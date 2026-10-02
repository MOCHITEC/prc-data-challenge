"""Build the LB file for the clipped-member stacker (plan_clipped_member.md): v16 + w_s (clip - v16) on matched rows.

    .venv/bin/python docs/gm_research_2026-10-01/apply_stack.py BASE_PARQUET CLIP_REFIT_PARQUET OUT_PARQUET [BASE_ID CLIP_ID]

The segment weights are fitted on all three pooled OOF folds of main_flseq-8615e6a194 and clip1800-935169f2ee (the
same rule as stack_clip.py: least squares in [0, 1] per airport x |c - b| band, w = 0 below 200 rows). v16 is the
full refit of main_flseq, so b at test time is the model whose OOF the weights were fitted on. Writes, never uploads."""

import sys
from pathlib import Path

import numpy as np
import polars as pl

from prc.predict import write_submission

LABEL, PROXY = "TAXITIME_SEC_mvt", "mvt_minus_aobt3"
WORK, EXP = Path("data/work_flseq"), Path("experiments")
BASE, CLIP = "main_flseq-8615e6a194", "clip1800-935169f2ee"


def seg_of(adep: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    gap = np.abs(c - b)
    band = np.where(gap < 60, 0, np.where(gap < 300, 1, 2))
    return np.char.add(adep.astype(str), band.astype(str))


def main(v16_path: str, clip_path: str, out_path: str, base_id: str = BASE, clip_id: str = CLIP) -> None:
    f = pl.read_parquet(WORK / "feat_train.parquet", columns=["MVT_ID_mvt", LABEL, PROXY, "ADEP_mvt"])
    d = (
        f.join(pl.read_parquet(EXP / f"oof_{base_id}.parquet").rename({"oof": "b"}), on="MVT_ID_mvt")
        .join(pl.read_parquet(EXP / f"oof_{clip_id}.parquet").rename({"oof": "c"}), on="MVT_ID_mvt")
        .filter(
            pl.col("b").is_not_null()
            & pl.col("c").is_not_null()
            & pl.col(LABEL).is_finite()
            & pl.col(PROXY).is_not_null()
        )
    )
    y, b, c = d[LABEL].to_numpy().astype(float), d["b"].to_numpy(), d["c"].to_numpy()
    seg = seg_of(d["ADEP_mvt"].to_numpy(), b, c)
    w = {}
    for s in np.unique(seg):
        m = seg == s
        dd = c[m] - b[m]
        den = float(np.sum(dd * dd))
        w[s] = float(np.clip(np.sum(dd * (y[m] - b[m])) / den, 0.0, 1.0)) if m.sum() >= 200 and den > 0 else 0.0
    print("weights:", {k: round(v, 3) for k, v in sorted(w.items())})

    rk = pl.read_parquet(WORK / "feat_rank.parquet", columns=["MVT_ID_mvt", PROXY, "ADEP_mvt"])
    v16 = pl.read_parquet(v16_path).rename({LABEL: "b"})
    cl = pl.read_parquet(clip_path).rename({LABEL: "c"})
    r = v16.join(rk, on="MVT_ID_mvt", how="left", maintain_order="left").join(
        cl, on="MVT_ID_mvt", how="left", maintain_order="left"
    )
    if r.height != v16.height or r["c"].null_count() or r["ADEP_mvt"].null_count():
        raise AssertionError("v16, the clip refit and the ranking features do not line up")
    rb, rc = r["b"].cast(pl.Float64).to_numpy(), r["c"].cast(pl.Float64).to_numpy()
    matched = r[PROXY].is_not_null().to_numpy()
    rs = seg_of(r["ADEP_mvt"].to_numpy(), rb, rc)
    ws = np.array([w.get(s, 0.0) for s in rs])
    pred = np.where(matched, rb + ws * (rc - rb), rb)
    print(
        f"matched rows {int(matched.sum()):,}: mean |change| {np.abs(pred - rb)[matched].mean():.2f} s, "
        f"max {np.abs(pred - rb).max():.0f} s; unmatched rows unchanged: {bool((pred[~matched] == rb[~matched]).all())}"
    )
    template = next(iter(sorted(Path("data/raw").rglob("submitting.parquet"))))
    ref = float(f[LABEL].median())
    write_submission(
        pl.DataFrame({"MVT_ID_mvt": r["MVT_ID_mvt"], "pred": pred}), template, Path(out_path), reference_median=ref
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:6])
