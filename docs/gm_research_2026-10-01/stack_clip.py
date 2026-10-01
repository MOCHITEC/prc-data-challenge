"""plan_clipped_member.md: out-of-fold segment stacker of the base and the clipped model A member (matched rows).

    .venv/bin/python docs/gm_research_2026-10-01/stack_clip.py BASE_ID CLIP_ID

Prediction = b + w_s (c - b) on matched rows, with w_s the least-squares weight in [0, 1] of segment s, fitted on the
other two pooled folds (jan / jul / forward) and applied to the third. Segments: airport x |c - b| band (< 60, 60-300,
> 300 s); fewer than 200 fitting rows -> w = 0. The control uses one global segment. Unmatched rows stay the base's.
Writes outputs/gm_stack/oof_{base,stacked,global}.parquet and runs prc.compare on them."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

LABEL, PROXY = "TAXITIME_SEC_mvt", "mvt_minus_aobt3"
WORK, EXP, OUT = Path("data/work_flseq"), Path("experiments"), Path("outputs/gm_stack")


def weights(y, b, c, seg, fit):
    w = {}
    for s in np.unique(seg[fit]):
        m = fit & (seg == s)
        d = c[m] - b[m]
        den = float(np.sum(d * d))
        w[s] = float(np.clip(np.sum(d * (y[m] - b[m])) / den, 0.0, 1.0)) if m.sum() >= 200 and den > 0 else 0.0
    return w


def main(base: str, clip: str) -> None:
    f = pl.read_parquet(WORK / "feat_train.parquet", columns=["MVT_ID_mvt", LABEL, PROXY, "ADEP_mvt"])
    folds = pl.read_parquet(EXP / f"oof_{base}_folds.parquet").filter("pooled").select("MVT_ID_mvt", "fold")
    d = (
        f.join(pl.read_parquet(EXP / f"oof_{base}.parquet").rename({"oof": "b"}), on="MVT_ID_mvt")
        .join(pl.read_parquet(EXP / f"oof_{clip}.parquet").rename({"oof": "c"}), on="MVT_ID_mvt")
        .join(folds, on="MVT_ID_mvt")
        .filter(pl.col("b").is_not_null() & pl.col("c").is_not_null() & pl.col(LABEL).is_finite())
    )
    y, b, c = d[LABEL].to_numpy().astype(float), d["b"].to_numpy(), d["c"].to_numpy()
    matched = d[PROXY].is_not_null().to_numpy()
    fold = d["fold"].to_numpy()
    gap = np.abs(c - b)
    band = np.where(gap < 60, 0, np.where(gap < 300, 1, 2))
    seg = np.char.add(d["ADEP_mvt"].to_numpy().astype(str), band.astype(str))
    one = np.full(len(y), "all")
    print(f"rows {len(y):,} (matched {int(matched.sum()):,}); folds {sorted(set(fold))}")
    print(
        f"matched RMSE: base {np.sqrt(np.mean((y[matched] - b[matched]) ** 2)):.2f}  clip "
        f"{np.sqrt(np.mean((y[matched] - c[matched]) ** 2)):.2f}"
    )
    out = {}
    for name, sg in (("stacked", seg), ("global", one)):
        p = b.copy()
        for k in sorted(set(fold)):
            fit, app = matched & (fold != k), matched & (fold == k)
            w = weights(y, b, c, sg, fit)
            ws = np.array([w.get(s, 0.0) for s in sg[app]])
            p[app] = b[app] + ws * (c[app] - b[app])
            if name == "global":
                print(f"  fold {k}: global w = {w.get('all', 0.0):.3f}")
            else:
                vals = np.array(list(w.values()))
                print(
                    f"  fold {k}: {len(w)} segments, w median {np.median(vals):.2f}, share w=0 {(vals == 0).mean():.2f}, "
                    f"share w=1 {(vals == 1).mean():.2f}"
                )
        out[name] = p
    OUT.mkdir(parents=True, exist_ok=True)
    for name, p in (("base", b), *out.items()):
        pl.DataFrame({"MVT_ID_mvt": d["MVT_ID_mvt"], "oof": p}).write_parquet(OUT / f"oof_{name}.parquet")
    for a_, b_ in (("base", "stacked"), ("base", "global"), ("global", "stacked")):
        print(f"\n######## prc.compare {a_} {b_}", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "prc.compare",
                a_,
                b_,
                "--work-dir",
                str(WORK),
                "--exp-dir",
                str(OUT),
                "--max-listed",
                "10",
            ],
            check=True,
        )


if __name__ == "__main__":
    main(*sys.argv[1:3])
