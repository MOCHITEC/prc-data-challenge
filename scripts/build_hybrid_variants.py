"""Build the three hybrid_blend_l2 variants of handoff/2026-10-02/README.md, for adsb_learn_cv/apply `--base <v>`.

    .venv/bin/python scripts/build_hybrid_variants.py            # all three
    .venv/bin/python scripts/build_hybrid_variants.py --check    # inputs and invariants only, writes nothing

Variants (b = the base the learned ADS-B residual is fitted on and applied to):
  l2m   CONTROL for l2w: production as it is (ranking file = hybrid_blend_l2 unchanged), but the unmatched OOF is
        production's model B validated with `fl_nm_rate_other` on the ranking's window (configs/flwin_target.yaml).
        main_flseq's own OOF validates on 12-month rates the ranking never has, so l2w vs l2 is not a fair CV
        comparison (it shows l2w ~ +1.6 s worse); l2w vs l2m is.
  l2w   matched = hybrid_blend_l2 (mean of the three L2 members); unmatched = the window-fixed model B
        (configs/flwin_both.yaml, `fe_fl_rate_window: both`) in place of main_flseq's
  l2c   matched = clipped-member segment stacker on the blend: b + w_s (c - b), c = main_flseq_osm_nolirf_l2_clip;
        unmatched = main_flseq (as in l2). Its fair control is l2.
  l2wc  both (the submission candidate)

Writes experiments/oof_hybrid_blend_<v>.parquet (MVT_ID_mvt, oof: every labelled pooled OOF row) and
outputs/hybrid_blend_<v>.parquet (the submission template's schema). Nothing is uploaded.

The stacker is docs/gm_research_2026-10-01/stack_clip.py's rule: per airport x |c - b| band (< 60, 60-300, > 300 s),
w = least squares in [0, 1], 0 below 200 fitting rows. OOF rows of fold k take weights fitted on the other pooled
folds (jan / jul / forward), so b stays out of fold for the residual's CV; the ranking rows take weights fitted on
all three. Labels are read only to fit those weights."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

# PRC_HANDOFF_ROOT: run against another tree (the stand-in test in handoff/2026-10-02/README.md uses it).
ROOT = Path(os.environ.get("PRC_HANDOFF_ROOT", Path(__file__).resolve().parent.parent))
EXP, OUT, HO = ROOT / "experiments", ROOT / "outputs", ROOT / "handoff" / "2026-10-02"
LABEL, PROXY = "TAXITIME_SEC_mvt", "mvt_minus_aobt3"
L2_MEMBERS = [
    "main_flseq_osm_nolirf_l2-ef204b44ff",
    "main_flseq_osm_nolirf_plainy_l2-c74a27239b",
    "main_flseq_osm_nolirf_cb_l2-4aa341b206",
]
UNM_BASE = "main_flseq-8615e6a194"
VARIANTS = ("l2m", "l2w", "l2c", "l2wc")


def seg_of(adep: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    gap = np.abs(c - b)
    return np.char.add(adep.astype(str), np.where(gap < 60, 0, np.where(gap < 300, 1, 2)).astype(str))


def fit_w(y: np.ndarray, b: np.ndarray, c: np.ndarray, seg: np.ndarray) -> dict[str, float]:
    w = {}
    for s in np.unique(seg):
        m = seg == s
        d = c[m] - b[m]
        den = float(np.sum(d * d))
        w[s] = float(np.clip(np.sum(d * (y[m] - b[m])) / den, 0.0, 1.0)) if m.sum() >= 200 and den > 0 else 0.0
    return w


def load_oof() -> pl.DataFrame:
    ft = pl.read_parquet(ROOT / "data/work_flseq/feat_train.parquet", columns=["MVT_ID_mvt", LABEL, PROXY, "ADEP_mvt"])
    folds = pl.read_parquet(EXP / f"oof_{L2_MEMBERS[0]}_folds.parquet").filter("pooled").select("MVT_ID_mvt", "fold")
    d = folds.join(ft, on="MVT_ID_mvt", how="left")
    for i, mem in enumerate(L2_MEMBERS):
        fm = pl.read_parquet(EXP / f"oof_{mem}_folds.parquet").filter("pooled").select("MVT_ID_mvt", "fold")
        if not fm.sort("MVT_ID_mvt").equals(folds.sort("MVT_ID_mvt")):
            raise AssertionError(f"{mem}: pooled folds differ from {L2_MEMBERS[0]}")
        d = d.join(pl.read_parquet(EXP / f"oof_{mem}.parquet").rename({"oof": f"m{i}"}), on="MVT_ID_mvt", how="left")
    d = d.join(pl.read_parquet(EXP / f"oof_{UNM_BASE}.parquet").rename({"oof": "ub"}), on="MVT_ID_mvt", how="left")
    d = d.join(pl.read_parquet(HO / "unm_window_oof.parquet").rename({"oof": "uw"}), on="MVT_ID_mvt", how="left")
    d = d.join(pl.read_parquet(HO / "unm_prod_testlike_oof.parquet").rename({"oof": "um"}), on="MVT_ID_mvt", how="left")
    d = d.join(pl.read_parquet(HO / "l2clip_oof.parquet").rename({"oof": "c"}), on="MVT_ID_mvt", how="left")
    d = d.with_columns(((pl.col("m0") + pl.col("m1") + pl.col("m2")) / 3.0).alias("blend"))
    mt, um = pl.col(PROXY).is_not_null(), pl.col(PROXY).is_null()
    for col, sel, what in (
        ("blend", mt, "blend"),
        ("c", mt, "clip member"),
        ("ub", um, "main_flseq model B"),
        ("uw", um, "window-fixed model B"),
    ):
        n = d.filter(sel & pl.col(col).is_null()).height
        if n:
            raise AssertionError(f"{n} pooled rows lack the {what} OOF")
    return d


def stacked_oof(d: pl.DataFrame) -> np.ndarray:
    y, b, c = d[LABEL].to_numpy().astype(float), d["blend"].to_numpy(), d["c"].to_numpy()
    matched, fold = d[PROXY].is_not_null().to_numpy(), d["fold"].to_numpy()
    seg = seg_of(d["ADEP_mvt"].to_numpy(), np.nan_to_num(b), np.nan_to_num(c))
    ok = matched & np.isfinite(y)
    out = b.copy()
    for k in np.unique(fold):
        w = fit_w(y[ok & (fold != k)], b[ok & (fold != k)], c[ok & (fold != k)], seg[ok & (fold != k)])
        app = matched & (fold == k)
        out[app] = b[app] + np.array([w.get(s, 0.0) for s in seg[app]]) * (c[app] - b[app])
        print(f"  stacker fold {k}: {len(w)} segments, median w {np.median(list(w.values())):.2f}")
    return out


def main(check: bool) -> None:
    d = load_oof()
    y = d[LABEL].to_numpy().astype(float)
    matched = d[PROXY].is_not_null().to_numpy()
    blend, ub, uw, umt = d["blend"].to_numpy(), d["ub"].to_numpy(), d["uw"].to_numpy(), d["um"].to_numpy()
    st = stacked_oof(d)
    oof = {
        "l2": np.where(matched, blend, ub),
        "l2m": np.where(matched, blend, umt),
        "l2w": np.where(matched, blend, uw),
        "l2c": np.where(matched, st, ub),
        "l2wc": np.where(matched, st, uw),
    }
    ok = np.isfinite(y)
    for k, p in oof.items():
        print(f"OOF {k:>4}: RMSE {np.sqrt(np.mean((y[ok] - p[ok]) ** 2)):8.3f}  (rows {int(ok.sum()):,})")

    tpl = pl.read_parquet(ROOT / "data/raw/submitting.parquet")
    base = pl.read_parquet(OUT / "hybrid_blend_l2.parquet")
    if not (
        base.schema == tpl.schema and base.height == tpl.height and (base["MVT_ID_mvt"] == tpl["MVT_ID_mvt"]).all()
    ):
        raise AssertionError("outputs/hybrid_blend_l2.parquet is not in the template's schema and order")
    rk = pl.read_parquet(ROOT / "data/work_flseq/feat_rank.parquet", columns=["MVT_ID_mvt", PROXY, "ADEP_mvt"])
    r = (
        base.rename({LABEL: "b"})
        .join(rk, on="MVT_ID_mvt", how="left", maintain_order="left")
        .join(
            pl.read_parquet(HO / "unm_window_rank.parquet").rename({"pred": "uw"}),
            on="MVT_ID_mvt",
            how="left",
            maintain_order="left",
        )
        .join(
            pl.read_parquet(HO / "l2clip_rank.parquet").rename({"pred": "c"}),
            on="MVT_ID_mvt",
            how="left",
            maintain_order="left",
        )
    )
    rmatched = r[PROXY].is_not_null().to_numpy()
    if r.filter(pl.col(PROXY).is_null())["uw"].null_count() or r.filter(pl.col(PROXY).is_not_null())["c"].null_count():
        raise AssertionError("the hand-off ranking files do not cover the ranking rows")
    rb, rc = r["b"].cast(pl.Float64).to_numpy(), r["c"].cast(pl.Float64).fill_null(0).to_numpy()
    w_all = fit_w(
        y[matched & ok],
        blend[matched & ok],
        d["c"].to_numpy()[matched & ok],
        seg_of(d["ADEP_mvt"].to_numpy()[matched & ok], blend[matched & ok], d["c"].to_numpy()[matched & ok]),
    )
    rs = seg_of(r["ADEP_mvt"].to_numpy(), rb, rc)
    r_st = rb + np.array([w_all.get(s, 0.0) for s in rs]) * (rc - rb)
    r_uw = r["uw"].cast(pl.Float64).fill_null(0).to_numpy()
    rank = {
        "l2m": rb.copy(),
        "l2w": np.where(rmatched, rb, r_uw),
        "l2c": np.where(rmatched, r_st, rb),
        "l2wc": np.where(rmatched, r_st, r_uw),
    }
    for k, p in rank.items():
        print(
            f"rank {k:>4}: rows changed vs hybrid_blend_l2 {int((np.rint(p) != rb).sum()):,}, mean |change| "
            f"{np.abs(p - rb).mean():.2f} s"
        )
    if check:
        print("--check: nothing written")
        return
    for k in VARIANTS:
        pl.DataFrame({"MVT_ID_mvt": d["MVT_ID_mvt"], "oof": oof[k]}).write_parquet(
            EXP / f"oof_hybrid_blend_{k}.parquet"
        )
        vals = pl.Series(LABEL, np.rint(rank[k])).cast(tpl.schema[LABEL])
        pl.DataFrame({"MVT_ID_mvt": r["MVT_ID_mvt"], LABEL: vals}).write_parquet(OUT / f"hybrid_blend_{k}.parquet")
        print(f"wrote experiments/oof_hybrid_blend_{k}.parquet and outputs/hybrid_blend_{k}.parquet")


if __name__ == "__main__":
    main("--check" in sys.argv)
