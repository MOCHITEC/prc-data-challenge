"""Pre-registered comparisons C1/C2 (PREREG.md §5) on one or more folds.

    python experiments/nbrnet_2026-10-15/nbrnet/nbr_eval.py --folds forward --seeds 1 --out <exp>/results/s7_forward.json

Rows: every labelled row of the evaluated folds (matched and unmatched). a2o is the regenerated OOF from the
pipeline's fold cache. NN predictions exist on matched rows only; elsewhere every arm is a2o. Every blend and every
NN-alone vector is projected onto [max(MVT-LOBT-3606, 0), MVT-LOBT+3606] where MVT-LOBT exists, and floored at 0.
Decision: prc.compare.verdict(prc.compare.robustness_curve(y, A, B, n_boot=1000))."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parent.parent

from prc.compare import (
    month_bootstrap_rmse_diff,
    per_group_diff,
    robustness_curve,
    strict_report,
    strict_verdict,
    verdict,
)
from prc.cv import per_airport, rmse

SLACK = 3606.0
A2O_EXP = "main_flseq_osm_nolirf_l2_opt_l6_cmix-1e01a07623"
FUTILITY_POINT = -0.10


def project(pred: np.ndarray, lobt_off: np.ndarray) -> np.ndarray:
    """Floor at 0, then the LOBT window where it exists (train._assemble + train._lobt_bound, same order)."""
    p = np.maximum(pred, 0.0)
    ok = np.isfinite(lobt_off)
    lo = np.maximum(lobt_off - SLACK, 0.0)
    hi = lobt_off + SLACK
    return np.where(ok, np.minimum(np.maximum(p, lo), hi), p)


def blend(a2o: np.ndarray, nn: np.ndarray, w_nn: float, lobt_off: np.ndarray) -> np.ndarray:
    """(1-w)·a2o + w·nn where nn exists (matched rows), a2o elsewhere; projected and floored."""
    has = np.isfinite(nn)
    raw = np.where(has, (1.0 - w_nn) * a2o + w_nn * np.where(has, nn, 0.0), a2o)
    return project(raw, lobt_off)


def comparison(y: np.ndarray, A: np.ndarray, B: np.ndarray, n_boot: int = 1000) -> dict:
    curve = robustness_curve(y, A, B, n_boot=n_boot)
    ok, why = verdict(curve)
    return {"adopt": bool(ok), "verdict": why, "curve": curve}


def load_frame(folds: list[str], seeds: list[int], folds_dir: Path, a2o_dir: Path, preds_dir: Path, tag: str = "") -> pl.DataFrame:
    parts = []
    for f in folds:
        meta = pl.read_parquet(folds_dir / f / "valid_meta.parquet")
        a2o = pl.read_parquet(a2o_dir / f"{f}.parquet").rename({"pred": "a2o"})
        df = meta.join(a2o, on="MVT_ID_mvt", how="left", maintain_order="left").with_columns(pl.lit(f).alias("fold"))
        if df["a2o"].null_count():
            raise AssertionError(f"{df['a2o'].null_count()} rows of {f} have no a2o OOF")
        for mode in ("ctx", "noctx"):
            acc = None
            for s in seeds:
                sfx = f"_{tag}" if tag else ""
                p = pl.read_parquet(preds_dir / f"{f}_s{s}_{mode}{sfx}.parquet").select("MVT_ID_mvt", pl.col("nn_pred").cast(pl.Float64).alias(f"_s{s}"))
                acc = p if acc is None else acc.join(p, on="MVT_ID_mvt", how="full", coalesce=True)
            seed_cols = [c for c in acc.columns if c.startswith("_s")]
            if acc.select(pl.any_horizontal([pl.col(c).is_null() for c in seed_cols]).sum()).item():
                raise AssertionError(f"{f} {mode}: the seeds do not cover the same rows")
            acc = acc.select(
                "MVT_ID_mvt",
                pl.mean_horizontal(seed_cols).alias(f"nn_{mode}"),
                *[pl.col(c).alias(f"nn_{mode}{c}") for c in seed_cols],  # per seed, descriptive
            )
            df = df.join(acc, on="MVT_ID_mvt", how="left", maintain_order="left")
        parts.append(df)
    return pl.concat(parts)


def evaluate(df: pl.DataFrame, n_boot: int = 1000) -> dict:
    y = df["y"].to_numpy()
    a2o = df["a2o"].to_numpy()
    lobt = df["mvt_minus_lobt"].fill_null(np.nan).to_numpy()
    matched = df["proxy"].is_not_null().to_numpy()
    nn_c = df["nn_ctx"].fill_null(np.nan).to_numpy()
    nn_n = df["nn_noctx"].fill_null(np.nan).to_numpy()
    if np.isfinite(nn_c[~matched]).any() or (~np.isfinite(nn_c[matched])).any():
        raise AssertionError("NN predictions must cover exactly the matched rows")
    arms = {
        "a2o": project(a2o, lobt),  # already projected by the pipeline; idempotent
        "nn_ctx_alone": blend(a2o, nn_c, 1.0, lobt),
        "nn_noctx_alone": blend(a2o, nn_n, 1.0, lobt),
        "blend_a2o75_ctx25": blend(a2o, nn_c, 0.25, lobt),
        "blend_a2o75_noctx25": blend(a2o, nn_n, 0.25, lobt),
    }
    apt = df["ADEP_mvt"].to_numpy()
    fold = df["fold"].to_numpy()
    out = {
        "n": len(y),
        "n_matched": int(matched.sum()),
        "rmse": {k: rmse(y, v) for k, v in arms.items()},
        "rmse_matched": {k: rmse(y[matched], v[matched]) for k, v in arms.items()},
        "per_fold": {f: {k: rmse(y[fold == f], v[fold == f]) for k, v in arms.items()} for f in np.unique(fold)},
        "per_airport": {k: per_airport(y, v, apt) for k, v in arms.items()},
        "max_abs_change_a2o_vs_pipeline": float(np.max(np.abs(arms["a2o"] - a2o))),
    }
    out["C1"] = comparison(y, arms["blend_a2o75_noctx25"], arms["blend_a2o75_ctx25"], n_boot)
    out["C2"] = comparison(y, arms["a2o"], arms["blend_a2o75_ctx25"], n_boot)
    # descriptive reports (no decision attached; PREREG §5)
    ym = df["ym"].to_numpy()
    seeds = sorted({c.split("_s")[-1] for c in df.columns if c.startswith("nn_ctx_s")})
    out["per_seed"] = {}
    for sd in seeds:
        c, n = df[f"nn_ctx_s{sd}"].fill_null(np.nan).to_numpy(), df[f"nn_noctx_s{sd}"].fill_null(np.nan).to_numpy()
        per = {"nn_ctx_alone": blend(a2o, c, 1.0, lobt), "nn_noctx_alone": blend(a2o, n, 1.0, lobt),
               "blend_a2o75_ctx25": blend(a2o, c, 0.25, lobt), "blend_a2o75_noctx25": blend(a2o, n, 0.25, lobt)}
        out["per_seed"][sd] = {
            "rmse": {k: rmse(y, v) for k, v in per.items()},
            "per_fold": {f: {k: rmse(y[fold == f], v[fold == f]) for k, v in per.items()} for f in np.unique(fold)},
            "C1_diff": rmse(y, per["blend_a2o75_ctx25"]) - rmse(y, per["blend_a2o75_noctx25"]),
            "C2_diff": rmse(y, per["blend_a2o75_ctx25"]) - rmse(y, arms["a2o"]),
        }
    for cid, (A, B) in {"C1": (arms["blend_a2o75_noctx25"], arms["blend_a2o75_ctx25"]),
                        "C2": (arms["a2o"], arms["blend_a2o75_ctx25"])}.items():
        d = out[cid]
        d["per_fold_curve"] = {f: comparison(y[fold == f], A[fold == f], B[fold == f], n_boot) for f in np.unique(fold)}
        d["per_airport_diff"] = per_group_diff(y, A, B, apt, n_boot=500)
        d["lanes"] = {name: {"n": int(m.sum()), "rmse_a": rmse(y[m], A[m]), "rmse_b": rmse(y[m], B[m]),
                             "diff": rmse(y[m], B[m]) - rmse(y[m], A[m])}
                      for name, m in (("matched", matched), ("unmatched", ~matched))}
        d["month_bootstrap"] = month_bootstrap_rmse_diff(y, A, B, ym, n_boot=n_boot)
        if "day" in df.columns:
            block = np.char.add(apt.astype(str), df["day"].to_numpy().astype(str))
            rep = strict_report(y, A, B, fold, apt, block, n_boot=n_boot)
            ok, why = strict_verdict(d["adopt"], rep)
            d["strict"] = {"adopt": bool(ok), "verdict": why, "report": rep}
    full = out["C1"]["curve"][0]
    out["futility"] = {
        "rule": "stop if C1 point >= -0.10 s or C1 full-data CI wholly above 0",
        "c1_point": full["diff"],
        "c1_ci95": full["ci95"],
        "stop": bool(full["diff"] >= FUTILITY_POINT or full["ci95"][0] > 0),
    }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", default=["forward"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--tag", default="")
    ap.add_argument("--folds-dir", type=Path, default=ROOT / "data/nbrnet/folds")
    ap.add_argument("--a2o-dir", type=Path, default=EXP / "a2o_oof" / A2O_EXP)
    ap.add_argument("--preds-dir", type=Path, default=EXP / "preds")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--work-dir", type=Path, default=None, help="pipeline feature cache, for airport-day blocks")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    df = load_frame(a.folds, a.seeds, a.folds_dir, a.a2o_dir, a.preds_dir, a.tag)
    if a.work_dir is not None:  # UTC date of MVT for the airport-day blocks, as prc.compare.main builds them
        days = pl.read_parquet(a.work_dir / "feat_train.parquet", columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"]).select(
            "MVT_ID_mvt", pl.col("MVT_TIME_UTC_mvt").dt.date().cast(pl.Utf8).alias("day")
        )
        df = df.join(days, on="MVT_ID_mvt", how="left", maintain_order="left")
        if df["day"].null_count():
            raise AssertionError("rows without a movement date")
    res = {"folds": a.folds, "seeds": a.seeds, "tag": a.tag, **evaluate(df, a.n_boot)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=float))
    short = {k: res[k] for k in ("n", "rmse")}
    short["C1"] = (res["C1"]["adopt"], res["C1"]["verdict"], res["C1"]["curve"][0]["ci95"])
    short["C2"] = (res["C2"]["adopt"], res["C2"]["verdict"], res["C2"]["curve"][0]["ci95"])
    short["futility_stop"] = res["futility"]["stop"]
    for cid in ("C1", "C2"):
        if "strict" in res[cid]:
            short[f"{cid}_strict"] = (res[cid]["strict"]["adopt"], res[cid]["strict"]["verdict"])
        short[f"{cid}_month_bootstrap"] = (res[cid]["month_bootstrap"]["diff_rmse"], res[cid]["month_bootstrap"]["ci95"])
    print(json.dumps(short, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
