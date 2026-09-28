"""Batch C forward-fold comparison helper (investigation 2026-09-16, items 10 / 11 / 23).

Reads only finished artefacts - `experiments/experiments.jsonl` and `experiments/oof_<exp_id>.parquet` - plus
`data/work/feat_train.parquet`; it trains nothing. The same statistics are needed by all three Batch C items,
so `linear-tree-model-b.py` and `holdout-month-decides-rounds.py` call this module rather than repeat it.

    POLARS_MAX_THREADS=2 .venv/bin/python docs/investigation_2026-09-16/scripts/tree-identity-saturation.py \
        inv_baseline_forward-xxxxxxxxxx inv_it10_leaf5-yyyyyyyyyy

Definitions follow plan.md section 0: y = TAXITIME_SEC_mvt; ms = mvt_minus_sched = MVT - SCHED; "unmatched" =
routing definition (mvt_minus_aobt3 null); sched-filled = |y - ms| <= 60 s (D's row-level test, also used by
cond-mean-vs-median.md); 2 h = 7,200 s, 4 h = 14,400 s, 12 h = 43,200 s.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
LABEL, PROXY, MS = "TAXITIME_SEC_mvt", "mvt_minus_aobt3", "mvt_minus_sched"
COLS = ["MVT_ID_mvt", LABEL, PROXY, MS, "ADEP_mvt", "MVT_TIME_UTC_mvt"]


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2))) if len(y) else float("nan")


def load_feats() -> pl.DataFrame:
    return pl.read_parquet(ROOT / "data/work/feat_train.parquet", columns=COLS).with_columns(
        (pl.col("MVT_TIME_UTC_mvt").dt.year() * 100 + pl.col("MVT_TIME_UTC_mvt").dt.month()).alias("ym")
    )


def records() -> dict[str, dict]:
    out = {}
    with open(ROOT / "experiments/experiments.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            out[r["exp_id"]] = r  # later records win, as `make compare` would also read the file by exp_id
    return out


def stats(exp_id: str, feats: pl.DataFrame, rec: dict | None) -> dict:
    oof = pl.read_parquet(ROOT / f"experiments/oof_{exp_id}.parquet")
    d = feats.join(oof, on="MVT_ID_mvt", how="inner").drop_nulls(["oof", LABEL])
    y, p = d[LABEL].to_numpy().astype(float), d["oof"].to_numpy().astype(float)
    unm = d[PROXY].is_null().to_numpy()
    ms = d[MS].to_numpy().astype(float)
    lirf = (d["ADEP_mvt"] == "LIRF").to_numpy()
    filled = np.abs(y - ms) <= 60.0
    s: dict = {
        "exp_id": exp_id,
        "months": sorted(d["ym"].unique().to_list()),
        "n": len(y),
        "rmse_total": rmse(y, p),
        "n_matched": int((~unm).sum()),
        "rmse_matched": rmse(y[~unm], p[~unm]),
        "n_unmatched": int(unm.sum()),
        "rmse_unmatched": rmse(y[unm], p[unm]),
        "unm_pred_gt_14400": int((p[unm] > 14_400).sum()),
        "unm_pred_gt_43200": int((p[unm] > 43_200).sum()),
        "unm_pred_max": float(p[unm].max()),
        "unm_pred_median": float(np.median(p[unm])),
        "rmse_lirf_unm": rmse(y[unm & lirf], p[unm & lirf]),
        "n_lirf_unm": int((unm & lirf).sum()),
    }
    if rec is not None:
        f = rec["folds"]["forward"]
        s |= {
            "rounds_unmatched": f.get("rounds_unmatched"),
            "rounds_matched": f.get("rounds_matched"),
            "train_seconds": rec["train_seconds"],
            "git_sha": rec["git_sha"],
            "rec_rmse_full": f["rmse_full"],
            "rec_rmse_clean": f["rmse_clean"],
            "rec_top20_share": f["top20_share_of_sse"],
        }
    # pred / ms on LIRF unmatched rows with ms >= 2 h, all rows and the sched-filled subset
    for tag, sel in (
        ("lirf_unm_ms2h", unm & lirf & (ms >= 7_200)),
        ("lirf_unm_ms2h_filled", unm & lirf & (ms >= 7_200) & filled),
        ("nonlirf_unm_ms2h", unm & ~lirf & (ms >= 7_200)),
    ):
        ratio = p[sel] / ms[sel]
        s[f"n_{tag}"] = int(sel.sum())
        if sel.sum():
            s[f"{tag}_ratio_p10"] = float(np.percentile(ratio, 10))
            s[f"{tag}_ratio_med"] = float(np.median(ratio))
            s[f"{tag}_ratio_p90"] = float(np.percentile(ratio, 90))
            s[f"{tag}_rmse"] = rmse(y[sel], p[sel])
            s[f"{tag}_pred_max"] = float(p[sel].max())
    return s


def rows_table(exp_ids: list[str], feats: pl.DataFrame) -> pl.DataFrame:
    """Per-row predictions of every experiment on the LIRF unmatched ms >= 2 h rows of the scored fold."""
    d = feats
    for e in exp_ids:
        oof = pl.read_parquet(ROOT / f"experiments/oof_{e}.parquet").rename({"oof": e})
        d = d.join(oof, on="MVT_ID_mvt", how="inner")
    return (
        d.filter(
            pl.col(PROXY).is_null()
            & (pl.col("ADEP_mvt") == "LIRF")
            & (pl.col(MS) >= 7_200)
            & pl.col(exp_ids[0]).is_not_null()
        )
        .select(["MVT_ID_mvt", "ym", LABEL, MS, *exp_ids])
        .sort(MS, descending=True)
    )


def train_side_counts() -> None:
    """Unmatched training rows available to model B on the forward fold (Jan-Oct 2025) by ms band.

    The hypothesis of item 10 is a comparison between `min_data_in_leaf` and the number of LIRF rows in the
    extreme ms bands, so the counts have to be taken on the fold's training side, not on the whole year."""
    f = load_feats().filter(pl.col(PROXY).is_null())
    for tag, d in (("all 12 months", f), ("forward train (Jan-Oct 2025)", f.filter(pl.col("ym") <= 202510))):
        lirf = d.filter(pl.col("ADEP_mvt") == "LIRF")
        ms, y = lirf[MS].to_numpy().astype(float), lirf[LABEL].to_numpy().astype(float)
        ms_all = d[MS].to_numpy().astype(float)
        print(
            f"{tag}: unmatched {d.height:,}; LIRF unmatched {lirf.height:,}; "
            f"LIRF ms>43200 {(ms > 43_200).sum()}; LIRF ms in (14400,43200] {((ms > 14_400) & (ms <= 43_200)).sum()}; "
            f"LIRF ms>=7200 {(ms >= 7_200).sum()}; LIRF y>=43200 {(y >= 43_200).sum()}; "
            f"LIRF y>=14400 {(y >= 14_400).sum()}; all-unmatched ms>43200 {(ms_all > 43_200).sum()}"
        )


def holdout_counts() -> None:
    """Composition of the inner early-stopping holdout that `_fit` carves out of the forward fold's training side.

    `_inner_holdout_mask` takes the latest `inner_holdout_months` calendar months of the frame it is given, and
    `predict_fold` gives it the matched rows (model A) and the unmatched rows (model B) separately, so the
    holdout is per model. Also prints the refit scaling factor train.height / (train.height - n_hold) that
    `_fit` applies to `best_iteration`."""
    f = load_feats().filter(pl.col("ym") <= 202510)  # forward fold training side
    for model, d in (("A (matched)", f.filter(pl.col(PROXY).is_not_null())), ("B (unmatched)", f.filter(pl.col(PROXY).is_null()))):
        for months, hold in ((1, [202510]), (2, [202509, 202510])):
            h = d.filter(pl.col("ym").is_in(hold))
            y = h[LABEL].to_numpy().astype(float)
            print(
                f"model {model}, inner_holdout_months={months}: train {d.height:,}, holdout {h.height:,} "
                f"({', '.join(str(m) for m in hold)}), y>=14400 {(y >= 14_400).sum()}, y>=43200 {(y >= 43_200).sum()}, "
                f"y>=7200 {(y >= 7_200).sum()}, max y {y.max():,.0f}, "
                f"refit scaling {d.height / (d.height - h.height):.4f}"
            )


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--train-counts":
        train_side_counts()
        argv = argv[1:]
    if argv and argv[0] == "--holdout-counts":
        holdout_counts()
        argv = argv[1:]
    exp_ids = argv or []
    if not exp_ids:
        return 0
    feats, recs = load_feats(), records()
    allst = [stats(e, feats, recs.get(e)) for e in exp_ids]
    keys = sorted({k for s in allst for k in s} - {"exp_id"})
    for k in ["rounds_unmatched", "rounds_matched", *[x for x in keys if x not in ("rounds_unmatched", "rounds_matched")]]:
        vals = []
        for s in allst:
            v = s.get(k)
            vals.append(f"{v:>16.4f}" if isinstance(v, float) else f"{str(v):>16}")
        print(f"{k:>26} " + " ".join(vals))
    print("\ncolumns: " + " | ".join(exp_ids))
    t = rows_table(exp_ids, feats)
    print(f"\nLIRF unmatched rows with ms >= 2 h in the scored fold: {t.height}")
    with pl.Config(tbl_rows=100, tbl_width_chars=200, fmt_str_lengths=40):
        print(t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
