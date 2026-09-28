"""Item 26 `fold-shape-vs-ranking` (docs/investigation_2026-09-16/plan.md).

No model training. Lazy scans, only the columns needed. Run with the project venv:

    POLARS_MAX_THREADS=1 .venv/bin/python docs/investigation_2026-09-16/scripts/fold-shape-vs-ranking.py

Checks
  A. exact train/valid month sets of every fold month_block_folds() produces on the real month keys
  B. per-month row counts and unmatched shares (routing definition = PROXY null; D definition = FLIGHT_ID_mvt null)
     for feat_train and feat_rank; MVT_ID uniqueness and null-label count (preconditions for reading overall.n)
  C. fold geometry vs the production geometry (12 months -> 2026-01 and 2026-07)
  D. experiments.jsonl: overall.n vs sum of fold n; overall.rmse_full vs pooled RMSE of {jan,jul,forward} and of
     {jan,jul_fwd,forward} -> which fold's rows are inside `overall`
  E. months carrying a prediction in oof_main-48c19bc830.parquet
  F. replay of the run_cv coalesce expression on a toy frame with two folds predicting the same rows
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "1")
import polars as pl  # noqa: E402

pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(200)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from prc.cv import month_block_folds  # noqa: E402  (pure function; no data, no training)

TRAIN = ROOT / "data/work/feat_train.parquet"
RANK = ROOT / "data/work/feat_rank.parquet"
EXPS = ROOT / "experiments/experiments.jsonl"
OOF = ROOT / "experiments/oof_main-48c19bc830.parquet"
LABEL, PROXY, TIME = "TAXITIME_SEC_mvt", "mvt_minus_aobt3", "MVT_TIME_UTC_mvt"


def ym(col: str = TIME) -> pl.Expr:
    return (pl.col(col).dt.year() * 100 + pl.col(col).dt.month()).alias("ym")


def month_table(path: Path, has_label: bool) -> tuple[pl.DataFrame, dict]:
    lf = pl.scan_parquet(path)
    names = lf.collect_schema().names()
    aobt = next((c for c in names if c.upper().startswith("AOBT_3")), None)
    cols = [TIME, "MVT_ID_mvt", PROXY, "FLIGHT_ID_mvt"] + ([LABEL] if has_label else []) + ([aobt] if aobt else [])
    lf = lf.select(cols).with_columns(ym())
    per_month = (
        lf.group_by("ym")
        .agg(
            pl.len().alias("n"),
            pl.col(PROXY).is_null().sum().alias("unm_routing"),
            pl.col("FLIGHT_ID_mvt").is_null().sum().alias("unm_flightid"),
            (pl.col(PROXY).is_null() & pl.col("FLIGHT_ID_mvt").is_not_null()).sum().alias("proxy_null_but_matched"),
            (pl.col(PROXY).is_not_null() & pl.col("FLIGHT_ID_mvt").is_null()).sum().alias("proxy_ok_but_unmatched"),
        )
        .sort("ym")
        .with_columns(
            (pl.col("unm_routing") / pl.col("n") * 100).round(3).alias("pct_routing"),
            (pl.col("unm_flightid") / pl.col("n") * 100).round(3).alias("pct_flightid"),
        )
        .collect()
    )
    aggs = [
        pl.len().alias("rows"),
        pl.col("MVT_ID_mvt").n_unique().alias("mvt_id_unique"),
        pl.col("MVT_ID_mvt").is_null().sum().alias("mvt_id_null"),
        pl.col(PROXY).is_null().sum().alias("unm_routing"),
        pl.col("FLIGHT_ID_mvt").is_null().sum().alias("unm_flightid"),
    ]
    if has_label:
        aggs += [pl.col(LABEL).is_null().sum().alias("label_null"), (~pl.col(LABEL).is_finite()).sum().alias("label_nonfinite")]
    if aobt:
        aggs += [(pl.col(aobt).is_null() != pl.col(PROXY).is_null()).sum().alias(f"{aobt}_null_ne_proxy_null")]
    totals = lf.select(aggs).collect().to_dicts()[0]
    return per_month, totals


def horizon(train_months: tuple[int, ...], valid_month: int) -> int:
    """Calendar months from the last training month to the validation month (negative = training extends past it)."""
    last = max(train_months)
    return (valid_month // 100 - last // 100) * 12 + (valid_month % 100 - last % 100)


def pooled_rmse(folds: dict, names: list[str]) -> float:
    n = sum(folds[k]["n"] for k in names)
    return math.sqrt(sum(folds[k]["n"] * folds[k]["rmse_full"] ** 2 for k in names) / n)


def main() -> None:
    print("=== B1. feat_train per month ===")
    tr_months, tr_tot = month_table(TRAIN, has_label=True)
    print(tr_months)
    print("totals:", tr_tot)
    print(f"train unmatched share: routing {tr_tot['unm_routing'] / tr_tot['rows'] * 100:.3f} %  "
          f"flightid {tr_tot['unm_flightid'] / tr_tot['rows'] * 100:.3f} %")

    print("\n=== B2. feat_rank per month ===")
    rk_months, rk_tot = month_table(RANK, has_label=False)
    print(rk_months)
    print("totals:", rk_tot)
    print(f"ranking unmatched share: routing {rk_tot['unm_routing'] / rk_tot['rows'] * 100:.3f} %  "
          f"flightid {rk_tot['unm_flightid'] / rk_tot['rows'] * 100:.3f} %")

    print("\n=== A. folds produced by month_block_folds on the real month keys ===")
    keys = tr_months["ym"].to_list()
    n_by_ym = dict(zip(tr_months["ym"].to_list(), tr_months["n"].to_list()))
    folds = month_block_folds(keys)
    print("fold order:", [f.name for f in folds])
    for f in folds:
        n_valid = sum(n_by_ym[m] for m in f.valid_months)
        n_train = sum(n_by_ym[m] for m in f.train_months)
        print(f"{f.name:8s} valid={f.valid_months} n_valid={n_valid:,}  train({len(f.train_months)} months)={f.train_months} n_train={n_train:,}")

    print("\n=== C. fold geometry vs production ===")
    prod_train = tuple(keys)
    print(f"{'fold':9s} {'train months':>12s} {'valid':>16s} {'horizon(mo)':>12s} {'train months after valid':>25s} {'crosses year':>13s} {'valid season':>13s}")
    for f in folds:
        hz = [horizon(f.train_months, v) for v in f.valid_months]
        after = [sum(1 for m in f.train_months if m > v) for v in f.valid_months]
        print(f"{f.name:9s} {len(f.train_months):12d} {str(f.valid_months):>16s} {str(hz):>12s} {str(after):>25s} {'no':>13s} {str([v % 100 for v in f.valid_months]):>13s}")
    hz = [horizon(prod_train, v) for v in (202601, 202607)]
    print(f"{'PROD':9s} {len(prod_train):12d} {str((202601, 202607)):>16s} {str(hz):>12s} {str([0, 0]):>25s} {'yes':>13s} {str([1, 7]):>13s}")

    print("\n=== D. experiments.jsonl: overall.n vs sum fold n; overall.rmse vs pooled fold RMSE ===")
    hdr = f"{'exp_id':32s} {'folds':28s} {'sum_n':>8s} {'overall.n':>9s} {'overall':>8s} {'pool(jan,jul,fwd)':>17s} {'pool(jan,jul_fwd,fwd)':>21s} {'jul.rmse':>8s} {'jul_fwd.rmse':>12s}"
    print(hdr)
    for line in EXPS.read_text().splitlines():
        d = json.loads(line)
        fl = d["folds"]
        s = sum(v["n"] for v in fl.values())
        o = d["overall"]
        names = list(fl)
        p1 = pooled_rmse(fl, [k for k in names if k != "jul_fwd"]) if any(k != "jul_fwd" for k in names) else float("nan")
        p2 = pooled_rmse(fl, [k for k in names if k != "jul"]) if "jul_fwd" in fl else float("nan")
        print(f"{d['exp_id']:32s} {','.join(names):28s} {s:8d} {o['n']:9d} {o['rmse_full']:8.3f} {p1:17.3f} {p2:21.3f} "
              f"{fl.get('jul', {}).get('rmse_full', float('nan')):8.2f} {fl.get('jul_fwd', {}).get('rmse_full', float('nan')):12.2f}")

    print("\n=== E. months carrying an OOF prediction in oof_main-48c19bc830.parquet ===")
    oof = pl.scan_parquet(OOF)
    print("oof columns:", oof.collect_schema().names())
    j = (
        pl.scan_parquet(TRAIN).select(["MVT_ID_mvt", TIME]).with_columns(ym()).drop(TIME)
        .join(oof, on="MVT_ID_mvt", how="left")
        .group_by("ym").agg(pl.len().alias("n"), pl.col("oof").is_not_null().sum().alias("n_oof"))
        .sort("ym").collect()
    )
    print(j)
    print("oof rows total:", oof.select(pl.len()).collect().item(), " non-null oof:", oof.select(pl.col("oof").is_not_null().sum()).collect().item())

    print("\n=== F. replay of the run_cv coalesce on a toy frame (fold 1 and fold 2 both predict rows 1,2) ===")
    feats = pl.DataFrame({"MVT_ID_mvt": [1, 2, 3, 4]})
    oof_t = feats.with_columns(pl.lit(None, dtype=pl.Float64).alias("oof"))
    preds = [pl.DataFrame({"MVT_ID_mvt": [1, 2], "pred": [10.0, 20.0]}),   # fold 1 ("jul")
             pl.DataFrame({"MVT_ID_mvt": [1, 2], "pred": [99.0, 99.0]}),   # fold 2 ("jul_fwd"), same rows
             pl.DataFrame({"MVT_ID_mvt": [3], "pred": [30.0]})]            # fold 3 ("forward")
    for p in preds:
        oof_t = (oof_t.join(p.rename({"pred": "_p"}), on="MVT_ID_mvt", how="left", maintain_order="left")
                 .with_columns(pl.coalesce([pl.col("oof"), pl.col("_p")]).alias("oof")).drop("_p"))
    print(oof_t)
    print("-> fold-2 values (99.0) absent; rows 1,2 keep fold-1 values; row 4 (never validated) stays null and is dropped by drop_nulls")


if __name__ == "__main__":
    main()
