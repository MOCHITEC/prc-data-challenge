"""Pre-test of docs/lirf_fill_2026-09-25/plan.md §1. Read-only: reads the feature cache and one OOF, writes stdout.

    .venv/bin/python docs/lirf_fill_2026-09-25/pretest.py <base_exp_id> [fold]

Trains a fill classifier on LIRF matched rows of the fold's training months (early stopping on the last
training month, the model from the months before it predicts the validation months), mixes its probability
into the base OOF on LIRF matched validation rows, and runs prc.compare's curve and verdict on the result,
next to a shuffled-probability control."""

from __future__ import annotations

import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from prc.compare import month_bootstrap_rmse_diff, robustness_curve, verdict
from prc.cv import month_block_folds, year_month_key
from prc.features import CATEGORICALS, numeric_feature_columns
from prc.train import LABEL, PROXY, SCHED_OFFSET, _matrix

AIRPORT = "LIRF"
FILL_TOL_S = 10
SEED = 20260925


def auc(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main(argv: list[str]) -> int:
    base_id = argv[0]
    fold_name = argv[1] if len(argv) > 1 else "forward"
    feats = pl.read_parquet("data/work/feat_train.parquet")
    months = year_month_key(feats)
    fold = next(f for f in month_block_folds(months.unique().to_list()) if f.name == fold_name)
    feats = feats.with_columns(
        months.alias("_ym"),
        ((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= FILL_TOL_S)
        .cast(pl.Int8)
        .alias("_fill"),
        pl.col("FLIGHT_mvt").cast(pl.Utf8).str.slice(0, 3).alias("flight_prefix"),
        pl.col("FLIGHT_mvt").cast(pl.Utf8).alias("flight_identity"),
    )
    lirf = feats.filter((pl.col("ADEP_mvt") == AIRPORT) & pl.col(PROXY).is_not_null() & pl.col(LABEL).is_not_null())
    train_months = sorted(fold.train_months)
    stop_month = train_months[-1]
    fit = lirf.filter(pl.col("_ym").is_in(train_months[:-1]))
    stop = lirf.filter(pl.col("_ym") == stop_month)
    val = lirf.filter(pl.col("_ym").is_in(list(fold.valid_months)))
    num = [c for c in numeric_feature_columns(lirf) if c not in (LABEL, "month", "_fill", "_ym")]
    cats = [c for c in CATEGORICALS if c in lirf.columns] + ["flight_prefix", "flight_identity"]
    stacked = pl.concat([f.select(num + cats) for f in (fit, stop, val)], how="vertical_relaxed")
    X, cols = _matrix(stacked, num, cats)
    ci = [cols.index(c) for c in cats]
    a, b = fit.height, fit.height + stop.height
    y_fit, y_stop, y_val = (f["_fill"].to_numpy() for f in (fit, stop, val))
    params = {
        "objective": "binary",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "num_threads": 4,
        "deterministic": True,
        "force_row_wise": True,
        "seed": SEED,
    }
    dtr = lgb.Dataset(X[:a], label=y_fit, categorical_feature=ci, params=params)
    dva = lgb.Dataset(X[a:b], label=y_stop, categorical_feature=ci, reference=dtr)
    booster = lgb.train(
        params, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)]
    )
    p = booster.predict(X[b:], num_iteration=booster.best_iteration)
    print(f"fold {fold_name}: LIRF matched rows fit {fit.height:,} / stop {stop.height:,} / validation {val.height:,}")
    print(f"fill rate fit {y_fit.mean():.2%}, stop {y_stop.mean():.2%}, validation {y_val.mean():.2%}")
    print(
        f"classifier: {booster.best_iteration} rounds, validation AUC {auc(y_val, p):.4f}, mean p {p.mean():.4f} "
        f"vs rate {y_val.mean():.4f}"
    )
    imp = sorted(zip(cols, booster.feature_importance("gain"), strict=True), key=lambda t: -t[1])[:12]
    tot = sum(v for _, v in zip(cols, booster.feature_importance("gain"), strict=True))
    print("top gain: " + ", ".join(f"{c} {v / tot:.1%}" for c, v in imp))

    # mix into the base OOF on exactly these rows, leave every other row of the fold as it was
    oof = pl.read_parquet(f"experiments/oof_{base_id}.parquet")
    lab = feats.filter(pl.col("_ym").is_in(list(fold.valid_months)) & pl.col(LABEL).is_not_null()).select(
        ["MVT_ID_mvt", LABEL, "MVT_TIME_UTC_mvt", "ADEP_mvt", PROXY, SCHED_OFFSET]
    )
    d = lab.join(oof, on="MVT_ID_mvt", how="inner").drop_nulls("oof")
    rng = np.random.default_rng(SEED)
    pv = pl.DataFrame({"MVT_ID_mvt": val["MVT_ID_mvt"], "_p": p, "_p_shuf": rng.permutation(p)})
    d = d.join(pv, on="MVT_ID_mvt", how="left", maintain_order="left")
    y = d[LABEL].to_numpy().astype(float)
    base = d["oof"].to_numpy().astype(float)
    ms = d[SCHED_OFFSET].to_numpy().astype(float)
    ym = year_month_key(d).to_numpy()
    on = d["_p"].is_not_null().to_numpy()
    out = {}
    for name, col in (("mix", "_p"), ("shuffled", "_p_shuf")):
        q = d[col].fill_null(0.0).to_numpy()
        mixed = np.where(on, (1 - q) * base + q * ms, base)
        mixed = np.maximum(mixed, 0.0)
        out[name] = mixed
    lirf_m = on
    print(f"\nrows in the comparison {len(y):,}; LIRF matched rows mixed {lirf_m.sum():,}")
    for name, pred in (("base", base), ("mix", out["mix"]), ("shuffled", out["shuffled"])):
        r = np.sqrt(np.mean((y - pred) ** 2))
        rl = np.sqrt(np.mean((y[lirf_m] - pred[lirf_m]) ** 2))
        matched = d[PROXY].is_not_null().to_numpy()
        rm = np.sqrt(np.mean((y[matched] - pred[matched]) ** 2))
        print(f"  {name:9s} fold {r:8.2f}   matched {rm:7.2f}   LIRF matched {rl:7.2f}")
    for name in ("mix", "shuffled"):
        curve = robustness_curve(y, base, out[name])
        ok, why = verdict(curve)
        mb = month_bootstrap_rmse_diff(y, base, out[name], ym)
        print(f"\n== base -> {name}")
        for rr in curve:
            print(
                f"  K={rr['dropped']:5d}  diff {rr['diff']:+7.2f}  CI95 [{rr['ci95'][0]:+7.2f}, {rr['ci95'][1]:+7.2f}]"
                f"  P(B better) {rr['b_better_prob']:.3f}"
            )
        print(f"  month bootstrap {mb}")
        print(f"  VERDICT: {'ADOPT' if ok else 'REJECT'} - {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
