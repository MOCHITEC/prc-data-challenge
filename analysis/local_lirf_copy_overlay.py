"""LIRF matched-lane schedule-copy blend (local).

At LIRF 15-21 % of matched labels are schedule copies (|y - MVT_minus_SCHED| <= 12 s) in every 2025 month. A
LightGBM classifier on the pipeline features gives q = P(copy); the prediction is moved toward the schedule value:
p' = p + A * q * (MVT_minus_SCHED - p), with A fixed in advance at 0.4 (the 2025 cross-month fits gave 0.25 and 0.65).

  check  : classifier fit on the 10 months other than Jan/Jul (Dec = early stopping), applied to the Jan+Jul OOF
           rows of the v23-equivalent prediction (cv_rows_cb_keep: b + c_learned); gain reported per month.
  apply  : classifier refit on all 12 months (rounds = check's best iteration x 1.1), applied to 2026 LIRF matched rows
           of a submission file.

    .venv/bin/python analysis/local_lirf_copy_overlay.py check
    .venv/bin/python analysis/local_lirf_copy_overlay.py apply <in.parquet> <out.parquet>
"""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

A = 0.4
APT = os.environ.get("APT", "LIRF")
S = os.environ["SCRATCH"]
DROP = {"TAXITIME_SEC_mvt", "copy", "MVT_ID_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt"}
P = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=1, seed=1)


def frames():
    T = (pl.scan_parquet("data/work_flseq/feat_train.parquet")
         .filter((pl.col("ADEP_mvt") == APT) & pl.col("mvt_minus_aobt3").is_not_null()).collect()
         .with_columns(((pl.col("TAXITIME_SEC_mvt") - pl.col("mvt_minus_sched")).abs() <= 12).alias("copy")))
    R = pl.read_parquet("data/work_flseq/feat_rank.parquet").filter(
        (pl.col("ADEP_mvt") == APT) & pl.col("mvt_minus_aobt3").is_not_null())
    num = [c for c, t in T.schema.items() if c not in DROP and t.is_numeric() and c in R.columns]
    cat = [c for c, t in T.schema.items() if c not in DROP and t == pl.String and c in R.columns]
    # one consistent code per category value across train and rank
    codes = {c: {v: i for i, v in enumerate(sorted(set(T[c].drop_nulls().to_list()) | set(R[c].drop_nulls().to_list())))}
             for c in cat}

    def X(df):
        return df.select(num + [pl.col(c).replace_strict(codes[c], default=None, return_dtype=pl.Int32).alias(c) for c in cat]
                         ).to_numpy().astype(np.float64)
    return T, R, num + cat, cat, X


def check():
    T, R, fn, cat, X = frames()
    m = T["month"].to_numpy()
    y = T["copy"].to_numpy().astype(int)
    tr, va = ~np.isin(m, [1, 7, 12]), m == 12
    Xt = X(T)
    mdl = lgb.train(P, lgb.Dataset(Xt[tr], y[tr], feature_name=fn, categorical_feature=cat), 2000,
                    valid_sets=[lgb.Dataset(Xt[va], y[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
    q = mdl.predict(Xt, num_iteration=mdl.best_iteration)
    Q = pl.DataFrame({"MVT_ID_mvt": T["MVT_ID_mvt"], "q": q, "ms": T["mvt_minus_sched"], "month": T["month"]})
    cv = pl.read_parquet(os.path.join(S, "learn", "cv_rows_cb_keep.parquet")).with_columns((pl.col("b") + pl.col("c_learned")).alias("p"))
    N, sse = cv.height, float(((cv["p"] - cv["y"]) ** 2).sum())
    L = cv.join(Q, on="MVT_ID_mvt", how="inner")
    for lab, sel in (("Jan", L["month"] == 1), ("Jul", L["month"] == 7), ("Jan+Jul", L["month"] > 0)):
        X_ = L.filter(sel)
        p, yy, qq, ms = [X_[c].to_numpy() for c in ("p", "y", "q", "ms")]
        g = float(((p - yy) ** 2 - (p + A * qq * (ms - p) - yy) ** 2).sum()) / N
        print(f"{lab:8s} n={X_.height:6d} gain {g:7.1f} MSE (2025 units) -> RMSE {np.sqrt(sse / N):.2f} -> "
              f"{np.sqrt((sse - g * N) / N):.2f}" if lab == "Jan+Jul" else f"{lab:8s} n={X_.height:6d} gain {g:7.1f} MSE")
    json.dump({"best_iter": int(mdl.best_iteration)}, open(os.path.join(S, "learn", f"copy_{APT}.json"), "w"))


def apply(src, dst):
    T, R, fn, cat, X = frames()
    it = int(json.load(open(os.path.join(S, "learn", f"copy_{APT}.json")))["best_iter"] * 1.1)
    mdl = lgb.train(P, lgb.Dataset(X(T), T["copy"].to_numpy().astype(int), feature_name=fn, categorical_feature=cat), it)
    q = mdl.predict(X(R))
    sub = pl.read_parquet(src)
    G = pl.DataFrame({"MVT_ID_mvt": R["MVT_ID_mvt"], "q": q, "ms": R["mvt_minus_sched"]}).join(
        sub.rename({"TAXITIME_SEC_mvt": "p"}), on="MVT_ID_mvt").with_columns(
        (pl.col("p") + A * pl.col("q") * (pl.col("ms") - pl.col("p"))).round(0).alias("new"))
    G = G.with_columns(pl.when(pl.col("new") < 0).then(pl.col("p")).otherwise(pl.col("new")).cast(pl.Int32).alias("new"))
    out = (sub.join(G.select("MVT_ID_mvt", "new"), on="MVT_ID_mvt", how="left", maintain_order="left")
           .with_columns(pl.coalesce("new", "TAXITIME_SEC_mvt").cast(pl.Int32).alias("TAXITIME_SEC_mvt")).select(sub.columns))
    assert (out["MVT_ID_mvt"] == sub["MVT_ID_mvt"]).all() and out.null_count().sum_horizontal().item() == 0
    d = out["TAXITIME_SEC_mvt"].cast(pl.Float64) - sub["TAXITIME_SEC_mvt"].cast(pl.Float64)
    out.write_parquet(dst)
    print(f"{dst}: {G.height} LIRF matched rows, {int((d != 0).sum())} changed, sum delta^2/N {float((d ** 2).sum()) / sub.height:.1f}, "
          f"mean q {float(np.mean(q)):.3f} (2025 copy rate {float(T['copy'].mean()):.3f})")


if __name__ == "__main__":
    check() if sys.argv[1] == "check" else apply(sys.argv[2], sys.argv[3])
