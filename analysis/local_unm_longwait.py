"""Unmatched non-LIRF departures that pushed on time and then waited (y >= 1 h, BLOCK ~ SCHED): a probability blend
toward MVT - SCHED, as for the LIRF schedule copies (local).

q = P(y >= 3600) from a LightGBM classifier on the unmatched non-LIRF rows of the 8 months that are not Jan/Jul/Nov/Dec
(Oct = early stopping). Applied to the 4-month OOF rows (b + c_B) of those months:
p' = p + A * q * (max(MVT - SCHED, p) - p). Gain per month and A grid; A is then fixed for the apply step.

    .venv/bin/python analysis/local_unm_longwait.py check
    .venv/bin/python analysis/local_unm_longwait.py apply <A> <in.parquet> <out.parquet>
"""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

S = os.environ["SCRATCH"]
DROP = {"TAXITIME_SEC_mvt", "lab", "MVT_ID_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt"}
P = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=50, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=1, seed=3)


def frames():
    sel = pl.col("mvt_minus_aobt3").is_null() & (pl.col("ADEP_mvt") != "LIRF")
    T = (pl.scan_parquet("data/work_flseq/feat_train.parquet").filter(sel).collect()
         .with_columns((pl.col("TAXITIME_SEC_mvt") >= 3600).alias("lab")))
    R = pl.read_parquet("data/work_flseq/feat_rank.parquet").filter(sel)
    num = [c for c, t in T.schema.items() if c not in DROP and t.is_numeric() and c in R.columns and T[c].null_count() < T.height]
    cat = [c for c, t in T.schema.items() if c not in DROP and t == pl.String and c in R.columns]
    codes = {c: {v: i for i, v in enumerate(sorted(set(T[c].drop_nulls().to_list()) | set(R[c].drop_nulls().to_list())))} for c in cat}

    def X(df):
        return df.select(num + [pl.col(c).replace_strict(codes[c], default=None, return_dtype=pl.Int32).alias(c) for c in cat]
                         ).to_numpy().astype(np.float64)
    return T, R, num + cat, cat, X


def check():
    T, R, fn, cat, X = frames()
    m = T["month"].to_numpy()
    y = T["lab"].to_numpy().astype(int)
    tr, va = ~np.isin(m, [1, 7, 11, 12, 10]), m == 10
    Xt = X(T)
    mdl = lgb.train(P, lgb.Dataset(Xt[tr], y[tr], feature_name=fn, categorical_feature=cat), 3000,
                    valid_sets=[lgb.Dataset(Xt[va], y[va])], callbacks=[lgb.early_stopping(200, verbose=False)])
    q = mdl.predict(Xt, num_iteration=mdl.best_iteration)
    from sklearn.metrics import roc_auc_score
    ev = np.isin(m, [1, 7, 11, 12])
    print(f"positives: train {y[tr].sum()}/{tr.sum()}, eval {y[ev].sum()}/{ev.sum()}; AUC eval {roc_auc_score(y[ev], q[ev]):.3f}; best_iter {mdl.best_iteration}")
    imp = sorted(zip(fn, mdl.feature_importance("gain")), key=lambda t: -t[1])[:8]
    tot = sum(mdl.feature_importance("gain"))
    print("top features:", [(f, round(g / tot, 3)) for f, g in imp])
    Q = pl.DataFrame({"MVT_ID_mvt": T["MVT_ID_mvt"], "q": q, "ms": T["mvt_minus_sched"]})
    cv = pl.read_parquet(os.path.join(S, "learn", "cv_rows_months.parquet")).with_columns((pl.col("b") + pl.col("c_B")).alias("p"))
    N = cv.height
    Lr = cv.join(Q, on="MVT_ID_mvt", how="inner").with_columns(pl.max_horizontal("ms", "p").alias("tgt"))
    for A in (0.25, 0.5, 0.75, 1.0):
        out = []
        for ymo in ("2025-01", "2025-07", "2025-11", "2025-12"):
            Z = Lr.filter(pl.col("ym") == ymo)
            p, yy, qq, t = [Z[c].to_numpy() for c in ("p", "y", "q", "tgt")]
            out.append(float(((p - yy) ** 2 - (p + A * qq * (t - p) - yy) ** 2).sum()) / N)
        print(f"A={A}: gain per month (MSE, 4-month units) " + " ".join(f"{g:+7.1f}" for g in out) + f" | total {sum(out):+7.1f}")
    json.dump({"best_iter": int(mdl.best_iteration)}, open(os.path.join(S, "learn", "unm_longwait.json"), "w"))


def apply(A, src, dst):
    T, R, fn, cat, X = frames()
    it = int(json.load(open(os.path.join(S, "learn", "unm_longwait.json")))["best_iter"] * 1.1)
    mdl = lgb.train(P, lgb.Dataset(X(T), T["lab"].to_numpy().astype(int), feature_name=fn, categorical_feature=cat), it)
    q = mdl.predict(X(R))
    sub = pl.read_parquet(src)
    G = (pl.DataFrame({"MVT_ID_mvt": R["MVT_ID_mvt"], "q": q, "ms": R["mvt_minus_sched"]})
         .join(sub.rename({"TAXITIME_SEC_mvt": "p"}), on="MVT_ID_mvt")
         .with_columns(pl.max_horizontal("ms", "p").alias("tgt"))
         .with_columns((pl.col("p") + A * pl.col("q") * (pl.col("tgt") - pl.col("p"))).round(0).cast(pl.Int32).alias("new")))
    out = (sub.join(G.select("MVT_ID_mvt", "new"), on="MVT_ID_mvt", how="left", maintain_order="left")
           .with_columns(pl.coalesce("new", "TAXITIME_SEC_mvt").cast(pl.Int32).alias("TAXITIME_SEC_mvt")).select(sub.columns))
    assert (out["MVT_ID_mvt"] == sub["MVT_ID_mvt"]).all() and out.null_count().sum_horizontal().item() == 0
    d = out["TAXITIME_SEC_mvt"].cast(pl.Float64) - sub["TAXITIME_SEC_mvt"].cast(pl.Float64)
    out.write_parquet(dst)
    print(f"{dst}: {G.height} unmatched non-LIRF rows, {int((d != 0).sum())} changed, sum delta^2/N {float((d ** 2).sum()) / sub.height:.1f}, "
          f"q mean {float(np.mean(q)):.4f} (2025 rate {float(T['lab'].mean()):.4f}), q>0.3: {int((q > 0.3).sum())}")


if __name__ == "__main__":
    check() if sys.argv[1] == "check" else apply(float(sys.argv[2]), sys.argv[3], sys.argv[4])
