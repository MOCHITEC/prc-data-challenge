"""LIRF unmatched departures: schedule-fill probability blend (local).

About half of LIRF's unmatched labels are BLOCK := SCHED fills (y = MVT - SCHED, often hours); the rest are ordinary
taxis. A classifier for fill (P1 stage 2 reached AUC 0.92 but was closed on an AUC gate, the blend never measured).
q from LightGBM on LIRF unmatched rows of the 8 months that are not Jan/Jul/Nov/Dec (Oct = early stop);
target t = q * (MVT - SCHED) + (1 - q) * T0, T0 = median label of non-fill LIRF unmatched rows of those months;
p' = p + A * (t - p) on the 4-month OOF (b + c_B). Gain per month for an A grid.

    .venv/bin/python analysis/local_lirf_unm_fill.py check
    .venv/bin/python analysis/local_lirf_unm_fill.py apply <A> <in.parquet> <out.parquet>
"""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

S = os.environ["SCRATCH"]
DROP = {"TAXITIME_SEC_mvt", "fill", "MVT_ID_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt"}
P = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=1, seed=5)


def frames():
    sel = pl.col("mvt_minus_aobt3").is_null() & (pl.col("ADEP_mvt") == "LIRF")
    T = (pl.scan_parquet("data/work_flseq/feat_train.parquet").filter(sel).collect()
         .with_columns(((pl.col("TAXITIME_SEC_mvt") - pl.col("mvt_minus_sched")).abs() <= 12).alias("fill")))
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
    y = T["fill"].to_numpy().astype(int)
    tr, va = ~np.isin(m, [1, 7, 11, 12, 10]), m == 10
    Xt = X(T)
    mdl = lgb.train(P, lgb.Dataset(Xt[tr], y[tr], feature_name=fn, categorical_feature=cat), 3000,
                    valid_sets=[lgb.Dataset(Xt[va], y[va])], callbacks=[lgb.early_stopping(200, verbose=False)])
    q = mdl.predict(Xt, num_iteration=mdl.best_iteration)
    from sklearn.metrics import roc_auc_score
    ev = np.isin(m, [1, 7, 11, 12])
    T0 = float(T.filter(pl.Series(tr) & ~pl.col("fill"))["TAXITIME_SEC_mvt"].median())
    print(f"rows: train {tr.sum()} (fill {y[tr].mean():.3f}), eval {ev.sum()} (fill {y[ev].mean():.3f}); AUC eval {roc_auc_score(y[ev], q[ev]):.3f}; "
          f"best_iter {mdl.best_iteration}; T0 {T0:.0f}")
    Q = pl.DataFrame({"MVT_ID_mvt": T["MVT_ID_mvt"], "q": q, "ms": T["mvt_minus_sched"]})
    cv = pl.read_parquet(os.path.join(S, "learn", "cv_rows_months.parquet")).with_columns((pl.col("b") + pl.col("c_B")).alias("p"))
    N = cv.height
    Lr = cv.join(Q, on="MVT_ID_mvt", how="inner").with_columns((pl.col("q") * pl.col("ms") + (1 - pl.col("q")) * T0).alias("t"))
    print("rows in 4-month OOF:", Lr.height)
    for A in (0.25, 0.5, 0.75, 1.0):
        out = []
        for ymo in ("2025-01", "2025-07", "2025-11", "2025-12"):
            Z = Lr.filter(pl.col("ym") == ymo)
            p, yy, t = [Z[c].to_numpy() for c in ("p", "y", "t")]
            out.append(float(((p - yy) ** 2 - (p + A * (t - p) - yy) ** 2).sum()) / N)
        print(f"A={A}: gain per month (MSE, 4-month units) " + " ".join(f"{g:+8.1f}" for g in out) + f" | total {sum(out):+8.1f}")
    json.dump({"best_iter": int(mdl.best_iteration), "T0": T0}, open(os.path.join(S, "learn", "lirf_unm_fill.json"), "w"))


def apply(A, src, dst):
    T, R, fn, cat, X = frames()
    cfg = json.load(open(os.path.join(S, "learn", "lirf_unm_fill.json")))
    T0all = float(T.filter(~pl.col("fill"))["TAXITIME_SEC_mvt"].median())
    mdl = lgb.train(P, lgb.Dataset(X(T), T["fill"].to_numpy().astype(int), feature_name=fn, categorical_feature=cat), int(cfg["best_iter"] * 1.1))
    q = mdl.predict(X(R))
    sub = pl.read_parquet(src)
    G = (pl.DataFrame({"MVT_ID_mvt": R["MVT_ID_mvt"], "q": q, "ms": R["mvt_minus_sched"]})
         .join(sub.rename({"TAXITIME_SEC_mvt": "p"}), on="MVT_ID_mvt")
         .with_columns((pl.col("q") * pl.col("ms") + (1 - pl.col("q")) * T0all).alias("t"))
         .with_columns((pl.col("p") + A * (pl.col("t") - pl.col("p"))).round(0).clip(0).cast(pl.Int32).alias("new")))
    out = (sub.join(G.select("MVT_ID_mvt", "new"), on="MVT_ID_mvt", how="left", maintain_order="left")
           .with_columns(pl.coalesce("new", "TAXITIME_SEC_mvt").cast(pl.Int32).alias("TAXITIME_SEC_mvt")).select(sub.columns))
    assert (out["MVT_ID_mvt"] == sub["MVT_ID_mvt"]).all() and out.null_count().sum_horizontal().item() == 0
    d = out["TAXITIME_SEC_mvt"].cast(pl.Float64) - sub["TAXITIME_SEC_mvt"].cast(pl.Float64)
    out.write_parquet(dst)
    print(f"{dst}: {G.height} LIRF unmatched rows, {int((d != 0).sum())} changed, sum delta^2/N {float((d ** 2).sum()) / sub.height:.1f}, "
          f"mean q {float(np.mean(q)):.3f} (2025 fill rate {float(T['fill'].mean()):.3f})")


if __name__ == "__main__":
    check() if sys.argv[1] == "check" else apply(float(sys.argv[2]), sys.argv[3], sys.argv[4])
