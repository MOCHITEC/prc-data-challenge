# LOMO copy classifier q for 2025 LIRF matched rows (same recipe as apply_overlays/copy_lomo), saved per row
import numpy as np, polars as pl, lightgbm as lgb
S='/tmp/claude-0/-home-user-prc-data-challenge/b411042e-f283-5ee9-9c9e-ffb2443b6ce1/scratchpad/'
DROP = {"TAXITIME_SEC_mvt", "copy", "MVT_ID_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt"}
P = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=2, seed=1)
T = (pl.scan_parquet("/home/user/prc-data-challenge/data/work_flseq/feat_train.parquet")
     .filter((pl.col("ADEP_mvt") == "LIRF") & pl.col("mvt_minus_aobt3").is_not_null()).collect()
     .with_columns(((pl.col("TAXITIME_SEC_mvt") - pl.col("mvt_minus_sched")).abs() <= 12).alias("copy")))
num = [c for c, t in T.schema.items() if c not in DROP and t.is_numeric()]
cat = [c for c, t in T.schema.items() if c not in DROP and t == pl.String]
codes = {c: {v: i for i, v in enumerate(sorted(set(T[c].drop_nulls().to_list())))} for c in cat}
X = T.select(num + [pl.col(c).replace_strict(codes[c], default=None, return_dtype=pl.Int32).alias(c) for c in cat]).to_numpy().astype(np.float64)
fn = num + cat
m = T["month"].to_numpy(); y = T["copy"].to_numpy().astype(int)
q = np.full(len(y), np.nan)
for k in range(1, 13):
    tr = m != k
    mdl = lgb.train(P, lgb.Dataset(X[tr], y[tr], feature_name=fn, categorical_feature=cat), 108)   # production: fixed 108 rounds
    q[m == k] = mdl.predict(X[m == k]); print(k, flush=True)
out = pl.DataFrame({"MVT_ID_mvt": T["MVT_ID_mvt"], "q": q, "ms_feat": T["mvt_minus_sched"].cast(pl.Float64), "mlobt_feat": T["mvt_minus_lobt"].cast(pl.Float64) if "mvt_minus_lobt" in T.columns else None, "month": T["month"]})
out.write_parquet(S + 'ideas/ht/copyq_lomo_2025.parquet')
print('features include mvt_minus_lobt:', 'mvt_minus_lobt' in num)
