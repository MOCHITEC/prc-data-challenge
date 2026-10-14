import numpy as np, polars as pl, lightgbm as lgb
f = pl.read_parquet("data/work/feat_train.parquet").filter(pl.col("TAXITIME_SEC_mvt").is_not_null())
f = f.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mo"))
m = f.filter(pl.col("mvt_minus_aobt3").is_not_null())
num = [c for c, d in m.schema.items() if d in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int8, pl.Int16, pl.UInt32, pl.UInt8)
       and c not in ("TAXITIME_SEC_mvt", "MVT_ID_mvt", "FLIGHT_ID_mvt", "month", "mo")]
tr, te = m.filter(~pl.col("mo").is_in([1, 7])), m.filter(pl.col("mo").is_in([1, 7]))
X = lambda d: d.select([pl.col(c).cast(pl.Float32) for c in num]).to_numpy()
params = dict(objective="regression", learning_rate=0.08, num_leaves=127, min_data_in_leaf=50, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4, seed=0)
bst = lgb.train(params, lgb.Dataset(X(tr), (tr["TAXITIME_SEC_mvt"] - tr["mvt_minus_aobt3"]).to_numpy()), 600)
y = pl.col("TAXITIME_SEC_mvt"); copy = pl.lit(False)
for c in ("mvt_minus_sched", "mvt_minus_aobt3", "mvt_minus_eobt1", "mvt_minus_lobt", "mvt_minus_iobt"):
    copy = copy | ((y - pl.col(c)).abs() <= 60).fill_null(False)
out = te.select("MVT_ID_mvt", "ADEP_mvt", "TAXITIME_SEC_mvt").with_columns(
    pl.Series("pred", te["mvt_minus_aobt3"].to_numpy() + bst.predict(X(te))),
    te.select((copy).alias("is_copy"))["is_copy"],
    (te["TAXITIME_SEC_mvt"] > 72000).alias("is_rollover"))
out = out.with_columns((pl.col("TAXITIME_SEC_mvt") - pl.col("pred")).alias("err"),
                       (~pl.col("is_copy") & ~pl.col("is_rollover")).alias("genuine"))
out.write_parquet("/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/errgroups/oof_matched_janjul.parquet")
imp = sorted(zip(bst.feature_importance("gain"), num), reverse=True)[:40]
open("/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/errgroups/feature_gain_top40.txt","w").write("\n".join(f"{g:.3e} {n}" for g,n in imp))
print(out.height, float(np.sqrt((out["err"]**2).mean())))
