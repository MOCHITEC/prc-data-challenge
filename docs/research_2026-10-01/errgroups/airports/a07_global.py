# Global reference with same categoricals as the specialists (ADEP + RUNWAY/STAND/TYPE/OPERATOR + stand x runway TE). Optional regime cols for 4 airports.
import sys, numpy as np, polars as pl, lightgbm as lgb
S="/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/errgroups/"
f=pl.scan_parquet("/home/user/prc-data-challenge/data/work/feat_train.parquet").filter(pl.col("TAXITIME_SEC_mvt").is_not_null()&pl.col("mvt_minus_aobt3").is_not_null())
sch=f.collect_schema()
num=[c for c,d in sch.items() if d in (pl.Float64,pl.Float32,pl.Int64,pl.Int32,pl.Int8,pl.Int16,pl.UInt32,pl.UInt8)
     and c not in ("TAXITIME_SEC_mvt","MVT_ID_mvt","FLIGHT_ID_mvt","month") and not c.endswith("_slog")]
CATS=["ADEP_mvt","RUNWAY_mvt","STAND_mvt","AIRCRAFT_TYPE_mvt","AIRCRAFT_OPERATOR_flt"]
f=f.select(num+CATS+["MVT_ID_mvt","MVT_TIME_UTC_mvt","TAXITIME_SEC_mvt"]).with_columns(pl.col(pl.Float64).exclude("MVT_ID_mvt").cast(pl.Float32)).collect()
f=f.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mo"),(pl.col("TAXITIME_SEC_mvt")-pl.col("mvt_minus_aobt3")).alias("g"))
for c in CATS[1:]: f=f.with_columns((pl.col("ADEP_mvt")+"|"+pl.col(c).fill_null("NA")).cast(pl.Categorical).to_physical().alias(c+"_c"))
f=f.with_columns(pl.col("ADEP_mvt").cast(pl.Categorical).to_physical().alias("ADEP_mvt_c"))
CC=[c+"_c" for c in CATS]
tr=~f["mo"].is_in([1,7]).to_numpy(); te=~tr
t=f.filter(pl.Series(tr)); gm=t.group_by("ADEP_mvt").agg(pl.col("g").median().alias("gm"))
s=t.group_by("ADEP_mvt","STAND_mvt","RUNWAY_mvt").agg(pl.col("g").median().alias("m"),pl.len().alias("n")).join(gm,on="ADEP_mvt")
s=s.with_columns(((pl.col("m")*pl.col("n")+pl.col("gm")*20)/(pl.col("n")+20)).alias("te_sr"))
f=f.join(s.select("ADEP_mvt","STAND_mvt","RUNWAY_mvt","te_sr"),on=["ADEP_mvt","STAND_mvt","RUNWAY_mvt"],how="left",maintain_order="left").join(gm,on="ADEP_mvt",how="left",maintain_order="left")
f=f.with_columns(pl.col("te_sr").fill_null(pl.col("gm")).cast(pl.Float32))
feats=num+CC+["te_sr"]
arm=sys.argv[1] if len(sys.argv)>1 else "G2"
if arm=="G2R":
    rg=pl.read_parquet(S+"airports/regime_feats.parquet").drop("apt","is_rank","t")
    RG=[c for c in rg.columns if c.startswith("rg_")]
    f=f.join(rg,on="MVT_ID_mvt",how="left",maintain_order="left"); feats+=RG
X=f.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy()
y=f["g"].to_numpy(); base=f["mvt_minus_aobt3"].to_numpy(); ids=f["MVT_ID_mvt"]
del f
P=dict(objective="regression",learning_rate=0.05,num_leaves=255,min_data_in_leaf=50,feature_fraction=1.0,bagging_fraction=0.8,
       bagging_freq=1,verbose=-1,num_threads=2,seed=0,cat_smooth=20,max_bin=127)
ds=lgb.Dataset(X[tr],y[tr],feature_name=feats,categorical_feature=CC,free_raw_data=True)
b=lgb.train(P,ds,700)
p=base[te]+b.predict(X[te])
pl.DataFrame({"MVT_ID_mvt":ids.filter(pl.Series(te)),arm:p}).write_parquet(S+f"airports/global_{arm}.parquet")
print("done",arm)
