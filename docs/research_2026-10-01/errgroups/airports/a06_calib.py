# per airport x runway (x stand-group) bias calibration of baseline pred, cross-fit Jan<->Jul (fit on one month, apply to other)
import numpy as np, polars as pl
S="/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/errgroups/"
o=pl.read_parquet(S+"oof_matched_janjul.parquet")
f=pl.scan_parquet("/home/user/prc-data-challenge/data/work/feat_train.parquet").select("MVT_ID_mvt","RUNWAY_mvt","STAND_mvt","MVT_TIME_UTC_mvt","hour_local").collect()
d=o.join(f,on="MVT_ID_mvt").with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mo"),pl.col("MVT_TIME_UTC_mvt").dt.date().alias("day"))
d=d.with_columns(pl.col("err").clip(-1800,1800).alias("ec"))  # robust target for bias
def run(keys,K,name):
    outs=[]
    for m_fit,m_app in [(1,7),(7,1)]:
        b=d.filter(pl.col("mo")==m_fit).group_by(keys).agg((pl.col("ec").sum()/(pl.len()+K)).alias("adj"))
        outs.append(d.filter(pl.col("mo")==m_app).join(b,on=keys,how="left").with_columns(pl.col("adj").fill_null(0)))
    x=pl.concat(outs).with_columns((pl.col("pred")+pl.col("adj")).alias("p2"))
    for A in ["LIRF","LTFM","LFPG","EGLL","ALL"]:
        y=x if A=="ALL" else x.filter(pl.col("ADEP_mvt")==A)
        g=y.with_columns(((pl.col("TAXITIME_SEC_mvt")-pl.col("pred"))**2).alias("a"),((pl.col("TAXITIME_SEC_mvt")-pl.col("p2"))**2).alias("b")).group_by("day").agg(pl.col("a").sum(),pl.col("b").sum(),pl.len())
        a,bb,n=g["a"].to_numpy(),g["b"].to_numpy(),g["len"].to_numpy(); r=np.random.default_rng(0)
        bs=[np.sqrt(bb[i].sum()/n[i].sum())-np.sqrt(a[i].sum()/n[i].sum()) for i in (r.integers(0,len(n),len(n)) for _ in range(500))]
        print(f"{name:28s} K={K:4d} {A:5s} dRMSE={np.sqrt(bb.sum()/n.sum())-np.sqrt(a.sum()/n.sum()):+.2f} [{np.percentile(bs,2.5):+.2f},{np.percentile(bs,97.5):+.2f}]")
for K in [50,500]:
    run(["ADEP_mvt"],K,"airport")
    run(["ADEP_mvt","RUNWAY_mvt"],K,"airport x runway")
    run(["ADEP_mvt","RUNWAY_mvt","STAND_mvt"],K,"airport x runway x stand")
    run(["ADEP_mvt","RUNWAY_mvt","hour_local"],K,"airport x runway x hour")
