# Per-airport specialist arms. usage: a04_specialist.py APT
import os, sys, numpy as np, polars as pl, lightgbm as lgb, datetime as dt
A=sys.argv[1]; ARMS=sys.argv[2].split(",") if len(sys.argv)>2 else ["S0","S1","SN","S0p","S0w"]
S="/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/errgroups/"
H=S+"airports/"
f=pl.scan_parquet("/home/user/prc-data-challenge/data/work/feat_train.parquet").filter((pl.col("ADEP_mvt")==A)&pl.col("TAXITIME_SEC_mvt").is_not_null()&pl.col("mvt_minus_aobt3").is_not_null()).collect()
f=f.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mo"))
num=[c for c,d in f.schema.items() if d in (pl.Float64,pl.Float32,pl.Int64,pl.Int32,pl.Int8,pl.Int16,pl.UInt32,pl.UInt8)
     and c not in ("TAXITIME_SEC_mvt","MVT_ID_mvt","FLIGHT_ID_mvt","month","mo")]
rg=pl.read_parquet(H+"regime_feats.parquet").filter(pl.col("apt")==A).drop("apt","is_rank","t")
RG=[c for c in rg.columns if c.startswith("rg_")]
f=f.join(rg,on="MVT_ID_mvt",how="left")
# noise columns: deterministic hash of MVT_ID (no signal)
rng=np.random.default_rng(12345)
for k in range(len(RG)):
    h=((f["MVT_ID_mvt"].cast(pl.Int64).to_numpy()*(2654435761+k*97))%1000003)/1000003.0
    f=f.with_columns(pl.Series(f"nz_{k}",h.astype(np.float32)))
NZ=[f"nz_{k}" for k in range(len(RG))]
CATS=["RUNWAY_mvt","STAND_mvt","AIRCRAFT_TYPE_mvt","AIRCRAFT_OPERATOR_flt"]
for c in CATS: f=f.with_columns(pl.col(c).fill_null("NA").cast(pl.Categorical).to_physical().alias(c+"_c"))
CC=[c+"_c" for c in CATS]
f=f.with_columns((pl.col("TAXITIME_SEC_mvt")-pl.col("mvt_minus_aobt3")).alias("g"))
t=f["MVT_TIME_UTC_mvt"]
tr_all=~f["mo"].is_in([1,7]); te=f["mo"].is_in([1,7])
# period masks
def nonrep(fr):
    ts=fr["MVT_TIME_UTC_mvt"]
    if A=="LFPG":
        return ((ts>=dt.datetime(2025,4,1,tzinfo=dt.timezone.utc))&(ts<dt.datetime(2025,6,1,tzinfo=dt.timezone.utc))) | \
               ((ts>=dt.datetime(2025,7,14,tzinfo=dt.timezone.utc))&(ts<dt.datetime(2025,12,8,tzinfo=dt.timezone.utc)))
    if A=="LTFM":
        return ts<dt.datetime(2025,4,17,tzinfo=dt.timezone.utc)
    return pl.Series([False]*fr.height)
nr=nonrep(f)
def te_enc(trm):
    # stand x runway median g, shrunk, fit on training rows only
    tr=f.filter(trm); gm=float(tr["g"].median())
    s=tr.group_by("STAND_mvt","RUNWAY_mvt").agg(pl.col("g").median().alias("m"),pl.len().alias("n"))
    s=s.with_columns(((pl.col("m")*pl.col("n")+gm*20)/(pl.col("n")+20)).alias("te_sr"))
    return f.join(s.select("STAND_mvt","RUNWAY_mvt","te_sr"),on=["STAND_mvt","RUNWAY_mvt"],how="left",maintain_order="left")["te_sr"].fill_null(gm)
P=dict(objective="regression",learning_rate=0.05,num_leaves=127,min_data_in_leaf=50,feature_fraction=1.0,bagging_fraction=0.8,
       bagging_freq=1,verbose=-1,num_threads=1,seed=int(os.environ.get('SEED','0')),cat_smooth=20,max_cat_to_onehot=8)
res=f.filter(te).select("MVT_ID_mvt","mvt_minus_aobt3","MVT_TIME_UTC_mvt")
for arm in ARMS:
    trm=tr_all.to_numpy().copy(); w=None
    if arm=="S0p": trm&=~nr.to_numpy()
    SUB={"S1day":["rg_day_own_share","rg_day_n_dep_rwy"],"S1win":["rg_own_share_pm30","rg_own_share_past30","rg_own_share_fut30","rg_share_change","rg_n_dep_rwy_pm30"],
         "S1arr":["rg_arr_on_own_pm30","rg_arr_share_on_own","rg_n_arr_rwy_pm30"],"S1chg":["rg_min_since_cfgchg","rg_min_until_cfgchg"]}
    feats=num+CC+(RG if arm=="S1" else [])+(NZ if arm=="SN" else [])+SUB.get(arm,[])
    if arm=="S0p" and nr.sum()==0: continue
    te_sr=te_enc(pl.Series(trm))
    X=f.select([pl.col(c).cast(pl.Float32) for c in feats]).with_columns(te_sr.cast(pl.Float32).alias("te_sr")).to_numpy()
    if arm=="S0w": w=np.where(nr.to_numpy()[trm],0.3,1.0)
    ds=lgb.Dataset(X[trm],f["g"].to_numpy()[trm],weight=w,feature_name=feats+["te_sr"],categorical_feature=CC,free_raw_data=True)
    b=lgb.train(P,ds,500)
    p=f["mvt_minus_aobt3"].to_numpy()[te.to_numpy()]+b.predict(X[te.to_numpy()])
    res=res.with_columns(pl.Series(arm,p))
    if arm=="S1":
        imp=sorted(zip(b.feature_importance("gain"),feats+["te_sr"]),reverse=True)
        print("S1 regime gain ranks:",[(i,n,round(g/sum(x for x,_ in imp)*100,2)) for i,(g,n) in enumerate(imp) if n.startswith("rg_")])
    print(A,arm,"trained on",int(trm.sum()),flush=True)
res.write_parquet(H+f"spec_{A}"+os.environ.get("SUF","")+".parquet")
