"""Residual LightGBM on baseline OOF err, 2-fold Jan<->Jul (train one month, eval the other). Control = same model w/o tx_ cols."""
import numpy as np, polars as pl, lightgbm as lgb, sys
E="/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/errgroups/"
base_cols=["mvt_minus_aobt3","aobt3_minus_eobt1","aobt3_minus_sched","nbr_mvt_minus_aobt_mean_60m","nbr_mvt_minus_aobt_mean_30m",
           "arr_taxi_in_mean_60m","dep_tko_prev_60m","hour_local","temp_c","precip_mm","vis_km","wind_ms","deicing_gate","month"]
o=pl.read_parquet(E+"oof_matched_janjul.parquet")
f=pl.scan_parquet("/home/user/prc-data-challenge/data/work/feat_train.parquet").select(["MVT_ID_mvt"]+base_cols).collect()
tx=pl.read_parquet(E+"longtaxi/tx_feats.parquet")
d=o.join(f,on="MVT_ID_mvt",how="left").join(tx,on="MVT_ID_mvt",how="left")
d=d.with_columns(pl.col("ADEP_mvt").cast(pl.Categorical).to_physical().cast(pl.Float32).alias("apt_code"),
                 pl.col("lday").cast(pl.Utf8).alias("day_s")).filter(~pl.col("is_rollover"))
TX=[c for c in tx.columns if c.startswith("tx_")]
sets={"control":["pred","apt_code"]+base_cols,"tx":["pred","apt_code"]+base_cols+TX,
      "tx_nodaylevel":["pred","apt_code"]+base_cols+[c for c in TX if "day" not in c]}
params=dict(objective="huber",alpha=600,learning_rate=0.03,num_leaves=31,min_data_in_leaf=400,feature_fraction=0.8,bagging_fraction=0.8,
            bagging_freq=1,lambda_l2=10,verbose=-1,num_threads=1,seed=0)
if len(sys.argv)>1 and sys.argv[1]=="l2": params["objective"]="regression"
res={}
for name,cols in sets.items():
    p=np.zeros(d.height)
    for m in (1,7):
        tr=d.filter(pl.col("month")!=m); te_mask=(d["month"]==m).to_numpy()
        X=lambda z: z.select([pl.col(c).cast(pl.Float32).fill_nan(None) for c in cols]).to_numpy()
        yt=tr["err"].clip(-3600,3600).to_numpy()
        b=lgb.train(params,lgb.Dataset(X(tr),yt),300)
        p[te_mask]=b.predict(X(d.filter(pl.col("month")==m)))
        if name=="tx" and m==1:
            imp=sorted(zip(b.feature_importance("gain"),cols),reverse=True)[:15]; print([ (n,f"{g:.2e}") for g,n in imp])
    res[name]=p
e0=d["err"].to_numpy(); y=d["TAXITIME_SEC_mvt"].to_numpy(); gen=d["genuine"].to_numpy()
day=(d["ADEP_mvt"]+"_"+d["day_s"]).to_numpy()
u,inv=np.unique(day,return_inverse=True)
rng=np.random.default_rng(0)
def rm(e,mask): return np.sqrt(np.mean(e[mask]**2))
def boot(ea,eb,mask,B=400):
    # cluster bootstrap over airport-days; diff of RMSE
    sa=np.bincount(inv[mask],ea[mask]**2,len(u)); sb=np.bincount(inv[mask],eb[mask]**2,len(u)); n=np.bincount(inv[mask],None,len(u))
    out=[]
    for _ in range(B):
        w=np.bincount(rng.integers(0,len(u),len(u)),minlength=len(u))
        out.append(np.sqrt((w*sb).sum()/(w*n).sum())-np.sqrt((w*sa).sum()/(w*n).sum()))
    return np.percentile(out,[2.5,97.5])
masks={"all_matched":np.ones(len(e0),bool),"genuine":gen,"genuine_long":gen&(y>=1800),"genuine_short":gen&(y<1800)}
for mn,mk in masks.items():
    print(mn, mk.sum(), "base", round(rm(e0,mk),2), {k:round(rm(e0-v,mk),2) for k,v in res.items()})
for k in ("tx","tx_nodaylevel"):
    for mn in ("all_matched","genuine"):
        mk=masks[mn]
        print(f"{k} vs control [{mn}]", round(rm(e0-res[k],mk)-rm(e0-res['control'],mk),3), boot(e0-res['control'],e0-res[k],mk))
        print(f"{k} vs baseline [{mn}]", round(rm(e0-res[k],mk)-rm(e0,mk),3), boot(e0,e0-res[k],mk))
for apt in sorted(set(d["ADEP_mvt"].to_list())):
    mk=gen&(d["ADEP_mvt"]==apt).to_numpy()
    print(apt, round(rm(e0,mk),1), round(rm(e0-res["control"],mk),1), round(rm(e0-res["tx"],mk),1))
np.save(E+"longtaxi/resid_tx_pred.npy",res["tx"]); np.save(E+"longtaxi/resid_ctrl_pred.npy",res["control"])
