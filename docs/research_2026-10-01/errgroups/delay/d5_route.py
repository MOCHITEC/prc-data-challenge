import polars as pl, numpy as np, lightgbm as lgb
E="/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/errgroups/"
W="/home/user/prc-data-challenge/data/work/"
o=pl.read_parquet(E+"oof_matched_janjul.parquet")
ex=["ADES_mvt","ARVT_1_flt","EOBT_1_flt","aobt3_minus_sched","aobt3_minus_lobt","lobt_minus_eobt1","aobt3_minus_eobt1","sched_minus_lobt","mvt_minus_aobt3","mvt_minus_sched","mvt_minus_lobt","mvt_minus_iobt","nbr_aobt_delay_mean_30m","nbr_mvt_minus_aobt_mean_30m","hour_local","dep_tko_prev_30m"]
f=pl.scan_parquet(W+"feat_train.parquet").select(["MVT_ID_mvt","MVT_TIME_UTC_mvt","RUNWAY_mvt"]+ex).filter(pl.col("MVT_ID_mvt").is_in(o["MVT_ID_mvt"].implode())).collect()
nf=pl.read_parquet(E+"delay/newfeat_train_janjul.parquet")
d=o.join(f,on="MVT_ID_mvt").join(nf,on="MVT_ID_mvt",how="left")
d=d.with_columns(mo=pl.col("MVT_TIME_UTC_mvt").dt.month(), day=pl.col("MVT_TIME_UTC_mvt").dt.date(),
   apt=pl.col("ADEP_mvt").cast(pl.Categorical).to_physical().cast(pl.Float64),
   dl=pl.col("aobt3_minus_sched").is_between(1800,10800,closed="left"))
rng=np.random.default_rng(0)
for k in range(9): d=d.with_columns(pl.Series(f"noise{k}",rng.normal(size=d.height)))
NEW=["aobt_sec_nz","lobt_minus_iobt","airborne","airborne_dev","nm_blk2land_excess","arvt3_minus_arvt1","aobt3_mod5","ovt_passed_me","ovt_i_passed"]
BASE=["pred","apt"]+ex
print("rows",d.height,"genuine",d["genuine"].sum(),"delayed genuine",d.filter("genuine","dl").height)
# ---------- residual model
P=dict(objective="regression",learning_rate=0.03,num_leaves=15,min_data_in_leaf=400,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,verbose=-1,num_threads=1,seed=0)
def crossfit(cols, subset=None, rounds=300):
    out=np.zeros(d.height)
    for m in (1,7):
        trm=(d["mo"]!=m).to_numpy(); tem=~trm
        if subset is not None: trm=trm&subset
        X=d.select([pl.col(c).cast(pl.Float64) for c in cols]).to_numpy()
        b=lgb.train(P,lgb.Dataset(X[trm],d["err"].to_numpy()[trm]),rounds)
        out[tem]=b.predict(X[tem])
    return out
def boot(e0,e1,grp,B=500):
    dd=pl.DataFrame({"g":grp,"a":e0**2,"b":e1**2}).group_by("g").agg(pl.col("a").sum(),pl.col("b").sum(),n=pl.len())
    a,b_,n=dd["a"].to_numpy(),dd["b"].to_numpy(),dd["n"].to_numpy()
    r=lambda w:np.sqrt((b_*w).sum()/(n*w).sum())-np.sqrt((a*w).sum()/(n*w).sum())
    pt=r(np.ones(len(a))); bs=[r(rng.multinomial(len(a),np.ones(len(a))/len(a))) for _ in range(B)]
    return pt,np.percentile(bs,2.5),np.percentile(bs,97.5)
grp=(d["ADEP_mvt"]+"_"+d["day"].cast(pl.Utf8)).to_numpy()
err=d["err"].to_numpy(); gen=d["genuine"].to_numpy(); dl=d["dl"].to_numpy()
dls=dl  # train only on delayed rows (subset) variant

d=d.with_columns(planned_flight_s=(pl.col("ARVT_1_flt")-pl.col("EOBT_1_flt")).dt.total_seconds().cast(pl.Float64),
  ades=pl.col("ADES_mvt").cast(pl.Categorical).to_physical().cast(pl.Float64))
d=d.with_columns(air_med=pl.col("airborne")-pl.col("airborne_dev"))
err=d["err"].to_numpy()
B2=[c for c in BASE if c not in ("ADES_mvt","ARVT_1_flt","EOBT_1_flt")]+["planned_flight_s","ades","air_med"]
NZ=[f"noise{k}" for k in range(4)]
ARMS=[("B2",B2),("B2+noise4",B2+NZ),("B2+airborne",B2+["airborne"]),("B2+airborne_dev",B2+["airborne_dev"]),
 ("B2+nm_blk2land",B2+["nm_blk2land_excess"]),("B2+arvt3-arvt1",B2+["arvt3_minus_arvt1"]),
 ("B2+arvt4",B2+["airborne","airborne_dev","nm_blk2land_excess","arvt3_minus_arvt1"])]
import json
R={}
for name,cols in ARMS:
    rp=crossfit(cols); e1=err-rp; R[name]=rp
    for lab,msk in [("all",np.ones_like(gen,bool)),("genuine",gen),("gen_delayed",gen&dl)]:
        pt,lo,hi=boot(err[msk],e1[msk],grp[msk])
        print(f"{name:18s} {lab:12s} {np.sqrt((err[msk]**2).mean()):.2f} -> {np.sqrt((e1[msk]**2).mean()):.2f} d={pt:+.2f} [{lo:+.2f},{hi:+.2f}]",flush=True)
# paired: arvt4 vs noise4
for a in ["B2+airborne_dev","B2+arvt4"]:
    for lab,msk in [("all",np.ones_like(gen,bool)),("genuine",gen),("gen_delayed",gen&dl)]:
        pt,lo,hi=boot((err-R["B2+noise4"])[msk],(err-R[a])[msk],grp[msk])
        print(f"PAIRED {a} vs B2+noise4 {lab}: d={pt:+.2f} [{lo:+.2f},{hi:+.2f}]",flush=True)
np.save("resid_B2.npy",np.vstack([R[k] for k in R]))
