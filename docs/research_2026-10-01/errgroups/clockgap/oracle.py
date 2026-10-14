import polars as pl, numpy as np, lightgbm as lgb
d=pl.read_parquet("resid_frame2.parquet").filter("genuine")
d=d.with_columns(hr=pl.col("MVT_TIME_UTC_mvt").dt.truncate("1h"))
e=d["early"].to_numpy(); err=d["err"].to_numpy()
S=(err[e]**2).sum(); print("early SSE share of genuine", S/(err**2).sum())
# (1) leave-one-out cell mean oracle on early rows, cell = airport x date x hour, and airport x date
for keys in (["ADEP_mvt","day"],["ADEP_mvt","hr"]):
    x=d.with_columns(s=pl.col("err").sum().over(keys),n=pl.len().over(keys))
    loo=((x["s"]-x["err"])/(x["n"]-1).clip(1)).to_numpy(); loo[(x["n"]<=1).to_numpy()]=0
    # shrink: k/(k+20)
    k=(x["n"]-1).to_numpy(); sh=loo*k/(k+20)
    print(keys,"oracle LOO cell-mean SSE reduction: early %.1f%% all %.2f%%"%(100*(1-((err[e]-sh[e])**2).sum()/S),100*(1-((err-sh)**2).sum()/(err**2).sum())))
# (2) membership oracle: model err on early rows only, train other month
mo=d["mo"].to_numpy()
cols=["pred","mvt_minus_aobt3","aobt3_minus_eobt1","aobt3_minus_sched","aobt3_minus_lobt","mvt_minus_eobt1","ap","ref_med","nbr_aobt_delay_mean_30m","nbr_mvt_minus_aobt_mean_30m","hour_local","stand_arrival_gap","stand_aobt_turn","ma_rel","day_dly","day_short","nb_short_30"]
X=d.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()
p=np.zeros(len(d))
P=dict(objective="regression",learning_rate=0.03,num_leaves=15,min_data_in_leaf=100,verbose=-1,num_threads=1,lambda_l2=10)
for m in (1,7):
    tr=e&(mo!=m); te=e&(mo==m)
    b=lgb.train(P,lgb.Dataset(X[tr],err[tr]),300); p[te]=b.predict(X[te])
print("membership oracle: early SSE reduction %.1f%%"%(100*(1-((err[e]-p[e])**2).sum()/S)))
print("early: mean err %.0f, sd err %.0f, constant-shift oracle reduction %.1f%%"%(err[e].mean(),err[e].std(),100*err[e].mean()**2/(err[e]**2).mean()))
# also classifier AUC for early with visible features, other-month
from sklearn.metrics import roc_auc_score
Pc=dict(objective="binary",learning_rate=0.05,num_leaves=31,min_data_in_leaf=200,verbose=-1,num_threads=1)
q=np.zeros(len(d))
for m in (1,7):
    tr=mo!=m; te=mo==m
    b=lgb.train(Pc,lgb.Dataset(X[tr],e[tr].astype(float)),300); q[te]=b.predict(X[te])
print("AUC early",roc_auc_score(e,q))
for t in (0.3,0.5,0.7):
    sel=q>t; print(t,"flagged",sel.sum(),"precision",e[sel].mean() if sel.sum() else 0,"recall",(e&sel).sum()/e.sum())
g=d["gap"].to_numpy(); pr=d["pred"].to_numpy(); ma=d["mvt_minus_aobt3"].to_numpy()
for t in (0.3,0.5,0.7):
    sel=q>t
    print(t,"err mean %.0f rmse %.0f  pred-ma mean %.0f  -gap mean %.0f"%(err[sel].mean(),np.sqrt((err[sel]**2).mean()),(pr-ma)[sel].mean(),(-g)[sel].mean()))
import collections
print(d.filter(pl.Series(q>0.5)).group_by("ADEP_mvt").len().sort("len",descending=True))
for lo,hi in ((0,.1),(.1,.3),(.3,.5),(.5,1.01)):
    s=(q>=lo)&(q<hi)
    print(f"q[{lo},{hi}) rows {s.sum()} early-rows {(s&e).sum()} share of early SSE {(err[s&e]**2).sum()/S:.3f} early err mean {err[s&e].mean():.0f} | all rows in bin err mean {err[s].mean():.0f} rmse {np.sqrt((err[s]**2).mean()):.0f}")
np.save("q_early.npy",q)
x=d.with_columns(q=pl.Series(q)).filter(pl.col("early"))
x=x.with_columns(qb=pl.col("q").cut([0.1,0.5]))
pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(250)
print(x.group_by("qb").agg(n=pl.len(),jan=(pl.col("mo")==1).mean(),deice=pl.col("deicing_gate").mean(),y=pl.col("TAXITIME_SEC_mvt").median(),ma=pl.col("mvt_minus_aobt3").median(),
   ae=pl.col("aobt3_minus_eobt1").median(),gap=pl.col("gap").median(),y30=(pl.col("TAXITIME_SEC_mvt")>1800).mean(),err=pl.col("err").mean(),
   topday=pl.col("day_short").mean()).sort("qb"))
print(x.filter(pl.col("q")<0.1).group_by("ADEP_mvt").agg(n=pl.len(),sse=(pl.col("err")**2).sum()).sort("sse",descending=True))
