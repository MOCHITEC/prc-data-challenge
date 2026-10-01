# Regime features from visible columns only: DEP MVT_TIME+RUNWAY of all rows, ARR MVT_TIME+RUNWAY. No BLOCK/TAXITIME.
import polars as pl, numpy as np, glob
R="/home/user/prc-data-challenge/data/raw/"
APTS=["LIRF","LTFM","LFPG","EGLL"]
cols=["MVT_ID_mvt","PHASE_mvt","ADEP_mvt","ADES_mvt","MVT_TIME_UTC_mvt","RUNWAY_mvt"]
src=sorted(glob.glob(R+"training_*.parquet"))+[R+"ranking.parquet"]
d=pl.concat([pl.scan_parquet(f).select(cols).with_columns(pl.lit("rank" in f).alias("is_rank")) for f in src]).collect()
d=d.with_columns(pl.when(pl.col("PHASE_mvt")=="DEP").then(pl.col("ADEP_mvt")).otherwise(pl.col("ADES_mvt")).alias("apt")).filter(pl.col("apt").is_in(APTS))
d=d.with_columns((pl.col("MVT_TIME_UTC_mvt").dt.epoch("s")).alias("t"))
W=1800
out=[]
for a in APTS:
    x=d.filter(pl.col("apt")==a).sort("t")
    dep=x.filter(pl.col("PHASE_mvt")=="DEP"); arr=x.filter(pl.col("PHASE_mvt")=="ARR")
    rw=[r for r in dep["RUNWAY_mvt"].drop_nulls().unique().to_list() if r!="NA"]
    rwa=[r for r in arr["RUNWAY_mvt"].drop_nulls().unique().to_list() if r!="NA"]
    t=dep["t"].to_numpy(); own=dep["RUNWAY_mvt"].fill_null("NA").to_numpy()
    def counts(times_by, lo, hi):
        M=np.zeros((len(t),len(times_by)),dtype=np.float32)
        for j,tt in enumerate(times_by):
            M[:,j]=np.searchsorted(tt,t+hi,side="right")-np.searchsorted(tt,t+lo,side="left")
        return M
    dt_={r:dep.filter(pl.col("RUNWAY_mvt")==r)["t"].to_numpy() for r in rw}
    at_={r:arr.filter(pl.col("RUNWAY_mvt")==r)["t"].to_numpy() for r in rwa}
    Dp=counts([dt_[r] for r in rw],-W,W); Dpast=counts([dt_[r] for r in rw],-W,0); Dfut=counts([dt_[r] for r in rw],0,W)
    Ap=counts([at_[r] for r in rwa],-W,W)
    idx={r:j for j,r in enumerate(rw)}; idxa={r:j for j,r in enumerate(rwa)}
    oj=np.array([idx.get(o,-1) for o in own]); oja=np.array([idxa.get(o,-1) for o in own])
    ii=np.arange(len(t))
    def own_share(M):
        tot=M.sum(1); v=np.where(oj>=0,M[ii,np.maximum(oj,0)],np.nan); return np.where(tot>0,v/np.maximum(tot,1),np.nan)
    f=dict(
      rg_own_share_pm30=own_share(Dp),
      rg_own_share_past30=own_share(Dpast), rg_own_share_fut30=own_share(Dfut),
      rg_n_dep_rwy_pm30=(Dp>=2).sum(1).astype(np.float32),
      rg_n_arr_rwy_pm30=(Ap>=2).sum(1).astype(np.float32),
      rg_arr_on_own_pm30=np.where(oja>=0,Ap[ii,np.maximum(oja,0)],0).astype(np.float32),
      rg_arr_share_on_own=np.where(Ap.sum(1)>0,np.where(oja>=0,Ap[ii,np.maximum(oja,0)],0)/np.maximum(Ap.sum(1),1),np.nan),
    )
    f["rg_share_change"]=f["rg_own_share_fut30"]-f["rg_own_share_past30"]
    # dominant dep runway set per 15-min bin (share>=10%), minutes since/until change
    b=(t//900).astype(np.int64)
    bins=np.unique(b)
    # per bin counts
    Bc=np.zeros((len(bins),len(rw)),dtype=np.float32)
    bi=np.searchsorted(bins,b)
    for j,r in enumerate(rw): np.add.at(Bc[:,j],bi[own==r],1)
    sh=Bc/np.maximum(Bc.sum(1,keepdims=True),1)
    key=(sh>=0.15).astype(np.int64)@(2**np.arange(len(rw)))
    chg=np.r_[True,key[1:]!=key[:-1]]
    last=np.maximum.accumulate(np.where(chg,np.arange(len(bins)),0))
    since=(bins-bins[last])*15.0
    nxt_idx=np.where(chg,np.arange(len(bins)),len(bins)); nxt=np.minimum.accumulate(nxt_idx[::-1])[::-1]
    nxt2=np.r_[nxt[1:],len(bins)]
    until=np.where(nxt2<len(bins),(bins[np.minimum(nxt2,len(bins)-1)]-bins)*15.0,1e4)
    f["rg_min_since_cfgchg"]=np.minimum(since[bi],720).astype(np.float32); f["rg_min_until_cfgchg"]=np.minimum(until[bi],720).astype(np.float32)
    # day-level (UTC day) own runway share and n runways
    day=(t//86400)
    df=pl.DataFrame({"day":day,"own":own})
    ds=df.group_by("day","own").len().with_columns((pl.col("len")/pl.col("len").sum().over("day")).alias("s"))
    f["rg_day_own_share"]=df.join(ds,on=["day","own"],how="left",maintain_order="left")["s"].to_numpy()
    dn=ds.filter(pl.col("s")>=0.05).group_by("day").len().rename({"len":"nd"})
    f["rg_day_n_dep_rwy"]=df.join(dn,on="day",how="left",maintain_order="left")["nd"].to_numpy().astype(np.float32)
    out.append(dep.select("MVT_ID_mvt","apt","is_rank","t").with_columns(*[pl.Series(k,np.asarray(v,dtype=np.float32)) for k,v in f.items()]))
    print(a,len(t),rw,rwa)
res=pl.concat(out); res.write_parquet("regime_feats.parquet")
pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(250)
fc=[c for c in res.columns if c.startswith("rg_")]
print(res.group_by("apt","is_rank").agg(*[pl.col(c).null_count().truediv(pl.len()).round(3).alias(c[3:]) for c in fc]).sort("apt","is_rank"))
print(res.group_by("apt","is_rank").agg(*[pl.col(c).mean().round(2).alias(c[3:]) for c in fc]).sort("apt","is_rank"))
