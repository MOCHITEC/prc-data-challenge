"""Candidate clock-gap features. Only visible columns: DEP MVT/AOBT_3/EOBT_1/LOBT/IOBT/SCHED/ARVT_*/stand/runway/operator,
ARR rows' full data. No DEP BLOCK/TAXITIME is read. Usage: feats.py train|rank"""
import sys, glob, polars as pl, numpy as np
which=sys.argv[1]
R="/home/user/prc-data-challenge/data/raw/"
files=sorted(glob.glob(R+"training_*.parquet")) if which=="train" else [R+"ranking.parquet"]
dcols=["MVT_ID_mvt","ADEP_mvt","ADES_mvt","STAND_mvt","RUNWAY_mvt","AIRCRAFT_OPERATOR_flt","MVT_TIME_UTC_mvt","SCHED_TIME_UTC_mvt","AOBT_3_flt","EOBT_1_flt","LOBT_flt","IOBT_flt","ARVT_1_flt","ARVT_3_flt"]
raw=pl.concat([pl.scan_parquet(f).select(dcols+["PHASE_mvt","BLOCK_TIME_UTC_mvt","TAXITIME_SEC_mvt"]) for f in files]).collect()
dep=raw.filter(pl.col("PHASE_mvt")=="DEP").select(dcols)
arr=raw.filter(pl.col("PHASE_mvt")=="ARR").select(pl.col("ADES_mvt").alias("AP"),"STAND_mvt","RUNWAY_mvt","BLOCK_TIME_UTC_mvt","MVT_TIME_UTC_mvt","TAXITIME_SEC_mvt")
del raw
s=lambda a,b:(pl.col(a)-pl.col(b)).dt.total_seconds().cast(pl.Float64)
dep=dep.with_columns(m_a=s("MVT_TIME_UTC_mvt","AOBT_3_flt"),a_e=s("AOBT_3_flt","EOBT_1_flt"),l_i=s("LOBT_flt","IOBT_flt"),
   e_l=s("EOBT_1_flt","LOBT_flt"),fltd=s("ARVT_3_flt","AOBT_3_flt")-s("ARVT_1_flt","EOBT_1_flt"),air=s("ARVT_3_flt","MVT_TIME_UTC_mvt"),
   t=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64),day=pl.col("MVT_TIME_UTC_mvt").dt.date())
# references from the same frame (non-label column m_a; computable on ranking the same way)
dep=dep.with_columns(ref_sr=pl.col("m_a").median().over(["ADEP_mvt","STAND_mvt","RUNWAY_mvt"]),n_sr=pl.len().over(["ADEP_mvt","STAND_mvt","RUNWAY_mvt"]),
                     ref_ar=pl.col("m_a").median().over(["ADEP_mvt","RUNWAY_mvt"]),
                     air_ref=pl.col("air").median().over(["ADEP_mvt","ADES_mvt"]))
dep=dep.with_columns(ref=pl.when(pl.col("n_sr")>=30).then("ref_sr").otherwise("ref_ar"))
dep=dep.with_columns(ma_rel=pl.col("m_a")-pl.col("ref"),air_rel=pl.col("air")-pl.col("air_ref"))
dep=dep.with_columns(short=(pl.col("ma_rel")< -300).cast(pl.Float64),dly=(pl.col("a_e")>900).cast(pl.Float64),rev=(pl.col("l_i").abs()>60).cast(pl.Float64))
arr=arr.with_columns(ti_rel=pl.col("TAXITIME_SEC_mvt")-pl.col("TAXITIME_SEC_mvt").median().over(["AP","STAND_mvt","RUNWAY_mvt"]),
                     t=pl.col("BLOCK_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64))
def win(src_t, src_v, q_t, w, excl_self=False):
    o=np.argsort(src_t); st=src_t[o]; sv=np.nan_to_num(src_v[o]); fin=np.isfinite(src_v[o]).astype(float)
    cs=np.concatenate([[0],np.cumsum(sv)]); cn=np.concatenate([[0],np.cumsum(fin)])
    lo=np.searchsorted(st,q_t-w,"left"); hi=np.searchsorted(st,q_t+w,"right")
    return cs[hi]-cs[lo], cn[hi]-cn[lo]
out=[]
for ap in dep["ADEP_mvt"].unique().to_list():
    d=dep.filter(pl.col("ADEP_mvt")==ap); a=arr.filter(pl.col("AP")==ap)
    t=d["t"].to_numpy(); res={"MVT_ID_mvt":d["MVT_ID_mvt"].to_numpy()}
    for c in ["ma_rel","short","dly","rev","a_e"]:
        v=d[c].to_numpy().astype(float)
        for w in (1800,3600):
            sm,n=win(t,v,t,w); own=np.isfinite(v)
            sm=sm-np.where(own,v,0); n=n-own
            res[f"nb_{c}_{w//60}"]=np.where(n>0,sm/np.maximum(n,1),np.nan)
    # operator +-3h
    op=d["AIRCRAFT_OPERATOR_flt"].fill_null("?").to_numpy()
    v=d["ma_rel"].to_numpy().astype(float); col=np.full(len(t),np.nan); cnt=np.zeros(len(t))
    for u in np.unique(op):
        ii=np.where(op==u)[0]; sm,n=win(t[ii],v[ii],t[ii],10800); own=np.isfinite(v[ii]); sm-=np.where(own,v[ii],0); n-=own
        col[ii]=np.where(n>0,sm/np.maximum(n,1),np.nan); cnt[ii]=n
    res["op3h_ma_rel"]=col; res["op3h_n"]=cnt
    at=a["t"].to_numpy(); av=a["ti_rel"].cast(pl.Float64).to_numpy()
    for w in (1800,3600):
        sm,n=win(at[np.isfinite(at)],av[np.isfinite(at)],t,w); res[f"arr_ti_rel_{w//60}"]=np.where(n>0,sm/np.maximum(n,1),np.nan)
    out.append(pl.DataFrame(res))
F=pl.concat(out)
dd=dep.select("MVT_ID_mvt","ADEP_mvt","day","ma_rel","l_i","e_l","fltd","air_rel","short","dly")
dd=dd.with_columns(day_short=pl.col("short").mean().over(["ADEP_mvt","day"]),day_dly=pl.col("dly").mean().over(["ADEP_mvt","day"]),
                   day_ma_rel=pl.col("ma_rel").mean().over(["ADEP_mvt","day"]))
F=dd.drop("ADEP_mvt","day","short","dly").join(F,on="MVT_ID_mvt")
F.write_parquet(f"cand_{which}.parquet")
print(F.height); print(F.null_count())
