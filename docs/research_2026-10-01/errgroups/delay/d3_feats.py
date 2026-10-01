# build delay-specific candidate features for train (Jan+Jul 2025 matched OOF rows) and rank (parity)
import polars as pl, numpy as np
W="/home/user/prc-data-challenge/data/work/"
E="/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/errgroups/"
C=["MVT_ID_mvt","ADEP_mvt","ADES_mvt","MVT_TIME_UTC_mvt","SCHED_TIME_UTC_mvt","LOBT_flt","IOBT_flt","EOBT_1_flt","ARVT_1_flt","AOBT_3_flt","ARVT_3_flt","AIRCRAFT_TYPE_mvt"]
def load(p, months=None):
    q=pl.scan_parquet(p).select(C)
    if months: q=q.filter(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months))
    return q.collect()
tr_all=pl.scan_parquet(W+"feat_train.parquet").select(["ADEP_mvt","ADES_mvt","MVT_TIME_UTC_mvt","ARVT_3_flt"]).collect()
rk=load(W+"feat_rank.parquet"); tr=load(W+"feat_train.parquet",[1,7])
both=pl.concat([tr_all.select("ADEP_mvt","ADES_mvt","MVT_TIME_UTC_mvt","ARVT_3_flt"),rk.select("ADEP_mvt","ADES_mvt","MVT_TIME_UTC_mvt","ARVT_3_flt")])
route=both.with_columns(air=(pl.col("ARVT_3_flt")-pl.col("MVT_TIME_UTC_mvt")).dt.total_seconds()).filter(pl.col("air").is_between(600,20*3600)).group_by("ADEP_mvt","ADES_mvt").agg(air_med=pl.col("air").median(),air_n=pl.len())
del tr_all,both
ep=lambda c: pl.col(c).dt.epoch("s").cast(pl.Float64)
def feats(d):
    d=d.join(route,on=["ADEP_mvt","ADES_mvt"],how="left")
    d=d.with_columns(
      aobt_sec_nz=(pl.col("AOBT_3_flt").dt.second()!=0).cast(pl.Int8),
      lobt_minus_iobt=(ep("LOBT_flt")-ep("IOBT_flt")),
      airborne=(ep("ARVT_3_flt")-ep("MVT_TIME_UTC_mvt")),
      nm_blk2land_excess=(ep("ARVT_3_flt")-ep("AOBT_3_flt"))-(ep("ARVT_1_flt")-ep("EOBT_1_flt")),
      arvt3_minus_arvt1=(ep("ARVT_3_flt")-ep("ARVT_1_flt")),
      sched_sec_nz=(pl.col("SCHED_TIME_UTC_mvt").dt.second()!=0).cast(pl.Int8),
      aobt3_mod5=(pl.col("AOBT_3_flt").dt.minute()%5==0).cast(pl.Int8))
    d=d.with_columns(airborne_dev=pl.col("airborne")-pl.col("air_med"))
    # overtake counts within airport: pushed after me (AOBT3 later) but took off before me; pushed before me but took off after
    out=[]
    for apt, g in d.group_by("ADEP_mvt"):
        a=g["AOBT_3_flt"].dt.epoch("s").cast(pl.Float64).to_numpy(); t=g["MVT_TIME_UTC_mvt"].dt.epoch("s").cast(pl.Float64).to_numpy()
        ok=np.isfinite(a)&np.isfinite(t)
        A,T=a[ok],t[ok]
        oa=np.argsort(A); As=A[oa]; Ts_by_a=T[oa]
        n_after_first=np.full(len(a),np.nan); n_before_after=np.full(len(a),np.nan); rank_gap=np.full(len(a),np.nan)
        Ts=np.sort(T)
        for i in np.where(ok)[0]:
            lo=np.searchsorted(As,a[i],side="right"); hi=np.searchsorted(As,t[i],side="left")
            seg=Ts_by_a[lo:hi]
            n_after_first[i]=(seg<t[i]).sum()  # pushed after me, airborne before me
            # pushed before me (within 2h), airborne after me
            lo2=np.searchsorted(As,a[i]-7200,side="left"); hi2=np.searchsorted(As,a[i],side="left")
            n_before_after[i]=(Ts_by_a[lo2:hi2]>t[i]).sum()
        out.append(g.select("MVT_ID_mvt").with_columns(pl.Series("ovt_passed_me",n_after_first),pl.Series("ovt_i_passed",n_before_after)))
    d=d.join(pl.concat(out),on="MVT_ID_mvt",how="left")
    return d.select("MVT_ID_mvt","aobt_sec_nz","lobt_minus_iobt","airborne","airborne_dev","nm_blk2land_excess","arvt3_minus_arvt1","sched_sec_nz","aobt3_mod5","ovt_passed_me","ovt_i_passed")
ft=feats(tr); fr=feats(rk)
ft.write_parquet(E+"delay/newfeat_train_janjul.parquet"); fr.write_parquet(E+"delay/newfeat_rank.parquet")
# parity: null rates and means, on matched rows (AOBT3 non-null)
mt=ft.join(tr.select("MVT_ID_mvt","AOBT_3_flt"),on="MVT_ID_mvt").filter(pl.col("AOBT_3_flt").is_not_null())
mr=fr.join(rk.select("MVT_ID_mvt","AOBT_3_flt"),on="MVT_ID_mvt").filter(pl.col("AOBT_3_flt").is_not_null())
for c in ft.columns[1:]:
    print(f"{c:22s} null train {mt[c].null_count()/mt.height:.4f} rank {mr[c].null_count()/mr.height:.4f} | mean train {mt[c].mean():.2f} rank {mr[c].mean():.2f} | med {mt[c].median():.1f} {mr[c].median():.1f}")
