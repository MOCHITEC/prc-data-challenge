"""LIRF copy-overlay re-weight: replace pull 0.4*q by q**1.5 on matched LIRF rows that are NOT gated (|ms - mlobt| <= 3606).
Usage: python -I apply_q15_to_v4.py <in_final.parquet> <out.parquet>
Row-by-row: p_pre = (p - 0.4*q*ms)/(1 - 0.4*q);  p_new = p_pre + q**1.5*(ms - p_pre); revert to p_pre if p_new < 0;
clip to [ceil(mlobt-3606), ceil(mlobt+3606)]; floor 0; round. Commutes with the Jan/Jul +1*delta dither and is disjoint from the gate rows."""
import polars as pl, numpy as np, sys
S='/tmp/claude-0/-home-user-prc-data-challenge/b411042e-f283-5ee9-9c9e-ffb2443b6ce1/scratchpad'
k,t='MVT_ID_mvt','TAXITIME_SEC_mvt'
inp,outp=sys.argv[1],sys.argv[2]
F=pl.read_parquet(inp)
R=(pl.scan_parquet('/home/user/prc-data-challenge/data/raw/final_ranking.parquet').filter((pl.col('PHASE_mvt')=='DEP')&(pl.col('ADEP_mvt')=='LIRF')&pl.col('AOBT_3_flt').is_not_null())
   .select(k,(pl.col('MVT_TIME_UTC_mvt')-pl.col('SCHED_TIME_UTC_mvt')).dt.total_seconds().cast(pl.Float64).alias('ms'),(pl.col('MVT_TIME_UTC_mvt')-pl.col('LOBT_flt')).dt.total_seconds().cast(pl.Float64).alias('mlobt')).collect())
Q=pl.read_parquet(f'{S}/ideas/ht/lc/copyq_unfixed_lc.parquet').select(k,'q')
X=R.join(Q,on=k,how='inner').filter((pl.col('ms')-pl.col('mlobt')).abs()<=3606)
Z=F.join(X,on=k,how='left',maintain_order='left')
p=pl.col(t).cast(pl.Float64); q=pl.col('q'); ms=pl.col('ms')
ppre=(p-0.4*q*ms)/(1-0.4*q)
pn=ppre+q**1.5*(ms-ppre); pn=pl.when(pn<0).then(ppre).otherwise(pn)
lo=(pl.col('mlobt')-3606).ceil(); hi=(pl.col('mlobt')+3606).ceil()
pn=pl.max_horizontal(pl.min_horizontal(pn,hi),lo).clip(0).round(0)
out=Z.select(k, pl.when(pl.col('q').is_not_null()).then(pn).otherwise(p).cast(pl.Int32).alias(t))
assert out.height==F.height and out[t].null_count()==0 and (out[t]>=0).all()
out.write_parquet(outp)
d=(out[t].cast(pl.Float64)-F[t].cast(pl.Float64)).to_numpy()
print(f'rows changed {(d!=0).sum()}, sum d^2/N {float((d**2).sum())/len(d):.1f} s^2, max|d| {int(np.abs(d).max())}')
