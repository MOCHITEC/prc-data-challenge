import polars as pl, numpy as np
pl.Config.set_tbl_rows(20); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(20)
S='/tmp/claude-0/-home-user-prc-data-challenge/b411042e-f283-5ee9-9c9e-ffb2443b6ce1/scratchpad'; L=S+'/ideas/ht/lc/'
q=pl.read_parquet(L+'copyq_unfixed_lc.parquet')
A=pl.read_parquet(L+'adsb_layer_detail_v2.parquet').select('MVT_ID_mvt',pl.col('b').alias('blend'),pl.col('new').alias('padsb'))
I=pl.read_parquet(L+'appear_increment_out_v2.parquet').rename({'TAXITIME_SEC_mvt':'aout'})
f=pl.read_parquet(f'{S}/ideas/ht/d26.parquet').filter(pl.col('matched')&(pl.col('ADEP_mvt')=='LIRF')).join(q,on='MVT_ID_mvt').join(A,on='MVT_ID_mvt').join(I,on='MVT_ID_mvt')
fj=pl.col('month').is_in([2,6])
f=f.with_columns(incr=pl.when(fj).then(pl.col('aout')-pl.col('blend').round(0)).otherwise(0.0), lo=(pl.col('mlobt')-3606).cast(pl.Float64), hi=(pl.col('mlobt')+3606).cast(pl.Float64),
                 imp=((pl.col('ms_feat')-pl.col('mlobt')).abs()>3606))
ov=pl.col('padsb')+0.4*pl.col('q')*(pl.col('ms_feat')-pl.col('padsb'))
f=f.with_columns(ov=pl.when(ov<0).then(pl.col('padsb')).otherwise(ov))
f=f.with_columns(chain=(pl.col('ov')+pl.col('incr')).clip(pl.col('lo'),pl.col('hi')).clip(0,None))
f=f.with_columns(resid=pl.col('rec')-pl.col('chain'))           # 0 on Feb/Jun; v48-lineage residual on Jan/Jul
f=f.with_columns(gated=pl.when(pl.col('imp')).then((pl.col('padsb')+pl.col('incr')+pl.col('resid')).clip(pl.col('lo'),pl.col('hi')).clip(0,None).round(0)).otherwise(pl.col('rec').cast(pl.Float64)))
f=f.with_columns(ch=pl.col('gated')-pl.col('rec'))
z=f.filter(pl.col('imp'))
print('resid on Feb/Jun impossible rows: max abs', z.filter(fj)['resid'].abs().max(), '; Jan/Jul resid rms', (z.filter(~fj)['resid']**2).mean()**.5)
print(z.group_by('month').agg(n=pl.len(), moved=(pl.col('ch').abs()>=1).sum(), sumsq=(pl.col('ch')**2).sum(), maxabs=pl.col('ch').abs().max(), mean=pl.col('ch').mean()).sort('month'))
for lab,cond in [('all',pl.lit(True)),('Feb/Jun',fj),('Jan/Jul',~fj)]:
    s=float((z.filter(cond)['ch']**2).sum()); print(f'{lab}: sum sq change {s:.3g} -> MSE {s/670790:.1f} s^2; gain at ratio 1: {s/670790/(2*233):.3f} s; at 2025 ratio 3.7: {3.7*s/670790/(2*233):.3f} s')
f.filter(pl.col('imp')&(pl.col('ch').abs()>=1)).select('MVT_ID_mvt','month',pl.col('rec').alias('old'),pl.col('gated').cast(pl.Int32).alias('new')).write_parquet(f'{S}/ideas/ht/lirf_copy_gate_rows.parquet')
print(f.filter(pl.col('imp')&(pl.col('ch').abs()>=1)).height,'rows written')
