import polars as pl, numpy as np
np.set_printoptions(linewidth=220, suppress=True)
S='/tmp/claude-0/-home-user-prc-data-challenge/b411042e-f283-5ee9-9c9e-ffb2443b6ce1/scratchpad'
k='MVT_ID_mvt'
d=pl.scan_parquet(f'{S}/ideas/ht/d26.parquet').select(k,'ADEP_mvt','matched','ms','mlobt','month','rec','v4','pre').filter((pl.col('ADEP_mvt')=='LIRF')&pl.col('matched')).collect()
q=pl.read_parquet(f'{S}/ideas/ht/lc/copyq_unfixed_lc.parquet')
print('q cols', q.columns, q.height)
A=pl.read_parquet(f'{S}/ideas/ht/lc/adsb_layer_detail_v2.parquet')
print('adsb detail cols', A.columns[:20])
A=A.select(k, pl.col('b').alias('blend'), pl.col('new').alias('padsb'))
I=pl.read_parquet(f'{S}/ideas/ht/lc/appear_increment_out_v2.parquet'); print('incr cols', I.columns)
d=d.join(q.select(k,'q','ms_feat'),on=k,how='left').join(A,on=k,how='left')
print('LIRF matched', d.height, 'with q', d['q'].is_not_null().sum(), 'with padsb', d['padsb'].is_not_null().sum())
d=d.filter(pl.col('q').is_not_null()).with_columns(imp=((pl.col('ms')-pl.col('mlobt')).abs()>3606))
d=d.with_columns(qb=pl.col('q').cut([0.05,0.1,0.2,0.3,0.5,0.7,0.8,0.9,0.95]))
f=lambda a,b: ((pl.col(a)-pl.col(b)).cast(pl.Float64)**2).mean().sqrt()
# by q-bin: does v4 carry a 0.4q pull toward ms?  predicted v2 chain: padsb+0.4q(ms-padsb); v4-rec
print(d.filter(~pl.col('imp')).group_by('qb').agg(n=pl.len(), qm=pl.col('q').mean(),
   rms_v4_rec=f('v4','rec'), rms_rec_padsb=f('rec','padsb'), rms_v4_padsb=f('v4','padsb'), rms_v4_pre=f('v4','pre'),
   rms_ms_padsb=f('ms','padsb'),
   # implied weight on (ms-padsb): regress (v4-padsb) on (ms-padsb) and (rec-padsb) on (ms-padsb)
   w_v4=((pl.col('v4')-pl.col('padsb'))*(pl.col('ms')-pl.col('padsb'))).sum()/((pl.col('ms')-pl.col('padsb'))**2).sum(),
   w_rec=((pl.col('rec')-pl.col('padsb'))*(pl.col('ms')-pl.col('padsb'))).sum()/((pl.col('ms')-pl.col('padsb'))**2).sum(),
   w_v4_pre=((pl.col('v4')-pl.col('pre'))*(pl.col('ms')-pl.col('pre'))).sum()/((pl.col('ms')-pl.col('pre'))**2).sum(),
   ).sort('qb'))
# per-bin implied w/q ratio for v4 and for rec
t=d.filter(~pl.col('imp')).group_by('qb').agg(qm=pl.col('q').mean(), w_v4=((pl.col('v4')-pl.col('padsb'))*(pl.col('ms')-pl.col('padsb'))).sum()/((pl.col('ms')-pl.col('padsb'))**2).sum(), w_rec=((pl.col('rec')-pl.col('padsb'))*(pl.col('ms')-pl.col('padsb'))).sum()/((pl.col('ms')-pl.col('padsb'))**2).sum()).sort('qb')
print(t.with_columns(r_v4=pl.col('w_v4')/pl.col('qm'), r_rec=pl.col('w_rec')/pl.col('qm')))
# is v4's pre-overlay base more copy-aware than v2's? compare inverted bases vs ms on high q
d=d.with_columns(v4pre=(pl.col('v4')-0.4*pl.col('q')*pl.col('ms'))/(1-0.4*pl.col('q')), v2pre=(pl.col('rec')-0.4*pl.col('q')*pl.col('ms'))/(1-0.4*pl.col('q')))
print(d.filter(~pl.col('imp')).group_by('qb').agg(n=pl.len(), rms_v4pre_v2pre=f('v4pre','v2pre'), rms_v2pre_padsb=f('v2pre','padsb'), rms_v4pre_padsb=f('v4pre','padsb'), rms_ms_v4pre=f('ms','v4pre'), rms_ms_v2pre=f('ms','v2pre'), rms_ms_pre=f('ms','pre'), mean_ms_minus_v4pre=(pl.col('ms')-pl.col('v4pre')).mean(), mean_ms_minus_v2pre=(pl.col('ms')-pl.col('v2pre')).mean()).sort('qb'))
# by month: rows and rec vs v4 differences at q>0.5
print(d.filter(~pl.col('imp')&(pl.col('q')>0.5)).group_by('month').agg(n=pl.len(), rms_v4_rec=f('v4','rec'), rms_v4pre_v2pre=f('v4pre','v2pre'), max_abs=(pl.col('v4')-pl.col('rec')).abs().max()).sort('month'))
# ms_feat vs ms
print('ms_feat==ms', (d['ms_feat']==d['ms']).mean(), 'nonmatch', (d['ms_feat']!=d['ms']).sum())
