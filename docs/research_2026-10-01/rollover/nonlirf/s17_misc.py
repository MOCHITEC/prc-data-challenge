# (a) DEP rows with BLOCK within 600 s of SCHED+-1 day: who are they and is y normal; (b) DST-change days: DEP label anomalies
import polars as pl
from datetime import date
O='/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/rollover/nonlirf/'
dep=pl.read_parquet(O+'dep_train_feat.parquet').with_columns(BmS=(pl.col('BLOCK_TIME_UTC_mvt')-pl.col('SCHED_TIME_UTC_mvt')).dt.total_seconds())
a=dep.filter(((pl.col('BmS')-86400).abs()<=600)|((pl.col('BmS')+86400).abs()<=600))
print('BLOCK ~ SCHED+-1d:',a.height); print(a.group_by('APT','unm').agg(pl.len(),pl.col('y').median(),(pl.col('y')>14400).sum().alias('tail')).sort('APT'))
d=dep.with_columns(day=pl.col('SCHED_TIME_UTC_mvt').dt.date())
dst=[date(2025,3,30),date(2025,10,26)]
for lab,f in [('DST days',pl.col('day').is_in(dst)),('other days',~pl.col('day').is_in(dst))]:
    x=d.filter(f)
    print(lab,x.height,'y<0',(x['y']<0).mean(),'y in 3000-4500',x['y'].is_between(3000,4500).mean(),'y in 6600-8100',x['y'].is_between(6600,8100).mean(),'y>4h',(x['y']>14400).sum(), 'ms in [-4500,-2700]', x['ms'].is_between(-4500,-2700).mean(), 'ms in [2700,4500]',x['ms'].is_between(2700,4500).mean())
