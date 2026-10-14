import polars as pl, numpy as np
f='/home/user/prc-data-challenge/data/raw/training_2025-07-01_2025-08-01.parquet'
d=pl.scan_parquet(f).filter(pl.col('PHASE_mvt')=='DEP').select('MVT_ID_mvt','ADEP_mvt','SCHED_TIME_UTC_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','FLIGHT_mvt','STAND_mvt','ADES_mvt').collect()
p=d.join(d,on=['ADEP_mvt','SCHED_TIME_UTC_mvt'],suffix='_2').filter(pl.col('MVT_ID_mvt')<pl.col('MVT_ID_mvt_2'))
p=p.with_columns(bo=(pl.col('BLOCK_TIME_UTC_mvt')<pl.col('BLOCK_TIME_UTC_mvt_2')),mo=(pl.col('MVT_TIME_UTC_mvt')<pl.col('MVT_TIME_UTC_mvt_2')),
  fo=(pl.col('FLIGHT_mvt')<pl.col('FLIGHT_mvt_2')),so=(pl.col('STAND_mvt')<pl.col('STAND_mvt_2')),do=(pl.col('ADES_mvt')<pl.col('ADES_mvt_2')))
print('pairs',p.height)
for c in ['bo','mo','fo','so','do']: print(c,'P(ID order agrees)',p[c].mean())
dis=p.filter(pl.col('bo')!=pl.col('mo'))
print('B/M disagree pairs',dis.height,'ID follows BLOCK',dis['bo'].mean())
print(p.group_by('ADEP_mvt').agg(pl.len(),pl.col('fo').mean(),pl.col('mo').mean(),pl.col('so').mean(),pl.col('do').mean()).sort('ADEP_mvt'))
