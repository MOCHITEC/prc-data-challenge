from common import *
df=load('train_ext.parquet')
r=df.filter(pl.col('y')>72000).sort('y')
r=r.with_columns(Bt=pl.col('B')+pl.duration(days=1), Bt_L=pl.col('BLOCK_TIME_UTC_mvt_L')+pl.duration(days=1))
r=r.with_columns(
  dS_U=pl.col('S').dt.date(), dB_U=pl.col('B').dt.date(), dM_U=pl.col('M').dt.date(),
  dS_L=pl.col('SCHED_TIME_UTC_mvt_L').dt.date(), dB_L=pl.col('BLOCK_TIME_UTC_mvt_L').dt.date(), dM_L=pl.col('MVT_TIME_UTC_mvt_L').dt.date(),
)
print(r.select('APT','FLIGHT_mvt','unm','SCHED_TIME_UTC_mvt_L','BLOCK_TIME_UTC_mvt_L','MVT_TIME_UTC_mvt_L','y','ms',
  (pl.col('dB_U')==pl.col('dS_U')).alias('B=S_U'),(pl.col('dB_L')==pl.col('dS_L')).alias('B=S_L'),
  (pl.col('dM_L')-pl.col('dS_L')).dt.total_days().alias('M-S_L'),
  (pl.col('BLOCK_TIME_UTC_mvt_L').dt.hour()).alias('Bh_L'), 'STAND_mvt','RUNWAY_mvt'))
