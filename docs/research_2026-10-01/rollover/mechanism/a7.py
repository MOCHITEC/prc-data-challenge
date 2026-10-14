from common import *
df=pl.read_parquet('train_dep_cls.parquet')
x=df.filter((pl.col('APT')=='LIRF')&pl.col('unm')&(pl.col('ms')>=4*3600)).sort('ms')
x=x.select('FLIGHT_mvt','cls',pl.col('SCHED_TIME_UTC_mvt_L').dt.strftime('%m-%d %H:%M').alias('S'),pl.col('BLOCK_TIME_UTC_mvt_L').dt.strftime('%m-%d %H:%M:%S').alias('B'),pl.col('MVT_TIME_UTC_mvt_L').dt.strftime('%m-%d %H:%M:%S').alias('Mv'),'y','ms','dd_L','STAND_mvt','RUNWAY_mvt')
for r in x.iter_rows(): print(*r)
