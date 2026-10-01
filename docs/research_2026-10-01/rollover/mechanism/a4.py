from common import *
df=pl.read_parquet('train_dep_cls.parquet')
x=df.filter((pl.col('APT')=='LIRF')&pl.col('unm')&(pl.col('dd_L')==1)&(pl.col('cls')!='fill')).sort('cls','MVT_TIME_UTC_mvt_L')
print(x.select('FLIGHT_mvt','cls','SCHED_TIME_UTC_mvt_L','BLOCK_TIME_UTC_mvt_L','MVT_TIME_UTC_mvt_L','y','ms','bs','STAND_mvt','RUNWAY_mvt','AIRCRAFT_TYPE_mvt'))
