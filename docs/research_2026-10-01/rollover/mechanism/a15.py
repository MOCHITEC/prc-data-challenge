from common import *
r=load('rank_ext.parquet')
r=r.with_columns(ma=(pl.col('M')-pl.col('AOBT_3_flt')).dt.total_seconds(),
  dd_L=(pl.col('MVT_TIME_UTC_mvt_L').dt.date()-pl.col('SCHED_TIME_UTC_mvt_L').dt.date()).dt.total_days())
print('null B', r.select(pl.col('B').is_null().mean()))
print('ma>40000:'); 
for x in r.filter(pl.col('ma').abs()>40000).select('APT','FLIGHT_mvt','unm',pl.col('S').dt.strftime('%m-%d %H:%M'),pl.col('M').dt.strftime('%m-%d %H:%M:%S'),pl.col('AOBT_3_flt').dt.strftime('%m-%d %H:%M'),'ms','ma').iter_rows(): print(*x)
x=r.filter((pl.col('APT')=='LIRF')&(pl.col('ms')>=12*3600)).sort('unm','ms')
print('LIRF ms>=12h')
for y in x.select('FLIGHT_mvt','unm',pl.col('SCHED_TIME_UTC_mvt_L').dt.strftime('%m-%d %H:%M').alias('S_L'),pl.col('MVT_TIME_UTC_mvt_L').dt.strftime('%m-%d %H:%M:%S').alias('M_L'),'ms',(pl.col('ms')/3600).round(2).alias('msh'),'dd_L','ma','STAND_mvt','RUNWAY_mvt','AIRCRAFT_TYPE_mvt').iter_rows(): print(*y)
# all-airport counts in bands
br=[12*3600,14*3600,16*3600,28*3600,36*3600]
print(r.filter(pl.col('ms')>=12*3600).with_columns(band=pl.col('ms').cut(br,left_closed=True)).group_by('APT','unm','band').len().sort('APT','unm','band'))
