from common import *
d=pl.read_parquet('train_dep_cls.parquet').filter((pl.col('APT')=='LIRF')&pl.col('unm'))
print(d.group_by('ADES_mvt').len().sort('len',descending=True).head(15))
tr=pl.read_parquet('train_ext.parquet').filter((pl.col('PHASE_mvt')=='ARR')&(pl.col('ADEP_mvt')=='LIRF'))
print('ARR rows from LIRF', tr.height)
big=d.filter(pl.col('ms')>=4*3600)
j=big.join(tr.select(pl.col('FLIGHT_mvt'),pl.col('ADES_mvt'),pl.col('MVT_TIME_UTC_mvt').alias('Ma'),pl.col('AOBT_3_flt').alias('Aa'),pl.col('SCHED_TIME_UTC_mvt').alias('Sa')),on=['FLIGHT_mvt','ADES_mvt'],how='inner')
j=j.with_columns(gap=(pl.col('Ma')-pl.col('M')).dt.total_seconds()/3600).filter(pl.col('gap').abs()<40)
for r in j.select('FLIGHT_mvt','ADES_mvt','cls',pl.col('S').dt.strftime('%m-%d %H:%M'),pl.col('M').dt.strftime('%m-%d %H:%M'),pl.col('Ma').dt.strftime('%m-%d %H:%M'),pl.col('Aa').dt.strftime('%m-%d %H:%M'),'gap','ms').sort('cls').iter_rows(): print(*r)
