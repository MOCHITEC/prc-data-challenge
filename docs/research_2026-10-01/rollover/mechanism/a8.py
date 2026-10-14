from common import *
a=load('train_ext.parquet','ARR')
# ARR: taxi-in = MVT - BLOCK? check sign
a=a.with_columns(bs=(pl.col('B')-pl.col('S')).dt.total_seconds(), bm=(pl.col('B')-pl.col('M')).dt.total_seconds(),
   dd_L=(pl.col('BLOCK_TIME_UTC_mvt_L').dt.date()-pl.col('SCHED_TIME_UTC_mvt_L').dt.date()).dt.total_days())
print(a.select(med=pl.col('y').quantile(0.5), idp=(pl.col('y')-(pl.col('M')-pl.col('B')).dt.total_seconds()).abs().max(), idm=(pl.col('y')+(pl.col('M')-pl.col('B')).dt.total_seconds()).abs().max()))
a=a.with_columns(cls=pl.when(pl.col('bs').abs()<=6).then(pl.lit('fill'))
   .when((pl.col('y').abs()-86400).is_between(-3600,7200)).then(pl.lit('roll'))
   .when(pl.col('y').abs()>7200).then(pl.lit('big')).otherwise(pl.lit('normal')))
a.write_parquet('train_arr_cls.parquet')
print(a.group_by('APT','unm').agg(n=pl.len(), fill=(pl.col('cls')=='fill').sum(), roll=(pl.col('cls')=='roll').sum(), big=(pl.col('cls')=='big').sum(), ypos=(pl.col('y')>0).mean()).sort('APT','unm'))
r=a.filter(pl.col('cls')=='roll').sort('APT')
for row in r.select('APT','unm','FLIGHT_mvt',pl.col('SCHED_TIME_UTC_mvt_L').dt.strftime('%m-%d %H:%M'),pl.col('BLOCK_TIME_UTC_mvt_L').dt.strftime('%m-%d %H:%M:%S'),pl.col('MVT_TIME_UTC_mvt_L').dt.strftime('%m-%d %H:%M:%S'),'y','ms').iter_rows(): print(*row)
