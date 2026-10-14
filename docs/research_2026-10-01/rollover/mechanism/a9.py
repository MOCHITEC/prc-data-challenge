from common import *
import numpy as np
d=pl.read_parquet('train_dep_cls.parquet').filter(pl.col('APT')=='LIRF')
a=pl.read_parquet('train_arr_cls.parquet').filter(pl.col('APT')=='LIRF')
d=d.with_columns(day=pl.col('SCHED_TIME_UTC_mvt_L').dt.date(), garb=~pl.col('FLIGHT_mvt').fill_null('').str.contains(r'^[A-Z]{3}\d'))
a=a.with_columns(day=pl.col('SCHED_TIME_UTC_mvt_L').dt.date())
# garble by class among LIRF unm
du=d.filter(pl.col('unm'))
print(du.group_by('cls').agg(n=pl.len(), garb=pl.col('garb').mean()))
print(du.filter(pl.col('ms')>=43200).group_by('cls').agg(n=pl.len(), garb=pl.col('garb').mean()))
print('matched LIRF DEP garb', d.filter(~pl.col('unm')).select(pl.col('garb').mean()))
# daily
dd=du.group_by('day').agg(nd=pl.len(), fd=(pl.col('cls')=='fill').mean())
dm=d.filter(~pl.col('unm')).group_by('day').agg(fdm=(pl.col('cls')=='fill').mean())
aa=a.group_by('day').agg(na=pl.len(), fa=(pl.col('cls')=='fill').mean(), fau=((pl.col('cls')=='fill')&pl.col('unm')).sum()/pl.col('unm').sum())
j=dd.join(dm,on='day').join(aa,on='day').drop_nulls()
print(j.select(pl.corr('fd','fa'), pl.corr('fd','fau').alias('c_fau'), pl.corr('fd','fdm').alias('c_fdm'), pl.corr('fdm','fa').alias('c_fdm_fa'), pl.len()))
# row-level logistic-ish: bucket ARR daily fill rate
j=j.with_columns(q=pl.col('fa').qcut(4,labels=['q1','q2','q3','q4']))
print(du.join(j.select('day','q','fa'),on='day').group_by('q').agg(n=pl.len(), fill=(pl.col('cls')=='fill').mean(), fa=pl.col('fa').mean()).sort('q'))
# fill rate by S hour, by runway, by month for unm
print(du.with_columns(h=pl.col('SCHED_TIME_UTC_mvt_L').dt.hour()//3).group_by('h').agg(n=pl.len(), fill=(pl.col('cls')=='fill').mean()).sort('h'))
print(du.group_by('RUNWAY_mvt').agg(n=pl.len(), fill=(pl.col('cls')=='fill').mean()).sort('n'))
print(du.with_columns(st=pl.col('STAND_mvt').str.slice(0,1)).group_by('st').agg(n=pl.len(), fill=(pl.col('cls')=='fill').mean()).sort('st'))
