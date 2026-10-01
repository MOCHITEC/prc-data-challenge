from common import *
D=pl.read_parquet('train_dep_cls.parquet').filter((pl.col('APT')=='LIRF'))
u=D.filter(pl.col('unm')&(pl.col('ms')>=12*3600))
print(u.group_by('dd_L').agg(n=pl.len(), roll=(pl.col('cls')=='roll').sum()))
m=D.filter(~pl.col('unm')).with_columns(ad=(pl.col('AOBT_3_flt')-pl.col('S')).dt.total_seconds()/3600)
print(m.with_columns(b=pl.col('ad').cut([0,0.5,1,2,3,4,6])).group_by('b').agg(n=pl.len(), fill=(pl.col('cls')=='fill').mean()).sort('b'))
r=pl.read_parquet('rank_ext.parquet').filter(pl.col('PHASE_mvt')=='DEP')
print(r.filter(pl.col('FLIGHT_mvt').is_in(['ITY340'])&(pl.col('MVT_TIME_UTC_mvt')-pl.col('AOBT_3_flt')).dt.total_seconds().gt(40000)).select('MVT_ID_mvt','FLIGHT_ID_mvt'))
lst=['WMTT3508','TAY9LA','WMT402','WMTMT4SG','NOSOS431','ITYTY672','RYR4852','TARAR753','SVII3006','ITYTY674','QFAA7006']
print(r.filter(pl.col('FLIGHT_mvt').is_in(lst)&(pl.col('ADEP_mvt')=='LIRF')&((pl.col('MVT_TIME_UTC_mvt')-pl.col('SCHED_TIME_UTC_mvt')).dt.total_seconds()>=43200)).select('MVT_ID_mvt','FLIGHT_mvt'))
