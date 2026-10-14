from common import *
import re
def deg(s):
    if s is None: return None
    for k in (3,2,1):
        if len(s)>3+k and s[3:3+k]==s[3-k:3]: return s[:3]+s[3+k:]
    return s
d=pl.read_parquet('train_dep_cls.parquet').filter((pl.col('APT')=='LIRF')&pl.col('unm')&(pl.col('ms')>=12*3600))
d=d.with_columns(F2=pl.col('FLIGHT_mvt').map_elements(deg,return_dtype=pl.String))
r=load('rank_ext.parquet').filter((pl.col('APT')=='LIRF')&pl.col('unm')&(pl.col('ms')>=12*3600))
r=r.with_columns(F2=pl.col('FLIGHT_mvt').map_elements(deg,return_dtype=pl.String))
TEN=list(TZ)
arr_t=pl.read_parquet('train_ext.parquet').filter((pl.col('PHASE_mvt')=='ARR')&(pl.col('ADEP_mvt')=='LIRF'))
arr_r=pl.read_parquet('rank_ext.parquet').filter((pl.col('PHASE_mvt')=='ARR')&(pl.col('ADEP_mvt')=='LIRF'))
for name,x,arr in [('train',d,arr_t),('rank',r,arr_r)]:
    print('==',name)
    for row in x.select('FLIGHT_mvt','F2','ADES_mvt','cls' if name=='train' else pl.lit('?').alias('c'),pl.col('S').dt.strftime('%m-%d %H:%M'),pl.col('M').dt.strftime('%m-%d %H:%M'),'ms').iter_rows():
        print(*row, '| in10' if row[2] in TEN else '')
    a=arr.with_columns(F2=pl.col('FLIGHT_mvt').map_elements(deg,return_dtype=pl.String))
    j=x.join(a.select('F2','ADES_mvt',pl.col('MVT_TIME_UTC_mvt').alias('Ma'),pl.col('SCHED_TIME_UTC_mvt').alias('Sa'),pl.col('AOBT_3_flt').alias('Aa'),pl.col('FLIGHT_mvt').alias('Fa')),on=['F2','ADES_mvt'])
    j=j.with_columns(gap=(pl.col('Ma')-pl.col('M')).dt.total_seconds()/3600).filter(pl.col('gap').abs()<48)
    print(j.select('FLIGHT_mvt','Fa','ADES_mvt',pl.col('S').dt.strftime('%m-%d %H:%M'),pl.col('M').dt.strftime('%m-%d %H:%M'),pl.col('Sa').dt.strftime('%m-%d %H:%M'),pl.col('Ma').dt.strftime('%m-%d %H:%M'),pl.col('Aa').dt.strftime('%m-%d %H:%M'),'gap'))
