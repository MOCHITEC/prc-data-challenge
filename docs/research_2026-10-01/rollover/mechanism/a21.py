from common import *
def deg(s):
    if s is None: return None
    for k in (3,2,1):
        if len(s)>3+k and s[3:3+k]==s[3-k:3]: return s[:3]+s[3+k:]
    return s
for name,path in [('train','train_dep_cls.parquet'),('rank',None)]:
    if path: D=pl.read_parquet(path).filter(pl.col('APT')=='LIRF')
    else:
        D=load('rank_ext.parquet').filter(pl.col('APT')=='LIRF').with_columns(cls=pl.lit('?'))
    D=D.with_columns(F2=pl.col('FLIGHT_mvt').map_elements(deg,return_dtype=pl.String))
    u=D.filter(pl.col('unm')&(pl.col('ms')>=4*3600))
    o=D.select('F2',pl.col('M').alias('Mo'),pl.col('MVT_ID_mvt').alias('ido'),pl.col('S').alias('So'),pl.col('unm').alias('unm_o'))
    j=u.join(o,on='F2').filter((pl.col('ido')!=pl.col('MVT_ID_mvt'))&((pl.col('Mo')-pl.col('M')).dt.total_seconds().abs()<=600))
    print(name, 'dup-MVT partners:', j.height)
    print(j.select('FLIGHT_mvt','cls',pl.col('S').dt.strftime('%m-%d %H:%M'),pl.col('M').dt.strftime('%m-%d %H:%M:%S'),pl.col('So').dt.strftime('%m-%d %H:%M'),pl.col('Mo').dt.strftime('%m-%d %H:%M:%S'),'unm_o','ms'))
