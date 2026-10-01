from common import *
def deg(s):
    if s is None: return None
    for k in (3,2,1):
        if len(s)>3+k and s[3:3+k]==s[3-k:3]: return s[:3]+s[3+k:]
    return s
D=pl.read_parquet('train_dep_cls.parquet').filter(pl.col('APT')=='LIRF').with_columns(F2=pl.col('FLIGHT_mvt').map_elements(deg,return_dtype=pl.String))
u=D.filter(pl.col('unm')&(pl.col('ms')>=12*3600))
oth=D.select('F2',pl.col('S').alias('So'),pl.col('M').alias('Mo'),pl.col('B').alias('Bo'),pl.col('unm').alias('unm_o'),pl.col('cls').alias('cls_o'),pl.col('FLIGHT_mvt').alias('Fo'),pl.col('MVT_ID_mvt').alias('ido'))
j=u.join(oth,on='F2').filter(pl.col('ido')!=pl.col('MVT_ID_mvt')).with_columns(dS=(pl.col('So')-pl.col('S')).dt.total_seconds()/3600, dM=(pl.col('Mo')-pl.col('M')).dt.total_seconds()/3600).filter(pl.col('dS').abs()<=48)
for r in j.sort('cls','S').select('FLIGHT_mvt','cls',pl.col('S').dt.strftime('%m-%d %H:%M'),pl.col('M').dt.strftime('%m-%d %H:%M'),'Fo','unm_o','cls_o',pl.col('So').dt.strftime('%m-%d %H:%M'),pl.col('Bo').dt.strftime('%m-%d %H:%M'),pl.col('Mo').dt.strftime('%m-%d %H:%M'),'dS','dM').iter_rows(): print(*r)
