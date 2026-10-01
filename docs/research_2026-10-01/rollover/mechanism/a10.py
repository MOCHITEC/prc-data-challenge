from common import *
d=pl.read_parquet('train_dep_cls.parquet')
def off(c): 
    s=pl.col(c).dt.second().cast(pl.Int32)
    return pl.when(s>30).then(s-60).otherwise(s)
d=d.with_columns(eM=off('M'), eB=off('B'), eS=off('S'))
print(d.group_by('APT').agg(n=pl.len(), eM_in6=(pl.col('eM').abs()<=6).mean(), eB_in6=(pl.col('eB').abs()<=6).mean(), eS0=(pl.col('eS')==0).mean(), anti=((pl.col('eM')+pl.col('eB')).abs()<=1).mean(), corr=pl.corr('eM','eB')).sort('APT'))
x=d.filter(pl.col('APT')=='LIRF')
print(x.group_by('cls').agg(n=pl.len(), eMabs=pl.col('eM').abs().mean(), anti=((pl.col('eM')+pl.col('eB')).abs()<=1).mean()))
# relation of jitter to y
x=x.filter(pl.col('cls')=='normal').with_columns(ymin=(pl.col('y')/60).round(0), s=pl.col('eM')-pl.col('eB'))
print(x.group_by(pl.col('y').cut([300,600,900,1200,1800,3600])).agg(n=pl.len(), absE=pl.col('eM').abs().mean(), sd=pl.col('s').std()))
