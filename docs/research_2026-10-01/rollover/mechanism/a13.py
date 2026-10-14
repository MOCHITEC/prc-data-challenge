from common import *
d=pl.read_parquet('train_dep_cls.parquet')
br=[4*3600,8*3600,12*3600,14*3600,16*3600,28*3600,36*3600]
x=d.filter(pl.col('ms')>=4*3600).with_columns(band=pl.col('ms').cut(br,left_closed=True))
t=x.group_by('APT','unm','band').agg(n=pl.len(), fill=(pl.col('cls')=='fill').sum(), roll=(pl.col('cls')=='roll').sum(), big=(pl.col('cls')=='big').sum(), normal=(pl.col('cls')=='normal').sum()).sort('APT','unm','band')
print(t.filter(pl.col('APT')=='LIRF'))
print(t.filter(pl.col('APT')!='LIRF').group_by('unm','band').agg(pl.col('n','fill','roll','big','normal').sum()).sort('unm','band'))
