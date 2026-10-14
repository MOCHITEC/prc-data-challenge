from common import *
df=pl.read_parquet('train_dep_cls.parquet').filter(pl.col('cls')!='fill')
# written B-S distribution in hours, per airport x unm, tails
df=df.with_columns(bsh=pl.col('bs')/3600, msh=pl.col('ms')/3600)
t=df.group_by('APT','unm').agg(n=pl.len(), bs_min=pl.col('bsh').min(), bs_q001=pl.col('bsh').quantile(0.0001), bs_max=pl.col('bsh').max(), bs_q999=pl.col('bsh').quantile(0.9999),
  n_bs_gt8=(pl.col('bsh')>8).sum(), n_bs_gt12=(pl.col('bsh')>12).sum(), n_bs_gt14=(pl.col('bsh')>14).sum(), n_bs_lt_m8=(pl.col('bsh')<-8).sum(),n_bs_lt_m12=(pl.col('bsh')<-12).sum(),
  n_ms_gt12=(pl.col('msh')>12).sum(), n_ms_gt12_roll=((pl.col('msh')>12)&(pl.col('cls')=='roll')).sum()).sort('APT','unm')
print(t)
# LIRF : histogram of bs hours for non-fill rows with bsh>4 or < -2
for u in [True, False]:
  x=df.filter((pl.col('APT')=='LIRF')&(pl.col('unm')==u))
  print('LIRF unm',u, x.with_columns(b=pl.col('bsh').floor()).group_by('b').agg(n=pl.len(), roll=(pl.col('cls')=='roll').sum()).filter((pl.col('b')>=4)|(pl.col('b')<=-2)).sort('b'))
