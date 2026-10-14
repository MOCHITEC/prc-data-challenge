from common import *
df=load('train_ext.parquet')
df=df.with_columns(
  dd_L=(pl.col('MVT_TIME_UTC_mvt_L').dt.date()-pl.col('SCHED_TIME_UTC_mvt_L').dt.date()).dt.total_days(),
  dd_U=(pl.col('M').dt.date()-pl.col('S').dt.date()).dt.total_days(),
  bs=(pl.col('B')-pl.col('S')).dt.total_seconds(),
)
df=df.with_columns(cls=pl.when(pl.col('bs').abs()<=6).then(pl.lit('fill'))
   .when((pl.col('y')-86400).is_between(-3600,7200)).then(pl.lit('roll'))
   .when(pl.col('y')>7200).then(pl.lit('big'))
   .otherwise(pl.lit('normal')))
df.write_parquet('train_dep_cls.parquet')
print(df.group_by('cls').len())
for dd in ['dd_L','dd_U']:
  print(dd)
  t=df.group_by('APT','unm',dd).agg(n=pl.len(),roll=(pl.col('cls')=='roll').sum(),fill=(pl.col('cls')=='fill').sum(),big=(pl.col('cls')=='big').sum()).filter(pl.col(dd)!=0).sort('APT','unm',dd)
  print(t)
