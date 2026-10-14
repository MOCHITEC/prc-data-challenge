from common import *
d=pl.read_parquet('train_dep_cls.parquet').filter((pl.col('APT')=='LIRF')&~pl.col('unm')).with_columns(ma=(pl.col('M')-pl.col('AOBT_3_flt')).dt.total_seconds())
print(d.select([pl.col('ma').quantile(q).alias(str(q)) for q in [0.001,0.01,0.05,0.5,0.95,0.99]]+[(pl.col('ma')<0).mean().alias('neg'),(pl.col('ma')<120).mean().alias('lt2m')]))
# y - ma relation (B vs AOBT3) for LIRF matched
print(d.select(((pl.col('B')-pl.col('AOBT_3_flt')).dt.total_seconds()).quantile(0.5).alias('B-A med'), ((pl.col('B')-pl.col('AOBT_3_flt')).dt.total_seconds().abs()<=300).mean().alias('within5m')))
# ITY340 history in training
t=pl.read_parquet('train_dep_cls.parquet').filter(pl.col('FLIGHT_mvt')=='ITY340').with_columns(ma=(pl.col('M')-pl.col('AOBT_3_flt')).dt.total_seconds())
print(t.select('cls',pl.col('S').dt.strftime('%m-%d %H:%M'),'y','ms','ma').describe())
print(t.group_by('cls').len())
