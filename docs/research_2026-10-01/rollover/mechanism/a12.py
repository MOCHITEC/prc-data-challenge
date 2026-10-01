from common import *
d=pl.read_parquet('train_dep_cls.parquet').filter((pl.col('APT')=='LIRF')&(pl.col('month')==7)).sort('MVT_ID_mvt')
print(d.select('MVT_ID_mvt','FLIGHT_mvt','S','B','M','cls').head(15))

# monotonic checks: fraction of consecutive pairs where key non-decreasing
for k in ['S','B','M']:
    print(k, d.select((pl.col(k).diff().dt.total_seconds()>=0).mean()))
# day-level
for k in ['S','B','M']:
    print(k,'date', d.select((pl.col(k).dt.date().diff().dt.total_days()>=0).mean()))
