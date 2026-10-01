from common import *
d=pl.read_parquet('train_dep_cls.parquet').with_columns(ma=(pl.col('M')-pl.col('AOBT_3_flt')).dt.total_seconds())
x=d.filter(pl.col('ma').abs()>40000).sort('ma')
print(x.group_by('APT','cls').len().sort('APT'))
for r in x.select('APT','FLIGHT_mvt','cls',pl.col('S').dt.strftime('%m-%d %H:%M'),pl.col('B').dt.strftime('%m-%d %H:%M'),pl.col('M').dt.strftime('%m-%d %H:%M'),pl.col('AOBT_3_flt').dt.strftime('%m-%d %H:%M'),'y','ms','ma').iter_rows(): print(*r)
