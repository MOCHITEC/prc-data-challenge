from common import *
for nm in ['train_ext.parquet','rank_ext.parquet']:
    a=load(nm,'ARR').with_columns(bs=(pl.col('B')-pl.col('S')).dt.total_seconds())
    print(nm, a.group_by('APT').agg(n=pl.len(), big=(pl.col('y').abs()>72000).sum(), y4h=(pl.col('y').abs()>14400).sum(), fill=(pl.col('bs').abs()<=6).mean()).sort('APT'))
    for r in a.filter(pl.col('y').abs()>72000).select('APT','FLIGHT_mvt','unm',pl.col('SCHED_TIME_UTC_mvt_L').dt.strftime('%m-%d %H:%M'),pl.col('MVT_TIME_UTC_mvt_L').dt.strftime('%m-%d %H:%M'),pl.col('BLOCK_TIME_UTC_mvt_L').dt.strftime('%m-%d %H:%M'),'y').iter_rows(): print(*r)
