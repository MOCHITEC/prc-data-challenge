import polars as pl
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(300)
a=pl.read_parquet('lirf.parquet').filter(pl.col('PHASE_mvt')=='ARR')
a=a.with_columns(mon=pl.col('SCHED_TIME_UTC_mvt').dt.month(), unm=pl.col('FLIGHT_ID_mvt').is_null(),
   ms=(pl.col('MVT_TIME_UTC_mvt')-pl.col('SCHED_TIME_UTC_mvt')).dt.total_seconds(),
   bs=(pl.col('BLOCK_TIME_UTC_mvt')-pl.col('SCHED_TIME_UTC_mvt')).dt.total_seconds(),
   ti=pl.col('TAXITIME_SEC_mvt'))
print('ARR identity check: taxi-in = BLOCK-MVT ?', a.select(((pl.col('BLOCK_TIME_UTC_mvt')-pl.col('MVT_TIME_UTC_mvt')).dt.total_seconds()-pl.col('ti')).abs().max()))
a=a.with_columns(cls=pl.when(pl.col('ti').abs()>72000).then(pl.lit('roll')).when(pl.col('bs').abs()<=6).then(pl.lit('fill')).otherwise(pl.lit('norm')))
print(a.group_by('src','unm','cls').agg(pl.len()).sort('src','unm','cls'))
# by ms band among unmatched
a=a.with_columns(mb=pl.col('ms').cut([-7200,7200,28800]))
print(a.filter(pl.col('unm')).group_by('src','mb','cls').len().sort('src','mb','cls'))
print(a.filter(pl.col('cls')=='roll').select('src','FLIGHT_mvt','unm','SCHED_TIME_UTC_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','ms','ti'))
# 2025 Jan+Jul vs 2026 for unmatched big |ms|
a.write_parquet('arr.parquet')
