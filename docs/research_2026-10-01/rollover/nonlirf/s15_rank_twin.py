# Ranking: rows flagged as twins, and ARR rows with |y|>4h (labels visible for ARR) by airport
import polars as pl
O='/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/rollover/nonlirf/'
rk=pl.read_parquet(O+'rank.parquet').join(pl.read_parquet(O+'twin_rank.parquet'),on='MVT_ID_mvt',how='left').with_columns(
    y=(pl.col('BLOCK_TIME_UTC_mvt')-pl.col('MVT_TIME_UTC_mvt')).dt.total_seconds(), unm=pl.col('FLIGHT_ID_mvt').is_null())
f=lambda x: x.with_columns([pl.col(c).dt.strftime('%m-%d %H:%M:%S') for c in ['SCHED_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','MVT_TIME_UTC_mvt']]).to_pandas().to_string(index=False)
print(f(rk.filter(pl.col('twin')).select('PHASE_mvt','APT','FLIGHT_mvt','unm','SCHED_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','MVT_TIME_UTC_mvt','y','dprev','dnext','sdiff_prev','sdiff_next')))
a=rk.filter((pl.col('PHASE_mvt')=='ARR')&(pl.col('y').abs()>14400))
print('ranking ARR |y|>4h:',a.height)
print(f(a.select('APT','FLIGHT_mvt','unm','SCHED_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','MVT_TIME_UTC_mvt','y').sort('APT')))
