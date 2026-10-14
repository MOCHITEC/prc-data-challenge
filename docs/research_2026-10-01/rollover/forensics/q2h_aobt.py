import polars as pl
OUT='/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/rollover/forensics/'
df=pl.read_parquet(OUT+'all.parquet').filter(pl.col('PHASE_mvt')=='DEP').with_columns(train=pl.col('file')!='ranking.parquet',
  ma=(pl.col('MVT_TIME_UTC_mvt')-pl.col('AOBT_3_flt')).dt.total_seconds(), y=pl.col('TAXITIME_SEC_mvt'))
t=df.filter(pl.col('ma')>43200)
print(t.group_by('train').agg(pl.len(), (pl.col('y')>72000).sum().alias('roll'), pl.col('ma').min().alias('mn'), pl.col('ma').max().alias('mx')))
print(t.filter(pl.col('train')).select('ADEP_mvt','FLIGHT_mvt','ma','y'))
print(t.filter(~pl.col('train')).select('ADEP_mvt','FLIGHT_mvt','MVT_TIME_UTC_mvt','AOBT_3_flt','SCHED_TIME_UTC_mvt','ma'))
