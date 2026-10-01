import polars as pl
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_cols(30); pl.Config.set_fmt_str_lengths(30); pl.Config.set_tbl_width_chars(250)
df=pl.read_parquet('all.parquet')
a=df.filter(pl.col('PHASE_mvt')=='ARR').with_columns(
  y=pl.col('TAXITIME_SEC_mvt'),
  ms=(pl.col('MVT_TIME_UTC_mvt')-pl.col('SCHED_TIME_UTC_mvt')).dt.total_seconds(),
  bs=(pl.col('BLOCK_TIME_UTC_mvt')-pl.col('SCHED_TIME_UTC_mvt')).dt.total_seconds(),
  ma=(pl.col('MVT_TIME_UTC_mvt')-pl.col('ARVT_3_flt')).dt.total_seconds(),
  matched=pl.col('FLIGHT_ID_mvt').is_not_null(),
  mon=pl.col('MVT_TIME_UTC_mvt').dt.strftime('%Y-%m'))
print('null y', a.group_by('src').agg(pl.col('y').is_null().sum(), pl.len()))
# bands
bands=[(-10**9,-72000),(-72000,-43200),(-43200,-7200),(-7200,-600),(-600,0),(0,3600),(3600,7200),(7200,43200),(43200,72000),(72000,100800),(100800,10**9)]
for src in ['train','rank']:
  s=a.filter(pl.col('src')==src)
  print(src, [(lo,hi,s.filter((pl.col('y')>=lo)&(pl.col('y')<hi)).height) for lo,hi in bands])
big=a.filter(pl.col('y').abs()>=20000).sort('y')
print(big.select('src','APT','ADEP_mvt','FLIGHT_mvt','matched','SCHED_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','MVT_TIME_UTC_mvt','y','ms','bs','ma','AIRCRAFT_TYPE_mvt','STAND_mvt'))
big.write_parquet('arr_big.parquet')
