import polars as pl
pl.Config.set_tbl_rows(50); pl.Config.set_tbl_cols(20); pl.Config.set_fmt_str_lengths(16); pl.Config.set_tbl_width_chars(300)
df=pl.read_parquet('all.parquet')
for stand,t0 in [('403','2025-04-24 22:12:02'),('805','2025-05-19 22:55:02')]:
  t0=pl.lit(t0).str.to_datetime(time_zone='UTC',time_unit='us')
  q=df.filter((pl.col('APT')=='LIRF')&(pl.col('STAND_mvt')==stand)&(pl.col('PHASE_mvt')=='DEP')&(pl.col('MVT_TIME_UTC_mvt')>t0)).sort('MVT_TIME_UTC_mvt').head(2)
  print(q.select('FLIGHT_mvt','ADES_mvt','FLIGHT_ID_mvt','SCHED_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','MVT_TIME_UTC_mvt','TAXITIME_SEC_mvt'))
  q=df.filter((pl.col('APT')=='LIRF')&(pl.col('FLIGHT_mvt').str.contains('IBE|WMT'))&(pl.col('PHASE_mvt')=='DEP')&(pl.col('MVT_TIME_UTC_mvt')>t0)&(pl.col('ADES_mvt').is_in(['LEMD','EGKK']))).sort('MVT_TIME_UTC_mvt').head(2)
  print(q.select('FLIGHT_mvt','ADES_mvt','FLIGHT_ID_mvt','STAND_mvt','SCHED_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','MVT_TIME_UTC_mvt','TAXITIME_SEC_mvt'))
