import polars as pl
pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(20)
OUT='/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/rollover/forensics/'
df=pl.read_parquet(OUT+'all.parquet').filter(pl.col('PHASE_mvt')=='DEP', pl.col('ADEP_mvt')=='LIRF', pl.col('FLIGHT_mvt').is_in(['ITY340','AEZ146R']))
d=pl.datetime(2026,7,13,time_zone='UTC'); d2=pl.datetime(2026,7,21,time_zone='UTC')
print(df.filter(pl.col('FLIGHT_mvt')=='ITY340', pl.col('SCHED_TIME_UTC_mvt').is_between(d,d2)).select('FLIGHT_mvt','ADES_mvt','MVT_ID_mvt','FLIGHT_ID_mvt','SCHED_TIME_UTC_mvt','MVT_TIME_UTC_mvt','AOBT_3_flt','LOBT_flt','RUNWAY_mvt','STAND_mvt','AIRCRAFT_TYPE_mvt').sort('SCHED_TIME_UTC_mvt'))
print(df.filter(pl.col('FLIGHT_mvt')=='AEZ146R').select('FLIGHT_mvt','SCHED_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','MVT_TIME_UTC_mvt','AOBT_3_flt','LOBT_flt','TAXITIME_SEC_mvt').sort('SCHED_TIME_UTC_mvt').filter(pl.col('SCHED_TIME_UTC_mvt').dt.month()==7))
