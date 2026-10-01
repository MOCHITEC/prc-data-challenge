import polars as pl
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_cols(30); pl.Config.set_fmt_str_lengths(30); pl.Config.set_tbl_width_chars(300)
df=pl.read_parquet('all.parquet')
d=df.filter((pl.col('PHASE_mvt')=='DEP')&(pl.col('src')=='train')&(pl.col('TAXITIME_SEC_mvt')>72000))
print(d.select('APT','ADES_mvt','FLIGHT_mvt','CALLSIGN_flt','FLIGHT_ID_mvt','SCHED_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','MVT_TIME_UTC_mvt','AOBT_3_flt','ARVT_3_flt','STAND_mvt','RUNWAY_mvt').sort('MVT_TIME_UTC_mvt'))
d.write_parquet('dep_roll.parquet')
