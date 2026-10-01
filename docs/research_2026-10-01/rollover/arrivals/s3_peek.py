import polars as pl
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_cols(30); pl.Config.set_fmt_str_lengths(30); pl.Config.set_tbl_width_chars(300)
df=pl.read_parquet('all.parquet')
c=['src','PHASE_mvt','APT','ADEP_mvt','ADES_mvt','FLIGHT_mvt','CALLSIGN_flt','FLIGHT_ID_mvt','SCHED_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','MVT_TIME_UTC_mvt','TAXITIME_SEC_mvt','AIRCRAFT_TYPE_mvt','STAND_mvt']
q=df.filter((pl.col('ADEP_mvt')=='LIRF')&(pl.col('ADES_mvt')=='EGLL')&(pl.col('MVT_TIME_UTC_mvt').dt.date().is_between(pl.date(2025,11,10),pl.date(2025,11,13)))&(pl.col('FLIGHT_mvt').str.contains('56|BAW')))
print(q.select(c).sort('MVT_TIME_UTC_mvt'))
q=df.filter((pl.col('ADEP_mvt')=='LIRF')&(pl.col('ADES_mvt')=='LFPG')&(pl.col('MVT_TIME_UTC_mvt').dt.date().is_between(pl.date(2025,6,24),pl.date(2025,6,27)))&(pl.col('FLIGHT_mvt').str.contains('33|ITY|AZA')))
print(q.select(c).sort('MVT_TIME_UTC_mvt'))
# LIRF ITYTY680 / unmatched odd strings examples, other phases
print(df.filter(pl.col('FLIGHT_mvt').str.contains('^ITYTY')).group_by('src','PHASE_mvt','APT').len())
print(df.filter(pl.col('FLIGHT_mvt').str.contains('ITY680|ITYTY680|AZA680')).select(c).sort('MVT_TIME_UTC_mvt').head(40))
