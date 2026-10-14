import polars as pl
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
df=pl.read_parquet('all.parquet')
TZ={'LIRF':'Europe/Rome','LSZH':'Europe/Zurich','EDDM':'Europe/Berlin','LEBL':'Europe/Madrid','LEMD':'Europe/Madrid','EDDF':'Europe/Berlin','LFPG':'Europe/Paris','EGLL':'Europe/London','EHAM':'Europe/Amsterdam','LTFM':'Europe/Istanbul'}
d=df.filter(pl.col('PHASE_mvt')=='DEP').with_columns(
  y=pl.col('TAXITIME_SEC_mvt'), matched=pl.col('FLIGHT_ID_mvt').is_not_null(),
  ms=(pl.col('MVT_TIME_UTC_mvt')-pl.col('SCHED_TIME_UTC_mvt')).dt.total_seconds(),
  dd_utc=(pl.col('MVT_TIME_UTC_mvt').dt.date()-pl.col('SCHED_TIME_UTC_mvt').dt.date()).dt.total_days())
parts=[]
for a,tz in TZ.items():
  s=d.filter(pl.col('APT')==a)
  s=s.with_columns(dd_loc=(pl.col('MVT_TIME_UTC_mvt').dt.convert_time_zone(tz).dt.date()-pl.col('SCHED_TIME_UTC_mvt').dt.convert_time_zone(tz).dt.date()).dt.total_days())
  parts.append(s)
d=pl.concat(parts)
d.write_parquet('dep_dd.parquet')
t=d.filter(pl.col('src')=='train')
for key in ['dd_utc','dd_loc']:
  print(key)
  print(t.group_by('matched',key).agg(n=pl.len(), roll=(pl.col('y')>72000).sum(), fill=((pl.col('y')-pl.col('ms')).abs()<=60).sum(), big2h=(pl.col('y')>7200).sum()).filter(pl.col(key).abs()<=3).sort('matched',key))
  print(t.filter(~pl.col('matched')).group_by('APT',key).agg(n=pl.len(), roll=(pl.col('y')>72000).sum(), fill=((pl.col('y')-pl.col('ms')).abs()<=60).sum(),big2h=(pl.col('y')>7200).sum()).filter(pl.col(key)!=0).sort('APT',key))
r=d.filter(pl.col('src')=='rank')
print('RANK'); print(r.group_by('matched','dd_loc').len().sort('matched','dd_loc'))
print(r.filter(~pl.col('matched')&(pl.col('dd_loc')!=0)).group_by('APT','dd_loc').len().sort('APT','dd_loc'))
