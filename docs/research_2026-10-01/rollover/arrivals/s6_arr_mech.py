# Does the ARR reporting share the DEP mechanism (BLOCK local date := SCHED local date)?
import polars as pl
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
df=pl.read_parquet('all.parquet')
TZ={'LIRF':'Europe/Rome','LSZH':'Europe/Zurich','EDDM':'Europe/Berlin','LEBL':'Europe/Madrid','LEMD':'Europe/Madrid','EDDF':'Europe/Berlin','LFPG':'Europe/Paris','EGLL':'Europe/London','EHAM':'Europe/Amsterdam','LTFM':'Europe/Istanbul'}
a=df.filter(pl.col('PHASE_mvt')=='ARR')
parts=[]
for ap,tz in TZ.items():
  s=a.filter(pl.col('APT')==ap).with_columns(
    dB=(pl.col('BLOCK_TIME_UTC_mvt').dt.convert_time_zone(tz).dt.date()-pl.col('SCHED_TIME_UTC_mvt').dt.convert_time_zone(tz).dt.date()).dt.total_days(),
    dM=(pl.col('MVT_TIME_UTC_mvt').dt.convert_time_zone(tz).dt.date()-pl.col('SCHED_TIME_UTC_mvt').dt.convert_time_zone(tz).dt.date()).dt.total_days())
  parts.append(s)
a=pl.concat(parts).with_columns(y=pl.col('TAXITIME_SEC_mvt'),matched=pl.col('FLIGHT_ID_mvt').is_not_null())
a.select('MVT_ID_mvt','src','APT','matched','dB','dM','y').write_parquet('arr_dd.parquet')
# opportunities: true in-block on a different local day than SCHED. Proxy: landing local date = SCHED date+1 (late arrivals past midnight)
print(a.group_by('src','APT','matched').agg(n=pl.len(), opp=(pl.col('dM')==1).sum(), shifted_neg=(pl.col('y')< -43200).sum(), shifted_pos=(pl.col('y')>43200).sum()).sort('src','APT','matched'))
