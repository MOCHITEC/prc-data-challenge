import polars as pl
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(30); pl.Config.set_fmt_str_lengths(30); pl.Config.set_tbl_width_chars(300)
df=pl.read_parquet('all.parquet')
APTS=['LIRF','LSZH','EDDM','LEBL','LEMD','EDDF','LFPG','EGLL','EHAM','LTFM']
d=pl.read_parquet('dep_dd.parquet').filter(pl.col('ADES_mvt').is_in(APTS))
a=df.filter((pl.col('PHASE_mvt')=='ARR')&pl.col('ADEP_mvt').is_in(APTS)).with_columns(
  ya=pl.col('TAXITIME_SEC_mvt'), amatched=pl.col('FLIGHT_ID_mvt').is_not_null())
del df
key=lambda c: pl.col(c).str.slice(0,3)+pl.col(c).str.slice(3).str.extract(r'(\d+)',1).cast(pl.Int64,strict=False).cast(pl.Utf8)
d=d.with_columns(fk=key('FLIGHT_mvt'))
a=a.with_columns(fk=key('FLIGHT_mvt'))
p=d.filter(pl.col('matched')).select('FLIGHT_ID_mvt','ADEP_mvt','ADES_mvt',dm=pl.col('MVT_TIME_UTC_mvt')).join(
  a.filter(pl.col('amatched')).select('FLIGHT_ID_mvt',am=pl.col('MVT_TIME_UTC_mvt')),on='FLIGHT_ID_mvt')
p=p.with_columns(ft=(pl.col('am')-pl.col('dm')).dt.total_seconds())
ft=p.group_by('ADEP_mvt','ADES_mvt').agg(ftmed=pl.col('ft').median(), ftsd=pl.col('ft').std())
print(ft.sort('ftsd',descending=True).head(5))
d=d.join(ft,on=['ADEP_mvt','ADES_mvt'],how='left').filter(pl.col('fk').is_not_null()&pl.col('ftmed').is_not_null())
d=d.with_columns(texp=(pl.col('MVT_TIME_UTC_mvt')+pl.duration(seconds=pl.col('ftmed'))).dt.cast_time_unit('us'))
A=a.filter(pl.col('fk').is_not_null()).select(pl.col('ADEP_mvt').alias('APT'),'ADES_mvt','fk',aid=pl.col('MVT_ID_mvt'),aflt=pl.col('FLIGHT_mvt'),amatched='amatched',am=pl.col('MVT_TIME_UTC_mvt'),ab=pl.col('BLOCK_TIME_UTC_mvt'),asch=pl.col('SCHED_TIME_UTC_mvt'),aobt=pl.col('AOBT_3_flt'),ya='ya',asrc=pl.col('src')).sort('am')
keep=['MVT_ID_mvt','src','APT','ADES_mvt','FLIGHT_mvt','fk','matched','dd_loc','y','ms','SCHED_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','MVT_TIME_UTC_mvt','AOBT_3_flt','ftmed','texp']
j=d.select(keep).sort('texp').join_asof(A,left_on='texp',right_on='am',by=['APT','ADES_mvt','fk'],strategy='nearest')
j=j.with_columns(lag=(pl.col('am')-pl.col('texp')).dt.total_seconds())
j.write_parquet('xapt_best.parquet')
print(j.group_by('src','matched').agg(n=pl.len(),linked30=(pl.col('lag').abs()<1800).sum(),linked2h=(pl.col('lag').abs()<7200).sum(),
   lagm1d=((pl.col('lag')+86400).abs()<7200).sum(), arrm=(pl.col('amatched')&(pl.col('lag').abs()<1800)).sum()).sort('src','matched'))
roll=pl.read_parquet('dep_roll.parquet').select('MVT_ID_mvt')
print(j.join(roll,on='MVT_ID_mvt').select('APT','ADES_mvt','FLIGHT_mvt','aflt','amatched','SCHED_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','MVT_TIME_UTC_mvt','am','ab','asch','aobt','ftmed','lag','y','ya'))
