import polars as pl
pl.Config.set_tbl_rows(250); pl.Config.set_tbl_cols(30); pl.Config.set_fmt_str_lengths(16); pl.Config.set_tbl_width_chars(320)
df=pl.read_parquet('all.parquet')
e=pl.read_parquet('lirf_elig.parquet')
a=df.filter((pl.col('PHASE_mvt')=='ARR')&(pl.col('APT')=='LIRF')&pl.col('STAND_mvt').is_not_null()).select(
  'STAND_mvt',ib=pl.col('BLOCK_TIME_UTC_mvt'),iflt=pl.col('FLIGHT_mvt'),iadep=pl.col('ADEP_mvt'),im=pl.col('MVT_TIME_UTC_mvt'),
  isch=pl.col('SCHED_TIME_UTC_mvt'),iy=pl.col('TAXITIME_SEC_mvt'),imatched=pl.col('FLIGHT_ID_mvt').is_not_null()).sort('ib')
# inbound = last ARR in-block at same stand before the DEP's MVT - 5 min  (stand occupant)
e=e.with_columns(tq=(pl.col('MVT_TIME_UTC_mvt')-pl.duration(minutes=5)).dt.cast_time_unit('us')).sort('tq')
j=e.join_asof(a,left_on='tq',right_on='ib',by='STAND_mvt',strategy='backward')
loc=lambda c: pl.col(c).dt.convert_time_zone('Europe/Rome').dt.date()
j=j.with_columns(gap_h=((pl.col('MVT_TIME_UTC_mvt')-pl.col('ib')).dt.total_seconds()/3600).round(1),
   ib_dd=(loc('ib')-loc('SCHED_TIME_UTC_mvt')).dt.total_days(),
   blk_dd=(loc('BLOCK_TIME_UTC_mvt')-loc('SCHED_TIME_UTC_mvt')).dt.total_days(),
   offb_proxy_dd=(pl.col('MVT_TIME_UTC_mvt').dt.offset_by('-20m').dt.convert_time_zone('Europe/Rome').dt.date()-loc('SCHED_TIME_UTC_mvt')).dt.total_days())
j.write_parquet('lirf_elig_rot.parquet')
t=j.filter(pl.col('src')=='train')
print(t.select('cls','FLIGHT_mvt','STAND_mvt','SCHED_TIME_UTC_mvt','MVT_TIME_UTC_mvt','y','ms_h','iflt','iadep','imatched','ib','gap_h','ib_dd','offb_proxy_dd').filter(pl.col('cls')=='ROLL'))
t2=t.with_columns(near=pl.col('gap_h')<12)
print(t2.group_by('cls','near','ib_dd').len().sort('cls','near','ib_dd'))
print('offb proxy x cls (train)'); print(t.group_by('offb_proxy_dd','cls').len().pivot(on='cls',index='offb_proxy_dd',values='len'))
# true block local date vs sched (train) for ok rows
print(t.group_by('cls','blk_dd').len().sort('cls','blk_dd'))
print('rank offb proxy', j.filter(pl.col('src')=='rank').group_by('offb_proxy_dd').len())
