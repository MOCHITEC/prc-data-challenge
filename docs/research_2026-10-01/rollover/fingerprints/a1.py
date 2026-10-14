import polars as pl
pl.Config.set_tbl_rows(400); pl.Config.set_tbl_cols(30); pl.Config.set_fmt_str_lengths(30); pl.Config.set_tbl_width_chars(250)
d=pl.read_parquet('lirf.parquet')
d=d.with_columns(ms=(pl.col('MVT_TIME_UTC_mvt')-pl.col('SCHED_TIME_UTC_mvt')).dt.total_seconds(),
                 y=pl.col('TAXITIME_SEC_mvt'))
dep=d.filter(pl.col('PHASE_mvt')=='DEP')
print('unmatched FLIGHT_ID null DEP:', dep.group_by('src').agg(pl.col('FLIGHT_ID_mvt').is_null().sum(), pl.len()))
u=dep.filter(pl.col('FLIGHT_ID_mvt').is_null())
u=u.with_columns(cls=pl.when(pl.col('y')>72000).then(pl.lit('roll')).when((pl.col('y')-pl.col('ms')).abs()<=60).then(pl.lit('fill')).otherwise(pl.lit('norm')))
# overall unmatched structure
print(u.filter(pl.col('src')=='train').group_by('cls').len())
print('null rates unmatched train:'); print(u.filter(pl.col('src')=='train').select([pl.col(c).is_null().mean().alias(c) for c in ['CALLSIGN_flt','AIRCRAFT_TYPE_mvt','STAND_mvt','RUNWAY_mvt','ADES_mvt','FLIGHT_RULE_mvt','SCHED_TIME_UTC_mvt','AOBT_3_flt']]))
# fill share by ms bucket
t=u.filter(pl.col('src')=='train').with_columns(mb=pl.col('ms').cut([-3600,0,1800,3600,7200,14400,28800,43200,72000]))
print(t.group_by('mb').agg(pl.len(),(pl.col('cls')=='roll').sum().alias('roll'),(pl.col('cls')=='fill').sum().alias('fill'),(pl.col('y').median()).alias('ymed')).sort('mb'))
print(u.filter(pl.col('src')=='rank').with_columns(mb=pl.col('ms').cut([-3600,0,1800,3600,7200,14400,28800,43200,72000])).group_by('mb').len().sort('mb'))
u.write_parquet('unm.parquet')
