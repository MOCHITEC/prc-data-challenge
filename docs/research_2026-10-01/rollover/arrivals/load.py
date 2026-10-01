import polars as pl, glob
R='/home/user/prc-data-challenge/data/raw/'
cols=['MVT_ID_mvt','FLIGHT_ID_mvt','FLIGHT_mvt','ADEP_mvt','ADES_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','AIRCRAFT_TYPE_mvt','RUNWAY_mvt','STAND_mvt','TAXITIME_SEC_mvt','CALLSIGN_flt','AIRCRAFT_OPERATOR_flt','AOBT_3_flt','ARVT_3_flt','ARVT_1_flt','EOBT_1_flt','LOBT_flt']
fs=sorted(glob.glob(R+'training_2025*.parquet'))
tr=pl.concat([pl.scan_parquet(f).select(cols) for f in fs]).collect()
tr=tr.with_columns(pl.lit('train').alias('src'))
rk=pl.scan_parquet(R+'ranking.parquet').select(cols).collect().with_columns(pl.lit('rank').alias('src'))
df=pl.concat([tr,rk])
df=df.with_columns(pl.when(pl.col('PHASE_mvt')=='DEP').then(pl.col('ADEP_mvt')).otherwise(pl.col('ADES_mvt')).alias('APT'))
df.write_parquet('all.parquet')
print(df.group_by('src','PHASE_mvt').len())
print(df.filter(pl.col('PHASE_mvt')=='ARR').group_by('APT').len().sort('len'))
