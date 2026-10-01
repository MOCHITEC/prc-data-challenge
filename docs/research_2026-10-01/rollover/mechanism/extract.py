import polars as pl, glob
cols=['MVT_ID_mvt','FLIGHT_ID_mvt','FLIGHT_mvt','ADEP_mvt','ADES_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','AIRCRAFT_TYPE_mvt','RUNWAY_mvt','STAND_mvt','TAXITIME_SEC_mvt','AOBT_3_flt','EOBT_1_flt','IOBT_flt','ARVT_3_flt','FLIGHT_RULE_mvt']
fs=sorted(glob.glob('/home/user/prc-data-challenge/data/raw/training_2025-*.parquet'))
df=pl.concat([pl.scan_parquet(f).select(cols) for f in fs]).collect()
df.write_parquet('train_ext.parquet')
print(df.shape, df.group_by('PHASE_mvt').len())
r=pl.scan_parquet('/home/user/prc-data-challenge/data/raw/ranking.parquet').select(cols).collect()
r.write_parquet('rank_ext.parquet'); print(r.shape, r.group_by('PHASE_mvt').len())
