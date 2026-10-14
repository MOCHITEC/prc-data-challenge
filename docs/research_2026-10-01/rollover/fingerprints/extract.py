import polars as pl, glob
R='/home/user/prc-data-challenge/data/raw/'
cols=['MVT_ID_mvt','FLIGHT_ID_mvt','FLIGHT_mvt','FLIGHT_RULE_mvt','ADEP_mvt','ADES_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','AIRCRAFT_TYPE_mvt','RUNWAY_mvt','STAND_mvt','TAXITIME_SEC_mvt','CALLSIGN_flt','AIRCRAFT_OPERATOR_flt','AOBT_3_flt','EOBT_1_flt']
fs=sorted(glob.glob(R+'training_*.parquet'))+[R+'ranking.parquet']
out=[]
for f in fs:
    src='rank' if 'ranking' in f else 'train'
    d=(pl.scan_parquet(f).select(cols)
       .filter(((pl.col('PHASE_mvt')=='DEP')&(pl.col('ADEP_mvt')=='LIRF'))|((pl.col('PHASE_mvt')=='ARR')&(pl.col('ADES_mvt')=='LIRF')))
       .with_columns(pl.lit(src).alias('src')).collect())
    print(f.split('/')[-1], d.shape, d['PHASE_mvt'].value_counts().to_dicts())
    out.append(d)
pl.concat(out).write_parquet('lirf.parquet')
