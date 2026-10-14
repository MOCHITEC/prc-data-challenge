# build compact training + ranking tables
import polars as pl, glob
D='/home/user/prc-data-challenge/data/raw/'
O='/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/rollover/nonlirf/'
cols=['MVT_ID_mvt','FLIGHT_ID_mvt','FLIGHT_mvt','FLIGHT_RULE_mvt','ADEP_mvt','ADES_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','AIRCRAFT_TYPE_mvt','RUNWAY_mvt','STAND_mvt','TAXITIME_SEC_mvt','LOBT_flt','CALLSIGN_flt','MARKET_SEGMENT_flt','IOBT_flt','FLIGHT_TYPE_flt','AIRCRAFT_TYPE_flt','AIRCRAFT_OPERATOR_flt','EOBT_1_flt','ARVT_1_flt','AOBT_3_flt','ARVT_3_flt']
for name,pat in [('train',D+'training_2025-*.parquet'),('rank',D+'ranking.parquet')]:
    lf=pl.scan_parquet(pat)
    sch=lf.collect_schema()
    df=lf.select([c for c in cols if c in sch]).with_columns(
        pl.when(pl.col('PHASE_mvt')=='DEP').then(pl.col('ADEP_mvt')).otherwise(pl.col('ADES_mvt')).alias('APT'))
    df.collect().write_parquet(O+f'{name}.parquet')
    print(name, pl.scan_parquet(O+f'{name}.parquet').select(pl.len()).collect().item())
