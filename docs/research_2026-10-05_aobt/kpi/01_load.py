# Load 2025 departures (selected columns) into a local working parquet (kept local, not redistributed).
import polars as pl, glob
cols=["MVT_ID_mvt","FLIGHT_ID_mvt","FLIGHT_mvt","FLIGHT_RULE_mvt","ADEP_mvt","PHASE_mvt","MVT_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt",
"SCHED_TIME_UTC_mvt","AIRCRAFT_TYPE_mvt","RUNWAY_mvt","STAND_mvt","TAXITIME_SEC_mvt","LOBT_flt","IOBT_flt","EOBT_1_flt","AOBT_3_flt",
"WK_TBL_CAT_flt","AIRCRAFT_TYPE_flt","MARKET_SEGMENT_flt","FLIGHT_TYPE_flt"]
fs=sorted(glob.glob('/home/user/prc-data-challenge/data/raw/training_2025-*.parquet'))
df=pl.concat([pl.scan_parquet(f).select(cols).filter(pl.col("PHASE_mvt")=="DEP") for f in fs]).collect()
df.write_parquet('dep2025.parquet')
print(df.shape)
print(df.group_by("ADEP_mvt").agg(n=pl.len(), stand_null=pl.col("STAND_mvt").is_null().mean(), rwy_null=pl.col("RUNWAY_mvt").is_null().mean(),
  blk_null=pl.col("BLOCK_TIME_UTC_mvt").is_null().mean(), act_null=pl.col("AIRCRAFT_TYPE_mvt").is_null().mean(), aobt3_null=pl.col("AOBT_3_flt").is_null().mean(),
  nstand=pl.col("STAND_mvt").n_unique(), nrwy=pl.col("RUNWAY_mvt").n_unique(), neg=(pl.col("TAXITIME_SEC_mvt")<0).sum(), gt2h=(pl.col("TAXITIME_SEC_mvt")>7200).sum(),
  rules=pl.col("FLIGHT_RULE_mvt").unique().str.join(","), mintime=pl.col("MVT_TIME_UTC_mvt").min(), maxtime=pl.col("MVT_TIME_UTC_mvt").max()).sort("ADEP_mvt"))
