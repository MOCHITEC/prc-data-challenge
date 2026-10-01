import os
os.environ["POLARS_MAX_THREADS"]="2"
os.environ["POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR"]="load_as_storage"
import polars as pl, glob
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(250)
R="/home/user/prc-data-challenge/data/raw/"
rank=pl.scan_parquet(R+"ranking.parquet")
tr=pl.scan_parquet(sorted(glob.glob(R+"training_*.parquet")))
FLT=["LOBT_flt","CALLSIGN_flt","ADEP_flt","ADES_flt","ADES_FILED_flt","MARKET_SEGMENT_flt","IOBT_flt","FLIGHT_RULE_flt","FLIGHT_TYPE_flt","AIRCRAFT_TYPE_flt","WK_TBL_CAT_flt","AIRCRAFT_OPERATOR_flt","EOBT_1_flt","ARVT_1_flt","AOBT_3_flt","ARVT_3_flt"]
MVT=["FLIGHT_ID_mvt","FLIGHT_mvt","FLIGHT_RULE_mvt","ADES_mvt","SCHED_TIME_UTC_mvt","AIRCRAFT_TYPE_mvt","RUNWAY_mvt","STAND_mvt"]

print("=== A. ranking months by phase, per-airport date span ===")
print(rank.group_by([pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m").alias("ym"),"PHASE_mvt"]).len().sort(["ym","PHASE_mvt"]).collect())
print(rank.filter(pl.col("PHASE_mvt")=="DEP").group_by("ADEP_mvt").agg([pl.col("MVT_TIME_UTC_mvt").min().alias("tmin"),pl.col("MVT_TIME_UTC_mvt").max().alias("tmax"),pl.len(),pl.col("FLIGHT_ID_mvt").is_null().mean().alias("unmatched_rate")]).sort("ADEP_mvt").collect())
print("training per-airport unmatched rate:")
print(tr.filter(pl.col("PHASE_mvt")=="DEP").group_by("ADEP_mvt").agg([pl.len(),pl.col("FLIGHT_ID_mvt").is_null().mean().alias("unmatched_rate")]).sort("ADEP_mvt").collect())
print("training unmatched rate by month:")
print(tr.filter(pl.col("PHASE_mvt")=="DEP").group_by(pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m").alias("ym")).agg([pl.len(),pl.col("FLIGHT_ID_mvt").is_null().mean().alias("unmatched_rate")]).sort("ym").collect())

print("=== B. null pattern of unmatched DEP rows: train vs rank ===")
for name,lf in [("train",tr),("rank",rank)]:
    u=lf.filter((pl.col("PHASE_mvt")=="DEP")&pl.col("FLIGHT_ID_mvt").is_null())
    print(name, u.select([pl.len().alias("n")]+[pl.col(c).is_null().mean().round(4).alias(c) for c in FLT+MVT]).collect().transpose(include_header=True))
print("=== B2. matched DEP rows: null rate of flt cols train vs rank ===")
for name,lf in [("train",tr),("rank",rank)]:
    u=lf.filter((pl.col("PHASE_mvt")=="DEP")&pl.col("FLIGHT_ID_mvt").is_not_null())
    print(name, u.select([pl.len().alias("n")]+[pl.col(c).is_null().mean().round(4).alias(c) for c in FLT+MVT]).collect().transpose(include_header=True))
