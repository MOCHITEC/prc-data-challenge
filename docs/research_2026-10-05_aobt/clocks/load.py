import os
os.environ.setdefault("POLARS_MAX_THREADS","2"); os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR","load_as_storage")
import polars as pl
OUT="/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/aobt/clocks/slim.parquet"
def s(a,b): return ((pl.col(a)-pl.col(b)).dt.total_milliseconds()/1000).cast(pl.Float64)
if __name__=="__main__":
    lf=pl.scan_parquet("/home/user/prc-data-challenge/data/raw/training_2025-*.parquet")
    lf=lf.with_columns(pl.when(pl.col("PHASE_mvt")=="DEP").then(pl.col("ADEP_mvt")).otherwise(pl.col("ADES_mvt")).alias("APT"))
    lf=lf.select("APT","PHASE_mvt","RUNWAY_mvt","STAND_mvt","AIRCRAFT_TYPE_mvt","WK_TBL_CAT_flt","MARKET_SEGMENT_flt",
        pl.col("MVT_TIME_UTC_mvt").alias("MVT"),pl.col("BLOCK_TIME_UTC_mvt").alias("BLOCK"),pl.col("SCHED_TIME_UTC_mvt").alias("SCHED"),
        pl.col("AOBT_3_flt").alias("AOBT3"),pl.col("EOBT_1_flt").alias("EOBT1"),pl.col("ARVT_3_flt").alias("ARVT3"),pl.col("LOBT_flt").alias("LOBT"))
    lf=lf.with_columns(month=pl.col("MVT").dt.month(),hour=pl.col("MVT").dt.hour(),
        d_mt=s("MVT","AOBT3"),d_ba=s("BLOCK","AOBT3"),y=s("MVT","BLOCK"),d_bs=s("BLOCK","SCHED"),d_be=s("BLOCK","EOBT1"),
        d_ma=s("MVT","ARVT3"),
        sec_a=pl.col("AOBT3").dt.second(),sec_b=pl.col("BLOCK").dt.second(),sec_m=pl.col("MVT").dt.second(),sec_v=pl.col("ARVT3").dt.second())
    lf.sink_parquet(OUT)
    print(pl.read_parquet(OUT).group_by("APT","PHASE_mvt").len().sort("APT","PHASE_mvt"))
