"""Where does a feature set gain? SSE decomposition of two OOF files by clock source / tail / NM match (report §4)."""
import sys
import numpy as np
import polars as pl
from reporting_fingerprint_common import CACHE, load_all

a, b = sys.argv[1], sys.argv[2]
A = pl.read_parquet(CACHE / a).select("MVT_ID_mvt", "y", "src", "apt", pl.col("pred").alias("pa"), pl.col("p_sched").alias("psa"))
B = pl.read_parquet(CACHE / b).select("MVT_ID_mvt", pl.col("pred").alias("pb"), pl.col("p_sched").alias("psb"))
meta = load_all().filter(pl.col("PHASE_mvt") == "DEP").select("MVT_ID_mvt", pl.col("FLIGHT_ID_mvt").is_null().alias("nm_null"),
        (pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("m_s"))
j = A.join(B, on="MVT_ID_mvt").join(meta, on="MVT_ID_mvt")
j = j.with_columns(((pl.col("pa") - pl.col("y")) ** 2).alias("ea"), ((pl.col("pb") - pl.col("y")) ** 2).alias("eb"),
                   pl.when(pl.col("y") > 3 * 3600).then(pl.lit(">3h")).when(pl.col("y") > 3600).then(pl.lit("1-3h")).otherwise(pl.lit("<1h")).alias("ybin"))
N = j.height
for key in ["src", "ybin", "nm_null"]:
    print(j.group_by(key).agg(pl.len(), (pl.col("ea").sum() / N).alias("mse_contrib_a"), (pl.col("eb").sum() / N).alias("mse_contrib_b"),
          pl.col("psa").mean().round(3).alias("p_sched_a"), pl.col("psb").mean().round(3).alias("p_sched_b")).with_columns(
          (pl.col("mse_contrib_b") - pl.col("mse_contrib_a")).round(0).alias("d_mse")).sort("d_mse"))
print("total RMSE a,b:", np.sqrt(j["ea"].mean()), np.sqrt(j["eb"].mean()))
