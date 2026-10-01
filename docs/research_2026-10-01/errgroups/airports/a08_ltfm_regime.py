# LTFM: check when the 3-independent-parallel regime begins; daily DEP runway count and y level per week (labels used only descriptively)
import polars as pl
f=pl.scan_parquet("/home/user/prc-data-challenge/data/work/feat_train.parquet").filter(pl.col("ADEP_mvt")=="LTFM").select("MVT_TIME_UTC_mvt","RUNWAY_mvt","TAXITIME_SEC_mvt","mvt_minus_aobt3").collect()
f=f.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.truncate("1w").alias("wk"))
w=f.group_by("wk").agg(pl.len(),pl.col("TAXITIME_SEC_mvt").filter(pl.col("TAXITIME_SEC_mvt")<7200).mean().round(0).alias("y"),
   pl.col("mvt_minus_aobt3").filter(pl.col("mvt_minus_aobt3").abs()<7200).mean().round(0).alias("mma"),
   ((pl.col("TAXITIME_SEC_mvt")-pl.col("mvt_minus_aobt3")).filter((pl.col("TAXITIME_SEC_mvt")-pl.col("mvt_minus_aobt3")).abs()<3600).mean().round(0)).alias("g"),
   (pl.col("RUNWAY_mvt")=="34L").mean().round(3).alias("34L"),(pl.col("RUNWAY_mvt")=="16R").mean().round(3).alias("16R"),
   (pl.col("RUNWAY_mvt").is_in(["35L","17R"])).mean().round(3).alias("35L17R")).sort("wk")
pl.Config.set_tbl_rows(60); print(w)
