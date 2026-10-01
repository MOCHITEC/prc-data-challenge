"""Test the 'date reconstructed nearest to SCHED' mechanism: distribution of recorded BLOCK-SCHED (bs) per airport,
esp. LIRF non-fill rows. Under a +-12h nearest-date rule, no LIRF row can have recorded bs outside a 24h window."""
import glob, polars as pl
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(20)
files=sorted(glob.glob("/home/user/prc-data-challenge/data/raw/training_*.parquet"))
d=(pl.scan_parquet(files).filter(pl.col("PHASE_mvt")=="DEP")
   .select("ADEP_mvt","FLIGHT_mvt","SCHED_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt","MVT_TIME_UTC_mvt","TAXITIME_SEC_mvt","AOBT_3_flt","LOBT_flt","RUNWAY_mvt","STAND_mvt","ADES_mvt")
   .with_columns(bs=(pl.col("BLOCK_TIME_UTC_mvt")-pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds(),
                 ms=(pl.col("MVT_TIME_UTC_mvt")-pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds(),
                 y=pl.col("TAXITIME_SEC_mvt"), unm=pl.col("AOBT_3_flt").is_null()).collect())
d=d.with_columns(fill=pl.col("bs").abs()<=10, roll=pl.col("y")>72000)
print("== per airport: recorded bs quantiles & tails (non-fill rows)")
print(d.filter(~pl.col("fill")).group_by("ADEP_mvt").agg(n=pl.len(),bs_max=pl.col("bs").max(),n_gt_10h=(pl.col("bs")>36000).sum(),n_gt_12h=(pl.col("bs")>43200).sum(),n_gt_13_7h=(pl.col("bs")>49438).sum(),n_lt_m10h=(pl.col("bs")<-36000).sum(),bs_min=pl.col("bs").min()).sort("ADEP_mvt"))
L=d.filter((pl.col("ADEP_mvt")=="LIRF")&~pl.col("fill"))
print("== LIRF non-fill rows with recorded bs > 6h (top 25), not rolled")
print(L.filter(~pl.col("roll")).sort("bs",descending=True).head(25).select("FLIGHT_mvt","SCHED_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt","MVT_TIME_UTC_mvt","y","bs","ms","unm"))
print("== LIRF non-fill non-roll rows by true delay bucket (h) ; plus rolled rows by implied true bs (bs+86400)")
L=L.with_columns(tbs=pl.when(pl.col("roll")).then(pl.col("bs")+86400).otherwise(pl.col("bs")))
print(L.filter(pl.col("tbs")>=6*3600).with_columns(h=(pl.col("tbs")/3600).floor()).group_by("h").agg(n=pl.len(),roll=pl.col("roll").sum(),unm=pl.col("unm").sum()).sort("h"))
print("== other airports: non-fill rows with true bs >= 12h, counts (should be non-zero if delays>12h exist and are recorded correctly)")
print(d.filter(~pl.col("fill")&(pl.col("ADEP_mvt")!="LIRF")&(pl.col("bs")>=43200)).group_by("ADEP_mvt").agg(n=pl.len(),unm=pl.col("unm").sum(),roll=pl.col("roll").sum()).sort("ADEP_mvt"))
print("== LIRF fill rows ms distribution vs non-fill (count by ms hour >= 6h)")
F=d.filter(pl.col("ADEP_mvt")=="LIRF").with_columns(h=(pl.col("ms")/3600).floor())
print(F.filter(pl.col("h")>=6).group_by("h").agg(n=pl.len(),fill=pl.col("fill").sum(),roll=pl.col("roll").sum(),unm=pl.col("unm").sum(),other=(~pl.col("fill")&~pl.col("roll")).sum()).sort("h"))
