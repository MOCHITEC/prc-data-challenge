import polars as pl, numpy as np
d=pl.read_parquet("joined.parquet").rename({"rev":"revised"})
tot=(d["err"]**2).sum()
d=d.with_columns(dl=pl.col("band").is_in(["30..60m","1..3h"]))
def tab(by,dd=d):
    t=dd.group_by(by).agg(n=pl.len(),rmse=(pl.col("err")**2).mean().sqrt(),bias=pl.col("err").mean(),
      sd_g=pl.col("g").std(),med_g=pl.col("g").median(),sse=(pl.col("err")**2).sum()/tot,
      p_g_pos600=(pl.col("g")>600).mean(),p_g_neg600=(pl.col("g")<-600).mean()).sort(by)
    with pl.Config(tbl_rows=200,tbl_cols=30,tbl_width_chars=250,float_precision=3): print(t)
tab(["band","revised"])
tab(["ADEP_mvt","dl"])
tab(["dl","FLIGHT_TYPE_flt"])
tab(["dl","MARKET_SEGMENT_flt"])
# where is BLOCK relative to clocks, for delayed rows with large |err|
s=lambda a,b:(pl.col(a)-pl.col(b)).dt.total_seconds()
d=d.with_columns(b_sched=s("BLOCK_TIME_UTC_mvt","SCHED_TIME_UTC_mvt"),b_eobt=s("BLOCK_TIME_UTC_mvt","EOBT_1_flt"),
  b_lobt=s("BLOCK_TIME_UTC_mvt","LOBT_flt"),b_iobt=s("BLOCK_TIME_UTC_mvt","IOBT_flt"),
  ebin=pl.col("err").cut([-600,-200,200,600],labels=["<-600","-600..-200","+-200","200..600",">600"]))
dl=d.filter("dl")
print(dl.group_by("ebin").agg(n=pl.len(),sse=(pl.col("err")**2).sum()/tot,med_g=pl.col("g").median(),
  med_b_sched=pl.col("b_sched").median(),med_aobt_sched=pl.col("aobt3_minus_sched").median(),
  near_sched5=(pl.col("b_sched").abs()<300).mean(),near_eobt5=(pl.col("b_eobt").abs()<300).mean(),
  near_lobt5=(pl.col("b_lobt").abs()<300).mean(),near_iobt5=(pl.col("b_iobt").abs()<300).mean(),
  block_min0=(pl.col("BLOCK_TIME_UTC_mvt").dt.second()==0).mean(), y_med=pl.col("TAXITIME_SEC_mvt").median(), pred_med=pl.col("pred").median()).sort("ebin"))
# fraction of delay recovered: BLOCK-SCHED vs AOBT3-SCHED
print(dl.select(pl.corr("b_sched","aobt3_minus_sched"), (pl.col("b_sched")/pl.col("aobt3_minus_sched")).median().alias("ratio_med")))
dl.write_parquet("delayed.parquet")
