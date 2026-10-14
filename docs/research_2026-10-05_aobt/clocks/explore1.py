import os; os.environ["POLARS_MAX_THREADS"]="2"
import polars as pl
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
d=pl.read_parquet("slim.parquet").filter(pl.col("PHASE_mvt")=="DEP")
q=lambda c,p: pl.col(c).quantile(p)
t=d.group_by("APT").agg(n=pl.len(),aobt_null=pl.col("AOBT3").is_null().mean(),
  ba_med=pl.col("d_ba").median(),ba_q25=q("d_ba",.25),ba_q75=q("d_ba",.75),ba_q05=q("d_ba",.05),ba_q95=q("d_ba",.95),
  ba_abs6=(pl.col("d_ba").abs()<=6).mean(),ba_0_60=((pl.col("d_ba")>=0)&(pl.col("d_ba")<60)).mean(),ba_m30_30=(pl.col("d_ba").abs()<30).mean(),ba_abs60=(pl.col("d_ba").abs()<=60).mean(),
  ba_abs180=(pl.col("d_ba").abs()<=180).mean(),
  mt_med=pl.col("d_mt").median(),y_med=pl.col("y").median()).sort("APT")
print(t)
# fine histogram of d_ba near zero, per airport, 10s bins in [-120,120]
h=d.filter(pl.col("d_ba").abs()<=120).with_columns(b=(pl.col("d_ba")/10).floor()*10).group_by("APT","b").len().pivot(on="APT",index="b",values="len").sort("b")
print(h)
