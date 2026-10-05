import os; os.environ["POLARS_MAX_THREADS"]="2"
import polars as pl
pl.Config.set_tbl_rows(300); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(260)
D=pl.read_parquet("slim.parquet")
a=D.filter(pl.col("PHASE_mvt")=="ARR")
q=lambda c,p: pl.col(c).quantile(p)
t=a.group_by("APT").agg(n=pl.len(),arvt_null=pl.col("ARVT3").is_null().mean(),arvt_s0=(pl.col("sec_v")==0).mean(),
  med=pl.col("d_ma").median(),iqr=q("d_ma",.75)-q("d_ma",.25),q05=q("d_ma",.05),q95=q("d_ma",.95),
  w6=(pl.col("d_ma").abs()<=6).mean(),w30=(pl.col("d_ma").abs()<=30).mean(),w60=(pl.col("d_ma").abs()<=60).mean(),w180=(pl.col("d_ma").abs()<=180).mean(),
  taxin_med=pl.col("y").median()*-1).sort("APT")
# DEP comparison: BLOCK vs AOBT3
d=D.filter(pl.col("PHASE_mvt")=="DEP").group_by("APT").agg(dep_ba_med=pl.col("d_ba").median(),dep_ba_iqr=q("d_ba",.75)-q("d_ba",.25),dep_w60=(pl.col("d_ba").abs()<=60).mean(),
  dep_q05=q("d_ba",.05),dep_q95=q("d_ba",.95),dep_early600=(pl.col("d_ba")<-600).mean(),dep_late600=(pl.col("d_ba")>600).mean())
t=t.join(d,on="APT").sort("APT"); print(t); t.write_csv("out/arr_vs_dep_clock_agreement.csv")
# monthly arrival agreement
mo=a.group_by("APT","month").agg(w60=(pl.col("d_ma").abs()<=60).mean(),med=pl.col("d_ma").median()).sort("APT","month")
mo.write_csv("out/arr_monthly.csv"); print(mo.pivot(on="month",index="APT",values="w60"))
# fine histogram of d_ma
h=a.filter(pl.col("d_ma").abs()<=60).with_columns(b=(pl.col("d_ma")/5).floor()*5).group_by("APT","b").len().pivot(on="APT",index="b",values="len").sort("b")
print(h)
