import numpy as np, polars as pl
E="/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/errgroups/"
tx=pl.read_parquet(E+"longtaxi/tx_feats.parquet")
day=tx.group_by("ADEP_mvt","lday","is_rank").agg([pl.col(c).first() for c in ["tx_day_mma_ratio","tx_day_tir","tx_day_tir_gt10","tx_day_late3h","tx_day_unm","tx_day_n_ratio"]]+[pl.len().alias("n")])
M=["tx_day_tir","tx_day_tir_gt10","tx_day_mma_ratio","tx_day_late3h","tx_day_unm"]
q=day.filter(~pl.col("is_rank")).group_by("ADEP_mvt").agg([pl.col(c).quantile(0.99).alias(c+"_p99") for c in M]+[pl.col(c).max().alias(c+"_max") for c in M])
d=day.join(q,on="ADEP_mvt").with_columns([(pl.col(c)>pl.col(c+"_p99")).cast(pl.Int8).alias(c+"_x") for c in M]+
   [(pl.col(c)>pl.col(c+"_max")).cast(pl.Int8).alias(c+"_beyond") for c in M])
d=d.with_columns(pl.sum_horizontal([c+"_x" for c in M]).alias("n_exceed"),pl.sum_horizontal([c+"_beyond" for c in M]).alias("n_beyond"))
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(14); pl.Config.set_fmt_str_lengths(12)
for r in (False,True):
    s=d.filter(pl.col("is_rank")==r)
    print("rank" if r else "2025", "days",s.height,"exceed>=1",(s["n_exceed"]>=1).sum(),"exceed>=2",(s["n_exceed"]>=2).sum(),"tir>p99",s["tx_day_tir_x"].sum(),"beyond2025max(any)",(s["n_beyond"]>=1).sum())
out=d.filter(pl.col("is_rank") & ((pl.col("n_exceed")>=2)|(pl.col("tx_day_tir_x")==1))).sort("ADEP_mvt","lday").select(
  "ADEP_mvt","lday","n",pl.col("tx_day_tir").round(0),pl.col("tx_day_tir_p99").round(0).alias("tir_p99_25"),pl.col("tx_day_tir_max").round(0).alias("tir_max_25"),
  pl.col("tx_day_tir_gt10").round(3).alias("ti>10m"),pl.col("tx_day_mma_ratio").round(2).alias("mma_r"),pl.col("tx_day_mma_ratio_p99").round(2).alias("mma_p99"),
  pl.col("tx_day_late3h").round(3).alias("late3h"),pl.col("tx_day_unm").round(3).alias("unm"),pl.col("tx_day_n_ratio").round(2).alias("n_r"),"n_exceed","n_beyond")
print(out)
out.write_csv(E+"longtaxi/flagged_days_2026.csv")
# 2025 analog: did those flags coincide with large oof error days?
o=pl.read_parquet(E+"oof_matched_janjul.parquet").join(tx.select("MVT_ID_mvt","lday"),on="MVT_ID_mvt").filter("genuine")
od=o.group_by("ADEP_mvt","lday").agg(pl.col("err").mean().alias("bias"),(pl.col("err")**2).mean().sqrt().alias("rmse"),pl.len().alias("rows"),((pl.col("TAXITIME_SEC_mvt")>=1800).mean()*100).alias("long%"))
j=od.join(d.filter(~pl.col("is_rank")),on=["ADEP_mvt","lday"])
print(j.group_by(pl.col("n_exceed").clip(0,2)).agg(pl.len(),pl.col("rows").sum(),pl.col("bias").mean().round(1),pl.col("rmse").mean().round(1),pl.col("long%").mean().round(1)).sort("n_exceed"))
print(j.group_by("tx_day_tir_x").agg(pl.len(),pl.col("bias").mean().round(1),pl.col("rmse").mean().round(1),pl.col("long%").mean().round(1)))
print(j.filter(pl.col("n_exceed")>=1).sort("rmse",descending=True).select("ADEP_mvt","lday","rows","bias","rmse","long%","tx_day_tir","tx_day_mma_ratio","tx_day_late3h","n_exceed").head(20))
