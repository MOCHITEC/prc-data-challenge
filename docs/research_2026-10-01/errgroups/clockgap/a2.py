import polars as pl, numpy as np
pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_rows(60)
d=pl.read_parquet("j.parquet")
print(d.group_by("grp").agg(n=pl.len(),gap_med=pl.col("gap").median(),ma_med=pl.col("m_a").median(),y_med=pl.col("TAXITIME_SEC_mvt").median(),
  ae0=(pl.col("a_e").abs()<60).mean(),al0=(pl.col("a_l").abs()<60).mean(),ai0=(pl.col("a_i").abs()<60).mean(),as0=(pl.col("a_s").abs()<60).mean(),
  ae_med=pl.col("a_e").median(),as_med=pl.col("a_s").median(),sec0=(pl.col("a3sec")==0).mean(),li0=(pl.col("l_i").abs()<60).mean(),
  el0=(pl.col("e_l").abs()<60).mean(),flt_d=(pl.col("flt3")-pl.col("flt1")).median()).sort("grp"))
# quantiles of gap in early group, by airport
e=d.filter(pl.col("grp")=="early")
print(e.select(pl.col("gap").quantile(q).alias(str(q)) for q in [.05,.25,.5,.75,.95]))
# deciles of m_a relative to airport-rwy median
d=d.with_columns(ma_rel=pl.col("m_a")-pl.col("m_a").median().over(["ADEP_mvt","RUNWAY_mvt"]))
d=d.with_columns(bin=pl.col("ma_rel").cut([-600,-300,-120,0,120,300,600,1200]))
print(d.group_by("bin").agg(n=pl.len(),early=(pl.col("grp")=="early").mean(),late=(pl.col("grp")=="late").mean(),gap_mean=pl.col("gap").mean(),err_mean=pl.col("err").mean(),rmse=(pl.col("err")**2).mean().sqrt()).sort("bin"))
d=d.with_columns(dbin=pl.col("a_s").cut([-600,0,600,1800,3600,10800]))
print(d.group_by("dbin").agg(n=pl.len(),early=(pl.col("grp")=="early").mean(),late=(pl.col("grp")=="late").mean(),gap_mean=pl.col("gap").mean(),err_mean=pl.col("err").mean(),rmse=(pl.col("err")**2).mean().sqrt()).sort("dbin"))
d=d.with_columns(ebin=pl.col("a_e").cut([-600,0,300,900,1800,3600]))
print(d.group_by("ebin").agg(n=pl.len(),early=(pl.col("grp")=="early").mean(),late=(pl.col("grp")=="late").mean(),gap_mean=pl.col("gap").mean(),err_mean=pl.col("err").mean(),rmse=(pl.col("err")**2).mean().sqrt()).sort("ebin"))
d.write_parquet("j.parquet")
