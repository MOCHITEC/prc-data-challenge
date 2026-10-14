import polars as pl, numpy as np
pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_rows(60)
d=pl.read_parquet("j.parquet")
d=d.with_columns(b_e=(pl.col("BLOCK_TIME_UTC_mvt")-pl.col("EOBT_1_flt")).dt.total_seconds(),
 b_s=(pl.col("BLOCK_TIME_UTC_mvt")-pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds(),
 b_l=(pl.col("BLOCK_TIME_UTC_mvt")-pl.col("LOBT_flt")).dt.total_seconds())
e=d.filter(pl.col("grp")=="early")
for c in ["b_e","b_s","b_l","a_e","a_s","gap","TAXITIME_SEC_mvt","m_a"]:
    print(c, e.select([pl.col(c).quantile(q).alias(str(q)) for q in [.05,.25,.5,.75,.95]]).row(0))
# fraction of BLOCK within ±3 min of EOBT1 / SCHED/ LOBT in early vs mid
for g in ["early","mid","late"]:
    x=d.filter(pl.col("grp")==g)
    print(g, [(c, round(float((x[c].abs()<=180).mean()),3)) for c in ["b_e","b_s","b_l"]])
# relationship: for delayed flights (a_e>900): gap vs a_e
x=d.filter(pl.col("a_e")>900)
print("corr gap,a_e among delayed",np.corrcoef(x["gap"],x["a_e"])[0,1])
x=x.with_columns(r=(pl.col("b_e")/pl.col("a_e")))
print(x.filter(pl.col("grp")=="early").select(pl.col("r").quantile(q).alias(str(q)) for q in [.1,.25,.5,.75,.9]))
# per airport early group stats
print(e.group_by("ADEP_mvt").agg(n=pl.len(),gap=pl.col("gap").median(),a_e=pl.col("a_e").median(),b_e=pl.col("b_e").median(),y=pl.col("TAXITIME_SEC_mvt").median(),ma=pl.col("m_a").median(),bias=pl.col("err").mean()).sort("n",descending=True))
