import polars as pl
pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_rows(40)
d=pl.read_parquet("resid_frame2.parquet").filter("genuine")
d=d.with_columns(grp=pl.when(pl.col("gap")< -600).then(pl.lit("early")).when(pl.col("gap")>600).then(pl.lit("late")).otherwise(pl.lit("mid")),
                 ae=(pl.col("AOBT_3_flt")-pl.col("EOBT_1_flt")).dt.total_seconds() if "EOBT_1_flt" in d.columns else pl.col("aobt3_minus_eobt1"))
sse=(d["err"]**2).sum()
for ap in ["LTFM","EGLL","LFPG","LIRF"]:
    x=d.filter(pl.col("ADEP_mvt")==ap)
    print(ap)
    print(x.group_by("proxy_at_mode").agg(n=pl.len(),early=(pl.col("grp")=="early").mean(),late=(pl.col("grp")=="late").mean(),sse=(pl.col("err")**2).sum()/sse,rmse=(pl.col("err")**2).mean().sqrt(),ma=pl.col("mvt_minus_aobt3").median()).sort("proxy_at_mode"))
x=d.filter(pl.col("ADEP_mvt")=="LTFM")
print(x.group_by("RUNWAY_mvt").agg(n=pl.len(),mode=pl.col("_mode").first(),early=(pl.col("grp")=="early").mean(),late=(pl.col("grp")=="late").mean(),rmse=(pl.col("err")**2).mean().sqrt(),sse=(pl.col("err")**2).sum()/sse).sort("n",descending=True))
x=x.with_columns(dl=pl.col("aobt3_minus_eobt1").cut([0,600,1800,3600]))
print(x.group_by("dl").agg(n=pl.len(),early=(pl.col("grp")=="early").mean(),late=(pl.col("grp")=="late").mean(),rmse=(pl.col("err")**2).mean().sqrt(),sse=(pl.col("err")**2).sum()/sse,gapmean=pl.col("gap").mean()).sort("dl"))
print(x.group_by("grp").agg(n=pl.len(),sse=(pl.col("err")**2).sum()/sse,bias=pl.col("err").mean(),y=pl.col("TAXITIME_SEC_mvt").median(),ma=pl.col("mvt_minus_aobt3").median(),gap=pl.col("gap").median()))
