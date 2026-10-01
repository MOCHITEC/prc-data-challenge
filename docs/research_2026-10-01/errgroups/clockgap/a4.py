import polars as pl, numpy as np
pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_rows(60)
d=pl.read_parquet("j.parquet").with_columns(early=(pl.col("grp")=="early"),mo=pl.col("MVT_TIME_UTC_mvt").dt.month(),day=pl.col("MVT_TIME_UTC_mvt").dt.date())
print(d.group_by("mo").agg(n=pl.len(),early=pl.col("early").mean()))
print(d.group_by(pl.col("l_i").abs()>60).agg(n=pl.len(),early=pl.col("early").mean(),late=(pl.col("grp")=="late").mean()))
print(d.group_by((pl.col("a_l")%300==0)).agg(n=pl.len(),early=pl.col("early").mean()))
# day x airport concentration
g=d.group_by(["ADEP_mvt","day"]).agg(n=pl.len(),k=pl.col("early").sum())
p=g["k"].sum()/g["n"].sum()
g=g.with_columns(r=pl.col("k")/pl.col("n"))
print("day-airport early rate var",float(((g["r"]-p)**2*g["n"]).sum()/g["n"].sum()),"binomial",float(p*(1-p)*(g.height/g["n"].sum())))
print(g.sort("k",descending=True).head(15))
# airport-hour cells: share of early group SSE in top cells
d=d.with_columns(hr=pl.col("MVT_TIME_UTC_mvt").dt.truncate("1h"))
# early fraction vs delay & m_a rel jointly
d=d.with_columns(ae_b=pl.col("a_e").cut([300,900,1800,3600]),mr_b=pl.col("ma_rel").cut([-300,0,300]))
print(d.group_by(["ae_b","mr_b"]).agg(n=pl.len(),early=pl.col("early").mean(),err=pl.col("err").mean(),rmse=(pl.col("err")**2).mean().sqrt()).sort(["ae_b","mr_b"]))
