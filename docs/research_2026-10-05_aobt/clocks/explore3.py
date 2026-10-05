import os; os.environ["POLARS_MAX_THREADS"]="2"
import polars as pl, numpy as np
pl.Config.set_tbl_rows(300); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
d=pl.read_parquet("slim.parquet").filter((pl.col("PHASE_mvt")=="DEP")&pl.col("d_mt").is_not_null()&(pl.col("d_mt").abs()<5400)&(pl.col("y").abs()<5400))
print(d.group_by("APT").agg(corr=pl.corr("d_mt","y"),sd_mt=pl.col("d_mt").std(),sd_y=pl.col("y").std(),sd_ba=(pl.col("y")-pl.col("d_mt")).std()).sort("APT"))
for apt,lo,hi in [("LTFM",1000,1600),("LEBL",500,1400),("EDDM",200,1000),("EHAM",450,750),("LEMD",800,1200)]:
    x=d.filter(pl.col("APT")==apt).with_columns(b=(pl.col("d_mt")/10).floor()*10).filter(pl.col("b").is_between(lo,hi))
    c=x.group_by("b").len().sort("b")
    print(apt," ".join(f"{int(b)}:{n}" for b,n in zip(c["b"],c["len"])))
