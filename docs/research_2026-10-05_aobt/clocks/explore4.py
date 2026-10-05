import os; os.environ["POLARS_MAX_THREADS"]="2"
import polars as pl, numpy as np
pl.Config.set_tbl_rows(300); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
d=pl.read_parquet("slim.parquet").filter((pl.col("PHASE_mvt")=="DEP")&pl.col("d_mt").is_not_null()&(pl.col("d_mt").abs()<5400))
d=d.with_columns(r=(pl.col("d_mt")%60))
for apt in ["EDDF","EHAM","LEBL","LEMD","LSZH"]:
    x=d.filter(pl.col("APT")==apt)
    h=np.histogram(x["r"].to_numpy(),bins=12,range=(0,60))[0]
    print(apt,"d_mt mod60 (5s bins) share:",np.round(h/h.sum(),3))
    x0=x.filter(pl.col("sec_a")==0); h=np.histogram(((x0["MVT"].dt.second())).to_numpy(),bins=12,range=(0,60))[0]
    print("    MVT sec dist (5s bins) where AOBT s0:",np.round(h/h.sum(),3))
    x1=x.filter(pl.col("sec_a")!=0)
    print("    AOBT3 non-zero-sec rows:",x1.height, "share |MVT-AOBT3 - k*60|<=1 :", ((x1["r"]<=1)|(x1["r"]>=59)).mean() if x1.height else None,
          " |BLOCK-AOBT3|<=1:",(x1["d_ba"].abs()<=1).mean() if x1.height else None)
