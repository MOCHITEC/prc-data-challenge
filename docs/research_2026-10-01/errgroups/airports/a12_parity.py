import polars as pl
r=pl.read_parquet("regime_feats.parquet").filter(pl.col("apt")=="LTFM").with_columns(pl.from_epoch("t").dt.strftime("%Y-%m").alias("ym"))
c=["rg_arr_on_own_pm30","rg_arr_share_on_own","rg_n_arr_rwy_pm30","rg_own_share_pm30","rg_n_dep_rwy_pm30"]
print(r.group_by("ym").agg(*[pl.col(x).fill_nan(None).mean().round(3) for x in c],(pl.col("rg_arr_on_own_pm30")>0).mean().round(3).alias("p_mixed")).sort("ym"))
