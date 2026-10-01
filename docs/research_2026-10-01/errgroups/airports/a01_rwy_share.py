import polars as pl, glob
R="/home/user/prc-data-challenge/data/raw/"
cols=["PHASE_mvt","ADEP_mvt","ADES_mvt","MVT_TIME_UTC_mvt","RUNWAY_mvt"]
d=pl.concat([pl.scan_parquet(f).select(cols) for f in sorted(glob.glob(R+"training_*.parquet"))+[R+"ranking.parquet"]]).collect()
d=d.with_columns(pl.when(pl.col("PHASE_mvt")=="DEP").then(pl.col("ADEP_mvt")).otherwise(pl.col("ADES_mvt")).alias("apt"),
                 pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m").alias("ym"))
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
for a in ["LIRF","LTFM","LFPG","EGLL"]:
    for ph in ["DEP","ARR"]:
        x=d.filter((pl.col("apt")==a)&(pl.col("PHASE_mvt")==ph)).group_by("ym","RUNWAY_mvt").len()
        x=x.with_columns((pl.col("len")/pl.col("len").sum().over("ym")*100).round(1).alias("p"))
        piv=x.pivot(on="RUNWAY_mvt",index="ym",values="p").sort("ym")
        print(a,ph); print(piv)
