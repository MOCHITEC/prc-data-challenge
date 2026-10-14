import polars as pl
pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(300); pl.Config.set_tbl_rows(80)
df=pl.read_parquet('dep2025.parquet').with_columns(
  y=pl.col("TAXITIME_SEC_mvt"), ya=(pl.col("MVT_TIME_UTC_mvt")-pl.col("AOBT_3_flt")).dt.total_seconds(),
  ym=(pl.col("TAXITIME_SEC_mvt")/60).round().cast(pl.Int32))
# share of most common taxi-minute value per airport (spike detector) vs neighbours
for a in ["LTFM","LIRF","EDDF","LSZH"]:
    s=df.filter(pl.col("ADEP_mvt")==a)
    vc=s["ym"].value_counts().sort("ym").filter(pl.col("ym").is_between(0,40))
    print(a, vc.with_columns(sh=pl.col("count")/s.height).select("ym","sh").transpose(include_header=False).to_numpy().round(3).tolist()[1][:40])
    vc2=s.with_columns(yam=(pl.col("ya")/60).round())["yam"].value_counts().sort("yam").filter(pl.col("yam").is_between(0,40))
    print(a,'AOBT3', vc2.with_columns(sh=pl.col("count")/s.height)["sh"].round(3).to_list())
# negatives
neg=df.filter(pl.col("y")<0)
print(neg.group_by("ADEP_mvt").agg(n=pl.len(), med=pl.col("y").median(), mn=pl.col("y").min(), ya_med=pl.col("ya").median()))
print(neg.filter(pl.col("ADEP_mvt")=="LSZH").select("FLIGHT_mvt","SCHED_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt","MVT_TIME_UTC_mvt","AOBT_3_flt","y","STAND_mvt","AIRCRAFT_TYPE_mvt").head(10))
