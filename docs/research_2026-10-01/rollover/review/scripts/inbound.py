"""For LIRF unmatched DEP rows with ms>=12h (train 2025 and ranking 2026): was the stand's last ARR in-block
(before take-off) BEFORE or AFTER the scheduled departure? i.e. late-inbound vs aircraft-on-ground. Label-free on ranking."""
import glob, polars as pl
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(260); pl.Config.set_tbl_cols(20)
cols=["MVT_ID_mvt","FLIGHT_mvt","ADEP_mvt","ADES_mvt","PHASE_mvt","MVT_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt","SCHED_TIME_UTC_mvt","TAXITIME_SEC_mvt","AOBT_3_flt","STAND_mvt","RUNWAY_mvt"]
def run(src, label):
    lf=pl.scan_parquet(src).select(cols)
    dep=(lf.filter((pl.col("PHASE_mvt")=="DEP")&(pl.col("ADEP_mvt")=="LIRF")&pl.col("AOBT_3_flt").is_null())
         .with_columns(ms=(pl.col("MVT_TIME_UTC_mvt")-pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds())
         .filter(pl.col("ms")>=43200).collect())
    arr=(lf.filter((pl.col("PHASE_mvt")=="ARR")&(pl.col("ADES_mvt")=="LIRF")&pl.col("STAND_mvt").is_not_null())
         .select(pl.col("STAND_mvt"),pl.col("BLOCK_TIME_UTC_mvt").alias("inb"),pl.col("FLIGHT_mvt").alias("in_flt"),pl.col("SCHED_TIME_UTC_mvt").alias("in_sched")).collect().sort("inb"))
    deps_all=(lf.filter((pl.col("PHASE_mvt")=="DEP")&(pl.col("ADEP_mvt")=="LIRF")).select("STAND_mvt",pl.col("MVT_TIME_UTC_mvt").alias("odep")).collect().sort("odep"))
    d=dep.sort("MVT_TIME_UTC_mvt").join_asof(arr,left_on="MVT_TIME_UTC_mvt",right_on="inb",by="STAND_mvt",strategy="backward")
    d=d.with_columns(inb_minus_sched_h=((pl.col("inb")-pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds()/3600).round(2),
                     mvt_minus_inb_h=((pl.col("MVT_TIME_UTC_mvt")-pl.col("inb")).dt.total_seconds()/3600).round(2),
                     in_delay_h=((pl.col("inb")-pl.col("in_sched")).dt.total_seconds()/3600).round(2))
    if label:
        d=d.with_columns(y=pl.col("TAXITIME_SEC_mvt")).with_columns(cls=pl.when((pl.col("y")-pl.col("ms")).abs()<=60).then(pl.lit("fill")).when(pl.col("y")>72000).then(pl.lit("roll")).otherwise(pl.lit("other")))
    else:
        d=d.with_columns(cls=pl.lit("?"))
    print(d.select("FLIGHT_mvt","STAND_mvt","RUNWAY_mvt","SCHED_TIME_UTC_mvt","MVT_TIME_UTC_mvt","ms","cls","in_flt","inb_minus_sched_h","in_delay_h","mvt_minus_inb_h").sort("cls","ms"))
    return d
files=sorted(glob.glob("/home/user/prc-data-challenge/data/raw/training_*.parquet"))
print("== TRAIN 2025"); t=run(files,True)
print(t.with_columns(late_inb=pl.col("inb_minus_sched_h")>0).group_by("cls","late_inb").agg(n=pl.len()).sort("cls","late_inb"))
print("== RANKING 2026"); r=run("/home/user/prc-data-challenge/data/raw/ranking.parquet",False)
