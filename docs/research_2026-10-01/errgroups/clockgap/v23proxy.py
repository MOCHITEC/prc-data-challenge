"""Adds v23-equivalent columns missing from the shared OOF baseline: label encodings (fit on the 10 months other than
Jan/Jul 2025), stand_arrival_gap / stand_aobt_turn, proxy_minus_mode_min, categorical codes."""
import polars as pl, numpy as np, glob
FT="/home/user/prc-data-challenge/data/work/feat_train.parquet"
d=pl.read_parquet("resid_frame.parquet")
lab=pl.scan_parquet(FT).select("ADEP_mvt","STAND_mvt","RUNWAY_mvt","hour_local","AIRCRAFT_OPERATOR_flt","MVT_TIME_UTC_mvt","TAXITIME_SEC_mvt","mvt_minus_aobt3").filter(~pl.col("MVT_TIME_UTC_mvt").dt.month().is_in([1,7])).collect()
L="TAXITIME_SEC_mvt"
l2=lab.filter((pl.col(L)>0)&(pl.col(L)<7200))
ref=l2.group_by(["ADEP_mvt","STAND_mvt","RUNWAY_mvt"]).agg(ref_p10=pl.col(L).quantile(.1),ref_p20=pl.col(L).quantile(.2),ref_med=pl.col(L).median(),ref_n=pl.len(),
      gref_med=(pl.col(L)-pl.col("mvt_minus_aobt3")).median()).filter(pl.col("ref_n")>=30)
arh=lab.group_by(["ADEP_mvt","RUNWAY_mvt","hour_local"]).agg(arh_med=pl.col(L).median(),arh_n=pl.len()).filter(pl.col("arh_n")>=10)
apt=lab.group_by("ADEP_mvt").agg(apt_med=pl.col(L).median())
mode=(lab.filter(pl.col("mvt_minus_aobt3").is_not_null()).with_columns(_pm=pl.col("mvt_minus_aobt3")//60).group_by(["ADEP_mvt","RUNWAY_mvt","_pm"]).len()
      .sort("len",descending=True).group_by(["ADEP_mvt","RUNWAY_mvt"]).first().select("ADEP_mvt","RUNWAY_mvt",pl.col("_pm").alias("_mode")))
k=pl.scan_parquet(FT).select("MVT_ID_mvt","STAND_mvt","RUNWAY_mvt","AIRCRAFT_OPERATOR_flt","AIRCRAFT_TYPE_mvt","hour_local").collect()
d=d.join(k.drop("hour_local"),on="MVT_ID_mvt",how="left")
d=d.join(ref,on=["ADEP_mvt","STAND_mvt","RUNWAY_mvt"],how="left").join(arh,on=["ADEP_mvt","RUNWAY_mvt","hour_local"],how="left").join(apt,on="ADEP_mvt",how="left").join(mode,on=["ADEP_mvt","RUNWAY_mvt"],how="left")
d=d.with_columns(proxy_minus_mode_min=pl.col("mvt_minus_aobt3")//60-pl.col("_mode"),proxy_at_mode=(pl.col("mvt_minus_aobt3")//60==pl.col("_mode")).cast(pl.Int8))
# stand_arrival_gap from ARR rows of Jan/Jul raw
R="/home/user/prc-data-challenge/data/raw/"
arr=pl.concat([pl.scan_parquet(R+f).filter(pl.col("PHASE_mvt")=="ARR").select(pl.col("ADES_mvt").alias("ADEP_mvt"),"STAND_mvt",pl.col("BLOCK_TIME_UTC_mvt").alias("_ab"),pl.col("AIRCRAFT_TYPE_mvt").alias("_at"))
     for f in ["training_2025-01-01_2025-02-01.parquet","training_2025-07-01_2025-08-01.parquet","training_2024-12-01_2025-01-01.parquet"] if glob.glob(R+f)]).collect().drop_nulls().sort("_ab")
x=d.select("MVT_ID_mvt","ADEP_mvt","STAND_mvt","MVT_TIME_UTC_mvt","AOBT_3_flt","AIRCRAFT_TYPE_mvt").drop_nulls(["MVT_TIME_UTC_mvt","STAND_mvt"]).sort("MVT_TIME_UTC_mvt")
x=x.join_asof(arr,left_on="MVT_TIME_UTC_mvt",right_on="_ab",by=["ADEP_mvt","STAND_mvt"],strategy="backward",allow_exact_matches=False)
gap=(pl.col("MVT_TIME_UTC_mvt")-pl.col("_ab")).dt.total_seconds()
x=x.select("MVT_ID_mvt",stand_arrival_gap=pl.when(gap<=21600).then(gap),stand_aobt_turn=pl.when(gap<=21600).then((pl.col("AOBT_3_flt")-pl.col("_ab")).dt.total_seconds()),
           stand_arrival_type_match=pl.when(gap<=21600).then((pl.col("AIRCRAFT_TYPE_mvt")==pl.col("_at")).cast(pl.Int8)))
d=d.join(x,on="MVT_ID_mvt",how="left")
for c in ["STAND_mvt","RUNWAY_mvt","AIRCRAFT_OPERATOR_flt","AIRCRAFT_TYPE_mvt"]:
    d=d.with_columns(pl.col(c).cast(pl.Categorical).to_physical().cast(pl.Int32).alias("cat_"+c))
d.write_parquet("resid_frame2.parquet"); print(d.height, d.select(pl.col(["ref_med","stand_arrival_gap","proxy_minus_mode_min"]).null_count()))
