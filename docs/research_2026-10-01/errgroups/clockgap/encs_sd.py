import polars as pl
FT="/home/user/prc-data-challenge/data/work/feat_train.parquet"
lab=pl.scan_parquet(FT).select("MVT_ID_mvt","ADEP_mvt","FLIGHT_mvt","STAND_mvt","AIRCRAFT_OPERATOR_flt","MVT_TIME_UTC_mvt","TAXITIME_SEC_mvt","mvt_minus_aobt3").filter(pl.col("mvt_minus_aobt3").is_not_null()).collect()
lab=lab.with_columns(g=pl.col("TAXITIME_SEC_mvt")-pl.col("mvt_minus_aobt3"),mo=pl.col("MVT_TIME_UTC_mvt").dt.month()).filter(pl.col("TAXITIME_SEC_mvt")<72000)
lab=lab.with_columns(ear=(pl.col("g")>600).cast(pl.Float64))   # g = AOBT3-BLOCK > 600 <=> BLOCK earlier by >10 min
tr=lab.filter(pl.col("mo")>=9)
fl=tr.group_by(["ADEP_mvt","FLIGHT_mvt"]).agg(fl_gmed=pl.col("g").median(),fl_early=pl.col("ear").mean(),fl_n=pl.len()).filter(pl.col("fl_n")>=10)
op=tr.group_by(["ADEP_mvt","AIRCRAFT_OPERATOR_flt"]).agg(op_early=pl.col("ear").mean(),op_gmed=pl.col("g").median(),op_n=pl.len()).filter(pl.col("op_n")>=20)
st=tr.group_by(["ADEP_mvt","STAND_mvt"]).agg(st_early=pl.col("ear").mean(),st_n=pl.len()).filter(pl.col("st_n")>=30)
d=pl.read_parquet("resid_frame2.parquet")
k=pl.scan_parquet(FT).select("MVT_ID_mvt","FLIGHT_mvt").collect()
d=d.join(k,on="MVT_ID_mvt",how="left").join(fl,on=["ADEP_mvt","FLIGHT_mvt"],how="left").join(op,on=["ADEP_mvt","AIRCRAFT_OPERATOR_flt"],how="left").join(st,on=["ADEP_mvt","STAND_mvt"],how="left")
d.write_parquet("resid_frame4.parquet")
print(d.select(pl.col(["fl_gmed","op_early","st_early"]).is_null().mean()))
