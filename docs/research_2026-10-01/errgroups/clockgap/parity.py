import polars as pl, glob
pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
FT="/home/user/prc-data-challenge/data/work/feat_train.parquet"
lab=pl.scan_parquet(FT).select("ADEP_mvt","FLIGHT_mvt","AIRCRAFT_OPERATOR_flt","STAND_mvt","TAXITIME_SEC_mvt","mvt_minus_aobt3","MVT_TIME_UTC_mvt").filter(pl.col("mvt_minus_aobt3").is_not_null()).collect()
fl_all=lab.group_by(["ADEP_mvt","FLIGHT_mvt"]).len().filter(pl.col("len")>=10)
fl_oth=lab.filter(~pl.col("MVT_TIME_UTC_mvt").dt.month().is_in([1,7])).group_by(["ADEP_mvt","FLIGHT_mvt"]).len().filter(pl.col("len")>=10)
rk=pl.scan_parquet("/home/user/prc-data-challenge/data/raw/ranking.parquet").filter((pl.col("PHASE_mvt")=="DEP")&pl.col("AOBT_3_flt").is_not_null()).select("ADEP_mvt","FLIGHT_mvt","MVT_TIME_UTC_mvt").collect()
jj=lab.filter(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in([1,7]))
r=rk.join(fl_all,on=["ADEP_mvt","FLIGHT_mvt"],how="left")
t=jj.join(fl_oth,on=["ADEP_mvt","FLIGHT_mvt"],how="left")
print("fl enc coverage: train Jan/Jul (fit other months) %.3f ; ranking (fit all 2025) %.3f"%(t["len"].is_not_null().mean(), r["len"].is_not_null().mean()))
print(r.with_columns(mo=pl.col("MVT_TIME_UTC_mvt").dt.month()).group_by("mo").agg(cov=pl.col("len").is_not_null().mean()))
# candidate features null/mean parity
a=pl.read_parquet("cand_train.parquet"); b=pl.read_parquet("cand_rank.parquet")
cols=[c for c in a.columns if c!="MVT_ID_mvt"]
def summ(x): return x.select([pl.col(c).fill_nan(None).is_null().mean().alias(c) for c in cols])
print("null rate train/rank"); print(pl.concat([summ(a),summ(b)]))
print(pl.concat([a.select([pl.col(c).fill_nan(None).median().alias(c) for c in cols]),b.select([pl.col(c).fill_nan(None).median().alias(c) for c in cols])]))
