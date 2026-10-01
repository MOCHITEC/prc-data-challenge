import polars as pl, numpy as np
P="/home/user/prc-data-challenge/data/work/feat_train.parquet"
cols=["MVT_ID_mvt","ADEP_mvt","RUNWAY_mvt","STAND_mvt","AIRCRAFT_OPERATOR_flt","WK_TBL_CAT_flt","MVT_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt","SCHED_TIME_UTC_mvt","AOBT_3_flt","EOBT_1_flt","LOBT_flt","IOBT_flt","ARVT_1_flt","ARVT_3_flt","TAXITIME_SEC_mvt","mvt_minus_aobt3"]
f=pl.scan_parquet(P).select(cols).filter(pl.col("mvt_minus_aobt3").is_not_null()&pl.col("TAXITIME_SEC_mvt").is_not_null()).collect()
o=pl.read_parquet("../oof_matched_janjul.parquet")
d=o.join(f.drop("ADEP_mvt","TAXITIME_SEC_mvt"),on="MVT_ID_mvt").filter("genuine")
d=d.with_columns(gap=(pl.col("BLOCK_TIME_UTC_mvt")-pl.col("AOBT_3_flt")).dt.total_seconds(),
  a_e=(pl.col("AOBT_3_flt")-pl.col("EOBT_1_flt")).dt.total_seconds(),
  a_l=(pl.col("AOBT_3_flt")-pl.col("LOBT_flt")).dt.total_seconds(),
  a_i=(pl.col("AOBT_3_flt")-pl.col("IOBT_flt")).dt.total_seconds(),
  a_s=(pl.col("AOBT_3_flt")-pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds(),
  l_i=(pl.col("LOBT_flt")-pl.col("IOBT_flt")).dt.total_seconds(),
  e_l=(pl.col("EOBT_1_flt")-pl.col("LOBT_flt")).dt.total_seconds(),
  m_a=pl.col("mvt_minus_aobt3"),
  a3sec=pl.col("AOBT_3_flt").dt.second(),
  arvt3_arvt1=(pl.col("ARVT_3_flt")-pl.col("ARVT_1_flt")).dt.total_seconds(),
  flt3=(pl.col("ARVT_3_flt")-pl.col("AOBT_3_flt")).dt.total_seconds(),
  flt1=(pl.col("ARVT_1_flt")-pl.col("EOBT_1_flt")).dt.total_seconds())
d=d.with_columns(grp=pl.when(pl.col("gap")< -600).then(pl.lit("early")).when(pl.col("gap")>600).then(pl.lit("late")).otherwise(pl.lit("mid")))
d.write_parquet("j.parquet")
sse=(d["err"]**2).sum()
print(d.group_by("grp").agg(n=pl.len(),sse=(pl.col("err")**2).sum()/sse,bias=pl.col("err").mean(),rmse=(pl.col("err")**2).mean().sqrt(),
  gap_med=pl.col("gap").median(),ma_med=pl.col("m_a").median(),y_med=pl.col("TAXITIME_SEC_mvt").median(),
  ae0=(pl.col("a_e").abs()<60).mean(),al0=(pl.col("a_l").abs()<60).mean(),ai0=(pl.col("a_i").abs()<60).mean(),as0=(pl.col("a_s").abs()<60).mean(),
  ae_med=pl.col("a_e").median(),sec0=(pl.col("a3sec")==0).mean(),li0=(pl.col("l_i").abs()<60).mean(),el_med=pl.col("e_l").median(),
  flt_d=(pl.col("flt3")-pl.col("flt1")).median()).sort("grp"))
print(d.group_by("ADEP_mvt").agg(n=pl.len(),early=(pl.col("grp")=="early").mean(),late=(pl.col("grp")=="late").mean(),
  sse_early=((pl.col("err")**2)*(pl.col("grp")=="early")).sum()/sse,sse_all=(pl.col("err")**2).sum()/sse).sort("sse_early",descending=True))
