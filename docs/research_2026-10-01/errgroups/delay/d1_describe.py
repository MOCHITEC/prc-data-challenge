import polars as pl, numpy as np
E="/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/errgroups/"
o=pl.read_parquet(E+"oof_matched_janjul.parquet")
cols=["MVT_ID_mvt","MVT_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt","SCHED_TIME_UTC_mvt","LOBT_flt","IOBT_flt","EOBT_1_flt","ARVT_1_flt","AOBT_3_flt","ARVT_3_flt",
 "mvt_minus_aobt3","aobt3_minus_sched","aobt3_minus_lobt","lobt_minus_eobt1","aobt3_minus_eobt1","FLIGHT_TYPE_flt","MARKET_SEGMENT_flt","RUNWAY_mvt","STAND_mvt"]
f=pl.scan_parquet("/home/user/prc-data-challenge/data/work/feat_train.parquet").select(cols).filter(pl.col("MVT_ID_mvt").is_in(o["MVT_ID_mvt"].implode())).collect()
d=o.join(f,on="MVT_ID_mvt").filter("genuine")
d=d.with_columns(g=(pl.col("AOBT_3_flt")-pl.col("BLOCK_TIME_UTC_mvt")).dt.total_seconds(),
  band=pl.col("aobt3_minus_sched").cut([-600,600,1800,3600,10800],labels=["<-10m","-10..10m","10..30m","30..60m","1..3h",">3h"]),
  aobt_eq_lobt=(pl.col("aobt3_minus_lobt")==0), aobt_min=(pl.col("AOBT_3_flt").dt.second()==0),
  rev=(pl.col("lobt_minus_eobt1")!=0), iobt_ne_lobt=(pl.col("IOBT_flt")!=pl.col("LOBT_flt")),
  blk_m_sched=(pl.col("BLOCK_TIME_UTC_mvt")-pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds())
d.write_parquet(E+"delay/joined.parquet")
tot=(d["err"]**2).sum()
def tab(by):
    t=d.group_by(by).agg(n=pl.len(),rmse=(pl.col("err")**2).mean().sqrt(),bias=pl.col("err").mean(),sd_err=pl.col("err").std(),
      sd_g=pl.col("g").std(),med_g=pl.col("g").median(),sse=(pl.col("err")**2).sum()/tot,
      eq_lobt=pl.col("aobt_eq_lobt").mean(),rev=pl.col("rev").mean(),iobt_ne=pl.col("iobt_ne_lobt").mean(),
      p_absg_gt600=(pl.col("g").abs()>600).mean()).sort(by)
    with pl.Config(tbl_rows=200,tbl_cols=30,tbl_width_chars=250): print(t)
print(d.height, np.sqrt((d["err"]**2).mean()))
tab("band")
tab(["band","aobt_eq_lobt"])
tab(["band","rev"])
# per airport for delayed band
d=d.with_columns(dl=pl.col("band").is_in(["30..60m","1..3h"]))
tab(["ADEP_mvt","dl"])
tab(["dl","FLIGHT_TYPE_flt"])
tab(["dl","MARKET_SEGMENT_flt"])
