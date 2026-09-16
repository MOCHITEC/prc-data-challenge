import os
os.environ["POLARS_MAX_THREADS"]="2"
os.environ["POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR"]="load_as_storage"
import polars as pl, glob
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(300)
R="/home/user/prc-data-challenge/data/raw/"
rank=pl.scan_parquet(R+"ranking.parquet")
tr=pl.scan_parquet(sorted(glob.glob(R+"training_*.parquet")))
sec=lambda a,b:(pl.col(a)-pl.col(b)).dt.total_seconds()
dep=tr.filter(pl.col("PHASE_mvt")=="DEP").with_columns([
  pl.col("FLIGHT_ID_mvt").is_null().alias("unm"),
  (sec("BLOCK_TIME_UTC_mvt","SCHED_TIME_UTC_mvt").abs()<=10).alias("blk_near_sched"),
  (pl.col("BLOCK_TIME_UTC_mvt").dt.second()==0).alias("blk_min_round"),
  (pl.col("AOBT_3_flt").dt.second()==0).alias("aobt_min_round"),
  sec("MVT_TIME_UTC_mvt","AOBT_3_flt").alias("proxy"),
  sec("BLOCK_TIME_UTC_mvt","AOBT_3_flt").alias("blk_minus_aobt"),
  sec("MVT_TIME_UTC_mvt","SCHED_TIME_UTC_mvt").alias("mvt_minus_sched"),
  sec("BLOCK_TIME_UTC_mvt","SCHED_TIME_UTC_mvt").alias("blk_minus_sched"),
  pl.col("TAXITIME_SEC_mvt").alias("y"),
]).with_columns(pl.when(pl.col("y")<0).then(pl.lit("neg")).when(pl.col("y")<30).then(pl.lit("0-30s")).when(pl.col("y")<7200).then(pl.lit("normal")).when(pl.col("y")<14400).then(pl.lit("2-4h")).when(pl.col("y")<72000).then(pl.lit("4-20h")).otherwise(pl.lit(">20h")).alias("bucket"))
print("=== C1c. buckets with |BLOCK-SCHED|<=10s tolerance ===")
print(dep.group_by("bucket").agg([pl.len(),pl.col("unm").mean().round(3).alias("unm"),pl.col("blk_near_sched").mean().round(3).alias("blk_near_sched"),(pl.col("blk_minus_sched").abs()<=120).mean().round(3).alias("blk_within2min_sched"),(pl.col("blk_minus_sched")< -3600).mean().round(3).alias("blk_gt1h_before_sched")]).sort("bucket").collect())
print("=== C1d. blk_near_sched rate by airport x matched (all rows) and its label distribution ===")
print(dep.group_by(["ADEP_mvt","unm"]).agg([pl.len(),pl.col("blk_near_sched").mean().round(4).alias("near_rate"),pl.col("y").filter(pl.col("blk_near_sched")).median().alias("y_med_near"),pl.col("y").filter(pl.col("blk_near_sched")).std().alias("y_sd_near"),(pl.col("y").filter(pl.col("blk_near_sched"))>=7200).mean().round(3).alias("gt2h_near"),pl.col("y").filter(~pl.col("blk_near_sched")).std().alias("y_sd_other")]).sort(["ADEP_mvt","unm"]).collect())
print("=== C1e. share of total sum(y^2) and of (y - median)^2 from blk_near_sched rows ===")
print(dep.select([(pl.col("y").cast(pl.Float64)-912)**2]).collect().sum())
print(dep.filter(pl.col("blk_near_sched")).select([(pl.col("y").cast(pl.Float64)-912)**2]).collect().sum())
print(dep.filter(pl.col("blk_near_sched")&pl.col("unm")).select([(pl.col("y").cast(pl.Float64)-912)**2]).collect().sum())
print("=== C3. negative labels ===")
print(dep.filter(pl.col("y")<0).select([pl.len(),pl.col("unm").mean().alias("unm"),pl.col("y").min().alias("ymin"),pl.col("y").median().alias("ymed"),pl.col("blk_near_sched").mean().alias("near"),pl.col("proxy").median().alias("proxy_med"),pl.col("ADEP_mvt").mode().first().alias("apt")]).collect())
print(dep.filter(pl.col("y")<0).group_by("ADEP_mvt").len().sort("len",descending=True).collect())
print("=== C4. unmatched vs matched label quantiles ===")
print(dep.group_by("unm").agg([pl.len(),pl.col("y").quantile(0.01).alias("p01"),pl.col("y").quantile(0.1).alias("p10"),pl.col("y").median().alias("med"),pl.col("y").quantile(0.9).alias("p90"),pl.col("y").quantile(0.99).alias("p99"),pl.col("y").std().alias("sd"),(pl.col("y")>=7200).mean().round(5).alias("gt2h"),(pl.col("y")>=14400).mean().round(5).alias("gt4h"),pl.col("blk_near_sched").mean().round(4).alias("near"),pl.col("blk_min_round").mean().round(3).alias("blk_round")]).collect())
print("=== C4b. matched rows: |BLOCK-AOBT_3| > 1h and the label there ===")
mm=dep.filter(~pl.col("unm"))
print(mm.select([pl.len(),(pl.col("blk_minus_aobt").abs()>3600).sum().alias("n_gap1h"),(pl.col("blk_minus_aobt").abs()>1800).sum().alias("n_gap30m"),(pl.col("blk_minus_aobt").abs()>600).mean().round(4).alias("rate_gap10m")]).collect())
print(mm.filter(pl.col("blk_minus_aobt").abs()>3600).group_by("ADEP_mvt").agg([pl.len(),pl.col("y").median().alias("y_med"),pl.col("proxy").median().alias("proxy_med"),pl.col("blk_near_sched").mean().round(2).alias("near")]).sort("len",descending=True).collect())
print("=== C5. matched: blk_minus_aobt & proxy per airport Jan vs Jul 2025 ===")
m=mm.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mo"))
print(m.filter(pl.col("mo").is_in([1,7])).group_by(["ADEP_mvt","mo"]).agg([pl.len(),pl.col("blk_minus_aobt").median().alias("blk_aobt_med"),pl.col("blk_minus_aobt").std().alias("blk_aobt_sd"),(pl.col("blk_minus_aobt").abs()<=60).mean().round(3).alias("within60"),pl.col("aobt_min_round").mean().round(3).alias("aobt_round"),pl.col("blk_min_round").mean().round(3).alias("blk_round"),pl.col("proxy").median().alias("proxy_med"),pl.col("proxy").std().alias("proxy_sd"),(pl.col("proxy")<0).mean().round(4).alias("proxy_neg"),(pl.col("proxy")>7200).mean().round(5).alias("proxy_gt2h"),pl.col("y").median().alias("y_med")]).sort(["ADEP_mvt","mo"]).collect())
print("=== C6. ranking matched: proxy per airport x month ===")
rk=rank.filter((pl.col("PHASE_mvt")=="DEP")&pl.col("AOBT_3_flt").is_not_null()).with_columns([sec("MVT_TIME_UTC_mvt","AOBT_3_flt").alias("proxy"),pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mo"),(pl.col("AOBT_3_flt").dt.second()==0).alias("aobt_min_round"),sec("AOBT_3_flt","EOBT_1_flt").alias("aobt_eobt"),sec("LOBT_flt","EOBT_1_flt").alias("lobt_eobt"),sec("MVT_TIME_UTC_mvt","SCHED_TIME_UTC_mvt").alias("mvt_sched")])
print(rk.group_by(["ADEP_mvt","mo"]).agg([pl.len(),pl.col("proxy").median().alias("proxy_med"),pl.col("proxy").quantile(0.9).alias("proxy_p90"),pl.col("proxy").std().alias("proxy_sd"),(pl.col("proxy")<0).mean().round(4).alias("proxy_neg"),(pl.col("proxy")>7200).mean().round(5).alias("proxy_gt2h"),pl.col("aobt_min_round").mean().round(3).alias("aobt_round"),pl.col("aobt_eobt").median().alias("aobt_eobt_med"),pl.col("lobt_eobt").median().alias("lobt_eobt_med"),pl.col("mvt_sched").median().alias("mvt_sched_med")]).sort(["ADEP_mvt","mo"]).collect())
print("--- same for training Jan/Jul 2025 ---")
print(m.filter(pl.col("mo").is_in([1,7])).with_columns([sec("AOBT_3_flt","EOBT_1_flt").alias("aobt_eobt"),sec("LOBT_flt","EOBT_1_flt").alias("lobt_eobt")]).group_by(["ADEP_mvt","mo"]).agg([pl.len(),pl.col("proxy").median().alias("proxy_med"),pl.col("proxy").quantile(0.9).alias("proxy_p90"),pl.col("proxy").std().alias("proxy_sd"),(pl.col("proxy")<0).mean().round(4).alias("proxy_neg"),(pl.col("proxy")>7200).mean().round(5).alias("proxy_gt2h"),pl.col("aobt_eobt").median().alias("aobt_eobt_med"),pl.col("lobt_eobt").median().alias("lobt_eobt_med"),pl.col("mvt_minus_sched").median().alias("mvt_sched_med"),pl.col("y").median().alias("y_med")]).sort(["ADEP_mvt","mo"]).collect())
print("=== C7. unmatched rows per airport x month: train (2025) vs rank (2026) ===")
ut=dep.filter(pl.col("unm")).group_by(["ADEP_mvt",pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mo")]).len().collect().pivot(on="mo",index="ADEP_mvt",values="len").sort("ADEP_mvt")
print(ut)
print(rank.filter((pl.col("PHASE_mvt")=="DEP")&pl.col("FLIGHT_ID_mvt").is_null()).group_by(["ADEP_mvt",pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mo")]).len().collect().pivot(on="mo",index="ADEP_mvt",values="len").sort("ADEP_mvt"))
print("=== D. duplicates ===")
for name,lf in [("train",dep),("rank",rank.filter(pl.col("PHASE_mvt")=="DEP"))]:
    d=lf.select(["MVT_ID_mvt","FLIGHT_ID_mvt","FLIGHT_mvt","ADEP_mvt","SCHED_TIME_UTC_mvt","MVT_TIME_UTC_mvt","RUNWAY_mvt","STAND_mvt"])
    print(name, d.select([pl.len().alias("n"),pl.col("MVT_ID_mvt").n_unique().alias("uniq_mvt_id"),pl.col("FLIGHT_ID_mvt").drop_nulls().len().alias("n_fid"),pl.col("FLIGHT_ID_mvt").drop_nulls().n_unique().alias("uniq_fid"),pl.struct(["ADEP_mvt","FLIGHT_mvt","SCHED_TIME_UTC_mvt"]).n_unique().alias("uniq_apt_flt_sched"),pl.struct(["ADEP_mvt","MVT_TIME_UTC_mvt","RUNWAY_mvt"]).n_unique().alias("uniq_apt_mvt_rwy"),pl.struct(["ADEP_mvt","STAND_mvt","MVT_TIME_UTC_mvt"]).n_unique().alias("uniq_apt_stand_mvt")]).collect())
    dupfid=d.filter(pl.col("FLIGHT_ID_mvt").is_not_null()).group_by("FLIGHT_ID_mvt").len().filter(pl.col("len")>1).collect()
    print(name,"FLIGHT_ID dup groups:",dupfid.height)
    if dupfid.height:
        print(d.filter(pl.col("FLIGHT_ID_mvt").is_in(dupfid["FLIGHT_ID_mvt"].head(4).implode())).sort("FLIGHT_ID_mvt").collect())
print("=== D2. same (ADEP, FLIGHT_mvt, SCHED) groups in training ===")
g=dep.filter(pl.col("FLIGHT_mvt").is_not_null()).group_by(["ADEP_mvt","FLIGHT_mvt","SCHED_TIME_UTC_mvt"]).agg([pl.len(),pl.col("y").min().alias("ymin"),pl.col("y").max().alias("ymax"),pl.col("unm").sum().alias("n_unm"),pl.col("MVT_TIME_UTC_mvt").min().alias("mvt_min"),pl.col("MVT_TIME_UTC_mvt").max().alias("mvt_max")]).filter(pl.col("len")>1).collect()
print("groups:",g.height,"rows:",g["len"].sum()); print(g.select([pl.col("n_unm").mean().alias("unm_per_group"),(pl.col("ymax")-pl.col("ymin")).median().alias("dy_med"),(pl.col("mvt_max")-pl.col("mvt_min")).dt.total_seconds().median().alias("dt_med")]))
print(g.sort("ymax",descending=True).head(8))
print("=== D3. ranking IDs overlap with training? ===")
tid=dep.select("MVT_ID_mvt").collect()["MVT_ID_mvt"]; rid=rank.select("MVT_ID_mvt").collect()["MVT_ID_mvt"]
print("MVT_ID overlap:",rid.is_in(tid).sum())
tf=tr.select("FLIGHT_ID_mvt").drop_nulls().collect()["FLIGHT_ID_mvt"]; rf=rank.select("FLIGHT_ID_mvt").drop_nulls().collect()["FLIGHT_ID_mvt"]
print("FLIGHT_ID overlap:",rf.is_in(tf).sum())
