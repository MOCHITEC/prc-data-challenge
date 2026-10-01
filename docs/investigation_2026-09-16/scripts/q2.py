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
  (pl.col("BLOCK_TIME_UTC_mvt")==pl.col("SCHED_TIME_UTC_mvt")).alias("blk_eq_sched"),
  (pl.col("BLOCK_TIME_UTC_mvt").dt.second()==0).alias("blk_min_round"),
  (pl.col("AOBT_3_flt").dt.second()==0).alias("aobt_min_round"),
  (pl.col("BLOCK_TIME_UTC_mvt").dt.date()!=pl.col("MVT_TIME_UTC_mvt").dt.date()).alias("blk_other_day"),
  sec("MVT_TIME_UTC_mvt","AOBT_3_flt").alias("proxy"),
  sec("BLOCK_TIME_UTC_mvt","AOBT_3_flt").alias("blk_minus_aobt"),
  sec("MVT_TIME_UTC_mvt","SCHED_TIME_UTC_mvt").alias("mvt_minus_sched"),
  sec("BLOCK_TIME_UTC_mvt","SCHED_TIME_UTC_mvt").alias("blk_minus_sched"),
  pl.col("AOBT_3_flt").is_null().alias("aobt_null"),
  pl.col("TAXITIME_SEC_mvt").alias("y"),
])
print("=== C0. matched but AOBT_3 null; MVT null; label null ===")
print(dep.select([((~pl.col("unm"))&pl.col("aobt_null")).sum().alias("matched_aobt_null"),
                  pl.col("MVT_TIME_UTC_mvt").is_null().sum().alias("mvt_null"),
                  pl.col("y").is_null().sum().alias("y_null"),
                  pl.col("BLOCK_TIME_UTC_mvt").is_null().sum().alias("blk_null"),
                  ((pl.col("y")-sec("MVT_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt")).abs()>1).sum().alias("identity_viol")]).collect())
print(rank.filter(pl.col("PHASE_mvt")=="DEP").select([(pl.col("FLIGHT_ID_mvt").is_not_null()&pl.col("AOBT_3_flt").is_null()).sum().alias("rank_matched_aobt_null"),pl.col("MVT_TIME_UTC_mvt").is_null().sum().alias("rank_mvt_null")]).collect())

print("=== C1. label tail buckets: composition ===")
b=dep.with_columns(pl.when(pl.col("y")<0).then(pl.lit("neg")).when(pl.col("y")<30).then(pl.lit("0-30s")).when(pl.col("y")<7200).then(pl.lit("normal")).when(pl.col("y")<14400).then(pl.lit("2-4h")).when(pl.col("y")<72000).then(pl.lit("4-20h")).when(pl.col("y")<100000).then(pl.lit("20-28h")).otherwise(pl.lit(">28h")).alias("bucket"))
print(b.group_by("bucket").agg([pl.len(),pl.col("unm").mean().round(3).alias("unmatched_share"),pl.col("blk_eq_sched").mean().round(3).alias("blk_eq_sched"),pl.col("blk_min_round").mean().round(3).alias("blk_min_round"),pl.col("blk_other_day").mean().round(3).alias("blk_other_day"),(pl.col("y").pow(2).sum()/1e9).round(2).alias("sum_y2_e9"),pl.col("ADEP_mvt").mode().first().alias("top_apt")]).sort("bucket").collect())
print("=== C1b. y>=7200: per airport x matched ===")
print(dep.filter(pl.col("y")>=7200).group_by(["ADEP_mvt","unm"]).len().sort(["ADEP_mvt","unm"]).collect())
print("=== C2. top 40 labels ===")
print(dep.filter(pl.col("y")>=14400).sort("y",descending=True).select(["ADEP_mvt","FLIGHT_mvt","unm","SCHED_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt","AOBT_3_flt","MVT_TIME_UTC_mvt","y","proxy","blk_minus_sched","blk_eq_sched","RUNWAY_mvt","STAND_mvt","AIRCRAFT_TYPE_mvt"]).head(40).collect())
print("=== C2b. y>=14400: (y mod 86400) and (y-86400) distribution ===")
print(dep.filter(pl.col("y")>=14400).select([(pl.col("y")%86400).alias("mod")]).select([pl.col("mod").quantile(q).alias(f"q{q}") for q in (0.1,0.25,0.5,0.75,0.9)]).collect())
print("=== C3. negative labels ===")
print(dep.filter(pl.col("y")<0).select([pl.len(),pl.col("unm").mean().alias("unm"),pl.col("y").min(),pl.col("y").median(),pl.col("blk_eq_sched").mean().alias("blk_eq_sched"),pl.col("proxy").median().alias("proxy_med")]).collect())
print("=== C4. unmatched vs matched: label quantiles & blk_eq_sched ===")
print(dep.group_by("unm").agg([pl.len(),pl.col("y").quantile(0.01).alias("p01"),pl.col("y").quantile(0.1).alias("p10"),pl.col("y").median().alias("med"),pl.col("y").quantile(0.9).alias("p90"),pl.col("y").quantile(0.99).alias("p99"),pl.col("y").std().alias("sd"),(pl.col("y")>=7200).mean().round(5).alias("gt2h"),(pl.col("y")>=14400).mean().round(5).alias("gt4h"),pl.col("blk_eq_sched").mean().round(4).alias("blk_eq_sched"),pl.col("blk_min_round").mean().round(3).alias("blk_min_round")]).collect())
print("=== C4b. blk_eq_sched rows: label stats & mvt_minus_sched ===")
print(dep.group_by(["blk_eq_sched","unm"]).agg([pl.len(),pl.col("y").median(),pl.col("y").std().alias("sd"),(pl.col("y")>=7200).mean().round(4).alias("gt2h"),pl.col("proxy").median().alias("proxy_med"),(pl.col("y")-pl.col("proxy")).std().alias("sd_y_minus_proxy")]).sort(["blk_eq_sched","unm"]).collect())
print("=== C5. matched: blk_minus_aobt & y-proxy per airport, Jan vs Jul 2025 ===")
m=dep.filter(~pl.col("unm")).with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mo"))
print(m.filter(pl.col("mo").is_in([1,7])).group_by(["ADEP_mvt","mo"]).agg([pl.len(),pl.col("blk_minus_aobt").median().alias("blk_aobt_med"),pl.col("blk_minus_aobt").std().alias("blk_aobt_sd"),(pl.col("blk_minus_aobt").abs()<=60).mean().round(3).alias("within60"),pl.col("aobt_min_round").mean().round(3).alias("aobt_round"),pl.col("blk_min_round").mean().round(3).alias("blk_round"),pl.col("proxy").median().alias("proxy_med"),(pl.col("proxy")<0).mean().round(4).alias("proxy_neg"),(pl.col("proxy")>7200).mean().round(5).alias("proxy_gt2h")]).sort(["ADEP_mvt","mo"]).collect())
print("=== C6. ranking matched: proxy per airport x month ===")
rk=rank.filter((pl.col("PHASE_mvt")=="DEP")&pl.col("AOBT_3_flt").is_not_null()).with_columns([sec("MVT_TIME_UTC_mvt","AOBT_3_flt").alias("proxy"),pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mo"),(pl.col("AOBT_3_flt").dt.second()==0).alias("aobt_min_round"),sec("AOBT_3_flt","EOBT_1_flt").alias("aobt_eobt"),sec("LOBT_flt","EOBT_1_flt").alias("lobt_eobt")])
print(rk.group_by(["ADEP_mvt","mo"]).agg([pl.len(),pl.col("proxy").median().alias("proxy_med"),pl.col("proxy").quantile(0.9).alias("proxy_p90"),pl.col("proxy").std().alias("proxy_sd"),(pl.col("proxy")<0).mean().round(4).alias("proxy_neg"),(pl.col("proxy")>7200).mean().round(5).alias("proxy_gt2h"),pl.col("aobt_min_round").mean().round(3).alias("aobt_round"),pl.col("aobt_eobt").median().alias("aobt_eobt_med"),pl.col("lobt_eobt").median().alias("lobt_eobt_med")]).sort(["ADEP_mvt","mo"]).collect())
print("--- same for training Jan/Jul 2025 ---")
print(m.filter(pl.col("mo").is_in([1,7])).with_columns([sec("AOBT_3_flt","EOBT_1_flt").alias("aobt_eobt"),sec("LOBT_flt","EOBT_1_flt").alias("lobt_eobt")]).group_by(["ADEP_mvt","mo"]).agg([pl.len(),pl.col("proxy").median().alias("proxy_med"),pl.col("proxy").quantile(0.9).alias("proxy_p90"),pl.col("proxy").std().alias("proxy_sd"),pl.col("aobt_eobt").median().alias("aobt_eobt_med"),pl.col("lobt_eobt").median().alias("lobt_eobt_med"),pl.col("y").median().alias("y_med")]).sort(["ADEP_mvt","mo"]).collect())

print("=== D. duplicates ===")
for name,lf in [("train",dep),("rank",rank.filter(pl.col("PHASE_mvt")=="DEP"))]:
    d=lf.select(["MVT_ID_mvt","FLIGHT_ID_mvt","FLIGHT_mvt","ADEP_mvt","SCHED_TIME_UTC_mvt","MVT_TIME_UTC_mvt","RUNWAY_mvt","STAND_mvt","CALLSIGN_flt"])
    out=d.select([
        pl.len().alias("n"),
        pl.col("MVT_ID_mvt").n_unique().alias("uniq_mvt_id"),
        pl.col("FLIGHT_ID_mvt").drop_nulls().len().alias("n_fid"),
        pl.col("FLIGHT_ID_mvt").drop_nulls().n_unique().alias("uniq_fid"),
        pl.struct(["ADEP_mvt","FLIGHT_mvt","SCHED_TIME_UTC_mvt"]).n_unique().alias("uniq_apt_flt_sched"),
        pl.struct(["ADEP_mvt","MVT_TIME_UTC_mvt","RUNWAY_mvt"]).n_unique().alias("uniq_apt_mvt_rwy"),
        pl.struct(["ADEP_mvt","STAND_mvt","MVT_TIME_UTC_mvt"]).n_unique().alias("uniq_apt_stand_mvt"),
    ]).collect()
    print(name, out)
    dupfid=d.filter(pl.col("FLIGHT_ID_mvt").is_not_null()).group_by("FLIGHT_ID_mvt").len().filter(pl.col("len")>1).collect()
    print(name,"FLIGHT_ID dup groups:",dupfid.height)
    if dupfid.height:
        ex=d.filter(pl.col("FLIGHT_ID_mvt").is_in(dupfid["FLIGHT_ID_mvt"].head(5).implode())).sort("FLIGHT_ID_mvt").collect()
        print(ex)
print("=== D2. same (ADEP, FLIGHT_mvt, SCHED) dup groups in training: label spread ===")
g=dep.group_by(["ADEP_mvt","FLIGHT_mvt","SCHED_TIME_UTC_mvt"]).agg([pl.len(),pl.col("y").min().alias("ymin"),pl.col("y").max().alias("ymax"),pl.col("unm").sum().alias("n_unm"),pl.col("MVT_TIME_UTC_mvt").min().alias("mvt_min"),pl.col("MVT_TIME_UTC_mvt").max().alias("mvt_max")]).filter(pl.col("len")>1).collect()
print("groups:",g.height,"rows:",g["len"].sum()); print(g.select([pl.col("n_unm").mean(),(pl.col("ymax")-pl.col("ymin")).median().alias("dy_med"),(pl.col("mvt_max")-pl.col("mvt_min")).dt.total_seconds().median().alias("dt_med")]))
print(g.sort("ymax",descending=True).head(10))
print("=== D3. ranking IDs overlap with training? ===")
tid=dep.select("MVT_ID_mvt").collect()["MVT_ID_mvt"]
rid=rank.select("MVT_ID_mvt").collect()["MVT_ID_mvt"]
print("MVT_ID overlap:",rid.is_in(tid).sum())
tf=tr.select("FLIGHT_ID_mvt").drop_nulls().collect()["FLIGHT_ID_mvt"]
rf=rank.select("FLIGHT_ID_mvt").drop_nulls().collect()["FLIGHT_ID_mvt"]
print("FLIGHT_ID overlap:",rf.is_in(tf).sum())
