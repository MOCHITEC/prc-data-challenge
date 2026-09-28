import os
os.environ["POLARS_MAX_THREADS"]="2"
os.environ["POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR"]="load_as_storage"
import polars as pl, glob, json
pl.Config.set_tbl_rows(120); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(300)
B="/home/user/prc-data-challenge/"
R=B+"data/raw/"
rank=pl.scan_parquet(R+"ranking.parquet")
tr=pl.scan_parquet(sorted(glob.glob(R+"training_*.parquet")))
ft=pl.scan_parquet(B+"data/work/feat_train.parquet"); fr=pl.scan_parquet(B+"data/work/feat_rank.parquet")
sch=ft.collect_schema()
print("=== F2. feature distribution shift: medians train vs rank for key numeric features ===")
keys=["mvt_minus_aobt3","mvt_minus_sched","mvt_minus_eobt1","mvt_minus_lobt","aobt3_minus_eobt1","aobt3_minus_sched","lobt_minus_eobt1","dep_tko_prev_30m","arr_land_prev_30m","arr_inblock_prev_30m","rwy_headway_prev_s","nbr_mvt_minus_aobt_mean_30m","nbr_aobt_delay_mean_30m","arr_taxi_in_mean_30m","arr_taxi_in_n_30m","window_coverage_60m","temp_c","vis_km","wind_ms","precip_mm","deicing_gate","low_vis","frozen_wx"]
keys=[k for k in keys if k in sch]
def stats(lf,tag):
    return lf.filter(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in([1,7])).group_by(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mo")).agg([pl.col(k).cast(pl.Float64).median().alias(k) for k in keys]).sort("mo").with_columns(pl.lit(tag).alias("src"))
print(pl.concat([stats(ft,"train"),stats(fr,"rank")]).sort(["mo","src"]).collect().transpose(include_header=True,header_name="feature"))
print("=== F3. arr_taxi_in (arrival labels visible in ranking) per airport: train Jan/Jul vs rank Jan/Jul ===")
arr=lambda lf,tag: lf.filter(pl.col("PHASE_mvt")=="ARR").with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mo")).filter(pl.col("mo").is_in([1,7])).group_by(["ADES_mvt","mo"]).agg([pl.len().alias("n_arr"),pl.col("TAXITIME_SEC_mvt").median().alias("taxi_in_med"),pl.col("TAXITIME_SEC_mvt").is_null().mean().round(3).alias("taxi_in_null"),pl.col("BLOCK_TIME_UTC_mvt").is_null().mean().round(3).alias("blk_null"),pl.col("FLIGHT_ID_mvt").is_null().mean().round(4).alias("unm")]).with_columns(pl.lit(tag).alias("src"))
print(pl.concat([arr(tr,"train"),arr(rank,"rank")]).collect().sort(["ADES_mvt","mo","src"]))
print("=== G. unmatched rows: category coverage & mix, train vs rank ===")
ud_tr=tr.filter((pl.col("PHASE_mvt")=="DEP")&pl.col("FLIGHT_ID_mvt").is_null()).collect()
ud_rk=rank.filter((pl.col("PHASE_mvt")=="DEP")&pl.col("FLIGHT_ID_mvt").is_null()).collect()
md_tr=tr.filter((pl.col("PHASE_mvt")=="DEP")&pl.col("FLIGHT_ID_mvt").is_not_null()).select(["FLIGHT_mvt","STAND_mvt","AIRCRAFT_TYPE_mvt","ADES_mvt","ADEP_mvt","TAXITIME_SEC_mvt"]).collect()
for c in ["FLIGHT_RULE_mvt","AIRCRAFT_TYPE_mvt","ADEP_mvt"]:
    t=ud_tr.group_by(c).len().with_columns((pl.col("len")/ud_tr.height).round(3).alias("share_tr")).drop("len")
    r=ud_rk.group_by(c).len().with_columns((pl.col("len")/ud_rk.height).round(3).alias("share_rk")).drop("len")
    print(t.join(r,on=c,how="full",coalesce=True).sort("share_rk",descending=True,nulls_last=True).head(15))
print("FLIGHT_mvt prefix (2 letters) share:")
pre=lambda df,tag: df.with_columns(pl.col("FLIGHT_mvt").str.slice(0,2).alias("pre")).group_by("pre").len().with_columns((pl.col("len")/df.height).round(3).alias(tag)).drop("len")
print(pre(ud_tr,"tr").join(pre(ud_rk,"rk"),on="pre",how="full",coalesce=True).sort("rk",descending=True,nulls_last=True).head(20))
print("ADES_mvt region share:")
reg=lambda df,tag: df.with_columns(pl.col("ADES_mvt").str.slice(0,2).alias("reg")).group_by("reg").len().with_columns((pl.col("len")/df.height).round(3).alias(tag)).drop("len")
print(reg(ud_tr,"tr").join(reg(ud_rk,"rk"),on="reg",how="full",coalesce=True).sort("rk",descending=True,nulls_last=True).head(12))
print("unmatched FLIGHT_mvt seen among matched training flights: train",ud_tr["FLIGHT_mvt"].is_in(md_tr["FLIGHT_mvt"]).mean(),"rank",ud_rk["FLIGHT_mvt"].is_in(md_tr["FLIGHT_mvt"]).mean())
print("unmatched STAND seen in train-unmatched STANDs: rank",ud_rk["STAND_mvt"].is_in(ud_tr["STAND_mvt"]).mean(), "seen in any train stand:",ud_rk["STAND_mvt"].is_in(md_tr["STAND_mvt"]).mean())
print("unmatched AIRCRAFT_TYPE seen in any train:",ud_rk["AIRCRAFT_TYPE_mvt"].is_in(md_tr["AIRCRAFT_TYPE_mvt"]).mean())
print("hour_utc dist unmatched train vs rank:")
h=lambda df,tag: df.group_by(pl.col("MVT_TIME_UTC_mvt").dt.hour().alias("h")).len().with_columns((pl.col("len")/df.height).round(3).alias(tag)).drop("len")
print(h(ud_tr,"tr").join(h(ud_rk,"rk"),on="h",how="full",coalesce=True).sort("h").transpose(include_header=True))
print("ranking unmatched: month split", ud_rk.group_by(pl.col("MVT_TIME_UTC_mvt").dt.month()).len())
print("=== G2. unmatched train: label by whether FLIGHT_mvt appears matched elsewhere ===")
print(ud_tr.with_columns(pl.col("FLIGHT_mvt").is_in(md_tr["FLIGHT_mvt"]).alias("seen")).group_by("seen").agg([pl.len(),pl.col("TAXITIME_SEC_mvt").median(),pl.col("TAXITIME_SEC_mvt").std().alias("sd"),(pl.col("TAXITIME_SEC_mvt")>=7200).mean().round(4).alias("gt2h"),(pl.col("TAXITIME_SEC_mvt")>=14400).sum().alias("n_gt4h")]))
print("=== H. worst OOF rows (latest main oof) ===")
import subprocess
oofs=sorted(glob.glob(B+"experiments/oof_main-*.parquet"),key=os.path.getmtime)
oof=pl.read_parquet(oofs[-1]); print(oofs[-1], oof.shape)
lab=tr.filter(pl.col("PHASE_mvt")=="DEP").select(["MVT_ID_mvt","ADEP_mvt","FLIGHT_mvt","FLIGHT_ID_mvt","SCHED_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt","AOBT_3_flt","MVT_TIME_UTC_mvt","TAXITIME_SEC_mvt","STAND_mvt","RUNWAY_mvt"]).collect()
j=lab.join(oof.drop_nulls("oof"),on="MVT_ID_mvt").with_columns(((pl.col("TAXITIME_SEC_mvt")-pl.col("oof"))**2).alias("se"))
print("n scored",j.height,"rmse",(j["se"].mean())**0.5)
tot=j["se"].sum()
w=j.sort("se",descending=True).head(30).with_columns((pl.col("se")/tot).round(4).alias("se_share"),pl.col("FLIGHT_ID_mvt").is_null().alias("unm"),(pl.col("BLOCK_TIME_UTC_mvt")==pl.col("SCHED_TIME_UTC_mvt")).alias("blk_eq_sched"))
print(w.select(["ADEP_mvt","FLIGHT_mvt","unm","blk_eq_sched","SCHED_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt","AOBT_3_flt","MVT_TIME_UTC_mvt","TAXITIME_SEC_mvt","oof","se_share"]))
print("share of SSE from rows with y>=7200 where pred<3600:", j.filter((pl.col("TAXITIME_SEC_mvt")>=7200)&(pl.col("oof")<3600))["se"].sum()/tot)
print("share of SSE from rows with pred>=7200 where y<3600 (false alarms):", j.filter((pl.col("oof")>=7200)&(pl.col("TAXITIME_SEC_mvt")<3600))["se"].sum()/tot, "n=",j.filter((pl.col("oof")>=7200)&(pl.col("TAXITIME_SEC_mvt")<3600)).height)
print("matched rows: SSE share from rows where |proxy - y| > 3600:", j.filter(pl.col("FLIGHT_ID_mvt").is_not_null()).with_columns((pl.col("MVT_TIME_UTC_mvt")-pl.col("AOBT_3_flt")).dt.total_seconds().alias("proxy")).filter((pl.col("proxy")-pl.col("TAXITIME_SEC_mvt")).abs()>3600)["se"].sum()/tot)
