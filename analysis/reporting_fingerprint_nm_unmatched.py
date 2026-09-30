"""NM-unmatched departures: who they are and whether other rows let us recover an off-block proxy (report §6)."""
import sys
sys.path.insert(0, __import__("os").path.dirname(__file__))
from reporting_fingerprint_common import *  # noqa
import numpy as np

pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(220)
df = load_all()
dep = df.filter(pl.col("PHASE_mvt") == "DEP").with_columns(pl.col("FLIGHT_ID_mvt").is_null().alias("nm_null"),
    sec_expr := (pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("m_s"))
u = dep.filter(pl.col("nm_null"))
print("unmatched by split/apt"); print(u.group_by("split", "apt").len().pivot(on="split", index="apt", values="len").sort("apt"))
tr = u.filter(pl.col("split") == "train")
print("which columns are filled in unmatched rows (train):")
print(tr.select([pl.col(c).is_not_null().mean().alias(c) for c in ["FLIGHT_mvt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "ADES_mvt", "FLIGHT_RULE_mvt", "CALLSIGN_flt", "AIRCRAFT_OPERATOR_flt"]]))
# flight-number prefix (airline designator), flight-rule
tr = tr.with_columns(pl.col("FLIGHT_mvt").str.extract(r"^([A-Z0-9]{2,3}?)(\d+)[A-Z]?$", 1).alias("pfx"),
                     pl.col("FLIGHT_mvt").str.contains(r"^\d").alias("starts_digit"),
                     pl.col("FLIGHT_mvt").str.len_chars().alias("fl_len"))
print(tr.group_by("FLIGHT_RULE_mvt").agg(pl.len(), pl.col("TAXITIME_SEC_mvt").mean().round(0), (pl.col("clock_src") == "sched").mean().round(3).alias("sched")).sort("len", descending=True))
print("clock_src of unmatched (train):", tr["clock_src"].value_counts().sort("count", descending=True))
print("y stats unmatched vs matched by src")
print(dep.filter(pl.col("split") == "train").group_by("nm_null", "clock_src").agg(pl.len(), pl.col("TAXITIME_SEC_mvt").mean().round(0).alias("ymean"), pl.col("TAXITIME_SEC_mvt").std().round(0).alias("ysd"), (pl.col("TAXITIME_SEC_mvt") > 3600).sum().alias("n_gt1h")).sort("nm_null", "len", descending=[False, True]))
# flight number re-occurrence: does the same FLIGHT_mvt at the same airport appear NM-matched on other days?
m = dep.filter(~pl.col("nm_null")).group_by("apt", "FLIGHT_mvt").agg(pl.len().alias("n_matched_same_fl"), pl.col("AIRCRAFT_OPERATOR_flt").mode().first().alias("op_mode"),
     pl.col("MARKET_SEGMENT_flt").mode().first().alias("seg_mode"), ((pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds()).median().alias("fl_med_mvt_aobt"),
     ((pl.col("AOBT_3_flt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds()).median().alias("fl_med_aobt_sched"))
for sp in ["train", "rank"]:
    x = u.filter(pl.col("split") == sp).join(m, on=["apt", "FLIGHT_mvt"], how="left")
    print(sp, "unmatched rows whose flight number is NM-matched elsewhere:", round(x["n_matched_same_fl"].is_not_null().mean(), 3), "n=", x.height)
x = tr.join(m, on=["apt", "FLIGHT_mvt"], how="left")
print(x.group_by(pl.col("n_matched_same_fl").is_not_null().alias("fl_seen_matched")).agg(pl.len(), pl.col("TAXITIME_SEC_mvt").mean().round(0), pl.col("TAXITIME_SEC_mvt").std().round(0).alias("ysd"),
     (pl.col("clock_src") == "sched").mean().round(3).alias("sched"), (pl.col("TAXITIME_SEC_mvt") > 3600).sum().alias("gt1h"), pl.col("m_s").median().alias("med_mvt_sched")))
# the gt1h unmatched rows
big = tr.filter(pl.col("TAXITIME_SEC_mvt") > 3600)
print("unmatched y>1h rows:", big.height, " by apt:", big["apt"].value_counts().sort("count", descending=True).to_dicts())
print(big.select("apt", "FLIGHT_mvt", "FLIGHT_RULE_mvt", "AIRCRAFT_TYPE_mvt", "STAND_mvt", "RUNWAY_mvt", "ADES_mvt", "clock_src", "TAXITIME_SEC_mvt", "m_s").sort("TAXITIME_SEC_mvt", descending=True).head(30))
# predicate: m_s (MVT - SCHED) is visible; how well does "m_s > 1h" flag big y among unmatched?
for thr in [1800, 3600, 7200, 3 * 3600]:
    s = tr.filter(pl.col("m_s") > thr)
    print(f"unmatched & MVT-SCHED>{thr}s: n={s.height}, share y>1h={round((s['TAXITIME_SEC_mvt']>3600).mean(),3)}, share sched={round((s['clock_src']=='sched').mean(),3)}, rank n={u.filter((pl.col('split')=='rank')&(pl.col('m_s')>thr)).height}")
