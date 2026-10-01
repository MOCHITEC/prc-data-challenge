"""LIRF: does the paired inbound arrival's fingerprint predict the departure's clock source? (report §4)"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from reporting_fingerprint_common import *  # noqa
from reporting_fingerprint_feats import build
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
fp = build()
d = load_all().filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("apt") == "LIRF")).join(fp, on="MVT_ID_mvt")
d = d.with_columns(((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= 6).cast(pl.Float64).alias("dep_copy"),
                   (pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("m_s"))
tr = d.filter(pl.col("split") == "train")
print("turnaround pairing coverage (train/rank):", d.group_by("split").agg(pl.col("ta_copy").is_not_nan().mean().round(3), (pl.col("ta_same_op") == 1).mean().round(3).alias("same_op")))
print("P(dep sched copy) by paired-arrival copy flag, same operator only:")
t = tr.filter(pl.col("ta_same_op") == 1)
print(t.group_by("ta_copy").agg(pl.len(), pl.col("dep_copy").mean().round(4)).sort("ta_copy"))
# within operator: centred
t2 = t.with_columns((pl.col("dep_copy") - pl.col("dep_copy").mean().over("AIRCRAFT_OPERATOR_flt")).alias("dc"))
print("within-operator uplift:", t2.group_by("ta_copy").agg(pl.len(), pl.col("dc").mean().round(4)).sort("ta_copy"))
# delayed departures only (MVT - SCHED > 1h): here a sched copy costs > 1h
t3 = t.filter(pl.col("m_s") > 3600)
print("delayed (MVT-SCHED>1h):", t3.group_by("ta_copy").agg(pl.len(), pl.col("dep_copy").mean().round(4), pl.col("TAXITIME_SEC_mvt").mean().round(0)).sort("ta_copy"))
# day-level rate buckets
print(tr.with_columns((pl.col("fa_dayop_rate") * 10).floor().clip(0, 6).alias("b")).group_by("b").agg(pl.len(), pl.col("dep_copy").mean().round(4)).sort("b"))
