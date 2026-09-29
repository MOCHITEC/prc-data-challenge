"""LEBL summer-2025 copy episode (report §7.2)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from reporting_fingerprint_common import *  # noqa
import numpy as np
pl.Config.set_tbl_rows(120); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
d = load_all().filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("apt") == "LEBL") & (pl.col("split") == "train"))
dd = (pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds()
d = d.with_columns((dd.abs() <= 6).cast(pl.Float64).alias("k0"), (((dd - 60).abs() <= 6) | ((dd + 60).abs() <= 6)).cast(pl.Float64).alias("k1"),
                   (pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("m_s"))
ex = (pl.col("k0").mean() - pl.col("k1").mean() / 2).round(4)
w = d.group_by(pl.col("MVT_TIME_UTC_mvt").dt.truncate("1w").alias("week")).agg(ex.alias("excess"), pl.len()).sort("week")
print(w)
s = d.filter(pl.col("month").is_in([7, 8, 9]))
print("summer by operator (top)")
print(s.group_by(pl.col("AIRCRAFT_OPERATOR_flt").str.slice(0, 8)).agg(pl.len(), ex.alias("excess")).sort("len", descending=True).head(12))
print("summer by stand first char"); print(s.group_by(pl.col("STAND_mvt").str.slice(0, 1).alias("st")).agg(pl.len(), ex.alias("excess")).sort("len", descending=True).head(10))
print("summer by delay bin (MVT-SCHED, 10 min)"); print(s.with_columns((pl.col("m_s") / 600).floor().clip(-1, 12).alias("b")).group_by("b").agg(pl.len(), ex.alias("excess"), pl.col("TAXITIME_SEC_mvt").mean().round(0)).sort("b"))
print("summer: how big is y for copies, sd:", s.filter(pl.col("k0") == 1).select(pl.col("TAXITIME_SEC_mvt").mean().alias("m"), pl.col("TAXITIME_SEC_mvt").std().alias("sd"), (pl.col("TAXITIME_SEC_mvt") > 3600).sum().alias("gt1h")))
print("summer hour-of-day"); print(s.group_by(pl.col("SCHED_TIME_UTC_mvt").dt.hour().alias("h")).agg(pl.len(), ex.alias("excess")).sort("h"))
