"""Period comparison 2025-01/07 vs 2026-01/07 of quantities observable in ranking.parquet (DEP + ARR rows).

Output: printed markdown tables (copied into docs/analysis/temporal_shift.md §1).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from temporal_shift_common import load

pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(250)
pl.Config.set_float_precision(3)
pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
pl.Config.set_tbl_hide_column_data_types(True)
pl.Config.set_tbl_hide_dataframe_shape(True)

d = load(["set", "PHASE_mvt", "APT", "year", "month", "date_local", "m_aobt3", "m_sched", "m_eobt1", "nm", "y",
          "RUNWAY_mvt", "STAND_mvt", "airline", "AIRCRAFT_TYPE_mvt", "MARKET_SEGMENT_flt", "hour_local",
          "aobt3_m_sched"])
d = d.with_columns((pl.col("year").cast(pl.Utf8) + "-" + pl.col("month").cast(pl.Utf8).str.zfill(2)).alias("ym"))
d_all = d
d = d.filter(pl.col("month").is_in([1, 7]))
# Jan->Feb and Jul->Aug 2025 serve as a within-year reference for "how big is a TVD"
dep_ref = d_all.filter((pl.col("PHASE_mvt") == "DEP") & pl.col("month").is_in([1, 2, 7, 8]))
PER = ["2025-01", "2026-01", "2025-07", "2026-07"]

dep = d.filter(pl.col("PHASE_mvt") == "DEP")
arr = d.filter(pl.col("PHASE_mvt") == "ARR")

print("\n## A. DEP volume / NM match / clock gaps by airport x period")
t = (dep.group_by("APT", "ym").agg(
    (pl.len() / pl.col("date_local").n_unique()).alias("dep_per_day"),
    (1 - pl.col("nm").mean()).alias("nm_unmatched"),
    pl.col("m_aobt3").median().alias("med_mvt_aobt3"),
    pl.col("m_aobt3").filter(pl.col("m_aobt3").abs() < 7200).mean().alias("mean_mvt_aobt3"),
    pl.col("m_sched").median().alias("med_mvt_sched"),
    (pl.col("aobt3_m_sched").abs() <= 60).mean().alias("aobt3_eq_sched60"),
    (pl.col("m_sched") > 3 * 3600).mean().alias("mvt_sched_gt3h"),
    (pl.col("m_sched") < 0).mean().alias("mvt_before_sched"),
).sort("APT", "ym"))
print(t)

print("\n## B. ARR taxi-in by airport x period")
ta = arr.filter(pl.col("y").is_between(0, 7200)).group_by("APT", "ym").agg(
    pl.len().alias("n"), pl.col("y").mean().alias("taxi_in_mean"), pl.col("y").median().alias("taxi_in_med"),
    pl.col("y").quantile(0.9).alias("taxi_in_p90")).sort("APT", "ym")
print(ta.pivot(on="ym", index="APT", values="taxi_in_mean").select(["APT", *PER]))
print(ta.pivot(on="ym", index="APT", values="taxi_in_med").select(["APT", *PER]))

print("\n## B2. ARR taxi-in unfiltered (including outliers) mean / share > 1h / share < 0")
ta2 = arr.group_by("APT", "ym").agg(pl.col("y").mean().alias("mean"), (pl.col("y") > 3600).mean().alias("gt1h"),
                                    (pl.col("y") <= 0).mean().alias("le0"), pl.col("y").is_null().mean().alias("null"))
print(ta2.sort("APT", "ym"))


def tvd(col: str, frame: pl.DataFrame, a: str, b: str) -> pl.DataFrame:
    """Total-variation distance of a categorical distribution between periods a and b, per airport, plus the share
    of period-b rows whose level never appears in ANY 2025 month (novel level)."""
    rows = []
    all25 = load(["set", "PHASE_mvt", "APT", col]).filter((pl.col("set") == "train") & (pl.col("PHASE_mvt") == "DEP"))
    for apt in sorted(frame["APT"].unique().to_list()):
        f = frame.filter(pl.col("APT") == apt)
        pa = f.filter(pl.col("ym") == a)[col].cast(pl.Utf8).value_counts(normalize=True, name="p")
        pb = f.filter(pl.col("ym") == b)[col].cast(pl.Utf8).value_counts(normalize=True, name="q")
        j = pa.join(pb, on=col, how="full", coalesce=True).fill_null(0)
        seen = set(all25.filter(pl.col("APT") == apt)[col].cast(pl.Utf8).drop_nulls().to_list())
        fb = f.filter(pl.col("ym") == b)[col].cast(pl.Utf8)
        novel = float((~fb.is_in(list(seen)) & fb.is_not_null()).mean())
        rows.append({"APT": apt, "col": col, "tvd": 0.5 * float(np.abs(j["p"] - j["q"]).sum()), "novel_share": novel})
    return pl.DataFrame(rows)


print("\n## C. Categorical shift (TVD) 2025-01 -> 2026-01 and 2025-07 -> 2026-07, DEP rows")
out = []
for col in ("RUNWAY_mvt", "STAND_mvt", "airline", "AIRCRAFT_TYPE_mvt", "hour_local"):
    for a, b in (("2025-01", "2026-01"), ("2025-07", "2026-07"), ("2025-01", "2025-02"), ("2025-07", "2025-08")):
        out.append(tvd(col, dep_ref, a, b).with_columns(pl.lit(b).alias("to")))
c = pl.concat(out)
print(c.pivot(on=["col", "to"], index="APT", values="tvd"))
print(c.pivot(on=["col", "to"], index="APT", values="novel_share"))

print("\n## D. Runway share by period (top runways) per airport, DEP")
rw = dep.group_by("APT", "ym", "RUNWAY_mvt").len().with_columns(
    (pl.col("len") / pl.col("len").sum().over("APT", "ym")).alias("share"))
print(rw.filter(pl.col("share") > 0.02).pivot(on="ym", index=["APT", "RUNWAY_mvt"], values="share")
      .select(["APT", "RUNWAY_mvt", *PER]).sort("APT", "RUNWAY_mvt"))
