"""Monthly levels per airport in 2025 (+ observable levels in 2026): is ARR taxi-in (or MVT-AOBT_3) a usable proxy
for the taxi-out level, and how does the unobservable offset BLOCK-AOBT_3 drift over the year?

Output: printed markdown tables (docs/analysis/temporal_shift.md §2).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from temporal_shift_common import load

pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(250)
pl.Config.set_float_precision(1)
pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
pl.Config.set_tbl_hide_column_data_types(True)
pl.Config.set_tbl_hide_dataframe_shape(True)

d = load(["set", "PHASE_mvt", "APT", "year", "month", "date_local", "m_aobt3", "y", "clock_src", "nm"])
d = d.with_columns((pl.col("year") * 100 + pl.col("month")).alias("ym"))
dep = d.filter(pl.col("PHASE_mvt") == "DEP")
arr = d.filter((pl.col("PHASE_mvt") == "ARR") & pl.col("y").is_between(0, 7200))

m_dep = dep.group_by("APT", "ym").agg(
    pl.col("y").filter(pl.col("y").is_between(0, 7200)).mean().alias("y_mean"),
    pl.col("y").mean().alias("y_mean_raw"),
    pl.col("m_aobt3").filter(pl.col("m_aobt3").is_between(0, 7200)).mean().alias("m_aobt3_mean"),
    (pl.col("m_aobt3") - pl.col("y")).filter(pl.col("y").is_between(0, 7200) & pl.col("m_aobt3").is_between(0, 7200))
    .mean().alias("offset_block_minus_aobt3"),
    (pl.col("clock_src") == "sched").mean().alias("src_sched"),
    (pl.col("clock_src") == "aobt3").mean().alias("src_aobt3"),
    (pl.col("clock_src") == "genuine").mean().alias("src_genuine"),
)
m_arr = arr.group_by("APT", "ym").agg(pl.col("y").mean().alias("taxi_in_mean"))
m = m_dep.join(m_arr, on=["APT", "ym"], how="left").sort("APT", "ym")

print("\n## A. per airport, 2025 months: y mean (0-7200), MVT-AOBT3 mean, offset, taxi-in, clock-source shares")
for col in ("y_mean", "m_aobt3_mean", "offset_block_minus_aobt3", "taxi_in_mean", "src_sched", "src_aobt3"):
    print(f"\n### {col}")
    fmt = pl.col(col).round(3) if col.startswith("src") else pl.col(col).round(0)
    print(m.with_columns(fmt).pivot(on="ym", index="APT", values=col, sort_columns=True))

print("\n## B. within-airport correlation across the 12 months of 2025 (monthly means)")
rows = []
tr = m.filter(pl.col("ym") < 202600)
for apt in sorted(tr["APT"].unique().to_list()):
    f = tr.filter(pl.col("APT") == apt)
    y, a, ti, off = (f[c].to_numpy() for c in ("y_mean", "m_aobt3_mean", "taxi_in_mean", "offset_block_minus_aobt3"))
    rows.append({"APT": apt, "corr(y,taxi_in)": np.corrcoef(y, ti)[0, 1], "corr(y,mvt-aobt3)": np.corrcoef(y, a)[0, 1],
                 "sd_month_y": y.std(), "sd_month_offset": off.std(),
                 "slope_y_on_taxi_in": np.polyfit(ti, y, 1)[0],
                 "offset_trend_s_per_month": np.polyfit(np.arange(len(off)), off, 1)[0]})
print(pl.DataFrame(rows))

print("\n## C. daily level: within-airport corr of daily y mean with daily taxi-in mean / daily MVT-AOBT3 mean (2025)")
dd = dep.filter(pl.col("ym") < 202600).group_by("APT", "date_local").agg(
    pl.col("y").filter(pl.col("y").is_between(0, 7200)).mean().alias("y"),
    pl.col("m_aobt3").filter(pl.col("m_aobt3").is_between(0, 7200)).mean().alias("a"),
    pl.len().alias("n"))
da = arr.filter(pl.col("ym") < 202600).group_by("APT", "date_local").agg(pl.col("y").mean().alias("ti"))
dd = dd.join(da, on=["APT", "date_local"])
rows = []
for apt in sorted(dd["APT"].unique().to_list()):
    f = dd.filter(pl.col("APT") == apt)
    rows.append({"APT": apt, "days": f.height,
                 "corr(y,ti)": np.corrcoef(f["y"], f["ti"])[0, 1],
                 "corr(y,a)": np.corrcoef(f["y"], f["a"])[0, 1],
                 "corr(y,n)": np.corrcoef(f["y"], f["n"])[0, 1],
                 "corr(y-a, ti)": np.corrcoef(f["y"] - f["a"], f["ti"])[0, 1]})
print(pl.DataFrame(rows).with_columns(pl.col(pl.Float64).round(3)))

print("\n## D. 2026 observable deltas vs same month 2025")
obs = m.select("APT", "ym", "m_aobt3_mean", "taxi_in_mean")
rows = []
for apt in sorted(obs["APT"].unique().to_list()):
    f = obs.filter(pl.col("APT") == apt)
    g = {r["ym"]: r for r in f.iter_rows(named=True)}
    for a, b in ((202501, 202601), (202507, 202607)):
        rows.append({"APT": apt, "to": b, "d_mvt_aobt3": g[b]["m_aobt3_mean"] - g[a]["m_aobt3_mean"],
                     "d_taxi_in": g[b]["taxi_in_mean"] - g[a]["taxi_in_mean"],
                     "y_2025": g[a]["m_aobt3_mean"]})
print(pl.DataFrame(rows).pivot(on="to", index="APT", values=["d_mvt_aobt3", "d_taxi_in"]))
