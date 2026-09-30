"""Deep dive LIRF / LFPG (+ the new-stand finding at EDDM and runway-config shifts).

Output: markdown tables (docs/analysis/temporal_shift.md §5).
"""

from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "2")
import polars as pl  # noqa: E402

from temporal_shift_common import load  # noqa: E402

pl.Config.set_tbl_rows(120)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(250)
pl.Config.set_float_precision(3)
pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
pl.Config.set_tbl_hide_column_data_types(True)
pl.Config.set_tbl_hide_dataframe_shape(True)

d = load(["set", "PHASE_mvt", "APT", "year", "month", "hour_local", "m_aobt3", "m_sched", "y", "clock_src", "nm",
          "RUNWAY_mvt", "STAND_mvt", "airline", "date_local"]).filter(pl.col("PHASE_mvt") == "DEP")
d = d.with_columns(pl.col("APT").cast(pl.Utf8), pl.col("RUNWAY_mvt").cast(pl.Utf8),
                   (pl.col("year") * 100 + pl.col("month")).alias("ym"),
                   pl.col("STAND_mvt").str.replace_all(r"\d+$", "").alias("stand_prefix"),
                   pl.col("STAND_mvt").str.slice(0, 1).alias("stand_1"))
tr = d.filter(pl.col("set") == "train")

print("\n## A. LIRF monthly mean y decomposed by clock source (contribution = share x mean y of class)")
lir = tr.filter(pl.col("APT") == "LIRF")
dec = lir.group_by("month", "clock_src").agg(pl.len().alias("n"), pl.col("y").mean().alias("ym_"),
                                             (pl.col("m_aobt3") - pl.col("y")).mean().alias("off"))
dec = dec.with_columns((pl.col("n") / pl.col("n").sum().over("month")).alias("share"))
dec = dec.with_columns((pl.col("share") * pl.col("ym_")).alias("contrib"))
print(dec.pivot(on="clock_src", index="month", values="contrib", sort_columns=True).sort("month"))
print(dec.pivot(on="clock_src", index="month", values="ym_", sort_columns=True).sort("month"))
print("offset BLOCK-AOBT3 by class")
print(dec.pivot(on="clock_src", index="month", values="off", sort_columns=True).sort("month"))

print("\n## A2. genuine-class offset (BLOCK-AOBT3) by month for all airports (median)")
g = tr.filter(pl.col("clock_src") == "genuine").group_by("APT", "month").agg(
    (pl.col("m_aobt3") - pl.col("y")).median().alias("off_med"))
print(g.pivot(on="month", index="APT", values="off_med", sort_columns=True).sort("APT"))

print("\n## B. LIRF / LFPG: y and RMSE-relevant stats by stand letter / prefix (train) and ranking share")
for apt in ("LIRF", "LFPG"):
    a = d.filter(pl.col("APT") == apt)
    s = a.group_by("stand_1").agg(
        (pl.col("set") == "train").sum().alias("n_train"),
        ((pl.col("set") == "rank")).sum().alias("n_rank"),
        pl.col("y").median().alias("y_med"), pl.col("y").std().alias("y_sd"),
        (pl.col("y") > 3 * 3600).sum().alias("n_gt3h"),
        (pl.col("clock_src") == "sched").mean().alias("src_sched"),
        (1 - pl.col("nm").mean()).alias("nm_unm"),
    ).sort("n_train", descending=True)
    print(f"### {apt}")
    print(s.head(25))

print("\n## C. LIRF y > 3h rows: characteristics (2025)")
big = tr.filter((pl.col("APT") == "LIRF") & (pl.col("y") > 3 * 3600))
print(big.group_by("clock_src").len().sort("len", descending=True))
print(big.group_by("hour_local").len().sort("hour_local"))
print(big.group_by("nm").len())
print(big.select(pl.col("m_sched").quantile(q).alias(f"m_sched_q{q}") for q in (0.1, 0.5, 0.9)))
print("LIRF rows with m_sched > 3h: count / share y>3h among them, by set")
print(d.filter((pl.col("APT") == "LIRF") & (pl.col("m_sched") > 3 * 3600)).group_by("set", "nm").agg(
    pl.len(), (pl.col("y") > 3 * 3600).mean().alias("share_y_gt3h"), (pl.col("clock_src") == "sched").mean()))

print("\n## D. LFPG per-runway taxi-out by month (median y), and share")
lf = tr.filter(pl.col("APT") == "LFPG")
print(lf.group_by("month", "RUNWAY_mvt").agg(pl.col("y").median()).filter(pl.col("RUNWAY_mvt").is_not_null())
      .pivot(on="RUNWAY_mvt", index="month", values="y", sort_columns=True).sort("month"))
print(d.filter(pl.col("APT") == "LFPG").group_by("ym", "RUNWAY_mvt").len().with_columns(
    (pl.col("len") / pl.col("len").sum().over("ym")).round(3).alias("s"))
      .pivot(on="RUNWAY_mvt", index="ym", values="s", sort_columns=True).sort("ym"))

print("\n## E. Novel stands in ranking (never seen in 2025) by airport and prefix")
seen = tr.select("APT", "STAND_mvt").unique().with_columns(pl.lit(True).alias("seen"))
rk = d.filter(pl.col("set") == "rank").join(seen, on=["APT", "STAND_mvt"], how="left")
nov = rk.filter(pl.col("seen").is_null() & pl.col("STAND_mvt").is_not_null())
print(nov.group_by("APT", "ym").len().sort("APT", "ym"))
print(nov.group_by("APT", "ym", "stand_prefix").agg(pl.len(), pl.col("STAND_mvt").unique().sort().head(8).alias("ex"))
      .filter(pl.col("len") > 20).sort("APT", "len", descending=[False, True]))
# MVT-AOBT3 on novel stands vs seen stands of the same airport-month: does the novel apron differ?
print(rk.with_columns(pl.col("seen").is_null().alias("novel")).filter(pl.col("m_aobt3").is_between(0, 7200))
      .group_by("APT", "ym", "novel").agg(pl.len(), pl.col("m_aobt3").median()).sort("APT", "ym", "novel")
      .filter(pl.col("APT").is_in(["EDDM", "EDDF", "LSZH"])))

print("\n## F. EGLL runway 09R departures share by month (easterly ops) and y by runway")
eg = d.filter(pl.col("APT") == "EGLL")
print(eg.group_by("ym").agg((pl.col("RUNWAY_mvt") == "09R").mean().alias("share_09R"),
                            pl.col("y").filter(pl.col("RUNWAY_mvt") == "09R").median().alias("y_09R"),
                            pl.col("y").filter(pl.col("RUNWAY_mvt") != "09R").median().alias("y_other"),
                            pl.col("m_aobt3").filter(pl.col("RUNWAY_mvt") == "09R").median().alias("ma_09R"),
                            pl.col("m_aobt3").filter(pl.col("RUNWAY_mvt") != "09R").median().alias("ma_other"))
      .sort("ym"))
