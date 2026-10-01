"""Row-level support check: 2026 rows whose MVT-AOBT3 lies where 2025 has few rows (EHAM snow days), and how y
relates to MVT-AOBT3 in those bins in 2025. Output: docs/analysis/temporal_shift.md §4."""
from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "2")
import polars as pl  # noqa: E402

from temporal_shift_common import load  # noqa: E402

pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(250)
pl.Config.set_float_precision(2)
pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
pl.Config.set_tbl_hide_column_data_types(True)
pl.Config.set_tbl_hide_dataframe_shape(True)

d = load(["set", "PHASE_mvt", "APT", "y", "m_aobt3", "month", "clock_src"]).filter(pl.col("PHASE_mvt") == "DEP")
d = d.with_columns(pl.col("APT").cast(pl.Utf8), pl.col("m_aobt3").cut([1200, 1800, 2400, 3600, 5400, 7200]).alias("bin"))
tr = d.filter(pl.col("set") == "train")
print("\n## A. 2025: y - (MVT-AOBT3) by MVT-AOBT3 bin (pooled; genuine+aobt3 classes only)")
print(tr.filter(pl.col("clock_src").is_in(["genuine", "aobt3"])).group_by("bin").agg(
    pl.len(), (pl.col("y") - pl.col("m_aobt3")).mean().alias("mean_gap"), (pl.col("y") - pl.col("m_aobt3")).median().alias("med_gap"),
    (pl.col("y") - pl.col("m_aobt3")).std().alias("sd_gap")).sort("bin"))
print("\n## B. rows per MVT-AOBT3 bin: EHAM 2025 all months vs EHAM ranking Jan 2026 vs all airports 2025")
t = d.group_by(pl.when(pl.col("set") == "rank").then(pl.col("APT") + "-" + pl.col("month").cast(pl.Utf8)).otherwise(
    pl.lit("2025-") + pl.col("APT")).alias("grp"), "bin").len()
print(t.filter(pl.col("grp").str.contains("EHAM|LFPG|LSZH")).pivot(on="bin", index="grp", values="len", sort_columns=True).sort("grp"))
print("\n## C. EHAM 2025, genuine+aobt3: gap by bin (does the EHAM-specific offset hold at large MVT-AOBT3?)")
print(tr.filter((pl.col("APT") == "EHAM") & pl.col("clock_src").is_in(["genuine", "aobt3"])).group_by("bin").agg(
    pl.len(), (pl.col("y") - pl.col("m_aobt3")).mean().alias("mean_gap"), (pl.col("y") - pl.col("m_aobt3")).median().alias("med_gap")).sort("bin"))

print("\n## D. 2025, genuine+aobt3 rows: gap y-(MVT-AOBT3) by row bin x day context (is a long MVT-AOBT3 an NM error on"
      " a normal day but real on a disrupted day?)")
from temporal_shift_features import features  # noqa: E402

f = features().filter(pl.col("set") == "train").with_columns(pl.col("APT").cast(pl.Utf8))
norm = f.group_by("APT").agg(pl.col("day_m_aobt3_mean").median().alias("norm"))
f = f.join(norm, on="APT").with_columns(
    (pl.col("day_m_aobt3_mean") / pl.col("norm")).cut([1.1, 1.3]).alias("day_ratio"),
    pl.col("m_aobt3").cut([1200, 1800, 2400, 3600, 5400]).alias("bin"),
    (pl.col("cw_nbr_m_aobt3_60m") / pl.col("norm")).cut([1.1, 1.3, 1.6]).alias("cw_ratio"))
g = f.filter(pl.col("clock_src").cast(pl.Utf8).is_in(["genuine", "aobt3"]))
for ctx in ("day_ratio", "cw_ratio"):
    print(f"### context = {ctx} (ratio to airport norm of daily mean MVT-AOBT3); cells: mean gap (n)")
    t = g.group_by("bin", ctx).agg(pl.len().alias("n"), (pl.col("y") - pl.col("m_aobt3")).mean().round(0).alias("gap"))
    t = t.with_columns((pl.col("gap").cast(pl.Int64).cast(pl.Utf8) + " (" + pl.col("n").cast(pl.Utf8) + ")").alias("cell"))
    print(t.pivot(on=ctx, index="bin", values="cell", sort_columns=True).sort("bin"))
