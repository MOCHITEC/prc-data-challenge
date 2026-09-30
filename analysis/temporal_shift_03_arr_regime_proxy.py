"""Does the ARRIVAL side reveal the airport's reporting regime? ARR rows keep BLOCK_TIME in ranking.parquet, so if the
share of arrivals whose in-block time is a copy of the scheduled time moves together with the share of departures
whose off-block is a copy of the scheduled time (clock_src == 'sched'), it is an observable, leak-free proxy for
the DEP clock-source mix on a given day / hour.

Output: markdown tables (docs/analysis/temporal_shift.md §3).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from temporal_shift_common import RAW, load

pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(250)
pl.Config.set_float_precision(3)
pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
pl.Config.set_tbl_hide_column_data_types(True)
pl.Config.set_tbl_hide_dataframe_shape(True)

TOL = 60
files = sorted(RAW.glob("training_2025-*.parquet")) + [RAW / "ranking.parquet"]
cols = ["MVT_ID_mvt", "PHASE_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
        "ARVT_1_flt", "ARVT_3_flt", "FLIGHT_ID_mvt"]
arr = pl.concat([pl.scan_parquet(f).select(cols).filter(pl.col("PHASE_mvt") == "ARR") for f in files],
                how="vertical_relaxed").collect()
ts = arr.schema["MVT_TIME_UTC_mvt"]
arr = arr.with_columns([pl.col(c).cast(ts) for c in ("BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "ARVT_1_flt",
                                                     "ARVT_3_flt")])


def gap(a: str, b: str) -> pl.Expr:
    return (pl.col(a) - pl.col(b)).dt.total_seconds().abs()


base = load(["MVT_ID_mvt", "date_local", "hour_local", "year", "month"])
arr = arr.join(base, on="MVT_ID_mvt", how="left").with_columns(
    (gap("BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt") <= TOL).alias("a_blk_eq_sched"),
    (gap("BLOCK_TIME_UTC_mvt", "ARVT_3_flt") <= TOL).alias("a_blk_eq_arvt3"),
    (gap("BLOCK_TIME_UTC_mvt", "ARVT_1_flt") <= TOL).alias("a_blk_eq_arvt1"),
    (gap("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt") <= TOL).alias("a_mvt_eq_sched"),
    (gap("MVT_TIME_UTC_mvt", "ARVT_3_flt") <= TOL).alias("a_mvt_eq_arvt3"),
    ((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("MVT_TIME_UTC_mvt")).dt.total_seconds() <= 0).alias("a_nonpos"),
    pl.col("FLIGHT_ID_mvt").is_null().alias("a_nm_unmatched"),
    pl.col("ADES_mvt").cast(pl.Categorical).alias("APT"),
)
flags = ["a_blk_eq_sched", "a_blk_eq_arvt3", "a_blk_eq_arvt1", "a_mvt_eq_sched", "a_mvt_eq_arvt3", "a_nonpos",
         "a_nm_unmatched"]

print("\n## A. ARR copy rates by airport (2025 all / 2026-01 / 2026-07)")
arr = arr.with_columns(pl.when((pl.col("year") == 2025) & ~pl.col("month").is_in([1, 7])).then(pl.lit("2025-other")).otherwise(
    pl.col("year").cast(pl.Utf8) + "-" + pl.col("month").cast(pl.Utf8).str.zfill(2)).alias("per"))
print(arr.group_by("APT", "per").agg([pl.col(f).mean() for f in flags]).sort("APT", "per"))

dep = load(["PHASE_mvt", "set", "APT", "date_local", "hour_local", "clock_src", "m_sched", "nm", "y", "m_aobt3"]) \
    .filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("set") == "train"))

for grain in (["APT", "date_local"], ["APT", "date_local", "hour_local"]):
    print(f"\n## B. correlation, grain = {grain} (2025): ARR flags vs DEP clock-source shares")
    ga = arr.filter(pl.col("year") == 2025).group_by(grain).agg([pl.col(f).mean() for f in flags] +
                                                                 [pl.len().alias("n_arr")])
    gd = dep.group_by(grain).agg(
        (pl.col("clock_src") == "sched").mean().alias("d_sched"),
        (pl.col("clock_src") == "aobt3").mean().alias("d_aobt3"),
        (pl.col("clock_src") == "eobt1").mean().alias("d_eobt1"),
        (pl.col("clock_src") == "genuine").mean().alias("d_genuine"),
        (1 - pl.col("nm").mean()).alias("d_nm_unmatched"),
        pl.len().alias("n_dep"))
    g = gd.join(ga, on=grain, how="inner").filter((pl.col("n_arr") >= 5) & (pl.col("n_dep") >= 5))
    rows = []
    for apt in sorted(g["APT"].unique().to_list()):
        f = g.filter(pl.col("APT") == apt)
        r = {"APT": apt, "n": f.height}
        for a in ("a_blk_eq_sched", "a_blk_eq_arvt3", "a_nm_unmatched"):
            for dcol in ("d_sched", "d_aobt3", "d_nm_unmatched"):
                x, y = f[a].to_numpy(), f[dcol].to_numpy()
                r[f"{a[2:]}~{dcol[2:]}"] = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None
        rows.append(r)
    print(pl.DataFrame(rows))
