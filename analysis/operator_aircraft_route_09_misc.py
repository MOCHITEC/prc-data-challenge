"""operator_aircraft_route 09: supporting tables.

A. NM-unmatched departures are (outside LIRF/LTFM) heavily delayed flights: BLOCK - SCHED by airport x NM match.
B. Inbound-link upper bound: latest ARR in-block at the same stand before (AOBT_3 - 10 min); y <= MVT - inblock.
"""

from __future__ import annotations

import polars as pl
from operator_aircraft_route_common import load_arr, load_dep, md


def main() -> None:
    d = load_dep(["ADEP_mvt", "FLIGHT_ID_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "split",
                  "STAND_mvt", "AOBT_3_flt", "airline", "y", "clock_src"])
    sec = lambda a, b: (pl.col(a) - pl.col(b)).dt.total_seconds().cast(pl.Float64)  # noqa: E731
    d = d.with_columns(sec("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("mms"),
                       sec("BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("bms"),
                       pl.col("FLIGHT_ID_mvt").is_not_null().alias("nm"))
    print("## A. NM 照合の有無と遅延（BLOCK − SCHED は学習のみ, MVT − SCHED は ranking も）\n")
    a = d.group_by("ADEP_mvt", "nm").agg(
        (pl.col("split") == "train").sum().alias("n_train"),
        pl.col("bms").median().alias("block−sched_med"), pl.col("bms").quantile(0.1).alias("p10"),
        (pl.col("bms") > 3600).mean().alias("share>1h"),
        pl.col("mms").filter(pl.col("split") == "train").median().alias("mvt−sched_med_train"),
        pl.col("mms").filter(pl.col("split") == "rank").median().alias("mvt−sched_med_rank"),
    ).sort("ADEP_mvt", "nm")
    print(md(a, "{:,.2f}"))

    print("\n## B. 到着便リンク（AOBT_3 − 10 分より前の同スタンド最終インブロック）による上限 y ≤ MVT − inblock\n")
    arr = load_arr(["ADES_mvt", "STAND_mvt", "BLOCK_TIME_UTC_mvt", "airline"]).drop_nulls(["STAND_mvt", "BLOCK_TIME_UTC_mvt"]).select(
        pl.col("ADES_mvt").alias("ADEP_mvt"), "STAND_mvt", pl.col("BLOCK_TIME_UTC_mvt").alias("inb"),
        pl.col("airline").alias("arr_al")).sort("inb")
    t = d.filter((pl.col("split") == "train") & pl.col("AOBT_3_flt").is_not_null() & pl.col("STAND_mvt").is_not_null())
    t = t.with_columns((pl.col("AOBT_3_flt") - pl.duration(minutes=10)).alias("anchor")).sort("anchor")
    t = t.join_asof(arr, left_on="anchor", right_on="inb", by=["ADEP_mvt", "STAND_mvt"], strategy="backward",
                    check_sortedness=False)
    t = t.with_columns(sec("MVT_TIME_UTC_mvt", "inb").alias("ub"), (pl.col("arr_al") == pl.col("airline")).alias("same_airline"))
    b = t.with_columns(pl.col("ub").truediv(60).cut([30, 45, 60, 90, 120, 240]).alias("ub_min")).group_by("ub_min", "same_airline").agg(
        pl.len().alias("n"), (pl.col("y") <= pl.col("ub")).mean().alias("valid"),
        (pl.col("y") > pl.col("ub") * 0.8).mean().alias("binding(y>0.8ub)")).sort("ub_min", "same_airline")
    print(md(b.drop_nulls("same_airline"), "{:.4f}"))


if __name__ == "__main__":
    main()
