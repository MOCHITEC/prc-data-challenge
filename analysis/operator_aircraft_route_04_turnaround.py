"""operator_aircraft_route 04: ARR -> DEP turnaround linking at the same stand.

For every departure (training and ranking) find the latest arrival that blocked in at the same airport and stand
before the departure's take-off (ranking keeps ARR BLOCK times and DEP MVT times, so this is leak-free). The link
is called "clean" when no other departure took off from that stand between the arrival's in-block and this
departure's take-off (a single stand holds one aircraft).

Physical constraint used: the departing aircraft cannot push back before its inbound blocked in, so for a correct
link BLOCK_dep >= inblock_arr, i.e. y <= ta_gap = MVT_dep - inblock_arr (an upper bound on y available at scoring
time).

Writes data/interim/oar_turn.parquet (MVT_ID_mvt + turnaround features) for script 06.
"""

from __future__ import annotations

import os

import numpy as np
import polars as pl
from operator_aircraft_route_common import INTERIM, load_arr, load_dep, md


def build() -> pl.DataFrame:
    dep = load_dep(["MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt",
                    "airline", "AIRCRAFT_TYPE_mvt", "split", "y", "clock_src", "FLIGHT_ID_mvt"])
    arr = load_arr(["ADES_mvt", "STAND_mvt", "BLOCK_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
                    "airline", "AIRCRAFT_TYPE_mvt", "TAXITIME_SEC_mvt"])
    arr = arr.filter(pl.col("STAND_mvt").is_not_null() & pl.col("BLOCK_TIME_UTC_mvt").is_not_null()).select(
        pl.col("ADES_mvt").alias("ADEP_mvt"), "STAND_mvt",
        pl.col("BLOCK_TIME_UTC_mvt").alias("arr_inblock"),
        pl.col("SCHED_TIME_UTC_mvt").alias("arr_sched"),
        pl.col("airline").alias("arr_airline"),
        pl.col("AIRCRAFT_TYPE_mvt").alias("arr_type"),
        pl.col("TAXITIME_SEC_mvt").alias("arr_taxi_in"),
    ).sort("arr_inblock")
    d = dep.filter(pl.col("STAND_mvt").is_not_null()).sort("MVT_TIME_UTC_mvt")
    d = d.join_asof(arr, left_on="MVT_TIME_UTC_mvt", right_on="arr_inblock", by=["ADEP_mvt", "STAND_mvt"],
                    strategy="backward", check_sortedness=False)
    prev = d.select("ADEP_mvt", "STAND_mvt", pl.col("MVT_TIME_UTC_mvt").alias("prev_dep_tko")).sort("prev_dep_tko")
    # previous *other* departure from the same stand: shift within stand
    d = d.sort("ADEP_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt").with_columns(
        pl.col("MVT_TIME_UTC_mvt").shift(1).over(["ADEP_mvt", "STAND_mvt"]).alias("prev_dep_tko")
    )
    del prev
    sec = lambda a, b: (pl.col(a) - pl.col(b)).dt.total_seconds().cast(pl.Float64)  # noqa: E731
    d = d.with_columns(
        sec("MVT_TIME_UTC_mvt", "arr_inblock").alias("ta_gap"),
        sec("SCHED_TIME_UTC_mvt", "arr_inblock").alias("ta_sched_slack"),
        sec("AOBT_3_flt", "arr_inblock").alias("ta_aobt3_minus_inblock"),
        sec("arr_inblock", "arr_sched").alias("arr_inblock_delay"),
        (pl.col("arr_airline") == pl.col("airline")).cast(pl.Int8).alias("ta_same_airline"),
        (pl.col("arr_type") == pl.col("AIRCRAFT_TYPE_mvt")).cast(pl.Int8).alias("ta_same_type"),
        ((pl.col("prev_dep_tko").is_null()) | (pl.col("prev_dep_tko") < pl.col("arr_inblock"))).cast(pl.Int8)
        .alias("ta_clean"),
    )
    d = d.with_columns(
        (
            (pl.col("ta_clean") == 1) & (pl.col("ta_same_airline") == 1) & (pl.col("ta_gap") < 36 * 3600)
        ).cast(pl.Int8).alias("ta_linked")
    )
    return d


def main() -> None:
    d = build()
    feats = ["ta_gap", "ta_sched_slack", "ta_aobt3_minus_inblock", "arr_inblock_delay", "ta_same_airline",
             "ta_same_type", "ta_clean", "ta_linked", "arr_taxi_in"]
    d.select(["MVT_ID_mvt"] + feats).write_parquet(os.path.join(INTERIM, "oar_turn.parquet"))

    print("## A. リンク率（スタンド既知の DEP のうち）\n")
    t = d.group_by("split", "ADEP_mvt").agg(
        pl.len().alias("n"), pl.col("arr_inblock").is_not_null().mean().alias("has_prev_arr"),
        pl.col("ta_clean").mean().alias("clean"), pl.col("ta_same_airline").mean().alias("same_airline"),
        pl.col("ta_linked").mean().alias("linked"),
        pl.col("ta_same_type").filter(pl.col("ta_linked") == 1).mean().alias("same_type|linked"),
        pl.col("ta_gap").filter(pl.col("ta_linked") == 1).median().alias("ta_gap_med_s"),
    ).sort("split", "ADEP_mvt", descending=[True, False])
    print(md(t, "{:.3f}"))

    tr = d.filter(pl.col("split") == "train")
    L = tr.filter(pl.col("ta_linked") == 1)
    print("\n## B. リンクの妥当性: 物理制約 y ≤ ta_gap（= MVT − 到着インブロック）の成立率\n")
    v = L.group_by("clock_src").agg(pl.len().alias("n"), (pl.col("y") <= pl.col("ta_gap") + 60).mean().alias("y<=gap+60"),
                                    (pl.col("y") <= pl.col("ta_gap")).mean().alias("y<=gap")).sort("n", descending=True)
    print(md(v, "{:.4f}"))

    print("\n## B2. 「直前インブロック」は自機の到着とは限らない: 次の機体が同スタンドに入った場合の下限 y ≥ ta_gap\n")
    print("ta_gap が短い（< 30 分）とき、その到着は自機の折返し元ではなく、自機の離脱後にスタンドへ入った次の機体。"
          "その場合 BLOCK_dep ≤ inblock_next なので y ≥ ta_gap（下限）が成り立つ。\n")
    g = tr.with_columns(pl.col("ta_gap").truediv(60).cut([5, 10, 15, 20, 25, 30, 40, 60, 120]).alias("gap_min")).group_by("gap_min").agg(
        pl.len().alias("n"), (pl.col("y") >= pl.col("ta_gap")).mean().alias("y>=gap（下限成立）"),
        (pl.col("ta_aobt3_minus_inblock") < 0).mean().alias("inblock>AOBT_3"),
        pl.col("ta_same_airline").mean().alias("same_airline"), pl.col("y").median().alias("y_med"),
        (pl.col("clock_src") == "sched").mean().mul(100).alias("sched%")).sort("gap_min")
    print(md(g, "{:.3f}"))
    lb = tr.filter(pl.col("ta_aobt3_minus_inblock") < 0)
    print(f"\n下限フラグ（同スタンドのインブロックが AOBT_3 より後）: 学習 {lb.height:,} 行（{lb.height / tr.height:.2%}）、"
          f"ranking {d.filter((pl.col('split') == 'rank') & (pl.col('ta_aobt3_minus_inblock') < 0)).height:,} 行。"
          f"下限成立率 y ≥ ta_gap = {(lb['y'] >= lb['ta_gap']).mean():.4f}\n")
    print(md(lb.group_by("clock_src").agg(pl.len().alias("n"), (pl.col("y") >= pl.col("ta_gap")).mean().alias("y>=lb"),
                                         pl.col("y").median().alias("y_med"), pl.col("ta_gap").median().alias("lb_med"))
             .sort("n", descending=True), "{:.3f}"))

    print("\n## C. ターン余裕（SCHED − 直前インブロック）と y・出所（学習, linked。注: slack<0 の行の「直前インブロック」は多くが次の機体）\n")
    L = L.with_columns(pl.col("ta_sched_slack").truediv(60).cut([-60, -15, 0, 15, 30, 45, 60, 90, 120, 240]).alias("slack_min"))
    c = L.group_by("slack_min").agg(
        pl.len().alias("n"), pl.col("y").median().alias("y_med"), pl.col("y").mean().alias("y_mean"),
        (pl.col("clock_src") == "sched").mean().mul(100).alias("sched%"),
        (pl.col("clock_src") == "aobt3").mean().mul(100).alias("aobt3%"),
        (pl.col("clock_src") == "genuine").mean().mul(100).alias("genuine%"),
        pl.col("ta_gap").median().alias("ta_gap_med"),
    ).sort("slack_min")
    print(md(c, "{:,.1f}"))

    print("\n## E. 到着遅延の伝搬: 到着インブロック遅延 × y 残差（学習, linked, 空港×スタンド中央値からの差）\n")
    ref = tr.group_by("ADEP_mvt", "STAND_mvt").agg(pl.col("y").clip(0, 3600).median().alias("ref"))
    L = L.join(ref, on=["ADEP_mvt", "STAND_mvt"], how="left").with_columns((pl.col("y").clip(0, 3600) - pl.col("ref")).alias("r"))
    f = L.with_columns(pl.col("arr_inblock_delay").truediv(60).cut([-15, 0, 15, 30, 60, 120]).alias("arr_delay_min")).group_by(
        "arr_delay_min").agg(pl.len().alias("n"), pl.col("r").mean().alias("r_mean"), pl.col("r").median().alias("r_med"),
                             (pl.col("clock_src") == "sched").mean().mul(100).alias("sched%"),
                             (pl.col("clock_src") == "aobt3").mean().mul(100).alias("aobt3%")).sort("arr_delay_min")
    print(md(f, "{:,.1f}"))
    r = L.drop_nulls(["r", "ta_sched_slack"])
    print(f"\ncorr(r, ta_sched_slack) = {np.corrcoef(r['r'], r['ta_sched_slack'].clip(-7200, 14400))[0, 1]:.4f}; "
          f"corr(r, arr_inblock_delay) = {np.corrcoef(r['r'], r['arr_inblock_delay'].fill_null(0).clip(-3600, 14400))[0, 1]:.4f}\n")


if __name__ == "__main__":
    main()
