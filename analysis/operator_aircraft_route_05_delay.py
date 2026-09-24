"""operator_aircraft_route 05: departure delay vs taxi-out, by carrier type.

delay proxy (leak-free): d_nm = AOBT_3 - EOBT_1 (NM actual vs filed off-block), d_mvt = MVT - SCHED - 900 s.
r = clip(y,0,3600) - median(airport x stand).
"""

from __future__ import annotations

import polars as pl
from operator_aircraft_route_01_airline import HUB
from operator_aircraft_route_common import load_dep, md


def main() -> None:
    d = load_dep(["ADEP_mvt", "STAND_mvt", "airline", "MARKET_SEGMENT_flt", "split", "y", "clock_src", "AOBT_3_flt",
                  "EOBT_1_flt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]).filter(pl.col("split") == "train")
    sec = lambda a, b: (pl.col(a) - pl.col(b)).dt.total_seconds().cast(pl.Float64)  # noqa: E731
    d = d.with_columns(sec("AOBT_3_flt", "EOBT_1_flt").alias("d_nm"), sec("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("mms"))
    ref = d.group_by("ADEP_mvt", "STAND_mvt").agg(pl.col("y").clip(0, 3600).median().alias("ref"))
    d = d.join(ref, on=["ADEP_mvt", "STAND_mvt"], how="left").with_columns((pl.col("y").clip(0, 3600) - pl.col("ref")).alias("r"))
    d = d.with_columns(
        pl.struct("ADEP_mvt", "airline").map_elements(lambda s: HUB.get(s["ADEP_mvt"]) == s["airline"], return_dtype=pl.Boolean)
        .alias("hub"),
        pl.col("d_nm").truediv(60).cut([-5, 5, 15, 30, 60, 120]).alias("delay_min"),
    )
    print("## A. NM 遅延（AOBT_3 − EOBT_1）× 残差 r と出所（学習）\n")
    a = d.group_by("delay_min").agg(pl.len().alias("n"), pl.col("r").mean().alias("r_mean"), pl.col("r").median().alias("r_med"),
                                    (pl.col("clock_src") == "sched").mean().mul(100).alias("sched%"),
                                    (pl.col("clock_src") == "aobt3").mean().mul(100).alias("aobt3%"),
                                    (pl.col("clock_src") == "eobt1").mean().mul(100).alias("eobt1%")).sort("delay_min")
    print(md(a, "{:,.1f}"))
    print("\n## B. 同、ハブキャリア / その他 × 市場区分（r の平均, 秒）\n")
    b = d.filter(pl.col("MARKET_SEGMENT_flt").is_in(["Mainline", "Lowcost", "Regional", "Cargo"])).group_by(
        "hub", "MARKET_SEGMENT_flt", "delay_min").agg(pl.col("r").mean().alias("r_mean"), pl.len().alias("n"))
    piv = b.with_columns(pl.col("delay_min").cast(pl.String)).pivot(on="delay_min", index=["hub", "MARKET_SEGMENT_flt"],
                                                                     values="r_mean").sort("hub", "MARKET_SEGMENT_flt")
    order = ["(-inf, -5]", "(-5, 5]", "(5, 15]", "(15, 30]", "(30, 60]", "(60, 120]", "(120, inf]"]
    piv = piv.select(["hub", "MARKET_SEGMENT_flt"] + [o for o in order if o in piv.columns])
    print(md(piv, "{:,.0f}"))
    print("\n## C. 主要航空会社ごとの 遅延 30 分超 − 遅延 ±5 分 の r 差（n≥300 ずつ）\n")
    c = d.filter(pl.col("delay_min").cast(pl.String).is_in(["(-5, 5]", "(30, 60]", "(60, 120]", "(120, inf]"])).with_columns(
        (pl.col("d_nm") > 1800).alias("late")).group_by("airline", "late").agg(pl.col("r").mean().alias("r"), pl.len().alias("n"))
    c = c.pivot(on="late", index="airline", values=["r", "n"]).filter((pl.col("n_true") >= 300) & (pl.col("n_false") >= 300))
    c = c.with_columns((pl.col("r_true") - pl.col("r_false")).alias("late_minus_ontime")).sort("late_minus_ontime")
    print(md(pl.concat([c.head(8), c.tail(8)]).unique(maintain_order=True), "{:,.1f}"))


if __name__ == "__main__":
    main()
