"""operator_aircraft_route 07: the FLIGHT_mvt string format as a reporting-system fingerprint (NM-unmatched rows).

For NM-unmatched departures the only candidate clock is SCHED, so the posterior-mean prediction is
    pred = p_sched * (MVT - SCHED) + (1 - p_sched) * g
Here p_sched is a smoothed group rate fit on the other months (leave-one-month-out over 2025) and g the group's
median genuine taxi time. We compare groupings with and without the FLIGHT_mvt format fingerprint.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from operator_aircraft_route_common import flight_format, load_dep, md


def main() -> None:
    d = load_dep(["ADEP_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "FLIGHT_ID_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
                  "split", "y", "clock_src", "ym", "airline", "AIRCRAFT_TYPE_mvt"])
    d = d.filter(pl.col("FLIGHT_ID_mvt").is_null()).with_columns(
        flight_format().alias("fmt"),
        ((pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64)).alias("mms"),
        pl.col("AIRCRAFT_TYPE_mvt").is_null().alias("type_null"),
    ).with_columns(pl.col("mms").truediv(3600).cut([1, 2, 6]).cast(pl.String).alias("mmsb"),
                   (pl.col("ADEP_mvt") == "LIRF").alias("lirf"))
    tr = d.filter(pl.col("split") == "train")
    rk = d.filter(pl.col("split") == "rank")

    print("## A. NM 未マッチ行の FLIGHT_mvt 書式（学習 / ranking）\n")
    a = d.group_by("lirf", "fmt").agg(
        (pl.col("split") == "train").sum().alias("n_train"), (pl.col("split") == "rank").sum().alias("n_rank"),
        (pl.col("clock_src") == "sched").mean().mul(100).alias("sched%"),
        (pl.col("clock_src") == "rollover_24h").sum().alias("rollover"),
        pl.col("y").median().alias("y_med"), pl.col("y").mean().alias("y_mean"), (pl.col("y") > 7200).sum().alias("n_y>2h"),
        pl.col("mms").median().alias("mms_med"), pl.col("type_null").mean().alias("type_null"),
    ).sort("lirf", "n_train", descending=[True, True])
    print(md(a, "{:,.1f}"))

    print("\n## B. 事後平均ルーティングの RMSE（NM 未マッチ行, 12 か月の leave-one-month-out）\n")
    groupings = {
        "空港": ["ADEP_mvt"],
        "空港×書式": ["ADEP_mvt", "fmt"],
        "空港×遅延帯": ["ADEP_mvt", "mmsb"],
        "空港×書式×遅延帯": ["ADEP_mvt", "fmt", "mmsb"],
        "空港×航空会社×遅延帯": ["ADEP_mvt", "airline", "mmsb"],
    }
    rows = []
    preds = {}
    for name, keys in groupings.items():
        out = []
        for m in tr["ym"].unique().sort().to_list():
            fit, ev = tr.filter(pl.col("ym") != m), tr.filter(pl.col("ym") == m)
            prior = fit.group_by("ADEP_mvt").agg((pl.col("clock_src") == "sched").mean().alias("p0"),
                                                 pl.col("y").filter(pl.col("clock_src") == "genuine").median().alias("g0"))
            gr = fit.group_by(keys).agg((pl.col("clock_src") == "sched").sum().alias("s"), pl.len().alias("n"),
                                        pl.col("y").filter(pl.col("clock_src") == "genuine").median().alias("g"))
            e = ev.join(prior, on="ADEP_mvt", how="left").join(gr, on=keys, how="left", nulls_equal=True)
            e = e.with_columns(((pl.col("s").fill_null(0) + 10 * pl.col("p0").fill_null(0.1)) / (pl.col("n").fill_null(0) + 10)).alias("p"),
                               pl.coalesce("g", "g0", pl.lit(1000.0)).alias("gg"))
            e = e.with_columns((pl.col("p") * pl.col("mms") + (1 - pl.col("p")) * pl.col("gg")).alias("pred"))
            out.append(e.select("ADEP_mvt", "ym", "y", "pred", "clock_src", "fmt"))
        o = pl.concat(out)
        preds[name] = o
        for scope, f in (("全空港", pl.lit(True)), ("LIRF", pl.col("ADEP_mvt") == "LIRF"), ("LIRF 以外", pl.col("ADEP_mvt") != "LIRF")):
            s = o.filter(f)
            rows.append((name, scope, s.height, float(np.sqrt(((s["y"] - s["pred"]) ** 2).mean())),
                         float(np.sqrt(((s.filter(pl.col("ym") % 100 == 1)["y"] - s.filter(pl.col("ym") % 100 == 1)["pred"]) ** 2).mean())),
                         float(np.sqrt(((s.filter(pl.col("ym") % 100 == 7)["y"] - s.filter(pl.col("ym") % 100 == 7)["pred"]) ** 2).mean()))))
    print(md(pl.DataFrame(rows, schema=["grouping", "scope", "n", "rmse_lomo", "rmse_jan", "rmse_jul"], orient="row"), "{:,.1f}"))
    # translate SSE change into full-set RMSE change (2,085,047 labelled departures, v7 OOF RMSE ~305)
    base = preds["空港×遅延帯"]
    best = preds["空港×書式×遅延帯"]
    dsse = float(((best["y"] - best["pred"]) ** 2).sum() - ((base["y"] - base["pred"]) ** 2).sum())
    N, R = 2_085_047, 305.0
    print(f"\n空港×遅延帯 → +書式 の SSE 差 = {dsse:.3e}（2025 全 DEP 換算の RMSE 差 ≈ "
          f"{np.sqrt(max(R**2 + dsse / N, 0)) - R:+.2f} 秒, 基準 RMSE {R}）\n")

    print("## C. ranking 側の該当行\n")
    print(md(rk.group_by("ADEP_mvt", "fmt").agg(pl.len().alias("n"), pl.col("mms").median().alias("mms_med"),
                                                (pl.col("mms") > 7200).sum().alias("n_mms>2h"))
             .filter(pl.col("ADEP_mvt") == "LIRF").sort("n", descending=True), "{:,.0f}"))


if __name__ == "__main__":
    main()
