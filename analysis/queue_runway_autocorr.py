"""queue_runway: where does the strong same-runway residual autocorrelation come from?

Variance decomposition of r = y - median(y | airport, stand, runway) into airport-day, airport-day-hour and
within-hour parts; the same for the observable proxy g = MVT - AOBT_3 and for the reporting gap
d = y - g = AOBT_3 - BLOCK. Then: how much of the (hidden) hour-level mean of r is recoverable from hour-level
means of observable quantities.

Run: POLARS_MAX_THREADS=2 .venv/bin/python analysis/queue_runway_autocorr.py
"""

from __future__ import annotations

import os

import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = os.path.join(ROOT, "data", "interim", "queue_runway")
Y = "TAXITIME_SEC_mvt"


def md(df: pl.DataFrame) -> str:
    cols = df.columns
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.iter_rows():
        lines.append("| " + " | ".join(f"{v:,.3f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def main() -> None:
    cols = ["MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt", Y, "is_rank", "clock_src",
            "mvt_minus_aobt3", "mvt_minus_sched", "arr_taxi_in_mean_30m", "hour_local"]
    df = pl.read_parquet(os.path.join(C, "base.parquet"), columns=cols)
    new = pl.read_parquet(os.path.join(C, "new.parquet"), columns=["MVT_ID_mvt", "nbr_seq_mvt_aobt_pm5",
                                                                    "sched_backlog_60m", "hw_mean_next5"])
    df = df.join(new, on="MVT_ID_mvt", how="left")
    tr = df.filter((pl.col("is_rank") == 0) & pl.col(Y).is_between(30, 7200))
    ref = tr.group_by("ADEP_mvt", "STAND_mvt", "RUNWAY_mvt").agg(
        pl.col(Y).median().alias("ref_med"), pl.col("mvt_minus_aobt3").median().alias("ref_g"), pl.len().alias("n")
    )
    tr = tr.join(ref, on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]).filter(pl.col("n") >= 30)
    tr = tr.filter(pl.col("mvt_minus_aobt3").is_between(0, 7200))
    tr = tr.with_columns(
        (pl.col(Y) - pl.col("ref_med")).alias("r"),
        (pl.col("mvt_minus_aobt3") - pl.col("ref_g")).alias("rg"),
        (pl.col(Y) - pl.col("mvt_minus_aobt3")).alias("d"),
        pl.col("MVT_TIME_UTC_mvt").dt.date().alias("day"),
        pl.col("MVT_TIME_UTC_mvt").dt.truncate("1h").alias("hr"),
    )
    out = []
    for sub_name, sub in (("phys(genuine+aobt3)", tr.filter(pl.col("clock_src").is_in(["genuine", "aobt3"]))),
                          ("genuine only", tr.filter(pl.col("clock_src") == "genuine")),
                          ("all clean", tr)):
        for v in ("r", "rg", "d"):
            z = sub.with_columns(
                pl.col(v).mean().over("ADEP_mvt").alias("m_a"),
                pl.col(v).mean().over("ADEP_mvt", "day").alias("m_d"),
                pl.col(v).mean().over("ADEP_mvt", "hr").alias("m_h"),
                pl.col(v).mean().over("ADEP_mvt", "RUNWAY_mvt", "hr").alias("m_rh"),
            )
            tot = ((z[v] - z["m_a"]) ** 2).sum()
            out.append({
                "subset": sub_name, "var": v, "sd_within_apt": float(np.sqrt(tot / z.height)),
                "share_day": float(((z["m_d"] - z["m_a"]) ** 2).sum() / tot),
                "share_hour_in_day": float(((z["m_h"] - z["m_d"]) ** 2).sum() / tot),
                "share_rwy_hour_extra": float(((z["m_rh"] - z["m_h"]) ** 2).sum() / tot),
                "share_within_rwy_hour": float(((z[v] - z["m_rh"]) ** 2).sum() / tot),
            })
    print("## variance decomposition (share of within-airport variance)")
    print(md(pl.DataFrame(out)))

    # correlation of d with itself along the sequence (reporting offset persistence)
    ph = tr.filter(pl.col("clock_src").is_in(["genuine", "aobt3"])).sort("ADEP_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt")
    ph = ph.with_columns(
        pl.col("r").shift(1).over("ADEP_mvt", "RUNWAY_mvt").alias("r_prev"),
        pl.col("rg").shift(1).over("ADEP_mvt", "RUNWAY_mvt").alias("rg_prev"),
        pl.col("d").shift(1).over("ADEP_mvt", "RUNWAY_mvt").alias("d_prev"),
    )
    rows = []
    for apt in sorted(ph["ADEP_mvt"].unique().to_list()):
        z = ph.filter(pl.col("ADEP_mvt") == apt)
        rows.append({
            "apt": apt,
            "corr(r,r_prev)": z.select(pl.corr("r", "r_prev")).item(),
            "corr(rg,rg_prev)": z.select(pl.corr("rg", "rg_prev")).item(),
            "corr(d,d_prev)": z.select(pl.corr("d", "d_prev")).item(),
            "corr(r,rg)": z.select(pl.corr("r", "rg")).item(),
            "sd_r": z["r"].std(), "sd_rg": z["rg"].std(), "sd_d": z["d"].std(),
            "med_d": z["d"].median(),
        })
    print("\n## lag-1 autocorrelation on the same runway (phys rows, 2025)")
    print(md(pl.DataFrame(rows)))

    # hour-level: how much of the hour-mean of r do hour-means of observables recover?
    h = ph.group_by("ADEP_mvt", "hr").agg(
        pl.len().alias("n"), pl.col("r").mean().alias("r_h"), pl.col("rg").mean().alias("rg_h"),
        pl.col("d").mean().alias("d_h"), pl.col("arr_taxi_in_mean_30m").mean().alias("ti_h"),
        pl.col("sched_backlog_60m").mean().alias("bl_h"),
    ).filter(pl.col("n") >= 5)
    rows = []
    for apt in sorted(h["ADEP_mvt"].unique().to_list()):
        z = h.filter(pl.col("ADEP_mvt") == apt).drop_nulls()
        X = np.column_stack([np.ones(z.height), z["rg_h"], z["ti_h"], z["bl_h"]])
        yv = z["r_h"].to_numpy()
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        res = yv - X @ beta
        rows.append({"apt": apt, "hours": z.height, "sd_r_h": float(yv.std()),
                     "R2_obs(rg_h,taxiin_h,backlog_h)": float(1 - res.var() / yv.var()),
                     "corr(r_h,d_h)": float(np.corrcoef(yv, z["d_h"])[0, 1]),
                     "corr(r_h,rg_h)": float(np.corrcoef(yv, z["rg_h"])[0, 1])})
    print("\n## hour-level means (airport x clock hour, >=5 phys deps)")
    print(md(pl.DataFrame(rows)))


if __name__ == "__main__":
    main()
