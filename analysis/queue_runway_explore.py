"""queue_runway: descriptive analysis (binned effects, autocorrelation, runway configuration).

Run after queue_runway_prep.py:  POLARS_MAX_THREADS=2 .venv/bin/python analysis/queue_runway_explore.py
Descriptive only: the reference median (airport x stand x runway) is fitted in-sample on 2025. The honest,
out-of-fold effect sizes are in queue_runway_eval.py.
"""

from __future__ import annotations

import os

import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = os.path.join(ROOT, "data", "interim", "queue_runway")
Y = "TAXITIME_SEC_mvt"


BASE_COLS = [
    "MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt", Y, "is_rank", "clock_src", "nm_matched",
    "mvt_minus_aobt3", "rwy_headway_prev_s", "rwy_headway_next_s", "dep_tko_prev_15m", "dep_tko_next_15m",
    "rwy_tko_prev_15m", "q_in_system_at_pushback", "q_in_system_at_tko", "q_in_system_at_pushback_rwy",
    "nbr_mvt_minus_aobt_mean_30m", "arr_taxi_in_mean_30m", "arr_land_prev_15m", "mvt_minus_sched",
]


def load() -> pl.DataFrame:
    base = pl.read_parquet(os.path.join(C, "base.parquet"), columns=BASE_COLS)
    new = pl.read_parquet(os.path.join(C, "new.parquet"))
    df = base.join(new, on="MVT_ID_mvt", how="left")
    return df


def eta2(x: np.ndarray, r: np.ndarray, q: int = 20) -> float:
    """Share of var(r) explained by quantile-bin means of x (missing = its own bin)."""
    ok = np.isfinite(r)
    x, r = x[ok], r[ok]
    fin = np.isfinite(x)
    edges = np.unique(np.nanquantile(x[fin], np.linspace(0, 1, q + 1))) if fin.sum() > 100 else np.array([0.0])
    b = np.where(fin, np.searchsorted(edges, x, side="right"), -1)
    _, inv, cnt = np.unique(b, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    means = sums / cnt
    return float(1 - np.sum((r - means[inv]) ** 2) / np.sum((r - r.mean()) ** 2))


def table(df: pl.DataFrame, col: str, edges: list[float], r: str = "r") -> pl.DataFrame:
    lab = pl.col(col).cut(edges) if edges else pl.col(col).cast(pl.Utf8)
    return (
        df.group_by(lab.alias("bin"))
        .agg(pl.len().alias("n"), pl.col(r).mean().round(1).alias("r_mean"), pl.col(Y).median().alias("y_med"))
        .sort("bin")
    )


def md(df: pl.DataFrame) -> str:
    cols = df.columns
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.iter_rows():
        lines.append("| " + " | ".join(f"{v:,.3f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def main() -> None:
    df = load()
    tr = df.filter((pl.col("is_rank") == 0) & pl.col(Y).is_between(30, 7200))
    ref = tr.group_by("ADEP_mvt", "STAND_mvt", "RUNWAY_mvt").agg(pl.col(Y).median().alias("ref_med"), pl.len().alias("ref_n"))
    tr = tr.join(ref, on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], how="left").filter(pl.col("ref_n") >= 30)
    tr = tr.with_columns((pl.col(Y) - pl.col("ref_med")).alias("r"))
    # "physical" subset: block time is a real off-block (genuine or copy of NM actual)
    phys = tr.filter(pl.col("clock_src").is_in(["genuine", "aobt3"]))
    print(f"rows clean={tr.height:,} phys={phys.height:,}  sd(r) clean={tr['r'].std():.1f} phys={phys['r'].std():.1f}")

    # ---- 1. variance explained, existing vs new ----
    feats_exist = [
        "mvt_minus_aobt3", "rwy_headway_prev_s", "rwy_headway_next_s", "dep_tko_prev_15m", "dep_tko_next_15m",
        "rwy_tko_prev_15m", "q_in_system_at_pushback", "q_in_system_at_tko", "q_in_system_at_pushback_rwy",
        "nbr_mvt_minus_aobt_mean_30m", "arr_taxi_in_mean_30m", "arr_land_prev_15m", "mvt_minus_sched",
    ]
    feats_new = [
        "hw_excess", "burst_pos", "burst_len", "burst_elapsed", "burst_remaining", "burst_pos_frac", "hw_mean_prev5",
        "hw_mean_next5", "hw_min_prev3", "arr_same_rwy_since_prevdep", "arr_same_rwy_prev_s", "arr_same_rwy_next_s",
        "arr_same_rwy_pm10m", "nbr_seq_mvt_aobt_prev5", "nbr_seq_mvt_aobt_next5", "nbr_seq_mvt_aobt_pm5",
        "nbr_seq_mvt_sched_pm5", "overtaken_by_aobt", "overtook_aobt", "rank_shift_aobt", "rwy_share_pm30",
        "cfg_n_dep_rwy", "cfg_n_arr_rwy", "cfg_since_change_bins", "cfg_until_change_bins", "sched_backlog_60m",
        "sched_demand_next30m", "sched_demand_prev30m", "arr_taxiing_at_tko", "arr_taxiing_at_aobt",
        "arr_taxi_in_landed_20m",
    ]
    rows = []
    for f in feats_exist + feats_new:
        for name, sub in (("phys", phys), ("clean", tr)):
            pass
        x = phys[f].cast(pl.Float64).to_numpy()
        rr = phys["r"].to_numpy()
        # within-airport eta2 averaged by rows (features have airport-specific scales)
        e_ap = []
        w = []
        for apt in phys["ADEP_mvt"].unique().to_list():
            m = (phys["ADEP_mvt"] == apt).to_numpy()
            e_ap.append(eta2(x[m], rr[m]))
            w.append(m.sum())
        nm0 = (phys["nm_matched"] == 0).to_numpy()
        rows.append(
            {
                "feature": f,
                "group": "existing" if f in feats_exist else "new",
                "eta2_phys_within_apt": float(np.average(e_ap, weights=w)),
                "null_rate_train": float(np.mean(~np.isfinite(x))),
                "null_rate_rank": float(df.filter(pl.col("is_rank") == 1)[f].is_null().mean()),
            }
        )
    et = pl.DataFrame(rows).sort("eta2_phys_within_apt", descending=True)
    print(md(et))

    # ---- 2. residual autocorrelation along the runway sequence ----
    print("\n## autocorr of r along same-runway take-off sequence (phys rows, consecutive both phys)")
    s = tr.sort("ADEP_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt").with_columns(
        pl.col("r").shift(1).over("ADEP_mvt", "RUNWAY_mvt").alias("r_prev"),
        pl.col("r").shift(-1).over("ADEP_mvt", "RUNWAY_mvt").alias("r_next"),
        pl.col("r").rolling_mean(11, center=True, min_samples=6).over("ADEP_mvt", "RUNWAY_mvt").alias("r_pm5_incl"),
        (pl.col("mvt_minus_aobt3").shift(1).over("ADEP_mvt", "RUNWAY_mvt")).alias("ma_prev"),
        pl.col("rwy_headway_prev_s").alias("gap"),
    ).with_columns(((pl.col("r_pm5_incl") * 11 - pl.col("r")) / 10).alias("r_pm5"))
    s = s.filter(pl.col("clock_src").is_in(["genuine", "aobt3"]))
    ac = []
    for apt in sorted(s["ADEP_mvt"].unique().to_list()) + ["ALL"]:
        z = s if apt == "ALL" else s.filter(pl.col("ADEP_mvt") == apt)
        z = z.filter(pl.col("r_prev").is_not_null() & pl.col("r_next").is_not_null() & pl.col("r_pm5").is_not_null())
        ac.append(
            {
                "apt": apt,
                "n": z.height,
                "corr_prev": z.select(pl.corr("r", "r_prev")).item(),
                "corr_next": z.select(pl.corr("r", "r_next")).item(),
                "corr_pm5mean": z.select(pl.corr("r", "r_pm5")).item(),
                "R2_oracle_pm5": z.select(pl.corr("r", "r_pm5")).item() ** 2,
                "corr_proxy_pm5": z.select(pl.corr("r", "nbr_seq_mvt_aobt_pm5")).item(),
                "corr_proxy_past30m": z.select(pl.corr("r", "nbr_mvt_minus_aobt_mean_30m")).item(),
                "corr_rpm5_vs_proxy": z.select(pl.corr("r_pm5", "nbr_seq_mvt_aobt_pm5")).item(),
            }
        )
    print(md(pl.DataFrame(ac)))
    # autocorr by time gap to previous dep
    z = s.filter(pl.col("r_prev").is_not_null())
    g = (
        z.group_by(pl.col("gap").cut([60, 90, 120, 180, 300, 600, 1800]).alias("gap_bin"))
        .agg(pl.len().alias("n"), pl.corr("r", "r_prev").alias("corr_prev"))
        .sort("gap_bin")
    )
    print(md(g))

    # ---- 3. binned effects of the new features (phys rows, within-airport-demeaned r) ----
    ph = phys.with_columns((pl.col("r") - pl.col("r").mean().over("ADEP_mvt")).alias("rd"))
    for col, edges in (
        ("burst_pos", [1, 2, 3, 5, 8, 12, 20]),
        ("burst_len", [1, 2, 4, 8, 15, 30]),
        ("burst_remaining", [0, 60, 180, 420, 900]),
        ("hw_excess", [-30, 0, 15, 45, 90, 180, 600]),
        ("hw_mean_next5", [80, 100, 120, 150, 200, 300, 600]),
        ("nbr_seq_mvt_aobt_next5", [500, 700, 900, 1100, 1300, 1600, 2000]),
        ("overtaken_by_aobt", [0, 1, 2, 4, 8]),
        ("overtook_aobt", [0, 1, 2, 4, 8]),
        ("sched_backlog_60m", [0, 2, 5, 10, 15, 20, 30]),
        ("arr_same_rwy_since_prevdep", [0, 1, 2, 4]),
        ("arr_taxiing_at_aobt", [0, 2, 4, 8, 12, 20]),
        ("cfg_since_change_bins", [0, 1, 2, 4, 8, 16]),
    ):
        print(f"\n### {col}")
        print(md(table(ph, col, edges, r="rd")))

    # ---- 4. configuration: within airport x stand x runway, does the arrival config matter? ----
    print("\n## runway configuration (arr set) effect within airport x runway x stand")
    cf = phys.filter(pl.col("cfg_config").is_not_null())
    cf = cf.with_columns(pl.col("r").mean().over("ADEP_mvt", "cfg_config").alias("m_cfg"))
    for apt in sorted(cf["ADEP_mvt"].unique().to_list()):
        z = cf.filter(pl.col("ADEP_mvt") == apt)
        # group by (runway, stand, arr-set) vs (runway, stand): extra variance explained
        z = z.with_columns(pl.col("cfg_config").str.split("|").list.get(1).alias("arrset"))
        g1 = z.with_columns(pl.col("r").mean().over("RUNWAY_mvt", "arrset").alias("m"))
        e = 1 - ((g1["r"] - g1["m"]) ** 2).sum() / ((z["r"] - z["r"].mean()) ** 2).sum()
        top = (
            z.group_by("RUNWAY_mvt", "arrset").agg(pl.len().alias("n"), pl.col("r").mean().round(1).alias("r_mean"))
            .filter(pl.col("n") > 2000).sort("RUNWAY_mvt", "n", descending=[False, True])
        )
        print(f"\n### {apt}: eta2(runway x arrival set | ref residual) = {e:.4f}")
        print(md(top.head(12)))


if __name__ == "__main__":
    main()
