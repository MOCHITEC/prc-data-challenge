"""operator_aircraft_route 03: how reproducible is a scheduled flight's taxi-out (and its clock source)?

* flight key = ADEP_mvt + FLIGHT_mvt (ranking coverage 78%).
* residual r = clip(y, 0, 3600) - median(y | airport x stand x local-hour-bucket), fit on the fit months only.
* out-of-time folds: jan / jul (fit on the other 11 months, as the repo's `fast` CV), and a strictly forward fold
  (fit Jan-Oct, evaluate Nov-Dec).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from operator_aircraft_route_01_airline import SRC, smoothed_logloss
from operator_aircraft_route_common import load_dep, md

CLIP = 3600.0


def base_ref(fit: pl.DataFrame) -> pl.DataFrame:
    return fit.group_by("ADEP_mvt", "STAND_mvt", "hb").agg(pl.col("yc").median().alias("ref"), pl.len().alias("ref_n"))


def attach_ref(df: pl.DataFrame, fit: pl.DataFrame) -> pl.DataFrame:
    ref = base_ref(fit)
    ap = fit.group_by("ADEP_mvt").agg(pl.col("yc").median().alias("ref_ap"))
    df = df.join(ref, on=["ADEP_mvt", "STAND_mvt", "hb"], how="left").join(ap, on="ADEP_mvt", how="left")
    return df.with_columns(
        pl.when(pl.col("ref_n") >= 10).then(pl.col("ref")).otherwise(pl.col("ref_ap")).alias("ref")
    ).with_columns((pl.col("yc") - pl.col("ref")).alias("r"))


def te(fit: pl.DataFrame, ev: pl.DataFrame, keys: list[str], col: str, k: float, name: str) -> pl.DataFrame:
    t = fit.group_by(keys).agg(pl.col(col).sum().alias("_s"), pl.len().alias(f"{name}_n"))
    ev = ev.join(t, on=keys, how="left", maintain_order="left")
    return ev.with_columns((pl.col("_s").fill_null(0) / (pl.col(f"{name}_n").fill_null(0) + k)).alias(name)).drop("_s")


def main() -> None:
    d = load_dep(
        ["MVT_ID_mvt", "ADEP_mvt", "FLIGHT_mvt", "STAND_mvt", "airline", "split", "y", "clock_src", "ym",
         "MVT_TIME_UTC_mvt", "FLIGHT_ID_mvt", "SCHED_TIME_UTC_mvt"]
    )
    d = d.with_columns(
        pl.col("STAND_mvt").fill_null("?"), pl.col("airline").fill_null("?"),
        pl.col("y").clip(0, CLIP).alias("yc"),
        (pl.col("SCHED_TIME_UTC_mvt").dt.hour() // 3).alias("hb"),
        pl.col("SCHED_TIME_UTC_mvt").dt.date().alias("day"),
    )
    tr = d.filter(pl.col("split") == "train")
    rk = d.filter(pl.col("split") == "rank")
    key = ["ADEP_mvt", "FLIGHT_mvt"]

    print("## A. 被覆: ranking DEP のうち 2025 学習に同じ (空港, FLIGHT_mvt) がある比率と、その学習件数\n")
    cnt = tr.group_by(key).len()
    r2 = rk.join(cnt, on=key, how="left").with_columns(pl.col("len").fill_null(0))
    cov = r2.group_by("ADEP_mvt").agg(
        pl.len().alias("rank_rows"),
        (pl.col("len") > 0).mean().alias("seen"),
        (pl.col("len") >= 20).mean().alias("n>=20"),
        (pl.col("len") >= 100).mean().alias("n>=100"),
        pl.col("len").median().alias("median_n"),
    ).sort("ADEP_mvt")
    allrow = r2.select(pl.lit("ALL").alias("ADEP_mvt"), pl.len().alias("rank_rows"), (pl.col("len") > 0).mean().alias("seen"),
                       (pl.col("len") >= 20).mean().alias("n>=20"), (pl.col("len") >= 100).mean().alias("n>=100"),
                       pl.col("len").median().cast(pl.Float64).alias("median_n"))
    print(md(pl.concat([cov, allrow], how="vertical_relaxed"), "{:.3f}"))
    # NM-unmatched coverage
    u = r2.filter(pl.col("FLIGHT_ID_mvt").is_null())
    print(f"\nNM 未マッチの ranking 行 {u.height:,} のうち既知の便: {(u['len'] > 0).mean():.3f}\n")

    # --- B. day-to-day reproducibility (in-sample, full year fit of the reference)
    print("## B. 同じ便の翌日以降の再現性（2025, 残差 r の相関）\n")
    t = attach_ref(tr, tr).sort("MVT_TIME_UTC_mvt")
    t = t.with_columns(
        pl.col("r").shift(1).over(key).alias("r_prev"),
        pl.col("day").shift(1).over(key).alias("day_prev"),
        pl.col("clock_src").shift(1).over(key).alias("src_prev"),
    ).with_columns((pl.col("day") - pl.col("day_prev")).dt.total_days().alias("gap_d"))
    # comparator: previous row at the same airport x stand x hour bucket (a different flight)
    t = t.with_columns(pl.col("r").shift(1).over(["ADEP_mvt", "STAND_mvt", "hb"]).alias("r_prev_stand"))
    rows = []
    for lo, hi, lab in ((1, 1, "翌日"), (2, 7, "2-7 日後"), (8, 60, "8-60 日後")):
        s = t.filter(pl.col("gap_d").is_between(lo, hi) & pl.col("r_prev").is_not_null())
        rows.append((lab, s.height, float(np.corrcoef(s["r"], s["r_prev"])[0, 1]),
                     float((s["clock_src"] == s["src_prev"]).mean()),
                     float(s.filter(pl.col("src_prev") == "sched")["clock_src"].eq("sched").mean()),
                     float(s.filter(pl.col("src_prev") != "sched")["clock_src"].eq("sched").mean())))
    s = t.filter(pl.col("r_prev_stand").is_not_null())
    rows.append(("比較: 同スタンド×時間帯の直前の別便", s.height, float(np.corrcoef(s["r"], s["r_prev_stand"])[0, 1]),
                 float("nan"), float("nan"), float("nan")))
    print(md(pl.DataFrame(rows, schema=["pair", "n", "corr_r", "same_src", "P(sched|prev sched)", "P(sched|prev not sched)"],
                          orient="row"), "{:.3f}"))

    # --- C. out-of-time target-encoding value on the residual
    print("\n## C. 月ブロック検証: 残差 r に対する target encoding の説明力（RMSE of r, 秒, clip 3600）\n")
    folds = {
        "jan": (tr.filter(pl.col("ym") % 100 != 1), tr.filter(pl.col("ym") % 100 == 1)),
        "jul": (tr.filter(pl.col("ym") % 100 != 7), tr.filter(pl.col("ym") % 100 == 7)),
        "fwd(Jan-Oct→Nov-Dec)": (tr.filter(pl.col("ym") % 100 <= 10), tr.filter(pl.col("ym") % 100 >= 11)),
    }
    res = []
    for fname, (fit, ev) in folds.items():
        fit = attach_ref(fit, fit)
        ev = attach_ref(ev, fit)
        ev = te(fit, ev, ["ADEP_mvt", "airline"], "r", 30, "te_al")
        fit2 = te(fit, fit, ["ADEP_mvt", "airline"], "r", 30, "te_al").with_columns((pl.col("r") - pl.col("te_al")).alias("r2"))
        ev = te(fit2, ev, key, "r2", 10, "te_fl")
        ev = te(fit, ev, key, "r", 10, "te_fl_only")
        ev = te(fit, ev, ["ADEP_mvt", "STAND_mvt", "airline"], "r", 20, "te_sa")
        y = ev["r"].to_numpy()
        cov = ev["te_fl_n"].fill_null(0).to_numpy() > 0
        for lab, p in (
            ("ref のみ（空港×スタンド×3h 中央値）", np.zeros_like(y)),
            ("+ 空港×航空会社 TE", ev["te_al"].to_numpy()),
            ("+ 空港×スタンド×航空会社 TE", ev["te_sa"].to_numpy()),
            ("+ 便 TE（単独）", ev["te_fl_only"].to_numpy()),
            ("+ 空港×航空会社 TE + 便 TE（残差に階層）", ev["te_al"].to_numpy() + ev["te_fl"].to_numpy()),
        ):
            e = (y - p) ** 2
            res.append((fname, lab, float(np.sqrt(e.mean())), float(np.sqrt(e[cov].mean())), float(cov.mean())))
    print(md(pl.DataFrame(res, schema=["fold", "predictor of r", "rmse_all", "rmse_covered", "coverage"], orient="row"),
             "{:.2f}"))

    # --- D. clock-source persistence by flight
    print("\n## D. 出所ラベルの便レベル持続性（月ブロック log loss, 7 クラス, α=20）\n")
    rows = []
    trs = tr.with_columns(pl.col("FLIGHT_mvt").fill_null("?"))
    for fname, (fit, ev) in {
        "jan": (trs.filter(pl.col("ym") % 100 != 1), trs.filter(pl.col("ym") % 100 == 1)),
        "jul": (trs.filter(pl.col("ym") % 100 != 7), trs.filter(pl.col("ym") % 100 == 7)),
        "fwd": (trs.filter(pl.col("ym") % 100 <= 10), trs.filter(pl.col("ym") % 100 >= 11)),
    }.items():
        for name, levels in {
            "airport>stand": [["ADEP_mvt"], ["ADEP_mvt", "STAND_mvt"]],
            "airport>airline": [["ADEP_mvt"], ["ADEP_mvt", "airline"]],
            "airport>airline>flight": [["ADEP_mvt"], ["ADEP_mvt", "airline"], key],
            "airport>stand>flight": [["ADEP_mvt"], ["ADEP_mvt", "STAND_mvt"], key],
        }.items():
            rows.append((fname, name, *smoothed_logloss(fit, ev, levels)))
    print(md(pl.DataFrame(rows, schema=["fold", "hierarchy", "logloss_all", "logloss_nm_unmatched", "brier_sched"],
                          orient="row"), "{:.4f}"))

    # overdispersion of per-flight sched rate vs binomial
    print("\n## E. 便ごとの SCHED コピー率の過分散（2025, 便あたり n≥30）\n")
    g = tr.group_by("ADEP_mvt", "FLIGHT_mvt").agg(pl.len().alias("n"), (pl.col("clock_src") == "sched").mean().alias("p"),
                                                   (pl.col("clock_src") == "aobt3").mean().alias("pa")).filter(pl.col("n") >= 30)
    rows = []
    for apt, s in g.group_by("ADEP_mvt"):
        n = s["n"].to_numpy(); p = s["p"].to_numpy(); pa = s["pa"].to_numpy()
        pbar = np.average(p, weights=n); pabar = np.average(pa, weights=n)
        obs = np.average((p - pbar) ** 2, weights=n); exp_ = np.average(pbar * (1 - pbar) / n, weights=n)
        obsa = np.average((pa - pabar) ** 2, weights=n); expa = np.average(pabar * (1 - pabar) / n, weights=n)
        rows.append((apt[0], len(n), pbar, obs / exp_, pabar, obsa / expa,
                     float((p > 0.5).mean())))
    print(md(pl.DataFrame(rows, schema=["airport", "flights", "sched_rate", "sched_var_ratio", "aobt3_rate",
                                        "aobt3_var_ratio", "share_flights_sched>50%"], orient="row").sort("airport"),
             "{:.3f}"))
    print("\nvar_ratio = 便間の率の分散 / 二項分布の期待分散（1 なら便ごとの差なし）\n")

    # --- F. LIRF: flight-level sched rate stability (H1 -> H2) and the high-sched flights
    print("## F. LIRF: 便ごとの SCHED コピー率の半期安定性（H1=1-6 月, H2=7-12 月, 各 n≥20）\n")
    L = tr.filter(pl.col("ADEP_mvt") == "LIRF")
    h1 = L.filter(pl.col("ym") % 100 <= 6).group_by("FLIGHT_mvt").agg(pl.len().alias("n1"), (pl.col("clock_src") == "sched").mean().alias("p1"))
    h2 = L.filter(pl.col("ym") % 100 >= 7).group_by("FLIGHT_mvt").agg(pl.len().alias("n2"), (pl.col("clock_src") == "sched").mean().alias("p2"))
    j = h1.join(h2, on="FLIGHT_mvt").filter((pl.col("n1") >= 20) & (pl.col("n2") >= 20))
    print(f"便数 {j.height}, corr(p_H1, p_H2) = {np.corrcoef(j['p1'], j['p2'])[0, 1]:.3f}\n")
    print(md(j.with_columns((pl.col("p1") > 0.5).alias("H1_sched>50%")).group_by("H1_sched>50%").agg(
        pl.len().alias("flights"), pl.col("p1").mean().alias("p_H1"), pl.col("p2").mean().alias("p_H2")), "{:.3f}"))
    g = L.group_by("FLIGHT_mvt").agg(pl.len().alias("n"), (pl.col("clock_src") == "sched").mean().alias("sched_rate"),
                                     pl.col("airline").mode().first().alias("airline"),
                                     pl.col("STAND_mvt").mode().first().alias("stand"),
                                     pl.col("y").mean().alias("y_mean")).filter(pl.col("n") >= 30)
    hi = g.filter(pl.col("sched_rate") > 0.5)
    nrk = rk.filter((pl.col("ADEP_mvt") == "LIRF") & pl.col("FLIGHT_mvt").is_in(hi["FLIGHT_mvt"].implode())).height
    print(f"\nsched>50% の便: {hi.height} / {g.height} 便（学習 {hi['n'].sum():,} 行）。ranking LIRF で該当便の行: {nrk:,}\n")
    print(md(hi.sort("n", descending=True).head(12), "{:,.2f}"))



if __name__ == "__main__":
    main()
