"""Read-only characterisation of LFPG's unmatched lane (2026-09-24).

Inputs (never written): data/work/feat_train.parquet, data/work/feat_rank.parquet,
experiments/oof_main_reparam-bc60017878.parquet + its _folds sidecar.

Notation: y = TAXITIME_SEC_mvt, ms = mvt_minus_sched = MVT - SCHED,
bs = BLOCK - SCHED (label-side only; BLOCK is null on all 344,841 ranking rows),
unmatched = routing definition (mvt_minus_aobt3 null -> model B).
Identity used throughout: y = MVT - BLOCK exactly, so bs = ms - y.

Run: POLARS_MAX_THREADS=2 .venv/bin/python docs/competitor_research_2026-09-21/scripts/lfpg_unmatched_lane.py
"""

from __future__ import annotations

import numpy as np
import polars as pl

BASE = "/home/user/prc-data-challenge/"
OOF = BASE + "experiments/oof_main_reparam-bc60017878.parquet"
FOLDS = BASE + "experiments/oof_main_reparam-bc60017878_folds.parquet"


def sec(a: str, b: str) -> pl.Expr:
    return (pl.col(a) - pl.col(b)).dt.total_seconds()


def load_train_slim() -> pl.DataFrame:
    cols = [
        "MVT_ID_mvt", "ADEP_mvt", "mvt_minus_aobt3", "TAXITIME_SEC_mvt", "mvt_minus_sched",
        "month", "hour_utc", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
        "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "FLIGHT_mvt", "ADES_mvt",
    ]
    return (
        pl.scan_parquet(BASE + "data/work/feat_train.parquet")
        .select(cols)
        .with_columns(
            pl.col("mvt_minus_aobt3").is_null().alias("unm"),
            sec("BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("bs"),
        )
        .collect()
    )


def load_oof(train: pl.DataFrame) -> pl.DataFrame:
    o = pl.read_parquet(OOF).drop_nulls("oof")
    f = pl.read_parquet(FOLDS).unique(subset=["MVT_ID_mvt"], keep="first")
    return (
        train.join(o, on="MVT_ID_mvt", how="inner")
        .join(f, on="MVT_ID_mvt", how="left")
        .with_columns((pl.col("oof") - pl.col("TAXITIME_SEC_mvt")).alias("resid"))
        .with_columns((pl.col("resid") ** 2).alias("sq"), pl.col("MVT_TIME_UTC_mvt").dt.date().alias("day"))
    )


def s1_per_airport(d: pl.DataFrame) -> None:
    print("\n== S1  per-airport unmatched OOF RMSE, all rows vs excluding y >= 20,000 ==")
    u = d.filter("unm")
    for a, g in sorted(u.group_by("ADEP_mvt"), key=lambda x: x[0]):
        y = g["TAXITIME_SEC_mvt"].to_numpy().astype(float)
        r = g["resid"].to_numpy()
        m = y < 20000
        print(f"{a[0]} n={len(y):5d} rmse={np.sqrt((r**2).mean()):9.1f} sse={(r**2).sum():.4e} | "
              f"excl y>=20k n={m.sum():5d} rmse={np.sqrt((r[m]**2).mean()):8.1f} (dropped {(~m).sum()})")


def s2_concentration(d: pl.DataFrame) -> None:
    print("\n== S2  SSE concentration inside each airport's unmatched lane ==")
    u = d.filter("unm")
    for a, g in sorted(u.group_by("ADEP_mvt"), key=lambda x: x[0]):
        s = np.sort(g["sq"].to_numpy())[::-1]
        tot = s.sum()
        drops = "  ".join(f"drop{k}={np.sqrt(s[k:].sum()/(len(s)-k)):.0f}" for k in (1, 3, 10, 50))
        print(f"{a[0]} n={len(s):5d} top1={100*s[0]/tot:5.1f}% top10={100*s[:10].sum()/tot:5.1f}% "
              f"top100={100*s[:100].sum()/tot:5.1f}% | {drops}")


def s3_edge(t: pl.DataFrame) -> None:
    print("\n== S3  the hard edge: P(unmatched | BLOCK-SCHED) around 3,600 s ==")
    for a, g in sorted(t.group_by("ADEP_mvt"), key=lambda x: x[0]):
        bs, un = g["bs"].to_numpy(), g["unm"].to_numpy()
        lo, hi = (bs >= 3300) & (bs < 3600), (bs >= 3600) & (bs < 3900)
        pl_, ph = un[lo].mean(), un[hi].mean()
        print(f"{a[0]} below n={lo.sum():6d} P={pl_:.4f} | above n={hi.sum():6d} P={ph:.4f} | "
              f"ratio {ph/max(pl_,1e-9):8.1f}x | frac of unmatched with bs>=3600: "
              f"{(bs[un] >= 3600).mean():.4f}")
    print("\n  pooled, 10-s bins, to locate the edge exactly:")
    bs, un = t["bs"].to_numpy(), t["unm"].to_numpy()
    for lo in range(3560, 3661, 10):
        m = (bs >= lo) & (bs < lo + 10)
        print(f"   [{lo},{lo+10}) n={m.sum():5d} unm={un[m].sum():4d} P={un[m].mean() if m.sum() else 0:.4f}")


def s4_nm_ceiling() -> None:
    print("\n== S4  the NM-side clamp on matched rows: BLOCK - LOBT ==")
    m = (
        pl.scan_parquet(BASE + "data/work/feat_train.parquet")
        .select(["mvt_minus_aobt3", "BLOCK_TIME_UTC_mvt", "LOBT_flt", "EOBT_1_flt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt"])
        .filter(pl.col("mvt_minus_aobt3").is_not_null())
        .with_columns(
            sec("BLOCK_TIME_UTC_mvt", "LOBT_flt").alias("b_lobt"),
            sec("AOBT_3_flt", "LOBT_flt").alias("a_lobt"),
            sec("LOBT_flt", "SCHED_TIME_UTC_mvt").alias("lobt_sched"),
        )
        .select(["b_lobt", "a_lobt", "lobt_sched"])
        .collect()
    )
    for c in m.columns:
        v = m[c].to_numpy()
        p = np.percentile(v, [1, 50, 99, 99.9])
        print(f"  {c:11s} n={len(v)} p1={p[0]:.0f} p50={p[1]:.0f} p99={p[2]:.0f} p99.9={p[3]:.0f} "
              f"max={v.max():.0f} frac>3600={(v>3600).mean():.4f} frac>3660={(v>3660).mean():.4f}")


def s5_lfpg_groups(d: pl.DataFrame) -> None:
    print("\n== S5  LFPG unmatched split by the edge, and by ms (prediction-time visible) ==")
    L = d.filter((pl.col("ADEP_mvt") == "LFPG") & pl.col("unm"))

    def rep(name: str, s: pl.DataFrame) -> None:
        if not s.height:
            return
        r = s["resid"].to_numpy()
        print(f"  {name:34s} n={len(r):5d} rmse={np.sqrt((r**2).mean()):9.1f} sse={(r**2).sum():.3e} "
              f"bias={r.mean():+9.1f} y_med={s['TAXITIME_SEC_mvt'].median():6.0f}")

    rep("ALL", L)
    rep("bs >= 3600 (late class)", L.filter(pl.col("bs") >= 3600))
    rep("bs <  3600", L.filter(pl.col("bs") < 3600))
    rep("bs <  3600, excl y>=20,000", L.filter((pl.col("bs") < 3600) & (pl.col("TAXITIME_SEC_mvt") < 20000)))
    rep("ms <  3600 (visible)", L.filter(pl.col("mvt_minus_sched") < 3600))
    rep("ms <  3600, excl y>=20,000", L.filter((pl.col("mvt_minus_sched") < 3600) & (pl.col("TAXITIME_SEC_mvt") < 20000)))
    rep("ms >= 3600, excl y>=20,000", L.filter((pl.col("mvt_minus_sched") >= 3600) & (pl.col("TAXITIME_SEC_mvt") < 20000)))
    rep("FLIGHT_mvt starts with FPO", L.filter(pl.col("FLIGHT_mvt").str.starts_with("FPO")))

    print("\n  LFPG matched vs unmatched RMSE at equal ms (y < 20,000):")
    for lane, flt in (("unm", pl.col("unm")), ("mat", ~pl.col("unm"))):
        s = d.filter((pl.col("ADEP_mvt") == "LFPG") & flt & (pl.col("TAXITIME_SEC_mvt") < 20000))
        y, p = s["TAXITIME_SEC_mvt"].to_numpy().astype(float), s["oof"].to_numpy()
        ms, r = s["mvt_minus_sched"].to_numpy().astype(float), s["resid"].to_numpy()
        print(f"   LFPG/{lane} n={len(y)} rmse={np.sqrt((r**2).mean()):.1f}")
        for lo, hi in ((-1e9, 1800), (1800, 3600), (3600, 5400), (5400, 7200), (7200, 10800), (10800, 14400), (14400, 1e9)):
            m = (ms >= lo) & (ms < hi)
            if m.sum() < 3:
                continue
            print(f"     ms[{lo:8.0f},{hi:8.0f}) n={m.sum():6d} y_med={np.median(y[m]):6.0f} "
                  f"sd(y)={y[m].std():7.1f} bias={r[m].mean():+7.1f} rmse={np.sqrt((r[m]**2).mean()):7.1f} "
                  f"sse_share={(r[m]**2).sum()/(r**2).sum():.3f}")


def s6_days(d: pl.DataFrame) -> None:
    print("\n== S6  day clustering at LFPG (non-rollover unmatched) and the matched lane on the same days ==")
    L = d.filter(pl.col("ADEP_mvt") == "LFPG")
    nr = L.filter(pl.col("unm") & (pl.col("TAXITIME_SEC_mvt") < 20000))
    tot = nr["sq"].sum()
    g = (nr.group_by("day").agg(n=pl.len(), sse=pl.col("sq").sum())
         .with_columns(share=(pl.col("sse") / tot).round(3), rmse=(pl.col("sse") / pl.col("n")).sqrt().round(0))
         .sort("sse", descending=True))
    print(g.head(8))
    m = L.filter(~pl.col("unm"))
    mm = (m.group_by("day").agg(n=pl.len(), sse=pl.col("sq").sum())
          .with_columns(rmse=(pl.col("sse") / pl.col("n")).sqrt().round(0)))
    print(f"  LFPG matched baseline RMSE {np.sqrt(m['sq'].sum()/m.height):.1f}; "
          f"per-day matched RMSE p50={np.percentile(mm['rmse'].to_numpy(),50):.0f} "
          f"p90={np.percentile(mm['rmse'].to_numpy(),90):.0f} max={mm['rmse'].max():.0f}")
    print(mm.filter(pl.col("day").is_in(g["day"][:5].to_list())).sort("rmse", descending=True))

    print("\n  SSE lift of airport-days ranked by their unmatched rate (all airports):")
    for lbl, df in (("all", d), ("excl 2 LFPG rollovers",
                                 d.filter(~((pl.col("ADEP_mvt") == "LFPG") & pl.col("unm") & (pl.col("TAXITIME_SEC_mvt") >= 20000))))):
        gg = (df.group_by(["ADEP_mvt", "day"]).agg(n=pl.len(), unm=pl.col("unm").sum(), sse=pl.col("sq").sum())
              .with_columns(rate=pl.col("unm") / pl.col("n")).sort("rate", descending=True))
        cn, cs = np.cumsum(gg["n"].to_numpy()), np.cumsum(gg["sse"].to_numpy())
        for frac in (0.01, 0.05, 0.10):
            i = int(np.searchsorted(cn, frac * df.height))
            print(f"   [{lbl}] top {100*frac:4.0f}% of rows -> {100*cs[i]/df['sq'].sum():5.1f}% of SSE "
                  f"(lift {(cs[i]/df['sq'].sum())/frac:.2f}x), rate cut {gg['rate'][i]:.4f}")


def s7_bound(d: pl.DataFrame) -> None:
    print("\n== S7  the ms-3600 upper bound as a post-processing rule (unmatched rows only) ==")
    u = d.filter("unm")
    y, p = u["TAXITIME_SEC_mvt"].to_numpy().astype(float), u["oof"].to_numpy()
    ms, apt = u["mvt_minus_sched"].to_numpy().astype(float), u["ADEP_mvt"].to_numpy()
    r = d["resid"].to_numpy()
    TOT, N = (r ** 2).sum(), len(r)
    R = np.sqrt(TOT / N)
    cap = ms - 3600
    newp = np.where(cap > 0, np.minimum(p, cap), p)
    ds = (newp - y) ** 2 - (p - y) ** 2
    print(f"  OOF N={N} RMSE={R:.4f}")
    for a in sorted(set(apt)):
        m = apt == a
        v = y[m] > ms[m] - 3600
        print(f"   {a} moved={(newp!=p)[m].sum():4d} dSSE={ds[m].sum():+.3e} "
              f"helped={(ds[m]<-1e-9).sum():3d} hurt={(ds[m]>1e-9).sum():3d} | truth violates the bound on "
              f"{v.sum():4d}/{m.sum():4d} ({v.mean():.3f})")
    for lbl, sel in (("all unmatched", np.ones_like(apt, bool)), ("non-LIRF", apt != "LIRF"),
                     ("6 post-hoc-chosen airports", np.isin(apt, ["EGLL", "LSZH", "EHAM", "LEBL", "EDDM", "LEMD"]))):
        print(f"   {lbl:28s} dSSE={ds[sel].sum():+.3e} dRMSE={np.sqrt((TOT+ds[sel].sum())/N)-R:+.4f}")


def s8_stake(d: pl.DataFrame) -> None:
    print("\n== S8  what LFPG's unmatched lane is worth ==")
    r = d["resid"].to_numpy()
    TOT, N = (r ** 2).sum(), len(r)
    R = np.sqrt(TOT / N)
    apt, unm = d["ADEP_mvt"].to_numpy(), d["unm"].to_numpy()
    y = d["TAXITIME_SEC_mvt"].to_numpy().astype(float)
    lane = (apt == "LFPG") & unm
    roll = lane & (y >= 20000)
    for lbl, m in (("all LFPG unmatched", lane), ("the 2 rollover rows", roll), ("LFPG unmatched non-rollover", lane & ~roll)):
        s = TOT - (r[m] ** 2).sum()
        print(f"   perfect on {lbl:30s} (n={m.sum():5d}) -> OOF RMSE {np.sqrt(s/N):.4f} ({np.sqrt(s/N)-R:+.4f})")
    for frac in (0.5, 0.1):
        s = TOT - frac * (r[lane & ~roll] ** 2).sum()
        print(f"   cut {100*frac:.0f}% of the non-rollover SSE           -> OOF RMSE {np.sqrt(s/N):.4f} ({np.sqrt(s/N)-R:+.4f})")

    rank = (pl.scan_parquet(BASE + "data/work/feat_rank.parquet")
            .select(["ADEP_mvt", "mvt_minus_aobt3", "mvt_minus_sched", "FLIGHT_mvt", "ADES_mvt", "month"])
            .with_columns(pl.col("mvt_minus_aobt3").is_null().alias("unm")).collect())
    RL = rank.filter(pl.col("unm") & (pl.col("ADEP_mvt") == "LFPG"))
    ms = RL["mvt_minus_sched"].to_numpy()
    one_s = 344841 * (2 * 282.7371 + 1)
    per_row = (r[lane & ~roll] ** 2).mean()
    print(f"   ranking frame: unmatched {rank['unm'].sum()} of {rank.height}; LFPG unmatched {RL.height}")
    print(f"   LFPG ranking unmatched ms>=3600 {(ms>=3600).mean():.4f}, ms<3600 n={(ms<3600).sum()}")
    print(f"   1 s of board RMSE = {one_s:.4e} SSE")
    print(f"   perfect on {RL.height} board rows at the OOF non-rollover rate "
          f"({per_row:.4e} SSE/row) = {RL.height*per_row/one_s:.3f} s")
    rate = 2 / 3773
    print(f"   rollover rate in LFPG unmatched training {rate:.5f} -> {rate*RL.height:.2f} expected rows on the board; "
          f"expected SSE {rate*RL.height*np.mean(r[roll]**2):.3e} = {rate*RL.height*np.mean(r[roll]**2)/one_s:.2f} s in expectation")


def main() -> None:
    t = load_train_slim()
    d = load_oof(t)
    print(f"train rows {t.height}, OOF rows {d.height}")
    s1_per_airport(d)
    s2_concentration(d)
    s3_edge(t)
    s4_nm_ceiling()
    s5_lfpg_groups(d)
    s6_days(d)
    s7_bound(d)
    s8_stake(d)


if __name__ == "__main__":
    main()
