"""P1-B -- the LIRF unmatched seasonal bias: is the mechanism real, and is the correction gateable?

Read-only. Reads data/work/*.parquet and experiments/oof_*.parquet; writes nothing into the pipeline.

    POLARS_MAX_THREADS=2 .venv/bin/python docs/competitor_research_2026-09-21/scripts/p1b_lirf_seasonal_bias.py

Sections, each printed with its own banner so the results note can cite `S<n>`:

  S1  LIRF-unmatched schedule-fill rate by calendar month, on all rows and on `ms >= 1800`.
  S2  The claimed +1,356 / t = 6.15 bias, reproduced on the OOF it was measured on
      (main_reparam-bc60017878) and re-measured on the production OOF (main_gap-62dd7c6e88).
  S3  The bias decomposed by month x fill status: is `month` anything other than a proxy for
      the month's fill rate?
  S4  Is the t independent? Day-block and month-block bootstraps, Kish effective n.
  S5  Calendar month alone as a fill classifier -- where it sits against the 0.9201 that closed
      P1 stages 4 and 5.
  S6  The three-arm post-hoc gate: base / noise / candidate, leave-one-month-out shift, decided by
      prc.compare.verdict.
  S7  The downside if the 2026 fill rate does not behave like 2025's.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from prc.compare import group_diff, month_bootstrap_rmse_diff, robustness_curve, verdict

LABEL = "TAXITIME_SEC_mvt"
BASE_OOF = "main_gap-62dd7c6e88"
DOC_OOF = "main_reparam-bc60017878"
MS = "mvt_minus_sched"

pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(25)


def banner(s: str) -> None:
    print("\n" + "=" * 100)
    print(s)
    print("=" * 100)


def load_frame() -> pl.DataFrame:
    tr = pl.read_parquet(
        ROOT / "data/work/feat_train.parquet",
        columns=[
            "MVT_ID_mvt",
            "ADEP_mvt",
            "MVT_TIME_UTC_mvt",
            "BLOCK_TIME_UTC_mvt",
            "SCHED_TIME_UTC_mvt",
            MS,
            "mvt_minus_aobt3",
            LABEL,
        ],
    )
    return tr.with_columns(
        [
            (pl.col("MVT_TIME_UTC_mvt").dt.year() * 100 + pl.col("MVT_TIME_UTC_mvt").dt.month()).alias("ym"),
            pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"),
            pl.col("MVT_TIME_UTC_mvt").dt.date().alias("day"),
            ((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= 10).alias(
                "f10"
            ),
            pl.col("mvt_minus_aobt3").is_null().alias("unmatched"),
        ]
    )


def attach_oof(tr: pl.DataFrame, exp_id: str) -> pl.DataFrame:
    oof = pl.read_parquet(ROOT / f"experiments/oof_{exp_id}.parquet")
    folds = pl.read_parquet(ROOT / f"experiments/oof_{exp_id}_folds.parquet")
    if "pooled" in folds.columns:
        folds = folds.filter(pl.col("pooled"))
    else:
        # older sidecars carry every fold including the shadowed jul_fwd; keep the pooled three.
        folds = folds.filter(pl.col("fold") != "jul_fwd")
    d = tr.join(oof, on="MVT_ID_mvt", how="inner").drop_nulls("oof")
    d = d.join(folds.select(["MVT_ID_mvt", "fold"]).unique(subset=["MVT_ID_mvt"]), on="MVT_ID_mvt", how="left")
    return d.with_columns((pl.col("oof") - pl.col(LABEL)).alias("resid"))


# ----------------------------------------------------------------------------------------------- S1


def s1(tr: pl.DataFrame) -> None:
    banner("S1  LIRF unmatched: schedule-fill rate by calendar month")
    lirf = tr.filter(pl.col("unmatched") & (pl.col("ADEP_mvt") == "LIRF"))
    print(f"LIRF unmatched training rows: {lirf.height}   (`ms` nulls: {lirf[MS].null_count()})")
    for name, sub in (("all `ms`", lirf), ("`ms` >= 1800", lirf.filter(pl.col(MS) >= 1800))):
        g = (
            sub.group_by("month")
            .agg([pl.len().alias("n"), pl.col("f10").mean().round(4).alias("fill10")])
            .sort("month")
        )
        print(f"\n-- {name} (n={sub.height}, pooled fill {sub['f10'].mean():.4f}) --")
        print(g)
    rk = pl.read_parquet(ROOT / "data/work/feat_rank.parquet", columns=["ADEP_mvt", "MVT_TIME_UTC_mvt", MS, "mvt_minus_aobt3"])
    rk = rk.filter(pl.col("mvt_minus_aobt3").is_null() & (pl.col("ADEP_mvt") == "LIRF")).with_columns(
        pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month")
    )
    print("\n-- scored (ranking) LIRF unmatched rows --")
    print(
        rk.group_by("month")
        .agg([pl.len().alias("n_all_ms"), (pl.col(MS) >= 1800).sum().alias("n_ms_ge_1800")])
        .sort("month")
    )
    print(f"total LIRF unmatched scored rows: {rk.height}; with ms>=1800: {(rk[MS] >= 1800).sum()}")


# ----------------------------------------------------------------------------------------------- S2


def bias_row(r: np.ndarray) -> tuple[float, float, float, int]:
    n = len(r)
    b = float(r.mean())
    se = float(r.std(ddof=1) / np.sqrt(n))
    return b, se, b / se, n


def s2(tr: pl.DataFrame) -> dict:
    banner("S2  The claimed +1,356 / t = 6.15, on the OOF it came from and on the production OOF")
    out = {}
    for exp in (DOC_OOF, BASE_OOF):
        d = attach_oof(tr, exp)
        lirf = d.filter(pl.col("unmatched") & (pl.col("ADEP_mvt") == "LIRF"))
        print(f"\n### {exp}   pooled OOF rows {d.height}   LIRF-unmatched OOF rows {lirf.height}")
        for cell, sub in (
            ("B-LIRF all `ms`", lirf),
            ("B-LIRF `ms`>=1800", lirf.filter(pl.col(MS) >= 1800)),
        ):
            for mon, s2_ in (
                ("all OOF months", sub),
                ("months 1+7 only", sub.filter(pl.col("month").is_in([1, 7]))),
            ):
                r = s2_["resid"].to_numpy()
                b, se, t, n = bias_row(r)
                rmse = float(np.sqrt((r**2).mean()))
                print(f"  {cell:22s} {mon:16s} n={n:5d}  RMSE={rmse:9.1f}  bias={b:+9.1f}  SE={se:7.1f}  t={t:5.2f}")
                out[(exp, cell, mon)] = (b, se, t, n)
        print("  -- per month (`ms`>=1800) --")
        g = (
            lirf.filter(pl.col(MS) >= 1800)
            .group_by("month")
            .agg(
                [
                    pl.len().alias("n"),
                    pl.col("resid").mean().round(1).alias("bias"),
                    pl.col(LABEL).mean().round(0).alias("E_y"),
                    pl.col("oof").mean().round(0).alias("E_pred"),
                ]
            )
            .sort("month")
        )
        print(g)
    return out


# ----------------------------------------------------------------------------------------------- S3


def s3(tr: pl.DataFrame) -> None:
    banner("S3  Does `month` carry anything beyond the month's fill rate?")
    d = attach_oof(tr, BASE_OOF)
    lirf = d.filter(pl.col("unmatched") & (pl.col("ADEP_mvt") == "LIRF") & (pl.col(MS) >= 1800))
    print(f"B-LIRF `ms`>=1800 OOF rows: {lirf.height}")
    print("\n-- bias by fill status, pooled over the four OOF months --")
    print(
        lirf.group_by("f10")
        .agg(
            [
                pl.len().alias("n"),
                pl.col("resid").mean().round(1).alias("bias"),
                (pl.col("resid").std(ddof=1) / pl.len().sqrt()).round(1).alias("se"),
                pl.col("resid").pow(2).mean().sqrt().round(1).alias("rmse"),
            ]
        )
        .sort("f10")
    )
    print("\n-- bias by month x fill status --")
    print(
        lirf.group_by(["month", "f10"])
        .agg([pl.len().alias("n"), pl.col("resid").mean().round(1).alias("bias")])
        .sort(["month", "f10"])
    )
    bf = float(lirf.filter(pl.col("f10"))["resid"].mean())
    bn = float(lirf.filter(~pl.col("f10"))["resid"].mean())
    print(f"\npooled filled bias {bf:+.1f}, pooled non-filled bias {bn:+.1f}")
    print("-- month bias, observed against the pure-composition prediction f*bf + (1-f)*bn --")
    rows = []
    for m in sorted(lirf["month"].unique().to_list()):
        sub = lirf.filter(pl.col("month") == m)
        f = float(sub["f10"].mean())
        obs = float(sub["resid"].mean())
        rows.append({"month": m, "n": sub.height, "fill": round(f, 3), "observed": round(obs, 1),
                     "composition": round(f * bf + (1 - f) * bn, 1), "residual": round(obs - (f * bf + (1 - f) * bn), 1)})
    print(pl.DataFrame(rows))


# ----------------------------------------------------------------------------------------------- S4


def s4(tr: pl.DataFrame) -> None:
    banner("S4  Is the t = 6 computed on independent rows?")
    d = attach_oof(tr, BASE_OOF)
    lirf = d.filter(
        pl.col("unmatched") & (pl.col("ADEP_mvt") == "LIRF") & (pl.col(MS) >= 1800) & pl.col("month").is_in([1, 7])
    )
    r = lirf["resid"].to_numpy()
    b, se, t, n = bias_row(r)
    print(f"months 1+7, B-LIRF ms>=1800:  n={n}  bias={b:+.1f}  iid SE={se:.1f}  iid t={t:.2f}")

    rng = np.random.default_rng(0)
    # day blocks
    for unit in ("day", "month", "fold"):
        keys = lirf[unit].to_numpy() if unit != "day" else lirf["day"].cast(pl.Utf8).to_numpy()
        uniq = np.unique(keys)
        idx = {u: np.flatnonzero(keys == u) for u in uniq}
        draws = np.empty(4000)
        for i in range(4000):
            pick = np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq), replace=True)])
            draws[i] = r[pick].mean()
        lo, hi = np.percentile(draws, [2.5, 97.5])
        print(
            f"  {unit:6s}-block bootstrap ({len(uniq):3d} blocks): bias {b:+.1f}  CI95 [{lo:+.1f}, {hi:+.1f}]  "
            f"SE {draws.std(ddof=1):.1f}  P(bias>0) {(draws > 0).mean():.3f}"
        )
    # Kish effective sample size over day blocks
    keys = lirf["day"].cast(pl.Utf8).to_numpy()
    sizes = np.array([np.sum(keys == u) for u in np.unique(keys)], float)
    print(f"  day blocks: {len(sizes)}, sizes min/median/max {sizes.min():.0f}/{np.median(sizes):.0f}/{sizes.max():.0f}")
    print(f"  Kish n_eff over day blocks = {sizes.sum() ** 2 / (sizes ** 2).sum():.1f} against n = {n}")
    # month-level view: only two observations
    for m in (1, 7):
        sub = lirf.filter(pl.col("month") == m)
        bb, ss, tt, nn = bias_row(sub["resid"].to_numpy())
        print(f"  month {m}: n={nn:4d}  bias={bb:+.1f}  iid SE={ss:.1f}  iid t={tt:.2f}")
    print("  -> the two scored months are 2 draws of the quantity that has to transfer to 2026.")


# ----------------------------------------------------------------------------------------------- S5


def s5(tr: pl.DataFrame) -> None:
    banner("S5  Calendar month alone as a fill classifier (against the 0.9201 that closed stages 4/5)")
    lirf = tr.filter(pl.col("unmatched") & (pl.col("ADEP_mvt") == "LIRF"))

    def auc(y: np.ndarray, s: np.ndarray) -> float:
        order = np.argsort(s, kind="mergesort")
        y = y[order]
        s = s[order]
        ranks = np.empty(len(s), float)
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and s[j + 1] == s[i]:
                j += 1
            ranks[i : j + 1] = 0.5 * (i + j) + 1
            i = j + 1
        npos, nneg = y.sum(), (1 - y).sum()
        if npos == 0 or nneg == 0:
            return float("nan")
        return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))

    y = lirf["f10"].cast(pl.Int8).to_numpy().astype(float)
    months = lirf["month"].to_numpy()
    # leave-one-month-out: the score for a row in month m is the fill rate of month m estimated
    # from the OTHER months -- which is degenerate (it cannot see m). Both the in-sample score and
    # the honest one are reported.
    rate = {m: float(lirf.filter(pl.col("month") == m)["f10"].mean()) for m in np.unique(months)}
    s_in = np.array([rate[m] for m in months])
    print(f"  in-sample calendar-month fill rate as a score: AUC = {auc(y, s_in):.4f}  (n={len(y)}, pos={int(y.sum())})")
    print("  leave-one-month-out: a calendar month has no other instance in a single training year,")
    print("  so the honest out-of-month score for month m is undefined -- there is nothing to estimate it from.")
    print("  For reference, P1 stage 2's best per-row classifier reached held-out AUC 0.9201 and FAILED its 0.93 gate.")


# ----------------------------------------------------------------------------------------------- S6


def lomo_shift(lirf: pl.DataFrame) -> np.ndarray:
    """Per-row shift to ADD to the prediction: minus the mean residual of the OTHER OOF months."""
    months = lirf["month"].to_numpy()
    resid = lirf["resid"].to_numpy()
    out = np.zeros(len(resid))
    for m in np.unique(months):
        other = months != m
        out[months == m] = -resid[other].mean()
    return out


def s6(tr: pl.DataFrame) -> None:
    banner("S6  The three-arm gate: base / noise / candidate, leave-one-month-out shift")
    d = attach_oof(tr, BASE_OOF)
    y = d[LABEL].to_numpy().astype(float)
    p_base = d["oof"].to_numpy().astype(float)
    month = d["month"].to_numpy()
    cell = (d["unmatched"] & (d["ADEP_mvt"] == "LIRF") & (d[MS] >= 1800)).to_numpy()
    print(f"pooled OOF rows {len(y)}; candidate cell (B-LIRF ms>=1800) {cell.sum()} rows")

    lirf = d.filter(pl.Series(cell))
    sh = lomo_shift(lirf)
    print("\n-- the leave-one-month-out shift actually applied, per OOF month --")
    for m in sorted(set(lirf["month"].to_list())):
        sel = lirf["month"].to_numpy() == m
        print(f"   month {m}: n={sel.sum():4d}  shift applied {sh[sel][0]:+9.1f} s "
              f"(this month's own bias {lirf.filter(pl.col('month') == m)['resid'].mean():+9.1f})")

    p_cand = p_base.copy()
    p_cand[cell] = np.clip(p_cand[cell] + sh, 0.0, None)

    # noise arm: identical estimator, applied to a control cell of the same size drawn from the
    # non-LIRF unmatched rows, which §4.2 of large-ms-population.md measured as unbiased.
    ctrl_pool = (d["unmatched"] & (d["ADEP_mvt"] != "LIRF") & (d[MS] >= 1800)).to_numpy()

    pool_idx = np.flatnonzero(ctrl_pool)
    noise_curves = []
    for seed in range(5):
        rng2 = np.random.default_rng(100 + seed)
        pick = rng2.choice(pool_idx, int(cell.sum()), replace=False)
        cmask = np.zeros(len(y), bool)
        cmask[pick] = True
        sub = d.filter(pl.Series(cmask))
        shn = lomo_shift(sub)
        p_noise = p_base.copy()
        p_noise[cmask] = np.clip(p_noise[cmask] + shn, 0.0, None)
        curve = robustness_curve(y, p_base, p_noise, n_boot=300, seed=1)
        noise_curves.append((seed, curve, p_noise))
        full = curve[0]
        k200 = next(r for r in curve if r["dropped"] == 200)
        print(f"   noise seed {seed}: K=0 {full['diff']:+.4f} [{full['ci95'][0]:+.4f},{full['ci95'][1]:+.4f}]  "
              f"K=200 {k200['diff']:+.4f} [{k200['ci95'][0]:+.4f},{k200['ci95'][1]:+.4f}]")

    print("\n### candidate against base -- whole-OOF robustness curve (the rule's own instrument)")
    curve = robustness_curve(y, p_base, p_cand, n_boot=500, seed=0)
    for r in curve:
        print(f"   K={r['dropped']:5d}  A={r['rmse_a']:9.4f}  B={r['rmse_b']:9.4f}  diff={r['diff']:+9.4f}  "
              f"CI95=[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  P(B)={r['b_better_prob']:.3f}")
    ok, why = verdict(curve)
    print(f"   VERDICT: {'ADOPT' if ok else 'REJECT'} - {why}")
    mb = month_bootstrap_rmse_diff(y, p_base, p_cand, month, n_boot=2000, seed=0)
    print(f"   month bootstrap: {mb['diff_rmse']:+.4f} [{mb['ci95'][0]:+.4f}, {mb['ci95'][1]:+.4f}]  "
          f"P(B)={mb['b_better_prob']:.3f}  n_months={mb['n_months']}")

    # the best noise arm, for the margin
    best = min(noise_curves, key=lambda t: t[1][0]["diff"])
    print(f"\n### candidate against the noise arm (seed {best[0]}, the most favourable of 5)")
    cn = robustness_curve(y, best[2], p_cand, n_boot=500, seed=0)
    for r in cn:
        print(f"   K={r['dropped']:5d}  diff={r['diff']:+9.4f}  CI95=[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  P(B)={r['b_better_prob']:.3f}")
    okn, whyn = verdict(cn)
    print(f"   VERDICT vs noise: {'ADOPT' if okn else 'REJECT'} - {whyn}")

    print("\n### where the change lives (reporting only, never the verdict)")
    print("   candidate cell:", group_diff(y, p_base, p_cand, cell, n_boot=500, seed=0))
    unm = d["unmatched"].to_numpy()
    print("   all unmatched :", group_diff(y, p_base, p_cand, unm, n_boot=500, seed=0))
    for m in sorted(set(month)):
        sel = cell & (month == m)
        g = group_diff(y, p_base, p_cand, sel, n_boot=300, seed=0)
        print(f"   cell, month {m}: n={g['n']:4d}  RMSE {g['rmse_a']:9.1f} -> {g['rmse_b']:9.1f}  diff {g['diff']:+9.1f}")

    print("\n### the same gate restricted to the two seasonally-matched folds (jan + jul)")
    keep = np.isin(month, [1, 7])
    c2 = robustness_curve(y[keep], p_base[keep], p_cand[keep], n_boot=500, seed=0)
    for r in c2:
        print(f"   K={r['dropped']:5d}  A={r['rmse_a']:9.4f}  B={r['rmse_b']:9.4f}  diff={r['diff']:+9.4f}  "
              f"CI95=[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  P(B)={r['b_better_prob']:.3f}")
    ok2, why2 = verdict(c2)
    print(f"   VERDICT (jan+jul only): {'ADOPT' if ok2 else 'REJECT'} - {why2}")

    print("\n### the oracle arm: the shift each month would need, fitted on that month itself")
    # Each month's own in-sample mean residual removed. Circular by construction: an upper bound on
    # what any month-keyed shift could achieve, not an achievable arm.
    mm = {m: float(lirf.filter(pl.col("month") == m)["resid"].mean()) for m in set(lirf["month"].to_list())}
    p_or = p_base.copy()
    p_or[cell] = np.clip(p_base[cell] - np.array([mm[m] for m in lirf["month"].to_list()]), 0.0, None)
    co = robustness_curve(y, p_base, p_or, n_boot=500, seed=0)
    for r in co:
        print(f"   K={r['dropped']:5d}  diff={r['diff']:+9.4f}  CI95=[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]")
    oko, whyo = verdict(co)
    print(f"   VERDICT (oracle, circular by construction -- an upper bound, not a result): "
          f"{'ADOPT' if oko else 'REJECT'} - {whyo}")


# ----------------------------------------------------------------------------------------------- S7


def s7(tr: pl.DataFrame) -> None:
    banner("S7  What the bet costs if the 2026 fill rate does not behave like 2025's")
    d = attach_oof(tr, BASE_OOF)
    lirf = d.filter(pl.col("unmatched") & (pl.col("ADEP_mvt") == "LIRF") & (pl.col(MS) >= 1800))
    n_scored = 356
    rk = pl.read_parquet(ROOT / "data/work/feat_rank.parquet", columns=["ADEP_mvt", MS, "mvt_minus_aobt3"])
    n_scored = int(
        rk.filter(pl.col("mvt_minus_aobt3").is_null() & (pl.col("ADEP_mvt") == "LIRF") & (pl.col(MS) >= 1800)).height
    )
    N, RMSE = 344841, 282.7371
    per_s = N * (2 * RMSE + 1)
    print(f"scored rows in the cell: {n_scored};  1 s of board RMSE = {per_s:.4e} of SSE")
    print("\n-- board effect of a shift of size -S applied to those rows, as a function of the TRUE bias b --")
    print("   dSSE = n*(S^2 - 2*S*b);  negative = better")
    for S in (881.0, 1204.0, 1356.0, 1382.0):
        row = [f"S={S:7.0f}:"]
        for b in (-1226.0, -73.0, 0.0, 440.0, 881.0, 1204.0, 1356.0):
            d_sse = n_scored * (S**2 - 2 * S * b)
            row.append(f"b={b:+7.0f} -> {d_sse / per_s:+6.2f}s")
        print("   " + "  ".join(row))
    print("\n   break-even is b = S/2: the shift is a loss whenever the true 2026 bias is below half the shift.")
    print("\n-- the four OOF months are the only evidence about what b can be --")
    print(
        lirf.group_by("month")
        .agg([pl.len().alias("n"), pl.col("resid").mean().round(0).alias("bias"), pl.col("f10").mean().round(3).alias("fill")])
        .sort("month")
    )


# ----------------------------------------------------------------------------------------------- S8


def s8(tr: pl.DataFrame) -> None:
    banner("S8  The two stronger forms of the correction, gated the same way")
    d = attach_oof(tr, BASE_OOF)
    y = d[LABEL].to_numpy().astype(float)
    p_base = d["oof"].to_numpy().astype(float)
    month = d["month"].to_numpy()
    cell = (d["unmatched"] & (d["ADEP_mvt"] == "LIRF") & (d[MS] >= 1800)).to_numpy()
    lirf = d.filter(pl.Series(cell))
    lm = lirf["month"].to_numpy()
    lr = lirf["resid"].to_numpy()
    months = np.unique(lm)

    # training-side fill rate per calendar month, from the LIRF unmatched training rows themselves.
    fill = {
        int(m): float(
            tr.filter(pl.col("unmatched") & (pl.col("ADEP_mvt") == "LIRF") & (pl.col(MS) >= 1800) & (pl.col("month") == m))[
                "f10"
            ].mean()
        )
        for m in range(1, 13)
    }
    print("  per-month training fill rate (ms>=1800):", {k: round(v, 3) for k, v in fill.items()})

    def gate(name: str, p_b: np.ndarray, keep: np.ndarray | None = None) -> None:
        yy, pa, pb = (y, p_base, p_b) if keep is None else (y[keep], p_base[keep], p_b[keep])
        c = robustness_curve(yy, pa, pb, n_boot=500, seed=0)
        for r in c:
            print(f"     K={r['dropped']:5d}  A={r['rmse_a']:9.4f}  B={r['rmse_b']:9.4f}  diff={r['diff']:+9.4f}  "
                  f"CI95=[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  P(B)={r['b_better_prob']:.3f}")
        ok, why = verdict(c)
        print(f"     VERDICT [{name}]: {'ADOPT' if ok else 'REJECT'} - {why}")

    # --- C2: the note's own test -- estimate on one scored month, apply to the other; the two
    # non-scored OOF months are left untouched. Uses the calendar (which months are scored), which
    # is legitimately known at serve time.
    print("\n### C2  scored-month LOMO (month 1 <-> month 7 only), the form large-ms-population.md §6.2 measured")
    b1 = float(lr[lm == 1].mean())
    b7 = float(lr[lm == 7].mean())
    sh2 = np.zeros(len(lr))
    sh2[lm == 1] = -b7
    sh2[lm == 7] = -b1
    print(f"     shift applied: month 1 gets {-b7:+.1f} (from month 7), month 7 gets {-b1:+.1f} (from month 1)")
    p2 = p_base.copy()
    p2[cell] = np.clip(p2[cell] + sh2, 0.0, None)
    gate("C2, whole OOF", p2)
    keep17 = np.isin(month, [1, 7])
    gate("C2, jan+jul folds only", p2, keep17)
    print("     cell only:", group_diff(y, p_base, p2, cell, n_boot=500, seed=0))
    for m in (1, 7):
        g = group_diff(y, p_base, p2, cell & (month == m), n_boot=300, seed=0)
        print(f"     cell month {m}: n={g['n']:4d} RMSE {g['rmse_a']:9.1f} -> {g['rmse_b']:9.1f} diff {g['diff']:+9.1f} "
              f"CI95 [{g['ci95'][0]:+.1f}, {g['ci95'][1]:+.1f}]")

    # --- C3: the mechanism's own form -- bias regressed on the month's fill rate, leave-one-month-out.
    print("\n### C3  fill-rate-keyed shift: bias ~ a + c*fill, fitted on the OTHER OOF months, LOMO over all four")
    sh3 = np.zeros(len(lr))
    for m in months:
        oth = [int(x) for x in months if x != m]
        bx = np.array([fill[o] for o in oth])
        by = np.array([lr[lm == o].mean() for o in oth])
        c, a = np.polyfit(bx, by, 1)
        pred = a + c * fill[int(m)]
        sh3[lm == m] = -pred
        print(f"     month {int(m):2d}: fit on {oth} -> bias_hat({fill[int(m)]:.3f}) = {pred:+8.1f}; "
              f"shift {-pred:+8.1f}; this month's actual bias {lr[lm == m].mean():+8.1f}")
    p3 = p_base.copy()
    p3[cell] = np.clip(p3[cell] + sh3, 0.0, None)
    gate("C3, whole OOF", p3)
    gate("C3, jan+jul folds only", p3, keep17)
    print("     cell only:", group_diff(y, p_base, p3, cell, n_boot=500, seed=0))
    mb = month_bootstrap_rmse_diff(y, p_base, p3, month, n_boot=2000, seed=0)
    print(f"     month bootstrap: {mb['diff_rmse']:+.4f} [{mb['ci95'][0]:+.4f}, {mb['ci95'][1]:+.4f}] P(B)={mb['b_better_prob']:.3f}")

    # --- how well does fill rate explain the month bias at all?
    print("\n### the regression the whole mechanism rests on, on its four points")
    bx = np.array([fill[int(m)] for m in months])
    by = np.array([lr[lm == m].mean() for m in months])
    c, a = np.polyfit(bx, by, 1)
    pred = a + c * bx
    ss_res = float(((by - pred) ** 2).sum())
    ss_tot = float(((by - by.mean()) ** 2).sum())
    print(f"     fill: {np.round(bx, 3).tolist()}   bias: {np.round(by, 0).tolist()}")
    print(f"     bias = {a:+.0f} {c:+.0f} * fill ;  R^2 = {1 - ss_res / ss_tot:.3f} on 4 points, 2 parameters")
    print(f"     implied bias at the 2025 Jan/Jul fill rates: {a + c * fill[1]:+.0f} / {a + c * fill[7]:+.0f}")


# ----------------------------------------------------------------------------------------------- S9


def s9(tr: pl.DataFrame) -> None:
    banner("S9  Three arms on the same rows: base / noise / candidate, for each candidate form")
    d = attach_oof(tr, BASE_OOF)
    y = d[LABEL].to_numpy().astype(float)
    p_base = d["oof"].to_numpy().astype(float)
    month = d["month"].to_numpy()
    fold = d["fold"].to_numpy()
    cell = (d["unmatched"] & (d["ADEP_mvt"] == "LIRF") & (d[MS] >= 1800)).to_numpy()
    fill = {
        int(m): float(
            tr.filter(
                pl.col("unmatched") & (pl.col("ADEP_mvt") == "LIRF") & (pl.col(MS) >= 1800) & (pl.col("month") == m)
            )["f10"].mean()
        )
        for m in range(1, 13)
    }

    def shifts(mask: np.ndarray, form: str) -> np.ndarray:
        """The shift to ADD, for the rows in `mask`, under one of the three estimator forms."""
        sub = d.filter(pl.Series(mask))
        mm = sub["month"].to_numpy()
        rr = (sub["oof"] - sub[LABEL]).to_numpy().astype(float)
        out = np.zeros(len(rr))
        months = np.unique(mm)
        if form == "C1":  # pooled other OOF months
            for m in months:
                out[mm == m] = -rr[mm != m].mean()
        elif form == "C2":  # the other scored month only; non-scored months untouched
            if 1 in months and 7 in months:
                out[mm == 1] = -rr[mm == 7].mean()
                out[mm == 7] = -rr[mm == 1].mean()
        elif form == "C3":  # bias ~ a + c*fill on the other months, evaluated at this month's fill
            for m in months:
                oth = [int(x) for x in months if x != m]
                if len(oth) < 2:
                    continue
                c, a = np.polyfit(np.array([fill[o] for o in oth]), np.array([rr[mm == o].mean() for o in oth]), 1)
                out[mm == m] = -(a + c * fill[int(m)])
        return out

    def apply(mask: np.ndarray, form: str) -> np.ndarray:
        p = p_base.copy()
        p[mask] = np.clip(p[mask] + shifts(mask, form), 0.0, None)
        return p

    views = {
        "whole OOF (jan+jul+forward)": np.ones(len(y), bool),
        "jan+jul folds only": np.isin(fold, ["jan", "jul"]),
        "forward fold only": fold == "forward",
    }
    pool = np.flatnonzero((d["unmatched"] & (d["ADEP_mvt"] != "LIRF") & (d[MS] >= 1800)).to_numpy())

    for form in ("C1", "C2", "C3"):
        print(f"\n################ form {form} ################")
        p_c = apply(cell, form)
        # noise arms: the identical estimator on an equally sized control cell drawn from the
        # non-LIRF unmatched ms>=1800 rows, which large-ms-population.md §4.2 measures as unbiased.
        noise = []
        for seed in range(5):
            pick = np.random.default_rng(1000 + seed).choice(pool, int(cell.sum()), replace=False)
            nm = np.zeros(len(y), bool)
            nm[pick] = True
            noise.append(apply(nm, form))
        for vname, keep in views.items():
            cur = robustness_curve(y[keep], p_base[keep], p_c[keep], n_boot=500, seed=0)
            f0 = cur[0]
            k2 = next(r for r in cur if r["dropped"] == 200)
            ok, _why = verdict(cur)
            print(f"  [{vname}]  n={int(keep.sum())}")
            print(f"    candidate vs base : K=0 {f0['diff']:+7.3f} [{f0['ci95'][0]:+7.3f},{f0['ci95'][1]:+7.3f}]   "
                  f"K=200 {k2['diff']:+7.3f} [{k2['ci95'][0]:+7.3f},{k2['ci95'][1]:+7.3f}]   "
                  f"VERDICT {'ADOPT' if ok else 'REJECT'}")
            n0 = [robustness_curve(y[keep], p_base[keep], pn[keep], n_boot=300, seed=0) for pn in noise]
            a0 = np.array([c[0]["diff"] for c in n0])
            a2 = np.array([next(r for r in c if r["dropped"] == 200)["diff"] for c in n0])
            print(f"    noise vs base (5) : K=0 median {np.median(a0):+7.3f} range [{a0.min():+7.3f},{a0.max():+7.3f}]   "
                  f"K=200 median {np.median(a2):+7.3f} range [{a2.min():+7.3f},{a2.max():+7.3f}]")
            worst = noise[int(np.argmin(a0))]
            cn = robustness_curve(y[keep], worst[keep], p_c[keep], n_boot=500, seed=0)
            fn0 = cn[0]
            kn2 = next(r for r in cn if r["dropped"] == 200)
            okn, _ = verdict(cn)
            print(f"    candidate vs noise: K=0 {fn0['diff']:+7.3f} [{fn0['ci95'][0]:+7.3f},{fn0['ci95'][1]:+7.3f}]   "
                  f"K=200 {kn2['diff']:+7.3f} [{kn2['ci95'][0]:+7.3f},{kn2['ci95'][1]:+7.3f}]   "
                  f"VERDICT {'ADOPT' if okn else 'REJECT'}")
            mb = month_bootstrap_rmse_diff(y[keep], p_base[keep], p_c[keep], month[keep], n_boot=2000, seed=0)
            print(f"    month bootstrap   : {mb['diff_rmse']:+7.3f} [{mb['ci95'][0]:+7.3f},{mb['ci95'][1]:+7.3f}]  "
                  f"P(B)={mb['b_better_prob']:.3f}  n_months={mb['n_months']}")


# ---------------------------------------------------------------------------------------------- S10


def s10() -> None:
    banner("S10  Is the 2026 scored frame a low-fill population? A measured answer, not an assumption")
    cols = ["ADEP_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", MS, "mvt_minus_aobt3", "FLIGHT_mvt"]
    tr = pl.read_parquet(ROOT / "data/work/feat_train.parquet", columns=cols).with_columns(
        [
            pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"),
            ((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= 10)
            .cast(pl.Float64)
            .alias("f10"),
            pl.col("FLIGHT_mvt").str.slice(0, 3).alias("pfx"),
        ]
    )
    lirf = tr.filter(pl.col("mvt_minus_aobt3").is_null() & (pl.col("ADEP_mvt") == "LIRF") & (pl.col(MS) >= 1800))
    rk = pl.read_parquet(ROOT / "data/work/feat_rank.parquet", columns=[c for c in cols if "BLOCK" not in c and "SCHED_TIME" not in c]).with_columns(
        [pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"), pl.col("FLIGHT_mvt").str.slice(0, 3).alias("pfx")]
    )
    rk = rk.filter(pl.col("mvt_minus_aobt3").is_null() & (pl.col("ADEP_mvt") == "LIRF") & (pl.col(MS) >= 1800))
    print(f"training LIRF-unmatched ms>=1800 rows {lirf.height}; scored {rk.height}")

    K = 5.0

    def table(fit: pl.DataFrame) -> tuple[dict, float]:
        g = float(fit["f10"].mean())
        t = fit.group_by("pfx").agg([pl.len().alias("n"), pl.col("f10").mean().alias("p")])
        return {r["pfx"]: (r["n"] * r["p"] + K * g) / (r["n"] + K) for r in t.iter_rows(named=True)}, g

    print("\n-- leave-one-month-out check that a prefix table predicts a MONTH's fill rate --")
    rows = []
    for m in range(1, 13):
        fit = lirf.filter(pl.col("month") != m)
        tb, g = table(fit)
        sub = lirf.filter(pl.col("month") == m)
        pred = np.array([tb.get(p, g) for p in sub["pfx"].to_list()])
        rows.append({"month": m, "n": sub.height, "actual": round(float(sub["f10"].mean()), 3),
                     "predicted": round(float(pred.mean()), 3), "seen_pfx": int(sum(p in tb for p in sub["pfx"].to_list()))})
    t12 = pl.DataFrame(rows)
    print(t12)
    a = t12["actual"].to_numpy()
    p = t12["predicted"].to_numpy()
    print(f"   12-month corr(actual, predicted) = {np.corrcoef(a, p)[0, 1]:.3f}; MAE = {np.abs(a - p).mean():.3f}; "
          f"actual spread {a.min():.3f}-{a.max():.3f}; predicted spread {p.min():.3f}-{p.max():.3f}")

    tb, g = table(lirf)
    pred = np.array([tb.get(x, g) for x in rk["pfx"].to_list()])
    print(f"\n-- the same table, fitted on all 2025, applied to the {rk.height} scored rows --")
    print(f"   predicted 2026 fill rate, all scored rows : {pred.mean():.3f}   (2025 pooled actual {g:.3f})")
    for m in (1, 7):
        sel = (rk["month"] == m).to_numpy()
        act = float(lirf.filter(pl.col("month") == m)["f10"].mean())
        print(f"   month {m}: scored n={int(sel.sum()):4d}  predicted {pred[sel].mean():.3f}   "
              f"2025 same-month actual {act:.3f}   2025 same-month predicted (LOMO) {t12.filter(pl.col('month') == m)['predicted'].item():.3f}")
    unseen = sum(x not in tb for x in rk["pfx"].to_list())
    print(f"   scored rows whose prefix is unseen in the 2025 LIRF-unmatched table: {unseen} of {rk.height}")
    # The prefix table shrinks hard, so its month means are compressed. Calibrate them against the
    # 12 observed month rates and map the scored predictions through the same line.
    slope, inter = np.polyfit(p, a, 1)
    ss = float(((a - (inter + slope * p)) ** 2).sum())
    st = float(((a - a.mean()) ** 2).sum())
    print("\n-- calibrating the compressed prediction against the 12 observed month rates --")
    print(f"   actual = {inter:+.3f} {slope:+.3f} * predicted ;  R^2 = {1 - ss / st:.3f} on 12 months")
    for m in (1, 7):
        sel = (rk["month"] == m).to_numpy()
        cal = inter + slope * pred[sel].mean()
        act25 = float(lirf.filter(pl.col("month") == m)["f10"].mean())
        print(f"   month {m}: calibrated 2026 fill estimate {cal:.3f}  against 2025 actual {act25:.3f}  "
              f"(difference {cal - act25:+.3f})")
    print("\n   NOTE: this estimates the 2026 fill rate from a 2025-fitted prefix table. It is evidence about")
    print("   the scored population's composition, not a measurement of 2026 labels, which do not exist.")


# ---------------------------------------------------------------------------------------------- S11


def s11(tr: pl.DataFrame) -> None:
    banner("S11  The bet, priced per scored month, under each belief about the 2026 bias")
    d = attach_oof(tr, BASE_OOF)
    lirf = d.filter(pl.col("unmatched") & (pl.col("ADEP_mvt") == "LIRF") & (pl.col(MS) >= 1800))
    b = {int(m): float(lirf.filter(pl.col("month") == m)["resid"].mean()) for m in (1, 7, 11, 12)}
    N, RMSE = 344841, 282.7371
    per_s = N * (2 * RMSE + 1)
    n_scored = {1: 91, 7: 265}
    # S8's line, refitted here so the number carries its own source
    fill = {
        int(m): float(
            tr.filter(pl.col("unmatched") & (pl.col("ADEP_mvt") == "LIRF") & (pl.col(MS) >= 1800) & (pl.col("month") == m))[
                "f10"
            ].mean()
        )
        for m in (1, 7, 11, 12)
    }
    c, a = np.polyfit(np.array([fill[m] for m in (1, 7, 11, 12)]), np.array([b[m] for m in (1, 7, 11, 12)]), 1)
    print(f"  S8 line refitted: bias = {a:+.0f} {c:+.0f} * fill")
    # the C2 shift is the OTHER scored month's bias
    shift = {1: -b[7], 7: -b[1]}
    print(f"  C2 shifts: month 1 {shift[1]:+.1f}, month 7 {shift[7]:+.1f}  (from 2025 biases {b[1]:+.1f} / {b[7]:+.1f})")
    beliefs = {
        "2026 == 2025 same month": {1: b[1], 7: b[7]},
        "2026 fill = S10 calibrated (0.589 / 0.290)": {1: a + c * 0.589, 7: a + c * 0.290},
        "2026 fill = 2025 same month (line)": {1: a + c * fill[1], 7: a + c * fill[7]},
        "no bias at all (b = 0)": {1: 0.0, 7: 0.0},
        "2026 looks like Nov/Dec (b = -777)": {1: -777.0, 7: -777.0},
    }
    print("\n  board effect of the C2 shift, dSSE = n*(S^2 - 2*S*b), negative = better:")
    for name, bb in beliefs.items():
        tot = 0.0
        parts = []
        for m in (1, 7):
            S = -shift[m]  # shift is negative; S is the size subtracted
            ds = n_scored[m] * (S**2 - 2 * S * bb[m])
            tot += ds
            parts.append(f"month {m}: b={bb[m]:+7.0f} -> {ds / per_s:+5.2f}s")
        print(f"   {name:44s}  " + "   ".join(parts) + f"   TOTAL {tot / per_s:+5.2f}s")
    print("\n  break-even true bias for each month's C2 shift (b = S/2):")
    for m in (1, 7):
        print(f"   month {m}: the shift pays only if the true 2026 bias exceeds {-shift[m] / 2:+.0f} s")


# ---------------------------------------------------------------------------------------------- S12


def s12() -> None:
    banner("S12  `month` is excluded -- but `temp_c` is in model B, and it carries the season")
    tr = pl.read_parquet(
        ROOT / "data/work/feat_train.parquet",
        columns=["ADEP_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", MS, "mvt_minus_aobt3", "temp_c"],
    ).with_columns(
        [
            pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"),
            ((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= 10)
            .cast(pl.Float64)
            .alias("f10"),
        ]
    )
    lirf = tr.filter(pl.col("mvt_minus_aobt3").is_null() & (pl.col("ADEP_mvt") == "LIRF") & (pl.col(MS) >= 1800))
    g = (
        lirf.group_by("month")
        .agg(
            [
                pl.len().alias("n"),
                pl.col("temp_c").mean().round(1).alias("temp_mean"),
                pl.col("temp_c").std().round(1).alias("temp_sd"),
                pl.col("f10").mean().round(3).alias("fill"),
            ]
        )
        .sort("month")
    )
    print(g)
    d = lirf.drop_nulls("temp_c")
    x = d["temp_c"].to_numpy()
    mt = np.array([float(d.filter(pl.col("month") == k)["temp_c"].mean()) for k in d["month"].to_list()])
    print(f"  R^2 of temp_c explained by calendar month = {1 - ((x - mt) ** 2).sum() / ((x - x.mean()) ** 2).sum():.3f}")

    def auc(y: np.ndarray, s: np.ndarray) -> float:
        o = np.argsort(s, kind="mergesort")
        y, s = y[o], s[o]
        r = np.empty(len(s))
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and s[j + 1] == s[i]:
                j += 1
            r[i : j + 1] = 0.5 * (i + j) + 1
            i = j + 1
        return float((r[y == 1].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (1 - y).sum()))

    y = d["f10"].to_numpy()
    fr = np.array([float(d.filter(pl.col("month") == k)["f10"].mean()) for k in d["month"].to_list()])
    print(f"  in-sample AUC(temp_c -> f10)          = {auc(y, -x):.4f}")
    print(f"  in-sample AUC(month fill rate -> f10) = {auc(y, fr):.4f}")
    print(f"  corr(month mean temp, month fill)     = {np.corrcoef(g['temp_mean'].to_numpy(), g['fill'].to_numpy())[0, 1]:.3f}")
    print("  -> the model is NOT informationally blind to the season. It holds a column that separates")
    print("     the fill label slightly better than the calendar month does, and does not use it this way.")


def main() -> None:
    tr = load_frame()
    print(f"feat_train rows {tr.height}")
    only = sys.argv[1:] or ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
    if "12" in only:
        s12()
    if "10" in only:
        s10()
    if "11" in only:
        s11(tr)
    for k, fn in (("1", s1), ("2", s2), ("3", s3), ("4", s4), ("5", s5), ("6", s6), ("7", s7), ("8", s8), ("9", s9)):
        if k in only:
            fn(tr)


if __name__ == "__main__":
    main()
