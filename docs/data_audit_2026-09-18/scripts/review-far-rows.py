"""Adversarial review of C3's "far-not-filled" set (2026-09-18).

Claim under review (docs/data_audit_2026-09-18/results/review-refutations.md, C3):
  39,224 scored matched rows that are non-LIRF, NOT sched-filled, and have
  |BLOCK - AOBT_3| > 600 s carry 12.65 % of total OOF SSE at RMSE 492.7 with 29
  labels >= 2 h, against 1,871 far-AND-filled non-LIRF rows at 0.81 % / RMSE 569.6.

No model training. Reads only (read-only):
  experiments/oof_main-48c19bc830.parquet   (MVT_ID_mvt, oof)
  data/work/feat_train.parquet              (2,085,047 DEP rows)
  data/work/feat_rank.parquet               (344,841 DEP rows)

Definitions (plan.md 2026-09-16 section 0):
  y     = TAXITIME_SEC_mvt   = MVT - BLOCK
  proxy = mvt_minus_aobt3    = MVT - AOBT_3
  d     = y - proxy          = AOBT_3 - BLOCK          ("far" = |d| > T)
  ms    = mvt_minus_sched    = MVT - SCHED
  fill  = |BLOCK - SCHED| <= 10 s  ==  |ms - y| <= 10 s (identity verified in fill-pattern.md)
  err   = y - oof            (positive = model under-predicts)
  matched = mvt_minus_aobt3 not null (routing definition)

Run: POLARS_MAX_THREADS=1 .venv/bin/python docs/data_audit_2026-09-18/scripts/review-far-rows.py
"""
import os

os.environ.setdefault("POLARS_MAX_THREADS", "1")
import math

import numpy as np
import polars as pl

ROOT = "/home/user/prc-data-challenge"
OOF = f"{ROOT}/experiments/oof_main-48c19bc830.parquet"
TRAIN = f"{ROOT}/data/work/feat_train.parquet"
RANK = f"{ROOT}/data/work/feat_rank.parquet"

T_MAIN = 600.0
COLS = [
    "MVT_ID_mvt", "ADEP_mvt", "AIRCRAFT_OPERATOR_flt", "TAXITIME_SEC_mvt",
    "mvt_minus_aobt3", "mvt_minus_sched", "aobt3_minus_sched", "aobt3_minus_eobt1",
    "mvt_minus_eobt1", "mvt_minus_lobt", "mvt_minus_iobt", "lobt_minus_eobt1",
    "month", "hour_utc", "queue_ahead", "nbr_aobt_delay_mean_30m",
    "nbr_mvt_minus_aobt_mean_30m", "arr_taxi_in_mean_30m", "dep_tko_prev_30m",
    "active_runways_60m", "planned_flight_s", "dest_distance_km",
    "MARKET_SEGMENT_flt", "AIRCRAFT_TYPE_mvt", "STAND_mvt",
]


def banner(s):
    print("\n" + "=" * 100)
    print(s)
    print("=" * 100)


def load():
    oof = pl.scan_parquet(OOF).filter(pl.col("oof").is_not_null())
    tr = pl.scan_parquet(TRAIN).select(COLS)
    df = (
        tr.join(oof, on="MVT_ID_mvt", how="inner")
        .with_columns(
            y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64),
            proxy=pl.col("mvt_minus_aobt3").cast(pl.Float64),
            ms=pl.col("mvt_minus_sched").cast(pl.Float64),
        )
        .with_columns(
            err=pl.col("y") - pl.col("oof"),
            d=pl.col("y") - pl.col("proxy"),
            matched=pl.col("mvt_minus_aobt3").is_not_null(),
            fill=((pl.col("ms") - pl.col("y")).abs() <= 10),
            lirf=(pl.col("ADEP_mvt") == "LIRF"),
        )
        .with_columns(absd=pl.col("d").abs())
        .collect()
    )
    return df


def sse(x):
    return float(np.sum(np.asarray(x, dtype=np.float64) ** 2))


def stats(sub, total_sse):
    e = sub["err"].to_numpy().astype(np.float64)
    n = len(e)
    if n == 0:
        return dict(n=0, rmse=float("nan"), share=0.0, mean=float("nan"),
                    median=float("nan"), bias_share=float("nan"), y2h=0)
    s = float(np.sum(e ** 2))
    return dict(
        n=n,
        rmse=math.sqrt(s / n),
        share=s / total_sse,
        mean=float(e.mean()),
        median=float(np.median(e)),
        bias_share=float(e.mean() ** 2 / (s / n)),
        y2h=int((sub["y"].to_numpy() >= 7200).sum()),
    )


def main():
    df = load()
    total_sse = sse(df["err"].to_numpy())
    n_all = df.height
    banner("0. FRAME")
    print(f"scored rows {n_all}  total OOF SSE {total_sse:.4e}  "
          f"RMSE {math.sqrt(total_sse / n_all):.3f}")
    m = df.filter(pl.col("matched"))
    u = df.filter(~pl.col("matched"))
    print(f"matched {m.height} RMSE {math.sqrt(sse(m['err']) / m.height):.3f} "
          f"SSE share {sse(m['err']) / total_sse:.4f}")
    print(f"unmatched {u.height} RMSE {math.sqrt(sse(u['err']) / u.height):.3f} "
          f"SSE share {sse(u['err']) / total_sse:.4f}")

    # ---------------- Q0: reproduce the 2x2 --------------------------------
    banner("Q0. 2x2 reproduction (matched scored rows), T = 600 s")
    print(f"{'cell':<34}{'n':>9}{'RMSE':>10}{'SSE share':>12}{'mean err':>11}"
          f"{'median':>9}{'bias sh':>9}{'y>=2h':>7}")
    cells = {}
    for lirf in (True, False):
        for fl in (True, False):
            for far in (True, False):
                sub = m.filter((pl.col("lirf") == lirf) & (pl.col("fill") == fl)
                               & ((pl.col("absd") > T_MAIN) == far))
                key = ("LIRF" if lirf else "nonLIRF", "fill" if fl else "nofill",
                       "far" if far else "near")
                st = stats(sub, total_sse)
                cells[key] = st
                print(f"{'/'.join(key):<34}{st['n']:>9}{st['rmse']:>10.1f}"
                      f"{st['share']:>12.4f}{st['mean']:>11.1f}{st['median']:>9.1f}"
                      f"{st['bias_share']:>9.4f}{st['y2h']:>7}")
    TARGET = m.filter((~pl.col("lirf")) & (~pl.col("fill")) & (pl.col("absd") > T_MAIN))

    # ---------------- Q1: is it just large |d| ? ---------------------------
    banner("Q1. Does 'not-filled / non-LIRF' add anything beyond |d| ?")
    far_all = m.filter(pl.col("absd") > T_MAIN)
    st = stats(far_all, total_sse)
    print(f"ALL matched |d| > 600 s: n {st['n']}  RMSE {st['rmse']:.1f}  "
          f"SSE share {st['share']:.4f}  y>=2h {st['y2h']}")
    print(f"  of which the claimed set is {TARGET.height / st['n']:.4f} of rows and "
          f"{cells[('nonLIRF', 'nofill', 'far')]['share'] / st['share']:.4f} of its SSE")
    near_all = m.filter(pl.col("absd") <= T_MAIN)
    stn = stats(near_all, total_sse)
    print(f"ALL matched |d| <= 600 s: n {stn['n']}  RMSE {stn['rmse']:.1f}  "
          f"SSE share {stn['share']:.4f}")

    print("\n(a) RMSE within |d| bands, by cell (matched scored rows)")
    bands = [(600, 900), (900, 1200), (1200, 1800), (1800, 3600), (3600, 1e9)]
    print(f"{'band (s)':<16}{'nonLIRF/nofill':>26}{'nonLIRF/fill':>26}"
          f"{'LIRF/nofill':>26}{'LIRF/fill':>26}")
    for lo, hi in bands:
        row = f"{lo}-{hi if hi < 1e8 else 'inf':<10}"
        out = []
        for lirf, fl in ((False, False), (False, True), (True, False), (True, True)):
            sub = m.filter((pl.col("lirf") == lirf) & (pl.col("fill") == fl)
                           & (pl.col("absd") > lo) & (pl.col("absd") <= hi))
            s = stats(sub, total_sse)
            out.append(f"n={s['n']:>6} rmse={s['rmse']:>7.1f}" if s["n"] else
                       f"n={0:>6} rmse={'--':>7}")
        print(f"{row:<16}" + "".join(f"{o:>26}" for o in out))

    print("\n(b) |d|-only model of squared error: 40 quantile bins of |d| over ALL "
          "matched scored rows;\n    predicted SSE of a set = sum of its rows' bin mean err^2.")
    absd = m["absd"].to_numpy().astype(np.float64)
    err = m["err"].to_numpy().astype(np.float64)
    qs = np.unique(np.quantile(absd, np.linspace(0, 1, 41)))
    binidx = np.clip(np.searchsorted(qs, absd, side="right") - 1, 0, len(qs) - 2)
    nb = len(qs) - 1
    bs = np.bincount(binidx, weights=err ** 2, minlength=nb)
    bc = np.bincount(binidx, minlength=nb).astype(np.float64)
    bmean = np.divide(bs, bc, out=np.zeros(nb), where=bc > 0)
    lirf_np = m["lirf"].to_numpy()
    fill_np = m["fill"].to_numpy()
    for name, mask in [
        ("claimed set (nonLIRF,nofill,far)", (~lirf_np) & (~fill_np) & (absd > T_MAIN)),
        ("nonLIRF,fill,far", (~lirf_np) & (fill_np) & (absd > T_MAIN)),
        ("LIRF,nofill,far", (lirf_np) & (~fill_np) & (absd > T_MAIN)),
        ("LIRF,fill,far", (lirf_np) & (fill_np) & (absd > T_MAIN)),
        ("all far", absd > T_MAIN),
        ("nonLIRF,nofill,near", (~lirf_np) & (~fill_np) & (absd <= T_MAIN)),
    ]:
        act = float(np.sum(err[mask] ** 2))
        pred = float(np.sum(bmean[binidx[mask]]))
        print(f"  {name:<34} n={int(mask.sum()):>7}  actual SSE {act:.4e}  "
              f"|d|-only predicted {pred:.4e}  ratio {act / pred:.4f}  "
              f"actual share {act / total_sse:.4f}  predicted share {pred / total_sse:.4f}")

    print("\n(c) |d|-distribution-matched control: for each claimed-set row, draw the same "
          "number of\n    rows per |d| bin from the OTHER far cells (fill or LIRF) where "
          "supply allows.")
    other = (absd > T_MAIN) & ~((~lirf_np) & (~fill_np))
    rng = np.random.default_rng(0)
    tgt_mask = (~lirf_np) & (~fill_np) & (absd > T_MAIN)
    tot_n, tot_sse_ctrl, tot_sse_tgt = 0, 0.0, 0.0
    for b in range(nb):
        want = int(((binidx == b) & tgt_mask).sum())
        pool = np.nonzero((binidx == b) & other)[0]
        if want == 0 or len(pool) == 0:
            continue
        take = min(want, len(pool))
        pick = rng.choice(pool, size=take, replace=False)
        tgt_pool = np.nonzero((binidx == b) & tgt_mask)[0]
        tpick = rng.choice(tgt_pool, size=take, replace=False)
        tot_n += take
        tot_sse_ctrl += float(np.sum(err[pick] ** 2))
        tot_sse_tgt += float(np.sum(err[tpick] ** 2))
    print(f"  matched-on-|d| pairs: n {tot_n}   claimed-set RMSE "
          f"{math.sqrt(tot_sse_tgt / tot_n):.1f}   control RMSE "
          f"{math.sqrt(tot_sse_ctrl / tot_n):.1f}")

    print("\n(d) correlation structure inside the claimed set")
    te = TARGET["err"].to_numpy().astype(np.float64)
    td = TARGET["d"].to_numpy().astype(np.float64)
    print(f"  corr(err, d) = {np.corrcoef(te, td)[0, 1]:.4f}   "
          f"corr(err, |d|) = {np.corrcoef(te, np.abs(td))[0, 1]:.4f}   "
          f"corr(|err|, |d|) = {np.corrcoef(np.abs(te), np.abs(td))[0, 1]:.4f}")
    print(f"  all matched: corr(err, d) = "
          f"{np.corrcoef(err, m['d'].to_numpy().astype(np.float64))[0, 1]:.4f}")

    # ---------------- Q2: computable at prediction time? --------------------
    banner("Q2. Is |BLOCK - AOBT_3| computable at prediction time?")
    rk = pl.scan_parquet(RANK).select(
        ["MVT_ID_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "AOBT_3_flt",
         "SCHED_TIME_UTC_mvt", "mvt_minus_aobt3", "ADEP_mvt", "aobt3_minus_sched",
         "aobt3_minus_eobt1", "mvt_minus_sched", "mvt_minus_eobt1", "lobt_minus_eobt1",
         "mvt_minus_lobt", "mvt_minus_iobt"]
    ).collect()
    print(f"ranking rows {rk.height}")
    for c in ["BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "SCHED_TIME_UTC_mvt",
              "AOBT_3_flt", "mvt_minus_aobt3"]:
        print(f"  {c:<22} null {rk[c].null_count():>8} "
              f"({rk[c].null_count() / rk.height:.4f})")
    print("  => d = y - proxy is a deterministic function of the LABEL y; "
          "BLOCK is null on every ranking row, so both 'far' and 'fill' are unobservable.")

    print("\nBest prediction-time-visible single-column proxies for the claimed set "
          "(target = nonLIRF & nofill & |d|>600, population = all matched scored rows):")
    target = tgt_mask.astype(np.int8)
    base = target.mean()
    print(f"  base rate {base:.4f}  ({int(target.sum())} of {len(target)})")
    cand = ["aobt3_minus_sched", "aobt3_minus_eobt1", "mvt_minus_aobt3", "mvt_minus_sched",
            "mvt_minus_eobt1", "mvt_minus_lobt", "mvt_minus_iobt", "lobt_minus_eobt1",
            "queue_ahead", "nbr_aobt_delay_mean_30m", "nbr_mvt_minus_aobt_mean_30m",
            "arr_taxi_in_mean_30m", "dep_tko_prev_30m", "active_runways_60m",
            "planned_flight_s", "dest_distance_km", "hour_utc"]

    def auc(x, t):
        ok = np.isfinite(x)
        x2, t2 = x[ok], t[ok]
        if t2.sum() == 0 or t2.sum() == len(t2):
            return float("nan")
        order = np.argsort(x2, kind="mergesort")
        ranks = np.empty(len(x2), dtype=np.float64)
        sx = x2[order]
        i = 0
        r = np.arange(1, len(x2) + 1, dtype=np.float64)
        while i < len(sx):
            j = i
            while j + 1 < len(sx) and sx[j + 1] == sx[i]:
                j += 1
            ranks[order[i:j + 1]] = r[i:j + 1].mean()
            i = j + 1
        npos, nneg = t2.sum(), len(t2) - t2.sum()
        return float((ranks[t2 == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))

    rows = []
    for c in cand:
        x = m[c].to_numpy().astype(np.float64)
        for xx, nm in ((x, c), (np.abs(x), f"|{c}|")):
            a = auc(xx, target)
            rows.append((nm, a, max(a, 1 - a)))
    rows.sort(key=lambda r: -r[2])
    for nm, a, ab in rows[:12]:
        print(f"  {nm:<32} AUC {a:.4f}  (|AUC-0.5| oriented {ab:.4f})")

    best_col = rows[0][0]
    print(f"\nPrecision/recall of the best oriented column ({best_col}) at sweep thresholds:")
    raw = best_col.strip("|")
    xb = m[raw].to_numpy().astype(np.float64)
    if best_col.startswith("|"):
        xb = np.abs(xb)
    sign = 1.0 if rows[0][1] > 0.5 else -1.0
    xs = sign * xb
    ok = np.isfinite(xs)
    print(f"{'threshold q':>12}{'flagged n':>12}{'precision':>12}{'recall':>10}{'F1':>9}"
          f"{'lift':>8}")
    bestf1 = (0, None)
    for q in [0.50, 0.75, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999]:
        thr = np.quantile(xs[ok], q)
        flag = ok & (xs >= thr)
        tp = int((flag & (target == 1)).sum())
        p = tp / max(flag.sum(), 1)
        r_ = tp / target.sum()
        f1 = 2 * p * r_ / (p + r_) if p + r_ > 0 else 0
        if f1 > bestf1[0]:
            bestf1 = (f1, (q, p, r_))
        print(f"{q:>12.3f}{int(flag.sum()):>12}{p:>12.4f}{r_:>10.4f}{f1:>9.4f}"
              f"{p / base:>8.2f}")
    print(f"  best F1 over the sweep: {bestf1[0]:.4f} at q={bestf1[1][0]}, "
          f"precision {bestf1[1][1]:.4f}, recall {bestf1[1][2]:.4f}")

    print("\nGroup-level precision ceiling (the strongest 'flag the cell' rule):")
    for grp in (["ADEP_mvt"], ["ADEP_mvt", "AIRCRAFT_OPERATOR_flt"],
                ["ADEP_mvt", "AIRCRAFT_OPERATOR_flt", "hour_utc"],
                ["ADEP_mvt", "STAND_mvt"]):
        g = (m.with_columns(t=pl.Series(target))
             .group_by(grp).agg(n=pl.len(), k=pl.col("t").sum())
             .with_columns(rate=pl.col("k") / pl.col("n"))
             .filter(pl.col("n") >= 50).sort("rate", descending=True))
        top = g.head(1)
        # recall obtainable by taking all cells with rate >= 2x base
        sel = g.filter(pl.col("rate") >= 2 * base)
        rec = sel["k"].sum() / target.sum()
        prec = sel["k"].sum() / max(sel["n"].sum(), 1)
        print(f"  {'x'.join(grp):<44} cells>=50 {g.height:>6}  max rate "
              f"{top['rate'][0]:.4f} (n {top['n'][0]})  cells>=2x base: "
              f"recall {rec:.4f} precision {prec:.4f}")

    # ---------------- Q3: oracle ceiling -----------------------------------
    banner("Q3. Oracle ceiling on the leaderboard metric")
    N, NM, NU = 344841, 339551, 5290
    LB = 284.3584
    sse_lb = N * LB ** 2
    rmse_m, rmse_u = 218.0, math.sqrt((sse_lb - NM * 218.0 ** 2) / NU)
    print(f"N {N} = matched {NM} + unmatched {NU}; LB {LB} -> total SSE {sse_lb:.4e}; "
          f"matched at 218.0 -> implied unmatched RMSE {rmse_u:.1f}")
    prev = TARGET.height / m.height
    n_rank = prev * NM
    mse_set = stats(TARGET, total_sse)["rmse"] ** 2
    print(f"prevalence of the claimed set among matched scored rows {prev:.5f} "
          f"-> {n_rank:.0f} ranking rows, own MSE {mse_set:.0f} "
          f"(RMSE {math.sqrt(mse_set):.1f})")
    for label, factor in [("perfect oracle (MSE -> 0)", 0.0),
                          ("halve their RMSE (MSE -> 0.25)", 0.25),
                          ("MSE -> matched baseline 218^2", 218.0 ** 2 / mse_set),
                          ("10 % MSE reduction", 0.90)]:
        new = sse_lb - n_rank * mse_set * (1 - factor)
        lb2 = math.sqrt(new / N)
        print(f"  {label:<36} LB {LB:.4f} -> {lb2:.4f}   gain {LB - lb2:+.3f} s")
    # same but applied only to the recall the best proxy achieves
    q, p, r_ = bestf1[1]
    new = sse_lb - n_rank * r_ * mse_set
    print(f"  perfect oracle on only the {r_:.1%} the best visible proxy recalls "
          f"(precision {p:.3f}) -> LB {math.sqrt(new / N):.4f}, "
          f"gain {LB - math.sqrt(new / N):+.3f} s (ignores damage to the "
          f"{1 - p:.1%} false positives)")

    # ---------------- Q4: bias or spread -----------------------------------
    banner("Q4. Bias vs spread on the claimed set")
    st = stats(TARGET, total_sse)
    print(f"n {st['n']}  RMSE {st['rmse']:.2f}  mean err {st['mean']:+.2f}  "
          f"median err {st['median']:+.2f}  bias share of MSE {st['bias_share']:.5f}")
    e = TARGET["err"].to_numpy().astype(np.float64)
    print(f"  sd {e.std(ddof=1):.2f}   p1 {np.percentile(e, 1):.0f}  p25 "
          f"{np.percentile(e, 25):.0f}  p75 {np.percentile(e, 75):.0f}  p99 "
          f"{np.percentile(e, 99):.0f}")
    print(f"  removing the global mean: RMSE {math.sqrt(((e - e.mean()) ** 2).mean()):.2f}"
          f"  (saves {st['rmse'] - math.sqrt(((e - e.mean()) ** 2).mean()):.3f} s)")
    print("\n  in-sample (oracle) group-mean removal INSIDE the set — an upper bound, "
          "no honesty correction:")
    for grp in (["ADEP_mvt"], ["ADEP_mvt", "AIRCRAFT_OPERATOR_flt"],
                ["ADEP_mvt", "AIRCRAFT_OPERATOR_flt", "hour_utc"],
                ["ADEP_mvt", "STAND_mvt"], ["month"]):
        g = TARGET.with_columns(
            gm=pl.col("err").mean().over(grp), gn=pl.len().over(grp))
        r_after = math.sqrt(float(((g["err"] - g["gm"]).to_numpy() ** 2).mean()))
        ngroups = g.select(pl.struct(grp).n_unique()).item()
        print(f"    {'x'.join(grp):<44} groups {ngroups:>6}  RMSE {st['rmse']:.2f} -> "
              f"{r_after:.2f}  ({st['rmse'] - r_after:+.2f} s, in-sample)")
    print("\n  per airport (claimed set vs that airport's own matched rows):")
    print(f"{'ADEP':<8}{'n set':>8}{'RMSE set':>11}{'mean err':>11}{'bias sh':>9}"
          f"{'apt RMSE':>11}{'ratio':>8}{'SSE share':>11}")
    for a in sorted(TARGET["ADEP_mvt"].unique().to_list()):
        s1 = stats(TARGET.filter(pl.col("ADEP_mvt") == a), total_sse)
        s2 = stats(m.filter(pl.col("ADEP_mvt") == a), total_sse)
        print(f"{a:<8}{s1['n']:>8}{s1['rmse']:>11.1f}{s1['mean']:>11.1f}"
              f"{s1['bias_share']:>9.4f}{s2['rmse']:>11.1f}"
              f"{s1['rmse'] / s2['rmse']:>8.2f}{s1['share']:>11.4f}")

    print("\n  Q4b — the same questions asked on a PREDICTION-TIME-IDENTIFIABLE set:")
    print("  rows flagged by the best visible proxy, |aobt3_minus_eobt1| >= quantile q "
          "(all matched scored rows)")
    print(f"{'q':>8}{'flagged n':>11}{'RMSE':>9}{'SSE share':>11}{'mean err':>10}"
          f"{'bias sh':>9}{'RMSE after global mean':>24}")
    for q in [0.90, 0.95, 0.98, 0.99, 0.999]:
        thr = np.quantile(xs[ok], q)
        flag = ok & (xs >= thr)
        ef = err[flag]
        r0 = math.sqrt((ef ** 2).mean())
        r1 = math.sqrt(((ef - ef.mean()) ** 2).mean())
        print(f"{q:>8.3f}{int(flag.sum()):>11}{r0:>9.1f}"
              f"{float((ef ** 2).sum()) / total_sse:>11.4f}{ef.mean():>10.1f}"
              f"{ef.mean() ** 2 / (r0 ** 2):>9.4f}{r1:>24.1f}")
    print("  (aobt3_minus_eobt1 is already a model feature — src/prc/features.py "
          "WIDE_RANGE_COLS — with corr(err, ·) = +0.0036 per matched-error.md §2.5)")

    # ---------------- Q5: threshold sweep ----------------------------------
    banner("Q5. Is the 600 s threshold load-bearing?")
    print(f"{'T (s)':>7}|{'nonLIRF & NOT filled':^44}|{'nonLIRF & filled':^44}|{'ratios':^22}")
    print(f"{'':>7}|{'n':>8}{'RMSE':>9}{'SSEshare':>11}{'y>=2h':>7}{'mean':>9}"
          f"|{'n':>8}{'RMSE':>9}{'SSEshare':>11}{'y>=2h':>7}{'mean':>9}"
          f"|{'RMSEf/nf':>11}{'SSEnf/f':>11}")
    for T in [60, 120, 180, 300, 450, 600, 900, 1200, 1800, 2400, 3600]:
        a = stats(m.filter((~pl.col("lirf")) & (~pl.col("fill")) & (pl.col("absd") > T)),
                  total_sse)
        b = stats(m.filter((~pl.col("lirf")) & (pl.col("fill")) & (pl.col("absd") > T)),
                  total_sse)
        rr = b["rmse"] / a["rmse"] if a["n"] and b["n"] else float("nan")
        sr = a["share"] / b["share"] if b["share"] else float("nan")
        print(f"{T:>7}|{a['n']:>8}{a['rmse']:>9.1f}{a['share']:>11.4f}{a['y2h']:>7}"
              f"{a['mean']:>9.1f}|{b['n']:>8}{b['rmse']:>9.1f}{b['share']:>11.4f}"
              f"{b['y2h']:>7}{b['mean']:>9.1f}|{rr:>11.3f}{sr:>11.2f}")

    print("\n  same sweep, oracle LB gain from perfect prediction on the "
          "nonLIRF-not-filled set:")
    print(f"{'T (s)':>7}{'n scored':>10}{'prevalence':>12}{'rank n':>9}{'RMSE':>9}"
          f"{'LB after':>11}{'gain':>8}")
    for T in [60, 120, 180, 300, 450, 600, 900, 1200, 1800, 2400, 3600]:
        s1 = stats(m.filter((~pl.col("lirf")) & (~pl.col("fill")) & (pl.col("absd") > T)),
                   total_sse)
        pv = s1["n"] / m.height
        nr = pv * NM
        lb2 = math.sqrt((sse_lb - nr * s1["rmse"] ** 2) / N)
        print(f"{T:>7}{s1['n']:>10}{pv:>12.5f}{nr:>9.0f}{s1['rmse']:>9.1f}"
              f"{lb2:>11.4f}{LB - lb2:>8.3f}")

    banner("DONE")


if __name__ == "__main__":
    main()
