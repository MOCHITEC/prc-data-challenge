"""Stamp rates: leakage-free 'BLOCK = SCHED' (schedule copy) rates per airline / destination / flight number (local).

copy := |y - (MVT - SCHED)| <= 12 s on 2025 departures. Rates are counted per airport x lane (matched = has AOBT_3)
and shrunk hierarchically with prior strength K (empirical Bayes, r = (c + K r_parent) / (n + K)):
  root (apt, lane) -> airline -> airline x ADES -> flight number;  root -> ADES;  (apt) -> airline (lanes pooled);
  delayed only (MVT - SCHED > 1800 s): root_d -> airline_d -> flight_d;  plus log1p counts.
  arrival side: same-month airline arrival copy rate (|BLOCK - SCHED| <= 12 s) at the airport (labels visible in 2026).
A 2025 row never uses its own month; in a leave-one-month-out fold (held-out month m) the training rows also exclude m
(nested). 2026 rows use all of 2025.

Airline = CALLSIGN 3-letter prefix, else FLIGHT 3-letter prefix, else FLIGHT first 2 characters.

  feats            : coverage and 2025 (month-out) vs 2026 (all-2025) distribution per airport
  lomo APT LANE    : 7 OOF months, per fold a copy classifier on the pipeline features (q0) and pipeline + stamp (q1),
                     trained on the 11 other months; writes $SCRATCH/agents/stamp/q_<APT>_<LANE>.parquet
                     (APT may be 'NONLIRF' for the pooled unmatched lane of the other nine airports)
  gains            : overlay gains on cv_rows_months7_v2 (p = b + c_C), per month, A fixed 0.4 and LOMO A,
                     day-block bootstrap, top-50 concentration
  seedcmp / diag   : LIRF matched seed control (needs SEED=2 lomo LIRF m) and band / top-row diagnostics
  apply SRC A [OUT]: LIRF matched 2026: production q0 reproduced, replaced by q1 where the flight number was seen in
                     2025 (flight-unseen stress was negative); writes OUT (default rank_delta.parquet)

Result (2026-10-07): NOT adopted. LIRF matched q1 vs production q0 at A = 0.4: +44 MSE [+28, +61] over 2 seeds, 6-7/7
months positive, but top-50 share 0.81-0.89 and the gain turns negative when flight numbers are unseen (48 % of
2026 July LIRF rows). Other airports' matched lanes, LIRF unmatched and non-LIRF unmatched: no gain.
Non-LIRF matched classifiers use learning_rate 0.1 and feature_fraction 0.5 (speed).

    .venv/bin/python analysis/local_stamp_rates.py feats
    .venv/bin/python analysis/local_stamp_rates.py lomo LIRF m
    .venv/bin/python analysis/local_stamp_rates.py gains
"""
import glob
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

S = os.environ["SCRATCH"]
O0 = os.path.join(S, "agents", "stamp")
SEED = int(os.environ.get("SEED", "0"))  # seed control: SEED=k shifts model seeds and validation days, writes to seed<k>/
O = os.path.join(O0, f"seed{SEED}") if SEED else O0
os.makedirs(O, exist_ok=True)
K = float(os.environ.get("K", "20"))
MONTHS = [1, 7, 8, 9, 10, 11, 12]
N_OOF = 1233234
DROP = {"TAXITIME_SEC_mvt", "copy", "MVT_ID_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt"}
P_M = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=2, seed=1)
P_U = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=2, seed=5)
SF = ["sr_root", "sr_al", "sr_al_ades", "sr_fl", "sr_ades", "sr_al_pool", "sr_root_d", "sr_al_d", "sr_fl_d",
      "ln_al", "ln_fl", "ln_fl_d", "arr_sr_al_m", "arr_sr_m"]


def airline(df):
    return df.with_columns(pl.coalesce(pl.col("CALLSIGN_flt").str.extract(r"^([A-Z]{3})"),
                                       pl.col("FLIGHT_mvt").str.extract(r"^([A-Z]{3})"),
                                       pl.col("FLIGHT_mvt").str.slice(0, 2), pl.lit("?")).alias("al"))


def dep_rows():
    """All departures, 2025 (with copy) and 2026 (copy null), minimal columns. Cached."""
    f = os.path.join(O0, "dep_rows.parquet")
    if os.path.exists(f):
        return pl.read_parquet(f)
    cols = ["MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "mvt_minus_aobt3", "mvt_minus_sched",
            "month", "MVT_TIME_UTC_mvt"]
    T = pl.scan_parquet("data/work_flseq/feat_train.parquet").select(cols + ["TAXITIME_SEC_mvt"]).collect()
    T = T.with_columns(pl.lit(2025).alias("yr"), ((pl.col("TAXITIME_SEC_mvt") - pl.col("mvt_minus_sched")).abs() <= 12).alias("copy"))
    R = pl.read_parquet("data/work_flseq/feat_rank.parquet", columns=cols).with_columns(
        pl.lit(2026).alias("yr"), pl.lit(None, pl.Boolean).alias("copy"), pl.lit(None, pl.Int32).alias("TAXITIME_SEC_mvt"))
    D = pl.concat([T, R.select(T.columns)])
    D = airline(D).with_columns(
        pl.col("ADEP_mvt").alias("apt"), pl.when(pl.col("mvt_minus_aobt3").is_null()).then(pl.lit("u")).otherwise(pl.lit("m")).alias("lane"),
        pl.col("ADES_mvt").fill_null("?").alias("ades"), pl.col("FLIGHT_mvt").fill_null("?").alias("fl"),
        (pl.col("mvt_minus_sched") > 1800).alias("dl"), pl.col("month").cast(pl.Int32),
        pl.col("MVT_TIME_UTC_mvt").dt.date().alias("day"))
    D = D.select("MVT_ID_mvt", "yr", "month", "day", "apt", "lane", "al", "ades", "fl", "dl", "copy", "mvt_minus_sched")
    D.write_parquet(f)
    return D


def arr_rates():
    """Same-month airline arrival copy rate per airport (2025 training ARR + 2026 ranking ARR). Cached."""
    f = os.path.join(O0, "arr_rates.parquet")
    if os.path.exists(f):
        return pl.read_parquet(f)
    cols = ["ADES_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt"]
    parts = []
    for fn in sorted(glob.glob("data/raw/training_2025-*.parquet")) + ["data/raw/ranking.parquet"]:
        parts.append(pl.scan_parquet(fn).filter(pl.col("PHASE_mvt") == "ARR").select(cols).collect())
    A = airline(pl.concat(parts)).with_columns(
        ((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= 12).fill_null(False).alias("c"),
        pl.col("MVT_TIME_UTC_mvt").dt.year().cast(pl.Int32).alias("yr"), pl.col("MVT_TIME_UTC_mvt").dt.month().cast(pl.Int32).alias("month"),
        pl.col("ADES_mvt").alias("apt"))
    root = A.group_by("apt", "yr", "month").agg(arr_sr_m=pl.col("c").mean())
    g = A.group_by("apt", "yr", "month", "al").agg(c=pl.col("c").sum(), n=pl.len())
    out = g.join(root, on=["apt", "yr", "month"]).with_columns(
        ((pl.col("c") + K * pl.col("arr_sr_m")) / (pl.col("n") + K)).alias("arr_sr_al_m")).select(
        "apt", "yr", "month", "al", "arr_sr_al_m")
    out = out.join(root, on=["apt", "yr", "month"])
    out.write_parquet(f)
    return out


def stamp(D, rows, extra=None):
    """Stamp features for `rows` (subset of D with yr/month), donors = 2025 rows of D, excluding each 2025 row's own month
    and, if given, month `extra` for every row."""
    don = D.filter(pl.col("yr") == 2025)
    if extra is not None:
        don = don.filter(pl.col("month") != extra)
    don = don.with_columns(pl.col("copy").cast(pl.Int32).alias("c"), (pl.col("copy") & pl.col("dl")).cast(pl.Int32).alias("cd"),
                           pl.col("dl").cast(pl.Int32).alias("nd"))
    R = rows

    def counts(keys, tag):
        nonlocal R
        g = don.group_by(keys + ["month"]).agg(pl.col("c").sum(), pl.len().alias("n"), pl.col("cd").sum(), pl.col("nd").sum())
        tot = g.group_by(keys).agg(pl.col("c", "n", "cd", "nd").sum())
        own = g.rename({"c": "c_o", "n": "n_o", "cd": "cd_o", "nd": "nd_o"}).with_columns(pl.lit(2025).alias("yr"))
        R = (R.join(tot, on=keys, how="left").join(own, on=keys + ["yr", "month"], how="left")
             .with_columns([(pl.col(a).fill_null(0) - pl.col(a + "_o").fill_null(0)).alias(f"{a}_{tag}") for a in ("c", "n", "cd", "nd")])
             .drop(["c", "n", "cd", "nd", "c_o", "n_o", "cd_o", "nd_o"]))

    def shrink(tag, parent, out, d=False):
        nonlocal R
        c, n = (f"cd_{tag}", f"nd_{tag}") if d else (f"c_{tag}", f"n_{tag}")
        R = R.with_columns(((pl.col(c) + K * pl.col(parent)) / (pl.col(n) + K)).alias(out))

    counts(["apt", "lane"], "root")
    counts(["apt", "lane", "al"], "al")
    counts(["apt", "lane", "al", "ades"], "ala")
    counts(["apt", "lane", "al", "ades", "fl"], "fl")
    counts(["apt", "lane", "ades"], "ades")
    counts(["apt"], "apt")
    counts(["apt", "al"], "alp")
    R = R.with_columns((pl.col("c_root") / pl.col("n_root")).alias("sr_root"), (pl.col("cd_root") / pl.col("nd_root")).alias("sr_root_d"),
                       (pl.col("c_apt") / pl.col("n_apt")).alias("sr_apt"))
    shrink("al", "sr_root", "sr_al")
    shrink("ala", "sr_al", "sr_al_ades")
    shrink("fl", "sr_al_ades", "sr_fl")
    shrink("ades", "sr_root", "sr_ades")
    shrink("alp", "sr_apt", "sr_al_pool")
    shrink("al", "sr_root_d", "sr_al_d", d=True)
    shrink("fl", "sr_al_d", "sr_fl_d", d=True)
    R = R.with_columns(pl.col("n_al").log1p().alias("ln_al"), pl.col("n_fl").log1p().alias("ln_fl"), pl.col("nd_fl").log1p().alias("ln_fl_d"))
    R = R.join(arr_rates(), on=["apt", "yr", "month", "al"], how="left")
    R = R.with_columns(pl.col("arr_sr_al_m").fill_null(pl.col("arr_sr_m")))
    # fill arr_sr_m for airlines absent from arrivals
    am = arr_rates().group_by("apt", "yr", "month").agg(pl.col("arr_sr_m").first().alias("_am"))
    R = R.join(am, on=["apt", "yr", "month"], how="left").with_columns(
        pl.col("arr_sr_m").fill_null(pl.col("_am")), pl.col("arr_sr_al_m").fill_null(pl.col("_am"))).drop("_am")
    return R.select(["MVT_ID_mvt", "n_al", "n_fl", "n_ades"] + SF)


def feats():
    D = dep_rows()
    F = stamp(D, D)
    J = D.join(F, on="MVT_ID_mvt")
    J.write_parquet(os.path.join(O, "stamp_feats_single.parquet"))
    pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_float("full")
    cov = J.group_by("apt", "lane", "yr").agg(
        n=pl.len(), al_seen=(pl.col("n_al") > 0).mean().round(3), fl_seen=(pl.col("n_fl") > 0).mean().round(3),
        fl_n20=(pl.col("n_fl") >= 20).mean().round(3), ades_seen=(pl.col("n_ades") > 0).mean().round(3)).sort("apt", "lane", "yr")
    print("coverage (share of rows whose key has donor rows in other months)\n", cov)
    dist = J.group_by("apt", "lane", "yr").agg(
        *[pl.col(c).mean().round(4).alias(f"{c}_mean") for c in ("sr_al", "sr_fl", "sr_fl_d", "arr_sr_al_m")],
        *[pl.col(c).quantile(0.9).round(4).alias(f"{c}_p90") for c in ("sr_fl",)],
        copy=pl.col("copy").mean().round(4)).sort("apt", "lane", "yr")
    print("distribution 2025 (month-out) vs 2026 (all 2025)\n", dist)
    # calibration of the single feature on 2025 (month-out): copy rate by sr_fl decile, matched lanes
    cal = (J.filter(pl.col("yr") == 2025).with_columns(pl.col("sr_fl").qcut(10, allow_duplicates=True).alias("bin"))
           .group_by("apt", "lane").agg(
        auc_proxy=pl.corr("sr_fl", pl.col("copy").cast(pl.Float64)).round(3), auc_proxy_al=pl.corr("sr_al", pl.col("copy").cast(pl.Float64)).round(3),
        corr_d=pl.corr("sr_fl_d", pl.col("copy").cast(pl.Float64)).filter(pl.col("dl")).round(3),
        corr_arr=pl.corr("arr_sr_al_m", pl.col("copy").cast(pl.Float64)).round(3)).sort("apt", "lane"))
    print("correlation with copy (2025, month-out): sr_fl, sr_al, sr_fl_d among delayed, arrival airline rate\n", cal)


def frames(apt, lane):
    sel = pl.col("mvt_minus_aobt3").is_null() if lane == "u" else pl.col("mvt_minus_aobt3").is_not_null()
    asel = (pl.col("ADEP_mvt") != "LIRF") if apt == "NONLIRF" else (pl.col("ADEP_mvt") == apt)
    T = (pl.scan_parquet("data/work_flseq/feat_train.parquet").filter(sel & asel).collect()
         .with_columns(((pl.col("TAXITIME_SEC_mvt") - pl.col("mvt_minus_sched")).abs() <= 12).alias("copy")))
    R = pl.read_parquet("data/work_flseq/feat_rank.parquet").filter(sel & asel)
    num = [c for c, t in T.schema.items() if c not in DROP and t.is_numeric() and c in R.columns and T[c].null_count() < T.height]
    cat = [c for c, t in T.schema.items() if c not in DROP and t == pl.String and c in R.columns]
    codes = {c: {v: i for i, v in enumerate(sorted(set(T[c].drop_nulls().to_list()) | set(R[c].drop_nulls().to_list())))} for c in cat}

    def X(df):
        return df.select(num + [pl.col(c).replace_strict(codes[c], default=None, return_dtype=pl.Int32).alias(c) for c in cat]
                         ).to_numpy().astype(np.float32)
    return T, R, num + cat, cat, X


def fit(P, X, y, fn, cat, day, rng, rounds=None):
    if rounds:
        return lgb.train(P, lgb.Dataset(X, y, feature_name=fn, categorical_feature=cat), rounds), rounds
    ud = np.unique(day)
    va = np.isin(day, rng.choice(ud, max(1, len(ud) // 10), replace=False))
    m = lgb.train(P, lgb.Dataset(X[~va], y[~va], feature_name=fn, categorical_feature=cat), 3000,
                  valid_sets=[lgb.Dataset(X[va], y[va], feature_name=fn, categorical_feature=cat)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    return m, m.best_iteration


def mask_fl(Ft, off):
    """Flight number unseen in the donor months (as for about half of 2026 July LIRF rows): flight level falls back to
    its parent and the flight counts are zero."""
    F = Ft.copy()
    ix = {n: off + i for i, n in enumerate(SF)}
    F[:, ix["sr_fl"]] = F[:, ix["sr_al_ades"]]
    F[:, ix["sr_fl_d"]] = F[:, ix["sr_al_d"]]
    F[:, ix["ln_fl"]] = 0
    F[:, ix["ln_fl_d"]] = 0
    return F


def lomo(apt, lane):
    D = dep_rows()
    T, R, fn, cat, X = frames(apt, lane)
    P = P_U if lane == "u" else (P_M if apt == "LIRF" else dict(P_M, learning_rate=0.1, feature_fraction=0.5))  # faster off LIRF
    P = dict(P, seed=P["seed"] + 1000 * SEED)
    Xt = X(T)
    del R
    y = T["copy"].to_numpy().astype(int)
    mo = T["month"].to_numpy()
    ids = T.select("MVT_ID_mvt")
    day = T["MVT_TIME_UTC_mvt"].dt.date().cast(pl.Int32).to_numpy()
    rows = D.filter(pl.col("yr") == 2025).join(ids, on="MVT_ID_mvt", how="semi")
    out, its = [], {"q0": [], "q1": []}
    for m in MONTHS:
        rng = np.random.default_rng(100 + m + 1000 * SEED)
        tr, te = mo != m, mo == m
        F = ids.join(stamp(D, rows, extra=m), on="MVT_ID_mvt", how="left", maintain_order="left").select(SF).to_numpy().astype(np.float32)
        Ft = np.hstack([Xt, F])
        m0, i0 = fit(P, Xt[tr], y[tr], fn, cat, day[tr], rng)
        rng = np.random.default_rng(100 + m + 1000 * SEED)
        m1, i1 = fit(P, Ft[tr], y[tr], fn + SF, cat, day[tr], rng)
        its["q0"].append(i0); its["q1"].append(i1)
        # test features: own month excluded only (= the single-exclusion features)
        out.append(pl.DataFrame({"MVT_ID_mvt": T["MVT_ID_mvt"].filter(te), "month": mo[te], "copy": y[te],
                                 "ms": T["mvt_minus_sched"].filter(te), "q0": m0.predict(Xt[te], num_iteration=i0),
                                 "q1": m1.predict(Ft[te], num_iteration=i1), "q1x": m1.predict(mask_fl(Ft[te], len(fn)), num_iteration=i1)}))
        imp = sorted(zip(m1.feature_importance("gain"), fn + SF), reverse=True)
        rank = {n: r for r, (_, n) in enumerate(imp)}
        print(apt, lane, m, "iters", i0, i1, "stamp gain-rank:", {s: rank[s] for s in SF if rank[s] < 40}, flush=True)
        del F, Ft
    pl.concat(out).write_parquet(os.path.join(O, f"q_{apt}_{lane}.parquet"))
    json.dump(its, open(os.path.join(O, f"q_{apt}_{lane}.json"), "w"))


def boot(g, day, B=2000, seed=0):
    ud, inv = np.unique(day, return_inverse=True)
    s = np.bincount(inv, weights=g, minlength=len(ud))
    rng = np.random.default_rng(seed)
    bs = np.array([s[rng.integers(0, len(ud), len(ud))].sum() for _ in range(B)]) / N_OOF
    return np.percentile(bs, [2.5, 97.5])


def logloss(y, q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


def gains():
    from sklearn.metrics import roc_auc_score
    cv = pl.read_parquet(os.path.join(S, "learn", "cv_rows_months7_v2.parquet")).with_columns((pl.col("b") + pl.col("c_C")).alias("p"))
    assert cv.height == N_OOF
    D = dep_rows().select("MVT_ID_mvt", "day")
    res = {}
    for f in sorted(glob.glob(os.path.join(O, "q_*_*.parquet"))):
        apt, lane = os.path.basename(f)[2:-8].rsplit("_", 1)
        L = cv.join(pl.read_parquet(f), on="MVT_ID_mvt", how="inner").join(D, on="MVT_ID_mvt", how="left")
        if lane == "u" and apt == "LIRF":  # leave the long-delay rows to the copy-band / wrap overlays
            L = L.filter(pl.col("ms") < 6 * 3600)
        p, y, q0, q1, ms, mo, c = [L[k].to_numpy() for k in ("p", "y", "q0", "q1", "ms", "month", "copy")]
        day = L["day"].cast(pl.Int32).to_numpy()
        prod = apt == "LIRF" and lane == "m"
        base = p + 0.4 * q0 * (ms - p) if prod else p
        e0 = (base - y) ** 2
        print(f"\n=== {apt} {lane}: OOF rows {L.height}, copy rate {c.mean():.3f}; baseline = {'p + 0.4 q0 (ms - p)' if prod else 'p'}")
        print(f"  classifier: logloss q0 {logloss(c, q0):.4f} q1 {logloss(c, q1):.4f} | AUC q0 {roc_auc_score(c, q0):.4f} q1 {roc_auc_score(c, q1):.4f}"
              + "".join(f" | m{m} AUC {roc_auc_score(c[mo == m], q0[mo == m]):.3f}->{roc_auc_score(c[mo == m], q1[mo == m]):.3f}"
                        for m in MONTHS if 0 < c[mo == m].sum() < (mo == m).sum()))
        qs = [("q0", q0), ("q1", q1)]
        if "q1x" in L.columns:  # flight-number-unseen stress: all rows, and half of the flights (2026 July LIRF: 48 %)
            q1x = L["q1x"].to_numpy()
            half = (L.join(dep_rows().select("MVT_ID_mvt", "fl"), on="MVT_ID_mvt", how="left")["fl"].hash(7) % 100 < 50).to_numpy()
            qs += [("q1x", q1x), ("q1half", np.where(half, q1x, q1))]
        for qn, q in qs:
            d = q * (ms - p)
            for lab in ("A=0.4", "A lomo"):
                if prod and qn == "q0" and lab == "A=0.4":
                    continue
                pn = np.empty_like(p); As = []
                for m in MONTHS:
                    te, tr = mo == m, mo != m
                    a = 0.4 if lab == "A=0.4" else float(np.clip(np.sum(d[tr] * (y[tr] - p[tr])) / max(np.sum(d[tr] ** 2), 1e-9), 0, 1))
                    As.append(round(a, 2)); pn[te] = p[te] + a * d[te]
                g = e0 - (pn - y) ** 2
                per = [round(float(g[mo == m].sum()) / N_OOF, 1) for m in MONTHS]
                tot = float(g.sum()) / N_OOF
                top = np.sort(g)[::-1][:50].sum() / N_OOF
                lo, hi = boot(g, day)
                print(f"  {qn} {lab:7s} gain {tot:+8.1f} [{lo:+.1f}, {hi:+.1f}]  months {per}  pos {sum(x > 0 for x in per)}/7  "
                      f"top50 {top:+.1f} (share {top / tot if tot > 0 else float('nan'):.2f})  A {As}")
                res[f"{apt}_{lane}_{qn}_{lab}"] = dict(gain=tot, ci=[lo, hi], months=per, top50=top, A=As)
    json.dump(res, open(os.path.join(O, "gains.json"), "w"), indent=1)


def seedcmp():
    """LIRF matched seed control: overlay gain (A = 0.4) of each q against seed-0 q0 as the production baseline."""
    cv = pl.read_parquet(os.path.join(S, "learn", "cv_rows_months7_v2.parquet")).with_columns((pl.col("b") + pl.col("c_C")).alias("p"))
    A_ = pl.read_parquet(os.path.join(O0, "q_LIRF_m.parquet"))
    B_ = pl.read_parquet(os.path.join(O0, "seed2", "q_LIRF_m.parquet")).select("MVT_ID_mvt", *[pl.col(c).alias(c + "_s2") for c in ("q0", "q1", "q1x")])
    L = cv.join(A_, on="MVT_ID_mvt").join(B_, on="MVT_ID_mvt").join(dep_rows().select("MVT_ID_mvt", "day"), on="MVT_ID_mvt", how="left")
    p, y, ms, mo = [L[k].to_numpy() for k in ("p", "y", "ms", "month")]
    day = L["day"].cast(pl.Int32).to_numpy()
    base = (p + 0.4 * L["q0"].to_numpy() * (ms - p) - y) ** 2
    for qn in ("q0_s2", "q1", "q1_s2", "q1x_s2"):
        g = base - (p + 0.4 * L[qn].to_numpy() * (ms - p) - y) ** 2
        print(f"{qn:7s} vs q0(seed0), A=0.4: {g.sum() / N_OOF:+.1f} {np.round(boot(g, day), 1)} months "
              f"{[round(float(g[mo == m].sum()) / N_OOF, 1) for m in MONTHS]}")
    # average of the two seeds
    for a, b in (("q0", "q0_s2"), ("q1", "q1_s2")):
        qa = (L[a].to_numpy() + L[b].to_numpy()) / 2
        print(f"mean({a},{b}) A=0.4 vs q0(seed0): {(base - (p + 0.4 * qa * (ms - p) - y) ** 2).sum() / N_OOF:+.1f}")
    b2 = (p + 0.2 * (L["q0"].to_numpy() + L["q0_s2"].to_numpy()) * (ms - p) - y) ** 2
    g = b2 - (p + 0.2 * (L["q1"].to_numpy() + L["q1_s2"].to_numpy()) * (ms - p) - y) ** 2
    print(f"2-seed q1 vs 2-seed q0, A=0.4: {g.sum() / N_OOF:+.1f} {np.round(boot(g, day), 1)} months "
          f"{[round(float(g[mo == m].sum()) / N_OOF, 1) for m in MONTHS]} top50 share {np.sort(g)[::-1][:50].sum() / g.sum():.2f}")


def diag():
    """LIRF matched: production-overlay reference, where q1 differs from q0, gain by |ms - p| band, top rows."""
    from sklearn.metrics import roc_auc_score
    cv = pl.read_parquet(os.path.join(S, "learn", "cv_rows_months7_v2.parquet")).with_columns((pl.col("b") + pl.col("c_C")).alias("p"))
    D = dep_rows().select("MVT_ID_mvt", "day", "fl", "al")
    L = cv.join(pl.read_parquet(os.path.join(O, "q_LIRF_m.parquet")), on="MVT_ID_mvt").join(D, on="MVT_ID_mvt", how="left")
    p, y, q0, q1, ms, mo, c = [L[k].to_numpy() for k in ("p", "y", "q0", "q1", "ms", "month", "copy")]
    day = L["day"].cast(pl.Int32).to_numpy()
    g_prod = (p - y) ** 2 - (p + 0.4 * q0 * (ms - p) - y) ** 2
    t = g_prod.sum() / N_OOF
    print(f"reference: production overlay vs p: {t:+.1f} {boot(g_prod, day)} months "
          f"{[round(float(g_prod[mo == m].sum()) / N_OOF, 1) for m in MONTHS]} top50 share {np.sort(g_prod)[::-1][:50].sum() / N_OOF / t:.2f}")
    g = (p + 0.4 * q0 * (ms - p) - y) ** 2 - (p + 0.4 * q1 * (ms - p) - y) ** 2
    ad = np.abs(ms - p)
    for lo, hi in ((0, 120), (120, 600), (600, 1800), (1800, 3600), (3600, 1e9)):
        s_ = (ad >= lo) & (ad < hi)
        auc = (roc_auc_score(c[s_], q0[s_]), roc_auc_score(c[s_], q1[s_])) if 0 < c[s_].sum() < s_.sum() else (np.nan, np.nan)
        print(f"|ms-p| [{lo},{hi}): n {s_.sum():6d} copy {c[s_].mean():.3f} mean q0 {q0[s_].mean():.3f} q1 {q1[s_].mean():.3f} "
              f"logloss {logloss(c[s_], q0[s_]):.4f}->{logloss(c[s_], q1[s_]):.4f} AUC {auc[0]:.3f}->{auc[1]:.3f} gain(A.4) {g[s_].sum() / N_OOF:+.1f}")
    o = np.argsort(-g)[:50]
    T50 = L[o].with_columns(g=pl.Series(g[o] / N_OOF))
    print("top-50 rows: distinct days", T50["day"].n_unique(), "flights", T50["fl"].n_unique(), "airlines", T50["al"].value_counts().sort("count", descending=True).head(5).to_dicts(),
          "copies", int(T50["copy"].sum()), "months", T50["month"].value_counts().sort("month").to_dicts())
    o = np.argsort(g)[:50]
    print("bottom-50 sum", round(float(g[o].sum()) / N_OOF, 1), "; rows with |q1-q0|>0.1:", int((np.abs(q1 - q0) > 0.1).sum()))


def apply_lirf(src, A, dst="rank_delta.parquet"):
    """LIRF matched 2026: undo the production overlay (q0 refit as in local_lirf_copy_overlay apply) and apply q1."""
    D = dep_rows()
    T, R, fn, cat, X = frames("LIRF", "m")
    its = json.load(open(os.path.join(O, "q_LIRF_m.json")))
    rows = D.join(pl.concat([T.select("MVT_ID_mvt"), R.select("MVT_ID_mvt")]), on="MVT_ID_mvt", how="semi")
    F = stamp(D, rows)
    Ft = T.select("MVT_ID_mvt").join(F, on="MVT_ID_mvt", how="left", maintain_order="left").select(SF).to_numpy().astype(np.float32)
    Fr = R.select("MVT_ID_mvt").join(F, on="MVT_ID_mvt", how="left", maintain_order="left").select(SF).to_numpy().astype(np.float32)
    y = T["copy"].to_numpy().astype(int)
    # production q0: exactly the production overlay's refit (its own frames, params and rounds)
    sys.path.insert(0, "analysis")
    os.environ["APT"] = "LIRF"
    import local_lirf_copy_overlay as prod
    T0, R0, fn0, cat0, X0 = prod.frames()
    assert (R0["MVT_ID_mvt"] == R["MVT_ID_mvt"]).all()
    it0 = int(json.load(open(os.path.join(S, "learn", "copy_LIRF.json")))["best_iter"] * 1.1)
    m0 = lgb.train(prod.P, lgb.Dataset(X0(T0), T0["copy"].to_numpy().astype(int), feature_name=fn0, categorical_feature=cat0), it0)
    q0 = m0.predict(X0(R0))
    del T0, R0
    it1 = int(np.median(its["q1"]) * 1.1)
    m1, _ = fit(P_M, np.hstack([X(T), Ft]), y, fn + SF, cat, None, None, rounds=it1)
    q1 = m1.predict(np.hstack([X(R), Fr]))
    seen = R.select("MVT_ID_mvt").join(F, on="MVT_ID_mvt", how="left", maintain_order="left")["n_fl"].to_numpy() > 0
    print(f"2026 LIRF matched rows {len(q1)}, flight seen in 2025: {seen.mean():.3f}; q1 used only where seen")
    q1 = np.where(seen, q1, q0)  # flight-unseen stress was negative: keep the production q there
    sub = pl.read_parquet(src)
    G = (pl.DataFrame({"MVT_ID_mvt": R["MVT_ID_mvt"], "q0": q0, "q1": q1, "ms": R["mvt_minus_sched"]})
         .join(sub.select("MVT_ID_mvt", pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("pc")), on="MVT_ID_mvt"))
    G = G.with_columns(((pl.col("pc") - 0.4 * pl.col("q0") * pl.col("ms")) / (1 - 0.4 * pl.col("q0"))).alias("pre"))
    G = G.with_columns((pl.col("pre") + A * pl.col("q1") * (pl.col("ms") - pl.col("pre"))).round(0).alias("new"))
    G = G.with_columns(pl.when(pl.col("new") < 0).then(pl.col("pc")).otherwise(pl.col("new")).alias("new"),
                       (pl.col("new") - pl.col("pc")).alias("delta"))
    G.write_parquet(os.path.join(O, "rank_lirf_q.parquet"))
    out = G.select("MVT_ID_mvt", "delta").filter(pl.col("delta") != 0)
    out.write_parquet(os.path.join(O, dst))
    d = out["delta"].to_numpy()
    print(f"rank_delta: {len(d)} rows changed, sum delta^2/N {float((d ** 2).sum()) / sub.height:.1f}, mean (over N) "
          f"{float(d.sum()) / sub.height:.3f}, mean over changed {d.mean():.1f}; mean q0 {q0.mean():.3f} q1 {q1.mean():.3f} "
          f"(2025 rate {y.mean():.3f}); corr(q0,q1) {np.corrcoef(q0, q1)[0, 1]:.3f}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "feats":
        feats()
    elif cmd == "lomo":
        lomo(sys.argv[2], sys.argv[3])
    elif cmd == "gains":
        gains()
    elif cmd == "seedcmp":
        seedcmp()
    elif cmd == "diag":
        diag()
    elif cmd == "apply":
        apply_lirf(sys.argv[2], float(sys.argv[3]), *sys.argv[4:5])
