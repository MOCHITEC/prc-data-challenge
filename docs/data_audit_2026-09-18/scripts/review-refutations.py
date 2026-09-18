"""Adversarial re-examination of three NEGATIVE findings of the 2026-09-18 data audit.

Target claims
  C1  arrival-derived features carry no usable signal   (results/arrival-rows.md §4)
  C2  the matched side has no recoverable bias left     (results/matched-error.md)
  C3  the LIRF fill splits into 12,146 harmful / 86,779 benign rows and the harmful
      kind does not generalise beyond LIRF             (results/fill-pattern.md item 3)

No model training. Group means, honest leave-one-month-out (LOMO) group-mean
subtraction, rank/AUC statistics, correlations against the MODEL RESIDUAL, threshold
sweeps, between-group variance shares against their chance term.  POLARS_MAX_THREADS=1,
lazy scans, minimal columns.  Writes nothing.

Usage:  python review-refutations.py [1] [2] [3]      (default: all three)
"""

from __future__ import annotations

import glob
import sys
import warnings

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(24)
pl.Config.set_tbl_width_chars(220)

ROOT = "/home/user/prc-data-challenge"
RAW = f"{ROOT}/data/raw"
TRAIN_FILES = sorted(glob.glob(f"{RAW}/training_*.parquet"))
FEAT = f"{ROOT}/data/work/feat_train.parquet"
OOF = f"{ROOT}/experiments/oof_main-48c19bc830.parquet"

OOF_MONTHS = [1, 7, 11, 12]
LEAKY_MONTHS = [1, 7]      # jan / jul folds train on months later than the validation month
CLEAN_MONTHS = [11, 12]    # forward fold: trains on 2025-01..10 only

SECTIONS = set(sys.argv[1:]) or {"1", "2", "3"}


# ---------------------------------------------------------------- helpers

def rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x)))


def pearson(a: np.ndarray, b: np.ndarray) -> tuple[int, float]:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30:
        return int(m.sum()), float("nan")
    a2, b2 = a[m], b[m]
    if a2.std() == 0 or b2.std() == 0:
        return int(m.sum()), float("nan")
    return int(m.sum()), float(np.corrcoef(a2, b2)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> tuple[int, float]:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30:
        return int(m.sum()), float("nan")
    ra = pl.Series(a[m]).rank("average").to_numpy().astype(float)
    rb = pl.Series(b[m]).rank("average").to_numpy().astype(float)
    return int(m.sum()), float(np.corrcoef(ra, rb)[0, 1])


def auc(score: np.ndarray, pos: np.ndarray) -> tuple[int, int, float]:
    """Mann-Whitney AUC of `score` for separating pos==1 from pos==0. NaN scores dropped."""
    m = np.isfinite(score)
    s, p = score[m], pos[m].astype(bool)
    n1, n0 = int(p.sum()), int((~p).sum())
    if n1 == 0 or n0 == 0:
        return n1, n0, float("nan")
    r = pl.Series(s).rank("average").to_numpy().astype(float)
    return n1, n0, float((r[p].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def codes_of(df: pl.DataFrame, keys: list[str]) -> np.ndarray:
    """Integer group codes for a (possibly multi-column) grouping; nulls become their own level."""
    e = pl.concat_str([pl.col(k).cast(pl.Utf8).fill_null("<null>") for k in keys], separator="\x1f")
    return df.select(e.cast(pl.Categorical).to_physical().alias("c"))["c"].to_numpy().astype(np.int64)


def lomo_gain(err: np.ndarray, month: np.ndarray, codes: np.ndarray, k: float = 0.0,
              fit_months: list[int] | None = None) -> tuple[int, float, float]:
    """Honest leave-one-month-out group-mean subtraction, the matched-error.md protocol.

    Group means of `err` are fitted on every OOF month except the held-out one and applied
    to the held-out one; `k` is the shrinkage denominator weight.  Returns
    (n groups, RMSE after, gain = RMSE before - RMSE after) over the rows scored.
    """
    G = int(codes.max()) + 1
    months = sorted(set(month.tolist()))
    adj = np.zeros_like(err)
    scored = np.zeros(len(err), dtype=bool)
    tot_s = np.bincount(codes, weights=err, minlength=G)
    tot_n = np.bincount(codes, minlength=G).astype(float)
    for m in months:
        hold = month == m
        if fit_months is not None:
            fit = np.isin(month, [x for x in fit_months if x != m])
            s = np.bincount(codes[fit], weights=err[fit], minlength=G)
            n = np.bincount(codes[fit], minlength=G).astype(float)
        else:
            s = tot_s - np.bincount(codes[hold], weights=err[hold], minlength=G)
            n = tot_n - np.bincount(codes[hold], minlength=G).astype(float)
        g = np.where(n > 0, s / (n + k), 0.0)
        adj[hold] = g[codes[hold]]
        scored |= hold
    before = rmse(err[scored])
    after = rmse((err - adj)[scored])
    return G, after, before - after


def between_share(err: np.ndarray, codes: np.ndarray) -> tuple[int, float, float]:
    """In-sample between-group share of sum of squares and the share expected by chance."""
    G = int(codes.max()) + 1
    n = np.bincount(codes, minlength=G).astype(float)
    s = np.bincount(codes, weights=err, minlength=G)
    sse = float(np.sum(err * err))
    between = float(np.sum(np.where(n > 0, s * s / np.maximum(n, 1), 0.0)))
    within_var = (sse - between) / max(len(err) - G, 1)
    chance = G * within_var / sse if sse > 0 else float("nan")
    return G, between / sse, chance


def deciles(x: np.ndarray, nbins: int = 10) -> np.ndarray:
    """Quantile bin codes; NaN gets its own bin."""
    out = np.full(len(x), nbins, dtype=np.int64)
    m = np.isfinite(x)
    if m.sum() == 0:
        return out
    q = np.unique(np.nanquantile(x[m], np.linspace(0, 1, nbins + 1)[1:-1]))
    out[m] = np.searchsorted(q, x[m], side="right")
    return out


def load_oof_frame(extra: list[str]) -> pl.DataFrame:
    base = ["MVT_ID_mvt", "TAXITIME_SEC_mvt", "ADEP_mvt", "month", "mvt_minus_aobt3"]
    cols = list(dict.fromkeys(base + extra))
    f = pl.scan_parquet(FEAT).select(cols)
    o = pl.scan_parquet(OOF).select(["MVT_ID_mvt", "oof"]).drop_nulls("oof")
    return (
        f.join(o, on="MVT_ID_mvt", how="inner")
        .with_columns((pl.col("TAXITIME_SEC_mvt").cast(pl.Float64) - pl.col("oof")).alias("err"))
        .collect()
    )


# =====================================================================================
# C3  the LIRF fill "harmful / benign" split: threshold robustness and domain of validity
# =====================================================================================

if "3" in SECTIONS:
    print("\n" + "=" * 100)
    print("C3  fill-pattern.md item 3: dfill = fill AND |BLOCK - AOBT_3| > 600 s")
    print("=" * 100)

    f3 = (
        pl.scan_parquet(FEAT)
        .select(["MVT_ID_mvt", "ADEP_mvt", "TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt",
                 "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "mvt_minus_aobt3", "month"])
        .join(pl.scan_parquet(OOF).select(["MVT_ID_mvt", "oof"]), on="MVT_ID_mvt", how="left")
        .with_columns(
            (pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs().alias("bs"),
            (pl.col("BLOCK_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().alias("ba"),
            pl.col("mvt_minus_aobt3").is_not_null().alias("matched"),
            pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y"),
        )
        .with_columns(
            (pl.col("bs") <= 10).alias("fill"),
            (pl.col("TAXITIME_SEC_mvt").cast(pl.Float64) - pl.col("oof")).alias("err"),
            (pl.col("ADEP_mvt") == "LIRF").alias("lirf"),
        )
        .collect()
    )
    print(f"\nA. frame: {f3.height} training departures, scored (oof non-null) {f3['oof'].is_not_null().sum()}")
    tot_sse = float(np.nansum(f3.filter(pl.col("oof").is_not_null())["err"].to_numpy() ** 2))
    print(f"   total OOF SSE {tot_sse:.4e}  (matched-error.md: 7.5275e10)")

    cens = (f3.group_by(["lirf", "matched"]).agg(
        n=pl.len(), fill=pl.col("fill").sum(),
        dfill=(pl.col("fill") & (pl.col("ba").abs() > 600)).sum(),
        ba_null_on_fill=(pl.col("fill") & pl.col("ba").is_null()).sum()).sort(["lirf", "matched"]))
    print("\nB. census reproduction (fill = |BLOCK-SCHED| <= 10 s, dfill = fill & |BLOCK-AOBT_3| > 600 s)")
    print(cens)

    fillm = f3.filter(pl.col("fill") & pl.col("matched"))
    print(f"\n   matched fill rows {fillm.height}; LIRF {fillm['lirf'].sum()}, non-LIRF {(~fillm['lirf']).sum()}")
    print(f"   claimed split: 12,146 harmful (6,823 LIRF + 5,323 non-LIRF) / 86,779 benign")

    print("\nC. THRESHOLD SWEEP on |BLOCK - AOBT_3| (matched fill rows only)")
    rows = []
    for T in [0, 60, 120, 300, 450, 600, 750, 900, 1200, 1800, 3600, 7200]:
        for tag, sub in (("LIRF", fillm.filter(pl.col("lirf"))), ("non-LIRF", fillm.filter(~pl.col("lirf")))):
            h = sub.filter(pl.col("ba").abs() > T)
            sc = h.filter(pl.col("oof").is_not_null())
            e = sc["err"].to_numpy() if sc.height else np.array([0.0])
            rows.append({
                "T": T, "set": tag, "n_harm": h.height,
                "share_of_fill": round(h.height / max(sub.height, 1), 4),
                "y>=2h": int((h["y"] >= 7200).sum()), "y>=1h": int((h["y"] >= 3600).sum()),
                "med_ba": None if h.height == 0 else float(h["ba"].median()),
                "n_scored": sc.height, "rmse": round(rmse(e), 1) if sc.height else None,
                "sse_share": round(float(np.sum(e * e)) / tot_sse, 5) if sc.height else None,
            })
    print(pl.DataFrame(rows).sort(["set", "T"]))

    print("\nD. is 600 s a knee?  density of |BLOCK-AOBT_3| on matched FILL rows vs matched NON-fill rows")
    edges = [0, 60, 120, 300, 600, 900, 1800, 3600, 7200, 10**9]
    for tag, sub in (("fill LIRF", fillm.filter(pl.col("lirf"))), ("fill non-LIRF", fillm.filter(~pl.col("lirf"))),
                     ("nonfill LIRF", f3.filter(pl.col("matched") & ~pl.col("fill") & pl.col("lirf"))),
                     ("nonfill non-LIRF", f3.filter(pl.col("matched") & ~pl.col("fill") & ~pl.col("lirf")))):
        v = sub["ba"].abs().drop_nulls().to_numpy()
        h = np.histogram(v, bins=edges)[0] / max(len(v), 1)
        print(f"   {tag:18s} n={len(v):7d}  " + " ".join(f"{x:.4f}" for x in h))
    print("   bands: 0-60 60-120 120-300 300-600 600-900 900-1800 1800-3600 3600-7200 >7200")

    print("\nE. non-LIRF 'benign' vs 'harmful' fill rows judged by OOF error, not by the label tail")
    rows = []
    base_nl = f3.filter(pl.col("matched") & ~pl.col("lirf") & pl.col("oof").is_not_null())
    e0 = base_nl["err"].to_numpy()
    rows.append({"cell": "non-LIRF matched, all", "n_scored": base_nl.height, "rmse": round(rmse(e0), 1),
                 "sse_share": round(float(np.sum(e0 * e0)) / tot_sse, 5), "y>=2h": int((base_nl["y"] >= 7200).sum())})
    for tag, sub in (("non-LIRF matched, not fill", f3.filter(pl.col("matched") & ~pl.col("lirf") & ~pl.col("fill"))),
                     ("non-LIRF fill, benign (|ba|<=600)", fillm.filter(~pl.col("lirf") & (pl.col("ba").abs() <= 600))),
                     ("non-LIRF fill, dfill (|ba|>600)", fillm.filter(~pl.col("lirf") & (pl.col("ba").abs() > 600))),
                     ("LIRF fill, benign", fillm.filter(pl.col("lirf") & (pl.col("ba").abs() <= 600))),
                     ("LIRF fill, dfill", fillm.filter(pl.col("lirf") & (pl.col("ba").abs() > 600)))):
        sc = sub.filter(pl.col("oof").is_not_null())
        e = sc["err"].to_numpy()
        rows.append({"cell": tag, "n_scored": sc.height, "rmse": round(rmse(e), 1),
                     "sse_share": round(float(np.sum(e * e)) / tot_sse, 5), "y>=2h": int((sub["y"] >= 7200).sum())})
    print(pl.DataFrame(rows))

    print("\nF. per-airport non-LIRF dfill: error, not label tail")
    nl = fillm.filter(~pl.col("lirf") & (pl.col("ba").abs() > 600)).filter(pl.col("oof").is_not_null())
    r = []
    for apt in sorted(nl["ADEP_mvt"].unique().to_list()):
        s = nl.filter(pl.col("ADEP_mvt") == apt)
        b = base_nl.filter(pl.col("ADEP_mvt") == apt)
        e, eb = s["err"].to_numpy(), b["err"].to_numpy()
        r.append({"apt": apt, "n_dfill_scored": s.height, "rmse_dfill": round(rmse(e), 1),
                  "rmse_apt_matched": round(rmse(eb), 1),
                  "ratio": round(rmse(e) / rmse(eb), 2),
                  "sse_share_total": round(float(np.sum(e * e)) / tot_sse, 5),
                  "med_err": round(float(np.median(e)), 1), "med_ba": round(float(s["ba"].median()), 1)})
    print(pl.DataFrame(r).sort("rmse_dfill", descending=True))

    print("\nG. the UNMATCHED rows: where the test cannot be computed at all")
    um = f3.filter(~pl.col("matched"))
    print(f"   unmatched training rows {um.height}; AOBT_3 non-null on {um['AOBT_3_flt'].is_not_null().sum()}"
          f"  -> |BLOCK-AOBT_3| computable on {um['ba'].is_not_null().sum()} of them")
    umf = um.filter(pl.col("fill"))
    print(f"   unmatched FILL rows {umf.height}  (LIRF {umf['lirf'].sum()}, non-LIRF {(~umf['lirf']).sum()})"
          f" - none can be classified as harmful/benign")
    r = []
    for tag, sub in (("LIRF unmatched fill", umf.filter(pl.col("lirf"))),
                     ("LIRF unmatched not-fill", um.filter(pl.col("lirf") & ~pl.col("fill"))),
                     ("non-LIRF unmatched fill", umf.filter(~pl.col("lirf"))),
                     ("non-LIRF unmatched not-fill", um.filter(~pl.col("lirf") & ~pl.col("fill")))):
        sc = sub.filter(pl.col("oof").is_not_null())
        e = sc["err"].to_numpy() if sc.height else np.array([np.nan])
        r.append({"cell": tag, "n": sub.height, "y_p50": float(sub["y"].median()), "y_p95": float(sub["y"].quantile(0.95)),
                  "y>=2h": int((sub["y"] >= 7200).sum()), "y>=2h_rate": round(float((sub["y"] >= 7200).mean()), 4),
                  "n_scored": sc.height, "rmse": None if not sc.height else round(rmse(e), 1),
                  "sse_share": None if not sc.height else round(float(np.sum(e * e)) / tot_sse, 5)})
    print(pl.DataFrame(r))
    print("\n   SSE accounting: share of total OOF SSE in rows the dfill test CAN classify vs CANNOT")
    can = f3.filter(pl.col("matched") & pl.col("oof").is_not_null())["err"].to_numpy()
    cannot = f3.filter(~pl.col("matched") & pl.col("oof").is_not_null())["err"].to_numpy()
    print(f"   classifiable (matched)  n={len(can):7d}  SSE share {np.sum(can*can)/tot_sse:.4f}")
    print(f"   unclassifiable (unmatched) n={len(cannot):7d}  SSE share {np.sum(cannot*cannot)/tot_sse:.4f}")
    dfa = fillm.filter(pl.col("ba").abs() > 600).filter(pl.col("oof").is_not_null())["err"].to_numpy()
    print(f"   all matched dfill       n={len(dfa):7d}  SSE share {np.sum(dfa*dfa)/tot_sse:.4f}")
    uf = umf.filter(pl.col("oof").is_not_null())["err"].to_numpy()
    print(f"   unmatched fill (unclassifiable) n={len(uf):5d}  SSE share {np.sum(uf*uf)/tot_sse:.4f}")

    print("\nH. does the FILL predicate itself survive into the ranking set?  (null structure)")
    rk = (pl.scan_parquet(f"{ROOT}/data/work/feat_rank.parquet")
          .select(["MVT_ID_mvt", "ADEP_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt",
                   "AOBT_3_flt", "SCHED_TIME_UTC_mvt", "mvt_minus_aobt3"]).collect())
    print(f"   ranking rows {rk.height}: BLOCK null {rk['BLOCK_TIME_UTC_mvt'].null_count()}, "
          f"TAXITIME null {rk['TAXITIME_SEC_mvt'].null_count()}, AOBT_3 null {rk['AOBT_3_flt'].null_count()}, "
          f"SCHED null {rk['SCHED_TIME_UTC_mvt'].null_count()}")

    print("\nE2. CONTROL the audit did not run: is the harm attributable to `fill`, or only to |BLOCK-AOBT_3|?")
    r = []
    for lab, lirf in (("LIRF", True), ("non-LIRF", False)):
        for fl in (True, False):
            for big in (True, False):
                sub = f3.filter(
                    pl.col("matched") & (pl.col("lirf") == lirf) & (pl.col("fill") == fl)
                    & ((pl.col("ba").abs() > 600) == big) & pl.col("oof").is_not_null())
                if not sub.height:
                    continue
                e = sub["err"].to_numpy()
                r.append({"set": lab, "fill": fl, "|ba|>600": big, "n_scored": sub.height,
                          "rmse": round(rmse(e), 1), "sse_share": round(float(np.sum(e * e)) / tot_sse, 5),
                          "y>=2h": int((sub["y"] >= 7200).sum())})
    print(pl.DataFrame(r))


# =====================================================================================
# C2  "the matched side has no recoverable bias left"
# =====================================================================================

if "2" in SECTIONS:
    print("\n" + "=" * 100)
    print("C2  matched-error.md: LOMO group-mean gain <= 0.024 s over 32 groupings; max |corr(err, X)| = 0.0140")
    print("=" * 100)

    schema = pl.scan_parquet(FEAT).collect_schema()
    NUMTYPES = (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt32, pl.UInt8)
    num_cols = [c for c, t in zip(schema.names(), schema.dtypes())
                if t in NUMTYPES and c not in ("MVT_ID_mvt", "FLIGHT_ID_mvt", "TAXITIME_SEC_mvt")]
    cat_cols = ["ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt",
                "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt", "FLIGHT_TYPE_flt",
                "ADES_REGION", "FLIGHT_RULE_mvt"]
    SWEPT22 = ["aobt3_minus_eobt1", "mvt_minus_aobt3", "nbr_aobt_delay_mean_30m", "mvt_minus_lobt",
               "mvt_minus_eobt1", "aobt3_minus_sched", "nbr_mvt_minus_aobt_mean_30m", "mvt_minus_sched",
               "arr_taxi_in_mean_30m", "queue_ahead", "active_runways_60m", "dep_tko_prev_30m",
               "deicing_gate", "low_vis", "dest_distance_km", "planned_flight_s", "hour_utc",
               "bank_position", "stand_rwy_share", "month", "dow", "rwy_headway_prev_s"]

    m2 = load_oof_frame(num_cols + cat_cols + ["BLOCK_TIME_UTC_mvt", "AOBT_3_flt"])
    m2 = m2.filter(pl.col("mvt_minus_aobt3").is_not_null())
    err = m2["err"].to_numpy().astype(float)
    mo = m2["month"].to_numpy().astype(int)
    print(f"\nA. matched scored rows {m2.height}, RMSE {rmse(err):.3f} (matched-error.md: 664,050 / 216.44)")
    print("   per OOF month:")
    for m in OOF_MONTHS:
        s = err[mo == m]
        print(f"     month {m:2d}  n {len(s):7d}  RMSE {rmse(s):8.3f}  mean {s.mean():8.3f}  sd {s.std():8.3f}"
              f"   fold {'LEAKY (trains on later months)' if m in LEAKY_MONTHS else 'clean forward'}")

    print("\nB. is the residual on the leaky-fold months LESS structured?  (tests matched-error.md I3 directly)")
    probe = [["ADEP_mvt"], ["ADEP_mvt", "AIRCRAFT_OPERATOR_flt"], ["ADEP_mvt", "hour_utc"],
             ["ADEP_mvt", "STAND_mvt"], ["AIRCRAFT_OPERATOR_flt"]]
    rows = []
    for keys in probe:
        for lab, mm in (("leaky 1+7", LEAKY_MONTHS), ("clean 11+12", CLEAN_MONTHS)):
            sub = m2.filter(pl.col("month").is_in(mm))
            e = sub["err"].to_numpy().astype(float)
            G, sh, ch = between_share(e, codes_of(sub, keys))
            rows.append({"grouping": "x".join(keys), "months": lab, "n": sub.height, "rmse": round(rmse(e), 2),
                         "sd_err": round(float(e.std()), 2), "groups": G,
                         "between": round(sh, 5), "chance": round(ch, 5), "excess": round(sh - ch, 5)})
    print(pl.DataFrame(rows))

    print("\nC. LOMO group-mean subtraction under four honesty protocols (gain in seconds of matched RMSE)")
    print("   P1 = all 4 OOF months, leave-one-out       (matched-error.md's protocol)")
    print("   P2 = P1 but SCORED only on the clean months 11+12")
    print("   P3 = fit on month 11 -> apply to month 12, and fit on 12 -> apply to 11 (clean rows only)")
    print("   P4 = strictly forward: fit on month 11 -> apply to month 12 only")
    GROUPINGS = [
        ["FLIGHT_TYPE_flt"], ["MARKET_SEGMENT_flt"], ["WK_TBL_CAT_flt"], ["dow"], ["hour_utc"],
        ["ADEP_mvt"], ["AIRCRAFT_OPERATOR_flt"], ["AIRCRAFT_TYPE_mvt"], ["ADES_REGION"], ["ADES_mvt"],
        ["ADEP_mvt", "hour_utc"], ["ADEP_mvt", "AIRCRAFT_OPERATOR_flt"], ["ADEP_mvt", "STAND_mvt"],
        ["ADEP_mvt", "RUNWAY_mvt"], ["ADEP_mvt", "AIRCRAFT_TYPE_mvt"], ["ADEP_mvt", "MARKET_SEGMENT_flt"],
        ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"],
    ]
    clean_mask = np.isin(mo, CLEAN_MONTHS)
    rows = []
    for keys in GROUPINGS:
        cd = codes_of(m2, keys)
        for k in (0.0, 50.0):
            G, after, gain = lomo_gain(err, mo, cd, k=k)
            # P2: same fits, scored on clean months only
            adj = np.zeros_like(err)
            for m in OOF_MONTHS:
                hold = mo == m
                s = np.bincount(cd[~hold], weights=err[~hold], minlength=int(cd.max()) + 1)
                n = np.bincount(cd[~hold], minlength=int(cd.max()) + 1).astype(float)
                g = np.where(n > 0, s / (n + k), 0.0)
                adj[hold] = g[cd[hold]]
            p2 = rmse(err[clean_mask]) - rmse((err - adj)[clean_mask])
            _, _, p3 = lomo_gain(err[clean_mask], mo[clean_mask], cd[clean_mask], k=k)
            # P4: fit on 11, apply to 12
            f = mo == 11
            t = mo == 12
            s = np.bincount(cd[f], weights=err[f], minlength=int(cd.max()) + 1)
            n = np.bincount(cd[f], minlength=int(cd.max()) + 1).astype(float)
            g = np.where(n > 0, s / (n + k), 0.0)
            p4 = rmse(err[t]) - rmse(err[t] - g[cd[t]])
            rows.append({"grouping": "x".join(keys), "k": k, "groups": G,
                         "P1_gain": round(gain, 4), "P2_gain": round(p2, 4),
                         "P3_gain": round(p3, 4), "P4_gain": round(p4, 4)})
    tb = pl.DataFrame(rows)
    print(tb.sort("P1_gain", descending=True))
    print("\n   best gain per protocol over all groupings x both k:")
    for c in ("P1_gain", "P2_gain", "P3_gain", "P4_gain"):
        b = tb.sort(c, descending=True).row(0, named=True)
        print(f"     {c}: {b[c]:+.4f} s  ({b['grouping']}, k={b['k']})   "
              f"n positive: {int((tb[c] > 0).sum())}/{tb.height}")

    print("\nD. THE COLUMN SWEEP.  |corr(err, X)| over EVERY numeric column, not 22")
    corrs = []
    for c in num_cols:
        if c in ("oof", "err"):
            continue
        x = m2[c].cast(pl.Float64).to_numpy()
        n, r = pearson(x, err)
        n2, rs = spearman(x, err)
        corrs.append({"col": c, "in_audit_22": c in SWEPT22, "n": n,
                      "pearson": None if not np.isfinite(r) else round(r, 5),
                      "spearman": None if not np.isfinite(rs) else round(rs, 5)})
    cf = pl.DataFrame(corrs).with_columns(pl.col("pearson").abs().alias("abs_p"))
    print(f"   columns swept here: {cf.height}   of which in the audit's 22: {int(cf['in_audit_22'].sum())}")
    print("   top 15 by |pearson|:")
    print(cf.sort("abs_p", descending=True, nulls_last=True).head(15))
    nsw = cf.filter(~pl.col("in_audit_22")).sort("abs_p", descending=True, nulls_last=True)
    print(f"\n   max |pearson| among the {nsw.height} columns NOT in the audit's 22: {nsw['abs_p'][0]}"
          f"  ({nsw['col'][0]})")
    print(f"   max |pearson| among the audit's 22 here: "
          f"{cf.filter(pl.col('in_audit_22')).sort('abs_p', descending=True, nulls_last=True)['abs_p'][0]}")
    print("   columns NOT in the audit's 22 (names):")
    print("   " + ", ".join(nsw["col"].to_list()))

    print("\nE. NONPARAMETRIC version of the same sweep: honest LOMO decile-mean gain, every numeric column")
    rows = []
    for c in num_cols:
        if c in ("oof", "err"):
            continue
        x = m2[c].cast(pl.Float64).to_numpy()
        if np.sum(np.isfinite(x)) < 1000:
            continue
        d = deciles(x, 10)
        if d.max() == d.min():
            continue
        for keys_lab, cd in (("decile", d), ("ADEP x decile", codes_of(m2, ["ADEP_mvt"]) * 12 + d)):
            G, after, gain = lomo_gain(err, mo, cd.astype(np.int64), k=50.0)
            _, sh, ch = between_share(err, cd.astype(np.int64))
            rows.append({"col": c, "grouping": keys_lab, "in_audit_22": c in SWEPT22, "groups": G,
                         "lomo_gain_k50": round(gain, 4), "between": round(sh, 5),
                         "chance": round(ch, 5), "excess": round(sh - ch, 5)})
    ef = pl.DataFrame(rows)
    print("   top 20 by honest LOMO gain (bar to beat: matched-error.md's +0.024 s):")
    print(ef.sort("lomo_gain_k50", descending=True).head(20))
    print(f"\n   positive gains: {int((ef['lomo_gain_k50'] > 0).sum())} of {ef.height};"
          f" > 0.024 s: {int((ef['lomo_gain_k50'] > 0.024).sum())}; > 0.1 s: {int((ef['lomo_gain_k50'] > 0.1).sum())}")
    print("   top 10 by between-group EXCESS (in-sample structure the correlation sweep cannot see):")
    print(ef.sort("excess", descending=True).head(10))

    print("\nF. does a group MEAN understate what a tree could do with the same grouping?")
    print("   honest LOMO with the grouping INTERACTED with the model's own prediction decile")
    oof_d = deciles(m2["oof"].to_numpy().astype(float), 10)
    px_d = deciles(m2["mvt_minus_aobt3"].to_numpy().astype(float), 10)
    rows = []
    for keys in [["ADEP_mvt"], ["AIRCRAFT_OPERATOR_flt"], ["ADEP_mvt", "STAND_mvt"],
                 ["MARKET_SEGMENT_flt"], ["ADEP_mvt", "hour_utc"]]:
        base = codes_of(m2, keys)
        for lab, cd in (("plain", base),
                        ("x oof decile", base * 12 + oof_d),
                        ("x proxy decile", base * 12 + px_d)):
            G, after, gain = lomo_gain(err, mo, cd.astype(np.int64), k=50.0)
            rows.append({"grouping": "x".join(keys) + " " + lab, "groups": G,
                         "rmse_after": round(after, 3), "gain": round(gain, 4)})
    print(pl.DataFrame(rows))

    print("\nG. WHY the protocol matters: the four OOF months come from THREE DIFFERENT MODELS")
    print("   fold-shape-vs-ranking.md: month 1 = `jan` fold model, month 7 = `jul` fold model,")
    print("   months 11+12 = the `forward` fold model.  LOMO across months therefore fits a")
    print("   correction on one model's residuals and applies it to another model's residuals.")
    print("   Within-fold transfer (11 <-> 12, same model) vs cross-fold transfer:")

    def fit_apply(cd: np.ndarray, fit_mask: np.ndarray, app_mask: np.ndarray, k: float = 50.0,
                  e: np.ndarray | None = None) -> float:
        e = err if e is None else e
        G = int(cd.max()) + 1
        s = np.bincount(cd[fit_mask], weights=e[fit_mask], minlength=G)
        n = np.bincount(cd[fit_mask], minlength=G).astype(float)
        g = np.where(n > 0, s / (n + k), 0.0)
        return rmse(e[app_mask]) - rmse(e[app_mask] - g[cd[app_mask]])

    rows = []
    for keys in [["ADEP_mvt"], ["AIRCRAFT_OPERATOR_flt"], ["ADEP_mvt", "AIRCRAFT_OPERATOR_flt"],
                 ["ADEP_mvt", "hour_utc"], ["ADEP_mvt", "STAND_mvt"], ["ADEP_mvt", "AIRCRAFT_TYPE_mvt"],
                 ["ADEP_mvt", "RUNWAY_mvt"], ["MARKET_SEGMENT_flt"], ["ADEP_mvt", "MARKET_SEGMENT_flt"],
                 ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]]:
        cd = codes_of(m2, keys)
        rows.append({
            "grouping": "x".join(keys),
            "11->12 (same model)": round(fit_apply(cd, mo == 11, mo == 12), 4),
            "12->11 (same model)": round(fit_apply(cd, mo == 12, mo == 11), 4),
            "7->12 (cross model)": round(fit_apply(cd, mo == 7, mo == 12), 4),
            "1->12 (cross model)": round(fit_apply(cd, mo == 1, mo == 12), 4),
            "1+7->11+12 (cross)": round(fit_apply(cd, np.isin(mo, [1, 7]), clean_mask), 4),
            "11+12->1+7 (cross)": round(fit_apply(cd, clean_mask, np.isin(mo, [1, 7])), 4),
        })
    print(pl.DataFrame(rows))

    print("\n   controls for the within-fold gain (grouping ADEP x operator, k=50):")
    cd = codes_of(m2, ["ADEP_mvt", "AIRCRAFT_OPERATOR_flt"])
    rng = np.random.default_rng(0)
    half = rng.random(m2.height) < 0.5
    print(f"     random half of 11+12 -> other half (same model, same months): "
          f"{fit_apply(cd, clean_mask & half, clean_mask & ~half):+.4f} s")
    perm = []
    for seed in range(20):
        r = np.random.default_rng(100 + seed)
        sh = err.copy()
        idx = np.where(clean_mask)[0]
        p = idx.copy()
        r.shuffle(p)
        sh[idx] = err[p]
        perm.append(fit_apply(cd, mo == 11, mo == 12, e=sh))
    print(f"     month-11-fit gain with err SHUFFLED inside 11+12, 20 draws: "
          f"mean {np.mean(perm):+.4f}, sd {np.std(perm):.4f}, max {np.max(perm):+.4f} s")
    for frac in (0.25, 0.5, 1.0):
        r = np.random.default_rng(7)
        sub = (r.random(m2.height) < frac) & (mo == 11)
        print(f"     fit on {frac:.0%} of month 11 ({int(sub.sum())} rows) -> month 12: "
              f"{fit_apply(cd, sub, mo == 12):+.4f} s")

    print("\n   same-model (11->12) gain for the whole grouping list and both k, sorted:")
    rows = []
    for keys in GROUPINGS + [["ADEP_mvt", "AIRCRAFT_OPERATOR_flt", "hour_utc"],
                             ["ADEP_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt"]]:
        cd = codes_of(m2, keys)
        for k in (0.0, 20.0, 50.0, 200.0):
            rows.append({"grouping": "x".join(keys), "k": k,
                         "gain_11to12": round(fit_apply(cd, mo == 11, mo == 12, k=k), 4),
                         "gain_12to11": round(fit_apply(cd, mo == 12, mo == 11, k=k), 4)})
    g2 = pl.DataFrame(rows)
    print(g2.sort("gain_11to12", descending=True).head(20))
    print(f"   positive in BOTH directions: "
          f"{int(((g2['gain_11to12'] > 0) & (g2['gain_12to11'] > 0)).sum())} of {g2.height}")

    print("\n   numeric-column deciles under the same-model protocol (11 -> 12), top 15:")
    rows = []
    apt_c = codes_of(m2, ["ADEP_mvt"])
    for c in num_cols:
        if c in ("oof", "err", "month"):
            continue
        x = m2[c].cast(pl.Float64).to_numpy()
        if np.sum(np.isfinite(x)) < 1000:
            continue
        d = deciles(x, 10)
        if d.max() == d.min():
            continue
        rows.append({"col": c, "in_audit_22": c in SWEPT22,
                     "gain_dec_11to12": round(fit_apply(d.astype(np.int64), mo == 11, mo == 12), 4),
                     "gain_aptdec_11to12": round(fit_apply((apt_c * 12 + d).astype(np.int64), mo == 11, mo == 12), 4)})
    df2 = pl.DataFrame(rows)
    print(df2.sort("gain_aptdec_11to12", descending=True).head(15))

    print("\nH. size and robustness of the within-model gain")
    cd = codes_of(m2, ["ADEP_mvt", "AIRCRAFT_OPERATOR_flt", "hour_utc"])
    G = int(cd.max()) + 1
    s = np.bincount(cd[mo == 11], weights=err[mo == 11], minlength=G)
    n = np.bincount(cd[mo == 11], minlength=G).astype(float)
    g = np.where(n > 0, s / (n + 20.0), 0.0)
    t = mo == 12
    e0, e1 = err[t], err[t] - g[cd[t]]
    print(f"   month 12, ADEP x operator x hour, k=20:  RMSE {rmse(e0):.3f} -> {rmse(e1):.3f}"
          f"  ({rmse(e0) - rmse(e1):+.4f} s)")
    # trimmed, to show it is not one tail row
    for cap in (600, 900, 1800):
        keep = np.abs(e0) <= cap
        print(f"     restricted to |err| <= {cap} ({keep.sum()} of {len(e0)} rows): "
              f"{rmse(e0[keep]) - rmse(e1[keep]):+.4f} s")
    # what the same relative SSE reduction would be worth on the FULL scored set
    all_sc = load_oof_frame([])
    ea = all_sc["err"].to_numpy().astype(float)
    tot = float(np.sum(ea * ea))
    mtot = float(np.sum(err * err))
    frac = 1.0 - (rmse(e1) / rmse(e0)) ** 2
    print(f"   matched SSE share of total {mtot / tot:.4f}; a {frac:.4%} matched-SSE reduction moves total RMSE "
          f"{rmse(ea):.3f} -> {np.sqrt((tot - mtot * frac) / len(ea)):.3f}")
