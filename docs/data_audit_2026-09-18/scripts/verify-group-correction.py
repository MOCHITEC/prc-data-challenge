"""Verification of review-refutations.md claim C2's decisive figure.

C2 reports: fitting group means of `err` on 2025-11 and applying them to 2025-12
(same model -- the `forward` fold), the grouping `airport x operator x hour` at k = 20
removes +1.016 s of matched RMSE on month 12, with the reverse direction +0.944 s.

This script re-implements that measurement independently and extends it:
  A  frame + fold->month attribution, i.e. which month pairs are ADMISSIBLE at all
  B  exact reproduction of +1.016 / +0.944 (two independent implementations)
  C  every ordered adjacent same-model block pair (month, half-month, 10-day, week)
  D  shrinkage (k) sweep -- plateau or single lucky setting
  E  shuffle null for EVERY configuration reported, with its spread
  F  trimming the largest residuals (caps, and the team's "drop 200 rows" rule)
  G  decomposition: global-offset part vs centred group-structure part; per-airport
     concentration; leave-one-airport-out
  H  arithmetic on the competition metric

No model training.  Group means, shuffles and arithmetic only.
POLARS_MAX_THREADS=1, lazy scans, minimal columns.  Writes nothing.

Usage:  POLARS_MAX_THREADS=1 .venv/bin/python verify-group-correction.py [A B C D E F G H]
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(220)

ROOT = "/home/user/prc-data-challenge"
FEAT = f"{ROOT}/data/work/feat_train.parquet"
OOF = f"{ROOT}/experiments/oof_main-48c19bc830.parquet"

SEC = set(x.upper() for x in sys.argv[1:]) or set("ABCDEFGHIJ")

# competition-metric constants given by the caller / from fold-shape-vs-ranking.md
N_RANK = 344_841
N_RANK_MATCHED = 339_551
N_RANK_UNMATCHED = N_RANK - N_RANK_MATCHED
LB_BEST = 284.3584
LB_MATCHED_RMSE = 218.0


def rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x)))


def codes_of(df: pl.DataFrame, keys: list[str]) -> np.ndarray:
    e = pl.concat_str([pl.col(k).cast(pl.Utf8).fill_null("<null>") for k in keys], separator="\x1f")
    return df.select(e.cast(pl.Categorical).to_physical().alias("c"))["c"].to_numpy().astype(np.int64)


def group_means(cd: np.ndarray, e: np.ndarray, fit: np.ndarray, k: float, G: int) -> np.ndarray:
    """Shrunken group means of `e` over the fit rows; groups unseen in fit get 0."""
    s = np.bincount(cd[fit], weights=e[fit], minlength=G)
    n = np.bincount(cd[fit], minlength=G).astype(float)
    return np.where(n > 0, s / (n + k), 0.0)


def gain(cd: np.ndarray, fit: np.ndarray, app: np.ndarray, k: float,
         e: np.ndarray, G: int | None = None) -> float:
    """Seconds of RMSE removed on the apply rows by the correction fitted on the fit rows."""
    G = int(cd.max()) + 1 if G is None else G
    g = group_means(cd, e, fit, k, G)
    return rmse(e[app]) - rmse(e[app] - g[cd[app]])


def gain_polars(df: pl.DataFrame, keys: list[str], fit: np.ndarray, app: np.ndarray,
                k: float) -> float:
    """Independent implementation of `gain` via polars group_by + join (no bincount)."""
    d = df.with_columns(pl.Series("err_", df["err"].to_numpy().astype(float)),
                        pl.Series("_fit", fit), pl.Series("_app", app))
    gm = (d.filter(pl.col("_fit"))
            .group_by(keys)
            .agg(pl.col("err_").sum().alias("s"), pl.len().alias("n"))
            .with_columns((pl.col("s") / (pl.col("n") + k)).alias("g"))
            .select(keys + ["g"]))
    a = d.filter(pl.col("_app")).join(gm, on=keys, how="left").with_columns(pl.col("g").fill_null(0.0))
    e0 = a["err_"].to_numpy()
    e1 = e0 - a["g"].to_numpy()
    return rmse(e0) - rmse(e1)


def load() -> pl.DataFrame:
    cols = ["MVT_ID_mvt", "TAXITIME_SEC_mvt", "month", "mvt_minus_aobt3", "MVT_TIME_UTC_mvt",
            "ADEP_mvt", "ADES_mvt", "ADES_REGION", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt",
            "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt", "FLIGHT_TYPE_flt",
            "hour_utc", "dow"]
    f = pl.scan_parquet(FEAT).select(cols)
    o = pl.scan_parquet(OOF).select(["MVT_ID_mvt", "oof"]).drop_nulls("oof")
    return (f.join(o, on="MVT_ID_mvt", how="inner")
             .with_columns((pl.col("TAXITIME_SEC_mvt").cast(pl.Float64) - pl.col("oof")).alias("err"))
             .collect())


GROUPINGS: dict[str, list[str]] = {
    "ADEP x operator x hour": ["ADEP_mvt", "AIRCRAFT_OPERATOR_flt", "hour_utc"],
    "ADEP x operator": ["ADEP_mvt", "AIRCRAFT_OPERATOR_flt"],
    "ADEP x hour": ["ADEP_mvt", "hour_utc"],
    "ADEP x actype": ["ADEP_mvt", "AIRCRAFT_TYPE_mvt"],
    "ADEP x stand": ["ADEP_mvt", "STAND_mvt"],
    "ADEP x stand x runway": ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"],
    "ADEP": ["ADEP_mvt"],
    "operator": ["AIRCRAFT_OPERATOR_flt"],
    "GLOBAL (one group)": [],
}
TOP = ["ADEP x operator x hour", "ADEP x operator", "ADEP x hour", "ADEP x actype",
       "ADEP x stand", "ADEP", "operator", "GLOBAL (one group)"]


# ---------------------------------------------------------------------------- A
print("=" * 104)
print("VERIFY  review-refutations.md C2: 'fit on 2025-11, apply to 2025-12, ADEP x operator x hour,")
print("        k = 20  ->  +1.016 s of matched RMSE'")
print("=" * 104)

df = load()
full_n = df.height
m = df.filter(pl.col("mvt_minus_aobt3").is_not_null())
err = m["err"].to_numpy().astype(float)
mo = m["month"].to_numpy().astype(int)
ts = m["MVT_TIME_UTC_mvt"].to_numpy()

if "A" in SEC:
    print("\nA. FRAME and which month pairs are admissible")
    print(f"   scored rows (non-null oof)            : {full_n}")
    print(f"   matched scored rows (proxy non-null)  : {m.height}   RMSE {rmse(err):.3f}")
    print("   per OOF month (fold attribution from investigation_2026-09-16/fold-shape-vs-ranking.md")
    print("   sec 3.1/3.3: jan->2025-01, jul->2025-07, forward->2025-11 and 2025-12):")
    for mm in sorted(set(mo.tolist())):
        s = err[mo == mm]
        fold = {1: "jan (leaky)", 7: "jul (leaky)", 11: "forward", 12: "forward"}[mm]
        print(f"     2025-{mm:02d}  n {len(s):7d}  RMSE {rmse(s):8.3f}  mean {s.mean():+8.3f}   fold {fold}")
    print("   -> the ONLY fold predicting more than one month is `forward` (2025-11, 2025-12).")
    print("      Admissible ordered adjacent same-model MONTH pairs: exactly 2 (11->12, 12->11).")
    print("      Sub-month blocks inside the forward fold are the only way to add replicates.")

clean = np.isin(mo, [11, 12])
f11, f12 = mo == 11, mo == 12

# ---------------------------------------------------------------------------- B
if "B" in SEC:
    print("\nB. EXACT REPRODUCTION of the decisive figure (two independent implementations)")
    keys = GROUPINGS["ADEP x operator x hour"]
    cd = codes_of(m, keys)
    G = int(cd.max()) + 1
    rows = []
    for k in (20.0, 50.0):
        rows.append({"k": k, "groups": G,
                     "11->12 bincount": round(gain(cd, f11, f12, k, err, G), 4),
                     "11->12 polars": round(gain_polars(m, keys, f11, f12, k), 4),
                     "12->11 bincount": round(gain(cd, f12, f11, k, err, G), 4),
                     "12->11 polars": round(gain_polars(m, keys, f12, f11, k), 4)})
    print(pl.DataFrame(rows))
    g = group_means(cd, err, f11, 20.0, G)
    e0 = err[f12]
    e1 = e0 - g[cd[f12]]
    print(f"   month 12 matched RMSE {rmse(e0):.4f} -> {rmse(e1):.4f}   gain {rmse(e0)-rmse(e1):+.4f} s")
    print(f"   apply-month rows {len(e0)};  groups seen in the fit month: "
          f"{int((np.bincount(cd[f11], minlength=G) > 0).sum())} of {G}; "
          f"apply rows falling in an unseen group: {int((np.bincount(cd[f11], minlength=G)[cd[f12]] == 0).sum())}")
    print("\n   the reviewer's 'both directions positive' count (19 groupings x 4 k = 76):")
    big = {**GROUPINGS, "ADEP x runway": ["ADEP_mvt", "RUNWAY_mvt"],
           "ADEP x segment": ["ADEP_mvt", "MARKET_SEGMENT_flt"],
           "ADEP x stand x actype": ["ADEP_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt"],
           "segment": ["MARKET_SEGMENT_flt"], "wake": ["WK_TBL_CAT_flt"],
           "flight type": ["FLIGHT_TYPE_flt"], "actype": ["AIRCRAFT_TYPE_mvt"],
           "ADES region": ["ADES_REGION"], "ADES": ["ADES_mvt"], "hour": ["hour_utc"], "dow": ["dow"]}
    del big["GLOBAL (one group)"]
    cnt = tot = 0
    for lab, kk in big.items():
        c = codes_of(m, kk)
        for k in (0.0, 20.0, 50.0, 200.0):
            a = gain(c, f11, f12, k, err)
            b = gain(c, f12, f11, k, err)
            tot += 1
            cnt += int(a > 0 and b > 0)
    print(f"     {cnt} of {tot} (grouping x k) combinations positive in BOTH directions "
          f"[review-refutations.md: 33 of 76]")


# --------------------------------------------------------------- block builder
def blocks_of(kind: str) -> tuple[np.ndarray, list[str]]:
    """Ordered contiguous time blocks inside the forward fold (2025-11 + 2025-12)."""
    day = m.select(pl.col("MVT_TIME_UTC_mvt").dt.ordinal_day().alias("d"),
                   pl.col("MVT_TIME_UTC_mvt").dt.day().alias("dom"))
    dom = day["dom"].to_numpy().astype(int)
    ordi = day["d"].to_numpy().astype(int)
    lab_idx = np.full(len(mo), -1, dtype=np.int64)
    if kind == "month":
        lab_idx[f11] = 0
        lab_idx[f12] = 1
        labels = ["2025-11", "2025-12"]
    elif kind == "half":
        for i, (mm, lo, hi) in enumerate([(11, 1, 15), (11, 16, 31), (12, 1, 15), (12, 16, 31)]):
            lab_idx[(mo == mm) & (dom >= lo) & (dom <= hi)] = i
        labels = ["Nov1-15", "Nov16-30", "Dec1-15", "Dec16-31"]
    elif kind == "tenday":
        spec = [(11, 1, 10), (11, 11, 20), (11, 21, 31), (12, 1, 10), (12, 11, 20), (12, 21, 31)]
        for i, (mm, lo, hi) in enumerate(spec):
            lab_idx[(mo == mm) & (dom >= lo) & (dom <= hi)] = i
        labels = ["Nov1-10", "Nov11-20", "Nov21-30", "Dec1-10", "Dec11-20", "Dec21-31"]
    elif kind == "week":
        start = ordi[clean].min()
        w = (ordi - start) // 7
        sel = clean & (w >= 0)
        lab_idx[sel] = w[sel]
        nw = int(w[sel].max()) + 1
        labels = [f"w{i+1}" for i in range(nw)]
    else:
        raise ValueError(kind)
    return lab_idx, labels


# ---------------------------------------------------------------------------- C
def null_spread(cd: np.ndarray, fit: np.ndarray, app: np.ndarray, k: float,
                blocks: list[np.ndarray], draws: int = 20, seed: int = 0) -> tuple[float, float]:
    """Shuffle null: permute `err` WITHIN each block (preserves each block's mean and
    marginal distribution, destroys only the group<->residual association)."""
    G = int(cd.max()) + 1
    out = []
    for s in range(draws):
        r = np.random.default_rng(1000 + seed * 97 + s)
        e = err.copy()
        for b in blocks:
            idx = np.where(b)[0]
            e[idx] = err[r.permutation(idx)]
        out.append(gain(cd, fit, app, k, e, G))
    return float(np.mean(out)), float(np.std(out))


if "C" in SEC:
    print("\nC. EVERY ordered adjacent same-model block pair (all inside the `forward` fold)")
    for kind in ("month", "half", "tenday", "week"):
        lab_idx, labels = blocks_of(kind)
        sizes = [int((lab_idx == i).sum()) for i in range(len(labels))]
        print(f"\n   granularity = {kind}: blocks " +
              ", ".join(f"{l}({n})" for l, n in zip(labels, sizes)))
        rows = []
        for gl in TOP:
            cd = codes_of(m, GROUPINGS[gl]) if GROUPINGS[gl] else np.zeros(m.height, dtype=np.int64)
            G = int(cd.max()) + 1
            rec: dict[str, object] = {"grouping": gl}
            vals = []
            for i in range(len(labels) - 1):
                a, b = lab_idx == i, lab_idx == i + 1
                fwd = gain(cd, a, b, 20.0, err, G)
                rev = gain(cd, b, a, 20.0, err, G)
                rec[f"{labels[i]}->{labels[i+1]}"] = round(fwd, 3)
                rec[f"{labels[i+1]}->{labels[i]}"] = round(rev, 3)
                vals += [fwd, rev]
            rec["mean"] = round(float(np.mean(vals)), 3)
            rec["min"] = round(float(np.min(vals)), 3)
            rec["n>0"] = f"{int(np.sum(np.array(vals) > 0))}/{len(vals)}"
            rows.append(rec)
        print(pl.DataFrame(rows))

    print("\n   fit-size control (the sub-month blocks are SMALLER fits, so compare like with like):")
    print("   ADEP x operator x hour, k = 20, fit on a random fraction of month 11 -> month 12")
    cd = codes_of(m, GROUPINGS["ADEP x operator x hour"])
    G = int(cd.max()) + 1
    rows = []
    for frac in (0.0625, 0.125, 0.25, 0.5, 1.0):
        gs = []
        for s in range(5):
            r = np.random.default_rng(11 + s)
            sub = f11 & (r.random(m.height) < frac)
            gs.append(gain(cd, sub, f12, 20.0, err, G))
        rows.append({"fit fraction of 2025-11": frac, "fit rows": int(frac * f11.sum()),
                     "gain mean (5 draws)": round(float(np.mean(gs)), 4),
                     "gain sd": round(float(np.std(gs)), 4)})
    print(pl.DataFrame(rows))

# ---------------------------------------------------------------------------- D
KS = (0.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0)
if "D" in SEC or "E" in SEC:
    print("\nD/E. SHRINKAGE SWEEP with a SHUFFLE NULL for every cell")
    print("   gain (s) fit 2025-11 -> 2025-12 | reverse | null mean +- sd (20 within-block permutations)")
    for gl in TOP:
        cd = codes_of(m, GROUPINGS[gl]) if GROUPINGS[gl] else np.zeros(m.height, dtype=np.int64)
        G = int(cd.max()) + 1
        rows = []
        for i, k in enumerate(KS):
            fwd = gain(cd, f11, f12, k, err, G)
            rev = gain(cd, f12, f11, k, err, G)
            nm, ns = null_spread(cd, f11, f12, k, [f11, f12], draws=20, seed=i)
            rows.append({"grouping": gl, "groups": G, "k": k,
                         "11->12": round(fwd, 4), "12->11": round(rev, 4),
                         "null mean": round(nm, 4), "null sd": round(ns, 4),
                         "z = (gain-null)/sd": round((fwd - nm) / ns if ns > 0 else float("nan"), 1)})
        print(pl.DataFrame(rows))

if "E" in SEC:
    print("\nE2. the reviewer's own null (permute err across the POOLED 11+12 rows) for comparison,")
    print("    ADEP x operator x hour k=20 and ADEP x operator k=50:")
    for gl, k in (("ADEP x operator x hour", 20.0), ("ADEP x operator", 50.0)):
        cd = codes_of(m, GROUPINGS[gl])
        G = int(cd.max()) + 1
        out = []
        for s in range(20):
            r = np.random.default_rng(100 + s)
            e = err.copy()
            idx = np.where(clean)[0]
            e[idx] = err[r.permutation(idx)]
            out.append(gain(cd, f11, f12, k, e, G))
        print(f"   {gl:24s} k={k:6.1f}  pooled-null {np.mean(out):+.4f} +- {np.std(out):.4f} "
              f"(max {np.max(out):+.4f})")

# ---------------------------------------------------------------------------- F
if "F" in SEC:
    print("\nF. TRIMMING the largest residuals (mask fixed by the UNCORRECTED |err|, so the row set")
    print("   does not depend on the correction).  ADEP x operator x hour, k = 20.")
    for gl, k in (("ADEP x operator x hour", 20.0), ("ADEP x operator", 50.0), ("ADEP x hour", 20.0)):
        cd = codes_of(m, GROUPINGS[gl])
        G = int(cd.max()) + 1
        for lab_dir, fit, app in (("11->12", f11, f12), ("12->11", f12, f11)):
            g = group_means(cd, err, fit, k, G)
            e0 = err[app]
            e1 = e0 - g[cd[app]]
            rows = []
            order = np.argsort(-np.abs(e0))
            for lab, keep in ([("untrimmed", np.ones(len(e0), bool))]
                              + [(f"|err| <= {c}", np.abs(e0) <= c) for c in (3600, 1800, 900, 600, 300)]
                              + [("drop top 200 |err|", ~np.isin(np.arange(len(e0)), order[:200])),
                                 ("drop top 0.1% |err|", ~np.isin(np.arange(len(e0)), order[:int(0.001 * len(e0))]))]):
                rows.append({"grouping": gl, "k": k, "dir": lab_dir, "trim": lab,
                             "rows kept": int(keep.sum()),
                             "rmse before": round(rmse(e0[keep]), 3),
                             "gain": round(rmse(e0[keep]) - rmse(e1[keep]), 4)})
            # where does the gain live: the dropped tail alone
            tail = np.abs(e0) > 600
            rows.append({"grouping": gl, "k": k, "dir": lab_dir, "trim": "ONLY |err| > 600",
                         "rows kept": int(tail.sum()), "rmse before": round(rmse(e0[tail]), 3),
                         "gain": round(rmse(e0[tail]) - rmse(e1[tail]), 4)})
            print(pl.DataFrame(rows))
    print("\n   SSE view (trim-independent): fraction of the apply block's SSE removed,")
    print("   ADEP x operator x hour k=20, 11->12, by |err| band of the uncorrected residual:")
    cd = codes_of(m, GROUPINGS["ADEP x operator x hour"])
    G = int(cd.max()) + 1
    g = group_means(cd, err, f11, 20.0, G)
    e0, e1 = err[f12], err[f12] - g[cd[f12]]
    tot0 = float(np.sum(e0 * e0))
    rows = []
    edges = [0, 100, 200, 400, 600, 900, 1800, 3600, 1e9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (np.abs(e0) >= lo) & (np.abs(e0) < hi)
        if s.sum() == 0:
            continue
        d = float(np.sum(e0[s] ** 2) - np.sum(e1[s] ** 2))
        rows.append({"|err| band": f"{lo:.0f}-{hi:.0f}" if hi < 1e8 else ">3600",
                     "n": int(s.sum()), "share of block rows": round(float(s.mean()), 4),
                     "share of block SSE": round(float(np.sum(e0[s] ** 2)) / tot0, 4),
                     "SSE removed": round(d, 1), "share of total SSE removed": round(d / (tot0 - float(np.sum(e1 * e1))), 4)})
    print(pl.DataFrame(rows))

# ---------------------------------------------------------------------------- G
if "G" in SEC:
    print("\nG. WHAT THE GAIN IS MADE OF")
    cd = codes_of(m, GROUPINGS["ADEP x operator x hour"])
    G = int(cd.max()) + 1
    for lab_dir, fit, app in (("11->12", f11, f12), ("12->11", f12, f11)):
        g = group_means(cd, err, fit, 20.0, G)
        e0 = err[app]
        adj = g[cd[app]]
        gl_ = float(err[fit].mean())
        print(f"\n   direction {lab_dir}: fit-block mean err {gl_:+.3f}, apply-block mean err {e0.mean():+.3f}")
        print(f"     full correction                : {rmse(e0) - rmse(e0 - adj):+.4f} s")
        print(f"     global-mean-only (1 group, k=0): {rmse(e0) - rmse(e0 - gl_):+.4f} s")
        print(f"     correction CENTRED on the apply block (mean removed): "
              f"{rmse(e0) - rmse(e0 - (adj - adj.mean())):+.4f} s")
        print(f"     correction mean on apply rows {adj.mean():+.4f}, sd {adj.std():.4f}")

    print("\n   per-airport decomposition of the 11->12 gain (ADEP x operator x hour, k=20):")
    g = group_means(cd, err, f11, 20.0, G)
    apt = m["ADEP_mvt"].to_numpy()[f12]
    e0, e1 = err[f12], err[f12] - g[cd[f12]]
    rows = []
    for a in sorted(set(apt.tolist())):
        s = apt == a
        rows.append({"ADEP": a, "n": int(s.sum()),
                     "rmse before": round(rmse(e0[s]), 2), "rmse after": round(rmse(e1[s]), 2),
                     "gain(own rows)": round(rmse(e0[s]) - rmse(e1[s]), 3),
                     "SSE removed": round(float(np.sum(e0[s] ** 2) - np.sum(e1[s] ** 2)), 0)})
    t = pl.DataFrame(rows).sort("SSE removed", descending=True)
    tot = float(np.sum(e0 ** 2) - np.sum(e1 ** 2))
    print(t.with_columns((pl.col("SSE removed") / tot).round(3).alias("share of removed SSE")))

    print("\n   leave-one-airport-out: gain on month 12 EXCLUDING each airport's rows")
    rows = []
    for a in sorted(set(apt.tolist())):
        keep = apt != a
        rows.append({"excluded ADEP": a, "rows": int(keep.sum()),
                     "gain": round(rmse(e0[keep]) - rmse(e1[keep]), 4)})
    print(pl.DataFrame(rows).sort("gain"))

    print("\n   cross-model transfer, for contrast (fit -> apply, ADEP x operator x hour, k=20):")
    rows = []
    for flab, fm in (("2025-01 (jan)", mo == 1), ("2025-07 (jul)", mo == 7),
                     ("2025-11 (fwd)", f11), ("2025-12 (fwd)", f12)):
        for alab, am in (("2025-01", mo == 1), ("2025-07", mo == 7), ("2025-11", f11), ("2025-12", f12)):
            if flab.startswith(alab):
                continue
            same = (flab[:7] in ("2025-11", "2025-12")) and (alab in ("2025-11", "2025-12"))
            rows.append({"fit": flab, "apply": alab, "same model": same,
                         "gain": round(gain(cd, fm, am, 20.0, err, G), 3)})
    print(pl.DataFrame(rows))

# ---------------------------------------------------------------------------- J
if "J" in SEC:
    print("\nJ. the report's secondary claim: the audit's own +0.024 s maximum is a STRUCTURE-FREE")
    print("   noise floor.  Leave-one-month-out over all four OOF months (the audit's P1 protocol):")
    rows = []
    for lab in ["GLOBAL (one group)", "ADEP", "ADEP x operator", "ADEP x operator x hour", "operator"]:
        cd = codes_of(m, GROUPINGS[lab]) if GROUPINGS[lab] else np.zeros(m.height, dtype=np.int64)
        G = int(cd.max()) + 1
        for k in (0.0, 50.0):
            adj = np.zeros_like(err)
            for mm in (1, 7, 11, 12):
                hold = mo == mm
                g = group_means(cd, err, ~hold, k, G)
                adj[hold] = g[cd[hold]]
            rows.append({"grouping": lab, "k": k, "groups": G,
                         "P1 LOMO gain (all 4 months)": round(rmse(err) - rmse(err - adj), 4)})
    print(pl.DataFrame(rows))
    print("   [matched-error.md sec 2.2: best honest gain over 32 groupings = +0.024 s;"
          " review-refutations.md: one-group LOMO = +0.0240 s]")

# ---------------------------------------------------------------------------- I
if "I" in SEC:
    print("\nI. IS THE CORRECTION TIME-LOCAL?  transfer gain vs the lag between fit and apply block")
    print("   (week blocks inside the forward fold; every ordered pair i->j, k = 20, averaged by |i-j|)")
    lab_idx, labels = blocks_of("week")
    nb = len(labels)
    for gl in ("ADEP x operator x hour", "ADEP x operator", "ADEP"):
        cd = codes_of(m, GROUPINGS[gl])
        G = int(cd.max()) + 1
        by_lag: dict[int, list[float]] = {}
        for i in range(nb):
            for j in range(nb):
                if i == j:
                    continue
                v = gain(cd, lab_idx == i, lab_idx == j, 20.0, err, G)
                by_lag.setdefault(abs(i - j), []).append(v)
        rows = []
        for lag in sorted(by_lag):
            v = np.array(by_lag[lag])
            rows.append({"grouping": gl, "lag (weeks)": lag, "n pairs": len(v),
                         "mean gain": round(float(v.mean()), 4), "sd": round(float(v.std()), 4)})
        print(pl.DataFrame(rows))
        # lag-agnostic control: fit on a RANDOM subset of the whole forward period of the
        # same size as one week block, apply to one held-out week
        gs = []
        for s in range(10):
            r = np.random.default_rng(300 + s)
            j = s % nb
            app = lab_idx == j
            pool = np.where(clean & ~app)[0]
            take = r.choice(pool, size=int((lab_idx == j).sum()), replace=False)
            fit = np.zeros(len(err), bool)
            fit[take] = True
            gs.append(gain(cd, fit, app, 20.0, err, G))
        print(f"     lag-agnostic control (random forward-period subset of one week's size -> a held-out "
              f"week): {np.mean(gs):+.4f} +- {np.std(gs):.4f}\n")

# ---------------------------------------------------------------------------- H
if "H" in SEC:
    print("\nH. ARITHMETIC ON THE COMPETITION METRIC")
    sse_tot = N_RANK * LB_BEST ** 2
    sse_m = N_RANK_MATCHED * LB_MATCHED_RMSE ** 2
    sse_u = sse_tot - sse_m
    r_u = float(np.sqrt(sse_u / N_RANK_UNMATCHED))
    print(f"   N {N_RANK} = matched {N_RANK_MATCHED} + unmatched {N_RANK_UNMATCHED}")
    print(f"   LB {LB_BEST}  ->  total SSE {sse_tot:.4e}")
    print(f"   matched at {LB_MATCHED_RMSE} -> matched SSE {sse_m:.4e}; implied unmatched RMSE {r_u:.1f}")
    print("\n   leaderboard RMSE as a function of the matched-RMSE improvement delta:")
    rows = []
    for d in (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.016, 1.5, 2.0):
        new = float(np.sqrt((N_RANK_MATCHED * (LB_MATCHED_RMSE - d) ** 2 + sse_u) / N_RANK))
        rows.append({"delta matched RMSE (s)": d, "LB RMSE": round(new, 4),
                     "LB gain (s)": round(LB_BEST - new, 4)})
    print(pl.DataFrame(rows))
    print(f"   dLB/ddelta near delta=0: "
          f"{(LB_BEST - float(np.sqrt((N_RANK_MATCHED * (LB_MATCHED_RMSE - 0.01) ** 2 + sse_u) / N_RANK))) / 0.01:.4f}"
          "  (seconds of LB per second of matched RMSE)")

print("\ndone.")
