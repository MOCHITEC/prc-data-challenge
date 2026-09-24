"""What model B loses when the flight plan is suspended, and whether anything visible stands in for it.

Read-only. Nothing is written to data/, outputs/, configs/ or src/. Reproduces every number in
docs/competitor_research_2026-09-21/results/missing-clock.md; each section prints its own tag (S1..S7).

    POLARS_MAX_THREADS=2 .venv/bin/python docs/competitor_research_2026-09-21/scripts/missing_clock.py [S1 S2 ...]

Sections (wall clock on a 4-CPU box with another training run resident):
  S1  what is still visible on an unmatched scored row, per airport            ~20 s   read-only
  S2  build the clock-substitute candidates (also proves they are computable)  ~15 s   read-only
  S3  correlation of every candidate with the clock model A is handed          ~60 s   read-only
  S4  the matched-lane counterfactual: four feature regimes, one fold          ~4 min  fits
  S5  does any candidate recover the blind model's loss                        ~25 min fits
  S6  association with the model-B OOF residual, raw and conditioned           ~4 min  small fits
  S7  pricing                                                                  ~10 s   read-only

Definitions used throughout.
  y      = TAXITIME_SEC_mvt = MVT_TIME - BLOCK_TIME               (the label)
  proxy  = mvt_minus_aobt3  = MVT_TIME - AOBT_3_flt               (model A's measured off-block clock)
  g      = y - proxy        = AOBT_3_flt - BLOCK_TIME             (what model A's tree actually fits)
  ms     = mvt_minus_sched  = MVT_TIME - SCHED_TIME
  unmatched = proxy is null = the routing rule in src/prc/train.py:398
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from prc.features import CATEGORICALS  # noqa: E402
from prc.train import (  # noqa: E402
    LABEL,
    PROXY,
    TrainConfig,
    attach_encodings,
    fit_encodings,
    visible_numeric_columns,
)

FEAT_TRAIN = ROOT / "data/work/feat_train.parquet"
FEAT_RANK = ROOT / "data/work/feat_rank.parquet"
OOF = ROOT / "experiments/oof_main_reparam-bc60017878.parquet"
OOF_FOLDS = ROOT / "experiments/oof_main_reparam-bc60017878_folds.parquet"
SCRATCH = Path("/tmp/missing_clock")
SCRATCH.mkdir(exist_ok=True)

# the four columns that go null with the flight plan but are NOT caught by model B's prefix rule
ALSO_NULL = ("stand_aobt_turn", "eobt_schedule_revision", "delay_neighbor_deviation", "planned_flight_s")
FLT_PREFIX = ("mvt_minus_aobt3", "aobt3", "lobt_minus", "mvt_minus_lobt", "mvt_minus_eobt1", "mvt_minus_iobt")
CAT_FLT = {"WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt"}
NEW4 = ["surface_occ", "nbr_aobt_minus_sched_60m", "nbr_unmatched_rate_60m", "stand_prev_dep_gap"]
WIT = ["witness_med", "witness_p90", "witness_n"]
MS_BANDS = [(-1e9, 3600), (3600, 5400), (5400, 7200), (7200, 10800), (10800, 1e9)]


# ---------------------------------------------------------------- S1
def s1_visible() -> None:
    print("\n=== S1  what is still visible on an unmatched scored row ===")
    r = pl.read_parquet(FEAT_RANK)
    u = r.filter(pl.col(PROXY).is_null())
    m = r.filter(pl.col(PROXY).is_not_null())
    print(f"scored departures {r.height}, unmatched {u.height} ({u.height / r.height:.4%}), matched {m.height}")
    print(u.group_by("ADEP_mvt").len().sort("len", descending=True))
    dead = [c for c in r.columns if u[c].null_count() == u.height]
    print(f"\n{len(dead)} columns 100% null on every unmatched scored row:")
    print("  " + ", ".join(dead))
    print("  (BLOCK_TIME_UTC_mvt and TAXITIME_SEC_mvt are blanked on matched rows too - they are the label side)")
    partial = [
        (c, u[c].null_count() / u.height, m[c].null_count() / m.height)
        for c in r.columns
        if 0 < u[c].null_count() < u.height
    ]
    print("\ncolumns partly null on unmatched (null rate unmatched / matched):")
    for c, a, b in sorted(partial, key=lambda z: -z[1]):
        print(f"  {c:30s} {a:7.4f} / {b:7.4f}")
    cols = ["AIRCRAFT_TYPE_mvt", "STAND_mvt", "RUNWAY_mvt", "ADES_mvt", "FLIGHT_mvt", "SCHED_TIME_UTC_mvt",
            "mvt_minus_sched", "stand_arrival_gap"]
    agg = [pl.len().alias("n")] + [(pl.col(c).null_count() / pl.len()).round(4).alias(c) for c in cols]
    print("\nper-airport null rate on unmatched scored rows:")
    print(u.group_by("ADEP_mvt").agg(agg).sort("n", descending=True))
    print("\nnumeric columns with <=2 distinct non-null values on the unmatched scored rows:")
    for c, dt in r.schema.items():
        if dt in (pl.String, pl.Datetime("us", "UTC")):
            continue
        s = u[c].drop_nulls()
        s = s.drop_nans() if dt in (pl.Float64, pl.Float32) else s
        if s.n_unique() <= 2:
            print(f"  {c:30s} unmatched {s.n_unique()}")
    print("\nrows with a FLIGHT_ID_mvt but no AOBT_3:", u.filter(pl.col("FLIGHT_ID_mvt").is_not_null()).height)
    print("LOBT_flt null <=> proxy null on the ranking frame:",
          bool((r["LOBT_flt"].is_null() == r[PROXY].is_null()).all()))


# ---------------------------------------------------------------- S2
def _build_candidates(path: Path, out: Path) -> None:
    """Four candidate clock-substitutes, each built only from columns that survive in ranking.parquet."""
    d = pl.read_parquet(path, columns=["MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt",
                                       "SCHED_TIME_UTC_mvt", "AOBT_3_flt", PROXY])
    d = d.with_columns([
        pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("_t"),
        pl.col("AOBT_3_flt").dt.epoch("s").cast(pl.Float64).alias("_a"),
        pl.col("SCHED_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("_s"),
    ])
    outs = []
    for apt in d["ADEP_mvt"].unique().to_list():
        pred = pl.col("ADEP_mvt").is_null() if apt is None else (pl.col("ADEP_mvt") == apt)
        g = d.filter(pred).sort("_t")
        t, a, s = g["_t"].to_numpy(), g["_a"].to_numpy(), g["_s"].to_numpy()
        # surface_occ: other departures already off-block but not yet airborne at this take-off.
        # Both sides are restricted to flights that HAVE an AOBT_3, or the difference drifts with the
        # cumulative count of unmatched rows instead of measuring occupancy.
        has = ~np.isnan(a)
        occ = (np.searchsorted(np.sort(a[has]), t, "right").astype(float)
               - np.searchsorted(np.sort(t[has]), t, "right").astype(float))
        # neighbours' actual off-block lateness against the SCHEDULE, window [t-3660, t-60)
        lat = a - s
        ok = ~np.isnan(lat)
        to, lo_ = t[ok], lat[ok]
        cs = np.concatenate([[0.0], np.cumsum(lo_)])
        hi = np.searchsorted(to, t - 60.0, "left")
        lo = np.searchsorted(to, t - 3660.0, "left")
        cnt = (hi - lo).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            nbr_lat = np.where(cnt > 0, (cs[hi] - cs[lo]) / cnt, np.nan)
        # share of the last hour's departures that have no NM record (the disruption-density idea)
        unm = g[PROXY].is_null().to_numpy().astype(float)
        cu = np.concatenate([[0.0], np.cumsum(unm)])
        h2, l2 = np.searchsorted(t, t, "left"), np.searchsorted(t, t - 3600.0, "left")
        c2 = (h2 - l2).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            nbr_unm = np.where(c2 > 0, (cu[h2] - cu[l2]) / c2, np.nan)
        # seconds since the previous departure from the same stand took off
        prev = np.full(len(t), np.nan)
        sa = np.array([x if x is not None else "" for x in g["STAND_mvt"].to_list()])
        for stand in set(sa.tolist()):
            if stand == "":
                continue
            idx = np.flatnonzero(sa == stand)
            tt = t[idx]
            prev[idx[1:]] = tt[1:] - tt[:-1]
        outs.append(pl.DataFrame({"MVT_ID_mvt": g["MVT_ID_mvt"], "surface_occ": occ,
                                  "nbr_aobt_minus_sched_60m": nbr_lat, "nbr_unmatched_rate_60m": nbr_unm,
                                  "stand_prev_dep_gap": prev}))
    pl.concat(outs).write_parquet(out)


def _build_witness(path: Path, out: Path) -> None:
    """merry-quicksand's congestion witness: airport x calendar-hour median of MVT - AOBT_3 over that
    hour's MATCHED departures, n >= 5 (results/21-repo-merry-quicksand.md §6.3)."""
    d = pl.read_parquet(path, columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", PROXY])
    d = d.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.truncate("1h").alias("_h"))
    w = (d.filter(pl.col(PROXY).is_not_null())
           .group_by(["ADEP_mvt", "_h"])
           .agg([pl.col(PROXY).median().alias("witness_med"),
                 pl.col(PROXY).quantile(0.9).alias("witness_p90"),
                 pl.len().alias("witness_n")])
           .with_columns([pl.when(pl.col("witness_n") >= 5).then(pl.col("witness_med")).alias("witness_med"),
                          pl.when(pl.col("witness_n") >= 5).then(pl.col("witness_p90")).alias("witness_p90")]))
    (d.join(w, on=["ADEP_mvt", "_h"], how="left")
       .select("MVT_ID_mvt", *WIT)
       .with_columns(pl.col("witness_n").fill_null(0))
       .write_parquet(out))


def s2_candidates() -> None:
    print("\n=== S2  build the candidate clock-substitutes and check they exist on scored unmatched rows ===")
    for src, tag in ((FEAT_TRAIN, "train"), (FEAT_RANK, "rank")):
        _build_candidates(src, SCRATCH / f"cand_{tag}.parquet")
        _build_witness(src, SCRATCH / f"wit_{tag}.parquet")
    r = (pl.read_parquet(FEAT_RANK, columns=["MVT_ID_mvt", "ADEP_mvt", PROXY])
           .join(pl.read_parquet(SCRATCH / "cand_rank.parquet"), on="MVT_ID_mvt")
           .join(pl.read_parquet(SCRATCH / "wit_rank.parquet"), on="MVT_ID_mvt"))
    u = r.filter(pl.col(PROXY).is_null())
    print(f"scored unmatched rows {u.height}; null/NaN rate of each candidate there:")
    for c in NEW4 + WIT:
        v = u[c].cast(pl.Float64)
        print(f"  {c:28s} {(v.is_null() | v.is_nan()).sum() / u.height:7.4f}")
    print(u.group_by("ADEP_mvt").agg([pl.len(), (pl.col("witness_med").is_null().sum() / pl.len()).round(4)
                                      .alias("witness_null")]).sort("ADEP_mvt"))


# ---------------------------------------------------------------- S3
def _spearman(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 100:
        return float("nan"), 0
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1]), int(ok.sum())


def s3_correlations() -> None:
    print("\n=== S3  is any candidate a measurement of the same event? ===")
    cands = ["stand_arrival_gap", "nbr_mvt_minus_aobt_mean_30m", "nbr_mvt_minus_aobt_mean_60m",
             "nbr_aobt_delay_mean_30m", "arr_taxi_in_mean_30m", "arr_taxi_in_mean_60m", "queue_ahead",
             "congestion_vs_usual_30m", "dep_tko_prev_30m", "mvt_minus_sched", *NEW4, *WIT]
    base = ["MVT_ID_mvt", "ADEP_mvt", LABEL, PROXY, "mvt_minus_sched", "stand_arrival_gap",
            "nbr_mvt_minus_aobt_mean_30m", "nbr_mvt_minus_aobt_mean_60m", "nbr_aobt_delay_mean_30m",
            "arr_taxi_in_mean_30m", "arr_taxi_in_mean_60m", "queue_ahead", "congestion_vs_usual_30m",
            "dep_tko_prev_30m"]
    t = (pl.read_parquet(FEAT_TRAIN, columns=base)
           .join(pl.read_parquet(SCRATCH / "cand_train.parquet"), on="MVT_ID_mvt", how="left")
           .join(pl.read_parquet(SCRATCH / "wit_train.parquet"), on="MVT_ID_mvt", how="left")
           .filter(pl.col(LABEL).is_not_null()))
    m = t.filter(pl.col(PROXY).is_not_null())
    print("matched labelled rows", m.height)

    g = m.filter(pl.col("stand_arrival_gap").is_not_null())
    y = g[LABEL].to_numpy().astype(float)
    sg = g["stand_arrival_gap"].to_numpy().astype(float)
    print("\nstand_arrival_gap tested AS AN UPPER BOUND on y (it is not one):")
    print(f"  n={len(y)}  P(y > gap) = {float((y > sg).mean()):.4f}   median slack gap-y = {np.median(sg - y):.0f} s")
    print(g.with_columns((pl.col(LABEL) > pl.col("stand_arrival_gap")).alias("viol"))
           .group_by("ADEP_mvt")
           .agg([pl.len().alias("n"), pl.col("viol").mean().round(4).alias("viol_rate"),
                 (pl.col("stand_arrival_gap") - pl.col(LABEL)).median().alias("med_slack")])
           .sort("ADEP_mvt"))

    proxy = m[PROXY].to_numpy().astype(float)
    lab = m[LABEL].to_numpy().astype(float)
    gg = lab - proxy
    bs = m["mvt_minus_sched"].to_numpy().astype(float) - lab
    sel = bs >= 3000
    print(f"\nSpearman on matched labelled rows (and on the {int(sel.sum())} delay-selected ones, BLOCK-SCHED>=3000):")
    print(f"{'candidate':30s}{'n':>10s}{'vs proxy':>10s}{'vs y':>9s}{'vs g':>9s}{'vs proxy|late':>15s}")
    for c in cands:
        v = m[c].cast(pl.Float64).to_numpy()
        a, n = _spearman(v, proxy)
        b, _ = _spearman(v, lab)
        cc, _ = _spearman(v, gg)
        dd, _ = _spearman(v[sel], proxy[sel])
        print(f"{c:30s}{n:>10d}{a:>10.4f}{b:>9.4f}{cc:>9.4f}{dd:>15.4f}")


# ---------------------------------------------------------------- shared fit harness
CFG = TrainConfig(name="missing_clock_cf", n_seeds=1)
ROUNDS = 400
SUBSAMPLE_PCT = 13  # of the matched training rows, by a hash of MVT_ID_mvt


def _load_fold():
    cand = (pl.read_parquet(SCRATCH / "cand_train.parquet")
              .join(pl.read_parquet(SCRATCH / "wit_train.parquet"), on="MVT_ID_mvt"))

    def lf():
        return (pl.scan_parquet(FEAT_TRAIN)
                  .with_columns((pl.col("MVT_TIME_UTC_mvt").dt.year() * 100
                                 + pl.col("MVT_TIME_UTC_mvt").dt.month()).alias("ym"))
                  .filter(pl.col(LABEL).is_not_null() & pl.col(LABEL).is_finite()))

    hashed = (pl.col("MVT_ID_mvt").cast(pl.Int64).hash(seed=7) % 100) < SUBSAMPLE_PCT
    tr = lf().filter((pl.col("ym") != 202501) & pl.col(PROXY).is_not_null() & hashed).collect()
    va = lf().filter((pl.col("ym") == 202501) & pl.col(PROXY).is_not_null()).collect()
    vu = lf().filter((pl.col("ym") == 202501) & pl.col(PROXY).is_null()).collect()
    enc = fit_encodings(tr, CFG)
    TR = attach_encodings(tr, enc).join(cand, on="MVT_ID_mvt", how="left")
    VA = attach_encodings(va, enc).join(cand, on="MVT_ID_mvt", how="left")
    VU = attach_encodings(vu, enc).join(cand, on="MVT_ID_mvt", how="left")
    del tr, va, vu
    gc.collect()
    return TR, VA, VU


def _weights(apt: np.ndarray, ms: np.ndarray):
    """Post-stratification weights taking the matched validation rows to the SCORED unmatched lane's
    airport x ms-band composition. Cells with fewer than 20 validation rows are dropped."""
    r = pl.read_parquet(FEAT_RANK, columns=["ADEP_mvt", PROXY, "mvt_minus_sched"]).filter(pl.col(PROXY).is_null())

    def bidx(v):
        o = np.full(len(v), -1)
        for i, (lo, hi) in enumerate(MS_BANDS):
            o[(v >= lo) & (v < hi)] = i
        return o

    ua, ub = r["ADEP_mvt"].to_numpy(), bidx(r["mvt_minus_sched"].to_numpy())
    mb = bidx(ms)
    cells = {}
    for a in sorted(set(apt)):
        for i in range(len(MS_BANDS)):
            w = float(((ua == a) & (ub == i)).sum())
            if w > 0 and ((apt == a) & (mb == i)).sum() >= 20:
                cells[(a, i)] = w
    return cells, mb, float(sum(cells.values())) / len(ua)


def _fit(TR, frames, cols, cats, target, adds):
    allc = cols + cats
    X = pl.concat([TR.select(allc)] + [f.select(allc) for f in frames], how="vertical_relaxed")
    for c in cats:
        s = X[c].cast(pl.Utf8).fill_null("__NA__")
        X = X.with_columns(s.cast(pl.Enum(sorted(s.unique().to_list()))).to_physical().cast(pl.Float32).alias(c))
    X = X.cast(pl.Float32).to_numpy()
    p = dict(CFG.params, seed=42, num_threads=2)
    d = lgb.Dataset(X[: TR.height], label=target, categorical_feature=[allc.index(c) for c in cats], params=p)
    mdl = lgb.train(p, d, num_boost_round=ROUNDS, callbacks=[lgb.log_evaluation(0)])
    out, i = [], TR.height
    for f, add in zip(frames, adds, strict=True):
        out.append(np.clip(mdl.predict(X[i: i + f.height]) + add, 0, None))
        i += f.height
    rk = {c: g for c, g in zip(allc, mdl.feature_importance("gain"), strict=True)}
    del X, d, mdl
    gc.collect()
    return out, rk


def _feature_sets(TR):
    num = [c for c in visible_numeric_columns(TR, CFG) if c not in set(NEW4) | set(WIT)]
    cats_all = [c for c in CATEGORICALS if c in TR.columns]
    cats_mvt = [c for c in cats_all if c not in CAT_FLT]
    cols_a = [c for c in num if c != PROXY] + [PROXY]
    cols_b = [c for c in num if not c.startswith(FLT_PREFIX) and c not in ALSO_NULL]
    cols_p = [c for c in num if not c.startswith(("mvt_minus_aobt3", "aobt3")) and c != "stand_aobt_turn"]
    return cols_a, cols_p, cols_b, cats_all, cats_mvt


# ---------------------------------------------------------------- S4
def s4_counterfactual() -> None:
    print("\n=== S4  the matched-lane counterfactual: how much worse is a model denied the flight plan ===")
    t0 = time.time()
    TR, VA, VU = _load_fold()
    cols_a, cols_p, cols_b, cats_all, cats_mvt = _feature_sets(TR)
    print(f"train {TR.height} matched rows (hash {SUBSAMPLE_PCT}%), valid {VA.height} matched / {VU.height} unmatched")
    print(f"feature counts  A {len(cols_a)}+{len(cats_all)}   P {len(cols_p)}+{len(cats_all)}   "
          f"B {len(cols_b)}+{len(cats_mvt)}")
    print("dropped from A to B:", sorted(set(cols_a) - set(cols_b)))
    yA = TR[LABEL].to_numpy().astype(float) - TR[PROXY].to_numpy().astype(float)
    yP = TR[LABEL].to_numpy().astype(float)
    res = {}
    res["A"], _ = _fit(TR, [VA, VU], cols_a, cats_all, yA, [VA[PROXY].to_numpy().astype(float), np.zeros(VU.height)])
    res["A_plain"], _ = _fit(TR, [VA, VU], cols_a, cats_all, yP, [0.0, 0.0])
    res["P"], _ = _fit(TR, [VA, VU], cols_p, cats_all, yP, [0.0, 0.0])
    res["B"], _ = _fit(TR, [VA, VU], cols_b, cats_mvt, yP, [0.0, 0.0])
    # model B's own sample size, same blind feature set
    small = TR.filter((pl.col("MVT_ID_mvt").cast(pl.Int64).hash(seed=7) % 1000) < 10)
    res["B_small"], _ = _fit(small, [VA, VU], cols_b, cats_mvt,
                             small[LABEL].to_numpy().astype(float), [0.0, 0.0])
    print(f"(B_small trained on {small.height} rows, model B's order of magnitude)")

    y = VA[LABEL].to_numpy().astype(float)
    ms = VA["mvt_minus_sched"].to_numpy().astype(float)
    apt = VA["ADEP_mvt"].to_numpy()
    bs = ms - y
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))  # noqa: E731
    cells, mb, cov = _weights(apt, ms)
    Z = sum(cells.values())

    def wr(p, apts=None):
        s = w = 0.0
        for (a, i), c in cells.items():
            if apts is not None and a not in apts:
                continue
            m = (apt == a) & (mb == i)
            s += c * float(np.mean((y[m] - p[m]) ** 2))
            w += c
        return float(np.sqrt(s / w))

    K = ["A", "A_plain", "P", "B", "B_small"]
    NL = [a for a in sorted(set(apt)) if a != "LIRF"]
    print(f"\n{'model':10s}{'RMSE matched':>14s}{'bias':>9s}{'reweighted':>12s}{'rw non-LIRF':>13s}  (coverage {cov:.3f})")
    for k in K:
        print(f"{k:10s}{rmse(y, res[k][0]):>14.2f}{float(np.mean(res[k][0] - y)):>+9.2f}"
              f"{wr(res[k][0]):>12.1f}{wr(res[k][0], NL):>13.1f}")
    print("\nper airport (A / P / B / B_small):")
    for a in sorted(set(apt)):
        m = apt == a
        v = [rmse(y[m], res[k][0][m]) for k in ("A", "P", "B", "B_small")]
        print(f"  {a} n={int(m.sum()):6d}  " + " / ".join(f"{x:7.1f}" for x in v) + f"   B-A {v[2] - v[0]:6.1f}")
    print("\nper ms band (A / P / B), with the label spread:")
    for lo, hi in [(-1e9, 1800), (1800, 3600), (3600, 5400), (5400, 7200), (7200, 10800), (10800, 14400), (14400, 1e9)]:
        m = (ms >= lo) & (ms < hi)
        if m.sum() < 5:
            continue
        v = [rmse(y[m], res[k][0][m]) for k in ("A", "P", "B")]
        print(f"  ms {lo:>9.0f}-{hi:<10.0f} n={int(m.sum()):6d}  " + " / ".join(f"{x:7.1f}" for x in v)
              + f"   sd(y) {float(np.std(y[m])):7.1f}")
    print("\nper BLOCK-SCHED band, the axis the lane is selected on (A / P / B):")
    for lo, hi in [(-1e9, 0), (0, 900), (900, 1800), (1800, 2700), (2700, 3300), (3300, 3606), (3606, 1e9)]:
        m = (bs >= lo) & (bs < hi)
        if m.sum() < 5:
            continue
        v = [rmse(y[m], res[k][0][m]) for k in ("A", "P", "B")]
        print(f"  bs {lo:>9.0f}-{hi:<10.0f} n={int(m.sum()):6d}  " + " / ".join(f"{x:7.1f}" for x in v))
    yu = VU[LABEL].to_numpy().astype(float)
    print(f"\nthe real unmatched jan rows (n={len(yu)}) scored by the blind model trained on MATCHED rows: "
          f"RMSE {rmse(yu, res['B'][1]):.1f}")
    for a in sorted(set(VU['ADEP_mvt'].to_numpy())):
        m = VU["ADEP_mvt"].to_numpy() == a
        print(f"   {a} n={int(m.sum()):5d} RMSE {rmse(yu[m], res['B'][1][m]):9.1f}")
    print("S4 elapsed", round(time.time() - t0, 1))


# ---------------------------------------------------------------- S5
def s5_substitutes() -> None:
    print("\n=== S5  does any candidate recover what the blind model lost ===")
    t0 = time.time()
    TR, VA, _ = _load_fold()
    _, _, base, _, cats = _feature_sets(TR)
    y = VA[LABEL].to_numpy().astype(float)
    ms = VA["mvt_minus_sched"].to_numpy().astype(float)
    apt = VA["ADEP_mvt"].to_numpy()
    cells, mb, _ = _weights(apt, ms)
    Z = sum(cells.values())
    rmse = lambda p: float(np.sqrt(np.mean((y - p) ** 2)))  # noqa: E731

    def wr(p):
        s = 0.0
        for (a, i), c in cells.items():
            m = (apt == a) & (mb == i)
            s += (c / Z) * float(np.mean((y[m] - p[m]) ** 2))
        return float(np.sqrt(s))

    yP = TR[LABEL].to_numpy().astype(float)
    (pv,), _ = _fit(TR, [VA], base, cats, yP, [0.0])
    b_all, b_w = rmse(pv), wr(pv)
    nbr = [c for c in base if c.startswith("nbr_mvt_minus_aobt_mean")]
    nob = [c for c in base if c not in nbr]
    print(f"base = model B's declared feature set, {len(base)} numeric + {len(cats)} categorical")
    print(f"the neighbour off-block clocks already inside it: {nbr}")
    print(f"\n{'variant':40s}{'RMSE matched':>14s}{'d':>8s}{'RMSE reweighted':>17s}{'d':>8s}  gain share")
    print(f"{'B_cf base':40s}{b_all:>14.2f}{0.0:>8.2f}{b_w:>17.2f}{0.0:>8.2f}")
    tests = [("+stand_arrival_gap", base + ["stand_arrival_gap"], ["stand_arrival_gap"]),
             ("+queue_ahead", base + ["queue_ahead"], ["queue_ahead"]),
             ("+congestion_vs_usual_30m", base + ["congestion_vs_usual_30m"], ["congestion_vs_usual_30m"]),
             ("+dest_distance_km", base + ["dest_distance_km"], ["dest_distance_km"]),
             *[(f"+{c}", base + [c], [c]) for c in NEW4],
             ("+all four new", base + NEW4, NEW4),
             ("+congestion witness", base + WIT, WIT),
             ("+witness + all four new", base + WIT + NEW4, WIT + NEW4),
             ("base MINUS neighbour clocks", nob, []),
             ("base MINUS neighbour clocks + witness", nob + WIT, WIT),
             ("+noise_control (control)", base + ["noise_control"], ["noise_control"])]
    for nm, cols, extra in tests:
        cols = list(dict.fromkeys(cols))
        (pv,), rk = _fit(TR, [VA], cols, cats, yP, [0.0])
        a, w = rmse(pv), wr(pv)
        tot = max(1e-9, sum(rk.values()))
        share = ", ".join(f"{c}:{rk.get(c, 0) / tot * 100:.2f}%" for c in extra)
        print(f"{nm:40s}{a:>14.2f}{a - b_all:>+8.2f}{w:>17.2f}{w - b_w:>+8.2f}  {share}")
    print("S5 elapsed", round(time.time() - t0, 1))


# ---------------------------------------------------------------- S6
def s6_residual() -> None:
    print("\n=== S6  association with the model-B OOF residual, raw and after conditioning ===")
    cond = ["mvt_minus_sched", "hour_local", "dow", "month", "dep_tko_prev_30m", "dep_tko_prev_60m",
            "dep_tko_next_30m", "arr_land_prev_30m", "arr_inblock_prev_30m", "rwy_headway_prev_s",
            "rwy_headway_next_s", "rwy_tko_prev_60m", "nbr_aobt_delay_mean_30m", "nbr_mvt_minus_aobt_mean_30m",
            "nbr_mvt_minus_aobt_mean_60m", "nbr_dep_n_60m", "arr_taxi_in_mean_30m", "arr_taxi_in_mean_60m",
            "minute_of_day_utc", "active_runways_60m", "dep_arr_ratio_30m"]
    cands = ["stand_arrival_gap", "queue_ahead", "congestion_vs_usual_30m", "dest_distance_km", *NEW4, *WIT]
    need = sorted(set(["MVT_ID_mvt", "ADEP_mvt", LABEL, PROXY, *cond, "stand_arrival_gap", "queue_ahead",
                       "congestion_vs_usual_30m", "dest_distance_km"]))
    t = (pl.read_parquet(FEAT_TRAIN, columns=need)
           .join(pl.read_parquet(SCRATCH / "cand_train.parquet"), on="MVT_ID_mvt", how="left")
           .join(pl.read_parquet(SCRATCH / "wit_train.parquet"), on="MVT_ID_mvt", how="left"))
    d = (pl.read_parquet(OOF).drop_nulls("oof")
           .join(pl.read_parquet(OOF_FOLDS), on="MVT_ID_mvt")
           .join(t, on="MVT_ID_mvt")
           .filter((pl.col("fold") != "jul_fwd") & pl.col(PROXY).is_null())
           .with_columns((pl.col("oof") - pl.col(LABEL)).alias("resid")))
    dx = d.filter(pl.col(LABEL) < 20000)
    print(f"unmatched OOF rows {d.height}, excluding the day-rollover labels {dx.height}")
    Z = np.column_stack([dx[c].cast(pl.Float64).to_numpy() for c in cond]
                        + [dx["ADEP_mvt"].cast(pl.Categorical).to_physical().cast(pl.Float64).to_numpy()])
    res = dx["resid"].to_numpy().astype(float)
    rng = np.random.default_rng(0)
    fold = rng.integers(0, 4, len(res))

    def residualise(v):
        out = np.full(len(v), np.nan)
        for k in range(4):
            tr, te = fold != k, fold == k
            ok = tr & np.isfinite(v)
            if ok.sum() < 200:
                continue
            p = dict(objective="regression", learning_rate=0.05, num_leaves=15, min_data_in_leaf=40,
                     verbose=-1, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                     num_threads=2, seed=1)
            mdl = lgb.train(p, lgb.Dataset(Z[ok], label=v[ok]), num_boost_round=150)
            out[te] = v[te] - mdl.predict(Z[te])
        return out

    r_res, r_abs = residualise(res), residualise(np.abs(res))
    print(f"\n{'candidate':28s}{'n':>7s}{'rho(c,resid)':>14s}{'rho(c,|resid|)':>16s}"
          f"{'partial rho':>13s}{'partial |r|':>13s}")
    for c in cands:
        v = dx[c].cast(pl.Float64).to_numpy()
        a, n = _spearman(v, res)
        b, _ = _spearman(v, np.abs(res))
        rv = residualise(v)
        pa, _ = _spearman(rv, r_res)
        pb, _ = _spearman(rv, r_abs)
        print(f"{c:28s}{n:>7d}{a:>14.4f}{b:>16.4f}{pa:>13.4f}{pb:>13.4f}")
    ctrl = rng.normal(size=len(res))
    a, _ = _spearman(ctrl, res)
    pa, _ = _spearman(residualise(ctrl), r_res)
    print(f"{'gaussian noise (control)':28s}{len(res):>7d}{a:>14.4f}{'':>16s}{pa:>13.4f}")
    print("\nper airport, rho(candidate, resid):")
    print(f"{'apt':6s}{'n':>6s}" + "".join(f"{c[:13]:>15s}" for c in cands))
    for a in sorted(dx["ADEP_mvt"].unique().to_list()):
        s = dx.filter(pl.col("ADEP_mvt") == a)
        rr = s["resid"].to_numpy().astype(float)
        print(f"{a:6s}{s.height:>6d}" + "".join(f"{_spearman(s[c].cast(pl.Float64).to_numpy(), rr)[0]:>15.3f}"
                                                for c in cands))


# ---------------------------------------------------------------- S7
def s7_price() -> None:
    print("\n=== S7  pricing ===")
    N, R = 344841, 282.7371
    per_sec = N * (2 * R + 1)
    print(f"board N={N} RMSE={R}: one second of RMSE = {per_sec:.4e} of squared error")
    r = pl.read_parquet(FEAT_RANK, columns=["ADEP_mvt", PROXY]).filter(pl.col(PROXY).is_null())
    n_u = r.height
    n_lirf = r.filter(pl.col("ADEP_mvt") == "LIRF").height
    print(f"scored unmatched rows {n_u}: LIRF {n_lirf}, other nine airports {n_u - n_lirf}")
    d = (pl.read_parquet(OOF).drop_nulls("oof")
           .join(pl.read_parquet(OOF_FOLDS), on="MVT_ID_mvt")
           .join(pl.read_parquet(FEAT_TRAIN, columns=["MVT_ID_mvt", "ADEP_mvt", LABEL, PROXY, "mvt_minus_sched"]),
                 on="MVT_ID_mvt")
           .filter(pl.col("fold") != "jul_fwd")
           .with_columns((pl.col("oof") - pl.col(LABEL)).alias("resid")))
    u = d.filter(pl.col(PROXY).is_null())
    rr = lambda f: float(np.sqrt((f["resid"] ** 2).mean()))  # noqa: E731
    print(f"OOF: pooled {rr(d):.4f}, matched {rr(d.filter(pl.col(PROXY).is_not_null())):.1f}, "
          f"unmatched {rr(u):.1f}, unmatched excl y>=20000 {rr(u.filter(pl.col(LABEL) < 20000)):.1f}, "
          f"non-LIRF excl {rr(u.filter((pl.col(LABEL) < 20000) & (pl.col('ADEP_mvt') != 'LIRF'))):.1f}")
    for tag, n, hi, lo in [("non-LIRF: observed -> sighted", n_u - n_lirf, 569.1, 361.6),
                           ("non-LIRF: blind+starved -> sighted (both counterfactual)", n_u - n_lirf, 591.4, 361.6),
                           ("non-LIRF: the training-size part of it alone", n_u - n_lirf, 591.4, 541.7),
                           ("non-LIRF: what stand_arrival_gap measurably recovers", n_u - n_lirf, 554.0, 539.1),
                           ("non-LIRF: what witness + the four new columns recover", n_u - n_lirf, 554.0, 547.5),
                           ("whole lane: observed -> sighted", n_u, 1141.3, 361.6),
                           ("whole lane: a 10% cut of its RMSE (the shape of the old -7 s)", n_u, 1141.3, 1027.2),
                           ("LIRF alone -> the non-LIRF unmatched level", n_lirf, 3437.2, 569.1)]:
        dsse = n * (hi * hi - lo * lo)
        print(f"  {tag:58s} n={n:5d}  {hi:7.1f} -> {lo:7.1f}  dRMSE {-dsse / per_sec:+7.3f} s")


SECTIONS = {"S1": s1_visible, "S2": s2_candidates, "S3": s3_correlations, "S4": s4_counterfactual,
            "S5": s5_substitutes, "S6": s6_residual, "S7": s7_price}

if __name__ == "__main__":
    want = [a for a in sys.argv[1:] if a in SECTIONS] or list(SECTIONS)
    if any(s in want for s in ("S3", "S4", "S5", "S6")) and "S2" not in want:
        if not (SCRATCH / "cand_train.parquet").exists():
            want = ["S2"] + want
    for s in want:
        SECTIONS[s]()
