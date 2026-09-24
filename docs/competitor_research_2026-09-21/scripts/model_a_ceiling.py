"""How much of model A's error is irreducible: four estimators of E[Var(y | visible)] on the matched lane.

Read-only. Nothing is written to data/, outputs/, configs/ or src/; no submission is made. Reproduces
every number in docs/competitor_research_2026-09-21/results/model-a-ceiling.md; each section prints its
own tag (S1..S7) and the document cites the tag beside each figure.

    POLARS_MAX_THREADS=2 .venv/bin/python docs/competitor_research_2026-09-21/scripts/model_a_ceiling.py [S1 S2 ...]

Sections (wall clock on an otherwise idle 4-CPU box):
  S1  reproduce the lane split; restate model A's error in the space it is actually fitted in   ~40 s
  S2  route 1  matched-twin ladder: within-cell spread of g, plain and covariate-adjusted       ~6 min
  S3  route 2  1-NN semivariance against the model's error ON THE SAME ROWS, post-stratified    ~9 min
  S4  route 3  twin-residual covariance -> the systematic share rho -> the floor                ~9 min
  S5  route 4  learning curve in training rows and in capacity                                  ~70 min  fits
  S6  what an oracle cell-mean correction of the OOF residual could recover                     ~2 min
  S7  the targets, and whether the +1.98 s OOF-to-board gap is stable                           ~30 s

Definitions used throughout.
  y      = TAXITIME_SEC_mvt = MVT_TIME - BLOCK_TIME                  (the label)
  proxy  = mvt_minus_aobt3  = MVT_TIME - AOBT_3_flt                  (model A's measured off-block clock)
  g      = y - proxy        = AOBT_3_flt - BLOCK_TIME                (what model A's tree actually fits)
  e      = oof - y          = ghat - g                               (model A's OOF residual)
  lane A = proxy is not null = the routing rule in src/prc/train.py:402

A feature is visible iff it reads neither BLOCK_TIME_UTC_mvt nor TAXITIME_SEC_mvt (src/prc/schema.py:52).
Every column used for matching below is visible by that rule; `g` and `e` are label-side and are only
ever the QUANTITY measured, never part of a key.

Why g and not y. proxy is exactly visible and enters the prediction additively, so
Var(y | visible) == Var(g | visible) identically, while Var(y | cell) > Var(g | cell) for any cell that
does not hold proxy fixed. Measuring the spread of g rather than y therefore removes the largest visible
axis for free and makes every matched-set estimate below tighter without assuming anything.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

FEAT_TRAIN = ROOT / "data/work/feat_train.parquet"
OOF = ROOT / "experiments/oof_main_gap-62dd7c6e88.parquet"
OOF_FOLDS = ROOT / "experiments/oof_main_gap-62dd7c6e88_folds.parquet"
EXPERIMENTS = ROOT / "experiments/experiments.jsonl"
SUBMISSIONS = ROOT / "experiments/submissions.jsonl"
SCRATCH = Path("/tmp/model_a_ceiling")
SCRATCH.mkdir(exist_ok=True)

APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]

# The matching metric. Eight continuous visible columns and four visible categoricals, chosen by LightGBM
# gain share on g (S1 prints it): these twelve carry 85% of the gain. STAND_mvt is not in the metric - it
# is the exact block every neighbour search runs inside, which is stricter.
CONT = [
    "aobt3_minus_eobt1",
    "mvt_minus_aobt3",
    "aobt3_minus_sched",
    "mvt_minus_eobt1",
    "mvt_minus_sched",
    "mvt_minus_lobt",
    "nbr_aobt_delay_mean_30m",
    "stand_arrival_gap",
]
CAT = ["RUNWAY_mvt", "ADES_mvt", "AIRCRAFT_OPERATOR_flt", "AIRCRAFT_TYPE_mvt"]
# The wide metric of S4.5. Adds the columns the narrow metric leaves out - the congestion block, the
# weather block and time-of-day - so that "d = 0" comes closer to "the whole visible vector agrees". If
# rho is stable between the two metrics, the 15% of gain the narrow metric omits carries no systematic
# error; if rho rises, the narrow metric was under-stating the recoverable share and the floor is lower.
CONT_WIDE = CONT + [
    "mvt_minus_iobt",
    "lobt_minus_eobt1",
    "nbr_mvt_minus_aobt_mean_30m",
    "dep_tko_prev_30m",
    "dep_tko_prev_60m",
    "arr_taxi_in_mean_30m",
    "rwy_headway_prev_s",
    "rwy_headway_next_s",
    "queue_ahead",
    "hour_local",
    "temp_c",
    "wind_ms",
    "vis_km",
]
# Continuous columns used as within-cell covariates in S2's adjusted arm (a superset of CONT).
ZCOV = CONT + [
    "mvt_minus_iobt",
    "lobt_minus_eobt1",
    "nbr_mvt_minus_aobt_mean_30m",
    "dep_tko_prev_30m",
    "arr_taxi_in_mean_30m",
    "rwy_headway_prev_s",
    "queue_ahead",
]

BOARD_N = 344_841
BOARD_V14 = 279.5234  # v14 = main_gap-62dd7c6e88, the OOF this note is built on
OOF_V14 = 277.5398  # S1 reproduces this


def _tag(s: str) -> None:
    print(f"\n{'=' * 100}\n{s}\n{'=' * 100}", flush=True)


def load_lane_a(extra: list[str] | None = None, with_oof: bool = True) -> pl.DataFrame:
    """Matched, labelled departures with g attached, and (optionally) the OOF residual and fold."""
    cols = sorted(
        set(
            ["MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "TAXITIME_SEC_mvt", "hour_local", "month",
             "MVT_TIME_UTC_mvt"]
            + CONT
            + CAT
            + (extra or [])
        )
    )
    a = pl.read_parquet(FEAT_TRAIN, columns=cols).filter(
        pl.col("mvt_minus_aobt3").is_not_null() & pl.col("TAXITIME_SEC_mvt").is_not_null()
    )
    a = a.with_columns((pl.col("TAXITIME_SEC_mvt") - pl.col("mvt_minus_aobt3")).alias("g"))
    if with_oof:
        fol = pl.read_parquet(OOF_FOLDS).filter(pl.col("pooled")).select("MVT_ID_mvt", "fold")
        a = a.join(pl.read_parquet(OOF), on="MVT_ID_mvt", how="left").join(fol, on="MVT_ID_mvt", how="left")
        a = a.with_columns((pl.col("oof") - pl.col("TAXITIME_SEC_mvt")).alias("e"))
    # the difficulty stratum used for every post-stratification below: within-airport quintile of the
    # single most informative visible column. Cheap, visible, and the same 50 cells everywhere.
    a = a.with_columns(
        (pl.col("aobt3_minus_eobt1").rank("ordinal").over("ADEP_mvt") / pl.len().over("ADEP_mvt") * 5)
        .floor()
        .clip(0, 4)
        .cast(pl.Int32)
        .alias("b_ae")
    )
    return a


def keycodes(frame: pl.DataFrame, keys: list[str]) -> np.ndarray:
    k = pl.concat_str([pl.col(x).cast(pl.Utf8).fill_null("~") for x in keys], separator="|")
    return frame.select(k.alias("k"))["k"].cast(pl.Categorical).to_physical().to_numpy()


# --------------------------------------------------------------------------------------------------
# S1 - the lane split, and model A restated in g-space
# --------------------------------------------------------------------------------------------------


def s1_baseline() -> None:
    _tag("S1  the lane split, and what model A is actually fitting")
    oof = pl.read_parquet(OOF)
    fol = pl.read_parquet(OOF_FOLDS).filter(pl.col("pooled"))
    ft = pl.read_parquet(
        FEAT_TRAIN, columns=["MVT_ID_mvt", "ADEP_mvt", "TAXITIME_SEC_mvt", "mvt_minus_aobt3", "mvt_minus_sched"]
    )
    d = fol.join(oof, on="MVT_ID_mvt").join(ft, on="MVT_ID_mvt")
    d = d.with_columns(
        [
            (pl.col("oof") - pl.col("TAXITIME_SEC_mvt")).alias("err"),
            pl.col("mvt_minus_aobt3").is_not_null().alias("A"),
        ]
    )
    err = d["err"].to_numpy()
    print(f"pooled OOF rows {d.height}, RMSE {np.sqrt((err ** 2).mean()):.4f}")
    sse_t = float((err**2).sum())
    for lane, name in ((True, "A (matched)"), (False, "B (unmatched)")):
        s = d.filter(pl.col("A") == lane)["err"].to_numpy()
        print(
            f"  lane {name:14s} n={len(s):7d}  RMSE {np.sqrt((s ** 2).mean()):9.4f}  "
            f"share of total SSE {float((s ** 2).sum()) / sse_t:.4f}"
        )
    print("\nper fold (from experiments.jsonl, exp_id main_gap-62dd7c6e88):")
    for line in open(EXPERIMENTS):
        rec = json.loads(line)
        if rec["exp_id"] != "main_gap-62dd7c6e88":
            continue
        for f, v in rec["folds"].items():
            m = v.get("matched", {})
            print(
                f"  {f:9s} n={v['n']:7d} full {v['rmse_full']:8.3f} | matched n={m.get('n', 0):7d} "
                f"RMSE {m.get('rmse_full', float('nan')):8.3f} | unmatched RMSE "
                f"{v.get('unmatched', {}).get('rmse_full', float('nan')):9.2f}"
            )

    a = d.filter(pl.col("A")).with_columns((pl.col("TAXITIME_SEC_mvt") - pl.col("mvt_minus_aobt3")).alias("g"))
    g = a["g"].to_numpy()
    y = a["TAXITIME_SEC_mvt"].to_numpy().astype(float)
    e = a["err"].to_numpy()
    mse = (e**2).mean()
    print("\nlane A restated in the space model A is fitted in (pred = proxy + ghat, so err = ghat - g):")
    print(f"  sd(y) {y.std():8.2f}   sd(g) {g.std():8.2f}   mean(g) {g.mean():8.2f}   median(g) {np.median(g):8.1f}")
    print(f"  robust sd(g) = IQR/1.349 = {(np.percentile(g, 75) - np.percentile(g, 25)) / 1.349:8.2f}"
          "   <- g is heavy-tailed; the RMSE lives in the tail")
    print(f"  model A RMSE {np.sqrt(mse):8.3f}   bias {e.mean():+7.3f}")
    print(f"  fraction of Var(g) explained {1 - mse / g.var():.4f}   of Var(y) {1 - mse / y.var():.4f}")

    print("\nper-airport, lane A only (this is the table the ceiling must be broken down over):")
    print(f"  {'ADEP':6s} {'n':>8s} {'RMSE':>9s} {'bias':>8s} {'sd(g)':>9s} {'share of lane-A SSE':>20s}")
    sse_a = float((e**2).sum())
    rows = []
    for ap in APTS:
        s = a.filter(pl.col("ADEP_mvt") == ap)
        ee = s["err"].to_numpy()
        gg = s["g"].to_numpy()
        rows.append((ap, len(ee), float(np.sqrt((ee**2).mean())), float(ee.mean()), float(gg.std()),
                     float((ee**2).sum()) / sse_a))
        print(f"  {ap:6s} {len(ee):8d} {rows[-1][2]:9.2f} {rows[-1][3]:+8.2f} {gg.std():9.2f} {rows[-1][5]:20.4f}")
    json.dump(rows, open(SCRATCH / "s1_per_airport.json", "w"))


# --------------------------------------------------------------------------------------------------
# S2 - route 1: the matched-twin ladder
# --------------------------------------------------------------------------------------------------


def _basis(z: np.ndarray) -> np.ndarray:
    """Piecewise-linear basis on one covariate: the level plus four hinges at its quintiles."""
    z = np.nan_to_num(z.astype(np.float64), nan=0.0)
    qs = np.unique(np.percentile(z, [20, 40, 60, 80]))
    return np.column_stack([z] + [np.maximum(z - q, 0) for q in qs])


def _within(codes: np.ndarray | None, gv: np.ndarray, Z: np.ndarray) -> tuple[float, int, int, int, int]:
    """Pooled within-cell SSE of gv after a within-cell least-squares fit on Z, with honest df.

    Returns (sse, df, rows_used, n_cells, rank). codes=None means one cell. Cells of size 1 carry no
    information about a within-cell spread and are dropped; that is the coverage number S2 reports."""
    p = Z.shape[1]
    if codes is None:
        gd = gv - gv.mean()
        Zd = Z - Z.mean(0) if p else Z
        nrow, ncell = len(gv), 1
    else:
        _, inv = np.unique(codes, return_inverse=True)
        cnt = np.bincount(inv)
        keep = (cnt >= 2)[inv]
        gd, Zd, inv = gv[keep], Z[keep], inv[keep]
        _, inv = np.unique(inv, return_inverse=True)
        cnt = np.bincount(inv)
        ncell, nrow = len(cnt), len(gd)
        gd = gd - (np.bincount(inv, weights=gd) / cnt)[inv]
        if p:
            Zm = np.empty((ncell, p))
            for j in range(p):
                Zm[:, j] = np.bincount(inv, weights=Zd[:, j]) / cnt
            Zd = Zd - Zm[inv]
    if p:
        beta, _, rank, _ = np.linalg.lstsq(Zd, gd, rcond=None)
        r = gd - Zd @ beta
    else:
        r, rank = gd, 0
    return float((r**2).sum()), int(nrow - ncell - rank), nrow, ncell, int(rank)


LADDER = [
    ("L1 ADEP", []),
    ("L2 +operator", ["AIRCRAFT_OPERATOR_flt"]),
    ("L3 +stand", ["AIRCRAFT_OPERATOR_flt", "STAND_mvt"]),
    ("L4 +runway", ["AIRCRAFT_OPERATOR_flt", "STAND_mvt", "RUNWAY_mvt"]),
    ("L5 +dest", ["AIRCRAFT_OPERATOR_flt", "STAND_mvt", "RUNWAY_mvt", "ADES_mvt"]),
    ("L6 +3h block", ["AIRCRAFT_OPERATOR_flt", "STAND_mvt", "RUNWAY_mvt", "ADES_mvt", "h3"]),
    ("L7 +actype", ["AIRCRAFT_OPERATOR_flt", "STAND_mvt", "RUNWAY_mvt", "ADES_mvt", "h3", "AIRCRAFT_TYPE_mvt"]),
    ("L8 +NM-delay quintile",
     ["AIRCRAFT_OPERATOR_flt", "STAND_mvt", "RUNWAY_mvt", "ADES_mvt", "h3", "AIRCRAFT_TYPE_mvt", "b_ae"]),
]


def s2_twin_ladder() -> None:
    _tag("S2  route 1 - the matched-twin ladder: within-cell spread of g as the cells tighten")
    a = load_lane_a(extra=ZCOV, with_oof=False).with_columns((pl.col("hour_local") // 3).alias("h3"))
    print(f"rows {a.height}, sd(g) {a['g'].std():.2f}")
    print("\nEvery cell key is built from visible columns only. `plain` is the within-cell sd of g; `+covar`")
    print("adds a within-cell least-squares fit on 15 visible continuous columns (75 hinge terms), which is")
    print("what lets a cell resolve a continuous feature it cannot bin. `control` re-runs the +covar arm with")
    print("the cell labels shuffled and the covariate rows permuted: it must return the within-ADEP sd of g")
    print("at every level, and a control that drifts means the degrees of freedom are being miscounted.\n")
    rng = np.random.default_rng(11)
    acc = {name: dict(sse=0.0, df=0, ssep=0.0, dfp=0, ssec=0.0, dfc=0, nrow=0, ncell=0) for name, _ in LADDER}
    per_apt: dict[str, dict] = {}
    for ap in APTS:
        s = a.filter(pl.col("ADEP_mvt") == ap)
        gv = s["g"].to_numpy().astype(float)
        n = len(gv)
        Z = np.hstack([_basis(s[c].to_numpy()) for c in ZCOV])
        Zp = Z[rng.permutation(n)]
        per_apt[ap] = {}
        for name, keys in LADDER:
            codes = np.zeros(n, dtype=np.int64) if not keys else keycodes(s, keys)
            sse, df, nrow, ncell, rank = _within(codes, gv, Z)
            ssep, dfp, _, _, _ = _within(codes, gv, np.zeros((n, 0)))
            cs = codes.copy()
            rng.shuffle(cs)
            ssec, dfc, _, _, _ = _within(cs, gv, Zp)
            R = acc[name]
            R["sse"] += sse; R["df"] += df; R["ssep"] += ssep; R["dfp"] += dfp
            R["ssec"] += ssec; R["dfc"] += dfc; R["nrow"] += nrow; R["ncell"] += ncell
            per_apt[ap][name] = (np.sqrt(sse / df), np.sqrt(ssep / dfp), nrow)
        print(f"  {ap} done", flush=True)
        del Z, Zp
        gc.collect()
    print(f"\n{'level':24s} {'plain sd':>9s} {'+covar':>9s} {'control':>9s} {'rows':>10s} {'cells':>9s} {'coverage':>9s}")
    for name, _ in LADDER:
        R = acc[name]
        print(
            f"  {name:22s} {np.sqrt(R['ssep'] / R['dfp']):9.2f} {np.sqrt(R['sse'] / R['df']):9.2f} "
            f"{np.sqrt(R['ssec'] / R['dfc']):9.2f} {R['nrow']:10d} {R['ncell']:9d} {R['nrow'] / a.height:9.3f}"
        )
    print("\nper airport, +covar sd / plain sd at L1, L5, L8:")
    for ap in APTS:
        r = per_apt[ap]
        print(
            f"  {ap:6s} L1 {r['L1 ADEP'][0]:7.1f}/{r['L1 ADEP'][1]:7.1f}   "
            f"L5 {r['L5 +dest'][0]:7.1f}/{r['L5 +dest'][1]:7.1f}   "
            f"L8 {r['L8 +NM-delay quintile'][0]:7.1f}/{r['L8 +NM-delay quintile'][1]:7.1f}  "
            f"n(L8)={r['L8 +NM-delay quintile'][2]}"
        )


# --------------------------------------------------------------------------------------------------
# S3 / S4 - the neighbour pass, shared
# --------------------------------------------------------------------------------------------------


def neighbour_pairs(a: pl.DataFrame, knn: int = 1, dmax: float = np.inf, shuffle_g: bool = False,
                    seed: int = 5, cont: list[str] | None = None) -> dict:
    """For every row, its k nearest visible twins inside its own STAND block.

    Distance: squared Euclidean over the eight z-scored continuous columns of CONT plus one unit per
    mismatched categorical in CAT. d == 0 therefore means the pair agrees on the stand and on all twelve
    metric columns exactly. Blocks smaller than 8 rows are skipped.
    """
    rng = np.random.default_rng(seed)
    cont = cont or CONT
    out = {k: [] for k in ("d", "gi", "gj", "ei", "ej", "fi", "fj", "ap", "ba", "ii", "dt")}
    fmap = {"jan": 0, "jul": 1, "forward": 2}
    for ap in APTS:
        s = a.filter(pl.col("ADEP_mvt") == ap)
        gv = s["g"].to_numpy().astype(float)
        if shuffle_g:
            gv = rng.permutation(gv)
        ev = s["e"].to_numpy().astype(float) if "e" in s.columns else np.full(len(gv), np.nan)
        fv = np.array([fmap.get(x, -1) for x in s["fold"].to_list()]) if "fold" in s.columns else np.full(len(gv), -1)
        ids = s["MVT_ID_mvt"].to_numpy()
        ba = s["b_ae"].to_numpy()
        tv = s["MVT_TIME_UTC_mvt"].dt.epoch("s").to_numpy().astype(np.float64)
        Z = np.column_stack([np.nan_to_num(s[c].to_numpy().astype(np.float64), nan=0.0) for c in cont])
        Z = (Z - Z.mean(0)) / np.maximum(Z.std(0), 1e-9)
        C = np.column_stack(
            [s[c].cast(pl.Categorical).to_physical().fill_null(-1).to_numpy() for c in CAT]
        ).astype(np.int64)
        stand = s["STAND_mvt"].cast(pl.Categorical).to_physical().fill_null(-1).to_numpy()
        o = np.argsort(stand, kind="stable")
        Z, C, gv, ev, fv, ids, ba, tv, st = Z[o], C[o], gv[o], ev[o], fv[o], ids[o], ba[o], tv[o], stand[o]
        bs = np.flatnonzero(np.r_[True, st[1:] != st[:-1]])
        be = np.r_[bs[1:], len(st)]
        for b0, b1 in zip(bs, be):
            n = b1 - b0
            if n < 8:
                continue
            zb, cb = Z[b0:b1], C[b0:b1]
            d2 = (zb**2).sum(1)[:, None] + (zb**2).sum(1)[None, :] - 2 * zb @ zb.T
            for j in range(C.shape[1]):
                d2 += (cb[:, j][:, None] != cb[None, :, j]).astype(np.float64)
            np.fill_diagonal(d2, np.inf)
            k = min(knn, n - 1)
            idx = np.argpartition(d2, k - 1, axis=1)[:, :k] if k > 1 else np.argmin(d2, 1)[:, None]
            dd = np.sqrt(np.maximum(np.take_along_axis(d2, idx, 1), 0))
            m = dd <= dmax
            if not m.any():
                continue
            ii = np.repeat(np.arange(n), k)[m.ravel()]
            jj = idx.ravel()[m.ravel()]
            out["d"].append(dd.ravel()[m.ravel()])
            for nm, arr in (("gi", gv), ("ei", ev), ("fi", fv), ("ba", ba)):
                out[nm].append(arr[b0:b1][ii])
            for nm, arr in (("gj", gv), ("ej", ev), ("fj", fv)):
                out[nm].append(arr[b0:b1][jj])
            out["ap"].append(np.full(int(m.sum()), ap))
            out["ii"].append(ids[b0:b1][ii])
            out["dt"].append(np.abs(tv[b0:b1][ii] - tv[b0:b1][jj]))
        print(f"  {ap} paired", flush=True)
    return {k: np.concatenate(v) for k, v in out.items()}


def s3_semivariance() -> None:
    _tag("S3  route 2 - 1-NN semivariance of g, against the model's error ON THE SAME ROWS")
    a = load_lane_a()
    P = neighbour_pairs(a, knn=1)
    d, gi, gj, ei = P["d"], P["gi"], P["gj"], P["ei"]
    gam = (gi - gj) ** 2 / 2
    se = ei**2
    has = np.isfinite(se)
    print(f"\nrows paired {len(d)} of {a.height}; 1-NN distance quantiles "
          f"{np.percentile(d, [5, 10, 25, 50, 75, 90, 95]).round(3)}")
    print("\ngamma(d) = E[(g_i - g_j)^2]/2 is an estimate of E[Var(g|x)] inflated by (mu_i - mu_j)^2/2, which")
    print("vanishes as d -> 0. The point of the third column is that the inflation can be read off directly:")
    print("model A's OOF RMSE is computed on exactly the rows in the same bin.\n")
    print(f"  {'d bin':18s} {'rows':>9s} {'rows in OOF':>12s} {'gamma^0.5':>10s} {'model RMSE':>11s} {'ratio':>7s}")
    for lo, hi in [(0, .25), (.25, .5), (.5, .75), (.75, 1.), (1., 1.25), (1.25, 1.5), (1.5, 2.), (2., 3.), (3., 1e9)]:
        m = (d > lo) & (d <= hi)
        mo = m & has
        if m.sum() < 200:
            continue
        gq, rq = np.sqrt(gam[m].mean()), np.sqrt(se[mo].mean())
        lbl = f"({lo:.2f},{hi:.2f}]" if hi < 1e8 else f"(>{lo:.2f})"
        print(f"  {lbl:18s} {m.sum():9d} {mo.sum():12d} {gq:10.2f} {rq:11.2f} {rq / gq:7.3f}")
    print(f"\n  whole lane A (all rows)  model RMSE {np.sqrt(se[has].mean()):.3f}")

    print("\npost-stratified to the lane's own composition (10 airports x 5 NM-delay quintiles = 50 cells):")
    print(f"  {'d <=':>6s} {'row cov':>9s} {'gamma^0.5':>10s} {'model RMSE':>11s} {'difference':>11s}")
    strat = np.array([f"{x}|{y}" for x, y in zip(P["ap"], P["ba"])])
    full = np.array([f"{x}|{y}" for x, y in zip(a["ADEP_mvt"].to_numpy(), a["b_ae"].to_numpy())])
    u, cnt = np.unique(full, return_counts=True)
    share = dict(zip(u, cnt / cnt.sum()))
    for dmax in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        tn = tm = cov = 0.0
        for s_, sh in share.items():
            m = (strat == s_) & (d <= dmax)
            mo = m & has
            if m.sum() < 30 or mo.sum() < 30:
                continue
            tn += sh * gam[m].mean()
            tm += sh * se[mo].mean()
            cov += sh
        gq, rq = np.sqrt(tn / cov), np.sqrt(tm / cov)
        print(f"  {dmax:6.2f} {float((d <= dmax).mean()):9.3f} {gq:10.2f} {rq:11.2f} {rq - gq:+11.2f}")

    Pc = neighbour_pairs(a, knn=1, shuffle_g=True)
    gc_ = (Pc["gi"] - Pc["gj"]) ** 2 / 2
    print(f"\ncontrol (g permuted within airport): gamma^0.5 overall {np.sqrt(gc_.mean()):.2f}, at d<=1.0 "
          f"{np.sqrt(gc_[Pc['d'] <= 1.0].mean()):.2f}; the within-ADEP sd of g is the number both must match.")


# --------------------------------------------------------------------------------------------------
# S4 - route 3: the twin-residual covariance
# --------------------------------------------------------------------------------------------------


def _rho(x: np.ndarray, y: np.ndarray, nboot: int = 400, seed: int = 0) -> tuple[float, float, float, float]:
    """(MSE, cov, rho, CI) for a set of twin pairs. rho = E[e_i e_j] / E[e^2]."""
    mse = float(((x**2 + y**2) / 2).mean())
    c = float((x * y).mean())
    r = np.random.default_rng(seed)
    n = len(x)
    bs = np.empty(nboot)
    for b in range(nboot):
        s = r.integers(0, n, n)
        bs[b] = (x[s] * y[s]).mean() / (((x[s] ** 2 + y[s] ** 2) / 2).mean())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return mse, c, c / mse, (float(lo), float(hi))


def s4_residual_covariance() -> None:
    _tag("S4  route 3 - twin-residual covariance: the systematic share of model A's error")
    print("""
The estimator. Write e = ghat(x) - g. For two rows with the SAME visible x, fitted by the SAME model,
    e_i = (ghat(x) - mu(x)) + (mu(x) - g_i)
and the second terms are independent across i with mean zero. Hence

    E[e_i e_j]  =  E[(ghat - mu)^2]           the part of the error the model gets WRONG about the mean
    E[e_i^2]    =  E[(ghat - mu)^2] + E[Var(g|x)]  =  MSE

so rho = E[e_i e_j]/MSE is the SYSTEMATIC (recoverable) share of the squared error and

    floor  =  RMSE * sqrt(1 - rho).

At d > 0 the two rows' approximation errors are not identical, so the measured covariance is an
UNDER-estimate of E[(ghat - mu)^2]; every floor printed below is therefore an UPPER bound that falls
towards the truth as d -> 0. Pairs are restricted to the same OOF fold, so ghat comes from one model.
""")
    a = load_lane_a(extra=CONT_WIDE)
    P = neighbour_pairs(a, knn=6, dmax=1.0)
    d, ei, ej, fi, fj, apv = P["d"], P["ei"], P["ej"], P["fi"], P["fj"], P["ap"]
    ok = np.isfinite(ei) & np.isfinite(ej) & (fi == fj)
    lane_rmse = float(np.sqrt(np.nanmean(a["e"].to_numpy() ** 2)))
    print(f"pairs at d<=1.0: {len(d)}; both in OOF and in the same fold: {ok.sum()}")
    print(f"lane A OOF RMSE (all 664,050 rows): {lane_rmse:.4f}\n")
    print(f"  {'d bin':16s} {'pairs':>8s} {'MSE^0.5':>9s} {'rho':>8s} {'rho 95% CI':>20s} {'floor|lane':>11s}")
    best = None
    for lo, hi in [(0, .1), (0, .25), (.25, .5), (.5, .75), (.75, 1.)]:
        m = ok & (d > lo) & (d <= hi)
        if m.sum() < 500:
            continue
        mse, c, rho, ci = _rho(ei[m], ej[m])
        if lo == 0 and hi == .25:
            best = (rho, ci)
        print(f"  ({lo:.2f},{hi:.2f}]{'':4s} {m.sum():8d} {np.sqrt(mse):9.2f} {rho:8.4f}  "
              f"[{ci[0]:.4f}, {ci[1]:.4f}] {lane_rmse * np.sqrt(max(1 - rho, 0)):11.2f}")
    print("\n'floor|lane' applies the measured rho to the WHOLE lane's MSE. That step assumes rho is the same")
    print("on rows with no close twin as on rows with one; the d-bin column is the evidence for it, and S3's")
    print("wide-d table extends it to d > 3.")

    print("""
PER FOLD. The pooled lane-A RMSE of 213.13 averages three folds that differ by 45 s (S1), and S5's
learning curve is measured on the forward fold alone. rho per fold makes the two sections comparable
without anyone having to assume the folds behave alike.""")
    fold_rmse = {}
    for nm, code in (("jan", 0), ("jul", 1), ("forward", 2)):
        ev = a.filter(pl.col("fold") == nm)["e"].to_numpy()
        fold_rmse[nm] = float(np.sqrt(np.nanmean(ev**2)))
    print(f"  {'fold':10s} {'pairs':>8s} {'rho':>8s} {'rho 95% CI':>20s} {'lane-A RMSE':>12s} {'floor':>9s}")
    for nm, code in (("jan", 0), ("jul", 1), ("forward", 2)):
        m = ok & (d <= 1.0) & (fi == code)
        if m.sum() < 500:
            continue
        mse, c, rho, ci = _rho(ei[m], ej[m])
        print(f"  {nm:10s} {m.sum():8d} {rho:8.4f}  [{ci[0]:.4f}, {ci[1]:.4f}] {fold_rmse[nm]:12.2f} "
              f"{fold_rmse[nm] * np.sqrt(max(1 - rho, 0)):9.2f}")

    print("""
ROBUSTNESS - how far apart in time are the twins? Two rows minutes apart on the same stand can share a
real operational shock (a closed taxiway, a de-icing queue) that is NOT a function of x. Such a shock
puts a positive Cov(g_i, g_j | x) into the estimator, inflating rho and pushing the floor DOWN. If rho
survives at large time separations, the covariance is approximation error and not a shared shock.""")
    dt = P["dt"] / 86400.0
    print(f"  {'|t_i - t_j|':18s} {'pairs':>8s} {'MSE^0.5':>9s} {'rho':>8s} {'rho 95% CI':>20s} {'floor|lane':>11s}")
    for lo, hi, lbl in [(0, 1, "< 1 day"), (1, 7, "1-7 days"), (7, 30, "7-30 days"),
                        (30, 120, "30-120 days"), (120, 1e9, "> 120 days")]:
        m = ok & (d <= 1.0) & (dt >= lo) & (dt < hi)
        if m.sum() < 500:
            continue
        mse, c, rho, ci = _rho(ei[m], ej[m])
        print(f"  {lbl:18s} {m.sum():8d} {np.sqrt(mse):9.2f} {rho:8.4f}  [{ci[0]:.4f}, {ci[1]:.4f}] "
              f"{lane_rmse * np.sqrt(max(1 - rho, 0)):11.2f}")

    print(f"""
WIDE METRIC - the narrow metric matches on 12 columns carrying 85% of the gain, so two rows at d = 0
can still differ on the other ~90 visible columns. If model A is systematically wrong ALONG those
columns, that error does not enter Cov(e_i, e_j) and rho is under-stated. Re-running the whole estimator
with {len(CONT_WIDE)} continuous columns instead of {len(CONT)} - the congestion block, the weather block and
time-of-day added - tests exactly that. A stable rho means the omitted columns carry no systematic error.
Distances are larger in the wider space, so the bins are wider and only the trend is read.""")
    Pw = neighbour_pairs(a, knn=6, dmax=2.0, cont=CONT_WIDE)
    dw, eiw, ejw, fiw, fjw = Pw["d"], Pw["ei"], Pw["ej"], Pw["fi"], Pw["fj"]
    okw = np.isfinite(eiw) & np.isfinite(ejw) & (fiw == fjw)
    print(f"  {'d bin (wide)':16s} {'pairs':>8s} {'MSE^0.5':>9s} {'rho':>8s} {'rho 95% CI':>20s} {'floor|lane':>11s}")
    for lo, hi in [(0, .5), (.5, 1.0), (1.0, 1.5), (1.5, 2.0)]:
        m = okw & (dw > lo) & (dw <= hi)
        if m.sum() < 500:
            continue
        mse, c, rho, ci = _rho(eiw[m], ejw[m])
        print(f"  ({lo:.2f},{hi:.2f}]{'':4s} {m.sum():8d} {np.sqrt(mse):9.2f} {rho:8.4f}  "
              f"[{ci[0]:.4f}, {ci[1]:.4f}] {lane_rmse * np.sqrt(max(1 - rho, 0)):11.2f}")
    del Pw, dw, eiw, ejw, fiw, fjw, okw
    gc.collect()

    print("\nNEGATIVE control - the partner residual is permuted, so no pair is a twin any more:")
    m = ok & (d <= 1.0)
    p = np.random.default_rng(2).permutation(int(m.sum()))
    mse, c, rho, ci = _rho(ei[m], ej[m][p])
    print(f"  rho {rho:+.4f}  [{ci[0]:+.4f}, {ci[1]:+.4f}]   (must contain 0)")

    print("\nPOSITIVE control - the same estimator applied to a deliberately crippled predictor of g:")
    print("  ghat := the out-of-half mean of g over (ADEP x AIRCRAFT_OPERATOR). It is a legitimate visible")
    print("  predictor and it is badly underfitted, so its error MUST be mostly systematic. If the estimator")
    print("  cannot see that, it cannot be trusted when it reports a small rho for model A.")
    gv = a["g"].to_numpy().astype(float)
    key = keycodes(a, ["ADEP_mvt", "AIRCRAFT_OPERATOR_flt"])
    K = int(key.max()) + 1
    half = np.random.default_rng(1).integers(0, 2, len(gv))
    ec = np.empty(len(gv))
    for h in (0, 1):
        tr = half != h
        cnt = np.bincount(key[tr], minlength=K)
        s = np.bincount(key[tr], weights=gv[tr], minlength=K)
        mu = np.where(cnt >= 5, s / np.maximum(cnt, 1), gv[tr].mean())
        ec[half == h] = mu[key[half == h]] - gv[half == h]
    print(f"  crippled predictor RMSE {np.sqrt((ec ** 2).mean()):.2f}  (model A: {lane_rmse:.2f})")
    a2 = a.with_columns(pl.Series("e", ec))
    P2 = neighbour_pairs(a2, knn=6, dmax=1.0, seed=5)
    d2, ci_, cj_ = P2["d"], P2["ei"], P2["ej"]
    for lo, hi in [(0, .25), (.25, .5), (.5, 1.)]:
        m = (d2 > lo) & (d2 <= hi)
        mse, c, rho, ci = _rho(ci_[m], cj_[m])
        print(f"  d in ({lo:.2f},{hi:.2f}] n={m.sum():8d} MSE^0.5 {np.sqrt(mse):7.2f} rho {rho:.4f} "
              f"[{ci[0]:.4f},{ci[1]:.4f}] -> floor on these rows {np.sqrt(max(mse - c, 0)):7.2f}")
    print("  Compare the last column with model A's own floor on the SAME rows (the d<=0.25 line above,")
    print("  MSE^0.5 * sqrt(1-rho)): two predictors 40% apart in RMSE must agree on the floor, and how far")
    print("  apart they land is the honest precision of this instrument.")

    print("\nPER AIRPORT (d <= 1.0, same fold). 'lane RMSE' is S1's per-airport lane-A OOF RMSE.")
    per = {r[0]: r for r in json.load(open(SCRATCH / "s1_per_airport.json"))} if (SCRATCH / "s1_per_airport.json").exists() else {}
    print(f"  {'ADEP':6s} {'pairs':>8s} {'rho':>8s} {'rho 95% CI':>20s} {'lane RMSE':>10s} {'floor':>9s} {'gap':>8s}")
    tot_sse = tot_floor = 0.0
    for ap in APTS:
        m = ok & (d <= 1.0) & (apv == ap)
        if m.sum() < 500:
            continue
        mse, c, rho, ci = _rho(ei[m], ej[m])
        lr = per[ap][2] if ap in per else float("nan")
        n = per[ap][1] if ap in per else 0
        fl = lr * np.sqrt(max(1 - rho, 0))
        tot_sse += n * lr * lr
        tot_floor += n * fl * fl
        print(f"  {ap:6s} {m.sum():8d} {rho:8.4f}  [{ci[0]:.4f}, {ci[1]:.4f}] {lr:10.2f} {fl:9.2f} {lr - fl:8.2f}")
    if tot_sse:
        n_all = sum(per[ap][1] for ap in APTS if ap in per)
        print(f"  {'POOLED':6s} {'':8s} {'':8s} {'':20s} {np.sqrt(tot_sse / n_all):10.2f} "
              f"{np.sqrt(tot_floor / n_all):9.2f} {np.sqrt(tot_sse / n_all) - np.sqrt(tot_floor / n_all):8.2f}")


# --------------------------------------------------------------------------------------------------
# S5 - route 4: the learning curve
# --------------------------------------------------------------------------------------------------


def s5_learning_curve(points=((127, 0.0625), (127, 0.125), (127, 0.25), (127, 0.5), (127, 1.0),
                              (31, 0.25), (511, 0.25))) -> None:
    """Each point is (num_leaves, fraction of the matched training pool).

    Cheapened deliberately and IDENTICALLY for every point, because the curve's shape is what is read and
    not its level: learning_rate 0.15 instead of production's 0.05, at most 400 rounds instead of 2000,
    one seed instead of three, and `deterministic`/`force_row_wise` off (those two are there for
    bit-reproducibility of a 0.1 s delta, which no conclusion here turns on, and they cost a large factor
    in wall clock on a shared box). Early stopping is KEPT - the same inner-holdout carve-out production
    uses - because a fixed round budget would flatter the large-n points and steepen the curve, which is
    the direction that would manufacture headroom.

    **The absolute level is therefore worse than production and must not be quoted as a production
    number.** configs/main_gap.yaml measures 198.53 on this fold with three seeds; the frac=1.0 point
    here is the same recipe cheapened, and the distance between them is the price of the cheapening.
    The capacity arm is run at frac=0.25 rather than 1.0 so that 31 / 127 / 511 leaves are compared at
    one fixed row count inside the budget."""
    _tag("S5  route 4 - the learning curve: is model A's error still falling in rows, or in capacity?")
    import time as _t

    from prc.cv import year_month_key
    from prc.train import LABEL, PROXY, TrainConfig, predict_fold

    ft = pl.read_parquet(FEAT_TRAIN)
    ft = ft.with_columns(year_month_key(ft).alias("ym"))
    val = ft.filter((pl.col("ym") % 100 >= 11) & pl.col(PROXY).is_not_null() & pl.col(LABEL).is_not_null())
    pool = ft.filter((pl.col("ym") % 100 < 11) & pl.col(PROXY).is_not_null() & pl.col(LABEL).is_not_null())
    del ft
    gc.collect()
    print(f"forward fold, matched rows only: train pool {pool.height}, validation {val.height}")
    print("configs/main_gap.yaml cheapened IDENTICALLY at every point: lr 0.15 (production 0.05), <=400 rounds")
    print("(production 2000), one seed (production three), deterministic/force_row_wise OFF. Early stopping is")
    print("kept, so each point is this recipe's best at that row count. The LEVEL is worse than production's")
    print("198.53 on this fold; only the SHAPE is read. Determinism is off because it costs ~10x wall clock")
    print("on this box for bit-reproducibility no conclusion here needs.\n")
    h = (pool["MVT_ID_mvt"].cast(pl.Utf8).hash(seed=99) % 10**6).to_numpy() / 10**6
    yv = val[LABEL].to_numpy().astype(float)
    apt = val["ADEP_mvt"].to_numpy()
    out = []
    for nl, f in points:
        sub = pool.filter(pl.Series(h < f))
        cfg = TrainConfig(
            name="lc", n_seeds=1, num_rounds=400, early_stopping=50, use_encodings=True,
            include_candidates=("stand_arrival_gap",), unmatched_sched_reparam=True, lobt_bound=True,
        )
        cfg.params = dict(cfg.params)
        cfg.params["num_leaves"] = nl
        cfg.params["learning_rate"] = 0.15
        cfg.params["deterministic"] = False
        cfg.params["force_row_wise"] = False
        t0 = _t.time()
        pr, info = predict_fold(sub, val, cfg)
        p = val.select("MVT_ID_mvt").join(pr, on="MVT_ID_mvt", how="left")["pred"].to_numpy()
        r = float(np.sqrt(np.nanmean((p - yv) ** 2)))
        rec = dict(
            leaves=nl, frac=f, n=sub.height, rmse=r, rounds=info.get("rounds_matched"),
            secs=round(_t.time() - t0, 1),
            by_airport={a: float(np.sqrt(np.nanmean((p[apt == a] - yv[apt == a]) ** 2))) for a in APTS},
        )
        out.append(rec)
        print(json.dumps({k: v for k, v in rec.items() if k != "by_airport"}), flush=True)
        json.dump(out, open(SCRATCH / "s5_learning_curve.json", "w"))
        del sub, pr, p
        gc.collect()
    base = [r for r in out if r["leaves"] == 127]
    if len(base) >= 4:
        n = np.array([r["n"] for r in base], float)
        m = np.array([r["rmse"] for r in base], float) ** 2
        best = None
        for alpha in np.linspace(0.05, 1.5, 300):
            X = np.column_stack([np.ones(len(n)), n ** (-alpha)])
            c, res, *_ = np.linalg.lstsq(X, m, rcond=None)
            r2 = float(((X @ c - m) ** 2).sum())
            if c[0] > 0 and (best is None or r2 < best[0]):
                best = (r2, alpha, c)
        if best:
            _, alpha, c = best
            print(f"\nfit MSE(n) = a + b*n^-alpha  ->  alpha {alpha:.3f}, asymptote sqrt(a) = {np.sqrt(c[0]):.2f}")
            print("  (the n -> infinity limit of THIS recipe on THESE features; it is not the floor, it is the")
            print("   error the recipe converges to given unlimited 2025-like rows.)")
            n100 = base[-1]["n"]
            for mult in (2, 4, 10):
                pred = np.sqrt(c[0] + c[1] * (n100 * mult) ** (-alpha))
                print(f"  extrapolated to {mult}x the training rows ({n100 * mult:,}): "
                      f"{pred:.2f}  ({pred - base[-1]['rmse']:+.2f} s)")
    caps = [r for r in out if r["frac"] == 1.0]
    if len(caps) > 1:
        print("\ncapacity arm at 100% of the rows:")
        for r in sorted(caps, key=lambda z: z["leaves"]):
            print(f"  num_leaves {r['leaves']:5d}  rounds {r['rounds']:5d}  RMSE {r['rmse']:8.3f}")


# --------------------------------------------------------------------------------------------------
# S6 - what an oracle cell-mean correction could recover
# --------------------------------------------------------------------------------------------------


def s6_cell_recovery() -> None:
    _tag("S6  an oracle cell-mean correction of the OOF residual: how much is left on the table")
    print("""
For any key K built from visible columns, the best correction constant within a cell is E[e|K]. This
section estimates it out-of-fold (5 random folds over rows, cell means from the other four) with ridge
shrinkage lambda, and reports the RMSE after subtracting it. lambda=0 is the raw cell mean; the control
shuffles the cell labels, and its best column is what a key carrying no information buys.

Note what this CAN and CANNOT answer. A cell-mean correction is exactly what a target encoding, a
per-group specialist or a per-group offset column would buy. It cannot see structure that varies
CONTINUOUSLY inside the cell - S4's rho does.
""")
    a = load_lane_a(extra=["AIRCRAFT_TYPE_mvt"]).filter(pl.col("e").is_not_null())
    a = a.with_columns((pl.col("hour_local") // 3).alias("h3"))
    e = a["e"].to_numpy().astype(float)
    N = len(e)
    print(f"lane A OOF rows {N}, RMSE {np.sqrt((e ** 2).mean()):.4f}\n")
    keys = [
        ("C1 ADEP", ["ADEP_mvt"]),
        ("C2 +operator", ["ADEP_mvt", "AIRCRAFT_OPERATOR_flt"]),
        ("C3 +stand", ["ADEP_mvt", "AIRCRAFT_OPERATOR_flt", "STAND_mvt"]),
        ("C4 +runway", ["ADEP_mvt", "AIRCRAFT_OPERATOR_flt", "STAND_mvt", "RUNWAY_mvt"]),
        ("C5 +dest", ["ADEP_mvt", "AIRCRAFT_OPERATOR_flt", "STAND_mvt", "RUNWAY_mvt", "ADES_mvt"]),
        ("C6 +3h block", ["ADEP_mvt", "AIRCRAFT_OPERATOR_flt", "STAND_mvt", "RUNWAY_mvt", "ADES_mvt", "h3"]),
        ("C7 +actype+delay", ["ADEP_mvt", "AIRCRAFT_OPERATOR_flt", "STAND_mvt", "RUNWAY_mvt", "ADES_mvt",
                              "h3", "AIRCRAFT_TYPE_mvt", "b_ae"]),
        ("S1 stand alone", ["ADEP_mvt", "STAND_mvt"]),
        ("S2 stand x rwy x 3h", ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "h3"]),
        ("S3 operator alone", ["AIRCRAFT_OPERATOR_flt"]),
    ]
    lams = (0.0, 5.0, 20.0, 50.0, 150.0, 500.0)
    rng = np.random.default_rng(3)
    fold = rng.integers(0, 5, N)

    def recover(codes):
        u, inv = np.unique(codes, return_inverse=True)
        K = len(u)
        res = {}
        for lam in lams:
            corr = np.zeros(N)
            for f in range(5):
                tr = fold != f
                te = ~tr
                cnt = np.bincount(inv[tr], minlength=K)
                s = np.bincount(inv[tr], weights=e[tr], minlength=K)
                corr[te] = (s / np.maximum(cnt + lam, 1e-9))[inv[te]]
            res[lam] = float(np.sqrt(((e - corr) ** 2).mean()))
        return res, K

    print(f"  {'cell key':22s} {'cells':>8s} " + " ".join(f"{'lam=' + str(int(l)):>10s}" for l in lams) + "   control")
    for name, k in keys:
        codes = keycodes(a, k)
        res, K = recover(codes)
        cs = codes.copy()
        rng.shuffle(cs)
        resc, _ = recover(cs)
        print(f"  {name:22s} {K:8d} " + " ".join(f"{res[l]:10.3f}" for l in lams) + f"   {min(resc.values()):9.3f}")
    print(f"\n  baseline (no correction) {np.sqrt((e ** 2).mean()):.4f}")


# --------------------------------------------------------------------------------------------------
# S7 - the targets, and the OOF-to-board gap
# --------------------------------------------------------------------------------------------------


def s7_targets() -> None:
    _tag("S7  the targets, and whether the +1.98 s OOF-to-board gap is stable")
    oof = pl.read_parquet(OOF)
    fol = pl.read_parquet(OOF_FOLDS).filter(pl.col("pooled"))
    ft = pl.read_parquet(FEAT_TRAIN, columns=["MVT_ID_mvt", "ADEP_mvt", "TAXITIME_SEC_mvt", "mvt_minus_aobt3"])
    d = (fol.join(oof, on="MVT_ID_mvt").join(ft, on="MVT_ID_mvt")
         .with_columns([(pl.col("oof") - pl.col("TAXITIME_SEC_mvt")).alias("err"),
                        pl.col("mvt_minus_aobt3").is_not_null().alias("A")]))
    a = d.filter(pl.col("A"))["err"].to_numpy()
    b = d.filter(~pl.col("A"))["err"].to_numpy()
    nA, nB = len(a), len(b)
    sseB = float((b**2).sum())
    print(f"held fixed: lane B, n={nB}, SSE={sseB:.4e}")
    print(f"\n  {'board target':>14s} {'implied OOF':>12s} {'lane A must reach':>18s} {'vs 213.13':>10s}")
    for board in (234.5, 230.0, BOARD_V14):
        oof_t = board - (BOARD_V14 - OOF_V14)
        need = np.sqrt(max((oof_t**2) * (nA + nB) - sseB, 0) / nA)
        print(f"  {board:14.4f} {oof_t:12.4f} {need:18.2f} {need / 213.1266 - 1:+9.1%}")
    print("\nrho that model A's error would have to be, for the floor to sit at each target")
    print("(floor = 213.1266 * sqrt(1 - rho), so rho = 1 - (target/213.1266)^2):")
    for t in (148.9, 141.7, 201.0):
        print(f"  target lane-A RMSE {t:7.1f}  ->  rho must be at least {1 - (t / 213.1266) ** 2:.4f}")

    print("\nThe OOF-to-board gap. It is ONE observation: v14 is the only submission whose board score can be")
    print("paired with a pooled three-fold OOF from the same single config (v10..v13 were hand-assembled from")
    print("two lineages; see experiments/submissions.jsonl).")
    rows = []
    for line in open(SUBMISSIONS):
        r = json.loads(line)
        if r.get("leaderboard_score"):
            rows.append((r["version"], r["exp_id"], r["leaderboard_score"]))
    for v, e_, s in rows:
        print(f"  v{v:<3d} {e_:38s} board {s:9.4f}")
    print(f"  v14  main_gap-62dd7c6e88                   board {BOARD_V14:9.4f}   pooled OOF {OOF_V14:9.4f}"
          f"   gap {BOARD_V14 - OOF_V14:+.4f}")

    print("\nHow much does that gap move if the pooling changes? The board is Jan+Jul 2026; the pooled OOF is")
    print("jan + jul + forward 2025. Recomputing the OOF on the fold subsets the board actually resembles:")
    fold = fol.join(oof, on="MVT_ID_mvt").join(ft, on="MVT_ID_mvt").with_columns(
        [(pl.col("oof") - pl.col("TAXITIME_SEC_mvt")).alias("err"),
         pl.col("mvt_minus_aobt3").is_not_null().alias("A")])
    for sel, lbl in ((("jan", "jul", "forward"), "all three (as shipped)"),
                     (("jan", "jul"), "jan + jul only (the board's months)"),
                     (("forward",), "forward only"), (("jan",), "jan only"), (("jul",), "jul only")):
        s = fold.filter(pl.col("fold").is_in(list(sel)))
        ee = s["err"].to_numpy()
        ea = s.filter(pl.col("A"))["err"].to_numpy()
        print(f"  {lbl:36s} n={len(ee):7d}  pooled {np.sqrt((ee ** 2).mean()):9.4f}  "
              f"lane A {np.sqrt((ea ** 2).mean()):8.2f}  implied gap to board {BOARD_V14 - np.sqrt((ee ** 2).mean()):+8.4f}")


SECTIONS = {
    "S1": s1_baseline,
    "S2": s2_twin_ladder,
    "S3": s3_semivariance,
    "S4": s4_residual_covariance,
    "S5": s5_learning_curve,
    "S6": s6_cell_recovery,
    "S7": s7_targets,
}

if __name__ == "__main__":
    want = [x for x in sys.argv[1:] if x in SECTIONS] or list(SECTIONS)
    if "S4" in want and "S1" not in want and not (SCRATCH / "s1_per_airport.json").exists():
        want = ["S1"] + want
    for s in want:
        t = time.time()
        SECTIONS[s]()
        print(f"\n[{s} done in {time.time() - t:.0f}s]", flush=True)
