"""How much of model B's error is irreducible: estimators of E[Var(y | visible)] on the UNMATCHED lane.

The lane-B counterpart of scripts/model_a_ceiling.py. Read-only: nothing is written to data/, outputs/,
configs/ or src/; no submission is made; no production model is refitted. Reproduces every number in
docs/competitor_research_2026-09-21/results/model-b-ceiling.md; each section prints its own tag (S1..S7)
and the document cites the tag beside each figure.

    POLARS_MAX_THREADS=2 .venv/bin/python docs/competitor_research_2026-09-21/scripts/model_b_ceiling.py [S1 S2 ...]

Sections:
  S1  the lane split, lane B's LIVE visible column set, and the importance fit that picks the metric
  S2  concentration: how few rows lane B's RMSE is actually made of (Kish n_eff, jackknife, blocks)
  S3  BIAS vs SPREAD: oracle cell-mean offsets in-sample and out-of-fold, with a shuffled control
  S4  route 1 - k-NN semivariance of bs, read against the model's error ON THE SAME ROWS
  S5  route 2 - twin-residual covariance -> the systematic share rho -> the floor
  S6  route 3 - the LIRF oracle fill gate, and the oracle on the broken-label rows
  S7  the board price through dBoard/dOOF = 0.747, with and without LIRF and the broken labels

Definitions.
  y      = TAXITIME_SEC_mvt = MVT_TIME - BLOCK_TIME                  (the label)
  ms     = mvt_minus_sched  = MVT_TIME - SCHED_TIME                  (VISIBLE on every lane-B row)
  bs     = ms - y           = BLOCK_TIME - SCHED_TIME                (label-side; the quantity measured)
  e      = oof - y                                                   (model B's OOF residual)
  lane B = mvt_minus_aobt3 is null = the routing rule in src/prc/train.py:686

WHY bs AND NOT y. `ms` is exactly visible on lane B (missing-clock.md S1.2: mvt_minus_sched is 0% null
at all ten airports on unmatched scored rows), so Var(y | visible) == Var(bs | visible) IDENTICALLY,
while Var(y | cell) > Var(bs | cell) for any cell that does not hold ms fixed. Measuring the spread of
bs therefore removes the largest visible axis for free. This is the same device as model_a_ceiling.py's
use of g, and it carries the same warning: bs is the LABEL IN DISGUISE (bs = ms - y). It is only ever
the QUANTITY MEASURED here, never a key, never a feature. Every column used to match rows below is
visible by src/prc/schema.py's rule (reads neither BLOCK_TIME_UTC_mvt nor TAXITIME_SEC_mvt).

CONTROLS ARE PART OF THE DESIGN, not added afterwards. S3 carries a shuffled-cell arm, S4 a
permuted-bs arm, S5 a permuted-partner arm AND a deliberately crippled positive control. This repo has
twice shipped a silent no-op (P1-stage2-gate-FAILED.md's AUC 1.0000 across every recipe including the
permuted control; missing-clock.md S7's degrees-of-freedom drift), so a route without a control is not
reported.
"""

from __future__ import annotations

import gc
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

APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]

# The narrow matching metric, chosen by LightGBM gain share on lane B's own 22,470 rows (S1 prints the
# fit). These twelve carry 87.5% of the gain - deliberately the same shape as model_a_ceiling.py's
# twelve-at-85%. STAND_mvt is a metric column here rather than the block, because lane B has 22,470 rows
# over ~2,500 stands and stand blocks are too small to search inside; the block is ADEP_mvt.
CONT = [
    "mvt_minus_sched",
    "queue_ahead",
    "stand_rwy_share",
    "stand_arrival_gap",
    "nbr_mvt_minus_aobt_mean_60m",
    "arr_taxi_in_mean_30m",
    "minute_of_day_utc",
    "stand_arrival_type_match",
]
CAT = ["RUNWAY_mvt", "STAND_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt"]

# The wide metric of S5.4: everything else with non-trivial gain, so that "d = 0" comes closer to "the
# whole visible vector agrees". If rho is stable between the two, the 12% of gain the narrow metric
# omits carries no systematic error.
CONT_WIDE = CONT + [
    "nbr_mvt_minus_aobt_mean_30m",
    "arr_taxi_in_mean_60m",
    "nbr_aobt_delay_mean_60m",
    "congestion_vs_usual_30m",
    "dep_tko_prev_60m",
    "dest_distance_km",
    "bank_position",
    "temp_c",
    "wind_ms",
    "month",
]

BOARD_N = 344_841
BOARD_V14 = 279.5234  # v14 = main_gap-62dd7c6e88, the OOF this note is built on
OOF_V14 = 277.5398
TRANSFER = 0.747  # dBoard/dOOF, model-a-ceiling.md S8.2, n = 1
GAP = BOARD_V14 - OOF_V14  # +1.9836, the brief's additive OOF->board gap

# A taxi time above this is not a taxi time. 40,000 s is 11.1 hours; the largest physically plausible
# pushback-to-wheels-up in this data set is well under 3 h. These rows are LABEL-SIDE anomalies and the
# threshold is a DESCRIPTION of rows already identified by their squared error, never a model input.
# S2.4 sweeps it, and sweeps the alternative `bs` criterion too.
BROKEN_Y = 40_000.0


def _tag(s: str) -> None:
    print(f"\n{'=' * 100}\n{s}\n{'=' * 100}", flush=True)


def neff(e: np.ndarray) -> float:
    """Kish effective sample size of an RMSE: (sum e^2)^2 / sum e^4.

    The number of equal-sized rows that would produce the same sampling variance for the mean squared
    error. For a lane whose SSE is one row, it is 1. P1B-lirf-seasonal-bias-REJECT.md S2.4 uses the same
    device on a mean; this is its second-moment analogue and it is the single most important statistic
    in this note.
    """
    s = e.astype(float) ** 2
    return float(s.sum() ** 2 / (s**2).sum())


def load_lane_b(extra: list[str] | None = None, with_oof: bool = True, oof_only: bool = False) -> pl.DataFrame:
    """Unmatched labelled departures with bs attached, and (optionally) the OOF residual and fold.

    All 22,470 of them by default - the twin routes draw partners from the whole unmatched universe,
    not only the 8,378 rows that happen to be in the pooled OOF, because a twin only has to supply a
    label. `oof_only` restricts to the OOF rows for the routes that need a residual on BOTH sides.
    """
    cols = sorted(
        set(
            ["MVT_ID_mvt", "ADEP_mvt", "TAXITIME_SEC_mvt", "mvt_minus_aobt3", "mvt_minus_sched",
             "MVT_TIME_UTC_mvt", "month", "hour_local"]
            + CONT_WIDE
            + CAT
            + (extra or [])
        )
    )
    b = pl.read_parquet(FEAT_TRAIN, columns=cols).filter(
        pl.col("mvt_minus_aobt3").is_null() & pl.col("TAXITIME_SEC_mvt").is_not_null()
    )
    b = b.with_columns((pl.col("mvt_minus_sched") - pl.col("TAXITIME_SEC_mvt")).alias("bs"))
    if with_oof:
        fol = pl.read_parquet(OOF_FOLDS).filter(pl.col("pooled")).select("MVT_ID_mvt", "fold")
        b = b.join(pl.read_parquet(OOF), on="MVT_ID_mvt", how="left").join(fol, on="MVT_ID_mvt", how="left")
        b = b.with_columns((pl.col("oof") - pl.col("TAXITIME_SEC_mvt")).alias("e"))
        if oof_only:
            b = b.filter(pl.col("e").is_not_null())
    # the difficulty stratum used for every post-stratification below: within-airport quintile of the
    # single most informative visible column (ms, 53.9% of gain). Cheap, visible, same 50 cells everywhere.
    b = b.with_columns(
        (pl.col("mvt_minus_sched").rank("ordinal").over("ADEP_mvt") / pl.len().over("ADEP_mvt") * 5)
        .floor()
        .clip(0, 4)
        .cast(pl.Int32)
        .alias("b_ms")
    )
    b = b.with_columns(
        (pl.col("TAXITIME_SEC_mvt") > BROKEN_Y).alias("broken"),
        # of the impossible labels, the ones a model CANNOT see coming: ms in the ordinary range, so
        # nothing visible marks them. The rest have ms of 40,000-73,000 s and are visibly extraordinary.
        ((pl.col("TAXITIME_SEC_mvt") > BROKEN_Y) & (pl.col("mvt_minus_sched") < 7_200)).alias("blind_broken"),
    )
    return b


def keycodes(frame: pl.DataFrame, keys: list[str]) -> np.ndarray:
    k = pl.concat_str([pl.col(x).cast(pl.Utf8).fill_null("~") for x in keys], separator="|")
    return frame.select(k.alias("k"))["k"].cast(pl.Categorical).to_physical().to_numpy()


def rmse(e: np.ndarray) -> float:
    return float(np.sqrt((e.astype(float) ** 2).mean()))


# --------------------------------------------------------------------------------------------------
# S1 - the lane split, lane B's live visible set, and the metric
# --------------------------------------------------------------------------------------------------


def s1_baseline() -> None:
    _tag("S1  the lane split, lane B's live visible columns, and the matching metric")
    import prc.features as F

    oof = pl.read_parquet(OOF)
    fol = pl.read_parquet(OOF_FOLDS).filter(pl.col("pooled"))
    ft = pl.read_parquet(
        FEAT_TRAIN,
        columns=["MVT_ID_mvt", "ADEP_mvt", "TAXITIME_SEC_mvt", "mvt_minus_aobt3", "mvt_minus_sched"],
    )
    j = (
        ft.join(oof, on="MVT_ID_mvt", how="inner")
        .join(fol.select("MVT_ID_mvt", "fold"), on="MVT_ID_mvt", how="inner")
        .with_columns(
            (pl.col("oof") - pl.col("TAXITIME_SEC_mvt")).alias("e"),
            (pl.col("mvt_minus_sched") - pl.col("TAXITIME_SEC_mvt")).alias("bs"),
        )
    )
    tot_sse = float((j["e"].to_numpy() ** 2).sum())
    print(f"\npooled OOF rows {j.height:,}  RMSE {rmse(j['e'].to_numpy()):.4f}")
    print(f"{'lane':<12}{'n':>10}{'RMSE':>12}{'bias':>10}{'median':>9}{'SSE share':>12}{'n_eff':>11}")
    for name, m in (("A (matched)", pl.col("mvt_minus_aobt3").is_not_null()),
                    ("B (unmatch)", pl.col("mvt_minus_aobt3").is_null())):
        s = j.filter(m)
        e = s["e"].to_numpy()
        print(f"{name:<12}{s.height:>10,}{rmse(e):>12.4f}{e.mean():>10.2f}{np.median(e):>9.1f}"
              f"{(e**2).sum()/tot_sse:>12.4f}{neff(e):>11.1f}")

    b = j.filter(pl.col("mvt_minus_aobt3").is_null())

    print("\n-- S1.1  lane B restated in the space the floor lives in")
    y = b["TAXITIME_SEC_mvt"].to_numpy().astype(float)
    bs = b["bs"].to_numpy()
    ms = b["mvt_minus_sched"].to_numpy()
    print(f"  sd(y)                 {y.std():>12.2f}")
    print(f"  sd(ms)   VISIBLE      {ms.std():>12.2f}")
    print(f"  sd(bs)   = Var(y|vis) axis the metric cannot remove  {bs.std():>12.2f}")
    print(f"  robust sd(bs) = IQR/1.349 {np.subtract(*np.percentile(bs,[75,25]))/1.349:>8.2f}")
    print(f"  model B RMSE          {rmse(b['e'].to_numpy()):>12.4f}")
    print(f"  bias                  {b['e'].to_numpy().mean():>12.3f}")
    print(f"  fraction of Var(bs) explained  {1 - (b['e'].to_numpy()**2).mean()/bs.var():>12.4f}")
    print("  [the last line is the direct analogue of model-a-ceiling.md S1.2's 0.7151]")

    print("\n-- S1.2  per airport, lane B only")
    print(f"{'ADEP':<7}{'n':>7}{'RMSE':>11}{'bias':>10}{'median':>9}{'sd(bs)':>11}{'SSE share':>11}{'n_eff':>9}")
    rows = []
    for (ap,), s in b.group_by(["ADEP_mvt"], maintain_order=False):
        e = s["e"].to_numpy()
        rows.append((ap, s.height, rmse(e), e.mean(), float(np.median(e)), float(s["bs"].to_numpy().std()),
                     float((e**2).sum() / (b['e'].to_numpy()**2).sum()), neff(e)))
    for r in sorted(rows, key=lambda r: -r[6]):
        print(f"{r[0]:<7}{r[1]:>7,}{r[2]:>11.2f}{r[3]:>10.1f}{r[4]:>9.1f}{r[5]:>11.1f}{r[6]:>11.4f}{r[7]:>9.1f}")

    print("\n-- S1.3  lane B's LIVE visible column set (the brief's 'enumerate it rather than assuming')")
    head = pl.read_parquet(FEAT_TRAIN, n_rows=5)
    num = F.numeric_feature_columns(head)
    cols_b = [c for c in num if not c.startswith(
        ("mvt_minus_aobt3", "aobt3", "lobt_minus", "mvt_minus_lobt", "mvt_minus_eobt1", "mvt_minus_iobt"))]
    cats = [c for c in F.CATEGORICALS if c in head.columns]
    need = list(dict.fromkeys(cols_b + cats + ["mvt_minus_aobt3", "TAXITIME_SEC_mvt"]))
    a = pl.read_parquet(FEAT_TRAIN, columns=need).filter(pl.col("mvt_minus_aobt3").is_null())
    print(f"  model A numeric {len(num)}   model B numeric {len(cols_b)}  + {len(cats)} categoricals"
          f"  = {len(cols_b)+len(cats)} advertised")
    dead, high = [], []
    for c in cols_b + cats:
        s = a[c]
        nn = s.null_count() / a.height
        nu = s.n_unique()
        if nu <= 1 or nn > 0.999:
            dead.append((c, nn, nu))
        elif nn > 0.20:
            high.append((c, nn, nu))
    print(f"  DEAD on every row model B ever sees ({len(dead)}):")
    for c, nn, nu in dead:
        print(f"     {c:<34} null {nn:.4f}  uniq {nu}")
    print(f"  live but >20% null ({len(high)}):")
    for c, nn, nu in high:
        print(f"     {c:<34} null {nn:.4f}  uniq {nu}")
    print(f"  ==> {len(cols_b)+len(cats)-len(dead)} LIVE columns, not {len(cols_b)+len(cats)}")
    del a
    gc.collect()

    print("\n-- S1.4  the importance fit that picks the metric (lane B's own 22,470 rows)")
    import lightgbm as lgb

    a = pl.read_parquet(
        FEAT_TRAIN, columns=list(dict.fromkeys(need + ["mvt_minus_sched"]))
    ).filter(pl.col("mvt_minus_aobt3").is_null())
    yv = a["TAXITIME_SEC_mvt"].to_numpy().astype(float)
    X = np.column_stack(
        [a[c].to_numpy().astype(np.float32) for c in cols_b]
        + [a[c].cast(pl.Categorical).to_physical().fill_null(-1).to_numpy().astype(np.float32) for c in cats]
    )
    names = cols_b + cats
    ds = lgb.Dataset(X, label=yv, feature_name=names, categorical_feature=cats, free_raw_data=False)
    m = lgb.train({"objective": "regression", "num_leaves": 127, "min_data_in_leaf": 50,
                   "learning_rate": 0.1, "num_threads": 2, "verbose": -1, "seed": 1},
                  ds, num_boost_round=300)
    g = m.feature_importance("gain")
    o = np.argsort(-g)
    cum = 0.0
    nc = {n: g[i] / g.sum() for i, n in enumerate(names) if n.startswith("noise_control")}
    for i in o[:22]:
        cum += g[i] / g.sum()
        print(f"     {names[i]:<34} {g[i]/g.sum()*100:6.2f}%   cum {cum*100:5.1f}%")
    print(f"  built-in noise controls: {', '.join(f'{k} {v*100:.2f}%' for k, v in sorted(nc.items()))}")
    print(f"  narrow metric ({len(CONT)+len(CAT)} cols) cumulative gain: "
          f"{sum(g[names.index(c)] for c in CONT+CAT if c in names)/g.sum()*100:.1f}%")
    del a, X, ds, m
    gc.collect()


# --------------------------------------------------------------------------------------------------
# S2 - concentration: how few rows is this RMSE made of?
# --------------------------------------------------------------------------------------------------


def s2_concentration() -> None:
    _tag("S2  concentration - how few rows lane B's 1,606.89 is actually made of")
    b = load_lane_b(with_oof=True, oof_only=True)
    e = b["e"].to_numpy()
    tot = float((e**2).sum())
    n = len(e)
    o = np.argsort(-(e**2))
    print(f"\nlane B  n={n:,}  RMSE {rmse(e):.4f}  SSE {tot:.5g}  Kish n_eff {neff(e):.2f}"
          f"  (lane A's n_eff is 19,738 - model-a-ceiling.md's routes have 2,600x the effective rows)")
    print("\n-- S2.1  SSE carried by the k worst rows")
    print(f"{'k':>6}{'share of lane-B SSE':>22}{'lane-B RMSE without them':>28}{'share of TOTAL OOF SSE':>26}")
    tot_all = tot + 3.0163e10
    for k in (1, 2, 5, 10, 17, 20, 33, 50, 100, 200, 500, 1000):
        s = float((e[o[:k]] ** 2).sum())
        print(f"{k:>6}{s/tot*100:>21.2f}%{np.sqrt((tot-s)/(n-k)):>28.2f}{s/tot_all*100:>25.2f}%")

    print("\n-- S2.2  jackknife: drop one row at a time, how far does the lane's RMSE move?")
    jk = np.sqrt((tot - e**2) / (n - 1))
    print(f"  full {rmse(e):.2f} ; worst single deletion -> {jk.min():.2f} (-{rmse(e)-jk.min():.2f} s)"
          f" ; jackknife sd of the lane RMSE {np.sqrt((n-1)/n*((jk-jk.mean())**2).sum()):.2f} s")
    r = np.random.default_rng(0)
    bsr = np.array([rmse(e[r.integers(0, n, n)]) for _ in range(2000)])
    print(f"  nonparametric bootstrap 95% CI for lane-B RMSE: "
          f"[{np.percentile(bsr,2.5):.1f}, {np.percentile(bsr,97.5):.1f}]  (point {rmse(e):.1f})")
    print("  [X] Any floor estimate for this lane inherits that interval. Quoting a lane-B floor to")
    print("      four significant figures would be a bug in the reader, not in the estimator.")

    print("\n-- S2.3  the blocks")
    print(f"{'block':<34}{'n':>7}{'RMSE':>11}{'SSE share':>11}{'n_eff':>9}{'bias':>11}")
    blocks = {
        "impossible labels (y > 11 h)": pl.col("broken"),
        "  of which INVISIBLE (ms < 2 h)": pl.col("blind_broken"),
        "LIRF, excl. broken": (pl.col("ADEP_mvt") == "LIRF") & ~pl.col("broken"),
        "LFPG, excl. broken": (pl.col("ADEP_mvt") == "LFPG") & ~pl.col("broken"),
        "other 8 airports, excl. broken": (~pl.col("ADEP_mvt").is_in(["LIRF", "LFPG"])) & ~pl.col("broken"),
    }
    for k, m in blocks.items():
        s = b.filter(m)
        ee = s["e"].to_numpy()
        print(f"{k:<34}{s.height:>7,}{rmse(ee):>11.2f}{(ee**2).sum()/tot:>11.4f}{neff(ee):>9.1f}{ee.mean():>11.1f}")

    print("\n-- S2.4  the impossible-label rows, and two threshold sweeps")
    print("  y = MVT - BLOCK. A taxi time of 11-36 hours is not a taxi time. The threshold is a")
    print("  DESCRIPTION of rows already identified by their squared error, not a model input -")
    print("  nothing downstream conditions on it, and S2.1 reaches the same rows without any threshold.")
    for thr in (10_000.0, 20_000.0, 40_000.0, 80_000.0):
        s = b.filter(pl.col("TAXITIME_SEC_mvt") > thr)
        ee = s["e"].to_numpy()
        print(f"    y > {thr:>9,.0f} s:  n={s.height:>4}  SSE share {(ee**2).sum()/tot:.4f}"
              f"  lane RMSE without them {np.sqrt((tot-(ee**2).sum())/(n-s.height)):>8.2f}")
    print("  the alternative criterion, bs = BLOCK - SCHED hours BEFORE schedule:")
    for thr in (-3600.0, -10_000.0, -40_000.0):
        s = b.filter(pl.col("bs") < thr)
        ee = s["e"].to_numpy()
        print(f"    bs < {thr:>9,.0f} s:  n={s.height:>4}  SSE share {(ee**2).sum()/tot:.4f}"
              f"  lane RMSE without them {np.sqrt((tot-(ee**2).sum())/(n-s.height)):>8.2f}")
    print("  [X] The two criteria disagree about WHICH rows are anomalous and agree almost exactly about")
    print("      HOW MUCH SSE the anomaly carries. bs < -10,000 catches 17 rows and 63.6%; y > 40,000")
    print("      catches 19 and 63.5%; ten rows are in both. The finding does not rest on the threshold.")
    br = b.filter(pl.col("broken")).sort("e")
    pl.Config.set_tbl_rows(30)
    pl.Config.set_tbl_width_chars(200)
    print(br.select("ADEP_mvt", "fold", "TAXITIME_SEC_mvt", "mvt_minus_sched", "bs", "oof", "e"))
    print("\n  Enrichment. The same criterion on the MATCHED lane:")
    am = pl.read_parquet(FEAT_TRAIN, columns=["mvt_minus_aobt3", "TAXITIME_SEC_mvt", "mvt_minus_sched"]).filter(
        pl.col("mvt_minus_aobt3").is_not_null() & pl.col("TAXITIME_SEC_mvt").is_not_null())
    nb = am.filter(pl.col("TAXITIME_SEC_mvt") > BROKEN_Y).height
    ub = load_lane_b(with_oof=False)
    nub = ub.filter(pl.col("broken")).height
    print(f"    matched   {nb:>6} / {am.height:>9,} = {nb/am.height*100:.4f}%")
    print(f"    unmatched {nub:>6} / {ub.height:>9,} = {nub/ub.height*100:.4f}%"
          f"   -> {(nub/ub.height)/(nb/am.height):.0f}x enriched in lane B")
    print("  [X] Broken labels and a missing NM flight plan are not independent events. That is the")
    print("      single fact that makes lane B's RMSE what it is.")
    print("\n  Are they VISIBLY distinguishable? ms is the only strong visible column (53.9% of gain):")
    print(f"    broken rows ms: {np.round(br['mvt_minus_sched'].to_numpy(),0)}")
    print(f"    lane-B ms quantiles p50/p90/p99: "
          f"{np.percentile(b['mvt_minus_sched'].to_numpy(),[50,90,99]).round(0)}")
    del am, ub
    gc.collect()


# --------------------------------------------------------------------------------------------------
# S3 - bias vs spread
# --------------------------------------------------------------------------------------------------


def s3_bias_vs_spread() -> None:
    _tag("S3  BIAS vs SPREAD - what an oracle first-moment correction can remove, and what survives OOF")
    print("""
The brief's warning, restated as arithmetic. For a cell K with n_K rows and mean residual b_K, removing
the cell mean removes exactly n_K * b_K^2 of SSE IN SAMPLE. That is free and it is also, on this lane,
largely fake: P1B-lirf-seasonal-bias-REJECT.md is the worked example of a real, large, identified first
moment that no honest estimator recovers. So every key below is reported THREE ways:

  in-sample   each cell's own mean subtracted from its own rows          - the oracle, circular
  out-of-fold the cell mean from the OTHER folds subtracted              - what is actually buyable
  control     the same out-of-fold arm with the cell LABELS shuffled     - what a key carrying nothing buys

A key whose out-of-fold arm does not beat its own shuffled control has no first moment to sell.
""")
    b = load_lane_b(oof_only=True)
    e = b["e"].to_numpy()
    fold = np.array(b["fold"].to_list())
    # A RANDOM 5-fold split as well as the production time folds. The production folds are MONTH-BLOCKED
    # (jan / jul / forward), so any key containing `month` is collinear with them and its out-of-fold arm
    # is structurally empty - which is P1B-lirf-seasonal-bias-REJECT.md's problem exactly. The random
    # split answers "does this structure exist at all"; the time split answers "does it transfer forward".
    rand5 = np.random.default_rng(2).integers(0, 5, len(e))
    tot = float((e**2).sum())
    rng = np.random.default_rng(11)

    keys = {
        "K1 ADEP": ["ADEP_mvt"],
        "K2 ADEP x ms-quintile": ["ADEP_mvt", "b_ms"],
        "K3 ADEP x month": ["ADEP_mvt", "month"],
        "K4 ADEP x RUNWAY": ["ADEP_mvt", "RUNWAY_mvt"],
        "K5 ADEP x STAND": ["ADEP_mvt", "STAND_mvt"],
        "K6 ADEP x ms-quintile x month": ["ADEP_mvt", "b_ms", "month"],
    }
    for scope, mask in (("whole lane B", np.ones(len(e), bool)),
                        ("excl. the 19 impossible labels", ~b["broken"].to_numpy()),
                        ("excl. impossible labels and LIRF", (~b["broken"].to_numpy())
                         & (np.array(b["ADEP_mvt"].to_list()) != "LIRF"))):
        ee, ff, rr = e[mask], fold[mask], rand5[mask]
        print(f"\n-- {scope}:  n={mask.sum():,}  base RMSE {rmse(ee):.3f}  bias {ee.mean():+.2f}")
        print(f"{'key':<32}{'cells':>7}{'in-samp':>10}{'in ctl':>10}{'rand5':>10}{'r5 ctl':>10}"
              f"{'time-OOF':>11}{'t ctl':>10}")
        for name, ks in keys.items():
            code = keycodes(b, ks)[mask]
            sh = rng.permutation(code)
            ncell = len(np.unique(code))
            cells = [
                rmse(ee - _cellmean(code, ee, None)),
                rmse(ee - _cellmean(sh, ee, None)),
                rmse(ee - _cellmean(code, ee, rr)),
                rmse(ee - _cellmean(sh, ee, rr)),
                rmse(ee - _cellmean(code, ee, ff)),
                rmse(ee - _cellmean(sh, ee, ff)),
            ]
            flag = "" if ("month" not in ks) else "   <- month is collinear with the folds"
            print(f"{name:<32}{ncell:>7,}" + "".join(f"{c:>10.2f}" if i != 4 else f"{c:>11.2f}"
                                                     for i, c in enumerate(cells)) + flag)
        gm = rmse(ee - _cellmean(np.zeros(len(ee), int), ee, ff))
        print(f"{'global constant (bias only)':<32}{1:>7}{rmse(ee-ee.mean()):>10.2f}{'-':>10}"
              f"{rmse(ee-_cellmean(np.zeros(len(ee),int),ee,rr)):>10.2f}{'-':>10}{gm:>11.2f}{'-':>10}")

    print("\n-- S3.1  the bias/spread split of lane-B SSE")
    for scope, mask in (("whole lane B", np.ones(len(e), bool)),
                        ("excl. the 19 impossible labels", ~b["broken"].to_numpy())):
        ee = e[mask]
        t = float((ee**2).sum())
        code = keycodes(b, ["ADEP_mvt"])[mask]
        within = ee - _cellmean(code, ee, None)
        print(f"  {scope}:  n={mask.sum():,}")
        print(f"    total SSE                            {t:>13.5g}  RMSE {rmse(ee):>9.2f}")
        print(f"    removed by the GLOBAL mean (pure bias){t-((ee-ee.mean())**2).sum():>13.5g}"
              f"  = {(t-((ee-ee.mean())**2).sum())/t*100:6.3f}% of SSE")
        print(f"    removed by ORACLE per-airport means   {t-(within**2).sum():>13.5g}"
              f"  = {(t-(within**2).sum())/t*100:6.3f}% of SSE")
        print(f"    within-airport (SPREAD)               {(within**2).sum():>13.5g}"
              f"  RMSE {rmse(within):>9.2f}")

def _cellmean(code: np.ndarray, e: np.ndarray, fold: np.ndarray | None) -> np.ndarray:
    """Per-cell mean residual. fold=None -> in-sample (oracle). Otherwise leave-one-fold-out."""
    out = np.zeros(len(e))
    if fold is None:
        s = np.bincount(code, weights=e)
        c = np.bincount(code)
        return (s / np.maximum(c, 1))[code]
    for f in np.unique(fold):
        m = fold == f
        s = np.bincount(code[~m], weights=e[~m], minlength=code.max() + 1)
        c = np.bincount(code[~m], minlength=code.max() + 1)
        mu = np.where(c > 0, s / np.maximum(c, 1), 0.0)
        out[m] = mu[code[m]]
    return out


# --------------------------------------------------------------------------------------------------
# S4 / S5 - twin routes
# --------------------------------------------------------------------------------------------------


def neighbour_pairs(b: pl.DataFrame, knn: int = 1, dmax: float = np.inf, shuffle_bs: bool = False,
                    seed: int = 5, cont: list[str] | None = None) -> dict:
    """For every lane-B row, its k nearest visible twins inside its own ADEP block.

    Distance: squared Euclidean over the z-scored continuous metric columns plus one unit per mismatched
    categorical. d == 0 therefore means the pair agrees on the airport and on all twelve metric columns
    exactly (STAND_mvt among them). Lane A blocked on STAND and searched inside it; lane B has 22,470
    rows over ~2,500 stands, so the block has to be the airport and the stand becomes a metric column.
    """
    rng = np.random.default_rng(seed)
    cont = cont or CONT
    out = {k: [] for k in ("d", "bi", "bj", "ei", "ej", "fi", "fj", "ap", "ba", "dt", "oi", "oj")}
    fmap = {"jan": 0, "jul": 1, "forward": 2}
    for ap in APTS:
        s = b.filter(pl.col("ADEP_mvt") == ap)
        if s.height < 8:
            continue
        bv = s["bs"].to_numpy().astype(float)
        if shuffle_bs:
            bv = rng.permutation(bv)
        ev = s["e"].to_numpy().astype(float) if "e" in s.columns else np.full(len(bv), np.nan)
        ov = s["broken"].to_numpy() if "broken" in s.columns else np.zeros(len(bv), bool)
        fv = (np.array([fmap.get(x, -1) for x in s["fold"].to_list()]) if "fold" in s.columns
              else np.full(len(bv), -1))
        ba = s["b_ms"].to_numpy()
        tv = s["MVT_TIME_UTC_mvt"].dt.epoch("s").to_numpy().astype(np.float64)
        Z = np.column_stack([np.nan_to_num(s[c].to_numpy().astype(np.float64), nan=0.0) for c in cont])
        Z = (Z - Z.mean(0)) / np.maximum(Z.std(0), 1e-9)
        C = np.column_stack(
            [s[c].cast(pl.Categorical).to_physical().fill_null(-1).to_numpy() for c in CAT]
        ).astype(np.int64)
        n = len(bv)
        d2 = (Z**2).sum(1)[:, None] + (Z**2).sum(1)[None, :] - 2 * Z @ Z.T
        for j in range(C.shape[1]):
            d2 += (C[:, j][:, None] != C[None, :, j]).astype(np.float64)
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
        for nm, arr in (("bi", bv), ("ei", ev), ("fi", fv), ("ba", ba), ("oi", ov)):
            out[nm].append(arr[ii])
        for nm, arr in (("bj", bv), ("ej", ev), ("fj", fv), ("oj", ov)):
            out[nm].append(arr[jj])
        out["ap"].append(np.full(int(m.sum()), ap))
        out["dt"].append(np.abs(tv[ii] - tv[jj]))
        del d2
        gc.collect()
        print(f"  {ap} paired ({n} rows)", flush=True)
    return {k: np.concatenate(v) for k, v in out.items()}


def s4_semivariance() -> None:
    _tag("S4  route 1 - 1-NN semivariance of bs, against the model's error ON THE SAME ROWS")
    print("""
gamma(d) = E[(bs_i - bs_j)^2]/2 estimates E[Var(bs|x)] = E[Var(y|x)], inflated by (mu_i-mu_j)^2/2 which
vanishes as d -> 0. Read alone it is a floor estimate; read beside model B's OOF RMSE on exactly the
same rows it says whether the model is at, above or below the spread of its own near-twins. The third
column is the point of the section - model-a-ceiling.md S3.2 records what happens to a reader who omits it.
""")
    b = load_lane_b()  # all 22,470 - a twin only has to supply a label
    P = neighbour_pairs(b, knn=1)
    d, bi, bj, ei = P["d"], P["bi"], P["bj"], P["ei"]
    gam = (bi - bj) ** 2 / 2
    se = ei**2
    has = np.isfinite(se)
    print(f"\nrows paired {len(d):,} of {b.height:,}; 1-NN distance quantiles "
          f"{np.percentile(d,[5,10,25,50,75,90,95]).round(3)}")
    print(f"{'d bin':<16}{'rows':>8}{'in OOF':>9}{'gamma^.5':>12}{'model RMSE same rows':>23}{'ratio':>8}{'n_eff(gamma)':>14}")
    edges = [0, .25, .5, .75, 1.0, 1.25, 1.5, 2.0, 3.0, np.inf]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d > lo) & (d <= hi)
        if m.sum() < 20:
            continue
        mm = m & has
        g = float(np.sqrt(gam[m].mean()))
        r = float(np.sqrt(se[mm].mean())) if mm.sum() else np.nan
        print(f"{f'({lo:.2f}, {hi:.2f}]':<16}{m.sum():>8,}{mm.sum():>9,}{g:>12.2f}{r:>23.2f}"
              f"{r/g:>8.3f}{neff(np.sqrt(gam[m])):>14.1f}")
    print(f"{'whole lane B':<16}{len(d):>8,}{has.sum():>9,}{np.sqrt(gam.mean()):>12.2f}"
          f"{np.sqrt(se[has].mean()):>23.2f}{np.sqrt(se[has].mean())/np.sqrt(gam.mean()):>8.3f}"
          f"{neff(np.sqrt(gam)):>14.1f}")

    print("\n-- S4.1  the same, with the broken-label rows removed from BOTH sides")
    keep = ~(P["oi"] | P["oj"])
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d > lo) & (d <= hi) & keep
        if m.sum() < 20:
            continue
        mm = m & has
        g = float(np.sqrt(gam[m].mean()))
        r = float(np.sqrt(se[mm].mean())) if mm.sum() else np.nan
        print(f"{f'({lo:.2f}, {hi:.2f}]':<16}{m.sum():>8,}{mm.sum():>9,}{g:>12.2f}{r:>23.2f}{r/g:>8.3f}")
    m = keep
    print(f"{'whole lane B':<16}{m.sum():>8,}{(m&has).sum():>9,}{np.sqrt(gam[m].mean()):>12.2f}"
          f"{np.sqrt(se[m&has].mean()):>23.2f}"
          f"{np.sqrt(se[m&has].mean())/np.sqrt(gam[m].mean()):>8.3f}")

    print("\n-- S4.2  post-stratified to the lane's own airport x ms-quintile composition")
    ap, ba = P["ap"], P["ba"]
    key = np.array([f"{a}|{x}" for a, x in zip(ap, ba)])
    w_target = {}
    kb = np.array([f"{a}|{x}" for a, x in zip(b["ADEP_mvt"].to_list(), b["b_ms"].to_numpy())])
    for k in np.unique(kb):
        w_target[k] = (kb == k).mean()
    for dm in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0):
        m = (d <= dm) & has
        if m.sum() < 100:
            continue
        num_g = num_r = wsum = 0.0
        for k in np.unique(key[m]):
            mk = m & (key == k)
            if mk.sum() < 5:
                continue
            w = w_target.get(k, 0.0)
            num_g += w * gam[mk].mean()
            num_r += w * se[mk].mean()
            wsum += w
        print(f"  d <= {dm:<5} coverage {m.sum()/has.sum():.3f}  strata weight {wsum:.3f}"
              f"  gamma^.5 {np.sqrt(num_g/wsum):>9.2f}   model RMSE same rows {np.sqrt(num_r/wsum):>9.2f}"
              f"   diff {np.sqrt(num_r/wsum)-np.sqrt(num_g/wsum):>+9.2f}")

    print("\n-- S4.3  CONTROL: bs permuted within airport. Must return the within-airport sd of bs.")
    Pc = neighbour_pairs(b, knn=1, shuffle_bs=True)
    gc_ = (Pc["bi"] - Pc["bj"]) ** 2 / 2
    sd_within = float(np.sqrt(
        b.group_by("ADEP_mvt").agg((pl.col("bs") - pl.col("bs").mean()).pow(2).sum().alias("s"),
                                   pl.len())
        .select((pl.col("s").sum() / pl.col("len").sum())).item()))
    print(f"  permuted gamma^.5 overall {np.sqrt(gc_.mean()):.2f}"
          f" ;  d<=1.0 {np.sqrt(gc_[Pc['d']<=1.0].mean()):.2f}"
          f" ;  within-ADEP sd(bs) {sd_within:.2f}")
    del P, Pc
    gc.collect()


def _rho(x: np.ndarray, y: np.ndarray, nboot: int = 2000, seed: int = 0) -> dict:
    """Twin-residual statistics for a set of pairs.

    rho_raw      = E[e_i e_j] / E[e^2]
    rho_centred  = E[(e_i - m)(e_j - m)] / E[e^2],  m = the mean residual on these pairs

    WHY BOTH, and why the centred one is the headline on lane B. E[e_i e_j] contains the square of the
    FIRST MOMENT: if the model is biased by m on this population, then even two unrelated rows have
    E[e_i e_j] = m^2 > 0. On lane A that term was 4.85^2 / 134.91^2 = 0.13% and could be ignored; on
    lane B the paired rows carry a bias of tens of seconds against an MSE of ~200^2 and it is worth
    several points of rho. The permuted-partner control in S5.1 is what detects this - it returns
    rho_raw = +0.066 with a CI that excludes zero, which is the signature of a first moment and NOT of
    a twin covariance. rho_centred therefore measures the recoverable SPREAD, and S3 has already shown
    that the first moment rho_raw would add is not recoverable out of fold at all.
    """
    x = x.astype(float)
    y = y.astype(float)
    m = float(np.concatenate([x, y]).mean())
    mse = float(((x**2 + y**2) / 2).mean())
    raw = float((x * y).mean()) / mse
    cen = float(((x - m) * (y - m)).mean()) / mse
    r = np.random.default_rng(seed)
    n = len(x)
    br = np.empty(nboot)
    bc = np.empty(nboot)
    for k in range(nboot):
        i = r.integers(0, n, n)
        xs, ys = x[i], y[i]
        ms = float(np.concatenate([xs, ys]).mean())
        d = ((xs**2 + ys**2) / 2).mean()
        br[k] = (xs * ys).mean() / d
        bc[k] = ((xs - ms) * (ys - ms)).mean() / d
    return {"mse": mse, "raw": raw, "cen": cen, "bias": m,
            "ci_raw": tuple(np.percentile(br, [2.5, 97.5])),
            "ci_cen": tuple(np.percentile(bc, [2.5, 97.5])),
            "neff": neff(np.sqrt(np.abs((x - m) * (y - m))))}


def _ci(lo: float, hi: float) -> str:
    return f"[{lo:+.3f}, {hi:+.3f}]"


def s5_rho() -> None:
    _tag("S5  route 2 - twin-residual covariance: the systematic share rho, and the floor")
    print("""
    e_i = (yhat(x) - mu(x)) + (mu(x) - y_i)      for two rows with the SAME visible x, one model
    E[e_i e_j] = E[(yhat - mu)^2]                the systematic, RECOVERABLE part
    E[e_i^2]   = E[(yhat - mu)^2] + E[Var(y|x)] = MSE
    rho = E[e_i e_j]/MSE ,  floor = RMSE * sqrt(1 - rho)

At d > 0 the covariance UNDER-states the systematic part, so every floor below is an UPPER bound that
falls towards the truth as d -> 0. Pairs are restricted to the same OOF fold so yhat comes from one
model. Six nearest neighbours per row, capped at d <= 1.0. Both rho_raw and rho_centred are reported;
see _rho's docstring for why the centred one is the headline here and the raw one is not.
""")
    b = load_lane_b(oof_only=True)
    P = neighbour_pairs(b, knn=6, dmax=1.0)
    same = (P["fi"] == P["fj"]) & np.isfinite(P["ei"]) & np.isfinite(P["ej"])
    d, ei, ej = P["d"][same], P["ei"][same], P["ej"][same]
    ap = P["ap"][same]
    brk = (P["oi"] | P["oj"])[same]
    dt = P["dt"][same]
    lane = rmse(b["e"].to_numpy())
    print(f"\npairs at d<=1.0 with both rows in the OOF and the same fold: {len(d):,}"
          f"   (lane-B rows {b.height:,}, lane-B RMSE {lane:.4f})")
    print(f"  [X] lane A had 140,520 such pairs. This is 1.1% of that, on a lane whose Kish n_eff is 7.6.")

    HDR = (f"{'':<26}{'pairs':>7}{'MSE^.5':>9}{'bias':>8}{'rho_raw':>9}{'rho_cen':>9}"
           f"{'rho_cen 95% CI':>20}{'n_eff':>8}")

    def row(label, m, lane_rmse=None, show_floor=True):
        if m.sum() < 30:
            print(f"{label:<26}{m.sum():>7,}   too few pairs to report")
            return None
        r = _rho(ei[m], ej[m])
        ci = r["ci_cen"]
        out = (f"{label:<26}{m.sum():>7,}{np.sqrt(r['mse']):>9.1f}{r['bias']:>8.1f}"
               f"{r['raw']:>9.4f}{r['cen']:>9.4f}{_ci(*ci):>20}{r['neff']:>8.1f}")
        if show_floor and lane_rmse is not None:
            fl = lane_rmse * np.sqrt(max(0.0, 1 - r["cen"]))
            flo = lane_rmse * np.sqrt(max(0.0, 1 - ci[1]))
            fhi = lane_rmse * np.sqrt(max(0.0, 1 - ci[0]))
            out += f"   floor {fl:>8.1f} [{flo:.1f}, {fhi:.1f}]"
        print(out)
        return r

    print("\n-- S5.0  rho by distance, on the rows that HAVE twins")
    print(HDR)
    for lo, hi in ((0, .25), (.25, .5), (.5, .75), (.75, 1.0)):
        row(f"({lo:.2f}, {hi:.2f}]", (d > lo) & (d <= hi))
    r_all = row("all d <= 1.0", d <= 1.0)
    print(f"  [X] MSE^.5 on the paired rows is {np.sqrt(r_all['mse']):.0f} against the lane's {lane:.0f}."
          f" The pairs are the ORDINARY")
    print("      rows; the lane's error lives on rows that have no twin. S5.2 is where that is handled.")

    print("\n-- S5.1  CONTROLS (both were part of the design, not added after the headline was read)")
    r = np.random.default_rng(3)
    m = d <= 1.0
    rc = _rho(ei[m], r.permutation(ej[m]))
    print(f"  negative control, partner permuted:")
    print(f"     rho_raw {rc['raw']:+.4f}  CI [{rc['ci_raw'][0]:+.4f}, {rc['ci_raw'][1]:+.4f}]"
          f"   {'contains 0' if rc['ci_raw'][0] < 0 < rc['ci_raw'][1] else 'EXCLUDES 0'}")
    print(f"     rho_cen {rc['cen']:+.4f}  CI [{rc['ci_cen'][0]:+.4f}, {rc['ci_cen'][1]:+.4f}]"
          f"   {'contains 0 - ok' if rc['ci_cen'][0] < 0 < rc['ci_cen'][1] else 'EXCLUDES 0 - investigate'}")
    print("     [X] THE RAW CONTROL FAILS AND THE CENTRED ONE PASSES. That is the whole argument for")
    print("         centring: rho_raw is counting the first moment, which S3 shows is not sellable.")
    print("  positive control: a deliberately crippled visible predictor of y - the out-of-half mean of")
    print("  y over (ADEP x ms-quintile). It is a legitimate visible predictor and it is badly")
    print("  underfitted, so it HAS systematic error. If the estimator cannot see that, it is useless.")
    code = keycodes(b, ["ADEP_mvt", "b_ms"])
    yv = b["TAXITIME_SEC_mvt"].to_numpy().astype(float)
    half = np.random.default_rng(7).integers(0, 2, len(yv))
    pred = np.zeros(len(yv))
    for h in (0, 1):
        sm = np.bincount(code[half != h], weights=yv[half != h], minlength=code.max() + 1)
        cn = np.bincount(code[half != h], minlength=code.max() + 1)
        mu = np.where(cn > 0, sm / np.maximum(cn, 1), yv.mean())
        pred[half == h] = mu[code[half == h]]
    ec = pred - yv
    b2 = b.with_columns(pl.Series("e", ec))
    P2 = neighbour_pairs(b2, knn=6, dmax=1.0)
    s2 = (P2["fi"] == P2["fj"]) & np.isfinite(P2["ei"]) & np.isfinite(P2["ej"])
    d2, e2i, e2j, ap2 = P2["d"][s2], P2["ei"][s2], P2["ej"][s2], P2["ap"][s2]
    brk2 = (P2["oi"] | P2["oj"])[s2]
    apv0 = np.array(b["ADEP_mvt"].to_list())
    brv0 = b["broken"].to_numpy()
    print("  Compared on EXACTLY the two blocks where S5.2 uses a measured rho, because a pooled")
    print("  comparison on this lane is decided by single outlier pairs (the caveat is S9's second bullet):")
    print(f"{'block':<28}{'arm':<12}{'RMSE on block':>14}{'pairs':>7}{'rho_cen':>9}{'rho_cen 95% CI':>20}")
    for nm, rowm, pm, pm2 in (
        ("LFPG, excl. impossible", (apv0 == "LFPG") & ~brv0,
         (d <= 1.0) & (ap == "LFPG") & ~brk, (d2 <= 1.0) & (ap2 == "LFPG") & ~brk2),
        ("other 8, excl. impossible", ~np.isin(apv0, ["LIRF", "LFPG"]) & ~brv0,
         (d <= 1.0) & ~np.isin(ap, ["LIRF", "LFPG"]) & ~brk,
         (d2 <= 1.0) & ~np.isin(ap2, ["LIRF", "LFPG"]) & ~brk2),
    ):
        for arm, ev, mk in (("model B", e_dummy_model := None, pm), ("crippled", None, pm2)):
            if arm == "model B":
                rr = _rho(ei[mk], ej[mk])
                br_ = rmse(b["e"].to_numpy()[rowm])
            else:
                rr = _rho(e2i[mk], e2j[mk])
                br_ = rmse(ec[rowm])
            print(f"{nm if arm=='model B' else '':<28}{arm:<12}{br_:>14.1f}{mk.sum():>7,}"
                  f"{rr['cen']:>9.4f}{_ci(*rr['ci_cen']):>20}")
    print("  [S] On both blocks the crippled predictor is worse AND measures a larger systematic share")
    print("      than model B. The estimator sees systematic error when it is there.")
    print("  [X] Pooled over the whole lane the positive control reads rho_cen -0.16, i.e. it FAILS.")
    print("      That is not a defect of the crippled arm, it is the same outlier-domination that S2")
    print("      measures as n_eff = 7.6: one pair containing an extreme row decides the pooled number.")
    print("      It is recorded rather than dropped, and it is the reason S5.2 is assembled block-wise.")

    print("\n-- S5.2  the floor, assembled BLOCK BY BLOCK rather than by carrying one rho to the lane")
    print("""  The lane's SSE and the lane's twins live on different rows. Applying a rho measured on the
  ordinary rows to the whole lane would credit the 19 impossible labels with a 32% recoverable share
  they cannot have. So each block gets its own rho where one is measurable, rho = 0 where it is not,
  and the lane floor is reassembled in SSE:   floor^2 = sum_block SSE_block * (1 - rho_block) / n.""")
    e_all = b["e"].to_numpy()
    apv = np.array(b["ADEP_mvt"].to_list())
    brv = b["broken"].to_numpy()
    blocks = [
        ("19 impossible labels", brv, (d < -1)),
        ("LIRF, excl. impossible", (apv == "LIRF") & ~brv, (d <= 1.0) & (ap == "LIRF") & ~brk),
        ("LFPG, excl. impossible", (apv == "LFPG") & ~brv, (d <= 1.0) & (ap == "LFPG") & ~brk),
        ("other 8, excl. impossible", ~np.isin(apv, ["LIRF", "LFPG"]) & ~brv,
         (d <= 1.0) & ~np.isin(ap, ["LIRF", "LFPG"]) & ~brk),
    ]
    print(f"\n{'block':<27}{'n':>6}{'SSE':>12}{'RMSE':>10}{'pairs':>7}{'rho_cen':>9}"
          f"{'rho 95% CI':>20}{'block floor':>13}")
    tot_sse = float((e_all**2).sum())
    acc_lo = acc = acc_hi = 0.0
    for name, rowmask, pairmask in blocks:
        sse = float((e_all[rowmask] ** 2).sum())
        nb_ = int(rowmask.sum())
        if pairmask.sum() >= 30:
            rr = _rho(ei[pairmask], ej[pairmask])
            rc_, ci = rr["cen"], rr["ci_cen"]
            pr = int(pairmask.sum())
        else:
            rc_, ci, pr = 0.0, (0.0, 0.0), int(pairmask.sum())
        acc += sse * (1 - max(0.0, rc_))
        acc_lo += sse * (1 - max(0.0, ci[1]))
        acc_hi += sse * (1 - max(0.0, ci[0]))
        tag = "" if pr >= 30 else "  (rho=0 assumed: too few pairs)"
        print(f"{name:<27}{nb_:>6,}{sse:>12.4g}{np.sqrt(sse/nb_):>10.1f}{pr:>7}{rc_:>9.4f}"
              f"{_ci(*ci):>20}{np.sqrt(sse*(1-max(0.0,rc_))/nb_):>13.1f}{tag}")
    n_all = len(e_all)
    print(f"\n  LANE B floor, reassembled:  {np.sqrt(acc/n_all):.1f}"
          f"   [{np.sqrt(acc_lo/n_all):.1f}, {np.sqrt(acc_hi/n_all):.1f}]"
          f"   against today's {lane:.1f}")
    print(f"  recoverable: {lane - np.sqrt(acc/n_all):.1f} s of lane-B RMSE"
          f"  ({(tot_sse-acc)/tot_sse*100:.2f}% of lane-B SSE)")
    print("  And the naive version, for contrast - one rho carried to the whole lane:")
    naive = lane * np.sqrt(max(0.0, 1 - r_all["cen"]))
    print(f"    {lane:.1f} * sqrt(1 - {r_all['cen']:.4f}) = {naive:.1f}  <- says {lane-naive:.0f} s is")
    print(f"    recoverable against the block-wise {lane-np.sqrt(acc/n_all):.0f} s. The difference is")
    print("    entirely the impossible labels being credited with a recoverable share they cannot have.")

    print("\n-- S5.2a  SENSITIVITY to the two blocks where rho could NOT be measured")
    print("  rho = 0 on the impossible labels and at LIRF is an ASSUMPTION, not a measurement. Those")
    print("  two blocks are 88.1% of lane-B SSE, so the floor is only as good as the assumption. What")
    print("  happens if it is wrong, in the direction that makes the floor LOWER (more recoverable):")
    sse_imp = float((e_all[brv] ** 2).sum())
    sse_lirf = float((e_all[(apv == "LIRF") & ~brv] ** 2).sum())
    rest = acc - sse_imp - sse_lirf
    print(f"{'rho(impossible)':>17}{'rho(LIRF)':>12}{'lane-B floor':>15}{'recoverable':>13}"
          f"{'board @0.747':>14}")
    for ri, rl in ((0.0, 0.0), (0.0, 0.20), (0.0, 0.50), (0.10, 0.27), (0.0, 1.0), (1.0, 1.0)):
        f2 = (rest + sse_imp * (1 - ri) + sse_lirf * (1 - rl)) / n_all
        fl = np.sqrt(f2)
        new_oof = np.sqrt((3.0163e10 + f2 * n_all) / 672_428)
        print(f"{ri:>17.2f}{rl:>12.2f}{fl:>15.1f}{lane-fl:>13.1f}"
              f"{BOARD_V14 + (new_oof - OOF_V14) * TRANSFER:>14.2f}")
    print("  The last two rows are not estimates. They are the arithmetic limit: LIRF perfectly")
    print("  predicted, and then BOTH unmeasurable blocks perfectly predicted - 88% of lane-B SSE")
    print("  deleted, with LFPG and the other eight still at their measured floors. S7 carries the")
    print("  one further step, lane B to exactly zero.")

    print("\n-- S5.3  is the covariance a shared operational shock? (time separation)")
    print(f"{'|t_i - t_j|':<26}{'pairs':>7}{'MSE^.5':>9}{'bias':>8}{'rho_raw':>9}{'rho_cen':>9}{'rho_cen 95% CI':>20}")
    for lab, lo, hi in (("< 1 day", 0, 86400), ("1-7 days", 86400, 7 * 86400),
                        ("7-30 days", 7 * 86400, 30 * 86400), ("30-365 days", 30 * 86400, 1e12)):
        mm = (d <= 1.0) & (dt >= lo) & (dt < hi)
        if mm.sum() < 30:
            continue
        rr = _rho(ei[mm], ej[mm])
        print(f"{lab:<26}{mm.sum():>7,}{np.sqrt(rr['mse']):>9.1f}{rr['bias']:>8.1f}"
              f"{rr['raw']:>9.4f}{rr['cen']:>9.4f}{_ci(*rr['ci_cen']):>20}")

    print("\n-- S5.4  the WIDE metric (22 columns instead of 12)")
    P3 = neighbour_pairs(b, knn=6, dmax=2.0, cont=CONT_WIDE)
    s3 = (P3["fi"] == P3["fj"]) & np.isfinite(P3["ei"]) & np.isfinite(P3["ej"])
    d3, e3i, e3j = P3["d"][s3], P3["ei"][s3], P3["ej"][s3]
    brk3 = (P3["oi"] | P3["oj"])[s3]
    print(f"{'d bin (wide)':<26}{'pairs':>7}{'MSE^.5':>9}{'bias':>8}{'rho_raw':>9}{'rho_cen':>9}{'rho_cen 95% CI':>20}")
    for lo, hi in ((0, 1.0), (1.0, 1.5), (1.5, 2.0)):
        mm = (d3 > lo) & (d3 <= hi) & ~brk3
        if mm.sum() < 30:
            continue
        rr = _rho(e3i[mm], e3j[mm])
        lab = f"({lo:.2f}, {hi:.2f}]"
        print(f"{lab:<26}{mm.sum():>7,}{np.sqrt(rr['mse']):>9.1f}{rr['bias']:>8.1f}"
              f"{rr['raw']:>9.4f}{rr['cen']:>9.4f}{_ci(*rr['ci_cen']):>20}")
    del P, P2, P3
    gc.collect()


def s6_oracles() -> None:
    _tag("S6  oracle bounds - what a PERFECT version of each identified lane-B route would be worth")
    print("""
These are ORACLES: they read the label. They are not floors and they are not achievable. They bound
from above what each named route could EVER return, which is exactly what is needed to decide whether
a route is worth a slot. model-a-ceiling.md S6 uses the same device.
""")
    b = load_lane_b(oof_only=True)
    e = b["e"].to_numpy()
    y = b["TAXITIME_SEC_mvt"].to_numpy().astype(float)
    ms = b["mvt_minus_sched"].to_numpy().astype(float)
    oof = b["oof"].to_numpy().astype(float)
    bs = b["bs"].to_numpy()
    ap = np.array(b["ADEP_mvt"].to_list())
    tot = float((e**2).sum())
    n = len(e)
    base = rmse(e)
    print(f"lane B baseline  n={n:,}  RMSE {base:.4f}")

    print("\n-- S6.1  the LIRF fill gate (P1-stage2-gate-FAILED.md closed the ACHIEVABLE version at 0.9201)")
    print("""  Production already applies the ms reparameterisation at LIRF (configs/main_gap.yaml:
  unmatched_sched_reparam, airports [LIRF]): model B fits y - ms there, so its prediction is ms + ghat
  and a `fill` row - one whose BLOCK stamp was filled from SCHED, so y == ms exactly - is the row where
  the right answer is ghat = 0. A PERFECT fill classifier is therefore exactly: set ghat = 0 where the
  row is a fill, leave the model alone elsewhere. That is the oracle below, and it is the ceiling on
  the whole stage-2 route.""")
    fill = np.abs(bs) <= 10.0
    L = ap == "LIRF"
    print(f"  LIRF OOF rows {int(L.sum())}, of which fill (|bs| <= 10 s) {int(fill[L].sum())}"
          f" = {fill[L].mean()*100:.1f}%")
    e_h = e.copy()
    e_h[L & fill] = (ms - y)[L & fill]
    print(f"  ORACLE fill gate at LIRF (ghat := 0 on fill rows, model untouched elsewhere):")
    print(f"    on the {int((L&fill).sum())} LIRF fill rows: RMSE {rmse(e[L&fill]):>9.1f}"
          f" -> {rmse(e_h[L&fill]):>7.1f}")
    print(f"    LIRF overall:  {rmse(e[L]):>9.1f} -> {rmse(e_h[L]):>9.1f}")
    print(f"    LANE B:        {base:>9.2f} -> {rmse(e_h):>9.2f}   ({rmse(e_h)-base:+.2f} s)")
    print(f"    SSE removed: {(tot-(e_h**2).sum())/tot*100:.2f}% of lane B")
    print("  [S] A PERFECT fill classifier is worth 30 s of lane-B RMSE and no more. P1 measured the")
    print("      best achievable at AUC 0.9201 against a 0.93 gate, so the achievable share of 30 s is")
    print("      a fraction of it. S7 prices the perfect version on the board.")
    print("  The same oracle applied at ALL TEN airports, not just LIRF:")
    e_g = e.copy()
    e_g[fill] = (ms - y)[fill]
    print(f"    {int(fill.sum())} fill rows lane-wide;  LANE B {base:>9.2f} -> {rmse(e_g):>9.2f}"
          f"   ({rmse(e_g)-base:+.2f} s)")
    print("  And the gate's DOWNSIDE if it fires when it should not - predict ms on EVERY LIRF row:")
    e_b = e.copy()
    e_b[L] = (ms - y)[L]
    print(f"    LIRF {rmse(e[L]):>9.1f} -> {rmse(e_b[L]):>9.1f} ;  LANE B {base:>9.2f} -> {rmse(e_b):>9.2f}"
          f"   ({rmse(e_b)-base:+.2f} s)")
    print("  [X] The route is asymmetric: +30 s if the gate is perfect, -291 s if it fires everywhere.")
    print("      That asymmetry is why P1's 0.93 gate was set where it was.")

    print("\n-- S6.2  the 19 impossible-label rows")
    brk = b["broken"].to_numpy()
    blind = b["blind_broken"].to_numpy()
    print(f"  {int(brk.sum())} rows, {(e[brk]**2).sum()/tot*100:.2f}% of lane-B SSE;"
          f" of which {int(blind.sum())} are VISIBLY ORDINARY (ms < 2 h)"
          f" and carry {(e[blind]**2).sum()/tot*100:.2f}%.")
    e_o = e.copy()
    e_o[brk] = 0.0
    print(f"  ORACLE that predicts all 19 EXACTLY:            lane B {base:>9.2f} -> {rmse(e_o):>9.2f}")
    e_o2 = e.copy()
    e_o2[blind] = 0.0
    print(f"  ORACLE on just the 2 invisible LFPG rows:       lane B {base:>9.2f} -> {rmse(e_o2):>9.2f}")
    print("  What would a REAL predictor do on the invisible ones? The conditional mean of y given")
    print("  visible x is (1-q)*ordinary + q*(ordinary + jump), q = the rate of the anomaly there:")
    ub = load_lane_b(with_oof=False)
    for ap_ in ("LFPG",):
        m = blind & (ap == ap_)
        q = ub.filter((pl.col("ADEP_mvt") == ap_) & pl.col("blind_broken")).height / ub.filter(
            pl.col("ADEP_mvt") == ap_).height
        jump = float(np.median(-bs[m]))
        print(f"    {ap_}: q = {q:.5f} ({ub.filter((pl.col('ADEP_mvt')==ap_) & pl.col('blind_broken')).height}"
              f" of {ub.filter(pl.col('ADEP_mvt')==ap_).height} labelled rows), median jump {jump:>9,.0f} s")
        print(f"          the optimal shift of EVERY {ap_} lane-B prediction is q*jump = {q*jump:>5.1f} s,")
        print(f"          which leaves {(1-q)*100:.3f}% of the jump on the row that actually has it.")
        eopt = e.copy()
        mm = np.array(b["ADEP_mvt"].to_list()) == ap_
        eopt[mm] = eopt[mm] - q * jump
        print(f"          applying it: lane B {base:>9.2f} -> {rmse(eopt):>9.2f} ({rmse(eopt)-base:+.3f} s)")
    del ub
    print("  [X] The conditional-mean predictor - the thing the floor is DEFINED as - eats essentially")
    print("      the whole squared error on these rows. That IS the floor, not a modelling failure.")
    print("      And the corruption is in BLOCK_TIME_UTC_mvt, which is one of the two columns blanked")
    print("      on every scored row (src/prc/schema.py:52). There is nothing to see it with.")
    print("  Direct check - the best constant shift applied to ALL lane-B predictions:")
    print(f"    shift {-e.mean():+.1f} s  ->  lane B {base:.2f} -> {rmse(e-e.mean()):.2f}")

    print("\n-- S6.3  the missing-clock route, priced on the same axis")
    print("  missing-clock.md S4 measured a fully-sighted counterfactual at 378.4 reweighted against")
    print("  554.0 blind, i.e. sight is a 1.31x penalty. Applying that ratio to the lane as an ORACLE:")
    for lab, m in (("whole lane B", np.ones(n, bool)),
                   ("excl. broken labels", ~brk),
                   ("excl. broken and LIRF", ~brk & (ap != "LIRF"))):
        print(f"    {lab:<24} RMSE {rmse(e[m]):>9.2f}  /1.31 -> {rmse(e[m])/1.31:>9.2f}"
              f"   (recovers {rmse(e[m])-rmse(e[m])/1.31:>8.2f} s of that subset)")
    print("  [X] The ratio is measured on MATCHED rows reweighted to the lane's composition. It does not")
    print("      apply to a broken label, which no amount of sight fixes - the clock that is missing is")
    print("      NM's, and the corruption is in the airport's own BLOCK stamp.")


# --------------------------------------------------------------------------------------------------
# S7 - the board price
# --------------------------------------------------------------------------------------------------


def s7_price() -> None:
    _tag("S7  what lane B's remaining headroom is worth on the board")
    b = load_lane_b(oof_only=True)
    e = b["e"].to_numpy()
    y = b["TAXITIME_SEC_mvt"].to_numpy().astype(float)
    ms = b["mvt_minus_sched"].to_numpy().astype(float)
    bs = b["bs"].to_numpy()
    ap = np.array(b["ADEP_mvt"].to_list())
    brk = b["broken"].to_numpy()
    L = ap == "LIRF"
    sse_b = float((e**2).sum())
    n_b = len(e)
    # lane A's SSE taken from the same OOF rather than hardcoded, so the table reproduces 277.5398
    oof = pl.read_parquet(OOF)
    fol = pl.read_parquet(OOF_FOLDS).filter(pl.col("pooled")).select("MVT_ID_mvt", "fold")
    ft = pl.read_parquet(FEAT_TRAIN, columns=["MVT_ID_mvt", "TAXITIME_SEC_mvt", "mvt_minus_aobt3"])
    ja = (ft.join(oof, on="MVT_ID_mvt", how="inner").join(fol, on="MVT_ID_mvt", how="inner")
          .filter(pl.col("mvt_minus_aobt3").is_not_null()))
    ea = (ja["oof"] - ja["TAXITIME_SEC_mvt"]).to_numpy()
    sse_a, n_a = float((ea**2).sum()), len(ea)
    del ft, ja, oof
    gc.collect()
    print(f"\npooled OOF {np.sqrt((sse_a+sse_b)/(n_a+n_b)):.4f} (target {OOF_V14})   board v14 {BOARD_V14}")
    print(f"lane A {np.sqrt(sse_a/n_a):.4f} on {n_a:,} rows;  lane B {np.sqrt(sse_b/n_b):.4f} on {n_b:,}")
    print(f"additive gap {GAP:+.4f}.  The LEVEL of that gap is unstable (model-a-ceiling.md S8.1 moves it")
    print(f"over 125 s by re-pooling the folds), so the priced column is the DERIVATIVE:")
    print(f"dBoard/dOOF = {TRANSFER} (n = 1, model-a-ceiling.md S8.2). Both columns are shown.")

    def price(label, new_sse_b, note=""):
        new_oof = np.sqrt((sse_a + new_sse_b) / (n_a + n_b))
        d = new_oof - OOF_V14
        print(f"{label:<45}{np.sqrt(new_sse_b/n_b):>10.1f}{new_oof:>11.2f}{d:>9.2f}"
              f"{BOARD_V14+d:>11.2f}{BOARD_V14+d*TRANSFER:>12.2f}  {note}")

    print(f"\n{'counterfactual on lane B':<45}{'lane B':>10}{'pooled OOF':>11}{'dOOF':>9}"
          f"{'board@1.0':>11}{'board@0.747':>12}")
    price("today (v14)", sse_b)
    print("  -- what the floor says is actually available --")
    for lab, fl in (("MEASURED FLOOR, block-wise (S5.2)", 1586.2),
                    ("   its 95% band, low end", 1573.8),
                    ("   its 95% band, high end", 1597.9),
                    ("floor if rho(LIRF)=0.27 like the rest (S5.2a)", 1476.6)):
        price(lab, fl**2 * n_b)
    print("  -- named routes, as ORACLES (upper bounds, not achievable) --")
    e_h = e.copy()
    e_h[L & (np.abs(bs) <= 10)] = (ms - y)[L & (np.abs(bs) <= 10)]
    price("PERFECT LIRF fill gate (S6.1)", float((e_h**2).sum()), "<- P1 stage 2's route, closed at AUC 0.9201")
    e3 = e.copy()
    e3[~brk] = e3[~brk] / 1.31
    price("full NM sight on non-impossible rows (S6.3)", float((e3**2).sum()), "<- missing-clock.md's route")
    e4 = e.copy()
    m9 = ~brk & ~L
    e4[m9] = e4[m9] / 1.31
    price("   ...only at the 9 non-LIRF airports", float((e4**2).sum()))
    price("the 19 impossible labels predicted exactly", float((e[~brk] ** 2).sum()),
          "<- not a route; nothing visible sees them")
    print("  -- the brief's arithmetic, for comparison --")
    for f in (0.75, 0.50):
        price(f"lane B RMSE x {f:.2f} ({(1-f)*100:.0f}% better)", sse_b * f * f)
    price("lane B -> 0", 0.0)

    print("\n-- S7.1  what fraction of lane B is even in play")
    print(f"  the 19 impossible labels:        {(e[brk]**2).sum()/sse_b*100:>6.2f}% of lane-B SSE"
          f"   = {(e[brk]**2).sum()/(sse_a+sse_b)*100:>5.2f}% of the WHOLE pooled OOF SSE")
    print(f"  LIRF, excl. impossible:          {(e[L&~brk]**2).sum()/sse_b*100:>6.2f}%"
          f"   = {(e[L&~brk]**2).sum()/(sse_a+sse_b)*100:>5.2f}%")
    print(f"  the other 9 airports, excl. imp.:{(e[~L&~brk]**2).sum()/sse_b*100:>6.2f}%"
          f"   = {(e[~L&~brk]**2).sum()/(sse_a+sse_b)*100:>5.2f}%")
    print("  [X] The block that ten months of feature work could plausibly touch - nine ordinary")
    print("      airports, no LIRF switch, no broken labels - is 5% of the problem's squared error,")
    print(f"      and its measured floor is {508.4:.0f} against {rmse(e[~L&~brk]):.0f} today (S5.2).")

    print("\n-- S7.2  the scored lane is a LOTTERY on how many impossible labels it contains")
    ub = load_lane_b(with_oof=False)
    rate = ub.filter(pl.col("broken")).height / ub.height
    print(f"  impossible-label rate on the 22,470 labelled unmatched rows: {rate*100:.4f}%"
          f"  ({ub.filter(pl.col('broken')).height} rows)")
    print(f"  5,290 scored unmatched rows x that rate = {5290*rate:.1f} expected")
    print("  Monte Carlo: draw k ~ Binomial(5290, rate), then k error magnitudes with replacement from")
    print("  the 19 observed OOF ones, and ask what they alone contribute to a 344,841-row board RMSE.")
    rng = np.random.default_rng(4)
    eb = np.abs(e[brk])
    sims = []
    for _ in range(20000):
        k = rng.binomial(5290, rate)
        sims.append(0.0 if k == 0 else float((rng.choice(eb, k) ** 2).sum()))
    sims = np.array(sims)
    q = np.percentile(sims, [5, 25, 50, 75, 95])
    print(f"{'percentile':<14}{'5%':>10}{'25%':>10}{'50%':>10}{'75%':>10}{'95%':>10}")
    print(f"{'their SSE':<14}" + "".join(f"{x:>10.3g}" for x in q))
    print(f"{'board RMSE':<14}" + "".join(f"{np.sqrt(x/BOARD_N):>10.1f}" for x in q)
          + "   <- seconds of board RMSE, in quadrature, from these rows ALONE")
    print(f"  mean over draws: {np.sqrt(sims.mean()/BOARD_N):.1f} s;  P(contribution > 100 s) = "
          f"{(sims/BOARD_N > 100**2).mean():.2f}")
    print("  [X] This is not a modelling uncertainty, it is a sampling one, and it is larger than every")
    print("      improvement this project is chasing. The board's lane-B term is set by how many broken")
    print("      BLOCK stamps happened to land in Jan and Jul 2026, which nobody can see or influence.")
    print("      It is also why the OOF->board gap's LEVEL is unstable (model-a-ceiling.md S8.1): the")
    print("      [X] inference is that these rows are a large part of that instability.")
    del ub
    gc.collect()


SECTIONS = {"S1": s1_baseline, "S2": s2_concentration, "S3": s3_bias_vs_spread,
            "S4": s4_semivariance, "S5": s5_rho, "S6": s6_oracles, "S7": s7_price}

if __name__ == "__main__":
    want = [a for a in sys.argv[1:] if a in SECTIONS] or list(SECTIONS)
    for k in want:
        t = time.time()
        SECTIONS[k]()
        print(f"\n[{k} done in {time.time()-t:.1f}s]", flush=True)
