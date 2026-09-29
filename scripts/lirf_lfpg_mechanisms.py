"""Where lane-A error sits at LIRF and LFPG, and one mechanism found by reading the rows (read-only).

Run:  POLARS_MAX_THREADS=2 .venv/bin/python scripts/lirf_lfpg_mechanisms.py [A B C D E F]

Nothing is written to outputs/, src/, configs/ or data/. Scratch artefacts go to $SCRATCH (default /tmp).
Every fit is a diagnostic: either on the production OOF residual or on a label-derived *diagnostic target*
(the fill flag) that is never used as a feature.

Parity rules obeyed throughout:
  * no feature reads BLOCK_TIME_UTC_mvt or TAXITIME_SEC_mvt of any DEPARTURE (own or neighbour);
  * the only BLOCK_TIME_UTC_mvt read by a feature is that of ARRIVAL rows, which ranking.parquet keeps on all
    344,693 scored arrivals (checked in section B) - the same source stand_arrival_gap already uses;
  * `bs = BLOCK - SCHED` of a departure appears only as the diagnostic label `fill`, never as a feature.

Sections
  A  where the error is: LIRF / LFPG lane-A SSE by kind (schedule fill, BLOCK-vs-LOBT band, worst days)
  B  the arrival mirror of the LIRF schedule fill: rates by airport and month, train and ranking
  C  does the arrival side predict the departure fill? month-blocked classifier at LIRF, with controls
  D  calibrated residual instrument (delay-mechanisms.md S5-S7 recipe) on the new arm, with controls
  E  extended instrument: same, but with mvt_minus_sched and the OOF prediction in every arm's base
  F  LFPG: worst days and the early-off-block rows
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import polars as pl

W = "/home/user/prc-data-challenge/"
OOF = W + "experiments/oof_main_gap-62dd7c6e88.parquet"
FOLDS = W + "experiments/oof_main_gap-62dd7c6e88_folds.parquet"
FEAT = W + "data/work/feat_train.parquet"
RAW_TRAIN = sorted(glob.glob(W + "data/raw/training_*.parquet"))
RAW_RANK = W + "data/raw/ranking.parquet"
SCRATCH = os.environ.get("SCRATCH", "/tmp")
FILL_TOL = 12  # seconds; every clock here is on a whole minute with about +-6 s of jitter (section A prints it)

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(250)
pl.Config.set_tbl_cols(20)


def secs(a: str, b: str) -> pl.Expr:
    return (pl.col(a) - pl.col(b)).dt.total_seconds()


def rss_mb() -> float:
    try:
        cg = open("/proc/self/cgroup").read()
        path = [ln.split(":")[2] for ln in cg.splitlines() if ":memory:" in ln][0]
        for ln in open(f"/sys/fs/cgroup/memory{path}/memory.stat"):
            if ln.startswith("total_rss "):
                return int(ln.split()[1]) / 2**20
    except Exception:  # noqa: BLE001
        pass
    return float("nan")


# --------------------------------------------------------------------------------------------- loading
def load_lane_a(columns: list[str] | None = None) -> pl.DataFrame:
    """Pooled OOF rows (jan/jul/forward; jul_fwd excluded), matched lane only, with r and g."""
    o = pl.read_parquet(OOF)
    fo = pl.read_parquet(FOLDS).filter(pl.col("pooled")).select("MVT_ID_mvt", "fold")
    d = pl.read_parquet(FEAT, columns=columns)
    return (
        d.join(o, on="MVT_ID_mvt")
        .join(fo, on="MVT_ID_mvt")
        .with_columns(
            (pl.col("oof") - pl.col("TAXITIME_SEC_mvt")).alias("r"),
            (pl.col("TAXITIME_SEC_mvt") - pl.col("mvt_minus_aobt3")).alias("g"),
        )
        .filter(pl.col("mvt_minus_aobt3").is_not_null())
    )


def arrivals(frame: str) -> pl.DataFrame:
    """ARR rows of the training or ranking frame. BLOCK here is the arrival IN-block, visible in ranking."""
    src = pl.scan_parquet(RAW_TRAIN) if frame == "train" else pl.scan_parquet(RAW_RANK)
    return (
        src.filter(pl.col("PHASE_mvt") == "ARR")
        .select(
            pl.col("ADES_mvt").alias("ADEP_mvt"),
            "STAND_mvt",
            pl.col("AIRCRAFT_OPERATOR_flt").alias("_a_op"),
            pl.col("BLOCK_TIME_UTC_mvt").alias("_a_block"),
            secs("BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("_a_bs"),
        )
        .collect()
        .with_columns(
            (pl.col("_a_bs").abs() <= FILL_TOL).cast(pl.Int8).alias("_a_fill"),
            pl.col("_a_block").dt.date().alias("_date"),
        )
    )


def build_arrival_fill(dep: pl.DataFrame, arr: pl.DataFrame) -> pl.DataFrame:
    """Arrival-side schedule-fill state for each departure. Reads only ARRIVAL block/sched times.

    arr_link_fill     the same-stand arrival that stand_arrival_gap links to (latest in-block before this
                      departure's take-off, within 6 h) had its in-block written as its schedule (|bs| <= 12 s)
    arr_link_bs_abs   |in-block - sched| of that arrival, seconds (continuous version of the flag)
    arr_link_same_op  that arrival is the same operator (the rotation is plausibly the same aircraft)
    arr_opday_fill    fill rate over this operator's arrivals at this airport on this UTC date, shrunk
                      toward the airport-day rate with k = 5
    arr_day_fill      fill rate over all arrivals at this airport on this UTC date
    """
    a = arr.drop_nulls(["ADEP_mvt", "_a_block"])
    day = a.group_by("ADEP_mvt", "_date").agg(pl.col("_a_fill").mean().alias("arr_day_fill"))
    opd = (
        a.group_by("ADEP_mvt", "_a_op", "_date")
        .agg(pl.col("_a_fill").sum().alias("_k"), pl.len().alias("_n"))
        .join(day, on=["ADEP_mvt", "_date"], how="left")
        .with_columns(((pl.col("_k") + 5 * pl.col("arr_day_fill")) / (pl.col("_n") + 5)).alias("arr_opday_fill"))
        .select("ADEP_mvt", pl.col("_a_op").alias("AIRCRAFT_OPERATOR_flt"), "_date", "arr_opday_fill")
    )
    link_src = a.drop_nulls("STAND_mvt").sort("_a_block").select("ADEP_mvt", "STAND_mvt", "_a_block", "_a_bs",
                                                                  "_a_fill", "_a_op")
    base = dep.select("MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "AIRCRAFT_OPERATOR_flt", "MVT_TIME_UTC_mvt").with_columns(
        pl.col("MVT_TIME_UTC_mvt").dt.date().alias("_date"))
    linked = (
        base.drop_nulls("MVT_TIME_UTC_mvt").sort("MVT_TIME_UTC_mvt")
        .join_asof(link_src, left_on="MVT_TIME_UTC_mvt", right_on="_a_block", by=["ADEP_mvt", "STAND_mvt"],
                   strategy="backward", allow_exact_matches=False, check_sortedness=False)
        .with_columns(((pl.col("MVT_TIME_UTC_mvt") - pl.col("_a_block")).dt.total_seconds() <= 21600).alias("_ok"))
        .select(
            "MVT_ID_mvt",
            pl.when("_ok").then(pl.col("_a_fill")).alias("arr_link_fill"),
            pl.when("_ok").then(pl.col("_a_bs").abs().cast(pl.Float64)).alias("arr_link_bs_abs"),
            pl.when("_ok").then((pl.col("_a_op") == pl.col("AIRCRAFT_OPERATOR_flt")).cast(pl.Int8))
            .alias("arr_link_same_op"),
        )
    )
    out = (
        base.join(linked, on="MVT_ID_mvt", how="left", maintain_order="left")
        .join(day, on=["ADEP_mvt", "_date"], how="left", maintain_order="left")
        .join(opd, on=["ADEP_mvt", "AIRCRAFT_OPERATOR_flt", "_date"], how="left", maintain_order="left")
        .with_columns(pl.col("arr_opday_fill").fill_null(pl.col("arr_day_fill")))
        .select("MVT_ID_mvt", "arr_link_fill", "arr_link_bs_abs", "arr_link_same_op", "arr_opday_fill",
                "arr_day_fill")
    )
    assert out.height == dep.height
    return out


ARR_COLS = ["arr_link_fill", "arr_link_bs_abs", "arr_link_same_op", "arr_opday_fill"]


def with_fill_features(A: pl.DataFrame) -> pl.DataFrame:
    dep = pl.read_parquet(FEAT, columns=["MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "AIRCRAFT_OPERATOR_flt",
                                         "MVT_TIME_UTC_mvt"])
    f = build_arrival_fill(dep, arrivals("train"))
    n0 = A.height
    A = A.join(f, on="MVT_ID_mvt", how="left", maintain_order="left")
    assert A.height == n0
    return A


def add_diag_labels(A: pl.DataFrame) -> pl.DataFrame:
    """Diagnostic labels only. `fill` and `bl` read the departure BLOCK and are never features."""
    return A.with_columns(
        (secs("BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").abs() <= FILL_TOL).alias("fill"),
        (secs("SCHED_TIME_UTC_mvt", "LOBT_flt").abs() > FILL_TOL).alias("sched_ne_lobt"),
        secs("BLOCK_TIME_UTC_mvt", "LOBT_flt").alias("bl"),
        secs("AOBT_3_flt", "LOBT_flt").alias("al"),
        secs("LOBT_flt", "SCHED_TIME_UTC_mvt").alias("lobt_minus_sched"),
        pl.col("MVT_TIME_UTC_mvt").dt.date().alias("date"),
    )


# ============================================================================================ sections
def A() -> None:
    d = add_diag_labels(load_lane_a())
    tot = float((d["r"] ** 2).sum())
    print("A lane-A rows %d, RMSE %.4f" % (d.height, np.sqrt(tot / d.height)))
    sec = d.filter(pl.col("ADEP_mvt") == "LIRF")["MVT_TIME_UTC_mvt"].dt.second()
    print("A LIRF MVT_TIME seconds-in-minute: share within +-6 s of :00 = %.4f" % ((sec <= 6) | (sec >= 54)).mean())
    # 1. the schedule fill by airport and by whether the plan was moved off the schedule
    t = (d.group_by("ADEP_mvt", "sched_ne_lobt")
         .agg(pl.len().alias("n"), pl.col("fill").sum().alias("n_fill"),
              ((secs("BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt") / 60).round().abs() == 1).sum().truediv(2)
              .alias("nbr_minute_avg"),
              (pl.col("r") ** 2).filter(pl.col("fill")).sum().truediv(tot).alias("fill_sse_share_laneA"),
              pl.col("r").filter(pl.col("fill")).mean().alias("fill_bias"),
              (pl.col("r") ** 2).filter(pl.col("fill")).mean().sqrt().alias("fill_rmse"))
         .sort("ADEP_mvt", "sched_ne_lobt"))
    print("A schedule fill (|BLOCK-SCHED|<=12 s) on lane A, split by SCHED != LOBT;"
          " nbr_minute_avg = mean count in the +-1 minute bins (the no-spike expectation)")
    print(t)
    # 2. LIRF and LFPG by BLOCK-LOBT band (label-side diagnostic)
    for apt in ("LIRF", "LFPG"):
        x = d.filter(pl.col("ADEP_mvt") == apt)
        at = float((x["r"] ** 2).sum())
        print("A %s lane-A n %d RMSE %.1f share of lane-A SSE %.4f" % (apt, x.height, np.sqrt(at / x.height), at / tot))
        k = x.with_columns(
            pl.when(pl.col("fill") & pl.col("sched_ne_lobt")).then(pl.lit("1 sched-fill, SCHED!=LOBT"))
            .when(pl.col("fill")).then(pl.lit("2 sched-fill, SCHED==LOBT"))
            .when(pl.col("bl") < -1200).then(pl.lit("3 BLOCK >20 min before LOBT, not fill"))
            .otherwise(pl.lit("4 rest")).alias("kind"))
        print(k.group_by("kind").agg(pl.len().alias("n"), (pl.col("r") ** 2).mean().sqrt().alias("rmse"),
                                     pl.col("r").mean().alias("bias"),
                                     ((pl.col("r") ** 2).sum() / at).alias("share_apt"),
                                     ((pl.col("r") ** 2).sum() / tot).alias("share_laneA")).sort("kind"))
        days = (x.group_by("date").agg(pl.len().alias("n"), (pl.col("r") ** 2).sum().alias("sse"),
                                       pl.col("r").mean().alias("bias"),
                                       pl.col("TAXITIME_SEC_mvt").median().alias("y_med"),
                                       pl.col("mvt_minus_aobt3").median().alias("proxy_med"),
                                       pl.col("mvt_minus_sched").median().alias("ms_med"),
                                       pl.col("fill").mean().alias("fill_rate"),
                                       (pl.col("fill") & pl.col("sched_ne_lobt")).sum().alias("n_fill_moved"),
                                       (pl.col("r") ** 2).filter(pl.col("fill")).sum().alias("sse_fill"),
                                       (pl.col("bl") < -1200).mean().alias("early_rate"))
                .with_columns((pl.col("sse") / at).alias("share"), (pl.col("sse_fill") / pl.col("sse")).alias(
                    "fill_share_of_day"))
                .sort("sse", descending=True))
        print("A %s ten worst days: %.3f of the airport's lane-A SSE; fill rows carry %.3f of those days' SSE"
              % (apt, days.head(10)["share"].sum(), days.head(10)["sse_fill"].sum() / days.head(10)["sse"].sum()))
        print(days.head(10).drop("sse", "sse_fill"))
        print("A %s median over all days: y_med %.0f proxy_med %.0f ms_med %.0f fill_rate %.3f" % (
            apt, days["y_med"].median(), days["proxy_med"].median(), days["ms_med"].median(),
            days["fill_rate"].median()))


def B() -> None:
    for frame in ("train", "rank"):
        a = arrivals(frame).drop_nulls("_a_bs").with_columns(pl.col("_a_block").dt.month().alias("m"))
        t = (a.group_by("ADEP_mvt").agg(pl.len().alias("n"), pl.col("_a_fill").mean().alias("fill"),
                                        ((pl.col("_a_bs") / 60).round().abs() == 1).mean().truediv(2)
                                        .alias("nbr_minute"))
             .with_columns((pl.col("fill") / pl.col("nbr_minute")).alias("spike")).sort("ADEP_mvt"))
        print("B %s arrivals: in-block == sched rate and spike over the +-1 minute bins" % frame)
        print(t)
        if frame == "train":
            print(a.filter(pl.col("ADEP_mvt") == "LIRF").group_by("m").agg(pl.col("_a_fill").mean()).sort("m"))
        else:
            print(a.filter(pl.col("ADEP_mvt") == "LIRF").group_by("m").agg(pl.len(), pl.col("_a_fill").mean()))
    # the departure-side relation (training only; the departure label is a diagnostic)
    d = add_diag_labels(with_fill_features(load_lane_a()))
    x = d.filter(pl.col("ADEP_mvt") == "LIRF")
    print("B LIRF lane A: departure schedule-fill rate by linked-arrival fill flag")
    print(x.group_by("arr_link_fill", "arr_link_same_op").agg(pl.len(), pl.col("fill").mean().alias("dep_fill"),
                                                              (pl.col("r") ** 2).mean().sqrt().alias("rmse"))
          .sort("arr_link_fill", "arr_link_same_op"))
    y = x.filter(pl.col("sched_ne_lobt"))
    print("B LIRF lane A, SCHED != LOBT only (the costly fill): dep fill rate by linked-arrival fill")
    print(y.group_by("arr_link_fill").agg(pl.len(), pl.col("fill").mean().alias("dep_fill"),
                                          (pl.col("r") ** 2).mean().sqrt().alias("rmse"),
                                          pl.col("r").mean().alias("bias")).sort("arr_link_fill"))
    print(y.with_columns(pl.col("arr_opday_fill").cut([0.05, 0.15, 0.3, 0.5]).alias("b"))
          .group_by("b").agg(pl.len(), pl.col("fill").mean().alias("dep_fill")).sort("b"))
    # stability across months: lift of the link flag per month
    print("B per-month dep-fill rate by link flag (LIRF lane A, all rows)")
    print(x.group_by("month", "arr_link_fill").agg(pl.len(), pl.col("fill").mean()).sort("month", "arr_link_fill"))


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def C() -> None:
    """Month-blocked fill classifier on ALL LIRF matched training rows (not only OOF rows).

    Target: departure schedule fill (diagnostic). Base = what the production model can already see about the
    row; + arrival = base + ARR_COLS; + permuted = base + ARR_COLS shuffled within month (control).
    Folds: leave one calendar month out, twelve folds.
    """
    import lightgbm as lgb

    cols = ["MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "AIRCRAFT_OPERATOR_flt", "AIRCRAFT_TYPE_mvt", "MARKET_SEGMENT_flt",
            "RUNWAY_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt",
            "AOBT_3_flt", "month", "hour_local", "dow", "mvt_minus_sched", "mvt_minus_lobt", "mvt_minus_aobt3",
            "mvt_minus_eobt1", "lobt_minus_eobt1", "aobt3_minus_sched", "stand_arrival_gap", "stand_aobt_turn",
            "CALLSIGN_flt"]
    d = pl.read_parquet(FEAT, columns=cols).filter((pl.col("ADEP_mvt") == "LIRF") & pl.col("mvt_minus_aobt3")
                                                   .is_not_null())
    f = build_arrival_fill(d, arrivals("train"))
    d = add_diag_labels(d.join(f, on="MVT_ID_mvt", how="left", maintain_order="left"))
    d = d.with_columns(
        (pl.col("mvt_minus_sched") - pl.col("mvt_minus_lobt")).alias("lobt_minus_sched_v"),  # visible form
        pl.col("CALLSIGN_flt").str.slice(0, 3).alias("pfx"))
    cats = ["STAND_mvt", "AIRCRAFT_OPERATOR_flt", "AIRCRAFT_TYPE_mvt", "MARKET_SEGMENT_flt", "RUNWAY_mvt", "ADES_mvt",
            "pfx"]
    d = d.with_columns([pl.col(c).fill_null("NA").cast(pl.Categorical).to_physical().cast(pl.Int32).alias(c + "_c")
                        for c in cats])
    base = [c + "_c" for c in cats] + ["hour_local", "dow", "mvt_minus_sched", "mvt_minus_lobt", "mvt_minus_aobt3",
                                       "mvt_minus_eobt1", "lobt_minus_eobt1", "aobt3_minus_sched",
                                       "stand_arrival_gap", "stand_aobt_turn", "lobt_minus_sched_v"]
    rng = np.random.default_rng(11)
    perm = {}
    mo = d["month"].to_numpy()
    for c in ARR_COLS:
        v = d[c].to_numpy().astype(np.float64).copy()
        for m in np.unique(mo):
            ix = np.where(mo == m)[0]
            v[ix] = v[rng.permutation(ix)]
        perm[c + "_perm"] = v
    d = d.with_columns([pl.Series(k, v) for k, v in perm.items()])
    y = d["fill"].to_numpy().astype(int)
    moved = d["sched_ne_lobt"].to_numpy()
    print("C LIRF matched training rows %d, fill rate %.4f; SCHED!=LOBT rows %d, fill rate %.4f"
          % (d.height, y.mean(), moved.sum(), y[moved].mean()))
    arms = {"base": base, "base+arrival": base + ARR_COLS, "base+permuted": base + [c + "_perm" for c in ARR_COLS],
            "arrival only": ARR_COLS + ["lobt_minus_sched_v"]}
    res = {}
    for name, feats in arms.items():
        X = d.select(feats).with_columns([pl.col(c).cast(pl.Float64) for c in feats]).to_numpy()
        p = np.full(len(y), np.nan)
        catix = [i for i, c in enumerate(feats) if c.endswith("_c")]
        for m in np.unique(mo):
            te = mo == m
            ds = lgb.Dataset(X[~te], label=y[~te], categorical_feature=catix, free_raw_data=True)
            params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50,
                          feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=0,
                          num_threads=2, deterministic=True, force_row_wise=True, cat_smooth=20, min_data_per_group=50)
            p[te] = lgb.train(params, ds, num_boost_round=300).predict(X[te])
        res[name] = p
        ll = -np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))
        per_m = [_auc(y[mo == m], p[mo == m]) for m in np.unique(mo)]
        print("C %-15s AUC all %.4f  AUC SCHED!=LOBT %.4f  logloss %.4f  per-month AUC min %.4f max %.4f"
              % (name, _auc(y, p), _auc(y[moved], p[moved]), ll, min(per_m), max(per_m)), flush=True)
        print("C %-15s per-month AUC %s" % (name, " ".join("%d:%.3f" % (m, a) for m, a in zip(np.unique(mo), per_m))))
    np.save(SCRATCH + "/lirf_fill_p.npy", np.vstack([d["MVT_ID_mvt"].to_numpy(), res["base"], res["base+arrival"],
                                                     res["base+permuted"]]))
    # 2x2 monthly wins: does the arrival arm beat base in every month?
    wins = sum(_auc(y[mo == m], res["base+arrival"][mo == m]) > _auc(y[mo == m], res["base"][mo == m])
               for m in np.unique(mo))
    print("C base+arrival beats base in %d of %d held-out months" % (wins, len(np.unique(mo))))


def _instrument(A: pl.DataFrame, base: list[str], arms: dict[str, list[str]], seeds: int = 3, tag: str = "D") -> dict:
    sys.path.insert(0, W + "docs/competitor_research_2026-09-21/scripts")
    from delay_mechanisms import month_blocked_gain  # the calibrated recipe, unmodified

    out = {}
    for name, extra in arms.items():
        v = [month_blocked_gain(A, base + extra, seed=s, num_leaves=8)["new"] for s in range(seeds)]
        out[name] = (float(np.mean(v)), float(np.std(v)))
        print("%s %-28s %.4f (seed sd %.4f)  rss %.0f MB" % (tag, name, out[name][0], out[name][1], rss_mb()),
              flush=True)
    return out


def _prep_instrument() -> pl.DataFrame:
    A = load_lane_a(["MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt", "TAXITIME_SEC_mvt", "mvt_minus_aobt3", "month",
                     "hour_local", "mvt_minus_sched", "mvt_minus_lobt", "noise_control", "noise_control_2",
                     "noise_control_3", "noise_control_4"])
    A = with_fill_features(A)
    rng = np.random.default_rng(5)
    mo = A["month"].to_numpy()
    ad = A["ADEP_mvt"].to_numpy()
    perm = {}
    for c in ARR_COLS:
        v = A[c].to_numpy().astype(np.float64).copy()
        for m in np.unique(mo):
            for a in np.unique(ad):
                ix = np.where((mo == m) & (ad == a))[0]
                v[ix] = v[rng.permutation(ix)]
        perm[c + "_perm"] = v
    return A.with_columns(
        [pl.Series(k, v) for k, v in perm.items()]
        + [pl.col("ADEP_mvt").cast(pl.Categorical).to_physical().cast(pl.Float64).alias("adep_code"),
           pl.col("RUNWAY_mvt").fill_null("NA").cast(pl.Categorical).to_physical().cast(pl.Float64).alias("rwy_code"),
           (pl.col("mvt_minus_sched") - pl.col("mvt_minus_lobt")).alias("lobt_minus_sched_v"),
           (pl.col("mvt_minus_sched") - pl.col("oof")).alias("ms_minus_pred")])


def D() -> None:
    A = _prep_instrument()
    BASE = ["adep_code", "hour_local", "mvt_minus_aobt3", "rwy_code"]
    NC = ["noise_control", "noise_control_2", "noise_control_3", "noise_control_4"]
    r = _instrument(A, BASE, {"base": [], "control4": NC, "arrival fill (4)": ARR_COLS,
                              "arrival fill permuted (4)": [c + "_perm" for c in ARR_COLS]})
    c = r["control4"][0]
    for k in ("arrival fill (4)", "arrival fill permuted (4)"):
        print("D %-28s vs control4 %+.4f  implied full-model lane-A gain (/0.57) %+.3f" % (k, r[k][0] - c,
                                                                                      (r[k][0] - c) / 0.57))


def E() -> None:
    """The fill correction is multiplicative in (ms - pred): a tree on r cannot size it without ms and the
    prediction. So every arm here, the control included, gets both in its base. Not calibrated against
    stand_arrival_gap; read only as arm minus its own permuted control."""
    A = _prep_instrument()
    BASE = ["adep_code", "hour_local", "mvt_minus_aobt3", "rwy_code", "ms_minus_pred", "lobt_minus_sched_v"]
    NC = ["noise_control", "noise_control_2", "noise_control_3", "noise_control_4"]
    r = _instrument(A, BASE, {"base+ms": [], "control4": NC, "arrival fill (4)": ARR_COLS,
                              "arrival fill permuted (4)": [c + "_perm" for c in ARR_COLS]}, tag="E")
    for k in ("arrival fill (4)",):
        print("E %-28s vs control4 %+.4f   vs its permuted twin %+.4f"
              % (k, r[k][0] - r["control4"][0], r[k][0] - r["arrival fill permuted (4)"][0]))
    # where does it come from: LIRF only vs rest
    for apt in ("LIRF",):
        sub = A.filter(pl.col("ADEP_mvt") == apt)
        rr = _instrument(sub, BASE, {"%s base+ms" % apt: [], "%s control4" % apt: NC,
                                     "%s arrival fill" % apt: ARR_COLS,
                                     "%s arrival permuted" % apt: [c + "_perm" for c in ARR_COLS]}, tag="E")
        print("E %s arrival vs permuted twin %+.4f on %d rows" % (apt, rr["%s arrival fill" % apt][0] -
                                                                   rr["%s arrival permuted" % apt][0], sub.height))


def F() -> None:
    d = add_diag_labels(load_lane_a())
    x = d.filter(pl.col("ADEP_mvt") == "LFPG")
    at = float((x["r"] ** 2).sum())
    w = x.sort("r").head(300)
    print("F LFPG 300 most negative residuals: share of LFPG SSE %.3f; BLOCK-LOBT median %.0f; AOBT3-LOBT median %.0f;"
          " stand_arrival_gap null %.3f median %.0f; fill %.3f" % (
              float((w["r"] ** 2).sum()) / at, w["bl"].median(), w["al"].median(),
              w["stand_arrival_gap"].is_null().mean(), w["stand_arrival_gap"].median(), w["fill"].mean()))
    print("F   vs all LFPG: BLOCK-LOBT median %.0f; AOBT3-LOBT median %.0f; gap null %.3f median %.0f" % (
        x["bl"].median(), x["al"].median(), x["stand_arrival_gap"].is_null().mean(), x["stand_arrival_gap"].median()))
    print(w.group_by("month").agg(pl.len()).sort("month"))
    print(w.with_columns(pl.col("CALLSIGN_flt").str.slice(0, 3).alias("pfx")).group_by("pfx").agg(pl.len())
          .sort("len", descending=True).head(8))
    print(w.with_columns(pl.col("STAND_mvt").str.slice(0, 1).alias("t")).group_by("t").agg(pl.len())
          .sort("len", descending=True).head(8))
    print(x.with_columns(pl.col("STAND_mvt").str.slice(0, 1).alias("t")).group_by("t").agg(pl.len())
          .sort("len", descending=True).head(8))


if __name__ == "__main__":
    todo = [a for a in sys.argv[1:]] or ["A", "B", "C", "D", "E", "F"]
    for s in todo:
        print("\n" + "=" * 30 + " " + s + " " + "=" * 30, flush=True)
        globals()[s]()
