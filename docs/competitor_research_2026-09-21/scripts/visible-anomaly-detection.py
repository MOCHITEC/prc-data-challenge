"""Can anomalous rows be told apart from visible information alone?

Read-only. No training run, no submission, no writes outside OUT.
Stages (pass as argv[1]): counts | marginal | cats | combo | widen | part2 | price | all

Definitions used throughout
  y      = TAXITIME_SEC_mvt (label; blanked on 2026 departures)
  ms     = mvt_minus_sched = MVT_TIME_UTC_mvt - SCHED_TIME_UTC_mvt   (VISIBLE on 2026 departures)
  lane   = nm_matched (1 = model A, has NM clocks; 0 = model B, NM clocks null)
  positive (corruption) = y > 72,000 s, the BLOCK-day-rollover signature
                          (docs/investigation_2026-09-16/results/block-day-rollover.md)
VISIBLE = computable on a 2026 scored departure row, which carries every feat_rank column except
BLOCK_TIME_UTC_mvt and TAXITIME_SEC_mvt (src/prc/schema.py BLANKED_DEP_COLS).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path("/home/user/prc-data-challenge")
OUT = Path("/tmp/claude-0/-home-user-prc-data-challenge/ca703460-62c0-5c84-91fe-d76c3d0e1e20/scratchpad/vad")
OUT.mkdir(parents=True, exist_ok=True)
TRAIN = ROOT / "data/work/feat_train.parquet"
RANK = ROOT / "data/work/feat_rank.parquet"
OOF = ROOT / "experiments/oof_main_reparam-bc60017878.parquet"

LABEL = "TAXITIME_SEC_mvt"
# Never a feature: the label, the blanked block clock, ids, the phase constant.
LABEL_DERIVED = {LABEL, "BLOCK_TIME_UTC_mvt", "MVT_ID_mvt", "FLIGHT_ID_mvt", "PHASE_mvt"}
CATS = [
    "ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt",
    "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt", "ADES_mvt", "ADES_REGION",
    "FLIGHT_RULE_mvt", "FLIGHT_RULE_flt",
]


def log(*a):
    print(*a, flush=True)


def numeric_visible(schema) -> list[str]:
    ok = (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt32, pl.UInt8, pl.Boolean)
    return [c for c, d in schema.items() if c not in LABEL_DERIVED and d in ok]


def load_train(cols=None) -> pl.DataFrame:
    lf = pl.scan_parquet(TRAIN)
    return lf.select(cols).collect() if cols else lf.collect()


def auc(score: np.ndarray, pos: np.ndarray) -> float:
    """Mann-Whitney AUC with ties at 0.5; nulls in score are ranked as a tied worst group."""
    s = score.astype(float).copy()
    finite = np.isfinite(s)
    if finite.sum() == 0:
        return float("nan")
    lo = np.nanmin(s[finite]) - 1.0
    s[~finite] = lo
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    sr = s[order]
    i = 0
    rank = np.arange(1, len(s) + 1, dtype=float)
    while i < len(s):
        j = i
        while j + 1 < len(s) and sr[j + 1] == sr[i]:
            j += 1
        r[order[i : j + 1]] = rank[i : j + 1].mean()
        i = j + 1
    npos, nneg = int(pos.sum()), int((~pos).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    return (r[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def pct_of(values: np.ndarray, x: float) -> float:
    """Percentile position of x inside `values` (mid-rank, nulls dropped), in %."""
    v = values[np.isfinite(values)]
    if len(v) == 0 or not np.isfinite(x):
        return float("nan")
    return 100.0 * ((v < x).sum() + 0.5 * (v == x).sum()) / len(v)


# ------------------------------------------------------------------ stage: counts
def stage_counts():
    t = load_train(["MVT_ID_mvt", "month", LABEL, "nm_matched", "ADEP_mvt", "mvt_minus_sched"])
    r = pl.scan_parquet(RANK).select(["MVT_ID_mvt", "month", "nm_matched", "ADEP_mvt", "mvt_minus_sched"]).collect()
    res = {
        "train_rows": t.height,
        "rank_rows": r.height,
        "train_unmatched": int((t["nm_matched"] == 0).sum()),
        "rank_unmatched": int((r["nm_matched"] == 0).sum()),
        "rank_months": t.group_by("month").len().sort("month").to_dicts(),
    }
    for thr, name in [(72000, "gt72000"), (43200, "ge43200"), (14400, "ge14400"), (7200, "ge7200")]:
        m = t[LABEL] > thr if thr == 72000 else t[LABEL] >= thr
        res[name] = {"n": int(m.sum()), "unmatched": int((m & (t["nm_matched"] == 0)).sum()),
                     "lirf": int((m & (t["ADEP_mvt"] == "LIRF")).sum())}
    log(json.dumps(res, indent=1, default=str))
    (OUT / "counts.json").write_text(json.dumps(res, indent=1, default=str))


# ------------------------------------------------------------------ stage: marginal
def stage_marginal():
    t = load_train()
    num = numeric_visible(t.schema)
    pos_mask = (t[LABEL] > 72000).to_numpy()
    lane = (t["nm_matched"] == 0).to_numpy()
    adep = t["ADEP_mvt"].to_numpy()
    ms = t["mvt_minus_sched"].to_numpy().astype(float)

    pops = {
        "R1_unmatched_all": lane,
        "R2_LIRF_unmatched": lane & (adep == "LIRF"),
        "R3_LIRF_unm_ms_12to28h": lane & (adep == "LIRF") & np.isfinite(ms) & (ms >= 43200) & (ms <= 100800),
        "R4_all_rows": np.ones(len(t), bool),
    }
    rows = []
    for pname, pmask in pops.items():
        p = pos_mask & pmask
        if p.sum() == 0:
            continue
        for c in num:
            v = t[c].to_numpy().astype(float)
            sub = v[pmask]
            subp = pos_mask[pmask]
            a = auc(sub, subp)
            nullrate_pop = float(np.mean(~np.isfinite(sub)))
            nullrate_pos = float(np.mean(~np.isfinite(v[p])))
            base = sub[~subp]
            pcts = np.array([pct_of(base, x) for x in v[p]])
            rows.append(dict(pop=pname, n_pop=int(pmask.sum()), n_pos=int(p.sum()), feature=c, auc=a,
                             pct_min=float(np.nanmin(pcts)) if np.isfinite(pcts).any() else float("nan"),
                             pct_med=float(np.nanmedian(pcts)) if np.isfinite(pcts).any() else float("nan"),
                             pct_max=float(np.nanmax(pcts)) if np.isfinite(pcts).any() else float("nan"),
                             n_pos_notnull=int(np.isfinite(v[p]).sum()),
                             null_rate_pop=nullrate_pop, null_rate_pos=nullrate_pos))
    df = pl.DataFrame(rows)
    df.write_parquet(OUT / "marginal.parquet")
    for pname in pops:
        d = df.filter(pl.col("pop") == pname)
        if d.height == 0:
            continue
        d = d.with_columns((pl.col("auc") - 0.5).abs().alias("sep")).sort("sep", descending=True)
        log(f"\n=== {pname}  n_pop={d['n_pop'][0]} n_pos={d['n_pos'][0]} — top 25 by |AUC-0.5| ===")
        log(d.select(["feature", "auc", "pct_min", "pct_med", "pct_max", "n_pos_notnull",
                      "null_rate_pop", "null_rate_pos"]).head(25).to_pandas().to_string(index=False))

    # per-row detail on the clock features inside R1
    detail_cols = ["mvt_minus_sched", "mvt_minus_sched_slog", "hour_utc", "stand_arrival_gap",
                   "stand_aobt_turn", "dep_tko_prev_60m", "arr_inblock_prev_60m", "queue_ahead",
                   "rwy_headway_prev_s", "dest_distance_km", "planned_flight_s", "window_coverage_60m"]
    base_mask = pops["R1_unmatched_all"] & ~pos_mask
    recs = []
    idx = np.flatnonzero(pos_mask & pops["R1_unmatched_all"])
    for i in idx:
        rec = {"ADEP": adep[i], "FLIGHT": t["FLIGHT_mvt"][int(i)], "y": int(t[LABEL][int(i)])}
        for c in detail_cols:
            v = t[c].to_numpy().astype(float)
            rec[c] = float(v[i]) if np.isfinite(v[i]) else None
            rec[c + "_pct"] = round(pct_of(v[base_mask], v[i]), 2)
        recs.append(rec)
    dd = pl.DataFrame(recs, strict=False).sort("y", descending=True)
    dd.write_parquet(OUT / "marginal_detail.parquet")
    log("\n=== per-row percentiles of the 15 unmatched positives inside R1 (unmatched training rows) ===")
    log(dd.to_pandas().to_string(index=False))


# ------------------------------------------------------------------ stage: cats
def stage_cats():
    t = load_train()
    pos = (t[LABEL] > 72000)
    lane = t["nm_matched"] == 0
    out = []
    extra = {"flight_prefix3": t["FLIGHT_mvt"].str.slice(0, 3)}
    frame = t.with_columns([v.alias(k) for k, v in extra.items()])
    for c in CATS + ["flight_prefix3", "month", "dow", "hour_utc", "stand_null", "runway_null",
                     "aobt3_null", "lobt_null", "eobt1_null", "diverted", "actype_mismatch",
                     "nm_matched", "anchor_ok", "stand_arrival_type_match", "deicing_gate", "low_vis"]:
        vals = frame[c]
        for scope, m in [("all", pl.Series([True] * t.height)), ("unmatched", lane)]:
            sub = frame.filter(m)
            sp = sub[LABEL] > 72000
            if sp.sum() == 0:
                continue
            pv = sub.filter(sp)[c]
            tab = pv.value_counts(sort=True)
            base = sub[c].value_counts()
            bmap = {r[c]: r["count"] for r in base.to_dicts()}
            n = sub.height
            for row in tab.to_dicts():
                lvl = row[c]
                k = row["count"]
                b = bmap.get(lvl, 0)
                out.append(dict(col=c, scope=scope, level=str(lvl), pos_n=k, pos_frac=k / int(sp.sum()),
                                pop_n=b, pop_frac=b / n, lift=(k / int(sp.sum())) / (b / n) if b else None,
                                precision_if_flag_all=k / b if b else None))
    d = pl.DataFrame(out)
    d.write_parquet(OUT / "cats.parquet")
    log("=== categorical / flag levels carrying positives, unmatched lane, sorted by precision ===")
    log(d.filter((pl.col("scope") == "unmatched") & (pl.col("pos_n") >= 2)).sort("precision_if_flag_all", descending=True)
        .head(40).to_pandas().to_string(index=False))
    log("\n=== same, scope=all ===")
    log(d.filter((pl.col("scope") == "all") & (pl.col("pos_n") >= 2)).sort("precision_if_flag_all", descending=True)
        .head(25).to_pandas().to_string(index=False))


# ------------------------------------------------------------------ classifier helpers
def build_xy(t: pl.DataFrame, feats: list[str]):
    import pandas as pd
    X = t.select(feats).to_pandas()
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype) == "string":
            X[c] = X[c].astype("category")
        elif X[c].dtype == bool:
            X[c] = X[c].astype("float32")
    return X


def lomo_scores(t: pl.DataFrame, feats: list[str], y: np.ndarray, seed=0, n_leaves=15, rounds=120, shuffle=False):
    """Leave-one-month-out LightGBM binary scores. Returns score vector aligned to t."""
    import lightgbm as lgb
    X = build_xy(t, feats)
    months = t["month"].to_numpy()
    yy = y.copy()
    if shuffle:
        rng = np.random.default_rng(seed)
        for m in np.unique(months):
            i = np.flatnonzero(months == m)
            yy[i] = rng.permutation(yy[i])
    sc = np.full(len(yy), np.nan)
    for m in np.unique(months):
        tr = months != m
        te = ~tr
        if yy[tr].sum() == 0:
            sc[te] = 0.0
            continue
        ds = lgb.Dataset(X[tr], label=yy[tr].astype(float), free_raw_data=False)
        p = dict(objective="binary", learning_rate=0.1, num_leaves=n_leaves, min_data_in_leaf=5,
                 feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                 verbose=-1, num_threads=2, deterministic=True, force_row_wise=True, seed=seed,
                 is_unbalance=True)
        b = lgb.train(p, ds, num_boost_round=rounds)
        sc[te] = b.predict(X[te])
    return sc, yy


def pr_at(score, y, ks=(1, 2, 5, 10, 20, 50, 100, 200, 500)):
    o = np.argsort(-score)
    out = []
    tot = int(y.sum())
    for k in ks:
        sel = o[:k]
        tp = int(y[sel].sum())
        out.append(dict(k=k, tp=tp, precision=tp / k, recall=tp / tot if tot else float("nan")))
    return out


# ------------------------------------------------------------------ stage: combo
def stage_combo():
    """Leave-one-month-out classifier on the unmatched lane, with a permutation control.

    Only the unmatched lane is fitted. 15 of the 16 positives are in it; on the full frame the
    task degenerates to "is this row unmatched", which is already a model feature (nm_matched)
    and the routing key, so an all-rows AUC would measure the lane, not the corruption.
    Rounds and leaves are kept small because other CPU-bound jobs share the four cores.
    """
    t = load_train()
    feats = [c for c in numeric_visible(t.schema) if c != "month"] + CATS
    feats = [c for c in feats if c in t.columns]
    sub = t.filter(t["nm_matched"] == 0)
    y = (sub[LABEL] > 72000).to_numpy().astype(int)
    sc, _ = lomo_scores(sub, feats, y, n_leaves=15, rounds=80)
    a_real = auc(sc, y.astype(bool))
    res = {"lane": "unmatched", "n": int(sub.height), "n_pos": int(y.sum()),
           "n_features": len(feats), "lomo_auc": a_real, "pr": pr_at(sc, y)}
    perm = []
    for s_ in range(30):
        sp, ys = lomo_scores(sub, feats, y, seed=100 + s_, n_leaves=15, rounds=80, shuffle=True)
        perm.append(auc(sp, ys.astype(bool)))
        log(f"  perm {s_}: {perm[-1]:.4f}")
    res["perm_auc"] = {"n": len(perm), "mean": float(np.mean(perm)), "sd": float(np.std(perm)),
                       "q05": float(np.quantile(perm, 0.05)), "q50": float(np.quantile(perm, 0.5)),
                       "q95": float(np.quantile(perm, 0.95)), "min": float(np.min(perm)),
                       "max": float(np.max(perm))}
    res["perm_p_one_sided"] = float((np.sum(np.array(perm) >= a_real) + 1) / (len(perm) + 1))

    ms = sub["mvt_minus_sched"].to_numpy().astype(float)
    res["ms_alone"] = {"auc": auc(ms, y.astype(bool)), "pr": pr_at(np.nan_to_num(ms, nan=-1e12), y)}
    rng = np.random.default_rng(0)
    idx = np.arange(len(y))
    null = []
    for _ in range(5000):
        pick = np.zeros(len(y), bool)
        pick[rng.choice(idx, 15, replace=False)] = True
        null.append(auc(ms, pick))
    null = np.array(null)
    res["ms_random_label_null"] = {"n": int(len(null)), "mean": float(null.mean()),
                                   "q95": float(np.quantile(null, 0.95)),
                                   "q99": float(np.quantile(null, 0.99)),
                                   "max": float(null.max()),
                                   "p_one_sided": float((np.sum(null >= res["ms_alone"]["auc"]) + 1) / (len(null) + 1))}
    tab = []
    yb = y.astype(bool)
    for thr in [7200, 14400, 21600, 28800, 36000, 43200, 50400, 57600, 64800, 72000, 86400]:
        f = np.isfinite(ms) & (ms >= thr)
        tab.append(dict(thr=thr, flagged=int(f.sum()), tp=int((f & yb).sum()),
                        precision=float((f & yb).sum() / max(f.sum(), 1)),
                        recall=float((f & yb).sum() / yb.sum())))
    res["ms_threshold_table"] = tab
    log(json.dumps(res, indent=1, default=float))
    log(pl.DataFrame(tab).to_pandas().to_string(index=False))
    (OUT / "combo.json").write_text(json.dumps(res, indent=1, default=float))
    np.save(OUT / "combo_score_unmatched.npy", sc)


# ------------------------------------------------------------------ stage: widen
def stage_widen():
    t = load_train()
    feats = [c for c in numeric_visible(t.schema) if c != "month"] + CATS
    sub = t.filter(t["nm_matched"] == 0)
    ms = sub["mvt_minus_sched"].to_numpy().astype(float)
    res = {}
    for name, thr in [("gt72000", 72000), ("ge43200", 43200), ("ge14400", 14400), ("ge7200", 7200)]:
        y = ((sub[LABEL] > thr) if thr == 72000 else (sub[LABEL] >= thr)).to_numpy().astype(int)
        sc, _ = lomo_scores(sub, feats, y, n_leaves=15, rounds=80)
        a = auc(sc, y.astype(bool))
        perm = [auc(*(lambda r: (r[0], r[1].astype(bool)))(lomo_scores(sub, feats, y, seed=200 + s, n_leaves=15, rounds=80, shuffle=True)))
                for s in range(8)]
        res[name] = {"n_pos_unmatched": int(y.sum()), "lomo_auc": a,
                     "ms_alone_auc": auc(ms, y.astype(bool)),
                     "perm_auc_mean": float(np.mean(perm)), "perm_auc_max": float(np.max(perm)),
                     "pr": pr_at(sc, y)}
        log(f"\n=== widen {name} ===")
        log(json.dumps(res[name], indent=1))
    # The all-rows variants are deliberately not fitted: see stage_combo's docstring.
    (OUT / "widen.json").write_text(json.dumps(res, indent=1, default=float))


# ------------------------------------------------------------------ stage: part2
def stage_part2():
    oof = pl.read_parquet(OOF).drop_nulls()
    t = load_train()
    j = t.join(oof, on="MVT_ID_mvt", how="inner")
    resid = (j["oof"] - j[LABEL]).to_numpy().astype(float)
    se = resid ** 2
    log(f"OOF rows {j.height} RMSE {np.sqrt(se.mean()):.4f} SSE {se.sum():.4e}")

    feats = [c for c in numeric_visible(t.schema) if c != "month"] + CATS
    # rank correlation of each visible feature with |resid|
    import scipy.stats as st
    rows = []
    aq = np.abs(resid)
    for c in [c for c in feats if c not in CATS]:
        v = j[c].to_numpy().astype(float)
        m = np.isfinite(v)
        if m.sum() < 1000:
            continue
        rho = st.spearmanr(v[m], aq[m]).statistic
        rows.append(dict(feature=c, spearman_abs_resid=float(rho), n=int(m.sum()),
                         null_rate=float(1 - m.mean())))
    d = pl.DataFrame(rows).with_columns(pl.col("spearman_abs_resid").abs().alias("a")).sort("a", descending=True)
    d.write_parquet(OUT / "part2_spearman.parquet")
    log("\n=== |Spearman| of visible features with |OOF residual| (top 25) ===")
    log(d.head(25).to_pandas().to_string(index=False))

    # group concentration
    grp = j.with_columns(pl.Series("se", se), pl.Series("ar", np.abs(resid)))
    for by in [["nm_matched"], ["ADEP_mvt"], ["nm_matched", "ADEP_mvt"], ["stand_null"], ["runway_null"]]:
        g = (grp.group_by(by).agg(n=pl.len(), sse=pl.col("se").sum(), rmse=pl.col("se").mean().sqrt(),
                                  p999=pl.col("ar").quantile(0.999))
             .with_columns((pl.col("sse") / se.sum()).alias("sse_share")).sort("sse", descending=True))
        log(f"\n=== SSE concentration by {by} ===")
        log(g.head(22).to_pandas().to_string(index=False))

    # heteroscedasticity model: predict log1p(se) from visible features, LOMO across the 4 OOF months
    import lightgbm as lgb
    X = build_xy(j, feats)
    months = j["month"].to_numpy()
    tgt = np.log1p(se)
    pred = np.full(len(tgt), np.nan)
    for m in np.unique(months):
        tr, te = months != m, months == m
        ds = lgb.Dataset(X[tr], label=tgt[tr], free_raw_data=False)
        p = dict(objective="regression", learning_rate=0.08, num_leaves=63, min_data_in_leaf=100,
                 feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                 num_threads=3, deterministic=True, force_row_wise=True, seed=3)
        b = lgb.train(p, ds, num_boost_round=300)
        pred[te] = b.predict(X[te])
    rho = st.spearmanr(pred, se).statistic
    o = np.argsort(-pred)
    res = {"n": int(len(se)), "spearman_pred_vs_se": float(rho), "rmse": float(np.sqrt(se.mean()))}
    for frac in (0.001, 0.005, 0.01, 0.05, 0.10, 0.25):
        k = int(len(se) * frac)
        res[f"sse_share_top_{frac}"] = float(se[o[:k]].sum() / se.sum())
        res[f"rmse_in_top_{frac}"] = float(np.sqrt(se[o[:k]].mean()))
        res[f"rmse_outside_top_{frac}"] = float(np.sqrt(se[o[k:]].mean()))
    # oracle for comparison
    oo = np.argsort(-se)
    for frac in (0.001, 0.01, 0.10):
        k = int(len(se) * frac)
        res[f"oracle_sse_share_top_{frac}"] = float(se[oo[:k]].sum() / se.sum())
    # importance from the last fold
    imp = sorted(zip(X.columns, b.feature_importance("gain")), key=lambda z: -z[1])[:25]
    res["top_gain_last_fold"] = [(a, float(v)) for a, v in imp]
    log("\n=== heteroscedasticity model (visible features -> log1p SE), LOMO over OOF months ===")
    log(json.dumps(res, indent=1))
    (OUT / "part2.json").write_text(json.dumps(res, indent=1, default=float))
    np.save(OUT / "part2_pred.npy", pred)
    np.save(OUT / "part2_se.npy", se)
    np.save(OUT / "part2_mvtid.npy", j["MVT_ID_mvt"].to_numpy())

    # does the association survive conditioning on the model's own prediction?
    base = np.column_stack([j["oof"].to_numpy(), (j["nm_matched"].to_numpy())])
    for m in np.unique(months):
        tr, te = months != m, months == m
        ds = lgb.Dataset(base[tr], label=tgt[tr])
        b2 = lgb.train(dict(objective="regression", learning_rate=0.08, num_leaves=31, min_data_in_leaf=200,
                            verbose=-1, num_threads=3, deterministic=True, force_row_wise=True, seed=3),
                       ds, num_boost_round=200)
        pred[te] = b2.predict(base[te])
    res2 = {"spearman_pred_vs_se_pred_only": float(st.spearmanr(pred, se).statistic)}
    o2 = np.argsort(-pred)
    for frac in (0.001, 0.01, 0.10):
        k = int(len(se) * frac)
        res2[f"sse_share_top_{frac}"] = float(se[o2[:k]].sum() / se.sum())
    log("\n=== control: prediction + lane only ===")
    log(json.dumps(res2, indent=1))
    (OUT / "part2_control.json").write_text(json.dumps(res2, indent=1, default=float))


# ------------------------------------------------------------------ stage: price
def stage_price():
    t = load_train(["month", LABEL, "nm_matched", "ADEP_mvt", "mvt_minus_sched"])
    r = pl.scan_parquet(RANK).select(["month", "nm_matched", "ADEP_mvt", "mvt_minus_sched"]).collect()
    import scipy.stats as st
    res = {}
    N = r.height
    rmse = 282.7371
    sse = N * rmse ** 2
    res["board"] = dict(N=N, rmse=rmse, sse=sse, sse_per_second_of_rmse=2 * N * rmse)
    pos = t[LABEL] > 72000
    # rate estimates
    def rate(mask_t, mask_r, name):
        nt, k = int(mask_t.sum()), int((pos & mask_t).sum())
        nr = int(mask_r.sum())
        lam = k / nt * nr
        lo, hi = (st.chi2.ppf(0.025, 2 * k) / 2 if k else 0.0), st.chi2.ppf(0.975, 2 * k + 2) / 2
        res[name] = dict(train_n=nt, train_pos=k, rank_n=nr, expected=lam,
                         poisson95=[lo / nt * nr, hi / nt * nr])
    rate(pl.Series([True] * t.height), pl.Series([True] * r.height), "naive_all_rows")
    rate(t["nm_matched"] == 0, r["nm_matched"] == 0, "unmatched_lane")
    rate(t["month"].is_in([1, 7]), pl.Series([True] * r.height), "months_1_and_7")
    rate((t["nm_matched"] == 0) & t["month"].is_in([1, 7]), r["nm_matched"] == 0, "unmatched_months_1_7")
    rate((t["nm_matched"] == 0) & (t["ADEP_mvt"] == "LIRF"),
         (r["nm_matched"] == 0) & (r["ADEP_mvt"] == "LIRF"), "LIRF_unmatched")
    # how many rank rows sit in the ms band the detector would flag
    ms_r = r["mvt_minus_sched"].to_numpy().astype(float)
    unm_r = (r["nm_matched"] == 0).to_numpy()
    ms_t = t["mvt_minus_sched"].to_numpy().astype(float)
    unm_t = (t["nm_matched"] == 0).to_numpy()
    posn = pos.to_numpy()
    band = []
    for thr in [14400, 28800, 43200, 50000, 57600, 72000, 86400]:
        ft = unm_t & np.isfinite(ms_t) & (ms_t >= thr)
        fr = unm_r & np.isfinite(ms_r) & (ms_r >= thr)
        band.append(dict(thr=thr, train_flagged=int(ft.sum()), train_tp=int((ft & posn).sum()),
                         train_precision=float((ft & posn).sum() / max(ft.sum(), 1)),
                         train_recall=float((ft & posn).sum() / posn.sum()),
                         rank_flagged=int(fr.sum()),
                         rank_flagged_per_train_month=float(ft.sum() / 12)))
    res["ms_band_rank_counts"] = band
    log(json.dumps(res, indent=1, default=float))
    log(pl.DataFrame(band).to_pandas().to_string(index=False))
    (OUT / "price.json").write_text(json.dumps(res, indent=1, default=float))


# ------------------------------------------------------------------ stage: hardcase
def stage_hardcase():
    """The two rows that dominate the OOF SSE are LFPG rows with a SMALL ms. Where do they sit?"""
    t = load_train()
    oofd = pl.read_parquet(OOF).drop_nulls()
    j = t.join(oofd, on="MVT_ID_mvt", how="inner").with_columns(
        ((pl.col("oof") - pl.col(LABEL)) ** 2).alias("se"))
    S = float(j["se"].sum())

    # 1. cost accounting of the ms signal the model ALREADY acts on (LIRF unmatched lane)
    lir = j.filter((pl.col("nm_matched") == 0) & (pl.col("ADEP_mvt") == "LIRF"))
    rows = []
    for lo, hi in [(-1e9, 3600), (3600, 7200), (7200, 14400), (14400, 43200), (43200, 1e9)]:
        b = lir.filter((pl.col("mvt_minus_sched") >= lo) & (pl.col("mvt_minus_sched") < hi))
        over = b.filter((pl.col("oof") - pl.col(LABEL)) > 0)
        under = b.filter((pl.col("oof") - pl.col(LABEL)) <= 0)
        rows.append(dict(ms_lo=lo, ms_hi=hi, n=b.height, n_rollover=int((b[LABEL] > 72000).sum()),
                         sse=float(b["se"].sum()), sse_share=float(b["se"].sum()) / S,
                         sse_overpredict=float(over["se"].sum()) / S,
                         sse_underpredict=float(under["se"].sum()) / S))
    log("\n=== LIRF unmatched OOF rows by ms band: where the SSE sits, over- vs under-prediction ===")
    log(pl.DataFrame(rows).to_pandas().to_string(index=False))

    # 2. the LFPG hard rows
    hard = j.filter((pl.col("ADEP_mvt") == "LFPG") & (pl.col(LABEL) >= 43200))
    log("\n=== LFPG rows with y >= 12 h in the OOF ===")
    log(hard.select(["FLIGHT_mvt", LABEL, "oof", "mvt_minus_sched", "month", "se"]).to_pandas().to_string(index=False))

    # percentiles of those rows inside LFPG unmatched
    pop = t.filter((pl.col("ADEP_mvt") == "LFPG") & (pl.col("nm_matched") == 0))
    posid = set(t.filter((pl.col("ADEP_mvt") == "LFPG") & (pl.col(LABEL) >= 43200))["MVT_ID_mvt"].to_list())
    num = numeric_visible(t.schema)
    base = pop.filter(~pl.col("MVT_ID_mvt").is_in(list(posid)))
    pr = t.filter(pl.col("MVT_ID_mvt").is_in(list(posid)))
    recs = []
    for c in num:
        bv = base[c].to_numpy().astype(float)
        if np.isfinite(bv).sum() < 50:
            continue
        pv = pr[c].to_numpy().astype(float)
        recs.append(dict(feature=c, n_base=int(np.isfinite(bv).sum()),
                         **{f"row{i}_val": (float(x) if np.isfinite(x) else None) for i, x in enumerate(pv)},
                         **{f"row{i}_pct": round(pct_of(bv, x), 2) for i, x in enumerate(pv)}))
    d = pl.DataFrame(recs, strict=False)
    pcols = [c for c in d.columns if c.endswith("_pct")]
    d = d.with_columns(pl.max_horizontal([(pl.col(c) - 50).abs() for c in pcols]).alias("extremeness")).sort(
        "extremeness", descending=True)
    log(f"\n=== percentile of the {pr.height} LFPG y>=12h rows inside {base.height} other LFPG unmatched rows (top 30) ===")
    log(d.head(30).to_pandas().to_string(index=False))
    d.write_parquet(OUT / "hardcase_lfpg.parquet")

    # 3. a detector aimed at the wide class among LFPG-like rows: leave-one-out at that size
    log("\n=== LFPG unmatched: any single visible feature with |AUC-0.5| > 0.25 for y>=12h? ===")
    y = pop[LABEL].to_numpy() >= 43200
    out = []
    for c in num:
        a = auc(pop[c].to_numpy().astype(float), y)
        if np.isfinite(a):
            out.append((c, a))
    out.sort(key=lambda z: -abs(z[1] - 0.5))
    log(pl.DataFrame(out, schema=["feature", "auc"], orient="row").head(20).to_pandas().to_string(index=False))


# ------------------------------------------------------------------ stage: lobt
def stage_lobt():
    """The matched lane has an EXACT visible bound on the label: |y - mvt_minus_lobt| <= 3606."""
    t = load_train(["ADEP_mvt", "FLIGHT_mvt", LABEL, "mvt_minus_lobt", "mvt_minus_aobt3", "nm_matched", "month"])
    m = t.filter(pl.col("nm_matched") == 1)
    y = m[LABEL].to_numpy().astype(float)
    l = m["mvt_minus_lobt"].to_numpy().astype(float)
    a = m["mvt_minus_aobt3"].to_numpy().astype(float)
    res = {"matched_n": int(m.height), "matched_pos": int((y > 72000).sum()),
           "identity_violations_abs_y_minus_lobt_gt_3606": int((np.abs(y - l) > 3606).sum())}
    for thr in (7200, 14400, 43200, 68394, 72000):
        f = l > thr
        res[f"lobt_gt_{thr}"] = dict(flagged=int(f.sum()), tp=int((y[f] > 72000).sum()),
                                     y_min=float(y[f].min()) if f.sum() else None,
                                     y_max=float(y[f].max()) if f.sum() else None)
    for thr in (43200, 72000):
        f = a > thr
        res[f"aobt3_gt_{thr}"] = dict(flagged=int(f.sum()), tp=int((y[f] > 72000).sum()))
    r = pl.scan_parquet(RANK).select(["MVT_ID_mvt", "ADEP_mvt", "FLIGHT_mvt", "month", "nm_matched",
                                      "mvt_minus_lobt", "mvt_minus_aobt3", "mvt_minus_sched"]).collect()
    sub = pl.read_parquet(ROOT / "outputs/main.parquet").rename({LABEL: "pred_v13"})
    j = r.join(sub, on="MVT_ID_mvt", how="left")
    res["rank_matched_lobt_gt_68394"] = int(((j["nm_matched"] == 1) & (j["mvt_minus_lobt"] > 68394)).sum())
    log(json.dumps(res, indent=1, default=float))
    log("\n=== 2026 matched rows with mvt_minus_lobt > 14400, with the shipped v13 prediction ===")
    log(j.filter((pl.col("nm_matched") == 1) & (pl.col("mvt_minus_lobt") > 14400))
        .sort("mvt_minus_lobt", descending=True).to_pandas().to_string(index=False))
    log("\n=== 2026 unmatched rows with ms >= 43200, with the shipped v13 prediction ===")
    log(j.filter((pl.col("nm_matched") == 0) & (pl.col("mvt_minus_sched") >= 43200))
        .select(["ADEP_mvt", "month", "mvt_minus_sched", "pred_v13"])
        .sort("mvt_minus_sched", descending=True).to_pandas().to_string(index=False))
    (OUT / "lobt.json").write_text(json.dumps(res, indent=1, default=float))


# ------------------------------------------------------------------ stage: part2b
def stage_part2b():
    """Light version of the part2 heteroscedasticity fit: 250k-row subsample, 31 leaves, 150 rounds.

    Use when the full fit does not fit in the CPU budget. Reports the same statistics.
    """
    import lightgbm as lgb
    import scipy.stats as st
    oofd = pl.read_parquet(OOF).drop_nulls()
    t = load_train()
    j = t.join(oofd, on="MVT_ID_mvt", how="inner")
    rng = np.random.default_rng(5)
    keep = np.zeros(j.height, bool)
    keep[rng.choice(j.height, 250000, replace=False)] = True
    j = j.filter(pl.Series(keep))
    resid = (j["oof"] - j[LABEL]).to_numpy().astype(float)
    se = resid ** 2
    tgt = np.log1p(se)
    feats = [c for c in numeric_visible(t.schema) if c != "month"] + CATS
    X = build_xy(j, feats)
    months = j["month"].to_numpy()
    params = dict(objective="regression", learning_rate=0.1, num_leaves=31, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, num_threads=2, deterministic=True, force_row_wise=True, seed=3)
    res = {"n": int(len(se)), "rmse_subsample": float(np.sqrt(se.mean())), "subsampled": True}
    for tag, cols in [("visible_all", None), ("pred_and_lane_only", ["oof_pred", "nm_matched"])]:
        if cols is None:
            XX = X
        else:
            XX = X[["nm_matched"]].copy()
            XX["oof_pred"] = j["oof"].to_numpy()
        pred = np.full(len(tgt), np.nan)
        for m in np.unique(months):
            tr, te = months != m, months == m
            b = lgb.train(params, lgb.Dataset(XX[tr], label=tgt[tr]), num_boost_round=150)
            pred[te] = b.predict(XX[te])
        o = np.argsort(-pred)
        r = {"spearman_pred_vs_se": float(st.spearmanr(pred, se).statistic)}
        for frac in (0.001, 0.01, 0.05, 0.10, 0.25):
            k = int(len(se) * frac)
            r[f"sse_share_top_{frac}"] = float(se[o[:k]].sum() / se.sum())
            r[f"rmse_in_top_{frac}"] = float(np.sqrt(se[o[:k]].mean()))
            r[f"rmse_outside_top_{frac}"] = float(np.sqrt(se[o[k:]].mean()))
        if cols is None:
            r["top_gain_last_fold"] = [(a, float(v)) for a, v in
                                       sorted(zip(XX.columns, b.feature_importance("gain")), key=lambda z: -z[1])[:20]]
        res[tag] = r
    oo = np.argsort(-se)
    for frac in (0.001, 0.01, 0.10):
        k = int(len(se) * frac)
        res[f"oracle_sse_share_top_{frac}"] = float(se[oo[:k]].sum() / se.sum())
    log(json.dumps(res, indent=1))
    (OUT / "part2b.json").write_text(json.dumps(res, indent=1, default=float))


STAGES = dict(counts=stage_counts, marginal=stage_marginal, cats=stage_cats, combo=stage_combo,
              widen=stage_widen, part2=stage_part2, price=stage_price, hardcase=stage_hardcase, part2b=stage_part2b, lobt=stage_lobt)

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name in (STAGES if which == "all" else [which]):
        log(f"\n################ stage {name} ################")
        STAGES[name]()
