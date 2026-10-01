"""Month-block evaluation of fingerprint features inside a lightweight clock-source mixture.

This is a *proxy* of src/prc/mixture.py (same classes, same posterior-mean routing via prc.clocks.routed_prediction,
same +-60 s label), but with a row-local base feature set instead of the full production features, so that the
*difference* made by a feature group can be measured cheaply. Folds: jan and jul 2025 held out in turn, the model
fitted on the other 11 months (the months scored on the leaderboard are Jan/Jul).

usage: python reporting_fingerprint_eval.py --apt LIRF --sets base,base+fa,base+fa+ta [--label tol60]
"""
from __future__ import annotations

import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from reporting_fingerprint_common import CACHE, load_all
from reporting_fingerprint_feats import build as build_fp, build_ff
from prc.clocks import routed_prediction

CLASSES = ["genuine", "aobt3", "sched", "eobt1", "lobt", "iobt"]
CANDS = [("aobt3", "AOBT_3_flt"), ("sched", "SCHED_TIME_UTC_mvt"), ("eobt1", "EOBT_1_flt"), ("lobt", "LOBT_flt"), ("iobt", "IOBT_flt")]
CATS = ["apt", "AIRCRAFT_OPERATOR_flt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "MARKET_SEGMENT_flt", "ADES_mvt", "FLIGHT_TYPE_flt"]
FA = ["fa_day_rate", "fa_day_n", "fa_dayop_rate", "fa_dayop_n", "fa_op_rate_split", "fa_op_n_split", "fa_daystand_rate",
      "fa_win2h_rate", "fa_win2h_n", "fa_opwin3h_rate", "fa_opwin3h_n"]
FF = ["k_fl_fmt", "k_fl_pfx", "fl_len", "fl_n_other", "fl_nm_rate_other"]
DUP = ["dup_n", "dup_n_unm_twins", "dup_n_m_twins", "dup_mvt_span"]
TA = ["ta_copy", "ta_gap_sched", "ta_gap_mvt", "ta_delay", "ta_same_op", "ta_taxi", "ta_next_inb_minus_sched"]


def sec(a, b):
    return (pl.col(a) - pl.col(b)).dt.total_seconds().cast(pl.Float64)


def frame(apt: str | None, label_tol: int, extra: list[pl.DataFrame] | None = None) -> pl.DataFrame:
    df = load_all().filter(pl.col("PHASE_mvt") == "DEP")
    if apt:
        df = df.filter(pl.col("apt").is_in(apt.split(",")))
    df = df.with_columns(
        [sec("MVT_TIME_UTC_mvt", c).alias(f"c_{k}") for k, c in CANDS]
        + [sec("AOBT_3_flt", "EOBT_1_flt").alias("g_a_e"), sec("AOBT_3_flt", "SCHED_TIME_UTC_mvt").alias("g_a_s"),
           sec("SCHED_TIME_UTC_mvt", "EOBT_1_flt").alias("g_s_e"), sec("LOBT_flt", "EOBT_1_flt").alias("g_l_e"),
           sec("SCHED_TIME_UTC_mvt", "LOBT_flt").alias("g_s_l"), sec("AOBT_3_flt", "LOBT_flt").alias("g_a_l"),
           pl.col("FLIGHT_ID_mvt").is_not_null().cast(pl.Float64).alias("nm_matched"),
           pl.col("MVT_TIME_UTC_mvt").dt.hour().cast(pl.Float64).alias("hour_utc"),
           pl.col("SCHED_TIME_UTC_mvt").dt.hour().cast(pl.Float64).alias("sched_hour_utc"),
           pl.col("MVT_TIME_UTC_mvt").dt.weekday().cast(pl.Float64).alias("dow")]
    )
    cc = [f"c_{k}" for k, _ in CANDS]
    df = df.with_columns(pl.max_horizontal(cc).alias("cand_max"), pl.min_horizontal(cc).alias("cand_min"),
                         pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Float64) for c in cc]).alias("cand_n"))
    df = df.with_columns((pl.col("cand_max") - pl.col("cand_min")).alias("cand_spread"))
    for c in CATS:
        df = df.with_columns(pl.col(c).cast(pl.Categorical).to_physical().cast(pl.Int32).alias(f"k_{c}"))
    if label_tol != 60:
        gap = lambda c: (pl.col("BLOCK_TIME_UTC_mvt") - pl.col(c)).dt.total_seconds().abs()  # noqa: E731
        ch = pl.when(pl.lit(False)).then(pl.lit("genuine"))
        for k, c in CANDS:
            ch = ch.when(gap(c) <= label_tol).then(pl.lit(k))
        df = df.with_columns(pl.when(pl.col("split") == "train").then(ch.otherwise(pl.lit("genuine"))).alias("clock_src"))
    else:
        df = df.with_columns(pl.col("clock_src").replace("rollover_24h", "genuine"))
    for e in extra or []:
        df = df.join(e, on="MVT_ID_mvt", how="left")
    return df


BASE = [f"c_{k}" for k, _ in CANDS] + ["g_a_e", "g_a_s", "g_s_e", "g_l_e", "g_s_l", "g_a_l", "nm_matched", "hour_utc",
                                        "sched_hour_utc", "dow", "cand_max", "cand_min", "cand_n", "cand_spread"] + [f"k_{c}" for c in CATS]


def fit_eval(df: pl.DataFrame, feats: list[str], months=(1, 7), threads=2, rounds=300, seed=0, return_pred=False, sub=1.0, gen_feats: list[str] | None = None):
    gen_feats = gen_feats or feats
    lab = df.filter(pl.col("split") == "train")
    cats = [f for f in feats if f in ("k_apt", "k_MARKET_SEGMENT_flt", "k_RUNWAY_mvt", "k_FLIGHT_TYPE_flt", "k_fl_fmt")]
    out = {}
    preds = []
    for m in months:
        tr = lab.filter(pl.col("month") != m)
        w_tr = None
        if sub < 1.0:  # keep every LIRF / NM-unmatched row, subsample the rest with inverse-probability weights
            keep_all = (tr["apt"] == "LIRF") | tr["FLIGHT_ID_mvt"].is_null()
            u = np.random.default_rng(seed + m).random(tr.height)
            sel = keep_all.to_numpy() | (u < sub)
            tr = tr.filter(pl.Series(sel))
            w_tr = np.where(((tr["apt"] == "LIRF") | tr["FLIGHT_ID_mvt"].is_null()).to_numpy(), 1.0, 1.0 / sub)
        te = lab.filter(pl.col("month") == m)
        Xtr = tr.select(feats).to_numpy().astype(np.float32)
        Xte = te.select(feats).to_numpy().astype(np.float32)
        Gtr = tr.select(gen_feats).to_numpy().astype(np.float32)
        Gte = te.select(gen_feats).to_numpy().astype(np.float32)
        gcats = [f for f in gen_feats if f in cats or f in ("k_apt", "k_MARKET_SEGMENT_flt", "k_RUNWAY_mvt", "k_FLIGHT_TYPE_flt", "k_fl_fmt")]
        ytr_cls = tr["clock_src"].replace_strict({c: i for i, c in enumerate(CLASSES)}, return_dtype=pl.Int32).to_numpy()
        yte_cls = te["clock_src"].replace_strict({c: i for i, c in enumerate(CLASSES)}, return_dtype=pl.Int32).to_numpy()
        y_tr = tr["TAXITIME_SEC_mvt"].to_numpy(); y_te = te["TAXITIME_SEC_mvt"].to_numpy()
        common = dict(learning_rate=0.1, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
                      bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=threads, seed=seed, max_cat_to_onehot=8,
                      cat_smooth=20, max_bin=63)
        real = np.isin(ytr_cls, [0, 1])
        dg = lgb.Dataset(Gtr[real], y_tr[real], weight=None if w_tr is None else w_tr[real], feature_name=gen_feats, categorical_feature=gcats, free_raw_data=True)
        gm = lgb.train({**common, "objective": "regression"}, dg, rounds)
        gen = gm.predict(Gte)
        dc = lgb.Dataset(Xtr, ytr_cls, weight=w_tr, feature_name=feats, categorical_feature=cats, free_raw_data=True)
        cm = lgb.train({**common, "objective": "multiclass", "num_class": len(CLASSES)}, dc, rounds)
        P = cm.predict(Xte)
        cand = te.select([f"c_{k}" for k, _ in CANDS]).to_numpy().astype(float)
        pred = np.clip(routed_prediction(P, cand, gen, CLASSES), 0, None)
        ll = float(-np.mean(np.log(np.clip(P[np.arange(len(P)), yte_cls], 1e-12, 1))))
        rm = float(np.sqrt(np.mean((pred - y_te) ** 2)))
        oracle = np.where(yte_cls > 0, cand[np.arange(len(cand)), np.clip(yte_cls - 1, 0, 4)], gen)
        out[m] = dict(n=len(y_te), rmse=rm, logloss=ll, rmse_gen_only=float(np.sqrt(np.mean((gen - y_te) ** 2))),
                      oracle=float(np.sqrt(np.mean((np.nan_to_num(oracle, nan=0) - y_te) ** 2))))
        imp = dict(zip(feats, cm.feature_importance("gain")))
        out[m]["top_clf_gain"] = sorted(((k, round(v / sum(imp.values()), 4)) for k, v in imp.items()), key=lambda x: -x[1])[:12]
        preds.append(pl.DataFrame({"MVT_ID_mvt": te["MVT_ID_mvt"], "date": te["date"], "apt": te["apt"], "y": y_te, "pred": pred,
                                   "gen": gen, "src": te["clock_src"], **{f"p_{c}": P[:, i] for i, c in enumerate(CLASSES)}}))
    allp = pl.concat(preds)
    out["pooled_rmse"] = float(np.sqrt(np.mean((allp["pred"].to_numpy() - allp["y"].to_numpy()) ** 2)))
    return (out, allp) if return_pred else out


def paired_boot(a: pl.DataFrame, b: pl.DataFrame, n=2000, seed=0):
    """Day-block paired bootstrap of RMSE(b) - RMSE(a)."""
    j = a.select("MVT_ID_mvt", "date", "y", pl.col("pred").alias("pa")).join(b.select("MVT_ID_mvt", pl.col("pred").alias("pb")), on="MVT_ID_mvt")
    g = j.group_by("date").agg(((pl.col("pa") - pl.col("y")) ** 2).sum().alias("sa"), ((pl.col("pb") - pl.col("y")) ** 2).sum().alias("sb"), pl.len().alias("n"))
    sa, sb, nn = g["sa"].to_numpy(), g["sb"].to_numpy(), g["n"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (n, len(g)))
    d = np.sqrt(sb[idx].sum(1) / nn[idx].sum(1)) - np.sqrt(sa[idx].sum(1) / nn[idx].sum(1))
    base = np.sqrt(sb.sum() / nn.sum()) - np.sqrt(sa.sum() / nn.sum())
    return dict(delta=float(base), ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], p_better=float((d < 0).mean()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apt", default=None)
    ap.add_argument("--sets", default="base,base+fa,base+fa+ta")
    ap.add_argument("--label", type=int, default=60)
    ap.add_argument("--rounds", type=int, default=150)
    ap.add_argument("--tag", default="")
    ap.add_argument("--sub", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    fp = build_fp()
    df = frame(a.apt, a.label, [fp, build_ff()])
    groups = {"base": BASE, "fa": FA, "ta": TA, "ff": FF, "dup": DUP}
    res, preds = {}, {}
    for s in a.sets.split(","):
        if "/" in s:  # "G/C": genuine-head feature groups / classifier feature groups
            gs, cs = s.split("/")
            gfe = [f for g in gs.split("+") for f in groups[g]]
            feats = [f for g in cs.split("+") for f in groups[g]]
        else:
            feats = [f for g in s.split("+") for f in groups[g]]
            gfe = feats
        t0 = time.time()
        r, p = fit_eval(df, feats, rounds=a.rounds, return_pred=True, sub=a.sub, seed=a.seed, gen_feats=gfe)
        r["sec"] = round(time.time() - t0)
        res[s], preds[s] = r, p
        p.write_parquet(CACHE / f"oof_{a.tag or (a.apt or 'ALL')}_{s.replace('/', '_over_')}_tol{a.label}_s{a.seed}.parquet")
        print(s, json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != 'top_clf_gain'}) for k, v in r.items()}, default=str), flush=True)
        print("  top gain (jan):", r[1]["top_clf_gain"], flush=True)
    sets = a.sets.split(",")
    for s in sets[1:]:
        print(f"paired {sets[0]} -> {s}:", paired_boot(preds[sets[0]], preds[s]))
        for apt in sorted(preds[s]["apt"].unique()):
            if preds[s].filter(pl.col("apt") == apt).height and len(sorted(preds[s]["apt"].unique())) > 1:
                pa, pb = preds[sets[0]].filter(pl.col("apt") == apt), preds[s].filter(pl.col("apt") == apt)
                print("   ", apt, round(float(np.sqrt(np.mean((pa["pred"] - pa["y"]).to_numpy() ** 2))), 2), "->",
                      round(float(np.sqrt(np.mean((pb["pred"] - pb["y"]).to_numpy() ** 2))), 2))
