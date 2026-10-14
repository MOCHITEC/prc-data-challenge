"""covw: coverage-matched training of the learned ADS-B residual (local, read-only analysis).

Hypothesis: 12:v2 is trained on 2025 adsb.lol coverage regimes that differ per airport from 2026's. Arms (all V2
params, fixed production rounds = 778, the 12:v2 rule; features of adsb_learn_features.design):
  A   production: all training months, unweighted
  B1  month-similarity weights: per airport x month, w = sum over 2026 targets (Jan, Jul) of
      exp(-((d - d_min) / 1)^2 / 2) on the coverage-descriptor distance of local_covw_coverage.py, floor 0.1,
      renormalised to mean 1 within airport
  B2  row-level importance weights from a label-free coverage-only domain classifier 2026 vs 2025 (LightGBM, OOF):
      w = (p / (1 - p))^0.5, clipped to [0.1, 10] x airport median, renormalised to mean 1 within airport
  C   A + coverage-normalised features (fixes per minute / fraction) replacing the raw fix counts
Validation: leave-one-month-out over the 12 months of 2025 (train on the other 11). Metrics on the swap scope
(ground-fix rows, LIRF and LTFM excluded; other rows count as zero change, all labelled rows in the denominator):
plain gain, importance-weighted gain (test rows weighted by the untempered domain ratio, estimating 2026 risk),
and gain on the "2026-like" cells (per airport the 2 closest 2025 months to Jan 2026 and to Jul 2026).
Paired day-block bootstrap (adsb_learn_cv.boot; weighted variant for the IW metric).

    POLARS_MAX_THREADS=2 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage .venv/bin/python analysis/local_covw_cv.py prep
    ... local_covw_cv.py weights
    ... local_covw_cv.py cv [arms]       # resumable, per fold x arm npz in $SCRATCH/agents/covw/folds
    ... local_covw_cv.py eval
    ... local_covw_cv.py apply12 <arm> <in.parquet> <with_lfpg yes|no>
Writes only under $SCRATCH/agents/covw/.
"""
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

ARGV = list(sys.argv)
sys.path.insert(0, "analysis")
import local_adsb_learn_months as M  # noqa: E402  (overwrites sys.argv)

L, CV = M.L, M.CV
OUT = L.OUT
RES = os.path.join(L.SCR, "agents", "covw")
FOLDS = os.path.join(RES, "folds")
os.makedirs(FOLDS, exist_ok=True)
V2 = dict(num_leaves=255, min_data_in_leaf=300, feature_fraction=0.7)
PRM = {**CV.PARAMS, **V2, "num_threads": 2}
ROUNDS = int(os.environ.get("ROUNDS", "778"))
MONTHS = ["2025-%02d" % m for m in range(1, 13)]
EXCL = ["LIRF", "LTFM"]
COUNTS = ["n_ground", "stand_nfix", "n_win", "n_win_ground", "post_n"]
NORM = ["r_post", "r_stand", "f_wing", "r_ground", "f_gap", "r_win"]
DOM = ["ground_fix", "post_n", "n_ground", "n_win", "n_win_ground", "stand_nfix", "post_max_gap", "gap_before_dep1",
       "has_stand", "has_push", "r_post", "f_wing", "cat_join", "cat_adep", "has_glast"]
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(220)
pl.Config.set_tbl_cols(30)


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024
    return 0.0


def norm_cols():
    dur = ((pl.col("t_glast") - pl.col("t_anchor")) / 60.0).clip(lower_bound=1.0)
    return [(pl.col("post_n") / dur).alias("r_post"),
            (pl.col("stand_nfix") / (pl.col("stand_dur") / 60.0).clip(lower_bound=1.0)).alias("r_stand"),
            (pl.col("n_win_ground") / pl.col("n_win").clip(lower_bound=1)).alias("f_wing"),
            (pl.col("n_ground") / ((pl.col("t_end") - pl.col("t_first")) / 60.0).clip(lower_bound=1.0)).alias("r_ground"),
            (pl.col("post_max_gap") / (dur * 60.0)).alias("f_gap"),
            (pl.col("n_win") / 270.0).alias("r_win")]


def dom_frame(F):
    return F.with_columns(norm_cols()).select(
        pl.col("ground_fix").cast(pl.Float32), *[pl.col(c).cast(pl.Float32) for c in DOM[1:8]],
        pl.col("stand_dur").is_not_null().cast(pl.Float32).alias("has_stand"), pl.col("t_push").is_not_null().cast(pl.Float32).alias("has_push"),
        pl.col("r_post").cast(pl.Float32), pl.col("f_wing").cast(pl.Float32),
        pl.col("join").replace_strict(["cs", "fn"], [0, 1], default=-1).cast(pl.Float32).alias("cat_join"),
        pl.col("ADEP_mvt").replace_strict(list(L.s1.AIRPORTS), list(range(10)), default=-1).cast(pl.Float32).alias("cat_adep"),
        pl.col("t_glast").is_not_null().cast(pl.Float32).alias("has_glast"))


# ----------------------------------------------------------------------------------------------- prep
def cmd_prep():
    keep = ["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "mvt_minus_aobt3", "b", "y", "day"]
    D = pl.concat([CV.load_labelled().select(keep), M.load_novdec().select(keep), M.load_novdec("aso", "aso").select(keep),
                   M.load_novdec("fmj", "fmj").select(keep)])
    F = pl.concat([pl.read_parquet(os.path.join(OUT, f)) for f in ("feat_2025.parquet", "feat_2025novdec.parquet", "feat_2025aso.parquet", "feat_2025fmj.parquet")],
                  how="diagonal_relaxed").drop("ADEP_mvt", "day")
    D = D.join(F, on="MVT_ID_mvt", how="left").with_columns(pl.col("icao").is_not_null().alias("has"),
                                                           pl.col("day").str.slice(0, 7).alias("ym"))
    del F
    days = sorted(D["day"].unique().to_list())
    D = D.with_columns(pl.col("day").replace_strict({d: i for i, d in enumerate(days)}).alias("day_idx"))
    print("rows", D.height, "has", int(D["has"].sum()), "days", len(days), flush=True)
    D.select("MVT_ID_mvt", "ADEP_mvt", "ym", "day", "day_idx", "y", "b", "has", pl.col("ground_fix").fill_null(False).alias("gf")
             ).write_parquet(os.path.join(RES, "meta.parquet"))
    H = D.filter("has")
    del D
    dom_frame(H).write_parquet(os.path.join(RES, "dom25.parquet"))
    X, feats, cats = L.design(H, "b")
    X = X.with_columns(norm_cols())
    allf = feats + NORM
    np.save(os.path.join(RES, "X.npy"), X.select(allf).to_numpy().astype(np.float32))
    json.dump({"feats": feats, "cats": cats, "allf": allf}, open(os.path.join(RES, "feats.json"), "w"))
    print("X", X.height, len(allf), "rss", round(rss_mb()), flush=True)
    del X, H
    G = pl.read_parquet(os.path.join(OUT, "feat_2026.parquet"))
    dom_frame(G).with_columns(pl.lit(G["day"].str.slice(0, 7)).alias("ym"), pl.lit(G["ADEP_mvt"]).alias("ADEP_mvt")
                              ).write_parquet(os.path.join(RES, "dom26.parquet"))


# ----------------------------------------------------------------------------------------------- weights
def cmd_weights():
    from sklearn.metrics import roc_auc_score
    meta = pl.read_parquet(os.path.join(RES, "meta.parquet")).filter("has")
    A = pl.read_parquet(os.path.join(RES, "dom25.parquet"))
    B = pl.read_parquet(os.path.join(RES, "dom26.parquet"))
    X = np.vstack([A.select(DOM).to_numpy(), B.select(DOM).to_numpy()]).astype(np.float32)
    yv = np.r_[np.zeros(A.height), np.ones(B.height)]
    fold = np.random.default_rng(0).integers(0, 5, len(yv))
    p = np.zeros(len(yv))
    prm = dict(objective="binary", num_leaves=63, learning_rate=0.05, min_data_in_leaf=200, feature_fraction=0.9,
               num_threads=2, verbose=-1, seed=1)
    for k in range(5):
        m = lgb.train(prm, lgb.Dataset(X[fold != k], yv[fold != k], feature_name=DOM, categorical_feature=["cat_join", "cat_adep"]), 300)
        p[fold == k] = m.predict(X[fold == k])
        print("dom fold", k, flush=True)
    auc = float(roc_auc_score(yv, p))
    apt = np.r_[meta["ADEP_mvt"].to_numpy(), B["ADEP_mvt"].to_numpy()]
    per = {a: round(float(roc_auc_score(yv[apt == a], p[apt == a])), 3) for a in sorted(set(apt)) if len(set(yv[apt == a])) == 2}
    print("domain AUC (coverage only, OOF):", round(auc, 4), per)
    imp = dict(zip(DOM, m.feature_importance("gain") / m.feature_importance("gain").sum()))
    print("gain share", {k: round(float(v), 3) for k, v in sorted(imp.items(), key=lambda t: -t[1])})
    p25 = np.clip(p[: A.height], 1e-4, 1 - 1e-4)
    ratio = p25 / (1 - p25) * (A.height / B.height)
    apt25 = meta["ADEP_mvt"].to_numpy()
    w_iw = np.zeros(A.height)
    w_b2 = np.zeros(A.height)
    for a in set(apt25):
        s = apt25 == a
        r = np.clip(ratio[s], 0, 20 * np.median(ratio[s]))
        w_iw[s] = r / r.mean()
        t = np.sqrt(ratio[s])
        med = np.median(t)
        t = np.clip(t, 0.1 * med, 10 * med)
        w_b2[s] = t / t.mean()
    # month-similarity weights B1
    Sim = pl.read_parquet(os.path.join(RES, "similarity.parquet")).with_columns(
        (pl.col("dist") - pl.col("dist").min().over("airport", "target")).alias("dd"))
    Sim = Sim.with_columns((-0.5 * pl.col("dd") ** 2).exp().alias("k")).group_by("airport", "ym").agg(pl.col("k").sum())
    Sim = Sim.with_columns(pl.col("k").clip(lower_bound=0.1))
    Wm = meta.select("ADEP_mvt", "ym").with_row_index("i").join(Sim.rename({"airport": "ADEP_mvt"}), on=["ADEP_mvt", "ym"], how="left").sort("i")
    k = Wm["k"].fill_null(1.0).to_numpy()
    w_b1 = np.zeros(A.height)
    for a in set(apt25):
        s = apt25 == a
        w_b1[s] = k[s] / k[s].mean()
    W = pl.DataFrame({"MVT_ID_mvt": meta["MVT_ID_mvt"], "ADEP_mvt": apt25, "ym": meta["ym"], "p26": p25,
                      "w_iw": w_iw, "w_B1": w_b1, "w_B2": w_b2})
    W.write_parquet(os.path.join(RES, "weights.parquet"))
    json.dump({"auc": auc, "auc_per_airport": per, "gain_share": {k_: float(v) for k_, v in imp.items()}},
              open(os.path.join(RES, "domain.json"), "w"), indent=1)
    print(W.group_by("ADEP_mvt", "ym").agg(pl.col("w_B1").mean().round(2), pl.col("w_B2").mean().round(2), pl.col("w_iw").mean().round(2))
          .pivot(on="ADEP_mvt", index="ym", values="w_B2").sort("ym"))
    print(W.group_by("ADEP_mvt", "ym").agg(pl.col("w_B1").mean().round(2)).pivot(on="ADEP_mvt", index="ym", values="w_B1").sort("ym"))
    ess = {a: round(float(w_b2[apt25 == a].sum() ** 2 / (w_b2[apt25 == a] ** 2).sum() / (apt25 == a).sum()), 3) for a in sorted(set(apt25))}
    print("B2 relative ESS per airport", ess)


# ----------------------------------------------------------------------------------------------- cv
ARMS = {"A": ("feats", None), "B1": ("feats", "w_B1"), "B2": ("feats", "w_B2"), "C": ("normf", None)}


def arm_feats(fj, kind):
    if kind == "feats":
        return fj["feats"]
    return [f for f in fj["feats"] if f not in COUNTS] + NORM


def cmd_cv(arms):
    fj = json.load(open(os.path.join(RES, "feats.json")))
    allf, cats = fj["allf"], fj["cats"]
    meta = pl.read_parquet(os.path.join(RES, "meta.parquet")).filter("has")
    W = pl.read_parquet(os.path.join(RES, "weights.parquet"))
    assert (W["MVT_ID_mvt"] == meta["MVT_ID_mvt"]).all()
    X = np.load(os.path.join(RES, "X.npy"), mmap_mode="r")
    r = np.clip(meta["y"].to_numpy() - meta["b"].to_numpy(), -CV.CLIP_Y, CV.CLIP_Y)
    ym = meta["ym"].to_numpy()
    ids = meta["MVT_ID_mvt"].to_numpy()
    order = [7, 1, 11, 4, 9, 2, 12, 6, 3, 10, 5, 8]  # Jan/Jul first so partial results are informative
    for mo in order:
        m_ = "2025-%02d" % mo
        te, tr = ym == m_, ym != m_
        for arm in arms:
            dst = os.path.join(FOLDS, f"{m_}_{arm}.npz")
            if os.path.exists(dst):
                continue
            kind, wcol = ARMS[arm]
            fn = arm_feats(fj, kind)
            idx = [allf.index(f) for f in fn]
            t0 = time.time()
            Xtr = np.asarray(X[tr][:, idx])
            w = None if wcol is None else W[wcol].to_numpy()[tr]
            ds = lgb.Dataset(Xtr, r[tr], weight=w, feature_name=fn, categorical_feature=cats, free_raw_data=True)
            ds.construct()
            del Xtr
            m = lgb.train(PRM, ds, num_boost_round=ROUNDS)
            del ds
            Xte = np.asarray(X[te][:, idx])
            c = np.clip(m.predict(Xte), -CV.CLIP_C, CV.CLIP_C)
            np.savez(dst, ids=ids[te], c=c)
            print(f"{m_} {arm}: train {int(tr.sum())} test {int(te.sum())} {time.time() - t0:.0f} s rss {rss_mb():.0f} MB", flush=True)
            del m, Xte


# ----------------------------------------------------------------------------------------------- eval
def wboot(d, w, di, nd, seed=CV.SEED):
    s1 = np.bincount(di, weights=d * w, minlength=nd)
    s0 = np.bincount(di, weights=w, minlength=nd)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, nd, size=(CV.B, nd))
    s = s1[pick].sum(1) / s0[pick].sum(1)
    return float((d * w).sum() / w.sum()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def like_cells(n=2):
    Sim = pl.read_parquet(os.path.join(RES, "similarity.parquet"))
    return set(Sim.filter(pl.col("rank") <= n).select(pl.concat_str("airport", pl.lit("|"), "ym")).to_series().to_list())


def cmd_eval():
    meta = pl.read_parquet(os.path.join(RES, "meta.parquet"))
    W = pl.read_parquet(os.path.join(RES, "weights.parquet")).select("MVT_ID_mvt", "w_iw")
    meta = meta.join(W, on="MVT_ID_mvt", how="left").with_columns(pl.col("w_iw").fill_null(1.0))
    ids = meta["MVT_ID_mvt"].to_numpy()
    pos = {v: i for i, v in enumerate(ids)}
    arms = [a for a in ARMS if all(os.path.exists(os.path.join(FOLDS, f"{m}_{a}.npz")) for m in MONTHS)]
    done_m = [m for m in MONTHS if all(os.path.exists(os.path.join(FOLDS, f"{m}_{a}.npz")) for a in ("A",))]
    print("complete arms", arms, "| months with A", len(done_m))
    c = {}
    for a in ARMS:
        c[a] = np.full(meta.height, np.nan)
        for m in MONTHS:
            p = os.path.join(FOLDS, f"{m}_{a}.npz")
            if os.path.exists(p):
                z = np.load(p)
                c[a][np.array([pos[v] for v in z["ids"]])] = z["c"]
    apt, ym, gf, has = meta["ADEP_mvt"].to_numpy(), meta["ym"].to_numpy(), meta["gf"].to_numpy(), meta["has"].to_numpy()
    scope = has & gf & ~np.isin(apt, EXCL)
    e0 = (meta["y"] - meta["b"]).to_numpy()
    di, nd = meta["day_idx"].to_numpy(), int(meta["day_idx"].max()) + 1
    wiw = meta["w_iw"].to_numpy()
    cells = like_cells()
    key = np.char.add(np.char.add(apt.astype(str), "|"), ym.astype(str))
    like = np.isin(key, list(cells))
    out = {}
    base_ok = ~np.isnan(c["A"])
    for a in [x for x in ARMS if x != "A"]:
        ok = ~np.isnan(c[a]) & base_ok
        avail = sorted({m for m in MONTHS if (ok & has & (ym == m)).any()})
        sel_m = np.isin(ym, avail)
        d = np.where(scope & ok, (e0 - c["A"]) ** 2 - (e0 - c[a]) ** 2, 0.0)
        d = np.where(np.isnan(d), 0.0, d)
        res = {"months": avail}
        print(f"\n=== {a} vs A (>0 = {a} better), swap scope, months {len(avail)}")
        for lab, sel in (("all", sel_m), ("all ex LFPG", sel_m & (apt != "LFPG")), ("2026-like cells", sel_m & like),
                         ("2026-like ex LFPG", sel_m & like & (apt != "LFPG")), ("Jan", ym == "2025-01"), ("Jul", ym == "2025-07")):
            if sel.sum() == 0:
                continue
            g, lo, hi = CV.boot(d[sel], di[sel], nd)
            res[lab] = (g, lo, hi)
            print(f"  {lab:20s} n={sel.sum():8d} gain {g:7.1f} [{lo:7.1f}, {hi:7.1f}]")
        sel = sel_m & has
        g, lo, hi = wboot(d[sel], wiw[sel], di[sel], nd)
        res["IW (2026-weighted, ADS-B rows)"] = (g, lo, hi)
        g2, lo2, hi2 = CV.boot(d[sel], di[sel], nd)
        print(f"  IW-weighted (ADS-B rows) gain {g:7.1f} [{lo:7.1f}, {hi:7.1f}] | unweighted same rows {g2:7.1f} [{lo2:7.1f}, {hi2:7.1f}]")
        res["unw ADS-B rows"] = (g2, lo2, hi2)
        rows = []
        for m in avail:
            s = ym == m
            g, lo, hi = CV.boot(d[s], di[s], nd)
            rows.append(dict(month=m, gain=round(g, 1), lo=round(lo, 1), hi=round(hi, 1)))
        res["per_month"] = rows
        print("  per month:", " ".join(f"{r_['month'][5:]}:{r_['gain']:+.0f}" for r_ in rows),
              f"| positive {sum(r_['gain'] > 0 for r_ in rows)}/{len(rows)}")
        rows = []
        for ap in sorted(set(apt) - set(EXCL)):
            s = sel_m & (apt == ap)
            g, lo, hi = CV.boot(d[s], di[s], nd)
            sl = s & like
            gl = CV.boot(d[sl], di[sl], nd) if sl.any() else (np.nan, np.nan, np.nan)
            rows.append(dict(airport=ap, gain=round(g, 1), lo=round(lo, 1), hi=round(hi, 1), like=round(gl[0], 1), like_lo=round(gl[1], 1), like_hi=round(gl[2], 1)))
        res["per_airport"] = rows
        print(pl.DataFrame(rows))
        dd = d[sel_m]
        o = np.argsort(-np.abs(dd))
        res["top50_share"] = float(1 - dd[o[50:]].sum() / dd.sum()) if dd.sum() != 0 else None
        print("  top-50 share", res["top50_share"])
        out[a] = res
    ga = np.where(scope & base_ok, e0 ** 2 - (e0 - np.nan_to_num(c["A"])) ** 2, 0.0)
    print("\nA gain over base on swap scope (all rows denominator):", round(float(ga[base_ok | ~has].mean()), 1))
    json.dump(out, open(os.path.join(RES, "eval.json"), "w"), indent=1)


# ----------------------------------------------------------------------------------------------- apply12
def cmd_apply12(arm, src, with_lfpg):
    """2026 swap delta 12:v2 -> 12:v2 <arm> (same 12 months, rounds, params; only the weights / features differ) on
    ground-fix rows, LIRF and LTFM excluded (LFPG only if with_lfpg == 'yes'; production carries 7:v2 there, so then
    old = 7:v2 at LFPG). b = file - c_old(b) by fixed point, new = file + c_new(b) - c_old(b) (as local_adsb_apply_swap)."""
    import local_adsb_apply_swap as W_
    fj = json.load(open(os.path.join(RES, "feats.json")))
    allf, cats = fj["allf"], fj["cats"]
    meta = pl.read_parquet(os.path.join(RES, "meta.parquet")).filter("has")
    Wt = pl.read_parquet(os.path.join(RES, "weights.parquet"))
    X = np.load(os.path.join(RES, "X.npy"), mmap_mode="r")
    r = np.clip(meta["y"].to_numpy() - meta["b"].to_numpy(), -CV.CLIP_Y, CV.CLIP_Y)
    ym = meta["ym"].to_numpy()
    kind, wcol = ARMS[arm]
    spec = [("old12", fj["feats"], None, None, ROUNDS), ("new12", arm_feats(fj, kind), wcol, None, ROUNDS)]
    if with_lfpg == "yes":
        sel7 = np.isin(ym, ["2025-%02d" % m for m in (1, 7, 8, 9, 10, 11, 12)])
        spec.append(("old7", fj["feats"], None, sel7, W_.rounds("7:v2")))
    models = {}
    for tag, fn, wc, sel, n in spec:
        idx = [allf.index(f) for f in fn]
        Xa = np.asarray(X[:, idx]) if sel is None else np.asarray(X[sel][:, idx])
        ra = r if sel is None else r[sel]
        w = None if wc is None else Wt[wc].to_numpy()
        ds = lgb.Dataset(Xa, ra, weight=w, feature_name=fn, categorical_feature=cats, free_raw_data=True)
        ds.construct()
        del Xa
        models[tag] = (lgb.train(PRM, ds, num_boost_round=n), fn)
        del ds
        print(f"{tag}: {len(ra)} rows, {n} rounds, weights {wc}, rss {rss_mb():.0f} MB", flush=True)
    del X
    sub = pl.read_parquet(src)
    rk = pl.scan_parquet("data/raw/ranking.parquet").select("MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt").collect()
    F26 = pl.read_parquet(os.path.join(OUT, "feat_2026.parquet"))
    excl = ["LTFM", "LIRF"] + ([] if with_lfpg == "yes" else ["LFPG"])
    G = (F26.drop("ADEP_mvt").join(sub.rename({"TAXITIME_SEC_mvt": "file"}).join(rk, on="MVT_ID_mvt"), on="MVT_ID_mvt")
         .filter(~pl.col("ADEP_mvt").is_in(excl) & pl.col("ground_fix").fill_null(False)).with_columns(pl.col("file").cast(pl.Float64)))
    lf = G["ADEP_mvt"].to_numpy() == "LFPG"

    def corr(tag, b):
        m, fn = models[tag]
        Xg, _, _ = L.design(G.with_columns(pl.Series("b", b)), "b")
        Xg = Xg.with_columns(norm_cols())
        return np.clip(m.predict(Xg.select(fn).to_numpy().astype(np.float32)), -CV.CLIP_C, CV.CLIP_C)

    def c_old(b):
        return np.where(lf, corr("old7", b), corr("old12", b)) if lf.any() else corr("old12", b)

    f = G["file"].to_numpy()
    b = f.copy()
    for _ in range(3):
        b = f - c_old(b)
    co, cn = c_old(b), corr("new12", b)
    new_v = np.round(f + cn - co)
    new_v = np.where(new_v < 0, f, new_v)
    R = G.select("MVT_ID_mvt", "ADEP_mvt", "file").with_columns(pl.Series("new", new_v), (pl.Series("new", new_v) - pl.col("file")).alias("delta"),
                                                                pl.Series("b", b), pl.Series("c_old", co), pl.Series("c_new", cn))
    R.select("MVT_ID_mvt", "delta", "ADEP_mvt", "file", "new", "b").write_parquet(os.path.join(RES, "rank_delta_12m.parquet"))
    d = R["delta"].to_numpy()
    summ = dict(arm=arm, input=src, excluded=excl, ground_rows=R.height, changed=int((d != 0).sum()),
                sum_d2_over_N=float((d ** 2).sum()) / sub.height, N=sub.height, mean=float(d.mean()),
                mean_abs=float(np.abs(d).mean()), p99_abs=float(np.quantile(np.abs(d), 0.99)), max_abs=float(np.abs(d).max()),
                rounds={t: n for t, *_, n in spec})
    json.dump(summ, open(os.path.join(RES, "rank_delta_12m_summary.json"), "w"), indent=1)
    print(summ)
    print(R.group_by("ADEP_mvt").agg(pl.len(), (pl.col("delta") != 0).sum().alias("changed"), pl.col("delta").mean().alias("mean"),
                                     pl.col("delta").abs().quantile(0.99).alias("p99_abs"),
                                     ((pl.col("delta") ** 2).sum() / sub.height).alias("sq_per_N")).sort("ADEP_mvt"))


if __name__ == "__main__":
    cmd = ARGV[1]
    if cmd == "prep":
        cmd_prep()
    elif cmd == "weights":
        cmd_weights()
    elif cmd == "cv":
        cmd_cv(ARGV[2].split(",") if len(ARGV) > 2 else list(ARMS))
    elif cmd == "eval":
        cmd_eval()
    elif cmd == "apply12":
        cmd_apply12(ARGV[2], ARGV[3], ARGV[4])
