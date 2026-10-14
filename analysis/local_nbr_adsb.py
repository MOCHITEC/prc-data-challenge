"""nbradsb: observed ground activity of neighbouring departures (ADS-B only) in the learned ADS-B residual (local).

Pre-registration: $SCRATCH/agents/nbradsb/prereg.md (written before any fit). Reuses the harness of
`analysis/local_adsb_rescat.py` (V2 params, leave-days-out K=5 per month, inner early-stopping split, clip, LTFM c = 0,
day-block bootstrap, paired arms) on two settings: months7 (Jan, Jul-Dec 2025) and months12 (all of 2025).
Arms: A0 production features (`design(D,"b")`), A4 = A0 + NB_COLS below.

Neighbour features (label-free: only ADS-B event times, MVT times, RUNWAY_mvt, hour_local; never BLOCK / y):
  pool = ADS-B departures (learn/feat_*.parquet) at the same airport with a valid ADS-B taxi duration
         dur = t_mvt - t_move in [60, 3600] s (t_move = ADS-B stand exit / start of taxi); the row itself excluded.
  ref  = median 2025 dur per airport x runway x local hour (fallback airport x hour, airport; groups >= 30)
  windows on the neighbour's t_mvt relative to own T = t_mvt: b30 (T-30', T], f30 (T, T+30'], s15 [T-15', T+15']
    nb_{w}_n, nb_{w}_mean, nb_{w}_med, nb_{w}_exc (mean of dur - ref)
  nb_rw30_n, nb_rw30_exc     same runway, |t_j - T| <= 30'
  nb_taxi_mvt                # pool aircraft taxiing at T (t_move_j <= T < t_mvt_j)
  nb_taxi_start              same at own start = coalesce(t_push, t_move) (null when neither is known)
  nb_cov60                   (# pool departures within +-60') / (# all departures within +-60', movement data)
  nb_taxi_adj                nb_taxi_mvt / max(nb_cov60, 0.05)

    POLARS_MAX_THREADS=2 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage SCRATCH=... \
        .venv/bin/python analysis/local_nbr_adsb.py feat         # -> nbfeat.parquet (2025 + 2026 ADS-B rows)
    ... local_nbr_adsb.py drift                                   # coverage + single-feature adversarial AUC
    ... local_nbr_adsb.py cv m12|m7 [A0,A4]                       # resumable per-fold npz in folds_<setting>/
    ... local_nbr_adsb.py eval m12|m7                             # gains, CIs, concentration, decision
    ... local_nbr_adsb.py apply12 <in.parquet> yes|no             # 2026 delta (only if adopted)
Writes only under $SCRATCH/agents/nbradsb/.
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
RES = os.path.join(L.SCR, "agents", "nbradsb")
os.makedirs(RES, exist_ok=True)
PRM = {**CV.PARAMS, **dict(num_leaves=255, min_data_in_leaf=300, feature_fraction=0.7, num_threads=2)}
FEATS25 = ("feat_2025.parquet", "feat_2025novdec.parquet", "feat_2025aso.parquet", "feat_2025fmj.parquet")
MONTHS = {"m7": [1, 7, 8, 9, 10, 11, 12], "m12": list(range(1, 13))}
WIN = {"b30": (-1800.0, 0.0, False, True), "f30": (0.0, 1800.0, False, True), "s15": (-900.0, 900.0, True, True)}
NB_COLS = ([f"nb_{w}_{s}" for w in WIN for s in ("n", "mean", "med", "exc")]
           + ["nb_rw30_n", "nb_rw30_exc", "nb_taxi_mvt", "nb_taxi_start", "nb_cov60", "nb_taxi_adj"])
ARMS = {"A0": [], "A4": NB_COLS}
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(220)
pl.Config.set_tbl_cols(20)


def avail_mb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    return 1e9


def gate(need_mb=2500.0):
    while avail_mb() < need_mb:
        print(f"  waiting: MemAvailable {avail_mb():.0f} MB < {need_mb:.0f}", flush=True)
        time.sleep(60)


# ----------------------------------------------------------------------------------------------- features
def mvt_all() -> pl.DataFrame:
    out = []
    for src, yr in (("train", "2025"), ("rank", "2026")):
        out.append(pl.scan_parquet(f"data/work_flseq/feat_{src}.parquet").filter(pl.col("PHASE_mvt") == "DEP")
                   .select("MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt", "hour_local",
                           pl.col("MVT_TIME_UTC_mvt").dt.epoch("ms").cast(pl.Float64).truediv(1000).alias("t_all"))
                   .collect().with_columns(pl.lit(yr).alias("yr")))
    return pl.concat(out)


def cmd_feat():
    A = mvt_all()
    cols = ["MVT_ID_mvt", "t_mvt", "t_push", "t_move"]
    P = pl.concat([pl.concat([pl.read_parquet(os.path.join(OUT, f), columns=cols) for f in FEATS25]),
                   pl.read_parquet(os.path.join(OUT, "feat_2026.parquet"), columns=cols)])
    assert P["MVT_ID_mvt"].n_unique() == P.height
    P = P.join(A, on="MVT_ID_mvt", how="inner")
    print("ADS-B rows", P.height, "| max |t_mvt - MVT| ", float((P["t_mvt"] - P["t_all"]).abs().max()), flush=True)
    P = P.with_columns((pl.col("t_mvt") - pl.col("t_move")).alias("dur"))
    P = P.with_columns(((pl.col("dur") >= 60) & (pl.col("dur") <= 3600)).fill_null(False).alias("valid"))
    # 2025 reference medians (label-free)
    R = P.filter(pl.col("valid") & (pl.col("yr") == "2025"))
    r1 = R.group_by("ADEP_mvt", "RUNWAY_mvt", "hour_local").agg(pl.col("dur").median().alias("r1"), pl.len().alias("n1"))
    r2 = R.group_by("ADEP_mvt", "hour_local").agg(pl.col("dur").median().alias("r2"), pl.len().alias("n2"))
    r3 = R.group_by("ADEP_mvt").agg(pl.col("dur").median().alias("r3"))
    P = (P.join(r1, on=["ADEP_mvt", "RUNWAY_mvt", "hour_local"], how="left").join(r2, on=["ADEP_mvt", "hour_local"], how="left")
         .join(r3, on="ADEP_mvt", how="left"))
    P = P.with_columns(pl.when(pl.col("n1") >= 30).then("r1").when(pl.col("n2") >= 30).then("r2").otherwise("r3").alias("ref"))
    P = P.with_columns((pl.col("dur") - pl.col("ref")).alias("exc"))
    print("valid share by yr/airport:\n", P.group_by("yr", "ADEP_mvt").agg(pl.col("valid").mean().round(3)).sort("ADEP_mvt", "yr")
          .pivot(on="yr", index="ADEP_mvt", values="valid"), flush=True)
    outs = []
    for (yr, apt), G in P.group_by(["yr", "ADEP_mvt"], maintain_order=True):
        t0 = time.time()
        G = G.sort("t_mvt")
        n = G.height
        T = G["t_mvt"].to_numpy()
        val = G["valid"].to_numpy()
        dur = G["dur"].to_numpy()
        exc = G["exc"].to_numpy()
        rw = G["RUNWAY_mvt"].to_numpy()
        tv = T[val]  # sorted (subset of sorted)
        dv, ev = dur[val], exc[val]
        rwv = rw[val]
        cs_d = np.concatenate([[0.0], np.cumsum(dv)])
        cs_e = np.concatenate([[0.0], np.cumsum(ev)])
        res = {}
        selfv = val.astype(float)
        for w, (lo, hi, lo_incl, hi_incl) in WIN.items():
            a = np.searchsorted(tv, T + lo, side="left" if lo_incl else "right")
            b = np.searchsorted(tv, T + hi, side="right" if hi_incl else "left")
            # self in window? own T lies in (T-30', T] and [T-15', T+15'] but not in (T, T+30']
            self_in = selfv * (1.0 if w in ("b30", "s15") else 0.0)
            cnt = (b - a) - self_in
            sd = cs_d[b] - cs_d[a] - np.where(self_in > 0, np.nan_to_num(dur), 0.0)
            se = cs_e[b] - cs_e[a] - np.where(self_in > 0, np.nan_to_num(exc), 0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                res[f"nb_{w}_n"] = cnt
                res[f"nb_{w}_mean"] = np.where(cnt > 0, sd / np.maximum(cnt, 1), np.nan)
                res[f"nb_{w}_exc"] = np.where(cnt > 0, se / np.maximum(cnt, 1), np.nan)
            med = np.full(n, np.nan)
            # position of self among valid rows (for exclusion from the median)
            vpos = np.cumsum(val) - 1
            for i in range(n):
                if cnt[i] <= 0:
                    continue
                s = dv[a[i]:b[i]]
                if self_in[i] > 0:
                    k = vpos[i] - a[i]
                    s = np.concatenate([s[:k], s[k + 1:]])
                med[i] = np.median(s)
            res[f"nb_{w}_med"] = med
        # same runway +-30'
        rn, rx = np.zeros(n), np.full(n, np.nan)
        for r_ in np.unique(rw):
            m = rw == r_
            mv = rwv == r_
            tr, er = tv[mv], ev[mv]
            ce = np.concatenate([[0.0], np.cumsum(er)])
            a = np.searchsorted(tr, T[m] - 1800, side="left")
            b = np.searchsorted(tr, T[m] + 1800, side="right")
            si = val[m].astype(float)
            c = (b - a) - si
            s = ce[b] - ce[a] - np.where(si > 0, np.nan_to_num(exc[m]), 0.0)
            rn[m] = c
            with np.errstate(invalid="ignore", divide="ignore"):
                rx[m] = np.where(c > 0, s / np.maximum(c, 1), np.nan)
        res["nb_rw30_n"], res["nb_rw30_exc"] = rn, rx
        # aircraft taxiing (pool intervals [t_move, t_mvt))
        starts = np.sort(G["t_move"].to_numpy()[val])
        ends = np.sort(tv)

        def taxiing(t):
            return (np.searchsorted(starts, t, side="right") - np.searchsorted(ends, t, side="right")).astype(float)

        res["nb_taxi_mvt"] = taxiing(T)  # self: t_mvt_i <= T so not counted
        st = G.select(pl.coalesce("t_push", "t_move")).to_series().to_numpy().astype(float)
        own_move = G["t_move"].to_numpy()
        tst = taxiing(np.nan_to_num(st, nan=-1e18))
        # self counted iff valid and t_move_i <= st < t_mvt_i
        self_c = val & (own_move <= st) & (st < T)
        res["nb_taxi_start"] = np.where(np.isnan(st), np.nan, tst - self_c)
        # coverage within +-60'
        ta = np.sort(A.filter((pl.col("yr") == yr) & (pl.col("ADEP_mvt") == apt))["t_all"].to_numpy())
        nall = np.searchsorted(ta, T + 3600, side="right") - np.searchsorted(ta, T - 3600, side="left") - 1
        nval = np.searchsorted(tv, T + 3600, side="right") - np.searchsorted(tv, T - 3600, side="left") - val
        res["nb_cov60"] = np.where(nall > 0, nval / np.maximum(nall, 1), np.nan)
        res["nb_taxi_adj"] = res["nb_taxi_mvt"] / np.maximum(np.nan_to_num(res["nb_cov60"], nan=0.0), 0.05)
        outs.append(pl.DataFrame({"MVT_ID_mvt": G["MVT_ID_mvt"], **{k: res[k].astype(np.float64) for k in NB_COLS}})
                    .with_columns(pl.lit(yr).alias("yr"), pl.lit(apt).alias("ADEP_mvt")))
        print(f"{yr} {apt}: {n} rows, {int(val.sum())} valid, {time.time() - t0:.0f} s", flush=True)
    F = pl.concat(outs)
    F.write_parquet(os.path.join(RES, "nbfeat.parquet"))
    print(F.describe())


# ----------------------------------------------------------------------------------------------- drift
def cmd_drift():
    from sklearn.metrics import roc_auc_score
    F = pl.read_parquet(os.path.join(RES, "nbfeat.parquet"))
    out = {}
    cov = F.with_columns(((pl.col("nb_b30_n") > 0) | (pl.col("nb_f30_n") > 0)).alias("any30"),
                         (pl.col("nb_s15_n") > 0).alias("any15"))
    T = cov.group_by("yr", "ADEP_mvt").agg(pl.col("any30").mean().round(3), pl.col("any15").mean().round(3)).sort("ADEP_mvt", "yr")
    print("share of ADS-B rows with >= 1 neighbour observation (b30 or f30 / s15):\n", T.pivot(on="yr", index="ADEP_mvt", values=["any30", "any15"]))
    # share of ALL departures (ranking / 2025 training) with >= 1 neighbour observation
    A = mvt_all().select("MVT_ID_mvt", "yr")
    tot = A.join(cov.select("MVT_ID_mvt", "any30", "any15"), on="MVT_ID_mvt", how="left").with_columns(
        pl.col("any30").fill_null(False), pl.col("any15").fill_null(False))
    S = tot.group_by("yr").agg(pl.len(), pl.col("any30").mean(), pl.col("any15").mean()).sort("yr")
    print("all departures:", S.rows())
    S2 = cov.group_by("yr").agg(pl.len(), pl.col("any30").mean(), pl.col("any15").mean()).sort("yr")
    print("ADS-B rows:", S2.rows())
    out["coverage_all_departures"] = S.rows()
    out["coverage_adsb_rows"] = S2.rows()
    out["coverage_per_airport"] = T.rows()
    rng = np.random.default_rng(0)
    a25 = F.filter(pl.col("yr") == "2025")
    a25 = a25[rng.choice(a25.height, 400000, replace=False)]
    Z = pl.concat([a25, F.filter(pl.col("yr") == "2026")])
    yv = (Z["yr"] == "2026").to_numpy().astype(int)
    fold = rng.integers(0, 5, Z.height)
    auc = {}
    for c in NB_COLS + ["_adep"]:
        if c == "_adep":
            X = Z.select(pl.col("ADEP_mvt").rank("dense").cast(pl.Float64)).to_numpy()
        else:
            X = Z.select(c).to_numpy().astype(np.float64)
        p = np.zeros(Z.height)
        for k in range(5):
            m = lgb.train(dict(objective="binary", num_leaves=31, learning_rate=0.1, num_threads=2, verbose=-1, min_data_in_leaf=200),
                          lgb.Dataset(X[fold != k], yv[fold != k]), 100)
            p[fold == k] = m.predict(X[fold == k])
        auc[c] = round(float(roc_auc_score(yv, p)), 4)
        print(f"  adversarial AUC 2025 vs 2026 {c:16s} {auc[c]:.4f}{'  FLAG > 0.6' if auc[c] > 0.6 else ''}", flush=True)
    out["auc"] = auc
    json.dump(out, open(os.path.join(RES, "drift.json"), "w"), indent=1)


# ----------------------------------------------------------------------------------------------- data
def load(setting: str):
    keep = ["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "mvt_minus_aobt3", "b", "y", "day"]
    parts = [CV.load_labelled().select(keep), M.load_novdec().select(keep), M.load_novdec("aso", "aso").select(keep)]
    files = list(FEATS25[:3])
    if setting == "m12":
        parts.append(M.load_novdec("fmj", "fmj").select(keep))
        files.append(FEATS25[3])
    D = pl.concat(parts)
    need = sorted(set(["MVT_ID_mvt", "icao", "t_mvt", "t_aobt3", "t_push", "t_move", "t_move_lo"] + L.EVENTS + L.PLAIN
                      + [c for c in L.CATS if c != "ADEP_mvt"]))
    F = pl.concat([pl.read_parquet(os.path.join(OUT, f), columns=need) for f in files], how="diagonal_relaxed")
    D = D.join(F, on="MVT_ID_mvt", how="left").with_columns(pl.col("icao").is_not_null().alias("has"),
                                                           pl.col("day").str.slice(0, 7).alias("ym"))
    del F
    N = pl.read_parquet(os.path.join(RES, "nbfeat.parquet")).filter(pl.col("yr") == "2025").drop("yr", "ADEP_mvt")
    D = D.join(N, on="MVT_ID_mvt", how="left")
    days = sorted(D["day"].unique().to_list())
    months = sorted({d[:7] for d in days})
    assert len(months) == len(MONTHS[setting]), months
    rng = np.random.default_rng(CV.SEED)
    fold_of = {}
    for m in months:
        ds = sorted(d for d in days if d.startswith(m))
        for i, j in enumerate(rng.permutation(len(ds))):
            fold_of[ds[j]] = i % CV.K

    def inner(tr_days, seed):
        r = np.random.default_rng(seed)
        val = set()
        for m in months:
            ds = sorted(d for d in tr_days if d.startswith(m))
            val |= {ds[j] for i, j in enumerate(r.permutation(len(ds))) if i % 5 == 0}
        return val

    D = D.with_columns(pl.col("day").replace_strict(fold_of).alias("cvfold"),
                       pl.col("day").replace_strict({d: i for i, d in enumerate(days)}).alias("day_idx"))
    return D, days, fold_of, inner


def cmd_cv(setting: str, arms: list):
    FD = os.path.join(RES, f"folds_{setting}")
    os.makedirs(FD, exist_ok=True)
    D, days, fold_of, inner = load(setting)
    nrow = D.height
    D = D.filter("has")
    X, feats, cats = L.design(D, "b")
    allf = feats + NB_COLS
    Xn = X.select(allf).to_numpy().astype(np.float32)
    del X
    r = np.clip(D["y"].to_numpy() - D["b"].to_numpy(), -CV.CLIP_Y, CV.CLIP_Y)
    cvf, day_arr, ids = D["cvfold"].to_numpy(), D["day"].to_numpy(), D["MVT_ID_mvt"].to_numpy()
    print(f"{setting}: rows {nrow}, ADS-B {D.height}, Xn {Xn.nbytes / 2**20:.0f} MB, avail {avail_mb():.0f} MB", flush=True)
    del D
    for k in range(CV.K):
        val_days = inner([d for d in days if fold_of[d] != k], CV.SEED + 1 + k)
        is_te = cvf == k
        is_val = (cvf != k) & np.isin(day_arr, list(val_days))
        is_fit = (cvf != k) & ~is_val
        for arm in arms:
            dst = os.path.join(FD, f"f{k}_{arm}.npz")
            if os.path.exists(dst):
                continue
            gate()
            t0 = time.time()
            fn = feats + ARMS[arm]
            idx = [allf.index(f) for f in fn]
            dtr = lgb.Dataset(Xn[np.ix_(is_fit, idx)], r[is_fit], feature_name=fn, categorical_feature=cats, free_raw_data=True)
            dva = lgb.Dataset(Xn[np.ix_(is_val, idx)], r[is_val], reference=dtr)
            m = lgb.train(PRM, dtr, num_boost_round=5000, valid_sets=[dva], callbacks=[lgb.early_stopping(200, verbose=False)])
            c = np.clip(m.predict(Xn[np.ix_(is_te, idx)], num_iteration=m.best_iteration), -CV.CLIP_C, CV.CLIP_C)
            imp = m.feature_importance("gain", iteration=m.best_iteration)
            np.savez(dst, ids=ids[is_te], c=c, best=int(m.best_iteration), imp=imp, feats=np.array(fn))
            sh = float(imp[len(feats):].sum() / imp.sum()) if ARMS[arm] else 0.0
            print(f"{setting} fold {k} arm {arm}: fit {is_fit.sum()} es {is_val.sum()} test {is_te.sum()} best_iter {m.best_iteration} "
                  f"| new-col gain share {sh:.4f} | {time.time() - t0:.0f} s", flush=True)
            del dtr, dva, m


def cmd_eval(setting: str):
    FD = os.path.join(RES, f"folds_{setting}")
    D, days, fold_of, inner = load(setting)
    D = D.select("MVT_ID_mvt", "ADEP_mvt", "ym", "has", "y", "b", "day_idx")
    ids = D["MVT_ID_mvt"].to_numpy()
    pos = {v: i for i, v in enumerate(ids)}
    c = {a: np.zeros(D.height) for a in ARMS}
    best, imps = {a: [] for a in ARMS}, {}
    for a in ARMS:
        tot = None
        for k in range(CV.K):
            z = np.load(os.path.join(FD, f"f{k}_{a}.npz"))
            ii = np.array([pos[v] for v in z["ids"]])
            c[a][ii] = z["c"]
            best[a].append(int(z["best"]))
            tot = z["imp"] if tot is None else tot + z["imp"]
            fn = list(z["feats"])
        imps[a] = {f: float(v / tot.sum()) for f, v in zip(fn, tot)}
    apt, ym, has = D["ADEP_mvt"].to_numpy(), D["ym"].to_numpy(), D["has"].to_numpy()
    for a in ARMS:
        c[a][apt == "LTFM"] = 0.0
    yv, bv = D["y"].to_numpy(), D["b"].to_numpy()
    e0 = yv - bv
    sq = {a: (e0 - c[a]) ** 2 for a in ARMS}
    di, nd = D["day_idx"].to_numpy(), len(days)
    out = {"setting": setting, "best_iters": best, "final_rounds": {a: int(np.median(v)) for a, v in best.items()},
           "new_col_gain_share": sum(imps["A4"][f] for f in NB_COLS),
           "new_col_gain": {f: round(imps["A4"][f], 5) for f in NB_COLS}}
    print("median best iter", out["final_rounds"], "| new-col gain share A4", round(out["new_col_gain_share"], 4))
    print("A0 gain over base (all rows):", round(float((e0 ** 2 - sq["A0"]).mean()), 1))
    d = sq["A0"] - sq["A4"]
    noLI = apt != "LIRF"
    dx = np.where(noLI, d, 0.0)  # what is applied (LIRF never swapped); all rows stay in the denominator
    sels = [("all", np.ones(D.height, bool)), ("Jan", ym == "2025-01"), ("Jul", ym == "2025-07"),
            ("Jan+Jul", np.isin(ym, ["2025-01", "2025-07"])), ("Aug-Oct", np.isin(ym, ["2025-08", "2025-09", "2025-10"])),
            ("Nov-Dec", np.isin(ym, ["2025-11", "2025-12"]))]
    if setting == "m12":
        sels.append(("Feb-Jun", np.isin(ym, ["2025-%02d" % m for m in range(2, 7)])))
    for name, dd in (("all airports", d), ("ex LIRF", dx)):
        out[name] = {}
        print(f"\n=== A4-A0 [{name}]: d = (y-b-c_A0)^2 - (y-b-c_A4)^2 (s^2, > 0 = A4 better)")
        for lab, sel in sels:
            g, lo, hi = CV.boot(dd[sel], di[sel], nd)
            sqa = np.where(noLI, sq["A4"], sq["A0"]) if name == "ex LIRF" else sq["A4"]
            dr = float(np.sqrt(sqa[sel].mean()) - np.sqrt(sq["A0"][sel].mean()))
            out[name][lab] = dict(n=int(sel.sum()), gain=g, lo=lo, hi=hi, d_rmse=dr)
            print(f"  {lab:8s} n={sel.sum():8d} gain {g:7.1f} [{lo:7.1f}, {hi:7.1f}] | dRMSE {dr:+.3f} s")
        o = np.argsort(-np.abs(dd))
        share = float(1 - dd[o[50:]].sum() / dd.sum()) if dd.sum() != 0 else float("nan")
        out[name]["top50_share"] = share
        print(f"  top-50 |d| rows carry share {share:.3f} of the gain; gain without them {dd[o[50:]].sum() / D.height:.1f}")
    rows = []
    for ap in sorted(set(apt)):
        for lab, sel0 in (("all", np.ones(D.height, bool)), ("Jan", ym == "2025-01"), ("Jul", ym == "2025-07")):
            sel = sel0 & (apt == ap)
            g, lo, hi = CV.boot(d[sel], di[sel], nd)
            rows.append({"airport": ap, "m": lab, "n": int(sel.sum()), "gain": round(g, 1), "lo": round(lo, 1), "hi": round(hi, 1)})
    print(pl.DataFrame(rows).pivot(on="m", index="airport", values=["gain", "lo", "hi"]))
    out["per_airport"] = rows
    r = out["ex LIRF"]
    passed = bool(r["all"]["lo"] > 0 and r["Jan"]["gain"] > 0 and r["Jul"]["gain"] > 0 and r["top50_share"] <= 0.5)
    lf = [x for x in rows if x["airport"] == "LFPG" and x["m"] == "all"][0]
    out["passed"] = passed
    out["lfpg_gain_positive"] = bool(lf["gain"] > 0)
    print(f"\nDECISION ({setting}, pre-registered rule on ex-LIRF): pass = {passed}; LFPG gain {lf['gain']} [{lf['lo']}, {lf['hi']}]")
    print("new-col gain shares:", out["new_col_gain"])
    json.dump(out, open(os.path.join(RES, f"eval_{setting}.json"), "w"), indent=1)
    pl.DataFrame({"MVT_ID_mvt": ids, "ym": ym, "ADEP_mvt": apt, "y": yv, "b": bv, "has": has,
                  **{f"c_{a}": c[a] for a in ARMS}}).write_parquet(os.path.join(RES, f"cv_rows_{setting}.parquet"))


# ----------------------------------------------------------------------------------------------- apply
def cmd_apply12(src: str, with_lfpg: str):
    """Production-matched swap (as local_adsb_rescat.cmd_apply12): ground-fix rows, LIRF and LTFM excluded:
    12:v2 -> 12:v2 + NB_COLS (12 months, rounds = W.rounds('12:v2') for both); at LFPG (with_lfpg == 'yes')
    7:v2 -> 7:v2 + NB_COLS (rounds 7:v2 rule / median m7 CV best iteration of A4). b = file - c_old(b) by fixed point,
    new = file + c_new(b) - c_old(b). Writes rank_delta_12m.parquet + summary json."""
    import local_adsb_apply_swap as W
    D12 = W.data()
    N = pl.read_parquet(os.path.join(RES, "nbfeat.parquet")).filter(pl.col("yr") == "2025").drop("yr", "ADEP_mvt")
    D12 = D12.join(N, on="MVT_ID_mvt", how="left")
    del N
    ym = D12["day"].str.slice(0, 7).to_numpy()
    X, feats, cats = L.design(D12, "b")
    r = np.clip(D12["y"].to_numpy() - D12["b"].to_numpy(), -CV.CLIP_Y, CV.CLIP_Y)
    allf = feats + NB_COLS
    Xn = X.select(allf).to_numpy().astype(np.float32)
    del X, D12
    sel7 = np.isin(ym, ["2025-%02d" % m for m in MONTHS["m7"]])
    P = {**CV.PARAMS, **W.V2, "num_threads": 2}
    spec = [("old12", feats, None, W.rounds("12:v2")), ("new12", allf, None, W.rounds("12:v2"))]
    if with_lfpg == "yes":
        ev7 = json.load(open(os.path.join(RES, "eval_m7.json")))
        spec += [("old7", feats, sel7, W.rounds("7:v2")), ("new7", allf, sel7, ev7["final_rounds"]["A4"])]
    models = {}
    for tag, fn, sel, n in spec:
        gate()
        idx = [allf.index(f) for f in fn]
        Xa = Xn[:, idx] if sel is None else Xn[np.ix_(sel, idx)]
        ra = r if sel is None else r[sel]
        ds = lgb.Dataset(Xa, ra, feature_name=fn, categorical_feature=cats, free_raw_data=True)
        models[tag] = (lgb.train(P, ds, num_boost_round=n), fn)
        del Xa, ds
        print(f"{tag}: {len(ra)} rows, {n} rounds, avail {avail_mb():.0f} MB", flush=True)
    del Xn
    sub = pl.read_parquet(src)
    rk = pl.scan_parquet("data/raw/ranking.parquet").select("MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt").collect()
    F26 = pl.read_parquet(os.path.join(OUT, "feat_2026.parquet"))
    N26 = pl.read_parquet(os.path.join(RES, "nbfeat.parquet")).filter(pl.col("yr") == "2026").drop("yr", "ADEP_mvt")
    excl = ["LTFM", "LIRF"] + ([] if with_lfpg == "yes" else ["LFPG"])
    G = (F26.drop("ADEP_mvt").join(sub.rename({"TAXITIME_SEC_mvt": "file"}).join(rk, on="MVT_ID_mvt"), on="MVT_ID_mvt")
         .filter(~pl.col("ADEP_mvt").is_in(excl) & pl.col("ground_fix").fill_null(False)).with_columns(pl.col("file").cast(pl.Float64))
         .join(N26, on="MVT_ID_mvt", how="left"))
    lf = G["ADEP_mvt"].to_numpy() == "LFPG"

    def corr(tag, b):
        m, fn = models[tag]
        Xg, _, _ = L.design(G.with_columns(pl.Series("b", b)), "b")
        return np.clip(m.predict(Xg.select(fn).to_numpy().astype(np.float32)), -CV.CLIP_C, CV.CLIP_C)

    def c_old(b):
        return np.where(lf, corr("old7", b), corr("old12", b)) if lf.any() else corr("old12", b)

    def c_new(b):
        return np.where(lf, corr("new7", b), corr("new12", b)) if lf.any() else corr("new12", b)

    f = G["file"].to_numpy()
    b = f.copy()
    for _ in range(3):
        b = f - c_old(b)
    new_v = np.round(f + c_new(b) - c_old(b))
    new_v = np.where(new_v < 0, f, new_v)
    R = G.select("MVT_ID_mvt", "ADEP_mvt", "file").with_columns(pl.Series("new", new_v), (pl.Series("new", new_v) - pl.col("file")).alias("delta"))
    R.select("MVT_ID_mvt", "delta", "ADEP_mvt", "file", "new").write_parquet(os.path.join(RES, "rank_delta_12m.parquet"))
    d = R["delta"].to_numpy()
    summ = dict(input=src, excluded=excl, ground_rows=R.height, changed=int((d != 0).sum()),
                sum_d2_over_N=float((d ** 2).sum()) / sub.height, N=sub.height, mean=float(d.mean()),
                mean_abs=float(np.abs(d).mean()), p99_abs=float(np.quantile(np.abs(d), 0.99)), max_abs=float(np.abs(d).max()),
                rounds={t: n for t, _, _, n in spec})
    json.dump(summ, open(os.path.join(RES, "rank_delta_12m_summary.json"), "w"), indent=1)
    print(summ)
    print(R.group_by("ADEP_mvt").agg(pl.len(), (pl.col("delta") != 0).sum().alias("changed"), pl.col("delta").mean().alias("mean"),
                                     pl.col("delta").abs().quantile(0.99).alias("p99_abs"),
                                     ((pl.col("delta") ** 2).sum() / sub.height).alias("sq_per_N")).sort("ADEP_mvt"))


if __name__ == "__main__":
    cmd = ARGV[1]
    if cmd == "feat":
        cmd_feat()
    elif cmd == "drift":
        cmd_drift()
    elif cmd == "cv":
        cmd_cv(ARGV[2], ARGV[3].split(",") if len(ARGV) > 3 else list(ARMS))
    elif cmd == "eval":
        cmd_eval(ARGV[2])
    elif cmd == "apply12":
        cmd_apply12(ARGV[2], ARGV[3])
