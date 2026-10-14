"""LFPG / EGLL unmatched-lane (AOBT_3 null) departures: SSE anatomy, label-artefact search and LOMO-validated
residual corrections on the 7-month OOF (p = b + c_C), replicated on the 12-month OOF.  Analysis only (local).

    .venv/bin/python analysis/local_lfpg_unm.py prep        # $SCRATCH/agents/lfpg/{raw_ctx,dep}.parquet
    .venv/bin/python analysis/local_lfpg_unm.py nbr         # matched-neighbour MVT-AOBT3 stats per unmatched row
    .venv/bin/python analysis/local_lfpg_unm.py fit|robust  # global non-LIRF residual GBM, LOMO on 7m (looks good)
    .venv/bin/python analysis/local_lfpg_unm.py rep12|clipped  # ...but fails on the 12m OOF (LTFM Feb/Oct)
    .venv/bin/python analysis/local_lfpg_unm.py lfpg_only|lfpg_boot|lfpg_apply  # LFPG-own clipped correction
Run with POLARS_MAX_THREADS=2 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage and SCRATCH exported.
"""
import os
import sys

import numpy as np
import polars as pl

S = os.environ["SCRATCH"]
O = f"{S}/agents/lfpg"
AP = ["LFPG", "EGLL"]
RAW = ["MVT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
       "SCHED_TIME_UTC_mvt", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt", "TAXITIME_SEC_mvt", "LOBT_flt",
       "CALLSIGN_flt", "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt", "MARKET_SEGMENT_flt", "AIRCRAFT_OPERATOR_flt",
       "FLIGHT_TYPE_flt", "FLIGHT_RULE_mvt", "ADES_flt"]


def raw_all():
    fs = sorted(f"data/raw/{f}" for f in os.listdir("data/raw") if f.startswith("training_"))
    tr = pl.concat([pl.scan_parquet(f).select(RAW) for f in fs]).filter(
        pl.col("ADEP_mvt").is_in(AP) | pl.col("ADES_mvt").is_in(AP)).collect()
    rk = pl.scan_parquet("data/raw/ranking.parquet").select(RAW).filter(
        pl.col("ADEP_mvt").is_in(AP) | pl.col("ADES_mvt").is_in(AP)).collect()
    return pl.concat([tr.with_columns(pl.lit("tr").alias("src")), rk.with_columns(pl.lit("rk").alias("src"))])


def adsb():
    cols = ["MVT_ID_mvt", "ground_fix", "t_mvt", "t_move", "t_push", "push", "post_dwell", "n_ground", "stand_dur",
            "t_first", "t_dep1", "t_to", "t_air1", "seen", "join"]
    fs = ["feat_2025.parquet", "feat_2025aso.parquet", "feat_2025novdec.parquet", "feat_2025fmj.parquet", "feat_2026.parquet"]
    out = []
    for f in fs:
        p = f"{S}/learn/{f}"
        if os.path.exists(p):
            sc = pl.scan_parquet(p)
            have = sc.collect_schema().names()
            out.append(sc.select([c for c in cols if c in have]).collect().with_columns(pl.lit(f).alias("adsb_src")))
    a = pl.concat(out, how="diagonal_relaxed").unique("MVT_ID_mvt", keep="first")
    return a


def prep():
    R = raw_all()
    R.write_parquet(f"{O}/raw_ctx.parquet")
    cv = pl.read_parquet(f"{S}/learn/{cvf}")
    A = adsb()
    sub = pl.read_parquet("/home/user/prc-data-challenge/outputs/v40_v39_7m_pos_12m.parquet").rename(
        {"TAXITIME_SEC_mvt": "v40"})
    dep = R.filter(pl.col("PHASE_mvt") == "DEP").join(A, on="MVT_ID_mvt", how="left")
    dep = dep.join(cv.select("MVT_ID_mvt", "ym", "y", "b", "has", "c_C"), on="MVT_ID_mvt", how="left")
    dep = dep.join(sub, on="MVT_ID_mvt", how="left")
    dep.write_parquet(f"{O}/dep.parquet")
    print(dep.group_by("src", "ADEP_mvt", pl.col("AOBT_3_flt").is_null().alias("unm"), pl.col("ym").is_not_null().alias("oof")).len().sort("src", "ADEP_mvt"))



def nbr():
    """Neighbour stats from matched departures (MVT - AOBT_3, known in 2026 too) around each unmatched row's MVT,
    plus unmatched-neighbour counts.  All 10 airports, train + rank.  -> $O/nbr.parquet"""
    cols = ["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "mvt_minus_aobt3", "RUNWAY_mvt"]
    T = pl.concat([pl.scan_parquet("data/work_flseq/feat_train.parquet").filter(pl.col("PHASE_mvt") == "DEP").select(cols).collect(),
                   pl.scan_parquet("data/work_flseq/feat_rank.parquet").filter(pl.col("PHASE_mvt") == "DEP").select(cols).collect()])
    T = T.with_columns(t=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s"), ms=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds())
    out = []
    for ap, g in T.group_by("ADEP_mvt"):
        g = g.sort("t")
        m = g.filter(pl.col("mvt_minus_aobt3").is_not_null())
        tm, vm = m["t"].to_numpy(), m["mvt_minus_aobt3"].to_numpy()
        un = g.filter(pl.col("mvt_minus_aobt3").is_null())
        tu = un["t"].to_numpy()
        res = {"MVT_ID_mvt": un["MVT_ID_mvt"].to_numpy()}
        for W in (900, 1800, 3600):
            lo = np.searchsorted(tm, tu - W); hi = np.searchsorted(tm, tu + W, side="right")
            q50 = np.full(len(tu), np.nan); q90 = q50.copy(); mx = q50.copy(); f40 = q50.copy(); n = hi - lo
            for i in range(len(tu)):
                if n[i] > 0:
                    v = vm[lo[i]:hi[i]]
                    q50[i] = np.median(v); q90[i] = np.quantile(v, 0.9); mx[i] = v.max(); f40[i] = (v > 2400).mean()
            res.update({f"nm_q50_{W}": q50, f"nm_q90_{W}": q90, f"nm_max_{W}": mx, f"nm_f40_{W}": f40, f"nm_n_{W}": n})
            lo2 = np.searchsorted(tu, tu - W); hi2 = np.searchsorted(tu, tu + W, side="right")
            res[f"nu_n_{W}"] = hi2 - lo2 - 1
        out.append(pl.DataFrame(res))
    pl.concat(out).write_parquet(f"{O}/nbr.parquet")



XF = ["nbr_mvt_minus_aobt_mean_30m", "nbr_mvt_minus_aobt_mean_60m", "nbr_dep_n_30m", "congestion_vs_usual_30m",
      "arr_taxi_in_mean_30m", "arr_taxi_in_mean_60m", "bk_sched_backlog_60m", "mvt_minus_sched", "hour_local",
      "frozen_wx", "deicing_gate", "low_vis", "wind_ms", "precip_mm", "temp_c", "queue_ahead", "dep_tko_prev_30m",
      "dep_tko_next_30m", "dup_n", "stand_null"]


def frame(rank=False, cvf="cv_rows_months7_v2.parquet"):
    nb = pl.read_parquet(f"{O}/nbr.parquet")
    src = "feat_rank" if rank else "feat_train"
    F = pl.scan_parquet(f"data/work_flseq/{src}.parquet").filter(pl.col("mvt_minus_aobt3").is_null() & (pl.col("PHASE_mvt") == "DEP")).select(
        ["MVT_ID_mvt", "ADEP_mvt"] + XF).collect()
    F = F.join(nb, on="MVT_ID_mvt", how="inner")
    if rank:
        return F
    cv = pl.read_parquet(f"{S}/learn/{cvf}").with_columns(p=pl.col("b") + pl.col("c_C"))
    return cv.join(F.drop("ADEP_mvt"), on="MVT_ID_mvt", how="inner").with_columns(e=pl.col("y") - pl.col("p"))


def feats(df):
    nm = [c for c in df.columns if c.startswith(("nm_", "nu_"))]
    ap = df["ADEP_mvt"].replace_strict({a: i for i, a in enumerate(sorted(["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]))}, return_dtype=pl.Int32)
    X = df.select(XF + nm + ["p"]).with_columns(ap=ap)
    return X.to_numpy().astype(np.float64), X.columns


def fit():
    """LOMO residual LightGBM on unmatched rows (excl. LIRF), target e = y - p (Huber), shrink factor grid."""
    import lightgbm as lgb
    D = frame()
    D = D.filter((pl.col("ADEP_mvt") != "LIRF") & (pl.col("y") < 40000))  # 2 easyJet outliers out of training only
    X, names = feats(D)
    e = D["e"].to_numpy(); ym = D["ym"].to_numpy(); aps = D["ADEP_mvt"].to_numpy()
    P = dict(objective="huber", alpha=600.0, learning_rate=0.03, num_leaves=7, min_data_in_leaf=80, feature_fraction=0.7,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=50.0, verbose=-1, num_threads=2, seed=7)
    pred = {}
    for R in (100, 300):
        out = np.zeros(len(e))
        for m in np.unique(ym):
            tr = ym != m
            bst = lgb.train(P, lgb.Dataset(X[tr], e[tr]), num_boost_round=R)
            out[~tr] = bst.predict(X[~tr])
        pred[R] = out
    np.save(f"{O}/fit_pred.npy", np.stack([pred[100], pred[300]]))
    N = 1233234
    for R, out in pred.items():
        for A in (0.25, 0.5, 1.0):
            g = e ** 2 - (e - A * out) ** 2
            pm = {m: g[ym == m].sum() / N for m in np.unique(ym)}
            pa = {a: round(g[aps == a].sum() / N, 2) for a in ("LFPG", "EGLL", "EHAM", "LTFM", "LSZH")}
            top2 = np.sort(g)[::-1][:2].sum() / N
            print(f"R={R} A={A}: gain/row={g.sum()/N:+.2f}  top2={top2:+.2f}  months+={sum(v>0 for v in pm.values())}/7 ",
                  " ".join(f"{k[-2:]}:{v:+.1f}" for k, v in pm.items()), pa)



def robust():
    """Robustness of the LOMO residual GBM: seeds, objectives, top-k concentration, day-block bootstrap."""
    import lightgbm as lgb
    D = frame().filter((pl.col("ADEP_mvt") != "LIRF") & (pl.col("y") < 40000))
    X, names = feats(D)
    e = D["e"].to_numpy(); ym = D["ym"].to_numpy(); aps = D["ADEP_mvt"].to_numpy()
    day = D.join(pl.scan_parquet("data/work_flseq/feat_train.parquet").select("MVT_ID_mvt", pl.col("MVT_TIME_UTC_mvt").dt.date().alias("d")).collect(),
                 on="MVT_ID_mvt", how="left")["d"].cast(pl.String).to_numpy()
    N = 1233234
    base = dict(learning_rate=0.03, num_leaves=7, min_data_in_leaf=80, feature_fraction=0.7, bagging_fraction=0.8,
                bagging_freq=1, lambda_l2=50.0, verbose=-1, num_threads=2)
    res = {}
    for tag, obj in [("huber600", dict(objective="huber", alpha=600.0)), ("l2", dict(objective="regression")),
                     ("huber300", dict(objective="huber", alpha=300.0))]:
        for seed in (1, 2, 3):
            out = np.zeros(len(e))
            for m in np.unique(ym):
                tr = ym != m
                bst = lgb.train({**base, **obj, "seed": seed}, lgb.Dataset(X[tr], e[tr]), num_boost_round=300)
                out[~tr] = bst.predict(X[~tr])
            res[(tag, seed)] = out
    rng = np.random.default_rng(0)
    ud = np.unique(day); idx = {d: np.where(day == d)[0] for d in ud}
    for (tag, seed), out in res.items():
        for A in (0.5,):
            g = e ** 2 - (e - A * out) ** 2
            gs = np.sort(g)[::-1]
            pm = [round(g[ym == m].sum() / N, 1) for m in np.unique(ym)]
            pa = {a: round(g[aps == a].sum() / N, 1) for a in np.unique(aps)}
            gd = np.array([g[idx[d]].sum() for d in ud])
            bs = [gd[rng.integers(0, len(ud), len(ud))].sum() / N for _ in range(500)]
            print(f"{tag} s{seed} A={A}: gain={g.sum()/N:+.2f} top10={gs[:10].sum()/N:+.2f} bottom10={gs[-10:].sum()/N:+.2f} "
                  f"dayboot5%={np.percentile(bs,5):+.2f} months={pm} {pa}")
    avg = np.mean([v for k, v in res.items() if k[0] == "huber600"], axis=0)
    np.save(f"{O}/robust_avg_huber600.npy", avg)
    for A in (0.3, 0.5, 0.7):
        g = e ** 2 - (e - A * avg) ** 2
        print("avg3 huber600 A", A, round(g.sum() / N, 2), [round(g[ym == m].sum() / N, 1) for m in np.unique(ym)],
              "LFPG", round(g[aps == "LFPG"].sum() / N, 2), "EGLL", round(g[aps == "EGLL"].sum() / N, 2))



PAR = dict(objective="huber", alpha=600.0, learning_rate=0.03, num_leaves=7, min_data_in_leaf=80, feature_fraction=0.7,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=50.0, verbose=-1, num_threads=2)
SEL = (pl.col("ADEP_mvt") != "LIRF")


def rep12():
    """Replicate the LOMO gain on the 12-month OOF file (different base fold models)."""
    import lightgbm as lgb
    D = frame(cvf="cv_rows_months12_v2.parquet").filter(SEL & (pl.col("y") < 40000))
    N12 = pl.scan_parquet(f"{S}/learn/cv_rows_months12_v2.parquet").select(pl.len()).collect().item()
    X, _ = feats(D); e = D["e"].to_numpy(); ym = D["ym"].to_numpy(); aps = D["ADEP_mvt"].to_numpy()
    out = np.zeros(len(e))
    for seed in (1, 2, 3):
        for m in np.unique(ym):
            tr = ym != m
            out[~tr] += lgb.train({**PAR, "seed": seed}, lgb.Dataset(X[tr], e[tr]), 300).predict(X[~tr]) / 3
    for A in (0.3, 0.5):
        g = e ** 2 - (e - A * out) ** 2
        print(f"12m A={A}: n={len(e)} gain/row(N12)={g.sum()/N12:+.2f}", [f"{m[-2:]}:{g[ym==m].sum()/N12:+.1f}" for m in np.unique(ym)],
              {a: round(g[aps == a].sum() / N12, 1) for a in np.unique(aps)})


def apply_rank():
    """Train on all 7 OOF months (3 seeds), apply to 2026 unmatched non-LIRF rows with p := v40; describe change."""
    import lightgbm as lgb
    D = frame().filter(SEL & (pl.col("y") < 40000))
    X, _ = feats(D); e = D["e"].to_numpy()
    bs = [lgb.train({**PAR, "seed": s}, lgb.Dataset(X, e), 300) for s in (1, 2, 3)]
    R = frame(rank=True).filter(SEL)
    v40 = pl.read_parquet("/home/user/prc-data-challenge/outputs/v40_v39_7m_pos_12m.parquet").rename({"TAXITIME_SEC_mvt": "p"})
    R = R.join(v40, on="MVT_ID_mvt", how="inner")
    XR, _ = feats(R)
    f = np.mean([b.predict(XR) for b in bs], axis=0)
    d = 0.5 * f
    R = R.with_columns(delta=pl.Series(d))
    R.select("MVT_ID_mvt", "ADEP_mvt", "p", "delta").write_parquet(f"{O}/rank_delta.parquet")
    print("rank rows touched:", R.height, "of", v40.height)
    print(R.group_by("ADEP_mvt").agg(pl.len(), mean=pl.col("delta").mean(), sd=pl.col("delta").std(), q01=pl.col("delta").quantile(.01), q99=pl.col("delta").quantile(.99)).sort("ADEP_mvt"))
    print(R["delta"].describe())
    print("sum delta^2 / 344841 =", float((d ** 2).sum() / 344841))
    # same on OOF for comparison of scale
    fo = np.mean([b.predict(X) for b in bs], axis=0) * 0.5
    print("OOF in-sample delta sd", fo.std(), "rank delta sd", d.std())



def clipped():
    """Constrained variant: residual GBM correction clipped to +-C, scored LOMO on both the 7m and 12m OOF files."""
    import lightgbm as lgb
    for cvf, NN in (("cv_rows_months7_v2.parquet", 1233234), ("cv_rows_months12_v2.parquet", 2085047)):
        D = frame(cvf=cvf).filter(SEL & (pl.col("y") < 40000))
        X, _ = feats(D); e = D["e"].to_numpy(); ym = D["ym"].to_numpy(); aps = D["ADEP_mvt"].to_numpy()
        for obj in ("huber", "regression"):
            out = np.zeros(len(e))
            for m in np.unique(ym):
                tr = ym != m
                out[~tr] = lgb.train({**PAR, "objective": obj, "seed": 1}, lgb.Dataset(X[tr], np.clip(e[tr], -3000, 3000)), 300).predict(X[~tr])
            for C in (50, 100, 200):
                for A in (0.5, 1.0):
                    d = np.clip(A * out, -C, C)
                    g = e ** 2 - (e - d) ** 2
                    pm = [g[ym == m].sum() / NN for m in np.unique(ym)]
                    print(f"{cvf[8:16]} {obj[:5]} C={C} A={A}: gain/row={g.sum()/NN:+.2f} months+={sum(v>0 for v in pm)}/{len(pm)} "
                          f"min_month={min(pm):+.1f} LFPG={g[aps=='LFPG'].sum()/NN:+.2f} EGLL={g[aps=='EGLL'].sum()/NN:+.2f} "
                          f"LTFM={g[aps=='LTFM'].sum()/NN:+.2f}")



def lfpg_only():
    """Pre-specified target airports only: global (non-LIRF) residual GBM applied to LFPG / EGLL rows, vs an
    airport-own model.  Clipped +-C.  Per-month signs on both 7m and 12m OOF files, top-row concentration."""
    import lightgbm as lgb
    for cvf, NN in (("cv_rows_months7_v2.parquet", 1233234), ("cv_rows_months12_v2.parquet", 2085047)):
        D = frame(cvf=cvf).filter(SEL & (pl.col("y") < 40000))
        X, _ = feats(D); e = D["e"].to_numpy(); ym = D["ym"].to_numpy(); aps = D["ADEP_mvt"].to_numpy()
        for mode in ("global", "own"):
            for ap in ("LFPG", "EGLL"):
                out = np.zeros(len(e)); sel = aps == ap
                for m in np.unique(ym):
                    tr = (ym != m) & (sel if mode == "own" else True)
                    te = (ym == m) & sel
                    bst = lgb.train({**PAR, "objective": "regression", "seed": 1,
                                     "min_data_in_leaf": 40 if mode == "own" else 80}, lgb.Dataset(X[tr], np.clip(e[tr], -3000, 3000)), 300 if mode == "global" else 150)
                    out[te] = bst.predict(X[te])
                for C in (50, 100, 200):
                    d = np.clip(0.5 * out, -C, C) * sel
                    g = (e ** 2 - (e - d) ** 2)[sel]
                    gs = np.sort(g)
                    pm = [g[ym[sel] == m].sum() / NN for m in np.unique(ym)]
                    print(f"{cvf[8:16]} {mode:6s} {ap} C={C}: gain/row={g.sum()/NN:+.2f} top5={gs[-5:].sum()/NN:+.2f} bot5={gs[:5].sum()/NN:+.2f} "
                          f"months+={sum(v>0 for v in pm)}/{len(pm)} " + " ".join(f"{v:+.1f}" for v in pm))



def lfpg_apply():
    """LFPG-own residual model (L2 on e clipped +-3000, 150 rounds, 0.5 shrink, clip +-100): OOF delta calibration,
    importances, and the 2026 rank deltas on v40 (trained on 7m and on 12m OOF)."""
    import lightgbm as lgb
    P2 = {**PAR, "objective": "regression", "seed": 1, "min_data_in_leaf": 40}
    R = frame(rank=True).filter(pl.col("ADEP_mvt") == "LFPG")
    v40 = pl.read_parquet("/home/user/prc-data-challenge/outputs/v40_v39_7m_pos_12m.parquet").rename({"TAXITIME_SEC_mvt": "p"})
    R = R.join(v40, on="MVT_ID_mvt", how="inner"); XR, names = feats(R)
    res = {}
    for cvf in ("cv_rows_months7_v2.parquet", "cv_rows_months12_v2.parquet"):
        D = frame(cvf=cvf).filter((pl.col("ADEP_mvt") == "LFPG") & (pl.col("y") < 40000))
        X, _ = feats(D); e = D["e"].to_numpy(); ym = D["ym"].to_numpy()
        out = np.zeros(len(e))
        for m in np.unique(ym):
            tr = ym != m
            out[~tr] = lgb.train(P2, lgb.Dataset(X[tr], np.clip(e[tr], -3000, 3000)), 150).predict(X[~tr])
        d = np.clip(0.5 * out, -100, 100)
        q = pl.DataFrame({"d": d, "e": e}).with_columns(b=pl.col("d").qcut(5, allow_duplicates=True))
        print(cvf, q.group_by("b").agg(pl.len(), dm=pl.col("d").mean(), em=pl.col("e").mean(), emed=pl.col("e").median()).sort("b"))
        bst = lgb.train(P2, lgb.Dataset(X, np.clip(e, -3000, 3000)), 150)
        print(" importance:", [(n, int(g / 1e6)) for g, n in sorted(zip(bst.feature_importance("gain"), names), reverse=True)[:8]])
        res[cvf] = np.clip(0.5 * bst.predict(XR), -100, 100)
    a, b = res.values()
    print("rank LFPG unmatched rows:", len(a), "corr(7m,12m deltas)=", np.corrcoef(a, b)[0, 1])
    for k, v in res.items():
        print(k, "delta mean %.1f sd %.1f q05 %.1f q50 %.1f q95 %.1f |d|>=99: %d" % (v.mean(), v.std(), *np.percentile(v, [5, 50, 95]), (np.abs(v) >= 99).sum()))
    R.select("MVT_ID_mvt", "p").with_columns(d7=pl.Series(a), d12=pl.Series(b)).write_parquet(f"{O}/rank_delta_lfpg.parquet")



def lfpg_boot():
    """LFPG-own correction: hyper-parameter sensitivity + day-block bootstrap 5th pct of the LOMO gain (7m & 12m)."""
    import lightgbm as lgb
    days = pl.scan_parquet("data/work_flseq/feat_train.parquet").select("MVT_ID_mvt", pl.col("MVT_TIME_UTC_mvt").dt.date().cast(pl.String).alias("d")).collect()
    rng = np.random.default_rng(0)
    for cvf, NN in (("cv_rows_months7_v2.parquet", 1233234), ("cv_rows_months12_v2.parquet", 2085047)):
        D = frame(cvf=cvf).filter((pl.col("ADEP_mvt") == "LFPG") & (pl.col("y") < 40000)).join(days, on="MVT_ID_mvt", how="left")
        X, _ = feats(D); e = D["e"].to_numpy(); ym = D["ym"].to_numpy(); day = D["d"].to_numpy()
        ud, inv = np.unique(day, return_inverse=True)
        for R, nl, mdl in ((150, 7, 40), (75, 7, 40), (300, 7, 40), (150, 15, 40), (150, 7, 100)):
            out = np.zeros(len(e))
            for m in np.unique(ym):
                tr = ym != m
                out[~tr] = lgb.train({**PAR, "objective": "regression", "seed": 2, "num_leaves": nl, "min_data_in_leaf": mdl},
                                     lgb.Dataset(X[tr], np.clip(e[tr], -3000, 3000)), R).predict(X[~tr])
            d = np.clip(0.5 * out, -100, 100)
            g = e ** 2 - (e - d) ** 2
            gd = np.bincount(inv, weights=g)
            bs = [gd[rng.integers(0, len(ud), len(ud))].sum() / NN for _ in range(1000)]
            pm = [g[ym == m].sum() / NN for m in np.unique(ym)]
            print(f"{cvf[8:16]} R={R} leaves={nl} mdl={mdl}: gain/row={g.sum()/NN:+.2f} boot5%={np.percentile(bs,5):+.2f} "
                  f"months+={sum(v>0 for v in pm)}/{len(pm)} days={len(ud)}")


if __name__ == "__main__":
    globals()[sys.argv[1]]()
