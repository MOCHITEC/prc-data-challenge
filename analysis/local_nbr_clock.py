"""Neighbour "clock" residual features for MATCHED departures (AOBT_3 present): excess MVT-AOBT_3 of other
departures vs a robust 2025 reference (airport x runway x local-hour median), in backward / forward / symmetric
windows around the take-off, per runway, around the own AOBT_3 ("following" pushbacks), plus gate-retention
(past EOBT_1, not yet AOBT_3) and departures-taxiing counts at own AOBT_3.  Only fields visible in 2026 are used.
A small, heavily regularised residual LightGBM on e = y - p (p = b + c_C) is validated leave-one-month-out on the
7-month and 12-month OOF files (LTFM excluded, EGLL optional).  Analysis only (local).

    .venv/bin/python analysis/local_nbr_clock.py feat              # $SCRATCH/agents/nbr/nbr_feat.parquet
    .venv/bin/python analysis/local_nbr_clock.py fit <var> <excl> [rounds]  # LOMO on 7m and 12m, per-month gains + day boot
    .venv/bin/python analysis/local_nbr_clock.py shift             # 2026 vs 2025 feature distribution
    NBR_OBJ=l2 .venv/bin/python analysis/local_nbr_clock.py apply new L 0.5 150  # (chosen config, gated) -> $SCRATCH/agents/nbr/rank_delta.parquet
Env NBR_OBJ = huber300 | huber900 | l2 (residual objective).  Run with POLARS_MAX_THREADS=2 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage and SCRATCH exported.
"""
import os
import sys

import numpy as np
import polars as pl

S = os.environ["SCRATCH"]
O = f"{S}/agents/nbr"
os.makedirs(O, exist_ok=True)
NN = {"7": 1233234, "12": 2085047}
CVF = {"7": "cv_rows_months7_v2.parquet", "12": "cv_rows_months12_v2.parquet"}
APS = sorted(["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"])
V40 = "/home/user/prc-data-challenge/outputs/v40_v39_7m_pos_12m.parquet"


def load_dep():
    cols = ["MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt", "EOBT_1_flt", "hour_local"]
    out = []
    for src in ("train", "rank"):
        out.append(pl.scan_parquet(f"data/work_flseq/feat_{src}.parquet").filter(pl.col("PHASE_mvt") == "DEP")
                   .select(cols).collect().with_columns(pl.lit(src).alias("src")))
    D = pl.concat(out)
    return D.with_columns(
        t=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64),
        a=pl.col("AOBT_3_flt").dt.epoch("s").cast(pl.Float64),
        eo=pl.col("EOBT_1_flt").dt.epoch("s").cast(pl.Float64),
        d=pl.col("MVT_TIME_UTC_mvt").dt.date(),
    ).with_columns(v=(pl.col("t") - pl.col("a")))


def _wmean(tq, ts, cs, lo_off, hi_off, lo_closed=True, hi_closed=True):
    """mean/count of values (cumsum cs over sorted ts) with ts in [tq+lo_off, tq+hi_off] (closure flags)."""
    lo = np.searchsorted(ts, tq + lo_off, side="left" if lo_closed else "right")
    hi = np.searchsorted(ts, tq + hi_off, side="right" if hi_closed else "left")
    n = hi - lo
    s = cs[hi] - cs[lo]
    return s, n


def feat():
    D = load_dep()
    # robust reference from 2025 only: airport x runway x local hour median of MVT - AOBT_3 (fallback airport x hour)
    tr = D.filter((pl.col("src") == "train") & pl.col("v").is_between(0, 7200))
    r1 = tr.group_by("ADEP_mvt", "RUNWAY_mvt", "hour_local").agg(ref1=pl.col("v").median(), n1=pl.len())
    r2 = tr.group_by("ADEP_mvt", "hour_local").agg(ref2=pl.col("v").median())
    D = D.join(r1, on=["ADEP_mvt", "RUNWAY_mvt", "hour_local"], how="left").join(r2, on=["ADEP_mvt", "hour_local"], how="left")
    D = D.with_columns(ref=pl.when(pl.col("n1") >= 50).then("ref1").otherwise("ref2"))
    D = D.with_columns(x=(pl.col("v") - pl.col("ref")).clip(-1800, 1800))
    D = D.with_columns(x=pl.when(pl.col("v").is_between(-600, 10800)).then("x").otherwise(None))
    res = []
    for (ap,), g in D.group_by(["ADEP_mvt"]):
        g = g.sort("t")
        t = g["t"].to_numpy(); a = g["a"].to_numpy(); eo = g["eo"].to_numpy(); x = g["x"].to_numpy()
        x_own = np.nan_to_num(x, nan=0.0); m_own = ~np.isnan(x)
        F = {"MVT_ID_mvt": g["MVT_ID_mvt"].to_numpy()}
        mk = ~np.isnan(x)
        tm, xm = t[mk], x[mk]
        cs = np.concatenate([[0.0], np.cumsum(xm)])
        hi600 = np.concatenate([[0.0], np.cumsum((xm > 600).astype(float))])
        # backward / forward / symmetric windows around own take-off (self excluded)
        for nm, lo, hi_, lc, hc in [("b30", -1800, 0, True, False), ("f15", 0, 900, False, True), ("f30", 0, 1800, False, True),
                                    ("s30", -1800, 1800, True, True), ("s60", -3600, 3600, True, True)]:
            s, n = _wmean(t, tm, cs, lo, hi_, lc, hc)
            if nm.startswith("s"):
                s = s - x_own * m_own; n = n - m_own
            F[f"nx_{nm}"] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
            F[f"nn_{nm}"] = n.astype(float)
            if nm in ("f30", "s30"):
                h, _ = _wmean(t, tm, hi600, lo, hi_, lc, hc)
                if nm.startswith("s"):
                    h = h - (x_own > 600) * m_own
                F[f"nhi_{nm}"] = np.where(n > 0, h / np.maximum(n, 1), np.nan)
        # per runway, symmetric 30 and forward 15
        rw = g["RUNWAY_mvt"].fill_null("?").to_numpy()
        for k in ("rx_s30", "rx_f15", "rn_s30"):
            F[k] = np.full(len(t), np.nan)
        for r in np.unique(rw):
            sel = rw == r
            ms = sel & mk
            trr, xr = t[ms], x[ms]
            csr = np.concatenate([[0.0], np.cumsum(xr)])
            s, n = _wmean(t[sel], trr, csr, -1800, 1800)
            s = s - (x_own * m_own)[sel]; n = n - m_own[sel]
            F["rx_s30"][sel] = np.where(n > 0, s / np.maximum(n, 1), np.nan); F["rn_s30"][sel] = n
            s, n = _wmean(t[sel], trr, csr, 0, 900, False, True)
            F["rx_f15"][sel] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
        # around own AOBT_3: pushbacks ordered by AOBT_3
        ma = mk & ~np.isnan(a)
        o = np.argsort(a[ma], kind="stable")
        aa, xa = a[ma][o], x[ma][o]
        csa = np.concatenate([[0.0], np.cumsum(xa)])
        aq = np.where(np.isnan(a), -1e18, a)
        s, n = _wmean(aq, aa, csa, -900, 900)
        s = s - x_own * m_own; n = n - m_own
        F["ax_p15"] = np.where((n > 0) & ~np.isnan(a), s / np.maximum(n, 1), np.nan)
        k = np.searchsorted(aa, aq, side="right")  # first pushback strictly after own AOBT
        hi5 = np.minimum(k + 5, len(aa)); n5 = hi5 - k
        F["ax_next5"] = np.where((n5 > 0) & ~np.isnan(a), (csa[hi5] - csa[k]) / np.maximum(n5, 1), np.nan)
        k0 = np.searchsorted(aa, aq, side="left"); lo5 = np.maximum(k0 - 5, 0); m5 = k0 - lo5
        F["ax_prev5"] = np.where((m5 > 0) & ~np.isnan(a), (csa[k0] - csa[lo5]) / np.maximum(m5, 1), np.nan)
        # gate retention at own AOBT_3: departures with EOBT_1 <= a < AOBT_3 ; taxiing: AOBT_3 <= a < MVT
        def stab(st, en, q):
            ok = ~np.isnan(st) & ~np.isnan(en) & (en > st)
            ss, ee = np.sort(st[ok]), np.sort(en[ok])
            return (np.searchsorted(ss, q, side="right") - np.searchsorted(ee, q, side="right")).astype(float)
        ret = stab(eo, a, aq); tax = stab(a, t, aq)
        F["ret_at_a"] = np.where(np.isnan(a), np.nan, ret)
        F["tax_at_a"] = np.where(np.isnan(a), np.nan, tax - (~np.isnan(a)))  # minus self
        # overdue minutes of retained flights is costly; use retention at own EOBT_1 too (scheduled-time view)
        eq = np.where(np.isnan(eo), -1e18, eo)
        F["ret_at_eo"] = np.where(np.isnan(eo), np.nan, stab(eo, a, eq))
        F["own_x"] = x
        res.append(pl.DataFrame(F))
    R = pl.concat(res, how="diagonal")
    R = R.with_columns([pl.col(c).cast(pl.Float64).fill_nan(None) for c in R.columns if c != "MVT_ID_mvt"])
    R = R.with_columns(own_rel_s30=pl.col("own_x") - pl.col("nx_s30"), own_rel_f15=pl.col("own_x") - pl.col("nx_f15"))
    R = R.join(D.select("MVT_ID_mvt", "ADEP_mvt", "src", "d", "hour_local", "ref"), on="MVT_ID_mvt", how="left")
    R.write_parquet(f"{O}/nbr_feat.parquet")
    print(R.shape)
    print(R.group_by("src").agg([pl.col(c).mean().round(1) for c in R.columns if c.startswith(("nx_", "rx_", "ax_", "ret", "tax", "own"))]))


NEW = ["nx_b30", "nx_f15", "nx_f30", "nx_s30", "nx_s60", "nn_f15", "nn_f30", "nn_s30", "nn_s60", "nhi_f30", "nhi_s30",
       "rx_s30", "rx_f15", "rn_s30", "ax_p15", "ax_next5", "ax_prev5", "ret_at_a", "tax_at_a", "ret_at_eo", "own_x",
       "own_rel_s30", "own_rel_f15"]
CTX = ["mvt_minus_aobt3", "nbr_mvt_minus_aobt_mean_30m", "hour_local_b", "has_i"]
WX = ["temp_c", "deicing_gate", "frozen_wx", "precip_mm", "vis_km"]


def frame(cvk, excl=""):
    NF = pl.read_parquet(f"{O}/nbr_feat.parquet").filter(pl.col("src") == "train").drop("src")
    base = pl.scan_parquet("data/work_flseq/feat_train.parquet").filter(pl.col("PHASE_mvt") == "DEP").select(
        "MVT_ID_mvt", "mvt_minus_aobt3", "nbr_mvt_minus_aobt_mean_30m", *WX).collect()
    cv = pl.read_parquet(f"{S}/learn/{CVF[cvk]}").with_columns(p=pl.col("b") + pl.col("c_C"))
    D = cv.join(NF.drop("ADEP_mvt"), on="MVT_ID_mvt", how="inner").join(base, on="MVT_ID_mvt", how="left")
    D = D.filter(pl.col("mvt_minus_aobt3").is_not_null() & (pl.col("ADEP_mvt") != "LTFM"))
    D = D.filter(~pl.col("ADEP_mvt").is_in(exclude(excl)))
    return D.with_columns(e=(pl.col("y") - pl.col("p")), hour_local_b=pl.col("hour_local"), has_i=pl.col("has").cast(pl.Int8))


def exclude(excl):
    """excl code letters: L = LIRF, G = EGLL, F = LFPG, E = EHAM (LTFM is always excluded)."""
    return [{"L": "LIRF", "G": "EGLL", "F": "LFPG", "E": "EHAM"}[c] for c in excl]


def Xmat(D, cols):
    ap = D["ADEP_mvt"].replace_strict({a: i for i, a in enumerate(APS)}, return_dtype=pl.Int32)
    return D.select(cols).with_columns(ap=ap).to_numpy().astype(np.float64)


VAR = {"new": NEW, "new_ctx": NEW + CTX, "new_wx": NEW + WX, "fwd": ["nx_f15", "nx_f30", "nn_f15", "nn_f30", "nhi_f30", "rx_f15", "ax_next5",
                                                  "ax_p15", "own_x", "own_rel_f15", "ret_at_a", "tax_at_a"],
       "cnt": ["ret_at_a", "tax_at_a", "ret_at_eo", "nn_f15", "nn_f30", "nn_s30", "nn_s60", "rn_s30"],
       "clock": ["nx_b30", "nx_f15", "nx_f30", "nx_s30", "nx_s60", "nhi_f30", "nhi_s30", "rx_s30", "rx_f15", "ax_p15",
                 "ax_next5", "ax_prev5", "own_x", "own_rel_s30", "own_rel_f15"]}
PAR = dict(objective="huber", alpha=300.0, learning_rate=0.03, num_leaves=7, min_data_in_leaf=1000, feature_fraction=0.7,
           bagging_fraction=0.7, bagging_freq=1, lambda_l2=200.0, verbose=-1, num_threads=2, seed=1, max_bin=63)
OBJ = os.environ.get("NBR_OBJ", "huber300")  # huber300 (median-ish: biased for RMSE) | huber900 | l2
PAR.update({"huber300": {}, "huber900": {"alpha": 900.0}, "l2": {"objective": "regression"}}[OBJ])


def lomo(D, cols, rounds=200):
    import lightgbm as lgb
    X = Xmat(D, cols); e = np.clip(D["e"].to_numpy(), -1800, 1800); ym = D["ym"].to_numpy()
    out = np.zeros(len(e))
    for m in np.unique(ym):
        tr = ym != m
        out[~tr] = lgb.train(PAR, lgb.Dataset(X[tr], e[tr], categorical_feature=[X.shape[1] - 1]), rounds).predict(X[~tr])
    return out


def report(D, out, NNk, tag, rng_seed=0):
    e = D["e"].to_numpy(); ym = D["ym"].to_numpy(); aps = D["ADEP_mvt"].to_numpy(); day = D["d"].cast(pl.String).to_numpy()
    ud, inv = np.unique(day, return_inverse=True)
    rng = np.random.default_rng(rng_seed)
    for A, C in ((0.3, 150), (0.5, 150), (0.7, 150), (1.0, 150)):
        dl = np.clip(A * out, -C, C)
        g = e ** 2 - (e - dl) ** 2
        pm = [g[ym == m].sum() / NNk for m in np.unique(ym)]
        gd = np.bincount(inv, weights=g)
        bs = [gd[rng.integers(0, len(ud), len(ud))].sum() / NNk for _ in range(400)]
        top = np.sort(gd)[::-1]
        print(f"  {tag} A={A} C={C}: gain/row={g.sum()/NNk:+.2f} boot5%={np.percentile(bs,5):+.2f} "
              f"months+={sum(v>0 for v in pm)}/{len(pm)} top5days={top[:5].sum()/NNk:+.2f} minus_top5={(g.sum()-top[:5].sum())/NNk:+.2f} | "
              + " ".join(f"{m[-2:]}:{v:+.1f}" for m, v in zip(np.unique(ym), pm)))
        print("     by ap:", {a: round(g[aps == a].sum() / NNk, 2) for a in np.unique(aps)})


def fit():
    var = sys.argv[2] if len(sys.argv) > 2 else "new"
    excl = sys.argv[3] if len(sys.argv) > 3 else ""
    rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    for cvk in ("7", "12"):
        D = frame(cvk, excl)
        out = lomo(D, VAR[var], rounds)
        np.save(f"{O}/lomo_{var}_{cvk}_x{excl}_r{rounds}_{OBJ}.npy", out)
        print(f"== {OBJ} {var} cv{cvk} excl={exclude(excl)} rounds={rounds} n={D.height} sd(pred)={out.std():.1f}", flush=True)
        report(D, out, NN[cvk], f"cv{cvk}")


def shift():
    NF = pl.read_parquet(f"{O}/nbr_feat.parquet").with_columns(mo=pl.col("d").dt.month(), yr=pl.col("d").dt.year())
    NF = NF.filter(pl.col("mo").is_in([1, 7]) & pl.col("own_x").is_not_null())
    cols = ["nx_s30", "nx_f15", "rx_s30", "ax_next5", "ret_at_a", "tax_at_a", "own_x", "own_rel_s30", "nn_s30"]
    print(NF.group_by("yr", "mo").agg([pl.col(c).mean().round(1) for c in cols]).sort("mo", "yr"))
    print(NF.group_by("ADEP_mvt", "yr", "mo").agg([pl.col(c).mean().round(1) for c in ["nx_s30", "own_x", "ret_at_a", "tax_at_a"]]
          + [pl.col("nx_s30").quantile(0.95).alias("nx_s30_q95")]).sort("ADEP_mvt", "mo", "yr"))


def gate_expr():
    """Switch the correction off in the de-icing / disrupted regime where the LOMO fit failed (Jan 2025):
    temp < 3 C or de-icing flag or mean neighbour excess (+-30 min) above 300 s."""
    return ((pl.col("temp_c").fill_null(99) < 3) | (pl.col("deicing_gate").fill_null(0) > 0)
            | (pl.col("nx_s30").fill_null(0) > 300))


def apply_rank():
    import lightgbm as lgb
    var = sys.argv[2] if len(sys.argv) > 2 else "new"
    excl = sys.argv[3] if len(sys.argv) > 3 else ""
    A, C = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5, float(sys.argv[5]) if len(sys.argv) > 5 else 100
    bs = []
    for cvk in ("7", "12"):  # average of models trained on each OOF's residuals (2 seeds each)
        D = frame(cvk, excl)
        X = Xmat(D, VAR[var]); e = np.clip(D["e"].to_numpy(), -1800, 1800)
        bs += [lgb.train({**PAR, "seed": s}, lgb.Dataset(X, e, categorical_feature=[X.shape[1] - 1]), 200) for s in (1, 2)]
    NF = pl.read_parquet(f"{O}/nbr_feat.parquet").filter(pl.col("src") == "rank").drop("src")
    base = pl.scan_parquet("data/work_flseq/feat_rank.parquet").filter(pl.col("PHASE_mvt") == "DEP").select(
        "MVT_ID_mvt", "mvt_minus_aobt3", "nbr_mvt_minus_aobt_mean_30m", *WX).collect()
    v40 = pl.read_parquet(V40)
    R = v40.join(NF, on="MVT_ID_mvt", how="inner").join(base, on="MVT_ID_mvt", how="left").filter(
        pl.col("mvt_minus_aobt3").is_not_null() & (pl.col("ADEP_mvt") != "LTFM"))
    R = R.filter(~pl.col("ADEP_mvt").is_in(exclude(excl)))
    R = R.with_columns(hour_local_b=pl.col("hour_local"), has_i=pl.lit(1, pl.Int8))
    f = np.mean([b.predict(Xmat(R, VAR[var])) for b in bs], axis=0)
    R = R.with_columns(gate=gate_expr())
    R = R.with_columns(delta=pl.when(pl.col("gate")).then(0.0).otherwise(pl.Series(np.clip(A * f, -C, C))))
    print("gated share by airport x month:", R.with_columns(mo=pl.col("d").dt.month()).group_by("ADEP_mvt", "mo").agg(
        pl.col("gate").mean().round(3)).sort("ADEP_mvt", "mo").pivot(on="mo", index="ADEP_mvt", values="gate"))
    out = v40.select("MVT_ID_mvt").join(R.select("MVT_ID_mvt", "delta"), on="MVT_ID_mvt", how="left").with_columns(pl.col("delta").fill_null(0.0))
    out.write_parquet(f"{O}/rank_delta.parquet")
    fo = np.clip(A * np.mean([b.predict(X) for b in bs], axis=0), -C, C)
    print("rows touched (|delta|>0.5):", int((out["delta"].abs() > 0.5).sum()), "of", out.height,
          "mean|delta| all:", round(float(out["delta"].abs().mean()), 2), "sum d^2/N:", round(float((out["delta"] ** 2).sum() / out.height), 2))
    print("OOF in-sample mean|delta|", round(float(np.abs(fo).mean()), 2), "rank mean|delta| (touched set)", round(float(np.abs(R["delta"]).mean()), 2))
    print(R.with_columns(mo=pl.col("d").dt.month()).group_by("ADEP_mvt", "mo").agg(pl.len(), mean=pl.col("delta").mean().round(2),
          mabs=pl.col("delta").abs().mean().round(2), q01=pl.col("delta").quantile(.01).round(1), q99=pl.col("delta").quantile(.99).round(1)).sort("ADEP_mvt", "mo"))


if __name__ == "__main__":
    {"feat": feat, "fit": fit, "shift": shift, "apply": apply_rank}[sys.argv[1]]()
