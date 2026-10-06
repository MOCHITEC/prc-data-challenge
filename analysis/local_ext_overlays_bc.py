"""External ideas B and C as post-hoc overlays on the 7-month OOF (p = b + c_C), leave-one-month-out (local).

Baseline = production overlays on the OOF: LIRF matched copy blend p + 0.4 q (MVT-SCHED - p) with q from a binary
LightGBM (local_lirf_copy_overlay.py parameters, fixed 109 rounds = production 99 x 1.1) fitted on the 11 2025 months
other than the row's month, then the LOBT identity clip +-3606.

C  isotonic calibration of q (kind-mango): map fitted on the LOMO q / copy labels of the 11 other months.
     C_iso1  : p + q_iso (s - p)              (expectation mixture, A = 1)
     C_isoA  : p + A q_iso (s - p), A fitted on the 6 other OOF months
     C_raw1  : p + q (s - p)                  (raw q, A = 1; control)
     C_rawA  : p + A q (s - p), A fitted on the 6 other OOF months (retune control)
B  NM AOBT_3 looks invalid:
     B_Z     : Phoenix v11 rule: AOBT_3 - LOBT > 7200, both proxies in [-7200, 172800], ADEP_flt == ADEP_mvt,
               not (LIRF and IOBT null) -> max(0, MVT - LOBT)
     B_Zw    : same trigger, p + w (MVT-LOBT - p), w fitted on the other 6 OOF months
     B_X<k>  : LOBT-residual expert on the gate |AOBT_3 - LOBT| > k s: LightGBM on e = y - p (features: NM clock
               offsets relative to p, airport, hour) trained on the other 6 OOF months, blended with a per-airport
               weight shrunk to the global one (javid form, k = 1000 pseudo-rows), weights fitted by an inner LOMO
               on the training months.
Gains: SSE reduction vs baseline / 1,233,234 (MSE units; ~480 MSE ~ 1 s at 238.8), per month, day-block bootstrap
(90 % interval over the 215 OOF days), and the share carried by the top 10 / top 50 rows.

    .venv/bin/python analysis/local_ext_overlays_bc.py q      # LOMO copy q -> $SCRATCH/agents/ext/lirf_q_lomo.parquet
    .venv/bin/python analysis/local_ext_overlays_bc.py eval
"""
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

S = os.environ["SCRATCH"]
O = f"{S}/agents/ext"
os.makedirs(O, exist_ok=True)
N = 1233234
OOFM = [1, 7, 8, 9, 10, 11, 12]
DROP = {"TAXITIME_SEC_mvt", "copy", "MVT_ID_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt"}
PQ = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
          bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=3, seed=1)
ROUNDS_Q = 109


def lomo_q():
    T = (pl.scan_parquet("data/work_flseq/feat_train.parquet")
         .filter((pl.col("ADEP_mvt") == "LIRF") & pl.col("mvt_minus_aobt3").is_not_null()).collect()
         .with_columns(((pl.col("TAXITIME_SEC_mvt") - pl.col("mvt_minus_sched")).abs() <= 12).alias("copy")))
    num = [c for c, t in T.schema.items() if c not in DROP and t.is_numeric()]
    cat = [c for c, t in T.schema.items() if c not in DROP and t == pl.String]
    codes = {c: {v: i for i, v in enumerate(sorted(set(T[c].drop_nulls().to_list())))} for c in cat}
    X = T.select(num + [pl.col(c).replace_strict(codes[c], default=None, return_dtype=pl.Int32).alias(c) for c in cat]
                 ).to_numpy().astype(np.float64)
    y = T["copy"].to_numpy().astype(int)
    m = T["month"].to_numpy()
    q = np.zeros(len(y))
    for k in range(1, 13):
        tr = m != k
        mdl = lgb.train(PQ, lgb.Dataset(X[tr], y[tr], feature_name=num + cat, categorical_feature=cat), ROUNDS_Q)
        q[~tr] = mdl.predict(X[~tr])
        print("month", k, "copy rate", y[~tr].mean().round(3), "mean q", q[~tr].mean().round(3), flush=True)
    pl.DataFrame({"MVT_ID_mvt": T["MVT_ID_mvt"], "month": m, "q": q, "copy": y.astype(bool),
                  "s": T["mvt_minus_sched"]}).write_parquet(f"{O}/lirf_q_lomo.parquet")


def stats(name, D, pn, base="p0"):
    """D: frame with y, base, month, day; pn: new prediction (numpy)."""
    y = D["y"].to_numpy()
    d = (D[base].to_numpy() - y) ** 2 - (pn - y) ** 2
    mo = D["month"].to_numpy()
    per_m = {k: d[mo == k].sum() / N for k in OOFM}
    day = D["dayk"].to_numpy()
    ud, inv = np.unique(day, return_inverse=True)
    perday = np.bincount(inv, weights=d) / N
    nd = len(ud)  # every OOF day has rows (215 days)
    rng = np.random.default_rng(0)
    bs = np.array([perday[rng.integers(0, nd, nd)].sum() for _ in range(4000)])
    o = np.sort(d)[::-1]
    tot = d.sum() / N
    nz = int((np.abs(pn - D[base].to_numpy()) > 1e-9).sum())
    print(f"{name:10s} gain {tot:+8.2f} MSE  touched {nz:6d}  months " + " ".join(f"{per_m[k]:+.1f}" for k in OOFM)
          + f"  boot90 [{np.percentile(bs, 5):+.1f},{np.percentile(bs, 95):+.1f}]  top10 {o[:10].sum() / N:+.1f}"
          f" top50 {o[:50].sum() / N:+.1f}  w/o top10 {(d.sum() - o[:10].sum()) / N:+.1f}", flush=True)
    return tot


def fitA(p, q, s, y, grid=np.linspace(0, 1.5, 61)):
    sse = [(((p + a * q * (s - p)) - y) ** 2).sum() for a in grid]
    return grid[int(np.argmin(sse))]


def evaluate():
    cv = pl.read_parquet(S + "/learn/cv_rows_months7_v2.parquet").with_columns(p=pl.col("b") + pl.col("c_C"))
    f = pl.read_parquet("data/work_flseq/feat_train.parquet", columns=[
        "MVT_ID_mvt", "MVT_TIME_UTC_mvt", "mvt_minus_aobt3", "mvt_minus_lobt", "mvt_minus_sched", "mvt_minus_eobt1",
        "mvt_minus_iobt", "lobt_minus_eobt1", "aobt3_minus_eobt1", "aobt3_minus_sched", "ADEP_flt", "IOBT_flt",
        "hour_local", "nbr_mvt_minus_aobt_mean_60m"])
    D = cv.join(f, on="MVT_ID_mvt", how="left").with_columns(
        month=pl.col("MVT_TIME_UTC_mvt").dt.month(), dayk=pl.col("MVT_TIME_UTC_mvt").dt.ordinal_day())
    Q = pl.read_parquet(f"{O}/lirf_q_lomo.parquet")
    D = D.join(Q.select("MVT_ID_mvt", "q", "copy"), on="MVT_ID_mvt", how="left")
    lirf = (D["ADEP_mvt"] == "LIRF").to_numpy() & D["q"].is_not_null().to_numpy()
    p, y, s = D["p"].to_numpy(), D["y"].to_numpy(), D["mvt_minus_sched"].to_numpy()
    q = D["q"].fill_null(0).to_numpy()
    mo = D["month"].to_numpy()
    pl_ = D["mvt_minus_lobt"].to_numpy()

    def lclip(x):
        ok = ~np.isnan(pl_)
        x = x.copy()
        x[ok] = np.clip(x[ok], pl_[ok] - 3606, pl_[ok] + 3606)
        return np.maximum(x, 0)

    pc = p.copy()
    pc[lirf] = p[lirf] + 0.4 * q[lirf] * (s[lirf] - p[lirf])
    p0 = lclip(pc)
    D = D.with_columns(p0=p0, pnoov=lclip(p))
    print("RMSE p (raw OOF) %.3f | + LIRF copy 0.4 + LOBT clip %.3f" % (np.sqrt(np.mean((p - y) ** 2)), np.sqrt(np.mean((p0 - y) ** 2))))
    stats("prod_copy", D, p0, base="pnoov")

    # ---------------- C: isotonic calibration of q ----------------
    print("\n== C (LIRF matched rows only are touched)")
    variants = {k: p0.copy() for k in ("C_iso1", "C_isoA", "C_raw1", "C_rawA")}
    Aiso, Araw = {}, {}
    qcal_all = np.zeros(len(y))
    for k in OOFM:
        cal = Q.filter(pl.col("month") != k)
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(cal["q"].to_numpy(), cal["copy"].to_numpy().astype(float))
        sel = lirf & (mo == k)
        qcal_all[sel] = ir.predict(q[sel])
    for k in OOFM:
        sel = lirf & (mo == k)
        oth = lirf & (mo != k)
        Aiso[k] = fitA(p[oth], qcal_all[oth], s[oth], y[oth])
        Araw[k] = fitA(p[oth], q[oth], s[oth], y[oth])
        for nm, qq, a in (("C_iso1", qcal_all, 1.0), ("C_isoA", qcal_all, Aiso[k]), ("C_raw1", q, 1.0), ("C_rawA", q, Araw[k])):
            variants[nm][sel] = lclip(p + a * qq * (s - p))[sel]
    print("A fitted (iso):", {k: round(v, 3) for k, v in Aiso.items()}, " (raw):", {k: round(v, 3) for k, v in Araw.items()})
    print("mean q raw %.3f iso %.3f copy rate %.3f (OOF LIRF matched)" % (q[lirf].mean(), qcal_all[lirf].mean(), D["copy"].to_numpy()[lirf].astype(float).mean()))
    for nm, v in variants.items():
        stats(nm, D, v)

    # ---------------- B: AOBT_3 invalid ----------------
    print("\n== B (matched rows)")
    pa = D["mvt_minus_aobt3"].to_numpy()
    dl = pl_ - pa  # = AOBT_3 - LOBT
    origin = (D["ADEP_flt"] == D["ADEP_mvt"]).fill_null(False).to_numpy()
    special = (D["ADEP_mvt"] == "LIRF").to_numpy() & D["IOBT_flt"].is_null().to_numpy()
    rng_ok = (pa >= -7200) & (pa <= 172800) & (pl_ >= -7200) & (pl_ <= 172800)
    trig = (dl > 7200) & rng_ok & origin & ~special
    trig = np.nan_to_num(trig.astype(float)).astype(bool)
    print("rule Z trigger rows", int(trig.sum()))
    bz = p0.copy()
    bz[trig] = np.maximum(0, pl_[trig])
    stats("B_Z", D, bz)
    bzw = p0.copy()
    for k in OOFM:
        sel, oth = trig & (mo == k), trig & (mo != k)
        r, dd = (y - p0)[oth], (pl_ - p0)[oth]
        w = float(np.clip((dd * r).sum() / max((dd * dd).sum(), 1e-9), 0, 1))
        bzw[sel] = lclip(p0 + w * (pl_ - p0))[sel]
    stats("B_Zw", D, bzw)

    feats = ["mvt_minus_aobt3", "mvt_minus_lobt", "mvt_minus_sched", "mvt_minus_eobt1", "mvt_minus_iobt",
             "lobt_minus_eobt1", "aobt3_minus_eobt1", "aobt3_minus_sched", "hour_local", "nbr_mvt_minus_aobt_mean_60m"]
    apts = sorted(D["ADEP_mvt"].unique().to_list())
    F = D.select([(pl.col(c).cast(pl.Float64) - (pl.col("p0") if c.startswith("mvt_minus") else 0)).alias(c) for c in feats]
                 + [pl.col("p0"), pl.col("ADEP_mvt").replace_strict({a: i for i, a in enumerate(apts)}, return_dtype=pl.Int32).alias("apt")]
                 ).to_numpy().astype(np.float64)
    fn = feats + ["p0", "apt"]
    e = y - p0
    PX = dict(objective="huber", alpha=600.0, learning_rate=0.03, num_leaves=15, min_data_in_leaf=200, lambda_l2=10.0,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=3, seed=1)
    aptv = F[:, -1].astype(int)

    def shrunk_w(dd, r, a_):
        mbar = np.mean(dd * dd) if len(dd) else 1.0
        kk = 1000.0
        aglob = np.clip((dd * r).sum() / max((dd * dd).sum(), 1e-9), 0, 1)
        out = {}
        for ap in np.unique(a_):
            s_ = a_ == ap
            out[ap] = float(np.clip(((dd[s_] * r[s_]).sum() + kk * mbar * aglob) / ((dd[s_] ** 2).sum() + kk * mbar), 0, 1))
        return aglob, out

    for gate_k in (1800, 3600):
        gate = np.nan_to_num((np.abs(dl) > gate_k).astype(float)).astype(bool) & ~np.isnan(pa) & ~np.isnan(pl_)
        bx = p0.copy()
        log = []
        for k in OOFM:
            sel = gate & (mo == k)
            trm = [m_ for m_ in OOFM if m_ != k]
            # inner LOMO on the 6 training months -> expert OOF deltas for weight fitting
            inner_d = np.full(len(y), np.nan)
            for j in trm:
                itr = gate & np.isin(mo, [m_ for m_ in trm if m_ != j])
                ite = gate & (mo == j)
                mdl = lgb.train(PX, lgb.Dataset(F[itr], e[itr], feature_name=fn, categorical_feature=["apt"]), 300)
                inner_d[ite] = mdl.predict(F[ite])
            tr = gate & np.isin(mo, trm)
            aglob, wa = shrunk_w(inner_d[tr], e[tr], aptv[tr])
            mdl = lgb.train(PX, lgb.Dataset(F[tr], e[tr], feature_name=fn, categorical_feature=["apt"]), 300)
            dk = mdl.predict(F[sel])
            w = np.array([wa.get(a_, aglob) for a_ in aptv[sel]])
            bx[sel] = lclip(p0 + 0)[sel] + 0  # placeholder for shape
            tmp = p0.copy()
            tmp[sel] = p0[sel] + w * dk
            bx[sel] = lclip(tmp)[sel]
            log.append((k, round(float(aglob), 3)))
        print(f"gate |AOBT3-LOBT|>{gate_k}: rows {int(gate.sum())}, global w per held-out month {log}")
        stats(f"B_X{gate_k}", D, bx)
        A_ = D.with_columns(dd=pl.Series((p0 - y) ** 2 - (bx - y) ** 2)).group_by("ADEP_mvt").agg(g=pl.col("dd").sum() / N).sort("ADEP_mvt")
        print("   per airport:", [(a, round(g, 1)) for a, g in A_.rows()])


if __name__ == "__main__":
    lomo_q() if sys.argv[1] == "q" else evaluate()
