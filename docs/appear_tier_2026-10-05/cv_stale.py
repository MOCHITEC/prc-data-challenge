"""plan_rebase_c12.md section 1-2: re-create in CV "the base moved but the ADS-B layers kept the old b" and compare
ways to put the 12-month appear on the moved base.

    SCRATCH=... POLARS_MAX_THREADS=2 .venv/bin/python docs/appear_tier_2026-10-05/cv_stale.py

Models as cv_lmo12.py (Jan and Jul 2025 left out whole; P12 / A12 on the other 11 months, P4 / A4 on the other three of
Jan, Jul, Nov, Dec; V2; fill). Each model is evaluated on the test rows at b (the a2 OOF) and at b' = b + w (b_alt - b)
on matched rows. Writes cv_stale_a2.json."""

import json
import os
import sys

sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv_months12 as M  # noqa: E402  (sets --base a2)

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

CV, L, AP, CM = M.CV, M.L, M.AP, M.CM
TEST = ("2025-01", "2025-07")
ALTS = {"flwin": ("flwin_both-33ed7358b9", 23.0), "tcdrop": ("tc_drop-86a3693e18", None)}  # (exp id, target RMS or w = 1)


def mats_at(D: pl.DataFrame, rows: np.ndarray, b: np.ndarray):
    Dt = D.filter(pl.Series(rows)).with_columns(pl.Series("b", b[rows]))
    X, feats, cats = L.design(Dt, "b")
    Xa, fa = AP.design(X, feats, "b")
    return X.select(feats).to_numpy().astype(np.float32), Xa.select(fa).to_numpy().astype(np.float32)


def main() -> None:
    CV.PARAMS = {**CV.PARAMS, **M.V2}
    D = M.load("fill")
    lane = pl.read_parquet(os.path.join(CV.ROOT, "data/work_flseq/feat_train.parquet"), columns=["MVT_ID_mvt", "mvt_minus_aobt3"])
    D = D.join(lane.rename({"mvt_minus_aobt3": "_m"}), on="MVT_ID_mvt", how="left")
    days = sorted(D["day"].unique().to_list())
    months = sorted({d[:7] for d in days})
    mo = D["day"].str.slice(0, 7).to_numpy()
    y, b, has, day_arr = D["y"].to_numpy(), D["b"].to_numpy(), D["has"].to_numpy(), D["day"].to_numpy()
    matched = D["_m"].is_not_null().to_numpy()
    app = D["own_ok"].is_not_null().to_numpy()
    di = D["day"].replace_strict({d: i for i, d in enumerate(days)}).to_numpy()
    apt = D["ADEP_mvt"].to_numpy()
    test = np.isin(mo, TEST)
    # moved bases
    bp = {"none": b.copy()}
    for name, (eid, target) in ALTS.items():
        o = pl.read_parquet(os.path.join(CV.ROOT, f"experiments/oof_{eid}.parquet"))
        alt = D.select("MVT_ID_mvt").join(o, on="MVT_ID_mvt", how="left")["oof"].to_numpy()
        d = np.where(matched & ~np.isnan(alt), alt - b, 0.0)
        rms = float(np.sqrt(np.mean(d[test & app] ** 2)))
        w = target / rms if target else 1.0
        bp[name] = b + w * d
        print(f"b' {name}: RMS(alt - b) on test appear rows {rms:.1f} s, w {w:.3f}, RMS(b' - b) {w * rms:.1f} s", flush=True)
    # training matrices at b (as cv_lmo12)
    X, feats, cats = L.design(D, "b")
    Xp = X.select(feats).to_numpy().astype(np.float32)
    Xa_df, fa = AP.design(X, feats, "b")
    Xa = Xa_df.select(fa).to_numpy().astype(np.float32)
    del X, Xa_df
    r = np.clip(y - b, -CV.CLIP_Y, CV.CLIP_Y)
    arms = {"P12": ("P", None), "A12": ("A", None), "P4": ("P", M.FOUR), "A4": ("A", M.FOUR)}
    c = {v: {k: np.zeros(len(y)) for k in arms} for v in bp}
    for i, m in enumerate(TEST):
        tr_days = [d for d in days if d[:7] != m]
        val_days = M.inner([x for x in months if x != m], tr_days, CV.SEED + 101 + i)
        is_te = has & (mo == m)
        te_mats = {v: mats_at(D, is_te, bv) for v, bv in bp.items()}
        for arm, (mat, rowsel) in arms.items():
            Xn, fn = (Xp, feats) if mat == "P" else (Xa, fa)
            is_tr = has & (mo != m) & (np.isin(mo, rowsel) if rowsel else True)
            is_val = is_tr & np.isin(day_arr, list(val_days))
            is_fit = is_tr & ~is_val
            dtr = lgb.Dataset(Xn[is_fit], r[is_fit], feature_name=fn, categorical_feature=cats, free_raw_data=True)
            dva = lgb.Dataset(Xn[is_val], r[is_val], reference=dtr)
            mdl = lgb.train(CV.PARAMS, dtr, num_boost_round=5000, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(200, verbose=False)])
            for v in bp:
                Xt = te_mats[v][0 if mat == "P" else 1]
                c[v][arm][is_te] = np.clip(mdl.predict(Xt, num_iteration=mdl.best_iteration), -CV.CLIP_C, CV.CLIP_C)
            print(f"  test {m} {arm}: best_iter {mdl.best_iteration}", flush=True)
    nd = len(days)
    out = {}
    for v in bp:
        k0, k1 = c["none"], c[v]  # models at the old b, at the moved b'
        B = bp[v]
        P = {"C0": B + k0["P12"] + k0["A4"] - k0["P4"],
             "old": B + k0["A12"],
             "req": B + k0["P12"] + k0["A4"] - k0["P4"] + (k1["A12"] - k1["P12"]) - (k1["A4"] - k1["P4"]),
             "swap": B + k0["P12"] + k1["A12"] - k1["P12"],
             "full": B + k1["A12"],
             "C0full": B + k1["P12"] + k1["A4"] - k1["P4"]}
        out[v] = {}
        for name in ("old", "req", "swap", "full", "C0full"):
            out[v][name] = {}
            for label, g in [("Jan+Jul", test)] + [(m_, mo == m_) for m_ in TEST] + \
                            [(a, test & (apt == a)) for a in sorted(set(apt[test])) if a != "LTFM"]:
                dd, lo, hi = CM._paired(y[g], P["C0"][g], P[name][g], di[g], nd)
                out[v][name][label] = {"n": int(g.sum()), "d_rmse": dd, "lo": lo, "hi": hi}
                if label in ("Jan+Jul", "2025-01", "2025-07"):
                    print(f"  [{v:6s}] {name:6s} vs C0 {label:>8s}  {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}]", flush=True)
    f = out["flwin"]
    s_ok = f["old"]["Jan+Jul"]["d_rmse"] > -0.3
    cands = {k: f[k]["Jan+Jul"] for k in ("swap", "req", "full") if f[k]["Jan+Jul"]["hi"] < 0}
    pick = None
    if s_ok and cands:
        pick = min(cands, key=lambda k: cands[k]["d_rmse"])
        if pick == "full" and "swap" in cands and cands["swap"]["d_rmse"] - cands["full"]["d_rmse"] < 0.1:
            pick = "swap"
    print(f"\nDECISION (plan_rebase_c12.md section 2): S {'supported' if s_ok else 'rejected'} "
          f"(old vs C0 {f['old']['Jan+Jul']['d_rmse']:+.3f}); construction -> {pick}")
    json.dump({"results": out, "S": "supported" if s_ok else "rejected", "pick": pick},
              open(os.path.join(L.OUT, "cv_stale_a2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
