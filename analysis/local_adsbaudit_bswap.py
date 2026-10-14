"""adsbaudit step 3a: is the learned ADS-B residual sensitive to WHICH base b it was trained on? (local, read-only)

Production trains the residual on b = the OOF of older bases (Jan+Jul: 3-member cb hybrid; the other 10 months: the
single anchored LGB) and applies it in 2026 on b = a newer, stronger blend. Emulation on Jan+Jul 2025 (feat_2025):
three bases of rising quality on the matched lane, b_single = osm_nolirf OOF (H2), b_plainy = (osm_nolirf + plainy)/2,
b_cb = (osm_nolirf + plainy + cb)/3; unmatched lane = main_flseq OOF in all three.
Leave-days-out K=5 (adsb_learn_cv folds and inner early-stopping split), V2 params (production). Arms train on
b_single or b_cb (option EXTRA=1 adds the 10 other months, single-member b, to every training set as production
does); each trained model predicts every test row with each of the three b's. Gain over ground-fix test rows
(LTFM/LIRF c = 0 as production) of mean (y-b_t)^2 - (y-b_t-c)^2, day-block bootstrap for paired differences.

    POLARS_MAX_THREADS=2 [EXTRA=1] .venv/bin/python analysis/local_adsbaudit_bswap.py
"""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "analysis")
import local_adsb_learn_months as M  # noqa: E402  (sets --base cb)

L, CV = M.L, M.CV
S = os.environ["SCRATCH"]
OUT = os.path.join(S, "agents", "adsbaudit")
EXTRA = os.environ.get("EXTRA") == "1"
P = {**CV.PARAMS, **dict(num_leaves=255, min_data_in_leaf=300, feature_fraction=0.7)}
ROOT = "experiments/"


def jj():
    D = CV.load_labelled()  # b = cb hybrid; oa = cb-mean in place, so rebuild the members
    oa = pl.read_parquet(ROOT + f"{CV.H2_A}.parquet").rename({"oof": "m_a"})
    op = pl.read_parquet(ROOT + f"{CV.PLAIN}.parquet").rename({"oof": "m_p"})
    oc = pl.read_parquet(ROOT + f"{CV.CB}.parquet").rename({"oof": "m_c"})
    D = D.join(oa, on="MVT_ID_mvt", how="left").join(op, on="MVT_ID_mvt", how="left").join(oc, on="MVT_ID_mvt", how="left")
    A = pl.col("mvt_minus_aobt3").is_not_null()
    D = D.with_columns(pl.when(A).then("m_a").otherwise("ob").alias("b_single"),
                       pl.when(A).then((pl.col("m_a") + pl.col("m_p")) / 2).otherwise("ob").alias("b_plainy"),
                       pl.when(A).then((pl.col("m_a") + pl.col("m_p") + pl.col("m_c")) / 3).otherwise("ob").alias("b_cb"))
    assert (D["b_cb"] - D["b"]).abs().max() < 1e-6
    F = pl.read_parquet(os.path.join(L.OUT, "feat_2025.parquet")).drop("ADEP_mvt", "day")
    return D.join(F, on="MVT_ID_mvt", how="inner")


def extra():
    keep = ["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "mvt_minus_aobt3", "b", "y", "day"]
    parts = [M.load_novdec().select(keep), M.load_novdec("aso", "aso").select(keep), M.load_novdec("fmj", "fmj").select(keep)]
    F = pl.concat([pl.read_parquet(os.path.join(L.OUT, f)) for f in ("feat_2025novdec.parquet", "feat_2025aso.parquet",
                                                                     "feat_2025fmj.parquet")], how="diagonal_relaxed").drop("ADEP_mvt", "day")
    E = pl.concat(parts).join(F, on="MVT_ID_mvt", how="inner")
    X, feats, _ = L.design(E, "b")
    return X.select(feats).to_numpy().astype(np.float32), np.clip((E["y"] - E["b"]).to_numpy(), -CV.CLIP_Y, CV.CLIP_Y).astype(np.float32)


def main():
    D = jj()
    days = sorted(D["day"].unique().to_list())
    fold_of = CV.assign_folds(days)
    D = D.with_columns(pl.col("day").replace_strict(fold_of).alias("cvfold"),
                       pl.col("day").replace_strict({d: i for i, d in enumerate(days)}).alias("day_idx"))
    BS = ["b_single", "b_plainy", "b_cb"]
    Xb, feats, cats = {}, None, None
    for bn in BS:
        X, feats, cats = L.design(D, bn)
        Xb[bn] = X.select(feats).to_numpy().astype(np.float32)
    y = D["y"].to_numpy()
    cvf, dayarr = D["cvfold"].to_numpy(), D["day"].to_numpy()
    apt = D["ADEP_mvt"].to_numpy()
    gnd = D["ground_fix"].fill_null(False).to_numpy() & ~np.isin(apt, ["LTFM", "LIRF"])
    print("rows", D.height, "ground (scored)", int(gnd.sum()))
    for bn in BS:
        e = y - D[bn].to_numpy()
        print(f"{bn}: RMSE on scored ground rows {np.sqrt((e[gnd] ** 2).mean()):.2f}")
    XE, rE = extra() if EXTRA else (None, None)
    if EXTRA:
        print("extra rows (10 months, single b)", len(rE))
    TR = ["b_single", "b_cb"]
    c = {(s, t): np.zeros(D.height) for s in TR for t in BS}
    for k in range(CV.K):
        val = CV.inner_split([d for d in days if fold_of[d] != k], CV.SEED + 1 + k)
        is_tr = cvf != k
        is_val = is_tr & np.isin(dayarr, list(val))
        is_fit = is_tr & ~is_val
        te = cvf == k
        for s in TR:
            r = np.clip(y - D[s].to_numpy(), -CV.CLIP_Y, CV.CLIP_Y)
            Xf, rf = Xb[s][is_fit], r[is_fit]
            if EXTRA:
                Xf, rf = np.vstack([Xf, XE]), np.concatenate([rf, rE])
            dtr = lgb.Dataset(Xf, rf, feature_name=feats, categorical_feature=cats, free_raw_data=True)
            dva = lgb.Dataset(Xb[s][is_val], r[is_val], reference=dtr)
            m = lgb.train(P, dtr, num_boost_round=3000, valid_sets=[dva], callbacks=[lgb.early_stopping(200, verbose=False)])
            del Xf, rf, dtr
            for t in BS:
                c[(s, t)][te] = np.clip(m.predict(Xb[t][te], num_iteration=m.best_iteration), -CV.CLIP_C, CV.CLIP_C)
            print(f"fold {k} train {s}: best_iter {m.best_iteration}", flush=True)
    di, nd = D["day_idx"].to_numpy(), len(days)
    res = {}
    G = {}
    for s in TR:
        for t in BS:
            e0 = y - D[t].to_numpy()
            cc = np.where(gnd, c[(s, t)], 0.0)
            G[(s, t)] = e0 ** 2 - (e0 - cc) ** 2
            g, lo, hi = CV.boot(G[(s, t)], di, nd)
            rm = np.sqrt(((e0 - cc) ** 2).mean())
            res[f"{s}->{t}"] = dict(gain=g, lo=lo, hi=hi, rmse_after=float(rm), c_mean=float(cc[gnd].mean()),
                                    c_sd=float(cc[gnd].std()))
            print(f"train {s:9s} -> test {t:9s}: gain {g:8.1f} [{lo:8.1f}, {hi:8.1f}] RMSE after {rm:.2f} "
                  f"| c mean {cc[gnd].mean():+.1f} sd {cc[gnd].std():.1f}")
    print("\nmismatch cost on the same test b (gain matched - gain mismatched), day-block CI:")
    for t in BS:
        g, lo, hi = CV.boot(G[("b_cb", t)] - G[("b_single", t)], di, nd)
        res[f"cb-single on {t}"] = dict(gain=g, lo=lo, hi=hi)
        print(f"  test {t}: (train cb) - (train single) {g:+.1f} [{lo:+.1f}, {hi:+.1f}]")
    print("\nper airport, test b_cb: train cb vs train single")
    for a in sorted(set(apt)):
        m_ = apt == a
        g, lo, hi = CV.boot((G[("b_cb", "b_cb")] - G[("b_single", "b_cb")])[m_], di[m_], nd)
        print(f"  {a}: {g:+8.1f} [{lo:+8.1f}, {hi:+8.1f}]   gain cb->cb {G[('b_cb', 'b_cb')][m_].mean():8.1f}")
    tag = "_extra" if EXTRA else ""
    json.dump(res, open(os.path.join(OUT, f"bswap{tag}.json"), "w"), indent=1)
    pl.DataFrame({"MVT_ID_mvt": D["MVT_ID_mvt"], "ADEP_mvt": apt, "y": y, "gnd": gnd,
                  **{t: D[t].to_numpy() for t in BS},
                  **{f"c_{s}_{t}": c[(s, t)] for s in TR for t in BS}}).write_parquet(os.path.join(OUT, f"bswap_rows{tag}.parquet"))


if __name__ == "__main__":
    main()
