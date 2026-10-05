"""Settings check for the learned ADS-B residual on the 4-month data (arm B of local_adsb_learn_months.py).

Same rows, folds, inner early-stopping split, features and clip as the arm-B run; only the LightGBM settings
change. Each variant is compared with the current settings (V0) by the paired day-block bootstrap of the per-row
gain difference, on Nov+Dec rows, Jan+Jul rows and all.

    POLARS_MAX_THREADS=2 .venv/bin/python analysis/local_adsb_tune_months.py
"""
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "analysis")
import local_adsb_learn_months as M  # noqa: E402

L, CV = M.L, M.CV
VARIANTS = {
    "V0 current": {},
    "V1 leaves127 min200": dict(num_leaves=127, min_data_in_leaf=200),
    "V2 leaves255 min300 ff0.7": dict(num_leaves=255, min_data_in_leaf=300, feature_fraction=0.7),
    "V3 leaves63 l2=50": dict(lambda_l2=50.0),
}


def main() -> None:
    keep = ["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "mvt_minus_aobt3", "b", "y", "day"]
    D = pl.concat([CV.load_labelled().select(keep), M.load_novdec().select(keep)])
    F = pl.concat([pl.read_parquet(os.path.join(L.OUT, "feat_2025.parquet")),
                   pl.read_parquet(os.path.join(L.OUT, "feat_2025novdec.parquet"))], how="diagonal_relaxed").drop("ADEP_mvt", "day")
    D = D.join(F, on="MVT_ID_mvt", how="left").with_columns(pl.col("icao").is_not_null().alias("has"),
                                                           pl.col("day").str.slice(0, 7).alias("ym"))
    days = sorted(D["day"].unique().to_list())
    months = sorted({d[:7] for d in days})
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
    X, feats, cats = L.design(D, "b")
    Xn = X.select(feats).to_numpy().astype(np.float64)
    has = D["has"].to_numpy()
    yv, bv = D["y"].to_numpy(), D["b"].to_numpy()
    r = np.clip(yv - bv, -CV.CLIP_Y, CV.CLIP_Y)
    cvf, day_arr, apt = D["cvfold"].to_numpy(), D["day"].to_numpy(), D["ADEP_mvt"].to_numpy()
    jj = np.isin(D["ym"].to_numpy(), ["2025-01", "2025-07"])
    e0 = yv - bv
    di, nd = D["day_idx"].to_numpy(), len(days)
    gains = {}
    for name, over in VARIANTS.items():
        P = {**CV.PARAMS, **over}
        c = np.zeros(D.height)
        its = []
        for k in range(CV.K):
            L.mem_gate()
            val_days = inner([d for d in days if fold_of[d] != k], CV.SEED + 1 + k)
            is_te = has & (cvf == k)
            is_tr = has & (cvf != k)
            is_val = is_tr & np.isin(day_arr, list(val_days))
            is_fit = is_tr & ~is_val
            dtr = lgb.Dataset(Xn[is_fit], r[is_fit], feature_name=feats, categorical_feature=cats, free_raw_data=True)
            dva = lgb.Dataset(Xn[is_val], r[is_val], reference=dtr)
            m = lgb.train(P, dtr, num_boost_round=5000, valid_sets=[dva], callbacks=[lgb.early_stopping(200, verbose=False)])
            its.append(int(m.best_iteration))
            c[is_te] = np.clip(m.predict(Xn[is_te], num_iteration=m.best_iteration), -CV.CLIP_C, CV.CLIP_C)
        c[apt == "LTFM"] = 0.0
        gains[name] = e0 ** 2 - (e0 - c) ** 2
        print(f"{name}: best iters {its}", flush=True)
    base = gains["V0 current"]
    for name, g in gains.items():
        parts = []
        for lab, sel in (("Nov+Dec", ~jj), ("Jan+Jul", jj), ("all", np.ones(D.height, bool))):
            if name == "V0 current":
                parts.append(f"{lab} gain {g[sel].mean():8.1f}")
            else:
                d, lo, hi = CV.boot((g - base)[sel], di[sel], nd)
                parts.append(f"{lab} {d:+7.1f} [{lo:+7.1f},{hi:+7.1f}]")
        print(f"{name:28s} " + " | ".join(parts), flush=True)


if __name__ == "__main__":
    main()
