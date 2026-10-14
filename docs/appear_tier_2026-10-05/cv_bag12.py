"""plan_bag12.md: average three seeds of the 12-month residual (P12 / A12), leave-whole-month-out CV (Jan, Jul 2025).

    SCRATCH=... POLARS_MAX_THREADS=2 .venv/bin/python docs/appear_tier_2026-10-05/cv_bag12.py

Rows, folds and early stopping as cv_lmo12.py. Writes cv_bag12_a2.json."""

import json
import os
import sys

sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv_months12 as M  # noqa: E402  (sets --base a2)

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

CV, L, AP, CM = M.CV, M.L, M.AP, M.CM
TEST = ("2025-01", "2025-07")
SEEDS = (0, 1, 2)


def main() -> None:
    base = {**CV.PARAMS, **M.V2, "num_threads": 4}
    D = M.load("fill")
    days = sorted(D["day"].unique().to_list())
    months = sorted({d[:7] for d in days})
    mo = D["day"].str.slice(0, 7).to_numpy()
    y, b, has, day_arr = D["y"].to_numpy(), D["b"].to_numpy(), D["has"].to_numpy(), D["day"].to_numpy()
    di = D["day"].replace_strict({d: i for i, d in enumerate(days)}).to_numpy()
    r = np.clip(y - b, -CV.CLIP_Y, CV.CLIP_Y)
    X, feats, cats = L.design(D, "b")
    mats = {"P": (X.select(feats).to_numpy().astype(np.float32), feats)}
    Xa, fa = AP.design(X, feats, "b")
    mats["A"] = (Xa.select(fa).to_numpy().astype(np.float32), fa)
    del X, Xa, D
    c = {(m, s): np.zeros(len(y)) for m in ("P", "A") for s in SEEDS}
    its = {}
    for i, mon in enumerate(TEST):
        tr_days = [d for d in days if d[:7] != mon]
        val_days = M.inner([x for x in months if x != mon], tr_days, CV.SEED + 101 + i)
        is_te = has & (mo == mon)
        is_tr = has & (mo != mon)
        is_val = is_tr & np.isin(day_arr, list(val_days))
        is_fit = is_tr & ~is_val
        for mat in ("P", "A"):
            Xn, fn = mats[mat]
            for s in SEEDS:
                params = {**base, "seed": CV.SEED + s}
                dtr = lgb.Dataset(Xn[is_fit], r[is_fit], feature_name=fn, categorical_feature=cats, free_raw_data=True)
                dva = lgb.Dataset(Xn[is_val], r[is_val], reference=dtr)
                mdl = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[dva],
                                callbacks=[lgb.early_stopping(200, verbose=False)])
                its[f"{mon} {mat}12 s{s}"] = int(mdl.best_iteration)
                c[(mat, s)][is_te] = np.clip(mdl.predict(Xn[is_te], num_iteration=mdl.best_iteration), -CV.CLIP_C, CV.CLIP_C)
                print(f"  test {mon} {mat}12 seed+{s}: best_iter {mdl.best_iteration}", flush=True)
    mA = np.mean([c[("A", s)] for s in SEEDS], axis=0)
    mP = np.mean([c[("P", s)] for s in SEEDS], axis=0)
    pred = {"single": b + c[("A", 0)], "bag": b + c[("P", 0)] + mA - mP, "bagfull": b + mA,
            "single_s1": b + c[("A", 1)], "single_s2": b + c[("A", 2)]}
    sel = np.isin(mo, TEST)
    nd = len(days)
    out = {}
    for key, a, b_ in (("bag vs single", "single", "bag"), ("bagfull vs single", "single", "bagfull"),
                       ("single s1 vs s0", "single", "single_s1"), ("single s2 vs s0", "single", "single_s2")):
        out[key] = {}
        for label, g in [("Jan+Jul", sel)] + [(m_, mo == m_) for m_ in TEST]:
            dd, lo, hi = CM._paired(y[g], pred[a][g], pred[b_][g], di[g], nd)
            out[key][label] = {"n": int(g.sum()), "d_rmse": dd, "lo": lo, "hi": hi}
            print(f"  {key:20s} {label:>8s}  {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}]", flush=True)
    j = out["bag vs single"]["Jan+Jul"]
    verdict = "ADOPT" if j["hi"] < 0 else "REJECT"
    print(f"\nDECISION (plan_bag12.md section 2): bag vs single {j['d_rmse']:+.3f} [{j['lo']:+.3f}, {j['hi']:+.3f}] -> {verdict}")
    json.dump({"rounds": its, "results": out, "verdict": verdict}, open(os.path.join(L.OUT, "cv_bag12_a2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
