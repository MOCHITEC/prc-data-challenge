"""plan_standmap12.md sections 2-3: the 12-month appear residual (A12) with the own-stand tables fill, fill12, rand12.

    SCRATCH=... POLARS_MAX_THREADS=2 .venv/bin/python docs/appear_tier_2026-10-05/cv_standmap12.py

Same rows, folds, early-stopping days and V2 parameters as cv_months12.py. Writes cv_standmap12_a2.json."""

import json
import os
import sys

sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv_months12 as M  # noqa: E402  (sets --base a2)
from cv_standmap import differs  # noqa: E402

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

CV, L, AP, CM = M.CV, M.L, M.AP, M.CM
ARMS = ("fill", "fill12", "rand12")


def main() -> None:
    CV.PARAMS = {**CV.PARAMS, **M.V2}
    frames = {v: M.load(v) for v in ARMS}
    D = frames["fill"]
    assert all((f["MVT_ID_mvt"] == D["MVT_ID_mvt"]).all() for f in frames.values())
    new = differs(frames["fill"], frames["fill12"])
    print(f"rows {D.height:,}; new rows (own-stand columns differ fill vs fill12) {int(new.sum()):,}; "
          f"rand12 differs from fill12 on {int(differs(frames['fill12'], frames['rand12']).sum()):,}", flush=True)
    days = sorted(D["day"].unique().to_list())
    months, fold_of = M.folds(days)
    cvf = D["day"].replace_strict(fold_of).to_numpy()
    di = D["day"].replace_strict({d: i for i, d in enumerate(days)}).to_numpy()
    y, b, has, day_arr = D["y"].to_numpy(), D["b"].to_numpy(), D["has"].to_numpy(), D["day"].to_numpy()
    apt = D["ADEP_mvt"].to_numpy()
    r = np.clip(y - b, -CV.CLIP_Y, CV.CLIP_Y)
    mats = {}
    for v in ARMS:
        X, feats, cats = L.design(frames[v], "b")
        X, feats = AP.design(X, feats, "b")
        mats[v] = (X.select(feats).to_numpy().astype(np.float32), feats, cats)
    del frames, X
    c = {v: np.zeros(len(y)) for v in ARMS}
    its = {v: [] for v in ARMS}
    for k in range(CV.K):
        val_days = M.inner(months, [d for d in days if fold_of[d] != k], CV.SEED + 1 + k)
        is_te = has & (cvf == k)
        is_tr = has & (cvf != k)
        is_val = is_tr & np.isin(day_arr, list(val_days))
        is_fit = is_tr & ~is_val
        for v in ARMS:
            Xn, feats, cats = mats[v]
            dtr = lgb.Dataset(Xn[is_fit], r[is_fit], feature_name=feats, categorical_feature=cats, free_raw_data=True)
            dva = lgb.Dataset(Xn[is_val], r[is_val], reference=dtr)
            m = lgb.train(CV.PARAMS, dtr, num_boost_round=5000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(200, verbose=False)])
            its[v].append(int(m.best_iteration))
            c[v][is_te] = np.clip(m.predict(Xn[is_te], num_iteration=m.best_iteration), -CV.CLIP_C, CV.CLIP_C)
            print(f"  fold {k} {v}: fit {int(is_fit.sum())} es {int(is_val.sum())} best_iter {m.best_iteration}", flush=True)
    pred = {v: b + c[v] for v in ARMS}
    nd = len(days)
    out = {}
    tests = [("fill12 vs fill", "fill", "fill12"), ("fill12 vs rand12", "rand12", "fill12"), ("rand12 vs fill", "fill", "rand12")]
    groups = [("new rows", new), ("all rows", np.ones(len(y), bool))] + \
             [(f"new rows {a}", new & (apt == a)) for a in sorted(set(apt[new]))]
    for key, a, b_ in tests:
        out[key] = {}
        for label, sel in groups:
            if sel.sum() < 50:
                continue
            dd, lo, hi = CM._paired(y[sel], pred[a][sel], pred[b_][sel], di[sel], nd)
            out[key][label] = {"n": int(sel.sum()), "d_rmse": dd, "lo": lo, "hi": hi}
            print(f"  {key:18s} {label:>14s} n={int(sel.sum()):>9,}  {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    ok1 = out["fill12 vs fill"]["new rows"]["hi"] < 0
    ok2 = out["fill12 vs rand12"]["new rows"]["hi"] < 0
    ok3 = not (out["fill12 vs fill"]["all rows"]["lo"] > 0)
    verdict = "ADOPT" if ok1 and ok2 and ok3 else "REJECT"
    print(f"\nDECISION (plan_standmap12.md section 3): new rows vs fill {ok1}, vs rand12 {ok2}, all rows not worse {ok3} "
          f"-> {verdict}")
    res = {"final_rounds": {v: int(np.median(x)) for v, x in its.items()}, "features": {v: mats[v][1] for v in ARMS},
           "params": CV.PARAMS, "n_new": int(new.sum()), "results": out, "verdict": verdict}
    json.dump(res, open(os.path.join(L.OUT, "cv_standmap12_a2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
