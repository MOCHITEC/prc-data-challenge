"""plan_lmo12.md: the 12-month appear delta under a leave-whole-month-out CV (test months Jan and Jul 2025).

    SCRATCH=... POLARS_MAX_THREADS=2 .venv/bin/python docs/appear_tier_2026-10-05/cv_lmo12.py

Same rows, base (a2), V2 parameters and early-stopping rule as cv_months12.py, but each test month is left out of every
model's training: P12 / A12 (and A12 on the fill12 table) on the other 11 months, P4 / A4 on the other three of
Jan, Jul, Nov, Dec. Writes cv_lmo12_a2.json and cv_lmo12_rows_a2.parquet."""

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


def main() -> None:
    CV.PARAMS = {**CV.PARAMS, **M.V2}
    F = {v: M.load(v) for v in ("fill", "fill12")}
    D = F["fill"]
    assert (F["fill12"]["MVT_ID_mvt"] == D["MVT_ID_mvt"]).all()
    days = sorted(D["day"].unique().to_list())
    months = sorted({d[:7] for d in days})
    mo = D["day"].str.slice(0, 7).to_numpy()
    y, b, has, day_arr = D["y"].to_numpy(), D["b"].to_numpy(), D["has"].to_numpy(), D["day"].to_numpy()
    di = D["day"].replace_strict({d: i for i, d in enumerate(days)}).to_numpy()
    apt = D["ADEP_mvt"].to_numpy()
    r = np.clip(y - b, -CV.CLIP_Y, CV.CLIP_Y)
    mats = {}
    X, feats, cats = L.design(D, "b")
    mats["P"] = (X.select(feats).to_numpy().astype(np.float32), feats, cats)
    Xa, fa = AP.design(X, feats, "b")
    mats["A"] = (Xa.select(fa).to_numpy().astype(np.float32), fa, cats)
    Xf, ff, cf = L.design(F["fill12"], "b")
    Xf, ff = AP.design(Xf, ff, "b")
    mats["Af12"] = (Xf.select(ff).to_numpy().astype(np.float32), ff, cf)
    keep = D.select("MVT_ID_mvt", "day", "ADEP_mvt", "y", "b", "has")
    del F, D, X, Xa, Xf
    arms = {"P12": ("P", None), "A12": ("A", None), "Af12": ("Af12", None), "P4": ("P", M.FOUR), "A4": ("A", M.FOUR)}
    c = {k: np.zeros(len(y)) for k in arms}
    its = {k: [] for k in arms}
    for i, m in enumerate(TEST):
        tr_days = [d for d in days if d[:7] != m]
        val_days = M.inner([x for x in months if x != m], tr_days, CV.SEED + 101 + i)
        is_te = has & (mo == m)
        for arm, (mat, rowsel) in arms.items():
            Xn, feats, cats = mats[mat]
            is_tr = has & (mo != m) & (np.isin(mo, rowsel) if rowsel else True)
            is_val = is_tr & np.isin(day_arr, list(val_days))
            is_fit = is_tr & ~is_val
            dtr = lgb.Dataset(Xn[is_fit], r[is_fit], feature_name=feats, categorical_feature=cats, free_raw_data=True)
            dva = lgb.Dataset(Xn[is_val], r[is_val], reference=dtr)
            mdl = lgb.train(CV.PARAMS, dtr, num_boost_round=5000, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(200, verbose=False)])
            its[arm].append(int(mdl.best_iteration))
            c[arm][is_te] = np.clip(mdl.predict(Xn[is_te], num_iteration=mdl.best_iteration), -CV.CLIP_C, CV.CLIP_C)
            print(f"  test {m} {arm}: fit {int(is_fit.sum())} es {int(is_val.sum())} best_iter {mdl.best_iteration}", flush=True)
    pred = {"final_12": b + c["A12"], "final_mix": b + c["P12"] + c["A4"] - c["P4"], "P12": b + c["P12"],
            "final_12f12": b + c["Af12"]}
    sel = np.isin(mo, TEST)
    nd = len(days)
    out = {}
    tests = [("final_12 vs final_mix", "final_mix", "final_12"), ("final_12 vs P12", "P12", "final_12"),
             ("final_mix vs P12", "P12", "final_mix"), ("final_12f12 vs final_12", "final_12", "final_12f12")]
    groups = [("Jan+Jul", sel)] + [(m, mo == m) for m in TEST] + [(a, sel & (apt == a)) for a in sorted(set(apt[sel]))]
    for key, a, b_ in tests:
        out[key] = {}
        for label, g in groups:
            dd, lo, hi = CM._paired(y[g], pred[a][g], pred[b_][g], di[g], nd)
            out[key][label] = {"n": int(g.sum()), "d_rmse": dd, "lo": lo, "hi": hi}
            print(f"  {key:26s} {label:>8s} n={int(g.sum()):>7,}  {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    j = out["final_12 vs final_mix"]["Jan+Jul"]
    rejected = j["d_rmse"] <= -0.3 and j["hi"] < 0
    print(f"\nDIAGNOSIS (plan_lmo12.md section 3): final_12 vs final_mix on Jan+Jul {j['d_rmse']:+.3f} "
          f"[{j['lo']:+.3f}, {j['hi']:+.3f}] -> H {'rejected' if rejected else 'supported'}")
    json.dump({"rounds": its, "params": CV.PARAMS, "results": out, "H": "rejected" if rejected else "supported"},
              open(os.path.join(L.OUT, "cv_lmo12_a2.json"), "w"), indent=1)
    keep.filter(pl.Series(sel)).with_columns([pl.Series(k, v[sel]) for k, v in pred.items()]).write_parquet(
        os.path.join(L.OUT, "cv_lmo12_rows_a2.parquet"))


if __name__ == "__main__":
    main()
