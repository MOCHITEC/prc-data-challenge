"""plan_cov26.md section 5: does a residual trained on sparse coverage lose value on denser test features?

    THIN=<thin dir> SCRATCH=<normal adsb dir> POLARS_MAX_THREADS=2 .venv/bin/python docs/appear_tier_2026-10-05/cv_cov26c.py

Same folds as cv_cov26b.py (Jan and Jul 2025 left out whole). The thin-trained A12 (EDDF thinned in every training
month) predicts the test month's EDDF rows twice: on their native (dense) features and on the thinned ones. The
normal-trained A12 on native features is the matched-density reference. Writes cv_cov26c_a2.json."""

import json
import os
import sys

sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv_cov26b as CB  # noqa: E402  (sets --base a2)

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

M, CV, L, AP, CM = CB.M, CB.CV, CB.L, CB.AP, CB.CM


def main() -> None:
    params = {**CV.PARAMS, **M.V2, "num_threads": 4}
    D = M.load("fill")
    Dt = CB.with_thin_eddf(D)
    assert (Dt["MVT_ID_mvt"] == D["MVT_ID_mvt"]).all()
    days = sorted(D["day"].unique().to_list())
    months = sorted({d[:7] for d in days})
    mo = D["day"].str.slice(0, 7).to_numpy()
    y, b, day_arr = D["y"].to_numpy(), D["b"].to_numpy(), D["day"].to_numpy()
    has = {"normal": D["has"].to_numpy(), "thin": Dt["has"].to_numpy()}
    di = D["day"].replace_strict({d: i for i, d in enumerate(days)}).to_numpy()
    apt = D["ADEP_mvt"].to_numpy()
    r = np.clip(y - b, -CV.CLIP_Y, CV.CLIP_Y)
    mats = {}
    for name, frame in (("normal", D), ("thin", Dt)):
        X, feats, cats = L.design(frame, "b")
        Xa, fa = AP.design(X, feats, "b")
        mats[name] = Xa.select(fa).to_numpy().astype(np.float32)
    del D, Dt, X, Xa
    c = {k: np.zeros(len(y)) for k in ("thin_on_dense", "thin_on_thin", "normal_on_dense")}
    for i, mon in enumerate(CB.TEST):
        tr_days = [d for d in days if d[:7] != mon]
        val_days = M.inner([x for x in months if x != mon], tr_days, CV.SEED + 101 + i)
        for arm in ("normal", "thin"):
            is_tr = has[arm] & (mo != mon)
            is_val = is_tr & np.isin(day_arr, list(val_days))
            is_fit = is_tr & ~is_val
            dtr = lgb.Dataset(mats[arm][is_fit], r[is_fit], feature_name=fa, categorical_feature=cats, free_raw_data=True)
            dva = lgb.Dataset(mats[arm][is_val], r[is_val], reference=dtr)
            mdl = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[dva], callbacks=[lgb.early_stopping(200, verbose=False)])
            for feat in (("normal", "thin") if arm == "thin" else ("normal",)):
                te = has[feat] & (mo == mon)
                key = f"{arm}_on_{'dense' if feat == 'normal' else 'thin'}"
                c[key][te] = np.clip(mdl.predict(mats[feat][te], num_iteration=mdl.best_iteration), -CV.CLIP_C, CV.CLIP_C)
            print(f"  test {mon} {arm}-trained: best_iter {mdl.best_iteration}", flush=True)
    g = np.isin(mo, CB.TEST) & (apt == "EDDF")
    nd = len(days)
    rm = {k: float(np.sqrt(np.mean((b[g] + v[g] - y[g]) ** 2))) for k, v in c.items()}
    rm["b"] = float(np.sqrt(np.mean((b[g] - y[g]) ** 2)))
    dd, lo, hi = CM._paired(y[g], b[g] + c["thin_on_thin"][g], b[g] + c["thin_on_dense"][g], di[g], nd)
    rf, rlo, rhi = CM._paired(y[g], b[g] + c["normal_on_dense"][g], b[g] + c["thin_on_dense"][g], di[g], nd)
    print(f"  EDDF n={int(g.sum()):,}  RMSE b {rm['b']:.2f}  thin-trained on dense {rm['thin_on_dense']:.2f}  "
          f"thin-trained on thinned {rm['thin_on_thin']:.2f}  normal-trained on dense {rm['normal_on_dense']:.2f}")
    print(f"  d = dense - thinned (thin-trained) {dd:+.3f} [{lo:+.3f}, {hi:+.3f}]; "
          f"thin-trained vs normal-trained on dense {rf:+.3f} [{rlo:+.3f}, {rhi:+.3f}]")
    verdict = "GO" if lo > 0 else "NO-GO"
    print(f"\nDECISION (plan_cov26.md section 5): {verdict}")
    out = {"n": int(g.sum()), **rm, "d": dd, "lo": lo, "hi": hi, "thin_vs_normal_on_dense": rf, "lo2": rlo, "hi2": rhi,
           "verdict": verdict}
    json.dump(out, open(os.path.join(L.OUT, "cv_cov26c_a2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
