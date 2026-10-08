"""plan_cov26.md section 4 (stage 2): train the 12-month appear residual on data whose EDDF ground fixes are thinned to
2026 density, and test both it and the current one on 2026-like (thinned) EDDF rows.

    THIN=<thin dir> SCRATCH=<normal adsb dir> POLARS_MAX_THREADS=2 .venv/bin/python docs/appear_tier_2026-10-05/cv_cov26b.py

Jan and Jul 2025 are left out whole (as cv_lmo12.py). Rows at EDDF carry the features rebuilt on thinned fixes in every
month (cov_thin.py); other airports keep the normal features. Writes cv_cov26b_a2.json."""

import json
import os
import sys

sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv_months12 as M  # noqa: E402  (sets --base a2)
from build_features import DAYS  # noqa: E402

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

CV, L, AP, CM = M.CV, M.L, M.AP, M.CM
TEST = ("2025-01", "2025-07")
THIN = os.path.join(os.environ["THIN"], "learn")


def with_thin_eddf(D: pl.DataFrame) -> pl.DataFrame:
    """D with the EDDF rows' features and appear columns replaced by the thinned build (all 12 months)."""
    days = DAYS["2025"] + DAYS["2025nd"] + DAYS["2025m8"]
    F = pl.concat([pl.read_parquet(os.path.join(THIN, f"feat_{d}.parquet")) for d in days], how="diagonal_relaxed")
    T = pl.concat([pl.read_parquet(os.path.join(THIN, f"appear_fill_{d}.parquet")) for d in days])
    E = D.filter(pl.col("ADEP_mvt") == "EDDF").select(CM.KEEP)
    E = E.join(F.drop("ADEP_mvt", "day"), on="MVT_ID_mvt", how="left").with_columns(pl.col("icao").is_not_null().alias("has"))
    E = E.drop([c for c in AP.COLS if c in E.columns]).join(T, on="MVT_ID_mvt", how="left")
    return pl.concat([D.filter(pl.col("ADEP_mvt") != "EDDF"), E.select(D.columns)], how="vertical_relaxed").sort("MVT_ID_mvt")


def main() -> None:
    params = {**CV.PARAMS, **M.V2, "num_threads": 4}
    D = M.load("fill")
    Dt = with_thin_eddf(D)
    assert (Dt["MVT_ID_mvt"] == D["MVT_ID_mvt"]).all()
    days = sorted(D["day"].unique().to_list())
    months = sorted({d[:7] for d in days})
    mo = D["day"].str.slice(0, 7).to_numpy()
    y, b, day_arr = D["y"].to_numpy(), D["b"].to_numpy(), D["day"].to_numpy()
    has_n, has_t = D["has"].to_numpy(), Dt["has"].to_numpy()
    di = D["day"].replace_strict({d: i for i, d in enumerate(days)}).to_numpy()
    apt = D["ADEP_mvt"].to_numpy()
    r = np.clip(y - b, -CV.CLIP_Y, CV.CLIP_Y)
    mats = {}
    for name, frame in (("normal", D), ("thin", Dt)):
        X, feats, cats = L.design(frame, "b")
        Xa, fa = AP.design(X, feats, "b")
        mats[name] = Xa.select(fa).to_numpy().astype(np.float32)
    del D, Dt, X, Xa
    c = {k: np.zeros(len(y)) for k in ("normal", "thin")}
    for i, mon in enumerate(TEST):
        tr_days = [d for d in days if d[:7] != mon]
        val_days = M.inner([x for x in months if x != mon], tr_days, CV.SEED + 101 + i)
        te = has_t & (mo == mon)  # test on 2026-like rows: thinned EDDF, normal elsewhere
        for arm in ("normal", "thin"):
            Xn, has = mats[arm], (has_n if arm == "normal" else has_t)
            is_tr = has & (mo != mon)
            is_val = is_tr & np.isin(day_arr, list(val_days))
            is_fit = is_tr & ~is_val
            dtr = lgb.Dataset(Xn[is_fit], r[is_fit], feature_name=fa, categorical_feature=cats, free_raw_data=True)
            dva = lgb.Dataset(Xn[is_val], r[is_val], reference=dtr)
            mdl = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[dva], callbacks=[lgb.early_stopping(200, verbose=False)])
            c[arm][te] = np.clip(mdl.predict(mats["thin"][te], num_iteration=mdl.best_iteration), -CV.CLIP_C, CV.CLIP_C)
            print(f"  test {mon} {arm}-trained: best_iter {mdl.best_iteration}", flush=True)
    sel = np.isin(mo, TEST)
    nd = len(days)
    out = {}
    for label, g in (("EDDF", sel & (apt == "EDDF")), ("other airports", sel & (apt != "EDDF")), ("all", sel)):
        rm = {k: float(np.sqrt(np.mean((v[g] - y[g]) ** 2))) for k, v in (("b", b), ("normal", b + c["normal"]), ("thin", b + c["thin"]))}
        dd, lo, hi = CM._paired(y[g], b[g] + c["normal"][g], b[g] + c["thin"][g], di[g], nd)
        out[label] = {"n": int(g.sum()), **rm, "d_rmse": dd, "lo": lo, "hi": hi}
        print(f"  {label:15s} n={int(g.sum()):>7,}  RMSE b {rm['b']:.2f}  normal-trained {rm['normal']:.2f}  "
              f"thin-trained {rm['thin']:.2f}  thin vs normal {dd:+.3f} [{lo:+.3f}, {hi:+.3f}]", flush=True)
    ok = out["EDDF"]["hi"] < 0 and not out["other airports"]["lo"] > 0
    verdict = "ADOPT" if ok else "REJECT"
    print(f"\nDECISION (plan_cov26.md section 4): EDDF {out['EDDF']['d_rmse']:+.3f}, other airports "
          f"{out['other airports']['d_rmse']:+.3f} -> {verdict}")
    out["verdict"] = verdict
    json.dump(out, open(os.path.join(L.OUT, "cv_cov26b_a2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
