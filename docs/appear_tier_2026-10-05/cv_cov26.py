"""plan_cov26.md sections 2-3: how much of the 12-month appear residual's value at EDDF survives 2026-like coverage.

    THIN=<thin dir> SCRATCH=<normal adsb dir> POLARS_MAX_THREADS=2 .venv/bin/python docs/appear_tier_2026-10-05/cv_cov26.py

A12 is trained on the normal data of the other 11 months (Jan and Jul 2025 left out whole, as cv_lmo12.py) and
predicts the test month's rows twice: with the normal features and with the features rebuilt on thinned EDDF fixes
(cov_thin.py). Writes cv_cov26_a2.json."""

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
THIN = os.path.join(os.environ["THIN"], "learn")


def thinned(D: pl.DataFrame) -> pl.DataFrame:
    """The test months' rows with features and appear columns from the thinned build (same steps as M.load)."""
    days = [d for d in L.DAYS["2025"]]
    F = pl.concat([pl.read_parquet(os.path.join(THIN, f"feat_{d}.parquet")) for d in days], how="diagonal_relaxed")
    T = pl.concat([pl.read_parquet(os.path.join(THIN, f"appear_fill_{d}.parquet")) for d in days])
    B = D.filter(pl.col("day").str.slice(0, 7).is_in(TEST)).select(CM.KEEP)
    B = B.join(F.drop("ADEP_mvt", "day"), on="MVT_ID_mvt", how="left").with_columns(pl.col("icao").is_not_null().alias("has"))
    return B.drop([c for c in AP.COLS if c in B.columns]).join(T, on="MVT_ID_mvt", how="left").sort("MVT_ID_mvt")


def stats(D: pl.DataFrame, label: str) -> dict:
    X = D.filter((pl.col("ADEP_mvt") == "EDDF") & pl.col("icao").is_not_null())
    s = {"rows": X.height, "ground_fix": float(X["ground_fix"].fill_null(False).mean()),
         "post_n_median": float(X["post_n"].median() or 0), "stand_run_missing": float(X["stand_dur"].is_null().mean()),
         "at_own_stand": float((X["d_dep1_own"] <= A_RIN).fill_null(False).mean())}
    print(f"  EDDF {label:14s} " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in s.items()), flush=True)
    return s


A_RIN = AP.R_IN


def main() -> None:
    params = {**CV.PARAMS, **M.V2, "num_threads": 4}
    D = M.load("fill")
    Dt = thinned(D)
    st = {"2025 normal": stats(D.filter(pl.col("day").str.slice(0, 7).is_in(TEST)), "2025 normal"),
          "2025 thinned": stats(Dt, "2025 thinned")}
    G = pl.read_parquet(os.path.join(L.OUT, "feat_2026.parquet")).join(
        pl.read_parquet(os.path.join(L.OUT, "appear_fill_2026.parquet")), on="MVT_ID_mvt", how="left")
    st["2026"] = stats(G, "2026")
    days = sorted(D["day"].unique().to_list())
    months = sorted({d[:7] for d in days})
    mo = D["day"].str.slice(0, 7).to_numpy()
    y, b, has, day_arr = D["y"].to_numpy(), D["b"].to_numpy(), D["has"].to_numpy(), D["day"].to_numpy()
    di = D["day"].replace_strict({d: i for i, d in enumerate(days)}).to_numpy()
    r = np.clip(y - b, -CV.CLIP_Y, CV.CLIP_Y)
    X, feats, cats = L.design(D, "b")
    Xa, fa = AP.design(X, feats, "b")
    Xn = Xa.select(fa).to_numpy().astype(np.float32)
    Xt0, ft, _ = L.design(Dt, "b")
    Xt, ft = AP.design(Xt0, ft, "b")
    assert ft == fa
    Xt = Xt.select(fa).to_numpy().astype(np.float32)
    ids_t = Dt["MVT_ID_mvt"].to_numpy()
    has_t = Dt["has"].to_numpy()
    pos = {v: i for i, v in enumerate(D["MVT_ID_mvt"].to_numpy())}
    rows_t = np.array([pos[v] for v in ids_t])
    c_n, c_t = np.zeros(len(y)), np.zeros(len(y))
    for i, mon in enumerate(TEST):
        tr_days = [d for d in days if d[:7] != mon]
        val_days = M.inner([x for x in months if x != mon], tr_days, CV.SEED + 101 + i)
        is_tr = has & (mo != mon)
        is_val = is_tr & np.isin(day_arr, list(val_days))
        is_fit = is_tr & ~is_val
        dtr = lgb.Dataset(Xn[is_fit], r[is_fit], feature_name=fa, categorical_feature=cats, free_raw_data=True)
        dva = lgb.Dataset(Xn[is_val], r[is_val], reference=dtr)
        mdl = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[dva], callbacks=[lgb.early_stopping(200, verbose=False)])
        te = has & (mo == mon)
        c_n[te] = np.clip(mdl.predict(Xn[te], num_iteration=mdl.best_iteration), -CV.CLIP_C, CV.CLIP_C)
        tt = (mo[rows_t] == mon) & has_t
        c_t[rows_t[tt]] = np.clip(mdl.predict(Xt[tt], num_iteration=mdl.best_iteration), -CV.CLIP_C, CV.CLIP_C)
        print(f"  test {mon} A12: best_iter {mdl.best_iteration}", flush=True)
    apt = D["ADEP_mvt"].to_numpy()
    sel = np.isin(mo, TEST)
    nd = len(days)
    out = {"stats": st}
    for label, g in (("EDDF", sel & (apt == "EDDF")), ("other airports", sel & (apt != "EDDF")), ("all", sel)):
        rm = {k: float(np.sqrt(np.mean((v[g] - y[g]) ** 2))) for k, v in (("b", b), ("normal", b + c_n), ("thinned", b + c_t))}
        dd, lo, hi = CM._paired(y[g], b[g] + c_n[g], b[g] + c_t[g], di[g], nd)
        out[label] = {"n": int(g.sum()), **rm, "value_normal": rm["b"] - rm["normal"], "value_thinned": rm["b"] - rm["thinned"],
                      "lost": dd, "lo": lo, "hi": hi}
        print(f"  {label:15s} n={int(g.sum()):>7,}  RMSE b {rm['b']:.2f}  normal {rm['normal']:.2f}  thinned {rm['thinned']:.2f}  "
              f"value lost {dd:+.3f} [{lo:+.3f}, {hi:+.3f}]", flush=True)
    lost = out["EDDF"]["lost"]
    verdict = "GO" if lost >= 3.0 else "NO-GO"
    print(f"\nDECISION (plan_cov26.md section 3): EDDF value lost under 2026-like coverage {lost:+.3f} s -> {verdict}")
    out["verdict"] = verdict
    json.dump(out, open(os.path.join(L.OUT, "cv_cov26_a2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
