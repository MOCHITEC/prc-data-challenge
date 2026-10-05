"""Finding 2b: add Aug-Oct 2025 to the learned ADS-B residual's training months (local, read-only).

Rows: Jan+Jul (cb-hybrid OOF), Nov+Dec (forward fold OOF) and Aug+Sep+Oct (local `aso` fold OOF: trained on the
other nine months). Matched lane = anchored LGB for Nov-Dec and Aug-Oct, unmatched = main_flseq.
Leave-days-out K=5 per month; two arms with the settings of adsb_learn_cv.py:
  B  fit on the training days of Jan, Jul, Nov, Dec (= the v37 recipe)
  C  fit on the training days of all seven months
Both predict the same test days of all seven months.

    POLARS_MAX_THREADS=2 .venv/bin/python analysis/local_adsb_learn_months7.py
"""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "analysis")
import local_adsb_learn_months as M  # noqa: E402

L, CV = M.L, M.CV
OUT = L.OUT


def main() -> None:
    keep = ["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "mvt_minus_aobt3", "b", "y", "day"]
    D = pl.concat([CV.load_labelled().select(keep), M.load_novdec().select(keep), M.load_novdec("aso", "aso").select(keep)])
    F = pl.concat([pl.read_parquet(os.path.join(OUT, f)) for f in ("feat_2025.parquet", "feat_2025novdec.parquet", "feat_2025aso.parquet")],
                  how="diagonal_relaxed").drop("ADEP_mvt", "day")
    D = D.join(F, on="MVT_ID_mvt", how="left").with_columns(pl.col("icao").is_not_null().alias("has"),
                                                           pl.col("day").str.slice(0, 7).alias("ym"))
    print("rows", D.height, D.group_by("ym").agg(pl.len(), pl.col("has").sum()).sort("ym").rows(), flush=True)
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
    cvf, day_arr, ym, apt = D["cvfold"].to_numpy(), D["day"].to_numpy(), D["ym"].to_numpy(), D["ADEP_mvt"].to_numpy()
    four = np.isin(ym, ["2025-01", "2025-07", "2025-11", "2025-12"])
    c = {"B": np.zeros(D.height), "C": np.zeros(D.height)}
    best = {"B": [], "C": []}
    for k in range(CV.K):
        L.mem_gate()
        val_days = inner([d for d in days if fold_of[d] != k], CV.SEED + 1 + k)
        is_te = has & (cvf == k)
        for arm, rowsel in (("B", four), ("C", np.ones(D.height, bool))):
            is_tr = has & (cvf != k) & rowsel
            is_val = is_tr & np.isin(day_arr, list(val_days))
            is_fit = is_tr & ~is_val
            dtr = lgb.Dataset(Xn[is_fit], r[is_fit], feature_name=feats, categorical_feature=cats, free_raw_data=True)
            dva = lgb.Dataset(Xn[is_val], r[is_val], reference=dtr)
            m = lgb.train(CV.PARAMS, dtr, num_boost_round=5000, valid_sets=[dva], callbacks=[lgb.early_stopping(200, verbose=False)])
            best[arm].append(int(m.best_iteration))
            c[arm][is_te] = np.clip(m.predict(Xn[is_te], num_iteration=m.best_iteration), -CV.CLIP_C, CV.CLIP_C)
            print(f"fold {k} arm {arm}: fit {is_fit.sum()} test {is_te.sum()} best_iter {m.best_iteration}", flush=True)
    json.dump({"best_iters": best, "final_rounds": {a: int(np.median(v)) for a, v in best.items()}, "features": feats},
              open(os.path.join(OUT, "cv_months7.json"), "w"), indent=1)
    for a in c:
        c[a][apt == "LTFM"] = 0.0
    e0 = yv - bv
    dB, dC = e0 ** 2 - (e0 - c["B"]) ** 2, e0 ** 2 - (e0 - c["C"]) ** 2
    di, nd = D["day_idx"].to_numpy(), len(days)
    print("\ngain = mean over all labelled rows of (y-b)^2-(y-b-c)^2 (s^2); C-B with a day-block bootstrap CI95")
    for lab, sel in (("Jan", ym == "2025-01"), ("Jul", ym == "2025-07"), ("Nov+Dec", np.isin(ym, ["2025-11", "2025-12"])),
                     ("Aug+Sep+Oct", np.isin(ym, ["2025-08", "2025-09", "2025-10"])), ("Jan+Jul", np.isin(ym, ["2025-01", "2025-07"])),
                     ("all", np.ones(D.height, bool))):
        g, lo, hi = CV.boot((dC - dB)[sel], di[sel], nd)
        print(f"{lab:12s} n={sel.sum():7d} gain B {dB[sel].mean():8.1f} C {dC[sel].mean():8.1f} | C-B {g:7.1f} [{lo:7.1f}, {hi:7.1f}]", flush=True)
    rows = []
    for a in sorted(set(apt)):
        for lab, sel0 in (("Jul", ym == "2025-07"), ("Nov+Dec", np.isin(ym, ["2025-11", "2025-12"])), ("Jan", ym == "2025-01")):
            sel = sel0 & (apt == a)
            g, lo, hi = CV.boot((dC - dB)[sel], di[sel], nd)
            rows.append({"airport": a, "months": lab, "n": int(sel.sum()), "C-B": round(g, 0), "lo": round(lo, 0), "hi": round(hi, 0)})
    pl.Config.set_tbl_rows(40)
    print(pl.DataFrame(rows))
    pl.DataFrame({"MVT_ID_mvt": D["MVT_ID_mvt"], "ym": D["ym"], "ADEP_mvt": D["ADEP_mvt"], "y": yv, "b": bv, "has": has,
                  "c_B": c["B"], "c_C": c["C"]}).write_parquet(os.path.join(OUT, "cv_rows_months7.parquet"))


if __name__ == "__main__":
    main()
