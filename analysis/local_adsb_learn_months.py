"""Does training the learned ADS-B residual on more 2025 months help? (local, read-only; finding 2 of 2026-10-05)

Rows: Jan+Jul 2025 (base = v23-equivalent cb hybrid OOF, folds jan/jul, as in adsb_learn_cv.py --base cb) and
Nov+Dec 2025 (same hybrid, OOF of the `forward` fold: trained on Jan-Oct). Nov-Dec 2025 ADS-B ground coverage is
close to Jan 2026's (LFPG 0.6-0.7, LIRF ~0.88, LEMD 0.98), unlike Jan 2025 (0 at all three).

Leave-days-out, K=5 day groups per month. Per fold two arms with the features, parameters, clip and early-stopping
rule of adsb_learn_cv.py:
  A  fit on the Jan+Jul training days only (the production recipe)
  B  fit on the training days of all four months
Both predict the same test days of all four months. Gain = mean over all labelled rows of (y-b)^2 - (y-b-c)^2.

    POLARS_MAX_THREADS=2 .venv/bin/python analysis/local_adsb_learn_months.py
"""
from __future__ import annotations

import glob
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "scripts")
import adsb_learn_features as L  # noqa: E402

_g = L.mem_gate
L.mem_gate = lambda limit_mb=12500.0: _g(12500.0)
sys.argv = [sys.argv[0], "--base", "cb"]
import adsb_learn_cv as CV  # noqa: E402

OUT = L.OUT
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(220)
pl.Config.set_tbl_cols(16)


def fw(stem: str, tag: str = "fw", fold: str = "forward") -> tuple[pl.DataFrame, pl.DataFrame]:
    f = sorted(glob.glob(f"experiments/oof_{stem}_{tag}-*_folds.parquet"))
    assert len(f) == 1, (stem, f)
    oof = f[0].replace("_folds.parquet", ".parquet")
    return pl.read_parquet(f[0]).filter(pl.col("pooled") & (pl.col("fold") == fold)), pl.read_parquet(oof)


def load_novdec(tag: str = "fw", fold: str = "forward") -> pl.DataFrame:
    fa, oa = fw("main_flseq_osm_nolirf", tag, fold)
    _, ob = fw("main_flseq", tag, fold)
    # matched lane: mean of whichever of the three v23 members have a forward-fold OOF (time budget: the anchored
    # LGB alone if plainy / cb were not run). Arms A and B share this base, so the comparison stays like for like.
    mem = [oa.rename({"oof": "m0"})]
    for i, stem in enumerate(("main_flseq_osm_nolirf_plainy", "main_flseq_osm_nolirf_cb"), 1):
        if glob.glob(f"experiments/oof_{stem}_{tag}-*_folds.parquet"):
            mem.append(fw(stem, tag, fold)[1].rename({"oof": f"m{i}"}))
    print("Nov+Dec matched-lane members:", len(mem))
    ft = pl.read_parquet("data/work_flseq/feat_train.parquet",
                         columns=["MVT_ID_mvt", "TAXITIME_SEC_mvt", "mvt_minus_aobt3", "ADEP_mvt", "MVT_TIME_UTC_mvt"])
    D = fa.select("MVT_ID_mvt").join(ft, on="MVT_ID_mvt", how="left").join(ob.rename({"oof": "ob"}), on="MVT_ID_mvt", how="left")
    for m_ in mem:
        D = D.join(m_, on="MVT_ID_mvt", how="left")
    mcols = [c for c in D.columns if c in ("m0", "m1", "m2")]
    D = D.with_columns(pl.when(pl.col("mvt_minus_aobt3").is_not_null())
                       .then(pl.mean_horizontal(mcols)).otherwise(pl.col("ob")).alias("b"),
                       pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y"),
                       pl.col("MVT_TIME_UTC_mvt").dt.date().cast(pl.String).alias("day"))
    assert D["b"].null_count() == 0 and D["y"].null_count() == 0
    return D


def main() -> None:
    keep = ["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "mvt_minus_aobt3", "b", "y", "day"]
    D = pl.concat([CV.load_labelled().select(keep), load_novdec().select(keep)])
    F = pl.concat([pl.read_parquet(os.path.join(OUT, "feat_2025.parquet")),
                   pl.read_parquet(os.path.join(OUT, "feat_2025novdec.parquet"))], how="diagonal_relaxed").drop("ADEP_mvt", "day")
    D = D.join(F, on="MVT_ID_mvt", how="left").with_columns(pl.col("icao").is_not_null().alias("has"),
                                                           pl.col("day").str.slice(0, 7).alias("ym"))
    print("rows", D.height, D.group_by("ym").agg(pl.len(), pl.col("has").sum()).sort("ym").rows())
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
    cvf, day_arr = D["cvfold"].to_numpy(), D["day"].to_numpy()
    jj = np.isin(D["ym"].to_numpy(), ["2025-01", "2025-07"])
    c = {"A": np.zeros(D.height), "B": np.zeros(D.height)}
    best = {"A": [], "B": []}
    for k in range(CV.K):
        L.mem_gate()
        tr_days = [d for d in days if fold_of[d] != k]
        val_days = inner(tr_days, CV.SEED + 1 + k)
        is_te = has & (cvf == k)
        for arm, rowsel in (("A", jj), ("B", np.ones(D.height, bool))):
            is_tr = has & (cvf != k) & rowsel
            is_val = is_tr & np.isin(day_arr, list(val_days))
            is_fit = is_tr & ~is_val
            dtr = lgb.Dataset(Xn[is_fit], r[is_fit], feature_name=feats, categorical_feature=cats, free_raw_data=True)
            dva = lgb.Dataset(Xn[is_val], r[is_val], reference=dtr)
            m = lgb.train(CV.PARAMS, dtr, num_boost_round=5000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(200, verbose=False)])
            best[arm].append(int(m.best_iteration))
            c[arm][is_te] = np.clip(m.predict(Xn[is_te], num_iteration=m.best_iteration), -CV.CLIP_C, CV.CLIP_C)
            print(f"fold {k} arm {arm}: fit {is_fit.sum()} es {is_val.sum()} test {is_te.sum()} best_iter {m.best_iteration}",
                  flush=True)
    import json
    json.dump({"best_iters": best, "final_rounds": {a: int(np.median(v)) for a, v in best.items()}, "features": feats},
              open(os.path.join(OUT, "cv_months.json"), "w"), indent=1)
    apt = D["ADEP_mvt"].to_numpy()
    c["A"][apt == "LTFM"] = 0.0  # production: c = 0 at LTFM
    c["B"][apt == "LTFM"] = 0.0
    e0 = yv - bv
    dA, dB = e0 ** 2 - (e0 - c["A"]) ** 2, e0 ** 2 - (e0 - c["B"]) ** 2
    di, nd = D["day_idx"].to_numpy(), len(days)
    ym = D["ym"].to_numpy()
    print("\ngain = mean over all labelled rows of (y-b)^2-(y-b-c)^2 (s^2); B-A with a day-block bootstrap CI95")
    for lab, sel in (("Jan+Jul", jj), ("Nov+Dec", ~jj), ("Nov", ym == "2025-11"), ("Dec", ym == "2025-12"),
                     ("all 4 months", np.ones(D.height, bool))):
        gA, gB = dA[sel].mean(), dB[sel].mean()
        g, lo, hi = CV.boot((dB - dA)[sel], di[sel], nd)
        rm0 = np.sqrt((e0[sel] ** 2).mean())
        print(f"{lab:13s} n={sel.sum():7d} RMSE base {rm0:7.2f} | A {np.sqrt(((e0 - c['A'])[sel] ** 2).mean()):7.2f} "
              f"B {np.sqrt(((e0 - c['B'])[sel] ** 2).mean()):7.2f} | gain A {gA:8.1f} B {gB:8.1f} | B-A {g:7.1f} [{lo:7.1f}, {hi:7.1f}]")
    rows = []
    for a in sorted(set(apt)):
        for lab, sel0 in (("Jan+Jul", jj), ("Nov+Dec", ~jj)):
            sel = sel0 & (apt == a)
            g, lo, hi = CV.boot((dB - dA)[sel], di[sel], nd)
            rows.append({"airport": a, "months": lab, "n": int(sel.sum()), "adsb": int((sel & has).sum()),
                         "gain_A": round(float(dA[sel].mean()), 0), "gain_B": round(float(dB[sel].mean()), 0),
                         "B-A": round(g, 0), "lo": round(lo, 0), "hi": round(hi, 0)})
    print(pl.DataFrame(rows))
    pl.DataFrame({"MVT_ID_mvt": D["MVT_ID_mvt"], "ym": D["ym"], "ADEP_mvt": D["ADEP_mvt"], "y": yv, "b": bv,
                  "has": has, "c_A": c["A"], "c_B": c["B"]}).write_parquet(os.path.join(OUT, "cv_rows_months.parquet"))


if __name__ == "__main__":
    main()
