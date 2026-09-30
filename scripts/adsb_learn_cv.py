"""Leave-days-out CV of the learned ADS-B residual model against the rule override, on Jan+Jul 2025.

Pre-registration: `docs/competitor_research_2026-09-21/results/adsb-learned-residual.md` section 0 (and 0a).
Written from scratch. It reads `$SCRATCH/learn/feat_2025.parquet` (from `adsb_learn_features.py`), the labels and
lanes in `data/work_flseq/feat_train.parquet`, and the two OOF files that make up H2. Nothing is written to `src/`,
`configs/`, `outputs/` or `data/`.

    POLARS_MAX_THREADS=2 .venv/bin/python scripts/adsb_learn_cv.py

Writes `$SCRATCH/learn/cv_rows.parquet` (per row: y, b, c_learned, c_rule, fold) and `$SCRATCH/learn/cv.json`
(the fold best iterations, used for the final fit, and the headline numbers).
"""

from __future__ import annotations

import json
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adsb_learn_features as L

ROOT = L.s1.REPO_DIR
OUT = L.OUT
SEED = 20260930
K = 5
B = 2000
CLIP_Y = 3600.0
CLIP_C = 1800.0
GUARD = 900.0
RULE_APTS = ["EDDM", "EHAM", "LEBL", "LSZH", "EGLL"]
PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, num_threads=2, seed=SEED, verbose=-1,
              max_bin=255)
H2_A = "oof_main_flseq_osm_nolirf-926741d779"   # matched lane
H2_B = "oof_main_flseq-8615e6a194"               # unmatched lane

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(250)
pl.Config.set_tbl_cols(30)


def head(s):
    print(f"\n{'=' * 100}\n{s}\n{'=' * 100}", flush=True)


def load_labelled() -> pl.DataFrame:
    fold = pl.read_parquet(ROOT + f"experiments/{H2_A}_folds.parquet").filter(
        pl.col("pooled") & pl.col("fold").is_in(["jan", "jul"]))
    foldb = pl.read_parquet(ROOT + f"experiments/{H2_B}_folds.parquet").filter(
        pl.col("pooled") & pl.col("fold").is_in(["jan", "jul"]))
    assert fold.sort("MVT_ID_mvt").equals(foldb.sort("MVT_ID_mvt")), "the two runs' jan/jul folds differ"
    ft = pl.read_parquet(ROOT + "data/work_flseq/feat_train.parquet",
                         columns=["MVT_ID_mvt", "TAXITIME_SEC_mvt", "mvt_minus_aobt3", "ADEP_mvt", "MVT_TIME_UTC_mvt"])
    oa = pl.read_parquet(ROOT + f"experiments/{H2_A}.parquet").rename({"oof": "oa"})
    ob = pl.read_parquet(ROOT + f"experiments/{H2_B}.parquet").rename({"oof": "ob"})
    D = (fold.select("MVT_ID_mvt", "fold").join(ft, on="MVT_ID_mvt", how="left")
         .join(oa, on="MVT_ID_mvt", how="left").join(ob, on="MVT_ID_mvt", how="left"))
    D = D.with_columns(pl.when(pl.col("mvt_minus_aobt3").is_not_null()).then("oa").otherwise("ob").alias("b"),
                       pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y"),
                       pl.col("MVT_TIME_UTC_mvt").dt.date().cast(pl.String).alias("day"))
    print("labelled rows", D.height, dict(D.group_by("fold").len().iter_rows()),
          "| null y", D["y"].null_count(), "null b", D["b"].null_count(), "| days", D["day"].n_unique(),
          "| lane A", int(D["mvt_minus_aobt3"].is_not_null().sum()))
    ref = os.path.join(os.path.dirname(L.SCR), "exp", "oof_hybrid_H2.parquet")
    if os.path.exists(ref):
        R = pl.read_parquet(ref)
        c = D.select("MVT_ID_mvt", "b").join(R.rename({R.columns[-1]: "ref"}), on="MVT_ID_mvt", how="inner")
        print(f"cross-check vs {ref}: {c.height} common rows, max |b - ref| {(c['b'] - c['ref']).abs().max()}")
    return D


def assign_folds(days: list[str]) -> dict[str, int]:
    rng = np.random.default_rng(SEED)
    out = {}
    for m in ("2025-01", "2025-07"):
        ds = sorted(d for d in days if d.startswith(m))
        for i, j in enumerate(rng.permutation(len(ds))):
            out[ds[j]] = i % K
    return out


def inner_split(train_days: list[str], seed: int) -> set[str]:
    rng = np.random.default_rng(seed)
    val = set()
    for m in ("2025-01", "2025-07"):
        ds = sorted(d for d in train_days if d.startswith(m))
        perm = rng.permutation(len(ds))
        val |= {ds[j] for i, j in enumerate(perm) if i % 5 == 0}
    return val


def boot(d: np.ndarray, day_idx: np.ndarray, n_days: int, seed: int = SEED) -> tuple[float, float, float]:
    sums = np.bincount(day_idx, weights=d, minlength=n_days)
    cnts = np.bincount(day_idx, minlength=n_days).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, n_days, size=(B, n_days))
    s = sums[pick].sum(axis=1) / cnts[pick].sum(axis=1)
    return float(d.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def rule_corr(D: pl.DataFrame, fold_of: dict[str, int]) -> np.ndarray:
    """v20 rule on the 2025 rows, biases per airport fitted on the other folds' days."""
    elig = (pl.col("ADEP_mvt").is_in(RULE_APTS) & pl.col("t_aobt3").is_not_null() & (pl.col("join") == "cs")
            & pl.col("t_push").is_not_null()).fill_null(False)
    E = D.with_columns(elig.alias("elig"), (pl.col("t_push") - (pl.col("t_mvt") - pl.col("y"))).alias("off"))
    c = np.zeros(E.height)
    for k in range(K):
        tr = E.filter(pl.col("elig") & (pl.col("cvfold") != k))
        bias = dict(tr.group_by("ADEP_mvt").agg(pl.col("off").median()).iter_rows())
        print(f"  rule fold {k}: bias " + ", ".join(f"{a} {bias[a]:+.1f}" for a in RULE_APTS if a in bias))
        te = E.with_row_index("i").filter(pl.col("elig") & (pl.col("cvfold") == k)).with_columns(
            (pl.col("t_mvt") - pl.col("t_push") + pl.col("ADEP_mvt").replace_strict(bias, default=None))
            .round(0).alias("y_a"))
        te = te.filter((pl.col("y_a") - pl.col("b")).abs() <= GUARD)
        c[te["i"].to_numpy()] = (te["y_a"] - te["b"]).to_numpy()
    return c


def main() -> None:
    L.mem_gate()
    head("S0  data")
    D = load_labelled()
    F = pl.read_parquet(os.path.join(OUT, "feat_2025.parquet")).drop("ADEP_mvt", "day")
    D = D.join(F, on="MVT_ID_mvt", how="left").with_columns(pl.col("icao").is_not_null().alias("has"))
    chk = D.filter(pl.col("has")).select(
        (pl.col("t_mvt") - pl.col("MVT_TIME_UTC_mvt").dt.epoch("ms") / 1000).abs().max().alias("max|t_mvt-MVT|"))
    print("ADS-B rows", int(D["has"].sum()), "of", D.height, "|", chk.row(0, named=True))
    days = sorted(D["day"].unique().to_list())
    assert len(days) == 62 and all(d[:7] in ("2025-01", "2025-07") for d in days), days
    fold_of = assign_folds(days)
    D = D.with_columns(pl.col("day").replace_strict(fold_of).alias("cvfold"),
                       pl.col("day").replace_strict({d: i for i, d in enumerate(days)}).alias("day_idx"))
    print("folds:", {k: sorted(d for d, f in fold_of.items() if f == k) for k in range(K)})
    X, feats, cats = L.design(D, "b")
    print(f"{len(feats)} features:", feats)

    head("S1  leave-days-out folds")
    c_learn = np.zeros(D.height)
    has = D["has"].to_numpy()
    yv, bv = D["y"].to_numpy(), D["b"].to_numpy()
    r = np.clip(yv - bv, -CLIP_Y, CLIP_Y)
    cv_fold = D["cvfold"].to_numpy()
    day_arr = D["day"].to_numpy()
    Xn = X.select(feats).to_numpy().astype(np.float64)
    best_its, imp = [], np.zeros(len(feats))
    for k in range(K):
        L.mem_gate()
        tr_days = [d for d in days if fold_of[d] != k]
        val_days = inner_split(tr_days, SEED + 1 + k)
        is_tr = has & (cv_fold != k)
        is_val = is_tr & np.isin(day_arr, list(val_days))
        is_fit = is_tr & ~is_val
        is_te = has & (cv_fold == k)
        dtr = lgb.Dataset(Xn[is_fit], r[is_fit], feature_name=feats, categorical_feature=cats, free_raw_data=True)
        dva = lgb.Dataset(Xn[is_val], r[is_val], reference=dtr)
        m = lgb.train(PARAMS, dtr, num_boost_round=5000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        best_its.append(int(m.best_iteration))
        imp += m.feature_importance("gain", iteration=m.best_iteration)
        c_learn[is_te] = np.clip(m.predict(Xn[is_te], num_iteration=m.best_iteration), -CLIP_C, CLIP_C)
        dk = (yv[cv_fold == k] - bv[cv_fold == k]) ** 2 - (yv[cv_fold == k] - bv[cv_fold == k] - c_learn[cv_fold == k]) ** 2
        print(f"fold {k}: fit {is_fit.sum()} / es {is_val.sum()} ({len(val_days)} days) / test {is_te.sum()} ADS-B rows; "
              f"best_iter {m.best_iteration}; test-fold gain (all rows) {dk.mean():.1f} s^2", flush=True)

    head("S2  rule override on the same rows (bias leave-days-out)")
    c_rule = rule_corr(D, fold_of)

    head("S3  results (gain = mean over ALL labelled rows of (y-b)^2 - (y-b-c)^2, s^2; day-block bootstrap)")
    e0 = yv - bv
    d_l = e0 ** 2 - (e0 - c_learn) ** 2
    d_r = e0 ** 2 - (e0 - c_rule) ** 2
    di = D["day_idx"].to_numpy()
    nd = len(days)
    res = {}
    rmse0 = float(np.sqrt((e0 ** 2).mean()))
    for name, d, c in (("learned", d_l, c_learn), ("rule", d_r, c_rule), ("learned - rule", d_l - d_r, None)):
        g, lo, hi = boot(d, di, nd)
        res[name] = {"gain": g, "lo": lo, "hi": hi}
        extra = ""
        if c is not None:
            rm = float(np.sqrt(((e0 - c) ** 2).mean()))
            res[name].update(rmse_base=rmse0, rmse_new=rm, n_changed=int((c != 0).sum()))
            extra = f" | RMSE {rmse0:.3f} -> {rm:.3f} ({rm - rmse0:+.3f} s) | rows changed {int((c != 0).sum())}"
        print(f"{name:15s} gain {g:9.1f}  CI95 [{lo:9.1f}, {hi:9.1f}]{extra}")
    adopt = res["learned"]["lo"] > 0 and res["learned - rule"]["lo"] > 0
    print(f"\nDECISION (pre-registered): learned CI low > 0: {res['learned']['lo'] > 0}; "
          f"paired (learned - rule) CI low > 0: {res['learned - rule']['lo'] > 0} -> {'ADOPT' if adopt else 'REJECT'}")

    print("\nper airport (gain over that airport's rows, s^2; RMSE change s):")
    apt = D["ADEP_mvt"].to_numpy()
    rows = []
    for a in sorted(set(apt)):
        m = apt == a
        gl, ll, hl = boot(d_l[m], di[m], nd)
        gr, lr, hr = boot(d_r[m], di[m], nd)
        gd, ld, hd = boot(d_l[m] - d_r[m], di[m], nd)
        r0 = np.sqrt((e0[m] ** 2).mean())
        rows.append({"ADEP": a, "n": int(m.sum()), "ads_b_rows": int(has[m].sum()),
                     "learned": round(gl, 1), "l_lo": round(ll, 1), "l_hi": round(hl, 1),
                     "rule": round(gr, 1), "r_lo": round(lr, 1), "r_hi": round(hr, 1),
                     "diff": round(gd, 1), "d_lo": round(ld, 1), "d_hi": round(hd, 1),
                     "rmse_base": round(float(r0), 2),
                     "d_rmse_learned": round(float(np.sqrt(((e0[m] - c_learn[m]) ** 2).mean()) - r0), 2),
                     "d_rmse_rule": round(float(np.sqrt(((e0[m] - c_rule[m]) ** 2).mean()) - r0), 2),
                     "chg_learned": int((c_learn[m] != 0).sum()), "chg_rule": int((c_rule[m] != 0).sum())})
    PA = pl.DataFrame(rows)
    print(PA)
    print("\nper month (info):")
    for mo in ("2025-01", "2025-07"):
        m = np.char.startswith(day_arr.astype(str), mo)
        print(f"  {mo}: learned {d_l[m].mean():.1f}, rule {d_r[m].mean():.1f}, diff {(d_l[m] - d_r[m]).mean():.1f}")
    print("\ncorrection distribution on ADS-B rows (learned):",
          {q: round(float(np.quantile(c_learn[has], q)), 1) for q in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)},
          "| |c| mean", round(float(np.abs(c_learn[has]).mean()), 1), "| at clip", int((np.abs(c_learn) >= CLIP_C).sum()))
    order = np.argsort(-imp)
    print("\ntop 20 features by gain (summed over folds):")
    for i in order[:20]:
        print(f"  {feats[i]:28s} {imp[i] / imp.sum():.3f}")
    print("\nbest iterations", best_its, "median", int(np.median(best_its)))

    D.select("MVT_ID_mvt", "day", "cvfold", "ADEP_mvt", "y", "b", "has").with_columns(
        pl.Series("c_learned", c_learn), pl.Series("c_rule", c_rule)).write_parquet(os.path.join(OUT, "cv_rows.parquet"))
    json.dump({"best_iters": best_its, "final_rounds": int(np.median(best_its)), "results": res, "adopt": adopt,
               "per_airport": rows, "folds": fold_of, "params": PARAMS, "features": feats,
               "importance": {feats[i]: float(imp[i] / imp.sum()) for i in order}},
              open(os.path.join(OUT, "cv.json"), "w"), indent=1)
    print("wrote", os.path.join(OUT, "cv_rows.parquet"), os.path.join(OUT, "cv.json"))


if __name__ == "__main__":
    main()
