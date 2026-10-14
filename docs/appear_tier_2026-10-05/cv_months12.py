"""plan_months12.md section 2-3: appear on 12 months of 2025 vs the production recipe (12-month plain residual +
4-month appear delta), V2 settings, fill stand table, base a2.

    SCRATCH=... POLARS_MAX_THREADS=2 .venv/bin/python docs/appear_tier_2026-10-05/cv_months12.py

Rows: Jan+Jul (jan / jul folds), Nov+Dec (forward), Aug-Oct (aso), Feb-Jun (fmj) OOF of the a2 base. K = 5 day groups
per month. Per fold, five models predict the same test days: P12 / A12 / S12 (plain / appear / shuffled appear, fit on
the training days of all 12 months) and P4 / A4 (plain / appear, fit on the training days of Jan, Jul, Nov, Dec only).
Writes cv_months12_a2.json (final rounds and features of P12 / A12) and cv_months12_rows_a2.parquet."""

import json
import os
import sys

sys.argv = [sys.argv[0], "--base", "a2"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv_months as CM  # noqa: E402

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

CV, L, AP = CM.CV, CM.L, CM.AP
V2 = dict(num_leaves=255, min_data_in_leaf=300, feature_fraction=0.7)
FOUR = ("2025-01", "2025-07", "2025-11", "2025-12")


def load_fold(fold: str) -> pl.DataFrame:
    """a2 base OOF of one custom fold (aso / fmj) from the *_asofmj runs."""
    def oof(exp_id: str):  # the run is named by the config's `name` (the base's), so its hash tells it apart
        f = os.path.join(CV.ROOT, f"experiments/oof_{exp_id}_folds.parquet")
        fo = pl.read_parquet(f).filter(pl.col("fold") == fold).select("MVT_ID_mvt")
        return fo, pl.read_parquet(f.replace("_folds.parquet", ".parquet"))
    fa, oa = oof("main_flseq_osm_nolirf_l2-f251478c1a")  # configs/main_flseq_osm_nolirf_l2_asofmj.yaml
    fb, ob = oof("main_flseq-87ea8fee1d")  # configs/main_flseq_asofmj.yaml
    assert fa.sort("MVT_ID_mvt").equals(fb.sort("MVT_ID_mvt")), f"{fold}: folds differ"
    ft = pl.read_parquet(os.path.join(CV.ROOT, "data/work_flseq/feat_train.parquet"),
                         columns=["MVT_ID_mvt", "TAXITIME_SEC_mvt", "mvt_minus_aobt3", "ADEP_mvt", "MVT_TIME_UTC_mvt"])
    D = (fa.join(ft, on="MVT_ID_mvt", how="left").join(oa.rename({"oof": "oa"}), on="MVT_ID_mvt", how="left")
         .join(ob.rename({"oof": "ob"}), on="MVT_ID_mvt", how="left").filter(pl.col("TAXITIME_SEC_mvt").is_not_null()))
    D = D.with_columns(pl.when(pl.col("mvt_minus_aobt3").is_not_null()).then("oa").otherwise("ob").alias("b"),
                       pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y"),
                       pl.col("MVT_TIME_UTC_mvt").dt.date().cast(pl.String).alias("day"))
    assert D["b"].null_count() == 0
    return D.select(CM.KEEP)


def load(variant: str = "fill") -> pl.DataFrame:
    """The 12 months with the appear columns of one own-stand table (fill here; fill12 / rand12 in plan_standmap12.md)."""
    D = pl.concat([CV.load_labelled().select(CM.KEEP), CM.load_novdec(), load_fold("aso"), load_fold("fmj")])
    assert D["MVT_ID_mvt"].n_unique() == D.height
    keys = ("2025", "2025nd", "2025m8")
    F = pl.concat([pl.read_parquet(os.path.join(L.OUT, f"feat_{k}.parquet")) for k in keys],
                  how="diagonal_relaxed").drop("ADEP_mvt", "day")
    D = D.join(F, on="MVT_ID_mvt", how="left").with_columns(pl.col("icao").is_not_null().alias("has"))
    T = pl.concat([pl.read_parquet(os.path.join(L.OUT, f"appear_{variant}_{k}.parquet")) for k in keys])
    return D.drop([c for c in AP.COLS if c in D.columns]).join(T, on="MVT_ID_mvt", how="left").sort("MVT_ID_mvt")


def folds(days: list[str]) -> tuple[list[str], dict[str, int]]:
    """K = 5 day groups within each month."""
    months = sorted({d[:7] for d in days})
    assert len(months) == 12, months
    rng = np.random.default_rng(CV.SEED)
    fold_of = {}
    for m in months:
        ds = sorted(d for d in days if d.startswith(m))
        for i, j in enumerate(rng.permutation(len(ds))):
            fold_of[ds[j]] = i % CV.K
    return months, fold_of


def inner(months: list[str], tr_days: list[str], seed: int) -> set[str]:
    """The early-stopping days: one in five of each month's training days."""
    r = np.random.default_rng(seed)
    val = set()
    for m in months:
        ds = sorted(d for d in tr_days if d.startswith(m))
        val |= {ds[j] for i, j in enumerate(r.permutation(len(ds))) if i % 5 == 0}
    return val


def main() -> None:
    CV.PARAMS = {**CV.PARAMS, **V2}
    D = load()
    days = sorted(D["day"].unique().to_list())
    months, fold_of = folds(days)

    D = D.with_columns(pl.col("day").replace_strict(fold_of).alias("cvfold"),
                       pl.col("day").replace_strict({d: i for i, d in enumerate(days)}).alias("day_idx"))
    S = CM.shuffled(D)
    print(f"rows {D.height:,} ({dict(D.group_by(pl.col('day').str.slice(0, 7)).len().sort('day').iter_rows())}); "
          f"ADS-B rows {int(D['has'].sum()):,}; appear rows {int(D['own_ok'].is_not_null().sum()):,}", flush=True)
    mats = {}
    for name, frame, appear in (("P", D, False), ("A", D, True), ("S", S, True)):
        X, feats, cats = L.design(frame, "b")
        if appear:
            X, feats = AP.design(X, feats, "b")
        mats[name] = (X.select(feats).to_numpy().astype(np.float32), feats, cats)
    y, b, has = D["y"].to_numpy(), D["b"].to_numpy(), D["has"].to_numpy()
    r = np.clip(y - b, -CV.CLIP_Y, CV.CLIP_Y)
    cvf, day_arr = D["cvfold"].to_numpy(), D["day"].to_numpy()
    four = np.isin(D["day"].str.slice(0, 7).to_numpy(), FOUR)
    arms = {"P12": ("P", None), "A12": ("A", None), "S12": ("S", None), "P4": ("P", four), "A4": ("A", four)}
    c = {k: np.zeros(D.height) for k in arms}
    its = {k: [] for k in arms}
    for k in range(CV.K):
        val_days = inner(months, [d for d in days if fold_of[d] != k], CV.SEED + 1 + k)
        is_te = has & (cvf == k)
        for arm, (mat, rowsel) in arms.items():
            Xn, feats, cats = mats[mat]
            is_tr = has & (cvf != k) & (rowsel if rowsel is not None else True)
            is_val = is_tr & np.isin(day_arr, list(val_days))
            is_fit = is_tr & ~is_val
            dtr = lgb.Dataset(Xn[is_fit], r[is_fit], feature_name=feats, categorical_feature=cats, free_raw_data=True)
            dva = lgb.Dataset(Xn[is_val], r[is_val], reference=dtr)
            m = lgb.train(CV.PARAMS, dtr, num_boost_round=5000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(200, verbose=False)])
            its[arm].append(int(m.best_iteration))
            c[arm][is_te] = np.clip(m.predict(Xn[is_te], num_iteration=m.best_iteration), -CV.CLIP_C, CV.CLIP_C)
            print(f"  fold {k} {arm}: fit {int(is_fit.sum())} es {int(is_val.sum())} best_iter {m.best_iteration}", flush=True)
    pred = {"final_12": b + c["A12"], "final_mix": b + c["P12"] + c["A4"] - c["P4"], "P12": b + c["P12"],
            "S12": b + c["S12"]}
    di, nd = D["day_idx"].to_numpy(), len(days)
    mo, apt = D["day"].str.slice(0, 7).to_numpy(), D["ADEP_mvt"].to_numpy()
    out = {}
    for a, b_ in (("final_mix", "final_12"), ("P12", "final_12"), ("S12", "final_12"), ("P12", "S12")):
        key = f"{b_} vs {a}"
        out[key] = {}
        groups = [("all rows", np.ones(len(y), bool))] + [(x, mo == x) for x in months] + \
                 [(x, apt == x) for x in sorted(set(apt))]
        for label, sel in groups:
            dd, lo, hi = CM._paired(y[sel], pred[a][sel], pred[b_][sel], di[sel], nd)
            out[key][label] = {"n": int(sel.sum()), "d_rmse": dd, "lo": lo, "hi": hi}
            print(f"  {key:24s} {label:>9s} n={int(sel.sum()):>7,}  {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    ok1 = out["final_12 vs final_mix"]["all rows"]["hi"] < 0
    ok2 = out["final_12 vs P12"]["all rows"]["hi"] < 0 and out["final_12 vs S12"]["all rows"]["hi"] < 0
    verdict = "ADOPT" if ok1 and ok2 else "REJECT"
    print(f"\nDECISION (plan_months12.md section 3): vs production recipe {ok1}, appear valid on 12 months {ok2} -> {verdict}")
    res = {"final_rounds": {k: int(np.median(v)) for k, v in its.items()}, "features": {"P12": mats["P"][1],
           "A12": mats["A"][1]}, "params": CV.PARAMS, "results": out, "verdict": verdict}
    json.dump(res, open(os.path.join(L.OUT, "cv_months12_a2.json"), "w"), indent=1)
    D.select("MVT_ID_mvt", "day", "ADEP_mvt", "y", "b", "has").with_columns(
        [pl.Series(k, v) for k, v in pred.items()]).write_parquet(os.path.join(L.OUT, "cv_months12_rows_a2.parquet"))


if __name__ == "__main__":
    main()
