"""4-month re-measure of the appear tier (plan.md section 6.2): plain4 / shuffle4 / appear4, all fit on Jan+Jul+Nov+Dec
2025, leave-days-out with K = 5 day groups per month, the residual's parameters, clip and early-stopping rule.

    SCRATCH=... POLARS_MAX_THREADS=2 .venv/bin/python docs/appear_tier_2026-10-05/cv_months.py

Base `a2` (v28 anchored L2 member on matched rows, main_flseq on unmatched): the jan/jul folds' OOF for Jan+Jul rows
(adsb_learn_cv.load_labelled) and the `forward` fold's OOF (fit on Jan-Oct) for Nov+Dec rows. ADS-B features:
feat_2025 / appear_2025 and feat_2025nd / appear_2025nd (docs/appear_tier_2026-10-05/build_features.py 2025 2025nd).
Writes $SCRATCH/learn/cv_months_a2.json (final rounds and feature lists per arm) and cv_months_rows_a2.parquet."""

import json
import os
import sys

sys.argv = [sys.argv[0], "--base", "a2"]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import adsb_appear as AP  # noqa: E402
import adsb_learn_cv as CV  # noqa: E402
import adsb_learn_features as L  # noqa: E402
from adsb_opdi import _paired  # noqa: E402

MONTHS = ("2025-01", "2025-07", "2025-11", "2025-12")
KEEP = ["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "mvt_minus_aobt3", "b", "y", "day"]


def load_novdec() -> pl.DataFrame:
    fa = pl.read_parquet(os.path.join(CV.ROOT, f"experiments/{CV.H2_A}_folds.parquet")).filter(
        pl.col("pooled") & (pl.col("fold") == "forward"))
    fb = pl.read_parquet(os.path.join(CV.ROOT, f"experiments/{CV.H2_B}_folds.parquet")).filter(
        pl.col("pooled") & (pl.col("fold") == "forward"))
    assert fa.sort("MVT_ID_mvt").equals(fb.sort("MVT_ID_mvt")), "forward folds differ"
    ft = pl.read_parquet(os.path.join(CV.ROOT, "data/work_flseq/feat_train.parquet"),
                         columns=["MVT_ID_mvt", "TAXITIME_SEC_mvt", "mvt_minus_aobt3", "ADEP_mvt", "MVT_TIME_UTC_mvt"])
    oa = pl.read_parquet(os.path.join(CV.ROOT, f"experiments/{CV.H2_A}.parquet")).rename({"oof": "oa"})
    ob = pl.read_parquet(os.path.join(CV.ROOT, f"experiments/{CV.H2_B}.parquet")).rename({"oof": "ob"})
    D = fa.select("MVT_ID_mvt").join(ft, on="MVT_ID_mvt", how="left").join(oa, on="MVT_ID_mvt", how="left").join(
        ob, on="MVT_ID_mvt", how="left")
    D = D.filter(pl.col("TAXITIME_SEC_mvt").is_not_null()).with_columns(
        pl.when(pl.col("mvt_minus_aobt3").is_not_null()).then("oa").otherwise("ob").alias("b"),
        pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y"),
        pl.col("MVT_TIME_UTC_mvt").dt.date().cast(pl.String).alias("day"))
    assert D["b"].null_count() == 0
    return D.select(KEEP)


def load() -> pl.DataFrame:
    D = pl.concat([CV.load_labelled().select(KEEP), load_novdec()])
    F = pl.concat([pl.read_parquet(os.path.join(L.OUT, f"feat_{k}.parquet")) for k in ("2025", "2025nd")],
                  how="diagonal_relaxed").drop("ADEP_mvt", "day")
    D = D.join(F, on="MVT_ID_mvt", how="left").with_columns(pl.col("icao").is_not_null().alias("has"))
    T = pl.concat([pl.read_parquet(os.path.join(L.OUT, f"appear_{k}.parquet")) for k in ("2025", "2025nd")])
    D = D.drop([c for c in AP.COLS if c in D.columns]).join(T, on="MVT_ID_mvt", how="left")
    return D


def shuffled(D: pl.DataFrame) -> pl.DataFrame:
    mode = AP.MODE
    AP.MODE = "appearshuf"
    try:
        T = D.select("MVT_ID_mvt", *AP.COLS)
        T.write_parquet(os.path.join(L.OUT, "appear_tmp_months.parquet"))
        S = AP.add(D.select("MVT_ID_mvt", "ADEP_mvt", "t_mvt"), "tmp_months")
    finally:
        AP.MODE = mode
        os.remove(os.path.join(L.OUT, "appear_tmp_months.parquet"))
    return D.drop(list(AP.COLS)).join(S.select("MVT_ID_mvt", *AP.COLS), on="MVT_ID_mvt", how="left")


def folds(days: list[str]) -> dict[str, int]:
    rng = np.random.default_rng(CV.SEED)
    out = {}
    for m in MONTHS:
        ds = sorted(d for d in days if d.startswith(m))
        for i, j in enumerate(rng.permutation(len(ds))):
            out[ds[j]] = i % CV.K
    return out


def inner(train_days: list[str], seed: int) -> set[str]:
    rng = np.random.default_rng(seed)
    val = set()
    for m in MONTHS:
        ds = sorted(d for d in train_days if d.startswith(m))
        perm = rng.permutation(len(ds))
        val |= {ds[j] for i, j in enumerate(perm) if i % 5 == 0}
    return val


def run_arm(D: pl.DataFrame, appear: bool, fold_of: dict[str, int], days: list[str]):
    X, feats, cats = L.design(D, "b")
    if appear:
        X, feats = AP.design(X, feats, "b")
    Xn = X.select(feats).to_numpy().astype(np.float64)
    y, b, has = D["y"].to_numpy(), D["b"].to_numpy(), D["has"].to_numpy()
    r = np.clip(y - b, -CV.CLIP_Y, CV.CLIP_Y)
    cvf, day_arr = D["cvfold"].to_numpy(), D["day"].to_numpy()
    c, its = np.zeros(D.height), []
    for k in range(CV.K):
        tr_days = [d for d in days if fold_of[d] != k]
        val_days = inner(tr_days, CV.SEED + 1 + k)
        is_tr = has & (cvf != k)
        is_val = is_tr & np.isin(day_arr, list(val_days))
        is_fit, is_te = is_tr & ~is_val, has & (cvf == k)
        dtr = lgb.Dataset(Xn[is_fit], r[is_fit], feature_name=feats, categorical_feature=cats, free_raw_data=True)
        dva = lgb.Dataset(Xn[is_val], r[is_val], reference=dtr)
        m = lgb.train(CV.PARAMS, dtr, num_boost_round=5000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        its.append(int(m.best_iteration))
        c[is_te] = np.clip(m.predict(Xn[is_te], num_iteration=m.best_iteration), -CV.CLIP_C, CV.CLIP_C)
        print(f"  fold {k}: fit {int(is_fit.sum())} es {int(is_val.sum())} test {int(is_te.sum())} best_iter "
              f"{m.best_iteration}", flush=True)
    return c, its, feats


def main() -> None:
    D = load()
    days = sorted(D["day"].unique().to_list())
    assert {d[:7] for d in days} == set(MONTHS), sorted({d[:7] for d in days})
    fold_of = folds(days)
    D = D.with_columns(pl.col("day").replace_strict(fold_of).alias("cvfold"),
                       pl.col("day").replace_strict({d: i for i, d in enumerate(days)}).alias("day_idx"))
    print(f"rows {D.height:,} ({dict(D.group_by(pl.col('day').str.slice(0, 7)).len().sort('day').iter_rows())}); "
          f"days {len(days)}; ADS-B rows {int(D['has'].sum()):,}; appear rows {int(D['own_ok'].is_not_null().sum()):,}",
          flush=True)
    arms, res = {}, {"final_rounds": {}, "features": {}}
    for name, frame, appear in (("plain4", D, False), ("shuffle4", shuffled(D), True), ("appear4", D, True)):
        print(f"arm {name}", flush=True)
        c, its, feats = run_arm(frame, appear, fold_of, days)
        arms[name] = D["b"].to_numpy() + c
        res["final_rounds"][name] = int(np.median(its))
        res["features"][name] = feats
    y, di, nd = D["y"].to_numpy(), D["day_idx"].to_numpy(), len(days)
    mo, apt = D["day"].str.slice(0, 7).to_numpy(), D["ADEP_mvt"].to_numpy()
    out = {}
    print("\nRMSE diff (B - A), day-bootstrap CI95:")
    for a, b_ in (("plain4", "appear4"), ("shuffle4", "appear4"), ("plain4", "shuffle4")):
        key = f"{b_} vs {a}"
        out[key] = {}
        for label, m in [("all rows", np.ones(len(y), bool))] + [(x, mo == x) for x in MONTHS] + \
                        [(x, apt == x) for x in sorted(set(apt))]:
            dd, lo, hi = _paired(y[m], arms[a][m], arms[b_][m], di[m], nd)
            out[key][label] = {"n": int(m.sum()), "d_rmse": dd, "lo": lo, "hi": hi}
            print(f"  {key:22s} {label:>9s} n={int(m.sum()):>7,}  {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    ok_p = out["appear4 vs plain4"]["all rows"]["hi"] < 0
    ok_s = out["appear4 vs shuffle4"]["all rows"]["hi"] < 0
    verdict = "ADOPT" if ok_p and ok_s else "REJECT"
    print(f"\nDECISION (plan.md section 6.2): vs plain4 {ok_p}, vs shuffle4 {ok_s} -> {verdict}")
    res.update(results=out, verdict=verdict, folds=fold_of)
    json.dump(res, open(os.path.join(L.OUT, "cv_months_a2.json"), "w"), indent=1)
    D.select("MVT_ID_mvt", "day", "ADEP_mvt", "y", "b", "has").with_columns(
        [pl.Series(f"p_{k}", v) for k, v in arms.items()]).write_parquet(os.path.join(L.OUT, "cv_months_rows_a2.parquet"))
    print("wrote", os.path.join(L.OUT, "cv_months_a2.json"))


if __name__ == "__main__":
    main()
