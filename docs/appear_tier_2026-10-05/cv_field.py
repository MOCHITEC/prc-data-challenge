"""plan_field.md section 3-4: cur (stage 1 + appear fill) vs field (cur + travel-time field columns) vs fshuf (field
columns permuted), all with v39's V2 residual settings, on the 4-month a2 rows and folds of cv_months.py.

    SCRATCH=... POLARS_MAX_THREADS=2 .venv/bin/python docs/appear_tier_2026-10-05/cv_field.py

Writes cv_field_a2.json (final_rounds / features under plain4 = cur and appear4 = field, verdict) for apply_delta."""

import json
import os
import sys

sys.argv = [sys.argv[0], "--base", "a2"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv_months as CM  # noqa: E402
import cv_standmap as CS  # noqa: E402
import field as FD  # noqa: E402

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

V2 = dict(num_leaves=255, min_data_in_leaf=300, feature_fraction=0.7)
CV, L, AP = CM.CV, CM.L, CM.AP


def run_arm(D: pl.DataFrame, with_field: bool, fold_of: dict, days: list):
    X, feats, cats = L.design(D, "b")
    X, feats = AP.design(X, feats, "b")
    if with_field:
        X, feats = FD.design(X, feats, "b")
    Xn = X.select(feats).to_numpy().astype(np.float64)
    y, b, has = D["y"].to_numpy(), D["b"].to_numpy(), D["has"].to_numpy()
    r = np.clip(y - b, -CV.CLIP_Y, CV.CLIP_Y)
    cvf, day_arr = D["cvfold"].to_numpy(), D["day"].to_numpy()
    c, its = np.zeros(D.height), []
    for k in range(CV.K):
        tr_days = [d for d in days if fold_of[d] != k]
        val_days = CM.inner(tr_days, CV.SEED + 1 + k)
        is_tr = has & (cvf != k)
        is_val = is_tr & np.isin(day_arr, list(val_days))
        is_fit, is_te = is_tr & ~is_val, has & (cvf == k)
        dtr = lgb.Dataset(Xn[is_fit], r[is_fit], feature_name=feats, categorical_feature=cats, free_raw_data=True)
        dva = lgb.Dataset(Xn[is_val], r[is_val], reference=dtr)
        m = lgb.train(CV.PARAMS, dtr, num_boost_round=5000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        its.append(int(m.best_iteration))
        c[is_te] = np.clip(m.predict(Xn[is_te], num_iteration=m.best_iteration), -CV.CLIP_C, CV.CLIP_C)
        print(f"  fold {k}: best_iter {m.best_iteration}", flush=True)
    return c, its, feats


def main() -> None:
    CV.PARAMS = {**CV.PARAMS, **V2}
    D = CS.load("fill")
    T = pl.concat([pl.read_parquet(os.path.join(L.OUT, f"field_{k}.parquet")) for k in ("2025", "2025nd")])
    D = D.join(T, on="MVT_ID_mvt", how="left").sort("MVT_ID_mvt")
    days = sorted(D["day"].unique().to_list())
    fold_of = CM.folds(days)
    D = D.with_columns(pl.col("day").replace_strict(fold_of).alias("cvfold"),
                       pl.col("day").replace_strict({d: i for i, d in enumerate(days)}).alias("day_idx"))
    print(f"rows {D.height:,}; field rows {int(D['t_appear_fA'].is_not_null().sum()):,}", flush=True)
    preds, res = {}, {"final_rounds": {}, "features": {}, "params": CV.PARAMS}
    for name, frame, wf in (("cur", D, False), ("fshuf", FD.shuffled(D), True), ("field", D, True)):
        print(f"arm {name}", flush=True)
        c, its, feats = run_arm(frame, wf, fold_of, days)
        preds[name] = D["b"].to_numpy() + c
        res["final_rounds"][name] = int(np.median(its))
        res["features"][name] = feats
    y, di, nd = D["y"].to_numpy(), D["day_idx"].to_numpy(), len(days)
    apt = D["ADEP_mvt"].to_numpy()
    far = ((D["d_dep1_own"] > AP.R_IN) & D["t_move_lo"].is_null()).fill_null(False).to_numpy()
    runoff = D["t_run_fA"].is_not_null().to_numpy()
    out = {}
    for a, b_ in (("cur", "field"), ("fshuf", "field"), ("cur", "fshuf")):
        key = f"{b_} vs {a}"
        out[key] = {}
        groups = [("all rows", np.ones(len(y), bool)), ("far first fix", far), ("run off stand", runoff)]
        groups += [(x, apt == x) for x in sorted(set(apt))]
        for label, m in groups:
            dd, lo, hi = CM._paired(y[m], preds[a][m], preds[b_][m], di[m], nd)
            out[key][label] = {"n": int(m.sum()), "d_rmse": dd, "lo": lo, "hi": hi}
            print(f"  {key:16s} {label:>14s} n={int(m.sum()):>7,}  {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    ok = out["field vs cur"]["all rows"]["hi"] < 0 and out["field vs fshuf"]["all rows"]["hi"] < 0
    verdict = "ADOPT" if ok else "REJECT"
    print(f"\nDECISION (plan_field.md section 4): -> {verdict}")
    res["final_rounds"]["plain4"], res["final_rounds"]["appear4"] = res["final_rounds"]["cur"], res["final_rounds"]["field"]
    res["features"]["plain4"], res["features"]["appear4"] = res["features"]["cur"], res["features"]["field"]
    res.update(results=out, verdict=verdict)
    json.dump(res, open(os.path.join(L.OUT, "cv_field_a2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
