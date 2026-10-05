"""Unmatched-lane view of the learned ADS-B residual model, and variant U (a separate LGB for unmatched rows).

Base: hybrid_blend_l2_pa_opt (section 10 of `docs/competitor_research_2026-09-21/results/adsb-learned-residual.md`).
Variant U is pre-registered in section 11 of that note, committed before `cv` was run. Written from scratch; same
features (`adsb_learn_features.design`), parameters, clipping and day-folds as `adsb_learn_cv.py`.

    POLARS_MAX_THREADS=2 .venv/bin/python scripts/adsb_learn_u.py report --base opt  # section-10 CV, unmatched rows
    POLARS_MAX_THREADS=2 .venv/bin/python scripts/adsb_learn_u.py cv     --base opt  # variant U, paired vs current
    POLARS_MAX_THREADS=2 .venv/bin/python scripts/adsb_learn_u.py apply  --base opt  # only if U was adopted

Unmatched = `mvt_minus_aobt3` null in `data/work_flseq/feat_train.parquet` (2025) / `feat_rank.parquet` (2026), the
lane `prc.train.predict_fold` uses.
"""

from __future__ import annotations

import json
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adsb_learn_cv as CV
import adsb_learn_features as L

assert CV.BASE_NAME == "opt", "section 11 is pre-registered on the hybrid_blend_l2_pa_opt base (--base opt)"
ROOT = L.s1.REPO_DIR
OUT_FILE = ROOT + "outputs/hybrid_blend_l2_pa_opt_adsb_u.parquet"
BASE_FILE = ROOT + "outputs/hybrid_blend_l2_pa_opt.parquet"
CUR_FILE = ROOT + "outputs/hybrid_blend_l2_pa_opt_adsb_learned.parquet"

pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(220)


def load() -> pl.DataFrame:
    """The section-10 CV rows (current model's corrections), with the lane and the features."""
    R = pl.read_parquet(os.path.join(L.OUT, "cv_rows_opt.parquet"))
    lane = pl.read_parquet(ROOT + "data/work_flseq/feat_train.parquet", columns=["MVT_ID_mvt", "mvt_minus_aobt3"])
    D = R.join(lane, on="MVT_ID_mvt", how="left", maintain_order="left").with_columns(
        pl.col("mvt_minus_aobt3").is_null().alias("unm"))
    assert D.height == R.height == 344419
    days = sorted(D["day"].unique().to_list())
    return D.with_columns(pl.col("day").replace_strict({d: i for i, d in enumerate(days)}).alias("day_idx"))


def block(name, d, di, nd):
    g, lo, hi = CV.boot(d, di, nd)
    return f"{name}: {g:10.1f} s^2  CI95 [{lo:10.1f}, {hi:10.1f}]", (g, lo, hi)


def report() -> None:
    L.mem_gate()
    D = load()
    nd = D["day_idx"].n_unique()
    e0 = (D["y"] - D["b"]).to_numpy()
    c = D["c_learned"].to_numpy()
    d = e0 ** 2 - (e0 - c) ** 2
    U = D["unm"].to_numpy()
    di = D["day_idx"].to_numpy()
    tot0 = float((e0 ** 2).sum())
    print(f"rows {D.height}; unmatched {U.sum()} ({100 * U.mean():.2f} %), of which with ADS-B "
          f"{int((U & D['has'].to_numpy()).sum())}")
    print(f"share of base squared error on unmatched rows: {(e0[U] ** 2).sum() / tot0:.3f}; after the learned "
          f"model: {((e0 - c)[U] ** 2).sum() / ((e0 - c) ** 2).sum():.3f}")
    for lab, m in (("all rows", np.ones_like(U)), ("matched", ~U), ("unmatched", U)):
        txt, _ = block(f"  gain {lab:9s} (mean over that lane's rows)", d[m], di[m], nd)
        print(f"{txt} | n {m.sum():6d} | RMSE {np.sqrt((e0[m] ** 2).mean()):8.2f} -> "
              f"{np.sqrt(((e0 - c)[m] ** 2).mean()):8.2f}")
    txt, _ = block("  unmatched contribution to the pooled gain (sum over unmatched / all rows)",
                   np.where(U, d, 0.0), di, nd)
    print(txt)
    print("\nunmatched rows by airport:")
    A = D.with_columns(pl.Series("d", d), pl.Series("e0", e0), pl.Series("e1", e0 - c)).filter(pl.col("unm"))
    rows = []
    for a in sorted(A["ADEP_mvt"].unique().to_list()):
        X = A.filter(pl.col("ADEP_mvt") == a)
        g, lo, hi = CV.boot(X["d"].to_numpy(), X["day_idx"].to_numpy(), nd)
        rows.append({"ADEP": a, "n": X.height, "ads_b": int(X["has"].sum()),
                     "rmse_before": float(np.sqrt((X["e0"] ** 2).mean())),
                     "rmse_after": float(np.sqrt((X["e1"] ** 2).mean())),
                     "gain": g, "lo": lo, "hi": hi})
    print(pl.DataFrame(rows))


def cv() -> None:
    L.mem_gate()
    CV.head("S0  data")
    D0 = CV.load_labelled()
    F = pl.read_parquet(os.path.join(L.OUT, "feat_2025.parquet")).drop("ADEP_mvt", "day")
    D0 = D0.join(F, on="MVT_ID_mvt", how="left").with_columns(pl.col("icao").is_not_null().alias("has"))
    R = load()
    assert D0["MVT_ID_mvt"].equals(R["MVT_ID_mvt"]) and np.allclose(D0["b"].to_numpy(), R["b"].to_numpy())
    days = sorted(D0["day"].unique().to_list())
    fold_of = CV.assign_folds(days)
    assert fold_of == json.load(open(os.path.join(L.OUT, "cv_opt.json")))["folds"]
    D0 = D0.with_columns(pl.col("day").replace_strict(fold_of).alias("cvfold"))
    X, feats, cats = L.design(D0, "b")
    Xn = X.select(feats).to_numpy().astype(np.float64)
    has = D0["has"].to_numpy()
    unm = D0["mvt_minus_aobt3"].is_null().to_numpy()
    assert (unm == R["unm"].to_numpy()).all()
    yv, bv = D0["y"].to_numpy(), D0["b"].to_numpy()
    r = np.clip(yv - bv, -CV.CLIP_Y, CV.CLIP_Y)
    cvf, day_arr = D0["cvfold"].to_numpy(), D0["day"].to_numpy()
    print(f"unmatched rows {unm.sum()}, with ADS-B {int((unm & has).sum())}")
    CV.head("S1  variant U: LGB on unmatched ADS-B rows only, leave-days-out")
    c_u = np.zeros(D0.height)
    best = []
    for k in range(CV.K):
        L.mem_gate()
        tr_days = [d for d in days if fold_of[d] != k]
        val_days = CV.inner_split(tr_days, CV.SEED + 1 + k)
        is_tr = has & unm & (cvf != k)
        is_val = is_tr & np.isin(day_arr, list(val_days))
        is_fit = is_tr & ~is_val
        is_te = has & unm & (cvf == k)
        dtr = lgb.Dataset(Xn[is_fit], r[is_fit], feature_name=feats, categorical_feature=cats)
        dva = lgb.Dataset(Xn[is_val], r[is_val], reference=dtr)
        m = lgb.train(CV.PARAMS, dtr, num_boost_round=5000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        best.append(int(m.best_iteration))
        c_u[is_te] = np.clip(m.predict(Xn[is_te], num_iteration=m.best_iteration), -CV.CLIP_C, CV.CLIP_C)
        print(f"fold {k}: fit {is_fit.sum()} / es {is_val.sum()} / test {is_te.sum()}; best_iter {m.best_iteration}",
              flush=True)
    CV.head("S2  paired comparison vs the current learned model (section 10)")
    c_cur = R["c_learned"].to_numpy()
    c_new = np.where(unm, c_u, c_cur)
    e0 = yv - bv
    di = R["day_idx"].to_numpy()
    nd = len(days)
    d = (e0 - c_cur) ** 2 - (e0 - c_new) ** 2  # > 0: U is better
    g, lo, hi = CV.boot(d, di, nd)
    rm_c, rm_n = float(np.sqrt(((e0 - c_cur) ** 2).mean())), float(np.sqrt(((e0 - c_new) ** 2).mean()))
    print(f"pooled (all {D0.height} rows): paired gain {g:.2f} s^2 CI95 [{lo:.2f}, {hi:.2f}] | RMSE {rm_c:.3f} -> "
          f"{rm_n:.3f} ({rm_n - rm_c:+.3f} s)")
    gu, lou, hiu = CV.boot(d[unm], di[unm], nd)
    print(f"unmatched only ({unm.sum()} rows): paired gain {gu:.1f} s^2 CI95 [{lou:.1f}, {hiu:.1f}] | RMSE "
          f"{np.sqrt(((e0 - c_cur)[unm] ** 2).mean()):.2f} -> {np.sqrt(((e0 - c_new)[unm] ** 2).mean()):.2f} "
          f"(base {np.sqrt((e0[unm] ** 2).mean()):.2f})")
    gb, lob, hib = CV.boot((e0 ** 2 - (e0 - c_u) ** 2)[unm], di[unm], nd)
    print(f"U vs no correction on unmatched rows: {gb:.1f} s^2 CI95 [{lob:.1f}, {hib:.1f}]")
    adopt = lo > 0
    print(f"\nDECISION (11): paired CI lower bound {lo:.2f} > 0: {adopt} -> {'ADOPT U' if adopt else 'keep the current model'}")
    apt = D0["ADEP_mvt"].to_numpy()
    print("per airport, unmatched rows (info): n, paired gain s^2")
    for a in sorted(set(apt[unm])):
        mm = unm & (apt == a)
        print(f"  {a}: n {mm.sum():5d}  ADS-B {int((mm & has).sum()):5d}  gain {d[mm].mean():10.1f}")
    print("best iterations", best, "median", int(np.median(best)))
    json.dump({"best_iters": best, "final_rounds": int(np.median(best)), "pooled": [g, lo, hi],
               "unmatched": [gu, lou, hiu], "u_vs_none_unmatched": [gb, lob, hib], "rmse_cur": rm_c, "rmse_new": rm_n,
               "adopt": adopt}, open(os.path.join(L.OUT, "cv_u.json"), "w"), indent=1)
    R.select("MVT_ID_mvt", "unm").with_columns(pl.Series("c_u", c_u), pl.Series("c_new", c_new)).write_parquet(
        os.path.join(L.OUT, "cv_rows_u.parquet"))


def apply() -> None:
    L.mem_gate()
    cj = json.load(open(os.path.join(L.OUT, "cv_u.json")))
    if not cj["adopt"]:
        print("U was not adopted: nothing is built")
        return
    D = CV.load_labelled()
    F = pl.read_parquet(os.path.join(L.OUT, "feat_2025.parquet")).drop("ADEP_mvt", "day")
    D = D.join(F, on="MVT_ID_mvt", how="inner").filter(pl.col("mvt_minus_aobt3").is_null())
    X, feats, cats = L.design(D, "b")
    r = np.clip(D["y"].to_numpy() - D["b"].to_numpy(), -CV.CLIP_Y, CV.CLIP_Y)
    mu = lgb.train(CV.PARAMS, lgb.Dataset(X.select(feats).to_numpy().astype(np.float64), r, feature_name=feats,
                                          categorical_feature=cats), num_boost_round=cj["final_rounds"])
    mu.save_model(os.path.join(L.OUT, "final_model_u.txt"))
    print(f"U final fit: {D.height} unmatched ADS-B rows, {cj['final_rounds']} rounds")
    tpl = pl.read_parquet(ROOT + "data/raw/submitting.parquet")
    base = pl.read_parquet(BASE_FILE)
    cur = pl.read_parquet(CUR_FILE)
    assert base["MVT_ID_mvt"].equals(tpl["MVT_ID_mvt"]) and cur["MVT_ID_mvt"].equals(tpl["MVT_ID_mvt"])
    rk = pl.scan_parquet(ROOT + "data/raw/ranking.parquet").select("MVT_ID_mvt", "ADEP_mvt").collect()
    lane = pl.read_parquet(ROOT + "data/work_flseq/feat_rank.parquet", columns=["MVT_ID_mvt", "mvt_minus_aobt3"])
    S = (base.rename({"TAXITIME_SEC_mvt": "base"}).join(rk, on="MVT_ID_mvt", how="left", maintain_order="left")
         .join(lane, on="MVT_ID_mvt", how="left", maintain_order="left"))
    print("2026 unmatched rows", int(S["mvt_minus_aobt3"].is_null().sum()))
    F26 = pl.read_parquet(os.path.join(L.OUT, "feat_2026.parquet"))
    G = (F26.drop("ADEP_mvt").join(S, on="MVT_ID_mvt", how="inner").filter(pl.col("mvt_minus_aobt3").is_null())
         .with_columns(pl.col("base").cast(pl.Float64).alias("b")))
    X26, f26, _ = L.design(G, "b")
    assert f26 == feats
    c = np.clip(mu.predict(X26.select(feats).to_numpy().astype(np.float64)), -CV.CLIP_C, CV.CLIP_C)
    G = G.with_columns(pl.Series("c", c)).with_columns(
        pl.when(pl.col("ADEP_mvt") == "LTFM").then(0.0).otherwise(pl.col("c")).alias("c"))
    G = G.with_columns((pl.col("b") + pl.col("c")).round(0).alias("new_f"))
    neg = int((G["new_f"] < 0).sum())
    G = G.with_columns(pl.when(pl.col("new_f") < 0).then(pl.col("base")).otherwise(pl.col("new_f").cast(pl.Int32))
                       .alias("new"))
    # matched rows: the current model's file; unmatched ADS-B rows: U; other unmatched rows: base (as in the current)
    cand = (cur.join(G.select("MVT_ID_mvt", "new"), on="MVT_ID_mvt", how="left", maintain_order="left")
            .join(S.select("MVT_ID_mvt", "base", "mvt_minus_aobt3"), on="MVT_ID_mvt", how="left", maintain_order="left")
            .with_columns(pl.when(pl.col("mvt_minus_aobt3").is_null()).then(pl.coalesce("new", "base"))
                          .otherwise(pl.col("TAXITIME_SEC_mvt")).cast(pl.Int32).alias("TAXITIME_SEC_mvt"))
            .select(tpl.columns))
    checks = {
        "rows == 344,841": cand.height == tpl.height == 344841,
        "schema == template": cand.schema == tpl.schema,
        "ids and order == template": bool((cand["MVT_ID_mvt"] == tpl["MVT_ID_mvt"]).all()),
        "no nulls": cand.null_count().sum_horizontal().item() == 0,
        "no negatives": int((cand["TAXITIME_SEC_mvt"] < 0).sum()) == 0,
        "matched rows == current file": bool(
            (cand["TAXITIME_SEC_mvt"] == cur["TAXITIME_SEC_mvt"]).filter(S["mvt_minus_aobt3"].is_not_null()).all()),
    }
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    assert all(checks.values())
    cand.write_parquet(OUT_FILE)
    dv = (cand["TAXITIME_SEC_mvt"] - cur["TAXITIME_SEC_mvt"]).cast(pl.Float64)
    print(f"wrote {OUT_FILE}; negatives reverted {neg}; rows differing from the current file {int((dv != 0).sum())}, "
          f"max |diff| {int(dv.abs().max())}")


if __name__ == "__main__":
    {"report": report, "cv": cv, "apply": apply}[sys.argv[1]]()
