"""Variants V1 (MLP) and V2 (0.5 LGB + 0.5 MLP) of the learned ADS-B residual model, on the hybrid_blend_l2_pa base.

Pre-registration: `docs/competitor_research_2026-09-21/results/adsb-learned-residual.md` section 9.1 (committed
before this was run). Written from scratch. It uses the same features (`adsb_learn_features.design`), the same rows,
day-folds and inner early-stopping days (`adsb_learn_cv`), and the section 8 LGB as the reference.

    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 POLARS_MAX_THREADS=2 \
      .venv/bin/python scripts/adsb_learn_mlp.py cv    --base pa   # -> learn/cv_rows_v2.parquet, learn/cv_v2.json
    ... scripts/adsb_learn_mlp.py apply --base pa                  # builds outputs/hybrid_blend_l2_pa_adsb_v2.parquet
                                                                   # only if a variant passed the decision rule
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time

import numpy as np
import polars as pl
from sklearn.neural_network import MLPRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adsb_learn_cv as CV
import adsb_learn_features as L

assert CV.BASE_NAME == "pa", "section 9 is pre-registered on the hybrid_blend_l2_pa base (--base pa)"
ROOT = L.s1.REPO_DIR
SEEDS = [CV.SEED + i for i in range(3)]
Y_SCALE = 600.0
MAX_EPOCHS = 60
PATIENCE = 8
MLP_KW = dict(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", alpha=1e-4, batch_size=512,
              learning_rate_init=1e-3, shuffle=True)
CAT_LEVELS = {f"cat_{c}": len(v) for c, v in L.CATS.items()}
OUT_FILE = ROOT + "outputs/hybrid_blend_l2_pa_adsb_v2.parquet"
BASE_FILE = ROOT + "outputs/hybrid_blend_l2_pa.parquet"


class Prep:
    """Quantile clip (fit rows' 0.5-99.5 %), standardise, NaN -> 0 with missing indicators; one-hot categoricals."""

    def __init__(self, num: list[str], cats: list[str]):
        self.num, self.cats = num, cats

    def fit(self, X: np.ndarray, C: np.ndarray) -> "Prep":
        self.lo, self.hi = np.nanpercentile(X, [0.5, 99.5], axis=0)
        Xc = np.clip(X, self.lo, self.hi)
        self.mu = np.nanmean(Xc, axis=0)
        sd = np.nanstd(Xc, axis=0)
        self.sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
        self.mu = np.where(np.isfinite(self.mu), self.mu, 0.0)
        self.miss = np.where(np.isnan(X).any(axis=0))[0]
        return self

    def transform(self, X: np.ndarray, C: np.ndarray) -> np.ndarray:
        Z = (np.clip(X, self.lo, self.hi) - self.mu) / self.sd
        M = np.isnan(Z[:, self.miss]).astype(np.float32)
        Z = np.nan_to_num(Z, nan=0.0)
        oh = []
        for j, c in enumerate(self.cats):
            k = CAT_LEVELS[c]
            codes = np.where(C[:, j] < 0, k, C[:, j]).astype(int)  # missing / unknown -> level k
            oh.append(np.eye(k + 1, dtype=np.float32)[codes])
        return np.hstack([Z.astype(np.float32), M] + oh)


def split_cols(X: pl.DataFrame, feats: list[str], cats: list[str]):
    num = [f for f in feats if f not in cats]
    return num, X.select(num).to_numpy().astype(np.float64), X.select(cats).to_numpy().astype(np.int64)


def train_one(Xf, yf, Xv, yv, seed, fixed_epochs=None):
    m = MLPRegressor(random_state=seed, **MLP_KW)
    best, best_state, best_ep, bad = np.inf, None, 0, 0
    n_ep = fixed_epochs or MAX_EPOCHS
    for ep in range(1, n_ep + 1):
        m.partial_fit(Xf, yf)
        if fixed_epochs:
            continue
        mse = float(np.mean((m.predict(Xv) - yv) ** 2))
        if mse < best:
            best, best_ep, bad = mse, ep, 0
            best_state = (copy.deepcopy(m.coefs_), copy.deepcopy(m.intercepts_))
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if not fixed_epochs:
        m.coefs_, m.intercepts_ = best_state
    return m, best_ep


def cv() -> None:
    L.mem_gate()
    CV.head("S0  data (base = hybrid_blend_l2_pa OOF)")
    D = CV.load_labelled()
    F = pl.read_parquet(os.path.join(L.OUT, "feat_2025.parquet")).drop("ADEP_mvt", "day")
    D = D.join(F, on="MVT_ID_mvt", how="left").with_columns(pl.col("icao").is_not_null().alias("has"))
    ref = pl.read_parquet(os.path.join(L.OUT, "cv_rows_pa.parquet"))
    assert ref["MVT_ID_mvt"].equals(D["MVT_ID_mvt"]) and np.allclose(ref["b"].to_numpy(), D["b"].to_numpy())
    days = sorted(D["day"].unique().to_list())
    fold_of = CV.assign_folds(days)
    assert fold_of == json.load(open(os.path.join(L.OUT, "cv_pa.json")))["folds"]
    D = D.with_columns(pl.col("day").replace_strict(fold_of).alias("cvfold"),
                       pl.col("day").replace_strict({d: i for i, d in enumerate(days)}).alias("day_idx"))
    X, feats, cats = L.design(D, "b")
    num, Xn, Xc = split_cols(X, feats, cats)
    has = D["has"].to_numpy()
    yv, bv = D["y"].to_numpy(), D["b"].to_numpy()
    r = np.clip(yv - bv, -CV.CLIP_Y, CV.CLIP_Y) / Y_SCALE
    cvf, day_arr = D["cvfold"].to_numpy(), D["day"].to_numpy()
    c_mlp = np.zeros(D.height)
    best_eps = {s: [] for s in SEEDS}
    CV.head("S1  V1 MLP, leave-days-out")
    for k in range(CV.K):
        L.mem_gate()
        tr_days = [d for d in days if fold_of[d] != k]
        val_days = CV.inner_split(tr_days, CV.SEED + 1 + k)
        is_tr = has & (cvf != k)
        is_val = is_tr & np.isin(day_arr, list(val_days))
        is_fit = is_tr & ~is_val
        is_te = has & (cvf == k)
        P = Prep(num, cats).fit(Xn[is_fit], Xc[is_fit])
        Zf, Zv, Zt = (P.transform(Xn[m], Xc[m]) for m in (is_fit, is_val, is_te))
        preds = []
        for s in SEEDS:
            t0 = time.time()
            m, ep = train_one(Zf, r[is_fit], Zv, r[is_val], s)
            best_eps[s].append(ep)
            preds.append(m.predict(Zt) * Y_SCALE)
            print(f"fold {k} seed {s}: best epoch {ep}, {time.time() - t0:.0f} s, inputs {Zf.shape[1]}, "
                  f"total_rss {L.mem_mb():.0f} MB", flush=True)
        c_mlp[is_te] = np.clip(np.mean(preds, axis=0), -CV.CLIP_C, CV.CLIP_C)

    CV.head("S2  paired comparison vs the current learned LGB (section 8)")
    c_lgb = ref["c_learned"].to_numpy()
    c_v2 = np.clip(0.5 * c_lgb + 0.5 * c_mlp, -CV.CLIP_C, CV.CLIP_C)
    e0 = yv - bv
    di, nd = D["day_idx"].to_numpy(), len(days)
    se_ref = (e0 - c_lgb) ** 2
    res = {}
    print(f"RMSE: base {np.sqrt((e0 ** 2).mean()):.3f} | current LGB {np.sqrt(se_ref.mean()):.3f}")
    for name, c in (("V1 MLP", c_mlp), ("V2 0.5 LGB + 0.5 MLP", c_v2)):
        d = se_ref - (e0 - c) ** 2
        g, lo, hi = CV.boot(d, di, nd)
        gb, lob, hib = CV.boot(e0 ** 2 - (e0 - c) ** 2, di, nd)
        rm = float(np.sqrt(((e0 - c) ** 2).mean()))
        res[name] = {"paired_gain": g, "lo": lo, "hi": hi, "gain_vs_base": gb, "gb_lo": lob, "gb_hi": hib,
                     "rmse": rm, "d_rmse_vs_lgb": rm - float(np.sqrt(se_ref.mean())), "passes": lo > 0}
        print(f"{name:22s} paired gain vs LGB {g:8.1f} s^2 CI95 [{lo:8.1f}, {hi:8.1f}] | gain vs base {gb:8.1f} "
              f"[{lob:8.1f}, {hib:8.1f}] | RMSE {rm:.3f} ({rm - np.sqrt(se_ref.mean()):+.3f} vs LGB) | "
              f"{'PASSES' if lo > 0 else 'does not pass'}")
    passing = [n for n, v in res.items() if v["passes"]]
    pick = max(passing, key=lambda n: res[n]["paired_gain"]) if passing else None
    print(f"\nDECISION (9.1): passing {passing or 'none'} -> {'ADOPT ' + pick if pick else 'keep the current model'}")
    apt = D["ADEP_mvt"].to_numpy()
    print("\nper airport (info): paired gain vs LGB, s^2")
    for a in sorted(set(apt)):
        m = apt == a
        print(f"  {a}: n {m.sum():6d}  V1 {np.mean(se_ref[m] - (e0[m] - c_mlp[m]) ** 2):9.1f}  "
              f"V2 {np.mean(se_ref[m] - (e0[m] - c_v2[m]) ** 2):9.1f}")
    print("corr(c_LGB, c_MLP) on ADS-B rows", round(float(np.corrcoef(c_lgb[has], c_mlp[has])[0, 1]), 3))
    D.select("MVT_ID_mvt", "day", "cvfold", "ADEP_mvt", "y", "b", "has").with_columns(
        pl.Series("c_lgb", c_lgb), pl.Series("c_mlp", c_mlp), pl.Series("c_v2", c_v2)).write_parquet(
        os.path.join(L.OUT, "cv_rows_v2.parquet"))
    json.dump({"results": res, "pick": pick, "best_epochs": {str(s): v for s, v in best_eps.items()},
               "final_epochs": {str(s): int(np.median(v)) for s, v in best_eps.items()}},
              open(os.path.join(L.OUT, "cv_v2.json"), "w"), indent=1)
    print("best epochs", best_eps)


def apply() -> None:
    import lightgbm as lgb

    L.mem_gate()
    cj = json.load(open(os.path.join(L.OUT, "cv_v2.json")))
    pick = cj["pick"]
    if not pick:
        print("no variant passed the pre-registered rule: nothing is built")
        return
    CV.head(f"S0  final fit of V1 on all 62 days (adopted: {pick}); epochs per seed {cj['final_epochs']}")
    D = CV.load_labelled()
    F = pl.read_parquet(os.path.join(L.OUT, "feat_2025.parquet")).drop("ADEP_mvt", "day")
    D = D.join(F, on="MVT_ID_mvt", how="inner")
    X, feats, cats = L.design(D, "b")
    num, Xn, Xc = split_cols(X, feats, cats)
    r = np.clip(D["y"].to_numpy() - D["b"].to_numpy(), -CV.CLIP_Y, CV.CLIP_Y) / Y_SCALE
    P = Prep(num, cats).fit(Xn, Xc)
    Z = P.transform(Xn, Xc)
    models = []
    for s in SEEDS:
        m, _ = train_one(Z, r, None, None, s, fixed_epochs=cj["final_epochs"][str(s)])
        models.append(m)
        print(f"seed {s}: {cj['final_epochs'][str(s)]} epochs, total_rss {L.mem_mb():.0f} MB", flush=True)

    CV.head("S1  apply to 2026 scored rows")
    tpl = pl.read_parquet(ROOT + "data/raw/submitting.parquet")
    base = pl.read_parquet(BASE_FILE)
    assert base.schema == tpl.schema and base.height == tpl.height and (base["MVT_ID_mvt"] == tpl["MVT_ID_mvt"]).all()
    rk = pl.scan_parquet(ROOT + "data/raw/ranking.parquet").select("MVT_ID_mvt", "ADEP_mvt").collect()
    S = base.rename({"TAXITIME_SEC_mvt": "base"}).join(rk, on="MVT_ID_mvt", how="left", maintain_order="left")
    F26 = pl.read_parquet(os.path.join(L.OUT, "feat_2026.parquet"))
    G = F26.drop("ADEP_mvt").join(S, on="MVT_ID_mvt", how="inner").with_columns(
        pl.col("base").cast(pl.Float64).alias("b"))
    X26, f26, _ = L.design(G, "b")
    assert f26 == feats
    _, Xn26, Xc26 = split_cols(X26, feats, cats)
    c_mlp = np.clip(np.mean([m.predict(P.transform(Xn26, Xc26)) for m in models], axis=0) * Y_SCALE,
                    -CV.CLIP_C, CV.CLIP_C)
    lgbm = lgb.Booster(model_file=os.path.join(L.OUT, "final_model_pa.txt"))
    c_lgb = np.clip(lgbm.predict(X26.select(feats).to_numpy().astype(np.float64)), -CV.CLIP_C, CV.CLIP_C)
    c = c_mlp if pick.startswith("V1") else np.clip(0.5 * c_lgb + 0.5 * c_mlp, -CV.CLIP_C, CV.CLIP_C)
    G = G.with_columns(pl.Series("c", c), pl.Series("c_lgb", c_lgb)).with_columns(
        pl.when(pl.col("ADEP_mvt") == "LTFM").then(0.0).otherwise(pl.col("c")).alias("c"))
    G = G.with_columns((pl.col("b") + pl.col("c")).round(0).alias("new_f"))
    neg = int((G["new_f"] < 0).sum())
    G = G.with_columns(pl.when(pl.col("new_f") < 0).then(pl.col("base")).otherwise(pl.col("new_f").cast(pl.Int32))
                       .alias("new"))
    cand = (base.join(G.select("MVT_ID_mvt", "new"), on="MVT_ID_mvt", how="left", maintain_order="left")
            .with_columns(pl.coalesce("new", "TAXITIME_SEC_mvt").cast(pl.Int32).alias("TAXITIME_SEC_mvt"))
            .select(tpl.columns))
    print(f"2026 ADS-B rows {G.height}; negatives reverted {neg}; corr(c_LGB, c_MLP) "
          f"{np.corrcoef(c_lgb, c_mlp)[0, 1]:.3f}")

    CV.head("S2  validation")
    ids_adsb = G["MVT_ID_mvt"].implode()
    checks = {
        "rows == 344,841": cand.height == tpl.height == 344841,
        "schema == template": cand.schema == tpl.schema,
        "ids and order == template": bool((cand["MVT_ID_mvt"] == tpl["MVT_ID_mvt"]).all()),
        "no nulls": cand.null_count().sum_horizontal().item() == 0,
        "no negatives": int((cand["TAXITIME_SEC_mvt"] < 0).sum()) == 0,
        "rows without ADS-B == base": bool(
            (cand.filter(~pl.col("MVT_ID_mvt").is_in(ids_adsb))["TAXITIME_SEC_mvt"]
             == base.filter(~pl.col("MVT_ID_mvt").is_in(ids_adsb))["TAXITIME_SEC_mvt"]).all()),
        "LTFM unchanged": bool(cand.join(S, on="MVT_ID_mvt").filter(pl.col("ADEP_mvt") == "LTFM")
                               .select((pl.col("TAXITIME_SEC_mvt") == pl.col("base")).all()).item()),
    }
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    assert all(checks.values())
    cand.write_parquet(OUT_FILE)
    assert pl.read_parquet(OUT_FILE).equals(cand)
    M = cand.rename({"TAXITIME_SEC_mvt": "new"}).join(S, on="MVT_ID_mvt").with_columns(
        (pl.col("new") - pl.col("base")).alias("delta"))
    ch = M.filter(pl.col("delta") != 0)
    print(f"wrote {OUT_FILE}; changed vs base {ch.height}; max |delta| {ch['delta'].abs().max()}; median |delta| "
          f"{ch['delta'].abs().median()}; RMS delta {float(np.sqrt((ch['delta'].cast(pl.Float64) ** 2).mean())):.1f}")
    v32 = ROOT + "outputs/hybrid_blend_l2_pa_adsb_learned.parquet"
    W = M.join(pl.read_parquet(v32).rename({"TAXITIME_SEC_mvt": "v32"}), on="MVT_ID_mvt")
    dv = (W["new"] - W["v32"]).cast(pl.Float64)
    print(f"vs {v32}: rows differing {int((dv != 0).sum())}; RMS {float(np.sqrt((dv ** 2).mean())):.1f} s; "
          f"max |diff| {int(dv.abs().max())}")
    json.dump({"pick": pick, "negatives_reverted": neg, "changed": ch.height, "checks": checks},
              open(os.path.join(L.OUT, "apply_v2.json"), "w"), indent=1)


if __name__ == "__main__":
    {"cv": cv, "apply": apply}[sys.argv[1]]()
