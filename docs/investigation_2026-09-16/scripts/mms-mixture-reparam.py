"""Investigation item 9 `mms-mixture-reparam` (plan.md §1): model-B fits on the ~22,470 unmatched training rows.

Reproduces model B's data path exactly as src/prc/train.py::predict_fold does it, without editing src/:
  * routing rule: mvt_minus_aobt3 null -> model B;
  * label-derived encodings fitted with fit_encodings on the training side of each fold (all rows, matched and
    unmatched, as predict_fold does) and joined with attach_encodings;
  * feature list = visible_numeric_columns minus the cols_b prefix exclusions, plus CATEGORICALS;
  * design matrix via _matrix on the stacked [train, target] frame; inner holdout via _inner_holdout_mask;
  * params = TrainConfig resolved from configs/main.yaml the way src/prc/pipeline.py does, then .for_unmatched();
    the only change is num_threads 2 (the OOF run used 4; deterministic + force_row_wise are set in both);
  * early stopping on the inner holdout, round count rescaled, refit on all rows under n_seeds seeds, averaged;
  * predictions clipped at 0 as _assemble does.
Folds: jan / jul / forward from prc.cv.month_block_folds (the three whose predictions make up
experiments/oof_main-48c19bc830.parquet: jul_fwd shares jul's validation months and is coalesced away there).

Steps
  (a) no-fit pre-check: corr(y, mvt_minus_sched) on LIRF unmatched vs LFPG unmatched (E H2), all airports listed;
  (0) baseline: model B as the code fits it, compared row by row with the OOF file;
  (A) variant A: target y - ms on LIRF rows only (ms added back at prediction), current target elsewhere;
  (A2) extra: target y - ms on every unmatched row (E H2's T = D - G form);
  (B) variant B: binary classifier P(filled | x), filled = |BLOCK - SCHED| <= 10 s, same cols_b / categoricals;
      prediction = P * ms + (1 - P) * baseline prediction; also hard P >= 0.5, LIRF-only gating, oracle P;
  (B2) extra: regression fitted on non-filled rows only, mixed with the same P (the hypothesis' "(1 - P) * taxi").
Evaluation: unmatched OOF RMSE per fold / pooled / per airport; worst-K rows removed (K = 20, 200) by the
compare.py convention (rows where either model errs most) with paired bootstrap CI; own-worst-K removed as well.

POLARS_MAX_THREADS=1, LightGBM num_threads 2, one fit at a time. Runtime ~20-30 min.
Writes per-row predictions to the scratchpad (item 14 reuses the baseline from there).
"""
import os
import sys
import time
from dataclasses import replace

os.environ["POLARS_MAX_THREADS"] = "1"
ROOT = "/home/user/prc-data-challenge"
sys.path.insert(0, f"{ROOT}/src")

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import yaml  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from prc.compare import robustness_curve  # noqa: E402
from prc.cv import month_block_folds, per_airport, rmse  # noqa: E402
from prc.features import CATEGORICALS  # noqa: E402
from prc.train import (  # noqa: E402
    LABEL,
    PROXY,
    TrainConfig,
    _inner_holdout_mask,
    _matrix,
    attach_encodings,
    fit_encodings,
    visible_numeric_columns,
)

pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(240)

FT = f"{ROOT}/data/work/feat_train.parquet"
OOF = f"{ROOT}/experiments/oof_main-48c19bc830.parquet"
SCRATCH = "/tmp/claude-0/-home-user-prc-data-challenge/ca703460-62c0-5c84-91fe-d76c3d0e1e20/scratchpad"
os.makedirs(SCRATCH, exist_ok=True)
PRED_OUT = f"{SCRATCH}/mms-mixture-reparam_preds.parquet"
FOLDS = ("jan", "jul", "forward")
MS = "mvt_minus_sched"
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ---------------------------------------------------------------- config, exactly as pipeline.main resolves it
def load_cfg() -> TrainConfig:
    spec = yaml.safe_load(open(f"{ROOT}/configs/main.yaml").read())
    fields = {k: v for k, v in spec.items() if k in TrainConfig.__dataclass_fields__}
    if "params" in fields:
        fields["params"] = {**TrainConfig().params, **fields["params"]}
    for k in ("folds", "exclude_features", "include_candidates"):
        if isinstance(fields.get(k), list):
            fields[k] = tuple(fields[k])
    cfg = TrainConfig(**fields)
    return replace(cfg, params={**cfg.params, "num_threads": 2})


CFG = load_cfg()
CFG_B = CFG.for_unmatched()
log("resolved TrainConfig:", {k: v for k, v in CFG.__dict__.items() if k != "params"})
log("model-B params (for_unmatched):", CFG_B.lgb_params(), "num_rounds", CFG_B.num_rounds, "n_seeds", CFG_B.n_seeds)

# ---------------------------------------------------------------- data (lazy, column-selected)
ENC_COLS = ["MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "hour_local", "AIRCRAFT_OPERATOR_flt", LABEL, PROXY,
            "MVT_TIME_UTC_mvt"]
lf = pl.scan_parquet(FT)
ym_expr = (pl.col("MVT_TIME_UTC_mvt").dt.year() * 100 + pl.col("MVT_TIME_UTC_mvt").dt.month()).alias("ym")
enc_all = lf.select(ENC_COLS).filter(pl.col(LABEL).is_not_null() & pl.col(LABEL).is_finite()).with_columns(ym_expr).collect()
unm = (
    lf.filter(pl.col(PROXY).is_null())
    .with_columns(
        ym_expr,
        ((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= 10).alias("f10"),
        (pl.col("ADEP_mvt") == "LIRF").alias("lirf"),
    )
    .collect()
)
assert unm[LABEL].null_count() == 0 and unm[MS].null_count() == 0
y_all = unm[LABEL].to_numpy().astype(float)
ms_all = unm[MS].to_numpy().astype(float)
f10_alt = np.abs(y_all - ms_all) <= 10
log(f"training rows with label {enc_all.height:,}; unmatched (routing) {unm.height:,}; f10 rows {int(unm['f10'].sum()):,}; "
    f"f10 == (|y-ms|<=10) on {int((unm['f10'].to_numpy() == f10_alt).sum()):,} rows")
months = sorted(enc_all["ym"].unique().to_list())
folds = {f.name: f for f in month_block_folds(months)}
log("folds available:", {k: (v.valid_months, len(v.train_months)) for k, v in folds.items()})

# ---------------------------------------------------------------- (a) pre-check, no fit
print("\n=== (a) corr(y, mvt_minus_sched) on unmatched training rows, per ADEP (routing definition) ===")
rows = []
for adep in sorted(unm["ADEP_mvt"].unique().to_list()):
    for sub, mask_expr in (("all", pl.lit(True)), ("ms>=2h", pl.col(MS) >= 7200), ("ms<2h", pl.col(MS) < 7200),
                           ("f10", pl.col("f10")), ("not f10", ~pl.col("f10"))):
        d = unm.filter((pl.col("ADEP_mvt") == adep) & mask_expr)
        if d.height < 3:
            continue
        y, m = d[LABEL].to_numpy().astype(float), d[MS].to_numpy().astype(float)
        pr = pearsonr(y, m)[0] if np.std(m) > 0 and np.std(y) > 0 else float("nan")
        sp = spearmanr(y, m)[0] if np.std(m) > 0 and np.std(y) > 0 else float("nan")
        slope = float(np.polyfit(m, y, 1)[0]) if np.std(m) > 0 else float("nan")
        rows.append({"ADEP": adep, "subset": sub, "n": d.height, "pearson": pr, "spearman": sp, "ols_slope": slope,
                     "y_med": float(np.median(y)), "ms_med": float(np.median(m)), "f10_rate": float(d["f10"].mean())})
corr_tbl = pl.DataFrame(rows)
print(corr_tbl.filter(pl.col("ADEP").is_in(["LIRF", "LFPG"])))
print("-- all airports, subset = all:")
print(corr_tbl.filter(pl.col("subset") == "all").sort("pearson", descending=True))
# LIRF matched for contrast
lm = lf.filter((pl.col("ADEP_mvt") == "LIRF") & pl.col(PROXY).is_not_null()).select([LABEL, MS]).collect()
print(f"LIRF matched (contrast): n {lm.height:,} pearson {pearsonr(lm[LABEL].to_numpy().astype(float), lm[MS].to_numpy().astype(float))[0]:.4f} "
      f"spearman {spearmanr(lm[LABEL].to_numpy(), lm[MS].to_numpy())[0]:.4f}")
del lm


# ---------------------------------------------------------------- fitting: a faithful copy of train._fit
def fit_b(u_tr: pl.DataFrame, u_tg: pl.DataFrame, y: np.ndarray, cfg_b: TrainConfig, cols_b: list[str],
          cat_cols: list[str], extra_params: dict | None = None) -> tuple[np.ndarray, dict]:
    """Early-stop on the inner holdout, rescale rounds, refit on all rows under n_seeds seeds, average.
    Returns predictions in model space for u_tg and info incl. holdout predictions (for AUC)."""
    params = dict(cfg_b.lgb_params(), **(extra_params or {}))
    hold = _inner_holdout_mask(u_tr, cfg_b.inner_holdout_months)
    stacked = pl.concat([f.select(cols_b + cat_cols) for f in (u_tr, u_tg)], how="vertical_relaxed")
    Xall, cols = _matrix(stacked, cols_b, cat_cols)
    del stacked
    cat_idx = [cols.index(c) for c in cat_cols]
    a, b = 0, u_tr.height
    rounds, best, n_hold, hold_pred = cfg_b.num_rounds, None, int(hold.sum()), None
    if 0 < n_hold < u_tr.height:
        Xtrain = Xall[a:b]
        dtr = lgb.Dataset(Xtrain[~hold], label=y[~hold], categorical_feature=cat_idx, params=params)
        dva = lgb.Dataset(Xtrain[hold], label=y[hold], categorical_feature=cat_idx, reference=dtr)
        booster = lgb.train(params, dtr, num_boost_round=cfg_b.num_rounds, valid_sets=[dva],
                            callbacks=[lgb.log_evaluation(0), lgb.early_stopping(cfg_b.early_stopping, verbose=False)])
        best = int(booster.best_iteration or cfg_b.num_rounds)
        rounds = int(min(cfg_b.num_rounds, max(10, round(best * u_tr.height / max(1, u_tr.height - n_hold)))))
        hold_pred = booster.predict(Xtrain[hold], num_iteration=best)
        del dtr, dva, booster, Xtrain
    dfull = lgb.Dataset(Xall[a:b], label=y, categorical_feature=cat_idx, free_raw_data=False, params=params)
    acc = np.zeros(u_tg.height)
    n_seeds = max(1, cfg_b.n_seeds)
    for k in range(n_seeds):
        p_k = dict(params, seed=cfg_b.seed + k, bagging_seed=cfg_b.seed + k, feature_fraction_seed=cfg_b.seed + k)
        model = lgb.train(p_k, dfull, num_boost_round=rounds, callbacks=[lgb.log_evaluation(0)])
        acc += model.predict(Xall[b:])
    return acc / n_seeds, {"rounds": rounds, "best_iter": best, "n_hold": n_hold, "hold_pred": hold_pred, "hold": hold,
                           "n_features": len(cols)}


# ---------------------------------------------------------------- per fold
parts, fold_info = [], {}
for fname in FOLDS:
    f = folds[fname]
    log(f"===== fold {fname}: valid {f.valid_months}, train {len(f.train_months)} months")
    tr_enc = enc_all.filter(pl.col("ym").is_in(list(f.train_months)))
    enc = fit_encodings(tr_enc, CFG)
    u_tr = attach_encodings(unm.filter(pl.col("ym").is_in(list(f.train_months))), enc)
    u_tg = attach_encodings(unm.filter(pl.col("ym").is_in(list(f.valid_months))), enc)
    del tr_enc
    num_cols = visible_numeric_columns(u_tr, CFG)
    cat_cols = [c for c in CATEGORICALS if c in u_tr.columns and c not in set(CFG.exclude_features)] if CFG.use_categoricals else []
    cols_b = [c for c in num_cols if not c.startswith(("mvt_minus_aobt3", "aobt3", "lobt_minus", "mvt_minus_lobt",
                                                        "mvt_minus_eobt1", "mvt_minus_iobt"))]
    assert MS in cols_b and "mvt_minus_sched_slog" in cols_b
    y_tr = u_tr[LABEL].to_numpy().astype(float)
    ms_tr = u_tr[MS].to_numpy().astype(float)
    lirf_tr = u_tr["lirf"].to_numpy()
    f10_tr = u_tr["f10"].to_numpy()
    y_tg = u_tg[LABEL].to_numpy().astype(float)
    ms_tg = u_tg[MS].to_numpy().astype(float)
    lirf_tg = u_tg["lirf"].to_numpy()
    f10_tg = u_tg["f10"].to_numpy()
    info = {"n_train": u_tr.height, "n_valid": u_tg.height, "n_num": len(cols_b), "n_cat": len(cat_cols),
            "train_lirf": int(lirf_tr.sum()), "train_f10": int(f10_tr.sum()), "train_f10_lirf": int((f10_tr & lirf_tr).sum())}
    log("fold shape:", info)

    # (0) baseline, exactly the code's fit
    p0, i0 = fit_b(u_tr, u_tg, y_tr, CFG_B, cols_b, cat_cols)
    info["base_rounds"], info["base_best"], info["n_hold"] = i0["rounds"], i0["best_iter"], i0["n_hold"]
    log(f"baseline rounds {i0['rounds']} (best {i0['best_iter']}, holdout {i0['n_hold']}) RMSE {rmse(y_tg, np.clip(p0, 0, None)):.1f}")

    # (A) target y - ms on LIRF rows only
    tA = y_tr - np.where(lirf_tr, ms_tr, 0.0)
    gA, iA = fit_b(u_tr, u_tg, tA, CFG_B, cols_b, cat_cols)
    pA = gA + np.where(lirf_tg, ms_tg, 0.0)
    info["A_rounds"], info["A_best"] = iA["rounds"], iA["best_iter"]
    log(f"variant A rounds {iA['rounds']} RMSE {rmse(y_tg, np.clip(pA, 0, None)):.1f}")

    # (A2) target y - ms on every unmatched row
    gA2, iA2 = fit_b(u_tr, u_tg, y_tr - ms_tr, CFG_B, cols_b, cat_cols)
    pA2 = gA2 + ms_tg
    info["A2_rounds"], info["A2_best"] = iA2["rounds"], iA2["best_iter"]
    log(f"variant A2 rounds {iA2['rounds']} RMSE {rmse(y_tg, np.clip(pA2, 0, None)):.1f}")

    # (B) classifier P(f10 | x)
    pP, iP = fit_b(u_tr, u_tg, f10_tr.astype(float), CFG_B, cols_b, cat_cols, {"objective": "binary"})
    auc_hold = roc_auc_score(f10_tr[iP["hold"]], iP["hold_pred"]) if iP["hold_pred"] is not None else float("nan")
    auc_val = roc_auc_score(f10_tg, pP)
    auc_val_lirf = roc_auc_score(f10_tg[lirf_tg], pP[lirf_tg]) if 0 < f10_tg[lirf_tg].sum() < lirf_tg.sum() else float("nan")
    auc_val_non = roc_auc_score(f10_tg[~lirf_tg], pP[~lirf_tg]) if 0 < f10_tg[~lirf_tg].sum() < (~lirf_tg).sum() else float("nan")
    info.update({"clf_rounds": iP["rounds"], "clf_best": iP["best_iter"], "auc_holdout": auc_hold, "auc_valid": auc_val,
                 "auc_valid_lirf": auc_val_lirf, "auc_valid_nonlirf": auc_val_non,
                 "valid_f10_rate": float(f10_tg.mean()), "valid_f10_rate_lirf": float(f10_tg[lirf_tg].mean()),
                 "P_mean": float(pP.mean()), "P_mean_lirf": float(pP[lirf_tg].mean())})
    log(f"classifier rounds {iP['rounds']} AUC holdout {auc_hold:.4f} valid {auc_val:.4f} (LIRF {auc_val_lirf:.4f}, non-LIRF {auc_val_non:.4f})")

    # (B2) regression on non-filled rows only
    nf = ~f10_tr
    pNF, iNF = fit_b(u_tr.filter(~pl.col("f10")), u_tg, y_tr[nf], CFG_B, cols_b, cat_cols)
    info["nofill_rounds"], info["nofill_best"], info["nofill_train"] = iNF["rounds"], iNF["best_iter"], int(nf.sum())
    log(f"non-filled regression rounds {iNF['rounds']} on {int(nf.sum()):,} rows")

    parts.append(pl.DataFrame({
        "MVT_ID_mvt": u_tg["MVT_ID_mvt"], "fold": fname, "ADEP_mvt": u_tg["ADEP_mvt"], "y": y_tg, "ms": ms_tg,
        "lirf": lirf_tg, "f10": f10_tg, "p_base": p0, "p_A": pA, "p_A2": pA2, "P": pP, "p_nofill": pNF,
    }))
    fold_info[fname] = info

pred = pl.concat(parts)
oof = pl.scan_parquet(OOF).select(["MVT_ID_mvt", "oof"]).collect()
pred = pred.join(oof, on="MVT_ID_mvt", how="left", maintain_order="left")
assert pred["oof"].null_count() == 0 and pred.height == 8378, (pred["oof"].null_count(), pred.height)
pred.write_parquet(PRED_OUT)
log("predictions written:", PRED_OUT)

# ---------------------------------------------------------------- variants (all clipped at 0 like _assemble)
y = pred["y"].to_numpy()
ms = pred["ms"].to_numpy()
lirf = pred["lirf"].to_numpy()
f10 = pred["f10"].to_numpy()
P = pred["P"].to_numpy()
clip = lambda v: np.clip(v, 0.0, None)
V = {
    "oof_file": pred["oof"].to_numpy(),
    "base(repro)": clip(pred["p_base"].to_numpy()),
    "A: y-ms@LIRF": clip(pred["p_A"].to_numpy()),
    "A2: y-ms@all": clip(pred["p_A2"].to_numpy()),
    "B: P*ms+(1-P)*base": clip(P * ms + (1 - P) * clip(pred["p_base"].to_numpy())),
    "B-hard: 1[P>=.5]": clip(np.where(P >= 0.5, ms, clip(pred["p_base"].to_numpy()))),
    "B-LIRF-only": clip(np.where(lirf, P * ms + (1 - P) * clip(pred["p_base"].to_numpy()), clip(pred["p_base"].to_numpy()))),
    "B2: P*ms+(1-P)*nofill": clip(P * ms + (1 - P) * clip(pred["p_nofill"].to_numpy())),
    "B2-LIRF-only": clip(np.where(lirf, P * ms + (1 - P) * clip(pred["p_nofill"].to_numpy()), clip(pred["p_base"].to_numpy()))),
    "oracleP*ms+(1-P)*base": clip(np.where(f10, ms, clip(pred["p_base"].to_numpy()))),
    "oracleP*ms+(1-P)*nofill": clip(np.where(f10, ms, clip(pred["p_nofill"].to_numpy()))),
    "nofill-reg alone": clip(pred["p_nofill"].to_numpy()),
}
base = V["base(repro)"]
folds_arr = pred["fold"].to_numpy()
adep = pred["ADEP_mvt"].to_numpy()

print("\n=== fold info ===")
for k, v in fold_info.items():
    print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()})

print("\n=== reproduction check: this run's baseline vs experiments/oof_main-48c19bc830.parquet on the 8,378 unmatched OOF rows ===")
d = base - V["oof_file"]
print(f"RMSE oof_file {rmse(y, V['oof_file']):.1f} | RMSE base(repro) {rmse(y, base):.1f} | RMS(base - oof) {np.sqrt(np.mean(d**2)):.1f} | "
      f"max|diff| {np.abs(d).max():.1f} | rows |diff|>1 {int((np.abs(d) > 1).sum())} | corr {np.corrcoef(base, V['oof_file'])[0,1]:.6f}")
for fn in FOLDS:
    m = folds_arr == fn
    dd = d[m]
    print(f"  {fn:8s} n {m.sum():5d} oof {rmse(y[m], V['oof_file'][m]):8.1f} repro {rmse(y[m], base[m]):8.1f} RMS diff {np.sqrt(np.mean(dd**2)):7.1f} max {np.abs(dd).max():8.1f}")

print("\n=== unmatched OOF RMSE per fold and pooled (all variants) ===")
hdr = f"{'variant':28s} " + " ".join(f"{fn:>9s}" for fn in FOLDS) + f" {'pooled':>9s} {'LIRF':>9s} {'nonLIRF':>9s} {'LIRF ms>=2h':>11s} {'own-20':>8s} {'own-200':>8s}"
print(hdr)
for name, p in V.items():
    se = (y - p) ** 2
    order = np.argsort(-se)
    own20 = rmse(y[order[20:]], p[order[20:]])
    own200 = rmse(y[order[200:]], p[order[200:]])
    l2 = lirf & (ms >= 7200)
    print(f"{name:28s} " + " ".join(f"{rmse(y[folds_arr == fn], p[folds_arr == fn]):9.1f}" for fn in FOLDS)
          + f" {rmse(y, p):9.1f} {rmse(y[lirf], p[lirf]):9.1f} {rmse(y[~lirf], p[~lirf]):9.1f} {rmse(y[l2], p[l2]):11.1f} {own20:8.1f} {own200:8.1f}")

print("\n=== per airport pooled RMSE ===")
airports = sorted(set(adep))
print(f"{'variant':28s} " + " ".join(f"{a:>8s}" for a in airports))
for name, p in V.items():
    pa = per_airport(y, p, adep)
    print(f"{name:28s} " + " ".join(f"{pa[a]:8.1f}" for a in airports))
print("n per airport:", {a: int((adep == a).sum()) for a in airports})

print("\n=== worst-K removed (compare.py convention: K rows where either model errs most), variant vs base(repro); diff<0 = variant better ===")
print(f"{'variant':28s} {'K':>5s} {'base':>9s} {'variant':>9s} {'diff':>9s} {'CI95':>22s} {'P(var better)':>14s}")
for name, p in V.items():
    if name == "base(repro)":
        continue
    for r in robustness_curve(y, base, p, drop_levels=(0, 20, 200), n_boot=500, seed=0):
        print(f"{name:28s} {r['dropped']:5d} {r['rmse_a']:9.1f} {r['rmse_b']:9.1f} {r['diff']:+9.1f} [{r['ci95'][0]:+9.1f},{r['ci95'][1]:+9.1f}] {r['b_better_prob']:14.3f}")

print("\n=== LIRF unmatched OOF rows with ms >= 2 h: prediction medians by f10 ===")
for flag in (True, False):
    m = lirf & (ms >= 7200) & (f10 == flag)
    print(f"f10={flag} n {m.sum()} y med {np.median(y[m]):.0f} ms med {np.median(ms[m]):.0f} P med {np.median(P[m]):.3f} | " +
          " ".join(f"{k}: {np.median(v[m]):.0f}" for k, v in V.items() if k in ("oof_file", "base(repro)", "A: y-ms@LIRF", "A2: y-ms@all", "B: P*ms+(1-P)*base", "B2: P*ms+(1-P)*nofill")))
    print("   RMSE: " + " ".join(f"{k}: {rmse(y[m], v[m]):.0f}" for k, v in V.items()))

print("\n=== rollover rows (y >= 72,000) and their predictions ===")
m = y >= 72000
for k, v in V.items():
    print(f"{k:28s} n {m.sum()} RMSE {rmse(y[m], v[m]):9.1f} SSE share of pooled {((y[m]-v[m])**2).sum()/((y-v)**2).sum():.3f}")

print("\n=== classifier calibration on validation rows: P bins vs observed f10 rate ===")
bins = np.array([0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0001])
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (P >= lo) & (P < hi)
    if m.sum():
        print(f"P in [{lo:.1f},{hi:.1f}) n {m.sum():5d} LIRF {int((m & lirf).sum()):4d} observed f10 {f10[m].mean():.3f} mean P {P[m].mean():.3f}")
print("\nP >= 0.5 rows by airport:", {a: int(((P >= 0.5) & (adep == a)).sum()) for a in airports})

print("\n=== predictions >= 4 h / >= 12 h (count) vs labels ===")
print(f"labels: >=4h {int((y >= 14400).sum())}, >=12h {int((y >= 43200).sum())}")
for k, v in V.items():
    print(f"{k:28s} >=4h {int((v >= 14400).sum()):3d} >=12h {int((v >= 43200).sum()):3d} non-LIRF >=4h {int(((v >= 14400) & ~lirf).sum()):3d}")

log("done")
