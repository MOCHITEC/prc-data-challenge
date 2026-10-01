"""Investigation item 14 `robust-loss-harmful` (plan.md §1), loss-function part only (clipping/shrink are measured
in results/tail-postprocessing-cap-shrink.md).

Model B (the ~22,470 unmatched training rows) refitted with robust objectives, on the same three folds
(jan / jul / forward) whose predictions form experiments/oof_main-48c19bc830.parquet, against the reproduced L2
baseline. Data path identical to src/prc/train.py::predict_fold (see the docstring of mms-mixture-reparam.py):
routing rule mvt_minus_aobt3 null; encodings fitted on the training side of each fold; cols_b + CATEGORICALS;
_matrix; _inner_holdout_mask; TrainConfig from configs/main.yaml as pipeline.py resolves it, then
.for_unmatched(); num_threads 2; n_seeds refits averaged; predictions clipped at 0.

Variants (each replaces only `objective`/`alpha`, or the target; the early-stopping metric is LightGBM's default
for that objective, i.e. what `params: {objective: ...}` in a config would give):
  L2 baseline (reproduced; reused from the item-9 run's per-row file when present, refitted otherwise)
  huber, alpha = LightGBM default (0.9 -> a 0.9 s quadratic zone, effectively L1 on this label scale)
  huber, alpha = 5000
  huber, alpha = 5000, early stopping on l2 (extra: isolates the metric choice from the objective)
  quantile, alpha = 0.5 (conditional median)
  log1p target: objective regression on log1p(max(y, 0)), prediction expm1 (240 negative labels are floored at 0
  for the target only; RMSE is always against the raw label)
Evaluation: unmatched OOF RMSE per fold / pooled / per airport; worst-K removed (K = 20, 200) by the compare.py
convention with paired bootstrap CI; own-worst-K; LIRF vs non-LIRF; LIRF ms >= 2 h by f10; rollover rows.

POLARS_MAX_THREADS=1, LightGBM num_threads 2, one fit at a time. Runtime ~15-25 min.
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
BASE_IN = f"{SCRATCH}/mms-mixture-reparam_preds.parquet"  # item 9's per-row file (has p_base)
PRED_OUT = f"{SCRATCH}/robust-loss-harmful_preds.parquet"
FOLDS = ("jan", "jul", "forward")
MS = "mvt_minus_sched"
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


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
log("model-B params (for_unmatched):", CFG_B.lgb_params(), "num_rounds", CFG_B.num_rounds, "n_seeds", CFG_B.n_seeds,
    "early_stopping", CFG_B.early_stopping, "inner_holdout_months", CFG_B.inner_holdout_months)

VARIANTS = {
    # name: (extra params, target transform, inverse transform)
    "huber a=0.9(default)": ({"objective": "huber"}, None, None),
    "huber a=5000": ({"objective": "huber", "alpha": 5000.0}, None, None),
    "huber a=5000, es=l2": ({"objective": "huber", "alpha": 5000.0, "metric": "l2"}, None, None),
    "quantile a=0.5": ({"objective": "quantile", "alpha": 0.5}, None, None),
    "log1p target": ({"objective": "regression"}, lambda v: np.log1p(np.clip(v, 0.0, None)), np.expm1),
}

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
log(f"training rows with label {enc_all.height:,}; unmatched (routing) {unm.height:,}; negative labels {int((unm[LABEL] < 0).sum())} "
    f"(min {unm[LABEL].min()}), zero labels {int((unm[LABEL] == 0).sum())}")
months = sorted(enc_all["ym"].unique().to_list())
folds = {f.name: f for f in month_block_folds(months)}

base_prev = None
if os.path.exists(BASE_IN):
    base_prev = pl.read_parquet(BASE_IN).select(["MVT_ID_mvt", "p_base"])
    log(f"reusing the reproduced L2 baseline from {BASE_IN} ({base_prev.height:,} rows)")


def fit_b(u_tr, u_tg, y, cfg_b, cols_b, cat_cols, extra_params=None):
    """Faithful copy of train._fit for one target frame (see mms-mixture-reparam.py)."""
    params = dict(cfg_b.lgb_params(), **(extra_params or {}))
    hold = _inner_holdout_mask(u_tr, cfg_b.inner_holdout_months)
    stacked = pl.concat([f.select(cols_b + cat_cols) for f in (u_tr, u_tg)], how="vertical_relaxed")
    Xall, cols = _matrix(stacked, cols_b, cat_cols)
    del stacked
    cat_idx = [cols.index(c) for c in cat_cols]
    a, b = 0, u_tr.height
    rounds, best, n_hold = cfg_b.num_rounds, None, int(hold.sum())
    if 0 < n_hold < u_tr.height:
        Xtrain = Xall[a:b]
        dtr = lgb.Dataset(Xtrain[~hold], label=y[~hold], categorical_feature=cat_idx, params=params)
        dva = lgb.Dataset(Xtrain[hold], label=y[hold], categorical_feature=cat_idx, reference=dtr)
        booster = lgb.train(params, dtr, num_boost_round=cfg_b.num_rounds, valid_sets=[dva],
                            callbacks=[lgb.log_evaluation(0), lgb.early_stopping(cfg_b.early_stopping, verbose=False)])
        best = int(booster.best_iteration or cfg_b.num_rounds)
        rounds = int(min(cfg_b.num_rounds, max(10, round(best * u_tr.height / max(1, u_tr.height - n_hold)))))
        del dtr, dva, booster, Xtrain
    dfull = lgb.Dataset(Xall[a:b], label=y, categorical_feature=cat_idx, free_raw_data=False, params=params)
    acc = np.zeros(u_tg.height)
    n_seeds = max(1, cfg_b.n_seeds)
    for k in range(n_seeds):
        p_k = dict(params, seed=cfg_b.seed + k, bagging_seed=cfg_b.seed + k, feature_fraction_seed=cfg_b.seed + k)
        model = lgb.train(p_k, dfull, num_boost_round=rounds, callbacks=[lgb.log_evaluation(0)])
        acc += model.predict(Xall[b:])
    return acc / n_seeds, {"rounds": rounds, "best_iter": best, "n_hold": n_hold}


parts, fold_info = [], {}
for fname in FOLDS:
    f = folds[fname]
    log(f"===== fold {fname}: valid {f.valid_months}, train {len(f.train_months)} months")
    enc = fit_encodings(enc_all.filter(pl.col("ym").is_in(list(f.train_months))), CFG)
    u_tr = attach_encodings(unm.filter(pl.col("ym").is_in(list(f.train_months))), enc)
    u_tg = attach_encodings(unm.filter(pl.col("ym").is_in(list(f.valid_months))), enc)
    num_cols = visible_numeric_columns(u_tr, CFG)
    cat_cols = [c for c in CATEGORICALS if c in u_tr.columns and c not in set(CFG.exclude_features)] if CFG.use_categoricals else []
    cols_b = [c for c in num_cols if not c.startswith(("mvt_minus_aobt3", "aobt3", "lobt_minus", "mvt_minus_lobt",
                                                        "mvt_minus_eobt1", "mvt_minus_iobt"))]
    y_tr = u_tr[LABEL].to_numpy().astype(float)
    y_tg = u_tg[LABEL].to_numpy().astype(float)
    info = {"n_train": u_tr.height, "n_valid": u_tg.height, "n_num": len(cols_b), "n_cat": len(cat_cols)}
    out = {"MVT_ID_mvt": u_tg["MVT_ID_mvt"], "fold": fname, "ADEP_mvt": u_tg["ADEP_mvt"], "y": y_tg,
           "ms": u_tg[MS].to_numpy().astype(float), "lirf": u_tg["lirf"].to_numpy(), "f10": u_tg["f10"].to_numpy()}

    if base_prev is not None:
        pb = u_tg.select("MVT_ID_mvt").join(base_prev, on="MVT_ID_mvt", how="left", maintain_order="left")["p_base"]
        assert pb.null_count() == 0
        out["p_base"] = pb.to_numpy()
        info["base_rounds"] = "reused"
    else:
        p0, i0 = fit_b(u_tr, u_tg, y_tr, CFG_B, cols_b, cat_cols)
        out["p_base"] = p0
        info["base_rounds"], info["base_best"] = i0["rounds"], i0["best_iter"]
        log(f"baseline rounds {i0['rounds']} RMSE {rmse(y_tg, np.clip(p0, 0, None)):.1f}")

    for name, (extra, fwd, inv) in VARIANTS.items():
        t = fwd(y_tr) if fwd else y_tr
        g, i = fit_b(u_tr, u_tg, t, CFG_B, cols_b, cat_cols, extra)
        p = inv(g) if inv else g
        out[name] = p
        info[f"{name} rounds"], info[f"{name} best"] = i["rounds"], i["best_iter"]
        log(f"{name:24s} rounds {i['rounds']:5d} (best {i['best_iter']}) RMSE {rmse(y_tg, np.clip(p, 0, None)):.1f}")
    parts.append(pl.DataFrame(out))
    fold_info[fname] = info

pred = pl.concat(parts)
oof = pl.scan_parquet(OOF).select(["MVT_ID_mvt", "oof"]).collect()
pred = pred.join(oof, on="MVT_ID_mvt", how="left", maintain_order="left")
assert pred["oof"].null_count() == 0 and pred.height == 8378
pred.write_parquet(PRED_OUT)
log("predictions written:", PRED_OUT)

y = pred["y"].to_numpy()
ms = pred["ms"].to_numpy()
lirf = pred["lirf"].to_numpy()
f10 = pred["f10"].to_numpy()
folds_arr = pred["fold"].to_numpy()
adep = pred["ADEP_mvt"].to_numpy()
clip = lambda v: np.clip(v, 0.0, None)
V = {"oof_file": pred["oof"].to_numpy(), "base(repro) L2": clip(pred["p_base"].to_numpy())}
for name in VARIANTS:
    V[name] = clip(pred[name].to_numpy())
base = V["base(repro) L2"]

print("\n=== fold info ===")
for k, v in fold_info.items():
    print(k, v)

print("\n=== unmatched OOF RMSE per fold and pooled ===")
print(f"{'variant':24s} " + " ".join(f"{fn:>9s}" for fn in FOLDS) + f" {'pooled':>9s} {'LIRF':>9s} {'nonLIRF':>9s} {'LIRF ms>=2h':>11s} {'own-20':>8s} {'own-200':>8s} {'bias':>8s} {'pred med':>8s}")
for name, p in V.items():
    se = (y - p) ** 2
    order = np.argsort(-se)
    l2 = lirf & (ms >= 7200)
    print(f"{name:24s} " + " ".join(f"{rmse(y[folds_arr == fn], p[folds_arr == fn]):9.1f}" for fn in FOLDS)
          + f" {rmse(y, p):9.1f} {rmse(y[lirf], p[lirf]):9.1f} {rmse(y[~lirf], p[~lirf]):9.1f} {rmse(y[l2], p[l2]):11.1f} "
          f"{rmse(y[order[20:]], p[order[20:]]):8.1f} {rmse(y[order[200:]], p[order[200:]]):8.1f} {np.mean(p - y):+8.1f} {np.median(p):8.1f}")

print("\n=== per airport pooled RMSE ===")
airports = sorted(set(adep))
print(f"{'variant':24s} " + " ".join(f"{a:>8s}" for a in airports))
for name, p in V.items():
    pa = per_airport(y, p, adep)
    print(f"{name:24s} " + " ".join(f"{pa[a]:8.1f}" for a in airports))
print("n per airport:", {a: int((adep == a).sum()) for a in airports})

print("\n=== worst-K removed (compare.py convention), variant vs base(repro) L2; diff<0 = variant better ===")
print(f"{'variant':24s} {'K':>5s} {'base':>9s} {'variant':>9s} {'diff':>9s} {'CI95':>22s} {'P(var better)':>14s}")
for name, p in V.items():
    if name == "base(repro) L2":
        continue
    for r in robustness_curve(y, base, p, drop_levels=(0, 20, 200), n_boot=500, seed=0):
        print(f"{name:24s} {r['dropped']:5d} {r['rmse_a']:9.1f} {r['rmse_b']:9.1f} {r['diff']:+9.1f} [{r['ci95'][0]:+9.1f},{r['ci95'][1]:+9.1f}] {r['b_better_prob']:14.3f}")

print("\n=== rows with normal labels (y < 2 h) vs tail labels (y >= 2 h): RMSE ===")
for name, p in V.items():
    m = y >= 7200
    print(f"{name:24s} y<2h n {int((~m).sum())} RMSE {rmse(y[~m], p[~m]):8.1f} | y>=2h n {int(m.sum())} RMSE {rmse(y[m], p[m]):9.1f} | y>=4h RMSE {rmse(y[y >= 14400], p[y >= 14400]):9.1f} (n {int((y >= 14400).sum())}) | rollover y>=72000 RMSE {rmse(y[y >= 72000], p[y >= 72000]):9.1f} (n {int((y >= 72000).sum())})")

print("\n=== LIRF unmatched OOF rows with ms >= 2 h by f10: prediction medians and RMSE ===")
for flag in (True, False):
    m = lirf & (ms >= 7200) & (f10 == flag)
    print(f"f10={flag} n {m.sum()} y med {np.median(y[m]):.0f} ms med {np.median(ms[m]):.0f}")
    for name, p in V.items():
        print(f"   {name:24s} pred med {np.median(p[m]):8.0f} pred/ms med {np.median(p[m] / ms[m]):6.3f} RMSE {rmse(y[m], p[m]):9.1f}")

print("\n=== predictions >= 2 h / 4 h / 12 h (count) vs labels ===")
print(f"labels: >=2h {int((y >= 7200).sum())} >=4h {int((y >= 14400).sum())} >=12h {int((y >= 43200).sum())}")
for name, p in V.items():
    print(f"{name:24s} >=2h {int((p >= 7200).sum()):4d} >=4h {int((p >= 14400).sum()):3d} >=12h {int((p >= 43200).sum()):3d} | non-LIRF >=2h {int(((p >= 7200) & ~lirf).sum()):3d} >=4h {int(((p >= 14400) & ~lirf).sum()):3d} | max {p.max():9.0f}")

log("done")
