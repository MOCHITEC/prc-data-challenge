"""Investigation item 30 `shared-b-all-rows` (plan.md §1), which also settles item 31's open question.

Question: does fitting model B on ALL labelled training rows (the removed `shared_b` mode, git 5b90919,
deleted in ea04ec9) beat the current `separate` fit on the unmatched rows it predicts?

The script reproduces `src/prc/train.py::predict_fold`'s model-B branch without editing `src/`:
  * routing rule: `mvt_minus_aobt3` null -> model B;
  * encodings fitted with `fit_encodings` on the whole labelled training side of the fold, joined with
    `attach_encodings` (exactly as predict_fold does);
  * feature list = `visible_numeric_columns` minus the six cols_b prefix exclusions, plus CATEGORICALS;
  * design matrix via `_matrix` on the stacked [train, target] frame; inner holdout via `_inner_holdout_mask`;
  * early stopping on the inner holdout, round count rescaled to the refit size, refit under n_seeds seeds and
    averaged, prediction clipped at 0 as `_assemble` does;
  * config resolved from `configs/inv_baseline_forward.yaml` (= configs/main.yaml + folds [forward] +
    num_threads 2), the config of the logged Batch C baseline `inv_baseline_forward-67e76a5dfc`.

Fold: `forward` only (train Jan-Oct 2025 = 1,757,038 labelled rows, inner holdout October, evaluate Nov+Dec
2025 = 328,009 rows, of which 3,005 unmatched). A pooled fit is a 1.76M-row fit, so one fold is the budget.

Variants (four fits, never two at once, in this order):
  S-full   split baseline, model B on the unmatched training rows only, full cols_b  (reproduces the logged run)
  S-restr  the same rows, cols_b minus the four `_flt` categoricals that are 100 % null on unmatched rows
  P-restr  POOLED: model B on every labelled training row, restricted column list (the item's mandated variant)
  P-full   POOLED: every labelled training row, full cols_b (the literal removed `shared_b` implementation)

Matched predictions are identical across all four by construction (model A is untouched), so the whole-fold
numbers reuse the matched OOF column of `experiments/oof_inv_baseline_forward-67e76a5dfc.parquet`.

Reporting: unmatched RMSE overall / per airport / LIRF vs rest; `prc.compare.robustness_curve` at K = 0/20/200/2000
with the paired bootstrap and `prc.compare.verdict`; `month_bootstrap_rmse_diff`; tree counts and the first
splits of the pooled boosters (nm_matched / eobt1_null / lobt_null); per-row table for LIRF unmatched ms >= 12 h.

POLARS_MAX_THREADS=1, LightGBM num_threads 2, one fit at a time. Runtime ~60-100 min.
`SMOKE=1` in the environment is a self-test path only (8 rounds, 1 seed, pooled rows subsampled to 50k): it
exercises every code path in minutes and produces no number quoted in the report.
"""
import json
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

from prc.compare import DROP_LEVELS, month_bootstrap_rmse_diff, robustness_curve, verdict  # noqa: E402
from prc.cv import month_block_folds, per_airport, rmse  # noqa: E402
from prc.features import CATEGORICALS, numeric_feature_columns  # noqa: E402
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

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(25)
pl.Config.set_tbl_width_chars(240)

FT = f"{ROOT}/data/work/feat_train.parquet"
BASE_OOF = f"{ROOT}/experiments/oof_inv_baseline_forward-67e76a5dfc.parquet"
CFG_YAML = f"{ROOT}/configs/inv_baseline_forward.yaml"
SCRATCH = "/tmp/claude-0/-home-user-prc-data-challenge/ca703460-62c0-5c84-91fe-d76c3d0e1e20/scratchpad"
os.makedirs(SCRATCH, exist_ok=True)
PRED_OUT = f"{SCRATCH}/shared-b-all-rows_preds.parquet"
MS = "mvt_minus_sched"
FOLD = "forward"
# The four categoricals that shared-b-no-transfer.md §3 G finds 100 % null on unmatched rows (training and
# ranking); they are the only cols_b columns present in one regime and absent in the other.
NULL_ON_UNMATCHED_CATS = ["AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt", "WK_TBL_CAT_flt"]
REGIME_COLS = ["nm_matched", "eobt1_null", "lobt_null"]
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ---------------------------------------------------------------- config, exactly as pipeline.main resolves it
def load_cfg(path: str) -> TrainConfig:
    spec = yaml.safe_load(open(path).read())
    fields = {k: v for k, v in spec.items() if k in TrainConfig.__dataclass_fields__}
    if "params" in fields:
        fields["params"] = {**TrainConfig().params, **fields["params"]}
    for k in ("folds", "exclude_features", "include_candidates"):
        if isinstance(fields.get(k), list):
            fields[k] = tuple(fields[k])
    return TrainConfig(**fields)


SMOKE = bool(os.environ.get("SMOKE"))
CFG = load_cfg(CFG_YAML)
CFG_B = CFG.for_unmatched()
if SMOKE:
    CFG_B = replace(CFG_B, num_rounds=8, n_seeds=1)
log("resolved TrainConfig:", {k: v for k, v in CFG.__dict__.items() if k != "params"})
log("model-A params:", CFG.lgb_params(), "num_rounds", CFG.num_rounds)
log("model-B params (for_unmatched):", CFG_B.lgb_params(), "num_rounds", CFG_B.num_rounds)
log(
    "for_unmatched() == cfg on (num_leaves, min_data_in_leaf, num_rounds):",
    (CFG_B.params["num_leaves"], CFG_B.params["min_data_in_leaf"], CFG_B.num_rounds)
    == (CFG.params["num_leaves"], CFG.params["min_data_in_leaf"], CFG.num_rounds),
    "-> the removed shared_b's capacity choice (cfg vs cfg.for_unmatched) is not a difference under this config",
)

# ---------------------------------------------------------------- data
lf = pl.scan_parquet(FT)
sch = lf.collect_schema()
# Column ORDER matters: numeric_feature_columns walks the frame schema and feature_fraction 0.8 draws its
# subset by column index, so a re-ordered selection fits different trees. Keep the parquet's own order.
_wanted = (
    set(c for c, dt in sch.items() if dt.is_numeric())
    | set(CATEGORICALS)
    | {"MVT_ID_mvt", "MVT_TIME_UTC_mvt", "PHASE_mvt"}
)
keep = [c for c in sch.names() if c in _wanted]
ym_expr = (pl.col("MVT_TIME_UTC_mvt").dt.year() * 100 + pl.col("MVT_TIME_UTC_mvt").dt.month()).alias("ym")
feats = (
    lf.select(keep)
    .filter(pl.col(LABEL).is_not_null() & pl.col(LABEL).is_finite())
    .with_columns(ym_expr)
    .collect()
)
log(f"labelled training frame: {feats.height:,} rows x {feats.width} columns")
months = sorted(feats["ym"].unique().to_list())
folds = {f.name: f for f in month_block_folds(months)}
f = folds[FOLD]
log(f"fold {FOLD}: train {f.train_months} -> valid {f.valid_months}")

tr_raw = feats.filter(pl.col("ym").is_in(list(f.train_months)))
tg_raw = feats.filter(pl.col("ym").is_in(list(f.valid_months)))
del feats
log(f"train rows {tr_raw.height:,} (record says 1,757,038); valid rows {tg_raw.height:,} (record says 328,009)")

enc = fit_encodings(tr_raw, CFG)
tr = attach_encodings(tr_raw, enc)
tg = attach_encodings(tg_raw, enc)
del tr_raw, tg_raw

# `ym` is this script's own fold key, not a feature: the pipeline's month column is called `month` and
# visible_numeric_columns drops it by name, so `ym` has to be dropped here to keep the feature set identical.
num_cols = [c for c in visible_numeric_columns(tr, CFG) if c != "ym"]
cat_cols = [c for c in CATEGORICALS if c in tr.columns and c not in set(CFG.exclude_features)] if CFG.use_categoricals else []
COLS_B = [
    c
    for c in num_cols
    if not c.startswith(("mvt_minus_aobt3", "aobt3", "lobt_minus", "mvt_minus_lobt", "mvt_minus_eobt1", "mvt_minus_iobt"))
]
assert MS in COLS_B and "mvt_minus_sched_slog" in COLS_B and "ym" not in COLS_B
CAT_RESTR = [c for c in cat_cols if c not in NULL_ON_UNMATCHED_CATS]
log(f"frame numerics {len(numeric_feature_columns(tr))}, visible {len(num_cols)}, cols_b numeric {len(COLS_B)}, "
    f"categoricals {len(cat_cols)} -> full feature set {len(COLS_B) + len(cat_cols)}; "
    f"restricted {len(COLS_B) + len(CAT_RESTR)} (drops {NULL_ON_UNMATCHED_CATS})")
print("cols_b numeric:", COLS_B, flush=True)
print("categoricals (full):", cat_cols, flush=True)
print("categoricals (restricted):", CAT_RESTR, flush=True)

u_tr = tr.filter(pl.col(PROXY).is_null())
u_tg = tg.filter(pl.col(PROXY).is_null())
m_tg = tg.filter(pl.col(PROXY).is_not_null())
log(f"unmatched: train {u_tr.height:,}, valid {u_tg.height:,}; matched valid {m_tg.height:,}")

y_tg = u_tg[LABEL].to_numpy().astype(float)
ms_tg = u_tg[MS].to_numpy().astype(float)
adep_tg = u_tg["ADEP_mvt"].to_numpy()

# null rates of the four dropped categoricals, re-checked on this fold's frames
for c in NULL_ON_UNMATCHED_CATS + ["AIRCRAFT_TYPE_mvt"]:
    log(f"  {c}: null rate train-matched {tr.filter(pl.col(PROXY).is_not_null())[c].null_count() / max(1, tr.height - u_tr.height):.4f} "
        f"train-unmatched {u_tr[c].null_count() / max(1, u_tr.height):.4f} valid-unmatched {u_tg[c].null_count() / max(1, u_tg.height):.4f}")


# ---------------------------------------------------------------- fitting: a faithful copy of train._fit
def fit_b(train_rows: pl.DataFrame, target_rows: pl.DataFrame, cfg_b: TrainConfig, feature_cols: list[str],
          cats: list[str]) -> tuple[np.ndarray, dict, lgb.Booster]:
    """Model B's fit path. Returns (prediction on target_rows, info, the first-seed refit booster)."""
    params = cfg_b.lgb_params()
    hold = _inner_holdout_mask(train_rows, cfg_b.inner_holdout_months)
    stacked = pl.concat([f_.select(feature_cols + cats) for f_ in (train_rows, target_rows)], how="vertical_relaxed")
    Xall, cols = _matrix(stacked, feature_cols, cats)
    del stacked
    cat_idx = [cols.index(c) for c in cats]
    a, b = 0, train_rows.height
    y = train_rows[LABEL].to_numpy().astype(float)
    rounds, best, n_hold = cfg_b.num_rounds, None, int(hold.sum())
    if 0 < n_hold < train_rows.height:
        Xtrain = Xall[a:b]
        dtr = lgb.Dataset(Xtrain[~hold], label=y[~hold], categorical_feature=cat_idx, params=params)
        dva = lgb.Dataset(Xtrain[hold], label=y[hold], categorical_feature=cat_idx, reference=dtr)
        booster = lgb.train(params, dtr, num_boost_round=cfg_b.num_rounds, valid_sets=[dva],
                            callbacks=[lgb.log_evaluation(0), lgb.early_stopping(cfg_b.early_stopping, verbose=False)])
        best = int(booster.best_iteration or cfg_b.num_rounds)
        rounds = int(min(cfg_b.num_rounds, max(10, round(best * train_rows.height / max(1, train_rows.height - n_hold)))))
        del dtr, dva, booster, Xtrain
        log(f"    early stopping: best_iter {best} on {n_hold:,} holdout rows -> {rounds} rounds for the refit")
    dfull = lgb.Dataset(Xall[a:b], label=y, categorical_feature=cat_idx, free_raw_data=False, params=params)
    acc = np.zeros(target_rows.height)
    n_seeds = max(1, cfg_b.n_seeds)
    first = None
    for k in range(n_seeds):
        p_k = dict(params, seed=cfg_b.seed + k, bagging_seed=cfg_b.seed + k, feature_fraction_seed=cfg_b.seed + k)
        model = lgb.train(p_k, dfull, num_boost_round=rounds, callbacks=[lgb.log_evaluation(0)])
        acc += model.predict(Xall[b:])
        if k == 0:
            first = model
        log(f"    seed {cfg_b.seed + k}: {model.num_trees()} trees fitted")
    info = {"rounds": rounds, "best_iter": best, "n_hold": n_hold, "n_train": train_rows.height,
            "n_features": len(cols), "n_trees_per_seed": first.num_trees(), "n_seeds": n_seeds, "cols": cols}
    return acc / n_seeds, info, first


def describe_splits(model: lgb.Booster, tag: str, cols: list[str]) -> dict:
    """Tree counts, root splits, and where the regime indicators are used.

    `_matrix` hands LightGBM a bare numpy array, so the booster names features Column_<i>; `cols` is the
    column order it was built from and is used to put the real names back."""
    tp = model.trees_to_dataframe()
    tp["threshold"] = tp["threshold"].astype(str)  # categorical splits carry strings like "3||8"
    names = {f"Column_{i}": c for i, c in enumerate(cols)}
    tp["split_feature"] = tp["split_feature"].map(lambda v: names.get(v, v))
    t = pl.from_pandas(tp)
    n_trees = int(t["tree_index"].max()) + 1 if t.height else 0
    internal = t.filter(pl.col("split_feature").is_not_null())
    roots = internal.filter(pl.col("node_depth") == 1)
    print(f"\n--- first splits, {tag} ({n_trees} trees, {internal.height} internal nodes) ---", flush=True)
    print("root split feature counts over all trees:")
    print(roots.group_by("split_feature").agg(pl.len().alias("n_trees_as_root"), pl.col("split_gain").mean().alias("mean_gain"))
          .sort("n_trees_as_root", descending=True).head(15))
    print("tree 0, nodes at depth 1-3 (in tree order):")
    print(t.filter((pl.col("tree_index") == 0) & pl.col("split_feature").is_not_null() & (pl.col("node_depth") <= 3))
          .select(["node_index", "node_depth", "split_feature", "threshold", "split_gain", "count"])
          .sort("node_depth"))
    out = {"n_trees": n_trees, "n_internal": internal.height}
    for c in REGIME_COLS:
        sub = internal.filter(pl.col("split_feature") == c)
        trees_with = sub["tree_index"].n_unique()
        out[c] = {
            "n_splits": sub.height,
            "n_trees_with_split": int(trees_with),
            "n_trees_root": int(roots.filter(pl.col("split_feature") == c).height),
            "min_depth": int(sub["node_depth"].min()) if sub.height else None,
            "median_depth": float(sub["node_depth"].median()) if sub.height else None,
            "total_gain": float(sub["split_gain"].sum()) if sub.height else 0.0,
        }
        print(f"  {c}: {out[c]}", flush=True)
    tot_gain = float(internal["split_gain"].sum())
    out["regime_gain_share"] = sum(out[c]["total_gain"] for c in REGIME_COLS) / tot_gain if tot_gain else float("nan")
    print(f"  regime indicators' share of total split gain: {out['regime_gain_share']:.4f}", flush=True)
    top = (internal.group_by("split_feature").agg(pl.col("split_gain").sum().alias("gain"), pl.len().alias("n"))
           .sort("gain", descending=True).head(15))
    print("top features by total split gain:")
    print(top)
    return out


# ---------------------------------------------------------------- run the four variants
VARIANTS = [
    ("S-full", "split", COLS_B, cat_cols),
    ("S-restr", "split", COLS_B, CAT_RESTR),
    ("P-restr", "pooled", COLS_B, CAT_RESTR),
    ("P-full", "pooled", COLS_B, cat_cols),
]
preds, infos, splits = {}, {}, {}
for tag, rows_mode, fcols, ccols in VARIANTS:
    src = u_tr if rows_mode == "split" else (tr.sample(50_000, seed=0) if SMOKE else tr)
    log(f"===== fit {tag}: rows {rows_mode} ({src.height:,}), features {len(fcols)} numeric + {len(ccols)} categorical")
    t_start = time.time()
    p, info, model = fit_b(src, u_tg, CFG_B, fcols, ccols)
    p = np.clip(p, 0.0, None)
    info["seconds"] = round(time.time() - t_start, 1)
    info["rows_mode"] = rows_mode
    preds[tag], infos[tag] = p, info
    log(f"  {tag}: {({k: v for k, v in info.items() if k != 'cols'})} unmatched RMSE {rmse(y_tg, p):.4f}")
    if rows_mode == "pooled":
        splits[tag] = describe_splits(model, tag, info["cols"])
    del model
    pl.DataFrame({"MVT_ID_mvt": u_tg["MVT_ID_mvt"], **{k: v for k, v in preds.items()}}).write_parquet(PRED_OUT)
    log(f"  predictions written to {PRED_OUT}")

# ---------------------------------------------------------------- baseline OOF cross-check
oof = pl.read_parquet(BASE_OOF).drop_nulls("oof")
joined_u = u_tg.select(["MVT_ID_mvt", LABEL, "ADEP_mvt"]).join(oof, on="MVT_ID_mvt", how="left", maintain_order="left")
p_logged_u = joined_u["oof"].to_numpy().astype(float)
assert joined_u["MVT_ID_mvt"].equals(u_tg["MVT_ID_mvt"]), "OOF join reordered the unmatched rows"
d = preds["S-full"] - p_logged_u
log(f"S-full vs logged OOF on the {len(d):,} unmatched valid rows: max |diff| {np.abs(d).max():.4f}, "
    f"mean |diff| {np.abs(d).mean():.4f}, RMSE logged {rmse(y_tg, p_logged_u):.4f} vs repro {rmse(y_tg, preds['S-full']):.4f}")
joined_m = m_tg.select(["MVT_ID_mvt", LABEL, "ADEP_mvt"]).join(oof, on="MVT_ID_mvt", how="left", maintain_order="left")
assert joined_m["oof"].null_count() == 0 and joined_u["oof"].null_count() == 0
y_m = joined_m[LABEL].to_numpy().astype(float)
p_m = joined_m["oof"].to_numpy().astype(float)
log(f"matched valid rows {joined_m.height:,} (record n=325,004), matched RMSE {rmse(y_m, p_m):.4f} (record 201.6633)")

# whole-fold frame: matched predictions are the logged ones for every variant (model A untouched)
y_all = np.concatenate([y_m, y_tg])
ym_m = m_tg["ym"].to_numpy()
ym_u = u_tg["ym"].to_numpy()
month_all = np.concatenate([ym_m, ym_u])
full_preds = {tag: np.concatenate([p_m, preds[tag]]) for tag in preds}
full_preds["logged"] = np.concatenate([p_m, p_logged_u])
log(f"whole-fold rows {len(y_all):,} (record 328,009); logged total RMSE {rmse(y_all, full_preds['logged']):.4f} (record 284.7038); "
    f"S-full total {rmse(y_all, full_preds['S-full']):.4f}")

# ---------------------------------------------------------------- results
print("\n=== unmatched RMSE, forward fold (n = %d) ===" % len(y_tg), flush=True)
rows = []
for tag in ["logged", "S-full", "S-restr", "P-restr", "P-full"]:
    p = p_logged_u if tag == "logged" else preds[tag]
    lirf = adep_tg == "LIRF"
    rows.append({
        "variant": tag,
        "rounds": infos[tag]["rounds"] if tag in infos else 45,
        "unmatched_rmse": rmse(y_tg, p),
        "LIRF": rmse(y_tg[lirf], p[lirf]),
        "non_LIRF": rmse(y_tg[~lirf], p[~lirf]),
        "total_fold": rmse(y_all, full_preds[tag]),
        "pred_med": float(np.median(p)),
        "pred_max": float(p.max()),
        "n_pred_ge_7200": int((p >= 7200).sum()),
        "n_pred_ge_43200": int((p >= 43200).sum()),
    })
print(pl.DataFrame(rows))

print("\n=== unmatched RMSE per airport ===", flush=True)
per = {"ADEP": sorted(set(adep_tg.tolist()))}
counts = [int((adep_tg == a).sum()) for a in per["ADEP"]]
per["n"] = counts
for tag in ["logged", "S-full", "S-restr", "P-restr", "P-full"]:
    p = p_logged_u if tag == "logged" else preds[tag]
    pa = per_airport(y_tg, p, adep_tg)
    per[tag] = [pa[a] for a in per["ADEP"]]
print(pl.DataFrame(per))

print("\n=== paired comparison, prc.compare convention (A = S-full repro baseline, B = variant) ===", flush=True)
comparisons = []
for scope, (yv, base, get) in {
    "unmatched only": (y_tg, preds["S-full"], lambda t: preds[t]),
    "whole fold": (y_all, full_preds["S-full"], lambda t: full_preds[t]),
}.items():
    for tag in ["S-restr", "P-restr", "P-full"]:
        curve = robustness_curve(yv, base, get(tag), drop_levels=DROP_LEVELS, n_boot=500)
        ok, why = verdict(curve)
        print(f"\n-- {scope}: A = S-full, B = {tag}   (rows {len(yv):,})")
        print(f"{'dropped':>8} {'A':>10} {'B':>10} {'diff':>9} {'CI95':>22} {'P(B)':>6}")
        for r in curve:
            ci = f"[{r['ci95'][0]:+8.2f},{r['ci95'][1]:+8.2f}]"
            print(f"{r['dropped']:>8} {r['rmse_a']:10.2f} {r['rmse_b']:10.2f} {r['diff']:+9.2f} {ci:>22} {r['b_better_prob']:6.3f}")
        mnth = month_all if scope == "whole fold" else ym_u
        mb = month_bootstrap_rmse_diff(yv, base, get(tag), mnth, n_boot=500)
        print(f"month-level bootstrap ({mb['n_months']} months): {mb['diff_rmse']:+.2f} s "
              f"[{mb['ci95'][0]:+.2f}, {mb['ci95'][1]:+.2f}] P(B better) = {mb['b_better_prob']:.3f}")
        print(f"VERDICT: {'ADOPT' if ok else 'REJECT'} - {why}")
        comparisons.append({"scope": scope, "b": tag, "curve": curve, "month_boot": mb, "adopt": ok, "why": why})

# also: pooled against the logged baseline OOF itself (unmatched only), for completeness
for tag in ["P-restr", "P-full"]:
    curve = robustness_curve(y_tg, p_logged_u, preds[tag], drop_levels=DROP_LEVELS, n_boot=500)
    ok, why = verdict(curve)
    print(f"\n-- unmatched only: A = logged OOF, B = {tag}: "
          + ", ".join(f"K={r['dropped']} {r['diff']:+.2f} [{r['ci95'][0]:+.2f},{r['ci95'][1]:+.2f}]" for r in curve))
    print(f"VERDICT: {'ADOPT' if ok else 'REJECT'} - {why}")

print("\n=== LIRF unmatched rows with ms >= 12 h in the forward validation set ===", flush=True)
sel = (adep_tg == "LIRF") & (ms_tg >= 43200)
tbl = {"MVT_ID": u_tg["MVT_ID_mvt"].to_numpy()[sel], "y": y_tg[sel], "ms": ms_tg[sel],
       "y_minus_ms": (y_tg - ms_tg)[sel]}
for tag in ["logged", "S-full", "S-restr", "P-restr", "P-full"]:
    p = p_logged_u if tag == "logged" else preds[tag]
    tbl[tag] = p[sel]
print(pl.DataFrame(tbl))
for band, lo in (("ms >= 12h", 43200), ("ms >= 4h", 14400), ("ms >= 2h", 7200)):
    s = (adep_tg == "LIRF") & (ms_tg >= lo)
    if not s.any():
        print(f"LIRF {band}: no rows")
        continue
    line = {"band": band, "n": int(s.sum()), "y_med": float(np.median(y_tg[s]))}
    for tag in ["logged", "S-full", "S-restr", "P-restr", "P-full"]:
        p = p_logged_u if tag == "logged" else preds[tag]
        line[f"{tag}_med"] = float(np.median(p[s]))
        line[f"{tag}_rmse"] = rmse(y_tg[s], p[s])
    print(json.dumps(line))

print("\n=== worst rows under each variant (top 10 by squared error of P-restr) ===", flush=True)
order = np.argsort((y_tg - preds["P-restr"]) ** 2)[::-1][:10]
w = {"ADEP": adep_tg[order], "y": y_tg[order], "ms": ms_tg[order]}
for tag in ["logged", "S-full", "P-restr", "P-full"]:
    p = p_logged_u if tag == "logged" else preds[tag]
    w[tag] = p[order]
print(pl.DataFrame(w))

for _i in infos.values():
    _i.pop("cols", None)
summary = {"infos": infos, "splits": splits,
           "unmatched_rmse": {t: (rmse(y_tg, p_logged_u) if t == "logged" else rmse(y_tg, preds[t]))
                              for t in ["logged", "S-full", "S-restr", "P-restr", "P-full"]},
           "comparisons": [{k: v for k, v in c.items() if k != "curve"} | {"curve": c["curve"]} for c in comparisons]}
open(f"{SCRATCH}/shared-b-all-rows_summary.json", "w").write(json.dumps(summary, indent=1, default=str))
log("done; summary written to", f"{SCRATCH}/shared-b-all-rows_summary.json")
