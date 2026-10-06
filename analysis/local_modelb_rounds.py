"""Model B (unmatched lane) round-count / early-stopping variants, through the real predict_fold path.

Trigger: the full-2025 refit of main_flseq logged rounds_unmatched = 39. Model B early-stops on the latest training
month (inner_holdout_months = 1), which is December for every fold except forward (Oct) and jul_fwd (Jun).

What it does, per fold (jan, jul, jul_fwd, forward, aso, fmj) and for the full 2025 -> ranking fit:
  * calls prc.train.predict_fold with the target frame restricted to unmatched rows (mvt_minus_aobt3 null), so
    model A is never fitted but encodings / columns / reparam / clip / LOBT bound are the production code;
  * prc.train._fit is replaced by a fitter that builds the SAME stacked matrix and runs the SAME early-stopping
    stage, but trains it to 2000 rounds while recording several holdout metrics, then emulates
    lgb.early_stopping (patience 100, or 300) on each metric; the 3-seed refit is trained once to the largest
    round count any variant needs and predicted at every variant's count (boosting is sequential, so the
    prefix of a longer deterministic fit IS the shorter fit);
  * prc.train._assemble is wrapped so each variant's prediction goes through the production post-processing.
Variants: prod (ES on l2), es_p300, es_huber600, es_l1, es_clip7200, es_clip3600, es_hold2 (two holdout months,
l2), fixed rounds 100/200/400/800/1600 (refit rounds, unscaled).

Outputs ($SCRATCH/agents/modelb/): fold_<name>.parquet (per-row y + variant preds), fold_<name>.json (rounds,
holdout stats, ES curves), rank_delta.parquet (written by --report when a variant is chosen).

    POLARS_MAX_THREADS=3 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage \
        .venv/bin/python analysis/local_modelb_rounds.py            # runs folds (skips finished ones)
    ... analysis/local_modelb_rounds.py --report [--choose VARIANT]  # bootstrap tables (+ rank delta)
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from dataclasses import replace

import lightgbm as lgb
import numpy as np
import polars as pl
import yaml

sys.path.insert(0, "src")
import prc.train as T  # noqa: E402
from prc.cv import month_block_folds, year_month_key  # noqa: E402
from prc.pipeline import drop_hidden_candidates, train_config_from_spec  # noqa: E402

OUT = os.path.join(os.environ["SCRATCH"], "agents", "modelb")
os.makedirs(OUT, exist_ok=True)
LABEL, PROXY = T.LABEL, T.PROXY
FIXED = (100, 200, 400, 800, 1600)
ES_VARIANTS = ("prod", "es_p300", "es_huber600", "es_l1", "es_clip7200", "es_clip3600", "es_hold2")
VARIANTS = ES_VARIANTS + tuple(f"fix{r}" for r in FIXED)
DROP = ["FLIGHT_ID_mvt", "PHASE_mvt", "BLOCK_TIME_UTC_mvt", "FLIGHT_mvt", "FLIGHT_RULE_mvt", "SCHED_TIME_UTC_mvt",
        "LOBT_flt", "CALLSIGN_flt", "ADEP_flt", "ADES_flt", "ADES_FILED_flt", "IOBT_flt", "FLIGHT_RULE_flt",
        "AIRCRAFT_TYPE_flt", "EOBT_1_flt", "ARVT_1_flt", "AOBT_3_flt", "ARVT_3_flt"]


def load_cfg():
    spec = yaml.safe_load(open("configs/main_flseq.yaml"))
    cfg = train_config_from_spec(spec)
    return replace(cfg, params={**cfg.params, "num_threads": 3})


def es_best(curve: np.ndarray, patience: int) -> int:
    """lgb.early_stopping emulation: 1-based best iteration, strict improvement, stop after `patience` misses."""
    best, best_i = np.inf, 0
    for i, s in enumerate(curve):
        if s < best:
            best, best_i = s, i
        elif i - best_i >= patience:
            break
    return best_i + 1


def scale(best: int, n: int, n_hold: int, cap: int) -> int:
    return int(min(cap, max(10, round(best * n / max(1, n - n_hold)))))


STATE: dict = {}


def variant_fit(train, target_expr, cfg, feature_cols, cat_cols, predict_frames, es_rows=None):
    """Production `_fit` (LightGBM branch, es_rows None) with every variant computed alongside."""
    assert es_rows is None and cfg.model == "lightgbm"
    hold = T._inner_holdout_mask(train, cfg.inner_holdout_months)
    hold2 = T._inner_holdout_mask(train, 2)
    n, n_hold, n_hold2 = train.height, int(hold.sum()), int(hold2.sum())
    frames = [train, *predict_frames]
    stacked = pl.concat([f.select(feature_cols + cat_cols) for f in frames], how="vertical_relaxed")
    Xall, cols = T._matrix(stacked, feature_cols, cat_cols)
    del stacked
    offs, i = [], 0
    for f in frames:
        offs.append((i, i + f.height))
        i += f.height
    cat_idx = [cols.index(c) for c in cat_cols]
    params = cfg.lgb_params()
    a, b = offs[0]
    y = target_expr(train)
    lab = train[LABEL].to_numpy().astype(float)
    shift = lab - y
    Xtrain = Xall[a:b]
    meta = {"n_train": n, "n_hold": n_hold, "n_hold2": n_hold2,
            "hold_months": sorted(set(year_month_key(train.filter(pl.Series(hold))).to_list())),
            "hold_y_gt3600": int((lab[hold] > 3600).sum()), "hold_y_gt7200": int((lab[hold] > 7200).sum()),
            "hold_y_sd": float(lab[hold].std()), "train_y_sd": float(lab[~hold].std())}

    _held = {}

    def _grab(preds, data):
        _held["p"] = np.asarray(preds, dtype=float)
        return ("grab", 0.0, False)

    def es_stage(mask):
        yh, sh = y[mask], shift[mask]
        lab_h = yh + sh
        curves = {k: [] for k in ("l2", "huber600", "l1", "clip7200", "clip3600")}
        dtr = lgb.Dataset(Xtrain[~mask], label=y[~mask], categorical_feature=cat_idx, params=params)
        dva = lgb.Dataset(Xtrain[mask], label=yh, categorical_feature=cat_idx, reference=dtr)
        bst = lgb.Booster(params, dtr)
        bst.add_valid(dva, "h")
        for it in range(cfg.num_rounds):
            bst.update()
            bst.eval_valid(feval=_grab)  # public route to the booster's cached raw scores on the valid set
            p = _held["p"]
            e = p - yh
            ae = np.abs(e)
            curves["l2"].append(float(np.mean(e * e)))
            curves["huber600"].append(float(np.mean(np.where(ae <= 600, 0.5 * e * e, 600 * (ae - 300)))))
            curves["l1"].append(float(np.mean(ae)))
            pl_ = p + sh
            curves["clip7200"].append(float(np.mean((pl_ - np.minimum(lab_h, 7200)) ** 2)))
            curves["clip3600"].append(float(np.mean((pl_ - np.minimum(lab_h, 3600)) ** 2)))
            # stop once every curve is past its patience-300 window (prod path never needs more)
            if it >= 400 and it % 50 == 0:
                if all(it + 1 - es_best(np.array(c), 300) >= 300 for c in curves.values()):
                    break
        del dtr, dva, bst
        return {k: np.array(v) for k, v in curves.items()}

    c1 = es_stage(hold)
    c2 = es_stage(hold2)
    best = {
        "prod": es_best(c1["l2"], cfg.early_stopping),
        "es_p300": es_best(c1["l2"], 300),
        "es_huber600": es_best(c1["huber600"], cfg.early_stopping),
        "es_l1": es_best(c1["l1"], cfg.early_stopping),
        "es_clip7200": es_best(c1["clip7200"], cfg.early_stopping),
        "es_clip3600": es_best(c1["clip3600"], cfg.early_stopping),
        "es_hold2": es_best(c2["l2"], cfg.early_stopping),
    }
    rounds = {k: scale(v, n, n_hold2 if k == "es_hold2" else n_hold, cfg.num_rounds) for k, v in best.items()}
    rounds.update({f"fix{r}": r for r in FIXED})
    meta["es_best_iter"] = best
    meta["rounds"] = rounds
    meta["curve_l2_hold1"] = c1["l2"][:600].round(1).tolist()
    meta["curve_clip7200_hold1"] = c1["clip7200"][:600].round(1).tolist()
    meta["curve_l2_hold2"] = c2["l2"][:600].round(1).tolist()
    del Xtrain
    gc.collect()

    need = sorted(set(rounds.values()))
    dfull = lgb.Dataset(Xall[a:b], label=y, categorical_feature=cat_idx, free_raw_data=False, params=params)
    s_, e_ = offs[1]
    acc = {r: np.zeros(e_ - s_) for r in need}
    for k in range(max(1, cfg.n_seeds)):
        p_k = dict(params, seed=cfg.seed + k, bagging_seed=cfg.seed + k, feature_fraction_seed=cfg.seed + k)
        model = lgb.train(p_k, dfull, num_boost_round=max(need), callbacks=[lgb.log_evaluation(0)])
        for r in need:
            acc[r] += model.predict(Xall[s_:e_], num_iteration=r)
    preds = {v: acc[rounds[v]] / max(1, cfg.n_seeds) for v in VARIANTS}
    STATE["g"] = preds
    STATE["meta"] = meta
    return [preds["prod"]], rounds["prod"]


_orig_assemble = T._assemble


def variant_assemble(parts, tr, tg, target_f, info, cfg=None):
    assert len(parts) == 1  # unmatched lane only
    part = parts[0]
    base = part["pred"].to_numpy() - STATE["g"]["prod"]  # the reparam shift predict_fold added
    outs = {}
    for v in VARIANTS:
        pv = part.with_columns(pl.Series("pred", base + STATE["g"][v]))
        o, _ = _orig_assemble([pv], tr, tg, target_f, dict(info), cfg)
        outs[v] = o["pred"].to_numpy()
    out, info = _orig_assemble(parts, tr, tg, target_f, info, cfg)
    assert np.array_equal(out["pred"].to_numpy(), outs["prod"])
    out = out.with_columns([pl.Series(f"p_{v}", outs[v]) for v in VARIANTS])
    return out, info


T._fit = variant_fit
T._assemble = variant_assemble


def load_frames(cfg, rank=False):
    p = "data/work_flseq/feat_rank.parquet" if rank else "data/work_flseq/feat_train.parquet"
    f = pl.read_parquet(p)
    f = drop_hidden_candidates(f, cfg).drop([c for c in DROP if c in f.columns])
    return f


def run():
    cfg = load_cfg()
    feats = load_frames(cfg)
    full_cols = T.visible_numeric_columns(drop_hidden_candidates(pl.read_parquet(
        "data/work_flseq/feat_train.parquet", n_rows=100), cfg), cfg)
    assert T.visible_numeric_columns(feats, cfg) == full_cols
    months = year_month_key(feats)
    folds = month_block_folds(months.unique().to_list())
    jobs = [(f.name, f.train_months, f.valid_months) for f in folds] + [("full", None, None)]
    for name, trm, vam in jobs:
        dst = os.path.join(OUT, f"fold_{name}.parquet")
        if os.path.exists(dst):
            print("skip", name)
            continue
        t0 = time.time()
        if name == "full":
            tr_f = feats
            va_f = load_frames(cfg, rank=True).filter(pl.col(PROXY).is_null())
        else:
            tr_f = feats.filter(months.is_in(list(trm)))
            va_f = feats.filter(months.is_in(list(vam)) & pl.col(PROXY).is_null())
        pred, info = T.predict_fold(tr_f, va_f, cfg)
        res = va_f.select(["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", LABEL]).join(
            pred, on="MVT_ID_mvt", how="left", maintain_order="left")
        res = res.with_columns(pl.lit(name).alias("fold"))
        meta = {**STATE["meta"], "rounds_unmatched_logged_style": info.get("rounds_unmatched"),
                "seconds": round(time.time() - t0, 1)}
        json.dump(meta, open(os.path.join(OUT, f"fold_{name}.json"), "w"))
        res.write_parquet(dst)
        print(name, "rounds", meta["rounds"], "hold", meta["hold_months"], f"{meta['seconds']}s", flush=True)
        if name != "full":
            y = res[LABEL].to_numpy().astype(float)
            for v in VARIANTS:
                print(f"   {v:12s} rmse {np.sqrt(np.mean((y - res['p_' + v].to_numpy()) ** 2)):8.1f}")
        del tr_f, va_f, pred, res
        gc.collect()


def day_boot(dd_by_day: np.ndarray, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    k = len(dd_by_day)
    idx = rng.integers(0, k, (B, k))
    return dd_by_day[idx].sum(1)


def report(choose: str | None):
    POOL = ["jan", "fmj", "jul", "aso", "forward"]  # the twelve 2025 months, each once
    folds = {}
    for name in POOL + ["jul_fwd"]:
        p = os.path.join(OUT, f"fold_{name}.parquet")
        if os.path.exists(p):
            folds[name] = pl.read_parquet(p)
    metas = {n: json.load(open(os.path.join(OUT, f"fold_{n}.json"))) for n in list(folds) + ["full"]
             if os.path.exists(os.path.join(OUT, f"fold_{n}.json"))}
    print("rounds per fold/variant:")
    for n, m in metas.items():
        print(f"  {n:8s} hold {m['hold_months']} n_hold {m['n_hold']} >3600:{m['hold_y_gt3600']} "
              f">7200:{m['hold_y_gt7200']} sd {m['hold_y_sd']:.0f}/{m['train_y_sd']:.0f}", m["rounds"])
    pooled = pl.concat([folds[n] for n in POOL if n in folds])
    sets = {**{n: folds[n] for n in folds}, "POOLED12": pooled}
    rows = []
    for sname, d in sets.items():
        d = d.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.date().alias("day"), pl.col(LABEL).cast(pl.Float64).alias("y"))
        for subset in ("all", "y<=3600"):
            s = d if subset == "all" else d.filter(pl.col("y") <= 3600)
            y = s["y"].to_numpy()
            se0 = (y - s["p_prod"].to_numpy()) ** 2
            for v in VARIANTS:
                se = (y - s["p_" + v].to_numpy()) ** 2
                dd = s.select(pl.col("day"), pl.Series("dd", se - se0)).group_by("day").agg(pl.col("dd").sum())["dd"].to_numpy()
                bs = day_boot(dd) / len(y)
                rows.append({"set": sname, "subset": subset, "variant": v, "n": len(y),
                             "rmse": float(np.sqrt(se.mean())), "rmse_prod": float(np.sqrt(se0.mean())),
                             "dmse": float((se - se0).mean()), "dmse_lo": float(np.percentile(bs, 2.5)),
                             "dmse_hi": float(np.percentile(bs, 97.5)), "p_better": float((bs < 0).mean())})
    R = pl.DataFrame(rows)
    R.write_parquet(os.path.join(OUT, "summary.parquet"))
    with pl.Config(tbl_rows=400, tbl_width_chars=200, float_precision=1):
        print(R.filter(pl.col("set") == "POOLED12").sort("subset", "dmse"))
        print(R.filter(pl.col("set") != "POOLED12", pl.col("subset") == "all").select(
            "set", "variant", "rmse", "dmse", "dmse_lo", "dmse_hi").pivot(on="set", index="variant", values="dmse"))
    # per airport, pooled 12 months, all rows
    d = pooled.with_columns(pl.col(LABEL).cast(pl.Float64).alias("y"))
    ap = d.group_by("ADEP_mvt").agg(
        pl.len().alias("n"), *[((pl.col("y") - pl.col("p_" + v)) ** 2).mean().sqrt().alias(v) for v in VARIANTS]
    ).sort("ADEP_mvt")
    with pl.Config(tbl_rows=40, tbl_width_chars=250, float_precision=0):
        print(ap)
        d2 = d.filter(pl.col("y") <= 3600)
        print(d2.group_by("ADEP_mvt").agg(
            pl.len().alias("n"), *[((pl.col("y") - pl.col("p_" + v)) ** 2).mean().sqrt().alias(v) for v in VARIANTS]
        ).sort("ADEP_mvt"))
    ap.write_parquet(os.path.join(OUT, "by_airport.parquet"))
    if choose:
        full = pl.read_parquet(os.path.join(OUT, "fold_full.parquet"))
        out = full.select("MVT_ID_mvt", "ADEP_mvt", pl.col("p_prod").alias("pred_prod"),
                          pl.col("p_" + choose).alias("pred_variant")).with_columns(
            (pl.col("pred_variant") - pl.col("pred_prod")).alias("delta"), pl.lit(choose).alias("variant"))
        out.write_parquet(os.path.join(OUT, "rank_delta.parquet"))
        print(out.describe())
        print(out.group_by("ADEP_mvt").agg(pl.len(), pl.col("delta").mean(), pl.col("delta").abs().max()).sort("ADEP_mvt"))


if __name__ == "__main__":
    if "--report" in sys.argv:
        ch = sys.argv[sys.argv.index("--choose") + 1] if "--choose" in sys.argv else None
        report(ch)
    else:
        run()
