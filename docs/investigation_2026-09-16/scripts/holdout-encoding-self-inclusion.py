"""Item 24 `holdout-encoding-self-inclusion`: the label encodings (`ref_*`, `arh_med`, `apt_med`, `ref_p10_fb`) are
fitted on the WHOLE training side, including the inner-holdout month that `_fit` then carves out for early
stopping, so the holdout rows' encodings contain their own labels. Effect on `best_iteration`, on the December
holdout RMSE, and on the tail of the resulting model?

Model B only. Three encoding variants, one early-stopping fit each (`_inner_holdout_mask(tr, 1)` → December 2025,
`lgb.early_stopping(100)`, `num_boost_round` 2,000, best_iteration scaled by n/(n−n_hold) exactly as train.py:295):

  A  production path      encodings fitted on all 12 labelled months, attached to every training row
  B  honest encodings     encodings fitted on months 1-11 only, attached to every training row
  C  holdout-only fix     months 1-11 rows keep A's encodings; the December holdout rows get B's
                          (the literal reading of the plan: "fit the encodings on months 1-11 and attach them to
                          December" — isolates the holdout leak from the change to the training side's own columns)

Then the consequence for the tail: refit on all 22,470 unmatched rows with the PRODUCTION (all-12) encodings at
each variant's round count and predict the 5,290 ranking unmatched rows, so that the only thing varying is the
round count the early stop chose.

Run:  POLARS_MAX_THREADS=1 .venv/bin/python docs/investigation_2026-09-16/scripts/holdout-encoding-self-inclusion.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "1")

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import yaml  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from prc.features import CATEGORICALS  # noqa: E402
from prc.train import (  # noqa: E402
    ENCODING_COLS,
    LABEL,
    PROXY,
    TrainConfig,
    _inner_holdout_mask,
    _matrix,
    attach_encodings,
    fit_encodings,
    visible_numeric_columns,
)

FEAT_TRAIN = REPO / "data/work/feat_train.parquet"
FEAT_RANK = REPO / "data/work/feat_rank.parquet"
OUT_DIR = Path(os.environ.get("SCRATCH", "/tmp")) / "item24"
NUM_THREADS = 2
ENC_COLS = ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "hour_local", "AIRCRAFT_OPERATOR_flt", LABEL, PROXY]
NM_PREFIXES = ("mvt_minus_aobt3", "aobt3", "lobt_minus", "mvt_minus_lobt", "mvt_minus_eobt1", "mvt_minus_iobt")
YM = (pl.col("MVT_TIME_UTC_mvt").dt.year() * 100 + pl.col("MVT_TIME_UTC_mvt").dt.month()).alias("ym")


def load_cfg() -> TrainConfig:
    """configs/main.yaml resolved the way prc.pipeline.main does (pipeline.py:128-137)."""
    spec = yaml.safe_load((REPO / "configs/main.yaml").read_text())
    fields = {k: v for k, v in spec.items() if k in TrainConfig.__dataclass_fields__}
    if "params" in fields:
        fields["params"] = {**TrainConfig().params, **fields["params"]}
    for k in ("folds", "exclude_features", "include_candidates"):
        if isinstance(fields.get(k), list):
            fields[k] = tuple(fields[k])
    return TrainConfig(**fields)


def cols_b_for(tr: pl.DataFrame, cfg: TrainConfig) -> tuple[list[str], list[str]]:
    num_cols = visible_numeric_columns(tr, cfg)
    cat_cols = [c for c in CATEGORICALS if c in tr.columns and c not in set(cfg.exclude_features)]
    return [c for c in num_cols if not c.startswith(NM_PREFIXES)], cat_cols


def early_stop(tr: pl.DataFrame, cfg: TrainConfig, cols_b: list[str], cat_cols: list[str], tag: str) -> dict:
    """train.py:_fit's early-stopping half, on this variant's frame."""
    hold = _inner_holdout_mask(tr, cfg.inner_holdout_months)
    X, cols = _matrix(tr.select(cols_b + cat_cols), cols_b, cat_cols)
    cat_idx = [cols.index(c) for c in cat_cols]
    params = cfg.lgb_params()
    y = tr[LABEL].to_numpy().astype(float)
    n_hold = int(hold.sum())
    t = time.time()
    dtr = lgb.Dataset(X[~hold], label=y[~hold], categorical_feature=cat_idx, params=params)
    dva = lgb.Dataset(X[hold], label=y[hold], categorical_feature=cat_idx, reference=dtr)
    evals: dict = {}
    booster = lgb.train(
        params, dtr, num_boost_round=cfg.num_rounds, valid_sets=[dva],
        callbacks=[lgb.log_evaluation(0), lgb.early_stopping(cfg.early_stopping, verbose=False), lgb.record_evaluation(evals)],
    )
    best = int(booster.best_iteration or cfg.num_rounds)
    rounds = int(min(cfg.num_rounds, max(10, round(best * tr.height / max(1, tr.height - n_hold)))))
    curve = np.sqrt(np.asarray(evals["valid_0"]["l2"], dtype=float))
    out = {
        "tag": tag, "n_train": tr.height, "n_hold": n_hold, "best_iteration": best, "rounds_refit": rounds,
        "holdout_rmse_best": float(curve[best - 1]), "holdout_rmse_min": float(curve.min()),
        "argmin": int(curve.argmin()) + 1, "n_iter_run": len(curve), "seconds": round(time.time() - t, 1),
        "curve_at": {str(i): float(curve[i - 1]) for i in (50, 100, 200, 300, 400, 500, 600, 700) if i <= len(curve)},
    }
    print(f"[{tag}] n={tr.height} hold={n_hold} best_iteration={best} rounds_refit={rounds} "
          f"holdout RMSE@best={out['holdout_rmse_best']:.2f} (min {out['holdout_rmse_min']:.2f} @{out['argmin']}) "
          f"iters run={len(curve)} {out['seconds']}s")
    return {"info": out, "booster": booster, "X": X, "hold": hold, "y": y, "cat_idx": cat_idx, "cols": cols}


def tail(v: np.ndarray) -> dict:
    return {
        "median": float(np.median(v)), "mean": float(np.mean(v)), "p99": float(np.quantile(v, 0.99)), "max": float(np.max(v)),
        "n_gt_7200": int((v > 7200).sum()), "n_gt_14400": int((v > 14400).sum()), "n_gt_43200": int((v > 43200).sum()),
    }


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = load_cfg()
    cfg = replace(base, n_seeds=1).for_unmatched()
    cfg = replace(cfg, params={**cfg.params, "num_threads": NUM_THREADS})
    print("resolved main.yaml:", json.dumps({k: v for k, v in base.__dict__.items() if k != "params"}, default=str))
    print("model-B params:", json.dumps(cfg.lgb_params(), sort_keys=True))

    lf = pl.scan_parquet(FEAT_TRAIN)
    labelled = lf.filter(pl.col(LABEL).is_not_null() & pl.col(LABEL).is_finite())
    enc_src = labelled.select([*ENC_COLS, YM]).collect()
    months = sorted(enc_src["ym"].unique().to_list())
    hold_ym = months[-1]
    print("labelled training rows:", enc_src.height, "| months:", months, "| holdout month:", hold_ym)
    enc_all = fit_encodings(enc_src.drop("ym"), cfg)
    enc_111 = fit_encodings(enc_src.filter(pl.col("ym") != hold_ym).drop("ym"), cfg)
    print("encoding table sizes  all-12:", {k: v.height for k, v in enc_all.items()},
          "| 1-11:", {k: v.height for k, v in enc_111.items()})
    print("rows used for the encodings: all-12", enc_src.height, "| 1-11", enc_src.filter(pl.col("ym") != hold_ym).height)
    del enc_src

    u_tr = labelled.filter(pl.col(PROXY).is_null()).collect()
    u_tg = pl.scan_parquet(FEAT_RANK).filter(pl.col(PROXY).is_null()).collect()
    tr_A, tr_B = attach_encodings(u_tr, enc_all), attach_encodings(u_tr, enc_111)
    tg_A = attach_encodings(u_tg, enc_all)
    assert tr_A.height == u_tr.height == tr_B.height and tg_A.height == u_tg.height
    hold0 = _inner_holdout_mask(tr_A, cfg.inner_holdout_months)
    # C: months 1-11 rows as in production, December holdout rows with the 1-11 encodings. Concatenating the two
    # blocks reorders rows, so the mask is recomputed from MVT_TIME_UTC_mvt on the concatenated frame.
    tr_C = pl.concat([tr_A.filter(~hold0), tr_B.filter(hold0)], how="vertical")
    assert tr_C.height == tr_A.height
    print(f"unmatched training rows {u_tr.height}, ranking {u_tg.height}; December holdout {int(hold0.sum())}")

    # ---- how much of the holdout's own labels sit in its encodings ----
    ym_tr = u_tr.select(YM)["ym"]
    dec = u_tr.filter(ym_tr == hold_ym)
    lab_all = labelled.select([*ENC_COLS, YM]).collect()
    self_stats: dict = {}
    for name, keys in (("ref", ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]), ("arh", ["ADEP_mvt", "RUNWAY_mvt", "hour_local"]), ("apt", ["ADEP_mvt"])):
        src = lab_all.filter(pl.col(LABEL).is_not_null())
        if name == "ref":
            src = src.filter((pl.col(LABEL) > 0) & (pl.col(LABEL) < 7200))
        grp = src.group_by(keys).agg([pl.len().alias("n_all"), (pl.col("ym") == hold_ym).sum().alias("n_dec")])
        j = dec.select(keys).join(grp, on=keys, how="left")
        share = (j["n_dec"] / j["n_all"]).drop_nulls()
        self_stats[name] = {
            "holdout_rows_with_a_group": int(j["n_all"].drop_nulls().len()),
            "median_group_n": float(j["n_all"].median() or float("nan")),
            "median_dec_share": float(share.median()) if len(share) else float("nan"),
            "mean_dec_share": float(share.mean()) if len(share) else float("nan"),
            "max_dec_share": float(share.max()) if len(share) else float("nan"),
            "share_gt_0.2": float((share > 0.2).mean()) if len(share) else float("nan"),
        }
        print(f"  self-inclusion {name}: {self_stats[name]}")
    del lab_all
    # how far the encoding columns actually move on the holdout rows
    encdiff = {}
    a_h, b_h = tr_A.filter(hold0), tr_B.filter(hold0)
    for c in ENCODING_COLS:
        d = (a_h[c].cast(pl.Float64) - b_h[c].cast(pl.Float64))
        both = d.drop_nulls()
        encdiff[c] = {
            "n_null_all12": int(a_h[c].null_count()), "n_null_1_11": int(b_h[c].null_count()),
            "n_changed": int((both != 0).sum()), "median_abs_change": float(both.abs().median()) if len(both) else 0.0,
            "p90_abs_change": float(both.abs().quantile(0.9)) if len(both) else 0.0,
        }
    print("  encoding-column change on the 1,649 holdout rows:", json.dumps(encdiff, indent=1, default=str))

    cols_b, cat_cols = cols_b_for(tr_A, cfg)
    print(f"cols_b {len(cols_b)} numeric + {len(cat_cols)} categorical")

    # ---- the three early-stopping fits ----
    runs = {}
    for tag, frame in (("A production (enc 1-12)", tr_A), ("B honest (enc 1-11 everywhere)", tr_B), ("C holdout-only (enc 1-11 on Dec)", tr_C)):
        runs[tag] = early_stop(frame, cfg, cols_b, cat_cols, tag)

    # ---- pure effect of the leak on the early-stopping signal: one model, two versions of the holdout ----
    rA = runs["A production (enc 1-12)"]
    XB, _ = _matrix(tr_B.select(cols_b + cat_cols), cols_b, cat_cols)
    yh = rA["y"][rA["hold"]]
    pred_leak = rA["booster"].predict(rA["X"][rA["hold"]], num_iteration=rA["info"]["best_iteration"])
    pred_clean = rA["booster"].predict(XB[hold0], num_iteration=rA["info"]["best_iteration"])
    same_model = {
        "rmse_holdout_enc_1_12": float(np.sqrt(np.mean((yh - pred_leak) ** 2))),
        "rmse_holdout_enc_1_11": float(np.sqrt(np.mean((yh - pred_clean) ** 2))),
        "rms_pred_diff": float(np.sqrt(np.mean((pred_leak - pred_clean) ** 2))),
        "n_pred_identical": int((pred_leak == pred_clean).sum()), "n": int(len(yh)),
    }
    print("same model A @best_iteration, holdout features swapped:", json.dumps(same_model))
    del XB

    # ---- consequence for the tail: production encodings, each variant's round count ----
    stacked = pl.concat([f.select(cols_b + cat_cols) for f in (tr_A, tg_A)], how="vertical_relaxed")
    Xall, cols = _matrix(stacked, cols_b, cat_cols)
    del stacked
    n_tr = tr_A.height
    cat_idx = [cols.index(c) for c in cat_cols]
    y_all = tr_A[LABEL].to_numpy().astype(float)
    dfull = lgb.Dataset(Xall[:n_tr], label=y_all, categorical_feature=cat_idx, free_raw_data=False, params=cfg.lgb_params())
    preds: dict[int, np.ndarray] = {}
    for tag, r in runs.items():
        rounds = r["info"]["rounds_refit"]
        if rounds in preds:
            continue
        p = dict(cfg.lgb_params(), seed=cfg.seed, bagging_seed=cfg.seed, feature_fraction_seed=cfg.seed)
        t = time.time()
        m = lgb.train(p, dfull, num_boost_round=rounds, callbacks=[lgb.log_evaluation(0)])
        preds[rounds] = m.predict(Xall[n_tr:])
        print(f"  refit at {rounds} rounds ({tag}) {time.time() - t:.1f}s")

    res = pl.DataFrame({"MVT_ID_mvt": tg_A["MVT_ID_mvt"], "ADEP_mvt": tg_A["ADEP_mvt"], "ms": tg_A["mvt_minus_sched"]})
    for rounds, p in preds.items():
        res = res.with_columns(pl.Series(f"r{rounds}", p))
    for k, path in (("v8_det_a", REPO / "outputs/det_a.parquet"), ("v4_probe", REPO / "outputs/probe_unmatched_from_v4.parquet")):
        ref = pl.scan_parquet(path).select(["MVT_ID_mvt", pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias(k)]).collect()
        res = res.join(ref, on="MVT_ID_mvt", how="left", maintain_order="left")
    res.write_parquet(OUT_DIR / "preds.parquet")
    print("\nranking tails by round count:")
    tails = {}
    for c in res.columns:
        if c.startswith("r") or c.startswith("v"):
            tails[c] = tail(res[c].to_numpy())
            print(f"  {c:>12}", {kk: (round(vv, 1) if isinstance(vv, float) else vv) for kk, vv in tails[c].items()})
    rk = [c for c in res.columns if c.startswith("r")]
    rms_pairs = {}
    ms = res["ms"].to_numpy()
    for i in range(len(rk)):
        for j in range(i + 1, len(rk)):
            a, b = res[rk[i]].to_numpy(), res[rk[j]].to_numpy()
            rms_pairs[f"{rk[i]}~{rk[j]}"] = {
                "overall": float(np.sqrt(np.mean((a - b) ** 2))),
                "ms_ge_2h": float(np.sqrt(np.mean((a[ms >= 7200] - b[ms >= 7200]) ** 2))),
                "ms_ge_12h": float(np.sqrt(np.mean((a[ms >= 43200] - b[ms >= 43200]) ** 2))),
                "lirf": float(np.sqrt(np.mean((a[(res["ADEP_mvt"] == "LIRF").to_numpy()] - b[(res["ADEP_mvt"] == "LIRF").to_numpy()]) ** 2))),
            }
    print("RMS between round counts:", json.dumps(rms_pairs, indent=1))

    out = {
        "holdout_month": hold_ym, "months": months, "n_unmatched_train": u_tr.height, "n_unmatched_rank": u_tg.height,
        "n_holdout": int(hold0.sum()), "enc_sizes": {"all12": {k: v.height for k, v in enc_all.items()}, "m1_11": {k: v.height for k, v in enc_111.items()}},
        "self_inclusion": self_stats, "encoding_change_on_holdout": encdiff, "cols_b": cols_b, "cat_cols": cat_cols,
        "params": cfg.lgb_params(), "runs": {k: v["info"] for k, v in runs.items()}, "same_model_swap": same_model,
        "tails": tails, "rms_between_round_counts": rms_pairs, "seconds": round(time.time() - t0, 1),
    }
    (OUT_DIR / "results.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"\ndone in {out['seconds']}s -> {OUT_DIR}")


if __name__ == "__main__":
    main()
