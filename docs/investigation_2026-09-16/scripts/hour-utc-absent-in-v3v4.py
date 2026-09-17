"""Item 22 `hour-utc-absent-in-v3v4`: did training model B without `hour_utc` (the v3/v4 weather-join bug, fixed in
34808da) produce the > 12 h LIRF predictions that v3/v4 emit (16 / 18 unmatched predictions > 43,200 s)?

Model B only, weather columns PRESENT (as in v3/v4), 127 leaves / min_data 50, one seed, early stopping exactly as
train.py:_fit does it (inner holdout = latest training month via _inner_holdout_mask, best_iteration scaled to the
full training size, capped at unmatched_num_rounds), then a refit on all 22,470 unmatched rows. Two matrices:
  WITH    hour_utc  (today's code path)
  WITHOUT hour_utc  (exclude_features: [hour_utc]; what v3/v4's matrix lacked)
Each refit is done under both seed semantics (old: `seed` only, as v3/v4's code did; HEAD: + bagging_seed /
feature_fraction_seed), so the hour_utc effect can be read within one semantics and the two are not confounded.
Not reproduced from v3/v4: linear_tree (v4 had it), the 1500-round cap, the absence of force_row_wise, and the
v3/v4-era METAR cache; see results/v4-config-not-reproducible.md.

Run:  POLARS_MAX_THREADS=1 .venv/bin/python docs/investigation_2026-09-16/scripts/hour-utc-absent-in-v3v4.py
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
OUT_DIR = Path(os.environ.get("SCRATCH", "/tmp")) / "item22"
NUM_THREADS = 2
ENC_COLS = ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "hour_local", "AIRCRAFT_OPERATOR_flt", LABEL, PROXY]
NM_PREFIXES = ("mvt_minus_aobt3", "aobt3", "lobt_minus", "mvt_minus_lobt", "mvt_minus_eobt1", "mvt_minus_iobt")


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


def load_model_b_frames(cfg: TrainConfig) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    lf = pl.scan_parquet(FEAT_TRAIN)
    labelled = lf.filter(pl.col(LABEL).is_not_null() & pl.col(LABEL).is_finite())
    enc_frame = labelled.select(ENC_COLS).collect()
    enc = fit_encodings(enc_frame, cfg)
    info = {"labelled_train_rows": enc_frame.height, **{k: v.height for k, v in enc.items()}}
    del enc_frame
    u_tr = labelled.filter(pl.col(PROXY).is_null()).collect()
    u_tg = pl.scan_parquet(FEAT_RANK).filter(pl.col(PROXY).is_null()).collect()
    tr, tg = attach_encodings(u_tr, enc), attach_encodings(u_tg, enc)
    assert tr.height == u_tr.height and tg.height == u_tg.height
    assert tg["MVT_ID_mvt"].equals(u_tg["MVT_ID_mvt"])
    return tr, tg, info


def cols_b_for(tr: pl.DataFrame, cfg: TrainConfig) -> tuple[list[str], list[str]]:
    num_cols = visible_numeric_columns(tr, cfg)
    cat_cols = [c for c in CATEGORICALS if c in tr.columns and c not in set(cfg.exclude_features)]
    cols_b = [c for c in num_cols if not c.startswith(NM_PREFIXES)]
    return cols_b, cat_cols


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def fit_variant(tr: pl.DataFrame, tg: pl.DataFrame, cfg: TrainConfig, tag: str) -> dict:
    """train.py:_fit re-expressed: same matrix, same holdout, same early stop and round scaling; the refit is done
    once per seed semantics on the same lgb.Dataset."""
    cols_b, cat_cols = cols_b_for(tr, cfg)
    hold = _inner_holdout_mask(tr, cfg.inner_holdout_months)
    stacked = pl.concat([f.select(cols_b + cat_cols) for f in (tr, tg)], how="vertical_relaxed")
    Xall, cols = _matrix(stacked, cols_b, cat_cols)
    del stacked
    n_tr = tr.height
    Xtr, Xtg = Xall[:n_tr], Xall[n_tr:]
    cat_idx = [cols.index(c) for c in cat_cols]
    params = cfg.lgb_params()
    y = tr[LABEL].to_numpy().astype(float)
    n_hold = int(hold.sum())
    out = {
        "tag": tag, "n_numeric": len(cols_b), "n_cat": len(cat_cols), "hour_utc_in_matrix": "hour_utc" in cols,
        "n_train": n_tr, "n_hold": n_hold, "params_early_stop": params,
    }
    t = time.time()
    dtr = lgb.Dataset(Xtr[~hold], label=y[~hold], categorical_feature=cat_idx, params=params)
    dva = lgb.Dataset(Xtr[hold], label=y[hold], categorical_feature=cat_idx, reference=dtr)
    evals: dict = {}
    booster = lgb.train(
        params, dtr, num_boost_round=cfg.num_rounds, valid_sets=[dva],
        callbacks=[lgb.log_evaluation(0), lgb.early_stopping(cfg.early_stopping, verbose=False), lgb.record_evaluation(evals)],
    )
    best = int(booster.best_iteration or cfg.num_rounds)
    rounds = int(min(cfg.num_rounds, max(10, round(best * n_tr / max(1, n_tr - n_hold)))))
    curve = evals["valid_0"]["l2"]
    out.update(
        {
            "best_iteration": best, "rounds_refit": rounds, "holdout_rmse_best": float(np.sqrt(curve[best - 1])),
            "holdout_rmse_iter1": float(np.sqrt(curve[0])), "n_iter_run": len(curve), "early_stop_seconds": round(time.time() - t, 1),
        }
    )
    # early-stop-fit importance of the two hour features (gain share)
    gain = booster.feature_importance("gain")
    tot = gain.sum()
    out["gain_share_es_fit"] = {c: float(gain[cols.index(c)] / tot) for c in ("hour_utc", "hour_local", "minute_of_day_utc", "mvt_minus_sched", "mvt_minus_sched_slog") if c in cols}
    del dtr, dva, booster

    dfull = lgb.Dataset(Xtr, label=y, categorical_feature=cat_idx, free_raw_data=False, params=params)
    preds = {}
    for sem, p in (("old", dict(params)), ("head", dict(params, seed=cfg.seed, bagging_seed=cfg.seed, feature_fraction_seed=cfg.seed))):
        t = time.time()
        m = lgb.train(p, dfull, num_boost_round=rounds, callbacks=[lgb.log_evaluation(0)])
        preds[sem] = m.predict(Xtg)
        gain = m.feature_importance("gain"); tot = gain.sum()
        out[f"gain_share_refit_{sem}"] = {c: float(gain[cols.index(c)] / tot) for c in ("hour_utc", "hour_local", "minute_of_day_utc", "mvt_minus_sched") if c in cols}
        out[f"refit_params_{sem}"] = p
        out[f"refit_seconds_{sem}"] = round(time.time() - t, 1)
    print(f"[{tag}] numeric {len(cols_b)} cat {len(cat_cols)} hour_utc_in_matrix={out['hour_utc_in_matrix']} | "
          f"holdout n={n_hold} best_iteration={best} rounds_refit={rounds} holdout RMSE@best={out['holdout_rmse_best']:.1f} "
          f"| gain share ES fit {json.dumps({k: round(v, 4) for k, v in out['gain_share_es_fit'].items()})}")
    out["preds"] = preds
    return out


def tail(v: np.ndarray) -> dict:
    return {
        "median": float(np.median(v)), "mean": float(np.mean(v)), "p99": float(np.quantile(v, 0.99)), "max": float(np.max(v)),
        "n_gt_7200": int((v > 7200).sum()), "n_gt_14400": int((v > 14400).sum()), "n_gt_43200": int((v > 43200).sum()),
    }


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = load_cfg()
    cfg_with = replace(base, n_seeds=1).for_unmatched()
    cfg_with = replace(cfg_with, params={**cfg_with.params, "num_threads": NUM_THREADS})
    cfg_without = replace(cfg_with, exclude_features=tuple(cfg_with.exclude_features) + ("hour_utc",))
    print("resolved main.yaml:", json.dumps({k: v for k, v in base.__dict__.items() if k != "params"}, default=str))
    print("model-B cfg (with):", json.dumps({k: v for k, v in cfg_with.__dict__.items() if k != "params"}, default=str))
    print("model-B params:", json.dumps(cfg_with.lgb_params(), sort_keys=True))

    tr, tg, info = load_model_b_frames(cfg_with)
    print("frames:", info, "u_tr", tr.height, "u_tg", tg.height, "| weather cols present:", "temp_c" in tr.columns)
    print("hour_utc null rate train/rank:", tr["hour_utc"].null_count(), tg["hour_utc"].null_count(),
          "| corr(hour_utc, hour_local) on u_tr:", round(float(np.corrcoef(tr["hour_utc"].to_numpy().astype(float), tr["hour_local"].to_numpy().astype(float))[0, 1]), 4))
    r_with = fit_variant(tr, tg, cfg_with, "WITH hour_utc")
    r_without = fit_variant(tr, tg, cfg_without, "WITHOUT hour_utc")
    print("cols_b WITH:", cols_b_for(tr, cfg_with)[0])

    ids = tg.select("MVT_ID_mvt")
    res = pl.DataFrame(
        {
            "MVT_ID_mvt": ids["MVT_ID_mvt"], "ADEP_mvt": tg["ADEP_mvt"], "month": tg["MVT_TIME_UTC_mvt"].dt.month(),
            "ms": tg["mvt_minus_sched"], "RUNWAY_mvt": tg["RUNWAY_mvt"], "hour_utc": tg["hour_utc"], "hour_local": tg["hour_local"],
            "with_old": r_with["preds"]["old"], "with_head": r_with["preds"]["head"],
            "without_old": r_without["preds"]["old"], "without_head": r_without["preds"]["head"],
        }
    )
    refs = {
        "v8_det_a": REPO / "outputs/det_a.parquet", "v4_probe": REPO / "outputs/probe_unmatched_from_v4.parquet",
        "v7_main": REPO / "outputs/main.parquet", "v2_two_clock": REPO / "outputs/v2_two_clock.parquet",
    }
    for k, path in refs.items():
        r = pl.scan_parquet(path).select(["MVT_ID_mvt", pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias(k)]).collect()
        res = res.join(r, on="MVT_ID_mvt", how="left", maintain_order="left")
        assert res[k].null_count() == 0, k
    res.write_parquet(OUT_DIR / "preds.parquet")

    out: dict = {"frames": info, "with": {k: v for k, v in r_with.items() if k != "preds"}, "without": {k: v for k, v in r_without.items() if k != "preds"}}
    print("\ntail counts on the 5,290 unmatched ranking rows:")
    out["tails"] = {}
    for k in ("with_old", "with_head", "without_old", "without_head", "v2_two_clock", "v8_det_a", "v7_main", "v4_probe"):
        out["tails"][k] = tail(res[k].to_numpy())
        print(f"  {k:>13}", {kk: (round(vv, 1) if isinstance(vv, float) else vv) for kk, vv in out['tails'][k].items()})
    # LIRF-only tail counts
    lirf = res.filter(pl.col("ADEP_mvt") == "LIRF")
    out["tails_lirf"] = {k: tail(lirf[k].to_numpy()) for k in ("with_old", "with_head", "without_old", "without_head", "v8_det_a", "v4_probe")}
    print("LIRF-only n>43200:", {k: v["n_gt_43200"] for k, v in out["tails_lirf"].items()}, "n>14400:", {k: v["n_gt_14400"] for k, v in out["tails_lirf"].items()})

    ms = res["ms"].to_numpy()
    pairs = [("with_old", "without_old"), ("with_head", "without_head"), ("with_old", "with_head"), ("without_old", "without_head"),
             ("with_old", "v4_probe"), ("without_old", "v4_probe"), ("with_head", "v8_det_a"), ("without_head", "v8_det_a"), ("v8_det_a", "v4_probe")]
    out["rms"] = {}
    print("\nRMS differences (overall / ms>=2h / ms>=12h / LIRF):")
    for a, b in pairs:
        A, B = res[a].to_numpy(), res[b].to_numpy()
        d = {"overall": rmse(A, B), "ms_ge_2h": rmse(A[ms >= 7200], B[ms >= 7200]), "ms_ge_12h": rmse(A[ms >= 43200], B[ms >= 43200]),
             "lirf": rmse(A[(res["ADEP_mvt"] == "LIRF").to_numpy()], B[(res["ADEP_mvt"] == "LIRF").to_numpy()])}
        out["rms"][f"{a}~{b}"] = d
        print(f"  {a:>12} ~ {b:<12} " + "  ".join(f"{k}={v:8.1f}" for k, v in d.items()))

    show = res.select(["MVT_ID_mvt", "ADEP_mvt", "month", "ms", "RUNWAY_mvt", "hour_utc", "hour_local", "with_old", "without_old", "with_head", "without_head", "v4_probe", "v8_det_a"])
    with pl.Config(tbl_rows=40, tbl_cols=20, tbl_width_chars=220, float_precision=0):
        print("\nLIRF unmatched ranking rows with ms >= 12 h:")
        t11 = show.filter((pl.col("ADEP_mvt") == "LIRF") & (pl.col("ms") >= 43200)).sort("ms", descending=True)
        print(t11)
        print("\nall unmatched ranking rows where any variant or v4 predicts > 43,200:")
        tbig = show.filter((pl.col("with_old") > 43200) | (pl.col("without_old") > 43200) | (pl.col("with_head") > 43200) | (pl.col("without_head") > 43200) | (pl.col("v4_probe") > 43200)).sort("ms", descending=True)
        print(tbig)
    out["lirf_ms_ge_12h"] = t11.to_dicts()
    out["rows_any_gt_43200"] = tbig.to_dicts()
    out["seconds"] = round(time.time() - t0, 1)
    (OUT_DIR / "results.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"\ndone in {out['seconds']}s -> {OUT_DIR}")


if __name__ == "__main__":
    main()
