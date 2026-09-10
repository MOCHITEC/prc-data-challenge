"""Training / cross-validation for the two-clock model (docs/04 §4).

Model A (NM-matched rows): target = TAXITIME - (MVT_TIME - AOBT_3)  i.e. BLOCK_TIME - AOBT_3 with sign so that
    prediction = (MVT_TIME - AOBT_3) - f(x)   ... we learn g(x) = TAXITIME - proxy and predict proxy + g(x).
Model B (unmatched rows): target = TAXITIME (raw seconds), schedule-based features.
Both L2 on raw seconds, all rows kept (no outlier filtering), per docs/04 §4-3.
Label-dependent encodings (stand x runway reference, target encodings) are fitted on the training side of each fold."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from .cv import Fold, metrics, month_block_folds, per_airport
from .features import CATEGORICALS, numeric_feature_columns

LABEL = "TAXITIME_SEC_mvt"
PROXY = "mvt_minus_aobt3"


@dataclass
class TrainConfig:
    name: str = "v1_two_clock"
    scheme: str = "fast"  # or "full"
    seed: int = 42
    params: dict = field(
        default_factory=lambda: {
            "objective": "regression",
            "learning_rate": 0.05,
            "num_leaves": 127,
            "min_data_in_leaf": 100,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "verbose": -1,
            "num_threads": 4,
            "deterministic": True,
        }
    )
    num_rounds: int = 2000
    early_stopping: int = 100
    use_categoricals: bool = True
    reference_min_count: int = 10

    def hash(self) -> str:
        return hashlib.sha1(json.dumps(self.__dict__, sort_keys=True, default=str).encode()).hexdigest()[:10]


# ---------- fold-local, label-dependent encodings ----------


def reference_table(train_dep: pl.DataFrame, min_count: int) -> pl.DataFrame:
    """PRU-style unimpeded reference per airport x stand x runway from the training side (P10, P20, median, n)."""
    lab = train_dep.filter(pl.col(LABEL).is_not_null() & (pl.col(LABEL) > 0) & (pl.col(LABEL) < 7200))
    return (
        lab.group_by(["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"])
        .agg(
            [
                pl.col(LABEL).quantile(0.10).alias("ref_p10"),
                pl.col(LABEL).quantile(0.20).alias("ref_p20"),
                pl.col(LABEL).median().alias("ref_med"),
                pl.len().alias("ref_n"),
            ]
        )
        .filter(pl.col("ref_n") >= min_count)
    )


def airport_runway_hour_table(train_dep: pl.DataFrame) -> pl.DataFrame:
    lab = train_dep.filter(pl.col(LABEL).is_not_null())
    return lab.group_by(["ADEP_mvt", "RUNWAY_mvt", "hour_local"]).agg(
        [pl.col(LABEL).median().alias("arh_med"), pl.len().alias("arh_n")]
    )


def attach_encodings(df: pl.DataFrame, ref: pl.DataFrame, arh: pl.DataFrame) -> pl.DataFrame:
    out = df.join(ref, on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], how="left")
    out = out.join(arh, on=["ADEP_mvt", "RUNWAY_mvt", "hour_local"], how="left")
    # hierarchical fallback for unseen stand x runway: airport x runway x hour median, then airport median
    apt_med = (
        df.filter(pl.col(LABEL).is_not_null()).group_by("ADEP_mvt").agg(pl.col(LABEL).median().alias("apt_med"))
        if LABEL in df.columns
        else None
    )
    if apt_med is not None and apt_med.height:
        out = out.join(apt_med, on="ADEP_mvt", how="left")
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("apt_med"))
    out = out.with_columns(
        [
            pl.coalesce([pl.col("ref_p10"), pl.col("arh_med"), pl.col("apt_med")]).alias("ref_p10_fb"),
            pl.col("ref_n").is_null().cast(pl.Int8).alias("ref_missing"),
        ]
    )
    return out


def _matrix(df: pl.DataFrame, feature_cols: list[str], cat_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    cols = feature_cols + cat_cols
    X = df.select(cols)
    for c in cat_cols:
        X = X.with_columns(
            pl.col(c).cast(pl.Utf8).fill_null("__NA__").cast(pl.Categorical).to_physical().cast(pl.Float64).alias(c)
        )
    return X.to_numpy().astype(np.float64), cols


def _fit_predict(
    train: pl.DataFrame,
    valid: pl.DataFrame,
    target: np.ndarray,
    cfg: TrainConfig,
    feature_cols: list[str],
    cat_cols: list[str],
    valid_target: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Fit LightGBM on `train` (labels = target, already in model space) and predict `valid`.
    If `valid_target` (same model space, all rows) is given, it is used for early stopping."""
    both = pl.concat(
        [train.select(feature_cols + cat_cols), valid.select(feature_cols + cat_cols)], how="vertical_relaxed"
    )
    Xb, cols = _matrix(both, feature_cols, cat_cols)
    Xtr, Xva = Xb[: train.height], Xb[train.height :]
    cat_idx = [cols.index(c) for c in cat_cols]
    dtr = lgb.Dataset(Xtr, label=target, categorical_feature=cat_idx, free_raw_data=False)
    callbacks = [lgb.log_evaluation(0)]
    valid_sets = []
    if valid_target is not None:
        valid_sets = [lgb.Dataset(Xva, label=valid_target, categorical_feature=cat_idx, reference=dtr)]
        callbacks.append(lgb.early_stopping(cfg.early_stopping, verbose=False))
    params = dict(cfg.params, seed=cfg.seed)
    booster = lgb.train(params, dtr, num_boost_round=cfg.num_rounds, valid_sets=valid_sets or None, callbacks=callbacks)
    return booster.predict(Xva, num_iteration=booster.best_iteration or None), int(
        booster.best_iteration or cfg.num_rounds
    )


def predict_fold(train_f: pl.DataFrame, valid_f: pl.DataFrame, cfg: TrainConfig) -> tuple[np.ndarray, dict]:
    """Fit the two-model system on train_f and predict valid_f (both feature frames from build_departure_features)."""
    ref = reference_table(train_f, cfg.reference_min_count)
    arh = airport_runway_hour_table(train_f)
    tr = attach_encodings(train_f, ref, arh)
    va = (
        attach_encodings(valid_f, ref, arh)
        if LABEL in valid_f.columns
        else attach_encodings(valid_f.with_columns(pl.lit(None, dtype=pl.Float64).alias(LABEL)), ref, arh)
    )
    num_cols = [c for c in numeric_feature_columns(tr) if c not in (LABEL, "month")]
    cat_cols = [c for c in CATEGORICALS if c in tr.columns] if cfg.use_categoricals else []
    pred = np.full(va.height, np.nan)
    info: dict = {}
    # Model A: matched rows -> learn residual over the NM proxy
    m_tr, m_va = tr.filter(pl.col(PROXY).is_not_null()), va.filter(pl.col(PROXY).is_not_null())
    if m_tr.height and m_va.height:
        y = m_tr[LABEL].to_numpy().astype(float) - m_tr[PROXY].to_numpy().astype(float)
        yv = None
        if m_va[LABEL].null_count() == 0:
            yv = m_va[LABEL].to_numpy().astype(float) - m_va[PROXY].to_numpy().astype(float)
        g, it = _fit_predict(m_tr, m_va, y, cfg, [c for c in num_cols if c != PROXY] + [PROXY], cat_cols, yv)
        idx = va.with_row_index("_i").filter(pl.col(PROXY).is_not_null())["_i"].to_numpy()
        pred[idx] = m_va[PROXY].to_numpy().astype(float) + g
        info["iters_matched"] = it
    # Model B: unmatched rows -> raw target on schedule-based features
    u_tr, u_va = tr.filter(pl.col(PROXY).is_null()), va.filter(pl.col(PROXY).is_null())
    if u_va.height:
        if u_tr.height >= 200:
            y = u_tr[LABEL].to_numpy().astype(float)
            cols_b = [
                c
                for c in num_cols
                if not c.startswith(
                    ("mvt_minus_aobt3", "aobt3", "lobt", "mvt_minus_lobt", "mvt_minus_eobt1", "mvt_minus_iobt")
                )
            ]
            yv = u_va[LABEL].to_numpy().astype(float) if u_va[LABEL].null_count() == 0 else None
            g, it = _fit_predict(u_tr, u_va, y, cfg, cols_b, cat_cols, yv)
            info["iters_unmatched"] = it
        else:  # fallback: reference/hour median
            g = (
                u_va["ref_p10_fb"]
                .fill_null(u_tr[LABEL].median() if u_tr.height else tr[LABEL].median())
                .to_numpy()
                .astype(float)
            )
        idx = va.with_row_index("_i").filter(pl.col(PROXY).is_null())["_i"].to_numpy()
        pred[idx] = g
    pred = np.where(np.isnan(pred), np.nanmedian(tr[LABEL].to_numpy().astype(float)), pred)
    pred = np.maximum(pred, 0.0)
    return pred, info


def run_cv(feats: pl.DataFrame, cfg: TrainConfig, out_dir: Path = Path("experiments")) -> dict:
    """Month-blocked CV on a labelled departure feature frame. Writes OOF predictions and a JSONL record."""
    t0 = time.time()
    months = feats["MVT_TIME_UTC_mvt"].dt.month()
    folds: list[Fold] = month_block_folds(months.unique().to_list(), cfg.scheme)
    oof = np.full(feats.height, np.nan)
    fold_metrics = {}
    for f in folds:
        tr_mask = months.is_in(list(f.train_months)).to_numpy()
        va_mask = months.is_in(list(f.valid_months)).to_numpy()
        tr_f, va_f = feats.filter(pl.Series(tr_mask)), feats.filter(pl.Series(va_mask))
        p, info = predict_fold(tr_f, va_f, cfg)
        oof[va_mask] = p
        y = va_f[LABEL].to_numpy().astype(float)
        fold_metrics[f.name] = {**metrics(y, p), **info, "by_airport": per_airport(y, p, va_f["ADEP_mvt"].to_numpy())}
    have = ~np.isnan(oof)
    y_all = feats[LABEL].to_numpy().astype(float)
    overall = metrics(y_all[have], oof[have])
    rec = {
        "exp_id": f"{cfg.name}-{cfg.hash()}",
        "config": {**cfg.__dict__},
        "git_sha": _git_sha(),
        "folds": fold_metrics,
        "overall": overall,
        "train_seconds": round(time.time() - t0, 1),
        "n_features": len(numeric_feature_columns(feats)),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "experiments.jsonl", "a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    pl.DataFrame({"MVT_ID_mvt": feats["MVT_ID_mvt"], "oof": oof}).write_parquet(
        out_dir / f"oof_{rec['exp_id']}.parquet"
    )
    return rec


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
