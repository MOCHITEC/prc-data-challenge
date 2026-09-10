"""Training / cross-validation for the two-clock model (docs/04 sections 4-5).

Model A (rows with an NM off-block time): learn g(x) = TAXITIME - (MVT_TIME - AOBT_3) and predict proxy + g(x).
    A tree cannot represent the identity part of y = proxy - f(x), so the reparameterisation is the point.
Model B (rows without one): learn TAXITIME directly from schedule-based features.
Both use L2 on raw seconds and keep every labelled row (no outlier filtering) - docs/04 section 4-3.

Invariants enforced here (each one was a real bug caught in review):
* Every label-derived table (stand x runway reference, airport-runway-hour median, airport median) is fitted on
  the training side of the fold and only joined onto the validation/ranking side.
* Early stopping never sees the evaluation fold: the stopping point comes from an inner holdout carved out of
  the training months, and the final model is refitted on all training rows with a fixed number of rounds. The
  same code path therefore produces the same model whether or not the target frame carries labels.
* Predictions are returned keyed by MVT_ID_mvt, never by row position.
"""

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

from .cv import Fold, metrics, month_block_folds, per_airport, year_month_key
from .features import CATEGORICALS, numeric_feature_columns

LABEL = "TAXITIME_SEC_mvt"
PROXY = "mvt_minus_aobt3"
ENCODING_COLS = ["ref_p10", "ref_p20", "ref_med", "ref_n", "arh_med", "arh_n", "apt_med", "ref_p10_fb", "ref_missing"]


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
            "min_data_in_leaf": 50,
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
    use_linear_tree: bool = False  # docs/03 section 8: reported worth about -10 s CLEAN RMSE; compare explicitly
    linear_lambda: float = 1.0
    reference_min_count: int = 30  # self-inclusion in a P10 over a small group is a leak of 1/n
    inner_holdout_months: int = 1  # months of the training side reserved for early stopping

    def hash(self) -> str:
        return hashlib.sha1(json.dumps(self.__dict__, sort_keys=True, default=str).encode()).hexdigest()[:10]

    def lgb_params(self) -> dict:
        p = dict(self.params, seed=self.seed)
        if self.use_linear_tree:
            p["linear_tree"] = True
            p["linear_lambda"] = self.linear_lambda
        return p


# ---------- label-derived tables: fitted on the training side only ----------


def reference_table(train_dep: pl.DataFrame, min_count: int) -> pl.DataFrame:
    """PRU-style unimpeded reference per airport x stand x runway (P10, P20, median, n)."""
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


def airport_runway_hour_table(train_dep: pl.DataFrame, min_count: int = 10) -> pl.DataFrame:
    lab = train_dep.filter(pl.col(LABEL).is_not_null())
    return (
        lab.group_by(["ADEP_mvt", "RUNWAY_mvt", "hour_local"])
        .agg([pl.col(LABEL).median().alias("arh_med"), pl.len().alias("arh_n")])
        .filter(pl.col("arh_n") >= min_count)
    )


def airport_median_table(train_dep: pl.DataFrame) -> pl.DataFrame:
    lab = train_dep.filter(pl.col(LABEL).is_not_null())
    return lab.group_by("ADEP_mvt").agg(pl.col(LABEL).median().alias("apt_med"))


def fit_encodings(train_dep: pl.DataFrame, cfg: TrainConfig) -> dict[str, pl.DataFrame]:
    return {
        "ref": reference_table(train_dep, cfg.reference_min_count),
        "arh": airport_runway_hour_table(train_dep),
        "apt": airport_median_table(train_dep),
    }


def attach_encodings(df: pl.DataFrame, enc: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Join the fitted tables onto any frame. Pure join: nothing is derived from df's own labels."""
    out = df.join(enc["ref"], on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], how="left", maintain_order="left")
    out = out.join(enc["arh"], on=["ADEP_mvt", "RUNWAY_mvt", "hour_local"], how="left", maintain_order="left")
    out = out.join(enc["apt"], on="ADEP_mvt", how="left", maintain_order="left")
    return out.with_columns(
        [
            pl.coalesce([pl.col("ref_p10"), pl.col("arh_med"), pl.col("apt_med")]).alias("ref_p10_fb"),
            pl.col("ref_n").is_null().cast(pl.Int8).alias("ref_missing"),
        ]
    )


# ---------- fitting ----------


def _matrix(df: pl.DataFrame, feature_cols: list[str], cat_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    cols = feature_cols + cat_cols
    X = df.select(cols)
    for c in cat_cols:
        X = X.with_columns(
            pl.col(c).cast(pl.Utf8).fill_null("__NA__").cast(pl.Categorical).to_physical().cast(pl.Float64).alias(c)
        )
    return X.to_numpy().astype(np.float64), cols


def _inner_split(train: pl.DataFrame, months: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Carve the latest `months` calendar months out of the training side for early stopping."""
    if months <= 0 or "MVT_TIME_UTC_mvt" not in train.columns:
        return train, train.head(0)
    keys = train.select(
        (pl.col("MVT_TIME_UTC_mvt").dt.year() * 100 + pl.col("MVT_TIME_UTC_mvt").dt.month()).alias("ym")
    )["ym"]
    uniq = sorted(keys.unique().to_list())
    if len(uniq) <= months:
        return train, train.head(0)
    hold = set(uniq[-months:])
    mask = keys.is_in(list(hold)).to_numpy()
    return train.filter(pl.Series(~mask)), train.filter(pl.Series(mask))


def _fit(
    train: pl.DataFrame,
    target_expr,
    cfg: TrainConfig,
    feature_cols: list[str],
    cat_cols: list[str],
    predict_frames: list[pl.DataFrame],
) -> tuple[list[np.ndarray], int]:
    """Early-stop on an inner holdout of `train`, refit on all of `train`, then predict each frame.

    `target_expr(frame) -> np.ndarray` builds the model-space target, so the caller decides the parameterisation.
    Returns one prediction array per frame (in model space) and the number of rounds actually used."""
    inner_tr, inner_va = _inner_split(train, cfg.inner_holdout_months)
    frames = [train, *predict_frames]
    if inner_va.height:
        frames = [inner_tr, inner_va, *frames]
    Xall, cols = _matrix(
        pl.concat([f.select(feature_cols + cat_cols) for f in frames], how="vertical_relaxed"), feature_cols, cat_cols
    )
    offs, i = [], 0
    for f in frames:
        offs.append((i, i + f.height))
        i += f.height
    cat_idx = [cols.index(c) for c in cat_cols]
    params = cfg.lgb_params()

    rounds = cfg.num_rounds
    if inner_va.height:
        (a, b), (c, d) = offs[0], offs[1]
        dtr = lgb.Dataset(
            Xall[a:b], label=target_expr(inner_tr), categorical_feature=cat_idx, free_raw_data=False, params=params
        )
        dva = lgb.Dataset(Xall[c:d], label=target_expr(inner_va), categorical_feature=cat_idx, reference=dtr)
        booster = lgb.train(
            params,
            dtr,
            num_boost_round=cfg.num_rounds,
            valid_sets=[dva],
            callbacks=[lgb.log_evaluation(0), lgb.early_stopping(cfg.early_stopping, verbose=False)],
        )
        best = int(booster.best_iteration or cfg.num_rounds)
        # scale the round count to the larger refit set, then cap
        rounds = int(min(cfg.num_rounds, max(10, round(best * train.height / max(1, inner_tr.height)))))
    fi = 2 if inner_va.height else 0
    (a, b) = offs[fi]
    dfull = lgb.Dataset(
        Xall[a:b], label=target_expr(train), categorical_feature=cat_idx, free_raw_data=False, params=params
    )
    model = lgb.train(params, dfull, num_boost_round=rounds, callbacks=[lgb.log_evaluation(0)])
    preds = [model.predict(Xall[s:e]) for (s, e) in offs[fi + 1 :]]
    return preds, rounds


def predict_fold(train_f: pl.DataFrame, target_f: pl.DataFrame, cfg: TrainConfig) -> tuple[pl.DataFrame, dict]:
    """Fit on train_f and predict target_f. Returns (DataFrame[MVT_ID_mvt, pred], info).

    target_f may be unlabelled (the ranking frame); the model is identical either way."""
    n_before = train_f.height
    train_f = train_f.filter(pl.col(LABEL).is_not_null() & pl.col(LABEL).is_finite())
    info: dict = {"train_rows": train_f.height, "dropped_unlabelled": n_before - train_f.height}
    enc = fit_encodings(train_f, cfg)
    tr, tg = attach_encodings(train_f, enc), attach_encodings(target_f, enc)
    if tg.height != target_f.height or not tg["MVT_ID_mvt"].equals(target_f["MVT_ID_mvt"]):
        raise AssertionError("attach_encodings changed row count or order")

    num_cols = [c for c in numeric_feature_columns(tr) if c not in (LABEL, "month")]
    cat_cols = [c for c in CATEGORICALS if c in tr.columns] if cfg.use_categoricals else []
    parts = []

    m_tr, m_tg = tr.filter(pl.col(PROXY).is_not_null()), tg.filter(pl.col(PROXY).is_not_null())
    if m_tg.height:
        if m_tr.height >= 200:
            cols_a = [c for c in num_cols if c != PROXY] + [PROXY]
            (g,), rounds = _fit(
                m_tr,
                lambda f: f[LABEL].to_numpy().astype(float) - f[PROXY].to_numpy().astype(float),
                cfg,
                cols_a,
                cat_cols,
                [m_tg],
            )
            info["rounds_matched"] = rounds
        else:
            info["rounds_matched"] = 0
            info["warning_matched"] = f"only {m_tr.height} labelled matched rows; predicting the proxy alone"
            g = np.zeros(m_tg.height)
        parts.append(pl.DataFrame({"MVT_ID_mvt": m_tg["MVT_ID_mvt"], "pred": m_tg[PROXY].to_numpy().astype(float) + g}))

    u_tr, u_tg = tr.filter(pl.col(PROXY).is_null()), tg.filter(pl.col(PROXY).is_null())
    if u_tg.height:
        cols_b = [
            c
            for c in num_cols
            if not c.startswith(
                ("mvt_minus_aobt3", "aobt3", "lobt_minus", "mvt_minus_lobt", "mvt_minus_eobt1", "mvt_minus_iobt")
            )
        ]
        if u_tr.height >= 200:
            (g,), rounds = _fit(u_tr, lambda f: f[LABEL].to_numpy().astype(float), cfg, cols_b, cat_cols, [u_tg])
            info["rounds_unmatched"] = rounds
        else:
            fallback = float(np.nanmedian(tr[LABEL].to_numpy().astype(float)))
            g = u_tg["ref_p10_fb"].fill_null(fallback).to_numpy().astype(float)
            info["rounds_unmatched"] = 0
        parts.append(pl.DataFrame({"MVT_ID_mvt": u_tg["MVT_ID_mvt"], "pred": g}))

    pred = pl.concat(parts) if parts else pl.DataFrame({"MVT_ID_mvt": tg["MVT_ID_mvt"].head(0), "pred": []})
    fallback = float(np.nanmedian(tr[LABEL].to_numpy().astype(float))) if tr.height else 0.0
    out = (
        target_f.select("MVT_ID_mvt")
        .join(pred, on="MVT_ID_mvt", how="left", maintain_order="left")
        .with_columns(pl.col("pred").fill_null(fallback).fill_nan(fallback).clip(lower_bound=0.0))
    )
    if out.height != target_f.height:
        raise AssertionError("prediction join changed the row count")
    return out, info


def feature_parity_report(train_f: pl.DataFrame, target_f: pl.DataFrame, cfg: TrainConfig) -> pl.DataFrame:
    """Null-rate of every model feature on the training side vs the target side.

    A feature that is dense in training and empty at prediction time (or vice versa) sends every row down one
    branch of the tree; docs/03 section 8 records a team losing a submission to exactly that."""
    enc = fit_encodings(train_f.filter(pl.col(LABEL).is_not_null()), cfg)
    tr, tg = attach_encodings(train_f, enc), attach_encodings(target_f, enc)
    cols = [c for c in numeric_feature_columns(tr) if c not in (LABEL, "month")]
    rows = [
        {
            "feature": c,
            "null_rate_train": tr[c].null_count() / max(1, tr.height),
            "null_rate_target": tg[c].null_count() / max(1, tg.height),
        }
        for c in cols
    ]
    return (
        pl.DataFrame(rows)
        .with_columns((pl.col("null_rate_target") - pl.col("null_rate_train")).abs().alias("gap"))
        .sort("gap", descending=True)
    )


def run_cv(feats: pl.DataFrame, cfg: TrainConfig, out_dir: Path = Path("experiments")) -> dict:
    """Month-blocked CV on a labelled departure feature frame. Writes OOF predictions and a JSONL record."""
    t0 = time.time()
    months = year_month_key(feats)
    folds: list[Fold] = month_block_folds(months.unique().to_list(), cfg.scheme)
    oof = pl.DataFrame({"MVT_ID_mvt": feats["MVT_ID_mvt"]}).with_columns(pl.lit(None, dtype=pl.Float64).alias("oof"))
    fold_metrics: dict[str, dict] = {}
    for f in folds:
        tr_f = feats.filter(months.is_in(list(f.train_months)))
        va_f = feats.filter(months.is_in(list(f.valid_months)))
        p, info = predict_fold(tr_f, va_f, cfg)
        joined = va_f.select(["MVT_ID_mvt", LABEL, "ADEP_mvt"]).join(p, on="MVT_ID_mvt", how="inner")
        y, pv = joined[LABEL].to_numpy().astype(float), joined["pred"].to_numpy().astype(float)
        fold_metrics[f.name] = {
            **metrics(y, pv),
            **info,
            "by_airport": per_airport(y, pv, joined["ADEP_mvt"].to_numpy()),
        }
        oof = (
            oof.join(p.rename({"pred": "_p"}), on="MVT_ID_mvt", how="left", maintain_order="left")
            .with_columns(pl.coalesce([pl.col("oof"), pl.col("_p")]).alias("oof"))
            .drop("_p")
        )
    scored = feats.select(["MVT_ID_mvt", LABEL]).join(oof, on="MVT_ID_mvt", how="inner").drop_nulls(["oof", LABEL])
    overall = metrics(scored[LABEL].to_numpy().astype(float), scored["oof"].to_numpy().astype(float))
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
    oof.write_parquet(out_dir / f"oof_{rec['exp_id']}.parquet")
    return rec


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
