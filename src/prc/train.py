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

import gc
import hashlib
import json
import subprocess
import time
from collections.abc import Callable
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
    # prc.mixture only (docs/06): tolerance for calling BLOCK_TIME a copy of another clock, and the round budget
    # for the clock-source classifier, which converges much sooner than the regressor.
    match_tol_s: int = 60
    clf_num_rounds: int = 400
    use_source_rates: bool = False  # historical P(clock source | airport, operator, stand, hour)
    source_rate_smoothing: float = 50.0
    # Temperature-scale the classifier against the routed squared error on the inner holdout. docs/06 8-0-4:
    # a posterior mean is only optimal when the probabilities are calibrated, and the cost of miscalibration
    # scales with the candidate value being multiplied.
    calibrate_temperature: bool = False

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


def _cat_dtypes(frames: list[pl.DataFrame], cat_cols: list[str]) -> dict[str, pl.Enum]:
    """One fixed category list per column, shared by every frame.

    polars assigns Categorical codes per frame, so encoding the training and prediction frames separately would
    give the same string different codes. A pl.Enum with an explicit category list is stable across frames, which
    lets us build one matrix per frame instead of concatenating them (the concatenated matrix is what ran the
    machine out of memory at full size)."""
    out: dict[str, pl.Enum] = {}
    for c in cat_cols:
        vals: set[str] = set()
        for f in frames:
            vals |= set(f[c].cast(pl.Utf8).drop_nulls().unique().to_list())
        out[c] = pl.Enum(["__NA__", *sorted(vals)])
    return out


def _matrix(
    df: pl.DataFrame, feature_cols: list[str], cat_cols: list[str], cat_dtypes: dict[str, pl.Enum]
) -> tuple[np.ndarray, list[str]]:
    """Dense float32 design matrix. float32 halves the footprint and LightGBM bins the values anyway."""
    cols = feature_cols + cat_cols
    X = df.select(cols).with_columns(
        [pl.col(c).cast(pl.Utf8).fill_null("__NA__").cast(cat_dtypes[c]).to_physical() for c in cat_cols]
    )
    X = X.with_columns([pl.col(c).cast(pl.Float32) for c in cols])
    return X.to_numpy(), cols


def _inner_holdout_mask(train: pl.DataFrame, months: int, min_share: float = 0.02) -> np.ndarray:
    """Boolean mask marking the latest `months` substantial calendar months of the training side.

    Used for early stopping and, in prc.mixture, for calibrating the routing temperature.

    "Substantial" matters: a movement file whose last flights spill past a month boundary contributes a month key
    holding a handful of rows, and taking that as the holdout silently reduces early stopping to those rows. The
    synthetic generator does exactly this, which is how the defect surfaced. Months holding less than `min_share`
    of the training rows are therefore not eligible as the holdout and simply stay on the training side."""
    empty = np.zeros(train.height, dtype=bool)
    if months <= 0 or "MVT_TIME_UTC_mvt" not in train.columns or train.height == 0:
        return empty
    keys = year_month_key(train)
    counts = dict(zip(*[s.to_list() for s in keys.value_counts().sort("ym")], strict=True))
    eligible = [m for m in sorted(counts) if counts[m] >= min_share * train.height]
    if len(eligible) <= months or len(counts) <= months:
        return empty
    return keys.is_in(eligible[-months:]).to_numpy()


def _fit(
    train: pl.DataFrame,
    target_fn,
    cfg: TrainConfig,
    feature_cols: list[str],
    cat_cols: list[str],
    predict_frames: list[pl.DataFrame],
    want_holdout: bool = False,
) -> tuple[list[np.ndarray], int, tuple[np.ndarray, np.ndarray] | None]:
    """Early-stop on an inner holdout of `train`, refit on all of `train`, then predict each frame.

    `target_fn(frame) -> np.ndarray` builds the model-space target, so the caller owns the parameterisation.
    Returns one prediction array per frame (in model space), the number of rounds actually used, and - when
    `want_holdout` is set - the inner holdout's boolean mask together with the *early-stopping* model's
    predictions on it.

    Those holdout predictions are the only genuinely out-of-sample predictions this function ever makes: the
    early-stopping model is fitted on the training months excluding the holdout, so anything calibrated on them
    is calibrated honestly, and the evaluation fold is never involved. The model is otherwise discarded, so
    asking for them costs one extra predict and no extra training."""
    cat_dtypes = _cat_dtypes([train, *predict_frames], cat_cols)
    Xtr, cols = _matrix(train, feature_cols, cat_cols, cat_dtypes)
    cat_idx = [cols.index(c) for c in cat_cols]
    params = cfg.lgb_params()
    y = target_fn(train)
    if not np.isfinite(y).all():
        raise AssertionError(f"{int((~np.isfinite(y)).sum())} non-finite training targets reached the fitter")

    dfull = lgb.Dataset(Xtr, label=y, categorical_feature=cat_idx, free_raw_data=False, params=params)
    rounds = cfg.num_rounds
    hold = _inner_holdout_mask(train, cfg.inner_holdout_months)
    holdout: tuple[np.ndarray, np.ndarray] | None = None
    if hold.any() and not hold.all():
        idx_in, idx_ho = np.flatnonzero(~hold), np.flatnonzero(hold)
        # subset() reuses the parent's binned data instead of materialising two more dense copies
        booster = lgb.train(
            params,
            dfull.subset(idx_in),
            num_boost_round=cfg.num_rounds,
            valid_sets=[dfull.subset(idx_ho)],
            callbacks=[lgb.log_evaluation(0), lgb.early_stopping(cfg.early_stopping, verbose=False)],
        )
        best = int(booster.best_iteration or cfg.num_rounds)
        # the refit sees more rows than the inner split, so scale the round count up in proportion
        rounds = int(min(cfg.num_rounds, max(10, round(best * len(y) / max(1, len(idx_in))))))
        if want_holdout:
            holdout = (hold, booster.predict(Xtr[idx_ho], num_iteration=best))
        del booster
        gc.collect()

    model = lgb.train(params, dfull, num_boost_round=rounds, callbacks=[lgb.log_evaluation(0)])
    del dfull, Xtr
    gc.collect()

    preds = []
    for f in predict_frames:
        Xp, _ = _matrix(f, feature_cols, cat_cols, cat_dtypes)
        preds.append(model.predict(Xp))
        del Xp
        gc.collect()
    return preds, rounds, holdout


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
            (g,), rounds, _ = _fit(
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
            (g,), rounds, _ = _fit(u_tr, lambda f: f[LABEL].to_numpy().astype(float), cfg, cols_b, cat_cols, [u_tg])
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


def run_cv(
    feats: pl.DataFrame,
    cfg: TrainConfig,
    out_dir: Path = Path("experiments"),
    predictor: Callable[[pl.DataFrame, pl.DataFrame, TrainConfig], tuple[pl.DataFrame, dict]] | None = None,
) -> dict:
    """Month-blocked CV on a labelled departure feature frame. Writes OOF predictions and a JSONL record.

    `predictor` has predict_fold's contract; pass prc.mixture.predict_fold_mixture to cross-validate the
    clock-source mixture instead. Whatever is passed must be the same function the ranking path uses, or the
    validated model is not the submitted one."""
    t0 = time.time()
    predictor = predictor or predict_fold
    months = year_month_key(feats)
    folds: list[Fold] = month_block_folds(months.unique().to_list(), cfg.scheme)
    oof = pl.DataFrame({"MVT_ID_mvt": feats["MVT_ID_mvt"]}).with_columns(pl.lit(None, dtype=pl.Float64).alias("oof"))
    fold_metrics: dict[str, dict] = {}
    for f in folds:
        tr_f = feats.filter(months.is_in(list(f.train_months)))
        va_f = feats.filter(months.is_in(list(f.valid_months)))
        p, info = predictor(tr_f, va_f, cfg)
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
        "predictor": getattr(predictor, "__name__", str(predictor)),
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
