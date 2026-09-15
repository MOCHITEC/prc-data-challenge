"""Training / cross-validation for the two-clock model (docs/README sections 4-5).

Model A (rows with an NM off-block time): learn g(x) = TAXITIME - (MVT_TIME - AOBT_3) and predict proxy + g(x).
    A tree cannot represent the identity part of y = proxy - f(x), so the reparameterisation is the point.
Model B (rows without one): learn TAXITIME directly from schedule-based features.

Splitting the two regimes is measured, not assumed. One model over everything - same reparameterisation, target
LABEL - coalesce(proxy, 0), free to branch on nm_matched - is worse by 74.10 s overall and still worse by 6.04 s
once the 2,000 largest errors are removed, so it is not a tail artefact. It loses on both groups at once
(matched 218.5 -> 226.2, unmatched 1,875.3 -> 2,697.2): one model has to spend its capacity telling the regimes
apart before it can fit either. Do not merge them back.
Both use L2 on raw seconds and keep every labelled row (no outlier filtering) - docs/README.

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
from dataclasses import dataclass, field, replace
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
            # deterministic alone does not bind: LightGBM still picks row-wise or column-wise histogram
            # construction by timing the first iterations, and on a loaded machine that choice flips, changing
            # the floating-point summation order and therefore the splits. Two runs whose only difference was
            # in model B's settings - which model A is fitted before and cannot see - gave model A 1273 and 959
            # rounds on the same fold. The documentation asks for one of force_row_wise / force_col_wise
            # alongside deterministic; row-wise is the one for many rows and few bins, which is this data
            # (2.08M rows, 72 features, 255 bins).
            "deterministic": True,
            "force_row_wise": True,
        }
    )
    num_rounds: int = 2000
    early_stopping: int = 100
    use_categoricals: bool = True
    use_linear_tree: bool = False  # docs/README: reported worth about -10 s CLEAN RMSE; compare explicitly
    linear_lambda: float = 1.0
    # The nine label-derived columns (stand x runway P10/P20/median, airport x runway x hour median, airport
    # median, and the coalesced fallback). Present since the first commit and never ablated; the flag exists to
    # settle whether they earn the 40 lines of fold-wise fitting they cost.
    use_encodings: bool = True
    reference_min_count: int = 30  # self-inclusion in a P10 over a small group is a leak of 1/n
    inner_holdout_months: int = 1  # months of the training side reserved for early stopping
    # The rows with no NM record are 1.2% of the data and roughly half the squared error. Model B sees ~22k
    # rows where model A sees 2.06M, so the obvious guess was that it needed *less* capacity, and measurement
    # says otherwise: 31 leaves is worse than 127, 255 is better than 127, 511 adds nothing. The effect is
    # small once the largest errors are excluded (-0.26 s at 2,000 dropped), so this is a weak result, not a
    # strong one - it is the current setting because nothing contradicts it, not because it was worth 34 s.
    unmatched_num_leaves: int = 255
    unmatched_min_data_in_leaf: int = 20
    unmatched_num_rounds: int = 3000
    # Averaging over seeds removes the luck of one random draw: the 2024 winner reported 1,612 -> 1,562 from
    # 20 seeds (docs/README). Cost is linear in n_seeds, so keep it small until the effect is measured.
    n_seeds: int = 1
    # Which folds to run. All three is the honest answer for a final number, but a screening run does not need
    # them: the matched rows of the forward fold alone are 324k, giving that statistic a resolution of about
    # 1.1 s - the same as the full three-fold set - at a third of the wall clock. forward is also the only fold
    # shaped like the real task, since jan and jul train on months that come after the month they predict.
    folds: tuple[str, ...] | None = None
    # Columns hidden from the model. Candidate features are always computed, so measuring one at a time is a
    # config change rather than a four-minute rebuild of the feature cache.
    exclude_features: tuple[str, ...] = ()

    def hash(self) -> str:
        return hashlib.sha1(json.dumps(self.__dict__, sort_keys=True, default=str).encode()).hexdigest()[:10]

    def for_unmatched(self) -> TrainConfig:
        """A copy sized for model B's much smaller training set."""
        return replace(
            self,
            params={
                **self.params,
                "num_leaves": self.unmatched_num_leaves,
                "min_data_in_leaf": self.unmatched_min_data_in_leaf,
            },
            num_rounds=self.unmatched_num_rounds,
        )

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
    """Design matrix as float32. LightGBM bins the input anyway, so float32 costs no accuracy and halves the
    matrix: at 2.1M rows x 60 features that is the difference between 1.7 GB and 0.85 GB."""
    cols = feature_cols + cat_cols
    X = df.select(cols)
    for c in cat_cols:
        X = X.with_columns(
            pl.col(c).cast(pl.Utf8).fill_null("__NA__").cast(pl.Categorical).to_physical().cast(pl.Float32).alias(c)
        )
    return X.cast(pl.Float32).to_numpy(), cols


def _inner_holdout_mask(train: pl.DataFrame, months: int) -> np.ndarray:
    """Boolean mask of the latest `months` calendar months of the training side, used for early stopping.

    A mask rather than two frames: the caller indexes one design matrix with it instead of materialising the
    training rows a second time."""
    empty = np.zeros(train.height, dtype=bool)
    if months <= 0 or "MVT_TIME_UTC_mvt" not in train.columns:
        return empty
    keys = train.select(
        (pl.col("MVT_TIME_UTC_mvt").dt.year() * 100 + pl.col("MVT_TIME_UTC_mvt").dt.month()).alias("ym")
    )["ym"]
    uniq = sorted(keys.unique().to_list())
    if len(uniq) <= months:
        return empty
    return keys.is_in(uniq[-months:]).to_numpy()


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
    hold = _inner_holdout_mask(train, cfg.inner_holdout_months)
    frames = [train, *predict_frames]
    stacked = pl.concat([f.select(feature_cols + cat_cols) for f in frames], how="vertical_relaxed")
    Xall, cols = _matrix(stacked, feature_cols, cat_cols)
    del stacked
    offs, i = [], 0
    for f in frames:
        offs.append((i, i + f.height))
        i += f.height
    cat_idx = [cols.index(c) for c in cat_cols]
    params = cfg.lgb_params()

    # target_expr is elementwise, so evaluate it once over the training block and index the result rather than
    # filtering the frame twice (each filter copies every column of a multi-million-row frame).
    a, b = offs[0]
    y = target_expr(train)
    rounds = cfg.num_rounds
    n_hold = int(hold.sum())
    if 0 < n_hold < train.height:
        Xtrain = Xall[a:b]
        dtr = lgb.Dataset(Xtrain[~hold], label=y[~hold], categorical_feature=cat_idx, params=params)
        dva = lgb.Dataset(Xtrain[hold], label=y[hold], categorical_feature=cat_idx, reference=dtr)
        booster = lgb.train(
            params,
            dtr,
            num_boost_round=cfg.num_rounds,
            valid_sets=[dva],
            callbacks=[lgb.log_evaluation(0), lgb.early_stopping(cfg.early_stopping, verbose=False)],
        )
        best = int(booster.best_iteration or cfg.num_rounds)
        # scale the round count to the larger refit set, then cap
        rounds = int(min(cfg.num_rounds, max(10, round(best * train.height / max(1, train.height - n_hold)))))
        del dtr, dva, booster, Xtrain  # the fancy-indexed copies are the peak of this function

    dfull = lgb.Dataset(Xall[a:b], label=y, categorical_feature=cat_idx, free_raw_data=False, params=params)
    # Refit under each seed and average. One seed is the default, and then this is the single fit it was before.
    acc = [np.zeros(e - s) for (s, e) in offs[1:]]
    for k in range(max(1, cfg.n_seeds)):
        p_k = dict(params, seed=cfg.seed + k, bagging_seed=cfg.seed + k, feature_fraction_seed=cfg.seed + k)
        model = lgb.train(p_k, dfull, num_boost_round=rounds, callbacks=[lgb.log_evaluation(0)])
        for i, (s_, e_) in enumerate(offs[1:]):
            acc[i] += model.predict(Xall[s_:e_])
    preds = [a_ / max(1, cfg.n_seeds) for a_ in acc]
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
    if not cfg.use_encodings:
        num_cols = [c for c in num_cols if c not in ENCODING_COLS]
    cat_cols = [c for c in CATEGORICALS if c in tr.columns] if cfg.use_categoricals else []
    if cfg.exclude_features:
        hidden = set(cfg.exclude_features)
        num_cols = [c for c in num_cols if c not in hidden]
        cat_cols = [c for c in cat_cols if c not in hidden]
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
            (g,), rounds = _fit(
                u_tr, lambda f: f[LABEL].to_numpy().astype(float), cfg.for_unmatched(), cols_b, cat_cols, [u_tg]
            )
            info["rounds_unmatched"] = rounds
        else:
            fallback = float(np.nanmedian(tr[LABEL].to_numpy().astype(float)))
            g = u_tg["ref_p10_fb"].fill_null(fallback).to_numpy().astype(float)
            info["rounds_unmatched"] = 0
        parts.append(pl.DataFrame({"MVT_ID_mvt": u_tg["MVT_ID_mvt"], "pred": g}))

    return _assemble(parts, tr, tg, target_f, info)


def _assemble(parts: list[pl.DataFrame], tr: pl.DataFrame, tg: pl.DataFrame, target_f: pl.DataFrame, info: dict):
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
    branch of the tree; docs/README records a team losing a submission to exactly that."""
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
    log_name: str = "experiments.jsonl",
) -> dict:
    """Month-blocked CV on a labelled departure feature frame. Writes OOF predictions and a JSONL record.

    log_name exists so a synthetic run cannot land in the real-data experiment log: its RMSE level is an
    artefact of the generator, and mixing the two would corrupt the CV-to-LB correspondence we track."""
    t0 = time.time()
    months = year_month_key(feats)
    folds: list[Fold] = month_block_folds(months.unique().to_list())
    if cfg.folds:
        wanted = set(cfg.folds)
        folds = [f for f in folds if f.name in wanted]
        if not folds:
            raise ValueError(f"cfg.folds={cfg.folds} matched none of the available folds")
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
    with open(out_dir / log_name, "a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    oof.write_parquet(out_dir / f"oof_{rec['exp_id']}.parquet")
    return rec


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
