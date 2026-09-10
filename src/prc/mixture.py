"""Mixture over clock sources: classify which clock wrote BLOCK_TIME, then route (docs/06 section 8, P1).

The target is MVT_TIME - BLOCK_TIME with MVT_TIME always known, so the problem is predicting BLOCK_TIME. For a
third of the labelled rows BLOCK_TIME is a copy of a timestamp we already hold, and for those rows the answer is
arithmetic rather than regression. This module therefore fits two heads:

  head 1  P(clock source | x), a multiclass classifier over {genuine, aobt3, sched, eobt1, lobt, iobt}
  head 2  E[taxi | x], the existing two-clock regressor, trained on the rows whose BLOCK_TIME really is an
          off-block time (source `genuine` or `aobt3` - AOBT_3 is the Network Manager's *actual* off-block, so
          those rows carry a true taxi time and dropping them would throw away a fifth of the signal)

and combines them with the L2-optimal posterior mean (prc.clocks.routed_prediction).

The invariants from prc.train carry over unchanged: label-derived tables are fitted on the training side only,
early stopping uses an inner holdout of the training months rather than the evaluation fold, and predictions are
keyed by MVT_ID_mvt rather than row position. Both heads see exactly the same feature columns.
"""

from __future__ import annotations

import gc

import numpy as np
import polars as pl

from .clocks import CANDIDATE_VALUE, GENUINE, classes, routed_prediction, with_candidates, with_clock_source
from .features import CATEGORICALS, numeric_feature_columns
from .train import LABEL, TrainConfig, _fit, attach_encodings, fit_encodings, predict_fold

# Sources whose BLOCK_TIME is an actual off-block time, so the row is a usable taxi-time observation.
REAL_OFFBLOCK_SOURCES = (GENUINE, "aobt3")

# Which clock a movement record copies is a property of the reporting chain - the airport's system, the handler
# filing the off-block, the stand, the time of day - far more than of the individual flight. So the rate at
# which each group historically produced each source is the natural feature for the classifier. Groups are
# ordered coarse to fine; a fine group with too few rows falls back on the global rate through the smoothing.
SOURCE_RATE_GROUPS: dict[str, list[str]] = {
    "apt": ["ADEP_mvt"],
    "apt_op": ["ADEP_mvt", "AIRCRAFT_OPERATOR_flt"],
    "apt_hour": ["ADEP_mvt", "hour_local"],
    "apt_stand": ["ADEP_mvt", "STAND_mvt"],
}


def source_rate_tables(train_src: pl.DataFrame, smoothing: float) -> dict[str, pl.DataFrame]:
    """Per-group counts of each clock source, fitted on the training side only.

    Counts rather than rates, because the rate has to be computed at attach time: on the training side a row's
    own class must be removed from its own group's counts first (see attach_source_rates)."""
    names = classes()
    n_all = train_src.height
    prior = {c: (train_src.filter(pl.col("clock_src") == c).height + 1) / (n_all + len(names)) for c in names}
    tables: dict[str, pl.DataFrame] = {"_prior": pl.DataFrame([{f"sr_prior_{c}": prior[c] for c in names}])}
    for tag, keys in SOURCE_RATE_GROUPS.items():
        if any(k not in train_src.columns for k in keys):
            continue
        wide = train_src.group_by([*keys, "clock_src"]).len().pivot(on="clock_src", index=keys, values="len")
        cnt_cols = []
        for c in names:  # a class absent from the training months still needs its column
            col = pl.col(c) if c in wide.columns else pl.lit(0, dtype=pl.UInt32)
            # fill only the count columns: filling the whole frame would put 0 into the string group keys
            cnt_cols.append(col.fill_null(0).cast(pl.Int64).alias(f"cnt_{tag}_{c}"))
        wide = wide.select([*keys, *cnt_cols])
        tables[tag] = wide.with_columns(
            pl.sum_horizontal([pl.col(f"cnt_{tag}_{c}") for c in names]).alias(f"srn_{tag}")
        )
    return tables


def attach_source_rates(
    df: pl.DataFrame, tables: dict[str, pl.DataFrame], smoothing: float, leave_one_out: bool = False
) -> pl.DataFrame:
    """Join the group counts and turn them into P(clock source | group).

    `leave_one_out` must be set on the training side and only there. Without it every training row's feature
    includes that row's own class, so the classifier sees a shaved-down version of its own label - the same
    self-inclusion defect that the stand x runway reference table guards against with min_count. At prediction
    time the row was never counted, so the feature is strictly less informative than in training, and the
    classifier's reliance on it is miscalibrated by exactly that amount.

    Rows whose group key is null (STAND_mvt and AIRCRAFT_OPERATOR_flt are both nullable) miss the join and fall
    back on the global prior; they were never counted, so leave-one-out subtracts nothing for them."""
    names = classes()
    prior = tables["_prior"]
    if leave_one_out and "clock_src" not in df.columns:
        raise ValueError("leave_one_out needs the clock_src column, which only labelled rows have")
    out = df
    for tag, keys in SOURCE_RATE_GROUPS.items():
        if tag not in tables:
            continue
        out = out.join(tables[tag], on=keys, how="left", maintain_order="left")
        joined = pl.col(f"srn_{tag}").is_not_null()
        # `own` is 1 for the class this row contributed to its own group and 0 otherwise; a row that missed the
        # join contributed nothing, so nothing is subtracted for it either
        if leave_one_out:
            own = {c: (joined & (pl.col("clock_src") == c)).cast(pl.Int64) for c in names}
            n = pl.when(joined).then(pl.col(f"srn_{tag}") - 1).otherwise(0)
        else:
            own = {c: pl.lit(0, dtype=pl.Int64) for c in names}
            n = pl.col(f"srn_{tag}").fill_null(0)
        out = out.with_columns(
            [
                (
                    (pl.col(f"cnt_{tag}_{c}").fill_null(0) - own[c] + smoothing * prior[f"sr_prior_{c}"][0])
                    / (n + smoothing)
                ).alias(f"sr_{tag}_{c}")
                for c in names
            ]
            + [pl.col(f"srn_{tag}").fill_null(0)]
        ).drop([f"cnt_{tag}_{c}" for c in names])
    if out.height != df.height:
        raise AssertionError("attach_source_rates changed the row count")
    return out


def _class_probs(train_src: pl.DataFrame, target: pl.DataFrame, cfg: TrainConfig) -> tuple[np.ndarray, dict]:
    """P(clock source | x) for every row of `target`."""
    names = classes()
    lookup = {c: i for i, c in enumerate(names)}
    num_cols = [c for c in numeric_feature_columns(train_src) if c not in (LABEL, "month")]
    cat_cols = [c for c in CATEGORICALS if c in train_src.columns] if cfg.use_categoricals else []
    ccfg = TrainConfig(
        **{
            **cfg.__dict__,
            "params": {**cfg.params, "objective": "multiclass", "num_class": len(names), "metric": "multi_logloss"},
            "num_rounds": cfg.clf_num_rounds,
            "use_linear_tree": False,  # a linear leaf model makes no sense for a softmax objective
        }
    )
    (probs,), rounds = _fit(
        train_src,
        lambda f: f["clock_src"].replace_strict(lookup, return_dtype=pl.Int32).to_numpy().astype(float),
        ccfg,
        num_cols,
        cat_cols,
        [target],
    )
    if probs.shape != (target.height, len(names)):
        raise AssertionError(f"classifier returned {probs.shape}, expected {(target.height, len(names))}")
    info = {
        "clf_rounds": rounds,
        "clf_classes": names,
        "clf_train_mix": {k: v for k, v in train_src["clock_src"].value_counts().iter_rows()},
    }
    return probs, info


def _candidate_matrix(df: pl.DataFrame) -> np.ndarray:
    """(n, 5) of what the target would be under each copied clock; NaN where that clock is missing."""
    cols = [CANDIDATE_VALUE[k] for k in CANDIDATE_VALUE]
    return df.select([pl.col(c).cast(pl.Float64) for c in cols]).to_numpy()


def predict_fold_mixture(train_f: pl.DataFrame, target_f: pl.DataFrame, cfg: TrainConfig) -> tuple[pl.DataFrame, dict]:
    """Same contract as prc.train.predict_fold: fit on train_f, return (DataFrame[MVT_ID_mvt, pred], info).

    target_f may be unlabelled - nothing here reads its label, so the ranking path and the CV path build the
    identical model."""
    n_before = train_f.height
    train_f = train_f.filter(pl.col(LABEL).is_not_null() & pl.col(LABEL).is_finite())
    info: dict = {"train_rows": train_f.height, "dropped_unlabelled": n_before - train_f.height}

    train_src = with_clock_source(with_candidates(train_f), cfg.match_tol_s, with_rollover=False)
    target_c = with_candidates(target_f)

    # head 2 first: it owns its own encodings, and freeing its frames before the classifier keeps the peak down
    real = train_src.filter(pl.col("clock_src").is_in(list(REAL_OFFBLOCK_SOURCES)))
    if real.height < 200:
        raise ValueError(f"only {real.height} rows with a real off-block time; cannot fit the taxi-time head")
    gen_pred, ginfo = predict_fold(real.drop("clock_src"), target_c, cfg)
    info.update({f"gen_{k}": v for k, v in ginfo.items()})
    if not gen_pred["MVT_ID_mvt"].equals(target_c["MVT_ID_mvt"]):
        raise AssertionError("the taxi-time head returned rows in a different order than the target frame")
    f_gen = gen_pred["pred"].to_numpy().astype(float)
    del real, gen_pred
    gc.collect()

    # head 1: the encodings the classifier sees are fitted on the training side only, as for the regressor
    enc = fit_encodings(train_src, cfg)
    tr_e, tg_e = attach_encodings(train_src, enc), attach_encodings(target_c, enc)
    del train_src  # tr_e carries clock_src, so the pre-encoding copy of the training frame is dead weight here
    gc.collect()
    if cfg.use_source_rates:
        rates = source_rate_tables(tr_e, cfg.source_rate_smoothing)
        tr_e = attach_source_rates(tr_e, rates, cfg.source_rate_smoothing, leave_one_out=True)
        tg_e = attach_source_rates(tg_e, rates, cfg.source_rate_smoothing, leave_one_out=False)
        info["source_rate_groups"] = [t for t in rates if t != "_prior"]
    if tg_e.height != target_c.height or not tg_e["MVT_ID_mvt"].equals(target_c["MVT_ID_mvt"]):
        raise AssertionError("attach_encodings changed the target row count or order")
    probs, cinfo = _class_probs(tr_e, tg_e, cfg)
    info.update(cinfo)
    del tr_e, tg_e
    gc.collect()

    pred = routed_prediction(probs, _candidate_matrix(target_c), f_gen, classes())
    routed_share = float((probs[:, 1:].sum(1) > 0.5).mean())
    info["routed_share_over_half"] = routed_share
    info["mean_p_genuine"] = float(probs[:, 0].mean())
    out = pl.DataFrame({"MVT_ID_mvt": target_c["MVT_ID_mvt"], "pred": pred}).with_columns(
        pl.col("pred").fill_nan(None).fill_null(float(np.nanmedian(f_gen))).clip(lower_bound=0.0)
    )
    if out.height != target_f.height:
        raise AssertionError("prediction frame changed the row count")
    return out, info
