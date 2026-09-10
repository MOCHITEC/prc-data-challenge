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
