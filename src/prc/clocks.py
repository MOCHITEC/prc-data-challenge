"""Which clock wrote BLOCK_TIME_UTC_mvt? (docs/06)

The target is TAXITIME = MVT_TIME - BLOCK_TIME and MVT_TIME is never missing at ranking time, so predicting the
target is exactly predicting BLOCK_TIME. BLOCK_TIME is whatever the airport's reporting system wrote there, and
measurement shows it is often a *copy* of another timestamp we already know: the Network Manager's actual
off-block (AOBT_3), the airport's own scheduled off-block (SCHED_TIME), or one of the NM estimates. For those
rows the answer is a known quantity, so the modelling problem is classification, not regression.

This module owns the definitions shared by the anatomy report, the mixture model and the tests.
"""

from __future__ import annotations

import numpy as np
import polars as pl

# The candidate clocks, in the priority order used to build a mutually exclusive label. AOBT_3 first because it
# is the most informative (an actual, not a planned, time); SCHED next because it is the only one that is never
# missing. IOBT/LOBT are near-duplicates of each other and jointly rare, so they share a class.
CANDIDATES: dict[str, str] = {
    "aobt3": "AOBT_3_flt",
    "sched": "SCHED_TIME_UTC_mvt",
    "eobt1": "EOBT_1_flt",
    "lobt": "LOBT_flt",
    "iobt": "IOBT_flt",
}
# What the target would be if BLOCK_TIME were a copy of that clock. These are the columns prc.features already
# builds (MVT_TIME - clock), so the routing target and the model feature are the same number by construction.
CANDIDATE_VALUE: dict[str, str] = {
    "aobt3": "mvt_minus_aobt3",
    "sched": "mvt_minus_sched",
    "eobt1": "mvt_minus_eobt1",
    "lobt": "mvt_minus_lobt",
    "iobt": "mvt_minus_iobt",
}
# Classes the model routes over. "genuine" means BLOCK_TIME matches no candidate, i.e. it really is a measured
# off-block time and the taxi time has to be regressed.
GENUINE = "genuine"
ROLLOVER = "rollover_24h"
MATCH_TOL_S = 60  # docs/06 section 1: the copy spike lives inside +-5s, the +-60s band still gives 38s of slack
ROLLOVER_TOL_S = 5400
DAY_S = 86_400


def with_candidates(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure the candidate value columns exist, recomputing them from the timestamps.

    prc.features already produces these for the model, so on a feature frame this is an identity; it exists so
    the anatomy report and the tests can work from raw timestamps. The candidate columns are cast to the MVT
    column's dtype first: a frame filtered down to rows without an NM record carries an all-null AOBT_3, which
    polars types as Null and refuses to subtract."""
    dtype = df.schema["MVT_TIME_UTC_mvt"]
    return df.with_columns(
        [
            (pl.col("MVT_TIME_UTC_mvt") - pl.col(col).cast(dtype)).dt.total_seconds().alias(CANDIDATE_VALUE[name])
            for name, col in CANDIDATES.items()
        ]
    )


def with_clock_source(df: pl.DataFrame, tol: int = MATCH_TOL_S, with_rollover: bool = True) -> pl.DataFrame:
    """Add the mutually exclusive clock_src label. Needs BLOCK_TIME, so labelled rows only."""
    dtype = df.schema["BLOCK_TIME_UTC_mvt"]

    def gap(col: str) -> pl.Expr:
        return (pl.col("BLOCK_TIME_UTC_mvt") - pl.col(col).cast(dtype)).dt.total_seconds().abs()

    if with_rollover:
        # BLOCK_TIME's date is one day early and the remainder is an ordinary taxi time (docs/06 section 4).
        chain = pl.when((pl.col("TAXITIME_SEC_mvt").cast(pl.Float64) - DAY_S).abs() < ROLLOVER_TOL_S).then(
            pl.lit(ROLLOVER)
        )
    else:
        chain = pl.when(pl.lit(False)).then(pl.lit(GENUINE))
    for name, col in CANDIDATES.items():
        chain = chain.when(gap(col) <= tol).then(pl.lit(name))
    return df.with_columns(chain.otherwise(pl.lit(GENUINE)).alias("clock_src"))


def classes(with_rollover: bool = False) -> list[str]:
    """Class list for the classifier. GENUINE is index 0 so argmax ties fall back to regression."""
    out = [GENUINE, *CANDIDATES]
    if with_rollover:
        out.append(ROLLOVER)
    return out


def routed_prediction(
    probs: np.ndarray, cand: np.ndarray, genuine_pred: np.ndarray, class_names: list[str]
) -> np.ndarray:
    """Posterior-mean prediction over the clock sources.

    Under L2 the optimum is E[y|x] = sum_k P(k|x) E[y|x,k]; for a copied clock E[y|x,k] is the known candidate
    value. Probability mass on a candidate that is null for this row (no NM record) cannot be spent, so it is
    moved onto the genuine branch rather than dropped - dropping it would shrink every such prediction toward
    zero. `cand` is (n, len(CANDIDATE_VALUE)) in CANDIDATE_VALUE order and may contain NaN.

    ROLLOVER, if present as a class, contributes DAY_S + genuine_pred: docs/06 section 4 measured the remainder
    after removing the day offset to be an ordinary taxi time."""
    if probs.shape != (len(genuine_pred), len(class_names)):
        raise ValueError(f"probs {probs.shape} does not match {len(genuine_pred)} rows x {len(class_names)} classes")
    names = list(CANDIDATE_VALUE)
    if cand.shape != (len(genuine_pred), len(names)):
        raise ValueError(f"cand {cand.shape} does not match {len(genuine_pred)} rows x {len(names)} candidates")
    num = np.zeros(len(genuine_pred))
    weight = np.zeros(len(genuine_pred))
    spare = np.zeros(len(genuine_pred))  # mass that has to fall back on the regression
    roll = np.zeros(len(genuine_pred))
    for k, name in enumerate(class_names):
        p = probs[:, k]
        if name == GENUINE:
            spare += p
        elif name == ROLLOVER:
            roll += p
        else:
            c = cand[:, names.index(name)]
            ok = np.isfinite(c)
            num += np.where(ok, p * np.nan_to_num(c), 0.0)
            weight += np.where(ok, p, 0.0)
            spare += np.where(ok, 0.0, p)
    total = weight + spare + roll
    return (num + spare * genuine_pred + roll * (DAY_S + genuine_pred)) / total
