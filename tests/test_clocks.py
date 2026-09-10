import datetime as dt

import numpy as np
import polars as pl
import pytest

from prc.clocks import (
    CANDIDATES,
    DAY_S,
    GENUINE,
    ROLLOVER,
    classes,
    routed_prediction,
    with_candidates,
    with_clock_source,
)

T0 = dt.datetime(2025, 3, 4, 8, 0, 0)


def _frame(rows: list[dict]) -> pl.DataFrame:
    """rows give offsets in seconds from T0 for each clock; taxi is derived so the frame stays consistent."""
    out = []
    for r in rows:
        block = T0 + dt.timedelta(seconds=r["block"])
        mvt = T0 + dt.timedelta(seconds=r["mvt"])
        rec = {
            "MVT_TIME_UTC_mvt": mvt,
            "BLOCK_TIME_UTC_mvt": block,
            "TAXITIME_SEC_mvt": int((mvt - block).total_seconds()),
        }
        for name, col in CANDIDATES.items():
            v = r.get(name)
            rec[col] = None if v is None else T0 + dt.timedelta(seconds=v)
        out.append(rec)
    return pl.DataFrame(out)


def test_source_label_priority_and_tolerance():
    f = _frame(
        [
            # BLOCK equals AOBT_3 exactly and SCHED too: AOBT_3 wins the priority order
            {"block": 0, "mvt": 900, "aobt3": 0, "sched": 0, "eobt1": 0, "lobt": 0, "iobt": 0},
            # 59s from SCHED is inside the +-60s band, AOBT_3 is far away
            {"block": 0, "mvt": 900, "aobt3": 5000, "sched": 59, "eobt1": 9000, "lobt": 9000, "iobt": 9000},
            # 61s from every candidate: nothing matches
            {"block": 0, "mvt": 900, "aobt3": 61, "sched": -61, "eobt1": 61, "lobt": 61, "iobt": 61},
            # only EOBT_1 matches
            {"block": 0, "mvt": 900, "aobt3": 4000, "sched": 4000, "eobt1": 10, "lobt": 4000, "iobt": 4000},
            # no NM record at all, BLOCK is a real off-block
            {"block": 0, "mvt": 900, "sched": 4000},
        ]
    )
    got = with_clock_source(f)["clock_src"].to_list()
    assert got == ["aobt3", "sched", GENUINE, "eobt1", GENUINE]


def test_rollover_outranks_a_clock_match():
    """A row whose BLOCK_TIME is a day early is a rollover even if some clock happens to sit next to it."""
    f = _frame([{"block": 0, "mvt": DAY_S + 900, "aobt3": 0, "sched": 0, "eobt1": 0, "lobt": 0, "iobt": 0}])
    assert with_clock_source(f)["clock_src"].to_list() == [ROLLOVER]
    assert with_clock_source(f, with_rollover=False)["clock_src"].to_list() == ["aobt3"]


def test_candidate_equals_the_label_when_block_is_a_copy():
    f = _frame([{"block": 120, "mvt": 1020, "aobt3": 120, "sched": 300}])
    g = with_candidates(f)
    assert g["cand_aobt3"][0] == float(g["TAXITIME_SEC_mvt"][0])  # the routed answer is exact
    assert g["cand_sched"][0] == 720.0
    assert g["cand_lobt"][0] is None  # a missing clock stays missing, it does not become 0


def test_routed_prediction_picks_an_available_candidate():
    names = list(CANDIDATES)
    cls = classes()
    probs = np.zeros((1, len(cls)))
    probs[0, cls.index("sched")] = 1.0
    cand = np.full((1, len(names)), np.nan)
    cand[0, names.index("sched")] = 4321.0
    assert routed_prediction(probs, cand, np.array([900.0]), cls)[0] == pytest.approx(4321.0)


def test_mass_on_a_missing_candidate_falls_back_instead_of_shrinking():
    """Half the mass on a null AOBT_3 must not drag the prediction toward zero."""
    names, cls = list(CANDIDATES), classes()
    probs = np.zeros((1, len(cls)))
    probs[0, cls.index("aobt3")] = 0.5
    probs[0, cls.index(GENUINE)] = 0.5
    cand = np.full((1, len(names)), np.nan)
    assert routed_prediction(probs, cand, np.array([900.0]), cls)[0] == pytest.approx(900.0)


def test_soft_mixture_is_the_posterior_mean():
    names, cls = list(CANDIDATES), classes()
    probs = np.zeros((1, len(cls)))
    probs[0, cls.index("sched")] = 0.25
    probs[0, cls.index(GENUINE)] = 0.75
    cand = np.full((1, len(names)), np.nan)
    cand[0, names.index("sched")] = 2000.0
    got = routed_prediction(probs, cand, np.array([1000.0]), cls)[0]
    assert got == pytest.approx(0.25 * 2000.0 + 0.75 * 1000.0)


def test_rollover_class_adds_a_day_to_the_regression():
    names, cls = list(CANDIDATES), classes(with_rollover=True)
    probs = np.zeros((1, len(cls)))
    probs[0, cls.index(ROLLOVER)] = 1.0
    cand = np.full((1, len(names)), np.nan)
    assert routed_prediction(probs, cand, np.array([900.0]), cls)[0] == pytest.approx(DAY_S + 900.0)


def test_shape_mismatch_is_an_error_not_a_broadcast():
    cls = classes()
    with pytest.raises(ValueError):
        routed_prediction(np.zeros((2, len(cls))), np.zeros((2, len(CANDIDATES))), np.array([900.0]), cls)
    with pytest.raises(ValueError):
        routed_prediction(np.zeros((1, len(cls))), np.zeros((1, len(CANDIDATES) - 1)), np.array([900.0]), cls)


def test_genuine_is_class_zero():
    assert classes()[0] == GENUINE and classes(with_rollover=True)[0] == GENUINE
