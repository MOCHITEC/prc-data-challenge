import datetime as dt

import numpy as np
import polars as pl
import pytest

from prc.clocks import (
    CANDIDATE_VALUE,
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
    assert g[CANDIDATE_VALUE["aobt3"]][0] == float(g["TAXITIME_SEC_mvt"][0])  # the routed answer is exact
    assert g[CANDIDATE_VALUE["sched"]][0] == 720.0
    assert g[CANDIDATE_VALUE["lobt"]][0] is None  # a missing clock stays missing, it does not become 0


def test_routed_prediction_picks_an_available_candidate():
    names = list(CANDIDATE_VALUE)
    cls = classes()
    probs = np.zeros((1, len(cls)))
    probs[0, cls.index("sched")] = 1.0
    cand = np.full((1, len(names)), np.nan)
    cand[0, names.index("sched")] = 4321.0
    assert routed_prediction(probs, cand, np.array([900.0]), cls)[0] == pytest.approx(4321.0)


def test_mass_on_a_missing_candidate_falls_back_instead_of_shrinking():
    """Half the mass on a null AOBT_3 must not drag the prediction toward zero."""
    names, cls = list(CANDIDATE_VALUE), classes()
    probs = np.zeros((1, len(cls)))
    probs[0, cls.index("aobt3")] = 0.5
    probs[0, cls.index(GENUINE)] = 0.5
    cand = np.full((1, len(names)), np.nan)
    assert routed_prediction(probs, cand, np.array([900.0]), cls)[0] == pytest.approx(900.0)


def test_soft_mixture_is_the_posterior_mean():
    names, cls = list(CANDIDATE_VALUE), classes()
    probs = np.zeros((1, len(cls)))
    probs[0, cls.index("sched")] = 0.25
    probs[0, cls.index(GENUINE)] = 0.75
    cand = np.full((1, len(names)), np.nan)
    cand[0, names.index("sched")] = 2000.0
    got = routed_prediction(probs, cand, np.array([1000.0]), cls)[0]
    assert got == pytest.approx(0.25 * 2000.0 + 0.75 * 1000.0)


def test_rollover_class_adds_a_day_to_the_regression():
    names, cls = list(CANDIDATE_VALUE), classes(with_rollover=True)
    probs = np.zeros((1, len(cls)))
    probs[0, cls.index(ROLLOVER)] = 1.0
    cand = np.full((1, len(names)), np.nan)
    assert routed_prediction(probs, cand, np.array([900.0]), cls)[0] == pytest.approx(DAY_S + 900.0)


def test_shape_mismatch_is_an_error_not_a_broadcast():
    cls = classes()
    with pytest.raises(ValueError):
        routed_prediction(np.zeros((2, len(cls))), np.zeros((2, len(CANDIDATE_VALUE))), np.array([900.0]), cls)
    with pytest.raises(ValueError):
        routed_prediction(np.zeros((1, len(cls))), np.zeros((1, len(CANDIDATE_VALUE) - 1)), np.array([900.0]), cls)


def test_genuine_is_class_zero():
    assert classes()[0] == GENUINE and classes(with_rollover=True)[0] == GENUINE


def test_temperature_softens_and_sharpens():
    from prc.clocks import apply_temperature

    p = np.array([[0.7, 0.2, 0.1]])
    assert apply_temperature(p, 1.0) is p  # identity is free
    soft = apply_temperature(p, 5.0)
    sharp = apply_temperature(p, 0.2)
    assert soft.sum() == pytest.approx(1.0) and sharp.sum() == pytest.approx(1.0)
    assert soft[0, 0] < p[0, 0] < sharp[0, 0]  # softening moves mass toward uniform
    assert soft[0, 2] > p[0, 2] > sharp[0, 2]
    assert soft.argmax() == sharp.argmax() == p.argmax()  # monotone, so the ranking never changes


def test_fit_temperature_softens_an_overconfident_classifier_on_a_huge_candidate():
    """The v6 failure in miniature: confident sched routing on a 100,000s candidate that is usually wrong."""
    from prc.clocks import fit_temperature

    rng = np.random.default_rng(0)
    n = 2000
    cls = classes()
    names = list(CANDIDATE_VALUE)
    # one row in ten really is schedule-reported; the classifier says 0.9 for all of them
    truly_sched = rng.random(n) < 0.1
    probs = np.zeros((n, len(cls)))
    probs[:, cls.index("sched")] = 0.9
    probs[:, cls.index(GENUINE)] = 0.1
    cand = np.full((n, len(names)), np.nan)
    cand[:, names.index("sched")] = 100_000.0
    genuine_pred = np.full(n, 900.0)
    y = np.where(truly_sched, 100_000.0, 900.0)

    best, curve = fit_temperature(probs, cand, genuine_pred, y, cls)
    assert best > 1.0, f"expected softening, got {best} with curve {curve}"
    assert curve[best] < curve[1.0]  # and it must actually beat the uncalibrated routing


def test_fit_temperature_leaves_a_calibrated_classifier_alone():
    from prc.clocks import fit_temperature

    cls = classes()
    names = list(CANDIDATE_VALUE)
    n = 1000
    probs = np.zeros((n, len(cls)))
    probs[:, cls.index("sched")] = 1.0  # certain, and right
    cand = np.full((n, len(names)), np.nan)
    cand[:, names.index("sched")] = 1200.0
    best, curve = fit_temperature(probs, cand, np.full(n, 900.0), np.full(n, 1200.0), cls)
    assert best <= 1.0 and curve[best] == pytest.approx(0.0, abs=1e-6)


def test_fit_temperature_declines_to_guess_from_too_few_rows():
    from prc.clocks import fit_temperature

    cls = classes()
    t, curve = fit_temperature(
        np.zeros((10, len(cls))), np.zeros((10, len(CANDIDATE_VALUE))), np.zeros(10), np.zeros(10), cls
    )
    assert t == 1.0 and curve == {}
