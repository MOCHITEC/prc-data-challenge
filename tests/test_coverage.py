"""P8-D: the coverage-parity gate.

Every rule is tested with a control beside it - a column that must NOT fire - because the failure this repo
keeps hitting is a check that passes everything. A gate that flags nothing and a gate that flags everything
both look "green enough" in a summary line, and both are useless.

The mutants these tests are written to kill, one per rule:
  * `nn_drop > DROP` -> `>=` or a flipped sign                    : test_nonnull_collapse_* (control column)
  * the `and` in the two-reference non-zero rule -> `or`          : test_nonzero_drop_needs_both_references
  * the DEAD rule deleted, or applied to the ranking side         : test_dead_column_fails_though_its_drop_is_zero
  * DEAD's threshold raised back to the plan's literal 0.1%       : test_near_dead_is_reported_not_failed
  * a rise promoted to a failure                                  : test_a_rise_is_reported_not_failed
  * the null-aware roll-up reverted to a plain weighted mean      : test_categoricals_are_never_reported_dead
  * the column set taken from `frame.columns`                     : test_gate_reads_the_columns_the_model_sees
  * the month-matched reference taken from the bare `month` column: test_month_matched_reference_is_the_year_before
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from prc.coverage import (
    ALL_AIRPORTS,
    ALL_PERIODS,
    _month_matched,
    check_coverage,
    coverage_frame,
    evaluate,
)
from prc.train import TrainConfig

AIRPORTS = ["EDDF", "LIRF"]


def _frame(periods: list[int], cols: dict, airports: list[str] = AIRPORTS, per_airport: int = 200) -> pl.DataFrame:
    """A feature frame with `per_airport` rows per (airport, period).

    `cols` maps a column name to a callable (airport, period, i) -> value, so a test can make one cell of the
    (column x airport x period) grid behave differently from every other cell.
    """
    rows = []
    for p in periods:
        when = dt.datetime(p // 100, p % 100, 15, 12, 0)
        for a in airports:
            for i in range(per_airport):
                rows.append({"ADEP_mvt": a, "MVT_TIME_UTC_mvt": when, **{c: f(a, p, i) for c, f in cols.items()}})
    return pl.DataFrame(rows).with_columns(pl.col("MVT_TIME_UTC_mvt").cast(pl.Datetime("us")))


TRAIN_PERIODS = [202501, 202507, 202511]
RANK_PERIODS = [202601, 202607]


def _run(train_cols: dict, rank_cols: dict, columns: list[str] | None = None, **kw) -> dict:
    numeric = sorted(train_cols)
    tr = _frame(TRAIN_PERIODS, train_cols)
    rk = _frame(RANK_PERIODS, rank_cols)
    return evaluate(
        coverage_frame(tr, numeric),
        coverage_frame(rk, sorted(rank_cols)),
        columns if columns is not None else numeric,
        rank_columns=sorted(rank_cols),
        **kw,
    )


def _rules(rep: dict, key: str, rule: str) -> list[dict]:
    return [r for r in rep[key] if r["rule"] == rule]


# ---------- (b) non-null collapse: the ec_* failure mode ----------


def test_nonnull_collapse_fails_and_a_stable_column_does_not():
    """A feed that is present in training and absent for one airport in one scored month is the -13.14 s
    defect the gate exists for. The control column is present everywhere and must stay silent."""
    live = {
        "ec_like": lambda a, p, i: 1.0,
        "stable": lambda a, p, i: float(i + 1),
    }
    served = {
        # present everywhere except LIRF in July 2026, where every value is null
        "ec_like": lambda a, p, i: None if (a == "LIRF" and p == 202607) else 1.0,
        "stable": lambda a, p, i: float(i + 1),
    }
    rep = _run(live, served)
    fails = _rules(rep, "failures", "NONNULL")
    assert [(f["column"], f["airport"], f["period"]) for f in fails] == [("ec_like", "LIRF", "202607")]
    assert fails[0]["drop_pts"] == pytest.approx(100.0)
    assert not [f for f in rep["failures"] if f["column"] == "stable"]
    # The same cell must ALSO fail the non-zero rule, because a null is a zero to the model: LightGBM's
    # numerical decision converts NaN to 0.0 when the split's feature had no missing value in training, which
    # is the mechanism that made this defect cost -13.14 s. A gate that computes the non-zero share over the
    # non-null rows only reports `nan` here and stays silent about the thing that actually happens at serve
    # time, so the null-counted-as-zero convention is asserted and not left to the docstring.
    nz = _rules(rep, "failures", "NONZERO")
    assert [(f["column"], f["airport"], f["period"]) for f in nz] == [("ec_like", "LIRF", "202607")]
    assert nz[0]["drop_pts"] == pytest.approx(100.0)


# ---------- (c) non-zero collapse, against two references ----------


def test_nonzero_drop_needs_both_references():
    """Seasonality is absorbed by the month-matched reference, and that conjunction is load-bearing.

    `seasonal` is 0 all winter and non-zero all summer, so a scored January looks like a 60-point collapse
    against the whole of 2025 and like nothing at all against January 2025. It must be reported, never
    failed. `broken` collapses against both references and must fail."""

    def nonzero_summer(a, p, i):
        return 0.0 if p % 100 == 1 else 5.0

    rep = _run(
        {"seasonal": nonzero_summer, "broken": lambda a, p, i: 5.0},
        {"seasonal": nonzero_summer, "broken": lambda a, p, i: 0.0},
    )
    assert {f["column"] for f in _rules(rep, "failures", "NONZERO")} == {"broken"}
    partial = _rules(rep, "warnings", "NONZERO_PARTIAL")
    assert {w["column"] for w in partial} == {"seasonal"}
    assert {w["period"] for w in partial} == {"202601"}


# ---------- (d) the extension: a column that was never alive ----------


def test_dead_column_fails_though_its_drop_is_zero():
    """The whole point of §5.3. `precip_like` is the literal 0.0 on both sides, so every share comparison the
    competitor's gate makes returns exactly 0 points of drop and passes it. `rare` is non-zero on 1% of rows -
    sparse, but real - and must not fail."""
    both = {
        "precip_like": lambda a, p, i: 0.0,
        "rare": lambda a, p, i: 1.0 if i % 100 == 0 else 0.0,
    }
    rep = _run(both, dict(both))
    dead = _rules(rep, "failures", "DEAD")
    assert [d["column"] for d in dead] == ["precip_like"]
    assert dead[0]["airport"] == ALL_AIRPORTS and dead[0]["period"] == ALL_PERIODS
    assert dead[0]["train_nonzero_pct"] == pytest.approx(0.0)
    # and the drop the competitor's rule looks at really is zero, so nothing else could have caught it
    assert not [f for f in rep["failures"] if f["rule"] in ("NONNULL", "NONZERO")]
    assert not [f for f in rep["failures"] if f["column"] == "rare"]


def test_near_dead_is_reported_not_failed():
    """0.0993% (`diverted`) is below the plan's literal 0.1% rule and above the fail tier. A build must not
    break on a column that is merely rare."""
    rep = _run(
        {"near_dead": lambda a, p, i: 1.0 if i == 0 else 0.0},  # 1 in 200 = 0.5%
        {"near_dead": lambda a, p, i: 1.0 if i == 0 else 0.0},
        near_dead_nonzero_pct=1.0,
        dead_nonzero_pct=0.001,
    )
    assert not rep["failures"]
    assert [w["column"] for w in _rules(rep, "warnings", "NEAR_DEAD")] == ["near_dead"]


# ---------- rises are reported, never failed ----------


def test_a_rise_is_reported_not_failed():
    """`deicing_gate` at LSZH rising in January 2026 is the real world and the reason weather might help."""
    rep = _run(
        {"gate_like": lambda a, p, i: 1.0 if i < 20 else 0.0},  # 10%
        {"gate_like": lambda a, p, i: 1.0 if i < 180 else 0.0},  # 90%
    )
    assert not rep["failures"]
    rises = _rules(rep, "warnings", "RISE")
    assert rises and all(r["rise_pts"] == pytest.approx(80.0) for r in rises)


# ---------- (a) absent column ----------


def test_a_column_absent_from_the_ranking_frame_fails():
    rep = _run(
        {"kept": lambda a, p, i: float(i), "gone": lambda a, p, i: float(i)},
        {"kept": lambda a, p, i: float(i)},
        columns=["kept", "gone"],
    )
    assert [(f["rule"], f["column"]) for f in rep["failures"]] == [("MISSING", "gone")]


# ---------- waivers ----------


def test_a_waiver_moves_a_failure_and_must_carry_a_reason():
    cols = {"precip_like": lambda a, p, i: 0.0}
    rep = _run(cols, dict(cols), allow={"DEAD:precip_like": "known dead, deleted by P8-B"})
    assert not rep["failures"]
    assert rep["waived"][0]["waiver_reason"] == "known dead, deleted by P8-B"
    with pytest.raises(ValueError, match="written reason"):
        _run(cols, dict(cols), allow={"DEAD:precip_like": ""})


# ---------- the two near-misses this module actually hit while being written ----------


def test_categoricals_are_never_reported_dead():
    """polars' sum treats a null as 0, so an n-weighted roll-up of a column whose non-zero share is null
    returns 0.0 - and every categorical would be failed as a constant. The roll-up is null-aware."""
    tr = _frame(TRAIN_PERIODS, {"num": lambda a, p, i: float(i + 1), "ADES_mvt": lambda a, p, i: "EGLL"})
    rk = _frame(RANK_PERIODS, {"num": lambda a, p, i: float(i + 1), "ADES_mvt": lambda a, p, i: "EGLL"})
    cov = coverage_frame(tr, ["num"], ["ADES_mvt"])
    roll = cov.filter(
        (pl.col("column") == "ADES_mvt") & (pl.col("airport") == ALL_AIRPORTS) & (pl.col("period") == ALL_PERIODS)
    )
    assert roll["nonzero_pct"].to_list() == [None]
    rep = evaluate(cov, coverage_frame(rk, ["num"], ["ADES_mvt"]), ["num", "ADES_mvt"])
    assert not rep["failures"]


def test_a_null_and_an_explicit_zero_are_the_same_thing_to_the_gate():
    """`half_null` is null on half the training rows and an explicit 0.0 on half the ranking rows, with the
    same 50% of live values on both sides. Counting the non-zero share over the non-null rows only would read
    the training side as 100% non-zero and call this a 50-point collapse; counting a null as a zero - which is
    what LightGBM does to it - reads 50% on both sides and correctly says nothing happened."""
    rep = _run(
        {"half_null": lambda a, p, i: None if i % 2 else 5.0},
        {"half_null": lambda a, p, i: 0.0 if i % 2 else 5.0},
    )
    assert not rep["failures"], rep["failures"]


def test_month_matched_reference_is_the_year_before():
    """A bare `month` column would make the scored January its own reference, and the whole month-matched
    arm of rule (c) would compare a frame against itself."""
    assert _month_matched("202601") == "202501"
    assert _month_matched("202607") == "202507"


def test_gate_reads_the_columns_the_model_sees_not_the_frame_width():
    """A candidate column is in the frame and hidden from the model unless a recipe names it. A gate built on
    `frame.columns` reports on columns nobody trains on - the exact duplication `visible_numeric_columns`
    exists to end (train.py:305-317)."""
    cols = {
        "temp_c": lambda a, p, i: 10.0,
        "noise_control": lambda a, p, i: 0.0,  # a CANDIDATE_COL, and dead: it would fail if it were visible
        "TAXITIME_SEC_mvt": lambda a, p, i: 600.0,
    }
    tr, rk = _frame(TRAIN_PERIODS, cols), _frame(RANK_PERIODS, cols)
    hidden = check_coverage(tr, rk, TrainConfig())
    assert not hidden["failures"]
    assert "noise_control" not in [c["column"] for c in hidden["failures"]]
    shown = check_coverage(tr, rk, TrainConfig(include_candidates=("noise_control",)))
    assert [(f["rule"], f["column"]) for f in shown["failures"]] == [("DEAD", "noise_control")]


# ---------- acceptance: the numbers P8-D §5.5 says the gate must produce on the real frames ----------

FEAT_TRAIN = Path("data/work/feat_train.parquet")
FEAT_RANK = Path("data/work/feat_rank.parquet")


@pytest.mark.skipif(not (FEAT_TRAIN.exists() and FEAT_RANK.exists()), reason="built feature frames not on disk")
def test_acceptance_on_the_real_frames():
    """P8-D's acceptance test, verbatim from the plan §5.5: four strict `dewp_spread_c` failures at LEMD and
    LSZH in January 2026 - independently reproduced from kind-mango's own four - the five identically-zero
    columns, and the LFPG plan-time rise as a report-only line."""
    import yaml

    spec = yaml.safe_load(Path("configs/main_gap.yaml").read_text())
    fields = {k: v for k, v in spec.items() if k in TrainConfig.__dataclass_fields__}
    for k in ("folds", "exclude_features", "include_candidates", "unmatched_sched_reparam_airports"):
        if isinstance(fields.get(k), list):
            fields[k] = tuple(fields[k])
    cfg = TrainConfig(**fields)
    rep = check_coverage(pl.read_parquet(FEAT_TRAIN), pl.read_parquet(FEAT_RANK), cfg)

    nonzero = sorted((f["column"], f["airport"], f["period"]) for f in rep["failures"] if f["rule"] == "NONZERO")
    assert nonzero == [
        ("dewp_spread_c", "LEMD", "202601"),
        ("dewp_spread_c", "LSZH", "202601"),
        ("dewp_spread_c_prev1h", "LEMD", "202601"),
        ("dewp_spread_c_prev1h", "LSZH", "202601"),
    ]
    assert sorted(f["column"] for f in rep["failures"] if f["rule"] == "DEAD") == [
        "precip_mm",
        "precip_mm_prev1h",
        "runway_null",
        "temp_x_precip",
        "temp_x_precip_prev1h",
    ]
    assert not [f for f in rep["failures"] if f["rule"] in ("MISSING", "NONNULL")]
    rises = {(w["column"], w["airport"], w["period"]) for w in rep["warnings"] if w["rule"] == "RISE"}
    assert ("lobt_minus_eobt1", "LFPG", "202607") in rises
