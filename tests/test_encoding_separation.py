"""P2 Fix 1 - the early-stopping holdout must not read its own labels.

Plan: `docs/competitor_research_2026-09-21/plans/P2-cross-fit-encodings.md` §4 Step 2 and §7 (T1, T2,
T6). The defect is BC-1: `predict_fold` fits the four label-derived tables on every labelled training
row and then `_fit` carves the inner early-stopping holdout out of that same already-encoded frame, so
the rows that choose `best_iteration` read encodings their own labels helped build.

The order of the tests is the point. `test_es_holdout_reads_its_own_labels_today` asserts the BEFORE
state and is expected to keep passing forever, because the repair is behind a config flag that defaults
to the current behaviour; it exists so the repair cannot be declared unnecessary later
(merry-quicksand's `test_incumbent_leak_is_real` discipline). Everything named `..._separated_...` or
`..._fix1_...` asserts the AFTER state and fails on the unrepaired code.

Every sensitivity test perturbs labels in TWO directions - the holdout month's and another month's -
because a test that only checks "nothing moved" passes just as happily on a column that is constant or
entirely null. This repo has shipped a silent no-op twice; the second direction is the control.
"""

from __future__ import annotations

from functools import lru_cache

import lightgbm as lgb
import numpy as np
import polars as pl
import pytest

from prc.cv import year_month_key
from prc.features import build_departure_features
from prc.synth import generate_month
from prc.train import (
    ENCODING_COLS,
    LABEL,
    OFFSET_COLS,
    PROXY,
    TrainConfig,
    _es_encoder,
    _inner_holdout_mask,
    attach_encodings,
    feature_parity_report,
    fit_encodings,
    predict_fold,
    reference_table,
)

VALUE_COLS = ["ref_p10", "ref_p20", "ref_med", "arh_med", "apt_med", "ref_p10_fb"]
BUMP = 600.0  # seconds; large against a taxi time, so a moved statistic is unmistakable


@lru_cache(maxsize=4)
def _feats(months=(1, 2, 3), seed0=41, per_day=60) -> pl.DataFrame:
    """60 departures per airport per day is not padding: `reference_table` keeps only stand x runway
    groups of at least `reference_min_count` = 30, and at the 8/day the rest of the suite uses the ref
    table comes out EMPTY - every `ref_*` assertion would then pass on an all-null column. Cached
    because the feature build is the expensive part of this module."""
    df = pl.concat([generate_month(2025, m, seed=seed0 + m, deps_per_airport_day=per_day) for m in months])
    out = build_departure_features(df).filter(pl.col(LABEL).is_not_null() & pl.col(LABEL).is_finite())
    # `generate_month` spills a handful of rows into the following calendar month, which would make
    # `_inner_holdout_mask` carve a sliver instead of a month and quietly weaken every test below.
    want = [202500 + m for m in months]
    out = out.filter(pl.Series("_k", year_month_key(out)).is_in(want))
    assert sorted(set(year_month_key(out).unique().to_list())) == want
    assert fit_encodings(out, TrainConfig())["ref"].height > 0, "fixture has no qualifying ref groups"
    return out


def _bump_month(frame: pl.DataFrame, ym: int) -> pl.DataFrame:
    """Add BUMP to every label in one calendar month. The features are untouched, so any encoding move
    is a move through the label and nothing else."""
    key = year_month_key(frame)
    return frame.with_columns(
        pl.when(pl.Series("_k", key) == ym).then(pl.col(LABEL) + BUMP).otherwise(pl.col(LABEL)).alias(LABEL)
    )


def _n_changed(a: pl.DataFrame, b: pl.DataFrame, col: str) -> int:
    x, y = a[col].cast(pl.Float64).to_numpy(), b[col].cast(pl.Float64).to_numpy()
    return int((np.nan_to_num(x, nan=-1e18) != np.nan_to_num(y, nan=-1e18)).sum())


def _setup(per_day=60):
    """The fold-training frame, its model-A lane, and the inner holdout the code would actually carve."""
    feats = _feats(per_day=per_day)
    cfg = TrainConfig(num_rounds=60, early_stopping=10, inner_holdout_months=1)
    # `_fit` receives the ALREADY-ENCODED lane frame, so the holdout block the repair is handed carries
    # the production encoding columns. Reproduce that here rather than a bare frame, or the test would be
    # exercising a shape the pipeline never produces.
    a_tr = attach_encodings(feats, fit_encodings(feats, cfg)).filter(pl.col(PROXY).is_not_null())
    hold = _inner_holdout_mask(a_tr, cfg.inner_holdout_months)
    assert 0 < int(hold.sum()) < a_tr.height, "the synthetic fixture must have a non-trivial inner holdout"
    hold_rows = a_tr.filter(pl.Series("_h", hold))
    months = sorted(set(year_month_key(feats).unique().to_list()))
    hold_ym = sorted(set(year_month_key(hold_rows).unique().to_list()))
    assert hold_ym == months[-1:], (hold_ym, months)
    return feats, cfg, a_tr, hold, hold_rows, months


# ---------------------------------------------------------------- T1: the defect, pinned


def test_es_holdout_reads_its_own_labels_today():
    """BEFORE state. With the production geometry, moving the holdout month's labels moves the holdout
    rows' own encodings. Asserted so the repair cannot later be called unnecessary."""
    feats, cfg, _a_tr, _hold, hold_rows, months = _setup()
    bare = hold_rows.drop([*ENCODING_COLS, *OFFSET_COLS])
    base = attach_encodings(bare, fit_encodings(feats, cfg))
    bumped = attach_encodings(bare, fit_encodings(_bump_month(feats, months[-1]), cfg))
    moved = {c: _n_changed(base, bumped, c) for c in VALUE_COLS}
    assert all(moved[c] > 0 for c in VALUE_COLS), moved


def test_validation_side_is_already_clean():
    """The half the existing suite covers, restated here so the two halves sit side by side: a row that
    is not in the encoder's fitting frame cannot move when its own labels move (guard G1)."""
    feats, cfg, _a_tr, _hold, _hold_rows, months = _setup()
    key = year_month_key(feats)
    tr = feats.filter(pl.Series("_k", key) != months[-1])
    va = feats.filter(pl.Series("_k", key) == months[-1])
    enc = fit_encodings(tr, cfg)
    a = attach_encodings(va, enc)
    b = attach_encodings(va.with_columns(pl.col(LABEL) + BUMP), enc)
    assert all(_n_changed(a, b, c) == 0 for c in ENCODING_COLS)


# ---------------------------------------------------------------- T2: Fix 1's acceptance test


def test_separated_es_encoder_ignores_the_holdout_months_labels():
    """AFTER state, direction 1. The holdout rows' encodings must not move when the holdout month's
    labels move."""
    feats, cfg, _a_tr, _hold, hold_rows, months = _setup()
    a = _es_encoder(feats, cfg, [])(hold_rows)
    b = _es_encoder(_bump_month(feats, months[-1]), cfg, [])(hold_rows)
    moved = {c: _n_changed(a, b, c) for c in ENCODING_COLS}
    assert all(v == 0 for v in moved.values()), moved


def test_separated_es_encoder_still_reads_the_other_months_labels():
    """AFTER state, direction 2 - the control. A column that ignored EVERY month would pass the test
    above and be useless. The other training months must still reach the holdout rows."""
    feats, cfg, _a_tr, _hold, hold_rows, months = _setup()
    a = _es_encoder(feats, cfg, [])(hold_rows)
    b = _es_encoder(_bump_month(feats, months[0]), cfg, [])(hold_rows)
    moved = {c: _n_changed(a, b, c) for c in VALUE_COLS}
    assert all(moved[c] > 0 for c in VALUE_COLS), moved


def test_separated_es_encoder_is_not_a_no_op():
    """The separated map must actually differ from the production map on these rows, and the stats the
    fit records must say so. Several arms returning identical numbers is this repo's known failure
    mode, not a result."""
    feats, cfg, _a_tr, _hold, hold_rows, months = _setup()
    stats: list[dict] = []
    out = _es_encoder(feats, cfg, stats)(hold_rows)
    prod = attach_encodings(hold_rows.drop([*ENCODING_COLS, *OFFSET_COLS]), fit_encodings(feats, cfg))
    assert sum(_n_changed(prod, out, c) for c in VALUE_COLS) > 0
    assert len(stats) == 1
    assert stats[0]["rows"] == hold_rows.height
    assert stats[0]["months_excluded"] == months[-1:]
    assert sum(stats[0]["cols_changed"].values()) > 0, stats


def test_separated_es_encoder_preserves_schema_and_row_order():
    """`_fit` stacks this block with the training frame, so a reordered or reshaped block would be a
    silent row-mismatch rather than an error."""
    _feats_, cfg, _a_tr, _hold, hold_rows, _months = _setup()
    out = _es_encoder(_feats_, cfg, [])(hold_rows)
    assert out.columns == hold_rows.columns
    assert out.height == hold_rows.height
    assert out["MVT_ID_mvt"].to_list() == hold_rows["MVT_ID_mvt"].to_list()


# ---------------------------------------------------------------- Fix 1 through the pipeline


def test_fix1_flag_defaults_to_current_behaviour():
    assert TrainConfig().es_holdout_encodings is False


def test_fix1_early_stopping_fires_and_uses_the_separated_encoder():
    """Guard G2's recorded weakness: an early-stopping test whose `num_rounds` is too small never fires
    and passes vacuously. Assert stopping happened before asserting anything about it."""
    feats = _feats(per_day=30)
    rk = build_departure_features(generate_month(2026, 3, seed=77, deps_per_airport_day=4, blank_dep=True))
    cfg = TrainConfig(num_rounds=400, early_stopping=10, inner_holdout_months=1, es_holdout_encodings=True)
    _p, info = predict_fold(feats, rk, cfg)
    assert info["rounds_matched"] < cfg.num_rounds, "early stopping never fired; the test would be vacuous"
    assert info["es_holdout_encodings"] is True
    rec = info["es_encoder_matched"]
    assert rec and rec[0]["rows"] > 0
    assert sum(rec[0]["cols_changed"].values()) > 0, rec


def test_fix1_only_channel_to_the_prediction_is_the_round_count():
    """No serve-side feature change: with the round count equal, the two arms must agree bit for bit.

    This is plan §4 Step 3's measurement (3) - feature-distribution parity is unchanged by
    construction, so it is asserted rather than measured."""
    feats = _feats(per_day=30)
    rk = build_departure_features(generate_month(2026, 3, seed=78, deps_per_airport_day=4, blank_dep=True))
    off = TrainConfig(num_rounds=400, early_stopping=10, inner_holdout_months=1)
    on = TrainConfig(num_rounds=400, early_stopping=10, inner_holdout_months=1, es_holdout_encodings=True)
    p_off, i_off = predict_fold(feats, rk, off)
    p_on, i_on = predict_fold(feats, rk, on)
    same_rounds = i_off.get("rounds_matched") == i_on.get("rounds_matched") and i_off.get(
        "rounds_unmatched"
    ) == i_on.get("rounds_unmatched")
    if same_rounds:
        assert np.array_equal(p_off["pred"].to_numpy(), p_on["pred"].to_numpy())
    else:  # the round count moved: that is the one permitted channel, and nothing else may have
        pytest.skip(f"rounds moved ({i_off.get('rounds_matched')} -> {i_on.get('rounds_matched')})")


def test_fix1_leaves_the_parity_report_untouched():
    """`feature_parity_report` is on the submission path (`predict.predict_ranking` refuses above a
    0.20 null-rate gap). Fix 1 must not move it: it changes no train-side and no serve-side column."""
    feats = _feats(per_day=30)
    rk = build_departure_features(generate_month(2026, 3, seed=79, deps_per_airport_day=4, blank_dep=True))
    off = feature_parity_report(feats, rk, TrainConfig())
    on = feature_parity_report(feats, rk, TrainConfig(es_holdout_encodings=True))
    assert off.equals(on)


# ---------------------------------------------------------------- T6: the order-statistic bound


def test_reference_min_count_does_not_bound_a_p10_to_one_over_n():
    """`train.py:88` says self-inclusion in a P10 over a group of n is "a leak of 1/n". That is the
    bound for a MEAN and it is false for this statistic.

    Plan §1.5 argues the point assuming linear interpolation. Measured here instead: polars 1.44.2's
    default quantile interpolation is `nearest`, so `ref_p10` on a group of n is the order statistic at
    rank round(0.10 * (n - 1)) VERBATIM - one member's own label, at weight 1.0, with no blending at
    all. The group of exactly `reference_min_count` = 30 below reads its P10 at rank 3.

    Two directions, so the test cannot pass on a constant column: the pivot row's label moves the
    statistic one-for-one, and a non-pivot row's does not move it at all. The leak is not diluted by
    n - it is concentrated on one row per group."""
    n, step = 30, 10.0
    base = pl.DataFrame(
        {
            "ADEP_mvt": ["EDDF"] * n,
            "STAND_mvt": ["A1"] * n,
            "RUNWAY_mvt": ["25C"] * n,
            LABEL: [100.0 + step * i for i in range(n)],
        }
    )

    def p10(frame: pl.DataFrame) -> float:
        return float(reference_table(frame, 30)["ref_p10"][0])

    def nudge(idx: int, delta: float) -> pl.DataFrame:
        return base.with_columns(
            pl.when(pl.int_range(pl.len()) == idx).then(pl.col(LABEL) + delta).otherwise(pl.col(LABEL)).alias(LABEL)
        )

    p0 = p10(base)
    assert p0 == base[LABEL][3], "P10 of a 30-row group is the 4th smallest label verbatim, not a blend"

    delta = step - 1.0  # small enough that the sort order is unchanged
    assert p10(nudge(3, delta)) - p0 == pytest.approx(delta), "the pivot row's label moves ref_p10 one-for-one"
    # the bound train.py:88 claims would be delta / n
    assert abs(p10(nudge(3, delta)) - p0) > 10 * (delta / n)
    # control: any other row moves it not at all, so the leak is concentrated rather than spread over n
    assert p10(nudge(20, delta)) == p0


# ---------------------------------------------------------------- the wiring itself


def _stage1_matrices(feats: pl.DataFrame, rk: pl.DataFrame, cfg: TrainConfig, monkeypatch):
    """The (fit, stopping) design matrices LightGBM is actually handed for early stopping.

    Recorded from inside `lgb.train` rather than inferred, because a mutation that computes the
    re-encoded block correctly and then hands LightGBM the leaked rows anyway survives every test that
    only exercises `_es_encoder` - it did, on the first pass of this audit."""
    seen: list[tuple] = []
    real = lgb.train

    def spy(params, train_set, *a, **kw):
        if kw.get("valid_sets"):
            seen.append((np.array(train_set.data, copy=True), np.array(kw["valid_sets"][0].data, copy=True)))
        return real(params, train_set, *a, **kw)

    monkeypatch.setattr(lgb, "train", spy)
    predict_fold(feats, rk, cfg)
    assert seen, "no early-stopping fit happened; the test would be vacuous"
    return seen[0]


def test_fix1_is_wired_into_the_stage1_fit(monkeypatch):
    """The stopping matrix must change and the fit matrix must not. Between them these two assertions
    are the whole of Option A: the holdout reads a different column, the model is fitted on the same
    one."""
    feats = _feats(per_day=30)
    rk = build_departure_features(generate_month(2026, 3, seed=80, deps_per_airport_day=4, blank_dep=True))
    base = {"num_rounds": 400, "early_stopping": 10, "inner_holdout_months": 1}
    fit_off, hold_off = _stage1_matrices(feats, rk, TrainConfig(**base), monkeypatch)
    fit_on, hold_on = _stage1_matrices(feats, rk, TrainConfig(**base, es_holdout_encodings=True), monkeypatch)
    assert fit_off.shape == fit_on.shape and hold_off.shape == hold_on.shape
    assert np.array_equal(np.nan_to_num(fit_off, nan=-1e18), np.nan_to_num(fit_on, nan=-1e18)), (
        "Fix 1 changed the rows the model is FITTED on; it must not"
    )
    assert not np.array_equal(np.nan_to_num(hold_off, nan=-1e18), np.nan_to_num(hold_on, nan=-1e18)), (
        "the early-stopping rows are unchanged: the re-encoded block was computed and then discarded"
    )
