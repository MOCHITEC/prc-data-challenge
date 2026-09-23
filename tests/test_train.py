import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from prc.cv import month_block_folds, paired_bootstrap_rmse_diff
from prc.features import build_departure_features
from prc.synth import generate_month
from prc.train import (
    CATEGORICALS,
    ENCODING_COLS,
    LOBT_OFFSET,
    PROXY,
    SCHED_OFFSET,
    TrainConfig,
    attach_encodings,
    feature_parity_report,
    fit_encodings,
    predict_fold,
    run_cv,
    sched_reparam_shift,
    unmatched_target,
)
from prc.train import _matrix


def test_folds_use_year_month_keys():
    f = month_block_folds([202501, 202502, 202507, 202511, 202512])
    assert [x.name for x in f] == ["jan", "jul", "jul_fwd", "forward"]
    assert f[0].valid_months == (202501,) and 202501 not in f[0].train_months
    # the same calendar month in two years must not be mixed into one fold
    f2 = month_block_folds([202501, 202507, 202601])
    jan = next(x for x in f2 if x.name == "jan")
    assert jan.valid_months == (202501, 202601)
    with pytest.raises(ValueError):
        month_block_folds([202501])


def test_paired_bootstrap_direction():
    y = np.random.default_rng(0).normal(900, 300, 5000)
    a = y + np.random.default_rng(1).normal(0, 100, 5000)
    b = y + np.random.default_rng(2).normal(0, 50, 5000)
    r = paired_bootstrap_rmse_diff(y, a, b, n_boot=200)
    assert r["diff_rmse"] < 0 and r["ci95"][1] < 0


def test_cv_runs_and_beats_median_baseline(tmp_path):
    df = pl.concat([generate_month(2025, m, seed=m, deps_per_airport_day=12) for m in (1, 6, 7)])
    feats = build_departure_features(df)
    cfg = TrainConfig(num_rounds=200, early_stopping=20)
    rec = run_cv(feats, cfg, out_dir=tmp_path)
    assert set(rec["folds"]) == {"jan", "jul", "jul_fwd"}
    y = feats["TAXITIME_SEC_mvt"].to_numpy()
    base = float(np.sqrt(np.mean((y - np.median(y)) ** 2)))
    assert rec["overall"]["rmse_full"] < base
    assert (tmp_path / "experiments.jsonl").exists()


def test_predict_fold_handles_unlabelled_ranking_frame():
    tr = build_departure_features(generate_month(2025, 7, seed=3, deps_per_airport_day=10))
    rk = build_departure_features(generate_month(2026, 7, seed=4, deps_per_airport_day=4, blank_dep=True))
    p, _info = predict_fold(tr, rk, TrainConfig(num_rounds=50, early_stopping=10))
    assert p.height == rk.height
    assert p["MVT_ID_mvt"].to_list() == rk["MVT_ID_mvt"].to_list()  # keyed by id, in the caller's order
    vals = p["pred"].to_numpy().astype(float)
    assert np.isfinite(vals).all() and (vals >= 0).all()


def test_encodings_do_not_depend_on_target_side_labels():
    """A validation row's own label must never reach its own features (the apt_med leak found in review)."""
    df = pl.concat([generate_month(2025, m, seed=m, deps_per_airport_day=8) for m in (1, 7)])
    feats = build_departure_features(df)
    tr = feats.filter(pl.col("MVT_TIME_UTC_mvt").dt.month() == 7)
    va = feats.filter(pl.col("MVT_TIME_UTC_mvt").dt.month() == 1)
    cfg = TrainConfig(num_rounds=30, early_stopping=10)
    enc = fit_encodings(tr, cfg)
    a = attach_encodings(va, enc)
    # blanking the target frame's labels must not change a single encoding value
    b = attach_encodings(va.with_columns(pl.lit(None, dtype=pl.Float64).alias("TAXITIME_SEC_mvt")), enc)
    for c in ENCODING_COLS:
        assert a[c].to_list() == b[c].to_list(), c
    # and the predictions must be identical too
    p1, _ = predict_fold(tr, va, cfg)
    p2, _ = predict_fold(tr, va.with_columns(pl.lit(None, dtype=pl.Float64).alias("TAXITIME_SEC_mvt")), cfg)
    assert np.allclose(p1["pred"].to_numpy(), p2["pred"].to_numpy())


def test_null_labels_are_dropped_not_treated_as_zero():
    df = generate_month(2025, 7, seed=11, deps_per_airport_day=10)
    feats = build_departure_features(df)
    holed = feats.with_columns(
        pl.when(pl.int_range(pl.len()) % 20 == 0)
        .then(None)
        .otherwise(pl.col("TAXITIME_SEC_mvt"))
        .alias("TAXITIME_SEC_mvt")
    )
    rk = build_departure_features(generate_month(2026, 7, seed=12, deps_per_airport_day=3, blank_dep=True))
    _p, info = predict_fold(holed, rk, TrainConfig(num_rounds=30, early_stopping=10))
    assert info["dropped_unlabelled"] == holed["TAXITIME_SEC_mvt"].null_count() > 0
    assert info["train_rows"] == holed.height - info["dropped_unlabelled"]


def test_parity_report_flags_a_column_that_vanishes_at_prediction_time():
    tr = build_departure_features(generate_month(2025, 7, seed=13, deps_per_airport_day=8))
    rk = build_departure_features(generate_month(2026, 7, seed=14, deps_per_airport_day=4, blank_dep=True))
    rep = feature_parity_report(tr, rk, TrainConfig())
    assert set(rep.columns) == {"feature", "null_rate_train", "null_rate_target", "gap"}
    assert rep["gap"].max() < 0.2, rep.head(5)  # the shipped feature set is parity-clean
    broken = rk.with_columns(pl.lit(None, dtype=pl.Float64).alias("mvt_minus_sched"))
    rep2 = feature_parity_report(tr, broken, TrainConfig())
    assert rep2.filter(pl.col("feature") == "mvt_minus_sched")["gap"].item() > 0.9


def test_early_stopping_never_reads_the_evaluation_fold():
    """The stopping round must not move when the target frame's labels are blanked.

    The older version of this check ran 30 rounds with a patience of 10, so early stopping never fired and a
    model that stopped on the evaluation fold passed it unchanged. Enough rounds for the callback to actually
    bite is the whole point."""
    df = pl.concat([generate_month(2025, m, seed=m, deps_per_airport_day=10) for m in (1, 5, 7)])
    feats = build_departure_features(df)
    tr = feats.filter(pl.col("MVT_TIME_UTC_mvt").dt.month() != 1)
    va = feats.filter(pl.col("MVT_TIME_UTC_mvt").dt.month() == 1)
    cfg = TrainConfig(num_rounds=400, early_stopping=10)
    _, info_labelled = predict_fold(tr, va, cfg)
    blank = va.with_columns(pl.lit(None, dtype=pl.Float64).alias("TAXITIME_SEC_mvt"))
    _, info_blank = predict_fold(tr, blank, cfg)
    assert info_labelled["rounds_matched"] < cfg.num_rounds, "early stopping never fired; the test proves nothing"
    assert info_labelled["rounds_matched"] == info_blank["rounds_matched"]


def test_predictions_follow_their_id_not_their_position():
    """Model A and model B fill different rows, so the two blocks must be re-keyed, never re-stacked."""
    tr = build_departure_features(generate_month(2025, 7, seed=11, deps_per_airport_day=10))
    tg = build_departure_features(generate_month(2025, 7, seed=12, deps_per_airport_day=6))
    # force both regimes and interleave them, so a positional assignment cannot accidentally be right
    half = tg.height // 2
    tg = tg.with_columns(
        pl.when(pl.int_range(pl.len()) % 2 == 0).then(None).otherwise(pl.col(PROXY)).alias(PROXY)
    ).head(2 * half)
    assert tg[PROXY].is_null().any() and tg[PROXY].is_not_null().any(), "fixture needs both regimes interleaved"
    cfg = TrainConfig(num_rounds=40, early_stopping=10, unmatched_num_rounds=40)
    straight, _ = predict_fold(tr, tg, cfg)
    shuffled_tg = tg.sample(fraction=1.0, shuffle=True, seed=5)
    shuffled, _ = predict_fold(tr, shuffled_tg, cfg)
    both = straight.rename({"pred": "a"}).join(shuffled.rename({"pred": "b"}), on="MVT_ID_mvt")
    assert both.height == tg.height
    assert np.allclose(both["a"].to_numpy(), both["b"].to_numpy()), "a row's prediction changed with its position"


def test_a_screening_run_can_use_one_fold():
    """Three folds is the honest number; one is what a screening run needs, so the选択 must be explicit."""
    df = pl.concat([generate_month(2025, m, seed=m, deps_per_airport_day=6) for m in (1, 7, 11)])
    feats = build_departure_features(df)
    cfg = TrainConfig(num_rounds=20, early_stopping=10, unmatched_num_rounds=20, folds=("jul",))
    rec = run_cv(feats, cfg, out_dir=Path(tempfile.mkdtemp()), log_name="t.jsonl")
    assert list(rec["folds"]) == ["jul"]
    with pytest.raises(ValueError, match="matched none"):
        run_cv(feats, replace(cfg, folds=("nope",)), out_dir=Path(tempfile.mkdtemp()), log_name="t.jsonl")


def test_a_forward_twin_exists_for_july_and_cannot_for_january():
    """The leak in the jan/jul folds is measurable only against the same month trained forward-only."""
    ms = [202501, 202503, 202507, 202511, 202512]
    by_name = {f.name: f for f in month_block_folds(ms)}
    assert "jul_fwd" in by_name
    jf = by_name["jul_fwd"]
    assert jf.valid_months == (202507,)
    assert all(m < 202507 for m in jf.train_months), "a forward fold may not train on later months"
    assert "jan_fwd" not in by_name, "January is the first month; there is nothing before it"


def test_candidate_columns_never_reach_the_model_unless_named():
    """v6 shipped with all nine screening columns because main.yaml did not opt out, and scored 350.62 against
    v4's 305.03. Hiding them is now the default and opting in is explicit."""
    from prc.features import CANDIDATE_COLS
    from prc.train import CATEGORICALS  # noqa: F401  - imported for the symbol check below

    tr = build_departure_features(generate_month(2025, 7, seed=21, deps_per_airport_day=6))
    tg = build_departure_features(generate_month(2025, 7, seed=22, deps_per_airport_day=4, blank_dep=True))
    assert all(c in tr.columns for c in CANDIDATE_COLS), "the columns must exist, just not be visible"

    base = TrainConfig(num_rounds=5, early_stopping=3, unmatched_num_rounds=5)
    assert base.include_candidates == (), "the default must hide every candidate"
    p_hidden, _ = predict_fold(tr, tg, base)
    p_shown, _ = predict_fold(tr, tg, replace(base, include_candidates=("noise_control",)))
    assert p_hidden.height == p_shown.height == tg.height


def _lirf_filled_month(year: int, month: int, seed: int, deps: int = 10, blank_dep: bool = False) -> pl.DataFrame:
    """A month in which every LIRF departure is unmatched and has BLOCK_TIME filled with SCHED_TIME.

    That is the regime the reparameterisation exists for (docs/investigation_2026-09-16/results/
    sched-filled-block.md): with BLOCK == SCHED the label equals `mvt_minus_sched` exactly, so a model fitted
    on `LABEL - mvt_minus_sched` learns ~0 there and everything it returns has to come from the shift. The
    generator already produces a few such rows; this turns the knob all the way up so a small fixture has them.
    """
    df = generate_month(year, month, seed=seed, deps_per_airport_day=deps, blank_dep=blank_dep)
    dep = (pl.col("ADEP_mvt") == "LIRF") & (pl.col("PHASE_mvt") == "DEP")
    mvt = pl.col("SCHED_TIME_UTC_mvt") + pl.duration(hours=3)
    df = df.with_columns(
        pl.when(dep).then(None).otherwise(pl.col("FLIGHT_ID_mvt")).alias("FLIGHT_ID_mvt"),
        pl.when(dep).then(None).otherwise(pl.col("AOBT_3_flt")).alias("AOBT_3_flt"),
        pl.when(dep).then(mvt).otherwise(pl.col("MVT_TIME_UTC_mvt")).alias("MVT_TIME_UTC_mvt"),
    )
    if not blank_dep:  # the ranking frame has no BLOCK_TIME and no label to fill in
        df = df.with_columns(
            pl.when(dep)
            .then(pl.col("SCHED_TIME_UTC_mvt"))
            .otherwise(pl.col("BLOCK_TIME_UTC_mvt"))
            .alias("BLOCK_TIME_UTC_mvt"),
            pl.when(dep).then(pl.lit(3 * 3600.0)).otherwise(pl.col("TAXITIME_SEC_mvt")).alias("TAXITIME_SEC_mvt"),
        )
    return df


def test_sched_reparam_shifts_exactly_the_configured_airport_rows():
    """The shift is `mvt_minus_sched` at the named airports and exactly zero everywhere else."""
    feats = build_departure_features(generate_month(2025, 7, seed=31, deps_per_airport_day=6))
    on = TrainConfig(unmatched_sched_reparam=True)
    shift = sched_reparam_shift(feats, on)
    ms = feats[SCHED_OFFSET].to_numpy().astype(float)
    lirf = (feats["ADEP_mvt"] == "LIRF").to_numpy()
    assert lirf.any() and (~lirf).any(), "fixture needs both airports"
    assert np.array_equal(shift[lirf], ms[lirf])
    assert (shift[~lirf] == 0.0).all()
    # and the target is the label minus that shift, on those rows only
    y = feats["TAXITIME_SEC_mvt"].to_numpy().astype(float)
    assert np.array_equal(unmatched_target(feats, on), y - np.where(lirf, ms, 0.0))
    # the airport set is config, not a constant: naming a second airport moves those rows too
    two = sched_reparam_shift(feats, replace(on, unmatched_sched_reparam_airports=("LIRF", "EHAM")))
    eham = (feats["ADEP_mvt"] == "EHAM").to_numpy()
    assert np.array_equal(two[eham], ms[eham]) and np.array_equal(two[lirf], ms[lirf])
    assert (two[~(lirf | eham)] == 0.0).all()
    # ...and naming none is a no-op even with the flag on
    assert (sched_reparam_shift(feats, replace(on, unmatched_sched_reparam_airports=())) == 0.0).all()


def test_sched_reparam_never_shifts_a_row_with_a_null_schedule_offset():
    """`mvt_minus_sched` has null rate 0 on the ranking set, but a null here would put a NaN in the labels."""
    feats = build_departure_features(generate_month(2025, 7, seed=32, deps_per_airport_day=6))
    lirf = (feats["ADEP_mvt"] == "LIRF").to_numpy()
    holed = feats.with_columns(
        pl.when((pl.col("ADEP_mvt") == "LIRF") & (pl.int_range(pl.len()) % 3 == 0))
        .then(None)
        .otherwise(pl.col(SCHED_OFFSET))
        .alias(SCHED_OFFSET)
    )
    null_ms = holed[SCHED_OFFSET].is_null().to_numpy()
    assert (null_ms & lirf).any(), "fixture needs LIRF rows without the offset"
    on = TrainConfig(unmatched_sched_reparam=True)
    shift = sched_reparam_shift(holed, on)
    assert (shift[null_ms] == 0.0).all()
    target = unmatched_target(holed, on)
    y = holed["TAXITIME_SEC_mvt"].to_numpy().astype(float)
    assert np.array_equal(target[null_ms], y[null_ms])
    assert np.isfinite(target).all()


def test_sched_reparam_is_off_by_default_and_leaves_the_model_unchanged():
    """With the flag off the target is the label itself and neither new field can reach the model.

    What this can show inside the repository is that the added arithmetic is `y - 0.0` / `pred + 0.0` and that
    the two new fields change nothing while the flag is off. That the fitted model is then the one the old code
    produced was checked on the real data instead: rerunning `configs/inv_baseline_forward.yaml` under this
    code reproduced `oof_inv_baseline_forward-67e76a5dfc` with every logged metric equal and 327,983 of 328,009
    predictions bit-identical, the other 26 within 4.5e-13
    (docs/investigation_2026-09-16/results/reparam-implementation.md)."""
    assert TrainConfig().unmatched_sched_reparam is False, "main.yaml must keep today's model"
    tr = build_departure_features(_lirf_filled_month(2025, 7, seed=33, deps=10))
    tg = build_departure_features(_lirf_filled_month(2025, 7, seed=34, deps=6))
    off = TrainConfig(num_rounds=40, early_stopping=10, unmatched_num_rounds=40)
    assert (sched_reparam_shift(tr, off) == 0.0).all()
    y = tr["TAXITIME_SEC_mvt"].to_numpy().astype(float)
    assert np.array_equal(unmatched_target(tr, off), y)  # `y - 0.0`, not merely close to y
    p_off, info_off = predict_fold(tr, tg, off)
    # the flag is what switches the code path: with it off, the airport set cannot matter
    p_other, _ = predict_fold(tr, tg, replace(off, unmatched_sched_reparam_airports=("LIRF", "EHAM")))
    assert (p_off["pred"].to_numpy() == p_other["pred"].to_numpy()).all()
    # ...and with it on but no airport named, the shift is zero and the model is the same one again
    p_empty, _ = predict_fold(tr, tg, replace(off, unmatched_sched_reparam=True, unmatched_sched_reparam_airports=()))
    assert (p_off["pred"].to_numpy() == p_empty["pred"].to_numpy()).all()
    assert "sched_reparam_train_rows" not in info_off


def test_sched_reparam_predictions_come_back_on_the_label_scale():
    """Fit on `LABEL - shift`, predict `LABEL`: the same call is the CV fold and the full-data refit.

    The target frame here is unlabelled, i.e. the ranking shape - predict_ranking reaches model B through this
    very call - so nothing in the add-back can come from the target side's labels."""
    tr = build_departure_features(_lirf_filled_month(2025, 7, seed=35, deps=12))
    tg = build_departure_features(_lirf_filled_month(2026, 7, seed=36, deps=6, blank_dep=True))
    cfg = TrainConfig(num_rounds=60, early_stopping=15, unmatched_num_rounds=60, unmatched_sched_reparam=True)
    pred, info = predict_fold(tr, tg, cfg)
    lirf = (tg["ADEP_mvt"] == "LIRF").to_numpy()
    unmatched = tg[PROXY].is_null().to_numpy()
    sel = lirf & unmatched
    assert sel.sum() > 20, "fixture needs LIRF unmatched rows"
    assert info["sched_reparam_train_rows"] == int(((tr["ADEP_mvt"] == "LIRF") & tr[PROXY].is_null()).sum()), (
        "the rows fitted on the shifted target are the LIRF unmatched ones"
    )
    assert info["sched_reparam_target_rows"] == int(sel.sum())
    ms = tg[SCHED_OFFSET].to_numpy().astype(float)
    joined = tg.select("MVT_ID_mvt").join(pred, on="MVT_ID_mvt", how="left", maintain_order="left")
    v = joined["pred"].to_numpy().astype(float)
    assert np.isfinite(v).all() and (v >= 0).all()
    # every LIRF row here is sched-filled, so the model space target is ~0 and the shift carries the prediction:
    # without the add-back these would sit near zero instead of near three hours.
    assert (v[sel] > 0.8 * ms[sel]).all(), (v[sel].min(), ms[sel].min())
    assert np.median(np.abs(v[sel] - ms[sel])) < 0.2 * float(np.median(ms[sel]))


def test_overall_names_the_folds_it_pooled_and_the_ones_it_shadowed(tmp_path):
    """`jul` and `jul_fwd` validate the same months, and `overall` keeps whichever the fold order reached first.

    That was silent: overall.n read 672,428 against a fold sum of 863,141 in production, and overall.rmse_full
    was the pooled OOF of {jan, jul, forward} with no record saying so
    (docs/investigation_2026-09-16/results/fold-shape-vs-ranking.md §3.3). The pooling is unchanged - the
    record now names it."""
    df = pl.concat([generate_month(2025, m, seed=m, deps_per_airport_day=4) for m in (1, 6, 7)])
    feats = build_departure_features(df)
    rec = run_cv(feats, TrainConfig(num_rounds=20, early_stopping=10), out_dir=tmp_path)
    assert list(rec["folds"]) == ["jan", "jul", "jul_fwd"]
    assert rec["overall"]["folds_pooled"] == ["jan", "jul"]
    assert rec["overall"]["folds_shadowed"] == ["jul_fwd"], "jul_fwd predicts rows jul already predicted"
    assert rec["folds"]["jul_fwd"]["n_into_overall"] == 0
    assert rec["folds"]["jul"]["n_into_overall"] == rec["folds"]["jul"]["n"]
    # overall.n is the folds that reached it, not the sum over the fold table
    assert rec["overall"]["n"] == sum(rec["folds"][f]["n_into_overall"] for f in rec["overall"]["folds_pooled"])
    assert rec["overall"]["n"] < sum(rec["folds"][f]["n"] for f in rec["folds"])


# --- the LOBT prediction bound (docs/competitor_research_2026-09-21/plans/P5-*) -------------------------


def _lobt_frames(seed: int = 5):
    df = pl.concat([generate_month(2025, m, seed=seed + m, deps_per_airport_day=8) for m in (1, 7)])
    feats = build_departure_features(df)
    tr = feats.filter(pl.col("MVT_TIME_UTC_mvt").dt.month() == 7)
    va = feats.filter(pl.col("MVT_TIME_UTC_mvt").dt.month() == 1)
    return tr, va


def test_lobt_bound_is_off_by_default_and_changes_nothing_when_on_but_never_binding():
    """Default builds must be bit-identical, and a slack wide enough to admit every prediction is a no-op."""
    tr, va = _lobt_frames()
    base = TrainConfig(num_rounds=30, early_stopping=10)
    off, info_off = predict_fold(tr, va, base)
    assert "lobt_bound_rows" not in info_off
    wide, info_wide = predict_fold(tr, va, replace(base, lobt_bound=True, lobt_bound_slack=1e9))
    assert info_wide["lobt_bound_rows"] == 0
    assert off["pred"].to_list() == wide["pred"].to_list()
    assert off["MVT_ID_mvt"].to_list() == wide["MVT_ID_mvt"].to_list()


def test_lobt_bound_moves_a_row_outside_its_window_to_the_boundary_and_leaves_the_rest_alone():
    """The mutation test for the bound: with a slack this tight it must bind, and only onto the edge."""
    tr, va = _lobt_frames()
    base = TrainConfig(num_rounds=30, early_stopping=10)
    off, _ = predict_fold(tr, va, base)
    # Wide enough that no row's window is inverted (that is a separate test), tight enough to bind here.
    slack = float(-min(va[LOBT_OFFSET].min(), 0.0)) + 300.0
    on, info = predict_fold(tr, va, replace(base, lobt_bound=True, lobt_bound_slack=slack))
    assert info["lobt_bound_rows"] > 0, f"a {slack:.0f} s window must bind here or the test proves nothing"
    m = va.select(["MVT_ID_mvt", LOBT_OFFSET]).join(
        off.rename({"pred": "before"}).join(on.rename({"pred": "after"}), on="MVT_ID_mvt"),
        on="MVT_ID_mvt",
    )
    for row in m.iter_rows(named=True):
        anchor, before, after = row[LOBT_OFFSET], row["before"], row["after"]
        if anchor is None:  # no NM record, no window
            assert after == before
            continue
        lo, hi = max(anchor - slack, 0.0), anchor + slack
        assert lo - 1e-9 <= after <= hi + 1e-9, row
        # a projection: unchanged inside the window, otherwise exactly the nearer edge
        assert after == pytest.approx(min(max(before, lo), hi), abs=1e-9), row
        assert abs(after - before) <= abs(hi - lo) + abs(before) + 1e-9


def test_lobt_bound_never_moves_a_prediction_away_from_a_truth_inside_the_window():
    """The property the whole change rests on: clipping onto an interval containing y cannot increase |p - y|."""
    rng = np.random.default_rng(0)
    slack, anchor = 3606.0, rng.uniform(-3000, 12000, 20000)
    y = anchor + rng.uniform(-slack, slack, anchor.size)  # a truth the identity admits
    p = rng.uniform(-5000, 20000, anchor.size)
    lo, hi = np.maximum(anchor - slack, 0.0), anchor + slack
    clipped = np.clip(np.maximum(p, 0.0), lo, hi)
    assert (np.abs(clipped - np.maximum(y, 0.0)) <= np.abs(np.maximum(p, 0.0) - np.maximum(y, 0.0)) + 1e-9).all()


def test_lobt_bound_refuses_an_inverted_window_rather_than_clipping_silently():
    """A row whose MVT precedes LOBT by more than the slack has no window the zero floor also admits."""
    tr, va = _lobt_frames()
    broken = va.with_columns(
        pl.when(pl.int_range(pl.len()) == 0).then(-9e4).otherwise(pl.col(LOBT_OFFSET)).alias(LOBT_OFFSET)
    )
    cfg = TrainConfig(num_rounds=30, early_stopping=10, lobt_bound=True)
    with pytest.raises(AssertionError, match="inverted"):
        predict_fold(tr, broken, cfg)


def test_lobt_bound_refuses_a_non_positive_slack():
    tr, va = _lobt_frames()
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="positive finite"):
            predict_fold(tr, va, TrainConfig(num_rounds=30, early_stopping=10, lobt_bound=True, lobt_bound_slack=bad))


# --- the fold sidecar and the two-lane report (P1 stage 0) ---------------------------------------------


def test_run_cv_writes_a_fold_sidecar_that_keeps_the_rows_overall_coalesced_away(tmp_path):
    """`overall` keeps one prediction per row, so a shadowed fold vanishes from it. The sidecar must not."""
    df = pl.concat([generate_month(2025, m, seed=m, deps_per_airport_day=6) for m in range(1, 13)])
    feats = build_departure_features(df)
    cfg = TrainConfig(name="sidecar", num_rounds=30, early_stopping=10)
    rec = run_cv(feats, cfg, out_dir=tmp_path, log_name="t.jsonl")
    side = pl.read_parquet(tmp_path / f"oof_{rec['exp_id']}_folds.parquet")
    assert side.columns == ["MVT_ID_mvt", "fold"]
    # one entry per (row, fold) actually predicted - so the total matches the fold table, not `overall`
    assert side.height == sum(rec["folds"][name]["n"] for name in rec["folds"])
    assert set(side["fold"].unique()) == set(rec["folds"])
    # and every shadowed fold is still represented, which is the whole point of the sidecar
    for name in rec["overall"]["folds_shadowed"]:
        assert side.filter(pl.col("fold") == name).height > 0, name
    # the wide OOF is untouched: still exactly the two columns every archived record and `compare` expect
    assert pl.read_parquet(tmp_path / f"oof_{rec['exp_id']}.parquet").columns == ["MVT_ID_mvt", "oof"]


def test_every_fold_and_overall_report_both_lanes_separately(tmp_path):
    """A pooled RMSE moves by tenths when model B moves by hundreds; the record must name which lane moved."""
    df = pl.concat([generate_month(2025, m, seed=20 + m, deps_per_airport_day=6) for m in range(1, 13)])
    feats = build_departure_features(df)
    cfg = TrainConfig(name="lanes", num_rounds=30, early_stopping=10)
    rec = run_cv(feats, cfg, out_dir=tmp_path, log_name="t.jsonl")
    for block in [rec["overall"], *rec["folds"].values()]:
        assert "matched" in block and "unmatched" in block
        assert block["matched"]["n"] + block["unmatched"]["n"] == block["n"], block["n"]
        for lane in ("matched", "unmatched"):
            if block[lane]["n"]:
                assert block[lane]["rmse_full"] > 0
                assert set(block[lane]["by_airport"]) <= set(feats["ADEP_mvt"].unique().to_list())


def test_the_design_matrix_is_a_function_of_the_data_not_of_its_chunking():
    """G11: `cast(pl.Categorical)` gave a different code permutation on every call over a chunked frame.

    The data survives a relabelling but the model does not: LightGBM splits a categorical on its code values,
    so permuting them is a different model. Replaying one production fit with only the codes relabelled moved
    early stopping by 164 rounds, which is the whole distance between two refits that were meant to be the
    same build. The frame must be multi-chunk for this test to mean anything - a single-chunk frame encodes
    stably even under the old code."""
    parts = [generate_month(2025, m, seed=m, deps_per_airport_day=8) for m in (1, 3, 5, 7)]
    feats = pl.concat([build_departure_features(p) for p in parts])
    cats = [c for c in CATEGORICALS if c in feats.columns]
    assert feats[cats[0]].n_chunks() > 1, "the frame must be chunked or this test cannot fail"
    nums = [c for c in ("mvt_minus_sched", "hour_local", "month") if c in feats.columns]
    draws = {_matrix(feats, nums, cats)[0].tobytes() for _ in range(4)}
    assert len(draws) == 1, f"the design matrix took {len(draws)} distinct values over 4 identical calls"
    # and the encoding must still separate every level it is given
    X, cols = _matrix(feats, nums, cats)
    for c in cats:
        n_levels = feats[c].cast(pl.Utf8).fill_null("__NA__").n_unique()
        assert len(np.unique(X[:, cols.index(c)])) == n_levels, c
