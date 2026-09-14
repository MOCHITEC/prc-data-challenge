import numpy as np
import polars as pl
import pytest

from prc.cv import month_block_folds, paired_bootstrap_rmse_diff
from prc.features import build_departure_features
from prc.synth import generate_month
from prc.train import (
    ENCODING_COLS,
    PROXY,
    TrainConfig,
    attach_encodings,
    feature_parity_report,
    fit_encodings,
    predict_fold,
    run_cv,
)


def test_folds_use_year_month_keys():
    f = month_block_folds([202501, 202502, 202507, 202511, 202512])
    assert [x.name for x in f] == ["jan", "jul", "forward"]
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
    assert set(rec["folds"]) == {"jan", "jul"}
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
