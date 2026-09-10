import numpy as np
import polars as pl
import pytest

from prc.clocks import CANDIDATE_VALUE, GENUINE, classes, with_candidates, with_clock_source
from prc.features import build_departure_features
from prc.mixture import REAL_OFFBLOCK_SOURCES, predict_fold_mixture
from prc.synth import generate_month
from prc.train import TrainConfig, run_cv

CFG = TrainConfig(num_rounds=60, early_stopping=10, clf_num_rounds=40)


def test_mixture_handles_an_unlabelled_ranking_frame():
    """The ranking frame has no BLOCK_TIME, so nothing in the mixture may read a clock source from it."""
    tr = build_departure_features(generate_month(2025, 7, seed=3, deps_per_airport_day=10))
    rk = build_departure_features(generate_month(2026, 7, seed=4, deps_per_airport_day=4, blank_dep=True))
    p, info = predict_fold_mixture(tr, rk, CFG)
    assert p.height == rk.height
    assert p["MVT_ID_mvt"].to_list() == rk["MVT_ID_mvt"].to_list()
    vals = p["pred"].to_numpy().astype(float)
    assert np.isfinite(vals).all() and (vals >= 0).all()
    assert info["clf_classes"] == classes()


def test_mixture_prediction_is_identical_whether_the_target_carries_labels():
    """Feature-availability parity's twin: the model must not change because the target happens to be labelled."""
    tr = build_departure_features(pl.concat([generate_month(2025, m, seed=m, deps_per_airport_day=8) for m in (6, 7)]))
    tg = build_departure_features(generate_month(2025, 8, seed=8, deps_per_airport_day=6))
    with_label, _ = predict_fold_mixture(tr, tg, CFG)
    blanked = tg.with_columns(
        pl.lit(None, dtype=tg.schema["TAXITIME_SEC_mvt"]).alias("TAXITIME_SEC_mvt"),
        pl.lit(None, dtype=tg.schema["BLOCK_TIME_UTC_mvt"]).alias("BLOCK_TIME_UTC_mvt"),
    )
    without, _ = predict_fold_mixture(tr, blanked, CFG)
    np.testing.assert_allclose(with_label["pred"].to_numpy(), without["pred"].to_numpy(), rtol=0, atol=1e-9)


def test_taxi_head_trains_only_on_rows_whose_block_time_is_a_real_off_block():
    tr = build_departure_features(pl.concat([generate_month(2025, m, seed=m, deps_per_airport_day=8) for m in (6, 7)]))
    tg = build_departure_features(generate_month(2025, 8, seed=9, deps_per_airport_day=5))
    _p, info = predict_fold_mixture(tr, tg, CFG)
    labelled = tr.filter(pl.col("TAXITIME_SEC_mvt").is_not_null())
    src = with_clock_source(with_candidates(labelled), CFG.match_tol_s, with_rollover=False)
    expected = src.filter(pl.col("clock_src").is_in(list(REAL_OFFBLOCK_SOURCES))).height
    assert info["gen_train_rows"] == expected
    assert expected < labelled.height  # some rows really are copies of a planned time, or the split is a no-op


def test_run_cv_accepts_the_mixture_predictor(tmp_path):
    df = pl.concat([generate_month(2025, m, seed=m, deps_per_airport_day=12) for m in (1, 6, 7)])
    feats = build_departure_features(df)
    rec = run_cv(feats, CFG, out_dir=tmp_path, predictor=predict_fold_mixture)
    assert rec["predictor"] == "predict_fold_mixture"
    y = feats["TAXITIME_SEC_mvt"].to_numpy()
    assert rec["overall"]["rmse_full"] < float(np.sqrt(np.mean((y - np.median(y)) ** 2)))


def test_a_copied_clock_is_predicted_exactly_when_the_classifier_is_certain():
    """End-to-end sanity on the routing arithmetic: a row the classifier sends to `sched` gets MVT - SCHED."""
    from prc.clocks import routed_prediction

    names, cls = list(CANDIDATE_VALUE), classes()
    probs = np.zeros((3, len(cls)))
    probs[0, cls.index("sched")] = 1.0
    probs[1, cls.index("aobt3")] = 1.0
    probs[2, cls.index(GENUINE)] = 1.0
    cand = np.full((3, len(names)), np.nan)
    cand[:, names.index("sched")] = [50_000.0, 900.0, 900.0]
    cand[:, names.index("aobt3")] = [800.0, 1_200.0, 1_100.0]
    got = routed_prediction(probs, cand, np.array([950.0, 950.0, 950.0]), cls)
    assert got[0] == pytest.approx(50_000.0)  # the delayed schedule-reported row, not clipped toward normality
    assert got[1] == pytest.approx(1_200.0)
    assert got[2] == pytest.approx(950.0)


def test_source_rates_are_fitted_on_the_training_side_and_joined_only():
    """The classifier's group rates must not see the target frame's own clock sources."""
    from prc.mixture import attach_source_rates, source_rate_tables

    tr = build_departure_features(generate_month(2025, 7, seed=21, deps_per_airport_day=10))
    tg = build_departure_features(generate_month(2025, 8, seed=22, deps_per_airport_day=8))
    tr_src = with_clock_source(with_candidates(tr.drop_nulls("TAXITIME_SEC_mvt")), 60, with_rollover=False)
    tables = source_rate_tables(tr_src, smoothing=50.0)

    a = attach_source_rates(tg, tables)
    # flipping every clock source on the target side cannot move a single attached rate
    flipped = tg.with_columns(pl.lit(None, dtype=tg.schema["BLOCK_TIME_UTC_mvt"]).alias("BLOCK_TIME_UTC_mvt"))
    b = attach_source_rates(flipped, tables)
    rate_cols = [c for c in a.columns if c.startswith("sr_")]
    assert rate_cols
    for c in rate_cols:
        assert a[c].to_list() == b[c].to_list()


def test_source_rates_sum_to_one_and_fall_back_to_the_prior():
    from prc.mixture import attach_source_rates, source_rate_tables

    tr = build_departure_features(generate_month(2025, 7, seed=23, deps_per_airport_day=10))
    tr_src = with_clock_source(with_candidates(tr.drop_nulls("TAXITIME_SEC_mvt")), 60, with_rollover=False)
    tables = source_rate_tables(tr_src, smoothing=50.0)
    got = attach_source_rates(tr, tables)
    for tag in ("apt", "apt_op", "apt_hour", "apt_stand"):
        cols = [f"sr_{tag}_{c}" for c in classes()]
        if cols[0] not in got.columns:
            continue
        total = got.select(pl.sum_horizontal([pl.col(c) for c in cols]).alias("t"))["t"].to_numpy()
        np.testing.assert_allclose(total, 1.0, atol=1e-9)

    # an airport that never appeared in training gets the global prior, not a null
    unseen = tr.head(3).with_columns(pl.lit("ZZZZ").alias("ADEP_mvt"))
    out = attach_source_rates(unseen, tables)
    prior = tables["_prior"]
    for c in classes():
        assert out[f"sr_apt_{c}"].to_list() == [prior[f"sr_prior_{c}"][0]] * 3
        assert out["srn_apt"].to_list() == [0, 0, 0]
