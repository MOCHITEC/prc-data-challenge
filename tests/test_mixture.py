import numpy as np
import polars as pl
import pytest

from prc.clocks import CANDIDATE_VALUE, GENUINE, classes, with_candidates, with_clock_source
from prc.features import build_departure_features
from prc.mixture import REAL_OFFBLOCK_SOURCES, SOURCE_RATE_GROUPS, predict_fold_mixture
from prc.synth import generate_month
from prc.train import LABEL, TrainConfig, run_cv

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

    a = attach_source_rates(tg, tables, 50.0)
    blanked = tg.with_columns(pl.lit(None, dtype=tg.schema["BLOCK_TIME_UTC_mvt"]).alias("BLOCK_TIME_UTC_mvt"))
    b = attach_source_rates(blanked, tables, 50.0)
    rate_cols = [c for c in a.columns if c.startswith("sr_")]
    assert rate_cols
    for c in rate_cols:
        assert a[c].to_list() == b[c].to_list()


def test_source_rates_sum_to_one_and_fall_back_to_the_prior():
    from prc.mixture import attach_source_rates, source_rate_tables

    tr = build_departure_features(generate_month(2025, 7, seed=23, deps_per_airport_day=10))
    tr_src = with_clock_source(with_candidates(tr.drop_nulls("TAXITIME_SEC_mvt")), 60, with_rollover=False)
    tables = source_rate_tables(tr_src, smoothing=50.0)
    got = attach_source_rates(tr, tables, 50.0)
    for tag in SOURCE_RATE_GROUPS:
        cols = [f"sr_{tag}_{c}" for c in classes()]
        if cols[0] not in got.columns:
            continue
        total = got.select(pl.sum_horizontal([pl.col(c) for c in cols]).alias("t"))["t"].to_numpy()
        np.testing.assert_allclose(total, 1.0, atol=1e-9)

    unseen = tr.head(3).with_columns(pl.lit("ZZZZ").alias("ADEP_mvt"))
    out = attach_source_rates(unseen, tables, 50.0)
    prior = tables["_prior"]
    for c in classes():
        # the fallback is (0 + s*prior) / (0 + s), which round-trips through a division
        assert out[f"sr_apt_{c}"].to_list() == pytest.approx([prior[f"sr_prior_{c}"][0]] * 3)
        assert out["srn_apt"].to_list() == [0, 0, 0]


def test_leave_one_out_removes_the_row_s_own_class_from_its_own_group():
    """Without this the classifier's feature carries a shaved-down copy of the row's own label."""
    from prc.mixture import attach_source_rates, source_rate_tables

    tr = build_departure_features(generate_month(2025, 7, seed=24, deps_per_airport_day=10))
    tr_src = with_clock_source(with_candidates(tr.drop_nulls("TAXITIME_SEC_mvt")), 60, with_rollover=False)
    tables = source_rate_tables(tr_src, smoothing=50.0)
    plain = attach_source_rates(tr_src, tables, 50.0, leave_one_out=False)
    loo = attach_source_rates(tr_src, tables, 50.0, leave_one_out=True)

    # for its own class the leave-one-out rate is strictly lower; for the others it is strictly higher
    # (the denominator lost a row too)
    for c in classes():
        own = pl.col("clock_src") == c
        if not tr_src.filter(own).height:
            continue
        idx = plain.with_row_index().filter(own)["index"].to_list()
        p_own = plain[f"sr_apt_{c}"].gather(idx).to_numpy()
        l_own = loo[f"sr_apt_{c}"].gather(idx).to_numpy()
        assert (l_own < p_own).all(), c
        other = next(o for o in classes() if o != c)
        assert (loo[f"sr_apt_{other}"].gather(idx).to_numpy() > plain[f"sr_apt_{other}"].gather(idx).to_numpy()).all()

    assert attach_source_rates(tr_src, tables, 50.0, leave_one_out=True).height == tr_src.height
    with pytest.raises(ValueError):  # the ranking frame has no clock_src, so leave-one-out is meaningless there
        attach_source_rates(tr_src.drop("clock_src"), tables, 50.0, leave_one_out=True)


def test_nullable_group_keys_survive_the_pivot_and_fall_back_to_the_prior():
    """STAND_mvt and AIRCRAFT_OPERATOR_flt are both nullable; filling the whole pivot would corrupt the keys."""
    from prc.mixture import attach_source_rates, source_rate_tables

    tr = build_departure_features(generate_month(2025, 7, seed=25, deps_per_airport_day=10))
    tr_src = with_clock_source(with_candidates(tr.drop_nulls("TAXITIME_SEC_mvt")), 60, with_rollover=False)
    tr_src = tr_src.with_columns(
        pl.when(pl.int_range(pl.len()) % 3 == 0).then(None).otherwise(pl.col("STAND_mvt")).alias("STAND_mvt"),
        pl.lit(None, dtype=pl.Utf8).alias("AIRCRAFT_OPERATOR_flt"),
    )
    tables = source_rate_tables(tr_src, smoothing=50.0)
    assert tables["apt_stand"].schema["STAND_mvt"] == pl.Utf8  # not silently cast by a blanket fill_null(0)
    out = attach_source_rates(tr_src, tables, 50.0, leave_one_out=True)
    assert out.height == tr_src.height
    prior = tables["_prior"]
    # every operator is null, so no row joins the operator group and all of them get the prior
    for c in classes():
        assert out[f"sr_apt_op_{c}"].to_list() == pytest.approx([prior[f"sr_prior_{c}"][0]] * out.height)
    null_stand = out.filter(pl.col("STAND_mvt").is_null())
    assert null_stand.height
    for c in classes():
        assert null_stand[f"sr_apt_stand_{c}"].to_list() == pytest.approx(
            [prior[f"sr_prior_{c}"][0]] * null_stand.height
        )


def test_calibration_uses_the_inner_holdout_and_never_the_target_frame():
    """The temperature must be fitted on training months the classifier did not see, not on the target."""
    tr = build_departure_features(
        pl.concat([generate_month(2025, m, seed=m, deps_per_airport_day=10) for m in (5, 6, 7)])
    )
    tg = build_departure_features(generate_month(2026, 7, seed=31, deps_per_airport_day=4, blank_dep=True))
    cfg = TrainConfig(num_rounds=60, early_stopping=10, clf_num_rounds=40, calibrate_temperature=True)
    p, info = predict_fold_mixture(tr, tg, cfg)
    assert p.height == tg.height
    assert "temperature" in info and info["temperature_holdout_rows"] > 0
    assert set(info["temperature_curve"]) >= {"1.0"}
    vals = p["pred"].to_numpy().astype(float)
    assert np.isfinite(vals).all() and (vals >= 0).all()

    # the target frame carries no labels at all here, so nothing about it could have entered the fit
    assert tg[LABEL].null_count() == tg.height


def test_calibration_is_off_by_default_and_then_changes_nothing():
    tr = build_departure_features(pl.concat([generate_month(2025, m, seed=m, deps_per_airport_day=8) for m in (6, 7)]))
    tg = build_departure_features(generate_month(2025, 8, seed=32, deps_per_airport_day=5))
    base = TrainConfig(num_rounds=60, early_stopping=10, clf_num_rounds=40)
    assert base.calibrate_temperature is False
    _p, info = predict_fold_mixture(tr, tg, base)
    assert "temperature" not in info
