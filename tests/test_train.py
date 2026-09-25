import numpy as np
import polars as pl
import pytest

from prc.cv import month_block_folds, paired_bootstrap_rmse_diff
from prc.features import build_departure_features
from prc.synth import generate_month
from prc.train import (
    ENCODING_COLS,
    TrainConfig,
    attach_encodings,
    feature_parity_report,
    fit_encodings,
    predict_fold,
    run_cv,
)


def test_folds_use_year_month_keys():
    f = month_block_folds([202501, 202502, 202507, 202511, 202512], "fast")
    assert [x.name for x in f] == ["jan", "jul", "forward"]
    assert f[0].valid_months == (202501,) and 202501 not in f[0].train_months
    # the same calendar month in two years must not be mixed into one fold
    f2 = month_block_folds([202501, 202507, 202601], "fast")
    jan = next(x for x in f2 if x.name == "jan")
    assert jan.valid_months == (202501, 202601)
    assert month_block_folds([202501, 202502, 202503], "full")[0].valid_months == (202501,)
    with pytest.raises(ValueError):
        month_block_folds([202501], "fast")


def test_paired_bootstrap_direction():
    y = np.random.default_rng(0).normal(900, 300, 5000)
    a = y + np.random.default_rng(1).normal(0, 100, 5000)
    b = y + np.random.default_rng(2).normal(0, 50, 5000)
    r = paired_bootstrap_rmse_diff(y, a, b, n_boot=200)
    assert r["diff_rmse"] < 0 and r["ci95"][1] < 0


def test_cv_runs_and_beats_median_baseline(tmp_path):
    df = pl.concat([generate_month(2025, m, seed=m, deps_per_airport_day=12) for m in (1, 6, 7)])
    feats = build_departure_features(df)
    cfg = TrainConfig(scheme="fast", num_rounds=200, early_stopping=20)
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
    # the shipped feature set is parity-clean. The v14 same-stand / same-runway neighbour groups are excluded:
    # their null rate is a function of traffic density, which this fixture halves on purpose (8 vs 4 flights per
    # day); on the real files the density matches (stand.md: 9.6% vs 9.5%) and the pipeline gate checks them.
    dense = ("occ_", "ta_", "seq_", "bk_")
    core = rep.filter(~pl.col("feature").str.starts_with(dense[0]))
    for pfx in dense[1:]:
        core = core.filter(~pl.col("feature").str.starts_with(pfx))
    assert core["gap"].max() < 0.2, core.head(5)
    broken = rk.with_columns(pl.lit(None, dtype=pl.Float64).alias("mvt_minus_sched"))
    rep2 = feature_parity_report(tr, broken, TrainConfig())
    assert rep2.filter(pl.col("feature") == "mvt_minus_sched")["gap"].item() > 0.9


def test_inner_holdout_ignores_a_month_that_is_only_a_spill_over():
    """A file whose last flights cross a month boundary used to become the entire early-stopping holdout."""
    import datetime as dt

    from prc.train import _inner_holdout_mask

    times = []
    for month, n in ((5, 500), (6, 500), (7, 500), (8, 1)):
        times += [dt.datetime(2025, month, 1) + dt.timedelta(hours=i) for i in range(n)]
    df = pl.DataFrame({"MVT_TIME_UTC_mvt": times})
    mask = _inner_holdout_mask(df, 1)
    ym = (df["MVT_TIME_UTC_mvt"].dt.year() * 100 + df["MVT_TIME_UTC_mvt"].dt.month()).to_numpy()
    assert mask.sum() == 500
    assert sorted(set(ym[mask].tolist())) == [202507]  # the real last month, not the one-row August


def test_inner_holdout_is_empty_when_there_are_too_few_months():
    import datetime as dt

    from prc.train import _inner_holdout_mask

    df = pl.DataFrame({"MVT_TIME_UTC_mvt": [dt.datetime(2025, 7, 1) + dt.timedelta(hours=i) for i in range(100)]})
    assert not _inner_holdout_mask(df, 1).any()  # one month cannot be both trained on and held out
    assert not _inner_holdout_mask(df, 0).any()


def test_inner_holdout_takes_several_months_when_asked():
    import datetime as dt

    from prc.train import _inner_holdout_mask

    times = []
    for month in (4, 5, 6, 7):
        times += [dt.datetime(2025, month, 1) + dt.timedelta(hours=i) for i in range(400)]
    df = pl.DataFrame({"MVT_TIME_UTC_mvt": times})
    mask = _inner_holdout_mask(df, 2)
    ym = (df["MVT_TIME_UTC_mvt"].dt.year() * 100 + df["MVT_TIME_UTC_mvt"].dt.month()).to_numpy()
    assert sorted(set(ym[mask].tolist())) == [202506, 202507]
