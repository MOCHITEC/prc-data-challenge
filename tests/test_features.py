import numpy as np
import polars as pl

from prc.features import (
    _counts_in_windows,
    _window_mean,
    build_departure_features,
    mask_departures,
    numeric_feature_columns,
)
from prc.synth import generate_month


def test_counts_exclude_self_and_use_half_open_windows():
    anchor = np.array([100.0, 200.0])
    events = np.array([40.0, 99.0, 100.0, 150.0, 200.0, 260.0])
    before, after = _counts_in_windows(anchor, events, 60)
    assert before.tolist() == [2, 1]  # [40,100): 40,99 ; [140,200): 150
    assert after.tolist() == [1, 1]  # (100,160]: 150 ; (200,260]: 260


def test_window_mean_ignores_nan_and_lag():
    anchor = np.array([1000.0])
    times = np.array([900.0, 950.0, 990.0, 1000.0])
    values = np.array([10.0, np.nan, 30.0, 999.0])
    mean, n = _window_mean(anchor, times, values, 120, lag_sec=60)  # [820, 940): only 900 -> 10
    assert n[0] == 1 and mean[0] == 10.0


def test_parity_same_features_for_training_and_ranking_views():
    df = generate_month(2025, 7, seed=3, deps_per_airport_day=8)
    f_train = build_departure_features(df)  # labels present, context masked internally
    f_rank = build_departure_features(mask_departures(df))  # ranking-like view
    cols = [c for c in numeric_feature_columns(f_train) if c not in ("TAXITIME_SEC_mvt",)]
    a = f_train.select(["MVT_ID_mvt"] + cols).sort("MVT_ID_mvt")
    b = f_rank.select(["MVT_ID_mvt"] + cols).sort("MVT_ID_mvt")
    for c in cols:
        x, y = a[c].to_numpy().astype(float), b[c].to_numpy().astype(float)
        assert np.allclose(np.nan_to_num(x, nan=-1), np.nan_to_num(y, nan=-1)), c
    # labels survive for the trainer only in the training view
    assert f_train["TAXITIME_SEC_mvt"].null_count() == 0
    assert f_rank["TAXITIME_SEC_mvt"].null_count() == f_rank.height


def test_no_label_leak_into_context_features():
    df = generate_month(2025, 1, seed=5, deps_per_airport_day=8)
    feats = build_departure_features(df)
    # Perturb all departure labels/block times massively; context features must not change at all.
    df2 = df.with_columns(
        [
            pl.when(pl.col("PHASE_mvt") == "DEP")
            .then(pl.col("TAXITIME_SEC_mvt") * 10)
            .otherwise(pl.col("TAXITIME_SEC_mvt"))
            .alias("TAXITIME_SEC_mvt"),
            pl.when(pl.col("PHASE_mvt") == "DEP")
            .then(pl.col("BLOCK_TIME_UTC_mvt") - pl.duration(hours=5))
            .otherwise(pl.col("BLOCK_TIME_UTC_mvt"))
            .alias("BLOCK_TIME_UTC_mvt"),
        ]
    )
    feats2 = build_departure_features(df2)
    cols = [c for c in numeric_feature_columns(feats) if c != "TAXITIME_SEC_mvt"]
    a, b = feats.sort("MVT_ID_mvt"), feats2.sort("MVT_ID_mvt")
    for c in cols:
        assert np.allclose(
            np.nan_to_num(a[c].to_numpy().astype(float), nan=-1), np.nan_to_num(b[c].to_numpy().astype(float), nan=-1)
        ), c


def test_local_hour_uses_airport_timezone():
    df = generate_month(2025, 7, seed=9, deps_per_airport_day=3)
    feats = build_departure_features(df)
    egll = feats.filter(pl.col("ADEP_mvt") == "EGLL")
    ltfm = feats.filter(pl.col("ADEP_mvt") == "LTFM")
    assert ((egll["hour_local"] - egll["hour_utc"]) % 24 == 1).all()  # BST = UTC+1 in July
    assert ((ltfm["hour_local"] - ltfm["hour_utc"]) % 24 == 3).all()  # Istanbul UTC+3
