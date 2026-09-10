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


def test_null_runway_and_null_airport_are_handled():
    """Real data may have either missing (docs/01 U11); one crashed the context path, the other dropped rows."""
    df = generate_month(2025, 7, seed=21, deps_per_airport_day=6)
    holed = df.with_columns(
        [
            pl.when((pl.col("PHASE_mvt") == "DEP") & (pl.int_range(pl.len()) % 7 == 0))
            .then(None)
            .otherwise(pl.col("RUNWAY_mvt"))
            .alias("RUNWAY_mvt"),
            pl.when((pl.col("PHASE_mvt") == "DEP") & (pl.int_range(pl.len()) % 11 == 0))
            .then(None)
            .otherwise(pl.col("ADEP_mvt"))
            .alias("ADEP_mvt"),
        ]
    )
    feats = build_departure_features(holed)
    assert feats.height == holed.filter(pl.col("PHASE_mvt") == "DEP").height  # no silent row loss
    assert feats["MVT_ID_mvt"].n_unique() == feats.height


def test_missing_anchor_time_gives_null_counts_not_zero():
    df = generate_month(2025, 7, seed=22, deps_per_airport_day=6)
    holed = df.with_columns(
        pl.when((pl.col("PHASE_mvt") == "DEP") & (pl.int_range(pl.len()) % 13 == 0))
        .then(None)
        .otherwise(pl.col("MVT_TIME_UTC_mvt"))
        .alias("MVT_TIME_UTC_mvt")
    )
    feats = build_departure_features(holed)
    no_anchor = feats.filter(pl.col("MVT_TIME_UTC_mvt").is_null())
    assert no_anchor.height > 0
    assert no_anchor["anchor_ok"].sum() == 0
    assert no_anchor["dep_tko_prev_15m"].null_count() == no_anchor.height  # not a misleading zero
    assert feats.filter(pl.col("MVT_TIME_UTC_mvt").is_not_null())["anchor_ok"].min() == 1


def test_candidate_spread_summarises_the_clock_disagreement():
    """cand_spread/cand_n describe how far apart the routing candidates are, which is where the loss lives."""
    f = build_departure_features(generate_month(2025, 7, seed=41, deps_per_airport_day=6))
    cands = ["mvt_minus_aobt3", "mvt_minus_sched", "mvt_minus_eobt1", "mvt_minus_lobt", "mvt_minus_iobt"]
    got = f.select(["cand_min", "cand_max", "cand_spread", "cand_n"] + cands).head(200)
    arr = got.select(cands).to_numpy()
    assert np.allclose(np.nanmin(arr, axis=1), got["cand_min"].to_numpy(), equal_nan=True)
    assert np.allclose(np.nanmax(arr, axis=1), got["cand_max"].to_numpy(), equal_nan=True)
    assert (got["cand_spread"].to_numpy() >= 0).all()
    assert got["cand_n"].to_list() == [int(n) for n in (~np.isnan(arr)).sum(axis=1)]


def test_rows_without_an_nm_record_have_only_the_schedule_candidate():
    f = build_departure_features(generate_month(2025, 7, seed=42, deps_per_airport_day=8))
    no_nm = f.filter(pl.col("nm_matched") == 0)
    if no_nm.height:  # SCHED_TIME is the one clock that is never missing
        assert no_nm["cand_n"].to_list() == [1] * no_nm.height
        assert no_nm["cand_spread"].to_list() == [0.0] * no_nm.height


def test_feature_cache_is_rebuilt_when_the_feature_builder_changes(tmp_path):
    """A new feature column used to leave the cached parquet in place and never reach the model."""
    from prc.pipeline import build_features, feature_fingerprint

    df = generate_month(2025, 7, seed=43, deps_per_airport_day=4)
    rk = generate_month(2026, 7, seed=44, deps_per_airport_day=2, blank_dep=True)
    full, _ = build_features(df, rk, use_weather=False, work=tmp_path)
    assert (tmp_path / "feat_fingerprint.txt").read_text().strip() == feature_fingerprint(False)

    # a matching fingerprint means the cache is trusted verbatim, however wrong it is
    full.head(3).write_parquet(tmp_path / "feat_train.parquet")
    truncated, _ = build_features(df, rk, use_weather=False, work=tmp_path)
    assert truncated.height == 3

    # a stale fingerprint forces the rebuild, which is the case that used to fail silently
    (tmp_path / "feat_fingerprint.txt").write_text("stale")
    rebuilt, _ = build_features(df, rk, use_weather=False, work=tmp_path)
    assert rebuilt.height == full.height
    assert (tmp_path / "feat_fingerprint.txt").read_text().strip() == feature_fingerprint(False)


def test_fingerprint_covers_every_module_that_decides_columns():
    """weather.py decides columns too; leaving it out of the hash reused a 20-column cache for a 6-column run."""
    import hashlib
    from pathlib import Path

    from prc.pipeline import FEATURE_SOURCES, feature_fingerprint

    assert set(FEATURE_SOURCES) == {"features.py", "weather.py"}
    src_dir = Path(__file__).resolve().parents[1] / "src" / "prc"
    for name in FEATURE_SOURCES:
        assert (src_dir / name).exists(), name
    expected = hashlib.sha1(b"".join((src_dir / n).read_bytes() for n in FEATURE_SOURCES) + b"True").hexdigest()
    assert feature_fingerprint(True) == expected[:12]
    assert feature_fingerprint(True) != feature_fingerprint(False)
