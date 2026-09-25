"""The feature cache is keyed by its inputs, not by its path.

A synthetic run and a real run share the default work directory. Reusing one cache for the other trains on
the wrong data while printing plausible CV numbers; it happened once and only a dtype error revealed it."""

import polars as pl
import pytest

from prc.pipeline import build_features, cache_fingerprint
from prc.synth import write_synthetic_dataset


@pytest.fixture(scope="module")
def frames(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    write_synthetic_dataset(raw, deps_per_airport_day=3, seed=7)
    train = pl.concat([pl.read_parquet(p) for p in sorted(raw.glob("training_*.parquet"))], how="diagonal")
    rank = pl.read_parquet(raw / "ranking.parquet")
    return raw, train, rank


def test_a_cache_from_other_inputs_is_rebuilt(frames, tmp_path, capsys):
    raw, train, rank = frames
    work = tmp_path / "work"
    f_tr, _ = build_features(train, rank, False, work, raw)

    # a second raw directory with a different number of rows: same paths, different inputs
    other = tmp_path / "other_raw"
    write_synthetic_dataset(other, deps_per_airport_day=5, seed=9)
    other_train = pl.concat([pl.read_parquet(p) for p in sorted(other.glob("training_*.parquet"))], how="diagonal")
    other_rank = pl.read_parquet(other / "ranking.parquet")

    capsys.readouterr()
    g_tr, _ = build_features(other_train, other_rank, False, work, other)
    assert "rebuilding" in capsys.readouterr().out
    assert g_tr.height != f_tr.height  # the stale frame was not handed back


def test_matching_inputs_hit_the_cache(frames, tmp_path, capsys):
    raw, train, rank = frames
    work = tmp_path / "work"
    build_features(train, rank, False, work, raw)
    capsys.readouterr()
    build_features(train, rank, False, work, raw)
    assert "reusing the cache" in capsys.readouterr().out


def test_editing_the_feature_builder_invalidates_the_cache(frames, tmp_path, monkeypatch):
    raw, train, rank = frames
    before = cache_fingerprint(train, rank, raw, False)
    monkeypatch.setattr("prc.pipeline._features_version", lambda: "0" * 16)
    assert cache_fingerprint(train, rank, raw, False) != before


def test_the_weather_flag_is_part_of_the_key(frames, tmp_path):
    raw, train, rank = frames
    assert cache_fingerprint(train, rank, raw, False) != cache_fingerprint(train, rank, raw, True)


def test_editing_the_weather_module_invalidates_a_weather_on_cache(frames, tmp_path, monkeypatch):
    """The trap the plan calls the highest-probability way to waste a day: the fingerprint hashed features.py
    only, so an edit to weather.py left every cached frame valid and the run reported the old numbers."""
    raw, train, rank = frames
    before = cache_fingerprint(train, rank, raw, True)
    monkeypatch.setattr("prc.pipeline._weather_version", lambda: "0" * 16)
    assert cache_fingerprint(train, rank, raw, True) != before


def test_editing_the_weather_module_does_not_invalidate_a_weather_off_cache(frames, tmp_path, monkeypatch):
    """The control. A `use_weather: false` recipe does not read weather.py, so a weather edit must not cost
    it a four-minute rebuild."""
    raw, train, rank = frames
    before = cache_fingerprint(train, rank, raw, False)
    monkeypatch.setattr("prc.pipeline._weather_version", lambda: "0" * 16)
    assert cache_fingerprint(train, rank, raw, False) == before


def test_the_wx_codes_flag_is_part_of_the_key(frames, tmp_path):
    raw, train, rank = frames
    assert cache_fingerprint(train, rank, raw, True, True) != cache_fingerprint(train, rank, raw, True, False)
    # ... and is ignored when there is no weather join to vary
    assert cache_fingerprint(train, rank, raw, False, True) == cache_fingerprint(train, rank, raw, False, False)
