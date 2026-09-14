"""The A/B split is now a measurable choice, so each mode has to actually produce sane predictions."""

import polars as pl
import pytest

from prc.features import build_departure_features
from prc.synth import write_synthetic_dataset
from prc.train import LABEL, PROXY, TrainConfig, predict_fold


@pytest.fixture(scope="module")
def frames(tmp_path_factory):
    paths = write_synthetic_dataset(tmp_path_factory.mktemp("raw"), months=[(2025, 1)], deps_per_airport_day=8)
    train = pl.concat([pl.read_parquet(p) for n, p in paths.items() if n.startswith("training_")], how="diagonal")
    f = build_departure_features(train)
    return f.head(f.height - 200), f.tail(200)


@pytest.mark.parametrize("mode", ["separate", "shared_b", "single"])
def test_every_split_mode_predicts_every_row(frames, mode):
    tr, tg = frames
    cfg = TrainConfig(name=f"t_{mode}", split_mode=mode, num_rounds=30, early_stopping=10, unmatched_num_rounds=30)
    out, info = predict_fold(tr, tg, cfg)
    assert out.height == tg.height
    assert out["MVT_ID_mvt"].equals(tg["MVT_ID_mvt"])
    assert out["pred"].is_finite().all() and (out["pred"] >= 0).all()
    assert info["split_mode"] == mode


def test_single_mode_still_uses_the_two_clock_offset(frames):
    """With no trees at all, 'single' must fall back to the proxy where it exists - not to zero."""
    tr, tg = frames
    cfg = TrainConfig(split_mode="single", num_rounds=1, early_stopping=1)
    out, _ = predict_fold(tr.head(10), tg, cfg)  # too few rows to fit: g is all zeros
    matched = tg.filter(pl.col(PROXY).is_not_null())
    if matched.height:
        got = out.join(matched.select("MVT_ID_mvt", PROXY), on="MVT_ID_mvt")
        assert (got["pred"] - got[PROXY].clip(lower_bound=0.0)).abs().max() < 1e-6


def test_unknown_split_mode_is_rejected(frames):
    tr, tg = frames
    with pytest.raises(ValueError, match="unknown split_mode"):
        predict_fold(tr, tg, TrainConfig(split_mode="both"))


def test_shared_b_changes_only_the_unmatched_rows(frames):
    """shared_b refits model B on every row and leaves model A alone, so only the no-NM rows may move."""
    tr, tg = frames
    assert tr.filter(pl.col(PROXY).is_null() & pl.col(LABEL).is_not_null()).height, "fixture needs both regimes"
    kw = {"num_rounds": 30, "early_stopping": 10, "unmatched_num_rounds": 30}
    a, _ = predict_fold(tr, tg, TrainConfig(split_mode="separate", **kw))
    b, _ = predict_fold(tr, tg, TrainConfig(split_mode="shared_b", **kw))
    d = a.rename({"pred": "sep"}).join(b.rename({"pred": "shr"}), on="MVT_ID_mvt")
    d = d.join(tg.select("MVT_ID_mvt", PROXY), on="MVT_ID_mvt")
    matched = d.filter(pl.col(PROXY).is_not_null())
    unmatched = d.filter(pl.col(PROXY).is_null())
    assert (matched["sep"] - matched["shr"]).abs().max() < 1e-9, "model A must be untouched"
    if unmatched.height:
        assert (unmatched["sep"] - unmatched["shr"]).abs().max() > 0, "model B must actually have been refitted"
