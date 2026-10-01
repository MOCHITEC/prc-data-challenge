"""GM #5 season weights (docs/gm_research_2026-10-01/plan_season_weight.md)."""

from datetime import datetime

import polars as pl
import pytest

from prc.features import build_departure_features
from prc.synth import generate_month
from prc.train import TrainConfig, predict_fold, season_weights


def test_rows_within_a_month_of_a_predicted_month_are_upweighted_circularly():
    tr = pl.DataFrame({"MVT_TIME_UTC_mvt": [datetime(2025, m, 1) for m in range(1, 13)]})
    tg = pl.DataFrame({"MVT_TIME_UTC_mvt": [datetime(2026, 1, 5), datetime(2026, 7, 5)]})
    assert season_weights(tr, tg, 1.0).to_list() == [2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2]


@pytest.fixture(scope="module")
def feats():
    raw = pl.concat([generate_month(2025, m, seed=m, deps_per_airport_day=6) for m in (1, 2, 3)], how="diagonal")
    return build_departure_features(raw)


def test_off_by_default_and_on_changes_the_fit(feats):
    m = feats["MVT_TIME_UTC_mvt"].dt.month()
    off = TrainConfig(num_rounds=30, early_stopping=5, n_seeds=1)
    assert off.season_weight == 0.0 and off.hash() == TrainConfig(num_rounds=30, early_stopping=5, n_seeds=1).hash()
    on = TrainConfig(num_rounds=30, early_stopping=5, n_seeds=1, season_weight=1.0)
    p0, i0 = predict_fold(feats.filter(m < 3), feats.filter(m == 3), off)
    p1, i1 = predict_fold(feats.filter(m < 3), feats.filter(m == 3), on)
    assert "season_weight_upweighted_share" not in i0
    assert 0 < i1["season_weight_upweighted_share"] < 1  # February is next to March, January is not
    assert p0["MVT_ID_mvt"].equals(p1["MVT_ID_mvt"]) and not p0["pred"].equals(p1["pred"])


def test_catboost_is_refused(feats):
    m = feats["MVT_TIME_UTC_mvt"].dt.month()
    cfg = TrainConfig(
        num_rounds=30,
        early_stopping=5,
        n_seeds=1,
        season_weight=1.0,
        model="catboost",
        unmatched_num_leaves=127,
        unmatched_min_data_in_leaf=50,
    )
    with pytest.raises(ValueError, match="season_weight"):
        predict_fold(feats.filter(m < 3), feats.filter(m == 3), cfg)
