"""A1, the arrival mirror of LIRF's schedule fill (prc.arrival_fill, TrainConfig.fe_arr_fill).

The flag must be invisible when off: an existing recipe keeps its experiment id, its design matrix and its
predictions to the byte, whether or not the four columns happen to be in the frame. When on, the columns go to
model A only, and model B's predictions do not move."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from prc.arrival_fill import (
    ARR_FILL_COLS,
    arrivals_of,
    attach_arrival_fill,
    build_arrival_fill,
    permute_within_month,
)
from prc.features import build_departure_features
from prc.pipeline import train_config_from_spec
from prc.synth import generate_month
from prc.train import PROXY, TrainConfig, predict_fold, unmatched_columns, visible_numeric_columns


def _t(h, m=0, s=0, day=1, month=7):
    return datetime(2025, month, day, h, m, s, tzinfo=UTC)


def _raw_arrivals() -> pl.DataFrame:
    # stand S1 at LIRF: a filled arrival (block == sched) at 08:00 and an unfilled one at 09:00 (5 min late);
    # stand S2: a filled arrival 7 h before the departure (outside the 6 h link window).
    return pl.DataFrame(
        {
            "PHASE_mvt": ["ARR", "ARR", "ARR", "DEP"],
            "ADES_mvt": ["LIRF", "LIRF", "LIRF", "LFPG"],
            "STAND_mvt": ["S1", "S1", "S2", "S1"],
            "AIRCRAFT_OPERATOR_flt": ["AZA", "RYR", "AZA", "AZA"],
            "BLOCK_TIME_UTC_mvt": [_t(8), _t(9, 5), _t(2), _t(8, 30)],
            "SCHED_TIME_UTC_mvt": [_t(8, 0, 5), _t(9), _t(2), _t(8, 30)],
        }
    )


def _deps() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "MVT_ID_mvt": [1.0, 2.0, 3.0, 4.0],
            "ADEP_mvt": ["LIRF", "LIRF", "LIRF", "LIRF"],
            "STAND_mvt": ["S1", "S1", "S2", None],
            "AIRCRAFT_OPERATOR_flt": ["AZA", "AZA", "AZA", "RYR"],
            # take-off 08:30 links the 08:00 arrival, 10:00 links the 09:05 one, 09:00 at S2 is 7 h after
            "MVT_TIME_UTC_mvt": [_t(8, 30), _t(10), _t(9), _t(11)],
        }
    )


def test_the_builder_links_the_latest_same_stand_arrival_within_six_hours():
    out = build_arrival_fill(_deps(), arrivals_of(_raw_arrivals()))
    assert out["MVT_ID_mvt"].to_list() == [1.0, 2.0, 3.0, 4.0]
    assert out["arr_link_fill"].to_list() == [1, 0, None, None]
    assert out["arr_link_bs_abs"].to_list() == [5.0, 300.0, None, None]
    assert out["arr_link_same_op"].to_list() == [1, 0, None, None]
    # LIRF day: 2 fills of 3 arrivals. AZA: 2 of 2, shrunk with k = 5 -> (2 + 5 * 2/3) / 7; RYR: 0 of 1 -> (0 + 5 * 2/3) / 6
    day = 2 / 3
    assert out["arr_opday_fill"].to_list() == pytest.approx([(2 + 5 * day) / 7] * 3 + [(0 + 5 * day) / 6])


def test_an_operator_with_no_arrivals_that_day_gets_the_airport_day_rate():
    deps = _deps().with_columns(pl.lit("XXX").alias("AIRCRAFT_OPERATOR_flt"))
    out = build_arrival_fill(deps, arrivals_of(_raw_arrivals()))
    assert out["arr_opday_fill"].to_list() == pytest.approx([2 / 3] * 4)
    assert out["arr_link_same_op"].to_list() == [0, 0, None, None]


def test_no_departure_block_time_or_label_is_read():
    raw = _raw_arrivals()
    moved = raw.with_columns(
        pl.when(pl.col("PHASE_mvt") == "DEP")
        .then(None)
        .otherwise(pl.col("BLOCK_TIME_UTC_mvt"))
        .alias("BLOCK_TIME_UTC_mvt")
    )
    assert build_arrival_fill(_deps(), arrivals_of(raw)).equals(build_arrival_fill(_deps(), arrivals_of(moved)))


def test_attach_appends_the_four_columns_and_permute_keeps_each_months_values():
    deps = _deps()
    out = attach_arrival_fill(deps, _raw_arrivals())
    assert out.columns == deps.columns + list(ARR_FILL_COLS)
    assert out.select(deps.columns).equals(deps)
    with pytest.raises(ValueError, match="already has"):
        attach_arrival_fill(out, _raw_arrivals())
    vals = pl.DataFrame({"a": np.arange(10, dtype=float), "b": np.arange(10, 20, dtype=float)})
    when = pl.Series([_t(1, month=1 + (i % 2)) for i in range(10)])
    p = permute_within_month(vals, when)
    jan = np.array([i % 2 == 0 for i in range(10)])
    for c in ("a", "b"):
        assert sorted(p[c].to_numpy()[jan]) == sorted(vals[c].to_numpy()[jan])
        assert sorted(p[c].to_numpy()[~jan]) == sorted(vals[c].to_numpy()[~jan])
    assert permute_within_month(vals, when).equals(p)  # fixed seed


def test_the_flag_is_off_by_default_and_existing_ids_are_unchanged():
    cfg = TrainConfig()
    assert cfg.fe_arr_fill is False and cfg.fe_arr_fill_permute is False
    # ids from experiments/experiments.jsonl, written before fe_arr_fill existed
    for path, want in (
        ("configs/main_flseq.yaml", "8615e6a194"),
        ("configs/main_flseq_osm.yaml", "bb204cd922"),
        ("configs/main_flseq_osm_nolirf.yaml", "926741d779"),
    ):
        assert train_config_from_spec(yaml.safe_load(Path(path).read_text())).hash() == want, path
    for base, a1 in (
        ("configs/main_flseq.yaml", "configs/main_flseq_a1.yaml"),
        ("configs/main_flseq_osm_nolirf.yaml", "configs/main_flseq_osm_nolirf_a1.yaml"),
    ):
        b, c = (yaml.safe_load(Path(p).read_text()) for p in (base, a1))
        assert c.pop("fe_arr_fill") is True
        assert {k: v for k, v in c.items() if k != "name"} == {k: v for k, v in b.items() if k != "name"}
        assert train_config_from_spec(c | {"fe_arr_fill": True}).hash() != train_config_from_spec(b).hash()


def test_permute_without_the_feature_is_refused():
    with pytest.raises(ValueError, match="fe_arr_fill_permute"):
        TrainConfig(fe_arr_fill_permute=True)


@pytest.fixture(scope="module")
def frames():
    raw = pl.concat([generate_month(2025, m, seed=m, deps_per_airport_day=6) for m in (1, 2, 3)], how="diagonal")
    feats = build_departure_features(raw)
    with_a1 = attach_arrival_fill(feats, raw)
    m = feats["MVT_TIME_UTC_mvt"].dt.month()
    return feats, with_a1, m


def test_the_synthetic_frame_links_some_arrivals(frames):
    _, with_a1, _ = frames
    assert with_a1["arr_link_fill"].is_not_null().sum() > 0


def test_with_the_flag_off_the_columns_change_nothing(frames):
    feats, with_a1, m = frames
    cfg = TrainConfig(num_rounds=30, early_stopping=5, n_seeds=1)
    assert visible_numeric_columns(with_a1, cfg) == visible_numeric_columns(feats, cfg)
    p0, i0 = predict_fold(feats.filter(m < 3), feats.filter(m == 3), cfg)
    p1, i1 = predict_fold(with_a1.filter(m < 3), with_a1.filter(m == 3), cfg)
    assert p0.equals(p1)
    assert i0 == i1 and "arr_fill_cols" not in i0


def test_with_the_flag_on_model_a_gets_the_columns_and_model_b_does_not_move(frames):
    feats, with_a1, m = frames
    off = TrainConfig(num_rounds=30, early_stopping=5, n_seeds=1)
    on = TrainConfig(num_rounds=30, early_stopping=5, n_seeds=1, fe_arr_fill=True)
    num = visible_numeric_columns(with_a1, on)
    assert set(ARR_FILL_COLS) <= set(num)
    assert not set(ARR_FILL_COLS) & set(unmatched_columns(num))
    assert unmatched_columns(num) == unmatched_columns(visible_numeric_columns(feats, off))
    p0, _ = predict_fold(feats.filter(m < 3), feats.filter(m == 3), off)
    p1, info = predict_fold(with_a1.filter(m < 3), with_a1.filter(m == 3), on)
    assert info["arr_fill_cols"] == list(ARR_FILL_COLS)
    unm = feats.filter(m == 3).filter(pl.col(PROXY).is_null())["MVT_ID_mvt"]
    assert unm.len() > 0
    sel = pl.col("MVT_ID_mvt").is_in(unm.implode())
    assert p0.filter(sel).equals(p1.filter(sel))
    with pytest.raises(ValueError, match="fe_arr_fill is on"):
        predict_fold(feats.filter(m < 3), feats.filter(m == 3), on)
