"""Fold-time candidate families from docs/article_fe_2026-09-24/plan.md.

Each test pins one property the screen's verdict depends on: a family that read a label it must not, or that
reached model B with a column that is null on every one of its rows, would make the measurement mean something
other than what the plan says it means."""

from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from prc.features import build_departure_features
from prc.synth import generate_month
from prc.train import (
    FREQ_KEYS,
    LABEL,
    PROXY,
    SCHED_OFFSET,
    TrainConfig,
    aggdiff_features,
    attach_encodings,
    cyclic_features,
    fit_encodings,
    frequency_features,
    meta_prior_features,
    noise_features,
    predict_fold,
    unmatched_columns,
    visible_numeric_columns,
)


@pytest.fixture(scope="module")
def feats() -> pl.DataFrame:
    df = pl.concat([generate_month(2025, m, seed=m, deps_per_airport_day=6) for m in (1, 2, 3)])
    return build_departure_features(df)


def _split(feats: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    m = feats["MVT_TIME_UTC_mvt"].dt.month()
    tr = feats.filter(m < 3)
    tg = feats.filter(m == 3)
    enc = fit_encodings(tr, TrainConfig(reference_min_count=1))
    return attach_encodings(tr, enc), attach_encodings(tg, enc)


def test_every_flag_is_off_by_default_and_the_default_fold_adds_no_column(feats):
    cfg = TrainConfig(num_rounds=30, early_stopping=5)
    assert not (cfg.fe_freq or cfg.fe_aggdiff or cfg.fe_cyclic or cfg.fe_meta_prior or cfg.fe_noise_cols)
    m = feats["MVT_TIME_UTC_mvt"].dt.month()
    _, info = predict_fold(feats.filter(m < 3), feats.filter(m == 3), cfg)
    assert "fold_feature_cols" not in info


def test_frequency_encodings_never_read_a_label(feats):
    tr, tg = _split(feats)
    a_tr, a_tg = frequency_features(tr, tg)
    shuffled = tr.with_columns(pl.col(LABEL).shuffle(seed=1))
    b_tr, b_tg = frequency_features(shuffled, tg.with_columns(pl.lit(None, dtype=pl.Float64).alias(LABEL)))
    for name in FREQ_KEYS:
        assert a_tr[name].equals(b_tr[name]) and a_tg[name].equals(b_tg[name])


def test_frequency_encodings_are_shares_within_the_airport_and_null_for_a_null_key(feats):
    tr, tg = _split(feats)
    a_tr, a_tg = frequency_features(tr, tg)
    for name in FREQ_KEYS:
        v = pl.concat([a_tr[name], a_tg[name]]).drop_nulls().to_numpy()
        assert v.size and (v > 0).all() and (v <= 1).all()
    # the unmatched rows carry no operator, so their operator share must be null, not the share of "no operator"
    nulls = a_tr.filter(pl.col("AIRCRAFT_OPERATOR_flt").is_null())
    assert nulls.height and nulls["freq_operator"].null_count() == nulls.height


def test_a_stand_seen_only_on_the_target_side_still_gets_a_share(feats):
    tr, tg = _split(feats)
    tg = tg.with_columns(
        pl.when(pl.int_range(pl.len()) < 5).then(pl.lit("NEWSTAND")).otherwise(pl.col("STAND_mvt")).alias("STAND_mvt")
    )
    _, a_tg = frequency_features(tr, tg)
    assert a_tg.filter(pl.col("STAND_mvt") == "NEWSTAND")["freq_stand"].null_count() == 0


def test_aggregate_differences_are_the_row_value_minus_the_fitted_group_value(feats):
    tr, tg = _split(feats)
    a_tr, a_tg = aggdiff_features(tr, tg, min_count=1)
    for f in (a_tr, a_tg):
        want = (f[PROXY] - f["ref_med"]).to_numpy()
        got = f["agg_proxy_minus_ref_med"].to_numpy()
        ok = np.isfinite(want)
        assert ok.any() and np.allclose(got[ok], want[ok])
        want_s = (f[SCHED_OFFSET] - f["ref_med"]).to_numpy()
        ok_s = np.isfinite(want_s)
        assert np.allclose(f["agg_sched_minus_ref_med"].to_numpy()[ok_s], want_s[ok_s])
    # the spread table is fitted on the training labels only: blanking the target labels changes nothing
    _, b_tg = aggdiff_features(tr, tg.with_columns(pl.lit(None, dtype=pl.Float64).alias(LABEL)), min_count=1)
    assert a_tg["agg_ref_iqr"].equals(b_tg["agg_ref_iqr"])


def test_cyclic_time_is_on_the_unit_circle_and_midnight_is_next_to_23_59():
    df = pl.DataFrame(
        {
            "hour_local": [0, 23, 12],
            "MVT_TIME_UTC_mvt": pl.Series(
                ["2025-01-01 22:00:00", "2025-01-01 21:59:00", "2025-01-01 11:00:00"]
            ).str.to_datetime(),
        }
    )
    out = cyclic_features(df)
    s, c = out["cyc_sin"].to_numpy(), out["cyc_cos"].to_numpy()
    assert np.allclose(s**2 + c**2, 1.0)
    assert np.hypot(s[0] - s[1], c[0] - c[1]) < 0.01  # one minute apart on the circle
    assert np.hypot(s[0] - s[2], c[0] - c[2]) > 1.9  # noon is opposite midnight


def test_noise_columns_are_a_function_of_the_id_alone(feats):
    a = noise_features(feats, 3)
    b = noise_features(feats.reverse(), 3).reverse()
    for k in (1, 2, 3):
        assert a[f"noise_{k}"].equals(b[f"noise_{k}"])
    assert not a["noise_1"].equals(a["noise_2"])


def test_model_b_is_never_given_a_proxy_derived_column():
    cols = [
        "dep_tko_prev_15m",
        "agg_proxy_minus_ref_med",
        "agg_proxy_cell_dev",
        "agg_sched_minus_ref_med",
        "meta_prior",
        "meta_prior_minus_proxy",
        "mvt_minus_aobt3",
        "agg_ref_iqr",
    ]
    assert unmatched_columns(cols) == ["dep_tko_prev_15m", "agg_sched_minus_ref_med", "meta_prior", "agg_ref_iqr"]


def test_the_training_side_meta_prior_is_out_of_fold(feats):
    """Shuffling the labels of one calendar month must leave that month's training-side prior unchanged: its
    prior comes from the model fitted on the other months."""
    tr, tg = _split(feats)
    cfg = TrainConfig()
    num = visible_numeric_columns(tr, cfg)
    cats = ["ADEP_mvt", "RUNWAY_mvt", "STAND_mvt"]
    a_tr, _ = meta_prior_features(tr, tg, cfg, num, cats, {})
    feb = tr["MVT_TIME_UTC_mvt"].dt.month() == 2
    shuffled = tr.with_columns(pl.when(feb).then(pl.col(LABEL).shuffle(seed=3)).otherwise(pl.col(LABEL)).alias(LABEL))
    b_tr, _ = meta_prior_features(shuffled, tg, cfg, num, cats, {})
    pa, pb = a_tr["meta_prior"].to_numpy(), b_tr["meta_prior"].to_numpy()
    fm = feb.to_numpy()
    assert np.allclose(pa[fm], pb[fm])
    assert not np.allclose(pa[~fm], pb[~fm])  # the other month's model did see February, so it must move


def test_the_meta_prior_never_reads_a_label_derived_column(feats):
    tr, tg = _split(feats)
    info: dict = {}
    cfg = TrainConfig()
    num = visible_numeric_columns(tr, cfg)
    meta_prior_features(tr, tg, cfg, num, [], info)
    assert info["meta_prior_input_cols"] == len(
        [c for c in unmatched_columns(num) if not c.startswith(("ref_", "arh_", "apt_", "op_offset", "agg_"))]
    )


@pytest.mark.parametrize(
    "flags",
    [{"fe_freq": True}, {"fe_aggdiff": True}, {"fe_cyclic": True}, {"fe_meta_prior": True}, {"fe_noise_cols": 2}],
)
def test_each_family_runs_end_to_end_and_says_which_columns_it_added(feats, flags):
    m = feats["MVT_TIME_UTC_mvt"].dt.month()
    cfg = replace(TrainConfig(num_rounds=30, early_stopping=5, reference_min_count=1), **flags)
    pred, info = predict_fold(feats.filter(m < 3), feats.filter(m == 3), cfg)
    assert pred.height == feats.filter(m == 3).height and pred["pred"].null_count() == 0
    assert info["fold_feature_cols"]
