from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from prc.train import LABEL, PROXY, SCHED_OFFSET, TrainConfig, _ucmix_blend


def _frame(n, seed):
    rng = np.random.default_rng(seed)
    apt = rng.choice(["LIRF", "EDDF"], n)
    x = rng.uniform(0, 1, n)
    ms = rng.uniform(1000, 20000, n)
    copy = x > 0.6
    y = np.where(copy, ms, 600 + 300 * x)
    proxy = np.where(rng.uniform(0, 1, n) < 0.7, y, np.nan)
    return pl.DataFrame(
        {"MVT_ID_mvt": np.arange(n, dtype=float) + seed * 1e6, "ADEP_mvt": apt, "x": x, SCHED_OFFSET: ms,
         LABEL: y, PROXY: proxy}
    ).with_columns(pl.col(PROXY).fill_nan(None))


def test_gated_rows_move_toward_truth_and_others_untouched():
    tr, tg = _frame(6000, 1), _frame(800, 2)
    tg = tg.with_columns(pl.when(pl.col("x") < 0.1).then(40000.0).otherwise(pl.col(SCHED_OFFSET)).alias(SCHED_OFFSET))
    cfg = replace(TrainConfig(), ucmix_airports=["LIRF"], num_rounds=80, early_stopping=10)
    pred_b = tg[SCHED_OFFSET].to_numpy().copy()
    info = {}
    out = _ucmix_blend(tr, tg, pred_b, cfg, ["x"], ["ADEP_mvt"], info)
    ms = tg[SCHED_OFFSET].to_numpy()
    gated = (tg["ADEP_mvt"].to_numpy() == "LIRF") & (ms < 28800)
    y = np.where(tg["x"].to_numpy() > 0.6, ms, 600 + 300 * tg["x"].to_numpy())
    assert np.array_equal(out[~gated], pred_b[~gated])
    assert np.abs(y[gated] - out[gated]).mean() < 0.7 * np.abs(y[gated] - pred_b[gated]).mean()
    assert info["ucmix"]["LIRF"]["target_rows"] == int(gated.sum())


def test_bad_weight_refused():
    tr, tg = _frame(2000, 1), _frame(10, 2)
    with pytest.raises(ValueError):
        _ucmix_blend(tr, tg, np.zeros(10), replace(TrainConfig(), ucmix_airports=["LIRF"], ucmix_weight=0.0),
                     ["x"], ["ADEP_mvt"], {})


def test_off_by_default():
    c = TrainConfig()
    assert c.ucmix_airports == [] and replace(c, ucmix_airports=["LIRF"]).hash() != c.hash()


def test_day_feats_are_label_free_and_order_preserving():
    from prc.train import UCDAY_COLS, _attach_day_feats

    f = _frame(500, 3).with_columns(
        pl.datetime(2025, 7, 1, time_zone="UTC").alias("MVT_TIME_UTC_mvt")
        + pl.duration(hours=pl.int_range(0, 500) % 72)
    )
    out = _attach_day_feats(f, f)
    assert out["MVT_ID_mvt"].equals(f["MVT_ID_mvt"])
    relabelled = _attach_day_feats(f.with_columns(pl.lit(1.0).alias(LABEL)), f.with_columns(pl.lit(1.0).alias(LABEL)))
    for c in UCDAY_COLS:
        assert out[c].equals(relabelled[c]), c
    first = f.filter((pl.col("ADEP_mvt") == "LIRF") & (pl.col("MVT_TIME_UTC_mvt").dt.day() == 1))
    row = out.filter(pl.col("MVT_ID_mvt") == first["MVT_ID_mvt"][0])
    assert row["ud_n_dep"][0] == first.height
    assert row["ud_n_unm"][0] == first[PROXY].is_null().sum()


def test_day_feats_run_through_the_blend():
    tr, tg = _frame(6000, 1), _frame(800, 2)
    t = pl.datetime(2025, 7, 1, time_zone="UTC") + pl.duration(hours=pl.int_range(0, pl.len()) % 240)
    tr, tg = tr.with_columns(t.alias("MVT_TIME_UTC_mvt")), tg.with_columns(t.alias("MVT_TIME_UTC_mvt"))
    cfg = replace(TrainConfig(), ucmix_airports=["LIRF"], ucmix_day_feats=True, num_rounds=40, early_stopping=10)
    u_tg = tg.filter(pl.col(PROXY).is_null())
    info = {}
    out = _ucmix_blend(tr, u_tg, u_tg[SCHED_OFFSET].to_numpy().copy(), cfg, ["x"], ["ADEP_mvt"], info, tg_all=tg)
    assert out.shape == (u_tg.height,) and np.isfinite(out).all()
    assert info["ucmix"]["LIRF"]["target_rows"] > 0
