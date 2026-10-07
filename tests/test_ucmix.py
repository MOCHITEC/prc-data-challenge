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
