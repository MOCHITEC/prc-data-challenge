from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from prc.train import LABEL, SCHED_OFFSET, TrainConfig, _gmodel_blend


def _frame(n, seed):
    rng = np.random.default_rng(seed)
    apt = rng.choice(["LIRF", "EDDF"], n)
    ms = rng.uniform(600, 6000, n)
    x = rng.uniform(0, 1, n)
    g = 300 + 2000 * x  # BLOCK - SCHED, learnable from x
    return pl.DataFrame(
        {"MVT_ID_mvt": np.arange(n, dtype=float) + seed * 1e6, "ADEP_mvt": apt, SCHED_OFFSET: ms, "x": x}
    ).with_columns(pl.Series(LABEL, ms - g))


def test_blend_moves_only_the_listed_airport_and_towards_the_truth():
    tr, tg = _frame(5000, 1), _frame(800, 2)
    cfg = replace(TrainConfig(), gmodel_airports=["LIRF"], gmodel_lambda=0.5, num_rounds=80, early_stopping=10)
    pred_b = np.full(tg.height, 1500.0)
    info = {}
    out = _gmodel_blend(tr, tg, pred_b, cfg, ["x"], ["ADEP_mvt"], info)
    lirf = tg["ADEP_mvt"].to_numpy() == "LIRF"
    y = tg[LABEL].to_numpy()
    assert np.array_equal(out[~lirf], pred_b[~lirf])
    assert np.sqrt(np.mean((y[lirf] - out[lirf]) ** 2)) < np.sqrt(np.mean((y[lirf] - pred_b[lirf]) ** 2))
    assert info["gmodel"]["LIRF"]["target_rows"] == int(lirf.sum())


def test_lambda_out_of_range_is_refused():
    tr, tg = _frame(2000, 1), _frame(100, 2)
    cfg = replace(TrainConfig(), gmodel_airports=["LIRF"], gmodel_lambda=1.5)
    with pytest.raises(ValueError):
        _gmodel_blend(tr, tg, np.zeros(tg.height), cfg, ["x"], ["ADEP_mvt"], {})


def test_off_by_default():
    c = TrainConfig()
    assert c.gmodel_airports == [] and replace(c, gmodel_airports=["LIRF"]).hash() != c.hash()
