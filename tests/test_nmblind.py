from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from prc.train import LABEL, TrainConfig, _nmblind_blend


def _frame(n, seed, label=True):
    rng = np.random.default_rng(seed)
    apt = rng.choice(["LIRF", "EDDF"], n)
    x = rng.uniform(0, 1, n)
    d = pl.DataFrame({"MVT_ID_mvt": np.arange(n, dtype=float) + seed * 1e6, "ADEP_mvt": apt, "x": x})
    return d.with_columns(pl.Series(LABEL, 600 + 900 * x)) if label else d


def test_blend_skips_reparam_airports_and_moves_others_toward_truth():
    tr, tg = _frame(4000, 1), _frame(600, 2)
    cfg = replace(
        TrainConfig(),
        nmblind_weight=0.5,
        unmatched_sched_reparam=True,
        unmatched_sched_reparam_airports=("LIRF",),
        num_rounds=80,
        early_stopping=10,
    )
    pred_b = np.full(tg.height, 2000.0)
    info = {}
    out = _nmblind_blend(tr, tg, pred_b, cfg, ["x"], ["ADEP_mvt"], info)
    lirf = tg["ADEP_mvt"].to_numpy() == "LIRF"
    y = tg[LABEL].to_numpy()
    assert np.array_equal(out[lirf], pred_b[lirf])
    assert np.abs(y[~lirf] - out[~lirf]).mean() < np.abs(y[~lirf] - pred_b[~lirf]).mean()
    assert info["nmblind"]["target_rows"] == int((~lirf).sum())


def test_weight_out_of_range_refused():
    tr, tg = _frame(1000, 1), _frame(10, 2)
    with pytest.raises(ValueError):
        _nmblind_blend(tr, tg, np.zeros(10), replace(TrainConfig(), nmblind_weight=1.5), ["x"], ["ADEP_mvt"], {})


def test_off_by_default():
    c = TrainConfig()
    assert c.nmblind_weight == 0.0 and replace(c, nmblind_weight=0.25).hash() != c.hash()
