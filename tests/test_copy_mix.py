from dataclasses import replace

import numpy as np
import polars as pl

from prc.train import LABEL, LOBT_OFFSET, PROXY, SCHED_OFFSET, TrainConfig, _copy_mixture


def _frame(n, seed, label=True):
    rng = np.random.default_rng(seed)
    apt = rng.choice(["LIRF", "EDDF"], n)
    ms = rng.uniform(600, 4000, n)
    proxy = rng.uniform(500, 1500, n)
    copy = rng.random(n) < np.where(ms > 2500, 0.8, 0.05)  # copies are predictable from ms
    y = np.where(copy, ms, proxy + rng.normal(0, 60, n))
    d = pl.DataFrame(
        {
            "MVT_ID_mvt": np.arange(n, dtype=float) + seed * 1e6,
            "ADEP_mvt": apt,
            SCHED_OFFSET: ms,
            PROXY: proxy,
            LOBT_OFFSET: ms,  # SCHED inside the window
            "x": ms,
        }
    )
    return d.with_columns(pl.Series(LABEL, y)) if label else d


def test_mixture_helps_where_copies_are_predictable_and_leaves_other_airports():
    tr, tg = _frame(6000, 1), _frame(3000, 2)
    cfg = replace(TrainConfig(), copy_mix_airports=["LIRF"], num_rounds=60, early_stopping=10)
    base = tg[PROXY].to_numpy().astype(float)  # a model A that ignores copies

    def target(f):
        return f[LABEL].to_numpy().astype(float) - f[PROXY].to_numpy().astype(float)

    info = {}
    out = _copy_mixture(tr, tg, base, target, False, cfg, ["x", PROXY], ["ADEP_mvt"], info)
    y = tg[LABEL].to_numpy()
    lirf = tg["ADEP_mvt"].to_numpy() == "LIRF"
    assert np.array_equal(out[~lirf], base[~lirf])
    rmse = lambda p: np.sqrt(np.mean((y[lirf] - p[lirf]) ** 2))
    assert rmse(out) < 0.7 * rmse(base)
    assert info["copy_mix"]["LIRF"]["target_rows"] == int(lirf.sum())


def test_p_is_zero_outside_the_lobt_window():
    tr, tg = _frame(6000, 1), _frame(500, 3)
    tg = tg.with_columns((pl.col(SCHED_OFFSET) + 10000).alias(LOBT_OFFSET))
    cfg = replace(TrainConfig(), copy_mix_airports=["LIRF"], num_rounds=60, early_stopping=10)

    def target(f):
        return f[LABEL].to_numpy().astype(float) - f[PROXY].to_numpy().astype(float)

    info = {}
    _copy_mixture(tr, tg, tg[PROXY].to_numpy().astype(float), target, False, cfg, ["x", PROXY], ["ADEP_mvt"], info)
    assert info["copy_mix"]["LIRF"]["target_mean_p"] == 0.0


def test_off_by_default():
    c = TrainConfig()
    assert c.copy_mix_airports == [] and replace(c, copy_mix_airports=["LIRF"]).hash() != c.hash()
