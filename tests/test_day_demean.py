from dataclasses import replace

import numpy as np
import polars as pl

from prc.train import TrainConfig, day_demean_features


def test_demeaned_columns_have_zero_day_mean_and_keep_order():
    rng = np.random.default_rng(0)
    n = 600
    f = pl.DataFrame(
        {
            "MVT_ID_mvt": np.arange(n, dtype=float),
            "ADEP_mvt": rng.choice(["LIRF", "EDDF"], n),
            "queue_ahead": rng.uniform(0, 10, n),
            "nbr_dep_n_30m": rng.integers(0, 20, n),
        }
    ).with_columns(
        (pl.datetime(2025, 7, 1, time_zone="UTC") + pl.duration(hours=pl.int_range(0, n) % 96)).alias("MVT_TIME_UTC_mvt")
    )
    out = day_demean_features(f)
    assert out["MVT_ID_mvt"].equals(f["MVT_ID_mvt"])
    assert {"dd_queue_ahead", "dd_nbr_dep_n_30m"} <= set(out.columns)
    g = out.group_by("ADEP_mvt", pl.col("MVT_TIME_UTC_mvt").dt.date()).agg(pl.col("dd_queue_ahead").mean())
    assert np.allclose(g["dd_queue_ahead"].to_numpy(), 0.0)


def test_off_by_default():
    c = TrainConfig()
    assert c.fe_day_demean is False and replace(c, fe_day_demean=True).hash() != c.hash()
