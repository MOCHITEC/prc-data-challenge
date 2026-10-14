import numpy as np
import polars as pl

from prc.train import NEW_STAND, TrainConfig, apply_new_stand_token


def _frames():
    n = 4000
    rng = np.random.default_rng(0)
    tr = pl.DataFrame(
        {
            "MVT_ID_mvt": np.arange(n, dtype=float),
            "ADEP_mvt": rng.choice(["EDDF", "EDDM"], n),
            "STAND_mvt": rng.choice(["A1", "A2", "B7"], n),
            "ref_p10": rng.uniform(300, 600, n),
            "ref_p20": rng.uniform(300, 600, n),
            "ref_med": rng.uniform(300, 600, n),
            "ref_n": np.full(n, 50, dtype=np.uint32),
            "arh_med": np.full(n, 800.0),
            "apt_med": np.full(n, 900.0),
        }
    ).with_columns(pl.col("ref_p10").alias("ref_p10_fb"), pl.lit(0).cast(pl.Int8).alias("ref_missing"))
    tg = pl.DataFrame(
        {
            "MVT_ID_mvt": [1e6, 1e6 + 1, 1e6 + 2, 1e6 + 3],
            "ADEP_mvt": ["EDDF", "EDDF", "EDDM", "EDDM"],
            "STAND_mvt": ["A1", "H12", None, "B7"],
        }
    )
    return tr, tg


def test_dropout_share_and_blanking():
    tr, tg = _frames()
    info = {}
    a, _ = apply_new_stand_token(tr, tg, 0.1, info)
    d = a.filter(pl.col("STAND_mvt") == NEW_STAND)
    assert 0.07 < d.height / a.height < 0.13 and info["stand_dropout_train_rows"] == d.height
    assert d["ref_p10"].null_count() == d.height and d["ref_n"].null_count() == d.height
    assert (d["ref_p10_fb"] == 800.0).all() and (d["ref_missing"] == 1).all()
    kept = a.filter(pl.col("STAND_mvt") != NEW_STAND)
    assert kept["ref_p10"].null_count() == 0 and (kept["ref_missing"] == 0).all()
    assert a["MVT_ID_mvt"].equals(tr["MVT_ID_mvt"])  # order and count unchanged
    # deterministic
    a2, _ = apply_new_stand_token(tr, tg, 0.1)
    assert a2["STAND_mvt"].equals(a["STAND_mvt"])


def test_target_unseen_stands_get_the_token_only():
    tr, tg = _frames()
    _, b = apply_new_stand_token(tr, tg, 0.1)
    assert b["STAND_mvt"].to_list() == ["A1", NEW_STAND, None, "B7"]
    assert b["MVT_ID_mvt"].equals(tg["MVT_ID_mvt"])


def test_off_by_default_and_id_unchanged():
    c = TrainConfig()
    assert c.stand_dropout == 0.0
    from dataclasses import replace

    assert replace(c, stand_dropout=0.05).hash() != c.hash()
