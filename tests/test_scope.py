from datetime import datetime

import polars as pl

from prc.scope import month_scope_columns
from prc.train import TrainConfig


def test_month_scope_counts_within_the_month_only():
    d = pl.DataFrame(
        {
            "MVT_ID_mvt": [1.0, 2.0, 3.0, 4.0],
            "ADEP_mvt": ["EDDF"] * 4,
            "FLIGHT_mvt": ["LH1", "LH1", "LH1", "LH1"],
            "FLIGHT_ID_mvt": [10.0, None, 12.0, 13.0],
            "STAND_mvt": ["A1", "A1", "A1", "A1"],
            "RUNWAY_mvt": ["25C", "07C", "25C", "25C"],
            "MVT_TIME_UTC_mvt": [
                datetime(2025, 1, 3),
                datetime(2025, 1, 9),
                datetime(2025, 2, 1),
                datetime(2025, 2, 2),
            ],
            "fl_nm_rate_other": pl.Series([0.0] * 4, dtype=pl.Float32),
            "stand_rwy_share": [0.0] * 4,
        }
    )
    out = month_scope_columns(d)
    assert out["MVT_ID_mvt"].equals(d["MVT_ID_mvt"])
    assert out["fl_nm_rate_other"].to_list() == [0.0, 1.0, 1.0, 1.0]  # Jan: the other is unmatched / matched
    assert out["stand_rwy_share"].to_list() == [0.5, 0.5, 1.0, 1.0]
    assert out.schema == d.schema


def test_off_by_default():
    from dataclasses import replace

    c = TrainConfig()
    assert c.fe_month_scope is False and replace(c, fe_month_scope=True).hash() != c.hash()
