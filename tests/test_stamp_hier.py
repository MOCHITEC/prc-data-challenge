from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import polars as pl

from prc.train import LABEL, SCHED_OFFSET, STAMP_COLS, TrainConfig, stamp_hier_features


def _frame(rows):
    # (flight, ades, month, ms, y)
    return pl.DataFrame(
        {
            "MVT_ID_mvt": [float(i) for i in range(len(rows))],
            "ADEP_mvt": ["LIRF"] * len(rows),
            "FLIGHT_mvt": [r[0] for r in rows],
            "ADES_mvt": [r[1] for r in rows],
            "MVT_TIME_UTC_mvt": [datetime(2025, r[2], 1, tzinfo=UTC) for r in rows],
            SCHED_OFFSET: [float(r[3]) for r in rows],
            LABEL: [None if r[4] is None else float(r[4]) for r in rows],
        }
    )


def test_hierarchy_shrinks_toward_parent_and_is_leave_month_out():
    tr = _frame(
        [
            ("AZA101", "LIML", 1, 3000, 3000),  # echo
            ("AZA101", "LIML", 7, 3000, 3000),  # echo
            ("AZA202", "LICC", 7, 3000, 900),
            ("RYR1", "LEMD", 7, 2000, 800),
        ]
    )
    tg = _frame([("AZA101", "LIML", 1, 5000, None), ("AZA999", "EGLL", 1, 5000, None)])
    a, b = stamp_hier_features(tr, tg)
    assert a["MVT_ID_mvt"].equals(tr["MVT_ID_mvt"]) and b["MVT_ID_mvt"].equals(tg["MVT_ID_mvt"])
    assert set(STAMP_COLS) <= set(b.columns)
    # target AZA101 is seen twice (2 echoes): its rate sits above the AZA carrier rate it is shrunk toward
    assert b["stamp_flt"][0] > b["stamp_car"][0]
    # unseen flight and destination fall back to the carrier rate, with zero count
    assert np.isclose(b["stamp_flt"][1], b["stamp_car"][1]) and b["stamp_flt_n"][1] == 0.0
    # training row 0 (Jan) only sees July: AZA101 seen once there
    assert a["stamp_flt_n"][0] == 1.0


def test_off_by_default():
    c = TrainConfig()
    assert c.fe_stamp_hier is False and replace(c, fe_stamp_hier=True).hash() != c.hash()
