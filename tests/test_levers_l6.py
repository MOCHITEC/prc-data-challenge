from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from prc.levers import L6_COLS, _retention_frame
from prc.train import TrainConfig

T0 = datetime(2025, 7, 1, 10, 0, tzinfo=UTC)


def _t(m):
    return None if m is None else T0 + timedelta(minutes=m)


def _dep(rows):
    return pl.DataFrame(
        {
            "MVT_ID_mvt": [float(i) for i in range(len(rows))],
            "ADEP_mvt": [r[0] for r in rows],
            "RUNWAY_mvt": [r[1] for r in rows],
            "EOBT_1_flt": [_t(r[2]) for r in rows],
            "AOBT_3_flt": [_t(r[3]) for r in rows],
            "SCHED_TIME_UTC_mvt": [_t(r[4]) for r in rows],
        },
        schema_overrides={c: pl.Datetime("us", "UTC") for c in ("EOBT_1_flt", "AOBT_3_flt", "SCHED_TIME_UTC_mvt")},
    )


def test_overdue_counts_by_hand():
    # (airport, runway, EOBT, AOBT_3, SCHED) in minutes after 10:00
    d = _dep(
        [
            ("EDDF", "25C", 0, 20, 0),  # 0: planned 10:00, pushed 10:20
            ("EDDF", "25C", 5, 30, 5),  # 1: overdue at 10:20 (5 <= 20 < 30)
            ("EDDF", "07C", 10, 40, 10),  # 2: overdue at 10:20, other runway
            ("EDDF", "25C", 25, 26, 25),  # 3: not yet planned at 10:20
            ("EGLL", "27R", 0, 50, 0),  # other airport: never counted for EDDF
        ]
    )
    f = _retention_frame(d)
    assert f.columns == ["MVT_ID_mvt", *L6_COLS]
    assert f["ret_overdue_eobt"].to_list()[0] == 2.0  # flights 1 and 2
    assert f["ret_overdue_eobt_rwy"].to_list()[0] == 1.0  # flight 1 only
    assert f["ret_overdue_sched"].to_list()[0] == 2.0
    assert f["ret_overdue_eobt"].to_list()[4] == 0.0
    # 15-min mean delay of pushes in [10:15, 10:30) seen from flight 1 (pushed 10:30): flights 0 (20 min) and 3 (1 min)
    assert np.isclose(f["ret_dly15"].to_list()[1], (20 * 60 + 1 * 60) / 2)
    assert f["ret_dly15"].to_list()[0] is None  # nothing pushed in [10:05, 10:20)


def test_null_clock_gives_null_and_ids_unchanged():
    d = _dep([("EDDF", "25C", 0, None, 0), ("EDDF", "25C", 5, 30, 5)])
    f = _retention_frame(d)
    assert f["ret_overdue_eobt"].to_list()[0] is None
    assert f["MVT_ID_mvt"].equals(d["MVT_ID_mvt"])


def test_off_by_default():
    from dataclasses import replace

    c = TrainConfig()
    assert c.fe_retention is False and replace(c, fe_retention=True).hash() != c.hash()
