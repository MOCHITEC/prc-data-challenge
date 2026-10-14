from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import polars as pl

from prc.train import CAR_COLS, LABEL, PROXY, SCHED_OFFSET, TrainConfig, carrier_echo_features


def _frame(rows):
    # (flight, airport, month, ms, y, matched)
    return pl.DataFrame(
        {
            "MVT_ID_mvt": [float(i) for i in range(len(rows))],
            "FLIGHT_mvt": [r[0] for r in rows],
            "ADEP_mvt": [r[1] for r in rows],
            "MVT_TIME_UTC_mvt": [datetime(2025, r[2], 1, tzinfo=UTC) for r in rows],
            SCHED_OFFSET: [float(r[3]) for r in rows],
            LABEL: [None if r[4] is None else float(r[4]) for r in rows],
            PROXY: [1.0 if r[5] else None for r in rows],
        }
    )


def test_rates_are_leave_month_out_on_training_rows():
    tr = _frame(
        [
            ("AZA101", "LIRF", 1, 3000, 3000, False),  # echo
            ("AZA102", "LIRF", 1, 3000, 1000, False),
            ("AZA103", "LIRF", 7, 3000, 3000, False),  # echo
            ("RYR9", "LIRF", 7, 2000, 900, True),
        ]
    )
    tg = _frame([("AZA555", "LIRF", 1, 5000, None, False), ("XX1", "EDDF", 1, 900, None, False)])
    a, b = carrier_echo_features(tr, tg)
    assert a["MVT_ID_mvt"].equals(tr["MVT_ID_mvt"]) and b["MVT_ID_mvt"].equals(tg["MVT_ID_mvt"])
    # row 0 (Jan) sees only July: AZA has 1 echo of 1 unmatched row; LIRF July prior = 1/2 over all, 1/1 over unmatched
    p_all = 0.5
    assert np.isclose(a["car_echo_all"][0], (1 + 20 * p_all) / (1 + 20))
    assert a["car_n_unm"][0] == 1.0
    # target AZA at LIRF sees all training rows: 2 echoes of 3 AZA rows
    p_all_full = 2 / 4
    assert np.isclose(b["car_echo_all"][0], (2 + 20 * p_all_full) / (3 + 20))
    assert b["car_echo_all"][1] is None and b["car_n_all"][1] == 0.0  # unseen airport/carrier
    assert set(CAR_COLS) <= set(b.columns)


def test_off_by_default():
    c = TrainConfig()
    assert c.fe_carrier_echo is False and replace(c, fe_carrier_echo=True).hash() != c.hash()
