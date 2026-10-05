"""OPDI airframe turnaround (docs/opdi_2026-09-30/plan.md)."""

from datetime import datetime

import numpy as np
import polars as pl

from prc.opdi import turnaround_table
from prc.train import TrainConfig, day_null_mask


def _opdi(tmp_path, rows):
    df = pl.DataFrame(
        rows,
        schema={
            "flt_id": pl.Utf8,
            "icao24": pl.Utf8,
            "ades": pl.Utf8,
            "first_seen": pl.Utf8,
            "last_seen": pl.Utf8,
        },
        orient="row",
    )
    f = tmp_path / "flight_list_202601.parquet"
    df.write_parquet(f)
    return [f]


def _deps(rows):
    return pl.DataFrame(
        rows,
        schema={
            "MVT_ID_mvt": pl.Float64,
            "ADEP_mvt": pl.Utf8,
            "CALLSIGN_flt": pl.Utf8,
            "MVT_TIME_UTC_mvt": pl.Datetime("us", "UTC"),
            "SCHED_TIME_UTC_mvt": pl.Datetime("us", "UTC"),
        },
        orient="row",
    )


def test_ground_time_is_this_lift_off_minus_the_same_airframes_previous_landing(tmp_path):
    files = _opdi(
        tmp_path,
        [
            ("DLH1", "abc", "EDDF", "2026-01-11 06:00:00", "2026-01-11 07:00:00"),  # inbound, lands at EDDF 07:00
            ("DLH2  ", "abc", "EGLL", "2026-01-11 08:30:10", "2026-01-11 09:40:00"),  # our departure, trailing spaces
            ("BAW9", "xyz", "LFPG", "2026-01-11 05:00:00", "2026-01-11 06:00:00"),  # inbound to another airport
            ("BAW10", "xyz", "EGLL", "2026-01-11 08:00:00", "2026-01-11 09:00:00"),
        ],
    )
    deps = _deps(
        [
            (1.0, "EDDF", "DLH2", datetime(2026, 1, 11, 8, 30), datetime(2026, 1, 11, 8, 0)),
            (2.0, "EDDF", "BAW10", datetime(2026, 1, 11, 8, 0), datetime(2026, 1, 11, 7, 30)),
            (3.0, "EDDF", "NOPE1", datetime(2026, 1, 11, 8, 0), datetime(2026, 1, 11, 7, 30)),
        ]
    )
    t = turnaround_table(deps, files).sort("MVT_ID_mvt")
    got = dict(zip(t["MVT_ID_mvt"].to_list(), t["opdi_ground_s"].to_list(), strict=True))
    assert got[1.0] == 5410.0  # 08:30:10 - 07:00:00
    assert got[2.0] is None  # previous leg ended at LFPG, not at the departure airport
    assert 3.0 not in got  # no OPDI flight with that callsign
    slack = dict(zip(t["MVT_ID_mvt"].to_list(), t["opdi_sched_slack_s"].to_list(), strict=True))
    assert slack[1.0] == 3600.0  # SCHED 08:00 - landing 07:00


def test_a_split_track_and_a_day_old_landing_are_not_turnarounds(tmp_path):
    files = _opdi(
        tmp_path,
        [
            ("AAA1", "a1", "EDDF", "2026-01-11 06:00:00", "2026-01-11 08:25:00"),  # 5 minutes before: a split track
            ("AAA2", "a1", "EGLL", "2026-01-11 08:30:00", "2026-01-11 09:30:00"),
            ("BBB1", "b1", "EDDF", "2026-01-09 06:00:00", "2026-01-09 07:00:00"),  # 49 hours before
            ("BBB2", "b1", "EGLL", "2026-01-11 08:00:00", "2026-01-11 09:00:00"),
        ],
    )
    deps = _deps(
        [
            (1.0, "EDDF", "AAA2", datetime(2026, 1, 11, 8, 30), datetime(2026, 1, 11, 8, 0)),
            (2.0, "EDDF", "BBB2", datetime(2026, 1, 11, 8, 0), datetime(2026, 1, 11, 7, 30)),
        ]
    )
    t = turnaround_table(deps, files)
    assert t["opdi_ground_s"].null_count() == 2


def test_the_day_blanking_is_a_function_of_the_date_alone():
    ts = pl.datetime_range(datetime(2025, 1, 1), datetime(2025, 12, 31, 23), "1h", eager=True, time_zone="UTC")
    df = pl.DataFrame({"MVT_TIME_UTC_mvt": ts})
    m = day_null_mask(df, 1 / 3)
    assert np.array_equal(m, day_null_mask(df.reverse(), 1 / 3)[::-1])
    by_day = pl.DataFrame({"d": ts.dt.date(), "m": m}).group_by("d").agg(pl.col("m").n_unique())
    assert (by_day["m"] == 1).all()  # whole days, never part of one
    assert 0.25 < m.mean() < 0.42


def test_opdi_is_off_by_default():
    cfg = TrainConfig()
    assert cfg.fe_opdi is False and cfg.opdi_target_null_day_share == 0.0


def test_opdi_exit_columns_join_by_id_and_blank_whole_days(monkeypatch):
    import prc.train as T

    table = pl.DataFrame({"MVT_ID_mvt": [1.0, 2.0], **{c: [10.0, 20.0] for c in T.OPDI_EXIT_COLS}})
    monkeypatch.setattr(T, "_opdi_exit_table", lambda: table)
    df = pl.DataFrame(
        {
            "MVT_ID_mvt": [2.0, 3.0, 1.0],
            "MVT_TIME_UTC_mvt": [datetime(2025, 1, 5), datetime(2025, 1, 5), datetime(2025, 1, 6)],
        }
    ).with_columns(pl.col("MVT_TIME_UTC_mvt").dt.replace_time_zone("UTC"))
    out = T.opdi_exit_features(df)
    assert out["MVT_ID_mvt"].to_list() == [2.0, 3.0, 1.0]
    assert out["opdi_exit_est_s"].to_list() == [20.0, None, 10.0]
    blank_all = T.opdi_exit_features(df, null_day_share=1.0)
    assert blank_all["opdi_exit_est_s"].null_count() == 3
    m_train = T.day_null_mask(df, 0.5, salt="train")
    assert len(m_train) == 3 and m_train[0] == m_train[1]  # same date, same decision
    cfg = T.TrainConfig()
    assert cfg.fe_opdi_exit is False and cfg.opdi_exit_train_null_day_share == 0.0
