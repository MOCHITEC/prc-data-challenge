"""Lever L5 (prc.levers, TrainConfig.fe_tx_nm): same-period tx_* context from other departures.

docs/research_2026-10-01/README.md. Hand-built values, self-exclusion, parity (no departure BLOCK_TIME or label is
read), and lane isolation (model B never sees the columns)."""

from datetime import UTC, datetime

import polars as pl
import pytest

from prc.levers import L5_COLS, attach_levers, tx_normal_tables
from prc.train import TrainConfig, unmatched_columns, visible_numeric_columns


def _t(h, m):
    return datetime(2025, 1, 15, h, m, tzinfo=UTC)


def _raw():
    # four LIRF departures; row 3 has no NM record; row 4 is 3.5 h late against schedule
    dep = pl.DataFrame(
        {
            "MVT_ID_mvt": [1.0, 2.0, 3.0, 4.0],
            "PHASE_mvt": ["DEP"] * 4,
            "ADEP_mvt": ["LIRF"] * 4,
            "ADES_mvt": ["AAAA"] * 4,
            "AIRCRAFT_TYPE_mvt": ["A320"] * 4,
            "RUNWAY_mvt": ["25"] * 4,
            "MVT_TIME_UTC_mvt": [_t(10, 0), _t(10, 10), _t(10, 20), _t(11, 30)],
            "AOBT_3_flt": [_t(9, 40), _t(9, 50), None, _t(11, 0)],
            "SCHED_TIME_UTC_mvt": [_t(9, 30), _t(9, 45), _t(10, 0), _t(8, 0)],
            "EOBT_1_flt": [None] * 4,
            "ARVT_1_flt": [None] * 4,
            "BLOCK_TIME_UTC_mvt": [_t(9, 41), _t(9, 52), _t(10, 2), _t(11, 3)],
            "TAXITIME_SEC_mvt": [1140, 1080, 1080, 1620],
        },
        schema_overrides={c: pl.Datetime("us", "UTC") for c in ("AOBT_3_flt", "EOBT_1_flt", "ARVT_1_flt")},
    )
    return dep


def _l5(raw_tr, raw_rk=None):
    raw_rk = raw_tr if raw_rk is None else raw_rk
    f_tr = raw_tr.select("MVT_ID_mvt")
    f_rk = raw_rk.select("MVT_ID_mvt")
    a, b = attach_levers(f_tr, f_rk, raw_tr, raw_rk, False, False, False, l5=True)
    return a, b


def test_hand_built_values_and_self_exclusion():
    out, _ = _l5(_raw())
    assert out.columns == ["MVT_ID_mvt", *L5_COLS]
    # proxies (MVT - AOBT_3): 1,200 / 1,200 / null / 1,800; airport normal = mean of those in [0, 7200] = 1,400
    norm = 1400.0
    # +-60 min around 10:00 contains rows 2 (1,200) and 3 (null) besides itself -> mean 1,200, n 1
    assert out["tx_mma_c60_n"].to_list() == [1.0, 1.0, 2.0, 0.0]
    assert out["tx_mma_c60_ratio"].to_list() == pytest.approx([1200 / norm, 1200 / norm, 1200 / norm, None])
    # forward (t, t+60m]: row 1 sees rows 2 (1,200) and 3 (null); row 3 sees nothing with a proxy before 11:20
    assert out["tx_mma_f60_n"].to_list() == [1.0, 0.0, 0.0, 0.0]
    # share without NM record in +-60 min (self excluded): row 1 sees rows 2, 3 -> 0.5; row 3 sees 1, 2 -> 0
    assert out["tx_unm_c60"].to_list() == pytest.approx([0.5, 0.5, 0.0, None])
    # late (MVT - SCHED > 3 h) only row 4; +-120 min of row 3 (10:20) contains rows 1, 2, 4 -> 1/3
    assert out["tx_late3h_c120"].to_list()[2] == pytest.approx(1 / 3)
    # take-offs in +-30 min, self excluded
    assert out["tx_tko_c60"].to_list() == [2.0, 2.0, 2.0, 0.0]


def test_no_departure_block_time_or_label_is_read():
    raw = _raw()
    moved = raw.with_columns(
        pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("BLOCK_TIME_UTC_mvt"),
        pl.lit(None, dtype=pl.Int32).alias("TAXITIME_SEC_mvt"),
    )
    a, _ = _l5(raw)
    b, _ = _l5(moved)
    assert a.equals(b)


def test_normals_come_from_the_training_frame_only():
    tr = _raw()
    rk = tr.with_columns(pl.col("AOBT_3_flt") - pl.duration(minutes=30))  # ranking proxies much longer
    _, b = _l5(tr, rk)
    mma, _ = tx_normal_tables(tr.select("MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"))
    assert mma == {"LIRF": pytest.approx(1400.0)}
    # ranking row 1 sees ranking row 2's proxy (3,000 s) over the TRAINING normal (1,400 s)
    assert b["tx_mma_c60_ratio"][0] == pytest.approx(3000 / 1400)


def test_lane_isolation_and_flag():
    cols = list(L5_COLS) + ["mvt_minus_sched"]
    frame = pl.DataFrame({c: [1.0] for c in cols})
    assert not set(L5_COLS) & set(visible_numeric_columns(frame, TrainConfig()))
    assert set(L5_COLS) <= set(visible_numeric_columns(frame, TrainConfig(fe_tx_nm=True)))
    assert not set(L5_COLS) & set(unmatched_columns(cols))
    assert TrainConfig().hash() == TrainConfig(fe_tx_nm=False).hash()
    assert TrainConfig().hash() != TrainConfig(fe_tx_nm=True).hash()
