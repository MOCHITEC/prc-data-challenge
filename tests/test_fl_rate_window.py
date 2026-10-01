"""The fl_nm_rate_other window-mismatch measurement (docs/gm_research_2026-10-01/plan_time_consistency.md §C)."""

from datetime import datetime

import polars as pl
import pytest

from prc.train import TrainConfig, fl_rate_window_features, fl_rate_window_rates


def _frame(rows):
    df = pl.DataFrame(
        rows,
        schema={
            "MVT_ID_mvt": pl.Float64,
            "ADEP_mvt": pl.Utf8,
            "FLIGHT_mvt": pl.Utf8,
            "FLIGHT_ID_mvt": pl.Float64,
            "MVT_TIME_UTC_mvt": pl.Datetime("us", "UTC"),
        },
        orient="row",
    )
    return df.with_columns(pl.lit(-1.0, dtype=pl.Float32).alias("fl_nm_rate_other"))


def _rows():
    # Flight AB1 at EDDF: matched in Jan and Jul, unmatched in Mar and Sep; the other flights of a Jan row are its
    # Jan/Jul twins only, so its window rate is 1.0 while its twelve-month rate would be 3 / 5.
    return [
        (1.0, "EDDF", "AB1", 10.0, datetime(2025, 1, 5)),
        (2.0, "EDDF", "AB1", 11.0, datetime(2025, 1, 6)),
        (3.0, "EDDF", "AB1", 12.0, datetime(2025, 7, 5)),
        (4.0, "EDDF", "AB1", None, datetime(2025, 3, 5)),
        (5.0, "EDDF", "AB1", None, datetime(2025, 9, 5)),
        (6.0, "EDDF", "AB1", 13.0, datetime(2025, 3, 6)),
    ]


def test_the_window_is_the_ranking_shape_months_m_and_m_plus_six():
    r = dict(fl_rate_window_rates(_frame(_rows())).iter_rows())
    assert r[1.0] == pytest.approx(1.0)  # Jan row: other flights = its Jan twin and the Jul flight, both matched
    assert r[4.0] == pytest.approx(1 / 2)  # Mar unmatched row: others = Sep (unmatched) and Mar 6 (matched)


def test_target_mode_changes_the_target_side_only_and_both_mode_changes_both():
    f = _frame(_rows())
    tr, tg = f.filter(pl.col("MVT_ID_mvt") > 2), f.filter(pl.col("MVT_ID_mvt") <= 2)
    info: dict = {}
    tr1, tg1 = fl_rate_window_features(tr, tg, "target", info)
    assert tr1["fl_nm_rate_other"].to_list() == tr["fl_nm_rate_other"].to_list()
    assert tg1["fl_nm_rate_other"].to_list() == pytest.approx([1.0, 1.0])
    tr2, _ = fl_rate_window_features(tr, tg, "both", {})
    assert tr2["fl_nm_rate_other"].to_list() != tr["fl_nm_rate_other"].to_list()
    assert tr2["MVT_ID_mvt"].to_list() == tr["MVT_ID_mvt"].to_list()


def test_a_ranking_like_frame_of_one_month_pair_is_unchanged_in_meaning():
    # Jan + Jul 2026 only: the window is the whole frame, as prc.features_extra computes it on the ranking frame.
    rows = [
        (11.0, "EDDF", "XY9", 1.0, datetime(2026, 1, 3)),
        (12.0, "EDDF", "XY9", None, datetime(2026, 7, 3)),
        (13.0, "EDDF", "XY9", 2.0, datetime(2026, 7, 4)),
    ]
    r = dict(fl_rate_window_rates(_frame(rows)).iter_rows())
    assert r[11.0] == pytest.approx(0.5) and r[12.0] == pytest.approx(1.0) and r[13.0] == pytest.approx(0.5)


def test_bad_mode_is_refused_and_the_flag_is_off_by_default():
    with pytest.raises(ValueError):
        fl_rate_window_features(_frame(_rows()), _frame(_rows()), "train", {})
    assert TrainConfig().fe_fl_rate_window == ""


def test_seq_window_swaps_cached_rows_only_and_keeps_dtypes(monkeypatch):
    import prc.train as T

    w = pl.DataFrame({"MVT_ID_mvt": [1.0, 2.0], "seq_burst_len__w": [5.0, 6.0], "fl_nm_rate_other__w": [0.5, 0.25]})
    monkeypatch.setattr(T, "_seq_window_table", lambda: w)
    f = _frame(_rows()).with_columns(pl.lit(1.0, dtype=pl.Float32).alias("seq_burst_len"))
    tr, tg = f.filter(pl.col("MVT_ID_mvt") > 1), f.filter(pl.col("MVT_ID_mvt") <= 1)
    info: dict = {}
    tr1, tg1 = T.seq_window_features(tr, tg, "target", info)
    assert tg1["seq_burst_len"].to_list() == [5.0] and tg1["fl_nm_rate_other"].to_list() == [0.5]
    assert tr1.equals(tr) and tg1.schema == tg.schema
    tr2, _ = T.seq_window_features(tr, tg, "both", {})
    assert tr2["seq_burst_len"].to_list() == [6.0, 1.0, 1.0, 1.0, 1.0]  # only id 2 is in the window cache
    assert TrainConfig().fe_seq_window == ""
