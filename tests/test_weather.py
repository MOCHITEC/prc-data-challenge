from datetime import datetime

import polars as pl
import pytest

from prc.weather import hourly_weather_table, join_weather, parse_metar_csv

SAMPLE = """station,valid,tmpf,dwpf,relh,drct,sknt,gust,p01i,vsby,wxcodes,skyc1,feel
EHAM,2026-01-06 05:25,30.20,28.40,92.85,50.00,12.00,M,T,1.24,-SN,OVC,21.60
EHAM,2026-01-06 05:55,30.20,28.40,92.85,60.00,14.00,25.00,0.01,0.62,SN BR,OVC,20.10
EHAM,2026-01-06 06:25,33.80,32.00,93.00,70.00,8.00,M,M,6.21,M,BKN,28.90
LEBL,2026-07-15 12:00,86.00,60.80,44.00,180.00,10.00,M,0.00,9.94,M,FEW,88.00
"""


def test_parse_and_units():
    df = parse_metar_csv(SAMPLE)
    assert df.height == 4
    row = df.filter((pl.col("station") == "EHAM") & (pl.col("obs_time_utc").dt.minute() == 55)).row(0, named=True)
    assert abs(row["temp_c"] - (-1.0)) < 0.05  # 30.2 F
    assert abs(row["precip_mm"] - 0.254) < 0.01  # 0.01 in
    assert abs(row["vis_km"] - 0.998) < 0.02  # 0.62 mi
    assert abs(row["wind_ms"] - 7.2) < 0.05  # 14 kt
    trace = df.filter(pl.col("obs_time_utc").dt.minute() == 25).sort("station").row(0, named=True)
    assert trace["precip_mm"] is not None and trace["precip_mm"] < 0.01  # trace 'T' kept as ~0


def test_deicing_gate_and_join():
    hourly = hourly_weather_table(parse_metar_csv(SAMPLE))
    eham5 = hourly.filter((pl.col("station") == "EHAM") & (pl.col("hour_utc").dt.hour() == 5)).row(0, named=True)
    assert eham5["deicing_gate"] == 1 and eham5["frozen_wx"] and eham5["low_vis"] == 1
    lebl = hourly.filter(pl.col("station") == "LEBL").row(0, named=True)
    assert lebl["deicing_gate"] == 0 and lebl["low_vis"] == 0
    dep = pl.DataFrame(
        {
            "MVT_ID_mvt": [1, 2],
            "ADEP_mvt": ["EHAM", "LEBL"],
            "MVT_TIME_UTC_mvt": [datetime(2026, 1, 6, 6, 10), datetime(2026, 7, 15, 12, 30)],
        },
        schema_overrides={"MVT_TIME_UTC_mvt": pl.Datetime("us", "UTC")},  # what prc.features actually produces
    )
    j = join_weather(dep, hourly)
    r = j.filter(pl.col("MVT_ID_mvt") == 1).row(0, named=True)
    assert r["deicing_gate"] == 0  # 06:25 obs is above 3 C
    assert r["deicing_gate_prev1h"] == 1  # the 05:xx hour was freezing with snow


@pytest.mark.parametrize("tz", [None, "UTC"])
def test_join_weather_accepts_either_time_zone_on_the_departure_side(tz):
    """The real feature frame is timezone-aware UTC and the METAR parse is naive; polars will not join those."""
    hourly = pl.DataFrame(
        {
            "station": ["EDDF", "EDDF"],
            "hour_utc": [datetime(2026, 1, 6, 5), datetime(2026, 1, 6, 6)],
            "temp_c": [-2.0, -1.0],
            "deicing_gate": [1, 0],
        }
    )
    dep = pl.DataFrame(
        {"MVT_ID_mvt": [1], "ADEP_mvt": ["EDDF"], "MVT_TIME_UTC_mvt": [datetime(2026, 1, 6, 6, 10)]},
        schema_overrides={"MVT_TIME_UTC_mvt": pl.Datetime("us", tz)},
    )
    out = join_weather(dep, hourly, columns=None, prev1h=None)  # column choice is tested separately
    assert out.height == 1
    assert out["deicing_gate"][0] == 0  # the take-off hour
    assert out["deicing_gate_prev1h"][0] == 1  # and the hour before it
    assert out["temp_c"][0] == -1.0


def test_join_weather_refuses_to_multiply_rows():
    """A duplicated station-hour would silently double-count departures downstream."""
    hourly = pl.DataFrame(
        {
            "station": ["EDDF", "EDDF"],
            "hour_utc": [datetime(2026, 1, 6, 6), datetime(2026, 1, 6, 6)],
            "temp_c": [-2.0, -1.0],
        }
    )
    dep = pl.DataFrame(
        {"MVT_ID_mvt": [1], "ADEP_mvt": ["EDDF"], "MVT_TIME_UTC_mvt": [datetime(2026, 1, 6, 6, 10)]},
        schema_overrides={"MVT_TIME_UTC_mvt": pl.Datetime("us", "UTC")},
    )
    with pytest.raises(AssertionError, match="changed the row count"):
        join_weather(dep, hourly, columns=None, prev1h=None)


def _full_hourly() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "station": ["EDDF", "EDDF"],
            "hour_utc": [datetime(2026, 1, 6, 5), datetime(2026, 1, 6, 6)],
            "temp_c": [-2.0, -1.0],
            "dewp_spread_c": [1.0, 1.5],
            "wind_ms": [4.0, 5.0],
            "gust_ms": [None, 9.0],
            "precip_mm": [0.3, 0.0],
            "vis_km": [0.8, 3.0],
            "frozen_wx": [True, False],
            "deicing_gate": [1, 0],
            "low_vis": [1, 0],
            "temp_x_precip": [-0.6, 0.0],
        }
    )


def _dep() -> pl.DataFrame:
    return pl.DataFrame(
        {"MVT_ID_mvt": [1], "ADEP_mvt": ["EDDF"], "MVT_TIME_UTC_mvt": [datetime(2026, 1, 6, 6, 10)]},
        schema_overrides={"MVT_TIME_UTC_mvt": pl.Datetime("us", "UTC")},
    )


def test_only_the_mechanism_columns_are_carried_by_default():
    """v11 carried all twenty and gained nothing net on the scored months; the winter columns are July noise."""
    from prc.weather import MECHANISM_COLUMNS, MECHANISM_PREV1H

    out = join_weather(_dep(), _full_hourly())
    added = [c for c in out.columns if c not in ("MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt")]
    assert sorted(added) == sorted([*MECHANISM_COLUMNS, *(f"{c}_prev1h" for c in MECHANISM_PREV1H)])
    assert "gust_ms" not in added and "frozen_wx" not in added and "temp_x_precip" not in added
    assert out["deicing_gate"][0] == 0 and out["deicing_gate_prev1h"][0] == 1
    assert "vis_km_prev1h" not in added  # visibility does not leave a queue behind it


def test_all_columns_are_available_on_request():
    out = join_weather(_dep(), _full_hourly(), columns=None, prev1h=None)
    assert "gust_ms" in out.columns and "gust_ms_prev1h" in out.columns


def test_a_misspelled_weather_column_is_rejected():
    with pytest.raises(ValueError, match="not in the hourly table"):
        join_weather(_dep(), _full_hourly(), columns=("deicing_gate", "temp_celsius"))
