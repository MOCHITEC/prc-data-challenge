"""Weather v2 candidate columns (docs/weather_v2_2026-09-25/plan.md)."""

from datetime import datetime

import numpy as np
import polars as pl
import pytest

from prc import train
from prc.weather import OM_COLS, WX_V2_COLS, join_hourly, parse_openmeteo, wx_code_table


def _metar(rows):
    return pl.DataFrame(
        rows,
        schema={"station": pl.Utf8, "obs_time_utc": pl.Datetime("us"), "temp_c": pl.Float64, "wx": pl.Utf8},
        orient="row",
    )


def test_present_weather_codes_map_to_the_right_columns_and_gate():
    m = _metar(
        [
            ("EDDF", datetime(2025, 1, 1, 0, 20), 1.0, "-RA"),  # cold rain: wet, gate
            ("EDDF", datetime(2025, 1, 1, 0, 50), 1.5, ""),
            ("EDDF", datetime(2025, 1, 1, 1, 50), 8.0, "SHRA"),  # warm shower: wet + shower, no gate
            ("EDDF", datetime(2025, 1, 1, 2, 50), -2.0, "BR"),  # mist at sub-zero: obscure, no precipitation, no gate
            ("EDDF", datetime(2025, 1, 1, 3, 50), -1.0, "-SN"),  # snow: frozen, gate
        ]
    )
    t = wx_code_table(m).sort("hour_utc")
    assert t.height == 4 and all(t[c].dtype == pl.Int8 for c in WX_V2_COLS)
    assert t["wxv2_wet"].to_list() == [1, 1, 0, 0]
    assert t["wxv2_shower"].to_list() == [0, 1, 0, 0]
    assert t["wxv2_obscure"].to_list() == [0, 0, 1, 0]
    assert t["wxv2_frozen"].to_list() == [0, 0, 0, 1]
    assert t["wxv2_deicing_gate"].to_list() == [1, 0, 0, 1]


def test_open_meteo_block_parses_to_station_hours():
    body = {
        "hourly": {
            "time": ["2026-01-05T00:00", "2026-01-05T01:00"],
            "precipitation": [0.0, 0.4],
            "snowfall": [0.0, 0.28],
            "snow_depth": [None, 0.03],
        }
    }
    t = parse_openmeteo(body, "EDDF")
    assert t["hour_utc"].to_list() == [datetime(2026, 1, 5, 0), datetime(2026, 1, 5, 1)]
    assert t["om_snowfall_cm"].to_list() == [0.0, 0.28] and t["om_snow_depth_m"].null_count() == 1
    assert "om_snow_depth_m" not in OM_COLS  # fetched, never given to the model (coverage rule)


def test_the_join_uses_the_take_off_hour_across_time_zones_and_never_multiplies_rows():
    dep = pl.DataFrame(
        {
            "MVT_ID_mvt": [1, 2, 3],
            "ADEP_mvt": ["EDDF", "EDDF", "LIRF"],
            "MVT_TIME_UTC_mvt": pl.Series(
                [datetime(2025, 1, 1, 0, 59), datetime(2025, 1, 1, 1, 0), datetime(2025, 1, 1, 0, 30)]
            ).dt.replace_time_zone("UTC"),
        }
    )
    table = pl.DataFrame(
        {
            "station": ["EDDF", "EDDF"],
            "hour_utc": [datetime(2025, 1, 1, 0), datetime(2025, 1, 1, 1)],
            "x": [10, 20],
        }
    )
    out = join_hourly(dep, table, ["x"])
    assert out["x"].to_list() == [10, 20, None]
    with pytest.raises(AssertionError):
        join_hourly(dep, pl.concat([table, table]), ["x"])


def test_weather_v2_features_add_exactly_the_mode_columns(monkeypatch):
    table = pl.DataFrame(
        {
            "station": ["EDDF"],
            "hour_utc": [datetime(2025, 1, 1, 0)],
            **{c: pl.Series([1], dtype=pl.Int8) for c in WX_V2_COLS},
            "om_precip_mm": [0.5],
            "om_snowfall_cm": [0.1],
        }
    )
    monkeypatch.setattr(train, "_weather_v2_table", lambda mode: table)
    dep = pl.DataFrame(
        {
            "MVT_ID_mvt": [1],
            "ADEP_mvt": ["EDDF"],
            "MVT_TIME_UTC_mvt": pl.Series([datetime(2025, 1, 1, 0, 30)]).dt.replace_time_zone("UTC"),
        }
    )
    b = train.weather_v2_features(dep, "B")
    c = train.weather_v2_features(dep, "C")
    assert set(b.columns) - set(dep.columns) == set(WX_V2_COLS)
    assert set(c.columns) - set(dep.columns) == set(WX_V2_COLS) | set(OM_COLS)
    assert np.isclose(c["om_snowfall_cm"][0], 0.1)


def test_weather_v2_is_off_by_default():
    assert train.TrainConfig().fe_weather_v2 == ""
