from datetime import datetime
from pathlib import Path

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
        schema_overrides={"MVT_TIME_UTC_mvt": pl.Datetime("us")},
    )
    j = join_weather(dep, hourly)
    r = j.filter(pl.col("MVT_ID_mvt") == 1).row(0, named=True)
    assert r["deicing_gate"] == 0  # 06:25 obs is above 3 C
    assert r["deicing_gate_prev1h"] == 1  # the 05:xx hour was freezing with snow


def test_the_join_keeps_the_hour_of_day_feature_and_survives_a_tz_aware_frame():
    """Two bugs in one: the join key used to be called hour_utc, which is also a feature, so weather runs
    silently trained without it - and the official files are UTC-aware while METAR parses naive, which polars
    refuses to join."""
    import datetime as dt

    from prc.weather import join_weather

    hourly = pl.DataFrame(
        {
            "station": ["EDDF", "EDDF"],
            "hour_utc": [dt.datetime(2025, 1, 1, 10), dt.datetime(2025, 1, 1, 11)],
            "tmpc": [1.0, 2.0],
        }
    )
    deps = pl.DataFrame(
        {
            "ADEP_mvt": ["EDDF", "EDDF"],
            "MVT_TIME_UTC_mvt": [dt.datetime(2025, 1, 1, 11, 5), dt.datetime(2025, 1, 1, 10, 30)],
            "hour_utc": [11, 10],  # the integer feature that must survive
        }
    ).with_columns(pl.col("MVT_TIME_UTC_mvt").dt.replace_time_zone("UTC"))

    out = join_weather(deps, hourly)
    assert out["hour_utc"].to_list() == [11, 10], "the hour-of-day feature was overwritten by the join key"
    assert out["tmpc"].to_list() == [2.0, 1.0], "weather did not land on the right hour"
    assert out["tmpc_prev1h"].to_list() == [1.0, None]


# ---------- P8-B: the wxcodes repair (docs/competitor_research_2026-09-21/results/P8-B-wxcodes-repair.md) ----------

REPAIR_SAMPLE = """station,valid,tmpf,dwpf,relh,drct,sknt,gust,p01i,vsby,wxcodes,skyc1,feel
EDDF,2026-01-10 08:50,30.20,28.40,92.85,50.00,12.00,M,0.00,5.00,-RA,OVC,21.60
EDDF,2026-01-11 08:50,30.20,28.40,92.85,50.00,12.00,M,0.00,5.00,-SN,OVC,21.60
EDDF,2026-07-10 08:50,77.00,60.80,58.00,50.00,12.00,M,0.00,5.00,-SHRA,BKN,80.00
EDDF,2026-07-11 08:50,77.00,60.80,58.00,50.00,12.00,M,0.00,5.00,M,SKC,80.00
EDDF,2026-07-12 08:50,68.00,60.80,78.00,50.00,12.00,M,0.00,0.50,BCFG,OVC,70.00
"""


def _repaired():
    return hourly_weather_table(parse_metar_csv(REPAIR_SAMPLE), wx_codes=True)


def test_wx_codes_defaults_to_the_shipped_columns():
    """The flag defaults to current behaviour. Without it the table is byte-for-byte what it always was -
    dead columns and all - so every archived recipe rebuilds to the frame it was measured on."""
    shipped = hourly_weather_table(parse_metar_csv(REPAIR_SAMPLE))
    assert "precip_mm" in shipped.columns and "temp_x_precip" in shipped.columns
    assert not {"wet_wx", "shower_wx", "obscuration_wx", "frozen_wx_i8"} & set(shipped.columns)
    assert hourly_weather_table(parse_metar_csv(SAMPLE)).equals(hourly_weather_table(parse_metar_csv(SAMPLE)))


def test_the_repair_swaps_the_dead_columns_for_live_ones():
    r = _repaired()
    assert "precip_mm" not in r.columns and "temp_x_precip" not in r.columns
    assert {"wet_wx", "shower_wx", "obscuration_wx", "frozen_wx_i8"} <= set(r.columns)
    for c in ("wet_wx", "shower_wx", "obscuration_wx", "frozen_wx_i8", "deicing_gate", "low_vis"):
        assert r.schema[c] == pl.Int8, c


@pytest.mark.parametrize(
    "code,wet,shower,obscure,frozen",
    [
        ("-RA", 1, 0, 0, 0),  # the single commonest code in the cache, 4,620 observations
        ("-SHRA", 1, 1, 0, 0),  # composes: a light rain shower is both
        ("-DZRA", 1, 0, 0, 0),
        ("+SHRA", 1, 1, 0, 0),
        ("-TSRA", 1, 1, 0, 0),
        ("VCSH", 0, 1, 0, 0),
        ("BR", 0, 0, 1, 0),
        ("BCFG", 0, 0, 1, 0),
        ("FZFG", 0, 0, 1, 1),  # freezing fog: obscuration AND frozen, via FZ
        ("-SN", 0, 0, 0, 1),
        ("-SHSN", 0, 1, 0, 1),
        ("", 0, 0, 0, 0),  # control: no present weather reported
        ("SKC", 0, 0, 0, 0),  # control: a sky code is not a present-weather code
    ],
)
def test_present_weather_codes_map_to_the_right_family(code, wet, shower, obscure, frozen):
    """The codes are the literal strings IEM serves in this cache, taken from the census of all 138,457
    observations. A substring match is the right matcher because METAR codes concatenate."""
    txt = (
        "station,valid,tmpf,dwpf,relh,drct,sknt,gust,p01i,vsby,wxcodes,skyc1,feel\n"
        f"EDDF,2026-01-10 08:50,41.00,32.00,70.00,50.00,12.00,M,0.00,5.00,{code or 'M'},OVC,40.00\n"
    )
    r = hourly_weather_table(parse_metar_csv(txt), wx_codes=True).row(0, named=True)
    assert (r["wet_wx"], r["shower_wx"], r["obscuration_wx"], r["frozen_wx_i8"]) == (wet, shower, obscure, frozen)


def test_the_repaired_gate_fires_on_freezing_rain_where_the_old_one_could_not():
    """The whole mechanism in one assertion. -1 C with `-RA` reported: the old gate needs `precip_mm > 0`,
    which IEM never serves, or a frozen code, which `-RA` is not - so it is 0. The repaired gate reads the
    observation and is 1. The control is the same code in July, where neither gate may fire."""
    old = hourly_weather_table(parse_metar_csv(REPAIR_SAMPLE)).sort("hour_utc")
    new = _repaired().sort("hour_utc")
    assert old["deicing_gate"].to_list() == [0, 1, 0, 0, 0]  # only the -SN hour
    assert new["deicing_gate"].to_list() == [1, 1, 0, 0, 0]  # the -RA hour as well
    jul = new.filter(pl.col("hour_utc").dt.month() == 7)
    assert jul["deicing_gate"].to_list() == [0, 0, 0]
    assert jul["wet_wx"].to_list() == [1, 0, 0]  # ... but the July rain hour is now visible, and was not


def test_the_repaired_columns_reach_the_model_and_the_dead_ones_no_longer_do():
    """`frozen_wx` is Boolean and `numeric_feature_columns` admits only float/int, so it has never reached
    the model. Four of the eighteen columns that did reach it were constants. After the repair the model is
    given 22 weather columns, none of them constant by construction."""
    import datetime as dt

    from prc.features import numeric_feature_columns

    dep = pl.DataFrame(
        {
            "MVT_ID_mvt": [1],
            "ADEP_mvt": ["EDDF"],
            "MVT_TIME_UTC_mvt": [dt.datetime(2026, 1, 10, 8, 30)],
        },
        schema_overrides={"MVT_TIME_UTC_mvt": pl.Datetime("us")},
    )
    before = set(numeric_feature_columns(join_weather(dep, hourly_weather_table(parse_metar_csv(REPAIR_SAMPLE)))))
    after = set(numeric_feature_columns(join_weather(dep, _repaired())))
    weather_before = before - {"MVT_ID_mvt"}
    weather_after = after - {"MVT_ID_mvt"}
    assert len(weather_before) == 18 and len(weather_after) == 22
    assert {"precip_mm", "precip_mm_prev1h", "temp_x_precip", "temp_x_precip_prev1h"} <= weather_before
    assert not {"precip_mm", "precip_mm_prev1h", "temp_x_precip", "temp_x_precip_prev1h"} & weather_after
    assert {"wet_wx", "wet_wx_prev1h", "frozen_wx_i8", "frozen_wx_i8_prev1h"} <= weather_after
    assert "frozen_wx" not in weather_after and "frozen_wx" not in weather_before  # Boolean, never visible


METAR_CACHE = Path("data/ref/metar")


@pytest.mark.skipif(not list(METAR_CACHE.glob("*.csv")), reason="the IEM cache is not on disk")
def test_acceptance_on_the_real_cache():
    """P8-B's acceptance test on all 138,457 cached observations at the ten airports.

    Three properties, each of which would have caught a plausible way of getting this wrong:
      * the repaired gate is a strict superset of the old one - the repair widens, it does not redefine;
      * `precip_mm` really is constant zero, so nothing was lost by dropping it;
      * the new columns are live in July, where every column the old gate depended on is flat."""
    raw = pl.concat([parse_metar_csv(p.read_text()) for p in sorted(METAR_CACHE.glob("*.csv"))], how="diagonal")
    assert raw.height > 100_000 and (raw["precip_mm"] > 0).sum() == 0
    old = hourly_weather_table(raw).sort(["station", "hour_utc"])
    shipped = Path("data/ref/metar_hourly.parquet")
    if shipped.exists():
        # The refactor that added the flag must not have moved the default path by a single value. This
        # compares against the parquet every archived weather experiment was actually built from.
        assert old.equals(pl.read_parquet(shipped).sort(["station", "hour_utc"]))
    new = hourly_weather_table(raw, wx_codes=True).sort(["station", "hour_utc"])
    assert old.height == new.height
    j = old.select(["station", "hour_utc", "deicing_gate"]).join(
        new.select(["station", "hour_utc", pl.col("deicing_gate").alias("new_gate"), "wet_wx"]),
        on=["station", "hour_utc"],
    )
    assert j.filter((pl.col("deicing_gate") == 1) & (pl.col("new_gate") != 1)).height == 0
    assert j.filter(pl.col("new_gate") == 1).height > j.filter(pl.col("deicing_gate") == 1).height
    jul26 = new.filter((pl.col("hour_utc").dt.year() == 2026) & (pl.col("hour_utc").dt.month() == 7))
    assert jul26["deicing_gate"].sum() == 0 and jul26["low_vis"].sum() == 0  # the old apparatus, flat
    assert jul26["wet_wx"].sum() > 0 and jul26["shower_wx"].sum() > 0  # the repaired channel, live
