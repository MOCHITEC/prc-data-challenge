"""Open weather data for the 10 competition airports (docs/README).

Primary source: Iowa Environmental Mesonet (IEM) ASOS/METAR archive - public domain, hourly METAR with
visibility and present-weather codes, which the reanalysis products do not offer. The service asks for at most
one request per second per IP, so fetches are chunked by station and year and cached on disk.

Secondary source (gap filling / snowfall): Open-Meteo historical API, CC BY 4.0. Attribution and licence notes
belong in docs/README.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import polars as pl
import requests

from .config import TARGET_AIRPORTS, data_dir

IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
IEM_FIELDS = ["tmpf", "dwpf", "relh", "drct", "sknt", "gust", "p01i", "vsby", "wxcodes", "skyc1", "feel"]
IEM_LICENCE = (
    "Iowa Environmental Mesonet ASOS archive, public domain (https://mesonet.agron.iastate.edu/request/download.phtml)"
)
REQUEST_SLEEP_S = 1.2  # IEM throttles to 1 request/s per IP


def _iem_request(station: str, start: str, end: str, timeout: int = 120) -> str:
    params = {
        "station": station,
        "data": ",".join(IEM_FIELDS),
        "tz": "UTC",
        "format": "onlycomma",
        "latlon": "no",
        "elev": "no",
        "missing": "M",
        "trace": "T",
        "direct": "no",
        "report_type": "3",
        "year1": start[:4],
        "month1": start[5:7],
        "day1": start[8:10],
        "year2": end[:4],
        "month2": end[5:7],
        "day2": end[8:10],
    }
    r = requests.get(IEM_URL, params=params, timeout=timeout)
    r.raise_for_status()
    return r.text


def fetch_metar(
    stations: list[str] | None = None,
    start: str = "2025-01-01",
    end: str = "2026-08-01",
    cache_dir: Path | None = None,
    refresh: bool = False,
) -> pl.DataFrame:
    """Fetch (and cache) hourly METAR rows for the given ICAO stations in [start, end)."""
    stations = stations or TARGET_AIRPORTS
    cache_dir = cache_dir or (data_dir() / "ref" / "metar")
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for st in stations:
        cache = cache_dir / f"{st}_{start}_{end}.csv"
        if refresh or not cache.exists() or cache.stat().st_size < 200:
            cache.write_text(_iem_request(st, start, end))
            time.sleep(REQUEST_SLEEP_S)
        frames.append(parse_metar_csv(cache.read_text()))
    return pl.concat([f for f in frames if f.height], how="diagonal") if frames else pl.DataFrame()


def parse_metar_csv(text: str) -> pl.DataFrame:
    """Parse the IEM 'onlycomma' response into typed columns (missing 'M', trace 'T')."""
    if not text.strip() or text.lstrip().startswith("#"):
        text = "\n".join(ln for ln in text.splitlines() if not ln.startswith("#"))
    if not text.strip():
        return pl.DataFrame()
    df = pl.read_csv(io.StringIO(text), infer_schema_length=0, null_values=["M", "", "null"])
    if "valid" not in df.columns:
        return pl.DataFrame()
    numeric = [c for c in ("tmpf", "dwpf", "relh", "drct", "sknt", "gust", "p01i", "vsby", "feel") if c in df.columns]
    df = df.with_columns(
        [
            pl.col("valid").str.to_datetime("%Y-%m-%d %H:%M", strict=False).alias("obs_time_utc"),
            *[pl.col(c).str.replace("T", "0.0001").cast(pl.Float64, strict=False).alias(c) for c in numeric],
        ]
    )
    return (
        df.with_columns(
            [
                ((pl.col("tmpf") - 32.0) * 5.0 / 9.0).alias("temp_c"),
                ((pl.col("dwpf") - 32.0) * 5.0 / 9.0).alias("dewp_c"),
                (pl.col("p01i") * 25.4).alias("precip_mm"),
                (pl.col("vsby") * 1.609344).alias("vis_km"),
                (pl.col("sknt") * 0.514444).alias("wind_ms"),
                (pl.col("gust") * 0.514444).alias("gust_ms"),
                pl.col("wxcodes").fill_null("").alias("wx"),
            ]
        )
        .select(
            [
                "station",
                "obs_time_utc",
                "temp_c",
                "dewp_c",
                "relh",
                "drct",
                "wind_ms",
                "gust_ms",
                "precip_mm",
                "vis_km",
                "wx",
            ]
        )
        .drop_nulls("obs_time_utc")
        .sort(["station", "obs_time_utc"])
    )


SNOW_CODES = ("SN", "SG", "PL", "IC", "GS", "GR", "FZ")
# Present-weather families the same feed already serves and this module has always discarded (P8-B).
# Matched as substrings of the space-separated code string, which is how METAR codes compose: "-SHRA" is a
# light rain shower and contains both "SH" and "RA"; "-DZRA" is drizzle with rain and contains both "DZ" and
# "RA"; "FZRA" is freezing rain and is both frozen and wet, which is correct.
WET_CODES = ("RA", "DZ")  # rain, drizzle
SHOWER_CODES = ("SH", "TS")  # showers, thunderstorms - convective, and in July the only live channel
OBSCURATION_CODES = ("FG", "BR", "HZ", "FU")  # fog, mist, haze, smoke - pairs with the existing vis_km


def _any_code(codes: tuple[str, ...]) -> pl.Expr:
    return pl.fold(pl.lit(False), lambda a, b: a | b, [pl.col("wx").str.contains(c, literal=True) for c in codes])


def hourly_weather_table(metar: pl.DataFrame, wx_codes: bool = False) -> pl.DataFrame:
    """One row per station-hour with the fields the model uses, including a de-icing gate.

    De-icing gate: precipitation or a frozen-precipitation code while the temperature is at or below 3 C -
    the January mechanism reported in docs/README (dry vs wet at sub-zero changes taxi-out by 400-1000 s).

    `wx_codes=False` is the shipped behaviour and stays the default.

    `wx_codes=True` repairs four identically-zero columns (P8-B). IEM serves the literal string "0.00" for
    `p01i` on 100% of 138,457 cached observations at all ten stations, so `precip_mm` is non-null and
    constant zero, `temp_x_precip` is its product and equally constant, and after the join their `_prev1h`
    copies make four dead columns out of the eighteen the model is given. Two consequences follow, and the
    second is the one that matters: `precip_mm > 0` never fires, so `deicing_gate` reduces exactly to
    `frozen_wx AND temp_c <= 3`, and the entire liquid-precipitation half of the gate has never existed.

    The information is not missing - it is in `wxcodes`, which this module already parses and then throws
    away everywhere except `SNOW_CODES`. Rain-family codes are live on 9.14% of the scored rows against the
    gate's 1.95%, and on 5,535 rows of July 2026, where `deicing_gate`, `low_vis` and `precip_mm` are zero on
    every row at every airport.

    Repaired, the table drops `precip_mm` and `temp_x_precip` and gains `wet_wx`, `shower_wx`,
    `obscuration_wx` and `frozen_wx_i8` - the last of which recovers a column that exists today but never
    reaches the model, because `frozen_wx` is Boolean and `numeric_feature_columns` admits only float and int
    dtypes. Net effect on what the model sees: 18 columns become 22, four of which were constants.
    """
    flags = [
        pl.col("obs_time_utc").dt.truncate("1h").alias("hour_utc"),
        (pl.col("temp_c") - pl.col("dewp_c")).alias("dewp_spread_c"),
        _any_code(SNOW_CODES).alias("frozen_wx"),
    ]
    aggs = [
        pl.col("temp_c").mean().alias("temp_c"),
        pl.col("dewp_spread_c").mean().alias("dewp_spread_c"),
        pl.col("wind_ms").max().alias("wind_ms"),
        pl.col("gust_ms").max().alias("gust_ms"),
        pl.col("precip_mm").sum().alias("precip_mm"),
        pl.col("vis_km").min().alias("vis_km"),
        pl.col("frozen_wx").any().alias("frozen_wx"),
    ]
    code_cols = ("wet_wx", "shower_wx", "obscuration_wx")
    if wx_codes:
        flags += [
            _any_code(WET_CODES).alias("wet_wx"),
            _any_code(SHOWER_CODES).alias("shower_wx"),
            _any_code(OBSCURATION_CODES).alias("obscuration_wx"),
        ]
        # `any()` over the hour, matching how `frozen_wx` has always been aggregated: an hour in which any
        # observation reported rain is a rain hour.
        aggs += [pl.col(c).any().alias(c) for c in code_cols]
    hourly = metar.with_columns(flags).group_by(["station", "hour_utc"]).agg(aggs)
    if not wx_codes:
        return hourly.with_columns(
            [
                ((pl.col("temp_c") <= 3.0) & ((pl.col("precip_mm") > 0) | pl.col("frozen_wx")))
                .cast(pl.Int8)
                .alias("deicing_gate"),
                (pl.col("vis_km") < 1.5).cast(pl.Int8).alias("low_vis"),
                (pl.col("temp_c") * ((pl.col("precip_mm") > 0).cast(pl.Float64))).alias("temp_x_precip"),
            ]
        ).sort(["station", "hour_utc"])
    return (
        hourly.with_columns(
            [
                # The `precip_mm > 0` term is replaced by the observation that actually reports liquid
                # precipitation. On the cached frames this widens the gate rather than redefining it: every
                # hour the old gate fired on still fires, because `frozen_wx AND temp_c <= 3` is retained.
                ((pl.col("temp_c") <= 3.0) & (pl.col("frozen_wx") | pl.col("wet_wx")))
                .cast(pl.Int8)
                .alias("deicing_gate"),
                (pl.col("vis_km") < 1.5).cast(pl.Int8).alias("low_vis"),
                pl.col("frozen_wx").cast(pl.Int8).alias("frozen_wx_i8"),
                *[pl.col(c).cast(pl.Int8).alias(c) for c in code_cols],
            ]
        )
        .drop("precip_mm")
        .sort(["station", "hour_utc"])
    )


# The join key is named for what it is and never reuses a feature name. An earlier version called it
# "hour_utc", which is also the integer hour-of-day feature: turning weather on overwrote that column with a
# timestamp and then dropped it, so every weather run silently trained without hour_utc. v3 and v4 were built
# that way.
WX_KEY = "_wx_hour"


def _utc_hour(expr: pl.Expr, like: pl.DataFrame, col: str) -> pl.Expr:
    """Truncate to the hour, matching the time zone of `like[col]`.

    The official parquet files type their timestamps as UTC-aware while the METAR archive parses naive, and
    polars refuses to join across the two rather than guessing."""
    tz = like.schema[col].time_zone if col in like.schema else None
    e = expr.dt.truncate("1h")
    return e.dt.replace_time_zone(tz) if tz is None else e.dt.convert_time_zone(tz)


def join_weather(dep_feats: pl.DataFrame, hourly: pl.DataFrame) -> pl.DataFrame:
    """Attach the weather hour of the take-off time (and the previous hour) to departure features."""
    w = hourly.rename({"station": "ADEP_mvt", "hour_utc": WX_KEY})
    prev = w.with_columns((pl.col(WX_KEY) + pl.duration(hours=1)).alias(WX_KEY)).rename(
        {c: f"{c}_prev1h" for c in w.columns if c not in ("ADEP_mvt", WX_KEY)}
    )
    dep_tz = dep_feats.schema["MVT_TIME_UTC_mvt"].time_zone
    w_tz = w.schema[WX_KEY].time_zone
    if dep_tz != w_tz:  # bring the weather side onto the departures' time zone, never the other way round
        w = w.with_columns(
            pl.col(WX_KEY).dt.replace_time_zone(w_tz or "UTC").dt.convert_time_zone(dep_tz)
            if dep_tz
            else pl.col(WX_KEY).dt.replace_time_zone(None)
        )
        prev = prev.with_columns(
            pl.col(WX_KEY).dt.replace_time_zone(w_tz or "UTC").dt.convert_time_zone(dep_tz)
            if dep_tz
            else pl.col(WX_KEY).dt.replace_time_zone(None)
        )
    out = dep_feats.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.truncate("1h").alias(WX_KEY))
    out = out.join(w, on=["ADEP_mvt", WX_KEY], how="left").join(prev, on=["ADEP_mvt", WX_KEY], how="left")
    return out.drop(WX_KEY)
