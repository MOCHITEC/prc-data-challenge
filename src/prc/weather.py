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


def hourly_weather_table(metar: pl.DataFrame) -> pl.DataFrame:
    """One row per station-hour with the fields the model uses, including a de-icing gate.

    De-icing gate: precipitation or a frozen-precipitation code while the temperature is at or below 3 C -
    the January mechanism reported in docs/README (dry vs wet at sub-zero changes taxi-out by 400-1000 s)."""
    frozen = pl.fold(pl.lit(False), lambda a, b: a | b, [pl.col("wx").str.contains(c) for c in SNOW_CODES])
    return (
        metar.with_columns(
            [
                pl.col("obs_time_utc").dt.truncate("1h").alias("hour_utc"),
                (pl.col("temp_c") - pl.col("dewp_c")).alias("dewp_spread_c"),
                frozen.alias("frozen_wx"),
            ]
        )
        .group_by(["station", "hour_utc"])
        .agg(
            [
                pl.col("temp_c").mean().alias("temp_c"),
                pl.col("dewp_spread_c").mean().alias("dewp_spread_c"),
                pl.col("wind_ms").max().alias("wind_ms"),
                pl.col("gust_ms").max().alias("gust_ms"),
                pl.col("precip_mm").sum().alias("precip_mm"),
                pl.col("vis_km").min().alias("vis_km"),
                pl.col("frozen_wx").any().alias("frozen_wx"),
            ]
        )
        .with_columns(
            [
                ((pl.col("temp_c") <= 3.0) & ((pl.col("precip_mm") > 0) | pl.col("frozen_wx")))
                .cast(pl.Int8)
                .alias("deicing_gate"),
                (pl.col("vis_km") < 1.5).cast(pl.Int8).alias("low_vis"),
                (pl.col("temp_c") * ((pl.col("precip_mm") > 0).cast(pl.Float64))).alias("temp_x_precip"),
            ]
        )
        .sort(["station", "hour_utc"])
    )


# ---- weather v2 (docs/weather_v2_2026-09-25/plan.md): candidate columns, joined inside the fold ----
#
# IEM serves no precipitation amount for non-US stations ("precipitation data is unavailable for non-US
# sites", IEM download page), so `precip_mm` above is zero on every station-hour and the precipitation term of
# `deicing_gate` never fires. The present-weather codes are served, and they say whether it is raining.
WET_CODES = ("RA", "DZ")  # rain, drizzle; substring match also catches -RA, +RA, SHRA, TSRA, FZRA, RADZ
SHOWER_CODES = ("SH", "TS")  # showers, thunderstorms
OBSCURE_CODES = ("FG", "BR", "HZ", "FU")  # fog, mist, haze, smoke
WX_V2_COLS = ["wxv2_wet", "wxv2_shower", "wxv2_obscure", "wxv2_frozen", "wxv2_deicing_gate"]

OPENMETEO_URL = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_LICENCE = "Open-Meteo historical weather API, data CC BY 4.0 (https://open-meteo.com/en/license)"
OPENMETEO_FIELDS = ("precipitation", "snowfall", "snow_depth")
# ECMWF IFS 9 km: covers both scored months, where ERA5-Land returns nulls for January 2026 (P8 §4.2). Pinned so
# the cache is reproducible rather than whatever `best_match` resolves to on the day.
OPENMETEO_MODEL = "ecmwf_ifs"
# om_snow_depth_m is fetched but NOT given to the model: it is null on most of 2025 (the IFS field starts
# being served part-way through the year, last null 2025-10-03) and populated on all of 2026, which is the
# dense-in-one-period / empty-in-the-other shape the coverage rule excludes (plan.md §3, §5).
OM_COLS = ["om_precip_mm", "om_snowfall_cm"]


def _any_code(codes: tuple[str, ...]) -> pl.Expr:
    return pl.fold(pl.lit(False), lambda a, b: a | b, [pl.col("wx").str.contains(c, literal=True) for c in codes])


def wx_code_table(metar: pl.DataFrame) -> pl.DataFrame:
    """One row per station-hour: which kinds of weather any METAR of that hour reported, as Int8."""
    hourly = (
        metar.with_columns(pl.col("obs_time_utc").dt.truncate("1h").alias("hour_utc"))
        .group_by(["station", "hour_utc"])
        .agg(
            pl.col("temp_c").mean().alias("_temp_c"),
            _any_code(WET_CODES).any().alias("wxv2_wet"),
            _any_code(SHOWER_CODES).any().alias("wxv2_shower"),
            _any_code(OBSCURE_CODES).any().alias("wxv2_obscure"),
            _any_code(SNOW_CODES).any().alias("wxv2_frozen"),
        )
    )
    return (
        hourly.with_columns(
            ((pl.col("_temp_c") <= 3.0) & (pl.col("wxv2_frozen") | pl.col("wxv2_wet"))).alias("wxv2_deicing_gate")
        )
        .with_columns([pl.col(c).cast(pl.Int8) for c in WX_V2_COLS])
        .select(["station", "hour_utc", *WX_V2_COLS])
        .sort(["station", "hour_utc"])
    )


def fetch_openmeteo(
    stations: list[str] | None = None,
    start: str = "2025-01-01",
    end: str = "2026-07-31",
    cache_dir: Path | None = None,
    refresh: bool = False,
    retries: int = 5,
) -> pl.DataFrame:
    """Hourly precipitation, snowfall and snow depth at each airport's coordinates, one cached request per station.

    The free tier's daily call budget is per IP and shared, so every station is fetched once and cached; a
    response carrying `"error"` raises instead of being written, because a partially-null column that looks
    like data is the failure mode the coverage gate exists for."""
    import json

    from .airports import airport_coords

    stations = stations or TARGET_AIRPORTS
    cache_dir = cache_dir or (data_dir() / "ref" / "openmeteo")
    cache_dir.mkdir(parents=True, exist_ok=True)
    coords = airport_coords().filter(pl.col("icao").is_in(stations))
    where = {r["icao"]: (r["lat"], r["lon"]) for r in coords.iter_rows(named=True)}
    frames = []
    for st in stations:
        cache = cache_dir / f"{st}_{OPENMETEO_MODEL}_{start}_{end}.json"
        if refresh or not cache.exists() or cache.stat().st_size < 200:
            if st not in where:
                raise KeyError(f"no coordinates for {st}")
            lat, lon = where[st]
            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": start,
                "end_date": end,
                "hourly": ",".join(OPENMETEO_FIELDS),
                "models": OPENMETEO_MODEL,
                "timezone": "UTC",
            }
            for attempt in range(retries):
                try:
                    r = requests.get(OPENMETEO_URL, params=params, timeout=120)
                    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                    status, reason = r.status_code, body.get("reason", "")
                except requests.exceptions.RequestException as e:  # a dropped connection mid-transfer
                    body, status, reason = {}, None, repr(e)
                if status == 200 and not body.get("error") and "hourly" in body:
                    cache.write_text(json.dumps(body))
                    break
                if attempt == retries - 1:
                    raise RuntimeError(f"Open-Meteo {st}: HTTP {status} {reason}")
                time.sleep(20 * (attempt + 1))
            time.sleep(REQUEST_SLEEP_S)
        frames.append(parse_openmeteo(json.loads(cache.read_text()), st))
    return pl.concat(frames) if frames else pl.DataFrame()


def parse_openmeteo(body: dict, station: str) -> pl.DataFrame:
    """Open-Meteo's hourly block as station-hour rows. Its `time` is naive UTC (we ask for timezone=UTC)."""
    h = body["hourly"]
    return pl.DataFrame(
        {
            "station": [station] * len(h["time"]),
            "hour_utc": pl.Series(h["time"]).str.to_datetime("%Y-%m-%dT%H:%M"),
            "om_precip_mm": pl.Series(h["precipitation"], dtype=pl.Float64),
            "om_snowfall_cm": pl.Series(h["snowfall"], dtype=pl.Float64),
            "om_snow_depth_m": pl.Series(h["snow_depth"], dtype=pl.Float64),
        }
    )


def join_hourly(dep: pl.DataFrame, table: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Attach `cols` of a station-hour table at the take-off hour. Refuses to change the row count or order."""
    key = "_wxv2_hour"
    tz = dep.schema["MVT_TIME_UTC_mvt"].time_zone
    t = table.rename({"station": "ADEP_mvt", "hour_utc": key}).select(["ADEP_mvt", key, *cols])
    t_tz = t.schema[key].time_zone
    if tz != t_tz:
        t = t.with_columns(
            pl.col(key).dt.replace_time_zone(t_tz or "UTC").dt.convert_time_zone(tz)
            if tz
            else pl.col(key).dt.replace_time_zone(None)
        )
    if t.select(pl.struct(["ADEP_mvt", key]).is_duplicated().any()).item():
        raise AssertionError("station-hour table has duplicate keys; the join would multiply rows")
    out = (
        dep.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.truncate("1h").alias(key))
        .join(t, on=["ADEP_mvt", key], how="left", maintain_order="left")
        .drop(key)
    )
    if out.height != dep.height or not out["MVT_ID_mvt"].equals(dep["MVT_ID_mvt"]):
        raise AssertionError("join_hourly changed the row count or order")
    return out


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
