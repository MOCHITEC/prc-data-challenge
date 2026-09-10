"""Open weather data for the 10 competition airports (docs/04 section 4-6, docs/03 section 9).

Primary source: Iowa Environmental Mesonet (IEM) ASOS/METAR archive - public domain, hourly METAR with
visibility and present-weather codes, which the reanalysis products do not offer. The service asks for at most
one request per second per IP, so fetches are chunked by station and year and cached on disk.

Secondary source (gap filling / snowfall): Open-Meteo historical API, CC BY 4.0. Attribution and licence notes
belong in docs/external_data.md.
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
    the January mechanism reported in docs/03 section 8 (dry vs wet at sub-zero changes taxi-out by 400-1000 s)."""
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


def _align_tz(col: pl.Expr, like: pl.DataType) -> pl.Expr:
    """Make a datetime column's time zone match `like`.

    The METAR archive is parsed as naive UTC while prc.features produces timezone-aware UTC, and polars refuses
    to join a datetime[us] key against a datetime[us, UTC] one. Both sides mean UTC, so this only relabels."""
    tz = getattr(like, "time_zone", None)
    return col.dt.replace_time_zone(tz) if tz else col.dt.replace_time_zone(None)


# The columns with an operational mechanism behind them, and the only ones carried by default.
#
# Measured on v11, which carried all twenty (ten fields plus a previous-hour copy of each): decisively better in
# January (-3.79s, P = 1.000) and probably worse in July (+2.45s, P better = 0.195), for nothing net on the two
# scored months. De-icing never fires in July, so every winter column is a noise dimension there, and twenty of
# them across 2M rows buy enough spurious splits to cancel the January gain.
#
# Kept: the de-icing gate and its inputs, and visibility, which triggers low-visibility procedures year-round.
# Dropped: gust_ms (97.7% null - only reported when present), dewp_spread_c and temp_x_precip (collinear with
# the gate's inputs), frozen_wx (already inside the gate). Only the de-icing gate keeps a previous-hour copy,
# because a de-icing queue persists past the weather that caused it, which is not true of visibility.
MECHANISM_COLUMNS = ("deicing_gate", "temp_c", "precip_mm", "vis_km", "low_vis", "wind_ms")
MECHANISM_PREV1H = ("deicing_gate",)


def join_weather(
    dep_feats: pl.DataFrame,
    hourly: pl.DataFrame,
    columns: tuple[str, ...] | None = MECHANISM_COLUMNS,
    prev1h: tuple[str, ...] | None = MECHANISM_PREV1H,
) -> pl.DataFrame:
    """Attach the weather hour of the take-off time (and selected previous-hour fields) to departure features.

    `columns=None` keeps every field, which is what v11 did; the default is MECHANISM_COLUMNS."""
    dep_dtype = dep_feats.schema["MVT_TIME_UTC_mvt"]
    w = hourly.rename({"station": "ADEP_mvt"}).with_columns(_align_tz(pl.col("hour_utc"), dep_dtype))
    if columns is not None:
        missing = [c for c in columns if c not in w.columns]
        if missing:
            raise ValueError(f"requested weather columns not in the hourly table: {missing}")
        w = w.select(["ADEP_mvt", "hour_utc", *columns])
    prev_cols = [c for c in w.columns if c not in ("ADEP_mvt", "hour_utc")]
    if prev1h is not None:
        prev_cols = [c for c in prev_cols if c in prev1h]
    prev = (
        w.select(["ADEP_mvt", "hour_utc", *prev_cols])
        .with_columns((pl.col("hour_utc") + pl.duration(hours=1)).alias("hour_utc"))
        .rename({c: f"{c}_prev1h" for c in prev_cols})
    )
    out = dep_feats.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.truncate("1h").alias("hour_utc"))
    out = out.join(w, on=["ADEP_mvt", "hour_utc"], how="left", maintain_order="left").join(
        prev, on=["ADEP_mvt", "hour_utc"], how="left", maintain_order="left"
    )
    # hourly_weather_table groups by station-hour so the right side is unique today, but a left join that
    # silently multiplied departures would be invisible downstream: every model here keys on MVT_ID_mvt, so
    # duplicated rows would be double-counted rather than rejected.
    if out.height != dep_feats.height:
        raise AssertionError(
            f"the weather join changed the row count {dep_feats.height} -> {out.height}; "
            "the hourly table must have one row per station-hour"
        )
    return out.drop("hour_utc")
