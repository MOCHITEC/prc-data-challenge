"""Airport coordinates, for the one thing the competition files cannot say: how far the flight is going.

Source: OurAirports (https://ourairports.com/data/), public domain. Cached on disk after the first fetch.

Why this and not another engineered column: seven features built from what the data already contained changed
nothing, because the model is already past the ceiling of that space (docs/features_log). The single thing that
did move the leaderboard was weather, which is information the files do not contain. Destination distance is
the same shape of idea. ADES_mvt is a categorical with 1,568 values, so a tree can memorise the destinations it
has seen and can do nothing with the rest; a distance lets it interpolate, and stands in for the fuel load that
decides how heavy the aircraft is on the ground."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import polars as pl
import requests

from .config import data_dir

URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"
LICENCE = "OurAirports (https://ourairports.com/data/), public domain"
EARTH_KM = 6371.0088


def airport_coords(cache: Path | None = None, timeout: int = 120) -> pl.DataFrame:
    """ICAO code -> latitude/longitude. One row per code, cached as parquet."""
    cache = cache or data_dir() / "external" / "airports.parquet"
    if cache.exists():
        return pl.read_parquet(cache)
    text = requests.get(URL, timeout=timeout).text
    raw = pl.read_csv(io.StringIO(text), infer_schema_length=0)
    out = (
        raw.select(
            [
                pl.coalesce([pl.col("icao_code"), pl.col("gps_code"), pl.col("ident")]).alias("icao"),
                pl.col("latitude_deg").cast(pl.Float64).alias("lat"),
                pl.col("longitude_deg").cast(pl.Float64).alias("lon"),
            ]
        )
        .drop_nulls()
        .filter(pl.col("icao").str.len_chars() == 4)
        .unique(subset="icao", keep="first")
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(cache)
    return out


def great_circle_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Haversine. Returns NaN wherever any coordinate is missing."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def add_destination_distance(dep: pl.DataFrame, coords: pl.DataFrame | None = None) -> pl.DataFrame:
    """Attach dest_distance_km: great-circle distance from ADEP_mvt to ADES_mvt."""
    c = coords if coords is not None else airport_coords()
    d = dep.join(c.rename({"icao": "ADEP_mvt", "lat": "_a_lat", "lon": "_a_lon"}), on="ADEP_mvt", how="left")
    d = d.join(c.rename({"icao": "ADES_mvt", "lat": "_d_lat", "lon": "_d_lon"}), on="ADES_mvt", how="left")
    km = great_circle_km(d["_a_lat"].to_numpy(), d["_a_lon"].to_numpy(), d["_d_lat"].to_numpy(), d["_d_lon"].to_numpy())
    return d.drop(["_a_lat", "_a_lon", "_d_lat", "_d_lon"]).with_columns(
        pl.Series("dest_distance_km", km).fill_nan(None)
    )
