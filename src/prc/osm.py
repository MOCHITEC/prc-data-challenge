"""Stand and runway-threshold geometry, for a stand-to-runway distance that also covers stands never seen in 2025.

Sources, both open (docs/competitor_research_2026-09-21/results/16-external-data.md §6):
* OpenStreetMap `aeroway=parking_position` (stand `ref` and position) via the Overpass API. ODbL: open, with
  share-alike on derived databases, so the stand table built here is ODbL and is attributed as such.
* OurAirports `runways.csv` for runway-end coordinates. Public domain.

Everything is cached on disk under data/external/, one file per airport, so a feature build never touches the
network once the caches exist.
"""

from __future__ import annotations

import io
import json
import re
import time
from pathlib import Path

import numpy as np
import polars as pl
import requests

from .airports import great_circle_km
from .config import TARGET_AIRPORTS, data_dir

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OSM_LICENCE = "OpenStreetMap contributors, ODbL 1.0 (https://www.openstreetmap.org/copyright)"
RUNWAYS_URL = "https://davidmegginson.github.io/ourairports-data/runways.csv"
RUNWAYS_LICENCE = "OurAirports (https://ourairports.com/data/), public domain"
# Overpass answers 406 to a request without a User-Agent, so name the project.
USER_AGENT = "prc-data-challenge-2026 feature research (resilient-hedgehog)"


def fetch_osm_aeroway(stations: list[str] | None = None, cache_dir: Path | None = None, retries: int = 6) -> dict:
    """Overpass `parking_position` and `runway` elements per airport, cached as the raw JSON response.

    The public instance drops connections and answers 504 under load; every failure backs off and retries, and
    only a parseable response with an `elements` list is ever written, so an error page cannot be cached."""
    stations = stations or TARGET_AIRPORTS
    cache_dir = cache_dir or (data_dir() / "external" / "osm")
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for st in stations:
        cache = cache_dir / f"parking_{st}.json"
        body = None
        if cache.exists():
            try:
                body = json.loads(cache.read_text())
                if "elements" not in body:
                    body = None
            except json.JSONDecodeError:
                body = None
        if body is None:
            query = (
                f'[out:json][timeout:170];area["icao"="{st}"]->.a;'
                '(nwr["aeroway"="parking_position"](area.a);nwr["aeroway"="runway"](area.a););out center tags;'
            )
            for attempt in range(retries):
                try:
                    r = requests.post(
                        OVERPASS_URL,
                        data={"data": query},
                        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                        timeout=200,
                    )
                    if r.status_code == 200:
                        body = r.json()
                        if "elements" in body:
                            cache.write_text(json.dumps(body))
                            break
                    reason = f"HTTP {r.status_code}"
                except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
                    reason = repr(e)
                if attempt == retries - 1:
                    raise RuntimeError(f"Overpass {st}: {reason}")
                time.sleep(60 * (attempt + 1))
            time.sleep(5)
        out[st] = body
    return out


OSM_API_MAP_URL = "https://api.openstreetmap.org/api/0.6/map"


def fetch_osm_map_bbox(
    station: str,
    bbox: tuple[float, float, float, float],
    cache_dir: Path | None = None,
    tiles: int = 2,
    retries: int = 4,
) -> dict:
    """The same element list as `fetch_osm_aeroway`, from the main OSM API instead of Overpass.

    Overpass's public instance drops most connections from this environment (the agent proxy records the tunnel
    closing about 7 s in), while the main API's `map` call answers. `map` returns every object in a bounding box
    capped at 50,000 nodes, so the airport box (min_lon, min_lat, max_lon, max_lat) is split into tiles x tiles
    requests, and only `aeroway` parking positions and runways are kept. A way's position is the mean of its
    nodes, which is what Overpass's `out center` gives to within metres for a stand-sized polygon."""
    import xml.etree.ElementTree as ET

    cache_dir = cache_dir or (data_dir() / "external" / "osm")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"parking_{station}.json"
    if cache.exists():
        body = json.loads(cache.read_text())
        if "elements" in body:
            return body
    lon0, lat0, lon1, lat1 = bbox
    nodes: dict[int, tuple[float, float]] = {}
    keep: dict[tuple[str, int], dict] = {}
    for i in range(tiles):
        for j in range(tiles):
            tb = (
                lon0 + (lon1 - lon0) * i / tiles,
                lat0 + (lat1 - lat0) * j / tiles,
                lon0 + (lon1 - lon0) * (i + 1) / tiles,
                lat0 + (lat1 - lat0) * (j + 1) / tiles,
            )
            for attempt in range(retries):
                try:
                    r = requests.get(
                        OSM_API_MAP_URL,
                        params={"bbox": ",".join(f"{v:.5f}" for v in tb)},
                        headers={"User-Agent": USER_AGENT},
                        timeout=300,
                    )
                    if r.status_code == 200:
                        break
                    reason = f"HTTP {r.status_code} {r.text[:120]}"
                except requests.exceptions.RequestException as e:
                    reason = repr(e)
                if attempt == retries - 1:
                    raise RuntimeError(f"OSM map {station} tile {i},{j}: {reason}")
                time.sleep(30 * (attempt + 1))
            root = ET.fromstring(r.content)
            for n in root.iter("node"):
                nodes[int(n.get("id"))] = (float(n.get("lat")), float(n.get("lon")))
            for el in list(root.iter("node")) + list(root.iter("way")):
                tags = {t.get("k"): t.get("v") for t in el.findall("tag")}
                if tags.get("aeroway") not in ("parking_position", "runway"):
                    continue
                rec = {"type": el.tag, "id": int(el.get("id")), "tags": tags}
                if el.tag == "node":
                    rec["lat"], rec["lon"] = float(el.get("lat")), float(el.get("lon"))
                else:
                    rec["_nd"] = [int(nd.get("ref")) for nd in el.findall("nd")]
                keep[(el.tag, rec["id"])] = rec
            time.sleep(2)
    elements = []
    for rec in keep.values():
        if rec["type"] == "way":
            pts = [nodes[k] for k in rec.pop("_nd") if k in nodes]
            if not pts:
                continue
            rec["center"] = {"lat": float(np.mean([p[0] for p in pts])), "lon": float(np.mean([p[1] for p in pts]))}
        elements.append(rec)
    body = {"source": OSM_API_MAP_URL, "bbox": list(bbox), "elements": elements}
    cache.write_text(json.dumps(body))
    return body


def normalise_stand(ref: str | None) -> str | None:
    """Upper-case, drop spaces, hyphens and leading zeros of the numeric part: 'b 41' -> 'B41', '01' -> '1'."""
    if ref is None:
        return None
    s = re.sub(r"[\s\-_/]", "", str(ref).upper())
    s = re.sub(r"(?<![0-9])0+(?=[0-9])", "", s)
    return s or None


def stand_table(osm: dict) -> pl.DataFrame:
    """One row per (airport, normalised stand ref) with the mean position of every element carrying that ref.

    A `ref` holding several stands ('2 2A', 'A26;A28') is split, so each listed stand gets the element's
    position; a stand drawn as several elements (one per aircraft-size stop line) is averaged."""
    rows = []
    for st, body in osm.items():
        for e in body.get("elements", []):
            tags = e.get("tags", {})
            if tags.get("aeroway") != "parking_position" or not tags.get("ref"):
                continue
            lat = e.get("lat", e.get("center", {}).get("lat"))
            lon = e.get("lon", e.get("center", {}).get("lon"))
            if lat is None or lon is None:
                continue
            for part in re.split(r"[;,]|\s+", tags["ref"].strip()):
                n = normalise_stand(part)
                if n:
                    rows.append((st, n, float(lat), float(lon)))
    df = pl.DataFrame(
        rows, schema={"icao": pl.Utf8, "stand": pl.Utf8, "lat": pl.Float64, "lon": pl.Float64}, orient="row"
    )
    return df.group_by(["icao", "stand"]).agg(pl.col("lat").mean(), pl.col("lon").mean(), pl.len().alias("n_elem"))


def fetch_runway_ends(cache: Path | None = None, timeout: int = 120) -> pl.DataFrame:
    """Runway-end coordinates per (airport, runway designator), from OurAirports. One row per end."""
    cache = cache or data_dir() / "external" / "runways.csv"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(requests.get(RUNWAYS_URL, timeout=timeout).text)
    raw = pl.read_csv(io.StringIO(cache.read_text()), infer_schema_length=0)
    raw = raw.filter(pl.col("closed") == "0")
    ends = []
    for side in ("le", "he"):
        ends.append(
            raw.select(
                pl.col("airport_ident").alias("icao"),
                pl.col(f"{side}_ident").str.strip_chars().alias("runway"),
                pl.col(f"{side}_latitude_deg").cast(pl.Float64, strict=False).alias("rlat"),
                pl.col(f"{side}_longitude_deg").cast(pl.Float64, strict=False).alias("rlon"),
            )
        )
    return pl.concat(ends).drop_nulls().unique(subset=["icao", "runway"], keep="first")


def add_stand_runway_distance(dep: pl.DataFrame, stands: pl.DataFrame, ends: pl.DataFrame) -> pl.DataFrame:
    """`osm_stand_rwy_km`: great-circle km from the stand's OSM position to the threshold the departure rolls
    from (the end named by RUNWAY_mvt). Null when either end of the pair is unknown. Row count and order kept."""
    d = (
        dep.select(["MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"])
        .with_columns(
            pl.col("STAND_mvt").map_elements(normalise_stand, return_dtype=pl.Utf8).alias("_stand"),
            pl.col("RUNWAY_mvt").cast(pl.Utf8).str.strip_chars().alias("_rwy"),
        )
        .join(
            stands.select(["icao", "stand", "lat", "lon"]),
            left_on=["ADEP_mvt", "_stand"],
            right_on=["icao", "stand"],
            how="left",
            maintain_order="left",
        )
        .join(ends, left_on=["ADEP_mvt", "_rwy"], right_on=["icao", "runway"], how="left", maintain_order="left")
    )
    km = great_circle_km(d["lat"].to_numpy(), d["lon"].to_numpy(), d["rlat"].to_numpy(), d["rlon"].to_numpy())
    out = dep.with_columns(pl.Series("osm_stand_rwy_km", np.where(np.isfinite(km), km, np.nan)).fill_nan(None))
    if out.height != dep.height:
        raise AssertionError("add_stand_runway_distance changed the row count")
    return out
