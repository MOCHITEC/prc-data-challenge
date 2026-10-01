"""Fill the OSM stand / runway caches that `fe_osm_dist` reads (data/external/osm/, data/external/runways.csv).

    .venv/bin/python scripts/fetch_osm_stands.py

Each airport is tried on the public Overpass instance first and, if that keeps failing, on the main OSM API's
`map` call over the aerodrome box below split into tiles (prc.osm.fetch_osm_map_bbox). Both return the same OSM
data. An airport whose cache already exists is not fetched again, so the fetch date of an existing cache is the
date it was first written: the script prints each cache's modification time and element count.

Data licences (docs/external_data.md): OpenStreetMap, ODbL 1.0; OurAirports runways.csv, public domain.
"""

from __future__ import annotations

import datetime as dt
import json
import sys

from prc.config import TARGET_AIRPORTS, data_dir
from prc.osm import fetch_osm_aeroway, fetch_osm_map_bbox, fetch_runway_ends, stand_table

# (lat, lon, half-height deg, half-width deg) around each aerodrome, as in analysis/stand_04_osm_fetch.py.
BOX = {
    "EDDF": (50.035, 8.56, 0.035, 0.07),
    "EDDM": (48.354, 11.786, 0.03, 0.07),
    "EGLL": (51.470, -0.454, 0.025, 0.06),
    "EHAM": (52.330, 4.765, 0.06, 0.08),
    "LEBL": (41.297, 2.078, 0.03, 0.06),
    "LEMD": (40.490, -3.567, 0.06, 0.05),
    "LFPG": (49.009, 2.548, 0.035, 0.07),
    "LIRF": (41.800, 12.245, 0.04, 0.05),
    "LSZH": (47.460, 8.553, 0.03, 0.04),
    "LTFM": (41.262, 28.735, 0.06, 0.09),
}


def main() -> int:
    cache_dir = data_dir() / "external" / "osm"
    for st in TARGET_AIRPORTS:
        try:
            fetch_osm_aeroway([st], retries=2)
            route = "overpass"
        except RuntimeError as e:
            print(f"{st}: Overpass failed ({e}); using the OSM map API", flush=True)
            lat, lon, dl, dn = BOX[st]
            # The map call refuses a tile holding more than 50,000 nodes; split finer until every tile fits.
            for tiles in (3, 5, 8):
                try:
                    fetch_osm_map_bbox(st, (lon - dn, lat - dl, lon + dn, lat + dl), tiles=tiles)
                    break
                except RuntimeError as e2:
                    if "too many nodes" not in str(e2) or tiles == 8:
                        raise
                    print(f"{st}: {tiles}x{tiles} tiles too large, splitting finer", flush=True)
            route = f"map-api ({tiles}x{tiles} tiles)"
        f = cache_dir / f"parking_{st}.json"
        n = len(json.loads(f.read_text())["elements"])
        when = dt.datetime.fromtimestamp(f.stat().st_mtime, dt.UTC).isoformat(timespec="seconds")
        print(f"{st}: {n} elements via {route}, cache written {when}", flush=True)
    ends = fetch_runway_ends()
    osm = {st: json.loads((cache_dir / f"parking_{st}.json").read_text()) for st in TARGET_AIRPORTS}
    stands = stand_table(osm)
    print(f"runway ends: {ends.height}; stands per airport:")
    print(stands.group_by("icao").len().sort("icao"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
