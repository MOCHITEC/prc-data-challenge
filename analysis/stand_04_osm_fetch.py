"""Fetch stand/gate positions and runway geometry for the 10 airports from OpenStreetMap (Overpass API, ODbL).

Output: SCRATCH/osm_<APT>.json (raw Overpass JSON). The public Overpass endpoint resets connections now and then
through the session proxy, so each request is retried.
Query: aeroway = parking_position | gate | holding_position (nodes and ways, with centre) and aeroway = runway
(ways, full geometry) inside a bounding box around each aerodrome.
"""

from __future__ import annotations

import json
import os
import subprocess
import time

from stand_common import SCRATCH

# (lat, lon, dlat, dlon) - aerodrome reference point and half-size of the box
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


def query(apt: str) -> str:
    lat, lon, dl, dn = BOX[apt]
    bb = f"{lat - dl},{lon - dn},{lat + dl},{lon + dn}"
    return (
        f"[out:json][timeout:120];("
        f'nwr["aeroway"~"^(parking_position|gate|holding_position)$"]({bb});'
        f'way["aeroway"="runway"]({bb});'
        f");out tags center geom;"
    )


def fetch_osm_api_tiles(apt: str, step_lat: float = 0.02, step_lon: float = 0.0275) -> None:
    """Fallback when Overpass keeps failing (LIRF on 2026-09-24): download the aerodrome box from the main OSM
    API (/api/0.6/map, small tiles) and convert the aeroway objects into the same JSON shape as Overpass."""
    import xml.etree.ElementTree as ET

    lat, lon, dl, dn = BOX[apt]
    tdir = f"{SCRATCH}/{apt.lower()}_tiles"
    os.makedirs(tdir, exist_ok=True)
    nodes: dict[int, tuple[float, float]] = {}
    ways: dict[int, tuple[dict, list[int]]] = {}
    tagged_nodes: dict[int, dict] = {}
    la = lat - dl
    while la < lat + dl - 1e-9:
        lo = lon - dn
        while lo < lon + dn - 1e-9:
            f = f"{tdir}/t_{la:.4f}_{lo:.4f}.xml"
            if not os.path.exists(f):
                bb = f"{lo:.4f},{la:.4f},{lo + step_lon:.4f},{la + step_lat:.4f}"
                subprocess.run(["curl", "-sS", "-m", "120", "-A", "prc-data-challenge-analysis/0.1", "-o", f,
                                f"https://api.openstreetmap.org/api/0.6/map?bbox={bb}"], check=False)
            root = ET.parse(f).getroot()
            for n in root.iter("node"):
                nid = int(n.get("id"))
                nodes[nid] = (float(n.get("lat")), float(n.get("lon")))
                tags = {t.get("k"): t.get("v") for t in n.iter("tag")}
                if tags.get("aeroway"):
                    tagged_nodes[nid] = tags
            for w in root.iter("way"):
                tags = {t.get("k"): t.get("v") for t in w.iter("tag")}
                if tags.get("aeroway") in ("parking_position", "gate", "holding_position", "runway"):
                    ways[int(w.get("id"))] = (tags, [int(nd.get("ref")) for nd in w.iter("nd")])
            lo += step_lon
        la += step_lat
    el = []
    for nid, tags in tagged_nodes.items():
        el.append({"type": "node", "id": nid, "lat": nodes[nid][0], "lon": nodes[nid][1], "tags": tags})
    for wid, (tags, nds) in ways.items():
        pts = [nodes[i] for i in nds if i in nodes]
        if not pts:
            continue
        el.append({"type": "way", "id": wid, "tags": tags,
                   "center": {"lat": sum(p[0] for p in pts) / len(pts), "lon": sum(p[1] for p in pts) / len(pts)},
                   "geometry": [{"lat": p[0], "lon": p[1]} for p in pts]})
    json.dump({"elements": el, "source": "api.openstreetmap.org/api/0.6/map tiles (ODbL)"}, open(f"{SCRATCH}/osm_{apt}.json", "w"))
    print(apt, "elements (OSM API tiles)", len(el))


def main() -> None:
    for apt in BOX:
        out = f"{SCRATCH}/osm_{apt}.json"
        if os.path.exists(out) and os.path.getsize(out) > 1000:
            continue
        for attempt in range(8):
            r = subprocess.run(
                ["curl", "-sS", "-m", "180", "-X", "POST", "-A", "prc-data-challenge-analysis/0.1",
                 "--data-urlencode", f"data={query(apt)}", "-o", out, "-w", "%{http_code}",
                 "https://overpass-api.de/api/interpreter"],
                capture_output=True, text=True,
            )
            ok = r.stdout.strip() == "200"
            if ok:
                try:
                    n = len(json.load(open(out))["elements"])
                    print(apt, "elements", n, flush=True)
                    break
                except Exception as e:  # noqa: BLE001
                    ok = False
                    print(apt, "bad json", e)
            print(apt, "attempt", attempt, r.stdout.strip(), r.stderr.strip()[:100], flush=True)
            time.sleep(5 + 5 * attempt)
        else:
            fetch_osm_api_tiles(apt)


if __name__ == "__main__":
    main()
