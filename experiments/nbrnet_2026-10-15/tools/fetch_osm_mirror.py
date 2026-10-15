"""Fill data/external/osm/parking_<ICAO>.json from an Overpass mirror when overpass-api.de refuses connections.

Same query text and cache format as prc.osm.fetch_osm_aeroway (which reads these files as its cache), so the
pipeline's code path is unchanged; only the server that answered differs. Used on 2026-10-15 when
overpass-api.de reset every connection (see RESULTS.md, deviations)."""

import json
import sys
import time

import requests

from prc.config import TARGET_AIRPORTS, data_dir
from prc.osm import USER_AGENT

MIRRORS = (
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)


def main() -> int:
    cache_dir = data_dir() / "external" / "osm"
    cache_dir.mkdir(parents=True, exist_ok=True)
    used = {}
    for st in TARGET_AIRPORTS:
        cache = cache_dir / f"parking_{st}.json"
        if cache.exists():
            try:
                if "elements" in json.loads(cache.read_text()):
                    used[st] = "cached"
                    continue
            except json.JSONDecodeError:
                pass
        query = (
            f'[out:json][timeout:170];area["icao"="{st}"]->.a;'
            '(nwr["aeroway"="parking_position"](area.a);nwr["aeroway"="runway"](area.a););out center tags;'
        )
        for attempt in range(3):
            for url in MIRRORS:
                try:
                    r = requests.post(url, data={"data": query}, headers={"User-Agent": USER_AGENT,
                                      "Accept": "application/json"}, timeout=200)
                    if r.status_code == 200:
                        body = r.json()
                        if body.get("elements"):
                            tmp = cache.with_suffix(".json.tmp")
                            tmp.write_text(json.dumps(body))
                            tmp.replace(cache)
                            used[st] = url
                            print(st, url, len(body["elements"]), flush=True)
                            break
                    print(st, url, "HTTP", r.status_code, flush=True)
                except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
                    print(st, url, type(e).__name__, flush=True)
            if st in used:
                break
            time.sleep(20)
        if st not in used:
            print("FAILED", st, flush=True)
        time.sleep(3)
    (cache_dir / "_sources.json").write_text(json.dumps(used, indent=1))
    print(json.dumps(used))
    return 0 if len(used) == len(TARGET_AIRPORTS) else 1


if __name__ == "__main__":
    sys.exit(main())
