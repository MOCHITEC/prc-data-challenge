# External data

The competition rules require any external data to be openly accessible and usable, documented, and under an open
licence. This file lists what we use, where it comes from, under which licence, and what we extracted from it.

## adsb.lol historical ADS-B traces (pushback override)

| | |
|---|---|
| **Source** | adsb.lol, a community ADS-B feeder network. The data is its daily `globe_history` archives. |
| **URL** | `https://github.com/adsblol/globe_history_2025/releases` and `https://github.com/adsblol/globe_history_2026/releases`. One release per UTC day, tag `vYYYY.MM.DD-planes-readsb-prod-0`, a tar split into 2 GB parts. Documentation: `https://www.adsb.lol/docs/open-data/historical/` |
| **Licence** | **Open Database License (ODbL) 1.0**, `http://opendatacommons.org/licenses/odbl/1.0/`. It is stated in each release's `README.txt` ("This database is made available under the Open Database License") and shipped as `LICENSE-ODbL.txt`. |
| **Attribution** (to carry in any write-up, report or submission that uses it) | **"Contains data from adsb.lol (globe_history), made available under the Open Database License (ODbL) 1.0."** |
| **Days used** | Labelled (2025): 2025-07-21, 2025-07-13, 2025-01-22, 2025-11-12, used to validate the rules and estimate the bias. Scored (2026): every day of January and July 2026, 62 days. |
| **What was extracted** | Only fixes within 8 km of the reference point of our 10 airports, on the ground or at ≤ 4,000 ft. Each fix keeps its ICAO24 address, time (UTC), lat/lon, ground flag, baro altitude, ground speed, track, flags, source, and a callsign filled within each leg. That is about 7–17 MB of parquet per day, out of a 2–4 GB archive. From those fixes we derive one value per departure: the **pushback time**. This is the first movement off a stationary stand, kept only when that motion runs against the reported track (backwards) and no later stop of ≥ 15 min follows before take-off. |
| **How** | `scripts/adsb_stage1_feasibility.py` does the extraction and detection, `scripts/adsb_ingest_all.py` streams the 62 days (one raw day on disk at a time, deleted after filtering), and `scripts/adsb_override_candidate.py` applies it. All three are written from scratch for this repository; no third-party extractor code was used or consulted. The archives are read as a tar stream straight from the split parts. Each `trace_full_*.json` is parsed with the Python standard library and filtered with numpy. |
| **Join to the competition data** | On `CALLSIGN_flt` at `ADEP_mvt`, taking the take-off nearest `MVT_TIME_UTC_mvt` within ±30 min. Every time is UTC. |
| **How it enters the prediction** | For a departure at EDDM, EGLL, EHAM, LEBL or LSZH with a detected pushback, the prediction becomes `MVT_TIME − pushback + b`. `b` is a per-airport median offset, estimated on the labelled 2025 days. The model's prediction is kept whenever the two disagree by more than 900 s. `BLOCK_TIME_UTC_mvt` is not used on scored rows (it is blank there). Details: `docs/competitor_research_2026-09-21/results/adsb-full-ingest-candidate.md`. |
| **Derived database** | Under ODbL, the filtered fixes and the per-departure pushback times are a derived database. If they are published, they must be released under ODbL with the attribution above. A prediction file (a "produced work") needs only the attribution notice. |

The airport reference points used by the 8 km filter come from **OurAirports** (`https://ourairports.com/data/`,
public domain), via `src/prc/airports.py`, cached at `data/external/airports.parquet`.

## OpenStreetMap stand positions and OurAirports runway ends (stand-to-runway distance)

| | |
|---|---|
| **Source** | OpenStreetMap (OSM), elements tagged `aeroway=parking_position` (stand `ref` and position) and `aeroway=runway` inside each of our 10 airports. Runway-end coordinates from OurAirports `runways.csv`. |
| **URL** | OSM through the Overpass API (`https://overpass-api.de/api/interpreter`) or, where Overpass drops the connection from this environment, the main OSM API `map` call (`https://api.openstreetmap.org/api/0.6/map`) over a box around the aerodrome split into tiles. Both serve the same OSM database. OurAirports: `https://davidmegginson.github.io/ourairports-data/runways.csv` (`https://ourairports.com/data/`). |
| **Licence** | OSM: **Open Database License (ODbL) 1.0** (`https://www.openstreetmap.org/copyright`). OurAirports: public domain (`https://ourairports.com/data/`). |
| **Attribution** (to carry in any write-up, report or submission that uses it) | **"© OpenStreetMap contributors, available under the Open Database License (ODbL) 1.0."** |
| **Fetch date** | The caches behind `outputs/main_flseq_osm.parquet` were fetched 2026-09-29 23:42 – 2026-09-30 00:14 UTC (EDDF, EDDM, EGLL, EHAM, LIRF, LTFM via Overpass; LEBL, LEMD, LFPG, LSZH via the `map` API). OSM changes daily, so a later fetch can give a slightly different stand table. |
| **What was extracted** | Per airport, one row per normalised stand `ref` with the mean position of the elements carrying it (`prc.osm.stand_table`); per runway end, its designator and threshold coordinates. |
| **How** | `scripts/fetch_osm_stands.py` fills the caches (`data/external/osm/parking_<ICAO>.json`, `data/external/runways.csv`; not in git). `src/prc/osm.py` holds the fetchers and the join. Written from scratch for this repository. |
| **Join to the competition data** | `STAND_mvt` (normalised: upper case, no spaces/hyphens, no leading zeros) at `ADEP_mvt` to the OSM stand ref, and `RUNWAY_mvt` to the OurAirports runway-end designator. |
| **How it enters the prediction** | One feature, `osm_stand_rwy_km`: great-circle km from the stand's OSM position to the threshold of the departure runway, null when either side is unknown (`fe_osm_dist: true`, `configs/main_flseq_osm.yaml`). Non-null on 90.7% of the labelled rows and 91.5% of the scored rows (EDDM 57.4% / 57.6%, lowest). |
| **Derived database** | The stand-position table is a derived database of OSM. If it is published, it must be released under ODbL with the attribution above. A prediction file (a "produced work") needs only the attribution notice. |

## Iowa Environmental Mesonet ASOS/METAR archive (weather, `use_weather: true`)

| | |
|---|---|
| **Source** | Iowa Environmental Mesonet (IEM), Iowa State University: its ASOS/METAR archive. Its download page names its upstream sources as Unidata IDD, NCEI ISD and MADIS One Minute ASOS. |
| **URL** | Download service `https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py`; page `https://mesonet.agron.iastate.edu/request/download.phtml`. |
| **Licence** | IEM's terms (`https://mesonet.agron.iastate.edu/disclaimer.php`, "Usage of IEM Products"), verbatim: "The materials found on this website are in the public domain and may be used freely by anyone for any lawful purpose. Attributing the Iowa Environmental Mesonet of Iowa State University would be appreciated." The same page states the data are provided "without any warranty of accuracy". |
| **Attribution** (requested, not required) | **"Weather observations from the Iowa Environmental Mesonet (IEM) of Iowa State University."** |
| **What was extracted** | METAR rows for the 10 airports' ICAO stations, 2025-01-01 to 2026-08-01, fields `tmpf, dwpf, relh, drct, sknt, gust, p01i, vsby, wxcodes, skyc1, feel` (`prc.weather.IEM_FIELDS`), cached as `data/ref/metar/<ICAO>_2025-01-01_2026-08-01.csv` (not in git). |
| **How** | `prc.weather.fetch_metar` (one request per station, 1.2 s apart, as IEM asks for at most one request per second), `prc.weather.hourly_weather_table` aggregates to station-hours, cached as `data/ref/metar_hourly.parquet`. Written from scratch for this repository. |
| **Join to the competition data** | `ADEP_mvt` to the station, and the UTC hour of `MVT_TIME_UTC_mvt` (and the hour before) to `hour_utc` (`prc.weather.join_weather`). |
| **How it enters the prediction** | The production recipes (`use_weather: true`, e.g. `configs/main_flseq.yaml`) carry temperature, dew-point spread, wind, gust, precipitation, visibility, a frozen-precipitation flag and derived de-icing / low-visibility flags, for the take-off hour and the hour before. |
