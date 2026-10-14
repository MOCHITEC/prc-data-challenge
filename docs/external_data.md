# External data

The competition rules require any external data to be openly accessible and usable, documented, and under an open
licence. This file lists every external source found in the code and notes: where it comes from, its licence, what
it is used for, and which script fetches or uses it. Updated 2026-10-14 for the integration branch.

## Summary

"Final chain" means the code that produced the final file (`README.md` §3). "Research only" means the source was
measured in notes or CV and did not reach any leaderboard submission or the final file. That status comes from
`experiments/submissions.jsonl` and the final-chain configs and scripts.

| source | licence | used for | status | main scripts |
|---|---|---|---|---|
| adsb.lol `globe_history` (ADS-B traces) | ODbL 1.0 | learned ADS-B residual, positional join, appear tier, 2026 stand map | **final chain** | `scripts/adsb_*.py`, `analysis/local_adsb_*.py`, `analysis/local_posjoin_*.py`, `docs/final_2026-10-11/adsb_layer.py`, `docs/appear_tier_2026-10-05/*.py` |
| OpenStreetMap (`aeroway=parking_position`, `aeroway=runway`) | ODbL 1.0 | stand positions: stand-to-runway distance feature, appear tier | **final chain** | `src/prc/osm.py`, `scripts/fetch_osm_stands.py`, `scripts/adsb_appear.py` |
| OurAirports `airports.csv`, `runways.csv` | public domain | airport reference points (8 km ADS-B filter), runway thresholds | **final chain** | `src/prc/airports.py`, `src/prc/osm.py` |
| Iowa Environmental Mesonet ASOS/METAR | public domain, attribution requested | hourly weather features (`use_weather: true`) | **final chain** | `src/prc/weather.py` |
| Open-Meteo historical weather API | CC BY 4.0 | precipitation / snow features in one weather arm (`configs/wx2_C.yaml`) | research only | `src/prc/weather.py` (`fetch_openmeteo`) |
| EUROCONTROL OPDI v0.0.2 flight list and flight events | EUROCONTROL disclaimer: attribution, non-commercial, no modification without permission | turnaround and stand-exit features (`fe_opdi`, `fe_opdi_exit`) | research only; a derived file is committed (see below) | `src/prc/opdi.py`, `scripts/opdi_events_ingest.py`, `scripts/adsb_opdi.py`, `docs/opdi_2026-09-30/`, `docs/opdi_events_2026-10-05/` |
| EUROCONTROL PRU taxi-out additional time (monthly, per airport) | EUROCONTROL disclaimer, as OPDI | check of a re-implementation of the PRU ATXOT indicator | research only; 240-row extract committed | `docs/research_2026-10-05_aobt/kpi/00_extract_pru.py`, `pru_published_2024_2025.csv` |
| EUROCONTROL ansperformance daily airport files (pre-departure delay, ATFM slot adherence, arrival ATFM delay) | EUROCONTROL disclaimer, as OPDI | leak and feature screening | research only; "All_Pre-Departure_Delay" ruled out as a label leak | `docs/research_2026-10-01/extdata/context/*.py`, `docs/research_2026-10-01/extdata/README.md` |
| X-Plane Scenery Gateway (stand coordinates) | not verified (notes say "GPLv2 or later, original text not checked") | probe of stand-coordinate coverage | research only | `docs/research_2026-10-01/extdata/offblock/xp.py` |
| ADS-B Exchange free samples (first day of each month) | terms forbid redistribution (per notes) | probe of LTFM ground coverage | research only, rejected | `docs/research_2026-10-01/extdata/offblock/adsbx_probe.py` |
| IANA time-zone database (through polars' time-zone support) | public domain | Europe/Rome UTC offset for the LIRF time-zone class | final chain | `docs/final_2026-10-11/apply_overlays.py`, `analysis/local_lirf_artefacts_apply.py` |

Not data inputs:
- the organisers' leaderboard API, `https://datacomp.opensky-network.org/api/competitions/<id>/leaderboard`. It is
  read by `src/prc/leaderboard.py` to check version numbers. Scores read from it are listed in
  `docs/DEVELOPMENT_HISTORY.md`.
- public competitor repositories. They were read for ideas only (`docs/competitor_research_2026-09-21/`).

### Derived data committed in this repository

| path | derived from | notice |
|---|---|---|
| `handoff/2026-10-13/adsb_store/`, `handoff/2026-10-13/june_adsb_for_adsb_layer/`, `handoff/2026-10-11/adsb/` (per-departure ADS-B features, positional joins, dwell, appear features, stand map) | adsb.lol, OpenStreetMap | Derived databases; made available under ODbL 1.0 with the attributions below |
| `docs/final_2026-10-11/adsb_models/`, `handoff/2026-10-11/adsb/learn/*.txt` (LightGBM text models) | trained on adsb.lol-derived features and the competition labels | Produced works |
| `handoff/2026-10-05/opdi_exitpark.parquet` (OPDI stand-exit columns per departure) | EUROCONTROL OPDI | Licence not open; see the release checklist |
| `docs/research_2026-10-05_aobt/kpi/pru_published_2024_2025.csv` | EUROCONTROL PRU taxi-out file | Licence not open; see the release checklist |

## adsb.lol historical ADS-B traces

| | |
|---|---|
| **Source** | adsb.lol, a community ADS-B feeder network. Its daily `globe_history` archives. |
| **URL** | `https://github.com/adsblol/globe_history_2025/releases` and `https://github.com/adsblol/globe_history_2026/releases`. There is one release per UTC day, tag `vYYYY.MM.DD-planes-readsb-prod-0`, a tar split into 2 GB parts. Some 2025 days (2025-05-28 to 06-10) are published only under `<tag>tmp` (handled in b424536). Documentation: `https://www.adsb.lol/docs/open-data/historical/` |
| **Licence** | **Open Database License (ODbL) 1.0**, `http://opendatacommons.org/licenses/odbl/1.0/`, stated in each release's `README.txt` and shipped as `LICENSE-ODbL.txt`. |
| **Attribution** | **"Contains data from adsb.lol (globe_history), made available under the Open Database License (ODbL) 1.0."** |
| **Days used** | Labelled 2025 days, all months (about 350 days in the integrator's store), to train and validate the learned residual and the appear tier. Scored 2026 days: January and July (62 days), then February and June for the final set. The June ingest had no 2026-06-11 on the integrator's side; see `handoff/2026-10-13/README.md`. |
| **What was extracted** | Only fixes within 8 km of the reference point of each of the 10 airports, on the ground or at ≤ 4,000 ft. Each fix keeps ICAO24, UTC time, lat/lon, ground flag, baro altitude, ground speed, track, flags, source and callsign. That is about 7–17 MB of parquet per day out of a 2–4 GB archive. Per departure, these fixes give the ground-movement features: push, first move, stand dwell, post-push stops, positions, and appear-tier geometry. |
| **How** | Extraction: `scripts/adsb_stage1_feasibility.py`. 2026 ingest: `scripts/adsb_ingest_all.py` (`--days` for Feb/Jun). 2025 ingest: `scripts/adsb_ingest_2025.py`. Features: `scripts/adsb_learn_features.py`. Residual CV and apply: `scripts/adsb_learn_cv.py`, `scripts/adsb_learn_apply.py`, `analysis/local_adsb_*.py`. Positional join: `analysis/local_adsb_posjoin.py`, `analysis/local_posjoin_*.py`, `analysis/local_dwell_end.py`. Appear tier: `scripts/adsb_appear.py`, `docs/appear_tier_2026-10-05/`. Final-set layer: `docs/final_2026-10-11/adsb_layer.py`, `docs/appear_tier_2026-10-05/final_adsb.py`. All of this code was written for this repository. |
| **Join to the competition data** | On `CALLSIGN_flt` at `ADEP_mvt`, taking the take-off nearest `MVT_TIME_UTC_mvt` within ±30 min. Rows without a callsign join use the positional join: take-off within ±90 s after an airport offset, runway heading and aircraft type. |
| **How it enters the prediction** | A LightGBM residual `c` on `y − b`, applied only to rows with ADS-B. See `docs/final_2026-10-11/adsb_layer_README.md` for the components, `docs/SOLUTION_OVERVIEW.md` §3 and `README.md` §2. Submissions v17–v20 used an earlier rule-based pushback override instead (`scripts/adsb_override_candidate.py`). |
| **Derived database** | The filtered fixes and per-departure features are a derived database. The committed copies (table above) are made available under ODbL 1.0 with the attribution above. A prediction file (a "produced work") needs only the attribution notice. |

## OpenStreetMap stand positions and OurAirports runway ends

| | |
|---|---|
| **Source** | OpenStreetMap (OSM): elements tagged `aeroway=parking_position` (stand `ref` and position) and `aeroway=runway` inside each of the 10 airports. Runway-end coordinates come from OurAirports `runways.csv`. |
| **URL** | OSM through the Overpass API (`https://overpass-api.de/api/interpreter`). Where Overpass dropped the connection, the main OSM API `map` call was used (`https://api.openstreetmap.org/api/0.6/map`), over tiles of a box around the aerodrome. Both serve the same database. OurAirports: `https://davidmegginson.github.io/ourairports-data/runways.csv` (`https://ourairports.com/data/`). |
| **Licence** | OSM: **ODbL 1.0** (`https://www.openstreetmap.org/copyright`). OurAirports: public domain. |
| **Attribution** | **"© OpenStreetMap contributors, available under the Open Database License (ODbL) 1.0."** |
| **Fetch date** | The caches were fetched 2026-09-29 23:42 – 2026-09-30 00:14 UTC. EDDF, EDDM, EGLL, EHAM, LIRF and LTFM came via Overpass; LEBL, LEMD, LFPG and LSZH via the `map` API. OSM changes daily, so a later fetch can give a slightly different stand table. |
| **What was extracted** | One row per airport and normalised stand `ref`, with the mean position of the elements carrying it (`prc.osm.stand_table`). One row per runway end, with its designator and threshold coordinates. |
| **How** | `scripts/fetch_osm_stands.py` fills the caches (`data/external/osm/parking_<ICAO>.json`, `data/external/runways.csv`; not in git). `src/prc/osm.py` holds the fetchers and the join. `scripts/adsb_appear.py` reads the same stand table. |
| **Join** | `STAND_mvt` at `ADEP_mvt`, normalised (upper case, no spaces or hyphens, no leading zeros), to the OSM stand ref. `RUNWAY_mvt` to the OurAirports runway-end designator. |
| **How it enters the prediction** | (1) Feature `osm_stand_rwy_km`: great-circle km from the stand to the departure runway threshold (`fe_osm_dist: true`). It is nulled at LIRF in the `*_nolirf` recipes. (2) The appear tier: distances from ADS-B fixes to the own stand and the nearest stand. |
| **Derived database** | The stand table is a derived database of OSM; if published it is under ODbL 1.0 with the attribution above. |

OurAirports `airports.csv` (`https://davidmegginson.github.io/ourairports-data/airports.csv`, public domain) gives
the airport reference points for the 8 km ADS-B filter (`src/prc/airports.py`; cache `data/external/airports.parquet`).
It also feeds the station coordinates for Open-Meteo and the candidate feature `dest_distance_km`, which is not in the
production recipes.

## Iowa Environmental Mesonet ASOS/METAR archive (weather, `use_weather: true`)

| | |
|---|---|
| **Source** | Iowa Environmental Mesonet (IEM), Iowa State University: its ASOS/METAR archive. Its upstream sources are Unidata IDD, NCEI ISD and MADIS One Minute ASOS. |
| **URL** | Download service `https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py`; page `https://mesonet.agron.iastate.edu/request/download.phtml`. |
| **Licence** | IEM's terms (`https://mesonet.agron.iastate.edu/disclaimer.php`), verbatim: "The materials found on this website are in the public domain and may be used freely by anyone for any lawful purpose. Attributing the Iowa Environmental Mesonet of Iowa State University would be appreciated." The data are provided "without any warranty of accuracy". |
| **Attribution** (requested) | **"Weather observations from the Iowa Environmental Mesonet (IEM) of Iowa State University."** |
| **What was extracted** | METAR rows for the 10 airports' stations, 2025-01-01 to 2026-08-01 (this covers February and June 2026). Fields: `tmpf, dwpf, relh, drct, sknt, gust, p01i, vsby, wxcodes, skyc1, feel` (`prc.weather.IEM_FIELDS`). Cached as `data/ref/metar/<ICAO>_2025-01-01_2026-08-01.csv` (not in git). |
| **How** | `prc.weather.fetch_metar` sends one request per station, 1.2 s apart. `prc.weather.hourly_weather_table` aggregates to station-hours (`data/ref/metar_hourly.parquet`). |
| **Join** | `ADEP_mvt` to the station; the UTC hour of `MVT_TIME_UTC_mvt` and the hour before to `hour_utc` (`prc.weather.join_weather`). |
| **How it enters the prediction** | Temperature, dew-point spread, wind, gust, precipitation, visibility, a frozen-precipitation flag, and derived de-icing and low-visibility flags. Computed for the take-off hour and the hour before. |

## Open-Meteo historical weather API (research only)

| | |
|---|---|
| **URL** | `https://archive-api.open-meteo.com/v1/archive`, model `ecmwf_ifs`, fields `precipitation, snowfall, snow_depth` (`src/prc/weather.py`, `OPENMETEO_*`). |
| **Licence** | Data **CC BY 4.0** (`https://open-meteo.com/en/license`). The free API tier is for non-commercial use. |
| **Attribution** | "Weather data by Open-Meteo.com, CC BY 4.0." |
| **Use** | Only the weather arm `configs/wx2_C.yaml` (`fe_weather_v2: C`), measured in `docs/weather_v2_2026-09-25/`. Not in any production recipe or in the final chain. |

## EUROCONTROL OPDI (research only)

| | |
|---|---|
| **Source** | EUROCONTROL Open Performance Data Initiative, v0.0.2: flight list (monthly) and flight events (10-day files), built from OpenSky ADS-B. `https://www.opdi.aero`. |
| **URL** | `https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_list/flight_list_{YYYYMM}.parquet`, `.../flight_events/flight_events_{span}.parquet`. |
| **Licence** | No open licence is stated on opdi.aero. The EUROCONTROL disclaimer applies (`https://ansperformance.eu/about/disclaimer/`): "It may be copied in whole or in part, provided that EUROCONTROL is mentioned as the source and it is not used for commercial purposes … may not be modified without prior written permission from EUROCONTROL." The code carries a LICENCE FLAG for this (`src/prc/opdi.py`, `scripts/opdi_events_ingest.py`). |
| **Use** | Turnaround time (`fe_opdi`, `configs/opdi_*.yaml`, `docs/opdi_2026-09-30/`). Stand-exit event (`fe_opdi_exit`, `configs/ox*.yaml`, `scripts/adsb_opdi.py`, `docs/opdi_events_2026-10-05/`, `handoff/2026-10-05/README.md`). All measured in CV only. The ledger and the final-chain configs contain no OPDI feature. |
| **Committed derivative** | `handoff/2026-10-05/opdi_exitpark.parquet` (136,388 rows: `MVT_ID_mvt` and five `opdi_exit_*` columns). |

## EUROCONTROL PRU taxi-out additional time (research only)

| | |
|---|---|
| **Source** | EUROCONTROL Performance Review Unit, monthly taxi-out additional time per airport. The local file name was `txo.xlsx` (sheet `DATA`). The repository does not record its download URL. Its columns (`YEAR, MONTH_NUM, …, TOTAL_ADD_TIME_MIN`) match EUROCONTROL's `Taxi-Out_Additional_Time.xlsx`. |
| **Licence** | EUROCONTROL disclaimer, as for OPDI. |
| **Use** | `docs/research_2026-10-05_aobt/kpi/00_extract_pru.py` extracts 2024–2025 rows for the 10 airports into `pru_published_2024_2025.csv` (240 rows). `02_baseline_validate.py` checks a re-implementation of the ATXOT indicator against it. Not a model input. It uses the vendored openpyxl (`kpi/pylib/`, MIT, licence files included). |

## EUROCONTROL ansperformance daily airport files (research only)

| | |
|---|---|
| **Files** | `apt_dly_2025/2026.csv`, `atfm_slot_adherence_2025/2026.csv`, `atc_pre_departure_delays_2025/2026.csv`, `all_pre_departure_delays_2025/2026.csv`. The local copies are not in git, and the download URLs are not recorded in the scripts. |
| **Licence** | EUROCONTROL disclaimer, as for OPDI. |
| **Use** | `docs/research_2026-10-01/extdata/context/{measure,leak,cv,cv2,resid}.py`. "All_Pre-Departure_Delay" was ruled out: its daily values correlate 0.998 with the label-derived daily mean delay (`docs/research_2026-10-01/extdata/README.md`). None of the files is a model input. |

## Other research probes (not used)

- **X-Plane Scenery Gateway** (`https://gateway.x-plane.com/apiv1/airport/{icao}`, `/scenery/{id}`):
  `docs/research_2026-10-01/extdata/offblock/xp.py`. Licence noted as "GPLv2 or later (original text not checked)".
  Judged a small gain at EDDM only; not implemented.
- **ADS-B Exchange samples** (`https://samples.adsbexchange.com/readsb-hist/{day}/{hh}0000Z.json.gz`):
  `docs/research_2026-10-01/extdata/offblock/adsbx_probe.py`. Two days of 62 covered, and the terms forbid
  redistribution; rejected.
- Sources listed as rejected (no code for them in `src/`, `scripts/` or the final chain) in `docs/research_2026-10-01/extdata/README.md` and
  `docs/competitor_research_2026-09-21/results/16-external-data.md`: Meteostat, OpenSky Trino and REST, adsb.fi,
  airplanes.live, the FAA aircraft characteristics database, ICAO Doc 8643, AIP and NOTAM.
