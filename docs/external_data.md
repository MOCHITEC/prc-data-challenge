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
