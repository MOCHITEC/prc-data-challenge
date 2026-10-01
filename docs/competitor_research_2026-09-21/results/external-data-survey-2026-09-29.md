# External-data survey, 2026-09-29

Question: "Is there any other external data we should still add?"

This note reports facts only. `[S]` means read from the cited source or measured in this sandbox on 2026-09-29.
`[R]` means relayed second-hand, not verified first-hand. A source that could not be read says **unknown**.
Nothing in `src/`, `configs/`, `data/` or `outputs/` was changed. The current branch is
`claude/model-b-and-arrival-features`, at HEAD `ca49f98` when this note was written.

---

## 0. Summary table

| source | licence (exact) | Jan + Jul 2026 available? | covers what | status in repo |
|---|---|---|---|---|
| adsb.lol `globe_history` | ODbL 1.0 | yes, every day | global ADS-B traces, ground fixes included | **in use**: pushback override at EDDM, EGLL, EHAM, LEBL, LSZH. No surface picture at LFPG, LIRF, LTFM |
| OSM `aeroway=parking_position` (stand to runway-end km) | ODbL | n/a (static, fetched 2026-09-28) | one physical column per stand | **passed 3 pre-registered conditions** on `origin/claude/beautiful-davinci-ehugk4`. **Not in the current branch** |
| OurAirports | public domain | n/a (static) | airport and runway coordinates | in use (`src/prc/airports.py`) |
| IEM ASOS/METAR | public domain | yes | hourly METAR, no precipitation for non-US sites | **in use** (`use_weather: true`), but **not listed in `docs/external_data.md`** |
| Open-Meteo (precip, snowfall) | CC BY 4.0 | yes | ERA5/IFS hourly grid weather | **measured and rejected** (weather-v2 arm C, davinci branch) |
| Meteostat bulk | CC BY-NC (per slot 16; licence page not reachable) | Jan yes, **Jul no** (files end 2026-03-29/30) | station weather | rejected |
| EUROCONTROL AIU / ansperformance.eu files | EUROCONTROL disclaimer: NC, no modification. [R] organiser: "can be considered open data for the challenge" | yes, measured (details in §3) | **per airport-day** or **per airport-month**, never per flight | not used |
| OPDI v0.0.2 | no licence stated. opdi.aero links the same EUROCONTROL disclaimer | yes (Jan 2026 file 20 MB, Jul 2026 file 53 MB) | OpenSky-derived flight list and OSM-crossing events | not used. The events route was refuted (0.8 % usable). OpenAir's icao24 turnaround feature was worse in their own E77 |
| OpenSky Network (REST/Trino/samples) | "General Terms of Use & Data License Agreement": non-profit research, education, government | REST: state vectors ≤ 1 h back, tracks ≤ 30 days back. Public samples end 2022-06-27. Trino needs an approved application | state vectors with `on_ground` | not used. [R] organiser: Trino **not allowed** |
| ADS-B Exchange | JETNET proprietary terms, no open licence | **only 2026-01-01 and 2026-07-01** are free (2 of 62 scored days) | readsb snapshots, traces, operations | not used |
| adsb.fi | "personal, non-commercial use only" | no historical archive documented | live API only | not used |
| airplanes.live | official terms page returned 403 to this sandbox. A third-party README says educational/non-commercial | no historical archive found | live API only | not used |

---

## 1. OSM stand-to-runway distance

### 1.1 Where it lives

- Branch: **`remotes/origin/claude/beautiful-davinci-ehugk4`**. There is no local branch of that name. [S] `git branch -a`
- Commits, oldest first [S] `git log origin/claude/beautiful-davinci-ehugk4`:
  - `45bc244`: pre-registration, pushed before any arm ran.
  - `d5669d2`: script `evaluate_unseen.py`.
  - `db23953`: noise1 arm.
  - `96503ff`: dist arm.
  - `7ba1cc2`: base_u arm.
  - `fafc7a3`: noise1_u arm.
  - `72d5807`: dist_u arm.
  - `f7b3c47`: verdict, *"OSM stand-to-runway distance passes all three pre-registered conditions: adoption candidate"*.
- Files on that branch [S]:
  - `docs/osm_stand_2026-09-28/plan.md`
  - `docs/osm_stand_2026-09-28/results.md`
  - `docs/osm_stand_2026-09-28/evaluate_unseen.py`
  - `docs/osm_stand_2026-09-28/run_arms.sh`
  - `src/prc/osm.py` (240 lines)
  - `tests/test_osm.py`
  - `configs/osm_*.yaml`
  - edits to `src/prc/train.py` and `docs/features_log.md`
- **Integration status: not integrated.** [S]
  - `git merge-base --is-ancestor f7b3c47 HEAD` returns false.
  - `src/prc/osm.py` does not exist on the current branch.
  - No OSM mention appears in `src/` or `configs/`, and `docs/external_data.md` does not mention OSM.
  - The merge base with the davinci branch is `08c2a74`.
  - The davinci branch carries other unmerged work: weather v2, article-FE, LIRF fill.

### 1.2 What was measured (quoted from `plan.md` and `results.md`, translated from Japanese)

- **Column.** `osm_stand_rwy_km` is the great-circle distance in km:
  - from the stand's OSM position (`aeroway=parking_position`, `ref` normalised, elements with the same ref averaged)
  - to the departure runway end with the same name as `RUNWAY_mvt` (OurAirports `runways.csv`).
  - If no match is found, the value is null.
- **Match rate, row-based, 2025 / 2026:** 90.7 % / 91.5 % overall. EDDM is lowest at 57.4 % / 57.6 %.
- **Base config.** Every arm is `configs/main_gap.yaml` (v14) with `folds: [forward]`.
- **Simulated-unseen test.** `sim_unseen_stands: 0.2` drops all training rows of 20 % of (airport, stand) pairs, selected by CRC32.
- **Arm results (forward RMSE):**

| arm | forward | matched rows |
|---|---:|---:|
| base | 219.72 | 198.53 |
| noise1 | 220.17 | 198.51 |
| dist | 219.70 | 198.23 |
| base_u | 225.42 | 203.59 |
| noise1_u | 225.42 | 203.60 |
| dist_u | 224.94 | 202.87 |

- **Condition 1: improvement on unseen-stand rows.** There are 65,242 such rows (19.9 %), 90.5 % of them OSM-matched.
  - base_u → dist_u: **−2.33 [−2.89, −1.75]** at K=0 and **−2.63 [−3.01, −2.25]** at K=200. Pass.
- **Condition 2: beats the noise twin.** noise1_u → dist_u: **−2.38 [−2.77, −2.02]** at K=200. Pass.
- **Condition 3: no harm on known stands.** base → dist on the normal forward fold: −0.02 [−0.73, +0.89] at K=0 and −0.34 [−0.50, −0.15] at K=200. Pass.
  - noise1 → dist at K=200: −0.21 [−0.36, −0.07]. This is the first column in that series to beat its noise twin at K=200.
- **Per airport, unseen-stand rows, base_u → dist_u:**

| airport | change (s) |
|---|---:|
| LTFM | −6.10 |
| EGLL | −5.83 |
| LEMD | −4.67 |
| EDDF | −3.23 |
| LFPG | −1.38 |
| EHAM | −1.13 |
| LIRF | +0.64 |
| LSZH | +0.96 |
| EDDM | +1.38 |
| LEBL | +1.47 |

- **The authors' own size estimate:**
  - Stands never seen in 2025 are **1,589 ranking rows (0.46 %)**. On these rows the mechanism is worth **"about −0.01 s"** overall.
  - If the known-stand effect is real, the gain is "about 0.2–0.3 s".
  - The stated case for adopting it is robustness on a hidden final set.
- **Pre-production steps the authors list:**
  - a full-fold run: jan / jul / jul_fwd / forward;
  - adding the ODbL attribution ("© OpenStreetMap contributors, ODbL") to `docs/README.md`;
  - noting the fetch date, 2026-09-28.
  - The cache `data/external/osm/` is not in git.
- **Fetch route.**
  - EDDF, EGLL, EHAM, LEBL, LIRF, LSZH: Overpass.
  - EDDM, LEMD, LFPG, LTFM: the OSM main-API `map` call, because Overpass kept dropping the connection.

---

## 2. Open ADS-B / ground-surveillance sources for LFPG, LIRF, LTFM

### 2.1 adsb.lol (already in use)

- **Licence.** ODbL 1.0, per each release `README.txt` (`docs/external_data.md`).
  - Issue `adsblol/globe_history_2026#6` (opened 2026-09-05) asks whether CC0 or ODbL governs, because repo pages disagree. It is closed with no visible maintainer reply. [S] https://github.com/adsblol/globe_history_2026/issues/6
- **2026 coverage.** Daily tags exist for every Jan and Jul 2026 day. [S] `25-top-teams-update-2026-09-29.md` §4.3
- **Ground coverage.** Measured in `adsb-stage1-feasibility.md`:
  - LFPG 0–0.3 % pushback, LIRF 0–1.4 %, LTFM 0 %. All three are "NO-GO".
  - LIRF had no ground fixes at all on 2026-07-20.
  - LTFM had no ground fixes on either day and was unjoinable in 2025.
- **The late-NM bucket.** Late-NM rows *"sit at LTFM (48), EGLL (36) and LFPG (11), the airports where ADS-B has no surface picture"*.
  - LFPG, LIRF and LTFM carry 9.4 pp of the 16.7 pp lane-A SSE share.
  - ADS-B reaches "essentially none of it". [S] `adsb-stage1-feasibility.md` §4

### 2.2 OpenSky Network

- **Licence.** `states/LICENSE.txt` in the public sample bucket is titled *"General Terms of Use & Data License Agreement"*. It grants a licence *"solely for the purpose of non-profit research, non-profit education, commercial internal testing and evaluation of the data, or for government purposes"*. [S] https://s3.opensky-network.org/data-samples/states/LICENSE.txt
- **Public state-vector samples.** The last prefix is `states/2022-06-27/`, and `states/2026*` is empty. [S] S3 listing of `https://s3.opensky-network.org/data-samples`
- **REST API.**
  - Authenticated users get state vectors only up to 1 hour in the past.
  - Tracks are not available from more than 30 days back.
  - State vectors include `on_ground`.
  - [S] https://openskynetwork.github.io/opensky-api/rest.html
- **Trino / historical DB.**
  - Access is by application, for "university-affiliated researchers, governmental organisations, and aviation authorities". [S] slot 16 §6.2, citing https://openskynetwork.github.io/opensky-api/trino.html
  - [R] Organiser on Discord (2026-09-11): *"Trino data access is NOT allowed for the challenge"*.
  - [R] Organiser (2026-09-17/18): the open-data acceptance *"does not extend to OSN state vectors"*.
  - Both are relayed by kind-mango's `RECAP.md` "Discord confirmations" (https://raw.githubusercontent.com/skylinkapi/prc-data-challenge-2026-kind-mango/main/RECAP.md, head `2a4ec1b`).
- **OpenSky surface coverage via OPDI** (slot 16 §4.3, 10-day Jan 2026 window, `exit-parking_position` events):
  - LFPG 192, LIRF 118, LTFM 148
  - against LSZH 4,355 and EDDF 1,674.
- The OpenSky COVID flightlist on Zenodo (CC BY) is no longer updated after January 2022. [S] https://zenodo.org/records/5557026

### 2.3 ADS-B Exchange

- **Free data.** *"Complete data from the 1st of each month are made available free of charge."* [S] https://www.adsbexchange.com/data-products/sample-data/
  - Probed: `readsb-hist/2026/07/01/120000Z.json.gz` and `2026/01/01/120000Z.json.gz` return HTTP 200.
  - `readsb-hist/2026/07/02/…` returns 404.
  - That is **2 of the 62 scored days**.
- **Licence.** The ADSBx terms redirect to https://www.jetnet.com/legal/terms-of-use/. They state proprietary ownership and prohibit *"publish, resell, transmit, broadcast, distribute … data acquired from the Services"*. No open licence is named. [S]
- **Ground positions.** Yes: `alt_baro == "ground"` in readsb format. One snapshot measured on 2026-07-01 (aircraft within 5 km, on ground) [S]:

| time (UTC) | LFPG | LIRF | LTFM | EDDM | EHAM |
|---|---:|---:|---:|---:|---:|
| 06:00Z | 3 | 1 | 3 | 16 | 45 |
| 12:00Z | 3 | 6 | 6 | 11 | 41 |

### 2.4 adsb.fi

- **Licence.** *"adsb.fi open data is for personal, non-commercial use only. You may not license, sell, rent, or lease any part of the data or the service."* [S] https://github.com/adsbfi/opendata
- **Historical data.** No historical endpoint is documented. The `adsbfi` GitHub org (6 repos) has no history archive. [S]

### 2.5 airplanes.live

- **Official terms.** `airplanes.live/api/` and `/api-docs/` return HTTP 403 to this sandbox, so the official terms are **unknown**.
  - A third-party README (https://github.com/Bellaposa/airplanes-live-mcp) states the API is for educational and non-commercial purposes. [R]
- **Historical data.** No historical archive was found. The `airplanes-live` GitHub org (10 repos) has none. The API is live-only (`airplanes-live/api-archive` README). [S]

---

## 3. EUROCONTROL AIU / ansperformance.eu files

- **Catalogue.** [S] https://ansperformance.eu/data/ (read 2026-09-29). All files are at `https://www.eurocontrol.int/performance/data/download/xls/<name>.xlsx`.
- **Last modified.** Every file below has HTTP `last-modified: Fri, 18 Sep 2026`. [S] HEAD requests

| file | granularity | catalogue period | Jan/Jul 2026 in file |
|---|---|---|---|
| `Airport_Traffic` | daily, per airport | Jan 2016 – Aug 2026 | catalogue says yes; not opened |
| `Airport_Arrival_ATFM_Delay` | daily, per airport, by cause | Jan 2019 – Aug 2026 | catalogue says yes; not opened |
| `ATFM_Slot_Adherence` | daily, per airport | Jan 2019 – Aug 2026 | catalogue says yes; not opened |
| `ATC_Pre-Departure_Delay` | daily, per airport | Jan 2016 – Jun 2026 | **measured: DATA sheet has rows for all 31 days of Jan and Jul 2026 at all 10 airports** (Jan–Jul 2026 complete). Columns: `FLT_DEP_1, FLT_DEP_IFR_2, DLY_ATC_PRE_2, FLT_DEP_3, DLY_ATC_PRE_3` |
| `All_Pre-Departure_Delay` | daily, per airport | Jan 2020 – Jun 2026 | **measured: same, all 31 days of Jan and Jul 2026 at all 10 airports.** Columns: `FLT_DEP_1, FLT_DEP_IFR_2, DLY_ALL_PRE_2` |
| `Taxi-Out_Additional_Time` | monthly, per airport | Jan 2018 – Jun 2026 | July 2026 present for all 10 airports (slot 16 §5) |
| `Taxi-In_Additional_Time`, `ASMA_Additional_Time` | monthly, per airport | Jan 2018 – Jun 2026 | not opened |
| `Airport_Arrival_ATFM_Delay_post_ops` | monthly | Jan 2016 – Dec 2025 | no 2026 |

- The two pre-departure measurements count rows only. Whether their delay values are non-null was not checked.
- **No per-flight file exists** in this catalogue.
- **Licence.** *"It may be copied in whole or in part, provided that EUROCONTROL is mentioned as the source and it is not used for commercial purposes … may not be modified without prior written permission from EUROCONTROL."* [S] https://ansperformance.eu/about/disclaimer/
- **Organiser ruling.** [R] 2026-09-17/18 on Discord: *"EUROCONTROL AIU daily airport files and ADSB.lol (ODbL 1.0) count as open data for the challenge if declared with attribution"*, *"can be considered open data for the challenge"*. Relayed by kind-mango's `RECAP.md`.
- **Prior use by others.** ahmetabdullahgultekin used ATXOT for validation only (r = 0.757) and ATFM files as features, and reported no RMSE delta (slot 16 §2.3).
- **Status in repo.** No EUROCONTROL PRU file is referenced in `src/` or `configs/` (slot 16 §7). No experiment was found.

---

## 4. Sources the repo notes say other teams used

- **merry-quicksand: adsb.lol "ADN" stacker.**
  - 161,036 matched rows moved at 9 airports: LFPG 4,596, LTFM 5, LIRF excluded.
  - Board 281.52 → 267.45 (−7,722 MSE).
  - Source: `25-top-teams-update-2026-09-29.md` §3.2, `21-repo-merry-quicksand.md`.
- **OpenAir (likable-eagle): OPDI flight-list ground time.**
  - 281.72 → 278.36 (≈ −3.4 s). [S] `16-external-data.md` §2.3
  - Their E77 `icao24` turnaround scored 236.06 vs 235.94, i.e. worse. [S] `18-turnaround-features.md` §8
  - Raw OpenSky was rejected: `first_seen` is lift-off. [S] `06-repo-openair.md`
- **ahmetabdullahgultekin.**
  - Used IEM METAR, OurAirports, EUROCONTROL ATXOT / ATFM / Slot Adherence.
  - Listed OSM as a candidate only.
  - Reported no RMSE deltas. [S] `16-external-data.md` §2.3
- **satam2 (knowledgeable-helicopter).** Explored EUMETSAT, lightning and IMERG sources. All were stopped. [S] `25-top-teams-update-2026-09-29.md` §3.3
- **upstanding-firefly.** Its team page says it uses *"OpenSky ADS-B data for an aircraft-noise research project"*. How this relates to its score is unknown. [S] `25-…` §3.3
- **kind-mango.** `RECAP.md` relays that on 09-09 *"METAR, OPDI, OurAirports, OSM, Eurocontrol NM, VRS all fine"* if declared. [R]
- **Ranks 1–23.** No public code. What external data they use is **unknown**. [S] `25-…` §3
- **PRC 2025 podium.** No external data. [S] `16-external-data.md` §2.2

---

## 5. Already rejected, and why

| source | verdict | reason | where |
|---|---|---|---|
| Meteostat | rejected | bulk files end 2026-03-29/30, so no July 2026; CC BY-NC | `16-external-data.md` §3.3 |
| raw OpenSky | refuted | access by application, non-commercial | `16-…` §6.2, `plans/00-master-candidate-list.md:801` |
| OPDI flight events as a taxi-out measurement | refuted | 232 of ~30,600 departures (0.8 %) usable; `entry-deicing_pad` fires once in 10 days | `16-…` §4.3, `00-master-candidate-list.md:803` |
| OPDI `icao24` turnaround | not pursued | OpenAir E77 worse (236.06 vs 235.94) | `plans/P4-turnaround-features.md:623` |
| OPDI flight-list join | risk recorded | join 85.2 % Jan 2025, 58.9 % Jan 2026, 91.2 % Jul 2026 | `16-…` §4.3 |
| Open-Meteo precipitation / snowfall (weather-v2 arm C) | **rejected by pre-registered rule** | base → C at K=200: −0.31 [−0.51, −0.13], against −0.27 for the 7-column noise arm. noise7 → C at K=200: −0.04 [−0.23, +0.15] | `docs/weather_v2_2026-09-25/results.md` on `origin/claude/beautiful-davinci-ehugk4`, commit `82064c3` |
| METAR present-weather repair (arm B) | rejected | same rule | same file |
| ICAO Doc 8643, OpenFlights, EUROCONTROL APD, NOTAMs | unverified / not open | pages 403/404; EAD needs registration | `16-…` §6.6 |
| adsb.lol at LEMD | excluded from override | see the ingest note | `adsb-full-ingest-candidate.md` |

`docs/external_data.md` (22 lines) lists only adsb.lol and OurAirports. It records no rejections. It does **not** list IEM ASOS, although:

- `configs/main_gap.yaml` sets `use_weather: true`;
- `src/prc/weather.py:22` fetches `https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py`.

---

## 6. Facts relevant to the question

1. Among the open, historical, 2026-covering ADS-B sources checked, adsb.lol is the only one that is both openly licensed (ODbL 1.0) and complete for all 62 scored days.
   - ADSBx is proprietary and has 2 free days.
   - adsb.fi and airplanes.live have no archive and are non-commercial.
   - OpenSky needs an application and is [R] ruled out for Trino and state vectors.
2. The survey found no open source with measured ground coverage at LFPG, LIRF or LTFM.
   - adsb.lol: 0–1.4 % pushback detection there.
   - OPDI/OpenSky: 118–192 stand-exit events in 10 days.
   - ADSBx: 1–6 ground aircraft within 5 km in two snapshots, against 41–45 at EHAM.
3. The EUROCONTROL daily files cover Jan and Jul 2026 but are per airport-day. They cannot single out a late-NM-stamp flight.
4. OSM distance is the only candidate in the repo with a passed pre-registered verdict. It is not merged, and its estimated overall board effect is −0.01 s (new stands) to 0.2–0.3 s (known-stand effect, "likely but not established").
