# SLOT 16 — external-data: external data sources used by aviation-competition winners, and their documented access paths

**Marking:** `[S]` = read from a named source (URL given) or measured by me in this sandbox against a named file; `[X]` = my own extrapolation, not stated anywhere. Every `[S]` carries a URL or a reproducible command.

**Date of all liveness checks: 2026-09-21.**

---

## 0. The coverage test everything is judged against

`[S]` The scored set is **exactly 2026-01 and 2026-07**. Measured on `data/raw/ranking.parquet`:

```
ranking PHASE_mvt: ARR 344,693 | DEP 344,841   (689,534 rows total)
MVT_TIME_UTC_mvt month counts: 2026-01 → 305,590 ; 2026-07 → 383,944
DEP rows at the 10 airports: Jan 2026 152,719 | Jul 2026 192,122
```

`[S]` 10 airports: **EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LTFM, LSZH** (https://prc-data-challenge-2026.netlify.app/data.html).

`[S]` Training = calendar 2025 (12 monthly files). Source: same page.

So the coverage bar is: **the source must carry data for January 2026 AND July 2026 at European airports.** A source that ends in March or June 2026 covers at most half the scored set. This report states the coverage end date for every source and marks it PASS / HALF / FAIL.

`[S]` Prize-eligibility wording, verbatim from https://prc-data-challenge-2026.netlify.app/eligibility.html:
- "All used external datasets are openly accessible/usable and documented."
- "All **additional datasets** used are openly available under an **open source license**."
- "All produced **source code** are made openly available on GitHub under the **GNU GPLv3 license**."

That second clause is the one that creates the licence problem in §6.

---

## 1. Summary table — every source, with the coverage verdict

| # | Source | Covers Jan 2026? | Covers Jul 2026? | Coverage end (verified) | Licence | Verdict |
|---|--------|------------------|------------------|--------------------------|---------|---------|
| 1 | IEM ASOS/METAR archive | **yes** (probed LSZH) | **yes** (probed EDDF, LTFM) | live, synced every 10 min, archive through 2026-09-21 | public domain (stated) | **PASS** |
| 2 | Open-Meteo historical (ERA5 / IFS) | **yes** (probed EDDF) | **yes** (probed EDDF) | 2026-09-19 (2-day lag) | CC BY 4.0; free tier non-commercial | **PASS** |
| 3 | OPDI flight list v0.0.2 | **yes** (`flight_list_202601.parquet` HTTP 200) | **yes** (`flight_list_202607.parquet` HTTP 200) | 2026-07 (monthly file) | not stated on opdi.aero; EUROCONTROL PRU disclaimer applies (NC + no-modify) | **PASS (data), LICENCE FLAG** |
| 4 | OPDI flight events / measurements v0.0.2 | **yes** (windows `20251231_20260110`, `20260130_20260209`) | **yes** (`20260629_20260709`, `20260729_20260808`) | 2026-07-31 | same as #3 | **PASS (data), LOW YIELD (§4.3)** |
| 5 | EUROCONTROL PRU daily airport files (Airport_Traffic, Airport_Arrival_ATFM_Delay, ATFM_Slot_Adherence) | yes | yes | Aug 2026 per catalogue; files re-published 2026-09-18 | **NC, no modification w/o written permission** | **PASS (data), LICENCE FLAG** |
| 6 | EUROCONTROL Taxi-Out_Additional_Time.xlsx (ATXOT monthly) | **yes** | **yes** (measured: month 7 present for all 10 airports) | 2026-07 (EDDF/EGLL have 2026-08) | as #5 | **PASS (data), LICENCE FLAG** |
| 7 | EUROCONTROL ATC_Pre-Departure_Delay.xlsx / All_Pre-Departure_Delay.xlsx | yes | catalogue says Jun 2026 — **unverified for July** | catalogue: Jun 2026; file re-published 2026-09-18 | as #5 | **HALF / UNVERIFIED** |
| 8 | OurAirports (airports.csv, runways.csv) | n/a (static reference) | n/a | `last-modified: 2026-09-21` — rebuilt daily | public domain (stated) | **PASS** |
| 9 | Meteostat bulk hourly | yes | **NO** | **2026-03-29/30 at all 7 European stations probed** | CC BY-NC (see §3.3) | **HALF → effectively FAIL** |
| 10 | OpenSky Network raw (state vectors / flights_data4, Trino) | yes (2013→) | yes | unlimited retention since 2013 | **access gated by application; non-commercial/research only** | **FAIL the "openly accessible" test** |
| 11 | OpenAP aircraft/engine properties | n/a (static) | n/a | n/a | **LGPL-3.0** (code + data) | **PASS** |
| 12 | OpenStreetMap aeroway (taxiway/parking_position/apron/runway) | n/a (live) | n/a | live | **ODbL** — share-alike, see §6 | **PASS with a caveat** |
| 13 | OpenFlights routes | — | — | **could not verify** (openflights.org/data.html → 404) | ODbL claimed | **UNVERIFIED** |
| 14 | ICAO Doc 8643 type designators | n/a | n/a | n/a | **could not verify** (icao.int → 403) — no open bulk licence found | **UNVERIFIED / likely FAIL** |
| 15 | EUROCONTROL Aircraft Performance Database | n/a | n/a | n/a | page moved to learningzone.eurocontrol.int; **not verified**, no bulk download found | **UNVERIFIED** |
| 16 | European NOTAMs (EAD) / airport capacity declarations | — | — | **not verified**; EAD requires registration, slot-coordinator declarations are PDFs | not open | **FAIL / not machine-readable** |

---

## 2. What published winners actually used (and what they abandoned)

### 2.1 PRC Data Challenge 2024 (task: actual take-off weight; metric RMSE)

`[S]` https://conormclaughlin.net/2024/12/reviewing-the-winning-ml-models-from-the-opensky-prc-data-challenge-what-did-they-do-differently/

| Team | Rank / RMSE | External source | Join described | Quantified gain reported |
|---|---|---|---|---|
| likable-jelly | 1st, 1561 | **OpenAP** (openap.dev) — physics fuel-flow/mass model | applied `FuelFlow` to the supplied trajectories | **none stated** ("probably the coolest thing" is the reviewer's phrase, not a number) |
| likable-jelly | | **METAR** (destination airport) | "spatial/temporal clustering of airports and times so that the most relevant METAR reports could be attributed to each flight" | **none stated** |
| faithful-engine | 2nd, 1635 | **METAR** (destination) | not described | **none stated** |
| faithful-engine | | **Acropole** (github.com/DGAC/Acropole) — fuel estimation | trajectories resampled to 4 s | **none stated** |
| tiny-rainbow | 3rd, 2218 | "external data on aircraft and airports" | not named | **none stated** |

`[S]` **No 2024 winner published a per-source RMSE delta.** Every external-data claim from 2024 is qualitative.

### 2.2 PRC Data Challenge 2025 (task: fuel burn; metric RMSE)

`[S]` https://prc-data-challenge-2025.netlify.app/outcome.html

- 1st **resourceful-quiver** 200.83 — **no external dataset mentioned in the review.**
- 2nd **jubilant-vase** 211.99 — **no external dataset mentioned.**
- 3rd **outspoken-tornado** 219.85 — **no external dataset mentioned.**
- 6th **honest-rose** — *tried and abandoned*: weather variables and economic variables, "found limited influence".
- 7th **unique-honey** — README links external sources, unnamed in the review.
- 11th **bright-lobster** — METAR, SkyVector airport information, passenger statistics.

**This is the single most important negative result in this slot: in the most recent edition, the entire prize podium was won without external data, and the one team that reported testing weather reported it did not help.** `[X]` That was a fuel-burn task where the trajectory already encodes the atmosphere; taxi-out is a surface-congestion task where weather is the mechanism, so the 2025 null does not transfer directly — but it does say external data is not automatically where the points are.

### 2.3 PRC Data Challenge 2026 — public repos

**ahmetabdullahgultekin/prc-taxiout-2026** `[S]` — the only public 2026 repo with a proper external-data register (`docs/external_data.md`, read via https://raw.githubusercontent.com/ahmetabdullahgultekin/prc-taxiout-2026/master/docs/external_data.md). Licence GPLv3. Verbatim inventory:

| Source | URL | Declared licence | Coverage they state | Use | Number they report |
|---|---|---|---|---|---|
| IEM ASOS/METAR | `https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py` | public domain | **11 airports, 2025-01-01 → 2026-07-31, 306,222 observations** | temp, dewpoint, visibility, wind, precip, present-weather codes; de-icing proxy | <0.03 % missing temp/visibility; 577 days covered at every airport |
| OurAirports | `https://davidmegginson.github.io/ourairports-data/` | public domain | 86,013 airports | coordinates → departure bearing; **runway counts and lengths** | — |
| EUROCONTROL ATXOT methodology PDF | `https://ansperformance.eu/library/ATXOT_indicator_documentation_mar23.pdf` | "EUROCONTROL public publication" | — | definition of the P10 unimpeded reference | — |
| EUROCONTROL Taxi-Out_Additional_Time.xlsx | `https://www.eurocontrol.int/performance/data/download/xls/Taxi-Out_Additional_Time.xlsx` | "EUROCONTROL public publication" | monthly through Jun 2026 | **validation only** (checking their ATXOT reimplementation) | **r = 0.757 overall; 0.87–0.98 at cold airports** |
| EUROCONTROL ATFM_Slot_Adherence.xlsx | `.../xls/ATFM_Slot_Adherence.xlsx` | as above | 2019-01-01 → 2026-07-31, 11 airports | downstream-restriction factor (slot counts, adherence) | — |
| EUROCONTROL Airport_Arrival_ATFM_Delay.xlsx | `.../xls/Airport_Arrival_ATFM_Delay.xlsx` | as above | 2019-01-01 → 2026-07-31, 30,451 airport-days | daily arrival ATFM delay by reason code | **de-icing reason column is entirely empty** |
| OpenStreetMap `aeroway=taxiway/parking_position` | — | **ODbL** | — | **candidate, not implemented** — "low priority; empirical P10 considered superior" | — |

`[S]` They report **no RMSE delta for any source.** The only numbers are the validation correlation and coverage counts.

**ChandanHegde07/OpenAir (LB 276.99)** `[S]` — https://github.com/ChandanHegde07/OpenAir, GPL-3.0.
- **OPDI ground time**, defined by them as same-aircraft `first_seen − previous last_seen` per icao24 (i.e. a turnaround gap from the **flight list**, not from the event milestones). Experiment E98 / v23 blended it into the gate-delay branch. `[S]` Progression recorded in our own slot-06 read and in their status log: **281.72 → 278.36 on the OPDI step**, i.e. **≈ −3.4 s RMSE**, then 276.99 after further work. Their stage-5 rule: OPDI ground time ≥ 7200 s → shrink 50 % toward the median, ~146 rows.
- `[S]` **Tried and rejected:** raw OpenSky. They found `first_seen` "is lift-off (MVT − fs ≈ −26 s)", i.e. it is *not* a parking event, so it could not serve as an off-block time. Coverage reported as "ranking icao coverage 219/383" on unmatched rows.
- `[S]` **Also falsified by them** (not external-data, but relevant to what external congestion data could buy): rolling queue features did not shrink the tail; previous-departure block intervals corr ≈ 0.008.

**This ≈ −3.4 s from OPDI is the only quantified external-data gain anywhere in the 2026 public record.**

**Phoenix-Ops-LTD/prc2026-taxiout (LB 273.56)**, **javidmardanov (LB 268.56)**, **sergiuv11 (LB 286.66)** `[S]` — per our slots 07/08/09, all of their published gains come from features built out of the supplied movement/NM tables (arrival-residual, peer-context, TimesFM on the airport series, LIRF overlays). **No external dataset is credited with a gain in any of them.**

### 2.4 DrivenData / NASA "Pushback to the Future"

`[S]` https://github.com/drivendataorg/nasa-airport-pushback — Phase-1 MAE (minutes): Team CDS 10.673, Moles 10.728, Oracle 2 11.054, FLHuskies2 11.105, Cuong_Syr 11.946; benchmark >19.
`[S]` The task was framed as predicting pushback "from **public air traffic and weather data**" — the weather (NOAA **LAMP**) was **supplied inside the competition feature set** along with `config`, `etd`, `first_position`, `mfs`, `runways`, `standtimes`, `tbfm`, `tfm`.
`[X]` I could not retrieve the rules page (drivendata.org problem-description URL 404) and therefore **cannot state whether external data was permitted**. Treat "NASA winners used external data" as **unestablished**. What is established is that the organisers considered airport-surface weather essential enough to ship it as a core table.

---

## 3. Weather sources — the decisive comparison

### 3.1 IEM ASOS / METAR — the only weather source verified live for both scored months

`[S]` Service page: https://mesonet.agron.iastate.edu/request/download.phtml — "an ever growing archive of automated airport weather observations from around the world"; data "synced from the real-time ingest every 10 minutes"; archive current through 2026-09-21; request cap 1,000 station-years; **"precipitation data is unavailable for non-US sites."**
`[S]` Licence text quoted in the competitor register: "The materials found on this website are in the public domain and may be used freely by any lawful purpose." → **satisfies the open-licence clause.**

`[S]` **Live probes I ran (raw METAR rows returned):**
```
EDDF 2026-07-15 → EDDF,2026-07-15 00:50,71.60,6.21,9.00,M   (tmpf,vsby,sknt,wxcodes)
LTFM 2026-07-15 → LTFM,2026-07-15 00:50,68.00
LSZH 2026-01-15 → LSZH,2026-01-15 00:50,32.00
```
Jan 2026 and Jul 2026 both return data, at a Turkish and a Swiss station, i.e. across the full network. **PASS.**

`[S]` Access pattern: `GET https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=<ICAO>&data=tmpf&data=dwpf&...&tz=Etc/UTC&format=onlycomma&report_type=3&year1=..&month1=..&day1=..&year2=..&month2=..&day2=..`. `report_type=3` selects routine hourly METAR. The service throttles to roughly 1 request/s per IP and can return `ERROR: server over capacity, please try later` (I hit this twice during this investigation) — retry logic is mandatory.

`[S]` **Direct cross-check against our own repo.** `docs/investigation_2026-09-16/findings.md` records that `deicing_gate`'s precipitation term is inert: `precip_mm > 0` occurs in **0 of 138,446** METAR rows, and asks whether "the IEM `p01i` field is genuinely empty or lost in `parse_metar_csv`". **The IEM download page answers it: "precipitation data is unavailable for non-US sites."** `p01i` is empty by design for every one of our 10 airports. The parser is not at fault, and no amount of re-fetching will populate it. **Frozen precipitation has to come from `wxcodes` (present-weather codes, which IEM does serve for European sites) or from a second source.**

### 3.2 Open-Meteo historical archive — the credible second source

`[S]` https://open-meteo.com/en/docs/historical-weather-api — models: ERA5 0.25° (1940→, "Daily with 5 days delay"), ERA5-Land 0.1°, ECMWF IFS 9 km (2017→, "Every 6 hours with no delay"), CERRA 5 km Europe-only (1985 → Jun 2021, so **CERRA is useless here**).
`[S]` Licence: **CC BY 4.0** on the data (https://open-meteo.com/en/license); free tier is non-commercial and capped at 10,000 calls/day. CC BY 4.0 satisfies the open-licence clause. The non-commercial cap is on the *service*, not the data.

`[S]` **Live probes:**
```
archive-api.open-meteo.com/v1/archive?latitude=50.033&longitude=8.570&start_date=2026-07-15&end_date=2026-07-15&hourly=temperature_2m
  → returns hourly series for 2026-07-15  (PASS for July 2026)
same endpoint 2026-09-16 → 2026-09-19, hourly temperature_2m
  → first 2026-09-16T00:00, last 2026-09-19T23:00, last non-null 2026-09-19T23:00
```
So the archive is current to **2 days behind wall clock**, and both scored months are fully inside it. **PASS.**

`[X]` Its value over IEM is `snowfall`, `precipitation`, `soil_temperature` — exactly the fields IEM withholds for non-US stations. Our own `src/prc/weather.py` docstring already names it as the "secondary source (gap filling / snowfall)" but the module as written only implements the IEM fetch.

### 3.3 Meteostat — verified NOT usable for the scored set

`[S]` Bulk endpoint `https://bulk.meteostat.net/v2/hourly/{year}/{station}.csv.gz` is live (HTTP 200). I decompressed the 2026 file for **seven** European stations and read the last row:

```
10637 (Frankfurt)  → 2026-03-29,22,...
03772 (Heathrow)   → 2026-03-29,22,...
06240 (Schiphol)   → 2026-03-29,22,...
07157              → 2026-03-29,22,...
08181              → 2026-03-29,22,...
16242              → 2026-03-30,01,...
17060              → 2026-03-30,01,...
06670              → 2026-03-29,22,...
```
Every file stops at **2026-03-29/30**, and `last-modified` on the year files is 2026-03-19/2026-03-30. The full-archive file `v2/hourly/10637.csv.gz` is older still (`last-modified: 2025-08-14`).

**Meteostat bulk covers January 2026 and does NOT cover July 2026 — roughly 56 % of the scored rows would get nulls.** A feature built on it would also be *systematically* null in one scored month and present in the other, which is worse than no feature. **Do not use.** `[X]` The Meteostat JSON API may be fresher than the bulk dumps; I did not test it, and it is rate-limited/keyed. I could not reach https://dev.meteostat.net/bulk/ (404) to read the licence page; Meteostat's published terms are commonly **CC BY-NC**, which `[X]` would additionally fail the open-source-licence clause — unverified here.

### 3.4 NOAA

`[X]` Not separately investigated. IEM's own page states its inputs are "Unidata IDD, NCEI ISD, and MADIS One Minute ASOS" — i.e. **IEM is already the convenient front-end to NOAA's ISD archive** for this purpose, with a public-domain statement attached. Going to NCEI directly buys nothing except latency.

---

## 4. OPDI — the deepest dive, because it is the one source with a published gain

### 4.1 What exists

`[S]` https://www.opdi.aero/data — tables **FLIGHT**, **EVENT**, **MEASUREMENT**, published "in collaboration with the OpenSky Network (OSN)", plus **AIRPORT** and **RUNWAY** which are just OurAirports CSVs re-served. Stated coverage **January 2022 – July 2026**, version **v0.0.2**.

`[S]` Download URL patterns (taken from the page HTML, not guessed):
```
https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_list/flight_list_{YYYYMM}.parquet
https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_events/flight_events_{YYYYMMDD}_{YYYYMMDD_next10}.parquet
https://www.eurocontrol.int/performance/data/download/OPDI/v002/measurements/measurements_{YYYYMMDD}_{YYYYMMDD_next10}.parquet
```
`[S]` **The 10-day windows are anchored at 2022-01-01, not at month starts.** `flight_events_20260701_20260711.parquet` returns **404**; `flight_events_20260629_20260709.parquet` exists. This trips people up. Each file "covers data up to, but not including, the last day listed in its name".

`[S]` **Liveness, verified by HTTP:**
```
flight_list_202601.parquet   200, 20,263,227 bytes, last-modified 2026-05-20
flight_list_202607.parquet   200, 53,159,289 bytes, last-modified 2026-08-17
flight_events / measurements windows 20251231_20260110, 20260130_20260209,
  20260629_20260709, 20260729_20260808 → all present (416/206 on a range probe, not 404)
flight_events_20260110_20260120.parquet → 200, 271,813,923 bytes
```
**Both scored months are downloadable today.** Note the volume: events run ~272 MB per 10-day window, so 2025 + Jan/Jul 2026 events is roughly **10–12 GB**.

`[S]` The ansperformance.eu catalogue page still lists OPDI as "Jan 2022 – May 2026". **That page is stale**; opdi.aero and the files themselves say July 2026. Trust the files.

### 4.2 What "ground time" means

`[S]` https://www.opdi.aero/concepts defines 12 milestones: off-block **T04**, end-of-pushback **T05**, enter-runway **T06**, lift-off **T08**, 40 NM **T11**, top-of-climb **T12**, top-of-descent **T13**, 40 NM **T14**, touch-down **T17**, runway-vacated **T19**, enter-parking-spot **T20**, on-block **T21**. It defines **taxi-out = T04 → T06**, taxi-in = T19 → T21, block-to-block = T04 → T21.

**But that is the conceptual model, not what v0.0.2 ships** (see §4.3). And it is *not* what OpenAir used: `[S]` OpenAir's "OPDI ground time" is `first_seen − previous last_seen` on the same `icao24` **from the flight list** — a turnaround gap between two consecutive tracked flights of the same airframe, not a milestone-derived ground time.

### 4.3 What I measured, and why the milestone route is a dead end

**Flight-list schema, read off `flight_list_202601.parquet` (709,137 rows):**
```
id Int64 | icao24 | flt_id | dof | adep | ades | adep_p | ades_p
registration | model | typecode | icao_aircraft_class | icao_operator
first_seen | last_seen | version
```
`[S]` `flt_id` carries **trailing spaces** (`'UAE42   '`) — strip before joining or you get zero matches. In the 2026 files `first_seen`/`dof` are **strings**; in the 2025 files they are already **datetimes**. Handle both.

`[S]` **`first_seen` ≈ take-off, not off-block.** Joining Jan-2026 ranking DEP rows to OPDI on `(CALLSIGN_flt, ADEP_mvt, date)`, the median of `first_seen − MVT_TIME_UTC_mvt` is **+12 s** (p10 −3787 s, p90 +68 s; the long negative tail is mis-joins). This independently reproduces OpenAir's "MVT − fs ≈ −26 s" and confirms **OPDI's flight list gives you no off-block time.**

`[S]` **Join rate on `(CALLSIGN_flt, ADEP_mvt, date)`, DEP rows only, at the 10 airports:**

| Period | competition DEP rows | joined | rate |
|---|---|---|---|
| Jan 2025 (training) | 153,706 | 130,924 | **85.2 %** |
| **Jan 2026 (scored)** | 152,719 | 90,023 | **58.9 %** |
| **Jul 2026 (scored)** | 192,122 | 175,290 | **91.2 %** |

**This is a serious covariate shift and the most important thing in this section.** Any OPDI-derived feature is present for 85 % of training rows, **59 % of January 2026 rows** and 91 % of July 2026 rows. A model that learns "when OPDI ground time is missing, the flight is unusual" will be badly miscalibrated in January 2026, which is 44 % of the scored set. `[X]` OpenAir's +? on OPDI was measured on the leaderboard, so their −3.4 s already absorbs this; but anyone constructing the feature without noticing the January hole is walking into a trap.

`[S]` **OPDI cannot fill the unmatched-row gap.** Our `findings.md` R44 records that `AIRCRAFT_OPERATOR_flt`, `MARKET_SEGMENT_flt`, `FLIGHT_TYPE_flt`, `WK_TBL_CAT_flt` are null on **100 %** of unmatched rows. The obvious idea — recover aircraft type and operator from OPDI — fails on the exact-key join, because the join key **is** `CALLSIGN_flt`, which is the null column: of the 2,572 Jan-2026 ranking rows with null callsign, the exact join recovers **0**.

`[S]` I then tested the fallback: a nearest-time `join_asof` on `(ADEP_mvt, |first_seen − MVT_TIME| ≤ 120 s)`. On the 2,572 unmatched Jan-2026 rows it matches **940 (36.5 %)**, of which 841 carry a `typecode`. **But I validated its precision on Jan-2025 training rows where the true callsign is known, and it is not good enough:**
```
Jan 2025, rows with known callsign: 153,706 → asof-matched 148,942 (84.2 %)
callsign agreement among asof matches: 77,405 / 148,942 = 52.0 %
typecode agreement: 57.5 % of 141,891
```
**Half the nearest-time matches are the wrong flight.** At a busy airport, departures are minutes apart and nearest-in-time is close to a coin flip. This route needs a one-to-one assignment (Hungarian / greedy with mutual-nearest + icao24 corroboration) before it is worth anything. As a naive asof join it would inject noise.

`[S]` **Flight-events v0.0.2 does NOT ship the T04–T21 milestone model.** Reading `flight_events_20260110_20260120.parquet` (7,479,646 rows; schema `id, flight_id, type, event_time, longitude, latitude, altitude, source, version, info`), the actual `type` values are:
```
level-start 2,488,466 | level-end 1,775,493 | first_seen 316,875 | last_seen 316,875
first/last-xing-fl50/70/100/245 ~200k each | top-of-climb 193,229 | top-of-descent 192,535
exit-runway 164,140 | entry-runway 163,586 | entry-taxiway 102,277 | exit-taxiway 98,303
landing 23,486 | take-off 20,492 | entry-parking_position 16,959 | exit-parking_position 16,907
exit-apron 9,659 | entry-apron 9,658 | entry-threshold 2,756 | exit-threshold 2,756
exit-hangar 114 | entry-hangar 114 | entry-deicing_pad 81 | exit-deicing_pad 81
```
The surface events are **OSM-geometry crossings**. The `info` field is JSON and is genuinely rich:
```json
{"osm_id":542823806,"osm_aeroway":"parking_position","osm_ref":"V174","osm_airport":"EDDF",
 "opdi_time_in_use_s":5,"osn_icao24":"3c6753","osn_flight_id":"DLH1TN  "}
```
— i.e. **stand reference, airport ICAO, icao24 and callsign, all inline**, so events can be joined without the flight list.

`[S]` **But the yield is far too low to be a taxi-out measurement.** Joining events to the flight list and restricting to our 10 airports, over that 10-day window (≈30,600 departures):

| Airport | exit-parking_position | entry-runway | take-off | entry-deicing_pad |
|---|---|---|---|---|
| EDDF | 1,674 | 6,103 | 764 | 1 |
| LSZH | 4,355 | 4,478 | 62 | 0 |
| EHAM | 260 | 6,705 | 1,552 | 0 |
| LEBL | 118 | 3,329 | 847 | 0 |
| EGLL | 384 | 5,748 | 20 | 0 |
| LFPG | 192 | 3,241 | 53 | 0 |
| EDDM | 179 | 2,837 | 67 | 0 |
| LIRF | 118 | 1,677 | 22 | 0 |
| LTFM | 148 | 755 | 2 | 0 |
| LEMD | 112 | 1,215 | 38 | 0 |

Flights with **both** an `exit-parking_position` and a `take-off`: **232** out of ~30,600 (**0.8 %**), and the implied taxi-out is frequently negative (EHAM median −330 s, LEBL −5215 s), i.e. the pairing is unreliable even where it exists. Flights with `exit-parking_position` + `entry-runway`: 3,936 (~13 %), but `entry-runway` fires on runway *crossings* during taxi, so its semantics are ambiguous without further work.

**Conclusion: the tempting idea — "OPDI events give us a near-direct measurement of the 2026 target" — is false at v0.0.2.** `entry-deicing_pad` fires **once in ten days across all ten airports**, so the de-icing-detection idea is also dead. The only OPDI signal with real coverage is the flight-list turnaround gap that OpenAir already used, and it is worth ≈ −3.4 s to them.

### 4.4 Licence

`[S]` `https://github.com/euctrl-pru/OPDI` (the pipeline) is **MIT**, and the README credits OpenSky, OurAirports and Uber H3 but states **no licence for the output data**.
`[S]` https://www.opdi.aero/about says only: "Open data can be freely used, reused, and distributed provided that the data source is attributed" — a description, not a licence grant, and it links to https://ansperformance.eu/about/disclaimer/, which is restrictive (§6).

---

## 5. EUROCONTROL / PRU public datasets on ansperformance.eu

`[S]` Full catalogue with per-dataset coverage: https://ansperformance.eu/data/. The airport-level ones that matter for taxi-out, with the catalogue's stated coverage:

| Dataset | Granularity | Coverage per catalogue | URL (`https://www.eurocontrol.int/performance/data/download/xls/…`) |
|---|---|---|---|
| Airport Traffic | **daily**, per airport | Jan 2016 – Aug 2026 | `Airport_Traffic.xlsx` |
| Airport Arrival ATFM Delays | **daily**, per airport, by delay cause | Jan 2019 – Aug 2026 | `Airport_Arrival_ATFM_Delay.xlsx` |
| ATFM Slot Adherence | **daily**, per airport | Jan 2019 – Aug 2026 | `ATFM_Slot_Adherence.xlsx` |
| ATC Pre-Departure Delay | **daily**, per airport | Jan 2016 – **Jun 2026** | `ATC_Pre-Departure_Delay.xlsx` |
| Total Pre-Departure Delay | **daily**, per airport | Jan 2020 – **Jun 2026** | `All_Pre-Departure_Delay.xlsx` |
| Taxi-Out Additional Time (ATXOT) | monthly, per airport | Jan 2018 – **Jun 2026** | `Taxi-Out_Additional_Time.xlsx` |
| Taxi-In Additional Time | monthly, per airport | Jan 2018 – Jun 2026 | `Taxi-In_Additional_Time.xlsx` |
| ASMA Additional Time | monthly, per airport | Jan 2018 – Jun 2026 | `ASMA_Additional_Time.xlsx` |
| Taxi-Time Planning Values | per IATA season | S2014 – **S2022** → **FAIL** | `Taxi_times_Planning_Data_S14_S22.xlsx` |

`[S]` **All five large ones are live and were re-published on 2026-09-18** (HTTP `last-modified: Fri, 18 Sep 2026`). Sizes: Airport_Traffic 96.5 MB, ATC_Pre-Departure_Delay 100.3 MB, ATFM_Slot_Adherence 120.4 MB, Airport_Arrival_ATFM_Delay 97.5 MB, Taxi-Out_Additional_Time 1.37 MB.

`[S]` **The catalogue's end dates are stale — I checked the actual file.** Downloading `Taxi-Out_Additional_Time.xlsx` and reading sheet `DATA` (9,972 rows; columns `YEAR, MONTH_NUM, MONTH_MON, APT_ICAO, APT_NAME, STATE_NAME, TF, VALID_FL, NO_REF, TOTAL_REF_NB_FL, TOTAL_REF_TIME_MIN, TOTAL_ADD_TIME_MIN, COMMENT, PIVOT_LABEL`):
```
max month with non-null TF in 2026, per airport:
  EDDF 8, EGLL 8, EDDM 7, EHAM 7, LEBL 7, LEMD 7, LFPG 7, LIRF 7, LSZH 7, LTFM 7
  LTFM 2026 months present: [1,2,3,4,5,6,7]
```
**July 2026 is present for all ten airports.** So ATXOT covers both scored months despite the catalogue saying "Jun 2026". `[X]` The same is very likely true of `ATC_Pre-Departure_Delay.xlsx` and `All_Pre-Departure_Delay.xlsx` (same publisher, same 2026-09-18 refresh), but those files are ~100 MB and I did not download them, so **I mark their July-2026 coverage UNVERIFIED** rather than assume.

`[S]` The `META` sheet documents the definitions: `TOTAL_REF_TIME_MIN` = total reference taxi-out minutes, `TOTAL_ADD_TIME_MIN` = total additional taxi-out minutes, `Avg. Additional Taxi-Out Time = ROUND(TOTAL_ADD_TIME_MIN / TOTAL_REF_NB_FL, 2)`. So a per-(airport, month) additional-taxi-out level is one division away.

`[S]` What a competitor bought with it: ahmetabdullahgultekin used it **for validation only**, reporting **r = 0.757** between their reimplemented ATXOT and the official series, **0.87–0.98 at cold airports** — and reports **no model gain**. `[S]` They used the daily ATFM files as features and again report no delta.

`[X]` The strongest form of this source for us is not the monthly ATXOT (12 values per airport per year is nearly useless as a feature) but the **daily** `Airport_Arrival_ATFM_Delay` + `ATFM_Slot_Adherence` + `Airport_Traffic`, which give a per-(airport, day) regulation-pressure vector for both scored months.

`[S]` **EUROCONTROL Network Operations Report:** I could not retrieve it — https://www.eurocontrol.int/network-operations-report returns 404. The monthly ATFM statistics it summarises are the same daily files above, so nothing is lost.

---

## 6. Licence flags — sources that could cost the prize

The eligibility page demands that "**All additional datasets used are openly available under an open source license.**"

### 6.1 RED FLAG — every `eurocontrol.int/performance/data/download/xls/*` file, and by extension OPDI

`[S]` https://ansperformance.eu/about/disclaimer/, verbatim:

> "This data is published by EUROCONTROL for information purposes. It may be copied in whole or in part, provided that EUROCONTROL is mentioned as the source and **it is not used for commercial purposes (i.e. for financial gain)**. The information in this document **may not be modified without prior written permission from EUROCONTROL**."

This is **not an open-source licence** under any standard definition: it forbids commercial use and forbids modification. No CC licence is named. `opdi.aero/about` links to this same disclaimer.

**Consequences:**
- ahmetabdullahgultekin's register labels five EUROCONTROL files "EUROCONTROL public publication" — that is a description, not a licence, and it does not satisfy the clause. **Their prize eligibility is exposed.**
- OpenAir's OPDI-derived ground-time feature is exposed by the same logic.
- Any feature we derive from these files is a "modification" of the data.

`[X]` **Mitigating read:** the organiser *is* the EUROCONTROL PRU, they built OPDI, they brand it "open data", and OPDI is the flagship of their own open-data initiative. It is implausible that they would disqualify a team for using their own published open data. `[X]` The pragmatic position is: use it, declare it precisely, quote the attribution requirement, and **do not claim it is CC-BY when it is not**. If prize money is actually in reach, ask the organisers on the Discord (https://discord.gg/RPh89jpVVz) and get the answer in writing. **The risk is real but small, and it is a documentation risk, not a technical one.**

### 6.2 RED FLAG — raw OpenSky Network

`[S]` https://openskynetwork.github.io/opensky-api/trino.html: Trino/MinIO access is for "university-affiliated researchers, governmental organisations, and aviation authorities"; "**Private or commercial entities must contact us for a licence**"; access is granted "based on application review", and they "reserve the right to decline applications lacking sufficient justification". `[S]` The general terms are non-commercial research/education.
`[S]` Tables: `state_vectors_data4` (unlimited retention since 2013), `flights_data4` (since 2016), plus raw Mode-S tables. **Temporal coverage is not the problem — accessibility is.** A dataset you must apply for and be approved to touch is not "openly accessible/usable", and the licence is not open-source.
**Verdict: do not use raw OpenSky directly. Use OPDI, which is OpenSky already processed and published.** `[S]` OpenAir reached the same practical conclusion for a different reason (`first_seen` is lift-off).

### 6.3 AMBER — OpenStreetMap aeroway geometry

`[S]` ODbL (stated by ahmetabdullahgultekin's register, who listed it as a candidate and did not implement it). ODbL **is** an open licence and passes the eligibility clause, but it is **share-alike on derived databases**. `[X]` A taxi-distance table built from OSM polygons is arguably a derived database and would carry an ODbL obligation onto that table — which is compatible with publishing it under ODbL alongside GPLv3 code, but it is a second licence in the repo. Note also that **OPDI's surface events are themselves OSM-derived** (`info.osm_id`, `info.osm_aeroway`), so this obligation may already be riding along with §4.

### 6.4 AMBER — Open-Meteo

`[S]` Data is CC BY 4.0 (passes). The **free API tier** is restricted to non-commercial use with a 10,000 calls/day cap. `[X]` A competition entry is non-commercial; the constraint is on service usage, not on the data licence. Fine, but attribute it.

### 6.5 CLEAR — IEM ASOS, OurAirports, OpenAP

`[S]` IEM: "in the public domain and may be used freely by any lawful purpose." OurAirports: "All data is released to the Public Domain and comes with no guarantee of accuracy." OpenAP: **LGPL-3.0** for code and data — GPLv3-compatible. All three pass cleanly.

### 6.6 UNVERIFIED — ICAO Doc 8643, EUROCONTROL APD, OpenFlights, NOTAMs

`[S]` icao.int/publications/DOC8643 returned **403**; openflights.org/data.html returned **404**; contentzone.eurocontrol.int/aircraftperformance redirects to learningzone.eurocontrol.int which I did not follow. **I cannot state the licence or the bulk-download availability of any of these.** `[X]` ICAO Doc 8643 is ICAO-copyright and has no open bulk distribution I am aware of; the practical open substitute is OpenAP (LGPL-3.0) or the `typecode`/`icao_aircraft_class` columns already inside the OPDI flight list.

`[X]` European NOTAMs come through EUROCONTROL EAD, which requires registration and is not open; airport capacity declarations are published by slot coordinators (ACL for the UK airports, Airport Coordination Germany, etc.) as **seasonal PDFs**, not machine-readable open data. Runway configuration schedules are not published as data for any of the 10 airports as far as I could find. **I verified none of this directly — treat the whole category as unavailable rather than as investigated.** Note that OPDI's `entry-runway` events carry `info.osm_ref`, which `[X]` is the nearest open proxy for "which runway was actually in use", and our own data already has `RUNWAY_mvt`.

### 6.7 Aircraft type / wake / MTOW — already in hand

`[S]` The competition file already supplies `AIRCRAFT_TYPE_mvt`, `AIRCRAFT_TYPE_flt` and `WK_TBL_CAT_flt`. `[S]` The OPDI flight list adds `typecode`, `model`, `icao_aircraft_class`, `registration`, `icao_operator`. `[S]` OpenAP (LGPL-3.0) supplies MTOW and engine properties per type. **There is no reason to touch a closed aircraft database.** `[S]` R44 in our own `findings.md` shows the gap is not the type table — it is that the type/operator columns are 100 % null on unmatched rows, and §4.3 shows OPDI does not fix that.

### 6.8 Airline / route reference

`[X]` `AIRCRAFT_OPERATOR_flt` and `MARKET_SEGMENT_flt` are already in the file for matched rows. OpenFlights routes were last refreshed years ago (I could not verify the date — page 404) and would not describe 2026 schedules. **No open, 2026-current route reference was found.** Treat this category as empty.

---

## 7. What our repo already has

`[S]` `src/prc/weather.py` — IEM ASOS is already wired in, fields `tmpf, dwpf, relh, drct, sknt, gust, p01i, vsby, wxcodes, skyc1, feel`, `report_type=3`, 1.2 s throttle, cached at `data/ref/metar/{ICAO}_2025-01-01_2026-08-01.csv` for all 10 airports plus `data/ref/metar_hourly.parquet`. **The cache window already spans both scored months.** The docstring names Open-Meteo as the "secondary source (gap filling / snowfall), CC BY 4.0" but the module does not implement it.
`[S]` `src/prc/airports.py` — OurAirports `airports.csv` already fetched, used for great-circle distance to `ADES_mvt`.
`[S]` Neither OPDI nor any EUROCONTROL PRU file is referenced anywhere in `src/` or `configs/`.
`[S]` Measured weather gain already on record (`findings.md`, paired on 328,009 forward-fold rows, no-weather → weather): matched gate rows **344.99 → 336.99** (mean residual −61.0 → −8.7), matched overall **203.35 → 202.48**, unmatched **2153.33 → 2116.25**.
`[S]` And the exposure is concentrated exactly where it matters: ranking gate rows **Jan 2026 = 6,729 vs Jan 2025 = 2,505** (×2.70); at EDDF one ranking month has more gate rows than the entire 2025 training year (1,223 vs 617), at LSZH 1,379 vs 1,119; **Jul 2025 and Jul 2026 have 0 gate hours at every station.**

---

## Applicable to resilient-hedgehog

Ranked by (evidence of gain) × (2026 coverage confirmed) × (ease of joining).

**1. Re-measure the existing IEM weather join with the fixed join — and stop trying to fix `p01i`.** `[S]` Gain already demonstrated (matched gate rows −8.0 s; ×2.70 more gate-hour exposure in scored January than in training January). `[S]` Coverage confirmed for both scored months by live probe, cache already on disk covering 2025-01-01 → 2026-08-01. Join already written. Licence clean (public domain). **And the open question in `findings.md` §5 is now answered: `p01i` is empty because IEM states "precipitation data is unavailable for non-US sites" — it is not a parser bug.** Zero download cost, highest expected value, and it removes a known blocker. Do this first.

**2. Add Open-Meteo `snowfall` / `precipitation` for the 10 airport coordinates.** `[S]` Covers exactly the field IEM withholds for European stations, which is the physical driver of the January de-icing effect our own `seasonal-deicing-shift.md` isolates. `[S]` Coverage verified through 2026-09-19 with a 2-day lag; both scored months inside. `[S]` CC BY 4.0 — clean. `[S]` The module docstring already names it, so the integration point exists. 10 airports × 19 months of hourly data is a handful of API calls. Cheap, licensed, and targeted at the one residual our own evidence says is live. `[X]` No competitor has published a number for this; the gain is inferred from our own gate-row residuals, not measured.

**3. EUROCONTROL daily airport files — `Airport_Arrival_ATFM_Delay`, `ATFM_Slot_Adherence`, `Airport_Traffic` — as per-(airport, day) regulation pressure.** `[S]` All three re-published 2026-09-18 and catalogued through Aug 2026; ATXOT verified by me to contain July 2026 for all ten airports, which is strong evidence the sibling files do too (`[X]` unverified for the two pre-departure-delay files, which stop at Jun 2026 in the catalogue — **check before building on those two**). Join is trivial: ICAO + date. `[S]` **But no one has reported a gain from them**: ahmetabdullahgultekin used them and published only a validation correlation, and OpenAir's own falsification list says rolling-queue/congestion features did not shrink the tail. `[S]` **Licence flag: NC + no-modification, not an open-source licence (§6.1).** Worth one experiment, not a programme; declare it precisely if it ships.

**4. OPDI flight-list turnaround gap (`first_seen − previous last_seen` per `icao24`).** `[S]` The **only** quantified external-data gain in the 2026 public record: OpenAir 281.72 → 278.36, **≈ −3.4 s**. `[S]` Both scored months downloadable (Jan 20 MB, Jul 53 MB). `[S]` Join is `(CALLSIGN_flt, ADEP_mvt, date)` after `str.strip()` on `flt_id`, with a dtype branch for 2025-vs-2026 files. **`[S]` The trap, measured here: the join rate is 85.2 % in Jan 2025 training but 58.9 % in Jan 2026 and 91.2 % in Jul 2026.** Build the feature with an explicit missingness indicator and validate per scored month, or January will punish you. `[S]` Licence flag as §6.1. Ranked 4th, not higher, because it costs a real integration for ~3 s that a rival already banked and that carries a covariate-shift hazard.

**5. Do NOT pursue OPDI flight-events as a taxi-out measurement.** `[S]` Measured, 10-day January 2026 window, all 10 airports: only **232 flights of ~30,600 (0.8 %)** have both an `exit-parking_position` and a `take-off` event, and the implied durations are frequently negative. `[S]` `entry-deicing_pad` fires **once** in ten days across all ten airports. `[S]` Cost is ~272 MB per 10-day window (~10–12 GB for 2025 + both scored months). **This is the expensive-looking idea that does not work; recording it here so nobody spends a day on it.** The one salvageable crumb: `info.osm_ref` gives a stand reference and `info.osm_airport` the ICAO, so `exit-parking_position` could enrich `STAND_mvt` at **LSZH (4,355 events in 10 days) and EDDF (1,674)** specifically — `[X]` speculative, and only those two airports have the density.

**6. Do NOT use Meteostat.** `[S]` Every one of seven European stations probed stops at **2026-03-29/30**. It covers January 2026 and not July 2026 — a feature that is systematically absent from 56 % of the scored rows is worse than no feature. `[X]` Its licence (CC BY-NC) would likely fail the eligibility clause as well.

**7. Do NOT use raw OpenSky.** `[S]` Access is by application, restricted to academic/governmental applicants, non-commercial — it fails "openly accessible/usable" on its face. `[S]` OpenAir independently found `first_seen` is lift-off, not a parking event, so it would not give off-block times anyway. OPDI is the licensed, pre-processed path to the same data.

**8. Treat NOTAMs, capacity declarations and runway-configuration schedules as unavailable.** `[X]` Not verified, but EAD requires registration, slot declarations are seasonal PDFs, and no open runway-config feed exists for these 10 airports. `[S]` `RUNWAY_mvt` is already in the file.

**One cross-cutting caution.** `[S]` In PRC 2025 — the most recent edition, same organisers, same metric family — **the entire prize podium used no external data**, and the one team that reported testing weather reported "limited influence". `[S]` In PRC 2026 the three strongest public repos (Phoenix-Ops 273.56, javidmardanov 268.56, sergiuv11 286.66) credit **zero** external datasets with a gain; every published gain comes from restructuring the supplied movement/NM tables. `[S]` The largest single documented external gain in this whole slot is **≈ −3.4 s** (OpenAir/OPDI), against our own gap of 305.03 → 243.58 at the top of the board. **External data is item 1–2 on the list because our own weather measurement is broken and cheap to fix, not because the external-data category is where the 60 seconds live.** It is not.
