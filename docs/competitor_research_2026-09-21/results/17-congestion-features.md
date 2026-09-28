# Slot 17 — congestion / queue / demand feature catalogue

Investigator: SubAgent. Date: 2026-09-21.
Marking: **[S]** = read from a named source (URL or file path inside a public repo). **[X]** = my own
extrapolation, never presented as fact.

All repository paths below are from public repos that I cloned and read at the code level:

| short name | repo | what it is |
|---|---|---|
| `NASA/` | `github.com/drivendataorg/nasa-airport-pushback` | the five prize winners of NASA "Pushback to the Future" |
| `OpenAir/` | `github.com/ChandanHegde07/OpenAir` | PRC 2026 public competitor |
| `Phoenix/` | `github.com/Phoenix-Ops-LTD/prc2026-taxiout` | PRC 2026 public competitor, team `zestful-fountain` |
| `Javid/` | `github.com/javidmardanov/prc-data-challenge-2026` | PRC 2026 public competitor, team `elegant-alligator` |

---

## 0. Ground facts about our own data that drive every leakage judgement

Measured directly on `data/raw/ranking.parquet` with polars during this investigation (no code modified,
read-only query). **[S]** — reproducible with a one-line polars scan:

| phase | rows | `BLOCK_TIME_UTC_mvt` null | `TAXITIME_SEC_mvt` null | `MVT_TIME_UTC_mvt` null | `AOBT_3_flt` null | `RUNWAY_mvt` null | `STAND_mvt` null |
|---|---:|---:|---:|---:|---:|---:|---:|
| ARR | 344,693 | **0** | **0** | 0 | 2,552 | 0 | 5 |
| DEP | 344,841 | **344,841 (100%)** | **344,841 (100%)** | 0 | 5,290 (1.53%) | 0 | 4 |

Ranking months: **2026-01 (305,590 rows) and 2026-07 (383,944 rows)** — two disconnected months, no
December/February context on either side. **[S]**

Four consequences that decide the whole leakage section:

1. **Every scored departure's block time and label are blank.** Any feature that aggregates *other
   departures' `TAXITIME_SEC_mvt` or `BLOCK_TIME_UTC_mvt`* is **uncomputable** at prediction time, not
   merely "risky". **[S]**
2. **Arrival rows in ranking are complete**, including `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt`. Taxi-**in**
   labels of neighbours *are* legitimately available. **[S]**
3. **`AOBT_3_flt` survives in ranking on 98.47% of departures.** `MVT − AOBT_3` is therefore a *reconstructable
   per-flight taxi duration proxy for every other flight*, and is the single legal substitute for
   "rolling actual taxi-out of neighbours". **[S]**
4. **Ranking appears to be the complete movement stream for those two months**, not a sample (305,590 ranking
   rows in Jan-2026 vs 307,257 training rows in Jan-2025 — `docs/data_profile.md` §1). Rolling counts computed
   on ranking are therefore on the same scale as those computed on training; there is no density
   discontinuity to correct. **[S]**

Relevant existing internal facts (`docs/data_profile.md`): `MVT − AOBT_3` predicts the target at RMSE
**384.9 s** against a global-mean RMSE of 546.4 s; airport×runway×stand mean is 469.1 s. **[S]**

---

## 1. The physical model the literature agrees on

- **Idris, Clarke, Bhuva & Kang (2002), "Queuing model for taxi-out time estimation", *Air Traffic Control
  Quarterly* 10(1):1–22.** The takeoff queue size *Q* is the dominant explanatory variable. Correlation of
  taxi-out with *Q* (takeoff queue size) versus *N* (number of departure aircraft present on the airport
  surface at pushback) improves from **R² = 0.19 to R² = 0.59**. **[S]** — source: search summary of
  `dspace.mit.edu/handle/1721.1/37322` ("Queuing Model for Taxi-Out Time Estimation"); I could not retrieve the
  primary PDF (MIT DSpace served a human-verification page and returned 405 to the fetcher), so treat the exact
  numbers as second-hand-but-named rather than read-from-the-PDF.
- **Herrema, Curran, Visser, Huet & Lacote (2018), "Taxi-Out Time Prediction Model at Charles de Gaulle
  Airport", *J. Aerospace Information Systems*, doi:10.2514/1.I010502.** Regression tree was the most efficient
  method, average error **1.6 minutes**; the top features out of 42 candidates include **unimpeded taxi time,
  congestion level, and number of departures in the last 20 minutes**. **[S]** — abstract/author list read from
  `research.tudelft.nl/en/publications/taxi-out-time-prediction-model-at-charles-de-gaulle-airport`; the
  feature list is from the search-engine summary of the same paper, not from the paywalled full text.
- **Lee & Malik (NASA), "Taxi-Out Time Prediction for Departures at Charlotte Airport Using Machine Learning
  Techniques", AIAA 2016-3910.** Feature list, verbatim: *"terminal concourse, gate, spot, runway, departure
  fix, aircraft model, weight class, taxi distance from gate to runway, scheduled pushback time of day,
  **number of departures on the surface by runway, number of arrivals on the surface by concourse**, and
  unimpeded taxi time."* Unimpeded taxi time is defined as *"the 10th percentile of the taxi time
  distributions … for aircraft with the same gate, spot and runway combination"*. The counts are deliberately
  **partitioned**: departures by assigned runway, arrivals by concourse of the assigned gate. **[S]**
- **EUROCONTROL PRU, "Additional taxi-out time performance indicator document", Ed. 01.00, 2023.** The official
  European unimpeded reference: group flights by **STAND × departure-runway combo**, take the **10th percentile
  of actual taxi-out times over a rolling 12 months** ending in the month under analysis, and only accept the
  reference if **at least 10 flights in the sample have taxi-out ≤ that P10**; otherwise the combo gets *no*
  reference. Additional taxi-out time = actual − reference. **[S]**
- **Vargo, Tien & Jafari (MITRE), "Airport Taxi Time Prediction and Alerting: A CNN Approach", arXiv:2111.09139.**
  Tarmac metadata, verbatim, as a 30-slice one-minute time series over the previous 30 minutes: *average
  taxi-in time of all aircraft currently on the tarmac; average taxi-out time of all aircraft currently on the
  tarmac; current arrival demand (frozen at top of hour); current departure demand (frozen at top of hour);
  current arrivals since top of hour, updated each minute; current departures since top of hour, updated each
  minute; **arrival demand +1h and +2h**; **departure demand +1h and +2h***. **[S]** — this is the canonical
  published precedent for *forward-looking* demand counts.
- **"Interpretable Taxi-Out Time Prediction of Departure Flights Using Stacking Ensemble Learning and SHAP
  Analysis", *Scientific Reports* (2026), s41598-026-40898-5.** Shenzhen Bao'an, 5,952 departures over 14 days.
  Features and SHAP shares: **arrival queue ≈ 18%**, **departure queue ≈ 18%**, airline category 9%,
  30-min average taxi time ≈ 7%, pre-pushback 30-min average ≈ 6%, taxi distance 6.2%. RMSE 140.5 s,
  MAE 99.7 s. Definitions as reported: departure queue = *"count of departing flights j that complete takeoff
  **during the taxi-out process**"*; arrival queue = *"count of arriving flights j that complete landing during
  the taxi-out process"*. **[S]**
  **Critical caveat:** those two top features are **endogenous** — the counting window is
  `[own pushback, own takeoff]`, whose *length is the target*. A longer taxi mechanically admits more events.
  This is the single most common trap in the literature and is called out independently below.

---

## 2. Exhaustive catalogue with exact definitions

Notation: query flight *i* at airport *a*; `t` = reference instant; `W` = window length. "Self-exclusion"
states whether row *i*'s own event is removed from the count.

### F1 — Trailing movement counts (before the query time)

| ID | definition | reference `t` | window | key | self | nulls / sparse | source |
|---|---|---|---|---|---|---|---|
| F1a `dep_prev_Wm` | count of DEP rows at same airport with `MVT_TIME ∈ (t−W, t]`, minus 1 | own `MVT_TIME` | W ∈ {1,5,10,15,30,60} min | airport | yes (−1) | count, never null | `OpenAir/analysis/03_congestion_state.py::add_rolling_counts` **[S]** |
| F1b `arr_prev_Wm` | count of ARR rows at same airport with landing `MVT_TIME ∈ (t−W, t]` | own `MVT_TIME` | same | airport | n/a (different phase) | 0 if none | same **[S]** |
| F1c `mov_prev_Wm` | `dep_prev_Wm + arr_prev_Wm` | own `MVT_TIME` | 5/15/60 | airport | yes | — | same **[S]** |
| F1d `DEP_count_-1_W` / `ARR_count_-1_W` | `searchsorted(events, t, 'left') − searchsorted(events, t−W, 'left')`, i.e. `[t−W, t)`, self at exactly `t` excluded by `side='left'` | own `MVT_TIME` | **W ∈ {900, 1800, 3600, 7200} s** | airport (arrivals keyed on `ADES_mvt`) | yes, by half-open interval | count | `Javid/scripts/train.py::features` **[S]** |
| F1e `airport_dep_previous_{15,30,60}m` | same, "exact-time events are excluded from trailing counts" | own `MVT_TIME` | 15/30/60 min | airport | yes | count | `Phoenix/traffic_features.py::movement_context` **[S]** |
| F1f `deps_nhr` / `arrs_nhr` | number of flights that departed / arrived at the airport in the past *n* hours, computed from the **runways** table filtered to `timestamp ∈ (now − n h, now]` | prediction timestamp | **3 h and 30 h** | airport | not excluded (constant per timestamp) | count | `NASA/4th Place/Phase 1/src/table_scripts/add_traffic.py` + KDD Tab. 3 **[S]** |
| F1g `runways_{15,30,60,90,120,360,720,1440}_totdep` / `_totarr` | rolling count of rows with a non-null actual departure (resp. arrival) runway time, evaluated on a **15-minute grid** of timestamps, `pandas.rolling(f"{p}min")` | 15-min grid instant | **8 windows, 15 min → 24 h** | airport | grid-level, self not excluded | `.count()`, never null | `NASA/1st Place/Phase 1/training/src/pushback_nasa/pipelines/feature_extraction/nodes.py::extract_runway_features` **[S]** |
| F1h `num_flights` | distinct GUFIs on the runway table in the **preceding clock hour**, joined by `date_hour` string | hour bucket | 1 h | airport | no | left-join → NaN if hour absent | `NASA/5th Place/Phase 1/src/features.py::get_num_dep_flights` **[S]** |
| F1i `current departures` | number of departures **since the top of the hour**, refreshed every minute, as one channel of a 30-minute time series | minute | expanding within hour | airport | no | — | MITRE arXiv:2111.09139 **[S]** |

**Resolutions that are actually reported in use:** 1, 5, 10, 15, 30, 60 min (OpenAir); 15, 30, 60, 120 min
(Javid, in seconds: 900/1800/3600/7200); 15/30/60 min (Phoenix); 15, 30, 60, 90, 120, 360, 720, 1440 min
(NASA 1st); 3 h and 30 h (NASA 4th); 1 h and 2 h ahead (MITRE). **[S]**

### F2 — Following counts (windows strictly AFTER the query time)

| ID | definition | window | key | self | source |
|---|---|---|---|---|---|
| F2a `DEP_count_+1_W` / `ARR_count_+1_W` | `searchsorted(events, t+W, 'left') − searchsorted(events, t, 'left')` = `[t, t+W)`; note this **includes** the query's own event at `t` (a constant +1 for DEP) | W ∈ {900, 1800, 3600, 7200} s | airport | **no** | `Javid/scripts/train.py::features` **[S]** |
| F2b `exp_deps_15min`, `exp_deps_30min` | count of flights whose **latest estimated departure-runway time** lies in `[ETD_i − W, ETD_i + W]` — i.e. a **symmetric ±W window centred on the query flight's own ETD**, not "the next W minutes" | W ∈ {15, 30} min | airport | **no** (own row is in the window) | `NASA/4th Place/.../add_traffic.py::count_expected_departures` **[S]** |
| F2c `nm_following_{airport,runway,stand}_{15,60}m_{count,mean,std,own_minus_mean}` | pool = DEP rows with matching `ADEP_mvt == ADEP_flt` and `proxy = MVT − AOBT_3 ∈ [0, 7200] s`; window is **strictly after and upper-inclusive: `(t, t+W]`**; per group: count, mean of `proxy`, population std, and `own proxy − mean` | W ∈ {15, 60} min | airport / airport×runway / airport×stand, **scoped within the same UTC calendar month** | yes — query itself *and every movement tied at the exact query instant* excluded | `Phoenix/nm_following_features.py` **[S]** |
| F2d `nm_following_*_next_wait_seconds`, `*_next_proxy` | time from `t` to the **nearest strictly later** departure movement in the group (capped at 7,200 s), and that movement's proxy (ties at that timestamp averaged) | — | same three groups | yes | same **[S]** |
| F2e `arrival/departure demand +1 h, +2 h` | projected demand over the next one and two hours, starting from the top of those hours, updated each minute | 1 h, 2 h | airport | no | MITRE arXiv:2111.09139 **[S]** |
| F2f `runway_next_gap` | seconds from `t` to the **next** departure on the same runway | — | airport×runway | yes | `Javid/scripts/train.py::features` **[S]** |

**Why "after the query time" is legitimate here [S]:** Phoenix's own `nm_following_features.py` docstring states
*"These observations do not establish availability before departure"*, and `METHOD.md` calls them *"retrospective
ranking context, not evidence of operational availability before departure."* Those features are legal because
the ranking file hands us the entire two-month movement stream at once; they are **batch-only** and would be
unavailable in a live system. The same is true of Javid's `direction=+1` counts. Phoenix's `METHOD.md` records
the v12 result from exactly these 30 features: **273.5557 official, improving v11 by 1.4181 s**, moving
12/131. **[S]**

### F3 — Same-runway counts, headways and gaps

| ID | definition | source |
|---|---|---|
| F3a `dep_rwy_prev_Wm` | DEP count with same `RUNWAY_mvt` and `MVT_TIME ∈ (t−W, t]`, minus 1, W ∈ {1,5,10,15,30,60} min | `OpenAir/analysis/03_congestion_state.py` **[S]** |
| F3b `runway_{dep,arr}_previous_{15,30,60}m` | same idea keyed on `(airport, runway)` | `Phoenix/traffic_features.py` **[S]** |
| F3c `runway_{dep,arr}_headway_seconds` | `t − (previous same-runway same-phase movement time)`, **clipped to 3,600 s**, and **3,600 when no previous event exists** (an explicit sentinel, not NaN) | `Phoenix/traffic_features.py` **[S]** |
| F3d `runway_prev_gap`, `runway_next_gap` | seconds to previous / next departure on the same runway; at the array ends the neighbour index is clamped, so the value degenerates to 0 | `Javid/scripts/train.py` **[S]** |
| F3e `runway_count_30min` | same-runway departures in a **centred** window `[t−900, t+900]` (name says 30 min, code is ±15 min) | `Javid/scripts/train.py` **[S]** |
| F3f `dep20_time` | minutes elapsed over the **last 20 actual departures** at the airport — `dep_time − dep_time.shift(20)` — merged backwards onto the query timestamp. Inverse throughput rather than a count. | `NASA/2nd Place/Phase 1/src/make_dataset.py::merge_runway_features` **[S]** |
| F3g `departure_runways` / `arrival_runways` | the *set* of runway names currently in use for departures / arrivals, as a categorical | `NASA/4th Place` KDD Tab. 3 **[S]** |
| F3h `number of departures on the surface **by runway**` | departures currently taxiing, partitioned by assigned runway | Lee & Malik, AIAA 2016-3910 **[S]** |

### F4 — Same-stand-area counts

| ID | definition | source |
|---|---|---|
| F4a `nm_neighbor_stand_*` / `nm_following_stand_*` | the F2c/F5-style statistics computed on the `airport × stand` group; **a missing stand does not form an observed group** (rows with `STAND` unknown get NaN rather than joining a giant `UNKNOWN` bucket) | `Phoenix/nm_neighbor_features.py`, `nm_following_features.py` **[S]** |
| F4b `number of arrivals on the surface **by concourse**` | arrivals currently taxiing in, partitioned by the concourse containing the assigned gate | Lee & Malik, AIAA 2016-3910 **[S]** |
| F4c `standtime_nhr` | average difference between "first position" (GUFI creation) and pushback, over all flights at the airport in the past *n* hours | `NASA/4th Place/.../add_averages.py::average_stand_time` + KDD Tab. 3 **[S]** |
| F4d `airport_stand`, `stand_prefix`, `stand_number`, `stand_runway` | stand identity decomposed so trees can generalise across an apron rather than memorise 1,899 stands | `Javid/scripts/train.py`, `Phoenix/traffic_features.py` **[S]** |

I found **no** source that counts *simultaneous occupancy of a stand area* (e.g. "other aircraft pushing back
within 3 stands of mine"). That specific feature is absent from every catalogue I read. **[X]** that it is
therefore unexplored rather than known-useless.

### F5 — Aircraft currently on the ground / taxiing

| ID | exact definition | self | nulls | source |
|---|---|---|---|---|
| F5a `deps_taxiing` | inner-join MFS(departures) with **standtimes** (so only flights with an actual pushback), left-join **runways**, keep rows where `departure_runway_actual_time` **is NaN** → flights that have pushed back and have no recorded takeoff. Count them. Scope is the whole 30-hour data window, not a rolling window. | not excluded | count | `NASA/4th Place/.../add_traffic.py::count_planes_taxiing(flights="departures")` **[S]** |
| F5b `arrs_taxiing` | runway rows with a valid `arrival_runway_actual_time`, left-joined to standtimes, keep those with **NaN `arrival_stand_actual_time`** → landed but not yet at the gate. | not excluded | count | same **[S]** |
| F5c `average taxi-in / taxi-out time of all aircraft **currently on the tarmac**` | per-minute channel over the preceding 30 min | — | — | MITRE arXiv:2111.09139 **[S]** |
| F5d `airport_queue_at_aobt` | on the (start, end) = (`AOBT_3`, `MVT_TIME`) intervals of all departures at the airport, `#(starts ≤ own AOBT) − #(ends ≤ own AOBT)`, then **minus 1 if the query's own interval is valid** → number of *other* aircraft between off-block and takeoff at the instant this flight pushes back. This is Idris's **N**. | **yes, explicit −1** | NaN when own `AOBT_3` missing | `Javid/scripts/queue_features.py::features` + `Javid/docs/queue_features.json` **[S]** |
| F5e `runway_queue_at_aobt` | same, restricted to `(ADEP_mvt, RUNWAY_mvt)` | yes | NaN | same **[S]** |

`Javid/docs/queue_features.json` records the interval admission rule verbatim: source intervals accepted only
for `duration ∈ (0, 21600] s` (lower bound exclusive), `2,400,891` of `2,429,888` rows valid, and
*"Queue at missing AOBT is NaN; empty neighbor means are NaN with count zero."* **[S]**

### F6 — The physical queue: pushed back but not yet airborne

| ID | exact definition | self | source |
|---|---|---|---|
| F6a `queue_at_mvt` | count of other departures *j* at the same airport with `AOBT_j < MVT_i ≤ MVT_j` — i.e. started taxi and not yet airborne **at the instant this flight takes off**. Implemented as `searchsorted(sorted AOBT, t, 'left') − searchsorted(sorted MVT, t, 'left') − 1`, floored at 0. | yes (`−1`) | `OpenAir/analysis/03_congestion_state.py::add_surface_proxy` **[S]** |
| F6b `airport_queue_at_takeoff`, `runway_queue_at_takeoff` | identical construction at `t = own MVT_TIME`; the interval is half-open so *"the interval ends at takeoff, so this count already excludes itself"* (code comment) | yes, structurally | `Javid/scripts/queue_features.py` **[S]** |
| F6c `push_prev_{1,5,15,30}m` | count of other departures whose `AOBT_3 ∈ (t−W, t]` — **pushbacks in the last W minutes** rather than a standing queue | not excluded in that loop | `OpenAir/analysis/03_congestion_state.py::add_surface_proxy` **[S]** |
| F6d "virtual departure queue" | ASDE-X-derived running queue, then linear regression on it | literature reference in MITRE arXiv:2111.09139 §I **[S]** |

**Anchor choice matters and is disputed.** `queue_at_aobt` (F5d) is evaluated at *pushback*, which is the
operationally correct instant (it is the state the pilot enters). `queue_at_takeoff` (F6a/F6b) is evaluated at
*takeoff*, which is later by exactly the target. Both OpenAir and Javid compute the takeoff-anchored version
anyway, because `MVT_TIME` is the one timestamp that is never missing in ranking. **[S]**

### F7 — Queue position / rank

| ID | exact definition | outcome | source |
|---|---|---|---|
| F7a ETD rank queue | bucket every flight's latest ETD to 15-minute groups; for each 15-min data-time group, `rank()` the flights by `departure_runway_estimated_time`; look the flight's own rank up. Built by replaying the estimate history at 22 successive 15-minute lags. | **"Queue feature ended up not being used in final model"** (verbatim code comment) | `NASA/2nd Place/Phase 1/src/make_dataset.py::get_queue_feature` **[S]** |
| F7b "length of the queue ahead of each flight, i.e. the number of flights scheduled to depart before a given airplane" | — | listed by the 2nd-place team under *"things that did not make it into the final workflow"* | `NASA/2nd Place/Phase 1/reports/DrivenData-Competition-Winner-Documentation.pdf` Q11 **[S]** |
| F7c `queue_ahead` | departures at the same airport between the flight's scheduled departure and its own takeoff | **rejected** in our own repo, `−0.16 [−0.40, +0.08]` vs a noise column | `docs/features_log.md` #1 **[S]** |
| F7d `bank_position` | position within a departure bank, 0 = head, 1 = tail | **rejected**, `−0.19 [−0.41, +0.07]` | `docs/features_log.md` #2 **[S]** |

**Every rank/position feature I found in any source was tried and dropped.** No source reports a positive
result for queue *position* as opposed to queue *size*. **[S]**

### F8 — Rolling neighbour taxi durations (the "recent taxi" family)

This is the family that actually earns money in the 2026 competition, and it is distinct from counting.

| ID | exact definition | window shape | self | nulls | source |
|---|---|---|---|---|---|
| F8a `dep_taxi_nhr` | average `departure_runway_actual_time − departure_stand_actual_time` over all departing flights in the past *n* hours (n = 30) | trailing | no | 0 substituted for NaN (`if math.isnan(x): x = 0`) | `NASA/4th Place/.../add_averages.py::average_taxi_time` **[S]** |
| F8b `arr_taxi_nhr` | same for arrivals: `arrival_stand_actual_time − arrival_runway_actual_time` | trailing 30 h | no | 0 for NaN | same **[S]** |
| F8c `taxi_times_mean/std` (3rd place) | mean and std of taxi time over standtime records in `[t − L, t)`, **L = 120 min, expanded by 60 min up to 180 min until at least 4 records are found** | trailing, adaptive | no | adaptive widening, then NaN | `NASA/3rd Place/Phase 1/helper.py::extract_combined_vals_taxi_error_estpush_v2` **[S]** |
| F8d roll10 `MVT − AOBT` | rolling mean of the previous 10 departures' `MVT − AOBT_3` at the airport, and at the runway | trailing, **count-based not time-based** | yes | — | `OpenAir/analysis/DISCOVERY_REPORT.md` §6, items 6–7 **[S]** |
| F8e `nm_neighbor_{airport,runway,stand}_{15,60}m_{count,mean,std,own_minus_mean}` + `last_age_seconds`, `last_proxy` | pool restricted to `ADEP_mvt == ADEP_flt` and `proxy = MVT − AOBT_3 ∈ [0,7200] s`; window `[t−W, t)` via `side='left'`, so the query row and simultaneous movements are excluded; grouped within the same UTC month; `last_proxy` averages all events tied at the most recent timestamp, `last_age_seconds` NaN if the last event is > 7,200 s old | trailing | yes | NaN with count 0 | `Phoenix/nm_neighbor_features.py` **[S]** |
| F8f `{airport,runway}_proxy_taxi_{mean,count}_{1800,7200}` | **symmetric/centred** window `[t − W, t + W]` with W ∈ {1800, 7200} s on the `MVT − AOBT_3` proxy, own value subtracted from both numerator and count | **centred** | yes, arithmetically | NaN mean, count 0 | `Javid/scripts/queue_features.py::neighbors` **[S]** |
| F8g `arr_taxi_mean_{1800,7200,21600}` | **centred** window mean of *arrival* `TAXITIME_SEC_mvt` (a real label, legal because ranking arrivals are complete) at the same airport | centred ±W, W ∈ {30 min, 2 h, 6 h} | no | denominator `max(1, n)` → 0 if window empty | `Javid/scripts/train.py::features` **[S]** |

### F9 — Peer lateness / disruption state

| ID | exact definition | source |
|---|---|---|
| F9a `dis_state_{15,30,60}m` | **mean over *other* departures at the same airport whose `AOBT_3` falls in `(t−W, t]` of that flight's `AOBT_3 − EOBT_1`** (lateness versus plan). Reference `t` = the query's own `MVT_TIME`. Only NM-matched flights with finite `aobt_eobt` contribute. **Self is explicitly subtracted** from both the sum and the count when the query's own `AOBT_3` falls inside the window. The mean is NaN when the surviving count `< 1`; counts are floored at 0. | `OpenAir/experiments/run_e16a.py::add_push_disruption`, `_window_from_prefix` **[S]** |
| F9b `dis_frac10_30m`, `dis_frac20_{15,30,60}m`, `dis_frac30_{30,60}m` | fraction of those peers whose lateness exceeds 600 / 1,200 / 1,800 s | same **[S]** |
| F9c `dis_n_{15,30,60}m`, `dis_n_dly20_{30,60}m` | the raw peer counts and the count of peers > 1,200 s late | same **[S]** |
| F9d `1hr_ETDP` / `delay_nhr` | average departure delay = average difference between estimated and true departure time over all flights at the airport in the past hour / past *n* hours | `NASA/4th Place` KDD Tab. 3 **[S]** |
| F9e `standtimes_maxdelay_{1,2,4,6,12,24}`, `standtimes_meandelay_{p}`, `standtimes_count_{p}` | `delay = actual departure-stand time − first ETD ever issued for that flight`; rolling max, mean and count over 1/2/4/6/12/24 **hours**, evaluated on a 15-minute-ceiled timestamp grid; self is excluded by construction because the query's own pushback has not happened yet | `NASA/1st Place/.../nodes.py::extract_standtime_features` **[S]** |
| F9f `error_etd_mean/std`, `pushback_error_etd_mean/std` | rolling mean/std of the ETD error and of the pushback-vs-ETD error over the adaptive 120–180 min window | `NASA/3rd Place/Phase 1/helper.py` **[S]** |

**F9 is the only congestion-family feature that survived a pre-registered acceptance test in the 2026
competition.** `OpenAir/status.md` C14, verbatim evidence: *"`dis_state_30m` = mean of other flights' AOBT−EOBT
in the previous 30 min (self excluded, AOBT ≤ scored MVT). corr vs own AOBT−EOBT = 0.45, vs frozen residual =
0.11 (own AOBT−EOBT vs residual = 0.02). … Adding it to the frozen L2 residual: Jan+Jul matched
**256.46 → 253.82**, >30 min **821 → 807**, <20 min +1.5 s; December matched **230.04 → 226.95**. Q8 mean
residual +60 → −3.5. >60 min Jan+Jul not improved. LIRF matched −11 s; LFPG ~0."* Verdict recorded as
*"KEEP `dis_state_30m` (+ `dis_frac20_30m`) … a real but small (~1% matched RMSE) uncompression of the
operational disruption tail, not a fix for 1–24 h bombs. **Do not sweep extra windows.**"* **[S]**

### F10 — Demand versus declared capacity

I searched all four competitor repos and the five NASA winners. **No source in this catalogue uses a declared
airport capacity.** The closest substitutes actually implemented:

| ID | definition | source |
|---|---|---|
| F10a `dep_arr_ratio_30m` | departures over arrivals in the previous 30 minutes — a *mix* rather than a load factor | `docs/features_log.md` #5, **rejected** `+0.03 [−0.16,+0.22]` **[S]** |
| F10b `congestion_vs_usual_30m` | current 30-min movement count divided by the historical typical count for that airport/hour — a **self-normalising demand/capacity proxy** | `docs/features_log.md` #6, **rejected** `+0.09 [−0.16,+0.31]` **[S]** |
| F10c `dep_ratio_15_60` | `dep_prev_15m / max(dep_prev_60m, 1)` — regime indicator, is the rate rising or falling | `OpenAir/analysis/03_congestion_state.py`; corr with target **−0.024** **[S]** |
| F10d `active_runways_60m` | number of distinct runways in use in the last hour — the capacity side of the ratio | `docs/features_log.md` #4, **rejected** `−0.06 [−0.29,+0.17]` **[S]** |
| F10e `departure_runways` / `arrival_runways` as categoricals | the runway configuration itself, which *is* the capacity state | `NASA/4th Place` KDD Tab. 3, `NASA/1st Place/.../extract_config_features` **[S]** |
| F10f `current departure demand` (frozen at top of hour) vs `current departures` (updated per minute) | demand and realised throughput as two separate channels, letting the model form the ratio | MITRE arXiv:2111.09139 **[S]** |

**[X]** The absence of declared-capacity features across every winning solution I read is itself informative:
either the data was not available, or the runway-configuration categorical already carries the capacity
information more cheaply. I found no source that tested a true declared-capacity ratio and reported the result.

### F11 — Unimpeded / reference taxi time (the denominator every congestion feature is implicitly measured against)

| ID | definition | source |
|---|---|---|
| F11a EUROCONTROL reference | P10 of actual taxi-out per `(STAND, departure runway)` combo over a **rolling 12 months**; combo dropped unless ≥ 10 flights have taxi-out ≤ P10 | EUROCONTROL ATXOT Ed. 01.00 **[S]** |
| F11b NASA unimpeded | P10 of taxi time per `(gate, spot, runway)` combo over a full calendar year | Lee & Malik AIAA 2016-3910 **[S]** |
| F11c CDG | "unimpeded taxi time" is among the top features of the regression tree | Herrema et al. 2018 **[S]** |
| F11d `geo_mean` | a per-`(airport, stand, runway)` mean of the target, used as the geometry baseline that congestion features are then added on top of | `OpenAir/status.md` C3 **[S]** |
| F11e `stand-rwy mean` | `airport × stand × runway` mean of the target; in-sample RMSE **469.04 s** vs airport-only 514.84 s | `OpenAir/analysis/output/03_congestion_state.txt` **[S]** |

**F11 is a target encoding.** Every variant is built from the label of other flights, and is therefore subject
to the leakage analysis in §4.

---

## 3. What each family actually bought — positives AND negatives

### 3.1 Correlation table measured on *our exact dataset* (2,085,047 training departures)

Source: `OpenAir/analysis/output/03_congestion_state.txt` lines 69–105. Columns: Pearson correlation with the
raw target; with the residual after subtracting the `airport×stand×runway` mean; with the residual after
subtracting the airport mean; with the residual after subtracting the `MVT − AOBT_3` proxy. **[S]**

| feature | corr_y | corr_stand-rwy-resid | corr_airport-resid | **corr_AOBT-proxy-resid** |
|---|---:|---:|---:|---:|
| `dep_prev_1m` | 0.0086 | 0.0092 | 0.0013 | −0.0450 |
| `dep_prev_5m` | 0.0403 | 0.0655 | 0.0502 | −0.0410 |
| `dep_prev_15m` | 0.0508 | 0.0690 | 0.0538 | −0.0603 |
| `dep_prev_60m` | 0.0707 | 0.0622 | 0.0604 | −0.0778 |
| `dep_rwy_prev_15m` | 0.1409 | 0.0346 | 0.0345 | 0.0007 |
| `dep_rwy_prev_60m` | **0.1667** | 0.0238 | 0.0331 | 0.0071 |
| `arr_prev_15m` | 0.0434 | 0.0161 | 0.0137 | −0.0536 |
| `arr_prev_60m` | 0.0256 | 0.0028 | −0.0049 | −0.0821 |
| **`queue_at_mvt`** | **0.2641** | **0.1534** | 0.1430 | **−0.1121** |
| `push_prev_15m` | 0.0135 | 0.0830 | 0.0259 | −0.0272 |
| `push_prev_30m` | 0.0660 | 0.1087 | 0.0735 | −0.0560 |
| `mov_prev_60m` | 0.0639 | 0.0435 | 0.0374 | −0.1049 |
| `dep_ratio_15_60` | −0.0240 | 0.0072 | −0.0111 | 0.0043 |

Note the brief's "previous-departure block intervals had correlation about 0.008" maps precisely to
`dep_prev_1m` corr_y = **0.0086**. **[S]**

The per-airport breakdown shows the effect is **not uniform** — `corr(dep_prev_15m, taxi)` is
EDDM **−0.000**, EHAM **−0.001**, LEMD **0.007**, LIRF 0.038 but EGLL 0.134, LTFM 0.132, LEBL 0.123,
LSZH 0.116, LFPG 0.114, EDDF 0.091. `corr(queue_at_mvt, taxi)` ranges from EDDM 0.027 / LEMD 0.029 to
EGLL **0.335** and LFPG 0.236. **[S]**

### 3.2 Nested in-sample OLS (same file)

| model | in-sample RMSE (s) |
|---|---:|
| airport mean | 514.84 |
| `airport × stand × runway` mean | 469.04 |
| `MVT − AOBT_3` proxy | 384.85 |
| stand-rwy mean + linear `dep_prev_15m` | 467.92 |
| stand-rwy mean + linear `queue_at_mvt` | 466.47 |
| OLS(AOBT proxy, `dep15`) | 352.51 |
| OLS(AOBT proxy, `dep15`, `rwy15`, `arr15`) | 350.85 |
| OLS(AOBT proxy, `dep15`, `rwy15`, `arr15`, `queue`) | **349.95** |

**Reading:** on top of geometry alone congestion buys ~2 s of in-sample RMSE; on top of the AOBT proxy the
whole congestion block buys **2.6 s in-sample**, which is well inside the noise envelope of any honest
out-of-sample test. **[S]**

### 3.3 The tail — the decisive negative

*"mean dep15 = **8.1** vs all 8.8"* for the `taxi > p99` rows (n = 20,834); *"mean queue = 12.5 vs all 10.0"*.
The worst 1% of taxi-outs happen at **below-average** departure density. **[S]** —
`OpenAir/analysis/output/03_congestion_state.txt` "HARD CASES" block.

`OpenAir/status.md` C10 (experiment E14): *"Traffic/queue corr(resid) ≈ 0. … `AOBT−EOBT` top quantile = 36% of
SSE (disruption tails). NEW: E15 should change the residual objective / tail handling, **not add linear
traffic, queue, or geometry tables**."* **[S]** That is the exact claim in the brief — *rolling queue features
did not shrink the tail* — confirmed at source.

`OpenAir/analysis/DISCOVERY_REPORT.md` §8: *"Congestion experts: **not justified.** Mean shifts of 1–3
minutes, EHAM has no volume effect, LIRF tail is not a congestion regime."* and §12: *"Dynamic surface-state
will not fix these."* **[S]**

### 3.4 Corroborating and contradicting evidence from the other teams

**Contradicting (congestion helped):**

- `NASA/4th Place` — LightGBM feature importance at KMEM places, in the **top 5**: `exp_deps_15min`
  (F2b), `dep_taxi_30hr` (F8a), `arr_taxi_30hr` (F8b), `standtime_30hr` (F4c). Write-up verbatim. **[S]**
  The KDD paper repeats this for KATL and states *"Many interval/average features regarding departures and
  arrivals such as `exp_deps_30min` … `dep_taxi_30hr` … and `standtime_30hr` … are also among the most
  important features. These features capture the busyness of the analyzed airport."* Whole-feature-set
  cumulative MAE 10.72 min vs a 14.33 min ETD−15 baseline. **[S]**
  *Caveat:* this is importance within a fitted model, not an ablation. Their feature selection was
  *"greedy bottom-up … only maintaining features that yielded a consistent significant improvement on the
  training and the validation data"*, which is at least a real inclusion test. **[S]**
- `Javid` — the dominant ensemble component `tfm_queue_d8` (blend weight **0.7986**) is described as
  *"base + TimesFM + weather + **queue** + interactions"*, and `queue_features.parquet` appears in every
  final training command including `tfm_queue_d10_v2_final` and `catboost_tail_d6_final`. Official v1 result
  **285.0706**; v3 line improved further. **[S]** *Caveat:* I found **no ablation** isolating the queue block
  in that repo — the 17.7%-weight sibling `no_tfm_full_d8` is "base features", which removes TimesFM and queue
  together.
- `Phoenix` — the strictly-past NM-neighbour block (F8e, 30 features) took v9 → v10 from 277.7530 to
  **275.1751 (−0.88)**, and the strictly-following block (F2c/F2d, 30 features) took v11 → v12 to
  **273.5557 (−1.4181)**. **[S]** *Caveat:* these are **taxi-duration** aggregates of neighbours, not counts.

**Corroborating (congestion did not help):**

- `NASA/2nd Place` (MAE 10.728, second of 450+ submissions) — two verbatim code comments:
  *"# Runway feature ended up not being used in the final model"* over `dep20_time` (F3f), and
  *"# Queue feature ended up not being used in final model"* over `get_queue_feature` (F7a). Their write-up
  Q11 lists under things that did not make it: *"A feature representing the length of the queue ahead of each
  flight, i.e. the number of flights scheduled to depart before a given airplane"* and *"Features
  incorporating the **rates of arrivals/departures** prior to a given flight"*. **[S]**
- `NASA/3rd Place` — *"We calculated taxi time using standtimes and arrivals, but this feature did not yield
  the expected improvements in our model."* **[S]**
- `Phoenix/METHOD.md` validation table: *"Traffic features with capped labels (rejected) | July–December |
  **434.145**"* against 382.284 for plain context at 1,000 trees — the traffic-feature variant was **52 s
  worse** and was rejected. **[S]**
- `OpenAir/status.md` C3 (E4, E5): *"on top of per-airport `AOBT + (AOBT−EOBT) + geo_mean`, adding traffic or
  queue changes Jan+Jul matched RMSE **303.53 → 303.64 / 303.56**. Linear incremental value ≈ 0. … those raw
  correlations were largely **proxies for the AOBT duration clock and geometry**."* **[S]**
- `OpenAir/status.md` §"Traffic/queue still absent from top importances" in the final CatBoost model. **[S]**
- **Our own repo**, `docs/features_log.md`, six congestion-family candidates measured against a matched
  noise column at `feature_fraction: 1.0` (200-row-trimmed RMSE, 95% CI): `queue_ahead` −0.16 [−0.40,+0.08];
  `bank_position` −0.19 [−0.41,+0.07]; `active_runways_60m` −0.06 [−0.29,+0.17]; `dep_arr_ratio_30m`
  +0.03 [−0.16,+0.22]; `congestion_vs_usual_30m` +0.09 [−0.16,+0.31]. All rejected — *below the measurement
  resolution of 0.25 s*, not harmful. **[S]**

**The reconciliation.** The positives are all *duration* aggregates (F8, F9). The negatives are all *count*
aggregates (F1, F2a/b, F3, F6, F7, F10). The two 2026 teams that gained (Phoenix, and OpenAir's `dis_state`)
gained from features whose **value** is a time difference of neighbouring flights; the teams that lost lost on
features whose value is a **count**. The NASA 4th place is the one apparent exception, and its top-5 list is
in fact three duration averages (`dep_taxi_30hr`, `arr_taxi_30hr`, `standtime_30hr`) plus one count
(`exp_deps_15min`). **[X]** — this generalisation is mine, but every data point above is sourced.

---

## 4. Leakage analysis: what is computable for a 2026 scored row

A 2026 scored row hands us: `MVT_TIME_UTC_mvt`, `SCHED_TIME_UTC_mvt`, `RUNWAY_mvt`, `STAND_mvt`, all NM
columns including `AOBT_3_flt` / `EOBT_1_flt` / `LOBT_flt` / `IOBT_flt` / `ARVT_1_flt` / `ARVT_3_flt` (98.47%),
and the complete arrival stream including arrival `BLOCK_TIME` and `TAXITIME_SEC`. It does **not** hand us
`BLOCK_TIME_UTC_mvt` or `TAXITIME_SEC_mvt` for *any* departure. §0 verifies all of this on the file. **[S]**

| family | computable for a 2026 scored row? | why / what breaks |
|---|---|---|
| F1 trailing DEP/ARR counts (airport, runway, stand) | **YES** | needs only other rows' `MVT_TIME` + `PHASE` + airport/runway/stand, all present. `OpenAir/DISCOVERY_REPORT.md` §5 marks all four count rows "Yes". **[S]** |
| F2 following DEP/ARR counts | **YES, batch only** | the whole two-month stream is in the file. Phoenix's own docstring: *"These observations do not establish availability before departure."* Legal for scoring; would not transfer to a live system. **[S]** |
| F2b `exp_deps_±W` around own ETD | **YES via `EOBT_1`/`SCHED`** | we have no evolving ETD stream, but `SCHED_TIME_UTC_mvt` and `EOBT_1_flt` are both present on ranking rows. **[X]** that `SCHED`-based counts are an adequate stand-in for NASA's Fuser-ETD counts — untested. |
| F3 same-runway counts, headways, gaps | **YES** | `RUNWAY_mvt` has **0 nulls** in ranking. **[S]** |
| F4 same-stand counts | **YES** (4–5 null stands only) | follow Phoenix's rule: a missing stand must **not** form an `UNKNOWN` group. **[S]** |
| F5d/F5e `queue_at_aobt` (Idris's *N*) | **YES on 98.47% of rows, NaN on the rest** | needs other flights' `AOBT_3` and `MVT_TIME` only. Javid computes it exactly this way on `ranking.parquet` and asserts `ranking_rows == 344841`. **[S]** |
| F6a/F6b `queue_at_takeoff` (Idris's *Q*) | **YES** | same inputs. **[S]** |
| F6c `push_prev_Wm` | **YES** | other flights' `AOBT_3` only. **[S]** |
| F7 queue rank among scheduled departures | **YES** | ranks over `SCHED_TIME` / `EOBT_1`, both present. But see §2 F7 — every source that tried it dropped it. **[S]** |
| F8a–F8c rolling mean of neighbours' **actual `TAXITIME_SEC_mvt`** | **NO — hard leak** | blank for all 344,841 ranking departures. `OpenAir/DISCOVERY_REPORT.md` §4 classifies "Other DEP `TAXITIME` / `BLOCK`" as class **C** with *"Rolling actual taxi-out of neighbours is **not** reproducible."* **[S]** A model trained on it would silently see NaN everywhere at scoring time. |
| F8d–F8f rolling mean of neighbours' **`MVT − AOBT_3`** | **YES — this is the legal substitute** | `OpenAir/DISCOVERY_REPORT.md` §5, verbatim: *"Rolling mean of `MVT − AOBT` of previous DEPs … **Yes — this is the usable 'recent taxi'**."* Phoenix, Javid and OpenAir all build it. **[S]** |
| F8g rolling mean of neighbours' **arrival** taxi-in | **YES** | ranking ARR `TAXITIME_SEC_mvt` has 0 nulls (§0). `OpenAir/DISCOVERY_REPORT.md`: *"All ARR fields including taxi-in | **A**"*. Empirically weak: *"EHAM −0.28, others ~0.1"*. **[S]** |
| F9 peer lateness `AOBT_3 − EOBT_1` of neighbours | **YES** | both columns present on ranking rows at 98.47%. OpenAir's implementation restricts contributors to NM-matched rows and self-excludes. **[S]** |
| F10b `congestion_vs_usual` (needs a historical baseline) | **YES but is a target-free encoding only if the baseline is built from counts, not labels** | if the "usual" level is a *movement count* it is safe; if it is a *mean taxi time* it becomes F11. **[X]** |
| F11 unimpeded / reference / `geo_mean` (P10 or mean of the **target** by stand×runway) | **YES but it is a target encoding** | the label exists only in the 2025 training months. A 2026 row gets its value from a *2025* lookup table. That is not a leak, it is an out-of-time encoding — but it must be built fold-wise during CV or in-fold CV RMSE will be optimistic. Our repo already has this machinery (`docs/features_log.md` exp. 7: *"既存のリーク防止機構に載せ、fold ごとに学習側だけで作る"*). **[S]** |

### 4.1 The three specific traps

**Trap 1 — endogenous windows.** Any window whose *boundary is the query's own takeoff and whose other
boundary is the query's own pushback* has a length equal to the target. `Scientific Reports` 2026's top two
SHAP features (departure queue x₃, arrival queue x₄, ≈18% each) are defined exactly that way: *"flights j that
complete takeoff **during the taxi-out process**"*. **[S]** `OpenAir/DISCOVERY_REPORT.md` §5 states the rule
explicitly: *"Do **not** count takeoffs during `[AOBT_i, MVT_i]` as a feature: that count scales with the
target (endogenous)."* **[S]** Our own `queue_ahead` (`docs/features_log.md` #1) was precisely this shape and
tested at zero. **[S]** Such a feature is *computable* at 2026 prediction time — `AOBT_3` and `MVT_TIME` are
both present — so it will not error; it will just encode the answer during training and then fail to be the
answer at scoring, because at scoring the target is not what generated it... **[X]** actually at scoring it
*is* generated from the same two timestamps, so it will not break; it will simply be a laundered copy of
`MVT − AOBT_3`, which the model already has directly. That is the mechanism by which it adds nothing.

**Trap 2 — neighbours' labels.** F8a/F8b/F8c and F11 all aggregate `TAXITIME_SEC_mvt`. For **departures** this
is blank in ranking → the feature is all-NaN at scoring while being informative in training. This is the
classic "works in CV, collapses on the leaderboard" failure. For **arrivals** it is fine (§0). **[S]**

**Trap 3 — month boundaries.** Ranking is Jan-2026 and Jul-2026 only. A trailing 24-hour or 30-hour window
(F1f, F1g, F8a) at the start of each ranking month has no history inside the file, while the same feature in
training always does. Phoenix handles this by scoping every group to *"the same UTC movement month"* so that
training and ranking are distorted identically; Javid handles it by *"Each supplied file is processed
independently; ranking contains two disconnected months."* **[S]** Any window longer than ~1 hour needs one of
those two conventions or its distribution shifts between fit and score. **[X]** for the "~1 hour" threshold —
that is my judgement, not a sourced number.

### 4.2 Null and sparse-window conventions actually used

| convention | who | exact |
|---|---|---|
| count features: 0, never null | OpenAir, Phoenix, Javid | counts are always defined **[S]** |
| mean features: NaN when count = 0 | OpenAir (`where=n>=1`), Phoenix (`where=count>0`), Javid (`where=count>0`) | **[S]** |
| mean features: **0** substituted for NaN | NASA 4th (`if math.isnan(avg): avg = 0`) | **[S]** — a different and arguably worse choice, since 0 is a legitimate-looking taxi time |
| adaptive window widening until ≥ 4 records | NASA 3rd (120 → 180 min in 60-min steps) | **[S]** |
| missing categorical → do **not** form a group | Phoenix (missing stand/runway excluded from the pool) | **[S]** |
| missing categorical → explicit `'UNK'` level | NASA 4th (ordinal encoders with a null category) | **[S]** |
| headway when no previous event | Phoenix: sentinel **3,600 s** (also the clip ceiling) | **[S]** |
| queue when own `AOBT` missing | Javid: **NaN** | **[S]** |
| exact-time ties | Phoenix: excluded from trailing counts and from following windows; tied "last"/"next" values **averaged** | **[S]** |

---

## 5. How the NASA pushback winners defined their congestion features (the closest analogue)

Final Phase-1 MAE (minutes): 1st Team CDS 10.673, 2nd Moles 10.728, 3rd Oracle 2 11.054, 4th FLHuskies2
11.105, 5th Cuong_Syr 11.946. All five trained **one tree model per airport**. **[S]** —
`github.com/drivendataorg/nasa-airport-pushback` README.

Structural analogy to us: their data is a movement stream (`runways` = actual runway arrival/departure times;
`standtimes` = actual gate pushback/in-block times; `etd` = evolving estimates; `mfs` = flight metadata), and
the rule is *"When making a prediction for a given flight at a given timestamp t, only auxiliary data with a
timestamp t′ ≤ t can be leveraged. In particular, the actual runway and the actual gate and runway departure
time of a flight are **not known yet** at the time of prediction. Such information about flights that have
**already departed** can however be used to estimate the current level of activity at an airport."* **[S]** —
Filienko et al., KDD 2023 UC, §2. **This is exactly our situation with one inversion: their forbidden-for-self
/ allowed-for-others field is the runway time; ours is the block time.**

**1st place (Team CDS, 298 features, Kedro + CatBoost ensemble):**
- `runways_{15,30,60,90,120,360,720,1440}_totdep` and `_totarr` — rolling counts of actual departures and
  arrivals, computed on a **15-minute grid of timestamps** rather than per flight, then joined. `extract_runway_features`. **[S]**
- `standtimes_{max,mean}delay_{1,2,4,6,12,24}h` and `standtimes_count_{p}` — peer lateness: `delay = actual
  pushback − first-ever ETD for that flight`, rolling max/mean/count at 1/2/4/6/12/24 hours on the same grid.
  `extract_standtime_features`. **[S]**
- `etd_roll_{60,120,240,360,720,1440}_{sum,max,std}` — rolling sum/max/std of the flight's **own** ETD
  revisions. Congestion enters indirectly through how much the ATM system is revising estimates. **[S]**
- No explicit surface queue. Their congestion is entirely *counts on a grid* + *peer lateness*.

**2nd place (Moles, Caltech, CatBoost per airport):** built both a queue-position feature (F7a) and a
throughput feature (F3f) and **used neither**. Verbatim code comments and the Q11 "did not make it" list.
Their final congestion signal is zero. **[S]**

**3rd place (Oracle 2, XGBoost + over/under-estimate classifier):** *"the most significant features were
derived from the estimated time of departure (most recent timepoint, average across the last 30 hours, and
standard deviation across the last 30 hours), airline code, prediction time, and taxi time. We calculated taxi
time using standtimes and arrivals, but **this feature did not yield the expected improvements**."* Their
rolling helper uses a **120-minute lookback expanded to 180 minutes until at least 4 records exist**, producing
mean/std of taxi time, of ETD error, of minutes-until-pushback, and of the pushback-vs-ETD error. **[S]**

**4th place (FLHuskies2, UW Tacoma, LightGBM per airport, and the only peer-reviewed writeup):** the fullest
and most useful catalogue. Verbatim from KDD 2023 Table 3: **[S]**

| feature | definition (verbatim) |
|---|---|
| `1hr_ETDP` | "Average departure delay, time between estimated and true departure time in the past hour." |
| `deps_nhr` | "Number of flights that have departed from the prediction airport in the past n hours." |
| `arrs_nhr` | "Number of flights that have arrived at the prediction airport within the past n hours." |
| `deps_taxiing` | "Number of flights that departed from the runway in the past 30 hours." (the *code* actually computes pushed-back-minus-taken-off; the table's wording is looser than `add_traffic.py`) |
| `exp_deps_nmin` | "Number of flights expected to depart from the prediction airport within the next n minutes." (the *code* uses a symmetric `[ETD−n, ETD+n]` window) |
| `delay_nhr` | "Average difference between the estimated and true departure time of all flights at the prediction airport within the past n hours." |
| `standtime_nhr` | "Average difference between the first position (timestamp of gufi creation) and pushback time … for all arrival flights at the prediction airport in the past n hours." |
| `dep_taxi_nhr` | "Average time that all departing flights in the past n hours spent taxiing." |
| `departure_runways` / `arrival_runways` | "Collection of runway names currently utilized at the airport for departures / arrivals." |

Instantiated windows in the code: `deps_3hr`, `deps_30hr`, `arrs_3hr`, `arrs_30hr`, `deps_taxiing`,
`arrs_taxiing`, `exp_deps_15min`, `exp_deps_30min`, `dep_taxi_30hr`, `arr_taxi_30hr`, `standtime_30hr`. **[S]**
Their selection procedure was *"greedy bottom-up … adding one feature at a time and only maintaining features
that yielded a consistent significant improvement on the training and the validation data"*. **[S]**
Top-5 importance at KMEM: destination airport, `exp_deps_15min`, `dep_taxi_30hr`, `arr_taxi_30hr`,
`standtime_30hr` — i.e. **four of the five are congestion features**. **[S]**

**5th place (Cuong_Syr, XGBoost per airport):** one congestion feature only — `num_flights`, distinct GUFIs on
the runways table in the preceding clock hour, joined on a `date_hour` string. **[S]**

**The spread across five winners separated by 1.3 minutes of MAE is 0 → 4 congestion features in the top 5.**
That, by itself, says the family is not decisive. **[X]**

---

## Applicable to resilient-hedgehog

Context that conditions everything below, from our own repo: `docs/features_log.md` records **10 consecutive
rejections**, six of them congestion-family, and the diagnosis that *model A's training target is
`AOBT_3 − BLOCK_TIME`, the offset between two recording clocks, not the physics of taxiing* — so congestion
features are being added in a region the proxy already dominates. It also records the measurement floor:
**a single extra column moves the score by ±0.5 s at `feature_fraction 0.8`**, the corrected screening
resolution is **0.25 s**, and the group-wise floor for static attributes
(`airport × stand × runway × hour × type`) is within-group RMSE **268.1** while the model already achieves
**~202** on matched rows. **[S]** Any candidate below whose expected effect is < 0.25 s cannot be
*distinguished* by our current instrument, whatever its true value.

Ordered by my read of expected value. "Computable" means: produces a non-degenerate value for a 2026 scored
row, verified against §0.

| # | candidate | (a) exact definition | (b) computable at prediction time? | (c) sourced evidence |
|---|---|---|---|---|
| **1** | `dis_state_30m` + `dis_frac20_30m` (peer lateness / disruption state, **F9a/F9b**) | mean over *other* NM-matched departures at the same airport with `AOBT_3 ∈ (MVT_i − 30 min, MVT_i]` of `(AOBT_3 − EOBT_1)`; self subtracted from sum and count when own `AOBT_3` is in window; NaN when surviving count < 1. `dis_frac20_30m` = fraction of those peers with lateness > 1,200 s. | **YES.** `AOBT_3` and `EOBT_1` both present on 98.47% of ranking departures; 1.53% get NaN, which is the same unmatched population the model already routes separately. | **FOR:** `OpenAir/status.md` C14 — matched RMSE **256.46 → 253.82** on Jan+Jul, **230.04 → 226.95** on December, >30 min bucket 821 → 807, mean residual in the top disruption octile +60 → −3.5 s. Kept after a pre-registered test. **[S]** The brief's "OpenAir gained from a disruption-state feature" is confirmed at source, with the exact window (30 min), key (airport), value (`AOBT−EOBT`) and self-exclusion. **AGAINST:** OpenAir's own verdict caps it — *"a real but small (~1% matched RMSE)"*, *">60 min Jan+Jul not improved"*, *"LFPG ~0"*, and explicitly *"Do not sweep extra windows."* **[S]** **[X]** On our scale, 1% of ~202 s matched RMSE is ~2 s, comfortably above our 0.25 s floor — this is the one congestion candidate I would expect to be *measurable*. |
| **2** | `nm_following_airport_{15,60}m_{count,mean,std,own_minus_mean}` (**F2c**), and the runway/stand variants | pool = DEP rows with `ADEP_mvt == ADEP_flt` and `proxy = MVT − AOBT_3 ∈ [0, 7200] s`; window **strictly after, upper-inclusive `(t, t+W]`** with `t = own MVT_TIME`; groups `airport`, `airport×runway`, `airport×stand`, each scoped **within the same UTC calendar month**; query itself and all exact-time ties excluded; missing stand/runway forms no group; NaN mean when count 0. | **YES, batch only.** The full two-month ranking stream is in the file. Phoenix's own docstring disclaims live availability; that does not affect leaderboard validity. Month scoping handles the Jan/Jul disconnection. | **FOR:** `Phoenix/METHOD.md` + `README.md` — v12 official **273.5557**, improving v11 by **1.4181 s**, 12/131. Local reused Jan+Jul 313.7463 → 312.9710 (−0.7753), 59/62 days improved. **[S]** This confirms the brief's "-1.4 from following-context NM features". **AGAINST:** the same METHOD.md records that the *LightGBM* version of the identical 30 features gained only **0.283 s**, below its own 0.5 s gate, and *"remains rejected"* — the gain is model-dependent (CatBoost depth 9, 4,999 trees) and does not reproduce in a LightGBM of 679 trees. **[S]** **[X]** Our model B is LightGBM; that is the closer of the two controls, and it is the one that failed. |
| **3** | `nm_neighbor_*` strictly-past twin of #2 (**F8e**) | identical, but window `[t−W, t)` via `side='left'`; plus `last_age_seconds` and `last_proxy` (ties averaged, NaN if the last event is > 7,200 s old). | **YES.** | **FOR:** v9 → v10 official **277.7530 → 275.1751 (−0.8819)**; component comparison improved all four held-out months (Jan 347.31→344.18, Jul 301.05→296.68, Feb 283.22→275.39, Aug 249.15→244.70). **[S]** **AGAINST:** `OpenAir/status.md` C2 — the same idea as roll10 `MVT−AOBT` gives only **318.0 → 316.2** *once `AOBT−EOBT` is already in the model*, and OpenAir's verdict is *"real but second-order"*. **[S]** Our model already carries the AOBT proxy as the baseline, so we are in OpenAir's "second-order" regime, not Phoenix's. |
| **4** | `arr_taxi_mean_{1800,7200,21600}` (**F8g**) — centred rolling mean of **arrival** taxi-in | centred window `[t−W, t+W]` on `TAXITIME_SEC_mvt` of ARR rows at the same airport, W ∈ {1800, 7200, 21600} s; denominator `max(1, n)`. | **YES, and it is a real label, not a proxy.** §0 verified: ranking ARR `TAXITIME_SEC_mvt` has **0 nulls** across 344,693 rows. This is the only place in the whole catalogue where neighbours' *actual* taxi labels are legally available to us. | **FOR:** used by `Javid/scripts/train.py` in the 0.7986-weight ensemble component of the best public solution (LB 268.56 line). **[S]** `NASA/4th Place` ranks `arr_taxi_30hr` **4th** in KMEM importance. **[S]** **AGAINST:** `OpenAir/DISCOVERY_REPORT.md` §6 — *"Recent taxi-in | Yes | available; weak / unstable (EHAM −0.28, others ~0.1)"*. **[S]** **[X]** Untried in our repo; `docs/features_log.md` lists no arrival-derived candidate at all. Of everything here this is the one genuinely *unexplored* axis with a live-data justification. |
| **5** | `queue_at_aobt` (**F5d/F5e**, Idris's *N*, anchored at **pushback** not takeoff) | on the set of `(AOBT_3, MVT_TIME)` intervals with duration in `(0, 21600] s` at the same airport (and separately at the same runway): `#(AOBT_j ≤ AOBT_i) − #(MVT_j ≤ AOBT_i) − 1`. NaN when own `AOBT_3` is missing. | **YES on 98.47%.** Javid runs precisely this over `ranking.parquet` and asserts 344,841 rows out. | **FOR:** Idris et al. 2002 — *N* is the classical surface-count variable (R² 0.19, improved to 0.59 by *Q*). **[S]** In Javid's 0.7986-weight component. **[S]** **AGAINST:** we already tested the takeoff-anchored, endogenous sibling (`queue_ahead`, `docs/features_log.md` #1) at **−0.16 [−0.40, +0.08]** — indistinguishable from a noise column. **[S]** `OpenAir/status.md` C3: adding queue on top of `AOBT + (AOBT−EOBT) + geo_mean` moves Jan+Jul matched **303.53 → 303.56**. **[S]** **[X]** The *pushback anchor* is the one variation we have not tested and it is not endogenous, which is a real difference from `queue_ahead` — but the prior from C3 is that it will land under our 0.25 s floor. |
| **6** | `queue_at_takeoff` (**F6a/F6b**, Idris's *Q*) | `#(AOBT_j < MVT_i) − #(MVT_j < MVT_i) − 1`, floored at 0, at airport and at `airport×runway`. | **YES on 98.47%.** | **FOR:** highest raw correlation of any count feature on our exact data — corr_y **0.2641**, corr with stand-rwy residual **0.1534**; per-airport up to **0.335 at EGLL**, 0.236 at LFPG. **[S]** **AGAINST, decisively:** corr with the **AOBT-proxy residual** is only **−0.1121**, and the nested OLS shows the entire congestion block buys **352.51 → 349.95 in-sample**. **[S]** Tail analysis: `taxi > p99` rows have **lower** `dep_prev_15m` (8.1 vs 8.8). **[S]** `OpenAir/status.md` C10: *"Traffic/queue corr(resid) ≈ 0"* against the fitted residual. **[S]** `NASA/2nd Place` built the equivalent and dropped it. **[S]** |
| **7** | trailing counts `dep_prev_{5,15,30,60}m`, `arr_prev_{15,30,60}m`, `dep_rwy_prev_{15,60}m`, headways (**F1, F3**) | as in §2, self-excluded, airport / airport×runway keys, counts never null. | **YES, trivially.** Only `MVT_TIME`, `PHASE`, `ADEP`/`ADES`, `RUNWAY` are needed, all 0-null in ranking. | **FOR:** `NASA/4th Place` KDD Tab. 3 and top-5 importance; every 2026 competitor computes some version. **[S]** **AGAINST:** the strongest negative stack in this report. Our own six rejections (`docs/features_log.md` #1–#6, all CIs straddling zero against a noise column). **[S]** `NASA/2nd Place` Q11 explicitly dropped *"features incorporating the rates of arrivals/departures prior to a given flight"*. **[S]** `Phoenix/METHOD.md`: traffic-feature variant **434.145 vs 382.284**, rejected. **[S]** OpenAir corr with AOBT-proxy residual ≤ 0.08 in magnitude for every window. **[S]** **[X]** I would not spend a measurement slot here. |
| **8** | forward counts `DEP_count_+1_{900,1800,3600,7200}` / `ARR_count_+1_*` (**F2a**) | `[t, t+W)` counts at the airport, keyed on `ADEP_mvt` for departures and `ADES_mvt` for arrivals; self included (constant +1 for DEP). | **YES, batch only.** | **FOR:** in `Javid/scripts/train.py`, i.e. in the best public solution's main component. **[S]** MITRE arXiv:2111.09139 uses forward demand at +1 h and +2 h as first-class channels. **[S]** **AGAINST:** never ablated in any source I read. Phoenix's gain (#2) came from forward **durations**, not forward **counts**. **[X]** — I could find no evidence either way for forward counts in isolation. |
| **9** | queue **position/rank** among flights scheduled to depart nearby (**F7**) | rank of own `SCHED_TIME`/`EOBT_1` among flights in a 15-min bucket. | **YES.** | **AGAINST, unanimously.** `NASA/2nd Place` built it (`get_queue_feature`) and left a code comment that it *"ended up not being used in final model"*, and listed *"the length of the queue ahead of each flight"* under things that did not make it. **[S]** Our `queue_ahead` −0.16 and `bank_position` −0.19, both indistinguishable from noise. **[S]** No source anywhere reports a positive rank result. **Do not build.** |
| **10** | demand ÷ declared capacity (**F10**) | movements in window ÷ declared hourly capacity of the airport. | **Only with external data** — declared capacity is not in the bundle. Our repo has an external-data precedent (weather via Iowa Mesonet; `dest_distance_km` via OurAirports). | **AGAINST / absent:** no source in this catalogue uses declared capacity. The in-data substitutes were all tried by us and rejected: `dep_arr_ratio_30m` +0.03, `congestion_vs_usual_30m` +0.09, `active_runways_60m` −0.06. **[S]** OpenAir's `dep_ratio_15_60` correlates **−0.024** with the target. **[S]** **[X]** The gap is real but the substitutes suggest the ceiling is low. |
| **11** | `dep_taxi_30hr`-style rolling mean of **neighbours' actual departure `TAXITIME_SEC_mvt`** (**F8a**) | rolling mean of the label over past departures. | **NO. This is the leak.** All 344,841 ranking departures have `TAXITIME_SEC_mvt` null (§0). The feature would be informative in training and all-NaN at scoring. | `OpenAir/analysis/DISCOVERY_REPORT.md` §4 classes it **C**: *"Rolling actual taxi-out of neighbours is not reproducible."* **[S]** Note that this is exactly the feature NASA 4th place ranked **3rd most important** — it was legal *there* because that competition blanked only the query flight's own runway time, not everyone's. **The single most transferable-looking idea in the NASA winners is the one that does not transfer.** |
| **12** | unimpeded / reference taxi time as a feature (**F11**) | EUROCONTROL rule: P10 of `TAXITIME_SEC_mvt` per `(STAND_mvt, RUNWAY_mvt)` over a rolling 12 months, dropped unless ≥ 10 flights sit at or below the P10. | **YES, but only as an out-of-time target encoding** — the 2026 rows would read a 2025 table. Must be built fold-wise or CV will be optimistic. | **FOR:** top feature at CDG (Herrema 2018), top feature at CLT (Lee & Malik), the official EUROCONTROL normaliser. **[S]** **AGAINST:** our repo has already measured the ceiling of this axis directly — within-group RMSE for `airport × stand × runway` is **307.5** and for `airport × stand × runway × hour × type` is **268.1**, while the model already sits at **~202** on matched rows (`docs/features_log.md`). *"静的・文脈の空間は飽和している"*. **[S]** A P10 encoding cannot beat a group mean that the model has already surpassed. **Do not build.** |

### Three judgements I would defend

1. **The count/duration split is the whole story.** Every congestion *count* family (F1, F2a/b, F3, F6, F7,
   F10) has at least one strong negative result and no clean positive ablation. Every congestion *duration*
   family (F8d–F8g, F9) has at least one measured positive in this exact competition. The mechanism is visible
   in OpenAir's table: counts correlate 0.01–0.17 with the target but **−0.11 to +0.08** with the residual
   after the AOBT proxy, whereas the durations are directly commensurate with the residual. **[S]** for every
   number, **[X]** for the generalisation.

2. **Our model's specific problem makes counts even less promising than the cross-team evidence suggests.**
   `docs/features_log.md` establishes that model A learns `AOBT_3 − BLOCK_TIME`, a clock-reconciliation
   residual. A count of aircraft on the taxiway has no mechanism by which it would predict a *reporting*
   discrepancy between NM and the airport. F9 (peer lateness) does have such a mechanism — a disrupted airport
   is one where reporting practice changes — which is consistent with it being the one that measured. **[X]**

3. **If exactly one slot is spent here, spend it on #1 (`dis_state_30m` + `dis_frac20_30m`), built exactly to
   OpenAir's specification** — 30-minute window, airport key, value `AOBT_3 − EOBT_1`, anchored at the query's
   own `MVT_TIME`, self subtracted from sum and count, NaN below count 1, no window sweep. It is the only
   candidate in this catalogue with a pre-registered positive result on our dataset, the only one with a
   mechanism aligned to what our model actually predicts, and the only one whose reported effect size (~1% of
   matched RMSE, ≈2 s) clears our 0.25 s measurement floor. Everything else in the count families has an
   expected effect that our instrument cannot resolve. **[X]** for the recommendation; **[S]** for each input.

### Loose ends I could not close

- I could not retrieve the primary PDFs of Idris et al. 2002 (MIT DSpace human-verification wall), Herrema et
  al. 2018 (paywall), or Wang/Brownlee/Woodward 2021 (Stirling repository behind an Azure WAF). The claims
  attributed to them above come from search-engine summaries of those specific papers and are labelled as
  such. **[S]** on the attribution, weaker on the exact numerals.
- No source I found ablates **forward counts** (F2a) in isolation. Javid ships them; nobody tests them.
- No source I found builds a **stand-proximity occupancy** feature (F4, "other aircraft pushing back within N
  stands of mine"). It is the one gap in the catalogue that is neither tested nor obviously useless.
- I found no A/B in `Javid/` isolating the 12-column `queue_features.parquet` block; its presence in the
  0.7986-weight component is suggestive but is not evidence of incremental value.
