# Why taxi-out delays actually happen — a mechanism map, a calibrated screen, and what survives it (2026-09-25)

**Script (read-only; nothing written to `outputs/`, `data/`, `src/`, `configs/`; no fit on the competition
target):** `docs/competitor_research_2026-09-21/scripts/delay_mechanisms.py`, run as
`POLARS_MAX_THREADS=2 .venv/bin/python docs/competitor_research_2026-09-21/scripts/delay_mechanisms.py [S1 … S7]`.
Every number below carries the section (`S1`, `S1b`, `S2`…`S7`) that prints it. S1–S4 are about two minutes each,
S5–S7 about four.

**Inputs.** `data/work/feat_train.parquet` (2,085,047 departures, 140 columns),
`data/work/feat_rank.parquet` (344,841 scored departures), `data/raw/ranking.parquet` (689,534 rows,
344,841 DEP + 344,693 ARR), `data/ref/metar/*.csv`, `experiments/oof_main_gap-62dd7c6e88.parquet` and
its `_folds` sidecar (672,428 pooled OOF rows over folds jan/jul/forward; `jul_fwd` excluded as
elsewhere), and `experiments/oof_main_reparam-bc60017878.parquet` for the instrument calibration.

**Notation.** `y` = `TAXITIME_SEC_mvt` = MVT − BLOCK. `proxy` = `mvt_minus_aobt3` = MVT − AOBT_3.
`g` = `y − proxy` = AOBT_3 − BLOCK. **Lane A** = `proxy` is not null (`src/prc/train.py:402`), 664,050
of the 672,428 pooled OOF rows, base RMSE **213.1266**. `r` = OOF prediction − `y`.

**Marking.** `[S]` = measured here by the named section. `[X]` = inference, marked as one.

**Parity.** No column proposed or built here reads `BLOCK_TIME_UTC_mvt` or `TAXITIME_SEC_mvt` of any
departure, own or neighbour. Neighbour statistics use `MVT_TIME_UTC_mvt` and `AOBT_3_flt` only, lagged
60 s, exactly as `src/prc/features.py:196-208` does. `bs = BLOCK − SCHED` appears nowhere.

---

## 0. The answer in seven sentences

1. **On 97 % of rows the question "what makes this taxi longer" is already answered by a column the model
   is handed.** The model's prediction is `proxy + ĝ`, so every second of lane-A error is an error about
   `g = AOBT_3 − BLOCK`, which is a *clock disagreement* and contains no taxi information at all. That is
   why every congestion candidate this project has built has died: congestion moves `y` and `proxy`
   together, and cancels out of `g`. [S] `S1`, and `model-a-ceiling.md` §1.2 independently.
2. **The disagreement is large, structured and specific to the stand event.** sd(`g`) = 399.3 s; it is not
   a rounding artefact (AOBT_3 is on a whole minute on 97.5 % of rows). On the arrival side, where both
   systems observe the *runway*, NM and the airport agree to sd 57.5 s. NM does not observe the stand, so
   AOBT_3 is an estimate and `g` is its error. [S] `S1`, `S1b`.
3. **How much the offset is worth varies sevenfold across airports and the model is forced to subtract it
   at 1.0 everywhere**: OLS slope of `y` on `proxy` runs LTFM 0.135, EDDM 0.253 … LIRF 0.949. At LTFM and
   LEBL the offset is a quantized table value on a fifth to a third of rows — the modal minute holds 20×
   and 30× the traffic of its neighbouring minutes. [S] `S1`, `S2`.
4. **The recoverable error is time-local, not row-local.** An oracle handed the true mean residual of each
   ADEP × calendar-date × local-hour cell recovers **−5.39 s** of lane A, against a shuffled-date control
   at **+0.07**; ADEP × date alone recovers −1.48 s and ADEP alone −0.11 s. LIRF's ten worst days are
   8.1 % of its days and 24.2 % of its lane-A SSE. [S] `S4`.
5. **That state is invisible.** The weighted R² of a month-blocked model of the cell-mean residual on
   thirteen visible cell aggregates (proxy, pushback delay, replanning rate, arrival taxi-in, counts,
   weather) is **−0.032**, worse than a Gaussian control at −0.007; at day level the best single
   correlation is day-median `mvt_minus_sched` at **+0.080** against a control at −0.009. [S] §4.3.
6. **The screen used here is calibrated, and nothing passed it.** A month-blocked residual-explanation
   instrument reads **−1.83 s** for `stand_arrival_gap`, whose true full-model lane-A gain measured on the
   same rows is **−3.213 s** — a transfer coefficient of **0.57**, with a control-arm seed sd of 0.003 and
   a base-to-control gap of 0.067. Eight mechanism families and 28 new columns read between **+0.27 and
   −0.071**. A candidate worth −0.5 s of lane A would have read −0.28 s. [S] `S6`, `S7`.
7. **[X] The honest headline: the taxi-out mechanisms the brief lists are all either already in the model
   or cancel out of the quantity the model actually fits, and the error that remains is a time-local
   airport state for which this dataset carries no observable.** What is left to build is small, and the
   three candidates below are ranked by mechanism and measurability, with pre-registered effects between
   −0.4 s and zero.

---

## 1. What is already in the model (enumerated, not assumed)

`src/prc/features.py` and `src/prc/train.py` produce a 140-column feature frame, of which 118 reach the
model as numerics plus 10 categoricals (`numeric_feature_columns`, `CATEGORICALS`). Enumerated:

| family | columns |
|---|---|
| row clocks | `mvt_minus_aobt3`, `mvt_minus_sched`, `mvt_minus_eobt1`, `mvt_minus_lobt`, `mvt_minus_iobt`, `lobt_minus_eobt1`, `aobt3_minus_eobt1`, `aobt3_minus_sched` + 8 `_slog` copies |
| null flags | `nm_matched`, `aobt3_null`, `lobt_null`, `eobt1_null`, `stand_null`, `runway_null`, `diverted`, `actype_mismatch`, `anchor_ok` |
| calendar | `month`, `dow`, `hour_utc`, `minute_of_day_utc`, `hour_local` |
| counts, 5 windows | `dep_tko_prev/next_{5,10,15,30,60}m`, `arr_land_prev/next_{…}`, `arr_inblock_prev_{…}` |
| runway | `rwy_headway_prev/next_s`, `rwy_tko_prev_{15,60}m`, `active_runways_60m`, `stand_rwy_share` |
| neighbour clocks | `nbr_aobt_delay_mean_{30,60}m`, **`nbr_mvt_minus_aobt_mean_{30,60}m`**, `nbr_dep_n_{30,60}m` |
| arrivals | `arr_taxi_in_mean_{30,60}m`, `arr_taxi_in_n_{30,60}m` |
| stand link | `stand_arrival_gap`, `stand_aobt_turn`, `stand_arrival_type_match` |
| demand shape | `queue_ahead`, `bank_position`, `dep_arr_ratio_30m`, `congestion_vs_usual_30m`, `window_coverage_60m` |
| exogenous | `dest_distance_km`, `planned_flight_s`, `eobt_schedule_revision`, `delay_neighbor_deviation` |
| weather ×2 (level + `_prev1h`) | `temp_c`, `dewp_spread_c`, `wind_ms`, `gust_ms`, `precip_mm`, `vis_km`, `frozen_wx`, `deicing_gate`, `low_vis`, `temp_x_precip` |
| categoricals | `ADEP_mvt`, `RUNWAY_mvt`, `STAND_mvt`, `AIRCRAFT_TYPE_mvt`, `WK_TBL_CAT_flt`, `AIRCRAFT_OPERATOR_flt`, `MARKET_SEGMENT_flt`, `FLIGHT_TYPE_flt`, `ADES_mvt`, `ADES_REGION` |
| label-derived, per fold | four encoding tables keyed on `ADEP_mvt` (`train.py:282-311`) |
| controls | `noise_control`, `_2`, `_3`, `_4` |

**Raw columns never used anywhere:** `ARVT_3_flt`, `CALLSIGN_flt`, `ADEP_flt`, `ADES_FILED_flt`
(only via `diverted`), `FLIGHT_RULE_mvt`/`_flt`, `FLIGHT_mvt`. **METAR fields fetched and then dropped**
(`src/prc/weather.py:23` fetches them, `:84-113` keeps a subset): **`drct` (wind direction)**, `skyc1`
(cloud), `relh`, `feel`. Wind *direction* has therefore never reached the model, although runway
configuration — the thing that rewrites every taxi route — is chosen by it.

### 1.1 The residual has no additive structure in any of them

`S3`. Out-of-fold (5 random folds, ridge λ=50, 20 quantile bins) per-bin constant correction on the
lane-A residual, in seconds of lane-A RMSE. Best of 118 numerics and 10 categoricals:

| column | Δ RMSE | | column | Δ RMSE |
|---|---:|---|---|---:|
| `AIRCRAFT_OPERATOR_flt` [cat] | −0.135 | | `nbr_mvt_minus_aobt_mean_60m` | −0.081 |
| `RUNWAY_mvt` [cat] | −0.131 | | `month` | −0.077 |
| `ADEP_mvt` [cat] | −0.113 | | `arr_land_prev_60m` | −0.073 |
| `temp_c_prev1h` | −0.095 | | `active_runways_60m` | −0.072 |
| `hour_local` | −0.085 | | **`noise_control_2`** | **−0.051** |
| `mvt_minus_aobt3` | −0.084 | | **`noise_control`** | **−0.051** |

**[S] Every one of the 128 keys lands within 0.09 s of a meaningless column.** This reproduces
`model-a-ceiling.md` §6 from a different direction and closes, again, the whole family of per-group
constants. The recoverable error is not an offset on any slice of the existing feature set.

---

## 2. The mechanism map

Each mechanism gets the two questions the brief asks: is it measurable at prediction time, and is it
already captured.

### 2.1 The mechanism that dominates everything else: the off-block stamp is an estimate

`S1`. On lane A the tree's target is `g = AOBT_3 − BLOCK`:

| quantity | value |
|---|---:|
| sd(`g`) | 399.3 s |
| median(`g`) | −55.0 s |
| `|g| ≤ 60 s` | 20.8 % of rows |
| AOBT_3 on a whole minute | **97.5 %** of rows |

So `g` is not quantization: the stamp's own granularity bounds that contribution at ±30 s against a
spread of 399 s.

**The arrival mirror settles what `g` is.** `data/raw/ranking.parquet` keeps `BLOCK_TIME_UTC_mvt` and
`TAXITIME_SEC_mvt` on all 344,693 scored **arrival** rows, and `ARVT_3_flt` on 342,141 of them. On those
rows `ARVT_3 − MVT_TIME` has median **+3 s** and sd **57.5 s**, and corr(`ARVT_3 − BLOCK`, −taxi-in) is
**0.986** [S] `S1b` — i.e. NM's arrival stamp *is* the landing stamp, and where both systems watch the
**runway** they agree to a minute. `[X]` The 399 s spread on the departure side is therefore a property
of the **stand** event, which NM does not observe and must estimate. `g` is the estimate's error, not
noise in the label.

**Consequence for feature design.** Anything that lengthens the real taxi lengthens `y` and `proxy`
together and cancels out of `g`. `[X]` This is the single best explanation on record for why
`queue_ahead` (288.90 vs base 285.52), `active_runways_60m` (290.14), `dep_arr_ratio_30m` (290.29),
`congestion_vs_usual_30m` (289.85) and `planned_flight_s` (290.25) all lost, and why surface occupancy,
neighbours' actual-vs-scheduled pushback and the congestion witness all landed inside the noise floor
(`missing-clock.md` §4.1, `model-a-ceiling.md` §6.1). It is not that the model already has congestion.
It is that on lane A congestion is not what is being predicted.

### 2.2 How much the offset is worth differs sevenfold, and the model must subtract it at 1.0

`S1`, OLS of `y` on `proxy` within airport:

| ADEP | n | b(y ~ proxy) | corr | lane-A RMSE |
|---|---:|---:|---:|---:|
| LTFM | 90,041 | **0.135** | 0.109 | 229.2 |
| EDDM | 51,286 | **0.253** | 0.169 | 175.7 |
| EGLL | 78,986 | 0.503 | 0.518 | 222.1 |
| LEBL | 55,501 | 0.562 | 0.515 | 195.3 |
| EHAM | 78,286 | 0.602 | 0.719 | 161.5 |
| LFPG | 76,471 | 0.631 | 0.570 | 241.5 |
| LEMD | 70,316 | 0.696 | 0.532 | 171.6 |
| EDDF | 71,489 | 0.846 | 0.653 | 175.3 |
| LSZH | 42,287 | 0.854 | 0.551 | 167.2 |
| LIRF | 49,387 | **0.949** | 0.626 | 345.1 |

**[X]** At LTFM the column the pipeline subtracts as a measurement explains 1.2 % of the label's variance.
The tree does recover this — regressing `r` on `proxy` within airport gives slopes of ≤ 0.024 in
magnitude and an *in-sample* per-airport linear recalibration moves lane A only 213.127 → 212.976 — so
this is a diagnosis, not a lever. It does say where "taxi mechanism" features could in principle matter:
at LTFM and EDDM, where the offset is nearly useless, the model is predicting the label outright.

### 2.3 Where the offset is a table value, not a measurement

`S2`. Minute histogram of `proxy` per airport, spike height measured against the median of the ten
neighbouring minute bins:

| ADEP | modal minute | rows | share | spike / neighbour | RMSE on spike | RMSE off spike |
|---|---:|---:|---:|---:|---:|---:|
| LEBL | 20 | 14,192 | 25.6 % | **30.1×** | 204 | 192 |
| LTFM | 19 | 26,859 | 29.8 % | **19.9×** | 211 | 236 |
| EDDM | 11 | 11,113 | 21.7 % | 5.1× | 136 | 185 |
| LSZH | 9 | 8,782 | 20.8 % | 3.3× | 135 | 175 |
| LEMD | 16 | 12,645 | 18.0 % | 2.5× | 154 | 175 |
| EDDF / EHAM / LIRF / LFPG / EGLL | — | — | 6.6–13.6 % | 1.1–1.9× | — | — |

LTFM's raw histogram is the clearest case: minute bins 17→191 rows, 18→8,587, 19→26,859, 20→18,962,
21→122, 22→117, 23→4,670, 24→14,787, 25→10,688, 26→44. Two tight blocks with a near-empty gap between
them, while the **label** `y` at LTFM is smooth over the same range (13 min→7,507, 14→7,807, 15→7,820,
16→7,318, 17→6,476). The 18–20 block is 31,636 movements off runway 36 and 20,239 off 35L; the 23–25
block is 12,570 off 18 and 8,298 off 17R (`S2` companion query). `[X]` This is NM assigning a nominal
taxi time per runway group, not measuring one.

**Measurable at prediction time?** Yes, from `MVT_TIME_UTC_mvt` and `AOBT_3_flt` alone. **Already
captured?** Only implicitly: the tree can reach it through `RUNWAY_mvt` × `proxy`, but the exact-equality
structure is expensive for a binned learner to represent. This becomes candidate **C1**.

### 2.4 Runway configuration, wind, and which runway this aircraft got

`RUNWAY_mvt` is non-null on 100 % of scored rows (`missing-clock.md` §1.2) and is a model categorical, so
*which* runway is known. What is not in the model is **why** — the wind. `drct` is fetched and dropped.
Built here (`S7`): `headwind_kt`, `crosswind_kt`, `rwy_wind_misalign = 1 − cos(drct − runway heading)`,
`wind_dir_deg`; plus configuration state from `RUNWAY_mvt` alone — `cfg_switch_recent` (modal departure
runway in [t−60 m, t−5 m) differs from [t−150 m, t−90 m)), `own_rwy_share_55m`, `n_rwy_active_55m`.
`drct` coverage: 90.0 % of the 138,457 cached METAR observations, per station 69.7 % (LSZH) to 97.5 %
(EHAM); 13,831–13,848 hours per station spanning 2025-01-01 → 2026-08-01, so both scored months are
covered. Result: **C3** and **C5** below.

### 2.5 De-icing

`S3` companion slice on lane A (`deicing_gate`, `frozen_wx`, `low_vis` are model columns):

| gate | n | RMSE | bias | share of lane-A SSE |
|---|---:|---:|---:|---:|
| `deicing_gate` = 1 | 5,018 | **343.3** | +9.6 | **1.96 %** on 0.76 % of rows |
| `frozen_wx` = true | 5,829 | 327.3 | +10.5 | 2.07 % on 0.88 % |
| `low_vis` = 1 | 11,332 | 291.7 | −6.4 | 3.20 % on 1.71 % |

Per airport on gate rows, the bias does not have one sign:

| ADEP | n (OOF) | bias | se | t | scored Jan-2026 gate rows |
|---|---:|---:|---:|---:|---:|
| LTFM | **81** | **+198.0** | 51.6 | +3.83 | **1,843** |
| EHAM | 792 | +28.8 | 7.3 | +3.93 | 1,037 |
| EDDF | 523 | +5.8 | 11.2 | +0.52 | 1,223 |
| LSZH | 965 | +0.2 | 9.1 | +0.02 | 1,379 |
| EDDM | 1,856 | −9.1 | 7.2 | −1.26 | 1,028 |
| LFPG | 544 | −16.7 | 26.7 | −0.62 | 213 |
| EGLL | 249 | +126.5 | — | — | **0** |

Two things follow. **[S] The EGLL half of the lead on record is moot for the board: the scored set has
zero EGLL de-icing-gate rows.** **[S] LTFM's de-icing experience is a single February 2025 cold snap** —
of its 2,825 training gate rows, 2,606 are in month 2, 136 in month 3, 83 in month 12, and the pooled OOF
covers months 1, 7, 11 and 12 only, which is why the out-of-fold evidence is 81 rows. The model will be
scored on 1,843 LTFM gate rows in January 2026 against out-of-fold evidence from 81. This becomes
candidate **C2** — and it is a robustness repair, not a new mechanism.

### 2.6 Everything else the brief lists

| mechanism | measured at prediction time by | already captured | verdict |
|---|---|---|---|
| runway queue | `queue_ahead`, `rwy_tko_prev_{15,60}m`, `rwy_headway_*`; built here: same-runway neighbour clock | yes, and it cancels out of `g` (§2.1) | C2/C6 arms: −0.041 and +0.085 vs control |
| stand→runway distance and routing | `STAND_mvt` × `RUNWAY_mvt` categoricals, `stand_rwy_share`; `proxy` itself *is* a route-length estimate at 5 airports (§2.3) | yes | dead |
| ATC flow restriction / CTOT | `lobt_minus_eobt1`, `mvt_minus_lobt`, `eobt_schedule_revision`; built here: destination-hour replanning state | yes at row level; the destination-state aggregate reads **+0.008 vs control** | dead |
| pushback conflicts, apron layout | built here: `pier_push_prev_10m` from `STAND_mvt` prefix + `AOBT_3_flt` | no — and it reads −0.033 vs control | dead |
| tow availability | nothing in the files | — | unmeasurable |
| aircraft class and speed | `WK_TBL_CAT_flt`, `AIRCRAFT_TYPE_mvt`, `dest_distance_km`, `planned_flight_s` | yes | dead |
| gate-hold policy | `aobt3_minus_eobt1`, `mvt_minus_sched` | yes | dead |
| time of day, bank structure | `hour_local`, `minute_of_day_utc`, `bank_position`, `congestion_vs_usual_30m` | yes | dead |
| surface degradation dispersion | built here: sd, coefficient of variation and tail mass of the neighbour clock | **no** — the model has only the mean | reads **+0.085 vs control**, dead |

A note on the last row, because it was the most promising idea going in: the model's single most valuable
column family is the *mean* neighbour off-block clock (removing it costs +5.60 s, `missing-clock.md` §3),
and it holds no second moment. Adding sd, CV and tail mass over 30 and 60 min makes the residual **worse** (+0.085)
than an arity-matched noise control. `[X]` Consistent with §2.1 — dispersion of taxi time is still a fact
about taxi time, and taxi time is not what is being predicted.

---

## 3. Where the error is

### 3.1 Concentrated in days, not in rows

`S4`. Lane-A SSE by airport-day:

| airport | days | top 5 days | top 10 days | top 20 days |
|---|---:|---:|---:|---:|
| LIRF | 123 | 14.9 % | **24.2 %** | 38.6 % |
| LFPG | 123 | 14.7 % | **22.0 %** | 33.1 % |

Worst LIRF day, 2025-07-13 (445 rows, RMSE 820.8, bias −306.5): day-median `y` **1,977 s** against a
normal-day 1,017, while day-median `proxy` is **962 s** against a normal 901. `[X]` The label moved by a
factor of two and the NM clock did not move at all — the failure is a divergence between the two stamps,
not an operational event that `proxy` would have shown. The day is visibly abnormal in other ways
(day-median `mvt_minus_sched` 4,200 s against a normal-day 1,264; replanning rate 0.557 against 0.482),
which §3.3 shows does not help. LFPG's worst day, 2025-01-17 (576 rows, RMSE 556.5), is the opposite
case — median `y` 1,800 against 958 *and* median `proxy` 1,679 against 901 — a day on which both clocks
moved together, and which the model handles far better (bias −46.7 against LIRF's −306.5). `[X]` The
distinction between the two is what separates a LIRF day from an LFPG day and is worth keeping in mind
when reading the per-airport table: LIRF's error is the clock diverging, LFPG's is a long day.

### 3.2 The oracle: the recoverable part is a time-local airport state

`S4`, leave-one-out group mean of the residual (the row's own residual excluded; ridge λ = 20):

| key | cells | Δ lane-A RMSE |
|---|---:|---:|
| `ADEP_mvt` | 10 | −0.114 |
| date | 123 | −0.320 |
| `ADEP_mvt` × date | 1,230 | −1.478 |
| `ADEP_mvt` × date × `RUNWAY_mvt` | 3,834 | −1.946 |
| `ADEP_mvt` × date × 2-hour block | 12,975 | −4.300 |
| **`ADEP_mvt` × date × `hour_local`** | **25,044** | **−5.386** |
| **control: dates permuted within airport** | 1,230 | **+0.070** |

**[S] There is a real, systematic, time-local component of lane-A error worth about five seconds, and it
is not a constant on any static key** (§1.1 and `model-a-ceiling.md` §6 both read ≈ 0.1–0.3 s on static
keys). `[X]` This is the same quantity as `model-a-ceiling.md` §4's ρ, now localized: the recoverable
share of the model's error lives in *when and where*, not in *which flight*.

### 3.3 …and it is invisible

Cell-level regression, `ADEP_mvt` × date × `hour_local`, cells with n ≥ 5 (22,783 cells, 659,063 rows,
weighted sd of the cell-mean residual 69.4 s). Month-blocked LightGBM, weighted R² of the cell-mean
residual:

| predictors | weighted R² |
|---|---:|
| 2 Gaussian controls | −0.007 |
| 13 visible aggregates (proxy, pushback delay, `ms`, arrival taxi-in, counts, weather, cell size) | **−0.032** |
| proxy-derived only | −0.028 |
| arrival-side only | −0.019 |
| weather only | −0.015 |

At **day** level (1,230 airport-days with n ≥ 100; weighted sd of the day-mean residual 28.1 s), the
weighted correlation of each day aggregate with the day-mean residual, against `noise_control` at −0.009:

`ms` **+0.080** · `vmin` −0.046 · cell size −0.049 · replanning rate +0.041 · `frozen_wx` +0.031 ·
`wmax` +0.029 · `proxy` +0.020 · pushback delay +0.019 · arrival taxi-in −0.011.

**[S] The best visible day-level signal explains 0.6 % of the day-level residual variance, which is itself
1.7 % of lane-A variance — 0.01 % of the total.** Two arms built on these aggregates and run through the
full instrument (§4) read **+0.27** and **+0.22** against control, i.e. they add noise.

`[X]` The reading: the model has already taken what the observable day state contains. The five seconds
in §3.2 would need an observation this dataset does not carry — the NM regulation list, an A-CDM TOBT/TSAT
feed, or the airport's own stand-reporting log.

---

## 4. The screen, and its calibration

`S5`, `S6`, `S7`. The instrument: fit the lane-A OOF residual `r` with LightGBM (leaves 8, lr 0.05,
min_data 200, 200 rounds), folds blocked by calendar quarter, on
`BASE = [ADEP code, hour_local, mvt_minus_aobt3, runway code]` plus the arm's four columns; report the
resulting lane-A RMSE. Every arm carries four columns so `feature_fraction` reshuffling is matched, and
the control arm is the four `noise_control` columns already in the frame.

**Sensitivity** (`S5`): injecting `r + N(0, k·sd(r))` reads 143.29 (k=1), 196.71 (k=3), **212.00 (k=10)**
against a base of 213.560 and a control of 213.502 — a signal carrying 1 % of the residual's variance is
seen at **−1.50 s** against the control. Seed spread over five seeds: base sd 0.007, control sd 0.003.

**Calibration against a known winner** (`S6`). `oof_main_reparam-bc60017878` and
`oof_main_gap-62dd7c6e88` differ in exactly one field of their configs — `include_candidates: []` versus
`['stand_arrival_gap']` (`experiments/experiments.jsonl`). On the same 664,050 rows:

| | lane-A RMSE |
|---|---:|
| before (`main_reparam`) | 216.3393 |
| after (`main_gap`) | 213.1266 |
| **true full-model gain** | **−3.2127** |
| instrument, base on `r_before` | 216.8037 (sd 0.009) |
| instrument, control4 | 216.7360 (sd 0.004) |
| instrument, `stand_arrival_gap` family | 214.9081 (sd 0.012) |
| **instrument reading vs control** | **−1.8278** |

**[S] Transfer coefficient 0.57.** `[X]` A candidate worth −0.5 s of lane A would read −0.28 s here,
against a control-arm seed sd of 0.004. The screen is not the reason nothing was found.

### 4.1 The sweep

`S7`, three seeds per arm, lane A, base 213.583 (sd 0.008), **control4 213.516 (sd 0.003)**. `S7` was
run twice, before and after the C8 builder was added; arm-to-arm reproducibility across the two runs is
0.001 s or better on every arm.

| arm | columns | RMSE | vs control | implied true lane-A gain (÷0.57) |
|---|---|---:|---:|---:|
| **C1 nominal proxy** | `proxy_at_mode`, `proxy_minus_mode`, `proxy_sec`, noise | 213.445 | **−0.071** | ≈ −0.12 |
| C3 wind direction | `headwind_kt`, `crosswind_kt`, `rwy_wind_misalign`, `wind_dir_deg` | 213.471 | −0.045 | ≈ −0.08 |
| C2 runway-local clock | `nbr_taxi_mean_rwy_30m`, `nbr_taxi_rwy_dev_30m`, `nbr_taxi_rwy_n_30m`, `rwy_share_60m` | 213.475 | −0.041 | ≈ −0.07 |
| C8 forward / centred clock | `nbr_fwd_30m`, `nbr_fwd_60m`, `nbr_ctr_3h`, `nbr_day` | 213.475 | −0.041 | ≈ −0.07 |
| C7 apron / pier | `nbr_taxi_mean_pier_30m`, `nbr_taxi_pier_dev_30m`, `pier_push_prev_10m`, noise | 213.483 | −0.033 | ≈ −0.06 |
| C4 gust repair | `gust_filled`, `gust_excess`, noise ×2 | 213.516 | +0.000 | 0 |
| C5 runway configuration | `cfg_switch_recent`, `own_rwy_share_55m`, `n_rwy_active_55m`, noise | 213.517 | +0.002 | 0 |
| destination regulation state † | `dest_replan_loo`, `dest_hold`, `dest_n`, noise | 213.524 | +0.008 | 0 |
| C6 neighbour dispersion | `nbr_taxi_sd_{30,60}m`, `nbr_taxi_tail_30m`, `nbr_taxi_cv_30m` | 213.601 | +0.085 | 0 |
| airport-hour arrival state † | `c_arr_taxi_dev_s3`, `c_arr_taxi`, `c_n_dev`, `c_ms_dev` | 213.736 | +0.220 | 0 |
| airport-hour regulation state † | `c_replan_rate_dev_s3`, `c_mvt_lobt_dev_s3`, `c_push_delay_dev_s3`, `c_replan_mag` | 213.784 | +0.268 | 0 |

† The three arms marked † were built in the same way but are not wired into `S7`, because they all read
*worse* than the control and carrying three dead builders in the script is not worth the twenty seconds
they cost every run. Their builders are recorded in §4.2 so the measurement can be repeated.

### 4.2 The three dead builders, for the record

* **destination regulation state** — group all 2,085,047 departures by (`ADES_mvt`, `ARVT_1_flt`
  truncated to the hour); per cell take the replanning rate (`lobt_minus_eobt1 ≠ 0`), median
  `mvt_minus_eobt1` and cell size; join back with the row's own contribution removed from the rate.
  1,050,835 cells, 240,628 with n ≥ 3, 50.8 % row coverage at that threshold.
* **airport-hour state** — group by (`ADEP_mvt`, `MVT_TIME_UTC_mvt` truncated to the hour); per cell take
  replanning rate, median `aobt3_minus_eobt1`, median `mvt_minus_lobt`, median `mvt_minus_sched`, median
  `arr_taxi_in_mean_60m`, cell size; subtract the airport × hour-of-day median of each; smooth over three
  consecutive hours, centred. 74,762 cells.

Per airport at LIRF, LTFM, LFPG and EGLL (one seed each, `S7` with an airport filter) no arm separates
from its control by more than 0.55 s against within-airport base-to-control gaps of 0.03–0.42 s.

---

## 5. The shortlist

Ranked by expected board effect. The repo's own conversion: a lane-A gain of X moves pooled OOF by
0.758·X (d(pooled RMSE)/d(lane-A RMSE) at the current split), and `model-a-ceiling.md` §8.2 measures
ΔBoard/ΔOOF = 0.747, so **board ≈ 0.57 × lane A**.

### C1 — `proxy_is_nominal`: flag the rows where NM's off-block stamp is a table value

* **Mechanism.** On 6.6–29.8 % of rows per airport, NM sets AOBT_3 to take-off minus a nominal per-runway
  taxi allowance rather than to an observed off-block, so on those rows the offset the pipeline subtracts
  at coefficient 1.0 carries no flight-specific information (`S2`: LEBL 30.1×, LTFM 19.9× the neighbouring
  minute; LTFM b(y~proxy) = 0.135, `S1`).
* **Columns.** `MVT_TIME_UTC_mvt`, `AOBT_3_flt`, `ADEP_mvt`, `RUNWAY_mvt`. Three outputs:
  `proxy_at_mode` (Int8), `proxy_minus_mode_min`, `proxy_sec_in_min`. The mode table is a **count** over
  feature columns — no label — so it is fitted once on the union of the training and ranking frames and
  is parity-safe by construction; do **not** fit it per frame, or train and serve will use different modes.
* **Cost.** About 12 lines in `features.py` next to `candidate_features`, one entry in `CANDIDATE_COLS`,
  one CV run. No new data, no cache rebuild beyond the usual four minutes.
* **Not already covered.** `S3`: no key in the model moves the residual by more than 0.09 s; the tree can
  reach `RUNWAY_mvt` × `proxy` but an exact-equality indicator is what a binned learner cannot cheaply
  build, and the spike is one bin wide.
* **Pre-registered effect: −0.05 to −0.15 s of board, point estimate −0.07 s.** Reasoning: instrument
  −0.070 ± 0.010 (3 seeds), transfer 0.57 → −0.12 s of lane A → ×0.57 → −0.07 s of board. It is above the
  instrument's floor at about 7σ of seed noise and below the repo's 0.25 s screening resolution, so it
  will read as a tie on a single CV run and needs the paired bootstrap to be seen at all.

### C2 — the January-2026 de-icing exposure at LTFM

* **Mechanism.** De-icing is done at the stand at some airports and on remote pads at others, so the same
  METAR gate buys a different taxi penalty per airport; the model carries one pooled response
  (`deicing_gate` interacting only through `ADEP_mvt`).
* **Columns.** `deicing_gate`, `frozen_wx`, `temp_c`, `ADEP_mvt` — all present. The change is not a new
  column but (a) P8-B's `wxcodes` repair, which replaces the dead `precip_mm` half of the gate, and (b)
  an explicit airport × gate term or a guarded offset at LTFM.
* **Cost.** Zero build; P8-B already produces `wet_wx`, `shower_wx`, `obscuration_wx`, `frozen_wx_i8`.
  One CV run plus one January-only slice.
* **Not already covered.** `S3`/§2.5: OOF bias on gate rows is +198.0 ± 51.6 at LTFM, +28.8 ± 7.3 at EHAM,
  and −9.1 ± 7.2 at EDDM. A single pooled response cannot be right at all three.
* **Pre-registered effect: −0.05 to −0.40 s of board, point estimate −0.15 s, and it may be zero or
  negative.** Reasoning: correcting a bias b on LTFM's 1,843 scored gate rows changes board SSE by
  1,843·b² against 344,841·284.36² = 2.79e10, so b = 198 gives −0.37 s and a shrunk b = 100 gives −0.09 s.
  **Risk, stated up front:** the estimate rests on 81 out-of-fold rows from December while the population
  it would be applied to is January; `P8-A-weather-remeasured.md` already recorded one sign flip
  (EGLL −33.2 → +63.3) in this exact family. Build the LTFM January slice as a separate arm and require
  it to hold on a month-blocked split before adopting.

### C3 — wind direction relative to the assigned runway

* **Mechanism.** Runway configuration is chosen by the wind and decides the taxi route; the model knows
  *which* runway but nothing about whether the airport is operating with, across or against the wind —
  the state in which configurations flip and aircraft are re-routed mid-taxi.
* **Columns.** `drct` and `sknt` from `data/ref/metar/*.csv` (already fetched by
  `src/prc/weather.py:23`, dropped at `:84-113`), plus `RUNWAY_mvt` → heading. Outputs `headwind_kt`,
  `crosswind_kt`, `rwy_wind_misalign`, `wind_dir_deg`.
* **Cost.** Four lines in `weather.py` (keep `drct` through the hourly aggregation), a 6-line heading
  parser, four columns in `features.py`. No new fetch — the cache already holds 138,457 observations to
  2026-08-01.
* **Not already covered.** Wind *direction* has never been in the feature set; `wind_ms` and `gust_ms` are
  scalars. `S3`: `wind_ms` reads −0.056, inside the −0.051 control.
* **Pre-registered effect: −0.00 to −0.06 s of board, point estimate −0.03 s.** Reasoning: instrument
  −0.045 ± 0.005 → −0.08 s of lane A → −0.045 s of board. Coverage caveat: `drct` is null on 10.0 % of
  cached observations overall and 30.3 % at LSZH, so the column must pass `src/prc/coverage.py` before it
  is proposed for production.

### Not recommended, and why

* **Neighbour dispersion / tail mass** (+0.085 vs control) — the obvious second moment of the model's most
  valuable family, and it makes the residual worse. §2.1 explains it.
* **Airport-hour state aggregates** (+0.22 to +0.27) — the direct attack on §3.2's five seconds. The state
  is real; these are not measurements of it.
* **Destination regulation state** (+0.008) — 50.8 % row coverage at n ≥ 3 per destination-hour, and no
  signal.
* **`gust_ms` null-versus-calm repair** (+0.000) — worth doing as hygiene (`gust_ms` is null on **96.6 %**
  of scored rows and 97.7 % of training rows because METAR omits the group when there is no gust, so the
  column conflates "steady wind" with "no observation"), but it buys nothing measurable and should ride
  along with C3 rather than be measured on its own.
* **Anything reading `ARVT_3_flt`** — legal (0.74 % null on scored arrival rows) but it is the landing
  stamp (median `ARVT_3 − MVT_TIME` = +3 s, sd 57.5 s), so for a departure it is post-take-off information
  about the *other* end of the flight with no taxi-out mechanism attached.

---

## 6. Two leads on record, judged

**EHAM de-icing and the EGLL sign flip.** `[S]` The EGLL half is moot: `feat_rank` has **zero** EGLL
de-icing-gate rows in either scored month (§2.5). EHAM is real but small: +28.8 ± 7.3 s on 792 OOF rows
and 1,037 scored rows, worth 1,037·28.8² / 2.79e10 → **−0.015 s of board** if corrected exactly. The
6,729-versus-2,505 January count is confirmed here as 6,729 scored gate rows in 2026-01 against 9,157 in
the whole of 2025 training, and the airport that carries the exposure is **LTFM (1,843)**, not EHAM.
That is why C2 is written about LTFM.

**LFPG `lobt_minus_eobt1` +22.2 points in July 2026.** `[S]` Not a feature lead — a composition shift.
At LFPG, rows with `lobt_minus_eobt1 ≠ 0` have lane-A RMSE **251.2** against **238.4** for the rest
(18,252 and 58,219 OOF rows). If the replanned share goes 23.9 % → 45.6 %, LFPG's lane-A RMSE rises to
about **244.0** from mix alone, on 11.6 % of scored rows, with no model change. `[X]` Expect roughly
+0.3 s of board in July 2026 that is nobody's fault and that no feature will remove; it is worth knowing
before reading a July regression as a defect. Two related facts found while checking it: `LTFM` has
`lobt_minus_eobt1 = 0` on **100 %** of its 90,041 lane-A rows — the column is dead at that airport — and
LIRF's replanned rows (24,280) carry **11.5 % of all lane-A SSE**, the largest single cell in the
airport × replanned table.

---

## 7. What I would build first

**C1, the nominal-proxy flag.** Not because it is the largest — C2 has the larger expected board effect —
but because it is the only candidate here that is a *mechanism statement about the data-generating
process* rather than a correlation, it is the only one that beat its control at more than 5σ of the
instrument's seed noise, it costs twelve lines and one CV run, and it attacks the exact place the
pipeline is structurally weakest: a reparameterization that subtracts `proxy` at 1.0 on rows where
`proxy` is a lookup table.

Build C2's measurement at the same time — it is free, it needs no new column, and its January-only slice
is the arm that decides whether the one genuinely large de-icing bias on the board is real or an artefact
of 81 December rows.

**[X] And I would not spend a third week on features for lane A.** §3.2 says the recoverable error is a
time-local airport state worth about five seconds; §3.3 says nothing in these files measures it; §4 says
the screen would have seen a −0.5 s candidate and saw nothing above −0.07. The next second is more likely
to come from the unmatched lane, from model capacity, or from a new observation — not from a new column
built out of what is already here.

---

## 8. Caveats, recorded rather than buried

1. **The instrument measures additive residual explanation.** A column can also help by changing which
   splits the main model makes, which this screen cannot see. The §6 calibration bounds that gap: a
   feature whose full-model gain was −3.21 s still read −1.83 s here, so the screen sees 57 % of a real
   effect, not 0 %. `[X]` It is a conservative screen, not a blind one.
2. **Lane A only.** Lane B (8,378 pooled rows, RMSE 1,606.9) carries 41.8 % of pooled SSE and is not
   touched here; `missing-clock.md` puts its ceiling at about five seconds and this note does not revisit
   that.
3. **The §3.2 oracle is partly the model's own local error, not new information.** Rows in one
   airport-hour cell sit in the same fold and share a fitted model, so their residuals are correlated by
   construction. That is exactly `model-a-ceiling.md` §4's ρ — the *systematic, recoverable* share — so
   the −5.39 s is a real recoverable quantity and not an artefact, but it is a restatement of a known
   number at a finer key, not an independent discovery of new headroom.
4. **Three seeds, not five.** Arm-to-arm differences below about 0.02 s should not be read.
5. **The LTFM de-icing bias rests on 81 out-of-fold rows.** Everything C2 claims inherits that.
