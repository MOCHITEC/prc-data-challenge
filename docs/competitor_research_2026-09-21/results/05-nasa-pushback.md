# Slot 05 — nasa-pushback: NASA / DrivenData "Pushback to the Future", code-level

Investigator: SubAgent (slot 05). Date: 2026-09-21.
Marking: **[S]** = read from a cited source (URL or path in the cloned repo). **[X]** = my own inference.

Everything marked [S] with a path like `1st Place/Phase 1/...` was read from a local clone of
`https://github.com/drivendataorg/nasa-airport-pushback` (cloned to `/tmp/nasa-airport-pushback`,
shallow clone of `main`, 2026-09-21). Paths like `nasa-airport-config/...` are from
`https://github.com/drivendataorg/nasa-airport-config` (`/tmp/nasa-airport-config`).

**Correction to prior repo knowledge.** The existing note "NASA pushback winners: all per-airport
models" is *incomplete and, on the one point where it was measured, wrong*. The 1st place solution
is an ensemble of **three per-airport models plus one global model** [S], and the 4th place team
published a controlled comparison in which, on the full feature set, the **global LightGBM beat the
per-airport LightGBM** (cumulative MAE 10.7232 vs 10.7686) [S]. Details in §4 and §5.

---

## 1. The competition pages

| Page | URL | Note |
|---|---|---|
| Competition group | https://www.drivendata.org/competitions/group/competition-nasa-airport-pushback/ | Open Arena (409 participants, no prizes), Prescreened Arena (49 participants, $50,000, US-university affiliation required), Phase 2 federated (13 participants, finalists only) [S] |
| Open Arena | https://www.drivendata.org/competitions/149/competition-nasa-airport-pushback/page/677/ | [S, linked from repo READMEs] |
| Prescreened Arena | https://www.drivendata.org/competitions/182/competition-nasa-airport-pushback-prescreened/page/714/ | the arena the prizes and published scores come from [S] |
| Phase 2 (federated) | https://www.drivendata.org/competitions/218/competition-nasa-airport-pushback-phase2/page/765/ | [S] |
| Benchmark blog | https://drivendata.co/blog/airport-pushback-benchmark | [S] |
| Winners blog | https://drivendata.co/blog/airport-pushback-finalists | [S] |
| Winners' code | https://github.com/drivendataorg/nasa-airport-pushback | [S] |

**Not found:** DrivenData published no per-arena leaderboard snapshot that I could retrieve, and there
is **no separate prize list for the Open Arena** — the Open Arena was a non-prize development arena
[S, group page]. The 1st-place team's own public repo `github.com/alsaco33/NASA-pushback`, linked from
their Phase 2 README [S, `1st Place/Phase 2/README.md`], **returns HTTP 404** as of 2026-09-21 [S] —
so their personal write-up is gone; only the DrivenData mirror survives.

## 2. Problem setup, and how it differs from ours

[S] `4th Place/Phase 1/reports/2023__KDD2023_Undergraduate_Consortium (2).pdf` (the FLHuskies KDD 2023
paper) plus the benchmark blog:

- Target: `minutes_until_pushback` = minutes from a prediction timestamp until the aircraft's **actual
  gate departure**. Integer minutes.
- Predictions are made **repeatedly for the same flight**, on a 15-minute grid, at every timestamp in
  the 5 hours before pushback. The 1st place target builder makes this explicit: for each flight it
  emits rows on `date_range(pushback - 300min, pushback, freq="15min")` [S,
  `1st Place/Phase 1/training/src/pushback_nasa/pipelines/create_targets/nodes.py`]. So there are
  ~6+ rows per flight (24.0 M instances from 3.51 M gufis) [S, KDD Tab. 2].
- **Availability rule:** at prediction time `t`, only auxiliary rows with timestamp `t' <= t` may be
  used, and only back to `t - 30 h` [S, benchmark blog; KDD §2]. The actual runway, actual runway
  departure time and actual gate departure time of the flight itself are unknown at `t`; the same
  quantities **for flights that have already departed are known and are the basis of all congestion
  features** [S, KDD §2].
- Metric: **MAE in minutes** [S, repo README].
- Baseline: `minutes_until_etd - 15`, i.e. NASA Fuser's estimated runway departure time minus 15 min.
  MAE **14.3250** on the FLHuskies validation set [S, KDD Tab. 4].

**[X] Structural differences from PRC 2026 that govern transfer:**

| | NASA pushback | PRC 2026 (ours) |
|---|---|---|
| metric | MAE (minutes) | **RMSE (seconds)** |
| rows per flight | ~6–20 (rolling 15-min grid) | 1 |
| anchor | a *prediction time* before pushback; the future is unknown | the **take-off time `MVT_TIME_UTC_mvt` is given for every ranking departure** [S, verified: `ranking.parquet` DEP rows have `MVT_TIME_UTC_mvt` null-rate 0.0] |
| strongest proxy | Fuser ETD (a live, repeatedly-revised estimate) | `AOBT_3_flt` (a single NM-reported off-block time) |
| realised taxi times of recent departures | available at inference | **hidden**: `TAXITIME_SEC_mvt` and `BLOCK_TIME_UTC_mvt` are null for 100% of ranking DEP rows [S, verified] |
| arrival taxi-in of recent arrivals | available | **available**: ranking ARR rows have `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` null-rate 0.0 [S, verified] |

That last pair of rows is the single most important finding for transfer and is spelled out in §9.

## 3. Final standings (Prescreened Arena, Phase 1)

[S] repo README table, cross-checked against KDD §4.2.4:

| Place | Team | Phase 1 MAE (min) | Phase 1 model | Phase 2 MAE |
|---|---|---|---|---|
| 1 | Team CDS (Priya Dhond, Alejandro Sáez — NYU) | **10.673** (10.6734) | ensemble of multiple CatBoost regressors per airport | 16.329 |
| 2 | Moles (Brian Hu, Nika Chuzhoy — Caltech) | 10.728 (10.7283) | one CatBoost regressor per airport | 36.589 |
| 3 | Oracle 2 (Suraj Rajendran, Matthew Love, Prathic Sundararajan — Weill Cornell) | 11.054 (11.0543) | per-airport XGBoost regressor + XGBoost over/under-estimate corrector | 12.848 |
| 4 | FLHuskies2 (UW Tacoma: Filienko, Lin, Tomlin, Robison, Pentyala, Nascimento, De Cock) | 11.105 (11.1046) | one LightGBM regressor per airport | 104.274 |
| 5 | Cuong_Syr (Syracuse: Kotary, Tran, Nguyen, Tran) | 11.946 | one XGBoost regressor per airport | 14.519 |

Prizes $15k / $12k / $8k / $7.5k / $7.5k [S, winners blog]. Spread 1st→5th is 1.27 min (12%); 1st→4th
is 0.43 min (4%). **[X] The top four are statistically close; the method differences below are worth
more as a catalogue of ideas than as a ranking of them.**

Additional public artefacts by the winners:
- FLHuskies KDD 2023 paper, bundled in the repo at `4th Place/Phase 1/reports/2023__KDD2023_Undergraduate_Consortium (2).pdf` [S].
- FLHuskies public working repo `https://github.com/krobison10/FLHuskies_PTTF` — exists, contains the
  competition scripts and the KDD paper [S].
- The other four teams have no public write-up beyond the `reports/DrivenData-Competition-Winner-Documentation.pdf`
  in the DrivenData repo, as far as I could find. **Not found:** any blog post, arXiv paper or slide
  deck by teams CDS, Moles, Oracle 2 or Cuong_Syr.

---

## 4. 1st place — Team CDS (CatBoost, 4 model families, 298 features)

Source: `1st Place/Phase 1/` [S] and its `reports/DrivenData-Competition-Winner-Documentation.pdf` [S].

### 4.1 Model families and the ensemble

Four families, all CatBoost [S, `submission/utilities.py::generate_predictions`,
`training/src/pushback_nasa/pipelines/train_models/nodes.py`, README]:

| family | scope | target |
|---|---|---|
| v0 | **one per airport** | `minutes_until_pushback − etd_time_till_est_dep` (residual against the ETD proxy) |
| v1 | **one per airport** | `minutes_until_pushback` (raw) |
| v2 | **one per airport** | same as v0, different parameters |
| global | **one across all 10 airports**, `feat_cat_airport` as a categorical | `minutes_until_pushback` |

The **global model deliberately sees only a subset of the columns**: `build_global_master` keeps
`gufi, timestamp, target` plus every column whose name contains `mfs` or `etd`, i.e. only the
flight-level identity/aircraft block and the ETD-revision block — **no weather, no congestion, no
runway config** [S, `pipelines/create_master/nodes.py::build_global_master`]. **[X] So the global
member is a deliberately decorrelated, low-variance ensemble partner, not a competitor to the
per-airport models.**

Ensemble [S, `submission/utilities.py:638-681`]:

```
for v in [0,1,2]:
    pred_v = model[v].predict(...)
    if v in (0,2): pred_v = clip(pred_v + etd_time_till_est_dep, 1, 299).astype(int)
    else:          pred_v = clip(pred_v,                        1, 299).astype(int)
global_pred = model["global_model"].predict(...)
minutes_until_pushback = clip((pred_0 + pred_1 + pred_2 + global_pred) / 4, 4, 260).astype(int)
if airport in ("KPHX","KMIA","KJFK"):
    minutes_until_pushback = clip(pred_1, 4, 260).astype(int)
```

Three points worth lifting:
1. **Equal weights**, no stacking, no weight search — just a mean of four.
2. Each member is **individually reconstructed and clipped to [1,299] and cast to int before
   averaging** — the residual members have the ETD proxy added back first.
3. **Per-airport override**: at KPHX, KMIA and KJFK the ensemble is thrown away and the plain-target
   per-airport model v1 is used alone. The README says the ensembling architecture was chosen to
   "minimize MAE in a held-out validation sample" [S], so this is a per-airport selection of ensemble
   structure on validation.

### 4.2 Hyperparameters

Identical for v0/v1 (v2 is described as "same as v0 with some parameter changes"; the changed values
are not in the repo — **not found**) [S, `train_models/nodes.py`]:

```
CatBoostRegressor(eta=0.01, depth=7, rsm=1, subsample=0.8, max_leaves=21,
                  l2_leaf_reg=3, min_data_in_leaf=5000, n_estimators=20000,
                  grow_policy="Lossguide", has_time=True, random_seed=4,
                  loss_function="MAE", boosting_type="Plain",
                  max_ctr_complexity=12, bootstrap_type="Bernoulli")
model.fit(..., eval_set=(x_val,y_val), use_best_model=True,
          cat_features=cat_features, early_stopping_rounds=60)
```

Notable: **`has_time=True`** (CatBoost stops randomly permuting rows for ordered target statistics and
uses the given order — the correct setting for time-ordered data), `min_data_in_leaf=5000` with
`max_leaves=21` (very heavily regularised, shallow-effective trees) and a very low `eta=0.01` with a
20,000-tree budget and early stopping. `max_ctr_complexity=12` allows deep categorical combinations.
The "hyperparameter search" in the shipped code is a degenerate loop over `depth in [7]`,
`eta in [0.01]` — i.e. the search was done offline and the winning point hard-coded [S].

### 4.3 Target transform and loss

- Loss is `MAE` (CatBoost's L1), matching the metric [S].
- Two of the four families regress the **residual against a proxy**: `target − etd_time_till_est_dep`,
  where `etd_time_till_est_dep` is the signed minutes from the prediction timestamp to the latest
  Fuser ETD [S, `train_catboost_diff`]. The proxy is added back at predict time.
- Training rows are filtered: `timestamp > "2020-11-02"` **and `target != 0`** [S]. Dropping the
  exact-zero rows removes the degenerate last row of each flight's grid.

### 4.4 Validation and leakage control

- Data span 2020-11-01 → 2022-12-02; `end_train = "2022-09-01"` [S, `conf/base/parameters.yml`].
  **A single forward time split**: everything before 2022-09-01 trains, the last ~3 months validate,
  `use_best_model=True` + `early_stopping_rounds=60` on that split. The README says the held-out test
  was 2022-09 … 2022-12 [S].
- The report answers the "what didn't work" question with: *"Time based cross validation: Averaging
  across folds based on time splits did not improve the performance of the models either"* [S].
- Leakage control is structural, not procedural: every feature block is built from rows at or before
  the timestamp, and at inference the feature window is `prediction_time − 26h → prediction_time`
  [S, `submission/utilities.py::catboost_predictions`].
- Federation-awareness also shaped it: *"we did not leak information that is airline-specific to
  other flights"* [S, report Q4].

### 4.5 The complete feature catalogue (298 features)

All from `training/src/pushback_nasa/pipelines/feature_extraction/nodes.py` [S], mirrored in
`submission/utilities.py`.

**(a) Flight identity / aircraft (`mfs_*`), per gufi-timestamp**
`mfs_cat_flightid` (GUFI token 0, bucketed against a hard-coded whitelist of ~38,700 flight ids else
`OTHER`), `mfs_cat_airline` (non-digit prefix of the flight id), `mfs_cat_flightno` (digits, whitelisted),
`mfs_flightno_length`, `mfs_cat_no_last` (last digit of the flight number), `mfs_cat_arrival`
(destination airport = GUFI token 2, whitelisted against ~100 airports else `OTHER`), `mfs_cat_last`,
`mfs_cat_sectolast` (last two GUFI tokens), `mfs_cat_hourdep` (hour of the scheduled departure parsed
out of GUFI tokens 3+4), `mfs_time_diff` (signed minutes between that scheduled departure and the
prediction timestamp), `mfs_cat_engineclass`, `mfs_cat_type` (whitelisted aircraft type),
`mfs_cat_majorcarrier` (whitelist `AAL UAL DAL JBU FDX SWA` else `OTHER`), `mfs_cat_flighttype`,
`mfs_departure`.
**[X] The whitelist-else-OTHER pattern is a deliberate cardinality cap that keeps CatBoost's CTRs stable.**

**(b) Runway configuration (`config_*`), airport-level, resampled to 5 min and ffilled**
- `config_arrival_cat_{r}` / `config_departure_cat_{r}` for each runway number `r`: a 5-level code
  `0=not in use, 1=…R, 2=…L, 3=…C, 4=plain` [S].
- `config_n_active_dep`, `config_n_active_arr` = number of runways in the config string (comma count + 1).
- `config_change_dep`, `config_change_arr` = boolean "config differs from the previous 5-min slot".
- `config_n_chang_dep_last_{p}`, `config_n_chang_arr_last_{p}` for `p ∈ {6,12,24,36,48,120}`
  **periods of 5 min** = number of changes in the last 30 min / 1 h / 2 h / 3 h / 4 h / 10 h.

**(c) ETD revision stream (`etd_*`), per gufi**, resampled to 15 min, last-wins, ffilled per gufi
- `etd_est_change` = signed minutes between consecutive ETD values for the same flight (0 at the
  flight's first row).
- `etd_diff_withfirst` = signed minutes between the current ETD and the **first ETD ever issued** for
  that flight — i.e. cumulative schedule drift.
- `etd_time_till_est_dep` = signed minutes from now to the current ETD. (This is the proxy.)
- `etd_roll_{p}_{sum,max,std}` of `etd_est_change` over rolling `p ∈ {60,120,240,360,720,1440}` minutes,
  **grouped by gufi**. 18 features describing how violently this flight's own estimate has been moving.

**(d) Calendar (`mom_*`)**: hour, day-of-week, month, day-of-month, ISO week-of-year, week-of-month,
`is_holiday` (US federal calendar), **`mom_days_until_holiday`** (days to the next federal holiday).

**(e) Weather (LAMP) — the most elaborate block**
- Past: 6-hour rolling `mean`/`min`/`max` of every LAMP variable, resampled to 15 min
  (`temp_{var}_{agg}_last6h`).
- **Forecast-lead bins**: mean of all forecasts with lead ≤ 3.5 h (`temp_{var}_next3`) and mean of all
  forecasts with lead in (3.5, 6.5] h (`temp_{var}_next36`).
- **Past change**: for `i = 1..6`, `change_{var}_{i}` = the variable minus its value `i` issues earlier
  *for the same `forecast_timestamp`* — i.e. how the forecast for a fixed future moment has been revised.
- **Future change**: for `i ∈ {1,3,6,9,12,24}`, `temp_futchange_{var}_{i}` = (value at lead 0.5 h) −
  (value at lead `i`+0.5 h) — the forecast gradient.
- Categorical encodings: `lightning_prob {L:0, M:1, N:2, H:3}` (note: **not monotone**, "None" sits
  between medium and high — an apparent bug that evidently did not matter), `cloud {CL:0, FW:1, SC:2,
  BK:3, OV:4}`, `precip` → float.
- Variables covered: temperature, wind_direction, wind_speed, wind_gust, cloud_ceiling, visibility,
  cloud, lightning_prob, precip.

**(f) Realised movements (`runways_*`), airport-level, on a 15-min grid**
`runways_{p}_totdep` and `runways_{p}_totarr` = count of **actual** runway departures / arrivals in the
last `p` minutes, for `p ∈ {15, 30, 60, 90, 120, 360, 720, 1440}` [S]. Sixteen nested cumulative
counters. **This is the whole congestion block.**

**(g) Stand times / realised pushback delay (`standtimes_*`), airport-level**
- Per flight: `delay` = actual departure stand time − **first** ETD ever issued for that flight, signed.
- Rolling over `p ∈ {1,2,4,6,12,24}` hours: `standtimes_maxdelay_{p}`, `standtimes_meandelay_{p}`,
  `standtimes_count_{p}`. So: how badly have *recently pushed-back* flights been running late, and how
  many of them were there.

**(h) TFM / TBFM (arrival demand) — built but NOT used.** `extract_tfm_features` /
`extract_tbfm_features` compute, for flights whose estimated runway *arrival* is within the next 3 h:
`n_gufis`, total/max/mean revision of the arrival estimate, and delay-per-gufi [S]. The merge lines are
**commented out** in `build_master` [S], and the report confirms *"TBFM/TFM features: These did not
improve the performance of the models"* [S]. **[X] Notable negative result: pure arrival-demand
look-ahead did not pay, while realised-movement look-back (block f) did.**

**Imputation / dtype:** `ffill().bfill().fillna(0)`, then **every non-categorical column is cast to
`int16`** [S, `adjust_types`]. **[X] This silently truncates all fractional features (rolling means,
std, weather) to integers and would overflow anything beyond ±32767. It apparently cost nothing, which
is itself evidence about how coarse the usable signal is.**

### 4.6 Post-processing and robustness

- Clip each member to `[1,299]`, cast to int; clip the ensemble to `[4,260]`, cast to int [S].
- **Three-level fallback chain** in `predict()` [S, `submission/solution.py`]:
  `catboost_predictions` → on any exception, `baseline_predictions` (= latest ETD − timestamp − 30 min,
  clipped to [1,299], NaN→15) → on any exception, a constant 30 minutes.
  **[X] A submission-safety pattern worth copying verbatim for a runtime-scored competition.**

---

## 5. 4th place — FLHuskies2 (LightGBM; the only team that *measured* global vs per-airport)

Source: `4th Place/Phase 1/` [S] and the KDD paper [S].

### 5.1 Model

```
lgb.train({"objective": "regression_l1", "metric": "mae",
           "num_leaves": 4096, "n_estimators": 128}, lgb.Dataset(X_train, y_train))
```
one model per airport, trained on the whole table with **no eval set and no early stopping** in the
shipped script [S, `src/train.py`]. `num_leaves=4096` and `n_estimators=128` were chosen with Optuna
on the validation set [S, KDD §4.2.1]. Learning rate left at the LightGBM default (0.1) [S]. No target
transform; the raw target with an L1 objective.

### 5.2 Global vs per-airport — the actual numbers

[S] KDD Tab. 4, cumulative MAE on their validation set:

| airport | baseline ETD−15 | MLP global | MLP local | **LGBM global** | **LGBM local** |
|---|---|---|---|---|---|
| KATL | 11.6135 | 8.0039 | 7.9851 | 7.8549 | 7.8682 |
| KCLT | 12.4652 | 9.4824 | 9.5111 | 9.3135 | 9.3352 |
| KDEN | 15.1364 | 12.0029 | 12.1098 | 11.9466 | 11.9446 |
| KDFW | 15.5376 | 12.3230 | 12.2801 | 12.0823 | 12.1768 |
| KJFK | 14.7204 | 11.5790 | 12.0592 | 11.1881 | 11.1143 |
| KMEM | 25.9130 | 18.7407 | 18.8432 | 17.7555 | 17.8176 |
| KMIA | 14.6529 | 12.4397 | 12.9948 | 11.9421 | 12.0881 |
| KORD | 14.5247 | 11.3000 | 11.4879 | 11.1351 | 11.2458 |
| KPHX | 12.5830 | 9.7497 | 9.6736 | 9.3849 | 9.4106 |
| KSEA | 10.1884 | 7.5612 | 9.6736 | 7.2195 | 7.2300 |
| **All** | **14.3250** | 10.9883 | 11.0500 | **10.7232** | **10.7686** |

Their own conclusion, verbatim: *"we observed that for smaller subsets of features the 'local model'
based approach would result in lower MAEs than the 'global model' based approach. When training
LightGBM models over the entire feature set from Tab. 3 however, the global model and the local model
based approaches are at par"* [S]. And the abstract: *"training one local model for each airport is
more memory efficient, while yielding a mean absolute error at par with a global model"* [S].

**[X] The operative rule is therefore: per-airport wins when the feature set is thin and the airport
identity has to be carried implicitly; a rich, explicitly airport-aware feature set makes the choice
a wash. Choosing per-airport on principle is not supported by the only measurement anyone published.**

### 5.3 Feature catalogue, with exact window definitions

From `src/table_scripts/*.py` [S] and KDD Tab. 3 [S]. `now` = prediction timestamp; all raw tables are
pre-filtered to `(now-30h, now]` by `filter_by_timestamp` [S].

**Congestion / demand — the block most relevant to us:**

| feature | exact definition [S] |
|---|---|
| `deps_3hr`, `deps_30hr` | count of rows in `runways` with a non-null `departure_runway_actual_time` in the last 3 h / 30 h — **realised take-offs** |
| `arrs_3hr`, `arrs_30hr` | same for `arrival_runway_actual_time` |
| `deps_taxiing` | **a state, not a flow**: inner-join `mfs(isdeparture)` × `standtimes`, left-join `runways`, keep rows where `departure_runway_actual_time` is null → flights that have pushed back but have not yet taken off |
| `arrs_taxiing` | rows of `runways` with a landing time, left-joined to `standtimes`, keeping those with a null `arrival_stand_actual_time` → landed but not yet on-block |
| `exp_deps_15min`, `exp_deps_30min` | **per-flight, not per-airport**: take *this* flight's latest ETD `T`, count all flights whose latest ETD falls in `[T−15, T+15]` (resp. ±30). A **symmetric window centred on the flight's own planned time** — a "departure-bank density", not a clock-window count |
| `standtime_30hr` | mean over the last 30 h of (`first_position.timestamp` − `departure_stand_actual_time`) for arriving flights — the origin-side turnaround signal |
| `dep_taxi_30hr` | mean over the last 30 h of (`departure_runway_actual_time` − `departure_stand_actual_time`) = **mean realised taxi-out of recent departures** |
| `arr_taxi_30hr` | mean over the last 30 h of (`arrival_stand_actual_time` − `arrival_runway_actual_time`) = **mean realised taxi-in of recent arrivals** |
| `minutes_until_etd` | (latest ETD − now) in minutes, cast to int — the proxy |
| `delay_nhr` (KDD Tab. 3), `gufi_lifespan` | listed in the paper's feature table; **`gufi_lifespan` is not in the Phase 1 `features` list in `train.py`** [S] — the paper's set is a superset of the shipped Phase 1 set |

**Reported top-5 importances at KMEM** [S, report]: `gufi_flight_destination_airport`,
**`exp_deps_15min`**, **`dep_taxi_30hr`**, **`arr_taxi_30hr`**, **`standtime_30hr`**. At KATL the paper
names ETD first, then **gufi lifespan**, then destination airport, then `exp_deps_30min`,
`dep_taxi_30hr`, `standtime_30hr` [S, KDD §4.2.2].
**[X] Three of the five most valuable features at KMEM are rolling means of *realised durations*
(taxi-out, taxi-in, turnaround) over a 30-hour window — not counts. Counts alone were not the win.**

**Runway configuration:** just the **raw config strings** `departure_runways` and `arrival_runways`
taken from the most recent `config` row, ordinal-encoded as opaque categoricals [S, `add_config.py`].
No derived counts in Phase 1. (Phase 2 adds a large derived block — §5.6.)

**Weather:** the single most recent LAMP forecast whose `forecast_timestamp` equals `now` rounded to
the hour; if absent, walk back hour by hour up to 30 h; if still absent, numerics → 0 and categoricals
→ `"UNK"` [S, `add_lamp.py`]. **Only the nowcast** — the report lists *"Utilizing weather predictions
for the next n hours instead of including exclusively current weather from LAMP"* among the things
tried that did not make it in [S].

**Identity/time:** `gufi_flight_major_carrier` and `gufi_flight_destination_airport` parsed from the
GUFI; `minute, hour, day, weekday, month, year`; `aircraft_engine_class, aircraft_type, major_carrier,
flight_type`.

### 5.4 Validation scheme

**A group split by flight, not a time split** [S, `src/preprocessing.py::train_test_split`]: the set of
GUFIs that formed the *Open Arena* validation set is used as the validation set here; every remaining
GUFI trains. Rationale given: *"separating the resulting dataset into training and validation dataset
by gufi, so that the same flight does not occur in both training and validation datasets, and both
datasets have flights that have varying departure times, in order to best emulate a real deployment
scenario"* [S, KDD §3.1]. The leaderboard models were then **retrained on train+validation** [S].
**[X] Note they did not guard against temporal leakage at all — same-day flights sit on both sides.
With ~6 correlated rows per flight, the grouping by GUFI was the binding constraint; the time
structure was not.**

### 5.5 Encoding, target, post-processing

- sklearn `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)` fitted **across all
  airports jointly**, pickled for inference [S, `src/encoders.py`].
- Missing handling: ffill, or mean, or an explicit `"UNK"` level — *"Unlike Kiliç and Sallan we do not
  remove rows with missing attributes"* [S, KDD §3.1].
- No target transform, no ensembling, **no clipping or rounding** [S].

### 5.6 Phase 2 feature additions (worth reading even though Phase 2 scored badly)

`4th Place/Phase 2/inference/` [S] adds:
- `add_etd_features.py`: **look-ahead departure demand done correctly**. Each ETD row is replicated
  forward in 15-min steps (up to 360 min) as long as `timestamp < ETD`, so that at any grid timestamp
  the airport's currently-known ETD set is materialised; then `estdep_next_{30,60,180,360}min` = count
  of flights whose current ETD falls within the next N minutes, plus a total `gufi` count.
- `add_runway_features.py`: `arrivals_last_{5,10,15,30,60,120,360}min` and
  `deps_last_{5,10,15,30,60,120,360}min`; `nunique` of runways used; per-runway binary "active for
  departure / arrival" flags including L/R disambiguation; rolling means of those flags over
  `{4,8,12,16,20,24}` 15-min periods (1 h … 6 h); **categorical previous configuration at lags 1..24**;
  `nchanges_last_{4,8,12,16,20,24}` = `nunique` of the config over that lag window.
  (This block is textually near-identical to 1st place's `config` block and to the *airport-config*
  competition's 1st-place code — **[X] it looks like it was adapted from the published `nasa-airport-config`
  winner's code**, see §8.)
- `add_global_lamp.py`, `add_estimated_flight_time.py`.

---

## 6. 2nd place — Moles (one CatBoost per airport, plus a "negative prediction" repair model)

Source: `2nd Place/Phase 1/` [S].

**Model** [S, `src/run_training.py`]:
```
CatBoostRegressor(iterations=800, learning_rate=0.12, l2_leaf_reg=15,
                  loss_function="MAE", eval_metric="MAE",
                  max_depth=9, max_bin=63, thread_count=-1)
```
per airport, with per-airport iteration overrides: **KMEM 1500, KJFK 1000, KMIA 1200** [S] — i.e. the
airports with the worst MAE got more trees. Fitted on a `Pool` with `cat_features = [aircraft_engine_class,
major_carrier, aircraft_type, flight_number, flight_arrival]`, **no eval set, no early stopping** [S].

**Scope:** strictly per-airport; the report says they considered a global model *"not ideal for the small
number of airports we were given"* and lists *"Partial globalization of the model, such as incorporating
delays at airports other than the airport a flight is departing from"* among things that did not make it
in [S].

**The second-stage repair model — their only "ensembling"** [S, `make_dataset.py::get_negative_predictions`,
`run_training.py::train_negative_model`, `run_inference.py`]:
1. Score the training set with each airport's main model.
2. Pool, **across all ten airports**, every training row whose prediction was `< 0`.
3. Train one small global CatBoost on just those rows: `iterations=800, learning_rate=0.01,
   l2_leaf_reg=20, max_depth=4, max_bin=63, loss_function="MAE"`.
4. At inference, any row whose main prediction is `< 0` is **re-predicted by that model**.

**[X] This is a targeted specialist for the region where the main model is structurally wrong,
selected by the main model's own output rather than by a data partition. It is the same shape as a
"model B for hard rows" and is the cheapest idea in the whole repository.**

**Features** [S, `src/make_dataset.py`]:
- ETD block via `merge_asof(..., by="gufi", direction="backward")`: `minutes_until_departure`
  (ETD − now), `minutes_after_etd` (now − the timestamp of the ETD message actually used),
  **`etd_slope`** = Δ(ETD) / Δ(timestamp) between consecutive messages for the same flight, and
  **`etd_slope_cumsum`** = cumulative Δ(ETD) / cumulative Δ(timestamp) since the flight's first message.
  Leakage is prevented by NaN-ing the shifted columns at gufi boundaries — the report presents exactly
  this as their leakage-prevention example [S].
- LAMP: each forecast row is replicated at `forecast_timestamp + 15/30/45 min` so a backward
  `merge_asof` on `forecast_timestamp` always finds a row [S, `get_lamp_features`].
- **Cyclical time encoding**: sin/cos of hour (period 24), day-of-week (7), day-of-month (**period 365**
  — apparently a bug, `dt.day` is 1–31), month (12), plus raw `year` [S]. The report explains the sin+cos
  pair as preventing collisions [S].
- Aircraft type is **whitelisted per airport** to the 9–15 most common types, everything else `"Other"` —
  a hard-coded dict of 10 lists in both the training and the inference script [S].
- `flight_number` and `flight_arrival` parsed from the GUFI, kept as high-cardinality categoricals.
- **Dropped on purpose:** `cloud_ceiling`, `flight_type`, `wind_speed` (and `isdeparture` per the report) [S].
- **No congestion features at all in the final model.** Two were written and abandoned:
  `merge_runway_features` computing `dep20_time` = the time span covered by the last 20 departures
  (an inverse departure rate), marked *"Runway feature ended up not being used in the final model"*,
  and `get_queue_feature` computing a **rank of the flight's ETD among all flights expected to depart
  in the same 15-min bucket** — marked *"Queue feature ended up not being used in final model"* [S].
  The report confirms: *"A feature representing the length of the queue ahead of each flight, i.e. the
  number of flights scheduled to depart before a given airplane"* and *"Features incorporating the rates
  of arrivals/departures prior to a given flight"* were both tried and dropped [S].

**[X] A 2nd place finish 0.055 min behind 1st with zero congestion features is the strongest single
piece of evidence in this slot that, once a good proxy plus flight identity is present, congestion
features are close to free-riding. This matches our own `features_log.md` result of 6 straight
rejections on the congestion axis.**

**Validation:** the report cites MAE as primary, MSE and "tendency to overfit" as secondary [S]; the
shipped training script has no held-out set. **Not found:** any description of how the iteration counts
(800/1000/1200/1500) were chosen.

**Post-processing:** none beyond `astype(int)` on the concatenated submission [S, `utils.py::postprocess_predictions`]
and the negative-prediction repair.

---

## 7. 3rd and 5th places

### 7.1 3rd — Oracle 2 (XGBoost, outlier-trimmed training, over/under-estimation corrector)

Source: `3rd Place/Phase 1/Train_Models.ipynb`, `Run_Inference.ipynb`, `helper.py` [S] and the report [S].

Three models per airport, **all `xgb.XGBRegressor()` / `xgb.XGBClassifier()` with default
hyperparameters — no tuning whatsoever** [S]:

1. **Internal regressor.** `GroupShuffleSplit(n_splits=1, train_size=0.4, test_size=0.6,
   random_state=42, groups=gufi)`. Fit on the 40%.
2. **Outlier trimming.** Predict on the 60%; compute `|error|`; drop rows with error above a
   **per-airport threshold — 30 min for the "bad" airports {KDFW, KJFK, KMEM, KMIA}, 20 min for
   {KATL, KCLT, KDEN, KORD, KPHX, KSEA}** [S]. Train the **final** regressor on the surviving rows of
   the 60%.
3. **Estimation classifier.** On the full 60%, label `1` if the final regressor under-predicted, `0`
   if it over-predicted. Fit an `XGBClassifier` on `features + [the regressor's own prediction]`.
   Store `median_underestimation` = median(actual − pred) over rows classified as under, and
   `median_overestimation` = median(pred − actual) over rows classified as over.

Inference [S, `Run_Inference.ipynb`]:
```
pred = regressor_lower.predict(X)
p    = estimate_classifier.predict_proba(X + [pred])[:,1]
pred = where(p > 0.5, pred + median_underestimation, pred)
pred = where(p < 0.5, pred - median_overestimation,  pred)
pred = int(round(pred))
```
The classifier's AUC is **0.64** and the report concedes *"The usefulness of it is limited because of
the estimator's accuracy but it still helps a certain amount"* [S].

Features [S]: `unix_time`; from ETD, per gufi-timepoint, over the messages with
`timestamp < timepoint < ETD`: `minutes_until_departure_from_timepoint`,
`minutes_until_departure_from_timestamp`, `mean/median/std_departure_from_timepoint`, `found_etd_vals`
(how many ETD messages exist); one-hot airline codes; and a **"taxi time to gate"** airport-level block
over a 120-minute look-back that **expands to 180 minutes if fewer than 4 records are found** [S,
`extract_combined_vals_taxi_error_estpush_v2`]: mean/std of realised taxi time, mean/std of ETD error
(|actual runway departure − estimated|), mean/std of realised `minutes_until_pushback`, mean/std of
pushback error, and `found_counts`. **[X] The adaptive window — widen until you have at least 4
observations — is a neat answer to sparse night-time traffic.**

Leakage control: the group split by gufi, and ETD filtered to `timestamp < timepoint`. They **drop rows
with any NaN** (`df_data.dropna()`) [S], unlike everyone else.

**[X] For an RMSE competition the median-offset correction is the wrong estimator (medians minimise L1),
and outlier trimming is actively dangerous. The *structure* — a first-pass model used to define a hard
subset, then a specialist — is transferable; the specific estimators are not.**

### 7.2 5th — Cuong_Syr (XGBoost, thin features, a leaky split)

Source: `5th Place/Phase 1/src/` [S].

```
XGBRegressor(objective="reg:absoluteerror", eval_metric="mae",
             max_depth=3, min_child_weight=3, colsample_bytree=0.9, subsample=0.6,
             gamma=0, reg_alpha=0.75, reg_lambda=0.45,
             learning_rate=lr, n_estimators=30000, seed=42)
model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_test,y_test)])
```
one per airport; `--lr` defaults to **2.5** in the CLI [S, `run_training.py`] — **[X] an implausibly
large learning rate for a 30,000-tree budget; either it was always overridden on the command line or
early stopping fired almost immediately.** The report highlights that XGBoost ≥1.7 lets you minimise
MAE directly rather than an RMSE proxy [S].

**Validation is `label_pd["is_train"] = (np.random.rand() >= 0.3)` per row** [S] — a uniform random
row-level split, with no grouping by flight and no time ordering. With ~6 highly-correlated rows per
flight this puts the same flight on both sides. **[X] Their early stopping was therefore tuned against
a leaked signal; that they still placed 5th says the leak mostly cost them, not helped them.**

Features [S, `features.py`]: `diff_depart_vs_ts` (ETD − now, via `merge_asof` backward, 30 h tolerance);
LAMP joined backward on `forecast_timestamp`; `airline` from the GUFI; `block_time` (5 coarse buckets:
early morning / morning / noon / afternoon / night); `day_in_wk` as a name string; the four mfs
categoricals; `departure_runway_actual` via `merge_asof` by gufi (**[X] this is *this flight's own*
actual departure runway — it is null at genuine prediction time and is filled with `"NA"`; in training
it may be populated for rows very close to departure, which is a subtle leak**); and `num_flights` =
number of distinct gufis in the `runways` table in the **previous clock hour bucket** (joined on a
`"YYYY-MM-DD HH"` string, so the window is 1–2 h wide depending on where in the hour you sit) [S].
One-hot via `pd.get_dummies`. No post-processing.

Tried and rejected: TFM-derived statistics on `arrival_runway_estimated_time`, which *"help to reduce
the MAE on Open Area but not on Prescreened Area"* [S].

---

## 8. Cross-cutting summary

### 8.1 Model / scope / loss

| | 1st | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|
| library | CatBoost | CatBoost | XGBoost | LightGBM | XGBoost |
| scope | 3 per-airport + 1 **global** | per-airport | per-airport | per-airport (global measured at par) | per-airport |
| loss | MAE | MAE | default (squared) | `regression_l1` | `reg:absoluteerror` |
| target | raw **and** residual-vs-ETD | raw | raw | raw | raw |
| ensembling | mean of 4, per-airport override | main + out-of-range repair model | regressor + sign classifier | none | none |
| clipping | [1,299] members, [4,260] final | none | none | none | none |
| validation | forward time split (last 3 mo) | none shipped | GroupShuffleSplit by gufi | group split by gufi | **random row split (leaky)** |
| tuned HPs | yes, hard-coded | per-airport iterations only | **none** | Optuna on 2 params | yes |

### 8.2 The operational features with a direct analogue in our data

| operational quantity | who used it and how | our analogue |
|---|---|---|
| **number of aircraft on the ground** | 4th: `deps_taxiing` (pushed back, not yet airborne) and `arrs_taxiing` (landed, not yet on-block) — computed as *set differences between two event tables*, not as counts in a window [S] | derivable; see §9 item 5 |
| **departures in a recent window** | 1st: `runways_{15,30,60,90,120,360,720,1440}min_totdep`, nested cumulative [S]. 4th: `deps_3hr`, `deps_30hr` [S], Phase 2 `deps_last_{5..360}min` [S]. 2nd: written (`dep20_time`) then **dropped** [S] | we already have `dep_tko_prev_{5,10,15,30,60}m`; **the long tail (2 h – 30 h) is missing** |
| **arrival demand look-ahead** | 1st: TFM/TBFM 3-hour arrival-estimate block — **built, then disabled, reported as no gain** [S]. 5th: TFM statistics — helped in one arena, not the other, dropped [S] | we have `arr_land_next_{w}m` |
| **departure demand look-ahead** | 4th Phase 1: `exp_deps_{15,30}min` = flights whose ETD is within ±N min of **this flight's own ETD** — 2nd-most-important feature at KMEM [S]. 4th Phase 2: `estdep_next_{30,60,180,360}min` at airport level [S] | derivable from `SCHED_TIME_UTC_mvt` / `EOBT_1_flt`; the *±window around the flight's own time* form is **not** in our feature set |
| **arrival / turnaround linkage** | 4th: `standtime_30hr` = mean(first-position time − departure-stand time) of *arriving* flights, i.e. the origin-side turnaround duration, in the last 30 h — top-5 at KMEM [S]. 1st: `standtimes_{max,mean}delay_{1..24}h` = how late recently-pushed-back flights were vs their first ETD [S] | `arr_taxi_in_mean_{30,60}m` exists; the **30-hour** horizon and the aircraft-level turnaround link do not |
| **runway configuration** | 1st: 5-level per-runway code, active-runway counts, change flags, `n_changes_last_{30min..10h}` [S]. 4th Phase 1: raw config strings as categoricals [S]; Phase 2: full derived block incl. lagged configs 1..24 and `nchanges_last_{4..24}` [S]. 2nd, 5th: essentially none | `RUNWAY_mvt` per movement (53 levels) + `rwy_tko_prev_{15,60}m` + headways; **no config-change counter, no runway-heading feature** |
| **weather** | 1st: past 6 h min/mean/max, forecast-lead bins (≤3.5 h, 3.5–6.5 h), forecast-revision deltas (i=1..6), forecast-gradient deltas (i ∈ {1,3,6,9,12,24}) [S]. 4th: **nowcast only**; lead-binned forecasts tried and rejected [S]. 2nd: nowcast, and dropped `wind_speed` and `cloud_ceiling` as unimportant [S] | weather block exists and is the one external source that has worked for us |

### 8.3 Lookahead-window design, as a pattern

Three distinct constructions appear, and they are not interchangeable [S]:
1. **Nested cumulative look-back** — count events in `(t−W, t]` for W in an increasing set. Everyone's
   default. 1st place: {15,30,60,90,120,360,720,1440} min.
2. **Disjoint buckets** — the airport-config 2nd place (MIT) builds, over a 30-minute look-back, both
   nested windows `[0,30],[0,15],[0,10],[0,5]` **and** disjoint 5-minute slices
   `[5,10],[10,15],[15,20],[20,25],[25,30]` [S, `nasa-airport-config/2nd Place/inference_src/src/feature_processing_functions.py::process_true_rate`].
   **[X] Disjoint slices let a tree express "the rate is rising" without needing a difference of two
   cumulative counts; nested windows cannot do this in one split.**
3. **Forward demand by materialisation** — replicate each still-pending estimate forward on the 15-min
   grid so that at every timestamp the *currently-known* future set exists, then count how many fall
   within the next N minutes. Used by 4th place Phase 2 and by the airport-config 1st place with
   `estdep_next_{30,60,…,720}min` and arrival equivalents `next_{30..360}min`
   [S, `nasa-airport-config/1st Place/code/src/ExtractFeatures.py::ExtractETDFeatures/ExtractERAFeatures`].
   MIT's `process_projected_rate` does the same with 30-min forward buckets out to a 10-bucket horizon [S].

---

## 9. Run-way Functions (airport configuration) — what transfers

Competition: https://www.drivendata.org/competitions/89/competition-nasa-airport-configuration/ ;
winners blog https://drivendata.co/blog/airport-configuration-winners ; code
https://github.com/drivendataorg/nasa-airport-config [S]. Metric is a mean aggregated log loss over
airport × lookahead × configuration; scores 0.07385 / 0.08956 / 0.09122 / 0.10583 for places 1–4 [S].

**Alejandro Sáez won 1st place in both competitions** (Stuytown2 here, Team CDS in pushback) [S,
both winners blogs], and the pushback code is visibly the same pipeline lineage.

1st place [S, `nasa-airport-config/1st Place/code/`]:
- **120 CatBoost multiclass classifiers = 10 airports × 12 lookaheads** {30,60,…,360 min}; a grid
  search over `eta ∈ {0.005,0.01,0.02,0.03,0.05} × depth ∈ {3,4,5,6,7}` **per model**, keeping the best
  by validation multiclass loss [S, `main.py`]. Fixed:
  `n_estimators=6000, l2_leaf_reg=20, min_data_in_leaf=1000, grow_policy="Lossguide", max_leaves=11,
  has_time=True, boosting_type="Plain", early_stopping_rounds=50`.
- Validation: **the last 5% of each airport's rows in time order** (`start_test = int(n*0.95)`), with
  competition test days excluded from training entirely [S].
- Post-processing: fill missing configs with 0 and **renormalise each (airport, timestamp, lookahead)
  group's probabilities to sum to 1** [S, `GeneratePredictions.py`]; a `BuildDummyPrediction` uniform
  fallback exists [S].

**Airport-state features worth stealing** [S, `ExtractFeatures.py`]:
- `feat_1_active_departurerunways` / `feat_1_active_arrivalrunways` = number of runways in use;
  **`feat_1_max_directions` / `feat_1_min_directions` / `feat_1_unique_directions`** = max/min/count-of-
  distinct **runway heading numbers** parsed out of the config string (L/R stripped) — a cheap encoding
  of "which way is the airport flowing" and "is it split".
- Rolling means of all of the above over `{4,8,12,16,20,24}` 15-min periods (1 h … 6 h).
- **`feat_1_cat_previous_config_{1..24}`** — the categorical configuration at each of the last 24
  15-minute lags, and `feat_1_nchanges_last_{4..24}` = `nunique` of those lags = how unstable the
  configuration has been.
- **Cross-airport globals**: for every one of the 10 airports, the rolling active-departure/arrival
  counts and change counts are **mapped onto every row of every other airport** by timestamp
  (`feat_1_{airport}_dep_roll_{i}`, `_arr_roll_{i}`, `_nchanges_{i}`) — i.e. the national system state
  is a feature for each local model [S].
- `feat_{group}_count_{15min,30min,45min,60min,90min,120min,240min,360min,480min,720min,24h}` — an
  11-window nested count template applied uniformly to each of the four event tables (runway arrival,
  runway departure, stand arrival, stand departure) [S, `ExtractGufiTimestampFeatures`].

2nd place (MIT) [S, `nasa-airport-config/2nd Place/inference_src/src/feature_processing_functions.py`]:
the nested+disjoint bucket design in §8.3 item 2, the forward-projected demand matrix in item 3, and a
hand-added **"Minimum Configuration Support"** floor because the models under-estimated how persistent
a configuration is [S, winners blog].

---

## 10. What I could not find

- Any **public leaderboard table** for either arena of the pushback competition, beyond the five prize
  scores in the repo README. Not found.
- **Team CDS's own repo** `github.com/alsaco33/NASA-pushback` — 404 [S].
- The **v2 hyperparameters** of the 1st place solution ("same as v0 with some parameter changes"); only
  the trained artefacts were shipped and the repo's `models/` directory contains only `.gitkeep` [S].
- Any **ablation of the 1st place ensemble** (what each of the four members scores alone). Their
  README's `images/errors.JPG` presumably shows it but the numbers are not in text. Not found.
- Any winner who **measured RMSE**. Only FLHuskies mention it, and only as a secondary metric with no
  numbers [S].
- Any winner who used a **monotonic constraint, quantile/Huber loss, or sample weighting**. None present
  in any of the five solutions.

---

## Applicable to resilient-hedgehog

Each item states the source, the concrete change, and whether our data actually contains the inputs.
Data availability claims marked **[S, verified]** were checked directly against
`data/raw/ranking.parquet` and `docs/data_profile.md` in this repo.

**A. Model structure / ensembling**

1. **Add a raw-target model as an ensemble partner to the residual model, and average.**
   1st place averaged two residual-vs-proxy models, one raw-target model and one global model with equal
   weights [S, `1st Place/Phase 1/submission/utilities.py:660-671`]. We already train on
   `TAXITIME − proxy` (docs/features_log.md); we do **not** appear to train a raw-`TAXITIME_SEC_mvt`
   partner. **Inputs: yes** — same feature table, only the label changes. Checkable in one config.

2. **Add a deliberately feature-starved global model as a 4th ensemble member.**
   1st place's global member saw only the flight-identity and proxy-revision columns plus an airport
   categorical — no weather, no congestion [S, `create_master/nodes.py::build_global_master`].
   Our analogue: a model on `{ADEP, AIRCRAFT_TYPE, AIRCRAFT_OPERATOR, ADES, WK_TBL_CAT,
   mvt_minus_aobt3, aobt3_minus_eobt1, lobt_minus_eobt1, calendar}` only. **Inputs: yes.**

3. **Select the ensemble structure per airport on validation, including "use one member alone".**
   1st place overrode the ensemble with the raw-target model at KPHX, KMIA and KJFK [S]. Our airports
   differ enormously in target sd (LIRF 1332 s vs LSZH 385 s [S, data_profile §3]) and in proxy quality
   (`aobt3_within_60s` ranges 0.087 at LTFM to 0.264 at LEBL [S, data_profile §5]). **Inputs: yes.**
   **[X] LTFM and LIRF are the obvious candidates for a different member weighting.**

4. **A repair model keyed on the main model's own output, not on a data partition.**
   2nd place pooled every training row where the main model predicted `< 0`, across all airports, and
   trained one small shallow CatBoost (depth 4, lr 0.01) used *only* on those rows at inference [S,
   `2nd Place/Phase 1/src/run_training.py::train_negative_model`]. We already have a model-B/model-A
   split by *matched vs unmatched*; the 2nd-place trigger is different and complementary — e.g. rows
   where model A predicts outside `[p0.1, p99.9]` of the airport's training target. **Inputs: yes.**

**B. Congestion / demand features not yet in `src/prc/features.py`**

5. **"Currently on the ground" as a state, not a flow.**
   4th place's `deps_taxiing` / `arrs_taxiing` are set differences between event tables, and were part
   of the feature set whose top-5 at KMEM was dominated by ground-state features [S, `4th Place/Phase 1/src/table_scripts/add_traffic.py`].
   Our analogues at anchor time `t` (the known take-off time of the target flight):
   - *arrivals on the ground*: `count(ARR rows at this airport with MVT_TIME ≤ t < BLOCK_TIME)`.
     **Inputs: yes [S, verified]** — ranking ARR rows have `BLOCK_TIME_UTC_mvt` null-rate **0.0**.
   - *departures on the ground*: `count(DEP rows with AOBT_3_flt ≤ t < MVT_TIME_UTC_mvt)`.
     **Inputs: yes, at 98.5% coverage [S, verified]** — ranking DEP rows have `AOBT_3_flt` null-rate
     **0.0153**; `BLOCK_TIME_UTC_mvt` is null-rate **1.0**, so `AOBT_3` is the only available off-block
     clock and the count inherits its ~±374 s noise (docs/features_log.md).
   Our existing `context_features` has only *flow* counters (`dep_tko_prev_*`, `arr_land_prev_*`,
   `arr_inblock_prev_*`), never an occupancy count.

6. **Extend the look-back windows to 2 h – 30 h.**
   Our `WINDOWS_MIN = (5, 10, 15, 30, 60)` [S, `src/prc/features.py:32`] tops out at one hour. 1st place
   used {15,30,60,90,120,360,720,1440} min for movement counts and {1,2,4,6,12,24} h for stand-delay
   aggregates [S]; 4th place's two most important non-identity features used a **30-hour** window [S].
   Six of our ten rejected experiments were congestion features **all at ≤60 min** (docs/features_log.md),
   so the long horizon has never been tested. **Inputs: yes**, pure recomputation.
   **[X] The 30-hour horizon measures "what kind of day is this airport having", which is a different
   quantity from the 30-minute horizon's "is there a queue right now"; our six rejections only rule out
   the latter.**

7. **Rolling mean of realised *durations*, not counts — using the arrival side, which survives.**
   `arr_taxi_30hr` was a top-5 feature [S]. We already have `arr_taxi_in_mean_{30,60}m`
   [S, `src/prc/features.py`]; extend it to 3 h and 30 h. **Inputs: yes [S, verified]** — ranking ARR
   rows have `TAXITIME_SEC_mvt` null-rate **0.0**.
   The departure-side twin `dep_taxi_30hr` is **not** available: ranking DEP rows have
   `TAXITIME_SEC_mvt` null-rate **1.0** [S, verified]. The computable surrogate is
   `mean(MVT_TIME_UTC_mvt − AOBT_3_flt)` over recent departures — which our
   `nbr_mvt_minus_aobt_mean_{30,60}m` already is, at ≤60 min only. **Extend that one to 3 h / 30 h.**

8. **Departure demand in a window centred on the flight's own planned time (not on the clock).**
   4th place's `exp_deps_15min` = count of flights whose current ETD lies in `[T−15, T+15]` where `T` is
   *this* flight's ETD — the 2nd-most-important feature at KMEM [S]. Our repo tested `bank_position`
   (rank within the bank) and rejected it, but never the **symmetric density count**.
   Our `T` candidates: `SCHED_TIME_UTC_mvt` (null-rate 0.0 in ranking [S, verified]) or
   `EOBT_1_flt`/`AOBT_3_flt` (null-rate 0.0153). **Inputs: yes.**

9. **Disjoint 5-minute slices in addition to nested windows.**
   MIT (airport-config 2nd) built `rate_[0,5], [5,10], [10,15], [15,20], [20,25], [25,30]` alongside
   `[0,30],[0,15],[0,10]` [S]. Our windows are purely nested. **Inputs: yes** — a one-line change to
   `_counts_in_windows`. **[X] Cheapest untested idea in this slot.**

**C. Runway / configuration features**

10. **Configuration-change counters and heading features.**
    1st place: `config_n_chang_dep_last_{6,12,24,36,48,120}` 5-min periods [S]; airport-config 1st:
    `nchanges_last_{4..24}` plus `max/min/unique` of the **runway heading numbers** in use [S].
    Ours has `active_runways_60m` (rejected) but **no change counter and no heading encoding**.
    Our `RUNWAY_mvt` has 53 levels with 100% ranking coverage [S, data_profile §7] and the heading is
    the numeric prefix of the designator. Derive: number of *distinct* runways used for departures in
    the last N min, number of distinct headings, number of times the departure-runway set changed in
    the last {1,2,4,6} h. **Inputs: yes.**
    **[X] Caveat: we observe realised runway *usage*, not a published configuration, so "change" is
    noisier than in the NASA data — a single off-nominal departure flips the set.**

**D. Weather**

11. **Forecast-lead-binned and forecast-gradient weather features.**
    1st place built (i) 6-hour rolling min/mean/max of observations, (ii) means over forecast-lead bins
    ≤3.5 h and 3.5–6.5 h, (iii) forecast-revision deltas, (iv) forecast-gradient deltas at leads
    {1,3,6,9,12,24} h [S]. Weather is the *only* external source that has worked for us
    (docs/features_log.md). **Inputs: depends on `src/prc/weather.py`'s source — needs checking.** If
    the source is observations only (e.g. METAR), (i) transfers and (ii)–(iv) do not.
    **[X] Counter-evidence: 4th place explicitly rejected multi-hour forecasts in favour of the nowcast,
    and 2nd place dropped `wind_speed` and `cloud_ceiling` as unimportant [S]. Expect a small effect.**

**E. Turnaround**

12. **Same-aircraft turnaround link.**
    1st place's `standtimes` delay block and 4th place's `standtime_30hr` both proxy turnaround pressure
    [S]. The direct form for us is the preceding arrival of the same aircraft/flight.
    **Inputs: partial — only 14.77% of ranking departures have a matching arrival row inside the ranking
    file [S, `docs/data_profile.md`, final line of §7: 域内便の割合 0.1477].** The airport-level surrogate (mean arrival taxi-in / count of arrivals
    on the ground, item 5/7) covers the remaining 85% and should be preferred.

**F. Validation and training discipline**

13. **A single forward time split is what the winner used, and fold-averaged time CV was explicitly
    reported as no help** [S, 1st place report Q11]. Our forward-fold + month-bootstrap design is
    consistent with this; no change implied. Worth recording as external confirmation.

14. **`has_time=True` if CatBoost is ever used.** 1st place set it in every model [S]; it stops CatBoost
    from permuting rows when building ordered target statistics. **Inputs: n/a, a model flag.** Our
    per-fold target encodings already handle this for LightGBM.

15. **Group the split by flight, not just by time, if rows are ever expanded per flight.** 3rd and 4th
    used `GroupShuffleSplit`/GUFI-disjoint splits; 5th used a random row split and it was the only
    clearly leaky scheme among the winners [S]. **Inputs: n/a** — our rows are one-per-flight, so this
    does not currently bite. It becomes relevant the moment any multi-row-per-flight scheme is tried.

**G. Post-processing**

16. **Clip the final prediction to a validated per-airport range.**
    1st place clipped members to `[1,299]` min and the ensemble to `[4,260]` min [S]. For RMSE this is
    strictly more valuable than for MAE. Our target has p01 = 291 s and p99 = 2339 s with a max of
    131,167 s [S, data_profile §3], and our unmatched group runs at RMSE ~2130 (docs/features_log.md).
    Clip predictions to, e.g., per-airport `[p0.5, p99.5]` of the training target. **Inputs: yes.**
    **[X] This cannot help on the rows our evaluation already excludes, but it protects the leaderboard
    against a single catastrophic prediction, which RMSE punishes quadratically.**

17. **A fallback chain in the submission path.** 1st place: model → ETD-minus-30 baseline → constant 30
    [S, `1st Place/Phase 1/submission/solution.py`]. Our analogue: model → `MVT_TIME − AOBT_3` (the
    proxy) → per-airport median taxi-out. **Inputs: yes** (`AOBT_3` at 98.5% in ranking [S, verified]).

**H. Explicit non-transfers — do not copy these**

18. **Integer rounding of predictions** (all five winners) is an MAE artefact; under RMSE, rounding to
    seconds adds variance for nothing. Skip.
19. **Median-offset correction by a sign classifier** (3rd place) optimises L1. Under RMSE the
    corresponding object is a conditional *mean* residual, which a second-stage regressor already
    estimates better. Skip the classifier; keep the two-stage idea only via item 4.
20. **Outlier trimming of the training set** (3rd place: drop rows with first-pass error > 20–30 min)
    is defensible under MAE and dangerous under RMSE, where the trimmed rows carry most of the loss.
    Our evaluation protocol already excludes the top 200 rows *for measurement*; do not also remove them
    from *training* without a controlled test.
21. **"Per-airport because everyone did per-airport"** is not supported. The only published measurement
    (4th place, KDD Tab. 4) has the global LightGBM at **10.7232** vs per-airport **10.7686** on the full
    feature set, and per-airport ahead only when the feature set was thin [S]. We have 10 airports, a
    strong airport categorical and airport-conditioned encodings; **[X] the expected gain from splitting
    is small and the variance cost at LIRF (160 k rows) and LSZH (135 k rows) is real.**
