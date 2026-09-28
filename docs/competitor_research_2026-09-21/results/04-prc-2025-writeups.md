# SLOT 04 — PRC Data Challenge 2025 writeups (fuel burn)

Agent: investigator. Date: 2026-09-21.
Legend: **[S]** = read directly from a source (URL given). **[X]** = my own extrapolation/inference.
All repos below were cloned and read at commit HEAD on 2026-09-21.

---

## 0. TL;DR for resilient-hedgehog

- The 2025 winner (`resourceful-quiver`, RMSE 200.83) was **one single LightGBM**, 48 features,
  no ensemble, no blending, **trained on a rate target (kg/s) instead of the raw quantity**, with
  the rate converted back by multiplying by segment duration. **[S]**
- The two biggest named gains in the winner's own ablation table: (a) **target = rate, not total**
  (≈ −18 RMSE), (b) **a crude physically-bounded latent-state feature** (heuristic mass) that beat
  the "smarter" recursive physics version (≈ −6 RMSE). **[S]**
- Almost every serious external-data idea **failed**: ERA5 wind (winner + rank 6), METAR
  (rank 11 + 2026 solo team), oil price / MSCI (rank 6), trajectory completion heuristics (winner).
  **[S]**
- **Per-phase / per-class specialist models were used by rank 7, 8 and 12 — none of them beat the
  single global model of rank 1.** **[S]**
- **`jubilant-vase` in 2026 is NOT the same team as `jubilant-vase` in 2025.** Codenames are
  re-randomised per edition. There is **no** methodological continuity to record. **[S]** — details §7.

---

## 1. Official sources

| what | URL | status |
|---|---|---|
| 2025 challenge site | https://ansperformance.eu/study/data-challenge/dc2025/ (301 → https://prc-data-challenge-2025.netlify.app/) | **[S]** |
| Ranking rules page | https://prc-data-challenge-2025.netlify.app/ranking.html | **[S]** — rules only, no table |
| **Winner Announcement / repo reviews** | **https://prc-data-challenge-2025.netlify.app/outcome.html** | **[S]** — the single richest source; full top-12 table + an organiser-written technical review of every qualifying repo |
| Ranking-evolution bump chart | https://prc-data-challenge-2025.netlify.app/media/bump_chart_annotated_final.png | **[S]** — Sep / Oct / Sep+Oct / Final RMSE for all 33 scored teams |
| Per-team submission histories (5 teams) | https://prc-data-challenge-2025.netlify.app/outcome_files/figure-html/unnamed-chunk-2-{1..5}.png | **[S]** |
| GitHub org (19 team repos) | https://github.com/PRC-Data-Challenge-2025 | **[S]** |
| JOAS special issue Vol.4 No.3 (2026) | https://journals.open.tudelft.nl/joas/issue/view/1058 | **[S]** |
| Editorial (dataset paper) | https://journals.open.tudelft.nl/joas/article/view/8750 | **[S]** |
| Leaderboard REST API | https://datacomp.opensky-network.org/api/competitions/71c49292-6139-425f-803a-52ee8730ba58/leaderboard?limit=200 | **[S]** — **now returns `{"items":[],"nextCursor":null}`**; the live 2025 leaderboard has been purged. The Observable notebook `@espinielli/dc25-leaderboard` renders empty for the same reason. Per-team submission *counts* therefore **cannot be recovered** except from the 5 published history figures. |

### Competition facts **[S]**
- Target: fuel burnt (kg) in irregular time segments of a flight; metric RMSE in kg.
  Source data = ACARS FOB telemetry (airframes.io) fused with re-decoded ADS-B (OpenSky) +
  EUROCONTROL flight list, Apr–Oct 2025. ~15k flights / 193,275 fuel intervals / 27 aircraft types.
  (editorial 8750)
- **179 teams registered, 53 teams submitted, 2,127 submissions total, 33 teams had a scored final
  submission, 19 teams appear in the official final ranking.** (outcome.html + bump chart)
- Two phases: unlimited phase-1 submissions scored on `fuel_rank_submission.parquet` (September
  intervals), then **exactly one** final submission on `fuel_final_submission.parquet` (October
  intervals). The organisers state the split exists "to avoid the possibility of *leaning* from the
  leaderboard results." (ranking.html)
- **Prize eligibility required a documented public GPLv3 repo. 14 of the 33 scored teams were
  excluded from the final ranking for having no repo** (marked `x = no repo` on the bump chart).
  Including **two teams that would otherwise have finished 3rd and ~9th** (Sep+Oct RMSE 214.94 and
  227.32). (outcome.html + bump chart)
- Prize pool 5000 EUR: 2500 / 1750 / 750. (outcome.html)

---

## 2. Final ranking — complete **[S]** (outcome.html; ranks 13–19 from the bump chart)

| # | team | final RMSE (kg) | country / affiliation | repo |
|---|---|---|---|---|
| 1 | resourceful-quiver | **200.83** | NL, TU Delft (5 PhD students) | PRC-Data-Challenge-2025/resourceful-quiver |
| 2 | jubilant-vase | **211.99** | FR, 3 data-science students | .../jubilant-vase |
| 3 | outspoken-tornado | **219.85** | BE, Euranova (8 people) | .../outspoken-tornado |
| 4 | merry-jacket | **220.67** | NL, 2 TU Delft BSc students | .../merry-jacket |
| 5 | zestful-mango | **221.25** | ES, 2 industry data scientists (undisclosed) | .../zestful-mango |
| 6 | honest-rose | **227.14** | DE, 2 independents (brothers) | .../honest-rose |
| 7 | unique-honey | **227.89** | DE, Univ. of Stuttgart (2) | .../unique-honey |
| 8 | tiny-zeppelin | **244.60** | DE, DLR Hamburg (7) | .../tiny-zeppelin |
| 9 | exuberant-emu | **254.10** | US, Embry-Riddle (13) | .../exuberant-emu |
| 10 | unique-quicksand | **256.27** | DE, industry (undisclosed, aircraft OEM) | .../unique-quicksand |
| 11 | bright-lobster | **303.02** | CY, KIOS / Univ. of Cyprus (6) | .../bright-lobster |
| 12 | wise-watermelon | **350.98** | — | .../wise-watermelon |
| 13 | sincere-glacier | 540.91 | — | .../sincere-glacier |
| 14 | unique-guitar | 701.90 | — | .../unique-guitar |
| 15 | likable-ant | 709.49 | — | .../likable-ant |
| 16 | confident-insect | 709.50 | — | .../confident-insect |
| 17 | polite-emu | 729.80 | — | .../polite-emu |
| 18 | nurturing-telephone | 738.16 | — | .../nurturing-telephone |
| 19 | confident-hedgehog | 5755.91 | — | .../confident-hedgehog |

**Verification of the two names carried in from prior research:**
- 1st = `resourceful-quiver`, **200.83**, LightGBM + domain knowledge — **CONFIRMED [S]**.
- 2nd = `jubilant-vase`, **211.99**, CatBoost 64.1% / LightGBM 19.9% / XGBoost 16.0%,
  GroupKFold by `flight_id` — **CONFIRMED [S]** (weights read verbatim from their README; the
  GroupKFold is in `train_blend.py` and `optimize_infinity.py`).
  Caveat: their README self-describes as "RMSE ~224 (Top 8)" — it was written mid-competition and
  never updated. **[S]**

### Time-generalisation signal (unique to 2025, directly relevant to 2026's temporal split)
The bump chart gives Sep (rank set) vs Oct (final set) vs combined. **[S]**
- Winner was stable: Sep 199.91 → Oct 201.42 → combined 200.83. Rank 1 in every column.
- **`bright-lobster` was 2nd on September (213.24) and finished 11th on the combined set (303.02)**
  — a collapse implying an October RMSE around 370 **[X, arithmetic from the two published numbers]**.
  Their own JOAS paper reports 204.6 validation / 214.92 "official hidden test" — consistent with
  the September number, i.e. **they tuned to the phase-1 leaderboard and it did not hold in October.**
- The editorial says this outright: *"Several teams that ranked well on one month's data dropped or
  rose on the other ... generalization across time periods was a real challenge."* **[S]**

---

## 3. Per-team technical extraction (top 11 + the excluded paper team)

### 1 — resourceful-quiver (200.83) — TU Delft
Sources: repo `PRC_2025_report.pdf` (near-complete JOAS-style paper), `notebooks/TrainModelAndSubmit.ipynb`,
`data/metadata/final_metadata.json`. Mirror: https://github.com/meldorr/PRC-2025 **[S]**

- **Model family:** **one single LightGBM regressor.** No ensemble, no blending, no seed-averaging.
  They explicitly evaluated Random Forest, CatBoost and XGBoost and **chose LightGBM alone** for
  histogram splits, speed and native NaN handling.
- **Final hyperparameters** (`final_metadata.json`): `n_estimators=7500, learning_rate=0.01,
  subsample=0.8, colsample_bytree=0.8, reg_lambda=0.5, reg_alpha=0.1, metric=rmse, random_state=46`.
  **No early stopping; final model fit on 100% of training rows.**
- **Target transformation — the single most quotable finding:** they train on **fuel flow `ff_kgs`
  (kg/s)**, then multiply the prediction by `seg_duration` to get kg.
  Their ablation, same features, same params:

  | target | RMSE (rank set) |
  |---|---|
  | fuel consumption (kg), full features + recursive mass | 220.56 |
  | fuel consumption (kg), ADSB+ACARS | 219.13 |
  | fuel consumption (kg), ADSB+ACARS+Vrate split | 218.47 |
  | **fuel flow (kg/s)**, same feature families | **199.9 – 205.9** |

  Quote: *"the distribution of fuel flow is considerably tighter and less skewed than the
  distribution of fuel consumption, which results in a more stable learning target."*
- **Target-space outlier filter:** drop training rows with `ff_kgs < 0.05` or `> 6.5` kg/s
  (physically implausible). Ablation: no filter 228.05; upper-only 205.00; both 205.85.
  → *"unrealistic low fuel-flow values introduce more noise than unrealistic high values."*
- **Feature groups (48 features total):**
  1. Basic: `seg_duration, seg_dist, flight_duration, full_flight_dist, aircraft_type (cat),
     phase (cat)`.
  2. Statistical embeddings of ADS-B signals inside the segment: `altitude_mean`,
     `groundspeed_mean/std`, `track_mean/std`, `vertical_rate_mean/std/min/max`.
     They deliberately **excluded** lat/lon/distance/timestamp embeddings as physically irrelevant.
  3. ACARS embeddings: `mach_mean/std, TAS_mean/std, CAS_mean/std`, with a cascade to impute
     missing airspeeds (CAS→TAS via OpenAP `cas2tas`, else Mach→TAS via `mach2tas`, else
     groundspeed), then recompute the other two for consistency.
  4. Mass block: `tow_est_kg, m_tow (MTOW), oew, mass_est_tf_mean/std, ff_kgs_est_mass_tf_mean/std`.
  5. **"Detailed vertical-rate split": the segment is cut into 10 equal-time sub-parts, and
     mean+std of vertical rate is computed per part → 20 extra features.** This is their way of
     telling the tabular model what *mix* of climb/cruise/descent the segment contains.
- **Mass estimation — the second big gain.** Two variants:
  - *Recursive mass*: LightGBM TOW model trained on **the PRC 2024 challenge ATOW dataset**
    (527,162 labelled trajectories), ~3000 kg RMSE, then mass integrated backwards from OpenAP
    fuel flow (5 iterations to converge). RMSE 203.09–206.5.
  - *Heuristic mass*: **no model at all** — linear interpolation in elapsed-time fraction between
    `0.8·MTOW` and `0.5·(MLW+OEW)`, clipped to [0,1]. RMSE **199.91**.
  - Their explanation: recursive mass propagates early fuel-flow errors along the whole flight and
    can go negative when ADS-B is missing near takeoff; the crude heuristic is **bounded, smooth
    and defined even with zero trajectory data**. *"even approximate mass information helps."*
- **External data:** PRC 2024 ATOW dataset; ERA5 via `fastmeteo`; OpenAP (`Prop`, `FuelFlow`);
  EUROCONTROL Aircraft Performance Database / SKYbrary for MTOW/OEW; a hand-patched airport
  elevation table (`data/apt_updated.parquet`).
- **What failed (their own words):**
  - **ERA5 wind made the model worse once mass features were present** (199.91 without wind vs
    208.71-config comparisons; "No Wind" beat "Full features" 199.91 vs 201.04). Reason given:
    *"the scale of meteorological data and trajectory data operate on two different scales ...
    interpolation on the sparse wind grid loses too much local information."* Also: ERA5 made every
    pipeline iteration slow.
  - **Trajectory completion/augmentation failed twice**: padding missing trajectory with
    origin/destination airport positions, and synthesising missing climb/descent with a fixed
    ROC/ROD both *"saw significantly higher RMSE on the rank set and were subsequently removed ...
    these steps added too much biased heuristics."*
  - Removing obvious GPS-spoofed/jammed sections: they concluded *"leaving them untouched yields
    better result."*
  - Removing **any single** mass sub-feature degraded RMSE → the mass block only works as a set.
- **Validation:** `KFold(n_splits=5, shuffle=False)` — i.e. **contiguous, non-shuffled folds**,
  explicitly *"using data that is not randomly splitted to avoid leakage from two identical flight
  id."* Fold RMSE is computed **after** converting back: `y*seg_duration` vs `pred*seg_duration`.
  They admit they steered mainly on the public rank-set score, not CV.
- **Post-processing:** none (prediction = `ff_pred * seg_duration`, no clipping).
- **Iterations:** the published submission-history figure shows **roughly 150+ phase-1 submissions
  between 14 Oct and 30 Nov**, first score ≈ 223, a long plateau at 210–245, several excursions
  above 300, and only in the last ~10 days a dense cluster at 200–210. **[S]**
- Feature selection was **backward**: *"Starting from all engineered feature groups, we successively
  removed or simplified features that did not provide a clear and robust improvement."*

### 2 — jubilant-vase (211.99) — 3 French/Italian data-science students
Sources: repo README, `train_blend.py`, `optimize_infinity.py`, `data_utils.py`, `mass_estimator.py`. **[S]**

- **Ensemble:** XGBoost + LightGBM + CatBoost, weights from `scipy.optimize.minimize` on OOF
  predictions with a sum-to-1 constraint and [0,1] bounds. Final weights
  **CatBoost 0.641 / LightGBM 0.199 / XGBoost 0.160**.
  Reported: XGB CV 246.8, LGB 245.4, Cat 244.9, **ensemble 241.0 CV → ~224 LB**.
  → the blend bought ≈ **4 RMSE over the best single model** on CV.
- **Validation: `GroupKFold(n_splits=5)` grouped by `flight_id`**, used both inside Optuna (50
  trials per library) and for the final OOF/blend. Test-time prediction = **average of the 5 fold
  models**, then weighted blend.
- **Target: raw `fuel_kg`, no transform.**
- **Outlier filter (train only):** drop rows with `fuel_kg/duration > 20 kg/s` **and** `fuel_kg > 10`.
- **Features:**
  - temporal: `duration_sec`, `time_since_takeoff` (cumsum of durations minus own), `is_missing_data`
    flag for "ghost" segments with 0 trajectory points (duration back-filled from `end-start`).
  - spatial: `distance_km` (haversine sum), `distance_direct_km` (GC start→end),
    `dist_flown` (cumulative), `track_change_total` (sum of |heading changes|).
  - per-segment mean/min/max/std of altitude, groundspeed, vertical rate, TAS, Mach.
  - **Lag/lead context features**: `prev_alt_mean, prev_gs_mean, prev_vrate_mean, prev_phase`
    (shift +1 within flight) and `next_phase` (shift −1). Numeric lags NaN-filled with 0,
    categorical with `'NA'`.
  - physics: `fuel_flow_acropole`, `fuel_flow_openap`, `fuel_estimated_kg`, plus `mass_source` and
    `fuel_source` as *categorical provenance features*.
- **Mass estimation by inverse physics:** for each *training* flight, `scipy.optimize.minimize_scalar`
  finds the `mass0` that best matches OpenAP fuel flow to the known fuel labels; then a per-aircraft-type
  regression `mass ≈ f(flight duration)` is fitted and applied to the test set; fallback 0.85·MTOW.
- **Categoricals:** `aircraft_type, fuel_source, mass_source, origin_icao, destination_icao, phase,
  typecode, prev_phase, next_phase` — all `fillna('UNKNOWN')` → pandas `category`; CatBoost gets them
  as native `cat_features`.
- **External data:** OpenAP, **Acropole** (DGAC QAR-derived fuel-flow model). ERA5 via `fastmeteo`
  was used **only to validate** their ACARS-based TAS reconstruction, **not in the model** —
  they state global ERA5 was infeasible (terabytes) and that sparse ACARS points interpolated along
  the trajectory are a lighter, accurate substitute.
- **Post-processing:** none beyond `fillna(0)` on unmatched template rows.
- **Iterations:** the published history shows **only ~8 phase-1 submissions**, all between 5 Nov and
  30 Nov: 433 → 301 → 301 → 720 → 651 → 724 → 225 → 230. **[S]** A second place reached with
  roughly 8 scored attempts, versus ~150 for the winner. **[X]** The implication is that the
  method, not the submission count, carried them.
- Organiser review: *"adopts a mass estimation strategy consistent with the previous competition and
  combines it with an ensemble of three gradient boosting libraries ... no accompanying paper and the
  README appears auto-generated(?)"*. **[S]**

### 3 — outspoken-tornado (219.85) — Euranova (BE), also competed in 2024
Sources: repo README (best config is written out verbatim), `prc_challenge/` package. **[S]**

- **Model: CatBoost only**, hyperparameters by Optuna. No ensemble in the winning config
  (an `AverageEnsemble_v0_0_0` exists but is not used).
- **Target: raw `fuel_kg`.** They *built* a `TrainOnFuelConsumptionPerSec_v0_0_0` wrapper
  (target = kg/s, predictions ×duration) but **it is not in the best configuration** — i.e. the
  rate trick was available to them and they did not ship it. **[S]** This is a 19-RMSE gap versus
  rank 1 that is at least partly attributable. **[X]**
- **Features (their stated best config, deliberately small):** aircraft type; aircraft features from
  OpenAP (MTOW + aerodynamic); fuel segment duration; time since takeoff; calendar features
  (day of week etc.); and trajectory aggregates
  `mean_altitude, range_altitude, delta_altitude, mean_TAS, max_TAS, mean_groundspeed,
  mean_vertical_rate, n_points`. `flight_id` and `idx` explicitly dropped.
- **Post-processing — an explicit physical clip:**
  `pred ← max(0, min(pred, 1.1 × max_fuel_kg_per_sec[aircraft_type] × segment_duration))`,
  where the per-type maximum rate is taken from the training set. (`post_cleaning/MaximumFuelSec.py`)
- **Also tried:** `TabDPT` (a tabular foundation model), LinearRegression, train-set mean and median
  baselines — all present in the model registry, none shipped.
- **Validation:** `scripts/split.py` 80/20 flight-level split with a fixed seed.
- Organiser review: *"predominantly machine-learning-driven ... limited evidence of domain-informed
  validation ... the codebase is highly structured, possibly to the point of being over-structured."*
- **Iterations:** ~25 phase-1 submissions, first on 7 Nov at 277, best ≈ 225. **[S]**

### 4 — merry-jacket (220.67) — 2 TU Delft BSc students
Sources: README, `source/ensemble_final.py`. **[S]**

- **The most instructive ensemble in the field: two models on two *different* targets.**
  - CatBoost trained on **log1p(fuel flow rate)**; at inference `expm1(pred) × duration.clip(lower=1)`.
  - XGBoost trained on **raw fuel_kg** (one-hot encoded categoricals, since XGB lacked native support).
  - Blend **0.70 CatBoost / 0.30 XGBoost**, weights **hand-tuned, not optimised**.
  Stated rationale: *"This dual-model approach, utilizing different target variables and categorical
  handling methods, provided diverse predictive inputs for the final ensemble, improving robustness."*
- **Validation: Group K-Fold**, with an explicit failure report:
  *"This grouping ensures that models are validated on entirely unseen flights, not just unseen
  segments of the same trained flights — **a pitfall which we found ourselves in when training and
  ranking performance showed high discrepancy.**"* **[S]** — the clearest statement in the whole
  field that row-random CV over segments of the same flight produces a CV/LB divergence.
- **Missing-data handling:** intervals with no trajectory get `is_missing_data=1` and **all kinematic
  values set to 0.0** — deliberately *not* imputed, *"to avoid imputation bias and allow the GBRT
  models to treat the zero state as a distinct, low-activity operational mode."*
- **Features (~35, all listed in the README):** aircraft geometry from the **FAA Aircraft
  Characteristics Database** (wingspan, length, tail height, wheelbase, gear config, approach speed,
  MTOW, MALW, wake category); route geometry (`route_distance_nm`, `elev_diff_ft`); **cyclical
  `hour_sin`/`hour_cos`** of takeoff hour; kinematic aggregates; and hand-built interaction proxies:
  `mass_ratio_proxy = MTOW·(1 − time_into_flight_ratio)`,
  `energy_rate_index = mass_proxy · vertical_rate`,
  `mass_speed_index = mass_proxy · TAS`,
  `aspect_ratio_proxy`, `tail_length_ratio`, `weight_delta_ratio`, `distance_nm_approx`.
- **Post-processing:** `clip(lower=1.0)` on final kg, NaN → column mean. Hyperparameters via Optuna.
- **No TOW estimation step** — the organiser review names this as the main gap.

### 5 — zestful-mango (221.25) — 2 Spanish industry data scientists
Sources: README, `notebooks/base_model_prototyping_gradient_boosting.ipynb`. **[S]**

- **Per-segment specialist models, split by *data quality*, not by phase:**
  two LightGBM models — `cf_` (complete flight: `trajectory_completeness ≥ 0.3`) and
  `uf_` (uncomplete: `< 0.3`) — each with its own feature list. The `uf_` list drops the
  `tas_available` categorical. **This is the only quality-conditioned specialist split in the field.**
- **Target: `fuel_ratio` = fuel_kg / segment_seconds** (a rate), reconverted by
  `fuel_from_ratio(ratio, time_h) = ratio · (time_h·3600)`. A `LABEL_LOG_TRANSFORM` switch exists
  and is set to **False**.
- **Validation: strictly temporal.** Rows are sorted by `start`, cut into chunks of 6000, and
  assigned `base_train` = first 9 chunks, `eval` = middle chunks, `test` = **last 6 chunks**.
  No random shuffling anywhere. They also have a greedy `partitioned_data_split` that packs whole
  groups into splits by a partition column.
- **Outliers:** IQR bounds computed on `fuel_ratio` (not on fuel_kg) → `is_outlier` flag.
- **Sample weighting was tried and abandoned:** three successive `metadata["weights"] = ...`
  assignments in the notebook (phase-based 5× for non-cruise, inverse-frequency by aircraft type
  capped at 10, completeness-based 5×) are all overwritten by `metadata["weights"] = 1`. **[S]**
- **Features (~40):** flight_time, flight_distance, origin/destination elevation, optimal cruise Mach,
  `aircraft_average_fuel_burn`, MTOW; **trajectory-quality features**
  (`trajectory_time, trajectory_ratio, n_points, trajectory_ceiling, trajectory_completeness,
  trajectory_max_gap_mins`); `time_since_takeoff_min, time_until_landing_min`; per-phase point counts
  and minutes (`n_points_climb/cruise/descent/level`, `time_mins_*`); **quartile triples**
  (q1/median/q3 rather than mean/std) of altitude, groundspeed, track, vertical rate;
  `avg_wind_component, avg_tas, avg_mach, avg_mass`.
- **A second approach — an LSTM over the cleaned trajectory — was started and explicitly abandoned
  for lack of time.** **[S]**
- **Iterations:** ~35 phase-1 submissions from ~22 Oct; first ≈ 265, best ≈ 220. **[S]**

### 6 — honest-rose (227.14) — 2 German independents
Sources: `PRC_2025_documentation.pdf` (16-page writeup), `Python scripts/`. **[S]**

- **Model:** XGBoost. R prototype → Python `PRC2025_V9.py`. **Plus a 10-model seed ensemble**
  (same config, different seeds, slightly shrunk subsample/colsample, predictions averaged).
- **Target: `fuel_kg_min` (kg per minute)** — again a **rate**, multiplied by `interval_min` at
  inference.
- **Per-segment specialist model — the most 2026-relevant one in the field:**
  they derive `status` from `pct_elapsed_mid`:
  `< 0 → taxi_out`, `> 100 → taxi_in`, else `inflight`.
  **All taxi rows are dropped from the main model and scored by a separate specialised taxi model**
  (`submission_intervals_v4_scored.csv`), then concatenated back. **[S]**
- **Validation — multi-axis grouped CV:** Bayesian optimisation (`bayes_opt`) whose objective is
  a *combination of the average and the worst-case RMSE* across
  **5 folds grouped by `flight_id`** (main), 3 folds grouped by `aircraft_type`, and
  3 folds grouped by **duration buckets**. *"This encourages robustness across flights, aircraft
  types and distance regimes."* On top, an outer 80/20 **flight-level** train/test split.
- **Feature versions V0–V6, each ablated:**
  - V0 route/timing: `gc_distance_km, elev_delta_ft, flight_duration_min, interval_min,
    t_since_takeoff_min, t_to_landing_min, pct_elapsed_mid, start_hour_utc, end_hour_utc,
    avg_speed_km_per_min`.
  - V1/V2 trajectory aggregates + heading; explicit interpolation/extrapolation rules for intervals
    that fall outside the recorded trajectory (three synthetic rows at start/mid/end from the nearest
    anchors, `source="interpolation"`).
  - V3 geographic regions — "interpretable but not critical".
  - **V4 weather (Open-Meteo Historical Forecast API at interval midpoints) — FAILED.**
    *"most weather variables ... received very low importance scores and did not consistently improve
    RMSE."*
  - V5 aircraft grouping (narrow/wide-body, manufacturer family) + discrete phase label from
    `vs_mean_fpm` thresholds — **kept**, on the grounds that a discrete label is robust when the
    continuous value is noisy, and that it makes per-phase error analysis possible.
  - **V6 oil price + MSCI World — FAILED.** *"both ... ended up with very low feature importance ...
    did not improve the validation or test RMSE in a reliable way."*
  - Verdict: *"V4 (weather) and V6 (macro variables) did not add measurable value and increased
    complexity and external dependencies."*
- **Encoding:** custom sparse one-hot (numeric dense block hstacked with a sparse `get_dummies`
  block into CSR), plus a column-space aligner that zero-fills train-only columns at submission time.
  High-cardinality categoricals (`origin_icao, dest_icao, weather_code_text, regions`) dropped.
- No TOW estimation — the organiser review names this as their main gap.

### 7 — unique-honey (227.89) — Univ. of Stuttgart
Sources: README, `Multi-models for fuel consumption-227(1).ipynb`. **[S]**

- **Per-phase specialist stacks.** Segments are labelled climb / cruise / descent by a custom
  `flight_phase(alt_delta, segment_cl, segment_cr, segment_de, ...)` function, then
  **three completely separate stacking ensembles** are built, one per phase.
- **Each stack:** base models = {LGBMRegressor (phase-specific params), GradientBoostingRegressor,
  XGBRegressor or RandomForestRegressor} → level-1 OOF matrix → **meta-model = LinearRegression**
  (Ridge was tried and commented out). Test-time base predictions are averaged across the 5 folds
  before the meta-model is applied.
- **Validation:** outer split is by **`flight_id`** (70/15/15 train/val/test), but the **inner
  stacking CV is `KFold(shuffle=True)` on rows, not grouped** → residual within-flight leakage in
  the meta-model fitting. **[X]** For the submission they refit on train+test combined.
- **Target: raw `fuel_kg`.** `np.log1p(fuel_kg)` is present but commented out.
- **Feature engineering of note — group-relative deviations**, all computable at test time:
  `altitude_mean_by_typecode = groupby(typecode).altitude_mean.mean()`,
  `altitude_diff_type = altitude_mean − type mean`,
  `speed_diff_phase = groundspeed_mean − phase mean`,
  `altitude_diff_flight = altitude_mean − flight mean`.
  A `fuel_diff_flight` feature was written and then **deleted with the comment
  "Problem on the submission file, we do not have target values"** — i.e. a target-leakage feature
  caught by hand. **[S]**
- **External data:** OpenAircraftType (wake turbulence category, engine type, engine count),
  ourairports.com (airport size class), ERA5.
- **Phase-conditioned outlier reasoning:** *"Climb phase: fuel varies widely → outliers may be real
  behavior — Confirmed"* — they refuse to trim climb outliers.
- Organiser review: *"the rationale behind feature engineering choices is only briefly discussed, and
  the use of flight phases largely mirrors approaches seen in other submissions."*

### 8 — tiny-zeppelin (244.60) — DLR Hamburg (7 researchers)
Sources: READMEs, `Gradient Boosting/*.py`, `orchestrator/use_models.py`. **[S]**

- **The most aggressive specialist decomposition in the field — and it finished 8th.**
  Routing table (`use_models.py`):
  `CR_heavy, CR_light, CL_heavy, CL_light, DE_heavy, DE_light`, plus
  **`UNKNOWN → all_phase_model.pkl`** as fallback for unclassifiable segments.
  That is **3 phases × 2 weight classes + 1 global fallback = 7 models.**
  The repo also contains `*_neo` and `*_old` variants (engine-generation split) that are in the
  pickle set but commented out of the routing table.
- **Weight class rule:** `weight_class = 'heavy' if MZFW > 100 (tonnes) else 'light'`.
- **Model family: sklearn `GradientBoostingRegressor`** (not LightGBM/XGBoost/CatBoost):
  `n_estimators=1000, max_depth=5, min_samples_split=10, learning_rate=0.005,
  loss='squared_error', n_iter_no_change=10, tol=1e-2`. One-hot encoders fitted **per class**.
- **Validation:** 80/20 split on **unique `flight_id`s** (`np.random.choice` on flight ids, then
  membership filter) — correct grouping, but no seed and `np.random.choice` **with replacement**,
  so the effective training fraction is ≈63%, not 80%. **[X]**
- **Target: raw fuel per segment.** No rate normalisation.
- Also contains a `duration_model.pkl` (predicting flight duration) and an `xgb_fuel_pipeline.pkl`.
- Organiser review: *"limited guidance on how to run or reproduce ... hardcoded Windows-specific
  paths ... a largely standard use of flight phases."*
- **Takeaway: 7 specialist models, 19 features, scored 244.60 — 44 RMSE worse than one global
  LightGBM with 48 features.** **[S] for the numbers, [X] for the causal reading.**

### 9 — exuberant-emu (254.10) — Embry-Riddle (13 people)
Source: `01-final_model.py` (the whole model is one file). **[S]**

- **Single LightGBM**, sklearn `Pipeline` with `OneHotEncoder(handle_unknown='ignore')` on
  `aircraft_type`. `extra_trees=True, num_leaves=768, max_bin=512, learning_rate=0.015,
  n_estimators=5000, subsample/colsample 0.7, reg_alpha 0.4, reg_lambda 2.0, min_child_samples 50`.
- **Only 10 features:** `segment_duration_min, mean_groundspeed (= distance/duration),
  distance_km, mean_altitude (= (max+min)/2), altitude_variation, mean_vertical_rate,
  std_vertical_rate, mean_mach (hardcoded to 0.0 — a dead feature), elevation_diff, aircraft_code`
  (LabelEncoded duplicate of the one-hot `aircraft_type`).
- **Target: raw `fuel_kg`.**
- **Validation: `train_test_split(test_size=0.05, random_state=42)` — row-random, ungrouped.**
  Segments of the same flight straddle the split. **[S]**
- **No external data, no mass, no post-processing.**
- **Reference point: ten plain features, one LightGBM, a leaky 5% holdout → 254.10, rank 9 of 19.**

### 10 — unique-quicksand (256.27) — German aircraft-OEM-linked team
Source: README + `Step 2 - Baseline Model cleaned.ipynb`. **[S]**

- **Their shipped final submission is a non-ML baseline: the median `fuel_rate` (kg/min) per
  `aircraft_type`, multiplied by `interval_min`.** They built XGBoost models that were better on
  their own data but *"As the leading teams ranking submissions are so much better than our best
  XGBoost model, we refrain from publishing ours. Instead, we publish and apply our baseline model
  to the final data ... we do not provide our 'best', but our most robust estimate."*
- Their **v0** baseline scored **RMSE 266.59** on the phase-1 rank set; the same baseline scored
  **256.27** on the final set → **rank 10 of 19.** **[S]**
- A hand-crafted rule overlay **backfired and was measured**: suspecting a lb/kg unit bug for A20N,
  they doubled the A20N rate in v1 → **RMSE went 266.59 → 281.51.** They then correctly concluded
  the ranking data carries the same outliers as the training data. **[S]**
- Their XGBoost feature list (not shipped): aircraft_type + `Num_Engines, MTOW, MLW, ac_L, ac_B2`
  (span²); interval duration and absolute/relative position in the flight; whole-flight duration,
  distance, bearing; and mid-interval altitude/TAS/GS/ROC with CAS→TAS conversion.
  They also tried `RANSACRegressor` to fight trajectory outliers.
- External data: **only** the FAA Aircraft Characteristics Database. They read the ICAO Engine
  Emissions Databank (takeoff/climb-out/approach/idle fuel flow per engine type) and a BADA A320
  performance file for calibration but did not use Acropole.
- **The most important number in this whole document for calibration purposes: a per-aircraft-type
  median-rate lookup is only ~27% worse than the winner and beats 9 of 19 published solutions.**

### 11 — bright-lobster (303.02 final / 213.24 September) — KIOS, Univ. of Cyprus
Sources: repo README, JOAS paper https://journals.open.tudelft.nl/joas/article/view/8764
("A Physics-Guided Gradient Boosting Framework for Open-Source Aviation Fuel Estimation",
Grigoriou, Eftychiou, Vitale, Pettemeridis, Souli, Kolios). Team identity confirmed by matching the
author list to the `bright-lobster` team page. **[S]**

- **Model: XGBoost**, top-10 configs from `RandomizedSearchCV` (200 iterations, 5-fold CV over
  `max_depth 7–9`, `learning_rate 0.06–0.08`, `n_estimators 850–950`), then retrained on 100% of the
  augmented data. Reported **204.6 validation / 214.92 on the official hidden set** in the paper.
- **Feature selection: `SequentialFeatureSelector` (forward, 5-fold CV, XGBoost base) reducing
  100+/133 candidates to a compact subset** (`selected_features_sfs.json` is in the repo).
- **Data augmentation — unique in the field:** *"Generates 25K synthetic widebody samples
  (A332/A333/A359/etc.) by perturbing real widebody training data with 5–15% Gaussian noise,
  emphasizing long segments (top 25% duration) to balance underrepresented widebody classes and
  reduce RMSE."* **[S]**
- **External data (the widest in the field):** METAR (downloaded per airport),
  **IATA Market Report passenger load factors Apr–Aug 2025 mapped to airport pairs**,
  SkyVector airport pages scraped for missing airport attributes, OpenAP `FuelFlow`
  (`use_synonym=True`) with **true dynamic mass tracking from takeoff** across 36 aircraft types,
  and — disclosed openly — **fuel-flow-per-phase numbers for the remaining types filled in by
  prompting Google Gemini Pro 2.5** (for A306, MD11, B77L imputation + WTC). **[S]**
- Trajectory filtering *"credited to last year's winning submission"* (traffic-library filter stack).
- **Validation: 80/20 random-ish train/validation. No grouping by flight is documented.** **[X]**
- **The cautionary tale:** best documented repo, richest external data, **2nd place on September
  data, 11th overall.** Their own tuning target (a validation split + the phase-1 leaderboard)
  did not transfer to October. **[S] for the ranks, [X] for the attribution to overfitting.**
- Organiser review: *"stands out for its strong emphasis on reproducibility, excellent documentation
  ... deep domain awareness through TOW estimation, use of METAR data, passenger statistics,
  SkyVector airport information ... and data augmentation via Gaussian noise for widebody aircraft."*
- **Iterations:** ~150 phase-1 submissions, all compressed into 4 Nov – 30 Nov. **[S]**

### 12 — wise-watermelon (350.98)
Organiser review: *"consists solely of two Python notebooks with no accompanying documentation or
in-code comments."* **[S]** Not analysed further.

### Excluded-but-published: kindly-garden (Sep+Oct 227.32, **no repo → not ranked**)
JOAS paper https://journals.open.tudelft.nl/joas/article/view/8766 —
Rade Kačar (Univ. of Belgrade) & Darko Ćulibrk (mtel, Banja Luka).
Team identity confirmed: the `kindly-garden` team page lists exactly these two names; their
paper's reported **227.32 kg** is exactly the Sep+Oct value carried by one of the `x = no repo`
rows on the bump chart. **[S]** — they would have placed ~9th had they published a repo.

- **73-dimensional feature space**, XGBoost + LightGBM, Optuna-tuned, K-fold CV.
  XGB CV 232.76, LGB CV 243.91, **submission 227.32**, 10-fold OOF RMSE 203.3.
- Their stated thesis: *"the systematic substitution of latent physical variables — mass, airspeed,
  phase-resolved fuel flow — in place of raw kinematic observables"*: ERA5 wind-corrected TAS,
  OpenAP drag/thrust coefficients, **total-energy-model mass estimates for climb and descent**,
  and phase-specific fuel-flow surrogates via the `traffic` phase detector.
- Data: ADS-B + ACARS + ERA5 pressure levels + METAR + FAA aircraft performance records.
- Also 2024 participants. **This team used ERA5 successfully where rank 1 and rank 6 did not** —
  but at 227 vs 200, so the successful use did not out-score dropping it. **[X]**

---

## 4. Cross-cutting answers to the four questions

### (a) Largest single gains, by the authors' own account **[S]**
| team | claimed biggest gain | measured |
|---|---|---|
| resourceful-quiver | **rate target (kg/s) instead of kg** | ~218.5 → ~205.9, i.e. **≈ −13 to −18 RMSE** |
| resourceful-quiver | **heuristic (crude, bounded) mass** over recursive (physics) mass | 205.85 → **199.91**, ≈ −6 |
| resourceful-quiver | fuel-flow range filter 0.05–6.5 kg/s | 228.05 → 205.00, **≈ −23** |
| resourceful-quiver | 10-part vertical-rate split of each segment | "stable improvements", included in final 48 |
| jubilant-vase | 3-library weighted blend on OOF | best single 244.9 → **241.0 CV**, ≈ −4 |
| merry-jacket | Group K-Fold fixing a CV/LB divergence | qualitative, no number |
| honest-rose | V0+V2+V5 core (route geometry, kinematics, phase label, aircraft grouping) | qualitative |
| bright-lobster | physics-guided OpenAP features + SFS + widebody augmentation | 204.6 val / 214.92 Sep |
| kindly-garden | replacing raw kinematics with latent physical variables | 227.32 |

**Pattern:** the two largest measured gains are (1) **re-expressing the target as a rate over the
segment's own duration** and (2) **filtering the target in rate space**. Both are cheap. Neither is
a modelling trick. **[X]**

### (b) Reported failures **[S]**
- **Weather / ERA5 / METAR:** rank 1 removed wind and *gained* 1.1 RMSE; rank 6 measured V4 weather
  as near-zero importance and dropped it; rank 2 used ERA5 only as a validation plot and stated
  global ERA5 was infeasible; the 2026 solo entrant (see §7) implemented and then rejected METAR.
  Only kindly-garden (unranked) reports using ERA5 productively.
- **Macroeconomic covariates (oil price, MSCI World):** rank 6, near-zero importance, no RMSE gain.
- **Trajectory completion / synthetic padding:** rank 1, *"significantly higher RMSE on the rank set."*
- **Hand rules / unit-correction overlays:** rank 10 doubled A20N rates on a unit-bug hypothesis and
  lost **15 RMSE** (266.59 → 281.51).
- **Row-random CV across segments of the same flight:** rank 4 names it explicitly as a pitfall they
  fell into ("training and ranking performance showed high discrepancy"). Rank 9 still shipped with
  a 5% random holdout.
- **Deep learning:** rank 5 started an LSTM over cleaned trajectories and abandoned it;
  rank 3 built a `TabDPT` (tabular foundation model) adapter and did not ship it. **No neural model
  appears in any of the 19 ranked solutions.** **[X]**
- **Sample weighting:** rank 5 wrote three weighting schemes (per-phase, inverse-aircraft-frequency,
  per-completeness) and ended with `weights = 1`.
- **Over-tuning on the phase-1 leaderboard:** the structural failure — 14 of 33 scored teams lost
  their ranking entirely for not publishing a repo, and bright-lobster fell 2nd→11th between
  September and October.

### (c) Submissions / iterations **[S]** (only 5 teams' histories were published)
| team | phase-1 submissions (visual count) | window | first → best |
|---|---|---|---|
| resourceful-quiver (1st) | ~150+ | 14 Oct – 30 Nov | 223 → 200 |
| jubilant-vase (2nd) | **~8** | 5 Nov – 30 Nov | 433 → 225 |
| outspoken-tornado (3rd) | ~25 | 7 Nov – 30 Nov | 277 → 225 |
| zestful-mango (5th) | ~35 | 22 Oct – 30 Nov | 265 → 220 |
| bright-lobster (11th) | ~150 | 4 Nov – 30 Nov | ~270 → **213** |
Field totals: **2,127 submissions across 53 submitting teams** (mean ≈ 40).
**The 2nd-place team reached 211.99 with roughly 8 scored attempts; the 11th-place team made ~150
and had the 2nd-best September score.** Submission count is not the lever. **[X]**

### (d) Per-segment / per-group specialist models **[S]**
| team | rank | RMSE | specialist scheme | net |
|---|---|---|---|---|
| resourceful-quiver | 1 | 200.83 | **none** — one global LightGBM; "phase" is a *categorical feature*, and the 10-part vertical-rate split encodes the phase *mix* inside a segment | best in field |
| jubilant-vase | 2 | 211.99 | none — `phase`, `prev_phase`, `next_phase` as categorical features | 2nd |
| honest-rose | 6 | 227.14 | **taxi vs inflight**: taxi_out/taxi_in rows removed from the main model and scored by a separate taxi model | 6th; not separately ablated |
| unique-honey | 7 | 227.89 | **3 per-phase stacking ensembles** (climb / cruise / descent) | 7th |
| tiny-zeppelin | 8 | 244.60 | **3 phases × 2 weight classes + 1 global fallback = 7 models** (plus unused neo/old engine-generation variants) | 8th |
| zestful-mango | 5 | 221.25 | **2 models split by trajectory completeness (≥ / < 0.3)** — a *data-quality* split, not a physical one | 5th, best of the specialist users |
| unique-quicksand | 10 | 256.27 | per-aircraft-type median rate (degenerate specialist) | 10th |

**Conclusion: no team that shipped physical-phase specialists beat any team that shipped a single
global model with phase as a feature.** The only specialist split that lands in the top 5 is
`zestful-mango`'s **data-quality** split. **[S] for the placements; [X] for the causal reading —
the teams differ in many other ways, so this is an association, not a controlled comparison.**
Note however that the 2026 winner-adjacent solo entry (§7) *did* measure a real per-airport
specialist gain under a pre-declared gate — so the lesson is "specialists must be gated, and the
grouping variable matters", not "specialists never work". **[X]**

### External data actually used by ranked teams **[S]**
| source | used by | verdict |
|---|---|---|
| **PRC 2024 ATOW challenge dataset** (527k labelled takeoff weights) | rank 1 (TOW model) | helped |
| OpenAP (Prop / FuelFlow / cas2tas / phase) | ranks 1, 2, 3, 7, 11, kindly-garden | ubiquitous |
| Acropole (DGAC QAR fuel-flow model) | rank 2 | used |
| FAA Aircraft Characteristics Database | ranks 4, 10 | used |
| OpenAircraftType (WTC, engine type/count) | rank 7 | used |
| ourairports.com | ranks 7, 11 | used |
| EUROCONTROL APD / SKYbrary (MTOW, OEW) | rank 1 | used |
| ERA5 (fastmeteo / cdsapi) | ranks 1 (dropped), 2 (validation only), 7, kindly-garden | mostly negative |
| METAR | rank 11, kindly-garden | no measured gain reported |
| IATA load factors, SkyVector scrape | rank 11 | no measured gain reported |
| Open-Meteo API, oil price, MSCI World | rank 6 | measured negative |

---

## 5. Things I could not find — stated explicitly

- **Per-team submission counts.** The leaderboard REST API now returns an empty item list and the
  Observable notebook that rendered it therefore shows nothing. Only 5 teams' phase-1 histories were
  published as static figures; my counts from those figures are visual estimates, not exact.
- **Any RMSE ablation from jubilant-vase, outspoken-tornado, unique-honey, tiny-zeppelin or
  exuberant-emu.** Only resourceful-quiver, honest-rose and (partly) bright-lobster and
  kindly-garden published feature-group ablations. There is no public "this feature bought X RMSE"
  table for the 2nd-place solution.
- **Whether honest-rose's separate taxi model helped.** The taxi model's code is not in the repo
  (only its scored output CSV is referenced) and no ablation is given.
- **A JOAS paper from the winner.** `resourceful-quiver`'s `PRC_2025_report.pdf` is a near-complete
  JOAS-format preprint but does **not** appear in the published Vol.4 No.3 issue; only the editorial,
  bright-lobster's paper and kindly-garden's paper are published there as of 2026-09-21.
  Their mirror repo is https://github.com/meldorr/PRC-2025.
- **The winner's k-fold CV numbers.** The report says *"For the simplicity of this report ... we
  omitted it from here"* — they steered on the rank-set score.
- **Identities/methods of ranks 12–19** beyond the organiser's one-paragraph reviews (12) and
  nothing at all (13–19).

---

## 6. Correction to prior repo research

Prior research recorded 2nd place as "CatBoost 64% / LightGBM 20% / XGBoost 16%, GroupKFold by
flight". **Confirmed exactly** (0.641 / 0.199 / 0.160, `GroupKFold(n_splits=5)` on `flight_id`).
The only refinement: the weights are **fitted by `scipy.optimize.minimize` on OOF predictions with
a sum-to-1 equality constraint**, not chosen by hand, and the inference path **averages the 5 fold
models per library before blending**. **[S]**

---

## 7. `jubilant-vase` 2025 vs `jubilant-vase` 2026 — the continuity question

**They are different teams. There is no continuity.** **[S]**

| | 2025 `jubilant-vase` | 2026 `jubilant-vase` |
|---|---|---|
| members | 3: Agathe Vernin (Sapienza), Louisa Djebroun (ENSAI), Ivo Douliery | **1: Sergiu Vincze, SevinHub** |
| country | France | Belgium |
| registered | 8 Oct 2025 | 1 Sep 2026 |
| source | https://prc-data-challenge-2025.netlify.app/teams/jubilant-vase.html | https://ansperformance.eu/study/data-challenge/dc2026/teams/jubilant-vase.html |

The PRC platform assigns adjective-noun codenames **per edition**; the collision is coincidence.
`github.com/sergiuv11/prc-data-challenge-2026` is authored by Sergiu Vincze (single commit,
2026-09-11) and contains **no reference to 2025, to fuel burn, or to the 2025 codebase**.
The 2025 `jubilant-vase` repo has no successor.

**Method comparison (for the record — slot 09 owns the deep read):**

| dimension | 2025 jubilant-vase | 2026 jubilant-vase (Vincze, V3, 286.656 s) |
|---|---|---|
| model | 3-library weighted blend (Cat/LGB/XGB) | **single LightGBM config, 4 fitted instances** (1 global + 3 airport specialists) |
| blending | `scipy.minimize` weights on OOF | fixed **50/50** global↔specialist at EDDF/EGLL/EHAM only |
| validation | GroupKFold by `flight_id` | **three temporal simulations** (composition / forward / seasonal) with pre-declared acceptance gates and 1,000–2,000-resample paired bootstrap; a candidate must win ≥95% of resamples and must not get >50% of its edge from one row |
| external data | Acropole, OpenAP; ERA5 only as a plot | **none shipped**; METAR built and rejected |
| post-processing | none | clip to ≥ 60 s; a calibrated LIRF linear overlay for missing-AOBT departures with takeoff > 14,400 s after schedule — **changed 15 of 215,876 rows** |
| specialists | none | EDDF/EGLL/EHAM; gains 2.375 s / 8.743 s / 1.815 s across the three gates, **100% bootstrap wins**, official LB 290.488 → **286.656** (−3.832 s) |
| rejected branches (logged) | — | target encoding, public METAR, CatBoost, LR changes, **more airport specialists**, alternative early stopping, congestion ablation, **seed ensembles**, **wake specialists**, completed-arrival state, active taxi queue, scheduled clock time, model-capacity selection, post-hoc calibration |

The only genuine methodological through-line between the two `jubilant-vase` entries is coincidental:
both use gradient boosting on a tabular featurisation. Treat them as unrelated. **[X]**

---

## Applicable to resilient-hedgehog

Ordered by expected value. Each item is checkable. I mark where the fuel→taxi-out transfer is
**direct**, **partial** or **does not transfer**.

**Transfers directly (both targets are a quantity accumulated over a duration, and RMSE is the metric):**

1. **[TRANSFERS — but the 2025 form does NOT apply.] Do not copy the "predict a rate, multiply by
   duration" trick blindly.** In 2025 the target was fuel over a *segment whose duration is given in
   the submission file* — dividing by a **known** duration is a pure, invertible reparameterisation
   that removed the dominant scale factor. In 2026 the target **is** the duration; there is no
   independently-given denominator. The transferable form is the *principle*: **find the known
   scale variable that dominates the target's variance and divide it out.** Candidate denominators
   to test for taxi-out: the Network Manager implied/planned taxi time, the airport×runway×stand
   historical median taxi-out, or an unimpeded-taxi-time reference. Concretely: fit on
   `y / nm_implied_taxi` or `y − airport_stand_runway_median` and multiply/add back. **Check:**
   does an otherwise identical LightGBM improve OOF RMSE (in seconds, after inversion) versus fitting
   `y` directly? Precedent: ranks 1, 5, 6 and 10 of 2025 all fit a rate; ranks 3, 7, 8, 9 fit the
   raw quantity, and the raw-quantity teams occupy the bottom of the top-10. **[S] for the ranks.**

2. **[TRANSFERS.] Filter training outliers in the *normalised* space, not the raw space, and filter
   the low tail harder than you expect.** Rank 1's measured gain from a target-space filter was
   **23 RMSE** — the largest single measured number in the field — and they specifically found that
   **implausibly low values hurt more than implausibly high values**. For taxi-out, the analogue is
   rows whose taxi-out is physically impossible for the stand→runway pair (too short) rather than
   the long tail. **Check:** compute per (airport, stand-area, runway) an empirical floor (e.g. a
   geometric minimum taxi time), drop or down-weight training rows below it, and measure OOF RMSE.
   Be careful: RMSE rewards the long tail, so do **not** trim the high tail without a gate.

3. **[TRANSFERS.] Group your CV by whatever creates duplicated structure, and prove it matters.**
   Rank 4 states in writing that row-random CV over segments of the same flight gave them a CV/LB
   divergence, and that GroupKFold fixed it. Our 2026 rows are one-per-departure, so `flight_id`
   grouping is moot — **but the same failure mode exists via near-duplicate operational context**:
   departures in the same 15-minute bank at the same airport share congestion features almost
   exactly. **Check:** compare OOF RMSE under (i) random KFold, (ii) GroupKFold on
   `airport × date × hour`, (iii) the temporal splits we already use. If (i) is materially optimistic
   versus (iii), any feature selected under (i) is suspect.

4. **[TRANSFERS.] The public leaderboard is a September; the prize is an October.** In 2025,
   bright-lobster was **2nd on the phase-1 set and 11th on the combined set** after ~150 leaderboard
   submissions, while jubilant-vase reached 2nd with ~8. **Check:** every architectural change must
   clear a *local, temporal* gate before it is submitted. The 2026 solo entrant's design
   (three temporal arms, pre-declared acceptance criteria, ≥95% of paired-bootstrap resamples, and a
   "no single row carries >50% of the edge" rule) is a ready-made template and is already public.

5. **[TRANSFERS.] A crude, bounded, always-defined proxy beats an accurate but fragile one.**
   Rank 1's heuristic mass (a straight line between 0.8·MTOW and 0.5·(MLW+OEW)) beat their OpenAP
   recursive mass by 6 RMSE, because the recursive version propagated errors and was undefined when
   trajectory data was missing. For taxi-out the analogous latent is **queue position / departure
   demand at off-block**. **Check:** prefer a queue feature that is defined for every row (e.g.
   a count of scheduled departures in ±T minutes, which never has gaps) over one that depends on
   complete movement data for neighbours and silently goes NaN or wrong when coverage is patchy.
   And: **measure whether missing-is-informative beats imputation** — rank 4 set all kinematics to
   `0.0` with an `is_missing_data` flag rather than impute, and rank 2 shipped `is_missing_data` and
   `mass_source`/`fuel_source` *provenance categoricals*. Provenance-of-feature as a categorical is
   cheap and was in both the 2nd and 4th place solutions.

6. **[TRANSFERS.] Sub-window decomposition of the prediction interval.** Rank 1's "detailed vertical
   rate" split each segment into **10 equal-time parts** and emitted mean+std per part (20 features),
   which let a tabular model see the *shape* of the interval, not just its average. Our interval is
   the taxi-out itself and is unobserved at prediction time — but the **lead-up** is observable.
   **Check:** cut the 60 minutes before AOBT into N equal windows at the departure airport and emit
   per-window departure counts, arrival counts and runway flow, instead of only aggregate counts over
   one window. That is the same idea applied to the only time axis we can see.

7. **[TRANSFERS.] Group-relative deviation features.** Rank 7 built `altitude_diff_type =
   altitude_mean − groupby(typecode).mean()` and `speed_diff_phase`. The taxi-out analogue:
   `nm_implied_taxi − median(nm_implied_taxi | airport, runway, stand_area)`,
   `scheduled_departures_in_window − median(… | airport, hour-of-week)`. These are label-free
   (no target encoding, no leakage) and were the distinguishing feature of a top-7 solution.
   Rank 7 also **deleted** a `fuel_diff_flight` feature on the grounds that the target is unavailable
   at submission time — a reminder to check every `groupby().transform()` for target dependence.

**Transfers partially:**

8. **[PARTIAL.] Ensembling bought ~4 RMSE (1.6%) for rank 2 and ~1 RMSE for rank 4's hand-tuned
   0.7/0.3 blend, versus ~23 for one outlier filter and ~18 for one target change.** Blending is
   the *last* 2%, not the first 20%. But note the **most interesting** ensemble in the field:
   merry-jacket blended **two models trained on two different target parameterisations**
   (CatBoost on log1p(rate), XGBoost on raw quantity). If item 1 gives us two viable
   parameterisations of taxi-out, blending them is a cheap source of genuine diversity — more so
   than three libraries on one target. **Check:** build the ratio-target and raw-target models,
   measure their OOF correlation, and only blend if correlation < ~0.98.

9. **[PARTIAL.] Post-processing should be a *physical clip*, not a hand rule.**
   Rank 3 shipped `pred ← max(0, min(pred, 1.1 × max_rate[aircraft_type] × duration))`;
   rank 4 shipped `clip(lower=1.0)`; the 2026 solo entrant clips to ≥60 s. By contrast, rank 10's
   *hypothesis-driven* overlay (doubling A20N rates) cost **15 RMSE**. **Check:** a lower clip at a
   physically defensible minimum taxi time per (airport, stand-area, runway) is safe; any overlay
   that encodes a *theory* about the data must pass a bootstrap gate on held-out time before it ships.

10. **[PARTIAL — and the 2025 evidence is negative.] Per-group specialists.**
    Four 2025 teams shipped per-phase or per-class specialists (ranks 5, 6, 7, 8); **none beat the
    single global model.** The one that placed best (rank 5) split on **data quality**
    (`trajectory_completeness ≥ 0.3`), not on physics. Transfer to 2026: a **per-airport** specialist
    has independent positive evidence from the 2026 task itself (+3.832 s official at EDDF/EGLL/EHAM,
    gated), so the 2025 negative result does **not** rule it out — it warns that the grouping
    variable must be chosen for *data volume and regime difference*, not for interpretability, and
    that adding more specialists has diminishing/negative returns (that entrant's own log rejects
    "more airport specialists" and "wake specialists"). **Check:** before building any specialist,
    verify the group has enough rows to support a full-capacity model, and require the same
    bootstrap gate on all three temporal arms.
    A **data-quality** split is the untried variant with 2025 support: e.g. one model for departures
    with complete movement/AOBT data and one for those without.

11. **[PARTIAL.] A degenerate group-median baseline is a real competitor and a real diagnostic.**
    `unique-quicksand`'s per-aircraft-type median rate scored **256.27 → rank 10 of 19**, ~27% off
    the winner. The 2026 analogue (hierarchical historical median by airport/runway/stand) is already
    in the public 2026 repo as a control and scores 348–537 s locally versus 250–405 s for a global
    LightGBM. **Check:** keep a group-median control in the harness permanently; any feature or
    architecture change that does not widen the gap to it is noise.

**Does not transfer:**

12. **Aircraft-mass and physics-model features (OpenAP/Acropole/BADA, TOW estimation, drag/thrust
    coefficients, total-energy mass).** These were the single most important feature family in 2025
    (every top-3 team used them; rank 1 called mass *"information that the model cannot fully infer
    from kinematic variables"*). **Taxi-out duration is essentially independent of aircraft mass** —
    it is set by ground topology, queueing and ATC. Wake category / aircraft size still matter, but
    as proxies for **separation minima and taxi speed**, not for fuel physics. Do not port the
    mass-estimation machinery.

13. **Trajectory statistical embeddings (mean/std/min/max of altitude, vertical rate, Mach, TAS over
    the prediction window).** The 2025 window is airborne and observable; the 2026 window is on the
    ground and its contents are the answer. The transferable residue is only item 6.

14. **Weather.** ERA5/METAR/Open-Meteo were measured as neutral-to-negative by ranks 1, 6 and 11 in
    2025 **and** rejected under gate by the 2026 solo entrant. The 2025 failure reason (grid/scale
    mismatch with a fast-moving aircraft) **does not** apply on the ground — an airport METAR is
    exactly co-located with a taxiing aircraft, and low-visibility procedures, snow/de-icing and
    runway-configuration changes plausibly drive taxi-out. So the 2025 evidence is **weak** evidence
    against weather for 2026; the 2026 entrant's own gated rejection is the stronger evidence.
    **Check:** if we test METAR, test the *operationally meaningful* derived flags (LVP/CAT-II-III
    conditions, freezing precipitation, crosswind on the active runway → runway config change),
    not raw temperature/wind columns.

15. **Macroeconomic covariates.** Measured as zero-importance by rank 6. Do not spend time here.

**Process items, all [S]-backed:**

16. **Backward feature elimination, not forward accumulation.** Rank 1: *"Starting from all
    engineered feature groups, we successively removed or simplified features that did not provide a
    clear and robust improvement."* Rank 11 used forward `SequentialFeatureSelector` on 133
    candidates and finished 11th. **Check:** our elimination decisions should be logged with the
    OOF delta and the gate outcome, the way the 2026 solo entrant logs every rejected branch.

17. **Ship the repo.** Fourteen of thirty-three scored 2025 teams — including the would-be 3rd place
    (214.94) and the JOAS-published kindly-garden (227.32) — were **removed from the final ranking**
    for not publishing a documented GPLv3 repo. Repository readiness is worth more RMSE-equivalent
    than any feature in this document. **[S]**

18. **Do not read across from a stale README.** jubilant-vase's public README still says
    *"RMSE ~224 (Top 8)"* while the team finished **2nd at 211.99**. When scraping competitor repos
    for claimed scores, prefer the organiser's ranking page over the repo's own text. **[S]**
