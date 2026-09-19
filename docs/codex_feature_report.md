# Independent feature investigation — 2026-09-18

All differences below are **candidate minus comparator RMSE, in seconds; negative is better**. Candidate screens measure residual corrections; feature refits are reported separately. No submission was created or uploaded.

## Scope and provenance

Generated **63 candidates**, four hash controls, an airport-only control and three existing-feature controls. The 63 comprise 62 nonconstant training columns plus the constant schedule-second diagnostic. None of the ten rejected features in brief §7 was rerun. Arrival continuity overlaps the prior audit, but is retested because that audit did not fit these features in a fresh baseline or validate their corrections within one model. No operator target encoding, planned duration, distance, queue-ahead or bank-position candidate was included.

Read the [brief](CODEX_BRIEF.md), [instrument audit](investigation_2026-09-16/results/decision-rule-blind-to-tail.md), [same-model refutation](data_audit_2026-09-18/results/review-refutations.md), [arrival audit](data_audit_2026-09-18/results/arrival-rows.md), [timestamp audit](data_audit_2026-09-18/results/timestamps.md), [precision investigation](investigation_2026-09-16/results/timestamp-precision-clusters.md), and production code/configuration.

The existing read-only feature cache has 2,085,047 training and 344,841 ranking departures. Its recorded feature-source fingerprint `590ceb8f8170ff6b` equals the current feature-code hash. New features are separate keyed parquet files in `outputs/codex_fe/`; production source and `configs/main.yaml` are unchanged. All experiment writes stay in this worktree, all LightGBM jobs are serial and use two threads; Polars uses two threads.

## Protocol and its limits

The [pre-outcome protocol](codex_fe_results/protocol.md) and [configuration](../configs/codex_fe_screen.yaml) specify a fresh January–August model predicting September–December. August chooses stopping rounds; the model is refitted on January–August. **Every calibration and evaluation residual belongs to this same fit.** The mixed historical OOF file is never loaded.

Baseline training: 1,387,414 departures; 742 matched rounds, 90 unmatched rounds; 409.9 seconds. One seed, full column sampling, two model regimes, LIRF unmatched schedule reparameterisation. This is a deliberately controlled experiment, not an exact reproduction of the three-seed production model.

1. Fit airport × regime × candidate-cell residual means on September. Numeric cells use September-only deciles (exact levels for ≤20 values); categorical cells use values. Add 100 zero-residual pseudo-observations per cell; unseen cells get zero. Evaluate every candidate on all 185,674 October departures.
2. Keep at most five novel candidates with October 95% upper CI < 0, a matched-row improvement, and a point improvement greater than every hash control and airport-only correction. Save selection before evaluating November–December.
3. Refit selected corrections on September+October and confirm on all 328,009 November–December departures. A separately preregistered shallow joint learner tests interactions across all candidates against existing context and context plus four hashes. It uses 15 leaves, 120 rounds, learning rate .03 and no evaluation early stopping.
4. Refit the actual January–August baseline with the selected bundle, all candidates, and a hash control. Keep baseline round counts and seed fixed. No correction is transferred across these refits.

Intervals resample **whole calendar days**, paired between predictions, preserving within-day airport/traffic dependence. October intervals use 2,000 draws and are exploratory, with no familywise claim. Confirmation uses 10,000 draws and 99.167% intervals (Bonferroni for five selected scalars plus the preregistered joint test). Feature-refit intervals use 98.75% coverage for two bundles × two comparators. Seven-day circular moving-block intervals are an additional sensitivity analysis. These are conditional test-sample intervals, not model-training uncertainty or a guarantee about another season/year.

Full RMSE is primary. Matched/unmatched results, month and airport breakdowns, SSE changes and tail counts are reported without deleting extremes. The “drop200” diagnostic excludes a fixed set of the baseline’s worst 200 rows, rather than choosing a different set for each candidate; it never determines acceptance. There are only two confirmation months, so a month-bootstrap is not presented as reliable uncertainty.

Limits: no local fold reproduces 12 training months → January/July 2026. The existing trainer fits encoding tables before its inner stopping split, so inner-holdout encodings include that holdout; outer evaluation labels never enter fitting. Fixed round counts in verification prevent candidate-specific stopping from confounding feature additions. A univariate binned correction can miss conditional signal; the joint screen and full all-feature refit address that limitation but do not exhaust all possible feature representations.

## Candidate rationale and complete October screen

Hypotheses, not measured facts: clock revisions may distinguish recording regimes; precision and identity fragments may expose reporting conventions; stand continuity may constrain available turnaround time; runway/operator histories may describe local proxy-clock behaviour; traffic and weather derivatives may capture changing operating conditions. The table measures their incremental residual-correction value.

| Candidate | Full ΔRMSE [95% day CI] | Matched Δ | Unmatched Δ | Drop200 Δ | Selected |
|---|---:|---:|---:|---:|:---:|
| `airport_only` | -0.595 [-1.164, -0.080] | -0.206 | -10.922 | -0.307 |  |
| `arrival_revision` | -0.638 [-1.210, -0.100] | -0.253 | -10.922 | -0.376 |  |
| `actual_remaining` | -0.576 [-1.169, -0.060] | -0.186 | -10.922 | -0.283 |  |
| `filed_remaining` | -0.578 [-1.173, -0.048] | -0.187 | -10.922 | -0.284 |  |
| `iobt_eobt_revision` | -0.597 [-1.164, -0.076] | -0.208 | -10.922 | -0.298 |  |
| `aobt_iobt_revision` | -0.630 [-1.220, -0.091] | -0.244 | -10.922 | -0.383 |  |
| `aobt_lobt_revision` | -0.639 [-1.231, -0.096] | -0.254 | -10.922 | -0.396 |  |
| `iobt_schedule_revision` | -0.617 [-1.181, -0.114] | -0.230 | -10.922 | -0.291 |  |
| `eobt_schedule_revision` | -0.690 [-1.255, -0.168] | -0.308 | -10.922 | -0.300 | yes |
| `lobt_iobt_revision` | -0.612 [-1.184, -0.092] | -0.224 | -10.922 | -0.322 |  |
| `duration_revision` | -0.606 [-1.184, -0.099] | -0.218 | -10.922 | -0.303 |  |
| `mvt_second` | -0.445 [-0.885, -0.056] | -0.193 | -7.171 | -0.304 |  |
| `mvt_minute5` | -0.429 [-0.806, -0.079] | -0.190 | -6.825 | -0.289 |  |
| `schedule_second` | -0.595 [-1.164, -0.080] | -0.206 | -10.922 | -0.307 |  |
| `schedule_minute5` | -0.589 [-1.157, -0.077] | -0.206 | -10.751 | -0.300 |  |
| `aobt_second` | -0.593 [-1.159, -0.077] | -0.204 | -10.922 | -0.308 |  |
| `aobt_minute5` | -0.573 [-1.139, -0.052] | -0.182 | -10.922 | -0.280 |  |
| `arvt_second` | -0.572 [-1.133, -0.069] | -0.181 | -10.922 | -0.278 |  |
| `arvt_minute5` | -0.578 [-1.149, -0.058] | -0.187 | -10.922 | -0.290 |  |
| `schedule_hour` | -0.304 [-0.657, +0.036] | -0.242 | -2.144 | -0.293 |  |
| `schedule_day_gap` | -0.593 [-1.164, -0.070] | -0.206 | -10.867 | -0.295 |  |
| `aobt_hour` | -0.621 [-1.191, -0.092] | -0.234 | -10.922 | -0.324 |  |
| `aobt_day_gap` | -0.595 [-1.163, -0.080] | -0.206 | -10.922 | -0.307 |  |
| `offblock_clock_spread` | -0.564 [-1.159, +0.004] | -0.172 | -10.922 | -0.307 |  |
| `offblock_unique` | -0.595 [-1.163, -0.080] | -0.206 | -10.922 | -0.309 |  |
| `flight_rule` | -0.596 [-1.162, -0.080] | -0.206 | -10.934 | -0.307 |  |
| `flight_prefix` | -0.448 [-0.670, -0.205] | -0.316 | -4.191 | -0.427 |  |
| `callsign_suffix` | -0.519 [-1.079, -0.026] | -0.124 | -10.922 | -0.192 |  |
| `flight_number_band` | -0.313 [-0.701, +0.051] | -0.194 | -3.597 | -0.319 |  |
| `stand_prefix` | -0.554 [-1.157, -0.010] | -0.206 | -9.806 | -0.280 |  |
| `stand_number` | -0.317 [-0.638, -0.015] | -0.111 | -5.772 | -0.247 |  |
| `stand_suffix` | -0.613 [-1.221, -0.073] | -0.211 | -11.286 | -0.289 |  |
| `runway_heading` | -0.182 [-0.703, +0.288] | -0.034 | -4.065 | +0.094 |  |
| `runway_side` | -0.351 [-0.973, +0.232] | -0.202 | -4.417 | -0.036 |  |
| `same_airport` | -0.584 [-1.163, -0.063] | -0.205 | -10.628 | -0.286 |  |
| `dest_region_mismatch` | -0.584 [-1.149, -0.071] | -0.194 | -10.922 | -0.297 |  |
| `temp_c_trend` | -0.369 [-0.774, +0.006] | -0.151 | -6.153 | -0.205 |  |
| `wind_ms_trend` | -0.225 [-0.597, +0.108] | -0.115 | -3.210 | -0.163 |  |
| `gust_ms_trend` | -0.592 [-1.158, -0.080] | -0.204 | -10.905 | -0.304 |  |
| `vis_km_trend` | -0.660 [-1.242, -0.120] | -0.226 | -12.183 | -0.326 |  |
| `precip_mm_trend` | -0.607 [-1.187, -0.083] | -0.207 | -11.225 | -0.308 |  |
| `dewp_spread_c_trend` | -0.202 [-0.501, +0.096] | -0.065 | -3.801 | -0.146 |  |
| `gust_excess` | -0.587 [-1.126, -0.115] | -0.202 | -10.798 | -0.306 |  |
| `dep_acceleration_5` | -0.348 [-0.607, -0.077] | -0.165 | -5.248 | -0.281 |  |
| `arr_acceleration_5` | -0.339 [-0.642, -0.035] | -0.163 | -5.062 | -0.207 |  |
| `dep_acceleration_15` | -0.339 [-0.527, -0.135] | -0.173 | -4.829 | -0.379 |  |
| `arr_acceleration_15` | -0.192 [-0.401, +0.038] | -0.172 | -0.868 | -0.277 |  |
| `dep_acceleration_30` | -0.241 [-0.455, -0.026] | -0.138 | -3.052 | -0.253 |  |
| `arr_acceleration_30` | -0.171 [-0.435, +0.091] | -0.150 | -0.852 | -0.198 |  |
| `proxy_neighbor_deviation` | -0.636 [-1.201, -0.117] | -0.251 | -10.922 | -0.345 |  |
| `delay_neighbor_deviation` | -0.677 [-1.264, -0.132] | -0.294 | -10.922 | -0.397 | yes |
| `runway_headway_asymmetry` | -0.316 [-0.544, -0.081] | -0.212 | -3.203 | -0.326 |  |
| `stand_arrival_gap` | -0.991 [-1.322, -0.651] | -0.655 | -10.364 | -0.815 | yes |
| `stand_arrival_taxi` | -0.441 [-0.728, -0.148] | -0.265 | -5.280 | -0.341 |  |
| `stand_arrival_type_match` | -0.761 [-1.319, -0.259] | -0.418 | -10.096 | -0.496 | yes |
| `stand_arrival_delay` | -0.408 [-0.690, -0.142] | -0.386 | -1.356 | -0.527 |  |
| `stand_sched_turn` | -0.653 [-0.937, -0.364] | -0.552 | -3.819 | -0.715 |  |
| `stand_aobt_turn` | -0.851 [-1.410, -0.346] | -0.482 | -10.922 | -0.544 | yes |
| `runway_proxy_mean` | -0.239 [-0.590, +0.118] | -0.107 | -3.752 | -0.208 |  |
| `runway_proxy_sd` | -0.432 [-0.815, -0.068] | -0.091 | -9.385 | -0.191 |  |
| `runway_proxy_n` | -0.236 [-0.490, +0.035] | -0.207 | -1.205 | -0.273 |  |
| `operator_proxy_mean` | -0.474 [-1.055, +0.066] | -0.075 | -10.922 | -0.160 |  |
| `operator_proxy_sd` | -0.516 [-1.106, +0.025] | -0.120 | -10.922 | -0.195 |  |
| `operator_proxy_n` | -0.541 [-1.122, -0.024] | -0.147 | -10.922 | -0.214 |  |
| `noise_0` | -0.167 [-0.626, +0.275] | -0.167 | -0.331 | -0.309 |  |
| `noise_1` | -0.230 [-0.480, +0.020] | -0.143 | -2.640 | -0.242 |  |
| `noise_2` | -0.194 [-0.460, +0.086] | -0.184 | -0.658 | -0.298 |  |
| `noise_3` | -0.428 [-0.828, -0.056] | -0.179 | -7.067 | -0.299 |  |
| `hour_local` | -0.331 [-0.575, -0.067] | -0.180 | -4.410 | -0.319 |  |
| `mvt_minus_sched` | -0.412 [-0.918, +0.058] | -0.148 | -7.422 | -0.246 |  |
| `mvt_minus_aobt3` | -0.577 [-1.171, -0.010] | -0.186 | -10.922 | -0.301 |  |

One hash control has a nominally significant October improvement. This is measured evidence that a single screen CI is insufficient. The screen is not a declaration that all nonselected features are useless.

### Joint October screen

| Arm | Full ΔRMSE [95% day CI] | Matched Δ | Unmatched Δ |
|---|---:|---:|---:|
| joint_context | -1.115 [-2.108, -0.220] | -0.259 | -23.793 |
| joint_noise | -0.770 [-1.748, +0.207] | -0.259 | -14.345 |
| joint_novel | -2.851 [-3.757, -1.997] | -1.640 | -36.301 |

## Locked November–December correction confirmation

| Correction | Full ΔRMSE [adjusted day CI] | November Δ | December Δ | Matched Δ | Unmatched Δ |
|---|---:|---:|---:|---:|---:|
| `stand_arrival_gap` | -0.781 [-1.177, -0.396] | -0.654 | -0.912 | -0.694 | -3.813 |
| `stand_aobt_turn` | -0.172 [-0.811, +0.535] | +0.344 | -0.704 | -0.522 | +6.955 |
| `stand_arrival_type_match` | -0.132 [-0.771, +0.566] | +0.384 | -0.665 | -0.500 | +7.427 |
| `eobt_schedule_revision` | -0.086 [-0.709, +0.609] | +0.449 | -0.640 | -0.426 | +6.955 |
| `delay_neighbor_deviation` | +0.029 [-0.609, +0.707] | +0.621 | -0.582 | -0.297 | +6.955 |
| `airport_only` | +0.034 [-0.587, +0.703] | +0.585 | -0.536 | -0.292 | +6.955 |
| `noise_0` | -0.076 [-0.380, +0.254] | +0.179 | -0.340 | -0.270 | +3.905 |
| `joint_context` | -0.201 [-1.656, +1.322] | +1.012 | -1.458 | -0.665 | +9.265 |
| `joint_noise` | -0.225 [-1.614, +1.155] | +0.915 | -1.405 | -0.666 | +8.734 |
| `joint_novel` | -2.478 [-3.773, -1.188] | -1.516 | -3.476 | -2.512 | -5.555 |

Baseline RMSE is **228.324**; joint-correction RMSE is **225.846**. Joint novel versus context-only: -2.277 [-2.928, -1.656]; versus context plus hashes: -2.254 [-2.823, -1.685]. These comparator intervals use the same adjusted confidence level and are supporting comparisons, not additional independently controlled discoveries. Airport-only and noise rows are controls, not selected hypotheses.

Only `stand_arrival_gap` among the five selected scalar corrections retains an adjusted CI below zero. The joint correction improves in both months and all ten airports. This supports recoverable signal within the fixed baseline; it does **not** authorize attaching this correction to a model trained on all 12 months.

## Availability and validation

Candidate generation passes a departure-label-masking test: replacing all departure labels and hidden block timestamps with nulls leaves candidates identical. Arrival joins use only arrivals already on-block before the known departure take-off, capped at six hours; they are candidate links, not guaranteed aircraft rotations. Neighbour histories exclude the current departure with a 60-second lag. Train/rank schemas, ID uniqueness, unseen-cell behaviour, train-only bin boundaries and exact paired-bootstrap arithmetic pass [checks.py](../scripts/codex_fe/checks.py). Lint and formatting checks pass on the experiment scripts.

`schedule_second` is constant in both sets; `precip_mm_trend` is constant in ranking only. [Availability rates](codex_fe_results/parity.json) and [cardinality audit](codex_fe_results/candidate_audit.json) are saved for every column. No ranking labels were used.

## Definitions

| Candidate | Definition / motivation |
|---|---|
| `arrival_revision` | Seconds ARVT_3_flt minus ARVT_1_flt; clock update/consistency information. |
| `actual_remaining` | Seconds ARVT_3_flt minus MVT_TIME_UTC_mvt; clock update/consistency information. |
| `filed_remaining` | Seconds ARVT_1_flt minus MVT_TIME_UTC_mvt; clock update/consistency information. |
| `iobt_eobt_revision` | Seconds IOBT_flt minus EOBT_1_flt; clock update/consistency information. |
| `aobt_iobt_revision` | Seconds AOBT_3_flt minus IOBT_flt; clock update/consistency information. |
| `aobt_lobt_revision` | Seconds AOBT_3_flt minus LOBT_flt; clock update/consistency information. |
| `iobt_schedule_revision` | Seconds IOBT_flt minus SCHED_TIME_UTC_mvt; clock update/consistency information. |
| `eobt_schedule_revision` | Seconds EOBT_1_flt minus SCHED_TIME_UTC_mvt; clock update/consistency information. |
| `lobt_iobt_revision` | Seconds LOBT_flt minus IOBT_flt; clock update/consistency information. |
| `duration_revision` | Change in end-to-end duration, separating off-block and arrival revisions. |
| `mvt_second` | Timestamp precision/reporting convention. |
| `mvt_minute5` | Five-minute timestamp rounding/reporting convention. |
| `schedule_second` | Timestamp precision/reporting convention. |
| `schedule_minute5` | Five-minute timestamp rounding/reporting convention. |
| `aobt_second` | Timestamp precision/reporting convention. |
| `aobt_minute5` | Five-minute timestamp rounding/reporting convention. |
| `arvt_second` | Timestamp precision/reporting convention. |
| `arvt_minute5` | Five-minute timestamp rounding/reporting convention. |
| `schedule_hour` | Off-block daily phase distinct from take-off hour. |
| `schedule_day_gap` | Known clock date disagreement; not the hidden block-date rollover. |
| `aobt_hour` | Off-block daily phase distinct from take-off hour. |
| `aobt_day_gap` | Known clock date disagreement; not the hidden block-date rollover. |
| `offblock_clock_spread` | Range of NM off-block clock revisions. |
| `offblock_unique` | Number of distinct NM off-block clocks. |
| `flight_rule` | Operational flight rule absent from baseline categoricals. |
| `flight_prefix` | Flight designator prefix on unmatched as well as matched rows. |
| `callsign_suffix` | Callsign suffix reporting/flight pattern. |
| `flight_number_band` | Coarse flight-number service family. |
| `stand_prefix` | Stand region sharing information across individual stands. |
| `stand_number` | Ordinal stand location proxy. |
| `stand_suffix` | Stand subdivision/remote-stand coding. |
| `runway_heading` | Share information across runway directions. |
| `runway_side` | Parallel runway side / helipad distinction. |
| `same_airport` | Local flight indicator. |
| `dest_region_mismatch` | Large filed/destination mismatch. |
| `temp_c_trend` | One-hour weather change rather than level. |
| `wind_ms_trend` | One-hour weather change rather than level. |
| `gust_ms_trend` | One-hour weather change rather than level. |
| `vis_km_trend` | One-hour weather change rather than level. |
| `precip_mm_trend` | One-hour weather change rather than level. |
| `dewp_spread_c_trend` | One-hour weather change rather than level. |
| `gust_excess` | Gustiness above mean wind. |
| `dep_acceleration_5` | Recent departure rate above trailing hourly rate. |
| `arr_acceleration_5` | Recent arrival rate above trailing hourly rate. |
| `dep_acceleration_15` | Recent departure rate above trailing hourly rate. |
| `arr_acceleration_15` | Recent arrival rate above trailing hourly rate. |
| `dep_acceleration_30` | Recent departure rate above trailing hourly rate. |
| `arr_acceleration_30` | Recent arrival rate above trailing hourly rate. |
| `proxy_neighbor_deviation` | Own proxy relative to recent airport reporting state. |
| `delay_neighbor_deviation` | Own off-block delay relative to recent airport state. |
| `runway_headway_asymmetry` | Local runway service interruption asymmetry. |
| `stand_arrival_gap` | Most recent completed same-stand arrival within six hours: stand_arrival_gap |
| `stand_arrival_taxi` | Most recent completed same-stand arrival within six hours: stand_arrival_taxi |
| `stand_arrival_type_match` | Most recent completed same-stand arrival within six hours: stand_arrival_type_match |
| `stand_arrival_delay` | Most recent completed same-stand arrival within six hours: stand_arrival_delay |
| `stand_sched_turn` | Most recent completed same-stand arrival within six hours: stand_sched_turn |
| `stand_aobt_turn` | Most recent completed same-stand arrival within six hours: stand_aobt_turn |
| `runway_proxy_mean` | Previous 60 minutes (lag 60 seconds), same airport and runway, known MVT-AOBT proxy mean. |
| `runway_proxy_sd` | Previous 60 minutes (lag 60 seconds), same airport and runway, known MVT-AOBT proxy sd. |
| `runway_proxy_n` | Previous 60 minutes (lag 60 seconds), same airport and runway, known MVT-AOBT proxy n. |
| `operator_proxy_mean` | Previous 60 minutes (lag 60 seconds), same airport and operator, known MVT-AOBT proxy mean. |
| `operator_proxy_sd` | Previous 60 minutes (lag 60 seconds), same airport and operator, known MVT-AOBT proxy sd. |
| `operator_proxy_n` | Previous 60 minutes (lag 60 seconds), same airport and operator, known MVT-AOBT proxy n. |
| `noise_0` | Deterministic hash negative control. |
| `noise_1` | Deterministic hash negative control. |
| `noise_2` | Deterministic hash negative control. |
| `noise_3` | Deterministic hash negative control. |

## Reproduction and artifacts

Run from this worktree with `POLARS_MAX_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src`. Use `.venv/bin/python scripts/codex_fe/candidates.py`, then `baseline.py`, `screen.py screen`, `screen.py confirm`, `refit.py noise`, `refit.py selected`, `refit.py all`, `evaluate_refits.py`, `checks.py`, and `report.py`. Run training commands serially. `run_stages.py` coordinates the middle stages after baseline completion; clear stale output metadata when intentionally starting a new experiment. Predictions are keyed parquet files under ignored `outputs/codex_fe/`; all aggregate evidence is under `docs/codex_fe_results/`.

**Commit limitation:** `git add` failed because `/home/user/prc-data-challenge/.git/worktrees/codex-fe/index.lock` is outside the writable workspace and the filesystem is read-only. No commit could be created on the requested branch. Files and results are preserved in this worktree; no workaround modified another checkout.

## NEW — continuation status (2026-09-18)

The continuation follows [continuation_protocol.md](codex_fe_results/continuation_protocol.md).
The prior findings above are preserved. The saved run completed only the noise feature refit;
selected/all refits are being completed before drawing feature-adoption conclusions.
The handoff's statement that the scalar gap improves at all ten airports is incorrect:
`confirmation.json` records improvements at seven, and worsening at EHAM (+0.374),
LEMD (+0.428), LSZH (+1.035). All-ten improvement applies to the joint correction.

Stage 1 commit attempt: `git add docs/codex_fe_results/continuation_protocol.md` failed with
`Unable to create '/home/user/prc-data-challenge/.git/worktrees/codex-fe/index.lock': Read-only file system`.
No workaround used; results will remain in the writable worktree.

### NEW — pipeline and failure diagnosis measurements

`stand_arrival_gap` now exists in `prc.features.candidate_features`, hidden by default via
`CANDIDATE_COLS`; `configs/codex_fe_gap.yaml` explicitly enables it. The production helper
matches the saved experimental column exactly on all 2,085,047 training and 344,841 ranking
rows ([pipeline_parity.json](codex_fe_results/pipeline_parity.json)). Existing feature tests
plus a boundary/key/order/opt-in test pass (8 tests). Exact-time arrivals are excluded;
six-hour equality is included; missing anchors and wrong airports yield null.
This changes the feature-source fingerprint, so an ordinary pipeline run would rebuild its
cache. This investigation continues reading the original cache and never writes `data/`.

[scalar_diagnosis.json](codex_fe_results/scalar_diagnosis.json) measures a specific failure:
`stand_aobt_turn`, `eobt_schedule_revision`, and `delay_neighbor_deviation` are 100% null
on unmatched confirmation rows. Their corrections are exactly airport-only there.
On 1,356 November unmatched rows they add **51,760,194 squared seconds**, +14.567 RMSE;
on 1,649 December unmatched rows they remove 7,678,201, −2.968 RMSE. The November mean
correction is −36.965 s while the mean evaluation residual is +28.826 s. The type-match
correction adds 51,684,670 squared seconds on November unmatched rows, despite having
nonmissing feature values; it is not literally the same correction.

Gap is nonmissing on 75.59% / 76.96% of November / December unmatched rows, reducing their
SSE by 13,847,863 / 10,195,215 and RMSE by 3.925 / 3.944 seconds. All four failed scalars
still improve pooled matched RMSE (prior confirmation table). **Inference:** scalar screening
partly rewarded airport calibration that failed to transfer to November unmatched traffic;
gap retained discriminating information in that regime. These post-hoc decompositions do
not prove a causal turnaround mechanism, or that the other features cannot help a joint fit.

All five selected columns are now opt-in production features via `configs/codex_fe_selected.yaml`.
The expanded parity check matches all five columns exactly on every train/ranking row; all eight
feature tests pass again. This integration adds no residual correction to any model.

The 11 cache/config tests also pass. `main.yaml` is unchanged. The opt-in configs use weather,
matching the cached baseline. They preserve this experiment's one seed, full column sampling
and 742/90 rounds; they are not a reproduction of the production three-seed ensemble.

Ten-month gap/selected refits generate their added columns directly with the production
helpers, using the original cached context. The eight-month refits used the saved sidecars;
full-row exact parity validates these as the same feature values. This avoids rebuilding or
writing the shared cache. It does not assert byte-identical results for a future full pipeline
rebuild (column order, stopping and production ensembling can change fitted trees).

A fresh [label-support audit](codex_fe_results/data_support.json) finds labels only in January–December
2025. January 2026 has 152,719 ranking departures and July 2026 has 192,122; both have zero
labels. Thus a full-twelve-month fit cannot have an honest subsequent local RMSE measurement.
I chose the ten-month sensitivity over unscored full-year inference: fitting/ranking prediction
alone would demonstrate execution, not establish the competition effect. No full-year model
or ranking prediction was produced in this continuation.

### NEW — actual feature-refit measurements

Results below use actual feature additions and fresh fits, without residual corrections.

| Comparison | ΔRMSE [day CI] | 7-day CI | Nov Δ | Dec Δ | Matched Δ | Unmatched Δ |
|---|---:|---:|---:|---:|---:|---:|
| noise_vs_baseline | -0.145 [-0.711, +0.426] | [-0.667, +0.396] | -0.187 | -0.101 | +0.028 | -4.037 |
| selected_vs_baseline | -3.663 [-5.048, -2.355] | [-5.053, -2.039] | -4.305 | -3.005 | -3.351 | -15.843 |
| selected_vs_noise | -3.518 [-4.939, -2.151] | [-4.991, -1.881] | -4.118 | -2.904 | -3.379 | -11.806 |
| all_vs_baseline | -4.333 [-6.314, -2.271] | [-6.046, -2.429] | -5.554 | -3.084 | -3.774 | -22.786 |
| all_vs_noise | -4.188 [-6.025, -2.216] | [-5.748, -2.429] | -5.366 | -2.983 | -3.802 | -18.749 |
| gap_vs_baseline | -3.242 [-4.282, -2.225] | [-4.226, -2.144] | -4.302 | -2.157 | -2.673 | -20.262 |
| gap_vs_noise | -3.097 [-4.279, -1.949] | [-4.194, -1.921] | -4.115 | -2.056 | -2.701 | -16.225 |
| selected_vs_gap | -0.421 [-1.261, +0.492] | [-1.324, +0.461] | -0.003 | -0.848 | -0.678 | +4.419 |
| all_vs_selected | -0.670 [-2.064, +0.711] | [-1.974, +0.498] | -1.249 | -0.079 | -0.422 | -6.943 |
| baseline_10m_vs_baseline | -2.689 [-4.923, -0.954] | [-5.121, -0.822] | -3.156 | -2.210 | -2.436 | -12.150 |

Selected/all bundle comparisons retain the original 98.75% intervals. Gap and training-size
comparisons use descriptive 95% intervals: their design follows the already-seen confirmation
outcomes. Seven-day intervals are a dependence sensitivity. These intervals omit training uncertainty.

Completed fit metadata:

- `refit_all.json`: 1,387,414 training rows, 742/90 rounds, 397.3 s.
- `refit_baseline_10m.json`: 1,757,038 training rows, 742/90 rounds, 235.0 s.
- `refit_gap.json`: 1,387,414 training rows, 742/90 rounds, 241.5 s.
- `refit_noise.json`: 1,387,414 training rows, 742/90 rounds, 248.0 s.
- `refit_selected.json`: 1,387,414 training rows, 742/90 rounds, 254.3 s.

Stage commit attempts remain blocked by the read-only git worktree index lock. Files are saved locally.
