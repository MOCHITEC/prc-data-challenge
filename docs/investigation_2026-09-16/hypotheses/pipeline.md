# Hypothesis generator: pipeline / evaluation / reproducibility (subagent output, verbatim, 2026-09-16)

"Verified" = measured in the agent's session with the command shown; "Supposed" = inference. No models trained (one <1 s 2,000-row LightGBM semantic check, item 2).

| # | id | Hypothesis | Needs training? | Status |
|---|---|---|---|---|
| 1 | v4-config-not-reproducible | `configs/submitted_v4.yaml` on today's code does not build v4: model B silently gets 255 leaves / min_data 20 / 3000 rounds (TrainConfig defaults), `hour_utc` is present, `force_row_wise` is on. | no | verified (config resolution); consequence supposed |
| 2 | refit-seed-semantics-changed | Commit d182fc1 made the refit pass explicit `bagging_seed`/`feature_fraction_seed`, which changes the booster even at n_seeds 1, while the early-stopping fit is unchanged — v2→v8 kept identical model-B rounds (658) but got a different model. | yes, model B only (~1–2 min) | semantics verified; magnitude supposed |
| 3 | lb-decided-by-20-lirf-rows | The 37 s LB gap between builds sits in ~20 LIRF unmatched rows: v3/v4 emit 16–18 unmatched predictions > 12 h (max 108,552 s), every current-code build emits 2 (max ≤ 68,581). | no | verified |
| 4 | hour-utc-absent-in-v3v4 | v3/v4's model B was trained without `hour_utc` and that absence, not weather or linear_tree, produced the >12 h LIRF predictions. | yes, model B only | supposed |
| 5 | holdout-six-rows-decide-rounds | Model B's round count is set by an L2 early-stop on 1,649 Dec-2025 rows containing 6 rows ≥ 4 h; the docs' "33 extreme rows" is the ≥ 12 h count, the ≥ 4 h count is 119. | no | verified counts; consequence supposed |
| 6 | holdout-encoding-self-inclusion | Label encodings (`ref_*`, `arh_med`, `apt_med`) are fitted on the whole training side including the inner-holdout month, so the early-stopping holdout sees its own labels. | yes (B cheap; A expensive) | code path verified; effect supposed |
| 7 | n-features-72-mislabel | "72 features" is frame width incl. `month`; the matrix was 80 numeric + 10 categorical (v1/v2/v8), 97+10 (v3/v4), 98+10 today. | no | verified |
| 8 | fold-shape-vs-ranking | No fold has the production shape (12 months → Jan+Jul 2026), `overall` silently discards `jul_fwd`, unmatched share differs (1.08 % train vs 1.53 % ranking). | no | verified structure; consequence supposed |
| 9 | decision-rule-blind-to-tail | `prc.compare` adopts only on the drop-200 CI, ignores the month bootstrap, and by construction removes the ~20 rows that moved the LB. | no | verified code; consequence supposed |
| 10 | cache-fingerprint-gaps | Feature-cache fingerprint hashes only `features.py`; `weather.py`, `airports.py`, METAR cache, `train.py` do not invalidate it. Current cache consistent. | no | verified |
| 11 | alignment-weather-probe-checked-ok | Row alignment, Int32 rounding, weather coverage for 2026, month-edge rows, and the v9 probe construction are NOT the cause. | no | verified negative |
| 12 | library-drift-refuted | Library version change 09-10→09-15 explains v4→v8. | no | verified negative (same dist-info since 09-10 07:25); no lockfile |

## Per-item detail

### 1. v4-config-not-reproducible
Resolved both configs through TrainConfig + for_unmatched():
```
configs/submitted_v4.yaml | model A: leaves 127, min_data 50, linear_tree True, force_row_wise True, rounds 1500, seeds 1
                          | model B: leaves 255, min_data 20, linear_tree True, rounds 3000
configs/main.yaml         | model A: 127/50, linear_tree None, rounds 2000, seeds 3 | model B: 127/50, rounds 2000
```
In v4-era code (`git show d7073bf:src/prc/train.py`, predict_fold) model B used cfg itself → 127/50/1500. Defaults unmatched_num_leaves=255 / min_data 20 / num_rounds 3000 at train.py:93-95 are exactly the v5 setting (325.22). v4 also trained without hour_utc and without force_row_wise. `git show a96e43a:configs/v4_linear_tree.yaml` had `scheme: fast` (field removed); submitted_v4.yaml has no unmatched_* keys.

### 2. refit-seed-semantics-changed
- `git log -S'bagging_seed' -- src/prc/train.py` → introduced in d182fc1 (2026-09-14); its comment "One seed is the default, and then this is the single fit it was before" is contradicted by the check below.
- Toy check (2,000 synthetic rows, num_leaves 15, 50 rounds, deterministic+force_row_wise): {seed:42} vs {seed:42, bagging_seed:42, feature_fraction_seed:42} → old-vs-old identical True; old-vs-new identical False, RMS diff 0.199.
- Pipeline records: v2 (`git show 4e807b5^:experiments/pipeline.jsonl`) rounds_unmatched 658; det_a/v8 rounds_unmatched 658. Same rounds, yet on unmatched rows RMS(v2 − v8) = 314.5 s (84/5,290 rows identical) vs 38.4 s on matched rows.
- det_a vs det_b (identical configs, same code): model B rounds bit-identical (commit 8dc2152); model A rounds 1126/1146/1131 vs 1273/1034/1173. Same-code model B is reproducible; v2→v8 delta is a code/param difference, not float noise.
- Confirm: refit model B (658 rounds, ~1 min) with params only vs with explicit sub-seeds; compare on the 5,290 ranking unmatched rows.

### 3. lb-decided-by-20-lirf-rows (verified on files, joined on MVT_ID_mvt to submitting.parquet)
```
UNMATCHED 5,290 rows          med    mean   p99     max    n>7200 n>14400 n>43200
v2  (old code, LB 312.02)    1062   1612  10256   68253   104    33      2
up_v3 (old, LB 306.29)       1092   1675   9359  107724    82    31     16
up_v4 (old, LB 305.03)       1112   1702   9457  108552    83    31     18
v5  (LB 325.22)              1060   1559   9697   61356   106    32      2
v7  (LB 342.59)              1082   1608  10169   66693   100    33      2
v8  (LB 343.80)              1066   1608  10184   68581   101    35      2
RMS(v8 − v4) on unmatched = 1735.5 s; top-20 rows carry 73.2 % of Σgap²; all 20 are LIRF (v4 40–108 k vs v8 30–40 k).
MATCHED 339,551 rows: RMS(v8 − v4) = 73.8 s → LB difference 1.43 s (v9 vs v4).
```
Training side: unmatched rows with y ≥ 43,200 s = 33; LIRF ≥ 14,400 s = 115 of 119. All training rows y ≥ 43,200: 34 of 2,085,047. main.yaml header claim "v5 predicted over 12 h twice as often as training" — measured: v5 emits 2 such predictions on 5,290 rows (training rate 33/22,470 ⇒ ~7.8 expected); v4 emits 18.
Supposed: true labels of those LIRF rows ≈ 86,400 s (day rollover). Cannot be checked without labels.

### 4. hour-utc-absent-in-v3v4
v3 already has 16 rows > 43,200 with no linear_tree → linear_tree not necessary; v7 (weather on, hour_utc on, seeds ×3) has 2. RMS(up_v3 − up_v4) on unmatched = 281 s. Confounded with item 2. Confirm: fit model B with exclude_features [hour_utc], weather on, 127/50, one seed, old seed semantics; count unmatched predictions > 43,200 (≥ ~15 confirms; 2 refutes).

### 5. holdout-six-rows-decide-rounds (polars on feat_train.parquet)
Unmatched rows 22,470; y ≥ 7,200: 481; ≥ 14,400: 119; ≥ 43,200: 33; ≥ 80,000: 15; y ≤ 0: 251. Dec-2025 unmatched (inner holdout, `_inner_holdout_mask` = latest month) = 1,649 rows, 6 with y ≥ 14,400. The holdout is the same month every run (train.py:235-249) — not a random draw; variation across configs comes from the model. Doc discrepancy: handoff §6 "極端値（y ≥ 4h 相当）は 33 行" — 33 is the ≥ 12 h count.

### 6. holdout-encoding-self-inclusion
predict_fold (train.py:317-318) fits ref/arh/apt/op on all labelled training rows, then `_fit` carves the holdout month out of the same frame; holdout rows' ref_p10/arh_med/apt_med contain their own labels (diluted by reference_min_count=30, arh min 10). Guard G2 tests only the evaluation fold's labels, not the holdout month's via encodings. Confirm: fit encodings on months 1–11, attach to Dec, same early-stop; compare best_iteration.

### 7. n-features-72-mislabel
Old numeric_feature_columns (`git show d7073bf:src/prc/features.py:293-300`) excluded TAXITIME_SEC_mvt but included month; 72 = 71 frame numerics + month. Current cache: numeric_feature_columns 99; visible_numeric_columns(main) 89 on raw frame, 98 after attach_encodings, +10 categoricals ⇒ 108 matrix columns. Weather adds 18. Derived: v2/v8 model A 71+9=80 numeric; v3/v4 70(−hour_utc)+18+9=97; today 98. n_features semantics in experiments.jsonl changed at 97e67c6. Also: window_coverage_60m is cast to Int32 (features.py:227-229) and holds only {0,1} — constant-ish, consistent train/rank.

### 8. fold-shape-vs-ranking
Ranking months 2026-01 (152,719 rows), 2026-07 (192,122). Folds (cv.py:96-121): jan, jul, jul_fwd (Jan–Jun → Jul), forward (Jan–Oct → Nov–Dec). run_cv builds OOF with pl.coalesce in fold order, so jul_fwd predictions never enter `overall`: overall.n = 672,428 while Σ fold n = 863,141. Unmatched share train 1.08 % vs ranking 1.53 % (Jan 2,453, Jul 2,837); training unmatched/month 1,082 (Mar)–3,938 (Jul). Routing rule `mvt_minus_aobt3` null in both paths (train.py:329,348): ranking 5,290 such rows, 7 with FLIGHT_ID_mvt (train 46 of 22,470) — consistent.

### 9. decision-rule-blind-to-tail
verdict() (compare.py:518-546) uses only robustness_curve; month_bootstrap_rmse_diff printed (line 584) but unused; robustness_curve (line 470) drops the K rows with the largest max(se_a, se_b) — exactly where two builds disagree most (the ~20 LIRF rows). paired_bootstrap_rmse_diff (cv.py:157-187) arithmetically correct. A `folds: [forward]` OOF compared with a 3-fold OOF silently restricts to Nov–Dec rows.

### 10. cache-fingerprint-gaps
cache_fingerprint (pipeline.py:250-266) = raw file names/sizes, row counts, use_weather, sha256(features.py) only. Current cache feature_code 590ceb8f8170ff6b == sha256(features.py) now; built 09-15 11:38:20 after last features.py commit (11:37:52); weather.py last changed 34808da (07:49) — consistent today. Not in key: weather.py, airports.py, data/ref/metar_hourly.parquet, METAR CSVs, train.py.

### 11. alignment-weather-probe-checked-ok (verified negative)
All seven submission files: 344,841 rows, ids in submitting.parquet order, MVT_ID_mvt Float64 / TAXITIME_SEC_mvt Int32 (np.rint, ≤0.5 s). v9 probe exact: RMS(v8 − v9) matched = 0.0; RMS(up_v4 − v9) unmatched = 0.0. METAR covers 2025-01-01 → 2026-07-31 all 10 stations; Jan/Jul 2026 744 h/station (LSZH 743); weather null rate feat_rank 0.0 (wind_ms 0.02 % Jul), feat_train ≤ 0.12 %. Both frames 132 columns, identical names/order/dtypes; MVT_TIME_UTC_mvt UTC-aware. Month-edge context rows 56. Unseen categories in ranking: STAND 284 rows, OPERATOR 177, ADES 174, AIRCRAFT_TYPE 18, RUNWAY 0. Side fact: backfilled uploaded_at for v1–v4 are ~9 h later than local outputs mtimes (JST labelled +00:00); affects only the daily counter.

### 12. library-drift-refuted (verified negative)
lightgbm 4.7.0, polars 1.44.2, numpy 2.4.6, pyarrow 25.0.1 dist-info all dated 2026-09-10 07:25 (before v1–v4). No lockfile; pyproject pins nothing.

## Not done
No make run, no model-B refit, no cache rebuild. Items 2, 4, 6 each need a model-B-only fit (~1–2 min) to move from supposed to measured; item 3's causal reading needs 2026 labels.
