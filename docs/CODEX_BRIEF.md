# Brief for an independent feature-engineering effort

Everything below is measured. Where a figure came from, the result file is named. No
interpretation, no recommendation, and no hypothesis of the briefer is included — the reader
designs its own investigation.

## 1. The task

Predict `TAXITIME_SEC_mvt` (seconds) for departures at ten European airports. Metric: RMSE.
Submission = 344,841 rows, `MVT_ID_mvt` + `TAXITIME_SEC_mvt`, in `data/raw/submitting.parquet`
order. Deadline 2026-10-11; five submissions per day.

## 2. The label is an exact timestamp difference

`TAXITIME_SEC_mvt == MVT_TIME_UTC_mvt − BLOCK_TIME_UTC_mvt` on 2,085,047 / 2,085,047 training
rows (`investigation_2026-09-16/results/sched-filled-block.md` §A).

## 3. What exists at prediction time

Measured on the 344,841 scored rows:

| column | populated |
|---|---|
| `MVT_TIME_UTC_mvt` (take-off) | 100 % |
| `SCHED_TIME_UTC_mvt` (scheduled off-block) | 100 % |
| `BLOCK_TIME_UTC_mvt` (actual off-block) | **0 %** |
| `TAXITIME_SEC_mvt` (label) | 0 % |
| `AOBT_3_flt` (NM's off-block record) | 98.5 % (339,551 rows) |
| the other 15 `_flt` columns | 98.5 %, all-or-nothing with `AOBT_3_flt`, 0 violations |

`SCHED_TIME` is anchored to off-block, not take-off: median(BLOCK − SCHED) = +416 s,
median(EOBT_1 − SCHED) = 0 s, median(MVT − SCHED) = +1,386 s, median label 907 s.

Arrival rows keep `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` in the ranking set (null rate 0).

## 4. The split

Rows where `AOBT_3_flt` is null go to a separate model: 22,470 training / 5,290 ranking rows
(1.53 %). They carry 48.1 % of squared error. Splitting beats one model by 74.10 s
(`findings.md` §1.1).

`AOBT_3 − BLOCK` on matched non-LIRF rows: median −56 s, **sd 364 s**, 21.2 % within 60 s.

## 5. Current state

| submission | LB |
|---|---|
| v4 | 305.027 |
| v10 (current best) | **284.3584** |

Leader 241.75. Standing 38 / 152. v10 = v4's matched predictions byte-identical + a new model B,
so its −20.67 s is attributable to model B alone.

Model: LightGBM, target `TAXITIME − proxy` where `proxy = MVT_TIME − AOBT_3`; at LIRF the
unmatched model's target is additionally shifted by `mvt_minus_sched`. Config `configs/main.yaml`.
Feature matrix: 98 numeric + 10 categorical for model A, 83 + 10 for model B.

## 6. Measured properties of the evaluation instrument

- `prc.compare`'s verdict reads only the K=0 and K=200 entries of the robustness curve. The 200
  rows it drops contain 84.5 % of the squared disagreement between two builds, 9 of 10 rollover
  rows and 17 of 18 rows with label ≥ 12 h (`results/decision-rule-blind-to-tail.md`).
- The OOF file `experiments/oof_main-48c19bc830.parquet` contains predictions from **three
  different models** (the jan, jul and forward folds). A correction fitted on one fold's residuals
  and applied to another's is measured to transfer negatively (−0.45 to −1.63), while within one
  model it transfers positively (`results/review-refutations.md` C2).
- No fold has the production geometry (12 months → Jan + Jul 2026). `overall` silently drops the
  `jul_fwd` fold (`results/fold-shape-vs-ranking.md`).
- With `feature_fraction: 0.8`, adding any column changes the column sampling in every tree. A
  hash column built to carry zero signal cost +0.53 s, indistinguishable from three real features
  (+0.49 / +0.39 / +0.50). Screening therefore needs `feature_fraction: 1.0` and a noise control
  column (`investigation_2026-09-16/docs/features_log.md`).
- Paired-bootstrap CI width at K=200 is 0.39–0.60 s on forward-only pairs; the unmatched-only CI
  is 385–724 s on three-fold pairs.

## 7. Feature attempts already measured and rejected

All ten, against a noise control on the forward fold, 200-row-dropped difference (+ is worse):
`queue_ahead` −0.16 [−0.40,+0.08], `bank_position` −0.19, `stand_rwy_share` −0.10,
`active_runways_60m` −0.06, `dep_arr_ratio_30m` +0.03, `congestion_vs_usual_30m` +0.09,
operator-offset encodings rejected, destination distance rejected, `planned_flight_s`
−0.21 [−0.49,+0.08]. (`docs/features_log.md`)

Ceiling measurements, with their protocol stated because they differ: airport × stand × runway ×
hour × aircraft-type within-group RMSE is 265.97 **in-sample** and 422.03 **out-of-month**; the
model is 216.44 on the same rows. Of 45 oracle groupings tried, the best honest one is 332.37
(`data_audit_2026-09-18/results/matched-error.md`).

## 8. Measured properties of the data

- At LIRF, `BLOCK_TIME` is filled with `SCHED_TIME` on 48.5 % of unmatched and 17.8 % of matched
  departures, and on 10.08 % of arrivals; elsewhere 1.9–3.7 %. Taxi-in-stratified, the spike runs
  16× (0–300 s band) to 648× (1,800–3,600 s) at LIRF and 0.94–1.40 at EGLL/LTFM
  (`results/review-arrtaxi.md`).
- 100,049 filled rows total; 12,146 of them also have |BLOCK − AOBT_3| > 600 s. Separately,
  39,224 scored rows are far-but-not-filled and carry 12.65 % of total squared error, against
  0.81 % for 1,871 far-and-filled non-LIRF rows (`results/review-refutations.md` C3).
- 16 training rows have date(MVT) − date(BLOCK) = 1 day; 10 of them are 34.7 % of OOF squared
  error (`results/block-day-rollover.md`).
- `RUNWAY_mvt == 'H'` at LSZH is a helipad (387 rows, median label 0 s, 79 ranking rows). EHAM
  stand HG01 is 990 A139 helicopters. 2,193 rows have ADEP == ADES. 2,930 rows are
  `FLIGHT_RULE == 'V'`, all unmatched. Deduplicated ranking total 536 rows; oracle bound on
  predicting all of them perfectly is +0.73 s (`results/review-shorttail.md`).
- `FLIGHT_ID_mvt` is not a per-leg key: of 315,269 departure↔arrival joins, 1,690 disagree on
  aircraft type. `STAND_mvt` is not a unique physical stand: 2,793 pushbacks overlap another
  flight's occupancy of the same code (`results/label-plausibility.md`).
- `LOBT_flt` equals `EOBT_1_flt` on 100.000 % of LTFM departures, every month, both sets
  (`results/timestamps.md` F1).
- Stand vocabulary steps in July 2026: 70 stands used in ≥11 of 12 training months have zero
  departures that month (2.654 % of it, against a 2025 baseline of 0.014–0.756 %); EDDM 103–108
  appear from nothing to 7.84 % of that airport's July departures while 110/111 go to zero
  (`results/recording-change.md` F2).
- Otherwise the recording instrument is stable across 2025 → 2026: schema, null rates, ID
  encodings, closed vocabularies, and the NM join rate are all inside their 2025 bands
  (`results/recording-change.md`).
- Adversarial AUC between 2025 training and the 2026 ranking set is 0.807; reweighting by the
  discriminator did not change verdict ordering (`docs/handoff.md` §5).
- DEP → most-recent-ARR at the same airport and stand covers 81.70 % of training and 81.01 % of
  ranking departures within six hours. Correlation of the label with that arrival's taxi-in is
  0.11 pooled, 0.047 after residualising on airport × month × hour, and ≤0.025 against the model
  residual at every airport. `airport × turnaround-gap decile` removes +0.263 s of matched RMSE
  under leave-one-month-out (`results/arrival-rows.md`, `results/review-refutations.md` C1).
- `FLIGHT_RULE_mvt` is present in the data and is not a feature (Utf8, in neither the categorical
  nor the numeric list).

## 9. Environment

Four cores, 15 GB. `.venv/bin/python`. `POLARS_MAX_THREADS` and LightGBM `num_threads` must be set
low if anything else is running. A forward-fold CV run is ~10–20 min; a full run with ranking
prediction ~40 min; the feature cache rebuilds in ~4 min when `src/prc/features.py` changes
(its sha256 is the cache key — `weather.py`, `airports.py` and `train.py` are NOT in the key).

`make run CONFIG=<yaml>` = CV + ranking prediction + gate, no upload.
`make compare A=<exp_id> B=<exp_id>` = the adopt/reject rule.
`make test`, `make lint`.
