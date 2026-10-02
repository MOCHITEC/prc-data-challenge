# Learned ADS-B residual model on top of hybrid H2 (2026-09-30)

Sibling note to `adsb-ingest-2025-labelled.md`, `adsb-full-ingest-candidate.md`, `adsb-coverage-breakdown.md`.
The only outside fact used is from `public-code-deep-dive-2026-09-30.md`: one competitor shipped a learned ADS-B
residual model. No competitor code was opened. The features, the model and the code are our own
(`scripts/adsb_learn_features.py`, `scripts/adsb_learn_cv.py`, `scripts/adsb_learn_apply.py`).

## 0. Pre-registration (written before any result was looked at)

This section was written and saved before the feature build or any model was run. It is not edited afterwards.
Anything added later goes in the sections below it.

**Rows.**
- Labelled set: all Jan 2025 and Jul 2025 departures, the OOF folds `jan` and `jul` of the H2 runs
  (153,706 + 190,713 = 344,419 rows).
- H2 OOF (`b`): taken by lane. A row is matched when `mvt_minus_aobt3` is non-null in
  `data/work_flseq/feat_train.parquet`. Matched rows take `oof` from `experiments/oof_main_flseq_osm_nolirf-926741d779`.
  Unmatched rows take it from `experiments/oof_main_flseq-8615e6a194`. Both fold files have `pooled` = true for
  the `jan` and `jul` entries.
- Label `y` = `TAXITIME_SEC_mvt` (= MVT − BLOCK). Target `r = y − b`.
- **ADS-B row**: the departure is in a `<tag>_joined.parquet` and has an aircraft (`icao` non-null, `join` ∈ {cs, fn}).
  All other rows get correction 0 but stay in every metric.

**Features** (per departure; label-free; nothing reads BLOCK, `g`, `fill`, `e_aobt3` or `e_to`).
- `ADEP` (categorical), `join` type, the stage-1 `stand_rule` (categorical), and the base prediction `b`.
- Stage-1 per-flight columns of the joined file: `ground_fix`, `n_ground`, `stand_dur`, `stand_nfix`, `dep1_gs`,
  `gap_before_dep1`, `opens_with_landing`, `push` (tail-first share), `post_dwell`, a flag for the gated `t_push`,
  and the stand-exit resolution `t_move − t_move_lo`.
- New raw-fix features, computed here from the aircraft's fixes at ADEP. The anchor is
  `a = coalesce(t_move_lo, t_dep1)` and the end is `e = coalesce(t_to, t_mvt + 120 s)`. Over the ground fixes in
  `[a, e]` they are:
  - times of the first fix with gs ≥ 3, ≥ 10 and ≥ 25 kt, the last ground fix, and the first airborne fix after `a`;
  - fix count, path length, straight displacement, stopped time (gs < 2 kt, steps ≤ 120 s), number of stops,
    maximum inter-fix gap and mean moving gs;
  - x/y in metres (east and north of the ARP) of the anchor fix and of the last ground fix.
  - Also counted: all fixes, and ground fixes, of this aircraft at ADEP within `[t_mvt − 4 h, t_mvt + 30 min]`.
- Every event time T ∈ {`t_first`, `t_dep1`, `t_move_lo`, `t_move`, `t_to`, the three gs thresholds, last ground
  fix, first airborne fix} enters three ways: `t_mvt − T`, `(t_mvt − T) − b`, and `T − t_aobt3` (null in lane B).

**Model.**
- LightGBM regression, L2 objective, trained on the ADS-B rows only.
- Target `r` clipped to ±3,600 s for training.
- Parameters: learning_rate 0.03, num_leaves 63, min_data_in_leaf 100, feature_fraction 0.8, bagging_fraction 0.8
  (freq 1), lambda_l2 10, num_threads 2, seed 20260930. Early stopping is 200 rounds, with a cap of 5,000.
- Correction `c` = the prediction clipped to ±1,800 s. There is no other shrinkage and no guard. Rows that are not
  ADS-B rows get `c = 0`. New prediction = `b + c`.

**CV.**
- Leave-days-out over the 62 labelled days, all 10 airports. Five folds of days, seed 20260930: each month's days
  are permuted and dealt round-robin, so every fold holds 6–7 Jan and 6–7 Jul days.
- In each outer fold, 20 % of the training days (dealt the same way from a second seeded permutation) are the
  early-stopping set. The early-stopped model predicts the held-out days, with no refit.
- **Final model**: all 62 days, with `num_boost_round` = the median best iteration of the 5 folds and no early
  stopping.

**Rule override on the same 2025 rows (the comparator).** This is the v20 rule, recomputed on the same 2025 rows.
- It fires at EDDM, EHAM, LEBL, LSZH and EGLL when `AOBT_3_flt` is non-null, `join` = cs and the gated `t_push` is
  non-null.
- `y_a = round(t_mvt − t_push + bias[ADEP])`. It applies only when `|y_a − b| ≤ 900 s`.
- `bias[ADEP]` = median(`t_push − t_block`) over the eligible rows of the **training days of the same outer fold**
  (leave-days-out).

**Metric.**
- Per row, `d = (y − b)² − (y − b − c)²` (s²). The gain is the mean of `d` over **all 344,419 rows**, i.e. the pooled
  MSE reduction. RMSE before and after are also reported.
- 95 % CI by a day-block bootstrap: 62 days resampled with replacement, B = 2,000, seed 20260930, gain = Σd / Σn.
- Reported for the learned model and for the rule. Also reported per airport (same bootstrap, the airport's own rows).
- **Paired difference** `d_learned − d_rule` per row, with the same day-block bootstrap.

**Decision rule.** ADOPT the learned model iff (1) the 2.5 % bound of its gain CI is > 0 and (2) the 2.5 % bound of
the paired difference (learned − rule) is > 0. Otherwise REJECT (keep v20 = H2 + rule).

**Production file.**
- The final model is applied to the 2026 scored rows, with the same feature code on the 2026 joined and fix files.
- The base is `outputs/hybrid_H2.parquet`, with no rule override.
- Output: `outputs/hybrid_H2_adsb_learned.parquet` = round(base + c). A negative value is reverted to base and
  counted. The file is written whatever the verdict, and it is not submitted.

### 0a. Addendum, still before any model result (coverage fact from the feature build)

The feature build gives these ADS-B rows (joined, with an aircraft): **2025: 286,008** (LTFM **38**) and
**2026: 290,822** (LTFM **5,161**). The model sees almost no LTFM training rows, and the CV cannot measure LTFM.
- **The production file sets c = 0 at LTFM.**
- The CV metric is unchanged: LTFM's 38 rows are scored as pre-registered.

This was fixed before any fold was trained. It is the only change to section 0.

---

## 1. Data checks (after the pre-registration)

- **Labelled rows.** 344,419: `jan` 153,706 and `jul` 190,713. Every row has a label and a base. There are 62 days
  and 339,046 lane-A rows.
- **H2 OOF.** Rebuilt by lane as in section 0. It equals `$SCRATCH/exp/oof_hybrid_H2.parquet` on all 344,419 rows
  (max |diff| 0.0).
- **ADS-B rows.** 286,008 in 2025 and 290,822 in 2026. In both years `t_mvt` in the joined files equals
  `MVT_TIME_UTC_mvt` exactly.
- **2026 inputs.**
  - The 2026 fixes and joined files come from the same `adsb_stage1_feasibility.py` `extract` and `analyse`, with
    the same filter: 10 airports, ≤ 8 km, ground or ≤ 4,000 ft. The join window is also the same:
    MVT in [D 02:00, D+1 00:00).
  - Between the 2026 analysis (2026-09-29) and the 2025 one (2026-09-30), `analyse` did not change. The only code
    change in between is the empty-final-chunk guard in `extract` (commit 416ef93).
  - The 2026 joined files have the same columns, minus the four label-derived ones (`g`, `fill`, `e_aobt3`,
    `e_to`). None of those four is read here.
- **Folds.** Each fold holds 12–14 days, 6–7 from each month. The day lists are in `$SCRATCH/adsb/learn/cv.json`.
- **Best iterations.** 393, 235, 395, 358 and 147, so the final fit uses **358 rounds**.

## 2. CV result (leave-days-out, all 344,419 rows, day-block bootstrap B = 2,000)

`gain` is the pooled MSE reduction in s² per row over all labelled rows. Rows without ADS-B count with d = 0.

| | gain (s²) | CI95 | RMSE (s) | rows changed |
|---|---:|---|---|---:|
| learned residual model | **8,697.9** | [7,691.4, 9,828.8] | 316.97 → **302.94** (−14.03) | 286,008 |
| rule override (v20 rule, bias leave-days-out) | 1,853.8 | [1,681.3, 2,046.1] | 316.97 → 314.03 (−2.94) | 37,482 |
| **paired: learned − rule** | **6,844.2** | **[5,902.7, 7,907.4]** | | |

**Verdict under the pre-registered rule: ADOPT.** Both CI lower bounds are > 0. The gain is 8,258 s² in Jan and
9,052 s² in Jul. The rule gains 2,271 s² in Jan and 1,517 s² in Jul.

- **Rule biases.** The rule's leave-days-out biases are stable across folds: EDDM +49, EHAM −3, LEBL +41–42, LSZH +31
  and EGLL +169–173 s.
- **Test folds.** The learned gain per test fold is 9,715, 8,777, 8,133, 7,818 and 8,899 s².

### Per airport (gain over that airport's rows, s²; day-block CI; RMSE change in s)

| ADEP | n | ADS-B rows | learned | CI95 | rule | CI95 | learned − rule | CI95 | ΔRMSE learned | ΔRMSE rule |
|---|---:|---:|---:|---|---:|---|---:|---|---:|---:|
| EDDF | 36,830 | 35,719 | 13,059 | [11,579, 14,655] | 0 | – | 13,059 | [11,579, 14,655] | −39.67 | 0 |
| EDDM | 27,091 | 26,762 | 13,114 | [9,976, 16,615] | 7,022 | [5,450, 8,690] | 6,091 | [4,353, 8,133] | −42.18 | −21.16 |
| EGLL | 40,210 | 39,862 | 8,447 | [6,160, 11,355] | 1,692 | [1,184, 2,306] | 6,755 | [4,782, 9,221] | −16.56 | −3.23 |
| EHAM | 40,949 | 39,702 | 15,145 | [13,603, 17,013] | 5,772 | [5,287, 6,300] | 9,373 | [8,035, 11,016] | −47.50 | −16.50 |
| LEBL | 28,986 | 28,489 | 11,657 | [10,167, 13,243] | 2,643 | [1,995, 3,304] | 9,013 | [7,846, 10,285] | −28.25 | −6.08 |
| LEMD | 35,328 | 29,667 | 3,383 | [2,492, 4,260] | 0 | – | 3,383 | [2,492, 4,260] | −9.97 | 0 |
| LFPG | 39,612 | 38,675 | 832 | [173, 1,531] | 0 | – | 832 | [173, 1,531] | −0.73 | 0 |
| LIRF | 26,528 | 25,339 | 14,404 | [3,926, 26,906] | 0 | – | 14,404 | [3,926, 26,906] | −12.28 | 0 |
| LSZH | 22,346 | 21,755 | 14,641 | [12,746, 16,859] | 3,009 | [2,603, 3,452] | 11,632 | [9,838, 13,711] | −40.44 | −7.62 |
| LTFM | 46,539 | 38 | 2 | [0, 5] | 0 | – | 2 | [0, 5] | −0.00 | 0 |

- Every airport's learned CI excludes zero, except LTFM, which has 38 rows.
- At each of the five rule airports, the learned model beats the rule and the paired CI excludes zero.
- **Top features by split gain, summed over the folds.**
  - The strongest are the base-relative ADS-B times: `res_t_move_lo` 0.113, `res_t_gs10` 0.092 and
    `res_t_gs3` 0.056.
  - Next come `ADEP` 0.049, `gap_before_dep1` 0.047, `mvt_m_t_gs10` 0.044, `res_t_dep1` 0.042,
    `t_to_m_aobt3` 0.037 and `base` 0.027.
- **CV corrections on ADS-B rows.**
  - Quantiles: p1 −253, p10 −97, p25 −40, p50 −12, p75 +11, p90 +87 and p99 +308 s.
  - The mean |c| is 58 s, and 3 rows hit the ±1,800 s clip.

## 3. Production file

`outputs/hybrid_H2_adsb_learned.parquet` (git-ignored, not submitted).
- **Model.** The final model was fitted on 286,008 ADS-B rows from all 62 days, with 358 rounds. It is saved as
  `$SCRATCH/adsb/learn/final_model.txt`.
- **Base.** The model is applied to `outputs/hybrid_H2.parquet` with no rule override, and c = 0 at LTFM (0a).
- **Validation.**
  - 344,841 rows. The schema `{MVT_ID_mvt: Float64, TAXITIME_SEC_mvt: Int32}` equals the template, and so do the
    ids and their order.
  - 0 nulls and 0 negatives. **2 rows came out negative and were reverted to base.**
  - Rows without ADS-B are identical to H2. LTFM is unchanged.
  - `python -m prc.validate_submission` prints OK, with only the usual > 10,800 s max warning.
- **Changes.**
  - **282,749 rows differ from hybrid_H2**, i.e. 81.99 % of scored rows, out of 290,822 with ADS-B features.
  - Over the changed rows, the RMS change is 119.9 s, the maximum |change| is 1,662 s, and the median |change| is
    46 s (p90 175, p99 429).
  - The signed change quantiles are p1 −299, p10 −119, p25 −57, p50 −12, p75 +27, p90 +118 and p99 +361.
  - The mean prediction goes from 1,039.04 to 1,033.16 s.
- **Against v20 (H2 + rule).**
  - On the 63,544 rows the rule changed, the learned change correlates 0.953 with the rule's change. The median
    learned change is −3 s, and the median rule change is −10 s.
  - RMS(new − v20) over all rows is 92.3 s.

| ADEP | scored | changed | mean Δ | median Δ | median \|Δ\| (changed) | p99 \|Δ\| | max \|Δ\| |
|---|---:|---:|---:|---:|---:|---:|---:|
| EDDF | 36,317 | 34,541 | −14.1 | −18 | 42 | 371 | 1,082 |
| EDDM | 25,966 | 25,186 | −0.6 | −7 | 71 | 544 | 974 |
| EGLL | 39,840 | 38,649 | +1.0 | −7 | 81 | 440 | 1,112 |
| EHAM | 38,182 | 36,070 | −1.1 | 0 | 63 | 512 | 1,394 |
| LEBL | 30,080 | 29,007 | +6.2 | −3 | 52 | 387 | 1,310 |
| LEMD | 36,954 | 34,941 | −26.9 | −22 | 54 | 364 | 1,001 |
| LFPG | 39,872 | 37,745 | −6.1 | −11 | 14 | 272 | 1,446 |
| LIRF | 26,899 | 24,591 | −11.9 | +1 | 15 | 398 | 1,662 |
| LSZH | 23,150 | 22,019 | −5.6 | −18 | 68 | 445 | 1,154 |
| LTFM | 47,581 | 0 | 0 | 0 | – | 0 | 0 |

### Caveat the CV cannot measure: ADS-B coverage differs between 2025 and 2026

Shares below are of ADS-B rows. `ground` = a ground fix before take-off. `stand` = a stand episode (`t_move`).
`push` = the gated `t_push`.

| ADEP | ground 2025 → 2026 | stand 2025 → 2026 | push 2025 → 2026 |
|---|---|---|---|
| EDDF | 0.99 → 0.94 | 0.47 → **0.28** | 0.09 → 0.04 |
| EDDM | **0.64 → 1.00** | 0.53 → 0.88 | 0.36 → 0.63 |
| EGLL | **0.61 → 0.98** | 0.19 → 0.71 | 0.06 → 0.44 |
| EHAM | 1.00 → 1.00 | 0.82 → 0.85 | 0.38 → 0.45 |
| LEBL | 0.99 → 0.99 | 0.52 → 0.65 | 0.24 → 0.33 |
| LEMD | **0.31 → 0.93** | 0.12 → 0.52 | 0.04 → 0.16 |
| LFPG | 0.18 → 0.58 | 0.00 → 0.03 | 0.00 → 0.00 |
| LIRF | 0.58 → **0.39** | 0.34 → 0.14 | 0.01 → 0.03 |
| LSZH | 0.98 → 0.99 | 0.72 → 0.72 | 0.18 → 0.26 |

- **The 2026 receiver coverage is much denser** at EDDM, EGLL, LEMD and LFPG, and thinner at EDDF (stand) and
  LIRF. The model was trained on the 2025 mix.
- **The correction spread moves with coverage.** 2025 CV vs 2026 production, p10/p50/p90 in s:
  - EGLL: −91/−15/+68 in the CV, −165/−10/+179 in 2026.
  - LEMD: −45/−10/+2 in the CV, −137/−26/+75 in 2026.
  - Elsewhere the spread is similar.
- **What the CV does and does not show.** It shows that the model extracts the signal when the coverage is there.
  It does not measure the 2026 coverage mix. The per-flight features (stand exit, speed thresholds) mean the same
  thing in both years, so denser coverage should help rather than hurt, but this is unverified.
- **Other caveats.**
  - The residual model is fitted on fold-OOF bases and applied to the full-fit H2 base. The base error
    distributions differ slightly.
  - The five rule airports were chosen earlier on 4 labelled days, 3 of which (01-22, 07-13, 07-21) are in this
    2025 set. The comparator is therefore, if anything, favoured.

## 4. Reproduce

```
export POLARS_MAX_THREADS=2 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
.venv/bin/python scripts/adsb_learn_features.py build && .venv/bin/python scripts/adsb_learn_features.py concat
.venv/bin/python scripts/adsb_learn_cv.py      # log: $SCRATCH/adsb/learn_cv.log
.venv/bin/python scripts/adsb_learn_apply.py   # log: $SCRATCH/adsb/learn_apply.log
```

- The feature build takes about 1 s per day.
- Peak RSS is below 1 GB for the feature build and below 1.5 GB for the CV.
- `num_threads` = 2.

Data: adsb.lol, Open Database License (ODbL) 1.0. "Contains data from adsb.lol, made available under the ODbL 1.0."

## 5. Re-run on the `hybrid_H2_plainy` base (2026-09-30, after v21 scored 254.7657)

**What changed and what did not.** The design is unchanged from section 0/0a: same features, hyperparameters and
clipping, the same 5 day-folds and B = 2,000, and the same LTFM = 0 rule in production. Only the base is swapped.
- `b` = the plainy OOF, following `plainy-blend.md`:
  - matched lane: `0.5 × oof_main_flseq_osm_nolirf-926741d779 + 0.5 × oof_main_flseq_osm_nolirf_plainy-b971f4b81b`;
  - unmatched lane: `oof_main_flseq-8615e6a194`.
- It equals `$SCRATCH/exp/oof_hybrid_H2_plainy.parquet` on all 344,419 rows (max |diff| 0.0).
- The plain run's jan/jul folds equal H2's.
- Target `r = y − b`.
- Commands: `adsb_learn_cv.py --base plainy` and `adsb_learn_apply.py --base plainy`. Logs:
  `$SCRATCH/adsb/learn_cv_plainy.log` and `learn_apply_plainy.log`.

**CV (344,419 rows, day-block bootstrap).**

| | gain (s²) | CI95 | RMSE (s) |
|---|---:|---|---|
| learned on the plainy base | **8,608.9** | [7,590.5, 9,711.5] | 316.334 → **302.421** (−13.913) |
| rule on the plainy base | 1,848.5 | [1,673.4, 2,047.1] | 316.334 → 313.399 (−2.935) |
| paired: learned − rule | 6,760.4 | [5,803.0, 7,814.8] | |

- The pre-registered rule gives **ADOPT** again.
- Best iterations were 479, 249, 377, 397 and 113, so the final fit uses **377 rounds**.
- Per-airport gains are within a few % of section 2. Every airport's learned CI excludes 0 except LTFM.

**Comparability on the same 2025 rows (paired, day-block bootstrap).**

| | pooled RMSE (s) |
|---|---:|
| H2 | 316.966 |
| plainy | 316.334 |
| H2 + learned (section 2) | 302.935 |
| **plainy + learned** | **302.421** |

**(plainy + learned) − (H2 + learned) = −0.513 s, CI95 [−0.852, −0.184].** As MSE: 310.7 s² [109.6, 483.4] in
favour of plainy + learned. Plainy's base advantage on these rows is −0.63 s. About 0.51 s of it survives the
residual model.

**Production file** `outputs/hybrid_H2_plainy_adsb_learned.parquet`. It is the final fit (286,008 rows, 62 days,
377 rounds) applied to `outputs/hybrid_H2_plainy.parquet`, with c = 0 at LTFM. It is git-ignored and not
submitted.
- **Validation.**
  - 344,841 rows. The schema, ids and order equal the template.
  - 0 nulls and 0 negatives. **2 negatives were reverted to base.**
  - Rows without ADS-B equal the base. LTFM is unchanged.
  - `prc.validate_submission` prints OK, with only the usual > 10,800 s warning.
- **Changes against its base.**
  - 282,627 rows (81.96 %) are changed.
  - Over the changed rows, the RMS change is 119.1 s, with median |Δ| 46 s, p90 175, p99 429 and max 1,614.
  - The signed change quantiles are p1 −296, p10 −117, p50 −12, p90 +120 and p99 +365.
  - The mean prediction goes from 1,039.02 to 1,033.35 s.
  - The per-airport spread matches section 3, including the wider 2026 corrections at EGLL and LEMD.
- **Against v21 (`hybrid_H2_adsb_learned`).** 330,283 rows differ, with RMS 29.6 s and max 1,902 s.
- In the committed script, the log line "changed vs hybrid_H2" was relabelled "changed vs base". The count is
  against `hybrid_H2_plainy`.

## 6. Re-run on the `hybrid_plainy_cb` base (2026-09-30, after v22 scored 254.2739)

**Setup.** The design is unchanged from section 0/0a. Only the base is swapped.
- `b` follows `catboost-blend.md`:
  - matched lane: `(oof_main_flseq_osm_nolirf-926741d779 + oof_main_flseq_osm_nolirf_plainy-b971f4b81b +
    oof_main_flseq_osm_nolirf_cb-6297d62f5f) / 3`;
  - unmatched lane: `oof_main_flseq-8615e6a194`.
- It equals `$SCRATCH/cbx/oof_hybrid_plainy_cb.parquet` on all 344,419 rows (max |diff| 0.0).
  - `$SCRATCH/dry/oof_hybrid_plainy_cb.parquet` differs from that file (max 1,726 s) and was not used.
  - The CB run's jan/jul folds equal H2's.
- Commands: `adsb_learn_cv.py --base cb` and `adsb_learn_apply.py --base cb`. Logs:
  `$SCRATCH/adsb/learn_cv_cb.log` and `learn_apply_cb.log`.
- **Restart checks.**
  - After the container restart, the memory cgroup path was re-read from `/proc/self/cgroup` (it is now
    `/process_api/01a0f24a-…/claude-code-bash`). The root `total_rss` was about 220 MB during the runs.
  - `feat_2025.parquet`, `feat_2026.parquet` and `cv_rows_plainy.parquet` were still in the scratchpad.

**CV (344,419 rows, day-block bootstrap B = 2,000).**

| | gain (s²) | CI95 | RMSE (s) |
|---|---:|---|---|
| learned on the cb base | **8,503.7** | [7,492.2, 9,631.2] | 314.771 → **300.961** (−13.811) |
| rule on the cb base | 1,807.9 | [1,632.7, 2,003.2] | 314.771 → 311.886 (−2.885) |
| paired: learned − rule | 6,695.7 | [5,747.4, 7,786.3] | |

- The pre-registered rule gives **ADOPT**.
- Best iterations were 415, 245, 432, 401 and 135, so the final fit uses **401 rounds**.
- Every airport's learned CI excludes 0 except LTFM. The per-airport gains are within a few % of section 5.

**Comparability on the same 2025 rows (paired, day-block bootstrap).**

| | pooled RMSE (s) |
|---|---:|
| plainy | 316.334 |
| cb | 314.771 |
| plainy + learned (section 5) | 302.421 |
| **cb + learned** | **300.961** |

**(cb + learned) − (plainy + learned) = −1.461 s, CI95 [−1.974, −1.030].** As MSE: 881.3 s² [705.3, 1,074.0].
cb's base advantage on these rows is −1.56 s, and about 1.46 s of it survives the residual model.

**Production file** `outputs/hybrid_plainy_cb_adsb_learned.parquet`. It is the final fit (286,008 rows, 62 days,
401 rounds) applied to `outputs/hybrid_plainy_cb.parquet`, with c = 0 at LTFM. It is git-ignored and not
submitted.
- **Validation.**
  - 344,841 rows. The schema, ids and order equal the template.
  - 0 nulls and 0 negatives. **2 negatives were reverted to base.**
  - Rows without ADS-B equal the base. LTFM is unchanged.
  - `prc.validate_submission` prints OK, with only the usual > 10,800 s warning.
- **Changes against its base.**
  - 283,203 rows (82.13 %) are changed.
  - Over the changed rows, the RMS change is 118.6 s, with median |Δ| 45 s, p90 173, p99 429 and max 1,744.
  - The signed change quantiles are p1 −300, p10 −116, p50 −12, p90 +118 and p99 +363.
  - The mean prediction goes from 1,039.29 to 1,033.91 s.
- **Against v22 (`hybrid_H2_plainy_adsb_learned`).** 331,137 rows differ, with RMS 29.5 s and max 946 s.

## 7. Re-run on the `hybrid_blend_l2` base (2026-10-01, after v23 scored 252.4287)

**Setup.** The design is unchanged from section 0/0a. Only the base is swapped.
- `b` follows `blend-l2.md` (P):
  - matched lane: `(oof_main_flseq_osm_nolirf_l2-ef204b44ff + oof_main_flseq_osm_nolirf_plainy_l2-c74a27239b +
    oof_main_flseq_osm_nolirf_cb_l2-4aa341b206) / 3`;
  - unmatched lane: `oof_main_flseq-8615e6a194`.
- It equals `$SCRATCH/l2exp/oof_hybrid_blend_l2_P.parquet` on all 344,419 rows (max |diff| 0.0).
- All three members' jan/jul folds equal H2's.
- Commands: `adsb_learn_cv.py --base l2` and `adsb_learn_apply.py --base l2`. Logs:
  `$SCRATCH/adsb/learn_cv_l2.log` and `learn_apply_l2.log`.
- **Restart checks.**
  - The memory cgroup path was re-read from `/proc/self/cgroup` (`/process_api/01a0f477-…/claude-code-bash`). The
    root `total_rss` was about 260–310 MB.
  - `feat_2025.parquet`, `feat_2026.parquet` and `cv_rows_cb.parquet` were present.

**CV (344,419 rows, day-block bootstrap B = 2,000).**

| | gain (s²) | CI95 | RMSE (s) |
|---|---:|---|---|
| learned on the blend_l2 base | **8,377.5** | [7,372.2, 9,451.3] | 314.470 → **300.855** (−13.615) |
| rule on the blend_l2 base | 1,798.7 | [1,635.4, 1,987.0] | 314.470 → 311.597 (−2.873) |
| paired: learned − rule | 6,578.8 | [5,644.7, 7,611.1] | |

- The pre-registered rule gives **ADOPT**.
- Best iterations were 495, 249, 462, 334 and 152, so the final fit uses **334 rounds**.
- Every airport's learned CI excludes 0 except LTFM.

**Comparability on the same 2025 rows (paired, day-block bootstrap).**

| | pooled RMSE (s) |
|---|---:|
| cb | 314.771 |
| blend_l2 | 314.470 |
| cb + learned (section 6) | 300.961 |
| blend_l2 + learned | 300.855 |

**(blend_l2 + learned) − (cb + learned) = −0.105 s, CI95 [−0.356, +0.127].** As MSE: 63.4 s² [−74.2, 200.3].
- The CI **includes zero**. Of blend_l2's base advantage on these rows (−0.30 s), about a third survives the
  residual model, and the difference is not resolved.
- The rows are Jan+Jul 2025 only, which is not the 672,428-row pooled set of `blend-l2.md`.

**Production file** `outputs/hybrid_blend_l2_adsb_learned.parquet`. It is the final fit (286,008 rows, 62 days,
334 rounds) applied to `outputs/hybrid_blend_l2.parquet`, with c = 0 at LTFM. It is git-ignored and not submitted.
- **Validation.**
  - 344,841 rows. The schema, ids and order equal the template.
  - 0 nulls and 0 negatives. **2 negatives were reverted to base.**
  - Rows without ADS-B equal the base. LTFM is unchanged.
  - `prc.validate_submission` prints OK, with only the usual > 10,800 s warning.
- **Changes against its base.**
  - 282,167 rows (81.83 %) are changed.
  - Over the changed rows, the RMS change is 116.4 s, with median |Δ| 45 s, p90 171, p99 419 and max 1,704.
  - The signed change quantiles are p1 −290, p10 −115, p50 −12, p90 +119 and p99 +358.
  - The mean prediction goes from 1,039.47 to 1,034.25 s.
- **Against v23 (`hybrid_plainy_cb_adsb_learned`).** 330,422 rows differ, with RMS 26.1 s and max 792 s.

## 8. Re-run on the `hybrid_blend_l2_pa` base (2026-10-02, after v28 scored 252.4111)

**Setup.** The design is unchanged from section 0/0a. Only the base is swapped.
- `b` follows `per-airport-member.md` (primary), with the same expression as `scripts/blend_pa.py`:
  - matched lane: `((0.5·a2 + 0.5·pa) + p2 + c2) / 3`, where
    - a2 = `oof_main_flseq_osm_nolirf_l2-ef204b44ff`,
    - pa = `oof_main_flseq_osm_nolirf_l2_pa-85f815edf4`,
    - p2 = `…plainy_l2-c74a27239b`,
    - c2 = `…cb_l2-4aa341b206`;
  - unmatched lane: `oof_main_flseq-8615e6a194`.
- It equals `logs/per_airport/scratch/oof_hybrid_blend_l2_pa.parquet` on all 344,419 rows (max |diff| 0.0).
- All four members' jan/jul folds equal H2's.
- Commands: `adsb_learn_cv.py --base pa` and `adsb_learn_apply.py --base pa`. Logs:
  `$SCRATCH/adsb/learn_cv_pa.log` and `learn_apply_pa.log`.
- **Restart checks.**
  - The memory cgroup path was re-read from `/proc/self/cgroup` (`/process_api/01a0fa39-…/claude-code-bash`). The
    root `total_rss` was about 250 MB.
  - `feat_2025.parquet`, `feat_2026.parquet` and `cv_rows_l2.parquet` were present.

**CV (344,419 rows, day-block bootstrap B = 2,000).**

| | gain (s²) | CI95 | RMSE (s) |
|---|---:|---|---|
| learned on the pa base | **8,265.3** | [7,232.0, 9,384.1] | 313.767 → **300.307** (−13.460) |
| rule on the pa base | 1,744.3 | [1,585.1, 1,926.5] | 313.767 → 310.975 (−2.792) |
| paired: learned − rule | 6,521.0 | [5,541.1, 7,608.1] | |

- The pre-registered rule gives **ADOPT**.
- Best iterations were 522, 293, 332, 333 and 150, so the final fit uses **332 rounds**.
- Every airport's learned CI excludes 0 except LTFM.

**Comparability on the same 2025 rows (paired, day-block bootstrap).**

| | pooled RMSE (s) |
|---|---:|
| blend_l2 | 314.470 |
| pa | 313.767 |
| blend_l2 + learned (section 7) | 300.855 |
| **pa + learned** | **300.307** |

**(pa + learned) − (blend_l2 + learned) = −0.548 s, CI95 [−0.800, −0.348].** As MSE: 329.5 s² [219.5, 439.3].
pa's base advantage on these rows is −0.70 s, and about 0.55 s of it survives the residual model.

**Production file** `outputs/hybrid_blend_l2_pa_adsb_learned.parquet`. It is the final fit (286,008 rows, 62 days,
332 rounds) applied to `outputs/hybrid_blend_l2_pa.parquet`, with c = 0 at LTFM. It is git-ignored and not
submitted.
- **Validation.**
  - 344,841 rows. The schema, ids and order equal the template.
  - 0 nulls and 0 negatives. **2 negatives were reverted to base.**
  - Rows without ADS-B equal the base. LTFM is unchanged.
  - `prc.validate_submission` prints OK, with only the usual > 10,800 s warning.
- **Changes against its base.**
  - 282,938 rows (82.05 %) are changed.
  - Over the changed rows, the RMS change is 115.1 s, with median |Δ| 44 s, p90 170, p99 416 and max 1,754.
  - The signed change quantiles are p1 −288, p10 −114, p50 −11, p90 +117 and p99 +353.
  - The mean prediction goes from 1,039.36 to 1,034.18 s.
- **Against `hybrid_blend_l2_adsb_learned`.** 324,921 rows differ, with RMS 16.4 s and max 509 s.

## 9. Improving the learned model: variants V1–V3 on the `hybrid_blend_l2_pa` base (2026-10-02)

Context: v32 (= section 8) scored 251.7333 on the board. Public fact (`public-code-deep-dive-2026-09-30.md` §1, §2):
merry-quicksand's ADS-B stage is a LightGBM + MLP stacker with 26 features, and its "joined mid-taxi" rows
(coverage code 2) carry 41 % of its ADS-B fold gain. No competitor code was opened.

### 9.0 How rows first seen mid-taxi are handled today (check; criterion fixed before computing it)

Coverage classes of the 344,419 labelled rows, from the joined/feature files:
- **K0**: no ADS-B feature row.
- **K1**: joined, but no ground fix before take-off.
- **K2 (mid-taxi first)**: there is a ground fix, but no stand episode was found (`t_move` null).
- **K3**: a stand episode was observed (`t_move` non-null).

The criterion was written before any number was computed. **K2 counts as under-served if its relative MSE
reduction under the current model (section 8 CV rows, `1 − MSE_learned / MSE_base`) is below half of K3's, and K2
holds ≥ 5 % of rows.**

**Check result** (`scripts/adsb_learn_v2_check.py`, log `$SCRATCH/adsb/learn_v2_check.log`; section 8 CV rows):

| class | rows | share | MSE base | MSE learned | rel. reduction | share of gain |
|---|---:|---:|---:|---:|---:|---:|
| K0 no ADS-B | 58,411 | 0.170 | 93,829 | 93,829 | 0 | 0 |
| K1 air only | 89,202 | 0.259 | 169,691 | 169,081 | 0.004 | 0.019 |
| **K2 mid-taxi first** | 82,786 | 0.240 | 75,095 | 63,063 | **0.160** | **0.350** |
| K3 stand seen | 114,020 | 0.331 | 62,039 | 46,286 | 0.254 | 0.631 |

**How K2 rows are handled today.** They are **not excluded**: every row with a joined aircraft gets the full feature
vector.
- With no stand run, the anchor falls back to `t_dep1`, the first departure fix after the last ≥ 10 min silence.
- The post-anchor features are then computed from the first fix seen: gs thresholds, path, stops, last ground fix
  and take-off. 80,204 of the 82,786 K2 rows have them.
- The model also sees `stand_rule`, `dep1_gs` (the speed at the first fix) and the first-fix position `x_a`/`y_a`.
- Stage-1 reasons within K2: `moving_when_first_seen` 56,701, no qualifying run 13,303, `run_at_dep1` whose run
  ends < 60 s before the phase end 12,739, `longest_run` 43.
- The first-fix gs median is 13.8 kt (p10 3.8, p90 30.5).
- K2 already carries 35 % of the gain, close to the 41 % merry-quicksand reports for its code 2.

**Criterion: K2 is not under-served.** Its relative reduction is 0.160, against half of K3's (0.127). **V3 is
therefore not run.** Other observations:
- By airport, K2 lags K3 most at EGLL (0.14 vs 0.36), LEMD (0.19 vs 0.52) and LSZH (0.24 vs 0.54).
- K1 (air only, 26 % of rows) gets almost nothing (0.004). It is not part of this task.

### 9.1 Pre-registration of V1/V2 (written and committed before any variant is run)

The base is the `hybrid_blend_l2_pa` OOF, as in section 8. The features (58), target, rows, 5 day-folds, the
inner 20 % early-stopping days per fold, the ±1,800 s correction clip, "no ADS-B → c = 0" and, in production, LTFM
c = 0 are all exactly as in section 0. **Reference = the current learned LGB** (section 8 CV rows,
`cv_rows_pa.parquet`; in production `final_model_pa.txt`).

**V1: MLP regressor** (sklearn 1.9.0 `MLPRegressor`; torch is not installed and nothing is installed).
- **Inputs.**
  - The 55 numeric features. Each is clipped to the 0.5–99.5 % quantiles of its fit rows, standardised with the fit
    rows' mean and sd, and NaN is set to 0. A 0/1 missing indicator is added for every feature that has a NaN in the
    fit rows.
  - The 3 categoricals (ADEP, join, stand_rule) are one-hot encoded, with an extra level for missing.
- **Target.** `r` clipped to ±3,600 s, divided by 600.
- **Network.** Hidden layers (128, 64), ReLU, Adam, learning_rate_init 1e-3, alpha (L2) 1e-4, batch 512.
- **Training.** Epochs via `partial_fit`, with the rows shuffled every epoch. At most 60 epochs, with early stopping
  on the inner early-stopping days' MSE (patience 8). The best epoch's weights are kept.
- **Seeds.** 3 (20260930 + 0/1/2), and the prediction is the mean of the 3. Prediction × 600, then clipped to
  ±1,800 s.
- **Final fit.** All 62 days, for each seed the median best epoch over the 5 folds, with no early stopping. The
  scaling is fitted on all rows.
- Threads: `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` = 2.

**V2.** `c = 0.5 × c_LGB + 0.5 × c_V1`, using the same folds' predictions. In production it is 0.5 × the current
final LGB plus 0.5 × the V1 final fit.

**V3.** Not run (check above).

**Metric.**
- Per row `d = (y − b − c_ref)² − (y − b − c_var)²`, the mean over all 344,419 rows. A positive value means the
  variant is better.
- 95 % CI from a day-block bootstrap (62 days, B = 2,000, seed 20260930).
- Also reported: RMSE, and per airport (information only).

**Decision.**
- A variant **passes** if the 2.5 % bound of its paired CI is > 0.
- If any variant passes, **adopt the single best** by point estimate among those that pass, and build
  `outputs/hybrid_blend_l2_pa_adsb_v2.parquet`: the final fit applied to `outputs/hybrid_blend_l2_pa.parquet`, with
  c = 0 at LTFM and negatives reverted to base. It is validated as before and not submitted.
- If no variant passes, keep the current model and build no file.
