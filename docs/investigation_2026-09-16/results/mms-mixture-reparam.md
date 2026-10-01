# Result — item 9 `mms-mixture-reparam` (2026-09-17)

Script: `docs/investigation_2026-09-16/scripts/mms-mixture-reparam.py` (model-B fits only, `POLARS_MAX_THREADS=1`,
LightGBM `num_threads: 2`, one fit at a time, runtime ≈ 6 min). Inputs: `data/work/feat_train.parquet`,
`configs/main.yaml`, `experiments/oof_main-48c19bc830.parquet`. Nothing under `src/` or `configs/` was modified;
the script imports `TrainConfig`, `fit_encodings`, `attach_encodings`, `visible_numeric_columns`, `_matrix`,
`_inner_holdout_mask` from `prc.train` and reproduces `predict_fold`'s model-B branch. Every number below was
produced by this run. Sibling reports used as inputs, not re-measured: item 1 (`sched-filled-block`) for the
identity `y − ms = SCHED − BLOCK` and the LIRF fill mechanism; item 3 (`lirf-regime-tail-count`) and item 8
(`cond-mean-vs-median`) for the LIRF bin composition; item 7 (`false-alarm-non-lirf`) for the non-LIRF
false-alarm rows; item 13 (`tail-postprocessing-cap-shrink`) for the baseline unmatched RMSE 2,296.1.

Definitions (plan §0): `y` = `TAXITIME_SEC_mvt`; `ms` = `mms` = feature `mvt_minus_sched`; **unmatched** =
routing definition (`mvt_minus_aobt3` null → model B), 22,470 training rows; **f10** = |BLOCK − SCHED| ≤ 10 s
(D's "fill"), which by item 1's identity equals |y − ms| ≤ 10 s — verified again here on all 22,470 rows
(1,124 rows under both spellings, agreeing row for row). Bins: 2 h = 7,200 s, 4 h = 14,400, 12 h = 43,200.

## 1. Hypothesis as stated (plan.md §1 item 9)

Model B should predict P(sched-filled | x)·mms + (1 − P)·taxi — equivalently, model the target y − mms on LIRF
rows, or a gate hold G on top of MVT − SCHED; a two-variable lookup already beats the GBDT on unmatched rows.

Sub-claims tested here (the plan lists them as **unverified**: "no trained version exists"):
(a) E H2's pre-check corr(TAXITIME, MVT − SCHED) on LIRF unmatched vs LFPG unmatched;
(b) variant A — model B with target y − `mvt_minus_sched` on LIRF unmatched rows, current target elsewhere;
(c) variant B — a P(sched-filled | x) classifier on the unmatched rows, prediction = P·ms + (1 − P)·regression-B.
The plan's already-verified no-training figures (LOMO lookup tables, oracle replacements) are not re-derived.

## 2. Procedure actually executed

**Resolved parameters.** `configs/main.yaml` was read exactly as `src/prc/pipeline.py::main` does (only keys that
are `TrainConfig` dataclass fields; `params` merged over the defaults), then `.for_unmatched()`:

| field | value |
|---|---|
| model-B `params` | objective regression, learning_rate 0.05, **num_leaves 127**, **min_data_in_leaf 50**, feature_fraction 0.8, bagging_fraction 0.8, bagging_freq 1, lambda_l2 1.0, verbose −1, **num_threads 2** (the OOF run used 4), deterministic true, force_row_wise true, seed 42 |
| `num_rounds` (model B) | 2,000 · `early_stopping` 100 · `inner_holdout_months` 1 · `n_seeds` 3 · `use_categoricals` true · `use_encodings` true · `reference_min_count` 30 · `use_linear_tree` false · `exclude_features` () |

**Data path per fold** (identical to `predict_fold`'s model-B branch): `fit_encodings` on the labelled training
side of the fold (all 2.08 M rows, matched and unmatched, as the code does) → `attach_encodings` onto the
unmatched training and validation frames → `visible_numeric_columns` → `cols_b` = numerics not prefixed by
`mvt_minus_aobt3` / `aobt3` / `lobt_minus` / `mvt_minus_lobt` / `mvt_minus_eobt1` / `mvt_minus_iobt`
(**84 numeric + 10 categorical = 94 columns**; `mvt_minus_sched` and `mvt_minus_sched_slog` are both present,
asserted in the script) → `_matrix` on the stacked [train, valid] frame → `_inner_holdout_mask` (latest 1 month
of the training side) → early stop → rescale rounds to the refit size → refit under 3 seeds and average →
`np.clip(pred, 0, None)` as `_assemble` does.

**Folds**: `jan`, `jul`, `forward` from `prc.cv.month_block_folds` — the three whose predictions make up
`experiments/oof_main-48c19bc830.parquet` (`jul_fwd` shares `jul`'s validation month and is coalesced away by
`run_cv`). Validation rows: jan 1,435 + jul 3,938 + forward 3,005 = **8,378**, joined back to the OOF file with
zero nulls, i.e. exactly the row set on which the plan's 2,296.1 is defined.

**Variants fitted** (five fits per fold, 15 in total, never two at once):

| id | what was fitted |
|---|---|
| base(repro) | model B as the code fits it (target `y`) |
| A | target `y − ms` on LIRF unmatched rows, `y` elsewhere; at prediction `ms` is added back on LIRF rows only |
| A2 (extra) | target `y − ms` on **every** unmatched row (E H2's T = D − G form) |
| B | LightGBM `objective: binary` on the label **f10**, same `cols_b` + categoricals; prediction = P·ms + (1 − P)·base |
| B2 (extra) | a second L2 regression fitted on the **non-filled** training rows only, mixed as P·ms + (1 − P)·that |

plus post-hoc combinations that needed no extra fit: `B-hard` (ms where P ≥ 0.5, else base), `B-LIRF-only` /
`B2-LIRF-only` (the mixture applied only to LIRF rows), and two **oracle** rows that substitute the true f10 flag
for P. Evaluation: unmatched OOF RMSE per fold, pooled, per airport, LIRF vs non-LIRF, and with the worst 20 /
200 rows removed — both by `prc.compare.robustness_curve` (the project convention: the K rows where *either*
model errs most, with a 500-sample paired bootstrap) and by each variant's own worst K.

## 3. Results

### 3.1 (a) Pre-check, no fit: corr(y, ms) on unmatched training rows

| group | n | Pearson | Spearman | OLS slope | y median | ms median | f10 rate |
|---|---|---|---|---|---|---|---|
| **LIRF unmatched** | 1,488 | **0.8964** | 0.5233 | 1.0619 | 3,922 | 6,657 | 0.4845 |
| **LFPG unmatched** | 3,773 | **0.0205** | 0.2206 | 0.0074 | 1,023 | 5,823 | 0.0034 |
| LIRF unm, ms ≥ 2 h | 627 | 0.9212 | 0.6052 | 1.1511 | 8,518 | 9,962 | 0.6316 |
| LIRF unm, ms < 2 h | 861 | 0.0682 | 0.2651 | 0.0759 | 1,321 | 5,526 | 0.3775 |
| LIRF unm, f10 | 721 | 1.0000 | 1.0000 | 1.00000 | 7,559 | 7,559 | 1 |
| LIRF unm, ¬f10 | 767 | 0.8189 | 0.0685 | 1.0592 | 1,070 | 5,995 | 0 |
| LFPG unm, ms ≥ 2 h | 990 | −0.0365 | 0.0610 | −0.0048 | 1,088.5 | 8,940 | 0.0000 |
| LFPG unm, f10 | 13 | 0.99999 | 1.0000 | 0.9997 | 1,388 | 1,384 | 1 |
| LIRF **matched** (contrast) | 159,216 | 0.2985 | 0.4134 | — | — | — | — |

All ten airports, unmatched, subset "all" (Pearson): LIRF 0.8964, LTFM 0.4425, EHAM 0.3421, LEBL 0.1964,
EDDF 0.1738, LEMD 0.1626, EDDM 0.1384, EGLL 0.0827, LSZH 0.0742, LFPG 0.0205.

The LIRF ¬f10 Pearson of 0.8189 on a subset whose Spearman is 0.0685 is carried by a handful of extreme rows
(the rollover and long-ms rows); the rank correlation shows no monotone relation there.

### 3.2 Baseline reproduction (the number the plan's 2,296.1 must be compared against)

| fold | n | `oof_main-48c19bc830` | this run | difference | rounds (file → this run) |
|---|---|---|---|---|---|
| jan | 1,435 | 3,223.0 | **3,247.5** | +24.5 | 393 → 566 (best 522, holdout 1,649 rows) |
| jul | 3,938 | 2,008.8 | **2,013.6** | +4.8 | 479 → 288 (best 262, holdout 1,649) |
| forward | 3,005 | 2,109.3 | **2,177.4** | +68.1 | 45 → 45 (best 41, holdout 1,913) |
| **pooled** | **8,378** | **2,296.1** | **2,326.5** | **+30.4 (+1.3 %)** | |

Row-level agreement: RMS(this run − OOF file) = 213.8 s, max |diff| 3,841.9 s, correlation 0.9967; 8,145 of the
8,378 rows differ by more than 1 s. Paired bootstrap of the two baselines: the OOF file is better by 30.5 s
[−56.6, −2.2] on all rows, but by +2.3 [−13.3, +19.6] once the worst 20 rows are removed and +1.8 [−5.7, +8.5]
at 200 — i.e. the whole difference is a handful of tail rows.

The two runs are not bit-identical because `num_threads` was 4 in the logged run and 2 here; `deterministic` and
`force_row_wise` fix the histogram construction mode but not the number of threads the reduction is summed over.
The visible consequence is the early-stopping round count (jan 393 → 566, jul 479 → 288, forward 45 → 45 —
unchanged on the fold with the shortest fit). **All variant comparisons below are against this run's own
baseline of 2,326.5**, which shares thread count, seeds and code path with them.

### 3.3 Unmatched OOF RMSE, all variants

| variant | jan | jul | forward | **pooled** | LIRF (537) | non-LIRF (7,841) | LIRF ms ≥ 2 h (239) | own-20 | own-200 |
|---|---|---|---|---|---|---|---|---|---|
| `oof_main-48c19bc830` | 3,223.0 | 2,008.8 | 2,109.3 | 2,296.1 | 7,179.3 | 1,450.2 | 10,463.2 | 1,173.8 | 608.3 |
| **base(repro)** | 3,247.5 | 2,013.6 | 2,177.4 | **2,326.5** | 7,326.3 | 1,451.7 | 10,697.9 | 1,172.4 | 605.9 |
| **A: y−ms @ LIRF** | 2,904.6 | 1,338.6 | 1,024.6 | **1,632.1** | 4,214.3 | 1,276.7 | 5,845.2 | 909.3 | 534.6 |
| A2: y−ms @ all | 3,204.9 | 2,253.0 | 2,952.6 | 2,696.7 | 6,641.3 | 2,179.4 | 9,691.9 | 1,258.7 | 637.3 |
| **B: P·ms+(1−P)·base** | 2,964.0 | 1,714.6 | 1,515.0 | **1,926.1** | 5,186.9 | 1,456.5 | 7,362.4 | 1,111.6 | 557.9 |
| B-hard: 1[P ≥ .5] | 2,936.1 | 1,904.5 | 1,363.6 | 1,961.7 | 5,410.0 | 1,451.7 | 7,696.3 | 1,107.3 | 523.3 |
| B-LIRF-only | 2,963.5 | 1,714.4 | 1,503.8 | 1,922.7 | 5,186.9 | 1,451.7 | 7,362.4 | 1,110.4 | 557.7 |
| B2: P·ms+(1−P)·nofill | 2,905.1 | 1,553.6 | 1,431.1 | 1,820.6 | 4,860.2 | 1,387.1 | 6,887.8 | 1,021.6 | 569.7 |
| B2-LIRF-only | 2,907.3 | 1,514.7 | 1,636.4 | 1,867.2 | 4,860.2 | 1,451.7 | 6,887.8 | 1,096.2 | 550.5 |
| oracle f10·ms+(1−·)·base | 3,151.9 | 1,863.4 | 2,059.1 | 2,203.3 | 6,706.7 | 1,451.4 | 9,846.1 | 962.4 | 469.8 |
| oracle f10·ms+(1−·)·nofill | 2,822.8 | 1,494.0 | 2,497.5 | 2,156.6 | 6,677.4 | 1,384.2 | 9,861.0 | 764.2 | 420.1 |
| non-filled regression alone | 3,219.2 | 2,081.3 | 2,982.9 | 2,646.2 | 9,014.0 | 1,384.7 | 13,096.2 | 1,488.0 | 727.3 |

"own-K" = the variant's own worst K rows removed. Variant A improves every fold, both airport groups, and both
tail-trimmed columns; it is the best of the fitted variants on every one of those cuts.

### 3.4 Per airport, pooled RMSE

| variant | EDDF | EDDM | EGLL | EHAM | LEBL | LEMD | LFPG | **LIRF** | LSZH | LTFM |
|---|---|---|---|---|---|---|---|---|---|---|
| oof file | 1,073.4 | 999.7 | 1,099.0 | 926.8 | 1,044.1 | 418.3 | 2,728.8 | 7,179.3 | 802.8 | 508.4 |
| base(repro) | 1,064.5 | 1,017.2 | 1,106.5 | 954.7 | 1,043.4 | 362.5 | 2,732.8 | 7,326.3 | 757.1 | 501.0 |
| **A** | **409.2** | 725.8 | 1,004.1 | **477.2** | **371.4** | 291.3 | 2,623.0 | **4,214.3** | 601.3 | 498.2 |
| A2 | 1,916.6 | 811.6 | 1,051.5 | 3,408.9 | 1,173.7 | 365.5 | 3,173.4 | 6,641.3 | 859.7 | 554.8 |
| B | 1,086.8 | 1,018.9 | 1,109.4 | 968.6 | 1,058.4 | 362.1 | 2,734.3 | 5,186.9 | 757.9 | 497.9 |
| B2 | 649.6 | 839.4 | 1,087.8 | 638.3 | 729.3 | 515.0 | 2,666.8 | 4,860.2 | 1,298.0 | 519.6 |

n per airport: EDDF 787, EDDM 402, EGLL 647, EHAM 1,353, LEBL 652, LEMD 416, LFPG 1,565, LIRF 537, LSZH 722,
LTFM 1,297. Variant A lowers RMSE at nine of ten airports (LEMD 362.5 → 291.3 included) and raises it at none;
the largest non-LIRF gains are EDDF 1,064.5 → 409.2, EHAM 954.7 → 477.2, LEBL 1,043.4 → 371.4. Variant B leaves
non-LIRF airports essentially untouched (≤ +23 s anywhere) because P ≥ 0.5 occurs almost only at LIRF (§3.6).

### 3.5 Tail-removed comparison against base(repro) (`prc.compare.robustness_curve`, 500 bootstrap draws)

Negative = variant better. K rows removed are those where *either* model errs most.

| variant | K=0 | CI95 | K=20 | CI95 | K=200 | CI95 |
|---|---|---|---|---|---|---|
| oof file | −30.5 | [−56.6, −2.2] | +2.3 | [−13.3, +19.6] | +1.8 | [−5.7, +8.5] |
| **A** | **−694.4** | [−1,051.0, −381.8] | **−185.1** | [−261.6, −111.2] | **−15.7** | [−32.2, +2.3] |
| A2 | +370.2 | [−149.5, +1,012.1] | +51.7 | [−66.6, +188.2] | +45.2 | [+25.9, +63.6] |
| **B** | −400.4 | [−656.5, −168.4] | −22.5 | [−68.8, +22.5] | −25.0 | [−41.5, −9.9] |
| B-hard | −364.8 | [−652.1, −106.8] | +2.8 | [−47.2, +51.4] | −49.4 | [−63.6, −36.8] |
| B-LIRF-only | −403.8 | [−659.5, −171.9] | −12.7 | [−54.4, +26.4] | −25.1 | [−41.3, −10.0] |
| B2 | −505.9 | [−787.8, −243.4] | −116.3 | [−186.5, −51.0] | −11.7 | [−27.8, +5.2] |
| B2-LIRF-only | −459.3 | [−725.3, −208.3] | −32.6 | [−77.6, +7.5] | −29.7 | [−43.1, −13.7] |
| oracle·base | −123.3 | [−189.3, −72.3] | −141.9 | [−185.2, −103.4] | −67.9 | [−80.3, −55.1] |
| oracle·nofill | −169.9 | [−548.0, +149.2] | −361.4 | [−438.2, −281.8] | −110.4 | [−132.8, −88.7] |
| nofill alone | +319.7 | [+35.9, +618.3] | +280.8 | [+162.6, +403.1] | +43.8 | [+8.3, +84.1] |

Variant A's advantage is large at K=0, still −185 s at K=20 with a CI excluding zero, and −15.7 s at K=200 with a
CI that includes zero ([−32.2, +2.3], P(A better) = 0.952). The B family is the mirror image: its K=0 gain is
mostly tail rows, but B / B-hard / B-LIRF-only / B2-LIRF-only still have CIs excluding zero at K=200.

### 3.6 Classifier behaviour (variant B)

| fold | rounds (best) | AUC inner holdout | AUC validation | AUC valid, LIRF | AUC valid, non-LIRF | valid f10 rate | valid f10 rate @LIRF | mean P | mean P @LIRF |
|---|---|---|---|---|---|---|---|---|---|
| jan | 112 (103) | 0.9811 | 0.9567 | 0.9177 | 0.9294 | 0.0328 | 0.4167 | 0.0315 | 0.5096 |
| jul | 143 (130) | 0.9830 | 0.9681 | 0.8371 | 0.9592 | 0.0399 | 0.3234 | 0.0388 | 0.3760 |
| forward | 63 (57) | 0.9584 | 0.9701 | 0.8869 | 0.9259 | 0.0496 | 0.7071 | 0.0466 | 0.5690 |

Calibration pooled over the three validation folds:

| P bin | n | of which LIRF | observed f10 rate | mean P |
|---|---|---|---|---|
| [0.0, 0.1) | 7,718 | 126 | 0.010 | 0.007 |
| [0.1, 0.3) | 313 | 104 | 0.173 | 0.171 |
| [0.3, 0.5) | 89 | 62 | 0.326 | 0.382 |
| [0.5, 0.7) | 94 | 84 | 0.585 | 0.610 |
| [0.7, 0.9) | 114 | 111 | 0.798 | 0.806 |
| [0.9, 1.0) | 50 | 50 | 0.900 | 0.941 |

Rows with P ≥ 0.5 by airport: LIRF 245, LSZH 11, EHAM 2, every other airport 0.

### 3.7 Where the error moves: LIRF unmatched rows with ms ≥ 2 h (239 OOF rows)

| subset | n | y median | ms median | P median | RMSE base | RMSE A | RMSE B | RMSE B2 | RMSE oracle |
|---|---|---|---|---|---|---|---|---|---|
| f10 = true | 129 | 10,560 | 10,560 | 0.801 | 5,694 | **2,158** | 2,421 | 3,566 | 4 |
| f10 = false | 110 | 1,112 | 9,838 | 0.297 | 14,513 | **8,293** | 10,531 | 9,389 | 14,513 |

Prediction medians on the f10 = true subset: base 8,757, A 9,508, B 9,861, B2 8,505 (ms median 10,560; the OOF
file's 8,639 is item 1's figure, reproduced). On the f10 = false subset, whose labels have median 1,112 while ms
has median 9,838, every variant still predicts 5,000–7,200 — this is where the residual error sits.

**Rollover rows** (y ≥ 72,000; n = 9, 59.8 % of base's pooled SSE): RMSE base 54,883, A 34,157, B 40,378,
B2 38,705, A2 47,296, oracle 54,883 (the oracle hands these rows to the base model, since they are not f10).

### 3.8 Tail-prediction counts (labels: 53 rows y ≥ 4 h, 17 rows y ≥ 12 h)

| variant | pred ≥ 4 h | pred ≥ 12 h | non-LIRF pred ≥ 4 h |
|---|---|---|---|
| oof file | 52 | 5 | 9 |
| base(repro) | 52 | 4 | 10 |
| **A** | 42 | 15 | **0** |
| A2 | 54 | 16 | 11 |
| B | 56 | 13 | 10 |
| B2 | 44 | 15 | 1 |

Variant A removes every non-LIRF prediction ≥ 4 h (10 → 0) — the false-alarm rows of item 7 — while raising the
count of predictions ≥ 12 h from 4 to 15 against 17 such labels.

## 4. Verdict per sub-claim

| sub-claim | verdict | deciding observation |
|---|---|---|
| (a) E H2's pre-check: corr(y, MVT − SCHED) is high at LIRF unmatched and not at LFPG unmatched | **CONFIRMED** | §3.1: Pearson 0.8964 (n = 1,488) vs 0.0205 (n = 3,773); 0.9212 vs −0.0365 restricted to ms ≥ 2 h. Qualifier: the LIRF correlation is a mixture artefact, not a uniform relation — it is 1.0000 on the 721 f10 rows (where y = ms by the item-1 identity), 0.0682 on the 861 rows with ms < 2 h, and its Spearman is only 0.5233. LIRF **matched** rows give 0.2985. |
| (b) Variant A (target y − ms on LIRF rows) beats the current target | **CONFIRMED on this OOF** | §3.3: pooled 2,326.5 → **1,632.1** (−694.4 [−1,051, −382], P = 1.000); better on each of the three folds (3,247.5 → 2,904.6; 2,013.6 → 1,338.6; 2,177.4 → 1,024.6), on LIRF (7,326 → 4,214) **and** on non-LIRF (1,452 → 1,277), at nine of ten airports with no airport worse, and with the worst 20 rows removed (−185.1 [−262, −111]). At 200 rows removed the gain is −15.7 s with CI [−32.2, +2.3], i.e. no longer separated from zero. |
| (b′) The same reparameterisation applied to **all** unmatched rows (E H2's T = D − G) | **REFUTED** | §3.3: A2 pooled 2,696.7 vs base 2,326.5 (+370.2); worse on jul and forward, worse on non-LIRF (1,452 → 2,179), EHAM 954.7 → 3,408.9. The gain is specific to the airport where the fill mechanism exists. |
| (c) Variant B (classifier mixture P·ms + (1 − P)·regression) beats the current model | **CONFIRMED on this OOF, by less than variant A** | §3.3: 2,326.5 → **1,926.1** (−400.4 [−657, −168]); B2 (mixing with a non-filled-only regression) 1,820.6 (−505.9 [−788, −243]). Both stay better than base with 200 rows removed (−25.0 [−41.5, −9.9] and −11.7 [−27.8, +5.2]). Neither reaches A. |
| The classifier is learnable at all | **CONFIRMED** | §3.6: validation AUC 0.9567 / 0.9681 / 0.9701 per fold (inner-holdout 0.9811 / 0.9830 / 0.9584); within LIRF alone 0.8371–0.9177; calibration monotone across six P bins (observed f10 rate 0.010 → 0.900). |
| "P(sched-filled)·mms + (1 − P)·taxi" as literally written (the (1 − P) branch being a taxi-only model) | **CONFIRMED as an improvement, REFUTED as the best form** | §3.3: B2 (1,820.6) beats B (1,926.1), so the taxi-only second component helps; but the non-filled regression *alone* is worse than the baseline (2,646.2, +319.7 [+36, +618]), and both stay above variant A (1,632.1). |
| A perfect fill flag would be enough | **REFUTED** | §3.3/§3.7: oracle f10 → ms gives only 2,203.3 (base 2,326.5) and 2,156.6 with the non-filled model, both worse than the *learned* mixtures, because the 9 rollover rows are not f10 and keep their base predictions (RMSE 54,883, 59.8 % of SSE); on the 129 f10 rows the oracle's RMSE is 4 s. |
| Where the remaining error sits | **MEASURED** | §3.7: after variant A, the 110 LIRF ms ≥ 2 h rows that are *not* filled (y median 1,112 against ms median 9,838) still carry RMSE 8,293, and the 9 rollover rows still carry RMSE 34,157. |
| Baseline reproduced to the plan's 2,296.1 | **CONFIRMED within 1.3 %** | §3.2: 2,326.5 vs 2,296.1 (+30.4); per fold +24.5 / +4.8 / +68.1; the gap vanishes once 20 rows are removed (+2.3 [−13.3, +19.6]). Round counts differ (jan 393 → 566, jul 479 → 288, forward 45 → 45) with `num_threads` 4 → 2. |

## 5. What cannot be determined locally (CV → LB transfer)

- **The f10 label is unobservable in the ranking set.** `BLOCK_TIME_UTC_mvt` is null on all 344,841 ranking rows
  (item 1 §K), so "sched-filled" cannot be computed there. What variant B needs at prediction time is only the
  classifier's **inputs** — `cols_b` plus the categoricals, all of which are present on ranking rows with
  `mvt_minus_sched` null rate 0 — and the classifier is fitted entirely on training-side labels within each
  fold. So B is implementable; what is not checkable locally is whether its P is calibrated on 2026 rows, since
  no 2026 fill flag and no 2026 label exists. The same holds for variant A, whose *target* uses f10 only through
  `y` on training rows; its prediction rule (add `ms` back on LIRF rows) needs no flag at all.
- **Whether the 2026 LIRF rows are filled at the training rate.** Item 1 and item 3 established the density match
  (383 ranking LIRF unmatched rows vs 397 in training Jan + Jul 2025; 181 vs 181 with ms ≥ 2 h) but not the fill
  rate. Every gain measured here is concentrated on rows whose 2026 counterparts' fill state is unknown.
- **The leaderboard effect of either variant.** No 2026 labels. The OOF gains here are on the 8,378 unmatched
  rows of four 2025 months; the ranking set has 5,290 unmatched rows in two 2026 months.
- **Whether the K = 200 result decides it.** By `prc.compare`'s own convention an effect whose CI includes zero at
  200 rows dropped "was those rows"; variant A is at −15.7 [−32.2, +2.3] there while variant B is at
  −25.0 [−41.5, −9.9]. Which of the two orderings survives to a different row set is not observable here.
- **The exact logged model.** This run is not bit-identical to `oof_main-48c19bc830` (§3.2); a rerun of the
  pipeline at `num_threads: 4` would be needed to fit the variants against a byte-exact baseline.

## 6. Facts vs inference

**Facts (measured in this run).** Every number in §3: the correlation table including LIRF 0.8964 / LFPG 0.0205
and the f10-subset correlations of 1.0000; the baseline reproduction 2,326.5 against the file's 2,296.1 with its
per-fold and per-round detail; all pooled, per-fold, per-airport and tail-trimmed RMSEs of the twelve prediction
rules; the bootstrap intervals of §3.5; the classifier's per-fold AUCs, calibration bins and the airport
distribution of P ≥ 0.5; the f10 / ¬f10 split of the 239 LIRF ms ≥ 2 h OOF rows; the 9 rollover rows' RMSEs and
their 59.8 % share of base SSE; the tail-prediction counts; the resolved `TrainConfig` and the 84 + 10 column
count of `cols_b`.

**Inference (not measured here).** That variant A's non-LIRF gain is *caused* by the model no longer having to
represent LIRF's ms-sized targets — the co-movement with the disappearance of all 10 non-LIRF predictions ≥ 4 h
(§3.8, item 7's false-alarm rows) is consistent with it, but no tree structure was inspected and no ablation
isolates it. That the LIRF Pearson of 0.8964 reflects the fill mechanism rather than a genuine taxi/schedule
relation (it rests on item 1's identity plus the f10-subset correlation of exactly 1). That the ordering of the
variants would persist on a byte-exact 4-thread baseline, on other folds, or on 2026 rows. That the residual
8,293 s RMSE on the non-filled LIRF rows is irreducible; nothing here bounds it.

**Definitional notes.** "Sched-filled" here is f10 (|BLOCK − SCHED| ≤ 10 s), D's definition; M's ±6 s and ≤ 120 s
and |y − ms| < 1,800 s variants were not used as classifier labels, so the classifier's AUC is specific to f10.
The unmatched set is the routing definition throughout (22,470 training / 8,378 OOF rows); the FLIGHT_ID-null
definition was not run separately for the fits. The plan's "2,296" is the OOF file's figure and is reported here
alongside this run's 2,326.5 rather than in place of it.
