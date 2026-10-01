# Result — item 14 `robust-loss-harmful` (2026-09-17)

Script: `docs/investigation_2026-09-16/scripts/robust-loss-harmful.py` (model-B fits only, `POLARS_MAX_THREADS=1`,
LightGBM `num_threads: 2`, one fit at a time, runtime ≈ 16 min). Inputs: `data/work/feat_train.parquet`,
`configs/main.yaml`, `experiments/oof_main-48c19bc830.parquet`, and the reproduced L2 baseline produced by
item 9's run (`scripts/mms-mixture-reparam.py`) so that the baseline is not fitted twice. Nothing under `src/`
or `configs/` was modified.

**This report covers only the loss-function half of item 14.** The clipping and shrink half is already measured
in `results/tail-postprocessing-cap-shrink.md` (item 13) and is not re-derived here; its figures are quoted once,
for the joint reading in §4.

Definitions (plan §0): `y` = `TAXITIME_SEC_mvt`; `ms` = `mvt_minus_sched`; **unmatched** = routing definition
(`mvt_minus_aobt3` null → model B), 22,470 training rows / 8,378 OOF rows; **f10** = |BLOCK − SCHED| ≤ 10 s.
2 h = 7,200 s, 4 h = 14,400, 12 h = 43,200.

## 1. Hypothesis as stated (plan.md §1 item 14, source M)

Huber / quantile / log-target / clipping worsen RMSE here because the tail is predictable signal.

The plan marks the clipping part **verified** (item 13 reproduced it) and the loss-function part **unverified**
("not fit"). Sub-claims tested here: (a) Huber worsens unmatched OOF RMSE; (b) quantile (α = 0.5) worsens it;
(c) a log1p-target regression worsens it; (d) the stated reason — that the deterioration is the tail, i.e. rows
the L2 model was getting right, rather than a uniform loss of accuracy.

## 2. Procedure actually executed

**Resolved parameters.** `configs/main.yaml` read exactly as `src/prc/pipeline.py::main` does, then
`.for_unmatched()`: objective regression, learning_rate 0.05, **num_leaves 127**, **min_data_in_leaf 50**,
feature_fraction 0.8, bagging_fraction 0.8, bagging_freq 1, lambda_l2 1.0, verbose −1, **num_threads 2**
(the logged OOF run used 4), deterministic true, force_row_wise true, seed 42; `num_rounds` 2,000,
`early_stopping` 100, `inner_holdout_months` 1, `n_seeds` 3, `use_categoricals` true, `use_encodings` true.

**Data path per fold**, identical to `predict_fold`'s model-B branch: `fit_encodings` on the labelled training
side → `attach_encodings` → `visible_numeric_columns` → `cols_b` (prefix exclusions) = **84 numeric + 10
categorical** → `_matrix` → `_inner_holdout_mask` (latest month of the training side) → early stop → rescale
rounds → refit under 3 seeds and average → `np.clip(pred, 0, None)`.

**Folds**: `jan`, `jul`, `forward` from `prc.cv.month_block_folds` — the three behind
`oof_main-48c19bc830.parquet`. Validation rows 1,435 + 3,938 + 3,005 = **8,378**, joined to the OOF file with
zero nulls.

**Variants** (each changes only `objective` / `alpha`, or the target; 15 fits total, never two at once):

| id | change |
|---|---|
| base(repro) L2 | none — the code's own fit (reused per-row from item 9's run) |
| huber α = 0.9 | `objective: huber` at LightGBM's default alpha. **On this label scale α = 0.9 means a 0.9-second quadratic zone**, i.e. effectively L1 |
| huber α = 5,000 | `objective: huber, alpha: 5000` |
| huber α = 5,000, es = l2 | same plus `metric: l2`, to separate the objective from the early-stopping metric |
| quantile α = 0.5 | `objective: quantile, alpha: 0.5` (conditional median) |
| log1p target | `objective: regression` on `log1p(max(y, 0))`, prediction `expm1` |

The 240 negative labels (min −12 s) are floored at 0 for the **log1p target only** (the 11 zero labels need no
flooring, `log1p(0) = 0`); RMSE is always measured against the raw label. Evaluation: unmatched OOF RMSE per fold, pooled, per airport, LIRF vs non-LIRF,
and with the worst 20 / 200 rows removed — both by `prc.compare.robustness_curve` (the K rows where *either*
model errs most, 500-sample paired bootstrap) and by each variant's own worst K.

**Baseline reproduction** (measured in item 9's run, same code path, same threads, carried over here):
pooled **2,326.5** against the OOF file's **2,296.1** (+30.4, +1.3 %); per fold jan 3,247.5 / 3,223.0,
jul 2,013.6 / 2,008.8, forward 2,177.4 / 2,109.3; RMS(repro − file) 213.8 s, correlation 0.9967. The gap
disappears once 20 rows are removed (+2.3 [−13.3, +19.6]). The two runs are not bit-identical because
`num_threads` was 4 in the logged run and 2 here; the visible consequence is the early-stopping round count
(unmatched rounds jan 393 → 566, jul 479 → 288, forward 45 → 45). **All comparisons below are against the
reproduced 2,326.5.**

## 3. Results

### 3.1 Unmatched OOF RMSE per fold and pooled

| variant | jan | jul | forward | **pooled** | LIRF (537) | non-LIRF (7,841) | LIRF ms ≥ 2 h (239) | own-20 | own-200 | bias | pred median |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `oof_main-48c19bc830` | 3,223.0 | 2,008.8 | 2,109.3 | 2,296.1 | 7,179.3 | 1,450.2 | 10,463.2 | 1,173.8 | 608.3 | +12.1 | 1,062.3 |
| **base(repro) L2** | 3,247.5 | 2,013.6 | 2,177.4 | **2,326.5** | 7,326.3 | 1,451.7 | 10,697.9 | 1,172.4 | 605.9 | +13.4 | 1,060.5 |
| huber α = 0.9 (default) | 4,271.3 | 3,825.5 | 3,409.1 | **3,764.6** | 13,931.5 | 1,360.3 | 20,650.7 | 1,825.8 | 742.9 | −112.8 | 1,308.1 |
| huber α = 5,000 | 3,228.9 | 2,189.4 | 2,431.9 | **2,481.9** | 8,460.4 | 1,296.1 | 12,474.3 | 1,011.4 | 548.0 | −58.0 | 1,037.0 |
| huber α = 5,000, es = l2 | 3,228.5 | 2,131.7 | 2,431.9 | **2,458.1** | 8,339.0 | 1,301.4 | 12,275.5 | 1,023.7 | 559.3 | −51.6 | 1,039.3 |
| quantile α = 0.5 | 3,418.9 | 2,788.5 | 2,204.2 | **2,720.2** | 9,574.9 | 1,275.8 | 14,171.2 | 995.3 | 504.7 | −158.7 | 1,010.9 |
| log1p target | 3,412.2 | 2,823.7 | 2,003.0 | **2,679.7** | 9,403.8 | 1,271.4 | 13,913.5 | 1,049.8 | 487.3 | −201.9 | 992.5 |

Every robust variant is worse than L2 pooled; every one is better than L2 on the non-LIRF rows and worse on the
LIRF rows; every one has a negative bias (L2's is +13.4). Two variants beat L2 on **one** fold each
(huber α = 5,000 on jan, 3,228.9 vs 3,247.5; log1p on forward, 2,003.0 vs 2,177.4) and lose on the other two.

Round counts: huber α = 0.9 never early-stopped in any fold (best = 2,000 = the cap), so its numbers are the
2,000-round model, not a converged early-stopped one. huber α = 5,000 stopped at 585 / 1,007 / 116 (best 539 /
917 / 105); quantile at 1,571 / 2,000 / 1,499; log1p at 133 / 133 / 207. The `es = l2` variant differs from
huber α = 5,000 only on jan and jul (1,786 / 2,000 / 116 rounds) and moves pooled RMSE by 24 s, so the
early-stopping metric is not what drives the result.

### 3.2 Per airport, pooled RMSE

| variant | EDDF | EDDM | EGLL | EHAM | LEBL | LEMD | LFPG | **LIRF** | LSZH | LTFM |
|---|---|---|---|---|---|---|---|---|---|---|
| oof file | 1,073.4 | 999.7 | 1,099.0 | 926.8 | 1,044.1 | 418.3 | 2,728.8 | 7,179.3 | 802.8 | 508.4 |
| base(repro) L2 | 1,064.5 | 1,017.2 | 1,106.5 | 954.7 | 1,043.4 | 362.5 | 2,732.8 | **7,326.3** | 757.1 | 501.0 |
| huber α = 0.9 | 535.1 | 825.0 | 1,050.2 | 882.7 | 449.6 | 384.8 | 2,645.7 | **13,931.5** | 922.0 | 608.5 |
| huber α = 5,000 | 524.0 | 726.7 | 1,154.2 | 497.7 | 478.6 | 356.9 | 2,627.1 | **8,460.4** | 550.2 | 500.0 |
| huber α = 5,000, es = l2 | 525.4 | 735.9 | 1,152.1 | 514.8 | 490.3 | 360.5 | 2,631.4 | **8,339.0** | 591.8 | 502.9 |
| quantile α = 0.5 | 414.0 | 730.2 | 991.3 | 477.4 | 341.6 | 280.0 | 2,632.6 | **9,574.9** | 536.9 | 485.5 |
| log1p target | 380.5 | 738.8 | 985.3 | 460.7 | 291.1 | 261.9 | 2,628.7 | **9,403.8** | 545.1 | 491.6 |

n per airport: EDDF 787, EDDM 402, EGLL 647, EHAM 1,353, LEBL 652, LEMD 416, LFPG 1,565, LIRF 537, LSZH 722,
LTFM 1,297. LIRF is the only airport where every robust loss is worse; at EDDF, EHAM and LEBL the robust losses
cut RMSE by half or more (EDDF 1,064.5 → 380.5, EHAM 954.7 → 460.7, LEBL 1,043.4 → 291.1 under log1p).
LFPG barely moves (2,732.8 → 2,627–2,646) because its OOF error is dominated by two January rows that every
variant under-predicts (item 7 §C3).

### 3.3 Tail-removed comparison against base(repro) L2 (`prc.compare.robustness_curve`, 500 draws)

Negative = variant better. The K rows removed are those where *either* model errs most.

| variant | K=0 | CI95 | K=20 | CI95 | K=200 | CI95 |
|---|---|---|---|---|---|---|
| oof file | −30.5 | [−56.6, −2.2] | +2.3 | [−13.3, +19.6] | +1.8 | [−5.7, +8.5] |
| huber α = 0.9 | **+1,438.0** | [+1,003.6, +1,819.1] | **+560.7** | [+336.6, +784.2] | **+104.4** | [+55.8, +143.3] |
| huber α = 5,000 | +155.4 | [−5.2, +297.4] | **−167.9** | [−257.8, −95.2] | **−29.9** | [−43.6, −16.8] |
| huber α = 5,000, es = l2 | +131.6 | [−25.4, +273.7] | **−151.4** | [−233.1, −76.1] | **−14.3** | [−27.1, −1.6] |
| quantile α = 0.5 | +393.7 | [+106.6, +652.7] | **−202.8** | [−306.7, −95.2] | **−50.2** | [−69.0, −31.1] |
| log1p target | +353.2 | [+47.8, +620.9] | **−192.7** | [−305.8, −89.8] | **−69.0** | [−91.1, −44.4] |

The sign reverses for four of the five variants between K=0 and K=20, with CIs excluding zero on both sides.
huber α = 0.9 is the exception: it is worse at every K, including 200 rows dropped.

### 3.4 Where the damage is: normal labels vs tail labels

| variant | y < 2 h (n = 8,215) | y ≥ 2 h (n = 163) | y ≥ 4 h (n = 53) | rollover y ≥ 72,000 (n = 9) |
|---|---|---|---|---|
| oof file | 1,092.4 | 14,519.7 | 24,830.7 | 53,881.1 |
| base(repro) L2 | 1,097.1 | 14,749.5 | 25,243.9 | 54,882.7 |
| huber α = 0.9 | **824.3** | 26,347.2 | 44,744.7 | 85,714.6 |
| huber α = 5,000 | 873.1 | 16,679.3 | 28,786.0 | 62,969.7 |
| huber α = 5,000, es = l2 | 887.8 | 16,457.6 | 28,387.5 | 62,099.9 |
| quantile α = 0.5 | 791.2 | 18,675.5 | 32,193.2 | 68,422.8 |
| log1p target | **688.5** | 18,579.7 | 31,832.1 | 64,624.7 |

Every robust loss improves the 8,215 normal rows (1,097 → 689–888, i.e. by 19–37 %) and loses far more on the
163 tail rows. The 9 rollover rows carry 59.8 % of L2's pooled SSE; under huber α = 0.9 their RMSE is 85,715 —
the model predicts essentially the median everywhere.

### 3.5 Tail-prediction counts (labels: 163 rows y ≥ 2 h, 53 ≥ 4 h, 17 ≥ 12 h)

| variant | pred ≥ 2 h | pred ≥ 4 h | pred ≥ 12 h | non-LIRF pred ≥ 2 h | non-LIRF pred ≥ 4 h | max prediction |
|---|---|---|---|---|---|---|
| oof file | 151 | 52 | 5 | 24 | 9 | 51,712 |
| base(repro) L2 | 148 | 52 | 4 | 23 | 10 | 49,415 |
| huber α = 0.9 | **0** | **0** | **0** | 0 | 0 | **1,462** |
| huber α = 5,000 | 131 | 41 | 1 | 2 | 0 | 47,793 |
| huber α = 5,000, es = l2 | 135 | 42 | 3 | 3 | 0 | 50,859 |
| quantile α = 0.5 | 121 | 37 | 0 | 0 | 0 | 31,577 |
| log1p target | 79 | 34 | 4 | 0 | 0 | 47,477 |

huber at the default alpha emits **no** prediction above 1,462 s on any of the 8,378 rows. Every robust variant
removes all 10 non-LIRF predictions ≥ 4 h — the false-alarm rows of item 7 — which is where their non-LIRF gain
comes from.

### 3.6 LIRF unmatched rows with ms ≥ 2 h (239 rows), by fill flag

| variant | f10 = true (129): pred median / pred/ms median / RMSE | f10 = false (110): pred median / pred/ms median / RMSE |
|---|---|---|
| oof file | 8,639 / 0.826 / 5,619 | 6,091 / 0.557 / 14,172 |
| base(repro) L2 | 8,757 / 0.821 / 5,694 | 6,104 / 0.552 / 14,513 |
| huber α = 0.9 | 1,418 / 0.133 / 18,167 | 1,387 / 0.142 / 23,228 |
| huber α = 5,000 | 9,269 / 0.835 / 6,534 | 5,457 / 0.516 / 16,972 |
| huber α = 5,000, es = l2 | 9,098 / 0.839 / 6,400 | 5,304 / 0.528 / 16,714 |
| quantile α = 0.5 | 8,922 / 0.771 / 8,975 | 4,914 / 0.444 / 18,490 |
| log1p target | 7,203 / 0.687 / 10,551 | 3,635 / 0.328 / 17,031 |

On the 129 sched-filled rows (where y = ms exactly, item 1) the robust losses shrink the prediction away from
ms — from pred/ms 0.821 under L2 down to 0.687 under log1p and 0.133 under default-alpha huber. The f10 = false
subset, whose labels have median 1,112 against ms median 9,838, gets *worse* under every robust loss too,
because the shrinkage moves those predictions below the L2 value but the SSE there is dominated by the handful
of rollover rows inside the subset rather than by the many small-label rows.

## 4. Verdict per sub-claim

| sub-claim | verdict | deciding observation |
|---|---|---|
| (a) Huber worsens unmatched OOF RMSE | **CONFIRMED** at both alphas | §3.1: α = 0.9 (LightGBM default) 2,326.5 → **3,764.6** (+1,438.0 [+1,004, +1,819]); α = 5,000 → **2,481.9** (+155.4 [−5.2, +297.4]); α = 5,000 with `metric: l2` → 2,458.1 (+131.6 [−25.4, +273.7]). The two α = 5,000 variants' K=0 CIs touch zero, so their *deterioration* is not itself separated from zero; their direction is confirmed on 2 of 3 folds each. |
| (b) Quantile (α = 0.5) worsens it | **CONFIRMED** | §3.1: 2,326.5 → **2,720.2** (+393.7 [+107, +653]); worse on all three folds (3,418.9 / 2,788.5 / 2,204.2 against 3,247.5 / 2,013.6 / 2,177.4). |
| (c) A log1p-target regression worsens it | **CONFIRMED** | §3.1: 2,326.5 → **2,679.7** (+353.2 [+48, +621]); better on forward alone (2,003.0 vs 2,177.4). |
| (d) The stated reason: "because the tail is predictable signal" | **CONFIRMED in mechanism, with the sharper form measured** | §3.4: every robust loss *improves* the 8,215 rows with y < 2 h (1,097 → 689–888) and loses on the 163 with y ≥ 2 h (14,750 → 16,458–26,347); §3.3: the sign of the comparison reverses once 20 rows are removed (huber 5,000 −167.9, quantile −202.8, log1p −192.7, all CIs excluding zero) and stays reversed at 200. So the deterioration is not a uniform loss of accuracy — it is concentrated in the tail rows, which is exactly what "the tail is signal" predicts. |
| (d′) Is the tail L2 fits genuinely predictable? | **CONFIRMED for the sched-filled component, REFUTED for the rollover component** | §3.6: on the 129 f10 rows L2 reaches RMSE 5,694 with pred/ms 0.821 (and item 9 shows the target reparameterisation gets this to 2,158), so those rows carry learnable structure the robust losses discard. But on the 9 rollover rows even L2 has RMSE 54,883 and they still carry 59.8 % of its SSE — L2's advantage there is that it over-predicts them less badly, not that it predicts them. |
| Huber at LightGBM's default alpha | **MEASURED — it is not a "mild" robustification** | α = 0.9 on a label measured in seconds is a 0.9-second quadratic zone, i.e. effectively L1. §3.5: it emits zero predictions above 1,462 s against 163 labels ≥ 2 h; §3.1: LIRF RMSE 7,326 → 13,932. It is also the only variant that never early-stopped (best = the 2,000-round cap in all three folds), so its figures are a capped model. It is the only variant still worse than L2 at 200 rows dropped (+104.4 [+55.8, +143.3]). |
| The early-stopping metric explains the result | **REFUTED** | §3.1: huber α = 5,000 with `metric: l2` gives 2,458.1 vs 2,481.9 with huber's own metric — 24 s of 155, on different round counts (1,786 / 2,000 / 116 vs 585 / 1,007 / 116). |
| Item 14 vs item 13 (the plan lists them as conflicting) | **RECONCILED, not in conflict** | Both find the same split. Item 13: capping *all* unmatched predictions worsens RMSE (2,296 → 3,373 at 7,200) while capping *non-LIRF* unmatched rows only improves it (2,296 → 2,207 at 3,600, total 334.58 → 327.05). Here: every robust loss improves non-LIRF (1,451.7 → 1,271–1,360) and worsens LIRF (7,326 → 8,339–13,932). The common finding is that model B's large LIRF predictions have large labels and its large non-LIRF predictions do not; a loss change cannot separate the two because it acts on all rows at once. |
| Baseline reproduced to the plan's 2,296.1 | **CONFIRMED within 1.3 %** | §2: 2,326.5 vs 2,296.1; per fold +24.5 / +4.8 / +68.1; the gap vanishes at 20 rows dropped (+2.3 [−13.3, +19.6]). |

## 5. What cannot be determined locally (CV → LB transfer)

- **The leaderboard effect of any of these losses.** The 2026 ranking labels are unknown. Everything above is on
  8,378 unmatched rows from four 2025 months; the ranking set has 5,290 unmatched rows in two 2026 months.
- **Whether the tail that L2 fits is present in 2026 at the training rate.** The gains and losses are decided by
  the 163 rows with y ≥ 2 h, and above all by the 9 rollover rows; items 1–3 established that the ranking LIRF
  density matches (383 vs 397 rows; 181 vs 181 with ms ≥ 2 h) but `BLOCK_TIME` is null on all 344,841 ranking
  rows, so neither the fill state nor the rollover count there is observable.
- **Whether the non-LIRF improvement would transfer.** It comes from suppressing predictions ≥ 4 h on rows whose
  2025 labels were 549–1,431 s (item 7 §C2); the corresponding 2026 rows' labels are unknown, and item 15 flags
  the EHAM January 2026 excess as a reason the 2025 label distribution may not hold there.
- **Whether a robust loss applied to non-LIRF rows only would help.** Not fitted — a per-row-group objective is
  not expressible through `configs/main.yaml`, and this item was scoped to the objective changes the plan names.
- **The byte-exact baseline.** This run is not bit-identical to `oof_main-48c19bc830` (`num_threads` 4 → 2);
  a 4-thread rerun would be needed to compare against the logged model exactly.
- **Whether a tuned alpha exists that keeps the non-LIRF gain without the LIRF loss.** Only two alphas were
  fitted (0.9 and 5,000); the curve between and beyond them was not traced.

## 6. Facts vs inference

**Facts (measured in this run).** Every number in §3: the pooled, per-fold, per-airport, LIRF / non-LIRF and
tail-trimmed RMSEs of all six prediction sets; the bootstrap intervals of §3.3 including the sign reversal
between K=0 and K=20; the round counts and the fact that huber α = 0.9 never early-stopped in any fold; the
y < 2 h / y ≥ 2 h / y ≥ 4 h / rollover decomposition; the tail-prediction counts including huber α = 0.9's
maximum prediction of 1,462 s and the disappearance of all 10 non-LIRF predictions ≥ 4 h under every robust
variant; the pred/ms medians on the 129 sched-filled LIRF rows; the resolved `TrainConfig`; the 240 negative and
11 zero labels among the 22,470 unmatched training rows. The baseline reproduction figures are carried over from
item 9's run, which used the same script-level fitting function and the same thread count.

**Inference (not measured here).** That huber α = 0.9's collapse is *because* a 0.9-second quadratic zone makes
the objective effectively L1 on a second-scale label (the alpha value and the resulting near-median predictions
are consistent with it; no intermediate alphas were fitted to trace the transition). That the robust losses'
non-LIRF gain and the non-LIRF cap gain of item 13 are the same effect reached two ways (both suppress the same
9–10 large-ms false-alarm rows, but no row-level attribution was computed across the two experiments). That the
ordering of these variants would survive on a byte-exact 4-thread baseline, on other folds, or on 2026 rows.
That the 163 tail rows' structure is what the robust losses discard rather than some correlated property of
those rows.

**Definitional notes.** "Huber at the LightGBM default" is α = 0.9, which LightGBM documents as a scale
parameter in the units of the label — here seconds — so it is not comparable to a default chosen for a
standardised target. `objective: quantile, alpha: 0.5` is LightGBM's pinball loss at the median, not an L1
objective with a different name (`regression_l1` was not fitted separately). The log1p variant floors the 240
negative labels at 0 in the target only. The unmatched set is the routing definition throughout; the
FLIGHT_ID-null definition was not run separately for the fits.
