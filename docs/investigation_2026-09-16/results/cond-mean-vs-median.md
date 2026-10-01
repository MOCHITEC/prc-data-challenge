# Result — item 8 `cond-mean-vs-median`

Script: `docs/investigation_2026-09-16/scripts/cond-mean-vs-median.py` (no training, `POLARS_MAX_THREADS=1`,
runtime ≈ 1 min; reads `data/work/feat_train.parquet` (9 columns, LIRF rows only), `data/work/feat_rank.parquet`
(7 columns), `experiments/oof_main-48c19bc830.parquet`, `outputs/main.parquet` (v7), `outputs/det_a.parquet` (v8),
`outputs/ab_split.parquet`, `outputs/v2_two_clock.parquet`, scratchpad `uploaded/v3.parquet`, `uploaded/v4.parquet`).
Every number below was re-derived by this script; none is copied from the hypothesis reports. Sibling results used
as inputs, not re-derived: item 1 (`y − ms = −(BLOCK − SCHED)` identity; fill ⇔ y ≈ ms), item 3 (v9 = v8 on matched
rows and v4 on unmatched rows; 11 ranking LIRF unmatched rows with ms ≥ 12 h).

Definitions (plan §0): `y` = `TAXITIME_SEC_mvt`; `ms` = MVT − SCHED (verified equal to feature `mvt_minus_sched` on
all 1,488 LIRF unmatched training rows); "unmatched" = routing definition (`AOBT_3_flt` null; the FLIGHT_ID-null
definition gives 1,487 LIRF rows, one fewer). Row classes used throughout: **filled** = |y − ms| ≤ 60 s;
**rollover** = y ≥ 72,000 s and not filled; **other** = remainder. Bins by ms: 2–4 h = [7,200, 14,400),
4–12 h = [14,400, 43,200), ≥ 12 h = [43,200, ∞).

## 1. Hypothesis as stated (plan §1 item 8, source M)

For LIRF & ms ≥ 12 h the label is two-valued {ms (62 %), ≈ 87k (38 %)}; the RMSE-optimal prediction is the mixture
mean (~ ms + 10k); a median/robust loss would pick ms. [supposed] for ms = 60k the mixture mean ≈ 70k gives
RMSE ≈ 13k vs ≈ 16.6k for predicting ms and ≈ 25–50k for the current plateau.

Sub-claims tested: (A) two-valued label in the ≥ 12 h bin with proportions 62 / 38; (B) the RMSE-optimal rule is the
mixture mean ≈ ms + 10k; (C) a median rule picks ms; (D) the [supposed] arithmetic for ms = 60k; (E) the plan's
"Verified" forward-fold line (n = 10: mean y 48,907, model 26,296, bin-mean 44,405, ms 42,960); (F) what each rule
implies on the 11 ranking rows (requested arithmetic, not a claim of M).

## 2. Procedure actually executed

1. S0–S1: LIRF unmatched training rows (1,488) with y, ms, month, OOF prediction (537 rows have one; OOF months
   1, 7, 11, 12). Composition per ms bin (filled / rollover / other), also M's strict bin (43,200, ∞) and per month.
2. S2: full histograms per bin of y (1-hour cells up to 12 h, 2-hour cells to 24 h, then 30-min cells) and of
   y − ms (fixed cells around 0 and around ±1 day); row listings for the ≥ 12 h bin (29 rows) and the non-filled
   4–12 h rows (15).
3. S3: RMSE per bin of six rules — pred = ms; pred = constant bin mean(y); pred = constant bin median(y);
   pred = ms + mean(y − ms); pred = ms + median(y − ms); pred = OOF `main-48c19bc830` — with in-sample statistics,
   on all rows and on the OOF rows only (common denominator with the model).
4. S4: the same rules leave-one-month-out (12 folds; statistics from the other 11 months of the bin), on all rows and
   on the OOF rows; additionally fold-shaped hold-outs jan / jul / forward (Nov + Dec) for comparability with M.
   S4b: M's forward-fold statement recomputed on months 11 + 12, ms ≥ 4 h.
5. S5: analytic two-point mixture y ∈ {m, R} with π and R from the training ≥ 12 h bin; M's [supposed] line.
6. S6: the 11 ranking LIRF unmatched rows with ms ≥ 12 h, predictions of v2 / v3 / v4 / v7 / v8 / ab_split, expected
   SSE and RMSE of each rule under the assumption "y = ms with probability π = training filled share of the bin,
   otherwise y drawn from the 11 training rollover labels"; sensitivity to π ∈ {0, 0.25, 0.5, 0.621, 0.75, 0.9, 1};
   the same for the 32 ranking rows in the 4–12 h bin. S6b: SSE(v8) − SSE(v4) on the 11 rows for every label
   assignment in {ms, R}, against the LB unmatched-side gap SSE(v8) − SSE(v9) = (343.7964² − 306.4555²) × 344,841.
7. S7: the 15 OOF rows of the ≥ 12 h bin with the model's prediction and each rule's error.

## 3. Results

### 3.1 Composition per ms bin (LIRF unmatched training rows)

| bin | n | filled (|y−ms| ≤ 60 s) | rollover (y ≥ 20 h) | other | ms median | y mean | y median | mean(y−ms) | median(y−ms) | rows with OOF |
|---|---|---|---|---|---|---|---|---|---|---|
| < 2 h | 861 | 326 (37.9 %) | 0 | 535 | 5,526 | 3,008 | 1,321 | −2,051 | −3,718 | 298 |
| 2–4 h | 500 | 295 (59.0 %) | 1 | 204 (40.8 %) | 9,178.5 | 6,380 | 7,736.5 | −3,158 | −4 | 184 |
| 4–12 h | 98 | 83 (84.7 %) | 0 | 15 (15.3 %) | 19,021.5 | 20,097 | 18,451 | −2,436 | −1 | 40 |
| ≥ 12 h | 29 | 18 (62.07 %) | 11 (37.93 %) | 0 | 56,520 | 67,859 | 59,641 | +7,112 | +4 | 15 |

M's strict bin (43,200, ∞): identical (29 rows, 18 filled = 0.6207, 11 rollover, 0 other); no row has ms = 43,200.
All 18 filled rows in the ≥ 12 h bin have |y − ms| ≤ 6 s. One filled row (ms 131,163, y 131,167, Feb) also lies
above 20 h; it is counted as filled, not rollover. The 11 rollover labels are y = 87,170 … 88,392
(= 86,400 + 770 … 1,992), mean 87,536, sd 381; their ms values are 50,581 … 93,535 (one row has ms > y).

≥ 12 h bin per month (n / filled / rollover): Jan 2/1/1, Feb 1/1/0, Mar 1/0/1, Apr 1/1/0, Jun 2/1/1, Jul 9/5/4,
Aug 5/5/0, Sep 4/3/1, Nov 2/0/2, Dec 2/1/1; May and Oct 0. 4–12 h bin: 83 filled + 15 other (0 rollover); the 15
non-filled rows have y = 723 … 1,681 and ms = 14,644 … 23,039 (13 of 15 with ms < 20,000).

### 3.2 Histograms (counts)

**2–4 h (n = 500).** y: [0, 0.5 h) 180, [0.5, 1 h) 15, [1, 2 h) 4, [2, 3 h) 237, [3, 4 h) 63, [24 h, 24.5 h) 1.
y − ms: [−4 h, −2 h) 131, [−2 h, −30 m) 69, [−30 m, −60 s) 1, [−60 s, +60 s] 295, (60 s, 30 m] 3, (20 h, 24 h] 1.
|y − ms| ≤ 10 s on 295 rows (identical to ≤ 60 s); y < 3,600 on 195 rows; rows in neither mode
(3,600 ≤ y < ms − 1,800): 5.

**4–12 h (n = 98).** y: [0, 0.5 h) 15, [4, 5 h) 30, [5, 6 h) 20, [6, 7 h) 8, [7, 8 h) 4, [8, 9 h) 3, [9, 10 h) 3,
[10, 11 h) 6, [11, 12 h) 9. y − ms: [−12 h, −4 h) 10, [−4 h, −2 h) 5, [−60 s, +60 s] 83. |y − ms| ≤ 10 s on 83 rows;
no row between the two modes.

**≥ 12 h (n = 29).** y: [12, 14 h) 9, [14, 16 h) 5, [16, 18 h) 2, [18, 20 h) 1, [24 h, 24.5 h) 10, [24.5 h, 25 h) 1,
≥ 25 h 1 (the filled 131,167 row). y − ms: [−2 h, −30 m) 1 (rollover with ms 93,535), [−60 s, +60 s] 18,
(60 s, 30 m] 1, (30 m, 2 h] 1, (2 h, 4 h] 1, (4 h, 12 h] 7. |y − ms| ≤ 10 s on 18 rows; y in [86,700, 88,800] on 11;
y < 3,600 on 0; rows in neither mode: 1 (the ms 93,535 rollover, whose y − ms is −6,365).

### 3.3 RMSE per bin, in-sample statistics (S3)

"all" = every training row of the bin (statistics from the same rows); "OOF" = rows with an OOF prediction, same
statistics, so that all six rules share one denominator with the model.

| rule | 2–4 h all (500) | 2–4 h OOF (184) | 4–12 h all (98) | 4–12 h OOF (40) | ≥ 12 h all (29) | ≥ 12 h OOF (15) |
|---|---|---|---|---|---|---|
| pred = ms | 6,359 | 6,390 | 6,296 | 5,006 | 14,100 | 15,266 |
| pred = const bin mean(y) | 5,623 | 4,488 | 11,575 | 10,830 | 21,661 | 19,403 |
| pred = const bin median(y) | 5,785 | 5,041 | 11,692 | 11,075 | 23,167 | 21,889 |
| pred = ms + mean(y − ms) | **5,520** | 4,723 | **5,805** | **4,833** | **12,175** | **12,783** |
| pred = ms + median(y − ms) | 6,357 | 6,387 | 6,295 | 5,006 | 14,098 | 15,264 |
| pred = OOF main-48c19bc830 | — | **4,191** | — | 7,951 | — | 36,882 |

Statistics used: 2–4 h mean(y) 6,380, median(y) 7,736, mean(y − ms) −3,158, median(y − ms) −4; 4–12 h 20,097 /
18,451 / −2,436 / −1; ≥ 12 h 67,859 / 59,641 / +7,112 / +4.

### 3.4 Leave-one-month-out (S4)

12 folds, statistics from the other 11 months of the bin. "pred = ms" and the OOF prediction need no statistics and
are unchanged.

| rule | 2–4 h all | 2–4 h OOF | 4–12 h all | 4–12 h OOF | ≥ 12 h all | ≥ 12 h OOF |
|---|---|---|---|---|---|---|
| pred = ms | 6,359 | 6,390 | 6,296 | 5,006 | 14,100 | 15,266 |
| pred = const bin mean(y) | 5,708 | 4,732 | 11,706 | 11,135 | 22,717 | 19,663 |
| pred = const bin median(y) | 5,834 | 5,195 | 11,734 | 11,197 | 26,000 | 22,714 |
| pred = ms + mean(y − ms) | **5,628** | 5,008 | **5,855** | **4,855** | **12,715** | **13,397** |
| pred = ms + median(y − ms) | 6,358 | 6,388 | 6,295 | 5,006 | 14,098 | 15,264 |
| pred = OOF main-48c19bc830 | — | **4,191** | — | 7,951 | — | 36,882 |

Fold-shaped hold-outs (jan / jul / forward = Nov + Dec), ≥ 12 h bin: const mean 19,684, const median 22,882,
ms + mean 13,206, ms + median 15,264, ms 15,266, OOF 36,882. Per-fold statistics in the ≥ 12 h bin: median(y − ms)
is 3 or 4 s in every fold; mean(y − ms) ranges 5,492 (Nov held out) … 8,593 (Aug held out). 4–12 h: median(y − ms)
−1 or −2 in every fold, mean −2,103 … −2,595. 2–4 h: median −4 (−2 when July is held out), mean −2,411 … −3,381.

Ordering of rules is the same in-sample and LOMO in every bin. The model beats every constant rule only in the
2–4 h bin (4,191 vs best LOMO constant 4,732); in 4–12 h it loses to the ms-anchored rules (7,951 vs 4,855) and in
≥ 12 h by a factor 2.4–2.8 (36,882 vs 13,397 / 15,266).

### 3.5 M's forward-fold statement (S4b)

LIRF unmatched, ms ≥ 4 h, months 11 + 12: n = 10, all with OOF; mean y **48,907**; mean OOF **26,296**; mean ms
**42,960** (M's three figures reproduced exactly). Bin mean of y from the other ten months: 29,473 with a single
bin ms ≥ 4 h; 19,466 (4–12 h, 6 rows) and 66,298 (≥ 12 h, 4 rows) with the plan's two bins (row-weighted 38,199).
M's **44,405 is not reproduced** with either binning; M's LOMO table used ADEP × mms-bin edges that are not stated.
RMSE on these 10 rows: OOF 30,918; ms 12,999; constant other-month mean 33,211; ms + other-month mean(y − ms) 13,377.

### 3.6 Two-point mixture arithmetic (S5)

With π = 18/29 = 0.6207 and R = 87,536 (constant), for a row with ms = m: RMSE(pred = m) = √(1−π)·|R − m|;
RMSE(pred = mixture mean m + (1−π)(R − m)) = √(π(1−π))·|R − m|; the median of the two-point law is m (π > 0.5).

| m | mixture mean (= m + offset) | RMSE pred = m | RMSE pred = mixture mean | RMSE pred = 35,000 plateau |
|---|---|---|---|---|
| 47,337 | 62,585 (+15,248) | 24,758 | 19,505 | 33,784 |
| 60,000 | 70,445 (+10,445) | 16,959 | 13,361 | 37,879 |
| 80,000 | 82,858 (+2,858) | 4,641 | 3,656 | 47,998 |
| 100,000 | 95,272 (−4,728) | 7,677 | 6,048 | 60,575 |
| 111,654 | 102,506 (−9,148) | 14,854 | 11,703 | 68,512 |

The rollover spread (sd 381) adds √((1−π)·381²) ≈ 235 s in quadrature and is negligible. The offset of the mixture
mean over ms is (1−π)(R − m): +10.4k only near m ≈ 60k, +15k at m ≈ 47k, negative above m ≈ 87.5k. The bin-level
mean(y − ms) = +7,112 is the row-average of this offset over the 29 training rows.

### 3.7 The 11 ranking LIRF unmatched rows with ms ≥ 12 h (S6)

| MVT_ID | mo | rwy | ms | v2 | v3 | v4 | v7 | v8 | ab_split |
|---|---|---|---|---|---|---|---|---|---|
| 213153677 | 7 | 16R | 47,337 | 68,253 | 52,066 | 53,155 | 66,693 | 66,778 | 55,711 |
| 213142469 | 7 | 25 | 47,705 | 36,278 | 53,103 | 53,459 | 35,434 | 37,306 | 71,322 |
| 213143669 | 7 | 25 | 48,539 | 34,887 | 54,267 | 54,226 | 33,485 | 33,640 | 62,319 |
| 213143319 | 7 | 25 | 48,776 | 37,077 | 50,364 | 50,937 | 35,683 | 38,739 | 61,051 |
| 202685083 | 1 | 25 | 48,841 | 37,301 | 52,095 | 52,545 | 36,782 | 37,704 | 65,297 |
| 202687258 | 1 | 25 | 53,520 | 32,309 | 54,957 | 55,198 | 30,644 | 31,617 | 65,226 |
| 202680942 | 1 | 25 | 56,520 | 34,836 | 59,495 | 60,401 | 33,444 | 34,236 | 58,905 |
| 213154026 | 7 | 25 | 60,656 | 31,617 | 64,154 | 66,225 | 31,551 | 33,197 | 64,617 |
| 213145102 | 7 | 16R | 61,322 | 67,002 | 60,338 | 60,845 | 66,037 | 68,581 | 63,988 |
| 213152353 | 7 | 16R | 87,484 | 33,882 | 80,051 | 79,686 | 32,688 | 34,552 | 50,227 |
| 213145579 | 7 | 25 | 111,654 | 40,865 | 107,724 | 108,552 | 39,636 | 40,267 | 57,344 |

Expected SSE under "y = ms w.p. π = 0.6207, else y uniformly from the 11 training rollover labels (mean 87,536)";
bin statistics from all 29 training rows; ΔLB = change of LB RMSE relative to the "pred = ms" rule when everything
else is fixed and the LB without the change is 305.03 (resp. 343.80): ΔLB = √(305.03² + ΔSSE/344,841) − 305.03.

| rule on the 11 rows | E[SSE] | E[RMSE] on the 11 rows | ΔLB @305.03 | ΔLB @343.80 |
|---|---|---|---|---|
| pred = ms | 4.489e9 | 20,201 | 0 | 0 |
| pred = const bin mean(y) = 67,859 | 4.508e9 | 20,243 | +0.09 | +0.08 |
| pred = const bin median(y) = 59,641 | 5.844e9 | 23,049 | +6.37 | +5.67 |
| pred = ms + mean(y − ms) = ms + 7,112 | 3.478e9 | 17,781 | −4.84 | −4.29 |
| pred = ms + median(y − ms) = ms + 4 | 4.488e9 | 20,199 | −0.00 | −0.00 |
| pred = per-row mixture mean π·ms + (1−π)·87,536 | 2.787e9 | 15,916 | −8.20 | −7.26 |
| pred = v7 (`outputs/main.parquet`) | 1.695e10 | 39,259 | +54.40 | +49.07 |
| pred = v8 (`det_a`) | 1.609e10 | 38,243 | +50.89 | +45.86 |
| pred = v2 | 1.623e10 | 38,413 | +51.47 | +46.39 |
| pred = ab_split (255/20) | 6.527e9 | 24,358 | +9.54 | +8.49 |
| pred = v4 (uploaded) | 3.739e9 | 18,437 | −3.58 | −3.18 |
| pred = v3 (uploaded) | 3.832e9 | 18,664 | −3.14 | −2.78 |

Sensitivity of E[RMSE] on the 11 rows to π (same rollover component): π = 1 (all filled): ms 0, const mean 20,579,
const median 19,502, ms + 7,112 → 7,112, v7 31,885, v8 31,063, v4 4,644, v3 4,180. π = 0.5: ms 23,194, const mean
20,135, ms + mean 20,017, v8 40,260, v4 21,006. π = 0 (all rollover): ms 32,801, const mean 19,681, ms + mean 27,400,
v8 47,717, v4 29,341. The constant bin mean is the only rule whose RMSE is nearly flat in π (19.7k–20.6k);
"pred = ms" beats "const mean" only for π > 0.62.

S6b, the v8 − v4 SSE difference on the same 11 rows under the two-valued assumption, against the LB identity
SSE(v8) − SSE(v4) = 8.674e9 of which unmatched side SSE(v8) − SSE(v9) = 8.373e9 (matched side 3.012e8):
all 11 filled → 1.038e10 (124 % of the unmatched gap); all 11 rollover → 1.558e10 (186 %); expectation at
π = 0.6207 → 1.235e10 (147 %); minimum over all 2¹¹ assignments and R ∈ [87,170, 88,392] → 5.467e9 (65 %);
maximum → 2.070e10 (247 %). Per row, v4 is closer to ms than v8 on all 11; under the rollover outcome v8 is closer
than v4 only on the two 16R rows (ms 47,337 and 61,322, where v7/v8 predict 66–69k).

Extension, the 32 ranking rows in the 4–12 h bin under that bin's training proportions (π = 0.8469, non-filled
component = 15 training labels, mean 1,127): E[RMSE] ms 9,967; const mean 12,095; const median 12,360;
ms + mean(y − ms) (= ms − 2,436) 9,357; per-row mixture mean 9,173; v7 10,629; v8 10,860; ab_split 10,486;
v2 10,733; v4 13,537; v3 13,403. ΔLB@305.03 vs "pred = ms": const mean +7.06, const median +8.02, ms + mean −1.80,
v7 +2.07, v8 +2.82, v4 +12.51, v3 +11.98.

### 3.8 The 15 OOF rows of the ≥ 12 h bin (S7)

7 filled, 8 rollover. Model predictions 32,352 … 51,712 on all 15. Errors y − OOF: rollover rows +35,508 …
+55,128 (all eight); filled rows −6,344 … +21,415 (five of seven positive, i.e. model below ms). "pred = ms" errors:
filled ≤ 6 s; rollover +30,960 … −6,365. "pred = ms + 7,112": filled −7,106 … −7,117; rollover −13,477 … +23,848.

## 4. Verdict per sub-claim

- **(A) Two-valued label {ms 62 %, ≈ 87k 38 %} for LIRF & ms ≥ 12 h: CONFIRMED.** 18/29 rows have |y − ms| ≤ 6 s,
  11/29 have y = 86,400 + 770 … 1,992 (mean 87,536, sd 381, independent of ms), 0 rows elsewhere. The proportion
  0.6207 is reproduced exactly under both the plan's [43,200, ∞) and M's (43,200, ∞). The ≈ 87k component is not
  an offset from ms: the residual y − ms of rollover rows spans −6,365 … +36,962. Caveat: 29 rows over 10 months
  (July alone 9); month-level proportions range from 0/2 to 5/5 filled.
  The other two bins are also two-valued but with a different second component: 4–12 h = 84.7 % filled + 15.3 %
  ordinary taxi (y 723 … 1,681), 2–4 h = 59.0 % filled + 40.8 % ordinary taxi (y < 3,600 on 195 of 204).
- **(B) The RMSE-optimal prediction is the mixture mean ≈ ms + 10k: CONFIRMED in direction, REFUTED as stated in
  two details.** Confirmed: the ms-anchored mean rule (ms + mean(y − ms)) has the lowest RMSE of all constant rules in
  every bin, in-sample and leave-one-month-out (≥ 12 h: 12,715 vs 14,100 for ms, LOMO; on the OOF rows 13,397 vs
  15,266 vs model 36,882), and the per-row mixture mean is the RMSE minimiser under the two-point law by
  construction (S5). Refuted details: (i) the bin offset is +7,112 in-sample and 5,492 … 8,593 across LOMO folds,
  not 10k; the per-row offset (1−π)(R − m) is +15.2k at m = 47k, +10.4k at m = 60k, and negative above m ≈ 87.5k, so
  "ms + 10k" holds only near m ≈ 60k; (ii) a **constant** bin mean of y (67,859) is worse than plain "pred = ms" in
  the ≥ 12 h and 4–12 h bins (LOMO 22,717 vs 14,100 and 11,706 vs 6,296) because ms varies by 43,861 … 131,163
  within the bin. The claim is true only for the ms-anchored formulation.
- **(C) A median/robust loss would pick ms: CONFIRMED for the conditional median given ms; UNDETERMINED for a trained
  model.** median(y − ms) is +4 s (≥ 12 h), −1 s (4–12 h), −4 s (2–4 h), and 3 … 4 s in every LOMO fold of the
  ≥ 12 h bin, so the median-anchored rule equals "pred = ms" to within 4 s and loses to the mean-anchored rule by
  1,383 (all rows) / 1,867 (OOF rows) RMSE in the ≥ 12 h bin, 440 / 151 in 4–12 h, 730 / 1,380 in 2–4 h. The
  constant bin median of y (59,641) is not ms and is the worst rule in the ≥ 12 h bin (LOMO 26,000). What a GBDT
  trained with an L1/Huber objective would output was not tested (no training).
- **(D) [supposed] arithmetic for ms = 60k: CONFIRMED.** With π = 0.6207 and R = 87,536: mixture mean 70,445
  (M: ≈ 70k), RMSE 13,361 for the mixture mean (M: ≈ 13k), 16,959 for pred = ms (M: ≈ 16.6k), 37,879 for a 35k
  plateau (M: 25–50k). The plateau figure is 33,784 … 68,512 across the 11 ranking ms values.
- **(E) Plan "Verified" forward-fold line: CONFIRMED for n = 10, mean y 48,907, model 26,296, ms 42,960; UNDETERMINED
  for bin-mean 44,405** (29,473 with one bin ms ≥ 4 h, 19,466 / 66,298 with the plan's two bins; M's bin edges are
  not stated in the plan).
- **(F) Implication on the 11 ranking rows: arithmetic only, not a verdict on the LB.** Under the training
  proportions, the mean-anchored rule beats "pred = ms" by 4.8 s of LB RMSE and the per-row mixture mean by 8.2 s,
  while v7 / v8 / v2 lose 51–54 s and v4 / v3 gain 3.1–3.6 s against "pred = ms" (base 305.03). The mean-vs-median
  question (≈ 5 s) is an order of magnitude smaller than the model-vs-any-ms-anchored-rule gap (≈ 50 s) on these
  rows. On the 32 rows of the 4–12 h bin the ordering differs: v7 / v8 are within +2–3 s of "pred = ms" and v4 / v3
  are 12 s worse.

## 5. What cannot be determined locally

- The labels of the 11 (and 32, 181) ranking LIRF rows; hence the actual filled / rollover / ordinary proportions
  in 2026 (BLOCK_TIME is null on every ranking row). All of §3.7 is expectation under the training proportions.
- Whether the expected SSE differences translate to the LB: the ΔLB column assumes every other row unchanged and
  ignores cross terms; S6b shows that, under the two-valued assumption at the training π, the 11 rows alone would
  account for 147 % of the unmatched-side LB gap v4 → v8 (124 % even if all 11 are filled), while the minimum over
  assignments is 65 %.
- The output of a model trained with a median/Huber/quantile objective (no training run; only the conditional
  median of the label was computed).
- M's LOMO bin edges behind "bin-mean 44,405".

## 6. Facts vs inference

**Facts (measured in this run):** every number in §3: bin compositions and histograms; the 18 / 11 / 0 split of the
≥ 12 h bin and the rollover label range 87,170 … 88,392; all RMSE tables in-sample and LOMO; per-fold medians of
y − ms (3 … 4 s in the ≥ 12 h bin); the S4b reproduction of 48,907 / 26,296 / 42,960 and non-reproduction of 44,405;
the analytic table of §3.6 given π and R; the predictions of the six files on the 11 ranking rows; the LB identity
numbers 8.674e9 / 8.373e9 / 3.012e8; the per-assignment v8 − v4 sums of S6b.

**Inference (not measured):** that the 2026 LIRF rows follow the training proportions (π = 0.6207, rollover ≈ 87.5k);
every E[SSE], E[RMSE] and ΔLB figure in §3.7, which rests on that assumption plus "all other rows fixed"; the reading
of S6b that, if the 11 rows were two-valued at the training proportions, v8 would have to beat v4 by ≥ 2–4e9 SSE on
the other 5,279 unmatched rows for the LB identity to hold; that the ordering of rules seen on 29 / 98 / 500 training
rows carries over to a model trained with a different objective.
