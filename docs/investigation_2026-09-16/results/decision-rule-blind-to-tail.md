# Result — item 27 `decision-rule-blind-to-tail` (2026-09-16)

Script: `docs/investigation_2026-09-16/scripts/decision-rule-blind-to-tail.py` (no training, `POLARS_MAX_THREADS=1`, lazy column-selected scans, two OOF files in memory at a time; 369 s). Every bootstrap number below is produced by the project's own functions (`prc.compare.robustness_curve / month_bootstrap_rmse_diff / verdict / load_pair`, `prc.cv.paired_bootstrap_rmse_diff`) with `n_boot=500`, the `make compare` default, seed 0.
Inputs: `data/work/feat_train.parquet` (9 columns), `experiments/oof_*.parquet` (35 files), `experiments/experiments.jsonl`.
Notation as in plan §0: `y = TAXITIME_SEC_mvt`; unmatched = routing definition (`mvt_minus_aobt3` null); "rollover row" = training row with y > 72,000 (block-day-rollover §3.2: 16 rows, 10 of them in the OOF months 1/7/11/12); "LIRF sched-filled" = LIRF unmatched with |BLOCK − SCHED| ≤ 10 s (fill10; identical to |y − ms| ≤ 60 s on every OOF row here, 233 rows).

## 1. Hypothesis as stated (plan §1 item 27)

`prc.compare` adopts only on the drop-200 CI, ignores the month bootstrap, and by construction removes the ~20 rows that moved the LB; screening resolution (≈ 0.25 s) cannot see structural wins, which in public logs are 2–28 s while single features give ≤ 1.4 s.

Sub-claims tested: (a) `verdict()` uses only `robustness_curve`; the month bootstrap is printed but unused; (b) `robustness_curve` drops the K rows with the largest max(se_a, se_b) and those rows are the rollover / LIRF sched-filled rows on which two builds disagree most; (c) unmatched-only paired CI width 62 s vs 0.77 s matched (H §5); (d) `paired_bootstrap_rmse_diff` is arithmetically correct; (e) a `folds: [forward]` OOF compared with a 3-fold OOF silently restricts to Nov–Dec rows; (f) screening resolution ≈ 0.25 s; the ten feature experiments scored −0.21…+0.09; which past verdicts flip under a different rule.

## 2. Procedure executed

A. Line numbers of every function at HEAD via `inspect`; static check of what `verdict()` receives.
B. Pairs det_a-cfc9417c52 → ab_split-ca87cb5af7 (the item pair; both 672,428 rows, model B 127/50 vs 255/20, code shas 70ef994 vs 4e807b5, 14 commits apart) and main-ab56551b5e → main-f9d57a661f (same setting change, 3-seed builds): reproduce the drop order (`np.argsort(np.maximum(se_a, se_b))[::-1]`, compare.py line 43) and count, for K = 20 / 200 / 2000, how many dropped rows are unmatched, LIRF-unmatched, rollover, sched-filled, matched, in each label band; share of SSE_A, SSE_B and Σ(p_a − p_b)² removed; rank of every rollover row and of the 20 largest-disagreement rows in the drop order; full `make compare` reproduction; K = 200 recomputed under five alternative drop rules.
C. Paired bootstrap on unmatched-only and matched-only rows for five pairs; width sensitivity to seed and n_boot.
D. `paired_bootstrap_rmse_diff` on y = [0,0,0], p_a = [1,2,3], p_b = [1,1,1]: hand computation, exhaustive enumeration of the 27 equiprobable resamples, pairing check (p_b = p_a), permutation check, NaN handling; `verdict()` boundary cases fed with hand-made curves.
E. `load_pair` (compare.py lines 122–129) on 3-fold vs forward-only, Jan-only vs forward-only, Jan-only vs Jan-only, Jul-only vs 3-fold; month bootstrap with 2 / 1 months.
F. 28 available pairs (12 three-fold, 15 forward-only, 1 mixed) judged by the current rule and by five alternatives: full CI only; month CI only; drop-200 CI only; full ∧ month; drop-2000 only. Flips listed.

## 3. Results

### 3.1 Code facts at HEAD (A)

| object | HEAD location | plan §1 citation |
|---|---|---|
| `robustness_curve` | compare.py 34–61 (drop order line 43, bootstrap call line 50 with `seed=seed`, default 0) | "line 470" |
| `month_bootstrap_rmse_diff` | compare.py 64–88 | — |
| `verdict(curve, sign_check_level=200)` | compare.py 91–119 | "518–546" |
| `load_pair` | compare.py 122–129 | — |
| `main` | compare.py 132–171; month bootstrap computed and printed at 157–161; `verdict(curve)` at 169 | "line 584" |
| `paired_bootstrap_rmse_diff` | cv.py 87–117 | "157–187" |

compare.py has 175 lines and cv.py 125 at HEAD (9bea987); none of the plan's line numbers exist in these files. Substantively: `verdict()` reads only the `dropped == 0` and `dropped == sign_check_level` entries of the curve (lines 100, 110); its signature has no month argument; `mb` is never passed to it. `DROP_LEVELS = (0, 20, 200, 2000)`, `SIGN_CHECK_LEVEL = 200`. Rule as coded: REJECT "worse" if full CI lo > 0; REJECT "no difference" if full CI hi ≥ 0; ADOPT if no K=200 entry exists; REJECT "tail-only" if K=200 CI hi ≥ 0; else ADOPT. Boundary checks (hand-made curves): full hi = 0.0 exactly → "no difference"; K=200 hi = 0.0 exactly → "tail-only"; curve without a K=200 entry → ADOPT "too few rows".

OOF file facts relevant to comparisons: every `oof_*.parquet` has 2,085,047 rows (one per training row, unique `MVT_ID_mvt`) with `oof` null on unscored rows; non-null counts 672,428 (3-fold), 328,009 (`folds: [forward]`), 190,713 (`[jul, jul_fwd]`), 153,706 (`[jan]`). train.py 438–440 fills the OOF by `coalesce([oof, _p])` in fold order `jan, jul, jul_fwd, forward`, so `jul_fwd` predictions never reach the file (the 4-fold runs main-48c19bc830, main-63086f0648, pw_*, f10 have 672,428 non-null rows, not 863,141).

### 3.2 Which rows `robustness_curve` drops (B)

det_a → ab_split, 672,428 rows; population: unmatched 8,378; LIRF unmatched 537; rollover 10; LIRF sched-filled 233 (129 with y ≥ 2 h); y ≥ 2 h 210; y ≥ 4 h 55; y ≥ 12 h 18; matched 664,050.

| K | unmatched | LIRF unm | rollover (of 10) | LIRF sched-filled (of 233) | y ≥ 2 h | y ≥ 4 h | y ≥ 12 h | matched | share SSE_A | share SSE_B | share Σ(p_a−p_b)² | ∩ top-K by (p_a−p_b)² |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 20 | 15 | 9 | 5 | 15 | 15 | 13 | 0 | 0.431 | 0.318 | 0.420 | 13 |
| 200 | 189 | 129 | 9 | 51 (46 of the 129 with y ≥ 2 h) | 76 | 40 | 17 | 11 | 0.539 | 0.429 | 0.845 | 101 |
| 2000 | 665 | 389 | 10 | 170 | 169 | 55 | 18 | 1,335 | 0.645 | 0.560 | 0.916 | 532 |

main-ab56551b5e → main-f9d57a661f gives the same picture: K=200 drops 188 unmatched / 129 LIRF-unm / 9 rollover / 50 sched-filled / 12 matched, 0.546 of SSE_A, 0.430 of SSE_B, 0.895 of Σ(p_a−p_b)².

Rollover rows: the 9 unmatched ones occupy drop ranks 0 and 2–9 in both pairs (LFPG y 84,240 predicted 717 / 876; LIRF y 87,170–88,132 predicted 31,211–52,058 by A and 47,722–68,823 by B). The one matched rollover row (LIRF y 87,002, predicted 85,228 / 85,246) is at rank 1,211 / 1,129, i.e. dropped only at K=2000.
Top-20 rows by (p_a − p_b)² (all unmatched; 14 LIRF, of which 9 rollover and 4 sched-filled y = 41,766–64,437; 6 non-LIRF rows with y 652–1,265 predicted 13,633–17,743 by A and 663–3,520 by B): 13 are dropped at K=20, all 20 at K=200.

`make compare` reproduction, det_a → ab_split: K=0 −34.58 [−51.99, −17.92]; K=20 −4.56 [−8.79, −0.53]; K=200 −0.27 [−0.85, +0.31]; K=2000 −0.24 [−0.39, −0.07]; month bootstrap (4 months) −34.58 [−44.37, −22.71], P(B) = 1.000; VERDICT REJECT tail-only. main-ab56551b5e → main-f9d57a661f: −36.60 [−54.47, −18.50] / −4.84 / −0.36 [−0.89, +0.18] / −0.25 [−0.37, −0.13]; month −36.60 [−47.79, −22.94]; REJECT tail-only. Matched-row RMSE in both pairs differs by 0.05 s (218.58 → 218.53; 217.67 → 217.62); unmatched 2,299 → 1,875 and 2,313 → 1,865.

K=200 under alternative choices of the 200 dropped rows (det_a → ab_split; second pair in brackets):

| rule for the 200 rows | RMSE A | RMSE B | diff [CI] | rollover dropped | unmatched dropped |
|---|---|---|---|---|---|
| max(se_a, se_b) (current) | 228.21 | 227.94 | −0.27 [−0.85, +0.31]  (−0.36 [−0.89, +0.18]) | 9 | 189 |
| se_a only (baseline's worst) | 227.57 | 229.40 | +1.83 [+0.76, +3.13]  (+1.35 [+0.41, +2.42]) | 9 | 183 |
| se_b only (candidate's worst) | 237.55 | 226.37 | −11.18 [−16.00, −6.72]  (−11.93) | 9 | 173 |
| largest label y (model-independent) | 248.14 | 239.71 | −8.43 [−12.95, −3.82]  (−8.67 [−13.53, −3.93]) | 10 | 157 |
| largest (p_a − p_b)² | 268.76 | 268.28 | −0.48 [−0.90, +0.04]  (−0.46 [−0.93, −0.05]) | 8 | 200 |
| 200 random rows | 336.24 | 301.65 | −34.59 [−49.35, −17.16] | 0 | 2 |

### 3.3 Unmatched-only / matched-only CI width (C)

| pair | rows | all: diff [CI] width | unmatched (n 8,378 / 3,005): diff [CI] width | matched: diff [CI] width |
|---|---|---|---|---|
| det_a → ab_split | 672,428 | −34.58 [−51.99, −17.92] 34.1 | −423.7 [−610.6, −225.6] **385.0** | −0.05 [−0.17, +0.08] **0.25** |
| ab_single → ab_split | 672,428 | −74.10 [−101.9, −44.8] 57.1 | −821.9 [−1187.8, −463.4] 724.4 | −7.67 [−8.35, −7.17] 1.17 |
| main-ab56551b5e → main-f9d57a661f | 672,428 | −36.60 36.0 | −447.6 [−660.0, −238.5] 421.5 | −0.05 [−0.11, +0.02] 0.14 |
| det_a → det_b | 672,428 | +0.02 [−0.06, +0.10] 0.17 | +0.00 [0.00, 0.00] 0.00 (unmatched predictions identical) | +0.04 [−0.08, +0.20] 0.28 |
| ff1_noise-47e9 → feat_f5 (forward-only) | 328,009 | +0.03 [−0.24, +0.26] 0.50 | +1.34 [−1.24, +4.33] 5.57 | −0.09 [−0.35, +0.19] 0.54 |

Unmatched widths move by ±5 % with seed (369.5 / 385.0) and n_boot (404.1 at 2,000). No available pair gives H §5's "62 s / 0.77 s"; the pair H measured is not identified in `docs/handoff.md`.

### 3.4 Arithmetic of `paired_bootstrap_rmse_diff` (D)

y = [0,0,0], p_a = [1,2,3], p_b = [1,1,1]: hand RMSE_A = √(14/3) = 2.160247, RMSE_B = 1, diff = −1.160247; function `diff_rmse` = −1.160247 (match). Exhaustive enumeration of the 27 index tuples gives 10 distinct diff values {−2, −1.708, −1.517, −1.380, −1.160, −1, −0.915, −0.732, −0.414, 0}, P(diff < 0) = 26/27 = 0.9630; the function (n_boot = 200,000) gives `b_better_prob` = 0.9632 and CI [−2.000, 0.000], which is the 2.5 / 97.5 percentile of that discrete distribution (mass 1/27 = 3.7 % at each extreme exceeds 2.5 %; `np.percentile` over the 27-value list itself interpolates to [−1.81, −0.27], a property of the percentile estimator, not an error). Pairing: p_b = p_a → CI exactly [0, 0]; p_b = reversed p_a (identical RMSE) → CI [−2.54, +2.54], so the same index vector is applied to both models. A NaN row is dropped (`n` = 3, `n_dropped` = 1, diff unchanged). `rng` is seeded once per call (seed 0 by default) and the same seed is reused at every K by `robustness_curve`.

### 3.5 `folds: [forward]` OOF vs 3-fold OOF (E)

| A | B | non-null A / B | rows entering | months | month bootstrap |
|---|---|---|---|---|---|
| det_a (3-fold) | feat_base-4908 (forward) | 672,428 / 328,009 | **328,009** (0.488 of A) | 202511: 162,332; 202512: 165,677 | n_months = 2, −3.30 [−5.62, −1.04]; only C(3,2) = 3 distinct month multisets |
| rep_jan_on (jan) | feat_base-4908 (forward) | 153,706 / 328,009 | **0** | — | `robustness_curve` → `[]`; `verdict` → (False, "no full-data comparison") |
| rep_jan_off (jan) | rep_jan_on (jan) | 153,706 / 153,706 | 153,706 | 202501 | n_months = 0, NaN CI (function returns NaN when < 2 months) |
| leak_check (jul, jul_fwd) | det_a (3-fold) | 190,713 / 672,428 | 190,713 (0.284 of B) | 202507 | n_months = 0, NaN |

`load_pair` inner-joins on `MVT_ID_mvt` and keeps rows where both predictions are non-null (lines 128–129); `main` prints `rows = …` (line 146) but no warning about the restriction. With the 4-month 3-fold files the month bootstrap draws from C(7,4) = 35 distinct multisets; with forward-only files from 3. Forward-only pairs consequently show month CIs such as −0.02 [−0.02, −0.02] (ff1_noise-47e9 → ff1_noise2) and −0.17 [−0.18, −0.17] (→ feat_f8).

### 3.6 Re-judging 28 pairs (F)

Current rule ADOPTs 2 of 28: ab_single → ab_split (−74.10 / −6.26 at K=200 / −6.04 at K=2000; H's numbers reproduced exactly) and det_a → feat_base-4908 (−3.30 / −1.20; a 3-fold vs forward-only comparison restricted to Nov–Dec, code shas 70ef994 vs 5347c3b). All 26 others REJECT.

Handoff / features_log numbers reproduced: ablations vs ab_split −0.94 [−6.14, +3.57] / +1.41 / +1.58; +37.37 / +1.25 / +1.24; +7.38 / +7.16 / +4.68 (exact). Features at K=200 vs noise control: f1 −0.16 [−0.41, +0.07], f2 −0.19 [−0.41, +0.07], f3 −0.10 [−0.37, +0.15], f5 −0.06 [−0.29, +0.16], f6 +0.03 [−0.17, +0.22], f7 +0.09 [−0.15, +0.31], f10 planned_flight −0.21 [−0.48, +0.09] with matched 218.49 → 218.11 (H: [−0.49, +0.08], 218.5 → 218.1); +0.49 / +0.39 / +0.50 / +0.53 for f1–f3 and noise vs base (features_log). README's "v2 → v6: −34.82 / −0.06 [−0.61, +0.52] / −0.26" is not reproduced by any available pair; the nearest are main-ab56551b5e → main-f9d57a661f (−36.60 / −0.36 / −0.25) and det_a → ab_split (−34.58 / −0.27 / −0.24); the v2-era OOF used in d9f1b45 is not in `experiments/`.

K=200 CI widths: forward-only pairs 0.39–0.60 s (half-width 0.20–0.30 s); 3-fold pairs 0.24–1.82 s. K=0 widths: forward-only 0.29–5.89 s; 3-fold 0.17–57.1 s. Largest full-data effects among available pairs: −74.10 (single → split model), −34.6 / −36.6 (model B 127/50 → 255/20), +37.37 (encodings removed), +7.38 (categoricals removed), −5.81 (main-63086 → main-48c19), −3.30 (cross-fold-shape), −2.89 (dest_distance vs noise, forward), ±3.4 (f1–f3 vs base, forward); every other feature pair |diff| ≤ 1.36 at K=0 and ≤ 0.26 at K=200.

Flips relative to the current rule (27 rule × pair cells; n/a where n_months < 2 never occurred):

| pair | full [CI] | K=200 [CI] | month [CI] | current | full only | month only | K=200 only | full ∧ month | K=2000 only |
|---|---|---|---|---|---|---|---|---|---|
| main-ab56551b5e → main-f9d57a661f | −36.60 [−54.47, −18.50] | −0.36 [−0.89, +0.18] | −36.60 [−47.79, −22.94] | REJECT | **ADOPT** | **ADOPT** | REJECT | **ADOPT** | **ADOPT** (−0.25 [−0.37, −0.13]) |
| det_a → ab_split | −34.58 [−51.99, −17.92] | −0.27 [−0.85, +0.31] | −34.58 [−44.37, −22.71] | REJECT | **ADOPT** | **ADOPT** | REJECT | **ADOPT** | **ADOPT** (−0.24 [−0.39, −0.07]) |
| main-63086f0648 → main-48c19bc830 | −5.81 [−10.90, −1.12] | +0.26 [−0.07, +0.58] | −5.81 [−9.78, −2.50] | REJECT | **ADOPT** | **ADOPT** | REJECT | **ADOPT** | REJECT (+0.18 [+0.05, +0.30]) |
| det_a → main-48c19bc830 (1 → 3 seeds) | −1.62 [−3.75, +0.55] | −2.58 [−2.96, −2.15] | −1.62 [−4.29, +1.87] | REJECT | REJECT | REJECT | **ADOPT** | REJECT | **ADOPT** (−2.26) |
| ff1_base → ff1_noise-36511 (noise column added) | −0.25 [−0.56, −0.00] | −0.02 [−0.25, +0.19] | −0.25 [−0.49, −0.02] | REJECT | **ADOPT** | **ADOPT** | REJECT | **ADOPT** | REJECT |
| ff1_noise2 → feat_f9_dist | −2.89 [−4.32, −1.56] | +0.02 [−0.24, +0.25] | −2.89 [−2.93, −2.85] | REJECT | **ADOPT** | **ADOPT** | REJECT | **ADOPT** | REJECT |
| ff1_noise-47e9 → feat_f8_opoffset | −0.17 [−0.40, +0.07] | −0.26 [−0.52, −0.01] | −0.17 [−0.18, −0.17] | REJECT | REJECT | **ADOPT** | **ADOPT** | REJECT | **ADOPT** |
| ff1_noise-47e9 → feat_f7_vsusual | −0.26 [−0.74, +0.14] | +0.09 [−0.15, +0.31] | −0.26 [−0.38, −0.14] | REJECT | REJECT | **ADOPT** | REJECT | REJECT | REJECT |
| ff1_noise-47e9 → ff1_noise2 (noise vs noise) | −0.02 [−0.16, +0.12] | −0.08 [−0.25, +0.11] | −0.02 [−0.02, −0.02] | REJECT | REJECT | **ADOPT** | REJECT | REJECT | REJECT |
| pw_noise2 → f10_plannedflight | −0.06 [−0.89, +0.70] | −0.21 [−0.48, +0.09] | −0.06 [−0.91, +0.77] | REJECT | REJECT | REJECT | REJECT | REJECT | **ADOPT** (−0.25 [−0.40, −0.09]) |
| feat_f4_noise → feat_f1_queue | +1.36 [−0.02, +2.59] | −0.16 [−0.41, +0.07] | +1.36 [+1.18, +1.54] | REJECT | REJECT | REJECT | REJECT | REJECT | **ADOPT** (−0.23 [−0.38, −0.07]) |
| ff1_noise-47e9 → feat_f5_rwyact | +0.03 [−0.24, +0.26] | −0.06 [−0.29, +0.16] | +0.03 [+0.00, +0.06] | REJECT | REJECT | REJECT | REJECT | REJECT | **ADOPT** (−0.20 [−0.33, −0.04]) |

The other 16 pairs (ab_single → ab_split ADOPT under every rule; the three ablations, both determinism pairs, pw_noise → pw_both, f1–f3/f4 vs base, f2/f3 vs noise, f6, noise-36511 → noise-47e9, and det_a → feat_base ADOPT under every rule) do not flip under any rule.

## 4. Verdict per sub-claim

| sub-claim | verdict | deciding observation |
|---|---|---|
| (a) `verdict()` uses only `robustness_curve`; the month bootstrap is printed but unused | CONFIRMED (line numbers in the plan are wrong) | compare.py 91–119 read only the K=0 and K=200 curve entries; signature has no month argument; `main` prints `mb` at 157–161 and calls `verdict(curve)` at 169. compare.py has 175 lines; "518–546 / 470 / 584" and cv.py "157–187" do not exist at HEAD. |
| "adopts only on the drop-200 CI" | REFUTED as stated, CONFIRMED as "adopts only if full CI **and** drop-200 CI both exclude 0" | Lines 103–119: full CI hi ≥ 0 → REJECT before K=200 is consulted; det_a → main-48c19bc830 has K=200 −2.58 [−2.96, −2.15] and is REJECTED because the full CI is [−3.75, +0.55]. |
| (b) `robustness_curve` drops the K rows with the largest max(se_a, se_b) | CONFIRMED | compare.py line 43. |
| (b) at K=200 the dropped rows include the rollover / LIRF sched-filled rows | CONFIRMED with counts | det_a → ab_split K=200: 9 of the 10 rollover rows (the matched one at rank 1,211), 51 of 233 LIRF sched-filled rows (46 of the 129 with y ≥ 2 h), 129 of 537 LIRF-unmatched rows, 17 of 18 y ≥ 12 h, 189 unmatched vs 11 matched. |
| (b) "exactly where two builds disagree most" | CONFIRMED on OOF rows | All 20 largest-disagreement rows are dropped at K=200 (13 already at K=20); the dropped 200 carry 84.5 % (89.5 %) of Σ(p_a − p_b)². Whether they are "the ~20 rows that moved the LB" is a statement about ranking rows and is UNDETERMINED (lirf-regime-tail-count §5). |
| (b) the K=200 statistic is a property of the models alone | REFUTED | Same pair, same 200-row budget: −0.27 (max se) / +1.83 (A's worst) / −11.18 (B's worst) / −8.43 (largest y) / −0.48 (largest disagreement). The current rule removes 53.9 % of SSE_A but 42.9 % of SSE_B. |
| (c) unmatched-only paired CI width 62 s vs 0.77 s matched | REFUTED on every available pair (H's pair unidentified) | Unmatched widths 385 / 724 / 421 s (3-fold pairs), 5.6 s (forward-only), 0.0 (det_a → det_b, identical model B); matched widths 0.14–1.17 s. Ratio unmatched : matched 1,500–3,000× on 3-fold pairs, 10× on the forward-only pair. |
| (d) `paired_bootstrap_rmse_diff` arithmetically correct | CONFIRMED | Point estimate matches by hand; P(diff < 0) 0.9632 vs exact 26/27; CI equals the percentile of the exhaustively enumerated distribution; pairing preserved (identical predictions → [0, 0]); NaN rows dropped and counted. |
| (e) `folds: [forward]` vs 3-fold silently restricts to Nov–Dec rows | CONFIRMED | det_a → feat_base-4908: 328,009 rows = months 202511 + 202512 only, 0.488 of A's rows, no warning; month bootstrap then has 2 months and 3 possible resamples. Jan-only vs forward-only: 0 rows → "no full-data comparison" REJECT. Single-month pairs: month CI NaN. |
| (f) screening resolution ≈ 0.25 s | CONFIRMED as half-width at K=200 on forward-only pairs | K=200 CI widths 0.39–0.60 s (half-width 0.20–0.30); at K=0 on the same pairs 0.29–5.89 s; 3-fold pairs 0.24–1.82 s at K=200. |
| (f) the ten feature experiments scored −0.21…+0.09 vs noise | CONFIRMED (−0.21 … +0.09 at K=200; f8 −0.26 [−0.52, −0.01] is outside that range) | §3.6; all H values reproduced to ±0.02 with the CI ends within ±0.02. |
| (f) "structural wins are 2–28 s, single features ≤ 1.4 s" | UNDETERMINED (external); locally: structural changes −74 / −35 / +37 / +7 s at K=0, features ≤ 3.4 s at K=0 and ≤ 0.5 s at K=200 | Only the local side is measurable. |
| Past verdicts flip under a different rule | CONFIRMED, list in §3.6 | Full-CI-only or month-only would ADOPT the two 255/20 model-B pairs (−34.6 / −36.6), main-63086 → main-48c19 (−5.81), a pure noise column (ff1_base → ff1_noise −0.25 [−0.56, −0.00]) and dest_distance (−2.89); month-only on forward files also ADOPTs noise → noise2 (−0.02 [−0.02, −0.02]); K=200-only would ADOPT 1 → 3 seeds (−2.58) and f8; K=2000-only would ADOPT the two model-B pairs, seeds, f1 vs noise, f5, f8, f10. |
| "the rule agreed with LB direction in two known cases" | UNDETERMINED | Requires the LB outcome of the specific uploads; not measurable here. README's v2 → v6 numbers (−34.82 / −0.06 / −0.26) cannot be reproduced because the v2-era OOF is not in `experiments/`. |

## 5. What cannot be determined locally

- Whether the rows dropped at K=200 on OOF correspond to the ranking rows that moved the LB (labels of the 5,290 unmatched ranking rows unavailable; lirf-regime-tail-count §5).
- Which pair H §5 measured for "62 s / 0.77 s"; which OOF file was "v2" in README §8 / commit d9f1b45.
- The LB agreement of any verdict, and the external step sizes quoted by E.
- Whether the plan's line numbers refer to some earlier or concatenated version of the files (git history of compare.py: d9f1b45 created it at 170 lines; no revision reached line 470).

## 6. Facts vs inference

**Facts (measured or read in this run):** all line numbers in §3.1; the coalesce fold-order behaviour in train.py; every count in §3.2 (rollover ranks, sched-filled and disagreement overlaps, SSE and Σgap² shares); the six alternative-rule numbers; the CI widths in §3.3; the exhaustive-enumeration agreement in §3.4; the row counts and months in §3.5; the 28 × 6 verdict table and the reproduction of H's / features_log's numbers in §3.6; det_a and det_b have identical unmatched predictions.

**Inference (not established here):** that the OOF disagreement rows and the LB-moving ranking rows are the same kind of row; that a different rule would have produced a better LB (no rule's LB outcome is measured); that H's 62 s figure was computed on a pair or definition of "unmatched" different from those tried; that the plan's line citations came from a different file version. No recommendation is made.
