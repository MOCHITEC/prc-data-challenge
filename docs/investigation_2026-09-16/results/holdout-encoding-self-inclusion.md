# Item 24 `holdout-encoding-self-inclusion` — result (2026-09-16)

Script: `docs/investigation_2026-09-16/scripts/holdout-encoding-self-inclusion.py` (model B only, 3 early-stopping
fits + 2 refits, 103 s total; `POLARS_MAX_THREADS=1`, LightGBM `num_threads: 2`, lazy column-selected scans).
Definitions follow plan §0: "unmatched" = routing definition (`mvt_minus_aobt3` null), 22,470 training / 5,290
ranking rows; `ms` = `mvt_minus_sched`.

## 1. Hypothesis as stated (plan.md §1 item 24, source P 6)

The label encodings (`ref_*`, `arh_med`, `apt_med`) are fitted on the whole training side including the
inner-holdout month, so the early-stopping holdout sees its own labels. Verified already by P as a code path
(`predict_fold` fits ref/arh/apt/op on all labelled training rows at train.py:317-318, then `_fit` carves the
holdout month out of the same frame). **Unverified: the effect on `best_iteration` and on the tail.**

Sub-claims tested: (A) the holdout rows' encodings really do contain their own labels, and how diluted;
(B) the effect on `best_iteration`; (C) the effect on the December holdout RMSE; (D) the effect on the tail of the
model that ships.

## 2. Procedure actually executed

### How the holdout was carved

Exactly as the code does it, by reusing `_inner_holdout_mask(tr, cfg.inner_holdout_months)` (train.py:235-249) on
model B's own training frame: it keys rows by `MVT_TIME_UTC_mvt.year()*100 + .month()`, sorts the 12 unique keys
`[202501 … 202512]` and marks the last `inner_holdout_months = 1` of them. That is **2025-12, 1,649 of the 22,470
unmatched training rows** — the same month and count P and M report (plan §4-3). No other split was used; the
early-stop fit trains on the remaining 20,821 rows and validates on those 1,649, and the refit round count is
`best_iteration × 22,470 / 20,821` capped at `unmatched_num_rounds`, exactly as train.py:293-295.

### The three encoding variants

`configs/main.yaml` resolved the way `prc.pipeline.main` does, then `.for_unmatched()`, `n_seeds` = 1,
`num_threads` = 2; weather on (as in `main.yaml`), so `cols_b` = **83 numeric + 10 categorical**. Encodings are
fitted with `fit_encodings` and attached with `attach_encodings`, never re-implemented:

| variant | encodings on months 1-11 rows | encodings on the 1,649 December holdout rows |
|---|---|---|
| **A** production path | fitted on all 12 labelled months (2,085,047 rows) | fitted on all 12 months — **contain the holdout's own labels** |
| **B** honest encodings | fitted on months 1-11 (1,919,370 rows) | fitted on months 1-11 |
| **C** holdout-only fix | fitted on all 12 months (unchanged) | fitted on months 1-11 |

C is the literal reading of the plan's procedure ("fit the encodings on months 1-11 and attach them to December")
and is the clean isolation: its early-stop **training** rows are byte-identical to A's, so C's booster is by
construction the same booster as A's and only the early-stopping *signal* differs.

Table sizes: all-12 → ref 6,434 / arh 1,024 / apt 10 / op 1,281 groups; months 1-11 → 6,304 / 1,017 / 10 / 1,247.

Resolved LightGBM parameters (all three fits and both refits):

```
{"bagging_fraction":0.8, "bagging_freq":1, "deterministic":true, "feature_fraction":0.8,
 "force_row_wise":true, "lambda_l2":1.0, "learning_rate":0.05, "min_data_in_leaf":50,
 "num_leaves":127, "num_threads":2, "objective":"regression", "seed":42, "verbose":-1}
early stopping: lgb.early_stopping(100), num_boost_round 2000
refits add: {"bagging_seed":42, "feature_fraction_seed":42}   (train.py:302, k = 0)
```

For the tail question both refits use the **production (all-12) encodings** on all 22,470 training rows and on the
5,290 ranking rows, so the only thing that varies between them is the round count the early stop chose.

## 3. Results

### 3.1 How much of its own label a holdout row sees (A)

Per December holdout row, the size of its encoding group and the share of that group contributed by December rows
(the group's own month), measured on all labelled training rows:

| table | keys | holdout rows with a group | median group n | median Dec share | mean | max | share > 20 % |
|---|---|---|---|---|---|---|---|
| ref (`ref_p10/p20/med/n`) | ADEP × STAND × RUNWAY | 1,649 / 1,649 | 428 | 8.84 % | 10.58 % | 100 % | 4.1 % |
| arh (`arh_med`) | ADEP × RUNWAY × hour_local | 1,649 / 1,649 | 3,638 | 8.54 % | 9.83 % | 100 % | 3.4 % |
| apt (`apt_med`) | ADEP | 1,649 / 1,649 | 239,552 | 8.26 % | 7.98 % | 8.47 % | 0 % |

Every holdout row's encodings include December rows; a row's own label is 1 of a median 428 (ref), 3,638 (arh) or
239,552 (apt) values, and December as a whole contributes ~8-10 % of each group. `reference_min_count = 30` and
the arh minimum of 10 bound the worst case, but do not prevent it: 4.1 % of holdout rows sit in a ref group that
is more than 20 % December, and the maximum share is 1.0 (a group consisting only of December rows).

How far the columns actually move when December is removed from the fit, on those 1,649 rows:

| column | rows changed | median abs change | p90 | nulls all-12 → 1-11 |
|---|---|---|---|---|
| `ref_p10` | 909 | 1.0 s | 6.0 s | 83 → 90 |
| `ref_p20` | 1,010 | 1.0 s | 6.0 s | 83 → 90 |
| `ref_med` | 1,240 | 1.5 s | 7.5 s | 83 → 90 |
| `ref_n` | 1,559 | 39 rows | 130 | 83 → 90 |
| `arh_med` | 1,382 | 2.0 s | 7.0 s | 17 → 19 |
| `arh_n` | 1,630 | 335 rows | 711 | 17 → 19 |
| `apt_med` | 1,106 | 1.0 s | 2.0 s | 0 → 0 |
| `ref_p10_fb` | 991 | 1.0 s | 7.0 s | 0 → 0 |
| `ref_missing` | 7 | — | — | 0 → 0 |

The label-derived *values* move by 1-2 s at the median; the *count* columns (`ref_n`, `arh_n`) move by 39 and 335,
which is the 8-10 % of rows removed.

### 3.2 Early stopping under the three variants

| variant | `best_iteration` | rounds after scaling | December holdout RMSE at best | iterations run |
|---|---|---|---|---|
| **A** production (enc 1-12) | **542** | 585 | **1,694.16** | 642 |
| **B** honest (enc 1-11 everywhere) | **596** | 643 | **1,689.30** | 696 |
| **C** holdout-only (enc 1-11 on Dec) | **542** | 585 | **1,692.19** | 642 |

Holdout RMSE along the curve:

| iteration | 50 | 100 | 200 | 300 | 400 | 500 | 600 |
|---|---|---|---|---|---|---|---|
| A | 1,831.1 | 1,799.3 | 1,744.7 | 1,724.0 | 1,711.6 | 1,708.3 | 1,700.6 |
| B | 1,838.0 | 1,813.1 | 1,743.2 | 1,726.8 | 1,708.4 | 1,702.7 | 1,691.6 |
| C | 1,827.5 | 1,794.1 | 1,740.1 | 1,718.8 | 1,707.7 | 1,706.5 | 1,699.1 |

Isolation with one fixed model (A's booster at its own `best_iteration` = 542, holdout features swapped):

| holdout features | RMSE on the 1,649 rows |
|---|---|
| all-12 encodings (production) | 1,694.16 |
| months 1-11 encodings | 1,692.19 |
| RMS of the per-row prediction difference | 146.40 (247 of 1,649 predictions unchanged) |

(These are C's numbers by construction: C's training rows equal A's, so it is the same booster.)

A's fit was also produced independently in item 22's "WITH `hour_utc`" run (`best_iteration` 542, rounds 585,
holdout RMSE 1,694.16) — the same configuration reproduces.

### 3.3 Consequence for the ranking predictions (the tail)

Refits on all 22,470 unmatched rows with production encodings, at each variant's round count:

| refit | median | mean | p99 | max | n > 7,200 | n > 14,400 | n > 43,200 |
|---|---|---|---|---|---|---|---|
| 585 rounds (A, C) | 1,083.2 | 1,593.9 | 10,057.5 | 66,076.5 | 100 | 35 | 2 |
| 643 rounds (B) | 1,080.6 | 1,591.3 | 10,328.4 | 67,199.5 | 99 | 35 | 2 |
| v8 = det_a (file) | 1,066.0 | 1,608.0 | 10,184.1 | 68,581.0 | 101 | 35 | 2 |
| v4 (probe file) | 1,112.0 | 1,702.5 | 9,457.5 | 108,552.0 | 83 | 31 | 18 |

RMS(585 − 643) on the 5,290 unmatched ranking rows: **63.3** overall, 91.5 on the 1,721 rows with ms ≥ 2 h, 533.2
on the 25 rows with ms ≥ 12 h, 130.9 at LIRF. For scale, item 21 measured the refit seed-key change at 314.5 s on
the same rows and `results/lirf-regime-tail-count.md` gives v8 − v4 at 1,735.5 s.

## 4. Verdict per sub-claim

- **(A) The early-stopping holdout sees its own labels through the encodings: CONFIRMED, and quantified.** All
  1,649 December holdout rows have ref / arh / apt encodings fitted on data that includes December; December's
  share of the group is ~8-10 % at the median, 100 % at the maximum, and 4.1 % of holdout rows sit in a ref group
  that is more than 20 % December. The code path was already established by P; this run measures the dilution.
- **(B) Effect on `best_iteration`: REFUTED as an effect of the holdout leak; CONFIRMED as an effect of the
  encoding fit as a whole.** Removing December from the holdout rows' encodings alone (C) leaves
  `best_iteration` at **542**, unchanged. Removing December from the encodings everywhere (B) moves it 542 → 596
  (rounds 585 → 643, +9.9 %), but that variant also changes the 20,821 training rows' own columns, so the shift
  is not attributable to the holdout leak.
- **(C) Effect on the December holdout RMSE: CONFIRMED in direction, negligible in size, and the sign is opposite
  to a leak advantage.** With the booster held fixed, the leaked encodings make the holdout look **1.97 s worse**
  (1,694.16 vs 1,692.19 on 1,649 rows, 0.12 %), not better — although the per-row predictions move by 146.4 s RMS,
  so the near-cancellation is an aggregate effect, not a "nothing changed". Under variant B the holdout RMSE at
  the variant's own best iteration is 1,689.30, 4.9 s below A.
- **(D) Effect on the tail: REFUTED as a material effect.** The only channel by which the leak can reach the
  shipped model is the round count, and the leak does not change it (542 in both A and C). Even taking B's round
  count, the ranking predictions move 63.3 s RMS, the count above 12 h stays at 2 and the count above 4 h at 35;
  none of the 18 rows where v4 predicts above 43,200 changes regime.

## 5. What cannot be determined locally

- Whether any of these differences move the leaderboard: the 5,290 unmatched ranking rows are unlabelled.
- Whether the same conclusion holds for **model A**, where the encodings are fitted on 2.06 M rows and the holdout
  is ~170 k rows: P flagged that fit as expensive and it was not run here. The dilution arithmetic differs
  (model A's holdout is the same December month but a far larger share of each ref group's rows is matched).
- Whether `best_iteration` is stable at all under this holdout: this is one seed and one month. Items 23
  (`holdout-month-decides-rounds`) and the observed spread of `rounds_unmatched` across builds (199 / 212 / 585 /
  658) bear on that; no CI or repeated-seed study was run here.
- Whether the +54 iterations of variant B are a real improvement or holdout noise: the curve is flat to within
  ~10 s between iterations 400 and 640 in all three variants (§3.2), which is the same order as the differences
  being compared.
- Whether the four-table encoding scheme should fit per fold at all (handoff §5 measures removing all nine columns
  at +37.37 s all rows / +450 s unmatched); that is a different experiment.

## 6. Facts vs inference

**Facts (measured in this run):** the holdout definition and its 1,649 rows; the encoding table sizes; every
self-inclusion statistic and column-change statistic in §3.1; `best_iteration` 542 / 596 / 542, rounds 585 / 643 /
585 and holdout RMSE 1,694.16 / 1,689.30 / 1,692.19 with the curves; the fixed-booster swap (1,694.16 → 1,692.19,
per-row RMS 146.40, 247 identical predictions); the ranking tails and RMS(585 − 643) = 63.3 / 91.5 / 533.2 / 130.9.

**Inference (not measured here):** (i) that variant C's booster is identical to A's — it follows from the
training rows being byte-identical and the same seeds, and is corroborated by C reproducing A's `best_iteration`
and iteration count exactly, but the two boosters were not compared tree by tree; (ii) that the round count is the
*only* channel from the holdout leak to the shipped model — true of the code as written (`_fit` uses the
early-stop booster for nothing but `best_iteration`), read from train.py:293-296, not tested by ablation;
(iii) that the 1.97 s difference is small relative to holdout noise — asserted from the flatness of the curve, not
from a confidence interval; (iv) that A's fit here is the same fit the production pipeline performs — it matches
item 22's independent reproduction and `main.yaml`, but the full `predict_fold` path (which also fits model A
first) was not executed.
