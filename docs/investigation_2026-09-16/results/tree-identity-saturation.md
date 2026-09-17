# Item 10 `tree-identity-saturation` — result (2026-09-17)

Batch C, forward-fold run. Source item: `plan.md` §1 #10 (from M `tree-identity-saturation`).
Script: `docs/investigation_2026-09-16/scripts/tree-identity-saturation.py` (reads finished artefacts only —
`experiments/experiments.jsonl`, `experiments/oof_<exp_id>.parquet`, `data/work/feat_train.parquet`; trains
nothing). Configs: `configs/inv_baseline_forward.yaml`, `configs/inv_it10_leaf5.yaml` (both new; no existing
config or `src/` file was modified).

Definitions follow plan.md §0: `y` = `TAXITIME_SEC_mvt`; `ms` = `mms` = `mvt_minus_sched` = MVT − SCHED;
"unmatched" = routing definition (`mvt_minus_aobt3` null → model B); sched-filled = |y − ms| ≤ 60 s;
2 h = 7,200 s, 4 h = 14,400 s, 12 h = 43,200 s.

## 1. Hypothesis (as stated in plan.md §1 #10)

"Model B saturates at ~35–40k because trees cannot represent y = mms and `min_data_in_leaf` 50 exceeds the
number of LIRF training rows with mms > 43,200 (29)."
Listed as unverified: "whether a smaller `min_data_in_leaf` restores pred/mms ≈ 1 on LIRF without harming
non-LIRF rows (see item 7)."

Sub-claims tested here:
(A) model B's unmatched predictions saturate at ≈ 35–40 k;
(B) `unmatched_min_data_in_leaf` = 50 exceeds the count of LIRF unmatched training rows with ms > 43,200;
(C) `unmatched_min_data_in_leaf` = 5 restores pred/ms ≈ 1 on LIRF;
(D) it does so without harming non-LIRF rows;
(E) the mechanism is "a tree cannot represent the identity y = ms".

## 2. Procedure actually executed

1. **Baseline run.** `configs/inv_baseline_forward.yaml` = `configs/main.yaml` verbatim + `folds: [forward]` +
   `params: {num_threads: 2}`. Command:
   `POLARS_MAX_THREADS=2 .venv/bin/python -m prc.pipeline --config configs/inv_baseline_forward.yaml --skip-predict`.
2. **Candidate run.** `configs/inv_it10_leaf5.yaml` = the baseline file with one line changed:
   `unmatched_min_data_in_leaf: 50` → `5`. `unmatched_num_leaves` stays 127. Same command, run afterwards, never
   concurrently.
3. `make compare A=inv_baseline_forward-67e76a5dfc B=inv_it10_leaf5-1ea7a3d3f5` (`POLARS_MAX_THREADS=1`).
4. The script above for per-group RMSE, pred/ms quantiles and the per-row LIRF table, plus its `--train-counts`
   mode for the training-side band counts.

Config diff (the only difference between the two YAML files besides `name:`):

```
--- configs/inv_baseline_forward.yaml
+++ configs/inv_it10_leaf5.yaml
-unmatched_min_data_in_leaf: 50
+unmatched_min_data_in_leaf: 5
```

Resolved parameters (printed from `TrainConfig` after the `pipeline.py` `params` merge; model B = `for_unmatched()`):

| | model A | model B (baseline) | model B (item 10) |
|---|---|---|---|
| num_leaves | 127 | 127 | 127 |
| min_data_in_leaf | 50 | 50 | **5** |
| num_rounds (cap) | 2000 | 2000 | 2000 |
| learning_rate | 0.05 | 0.05 | 0.05 |
| linear_tree | absent | absent | absent |
| num_threads | 2 | 2 | 2 |
| n_seeds / early_stopping / inner_holdout_months | 3 / 100 / 1 | 3 / 100 / 1 | 3 / 100 / 1 |

Records: `experiments/experiments.jsonl`, ids **`inv_baseline_forward-67e76a5dfc`** (git_sha 4bc9cb4, 1,186.5 s)
and **`inv_it10_leaf5-1ea7a3d3f5`** (git_sha c1ba6f0, 1,070.2 s); OOF files
`experiments/oof_inv_baseline_forward-67e76a5dfc.parquet` and `experiments/oof_inv_it10_leaf5-1ea7a3d3f5.parquet`
(each 2,085,047 rows, 328,009 non-null = months 202511 + 202512). The two git shas differ because other agents
committed during the runs; `git diff 3d7cfc7..c1ba6f0 -- src/` is empty, so both runs are the same model code.
Forward fold = train Jan–Oct 2025, inner holdout (early stopping) = October, evaluate Nov+Dec 2025.

## 3. Results

### 3.1 Training-side counts (script `--train-counts`, no model involved)

| set | unmatched rows | LIRF unmatched | LIRF ms > 43,200 | LIRF ms ∈ (14,400, 43,200] | LIRF ms ≥ 7,200 | LIRF y ≥ 43,200 | all-unmatched ms > 43,200 |
|---|---|---|---|---|---|---|---|
| all 12 months | 22,470 | 1,488 | **29** | **98** | 627 | 30 | 75 |
| forward training side (Jan–Oct 2025) | 19,465 | 1,348 | **25** | 92 | 569 | 26 | 61 |

M's 29 and 98 are reproduced exactly on the 12-month set. The model actually fitted in this item sees the
Jan–Oct subset: 25 rows above 12 h.

### 3.2 Forward-fold metrics (328,009 scored rows)

| quantity | baseline (50) | item 10 (5) | change |
|---|---|---|---|
| `rounds_unmatched` | **45** | **49** | +4 |
| `rounds_matched` | 939 | 939 | 0 |
| RMSE total (`rmse_full`) | **284.7038** | **234.9741** | −49.73 |
| RMSE clean | 224.5794 | 210.3057 | −14.27 |
| RMSE matched (n = 325,004) | **201.6633** | **201.6633** | 0.0000 |
| RMSE unmatched (n = 3,005) | **2,109.32** | **1,276.05** | −833.27 |
| RMSE LIRF unmatched (n = 140) | 8,578.38 | 5,242.08 | −3,336.30 |
| RMSE non-LIRF unmatched ms ≥ 2 h (n = 596) | 2,132.29 | 943.72 | −1,188.57 |
| top-20 share of fold SSE | 0.4464 | 0.1941 | −0.2523 |
| unmatched predictions > 14,400 | 17 | 9 | −8 |
| unmatched predictions > 43,200 | 0 | 4 | +4 |
| max unmatched prediction | 34,539.68 | 68,801.03 | +34,261 |
| median unmatched prediction | 1,079.51 | 1,062.73 | −16.8 |

Model A is untouched by the change and its RMSE is identical to four decimals, as the design predicts
(`for_unmatched()` is applied to model B only).

### 3.3 pred/ms on LIRF unmatched rows with ms ≥ 2 h (the item's requested statistic)

58 such rows are scored in the forward fold (45 of them sched-filled). M's reference figure — median 0.83,
p10–p90 [0.55, 1.10] — was measured on the sched-filled rows of the 3-fold `oof_main-48c19bc830`, so it is not
the same row set as any column below.

| row set | n | statistic | baseline (50) | item 10 (5) |
|---|---|---|---|---|
| LIRF unmatched, ms ≥ 2 h | 58 | median pred/ms | **0.7021** | **0.7383** |
| | | p10 | 0.4248 | 0.4564 |
| | | p90 | 0.8579 | 0.8903 |
| | | RMSE | 13,119.34 | 7,789.21 |
| | | max pred | 34,539.68 | 68,801.03 |
| LIRF unmatched, ms ≥ 2 h, sched-filled | 45 | median pred/ms | **0.7893** | **0.7473** |
| | | p10 | 0.5827 | 0.5586 |
| | | p90 | 0.8779 | 0.8850 |
| | | RMSE | 4,984.98 | 5,012.19 |
| | | max pred | 32,399.49 | 48,922.99 |
| non-LIRF unmatched, ms ≥ 2 h | 596 | median pred/ms | 0.1418 | 0.1359 |
| | | p10 / p90 | 0.0782 / 0.2122 | 0.0730 / 0.2121 |
| | | max pred | 16,879.98 | 6,287.75 |

The ten largest-ms rows of that set (`y`, `ms`, then the two predictions):

| ym | y | ms | baseline | item 10 |
|---|---|---|---|---|
| 202512 | 87,605 | 86,102 | 34,093.3 | 68,801.0 |
| 202511 | 87,598 | 60,599 | 34,539.7 | 64,488.7 |
| 202511 | 87,480 | 56,520 | 32,351.7 | 50,583.1 |
| 202512 | 47,767 | 47,763 | 32,399.5 | 48,923.0 |
| 202512 | 39,120 | 39,120 | 20,890.4 | 14,545.3 |
| 202512 | 38,633 | 38,636 | 29,713.7 | 30,395.5 |
| 202511 | 37,736 | 37,738 | 25,511.5 | 28,445.8 |
| 202511 | 29,282 | 29,281 | 24,317.9 | 26,936.9 |
| 202512 | 19,376 | 19,378 | 19,075.8 | 17,026.3 |
| 202512 | 14,471 | 14,465 | 10,065.3 | 9,156.9 |

### 3.4 `make compare` (verbatim)

```
.venv/bin/python -m prc.compare inv_baseline_forward-67e76a5dfc inv_it10_leaf5-1ea7a3d3f5
A = inv_baseline_forward-67e76a5dfc
B = inv_it10_leaf5-1ea7a3d3f5
rows = 328,009
top-20 share of B's squared error: 0.194

 dropped         A         B     diff                 CI95   P(B)
       0    284.70    234.97   -49.73    [ -91.52, -13.99]  1.000
      20    212.20    213.33    +1.13    [  +0.17,  +2.17]  0.008
     200    198.82    199.02    +0.20    [  -0.02,  +0.42]  0.038
    2000    177.39    177.44    +0.05    [  -0.02,  +0.12]  0.070

month-level bootstrap (2 months): -49.73 s [-55.19, -44.23]  P(B better) = 1.000

   matched: n=325,004  A=   201.7  B=   201.7
 unmatched: n=  3,005  A=  2109.3  B=  1276.0

VERDICT: REJECT - tail-only: -49.73 s on all rows, but +0.20 s [-0.02, +0.42] once the worst 200 rows are removed - the effect is those rows, and the ranking set has different ones
```

## 4. Verdict per sub-claim

| sub-claim | verdict | deciding observation |
|---|---|---|
| (A) model B saturates at ≈ 35–40 k | **CONFIRMED** on this fold | baseline's largest unmatched OOF prediction is 34,539.68 and 0 predictions exceed 43,200, while three scored rows have y ≈ 87.5 k and one has ms = 86,102 |
| (B) `min_data_in_leaf` 50 > number of LIRF training rows with ms > 43,200 | **CONFIRMED** (arithmetic) | 29 on all 12 months (M's figure reproduced), 25 on the forward training side; both < 50 |
| (C) `min_data_in_leaf` 5 restores pred/ms ≈ 1 on LIRF | **REFUTED as stated** | median pred/ms moves 0.7021 → 0.7383 on the 58 rows and 0.7893 → **0.7473** (away from 1) on the 45 sched-filled rows; p90 0.858 → 0.890. The ceiling moves (max 34,540 → 68,801; 4 predictions above 12 h where the baseline had 0), the central ratio does not |
| (D) without harming non-LIRF rows | **CONFIRMED** on this fold | non-LIRF unmatched ms ≥ 2 h RMSE 2,132.29 → 943.72, median pred/ms 0.1418 → 0.1359, max prediction 16,880 → 6,288; matched RMSE identical to 4 dp; no non-LIRF row is among the 4 predictions above 12 h |
| (E) mechanism "a tree cannot represent y = ms" | **UNDETERMINED** | the run changes leaf size only; it cannot separate "leaf minimum too large" from "piecewise-constant fit of an identity". The sched-filled ratio staying at 0.75 with leaves of 5 rows is consistent with the mechanism but does not isolate it (item 11's `linear_tree` is the variant that addresses representation) |
| unmatched RMSE effect on the forward fold | **fact**: 2,109.32 → 1,276.05 (−833.27), total 284.70 → 234.97 (−49.73) | §3.2 |
| does the change pass the repo's decision rule | **no** | `make compare` REJECT, "tail-only": already +1.13 s [+0.17, +2.17] at K = 20 and +0.20 s [−0.02, +0.42] at K = 200 |

## 5. Not determinable locally

- **CV → LB transfer.** Nothing here measures the leaderboard. `decision-rule-blind-to-tail.md` shows the
  adopt/reject rule consults only the K = 0 and K = 200 entries, and that at K = 200 the dropped rows are
  exactly the rollover / LIRF sched-filled rows this change moves; the −49.73 s therefore lives entirely on
  rows the rule discards by construction, and whether those rows recur in the 2026 ranking set is unknown
  (the ranking labels do not exist locally — `lirf-regime-tail-count.md` §5).
- **The forward fold's known limits** (`fold-shape-vs-ranking.md` §3.2, §5): no fold has the production shape;
  the forward fold trains on 10 months, predicts Nov+Dec 2025 at horizons 1–2 months, never crosses a year
  boundary and never predicts a January or a July. Its 58 LIRF unmatched ms ≥ 2 h rows are not the ranking
  set's 181, and its 3 rollover rows are not the ranking set's unknown number.
- Whether 5 is a good value: no sweep was run (only 50 and 5), and no second seed set beyond the config's
  `n_seeds: 3` refit.
- Whether the same change behaves the same way on the `jan` / `jul` folds or in a 12-month production fit
  (rounds differ per fold: 45 here vs 199–658 reported for production in plan.md §1 #23).

## 6. Facts vs inference

**Facts (measured in this run or read from the artefacts named above).** Every number in §3, including: the
two exp_ids and their OOF files; `rounds_unmatched` 45 → 49 and `rounds_matched` 939 → 939; the forward-fold
RMSEs 284.7038 / 234.9741 total, 201.6633 / 201.6633 matched, 2,109.32 / 1,276.05 unmatched; the pred/ms
quantiles; the prediction counts above 4 h (17 → 9) and 12 h (0 → 4); the maximum unmatched prediction
34,539.68 → 68,801.03; the training-side band counts (29 / 98 all-year, 25 / 92 forward); the `make compare`
block quoted verbatim; the resolved parameter table (printed from `TrainConfig.for_unmatched()` /
`lgb_params()`); that `git diff 3d7cfc7..c1ba6f0 -- src/` is empty.

**Inference (not established by this run).**
- That the baseline's 34.5 k ceiling on this fold is the same phenomenon as the 28–40 k ceiling M measured on
  v8's *ranking* predictions: the row sets differ and the ranking labels are unknown.
- That the drop in unmatched RMSE is caused by the four rows now predicted above 12 h rather than by the
  broader redistribution: the K = 20 line (+1.13 s, CI excluding 0 on the worse side) shows the sign flips once
  20 rows are removed, but no per-row attribution of the 833 s was computed.
- That the improvement in non-LIRF unmatched RMSE (2,132 → 944) is a property of the leaf size rather than of
  the 4 extra rounds; the two changes are confounded within a single run.
- That any of this transfers to the leaderboard. No recommendation is made.
