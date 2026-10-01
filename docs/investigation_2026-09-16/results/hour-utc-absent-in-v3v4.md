# Item 22 `hour-utc-absent-in-v3v4` — result (2026-09-16)

Script: `docs/investigation_2026-09-16/scripts/hour-utc-absent-in-v3v4.py` (model B only, 2 early-stop fits +
4 refits, 74 s total; `POLARS_MAX_THREADS=1`, LightGBM `num_threads: 2`, lazy column-selected scans).
Definitions follow plan §0: "unmatched" = routing definition (`mvt_minus_aobt3` null), 22,470 training / 5,290
ranking rows; `ms` = `mvt_minus_sched`; 12 h = 43,200 s.

## 1. Hypothesis as stated (plan.md §1 item 22, sources P 4; M `v5-verdict-confounded`)

v3/v4's model B was trained without `hour_utc` (weather-join bug, handoff §7-4, fixed in `34808da`) and **that
absence, not weather or linear_tree, produced the > 12 h LIRF predictions**. Decision rule given in the plan's
procedure: fit model B with `exclude_features [hour_utc]`, weather on, 127/50, one seed, old seed semantics, and
count unmatched ranking predictions > 43,200 — "≥ ~15 confirms; 2 refutes" (v3 has 16, v4 has 18, every
current-code build has 2).

## 2. Procedure actually executed

Model B's data path reproduced from the `prc.train` helpers (identical to item 21's §2 steps 1-3, except that the
**weather columns are kept**, as in v3/v4):

1. `configs/main.yaml` resolved the way `prc.pipeline.main` does, then `.for_unmatched()`; `n_seeds` = 1;
   `num_threads` = 2. Weather is on in `main.yaml` and the cached `feat_train.parquet` / `feat_rank.parquet`
   carry the 20 weather columns.
2. Encodings fitted with `fit_encodings` on all 2,085,047 labelled training rows (ref 6,434 / arh 1,024 / apt 10 /
   op 1,281), attached with `attach_encodings`; unmatched training 22,470, unmatched ranking 5,290; row order
   asserted unchanged.
3. Two matrices through `_matrix` on the stacked [train, ranking] frame:
   - **WITH `hour_utc`** — `cols_b` per train.py:350-356 = **83 numeric** + 10 categorical (the 65 of item 21 plus
     18 weather columns; `frozen_wx` / `frozen_wx_prev1h` are not numeric-typed in the cache and drop out);
   - **WITHOUT** — the same with `exclude_features = ("hour_utc",)` → **82 numeric** + 10 categorical.
4. **Early stopping as in the code**, not a fixed round count: `_inner_holdout_mask(tr, 1)` → the latest training
   month (December 2025) = **1,649 of 22,470 rows**; `lgb.early_stopping(100)` with `num_boost_round` = 2,000 on
   the remaining 20,821; `best_iteration` then scaled by `n_train / (n_train − n_hold)` and capped, exactly as
   train.py:293-295.
5. Each variant refitted on all 22,470 rows under **both** seed conventions, so that item 21's effect is not
   confounded with this one: `old` = `params` with `seed` only (what v3/v4's code did), `head` = + `bagging_seed`
   / `feature_fraction_seed`.
6. Comparison files: `outputs/probe_unmatched_from_v4.parquet` (v4's unmatched predictions), `outputs/det_a.parquet`
   (v8), `outputs/main.parquet` (v7), `outputs/v2_two_clock.parquet` (v2). Tail counts are taken on the raw float
   predictions; `>` and `≥` give the same counts at these thresholds.

Resolved LightGBM parameters (both variants, both fits):

```
{"bagging_fraction":0.8, "bagging_freq":1, "deterministic":true, "feature_fraction":0.8,
 "force_row_wise":true, "lambda_l2":1.0, "learning_rate":0.05, "min_data_in_leaf":50,
 "num_leaves":127, "num_threads":2, "objective":"regression", "seed":42, "verbose":-1}
refit "head" adds: {"bagging_seed":42, "feature_fraction_seed":42}
```

**Not reproduced from the v3/v4 build** (see `results/v4-config-not-reproducible.md` §3.3): `linear_tree` (v4 had
it), the 1,500-round cap, the absence of `force_row_wise`, the v3/v4-era METAR cache and the v3/v4-era weather
join itself. This run removes `hour_utc` from today's matrix; it does not re-execute the buggy join.

## 3. Results

### 3.1 The two fits

| | WITH `hour_utc` | WITHOUT `hour_utc` |
|---|---|---|
| numeric / categorical columns | 83 / 10 | 82 / 10 |
| inner holdout (Dec 2025) | 1,649 rows | 1,649 rows |
| `best_iteration` | 542 | 599 |
| rounds after scaling (refit) | 585 | 646 |
| holdout RMSE at iteration 1 | 3,072.9 | 3,073.1 |
| **holdout RMSE at `best_iteration`** | **1,694.2** | **1,653.6** |
| gain share of `hour_utc` (early-stop fit) | 0.0024 | — |
| gain share of `hour_local` / `minute_of_day_utc` | 0.0024 / 0.0060 | 0.0032 / 0.0076 |
| gain share of `mvt_minus_sched` (+ `_slog`) | 0.417 (+0.105) | 0.457 (+0.072) |

`hour_utc` null rate is 0 in both frames; its correlation with `hour_local` on the 22,470 unmatched training rows
is 0.597. The column carries 0.24 % of the early-stop fit's gain.

### 3.2 Tail counts on the 5,290 unmatched ranking rows

| prediction set | median | mean | p99 | max | n > 7,200 | n > 14,400 | **n > 43,200** |
|---|---|---|---|---|---|---|---|
| WITH `hour_utc`, old seeds | 1,075.9 | 1,597.2 | 10,179.0 | 65,556.9 | 103 | 36 | **2** |
| WITH `hour_utc`, HEAD seeds | 1,083.2 | 1,593.9 | 10,057.5 | 66,076.5 | 100 | 35 | **2** |
| **WITHOUT `hour_utc`, old seeds** | 1,092.6 | 1,606.4 | 10,218.2 | 68,331.4 | 97 | 36 | **2** |
| WITHOUT `hour_utc`, HEAD seeds | 1,087.2 | 1,613.0 | 10,096.4 | 67,697.1 | 99 | 34 | **2** |
| v2 (file, LB 312.02) | 1,062.5 | 1,611.7 | 10,256.2 | 68,253.0 | 104 | 33 | 2 |
| v7 (file, 342.59) | 1,082.0 | 1,608.4 | 10,168.6 | 66,693.0 | 100 | 33 | 2 |
| v8 (file, 343.80) | 1,066.0 | 1,608.0 | 10,184.1 | 68,581.0 | 101 | 35 | 2 |
| **v4 (file, 305.03)** | 1,112.0 | 1,702.5 | 9,457.5 | **108,552.0** | 83 | 31 | **18** |

The plan's decision rule ("≥ ~15 confirms; 2 refutes") lands on **2** for the configuration it specifies
(WITHOUT `hour_utc`, weather on, 127/50, one seed, old seed semantics). All four variants give 2; every LIRF
count is the same 2 (both > 43,200 rows are LIRF in every variant, as in v4's 18/18).

### 3.3 RMS differences

| pair | overall | ms ≥ 2 h (1,721) | ms ≥ 12 h (25) | LIRF (383) |
|---|---|---|---|---|
| WITH vs WITHOUT, old seeds (the `hour_utc` effect) | 300.8 | 416.9 | 1,062.1 | 711.4 |
| WITH vs WITHOUT, HEAD seeds (the same effect) | 338.4 | 478.5 | 1,200.7 | 841.8 |
| old vs HEAD seeds, WITH (item 21's effect) | 319.0 | 436.7 | 833.1 | 785.3 |
| old vs HEAD seeds, WITHOUT (item 21's effect) | 339.8 | 492.3 | 1,177.9 | 730.7 |
| WITH, old seeds vs v4 | 1,783.2 | 3,052.6 | 20,730.7 | 6,231.2 |
| **WITHOUT, old seeds vs v4** | **1,802.3** | **3,085.5** | **21,141.7** | **6,330.1** |
| WITH, HEAD seeds vs v8 | 397.3 | 563.3 | 1,610.8 | 966.9 |
| WITHOUT, HEAD seeds vs v8 | 393.2 | 554.1 | 921.3 | 890.0 |
| v8 vs v4 (files, for scale) | 1,735.5 | 2,967.0 | 20,398.9 | 6,127.8 |

Removing `hour_utc` moves the predictions about as far as changing the seed keys does (301 vs 319 s under old
seeds) and **away** from v4, not towards it: 1,802.3 without vs 1,783.2 with, against a v8-to-v4 distance of
1,735.5. Both variants stay ~390-400 s from v8, which is the weather-on/weather-off difference (v8 was built
without weather) plus the seed effect.

### 3.4 The 11 ranking LIRF unmatched rows with ms ≥ 12 h

| MVT_ID | mo | ms | rwy | h_utc | h_loc | WITH/old | WITHOUT/old | WITH/head | WITHOUT/head | v4 | v8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 213145579 | 7 | 111,654 | 25 | 21 | 23 | 40,697 | 39,507 | 39,743 | 39,340 | **108,552** | 40,267 |
| 213152353 | 7 | 87,484 | 16R | 10 | 12 | 33,272 | 31,546 | 33,133 | 34,234 | **79,686** | 34,552 |
| 213145102 | 7 | 61,322 | 16R | 13 | 15 | 65,557 | 65,846 | 66,025 | 67,697 | 60,845 | 68,581 |
| 213154026 | 7 | 60,656 | 25 | 8 | 10 | 31,851 | 31,499 | 31,113 | 32,185 | **66,225** | 33,197 |
| 202680942 | 1 | 56,520 | 25 | 11 | 12 | 34,015 | 33,881 | 32,697 | 33,358 | **60,401** | 34,236 |
| 202687258 | 1 | 53,520 | 25 | 20 | 21 | 30,236 | 29,416 | 30,257 | 31,653 | **55,198** | 31,617 |
| 202685083 | 1 | 48,841 | 25 | 14 | 15 | 37,520 | 38,009 | 35,719 | 38,132 | **52,545** | 37,704 |
| 213143319 | 7 | 48,776 | 25 | 3 | 5 | 36,469 | 36,668 | 35,562 | 36,893 | **50,937** | 38,739 |
| 213143669 | 7 | 48,539 | 25 | 21 | 23 | 34,000 | 33,720 | 32,228 | 33,971 | **54,226** | 33,640 |
| 213142469 | 7 | 47,705 | 25 | 7 | 9 | 35,733 | 36,918 | 34,020 | 36,656 | **53,459** | 37,306 |
| 213153677 | 7 | 47,337 | 16R | 9 | 11 | 65,189 | 68,331 | 66,076 | 67,274 | 53,155 | 66,778 |

On the 9 rows where v4 predicts ≈ `ms` (50,937-108,552), every variant of this run predicts 29,416-40,697 — the
same band as v2/v7/v8 — whether or not `hour_utc` is in the matrix. On the two 16R rows both families predict
53-68 k and the ordering between them is not stable.

18 ranking rows have any variant or v4 above 43,200 (full table in the script output and in `results.json`); all
18 are LIRF and 11 of them have `ms` ≥ 12 h. Across those 18 rows the WITH-minus-WITHOUT difference never exceeds
3,143 s (3,195 s under HEAD seeds), while the largest gap between a variant and v4 is 69,212 s.

## 4. Verdict per sub-claim

- **(A) v3/v4's model B was trained without `hour_utc`: CONFIRMED** — as a code fact only, inherited from
  `results/v4-config-not-reproducible.md` §3.3-4 and the comment at `weather.py:166-168`. Nothing in this run
  re-tests it; the v3/v4 join was not executed.
- **(B) That absence produced v3/v4's > 12 h LIRF predictions: REFUTED.** The plan's own decision rule was
  applied to the configuration the plan specified (no `hour_utc`, weather on, 127/50, one seed, old seed
  semantics) and returns **2** predictions > 43,200, not ≥ 15. Dropping `hour_utc` moves the predictions
  *away* from v4 (RMS 1,802.3 vs 1,783.2 with it) and changes none of the 11 LIRF ms ≥ 12 h rows into the v4
  regime: the largest, ms = 111,654, is predicted 39,507 without `hour_utc` against v4's 108,552.
- **(C) "Not weather or linear_tree" — the exclusive part of the hypothesis: REFUTED for `hour_utc` as the sole
  cause; the alternatives are UNDETERMINED here.** This run holds weather on and `linear_tree` off; the residual
  ~1,800 s distance to v4 is unexplained by the factor tested. `linear_tree` on model B is plan item 11, and the
  v3/v4-era METAR cache and join are not recoverable (item 20 §5).
- **(D) `hour_utc` is a low-value feature for model B: CONFIRMED (measurement, weak effect).** 0.24 % gain share;
  removing it *lowers* the December holdout RMSE from 1,694.2 to 1,653.6 and raises `best_iteration` from 542 to
  599. This is one holdout of 1,649 rows and one seed, so the sign is not established (see §5).
- **(E) The `hour_utc` effect is confounded with item 21's seed change (P's note): CONFIRMED as a magnitude
  statement.** Under one seed convention the `hour_utc` effect is 300.8-338.4 s RMS while the seed-key effect on
  the same rows is 319.0-339.8 s: the two are the same size, so a single before/after comparison across the
  d182fc1 boundary cannot separate them. This run separates them by crossing the two factors.

## 5. What cannot be determined locally

- Whether the December-holdout improvement from dropping `hour_utc` (1,694.2 → 1,653.6) is real: it is one
  holdout month, 1,649 rows containing 6 rows ≥ 4 h (plan §4-3), one seed, and item 23 questions this holdout's
  resolution generally. No CI was computed and no fold-wise ablation was run.
- What actually produced v4's 18 predictions > 43,200. Candidates left standing: `linear_tree` on model B
  (item 11), the v4-era METAR cache / weather join, the 1,500-round cap, the absence of `force_row_wise`, or the
  library versions of 2026-09-10 — none is reproducible from this repository (item 20 §5).
- Any leaderboard consequence: the 5,290 unmatched ranking rows have no labels here.
- Whether the same conclusion holds for model A: only model B was fitted.

## 6. Facts vs inference

**Facts (measured in this run):** the two matrices' widths (83+10 / 82+10); the holdout size 1,649;
`best_iteration` 542 / 599 and refit rounds 585 / 646; holdout RMSE 1,694.2 / 1,653.6; every gain share; every
tail count and RMS figure in §3.2-3.4, including the four variants' `n > 43,200` = 2; `hour_utc` null rate 0 and
its 0.597 correlation with `hour_local`.

**Inference (not measured here):** (i) that v3/v4 lacked `hour_utc` — code reading, carried over from item 20;
(ii) that this run's "WITHOUT" matrix is what v3/v4's model B saw — it is today's matrix minus one column, while
v3/v4 also differed in `linear_tree`, the round cap, `force_row_wise`, the METAR cache and the library versions,
so the refutation is of the isolated factor, not of a faithful v4 replica; (iii) that the ~390-400 s distance
from v8 is the weather columns plus the seed keys — plausible from item 21's 314.5 s but not decomposed here;
(iv) that `frozen_wx` / `frozen_wx_prev1h` are absent from `cols_b` for dtype reasons — read off
`numeric_feature_columns`, not separately tested.
