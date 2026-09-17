# Item 21 `refit-seed-semantics-changed` — result (2026-09-16)

Script: `docs/investigation_2026-09-16/scripts/refit-seed-semantics-changed.py` (model B only, three refits of 658
rounds each, 40 s total; `POLARS_MAX_THREADS=1`, LightGBM `num_threads: 2`, lazy column-selected scans).
Definitions follow plan §0: "unmatched" = routing definition (`mvt_minus_aobt3` null → model B), 22,470 training /
5,290 ranking rows; `ms` = `mvt_minus_sched`; thresholds 2 h = 7,200, 12 h = 43,200.

## 1. Hypothesis as stated (plan.md §1 item 21, sources P 2; M `seed-avg-no-effect-B`)

Commit d182fc1 made the refit pass explicit `bagging_seed` / `feature_fraction_seed`, which changes the booster
even at `n_seeds` 1 while the early-stopping fit is unchanged; v2→v8 (LB 312.02 → 343.80) kept identical model-B
rounds (658) but produced a different model. Unverified part per the plan: the magnitude on the 5,290 ranking
unmatched rows.

Sub-claims tested: (A) the early-stopping fit is unchanged by d182fc1; (B) the refit differs at `n_seeds` 1;
(C) the difference is reproducible, not float noise; (D) its magnitude on the 5,290 ranking unmatched rows;
(E) whether it accounts for the v2→v8 model-B difference; (F) whether it explains the v4 → current-code gap.

## 2. Procedure actually executed

Model B's data path was reproduced from the helpers in `prc.train`, not re-implemented:

1. `configs/main.yaml` loaded exactly as `prc.pipeline.main` does (field filter, `params` merge, list→tuple), then
   `.for_unmatched()`; `n_seeds` forced to 1; the 20 weather columns hidden through `exclude_features` (v2 and
   det_a/v8 were both built with `use_weather: false`); `num_threads` set to 2.
2. Training rows: `feat_train.parquet` filtered to `TAXITIME_SEC_mvt` not null and finite (2,085,047 rows).
   Encodings fitted on **all** labelled training rows with `fit_encodings` (ref 6,434 / arh 1,024 / apt 10 / op
   1,281 groups) and attached with `attach_encodings` to the unmatched training rows (22,470) and to the unmatched
   ranking rows of `feat_rank.parquet` (5,290); row count and `MVT_ID_mvt` order asserted unchanged.
3. `cols_b` = `visible_numeric_columns(tr, cfg)` minus the six NM-clock prefixes, exactly the list comprehension at
   train.py:350-356 → **65 numeric columns**; categoricals = the 10 `CATEGORICALS` → matrix (27,760, 75), built by
   `_matrix` on the stacked [train, ranking] frame as `_fit` does.
4. **No early stopping.** `num_boost_round` fixed at **658**, the `rounds_unmatched` recorded for both v2
   (`git show 4e807b5^:experiments/pipeline.jsonl`) and det_a/v8 (`experiments/pipeline.jsonl`). Three refits on
   the same `lgb.Dataset`:
   - (a) `params` with `seed` only — the refit as `train.py` did it before d182fc1 (`lgb.train(params, dfull, …)`);
   - (a2) the same call again — bit-identity check;
   - (b) `params` + `seed`/`bagging_seed`/`feature_fraction_seed` = 42 — the refit at HEAD (train.py:301-302) with
     `cfg.n_seeds` = 1, i.e. the `k = 0` iteration of the seed loop.
5. Predictions compared with `outputs/v2_two_clock.parquet` (v2, LB 312.02), `outputs/det_a.parquet` (v8, 343.80),
   `outputs/probe_unmatched_from_v4.parquet` (v4's unmatched predictions, LB 305.03 / probe v9 306.46) and
   `outputs/main.parquet` (v7). Submission files hold `Int32`, so comparisons against them use the pipeline's own
   output transform — `clip(lower_bound=0)` (`_assemble`, train.py:377) then `np.rint` (`write_submission`).

Resolved LightGBM parameters (printed by the script; identical for all three fits except the seed keys):

```
early-stop fit / refit (a): {"bagging_fraction":0.8, "bagging_freq":1, "deterministic":true,
  "feature_fraction":0.8, "force_row_wise":true, "lambda_l2":1.0, "learning_rate":0.05,
  "min_data_in_leaf":50, "num_leaves":127, "num_threads":2, "objective":"regression",
  "seed":42, "verbose":-1}
refit (b): the same dict + {"bagging_seed":42, "feature_fraction_seed":42}
```

(`num_leaves` 127 / `min_data_in_leaf` 50 come from `main.yaml`'s `unmatched_*` keys through `for_unmatched()`;
`num_threads` is 2 here against 4 in the recorded builds.)

## 3. Results

### 3.1 Determinism and identity with the shipped builds

| check | result |
|---|---|
| (a) run 1 vs run 2, raw float predictions | **bit-identical**, max abs difference 0.0 on all 5,290 rows |
| (a) vs `v2_two_clock.parquet` after clip+rint | **identical on all 5,290 rows**; max deviation of the raw float from the stored Int32 = 0.499929 (i.e. the rounding itself) |
| (b) vs `det_a.parquet` (v8) after clip+rint | **identical on all 5,290 rows**; max deviation 0.499938 |
| (a) vs (b), raw floats | **0 of 5,290 rows equal** |

So this run reproduces v2's model B with the old refit call and v8's model B with the new one, from the same data,
the same 658 rounds and the same `params` — the only difference being the two seed keys.

### 3.2 Magnitude of (a) − (b) on the 5,290 ranking unmatched rows

RMS difference, raw floats and in submission space (clip at 0 then rint, which is what the leaderboard saw):

| subset | n | RMS(a − b) float | RMS(a − b) submission space | RMS(a − v4) | RMS(b − v4) |
|---|---|---|---|---|---|
| all unmatched | 5,290 | 327.8 | **314.5** | 1,748.5 | 1,735.5 |
| ms ≥ 2 h | 1,721 | 455.2 | 431.4 | 2,989.0 | 2,967.0 |
| ms ≥ 12 h | 25 | 1,055.5 | 1,055.6 | 20,514.0 | 20,398.9 |
| ms < 2 h | 3,569 | 243.6 | 238.5 | 472.3 | 468.5 |

(The float/submission-space gap is the clip at 0, not the rounding: 5 of the 20 largest disagreement rows have a
negative raw prediction in one or both fits, and clipping collapses those differences.)

By airport (submission space; the last two columns are each fit's distance to v4):

| ADEP | n | RMS(a − b) | RMS(a − v4) | RMS(b − v4) |
|---|---|---|---|---|
| LIRF | 383 | **800.2** | 6,132.8 | 6,127.8 |
| EGLL | 248 | 347.1 | 1,432.9 | 1,278.4 |
| EDDM | 238 | 278.6 | 644.5 | 539.2 |
| EHAM | 1,206 | 269.3 | 593.5 | 583.9 |
| LFPG | 734 | 226.7 | 505.9 | 472.1 |
| LSZH | 659 | 216.0 | 559.3 | 560.8 |
| EDDF | 421 | 213.8 | 434.3 | 404.5 |
| LEMD | 238 | 213.6 | 461.1 | 467.1 |
| LTFM | 721 | 200.4 | 373.2 | 327.2 |
| LEBL | 442 | 183.4 | 497.5 | 470.8 |

Concentration of Σ(a − b)² (raw floats): top-20 rows 18.7 %, LIRF 43.5 %, ms ≥ 2 h rows 62.7 %. For comparison,
plan §1 item 3 / `results/lirf-regime-tail-count.md` give the v8 − v4 gap as 90.3 % LIRF and 80.4 % in 20 rows:
the seed-key difference is spread far more evenly than the v4-family difference.

### 3.3 Tail counts on the same 5,290 rows

| prediction set | median | mean | p99 | max | n > 7,200 | n > 14,400 | n > 43,200 |
|---|---|---|---|---|---|---|---|
| (a) seed only | 1,062.4 | 1,598.7 | 10,256.5 | 68,252.9 | 104 | 33 | 2 |
| (b) explicit sub-seeds | 1,066.2 | 1,596.1 | 10,183.9 | 68,580.7 | 101 | 35 | 2 |
| v2 (file) | 1,062.5 | 1,611.7 | 10,256.2 | 68,253.0 | 104 | 33 | 2 |
| v8 = det_a (file) | 1,066.0 | 1,608.0 | 10,184.1 | 68,581.0 | 101 | 35 | 2 |
| v4 (probe file) | 1,112.0 | 1,702.5 | 9,457.5 | 108,552.0 | 83 | 31 | 18 |

The seed-key change moves 3 rows across the 2 h line and 2 across the 4 h line; it moves none across the 12 h line
(2 in both) — v4's 18 are not reachable from either.

### 3.4 The 11 ranking LIRF rows with ms ≥ 12 h

| MVT_ID | mo | ms | rwy | (a) | (b) | v2 | v8 | v4 |
|---|---|---|---|---|---|---|---|---|
| 213145579 | 7 | 111,654 | 25 | 40,865 | 40,267 | 40,865 | 40,267 | 108,552 |
| 213152353 | 7 | 87,484 | 16R | 33,882 | 34,552 | 33,882 | 34,552 | 79,686 |
| 213145102 | 7 | 61,322 | 16R | 67,002 | 68,581 | 67,002 | 68,581 | 60,845 |
| 213154026 | 7 | 60,656 | 25 | 31,617 | 33,197 | 31,617 | 33,197 | 66,225 |
| 202680942 | 1 | 56,520 | 25 | 34,836 | 34,236 | 34,836 | 34,236 | 60,401 |
| 202687258 | 1 | 53,520 | 25 | 32,309 | 31,617 | 32,309 | 31,617 | 55,198 |
| 202685083 | 1 | 48,841 | 25 | 37,301 | 37,704 | 37,301 | 37,704 | 52,545 |
| 213143319 | 7 | 48,776 | 25 | 37,077 | 38,739 | 37,077 | 38,739 | 50,937 |
| 213143669 | 7 | 48,539 | 25 | 34,887 | 33,640 | 34,887 | 33,640 | 54,226 |
| 213142469 | 7 | 47,705 | 25 | 36,278 | 37,306 | 36,278 | 37,306 | 53,459 |
| 213153677 | 7 | 47,337 | 16R | 68,253 | 66,778 | 68,253 | 66,778 | 53,155 |

(a) equals the v2 column and (b) equals the v8 column row for row, as in §3.1. Both fits sit 20–21 k RMS from v4
on these rows; the (a) − (b) spread here is ±1,600 s.

## 4. Verdict per sub-claim

- **(A) The early-stopping fit is unchanged by d182fc1: CONFIRMED** (code reading, not re-measured here). In both
  versions the early-stop `lgb.train` receives `params` = `cfg.lgb_params()` with no sub-seed keys
  (train.py:286-292 at HEAD, identical text at 9f31cef); d182fc1's diff touches only the refit block. This run
  fixed the rounds at 658 and therefore ran no early stopping at all.
- **(B) The refit differs at `n_seeds` = 1: CONFIRMED.** 0 of 5,290 rows equal; RMS 327.8 raw / 314.5 in
  submission space. The code comment at train.py:299 ("One seed is the default, and then this is the single fit it
  was before") is false as written.
- **(C) The difference is reproducible, not float noise: CONFIRMED.** (a) repeated gives bit-identical
  predictions (max abs diff 0.0), so the 314.5 s is entirely attributable to the two seed keys.
- **(D) Magnitude on the 5,290 ranking unmatched rows: MEASURED** — 314.5 s overall, 431.4 on the 1,721 ms ≥ 2 h
  rows, 1,055.6 on the 25 ms ≥ 12 h rows, 800.2 at LIRF, 183–347 at every other airport.
- **(E) It accounts for the whole v2 → v8 model-B difference: CONFIRMED.** (a) reproduces
  `v2_two_clock.parquet` and (b) reproduces `det_a.parquet` on all 5,290 rows to the Int32 the files store, from
  one data path, one round count and one `params` dict. P's independently measured RMS(v2 − v8) = 314.5 on
  unmatched rows is reproduced exactly as RMS(a − b). Everything else that changed between the two builds —
  `force_row_wise`, `for_unmatched()`, the refactor into `_assemble`, `num_threads` 4 → 2 here — left model B's
  booster unchanged on this data, because otherwise the two identities could not both hold.
- **(F) It explains the v4 → current-code regression: REFUTED.** Both fits sit at 1,748.5 (a) and 1,735.5 (b)
  RMS from v4, i.e. the seed change moves the predictions ~2 % of the way along a 1,736 s gap, and in neither
  direction systematically (it is closer to v4 at 6 airports and further at 4). Neither fit emits more than 2
  predictions > 43,200 against v4's 18.

## 5. What cannot be determined locally

- Whether the 314.5 s difference costs or gains leaderboard points: the labels of the 5,290 unmatched ranking
  rows are unknown. v2 (312.02) and v8 (343.80) differ by 31.8 s on the leaderboard, but they also differ in
  model A (P: matched RMS 38.4) and the probe identity in `results/lirf-regime-tail-count.md` attributes the
  v4 → v8 unmatched-side gap, not the v2 → v8 one.
- Which of the two seed conventions is better in expectation: that needs either labels or a multi-seed study,
  and one draw of each is one sample.
- Whether the same identity holds for model A (not fitted here; P records model A's rounds as *not* reproducible
  across det_a/det_b, 1126/1146/1131 vs 1273/1034/1173).
- Whether v2's build used exactly this matrix: the reproduction is evidence that it did, but the v2-era code was
  not executed in this run (see §6).

## 6. Facts vs inference

**Facts (measured in this run):** every number in §3, including the two exact identities, the 0.0 max difference
between (a) run 1 and run 2, the 0/5,290 equality between (a) and (b), all RMS tables, tail counts and the 11-row
table; the resolved parameter dicts in §2; `cols_b` = 65 numeric + 10 categorical columns; 22,470 / 5,290 rows.

**Inference (not measured here):** (i) that the early-stopping fit is unaffected — read from the two versions of
`_fit`, not executed, since this run pinned the rounds; (ii) that `force_row_wise`, `for_unmatched()` and the
`num_threads` change are booster-neutral for model B — deduced from the two identities holding simultaneously,
not tested one factor at a time; (iii) that `outputs/v2_two_clock.parquet` is the file uploaded as v2 and
`outputs/det_a.parquet` the one uploaded as v8 — taken from `experiments/submissions.jsonl`; (iv) that the v2-era
`cols_b` was the same 65-column list — supported by the exact reproduction, but the 9f31cef code path was not run.
