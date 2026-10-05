# Hyperparameter retune of the pooled anchored LGB+L2 member (model A) — pre-registration (2026-10-02)

**Status (2026-10-03): done. Member ADOPT (−2.41 s [−2.85, −2.02], K=200 −3.08 [−3.35, −2.85]); blend
`hybrid_blend_l2_pa` → `hybrid_blend_l2_pa_opt` ADOPT (−0.89 s [−1.10, −0.72], K=200 −1.17 [−1.31, −1.03]);
`outputs/hybrid_blend_l2_pa_opt.parquet` written and validated. Nothing uploaded or submitted.** §0 was written and
committed (3026785) before any tuning trial, confirmation or blend run. Results will be appended as §1+ without editing §0 (except this status line and the one measured
number §0.4 asks to be filled in, the trial budget N, which is fixed by a rule stated here before it is measured).

## 0. Pre-registration

Full test suite before any run: 251 passed (`logs/retune/pytest_full.txt`).

### 0.1 Why

`public-code-deep-dive-2026-09-30.md` (row 4 of its lever table and the v76 entry): kind-mango reported −0.88 s on
the board at v76 for `linear_tree` + `linear_lambda` 15 + an Optuna retune (613 leaves, lr 0.015), bundled, so the
retune's own share is unknown. `linear-tree.md` REJECTED `linear_tree` here, so this tests the remaining part: a
retune of the plain-tree parameters. Our LightGBM parameters (lr 0.05, 127 leaves, `min_data_in_leaf` 50,
`feature_fraction` 0.8, bagging 0.8/1, `lambda_l2` 1) were set early and never searched jointly. Prior capacity
facts (`model-a-ceiling.md` §5.3): 31 / 127 / 511 leaves gave lane-A 213.229 / 213.125 / 213.376, so capacity
alone is flat; a joint search over regularisation and step size has not been done. No competitor code was opened.
**Prior: a gain on the member of ≤ 0.5 s; an inconclusive confirmation is a likely outcome.**

### 0.2 What is tuned, and the code

1. **Only the pooled anchored LGB+L2 member** `configs/main_flseq_osm_nolirf_l2.yaml`
   (`main_flseq_osm_nolirf_l2-ef204b44ff`), and only its **model A** (matched lane). CatBoost is not retuned; the
   plain-y and per-airport members are only rebuilt with the winning parameters if §0.5 adopts them.
2. New `TrainConfig.matched_params` (dict, default `{}`, out of the hash when empty): LightGBM parameters merged
   over `params` for model A only (`TrainConfig.for_matched()`); model B always uses `params`. Needed because
   `params` alone would also move model B (`for_unmatched` copies every key except the two capacity overrides).
   Refused under CatBoost and for the keys in `MATCHED_PARAMS_REFUSED` (objective, seed, determinism, threads,
   verbosity, linear-tree keys). With it empty `predict_fold` makes exactly the calls it made before.
3. `tests/test_matched_params.py`: every recipe in `configs/` at 548538c (125) keeps its experiment id and the
   resolved model A / model B parameters byte for byte; the default, plain-y and per-airport paths give
   predictions byte-identical to 548538c on a fixed synthetic fold; with an override, model B's predictions are
   byte-identical and model A's move, and the override equals setting the same `params` on model A; refusals; any
   `*_opt.yaml` recipe equals its base plus `matched_params` only; the retune script's base config, search space,
   trial-0 = current params, determinism of the sampler and the tuning fold's months.
   `tests/test_per_airport_member.py`'s recipe-list guard is widened to ignore `*_opt.yaml` (they are checked here).
4. `scripts/retune_lgb.py` — the search (below).

### 0.3 Tuning split, objective, search, budget

**Optuna is not installed in `.venv` and is not in `requirements-lock.txt`/`pyproject.toml`; nothing is installed.**
The search is a plain random search (`scripts/retune_lgb.py`).

- **Tuning split `tune_oct`: train 2025-01 … 2025-09, validate 2025-10.** The comparison folds validate on 2025-01
  (`jan`), 2025-07 (`jul`, `jul_fwd`) and 2025-11+12 (`forward`). October is validated by none of them, so the
  selection never reads a row the confirmation is scored on. (Jan+jul tuning would reuse two of the three pooled
  comparison months; any split that validates on a comparison month has the same problem.) It is forward-shaped
  like the real task (train only on what came before) and October is the month nearest the forward fold. The
  tuning fold does train on January and July labels; that is training, not selection, and does not bias the
  confirmation. Early stopping uses the recipe's inner holdout (the latest training month, 2025-09), as always.
- **Objective:** RMSE of the member's model-A prediction (proxy + g, zero floor, LOBT bound, exactly as
  `predict_fold`) on October's **matched** rows (`mvt_minus_aobt3` non-null; 183,761 rows labelled in Oct, minus
  any unlabelled). Model B is not fitted in a trial. 3 seeds, 2,000-round cap, early stopping 100 — the recipe's
  own settings; only the six parameters below change.
- **Search space** (independent draws; trial `i ≥ 1` uses `numpy.random.default_rng(20261002 + i)`, so the
  points are a function of the index alone and a restart resamples the same ones):

  | parameter | distribution | current |
  |---|---|---|
  | `num_leaves` | log-uniform integer [31, 511] | 127 |
  | `min_data_in_leaf` | log-uniform integer [10, 500] | 50 |
  | `feature_fraction` | uniform [0.4, 1.0] | 0.8 |
  | `bagging_fraction` | uniform [0.5, 1.0] (`bagging_freq` stays 1) | 0.8 |
  | `lambda_l2` | log-uniform [0.01, 100] | 1.0 |
  | `learning_rate` | log-uniform [0.03, 0.12] | 0.05 |

  Trial 0 is the current parameters (no override) — the reference, and the cost measurement.
- **Budget rule (fixed now):** run trial 0 first and measure its wall time `t0` (s). **N = floor(43,200 / t0)**
  trials in total including trial 0, clipped to [12, 60] — i.e. what fits in ~12 h at the current recipe's cost.
  N is written into §1 once `t0` is known and is not changed afterwards. Safety stop: no new trial is started once
  the cumulative trial wall time (gate waits excluded) exceeds 16 h; the trials finished by then are the study.
- **Persistence/resume:** each finished trial is written atomically to `logs/retune/trials/trial_NNN.json` (and its
  October predictions to `logs/retune/preds/`) before the next starts; a restarted study skips finished indices.
- **Resources:** one trial at a time, each in its own subprocess. Before each trial the study polls every 60 s until
  `ps -eo args | grep -c '[a]dsb_learn_'` is 0 and the cgroup `total_rss` (own and parent, from `/proc/self/cgroup`)
  is < 2,500 MB. `total_rss` is sampled every 10 s during a trial; above 13,000 MB the trial is killed, recorded as
  `killed_memory`, and not retried.
- **Selection:** the best trial = lowest October matched RMSE among the finished trials. If that is trial 0, no
  candidate exists: the current parameters stay and nothing below is run.

### 0.4 Confirmation (the decision for the member)

`configs/main_flseq_osm_nolirf_l2_opt.yaml` = `main_flseq_osm_nolirf_l2.yaml` + `matched_params:` the best trial's
six values (nothing else). Run its CV on the four production folds (jan, jul, jul_fwd, forward; pooled OOF = jan,
jul, forward = 672,428 rows), `--skip-predict --fold-cache logs/retune/fold_cache`, gated as above, and

`python -m prc.compare main_flseq_osm_nolirf_l2-ef204b44ff <opt id> --work-dir data/work_flseq`.

**Rule (`prc.compare.verdict`): ADOPT only if the full CI (K=0) and the drop-200 CI both exclude zero on the
improving side; REJECT if both exclude zero on the worse side; anything else keeps the current parameters.**
Reported: the K = 0/20/200/2000 curve, month bootstrap, matched/unmatched lane, per airport. Expected and checked:
unmatched lane predictions byte-identical to `ef204b44ff` (model B untouched).

### 0.5 If the member is ADOPTED: rebuild M1 and M2, then the blend decision

- `configs/main_flseq_osm_nolirf_l2_pa_opt.yaml` (PA + the same `matched_params`) and
  `configs/main_flseq_osm_nolirf_plainy_l2_opt.yaml` (plain-y + the same `matched_params`): CV on the four folds
  (fold cache), then full fits (`--skip-cv`) of all three `_opt` recipes. CatBoost (M3) and the unmatched lane
  (`main_flseq` model B) are unchanged.
- **`hybrid_blend_l2_pa_opt`**, matched lane `((0.5·a2o + 0.5·pao) + p2o + c2) / 3`, unmatched lane `main_flseq`.
  Base **`hybrid_blend_l2_pa`** = `((0.5·a2 + 0.5·pa) + p2 + c2) / 3` rebuilt from the existing OOFs (must
  reproduce 269.21, `per-airport-member.md` §2). Hybrid OOFs are written to a scratch dir, and
  `python -m prc.compare hybrid_blend_l2_pa hybrid_blend_l2_pa_opt --work-dir data/work_flseq --exp-dir <scratch>`
  with the same rule decides. Report the K curve, month bootstrap, lane split, per month, per airport.
- **Only if that is also ADOPT:** write `outputs/hybrid_blend_l2_pa_opt.parquet` (matched rows
  `rint(((0.5·a2o + 0.5·pao) + p2o + c2) / 3)` from the full-fit files, unmatched rows from
  `outputs/main_flseq.parquet`) and validate: 344,841 rows, schema equal to the template, ids and order equal to
  `data/raw/submitting.parquet`, no nulls, no negatives, unmatched rows equal to `outputs/hybrid_blend_l2_pa.parquet`,
  `prc.validate_submission` OK. Otherwise no file.
- Nothing is uploaded or submitted, and no ADS-B residual is applied (v32 = `hybrid_blend_l2_pa` + the learned
  ADS-B residual; applying it to a new base is a separate step).

### 0.6 Runs and resources

All runs alone (one training at a time), gated as in §0.3, logged to `logs/retune/` (git-ignored) with exit codes
checked, peak cgroup `total_rss` sampled every 10 s, killed above 13,000 MB. Reference: pooled anchored+L2 CV
4,134 s / 11,231 MB; PA CV 3,636 s / 8,909 MB.

## 1. Tuning study

**Budget (§0.3 rule):** trial 0 (current params) took t0 = 1,340 s wall (1,264 s fitting; 1,571,364 training rows,
183,761 October matched rows; 1,072 rounds; peak total_rss 7,282 MB own / 7,496 MB parent).
**N = floor(43,200 / 1,340) = 32 trials** (0 … 31), fixed here before trial 1 started.

**Run:** `scripts/retune_lgb.py study --n 32`, 2026-10-02 05:47 → 18:16 UTC (trial 0 alone first, then 1–31). All 32
trials finished (`status ok`); none killed, none failed, no restart needed. Sum of trial wall times 44,344 s
(12.3 h); peak total_rss over all trials 7,607 MB (parent cgroup). 7 trials hit the 2,000-round cap (all with
lr ≤ 0.045 or few leaves). Gate waits: none after the first trial (no `adsb_learn_` process at any later start).

October matched RMSE, all 32 trials, best first (`logs/retune/summary.txt`; d = vs trial 0):

```
idx  rmse_m    d_vs_0  leaves  min_leaf  ff     bf     l2       lr      rounds  wall_s  peak_MB
  8  204.204   -3.059     357        11  0.418  0.646  8.928    0.03462    1200    1590  7470
 12  204.593   -2.670     278        65  0.470  0.935  2.421    0.05272    1002    1240  7515
 16  204.775   -2.488     360        45  0.420  0.867  27.3     0.08723     498     780  7521
 31  205.028   -2.235     273        40  0.588  0.958  0.02832  0.03807    1386    1910  7492
  9  205.602   -1.661     102        11  0.592  0.863  0.5843   0.03573    2000    2030  7300
 22  205.628   -1.635     260        68  0.602  0.841  3.839    0.03168    1428    2540  7497
 29  205.715   -1.548      84        93  0.476  0.844  34.78    0.07773    1653    1080  7227
  1  206.203   -1.060     181        86  0.599  0.823  86.48    0.05295    1082    1520  7467
  3  206.383   -0.880      80        48  0.512  0.724  3.249    0.08619     969     810  7261
 23  206.446   -0.816      49        17  0.588  0.835  0.02125  0.06409    2000    1590  7450
 19  206.622   -0.641      57        11  0.498  0.565  0.01616  0.04455    2000    1340  7361
 28  206.858   -0.404     114       110  0.518  0.737  82.71    0.04257    1780    1610  7476
 20  207.014   -0.249      46        10  0.499  0.630  0.03716  0.04401    2000    1340  7183
  0  207.263   +0.000     127        50  0.800  0.800  1        0.05      1072    1340  7496
  2  207.332   +0.069      36        49  0.774  0.967  0.02234  0.1121    1394    1170  7045
 17  207.887   +0.624      63       180  0.489  0.767  0.4114   0.1132    1121     880  7363
 14  208.299   +1.036      82        88  0.775  0.749  7.774    0.04283    1598    1750  7457
 10  208.374   +1.111     172        50  0.935  0.826  2.789    0.08558     402     750  7470
 18  208.383   +1.121     355        17  0.916  0.686  7.15     0.1039     203     530  7511
 13  208.599   +1.337      68       100  0.685  0.706  0.01043  0.08665     934     980  7050
 15  208.661   +1.399     315        73  0.974  0.885  1.546    0.0767     301     820  7235
 11  208.747   +1.484      85        25  0.809  0.555  12.27    0.05107    1224    1250  7450
  4  208.782   +1.519     301        35  0.997  0.612  1.563    0.04018     575    1220  7515
  7  208.837   +1.575     117        51  0.770  0.605  64.38    0.0527     973    1350  7349
 25  208.956   +1.693     493       431  0.420  0.722  0.3413   0.03652    1502    2670  7607
 30  209.628   +2.365     387       108  0.945  0.717  0.7835   0.05254     451    1000  7367
 24  209.629   +2.366     281       117  0.945  0.734  0.01201  0.06277     419     950  7380
  5  209.905   +2.642      54       452  0.702  0.955  0.2088   0.05438    2000    2100  7365
 21  210.924   +3.661      75       103  0.970  0.658  59.1     0.04313    1343    1510  7455
 26  211.181   +3.918      32        19  0.793  0.511  1.744    0.0327    2000    1330  7281
 27  211.582   +4.319     102       429  0.987  0.931  16.39    0.05951    1095    1430  7356
  6  214.581   +7.318      44       446  0.995  0.698  11.17    0.0349    2000    1930  7462
```

**Best: trial 8 — `num_leaves` 357, `min_data_in_leaf` 11, `feature_fraction` 0.4178, `bagging_fraction` 0.6457,
`lambda_l2` 8.928, `learning_rate` 0.03462** (1,200 rounds): 204.204 vs 207.263, **−3.06 s on the tuning fold**,
lower at all ten airports (−1.78 EDDF … −4.72 EGLL). This is a selection-biased number (best of 31 draws); §2 is
the decision. Reading of the table (not a test): the eight best trials all have `feature_fraction` ≤ 0.61, and every
trial with `feature_fraction` ≥ 0.9 is worse than trial 0; large `min_data_in_leaf` (≥ 400) is the worst region.

## 2. Confirmation (four folds)

`configs/main_flseq_osm_nolirf_l2_opt.yaml` = base + `matched_params` of trial 8 → exp id
**`main_flseq_osm_nolirf_l2_opt-7244b3004f`**. CV started 18:16:52 UTC (`logs/retune/run.sh opt_cv`).

`logs/retune/run.sh opt_cv --config configs/main_flseq_osm_nolirf_l2_opt.yaml --work-dir data/work_flseq
--skip-predict --fold-cache logs/retune/fold_cache`: exit 0, not killed, wall 6,992 s, peak total_rss 11,028 MB
(parent 11,052 MB). (A reported container restart at ~18:26 UTC did not stop this run: it was still alive in the
previous session's cgroup. A duplicate launched by mistake at 18:26:57 was killed after ~10 s, before it had loaded
features; the original run, its fold cache and its log are the record.)

Per fold, base `ef204b44ff` → opt `7244b3004f`, matched RMSE (rounds_matched):

| fold | base | opt |
|---|---|---|
| jan | 201.51 (1,269) | 197.64 (1,237) |
| jul | 241.45 (1,208) | 238.14 (1,415) |
| jul_fwd (shadowed) | 250.19 (903) | 246.09 (1,214) |
| forward | 196.73 (978) | 193.97 (1,512) |

`python -m prc.compare main_flseq_osm_nolirf_l2-ef204b44ff main_flseq_osm_nolirf_l2_opt-7244b3004f --work-dir
data/work_flseq` (`logs/retune/compare_member.txt`), 672,428 pooled OOF rows:

| K | A | B | diff [CI95] |
|---|---|---|---|
| 0 | 273.57 | 271.17 | **−2.41 [−2.85, −2.02]** |
| 20 | 229.10 | 226.23 | −2.88 [−3.14, −2.64] |
| 200 | 215.42 | 212.34 | **−3.08 [−3.35, −2.85]** |
| 2000 | 192.26 | 189.39 | −2.87 [−3.03, −2.73] |

Month bootstrap −2.41 [−2.63, −2.21]. Matched lane 211.3 → 208.1; unmatched 1,571.0 → 1,571.0 and all 8,378
pooled unmatched predictions exactly equal (model B untouched, as required). Matched RMSE lower at all ten airports
(EDDF −2.66, EDDM −3.46, EGLL −3.72, EHAM −2.44, LEBL −1.91, LEMD −2.07, LFPG −4.31, LIRF −3.57, LSZH −3.15,
LTFM −3.84).

**Verdict: ADOPT** (`VERDICT … ADOPT - adopt (-2.41 s, still -3.08 s at 200 rows dropped)`). The tuning-fold gain
(−3.06 s, selection-biased) transferred to the untouched comparison months at −3.2 s on the matched lane. §0.5 runs.

## 3. M1 / M2 rebuild and the blend decision

`configs/main_flseq_osm_nolirf_l2_pa_opt.yaml` (`main_flseq_osm_nolirf_l2_pa_opt-8d54effc06`) and
`configs/main_flseq_osm_nolirf_plainy_l2_opt.yaml` (`main_flseq_osm_nolirf_plainy_l2_opt-dd8b8e2c7c`): base recipe +
the same `matched_params`. Runs, in order, one at a time (`logs/retune/chain.sh`, each step gated by
`logs/retune/run2.sh`, which also sums total_rss over every session cgroup): PA CV, plain-y CV, then full fits of
`_opt`, `_pa_opt`, `_plainy_l2_opt`. Started 20:15:54 UTC.

Two container events: none touched a result. (1) ~18:26 UTC, see §2. (2) ~23:55 UTC a real restart killed the chain
during the plain-y CV's forward fold; jan/jul/jul_fwd were in the fold cache. `chain.sh` was relaunched at 23:56:02,
skipped the finished PA step, reloaded the three cached folds and fitted forward only (first attempt's log:
`logs/retune/plainy_opt_cv_attempt1.log`).

| step | exp id | exit | wall s | peak total_rss MB (own / all) |
|---|---|---|---:|---:|
| PA CV | `main_flseq_osm_nolirf_l2_pa_opt-8d54effc06` | 0 | 7,177 | 9,007 / 9,177 |
| plain-y CV (resumed; forward fold only) | `main_flseq_osm_nolirf_plainy_l2_opt-dd8b8e2c7c` | 0 | 1,302 | 10,781 / 10,947 |
| full fit, pooled `_opt` | | 0 | 1,633 | 9,579 / 9,746 |
| full fit, PA `_opt` | | 0 | 1,263 | 7,613 / 7,778 |
| full fit, plain-y `_opt` | | 0 | 1,332 | 9,695 / 9,861 |

Member CV, matched RMSE per fold, base → opt (overall pooled):

| member | jan | jul | jul_fwd | forward | pooled matched | overall |
|---|---|---|---|---|---|---|
| M1 pooled (a2) | 201.51 → 197.64 | 241.45 → 238.14 | 250.19 → 246.09 | 196.73 → 193.97 | 211.307 → 208.144 | 273.57 → 271.17 |
| M1 per-airport (pa) | 200.04 → 196.52 | 243.87 → 238.74 | 251.61 → 247.34 | 195.71 → 192.33 | 211.307 → 207.346 | 273.57 → 270.56 |
| M2 plain-y (p2) | 204.13 → 199.22 | 242.08 → 239.00 | 250.99 → 247.18 | 197.05 → 194.09 | 212.233 → 208.816 | 274.28 → 271.68 |

Model B is untouched in all three: CV unmatched-lane numbers equal to the base runs, and in each full-fit file the
5,290 unmatched rows equal the base file's exactly (matched rows move by RMS 39.67 / 46.73 / 43.97 s).

**Blend.** `scripts/blend_pa_opt.py oof --scratch logs/retune/scratch` reproduces the base `hybrid_blend_l2_pa`
at **269.21** (as `per-airport-member.md` §2); `hybrid_blend_l2_pa_opt` 268.32. `prc.compare hybrid_blend_l2_pa
hybrid_blend_l2_pa_opt --work-dir data/work_flseq --exp-dir logs/retune/scratch` (`logs/retune/compare_blend.txt`):

| K | A | B | diff [CI95] |
|---|---|---|---|
| 0 | 269.21 | 268.32 | **−0.89 [−1.10, −0.72]** |
| 20 | 224.65 | 223.59 | −1.06 [−1.22, −0.92] |
| 200 | 211.15 | 209.99 | **−1.17 [−1.31, −1.03]** |
| 2000 | 188.28 | 187.17 | −1.11 [−1.19, −1.03] |

Month bootstrap −0.89 [−0.99, −0.79]. **Verdict: ADOPT.** Lane split (`detail_blend.txt`): matched −1.17
[−1.32, −1.00], unmatched +0.00, all except LIRF −0.98 [−1.24, −0.77]. Per calendar month (all rows): Jan −1.01,
Jul −0.85, Nov −0.71, Dec −1.04, every CI below zero. Per airport (all rows): LIRF −0.57 [−1.08, −0.08], EGLL −1.50,
LFPG −0.95, LTFM −1.50, EHAM −1.06, EDDM −0.90, EDDF −0.50, LEMD −0.86, LEBL −0.67, LSZH −1.05; all ten below zero.
Reading: each retuned LGB member gains ~3.2–4.0 s on the matched lane alone, but the blend gains 1.17 s there: the
CatBoost third is unchanged, and the retuned members are partly the same errors the blend was already averaging away.

## 4. Output (ADOPT, so §0.5 applies)

`scripts/blend_pa_opt.py full` → **`outputs/hybrid_blend_l2_pa_opt.parquet`** (`logs/retune/full.txt`):
- matched rows `rint(((0.5·a2o + 0.5·pao) + p2o + c2) / 3)` from `outputs/main_flseq_osm_nolirf_l2_opt.parquet`,
  `outputs/main_flseq_osm_nolirf_l2_pa_opt.parquet`, `outputs/main_flseq_osm_nolirf_plainy_l2_opt.parquet`,
  `outputs/main_flseq_osm_nolirf_cb_l2.parquet`; unmatched rows from `outputs/main_flseq.parquet`.
- 344,841 rows (= template; 339,551 matched, 5,290 unmatched); schema `{MVT_ID_mvt: double, TAXITIME_SEC_mvt: int32}`
  equals the template; ids and order equal `data/raw/submitting.parquet`; 0 nulls, 0 negatives, min 0, max 113,641.
- Against `outputs/hybrid_blend_l2_pa.parquet`: differs on 324,976 rows, all matched (unmatched untouched), max |d|
  820 s, RMS 20.44 s.
- `prc.validate_submission` OK (only the usual > 10,800 s max warning); the three `_opt` member files also pass.

Nothing was uploaded or submitted, and no ADS-B residual was applied: v32 = `hybrid_blend_l2_pa` + the learned
ADS-B residual, so the like-for-like candidate is this file + the same residual, a separate step. Expect less on
the board than the OOF gain (the per-airport member's −0.77 s OOF came with a smaller board gain).

Code and tests: §0.2. Full suite 251 passed before any run (`logs/retune/pytest_full.txt`); `tests/test_matched_params.py`
(11 tests, incl. the `_opt` recipes vs their bases and the blend script's members/formula) passes (with test_per_airport_member.py: 19 passed, `logs/retune/pytest_final.txt`) with the three
`_opt` recipes in place. Logs (git-ignored): `logs/retune/`.
