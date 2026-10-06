# Hyperparameter retune of the anchored CatBoost+L2 member M3 (model A) — pre-registration (2026-10-05)

**Status (2026-10-06): member ADOPT (−1.86 s [−2.36, −1.46], K=200 −2.18 [−2.65, −1.76]); blend `hybrid_blend_l2_pa_opt_mlp` → `_cbopt` ADOPT (−0.18 s [−0.28, −0.09], K=200 −0.21 [−0.34, −0.11]); R6 secondary −0.07 [−0.14, −0.01], K=200 −0.09 [−0.16, −0.01]. Full fit running.** §0 is written and committed before any tuning trial,
confirmation or blend run. Results will be appended as §1+ without editing §0, except this status line and the one
measured number §0.3 asks to be filled in (the trial budget N, fixed by a rule stated here before it is measured).

## 0. Pre-registration

Full test suite before any run: 269 passed (262 before this change, plus the 7 new tests in
`tests/test_cb_matched_params.py`; `logs/cbretune/pytest_full.txt`).

### 0.1 Why, and the prior

The board best is v35 = 245.935: base `outputs/hybrid_blend_l2_pa_opt_mlp.parquet` (`mlp-member.md`) + the learned
ADS-B residual (`adsb-learned-residual.md` §12) + the positional ADS-B overlay on 430 unmatched rows
(`posjoin-on-mlp.md`). In that base, the matched lane is the equal-weight mean of four members:
- M1 = 0.5 × pooled anchored LGB+L2 (retuned, `main_flseq_osm_nolirf_l2_opt-7244b3004f`) + 0.5 × per-airport anchored
  LGB+L2 (retuned, `main_flseq_osm_nolirf_l2_pa_opt-8d54effc06`);
- M2 = plain-y LGB+L2 (retuned, `main_flseq_osm_nolirf_plainy_l2_opt-dd8b8e2c7c`);
- M3 = anchored CatBoost+L2 (`main_flseq_osm_nolirf_cb_l2-4aa341b206`);
- M4 = sklearn MLP (`main_flseq_osm_nolirf_l2_mlp-2b9a3b9348`).
The unmatched lane is `main_flseq-8615e6a194`'s model B.

M3's settings were never searched: lr 0.2, depth 6, `max_ctr_complexity` 1, up to 5,000 rounds, 1 seed
(`catboost-blend.md`, `blend-l2.md`), with CatBoost's library defaults for everything else (resolved and logged:
`l2_leaf_reg` 3, `border_count` 254, MVS bootstrap with `subsample` 0.8, `one_hot_max_size` 2; `random_strength` 1).
Its early-stopping best counts were 3,136–3,940 of 5,000, so it is near its round cap. The LGB retune
(`optuna-retune.md`) gained −2.41 s on its member, but only −0.89 s on the blend, where the other members already
average away part of the error. M3 is 212.40 matched alone (M1 205.94) and is the member with 1/4 weight that is
the furthest behind M1 among the trees.

**Prior:** the member gains 0–2 s on the matched lane; the blend gains 0–0.4 s. P(member ADOPT) ≈ 0.5,
P(blend ADOPT | member ADOPT) ≈ 0.5. A retuned CatBoost could also become more correlated with the LightGBM
members, which would shrink its blend value.

### 0.2 What is tuned, and the code

1. **Only M3** `configs/main_flseq_osm_nolirf_cb_l2.yaml` (`main_flseq_osm_nolirf_cb_l2-4aa341b206`), and only its
   **model A** (matched lane). M1, M2, M4 and the unmatched lane are not touched.
2. New **`TrainConfig.cb_matched_params`** (dict, default `{}`, in `_HASH_OPTIONAL`, so out of the hash when empty):
   CatBoost settings merged over the resolved `cb_params` for model A only (`TrainConfig.for_matched()`); model B
   always uses `cb_params`. It is `matched_params`' CatBoost twin, needed for the same reason: under CatBoost
   `cb_params` also configures model B, so the unmatched predictions of the member would move and the member
   comparison would not isolate model A. Refused unless `model: catboost`, and for the keys in `CB_OWNED_KEYS`
   (iterations, seeds, stopping, `has_time`, `cat_features`) and `CB_MATCHED_PARAMS_REFUSED` (`loss_function`,
   `eval_metric`, `thread_count`, `verbose`, `allow_writing_files`, `task_type`). When set, `predict_fold` records it
   in `info["cb_matched_params"]`; when empty it makes exactly the calls and records it made before.
3. **`tests/test_cb_matched_params.py`** (7 tests):
   - all 129 recipes in `configs/` at f830720 keep their experiment id and the resolved model A / model B
     parameters byte for byte (any new recipe must be `*_cbopt.yaml`);
   - `predict_fold`'s predictions and `info` hash exactly as under the f830720 code on a fixed synthetic fold, for
     four arms: LightGBM default, LightGBM `matched_params`, CatBoost default, CatBoost `cb_params` (digests computed
     by running f830720 before the field existed, and identical after it was added);
   - refusals and the hash;
   - with an override, model B's predictions and record are byte-identical, model A moves, and model A equals the
     fit with the same settings in `cb_params`;
   - any `*_cbopt.yaml` recipe equals its base plus `cb_matched_params` only, with keys from the search space;
   - the retune script's base config, tuning fold, search space, trial 0 = current settings, determinism of the
     sampler, and that every sampled point is accepted by `TrainConfig`;
   - the blend script's member ids, files and formulas.
   The recipe-list guards in `tests/test_matched_params.py`, `tests/test_per_airport_member.py` and
   `tests/test_mlp_member.py` are widened to ignore `*_cbopt.yaml`, which this file checks.
4. **`scripts/retune_cb.py`** (the search, §0.3) and **`scripts/blend_cbopt.py`** (§0.5). A dry run of
   `blend_cbopt.py oof` with M3 standing in for M3' reproduces the base `hybrid_blend_l2_pa_opt_mlp` at **267.57**
   (`mlp-member.md` §2), and `build_full` with the stand-in reproduces `outputs/hybrid_blend_l2_pa_opt_mlp.parquet`
   exactly (0 rows differ).

### 0.3 Tuning split, objective, search, budget (the `optuna-retune.md` design, reused)

**Optuna is not installed; nothing is installed. The search is a plain random search (`scripts/retune_cb.py`).**

- **Tuning split `tune_oct`: train 2025-01 … 2025-09, validate 2025-10** (the same `TUNE_FOLD` object as
  `scripts/retune_lgb.py`). M3's comparison folds validate on 2025-01 (`jan`), 2025-07 (`jul`) and 2025-11+12
  (`forward`); October is validated by none of them. Early stopping uses the recipe's inner holdout (the latest
  training month, 2025-09), as always.
- **Objective:** RMSE of the member's model-A prediction (proxy + g, zero floor, LOBT bound, exactly as
  `predict_fold`) on October's matched rows (`mvt_minus_aobt3` non-null). Model B is not fitted in a trial. The
  recipe's own settings otherwise: 1 seed, 5,000-round cap, early stopping 100, the ES-then-refit procedure of
  `_fit_catboost`.
- **Search space** (independent draws in this key order, one uniform each; trial `i ≥ 1` uses
  `numpy.random.default_rng(20261005 + i)`, so the points are a function of the index alone and a restart resamples
  the same ones):

  | parameter | distribution | current |
  |---|---|---|
  | `depth` | uniform integer {4, …, 10} | 6 |
  | `learning_rate` | log-uniform [0.08, 0.4] | 0.2 |
  | `l2_leaf_reg` | log-uniform [0.3, 30] | 3 |
  | `random_strength` | log-uniform [0.1, 10] | 1 |
  | `subsample` (MVS bootstrap, which is the resolved default; `bagging_temperature` applies only to the Bayesian bootstrap) | uniform [0.5, 1.0] | 0.8 |
  | `border_count` | log-uniform integer [32, 254] | 254 |
  | `one_hot_max_size` | choice {2, 4, 16} | 2 |
  | `max_ctr_complexity` | choice {1, 2} | 1 |

  Our code passes any non-owned key through `cb_params`, so `one_hot_max_size` and `max_ctr_complexity` are
  supported and are searched. `border_count` stays ≤ 254 (one byte per quantised value, as now).
  Trial 0 is the current settings (no override) — the reference, and the cost measurement.
- **Budget rule (fixed now):** run trial 0 first and measure its wall time `t0` (s).
  **N = floor(43,200 / t0)** trials in total including trial 0 (no clipping). N is written into §1 once `t0` is
  known and is not changed afterwards. Safety stop (as in `optuna-retune.md`): no new trial is started once the
  cumulative trial wall time exceeds 16 h; the trials finished by then are the study.
- **Persistence/resume:** each finished trial is written atomically to `logs/cbretune/trials/trial_NNN.json` (and its
  October predictions to `logs/cbretune/preds/`) before the next starts; a restarted study skips finished indices and
  reruns an interrupted one from scratch (bit-identical: fixed seed and thread count).
- **Resources:** one trial at a time, each in its own subprocess. Before each trial the study waits (60 s polls)
  until no `adsb_learn_` or `prc.pipeline` process runs and the cgroup `total_rss` (own and parent, from
  `/proc/self/cgroup`) is < 2,500 MB. `total_rss` is sampled every 10 s; above 13,000 MB the trial is killed,
  recorded as `killed_memory`, and not retried.
- **Selection:** the best trial = lowest October matched RMSE among the finished trials. If that is trial 0, no
  candidate exists: the current settings stay and nothing below is run.

### 0.4 Confirmation (the decision for the member)

`configs/main_flseq_osm_nolirf_cb_l2_cbopt.yaml` = `main_flseq_osm_nolirf_cb_l2.yaml` + `cb_matched_params:` the
best trial's eight values (nothing else). Run its CV on the recipe's folds, `--skip-predict --fold-cache
logs/cbretune/fold_cache`, gated as above, and

`python -m prc.compare main_flseq_osm_nolirf_cb_l2-4aa341b206 <cbopt id> --work-dir data/work_flseq`.

**Folds.** M3's recipe has `folds: [jan, jul, forward]` and has never run `jul_fwd` (`catboost-blend.md`). Those three
are every fold of the pooled OOF (672,428 rows); the fourth production fold, `jul_fwd`, is shadowed by `jul` in the
pooled OOF of every member and enters no comparison. The confirmation therefore runs the recipe's three folds, which
are all the rows `prc.compare` scores, and keeps the `folds` key equal to the base (so the recipe is base +
`cb_matched_params` only).

**Rule (`prc.compare.verdict`): ADOPT only if the full CI (K=0) and the drop-200 CI both exclude zero on the
improving side; REJECT if both exclude zero on the worse side; anything else keeps the current settings.**
Reported: the K = 0/20/200/2000 curve, month bootstrap, matched/unmatched lane, per airport, per fold. Expected and
checked: unmatched-lane predictions byte-identical to `4aa341b206` (model B untouched).

### 0.5 If the member is ADOPTED: the blend decision

- **`hybrid_blend_l2_pa_opt_mlp_cbopt`**: matched lane `((0.5·a2o + 0.5·pao) + p2o + c2o + m4) / 4`, unmatched lane
  `main_flseq`. Base **`hybrid_blend_l2_pa_opt_mlp`** = `((0.5·a2o + 0.5·pao) + p2o + c2 + m4) / 4` rebuilt from the
  existing OOFs (must reproduce 267.57). `scripts/blend_cbopt.py oof --scratch logs/cbretune/scratch --c2o <id>`, then
  `python -m prc.compare hybrid_blend_l2_pa_opt_mlp hybrid_blend_l2_pa_opt_mlp_cbopt --work-dir data/work_flseq
  --exp-dir logs/cbretune/scratch` with the same rule decides. Reported: K curve, month bootstrap, and
  `blend_cbopt.py detail` (lane split, per month, per airport); member error correlations (report only).
- **Only if that is also ADOPT:** full fit `prc.pipeline --config configs/main_flseq_osm_nolirf_cb_l2_cbopt.yaml
  --work-dir data/work_flseq --skip-cv` → `outputs/main_flseq_osm_nolirf_cb_l2_cbopt.parquet`; then
  `scripts/blend_cbopt.py full` → **`outputs/hybrid_blend_l2_pa_opt_mlp_cbopt.parquet`** (matched rows
  `rint(((0.5·a2o + 0.5·pao) + p2o + c2o + m4) / 4)` from the full-fit files, unmatched rows from
  `outputs/main_flseq.parquet`), and validate: 344,841 rows, schema equal to the template, ids and order equal to
  `data/raw/submitting.parquet`, no nulls, no negatives, unmatched rows equal to
  `outputs/hybrid_blend_l2_pa_opt_mlp.parquet`, `prc.validate_submission` OK. Otherwise no full fit and no file.
- Nothing is uploaded or submitted, and no ADS-B residual or overlay is applied (v35 = the base + residual +
  overlay; applying them to a new base is a separate step).

### 0.6 Runs and resources

All runs alone (one training at a time), gated as in §0.3, logged to `logs/cbretune/` (git-ignored) with exit codes
checked, peak cgroup `total_rss` sampled every 10 s (own and all sessions), killed above 13,000 MB (memory limit
13,681 MB). Reference: M3's CV + full fit 16,638 s (CV 10,819 s), peak 11,977 MB (`blend-l2.md`).

## 1. Tuning study

**Budget (§0.3 rule):** trial 0 (current settings) took t0 = **3,230 s** wall (3,202 s fitting; 1,571,364 training
rows, 183,761 October matched rows; ES best 3,993, refit 4,521 rounds; October matched RMSE 206.673; peak
total_rss 7,912 MB own / 8,132 MB parent). **N = floor(43,200 / 3,230.3) = 13 trials** (0 … 12), fixed here before
trial 1 started.

**Run:** `scripts/retune_cb.py study --n 13`, 2026-10-05 05:50 (trial 0 alone) / 06:49 → 23:11 UTC. Container restarts
at ~07:25 and ~08:00 UTC killed trial 1 twice (no record either time; study relaunched at 07:28:03 and 08:02:50, trial 1
rerun from scratch with the same pre-drawn settings, as §0.3 says). A later restart (~20:50) did not stop the study.
**The pre-registered 16 h safety stop ended the study after 10 trials (0 … 9):** cumulative trial wall 57,735 s
exceeded 16 h before trial 10 (`stop: cumulative trial wall 57735 s exceeds 16.0 h before trial 10`). Trials 10–12
were not run. All 10 finished `ok`; none killed. Peak total_rss over all trials 11,208 MB (trial 6, parent cgroup).
Trials cost 1,570–13,241 s (trial 0: 3,230 s); low learning rates and depth ≥ 8 with `max_ctr_complexity` 2 are the
slow ones; four trials reached the 5,000-round cap at refit.

October matched RMSE, all 10 trials, best first (`logs/cbretune/summary.txt`; d = vs trial 0):

```
idx  rmse_m    d_vs_0  depth  lr      l2      rs      subs   border  ohe  ctr  es_best rounds  wall_s  peak_MB
  7  204.424   -2.249      8  0.1554  18.24   2.015   0.806     153    4    2     4573    5000   13241  9655
  0  206.673   +0.000      6  0.2     3       1       0.800     254    2    1     3993    4521    3230  8132
  4  207.962   +1.289      7  0.1558  5.15    0.2621  0.976      74   16    1     3883    4397    4210  8938
  9  208.970   +2.297      9  0.1723  0.5117  5.498   0.798      74    4    2     1391    1575    4510  9180
  1  209.058   +2.385      9  0.1344  29.22   0.2796  0.774      50    4    1     4435    5000    5670  8351
  8  209.782   +3.109      6  0.1168  6.936   0.1656  0.886      71    2    1     4994    5000    4030  8124
  5  210.274   +3.601     10  0.08406  0.3438  0.3825  0.869      40    4    1     3525    3991    9061  8105
  6  210.782   +4.109      6  0.08303  1.313   2.829   0.721      42   16    2     4995    5000    7501  11208
  2  211.190   +4.517      5  0.3838  3.042   6.575   0.665      78    2    2     2791    3160    4710  9399
  3  217.161  +10.487      4  0.3815  28.83   0.6212  0.881      40    4    1     2100    2378    1570  8094
```

**Best: trial 7 — `depth` 8, `learning_rate` 0.1554, `l2_leaf_reg` 18.24, `random_strength` 2.015, `subsample`
0.8057, `border_count` 153, `one_hot_max_size` 4, `max_ctr_complexity` 2** (ES best 4,573, refit capped at 5,000):
204.424 vs 206.673, **−2.25 s on the tuning fold** (selection-biased, best of 9 draws). Per airport vs trial 0: LIRF
−9.75, LTFM −2.97, LSZH −2.92, LEMD −2.36, EDDF −2.08, EHAM −1.20, LEBL −0.74, LFPG −0.55, EDDM +0.46, EGLL +2.10.
Only one of nine random draws beat the current settings. Trial 7 cost 4.1× trial 0, so §2's CV is expected to take
roughly 4× M3's 10,819 s.

## 2. Confirmation

`configs/main_flseq_osm_nolirf_cb_l2_cbopt.yaml` = base + `cb_matched_params` of trial 7 → exp id
**`main_flseq_osm_nolirf_cb_l2_cbopt-9d30cf6882`** (folds jan, jul, forward, as §0.4).

**Attempt 1 (2026-10-05 23:14:06 UTC): killed by the memory guard after 111 s.**
`logs/cbretune/run.sh cbopt_cv --config configs/main_flseq_osm_nolirf_cb_l2_cbopt.yaml --work-dir data/work_flseq
--skip-predict --fold-cache logs/cbretune/fold_cache`: rc 137, total_rss 13,460 MB own / 13,693 MB all sessions at
23:15:57 (limit 13,681 MB; guard 13,000 MB). Trace (own cgroup, MB): 8,149 (11 s) → 10,098 → 12,134 (41 s) →
12,3xx–12,7xx → 13,460 (111 s), in the first fold, before any fold was cached. For comparison, the base M3 CV
(`blend-l2.md`, same recipe without `cb_matched_params`) reached 11,559 MB at 30 s and peaked at 11,977 MB; in the
tuning study, trial 7 peaked 1.5 GB above trial 0 (9,655 vs 8,132 MB). So the trial-7 settings
(`max_ctr_complexity` 2, depth 8) need about 1.5 GB more than the CV process's 12 GB, which does not fit.
Per the run rules, nothing was retried; no confirmation verdict exists. §0.4/§0.5 are on hold.

**Attempt 2 (coordinator-approved retry, memory only; no model parameter, fold or rule changed).** What was used:
1. **Each fold in its own process, through the fold cache** — `scripts/cv_per_fold.py run`. A fold process builds the
   labelled feature frame with the pipeline's own calls (`retune_lgb.train_features`), slices the fold exactly as
   `run_cv` does, **drops the full 2.08M-row frame** (`del` + `gc.collect()`), and fits through `prc.train._cached_fold`
   under the same exp id, so the fold lands in `logs/cbretune/fold_cache/<exp_id>/<fold>.*` as `run_cv` writes it.
   Then the unchanged `prc.pipeline --skip-predict --fold-cache logs/cbretune/fold_cache` loads all three folds and
   writes the OOF and the record (its documented resume path).
2. **Raw frames / unused columns:** the pipeline already frees the raw frames and drops hidden candidate columns
   (`drop_hidden_candidates`); the per-fold process does the same and in addition frees the full feature frame. No
   other column is dropped.
3. **Polars threads: not changed.**
4. **CatBoost `thread_count`: not changed (stays 4).** Checked: on a 60k-row toy problem with depth 8,
   `max_ctr_complexity` 2 and two categoricals, `thread_count` 4, 2 and 1 gave bit-identical predictions. But
   `thread_count` is part of the hashed resolved `cb_params` (it would change the exp id) and per-thread buffers are
   not where the memory is, so it was left alone.
5. **Snapshots** (restart resume, not memory): new env flag `PRC_CB_SNAPSHOT_DIR` (default unset → the fitter's
   calls are unchanged). Set, every CatBoost fit gets `save_snapshot`, its own `snapshot_file`
   (`<dir>/<digest>.cbsnap`, digest of params, stage, round cap, seed, stopping rule, label, holdout mask and the
   training block), `snapshot_interval` 300 s, and `allow_writing_files: True` with a `train_dir` beside it. The driver
   sets it per fold (`fold_cache/snapshots/<fold>`) and deletes a fold's snapshots once its cache entry exists.

Checks before launch:
- `tests/test_cb_matched_params.py` (+3 tests): snapshots on vs off give identical predictions and `info` on a
  synthetic fold with the trial-7-like settings (`max_ctr_complexity` 2), and a second run reading the finished
  snapshots is identical; a fit killed mid-training (subprocess, snapshot interval 1 s) and rerun equals the
  unsnapshotted reference exactly; on a four-fold synthetic frame, the per-fold path (slice + `_cached_fold` per fold,
  then `run_cv` reading that cache) writes fold parquet/JSON, OOF and `overall` identical to a plain `run_cv`.
- Real data: the per-fold feature frame equals the pipeline's (`main` with `run_cv` intercepted): 2,085,047 × 155,
  row-hash digest `490c347f3ca74061` both (`logs/cbretune/feat_eq_*.txt`).
- Full suite: 272 passed (`logs/cbretune/pytest_full2.txt`). The unchanged-path digests (`PRED_BEFORE`, all 129
  recipe ids) still hold.
- Gate: `/proc/self/cgroup` → `process_api/01a10ddc-…/claude-code-bash`. total_rss right before: 244 MB in this
  session's cgroup (the `claude` CLI ≈ 350 MB RSS, `environment-manager` ≈ 44 MB, shell), 0 MB in the previous
  session's cgroup (`01a10b15-…`, only 2.0 GB page cache, no processes). So attempt 1's 13,693 MB "all sessions" was
  the pipeline's 13,460 MB plus ~230 MB of these tools.
Rule unchanged: if a fold goes over 13,000 MB again, M3 tuning is closed as "not runnable within memory".

Attempt 2 launched 2026-10-05 23:29:08 UTC (`scripts/cv_per_fold.py run`, `logs/cbretune/per_fold.log`). Note: CatBoost
resolves a relative `snapshot_file` under `train_dir`, so the snapshots sit at
`fold_cache/snapshots/<fold>/<digest>_info/logs/cbretune/fold_cache/snapshots/<fold>/<digest>.cbsnap`; the path is
deterministic, so a restarted fold finds them (≈ 73 MB each).

| fold | start (UTC) | wall s | peak total_rss MB (own / all sessions) | ES best → refit rounds (model A) | matched RMSE: base M3 → cbopt |
|---|---|---:|---:|---|---|
| jan | 10-05 23:29:08 | 8,141 | 12,145 / 12,401 | 2,361 → 2,583 | 202.50 → 201.07 |
| jul | 10-06 01:44:49 | 8,751 | 11,769 / 12,044 | 2,621 → 2,872 | 244.16 → 240.45 |
| forward | 10-06 04:10:41 | 10,552 | 11,064 / 11,330 | (see record) | 196.73 → 194.69 |

All three folds exited 0, none killed, no restart during attempt 2; peak 12,145 MB own / 12,401 MB all sessions (jan).
CV wall 27,444 s (23:29:08 → 07:06:33). The record and OOF were then written by the unchanged pipeline reading the
cache (`run.sh cbopt_cv_assemble … --skip-predict --fold-cache logs/cbretune/fold_cache`: exit 0, 20 s, 5,978 MB;
all three folds "loaded from" the cache). CV `main_flseq_osm_nolirf_cb_l2_cbopt-9d30cf6882`: full 272.53 (base
274.39); jan 338.32, jul 301.68, forward 213.81.

`python -m prc.compare main_flseq_osm_nolirf_cb_l2-4aa341b206 main_flseq_osm_nolirf_cb_l2_cbopt-9d30cf6882 --work-dir
data/work_flseq` (`logs/cbretune/compare_member.txt`), 672,428 pooled OOF rows:

| K | A | B | diff [CI95] |
|---|---|---|---|
| 0 | 274.39 | 272.53 | **−1.86 [−2.36, −1.46]** |
| 20 | 233.52 | 231.33 | −2.18 [−2.67, −1.70] |
| 200 | 218.52 | 216.34 | **−2.18 [−2.65, −1.76]** |
| 2000 | 194.07 | 192.07 | −2.00 [−2.25, −1.76] |

Month bootstrap −1.86 [−2.68, −1.16]. Matched lane 212.40 → 209.96; unmatched 1,570.7 → 1,570.7, and all 8,378 pooled
unmatched predictions are exactly equal (model B untouched, as required). Matched RMSE per airport (cbopt − base):
LIRF −9.82, LFPG −3.22, LSZH −2.06, EDDF −1.82, LEMD −1.80, EHAM −1.68, EDDM −1.10, LEBL −0.63, LTFM −0.62,
EGLL −0.34 — lower at all ten. **Verdict: ADOPT** (`ADOPT - adopt (-1.86 s, still -2.18 s at 200 rows dropped)`).

## 3. Blend decision (§0.5)

`scripts/blend_cbopt.py oof --scratch logs/cbretune/scratch --c2o main_flseq_osm_nolirf_cb_l2_cbopt-9d30cf6882`
(`logs/cbretune/oof.txt`) reproduces the base `hybrid_blend_l2_pa_opt_mlp` at **267.57**; candidate 267.39. Member
matched RMSE: M1 205.94, M2 208.82, M3 212.40 → M3' 209.96, M4 213.27. Matched-lane error correlations of M3' with
M1 / M2 / M4: 0.9513 / 0.9426 / 0.9146 (M3: 0.9414 / 0.9355 / 0.9054) — M3' is better alone but more correlated.

`python -m prc.compare hybrid_blend_l2_pa_opt_mlp hybrid_blend_l2_pa_opt_mlp_cbopt --work-dir data/work_flseq
--exp-dir logs/cbretune/scratch` (`logs/cbretune/compare_blend.txt`):

| K | A | B | diff [CI95] |
|---|---|---|---|
| 0 | 267.57 | 267.39 | **−0.18 [−0.28, −0.09]** |
| 20 | 222.70 | 222.48 | −0.22 [−0.32, −0.11] |
| 200 | 209.11 | 208.90 | **−0.21 [−0.34, −0.11]** |
| 2000 | 186.49 | 186.25 | −0.24 [−0.31, −0.17] |

Month bootstrap −0.18 [−0.33, −0.06]. **Verdict: ADOPT** (`ADOPT - adopt (-0.18 s, still -0.21 s at 200 rows dropped)`).
Detail (`logs/cbretune/detail.txt`): matched −0.24 [−0.36, −0.13]; unmatched +0.00; all except LIRF −0.08 [−0.17,
+0.01]. Months (all rows): 2025-01 −0.00 [−0.13, +0.12], 2025-07 −0.36 [−0.58, −0.14], 2025-11 −0.19 [−0.39, +0.02],
2025-12 −0.16 [−0.34, −0.01]. Airports: LIRF −0.83 [−1.24, −0.42], LFPG −0.25, LEMD −0.21, EDDF −0.19, EHAM −0.10,
LSZH −0.06, LEBL −0.02, EGLL +0.05, LTFM +0.10, EDDM +0.12; only LIRF, LFPG and LEMD have CIs below zero. Most of the
gain is at LIRF.

**Secondary, report only (requested by the coordinator after §0): R6 with M3' substituted.** `blend_cbopt.py r6`
rebuilds R6 (`rejected-rereview.md` §3) from the OOFs exactly (max |diff| vs `logs/rereview/oof_rr_R6_pa5_lofo.parquet`
0.0; 267.25) and refits the LOFO sum-to-one weights the same way with M3' in place of M3 (`logs/cbretune/r6.txt`):
- weights (a2o, pao, p2o, M3', m4), held-out forward 0.118 / 0.327 / 0.138 / 0.181 / 0.236; jan 0.048 / 0.339 /
  0.174 / 0.216 / 0.224; jul 0.001 / 0.411 / 0.196 / 0.190 / 0.202; all folds 0.0523 / 0.3578 / 0.1721 / 0.1973 /
  0.2205 (M3's all-fold weight in R6 was 0.1470).
- `prc.compare rr_R6_pa5_lofo rr_R6_pa5_lofo_cbopt` (`logs/cbretune/compare_r6.txt`): K=0 **−0.07 [−0.14, −0.01]**,
  K=20 −0.09 [−0.16, −0.02], K=200 **−0.09 [−0.16, −0.01]**, K=2000 −0.11 [−0.16, −0.06]; month bootstrap −0.07
  [−0.16, +0.00]. The rule's two CIs exclude zero (verdict() prints ADOPT); this is a secondary and is not the decision.

The primary blend is ADOPT, so §0.5's full fit and output follow.
