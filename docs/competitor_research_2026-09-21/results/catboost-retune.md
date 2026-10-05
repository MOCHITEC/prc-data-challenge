# Hyperparameter retune of the anchored CatBoost+L2 member M3 (model A) — pre-registration (2026-10-05)

**Status (2026-10-05): study done (10 trials, 16 h safety stop); best trial 7 −2.25 s on October. Confirmation CV killed by the 13,000 MB memory guard (13,460 MB at 111 s); on hold, no verdict, no blend, no output.** §0 is written and committed before any tuning trial,
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
