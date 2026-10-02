# Hyperparameter retune of the pooled anchored LGB+L2 member (model A) — pre-registration (2026-10-02)

**Status: pre-registered, nothing run.** §0 is written and committed before any tuning trial, confirmation or
blend run. Results will be appended as §1+ without editing §0 (except this status line and the one measured
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
