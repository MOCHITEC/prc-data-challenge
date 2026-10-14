# Is the base LightGBM ("opt" params) tuned out?  (worker: tune, 2026-10-08)

Status: jan/jul screen done for the arms below (tables regenerated 2026-10-13 from runs/results.jsonl, which now holds cat_tight's 3-seed replicates); Feb+Jun screen PREPARED, NOT RUN (see `## Feb+Jun screen` at the end).

## Setup found
- "opt" = `matched_params` of `configs/main_flseq_osm_nolirf_l2_opt.yaml` (model A / matched lane only):
  num_leaves 357, min_data_in_leaf 11, feature_fraction 0.4178, bagging_fraction 0.6457 (freq 1), lambda_l2 8.928,
  lr 0.03462; objective regression (L2), 2000-round cap, ES 100 on the inner holdout (latest training month),
  refit on all training rows with rounds scaled, 3 seeds averaged. Source: 32-trial random search on a single
  tune_oct fold (docs/competitor_research_2026-09-21/results/optuna-retune.md, scripts/retune_lgb.py).
  Search box: leaves [31,511], min_data [10,500], ff [0.4,1.0], bf [0.5,1.0], l2 [0.01,100], lr [0.03,0.12].
  **The winner sits at/near the box edge on ff (0.418 vs 0.4), min_data (11 vs 10) and lr (0.035 vs 0.03)**;
  objective, cat_* and lr<0.03 were never searched (objective/metric are refused by `matched_params`).
- Baseline CV record `main_flseq_osm_nolirf_l2_opt-7244b3004f` (pb-work/experiments/experiments.jsonl, two runs):
  jan matched 197.644 / 197.558 (rounds 1237 / 1761), jul matched 238.144 / 238.174 (rounds 1415 / 1440).
  -> run-to-run noise of the production protocol is ~0.1 s on a fold, ES round count is very unstable.

## Harness (nothing in src/ modified)
- `dump_folds.py`: production feature build (`retune_lgb.train_features`, same calls as `prc.pipeline.main`) +
  `predict_fold` with `prc.train._fit` monkey-patched to save model A's exact stacked design matrix
  (same `_matrix`, same inner-holdout mask, target g = y - proxy) for the jan and jul folds; model B skipped.
- `run_arms.py`: LightGBM 4.7.0 on those matrices; matched-lane RMSE after zero floor + LOBT bound (as `_assemble`).
  Mode `es` = the early-stopping fit only (train = 10 months, ES on the 11th, predict the fold month at best iter):
  1 fit/fold, used for screening. Mode `full` = exact production protocol (ES + scaled-round refit x 3 seeds).
- Matched-lane delta -> overall-fold delta: roughly x0.57 (jan) / x0.80 (jul), since model B rows are untouched.
- `watch.sh` hard-codes the memory cgroup id of the 2026-10-08 container; in later containers that path does not exist, so its
  kill never fires (peak logging still works). `watch_fj.sh` resolves the cgroup from /proc/<pid>/cgroup at runtime.

## Results (screening, mode es, single seed 42 unless arm says otherwise; d = arm - base, seconds)

| arm | jan RMSE_m | d jan | jul RMSE_m | d jul | mean d (matched) | ~d overall | rounds jan/jul | min/fold |
|---|---|---|---|---|---|---|---|---|
| base | 200.181 | +0.000 | 239.892 | +0.000 | +0.000 | +0.000 | 1543/1485 | 24/9 |
| seed43 | 199.985 | -0.195 | 239.903 | +0.010 | -0.092 | -0.052 | 1613/1680 | 10/9 |
| lr_half | 199.356 | -0.824 | 239.329 | -0.564 | -0.694 | -0.460 | 2890/3386 | 17/21 |
| ff_025 | 199.440 | -0.740 | 239.773 | -0.119 | -0.430 | -0.259 | 1960/1946 | 8/7 |
| leaves_714 | 199.879 | -0.302 | 240.192 | +0.300 | -0.001 | +0.034 | 904/1058 | 7/8 |
| mdl_40 | 200.691 | +0.510 | 240.414 | +0.521 | +0.516 | +0.354 | 1469/1468 | 8/8 |
| huber_300 | 201.135 | +0.955 | 246.736 | +6.843 | +3.899 | +3.009 | 1997/1992 | 11/10 |
| cat_loose | 201.495 | +1.314 | 243.295 | +3.403 | +2.359 | +1.736 | 1164/1272 | 6/6 |
| seed44 | 200.238 | +0.057 | 240.127 | +0.234 | +0.146 | +0.110 | 1881/1559 | 9/8 |
| lr_half_s43 | 199.570 | -0.611 | 238.836 | -1.056 | -0.834 | -0.597 | 2810/3025 | 16/17 |
| lr_half_s44 | 199.365 | -0.816 | 239.246 | -0.646 | -0.731 | -0.491 | 3066/2354 | 16/12 |
| ff_025_s43 | 199.597 | -0.584 | 239.228 | -0.664 | -0.624 | -0.432 | 1999/1931 | 7/7 |
| ff_025_s44 | 199.267 | -0.914 | 239.756 | -0.136 | -0.525 | -0.315 | 1974/1980 | 8/7 |
| cat_tight | 199.099 | -1.082 | 238.897 | -0.995 | -1.038 | -0.706 | 1996/1894 | 10/10 |
| mdl_4 | 200.469 | +0.288 | 240.469 | +0.576 | +0.432 | +0.313 | 1542/1531 | 9/9 |
| cat_tight_s43 | 199.412 | -0.769 | 238.628 | -1.264 | -1.017 | -0.725 | 1998/1962 | 11/10 |
| cat_tight_s44 | 199.269 | -0.912 | 239.150 | -0.742 | -0.827 | -0.557 | 1976/1924 | 11/11 |

3-seed average (ES model, seeds 42/43/44 averaged in g-space; closest to the production 3-seed protocol):

| arm | jan RMSE_m | d jan | jul RMSE_m | d jul | mean d (matched) | ~d overall |
|---|---|---|---|---|---|---|
| base | 198.979 | +0.000 | 238.641 | +0.000 | +0.000 | +0.000 |
| lr_half | 198.822 | -0.157 | 238.379 | -0.262 | -0.210 | -0.150 |
| ff_025 | 198.192 | -0.787 | 238.210 | -0.431 | -0.609 | -0.397 |
| cat_tight | 198.119 | -0.859 | 237.589 | -1.052 | -0.956 | -0.666 |

Noise reference: base seeds 42/43/44 (same params): jan 200.18/199.99/200.24, jul 239.89/239.90/240.13 -> single-fit seed sd ~0.13 s per fold (~0.1 s on the 2-fold mean). Averaging 3 seeds alone gives -1.2 s per fold, so single-seed gains from variance-reducing knobs (lower lr, lower ff) must be re-checked at 3 seeds (production averages 3 seeds).

## Interim reading (updated as arms land)
- Capacity is at its optimum: num_leaves 714 (null: -0.30/+0.30), min_data_in_leaf 4 (+0.29/+0.58) and 40 (+0.51/+0.52) both worse.
- huber (alpha 300) is clearly worse under RMSE (+1.0 jan, +6.8 jul); L2 stays.
- lr halved (rounds ~2x, cap 4000): -0.69 single-seed but only -0.21 matched (~-0.15 overall) at 3 seeds, for 2x train cost -> mostly variance reduction that 3-seed averaging already buys.
- feature_fraction 0.25 (below the old search box floor 0.4): -0.61 matched (~-0.40 overall) at 3 seeds, survives averaging; runs at the 2000-round cap.
- Categorical regularisation matters: cat_loose (cat_smooth 1, cat_l2 1, min_data_per_group 20) +1.3/+3.4; cat_tight (50/50/200) -1.08/-1.00 single-seed and -0.86 (jan) / -1.05 (jul) at 3 seeds (-0.96 matched, ~-0.67 overall): the largest 3-seed gain screened, ahead of ff_025 (-0.61 matched) and lr_half (-0.21). Per-seed cat_tight: jan 199.10/199.41/199.27, jul 238.90/238.63/239.15 (sd ~0.16/0.26), all three seeds below base on both folds.

## Feb+Jun screen (prepared 2026-10-13, NOT RUN - waits for the 12 GB production fit to finish)
Protocol: every lever must also pass a Feb+Jun check, pooled Feb+Jun delta vs base <= 0 (matched-lane RMSE, mode es,
seed 42). Folds `feb` = 202502 held out (142,622 labelled matched target rows), `jun` = 202506 held out (180,640);
train = the other 11 months, ES holdout = the latest training month 202512 (164,028 rows), exactly the jan/jul structure.
- `dump_folds_fj.py`: dump_folds.py + `extra_folds()` (feb/jun built like cv.month_block_folds' jan/jul; meta.json also
  records valid/train/ES months). Same config as the cached jan/jul: `main_flseq_osm_nolirf_l2_opt` (NOT the L6/cmix
  production config `a2o_l6_cmix_ff025`: no `ret_*` columns, ff 0.4178 base, no LIRF copy mixture).
- `queue_fj.sh [arms]`: preflight (MemAvailable >= 10 GB else exit 2) -> dump feb, jun (serial, one process each, ~8.2 GB
  RSS, ~1.5 min each with the feature cache) -> run_arms.py es feb,jun per arm (serial, `watch_fj.sh` watchdog, ~3.7 GB
  RSS, ~7-10 min per fold-fit) -> update_readme.sh -> `summarize_fj.py` -> `fj_screen.md`. Default arms: base cat_tight ff_025
  (6 ES fits, ~1 h; re-runnable, finished parts are skipped).
- Launch: `nohup $S/queue_fj.sh > $S/logs/queue_fj.log 2>&1 &` with S = this directory; result table in `fj_screen.md`.
