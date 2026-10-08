# ucmix + LIRF same-day context (ucday): pre-registration (2026-10-08, before the run)

## Why
- The error decomposition of 2026-10-07 found an unmatched-lane airport-day persistence ceiling of −1.35 s (other hours of the same day, label means). Almost all of it is at LIRF.
- Label means are not available on the target side. This arm tests whether label-free same-day context carries part of it into the ucmix classifier and regressor.
- The context columns are: the day's departure count, the unmatched count and share, the unmatched rows' median MVT − SCHED, and how many of them have MVT − SCHED in [2 h, 4 h).

## Change
- `ucmix_day_feats: true` on top of `main_flseq_osm_nolirf_l2_opt_ucmix`. The implementation is `train._attach_day_feats`.
- Training rows aggregate over the training frame; target rows aggregate over the full target frame.
- Only the inputs of the ucmix classifier and regressor change. Model B and every other row are untouched.

## Comparison
main_flseq_osm_nolirf_l2_opt_ucday vs main_flseq_osm_nolirf_l2_opt_ucmix-b581da8fb2 (all folds).

## Verdict: ADOPT only if all hold
1. Only LIRF unmatched rows change.
2. On LIRF unmatched rows, the airport-day bootstrap CI excludes 0 on the improving side.
3. jan, jul and forward each improve on LIRF unmatched rows.
4. `prc.compare` gives ADOPT on the whole frame.
