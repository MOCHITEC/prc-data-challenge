# LIRF G-model blend into model B (gmodel_airports) — pre-registration (2026-10-05)

## Background
In `copy.md` item 2, the OpenAir idea, measured on our data, gave −138 s on the 1,475 LIRF unmatched rows
(12-month leave-month-out, CI excludes 0).
- λ = 0.5 was the best value whether it was chosen on Jan or on Jul.
- λ = 0.25 gave −92 s, also with a CI that excludes 0.

## Change
`gmodel_airports` / `gmodel_lambda` (train.py `_gmodel_blend`; default empty, existing ids unchanged).
- Ĝ = (MVT − SCHED) − y is fitted on the training side of the fold, on every labelled LIRF departure (matched rows
  included), with model B's columns.
- On LIRF unmatched target rows: pred = (1 − λ)·B + λ·((MVT − SCHED) − Ĝ).
- **λ is fixed at 0.25 before the run** (the conservative choice; λ = 1 lost 12 s on OpenAir's LB).

## Arms (all folds)
| arm | config |
|---|---|
| base | main_flseq_osm_nolirf_l2_opt (a2o; the same run serves the L6 and cmix pairs) |
| gm | main_flseq_osm_nolirf_l2_opt_gm |

## Verdict (fixed before the runs)
The change touches only LIRF unmatched rows (about 0.11% of the ranking set), so whole-frame CIs are uninformative.
- **ADOPT** only if all of the following hold:
  1. On LIRF unmatched rows, the pooled paired-bootstrap CI excludes 0 on the improving side.
  2. Both the jan and the jul fold improve on those rows.
  3. Matched rows are bit-identical between the arms.
- `prc.compare` is reported for completeness, and so is its "unmatched @ LIRF" line.
