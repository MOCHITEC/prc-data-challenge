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

## Result (2026-10-05, `gm_verdict.txt`)
**REJECT.** main_flseq_osm_nolirf_l2_opt-7244b3004f → main_flseq_osm_nolirf_l2_opt_gm-67f49a5d62.
- Condition 3 met: matched rows are bit-identical.
- Condition 1 failed: on LIRF unmatched rows the pooled change is −10.3 s, CI [−57.8, +31.9] (includes 0).
- Condition 2 was not met cleanly:
  - jan improves by −115.9 s (n = 60);
  - jul is −1.0 s, essentially flat (n = 337);
  - forward is worse by +18.3 s (n = 140).
- Whole frame: −0.12 s.
- Reading: the stand-alone measurement (−138 s over 12 months, λ = 0.5) did not carry over to production model B.
  Production model B already applies the LIRF schedule reparameterisation, which captures most of what Ĝ adds.
  Not adopted; the code stays off.
