# Hierarchical schedule-stamp rates for model B (fe_stamp_hier) — pre-registration (2026-10-06)

## Background
- **Source:** versatile-violin (github.com/arnavhm/prc_26). Its v44 "stamp rates" use 7 columns over airline → destination → flight number, shrunk and never using their own month.
- **Their evidence:**
  - month-fold screen −2.38 s, 10/10 months;
  - held-out −4.6 s;
  - LB −3.91, confounded with another change.
- `fe_carrier_echo` (pending) is the airline-only level.

## Change
`fe_stamp_hier` (train.py `stamp_hier_features`; default off, existing ids unchanged).
- **Echo flag:** |y − (MVT − SCHED)| ≤ 60 s.
- **Columns:**
  - `stamp_car`: airport × carrier, shrunk toward the airport (k = 20);
  - `stamp_car_ades`: airport × carrier × destination, shrunk toward `stamp_car`;
  - `stamp_flt`: × flight number, shrunk toward `stamp_car_ades`;
  - `stamp_flt_n`: the flight-number count.
- **Fitting:** target rows use all training-side rows; each training row uses the training rows of the other months.
- **Lane:** model B only. Model A never sees the columns.

## Arm (all folds)
- base: main_flseq_osm_nolirf_l2_opt-7244b3004f.
- stamp: main_flseq_osm_nolirf_l2_opt_stamp.

## Verdict (fixed before the run)
Same as `carrier_plan.md`. ADOPT only if all three hold:
1. Matched rows are bit-identical.
2. On unmatched rows, the airport-day block-bootstrap CI excludes 0 on the improving side.
3. Both jan and jul improve on unmatched rows.

Report the 2025-vs-2026 adversarial AUC of `stamp_flt` (flight-number churn risk).
