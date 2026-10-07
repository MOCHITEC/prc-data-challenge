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

## Result (2026-10-07, `verdict.txt`)
**REJECT** under the pre-registered rule.
main_flseq_osm_nolirf_l2_opt-7244b3004f → main_flseq_osm_nolirf_l2_opt_stamp-84be677e9d.

| condition | result | met? |
|---|---|---|
| 1. matched rows bit-identical | identical | yes |
| 2. unmatched rows, CI excludes 0 on the improving side | 1571.6 → 1556.6, −14.93 s [−40.56, +4.39] — the CI includes 0 | **no** |
| 3. jan and jul both improve on unmatched rows | jan −22.00, jul −30.93 | yes |

- **forward gets worse:** +14.19.
- **Per airport:** LIRF −90.8 dominates. EDDF +4.5, EDDM +3.6 and LEMD +3.7 get worse; the other airports are within ±10.
- **Whole frame:** −1.07 s (270.12 vs 271.20). This is attractive, but it rests on LIRF unmatched tail rows (537 rows, RMSE in the thousands) and fails the CI.
- **Reading:** the hierarchy is far stronger than the airline-only rate (−6.3), and the signal sits at LIRF.
  - A LIRF-only version would be a **new** hypothesis chosen after seeing these numbers. It would need its own pre-registration, and its result would carry the selection caveat. It is not adopted here.
