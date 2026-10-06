# Carrier × airport echo rate for model B (fe_carrier_echo) — pre-registration (2026-10-06)

## Background
`copy.md` item 3 (the tip shared on Discord).
- **12-month leave-month-out on a simplified model B:** −25.4 s on the unmatched lane, CI excludes 0.
- **Jan+Jul only:** −9.6 s, CI spans 0.
- **Month-to-month stability:** correlation 0.85; LTFM is the exception at 0.04.

## Change
`fe_carrier_echo` (train.py `carrier_echo_features`, default off, existing ids unchanged).
- Four columns: `car_echo_unm`, `car_n_unm`, `car_echo_all`, `car_n_all`.
- Smoothed rate: (k + 20·p_airport) / (n + 20) of |y − (MVT − SCHED)| ≤ 60 s, per (airport, carrier).
- Carrier = the ICAO prefix of the flight number, or its first two characters.
- Fitting:
  - target rows use the training side of the fold;
  - each training row uses the other training months only.
- Hidden from model A.

## Arms (all folds)
| arm | config |
|---|---|
| base | main_flseq_osm_nolirf_l2_opt-7244b3004f (already run) |
| car | main_flseq_osm_nolirf_l2_opt_car |

## Verdict (fixed before the run)
**ADOPT** only if all of the following hold:
1. Matched rows are bit-identical.
2. On the unmatched rows, the airport-day block-bootstrap CI excludes 0 on the improving side.
3. Both the jan and the jul fold improve on the unmatched rows.

Per-airport results and `prc.compare` are reported, but not used for the decision.

## Result (2026-10-06, `car_verdict.txt`)
**REJECT.** main_flseq_osm_nolirf_l2_opt-7244b3004f → main_flseq_osm_nolirf_l2_opt_car-574127d012.

| condition | result | met? |
|---|---|---|
| 1. matched rows bit-identical | identical | yes |
| 2. unmatched CI excludes 0 on the improving side | 1571.6 → 1565.3, −6.29 s [−20.80, +4.91] — the CI includes 0 | **no** |
| 3. jan and jul improve on unmatched rows | jan −1.55, jul −18.59 | yes |

- forward is worse: +4.31.
- Most of the change is LIRF (−34.2 s). EDDM (+7.1) and LEMD (+6.4) get worse.
- Whole frame: −0.45 s.
- **Reading:** this is the same pattern as in `copy.md` (12-month gain, Jan+Jul gain not significant). The airline-only
  rate does not pass. The finer hierarchy (`fe_stamp_hier`, queued) is the next test of this idea.
