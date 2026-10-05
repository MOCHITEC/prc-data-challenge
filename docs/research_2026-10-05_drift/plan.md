# 2025/2026 drift screen and month-scoped frame columns — pre-registration (2026-10-05)

## Measured before this plan [S]
`auc.py`: single-column adversarial AUC (LightGBM, 2-fold), Jan+Jul 2025 vs Jan+Jul 2026, over every numeric frame
column plus the lever columns L1/L2/L3/L5 (`auc.csv`).
- **L5 is not drifting** (tx_* 0.52–0.55). The idea that in-frame aggregates fingerprint the year does not explain
  the L5 miss (CV −0.32, LB +0.12).
- The top drifters are frame-scoped shares/percentiles: `fl_nm_rate_other` 0.747, `stand_rwy_share` 0.686,
  `seq_hw_prev_pairq` 0.674. Training is a 12-month frame and ranking a 2-month one, so the same statistic has a
  different resolution.
- `month_scope.py`: counted within the calendar month, `fl_nm_rate_other` drops to 0.521 and `stand_rwy_share` to
  0.620. The rest of the latter is the 2026 runway mix, which is real.

## Change
`fe_month_scope` (prc.scope, default off, existing ids unchanged): both columns are recomputed within the
calendar month in both frames. `seq_hw_prev_pairq` is left as it is for now (its percentile lives inside
`_runway_sequence`).

## Arms and verdict
- `ms_scope` = `ns_base` (main_flseq_osm_nolirf + folds [forward]) + `fe_month_scope: true`. Compared against
  `ns_base-2c65506a2d`.
- The forward CV cannot see the frame-scope mismatch, because train and validation are both 2025. So the rule is
  "do no harm":
  - adopt the month scope if `prc.compare ns_base ms_scope` has a K=200 CI upper end ≤ +0.10 s;
  - and the AUC of both columns stays as measured above.
- Its LB effect cannot be measured offline; that is stated, not hidden.
