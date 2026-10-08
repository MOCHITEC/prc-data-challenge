# Matched-blend re-weighting: honest leave-one-fold-out measurement (2026-10-08)

## Setup
- **Members:** a2o (plus the LIRF cmix delta), pao, p2o and m4.
- **c2 is not local.** Its id is main_flseq_osm_nolirf_cb_l2_cbopt-9d30cf6882.
- **Baseline:** production weights without c2, (0.5·a2o + 0.5·pao + p2o + m4)/3.
- **Rows:** 664,050 matched rows from Jan, Jul, Nov and Dec 2025.
- **Validation:** weights are fitted on 2 folds and tested on the 3rd. CIs come from a 1,230-block airport-day bootstrap.
- **Units:** "w" is whole-frame seconds, about 0.75 × the change in matched-lane RMSE.

## Results

| Option | Δ w | 95% CI | Months | Top-50 share |
|---|---|---|---|---|
| Global simplex, all rows | −0.31 | [−0.43, −0.20] | all 4 negative | 0.59 (fails) |
| Per-airport, all rows | −0.43 | [−0.59, −0.28] | all 4 negative | 0.60 (fails) |
| LGB stacker | −0.22 | [−0.38, −0.04] | Dec +0.07 | reject |
| **Global, non-LIRF rows only** | **−0.22** | **[−0.30, −0.14]** | all 4 negative | 0.475 |
| Per-airport, non-LIRF rows only | −0.25 | [−0.32, −0.17] | all 4 negative | 0.44 |
| Inner split only (a2o:pao = 0.1:0.9), non-LIRF | −0.12 | — | all folds negative | — |

- **Why the all-row fits fail:** 32–41 of their top-50 rows are LIRF schedule copies. The fit also double-counts the LIRF copy overlay.
- **Consistent signal:** outside LIRF, pao is under-weighted, and p2o and m4 are over-weighted.

## Candidate recipe (non-LIRF matched rows; LIRF keeps production weights)
- **Without c2:** 0.5·(0.2·a2o + 0.8·pao) + 0.25·p2o + 0.25·m4.
- **With c2:** 0.4·(0.2·a2o + 0.8·pao) + 0.2·(p2o + c2 + m4). Not measured here; it needs the c2 OOF.
- **Expected:** about −0.2 s. The ADS-B residual, which is trained on y − blend, may absorb part of it.
