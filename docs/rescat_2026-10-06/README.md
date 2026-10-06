# ADS-B residual + apron category (2026-10-06): adopted for the batch

Requested by peaceful-babbage, who cited another team: airline prefix in the correction model, −9.25 s on their
board. The arms, rule and vocabulary were pre-registered (`prereg.md`). The CV uses the months7 folds and the
day-block bootstrap. Gains are MSE over all labelled rows.

| arm vs production V2 (A0) | all | without LIRF | without LIRF+LFPG | Jan / Jul / Nov–Dec | top-50 share |
|---|---|---|---|---|---|
| A1 + airline | +689 | +27 [−15, +70] | — | — | 0.38 |
| A2 + airline + apron | +1007 | +308 | +290 | — | 0.26 |
| **A3 + apron only** | **+482 [391, 572]** | **+386 [347, 426]** | **+326 [297, 356]** | +651 / +728 / +401 | 0.12 |

- **A3:** every airport is positive and every per-airport CI excludes 0.
- **Airline:** a LIRF-only effect. On top of apron it hurts where production swaps the residual (A2 − A3 without
  LIRF+LFPG: −36).
- **Drift:** 0.04 % of 2026 rows have an unseen apron and 1.8 % a rare one. Adversarial AUC for apron is 0.54.

## 2026 delta
The delta is matched to production: at every airport except LFPG, 12:v2 → 12:v2+apron (12 months, 778 rounds);
at LFPG, 7:v2 → 7:v2+apron. It applies to ground-fix rows only, with LIRF and LTFM excluded.

- 230 454 rows change; Σ Δ² / N = 652; mean +1.1 s; p99 |Δ| 126 s.
- File: `$SCRATCH/agents/rescat/rank_delta_12m.parquet`.
- Expected +300–400 MSE (0.6–0.8 s).
- Caveat: apron was CV-tested on top of the 7-month model only.

Script: `analysis/local_adsb_rescat.py`.
