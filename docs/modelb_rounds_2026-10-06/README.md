# Model B (unmatched lane) stopping rounds (2026-10-06): no robust gain

Since the LIRF schedule reparametrisation, model B early-stops at 35–56 rounds whenever the holdout is December
(jan, jul, aso, fmj folds and the full 2025 refit, which logs 36–45). With a June or October holdout it runs
114–161 rounds.

The real `predict_fold` path on 6 folds covers all 12 months. Production is reproduced exactly (jan 56 rounds,
RMSE 2867.8).

| variant | result |
|---|---|
| more rounds (≥ 400), robust stopping metrics (Huber, L1, clip 7200) | clearly worse |
| fixed 100 rounds | pooled ΔMSE on unmatched rows −30.7k [−58.8k, +1.8k]; 4/6 folds better |

The fixed-100 gain comes entirely from LIRF (+44k share) and LTFM (+12k). All other 8 airports get worse.

- **LIRF:** better in 3 folds, worse in 3. The top 10 rows hold 55 % of the gain.
- **LTFM:** the gain is almost all from the fmj fold (Feb 2025 snow), on rows with y > 3600.

This is the same heavy-tailed, few-row pattern that lost v41. Rejected.
Script: `analysis/local_modelb_rounds.py`.
