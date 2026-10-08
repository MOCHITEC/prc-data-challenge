# Coverage-matched training of the ADS-B residual (2026-10-08): rejected

## Coverage match between 2026 and 2025
- Nov–Dec 2025 is the closest analogue at most airports.
- EDDF 2026 is sparser, and EGLL July 2026 denser, than every 2025 month.
- Coverage-only domain classifier, 2026 vs 2025: AUC 0.845.

## Validation
Leave-one-month-out over the 12 months (48 fits, V2 params, 778 rounds). Gains vs production, in MSE (positive =
better):

| arm | all | 2026-like cells | importance-weighted | months positive |
|---|---|---|---|---|
| B1 month-similarity weights | −40.9 | −0.7 | −57.9 | 3/12 |
| B2 domain importance weights | −98.8 | −68.0 | −35.7 | 0/12 |
| C rate features | −10.7 | −33.5 | −23.8 | 4/12 |

None passes, and no delta was built. Reweighting loses more effective data than it gains in regime match. EDDF
and EGLL in 2026 are outside every 2025 regime, so reweighting cannot reach them.
Scripts: `analysis/local_covw_*.py`.
