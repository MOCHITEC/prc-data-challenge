# Neighbour-clock residual (2026-10-06): small, stable, parked for a batch

## Features
New matched-lane features, built only from fields visible in 2026. "Excess" means MVT − AOBT_3 minus the 2025
median per airport × runway × local hour.

- Excess means and counts in backward and forward windows around the take-off, also per runway.
- "Following" clocks around the flight's own AOBT_3.
- Gate retention: flights past EOBT_1 that have not yet pushed.
- Departures taxiing at AOBT_3.

## Model
- Small L2 LightGBM on e = y − p. LTFM and LIRF are excluded.
- Change = 0.5 × prediction, clipped to ±150 s.
- Gated off on cold or de-icing hours and when the neighbour excess is above 300 s. The gate was chosen after
  January 2025 failed.

## Validation
Leave one month out (SSE reduction / N):

| OOF | Gain | Months positive | Bootstrap 5th percentile |
|---|---|---|---|
| 7-month | +21.1 | 7/7 | +17.9 |
| 12-month | +25.3 | 12/12 | +23.2 |

Every airport is positive on both OOFs.

## 2026 effect
- The gate switches off most January rows at EDDF, LSZH, EDDM and EHAM, so the change mostly applies to July.
- Mean |delta| is 1.1 s, with a range of −13 to +28 s.
- Expected board effect: about 0.03–0.05 s.

The 2026 deltas are in `$SCRATCH/agents/nbr/rank_delta.parquet`. Script: `analysis/local_nbr_clock.py`.
