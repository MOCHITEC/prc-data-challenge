# Further checks on v34 (2026-10-05): no further large lever found

All checks are label-free on 2026 or validated against 2025 labels. Nothing was submitted.

| # | question | result | value |
|---|---|---|---|
| 1 | Other airport-days like EHAM's snow days? | Flagged against each airport's 2025 range: LSZH 01-10 (median MVT−AOBT_3 1,447 s vs a 2025 p99 of 716), LFPG 01-05..07 (1,617–1,621 vs 1,143), EDDF 01-29, EGLL 01-14. **July has none.** | — |
| 2 | Is v34 off on those days? | Matched rows with an ADS-B push: rms(v34 − ADS-B) is 159 s on flagged days and 234 s on normal days, so they are fine. Unmatched rows with no ADS-B: the same-30-min matched median is worse than the model in 2025 (853 vs 745 s on extreme slots). The best 2025 blend (w = 0.25) would be worth ≈ 20 MSE in 2026. | none |
| 3 | Unmatched rows joined by flight number (fn) | The learned residual is already at ADS-B accuracy in 2025. The large 2026 gap is 5 LIRF schedule-fill rows, correctly left alone. | none |
| 4 | Did NM AOBT_3 vs the ADS-B push shift between 2025 and 2026? | Shifts are at most ±55 s (LSZH +32..+48, EDDM +55 / −40). Even fully uncorrected that is ≈ 100 MSE. | small |
| 5 | v34 bias estimated from ADS-B push + the 2025 offset | The method reproduces 2025's true bias within 5–20 s. In 2026 every airport-month is within ±35 s (LFPG Jan: n = 10). The stand-exit-only variant fails its 2025 check, so it is not used. | none |
| 6 | Arrival mirror (2026 arrivals carry labels) | Fill rates and taxi-in are stable year on year (LIRF Jul fill 8.9 → 10.6 %; EHAM Jul taxi-in −77 s). No convention change. | — |
| 7 | Learned residual on LIRF unmatched rows with MVT−SCHED > 4 h (81 % schedule fills) | Hurts in 2025 (−330 MSE, 27 rows), but the sign flips at 3 h, so it is threshold-sensitive. In 2026, 21 rows carry a correction (median −862 s). Zeroing it is worth ≈ 30 MSE at best. Noted for the extreme-row work. | ≈ 0.06 s |
| 8 | Airport-hour error sharing: do ADS-B rows' residuals transfer to non-ADS-B rows? | corr 0.11 in a 1 h window, 0.02 per day. Best shrink ≈ 60 MSE in 2025. | ≈ 0.1 s |
| 9 | Arrival-side information (stand arrival taxi, etc.) | Already in the feature set (`ta_taxi`, `arr_taxi_in_mean_*`, `occ_*`). | — |
