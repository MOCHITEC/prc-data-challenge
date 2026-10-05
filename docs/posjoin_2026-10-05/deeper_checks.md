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

## Follow-up: the learned residual on positional-join rows (candidate, not submitted)

v34 used raw ADS-B only on rows with a pushback and no long stop. The remaining positional-join rows can get the
learned residual itself: `local_posjoin_learnfeat.py` builds its features, and `local_posjoin_learned_eval.py`
checks them against the 2025 labels. None of these rows was ever in the residual's training set.

| 2025 rows | n | RMSE base → learned | gain (MSE) |
|---|---|---|---|
| unmatched, non-overlay, non-LIRF | 724 | 569 → 501 | +151.5 |
| matched, non-LIRF | 100 | 416 → 386 | +6.9 |
| LIRF | 19 | 6,720 → 7,061 | −259.8 (excluded) |

Long-stop rows stay on the v34 distance rule. On 2025's long stops (mostly tows) the learned residual is better:
RMSE 369, against 539 for the base and 787 for the rule. But the 2026 snow-day stops are de-icing, and the board
gain of v34 (2,090 MSE) exceeded the estimate, which says the rule worked there.

The candidate is v34 plus the residual on 890 rows (`local_posjoin_learned_apply.py`). Its squared change is
307 MSE, and the expected gain is about 150–250 MSE. It is held back so it can be combined with finding 2.

## v35 candidate (held, not submitted): v34 plus three overlays

| step | script | rows | 2025 check | expected board gain |
|---|---|---|---|---|
| learned residual on positional-join rows | `local_posjoin_learned_apply.py` | 890 | +151.5 MSE (unmatched) / +6.9 (matched) | ≈ 150–250 |
| LIRF schedule-copy blend, `p + 0.4·q·(MVT−SCHED − p)` | `local_lirf_copy_overlay.py` | 26,277 | +390 MSE with A fixed in advance (Jan +15, Jul +375); +407 with p = base, i.e. without ADS-B (the 2026-July case) | ≈ 300–400 |
| LOBT identity clip ±3,606 s | `local_lobt_clip.py` | 17 | identity holds on all 2.06 M matched 2025 rows | ≈ 1–6 |

For the copy blend, the classifier is refit on all 12 months for 2026 (rounds = the check's best iteration × 1.1).
On 2026 LIRF its mean q is 0.167, against a 2025 copy rate of 0.178.

Compared with v34, the candidate changes 27,179 rows, and its squared change is 524 MSE.

## Data checks, round 3 (2026-10-05)

| # | question | result |
|---|---|---|
| 10 | Rows where p (b + learned) and ADS-B disagree by more than 10 min, 2025 (171 rows) | The model is right and ADS-B is wrong: RMSE 723 for p against 7,269 for ADS-B. Below 2 min (41k rows) both are about 90 s. The learned residual already weighs ADS-B well, so there is nothing left on push rows. |
| 11 | Cross-month group bias on the 4-month OOF (b + c_B), matched rows with y < 90 min | Leave-one-month-out shrunk corrections (k = 200): stand +27.5 MSE; operator −29, runway −94, aircraft type −44, ADES −52, local hour −82. Not transferable. |
| 12 | July 2026 runway mix against July 2025 | LFPG 27R falls from 16 % to 0.1 % (closed). LFPG median MVT−AOBT_3 rises by 60–120 s per runway. EGLL 09R +27 points (wind). The anchored model absorbs this through MVT−AOBT_3; the unmatched LFPG July exposure is ≈ 7 MSE. |
| 13 | Time-stamp resolution | 5 s at EDDM, EGLL, LFPG, LIRF and LTFM, and for BLOCK at LEBL and LEMD (P(sec = 0) = 1/12); 1 s elsewhere. No lever. |
| 14 | Learned-residual settings on the 4-month data (`local_adsb_tune_months.py`), paired against the current settings | V1 (127 leaves, min 200): +279 MSE [+156, +421]. **V2 (255 leaves, min 300, ff 0.7): +376 [+193, +582]**, with Nov+Dec +302 [+209, +395] and Jan+Jul +447 [+99, +842]. V3 (l2 = 50): +39 [−6, +88]. V2 is carried into the next candidate. |
