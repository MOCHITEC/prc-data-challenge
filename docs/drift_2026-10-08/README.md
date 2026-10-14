# 2026 prediction-level drift vs 2025 (2026-10-08): no actionable fix

Question: does v47 predict a level for any airport-month of 2026 that 2025 never showed, and if so, is it a bug?
Measure: anchored residual res = y − (MVT − AOBT_3) = AOBT_3 − BLOCK, mean clipped at ±1200 s. 2025 uses labels,
2026 uses the v47 prediction.

## Outliers
- **Snow:** LTFM Jan (+117 vs Jan 2025), EDDM Jan, LSZH Jan. Spikes fall on frozen-precip METAR days (LTFM 12, 18, 19 Jan).
  This matches `docs/weather_2026-10-06` (snow uplift is already in the base). Not a bug.
- **LFPG Jul 2026:** pred res is −64. Every 2025 month at LFPG was +5 to +60, and every day of Jul 2026 is −41 to −89.
  - All base members (LGB opt/pa/plainy, L6, MLP) and all past submissions agree, so it comes from the features,
    not from the blend or an overlay.
  - **Cause, a change in how the source data is recorded:**
    - The share of rows with EOBT_1 later than LOBT is 0.06 in every 2025 month and 0.34 in Jul 2026.
    - The share with EOBT_1 = LOBT drops from 0.72–0.77 to 0.53.
    - Median AOBT_3 − EOBT_1 is 120 s, against 240–600 s in 2025.
  - **Milder versions elsewhere:** EGLL Jul 2026 (EOBT_1 = SCHED 0.40 vs 0.47–0.57), and LTFM Jul 2026
    (LOBT = SCHED 0.79 vs 0.54–0.65).
  - **Why the model's response is still defensible:** in 2025 the label residual follows AOBT_3 − EOBT_1
    (coefficient about 0.2), not LOBT and not SCHED, at every airport. This holds also on rows where EOBT_1 > LOBT
    (LFPG coefficient 0.217 vs 0.022).
  - **ADS-B check:** AOBT_3 relative to first movement (≥ 3 kt) is unchanged in Jul 2026 (−617 vs −590 s). The 2026
    residual by AOBT − EOBT_1 bucket matches 2025. So the level drop is a change in the input distribution that the
    2025 relation explains.
  - **Risk:** if the new EOBT_1 > LOBT rows are a new kind of row (unknowable without labels), LFPG Jul is
    under-predicted by about 100 s on about 21k rows, worth about 1 s. A level shift is a ±1 s gamble that only a
    leaderboard probe could settle.
- **LFPG Jul 2025 seq_arr_* oddity:** the 2025 summer months are the anomaly (`pm10` 0.7–1.35 against about 0.1
  otherwise). 2026 looks like normal months. Not a consistency bug.

## Verdict
No validated fix. Scripts: `analysis/drift_2026-10-08/*.py`.

## Same-day worker results (summarised)
- **Residual-segment search** on the 5-fold 2025 OOF (2.09M rows):
  - No correction replicates at ≥ 0.2 s.
  - Airport-level bias is ±10–30 s and changes with season and training window. Carrying it forward costs up to 0.5 s.
  - LIRF unmatched ×1.03 gives +0.5 s, but 10 rows carry 120 % of the gain. Rejected.
- **Blend weight / stacking** on the Jan/Jul OOF (anchored, plain, CatBoost), cross-season:
  - Best global or hour-bucket weights: −0.1 s, with a CI that crosses 0.
  - Per-airport weights, per-airport intercepts and a LightGBM stacker are worse by +0.2 to +0.5 s.
