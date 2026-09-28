# Hypothesis generator: data / labels / target / leakage / shift (subagent output, verbatim, 2026-09-16)

All numbers measured in the agent's session on data/raw/*.parquet, data/work/feat_*.parquet, data/ref/metar_hourly.parquet, experiments/oof_main-48c19bc830.parquet (latest main OOF, 672,428 rows, RMSE 334.58) and outputs/main.parquet. Scripts: docs/investigation_2026-09-16/scripts/q1.py … q5.py (1–3 min each, POLARS_MAX_THREADS=2, no training). "supposed" = nothing measured.

Notation: ms = MVT_TIME − SCHED_TIME (s); fill = |BLOCK_TIME − SCHED_TIME| ≤ 10 s; proxy = MVT_TIME − AOBT_3; unmatched = FLIGHT_ID_mvt null.

| id | hypothesis | training run? | status |
|---|---|---|---|
| lirf-sched-filled-block | At LIRF, when no real off-block exists BLOCK_TIME is filled with SCHED_TIME (±10 s), so the label becomes ms; the tail is this artefact, not taxiing. | no | mechanism verified; LB effect supposed |
| lirf-fill-mixture-prediction | For LIRF unmatched rows with ms ≥ 2 h the label is a Bernoulli mixture (p≈0.63 of y≈ms, else normal taxi); most OOF error there comes from firing on the non-filled rows; p partly separable by callsign prefix, delay magnitude, hour. | no | facts verified; separability partly verified |
| block-day-rollover | The 16 labels > 20 h are BLOCK_TIME with the date off by one day (label = true taxi + 86,400); 10 such rows are 34.7 % of OOF SSE. | no | verified in training; ranking count supposed |
| lirf-matched-fill | LIRF matched rows have filled BLOCK 17.8 %; when AOBT_3 is far from SCHED the residual y − proxy = AOBT_3 − SCHED with sd 1,138 s, making LIRF the worst matched airport (OOF RMSE 347, 7.9 % of SSE). | no | verified |
| eham-jan2026-disruption | EHAM Jan 2026 ranking contains a multi-day disruption (Jan 2–7, 9, 29) with 68–123 unmatched departures/day (normal 5–15), proxy sd up to 1,963; training has almost no analogue. | no | shift verified; effect supposed |
| winter-2026-deicing-shift | Jan 2026 has 2–4× more de-icing-gate hours than Jan 2025 at EHAM/LSZH/EDDF/EDDM; LTFM 9.5 % vs 0 %. | no | shift verified; effect supposed |
| unmatched-composition-shift | Ranking unmatched rows are a different population (1.53 % vs 1.08 %, EHAM/LSZH/KLM heavy, 70 % vs 82 % "flights seen among matched"). | no | shift verified; effect supposed |
| sched-anchored-delay-shift | 2026 delay levels differ by month from 2025 (Jan 2026 ms +300–420 s at EDDF/EDDM/LTFM, Jul 2026 −140–240 s), shifting every SCHED/EOBT-anchored feature. | no | shift verified; effect supposed |
| matched-proxy-outliers | 1,808 matched rows have |BLOCK − AOBT_3| > 1 h (LTFM: stale early AOBT_3; EGLL/LFPG: BLOCK wrong); 22 ranking matched rows predicted ≥ 2 h. | no | verified (2.5 % OOF SSE) |
| return-to-stand-twins | Departures recorded twice with same (ADEP, FLIGHT, SCHED), take-off within 2 min; matched twin carries earlier pushback's BLOCK (median y 3,170–3,834); 84 ranking rows in such groups. | no | verified; small |
| tail-count-per-set | CV/LB divergence dominated by how many LIRF coin-flip rows each evaluation set contains (52–337 per training month; ranking Jan 107 / Jul 276, 181 with ms ≥ 2 h). | no | counts verified; attribution supposed |
| checked-not-a-cause | Label identity, negative labels, weather coverage/parity, null parity, ID overlap, timezone types, routing of 46 AOBT-null matched rows — clean. | no | verified |

## Per-item detail

### 1. lirf-sched-filled-block (q2b C1c/C1d, q4 I/I2/I3)
- Label bucket composition (2,085,047 training departures): 2–4 h: 462 rows, 78 % unmatched, 71 % fill; 4–20 h: 106 rows, 98 % unmatched, 94 % fill; > 20 h: 16 rows, 12.5 % fill, 75 % BLOCK > 1 h before SCHED.
- Of 481 unmatched rows with y ≥ 2 h, 82.5 % have |y − ms| ≤ 60 s. Of 103 matched rows with y ≥ 2 h, 34 %.
- Fill rate LIRF unmatched 48.4 % (filled rows: y median 7,562, 55 % ≥ 2 h); LIRF matched 17.8 %; all other airports 0–5.7 %.
- In the 2–20 h ms band: LIRF unmatched 622 rows, fill 64 %, y≈ms 64 %, y median 8,489; every other airport × group has fill 0.00–0.01.
- Identity TAXITIME = MVT − BLOCK holds on all rows.
- Ranking: LIRF unmatched ms ≥ 2 h = 181 rows; outputs/main.parquet predicts median 7,041 for them (86 predicted ≥ 2 h).

### 2. lirf-fill-mixture-prediction (q5 N, O)
- OOF: LIRF unmatched ms ≥ 2 h = 239 rows = 34.8 % of total OOF SSE. Filled subset 129 rows: 5.4 % of SSE (pred median 8,639 vs y 10,560). Non-filled 110 rows: 29.3 % of SSE (pred median 6,091 vs y median 1,112, RMSE 14,172). Non-LIRF unmatched ms ≥ 2 h: 1,837 rows, 6.4 %.
- Separability on 627 LIRF unmatched ms ≥ 2 h training rows (fill 63 %): by ms hours 2 h 60 %, 3 h 54 %, 4 h 73 %, 5 h 87 %, 6 h 89 %, 7–13 h 100 % (n=34); by take-off hour UTC 0–6 h 82–100 %, 10–13 h 43–53 %, 15 h 31 %; by FLIGHT prefix ISR/AEZ/TWB/LAV/MSC/ETH 100 %, NOS 89 %, WZZ 69 %, WMT 60 %, RYR 48 %, EJU 27 %, ITY 21 %, UAL 10 %; FLIGHT_mvt not seen among matched training flights → 92 % (n=89) vs 58 % (n=527). AIRCRAFT_TYPE_mvt null on all 627; FLIGHT_RULE all "I"; runway 25 in 554/627.

### 3. block-day-rollover (q2 C2/C2b, q5 N)
- 16 rows y > 20 h; y − 86,400 = 602…1,992 s for 13 of 16 (e.g. AMX071: SCHED 06-24 21:15, BLOCK 06-24 15:29:54, MVT 06-25 16:03, y 88,392). 10 such rows in OOF = 34.7 % of OOF SSE (RMSE within group 51,121, pred median 37,131). Training ms bands: 20–28 h unmatched 23 rows (17.4 % y ≥ 2 h), > 28 h unmatched 3; ranking has 6 + 1 unmatched and 36 + 6 matched rows in those bands. Training: 15 rows over 12 months.

### 4. lirf-matched-fill (q2b C1d/C4b, q4 J, q5 N/N2)
- LIRF matched fill 17.8 % (by month 14.5–20.6 %); |AOBT_3 − SCHED| ≤ 600 s: fill 26.6 %, residual sd 253; > 600 s: fill 8.4 % (≈6,460 rows), residual sd 1,138, y sd 1,203. Fill by operator (hashed, n>2,000): 54 %, 25 %, 22 %, 19 %, 14 %, 11 %, ≤5.6 % rest; by segment Cargo 47 %, Non-Scheduled 45 %, Lowcost 22 %, Mainline 14 %. Matched |BLOCK − AOBT_3| > 1 h at LIRF: 457, y median 5,274, proxy median 1,025, 80 % filled. OOF LIRF matched RMSE 347 (next LFPG 246, EHAM 162), 7.9 % SSE; filled ∧ far subset (1,856 OOF rows) 3.0 % SSE, RMSE 1,106.

### 5. eham-jan2026-disruption (q1 A, q2b C7, q4 K, q5 P)
- EHAM unmatched Jan 2026 = 805 vs Jan 2025 = 293 (Jul 401 vs 515). Per day Jan 2026: d2 94, d3 108, d4 123, d5 84, d6 68, d7 18, d9 49, d29 69, else 1–19; Jan 2025 max 37. Proxy sd Jan 2026 d3 1,502, d4 1,963, d5 1,848, d6 1,644, d7 1,440 vs Jan 2025 220–600. AOBT_3 minute-rounding EHAM 0.676 (Jan 2026) vs 0.749 (Jan 2025) — only airport not ~100 % rounded. METAR de-icing-gate hours EHAM 12.8 % vs 5.1 %. Training EHAM unmatched on Jan 5 & 18 2025 (n=72): median 1,006, p90 3,580 vs other Jan days (n=221) 516 / 1,233. outputs/main.parquet EHAM Jan unmatched median 1,524, p90 3,152.

### 6. winter-2026-deicing-shift (q4 K)
- De-icing-gate hour share Jan 2026 / Jan 2025: EHAM 0.128/0.051, LSZH 0.180/0.059, EDDM 0.112/0.069, LTFM 0.095/0.000, EDDF 0.083/0.023, LFPG 0.015/0.031, EGLL 0.001/0.011. Low-visibility lower in 2026 at EHAM (0.031/0.109), LFPG (0.027/0.073). Weather cache complete; null parity ≤ 0.02 pp.

### 7. unmatched-composition-shift (q1 A/B, q3 G/G2)
- Unmatched rate 1.53 % (5,283; 2,449 Jan / 2,834 Jul) vs 1.08 % (22,424; handoff's 22,470/5,290 count proxy-null, adding 46/7 matched AOBT_3-null rows). By airport ranking/training: EHAM 3.16/1.65, LSZH 2.85/1.70, LEBL 1.47/1.03, LIRF 1.42/0.93, EDDF 1.16/0.85, LEMD 0.64/0.44. Training rate by month 0.62–2.06 % (peak Jul 2025). Unmatched airport share: EHAM 22.8 vs 18.2, LSZH 12.5 vs 10.2, LFPG 13.9 vs 16.8, LTFM 13.6 vs 16.2. FLIGHT prefix KL 7.4 vs 3.0, TH 10.8 vs 14.5, HB 2.7 vs 4.5. FLIGHT_RULE V 7.9 vs 13.1. Seen among matched flights 69.7 vs 81.9 %. Null pattern identical. Hour distribution near-identical. Training label by seen: median 962 / gt2h 2.08 % vs 846 / 2.27 %.

### 8. sched-anchored-delay-shift (q2b C5/C6, q3 F2)
- Matched ms median Jan 2026 vs Jan 2025: EDDF 1,545/1,183, EDDM 1,324/1,025, LTFM 1,615/1,194, LSZH 1,509/1,232, EHAM 1,451/1,371; Jul 2026 vs 2025: EDDF 1,456/1,693, EDDM 1,255/1,374, LIRF 1,864/2,038, LTFM 1,262/1,436, EGLL 1,678/1,918. aobt3_minus_sched median Jan 420 vs 240, Jul 481 vs 720. aobt3_minus_eobt1 LTFM Jan 0 vs −180, Jul −60 vs +120. Proxy medians stable (≤50 s); proxy sd higher Jan 2026 at EHAM (740 vs 476), EDDF, EDDM, LSZH. Congestion counts, headways, taxi-in means: no shift.

### 9. matched-proxy-outliers (q2b C4b, q5 N/Q)
- |BLOCK − AOBT_3| > 1 h on 1,808 matched rows; > 30 min 10,554; > 10 min 6.8 %. LTFM 657 rows: proxy median 4,692, y median 850. EGLL 296 / LFPG 186 / EDDM 45: y median 5,035 / 3,328 / 3,722, proxy 600 / 658 / 119. Matched |proxy − y| > 1 h: 689 OOF rows, 2.5 % SSE. Ranking matched predicted ≥ 2 h: EGLL 7, LIRF 7, EHAM 6, LFPG 2 (22). Ranking matched AOBT_3 null: 7.

### 10. return-to-stand-twins (q2b D/D2, q4 L)
- 223 pairs in training (450 rows), 129 mixed; take-offs within 120 s in 68–89 %; matched twin y median 3,170–3,834 vs unmatched 966. Ranking 84 rows in such groups (LFPG 38, LTFM 28, EDDF 10, LSZH 8). No MVT_ID overlap ranking/training; 4 FLIGHT_ID overlaps. Upper bound LB effect ≈ 0.8 % of SSE.

### 11. tail-count-per-set (q4 J, q2b C7, q5 P)
- LIRF unmatched per training month 52–337 (Jul 2025 = 337); fill rate 32–76 %; y ≥ 2 h share 22–48 %. Ranking LIRF unmatched Jan 107, Jul 276; ms ≥ 2 h: 181 (median ms 10,135). OOF: LIRF unmatched 537 rows = 36.8 % SSE; LFPG unmatched 1,562 rows = 15.5 % (RMSE 2,731); other unmatched airports ≤ 1.5 % each. Submissions predict ≥ 2 h on 103–130 ranking rows (training frequency implies ≈97), ≥ 12 h on 3–19 (implies ≈6).

### 12. checked-not-a-cause
- TAXITIME == MVT − BLOCK within 1 s on 100 %; no null labels. Negative labels 369 rows (min −12 s; LSZH 298, LFPG 61). Weather complete, no nulls in temp/vis/wind (gust 97 % null both sets). Null parity: only NM-derived columns differ; dest_distance_km 0.52 vs 0.29 %. Arrivals in ranking carry full TAXITIME/BLOCK; taxi-in medians within ~50 s except EHAM Jul (335 vs 412). All timestamps Datetime(us, UTC). Routing key proxy-null sends 46 training / 7 ranking matched rows to model B. Ranking months Jan 2026 152,719 DEP, Jul 2026 192,122, full coverage.
