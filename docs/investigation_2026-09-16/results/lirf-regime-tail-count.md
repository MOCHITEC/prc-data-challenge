# Item 3 `lirf-regime-tail-count` — verification result (2026-09-16)

Script: `docs/investigation_2026-09-16/scripts/lirf-regime-tail-count.py` (no training, `POLARS_MAX_THREADS=1`, column-selected scans; ~4 min).
Definitions follow plan §0: "unmatched" = routing definition (`AOBT_3_flt` null → model B) unless stated; `ms` = `MVT_TIME − SCHED_TIME` (verified identical to feature `mvt_minus_sched` on all training and ranking rows, 0 nulls); `fill10` = |BLOCK − SCHED| ≤ 10 s; `y` = `TAXITIME_SEC_mvt`. Thresholds 2 h = 7,200, 4 h = 14,400, 12 h = 43,200.

## 1. Hypothesis as stated (plan §1 item 3)

LIRF unmatched rows are a distinct data source (no aircraft type, sched-filled block) present at the same density in the ranking set; the LB score, the differences between submissions, and the CV/LB disagreement are decided by a few dozen LIRF rows whose count varies per evaluation set.

Sub-claims tested: (A) distinct data source; (B) same density in ranking; (C) LB score decided by a few dozen LIRF rows; (D) differences between submissions decided by those rows; (E) CV/LB disagreement decided by the per-set count of those rows.

## 2. Procedure actually executed

1. `feat_train.parquet` (2,085,047 rows), 12 columns: per ADEP × unmatched profile (n, actype-null rate, fill10/fill120, y median, y ≥ 2 h/4 h/12 h counts, ms ≥ 2 h count, FLIGHT_RULE). Both unmatched definitions counted.
2. `feat_rank.parquet` (344,841 rows), 12 columns: same counts per month (Jan/Jul 2026); density expressed as LIRF-unmatched / LIRF departures and as LIRF share of all unmatched rows, compared with each 2025 training month.
3. `experiments/oof_main-48c19bc830.parquet` joined to training on `MVT_ID_mvt`: SSE share by ADEP × unmatched, top-k rows by squared error and their composition, per-month RMSE with and without LIRF-unmatched rows.
4. Submission files joined to `data/raw/submitting.parquet` / `feat_rank` on `MVT_ID_mvt` (344,841 unique ids, full intersection): v2 = `outputs/v2_two_clock.parquet`; v3, v4 = `scratchpad/uploaded/v3.parquet`, `v4.parquet` (available); v7 = `outputs/main.parquet` (its 09-15 15:52 timestamp matches the `oof_main-48c19bc830` pipeline entry; `submissions.jsonl` records v7 as `outputs/main.parquet` uploaded 15:53); v8 = `outputs/det_a.parquet`; v9 = `outputs/probe_unmatched_from_v4.parquet`. **v5 and v6 uploaded files are unavailable** (both were `outputs/main.parquet`, since overwritten). `outputs/v5_unmatched_capacity.parquet` (09-14 02:04) is included as `v5cand`; `outputs/ab_split.parquet` (255/20 configuration) as a reference. Per-submission unmatched prediction table; pairwise Σgap² by airport; top-20 disagreement rows; predictions on the 181 LIRF unmatched ms ≥ 2 h rows.
5. LB identities from v4 / v8 / v9 (v9 verified as v8 on every matched row and v4 on every unmatched row): SSE(v9) − SSE(v4) = matched-side gap, SSE(v8) − SSE(v9) = unmatched-side gap. Implied unmatched RMSE per LB score with matched RMSE fixed at 218 and 226.

## 3. Results

### 3.1 Training: LIRF unmatched vs all other ADEP × group (S1)

Counts: 2,085,047 rows; unmatched (routing) 22,470; FLIGHT_ID-null 22,424; 46 rows AOBT_3-null with FLIGHT_ID present; 0 rows the other way.

| ADEP | unm | n | actype_null | fill10 | fill120 | y_med | share y≥2h | n y≥2h | n y≥4h | n y≥12h | n ms≥2h |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **LIRF** | **true** | **1,488** | **0.9993** | **0.485** | **0.486** | **3,922** | **0.295** | **439** | **115** | **30** | **627** |
| LIRF | false | 159,216 | 0.0 | 0.178 | 0.257 | 1,024 | 0.0003 | 41 | 1 | 1 | 2,751 |
| LFPG | true | 3,773 | 0.0013 | 0.003 | 0.014 | 1,023 | 0.0019 | 7 | 2 | 2 | 990 |
| EHAM | true | 4,084 | 0.0002 | 0.027 | 0.062 | 538.5 | 0.0002 | 1 | 0 | 0 | 705 |
| LTFM | true | 3,645 | 0.0019 | 0.050 | 0.179 | 1,009 | 0.0052 | 19 | 0 | 0 | 266 |
| LSZH | true | 2,291 | 0.0031 | 0.023 | 0.078 | 673 | 0.0017 | 4 | 2 | 1 | 233 |
| EGLL | true | 1,414 | 0.0 | 0.003 | 0.005 | 1,384 | 0.0078 | 11 | 0 | 0 | 513 |
| EDDF / EDDM / LEBL / LEMD | true | 1,956 / 1,010 / 1,862 / 947 | 0.0 | ≤ 0.019 | ≤ 0.024 | 891–1,096 | 0.0 | 0 | 0 | 0 | 228–581 |
| all 10 airports | false | 132,616–269,320 | 0.0 | 0.011–0.057 (LIRF 0.178) | 0.114–0.202 (LIRF 0.257) | 712–1,319 | ≤ 0.0003 | 0–41 | 0–2 | 0–1 | 1,114–5,748 |

LIRF unmatched detail: 1,487/1,488 AIRCRAFT_TYPE_mvt null; FLIGHT_RULE_mvt single value "I"; fill10 0.4845, fill120 0.4859, |y − ms| ≤ 60 s 0.4852. LIRF unmatched with ms ≥ 2 h: 627 rows, 627 actype-null, fill10 0.6316, |y − ms| ≤ 60 s 0.6316, runway 25 on 554, y median 8,518.

Share of all training tail rows that are LIRF unmatched: y ≥ 2 h 439/584 (unmatched 481); y ≥ 4 h 115/122 (unmatched 119); y ≥ 12 h 30/34 (unmatched 33); y > 20 h 13/16 (unmatched 15). LIRF matched: 41 / 1 / 1 / 1.

### 3.2 Per-month LIRF unmatched counts, training 2025 vs ranking 2026 (S2)

| set | month | LIRF unm | actype null | ms≥2h | ms≥4h | ms≥12h | ms max | fill10 | y≥2h | y≥4h | y≥12h | y med | LIRF dep | all unm | LIRF unm % of LIRF dep | LIRF % of all unm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train | 1 | 60 | 60 | 24 | 6 | 2 | 60,118 | 0.417 | 13 | 6 | 2 | 1,316 | 11,317 | 1,435 | 0.53 | 4.2 |
| train | 2 | 58 | 58 | 23 | 6 | 1 | 131,163 | 0.759 | 21 | 6 | 1 | 6,175 | 10,703 | 1,110 | 0.54 | 5.2 |
| train | 3 | 59 | 59 | 27 | 4 | 1 | 65,463 | 0.729 | 28 | 4 | 1 | 7,130 | 12,544 | 1,082 | 0.47 | 5.5 |
| train | 4 | 70 | 69 | 31 | 7 | 1 | 59,640 | 0.671 | 25 | 6 | 1 | 5,094.5 | 14,028 | 1,085 | 0.50 | 6.5 |
| train | 5 | 99 | 99 | 35 | 6 | 0 | 22,794 | 0.465 | 32 | 7 | 0 | 4,974 | 14,541 | 1,605 | 0.68 | 6.2 |
| train | 6 | 185 | 185 | 78 | 12 | 2 | 67,686 | 0.486 | 54 | 10 | 3 | 1,922 | 14,696 | 2,502 | 1.26 | 7.4 |
| train | 7 | 337 | 337 | 157 | 39 | 9 | 93,535 | 0.323 | 84 | 35 | 9 | 1,447 | 15,211 | 3,938 | 2.22 | 8.6 |
| train | 8 | 197 | 197 | 87 | 17 | 5 | 53,525 | 0.452 | 63 | 15 | 5 | 3,127 | 15,459 | 2,330 | 1.27 | 8.5 |
| train | 9 | 168 | 168 | 71 | 15 | 4 | 64,854 | 0.435 | 42 | 12 | 4 | 1,445 | 14,667 | 2,465 | 1.15 | 6.8 |
| train | 10 | 115 | 115 | 36 | 5 | 0 | 25,559 | 0.487 | 28 | 4 | 0 | 2,397 | 14,142 | 1,913 | 0.81 | 6.0 |
| train | 11 | 52 | 52 | 21 | 4 | 2 | 60,599 | 0.615 | 18 | 4 | 2 | 5,733 | 11,328 | 1,356 | 0.46 | 3.8 |
| train | 12 | 88 | 88 | 37 | 6 | 2 | 86,102 | 0.761 | 31 | 6 | 2 | 6,034.5 | 12,068 | 1,649 | 0.73 | 5.3 |
| **rank** | **1** | **107** | **107** | **52** | **15** | **3** | **56,520** | n/a | n/a | n/a | n/a | n/a | **11,061** | **2,453** | **0.97** | **4.4** |
| **rank** | **7** | **276** | **276** | **129** | **28** | **8** | **111,654** | n/a | n/a | n/a | n/a | n/a | **15,838** | **2,837** | **1.74** | **9.7** |

Totals: training LIRF unmatched 1,488 (ms ≥ 2 h 627, ≥ 4 h 127, ≥ 12 h 29); training Jan+Jul 2025: 397 (ms ≥ 2 h 181, ≥ 4 h 45, ≥ 12 h 11; y ≥ 2 h 97, ≥ 4 h 41, ≥ 12 h 11); ranking Jan+Jul 2026: 383 (ms ≥ 2 h 181, ≥ 4 h 43, ≥ 12 h 11; ms max 111,654; median ms given ≥ 2 h 10,135). Ranking BLOCK_TIME null on 100 % of rows (fill rate not observable). FLIGHT_ID-null definition gives identical LIRF counts (107 / 276; 52 / 129). Ranking unmatched totals: Jan 2,453 / Jul 2,837 (routing), 2,449 / 2,834 (FLIGHT_ID-null).

Ranking unmatched rows with ms ≥ 2 h by airport: EHAM 559 (ms ≥ 12 h: 8), LFPG 258 (2), LIRF 181 (11), LEBL 155, LSZH 152, EDDF 111 (4), EDDM 87, EGLL 86, LEMD 84, LTFM 48. Ranking unmatched per airport, Jan/Jul: EDDF 212/209, EDDM 119/119, EGLL 112/136, EHAM 805/401, LEBL 77/365, LEMD 63/175, LFPG 325/409, LIRF 107/276, LSZH 300/359, LTFM 333/388.

### 3.3 OOF `oof_main-48c19bc830` (S3)

672,428 rows, months 1, 7, 11, 12; RMSE 334.58; unmatched 8,378 rows RMSE 2,296.1; matched RMSE 216.4; unmatched-side SSE share 58.7 %.

| ADEP × unm | n | SSE % | RMSE | n y≥2h |
|---|---|---|---|---|
| LIRF unm | 537 | 36.77 | 7,179 | 146 |
| LFPG unm | 1,565 | 15.48 | 2,729 | 6 |
| LIRF matched | 49,387 | 7.90 | 347 | 17 |
| LTFM matched | 90,041 | 6.57 | 234 | 1 |
| LFPG matched | 76,471 | 6.15 | 246 | 10 |
| EGLL matched | 78,986 | 5.55 | 230 | 16 |
| other 6 matched | 42,287–78,286 | 1.60–3.03 | 162–197 | 0–2 |
| EHAM unm | 1,353 | 1.54 | 927 | 1 |
| other 7 unm | 402–1,297 | 0.10–1.20 | 418–1,099 | 0–7 |

Top-k rows by squared error: top-10 = 39.1 % of SSE (8 LIRF-unm, 2 other-unm, all y ≥ 12 h); top-20 = 43.4 % (15 LIRF-unm, 5 other-unm, 0 matched; 13 y ≥ 12 h); top-50 = 48.3 % (29 / 21 / 0); top-100 = 51.5 % (64 / 34 / 2); top-200 = 54.8 % (123 / 61 / 16); top-2000 = 65.2 % (341 / 233 / 1,426). RMSE without top-20 = 251.8; without top-2000 = 197.8. Unmatched RMSE excluding LIRF-unmatched rows = 1,450.2 (n 7,841); LIRF-unmatched RMSE 7,179.3 (n 537).

Top-20 OOF rows: 2 LFPG rollovers (y 84,240 / 58,206, ms 1,740 / 2,043, pred 922 / 992), 8 LIRF rollovers (y 87,170–88,132, ms 56,520–93,535, pred 32,352–51,678), 5 LIRF fill10 rows with y = ms (25,383 / 64,437 / 56,633 / 39,120 / 47,767; preds 49,316 / 43,022 / 37,844 / 20,890 / 32,399), 2 LIRF false alarms (y 1,377 / 1,926, ms 20,878 / 12,723, pred 19,659 / 17,244), EDDF / LEBL / LFPG false alarms (y 661 / 865 / 652, ms 89,185 / 45,569 / 89,156, pred 16,880 / 16,216 / 15,812).

Per OOF month:

| mo | n | RMSE | LIRF unm | LIRF unm y≥2h | y≥12h | LIRF-unm SSE % | RMSE without LIRF-unm rows |
|---|---|---|---|---|---|---|---|
| 1 | 153,706 | 373.0 | 60 | 13 | 2 | 17.9 | 338.0 |
| 7 | 190,713 | 378.4 | 337 | 84 | 9 | 49.6 | 268.6 |
| 11 | 162,332 | 291.5 | 52 | 18 | 2 | 45.8 | 214.5 |
| 12 | 165,677 | 277.9 | 88 | 31 | 2 | 31.1 | 230.6 |

### 3.4 Submission files on the 5,290 unmatched ranking rows (S4)

v9 probe: equals v8 on all 339,551 matched rows and v4 on all 5,290 unmatched rows (5,266 rows where v9 ≠ v8, i.e. 24 unmatched rows where v4 = v8 by coincidence).

| sub (LB) | med | mean | p99 | max | n>7200 | n>14400 | n>43200 | n>43200 LIRF | n>7200 LIRF |
|---|---|---|---|---|---|---|---|---|---|
| v2 (312.02) | 1,062.5 | 1,611.7 | 10,243 | 68,253 | 104 | 33 | 2 | 2 | 95 |
| v3 (306.29) | 1,092 | 1,674.6 | 9,358 | 107,724 | 82 | 31 | 16 | 16 | 82 |
| v4 (305.03) | 1,112 | 1,702.5 | 9,452 | 108,552 | 83 | 31 | 18 | 18 | 81 |
| v7 (342.59) | 1,082 | 1,608.4 | 10,153 | 66,693 | 100 | 33 | 2 | 2 | 90 |
| v8 (343.80) | 1,066 | 1,608.0 | 10,183 | 68,581 | 101 | 35 | 2 | 2 | 95 |
| v9 (306.46) | = v4 | = v4 | = v4 | = v4 | 83 | 31 | 18 | 18 | 81 |
| v5cand (file 09-14 02:04; upload identity unverified) | 1,060 | 1,559.1 | 9,693 | 61,356 | 106 | 32 | 2 | 2 | 88 |
| ab_split (255/20, never uploaded) | 1,073 | 1,644.7 | 10,119 | 71,322 | 102 | 35 | 11 | 11 | 95 |

`>` and `≥` give identical counts at 7,200 and 43,200. All rows (344,841): predictions ≥ 2 h: v2 128, v3 104, v4 103, v7 122, v8 124, v9 106, v5cand 130 (matched-row share 20–24); ≥ 12 h: v2 3, v3 17, v4 19, v7 3, v8 3, v9 19, v5cand 3 (matched 1 in every file). Training label rates: unmatched y ≥ 2 h 0.02141 (× 5,290 = 113.2), y ≥ 12 h 0.00147 (× 5,290 = 7.8); all rows y ≥ 2 h 0.000280 (× 344,841 = 96.6), y ≥ 12 h 0.0000163 (× 344,841 = 5.6).

Pairwise gaps (unmatched rows; matched RMS in last column):

| pair | RMS unm | LIRF share of Σgap² | ms≥2h rows share | top-20 share | RMS on 1,721 ms≥2h rows | RMS elsewhere | matched RMS |
|---|---|---|---|---|---|---|---|
| v8 − v4 | 1,735.5 | 90.3 % | 95.1 % | 80.4 % | 2,967 | 469 | 73.8 |
| v7 − v4 | 1,776.6 | 90.5 % | 95.7 % | 82.0 % | 3,048 | 446 | 64.3 |
| v2 − v4 | 1,748.5 | 89.1 % | 95.1 % | 80.2 % | 2,989 | 472 | 72.9 |
| v3 − v4 | 281.4 | 24.4 % | 62.6 % | 19.6 % | 390 | 209 | 40.2 |
| v8 − v2 | 314.5 | 46.9 % | 61.2 % | 19.1 % | 431 | 239 | 38.4 |
| v7 − v8 | 342.7 | 46.0 % | 63.9 % | 22.5 % | 480 | 251 | 45.9 |
| v8 − v5cand | 579.5 | 46.2 % | 77.9 % | 35.6 % | 897 | 332 | 38.1 |

Σ(v8 − v4)² by airport: LIRF 383 rows 1.438e10 (90.3 %, RMS 6,128); EHAM 2.6 %; EGLL 2.5 % (RMS 1,278); LSZH 1.3 %; LFPG 1.0 %; others ≤ 0.6 %.

Top-20 disagreement rows v8 vs v4 (unmatched; 18 LIRF + 2 EGLL; 18 actype-null; all 20 ms ≥ 2 h, 10 ms ≥ 12 h; 6 Jan / 14 Jul):

| MVT_ID | ADEP | mo | ms | rwy | v2 | v3 | v4 | v7 | v8 | v5cand | (v8−v4)² |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 213145579 | LIRF | 7 | 111,654 | 25 | 40,865 | 107,724 | 108,552 | 39,636 | 40,267 | 34,027 | 4.66e9 |
| 213152353 | LIRF | 7 | 87,484 | 16R | 33,882 | 80,051 | 79,686 | 32,688 | 34,552 | 30,759 | 2.04e9 |
| 213154026 | LIRF | 7 | 60,656 | 25 | 31,617 | 64,154 | 66,225 | 31,551 | 33,197 | 29,731 | 1.09e9 |
| 202680942 | LIRF | 1 | 56,520 | 25 | 34,836 | 59,495 | 60,401 | 33,444 | 34,236 | 32,003 | 6.85e8 |
| 213147307 | LIRF | 7 | 33,057 | 25 | 32,453 | 57,651 | 57,016 | 30,493 | 32,024 | 32,982 | 6.25e8 |
| 202687258 | LIRF | 1 | 53,520 | 25 | 32,309 | 54,957 | 55,198 | 30,644 | 31,617 | 29,492 | 5.56e8 |
| 202681830 | LIRF | 1 | 40,615 | 25 | 28,336 | 50,636 | 51,087 | 28,662 | 28,965 | 26,243 | 4.89e8 |
| 213143669 | LIRF | 7 | 48,539 | 25 | 34,887 | 54,267 | 54,226 | 33,485 | 33,640 | 30,657 | 4.24e8 |
| 202680695 | LIRF | 1 | 39,365 | 25 | 33,842 | 53,610 | 52,699 | 31,757 | 32,651 | 30,872 | 4.02e8 |
| 213142469 | LIRF | 7 | 47,705 | 25 | 36,278 | 53,103 | 53,459 | 35,434 | 37,306 | 33,896 | 2.61e8 |
| 213145828 | LIRF | 7 | 36,062 | 16R | 33,034 | 47,004 | 49,002 | 32,025 | 33,741 | 28,408 | 2.33e8 |
| 202685083 | LIRF | 1 | 48,841 | 25 | 37,301 | 52,095 | 52,545 | 36,782 | 37,704 | 33,791 | 2.20e8 |
| 213153677 | LIRF | 7 | 47,337 | 16R | 68,253 | 52,066 | 53,155 | 66,693 | 66,778 | 56,026 | 1.86e8 |
| 213142471 | LIRF | 7 | 25,564 | 25 | 34,471 | 21,783 | 21,945 | 33,825 | 34,899 | 31,381 | 1.68e8 |
| 213154831 | LIRF | 7 | 39,060 | 25 | 31,059 | 42,958 | 42,993 | 30,701 | 30,687 | 29,791 | 1.51e8 |
| 212234819 | EGLL | 7 | 25,438 | 09R | 15,608 | 129 | 100 | 12,193 | 12,299 | 8,124 | 1.49e8 |
| 213143319 | LIRF | 7 | 48,776 | 25 | 37,077 | 50,364 | 50,937 | 35,683 | 38,739 | 38,226 | 1.49e8 |
| 212237731 | EGLL | 7 | 13,199 | 09R | 11,675 | 994 | 1,603 | 12,369 | 12,700 | 11,067 | 1.23e8 |
| 213142746 | LIRF | 7 | 41,395 | 25 | 36,567 | 46,872 | 47,148 | 35,334 | 36,923 | 33,493 | 1.05e8 |
| 202681707 | LIRF | 1 | 25,619 | 25 | 31,384 | 20,399 | 21,201 | 30,953 | 31,170 | 28,043 | 9.94e7 |

On these 20 rows: median v4 − ms = +3,907, median v8 − ms = −10,218; median v4/ms 1.084, v8/ms 0.784.

On the 181 ranking LIRF unmatched ms ≥ 2 h rows (ms median 10,135): prediction median v2 7,252 / v3 6,697 / v4 6,654 / v7 7,041 / v8 7,067 / v5cand 6,727; p90 32,659 / 42,452 / 43,127 / 31,757 / 33,197 / 30,657; n pred ≥ 2 h 91 / 82 / 81 / 86 / 89 / 87; n pred ≥ 12 h 2 / 16 / 18 / 2 / 2 / 2; median pred/ms 0.696 / 0.722 / 0.712 / 0.690 / 0.693 / 0.656. The 11 rows with ms ≥ 12 h (47,337–111,654): v3/v4 predict 50,364–108,552 (≈ ms on 9 of 11; 52,066 / 53,155 and 60,338 / 60,845 on the two 16R rows with ms 47,337 / 61,322), v2/v7/v8 predict 28,965–40,267 on 9 of them and 66,037–68,581 on the two 16R rows; v5cand 29,492–61,356.

### 3.5 Leaderboard identities (S5)

N = 344,841; unmatched 5,290; matched 339,551. SSE(v8) − SSE(v4) = (343.7964² − 305.027²)·N = 8.674e9. Because v9 = v8 on matched rows and v4 on unmatched rows: matched part = SSE(v9) − SSE(v4) = 3.012e8 (3.5 %); unmatched part = SSE(v8) − SSE(v9) = 8.373e9 (96.5 %). If matched RMSE(v4) = 218, matched RMSE(v8) = 220.03. Local Σ(v8 − v4)² on unmatched = 1.593e10; LB unmatched gap / Σgap² = 0.53 (the cross term 2·Σ gap·(v4 − y) = −7.56e9 requires labels to decompose further).

Implied unmatched RMSE with matched fixed at 218 (226): v1 2,174 (2,120); v2 1,816 (1,752); v3 1,751 (1,684); v4 1,736 (1,669); v5 1,961 (1,902); v6 2,228 (2,176); v7 2,145 (2,091); v8 2,157 (2,104); v9 1,753 (1,686).

The unmatched SSE gap v8 − v4 (8.373e9) spread evenly: over 20 rows = 20,461 s RMS per row; over 181 rows = 6,801; over 5,290 = 1,258.

Hypothetical label scenarios (inference, not measurement): if all 181 LIRF unmatched ms ≥ 2 h rows had y = ms, SSE(v8) − SSE(v4) on them = 9.93e9 (119 % of the LB unmatched gap); if all had y = 87,600 (rollover), 1.82e10 (217 %); if only the 11 ms ≥ 12 h rows had y = ms, 1.04e10 (124 %).

## 4. Verdict per sub-claim

- **(A) LIRF unmatched rows are a distinct data source (actype-null, sched-filled block): CONFIRMED.** 1,487/1,488 actype-null vs ≤ 0.31 % in every other ADEP × group (matched groups 0.0); fill10 0.485 vs ≤ 0.057 for every other unmatched airport and 0.178 for LIRF matched; y median 3,922 vs 538–1,384 elsewhere; 439/584 of all y ≥ 2 h and 115/122 of y ≥ 4 h training rows are LIRF unmatched. The independent re-derivation reproduces M's 1,488 / 1,487 / 3,922 / 29.5 % and D's 627 / 554 exactly.
- **(B) Present at the same density in the ranking set: CONFIRMED for the Jan+Jul aggregate, REFUTED month-by-month.** Aggregate: 383 ranking vs 397 training (Jan+Jul 2025); ms ≥ 2 h 181 vs 181; ms ≥ 4 h 43 vs 45; ms ≥ 12 h 11 vs 11 (M's "same density" numbers reproduced exactly). Per month: Jan 2026 107 vs Jan 2025 60 (0.97 % vs 0.53 % of LIRF departures; ms ≥ 2 h 52 vs 24; ms ≥ 4 h 15 vs 6), Jul 2026 276 vs Jul 2025 337 (1.74 % vs 2.22 %; ms ≥ 2 h 129 vs 157). Training months range 52–337 (0.46–2.22 % of LIRF departures). The sched-fill rate of ranking rows is unobservable (BLOCK_TIME null on 100 % of ranking rows).
- **(C) The LB score is decided by a few dozen LIRF rows: UNDETERMINED (labels unavailable).** Decided locally: (i) 96.5 % of the v4→v8 LB SSE gap sits on the 5,290 unmatched rows (exact identity from v9); (ii) 90.3 % of Σ(v8 − v4)² on those rows is LIRF and 80.4 % is in 20 rows (18 LIRF, 2 EGLL); (iii) on OOF, LIRF unmatched 537 rows carry 36.8 % of SSE and the top-20 OOF rows (15 LIRF-unm) carry 43.4 %. Which unmatched rows carry the LB gap cannot be located without labels; the LB gap is 0.53 × the local Σgap², so the disagreement rows are not uniformly "v4 right, v8 wrong".
- **(D) Differences between submissions are decided by those rows: CONFIRMED for the v4-family vs v2/v7/v8-family difference; REFUTED as a general statement.** v8 − v4, v7 − v4, v2 − v4: LIRF share 89–90.5 %, top-20 share 80–82 %, RMS on ms ≥ 2 h rows 2,967–3,048 vs 446–472 elsewhere; matched RMS 64–74 (LB-verified matched contribution 3.5 %). But v3 − v4 (LB 306.29 vs 305.03), v8 − v2 (343.80 vs 312.02), v7 − v8 and v8 − v5cand have LIRF shares 24–47 % and top-20 shares 19–36 %; the 32 s v2→v8 LB change is not accompanied by an LIRF-tail prediction change (v2 and v8 predict 40,865 / 40,267 etc. on the same rows; unmatched RMS v8 − v2 = 314.5, matched 38.4).
- **(E) The CV/LB disagreement is decided by the per-set count of these rows: CONFIRMED on the CV side, UNDETERMINED on the LB side.** Per OOF month the LIRF-unmatched SSE share is 17.9 % / 49.6 % / 45.8 % / 31.1 % (Jan / Jul / Nov / Dec) and per-month RMSE 373.0 / 378.4 / 291.5 / 277.9 becomes 338.0 / 268.6 / 214.5 / 230.6 without those 60 / 337 / 52 / 88 rows; the ranking set's counts (107 / 276) match neither training month. Whether the LB responds the same way requires labels.

Additional facts vs the source reports: P's table for v2 / v3 / v4 / v7 / v8 is reproduced to within rounding of p99 (10,243 vs 10,256; 9,452 vs 9,457; 10,153 vs 10,169). P's v5 row matches `outputs/v5_unmatched_capacity.parquet` (1,060 / 1,559 / 9,693 / 61,356 / 106 / 32 / 2), a file dated 09-14 02:04 while the uploaded v5 was `outputs/main.parquet` at 09:54; the two cannot be shown identical. P's "top-20 rows carry 73.2 % of Σgap²; all 20 are LIRF" is not reproduced: this run gives 80.4 % with 18 LIRF and 2 EGLL (ms 25,438 / 13,199; v4 100 / 1,603, v8 12,299 / 12,700). D's "≥ 2 h on 103–130, ≥ 12 h on 3–19" is reproduced exactly on all rows including v5cand. M's RMS 2,967 / 469 and the 1,721-row count are reproduced. E H1's requested measurement (unmatched SSE share by airport on OOF): LIRF 36.77 %, LFPG 15.48 %, EHAM 1.54 %, others ≤ 1.20 %; unmatched side total 58.7 % on this OOF (H's 48.1 % is from a different OOF, plan §4-9).

## 5. What cannot be determined locally

- Labels of the 5,290 unmatched ranking rows, hence: the LB-side SSE share of LIRF rows; how many of the 11 (ms ≥ 12 h) / 43 (≥ 4 h) / 181 (≥ 2 h) LIRF rows are sched-filled (y = ms) vs rollover vs ordinary; the ranking fill rate (BLOCK_TIME is null on every ranking row).
- Which unmatched rows carry the 8.373e9 unmatched-side LB gap between v4 and v8; only the total is fixed by the v9 identity.
- The exact uploaded v5 and v6 prediction files (overwritten `outputs/main.parquet`).
- Matched RMSE on the ranking set (218 / 226 are assumptions used only for the implied-unmatched-RMSE table).

## 6. Facts vs inference

**Facts (measured in this run):** every number in §3; the v9 = v8-matched + v4-unmatched identity; the 96.5 % / 3.5 % LB split of the v4→v8 SSE gap; the 383 vs 397 / 181 vs 181 / 11 vs 11 counts; the per-month 107 vs 60 and 276 vs 337; the OOF SSE shares and per-month decomposition; the pairwise Σgap² tables.

**Inference (not measured):** that ranking LIRF rows are sched-filled at the training rate (BLOCK unobservable); that the LB gap is concentrated in the top-20 disagreement rows (only their prediction gap is concentrated); the implied unmatched RMSE table (assumes matched RMSE); the hypothetical y = ms / rollover SSE computations in §3.5; that `v5_unmatched_capacity.parquet` is the uploaded v5.
