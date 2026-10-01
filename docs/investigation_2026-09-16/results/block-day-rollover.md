# Result — item 2 `block-day-rollover` (2026-09-16)

Script: `docs/investigation_2026-09-16/scripts/block-day-rollover.py` (no training, `POLARS_MAX_THREADS=1`, lazy scans, 19 of 132 columns).
Inputs: `data/work/feat_train.parquet` (2,085,047 rows, all `PHASE_mvt == "DEP"`), `data/work/feat_rank.parquet` (344,841 rows), `experiments/oof_main-48c19bc830.parquet` (672,428 non-null rows), `outputs/main.parquet`, and every other `experiments/oof_*.parquet` for §3.6.
Notation as in plan §0: `y = TAXITIME_SEC_mvt`, `ms = MVT − SCHED` (= `mvt_minus_sched`), unmatched = **routing** (`mvt_minus_aobt3` null: 22,470 / 5,290) unless stated; D's definition (`FLIGHT_ID_mvt` null: 22,424 / 5,283) is reported where it differs. "Rollover row" below = training row with `y > 72,000` (20 h) unless another definition is named.

## 1. Hypothesis as stated (plan §1 item 2)

A handful of labels per year are `BLOCK_TIME` with the date off by one day (label = true taxi + 86,400); these rows dominate SSE and are predictable only as a mixture component.
Sub-claims taken from the "Verified" block: (a) 16 training rows y > 20 h, 13 of 16 with y − 86,400 in 602…1,992 (D); (b) MVT and BLOCK differ by 1 day + 13–30 min (M); (c) OOF: 10 rows y > 20 h = 34.7 % of SSE, RMSE 51,121, pred median 37,131 (D); 8 rows ≈ 86,400 + taxi = 52.7 % of unmatched SSE (≈ 25 % of total), RMSE 53,927; top-25 = 76.3 % (M); 10 rows ≈ 1 day = 24.1 % (H); (d) training ms bands 20–28 h unmatched 23 (17.4 % y ≥ 2 h), > 28 h unmatched 3 (D); (e) ranking 6 + 1 unmatched, 36 + 6 matched rows in those bands (D); 11 LIRF ranking rows mms ≥ 12 h, training split 62 % sched-filled / 38 % rollover (M).

## 2. Procedure executed

1. Training: counts of y ≥ 2 h / 4 h / 12 h / > 20 h / ≥ 80,000 / ≥ 86,400 / ≥ 28 h, negatives, `y == MVT − BLOCK` identity, split by both unmatched definitions.
2. Full list of the y > 20 h rows with SCHED / BLOCK / MVT / y / ms / y − 86,400 / BLOCK − SCHED / date differences / fill flags / AOBT_3 / MVT_ID.
3. Per-row interpretation test: implied BLOCK − SCHED if the BLOCK date is one day early, vs implied MVT − SCHED if the MVT date is one day late.
4. Listing of 12 h ≤ y ≤ 20 h rows (rollovers hiding below 20 h?).
5. Alternative mechanism: search for a same-(ADEP, FLIGHT) departure whose MVT date equals the rollover row's BLOCK date.
6. Training ms bands 12–20 h / 20–28 h / > 28 h × matched/unmatched (both definitions): n, y ≥ 2 h share, n y > 20 h, y == ms share, fill share.
7. OOF main-48c19bc830 joined on MVT_ID: SSE share of the rollover rows under D's, M's and H's definitions, with the denominator stated (total SSE vs unmatched SSE); reconstruction of M's 8-row set; RMSE with the rows removed; top-25 unmatched rows.
8. Same SSE-share statistics on every other `experiments/oof_*.parquet` (to locate H's 24.1 %).
9. Ranking: BLOCK / label / ms nullness; ms bands × matched/unmatched (both definitions); the 11 LIRF unmatched rows with ms ≥ 12 h with `outputs/main.parquet` predictions.
10. Training LIRF unmatched rows with ms ≥ 12 h: class (rollover y > 20 h / sched-filled |y − ms| ≤ 60 / other) against visible features (ms bucket, take-off hour, runway, flight, dow, month).

## 3. Results

### 3.1 Training tail counts (2,085,047 rows)

| quantity | value |
|---|---|
| unmatched (routing / FLIGHT_ID) | 22,470 / 22,424 |
| y ≥ 7,200 (2 h) | 584 |
| y ≥ 14,400 (4 h) | 122 |
| y ≥ 43,200 (12 h) | 34 (33 unmatched under both definitions + 1 matched) |
| y > 72,000 (20 h) | **16** (15 unmatched under both definitions + 1 matched) |
| y ≥ 80,000 | 16 (same rows) |
| y ≥ 86,400 | 15 |
| y > 100,800 (28 h) | 1 |
| y ≤ −72,000 | 0 (negative labels: 369, none below −20 h) |
| rows with |y − (MVT − BLOCK)| > 1 s | 0 |

### 3.2 Full list of the 16 training rows y > 20 h (sorted by y)

`unm` = routing definition (identical to FLIGHT_ID definition on these rows). `y−1d` = y − 86,400. `B−S` = BLOCK − SCHED (s). `OOF pred` from main-48c19bc830 (— = month not in that OOF; OOF covers months 1, 7, 11, 12 only).

| # | ADEP | FLIGHT | unm | SCHED (UTC) | BLOCK (UTC) | MVT (UTC) | y | ms | y−1d | B−S | RWY | actype | OOF pred |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | LIRF | NOSOS739 | yes | 2025-02-10 05:25:00 | 2025-02-10 05:24:56 | 2025-02-11 17:51:03 | 131,167 | 131,163 | 44,767 | −4 | 25 | null | — |
| 2 | LIRF | AMX071 | yes | 2025-06-24 21:15:00 | 2025-06-24 15:29:54 | 2025-06-25 16:03:06 | 88,392 | 67,686 | 1,992 | −20,706 | 25 | null | — |
| 3 | LIRF | CHHHH438 | yes | 2025-07-20 09:05:00 | 2025-07-20 08:00:04 | 2025-07-21 08:28:56 | 88,132 | 84,236 | 1,732 | −3,896 | 25 | null | 42,846 |
| 4 | LIRF | NOSOS610 | yes | 2025-12-27 18:40:00 | 2025-12-27 18:14:57 | 2025-12-28 18:35:02 | 87,605 | 86,102 | 1,205 | −1,503 | 25 | null | 34,093 |
| 5 | LIRF | BAWW561D | yes | 2025-11-11 19:25:00 | 2025-11-11 11:55:01 | 2025-11-12 12:14:59 | 87,598 | 60,599 | 1,198 | −26,999 | 25 | null | 34,540 |
| 6 | LIRF | ITYTY630 | yes | 2025-09-28 08:35:00 | 2025-09-27 22:18:58 | 2025-09-28 22:38:01 | 87,543 | 50,581 | 1,143 | −36,962 | 34L | null | — |
| 7 | LIRF | ITYTY680 | yes | 2025-11-03 21:40:00 | 2025-11-03 13:04:00 | 2025-11-04 13:22:00 | 87,480 | 56,520 | 1,080 | −30,960 | 34L | null | 32,352 |
| 8 | LIRF | VOE6BB | yes | 2025-07-02 20:40:00 | 2025-07-02 16:44:59 | 2025-07-03 17:01:00 | 87,361 | 73,260 | 961 | −14,101 | 25 | null | 39,292 |
| 9 | LSZH | NJE389D | yes | 2025-05-14 14:40:00 | 2025-05-13 14:39:58 | 2025-05-14 14:55:39 | 87,341 | 939 | 941 | −86,402 | 28 | C56X | — |
| 10 | LIRF | SWUU1919 | yes | 2025-03-27 16:30:00 | 2025-03-27 10:26:56 | 2025-03-28 10:41:03 | 87,247 | 65,463 | 847 | −21,784 | 25 | null | — |
| 11 | LIRF | ITYTY680 | yes | 2025-07-29 20:30:00 | 2025-07-29 12:32:57 | 2025-07-30 12:46:03 | 87,186 | 58,563 | 786 | −28,623 | 25 | null | 51,678 |
| 12 | LIRF | EXSXS1GT | yes | 2025-01-24 10:55:00 | 2025-01-24 03:24:01 | 2025-01-25 03:36:58 | 87,177 | 60,118 | 777 | −27,059 | 25 | null | 34,970 |
| 13 | LIRF | RYRR90BN | yes | 2025-07-13 13:00:00 | 2025-07-13 14:46:05 | 2025-07-14 14:58:55 | 87,170 | 93,535 | 770 | +6,365 | 25 | null | 41,118 |
| 14 | LIRF | ITY332 | yes | 2025-06-25 21:59:00 | 2025-06-25 01:01:06 | 2025-06-26 01:13:54 | 87,168 | 11,694 | 768 | −75,474 | 25 | null | — |
| 15 | LIRF | AEZ146R | **no** | 2025-07-14 20:00:00 | 2025-07-14 19:59:59 | 2025-07-15 20:10:01 | 87,002 | 87,001 | 602 | −1 | 25 | B738 | 84,798 |
| 16 | LFPG | EJU983W | yes | 2025-01-19 18:25:00 | 2025-01-18 19:30:00 | 2025-01-19 18:54:00 | 84,240 | 1,740 | −2,160 | −82,500 | 08L | A319 | 922 |

Row 15 (the only matched one) has `AOBT_3 = 2025-07-14 19:57:00`, `mvt_minus_aobt3 = 87,181`.

### 3.3 Diagnostics on the 16 rows

| observation | value |
|---|---|
| y − 86,400 sorted | −2,160, 602, 768, 770, 777, 786, 847, 941, 961, 1,080, 1,143, 1,198, 1,205, 1,732, 1,992, 44,767 |
| rows with 602 ≤ y − 86,400 ≤ 1,992 | **14** of 16 (D wrote 13) |
| rows with 0 < y − 86,400 < 7,200 | 14 of 16 |
| date(MVT) − date(BLOCK) = 1 day | 16 of 16 (none = 2) |
| date(BLOCK) = date(SCHED) and date(MVT) = date(SCHED) + 1 | 13 of 16 |
| date(BLOCK) = date(SCHED) − 1 and date(MVT) = date(SCHED) | 3 of 16 (rows 6, 9, 16) |
| |ms| < 2 h (MVT consistent with SCHED) | 2 of 16 (rows 9, 16) |
| BLOCK earlier than SCHED by > 1 h | 12 of 16 |
| BLOCK within ±10 s of SCHED (D fill) / ±6 s (M fill) | 2 / 2 (rows 1, 15) |
| |y − ms| ≤ 60 s (y == ms) | 2 (rows 1, 15) |
| |y − ms| < 1,800 s (M's sched-filled rule) | 3 (rows 1, 4, 15) |
| BLOCK seconds-of-minute | 0, 0, 1, 1, 4, 5, 6, 54, 56, 56, 57, 57, 58, 58, 59, 59 (all within 6 s of a minute boundary, incl. LSZH and LFPG) |
| airports | LIRF 14, LSZH 1, LFPG 1 |
| matched / unmatched | 1 / 15 (both definitions) |
| months (2025) | Jan 2, Feb 1, Mar 1, May 1, Jun 2, Jul 5, Sep 1, Nov 2, Dec 1 → 16 rows in 9 distinct months (15 unmatched rows in 9 months) |
| same-(ADEP, FLIGHT) departure with MVT on the BLOCK date | **0** of 16 |

Interpretation test (which single-field date shift makes SCHED / BLOCK / MVT mutually consistent):

| row | if BLOCK date −1: true BLOCK − SCHED | if MVT date +1: true MVT − SCHED |
|---|---|---|
| 1 NOSOS739 | 86,396 | 44,763 |
| 2 AMX071 | 65,694 | −18,714 |
| 3 CHHHH438 | 82,504 | −2,164 |
| 4 NOSOS610 | 84,897 | −298 |
| 5 BAWW561D | 59,401 | −25,801 |
| 6 ITYTY630 | 49,438 | −35,819 |
| 7 ITYTY680 | 55,440 | −29,880 |
| 8 VOE6BB | 72,299 | −13,140 |
| 9 NJE389D (LSZH) | **−2** | −85,461 |
| 10 SWUU1919 | 64,616 | −20,937 |
| 11 ITYTY680 | 57,777 | −27,837 |
| 12 EXSXS1GT | 59,341 | −26,282 |
| 13 RYRR90BN | 92,765 | 7,135 |
| 14 ITY332 | 10,926 | −74,706 |
| 15 AEZ146R (matched) | 86,399 | **601** |
| 16 EJU983W (LFPG) | 3,900 | −84,660 |

Rows 12 h ≤ y ≤ 20 h (n = 18): 17 LIRF unmatched rows with |BLOCK − SCHED| ≤ 6 s and y == ms (43,863…64,849), plus 1 LFPG unmatched row (y 58,206, ms 2,043, BLOCK − SCHED −56,163, y − 86,400 = −28,194). None has y − 86,400 in (0, 7,200). All rows with y ≥ 80,000 are the 16 rows above.

### 3.4 Training ms bands (identical under both unmatched definitions)

| ms band | matched n | matched y ≥ 2 h | matched y > 20 h | unmatched n | unmatched y ≥ 2 h | unmatched y > 20 h | unmatched y == ms | unmatched LIRF |
|---|---|---|---|---|---|---|---|---|
| 12–20 h | 335 | 0.000 | 0 | 49 | 0.490 | 7 | 0.347 | 24 |
| 20–28 h | 167 | 0.006 | 1 (row 15) | **23** | **0.174** | 4 | 0.000 | 4 |
| > 28 h | 49 | 0.000 | 0 | **3** | 0.333 | 1 (row 1) | 0.333 | 1 |

### 3.5 OOF main-48c19bc830 (672,428 rows; months 1, 7, 11, 12; RMSE 334.58)

Unmatched (routing) 8,378 rows, RMSE 2,296.1, 58.68 % of total SSE; (FLIGHT_ID) 8,358 rows, RMSE 2,298.6, 58.67 %. Matched RMSE 216.4.
All 10 training rollover rows in months 1, 7, 11, 12 (2 + 5 + 2 + 1) are present in the OOF.

| row set | n (unmatched) | share of **total** SSE | share of **unmatched** SSE (unmatched rows only) | RMSE | pred median | y median |
|---|---|---|---|---|---|---|
| D: y > 20 h, all rows | 10 (9) | **0.3472** | 0.5916 | 51,121 | 37,131 | 87,274 |
| y > 20 h, unmatched only | 9 (9) | 0.3471 | **0.5916** | 53,881 | 34,970 | 87,361 |
| M-reconstruction: y > 20 h, unmatched, |y − ms| ≥ 1,800 | **8** (8) | 0.3091 | **0.5267** | **53,927** | 37,131 | 87,274 |
| 0 < y − 86,400 < 7,200 (all rows) | 9 (8) | 0.2550 | 0.4345 | 46,178 | 39,292 | 87,361 |
| 0 < y − 86,400 < 7,200, unmatched | 8 (8) | 0.2549 | 0.4344 | 48,973 | 37,131 | 87,420 |
| y ≥ 86,400 | 9 (8) | 0.2550 | 0.4345 | 46,178 | 39,292 | 87,361 |
| H: y ≥ 80,000 | 10 (9) | 0.3472 | 0.5917 | 51,121 | 37,131 | 87,274 |
| y ≥ 12 h | 18 (17) | 0.4068 | 0.6934 | 41,248 | 38,101 | 85,621 |
| y ≥ 4 h | 55 (53) | 0.4343 | 0.7402 | 24,381 | 24,287 | 25,383 |
| y ≥ 2 h | 210 (163) | 0.4601 | 0.7842 | 12,843 | 7,698 | 9,632 |
| y > 20 h at LIRF | 9 (8) | 0.2550 | 0.4345 | 46,178 | 39,292 | 87,361 |
| y > 20 h not LIRF (= LFPG row 16) | 1 (1) | 0.0922 | 0.1572 | 83,318 | 922 | 84,240 |

Per-row SSE of the 10 OOF rollover rows (share of total / of unmatched): LFPG EJU983W 6.94e9 (0.0922 / 0.1572); LIRF ITYTY680-Nov 3.04e9 (0.0404 / 0.0688); NOSOS610 2.86e9 (0.0380 / 0.0648); BAWW561D 2.82e9 (0.0374 / 0.0637); EXSXS1GT 2.73e9 (0.0362 / 0.0617); VOE6BB 2.31e9 (0.0307 / 0.0523); RYRR90BN 2.12e9 (0.0282 / 0.0480); CHHHH438 2.05e9 (0.0272 / 0.0464); ITYTY680-Jul 1.26e9 (0.0167 / 0.0285); AEZ146R (matched, pred 84,798) 4.86e6 (0.0001 / 0.0001).

Removing the 10 rows: total RMSE 334.58 → 270.34; unmatched RMSE 2,296.1 → 1,468.2.
Top-25 unmatched rows by SSE = 76.33 % of unmatched SSE; 9 of the 25 are rollover rows (y > 20 h), 15 have y ≥ 4 h; the other 10 are 6 LIRF sched-filled rows (y == ms 25,381…64,438, predicted 20,890…49,316) and 4 non-LIRF rows with normal y (549…1,265) and ms 45,569…151,639 predicted 14,715…16,880.

### 3.6 Other OOF files (same join; `y > 20 h` always 10 rows on the 672,428-row files)

| file | RMSE | unmatched RMSE | y > 20 h share of total | y ≥ 4 h (55 rows) share | y ≥ 2 h (210 rows) share |
|---|---|---|---|---|---|
| oof_main-48c19bc830 | 334.58 | 2,296.1 | 0.347 | 0.434 | 0.460 |
| **oof_main-f9d57a661f** | **300.17** | **1,864.9** | **0.241** | **0.336** | **0.367** |
| oof_ab_split-ca87cb5af7 | 301.63 | 1,875.3 | 0.243 | 0.339 | 0.370 |
| oof_ab_single-3978ee26d2 | 375.73 | 2,697.2 | 0.399 | 0.504 | 0.526 |
| oof_abl_vanilla_params-c2251eb080 | 300.69 | 1,850.9 | 0.237 | 0.337 | 0.371 |
| oof_repro_v2 / det_a / det_b | 336.2 | 2,299.0 | 0.342 | 0.429 | 0.456 |
| oof_main-63086f0648 | 340.40 | 2,364.1 | 0.367 | 0.450 | 0.474 |
| 328,009-row files (feat_*, ff1_*, weather_check) | 285.5–290.3 | 2,116–2,174 | 3 rows, 0.329–0.334 | 10 rows, 0.364–0.367 | 61 rows, 0.383–0.385 |

`oof_main-f9d57a661f` reproduces H §3 exactly (10 rows 24.1 %, 55 rows 33.6 %, 210 rows 36.7 %, model-B RMSE 1,864.9). Across all 672,428-row OOF files the 10 rows are 23.7 %–39.9 % of total SSE and 50.1 %–62.2 % of unmatched SSE.

### 3.7 Ranking (feat_rank, 344,841 rows)

`BLOCK_TIME_UTC_mvt` null on 344,841 / 344,841 rows; `TAXITIME_SEC_mvt` null on all; `ms` null on 0. Unmatched 5,290 (routing) / 5,283 (FLIGHT_ID).

| ms band | matched n (routing / FID) | unmatched n (routing / FID) | LIRF among unmatched | ms median (unmatched) |
|---|---|---|---|---|
| 12–20 h | 73 / 74 | 18 / 17 | 9 | 53,147 |
| 20–28 h | **36** / 36 | **6** / 6 | 1 | 83,764 |
| > 28 h | **6** / 6 | **1** / 1 | 1 | 111,654 |

LIRF unmatched ranking rows with ms ≥ 2 h / 4 h / 12 h: 181 / 43 / **11**. Non-LIRF unmatched ms ≥ 12 h: 14.
The 11 LIRF rows (all `AIRCRAFT_TYPE_mvt` null), with `outputs/main.parquet` prediction:

| FLIGHT | month | SCHED | MVT | ms | hour | RWY | main pred |
|---|---|---|---|---|---|---|---|
| WMTT3508 | 1 | 2026-01-02 19:45 | 2026-01-03 11:27:00 | 56,520 | 11 | 25 | 33,444 |
| TAY9LA | 1 | 2026-01-22 05:45 | 2026-01-22 20:37:00 | 53,520 | 20 | 25 | 30,644 |
| WMT402 | 1 | 2026-01-14 00:40 | 2026-01-14 14:14:01 | 48,841 | 14 | 25 | 36,782 |
| WMTMT4SG | 7 | 2026-07-04 17:50 | 2026-07-05 07:05:05 | 47,705 | 7 | 25 | 35,434 |
| NOSOS431 | 7 | 2026-07-10 14:50 | 2026-07-11 21:50:54 | 111,654 | 21 | 25 | 39,636 |
| ITYTY672 | 7 | 2026-07-09 20:10 | 2026-07-10 13:12:02 | 61,322 | 13 | 16R | 66,037 |
| RYR4852 | 7 | 2026-07-06 08:10 | 2026-07-06 21:38:59 | 48,539 | 21 | 25 | 33,485 |
| TARAR753 | 7 | 2026-07-05 14:20 | 2026-07-06 03:52:56 | 48,776 | 3 | 25 | 35,683 |
| SVII3006 | 7 | 2026-07-16 15:30 | 2026-07-17 08:20:56 | 60,656 | 8 | 25 | 31,551 |
| ITYTY674 | 7 | 2026-07-19 20:00 | 2026-07-20 09:08:57 | 47,337 | 9 | 16R | 66,693 |
| QFAA7006 | 7 | 2026-07-19 10:00 | 2026-07-20 10:18:04 | 87,484 | 10 | 16R | 32,688 |

### 3.8 Training LIRF unmatched rows with ms ≥ 12 h (n = 29): rollover vs sched-filled

Class by y > 20 h first, else |y − ms| ≤ 60: rollover 12 (y median 87,512, ms median 66,575), sched-filled 17 (y median 48,965 = ms), other 0. Row 1 (NOSOS739) is in both sets (y > 20 h **and** y == ms), so by "y == ms first" the split is 18 / 11 = 62.1 % / 37.9 %; under M's |y − ms| < 1,800 rule it is also 18 / 11 (row 4 NOSOS610 moves to sched-filled, row 1 stays). All 29 have `AIRCRAFT_TYPE_mvt` null and `FLIGHT_RULE_mvt = "I"`.

| ms bucket (h) | n | rollover (y > 20 h) | y == ms |
|---|---|---|---|
| 12 | 7 | 0 | 7 |
| 13 | 2 | 0 | 2 |
| 14 | 4 | 1 | 3 |
| 15 | 3 | 1 | 2 |
| 16 | 4 | 3 | 1 |
| 17 | 1 | 0 | 1 |
| 18 | 3 | 2 | 1 |
| 20 | 1 | 1 | 0 |
| 23 | 2 | 2 | 0 |
| 25 | 1 | 1 | 0 |
| 36 | 1 | 1 (row 1, also y == ms) | 0 |

Runway: rollover rows 25 × 10, 34L × 2; y == ms rows 25 × 11, 16R × 6. Take-off hour UTC of rollover rows: 3, 8, 10, 12, 12, 13, 14, 16, 17, 17, 18, 22; of y == ms rows: 0, 2, 3, 5, 6, 8, 8, 8, 8, 10, 10, 10, 11, 12, 18, 21, 23. BLOCK − SCHED: rollover rows −36,962…−1,503 and +6,365 (row 1: −4); y == ms rows −6…+6. FLIGHT ITYTY680 is a rollover twice (Jul, Nov); ITYTY672 / ITYTY674 are y == ms four times (all 16R).
Non-LIRF unmatched rows with ms ≥ 12 h: 46 (EDDF 18, EHAM 14, LFPG 8, EDDM 3, LEBL 2, LTFM 1), 0 rollover, 0 y == ms.

## 4. Verdicts per sub-claim

| sub-claim | verdict | deciding observation |
|---|---|---|
| 16 training rows y > 20 h (D) | CONFIRMED | §3.1: 16; 15 unmatched under both definitions + 1 matched. |
| y − 86,400 = 602…1,992 for 13 of 16 (D) | REFUTED on the count, CONFIRMED on the range | 14 of 16 fall in [602, 1,992]; the two others are −2,160 (LFPG) and 44,767 (row 1, y == ms). |
| "MVT and BLOCK differ by 1 day + 13–30 min" (M) | CONFIRMED (range 10–33 min) | 14 rows: 1 day + 602…1,992 s; date(MVT) − date(BLOCK) = 1 day on all 16. |
| Label = true taxi + 86,400 | CONFIRMED for 14 of 16 as an arithmetic description | 14 rows have y − 86,400 in (0, 7,200); row 1 is a sched-filled row (BLOCK = SCHED − 4 s, y = ms = 131,163); row 16 (LFPG) has BLOCK time-of-day 36 min after MVT time-of-day, so y − 86,400 < 0. |
| The day error sits in `BLOCK_TIME` (rather than MVT_TIME) | UNDETERMINED | No single-field one-day shift makes SCHED/BLOCK/MVT consistent for the 12 LIRF rows (BLOCK 1–10 h before SCHED **and** MVT 3–26 h after SCHED; §3.3 table). For the only row with an independent clock (row 15, matched), AOBT_3 19:57 agrees with BLOCK 19:59:59 and SCHED 20:00 on 07-14, which is consistent with MVT carrying the wrong date, not BLOCK. For rows 9 and 16 the BLOCK-date shift gives plausible values (−2 s, +3,900 s). Label value is identical under either attribution. |
| Alternative: BLOCK borrowed from previous day's same-callsign departure | REFUTED (within training DEP rows) | 0 of 16 rows have a same-(ADEP, FLIGHT) departure with MVT on the BLOCK date. |
| "A handful per year" / "15 rows over 12 months" (D) | CONFIRMED | 15 unmatched (16 total) rollover-band rows in 2025, spread over 9 of 12 months (Jul 5). |
| OOF 10 rows = 34.7 % of SSE, RMSE 51,121, pred median 37,131 (D) | CONFIRMED | §3.5 on main-48c19bc830: 10 rows, 0.3472 of total SSE, RMSE 51,121, pred median 37,131. |
| 8 rows = 52.7 % of unmatched SSE, RMSE 53,927 (M) | CONFIRMED, with the set identified | 9 unmatched y > 20 h rows minus NOSOS610 (|y − ms| = 1,503 < 1,800, classed sched-filled by M's rule): 8 rows, 0.5267 of unmatched SSE, RMSE 53,927. |
| "≈ 25 % of total" for those 8 rows (M) | REFUTED as stated | The 8 rows are 30.9 % of total SSE; 25.5 % is the share of the 9 rows with y ≥ 86,400 (or of the 9 LIRF rows y > 20 h). |
| Top-25 unmatched rows = 76.3 % of unmatched SSE (M) | CONFIRMED | 0.7633; 9 of the 25 are rollover rows. |
| 10 rows y ≈ 1 day = 24.1 % of OOF SSE (H) | CONFIRMED on a different OOF | Reproduced exactly on `oof_main-f9d57a661f` (RMSE 300.17, unmatched 1,864.9); on main-48c19bc830 the same 10 rows are 34.7 %. Discrepancy §4-4 is a denominator/OOF-file difference, not a counting error. |
| Training ms bands: 20–28 h unmatched 23 (17.4 % y ≥ 2 h), > 28 h unmatched 3 (D) | CONFIRMED | §3.4; identical under both unmatched definitions. |
| Ranking 6 + 1 unmatched, 36 + 6 matched in those bands (D) | CONFIRMED | §3.7; identical under both unmatched definitions (12–20 h band differs by 1 row: 18/73 vs 17/74). |
| Ranking 11 LIRF unmatched rows mms ≥ 12 h; training 62 % sched-filled / 38 % rollover (M) | CONFIRMED | 11 rows (§3.7); training 18 / 11 of 29 = 62.1 % / 37.9 % with y == ms taking precedence (row 1 is both). |
| "Predictable only as a mixture component" / no visible feature separates rollover from sched-filled | UNDETERMINED (n = 29) | ms bucket gives a gradient (ms 12–13 h: 0 of 9 rollover; 14–18 h: 7 of 15; ≥ 20 h: 5 of 5, one of which is also y == ms), runway 16R 0 of 6 vs 34L 2 of 2, take-off hours overlap; BLOCK − SCHED would separate them (|B−S| ≤ 6 vs ≥ 1,503) but BLOCK is null on every ranking row. |
| These rows "dominate SSE" | CONFIRMED under every definition | 24.1 %–39.9 % of total SSE across the 672,428-row OOF files; 50.1 %–62.2 % of unmatched SSE; removing the 10 rows moves main-48c19bc830 from 334.58 to 270.34. |

## 5. What cannot be determined locally

- The number of rollover rows in the ranking set: `BLOCK_TIME_UTC_mvt` and the label are null on all 344,841 ranking rows; only `ms` is observable. The 11 LIRF unmatched rows with ms ≥ 12 h and the 6 + 1 unmatched / 36 + 6 matched rows in the 20–28 h / > 28 h ms bands are the candidates by ms alone.
- Which timestamp carries the wrong date for the 15 unmatched rollover rows (no independent off-block clock; AOBT_3 is null on them).
- Whether the ranking LIRF rows with ms ≥ 12 h are sched-filled (y = ms) or rollover (y ≈ 87k) at the training proportions.
- The LB effect of any treatment of these rows.

## 6. Facts vs inference

**Facts (measured above):** all counts, lists and shares in §3; the identity `y = MVT − BLOCK` holds on every row; 14 of 16 rows y > 20 h have y − 86,400 ∈ [602, 1,992]; date(MVT) − date(BLOCK) = 1 day on all 16; BLOCK is 1–10 h before SCHED on 12 LIRF rows; AOBT_3 agrees with BLOCK and SCHED on the single matched rollover row; M's 8-row set equals the 9 unmatched y > 20 h rows minus NOSOS610; H's 24.1 % corresponds to `oof_main-f9d57a661f`; BLOCK is null on all ranking rows; 0 same-callsign departures on the BLOCK date.

**Inference (not established by the data here):** that the corrupted field is `BLOCK_TIME` (the evidence is compatible with a wrong MVT date for row 15 and is indeterminate for the 12 LIRF rows whose three timestamps are mutually inconsistent under any single one-day shift); that "true taxi" for these rows is y − 86,400 (arithmetically plausible, 10–33 min, but the true taxi time is not observed); that the ranking set contains rollover rows at the training density (≈ 15 per 12 months of unmatched rows would imply ≈ 2–3 in Jan + Jul 2026; not observable); that a mixture model is the only way to predict them; that M's 62 % / 38 % training split transfers to the 11 ranking rows.
