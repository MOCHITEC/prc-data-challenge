# 16. `seasonal-deicing-shift` — results (2026-09-16)

Source item: plan.md §1 #16 (merged from D `winter-2026-deicing-shift` and E H11). Script:
`docs/investigation_2026-09-16/scripts/seasonal-deicing-shift.py` (run with
`POLARS_MAX_THREADS=1 .venv/bin/python …`; lazy scans; ≤ 12 columns per scan; no model training).
Code read at HEAD `1d646dd`: `src/prc/weather.py` (`hourly_weather_table`, `join_weather`, `WX_KEY`),
`src/prc/pipeline.py:41–99`, `src/prc/compare.py`, `configs/main.yaml:15`, `experiments/experiments.jsonl`.
Builds on `results/fold-shape-vs-ranking.md` for the fold provenance of the OOF months (Jan and Jul 2025 in
`oof_main-48c19bc830` are the leaky `jan` / `jul` folds; Nov and Dec are the `forward` fold).

Definitions: `deicing_gate` = `(temp_c ≤ 3) ∧ (precip_mm > 0 ∨ frozen_wx)` per station-hour
(`weather.py:hourly_weather_table`), joined onto departures by `(ADEP_mvt, truncate(MVT_TIME_UTC_mvt, 1h))`.
"Gate hour share" = mean of `deicing_gate` over the hours of a station-month (D's quantity). "Gate row share" =
share of departure rows whose take-off hour has `deicing_gate == 1`. Residual = `oof − y`. `unmatched` =
`mvt_minus_aobt3` null (§0 routing definition). RMSE figures are in seconds.

## 1. Hypothesis (as consolidated in plan.md)

Jan 2026 weather (2–4× more de-icing-gate hours than Jan 2025) and Jan/Jul seasonality (elevated label mean/SD)
place the ranking months outside what the 2025-trained model saw. Sub-claims carried in the plan:

- (a) Gate hour share Jan 2026 / Jan 2025: EHAM 0.128/0.051, LSZH 0.180/0.059, EDDM 0.112/0.069, LTFM 0.095/0.000,
  EDDF 0.083/0.023, LFPG 0.015/0.031, EGLL 0.001/0.011; low-visibility lower in 2026 at EHAM (0.031/0.109) and
  LFPG (0.027/0.073) (D).
- (b) Weather cache complete; weather-feature null parity train vs rank ≤ 0.02 pp (D).
- (c) Jan/Jul 2025 label mean 995/1026 s, SD 605/745 "elevated vs mid-year"; LIRF SD 1,332 s (E A3, external repo).
- (d) Unverified in the plan: per-month OOF RMSE; residual by airport Jan vs rest; the weather effect re-measured
  with the fixed join (v3/v4 were trained with the `hour_utc`-overwriting join bug).
- (e) The inference itself: the ranking months lie outside the training distribution in a way that matters.

## 2. Procedure

1. **A** `data/ref/metar_hourly.parquet` (138,446 station-hours, 2025-01-01 00:00 – 2026-07-31 23:00, unique on
   `(station, hour_utc)`): re-derived `deicing_gate` from `temp_c`, `precip_mm`, `frozen_wx` and compared with the
   stored column; hours per station-month vs calendar hours (coverage); gate / frozen / precip / temp ≤ 3 /
   low-vis shares per station × month for all 19 months; Jan 2026 vs the maximum of any 2025 month.
2. **B** `experiments/oof_main-48c19bc830.parquet` inner-joined to `feat_train.parquet` on `MVT_ID_mvt`, rows with
   non-null `oof` (672,428): RMSE, n, mean / median residual, median |residual|, share of SSE, by
   `deicing_gate` (× ADEP, × matched/unmatched, with and without y ≤ 7,200), by month (× ADEP, × unmatched), Jan vs
   Jul/Nov/Dec per ADEP (matched), and gate vs no-gate inside Jan 2025 per ADEP.
3. **C** Label mean / SD / median / p90 / count ≥ 2 h by month, by airport, and by airport × month on all
   2,085,047 training rows; repeated on matched rows and on y < 7,200.
4. **D** `feat_rank.parquet`: rows with `deicing_gate == 1` by ADEP × month (and how many are unmatched); the same
   table for Jan / Jul 2025 training rows; row-level `deicing_gate` and `temp_c` re-joined from the METAR table on
   the truncated take-off hour and compared with the cached columns in both frames; dtype and range of the
   `hour_utc` feature.
5. **E** Null rate of all 20 weather columns (10 current-hour + 10 `_prev1h`) in `feat_train` vs `feat_rank`.
6. **F** Training label by `deicing_gate` (matched / unmatched; per airport; Jan 2025 only).
7. **G** `precip_mm` over the whole METAR table; paired comparison of `oof_det_a-cfc9417c52` (sha 70ef994,
   72 features, no weather) and `oof_weather_check-1776e5e01c` (sha 34808da = the `WX_KEY` fix commit, 90 features)
   on the 328,009 forward-fold rows both scored, by `deicing_gate`, matched/unmatched, month, ADEP.

## 3. Results

### 3.1 METAR gate hour share (A)

- Stored `deicing_gate` equals the re-derived expression on all 138,446 hours (0 mismatches; 3 null `temp_c`).
- Coverage: every station-month ≥ 0.9958 of calendar hours (min: LIRF 13,831 / 13,896 h); no station-month below
  0.99. 277 training rows (0.013 %) and 0 ranking rows have a take-off hour absent from the table.
- `precip_mm > 0` occurs in **0 of 138,446 hours** (max 0.0, 0 nulls, all 10 stations). Every one of the
  1,326 gate hours is `frozen_wx ∧ temp_c ≤ 3`; `gate_without_frozen` = 0. The precipitation term of the gate
  definition is inert on this cache (1,413 frozen hours; 87 of them at temp > 3 °C).

Gate hour share, station × month (all 19 months; 0.0 omitted where a whole row is zero):

| station | 25-01 | 25-02 | 25-03 | 25-10 | 25-11 | 25-12 | **26-01** | 26-02 | 26-03 | 26-04 | Apr–Sep 25, May–Jul 26 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EDDF | 0.0228 | 0.0074 | 0 | 0 | 0.0139 | 0.0067 | **0.0833** | 0.0357 | 0 | 0 | 0 |
| EDDM | 0.0685 | 0.0357 | 0.0148 | 0.0067 | 0.0542 | 0.1075 | **0.1116** | 0.0744 | 0.0403 | 0 | 0 |
| EGLL | 0.0108 | 0 | 0 | 0 | 0.0014 | 0.0013 | **0.0013** | 0 | 0 | 0 | 0 |
| EHAM | 0.0511 | 0.0253 | 0.0269 | 0 | 0.0167 | 0.0027 | **0.1277** | 0.0134 | 0.0013 | 0 | 0 |
| LEBL | 0 | 0 | 0 | 0 | 0 | 0 | **0** | 0 | 0 | 0 | 0 |
| LEMD | 0.0013 | 0 | 0.0013 | 0 | 0 | 0.0027 | **0.0040** | 0 | 0 | 0 | 0 |
| LFPG | 0.0309 | 0.0223 | 0 | 0 | 0.0125 | 0 | **0.0148** | 0.0030 | 0 | 0 | 0 |
| LIRF | 0 | 0 | 0 | 0 | 0 | 0 | **0** | 0 | 0 | 0 | 0 |
| LSZH | 0.0591 | 0.0312 | 0 | 0 | 0.0389 | 0.0336 | **0.1801** | 0.0640 | 0.0363 | 0.0083 | 0 |
| LTFM | 0 | 0.1756 | 0.0054 | 0 | 0 | 0.0027 | **0.0954** | 0.0521 | 0 | 0 | 0 |

Jan 2026 vs Jan 2025 (gate hours in parentheses): EDDF 0.0833 (62) / 0.0228 (17) ×3.65; EDDM 0.1116 (83) /
0.0685 (51) ×1.63; EGLL 0.0013 (1) / 0.0108 (8) ×0.12; EHAM 0.1277 (95) / 0.0511 (38) ×2.50; LEMD 0.0040 (3) /
0.0013 (1); LFPG 0.0148 (11) / 0.0309 (23) ×0.48; LSZH 0.1801 (134) / 0.0591 (44) ×3.05; LTFM 0.0954 (71) / 0 (0);
LEBL and LIRF 0 / 0. Jul 2025, Jul 2026: 0 at every station. Low-vis Jan 2026 / Jan 2025: EHAM 0.0309 / 0.1089,
LFPG 0.0269 / 0.0726 (also LSZH 0.1277 / 0.0753, EDDM 0.0591 / 0.0417). Mean temperature Jan 2026 vs Jan 2025:
EDDF 1.0 / 3.3, EDDM −1.7 / 0.8, EHAM 2.7 / 3.8, LSZH −0.5 / 2.0, LTFM 7.8 / 8.7, EGLL 5.6 / 4.4, LFPG 5.2 / 4.1.

Jan 2026 exceeds the maximum of all twelve 2025 months at EDDF (max 0.0228, Jan), EHAM (0.0511, Jan), LSZH
(0.0591, Jan), LEMD (0.0027) and marginally EDDM (0.1116 vs 0.1075 in Dec 2025). It does **not** at LTFM (Feb 2025
= 0.1756), EGLL, LFPG, LEBL, LIRF.

### 3.2 OOF residuals, `oof_main-48c19bc830` (B)

Join: 672,428 rows, `MVT_ID_mvt` unique; overall RMSE 334.582, mean residual +4.478 (record: 334.582 / bias 4.478).
94 joined rows have null `deicing_gate` (no METAR hour).

By gate (all rows / matched / unmatched):

| subset | gate | n | RMSE | mean res | median res | median abs res | % SSE | y mean | pred mean |
|---|---|---|---|---|---|---|---|---|---|
| all | 0 | 667,204 | 334.3 | +4.5 | 14.8 | 99.8 | 99.03 | 1,002 | 1,006 |
| all | 1 | 5,130 | 376.7 | +7.8 | 22.3 | 141.1 | 0.97 | 1,361 | 1,369 |
| matched | 0 | 658,938 | 215.2 | +4.4 | 14.5 | 99.1 | 40.52 | 996 | 1,001 |
| matched | 1 | 5,018 | 345.7 | +8.2 | 22.1 | 137.7 | 0.80 | 1,350 | 1,359 |
| unmatched | 0 | 8,266 | 2,308.2 | +12.4 | 62.3 | 196.5 | 58.50 | 1,430 | 1,443 |
| unmatched | 1 | 112 | 1,070.7 | −12.0 | 95.3 | 549.0 | 0.17 | 1,863 | 1,851 |

`deicing_gate_prev1h == 1` (5,462 rows): RMSE 487.5, mean res −4.1, 1.72 % SSE.

By gate × ADEP, matched rows (gate = 0 → gate = 1); LEBL / LIRF have no gate rows, LEMD 8:

| ADEP | n₁ | RMSE₀ → RMSE₁ | mean res₀ → res₁ | median res₁ | y mean₀ → y mean₁ |
|---|---|---|---|---|---|
| EDDF | 523 | 177.9 → 260.1 | +2.2 → −6.8 | 15.0 | 881 → 1,080 |
| EDDM | 1,856 | 171.3 → 309.6 | −1.3 → −9.6 | 10.4 | 839 → 1,365 |
| EGLL | 249 | 229.4 → 382.3 | +10.6 → +121.8 | 96.5 | 1,371 → 1,450 |
| EHAM | 792 | 161.5 → 214.1 | −2.4 → +35.6 | 19.3 | 787 → 1,024 |
| LFPG | 544 | 241.2 → 626.9 | +3.7 → −25.1 | 24.4 | 1,045 → 2,303 |
| LSZH | 965 | 165.2 → 284.7 | −1.0 → +0.9 | 30.6 | 750 → 1,151 |
| LTFM | 81 | 233.9 → 517.2 | +5.2 → +201.6 | 130.0 | 1,052 → 1,634 |

Removing y > 7,200 changes these matched figures by ≤ 0.1 % of n and ≤ 41 s of RMSE (LFPG 626.9 → 586.4); the
gate-row RMSE inflation is not a tail artefact. Unmatched gate rows are 112 in total (max 29 per airport).

By month (fold provenance from `fold-shape-vs-ranking.md`: Jan = `jan` fold, Jul = `jul` fold, both trained on
the 11 other months; Nov/Dec = `forward` fold):

| month | n | RMSE all | mean res | % SSE | matched n / RMSE / mean res | unmatched n / RMSE / mean res |
|---|---|---|---|---|---|---|
| 2025-01 | 153,706 | 373.0 | +5.9 | 28.40 | 152,271 / 206.2 / +6.4 | 1,435 / 3,223.0 / −47.9 |
| 2025-07 | 190,713 | 378.4 | +4.1 | 36.28 | 186,775 / 247.3 / +3.1 | 3,938 / 2,008.8 / +55.8 |
| 2025-11 | 162,332 | 291.5 | −0.9 | 18.32 | 160,976 / 198.3 / −0.4 | 1,356 / 2,345.5 / −50.1 |
| 2025-12 | 165,677 | 277.9 | +8.7 | 17.00 | 164,028 / 204.9 / +8.7 | 1,649 / 1,893.1 / +11.1 |

Per-month figures match the record's fold RMSEs (372.97 / 378.42; forward pooled 284.68). Median |residual| by
month: 96.0 / 107.5 / 97.0 / 98.8.

By month × ADEP, matched rows, RMSE (mean res in parentheses):

| ADEP | Jan 25 | Jul 25 | Nov 25 | Dec 25 | Jan vs Jul+Nov+Dec (RMSE / mean res) |
|---|---|---|---|---|---|
| EDDF | 173.8 (−0.7) | 189.2 (+14.1) | 178.4 (−10.5) | 169.5 (+3.2) | 173.8 vs 179.9 / −0.7 vs +2.8 |
| EDDM | 170.2 (+11.2) | 170.1 (−5.8) | 163.0 (−12.6) | 208.5 (+3.5) | 170.2 vs 180.3 / +11.2 vs −5.2 |
| EGLL | 233.0 (+19.7) | 254.6 (+8.7) | 204.3 (+1.1) | 224.0 (+14.3) | 233.0 vs 229.1 / +19.7 vs +8.1 |
| EHAM | 174.4 (+2.2) | 166.6 (+9.4) | 154.2 (−16.7) | 152.0 (−4.4) | 174.4 vs 158.2 / +2.2 vs −3.3 |
| LEBL | 144.1 (+1.1) | 262.5 (+0.3) | 159.2 (+8.9) | 177.4 (+10.0) | 144.1 vs 209.0 / +1.1 vs +6.0 |
| LEMD | 159.7 (+4.1) | 190.2 (+17.2) | 170.2 (+20.7) | 165.9 (+15.5) | 159.7 vs 176.0 / +4.1 vs +17.8 |
| LFPG | 258.2 (+14.1) | 249.2 (+13.5) | 247.9 (−16.1) | 229.1 (+1.2) | 258.2 vs 242.3 / +14.1 vs +0.2 |
| LIRF | 284.7 (+2.2) | 442.4 (+4.5) | 294.8 (+6.2) | 309.3 (−2.3) | 284.7 vs 363.5 / +2.2 vs +2.9 |
| LSZH | 173.7 (+0.8) | 182.2 (+5.5) | 157.1 (−7.7) | 158.8 (−3.8) | 173.7 vs 167.5 / +0.8 vs −1.5 |
| LTFM | 221.2 (+4.0) | 274.2 (−30.6) | 211.0 (+18.2) | 219.5 (+34.1) | 221.2 vs 238.2 / +4.0 vs +5.8 |

All rows (matched + unmatched) Jan 2025: LFPG 800.8 (15.25 % of total SSE, LFPG unmatched 1,539 rows carry
15.43 %), LIRF 646.9 (6.29 %); Jul 2025: LIRF 1,040.4 (21.87 %). Within Jan 2025, matched rows, RMSE gate = 1 vs
gate = 0: EDDF 209.7 / 173.3, EDDM 287.6 / 160.8, EGLL 399.9 / 230.7 (mean res +176.9 / +18.2), EHAM 227.6 / 172.4
(+48.2 / +0.7), LFPG 569.9 / 245.8, LSZH 341.5 / 160.8; gate row share in Jan 2025 OOF: EDDM 5.5 %, LSZH 4.7 %,
EHAM 3.1 %, LFPG 2.4 %, EDDF 1.3 %, EGLL 1.0 %, others 0.

### 3.3 Label mean / SD by month and airport, 2025 (C)

All 2,085,047 rows (rank among 12 months, 1 = highest):

| month | n | mean (rank) | SD (rank) | median | p90 | n ≥ 2 h | matched-only mean / SD | y < 7,200 mean / SD |
|---|---|---|---|---|---|---|---|---|
| 01 | 153,706 | **995** (4.5) | **605** (3) | 906 | 1,495 | 29 | 992 / 442 | 992 / 442 |
| 02 | 143,732 | 1,002 (2) | 628 (2) | 904 | 1,494 | 46 | 994 / 466 | 998 / 491 |
| 03 | 164,449 | 972 (11) | 462 (10) | 905 | 1,442 | 31 | 970 / 384 | 970 / 386 |
| 04 | 175,288 | 971 (12) | 436 (12) | 908 | 1,431 | 27 | 970 / 370 | 969 / 373 |
| 05 | 185,202 | 987 (9) | 469 (9) | 925 | 1,447 | 39 | 984 / 389 | 984 / 390 |
| 06 | 183,142 | 973 (10) | 555 (5) | 902 | 1,443 | 64 | 969 / 407 | 969 / 411 |
| 07 | 190,713 | **1,026** (1) | **745** (1) | 948 | 1,512 | 120 | 1,016 / 502 | 1,016 / 459 |
| 08 | 191,182 | 994 (6) | 534 (6) | 929 | 1,461 | 74 | 989 / 406 | 989 / 409 |
| 09 | 183,950 | 990 (8) | 560 (4) | 914 | 1,480 | 50 | 985 / 405 | 985 / 408 |
| 10 | 185,674 | 992 (7) | 437 (11) | 925 | 1,479 | 43 | 990 / 409 | 990 / 408 |
| 11 | 162,332 | 997 (3) | 532 (7) | 927 | 1,495 | 23 | 994 / 408 | 995 / 413 |
| 12 | 165,677 | 995 (4.5) | 515 (8) | 911 | 1,493 | 38 | 991 / 409 | 992 / 414 |

By airport, all rows: LIRF mean 1,195, **SD 1,332**, 480 rows ≥ 2 h; next-highest SD LFPG 453, LTFM 428, EGLL 422;
LIRF matched-only SD by month 421–1,021 (Jul), all-rows 775–2,167 (Jul).
Airport × month means (all rows): Jan is the highest month only at EDDF (885; Nov 895 is higher), EDDM (870; Dec 947
higher), LFPG (1,080; Nov 1,106 higher), LSZH (775, highest); Jul is highest at EGLL (1,431), LEBL (1,008), LIRF
(1,406), LTFM second (1,094; Feb 1,261). SD Jan is highest of the year at EDDF (399), EHAM (440), LSZH (413 all /
400 matched), EGLL second (462 vs Jul 527), LFPG all-rows 937 (matched 552, Nov 470).

### 3.4 Ranking rows in gate hours; join check (D)

| ADEP | Jan 2025 rows / gate = 1 (share) [unmatched] | Jan 2026 rows / gate = 1 (share) [unmatched] | ratio of shares | training full-year matched gate rows |
|---|---|---|---|---|
| EDDF | 15,496 / 206 (0.0133) [0] | 15,667 / 1,223 (0.0781) [55] | 5.87 | 617 |
| EDDM | 11,266 / 628 (0.0557) [12] | 10,953 / 1,028 (0.0939) [14] | 1.69 | 2,351 |
| EGLL | 19,420 / 192 (0.0099) [5] | 19,283 / 0 (0) [0] | 0 | 249 |
| EHAM | 18,460 / 581 (0.0315) [16] | 16,232 / 1,037 (0.0639) [263] | 2.03 | 1,068 |
| LEBL | 12,234 / 0 | 12,201 / 0 | — | 0 |
| LEMD | 16,510 / 3 | 16,802 / 6 | 2.0 | 8 |
| LFPG | 17,900 / 432 (0.0241) [12] | 18,129 / 213 (0.0117) [21] | 0.49 | 791 |
| LIRF | 11,317 / 0 | 11,061 / 0 | — | 0 |
| LSZH | 9,735 / 463 (0.0476) [10] | 9,995 / 1,379 (0.1380) [75] | 2.90 | 1,119 |
| LTFM | 21,368 / 0 (0) [0] | 22,396 / 1,843 (0.0823) [34] | ∞ | 2,637 |
| **total** | 153,706 / 2,505 (0.0163) [55] | 152,719 / 6,729 (0.0441) [462] | 2.70 | 8,840 (+317 unmatched) |

Jul 2025 and Jul 2026: 0 gate rows at every airport. Jan 2026 alone holds 6,729 gate rows against 9,157 in the
entire 2025 training year; at EDDF (1,223 vs 617) and LSZH (1,379 vs 1,119) one ranking month has more gate rows
than the whole training set. Low-vis rows Jan 2026 / Jan 2025: 3,141 / 4,396 (EHAM 396 / 1,901, LFPG 494 / 1,158,
LSZH 1,028 / 601, EDDM 592 / 354).

Join check: re-joining the METAR table on `(ADEP, truncate(MVT_TIME, 1h))` reproduces the cached `deicing_gate`
and `temp_c` on every row of both frames (0 mismatches; 277 training rows with no METAR hour, 0 ranking rows).
`hour_utc` in `feat_train` is `Int8` with range 0–23, i.e. the cached features carry the hour-of-day feature and
were built with the `WX_KEY` join, not the v3/v4-era one. `configs/main.yaml:15` sets `use_weather: true` (same at
`main-48c19bc830`'s sha 97e67c6); the record has `n_features: 99`.

### 3.5 Weather-feature null parity (E)

18 of 20 columns: train 0.013–0.020 % null, rank 0–0.014 %, |difference| ≤ 0.020 pp. `gust_ms` 97.705 % vs
96.620 % (−1.085 pp); `gust_ms_prev1h` 97.721 % vs 96.670 % (−1.051 pp). Max |diff| over all 20 = 1.085 pp.

### 3.6 Training label by gate (F)

Matched rows: gate = 0 n 2,053,463 mean 985 / median 911 / SD 412; gate = 1 n 8,840 mean 1,530 / median 1,222 /
SD 989 (5 rows ≥ 2 h). Unmatched: gate = 0 n 22,150 mean 1,350 / median 946; gate = 1 n 317 mean 3,733 / median
4,028 / p90 6,590 (18 rows ≥ 2 h). Matched by airport, mean gate 0 → 1: EDDF 862 → 1,076, EDDM 802 → 1,375,
EGLL 1,363 → 1,450, EHAM 786 → 1,111, LEMD 1,014 → 1,236 (n 8), LFPG 1,013 → 2,244, LSZH 738 → 1,157, LTFM
1,041 → 1,896 (n 2,637, all from Feb 2025 and Dec 2025 — Jan 2025 had 0). Jan 2025 only, matched: EDDF 883 → 914,
EDDM 840 → 1,356, EGLL 1,329 → 1,325, EHAM 801 → 1,092, LFPG 1,044 → 2,114, LSZH 755 → 1,259.

### 3.7 Weather on/off with the fixed join (G)

`weather_check-1776e5e01c` (sha 34808da, the commit that introduced `WX_KEY`; forward fold only) vs
`det_a-cfc9417c52` (sha 70ef994, no weather). Commits between them: c5e9908 (record), 9ab8758 (perf), 8dc2152
(`force_row_wise`), 34808da (weather fix) — not a pure weather diff. Record-level forward-fold RMSE 288.880 → 285.746
(−3.13; `docs/README.md:89` reports −3.13 [−5.23, −1.26] and −1.06 [−1.40, −0.73] after the worst 200 rows are
removed, `prc.compare` method; the 200-row curve was not re-run here). Paired on the 328,009 Nov/Dec 2025 rows:

| subset | n | RMSE no-wx → wx | Δ | mean res no-wx → wx |
|---|---|---|---|---|
| matched | 325,004 | 203.35 → 202.48 | −0.87 | −1.9 → +4.4 |
| unmatched | 3,005 | 2,153.33 → 2,116.25 | −37.08 | −51.8 → −18.0 |
| matched, gate 0 | 322,369 | 201.82 → 201.05 | −0.77 | −1.4 → +4.5 |
| matched, gate 1 | 2,568 | 344.99 → 336.99 | −8.00 | −61.0 → −8.7 |
| unmatched, gate 1 | 57 | 1,078.22 → 1,007.18 | −71.04 | −346.8 → −138.6 |

Matched gate rows by airport, Δ RMSE (mean res no-wx → wx): EDDF n 317 −22.86 (−67.4 → −16.2); EDDM 1,240 −7.79
(−75.2 → −18.2); EGLL 62 −33.22; EHAM 227 +6.71 (−22.8 → +6.6); LFPG 124 −19.40 (−214.1 → −89.1); LSZH 512 −1.56;
LTFM 81 +18.96 (+29.0 → +207.8); LEMD 5 +10.04. Matched gate-0 rows by airport: Δ between −4.16 (EHAM) and +1.72
(LTFM). Nov/Dec matched: −1.14 / −0.61. Without weather the model under-predicts gate rows (mean res −61.0);
with weather the under-prediction is largely removed (−8.7) except at LTFM, which had 0 Jan 2025 gate rows and
whose 81 Nov/Dec gate rows are over-predicted by +208 s.

## 4. Verdict per sub-claim

| sub-claim | verdict | basis |
|---|---|---|
| (a) Gate hour shares Jan 2026 / Jan 2025 for the 7 airports and the two low-vis figures | **CONFIRMED** — all 16 numbers reproduced to the quoted precision (§3.1) | independent recomputation from `metar_hourly.parquet`; stored gate ≡ definition |
| (a′) "2–4× more de-icing-gate hours" | **CONFIRMED** for EHAM ×2.50, LSZH ×3.05, EDDF ×3.65 (LTFM 0 → 71 h); **REFUTED** as a general statement: EDDM ×1.63, LFPG ×0.48, EGLL ×0.12, LEBL/LIRF 0/0; Jul 2026 = 0 everywhere | §3.1 |
| (a″) Jan 2026 gate share is outside the 2025 range the model saw | **CONFIRMED** at EDDF, EHAM, LSZH (and marginally EDDM); **REFUTED** at LTFM (Feb 2025 0.1756 > Jan 2026 0.0954) and everywhere else | §3.1 last table |
| (b) Weather cache complete | **CONFIRMED** (coverage ≥ 0.9958 every station-month; 0.013 % training rows unmatched to an hour) | §3.1 |
| (b′) Null parity ≤ 0.02 pp | **CONFIRMED** for 18 of 20 weather columns; **REFUTED** for `gust_ms` / `gust_ms_prev1h` (1.085 / 1.051 pp; both ~97 % null) | §3.5 |
| (c) Jan/Jul 2025 mean 995/1026, SD 605/745; LIRF SD 1,332 | **CONFIRMED** — exact match on all five numbers | §3.3 |
| (c′) "elevated vs mid-year" | **CONFIRMED** for Jul (rank 1 on mean and SD, in every variant); for Jan **CONFIRMED** on SD (rank 3; but Feb 628 > Jan 605) and **REFUTED** on mean (995 = Dec 995, < Nov 997, < Feb 1,002; rank 4.5/12). The Jan/Jul SD excess over mid-year is a tail effect: on y < 7,200 the SDs are Jan 442 / Jul 459 vs 373–414 for the other months, and matched-only Jan 442 / Jul 502 | §3.3 |
| (d₁) Per-month OOF RMSE | measured: 373.0 / 378.4 / 291.5 / 277.9 (Jan / Jul / Nov / Dec); matched-only 206.2 / 247.3 / 198.3 / 204.9. Jan and Jul are the leaky folds, so the comparison with Nov/Dec is not a horizon comparison | §3.2 |
| (d₂) Residual by airport, Jan vs rest | measured: matched Jan RMSE higher than Jul+Nov+Dec at EHAM (+16), LFPG (+16), LSZH (+6), EGLL (+4); lower at the other six. Jan mean residual within ±20 s at every airport (largest EGLL +19.7, LFPG +14.1, EDDM +11.2) | §3.2 |
| (d₃) Weather effect with the fixed join | one existing measurement (`weather_check` at the fix commit, forward fold): −3.13 s record-level; paired −0.87 matched / −37.1 unmatched; gate rows −8.0 (2,568 rows) with mean residual −61 → −9 | §3.7 |
| gate rows are harder for the model | **CONFIRMED** — matched gate rows RMSE 345.7 vs 215.2; inflation ×1.3–2.6 at every airport with gate rows; median |res| 138 vs 99; not driven by y > 7,200 | §3.2 |
| model systematically mis-predicts gate rows | **REFUTED** for the pooled figure (mean res +8.2, median +22); **CONFIRMED** locally at EGLL (+122, n 249) and LTFM (+202, n 81), both airports with few training gate rows | §3.2 |
| ranking Jan 2026 has more gate rows than Jan 2025 | **CONFIRMED** — 6,729 vs 2,505 (share ×2.70); 462 vs 55 of them unmatched (EHAM 263) | §3.4 |
| the cached features used the fixed join | **CONFIRMED** — `hour_utc` Int8 0–23 present; row-level gate/temp identical to a fresh re-join on both frames | §3.4 |
| (e) the shift "places the ranking months outside what the model saw" in a way that changes the score | **UNDETERMINED** — see §5 | |

## 5. Not determinable locally

- The effect of the Jan 2026 gate-hour excess on the LB needs ranking labels. What is measurable is only the
  exposure: 6,729 Jan 2026 rows (4.4 % of that month, 1.95 % of the ranking set) are in gate hours, vs 2,505 in
  Jan 2025; gate rows carry 0.97 % of OOF SSE at their 2025 frequency (0.76 % of OOF rows) with RMSE 376.7.
- Whether the model extrapolates correctly at LTFM (1,843 Jan 2026 gate rows; training gate rows exist only in
  Feb/Dec 2025; Nov/Dec forward-fold gate rows over-predicted by +208 s, n 81) and EDDF (1,223 vs 617 in all of
  training) cannot be tested without labels.
- The July half of the ranking set (55.7 % of rows) has zero gate hours in both years; nothing in this item
  bears on it beyond the Jul 2025 label statistics.
- The clean "weather effect with the fixed join" needs a controlled pair (same sha, weather on/off); the only
  existing pair spans four commits including `force_row_wise`. The `prc.compare` 200-row curve quoted in
  `docs/README.md` was not re-run here.
- The gate is currently `frozen_wx ∧ temp ≤ 3` because `precip_mm` is identically 0 in the cache; whether the IEM
  `p01i` field is genuinely empty for these stations or lost in `parse_metar_csv` (it maps `T` → 0.0001 and
  `M` → null) needs the raw CSVs under `data/ref/metar/`, which were not read here.

## 6. Facts vs inference

**Facts (measured here or read from code/records at HEAD 1d646dd):**
- METAR cache: 138,446 station-hours, complete to ≥ 99.58 % per station-month, Jan 2025 – Jul 2026; stored
  `deicing_gate` equals its definition on every hour; `precip_mm > 0` on 0 hours; all 1,326 gate hours are
  frozen-code hours at ≤ 3 °C.
- Gate hour shares per station × month as tabulated in §3.1; D's 16 quoted figures reproduced exactly.
- Jan 2026 gate share exceeds every 2025 month at EDDF, EHAM, LSZH, LEMD, EDDM; not at LTFM, EGLL, LFPG, LEBL, LIRF.
- OOF `main-48c19bc830`: 672,428 rows, RMSE 334.582; by gate, month, airport as in §3.2; per-month RMSE equals the
  record's fold RMSEs; Jan/Jul rows are from the leaky folds (sibling report).
- Training label statistics per month / airport as in §3.3; E's 995 / 1,026 / 605 / 745 / 1,332 are exact.
- Ranking gate rows: Jan 2026 6,729 (462 unmatched), Jul 2026 0; per airport as in §3.4; training Jan 2025 2,505,
  Jul 2025 0; training full year 9,157.
- Cached `deicing_gate` / `temp_c` in both frames equal a fresh hour-join; `hour_utc` Int8 0–23 present;
  `configs/main.yaml` `use_weather: true` at HEAD and at 97e67c6.
- Null parity: 18 columns ≤ 0.020 pp; `gust_ms` −1.085 pp, `gust_ms_prev1h` −1.051 pp.
- Training label by gate (§3.6); paired det_a / weather_check differences on Nov/Dec 2025 rows (§3.7);
  `weather_check` sha 34808da is the `WX_KEY` commit; four commits separate it from det_a.

**Inference (not measured):**
- That the gate-row RMSE inflation (×1.3–2.6) is "difficulty of the weather regime" rather than of the hours
  those rows fall in follows only from the per-airport consistency; no counterfactual was measured.
- That the model's small pooled gate-row bias (+8 s) would persist on Jan 2026, where gate rows are 2.7× more
  frequent and at LTFM/EDDF exceed anything in training, is not implied by any table here; the LTFM +208 s
  forward-fold bias on 81 rows is the only in-sample sign of extrapolation error and rests on a small n.
- That the Jan/Jul SD excess is a tail effect is inferred from its disappearance on y < 7,200 and on matched rows;
  the mechanism of that tail is items 1–3, not this item.
- The −3.13 s / −0.87 s weather-on numbers are attributed to the weather join because the intervening commits are
  described as determinism/perf changes; this was not isolated.
