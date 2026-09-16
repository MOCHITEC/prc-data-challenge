# Item 15 `eham-jan2026-disruption` — verification result (2026-09-16)

Script: `docs/investigation_2026-09-16/scripts/eham-jan2026-disruption.py` (no training, `POLARS_MAX_THREADS=1`, column-selected lazy scans; ~3 min).
Definitions follow plan §0: "unmatched" = routing definition (`AOBT_3_flt` null → model B; 22,470 / 5,290 rows). D's definition (`FLIGHT_ID_mvt` null) gives identical per-day counts at EHAM in January (every day, both years). `proxy` = `MVT_TIME − AOBT_3` (matched rows). `ms` = `MVT_TIME − SCHED_TIME`. `y` = `TAXITIME_SEC_mvt`. "day" = UTC calendar date of `MVT_TIME_UTC_mvt` (as in D's q4 K). OOF = `experiments/oof_main-48c19bc830.parquet` (672,428 non-null rows, months 1/7/11/12, RMSE 334.58). Predictions: `outputs/main.parquet` (= v7) and `outputs/probe_unmatched_from_v4.parquet` (= v9: v4 on the 5,290 unmatched rows).

**Disruption day (plan definition):** a day whose unmatched count exceeds 3 × the median daily unmatched count of that airport in that month (median over the month's calendar days; days with 0 unmatched count as 0). A **strict variant** (3× rule AND ≥ 30 unmatched) is reported alongside because the plain rule fires on 1–8 unmatched rows at airport-months whose median is 0–2 (LEMD Jan 2025 median 0 → 13 "disruption days"; 9 such airport-months in total).

## 1. Hypothesis as stated (plan §1 item 15)

EHAM Jan 2026 ranking contains a multi-day disruption (Jan 2–7, 9, 29) with 68–123 unmatched departures/day (normal 5–15); training has almost no analogue.

Sub-claims tested: (A) the per-day counts and proxy sd quoted by D; (B) the listed days are disruption days under the 3× rule; (C) training has almost no analogue; (D) AOBT_3 minute-rounding and METAR de-icing-gate figures; (E) label / OOF behaviour on training disruption days vs normal days; (F) what v7 and v9 predict for the affected rows; (G) LB effect (unverified in the plan).

## 2. Procedure actually executed

1. `feat_train` (2,085,047 rows) and `feat_rank` (344,841 rows), 13 columns each: per ADEP × day counts of departures, unmatched (both definitions), proxy sd / median / p99, AOBT_3 second==0 share, for Jan/Jul 2025 and Jan/Jul 2026 (S1); month medians and the 3× rule for all 10 airports × 4 months (S2); the same rule over all 12 training months as an extension (S2b); strict variant (S7a).
2. OOF joined to training on `MVT_ID_mvt`: label quantiles and OOF RMSE, matched vs unmatched, on disruption vs normal days (S3a–d); EHAM unmatched rows by daily-unmatched-count band and by `ms` band (S3e, S7f); label − proxy on EHAM matched rows (S7d).
3. Ranking rows on 2026 disruption days by airport × month × matched/unmatched (S4, S7a).
4. `metar_hourly.parquet` aggregated per station-day (temp mean/min, precip sum, min visibility, hours of low_vis / frozen_wx / deicing_gate, max wind), joined to the day table (S5a–b); monthly EHAM shares (S5c); the departure-row `deicing_gate` feature (S5d); precip content and the effective gate definition (S7e).
5. v7 / v9 predictions for EHAM Jan 2026 rows, disruption vs normal × matched/unmatched, per day, per `ms` band; all-airport disruption-day predictions; SSE bound (S6a–e); prediction − proxy on matched rows (S7d).
6. Supplementary: training airport-days with proxy sd ≥ 1,400 or proxy median ≥ 1,500 (S7b); EHAM Jan 2026 departures vs same-weekday median (S7c); FLIGHT prefix mix (S7g).

## 3. Results

### 3.1 EHAM January per day (S1b, S5a, S7c)

| Jan day | unm 2025 | unm 2026 | proxy sd 2025 | proxy sd 2026 | proxy med 2026 | dep 2026 / weekday median | frozen_wx h 2026 | deice h 2026 | t_mean 2026 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 | 9 | 271 | 396 | 754 | – | 8 | 4 | 4.2 |
| **2** | 9 | **94** | 426 | 1,004 | 963 | 0.78 | 10 | 9 | 2.9 |
| **3** | 7 | **108** | 520 | 1,502 | 1,998 | 0.75 | 17 | 17 | 1.6 |
| **4** | 15 | **123** | 407 | 1,963 | 2,352 | 0.57 | 16 | 16 | 1.3 |
| **5** | 35 (dis) | **84** | 1,036 | 1,848 | 2,110 | 0.35 | 8 | 8 | −0.3 |
| **6** | 16 | **68** | 597 | 1,644 | 1,142 | 0.53 | 6 | 6 | −0.8 |
| 7 | 7 | 18 | 226 | 1,440 | 1,573 | 0.34 | 14 | 14 | 0.9 |
| 8 | 3 | 14 | 287 | 367 | 795 | – | 0 | 0 | 1.8 |
| **9** | 10 | **49** | 459 | 1,172 | 1,185 | 0.83 | 8 | 8 | 1.0 |
| 18 | 37 (dis) | 8 | 869 | 346 | 1,006 | – | 0 | 0 | 3.2 |
| 25 | 8 | 19 | 260 | 741 | 1,127 | 0.99 | 0 | 0 | −0.8 |
| **29** | 8 | **69** | 225 | 971 | 923 | 0.99 | 7 | 7 | 0.1 |
| other days | 2–20 | 1–12 | 220–665 | 220–575 | 727–1,158 | – | 0–3 | 0–3 | – |

- EHAM Jan totals: unmatched 805 (2026) vs 293 (2025); departures 16,232 vs 18,460. Jul: 401 vs 515. Both unmatched definitions give the same daily counts at EHAM.
- Monthly median daily unmatched count: EHAM Jan 2025 = 7, Jul 2025 = 14, Jan 2026 = 9, Jul 2026 = 13. 3× thresholds: 21 / 42 / 27 / 39.
- EHAM disruption days (3× rule): **Jan 2, 3, 4, 5, 6, 9, 29 2026** (7 days; 595 unmatched, 2,187 matched rows); **Jan 5 and Jan 18 2025**; **Jul 2 2025** (50 unmatched); Jul 2026 none (max 30). Jan 7 2026 (18 unmatched) is *not* a disruption day by count, although its proxy sd (1,440), proxy median (1,573) and departure count (194 = 34 % of the Wednesday median) match the flagged days.
- Departures on Jan 4–7 2026 are 34–57 % of the same-weekday January median; on Jan 2–3 and 9, 75–83 %; on Jan 29, 99 %. Jan 5 / 18 2025: 94–95 %.
- Proxy sd Jan 2025 by day: 220–665 on 28 days, 869 (Jan 18) and 1,036 (Jan 5). D's "220–600" range omits Jan 5, 11 and 18.
- AOBT_3 second==0 share, row-weighted: EHAM 0.749 (Jan 2025) / 0.676 (Jan 2026) / 0.886 (Jul 2025) / 0.906 (Jul 2026); all other airports 0.964–1.000 in both Januaries. Per day the share moves between 0.43 and 0.96 at EHAM in both years; it is not specifically lower on disruption days (0.50–0.77).

### 3.2 All 10 airports × 4 months (S1a, S2, S7a)

| airport | Jan 25 med / max / dis-days | Jul 25 med / max / dis | Jan 26 med / max / dis | Jul 26 med / max / dis |
|---|---|---|---|---|
| EDDF | 4 / 53 / 1 | 10 / 88 / 2 | 3 / 34 / 8 | 4 / 37 / 4 |
| EDDM | 2 / 13 / 1 | 5 / 15 / 0 | 2 / 33 / 4 | 2 / 21 / 4 |
| EGLL | 3 / 23 / 3 | 3 / 76 / 9 | 1 / 35 / 9 | 4 / 13 / 1 |
| **EHAM** | **7 / 37 / 2** | **14 / 50 / 1** | **9 / 123 / 7** | **13 / 30 / 0** |
| LEBL | 2 / 9 / 1 | 10 / 57 / 3 | 2 / 8 / 2 | 10 / 31 / 2 |
| LEMD | 0 / 8 / 13 | 4 / 70 / 5 | 1 / 9 / 6 | 4 / 18 / 4 |
| LFPG | 5 / 25 / 3 | 14 / 66 / 4 | 5 / 49 / 6 | 10 / 31 / 2 |
| LIRF | 2 / 6 / 0 | 9 / 34 / 1 | 3 / 14 / 2 | 7 / 34 / 1 |
| LSZH | 5 / 16 / 1 | 9 / 41 / 2 | 5 / 61 / 5 | 10 / 29 / 0 |
| LTFM | 7 / 15 / 0 | 12 / 55 / 1 | 10 / 20 / 0 | 13 / 18 / 0 |

- Plain rule: 120 airport-days in the 4 months (2025: 53 days / 1,467 unmatched rows; 2026: 67 days / 1,700 unmatched rows). Strict variant (≥ 30): 42 airport-days (2025: 22 / 1,136 rows; 2026: 20 / 1,080 rows).
- Jan 2–5 2026 is flagged at EDDF, EDDM, EGLL, LFPG, LSZH, LEMD, LEBL and LIRF as well as EHAM, mostly with small counts; under the strict variant only EGLL Jan 4 (35), LFPG Jan 5 (44) and Jan 7 (49) join EHAM in that week. Jan 5 2025 is flagged at EDDF (53), EDDM (13), EGLL (23), EHAM (35), LEBL (9), LEMD (8) and LSZH (16) — a training multi-airport day.
- The largest 2026 non-EHAM disruption days: LSZH Jan 22 (61), LFPG Jan 7 (49), Jan 5 (44), EDDF Jul 17 (37), EGLL Jan 4 (35), EDDF Jan 26 (34), LIRF Jul 2 (34), LSZH Jan 10 (34), EDDM Jan 12 (33). Their proxy sd is 240–1,069 — none reaches the EHAM Jan 3–6 2026 range (1,502–1,963).

### 3.3 Training analogues (S2b, S7b)

- Training airport-days with ≥ 68 unmatched (all 2025): 11 — EDDF 06-01 (107), **EHAM 09-15 (104)**, LFPG 12-05 (100), EDDF 07-27 (88), LTFM 02-19 (85), LTFM 02-23 (84), LFPG 06-26 (81), LFPG 11-22 (77), EGLL 07-06 (76), **EHAM 09-16 (71)**, LEMD 07-03 (70). Proxy sd on those days 230–1,050.
- EHAM top days by unmatched count in 2025: Sep 15 (104), Sep 16 (71), Jul 2 (50), Oct 4 (38), Jan 18 (37), Jan 5 (35), then ≤ 32. September is not an OOF month.
- EHAM unmatched labels on the two ≥ 68 days (175 rows): y p10/p50/p90/max = 437 / 714 / 1,071 / 1,615; 40–67 band (Jul 2 2025, 50 rows): 170 / 583 / 1,129 / 1,602; 20–39 band (622 rows): 205 / 758 / 1,529 / 3,903; < 10 band (1,218 rows): 113 / 352 / 1,092 / 3,129.
- Matched EHAM rows on the ≥ 68 days (1,207): y − proxy sd 702, proxy sd 723 (normal bands 259–367 / 357–503).
- Training airport-days with matched proxy median ≥ 1,500 s: 23 of 3,650; maximum proxy sd among them 1,118 (EGLL 12-31). The only training days with proxy median ≥ 1,800 are LTFM Feb 11, 19, 22, 23 2025 (proxy median 2,084–2,640, proxy sd 34–736, 10–85 unmatched; unmatched y median 2,576–5,409 on those days). No training airport-day combines proxy sd ≥ 1,400 with proxy median ≥ 1,500. Ranking airport-days with proxy median ≥ 1,500: 13, led by EHAM Jan 4 (2,352 / sd 1,963), Jan 5 (2,110 / 1,848), Jan 3 (1,998 / 1,502); the other ten have proxy sd 262–1,440 (EHAM Jan 7: 1,573 / 1,440).
- EHAM's highest training proxy median is 1,204 (Mar 24 2025).

### 3.4 Labels and OOF on training disruption days (S3, S7a, S7d, S7f)

Jan + Jul 2025, plain rule:

| group | n | OOF RMSE | OOF bias (median oof − y) | y p10 / p50 / p90 / p99 / max | n y ≥ 2 h | ms median |
|---|---|---|---|---|---|---|
| normal, matched | 309,533 | 223.8 | +15 | 562 / 908 / 1,490 / 2,458 / 87,002 | 31 | 1,445 |
| normal, unmatched | 3,906 | 2,710.2 | +34 | 324 / 918 / 1,620 / 12,770 / 88,132 | 93 | 5,336 |
| disruption, matched | 29,513 | 284.5 | +31 | 648 / 1,069 / 1,796 / 3,485 / 9,611 | 7 | 2,036 |
| disruption, unmatched | 1,467 | 1,198.7 | +55 | 647 / 1,146 / 1,930 / 7,904 / 40,262 | 18 | 6,522 |

Strict variant: disruption unmatched 1,136 rows RMSE 1,301.7 (y p50 1,127, p90 1,910); normal unmatched 4,237 rows RMSE 2,610.5. All 12 months (per airport-month rule): disruption unmatched 4,531 rows (2,192 in OOF) RMSE 1,252.3 vs normal 17,939 (6,186) RMSE 2,566.0; disruption matched RMSE 278.5 vs 210.0.

EHAM January 2025 (disruption = Jan 5, 18; identical to D's day set):

| group | n | OOF RMSE | y p10 / p50 / p90 / p99 / max |
|---|---|---|---|
| normal, matched | 17,136 | 167.1 | 462 / 724 / 1,157 / 2,321 / 5,590 |
| normal, unmatched | 221 | 824.6 | 126 / 516 / 1,233 / 3,652 / 11,032 |
| disruption, matched | 1,031 | 268.3 | 475 / 764 / 2,377 / 4,396 / 5,298 |
| disruption, unmatched | 72 | 1,076.7 | 652 / 1,006 / 3,576 / 3,839 / 3,903 |

- EHAM, all 2025, unmatched: disruption-day rows (335) RMSE 881.9, y p50 746, p90 1,289; normal (3,749) RMSE 931.2, y p50 486, p90 1,115.
- Per airport (Jan+Jul), unmatched RMSE disruption vs normal: EDDF 788 / 791; EGLL 1,350 / 838; LEBL 1,369 / 805; LFPG 831 / 4,252; LIRF 4,744 / 6,765; LSZH 571 / 969; LTFM 379 / 597; LEMD 353 / 565; EDDM 477 / 662. Matched RMSE is higher on disruption days at every airport except LEMD (172 / 178).
- EHAM matched rows, Jan 2025: y − proxy median −102 (p10 −498, p90 +122) on disruption days vs −100 (−388, +103) on normal days; OOF RMSE 268 vs 167.
- EHAM unmatched training rows by `ms` band (all 2025): ms < 2 h (3,379) y p50 427, p90 1,047; 2 h–10,000 s (435) 838 / 1,522; 10,000 s–4 h (179) 839 / 1,304; ≥ 4 h (91) 788 / 1,424, max 3,849; y ≥ 2 h: 1 row in total (ms < 2 h). No EHAM unmatched training row has y ≥ 4,000 s except one (11,032). EHAM Jan 5 / 18 2025 unmatched with ms 2 h–10,000 s (25 rows): y p50 2,945, p90 3,736, OOF bias median −751.

### 3.5 Ranking rows on 2026 disruption days (S4, S7a)

- Plain rule: 1,700 unmatched (32.1 % of 5,290) and 33,529 matched ranking rows fall on disruption days. Strict variant: 1,080 unmatched / 8,705 matched.
- By airport (plain rule, unmatched / matched, Jan | Jul): EHAM 595 / 2,187 | 0 / 0; LFPG 185 / 3,461 | 62 / 1,374; LSZH 166 / 1,548 | –; EDDF 144 / 3,918 | 86 / 2,588; EGLL 91 / 5,524 | 13 / 661; EDDM 72 / 1,357 | 51 / 1,791; LEMD 38 / 3,266 | 59 / 2,536; LIRF 27 / 824 | 34 / 482; LEBL 15 / 919 | 62 / 1,093.
- EHAM Jan 2026 disruption-day unmatched rows: 595 of 805 (74 %); ms median 10,218 s; 467 have ms ≥ 2 h (training EHAM unmatched with ms ≥ 2 h: 705 rows in 12 months). EHAM disruption-day matched rows: ms median 3,867, 594 with ms ≥ 2 h.
- FLIGHT prefix of the 595: KL 50.3 % (training EHAM unmatched: 12.3 %, y median 713, 0 % ≥ 1 h), HV 7.9 % (6.9 %), DL 4.5 % (1.2 %), EJ 3.9 % (3.2 %); ZX 1.7 % (training 20.8 %, y median 174).

### 3.6 METAR (S5, S7e)

- `precip_mm` is 0 in all 138,446 station-hours of `metar_hourly.parquet`. Consequently `deicing_gate` equals `frozen_wx AND temp_c ≤ 3` on every row (1,326 gate hours; 0 disagreements); `temp_x_precip` is identically 0.
- EHAM monthly shares (hour-weighted): deicing_gate 0.051 (Jan 2025) / 0.128 (Jan 2026); frozen_wx 0.060 / 0.134; low_vis 0.109 / 0.031; mean temp 3.8 / 2.7 °C. Jul: all 0.
- EHAM Jan 2026, mean of daily values, disruption (7 days) vs normal (24): frozen_wx 10.3 vs 1.2 h; deicing_gate 10.1 vs 1.0 h; t_mean 0.8 vs 3.3 °C; t_min −1.4 vs 1.0; vis_min 2.2 vs 3.8 km; low_vis 1.0 vs 0.7 h; max wind 8.3 vs 7.3 m/s. Every disruption day has ≥ 6 frozen hours; two non-flagged days also do (Jan 1: 8 h, 9 unmatched; Jan 7: 14 h, 18 unmatched). After Jan 12 no EHAM day has more than 3 frozen hours; Jan 29 has 7.
- EHAM Jan 2025 disruption days: Jan 5 (4 frozen h, vis_min 1.4 km, t_min 0), Jan 18 (12 frozen h, 13 low-vis h, t_mean −1.0). Jan 21 2025 (9 frozen h, 16 low-vis h) had 7 unmatched.
- Other airports, Jan 2026 disruption vs normal days: frozen hours EDDF 5.5 / 0.8, EDDM 5.2 / 2.3, LSZH 8.6 / 3.5, LFPG 1.8 / 0.0, EGLL 0.1 / 0.0, LEMD 0.5 / 0.0, LEBL 0 / 0, LIRF 0 / 0. July disruption days everywhere: 0 frozen hours, temp 19–27 °C.
- Departure-row feature `deicing_gate` (take-off hour), EHAM Jan 2026: 0.417 on disruption-day unmatched rows, 0.252 on disruption-day matched, 0.071 / 0.017 on normal days. Jan 2025: 0.111 / 0.193 / 0.036 / 0.021. Training EHAM unmatched rows with gate = 1 (35, 29 in OOF): y p50 1,102, p90 2,484, OOF bias median +496.

### 3.7 Predictions (S6, S7d, S7f)

EHAM Jan 2026 (v7 = `main.parquet`; v9 = `probe_unmatched_from_v4.parquet`):

| group | n | v7 p10 / p50 / p90 / max | v7 ≥ 2 h | v9 p10 / p50 / p90 / max | v9 ≥ 2 h | RMS(v7 − v9) | median(v7 − v9) |
|---|---|---|---|---|---|---|---|
| normal, matched | 13,240 | 531 / 746 / 1,183 / 10,305 | 1 | 528 / 739 / 1,151 / 10,109 | 1 | 78 | +6 |
| normal, unmatched | 210 | 156 / 664 / 2,054 / 6,545 | 0 | 120 / 670 / 1,837 / 7,418 | 1 | 481 | +12 |
| disruption, matched | 2,187 | 603 / 1,006 / 3,742 / 9,596 | 5 | 603 / 1,000 / 3,716 / 9,763 | 4 | 159 | +6 |
| disruption, unmatched | 595 | 767 / 1,846 / 3,263 / 6,930 | 0 | 912 / 2,134 / 3,401 / 5,694 | 0 | 713 | −129 |

- All 805 EHAM Jan 2026 unmatched rows: v7 median 1,524, p90 3,152 (D's figure reproduced); v9 median 1,693, p90 3,297.
- Per disruption day, v7 / v9 unmatched medians: Jan 2 1,192 / 1,278; Jan 3 2,012 / 2,466; Jan 4 2,238 / 2,675; Jan 5 2,352 / 2,588; Jan 6 1,463 / 1,619; Jan 9 2,396 / 2,491; Jan 29 1,499 / 1,433. v9 (v4 model B) is higher than v7 on 6 of 7 days; RMS difference 383–1,051 per day.
- Disruption-day unmatched by `ms` band, v7 median: < 2 h (128) 1,627; 2 h–10,000 s (163) 1,918; 10,000 s–4 h (176) 1,943; ≥ 4 h (128) 1,796 (p10 57). v9: 1,587 / 2,306 / 2,258 / 2,473.
- Matched disruption-day rows: v7 − proxy median −115 (p10 −482, p90 +106) vs −103 (−317, +34) on normal days; training EHAM y − proxy on Jan 2025 disruption days −102 (−498, +122).
- All ranking unmatched rows on disruption days (1,700): v7 median 1,430, p90 3,072, 16 ≥ 2 h; normal-day unmatched (3,590): 984 / 2,060, 84 ≥ 2 h. The ≥ 2 h predictions on disruption days are LIRF (7 Jan + 8 Jul) and EGLL Jul (1).
- SSE bound (facts of arithmetic): the 595 rows are 0.173 % of the ranking set (11.2 % of unmatched). If their RMSE were 1,000 / 2,000 / 3,000 / 5,000 s, the contribution to overall MSE would be 1,725 / 6,902 / 15,529 / 43,136 s², i.e. an overall score of 307.8 / 316.1 / 329.5 / 369.0 on top of an otherwise-305.0 submission.

## 4. Verdict per sub-claim

- **(A) Per-day counts and proxy sd as quoted by D — CONFIRMED.** d2 94, d3 108, d4 123, d5 84, d6 68, d7 18, d9 49, d29 69, others 1–19; Jan 2025 max 37; proxy sd d3 1,502, d4 1,963, d5 1,848, d6 1,644, d7 1,440; 805 vs 293 and 401 vs 515; AOBT_3 rounding 0.676 vs 0.749; METAR gate share 0.128 vs 0.051; training Jan 5/18 2025 unmatched n = 72, median 1,006, p90 3,576 (D: 3,580 — interpolation) vs 516 / 1,233; v7 median 1,524, p90 3,152 — all reproduced. D's Jan 2025 proxy-sd range "220–600" omits Jan 5 (1,036), Jan 18 (869) and Jan 11 (665).
- **(B) "Jan 2–7, 9, 29" are disruption days with 68–123 unmatched/day, normal 5–15 — CONFIRMED with two corrections.** Under the 3× rule the EHAM Jan 2026 disruption days are Jan 2–6, 9, 29 (7 days, 595 unmatched rows). Jan 7 (18 unmatched) does not pass the count rule; it does match the flagged days on proxy sd (1,440), proxy median (1,573) and departure shortfall (34 % of weekday median). Jan 9 has 49, below the stated 68–123. The normal-day median is 9 (range 1–19), consistent with "5–15".
- **(C) Training has almost no analogue — CONFIRMED for the combination, REFUTED for the unmatched count alone.** By unmatched count, EHAM has Sep 15–16 2025 (104, 71), Jul 2 (50) and Jan 5/18 (35, 37); all airports have 11 days ≥ 68 in 2025. By matched proxy sd (1,502–1,963) with proxy median ≈ 2,000–2,350 and departures at 35–57 % of normal, no training airport-day matches: the highest training proxy sd on any day is 1,118 (excluding one LIRF day whose sd 3,993 comes from a single outlier row, proxy median 896); the only training days with proxy median ≥ 1,800 are four LTFM days in Feb 2025 with proxy sd ≤ 736 and no departure shortfall check performed.
- **(D) AOBT_3 rounding and METAR gate figures — CONFIRMED**, with the addition that `precip_mm` is zero in every METAR hour, so `deicing_gate` is exactly `frozen_wx & temp ≤ 3 °C`.
- **(E) Labels / OOF differ on training disruption days — CONFIRMED as a shift, direction mixed.** Unmatched labels are higher on disruption days (Jan+Jul: p50 1,146 vs 918, p90 1,930 vs 1,620; EHAM Jan: 1,006 vs 516, 3,576 vs 1,233) but the extreme tail (p99, max, y ≥ 2 h counts) is smaller, so pooled unmatched OOF RMSE is lower on disruption days (1,199 vs 2,710); at EHAM Jan 2025 it is higher (1,077 vs 825). Matched OOF RMSE is higher on disruption days at 9 of 10 airports (pooled 284.5 vs 223.8; EHAM Jan 268 vs 167).
- **(F) What v7 / v9 predict for the affected rows — CONFIRMED (measured).** Disruption-day unmatched: v7 median 1,846 (p90 3,263), v9 median 2,134 (p90 3,401), none ≥ 2 h; normal-day unmatched 664 / 670. These predictions exceed every EHAM unmatched training-label median observed on high-count days (714 on the ≥ 68 days, 1,006 on Jan 5/18 2025) and the EHAM training p90 for ms ≥ 2 h rows (1,304–1,522); they are below the Jan 5/18 2025 p90 (3,576).
- **(G) LB effect — UNDETERMINED.** No labels; bound only: the 595 rows move the score by ≤ +3 s at RMSE 1,000 and ≤ +11 s at RMSE 2,000 (§3.7).

## 5. Not determinable locally

- Ranking labels for the 595 (or 805) EHAM Jan 2026 unmatched rows and the 2,187 disruption-day matched rows; hence the actual error and the LB effect of the event.
- Whether the Jan 2–7 2026 event's unmatched rows carry the ordinary label mechanism (BLOCK from a real off-block) or an artefact (sched-fill, rollover) — training EHAM has fill10 0.027 and no sched-fill on its disruption days, but the ranking rows cannot be checked.
- The operational cause of the event (weather vs other); only METAR co-occurrence is measurable here.
- The cause of the Sep 15–16 2025 EHAM high-unmatched days (proxy sd 711–731, no frozen weather, labels benign) — not an OOF month, so its model-B error is unknown.

## 6. Facts vs inference

**Facts (measured in this run):** every number in §3; the day lists in §3.1–3.2; identity `deicing_gate == frozen_wx & temp_c ≤ 3` on all METAR rows; zero `precip_mm` in all 138,446 hours; equality of the two unmatched definitions on every EHAM January day; v7 / v9 prediction quantiles; the SSE arithmetic in §3.7.

**Inference (not established by the data here):**
- That Jan 2–7 2026 at EHAM was a winter-weather operational disruption. Supporting facts: 6–17 frozen-precipitation hours on each flagged day, departures at 34–83 % of weekday norms, matched taxi proxies doubled, KLM share of unmatched rows 50 % vs 12 % in training. The METAR co-occurrence is not causal evidence; Jan 1 and Jan 7 2026 had 8 and 14 frozen hours with 9 and 18 unmatched rows.
- That the multi-airport flags on Jan 2–5 2026 and Jan 5 2025 reflect one network-wide event rather than independent per-airport noise; under the strict variant only EGLL Jan 4 and LFPG Jan 5/7 2026 remain flagged outside EHAM.
- That "training has no analogue" implies model-B error on these rows is larger than on normal rows. Training evidence points both ways: unmatched labels on disruption days are higher in the middle of the distribution but lack the extreme tail, and pooled OOF RMSE is lower; at EHAM the OOF RMSE was higher (1,077 vs 825) on 72 rows. The predictions v7/v9 emit for the 595 rows (median 1,846 / 2,134) are already above every EHAM disruption-day training-label median, so the direction of any error is not implied by the training analogues.
- That the 3× rule is a suitable operational definition: it fires on 1–8 rows at low-median airport-months (LEMD Jan 2025: 13 days) and misses EHAM Jan 7 2026; the strict variant (≥ 30) removes the former and does not fix the latter.
