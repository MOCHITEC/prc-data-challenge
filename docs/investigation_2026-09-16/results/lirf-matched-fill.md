# Item 4 `lirf-matched-fill` — verification result (2026-09-16)

Script: `docs/investigation_2026-09-16/scripts/lirf-matched-fill.py` (no training, `POLARS_MAX_THREADS=1`, ~3 s wall).
Inputs: `data/work/feat_train.parquet`, `data/work/feat_rank.parquet`, `experiments/oof_main-48c19bc830.parquet`.
All numbers below were re-derived by this script; none are copied from the hypothesis reports.

## 1. Hypothesis as stated (plan §1 item 4, source D)

> LIRF matched rows also have sched-filled BLOCK; when AOBT_3 is far from SCHED the residual
> y − proxy = AOBT_3 − SCHED with large sd, making LIRF the worst matched airport.

Sub-claims tested:

- S1 LIRF matched rows have "filled" BLOCK (|BLOCK − SCHED| ≤ 10 s) at 17.8 % (14.5–20.6 % by month), far above other airports.
- S2 On filled rows the proxy residual y − proxy equals AOBT_3 − SCHED; for |AOBT_3 − SCHED| > 600 s its sd is ≈1,138 s (fill 8.4 %, ≈6,460 rows), vs sd 253 for ≤ 600 s (fill 26.6 %).
- S3 Fill by operator (n > 2,000): 54 / 25 / 22 / 19 / 14 / 11 %, rest ≤ 5.6 %; by segment Cargo 47 %, Non-Scheduled 45 %, Lowcost 22 %, Mainline 14 %.
- S4 LIRF matched rows with |BLOCK − AOBT_3| > 1 h: 457, y median 5,274, proxy median 1,025, 80 % filled.
- S5 OOF (`oof_main-48c19bc830`): LIRF matched RMSE 347 (next LFPG 246, EHAM 162 lowest), 7.9 % of SSE; filled ∧ far subset 1,856 rows, 3.0 % SSE, RMSE 1,106; actype-null among LIRF matched: 0.
- S6 (causal wording) the fill is what makes LIRF the worst matched airport.
- S7 (listed as unverified) the fill state is recoverable for matched rows from available columns.
- Extension requested by the coordinator: fill by operator × hour × market segment; whether the same operator values appear in ranking LIRF matched rows and how many rows.

## 2. Procedure actually executed

Definitions (plan §0): `y = TAXITIME_SEC_mvt`; `proxy = MVT − AOBT_3`; `resid = y − proxy`; `as_ = AOBT_3 − SCHED`;
`bs = BLOCK − SCHED`; `fill = |bs| ≤ 10 s` (D's threshold; ±6 s and ≤ 120 s variants also computed);
`far = |as_| > 600 s`; matched = `AOBT_3` non-null (routing definition); D's FLIGHT_ID definition also counted.

1. Frame sanity on `feat_train` (16 columns selected via `scan_parquet`): row count, PHASE, matched counts under both definitions, exactness of `proxy == mvt_minus_aobt3`, `as_ == aobt3_minus_sched`, `y == MVT − BLOCK`, `hour_utc == hour(MVT)`.
2. Fill rate by ADEP × matched (three thresholds); fill by ADEP × |AOBT_3 − SCHED| band (matched rows, all airports).
3. LIRF matched (159,216 rows, collected in memory): fill by month; by `far`; by seven |AOBT_3 − SCHED| bands; sd of `resid` and `y` split by fill state; identity check `resid == as_` on filled rows.
4. Decomposition of Σresid² over fill × far cells; sd(resid) by airport with and without filled rows.
5. Rows with |BLOCK − AOBT_3| > 1 h.
6. Fill by operator (n > 2,000 and n ≥ 200 & fill ≥ 0.30), by segment, by hour_utc, by operator × segment × hour_utc (cells n ≥ 100); between-group variance share R² = Σ n_g (p_g − p)² / (N p (1 − p)) of the fill flag for several groupings (group means only, no model fit).
7. `feat_rank` LIRF rows: matched count by month, distinct operators, overlap with training LIRF-matched operators, ranking rows per training-fill bucket of their operator, expected filled count under training cell rates, `far` share, segment counts.
8. OOF join on `MVT_ID_mvt` (rows with non-null `oof`): RMSE / SSE share by ADEP × matched; LIRF matched by fill × far; RMSE by airport excluding filled rows; |BLOCK − AOBT_3| > 1 h rows; correlation of OOF error with `as_` on filled ∧ far rows and on the not-filled ∧ far control.
9. OOF file coverage (row count, null count, months).

## 3. Results

### 3.1 Sanity (feat_train)

| quantity | value |
|---|---|
| rows / PHASE = DEP | 2,085,047 / 2,085,047 |
| matched (AOBT_3 non-null) / unmatched | 2,062,577 / 22,470 |
| matched (FLIGHT_ID non-null) | 2,062,623 (46 rows FLIGHT_ID non-null & AOBT_3 null) |
| max \|proxy − mvt_minus_aobt3\|, max \|as_ − aobt3_minus_sched\|, max \|y − (MVT − BLOCK)\| | 0.0 / 0.0 / 0.0 |
| rows with hour_utc ≠ hour(MVT) | 0 |
| LIRF rows total / matched (routing) / matched (FLIGHT_ID) / unmatched | 160,704 / 159,216 / 159,217 / 1,488 |
| AIRCRAFT_TYPE_mvt null among LIRF matched; operator null; segment null | 0 / 0 / 0 |

### 3.2 Fill rate by airport × matched (S1)

| ADEP | matched n | fill ≤10 s | fill ±6 s | fill ≤120 s | unmatched n | unm. fill ≤10 s |
|---|---|---|---|---|---|---|
| LIRF | 159,216 | **0.178** | 0.178 | 0.257 | 1,488 | 0.484 |
| LEBL | 177,843 | 0.057 | 0.057 | 0.179 | 1,862 | 0.019 |
| LTFM | 269,320 | 0.051 | 0.051 | 0.202 | 3,645 | 0.050 |
| EDDM | 166,324 | 0.050 | 0.050 | 0.198 | 1,010 | 0.000 |
| EGLL | 238,132 | 0.049 | 0.049 | 0.183 | 1,414 | 0.003 |
| LEMD | 211,295 | 0.045 | 0.045 | 0.160 | 947 | 0.002 |
| LFPG | 235,779 | 0.042 | 0.042 | 0.156 | 3,773 | 0.003 |
| EDDF | 228,185 | 0.012 | 0.007 | 0.137 | 1,956 | 0.002 |
| EHAM | 243,867 | 0.012 | 0.008 | 0.135 | 4,084 | 0.027 |
| LSZH | 132,616 | 0.011 | 0.007 | 0.114 | 2,291 | 0.023 |

LIRF matched filled rows: 28,369. Fill by month (LIRF matched): Jan .191, Feb .202, Mar .206, Apr .173, May .176, Jun .168, Jul .145, Aug .167, Sep .159, Oct .174, Nov .198, Dec .201 (range 0.145–0.206).

Fill rate by |AOBT_3 − SCHED| band, matched rows, all airports:

| ADEP | ≤60 | 60–300 | 300–600 | 600–1800 | 1800–3600 | 3600–7200 | >7200 |
|---|---|---|---|---|---|---|---|
| **LIRF** | **0.372** | **0.296** | **0.183** | **0.098** | **0.069** | **0.050** | 0.006 |
| LEBL | 0.176 | 0.102 | 0.041 | 0.019 | 0.004 | 0.000 | 0.000 |
| EDDM | 0.158 | 0.092 | 0.017 | 0.010 | 0.000 | 0.000 | 0.000 |
| LEMD | 0.154 | 0.094 | 0.020 | 0.005 | 0.001 | 0.000 | 0.000 |
| LFPG | 0.149 | 0.092 | 0.039 | 0.007 | 0.000 | 0.000 | 0.000 |
| EGLL | 0.113 | 0.089 | 0.029 | 0.005 | 0.004 | 0.001 | 0.000 |
| LTFM | 0.070 | 0.081 | 0.081 | 0.019 | 0.004 | 0.005 | 0.000 |
| LSZH | 0.059 | 0.026 | 0.006 | 0.001 | 0.000 | 0.000 | 0.000 |
| EHAM | 0.044 | 0.026 | 0.004 | 0.000 | 0.000 | 0.000 | 0.000 |
| EDDF | 0.039 | 0.030 | 0.004 | 0.000 | 0.000 | 0.000 | 0.000 |

LIRF band sizes: 15,031 / 35,700 / 31,572 / 48,566 / 19,511 / 6,903 / 1,933.

### 3.3 |AOBT_3 − SCHED| bands, residual sd (S2) — LIRF matched

| far (>600 s) | n | fill | n_fill | sd resid all | sd resid filled | sd resid not filled | sd y all | sd y filled | sd y not filled | y med | proxy med | resid med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| false | 82,303 | 0.266 | 21,916 | 271.1 | **252.6** | 276.3 | 435.0 | 652.8 | 321.2 | 964 | 900 | 59 |
| true | 76,913 | **0.084** | **6,453** | 657.6 | **1,138.3** | 476.7 | 690.3 | **1,202.7** | 506.4 | 1,132 | 900 | 183 |

Finer bands (LIRF matched; sd resid filled / not filled): ≤60 s 48.7 / 260.4; 60–300 196.1 / 262.4; 300–600 335.5 / 294.5; 600–1800 501.9 / 398.5; 1800–3600 574.3 / 520.7; 3600–7200 749.4 / 660.6; >7200 1,154.0 / 750.7 (11 filled rows).

Identity on the 28,369 filled rows: |resid − as_| ≤ 10 s on 100.0 %; max 6 s; median 3 s. (sd resid filled == sd as_ filled to 0.1 s in every band.)

Σresid² decomposition, LIRF matched (Σ over all 159,216 rows):

| fill | far | n | share of n | share of Σresid² | sd resid | mean resid | median \|resid\| |
|---|---|---|---|---|---|---|---|
| false | false | 60,387 | 37.9 % | 9.9 % | 276.3 | 71 | 175 |
| false | true | 70,460 | 44.3 % | 41.3 % | 476.7 | 250 | 237 |
| true | false | 21,916 | 13.8 % | 3.5 % | 252.6 | 122 | 183 |
| true | true | 6,453 | **4.1 %** | **45.3 %** | 1,138.3 | 1,476 | 1,135 |

sd(resid) LIRF matched: all rows 514.6; excluding filled 406.9; excluding filled ∧ far 388.8.

sd(resid) by airport, matched rows (sorted by all-rows sd):

| ADEP | n | fill | sd resid all | sd resid excl. filled | sd resid excl. filled∧far | RMS resid | p99 \|resid\| | n \|BLK−AOBT\|>1h |
|---|---|---|---|---|---|---|---|---|
| LIRF | 159,216 | 0.178 | **514.6** | 406.9 | 388.8 | 557.4 | 2,282 | 457 |
| LTFM | 269,320 | 0.051 | 475.0 | **475.7** | 468.3 | 530.6 | 1,624 | 657 |
| EGLL | 238,132 | 0.049 | 417.9 | 420.4 | 413.0 | 418.5 | 1,564 | 296 |
| LFPG | 235,779 | 0.042 | 377.9 | 379.5 | 375.2 | 379.5 | 1,085 | 186 |
| EDDM | 166,324 | 0.050 | 339.2 | 341.0 | 335.7 | 348.9 | 1,375 | 45 |
| LEBL | 177,843 | 0.057 | 304.0 | 295.3 | 292.3 | 311.4 | 1,204 | 13 |
| EHAM | 243,867 | 0.012 | 292.6 | 293.7 | 292.5 | 329.6 | 1,258 | 54 |
| LEMD | 211,295 | 0.045 | 274.3 | 273.8 | 271.0 | 276.4 | 1,079 | 20 |
| LSZH | 132,616 | 0.011 | 261.8 | 262.0 | 261.4 | 268.4 | 804 | 30 |
| EDDF | 228,185 | 0.012 | 249.5 | 249.8 | 249.1 | 255.1 | 745 | 50 |

### 3.4 |BLOCK − AOBT_3| > 1 h at LIRF matched (S4)

n = 457; y median 5,274; proxy median 1,025; fill 0.799 (365 filled / 92 not); far 0.991; BLOCK < AOBT_3 in 99.6 %; AOBT_3 − SCHED median 4,260 s. Not-filled subset (92): y median 5,155.5, proxy median 1,015, as_ median 4,620.

### 3.5 Fill by operator / segment / hour (S3) and extension

Operators with n > 2,000 among LIRF matched (11 of 181 distinct):

| operator (hash prefix) | n | fill | segment mode | far | sd resid |
|---|---|---|---|---|---|
| 48910dcb | 8,000 | **0.543** | Lowcost | 0.322 | 507.4 |
| 15411f3d | 13,312 | 0.253 | Lowcost | 0.427 | 543.4 |
| 59972ea6 | 2,583 | 0.220 | Lowcost | 0.491 | 613.3 |
| 3ba01c3b | 19,814 | 0.194 | Lowcost | 0.456 | 411.8 |
| 65b92f7a | 44,184 | 0.144 | Mainline | 0.501 | 390.3 |
| 4c1c009e | 5,312 | 0.107 | Lowcost | 0.485 | 545.6 |
| 4ac51f35 | 2,241 | 0.056 | Mainline | 0.501 | 506.4 |
| 1901074f | 5,077 | 0.050 | Lowcost | 0.464 | 304.3 |
| 11dca3fa | 3,064 | 0.048 | Mainline | 0.437 | 299.3 |
| aa14d43d | 2,304 | 0.046 | Mainline | 0.504 | 316.6 |
| 8e92b7d7 | 2,723 | 0.044 | Mainline | 0.448 | 651.1 |

Smaller operators (n ≥ 200) with fill ≥ 0.30: c9dc9ccb 345 rows fill 1.000 (Lowcost); fc83d369 270 rows 1.000 (Mainline); 5689a906 385 rows 0.992 (Mainline); 66491b1a 256 rows 0.906 (Cargo); 36947b38 741 rows 0.800 (Mainline); 4434948b 214 rows 0.435; 7da5f0e6 785 rows 0.372 (Cargo); 14853ee6 507 rows 0.359 (Cargo); 116ac666 951 rows 0.342 (Lowcost).

By segment (LIRF matched): Mainline 85,897 rows fill 0.136; Lowcost 65,218 / 0.222; Regional 4,750 / 0.155; Cargo 1,748 / **0.469**; Non-Scheduled 1,081 / **0.452**; Other 461 / 0.210; Business 51 / 0.529; Military 10 / 0.700.

By hour_utc (LIRF matched): hours with n ≥ 1,000 range 0.137 (16 h) – 0.251 (3 h); 0–2 h and 23 h (n = 93 / 54 / 171 / 310) 0.280 / 0.370 / 0.246 / 0.265; broadly 0.20–0.25 at 3–7 h, 0.14–0.20 at 8–22 h.

Operator × segment × hour_utc: 1,961 cells; 352 cells with n ≥ 100 covering 138,658 rows and 24,508 of the 28,369 filled rows. Among n ≥ 100 cells: 7 cells (1,219 rows) have fill ≥ 0.8 (operators c9dc9ccb h11 1.000 n139; fc83d369 h19 1.000 n173; 465c27aa h9 0.972 n106; 66491b1a Cargo h21 0.897 n175; 36947b38 h8/h19/h20 0.855/0.816/0.814); 90 cells (16,854 rows) have fill ≤ 0.05. Largest operator 65b92f7a (Mainline) varies 0.096 (h17) – 0.201 (h6) across hours; operator 48910dcb varies 0.384 (h21) – 0.781 (h5).

Between-group variance share of the fill flag (in-sample group means):

| grouping | groups | R² |
|---|---|---|
| operator | 181 | 0.156 |
| segment | 8 | 0.022 |
| hour_utc | 24 | 0.006 |
| far | 2 | 0.057 |
| operator × hour | 1,554 | 0.175 |
| operator × segment | 287 | 0.157 |
| operator × segment × hour | 1,961 | 0.178 |
| operator × segment × hour × far | 2,937 | 0.238 |
| operator × segment × hour × month | 9,710 | 0.231 |

### 3.6 Ranking LIRF matched rows — operator overlap (extension)

- feat_rank LIRF rows 26,899: matched 26,516 (Jan 10,954, Jul 15,562), unmatched 383; FLIGHT_ID definition gives the same 26,516.
- Distinct operators 129; 119 appear among training LIRF matched operators. Rows with a seen operator **26,366 of 26,516 (99.4 %)**; unseen-operator rows 150 (10 operators); null operator 0.
- The 11 big training operators account for 18,246 ranking rows: 48910dcb (train fill 0.543) 1,563 rows; 0.2–0.5 bucket (15411f3d, 59972ea6) 3,150; 0.1–0.2 bucket (3ba01c3b, 65b92f7a, 4c1c009e) 11,079; < 0.1 bucket 2,454; remaining 8,270 rows belong to operators with n_train ≤ 2,000 or unseen.
- The high-fill small operators also appear: 36947b38 (train fill 0.800) 121 ranking rows; 116ac666 (0.342) 120; 7da5f0e6 (0.372) 128; b04ffa54 (0.277) 338; 95828822 (0.259) 243; f3270be8 (0.299) 103.
- Expected number of filled rows in ranking LIRF matched if training rates apply: 4,796 (cell rates, 26,116 rows with a matching cell), 4,884 (operator rates), 4,725 (flat 0.178).
- `far` share ranking 0.521 vs training 0.483. Segments: Mainline 14,187, Lowcost 10,742, Regional 1,019, Cargo 295, Non-Scheduled 157, Other 72, Military 32, Business 12.

### 3.7 OOF `oof_main-48c19bc830` (S5)

File coverage: 2,085,047 rows (one per training MVT_ID); `oof` non-null on 672,428 rows = months 1 (153,706), 7 (190,713), 11 (162,332), 12 (165,677); null on the other 8 months. RMSE on non-null rows 334.58.

By ADEP × matched (SSE share of total OOF SSE):

| ADEP | matched n | RMSE | SSE % | | unmatched n | RMSE | SSE % |
|---|---|---|---|---|---|---|---|
| LIRF | 49,387 | **347.1** | **7.90** | | 537 | 7,179.3 | 36.77 |
| LFPG | 76,471 | 246.1 | 6.15 | | 1,565 | 2,728.8 | 15.48 |
| LTFM | 90,041 | 234.3 | 6.57 | | 1,297 | 508.4 | 0.45 |
| EGLL | 78,986 | 230.0 | 5.55 | | 647 | 1,099.0 | 1.04 |
| LEBL | 55,501 | 196.6 | 2.85 | | 652 | 1,044.1 | 0.94 |
| EDDF | 71,489 | 178.6 | 3.03 | | 787 | 1,073.4 | 1.20 |
| EDDM | 51,286 | 178.2 | 2.16 | | 402 | 999.7 | 0.53 |
| LEMD | 70,316 | 172.3 | 2.77 | | 416 | 418.3 | 0.10 |
| LSZH | 42,287 | 168.9 | 1.60 | | 722 | 802.8 | 0.62 |
| EHAM | 78,286 | 162.1 | 2.73 | | 1,353 | 926.8 | 1.54 |

LIRF matched OOF by fill × far:

| fill | far | n | RMSE | err median | y med | oof med | % total SSE | % LIRF-matched SSE |
|---|---|---|---|---|---|---|---|---|
| false | false | 19,057 | 204.9 | +15.5 | 954 | 961 | 1.06 | 13.4 |
| false | true | 21,380 | 355.7 | +72.5 | 1,074 | 1,155 | 3.59 | 45.5 |
| true | false | 7,094 | 157.8 | −7.5 | 1,010 | 980 | 0.23 | 3.0 |
| true | true | **1,856** | **1,105.6** | **−450.7** | 2,103.5 | 1,483 | **3.01** | 38.1 |

Matched OOF RMSE excluding filled rows: LIRF **294.4** (excl. filled ∧ far 278.3); LFPG 247.8; LTFM 236.2; EGLL 231.6; LEBL 183.4; EDDM 180.1; EDDF 178.6; LEMD 171.8; LSZH 168.6; EHAM 162.4 — LIRF remains first.

LIRF matched OOF rows with |BLOCK − AOBT_3| > 1 h: 157, RMSE 2,505.3, 1.31 % of total SSE, fill 0.720.

Filled ∧ far rows (1,856): corr(err, as_) = −0.694; corr(oof − proxy, as_) = +0.698; median (oof − proxy) 569 vs median as_ 1,140 (= median resid 1,135); median err −451.
Control, not filled ∧ far (21,380): corr(err, as_) = 0.050; corr(oof − proxy, as_) = 0.265; median (oof − proxy) 242 vs median resid 123.

## 4. Verdicts per sub-claim

| # | sub-claim | verdict | deciding observation |
|---|---|---|---|
| S1 | LIRF matched fill 17.8 %, 14.5–20.6 % by month, far above other airports | **CONFIRMED** | 0.1782 (28,369 / 159,216); months 0.145–0.206; other airports 0.011–0.057 at ≤ 10 s. LIRF exceeds every other airport in every \|AOBT_3−SCHED\| band up to 7,200 s (e.g. 600–1800 s: 0.098 vs ≤ 0.019). Caveat: at ≤ 120 s LIRF is 0.257 vs 0.114–0.202 elsewhere, so the distinctiveness is specific to the tight threshold (BLOCK exactly on SCHED's minute). |
| S2a | On filled rows resid = AOBT_3 − SCHED | **CONFIRMED** | 100 % within 10 s, max 6 s (algebraic identity, holds by construction of `fill`). |
| S2b | \|AOBT_3−SCHED\| ≤ 600: fill 26.6 %, sd 253; > 600: fill 8.4 %, ≈6,460 rows, sd 1,138, y sd 1,203 | **CONFIRMED** | 0.266 / 252.6; 0.084 / 6,453 rows / 1,138.3 / 1,202.7. |
| S3 | Fill by operator 54/25/22/19/14/11 %, rest ≤ 5.6 %; segments Cargo 47, NS 45, Lowcost 22, Mainline 14 | **CONFIRMED** | 0.543/0.253/0.220/0.194/0.144/0.107, then 0.056–0.044; Cargo 0.469, NS 0.452, Lowcost 0.222, Mainline 0.136. |
| S4 | \|BLOCK−AOBT_3\| > 1 h: 457 rows, y med 5,274, proxy med 1,025, 80 % filled | **CONFIRMED** | 457 / 5,274 / 1,025 / 0.799. |
| S5 | OOF LIRF matched RMSE 347, 7.9 % SSE, LFPG 246 next, EHAM 162; filled∧far 1,856 rows 3.0 % SSE RMSE 1,106; actype-null 0 | **CONFIRMED** | 347.1 / 7.90 % / 246.1 / 162.1; 1,856 / 3.01 % / 1,105.6; 0. |
| S6 | The fill makes LIRF the worst matched airport | **REFUTED as sufficient cause; CONFIRMED as largest single contributor** | Removing all filled rows lowers LIRF matched OOF RMSE 347.1 → 294.4, but LIRF stays first (LFPG 247.8). In raw proxy-residual sd the fill does account for LIRF's top rank (514.6 → 406.9, below LTFM 475.7 and EGLL 420.4). The 6,453 filled∧far rows are 4.1 % of LIRF matched rows and 45.3 % of Σresid²; in OOF they are 38.1 % of LIRF-matched SSE. The not-filled∧far cell (21,380 OOF rows, RMSE 355.7) carries more SSE (45.5 %). |
| S7 | Fill state recoverable from available columns (matched rows) | **UNDETERMINED (no model fit); group-mean evidence weak** | In-sample R² of the fill flag on group means: operator 0.156, operator×segment×hour 0.178, +far 0.238 (2,937 groups). Only 7 cells (1,219 rows, 0.8 % of LIRF matched) have fill ≥ 0.8; 90 cells (16,854 rows) have fill ≤ 0.05. The model already partially recovers the offset on filled∧far OOF rows: corr(oof − proxy, as_) = 0.698, median recovered offset 569 of 1,140 s. |
| Ext-1 | Fill by operator × hour × segment | **DONE (facts in §3.5)** | 1,961 cells; hour adds R² 0.156 → 0.175 over operator alone; segment adds 0.001. |
| Ext-2 | Same operator values appear in ranking LIRF matched rows; how many rows | **CONFIRMED** | 119 of 129 ranking operators seen in training LIRF matched; 26,366 of 26,516 rows (99.4 %); 150 rows from 10 unseen operators; the 0.543-fill operator contributes 1,563 ranking rows, the 0.800-fill operator 121. Expected filled rows under training rates ≈ 4,725–4,884. |

Additional facts found while verifying (not part of the stated hypothesis):

- F1 At LIRF the fill rate falls monotonically with |AOBT_3 − SCHED| (0.372 → 0.296 → 0.183 → 0.098 → 0.069 → 0.050 → 0.006) and at > 7,200 s equals the other airports' rate (0.000–0.000). At other airports fill is 0.04–0.18 at ≤ 60 s and ≤ 0.02 beyond 600 s.
- F2 `oof_main-48c19bc830.parquet` has 2,085,047 rows, `oof` non-null on 672,428 (months 1, 7, 11, 12 of 2025). Plan §0's "672,428 rows" is the non-null count. Aggregates over the file that do not filter nulls report inflated `n` (e.g. 159,216 instead of 49,387 for LIRF matched) while RMSE/SSE are unaffected (polars skips nulls).
- F3 Under the routing definition the LFPG unmatched OOF cell has 1,565 rows (D reports 1,562 under the FLIGHT_ID definition); SSE share identical at 15.48 %.
- F4 On filled∧far OOF rows the model under-predicts (median err −451; oof median 1,483 vs y median 2,103); on not-filled∧far rows it over-predicts slightly (median err +72.5).

## 5. What cannot be determined locally

- Whether ranking LIRF matched rows are filled at the training rate (labels unknown); only the operator/segment/hour/far composition can be compared (done: operator overlap 99.4 %, far share 0.521 vs 0.483, so the expected filled count under training rates is 4.7–4.9 k of 26,516).
- The LB effect of any treatment of filled rows (needs a probe submission; out of scope).
- Whether "fill" at LIRF is an upstream data-filling step (BLOCK := SCHED) or genuinely on-time minute-resolution off-block times: the columns available carry no provenance flag. F1 constrains but does not decide this (see §6).
- Whether a trained classifier would recover the fill state better than the group-mean R² ≤ 0.24 indicates (would require a model fit, which this item does not permit).

## 6. Facts vs inference

Facts (all measured by the script): every number in §3 and in the deciding-observation column of §4.

Inference (not measured):

- I1 The hypothesis's causal wording ("making LIRF the worst matched airport") is only partly supported: the filled rows explain the LIRF-specific excess in proxy-residual variance (sd 514.6 → 406.9) and about half of the OOF RMSE gap to the next airport (347.1 − 246.1 = 101 → 294.4 − 247.8 = 47), but a residual LIRF excess (294 vs 248) remains on non-filled rows and is not explained by this item.
- I2 F1 (fill rate decreasing with |AOBT_3 − SCHED| and vanishing above 2 h) is what a mixture of (a) an on-time-coincidence component present at all airports and (b) a LIRF-specific excess concentrated at small-to-moderate delays would produce; it is not what a fill process independent of the actual pushback delay would produce (that would give a flat rate). Which mechanism operates is not decidable from these columns.
- I3 The 99.4 % operator overlap and the 1,563 ranking rows of the 0.543-fill operator mean that if the fill process is stationary from 2025 to 2026, the ranking set contains on the order of 4.7–4.9 k filled LIRF matched rows, of which roughly 1.1 k would be filled∧far rows of the kind that had OOF RMSE 1,106 (either 0.084 × 13,815 ranking far rows ≈ 1,160, or the training filled∧far fraction 6,453 / 159,216 = 4.05 % × 26,516 ≈ 1,074). This is an extrapolation, not a measurement.
