# Result — item 1 `sched-filled-block`

Script: `docs/investigation_2026-09-16/scripts/sched-filled-block.py` (no training, `POLARS_MAX_THREADS=1`,
runtime ≈ 1 min, reads `data/work/feat_train.parquet` / `feat_rank.parquet` (12 columns),
`experiments/oof_main-48c19bc830.parquet`, `outputs/main.parquet`). All numbers below were re-derived by
this script; none are copied from the hypothesis reports.

## 1. Hypothesis as stated (plan.md §1 item 1)

At LIRF, when no real off-block time exists, `BLOCK_TIME` is filled with `SCHED_TIME`, so the label becomes
`y = MVT − SCHED` (= feature `mvt_minus_sched`); the extreme tail is this artefact, not taxiing, and is
deterministic given a feature model B already has.

## 2. Procedure actually executed

Definitions (plan §0 / §4), all computed side by side:

| symbol | definition |
|---|---|
| `unm_routing` | `mvt_minus_aobt3` null (model-B routing rule; H/P/M) |
| `unm_fid` | `FLIGHT_ID_mvt` null (D) |
| `f10` / `f6` / `f120` | \|BLOCK − SCHED\| ≤ 10 s (D) / ≤ 6 s (M) / ≤ 120 s (M) |
| `r60` / `r1800` | \|y − ms\| ≤ 60 s (D) / < 1,800 s (M) |
| `ms` | feature `mvt_minus_sched`; also recomputed as MVT − SCHED from timestamps |

Steps: (A) label identity; (B) unmatched definitions; (C) label buckets 2–4 h / 4–20 h / > 20 h;
(D) y ≥ 2 h rows, share with y ≈ ms, by matched status; (E) y ≥ 4 h unmatched rows: residual y − ms
quantiles, BLOCK == SCHED, per-airport, listing of the 17 non-filled rows; (F) fill rate per ADEP × matched;
(G) ms band 2–20 h per ADEP × matched, ms ≥ 2 h LIRF vs others, LIRF unmatched by finer ms band;
(H) BLOCK seconds-of-minute alignment; (I) LIRF unmatched profile, 2×2 fill × y ≥ 2 h, by month;
(J) OOF `main-48c19bc830` (672,428 scored rows of 2,085,047 in the file; the other 1,412,619 have null
`oof`): SSE share of filled rows under each definition, plus two oracle replacements; (K) ranking set:
BLOCK/label availability, LIRF unmatched ms ≥ 2 h count and `outputs/main.parquet` predictions;
(L) LIRF unmatched ms ≥ 2 h: label distribution of filled vs non-filled rows. Code inspection of
`src/prc/train.py` for `cols_b`.

## 3. Results

### A. Identity and algebra (2,085,047 training departures)

| check | count |
|---|---|
| `TAXITIME_SEC_mvt == MVT − BLOCK` exactly | 2,085,047 / 2,085,047 |
| null labels / null BLOCK / null SCHED | 0 / 0 / 0 |
| feature `mvt_minus_sched == MVT − SCHED` (recomputed) | 2,085,047 / 2,085,047 |
| `(y − ms) == −(BLOCK − SCHED)` exactly | 2,085,047 / 2,085,047 |

Consequence: "fill" (BLOCK ≈ SCHED at tolerance k) and "y ≈ ms" (at the same tolerance k) are the same
predicate; `f10 ⇔ |y − ms| ≤ 10`, `f6 ⇔ |y − ms| ≤ 6`. D's `fill` and D's `y_eq_ms` differ only by
threshold (10 vs 60 s).

### B. Unmatched definitions

routing 22,470; FLIGHT_ID-null 22,424; routing-only 46; fid-only 0. Ranking: 5,290 / 5,283 / 7 / 0.
Every table below was produced under both definitions; where not shown separately, the numbers were
identical to the stated precision (the 46 rows are all y < 2 h, LIRF-unrelated).

### C. Label buckets

| bucket | n | unm (routing) | unm (fid) | f10 | f6 | f120 | r1800 | BLOCK > 1 h before SCHED | LIRF share |
|---|---|---|---|---|---|---|---|---|---|
| < 2 h | 2,084,463 | 0.011 | 0.011 | 0.048 | 0.046 | 0.170 | 0.851 | 0.000 | 0.077 |
| 2–4 h | 462 | 0.784 | 0.784 | 0.710 | 0.710 | 0.727 | 0.870 | 0.054 | 0.788 |
| 4–20 h | 106 | 0.981 | 0.981 | 0.943 | 0.943 | 0.943 | 0.962 | 0.038 | 0.962 |
| > 20 h | 16 | 0.938 | 0.938 | 0.125 | 0.125 | 0.125 | 0.188 | 0.750 | 0.875 |

### D. y ≥ 2 h rows (584 total)

| group | n | r60 | r1800 | f10 | LIRF share |
|---|---|---|---|---|---|
| unmatched (either def) | 481 | 0.825 | 0.871 | 0.825 | 0.913 |
| matched (either def) | 103 | 0.340 | 0.854 | 0.320 | 0.398 |

### E. Unmatched y ≥ 4 h (n = 119 under both definitions)

- Residual y − ms quantiles (0, .1, .25, .5, .75, .9, 1): [−6,365, −5, −2, 1, 4, 20,706, 86,402].
  (M reported the 0.9 quantile as 20,922; the difference is quantile interpolation — the two nearest
  order statistics are 20,706 and 21,784.)
- Within 1,800 s: 102 / 119. Median residual 1 s. f6 = f10 = f120 = 101 (one row has 1,800 > |resid| > 120).
- LIRF 115 / 119 (all 115 AIRCRAFT_TYPE_mvt null); all 102 within-1,800 rows are LIRF; 0 non-LIRF within 1,800.
- Per airport: LIRF 115 (102 r1800, y median 24,900), LSZH 2 (0), LFPG 2 (0).
- The 17 rows not within 1,800 s: 13 rows with y in 87,168–88,392 (BLOCK date one day before MVT date;
  11 LIRF, 1 LSZH 87,341, 1 LFPG 84,240 — the LFPG one is 84,240), LFPG 58,206, LIRF 31,268, LIRF 25,018,
  LSZH 18,409. Full listing in script output section E.
- 2 h ≤ y < 4 h unmatched: 362 rows, r1800 0.8757, f10 0.8177; LIRF 324 (299 r1800); non-LIRF within 1,800:
  18 rows (EGLL 8, LTFM 6, LFPG 2, LSZH 2), of which f10 = 1 (EGLL).

### F. Fill rate by ADEP × unmatched (routing)

| ADEP | group | n | f10 | f6 | f120 | y median (f10 rows) | ≥ 2 h share (f10 rows) | y median (non-fill) | ≥ 2 h (non-fill) |
|---|---|---|---|---|---|---|---|---|---|
| LIRF | unmatched | 1,488 | 0.4845 | 0.4845 | 0.4859 | 7,559 | 0.549 | 1,070 | 0.056 |
| LIRF | matched | 159,216 | 0.1782 | 0.1782 | 0.2567 | 1,091 | 0.001 | 1,018 | 0.0001 |
| LEBL | matched | 177,843 | 0.0565 | 0.0565 | 0.1788 | 926 | 0 | | |
| LTFM | matched | 269,320 | 0.0515 | 0.0515 | 0.2022 | 961 | 0 | | |
| EDDM | matched | 166,324 | 0.0501 | 0.0501 | 0.198 | 725 | 0 | | |
| LTFM | unmatched | 3,645 | 0.0499 | 0.0499 | 0.1789 | 964 | 0 | | |
| EGLL | matched | 238,132 | 0.049 | 0.049 | 0.1831 | 1,318 | 0 | | |
| LEMD | matched | 211,295 | 0.0448 | 0.0448 | 0.1603 | 986 | 0 | | |
| LFPG | matched | 235,779 | 0.0419 | 0.0419 | 0.1563 | 909 | 0 | | |
| EHAM | unmatched | 4,084 | 0.0274 | 0.0269 | 0.0619 | 232 | 0 | | |
| LSZH | unmatched | 2,291 | 0.0227 | 0.0205 | 0.0781 | 2.5 | 0 | | |
| all other cells | | | ≤ 0.0188 | | ≤ 0.137 | | 0 (EGLL unm 0.25 on 4 rows) | | |

Under the FLIGHT_ID definition LIRF unmatched: n 1,487, f10 0.4842, f120 0.4855, y median (f10) 7,562.5,
≥ 2 h share 0.550. At non-LIRF airports, f10 rows have y median in the normal range (232–1,388) and
≥ 2 h share 0 (except EGLL unmatched: 4 f10 rows, 1 ≥ 2 h).

### G. ms bands

ms in [7,200, 72,000), unmatched (routing): LIRF 622 rows, f10 0.635, r60 0.635, f120 0.637, y median
8,489.5, y ≥ 2 h 0.656. Every other ADEP × group cell: f10 ≤ 0.011 (LIRF matched 0.011 on 2,750 rows; all
non-LIRF cells 0.000–0.002).

ms ≥ 7,200 (no upper bound), unmatched: LIRF 627 rows f10 0.6316 / f6 0.6316 / f120 0.6332 / r1800 0.6396,
y median 8,518; non-LIRF 4,322 rows (4,318 under fid) f10 0.0002 / f120 0.0005 / r1800 0.0081, y median 1,076.
Non-LIRF per airport (ms ≥ 2 h, unmatched): f10 max 0.0019 (EGLL), f120 max 0.0038 (LTFM), r1800 max
0.0677 (LTFM, 266 rows).

LIRF unmatched by ms band:

| ms band | n | f10 | r1800 | y median | y ≥ 2 h | y ≥ 20 h |
|---|---|---|---|---|---|---|
| < 0 | 16 | 0.000 | 0.688 | 606 | 0 | 0 |
| 0–1 h | 97 | 0.474 | 0.598 | 2,229 | 0.247 | 0 |
| 1–2 h | 748 | 0.373 | 0.381 | 1,316 | 0.003 | 0 |
| 2–4 h | 500 | 0.590 | 0.598 | 7,737 | 0.602 | 1 |
| 4–12 h | 98 | 0.847 | 0.847 | 18,451 | 0.847 | 0 |
| 12–20 h | 24 | 0.708 | 0.708 | 54,209 | 1.000 | 7 |
| ≥ 20 h | 5 | 0.200 | 0.400 | 87,605 | 1.000 | 5 |

### H. BLOCK seconds-of-minute (min(sec, 60 − sec) ≤ 6)

| group | n | within 6 s | BLOCK sec == 0 | SCHED sec == 0 | MVT sec == 0 |
|---|---|---|---|---|---|
| LIRF matched | 159,216 | 1.000 | 0.083 | 1.000 | 0.084 |
| LIRF unmatched | 1,488 | 1.000 | 0.092 | 1.000 | 0.081 |
| non-LIRF matched | 1,903,361 | 0.754 | 0.062 | 1.000 | 0.049 |
| non-LIRF unmatched | 20,982 | 0.741 | 0.063 | 1.000 | 0.053 |

Non-LIRF unmatched by ADEP: EDDF 0.224, EHAM 0.469, LSZH 0.251; EDDM/EGLL/LFPG/LTFM 1.000, LEBL 0.988,
LEMD 0.999. (BLOCK "within 6 s of a minute" is thus an airport-recording property, not LIRF-specific;
what is LIRF-specific is BLOCK == SCHED.)

### I. LIRF unmatched profile (routing, 1,488 rows)

AIRCRAFT_TYPE_mvt null 1,487 (LIRF matched: 0); y median 3,922; y ≥ 2 h 29.5 %; y ≥ 4 h 115.

2×2 fill × label:

| f10 | y ≥ 2 h | n | ms median | y median |
|---|---|---|---|---|
| no | no | 724 | 6,000 | 1,024 |
| no | yes | 43 | 2,882 | 8,332 |
| yes | no | 325 | 5,765 | 5,770 |
| yes | yes | 396 | 10,258 | 10,255 |

By month: n 52–337 (Jul 337); f10 0.323 (Jul) – 0.761 (Dec); y ≥ 2 h 0.217–0.475; ms ≥ 2 h 0.313–0.466.

### J. OOF `main-48c19bc830` (672,428 scored rows; total RMSE 334.58; SSE 7.5275e10)

Unmatched (routing) 8,378 rows, RMSE 2,296.1, 58.7 % of total SSE. (fid: 8,358 rows, 2,298.6.)

| subset of unmatched OOF rows | n | share of unmatched SSE | share of total SSE | RMSE |
|---|---|---|---|---|
| f6 | 350 | 0.1036 | 0.0608 | 3,616 |
| f10 | 353 | 0.1036 | 0.0608 | 3,601 |
| f120 | 632 | 0.1052 | 0.0617 | 2,711 |
| r1800 | 2,348 | 0.1933 | 0.1134 | 1,907 |
| f10 ∧ ms ≥ 2 h | 129 | 0.0922 | | 5,619 (pred/ms median 0.826) |
| ¬f10 ∧ ms ≥ 2 h | 1,947 | 0.6099 | | 3,720 |
| LIRF ∧ ms ≥ 2 h | 239 (129 f10) | 0.5924 | 0.3476 | |

Oracle replacements on unmatched OOF rows: predicting f10 rows exactly → unmatched RMSE 2,296.1 → 2,173.8
(identical to "pred := ms on f10 rows", by the identity in A); predicting r1800 rows exactly → 2,062.3;
pred := ms on all LIRF unmatched rows with ms ≥ 2 h, no flag → 1,892.7.

### K. Ranking set (344,841 departures)

BLOCK_TIME null 344,841; TAXITIME null 344,841; `mvt_minus_sched` null 0. LIRF unmatched: 383 rows
(routing and fid identical), all AIRCRAFT_TYPE null; ms ≥ 2 h 181 (Jan 52, Jul 129; ms ≥ 4 h 15 + 28 = 43;
ms ≥ 12 h 3 + 8 = 11). `outputs/main.parquet` on those 181: median 7,041, 86 predicted ≥ 2 h, pred/ms median
0.690. Training LIRF unmatched Jan + Jul 2025: 397 rows. Training f10 rate for LIRF unmatched ms ≥ 2 h: 0.6316.

### L. LIRF unmatched ms ≥ 2 h (627 rows): filled vs non-filled labels

| f10 | n | y p10 | y median | y p90 | y max | y < 1 h | y ≥ 2 h | y ≥ 20 h | ms median |
|---|---|---|---|---|---|---|---|---|---|
| yes | 396 | 7,620 | 10,255 | 28,553 | 131,167 | 0.000 | 1.000 | 1 | 10,258 |
| no | 231 | 710 | 1,090 | 3,116 | 88,392 | 0.909 | 0.074 | 12 | 9,422 |

### Code

`src/prc/train.py` lines 350–356: `cols_b` = numeric columns not prefixed by `mvt_minus_aobt3`, `aobt3`,
`lobt_minus`, `mvt_minus_lobt`, `mvt_minus_eobt1`, `mvt_minus_iobt`; `mvt_minus_sched` and
`mvt_minus_sched_slog` carry none of those prefixes and are not in `configs/main.yaml`'s excludes (the file
has no `exclude_features` key), so both are given to model B.

## 4. Per sub-claim

| sub-claim | verdict | deciding observation |
|---|---|---|
| `y = MVT − BLOCK` on every training row; `mvt_minus_sched = MVT − SCHED` | CONFIRMED | 2,085,047 / 2,085,047 exact for both (A). |
| "fill" (BLOCK ≈ SCHED) and "y ≈ ms" are the same condition | CONFIRMED (algebraic) | `y − ms = SCHED − BLOCK` exactly on all rows (A); f10 and r60 rates coincide wherever both are reported (D, G). |
| Label buckets 2–4 h 462 rows / 78 % unmatched / 71 % fill; 4–20 h 106 / 98 % / 94 %; > 20 h 16 / 12.5 % fill / 75 % BLOCK > 1 h before SCHED | CONFIRMED | Table C, identical under both unmatched definitions. |
| Of 481 unmatched y ≥ 2 h rows, 82.5 % have \|y − ms\| ≤ 60; of 103 matched, 34 % | CONFIRMED | Table D: 0.825 / 0.340. |
| y ≥ 4 h unmatched n = 119, 102 within 1,800 s, median residual 1 s, BLOCK == SCHED ±6 s on those rows, 115 LIRF | CONFIRMED | E: 119 / 102 / 1.0 / f6 = 101 of the 102 (one row has 120 < \|resid\| < 1,800) / 115. Quantile 0.9 is 20,706 not 20,922 (interpolation). |
| "None at non-LIRF except two LFPG rollovers" (M) | REFUTED as worded, CONFIRMED in substance | Non-LIRF y ≥ 4 h unmatched rows are 4, not 2: LSZH 87,341 (rollover), LFPG 84,240 (rollover), LFPG 58,206, LSZH 18,409. None of the 4 is sched-filled; the two rollovers are one LSZH and one LFPG, not two LFPG. |
| 2 h ≤ y < 4 h unmatched (n = 362): 87.6 % within 1,800 s | CONFIRMED | E: 362, 0.8757. LIRF 324 rows (299 within 1,800 s, 25 not); non-LIRF 38 rows (18 within 1,800 s, 20 not); of the 18 non-LIRF within-1,800 rows only 1 (EGLL) is f10, i.e. the non-LIRF part is ordinary long delay, not fill. |
| Fill rate LIRF unmatched 48.4 % (filled y median 7,562, 55 % ≥ 2 h); LIRF matched 17.8 %; others 0–5.7 % | CONFIRMED | F: 0.4845 (routing) / 0.4842 (fid); y median 7,559 / 7,562.5; 0.549 / 0.550; 0.1782; max other cell 0.0565 (LEBL matched). |
| 48.6 % of LIRF unmatched have \|BLOCK − SCHED\| ≤ 120 s (M) | CONFIRMED | F: f120 0.4859 (routing), 0.4855 (fid). |
| ms 2–20 h band: LIRF unmatched 622 rows, fill 64 %, y≈ms 64 %, y median 8,489; other cells ≤ 0.01 | CONFIRMED | G: 622, 0.635, 0.635, 8,489.5; max other cell 0.011 (LIRF matched). |
| Sched-filled fraction given ms ≥ 2 h: LIRF 0.633, others ≤ 0.004 | CONFIRMED (f6/f10/f120); r1800 gives ≤ 0.0081 overall, 0.0677 at LTFM | G: LIRF 0.6316 (f6, f10) / 0.6332 (f120) / 0.6396 (r1800) on 627 rows; non-LIRF 0.0002 / 0.0005 / 0.0081 on 4,322 rows. The "≤ 0.004" bound holds only for the BLOCK − SCHED definitions, not for M's 1,800 s residual definition. |
| BLOCK seconds within 6 s of a minute boundary: 100 % LIRF (both groups), 74 % other unmatched | CONFIRMED | H: 1.000 / 1.000 / 0.741 (non-LIRF matched 0.754). Per airport this is bimodal (≈1.0 at 6 airports, 0.22–0.47 at EDDF/EHAM/LSZH). |
| `mvt_minus_sched` (and `_slog`) in `cols_b`; ranking null rate 0 | CONFIRMED | Code inspection (prefix exclusion list) and K: ms null 0 of 344,841. |
| OOF main-48c19bc830: 632 sched-filled rows carry 10.5 % of unmatched SSE | CONFIRMED (under f120) | J: f120 n = 632, share 0.1052. Under f10: 353 rows, 0.1036; under r1800: 2,348 rows, 0.1933. The "632" is M's ≤ 120 s definition. |
| Ranking LIRF unmatched ms ≥ 2 h = 181; main.parquet median 7,041; 86 predicted ≥ 2 h | CONFIRMED | K: 181 / 7,041 / 86, both definitions. |
| "The extreme tail is this artefact" | CONFIRMED for the 4–20 h band; REFUTED for > 20 h | C: 4–20 h fill 0.943 (100 of 106 rows); 2–4 h 0.71; > 20 h 0.125 (2 of 16) — the > 20 h rows are the day-rollover pattern (item 2), 75 % with BLOCK > 1 h before SCHED. Of the 119 unmatched y ≥ 4 h rows, 102 are filled and 17 are not, of which 13 are rollovers (y − 86,400 in 768–1,992). |
| "deterministic given a feature model B already has" | CONFIRMED conditionally, UNDETERMINED for practical use | Given the fill flag, y = ms exactly (A). The flag itself is `BLOCK_TIME`-derived, and BLOCK_TIME is null on all 344,841 ranking rows (K), so the flag is not observable at prediction time. At LIRF unmatched ms ≥ 2 h the label is a mixture: 63.2 % y = ms (median 10,255), 36.8 % with y median 1,090 (90.9 % < 1 h) plus 12 rollover rows (L). Oracle-flag replacement lowers unmatched OOF RMSE 2,296 → 2,174; flag-free "pred := ms" on all LIRF ms ≥ 2 h rows gives 1,893 (J). |
| LIRF-specific mechanism (not present elsewhere) | CONFIRMED | G: every non-LIRF airport × group cell has f10 ≤ 0.002 in the ms 2–20 h band; F: f10 rows at non-LIRF airports have y medians 232–1,388 and ≥ 2 h share 0 (EGLL unmatched: 1 of 4). At non-LIRF airports BLOCK == SCHED (5–6 % of matched rows at LEBL/LTFM/EDDM/EGLL/LEMD/LFPG) coincides with normal labels. |

## 5. What cannot be determined locally

- Whether ranking LIRF unmatched rows are sched-filled at the training rate (0.6316 of ms ≥ 2 h rows, 0.4845
  overall): `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` are null on all 344,841 ranking departures, so the
  fill state and the labels are both unobservable. The only local evidence is the density match
  (383 ranking LIRF unmatched rows vs 397 in training Jan + Jul 2025; 181 vs 24 + 157 = 181 with ms ≥ 2 h).
- The LB effect of any change to model B's handling of these rows (needs 2026 labels).
- Whether the "filled" BLOCK_TIME is literally a copy of SCHED_TIME in the source system or a coincidence of
  two minute-rounded clocks: BLOCK second-of-minute is within 6 s of a boundary on 100 % of LIRF rows, and
  SCHED seconds are 0 on 100 % of all rows, so exact equality at ±6 s is consistent with a copy but the
  source-system behaviour is not inspectable here.

## 6. Facts vs inference

**Facts (measured above):** every number in §3; the algebraic identity `y − ms = SCHED − BLOCK`; BLOCK and
label null on all ranking departures; `mvt_minus_sched` visible to model B by the `cols_b` prefix rule; the
four non-LIRF unmatched y ≥ 4 h rows (LSZH 87,341 / LFPG 84,240 / LFPG 58,206 / LSZH 18,409); the OOF file
holds 2,085,047 rows of which 672,428 have a non-null `oof` (the plan's "672,428 rows" refers to the scored
subset).

**Inference (not measured):** that the LIRF BLOCK == SCHED rows arise from a fill in the data source rather
than from actual off-block at the scheduled minute; that the 13 rows with y ≈ 86,400 + 768…1,992 are a date
error (item 2's domain); that the ranking LIRF rows follow the training mixture proportions.

**Definitional notes for cross-item comparison:** M's "632 sched-filled rows / 10.5 %" uses ≤ 120 s; D's
"fill" uses ≤ 10 s (353 OOF rows, 10.4 %); the two agree on SSE share because the extra 279 rows at
10 < |BLOCK − SCHED| ≤ 120 s have small errors. M's "≤ 0.004 elsewhere" holds for the BLOCK − SCHED
definitions but not for the |y − ms| < 1,800 s definition (0.0081 pooled, 0.0677 at LTFM), which admits
ordinary long delays. The two unmatched definitions give identical results for every LIRF-related quantity
(the 46 / 7 routing-only rows are not LIRF tail rows).
