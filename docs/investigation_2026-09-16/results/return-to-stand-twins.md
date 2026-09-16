# Item 6 `return-to-stand-twins` — verification result (2026-09-16)

Script: `docs/investigation_2026-09-16/scripts/return-to-stand-twins.py` (no training; `POLARS_MAX_THREADS=1`; ~4 s wall).
Inputs: `data/raw/training_*.parquet` and `data/raw/ranking.parquet` (DEP rows, 14 columns selected),
`experiments/oof_main-48c19bc830.parquet`, `outputs/main.parquet` (+ the other 12 `outputs/*.parquet` for one table).
All numbers below were re-derived by this script; none are copied from D's report.

## 1. Hypothesis as stated (plan.md §1 item 6, source D)

Departures recorded twice with the same (ADEP, FLIGHT, SCHED) and take-off within 2 min; the matched twin carries the
earlier pushback's BLOCK, so its label is inflated. Verified by D: 223 pairs in training (450 rows), 129 mixed;
take-offs within 120 s in 68–89 %; matched twin y median 3,170–3,834 vs unmatched 966. Ranking 84 rows in such groups
(LFPG 38, LTFM 28, EDDF 10, LSZH 8). No MVT_ID overlap ranking/training; 4 FLIGHT_ID overlaps. Upper bound LB effect
≈ 0.8 % of SSE. Unverified: actual LB effect.

## 2. Procedure executed

1. Training DEP rows (2,085,047; 2,084,859 with non-null `FLIGHT_mvt`). Group by `(ADEP_mvt, FLIGHT_mvt, SCHED_TIME_UTC_mvt)`;
   tabulate group sizes; collect all rows in groups of size ≥ 2; sort by `MVT_TIME_UTC_mvt` within group.
2. For size-2 groups ("pairs"): classify by (unmatched?, unmatched?) in MVT order, using both definitions of unmatched
   (§0: `FLIGHT_ID_mvt` null = D; `AOBT_3_flt` null = routing). Compute `dt_mvt = MVT2 − MVT1`, `dt_blk = BLOCK2 − BLOCK1`,
   label medians, stand/runway equality.
3. For mixed pairs (one matched, one unmatched): orient as (matched twin m, unmatched twin u); test the mechanism
   `BLOCK_m < BLOCK_u`, `y_m > y_u`, the identity `y_m − y_u = (BLOCK_u − BLOCK_m) − (MVT_u − MVT_m)`, and where the
   matched twin's own `AOBT_3` sits relative to the two BLOCK times; proxy `MVT_m − AOBT_3_m`.
4. Twin rows vs the training label tail (y ≥ 2 h, ≥ 4 h).
5. Join twin rows to `oof_main-48c19bc830` (672,428 non-null OOF rows, RMSE 334.58 re-derived): SSE share, RMSE, residuals
   per class.
6. Ranking DEP rows (344,841): same grouping; join `outputs/main.parquet` on `MVT_ID_mvt`; per-row list of all 84 rows;
   per-pair prediction comparison; medians for all 13 output files.
7. ID overlap ranking vs training: MVT_ID; FLIGHT_ID on DEP rows only and (as D did) on all phases.
8. Upper-bound arithmetic against `SSE_LB = 305.03² × 344,841 = 32,085,144,926` under five stated assumptions.

## 3. Results

### 3.1 Training group structure

| group size | groups | rows |
|---|---|---|
| 1 | 2,084,409 | 2,084,409 |
| 2 | 223 | 446 |
| 4 | 1 (EDDM BPO721 2025-07-18, 4 unmatched rows, MVT within 3 s, stands 611–614, y 361–366) | 4 |

Rows in groups of size > 1: **450** (223 pairs + one quadruple). `unm_fid` and `unm_route` agree on all 450 twin rows.
By airport: LFPG 232 (unmatched rate 0.466, y median 1,563.5), LTFM 130 (0.015, 950.5), EDDF 70 (0.314, 679.5),
EDDM 16 (0.312, 547), LSZH 2 (1.0, 848.5).

### 3.2 The 223 pairs by kind (first = earlier MVT)

| kind | n | dt_mvt ≤ 120 s | dt_mvt median (s) | y1 med | y2 med | y1 > y2 | dt_blk med (s) | same stand | same rwy |
|---|---|---|---|---|---|---|---|---|---|
| mat,mat | 91 | 5.5 % | 13,554 | 908 | 926 | 48.4 % | 13,327 | 8.8 % | 83.5 % |
| mat,unm | 47 | **89.4 %** | 16 | **3,834** | 964 | 93.6 % | 3,954 | 35.6 % | 85.1 % |
| unm,mat | 82 | **68.3 %** | 44 | 984.5 | **3,170** | 20.7 % | −2,339.5 | 42.7 % | 86.6 % |
| unm,unm | 3 | 0 % | 8,462 | 6 | 1,691 | 0 % | 5,159 | 66.7 % | 66.7 % |

Mixed pairs = 47 + 82 = **129**. MVT_IDs are distinct within every pair. No pair has `dt_mvt == 0`.
dt_mvt quantiles over all 223 pairs (0/.1/.25/.5/.75/.9/1): 1 / 9 / 23 / 3,900 / 15,001 / 26,101 / 115,148 s.

Kind × airport (n, of which dt_mvt ≤ 120 s): LFPG mat,mat 10 (5), mat,unm 46 (42), unm,mat 58 (56), unm,unm 2 (0);
LTFM mat,mat 63 (0), unm,mat 2 (0); EDDF mat,mat 13 (0), unm,mat 22 (0); EDDM mat,mat 5 (0), mat,unm 1 (0); LSZH unm,unm 1 (0).
**Every pair with take-off within 120 s is at LFPG (103 pairs: 98 mixed + 5 mat,mat).** Non-LFPG pairs are hours apart
(median dt 2.9–8.4 h): same flight number and SCHED, but different take-offs.

### 3.3 Mechanism on mixed pairs

All 129 mixed pairs: `BLOCK_m < BLOCK_u` in 104; `y_m > y_u` in 109; y_m median 3,550 (p10/p90 614 / 8,067), y_u median 966
(516 / 1,556); y_m ≥ 2 h in 26, y_u ≥ 2 h in 0. Matched twin is the earlier MVT row in 47, the later in 82.

**Same-take-off mixed pairs (|dt_mvt| ≤ 120 s): n = 98, all LFPG.**
- `BLOCK_m < BLOCK_u`: **98 / 98**. `y_m > y_u`: **98 / 98**. y_m median **4,742.5**, y_u median **1,083**.
- Identity `y_m − y_u = (BLOCK_u − BLOCK_m) − (MVT_u − MVT_m)`: max |difference| = **0.0** on 98 / 98 (labels are exactly
  MVT − BLOCK on both rows; the inflation is exactly the gap between the two BLOCK times).
- `y_m − y_u` quantiles .1/.5/.9: 1,257 / 3,857 / 7,036 s. `BLOCK_u − BLOCK_m` (all mixed) .1/.5/.9: −9,773 / 2,880 / 7,020.
- The matched twin's own `AOBT_3`: `BLOCK_m − AOBT_3_m` .1/.5/.9 = −7,319 / −2,585 / +180 (22 / 129 within 300 s);
  `BLOCK_u − AOBT_3_m` = −9,725 / −302 / +539 (48 / 129 within 300 s). I.e. NM's AOBT_3 sits near the *later* (unmatched
  twin's) BLOCK, not near the matched twin's own BLOCK.
- proxy `MVT_m − AOBT_3_m` on the 98: .1/.5/.9 = 638 / 885 / 1,327 s (a normal taxi); `y_m − proxy_m` = 1,200 / 3,896 / 7,442.
- Examples (largest y_m): LFPG DAL265 2025-07-09 y_m 15,059 vs y_u 895 (BLOCK_m 13:09:59, BLOCK_u 17:06:02, AOBT_3 17:10:00,
  MVT 17:20:58); CAO1026 12,578 / 1,745; DAL153 11,737 / 1,806; AFR78KN 10,148 / 1,629.

### 3.4 Twin rows vs the training label tail

Training y ≥ 2 h: 584 rows; y ≥ 4 h: 122. Twin rows: y ≥ 2 h **28** (all 28 are matched twins), y ≥ 4 h **1** (max 15,059).
Twin rows y quantiles .5/.9/.99: 1,075 / 5,635 / 10,148. Unmatched twin rows (139): y median 964, none ≥ 2 h.

### 3.5 OOF error carried by twin rows (`oof_main-48c19bc830`, 672,428 rows, SSE 7.5275e10, RMSE 334.58)

| class | n with OOF | RMSE | SSE share | y med | oof med | resid med |
|---|---|---|---|---|---|---|
| all twin rows | 166 | 1,289.1 | **0.366 %** | | | |
| k=2 matched | 115 | 1,365.1 | 0.285 % | 1,082 | 1,152 | +1.6 |
| k=2 unmatched | 47 | 1,140.0 | 0.081 % | 1,028 | 1,096 | +60 |
| k=4 unmatched | 4 | 336.9 | 0.001 % | 362 | 274 | −90 |
| matched twin of same-take-off mixed pair | 35 | **2,358.2** | **0.259 %** | 4,564 | 2,260 | **−1,544** |
| unmatched twin of same-take-off mixed pair | 35 | 416.6 | 0.008 % | | | +26.5 |

On the 35 matched same-take-off twins: oof .1/.5/.9 = 1,029 / 2,260 / 7,560 vs y 2,267 / 4,564 / 8,563; oof ≥ 2,600 in 14 / 35;
resid .1/.5/.9 = −3,922 / −1,544 / +148. (Context: all OOF rows with y ≥ 2 h carry 46.01 % of OOF SSE.)

### 3.6 Ranking set

Ranking DEP rows 344,841; rows in (ADEP, FLIGHT, SCHED) groups with k ≥ 2: **84** in **41** groups (39 pairs, 2 triples).
By airport: **LFPG 38, LTFM 28, EDDF 10, LSZH 8** (unmatched rate 0.45 / 0.07 / 0.20 / 0.88). By month: Jan 46, Jul 38.
`BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` are null on all 84 (so BLOCK ordering cannot be checked in the ranking set).

Pairs by kind × airport (n, of which dt_mvt ≤ 120 s): LFPG mixed 17 (**16**), LFPG mat,mat 2 (1: KLM26E, dt 3 s);
LTFM mat,mat 12 (0), LTFM mixed 2 (0); EDDF mat,mat 3 (0), EDDF mixed 2 (0); LSZH mixed 1 (0).
Triples: LSZH SCOUT11 2026-01-17 and 2026-01-21, 3 unmatched rows each, MVT within 11 s (analogue of the EDDM quadruple).
Two LFPG pairs have identical MVT (dt = 0: ACI501, TSC251); none in training.

**Rows that match the hypothesised pattern (mixed, same take-off): 16 pairs = 32 rows, all LFPG; matched twins at risk: 16.**
The other 52 rows are pairs hours apart (46 rows) or all-unmatched triples (6 rows).

### 3.7 What `outputs/main.parquet` predicts on the 84 rows

| | n | pred median | pred mean | min | max | ≥ 3,000 | ≥ 2 h |
|---|---|---|---|---|---|---|---|
| matched (`unm_fid` false) | 56 | 1,086 | 1,703 | 430 | 6,020 | 7 | 0 |
| unmatched | 28 | 949.5 | 1,038 | 106 | 3,036 | 1 | 0 |

By airport × matched: LFPG matched 21 rows median 1,197 max 6,020; LFPG unmatched 17 median 967 max 2,366; LTFM matched 26 median 1,091.5
max 2,012; LTFM unmatched 2 median 2,024 max 3,036; EDDF matched 8 median 887; EDDF unmatched 2 median 1,379.5; LSZH matched 1 (693);
LSZH unmatched 7 median 261.

Mixed ranking pairs (22): pred_m median 1,267.5 vs pred_u median 1,021.5; pred_m > pred_u in 16 / 22.
**The 16 same-take-off mixed pairs**: pred_m sorted = 641, 939, 1,027, 1,045, 1,083, 1,167, 1,197, 1,748, 2,636, 2,934, 3,867, 5,753,
5,883, 5,946, 5,988, 6,020 (8 / 16 ≥ 2,600; sum 47,874); pred_u sorted = 613 … 1,882 (sum 16,804); pred_m > pred_u in 13 / 16.
Per-row list of all 84 rows is in the script output (section E).

Other output files on the same 84 rows (median matched / median unmatched / max): v1_baseline 1,082.5 / 949 / 5,479;
v2_two_clock 1,104.5 / 924 / 6,226; v5 1,088 / 975.5 / 6,242; v7 1,100.5 / 1,003.5 / 6,175; main 1,086 / 949.5 / 6,020;
probe_unmatched_from_v4 1,101.5 / 1,020 / 6,077. All 13 files agree to within ~60 s on the medians.

### 3.8 ID overlap ranking vs training

- MVT_ID (DEP vs DEP): **0**.
- FLIGHT_ID, DEP vs DEP: **0**. FLIGHT_ID, all phases (D's method): **4**, all four are ranking **ARR** rows on 2026-01-01
  00:19–01:38 UTC whose FLIGHT_ID belongs to a training DEP on 2025-12-31 21:34–23:07 UTC (THY2KS EGLL→LTFM, BAW492 EGLL→LEBL,
  TK1956/THY2GL EHAM→LTFM, AFR1390 LFPG→LTFM): year-boundary flights, unrelated to twins.
- Ranking twin keys that are also twin keys in training: 0. Ranking twin rows whose FLIGHT_ID appears in training: 0.

### 3.9 Upper-bound arithmetic (N = 344,841; LB v4 = 305.03; SSE_LB = 32,085,144,926)

Assumption in every row: the named rows currently carry the stated per-row error and a fix would remove it entirely,
all other rows unchanged (RMSE_new = sqrt(305.03² − n·e²/N)).

| | rows n | per-row error e (s) | n·e² | share of SSE_LB | RMSE 305.03 → | Δ |
|---|---|---|---|---|---|---|
| (a) all 84 twin rows, e = OOF RMSE on training twin rows | 84 | 1,289.1 | 1.396e8 | 0.435 % | 304.37 | 0.66 |
| (b) 16 matched same-take-off twins, e = OOF RMSE on that class | 16 | 2,358.2 | 8.90e7 | 0.277 % | 304.61 | 0.42 |
| (c) 16, e = median(y_m) − median(y_u) in training | 16 | 3,659.5 | 2.143e8 | 0.668 % | 304.01 | 1.02 |
| (d) all 84, e = training matched-twin median label (pessimistic, contradicts 3.5: unmatched twins have OOF RMSE 417) | 84 | 3,550 | 1.059e9 | 3.299 % | 299.96 | 5.07 |
| (e) 16, e = OOF residual median on that class | 16 | 1,544.4 | 3.82e7 | 0.119 % | 304.85 | 0.18 |

Under the assumptions consistent with the training evidence ((a)–(c), (e)): **0.12–0.67 % of SSE_LB, ΔRMSE 0.18–1.02**.
D's "≈ 0.8 %" is within a factor 1.2 of (c) but its exact assumption is not stated in D's report, so it is not reproducible as written.

## 4. Verdicts per sub-claim

| sub-claim | verdict | deciding observation |
|---|---|---|
| Departures recorded twice with same (ADEP, FLIGHT, SCHED) | CONFIRMED | 223 size-2 groups + 1 size-4 group = 450 training rows (3.1) |
| … with take-off within 2 min | CONFIRMED for LFPG only | 103 / 223 pairs have dt_mvt ≤ 120 s, all 103 at LFPG; 0 / 106 non-LFPG pairs (3.2). The 84-row ranking count and the 450-row training count both mix this pattern with same-flight-number pairs hours apart |
| Matched twin carries the earlier pushback's BLOCK | CONFIRMED | BLOCK_m < BLOCK_u in 98 / 98 same-take-off mixed pairs; 104 / 129 over all mixed pairs (3.3) |
| … so its label is inflated | CONFIRMED | y_m > y_u in 98 / 98; y_m median 4,742.5 vs y_u 1,083; y_m − y_u equals BLOCK_u − BLOCK_m − (MVT_u − MVT_m) exactly (3.3) |
| 223 pairs / 450 rows / 129 mixed | CONFIRMED | 3.1, 3.2 |
| take-offs within 120 s in 68–89 % | CONFIRMED | mat,unm 89.4 %, unm,mat 68.3 % (these are the two mixed kinds; mat,mat 5.5 %) |
| matched twin y median 3,170–3,834 vs unmatched 966 | CONFIRMED | 3,834 (mat,unm) / 3,170 (unm,mat); unmatched 964 / 984.5; 966 over all 129 mixed |
| Ranking 84 rows (LFPG 38, LTFM 28, EDDF 10, LSZH 8) | CONFIRMED | 3.6. Only 32 of the 84 (16 LFPG pairs) fit the same-take-off mixed pattern |
| No MVT_ID overlap | CONFIRMED | 0 |
| 4 FLIGHT_ID overlaps | CONFIRMED as a count, REFUTED as relevant | 4 = ranking ARR rows of year-boundary flights; DEP-vs-DEP overlap 0; none involve twin rows (3.8) |
| Upper bound LB effect ≈ 0.8 % of SSE | CONFIRMED in magnitude (sub-1 %) | 0.12–0.67 % under assumptions consistent with training OOF; the 0.8 % figure itself is not reproducible because its assumption is unstated (3.9) |
| Tail relevance (implicit: twins explain the extreme tail) | REFUTED | twin rows: 28 of 584 training rows y ≥ 2 h, 1 of 122 y ≥ 4 h (max 15,059); 0.366 % of OOF SSE vs 46 % for all y ≥ 2 h rows (3.4, 3.5) |
| Model already handles it | UNDETERMINED / partial | OOF on the 35 matched same-take-off twins: median pred 2,260 vs y 4,564, 14/35 ≥ 2,600, residual median −1,544; main.parquet gives 8 / 16 ranking matched twins ≥ 2,600 and 13 / 16 above their unmatched twin (3.5, 3.7). Predictions are elevated but by roughly half the training inflation |

## 5. What cannot be determined locally

- Ranking labels are hidden, so whether the 16 ranking matched twins actually carry inflated labels, and the actual LB
  effect, cannot be measured. `BLOCK_TIME_UTC_mvt` is null on all 84 ranking twin rows, so the BLOCK-ordering test (98/98 in
  training) cannot be repeated on the ranking set; only the MVT-proximity and matched/unmatched structure is observable there.
- Whether the twin mechanism at LFPG in 2026 has the same inflation distribution as in 2025 (16 ranking pairs vs 98 training pairs
  over 12 months; ranking rate 16 pairs / 2 months vs training 98 / 12 months ≈ 8.2 per month — comparable in order but not testable).
- The exact assumption behind D's "≈ 0.8 %".

## 6. Facts vs inference

**Facts (measured):** everything in §3; in particular: 450 twin rows / 223 pairs / 129 mixed; 103 same-take-off pairs all at LFPG;
98/98 same-take-off mixed pairs have BLOCK_m < BLOCK_u and y_m > y_u with the label difference equal to the BLOCK gap exactly;
AOBT_3 of the matched twin is closer to the unmatched twin's BLOCK (median −302 s) than to its own (median −2,585 s);
twin rows carry 0.366 % of OOF SSE (matched same-take-off twins 0.259 %, RMSE 2,358, residual median −1,544);
ranking: 84 rows, 16 same-take-off mixed pairs (all LFPG), main.parquet predicts 641–6,020 on the 16 matched twins
(8 ≥ 2,600), 613–1,882 on their unmatched twins; MVT_ID overlap 0; FLIGHT_ID DEP-vs-DEP overlap 0, all-phase overlap 4 (ARR rows);
arithmetic bounds 0.12–0.67 % of SSE_LB (ΔRMSE 0.18–1.02) under stated assumptions.

**Inference (not measured):** (i) that the two rows are one physical departure with a return to stand — supported by identical
flight/SCHED, MVT within seconds, the matched twin's earlier BLOCK, and NM AOBT_3 aligning with the later BLOCK, but the
operational cause is not in the data; (ii) that the non-LFPG "pairs" hours apart (LTFM 63 mat,mat pairs etc.) are a different
phenomenon (re-timed or repeated flight numbers) — inferred from dt medians of 2.9–8.4 h and no label inflation pattern
(y1 ≈ y2 medians 908 / 926); (iii) that the model's elevated predictions on some matched twins come from features such as
`mvt_minus_sched` rather than from the proxy (the proxy is a normal 885 s median for these rows) — not tested here;
(iv) that a fix would remove the full error on the 16 rows (the assumption behind every row of §3.9).
