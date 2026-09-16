# Item 12 `rule-overlays-extreme-rows` — verification result (2026-09-16)

Script: `docs/investigation_2026-09-16/scripts/rule-overlays-extreme-rows.py` (no training, `POLARS_MAX_THREADS=1`,
lazy scans of 14 columns from `data/work/feat_train.parquet` / `feat_rank.parquet`, plus
`experiments/oof_main-48c19bc830.parquet`, `outputs/main.parquet` (v7), `outputs/probe_unmatched_from_v4.parquet` (v9)).
Every number below was re-derived by this script; none is copied from the hypothesis reports. Definitions follow
plan §0: `y = TAXITIME_SEC_mvt`; `pa = MVT − AOBT_3` (= feature `mvt_minus_aobt3`); `pl = MVT − LOBT`
(= `mvt_minus_lobt`); `ms = MVT − SCHED` (= `mvt_minus_sched`); unmatched = routing definition (`mvt_minus_aobt3`
null). 2 h = 7,200 s, 4 h = 14,400 s, 12 h = 43,200 s.

## 1. Hypothesis as stated (plan §1 item 12, source E H5)

Rule overlays used by public teams for extreme rows transfer to this pipeline:
(Z) zestful-fountain / Phoenix-Ops v11: `AOBT_3 − LOBT > 7200` and both proxies in `[−7200, 172800]` → predict
`max(0, MVT − LOBT)` (reported −0.2 LB on 21–31 rows);
(O) OpenAir Stage 5: OPDI ground time ≥ 7,200 s → shrink 50 % toward median (~146 rows, 281.72 → 278.36);
(S) sergiuv11: "LIRF overlay at 14,400 s" (no formula given in the source).
Plan flag: "Unverified: everything locally."

## 2. Procedure actually executed

Local reading of each rule (the source text is a one-line summary; where it is ambiguous the reading is stated
and, where possible, several readings are tested):

| id | trigger | prediction on trigger rows | needs |
|---|---|---|---|
| Z | `AOBT_3 − LOBT > 7200` ∧ `pa ∈ [−7200, 172800]` ∧ `pl ∈ [−7200, 172800]` ("both proxies" read as the two NM clock proxies MVT − AOBT_3 and MVT − LOBT) | `max(0, MVT − LOBT)` | AOBT_3, LOBT |
| Z-gap | `AOBT_3 − LOBT > 7200` only (sensitivity) | same | AOBT_3, LOBT |
| O-a / O-l / O-s | `pa ≥ 7200` / `pl ≥ 7200` / `ms ≥ 7200` (local analogues of "ground time"; OPDI is not in the repository — `grep -ri opdi` hits only the two hypothesis documents) | `0.5·pred + 0.5·median`, median = per-airport median of y on the training months not in the OOF file (2–6, 8–10); global-median variant also run | AOBT_3 / LOBT / SCHED |
| O-osU / O-osM | O-s restricted to unmatched / matched rows | as O | SCHED |
| S-cap | every LIRF row | `min(pred, 14400)` | — |
| S-ms | LIRF ∧ `ms ≥ 14400` | `ms` | SCHED |
| S-msU | LIRF unmatched ∧ `ms ≥ 14400` | `ms` | SCHED |
| S-floor | LIRF ∧ `ms ≥ 14400` | `max(pred, 14400)` | — |

Steps: (A) identity checks and null structure of AOBT_3 / LOBT / SCHED in training and ranking; (B) trigger counts on
the 2,085,047 training rows by ADEP × matched; (C) medians for the shrink rule; (D) on the 672,428 OOF-scored rows
(months 1, 7, 11, 12; total RMSE 334.5821; unmatched 8,378 rows 2,296.06; matched 664,050 rows 216.436): per trigger the
row count, RMSE of (a) the OOF prediction, (b) the rule prediction, (c) `pa`, (d) `ms`, the counterfactual total /
unmatched / matched OOF RMSE with the rule applied, and improved/worsened row counts; (E) on the 344,841 ranking rows:
trigger counts by ADEP × matched × month and what v7 / v9 predict there vs the rule; (F) decomposition of the one
improving reading (S-msU) by row category and month, and a threshold sweep 7,200 / 14,400 / 43,200.

## 3. Results

### A. Column availability (decides which rules can fire where)

| set | rows | AOBT_3 null | LOBT null | LOBT null ∧ unmatched | LOBT null ∧ matched | SCHED / `ms` null |
|---|---|---|---|---|---|---|
| training | 2,085,047 | 22,470 | 22,470 | 22,470 | 0 | 0 |
| ranking | 344,841 | 5,290 | 5,290 | 5,290 | 0 | 0 |

LOBT is null on exactly the unmatched rows (both sets). Consequently rules Z, Z-gap, O-a and O-l can fire only on
matched (model-A) rows; only the `ms`-based readings can touch model-B rows. Identities: `y == MVT − BLOCK` on
2,085,047 / 2,085,047; feature `mvt_minus_aobt3` / `mvt_minus_lobt` equal the recomputed timestamp differences on all
2,062,577 non-null rows (ranking 339,551 / 339,551); `mvt_minus_sched` on all rows. `AOBT_3 − LOBT` on matched
training rows: quantiles q0.001 −2,280, q0.1 −420, median 240, q0.9 1,200, q0.99 2,760, q0.999 3,840, max 20,421.

### B. Trigger counts, training (2,085,047 rows)

| trigger | n | unmatched | matched | y median | n y ≥ 2 h | n y ≥ 4 h | by ADEP |
|---|---|---|---|---|---|---|---|
| Z | 89 | 0 | 89 | 7,540 | 47 | 2 | LFPG 33, EGLL 27, LIRF 8, EDDF 5, LSZH/EHAM/LEMD/LTFM 3, EDDM/LEBL 2 |
| Z-gap | 111 | 0 | 111 | — | — | — | range condition removes 22 rows, all by the `pa` lower bound (−7,200); the `pl` bound never binds |
| O-a | 19 | 0 | 19 | 2,034 | 6 | 1 | LTFM 9 (y med 1,669), LIRF 6 (y med 12,811), EHAM/EDDF/EDDM/EGLL 1 |
| O-l | 144 | 0 | 144 | 7,427.5 | 80 | 3 | LFPG 49, EGLL 47, LIRF 29, LTFM 8, EDDF 4, EHAM 4, LSZH 2, EDDM 1 |
| O-s | 39,852 | 4,949 | 34,903 | 1,024 | 552 | 116 | matched LTFM 5,748 … LSZH 1,114 (y med 769–1,441, y ≥ 2 h ≤ 1.5 %); unmatched LFPG 990, EHAM 705, **LIRF 627 (y med 8,518, 65.9 % ≥ 2 h)**, LEBL 581 … |
| S-ms | 510 | 127 | 383 | 1,260 | 114 | 113 | LIRF matched 383: y med 1,086, 0.5 % ≥ 2 h; LIRF unmatched 127: y med 20,887, 88.2 % ≥ 2 h |

Rule Z on all 89 training trigger rows: RMSE(rule, y) 1,631; RMSE(`pa`, y) 7,990; RMSE(`ms`, y) 4,229; rule within
10 min of y on 30.3 % of rows (`pa` 0 %, `ms` 24.7 %); median (y − rule) = −1,023 s. In the LFPG / EGLL / LIRF
cells (68 of 89 rows) `pa` median is 601–1,079 s while y median is 7,954–9,636 s and rule ≈ `pl` ≈ `ms` ≈ 9,000–9,600 s;
in the other seven airports (21 rows) `pa` median is −4,779 to −6,570 s, y median 635–1,320 s and the rule
(1,293–3,792 s) overshoots.

### C. Medians used for the shrink rule (training months 2–6, 8–10)

Per-airport y median: EDDF 834, EDDM 772, EGLL 1,319, EHAM 746, LEBL 918, LEMD 990, LFPG 952, LIRF 1,027, LSZH 710,
LTFM 962; global 910. (Unmatched-only LIRF median for reference: 4,799.)

### D. OOF `main-48c19bc830` (672,428 rows; baseline total 334.5821, unmatched 2,296.06, matched 216.436)

| rule | n trig (unm / mat) | y med | (a) OOF | (b) rule | (c) `pa` | (d) `ms` | total after (Δ) | unmatched after | matched after | improved / worsened |
|---|---|---|---|---|---|---|---|---|---|---|
| Z | 44 (0 / 44) | 7,231 | 1,512.1 | 1,614.8 | 7,781.5 | 4,316.4 | 334.6135 (**+0.031**) | 2,296.06 | 216.485 | 19 / 25 |
| Z-gap | 58 (0 / 58) | 4,469.5 | 1,404.5 | 1,514.4 | 8,531.0 | 5,387.8 | 334.6235 (+0.041) | 2,296.06 | 216.500 | 21 / 30 (7 unchanged) |
| O-a, airport med | 5 (0 / 5) | 2,106 | 2,982.2 | 20,026.7 | 4,817.8 | 3,863.0 | 338.9119 (+4.33) | 2,296.06 | 223.153 | 3 / 2 |
| O-a, global med | 5 | | 2,982.2 | 20,043.6 | | | 338.9193 (+4.34) | | 223.164 | 3 / 2 |
| O-l, airport med | 75 (0 / 75) | 7,195 | 1,551.8 | 6,233.3 | 6,182.1 | 2,459.5 | 340.6028 (+6.02) | 2,296.06 | 225.745 | 8 / 67 |
| O-l, global med | 75 | | 1,551.8 | 6,288.2 | | | 340.7152 (+6.13) | | 225.917 | 8 / 67 |
| O-s, airport med | 15,375 (2,076 / 13,299) | 1,044 | 1,467.7 | 1,984.7 | 929.5 [n=13,299] | 13,991.6 | 390.8400 (+56.26) | 2,852.34 | 228.117 | 6,737 / 8,638 |
| O-s, global med | 15,375 | | 1,467.7 | 1,991.1 | | | 391.5858 (+57.00) | 2,856.60 | 228.739 | 6,559 / 8,816 |
| O-osU (unmatched only) | 2,076 (2,076 / 0) | 1,151 | 3,865.0 | 5,147.4 | — | 11,706.0 | 384.2233 (+49.64) | 2,852.34 | 216.436 | 1,073 / 1,003 |
| O-osM (matched only) | 13,299 (0 / 13,299) | 1,026 | 398.2 | 646.4 | 929.5 | 14,315.5 | 342.1601 (+7.58) | 2,296.06 | 228.117 | 5,664 / 7,635 |
| S-cap | 49,924 LIRF (537 / 49,387) | 1,018 | 820.7 | 1,217.4 | 561.6 [n=49,387] | 2,037.9 | 414.6928 (+80.11) | 3,075.88 | 234.040 | 8 / 36 (49,880 unchanged) |
| S-ms | 206 (55 / 151) | 1,329 | 10,567.3 | 16,031.6 | 943.3 [n=151] | 16,031.6 | 395.5656 (+60.98) | 1,752.57 | 345.969 | 52 / 154 |
| **S-msU** | 55 (55 / 0) | 23,472 | 20,419.8 | 9,043.5 | — | 9,043.5 | **290.7393 (−43.84)** | **1,752.57** | 216.436 | 51 / 4 |
| S-floor | 206 (55 / 151) | 1,329 | 10,567.3 | 15,369.9 | 943.3 | 16,031.6 | 387.4355 (+52.85) | 2,297.05 | 292.283 | 12 / 153 (41 unchanged) |

Rule Z detail (44 OOF rows: EGLL 16, LFPG 12, LIRF 4, EDDF 3, LEBL/LSZH/EDDM 2, EHAM/LTFM/LEMD 1): SSE share of
total 0.134 %. On the 29 rows with `pa` in 546–1,743 s (EGLL/LFPG/LIRF/LSZH pattern), y is 5,997–15,059, the OOF
prediction is mostly below y (err −3,831 … +2,180) and the rule is mostly above y (err −2,403 … +3,485; 22 of 29 within
±1,500). On the 15 rows with `pa` in −7,199 … −3,842 (BLOCK-side gap, item 5), y is 258–2,942, OOF errors are
−2,341 … +2,180 and the rule adds 455–3,279 s of positive error on 13 of 15. The two effects net to +0.03 s total.

O-a trigger (5 OOF rows) coincides with item 5's "matched proxy ≥ 2 h in OOF" set (n=5, RMSE 2,982.2); the shrink
halves the LIRF rollover row's prediction (y 87,002, pred 84,798 → 42,913) and drives the rule RMSE to 20,027.

### E. Ranking set (344,841 rows; v7 = `outputs/main.parquet`, v9 = `outputs/probe_unmatched_from_v4.parquet`)

| rule | n trig (unm / mat) | by ADEP × month | v7 med [min, max] | v9 med [min, max] | rule med [min, max] | RMS(rule − v7) / (rule − v9) | Σ(rule − v7)² | pred ≥ 2 h: v7 / v9 / rule |
|---|---|---|---|---|---|---|---|---|
| Z | **21** (0 / 21) | EGLL Jul 6 + Jan 1, LIRF Jul 4 + Jan 2, LFPG Jul 3 + Jan 2, LTFM/EHAM Jan 1, EDDF Jul 1 | 7,157 [0, 10,305] | 7,287 [0, 10,109] | 8,578 [3,242, 13,414] | 2,726 / 2,743 | 1.560e8 | 10 / 11 / 14 |
| Z-gap | 27 (0 / 27) | as above + 6 | 4,470 [0, 10,305] | 4,527 | 8,037 [0, 13,414] | 2,443 / 2,458 | 1.612e8 | 10 / 11 / 14 |
| O-a | 5 (0 / 5) | LFPG 2, LIRF 1, LTFM 1, EDDM 1 | 3,924 [3,298, 84,866] | 3,925 [3,181, 84,814] | 2,438 [2,054, 42,946.5] | 18,946 / 18,940 | 1.795e9 | 2 / 2 / 1 |
| O-l | 94 (0 / 94) | 11 cells | 5,610.5 [3,844, 84,866] | 5,630.5 | 3,201 [2,398, 42,946.5] | 5,118 / 5,124 | 2.463e9 | 21 / 22 / 1 |
| O-s | 10,180 (1,721 / 8,459) | all airports | 1,153.5 [0, 84,866] | 1,159 [0, 108,552] | 1,042 [355, 42,946.5] | 1,204 / 1,997 | 1.477e10 | 113 / 103 / 38 |
| O-osU | 1,721 (1,721 / 0) | all airports | 1,472 [0, 66,693] | 1,578 [8, 108,552] | 1,166 [355, 33,860] | 2,512 / 4,619 | 1.086e10 | 92 / 81 / 37 |
| S-cap | 26,899 LIRF (383 / 26,516) | | 1,064 | 1,068 | 1,064 [0, 14,400] | 478 / 888 | 1.844e9 | 98 / 90 / 98 |
| S-ms | 138 (43 / 95) | LIRF | 1,396 [0, 84,866] | 1,415.5 [0, 108,552] | 18,476 [14,465, 111,654] | 17,966 / 15,816 | 4.455e10 | 42 / 42 / 138 |
| S-msU | 43 (43 / 0) | LIRF Jul 28, Jan 15 | 30,701 [5,030, 66,693] | 37,517 [4,319, 108,552] | 29,455 [14,945, 111,654] | 16,937 / 7,419 | 1.234e10 | 41 / 41 / 43 |

Rule Z, 21 ranking rows: 14 have `pa` 594–1,496 s and `pl` = rule 8,037–13,414 s (EGLL 7 rows all `pa` 594–602; LIRF 5;
LFPG 2 with `pl` 9,010 / 10,197 but v7 5,394 / 3,867; EHAM 1 with `pa` 5,425, `pl` 13,414, v7 10,305); 7 have `pa`
−3,302 … −5,402 and rule 3,242–5,039 vs v7 0–4,470. 10 of the 21 are in item 5's list of 22 matched rows predicted
≥ 2 h by v7. Arithmetic bound on the LB change from moving these 21 predictions (labels unknown): with Σ(rule − v7)² =
1.560e8 over 344,841 rows, the RMSE change lies in [−0.74, +0.74] s at LB 305.03 and [−0.66, +0.66] s at 342.59, the
extremes corresponding to the rule being exactly right / v7 exactly right on every trigger row.

S-msU, 43 ranking rows (the plan's "43 LIRF unmatched ms ≥ 4 h"): v7 predicts 5,030–66,693 (median 30,701), v9 (= v4
on unmatched rows) 4,319–108,552 (median 37,517); the rule (= `ms`) 14,945–111,654. On the 11 rows with `ms` ≥ 12 h,
v9 is within ±10 % of `ms` on 9 (e.g. 108,552 vs 111,654; 79,686 vs 87,484), v7 predicts 30,644–39,636 on 9 of them and
66,037 / 66,693 on the two rows with `ms` 61,322 / 47,337 (16R rows in item 3). RMS(rule − v9) 7,419 vs RMS(rule − v7)
16,937: v9/v4 already sits far closer to "predict `ms`" on these rows than v7 does.

### F. S-msU decomposition (LIRF unmatched OOF rows: 537, SSE 2.768e10 = 36.77 % of total)

Rows by category among the trigger set (`y ≈ ms` = |y − ms| ≤ 1,800; rollover = y > 72,000):

| T | n trig | filled: n, SSE before → after | rollover: n, SSE before → after | other: n, SSE before → after | total OOF RMSE after | unmatched after |
|---|---|---|---|---|---|---|
| 7,200 | 239 | 131: 6.946e9 → 2.35e6 (y med 10,560) | 7: 1.632e10 → 3.494e9 (y med 87,361, ms med 60,599, oof med 39,292) | 101: 2.896e9 → 8.514e9 (y med 1,077, ms med 9,535) | 301.487 | 1,892.7 |
| 14,400 | 55 | 44: 6.140e9 → 2.26e6 (y med 21,872) | 7: as above | 4: 4.71e8 → 1.002e9 (y 723–1,377, ms 15,178–20,878) | 290.739 | 1,752.6 |
| 43,200 | 15 | 8: 4.082e9 → 2.26e6 (y med 47,068) | 7: as above | 0 | 294.617 | 1,803.8 |

At T = 14,400 the SSE reduction is 1.843e10, of which 1.283e10 (69.6 %) comes from the 7 rollover rows (where `ms` =
56,520–93,535 is closer to y ≈ 87,200–88,100 than the OOF predictions 32,352–51,678) and 6.137e9 (33.3 %) from the 44
filled rows, offset by +5.32e8 on the 4 false-alarm rows. Removing the rollover contribution by arithmetic
(7.5275e10 − 6.137e9 + 5.32e8 over 672,428 rows) leaves a total RMSE of ≈ 321.9 (−12.7). The T = 7,200 variant reproduces
item 1's "pred := ms on all LIRF unmatched ms ≥ 2 h rows → unmatched 1,892.7" exactly. Per month (T = 14,400): Jan 6
rows, 372.97 → 349.70; Jul 39 rows, 378.42 → 318.36; Nov 4 rows, 291.46 → 241.27; Dec 6 rows, 277.88 → 236.47. The
same `pred := ms` applied to matched LIRF rows with `ms` ≥ 14,400 (151 OOF rows, y median 1,086) raises matched RMSE
216.4 → 346.0 (row S-ms), i.e. the reading is only viable with the unmatched restriction.

## 4. Verdict per sub-claim

| # | sub-claim | verdict | deciding observation |
|---|---|---|---|
| 1 | Rule Z is applicable in this pipeline (columns available) | CONFIRMED for matched rows, REFUTED for unmatched rows | AOBT_3 and LOBT are null on exactly the 22,470 / 5,290 unmatched rows (A); the rule can never fire on a model-B row. |
| 2 | Rule Z selects "21–31 rows" on the ranking set | CONFIRMED | 21 rows under the full trigger, 27 under the gap-only trigger (E). Training: 89 / 111. |
| 3 | Rule Z improves RMSE here | REFUTED on OOF (no evidence of transfer) | 44 OOF rows: rule RMSE 1,614.8 vs OOF 1,512.1; total 334.5821 → 334.6135 (+0.031); 19 improved / 25 worsened (D). The rule beats the OOF model on the 29 `pa` ≈ 600–1,700 s rows (matched BLOCK-before-AOBT_3 gap rows) and loses on the 15 `pa` < −3,800 rows. On the ranking set the effect is bounded to ±0.74 s (E). |
| 4 | Rule Z's prediction is better than the proxies on its trigger rows | CONFIRMED vs `pa` and `ms` | Training 89 rows: rule 1,631 vs `pa` 7,990 vs `ms` 4,229; OOF 44 rows: 1,614.8 vs 7,781.5 vs 4,316.4. It is not better than the existing model (sub-claim 3). |
| 5 | OpenAir rule (OPDI ground ≥ 7,200 → 50 % toward median) transfers | UNDETERMINED as stated; REFUTED for every local analogue | OPDI data absent (grep). Analogues `pa` / `pl` / `ms` ≥ 7,200 with airport or global median all worsen total OOF RMSE: +4.33 / +6.02 / +56.26 (airport median), +4.34 / +6.13 / +57.00 (global); unmatched-only `ms` variant +49.64, matched-only +7.58 (D). Trigger sizes (OOF 5 / 75 / 15,375; ranking 5 / 94 / 10,180) do not match "~146 rows", so none of the analogues is the OpenAir trigger. |
| 6 | sergiuv11 "LIRF overlay at 14,400 s" transfers | UNDETERMINED (formula not in source); under the four readings tested: three REFUTED, one improves | S-cap +80.11, S-ms +60.98, S-floor +52.85; S-msU −43.84 (total 334.58 → 290.74; unmatched 2,296 → 1,753; 51 of 55 rows improved) (D). |
| 7 | The S-msU gain is a tail-row property | CONFIRMED, with 69.6 % of the SSE reduction on 7 rollover rows | F: rollover 1.283e10 of 1.843e10; filled 44 rows 6.137e9; 4 false alarms +5.32e8. Without the rollover rows ≈ −12.7 s. Per-month gains 23–60 s (F). |
| 8 | The rules target "extreme rows" | CONFIRMED for Z / O-l / S-ms in the sense of `ms` ≥ 2 h rows; REFUTED in the sense of model-B tail rows for Z / O-a / O-l | Z / O-a / O-l trigger only matched rows (B); their training y medians are 7,540 / 2,034 / 7,427 with 2 / 1 / 3 rows ≥ 4 h. The 119 unmatched training rows with y ≥ 4 h (items 1–3) are reachable only through `ms`. |
| 9 | Plan's external figures (−0.2 LB, 281.72 → 278.36, 21–31 rows) | UNDETERMINED except the row count (CONFIRMED, 21) | LB effects need labels; the OpenAir stage needs OPDI. |

## 5. What cannot be determined locally

- The OpenAir trigger itself: OPDI ground time is not in `feat_train` / `feat_rank` / `data/` (only mentioned in
  `hypotheses/external.md` and `plan.md`); the analogues tested are the three local "ground time" clocks, none of which
  selects ~146 ranking rows.
- The exact sergiuv11 rule: external.md gives only "LIRF overlay at 14,400 s"; the repository METHOD files of the three
  public teams are not in this repo, so the readings in §2 are this script's, not theirs. Likewise whether Phoenix-Ops'
  "both proxies" means MVT − AOBT_3 and MVT − LOBT (the reading used) — the `pl` bound never binds and the `pa` bound
  removes 22 of 111 training / 6 of 27 ranking gap rows, so the count 21 depends on the `pa` reading.
- LB effect of any rule: ranking labels unknown. For Z the arithmetic bound is ±0.74 s (E); for S-msU the 43 ranking
  rows would move v7 by RMS 16,937 (Σ 1.234e10, i.e. up to ±~57 s of LB RMSE at 305 if the rule were entirely right or
  entirely wrong), and v9/v4 already predicts close to `ms` on 9 of the 11 `ms` ≥ 12 h rows.
- How many of the 43 ranking S-msU rows are filled / rollover / ordinary (BLOCK_TIME null on all ranking rows; item 1 §5).

## 6. Facts vs inference

**Facts (measured in this run):** every count, median and RMSE in §3; LOBT null ⇔ AOBT_3 null ⇔ unmatched on both
sets; rule Z trigger 89 / 44 / 21 rows (training / OOF / ranking); the OOF deltas +0.031 (Z), +4.33 … +57.00 (O
analogues), +52.85 … +80.11 (S-cap / S-ms / S-floor), −43.84 (S-msU); the S-msU decomposition (7 rollover rows =
69.6 % of the gain); the reproduction of item 1's 1,892.7 at T = 7,200; the 21 Z ranking rows and their v7 / v9 / rule
values; that 10 of the 21 are in item 5's 22-row list.

**Inference (not established here):** that the local readings of rules Z, O and S match what the public teams
implemented; that the ranking LIRF unmatched rows with `ms` ≥ 14,400 contain filled / rollover rows in the training
proportions (the S-msU gain on the ranking set depends entirely on this); the ±0.74 s / ±57 s LB bounds (arithmetic
on prediction changes, assuming a hypothetical LB RMSE); the sentence "v9/v4 already sits closer to predict-`ms`"
being the reason for v4's better LB (only the prediction geometry is measured, not the labels); that the Z rule's gain
on `pa` ≈ 600 s rows and loss on `pa` < −3,800 rows reflects item 5's two distinct gap mechanisms (the sign split is
measured, the mechanism is item 5's inference).
