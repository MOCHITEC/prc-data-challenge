# Result — item 7 `false-alarm-non-lirf`

Script: `docs/investigation_2026-09-16/scripts/false-alarm-non-lirf.py` (no training, `POLARS_MAX_THREADS=1`,
runtime ≈ 1 min; reads 9 columns of `data/work/feat_train.parquet`, 6 of `feat_rank.parquet`,
`experiments/oof_main-48c19bc830.parquet`, `data/raw/submitting.parquet`, `outputs/*.parquet` and the bucket
copies `scratchpad/uploaded/v2,v3,v4.parquet`). Every number below was re-derived by this script; none is
copied from the hypothesis reports. Item 1's report (`results/sched-filled-block.md`) is relied on only for the
algebraic identity `y − ms = SCHED − BLOCK` and the LIRF fill mechanism; nothing from it is re-measured here
except where a LIRF row appears as a contrast line.

## 1. Hypothesis as stated (plan.md §1 item 7, source M)

Model B fires 15–20k predictions on large-`ms` rows at non-LIRF airports whose labels are normal
(sched-filled fraction ≈ 0); the ranking set (EHAM) contains many such rows.

Sub-claims taken from the "Verified" block: (a) non-LIRF unmatched, `ms` in (14,400, 43,200]: 368 rows,
sched-filled fraction 0.000; (43,200, ∞): 46 rows, 0.000; (b) top-25 OOF SSE rows include EDDF y=661 pred
16,880 (ms 89,185), EHAM y=549 pred 15,335 (ms 151,639), LFPG y=652 pred 15,812; (c) ranking EHAM 559 rows
ms ≥ 2 h, 148 ≥ 4 h, 8 ≥ 12 h; v5 predicted 7–10k on those, v8 2–7k, v4 0.3–5k; v4 predicted lower than v8/v5
on large-ms non-LIRF rows (EHAM ms 85,535 → v4 2,865 vs v5 8,626); (d) D: non-LIRF unmatched ms ≥ 2 h in OOF
= 1,837 rows, 6.4 % of SSE. Unverified in the plan: share of v4's LB advantage attributable to lower non-LIRF
predictions.

## 2. Procedure executed

Definitions (plan §0 / §4), always computed side by side:

| symbol | definition |
|---|---|
| `unm_routing` | `mvt_minus_aobt3` null (model-B routing rule; H/P/M): 22,470 training / 5,290 ranking |
| `unm_fid` | `FLIGHT_ID_mvt` null (D): 22,424 / 5,283 |
| `f6` / `f10` / `f120` | \|BLOCK − SCHED\| ≤ 6 / 10 / 120 s |
| `r1800` | \|y − ms\| < 1,800 s (M's sched-filled definition) |
| `ms` | feature `mvt_minus_sched` (= MVT − SCHED; ranking null rate 0) |
| task bins | 2–4 h [7,200, 14,400), 4–12 h [14,400, 43,200), ≥ 12 h [43,200, ∞) |
| M's bins | (14,400, 43,200], (43,200, ∞) |

Steps: (A) unmatched definitions and the 46 routing-only rows; (B) training labels of unmatched rows with
ms ≥ 2 h, LIRF vs non-LIRF × bin and per airport × bin, with all four sched-filled rates and label quantiles;
listing of every non-LIRF unmatched row with ms ≥ 4 h and y ≥ 2 h; (B2) airport-level label medians (all rows,
unmatched rows, unmatched rows with ms ≥ 4 h, leave-one-month-out unmatched median); (C) OOF
`main-48c19bc830` (672,428 scored rows of 2,085,047): label quantiles, prediction quantiles, RMSE, SSE share per
LIRF/non-LIRF × bin and per airport × bin; (C2) all unmatched OOF rows with prediction ≥ 4 h; (C3) top-25 SSE
rows; (D) counterfactual: OOF predictions of non-LIRF unmatched rows with ms ≥ 4 h (also ≥ 2 h, ≥ 12 h) replaced
by five airport-level constants plus the oracle, under both unmatched definitions; (E) ranking set: non-LIRF
unmatched rows per airport × bin vs training Jan + Jul 2025, and each submission file's predictions per bin,
per airport × bin, and on the 8 EHAM ms ≥ 12 h rows; (F) expected SSE of each submission on the 280 non-LIRF
unmatched ranking rows with ms ≥ 4 h **if** their labels follow the training label distribution of the same
(ADEP, bin) cell — a conditional computation, not a measurement.

Submission file → version mapping, fixed from `experiments/submissions.jsonl` and identity checks in the
script: v4 = bucket copy `uploaded/v4.parquet` (equals `outputs/probe_unmatched_from_v4.parquet` on all 5,290
unmatched rows); v8 = `outputs/det_a.parquet` (equals the probe on all 339,551 matched rows); v9 = the probe;
v7 = `outputs/main.parquet` (mtime 09-15 15:52, uploaded 15:53); v3 = bucket copy `uploaded/v3.parquet` (the
logged repo file `outputs/v3_two_clock_weather.parquet` no longer exists); `v5cand` =
`outputs/v5_unmatched_capacity.parquet`, a **candidate** only (the uploaded `outputs/main.parquet` of 09-14 was
overwritten). v2 exists in two disagreeing copies, both reported: `v2` = repo `outputs/v2_two_clock.parquet`,
`v2up` = bucket copy `uploaded/v2.parquet` (see §5).

## 3. Results

### A. Unmatched definitions

Training: routing 22,470 / fid 22,424 / routing-only 46 / fid-only 0. Of the 46 routing-only rows, 4 are
non-LIRF with ms ≥ 2 h (LEBL 4–12 h y 783; LFPG 4–12 h y 730; LEMD 2–4 h y 1,097; LTFM 2–4 h y 780).
Ranking: 5,290 / 5,283; 1 routing-only non-LIRF row with ms ≥ 2 h. All tables were produced under both
definitions; the differences are confined to those rows and never change a reported figure beyond the last
digit shown (368 → 366 rows in M's (4 h, 12 h] bin; OOF non-LIRF numbers identical; counterfactual identical).

### B. Training labels, unmatched rows with ms ≥ 2 h (routing definition; fid in parentheses where different)

| group | bin | n | f6 | f10 | f120 | r1800 | y p10 | y p50 | y p90 | y max | y ≥ 2 h | y ≥ 4 h | ms median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| non-LIRF | 2–4 h | 3,907 (3,905) | 0.0003 | 0.0003 | 0.0005 | 0.009 | 616 | 1,080 | 1,973 | 11,167 | 0.0082 | 0 | 8,676 |
| non-LIRF | 4–12 h | 369 (367) | 0 | 0 | 0 | 0 | 599 | 1,019 | 1,866 | 8,937 | 0.0136 | 0 | 17,935 |
| non-LIRF | ≥ 12 h | 46 | 0 | 0 | 0 | 0 | 659 | 1,099 | 1,724 | 4,667 | 0 | 0 | 70,338 |
| LIRF | 2–4 h | 500 | 0.590 | 0.590 | 0.592 | 0.598 | 825 | 7,736 | 11,233 | 87,168 | 0.602 | 0.002 | 9,179 |
| LIRF | 4–12 h | 98 | 0.847 | 0.847 | 0.847 | 0.847 | 1,273 | 18,451 | 38,956 | 42,661 | 0.847 | 0.847 | 19,022 |
| LIRF | ≥ 12 h | 29 | 0.621 | 0.621 | 0.621 | 0.655 | 45,706 | 59,641 | 87,710 | 131,167 | 1.0 | 1.0 | 56,520 |

M's bins, non-LIRF unmatched: (14,400, 43,200] n = 368 (fid: 366), all four sched-filled rates 0.000, y median
1,019, max 8,937; (43,200, ∞) n = 46, all rates 0.000, y median 1,099, max 4,667.

Per airport × bin (non-LIRF unmatched, routing): n / y p10 / p50 / p90 / max / n with y ≥ 2 h

| ADEP | 2–4 h | 4–12 h | ≥ 12 h |
|---|---|---|---|
| EDDF | 463 / 554 / 1,013 / 1,498 / 2,532 / 0 | 39 / 673 / 1,151 / 1,519 / 2,111 / 0 | 18 / 826 / 1,212 / 1,810 / 4,667 / 0 |
| EDDM | 215 / 663 / 963 / 2,705 / 6,899 / 0 | 10 / 729 / 929 / 1,314 / 2,946 / 0 | 3 / 728 / 790 / 1,072 / 1,142 / 0 |
| EGLL | 473 / 1,024 / 1,444 / 2,154 / 11,167 / 11 | 40 / 887 / 1,354 / 1,936 / 4,078 / 0 | — |
| EHAM | 614 / 397 / 838 / 1,487 / 3,903 / 0 | 77 / 301 / 776 / 1,229 / 3,841 / 0 | 14 / 595 / 1,090 / 1,622 / 3,849 / 0 |
| LEBL | 532 / 678 / 1,100 / 1,536 / 3,273 / 0 | 47 / 692 / 1,065 / 1,517 / 1,866 / 0 | 2 / 870 / 888 / 906 / 911 / 0 |
| LEMD | 272 / 733 / 1,150 / 1,608 / 3,465 / 0 | 14 / 781 / 1,158 / 2,199 / 2,315 / 0 | — |
| LFPG | 894 / 662 / 1,139 / 2,286 / 9,182 / 5 | 88 / 610 / 960 / 1,446 / 3,486 / 0 | 8 / 566 / 954 / 1,491 / 1,744 / 0 |
| LSZH | 221 / 484 / 806 / 1,287 / 9,652 / 2 | 12 / 558 / 717 / 905 / 1,272 / 0 | — |
| LTFM | 223 / 713 / 1,267 / 6,953 / 10,309 / 14 | 42 / 734 / 1,266 / 7,206 / 8,937 / 5 | 1 / 727 / 727 / 727 / 727 / 0 |

Sched-filled counts in these 22 cells: f10 = 1 (EGLL 2–4 h), f120 = 2 (EGLL, LTFM 2–4 h), r1800 = 35 (EGLL 12,
LTFM 18, LFPG 2, LSZH 2, EDDM 1 — all in the 2–4 h bin; 0 in the 4–12 h and ≥ 12 h bins).

Every non-LIRF unmatched row with ms ≥ 4 h and y ≥ 2 h (5 rows, all LTFM, all February): (ms, y, BLOCK − SCHED)
= (17,098, 8,937, +8,161), (17,034, 8,748, +8,286), (14,400, 7,800, +6,600), (20,879, 7,618, +13,261),
(14,526, 7,272, +7,254). None is sched-filled under any definition. No non-LIRF unmatched row with ms ≥ 4 h
has |y − ms| < 1,800 s.

Airport label medians (training): see script section B2. Non-LIRF: median of all rows 711–1,319; of unmatched
rows 539 (EHAM) – 1,384 (EGLL); of unmatched rows with ms ≥ 4 h 717 (LSZH) – 1,355 (EGLL), except LTFM 1,265 with
mean 3,163 (the five February rows). LOMO unmatched medians move by ≤ 60 s per airport (EHAM 501–558,
EDDM 849–896). LIRF for contrast: unmatched median 3,922, ms ≥ 4 h median 20,887, mean 31,003.

### C. OOF `main-48c19bc830` (672,428 scored rows; total RMSE 334.58; SSE 7.5275e10; unmatched 8,378 rows, RMSE 2,296.1, 58.7 % of SSE)

Pooled: non-LIRF unmatched ms ≥ 2 h = 1,837 rows, RMSE 1,624, **6.44 % of total SSE** (10.97 % of unmatched
SSE); LIRF unmatched ms ≥ 2 h = 239 rows, RMSE 10,463, 34.76 % of total SSE. Identical under both definitions.

| group | bin | n | y p10/p50/p90/max | pred p10/p50/p90/max | RMSE | share total SSE | share unm SSE | n pred ≥ 2 h | n pred ≥ 4 h | n y ≥ 2 h |
|---|---|---|---|---|---|---|---|---|---|---|
| non-LIRF | 2–4 h | 1,673 | 654 / 1,136 / 2,269 / 11,167 | 765 / 1,221 / 1,992 / 13,102 | 1,029 | 0.0235 | 0.0401 | 4 | 0 | 14 |
| non-LIRF | 4–12 h | 143 | 626 / 1,054 / 1,851 / 5,170 | **0** / 1,022 / 2,179 / 14,212 | 1,650 | 0.0052 | 0.0088 | 2 | 0 | 0 |
| non-LIRF | ≥ 12 h | 21 | 657 / 1,093 / 1,433 / 3,849 | 6,391 / 12,950 / 16,216 / 16,880 | 11,307 | **0.0357** | 0.0608 | 16 | 9 | 0 |
| LIRF | 2–4 h | 184 | 728 / 2,350 / 10,781 / 13,849 | 2,929 / 6,662 / 9,806 / 25,219 | 4,191 | 0.0429 | 0.0732 | 71 | 3 | 87 |
| LIRF | 4–12 h | 40 | 13,162 / 18,668 / 38,682 / 41,766 | 8,603 / 19,834 / 36,313 / 49,316 | 7,951 | 0.0336 | 0.0573 | 38 | 25 | 36 |
| LIRF | ≥ 12 h | 15 | 45,537 / 87,170 / 87,602 / 88,132 | 33,077 / 38,333 / 49,193 / 51,712 | 36,882 | 0.2711 | 0.4620 | 15 | 15 | 15 |

Per airport × bin, non-LIRF unmatched (n / pred median / pred max / RMSE / share of total SSE / n pred ≥ 4 h):

| ADEP | 2–4 h | 4–12 h | ≥ 12 h |
|---|---|---|---|
| EDDF | 199 / 1,124 / 2,675 / 440 / 0.0005 / 0 | 16 / 1,080 / 7,645 / 1,913 / 0.0008 / 0 | 7 / 9,385 / 16,880 / 10,044 / 0.0094 / 2 |
| EDDM | 105 / 1,131 / 2,536 / 1,229 / 0.0021 / 0 | 6 / 1,199 / 1,444 / 733 / 0.0000 / 0 | 1 / 14,011 / 14,011 / 13,221 / 0.0023 / 0 |
| EGLL | 232 / 1,618 / 8,023 / 1,520 / 0.0071 / 0 | 15 / 1,858 / 6,285 / 2,180 / 0.0009 / 0 | — |
| EHAM | 237 / 970 / 3,979 / 536 / 0.0009 / 0 | 37 / 851 / 14,212 / 2,303 / 0.0026 / 0 | 7 / 10,480 / 15,335 / 9,875 / 0.0091 / 2 |
| LEBL | 190 / 1,228 / 5,394 / 544 / 0.0007 / 0 | 17 / 930 / 3,870 / 964 / 0.0002 / 0 | 2 / 15,465 / 16,216 / 14,598 / 0.0057 / 2 |
| LEMD | 129 / 1,230 / 1,977 / 365 / 0.0002 / 0 | 10 / 1,107 / 2,164 / 972 / 0.0001 / 0 | — |
| LFPG | 431 / 1,318 / 8,807 / 1,113 / 0.0071 / 0 | 31 / 997 / 3,065 / 660 / 0.0002 / 0 | 4 / 15,306 / 16,423 / 13,180 / 0.0092 / 3 |
| LSZH | 83 / 915 / 13,102 / 1,878 / 0.0039 / 0 | 4 / 508 / 1,376 / 397 / 0.0000 / 0 | — |
| LTFM | 67 / 1,208 / 2,221 / 1,030 / 0.0009 / 0 | 7 / 1,050 / 1,881 / 1,724 / 0.0003 / 0 | — |

Per airport, all unmatched OOF rows: LIRF 537 rows 36.77 % of total SSE (RMSE 7,179; 30.47 % from its 55 rows
with ms ≥ 4 h); LFPG 1,565 rows 15.48 % (RMSE 2,729; only 0.94 % from its 35 rows with ms ≥ 4 h — the rest is
the two January rollover/long rows with ms 1,740 and 2,043, ranks 1–2 in C3); EHAM 1.54 % (1.17 % from ms ≥ 4 h);
EDDF 1.20 % (1.02 %); EGLL 1.04 %; LEBL 0.94 % (0.59 %); LSZH 0.62 %; EDDM 0.53 %; LTFM 0.45 %; LEMD 0.10 %.

**C2. Predictions ≥ 4 h on unmatched OOF rows.** Non-LIRF: **9 rows** (LIRF: 43). Their predictions
14,436–16,880 (6 of the 9 in [15k, 20k]); their labels 549–1,431 (median 865); none has y ≥ 2 h; none is
sched-filled; together 2.60 % of total SSE. All 9 have ms ≥ 45,569 (EDDF 89,185 / 54,692; LFPG 88,502 / 89,156 /
47,338; LEBL 45,569 / 68,771; EHAM 151,639 / 66,059). Non-LIRF unmatched rows predicted in [2 h, 4 h): 15
(EDDF 4, EHAM 3, LFPG 3, LEBL 2, EGLL 1, EDDM 1, LSZH 1), labels 733–1,744, none ≥ 2 h. Matched OOF rows
predicted ≥ 2 h: 36. In the 4–12 h bin the non-LIRF prediction p10 is **0** (the model outputs 0 for more than
10 % of those 143 rows; p10 of the labels is 626) — an under-prediction failure in the same bin, not a false alarm.

**C3. Top-25 SSE rows** (44.79 % of total SSE): ranks 1–2 LFPG unmatched y 84,240 / 58,206 (ms 1,740 / 2,043,
pred 922 / 992); ranks 3–15, 17, 19 LIRF unmatched; **10 non-LIRF rows at ranks 16, 18, 20–25 with ms 45,569–151,639,
y 549–1,431, pred 14,715–16,880** — these are 8 of the 9 rows in C2 plus none others (the 9th, EHAM ms 66,059
pred 14,436, is rank > 25). The three rows quoted by M appear exactly: EDDF y 661 pred 16,880 ms 89,185 (rank 16,
0.35 % of total SSE), EHAM y 549 pred 15,335 ms 151,639 (rank 23, 0.29 %), LFPG y 652 pred 15,812 ms 89,156
(rank 20, 0.31 %). The 10 non-LIRF top-25 rows carry 15.95 % of total SSE **only when the two LFPG rows at ranks
1–2 are included**; the 8 large-ms false-alarm rows among them carry 2.39 %.

### D. Counterfactual OOF RMSE (non-LIRF unmatched predictions replaced; identical under both unmatched definitions)

| rows replaced | replacement | n | total RMSE 334.58 → | unmatched RMSE 2,296.1 → | SSE on those rows (share of total) |
|---|---|---|---|---|---|
| ms ≥ 4 h | airport median of unmatched training labels | 164 | **327.88** | 2,217.2 | 3.074e9 → 9.06e7 (0.0408 → 0.0012) |
| ms ≥ 4 h | LOMO airport median of unmatched labels | 164 | 327.89 | 2,217.2 | → 9.16e7 |
| ms ≥ 4 h | airport median of all training labels | 164 | 327.88 | 2,217.1 | → 9.00e7 |
| ms ≥ 4 h | airport median of unmatched labels with ms ≥ 4 h | 164 | 327.86 | 2,216.9 | → 7.96e7 |
| ms ≥ 4 h | airport mean of unmatched labels with ms ≥ 4 h | 164 | 327.86 | 2,216.8 | → 7.82e7 |
| ms ≥ 4 h | oracle pred := y | 164 | 327.68 | 2,214.7 | → 0 |
| ms ≥ 12 h | airport median of unmatched labels | 21 | 328.60 | 2,225.6 | 2.685e9 → 1.48e7 (0.0357 → 0.0002) |
| ms ≥ 12 h | oracle | 21 | 328.56 | 2,225.2 | → 0 |
| ms ≥ 2 h | airport median of unmatched labels | 1,837 | 328.44 | 2,223.7 | 4.846e9 → 2.107e9 (0.0644 → 0.0280) |
| ms ≥ 2 h | oracle | 1,837 | 323.63 | 2,166.4 | → 0 |
| LIRF ms ≥ 4 h (contrast) | airport median of unmatched labels | 55 | 455.65 | 3,598.7 | 2.293e10 → 8.726e10 |
| all airports ms ≥ 4 h (contrast) | airport median of unmatched labels | 219 | 450.75 | 3,548.9 | 2.601e10 → 8.735e10 |

Per-airport gain of the ms ≥ 4 h median replacement (share of total SSE): EHAM 0.0113 (44 rows), EDDF 0.0101
(23), LFPG 0.0093 (35), LEBL 0.0058 (19), EDDM 0.0023 (7), EGLL 0.0008 (15), LEMD 0.0001 (10), LSZH 0.0000 (4),
LTFM −0.0000 (7; SSE 2.08e7 → 2.42e7, the only airport where the median is worse than the model). The 21
ms ≥ 12 h rows account for 2.685e9 of the 3.074e9 SSE on the 164 ms ≥ 4 h rows (87 %); the 143 rows in 4–12 h for
the remaining 3.9e8.

### E. Ranking set (344,841 rows; unmatched 5,290 routing / 5,283 fid)

Identity checks: v9 == v4 on all 5,290 unmatched rows; v9 == v8 on all 339,551 matched rows. Repo
`outputs/v2_two_clock.parquet` vs bucket copy `uploaded/v2.parquet`: 341,701 differing rows, RMS 227.4 s overall,
**1,766.9 s on unmatched rows**, 62.3 s on matched; the bucket "v2" is within 55.3 s RMS (260.8 on unmatched) of
the bucket v3; the old-code worktree rebuild `scratchpad/wt_v2/outputs_v2/v2_two_clock.parquet` equals the repo
v2 **exactly on all 5,290 unmatched rows** (RMS 0.0) and is 34.9 s RMS from it on matched rows. `v5cand` vs v7:
RMS 534.4 on unmatched rows.

Non-LIRF unmatched ranking rows with ms ≥ 2 h: **1,540** (2–4 h 1,260; 4–12 h 266; ≥ 12 h 14). Training Jan + Jul
2025 in the same cells: 1,133 / 97 / 11. EHAM: **559 / 148 / 8** confirmed (411 in 2–4 h, 140 in 4–12 h, 8 ≥ 12 h;
fid definition also 559); 502 of the 559 are January 2026 (364 + 133 + 5); training EHAM unmatched Jan + Jul 2025
ms ≥ 2 h = 194, whole-year ms ≥ 4 h = 91 vs 148 in the two ranking months. Other airports: EDDF 111, EDDM 87,
EGLL 86, LEBL 155, LEMD 84, LFPG 258, LSZH 152, LTFM 48 (ms ≥ 2 h).

Predictions on non-LIRF unmatched ranking rows, pooled (p10 / p50 / p90 / max; n ≥ 2 h / ≥ 4 h):

| bin (n) | v2 (repo) | v2up (bucket) | v3 | v4 = v9 | v5cand | v7 | v8 |
|---|---|---|---|---|---|---|---|
| 2–4 h (1,260) | 734 / 1,314 / 2,833 / 11,675; 1 / 0 | 739 / 1,322 / 2,947 / 6,149; 0 / 0 | 739 / 1,326 / 2,990 / 6,382; 0 / 0 | 759 / 1,394 / 2,990 / 5,694; 0 / 0 | 780 / 1,234 / 2,651 / 11,067; 1 / 0 | 735 / 1,336 / 2,859 / 12,369; 1 / 0 | 714 / 1,312 / 2,846 / 12,700; 1 / 0 |
| 4–12 h (266) | **0** / 1,453 / 3,194 / 15,608; 3 / 1 | 690 / 1,640 / 3,551 / 6,204; 0 / 0 | 567 / 1,557 / 3,509 / 4,700; 0 / 0 | 503 / 1,719 / 3,625 / 5,677; 0 / 0 | 511 / 1,189 / 2,929 / 12,224; 4 / 0 | **1** / 1,474 / 3,140 / 13,730; 3 / 0 | **0** / 1,392 / 3,201 / 12,299; 3 / 0 |
| ≥ 12 h (14) | 1,679 / 4,364 / 7,533 / 10,437; 2 / 0 | 1,034 / 2,829 / 5,120 / 5,224; 0 / 0 | 1,033 / 2,510 / 5,111 / 5,140; 0 / 0 | 740 / 2,690 / 4,769 / 5,013; 0 / 0 | **7,477 / 8,498 / 9,877 / 10,183; 13 / 0** | 2,628 / 5,741 / 8,849 / 10,899; 2 / 0 | 1,874 / 3,927 / 7,024 / 9,565; 1 / 0 |

Exactly **one** non-LIRF unmatched ranking row is predicted ≥ 4 h by any submission: EGLL July ms 25,438 — v2
15,608, v7 12,193, v8 12,299, v5cand 8,124, v4 100, v3 129, v2up 110. No submission predicts ≥ 12 h on any
non-LIRF unmatched ranking row. Maximum non-LIRF unmatched prediction per file: v2 15,608; v5cand 12,224;
v7 13,730; v8 12,700; v4 5,694; v3 6,382; v2up 6,204.

The 8 EHAM ms ≥ 12 h rows (ms 51,378–85,535): v5cand 7,160–8,864; v8 1,750–6,319; v7 2,465–6,930; v4 332–4,769;
v3 668–5,111; v2 (repo) 1,672–6,311; v2up 605–5,120. The ms 85,535 row: v4 2,865, v5cand 8,626, v8 3,802, v7 5,865.
EHAM 4–12 h (140 rows): medians v4 2,348 / v3 2,386 / v2up 2,256 / v8 1,779 / v2 1,770 / v7 1,659 / v5cand 1,539
(v4 is the **highest** here, not the lowest); p10 v7 7, v2 0, v8 255.

Per airport × bin medians (all files) are in script section E; the largest per-cell median of any file on a
non-LIRF cell is v5cand 9,858 (LFPG ≥ 12 h, 2 rows) and v5cand 9,289 (EDDF ≥ 12 h, 4 rows).

LIRF contrast (same files, LIRF unmatched ≥ 12 h, 11 rows): v4/v3/v2up medians 55,198 / 54,957 / 56,503 with
11 of 11 ≥ 12 h; v2 (repo)/v5cand/v7/v8 medians 36,278 / 33,791 / 35,434 / 37,306 with 2 of 11 ≥ 12 h.

### F. Conditional expected SSE on the 280 non-LIRF unmatched ranking rows with ms ≥ 4 h

Assumption (stated, not verified): ranking labels in each (ADEP, bin) cell follow the training label distribution
of that cell (mean μ, variance σ²); E[(p − y)²] = (p − μ)² + σ². Under it:

| file | E[SSE] | E[SSE] / 344,841 (added to LB MSE) |
|---|---|---|
| v2 (repo) | 1.270e9 | 3,683 |
| v2up | 8.44e8 | 2,446 |
| v3 | 7.96e8 | 2,307 |
| v4 = v9 | 8.80e8 | 2,551 |
| v5cand | 1.532e9 | 4,443 |
| v7 | 1.269e9 | 3,680 |
| v8 | 1.128e9 | 3,270 |
| pred := cell mean (irreducible) | 1.32e8 | 383 |

E[SSE_v4] − E[SSE_v8] = −2.48e8 = **2.9 %** of the measured total SSE gap (343.80² − 305.03²)·344,841 = 8.675e9;
v4 vs v7: −3.89e8 = 4.6 % of 8.388e9. RMS(v4 − v8): 1,548 s on these 280 rows, 562 s on all non-LIRF unmatched
ranking rows, 6,128 s on LIRF unmatched rows; Σ(v4 − v8)² / 344,841 = 4,502 over all non-LIRF unmatched rows vs
41,705 over LIRF unmatched rows.

## 4. Per sub-claim

| sub-claim | verdict | deciding observation |
|---|---|---|
| Non-LIRF unmatched, ms in (14,400, 43,200]: 368 rows, sched-filled 0.000; (43,200, ∞): 46 rows, 0.000 | CONFIRMED | B: 368 (fid 366) and 46; f6 = f10 = f120 = r1800 = 0.000 in both bins. Labels: median 1,019 / 1,099, max 8,937 / 4,667; 0 rows with y ≥ 4 h; 5 LTFM February rows with y 7,272–8,937 are the only y ≥ 2 h. |
| "labels are normal" at non-LIRF for large ms (also 2–4 h) | CONFIRMED with a stated exception | B: 2–4 h bin 3,907 rows, y ≥ 2 h 0.82 % (32 rows: LTFM 14, EGLL 11, LFPG 5, LSZH 2), f10 0.0003, r1800 0.009. |
| Model B "fires 15–20k" on such rows (OOF) | CONFIRMED for the ≥ 12 h bin only, REFUTED for 2–12 h | C/C2: 9 non-LIRF unmatched OOF rows predicted ≥ 4 h, all with ms ≥ 45,569, preds 14,436–16,880 (6 in [15k, 20k]), labels 549–1,431; ≥ 12 h bin pred median 12,950, 16 of 21 ≥ 2 h. In the 4–12 h bin 0 of 143 predicted ≥ 4 h (median 1,022, p10 = 0); in 2–4 h 0 of 1,673. |
| Top-25 SSE includes EDDF y 661 pred 16,880 (ms 89,185); EHAM y 549 pred 15,335 (ms 151,639); LFPG y 652 pred 15,812 | CONFIRMED | C3: ranks 16, 23, 20 with exactly those values. 8 such rows in the top 25 (2.39 % of total SSE); the other 2 non-LIRF top-25 rows are the LFPG y 84,240 / 58,206 under-predictions (ranks 1–2, 13.6 %). |
| D: non-LIRF unmatched ms ≥ 2 h in OOF = 1,837 rows, 6.4 % of SSE | CONFIRMED | C: 1,837 rows, 0.0644 of total SSE, both definitions. Of that, 3.57 % is the ≥ 12 h bin (21 rows), 0.52 % the 4–12 h bin, 2.35 % the 2–4 h bin. |
| Ranking EHAM 559 / 148 / 8 | CONFIRMED | E: 559 / 148 / 8 under both definitions. 502 of 559 are January 2026; training Jan + Jul 2025 EHAM: 194. |
| v5 predicted 7–10k on the EHAM rows, v8 2–7k, v4 0.3–5k | CONFIRMED for the 8 ms ≥ 12 h rows, using `v5cand` | E: v5cand 7,160–8,864; v8 1,750–6,319; v4 332–4,769. `v5cand` is not verified to be the uploaded v5 (§5). On the 140 EHAM 4–12 h rows the order is reversed: v4 median 2,348 is the highest, v5cand 1,539 the lowest. |
| v4 lower than v8/v5 on large-ms non-LIRF rows (EHAM ms 85,535 → 2,865 vs v5 8,626) | CONFIRMED for ≥ 12 h; REFUTED for 4–12 h | E: 2,865 / 8,626 exact (v8 3,802). ≥ 12 h pooled medians v4 2,690 < v8 3,927 < v7 5,741 < v5cand 8,498. 4–12 h pooled medians v4 1,719 > v2up 1,640 > v3 1,557 > v7 1,474 > v2 1,453 > v8 1,392 > v5cand 1,189. |
| "the ranking set contains many such rows" (rows on which model B fires 15–20k) | REFUTED as worded; large-ms rows are many, high predictions are not | E: 1,540 non-LIRF unmatched ranking rows with ms ≥ 2 h (280 ≥ 4 h, 14 ≥ 12 h), i.e. 1.36× / 2.9× / 1.3× the training Jan + Jul density. But exactly 1 non-LIRF row is predicted ≥ 4 h by any file (EGLL ms 25,438: v2 15,608, v8 12,299, v7 12,193, v4 100), no file predicts ≥ 12 h on any, and the maximum non-LIRF prediction is 15,608 (v2), 12,700 (v8), 13,730 (v7), 5,694 (v4). |
| Counterfactual: replacing non-LIRF unmatched ms ≥ 4 h predictions by the airport-median label | MEASURED | D: OOF total 334.58 → 327.88 (unmatched 2,296.1 → 2,217.2); oracle bound 327.68; the median recovers 97 % of the oracle SSE reduction on those 164 rows; 87 % of the gain is the 21 ms ≥ 12 h rows; LOMO median 327.89; identical under both definitions. Same replacement on LIRF rows: 334.58 → 455.65. |
| Share of v4's LB advantage from lower non-LIRF predictions (plan "unverified") | UNDETERMINED (conditional estimate only) | F: under the training-label-distribution assumption, v4's expected SSE on the 280 rows is 2.48e8 below v8's = 2.9 % of the measured v4→v8 SSE gap (4.6 % of the v4→v7 gap). The measured gap itself is 8.675e9; Σ(v4 − v8)² on all non-LIRF unmatched ranking rows is 1.55e9 (upper bound on any attribution regardless of labels) vs 1.44e10 on LIRF unmatched rows. |
| Whether an ADEP × ms interaction fixes it inside the GBDT (plan "unverified") | NOT TESTED | requires a fit (out of scope: no training). |

## 5. What cannot be determined locally

- Ranking labels: whether the 280 non-LIRF unmatched ms ≥ 4 h ranking rows (EHAM 148) have training-like labels
  (median ≈ 1,000 s). Table F is conditional on that; the EHAM January 2026 excess (item 15) is a reason the
  assumption may not hold, and this item does not test it.
- Which file the leaderboard scored as **v2** (312.02): the repo `outputs/v2_two_clock.parquet` (reproduced
  exactly on unmatched rows by the old-code worktree rebuild) and the bucket copy `uploaded/v2.parquet` differ by
  1,767 s RMS on unmatched rows, and the bucket copy is within 261 s RMS of bucket v3. `submissions.jsonl` names
  the repo path; the bucket copy's provenance (download time 09-15 16:01, same as v3/v4) is not recorded. Both are
  reported wherever v2 appears.
- Whether `outputs/v5_unmatched_capacity.parquet` is the uploaded v5 (325.22); the logged file
  `outputs/main.parquet` of 09-14 was overwritten. All "v5" statements above are about `v5cand`.
- The LB effect of any change to non-LIRF unmatched predictions (needs 2026 labels).

## 6. Facts vs inference

**Facts (measured in this script):** every number in §3; identity `y − ms = SCHED − BLOCK` used to equate the
f-k and r-k definitions (from item 1); v9 = v4 on unmatched / v8 on matched; det_a = v8's matched rows; the repo
v2 = old-code rebuild on all unmatched rows; the two unmatched definitions differ on 46 training / 7 ranking rows
of which 4 / 1 are non-LIRF with ms ≥ 2 h, without changing any figure above beyond its last digit.

**Inference (not measured):** that the 9 OOF rows predicted 14–17k at non-LIRF are caused by the LIRF sched-filled
regime leaking through `mvt_minus_sched` (the co-location of all 9 in the ms ≥ 45k region where LIRF labels ≈ ms
is consistent with it, but the tree structure was not inspected); that ranking labels follow training cell
distributions (Table F); that the bucket "v2" copy is a mislabelled v3-family file (its 261 s RMS distance to v3
on unmatched rows is suggestive, not conclusive).

**Definitional notes for cross-item comparison:** M's "368 / 46" uses bins (14,400, 43,200] / (43,200, ∞) and
the routing definition; the task's [14,400, 43,200) / [43,200, ∞) bins give 369 / 46 (one row has ms exactly
14,400: LTFM February, y 7,800). D's "1,837 / 6.4 %" is the OOF file's non-LIRF unmatched ms ≥ 2 h subset and is
the same under both unmatched definitions. "15–20k" in the hypothesis corresponds in the data to the 9 OOF rows
with pred 14,436–16,880 and to **no** ranking prediction of any file (non-LIRF max 15,608 in v2, on one EGLL row
where v4 predicts 100).
