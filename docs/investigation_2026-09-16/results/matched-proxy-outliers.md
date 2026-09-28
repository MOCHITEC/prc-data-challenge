# Item 5 `matched-proxy-outliers` — verification report (2026-09-16)

Script: `docs/investigation_2026-09-16/scripts/matched-proxy-outliers.py` (POLARS_MAX_THREADS=1, `scan_parquet` with 15 selected columns, no training, run time ≈ 1 min).
Inputs: `data/work/feat_train.parquet` (2,085,047 DEP rows), `data/work/feat_rank.parquet` (344,841 DEP rows), `experiments/oof_main-48c19bc830.parquet` (672,428 non-null rows), `outputs/main.parquet` (344,841 rows).

Definitions (plan §0): `y = TAXITIME_SEC_mvt`; `proxy = MVT − AOBT_3`; `ms = MVT − SCHED`; `gap = BLOCK − AOBT_3` (D's `blk_minus_aobt`; positive = BLOCK later than AOBT_3); `fill = |BLOCK − SCHED| ≤ 10 s`; **matched = routing definition (AOBT_3 not null)** unless stated; D's definition (FLIGHT_ID not null) is reported alongside where it changes a count. 1 h = 3,600 s, 2 h = 7,200 s.

## 1. Hypothesis as stated (plan §1 item 5, source D)

A set of matched rows has |BLOCK − AOBT_3| > 1 h (LTFM: stale early AOBT_3; EGLL/LFPG: BLOCK wrong), producing large errors under the proxy decomposition; some ranking matched rows are predicted ≥ 2 h.
Claimed: |gap| > 1 h on 1,808 matched rows; > 30 min 10,554; > 10 min 6.8 %. LTFM 657 rows: proxy median 4,692, y median 850. EGLL 296 / LFPG 186 / EDDM 45: y median 5,035 / 3,328 / 3,722, proxy 600 / 658 / 119. Matched |proxy − y| > 1 h: 689 OOF rows, 2.5 % SSE. Ranking matched predicted ≥ 2 h: EGLL 7, LIRF 7, EHAM 6, LFPG 2 (22). Ranking matched AOBT_3 null: 7. Matched y ≥ 2 h = 103 training rows (47 in OOF, 0.9 % of matched SSE, y − proxy median 6,960) (M).
Unverified per plan: whether the 22 ranking rows are "the same kind"; LB effect.

## 2. Procedure actually executed

A. Training: derive y, proxy, ms, gap, fill from timestamps; verify `y == MVT − BLOCK` and `proxy == mvt_minus_aobt3`; count matched rows by |gap| threshold; per-airport table of the |gap| > 1 h group with sign of gap, fill rate, AOBT_3 − SCHED, AOBT_3 − EOBT_1; sign split at LTFM / EGLL / LFPG / EDDM / LIRF; matched rows with y ≥ 2 h and with proxy ≥ 2 h.
B. OOF: inner join on `MVT_ID_mvt`; total RMSE; SSE share of matched |proxy − y| > 1 h, |gap| > 1 h, |gap| > 30 min, y ≥ 2 h, proxy ≥ 2 h (extension), pred ≥ 2 h; per-airport breakdown of |gap| > 1 h with pred vs proxy vs y.
C. Ranking: join `feat_rank` with `outputs/main.parquet`; count matched rows predicted ≥ 2 h under both matched definitions, with and without D's `ms ≥ 2 h` pre-filter; list all 22 with proxy / ms / airport / operator (extension); the 7 AOBT_3-null matched rows; ranking matched rows with proxy ≥ 2 h.
D. "Same kind" check: define the visible signature of the 22 rows (S1: proxy < 2,000 s ∧ AOBT_3 − SCHED ≥ 5,400 s; S2: 4,000 ≤ proxy < 7,200 s) and look up, in training and OOF, what y was for matched rows carrying the same signature, per airport; classify EGLL/LIRF/LFPG/LTFM S1 training rows as y≈proxy / y≈ms / neither (±1,800 s).

## 3. Results

### 3.1 Training (2,062,577 matched rows)

Identity `y == MVT − BLOCK`: 0 violations. `proxy == mvt_minus_aobt3` and `ms == mvt_minus_sched` features: max |diff| 0.0. Matched(routing) 2,062,577; matched(D) 2,062,623; FLIGHT_ID present but AOBT_3 null: 46.

| threshold on \|gap\| | rows | rate |
|---|---|---|
| > 1 h | **1,808** | 0.088 % |
| > 30 min | **10,554** | 0.51 % |
| > 10 min | 140,633 | **6.82 %** |

(Identical under D's definition: 2,062,577 / 1,808 / 10,554.)

|gap| > 1 h by airport (n = 1,808; 96 of them have y ≥ 2 h, 11 have proxy ≥ 2 h):

| ADEP | n | y med | proxy med | gap med | share gap > 0 | fill | AOBT−SCHED med | AOBT−EOBT med | ms med |
|---|---|---|---|---|---|---|---|---|---|
| LTFM | 657 | 850 | 4,692 | +3,826 | 0.88 | 0.06 | −3,575 | −3,609 | 1,198 |
| LIRF | 457 | 5,274 | 1,025 | −4,256 | 0.00 | 0.80 | 4,260 | 1,200 | 5,335 |
| EGLL | 296 | 5,034.5 | 600 | −4,615.5 | 0.01 | 0.03 | 5,340 | 4,860 | 5,999 |
| LFPG | 186 | 3,328 | 657.5 | −5,344 | 0.06 | 0.02 | 8,599.5 | 4,260 | 7,196 |
| EHAM | 54 | 774 | −3,511 | −4,544.5 | 0.28 | 0.02 | 10,050 | 1,140 | 6,133 |
| EDDF | 50 | 1,126.5 | −3,155.5 | −4,940 | 0.02 | 0.00 | 8,820 | 2,880 | 6,647.5 |
| EDDM | 45 | 3,722 | 119 | −4,199 | 0.16 | 0.02 | 6,420 | 3,780 | 5,278 |
| LSZH | 30 | 771.5 | −4,400.5 | −5,345 | 0.00 | 0.00 | 9,420 | 2,820 | 5,109 |
| LEMD | 20 | 1,039.5 | −5,148 | −5,970.5 | 0.00 | 0.00 | 13,590 | 5,610 | 8,024 |
| LEBL | 13 | 885 | −5,846 | −6,781 | 0.00 | 0.00 | 13,200 | 5,160 | 7,229 |

Baseline (all matched rows) per airport: gap median −118…+296 s, gap sd 249–515 s; LTFM has the largest positive gap median (+296) and the most |gap| > 1 h rows (657 of 269,320).

LTFM sign split: gap > 0 (AOBT_3 earlier than BLOCK by > 1 h): 579 rows, y med 842, proxy med 4,741, AOBT_3 − SCHED med −3,670, AOBT_3 − EOBT_1 med −3,684, fill 0.07. gap < 0: 78 rows, y med 4,046, proxy med −0.5. The 579 rows are spread over all 12 months (15–93 per month) and 530 of 657 belong to one operator hash (`95828822…`).
EGLL: gap < 0 292 rows (y med 5,038.5, proxy 599, fill 0.03, ms med 5,999), gap > 0 4 rows. LFPG: gap < 0 174 (y 4,600, proxy 594, fill 0.02), gap > 0 12. EDDM: gap < 0 38 (y 3,846, proxy 87.5), gap > 0 7. LIRF: gap < 0 455 (y 5,274, proxy 1,025, **fill 0.80**), gap > 0 2.

Matched y ≥ 2 h: **103** rows (LIRF 41, LFPG 30, EGLL 26, LTFM 2, LSZH 2, EHAM 2); (y − proxy) median 7,142; fill 0.32; |gap| > 1 h on 93.2 % of them.
Matched proxy ≥ 2 h: **19** rows only (LTFM 9 with y med 1,669 and 0 % y ≥ 2 h; LIRF 6 with y med 12,811 and 83 % y ≥ 2 h; EDDM/EHAM/EDDF/EGLL 1 each); y ≥ 2 h share 31.6 %.

### 3.2 OOF (`oof_main-48c19bc830`, 672,428 rows, total RMSE 334.58; matched 664,050 rows, RMSE 216.44, 41.3 % of total SSE)

| subset (matched) | n | SSE / total | SSE / matched | RMSE in | pred med | y med | proxy med |
|---|---|---|---|---|---|---|---|
| \|proxy − y\| > 1 h | **689** | **0.0251** | 0.0608 | 1,657 | 1,844 | 2,150 | 962 |
| \|BLOCK − AOBT_3\| > 1 h | 689 | 0.0251 | 0.0608 | 1,657 | 1,844 | 2,150 | 962 |
| \|BLOCK − AOBT_3\| > 30 min | 3,703 | 0.0752 | 0.1821 | 1,237 | 1,923 | 2,768 | 895 |
| y ≥ 2 h | **47** | 0.0036 | **0.0088** | 2,410 | 7,146 | 8,105 | 1,015 |
| proxy ≥ 2 h (extension) | **5** | 0.0006 | 0.0014 | **2,982** | 4,793 | 2,106 | 7,796 |
| — proxy ≥ 2 h ∧ y < 2 h | 3 | 0.0002 | 0.0004 | 2,040 | 3,574 | 1,687 | 7,523 |
| — proxy ≥ 2 h ∧ y ≥ 2 h | 2 | 0.0004 | 0.0010 | 3,999 | 45,281 | 48,987 | 47,488.5 |
| — proxy < 2 h ∧ y ≥ 2 h | 45 | 0.0032 | 0.0078 | 2,315 | 7,146 | 8,103 | 1,014 |
| pred ≥ 2 h | 36 | 0.0017 | 0.0042 | 1,910 | 7,954 | 8,168.5 | 781.5 |
| pred ≥ 2 h ∧ y < 1 h | 1 | 0.0005 | 0.0011 | 5,907 | 7,717 | 1,810 | 1,445 |

(Because the matched identity y = proxy − gap holds exactly, `|proxy − y| > 1 h` and `|gap| > 1 h` are the same set.)

|gap| > 1 h in OOF, by airport:

| ADEP | n | SSE/total | RMSE in | y med | proxy med | pred med | \|pred − proxy\| med | \|pred − y\| med |
|---|---|---|---|---|---|---|---|---|
| LTFM | 218 | 0.0022 | 871 | 870 | 4,678.5 | 1,095 | 3,676 | 284 |
| LIRF | 157 | **0.0131** | 2,505 | 5,337 | 1,020 | 3,369 | 2,595 | 2,256 |
| EGLL | 138 | 0.0034 | 1,360 | 4,918 | 600 | 4,537 | 4,479 | 801 |
| LFPG | 78 | 0.0040 | 1,954 | 2,401.5 | −1,563.5 | 1,873 | 4,988 | 938 |
| EDDF / EHAM / EDDM / LSZH / LEBL / LEMD | 27 / 21 / 21 / 15 / 7 / 7 | ≤ 0.0007 each | 759–1,688 | 733–2,029 | −5,979…+119 | 439–2,001 | 3,521–7,690 | 663–1,094 |

Extension — matched rows with proxy > 7,200 in OOF: **n = 5**; RMSE(pred, y) = **2,982.2**; RMSE(proxy, y) = 4,817.8; RMSE(pred, proxy) = 3,537.8; SSE share of total 0.06 %. The five rows: EHAM Jul y 10,972 proxy 7,796 pred 5,764; EGLL Jul y 1,687 proxy 8,163 pred 4,793; LIRF Jul y 87,002 proxy 87,181 pred 84,798 (fill = true); LTFM Nov y 2,106 proxy 7,363 pred 3,574; LTFM Jan y 1,493 proxy 7,523 pred 2,321.

### 3.3 Ranking (`feat_rank` ⋈ `outputs/main.parquet`, 344,841 rows; BLOCK_TIME and TAXITIME are 100 % null in ranking DEP rows)

Matched(routing) 339,551; matched(D) 339,558; matched(D) with AOBT_3 null: **7** (LEMD 2, LFPG 2, LTFM 2, LEBL 1; predictions 764–10,899; the LFPG Jan row has ms 52,555 and pred 10,899). Unmatched(routing) 5,290.

Ranking rows predicted ≥ 2 h: all rows 122; unmatched(routing) 100; matched:

| matched definition | pred ≥ 2 h | of which ms ≥ 2 h (D's q5 Q filter) | by airport |
|---|---|---|---|
| routing (AOBT_3 not null) | **22** | 21 | LIRF 8, EGLL 7, EHAM 6, LFPG 1 |
| D (FLIGHT_ID not null) | 23 | **22** | LIRF 8, EGLL 7, EHAM 6, LFPG 2 |

D's "22 = EGLL 7, LIRF 7, EHAM 6, LFPG 2" is reproduced only as matched(D) ∧ ms ≥ 2 h; the routing-definition count without the ms filter is also 22 but with LIRF 8 / LFPG 1 (the LIRF row ITY340 has ms 94,560 and is dropped by nothing; the LIRF row TKJ4QU has ms 6,837 < 7,200 and is dropped by D's filter; D's second LFPG row is the AOBT_3-null row above, routed to model B).

Extension — the 22 matched(routing) rows predicted ≥ 2 h:

| MVT_ID | ADEP | month | FLIGHT | pred | proxy | ms | AOBT−SCHED | AOBT−EOBT | pred − proxy |
|---|---|---|---|---|---|---|---|---|---|
| 212247168 | EGLL | 7 | BAW494 | 8,885 | 602 | 10,502 | 9,900 | 10,200 | 8,283 |
| 212240195 | EGLL | 7 | BAW358 | 8,650 | 594 | 11,154 | 10,560 | 8,160 | 8,056 |
| 202183793 | EGLL | 1 | BAW3KG | 8,016 | 601 | 10,381 | 9,780 | 7,680 | 7,415 |
| 212245340 | EGLL | 7 | BAW454 | 7,692 | 604 | 7,564 | 6,960 | 7,080 | 7,088 |
| 212242170 | EGLL | 7 | BAW872 | 7,688 | 600 | 8,220 | 7,620 | 7,920 | 7,088 |
| 212240402 | EGLL | 7 | BAW396U | 7,677 | 597 | 8,037 | 7,440 | 7,740 | 7,080 |
| 212245428 | EGLL | 7 | BAW336 | 7,593 | 598 | 9,298 | 8,700 | 9,000 | 6,995 |
| 202307775 | EHAM | 1 | 5Y5579 | 10,305 | 5,425 | 13,414 | 7,989 | 7,989 | 4,880 |
| 202319332 | EHAM | 1 | MP9411 | 9,596 | 5,429 | 15,629 | 10,200 | 5,400 | 4,167 |
| 202320467 | EHAM | 1 | EK148 | 8,890 | 5,437 | 28,014 | 22,577 | 6,077 | 3,453 |
| 202310547 | EHAM | 1 | 6E0022 | 8,743 | 5,392 | 30,712 | 25,320 | 5,520 | 3,351 |
| 202304368 | EHAM | 1 | KL0597 | 8,026 | 5,392 | 11,692 | 6,300 | 5,700 | 2,634 |
| 202309192 | EHAM | 1 | KE8536 | 7,237 | 5,430 | 70,192 | 64,762 | 4,642 | 1,807 |
| 212756714 | LFPG | 7 | AFR19DE | 12,304 | 13,313 | 13,620 | 307 | 307 | −1,009 |
| 213147416 | LIRF | 7 | ITY340 | 84,866 | 86,340 | 94,560 | 8,220 | −60 | −1,474 |
| 202681723 | LIRF | 1 | ISR342 | 10,166 | 1,079 | 8,099 | 7,020 | 960 | 9,087 |
| 213155902 | LIRF | 7 | TKJ4QU | 9,070 | 1,077 | 6,837 | 5,760 | 660 | 7,993 |
| 213143101 | LIRF | 7 | WMT4078 | 8,961 | 957 | 13,317 | 12,360 | 10,200 | 8,004 |
| 213146889 | LIRF | 7 | DAL113 | 8,552 | 1,138 | 8,578 | 7,440 | 7,800 | 7,414 |
| 213147461 | LIRF | 7 | AMX071 | 8,513 | 1,496 | 8,816 | 7,320 | 7,320 | 7,017 |
| 213148774 | LIRF | 7 | ASA181 | 7,980 | 1,195 | 8,335 | 7,140 | 7,140 | 6,785 |
| 213148460 | LIRF | 7 | RYRYR5FY | 7,628 | 895 | 7,795 | 6,900 | 6,000 | 6,733 |

Summary of the 22: proxy ≥ 2 h on **2** (LFPG AFR19DE 13,313; LIRF ITY340 86,340); the other 20 have proxy 594–5,437 s. ms ≥ 2 h on 21. pred − proxy median 7,006, max 9,087. All 7 EGLL rows carry the same operator hash (`8e92b7d7…`) and proxy 594–604 s, and 6 of 7 are July. All 6 EHAM rows are January with proxy 5,392–5,437 s (a 45 s band). Ranking matched rows with proxy ≥ 2 h: 5 in total (LFPG 2, LIRF 1, LTFM 1, EDDM 1); the LTFM (proxy 7,277, pred 3,298) and EDDM (proxy 7,615, pred 3,336) rows are predicted well below proxy.

### 3.4 Signature lookup in training / OOF ("same kind?")

S1 = matched ∧ proxy < 2,000 ∧ AOBT_3 − SCHED ≥ 5,400 (the EGLL / LIRF-type signature of 14 of the 22): training 45,300 rows (2.2 % of matched); ranking 10,174 rows (3.0 %).

| ADEP | train n | y med | y ≥ 2 h share | \|y − ms\| ≤ 30 min | \|gap\| > 1 h | OOF n | OOF pred med | OOF pred ≥ 2 h share | OOF RMSE | ranking n | ranking pred ≥ 2 h |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EGLL | 3,756 | 1,382 | 0.007 | 0.023 | 0.039 | 1,544 | 1,404 | 0.012 | 472 | 765 | 7 |
| LIRF | 3,871 | 1,088 | 0.009 | 0.018 | 0.024 | 1,420 | 1,220 | 0.004 | 625 | 1,080 | 7 |
| LFPG | 6,834 | 971 | 0.004 | 0.006 | 0.021 | 2,675 | 1,051 | 0.002 | 462 | 1,542 | 0 |
| LTFM | 6,266 | 956 | 0.000 | 0.002 | 0.006 | 2,428 | 1,016 | 0.000 | 386 | 1,428 | 0 |
| EHAM | 6,229 | 783 | 0.000 | 0.000 | 0.006 | 2,428 | 762 | 0.000 | 211 | 1,156 | 0 |
| other 5 | 1,905–5,103 | 765–1,071 | ≤ 0.001 | ≤ 0.001 | ≤ 0.014 | 664–1,729 | 797–1,084 | ≤ 0.001 | 251–439 | 539–1,224 | 0 |

Within S1, the label is y ≈ proxy (±1,800 s) on EGLL 3,533 / 3,756 (94.1 %), LIRF 3,683 / 3,871 (95.1 %), LFPG 6,601 / 6,834 (96.6 %), LTFM 6,156 / 6,266 (98.2 %); y ≈ ms on EGLL 87 (2.3 %, y med 5,988), LIRF 69 (1.8 %, y med 7,069), LFPG 44 (0.6 %), LTFM 14 (0.2 %).

S2 = matched ∧ 4,000 ≤ proxy < 7,200 (the EHAM-January signature of 6 of the 22): training LTFM 1,152 (y med 906, |y − ms| ≤ 30 min 90.9 %, |gap| > 1 h 49.7 %), EGLL 211 (y med 3,294, y ≥ 2 h 0.000), EHAM 96 (y med 3,335, y ≥ 2 h 0.010), LFPG 44 (y med 2,223); OOF EHAM S2 60 rows pred med 3,857, pred ≥ 2 h share 0.000, RMSE 819. Ranking S2: LTFM 299 (pred med 944), **EHAM 283 (271 in January, pred med 4,500, 6 predicted ≥ 2 h)**, LFPG 52, EGLL 42.

## 4. Verdicts per sub-claim

| # | sub-claim | verdict | deciding observation |
|---|---|---|---|
| 1 | 1,808 matched rows with \|BLOCK − AOBT_3\| > 1 h; 10,554 > 30 min; 6.8 % > 10 min | **CONFIRMED** | 1,808 / 10,554 / 6.82 % (140,633) on the 2,062,577 routing-matched rows; identical under D's definition. |
| 2 | LTFM 657 rows, proxy med 4,692, y med 850; EGLL 296 / LFPG 186 / EDDM 45 with y med 5,035 / 3,328 / 3,722 and proxy 600 / 658 / 119 | **CONFIRMED** | Exact match: LTFM 657 / 4,692 / 850; EGLL 296 / 5,034.5 / 600; LFPG 186 / 3,328 / 657.5; EDDM 45 / 3,722 / 119. LIRF (457, y 5,274, proxy 1,025, fill 0.80) is the second-largest group and is not named in the hypothesis text. |
| 3 | LTFM mechanism = "stale early AOBT_3" | **CONFIRMED as a data pattern** (the label "stale" is inference) | 579 of 657 LTFM rows have AOBT_3 earlier than BLOCK by > 1 h (gap med +3,826), AOBT_3 − SCHED med −3,670 and AOBT_3 − EOBT_1 med −3,684 (AOBT_3 about 1 h before both schedule and EOBT), y med 842 ≈ LTFM baseline 963, fill 0.07; 530 / 657 from one operator hash. The direction (AOBT_3 too early, y normal) is a fact; whether the cause is a stale message cannot be read from the data. |
| 4 | EGLL / LFPG mechanism = "BLOCK wrong" | **UNDETERMINED** (direction confirmed, attribution not) | EGLL 292 / 296 and LFPG 174 / 186 rows have BLOCK > 1 h before AOBT_3 (gap med −4,616 / −5,344), proxy normal (599 / 594 ≈ baselines 1,324 / 900), y ≈ ms − ~1,000 (EGLL y 5,038 vs ms 5,999). Fill (≤ 10 s) is only 0.03 / 0.02, so these are not the LIRF-type sched-filled BLOCK. Whether BLOCK or AOBT_3 is the erroneous timestamp cannot be decided from these two sources alone; a 1–1.5 h delay held on stand after the recorded BLOCK is equally consistent with the numbers. |
| 5 | Matched \|proxy − y\| > 1 h: 689 OOF rows, 2.5 % of SSE | **CONFIRMED** | 689 rows, 2.51 % of total OOF SSE (6.08 % of matched SSE), RMSE within 1,657. LIRF alone carries 1.31 pp of the 2.51 pp. |
| 6 | Matched y ≥ 2 h = 103 training / 47 OOF rows, 0.9 % of matched SSE, y − proxy med 6,960 (M) | **CONFIRMED** (median differs slightly) | 103 training rows; 47 OOF rows = 0.88 % of matched SSE; y − proxy median 7,142 on the 103 training rows (M's 6,960 presumably on the 47 OOF rows; not recomputed on exactly M's subset). |
| 7 | Ranking matched predicted ≥ 2 h: 22 = EGLL 7, LIRF 7, EHAM 6, LFPG 2 | **CONFIRMED with definitional caveat** | 22 under routing-matched (LIRF 8, EGLL 7, EHAM 6, LFPG 1); D's exact split reproduces only as FLIGHT_ID-matched ∧ ms ≥ 2 h (22 of 23). The two definitions differ by one AOBT_3-null LFPG row (pred 10,899, model B) and one LIRF row with ms 6,837. |
| 8 | Ranking matched AOBT_3 null: 7 | **CONFIRMED** | 7 rows (LEMD 2, LFPG 2, LTFM 2, LEBL 1), routed to model B by the AOBT_3-null rule. |
| 9 | The 22 ranking rows are "proxy outliers" of the same kind as the 1,808 training rows | **REFUTED for 20 of 22** | Only 2 of 22 have proxy ≥ 2 h (LFPG 13,313; LIRF 86,340). The other 20 have proxy 594–5,437 s; their predictions exceed proxy by 1,807–9,087 s. The training |gap| > 1 h set is defined by a BLOCK − AOBT_3 gap, which is unobservable in ranking (BLOCK 100 % null); so membership cannot be tested directly, and by the only visible quantity (proxy) they are not outliers. |
| 10 | "Same kind" by visible signature | **UNDETERMINED (labels absent); training analogue quantified** | The 14 EGLL/LIRF-type rows share signature S1 with 45,300 training rows on which y ≥ 2 h occurs on 0.7 % (EGLL) / 0.9 % (LIRF) and y ≈ proxy on 94–95 %; the OOF model nonetheless predicted ≥ 2 h on 1.2 % of EGLL S1 and 0.4 % of LIRF S1 rows (ranking: 7 / 765 = 0.9 % EGLL, 7 / 1,080 = 0.6 % LIRF). The 6 EHAM rows share S2 with 96 EHAM training rows (y med 3,335, y ≥ 2 h 1.0 %; OOF pred ≥ 2 h share 0.000) — the ranking EHAM S2 population is 283 rows, 271 in January, vs 96 in a full training year. |
| 11 | These rows produce "large errors under the proxy decomposition" | **CONFIRMED in OOF for the training set** | Rows with |gap| > 1 h: RMSE 1,657 vs matched overall 216; matched pred ≥ 2 h in OOF (36 rows) has y med 8,168 and only 1 row with y < 1 h, i.e. the ≥ 2 h predictions on matched rows were not mostly false alarms in OOF. |
| 12 | Extension: OOF RMSE of matched rows with proxy > 7,200 | **measured** | n = 5, RMSE(pred, y) 2,982; RMSE(proxy, y) 4,818; pred sits between proxy and y on 3 of 5 rows (below both on the EHAM and LIRF rows). SSE share 0.06 %. Too few rows for any distributional claim. |
| 13 | LB effect | **UNDETERMINED** | Requires ranking labels. |

## 5. What cannot be determined locally

- Ranking labels: whether any of the 22 rows has y ≥ 2 h, hence the LB effect of these predictions (an upper bound is not computable without assuming a label distribution).
- Whether BLOCK or AOBT_3 is the erroneous timestamp at EGLL/LFPG/EDDM (needs an independent third timestamp or source documentation).
- Whether the LTFM early-AOBT_3 pattern is a stale message, a different AOBT semantic for one operator, or a genuine early off-block followed by a long hold; only its shape (one operator, all months, AOBT_3 ≈ 1 h before SCHED and EOBT, y normal) is observable.
- Whether the EHAM January S2 concentration (271 ranking rows vs 96 training rows/year) reflects the same generating process as training.

## 6. Facts vs inference

Facts (all measured in this run): every count and median in §3; identity y = MVT − BLOCK with 0 violations; `|proxy − y| > 1 h` ≡ `|gap| > 1 h` on matched rows; BLOCK and TAXITIME 100 % null in ranking DEP rows; the 22-row list; 2 of 22 with proxy ≥ 2 h; S1/S2 conditional label tables; operator concentration (LTFM 530 / 657 one hash; EGLL 7 / 7 one hash).

Inference (not established by the data here): the words "stale" (LTFM) and "wrong" (EGLL/LFPG); that the model's ≥ 2 h predictions on the 20 normal-proxy ranking rows are driven by the AOBT_3 − SCHED / ms features learned from the y ≈ ms minority (EGLL 87, LIRF 69 rows) inside S1 — consistent with the pred − proxy ≈ ms − proxy − ~1,000 pattern in the table but not tested by feature attribution; that the ranking rows would resolve the same way as the training analogues.
