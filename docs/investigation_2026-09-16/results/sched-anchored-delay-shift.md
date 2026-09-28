# 18. `sched-anchored-delay-shift` — results (2026-09-16)

Source item: plan.md §1 #18 (from D `sched-anchored-delay-shift`). Script:
`docs/investigation_2026-09-16/scripts/sched-anchored-delay-shift.py` (run with
`POLARS_MAX_THREADS=1 .venv/bin/python …`; lazy scans, 4–12 columns per scan; no model training, no
`prc.adversarial` run). Inputs: `data/work/feat_train.parquet` (2,085,047 DEP rows), `data/work/feat_rank.parquet`
(344,841 DEP rows), `experiments/oof_main-48c19bc830.parquet`, `data/work/adv_score.parquet`.
Definitions follow plan §0: matched = routing definition (`mvt_minus_aobt3` not null); `ms` = `mvt_minus_sched`;
proxy = `mvt_minus_aobt3`; residual = `y − oof`. Sibling report built on: `results/fold-shape-vs-ranking.md`
(month composition of the OOF file: 2025-01 / 07 / 11 / 12 only; 2025-07 predictions are the leaky `jul` fold's).

## 1. Hypothesis (as consolidated in plan.md)

2026 delay levels differ by month from 2025 — Jan 2026 later than Jan 2025, Jul 2026 earlier than Jul 2025 —
shifting every SCHED/EOBT-anchored feature (`mvt_minus_sched`, `aobt3_minus_sched`, `aobt3_minus_eobt1`), while
the proxy (`mvt_minus_aobt3`) medians, congestion counts, headways and taxi-in means do not move. D's specific
numbers: matched ms median Jan 2026/2025 EDDF 1,545/1,183, EDDM 1,324/1,025, LTFM 1,615/1,194, LSZH 1,509/1,232,
EHAM 1,451/1,371; Jul EDDF 1,456/1,693, EDDM 1,255/1,374, LIRF 1,864/2,038, LTFM 1,262/1,436, EGLL 1,678/1,918;
`aobt3_minus_sched` median Jan 420 vs 240, Jul 481 vs 720; `aobt3_minus_eobt1` LTFM Jan 0 vs −180, Jul −60 vs
+120; proxy medians stable (≤ 50 s); proxy sd higher in Jan 2026 at EHAM (740 vs 476), EDDF, EDDM, LSZH.
Unverified in the plan: effect on predictions.

## 2. Procedure

A. Matched rows of Jan/Jul 2025 (feat_train) and Jan/Jul 2026 (feat_rank): per airport × month and pooled, n,
   median of `mvt_minus_sched`, `aobt3_minus_sched`, `aobt3_minus_eobt1`, `mvt_minus_aobt3`, sd of the proxy;
   pooled quantiles (p10–p90) and shares > 60 / > 120 min of `mvt_minus_sched`.
B. Same rows: medians and means (NaN treated as null) of `dep_tko_prev_30m/60m`, `arr_land_prev_30m`,
   `arr_inblock_prev_30m`, `nbr_dep_n_30m`, `rwy_headway_prev_s/next_s` (plus p95 and null rate),
   `arr_taxi_in_mean_30m`, `nbr_mvt_minus_aobt_mean_30m`, `queue_ahead`.
C. OOF file inner-joined to feat_train on `MVT_ID_mvt`, non-null `oof` only (672,428 rows, RMSE 334.58,
   664,050 matched): n, RMSE, mean/median residual, SSE share by bin of `mvt_minus_sched` and of
   `aobt3_minus_sched` (minutes: <−15, −15..0, 0..15, 15..30, 30..60, 60..120, >120; left-closed), for all rows,
   matched only, unmatched only, and matched × OOF month.
D. Share of matched rows per bin: training all-2025 / OOF-scored / Jan 2025 / Jul 2025 vs ranking all /
   Jan 2026 / Jul 2026; also all rows (matched + unmatched) for `mvt_minus_sched`.
E. `adv_score.parquet`: schema, coverage, mean/median `adv` and share > 0.5 by bin × matched, by month, by airport.
F. Arithmetic only: matched per-bin OOF MSE re-weighted with the ranking bin shares (no refit).

## 3. Results

### 3.1 SCHED/EOBT-anchored features, matched rows, 2025 → 2026 (medians, seconds)

`n` = matched rows (2025 / 2026). Δ = 2026 − 2025.

| airport | n Jan | ms Jan 25→26 (Δ) | n Jul | ms Jul 25→26 (Δ) | aobt3−sched Jan (Δ) | aobt3−sched Jul (Δ) | aobt3−eobt1 Jan (Δ) | aobt3−eobt1 Jul (Δ) |
|---|---|---|---|---|---|---|---|---|
| EDDF | 15,332 / 15,455 | 1,183 → 1,545 (+362) | 20,899 / 20,441 | 1,693 → 1,456 (−237) | 240 → 540 (+300) | 780 → 540 (−240) | 240 → 300 (+60) | 420 → 300 (−120) |
| EDDM | 11,201 / 10,834 | 1,025 → 1,324 (+299) | 15,653 / 14,894 | 1,374 → 1,255 (−119) | 300 → 540 (+240) | 660 → 540 (−120) | 180 → 300 (+120) | 360 → 300 (−60) |
| EGLL | 19,292 / 19,171 | 1,494 → 1,499 (+5) | 20,452 / 20,421 | 1,918 → 1,678 (−240) | 60 → 120 (+60) | 480 → 240 (−240) | 60 → 120 (+60) | 300 → 240 (−60) |
| EHAM | 18,167 / 15,427 | 1,371 → 1,451 (+80) | 21,974 / 21,549 | 1,644 → 1,441 (−203) | 420 → 480 (+60) | 660 → 420 (−240) | 275 → 274 (−1) | 360 → 240 (−120) |
| LEBL | 12,168 / 12,124 | 1,074 → 1,128.5 (+54.5) | 16,322 / 17,514 | 1,733 → 1,583 (−150) | 240 → 240 (0) | 840 → 720 (−120) | 180 → 180 (0) | 480 → 420 (−60) |
| LEMD | 16,478 / 16,739 | 1,346 → 1,524 (+178) | 18,541 / 19,977 | 1,712 → 1,623 (−89) | 360 → 540 (+180) | 720 → 660 (−60) | 180 → 240 (+60) | 420 → 360 (−60) |
| LFPG | 17,684 / 17,804 | 1,620 → 1,625 (+5) | 21,048 / 21,334 | 1,916 → 1,624 (−292) | 600 → 660 (+60) | 960 → 600 (−360) | 360 → 300 (−60) | 600 → 120 (−480) |
| LIRF | 11,257 / 10,954 | 1,144 → 1,145 (+1) | 14,874 / 15,562 | 2,038 → 1,864 (−174) | 240 → 240 (0) | 1,080 → 900 (−180) | 180 → 180 (0) | 420 → 360 (−60) |
| LSZH | 9,556 / 9,695 | 1,232 → 1,509 (+277) | 12,266 / 12,796 | 1,635 → 1,528 (−107) | 540 → 720 (+180) | 960 → 840 (−120) | 420 → 480 (+60) | 600 → 540 (−60) |
| LTFM | 21,136 / 22,063 | 1,194 → 1,615 (+421) | 24,746 / 24,797 | 1,436 → 1,262 (−174) | −180 → 180 (+360) | 180 → 60 (−120) | −180 → 0 (+180) | 120 → −60 (−180) |
| **pooled** | 152,271 / 150,266 | 1,267 → 1,451 (+184) | 186,775 / 189,285 | 1,684 → 1,503 (−181) | 240 → 420 (+180) | 720 → 481 (−239) | 180 → 240 (+60) | 420 → 240 (−180) |

Every one of D's quoted medians is reproduced exactly (EDDF/EDDM/LTFM/LSZH/EHAM Jan; EDDF/EDDM/LIRF/LTFM/EGLL Jul;
pooled aobt3−sched 420/240 and 481/720; LTFM aobt3−eobt1 0/−180 and −60/+120).

Pooled `mvt_minus_sched` distribution, matched rows:

| month | n | p10 | p25 | p50 | p75 | p90 | share > 60 min | share > 120 min |
|---|---|---|---|---|---|---|---|---|
| 2025-01 | 152,271 | 500 | 833 | 1,267 | 2,042 | 3,355 | 8.6 % | 1.9 % |
| 2026-01 | 150,266 | 592 | 905 | 1,451 | 2,405 | 4,020 | 12.4 % | 3.0 % |
| 2025-07 | 186,775 | 733 | 1,094 | 1,684 | 2,650 | 4,084 | 13.4 % | 2.5 % |
| 2026-07 | 189,285 | 692 | 1,017 | 1,503 | 2,353 | 3,720 | 10.8 % | 2.1 % |

### 3.2 Proxy (`mvt_minus_aobt3`) median and sd, matched rows

| airport | proxy med Jan 25→26 (Δ) | proxy med Jul 25→26 (Δ) | proxy sd Jan 25→26 (Δ) | proxy sd Jul 25→26 (Δ) |
|---|---|---|---|---|
| EDDF | 907 → 954 (+47) | 896 → 929 (+33) | 330.4 → 375.3 (+44.9) | 256.3 → 215.9 (−40.4) |
| EDDM | 720 → 722 (+2) | 721 → 718 (−3) | 294.0 → 354.8 (+60.9) | 177.1 → 159.0 (−18.1) |
| EGLL | 1,320 → 1,317 (−3) | 1,376 → 1,321 (−55) | 494.1 → 475.7 (−18.4) | 464.8 → 408.4 (−56.4) |
| EHAM | 812 → 863 (+51) | 900 → 964 (+64) | 475.9 → 740.1 (+264.3) | 406.0 → 399.0 (−7.0) |
| LEBL | 758 → 760 (+2) | 767 → 765 (−2) | 283.1 → 289.1 (+6.0) | 329.2 → 301.9 (−27.4) |
| LEMD | 984 → 987 (+3) | 985 → 981 (−4) | 221.1 → 247.0 (+25.9) | 247.9 → 219.5 (−28.4) |
| LFPG | 899 → 897 (−2) | 898 → 959 (+61) | 484.9 → 491.4 (+6.5) | 368.3 → 376.3 (+8.0) |
| LIRF | 902 → 900 (−2) | 901 → 903 (+2) | 208.1 → 218.5 (+10.4) | 750.7 → 737.1 (−13.6) |
| LSZH | 667 → 679 (+12) | 688 → 670 (−18) | 311.4 → 347.6 (+36.2) | 138.4 → 137.8 (−0.6) |
| LTFM | 1,434 → 1,444 (+10) | 1,198 → 1,197 (−1) | 233.4 → 297.7 (+64.4) | 359.5 → 378.3 (+18.8) |
| **pooled** | 964 → 1,001 (+37) | 957 → 965 (+8) | 419.6 → 470.7 (+51.1) | 433.6 → 419.0 (−14.6) |

Proxy-median |Δ| exceeds 50 s in 4 of 20 airport-months (EHAM Jul +64, LFPG Jul +61, EGLL Jul −55, EHAM Jan +51);
the other 16 are within ±47 s (13 within ±12 s). Proxy sd Jan 2026 is higher at 9 of 10 airports (EGLL −18 is the
exception); Jul 2026 sd is lower or flat at 8 of 10 (LFPG +8, LTFM +19).

### 3.3 Congestion counts, headways, taxi-in (same matched rows)

Pooled medians, 2025 → 2026: `dep_tko_prev_30m` Jan 16 → 16, Jul 18 → 18; `dep_tko_prev_60m` 31 → 31, 36 → 36;
`arr_land_prev_30m` 13 → 13, 16 → 16; `arr_inblock_prev_30m` 13 → 13, 16 → 16; `nbr_dep_n_30m` 16 → 16, 18 → 18;
`rwy_headway_prev_s` 122 → 122, 118 → 118; `rwy_headway_next_s` 122 → 122, 118 → 118 (p95 777/779 → 780/780 Jan,
601/601 → 603/602 Jul). Pooled means of the count features differ by ≤ 0.33 (e.g. `dep_tko_prev_60m` Jul 35.41 →
35.73). Per airport, every count-feature median differs by ≤ 2 (largest: EHAM Jan `dep_tko_prev_30m` 20 → 18,
`dep_tko_prev_60m` 38 → 36, `nbr_dep_n_30m` 20 → 18; LTFM Jan `arr_land_prev_30m` 13 → 15); headway medians differ
by ≤ 9 s (LSZH Jan 128 → 137), headway p95 by ≤ 118 s (LTFM Jan next 894 → 776; LIRF Jan prev 844 → 960).

`arr_taxi_in_mean_30m` median: pooled Jan 547.4 → 544.1 (−3.2), Jul 556.6 → 519.2 (−37.4). Per airport Jul:
EHAM 506.0 → 355.7 (−150.3), EDDF 607.1 → 538.1 (−69.0), LFPG 603.8 → 575.3 (−28.5), LTFM 881.9 → 854.9 (−27.1),
EGLL −17.6, LEMD −12.9, EDDM −11.0, LSZH +3.4, LEBL +12.3, LIRF +15.6. Per airport Jan: LIRF 601.7 → 557.8 (−43.9),
LTFM −32.1, LSZH +29.8, EHAM +19.1, EGLL +16.6, EDDM +11.9, LEBL +7.0, LFPG +6.8, EDDF +1.9, LEMD −9.7.

`queue_ahead` median: pooled Jan 12 → 14 (+2), Jul 18 → 17 (−1); LTFM Jan 14 → 19, EDDF Jan 13 → 16, EDDM Jan 8 → 10,
LEMD Jan 13 → 15; Jul EDDF 22 → 18, EGLL 22 → 19, LFPG 22 → 19, EHAM 22 → 20. `nbr_mvt_minus_aobt_mean_30m` median:
pooled +21.7 Jan, +21.2 Jul; LTFM Jan +104.1, EDDF Jan +51.3, EHAM +48.4 / +49.5, LFPG Jul +67.2, others within ±29.

Headway means (not medians) differ strongly between the sets (pooled `rwy_headway_next_s` mean Jan 957 → 5,522;
`rwy_headway_prev_s` Jul 779 → 4,177) because 150–251 matched rows per month in training and 217 (Jan) / 177 (Jul)
in ranking carry a headway > 86,400 s; the maximum in ranking is 17,189,824 s (Jan `next`, Jul `prev` — the
Jan→Jul gap), in training up to 17,807,404 s. Medians and p95 are unaffected. Not part of this item's hypothesis;
recorded because the headway "mean" row of the output is otherwise misleading.

### 3.4 OOF residuals by bin (`oof_main-48c19bc830` ⋈ feat_train, 672,428 rows, RMSE 334.58, mean residual −4.48)

By `mvt_minus_sched` bin:

| bin (min) | all: n | RMSE | mean res | med res | SSE share | matched: n | RMSE | mean res | med res | SSE share | unmatched: n | RMSE | mean res | SSE share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| <−15 | 758 | 296.9 | +5.1 | +4.0 | 0.1 % | 712 | 216.5 | +23.6 | +7.8 | 0.0 % | 46 | 852.5 | −280.5 | 0.0 % |
| −15..0 | 4,800 | 218.2 | +4.8 | −1.8 | 0.3 % | 4,600 | 133.4 | +7.3 | +1.0 | 0.1 % | 200 | 856.4 | −53.0 | 0.2 % |
| 0..15 | 152,715 | 121.7 | −3.1 | −6.9 | 3.0 % | 151,796 | 119.6 | −2.7 | −6.6 | 2.9 % | 919 | 308.0 | −81.5 | 0.1 % |
| 15..30 | 268,937 | 233.4 | −4.4 | −13.2 | 19.5 % | 268,233 | 167.9 | −4.7 | −13.1 | 10.0 % | 704 | 3,172.8 | +78.5 | 9.4 % |
| 30..60 | 175,000 | 292.7 | −5.2 | −24.4 | 19.9 % | 174,624 | 257.0 | −5.6 | −24.4 | 15.3 % | 376 | 3,031.7 | +184.0 | 4.6 % |
| 60..120 | 54,843 | 426.2 | −8.4 | −36.8 | 13.2 % | 50,786 | 387.2 | −5.0 | −35.4 | 10.1 % | 4,057 | 760.6 | −51.4 | 3.1 % |
| >120 | 15,375 | 1,467.7 | +0.5 | −27.1 | 44.0 % | 13,299 | 398.2 | −5.5 | −24.4 | 2.8 % | 2,076 | 3,865.0 | +39.2 | 41.2 % |

By `aobt3_minus_sched` bin (matched rows; `null` = the 8,378 unmatched OOF rows, RMSE 2,296.1, 58.7 % of SSE):

| bin (min) | n | RMSE | mean res | med res | y med | pred med | SSE share | share of matched |
|---|---|---|---|---|---|---|---|---|
| <−15 | 10,351 | 174.6 | −2.9 | −9.2 | 784 | 799 | 0.4 % | 1.6 % |
| −15..0 | 163,742 | 141.5 | −5.1 | −8.4 | 865 | 883 | 4.4 % | 24.7 % |
| 0..15 | 270,804 | 174.7 | −3.5 | −11.6 | 910 | 929 | 11.0 % | 40.8 % |
| 15..30 | 111,468 | 237.0 | −4.2 | −22.0 | 972 | 1,008 | 8.3 % | 16.8 % |
| 30..60 | 71,768 | 318.3 | −6.5 | −34.1 | 1,018 | 1,064 | 9.7 % | 10.8 % |
| 60..120 | 26,115 | 401.5 | −2.8 | −31.8 | 1,020 | 1,067 | 5.6 % | 3.9 % |
| >120 | 9,802 | 392.2 | −9.2 | −25.1 | 1,010 | 1,034 | 2.0 % | 1.5 % |

Matched rows by `mvt_minus_sched` bin × OOF month (mean residual; RMSE):

| bin (min) | 2025-01 | 2025-07 (leaky `jul` fold) | 2025-11 | 2025-12 |
|---|---|---|---|---|
| 0..15 | −0.1; 116.3 (n 42,994) | −4.3; 122.5 (28,907) | −1.5; 120.5 (41,516) | −5.5; 120.2 (38,379) |
| 15..30 | −4.7; 168.7 (60,440) | −4.7; 170.6 (70,763) | −0.9; 168.1 (69,114) | −8.4; 164.2 (67,916) |
| 30..60 | −14.9; 260.3 (33,940) | −3.3; 265.6 (61,205) | +4.1; 251.8 (37,606) | −10.3; 245.9 (41,873) |
| 60..120 | −15.7; 372.2 (10,204) | +4.6; 422.6 (20,347) | +5.2; 348.5 (8,873) | −20.4; 362.0 (11,362) |
| >120 | −12.3; 382.0 (2,911) | −3.8; 447.7 (4,769) | −6.8; 341.1 (2,443) | −0.8; 373.8 (3,176) |

Bin shares within each OOF month (matched): Jan 2025 0..15 = 28.2 %, 30..60 = 22.3 %, ≥ 60 = 8.6 %;
Jul 2025 15.5 % / 32.8 % / 13.5 %; Nov 25.8 % / 23.4 % / 7.0 %; Dec 23.4 % / 25.5 % / 8.8 %.

### 3.5 Bin shares, ranking vs training (matched rows, % of rows)

`mvt_minus_sched`:

| bin (min) | train all 2025 | OOF-scored | rank all 2026 | Δ all | Jan 2025 | Jan 2026 | Δ Jan | Jul 2025 | Jul 2026 | Δ Jul |
|---|---|---|---|---|---|---|---|---|---|---|
| <−15 | 0.1 | 0.1 | 0.1 | −0.0 | 0.1 | 0.1 | −0.0 | 0.1 | 0.1 | +0.0 |
| −15..0 | 0.7 | 0.7 | 0.6 | −0.1 | 1.0 | 0.8 | −0.3 | 0.3 | 0.4 | +0.0 |
| 0..15 | 23.0 | 22.9 | 20.7 | −2.3 | 28.2 | 23.2 | −5.0 | 15.5 | 18.7 | +3.2 |
| 15..30 | 41.5 | 40.4 | 39.7 | −1.7 | 39.7 | 37.2 | −2.5 | 37.9 | 41.7 | +3.8 |
| 30..60 | 26.0 | 26.3 | 27.4 | +1.4 | 22.3 | 26.2 | +4.0 | 32.8 | 28.3 | −4.4 |
| 60..120 | 7.0 | 7.6 | 9.1 | +2.0 | 6.7 | 9.5 | +2.8 | 10.9 | 8.7 | −2.2 |
| >120 | 1.7 | 2.0 | 2.5 | +0.8 | 1.9 | 3.0 | +1.1 | 2.6 | 2.1 | −0.4 |

`aobt3_minus_sched`:

| bin (min) | train all 2025 | OOF-scored | rank all 2026 | Δ all | Jan 2025 | Jan 2026 | Δ Jan | Jul 2025 | Jul 2026 | Δ Jul |
|---|---|---|---|---|---|---|---|---|---|---|
| <−15 | 1.6 | 1.6 | 1.1 | −0.5 | 2.3 | 1.5 | −0.7 | 1.0 | 0.8 | −0.2 |
| −15..0 | 24.4 | 24.7 | 23.1 | −1.3 | 30.2 | 26.2 | −4.0 | 16.6 | 20.7 | +4.1 |
| 0..15 | 41.8 | 40.8 | 40.2 | −1.6 | 40.1 | 38.1 | −2.0 | 38.7 | 41.9 | +3.2 |
| 15..30 | 16.9 | 16.8 | 17.2 | +0.3 | 14.0 | 16.0 | +1.9 | 20.7 | 18.2 | −2.5 |
| 30..60 | 10.4 | 10.8 | 11.8 | +1.4 | 8.7 | 11.3 | +2.5 | 15.5 | 12.3 | −3.2 |
| 60..120 | 3.6 | 3.9 | 4.7 | +1.1 | 3.4 | 4.9 | +1.5 | 5.7 | 4.6 | −1.1 |
| >120 | 1.2 | 1.5 | 1.8 | +0.5 | 1.4 | 2.1 | +0.7 | 1.9 | 1.5 | −0.4 |

All rows (matched + unmatched), `mvt_minus_sched`: train all 22.9 / 41.1 / 25.8 / 7.5 / 1.9 % for 0..15 / 15..30 /
30..60 / 60..120 / >120; rank all 20.5 / 39.3 / 27.0 / 9.6 / 3.0 %; Jan 2026 22.9 / 36.7 / 25.9 / 9.9 / 3.6 %
(Jan 2025 28.1 / 39.4 / 22.1 / 7.1 / 2.1 %); Jul 2026 18.5 / 41.2 / 28.0 / 9.3 / 2.4 % (Jul 2025 15.3 / 37.2 / 32.2 /
11.8 / 3.1 %).

### 3.6 `data/work/adv_score.parquet`

Schema: `{MVT_ID_mvt: Float64, adv: Float64}`; 2,085,047 rows, `MVT_ID_mvt` unique, all join to feat_train, none
to feat_rank; `adv` min 0.000043, max 0.955, mean 0.112, median 0.097, no nulls. `src/prc/adversarial.py:main()`
writes no file; the file (mtime 2026-09-14 11:15) has no producer in the repository, so the column is taken to be
the per-row P(ranking) described in that module's docstring but this is not verifiable here.

Mean `adv` by bin: `mvt_minus_sched`, matched: <−15 0.095, −15..0 0.097, 0..15 0.106, 15..30 0.110, 30..60 0.114,
60..120 0.131, >120 0.139; unmatched: 0.130, 0.131, 0.123, 0.126, 0.130, 0.138, 0.152. `aobt3_minus_sched`
(matched): <−15 0.086, −15..0 0.110, 0..15 0.110, 15..30 0.111, 30..60 0.120, 60..120 0.133, >120 0.138; null
(unmatched) 0.138. Share with `adv` > 0.5 is ≤ 1.3 % in every bin (matched ≤ 0.9 %). By month: Jan 0.138, Feb 0.115,
Mar–Oct 0.099–0.108, Nov 0.132, Dec 0.148. By airport (matched): 0.090 (LFPG) – 0.133 (LEMD); unmatched 0.110–0.171.

### 3.7 Re-weighting arithmetic (matched rows; per-bin OOF MSE held fixed)

Matched OOF RMSE with OOF bin weights 216.44 → 223.44 with the ranking `mvt_minus_sched` bin weights (+7.0 s),
→ 221.92 with the ranking `aobt3_minus_sched` bin weights (+5.5 s). Mean residual −4.38 → −4.47 / −4.40.

## 4. Verdict per sub-claim

| sub-claim | verdict | basis |
|---|---|---|
| Matched ms median up in Jan 2026 at EDDF/EDDM/LTFM/LSZH/EHAM, down in Jul 2026 at EDDF/EDDM/LIRF/LTFM/EGLL (D's ten figures) | **CONFIRMED** — all ten reproduced exactly (§3.1). Refinement (fact): Jan rise is not airport-wide — EGLL +5, LFPG +5, LIRF +1, LEBL +54.5; Jul fall is at all 10 airports (−89 … −292); pooled +184 / −181 |
| `aobt3_minus_sched` median Jan 420 vs 240, Jul 481 vs 720 | **CONFIRMED** (pooled; per airport Jan Δ 0 … +360, Jul Δ −60 … −360) |
| `aobt3_minus_eobt1` LTFM Jan 0 vs −180, Jul −60 vs +120 | **CONFIRMED**; pooled Jan 180 → 240, Jul 420 → 240 |
| "shifting every SCHED/EOBT-anchored feature" | **CONFIRMED** for the three features checked, at the pooled level and in the same direction per month; not checked: `mvt_minus_eobt1`, `mvt_minus_lobt`, `mvt_minus_iobt`, `lobt_minus_eobt1` |
| Proxy medians stable (≤ 50 s) | **CONFIRMED in substance, threshold REFUTED as stated**: 4 of 20 airport-months exceed 50 s (max +64 EHAM Jul); pooled +37 / +8 |
| Proxy sd higher in Jan 2026 at EHAM (740 vs 476), EDDF, EDDM, LSZH | **CONFIRMED** (740.1 / 475.9; +44.9 / +60.9 / +36.2); also LTFM +64.4, LEMD +25.9; EGLL −18.4; pooled +51.1. Jul 2026 sd lower or flat at 8 of 10 |
| Congestion counts: no shift | **CONFIRMED** — pooled medians identical; per-airport median Δ ≤ 2; pooled mean Δ ≤ 0.33 |
| Headways: no shift | **CONFIRMED** on medians (identical pooled; ≤ 9 s per airport) and p95 (pooled ≤ 3 s). Means are not comparable (cross-gap values > 1 day in both sets, §3.3) |
| Taxi-in means: no shift | **PARTIALLY REFUTED** — pooled Jul −37 s; EHAM Jul −150 s and EDDF Jul −69 s at the median; Jan pooled −3 s; 18 of 20 airport-months within ±45 s |
| Ranking bin composition differs from training | **CONFIRMED** — Jan 2026 moves 7.9 pp of matched rows into ms ≥ 30 min (vs Jan 2025); Jul 2026 moves 7.0 pp out; all-2026 vs all-2025 +4.2 pp into ≥ 30 min |
| OOF error depends on the bin | **CONFIRMED** — matched RMSE 119.6 (0..15) → 167.9 → 257.0 → 387.2 → 398.2 (>120); by aobt3 bin 141.5 → 392.2. Mean residual per bin within −9.2 … +7.3 s (bins with n ≥ 4,600); per month the same bin's mean residual ranges over ~20–25 s (30..60: −14.9 Jan, +4.1 Nov; 60..120: −20.4 Dec, +5.2 Nov) |
| `adv` score differs by bin | **CONFIRMED as a gradient** — matched mean 0.095 → 0.139 across ms bins, 0.086 → 0.138 across aobt3 bins; smaller than the month range (0.099–0.148) and no bin has > 1.3 % of rows scored > 0.5 |
| Effect on predictions / LB | **UNDETERMINED** — see §5 |

## 5. Not determinable locally

- Whether per-bin error in 2026 equals per-bin OOF error in 2025 (needs ranking labels). The +7.0 s in §3.7 is
  arithmetic under that assumption and is not a measurement.
- Whether the Jan 2026 shift toward high-error bins and the Jul 2026 shift away from them net out on the LB
  (Jul is 55.7 % of ranking rows; the per-bin MSE differences are largest where Jan gains share).
- Whether the ms shift is a change in operations (delay) or in the `SCHED_TIME` source (e.g. schedule granularity);
  `aobt3_minus_sched` and `aobt3_minus_eobt1` moving together with `mvt_minus_sched` while the proxy is stable is
  consistent with either.
- Provenance of `adv_score.parquet` (no producer in the repository), and whether the model used calendar columns.
- Whether the month-dependent sign of the bin residual (Jan negative, Nov positive) is a property of the forward
  folds' training windows or of the months themselves (the 2025-07 column is the leaky `jul` fold per
  `results/fold-shape-vs-ranking.md`).

## 6. Facts vs inference

**Facts (measured on the files named above):**
- All 14 numbers D quotes for this item are reproduced exactly (ten ms medians, two pooled `aobt3_minus_sched`
  medians, two LTFM `aobt3_minus_eobt1` medians, and the EHAM proxy sd pair 740 / 476 to the nearest second).
- Pooled matched `mvt_minus_sched` median: Jan 1,267 → 1,451; Jul 1,684 → 1,503. Share > 60 min: Jan 8.6 → 12.4 %,
  Jul 13.4 → 10.8 %. Share > 120 min: Jan 1.9 → 3.0 %, Jul 2.5 → 2.1 %.
- Proxy median Δ per airport-month: 16 of 20 within ±47 s, 4 between 51 and 64 s; pooled +37 (Jan), +8 (Jul).
- Proxy sd Δ: Jan +6 … +264 at 9 airports, EGLL −18; Jul −56 … +19.
- Count features and headway medians: pooled identical; per-airport median Δ ≤ 2 counts / ≤ 9 s.
- `arr_taxi_in_mean_30m` median: EHAM Jul −150.3, EDDF Jul −69.0, pooled Jul −37.4, pooled Jan −3.2.
- `queue_ahead` median pooled Jan 12 → 14, Jul 18 → 17; `nbr_mvt_minus_aobt_mean_30m` median pooled +21.7 / +21.2.
- Headway values > 86,400 s: 150–251 matched rows per training month; 217 / 177 in Jan / Jul 2026; ranking maximum
  17,189,824 s.
- OOF matched RMSE by ms bin rises monotonically 119.6 → 398.2; the >120 bin is 2.0 % of matched rows and 2.8 % of
  matched SSE; for all rows the >120 bin is 2.3 % of rows and 44.0 % of SSE, of which the 2,076 unmatched rows
  contribute 41.2 points.
- Mean residual per bin (matched, n ≥ 4,600): −2.7 … −5.6 (ms), −2.8 … −9.2 (aobt3). Median residual grows more
  negative with the bin: −6.6 (0..15) → −35.4 (60..120).
- Ranking matched bin shares vs training (§3.5); OOF-scored shares are within 1.1 pp of all-2025 shares (15..30: 40.4 vs 41.5).
- `adv_score.parquet`: schema, coverage and per-bin means as in §3.6.
- Re-weighted matched RMSE 216.44 → 223.44 / 221.92 (§3.7) is exact arithmetic on the measured per-bin MSEs.

**Inference (not measured):**
- Calling the ms shift a "delay level" change presumes `SCHED_TIME` semantics are unchanged between years; the data
  are equally consistent with a schedule-field change, since the proxy and the congestion counts do not move.
- The +7.0 s re-weighting figure describes what the matched OOF RMSE would have been had the 2025 rows carried the
  2026 bin mix with unchanged per-bin error; it says nothing about the actual 2026 error and ignores the
  month-dependence of the bin residuals shown in §3.4.
- The mild `adv` gradient across bins is read as "the classifier uses delay-anchored features a little", but the
  file's producer and feature set are not known, so this stays an inference.
- The Jul 2026 taxi-in drop at EHAM/EDDF is outside this item's hypothesis; whether it relates to the EHAM Jan
  disruption item (#15) or to layout changes is not examined here.
