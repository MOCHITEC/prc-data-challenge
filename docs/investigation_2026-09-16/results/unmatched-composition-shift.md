# Item 17 `unmatched-composition-shift` — verification result (2026-09-16)

Script: `docs/investigation_2026-09-16/scripts/unmatched-composition-shift.py` (no training, `POLARS_MAX_THREADS=1`,
10-column scans of `data/work/feat_train.parquet` / `feat_rank.parquet`, `experiments/oof_main-48c19bc830.parquet`; ~3 min).
Definitions follow plan §0. Both "unmatched" definitions are computed throughout: **routing** = `mvt_minus_aobt3` null
(22,470 / 5,290 rows); **flightid** = `FLIGHT_ID_mvt` null (22,424 / 5,283). Numbers below are the routing definition
unless marked; the flightid definition differs by ≤ 0.05 pp on every share and ≤ 3 s on every RMSE (full tables in the
script output). `y` = `TAXITIME_SEC_mvt`; 2 h = 7,200 s. "seen" = the row's `FLIGHT_mvt` occurs among the `FLIGHT_mvt`
values of matched training departures (D's definition, q3 G; 44,861 distinct flight numbers). Prefix = first 2 / 3
characters of `FLIGHT_mvt`; hour = UTC hour of `MVT_TIME_UTC_mvt`. Sets: `train_all` (2025, 12 months), `train_JanJul25`
(Jan + Jul 2025 = same calendar months as the ranking set), `rank_Jan26`, `rank_Jul26`, `rank_all`.

## 1. Hypothesis (plan §1 #17, from D and P 8)

Ranking unmatched rows are a different population from training unmatched rows: (a) higher unmatched rate (1.53 % vs
1.08 %); (b) EHAM/LSZH-heavy by airport; (c) KLM-heavy by flight prefix (KL 7.4 vs 3.0 %), TH and HB lighter;
(d) fewer VFR (FLIGHT_RULE V 7.9 vs 13.1 %); (e) fewer "flights seen among matched training flights" (69.7 vs 81.9 %);
(f) hour distribution and null pattern near-identical; (g) training label by "seen": median 962 / y ≥ 2 h 2.08 % (seen)
vs 846 / 2.27 % (unseen). Unverified in the sources: the effect on model B error. Extension asked by the plan: reweight
unmatched OOF SSE per (airport × month) cell to the ranking composition.

## 2. Procedure

1. Preconditions: PHASE, month keys, `MVT_ID_mvt` uniqueness, label nulls (ranking labels are null on all 344,841 rows).
2. Unmatched counts and rates for the five sets under both definitions.
3. Composition of unmatched rows (share %) by airport, prefix-2, prefix-3, FLIGHT_RULE, hour, seen, AIRCRAFT_TYPE null
   for all five sets; total-variation distance (TV = ½ Σ|p − q|) between sets per dimension; unmatched rate per
   airport (unmatched / departures at that airport); per airport × month counts Jan25/Jul25 vs Jan26/Jul26.
4. "seen" decomposed: by airport × month; against matched *ranking* flights as the reference set; and the same
   flight-number-overlap statistic on *matched* rows (to test whether the drop is specific to unmatched rows).
5. Training label (y median, share y ≥ 2 h, n ≥ 2 h / ≥ 4 h, mean) by every subgroup of unmatched training rows;
   composition-implied label statistics = training per-cell stats weighted by ranking cell shares.
6. OOF: `oof_main-48c19bc830` joined on `MVT_ID_mvt` to unmatched training rows (8,378 rows; months 1, 7, 11, 12).
   Cells (airport × month) restricted to OOF months 1 and 7 (the ranking calendar months); expected unmatched RMSE
   under ranking composition = sqrt(Σ_c w_c^rank · MSE_c^OOF). **Assumption stated:** the per-cell MSE on ranking rows
   equals the per-cell OOF MSE; only the cell weights change. Variants: airport-only cells (all 4 OOF months),
   month-only cells, cells after dropping the top-10 / 20 / 50 OOF unmatched rows by squared error, (airport × month
   × seen) cells, and the label-only analogue (within-cell label SD reweighted).

## 3. Results

### 3.1 Unmatched rate (both definitions)

| set | rows | unmatched (routing) | % | unmatched (flightid) | % |
|---|---|---|---|---|---|
| train_all | 2,085,047 | 22,470 | 1.078 | 22,424 | 1.075 |
| train_JanJul25 | 344,419 | 5,373 | 1.560 | 5,366 | 1.558 |
| rank_Jan26 | 152,719 | 2,453 | 1.606 | 2,449 | 1.604 |
| rank_Jul26 | 192,122 | 2,837 | 1.477 | 2,834 | 1.475 |
| rank_all | 344,841 | 5,290 | 1.534 | 5,283 | 1.532 |

Same-calendar-month training rate is 1.560 % vs ranking 1.534 % (matches sibling report #26 §3.4).

### 3.2 Composition of unmatched rows (share % of unmatched rows in the set; routing definition)

**Airport** (unmatched-rate per airport = unmatched / departures at that airport, in the second block):

| ADEP | train_all | train_JanJul25 | rank_Jan26 | rank_Jul26 | rank_all | Δ vs all (pp) | Δ vs JanJul25 (pp) | rate train_all | rate JanJul25 | rate Jan26 | rate Jul26 | rate rank_all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EHAM | 18.18 | 15.04 | 32.82 | 14.13 | 22.80 | +4.62 | +7.76 | 1.647 | 1.973 | 4.959 | 1.827 | 3.159 |
| LFPG | 16.79 | 16.38 | 13.25 | 14.42 | 13.88 | −2.92 | −2.50 | 1.575 | 2.222 | 1.793 | 1.881 | 1.841 |
| LTFM | 16.22 | 12.23 | 13.58 | 13.68 | 13.63 | −2.59 | +1.40 | 1.335 | 1.412 | 1.487 | 1.541 | 1.515 |
| LSZH | 10.20 | 9.75 | 12.23 | 12.65 | 12.46 | +2.26 | +2.71 | 1.698 | 2.345 | 3.002 | 2.729 | 2.847 |
| LEBL | 8.29 | 9.23 | 3.14 | 12.87 | 8.36 | +0.07 | −0.88 | 1.036 | 1.711 | 0.631 | 2.042 | 1.469 |
| EDDF | 8.70 | 11.15 | 8.64 | 7.37 | 7.96 | −0.75 | −3.19 | 0.850 | 1.626 | 1.353 | 1.012 | 1.159 |
| LIRF | 6.62 | 7.39 | 4.36 | 9.73 | 7.24 | +0.62 | −0.15 | 0.926 | 1.497 | 0.967 | 1.743 | 1.424 |
| EGLL | 6.29 | 8.67 | 4.57 | 4.79 | 4.69 | −1.60 | −3.98 | 0.590 | 1.159 | 0.581 | 0.662 | 0.622 |
| EDDM | 4.49 | 4.41 | 4.85 | 4.19 | 4.50 | 0.00 | +0.09 | 0.604 | 0.875 | 1.086 | 0.793 | 0.917 |
| LEMD | 4.21 | 5.75 | 2.57 | 6.17 | 4.50 | +0.28 | −1.25 | 0.446 | 0.875 | 0.375 | 0.868 | 0.644 |

Counts per airport × month (train Jan25 / Jul25 → rank Jan26 / Jul26; ratio): EDDF 164/435 → 212/209 (1.29 / 0.48);
EDDM 65/172 → 119/119 (1.83 / 0.69); EGLL 128/338 → 112/136 (0.88 / 0.40); EHAM 293/515 → 805/401 (2.75 / 0.78);
LEBL 66/430 → 77/365 (1.17 / 0.85); LEMD 32/277 → 63/175 (1.97 / 0.63); LFPG 216/664 → 325/409 (1.50 / 0.62);
LIRF 60/337 → 107/276 (1.78 / 0.82); LSZH 179/345 → 300/359 (1.68 / 1.04); LTFM 232/425 → 333/388 (1.44 / 0.91).
Every airport except LSZH has fewer unmatched rows in Jul 2026 than Jul 2025; every airport except EGLL has more in
Jan 2026 than Jan 2025. EHAM's excess is Jan 2026 only (805; Jul 2026 401 < Jul 2025 515).

**Prefix-2** (top by rank_all): TH 14.59 / 11.61 / 10.48 / 11.07 / **10.79**; KL 2.97 / 3.22 / **15.17** / 0.60 / **7.35**;
AF 6.33 / 6.37 / 6.16 / 4.37 / 5.20; SW 2.10 / 1.92 / 4.40 / 3.21 / 3.76; RY 2.98 / 4.04 / 0.94 / 4.93 / 3.08;
HV 1.42 / 1.51 / 2.53 / 2.86 / 2.70; HB 4.49 / 3.57 / 2.20 / 3.07 / **2.67**; PG 0.99 / 0.35 / 2.94 / 2.36 / 2.63;
DA 2.81 / 2.27 / 2.12 / 2.82 / 2.50; ZX 3.77 / 2.23 / 2.41 / 2.33 / 2.36; LH 2.71 / 3.80 / 3.18 / 1.66 / 2.36
(order: train_all / train_JanJul25 / rank_Jan26 / rank_Jul26 / rank_all).
KL rows among ranking unmatched: 389, of which 331 are EHAM Jan 2026 and 12 EHAM Jul 2026. KL share of EHAM unmatched
rows: all-train 12.3 %, Jan 2025 19.8 % (n 293), **Jan 2026 41.1 % (n 805)**. KL share of EHAM *matched* departures:
train 51.7 %, Jan 2026 52.5 %, Jul 2026 51.3 % (unchanged).

**Prefix-3** (top by rank_all): THY 14.48 / 11.56 / 10.40 / 10.96 / 10.70; AFR 6.09 / 6.16 / 5.71 / 4.34 / 4.97;
KL1 1.87 / 1.94 / 9.05 / 0.39 / 4.40; SWR 2.04 / 1.86 / 4.24 / 3.00 / 3.57; RYR 2.98 / 4.04 / 0.94 / 4.93 / 3.08;
PGT 0.99 / 0.35 / 2.94 / 2.36 / 2.63; ZXP 3.77 / 2.23 / 2.41 / 2.33 / 2.36; KL0 0.43 / 0.39 / 4.69 / 0.04 / 2.19;
EJU 2.19 / 1.92 / 1.75 / 2.08 / 1.93; WMT 1.00 / 1.47 / 0.77 / 1.90 / 1.38; BAW 1.71 / 3.18 / 1.06 / 1.23 / 1.15.

**FLIGHT_RULE**: I 86.78 / 90.69 / 92.62 / 91.33 / **91.93**; V 13.04 / 9.14 / 7.09 / 8.64 / **7.92**; null 0.18 / 0.17 /
0.29 / 0.04 / 0.15 (train 40 rows, rank 8).

**Hour (UTC)**: largest |Δ| rank_all − train_all = 0.89 pp (hour 6), rank_all − train_JanJul25 = 1.38 pp (hour 16); all
24 shares within ±1.4 pp.

**seen**: true 81.66 / 86.97 / 82.63 / **58.05** / **69.45**; false 18.05 / 12.94 / 17.04 / 41.42 / 30.11; FLIGHT_mvt null
0.29 / 0.09 / 0.33 / 0.53 / 0.43. Flightid definition: 81.87 / — / — / — / 69.73 (D's 81.9 / 69.7 reproduced).

**AIRCRAFT_TYPE null**: 6.71 / 7.48 / 4.40 / 9.87 / 7.33 (1,507 train rows, 388 rank rows; per sibling #3, 1,487 and 383
of these are LIRF).

**Total-variation distance** between compositions:

| dimension | rank_all vs train_all | rank_all vs train_JanJul25 | rank_Jan26 vs train_all | rank_Jul26 vs train_all | rank_Jan26 vs rank_Jul26 | train_JanJul25 vs train_all |
|---|---|---|---|---|---|---|
| airport | 0.079 | 0.120 | 0.170 | 0.121 | 0.206 | 0.081 |
| prefix-2 | 0.197 | 0.201 | 0.276 | 0.233 | 0.308 | 0.136 |
| prefix-3 | 0.242 | 0.253 | 0.328 | 0.290 | 0.374 | 0.161 |
| FLIGHT_RULE | 0.053 | 0.014 | 0.061 | 0.046 | 0.016 | 0.041 |
| hour | 0.048 | 0.066 | 0.090 | 0.068 | 0.128 | 0.070 |
| seen | 0.125 | 0.176 | 0.013 | 0.239 | 0.249 | 0.054 |
| actype null | 0.006 | 0.002 | 0.023 | 0.032 | 0.055 | 0.008 |

For every dimension the Jan 2026 vs Jul 2026 distance is larger than the rank-vs-train distance; the within-training
seasonal distance (JanJul25 vs all) is 0.04–0.16.

### 3.3 The "seen" gap decomposed

Seen rate of unmatched rows by airport × month (seen among matched *training* flights; last two columns: seen among
matched *ranking* flights):

| ADEP | train Jan25 | train Jul25 | rank Jan26 | rank Jul26 | rank Jan26 (ref = ranking matched) | rank Jul26 (ref = ranking matched) |
|---|---|---|---|---|---|---|
| EDDF | 95.1 | 97.0 | 96.2 | 88.5 | 94.3 | 96.2 |
| EDDM | 96.9 | 93.6 | 95.0 | 51.3 | 92.4 | 93.3 |
| EGLL | 99.2 | 99.7 | 100.0 | 76.5 | 99.1 | 97.1 |
| EHAM | 90.4 | 93.2 | 95.2 | 81.3 | 92.5 | 89.8 |
| LEBL | 89.4 | 96.5 | 97.4 | 47.9 | 89.6 | 96.4 |
| LEMD | 96.9 | 98.6 | 95.2 | 45.7 | 95.2 | 94.9 |
| LFPG | 99.1 | 99.2 | 97.2 | 56.2 | 95.1 | 97.1 |
| LIRF | 81.4 | 93.1 | 76.8 | 35.2 | 76.8 | 87.7 |
| LSZH | 52.0 | 72.5 | 74.7 | 70.2 | 72.7 | 72.1 |
| LTFM | 59.1 | 40.7 | 24.3 | 36.6 | 38.1 | 17.8 |

Training unmatched seen-rate by month: 72.0–88.5 % (min Nov, max Jul); Jan 2026 82.9 %, Jul 2026 58.4 %.

Reference-set symmetry: ranking unmatched rows seen among matched **ranking** flights = **81.66 %** (training within-set
rate 81.90 %); training unmatched rows seen among matched **ranking** flights = 60.13 %; ranking unmatched seen among
matched training-or-ranking flights = 85.72 %.

The same overlap statistic on **matched** rows: matched ranking rows whose `FLIGHT_mvt` appears among matched training
flights = 78.85 % overall, **Jan 2026 98.05 %, Jul 2026 63.61 %**; matched training rows whose flight appears among
matched ranking flights = 68.38 % (Jan 2025 61.60 %, Jul 2025 63.16 %).

### 3.4 Training label by subgroup (unmatched training rows, n 22,470; y median 950.5, y ≥ 2 h 2.141 %, mean 1,384, RMS 3,669)

| subgroup | n_train | y median | y ≥ 2 h % | n ≥ 2 h | n ≥ 4 h | share train % | share rank % |
|---|---|---|---|---|---|---|---|
| LIRF | 1,488 | 3,922 | 29.50 | 439 | 115 | 6.62 | 7.24 |
| EGLL | 1,414 | 1,384 | 0.78 | 11 | 0 | 6.29 | 4.69 |
| LTFM | 3,645 | 1,009 | 0.52 | 19 | 0 | 16.22 | 13.63 |
| LFPG | 3,773 | 1,023 | 0.19 | 7 | 2 | 16.79 | 13.88 |
| LSZH | 2,291 | 673 | 0.17 | 4 | 2 | 10.20 | 12.46 |
| EHAM | 4,084 | 538.5 | 0.02 | 1 | 0 | 18.18 | 22.80 |
| EDDF / EDDM / LEBL / LEMD | 1,956 / 1,010 / 1,862 / 947 | 891–1,096 | 0.00 | 0 | 0 | 25.7 | 25.3 |
| rule I | 19,500 | 1,009 | 2.45 | 477 | 118 | 86.78 | 91.93 |
| rule V | 2,930 | 232 | 0.14 | 4 | 1 | 13.04 | 7.92 |
| seen = true | 18,349 | 962 | 2.07 | 380 | 82 | 81.66 | 69.45 |
| seen = false | 4,056 | 846 | 2.27 | 92 | 32 | 18.05 | 30.11 |
| FLIGHT_mvt null | 65 | 1,915 | 13.85 | 9 | 5 | 0.29 | 0.43 |
| actype null = true | 1,507 | 2,643 | 29.13 | 439 | 115 | 6.71 | 7.33 |
| actype null = false | 20,963 | 912 | 0.20 | 42 | 4 | 93.29 | 92.67 |
| prefix-2 KL | 667 | 747 | 0.15 | 1 | 0 | 2.97 | 7.35 |
| prefix-2 TH | 3,278 | 1,017 | 0.55 | 18 | 1 | 14.59 | 10.79 |
| prefix-2 HB | 1,010 | 602 | 0.20 | 2 | 1 | 4.49 | 2.67 |
| prefix-2 RY | 669 | 1,195 | 5.83 | 39 | 7 | 2.98 | 3.08 |
| prefix-3 WMT | 224 | 1,361 | 22.32 | 50 | 8 | 1.00 | 1.38 |
| prefix-2 ZX | 848 | 174 | 0.00 | 0 | 0 | 3.77 | 2.36 |
| hour 23 / 0 / 1 | 247 / 124 / 58 | 1,141 / 1,180 / 1,622 | 11.3 / 11.3 / 20.7 | 28 / 14 / 12 | 11 / 8 / 4 | 1.10 / 0.55 / 0.26 | 1.36 / 0.68 / 0.19 |
| hours 6–19 | 501–1,859 each | 775–990 | 0.9–2.5 | — | — | 82.7 | 82.0 |

D's "seen" figures reproduced: 962 / 2.08 % vs 846 / 2.27 % (this run 962 / 2.07 vs 846 / 2.27).

Composition-implied y ≥ 2 h share (training per-cell rate × ranking cell share; training mix = 2.141 %): airport 2.297 %,
prefix-2 2.077 %, prefix-3 2.006 %, FLIGHT_RULE 2.263 %, hour 2.166 %, seen 2.131 %, actype null 2.322 %. Implied mean y
(training 1,384): 1,371 / 1,377 / 1,356 / 1,446 / 1,388 / 1,357 / 1,419. Airport × month (Jan, Jul) cells: training
Jan+Jul 2025 rate 2.066 % → 1.991 % under the ranking mix, i.e. ≈ 105 rows with y ≥ 2 h expected among 5,290 (LIRF
cells: Jan 21.67 % of 60, Jul 24.93 % of 337; every non-LIRF cell ≤ 2.07 %).

### 3.5 OOF reweighting (`oof_main-48c19bc830`, unmatched rows; routing definition)

OOF unmatched: 8,378 rows, RMSE 2,296.1 (months 1 / 7 / 11 / 12: n 1,435 / 3,938 / 1,356 / 1,649; RMSE 3,223.0 /
2,008.8 / 2,345.5 / 1,893.1; SSE share 33.8 / 36.0 / 16.9 / 13.4 %). Months 1 + 7 only: 5,373 rows, RMSE 2,394.2.
(Per sibling #26, the month-7 predictions are the leaky `jul` fold's.)

(airport × month) cells, months 1 and 7:

| mo | ADEP | n_oof | n_rank | w_oof | w_rank | cell RMSE | n y≥2h | top-1 row share of cell SSE | MSE·w_rank share % |
|---|---|---|---|---|---|---|---|---|---|
| 1 | LFPG | 216 | 325 | 0.040 | 0.061 | 6,905.5 | 2 | 0.674 | 42.55 |
| 7 | LIRF | 337 | 276 | 0.063 | 0.052 | 6,341.6 | 84 | 0.170 | 30.47 |
| 1 | LIRF | 60 | 107 | 0.011 | 0.020 | 7,982.3 | 13 | 0.713 | 18.72 |
| 1 | EHAM | 293 | 805 | 0.055 | 0.152 | 893.1 | 1 | 0.505 | 1.76 |
| 7 | LEBL | 430 | 365 | 0.080 | 0.069 | 1,077.2 | 0 | 0.472 | 1.16 |
| 7 | LSZH | 345 | 359 | 0.064 | 0.068 | 991.4 | 1 | 0.445 | 0.97 |
| 1 | EDDF | 164 | 212 | 0.031 | 0.040 | 1,239.2 | 0 | 0.325 | 0.89 |
| 7 | EGLL | 338 | 136 | 0.063 | 0.026 | 1,299.2 | 7 | 0.164 | 0.63 |
| other 12 cells | 32–664 | 63–409 | | | 286–949 | 0–1 | 0.17–0.59 | ≤ 0.6 each |

Expected unmatched RMSE:

| cells | reference (OOF composition) | under ranking composition | ratio |
|---|---|---|---|
| **(airport × month), months 1+7** | **2,394.2** | **2,624.0** | **1.096** |
| airport only, all 4 OOF months | 2,296.1 | 2,317.0 | 1.009 |
| month only (1, 7) | 2,394.2 | 2,642.2 | 1.104 |
| (airport × month) after dropping top-10 OOF unmatched rows | 1,275.8 | 1,267.2 | 0.993 |
| … after dropping top-20 | 1,152.7 | 1,118.2 | 0.970 |
| … after dropping top-50 | 979.3 | 979.6 | 1.000 |
| (airport × month × seen), months 1+7 (23 ranking rows in cells with no OOF rows, excluded) | 2,394.2 | 2,520.4 | 1.053 |
| label-only analogue: within-cell label SD, train Jan+Jul25 → ranking mix | 3,710.6 | 3,909.1 | 1.053 |

Flightid definition: 2,395.5 → 2,623.9 (1.095); 2,298.6 → 2,318.3 (1.009); 1,276.3 → 1,267.7; 1,153.0 → 1,118.6;
979.4 → 979.9; 2,395.5 → 2,519.9 (1.052).

OOF unmatched RMSE by subgroup (all 4 OOF months): seen = true 2,205.3 (n 6,911; y med 1,015, pred med 1,091),
seen = false 2,669.6 (n 1,450; 851.5 / 951), FLIGHT_mvt null 3,631.0 (n 17); actype null 7,120.5 (n 546) vs 1,450.8
(n 7,832); FLIGHT_RULE V 602.4 (n 860) vs I 2,417.6 (n 7,503).

## 4. Verdict per sub-claim

| sub-claim | verdict | basis |
|---|---|---|
| (a) unmatched rate 1.53 % vs 1.08 % | **CONFIRMED as stated; REFUTED as a population difference** | 1.534 vs 1.078 (routing) / 1.532 vs 1.075 (flightid). Same-calendar-month training rate is 1.560 %; Jan 2026 is above Jan 2025 (1.606 vs 0.934 %) and Jul 2026 below Jul 2025 (1.477 vs 2.065 %). The aggregate gap is the Jan/Jul month mix of the ranking set (#26 §3.4 reproduced). |
| (b) EHAM/LSZH-heavy by airport | **CONFIRMED for EHAM Jan 2026 and LSZH; REFUTED as a whole-ranking-set property** | EHAM 22.80 vs 18.18 % (vs JanJul25 15.04); the excess is entirely Jan 2026 (805 rows, 32.8 % of Jan unmatched, rate 4.96 %); Jul 2026 EHAM (401, 14.1 %, 1.83 %) is below Jul 2025 (515). LSZH 12.46 vs 10.20 (vs JanJul25 9.75), rate 2.85 vs 1.70 (vs 2.35), both months. LFPG/LTFM/EGLL/EDDF shares are lower. Airport TV distance rank vs train 0.079, smaller than Jan26 vs Jul26 (0.206) and about equal to the training's own Jan+Jul vs all-year distance (0.081). Per-airport rates D quotes (3.16/1.65, 2.85/1.70, 1.47/1.04, 1.42/0.93, 1.16/0.85, 0.64/0.45) reproduced; against the same months (JanJul25) only EHAM (3.16 vs 1.97), LSZH (2.85 vs 2.35), LTFM (1.52 vs 1.41) and EDDM (0.92 vs 0.88) are higher, the other six are lower. |
| (c) KL-heavy; TH, HB lighter | **CONFIRMED (KL 7.35 vs 2.97; TH 10.79 vs 14.59; HB 2.67 vs 4.49); the KL excess is one cell** | 331 of 389 KL ranking unmatched rows are EHAM Jan 2026; Jul 2026 KL share 0.60 %. KL is 41.1 % of EHAM Jan 2026 unmatched vs 19.8 % Jan 2025 / 12.3 % all-train, while KL's share of EHAM matched departures is unchanged (51.7 / 52.5 / 51.3 %). TH vs same months 11.61 → 10.79 (−0.8 pp); HB 3.57 → 2.67. Prefix TV distances rank vs train (0.20 / 0.24) are smaller than Jan26 vs Jul26 (0.31 / 0.37). |
| (d) fewer VFR (V 7.9 vs 13.1 %) | **CONFIRMED vs all-year; mostly seasonal** | 7.92 vs 13.04 %; vs Jan+Jul 2025 9.14 % (TV 0.014). |
| (e) fewer "seen among matched training flights" (69.7 vs 81.9 %) | **CONFIRMED as measured; REFUTED as a property of the unmatched population** | 69.45 vs 81.66 (routing) / 69.73 vs 81.87 (flightid). Jan 2026 82.6 % (= training); Jul 2026 58.1 %. The same statistic on **matched** ranking rows is 98.05 % (Jan 2026) and 63.61 % (Jul 2026): the Jul 2026 flight-number overlap with 2025 is low for matched and unmatched rows alike. Against the ranking set's own matched flights, ranking unmatched rows are seen at 81.66 %, the training within-set rate. The drop is concentrated at LIRF (35.2 %), LEMD (45.7), LEBL (47.9), EDDM (51.3), LFPG (56.2) in Jul 2026, each ≥ 93 % in Jul 2025. |
| (f) hour and null pattern near-identical | **CONFIRMED** | hour TV 0.048 (rank vs train_all), all 24 shares within ±1.4 pp; actype-null 7.33 vs 6.71 % (TV 0.006); FLIGHT_RULE null 0.15 vs 0.18 %; FLIGHT_mvt null 0.43 vs 0.29 %. |
| (g) label by seen: 962 / 2.08 % vs 846 / 2.27 % | **CONFIRMED** | 962 / 2.071 vs 846 / 2.268 (n 18,349 / 4,056). Unseen rows have lower median and near-equal tail rate; OOF RMSE unseen 2,670 vs seen 2,205. |
| "different population" (overall) | **REFUTED on the measured dimensions** | every rank-vs-train TV distance is below the Jan26-vs-Jul26 distance and, for airport / hour / rule / actype, within 1–2× the training's own Jan+Jul-vs-all-year distance; the two large deviations (EHAM-KL Jan 2026; seen in Jul 2026) are each one cell / one mechanism (§3.2–3.3). |
| Extension: effect of composition on unmatched RMSE | **UNDETERMINED in sign; magnitude bounded** | reweighted expected unmatched RMSE 2,624 vs 2,394 (+9.6 %) under the stated assumption, but 92 % of the expected MSE sits in three cells (LFPG-Jan, LIRF-Jul, LIRF-Jan) whose MSE is set by 1–13 rows (top-1 row = 67 % and 71 % of cell SSE in LFPG-Jan and LIRF-Jan); month-only reweighting alone gives +10.4 %, airport-only 0.9 %; after removing the top-10 rows the ratio is 0.993 and after top-50 it is 1.000. The composition-implied tail rate moves from 2.066 % to 1.991 % (≈ 105 rows y ≥ 2 h among 5,290). |

## 5. Not determinable locally

- Ranking labels are null on all 344,841 rows, so the actual unmatched RMSE on the ranking set, the true tail count,
  and whether per-cell error equals the OOF per-cell error (the reweighting assumption) cannot be checked.
- Whether the Jul 2026 flight-number turnover (matched overlap 63.6 % vs 98.1 % in Jan 2026) is a schedule change,
  a callsign-format change or a data-extraction difference is not visible in these columns.
- Whether the EHAM Jan 2026 KL-heavy unmatched excess shares the mechanism of item 15 (`eham-jan2026-disruption`) is not
  tested here beyond the coincidence of rows.
- The OOF month-7 cells are produced by the leaky `jul` fold (sibling #26); a forward-fold OOF for July does not exist
  for the unmatched rows in `oof_main-48c19bc830`, so the reweighting cannot be repeated on a production-shaped fold.

## 6. Facts vs inference

**Facts (measured in this run):** every number in §3; reproduction of D's shares (EHAM 22.8/18.2, LSZH 12.5/10.2,
LFPG 13.9/16.8, LTFM 13.6/16.2, KL 7.4/3.0, TH 10.8/14.5, HB 2.7/4.5, V 7.9/13.1, seen 69.7/81.9, label-by-seen
962/2.08 vs 846/2.27) and P 8's counts (22,470 / 5,290; Jan 2,453 / Jul 2,837); the 46 / 7 row difference between
definitions changes no share by more than 0.05 pp and no RMSE by more than 3 s; the ranking set's Jan/Jul split of the
KL excess (331 of 389 in EHAM Jan 2026) and of the seen-rate drop (Jul 2026 only); the matched-row overlap figures
98.05 / 63.61 %; the reweighting table and its sensitivity to the top-10/20/50 rows.

**Inference (not measured):** that the aggregate 1.53-vs-1.08 gap and the V-share gap are "seasonal" (they equal the
Jan+Jul-vs-all-year difference within the training year, but the 2025 Jan/Jul rates are themselves single observations);
that the seen-rate drop is a flight-number turnover between the 2025 and 2026 summer schedules (the symmetric overlap
statistics are consistent with it; no schedule data was inspected); that the reweighted 2,624 has any bearing on the
ranking-set error (it assumes per-cell MSE transfers and is decided by fewer than 10 OOF rows); that EHAM Jan 2026
KL rows belong to the disruption of item 15.
