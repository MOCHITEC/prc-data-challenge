# Result — lens `arrival-rows`: census of the ARRIVAL rows and of what they join to

Script: `docs/data_audit_2026-09-18/scripts/arrival-rows.py` (no training, `POLARS_MAX_THREADS=1`,
runtime ≈ 8 min, reads only `data/raw/training_*.parquet` and `data/raw/ranking.parquet`, 21 columns,
lazy scan + `join_asof`). Every number below was produced by that script; nothing is copied from the
hypothesis reports. Context read first: `docs/investigation_2026-09-16/hypotheses/external.md` A3/H3,
`docs/investigation_2026-09-16/plan.md` item 32 and §0, `docs/investigation_2026-09-16/results/sched-filled-block.md`.

---

## 1. What I checked

1. **Census.** ARR vs DEP row counts, per airport and per month, in training and in the ranking set.
2. **Column population.** Null rate of all 21 columns on ARR vs DEP, separately for training and ranking.
3. **Taxi-in distribution.** Identity check, then per airport, per arrival runway, per stand, per hour of
   landing; plus 2025 vs 2026 Jan/Jul comparison at the same airports.
4. **The stand join.** For each departure, the most recent arrival at the same airport and the same
   `STAND_mvt` strictly before it, by backward `join_asof` keyed on `(airport, STAND_mvt)`.
   Departure anchor `SCHED_TIME_UTC_mvt`, arrival anchor `BLOCK_TIME_UTC_mvt` (the only clocks that are
   non-null on both sides in both sets). Coverage at seven cut-offs, gap quantiles, per airport, split by
   NM-match status, plus a miss autopsy and an anchor-sensitivity check against the (training-only)
   actual off-block.
5. **Inbound-leg identity.** Whether `FLIGHT_ID_mvt` ever links an arrival to a departure; agreement of
   `AIRCRAFT_TYPE_mvt`, `AIRCRAFT_OPERATOR_flt`, `CALLSIGN_flt`, `FLIGHT_mvt` across the stand link, by
   gap band; how often the link is *unique* (a second arrival on the same stand in the same window).
6. **Signal check, no training.** corr(y, ·) for (a) previous arrival's taxi-in, (b) turnaround gap
   `dep SCHED − arr BLOCK`, (c) arrivals at the airport in the preceding 30 / 60 min (counted on landing
   time and on in-block time), split matched/unmatched, per airport, raw and residualised on
   (airport, month, hour-of-day) medians; plus decile tables and the LIRF-unmatched tail.
7. **Pathologies on ARR rows.** Negative and zero durations, extreme taxi-in, the day-rollover pattern,
   the LIRF schedule-fill (`BLOCK == SCHED`) including a spike-vs-neighbouring-minute test, seconds-of-minute
   recording granularity, and the actype-null "second source".

Definitions: **unmatched** = routing definition, `AOBT_3_flt` null (22,470 training / 5,290 ranking —
reproduced exactly). `y` = `TAXITIME_SEC_mvt` on the departure. **taxi-in** = `TAXITIME_SEC_mvt` on the
arrival. **gap** = `dep SCHED_TIME − arr BLOCK_TIME` (seconds, always ≥ 0 by construction of the backward join).

---

## 2. Findings

### 2.0 Coverage first — this gates everything else

The asof join with **unbounded** lookback matches 99.91 % of training and 99.58 % of ranking departures,
but that number is meaningless: the gap p99 is 153,005 s (training) and the max is 26,705,653 s (309 days).
The usable figure is coverage **within a time band**.

| set | group | n dep | any | ≤1 h | ≤2 h | ≤4 h | ≤6 h | ≤12 h | ≤24 h | ≤72 h |
|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN | all | 2,085,047 | 0.9991 | 0.3158 | 0.6042 | 0.7599 | **0.8170** | **0.9398** | 0.9800 | 0.9942 |
| TRAIN | matched | 2,062,577 | 0.9991 | 0.3161 | 0.6050 | 0.7603 | 0.8175 | 0.9406 | 0.9803 | 0.9943 |
| TRAIN | unmatched | 22,470 | 0.9959 | 0.2924 | 0.5335 | 0.7188 | **0.7769** | **0.8629** | 0.9525 | 0.9838 |
| RANK | all | 344,841 | 0.9958 | 0.3021 | 0.5907 | 0.7492 | **0.8101** | **0.9311** | 0.9741 | 0.9886 |
| RANK | matched | 339,551 | 0.9959 | 0.3021 | 0.5916 | 0.7501 | 0.8110 | 0.9323 | 0.9745 | 0.9888 |
| RANK | unmatched | 5,290 | 0.9904 | 0.2998 | 0.5278 | 0.6938 | **0.7558** | **0.8514** | 0.9490 | 0.9773 |

**Training and ranking coverage agree to within 0.7 pp at every cut-off.** The unmatched rows — where our
error is concentrated — are covered 4–8 pp worse than matched rows at every band, and the shortfall is the
same in both sets (≤6 h: matched 0.8175/0.8110, unmatched 0.7769/0.7558).

Gap distribution (seconds), matched-to-a-stand departures only:

| set | group | n | min | p01 | p05 | p10 | p25 | p50 | p75 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN | all | 2,083,167 | 0 | 487 | 1,620 | 2,232 | 3,236 | 5,215 | 13,501 | 34,021 | 47,519 | 153,005 | 26,705,653 |
| TRAIN | matched | 2,060,789 | 0 | 521 | 1,623 | 2,251 | 3,237 | 5,189 | 13,454 | 33,904 | 47,096 | 151,019 | 26,705,653 |
| TRAIN | unmatched | 22,378 | 0 | 135 | 781 | 1,499 | 3,126 | 6,426 | 16,978 | 54,131 | 80,013 | 312,419 | 18,689,525 |
| RANK | all | 343,391 | 0 | 467 | 1,565 | 2,225 | 3,296 | 5,401 | 14,165 | 34,909 | 50,524 | 176,675 | 17,084,358 |
| RANK | matched | 338,152 | 0 | 477 | 1,599 | 2,242 | 3,297 | 5,398 | 14,102 | 34,735 | 49,925 | 175,081 | 17,084,358 |
| RANK | unmatched | 5,239 | 0 | 176 | 837 | 1,555 | 3,115 | 6,423 | 19,857 | 55,824 | 78,595 | 420,649 | 15,841,538 |

The distribution is heavy-tailed on both sides and *more* heavy-tailed for unmatched rows (p90 54,131 vs
33,904 s in training). Median gap ≈ 1 h 27 min (training) / 1 h 30 min (ranking).

Coverage per airport (≤6 h / ≤12 h), training vs ranking, is stable airport by airport:

| apt | TRAIN ≤6h | RANK ≤6h | TRAIN ≤12h | RANK ≤12h | TRAIN gap p50 | RANK gap p50 |
|---|---|---|---|---|---|---|
| EDDF | 0.8385 | 0.8273 | 0.9416 | 0.9228 | 5,618 | 5,932 |
| EDDM | 0.8324 | 0.8138 | 0.9240 | 0.9123 | 4,201 | 4,139 |
| EGLL | 0.8047 | 0.8062 | 0.9597 | 0.9551 | 6,600 | 6,600 |
| EHAM | 0.8422 | 0.8313 | 0.9432 | 0.9265 | 4,658 | 4,976 |
| LEBL | 0.8109 | 0.8113 | 0.9506 | 0.9363 | 3,778 | 3,957 |
| LEMD | 0.7471 | 0.7477 | 0.9140 | 0.9122 | 5,640 | 5,517 |
| LFPG | 0.8360 | 0.8314 | 0.9384 | 0.9341 | 5,342 | 5,635 |
| LIRF | 0.8819 | 0.8794 | 0.9637 | 0.9568 | 4,017 | 4,144 |
| LSZH | 0.7594 | 0.7523 | 0.9046 | 0.8970 | 4,385 | 4,533 |
| LTFM | 0.8095 | 0.8003 | 0.9451 | 0.9418 | 6,718 | 6,905 |

Unmatched rows per airport, coverage ≤6 h: worst is **EHAM 0.6533 training / 0.5945 ranking** on 4,084 /
1,206 rows (this is also the largest unmatched population). Others run 0.67–0.86.

Miss autopsy (departures with no earlier same-stand arrival at all): training 1,880 misses = 28 with a null
`STAND_mvt`, 141 on a stand no arrival ever used, 1,711 on a stand that arrivals do use but with no earlier
arrival in the window of the file. Ranking 1,450 misses = 4 / 133 / 1,313.

Anchor sensitivity: using the departure's **actual** off-block instead of its scheduled off-block (possible
in training only) changes coverage by < 0.1 pp (≤6 h 0.8179 vs 0.8170; ≤12 h 0.9408 vs 0.9398), so the
schedule anchor — the only one available in the ranking set — is not a material distortion of coverage.

Degenerate gaps: **no negative gaps** (0 by construction); 213 training / 24 ranking gaps are exactly 0;
12,220 training (0.59 %) and 2,203 ranking (0.64 %) gaps are under 300 s, spread evenly over all ten airports
(869–1,639 per airport in training).

### 2.1 Census

| set | DEP | ARR | ARR/DEP |
|---|---|---|---|
| TRAIN (2025-01 … 2025-12) | 2,085,047 | 2,082,750 | 0.9989 |
| RANK (2026-01 and 2026-07) | 344,841 | 344,693 | 0.9996 |

Per airport the ratio is 0.9946–1.0034 (training) and 0.9964–1.0038 (ranking) — arrivals and departures are
one-to-one at the airport level. Arrivals per airport, training / ranking: LTFM 273,886 / 47,764;
EHAM 247,706 / 38,143; EGLL 239,511 / 39,837; LFPG 238,835 / 39,733; EDDF 230,122 / 36,253;
LEMD 211,091 / 37,093; LEBL 179,081 / 29,973; EDDM 167,323 / 25,935; LIRF 160,504 / 26,839; LSZH 134,691 / 23,123.

Arrivals per month, training: 143,680 (Feb) to 190,865 (Aug). Ranking: **2026-01 152,871** and
**2026-07 191,822**. The full airport × month tables are in the script output (A3 / A3b). All arrival rows,
in both sets, land at one of the same ten airports; `ADES_mvt == apt` on 2,082,750 / 2,082,750 and
344,693 / 344,693 rows.

### 2.2 What is populated on arrivals versus departures

Null rate by column:

| column | TRAIN DEP | TRAIN ARR | RANK DEP | RANK ARR |
|---|---|---|---|---|
| `BLOCK_TIME_UTC_mvt` | 0.0 | 0.0 | **1.0** | **0.0** |
| `TAXITIME_SEC_mvt` | 0.0 | 0.0 | **1.0** | **0.0** |
| `MVT_TIME_UTC_mvt` / `SCHED_TIME_UTC_mvt` / `RUNWAY_mvt` / `MVT_ID_mvt` | 0.0 | 0.0 | 0.0 | 0.0 |
| `STAND_mvt` | 0.00001 | 0.00002 | 0.00001 | 0.00001 |
| `FLIGHT_ID_mvt` | 0.01075 | 0.00408 | 0.01532 | 0.00409 |
| `AIRCRAFT_TYPE_mvt` | 0.00072 | 0.00007 | 0.00113 | 0.00006 |
| `FLIGHT_mvt` | 0.00009 | 0.00011 | 0.00017 | 0.00011 |
| all eight `*_flt` columns | 0.01078 | 0.00742 | 0.01534 | 0.00740 |

The `*_flt` block is strictly all-or-nothing: `AOBT_3_flt` null agrees with `CALLSIGN_flt` null,
`AIRCRAFT_OPERATOR_flt` null and `MARKET_SEGMENT_flt` null with **zero disagreements** in all four
set × phase combinations. So "NM-unmatched" (the model-B routing rule) means "no flight record at all",
and operator / callsign / market segment are unavailable on *every* unmatched departure.

Arrival rows are better populated than departure rows on every key: 2.6× fewer null `FLIGHT_ID`,
10× fewer null `AIRCRAFT_TYPE`, and — decisively — `BLOCK_TIME` and `TAXITIME` are present on all 344,693
ranking arrivals while both are null on all 344,841 ranking departures.

Stand vocabularies agree: per airport the arrivals use 134–411 distinct stands and cover all but 0–32 of the
stands the departures use (training) and all but 0–10 (ranking).

### 2.3 Taxi-in distribution

Identity: `TAXITIME_SEC_mvt == BLOCK_TIME − MVT_TIME` exactly on 2,082,749 / 2,082,750 training arrivals
(one row has a null taxi time) and 344,693 / 344,693 ranking arrivals. On arrivals `MVT_TIME` is the
landing and `BLOCK_TIME` the in-block, i.e. the sign convention is the mirror of departures.

Per airport (training, seconds):

| apt | n | mean | p01 | p25 | p50 | p75 | p95 | p99 | max | <0 | >1 h |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EDDF | 230,122 | 582.6 | 249 | 441 | 541 | 676 | 980 | 1,330 | 4,693 | 0 | 0.00002 |
| EDDM | 167,323 | 367.5 | 171 | 291 | 355 | 424 | 597 | 837 | 12,245 | 0 | 0.00001 |
| EGLL | 239,511 | 525.7 | 178 | 353 | 428 | 601 | 1,088 | 1,796 | 9,191 | 0.00005 | 0.00023 |
| EHAM | 247,706 | 500.9 | 114 | 265 | 448 | 698 | 969 | 1,244 | 18,927 | 0.00001 | 0.00007 |
| LEBL | 179,081 | 235.6 | 76 | 152 | 207 | 287 | 451 | 624 | 3,573 | 0.00001 | 0 |
| LEMD | 211,091 | 525.8 | 127 | 316 | 495 | 675 | 1,014 | 1,308 | 3,458 | 0 | 0 |
| LFPG | 238,835 | 588.0 | 240 | 429 | 544 | 668 | 964 | 1,433 | 77,756 | 0.00011 | 0.00017 |
| LIRF | 160,504 | 648.6 | 233 | 478 | 592 | 715 | 1,144 | 1,980 | 87,244 | 0 | 0.00182 |
| LSZH | 134,691 | 356.7 | 156 | 260 | 329 | 413 | 580 | 793 | 604,797 | 0.00131 | 0.00010 |
| LTFM | 273,886 | 827.0 | 241 | 593 | 773 | 1,028 | 1,441 | 1,729 | 8,647 | 0 | 0.00005 |

Ranking medians are 336–769 and track training closely (EDDM 357 vs 355, LEBL 211 vs 207, LTFM 769 vs 773).

By hour of landing (pooled, training): median 432 s (17 h) to 611 s (02 h); the night hours 00–03 are the
slowest, 17–18 h the fastest. Spread across the day is ≈ 180 s on the median.

By arrival runway the within-airport spread is large: LTFM 35R p50 667 vs 34L 1,137; EDDF 07L 488 vs 25R 672;
LEMD 32L 321 vs 32R 611; EHAM 06 244 vs 18R 745.

By stand (stands with n ≥ 30): the standard deviation of per-stand median taxi-in is 56 s (EDDM) to 206 s
(LEMD); the min-to-max range of per-stand medians runs e.g. LEMD 216 → 1,143 s, LTFM 208 → 1,141 s,
EHAM 170 → 819 s. Stand is therefore a much stronger discriminator of taxi-in than hour of day.

2025 → 2026 comparison at Jan and Jul (mean taxi-in, seconds): the largest moves are EHAM Jul −69.7,
EDDF Jul −66.5, EHAM Jan +67.0, LIRF Jan −43.2, LSZH Jan +35.2; the other 15 airport × month cells move by
less than 31 s. Median moves are within ±77 s everywhere.

### 2.4 Does the inbound leg identify the outbound one?

**`FLIGHT_ID_mvt` is a leg id, not a turnaround id.** 315,269 training (50,923 ranking) ARR×DEP pairs share a
`FLIGHT_ID`; of these **313,639 (99.48 %) are the *same leg*** — the arrival is that departure's own landing
at its `ADES` — and 315,262 of them land *after* the departure's off-block. Only 591 training
(117 ranking) pairs are same-airport, i.e. could be a turnaround. There is no aircraft registration column.
So the only available link from an outbound to its inbound leg is **stand + time**.

Agreement across the stand link, by gap band:

| set | gap | n | `AIRCRAFT_TYPE_mvt` equal | `AIRCRAFT_OPERATOR_flt` equal | `CALLSIGN_flt` equal | `FLIGHT_mvt` equal |
|---|---|---|---|---|---|---|
| TRAIN | 0–1 h | 657,004 | 0.9813 | 0.9725 | 0.0189 | 0.0190 |
| TRAIN | 1–4 h | 927,275 | 0.9548 | 0.9460 | 0.0156 | 0.0162 |
| TRAIN | 4–12 h | 375,252 | 0.8209 | 0.8681 | 0.0064 | 0.0067 |
| TRAIN | > 12 h | 123,636 | 0.6504 | 0.7622 | 0.1000 | 0.1052 |
| RANK | 0–1 h | 103,935 | 0.9779 | 0.9651 | 0.0269 | 0.0248 |
| RANK | 1–4 h | 154,420 | 0.9513 | 0.9406 | 0.0175 | 0.0180 |
| RANK | 4–12 h | 62,707 | 0.8231 | 0.8644 | 0.0065 | 0.0067 |
| RANK | > 12 h | 22,329 | 0.6296 | 0.7476 | 0.0844 | 0.0880 |

Aircraft-type agreement is 98.1 % at gaps under an hour and decays to 65.0 % above 12 h; operator behaves the
same way. Callsign and flight number essentially never agree (1.6–2.7 %) — the outbound leg carries a
different flight number — except in the > 12 h band (10 %), which is the same flight number the next day.

"Tied to a specific inbound flight" — stand match **and** same aircraft type **and** same operator **and**
gap in (0, 12 h]:

| set | group | n | stand+type+operator ≤12 h | stand+type ≤12 h |
|---|---|---|---|---|
| TRAIN | all | 2,085,047 | 0.8479 | 0.8808 |
| TRAIN | matched | 2,062,577 | 0.8571 | 0.8832 |
| TRAIN | unmatched | 22,470 | **0.0000** | 0.6661 |
| RANK | all | 344,841 | 0.8313 | 0.8694 |
| RANK | matched | 339,551 | 0.8443 | 0.8727 |
| RANK | unmatched | 5,290 | **0.0000** | 0.6535 |

The 0.0000 is not an error: operator comes from the `*_flt` block, which is entirely null on every
unmatched departure (§2.2). On unmatched rows the strongest available tie is stand + aircraft type, at 66 %.

**The tie is not unique.** Of the 1,959,551 training departures with a same-stand arrival within 12 h,
1,457,582 (**74.5 %**) have a *second* same-stand arrival also within 12 h, and 662,390 (33.8 %) have one
within 4 h. So "most recent arrival on this stand" identifies a candidate, not the aircraft.

### 2.5 Signal check (correlations, no training)

Pearson corr(y, feature), training. "stand-matched" = a previous same-stand arrival within 12 h.

| group | scope | n | y p50 | y sd | taxi_in | turn_gap | arr_land_30 | arr_land_60 | arr_blk_30 | arr_blk_60 |
|---|---|---|---|---|---|---|---|---|---|---|
| all | stand-matched ≤12 h | 1,959,551 | 912 | 529.4 | 0.0917 | 0.0289 | 0.0210 | 0.0061 | 0.0094 | 0.0001 |
| all | no stand match | 125,496 | 903 | 763.9 | 0.1169 | 0.0259 | −0.0197 | −0.0213 | −0.0222 | −0.0203 |
| matched | stand-matched ≤12 h | 1,940,162 | 912 | 413.5 | 0.1115 | 0.0360 | 0.0336 | 0.0151 | 0.0189 | 0.0076 |
| matched | no stand match | 122,415 | 904 | 475.7 | 0.1595 | 0.0198 | −0.0130 | −0.0134 | −0.0170 | −0.0123 |
| unmatched | stand-matched ≤12 h | 19,389 | 959 | 3,324.1 | 0.0765 | 0.0261 | −0.0907 | −0.0997 | −0.0934 | −0.1024 |
| unmatched | no stand match | 3,081 | 783 | 3,828.1 | 0.1295 | 0.0583 | −0.1202 | −0.1294 | −0.1187 | −0.1251 |

Per airport, NM-matched rows with a stand match ≤12 h — **the sign of the taxi-in correlation flips by
airport**:

| apt | n | taxi_in | turn_gap | arr_land_30 | arr_land_60 |
|---|---|---|---|---|---|
| LEMD | 193,119 | **0.3528** | −0.0371 | 0.0723 | 0.0637 |
| EDDF | 214,945 | 0.1824 | 0.0234 | −0.0506 | −0.0690 |
| EDDM | 153,709 | 0.1219 | 0.1274 | −0.0723 | −0.1105 |
| LFPG | 221,433 | 0.0671 | 0.0842 | −0.0053 | −0.0038 |
| LSZH | 120,339 | 0.0480 | 0.0681 | −0.1048 | −0.0868 |
| EGLL | 228,591 | 0.0461 | −0.0616 | 0.0413 | 0.0824 |
| LIRF | 153,535 | 0.0253 | −0.0179 | 0.0374 | 0.0383 |
| LTFM | 254,562 | −0.0175 | 0.0474 | −0.0617 | −0.0920 |
| EHAM | 230,782 | −0.0705 | 0.0495 | −0.0295 | −0.0555 |
| LEBL | 169,147 | **−0.0943** | −0.1174 | **0.1590** | **0.1711** |

After residualising both y and the feature on their (airport, month, hour-of-day) medians, the pooled
correlations drop to taxi_in **0.0468**, arr_land_60 **−0.0245**, turn_gap **0.0567**; per airport the
residualised taxi-in correlation is LEMD 0.345, EDDF 0.159, EDDM 0.112, LFPG 0.054, LSZH 0.051, LIRF 0.023,
EGLL 0.018, LTFM −0.014, LEBL −0.070, EHAM −0.076. Much of the pooled raw correlation is therefore
airport / time-of-day structure the model already has, not turnaround information.

By gap band (NM-matched), corr(y, taxi_in) is 0.0988 (0–30 m), 0.1173 (30–60 m), 0.1261 (1–2 h),
0.0791 (2–4 h), 0.0833 (4–12 h), 0.1595 (> 12 h) — it does **not** strengthen as the turnaround tightens.

Decile tables (NM-matched, stand-matched ≤12 h) show the effect sizes directly: across taxi-in deciles
(p50 174 s → 1,114 s) mean y rises monotonically 915.9 → 1,094.7 s, i.e. **179 s of y for a 940 s swing in
taxi-in**. Across arr_land_60 deciles (9 → 53 arrivals/hour) mean y is *non-monotone*: 965.6, 931.0, 950.3,
956.9, 976.0, 1,000.7, 1,093.4, 1,148.3, 997.2, **858.3** — it peaks at the 8th decile and falls to its
minimum at the busiest decile. Across turn_gap deciles the same inverted-U appears (952 → 925 → 1,089 → 969).

**Unmatched rows.** The arrival-count features correlate *negatively* with y on unmatched rows in both
scopes (−0.09 to −0.13) and per airport: EDDM −0.175, EHAM −0.172, LTFM −0.157, LIRF −0.090, versus
positive at LEBL +0.086 and LEMD +0.082. Per-airport unmatched taxi-in correlations: EHAM +0.325,
LEMD +0.325, LSZH +0.146, EDDF +0.125, LFPG +0.033, EDDM +0.021, LIRF +0.016, LTFM −0.000, EGLL −0.020,
LEBL −0.224.

**The LIRF tail specifically.** LIRF unmatched rows split by schedule-fill: filled (n = 721, y p50 7,559)
corr(y, taxi_in) = 0.0081, corr(y, turn_gap) = 0.0799; non-filled (n = 767, y p50 1,070) corr(y, taxi_in) =
0.0026. Comparing LIRF unmatched rows with y ≥ 2 h (n = 439) against the rest (n = 1,049): previous-arrival
taxi-in medians are **identical** (602 vs 601 s) and arrivals-in-60-min medians are **identical** (25 vs 25);
only the turnaround gap differs (median 9,152 vs 4,319 s) and stand coverage is lower on the tail
(0.838 vs 0.924).

Arrival-count distributions are stable between the sets (p10/p50/p90 of arr_land_60: training 14/32/48,
ranking 13/32/46), so these features do not shift from 2025 to 2026.

### 2.6 Pathologies on arrival rows

**Schedule-fill at LIRF — present on arrivals, at about one fifth of the departure rate.**
`BLOCK_TIME == SCHED_TIME` (±10 s) on **10.08 %** of LIRF training arrivals and **9.74 %** of LIRF ranking
arrivals, against 0.94–3.08 % at the other nine airports. The spike test (mass at exactly 0 minutes divided
by the average mass in minutes −2, −1, +1, +2) is **5.27 at LIRF and 0.99–1.11 at every other airport**, so
the LIRF mass at exactly zero is ≈ 5× what the neighbouring minutes imply. Median `BLOCK − SCHED` at LIRF
is exactly 0 s (other airports −186 … +229 s).

The filled rows carry a visibly different taxi-in: LIRF training filled (n = 16,178) p05/p50/p95 =
233 / **830** / 2,221 s versus non-filled (n = 144,326) 349 / **589** / 963 s; ranking filled (n = 2,615)
189 / **843** / 2,213 versus 303 / **550** / 968. At EGLL the same flag selects rows whose taxi-in is
essentially unchanged (p50 471 vs 427, 0 rows > 2 h). So on LIRF arrivals, `BLOCK_TIME` behaves like a fill
and `TAXITIME_SEC_mvt` on those rows is `SCHED − landing`, not a taxi time — the same mechanism as
`results/sched-filled-block.md` describes for LIRF departures, at 10 % instead of 48 % (unmatched) / 18 %
(matched). LIRF arrivals by month: fill rate 8.7–12.8 % (training), 8.6 % Jan 2026 / 10.6 % Jul 2026.

**Negative durations.** 220 training arrivals (0.0106 %) and 39 ranking arrivals (0.0113 %) have taxi-in < 0;
min −2,652 s (training) and −12 s (ranking). Concentrated at LSZH: 177 / 220 training (rate 0.00131) and
35 / 39 ranking (rate 0.00151); the rest are LFPG 26/3, EGLL 12, EHAM 2, LTFM 1/1, EDDF 1, LEBL 1.
16 training and 5 ranking arrivals have taxi-in exactly 0.

**Extremes and day rollover.** Taxi-in > 2 h: 141 training rows (LIRF 119, LSZH 9, LFPG 6, EGLL 3, EHAM 2,
EDDM 1, LTFM 1) and 27 ranking rows (LIRF 17, EHAM 8, LFPG 2). Taxi-in > 12 h: 5 training, 1 ranking.
The departure day-rollover signature (value ≈ 86,400 + a plausible duration) appears on only **2** training
arrivals, both LIRF, residual ≈ 815 s, and on **0** ranking arrivals. The single largest training value is
LSZH 604,797 s (7 days), which matches no rollover pattern.

**Recording granularity.** Fraction of arrivals whose in-block second-of-minute is within 6 s of a boundary:
EDDM / EGLL / LFPG / LIRF / LTFM / LEBL / LEMD ≈ 1.00, EDDF 0.223, EHAM 0.239, LSZH 0.221. Landing time:
EDDM / EGLL / LIRF / LTFM / LFPG ≈ 1.00, LEBL 0.216, LEMD 0.216, EDDF 0.217, EHAM 0.226, LSZH 0.218.
LEBL and LEMD record in-block to the minute but landing to the second.

**The LIRF "second source" is a departure-side population.** Arrivals with a null `AIRCRAFT_TYPE_mvt`:
LIRF 125 training / 16 ranking, against 1,487 / 383 on LIRF departures. Arrivals with a null `FLIGHT_ID`:
8,506 training (EHAM 3,165, LTFM 3,037, LSZH 1,154, LFPG 308, LEMD 256, LEBL 238, LIRF 125, EDDF 114,
EDDM 80, EGLL 29) and 1,409 ranking; their taxi-in p50 is 472 s (training) / 494 s (ranking), only 6 of them
exceed 2 h, and their fill rate is 2.4 % — they look like ordinary arrivals with a missing flight record,
not like the LIRF unmatched departures.

---

## 3. What is CLEAN and what is unusable

**CLEAN — arrival rows are, as data, in better shape than departure rows.**
- The taxi-in identity holds exactly: 2,082,749 / 2,082,750 training and 344,693 / 344,693 ranking arrivals
  satisfy `TAXITIME_SEC_mvt == BLOCK_TIME − MVT_TIME`.
- `BLOCK_TIME` and `TAXITIME` are non-null on **all** 344,693 ranking arrivals; `STAND_mvt` is null on 5,
  `RUNWAY_mvt` on 0. This is a genuine 2026 observation window that departures do not provide.
- Null rates on arrivals are lower than on departures for `FLIGHT_ID` (0.0041 vs 0.0153 in ranking),
  `AIRCRAFT_TYPE_mvt` (0.00006 vs 0.00113) and the `*_flt` block (0.0074 vs 0.0153).
- Arrival and departure counts are one-to-one per airport (ratio 0.995–1.004), all arrivals land at the ten
  target airports, and the stand vocabularies overlap almost completely.
- Arrival-count and taxi-in distributions do not shift materially between 2025 and 2026 (means within ±70 s,
  arr_land_60 deciles within one arrival).
- **No day-rollover problem on arrivals**: 2 training rows, 0 ranking rows, versus 16 training departures.
- **Coverage of the stand join is stable between training and ranking** at every cut-off (within 0.7 pp),
  and the schedule anchor loses < 0.1 pp against the actual off-block anchor. If a stand-join feature is
  built on training it will be populated at the same rate on the ranking set.

**UNUSABLE or contaminated, with evidence.**
- **Unbounded-lookback coverage (99.9 %) is an artefact.** Gap p99 153,005 s, max 26.7 M s. Any statement of
  coverage must be banded; the honest numbers are ≤6 h 0.817 / 0.810 and ≤12 h 0.940 / 0.931.
- **`TAXITIME_SEC_mvt` on LIRF arrivals is contaminated on ~10 % of rows** (spike ratio 5.27 vs ≈ 1.0
  elsewhere; filled-row taxi-in p50 830 vs 589 and p95 2,221 vs 963). The contaminating flag
  `BLOCK == SCHED` *is* observable on arrivals in the ranking set, unlike on departures.
- **Operator / callsign linkage is unavailable exactly where it would matter.** `AOBT_3_flt` null ⇔ all
  `*_flt` null with zero exceptions, so all 22,470 training and 5,290 ranking unmatched departures have no
  operator and no callsign; "stand + type + operator" identification is 0.0000 on them by construction.
- **Callsign and flight number do not link the legs at all** (1.6–2.7 % agreement at short gaps).
  `FLIGHT_ID_mvt` links an arrival to a departure in 99.48 % of cases as the *same leg* at the destination,
  not as a turnaround.
- **The stand link is not an aircraft link.** 74.5 % of stand-matched departures have a second same-stand
  arrival inside the same 12 h window.
- **On the LIRF unmatched tail the arrival features are flat.** Previous-arrival taxi-in medians for
  y ≥ 2 h vs y < 2 h are 602 vs 601 s and arrivals-in-60-min are 25 vs 25; corr(y, taxi_in) is 0.008 on
  filled rows and 0.003 on non-filled ones.
- **Arrival-count features have the wrong sign on unmatched rows** (−0.09 to −0.13 pooled, −0.175 at EDDM,
  −0.172 at EHAM) while being weakly positive on matched rows, and the decile table for matched rows is
  non-monotone with its *lowest* mean y in the busiest decile (858 s at 53 arrivals/hour vs 1,148 s at 40).
- **Negative taxi-in at LSZH** (0.131 % training, 0.151 % ranking, min −2,652 s) and the LSZH maximum of
  604,797 s: at least those rows are not taxi times.

---

## 4. Facts and inference

**Facts (measured by the script, in §2).** All counts, null rates, quantiles, coverage fractions,
correlations, decile tables and spike ratios above. In particular: coverage ≤6 h of 0.8170 (training) and
0.8101 (ranking), ≤12 h of 0.9398 and 0.9311, with unmatched rows at 0.7769 / 0.7558 and 0.8629 / 0.8514;
the gap quantiles; 313,639 of 315,269 shared-`FLIGHT_ID` ARR×DEP pairs being the same leg; aircraft-type
agreement 0.9813 at 0–1 h decaying to 0.6504 above 12 h; 74.5 % of stand links being non-unique within 12 h;
the `*_flt` all-or-nothing property with zero disagreements; the LIRF arrival `BLOCK == SCHED` rate of
0.10079 (training) / 0.09743 (ranking) with spike ratio 5.27 against 0.99–1.11 elsewhere; 220 / 39 negative
taxi-in values with LSZH carrying 177 / 35 of them; 2 / 0 rollover-shaped arrival rows.

**Inference (not measured).**
- That the LIRF arrival rows with `BLOCK == SCHED` are a *fill* rather than 16,178 aircraft that happened to
  reach the gate at the scheduled minute. The evidence is the 5.27× excess over the neighbouring minutes and
  the shifted taxi-in on exactly those rows; the source system is not inspectable here, and LIRF records all
  its times on minute boundaries (100 % within 6 s), which is also consistent with rounding.
- That the 591 same-airport shared-`FLIGHT_ID` pairs are return-to-stand events rather than genuine
  turnarounds recorded under one id.
- That the non-monotone arrival-count deciles reflect a confound (busy hours are also hours with favourable
  runway configurations or shorter scheduled taxi routes) rather than congestion genuinely reducing taxi-out.
  Nothing here separates those.
- That the sign flip of corr(y, taxi_in) between LEMD (+0.35) and EHAM/LEBL (−0.07/−0.09) reflects different
  airport layouts (shared vs segregated taxiways) rather than a data defect. Not tested.
- That aircraft-type agreement of 0.65 in the > 12 h band approximates the agreement rate of two *unrelated*
  aircraft on the same stand, and hence that the excess to 0.98 at short gaps measures real inbound-leg
  identification. No independent random-pairing baseline was computed.
- Whether ranking LIRF arrivals are filled at the training rate is measurable here (0.0974 vs 0.1008) — that
  is a fact — but whether the corresponding ranking *departures* are filled at the training rate remains
  unobservable, as `results/sched-filled-block.md` §5 states.

**Definitional notes for cross-item comparison.** §2.6's "fill" uses `|BLOCK − SCHED| ≤ 10 s`, matching D's
`f10`. The spike table (I2) rounds `BLOCK − SCHED` to the nearest minute, so its "at 0" bucket is wider than
`f10` at the five airports that record seconds (EDDF 0.0278 vs 0.0097, EHAM 0.0258 vs 0.0094,
LSZH 0.0276 vs 0.0103) and identical at the five that record whole minutes, LIRF included (0.10079 both).
"Unmatched" is the routing definition (`AOBT_3_flt` null): 22,470 training / 5,290 ranking, reproduced.
"Coverage" always means the backward asof join keyed on `(airport, STAND_mvt)` with the departure anchored
at `SCHED_TIME_UTC_mvt` and the arrival at `BLOCK_TIME_UTC_mvt`.
