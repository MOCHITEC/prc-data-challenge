# Data audit 2026-09-18 — lens: timestamp integrity across the nine time columns

Script: `docs/data_audit_2026-09-18/scripts/timestamps.py` (no training, `POLARS_MAX_THREADS=1`,
lazy scans, streaming engine; stages `inv eq eqop ord prec span cross derived spike spike2 spike3
extreme arrfill origin distinct extras grid`; total runtime ≈ 35 min). Sources: the 13
`data/raw/training_*.parquet` files and `data/raw/ranking.parquet` for raw columns,
`data/work/feat_train.parquet` / `feat_rank.parquet` for derived deltas. Every number below was
produced by that script; none is copied from an earlier document except where explicitly marked
"(prior audit)".

## 0. Conventions

- Short names: `MVT` = `MVT_TIME_UTC_mvt`, `BLOCK` = `BLOCK_TIME_UTC_mvt`, `SCHED` =
  `SCHED_TIME_UTC_mvt`, `LOBT` = `LOBT_flt`, `IOBT` = `IOBT_flt`, `EOBT1` = `EOBT_1_flt`,
  `AOBT3` = `AOBT_3_flt`, `ARVT1` = `ARVT_1_flt`, `ARVT3` = `ARVT_3_flt`.
- `train` = the 2025 training files (4,167,797 rows: 2,085,047 `PHASE_mvt = DEP`, 2,082,750 `ARR`);
  `rank` = `ranking.parquet` (689,534 rows: 344,841 DEP, 344,693 ARR). Months are the calendar month
  of `MVT`; the ranking set holds 2026-01 and 2026-07 only.
- `apt` = the challenge airport of the row: `ADEP_mvt` on DEP rows, `ADES_mvt` on ARR rows.
- **Copy test.** Exact equality between two minute-rounded clocks happens by coincidence, so a raw
  equality rate proves nothing. Throughout, a copy is claimed only from a *spike against the local
  density*: bucket `k` holds the rows with `|A − B − 60k| ≤ 6 s`; `copy_ratio = n(k=0) / mean(n(k=±1))`
  and `copy_excess = n(k=0) − mean(n(k=±1))`. `copy_ratio ≈ 1` means "no copy, the equalities are the
  ordinary mass at zero difference". The ±6 s tolerance is needed because `BLOCK` and `MVT` at six
  airports live on a 13-value second grid (§3.5), so a copy shows up at ±6 s, not at exactly 0.
- Note on nulls: `BLOCK` and `TAXITIME_SEC_mvt` are null on all 344,841 **ranking DEP** rows, but are
  **present on all 344,693 ranking ARR rows**. Several checks below turn on that asymmetry.

## 1. What was checked

Coverage list, so that anything not on it was not looked at.

1. Row counts, null counts and min/max per column, per set × phase (`inv_nullrange`).
2. Sentinel and boundary values per column, per set × phase: rows before 2024 / after 2026, exact
   midnight `00:00:00`, sub-second components, second-of-minute == 0 rate (`inv_sentinel`).
3. **All 36 column pairs**, exact equality and near equality (≤ 6 s, ≤ 60 s), for every
   set × phase × airport × month cell (280 cells × 36 pairs; `eq_cell.parquet`, rolled up to
   `eq_by_set_phase`, `eq_by_apt`, `eq_by_month`).
4. Copy test (§0) on the 14 pairs with a visible equality cluster, three ways: minute-bin histogram
   (`spike_excess`), exact-second neighbours (`spike2_exact`), ±6 s tolerance buckets globally and
   per airport (`spike3_total`, `spike3.parquet`).
5. The same equality census keyed on the flight's **origin** `ADEP_mvt` rather than the row's airport,
   for both phases, for the five off-block pairs (`origin_eq.parquet`).
6. The same census per `AIRCRAFT_OPERATOR_flt` (95 training / 33 ranking operators with ≥ 2,000 DEP
   rows) for 8 pairs (`eqop.parquet`, `eqop_summary`).
7. How many *distinct* values the four off-block stamps `SCHED/LOBT/IOBT/EOBT1` carry per row, per
   airport per set (`distinct_offblock_family`).
8. **Ordering violations** for 15 rules whose sign is implied by physics or by the NM meaning of the
   field (10 on DEP, 5 on ARR), counted per set × phase × airport and per month, with the violation
   median and maximum and counts above 60 s and above 1 h (`ord_total`, `ord_by_apt`, `ord_by_month`).
   Rules: `BLOCK ≤ MVT`, `AOBT3 ≤ MVT`, `MVT ≤ ARVT3`, `MVT ≤ ARVT1`, `AOBT3 ≤ ARVT3`,
   `EOBT1 ≤ ARVT1`, `IOBT ≤ ARVT1`, `LOBT ≤ ARVT3`, `BLOCK ≤ ARVT3`, `SCHED ≤ ARVT1` (DEP);
   `MVT ≤ BLOCK`, `ARVT3 ≤ BLOCK`, `AOBT3 ≤ MVT`, `AOBT3 ≤ ARVT3`, `EOBT1 ≤ ARVT1` (ARR).
9. Quantiles (0, 0.001, 0.5, 0.999, 1) and the count beyond ±7 days for 17 pairwise differences, per
   set × phase (`span_quantiles`).
10. Day-scale anomaly counts for 11 explicit conditions, per set × airport × month, plus a row-level
    listing of the 15 worst arrival cases in each set (`extreme_total`, `extreme_by_apt`,
    `extreme_by_month`, `extreme_arr_taxiin_*`).
11. **Rounding and precision**: full second-of-minute histogram per column per airport per set per
    phase, with distinct-value count, `P(sec = 0)`, `P(within 6 s of a minute)` and entropy
    (`prec_sec_summary`, `prec_sec.parquet`); minute-of-hour multiples of 5/10/15
    (`prec_minute_grid`); the explicit 13-value grid shares for `MVT` and `BLOCK` (`grid_sec_share`);
    the `ARVT3` grid rate month by month (`extras_arvt3_grid_by_month`).
12. **Cross-row integrity**: for every `FLIGHT_ID_mvt` present as both a DEP and an ARR row in the
    same set (315,269 training / 50,923 ranking pairs), whether each of the nine stamps agrees
    between the two rows, and whether the DEP row's `ARVT3` matches the ARR row's `MVT` (landing) or
    its `BLOCK` (on-block) (`cross_train`, `cross_rank`).
13. **Derived-column consistency**: whether the eight `*_minus_*` features and `planned_flight_s` in
    `feat_train`/`feat_rank` equal the raw timestamp differences exactly, and their null alignment
    (`derived_train`, `derived_rank`).
14. Whether each monthly training file contains only its own month of `MVT` (`grid_file_month`).
15. **2025 vs 2026**: every table above is cut by `set`; the copy rates, ordering-violation rates,
    precision rates and extreme-value rates are all reported for both, and month by month where the
    rate moved.

Not checked (out of this lens): the label's own distribution, congestion/weather features, the
model, anything about `RUNWAY_mvt` / `STAND_mvt` / operator identity beyond using the operator as a
grouping key.

## 2. Findings

Order is arbitrary. "Actionable" here means only: the subset is identifiable from columns that are
non-null in the ranking set.

### F1. `LOBT_flt` is an exact copy of `EOBT_1_flt` on 100.000 % of LTFM departures

- **What the column contains instead:** at LTFM, `LOBT_flt` ("last off-block time", the field that is
  supposed to be the *updated* off-block reference) never differs from the filed estimate
  `EOBT_1_flt`. Elsewhere it differs on 26–50 % of rows.
- **Rows:** 269,320 / 269,320 training DEP rows at LTFM and 46,860 / 46,860 ranking DEP rows at LTFM.
  Exactly 1.000000 in **every one of the 14 months** (`extras_ltfm_lobt_eobt1_by_month`), 2025 and
  2026 alike. Per-airport equality rate for comparison (train / rank): LTFM 1.000/1.000,
  LSZH 0.696/0.735, EHAM 0.695/0.698, EDDF 0.685/0.696, LEMD 0.659/0.639, EDDM 0.656/0.668,
  LEBL 0.638/0.668, LFPG 0.744/0.647, EGLL 0.586/0.537, LIRF 0.503/0.502.
- **Operators:** it is an airport property, not an operator property — the largest operator at LTFM
  sits at 0.9664 (train) / 0.9647 (rank) overall, i.e. below 1 once its non-LTFM departures are
  included. Keyed on the flight's origin instead of the row's airport, LTFM-origin ARR rows give
  20,481 / 20,487 (train) and 3,518 / 3,520 (rank) — the handful of exceptions are ARR rows whose DEP
  counterpart is not in the set.
- **Identifiable in the ranking set:** yes — `LOBT_flt` and `EOBT_1_flt` are both non-null on
  334,261 of 344,841 ranking DEP rows, and `ADEP_mvt` is non-null on all of them.
- **Consequence for the derived columns (measured, §2 F11):** `lobt_minus_eobt1` is identically 0 and
  `mvt_minus_lobt` is identically equal to `mvt_minus_eobt1` on every LTFM departure in both sets.
- **Query:**
  ```python
  pl.scan_parquet("data/raw/training_*.parquet").filter(
      (pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LTFM")
  ).select(
      pl.len(),
      (pl.col("LOBT_flt") == pl.col("EOBT_1_flt")).sum(),
  ).collect()           # -> 269320, 269320   (same on ranking.parquet: 46860, 46860)
  ```

### F2. `SCHED`, `LOBT`, `IOBT`, `EOBT1` are four names for largely one value

- **What the columns contain instead:** the four off-block stamps are a nested copy structure, not
  four measurements. Copy ratios on DEP rows (train / rank), ±6 s buckets:

  | pair | n both | n equal (±6 s) | equality rate | copy_ratio | copy_excess |
  |---|---|---|---|---|---|
  | `LOBT == IOBT` train | 2,062,577 | 2,051,361 | 0.9945 | **4,620** | 2,050,917 |
  | `LOBT == IOBT` rank | 339,551 | 337,389 | 0.9936 | **2,835** | 337,270 |
  | `SCHED == IOBT` train | 2,062,577 | 1,534,587 | 0.7440 | **582** | 1,531,950 |
  | `SCHED == IOBT` rank | 339,551 | 246,913 | 0.7272 | **620** | 246,515 |
  | `SCHED == LOBT` train | 2,062,577 | 1,526,520 | 0.7401 | **530** | 1,523,638 |
  | `SCHED == EOBT1` train | 2,062,577 | 1,145,218 | 0.5552 | **108** | 1,134,576 |
  | `SCHED == EOBT1` rank | 339,551 | 175,930 | 0.5181 | **109** | 174,321 |
  | `LOBT == EOBT1` train | 2,062,577 | 1,444,439 | 0.7003 | **101** | 1,430,184 |
  | `IOBT == EOBT1` train | 2,062,577 | 1,433,439 | 0.6950 | **98** | 1,418,744 |

  On DEP rows the number of *distinct* values among `{SCHED, LOBT, IOBT, EOBT1}` is 1 on 37.1–66.4 %
  of rows (by airport), 2 on 33.0–45.8 %, 3 on 0.59–17.1 %, and 4 on at most 0.015 %
  (`distinct_offblock_family`). LTFM has the smallest 3-distinct share (0.0066 train / 0.0059 rank,
  because of F1); LIRF the largest (0.1528 / 0.1714).
- **Rows / airports / months:** universal — every airport and every month, both sets. Per-operator
  spread (ops with ≥ 2,000 DEP rows): `LOBT == IOBT` 0.940–1.000 with 92 of 95 training operators
  above 0.98; `SCHED == IOBT` 0.470–0.959; `SCHED == EOBT1` 0.248–0.848.
- **Identifiable in the ranking set:** yes, all four columns are present.
- **Query:** as F1 with the pair substituted; the copy ratio is
  `stage_spike3` in the script (`spike3_total.csv`).

### F3. `AOBT_3_flt` is after take-off on ~0.05 % of departures, by up to 3.2 h

- **What the column contains instead:** on these rows `AOBT_3` ("actual off-block") post-dates
  `MVT_TIME` (take-off), so it cannot be that flight's off-block time. It is the sole input to
  `mvt_minus_aobt3`, model A's proxy, which is negative on exactly these rows.
- **Rows:** 1,012 of 2,062,577 training DEP rows (0.000491) and 169 of 339,551 ranking DEP rows
  (0.000498). Of those, 720 / 112 exceed 60 s and **219 / 31 exceed one hour**; the maximum is
  11,521 s (3 h 12 m) in training and 11,519 s in ranking.
- **Airports (violation rate, train → rank):** LTFM 0.00129 → 0.00062, LFPG 0.00080 → 0.00084,
  EHAM 0.00041 → 0.00062, EDDF 0.00044 → 0.00047, LSZH 0.00042 → **0.00120**,
  LIRF 0.00017 → **0.00087**, EGLL 0.00033 → **0.000025**, EDDM 0.00013 → 0.000039,
  LEBL 0.00026 → 0.00024, LEMD 0.00023 → 0.00022. The pooled rate is stable but the per-airport
  rates move by factors of 3–13 between 2025 and 2026 in both directions.
- **Months:** training monthly rate 0.00021 (Feb) – 0.00095 (Jul); ranking 0.00041 (Jan 2026),
  0.00057 (Jul 2026).
- **Identifiable in the ranking set:** yes — `MVT` and `AOBT_3` are both non-null on 339,551 ranking
  DEP rows, and the sign of `mvt_minus_aobt3` shows it directly.
- **Query:**
  ```python
  pl.scan_parquet("data/raw/ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP").select(
      ((pl.col("AOBT_3_flt") - pl.col("MVT_TIME_UTC_mvt")).dt.total_seconds() > 0).sum(),
      ((pl.col("AOBT_3_flt") - pl.col("MVT_TIME_UTC_mvt")).dt.total_seconds() > 3600).sum(),
  ).collect()           # -> 169, 31
  ```

### F4. The LIRF `BLOCK := SCHED` fill also exists on **arrival** rows, where it is visible in the ranking set

- **What the column contains instead:** on LIRF ARR rows `BLOCK_TIME_UTC_mvt` (in-block) carries
  `SCHED_TIME_UTC_mvt` (scheduled in-block), so `TAXITIME_SEC_mvt` on those rows is
  `SCHED − MVT` = "schedule buffer + taxi-in", not taxi-in. This is the arrival twin of the known
  departure finding (`sched-filled-block.md`), which examined DEP rows only.
- **Rows:** ±6 s bucket at LIRF ARR: **16,178 of 160,504 training rows (10.08 %)** and **2,615 of
  26,839 ranking rows (9.74 %)**, against a local density of 3,023 / 485 — `copy_ratio` **5.35**
  (train) and **5.39** (rank), `copy_excess` 13,155 / 2,130.
- **Airports:** LIRF is the only airport with a copy spike on ARR rows. The other nine have
  `copy_ratio` 0.77–1.31 (EDDF 0.96/0.77, EDDM 1.02/1.01, EGLL 1.11/1.10, EHAM 1.07/1.05,
  LEBL 1.04/1.00, LEMD 0.98/1.06, LFPG 1.00/1.02, LSZH 1.05/1.31, LTFM 1.03/1.01).
- **Months:** present in all 14 months; LIRF ARR `copy_ratio` 4.28–6.31 across 2025 and
  4.22 (Jan 2026) / 6.39 (Jul 2026); ±6 s rate 0.0874–0.1276 (2025), 0.0857 / 0.1057 (2026). No
  drift.
- **What it produces:** 292 of 160,504 LIRF training arrivals have taxi-in > 1 h (95 of them inside
  the filled bucket), 119 > 2 h and 3 > 12 h, max **87,244 s**; ranking LIRF has 42 > 1 h (12 filled),
  17 > 2 h, max 41,885 s. Pooled over all arrivals the training maximum is **604,797 s** (LSZH, §F8)
  and LFPG has 77,756 s / 50,517 s.
- **Identifiable in the ranking set:** **yes, directly** — `BLOCK`, `SCHED` and
  `TAXITIME_SEC_mvt` are all non-null on ranking ARR rows. This is the one place where the fill state
  itself is observable at prediction time.
- **Query:**
  ```python
  d = (pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds()
  pl.scan_parquet("data/raw/ranking.parquet").filter(
      (pl.col("PHASE_mvt") == "ARR") & (pl.col("ADES_mvt") == "LIRF")
  ).select(
      pl.len(), (d.abs() <= 6).sum(), ((d - 60).abs() <= 6).sum(), ((d + 60).abs() <= 6).sum(),
  ).collect()           # -> 26839, 2615, 271, 251   (ratio 2615 / 261 = 5.39)
  ```

### F5. A second, smaller `BLOCK := SCHED` cluster at LEBL and LEMD on departures

- **What the column contains instead:** the same fill as at LIRF, at a much lower rate.
- **Rows (training DEP, ±6 s bucket, local density, `copy_ratio`, `copy_excess`):**
  LIRF 29,090 / 3,955 / **7.36** / 25,135; **LEBL 10,086 / 7,184 / 1.40 / 2,902**;
  **LEMD 9,470 / 8,060 / 1.18 / 1,410**; LSZH 1,011 / 833 / 1.21 / 178; LFPG 9,883 / 9,009 / 1.10 / 874;
  EGLL 11,668 / 10,729 / 1.09 / 940; EHAM 1,960 / 1,852 / 1.06 / 108; LTFM 14,046 / 13,623 / 1.03 / 424;
  EDDM 8,341 / 8,199 / 1.02 / 142; EDDF 1,725 / 1,683 / 1.03 / 42.
  Pooled: 97,280 in the bucket vs 65,123 local, excess **32,157**.
- **Identifiable in the ranking set:** **no** for departures — `BLOCK` is null on all 344,841 ranking
  DEP rows, so neither the rate nor the membership can be observed there.
- **Note on the prior audit:** its raw rate "LEBL matched f10 0.0565" is reproduced here
  (10,086 / 177,843 = 0.0567 over all LEBL DEP rows); against the local density only 29 % of those
  rows are an excess. The LIRF figure survives the same correction almost intact (86 % excess).
- **Query:** `stage_spike3` restricted to `A = BLOCK, B = SCHED` (`spike3.parquet`).

### F6. A `AOBT_3 := EOBT_1` cluster at EGLL only

- **What the column contains instead:** on these rows the "actual" off-block equals the filed
  estimate to the second.
- **Rows:** ±6 s bucket at EGLL DEP 18,190 of 238,132 training rows against a local density of
  15,391 — `copy_ratio` **1.18**, excess **2,799**; ranking 2,874 of 39,592 against 2,434 —
  `copy_ratio` **1.18**, excess **440**. Identical ratio in both sets.
- **Every other airport** is at `copy_ratio` 0.99–1.05 (train) and 0.93–1.07 (rank), i.e. no cluster:
  the pooled `EOBT1 == AOBT3` rate of 5.02 % (train) / 5.13 % (rank) is almost entirely the ordinary
  mass at zero difference, **not** a copy (see also §3.1).
- **Months:** the EGLL rate runs 0.058–0.096 through 2025 with a seasonal shape (highest Jan–Apr,
  lowest Jul–Sep) and 0.0815 (Jan 2026) / 0.0642 (Jul 2026) — consistent with 2025's same months.
- **Identifiable in the ranking set:** yes — both columns present; the flag is
  `(AOBT_3_flt - EOBT_1_flt).abs() <= 6` at `ADEP_mvt == "EGLL"`. The excess is 15 % of the flagged
  rows, so the flag selects a mixture, not a pure subset.
- **Query:** `spike3.parquet` filtered to `A = EOBT1, B = AOBT3`.

### F7. `ARVT_3_flt`'s recording precision switches airport by airport, mid-series, including inside the ranking months

- **What the column contains instead:** at some airport-months `ARVT_3` is a second-resolution time;
  at others it is a minute-rounded time reported on the 13-value grid of §3.5. The regime flips
  without warning. Share of ARR rows within 6 s of a minute boundary (0.217 = uniform seconds,
  ≈ 1.0 = fully gridded):

  | month | EGLL | EHAM | LEBL | LEMD | LFPG | LIRF | LTFM | EDDF | EDDM | LSZH |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 2025-01 | 0.220 | 0.215 | 0.218 | 0.885 | 0.787 | 0.830 | 0.996 | 0.218 | 0.222 | 0.221 |
  | 2025-03 | 0.244 | 0.214 | 0.224 | 0.768 | 0.798 | 0.841 | 0.984 | 0.213 | 0.222 | 0.221 |
  | 2025-04 | **0.834** | 0.213 | 0.223 | 0.801 | 0.775 | 0.839 | 0.991 | 0.218 | 0.218 | 0.218 |
  | 2025-07 | 0.824 | 0.220 | **0.388** | 0.806 | 0.887 | 0.838 | 0.996 | 0.218 | 0.223 | 0.221 |
  | 2025-08 | 0.853 | 0.218 | **0.839** | 0.792 | 0.910 | 0.841 | 0.996 | 0.212 | 0.222 | 0.223 |
  | 2025-11 | 0.623 | 0.217 | 0.786 | 0.803 | 0.808 | 0.828 | 0.998 | 0.217 | 0.216 | 0.223 |
  | **2026-01** | **0.589** | 0.217 | 0.802 | 0.812 | **0.597** | 0.808 | 0.997 | 0.215 | 0.222 | 0.214 |
  | **2026-07** | **0.925** | **0.552** | 0.717 | 0.881 | 0.877 | 0.828 | 0.998 | 0.217 | 0.219 | 0.217 |

  EGLL flips from ungridded to gridded between 2025-03 and 2025-04; LEBL between 2025-07 and 2025-08;
  **EHAM flips only in 2026-07, a month with no training analogue** (0.215–0.222 in all 12 training
  months and in 2026-01, then 0.552).
- **Identifiable in the ranking set:** yes — `ARVT_3_flt` is non-null on 342,141 ranking ARR and
  339,551 ranking DEP rows, and the second-of-minute is readable per row.
- **Query:**
  ```python
  s = pl.col("ARVT_3_flt").dt.second()
  pl.scan_parquet("data/raw/ranking.parquet").filter(
      (pl.col("PHASE_mvt") == "ARR") & (pl.col("ADES_mvt") == "EHAM")
  ).group_by(pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m")).agg(
      pl.len(), ((s <= 6) | (s >= 54)).mean()
  ).collect()           # -> 2026-01 0.2168 ; 2026-07 0.5523
  ```

### F8. Day- and week-scale date errors in `MVT`/`BLOCK`/`SCHED`/`ARVT` — present in both sets

- **What the columns contain instead:** on a small set of rows one stamp carries the wrong date, so
  the implied duration is a multiple of a day (or, once, a week) plus a plausible remainder.
- **Counts (train / rank):**

  | condition | phase | train | rate | rank | rate |
  |---|---|---|---|---|---|
  | `|MVT − SCHED| > 1 day` | DEP | 153 | 7.3e-5 | 28 | 8.1e-5 |
  | `|SCHED − EOBT1| > 1 day` | DEP | 103 | 4.9e-5 | 15 | 4.3e-5 |
  | `MVT − BLOCK > 1 day` | DEP | 15 | 7.2e-6 | — | BLOCK null |
  | `ARVT3 < MVT` | DEP | 27 | 1.3e-5 | 8 | 2.4e-5 |
  | `|MVT − SCHED| > 1 day` | ARR | 312 | 1.5e-4 | 73 | 2.1e-4 |
  | `BLOCK − MVT > 1 h` (taxi-in) | ARR | 440 | 2.1e-4 | **145** | **4.2e-4** |

- **Extreme values:** DEP `MVT − SCHED` ranges −113,526 … +270,508 s (train) and −85,810 … +174,412 s
  (rank); DEP `AOBT3 − SCHED` −115,020 … +269,280 s; DEP `SCHED − EOBT1` −268,800 … +114,300 s;
  DEP `ARVT3 − MVT` min −84,061 s (train) / −81,600 s (rank); ARR `MVT − BLOCK` min **−604,797 s**
  (exactly 7 days 0 h 0 m 3 s, LSZH 2025-08: `MVT` 2025-08-12 10:29:01 against `BLOCK` 2025-08-19
  10:28:58 and `SCHED` 2025-08-19 10:30:00). No difference in any pair exceeds ±7 days except that
  one row.
- **Airports:** DEP `|MVT − SCHED| > 1 day` — EDDF 99/10, EHAM 34/14, LFPG 9/0, LTFM 7/0, LIRF 3/3,
  LEBL 1/0, EDDM 0/1. ARR `BLOCK − MVT > 1 h` — train LIRF 292, EGLL 55, LFPG 41, EHAM 18, LSZH 14,
  LTFM 14, EDDF 5, EDDM 1; rank EHAM 72, LIRF 42, LFPG 22, EGLL 8, EDDF 1. The composition of the
  arrival outliers moves from LIRF-dominated in 2025 to EHAM-dominated in 2026-01.
- **Identifiable in the ranking set:** yes for every condition except those involving DEP `BLOCK`.
- **Query:**
  ```python
  pl.scan_parquet("data/raw/ranking.parquet").filter(pl.col("PHASE_mvt") == "ARR").select(
      ((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("MVT_TIME_UTC_mvt")).dt.total_seconds() > 3600).sum()
  ).collect()           # -> 145
  ```

### F9. `ARVT_3` on arrival rows is after the on-block time on 0.06–0.10 % of rows, and the rate rose 55 % in 2026

- **What the column contains instead:** `ARVT_3` is an actual **landing** time (established in F12,
  not an in-block time), so `ARVT3 ≤ BLOCK` must hold on ARR rows. It does not on
  **1,311 of 2,067,300 training ARR rows (0.000634)** and **336 of 342,141 ranking ARR rows
  (0.000982)**; 854 / 268 of those exceed 60 s; maximum 3,377 s (train) / 3,625 s (rank).
- **Months:** the training rate rises monotonically through the year — 0.000584 (Jan), 0.000368
  (Apr), 0.000651 (Jul), 0.001168 (Oct), 0.000698 (Dec) — and the ranking months are 0.000811
  (Jan 2026) and 0.001119 (Jul 2026).
- **Airports (train → rank):** LIRF 0.00230 → **0.00748**, EDDF 0.00064 → 0.00103,
  EHAM 0.00088 → 0.00045, LEBL 0.00066 → 0.00070, EDDM 0.00061 → 0.00035, LFPG 0.00050 → 0.00023,
  LEMD 0.00040 → 0.00035, LSZH 0.00029 → 0.00052, LTFM 0.00029 → 0.00032, EGLL 0.00019 → 0.00008.
- **Identifiable in the ranking set:** yes — both columns present on ARR rows.
- **Query:**
  ```python
  pl.scan_parquet("data/raw/ranking.parquet").filter(pl.col("PHASE_mvt") == "ARR").select(
      pl.len(), ((pl.col("ARVT_3_flt") - pl.col("BLOCK_TIME_UTC_mvt")).dt.total_seconds() > 0).sum()
  ).collect()           # -> 344693 (342141 with both non-null), 336
  ```

### F10. `MVT` is after the *planned* arrival `ARVT_1` on 0.18–0.21 % of departures

- **What the column contains instead:** `ARVT_1` (filed arrival) pre-dates the flight's own take-off,
  so it is a stale plan, not that flight's planned arrival. `ARVT_1` is one of the two inputs to
  `planned_flight_s`.
- **Rows:** 3,786 of 2,062,577 training DEP rows (0.00184) and **728 of 339,551 ranking DEP rows
  (0.00214)**; 3,427 / 673 exceed 60 s; 33 / 9 exceed one hour; maximum 83,542 s (train) /
  82,162 s (rank), both at LIRF, i.e. a day error.
- **Airports (train → rank):** EHAM 0.00326 → **0.00535**, LSZH 0.00335 → **0.00511**,
  LFPG 0.00354 → 0.00194, EGLL 0.00188 → 0.00141, EDDF 0.00135 → **0.00220**, EDDM 0.00144 → 0.00163,
  LEBL 0.00166 → 0.00199, LEMD 0.00101 → 0.00158, LIRF 0.00066 → 0.00091, LTFM 0.00039 → 0.00045.
  Eight of the ten airports have a higher rate in 2026 than in 2025.
- **Months:** training 0.00084 (Apr) – 0.00288 (Jul); ranking **0.00313 (Jan 2026)**, 0.00136
  (Jul 2026). Jan 2026 exceeds every training month.
- The companion rule `SCHED ≤ ARVT1` is violated on 624 / 82 rows (209 / 28 of them by more than an
  hour, max 103,572 s), concentrated at EHAM (150 of the 209 training > 1 h cases).
- **Identifiable in the ranking set:** yes.
- **Query:**
  ```python
  pl.scan_parquet("data/raw/ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP").select(
      ((pl.col("MVT_TIME_UTC_mvt") - pl.col("ARVT_1_flt")).dt.total_seconds() > 0).sum()
  ).collect()           # -> 728
  ```

### F11. LFPG's whole timestamp regime moves in 2026-07

- **What changes:** the equality structure of §F2 at LFPG, which is stable through all twelve training
  months, breaks in the second ranking month.

  | LFPG DEP, equality rate | 2025-04 | 2025-08 | 2025-12 | 2026-01 | **2026-07** |
  |---|---|---|---|---|---|
  | `LOBT == EOBT1` | 0.7345 | 0.7437 | 0.7832 | 0.7810 | **0.5351** |
  | `IOBT == EOBT1` | 0.7339 | 0.7432 | 0.7827 | 0.7807 | **0.5341** |
  | `SCHED == EOBT1` | 0.6744 | 0.6437 | 0.5797 | 0.5442 | **0.3725** |
  | `EOBT1 == AOBT3` (±6 s) | 0.0463 | 0.0359 | 0.0463 | 0.0463 | **0.0751** |
  | `ARVT3` on the minute grid (ARR) | 0.775 | 0.910 | 0.805 | **0.597** | 0.877 |

  The training range for `LOBT == EOBT1` at LFPG is 0.720–0.783 across twelve months; 2026-07's
  0.535 is far outside it. `SCHED == EOBT1` also trends downward through 2025 (0.674 → 0.580) and
  continues (0.544, 0.373).
- **Rows:** 39,138 ranking LFPG DEP rows, of which 19,254 in 2026-07 (the month with the shift).
- **Identifiable in the ranking set:** yes.
- **Query:** `eq_cell.parquet` filtered to `apt = "LFPG", PHASE_mvt = "DEP", B = "EOBT1"`; the script
  writes it as `extras_lfpg_eobt1_by_month`.

### F12. `ARVT_3` is a landing time, `BLOCK` on ARR rows is the in-block time — verified, not assumed

- Of the 315,269 training `FLIGHT_ID_mvt` values that appear as both a DEP and an ARR row, the DEP
  row's `ARVT_3` is within 60 s of the ARR row's `MVT` on **274,432 (87.0 %)** and exactly equal on
  9,162 (2.9 %), but within 60 s of the ARR row's `BLOCK` on only **686 (0.22 %)** and exactly equal
  on 6. Ranking: 45,357 / 50,923 (89.1 %) near `MVT`, 64 near `BLOCK`.
- This fixes the semantics used by rules F3, F9 and F10 and is recorded here because no data
  dictionary in the repository states it.

## 3. What was checked and found CLEAN

Equally explicit, so these areas need not be re-audited.

### 3.1 Pairs with **no** copy cluster

Under the §0 copy test on DEP rows, these pairs have `copy_ratio` indistinguishable from 1 and are
therefore *not* copies, however large their raw equality rate looks:

| pair | raw equality rate train / rank | `copy_ratio` train / rank |
|---|---|---|
| `EOBT1 == AOBT3` | 0.0504 / 0.0513 | **1.04 / 1.04** |
| `IOBT == AOBT3` | 0.0430 / 0.0425 | 1.03 / 1.02 |
| `LOBT == AOBT3` | 0.0429 / 0.0425 | 1.03 / 1.03 |
| `SCHED == AOBT3` | 0.0350 / 0.0338 | 1.03 / 1.03 |
| `BLOCK == AOBT3` | 0.0065 (0.0789 at ±6 s) | 1.08 pooled; ≤ 1.10 at nine airports (LIRF 1.27, F5's neighbour) |
| `ARVT1 == ARVT3` | 0.00068 / 0.00071 | **1.00 / 0.97** |
| `MVT == AOBT3` | 2.2e-5 / 2.4e-5 | 1.44 / 1.87 on 539 / 116 rows |

So `AOBT_3_flt` is a genuinely independent actual-time measurement at every airport except for the
small EGLL excess of F6; the 103,632 training rows where it equals `EOBT_1` are the ordinary mass at
zero difference of a distribution whose median is +240 s. Per operator the `EOBT1 == AOBT3` rate runs
0.020–0.089 (95 training operators) and 0.032–0.080 (33 ranking operators) with no outlier.
`ARVT_1 == ARVT_3` likewise shows no copy anywhere, on any airport or operator (operator max 0.0030).

### 3.2 Ordering rules with zero or negligible violations

| rule | phase | train violations | rank violations |
|---|---|---|---|
| `AOBT3 ≤ ARVT3` | DEP and ARR | **0 of 2,062,577 / 0 of 2,067,301** | **0 of 339,551 / 0 of 342,141** |
| `EOBT1 ≤ ARVT1` | DEP and ARR | **0 / 0** | **0 / 0** |
| `BLOCK ≤ ARVT3` | DEP | 5 (2 > 1 h) | BLOCK null |
| `IOBT ≤ ARVT1` | DEP | 40 (0 > 1 h, max 1,735 s) | 5 (max 616 s) |
| `LOBT ≤ ARVT3` | DEP | 46 (1 > 1 h) | 10 (max 2,912 s) |
| `AOBT3 ≤ MVT` | ARR | 7 (max 1,500 s) | 1 (299 s) |
| `MVT ≤ BLOCK` | ARR | 220, of which 209 ≤ 60 s | 39, **all ≤ 12 s** |

`planned_flight_s = ARVT_1 − EOBT_1` is strictly positive on all 2,062,577 training and 339,551
ranking rows: 0 negative, 0 zero, minimum 584 s (train) / 654 s (rank).

### 3.3 Negative taxi-out is quantisation, not a data error

`BLOCK > MVT` on DEP rows: **369 of 2,085,047 training rows (0.000177), every one of them by at most
12 s** (median 5–8.5 s), i.e. inside the ±6 s grid of §3.5 applied to two stamps. Airports: LSZH 298
(0.00221), LFPG 61 (0.000255), LEBL 5, EHAM 3, LTFM 2; EDDF, EDDM, EGLL, LEMD and LIRF have none.
There is no row on which the off-block time is grossly after take-off. (Not checkable on ranking DEP
rows — `BLOCK` is null there.)

### 3.4 No sentinel, epoch, midnight or sub-second values

Across all nine columns, both sets, both phases: **0 rows** with a year before 2024 or after 2026;
**0 rows** with a non-zero microsecond component. Exact `00:00:00` is *under*-represented, not a
sentinel: e.g. `EOBT_1` on training DEP rows has 148 midnight values out of 2,062,577
(7.2e-5 against 6.9e-4 expected if minutes were uniform); the largest midnight count anywhere is
1,379 (`EOBT_1`, training ARR, 2,067,301 rows). Value ranges are exactly the expected windows:
training `MVT` 2025-01-01 00:05:56 … 2025-12-31 23:57:36, ranking `MVT` 2026-01-01 00:01:55 …
2026-07-31 23:56:19; the only stamps reaching outside their month set are `SCHED` (back to 2024-12-28
on ARR rows) and `ARVT1`/`ARVT3` (forward to 2026-01-01 on training DEP rows, 2026-08-01 on ranking
DEP rows) — both of which are correct for overnight flights.

Null rates are exactly two values and are stable: `MVT`, `SCHED` 0 everywhere; `BLOCK` 0 on DEP and
ARR in training (one single null on a training ARR row) and 1.0 on ranking DEP / 0 on ranking ARR;
the six `_flt` stamps share one null pattern — 0.010777 on training DEP, 0.007418 on training ARR,
0.015340 on ranking DEP, 0.007404 on ranking ARR — i.e. they are null together, always.

### 3.5 Rounding and precision (extends the prior audit's MVT result)

- The 13-value second grid `{54…59, 0…6}` found by the prior audit for `MVT` at EDDM/EGLL/LFPG/LIRF/
  LTFM applies **identically to `BLOCK`** at the same airports, with the same shares:
  second 54 → 0.0617–0.0637, seconds 55–59 and 0–5 → 0.0817–0.0847 each, second 6 → 0.0206–0.0215
  (training DEP; ranking DEP `MVT` 0.0620–0.0876 / 0.0203–0.0209). So the grid is a property of the
  airport's movement feed, not of one column. `BLOCK` is also on the grid at LEBL (0.9995 of DEP
  rows within 6 s), LEMD (0.9999) and EGLL (1.0000); it is *not* at EDDF (0.217), EHAM (0.223) or
  LSZH (0.249), which report true seconds.
- Column-level precision, pooled over airports, DEP rows (train / rank):
  `SCHED` `P(sec = 0)` **1.0000 / 1.0000** and 0.9997 / 0.9996 on a 5-minute grid;
  `IOBT` **1.0000 / 1.0000** and 0.954 / 0.949;
  `LOBT` 0.9969 / 0.9977 and 0.950 / 0.944;
  `EOBT1` 0.9970 / 0.9978 and 0.851 / 0.840;
  `AOBT3` 0.977 / 0.974 and 0.199 / 0.198;
  `ARVT3` 0.300 / 0.325; `MVT` 0.0513 / 0.0515; `BLOCK` 0.0640 / — .
  `ARVT_1` is the only column with genuinely uniform seconds everywhere: `P(within 6 s)`
  0.2135–0.2230 at all ten airports, both sets, both phases (13/60 = 0.2167).
- All of the above is the same in 2025 and in 2026 for every column except `ARVT_3` (finding F7).

### 3.6 Derived columns match the raw timestamps exactly

In `feat_train.parquet` (2,085,047 rows) and `feat_rank.parquet` (344,841 rows), all eight of
`mvt_minus_aobt3`, `mvt_minus_sched`, `mvt_minus_eobt1`, `mvt_minus_lobt`, `mvt_minus_iobt`,
`lobt_minus_eobt1`, `aobt3_minus_eobt1`, `aobt3_minus_sched` and also `planned_flight_s` equal the
corresponding raw difference to within 1e-6 s on **every row**, with **0** value mismatches and **0**
null-pattern mismatches in either file. `planned_flight_s` is null on exactly the 22,470 / 5,290 rows
where the `_flt` block is null.

### 3.7 Cross-row consistency of the `_flt` block

For the 315,269 training (50,923 ranking) DEP/ARR pairs sharing a `FLIGHT_ID_mvt`, all six `_flt`
stamps — `LOBT`, `IOBT`, `EOBT1`, `AOBT3`, `ARVT1`, `ARVT3` — are **identical on 315,234 / 315,234**
(50,923 / 50,923) of the pairs where both sides are non-null. There is no row on which the flight
plan differs between the departure and arrival view of the same flight. `MVT` and `BLOCK` never
agree (0 pairs), as they should not, and `SCHED` agrees on 108 / 18 pairs. Only **7 training and 2
ranking** pairs have the departure `MVT` after the arrival `MVT` of the same flight.

### 3.8 Monthly file partitioning

Each `training_YYYY-MM-01_*.parquet` file contains that month's `MVT_TIME` except for **6 single
rows** spread over 6 files, each holding one row from the previous month (2025-02, -06, -07, -08,
-09, -11). The partition key is evidently not `MVT_TIME`; no row is duplicated across files and the
totals match the profile exactly (4,167,797).

## 4. Facts and inference kept apart

**Facts** — everything in §2 and §3 above: every count, rate, quantile, copy ratio and per-airport /
per-month table, all re-derived by `timestamps.py` from `data/raw/*.parquet` and
`data/work/feat_*.parquet`.

**Inference — not measured here:**

- That the exact-equality clusters with high `copy_ratio` (F1, F2, F4, F5, F6) arise from one field
  being *written from* another in the source system, rather than from two independent processes
  happening to agree. What is measured is only that the equality mass is 5× to 4,600× the local
  density, which no smooth difference distribution can produce.
- That the rows in F8 are date errors rather than genuine week-long gate occupancies. What is
  measured is that the residual after removing whole days is in the ordinary range for that
  duration, and that `BLOCK` and `SCHED` agree on the wrong date while `MVT` sits on the right one.
- That the `AOBT_3 > MVT` rows of F3 are a reporting error rather than a return-to-stand and second
  departure recorded under one movement id.
- That the `ARVT_3` precision switches of F7 are changes in the upstream feed rather than a change
  in the mix of data providers per airport-month.
- Any statement about what these findings do to a model's error. Nothing in this lens was trained,
  scored or compared against an OOF file.
- The direction of causality in F11: whether LFPG's 2026-07 shift is a change in the NM feed, in the
  airport's schedule publication, or in the composition of flights that month.

**Definitional notes for cross-item comparison:** the prior audit's `f10` (`|BLOCK − SCHED| ≤ 10 s`)
and `f6` (≤ 6 s) are raw rates; §F5 reports the same rows against a local-density baseline, which is
why "LEBL 5.7 %" and "LEBL excess 2,902 rows (29 % of the bucket)" are both correct and not in
conflict. The prior audit examined DEP rows only; every ARR-row number here (F4, F8, F9, F12, §3.2,
§3.7) is new territory rather than a restatement.
