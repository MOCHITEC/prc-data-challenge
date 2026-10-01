# What the large-`ms` rows actually are (2026-09-24)

**Script (read-only, no training, ~3 min):**
`docs/competitor_research_2026-09-21/scripts/large_ms_population.py`, run as
`POLARS_MAX_THREADS=4 .venv/bin/python docs/competitor_research_2026-09-21/scripts/large_ms_population.py`.
Every number below carries the section tag (`S0`…`S8c`) of that script that prints it.
Print order is `S0 S1 S1b S2 S2b S2c S3 S3b S4 S4b S4c S4d S4e S5 S5b S6 S8 S8b S8c S7 S7b S7c S7d`.

**Inputs (never written).** `data/work/feat_train.parquet` (2,085,047 rows, all `PHASE_mvt == "DEP"`),
`data/work/feat_rank.parquet` (344,841), `experiments/oof_main_reparam-bc60017878.parquet`
(672,428 non-null `oof`, RMSE 281.8411) with its `_folds` sidecar.

**Notation.** `y = TAXITIME_SEC_mvt`; `ms = mvt_minus_sched = MVT − SCHED`; `bs = BLOCK − SCHED`.
`y = MVT − BLOCK` holds on all 2,085,047 rows (0 violations at 1 s tolerance) [S] `S0`, so
`bs ≡ ms − y` — **`bs` is a label-side quantity**, `BLOCK_TIME_UTC_mvt` is null on 344,841 / 344,841
scored rows. Lane = the routing definition the code uses: unmatched ⇔ `mvt_minus_aobt3` null
(22,470 train / 5,290 rank).

**Bands.** `<30m`, `30-45m`, `45-60m`, `60-75m`, `75-90m`, `90-120m`, `2-3h`, `3-4h`, `4-6h`,
`6-8h`, `8-12h`, `>=12h` (edges 1800, 2700, 3600, 4500, 5400, 7200, 10800, 14400, 21600, 28800,
43200 s). Finer below 2 h because that is where the mass is.

**Sidecar bug worth recording.** `oof_main_reparam-bc60017878_folds.parquet` has 863,141 rows for
672,428 OOF rows: `jul` and `jul_fwd` list the same 190,713 `MVT_ID_mvt`. A naive join duplicates
every July row and silently reports RMSE 288.66 instead of 281.84. The script deduplicates. [S] `S5`

---

## 0. The answer in six sentences

1. **The population is not rare and it is not the tail.** `ms ≥ 30 min` covers **733,271 of
   2,085,047 training rows (35.2 %)** and **136,521 of 344,841 scored rows (39.6 %)**; the
   `≥ 12 h` band that the previous investigation classified is 626 training rows (0.09 % of it). [S] `S1`, `S6`
2. **Almost all of it is late pushback, not long taxi.** Median `y` is flat at **1,014–1,085 s in
   every band from 30-45m to ≥12h on the matched lane**; `bs` absorbs essentially all of `ms`.
   90.9 % of `ms ≥ 1800` rows have `y < 1800`. [S] `S1`, `S2`
3. **Four kinds exist beyond that, all label-defined, and three of them are LIRF-only.** The
   schedule-fill (`BLOCK == SCHED`) is a genuine 5.35× spike **only at LIRF**; at the other nine
   airports the `|bs| ≤ 10 s` window holds *fewer* rows than the neighbouring windows. [S] `S3b`
4. **The crux: the kinds are not identifiable, but the cell is.** Inside a fixed `ms` band, `ms`
   itself has AUC 0.48–0.61 for `y ≥ 1800` in both lanes — no information. In the **matched** lane
   the NM clocks do separate it (`mvt_minus_eobt1` AUC 0.76–0.87) and are already model-A features.
   In the **unmatched** lane there are no clocks at all (all four are 100 % null), and the only
   separator is **`ADEP == LIRF`**, equivalently `AIRCRAFT_TYPE_mvt` null — also already a model
   categorical. [S] `S0`, `S4`, `S4d`, `S4e`
5. **One genuine, prediction-time-visible, systematic mis-prediction survives**: the model
   **over-predicts LIRF unmatched rows with `ms ≥ 1800`** by +710 to +4,136 s in the 75m–4h bands
   (t = 4.1–5.1), because the LIRF schedule-fill rate is **seasonal** (0.33 in July, 0.76 in
   December) and the model holds a year-average weight. The scored frame is months 1 and 7 — the two
   lowest-fill months of 2025. [S] `S7b`, `S7c`
6. **Price: 2.6–3.4 s of board RMSE, on 356 scored rows, contingent on 2026 behaving like 2025.**
   Everything else prices at or below 0.03 s. [S] `S7c`, `S7d`

---

## 1. Map of the population (task item 1)

### 1.1 Where the mass is

`S1`. Query: `band_expr(ms)` × `mvt_minus_aobt3.is_null()` on `feat_train.parquet`, aggregating
`len()`, quantiles of `y`, quantiles of `bs = ms − y`.

| band | matched n | unmatched n | total | % of `ms≥1800` | `y` p50 (mat) | `y` p50 (unm) | `y` p99 (mat) | `bs` p50 (mat) | frac `bs<0` (mat) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `<30m` | 1,346,139 | 5,637 | 1,351,776 | — | 852 | 710 | 1,732 | 121 | 0.365 |
| **30-45m** | 378,785 | 797 | **379,582** | **51.8 %** | 1,070 | 1,017 | 2,341 | 1,081 | 0.022 |
| **45-60m** | 157,705 | 325 | **158,030** | **21.6 %** | 1,079 | 1,014 | 3,006 | 1,979 | 0.008 |
| **60-75m** | 74,426 | 844 | **75,270** | **10.3 %** | 1,079 | 652 | 3,773 | 2,883 | 0.006 |
| **75-90m** | 36,900 | 4,183 | **41,083** | **5.6 %** | 1,085 | 928 | 4,561 | 3,780 | 0.005 |
| **90-120m** | 33,719 | 5,735 | **39,454** | **5.4 %** | 1,030 | 1,083 | 4,863 | 5,036 | 0.003 |
| 2-3h | 22,070 | 3,597 | 25,667 | 3.5 % | 1,020 | 1,113 | 4,391 | 7,332 | 0.001 |
| 3-4h | 6,557 | 810 | 7,367 | 1.0 % | 1,022 | 1,144 | 3,717 | 11,064 | 0.001 |
| 4-6h | 4,022 | 334 | 4,356 | 0.59 % | 1,014 | 1,144 | 3,114 | 15,604 | 0.000 |
| 6-8h | 1,104 | 94 | 1,198 | 0.16 % | 966 | 1,077 | 2,587 | 23,522 | 0.000 |
| 8-12h | 599 | 39 | 638 | 0.09 % | 968 | **29,282** | 2,762 | 32,082 | 0.000 |
| ≥12h | 551 | 75 | 626 | 0.09 % | 1,059 | 1,431 | 2,492 | 64,717 | 0.002 |

**The five bands 30-45m … 90-120m are 693,419 rows = 94.6 % of the `ms ≥ 1800` population.** [S]
The bands the previous work covered (`≥12h`) are 626 rows, and its 75-row figure was the unmatched
slice of them.

**The single most important row in this table is the `y` p50 column.** It does not move. A matched
row with `ms` of eleven hours has the same median taxi time (1,059 s) as a matched row with `ms` of
35 minutes (1,070 s). `ms` is a pushback-delay meter, not a taxi meter, and the `bs` p50 column
(1,081 → 64,717) says where all of it went. [S] `S1`

### 1.2 Per airport

`S1b`. `ms ≥ 1800` totals, and the `≥12h` extreme:

| ADEP | n (`ms≥1800`) | n unmatched | n `y≥1800` | n `\|bs\|≤10` | `y` p50 | `bs` p50 | n in `≥12h` band |
|---|---:|---:|---:|---:|---:|---:|---:|
| LFPG | 104,322 | 3,575 | 8,444 | 291 | 1,072 | 1,617 | 38 |
| EGLL | 99,282 | 1,411 | **22,341** | 1,149 | **1,501** | 1,026 | 2 |
| EHAM | 85,289 | 2,466 | 2,226 | 27 | 822 | 1,852 | 120 |
| EDDF | 80,956 | 1,822 | 2,296 | 29 | 967 | 1,651 | 360 |
| LEMD | 77,679 | 936 | 3,230 | 180 | 1,093 | 1,559 | 7 |
| LTFM | 76,944 | 1,322 | 11,113 | 550 | 1,140 | 1,384 | 26 |
| LEBL | 61,157 | 1,731 | 2,678 | 484 | 1,079 | 1,740 | 14 |
| **LIRF** | 61,016 | 1,412 | 11,519 | **5,195** | 1,196 | 1,440 | 31 |
| LSZH | 43,891 | 1,185 | 980 | 7 | 804 | 1,729 | 2 |
| EDDM | 42,735 | 973 | 2,136 | 55 | 890 | 1,675 | 26 |

EGLL is the long-taxi airport (`y` p50 1,501, 22.5 % of its large-`ms` rows have `y ≥ 1800`).
LIRF is the schedule-fill airport. Everything else is unremarkable. [S] `S1b`

---

## 2. The kinds (task item 2)

### 2.1 Definitions

Ordered and mutually exclusive, applied to training rows with `ms ≥ 1800`. **Every one of these is
defined using the label.** Section 3 asks separately whether it can be recognised without it.

| kind | definition | n | share | n unmatched | `y` p50 | `bs` p50 |
|---|---|---:|---:|---:|---:|---:|
| **latepush** | `y < 1800` — `ms` is pushback delay | **666,301** | **90.87 %** | 14,867 | 1,019 | 1,623 |
| **modtaxi** | `1800 ≤ y < 3600` | 56,595 | 7.72 % | 884 | 2,047 | 717 |
| **schedfill** | `\|bs\| ≤ 10` (`BLOCK == SCHED`, so `y == ms`) | 7,965 | 1.09 % | 710 | 2,292 | 0 |
| **longtaxi** | `y ≥ 3600` | 2,396 | 0.33 % | 359 | 4,260 | 894 |
| **rollover** | `y > 72,000` (BLOCK date one day off) | 14 | 0.002 % | 13 | 87,421 | −21,245 |

Query: `kind_expr()` in the script; `S2`. The partition is near-exhaustive by construction —
`ms = bs + y`, so for `ms ≥ 3600` a row cannot have both `y < 1800` and `bs < 1800`. There is no
"other" category; nothing fell outside. [S]

### 2.2 The schedule fill is LIRF-only — the other airports' `bs ≈ 0` rows are coincidence

`S3b`. `|bs| ≤ 10` is a 21-second window. Compare its occupancy with the mean of four equal-width
windows displaced by ±600 and ±1,200 s at the same airport (`ms ≥ 1800`):

| ADEP | n | n at `\|bs\|≤10` | expected from neighbours | excess | ratio | of which unmatched |
|---|---:|---:|---:|---:|---:|---:|
| **LIRF** | 61,016 | **5,195** | 971.8 | **+4,223** | **5.35×** | 691 |
| EGLL | 99,282 | 1,149 | 1,412.8 | −264 | 0.81× | 3 |
| LEBL | 61,157 | 484 | 700.8 | −217 | 0.69× | 2 |
| LTFM | 76,944 | 550 | 899.0 | −349 | 0.61× | 13 |
| LFPG | 104,322 | 291 | 1,143.2 | −852 | 0.25× | 1 |
| LEMD | 77,679 | 180 | 909.5 | −730 | 0.20× | 0 |
| EDDM | 42,735 | 55 | 467.5 | −413 | 0.12× | 0 |
| EHAM | 85,289 | 27 | 269.8 | −243 | 0.10× | 1 |
| EDDF | 80,956 | 29 | 311.2 | −282 | 0.09× | 0 |
| LSZH | 43,891 | 7 | 160.5 | −154 | 0.04× | 0 |

**Only LIRF has a spike.** [S] Everywhere else the count at `bs ≈ 0` is at or below the local
density of `bs`, so the EGLL 1,149 / LEBL 484 / LTFM 550 rows in the `schedfill` row of §2.1 are
ordinary long-taxi rows that happen to have `BLOCK` within 10 s of `SCHED`, not a fill. **The
correct count of the schedule-fill kind is ~4,200 excess rows, all LIRF** (5,195 observed −
971.8 expected), of which ~690 are in the unmatched lane. `bs == 0` exactly: LIRF 456, next EGLL 91.

### 2.3 Per-airport spread of each kind

`S2b`, rates = kind n / that airport's `ms ≥ 1800` n:

| ADEP | latepush | modtaxi | longtaxi | `\|bs\|≤10` | rollover |
|---|---:|---:|---:|---:|---:|
| EDDF | 0.972 | 0.027 | 0.0009 | 0.0004 | 0 |
| EDDM | 0.950 | 0.047 | 0.0013 | 0.0013 | 0 |
| **EGLL** | **0.775** | **0.206** | 0.0071 | 0.0116 | 0 |
| EHAM | 0.974 | 0.025 | 0.0009 | 0.0003 | 0 |
| LEBL | 0.956 | 0.036 | 0.00002 | 0.0079 | 0 |
| LEMD | 0.958 | 0.039 | 0.0000 | 0.0023 | 0 |
| LFPG | 0.919 | 0.074 | 0.0037 | 0.0028 | 0 |
| **LIRF** | 0.811 | 0.099 | 0.0049 | **0.0851** | **0.00023** |
| LSZH | 0.978 | 0.022 | 0.0005 | 0.0002 | 0 |
| **LTFM** | 0.856 | 0.127 | 0.0101 | 0.0072 | 0 |

**Kinds that exist only at certain airports:** the fill is LIRF-only (§2.2); the rollover is LIRF
(14 of 14 at `ms ≥ 1800`; the other two of the 16 known rollovers sit at `ms < 1800` at LFPG and
LSZH, which is why they are invisible in this cut). `longtaxi` is essentially absent at LEMD (0) and
LEBL (1) and concentrated at LTFM (778), EGLL (702), LIRF (299), LFPG (385). [S] `S2b`, `S2c`

### 2.4 The unmatched lane, kind by airport

`S2c`, `ms ≥ 1800`, unmatched only (16,833 rows):

| ADEP | latepush | modtaxi | longtaxi | schedfill | rollover |
|---|---:|---:|---:|---:|---:|
| LFPG | 3,275 | 256 | 43 | 1 | 0 |
| EHAM | 2,387 | 68 | 10 | 1 | 0 |
| EDDF | 1,786 | 33 | 3 | 0 | 0 |
| LEBL | 1,674 | 55 | 0 | 2 | 0 |
| EGLL | 1,168 | 198 | 42 | 3 | 0 |
| LSZH | 1,164 | 18 | 3 | 0 | 0 |
| LTFM | 971 | 123 | 215 | 13 | 0 |
| LEMD | 900 | 36 | 0 | 0 | 0 |
| EDDM | 896 | 67 | 10 | 0 | 0 |
| **LIRF** | **646** | **30** | **33** | **690** | **13** |

**Non-LIRF unmatched rows are 20 schedule-fills out of 15,421 (0.13 %).** LIRF unmatched rows are
49 % fill. These are two different populations wearing the same lane label. [S] `S2c`

---

## 3. The crux — is the kind identifiable at prediction time? (task item 3)

### 3.1 The unmatched lane has no clocks at all

`S0`. On all 22,470 unmatched training rows: `mvt_minus_aobt3` non-null **0**, `mvt_minus_lobt` **0**,
`mvt_minus_eobt1` **0**, `mvt_minus_iobt` **0**. `AIRCRAFT_OPERATOR_flt` non-null **0**,
`MARKET_SEGMENT_flt` **0**. (46 rows have `nm_matched == 1` with `aobt3_null == 1`; they carry no
clock either.) So "which `_flt` columns are null" is not a feature — it is the lane indicator, exactly.
**Model B has no off-block observation and no operator identity.** [S]

### 3.2 `ms` itself carries nothing within a band

`S4d`. AUC for `y ≥ 1800` inside each band, matched lane:

| band | n | base | **ms** | `mvt_minus_aobt3` | `mvt_minus_lobt` | **`mvt_minus_eobt1`** | `hour_utc` | `congestion_vs_usual_30m` | `dest_distance_km` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30-45m | 378,785 | 0.076 | 0.564 | 0.760 | 0.731 | **0.762** | 0.476 | 0.451 | 0.595 |
| 45-60m | 157,705 | 0.108 | 0.507 | 0.738 | 0.730 | **0.769** | 0.443 | 0.448 | 0.600 |
| 60-75m | 74,426 | 0.115 | 0.498 | 0.730 | 0.725 | **0.765** | 0.431 | 0.459 | 0.605 |
| 75-90m | 36,900 | 0.120 | 0.507 | 0.714 | 0.797 | **0.805** | 0.417 | 0.456 | 0.598 |
| 90-120m | 33,719 | 0.105 | 0.479 | 0.736 | 0.831 | **0.838** | 0.427 | 0.473 | 0.585 |
| 2-3h | 22,070 | 0.088 | 0.493 | 0.710 | 0.811 | **0.826** | 0.447 | 0.494 | 0.565 |
| 3-4h | 6,557 | 0.079 | 0.484 | 0.723 | 0.813 | **0.835** | 0.431 | 0.484 | 0.520 |
| 4-6h | 4,022 | 0.070 | 0.503 | 0.699 | 0.794 | **0.816** | 0.513 | 0.469 | 0.551 |
| 6-8h | 1,104 | 0.052 | 0.509 | 0.704 | 0.822 | **0.853** | 0.500 | 0.443 | 0.490 |
| 8-12h | 599 | 0.072 | 0.500 | 0.692 | 0.664 | **0.715** | 0.462 | 0.530 | 0.479 |
| ≥12h | 551 | 0.058 | 0.513 | 0.778 | 0.824 | **0.871** | 0.440 | 0.480 | 0.549 |

Unmatched lane, same test:

| band | n | base | ms | `hour_utc` | `dow` | `queue_ahead` | `dest_distance_km` | **`actype_null`** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 30-45m | 797 | 0.124 | 0.563 | 0.559 | 0.511 | 0.495 | 0.579 | 0.597 |
| 45-60m | 325 | 0.157 | 0.604 | 0.407 | 0.520 | 0.414 | 0.548 | 0.593 |
| 60-75m | 844 | 0.062 | 0.413 | 0.468 | 0.515 | 0.334 | 0.611 | 0.596 |
| 75-90m | 4,183 | 0.034 | 0.525 | 0.528 | 0.477 | 0.251 | 0.483 | **0.755** |
| 90-120m | 5,735 | 0.114 | 0.610 | 0.454 | 0.505 | 0.402 | 0.525 | 0.638 |
| 2-3h | 3,597 | 0.173 | 0.532 | 0.477 | 0.531 | 0.336 | 0.518 | 0.675 |
| 3-4h | 810 | 0.235 | 0.493 | 0.413 | 0.581 | 0.311 | 0.539 | 0.657 |
| 4-6h | 334 | 0.252 | 0.493 | 0.495 | 0.561 | 0.242 | 0.488 | **0.770** |
| 6-8h | 94 | 0.160 | 0.431 | 0.354 | 0.532 | 0.166 | 0.428 | **0.894** |
| ≥12h | 75 | 0.427 | 0.326 | 0.429 | 0.384 | 0.216 | 0.575 | **0.953** |

**Conditioning on the band, `ms` is uninformative in both lanes** (0.48–0.61). Hour, day of week,
stand/runway nullness, and the traffic-context columns are all at or below 0.6 and several are
*inverse* (`queue_ahead` 0.17–0.50, `dep_tko_prev_30m` 0.25–0.47) — a high-`ms` row happens in quiet
traffic, which is a composition fact about delayed departures, not a taxi predictor. [S] `S4d`

### 3.3 What *is* identifiable — matched lane

`mvt_minus_eobt1`, `mvt_minus_lobt` and `mvt_minus_aobt3` separate long taxi from late pushback at
AUC 0.69–0.87 in every band. **All three are already model-A features** (`src/prc/train.py:405-419`
hands `mvt_minus_aobt3` to model A as the proxy). So the split *is* resolved at prediction time and
the model already holds the instrument. [S] `S4d`

The resolution is not perfect: over `ms ≥ 1800` matched rows (n = 716,438), `mvt_minus_aobt3 − y`
has p50 −6 s but only **35.9 % within ±120 s** and 69.4 % within ±300 s; using it directly as `ŷ`
gives RMSE **507.4**, bias −78.1, p25 −236, p75 +140. AOBT_3 is *not* a copy of BLOCK. [S] `S3`, `S4d`

### 3.4 What is identifiable — unmatched lane: only `ADEP == LIRF`

`S4e`. `ms ≥ 1800`, unmatched, split on LIRF (equivalently `AIRCRAFT_TYPE_mvt` null — 383 of the
388 `actype_null` unmatched scored rows are LIRF, `S6`):

| band | non-LIRF n | non-LIRF `P(y≥1800)` | non-LIRF `E[y]` | LIRF n | LIRF `P(y≥1800)` | **LIRF `E[y]`** |
|---|---:|---:|---:|---:|---:|---:|
| 30-45m | 773 | 0.101 | 1,087 | 24 | 0.875 | 3,696 |
| 45-60m | 312 | 0.131 | 1,110 | 13 | — | — |
| 60-75m | 819 | 0.050 | 784 | 25 | 0.440 | 2,646 |
| 75-90m | 3,918 | 0.016 | 972 | 265 | 0.298 | 2,182 |
| 90-120m | 5,277 | 0.084 | 1,185 | 458 | 0.465 | 3,370 |
| 2-3h | 3,213 | 0.117 | 1,297 | 384 | 0.643 | 5,884 |
| 3-4h | 694 | 0.169 | 1,401 | 116 | 0.629 | 8,022 |
| 4-6h | 270 | 0.126 | 1,377 | 64 | 0.781 | **13,767** |
| 6-8h | 81 | 0.037 | 1,076 | 13 | — | — |
| 8-12h | — | — | — | 21 | 1.000 | **37,742** |
| ≥12h | 46 | 0.065 | 1,206 | 29 | 1.000 | **67,859** |

**Non-LIRF unmatched `E[y]` is flat at 784–1,401 s at every band up to twelve hours.** Large `ms`
there means *nothing at all* about the taxi. LIRF unmatched `E[y]` rises by a factor of 18.
`ADEP_mvt` and `AIRCRAFT_TYPE_mvt` are both already model categoricals (`src/prc/features.py:33-44`).
[S] `S4e`

### 3.5 Inside LIRF unmatched: the callsign prefix is a real but partly redundant handle

The fill rate among LIRF unmatched `ms ≥ 1800` rows is 0.489 (690 / 1,412). `S4b` by 3-letter
callsign prefix (whole year, n ≥ 8, selected rows):

| prefix | n | fill rate | `y` p50 | prefix | n | fill rate | `y` p50 |
|---|---:|---:|---:|---|---:|---:|---:|
| ISR | 58 | **1.000** | 6,983 | RYR | 237 | 0.333 | 1,206 |
| LAV | 23 | **1.000** | 7,747 | ITY | 86 | 0.116 | 1,089 |
| ETH | 21 | **1.000** | 6,836 | EJU | 40 | 0.100 | 838 |
| TKJ | 20 | **1.000** | 6,776 | EZY | 25 | 0.080 | 1,145 |
| TWB | 19 | **1.000** | 8,635 | KLM | 17 | **0.000** | 731 |
| MSC | 12 | **1.000** | 9,245 | EWG | 14 | **0.000** | 1,015 |
| KMM | 18 | 0.944 | 7,525 | BAW | 12 | **0.000** | 1,406 |
| AEZ | 46 | 0.826 | 6,934 | ELY | 12 | **0.000** | 1,257 |
| WZZ | 57 | 0.737 | 6,790 | UAL | 17 | 0.059 | 2,336 |
| WMT | 175 | 0.486 | 1,689 | WJA | 8 | **0.000** | 1,197 |

Out-of-time test (`S8`): fit the prefix fill-rate table on one half-year, score the other.

| split | fit n | test n | prefix known | **AUC(prefix rate)** | AUC(ADES rate) | AUC(ms) | AUC(hour) |
|---|---:|---:|---:|---:|---:|---:|---:|
| H1 → H2 | 489 | 923 | 0.696 | **0.7748** | 0.6996 | 0.6669 | 0.5737 |
| H2 → H1 | 923 | 489 | 0.718 | **0.8024** | 0.7893 | 0.6797 | 0.5480 |

**`FLIGHT_mvt` is present on 344,783 of 344,841 scored rows and is NOT a model feature**
(`CATEGORICALS`, `src/prc/features.py:33-44`, lists `ADEP_mvt`, `RUNWAY_mvt`, `STAND_mvt`,
`AIRCRAFT_TYPE_mvt`, `WK_TBL_CAT_flt`, `AIRCRAFT_OPERATOR_flt`, `MARKET_SEGMENT_flt`,
`FLIGHT_TYPE_flt`, `ADES_mvt`, `ADES_REGION`; `AIRCRAFT_OPERATOR_flt` is 100 % null in this lane).
So operator identity is genuinely unavailable to model B today. **But `ADES_mvt` is a categorical
the model already has and it recovers AUC 0.70 / 0.79 of the same signal**, so the prefix's
incremental content over what the model already sees is modest. [X] Treating the prefix as an
unexploited lever would be overclaiming; it is an unexploited *column*, with a measured
0.08 / 0.01 AUC margin over one the model holds.

Hard-assigning on the prefix rate (predict `ms` if rate ≥ 0.5, else 1,000 s) gives RMSE 8,029 /
9,938 against `sd(y)` of 11,262 / 11,742 — better than nothing, far worse than the fitted model
(§4.3). [S] `S8`

### 3.6 Explicit statement per kind

| kind | identifiable WITHOUT the label? | what would identify it | evidence |
|---|---|---|---|
| **latepush** | **Partly.** In the matched lane, yes, via `mvt_minus_eobt1`/`lobt`/`aobt3` (AUC 0.69–0.87), already used. In the unmatched lane, only through `ADEP ≠ LIRF`, which makes it 96 % of that cell. | NM clocks / LIRF flag | `S4d`, `S4e` |
| **modtaxi** | Same as latepush — the same instruments, the same AUCs. The boundary `y = 1800` is not separately visible. | — | `S4d` |
| **longtaxi** | **No.** In the unmatched lane no feature reaches AUC 0.62 for it (best: `dest_distance_km` 0.615, `ms` 0.593, all others inside 0.36–0.42). In the matched lane the clocks give 0.59–0.79, which is the same instrument as above, not a separate handle. | — | `S4`, `S4d` |
| **schedfill** | **Partly, and only as a rate.** `ADEP == LIRF` ∧ unmatched raises the base rate from 0.0013 to 0.489 (a 376× lift on a fully visible condition). Inside that cell, the callsign prefix gives out-of-time AUC 0.77–0.80, `ADES_mvt` 0.70–0.79, `ms` 0.67. No rule identifies an individual row. | `ADEP`, `AIRCRAFT_TYPE` nullness, `ms` band, callsign prefix | `S2c`, `S4b`, `S8` |
| **rollover** | **No.** 16 rows a year. `ms` has AUC 0.994 within the unmatched lane but the base rate is 0.0008 and every `ms ≥ 12 h` LIRF unmatched row is *either* a rollover *or* a fill (12 / 17 in the ≥12h band) — `ms` tells you the row is one of the two, never which. The two non-LIRF rollovers (LFPG EJU983W, EJU42AY) have `ms` of 1,740 and 2,043 and look completely ordinary. | nothing | `S4`, `S4b`, and `block-day-rollover.md`, `lfpg-unmatched-lane.md:§1.1` |

**Plainly: none of the five kinds is individually identifiable on a scored row.** What is
identifiable is the *cell* — `(lane, LIRF or not, ms band)` — inside which the kinds sit in a
mixture whose weights the cell pins down. Section 4 asks whether the model is already at that
mixture's optimum.

---

## 4. How well the current model does, per kind and per band (task item 4)

`S5`. OOF `oof_main_reparam-bc60017878`, 672,428 rows, RMSE **281.8411**, total SSE **5.3414e10**.
1 s of OOF RMSE = `672428 × (2×281.8411 + 1)` = **3.797e8** of SSE.

**`ms ≥ 1800` rows are 245,218 of 672,428 (36.5 %) and carry 67.9 % of total SSE.** [S] `S7`

### 4.1 By label-defined kind (`ms ≥ 1800`)

| kind | lane | n | RMSE | **bias** | share of total SSE | `n·bias²` |
|---|---|---:|---:|---:|---:|---:|
| latepush | matched | 213,395 | 222.0 | **+49.4** | 0.1969 | 5.21e8 |
| latepush | unmatched | 5,688 | 942.1 | **+221.1** | 0.0945 | 2.78e8 |
| modtaxi | matched | 22,134 | 522.9 | **−296.3** | 0.1133 | 1.94e9 |
| modtaxi | unmatched | 489 | 1,320.0 | **−461.6** | 0.0160 | 1.04e8 |
| longtaxi | matched | 839 | 1,520.0 | **−1,031.5** | 0.0363 | 8.93e8 |
| longtaxi | unmatched | 98 | 6,937.4 | **−3,801.5** | 0.0883 | 1.42e9 |
| schedfill | matched | 2,340 | 1,077.4 | **−754.6** | 0.0509 | 1.33e9 |
| schedfill | unmatched | 226 | 1,959.8 | **−1,314.3** | 0.0163 | 3.90e8 |
| rollover | matched | 1 | 2,144.8 | −2,144.8 | 0.0001 | 4.6e6 |
| rollover | unmatched | 8 | 21,104.6 | **−16,328.5** | 0.0667 | 2.13e9 |

**The model over-predicts late pushbacks and under-predicts every taxi-heavy kind, in both lanes,
in every band.** That is exactly what a conditional-mean predictor does on a mixture it cannot
resolve — the two biases are the two sides of the same regression to the cell mean, and they
offset. `S5` "by kind x band" shows the pattern holds band by band: `latepush` bias +36.6 →
+137.1 monotonically with the band, `schedfill` bias −489 → −1,964, `longtaxi` −1,061 → −1,749.
**None of this is a lever**, because the kind is not visible (§3.6). [X]

### 4.2 By visible cell — this is the one that counts

`S7b`. Cells are `(lane, LIRF/not, band)`, all three components visible on a scored row.

| cell | n (OOF) | RMSE | **bias** | n (scored) | s if perfect | s if bias closed |
|---|---:|---:|---:|---:|---:|---:|
| **A-matched**, all bands | 238,709 | 298.2 | **+5.67** | 132,236 | 60.20 | **0.022** |
| **B-other** (unmatched, not LIRF) | 5,997 | 962.8 | **−36.60** | 3,929 | 18.64 | **0.027** |
| **B-LIRF** (unmatched, LIRF) | 512 | 4,304.9 | **+881.3** | 356 | 33.77 | **1.416** |

Matched lane band by band (`S5`, "MATCHED lane by band"): bias **+1.0, +3.6, +4.6, +5.3, +6.6, +6.8,
+7.7, +13.0, +25.5, +29.8, +46.1** — never above 46 s, and the largest values sit on 197–388 rows.
The matched lane is calibrated. [S]

Non-LIRF unmatched band by band (`S5`, "UNMATCHED lane by band x (LIRF / not)"): bias **−260, −13,
+77, +50, −30, −119, −85, −16, −68, +735** — no pattern, no trend with the band, and the −260 on
263 rows in the 30-45m band is the two known corrupted LFPG labels
(`lfpg-unmatched-lane.md:§1.1`). [S]

**LIRF unmatched is the exception and the only one:**

| band | n | **bias** | SE | **t** | n scored | s if bias closed | s SE-adjusted |
|---|---:|---:|---:|---:|---:|---:|---:|
| 30-45m | 7 | −363 | 968 | 0.38 | 4 | 0.003 | 0.000 |
| 45-60m | 7 | −1,916 | 1,205 | 1.59 | 4 | 0.075 | 0.045 |
| 60-75m | 6 | +993 | 996 | 1.00 | 13 | 0.066 | 0.000 |
| **75-90m** | 88 | **+710** | 172 | **4.14** | 50 | 0.129 | **0.121** |
| **90-120m** | 165 | **+901** | 178 | **5.06** | 104 | 0.432 | **0.415** |
| **2-3h** | 137 | **+1,310** | 301 | **4.35** | 107 | 0.940 | **0.890** |
| **3-4h** | 47 | **+4,136** | 968 | **4.27** | 31 | 2.715 | **2.566** |
| 4-6h | 26 | +837 | 1,109 | 0.75 | 17 | 0.061 | 0.000 |
| 6-8h | 5 | −1,325 | 1,052 | 1.26 | 4 | 0.036 | 0.013 |
| 8-12h | 9 | +238 | 534 | 0.45 | 11 | 0.003 | 0.000 |
| ≥12h | 15 | **−9,402** | 3,994 | 2.35 | 11 | 4.978 | 4.079 |

SE-adjusted uses `max(0, b̂² − SE²)`, the unbiased estimate of the recoverable squared bias, since
`E[b̂²] = b² + SE²`. Four bands clear t = 4. [S] `S7c`

### 4.3 Why the LIRF-unmatched bias exists: the fill rate is seasonal

`S7c`, `S8`. OOF bias on LIRF unmatched `ms ≥ 1800` by month, against the training fill rate for
that month:

| month | fill rate (train, `ms≥1800`) | OOF n | OOF **bias** | `E[y]` | `E[pred]` |
|---|---:|---:|---:|---:|---:|
| **1** | **0.455** | 55 | **+1,204** | 7,930 | 9,133 |
| 2 | 0.769 | — | — | — | — |
| 3 | 0.796 | — | — | — | — |
| 4 | 0.672 | — | — | — | — |
| 5 | 0.495 | — | — | — | — |
| 6 | 0.483 | — | — | — | — |
| **7** | **0.326** | 328 | **+1,382** | 6,116 | 7,498 |
| 8 | 0.456 | — | — | — | — |
| 9 | 0.436 | — | — | — | — |
| 10 | 0.491 | — | — | — | — |
| 11 | 0.627 | 51 | **−1,226** | 9,539 | 8,313 |
| 12 | 0.756 | 78 | **−73** | 8,596 | 8,523 |

**The sign of the bias tracks the month's fill rate exactly.** [S] The model predicts close to the
year-average mixture weight (~0.49) and is therefore too high in low-fill months and too low in
high-fill ones. Note that `month` is *deliberately excluded* from the numeric feature set
(`src/prc/train.py:244`: `if c not in (LABEL, "month")`), so the model has no way to track this.

The same test on non-LIRF unmatched rows gives bias −60 (Jan), −47 (Jul), −18 (Nov), −1 (Dec) —
no seasonality, nothing to correct. [S] `S7c`

`S8b` also shows the model is **not** merely predicting the cell mean: its SSE beats the cell's own
realised-mean oracle in five of the six LIRF-unmatched bands with n ≥ 10 (e.g. 2-3h: 1.70e9 vs
2.12e9; ≥12h: 3.59e9 vs 5.58e9). It is already resolving part of the mixture, presumably through
`ADES_mvt`, `STAND_mvt` and `RUNWAY_mvt`. Only the 3-4h band (n = 47) is worse than its cell mean.
[S]

### 4.4 The biggest single rows

`S5`, top-20 by SSE within `ms ≥ 1800` — 1 LFPG corrupted label (EJU42AY, 3.28e9), 5 LIRF rollovers
(8.98e8 … 3.20e8), and **13 LIRF unmatched `latepush`/`modtaxi` rows where the model predicted
`ŷ ≈ ms`** (e.g. PGT2WE, `ms` 20,878, `y` 1,377, pred 20,694, SSE 3.73e8; RYR68SF, `ms` 15,301,
`y` 723, pred 12,209). Those 13 rows are the cost of the mixture, and they are the rows the §4.3
shift would pull down. [S]

---

## 5. How many such rows are in the 2026 scored frame (task item 5)

`S6`. 344,841 rows, months **1 and 7 only** (152,719 + 192,122).

| band | matched n | unmatched n | total | `ms` p50 (mat) |
|---|---:|---:|---:|---:|
| `<30m` | 207,315 | 1,005 | 208,320 | 1,079 |
| **30-45m** | 63,884 | 152 | **64,036** | 2,159 |
| **45-60m** | 29,157 | 66 | **29,223** | 3,065 |
| **60-75m** | 15,153 | 154 | **15,307** | 3,963 |
| **75-90m** | 7,926 | 884 | **8,810** | 4,886 |
| **90-120m** | 7,657 | 1,308 | **8,965** | 6,115 |
| 2-3h | 5,231 | 1,053 | 6,284 | 8,406 |
| 3-4h | 1,690 | 345 | 2,035 | 12,236 |
| 4-6h | 1,052 | 226 | 1,278 | 16,864 |
| 6-8h | 258 | 45 | 303 | 24,540 |
| 8-12h | 113 | 27 | 140 | 31,863 |
| ≥12h | 115 | 25 | 140 | 64,863 |
| **`ms ≥ 1800` total** | **132,236** | **4,285** | **136,521** | (39.6 % of the frame) |

`ms ≥ 3600`: **43,262**. Per airport (`ms ≥ 1800`): LFPG 18,328, EGLL 16,745, LTFM 16,412,
LEMD 15,417, EHAM 14,794, EDDF 14,313, LIRF 11,557, LEBL 11,231, LSZH 9,412, EDDM 8,312. [S]

**The identifiable cells, sized:**

| cell | scored rows | per-band detail |
|---|---:|---|
| A-matched, `ms ≥ 1800` | **132,236** | of which 9,910 have `mvt_minus_aobt3 ≥ 1800` and 600 have it ≥ 3600 — the visible "long taxi" candidates |
| B-other (unmatched, not LIRF), `ms ≥ 1800` | **3,929** | 148 / 62 / 141 / 834 / 1,204 / 946 / 314 / 209 / 41 / 16 / 14 by band |
| **B-LIRF** (unmatched, LIRF), `ms ≥ 1800` | **356** | 4 / 4 / 13 / 50 / 104 / 107 / 31 / 17 / 4 / 11 / 11 by band; 91 in month 1, 265 in month 7 |

LIRF unmatched total (any `ms`) = **383** (107 in month 1, 276 in month 7); `AIRCRAFT_TYPE_mvt` is
null on all 383, and on only 5 other unmatched rows in the whole frame (LTFM 4, LSZH 1). **The
`ADEP == LIRF ∧ unmatched` cell is exactly recoverable at prediction time.** [S] `S6`, `S8c`

Prefix coverage on those 356 rows (`S8c`): 319 have a prefix seen in the training LIRF-unmatched
table; by fitted fill rate — ≥0.8: **51 rows**; 0.5–0.8: 27; 0.2–0.5: 138; <0.2: 103; unseen: 37.

**A kind with 3 rows is a curiosity; a kind with 3,000 is a target.** By that test: the rollover is
a curiosity (expected ~2 rows in the scored frame, from 16/year over 12 months against 2 scored
months); the LIRF fill is **~170 rows** (356 × 0.49, or ~120 if the 2025 Jan/Jul rates of 0.455 and
0.326 hold); the non-LIRF unmatched large-`ms` population is 3,929 rows and is **unbiased**; the
matched large-`ms` population is 132,236 rows and is **calibrated to within 46 s in every band**.

---

## 6. Price it honestly (task item 6)

`N = 344,841`, `RMSE = 282.7371`, so **1 s of board RMSE = `344841 × (2×282.7371 + 1)` = 1.9534e8**
of squared error. `S7`.

### 6.1 Ceilings (not achievable, stated to bound the rest)

| population | scored rows | OOF SSE/row | **s if perfect** |
|---|---:|---:|---:|
| all `ms ≥ 1800` | 136,521 | 1.479e5 | **103.4** |
| A-matched `ms ≥ 1800` | 132,236 | 8.892e4 | 60.2 |
| B-other `ms ≥ 1800` | 3,929 | 9.268e5 | 18.6 |
| B-LIRF `ms ≥ 1800` | 356 | 1.853e7 | 33.8 |

These are what zero error would be worth. They are not targets. [S] `S7`, `S7b`

### 6.2 Closing the bias, per identifiable cell

| cell | bias | scored rows | **s from closing bias** | verdict |
|---|---:|---:|---:|---|
| A-matched (all bands pooled and band by band) | +1 … +46 | 132,236 | **0.022** | nothing there |
| B-other (all bands) | −37 | 3,929 | **0.027** | nothing there |
| **B-LIRF, whole-year bias** | +881 | 356 | **1.416** | real |
| **B-LIRF, months 1+7 bias** | **+1,356** (SE 220, t = 6.15) | 356 | **3.35 naive / 3.26 SE-adjusted** | the finding |

`S7d`. Restricting the OOF to the two months the scored frame contains: LIRF unmatched `ms ≥ 1800`,
n = 383, RMSE 4,314, bias **+1,356**, SE 220, **t = 6.15**. [S]

**Leave-one-month-out transfer test** (estimate the shift on one scored month, apply it to the
other — no label from the month being corrected):

| shift estimated on | applied to | n | SSE before → after | change | scaled to 356 scored rows |
|---|---|---:|---|---:|---:|
| month 1 (+1,204 s) | month 7 | 328 | 5.857e9 → 5.241e9 | **−10.5 %** | **+3.42 s** |
| month 7 (+1,382 s) | month 1 | 55 | 1.273e9 → 1.195e9 | **−6.1 %** | **+2.58 s** |

The same test on non-LIRF unmatched rows moves 0.56 % and 0.09 % of that cell's SSE — noise. [S] `S7d`

### 6.3 What a realistic partial improvement is worth

* **Closing the LIRF-unmatched bias alone: 2.6–3.4 s** (the LOMO range), on 356 scored rows.
  Achievable mechanically — a constant offset on a fully visible cell.
* **Cutting 10 % of B-LIRF's SSE: 3.38 s.** The bias shift is most of that, so the two are not
  additive: closing the bias *is* roughly the realistic 10 %.
* **Cutting 10 % of B-other's SSE: 1.86 s** — but there is no bias to close there (0.027 s) and
  `lfpg-unmatched-lane.md:§5` already measured every bound-style rule on that lane at ≤ 0.02 s. This
  would have to come from variance reduction with no instrument. **[X] Not a candidate.**
* **Cutting 10 % of A-matched's SSE: 6.02 s** across 132,236 rows — that is not a large-`ms`
  question at all, it is "make model A 5 % better", and the bias is already 5.7 s. **[X] Not a
  candidate from this analysis.**
* **The rollover lottery.** ~2 expected rollover rows in the scored frame at ~4e8 SSE each ≈ 4 s of
  board RMSE in expectation, concentrated in a coin flip. Unwinnable; see
  `lfpg-unmatched-lane.md:§6`. **[X]**

---

## 7. Verdict, and the honest caveats

**What the large-`ms` rows are:** overwhelmingly ordinary flights that pushed back late. 90.9 % of
the `ms ≥ 1800` population has `y < 1800`, and the median taxi time is the same (≈1,050 s) whether
`ms` is 35 minutes or 11 hours. The band is, for the most part, **exactly what the question
allowed for as an answer: variance, not a kind.** [S] §1.1

**The kinds that do exist are label-defined and none is individually identifiable.** §3.6 states
this per kind. Presenting `latepush` vs `longtaxi` vs `schedfill` as a modelling target would be
describing the residual, which is the failure mode this project has closed five candidates for.

**One thing survives as a candidate, and it is a cell, not a kind.** `ADEP == LIRF ∧ unmatched`
(356 scored rows, exactly recoverable from `ADEP_mvt` and `mvt_minus_aobt3` nullness) carries a
+1,356 s over-prediction in the two scored months, at t = 6.15, cross-validated across those months
at −6 % to −10 % of the cell's SSE, worth **2.6–3.4 s of board RMSE**. Its mechanism is understood:
the LIRF schedule-fill rate is seasonal (0.33 July, 0.76 December), `month` is excluded from the
feature set by design (`src/prc/train.py:244`), and the model consequently carries a year-average
mixture weight into two months that sit far below it.

**Three things that must be said against it. [X]**

1. **The transfer test spans 2025-Jan → 2025-Jul, not 2025 → 2026.** It establishes that the shift
   is not month-specific *within* 2025. It does not establish that LIRF's 2026 feed behaves the same
   way. If the fill rate in 2026 Jan/Jul resembles 2025 Nov/Dec instead, the same shift costs
   roughly what it would have gained.
2. **The effect is 383 OOF rows and four bands.** The 3-4h band, which carries 2.57 s of the
   3.26 s, is 47 OOF rows and 31 scored rows. The SE-adjustment is applied, but a t of 4.3 on 47
   rows is not the same evidence as a t of 4.3 on 47,000.
3. **It has not been ablated.** Everything above is measured on an OOF produced for another purpose.
   Until a run with the correction shows the number on the same validation design, this is a
   candidate, not a lever.

**What is closed by this analysis. [X]** The 1 h – 12 h band in the *matched* lane (132,236 scored
rows) is calibrated to within 46 s in every band and holds no structure that the NM clocks do not
already supply. The non-LIRF *unmatched* lane at large `ms` (3,929 scored rows) is unbiased at
−37 s and has no instrument of any kind — no clock, no operator, no aircraft type, and `ms` itself
at AUC 0.49–0.61 within band. Those two together are 136,165 of the 136,521 scored rows in the
population. There is no identifiable kind in them and the band is variance.

---

## 8. Facts vs inference

**Facts (measured above, each with its script section).** All counts, bands, kind definitions and
per-airport tables in §1–§2; the `|bs| ≤ 10` spike test and its result that only LIRF has one
(`S3b`); the complete absence of NM clocks and `_flt` categoricals in the unmatched lane (`S0`);
every AUC in §3 (`S3`, `S4`, `S4b`, `S4d`, `S8`); the OOF n / RMSE / bias / SSE share tables in §4
(`S5`, `S7b`, `S7c`); the per-month fill rate and per-month OOF bias (`S8`, `S7c`); the scored-frame
counts in §5 (`S6`, `S8c`); the leave-one-month-out shift test (`S7d`); the folds-sidecar
duplication (`S5`).

**Inference (not established by the data here).** That the LIRF-unmatched over-prediction will be
present in the 2026 Jan/Jul data at the 2025 magnitude (the mechanism is measured, the transfer to
2026 is assumed); that the callsign prefix would add to the model over `ADES_mvt` (an AUC margin of
0.08 / 0.01 out-of-time is measured; the incremental value inside a fitted model is not); that the
traffic-context columns' sub-0.5 AUCs reflect delayed departures landing in quiet traffic rather
than something else; that ~2 rollover rows sit in the scored frame; that the model's advantage over
the cell-mean oracle (`S8b`) comes from `ADES_mvt`/`STAND_mvt`/`RUNWAY_mvt` specifically.
