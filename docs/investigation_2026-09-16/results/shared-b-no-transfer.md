# Item 31 `shared-b-no-transfer` — no-training pre-check (2026-09-16)

Script: `docs/investigation_2026-09-16/scripts/shared-b-no-transfer.py` (no training, `POLARS_MAX_THREADS=1`,
lazy column-selected scans, ≈10 s wall). Inputs: `data/work/feat_train.parquet` (2,085,047 rows); `feat_rank.parquet`
only for prediction-time column availability on the 5,290 unmatched ranking rows. All numbers below were re-derived
by this script; none are copied from the hypothesis reports. Code facts come from `src/prc/train.py`,
`src/prc/features.py`, `configs/main.yaml` and the removed `shared_b` implementation (git 5b90919, removed in ea04ec9).

Definitions (plan §0): "unmatched" = routing definition (`mvt_minus_aobt3` null → model B), 22,470 rows; D's
`FLIGHT_ID_mvt`-null definition (22,424) gives identical tail counts (y ≥ 2 h 481 / y ≥ 4 h 119 under both; the 46
routing-only rows have y < 2 h). `y` = `TAXITIME_SEC_mvt`; `ms` = `mvt_minus_sched` = MVT − SCHED; `f10` =
|BLOCK − SCHED| ≤ 10 s; `f120` = ≤ 120 s. 2 h = 7,200 s, 4 h = 14,400 s, 12 h = 43,200 s.

## 1. Hypothesis — both sides as stated

**Side A (item 30; E H7, H §8, docs/README.md Phase 2' row a, commit 5b90919):** training model B on all ≈2.08 M rows
with the schedule-only feature set (`shared_b`) improves unmatched predictions, because the discarded 2,062,577 matched
rows "carry the same airport/hour/aircraft structure" (5b90919) and because "the extreme examples go from 33 to
≈2,000" (docs/README.md §Phase 2'). Pre-check proposed by E: per-airport y distribution unmatched vs matched.

**Side B (item 31; M `shared-b-no-transfer`):** pooling will not help the tail because the LIRF actype-null
sched-filled regime does not exist among matched rows: matched |BLOCK − SCHED| ≤ 120 s is 11–26 % per airport, but
matched y ≥ 2 h is 103 rows of 2.06 M; of 122 training rows y ≥ 4 h, 119 are unmatched; actype-null among matched
LIRF rows is 0.

This report executes the pre-check only. The effect of the run itself is item 30.

## 2. Procedure actually executed

A. Sanity: row counts under both unmatched definitions; label identity `y − ms = SCHED − BLOCK` (max |y − ms + (BLOCK − SCHED)| = 0 on all rows); `nm_matched` vs the routing flag.
B. Per ADEP × {matched, unmatched}: n, y median / mean / p90 / p99 / max, share y ≥ 2 h, share y ≥ 4 h, counts y ≥ 1 h / 2 h / 4 h / 12 h; pooled totals; FLIGHT_ID definition alongside.
C. Fill rate f10 and f120 per ADEP × group; f10 rows' y median and y ≥ 2 h share; count f10 ∧ y ≥ 2 h.
D. `AIRCRAFT_TYPE_mvt` null rate per ADEP × group; labels of matched actype-null rows.
E. Rows carrying the visible signature of the LIRF unmatched extreme rows (LIRF ∧ actype null ∧ ms ≥ 2 h) among matched rows, plus relaxations (LIRF ∧ ms ≥ 2 h with actype present; split by f10; non-LIRF ∧ ms ≥ 2 h), with label distributions; BLOCK − AOBT_3 on the matched LIRF ∧ ms ≥ 2 h ∧ f10 rows.
F. P(y ≥ 2 h | ms band), y median, f10 and median(y/ms) by LIRF / non-LIRF × matched / unmatched × 7 ms bands.
G. `cols_b` reconstructed by the exact rule in `train.py` (numeric columns minus LABEL / `month`, minus the nine `CANDIDATE_COLS` which `configs/main.yaml` does not include, minus the six prefixes `mvt_minus_aobt3`, `aobt3`, `lobt_minus`, `mvt_minus_lobt`, `mvt_minus_eobt1`, `mvt_minus_iobt`), plus the 10 `CATEGORICALS`: non-null rate on matched / unmatched training rows and on unmatched ranking rows; number of distinct values per group. G2: min/max per group of the eight flag columns in `cols_b`.
H. Extra "extreme" examples pooling would add per airport (matched counts at y ≥ 1 h / 2 h / 4 h / 12 h next to unmatched counts); pooled-set size at seven label thresholds; profile of the 103 matched y ≥ 2 h rows vs the 481 unmatched ones; listing of the 3 matched y ≥ 4 h rows; unmatched share of pooled rows and of pooled tail rows.

## 3. Results

### A. Sanity

2,085,047 rows; unmatched (routing) 22,470; FLIGHT_ID-null 22,424; routing-only 46, fid-only 0. `nm_matched` equals
¬routing on 2,085,001 rows (the 46 routing-only rows have `nm_matched` = 1). y and ms have no nulls;
`y − ms + (BLOCK − SCHED)` = 0 exactly on every row.

### B. Label distribution per airport, matched vs unmatched

| ADEP | group | n | y med | y mean | p90 | p99 | max | share ≥ 2 h | share ≥ 4 h | n ≥ 1 h | n ≥ 2 h | n ≥ 4 h | n ≥ 12 h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EDDF | matched | 228,185 | 836 | 863 | 1,249 | 1,816 | 6,566 | 0 | 0 | 70 | 0 | 0 | 0 |
| EDDF | unmatched | 1,956 | 958 | 975 | 1,398 | 2,111 | 6,907 | 0 | 0 | 3 | 0 | 0 | 0 |
| EDDM | matched | 166,324 | 774 | 811 | 1,134 | 1,916 | 6,850 | 0 | 0 | 48 | 0 | 0 | 0 |
| EDDM | unmatched | 1,010 | 891 | 1,036 | 1,570 | 3,661 | 6,899 | 0 | 0 | 11 | 0 | 0 | 0 |
| EGLL | matched | 238,132 | 1,319 | 1,363 | 1,806 | 2,697 | 11,452 | 0.0001 | 0 | 685 | 26 | 0 | 0 |
| EGLL | unmatched | 1,414 | 1,384 | 1,568 | 2,028 | 6,303 | 11,167 | 0.0078 | 0 | 45 | 11 | 0 | 0 |
| EHAM | matched | 243,867 | 744 | 788 | 1,158 | 1,767 | 10,972 | 0.0000 | 0 | 71 | 2 | 0 | 0 |
| EHAM | unmatched | 4,084 | 539 | 600 | 1,133 | 2,263 | 11,032 | 0.0002 | 0 | 11 | 1 | 0 | 0 |
| LEBL | matched | 177,843 | 905 | 956 | 1,355 | 1,957 | 3,604 | 0 | 0 | 1 | 0 | 0 | 0 |
| LEBL | unmatched | 1,862 | 1,039 | 1,043 | 1,484 | 2,261 | 3,273 | 0 | 0 | 0 | 0 | 0 | 0 |
| LEMD | matched | 211,295 | 984 | 1,014 | 1,402 | 1,932 | 3,598 | 0 | 0 | 0 | 0 | 0 | 0 |
| LEMD | unmatched | 947 | 1,096 | 1,116 | 1,547 | 2,164 | 3,465 | 0 | 0 | 0 | 0 | 0 | 0 |
| LFPG | matched | 235,779 | 954 | 1,017 | 1,443 | 2,402 | 21,308 | 0.0001 | 0.0000 | 351 | 30 | 2 | 0 |
| LFPG | unmatched | 3,773 | 1,023 | 1,184 | 1,689 | 3,783 | 84,240 | 0.0019 | 0.0005 | 46 | 7 | 2 | 2 |
| **LIRF** | **matched** | 159,216 | 1,024 | 1,145 | 1,682 | 3,299 | 87,002 | 0.0003 | 0.0000 | 1,224 | 41 | 1 | 1 |
| **LIRF** | **unmatched** | 1,488 | **3,922** | **6,531** | **11,819** | **59,641** | 131,167 | **0.2950** | **0.0773** | 745 | 439 | 115 | 30 |
| LSZH | matched | 132,616 | 712 | 742 | 1,080 | 1,705 | 9,538 | 0.0000 | 0 | 21 | 2 | 0 | 0 |
| LSZH | unmatched | 2,291 | 673 | 680 | 1,088 | 1,842 | 87,341 | 0.0017 | 0.0009 | 5 | 4 | 2 | 1 |
| LTFM | matched | 269,320 | 963 | 1,050 | 1,501 | 2,568 | 7,500 | 0.0000 | 0 | 574 | 2 | 0 | 0 |
| LTFM | unmatched | 3,645 | 1,009 | 1,302 | 1,802 | 6,473 | 10,309 | 0.0052 | 0 | 221 | 19 | 0 | 0 |
| all | matched | 2,062,577 | 911 | 987 | 1,464 | 2,303 | 87,002 | 0.0000 | 0.0000 | 3,045 | 103 | 3 | 1 |
| all | unmatched | 22,470 | 951 | 1,384 | 1,731 | 10,211 | 131,167 | 0.0214 | 0.0053 | 1,087 | 481 | 119 | 33 |

At nine of ten airports the unmatched median is within −205.5 (EHAM) … +133.5 (LEBL) s of the matched median and the
unmatched p99 exceeds the matched p99 by +137 (LSZH) … +3,905 (LTFM) s. LIRF is the exception on every statistic: unmatched median 3.8× matched, p90 7.0×, p99 18×.

### C. Fill rate matched vs unmatched by airport

| ADEP | f10 matched | f10 unmatched | f120 matched | f120 unmatched | n f10 ∧ y ≥ 2 h matched | n f10 ∧ y ≥ 2 h unmatched |
|---|---|---|---|---|---|---|
| EDDF | 0.0124 | 0.0015 | 0.1366 | 0.0118 | 0 | 0 |
| EDDM | 0.0501 | 0.0000 | 0.1980 | 0.0050 | 0 | 0 |
| EGLL | 0.0490 | 0.0028 | 0.1831 | 0.0050 | 1 | 1 |
| EHAM | 0.0123 | 0.0274 | 0.1352 | 0.0619 | 0 | 0 |
| LEBL | 0.0565 | 0.0188 | 0.1788 | 0.0236 | 0 | 0 |
| LEMD | 0.0448 | 0.0021 | 0.1603 | 0.0042 | 0 | 0 |
| LFPG | 0.0419 | 0.0034 | 0.1563 | 0.0143 | 0 | 0 |
| **LIRF** | **0.1782** | **0.4845** | 0.2567 | 0.4859 | **32** | **396** |
| LSZH | 0.0112 | 0.0227 | 0.1136 | 0.0781 | 0 | 0 |
| LTFM | 0.0515 | 0.0499 | 0.2022 | 0.1789 | 0 | 0 |

Matched f120 ranges 0.114 (LSZH) – 0.257 (LIRF). On matched f10 rows the y median is in the normal range at every
airport (725 – 1,318 s) and the y ≥ 2 h share is ≤ 0.0011 (LIRF: 32 of 28,369 filled matched rows). On LIRF unmatched
f10 rows: y median 7,559, y ≥ 2 h share 0.549.

### D. `AIRCRAFT_TYPE_mvt` null rate

Matched: 0 null rows at every airport (0 of 2,062,577). Unmatched: LIRF 1,487 / 1,488 (0.9993); LSZH 7, LTFM 7,
LFPG 5, EHAM 1; EDDF / EDDM / EGLL / LEBL / LEMD 0. Total 1,507 of 22,470. The table "matched actype-null rows: labels"
is empty.

### E. The LIRF unmatched extreme signature among matched rows

| subset | n | actype null | f10 | y p10 | y med | y p90 | y max | ms med | share ≥ 2 h | share ≥ 4 h | n ≥ 2 h | n ≥ 4 h | n ≥ 12 h | share \|y − ms\| ≤ 60 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| unmatched: LIRF ∧ actype null ∧ ms ≥ 2 h (the regime) | 627 | 1.000 | 0.632 | 898 | 8,518 | 21,292 | 131,167 | 9,962 | 0.659 | 0.180 | 413 | 113 | 30 | 0.632 |
| **matched: LIRF ∧ actype null ∧ ms ≥ 2 h (same signature)** | **0** | – | – | – | – | – | – | – | – | – | 0 | 0 | 0 | – |
| matched: LIRF ∧ ms ≥ 2 h (actype present) | 2,751 | 0 | 0.012 | 726 | 1,134 | 1,983 | 87,002 | 9,242 | 0.0145 | 0.0004 | 40 | 1 | 1 | 0.012 |
| matched: LIRF ∧ ms ≥ 2 h ∧ f10 | 32 | 0 | 1.000 | 7,386 | 8,099 | 11,629 | 87,002 | 8,099 | 1.000 | 0.031 | 32 | 1 | 1 | 1.000 |
| matched: LIRF ∧ ms ≥ 2 h ∧ ¬f10 | 2,719 | 0 | 0 | 725 | 1,131 | 1,919 | 13,133 | 9,246 | 0.0029 | 0 | 8 | 0 | 0 | 0 |
| matched: non-LIRF ∧ ms ≥ 2 h | 32,152 | 0 | 0 | 605 | 1,013 | 1,684 | 21,308 | 9,599 | 0.0019 | 0.0001 | 62 | 2 | 0 | 0.0001 |
| unmatched: non-LIRF ∧ ms ≥ 2 h | 4,322 | 0.0007 | 0.0002 | 613 | 1,076 | 1,931 | 11,167 | 8,920 | 0.0086 | 0 | 37 | 0 | 0 | 0.0002 |
| matched: LIRF all | 159,216 | 0 | 0.178 | 670 | 1,024 | 1,682 | 87,002 | 1,441 | 0.0003 | 0.0000 | 41 | 1 | 1 | 0.205 |
| unmatched: LIRF all | 1,488 | 0.9993 | 0.485 | 769 | 3,922 | 11,819 | 131,167 | 6,657 | 0.295 | 0.077 | 439 | 115 | 30 | 0.485 |

The 32 matched LIRF ∧ ms ≥ 2 h ∧ f10 rows: BLOCK − AOBT_3 quantiles p10 / med / p90 = −8,943 / −6,810 / −845 s;
28 of 32 have |BLOCK − AOBT_3| > 1 h (these are the item-5 "matched proxy outliers" at LIRF); all 32 have y ≥ 2 h;
1 has y ≥ 4 h (the 87,002 s row, a rollover with BLOCK − SCHED = −1 s).

### F. What the pooled rows say about the ms → y mapping

| airport | group | ms band | n | y med | y p90 | P(y ≥ 2 h) | n ≥ 2 h | f10 | med(y/ms) |
|---|---|---|---|---|---|---|---|---|---|
| LIRF | matched | 0–1 h | 142,057 | 1,020 | 1,627 | 0.0000 | 0 | 0.193 | 0.90 |
| LIRF | unmatched | 0–1 h | 97 | 2,229 | 8,103 | 0.247 | 24 | 0.474 | 1.01 |
| LIRF | matched | 1–2 h | 13,228 | 1,150 | 2,769 | 0.0001 | 1 | 0.069 | 0.26 |
| LIRF | unmatched | 1–2 h | 748 | 1,316 | 6,592 | 0.0027 | 2 | 0.373 | 0.23 |
| LIRF | matched | 2–4 h | 2,368 | 1,135 | 1,983 | **0.016** | 38 | 0.013 | **0.12** |
| LIRF | unmatched | 2–4 h | 500 | 7,737 | 11,226 | **0.602** | 301 | 0.590 | **1.00** |
| LIRF | matched | 4–12 h | 381 | 1,086 | 1,915 | **0.003** | 1 | 0.000 | **0.06** |
| LIRF | unmatched | 4–12 h | 98 | 18,451 | 38,886 | **0.847** | 83 | 0.847 | **1.00** |
| LIRF | matched | 12–20 h | 1 | 715 | 715 | 0 | 0 | 0 | 0.02 |
| LIRF | unmatched | 12–20 h | 24 | 54,209 | 87,543 | 1.000 | 24 | 0.708 | 1.00 |
| LIRF | matched | ≥ 20 h | 1 | 87,002 | 87,002 | 1.000 | 1 | 1.000 | 1.00 |
| LIRF | unmatched | ≥ 20 h | 5 | 87,605 | 131,167 | 1.000 | 5 | 0.200 | 1.02 |
| non-LIRF | matched | 2–4 h | 26,259 | 1,015 | 1,692 | 0.0022 | 57 | 0.000 | 0.11 |
| non-LIRF | unmatched | 2–4 h | 3,907 | 1,080 | 1,972 | 0.0082 | 32 | 0.0003 | 0.12 |
| non-LIRF | matched | 4–12 h | 5,344 | 987 | 1,606 | 0.0009 | 5 | 0.000 | 0.05 |
| non-LIRF | unmatched | 4–12 h | 369 | 1,019 | 1,866 | 0.0136 | 5 | 0.000 | 0.05 |
| non-LIRF | matched | 12–20 h / ≥ 20 h | 334 / 215 | 1,041 / 1,092 | 1,627 / 1,638 | 0 / 0 | 0 | 0 | 0.019 / 0.012 |
| non-LIRF | unmatched | 12–20 h / ≥ 20 h | 25 / 21 | 1,142 / 1,093 | 1,704 / 1,744 | 0 / 0 | 0 | 0 | 0.019 / 0.013 |

(Bands < 0 and 0–1 h for non-LIRF, and the full 28-row table, are in the script output.) At non-LIRF airports the
matched and unmatched rows agree band by band (P(y ≥ 2 h) ≤ 0.014, med(y/ms) 0.05–0.12 for ms ≥ 2 h). At LIRF the
matched rows with ms ≥ 2 h have P(y ≥ 2 h) = 0.016 / 0.003 and med(y/ms) = 0.12 / 0.06, where the unmatched rows have
0.60 / 0.85 and 1.00 / 1.00. The LIRF matched 2–4 h band alone (2,368 rows) is 4.7× the size of the LIRF unmatched
2–4 h band (500 rows) and carries the opposite label behaviour.

### G. Feature columns a pooled model B would have

Frame: 99 numeric feature columns; 89 visible under `configs/main.yaml` (no `exclude_features`, no
`include_candidates`, so the nine `CANDIDATE_COLS` are hidden); **`cols_b` = 74 columns**; 15 hidden from model B
by prefix (`mvt_minus_aobt3`, `mvt_minus_eobt1`, `mvt_minus_lobt`, `mvt_minus_iobt`, `lobt_minus_eobt1`,
`aobt3_minus_eobt1`, `aobt3_minus_sched`, `aobt3_null` and the seven `_slog` copies). `frozen_wx`,
`frozen_wx_prev1h` are Boolean and never features. The nine fold-derived encodings (`ref_*`, `arh_*`, `apt_med`,
`ref_p10_fb`, `ref_missing`) are joined at fit time on (ADEP, STAND, RUNWAY) / (ADEP, RUNWAY, hour_local) / ADEP,
all of which are ≥ 99.95 % non-null on unmatched rows.

Non-null rate on matched / unmatched training rows / unmatched ranking rows, and distinct values on unmatched
training rows:

| class | n cols | columns |
|---|---|---|
| `cols_b` numeric, ≥ 99 % non-null on both groups and varying on unmatched rows | 66 | `mvt_minus_sched`, `mvt_minus_sched_slog`, `dow`, `hour_utc`, `hour_local`, `minute_of_day_utc`, `nm_matched`, `stand_null`, 25 `dep_tko_*` / `arr_land_*` / `arr_inblock_*` window counts, `rwy_headway_prev_s` / `_next_s` (+ `_slog`), `rwy_tko_prev_15m` / `_60m`, `window_coverage_60m`, `nbr_aobt_delay_mean_30m` / `_60m`, `nbr_mvt_minus_aobt_mean_30m` / `_60m`, `nbr_dep_n_30m` / `_60m`, `arr_taxi_in_mean_30m` / `_60m`, `arr_taxi_in_n_30m` / `_60m`, 16 weather columns (`temp_c`, `dewp_spread_c`, `wind_ms`, `precip_mm`, `vis_km`, `deicing_gate`, `low_vis`, `temp_x_precip` and their `_prev1h`; non-null 0.9997–1.0) |
| `cols_b` numeric, ≥ 99 % non-null on both groups but a single value on unmatched rows | 6 | `actype_mismatch` (0 on all unmatched; 0/1 on matched), `diverted` (0 on all unmatched; 0/1 on matched), `anchor_ok` (1 everywhere), `runway_null` (0 everywhere), `eobt1_null` (**0 on all matched, 1 on all unmatched**), `lobt_null` (**0 on all matched, 1 on all unmatched**) |
| `cols_b` numeric, mostly null on both groups | 2 | `gust_ms` (0.023 / 0.027 / 0.028), `gust_ms_prev1h` |
| categorical, non-null on both groups and varying on unmatched rows | 5 | `ADEP_mvt`, `RUNWAY_mvt`, `STAND_mvt` (0.9995), `ADES_mvt` (0.9981), `ADES_REGION` (0.9981) |
| categorical, null on 100 % of unmatched rows (training and ranking) | 4 | `AIRCRAFT_OPERATOR_flt`, `MARKET_SEGMENT_flt`, `FLIGHT_TYPE_flt`, `WK_TBL_CAT_flt` (each becomes the single value `__NA__` on unmatched rows via `_matrix`) |
| categorical, partly null on unmatched rows | 1 | `AIRCRAFT_TYPE_mvt`: matched 1.000, unmatched 0.933 (training) / 0.927 (ranking); the null rows are the LIRF unmatched rows (D) |

G2 (regime indicators inside `cols_b`): `nm_matched` is 1 on all 2,062,577 matched rows and 0 on 22,424 of the
22,470 unmatched rows (1 on the 46 routing-only rows); `eobt1_null` and `lobt_null` are 0 on every matched row and 1
on every unmatched row. Three numeric columns given to model B therefore separate the two regimes exactly with one
split each; the four `_flt` categoricals do the same via `__NA__`. `train.py::_fit` passes no sample weights (the
word "weight" does not occur in the file); the removed `shared_b` mode (5b90919) fitted `u_tr = tr` (all rows) with
`cols_b` unchanged and used the main-model capacity (`cfg`, not `cfg.for_unmatched()`).

### H. Extra "extreme" training examples pooling would add to model B

| ADEP | n unmatched | n matched | y ≥ 1 h unm | y ≥ 1 h matched | y ≥ 2 h unm | **y ≥ 2 h matched (added)** | matched share of pooled ≥ 2 h | y ≥ 4 h unm | **y ≥ 4 h matched (added)** | y ≥ 12 h unm | y ≥ 12 h matched |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EDDF | 1,956 | 228,185 | 3 | 70 | 0 | 0 | – | 0 | 0 | 0 | 0 |
| EDDM | 1,010 | 166,324 | 11 | 48 | 0 | 0 | – | 0 | 0 | 0 | 0 |
| EGLL | 1,414 | 238,132 | 45 | 685 | 11 | 26 | 0.70 | 0 | 0 | 0 | 0 |
| EHAM | 4,084 | 243,867 | 11 | 71 | 1 | 2 | 0.67 | 0 | 0 | 0 | 0 |
| LEBL | 1,862 | 177,843 | 0 | 1 | 0 | 0 | – | 0 | 0 | 0 | 0 |
| LEMD | 947 | 211,295 | 0 | 0 | 0 | 0 | – | 0 | 0 | 0 | 0 |
| LFPG | 3,773 | 235,779 | 46 | 351 | 7 | 30 | 0.81 | 2 | 2 | 2 | 0 |
| **LIRF** | 1,488 | 159,216 | 745 | 1,224 | **439** | **41** | **0.085** | **115** | **1** | 30 | 1 |
| LSZH | 2,291 | 132,616 | 5 | 21 | 4 | 2 | 0.33 | 2 | 0 | 1 | 0 |
| LTFM | 3,645 | 269,320 | 221 | 574 | 19 | 2 | 0.10 | 0 | 0 | 0 | 0 |
| **total** | 22,470 | 2,062,577 | 1,087 | 3,045 | **481** | **103** | 0.176 | **119** | **3** | 33 | 1 |

Pooled-set size by label threshold (all rows = unmatched + matched): y ≥ 1,800 s: 74,626 = 2,023 + 72,603;
≥ 2,700: 11,736 = 1,222 + 10,514; ≥ 3,600: 4,132 = 1,087 + 3,045; ≥ 5,400: 1,235 = 777 + 458; ≥ 7,200: 584 = 481 + 103;
≥ 14,400: 122 = 119 + 3; ≥ 43,200: 34 = 33 + 1. Unmatched rows are 1.08 % of pooled rows, 82.4 % of pooled y ≥ 2 h
rows, 97.5 % of pooled y ≥ 4 h rows, 97.1 % of pooled y ≥ 12 h rows.

Profile of the 103 matched y ≥ 2 h rows (added examples) vs the 481 unmatched ones (target regime):

| ADEP | matched n | actype null | f10 | ms ≥ 2 h | \|y − ms\| ≤ 60 | \|BLOCK − AOBT_3\| > 1 h | y med | ms med | ≥ 4 h | | unmatched n | actype null | f10 | \|y − ms\| ≤ 60 | y med | ≥ 4 h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LIRF | 41 | 0 | 0.78 | 0.98 | 0.78 | 0.88 | 8,103 | 8,283 | 1 | | 439 | 439 | 0.90 | 0.90 | 10,196 | 115 |
| LFPG | 30 | 0 | 0.00 | 1.00 | 0.03 | 1.00 | 8,262 | 9,511 | 2 | | 7 | 0 | 0 | 0 | 9,115 | 2 |
| EGLL | 26 | 0 | 0.04 | 1.00 | 0.08 | 1.00 | 8,734 | 9,933 | 0 | | 11 | 0 | 0.09 | 0.09 | 8,464 | 0 |
| LTFM / LSZH / EHAM | 2 / 2 / 2 | 0 | 0 | 1.00 | 0 | 1 / 1 / 0 | 7,409 / 9,312 / 9,861 | | 0 | | 19 / 4 / 1 | 0 | 0 | 0 | 7,733 / 14,031 / 11,032 | 0 / 2 / 0 |

The 3 matched y ≥ 4 h rows: LIRF 192622644 (Jul, B738, rwy 25, y 87,002, ms 87,001, BLOCK − SCHED −1 s,
BLOCK − AOBT_3 +179 s; a filled day-rollover row); LFPG 184881387 (Mar, B734, y 21,308, ms 21,072, BLOCK − AOBT_3
−20,657); LFPG 191767771 (Jul, B764, y 15,059, ms 16,858, BLOCK − AOBT_3 −14,401). None is actype-null.

## 4. Verdicts per sub-claim

| # | side | sub-claim | verdict | deciding observation |
|---|---|---|---|---|
| A1 | A | The matched rows "carry the same airport/hour/aircraft structure" as the unmatched rows (5b90919) | CONFIRMED for airport / hour / runway / stand / traffic-window / weather columns; REFUTED for aircraft and operator columns | G: 66 `cols_b` numerics and `ADEP`, `RUNWAY`, `STAND`, `ADES`, `ADES_REGION` are ≥ 99.8 % non-null and varying on both groups. `AIRCRAFT_OPERATOR_flt`, `MARKET_SEGMENT_flt`, `FLIGHT_TYPE_flt`, `WK_TBL_CAT_flt` are null on 100 % of unmatched rows (training and ranking); `AIRCRAFT_TYPE_mvt` is null on 1,487 / 1,488 LIRF unmatched rows and 0 matched rows (D). |
| A2 | A | "Extreme examples go from 33 to ≈ 2,000" (docs/README.md Phase 2' a) | REFUTED | H: the 33 are unmatched y ≥ 12 h; pooling adds 1 (→ 34). At y ≥ 4 h: 119 → 122; at y ≥ 2 h: 481 → 584. No label threshold maps 33 unmatched rows onto ≈ 2,000 pooled rows (y ≥ 1 h gives 1,087 → 4,132; y ≥ 45 min gives 1,222 → 11,736). |
| A3 | A | Per-airport y distribution unmatched vs matched (E's pre-check) | DONE — nine airports alike, LIRF unlike | B: at EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LSZH, LTFM the unmatched median is within −205.5 … +133.5 s of the matched median; at LIRF 3,922 vs 1,024, p90 11,819 vs 1,682, share y ≥ 2 h 0.295 vs 0.0003. |
| A4 | A | Pooling improves unmatched predictions | UNDETERMINED — needs the item-30 run | No training here. |
| B1 | B | Matched \|BLOCK − SCHED\| ≤ 120 s is 11–26 % per airport | CONFIRMED | C: f120 matched 0.114 (LSZH) – 0.257 (LIRF). |
| B2 | B | Matched y ≥ 2 h is 103 rows of 2.06 M | CONFIRMED | B/H: 103 of 2,062,577 (routing) and 103 of 2,062,623 (FLIGHT_ID). |
| B3 | B | Of 122 training rows y ≥ 4 h, 119 unmatched | CONFIRMED | B/H: 119 + 3 = 122 under both definitions. |
| B4 | B | actype-null among matched LIRF rows: 0 | CONFIRMED | D: 0 of 159,216 (0 of 2,062,577 at all airports). |
| B5 | B | "The LIRF actype-null sched-filled regime does not exist among matched rows" | CONFIRMED for the actype-null signature; QUALIFIED for the fill mechanism | E: matched rows with LIRF ∧ actype null ∧ ms ≥ 2 h: **0**. The label mechanism BLOCK = SCHED with ms ≥ 2 h does occur in **32** matched LIRF rows (all y ≥ 2 h, y = ms, 28 with \|BLOCK − AOBT_3\| > 1 h) against 396 in unmatched LIRF rows; those 32 are 1.2 % of the 2,751 matched LIRF ms ≥ 2 h rows, whereas the unmatched rate is 63.2 % of 627. |
| B6 | B | Pooling will not help the tail | UNDETERMINED — needs the item-30 run; pre-check facts stated in §6 | The pre-check can state what pooling adds (H) and what mapping the added rows encode (F), not the fitted model's behaviour. |

## 5. Not determinable locally

- The effect of the pooled fit on unmatched OOF / LB error (item 30's full run; the plan states the same run settles
  item 31). Whether LightGBM, given `nm_matched` / `eobt1_null` / `lobt_null` / `__NA__` categoricals (G2), separates
  the regimes early (in which case the 2.06 M matched rows influence the unmatched leaves only through shared
  splits above that point and through the early-stopping / capacity setting) or shares leaves across regimes
  (in which case the LIRF matched ms ≥ 2 h rows, 4.4× as numerous as the unmatched ones and with P(y ≥ 2 h) 0.016 vs
  0.60–0.85, pull the LIRF unmatched tail predictions down) is a property of the fit, not of the data.
- The capacity and early-stopping behaviour under `shared_b` (5b90919 used the main-model parameters and the same
  inner holdout month; the current `configs/main.yaml` values are 127 / 50 / 2,000 for model B and 2,000 rounds /
  early stopping 100 for the main model); this determines how many of the 34 pooled y ≥ 12 h rows can form a leaf.
- Whether the four `_flt` categoricals and `AIRCRAFT_TYPE_mvt`, which a pooled model would learn on matched rows,
  change the prediction on ranking unmatched rows where they are `__NA__` (100 % / 7.3 %).
- LB effect (ranking labels unknown).

## 6. Facts vs inference

**Facts (measured by the script or read from code):** every number in §3; the reconstructed `cols_b` of 74 columns;
`nm_matched`, `eobt1_null`, `lobt_null` inside `cols_b` and constant per regime; the four `_flt` categoricals null on
every unmatched row in training and ranking; no sample weights in `_fit`; the removed `shared_b` mode fitted all rows
with the same `cols_b` and the main-model capacity; the docs/README.md sentence "極端例が 33 個 → ~2,000 個"; the 5b90919
commit message; pooling adds 103 / 3 / 1 rows at y ≥ 2 h / 4 h / 12 h, of which 41 / 1 / 1 are LIRF; the 32 matched
LIRF filled ∧ ms ≥ 2 h rows and their BLOCK − AOBT_3 offsets.

**Inference (not measured):**
- I1 The "33 → ≈ 2,000" count in docs/README.md does not correspond to any label-threshold count in the training set;
  it may have referred to a different quantity (e.g. rows with large ms, or a different threshold). Which one is not
  recoverable from the sentence.
- I2 Both sources' framing is partly right on the data: the non-LIRF unmatched rows look like the matched rows
  (B, F) and could in principle borrow their structure; the LIRF unmatched tail has no matched counterpart with the
  same signature (E) and the matched LIRF rows with the same `ms` carry the opposite label behaviour (F). Whether the
  first effect outweighs the second in RMSE terms is the item-30 run.
- I3 Because three visible numeric columns and four categoricals separate the regimes exactly (G2), a pooled tree
  model is able to reproduce the "separate" fit as a special case; the pooled data therefore need not degrade the
  LIRF tail, but it also cannot supply LIRF-tail examples that do not exist (H: +41 rows at y ≥ 2 h, +1 at y ≥ 4 h).
  This is a statement about what the training set contains, not about what a fitted model does.
- I4 The 5b90919 phrase "same … aircraft structure" cannot apply to the LIRF unmatched rows, since `AIRCRAFT_TYPE_mvt`
  is null on 1,487 of 1,488 of them and every operator / segment / type column is null on all unmatched rows; aircraft
  structure learned on matched rows has no key to be looked up with on the rows model B predicts.
