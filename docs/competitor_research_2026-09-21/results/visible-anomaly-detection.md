# Can anomalous rows be told apart from information visible at prediction time? (2026-09-24)

**Script (read-only; no training run, no submission, nothing written outside `/tmp`):**
`docs/competitor_research_2026-09-21/scripts/visible-anomaly-detection.py`, run stage by stage as
`POLARS_MAX_THREADS=<1-3> .venv/bin/python docs/competitor_research_2026-09-21/scripts/visible-anomaly-detection.py <stage>`
with `<stage>` in `counts | marginal | cats | combo | widen | part2 | price | hardcase | lobt`.
Artefacts land in a scratch directory; nothing in `src/`, `configs/` or `outputs/` was touched.
Fits were kept small and, where the text says so, subsampled; a few one-off checks were run
inline from the shell and are marked **(inline)** with the query given in full.

**Inputs (read, never written).** `data/work/feat_train.parquet` (2,085,047 rows, all
`PHASE_mvt == "DEP"`, 137 columns), `data/work/feat_rank.parquet` (344,841 scored 2026 departure
rows, same 137 columns), `experiments/oof_main_reparam-bc60017878.parquet` (672,428 non-null
`oof`, months 1/7/11/12, pooled RMSE **281.8411**), `outputs/main.parquet` (the shipped v13
predictions, 344,841 rows), `data/raw/submitting.parquet` (344,841 rows — the scored set is
exactly the departure rows of `feat_rank`).

**Notation.** `y = TAXITIME_SEC_mvt`; `ms = mvt_minus_sched = MVT − SCHED`
(`src/prc/features.py:63`); lane = `nm_matched` (1 = model A, has the NM clocks; 0 = model B, all
four NM clocks null). `y = MVT − BLOCK` holds on all 2,085,047 training rows.

**What "visible" means here, and it is checked, not assumed.** [S] (inline)
`pl.scan_parquet('data/work/feat_rank.parquet').collect().null_count()` over all 137 columns:
exactly **two** columns are null on all 344,841 scored rows — `BLOCK_TIME_UTC_mvt` and
`TAXITIME_SEC_mvt`, i.e. `BLANKED_DEP_COLS` in `src/prc/schema.py:52`. Everything else is
present at the rates you would expect (`mvt_minus_aobt3` and the other NM offsets null on
5,290 / 344,841 = 1.534 %; `stand_arrival_gap` null on 18.27 %; `gust_ms` null on 96.6 %). So a
feature is *visible* iff it does not read `BLOCK_TIME_UTC_mvt` or `TAXITIME_SEC_mvt`. Every
candidate below is labelled accordingly; the standing trap in this project is `bs = BLOCK − SCHED`,
which equals `ms − y` identically and is therefore a re-expression of the label.

---

## 0. The answer

1. **Yes for the matched lane, and it is exact, not statistical.** `|y − mvt_minus_lobt| ≤ 3,606 s`
   holds on **all 2,062,623 matched training rows, 0 violations**, so `mvt_minus_lobt` — visible —
   pins `y` to a ±1 h window. `mvt_minus_lobt > 43,200` selects **1 row of 2,062,623**, and it is
   the single matched corrupted row (precision 1.000, recall 1.000, zero false positives).
   The 2026 frame contains **exactly one** such row, and the shipped v13 prediction for it is
   84,866 s — the production `lobt_bound` already acts on it. §2.1, §5.2. [S]
2. **Mostly yes for the unmatched lane too, and the claim that "visible information points the
   wrong way" is wrong as a general statement.** Of the 15 unmatched corrupted rows, **12 sit above
   the 95th percentile of `ms` in their own lane and 11 above the 99.8th**; `ms` alone scores
   **AUC 0.8896** over the 22,424-row lane and **0.9895** inside LIRF's 1,487. The two rows the
   earlier claim cited (LSZH `ms` 939, LFPG `ms` 1,740) are percentile 15.6 and 24.6 — they are
   the exceptions, quoted as if they were the rule. §2.2, §2.3. [S]
3. **But the detector buys nothing, because the model already has it and already acts on it.**
   `mvt_minus_sched` is a model feature, and `unmatched_sched_reparam` at LIRF is precisely the
   mechanism that converts it into large predictions. The shipped v13 file already predicts
   30,644–66,693 s on the eleven 2026 LIRF unmatched rows with `ms ≥ 43,200`. Replacing the
   prediction on the best-precision visible cell with 87,000 s measures **+16.2 s of OOF RMSE**
   (281.84 → 298.07); the best *additive* shift on that cell, fitted leave-one-month-out, is
   −1.5 s at one threshold, +1.2 s at another, and changes sign across folds. §5.3, §5.4. [S]
4. **The rows that actually cost us are the ones nothing visible reaches.** Two LFPG rows are
   **19.1 %** of the whole OOF squared error (one of them 13.0 % on its own). Both have a *small*
   `ms` (1,740 and 2,043 s, percentile 4.98 and 6.09 among 3,763 LFPG unmatched peers) and sit
   inside the bulk of every other visible feature; the two disagree in sign on every feature where
   either is extreme. With n = 2 there is nothing to fit and nothing to claim. §3. [S]
5. **Combination does not rescue it at this class size.** A leave-one-month-out LightGBM over the
   full visible feature set scores **below `ms` alone** on the unmatched lane. With 15 positives
   the permutation control is the point of the exercise, not a footnote. §4. [S]
6. **Part 2 (heteroscedasticity) is the part with a real, reusable answer**, and it is not a lever
   either: visible features predict *error magnitude* well, but almost all of that is the two
   things the model already conditions on — lane and airport. §6.
7. **The "well under one expected corrupt row in 2026" estimate is wrong.** Every stratification
   gives **2.6 to 7.0** expected, with Poisson 95 % lower bounds of 1.5 to 2.8. §7.1. [S]

---

## 1. The positive class, and why 16 is the number

`counts` stage. Query: counts on `feat_train.parquet` of `y > 72,000` / `y ≥ 43,200` / `≥ 14,400` /
`≥ 7,200`, split by `nm_matched == 0` and `ADEP_mvt == "LIRF"`.

| class | n | unmatched | LIRF |
|---|---:|---:|---:|
| `y > 72,000` (the day-rollover signature) | **16** | 15 | 14 |
| `y ≥ 43,200` | 34 | 33 | 31 |
| `y ≥ 14,400` | 122 | 119 | 116 |
| `y ≥ 7,200` | 584 | 481 | 480 |

[S]. Matches `docs/investigation_2026-09-16/results/block-day-rollover.md` §3.1 exactly.

**15 of the 16 are in the unmatched lane, so the lane is the analysis unit.** [S] (inline) A
projection of the 16 rows onto the clock columns shows `mvt_minus_aobt3`, `mvt_minus_lobt`,
`mvt_minus_eobt1`, `mvt_minus_iobt` all null on 15 of 16; only `AEZ146R` (LIRF, 2025-07-14) is
matched, with `mvt_minus_aobt3 = 87,181`. Comparing the 15 against the 2.08 M-row population as a
whole would therefore measure "is this row unmatched", which is already a model feature
(`nm_matched`) and already the routing key. Every per-feature number below is computed **inside the
lane**.

---

## 2. Feature by feature (task item 1.1)

### 2.1 The matched lane: an exact visible bound, already in production

`lobt` stage. Query: on `feat_train.parquet` restricted to `nm_matched == 1` (2,062,623 rows),
`(|y − mvt_minus_lobt| > 3606).sum()`, then counts and label ranges under `mvt_minus_lobt > thr`.

| statement | value |
|---|---|
| matched training rows | 2,062,623 |
| rows violating `\|y − mvt_minus_lobt\| ≤ 3,606` | **0** |
| matched rows with `y > 72,000` | 1 |
| `mvt_minus_lobt > 7,200` | 143 flagged, 1 corrupt, `y` 3,777…87,002 |
| `mvt_minus_lobt > 14,400` | 5 flagged, 1 corrupt, `y` 12,578…87,002 |
| `mvt_minus_lobt > 43,200` | **1 flagged, 1 corrupt** — precision 1.000, recall 1.000 |
| `mvt_minus_aobt3 > 43,200` | 1 flagged, 1 corrupt |

[S]. `mvt_minus_lobt` is **computable without the label**: `LOBT_flt` and `MVT_TIME_UTC_mvt` are
both populated on 98.47 % of 2026 scored rows. The bound is not a statistic — it is the identity
already derived in `src/prc/train.py:118-130` and exploited by `lobt_bound: true` in
`configs/main_reparam.yaml`. On the matched lane the answer to the brief's question is therefore
an unqualified **yes**, with zero false positives in 2.06 M rows, and the lever is already pulled.

### 2.2 The unmatched lane: `ms` separates, and it is the only thing that does

`marginal` stage. Query: for each of the 118 numeric visible columns, Mann-Whitney AUC of the
column over the 22,424-row unmatched lane with the 15 corrupted rows as positives, plus the
percentile position of each positive inside the 22,409 negatives.

| feature | AUC (R1: unmatched lane, n=22,424, 15 pos) | AUC (R2: LIRF unmatched, n=1,487, 13 pos) | null on positives |
|---|---:|---:|---:|
| **`mvt_minus_sched`** (`= ms`) | **0.8896** | **0.9895** | 0/15 |
| `mvt_minus_sched_slog` | 0.8896 | 0.9895 | 0/15 |
| `queue_ahead` | 0.8254 | 0.9219 | 0/15 |
| `stand_arrival_type_match` | 0.1409 | — (100 % null in R2) | 14/15 |
| `arr_land_prev_60m` | 0.2312 | 0.3309 | 0/15 |
| `arr_inblock_prev_30m` | 0.2378 | — | 0/15 |
| `arr_land_prev_30m` | 0.2439 | 0.3217 | 0/15 |
| all 19 NM-clock columns | undefined | undefined | **15/15 null** |

[S]. Reading:

* **`ms` is the separator, and `queue_ahead` is `ms` wearing a hat.** `queue_ahead` counts the
  departures that left between this flight's `SCHED` and its own take-off
  (`src/prc/features.py:374-399`), so it is a monotone-ish function of `ms` by construction. Its
  AUC is lower than `ms`'s in both populations; it adds nothing. Both are label-free. [X]
* **The "arrivals" block points the other way and it is not a detector.** `arr_land_prev_60m`,
  `arr_inblock_prev_30m` and friends run AUC 0.21-0.33, i.e. the corrupted rows are in *quiet*
  traffic. That is a consequence of when they happen (a flight 16-25 h behind schedule goes at
  whatever hour it is ready), not a signal about the record. The percentile spread is 0.05 to 79,
  so it separates nothing at the row level. [X]
* **Nullness is not a signature inside the lane.** `stand_arrival_type_match` is null on 14 of 15
  positives against a lane base rate of 25.9 % — lift 3.6, `P(≥14 of 15 | 0.259) ≈ 7e-8` taken alone,
  but that column is 100 % null at LIRF unmatched, so this is the LIRF indicator again, not
  information. Inside LIRF (R2) it is undefined. [X]
* **Every NM-derived column is 100 % null on the positives by construction** — that is what
  "unmatched" means. They cannot participate.

### 2.3 Where the 15 actually sit, row by row

`marginal` stage, per-row percentiles against the 22,409 negatives of the same lane. This is the
table the earlier claim should have produced.

| ADEP | FLIGHT | y | `ms` | **`ms` pct** | hour pct | `stand_arrival_gap` pct | `dep_tko_prev_60m` pct | `queue_ahead` pct | `rwy_headway_prev_s` pct | `dest_distance_km` pct |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LIRF | NOSOS739 | 131,167 | 131,163 | **100.00** | 75.3 | 30.8 | 7.4 | 99.93 | 71.9 | 21.7 |
| LIRF | AMX071 | 88,392 | 67,686 | **99.88** | 69.3 | 77.8 | 35.5 | 99.78 | 18.9 | 99.1 |
| LIRF | CHHHH438 | 88,132 | 84,236 | **99.92** | 13.2 | 45.4 | 63.3 | 99.90 | 3.0 | 97.1 |
| LIRF | NOSOS610 | 87,605 | 86,102 | **99.93** | 80.6 | 42.0 | 10.4 | 99.84 | 43.8 | 92.7 |
| LIRF | BAWW561D | 87,598 | 60,599 | **99.86** | 39.2 | null | 12.1 | 98.94 | 61.9 | 63.2 |
| LIRF | ITYTY630 | 87,543 | 50,581 | **99.81** | 97.1 | 52.7 | 1.9 | 99.83 | 96.0 | 94.7 |
| LIRF | ITYTY680 | 87,480 | 56,520 | **99.83** | 47.0 | null | 35.5 | 98.64 | 93.6 | 99.8 |
| LIRF | VOE6BB | 87,361 | 73,260 | **99.91** | 75.3 | 54.5 | 25.0 | 99.83 | 43.8 | 53.7 |
| LSZH | NJE389D | 87,341 | 939 | **15.64** | 54.9 | 84.2 | 6.3 | 6.44 | 29.6 | 28.0 |
| LIRF | SWUU1919 | 87,247 | 65,463 | **99.88** | 24.5 | null | 63.3 | 99.59 | 45.4 | 22.2 |
| LIRF | ITYTY680 | 87,186 | 58,563 | **99.84** | 39.2 | 72.2 | 52.2 | 99.61 | 17.0 | 99.8 |
| LIRF | EXSXS1GT | 87,177 | 60,118 | **99.86** | 1.8 | 18.9 | 0.6 | 99.40 | 36.4 | 72.2 |
| LIRF | RYRR90BN | 87,170 | 93,535 | **99.97** | 54.9 | 38.2 | 49.4 | 99.91 | 1.5 | 23.1 |
| LIRF | ITY332 | 87,168 | 11,694 | **95.47** | 0.5 | 74.6 | 0.1 | 7.95 | 96.3 | 53.8 |
| LFPG | EJU983W | 84,240 | 1,740 | **24.58** | 80.6 | null | 27.5 | 19.62 | 76.7 | 28.8 |

[S]. **12 of 15 are at or above the 95th percentile of `ms` and 11 above the 99.8th.** The two
rows offered as evidence that visible information "points the wrong way" are rows 9 and 15 of this
table. They do point the wrong way — and they are 2 of 15. `hour_utc`, `dow`, `month`,
`stand_arrival_gap`, `dep_tko_prev_60m`, `rwy_headway_prev_s` and `dest_distance_km` are all
scattered across their ranges with no pattern, which is the expected picture for 15 draws.

**Mechanism, and why `ms` is not circular.** [X] `ms` does not read `BLOCK_TIME_UTC_mvt`, so it is
not a re-expression of the label. The link is physical: the day-rollover rows are flights that went
off-block 15-25 h after their scheduled time (`docs/investigation_2026-09-16/results/block-day-rollover.md`
§3.2, "if BLOCK date −1" column: 49,438…92,765 s for 12 of 16), and `ms = MVT − SCHED` measures
exactly that delay from the uncorrupted side. The recording fault and the large `ms` share one
cause — an enormously delayed departure — rather than one being derived from the other.

### 2.4 Categoricals, identity, calendar

`cats` stage. Query: for every categorical, flag and calendar column, the level counts among the
15 positives against the lane base rate, reported as lift and as the precision you would get by
flagging the whole level.

| column | level | positives | lane base rate | lift | precision if you flag the level |
|---|---|---:|---:|---:|---:|
| `ADES_mvt` | SAEZ | 2 | 0.107 % | 124.6 | 0.083 |
| `flight_prefix3` | NOS | 2 | 0.143 % | 93.4 | 0.063 |
| `flight_prefix3` | ITY | 4 | 0.700 % | 38.1 | 0.025 |
| `ADEP_mvt` | **LIRF** | **13** | 6.63 % | 13.1 | 0.0087 |
| `AIRCRAFT_TYPE_mvt` | null | 13 | 6.72 % | 12.9 | 0.0086 |
| `RUNWAY_mvt` | 25 | 11 | 5.86 % | 12.5 | 0.0084 |
| `hour_utc` | 18 | 2 | 4.30 % | 3.1 | 0.0021 |
| `month` | 11 | 2 | 6.02 % | 2.2 | 0.0015 |
| `dow` | 3 | 4 | 13.86 % | 1.9 | 0.0013 |
| `stand_null`, `runway_null`, `diverted`, `actype_mismatch`, `anchor_ok` | — | 15/15 at the modal value | ≈100 % | 1.00 | base rate |
| `AIRCRAFT_OPERATOR_flt`, `MARKET_SEGMENT_flt`, `FLIGHT_TYPE_flt`, `WK_TBL_CAT_flt` | null | 15/15 | 100 % | 1.00 | base rate |

[S]. Reading: **nothing categorical is a detector.** `ADES_mvt == SAEZ` and the `NOS`/`ITY` flight
prefixes have big lifts on 2-4 rows and 0.025-0.083 precision — at n = 15, two rows landing in a
0.1 %-of-the-lane level is a coin-flip event you will find in *some* column if you look at
enough of them, and 12 columns × dozens of levels were looked at. `ADEP_mvt == LIRF`,
`AIRCRAFT_TYPE_mvt` null and `RUNWAY_mvt == 25` are the same fact three times (LIRF's unmatched
rows carry no aircraft type and mostly use 25) and are already model categoricals. The operator
block is **entirely null on the unmatched lane** — all four `_flt` categoricals are 100 % null
there — so operator identity contributes nothing by construction. Calendar columns are flat:
the largest month lift is 2.2 and the largest hour lift 3.1, both on 2 rows. Stand is not a
signature: no stand appears twice.

---

## 3. The rows that actually cost money, and they are the invisible ones

`hardcase` stage, plus an inline join of `oof_main_reparam-bc60017878.parquet` onto
`feat_train.parquet` on `MVT_ID_mvt`.

**Error concentration, re-measured on the current OOF.** [S] (inline) 672,428 rows, pooled
RMSE 281.8411, SSE 5.3414e10; worst row 12.96 %, worst 10 rows 26.91 %, worst 100 rows 34.94 %,
worst 1,000 rows 47.00 %. (Confirms the brief's figures.)

**The top of that list is not what the corruption story predicts.** [S]

| rank | ADEP | FLIGHT | y | OOF pred | `ms` | SSE share |
|---:|---|---|---:|---:|---:|---:|
| 1 | LFPG | EJU983W | 84,240 | 1,025 | 1,740 | **12.96 %** |
| 2 | LFPG | EJU42AY | 58,206 | 936 | 2,043 | **6.14 %** |
| 3 | LIRF | ITYTY680 | 87,480 | 57,514 | 56,520 | 1.68 % |
| 4 | LIRF | ITYTY680 | 87,186 | 57,570 | 58,563 | 1.64 % |
| 5 | LIRF | EXSXS1GT | 87,177 | 60,190 | 60,118 | 1.36 % |
| 6 | LIRF | BAWW561D | 87,598 | 61,277 | 60,599 | 1.30 % |
| 7 | LIRF | PGT2WE | **1,377** | **20,694** | 20,878 | 0.70 % |
| 8 | LIRF | VOE6BB | 87,361 | 69,477 | 73,260 | 0.60 % |
| 9 | LIRF | NBT40H | **842** | **12,936** | 15,361 | 0.27 % |
| 10 | LIRF | RYR68SF | **723** | **12,209** | 15,301 | 0.25 % |

All ten corrupted rows present in this OOF together carry **19.64 %** of its SSE, and **13.0 of
those 19.6 points are the single LFPG row**. The nine LIRF corrupted rows carry 6.68 %, because
the model already predicts 57,514 to 91,813 on them. Rows 7, 9 and 10 (and 9 more of the top 25)
are the *opposite* error: LIRF unmatched rows with a large `ms` and a perfectly ordinary label,
over-predicted by 10,000-19,000 s. **That is the ms detector already firing, and already paying
its false-positive bill.**

**Where the LIRF unmatched SSE sits, by `ms` band** (`hardcase`; OOF rows only):

| `ms` band | n | corrupted | SSE share of OOF | of which over-prediction | of which under-prediction |
|---|---:|---:|---:|---:|---:|
| < 1 h | 39 | 0 | 0.46 % | 0.09 % | 0.37 % |
| 1-2 h | 259 | 0 | 2.11 % | **1.62 %** | 0.49 % |
| 2-4 h | 184 | 0 | 7.06 % | **6.25 %** | 0.81 % |
| 4-12 h | 40 | 0 | 1.65 % | **1.43 %** | 0.23 % |
| ≥ 12 h | 15 | 8 | 6.72 % | 0.04 % | **6.68 %** |

[S]. The ≥ 12 h band is where the corruption lives and the model under-predicts; the 1-12 h bands
immediately below it are where the same signal has *already* been over-applied, at a combined cost
of 9.3 % of SSE. Any proposal to lean harder on `ms` has to explain why it will not also deepen
that 9.3 %.

**The two LFPG rows against their own peers.** `hardcase` stage. Query: percentile position of
each of the two LFPG rows with `y ≥ 43,200` inside the other 3,763 LFPG unmatched training rows,
over all 118 numeric visible columns, ranked by the larger of the two deviations from the median.

| feature | row A (EJU983W) value / pct | row B (EJU42AY) value / pct |
|---|---|---|
| `wind_ms_prev1h` | 1.03 / **1.1** | 7.20 / **88.4** |
| `temp_c` | 0.0 / 3.6 | 7.0 / 16.7 |
| `arr_land_prev_5m` | 5 / 95.3 | 1 / 16.5 |
| **`mvt_minus_sched`** | 1,740 / **4.98** | 2,043 / **6.09** |
| `queue_ahead` | 15 / 5.3 | 27 / 10.1 |
| `rwy_tko_prev_15m` | 4 / 32.3 | 9 / 95.0 |
| `minute_of_day_utc` | 110 / 94.4 | −115 / 5.6 |
| `stand_arrival_gap` | null | null |
| `dow` | 7 / 92.3 | 4 / 44.6 |

[S]. **No feature places both rows on the same side of the distribution except the clock features,
and those place them both LOW.** The largest deviation from the median anywhere in the 118 columns
is 48.9 percentile points, on `wind_ms_prev1h`, where the two rows disagree by 87 points. The best
single-feature AUC for `y ≥ 43,200` inside LFPG unmatched is 0.028 (`month`: both rows are
January) — with 2 positives in 3,765 rows, an AUC of 0.028 or 0.972 is what one draw looks like.
The only thing the two share besides the small `ms` is a null `stand_arrival_gap` (lane base rate
13.5 %, so `P ≈ 0.018` for both — one uncorrected coincidence out of 118 columns tested).

**Verdict for §3: for the two rows that carry a fifth of our squared error, nothing visible
separates them, and this is now demonstrated at each feature rather than asserted.** The reason it
cannot be settled better is n = 2; the honest statement is that the test has essentially no power,
not that the answer is no. [X]

---

## 4. Combination, and what chance looks like at n = 15 (task item 1.2)

`combo` stage. **Fitted on the unmatched lane only** — see §9.1 for why. 22,424 rows, 15
positives, 117 visible features (every numeric visible column except `month`, plus the 12
categoricals), LightGBM binary objective, `num_leaves 15`, `min_data_in_leaf 5`, 80 rounds,
`is_unbalance`, **leave-one-month-out over the 12 training months** so no fold sees a positive
from its own month. The permutation control repeats the entire pipeline 30 times with the labels
shuffled **within month**, which preserves the 15-positives-in-12-months structure.

| quantity | value |
|---|---|
| LOMO AUC, 117 visible features | **0.8609** |
| LOMO AUC, `mvt_minus_sched` alone (no fitting at all) | **0.8896** |
| permutation null (30 runs): mean / sd | 0.5334 / 0.0544 |
| permutation null: 5th / 50th / 95th pct | 0.4475 / 0.5348 / 0.6261 |
| permutation null: min / max | 0.4197 / **0.6726** |
| one-sided p for the fitted model | **0.032** (= 1/31, the floor at 30 permutations) |
| random-label null for `ms` alone (5,000 draws of 15 positives, fixed score) | mean 0.5026, 95th 0.6254, 99th 0.6787, max 0.7530 |
| one-sided p for `ms` alone | **2.0e-4** (= 1/5,001, the floor at 5,000 draws) |

Precision / recall at the top of each ranking:

| k | fitted model TP | precision | recall | `ms` alone TP | precision | recall |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 1 | 0.20 | 0.07 | 1 | 0.20 | 0.07 |
| 20 | 2 | 0.10 | 0.13 | 3 | 0.15 | 0.20 |
| **50** | 5 | 0.10 | 0.33 | **11** | **0.22** | **0.73** |
| 100 | 9 | 0.09 | 0.60 | 12 | 0.12 | 0.80 |
| 200 | 11 | 0.055 | 0.73 | 12 | 0.06 | 0.80 |

[S]. **What 15 positives does to these numbers, stated plainly.** (a) The permutation null has a
95th percentile of **0.626** and a maximum of **0.673**: at this class size, any reported AUC below
about 0.68 is indistinguishable from a lucky shuffle, so "AUC 0.65, promising" would be a
meaningless sentence. (b) The smallest p-value obtainable from 30 permutations is 1/31 = 0.032; the
measured 0.8609 is above every one of the 30, and that is all 30 permutations can say. The `ms`
null is cheap to resample because the score is fixed, so it gets 5,000 draws and a floor of 2e-4;
there the evidence is genuinely strong. (c) Every precision figure in the table is a ratio with a
numerator between 1 and 12, so its standard error is of the same order as itself. (d) **The fitted
model is worse than the single feature it contains.** With 15 positives spread over 12 folds, each
fold trains on 13-14 of them; LightGBM spends its capacity on the other 116 columns and loses
0.029 of AUC doing so. That is the expected outcome, and it is the reason this section does not
end in a model.

---

## 5. What a detector would be worth, and what it would cost (task item: PRICE BOTH)

### 5.1 The exchange rate

[S] (inline) `N = 344,841`, board RMSE `282.7371`, so board `SSE = N × RMSE² = 2.7567e10` and one
second of RMSE at the margin is `2 × N × RMSE = 1.950e8` of squared error. Single-row moves, with
the resulting board RMSE computed exactly (not linearised):

| action on **one** row | Δ SSE | board RMSE | Δ RMSE |
|---|---:|---:|---:|
| a corrupted row (`y ≈ 87,000`) currently predicted 1,000 → predicted correctly | −7.396e9 | 241.85 | **−40.88** |
| … currently predicted 5,000 → correct | −6.724e9 | 245.85 | −36.89 |
| … currently predicted 30,000 → correct | −3.249e9 | 265.55 | −17.18 |
| … currently predicted 60,000 → correct | −7.290e8 | 278.97 | −3.76 |
| … currently predicted 84,000 → correct | −9.000e6 | 282.69 | −0.05 |
| a **normal** row (`y = 900`) moved to 10,000 | +8.281e7 | 283.16 | +0.42 |
| … moved to 30,000 | +8.468e8 | 287.05 | +4.31 |
| … moved to 60,000 | +3.493e9 | 300.12 | +17.38 |
| … moved to 87,000 | +7.413e9 | 318.49 | **+35.76** |

[X] **The bet is close to symmetric at the extremes**: catching one corrupted row that is currently
predicted short is worth ≈ 41 s; calling one normal row corrupted costs ≈ 36 s. A rule that
predicts 87,000 on a flagged row therefore needs precision above roughly **0.47** merely to break
even, and that is the *best* case, where the corrupted rows are ones we currently predict at 1,000.
The corrupted rows we currently predict at 57,000-91,000 are worth 0.05-4 s each, so the realistic
break-even precision is far higher than 0.47.

### 5.2 Matched lane: priced, and already banked

The `mvt_minus_lobt` bound has precision 1.000 and recall 1.000 on the matched lane, so its
false-positive cost is zero by the identity. It is already applied (`lobt_bound: true`). [S] (inline)
The one 2026 matched row it binds on is LIRF `ITY340`, July, `mvt_minus_lobt = 86,820`,
`mvt_minus_aobt3 = 86,340`, shipped v13 prediction **84,866** — inside the guaranteed window
`[83,214, 90,426]`. Value of re-deriving it: **0 s**, because it is in production. Value of *not*
having it: if that row were predicted at the lane median (~900 s) instead, the board would be
about 41 s worse. [X]

### 5.3 Unmatched lane: the operating points, and what acting on them measures

`combo` / `price` stages. Query: `ms ≥ thr` on the 22,424-row unmatched training lane against
`y > 72,000`, and the same threshold counted on the 5,283 unmatched rows of `feat_rank`.

| `ms ≥` | train flagged | corrupted caught | precision | recall | 2026 rows flagged |
|---:|---:|---:|---:|---:|---:|
| 14,400 | 540 | 12 | 0.022 | 0.80 | 322 |
| 28,800 | 113 | 12 | 0.106 | 0.80 | 51 |
| 43,200 | 75 | 12 | 0.160 | 0.80 | 24 |
| **50,400** | 55 | 12 | **0.218** | **0.80** | 18 |
| 57,600 | 46 | 10 | 0.217 | 0.67 | 13 |
| 72,000 | 26 | 5 | 0.192 | 0.33 | 7 |
| 86,400 | 18 | 2 | 0.111 | 0.13 | 3 |
| **LIRF & `ms ≥ 43,200`** | **29** | **12** | **0.414** | 0.80 | **11** |

[S]. Best precision anywhere is **0.414** — below the 0.47 break-even of §5.1 even before the
discount for corrupted rows we already predict large. And the measured counterfactual confirms it:

[S] (inline) replacing the OOF prediction with 87,000 on the flagged cells and re-scoring the whole
672,428-row OOF:

| rule | flagged | corrupted among them | Δ SSE | OOF RMSE |
|---|---:|---:|---:|---:|
| baseline | — | — | — | 281.841 |
| LIRF ∧ unmatched ∧ `ms ≥ 43,200` → 87,000 | 15 | 8 | **+6.33e9** | **298.07** |
| LIRF ∧ unmatched ∧ `ms ≥ 43,200` → 60,000 | 15 | 8 | +3.49e9 | 290.90 |
| unmatched ∧ `ms ≥ 50,400` → 87,000 | 25 | 8 | +1.09e11 | 490.86 |
| **oracle**: set the 10 corrupted rows to their true labels | 10 | 10 | −1.05e10 | **252.65** |

**Acting on the best available visible detector makes the model 16.2 s worse.** The oracle is worth
29.2 s of OOF RMSE, and the gap between 29.2 and −16.2 is the whole answer: the information is
real, the model has already extracted what is extractable from it, and the remainder is not
addressable by a threshold rule.

### 5.4 The subtler action, tested honestly

Instead of forcing a value, shift the flagged cell's predictions by a constant `c`. Fitted on the
cell's own residuals this always "gains" — `ms ≥ 50,400` at LIRF gives `c* = +13,376 s` and an
apparent −4.76 s — which is exactly the six-candidates-died-at-review failure mode: the shift is
fitted on the residuals it is then scored against, so it is a description of the residual, not a
lever. [X] Fitted **leave-one-month-out** over the four OOF months and applied to the held-out
month: [S] (inline)

| cell | LOMO Δ RMSE | per-fold Δ SSE (months 1 / 7 / 11 / 12) | rows per fold |
|---|---:|---|---|
| LIRF unm `ms ≥ 14,400` | **+1.18** | −8.3e7 / **+6.4e8** / −1.4e8 / +2.9e7 | 6 / 39 / 4 / 6 |
| LIRF unm `ms ≥ 28,800` | −1.57 | −1.8e8 / −2.4e8 / −3.9e8 / **+2.2e8** | 4 / 12 / 4 / 4 |
| LIRF unm `ms ≥ 43,200` | −1.45 | −3.3e8 / **+2.2e8** / −6.5e8 / **+2.0e8** | 2 / 9 / 2 / 2 |
| LIRF unm `ms ≥ 50,400` | −3.55 | −3.8e8 / −2.9e8 / −9.0e8 / **+2.4e8** | 2 / 5 / 2 / 1 |
| LIRF unm, all | +0.34 | −6.9e7 / −5.9e7 / **+1.9e8** / +7.0e7 | 60 / 337 / 52 / 88 |
| non-LIRF unm `ms ≥ 14,400` | −0.000 | — | 32 / 76 / 20 / 36 |

[X] **Do not ship this.** The sign of the effect flips with the threshold (+1.18 at 4 h, −3.55 at
14 h), at least one fold is positive in every row of the table, and the folds carry 1-9 rows each.
The threshold that looks best is the one with the fewest rows. This is the shape of an artefact.

---

## 7. How many corrupted rows are actually in the 2026 scored frame?

### 7.1 The "well under one" estimate is wrong

`price` stage. Query: `k / n_train × n_rank` with an exact Poisson (chi-square) 95 % interval on
`k`, under five different stratifications.

| basis | train n | corrupted k | 2026 n | **expected** | Poisson 95 % |
|---|---:|---:|---:|---:|---|
| all rows, all 12 months | 2,085,047 | 16 | 344,841 | **2.65** | 1.51 – 4.30 |
| unmatched lane | 22,424 | 15 | 5,283 | **3.53** | 1.98 – 5.83 |
| LIRF unmatched only | 1,487 | 13 | 383 | **3.35** | 1.78 – 5.73 |
| training months 1 and 7 only | 344,419 | 7 | 344,841 | **7.01** | 2.82 – 14.44 |
| unmatched lane, months 1 and 7 | 5,366 | 6 | 5,283 | **5.91** | 2.17 – 12.86 |

[S]. **Every basis gives 2.6 to 7.0 expected corrupted rows, and every 95 % lower bound is above
1.5.** The months-1-and-7 basis is the most relevant one — the scored frame is exactly
January 2026 (152,719) plus July 2026 (192,122), and January and July 2025 together contain 7 of
the 16 corrupted rows in a near-identical row count — and it is the highest. Whoever put the number
"well under one" on the record should say what it was computed from; no stratification I can
construct reaches it. [X]

### 7.2 What that means for the board

[X] At 3 expected corrupted rows, and taking the LIRF-like case where the model already predicts
large (worth ~1-4 s each to fix) versus the LFPG-like case where it predicts short (worth ~40 s
each), the 2026 board score carries somewhere between **3 and 80 s of RMSE of pure record
corruption**, entirely determined by how many of the 2-7 rows are of the LFPG kind. Our current
282.7371 is, in this sense, mostly a draw from that lottery rather than a measurement of the
model. That is not actionable — it is a reason to distrust small differences on the leaderboard,
which is the same conclusion `docs/README`'s §6 decision rule reaches from the other end.

[S] (inline) Sanity check that the 2026 frame contains the *shape* of the training positives:
24 unmatched 2026 rows have `ms ≥ 43,200` (11 of them at LIRF), 7 have `ms ≥ 72,000`, 3 have
`ms ≥ 86,400`. The shipped v13 predictions on the 11 LIRF ones are 30,644 / 31,551 / 32,688 /
33,444 / 33,485 / 35,434 / 35,683 / 36,782 / 39,636 / 66,037 / 66,693 — i.e. **the model has
already committed to the corruption hypothesis on those eleven rows**. The 13 non-LIRF rows in the
same `ms` band (EHAM, EDDF, LFPG) get 2,465 to 8,849, correctly, because non-LIRF large-`ms` rows
in training have ordinary labels.

---

## 8. Label-freeness audit (required for every candidate in this file)

The rule applied: a quantity is label-free iff it can be computed from `feat_rank.parquet`, whose
only null-everywhere columns are `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt`. [S] (inline, the
`null_count()` query in the header).

| quantity used above | label-free on a 2026 scored row? | why |
|---|---|---|
| `mvt_minus_sched` (`ms`) | **Yes** | `MVT_TIME_UTC_mvt` − `SCHED_TIME_UTC_mvt`, both populated on 344,841/344,841 |
| `mvt_minus_lobt` | **Yes**, on 98.466 % of rows | `LOBT_flt` null on 5,290/344,841 |
| `mvt_minus_aobt3`, `mvt_minus_eobt1`, `mvt_minus_iobt` | **Yes**, same 98.466 % | NM clocks, null only where the row is unmatched |
| `queue_ahead` | **Yes** | counts other departures' `MVT_TIME_UTC_mvt` between this row's `SCHED` and its own take-off |
| `nm_matched`, `aobt3_null`, `lobt_null`, `eobt1_null` | **Yes** | derived from NM nullness |
| `ADEP_mvt`, `RUNWAY_mvt`, `STAND_mvt`, `AIRCRAFT_TYPE_mvt`, `FLIGHT_mvt` and its prefix, `ADES_*` | **Yes** | present on every scored row |
| `hour_utc`, `dow`, `month`, `minute_of_day_utc` | **Yes** | from `MVT_TIME_UTC_mvt` |
| the traffic-context block (`dep_tko_*`, `arr_land_*`, `arr_inblock_*`, `rwy_headway_*`, `nbr_*`, `arr_taxi_in_*`) | **Yes** | built from the masked frame (`src/prc/features.py:528`), arrivals keep their block times |
| `stand_arrival_gap`, `stand_aobt_turn`, `stand_arrival_type_match` | **Yes** | arrival-side `BLOCK_TIME_UTC_mvt` is present on all 344,693 arrival rows |
| the weather block | **Yes** | external, joined on airport and hour |
| the model's own OOF / ranking prediction | **Yes** | a function of visible features only |
| **`bs = BLOCK − SCHED`** | **NO** | `≡ ms − y`. Any "feature" built on it is the label. |
| **`y − ms`, `y − 86,400`, the residual, `|resid|`, the SSE rank** | **NO** | all read `y` |
| **a per-cell additive shift fitted on OOF residuals** | **borderline** | the *rule* is label-free, the *constant* is fitted on labels; that is legitimate only if fitted out-of-fold, which §5.4 does, and it then fails |

---

## 9. What I did not do, and what would change the answer

1. **The all-rows classifier was deliberately not fitted.** 15 of the 16 positives are unmatched,
   so a classifier over all 2.08 M rows measures `nm_matched` — already a model feature and the
   routing key — and would report a flattering AUC for no reason. Everything in §2 and §4 is
   computed inside the lane. [X]
2. **n = 15 (or 13 at LIRF, or 2 at LFPG) is the binding constraint, not the feature set.** No
   amount of feature engineering fixes a class of that size; the permutation control in §4 exists
   precisely so the reader can see how much apparent structure 15 random rows produce.
3. **What would change the answer.** (a) Organiser confirmation of the fault's mechanism — if
   BLOCK is written with the *scheduled* date whenever the flight crosses midnight relative to
   schedule, then `ms` plus a midnight-crossing indicator is a near-complete detector and the LFPG
   rows are a different fault. (b) More positives: pooling 2024 and 2025 training data, if it were
   available, would take the class from 16 to ~30-50 and make §4 worth re-running. (c) Anything
   that recovers a second off-block clock for the unmatched lane — that is the real gap, and it is
   the same gap `lfpg-unmatched-lane.md` §0.4 reached from the other side.
4. **Not attempted:** per-airport thresholds beyond LIRF (no other airport has enough positives),
   a mixture model on `y` (that is a label-side model, not a detector), and any change to `src/`,
   `configs/` or `outputs/`.

---

## 10. One more visible signal, for completeness: the calendar-day crossing

[S] (inline) `daydiff = date(MVT_TIME_UTC_mvt) − date(SCHED_TIME_UTC_mvt)`, computed on the
unmatched training lane. Label-free (neither field is blanked).

| `daydiff` | rows | corrupted | precision |
|---:|---:|---:|---:|
| −1 | 3 | 0 | 0.000 |
| 0 | 22,204 | 3 | 0.0001 |
| **+1** | **216** | **12** | **0.0556** |
| +2 | 1 | 0 | 0.000 |

AUC 0.8954 — marginally better than `ms` (0.8896) and, since 13 of the 16 corrupted rows have
`date(MVT) = date(SCHED) + 1` (`block-day-rollover.md` §3.3), the same signal seen through a
coarser lens. `daydiff ≥ 1 ∧ LIRF` gives precision 0.179 at recall 0.80 — worse than
`ms ≥ 43,200 ∧ LIRF` (0.414). The 2026 frame has **65** unmatched rows with `daydiff ≥ 1`, 25 of
them at LIRF. It adds nothing over `ms` and is priced the same way.
