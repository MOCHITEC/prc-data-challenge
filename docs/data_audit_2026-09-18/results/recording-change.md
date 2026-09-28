# Recording change 2025 → 2026 — is the INSTRUMENT different?

Script: `docs/data_audit_2026-09-18/scripts/recording-change.py` (no training, `POLARS_MAX_THREADS=1`,
lazy month-by-month aggregation over `data/raw/*.parquet`; ~25 min total across stages 0-8b).

**Lens.** Not "are the flights different" — that is `unmatched-composition-shift.md`,
`sched-anchored-delay-shift.md`, `seasonal-deicing-shift.md`, `eham-jan2026-disruption.md` and
findings.md §1.4, none of which is re-derived here. This report asks only whether the *recording
apparatus* changed: null rates, distinct counts, value ranges, timestamp granularity, the NM join
mechanism, categorical vocabularies, identifier encodings, and the writer that produced the files.

**Unit of analysis.** 14 month-partitions, not 24: the 12 training files are one calendar month each,
and `ranking.parquet` is split on `month(MVT_TIME_UTC_mvt)` into 2026-01 and 2026-07. Every partition
is further split by `PHASE_mvt` into **DEP** (344,841 ranking rows = the scored set) and **ARR**
(344,693 ranking rows, which keep `TAXITIME_SEC_mvt` and `BLOCK_TIME_UTC_mvt` in 2026 and are
therefore the only direct window onto 2026 recording). 28 partition × phase cells in all.
For DEP the audited airport is `ADEP_mvt`; for ARR it is `ADES_mvt`.

Row counts per partition (DEP / ARR): 2025-01 153,706/153,551 … 2025-12 165,677/165,871,
2026-01 152,719/152,871, 2026-07 192,122/191,822. Training DEP sums to 2,085,047 and ranking DEP to
344,841, matching the earlier work. `ranking.parquet` contains exactly these two months and nothing
else (152,719+192,122+152,871+191,822 = 689,534 = the file's row count).

Throughout, "**inside**" means the 2026 value falls within the min–max envelope of the twelve 2025
monthly values for the same column and phase. That envelope is the only baseline available; with
twelve observations it is a weak test for a small step and a strong one for a large step. Where the
2026 value is outside, the size of the excess over the nearest 2025 bound is given.

---

## 1. What was checked

| # | stage | what, per month and per phase | columns |
|---|---|---|---|
| 0 | file metadata | parquet writer, format version, row-group count, compression, column encodings, `ARROW:schema` blob byte-identity, arrow-schema equality | all 14 files |
| 1 | column census | row count, null %, `n_distinct`, `n_distinct` per 1,000 rows, lexical/numeric min and max | all 30 columns × 14 months × 2 phases = 840 cells |
| 2 | timestamp granularity | share of values with second-of-minute = 0, ≡0 mod 10, ≡0 mod 30; number of distinct second values; count of non-zero sub-second components; share on the 5-minute grid | 9 timestamp columns × 14 × 2 |
| 2b | seconds histogram | full 60-bin second-of-minute histogram, TV distance vs uniform, top-3 bins | `MVT_TIME`, `BLOCK_TIME`, `AOBT_3` (DEP) |
| 3 | NM join | join rate (`FLIGHT_ID_mvt` non-null), unmatched count and %, `AOBT_3` non-null %, joined-but-no-AOBT3, unjoined-but-AOBT3, joined-but-no-callsign, unjoined-but-callsign — overall and per airport | 14 × 2, and 10 airports × 14 × 2 |
| 3b | `_flt` block integrity | null rate of all 16 `_flt` columns **conditional on the join succeeding** | 16 × 14 (DEP) |
| 4 | vocabulary | distinct values, values new vs pooled 2025, rows carried by new values, values lost, unseen-in-training rows over the whole ranking set, within-2025 monthly churn baseline, TV distance of the frequency distribution restricted to values present on both sides | 12 single columns + `ADEP+STAND` and `ADEP+RUNWAY` pairs |
| 5 | identifiers | min, max, span, median, distinct count, null count, non-integrality count, per-month fill density, train↔rank set intersection, rank-correlation with `MVT_TIME` | `MVT_ID_mvt`, `FLIGHT_ID_mvt` |
| 6 | arrival rows | label distribution (min, p05, median, p95, p999, max, ≤0, ≥2 h), `BLOCK−MVT ≡ TAXITIME` identity residual, `|BLOCK−SCHED| ≤ 10 s` fill rate, unmatched/stand-null/runway-null/actype-null rates — overall and per airport | 14 × ARR, with DEP counterpart |
| 7 | string format | min/max/mean character length, share matching `^[A-Z0-9]+$`, share containing a lowercase letter, share containing whitespace | 10 string columns × 14 (DEP) |
| 8 | feed coverage | rows per airport, distinct calendar days with ≥1 departure, airport share of departures | 10 airports × 14 |
| 8b | second-of-minute drift | TV distance of the `MVT_TIME` second histogram against the pooled-2025 histogram; share within 5 s of a minute boundary | 14 (DEP) |

Ad-hoc drill-downs (same script's outputs, queried afterwards): per-airport `STAND` distribution TV
against pooled 2025; stand "retirement" and "birth" statistics with a within-2025 baseline; the
EDDM 103–108 time series on both phases.

---

## 2. Findings

### F1 — The ranking file was written by a different R installation than the training files

| file(s) | parquet `created_by` | format | row groups | compression | encodings | `ARROW:schema` (base64 bytes) | R writer version |
|---|---|---|---|---|---|---|---|
| 12 × `training_*.parquet` | `parquet-cpp-arrow version 24.0.0` | 2.6 | 1 | SNAPPY | PLAIN, RLE, RLE_DICTIONARY | 2,744 B, byte-identical across all 12 | **4.6.1** |
| `ranking.parquet` | `parquet-cpp-arrow version 24.0.0` | 2.6 | 1 | SNAPPY | PLAIN, RLE, RLE_DICTIONARY | 2,744 B, 3 bytes differ | **4.5.0** |

The `AIRCRAFT_OPERATOR_flt` field carries an `arrow.r.vctrs` extension type whose
`ARROW:extension:metadata` is an R `serialize(ascii = TRUE)` blob beginning
`A\n3\n<writer R version>\n<min reader R version>\n5\nUTF-8\n`, with version encoded as
65536·major + 256·minor + patch. The training files carry `263681` = **4.6.1**; `ranking.parquet`
carries `263424` = **4.5.0**. Both carry `197888` = 3.5.0 as the minimum reader. Decoding the base64
blobs and diffing them byte by byte: exactly **3 of 2,056 bytes differ**, and all three are inside
that version number. `submitting.parquet` has a 280-byte schema blob with no extension type.
`schema_arrow.equals(..., check_metadata=False)` is `True` for all 13 data files against 2025-01;
with metadata it is `True` for the 12 training files and `False` only for `ranking.parquet`.

**Size of the change:** one minor R version. Nothing else in the file envelope differs — same arrow
C++ writer build, same format version, same single-row-group layout, same compression codec, same
per-column encoding set, same 30-column schema in the same order with the same types.

### F2 — The stand vocabulary steps at July 2026

The `STAND_mvt` value set is airport-local, so the meaningful unit is the `(ADEP, STAND)` pair.

**(a) Retirement.** Stands used in ≥ 11 of the 12 training months that have **zero** departures in a
given month, and the share of that month's departures they would have carried at their mean 2025 rate
(within-2025 rows use a leave-one-month-out version of the same rule, so the baseline is honest):

| month | stands with zero use | implied share of that month's departures |
|---|---|---|
| 2025-01 … 2025-11 | 3 – 12 | 0.014 % – 0.286 % |
| 2025-12 | 23 | 0.756 % |
| **2026-01** | **21** | **0.939 %** |
| **2026-07** | **70** | **2.654 %** |

Jul 2026 is **3.5×** the largest 2025 monthly value and 2.8× Jan 2026; Jan 2026 is itself 1.24× the
2025 maximum, which is Dec 2025 — the 2025 series is not flat and its last month is already its
highest. Restricted to stands that were
also still in use in Jan 2026: 62 stands, carrying **47,124 training departures = 2.260 %** of the
2,085,047 training rows. A further 8 stands are absent in both 2026 months (14,065 rows = 0.675 %:
LIRF 3, LSZH 4, EGLL 1). The largest individual retirements (mean 2025 departures/month → Jan 2026 →
Jul 2026): EHAM/B15 260→223→0, EHAM/B17 256→218→0, EHAM/B23 255→196→0, EDDF/A28 251→198→0,
EDDF/A26 248→204→0, LIRF/501 227→0→0, LIRF/503 221→0→0, LIRF/502 217→0→0, EDDM/111 189→156→0,
LFPG/E44 162→53→0, LEMD/522 154→156→0, EDDM/110 152→145→0.

**(b) Birth.** `(ADEP, STAND)` pairs absent from all of 2025 and from Jan 2026, appearing in Jul 2026:
**27 pairs, 1,434 departures = 0.746 %** of Jul-2026 departures. Against pooled 2025 alone (allowing
pairs that also appeared in Jan 2026) it is 31 pairs / 1,549 rows / 0.806 %. Within-2025 baseline for
the same statistic (new pairs vs all earlier months): 0.0075 % – 0.228 % per month, so Jul 2026 is
**3.3×** the 2025 maximum. Jan 2026 is 19 pairs / 40 rows / 0.026 % — inside the 2025 range.

**(c) The largest single block is at EDDM.** Stands `103`–`108`:

| | 2025-01 … 2025-12 (each) | 2026-01 | 2026-07 |
|---|---|---|---|
| EDDM departures on stands 103–108 | 0 | 0 | 1,177 (**7.84 %** of EDDM departures) |
| EDDM arrivals on stands 103–108 | 0 | 0 | 1,177 (**7.84 %** of EDDM arrivals) |
| EDDM/110 departures | 138 – 180 | 145 | 0 |
| EDDM/111 departures | 155 – 205 | 156 | 0 |

EDDM/233, /234, /255, /256 also fall from 54–143 departures per month through Jan 2026 to 0 or 1 in
Jul 2026. The EDDM stand count is unchanged (164–182 distinct stands per month in every partition,
173 in Jul 2026), and EDDM/244, /245, /247 roughly double. The second block is at EDDF: `J5`, `J10`,
`V333`, `V335`, `V337`, `XDP1M` are new (196 rows), while `A28`, `A26`, `V171A`, `V171B`, `V173B` go
to zero and `A38`, `J1`, `J2`, `J4`, `J8`, `H5`, `H6`, `H8` — all present in 2025 at other times —
jump to 111–264 departures each.

**(d) Distribution-level size.** TV distance of each month's within-airport stand distribution against
the pooled-2025 distribution:

| airport | 2025 min | 2025 max | 2026-01 | 2026-07 | Jul26 ÷ 2025 max |
|---|---|---|---|---|---|
| EDDF | 0.0385 | 0.0779 | 0.0737 | **0.1824** | **2.34** |
| EDDM | 0.0664 | 0.1135 | 0.0962 | **0.1776** | **1.56** |
| EGLL | 0.0292 | 0.0408 | 0.0459 | 0.0555 | 1.36 |
| EHAM | 0.0466 | 0.0748 | 0.0736 | 0.0922 | 1.23 |
| LSZH | 0.0467 | 0.1127 | 0.1558 | 0.1377 | 1.22 |
| LEMD | 0.0492 | 0.0746 | 0.0711 | 0.0879 | 1.18 |
| LFPG | 0.0613 | 0.1018 | 0.1105 | 0.1168 | 1.15 |
| LIRF | 0.0387 | 0.1195 | 0.1300 | 0.1301 | 1.09 |
| LTFM | 0.0382 | 0.0592 | 0.0487 | 0.0630 | 1.06 |
| LEBL | 0.0422 | 0.0931 | 0.1052 | 0.0708 | 0.76 |

Only EDDF and EDDM are clearly outside; LSZH's largest deviation is in Jan 2026 (0.1558 vs 0.1127).

### F3 — Unseen-value census: earlier figures reproduced exactly, and extended

Over the whole ranking DEP set (344,841 rows), values not present among 2025 departures:

| column | unseen values | unseen rows | 2026-01 (values / rows / % of month) | 2026-07 (values / rows / % of month) |
|---|---|---|---|---|
| `STAND_mvt` | 30 | **284** | 16 / 28 / 0.018 % | 17 / 256 / 0.133 % |
| `AIRCRAFT_OPERATOR_flt` | 30 | **177** | 3 / 4 / 0.003 % | 28 / 173 / 0.090 % |
| `ADES_mvt` | 62 | **174** | 22 / 24 / 0.016 % | 41 / 150 / 0.078 % |
| `AIRCRAFT_TYPE_mvt` | 8 | **18** | 2 / 9 / 0.006 % | 6 / 9 / 0.005 % |
| `RUNWAY_mvt` | 0 | **0** | 0 / 0 / 0 % | 0 / 0 / 0 % |
| `ADES_FILED_flt` | 51 | 154 | 19 / 21 / 0.014 % | 32 / 133 / 0.069 % |
| `AIRCRAFT_TYPE_flt` | 6 | 9 | 1 / 1 / 0.001 % | 5 / 8 / 0.004 % |
| `MARKET_SEGMENT_flt`, `FLIGHT_TYPE_flt`, `WK_TBL_CAT_flt`, `ADEP_mvt`, `FLIGHT_RULE_mvt` | 0 | 0 | 0 | 0 |
| **`ADEP+STAND_mvt`** | **46** | **1,589** | 19 / 40 / 0.026 % | 31 / 1,549 / 0.806 % |
| `ADEP+RUNWAY_mvt` | 1 | 2 | 0 / 0 | 1 / 2 (`LFPG/NA`) |

The five figures quoted by the earlier work — STAND 284, OPERATOR 177, ADES 174, AIRCRAFT_TYPE 18,
RUNWAY 0 — are reproduced to the row. **The extension that matters is the pair column**: treating a
stand as airport-local raises the unseen count from 284 to **1,589 rows (0.461 % of the ranking set)**,
a factor of 5.6, and concentrates it almost entirely in Jul 2026 (1,549 of 1,589). The single-column
view understates it because stand labels such as `103` or `J5` exist at some other airport in 2025.

Within-2025 monthly churn for reference (rows carried by values new relative to all earlier months):
`STAND_mvt` 0.003 %–0.109 %; `ADES_mvt` 0.027 %–0.144 %; `AIRCRAFT_TYPE_mvt` 0.002 %–0.028 %;
`AIRCRAFT_OPERATOR_flt` 0.002 %–0.385 %; `RUNWAY_mvt` 0 %–0.0035 %; `ADEP+STAND_mvt` 0.008 %–0.228 %.
On the single-column statistics both 2026 months sit inside this band; on the pair statistic Jul 2026
does not.

### F4 — `AOBT_3_flt` minute-rounding rate drops below the 2025 band in January 2026, on both phases

Share of non-null `AOBT_3_flt` with second-of-minute = 0:

| phase | 2025 min | 2025 max | 2026-01 | 2026-07 |
|---|---|---|---|---|
| DEP | 96.595 % (Jan) | 98.448 % (Jul) | **95.799 %** (−0.796 pp) | **98.723 %** (+0.275 pp) |
| ARR | 88.974 % (Dec) | 90.338 % (Oct) | **87.711 %** (−1.263 pp) | 89.641 % (inside) |

The within-2025 DEP spread is 1.853 pp, so the two DEP excesses are 43 % and 15 % of the historical
spread; the ARR spread is only 1.364 pp, so the ARR January excess is 93 % of it. The `≡ 0 mod 10` series moves in lockstep
(DEP −0.705 / +0.246 pp, ARR −1.134 pp). That the same January dip appears on both phases makes it a
property of the `AOBT_3` feed rather than of the departure extract. The non-zero-second tail also
changes shape on DEP: the top non-zero second bins in Jan 2026 are 37 (0.13 %) and 40 (0.12 %), where
every 2025 month's top non-zero bins are 14–22. Everything else about the column is unchanged:
60 distinct second values, no sub-second component, minute ≡ 0 mod 5 at 20.19 %–20.51 % in all
14 months. The 2025 DEP series is itself seasonal (low Jan/Mar/Nov, high Jun–Aug) and the 2026 values
extend that swing in the same direction rather than reversing it.

### F5 — `ARVT_3_flt` is more minute-rounded in July 2026 than in any 2025 month

Share of non-null `ARVT_3_flt` with second-of-minute = 0:

| phase | 2025 series | 2026-01 | 2026-07 |
|---|---|---|---|
| DEP | 26.6, 28.0, 26.6, 30.3, 31.0, 30.9, 29.8, 30.8, 31.4, 30.9, 30.4, 31.2 % | 29.387 % (inside) | **34.972 %** (**+3.59 pp** over the 2025 max) |
| ARR | 38.5, 39.8, 36.2, 44.6, 46.3, 47.1, 48.2, 53.9, 55.0, 50.7, 49.2, 51.6 % | 47.492 % (inside) | **59.119 %** (**+4.17 pp** over the 2025 max) |

The DEP excess is 75 % of the whole 2025 DEP spread. On ARR the 2025 series already rises steadily
(36 % in March to 55 % in September), so July 2026 is closer to a continuation of a trend than to a
step; on DEP the 2025 series is flat from April onward (≈ 30–31 %) and July 2026 is a step.
`ARVT_3` on a departure row is the arrival time of that flight at its destination — the most
downstream field in the record.

### F6 — Small residual steps, listed for completeness

| observation | 2025 range | 2026-01 | 2026-07 |
|---|---|---|---|
| ARR label p99.9 (s) | 2,230 – 2,637 | **3,175** (+20 % over max) | 2,340 (inside) |
| `MVT_TIME` second-histogram TV vs pooled 2025 (DEP) | 0.00554 – 0.01212 | **0.01400** (+15 %) | 0.00772 (inside) |
| `MVT_TIME` share within 5 s of a minute boundary (DEP) | 56.118 % – 57.289 % | **57.686 %** (+0.40 pp) | 56.161 % (inside) |
| `MVT_TIME` sec = 0 share (DEP / ARR) | 5.020–5.218 / 5.031–5.210 % | **5.264 / 5.304 %** (+0.05 / +0.09 pp) | 5.055 / 5.128 % (inside) |
| `LOBT_flt` and `EOBT_1_flt` sec = 0 share (DEP) | 99.490–99.842 / 99.496–99.846 % | **99.896 / 99.898 %** (+0.054 / +0.052 pp) | inside |
| `ARVT_1_flt` sec = 0 share (DEP) | 1.654 – 1.747 % | **1.815 %** (+0.068 pp) | 1.747 % (inside) |
| ARR overall `BLOCK == SCHED` fill | 2.273 – 2.741 % | **2.220 %** (−0.053 pp) | 2.574 % (inside) |
| ARR taxi-in median, cells outside their airport's 2025 range | — | EHAM +13 s, LSZH +5 s, EGLL +4 s, LIRF −10.5 s, LTFM −1 s | EDDF −8.5 s, LIRF +2 s |
| ARR `BLOCK == SCHED` fill, cells outside range | — | EDDM −0.27, LIRF −0.17, LFPG −0.16, EDDF −0.11 pp | EDDF −0.04 pp |
| `ADES_FILED_flt` distinct per 1,000 ARR rows | 0.457 – 0.755 | **0.903** | 0.553 (inside) |
| EHAM departure unmatched rate | 1.127 % – 2.502 % | **4.959 %** | 1.827 % (inside) |
| LSZH departure unmatched rate | 0.954 % – 2.736 % | **3.002 %** | 2.729 % (inside) |
| airport share of departures: EHAM | 11.607 % – 12.420 % | **10.629 %** | 11.425 % (inside) |
| airport share of departures: LTFM | 12.491 % – 13.902 % | **14.665 %** | 13.109 % (inside) |
| airport share of departures: EGLL | 10.894 % – 12.635 % | 12.626 % (inside) | **10.700 %** |

Every timestamp-granularity excess in this table other than F4's and F5's is at most 0.094 pp, i.e.
under one row in a thousand. The ARR taxi-in and fill excesses are at most 2–4 % of the statistic.

The EHAM and LSZH departure unmatched rates and the airport shares are flight-population facts already
covered by the earlier reports; they are listed here only because they are the cells where a
per-month recording scan lights up, and because §3 uses the arrival side to show they are *not* join
failures.

---

## 3. What is CLEAN

Each item below is stated as a property that holds in **all 14 month-partitions**, i.e. the 2026 value
is inside the 2025 envelope or identical to it.

**File envelope.** One parquet writer (`parquet-cpp-arrow 24.0.0`), format 2.6, one row group per
file, SNAPPY, the encoding set {PLAIN, RLE, RLE_DICTIONARY} for every column of every file. The same
30 columns in the same order with identical arrow types in all 13 data files; the only metadata
difference anywhere is F1's 3 bytes.

**Partitioning.** `ranking.parquet` holds exactly two calendar months by `MVT_TIME_UTC_mvt` and
nothing else. Every one of the 10 airports has a departure on every calendar day of both 2026 months
(31/31 and 31/31), as in every 2025 month.

**Null rates.** Of the 30 columns × 2 phases, the only null rates that leave the 2025 envelope are
`TAXITIME_SEC_mvt` and `BLOCK_TIME_UTC_mvt` on DEP rows, which go from 0 % to exactly 100 % — the
designed masking of the target. `SCHED_TIME_UTC_mvt`, `MVT_TIME_UTC_mvt`, `MVT_ID_mvt`, `ADEP_mvt`,
`PHASE_mvt` and `RUNWAY_mvt` have 0 nulls in all 28 cells. `STAND_mvt` 0.0005 %–0.0046 % in 2025 vs
0.0007 % / 0.0016 %; `AIRCRAFT_TYPE_mvt` 0.034 %–0.178 % vs 0.071 % / 0.146 %. The remaining
columns whose null rate moves at all move by ≤ 0.02 pp.

**The `_flt` block is all-or-nothing with the NM join, unchanged.** In all 28 cells the count of rows
with `FLIGHT_ID_mvt` null but `AOBT_3_flt` present is **0**, and likewise for `CALLSIGN_flt`.
Conditional on the join succeeding, every one of the 16 `_flt` columns has a null rate of
0.000 %–0.009 % in every one of the 14 months, 2026 included (2026-01: 0.003 % for 14 columns,
0.009 % for `FLIGHT_RULE_flt` and `WK_TBL_CAT_flt`; 2026-07: 0.002 % / 0.004 %). The join either
attaches the whole flight record or none of it, in 2026 exactly as in 2025.

**The NM join rate itself.** Departures: 2025 unmatched 0.618 %–2.063 %, 2026-01 **1.604 %**,
2026-07 **1.475 %** — both inside. Arrivals: 2025 unmatched 0.372 %–0.458 %, 2026-01 **0.417 %**,
2026-07 **0.402 %** — both near the middle. Per airport on the arrival side, every one of the 20
2026 cells is inside that airport's 2025 range (EHAM 1.231 % / 1.082 % vs 1.020 %–1.545 %;
LSZH 0.863 % / 0.813 % vs 0.416 %–1.348 %; LTFM 1.188 % / 1.431 % vs 0.878 %–1.369 %, the one
marginal case). `joined but no AOBT_3` on arrivals is 479–659 rows per month in 2025 and 504 / 639 in
2026. Since departures and arrivals are matched by the same mechanism against the same NM flight
table, an arrival-side join rate that does not move is direct evidence that the join did not change.

**Timestamp recording granularity.** No sub-second component anywhere: `dt.microsecond() != 0` counts
**0** for all 9 timestamp columns in all 28 cells. `SCHED_TIME_UTC_mvt` has second = 0 on 100 % of
rows in all 14 months and both phases, and `IOBT_flt` likewise; `SCHED_TIME` sits on the 5-minute
grid at 99.944 %–99.981 % (DEP) in all 14 months, 2026 included. `MVT_TIME` uses all 60 second values
in every month, is minute ≡ 0 mod 5 on 19.92 %–20.15 % of rows in all 14 months, and its clustering
near the minute boundary (56.1 %–57.7 % within ±5 s) is unchanged. `BLOCK_TIME` on ARR rows has
second = 0 on 6.268 %–6.559 % of rows in all 14 months. Of the 54 (column × phase × statistic) 2026
cells checked, the only ones outside the 2025 envelope by more than 0.1 pp are `AOBT_3` (F4) and
`ARVT_3` (F5); the rest — `MVT_TIME`, `BLOCK_TIME`, `SCHED_TIME`, `LOBT`, `IOBT`, `EOBT_1`, `ARVT_1` —
are inside or outside by 0.008–0.094 pp (F6).

**Identifier columns mean the same thing.** `MVT_ID_mvt`: 0 nulls, 0 non-integral values and
100 % distinct in all 14 months; the train and rank ID sets are **disjoint** (intersection 0 of
4,167,797 and 689,534); the ranges are continuous and forward-monotone across the boundary
(2025-12 max 202,064,541 → 2026-01 min 202,180,234, a gap of 115,693, against monthly spans of
0.88 M–2.36 M); the per-month fill density (rows ÷ ID span) is 0.157–0.326 in 2025 and 0.243 / 0.240
in 2026; the rank-correlation with `MVT_TIME` inside a month is 0.052–0.073 in all 14 months, i.e.
the ID is not time-ordered within a month either before or after the boundary. `FLIGHT_ID_mvt`:
0 non-integral, ranges continuous with a 4-ID overlap exactly at the Dec-2025/Jan-2026 seam
(train max 294,567,590, rank min 294,555,620), distinct-per-1,000-rows 979–994 in 2025 vs 984 / 985
in 2026. No renumbering, no offset, no re-encoding.

**String formats.** `FLIGHT_mvt` length 3–8, mean 6.194–6.221 (2026: 6.202 / 6.198);
`CALLSIGN_flt` 3–7, mean 6.279–6.317 (2026: 6.280 / 6.272); `ADES_mvt` exactly 4;
`AIRCRAFT_TYPE_mvt` 3–4; `RUNWAY_mvt` 1–3; `STAND_mvt` 1–7, mean 3.003–3.028. All of these are
100 % `[A-Z0-9]`, 0 % containing a lowercase letter, 0 % containing whitespace, in all 14 months.
`AIRCRAFT_OPERATOR_flt` is a 64-character all-lowercase-hex digest in all 14 months (length min =
max = 64), and the digest is stable across the boundary — only 30 of 676 training operator values
are new in the ranking set, on 177 rows. `MARKET_SEGMENT_flt` is mixed-case free text of length 5–13
in all 14 months.

**Closed vocabularies are identical.** `MARKET_SEGMENT_flt` 8 values, `FLIGHT_TYPE_flt` 5,
`WK_TBL_CAT_flt` 4, `FLIGHT_RULE_mvt` 2, `ADEP_mvt` 10 — zero new and zero lost values in either
2026 month, and frequency TV distances of 0.0002–0.046 against pooled 2025, at or below the
2025-01-vs-2025-07 baseline for the same column.

**`RUNWAY_mvt` is clean on every test.** 0 nulls in all 28 cells; 0 unseen values over the whole
ranking set; 0 new values in either 2026 month; the `ADEP+RUNWAY` pair produces exactly 1 new value
on 2 rows (`LFPG/NA`); and the per-airport runway-share TV against pooled 2025 is inside the 2025
range at **all ten airports in both 2026 months** (ratios to the 2025 max 0.07–0.96).

**Frequency shift of values that exist on both sides is at or below the within-2025 seasonal level**
for every vocabulary column. TV against pooled 2025, restricted to shared values
(2026-01 / 2026-07, with the 2025-01-vs-2025-07 baseline in brackets): `STAND_mvt`
0.078 / 0.094 [0.090]; `RUNWAY_mvt` 0.126 / 0.112 [0.181]; `ADES_mvt` 0.076 / 0.076 [0.112];
`AIRCRAFT_OPERATOR_flt` 0.081 / 0.063 [0.086]; `MARKET_SEGMENT_flt` 0.046 / 0.033 [0.043];
`ADEP_mvt` 0.040 / 0.019 [0.038]; `FLIGHT_RULE_mvt` 0.0004 / 0.0002 [0.0007]. The two columns
slightly above their baseline are `AIRCRAFT_TYPE_mvt` 0.061 / 0.069 [0.041] and
`AIRCRAFT_TYPE_flt` 0.071 / 0.079 [0.054].

**Value ranges.** Of 60 column × phase envelopes, the only 2026 values outside the 2025 min–max are
the ones that must be: the two monotone ID columns and the eight timestamp columns (all "max >",
because 2026 is later), plus `CALLSIGN_flt` on ARR rows where the 2026-01 lexical minimum is below
`2AMBO`. Every categorical column's lexical min and max in 2026 are inside the 2025 envelope, and
§3's format checks show no new character classes.

**The arrival rows — the window onto 2026 — behave identically.**
- Label present on 344,693 of 344,693 ranking arrivals (2025 has one null in 2025-11).
- The label identity `TAXITIME_SEC_mvt == BLOCK_TIME − MVT_TIME` holds with **maximum absolute
  residual 0 seconds in all 14 months**, 2026-01 and 2026-07 included. The label is computed the same
  way in 2026.
- Label distribution: median 466–490 s in 2025 vs **487 / 470**; p05 162–184 vs 181 / 172;
  p95 1,069–1,147 vs 1,134 / 1,104; count ≤ 0 s 6–31 vs 21 / 23; count ≥ 2 h 3–25 vs 14 / 13. Only
  p99.9 leaves the envelope (F6).
- Taxi-in median per airport: 13 of the 20 2026 cells are inside their airport's 2025 range; the
  seven that are not miss it by 1–13 seconds (F6), i.e. by ≤ 2.2 % of the statistic.
- The LIRF `BLOCK == SCHED` fill pathology is present on the arrival side at essentially its 2025
  rate: LIRF 8.74 %–12.76 % in 2025 vs **8.57 % / 10.57 %** in 2026 (Jan 0.17 pp below the 2025
  minimum). Fourteen of the twenty 2026 airport cells are inside their 2025 range and the other six
  miss it by 0.04–0.27 pp (F6). This is the one departure-side pathology the audit can still observe
  in 2026, and it has not gone away.
- Arrival null patterns: stand-null 0 %–0.0047 % vs 0.0007 % / 0.0021 %; runway-null 0 % everywhere;
  actype-null 0.0012 %–0.0137 % vs 0.0046 % / 0.0073 %.

---

## 4. Facts and inference, separated

**Facts — measured in this run, from `data/raw/*.parquet` only.** Every number in §1–§3: the file
metadata table and the 3-byte `ARROW:schema` difference with the decoded R versions 4.6.1 / 4.5.0;
all 840 per-column × month × phase null, distinct, min and max values; the timestamp
second-of-minute and sub-second tables; the join-rate tables overall and per airport for both phases;
the conditional `_flt` null table; the vocabulary tables including the reproduction of STAND 284 /
OPERATOR 177 / ADES 174 / AIRCRAFT_TYPE 18 / RUNWAY 0 and the `ADEP+STAND` extension to 1,589 rows;
the stand retirement and birth counts with their within-2025 baselines; the EDDM 103–108 and
EDDM/110–111 time series on both phases; the identifier range, density, integrality and overlap
figures; the arrival-row label identity residual of 0 s in all 14 months and the arrival LIRF fill
rates; the string-format fingerprints; the per-airport day coverage.

**Inference — not measured here.**
- That the R-version difference means the ranking file and the training files were produced by
  *different runs* of the export pipeline. What is measured is one differing version string in one
  metadata blob. It is consistent with a single pipeline run on two machines, with a re-export of the
  training files after the ranking file was built, or with the ranking file having been generated
  earlier; nothing in the data distinguishes these. No behavioural difference between the two files
  has been traced to it, and none of §3's clean results would be expected to survive a genuine
  format change.
- That the EDDM 103–108 / 110–111 substitution is a **renumbering** of the same physical stands rather
  than the opening of new ones and the closure of others. The evidence is only that the counts trade
  off at roughly the same magnitude, in the same month, at the same airport, on both phases. No
  airport layout, AIP or NOTAM source was consulted, and the five months Feb–Jun 2026 are not in the
  data, so the date of the change is bounded only to (2026-01-31, 2026-07-01].
- That the 2.654 % of Jul-2026 departures on retired stands and the 0.746 % on newborn stands are the
  *same* rows re-labelled. They are not matched row by row here.
- That F4's `AOBT_3` rounding step and F5's `ARVT_3` step indicate a change in the upstream feed. Both
  are small relative to their own 2025 seasonal swing, and the 2025 series has only one observation
  per calendar month, so a genuine seasonal effect and a recording change are not separable with this
  data. The two columns are the only timestamps that move at all.
- That the arrival rows are a valid proxy for departure-side recording. They share the file, the
  schema, the `MVT_ID` generator, the NM join and the `TAXITIME` formula, but they are different
  records of different events; a change confined to the departure feed would be invisible in them.
  The specific departure-side columns that could hide such a change and cannot be cross-checked on
  arrivals are `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt`, which are null on all 344,841 ranking
  departure rows.
- That the Jan-2026 EHAM and LSZH departure unmatched rates in F6 are not join failures. The
  arrival-side join rate at those airports stays inside its 2025 range in the same month, which is
  evidence for that reading but does not prove the departure excess is an operational event; the
  earlier reports own that question.
- Anything about the ranking labels, which are null on all 344,841 departure rows, so no
  recording property of the target on the scored side can be checked directly.
