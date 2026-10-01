# LIRF and LFPG lane A: what the worst rows share, and one mechanism that is visible (2026-09-29)

**Script (read-only; nothing written to `outputs/`, `src/`, `configs/`, `data/`; no production fit):**
`scripts/lirf_lfpg_mechanisms.py`, run as
`SCRATCH=<dir> POLARS_MAX_THREADS=2 .venv/bin/python scripts/lirf_lfpg_mechanisms.py <A..J>`.
Every number below carries the section that prints it (`[A]` … `[J]`). `[inline]` = a one-off shell query,
described where it is used. `[X]` = an inference, marked as one. G needs C's saved probabilities, so run C
before G. The peak `total_rss` of any section was 711 MB (`[J]`), against the 13,681 MB limit.

**Inputs.** `data/work/feat_train.parquet`, `data/work/feat_rank.parquet`, `data/raw/training_*.parquet`
and `data/raw/ranking.parquet` (for the **arrival** rows), `experiments/oof_main_gap-62dd7c6e88.parquet` and its
`_folds` sidecar (pooled jan/jul/forward, `jul_fwd` left out as elsewhere). Lane A = `mvt_minus_aobt3` not null:
664,050 rows, RMSE **213.1266** `[A]`, which reproduces `delay-mechanisms.md`.

**Parity.** The one new feature family reads `BLOCK_TIME_UTC_mvt` and `SCHED_TIME_UTC_mvt` of **arrival** rows
only. `ranking.parquet` keeps arrival BLOCK on all 344,693 scored arrivals, and `stand_arrival_gap` reads it
already. `[I]` asserts that departure BLOCK is null on every `feat_rank` row. The departure's own
`|BLOCK − SCHED|` is used only as a diagnostic label, never as a feature, and `[J]` asserts that the label is
absent from the feature list. **Clock resolution:** every `MVT_TIME` at LIRF sits within ±6 s of a whole minute
(share 1.0000 `[A]`), so "equal" below means `|Δ| ≤ 12 s`.

---

## 0. The answer

1. **The largest single piece of LIRF's error is one mechanism: the airport writes the timetable time as the
   off-block time for a flight whose plan has moved.** 863 LIRF lane-A rows (0.13 % of lane A, 1.75 % of
   LIRF) have `BLOCK == SCHED` to the minute while `LOBT ≠ SCHED`. They carry **5.69 % of all lane-A SSE and
   29.2 % of LIRF's**, with RMSE 1,410 s and bias −966 s. On those rows `y = mvt_minus_sched` exactly. The
   exact-minute bin holds 863 rows against an average of 50 in each neighbouring minute, a 17× spike. No other
   airport has it: their equivalent cells are ≤ 0.18 % of lane-A SSE, and LTFM's 1,284 rows sit on 1,256
   neighbours, so there is no spike. `[A]`
2. **LIRF's ten worst days are mostly the same thing.** Fill rows carry **44.3 %** of the SSE on those days
   `[A]`. On the worst day (2025-07-13) they carry 32 %. Most of the rest of that day reads `BLOCK ≈ LOBT`
   while `AOBT_3 ≈ MVT − 15 min`. `[X]` NM did not see the real off-block and fell back to a nominal taxi
   time. §1.3 below covers this.
3. **The fill has a visible mirror: LIRF also fills *arrival* in-block times with the schedule, and the
   arrival record is not blanked.** The LIRF arrival in-block equals SCHED on 10.1 % of training arrivals
   and 9.7 % of scored arrivals, a 5.35× and 5.39× spike. Every other airport is ≤ 1.11× `[B]`. The departure
   fill rate is **0.399** when the same-stand, same-operator arrival that `stand_arrival_gap` links to was
   filled, against **0.156** when it was not. On moved-plan rows it is **0.274 against 0.052**. The lift holds
   in all four OOF months `[B]`.
4. **It survives month-blocked tests, and it is small.** Adding four arrival columns lifts a
   leave-one-month-out LIRF fill classifier from AUC 0.8659 to 0.8715, and it wins in **12 of 12** held-out
   months. The permuted control reads 0.8656 `[C]`. On the calibrated residual instrument the arm reads
   **−0.071 s against control4**, and its permuted twin reads +0.002 `[D]`. That implies about −0.12 s of
   lane A. With `ms` and the prediction in every arm's base, it reads −0.166 against its permuted twin. The
   sign is negative in **all three quarter folds**: jan −0.18, jul −0.08, nov-dec −0.23 `[E]`, `[H]`.
5. **The ceiling is large and this does not reach it.** An oracle that knew which moved-plan LIRF rows are
   fills would gain **−6.15 s** of lane A `[H]`. The production model already separates fills about as well
   as the base classifier (AUC 0.87, 0.92 on moved rows `[C]`). The arrival side adds only about 0.006 AUC.
6. **LFPG has no counterpart.** Fill rows carry 0.4 % of LFPG's SSE `[A]`. 89 % of LFPG's SSE is diffuse.
   Its worst days are snow and high-delay days on which both clocks moved together. Its 300 worst
   under-predictions (21 % of LFPG SSE `[F]`) are "left the stand at LOBT, NM clock 37 min later", and I
   found no visible separator for them (§3).
7. **Three things on the record read differently after this** (§4). The LIRF slope of 0.949 is inflated by
   the fill. The "LIRF replanned rows = 11.5 % of SSE" lead is about half fill. And the closed P1 gate
   recipe now reproduces at 0.9258, not 0.9201.

---

## 1. LIRF: what the worst rows have in common

### 1.1 Reading the forty worst rows

In the 40 largest `|r|` rows at LIRF `[inline]`, 28 have `BLOCK_TIME` within 12 s of `SCHED_TIME` (WMT404
11:30:02 vs 11:30:00, WMT8PV 10:15:03 vs 10:15:00, RYR51XV 09:44:58 vs 09:45:00, …). Their `LOBT`, `EOBT_1`
and `AOBT_3` are 40–110 min later, and their residual is −3,298 to −5,192 s. The model predicts a normal
taxi on each of them, because `proxy` is normal (~900 s), while the label is `MVT − SCHED`. The other 12 include one
over-prediction (UAL41, +3,847); the rest are the kind in §1.3.

### 1.2 The fill, decomposed `[A]`

| LIRF lane-A kind | n | RMSE | bias | share of LIRF SSE | share of lane-A SSE |
|---|---:|---:|---:|---:|---:|
| sched-fill, `SCHED ≠ LOBT` | **863** | **1,410.2** | **−966.2** | **0.292** | **0.0569** |
| sched-fill, `SCHED == LOBT` | 8,087 | 292.7 | −67.2 | 0.118 | 0.0230 |
| `BLOCK` > 20 min before `LOBT`, not fill | 203 | 1,353.3 | −809.7 | 0.063 | 0.0123 |
| rest | 40,234 | 277.7 | +41.9 | 0.527 | 0.1028 |

When `SCHED == LOBT`, a schedule fill is also roughly the planned off-block, so the damage is small (RMSE
293). When the plan has moved, the fill is wrong by the whole delay. The fill rate on moved rows falls to 0
once `LOBT − SCHED ≥ 70 min` `[inline]`. That is the `|BLOCK − LOBT| ≤ 3,606 s` matching window: a fill that
far out would fall outside it, and the row would not be on lane A.

Which plans get filled is strongly operator-shaped `[inline]`. The home carrier ITY is at 0.8 %. TKJ, ISR and
TWB are at 100 %, UPS at 86 %, and cargo and non-scheduled flights at about 42 %. `[X]` This points to a
handler that does not report actual times. `AIRCRAFT_OPERATOR_flt` is already a model categorical, which is
why the base classifier in §2.2 starts at AUC 0.87.

### 1.3 The worst days

| day `[A]` | n | bias | median `y` | median `proxy` | median `ms` | fill rate | fill share of the day's SSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2025-07-13 | 445 | −306.5 | 1,977 | 962 | 4,200 | 0.101 | 0.319 |
| 2025-07-28 | 493 | −21.1 | 1,508 | 1,197 | 3,538 | 0.126 | 0.331 |
| 2025-07-03 | 455 | +22.1 | 1,076 | 896 | 2,339 | 0.123 | 0.439 |
| 2025-07-04 | 461 | −4.0 | 1,137 | 899 | 2,335 | 0.161 | 0.556 |
| 2025-07-06 | 506 | −10.7 | 1,131 | 897 | 2,157 | 0.152 | 0.623 |
| median of all LIRF days | | | 1,014 | 900 | 1,264 | 0.189 | |

All ten of the worst days are in July 2025, and all have high `ms` (heavy delay). Delay is what turns a fill
into a large error. On 2025-07-13, reading every sixth row `[inline]`, the non-fill rows from 07:00 onward
show `g ≈ AOBT_3 − LOBT` (1,979 vs 1,980; 1,444 vs 1,440; 2,758 vs 2,760; 3,177 vs 3,180) and
`BLOCK − LOBT` near zero, while `proxy` stays at 780–1,200. `[X]` Those aircraft left the stand at about
their LOBT and queued. NM's `AOBT_3` at LIRF is not an observed off-block on these rows: it is take-off minus
a nominal allowance. That fits LIRF's day-median `proxy`, which is 896–1,197 on its ten worst days and 900 on the median day `[A]`, while the day-median `y` reaches 1,977. Across all airports this kind
(`g > 1,200 s`, not a fill) is 6,152 rows carrying **16.7 % of lane-A SSE**. LIRF has 910 of them (2.9 % of
lane-A SSE), and LFPG is the largest cell `[inline]`. I found nothing visible that separates it. The
airport-hour state arms in `delay-mechanisms.md` §4 are the closest test on record, and they read worse than
their control.

---

## 2. The mechanism: the arrival mirror of the schedule fill

### 2.1 Visible, and present on the scored set `[B]`, `[I]`

| | LIRF arrivals, in-block == sched | spike over ±1 min | every other airport |
|---|---:|---:|---:|
| training 2025 | 0.1008 | 5.35× | ≤ 1.11× |
| ranking 2026 | 0.0974 (Jan 0.086, Jul 0.106) | 5.39× | ≤ 1.10× |

Columns, built by `build_arrival_fill` (arrival rows only):

- `arr_link_fill` (Int8): the same-stand arrival that `stand_arrival_gap` links to had its in-block
  written as its schedule.
- `arr_link_bs_abs`: that arrival's `|in-block − sched|`.
- `arr_link_same_op`: that arrival has the same operator as this departure.
- `arr_opday_fill`: the operator × airport × UTC-date arrival fill rate, shrunk toward the airport-day rate
  with k = 5.

On LIRF lane A, link coverage is 0.879 in training and 0.878 on the scored rows `[I]`.

**Departure fill rate by linked-arrival fill** (LIRF lane A, the departure label used as a diagnostic) `[B]`:

| | linked arrival not filled | linked arrival filled |
|---|---:|---:|
| all rows, same operator | 0.156 (37,672) | **0.399** (3,273) |
| moved-plan rows (`SCHED ≠ LOBT`) | 0.052 (11,471) | **0.274** (492) |
| month 1 / 7 / 11 / 12 | 0.172 / 0.120 / 0.169 / 0.178 | 0.341 / 0.362 / 0.453 / 0.416 |

The graded version on moved-plan rows: `arr_opday_fill` ≤ 0.05 → 0.026; 0.05–0.15 → 0.071; 0.15–0.30 →
0.163; 0.30–0.50 → 0.423 `[B]`. At the day level, departure and arrival fill rates correlate at 0.447, and
0.397 within month `[inline]`. `[X]` It is a contemporaneous estimate of the seasonal fill rate that
`P1B-lirf-seasonal-bias-REJECT.md` could not estimate out of month.

### 2.2 Does it add to what the model has? A month-blocked classifier `[C]`

The data are all 159,216 LIRF matched training rows, with leave-one-calendar-month-out folds (twelve).
"base" is what the production model can see about the row: stand, operator, type, segment, runway,
destination, callsign prefix, hour, dow, the eight NM/schedule clocks, `stand_arrival_gap` and
`stand_aobt_turn`.

| arm | AUC | AUC, `SCHED ≠ LOBT` | logloss | per-month AUC min / max |
|---|---:|---:|---:|---:|
| base | 0.8659 | 0.9187 | 0.3187 | 0.8445 / 0.8824 |
| **base + arrival (4)** | **0.8715** | **0.9246** | **0.3116** | 0.8515 / 0.8833 |
| base + arrival permuted within month (control) | 0.8656 | 0.9183 | 0.3190 | 0.8443 / 0.8826 |
| arrival only | 0.6993 | 0.7998 | 0.4212 | — |

base + arrival beats base in **12 of 12** held-out months. The control is indistinguishable from base.

### 2.3 On the residual: the calibrated instrument and its extension

`[D]` uses `delay_mechanisms.month_blocked_gain` unchanged, with the base `[adep, hour_local, proxy,
runway]`. It runs three seeds and folds by calendar quarter; the transfer coefficient 0.57 comes from
`delay-mechanisms.md` §4.

| arm `[D]` | lane-A RMSE | seed sd | vs control4 |
|---|---:|---:|---:|
| base | 213.5826 | 0.0077 | |
| control4 | 213.5158 | 0.0027 | 0 |
| **arrival fill (4)** | **213.4451** | 0.0004 | **−0.0706** |
| arrival fill permuted (4) | 213.5181 | 0.0051 | +0.0023 |

A fill correction scales with `ms − pred`, and that instrument cannot see it. `[E]` therefore puts
`ms − oof` and `LOBT − SCHED` (visible as `mvt_minus_sched − mvt_minus_lobt`) in every arm's base, the control
included. `[E]` is **not calibrated**, so read it only against its own permuted twin:

| `[E]` | lane-A RMSE | vs permuted twin |
|---|---:|---:|
| base + ms | 213.3649 | |
| control4 | 213.3435 | |
| **arrival fill (4)** | **213.1794** | **−0.166** |
| arrival fill permuted (4) | 213.3454 | 0 |
| LIRF rows only: arrival vs permuted (LIRF RMSE) | | **−0.922** |

**Per quarter fold** `[H]` (`[E]` recipe, three-seed mean prediction):

| fold | lane A, arrival − permuted | LIRF, arrival − permuted | non-LIRF |
|---|---:|---:|---:|
| jan | **−0.177** | −1.071 | −0.075 |
| jul | **−0.081** | −0.502 | −0.010 |
| nov-dec | **−0.226** | −1.299 | −0.097 |

The sign is negative in every fold, and July is the weakest fold, as it was for C1. July is also LIRF's
lowest-fill month. `[X]` The small non-LIRF gain probably comes from `arr_link_bs_abs` acting as inbound
lateness (turnaround pressure), not from a fill.

### 2.4 What does not work: a post-hoc mixture correction `[G]`

The correction was `oof + (p_arr − p_base)(ms − oof)/(1 − p_base)`, with no fitted parameter and p taken
from `[C]`. It moved LIRF RMSE by −0.62 overall, but +0.97 on jan, +4.32 on jul and −6.07 on forward. The
same move driven by the permuted p read +4.09. The signal is real, but multiplying a small probability change
by a 1,000–3,000 s lever is too noisy to ship as an overlay. **It has to go in as model columns, not as a
post-process.**

### 2.5 The unmatched lane: no help `[J]`

P1 recipe (c−) was re-run on today's frame. It trains on matched and unmatched rows, weights unmatched rows
×8, uses prefix and identity, and has no stake weight. Held-out AUC on the 1,488 LIRF unmatched rows:

| arm | AUC | logloss |
|---|---:|---:|
| (c−) as recorded | 0.9258 | 0.3559 |
| + arrival | 0.9267 | 0.3523 |
| + arrival, permuted | 0.9264 | 0.3546 |

That is within noise, so the P1 gate stays closed on this evidence.

---

## 3. LFPG: where its error is, and why nothing here fixes it

- **No fill.** Schedule-fill rows carry 0.40 % of LFPG's lane-A SSE, and their bias is −59 to −137 `[A]`.
- **Worst days** (ten days carry 22.0 % of LFPG SSE `[A]`): 2025-01-17, 11-22, 12-05, 01-21, 11-21 and so on.
  Median `ms` is 2,400–4,414 against a normal 1,676, and both clocks moved together (01-17: median `y`
  1,800, median `proxy` 1,679). Bias on those days is small (−61 to +50). This is a variance problem, not a
  missing offset.
- **The snow days are variance, not bias.** Reading 2025-11-22 `[inline]`: in the morning, with the METAR
  de-icing gate on, `g` is strongly positive. In the afternoon the gate is off at 0 °C and `g` turns
  strongly negative (−600 to −1,800; `BLOCK` *after* `AOBT_3`). Rows with no gate but frozen weather in the
  previous 12 h have LFPG bias **+9.0 s**, RMSE 341 against 241 overall `[inline]`. A snow-memory column
  would have nothing to shift.
- **The 300 worst under-predictions** (21.0 % of LFPG SSE `[F]`): median `BLOCK − LOBT` +176 s, against a
  median `AOBT_3 − LOBT` of +2,220 s. `stand_arrival_gap` is null on 24 % of them with median 8,278 s, so the
  stand was not reused and the v14 mechanism does not fire. AFR is 52 %. Hall K is over-represented (11.7 %
  against 6.8 % of LFPG rows). They are spread across all four OOF months (85 / 104 / 61 / 50). `[X]` These
  are either an early off-block with a later NM stamp, or a return to stand. I found no column that
  separates them.
- The **replanned-share shift** (23.9 % → 45.6 %) is a composition effect, as `delay-mechanisms.md` §6
  records. I did not find a mechanism behind it.

---

## 4. Things on the record that read differently now

1. **`delay-mechanisms.md` §2.2, LIRF slope 0.949:** the fill inflates it. On non-fill LIRF rows,
   `b(y ~ proxy)` is **0.743** and corr is **0.355** (all rows: 0.949 / 0.626). LFPG (0.630 / 0.570) and
   LTFM (0.134 / 0.108) do not change `[inline]`. With fills removed, LIRF sits with LTFM and EDDM, where the
   offset explains little. `[X]` At those airports `g` is closer to "taxi minus a nominal" than to a clock
   disagreement, so the brief's point that taxi mechanisms cancel out of `g` holds least there.
2. **"LIRF replanned rows = 11.5 % of lane-A SSE"** reproduces: 0.0295 + 0.0157 + 0.0325 + 0.0378 = 0.1155
   `[inline]`. But **5.35 points of it are fill rows**, 3.78 of them moved-plan fills. What costs is the plan
   moving off the schedule (`SCHED ≠ LOBT`) combined with a fill, not `lobt_minus_eobt1 ≠ 0`. Moved-plan
   fills with `lobt_minus_eobt1 == 0` add another 1.91 points.
3. **P1 stage 2, recipe (c−):** recorded at 0.9201, it now reproduces at **0.9258** on the current 140-column
   frame `[J]`. The frame has gained `stand_arrival_gap` and others since, and threads differ. The gate is
   still 0.93, so the verdict does not change.
4. **LTFM:** `EOBT_1 == LOBT` on **100 %** of lane-A rows `[inline]` (other airports 51–76 %), which is why
   `lobt_minus_eobt1` is identically zero there. `[X]` NM's first-filed EOBT at LTFM is the last one, so the
   column carries no planning history there. There is nothing to build from it.

**Side check, closed:** `MVT_ID_mvt` follows `SCHED` order, not `BLOCK` order. Consecutive-pair concordance
for LIRF departures in July 2025 is 0.994 with SCHED, 0.546 with BLOCK and 0.542 with MVT; EGLL is the same
`[inline]`. The ID carries no off-block information.

---

## 5. Ranked shortlist, with pre-registered effects

Conversions are the repo's own: board ≈ 0.57 × lane A, and the instrument sees 0.57 of a true lane-A gain.

### A1 — arrival-mirror fill state (the only candidate)

- **Mechanism.** At LIRF the recording process writes the schedule as the block time. When it does that for
  a flight's inbound arrival, the departure of the same rotation is 2.6× as likely to be filled
  (0.399 / 0.156). The arrival record is not blanked.
- **Columns.** `arr_link_fill`, `arr_link_bs_abs`, `arr_link_same_op`, `arr_opday_fill`, built from ARR rows
  of the same frame (train from training arrivals, rank from ranking arrivals). They join on the stand link
  that `stand_arrival_features` already computes. The builder is `build_arrival_fill` in the script, about
  40 lines.
- **Variant to run alongside.** The same columns nulled outside LIRF, because the signal is LIRF-only and
  the non-LIRF reading (−0.01 to −0.10 `[H]`) is unexplained.
- **Not already covered.** `[C]`: +0.0056 AUC over a base holding the operator, stand, prefix and every
  clock, in 12 of 12 months. The permuted control gives +0.0000.
- **Pre-registered effect.** Lane-A OOF **−0.10 to −0.30 s, point −0.15**. Board **−0.05 to −0.17 s, point
  −0.08**. Reasoning: `[D]` −0.071 ÷ 0.57 = −0.12 s of lane A. `[E]` −0.166 is uncalibrated, and ÷ 0.57 would
  be −0.29. The point estimate sits between them, weighted to the calibrated reading.
- **Kill conditions, fixed now.**
  - (a) The LIRF lane-A RMSE must fall in each of jan, jul and forward, and in jul_fwd.
  - (b) The pooled change must be negative on the paired bootstrap.
  - (c) An arity-matched permuted twin run must not read within 50 % of the arm.
  - jul is the fold most likely to fail. It read −0.08 `[H]`, the weakest, as C1's did. If jul_fwd reads ≥ 0,
    close it.
- **Risk stated up front.** This is the same size as C1 on the same instrument (−0.071 vs −0.070), and C1 did
  not transfer. What differs: its sign held in all three quarter folds `[H]` and in 12 of 12 months of an
  independent classifier `[C]`. C1 never had either check.

### Not recommended

- **Fill-probability mixture overlay** `[G]`: fold signs +/+/−, so do not ship it.
- **Arrival columns for the unmatched lane** `[J]`: +0.0009 AUC, within noise.
- **Snow-memory or de-icing persistence at LFPG:** the rows are unbiased (+9.0 s), so it would only add
  variance.
- **`MVT_ID` order:** it follows SCHED, not BLOCK.

### Where the rest of the error is, and why I found nothing for it

- The **late NM off-block** kind is 16.7 % of lane-A SSE (6,152 rows, `g > 1,200`, not a fill; LFPG 3.8 %,
  LIRF 2.9 %, LTFM 2.7 %). On those rows the aircraft left the stand near LOBT, and NM's `AOBT_3` sits
  25–45 min later at a nominal distance from take-off. It is defined by the label. I looked for a visible
  separator in:
  - the stand link, backward and via `stand_aobt_turn`;
  - arrival state;
  - weather memory;
  - operator, stand hall and month;
  - exact `proxy` values (LIRF fill rates by proxy minute range only 3–14 %).

  None separated it. It is what `delay-mechanisms.md` §3 calls the invisible time-local state, and at LIRF
  and LFPG it is concentrated in high-delay days.
- **Oracle ceiling, for scale:** knowing the LIRF moved-plan fills exactly is worth −6.15 s of lane A, and
  all LIRF fills −8.69 s `[H]`. A1 recovers about 2–5 % of that.
