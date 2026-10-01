# What LFPG's unmatched lane actually is (2026-09-24)

**Script (read-only, no training, ~90 s):**
`docs/competitor_research_2026-09-21/scripts/lfpg_unmatched_lane.py`, run as
`POLARS_MAX_THREADS=2 .venv/bin/python docs/competitor_research_2026-09-21/scripts/lfpg_unmatched_lane.py`.
Every number below is tagged with the section of that script (`S1`…`S8`) that prints it.

**Inputs.** `data/work/feat_train.parquet` (2,085,047 rows, all `PHASE_mvt == "DEP"`),
`data/work/feat_rank.parquet` (344,841), `experiments/oof_main_reparam-bc60017878.parquet`
(672,428 non-null `oof`) with its sidecar `..._folds.parquet`.

**Notation.** `y` = `TAXITIME_SEC_mvt`; `ms` = `mvt_minus_sched` = MVT − SCHED (`src/prc/features.py:62`);
`bs` = BLOCK − SCHED. Unmatched = the routing definition the code uses, `mvt_minus_aobt3` null
(`src/prc/train.py:40`, `:402`, `:421`). `y = MVT − BLOCK` holds exactly on all 2,085,047 rows [S],
so `bs = ms − y` identically — `bs` is a **label-side** quantity: `BLOCK_TIME_UTC_mvt` is blanked on
every departure in both frames (`src/prc/schema.py:52`, `src/prc/features.py:46-50`) and is null on
344,841 / 344,841 ranking rows.

---

## 0. The answer in four sentences

1. **LFPG's unmatched RMSE of 2,623.0 is two rows.** They carry **94.8 %** of the lane's OOF squared
   error; with them removed the lane measures **600.1**, which puts LFPG third, behind EGLL (998.4)
   and EDDM (717.4), and in the same band as LSZH (585.4). [S] `S1`, `S2`
2. **Both rows are already-catalogued label corruption, not LFPG-specific structure.** They are the
   LFPG entries of `docs/investigation_2026-09-16/results/block-day-rollover.md` — `BLOCK_TIME` on
   the *previous* calendar day. They are invisible at prediction time. [S] `S1`, and `block-day-rollover.md:63, :109`
3. **The population itself has a clean, previously unrecorded mechanism: the unmatched lane is a
   delay-selected population with an exact hard edge at one hour.** `P(unmatched)` at LFPG jumps
   from 0.0058 to 0.2224 as `bs` crosses 3,600 s — a 38.5× step at an exact boundary — and the
   matched side shows the clamp that causes it: `BLOCK − LOBT` has **maximum 3,606 s over all
   2,062,577 matched rows**. A flight that goes off-block more than an hour after its last filed
   off-block time loses its NM record, and lands in model B. [S] `S3`, `S4`
4. **There is no lever here.** The bound that the mechanism implies (`y ≤ ms − 3,600`) is worth
   **−0.018 s** applied to all non-LIRF unmatched rows, the two big rows are unpredictable, and the
   remaining 600.1 is variance, not bias: model B has no off-block clock at all, and at equal `ms`
   it measures 899.6 where model A measures 527.1. [S] `S7`, `S5`

---

## 1. Where the squared error actually sits (task item 2)

`S2`, unmatched rows of the pooled OOF, per airport:

| ADEP | n | RMSE | top-1 share | top-10 share | top-100 share | drop 1 | drop 3 | drop 10 | drop 50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **LFPG** | 1,565 | **2,623.0** | **64.3 %** | 96.8 % | 98.7 % | **1,567** | **567** | 469 | 358 |
| LIRF | 537 | 4,232.2 | 9.3 % | 46.0 % | 82.2 % | 4,034 | 3,650 | 3,139 | 2,421 |
| EGLL | 647 | 998.4 | 13.8 % | 59.5 % | 93.6 % | 928 | 827 | 641 | 345 |
| EDDM | 402 | 717.4 | 15.9 % | 56.5 % | 93.6 % | 659 | 578 | 479 | 301 |
| EHAM | 1,353 | 468.5 | 39.0 % | 53.8 % | 84.5 % | 366 | 352 | 320 | 236 |

**LFPG and LIRF have opposite shapes.** LIRF is a broad regime — dropping its worst 50 rows still
leaves 2,421. LFPG is one row: 64.3 % of the lane's SSE is a single observation, 96.8 % is ten.

`S1`, the same lanes with `y ≥ 20,000 s` removed:

| ADEP | n | RMSE all | n excl | RMSE excl | rows dropped |
|---|---:|---:|---:|---:|---:|
| **LFPG** | 1,565 | 2,623.0 | 1,563 | **600.1** | **2** |
| LIRF | 537 | 4,232.2 | 506 | 3,437.2 | 31 |
| EGLL | 647 | 998.4 | 647 | 998.4 | 0 |
| EDDM | 402 | 717.4 | 402 | 717.4 | 0 |
| every other airport | — | — | — | unchanged | 0 |

Query: `u.group_by("ADEP_mvt")` on the OOF-joined frame, RMSE over `resid = oof − y`, once on all
rows and once on `y < 20000`.

### 1.1 The two rows

`S1` / the OOF-joined frame filtered to `ADEP_mvt == "LFPG" & unm & y >= 20000`:

| `MVT_ID_mvt` | FLIGHT | fold | SCHED (UTC) | BLOCK (UTC) | MVT (UTC) | y | ms | pred | resid | row SSE |
|---:|---|---|---|---|---|---:|---:|---:|---:|---:|
| 182378789 | EJU983W | jan | 2025-01-19 18:25:00 | **2025-01-18** 19:30:00 | 2025-01-19 18:54:00 | 84,240 | 1,740 | 1,025.4 | −83,214.6 | 6.925e9 |
| 182377662 | EJU42AY | jan | 2025-01-23 14:35:00 | **2025-01-22** 22:58:57 | 2025-01-23 15:09:03 | 58,206 | 2,043 | 935.6 | −57,270.4 | 3.280e9 |

Together 1.020e10 of the lane's 1.077e10 (94.8 %). Both are in
`docs/investigation_2026-09-16/results/block-day-rollover.md` already: EJU983W is its row 16
(`:63`), EJU42AY is the single non-LIRF row in its 12 h ≤ y ≤ 20 h list (`:109`). In both the BLOCK
date is one calendar day before the MVT date while SCHED and MVT agree, so the *label* is wrong, not
the flight.

**[X] They cannot be caught at prediction time and it is not close.** Everything visible about them
is ordinary: `ms` 1,740 and 2,043 against a lane median of 5,823; stand C03 / D06; runway 08L / 26R;
A319 / A320; `AIRCRAFT_TYPE_mvt` present. The only column that betrays them is `BLOCK_TIME_UTC_mvt`,
which is null on all 344,841 scored rows. Any rule that would have fired on them would have to fire
on the ~1,500 lane rows that look identical.

---

## 2. The mechanism: an exact one-hour selection edge (task item 4)

This is the new finding about the population, and it is not LFPG-specific — LFPG is simply where it
is sharpest and where it matters most.

### 2.1 The edge

`S3`, `P(unmatched | bs)` measured on all departures at each airport, in the two 300-s bins either
side of 3,600:

| ADEP | n in [3300,3600) | P below | n in [3600,3900) | P above | step | share of that airport's unmatched rows with `bs ≥ 3600` |
|---|---:|---:|---:|---:|---:|---:|
| LEBL | 1,724 | 0.0006 | 1,482 | 0.1538 | 265× | 0.9076 |
| EDDM | 1,090 | 0.0018 | 869 | 0.1657 | 90× | 0.9455 |
| EGLL | 1,619 | 0.0019 | 1,330 | 0.1511 | 82× | 0.9639 |
| **LFPG** | 2,597 | 0.0058 | 2,221 | **0.2224** | **38.5×** | **0.8834** |
| EHAM | 2,500 | 0.0076 | 2,050 | 0.1649 | 22× | 0.5424 |
| EDDF | 1,983 | 0.0076 | 1,597 | 0.1634 | 22× | 0.8809 |
| LSZH | 1,213 | 0.0115 | 941 | 0.1796 | 16× | 0.4465 |
| LTFM | 1,536 | 0.0215 | 1,260 | 0.0611 | 2.8× | 0.1654 |
| LEMD | 1,834 | 0.0000 | 1,485 | 0.0902 | ∞ (0 below) | 0.9820 |
| LIRF | 1,417 | 0.0000 | 1,089 | 0.0808 | ∞ (0 below) | 0.4503 |

Pooled at 10-s resolution the transition sits between 3,600 and 3,620 (`S3`): `[3590,3600)` P=0.0115,
`[3600,3610)` P=0.0193, `[3610,3620)` P=0.1588. SCHED has second-of-minute 0 on 100 % of rows and
BLOCK is within 6 s of a minute boundary on 100 % of LFPG and LIRF unmatched rows, so "one hour at
minute resolution" is the natural reading of a 3,600–3,610 edge. **[X]**

Query (per airport): `bs = (BLOCK − SCHED).total_seconds()`, then the mean of the
`mvt_minus_aobt3.is_null()` flag inside each bs bin.

### 2.2 The clamp that causes it

`S4`, on the 2,062,577 **matched** rows (where the NM columns exist):

| quantity | p1 | p50 | p99 | p99.9 | **max** | frac > 3,600 | frac > 3,660 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `BLOCK − LOBT` | −838 | 251 | 2,623 | 3,474 | **3,606** | 0.0001 | **0.0000** |
| `AOBT_3 − LOBT` | −1,080 | 240 | 2,760 | 3,840 | 20,421 | 0.0017 | 0.0015 |
| `LOBT − SCHED` | −600 | 0 | 7,200 | 21,000 | 268,800 | 0.0320 | 0.0318 |

**No matched row anywhere in the training data has an actual off-block more than 3,606 s after its
last filed off-block time.** That is a hard ceiling, not a tail. **[X]** The reading is the
Eurocontrol NM behaviour: a flight plan that has not departed by LOBT + 1 h is suspended and drops
out of the flight table, so the movement record has nothing to join to. `LOBT − SCHED` has median 0,
which is why the ceiling shows up as an edge in `BLOCK − SCHED` too, smeared by the 3.2 % of rows
whose LOBT was re-filed more than an hour after schedule.

**So: the unmatched lane is, to first order, the set of departures that went off-block at least an
hour late.** At LFPG that is 88.3 % of it; at LTFM only 16.5 %, which is why the lanes do not look
alike across airports.

### 2.3 What that predicts, and it holds

If the lane is delay-selected, its size should track disruption. `S6` / the daily table at LFPG:
the unmatched rate has median 1.08 % over 365 days and reaches **15.67 % on 2025-12-05** and
13.68 % on 2025-11-22. The lane is also summer-heavy in a way the matched lane is not (monthly
unmatched counts 216/123/106/152/265/454/**664**/403/389/316/372/313 against a flat 17k–21k matched).

### 2.4 The other mechanisms this is *not*

* **Not LIRF's fill.** `|BLOCK − SCHED| ≤ 10 s` on 0.0034 of LFPG unmatched rows against 0.4845 at
  LIRF; `corr(y, ms)` +0.0205 against +0.8964; `y == ms` (±1 s) on 4 rows of 3,773 against 193 of
  1,488. Exact equalities BLOCK == SCHED: 3 rows. Confirms the brief's figures.
* **Not an information blackout.** On LIRF unmatched rows `AIRCRAFT_TYPE_mvt` is null on 99.93 %.
  On **LFPG** unmatched rows it is null on **0.13 %**, `STAND_mvt` on 0.08 %, `RUNWAY_mvt` on 0.00 %,
  `ADES_mvt` on 0.05 %. The four `_flt` categoricals (`WK_TBL_CAT`, `AIRCRAFT_OPERATOR`,
  `MARKET_SEGMENT`, `FLIGHT_TYPE`) are 100 % null in every airport's unmatched lane — that is
  definitional, they come from the NM record. **LFPG's unmatched rows look like ordinary LFPG
  departures minus the flight plan.**
* **Not a rounding or discreteness artefact.** `SCHED` second == 0 on 100 % of rows, `MVT` second
  == 0 on 8.1 %, `BLOCK` second == 0 on 8.8 % — identical between LFPG and LIRF, so nothing
  discriminates.

---

## 3. Residual behaviour once the two rows are set aside (task item 3)

`S5`, LFPG, `y < 20,000`, matched lane against unmatched lane in the same `ms` band:

| `ms` band | unm n | unm sd(y) | unm bias | unm RMSE | unm SSE share | mat n | mat sd(y) | mat RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| < 1,800 | 42 | 395 | +161.0 | 354.8 | 0.009 | 40,636 | 254 | 161.3 |
| 1,800–3,600 | 72 | 443 | +9.2 | 410.8 | 0.022 | 25,001 | 446 | 260.2 |
| 3,600–5,400 | 384 | 410 | +66.2 | 396.6 | 0.107 | 6,702 | 672 | 377.0 |
| 5,400–7,200 | 599 | 517 | −52.9 | 416.1 | 0.184 | 2,003 | 856 | 439.0 |
| **7,200–10,800** | **339** | 1,017 | −67.4 | **899.6** | **0.487** | 1,386 | 999 | **527.1** |
| 10,800–14,400 | 92 | 1,352 | −324.4 | 1,033.6 | 0.175 | 441 | 912 | 492.1 |
| ≥ 14,400 | 35 | 550 | +84.0 | 499.6 | 0.016 | 302 | 979 | 376.3 |
| **whole lane** | **1,563** | 751 | **−38.6** | **600.1** | 1.000 | 76,471 | — | **245.9** |

Three things follow. **[X]**

* **The lane is not biased.** Overall bias −38.6 s on 1,563 rows. Removing it entirely would recover
  `1563 × 38.6² ≈ 2.3e6` of 5.63e8 SSE — 0.4 %, about 0.006 s of OOF RMSE. There is no
  over-prediction of a bulk and no systematic under-prediction of a tail; the −324 in the
  10,800–14,400 band is 92 rows and worth `92 × 324² ≈ 9.7e6`, 0.026 s.
* **Two thirds of the remaining SSE (0.487 + 0.175) is the 431 rows with `ms ≥ 7,200`,** and `ms`
  is a feature model B already holds (`src/prc/features.py:62`, `:63`).
* **The gap is the missing clock, not the airport.** At `ms` 7,200–10,800 the two lanes have the
  same label spread (sd 1,017 vs 999) and the matched lane is 1.7× more accurate, because model A is
  handed `MVT − AOBT_3` as a proxy (`src/prc/train.py:405-419`) while model B has no off-block
  observation of any kind.

### 3.1 The non-rollover error is day-clustered, and the days are not lane-specific

`S6`. LFPG unmatched non-rollover SSE by day, top 5 of 122 OOF days = **56.5 %**, top 10 = 68.9 %:

| day | unm n | unm SSE | share | unm RMSE | **matched-lane RMSE that day** |
|---|---:|---:|---:|---:|---:|
| 2025-11-22 | 77 | 1.040e8 | 0.185 | 1,162 | **583** (the worst matched day of the year) |
| 2025-12-05 | 100 | 8.692e7 | 0.154 | 932 | **502** (the second worst) |
| 2025-12-26 | 6 | 6.083e7 | 0.108 | 3,184 | 238 |
| 2025-07-16 | 12 | 4.151e7 | 0.074 | 1,860 | 251 |
| 2025-01-13 | 8 | 2.465e7 | 0.044 | 1,755 | 243 |

LFPG matched-lane RMSE is 245.9 overall, p50 223 / p90 294 per day. **The two days that carry a
third of the unmatched lane's non-rollover error are the two days the matched lane is worst too.**
That is an airport-wide disruption signature, not an unmatched-lane defect. The remaining days in
the list are single flights (2025-12-26 is 6 rows, of which CES570 to ZSPD alone is 6.0e7).

---

## 4. Is it one thing or several? (task item 5)

**Three populations, of which only two matter.**

`S5`, LFPG unmatched, split on the edge:

| group | n | RMSE | SSE | bias | y median |
|---|---:|---:|---:|---:|---:|
| ALL | 1,565 | 2,623.0 | 1.077e10 | −120.8 | 1,087 |
| `bs ≥ 3600` — the late class | 1,426 (91.1 %) | **501.5** | 3.587e8 | −14.5 | 1,084 |
| `bs < 3600` | 139 | 8,653.5 | 1.041e10 | −1,211.3 | 1,199 |
| `bs < 3600`, excl the 2 rollovers | 137 | 1,221.0 | 2.042e8 | −203.6 | 1,195 |

* **(a) The late class** — 91 % of the lane, RMSE 501.5, unbiased. Ordinary flights that pushed back
  an hour or more late. Nothing anomalous about them except that they have no flight plan.
* **(b) The on-time unmatched class** — 139 OOF rows / 440 training rows. This is where both
  corrupted labels sit. Its largest identifiable component is one recurring night freighter:
  `FLIGHT_mvt == "FPO7SN"`, LFPG → EINN at 03:00 daily, **200 of the 3,773 LFPG unmatched training
  rows** (and 200 of 347 LFPG→EINN departures overall, an unmatched rate of 57.6 % against the
  airport's 1.58 %). Its predictions are fine — RMSE 274.5 on the 71 OOF rows whose `FLIGHT_mvt`
  starts with `FPO`, bias −9.0 — so it is a composition fact, not an error source. The rest of the
  group is TAY 28 / FDX 26 / AFR 77 / DAL 10 / UPS, i.e. mostly night cargo.
* **(c) Two corrupted labels**, §1.1.

### 4.1 The crux: nothing here separates large errors from small ones at prediction time

The one prediction-time split that looked promising fails cleanly. `ms < 3600` **implies** `bs < 3600`
(because `bs = ms − y` and `y > 0` on all but 3 of the 1,565 OOF rows; verified directly — every `ms < 3600` row does have `bs < 3600`), so it is a visible necessary condition for group (b). `S5`:

| LFPG unmatched subset | n | RMSE |
|---|---:|---:|
| `ms < 3600` (visible) | 116 | 9,387.3 |
| `ms < 3600`, excl the 2 rollovers | 114 | **391.1** |
| `ms ≥ 3600`, excl the 2 rollovers | 1,449 | **613.6** |

The whole apparent signal is the two rollover rows happening to land there. Once they are removed,
`ms < 3600` is the **best-behaved** part of the lane, not the worst. **There is no prediction-time
handle on group (b), and the 2,623 is group (c), which has none either.**

Per airport (`S7`, violation column), the fraction of unmatched rows where the truth violates
`y ≤ ms − 3,600` is LEMD 0.007, EGLL 0.034, EDDM 0.037, LEBL 0.067, **LFPG 0.089**, EDDF 0.102,
EHAM 0.426, LIRF 0.497, LSZH 0.528, LTFM 0.823 — so even the mechanism's own bound is airport-
dependent and does not hold where LIRF's fill or LTFM's no-flight-plan population dominates.

### 4.2 One candidate the evidence does point at, stated as a candidate

**[X]** The lane's *density* is a disruption indicator and it is computable at prediction time on
the scored frame (routing is known for every row before any label is). `S6`, ranking airport-days by
their unmatched rate over the whole OOF:

| rows covered | share of total SSE | lift | rate cut |
|---|---:|---:|---:|
| top 1 % | 3.8 % | **3.77×** | 0.0908 |
| top 5 % | 17.2 % | 3.44× | 0.0365 |
| top 10 % | 25.8 % | 2.58× | 0.0233 |

(excluding the two LFPG rollovers, so the lift is not an artefact of them.) This is an
**airport-wide** candidate feature — "share of departures in the last N minutes with no NM record" —
not an unmatched-lane one, and it competes with `congestion_vs_usual_30m` and the rest of the
existing context block, which may already carry it. **Not measured. Do not describe it as a lever
until an ablation says so.**

---

## 5. What a bound built from the mechanism is worth (measured, and it is nothing)

`S7`. Rule: `pred := min(pred, ms − 3600)` on unmatched rows where `ms − 3600 > 0`. This is exactly
the model-B analogue of the existing LOBT bound (`src/prc/train.py:453-484`), which is inert on
unmatched rows because `mvt_minus_lobt` is 100 % null there.

| scope | rows moved | helped / hurt | ΔSSE | ΔRMSE (OOF) |
|---|---:|---:|---:|---:|
| all unmatched | 753 | 478 / 275 | +1.388e9 | **+3.6396** |
| non-LIRF | 333 | 266 / 67 | −6.887e6 | **−0.0182** |
| EGLL LSZH EHAM LEBL EDDM LEMD (chosen **after** seeing the signs) | 181 | 164 / 17 | −4.748e7 | −0.1253 |
| LFPG alone | 77 | 61 / 16 | **+3.010e6** | (positive — it hurts) |

**LFPG is one of the two airports where the bound is positive**, alongside EDDF (+1.49e6) and next
to LTFM (+3.61e7) and LIRF (+1.395e9). That reproduces, from a different direction, the observation
in `P1-stage1-cap-CLOSED.md` that LFPG resists caps: 8.9 % of its unmatched rows genuinely sit above
the bound, and its long predictions are more often right than wrong. The −0.1253 row is
post-hoc airport selection on the measured sign and should not be treated as an effect.

---

## 6. What is at stake, honestly (task item 6)

Exchange rate on the board: `N = 344,841`, `RMSE = 282.7371`, so 1 s of RMSE = `344841 × (2×282.7371 + 1)`
= **1.9534e8** of squared error. On the OOF (`N = 672,428`, RMSE 281.8411) 1 s = 3.797e8.

`S8`:

| counterfactual | OOF RMSE | Δ |
|---|---:|---:|
| as shipped | 281.8411 | — |
| perfect on all 1,565 LFPG unmatched rows | 251.8363 | **−30.00** |
| perfect on the 2 rollover rows only | 253.4930 | **−28.35** |
| perfect on the 1,563 non-rollover LFPG unmatched rows | 280.3520 | **−1.49** |
| cut 50 % of the non-rollover SSE | 281.0975 | −0.74 |
| cut 10 % of the non-rollover SSE | 281.6925 | −0.15 |

**On the board.** `feat_rank.parquet` has **734 LFPG unmatched rows** of 5,290 unmatched and 344,841
total (`S8`; LFPG is the second-largest unmatched population after EHAM's 1,206). Their composition
matches training: `ms ≥ 3600` on 94.4 % (90.1 % in training), only 41 rows with `ms < 3600`,
`AIRCRAFT_TYPE_mvt` null on 0.

* A **perfect** model of those 734 rows, at the OOF non-rollover error rate of 3.6015e5 SSE/row, is
  worth **1.35 s**. That is the whole honest prize, and it requires zero error on 734 rows we
  currently predict at RMSE 600.
* A **realistic** partial improvement — say 10 % of their SSE, which is what removing all remaining
  bias plus some of the `ms ≥ 7,200` variance would look like — is worth **0.14 s**.
* The **rollover lottery** is the real money and cannot be won. The LFPG unmatched rollover rate in
  training is 2 / 3,773 = 5.3e-4, so 0.39 such rows are expected among the 734, at ~5.1e9 SSE each:
  **≈ 10.2 s of board RMSE in expectation, concentrated in a coin flip.** Whether v13's 282.7371
  contains one of these is not knowable from here; if it does, roughly 13 s of the score is one row,
  and if it does not, a future submission may pick one up. Nothing in the feature set can change
  that — it is the score's variance, not its bias.

**[X] Conclusion for the candidate list: close it.** LFPG's unmatched lane is a delay-selected
population whose error is (i) two corrupted labels nobody can see coming, (ii) two airport-wide
disruption days that hurt the matched lane just as badly, and (iii) 431 high-`ms` rows where model B
is doing about as well as an off-block-blind model can. The one-hour edge in §2 is a genuine and
previously unrecorded fact about the data — it explains *why* the lane exists and why it differs so
much between airports — but every rule derivable from it measures at or below noise. The only
unspent idea it suggests (§4.2) is an airport-wide feature, not an LFPG one, and it is untested.
