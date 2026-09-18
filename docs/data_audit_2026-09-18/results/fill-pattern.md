# Lens: the sched-filled `BLOCK_TIME` phenomenon, characterised everywhere it occurs

Script: `docs/data_audit_2026-09-18/scripts/fill-pattern.py` (no training; group means, point-biserial
correlations and rank statistics only; `POLARS_MAX_THREADS=1`; 2 min 32 s wall; sections A–K).
Inputs, read-only: `data/work/feat_train.parquet` (2,085,047 departures), `data/work/feat_rank.parquet`
(344,841), `experiments/oof_main-48c19bc830.parquet` (2,085,047 rows, `oof` non-null on 672,428).
Every number below was produced by this script. Numbers carried from the 2026-09-16 investigation are
marked as such and are not re-derived.

Definitions (`docs/investigation_2026-09-16/plan.md` §0 unless stated):
`y` = `TAXITIME_SEC_mvt`; `ms` = `mvt_minus_sched`; `proxy` = MVT − AOBT_3; `resid` = y − proxy
(= AOBT_3 − BLOCK); matched = `AOBT_3` non-null (routing definition); `fill` / `f10` = |BLOCK − SCHED| ≤ 10 s;
`far` = |AOBT_3 − SCHED| > 600 s. **New in this file:** `dfill` = `fill` ∧ |BLOCK − AOBT_3| > 600 s
("distorted fill"), and `nosupd` = (SCHED == LOBT) ∧ (SCHED == EOBT_1) ("no schedule update").

---

## 1. What was checked

1. **Census.** Fill rate at three thresholds (±6 s, ±10 s, ±120 s) by ADEP × matched, and by each of
   month, operator, market segment, hour, stand, runway, destination, wake category, flight type, in
   training. Same marginals for the ranking set wherever the driving column survives. (§A–C, §G)
2. **The 12-month stability** of each airport × matched series, and of the LIRF unmatched series. (§B2–B5)
3. **Whether the non-LIRF filled rows are the same phenomenon**: for every filled matched row, is
   |BLOCK − AOBT_3| small (the flight really did push back on its scheduled minute) or large (BLOCK
   displaced from the real off-block)? Per airport, with the direction and size of the resulting label
   displacement and its label tail. (§D)
4. **Separability of the fill on MATCHED rows** — the 28,369 LIRF and 70,556 other-airport rows that were
   never modelled. In-sample between-group R² and honest leave-one-month-out (LOMO) smoothed group-mean
   R² and AUC, over 18 groupings of ranking-visible columns, plus point-biserial correlations of 12
   continuous candidates. (§E, §J5)
5. **Airport-free signals**: the EOBT_1 / LOBT / IOBT equality structure, and the seconds-of-minute
   pattern of BLOCK / MVT / SCHED. Per airport, per `far` stratum, on fill and on `dfill` separately. (§F, §J2–J4)
6. **Unmatched rows**: which columns survive at all, and how separable the fill is from those only
   (flight-designator prefix, stand, runway, hour, `ms` band). (§I)
7. **Ranking set**: null structure of every driver; composition Jan/Jul 2026 against training; expected
   filled row counts per airport × month under training cell rates. (§G, §K)
8. **Cost**: squared-error share of filled rows on `oof_main-48c19bc830`, split matched/unmatched and by
   airport, with oracle-exact replacements. (§H)

---

## 2. Findings

### 2.1 The census: 100,049 filled rows, four distinct populations (§A, §B1)

| population | rows | fill rate | identifiable at prediction time? |
|---|---|---|---|
| LIRF unmatched | 721 of 1,488 | **0.4845** | fill state: **no**; the rows: yes (ADEP + `mvt_minus_aobt3` null) |
| LIRF matched | 28,369 of 159,216 | **0.1782** | fill state: **no**; the rows: yes |
| non-LIRF matched | **70,556** of 1,903,361 | **0.0371** | fill state: **no**; the rows: yes |
| non-LIRF unmatched | 403 of 20,982 | **0.0192** | fill state: **no**; the rows: yes |
| all training departures | 100,049 of 2,085,047 | 0.0480 | |

Query: `abs((BLOCK_TIME_UTC_mvt - SCHED_TIME_UTC_mvt).dt.total_seconds()) <= 10`, grouped by
`ADEP_mvt` × `mvt_minus_aobt3.is_null()`. §B1 also carries f6 and f120 per cell.

Re-verified identity (§A): `Σ |(y − ms) + (BLOCK − SCHED)| = 0` over all 2,085,047 rows, so "filled" and
"y = ms" remain the same predicate at the same tolerance.

### 2.2 The rate is stable over the twelve training months at every airport, for matched rows (§B2)

Matched f10 per airport × month, min–max over the 12 months: EDDF 0.0086–0.0157, EDDM 0.0389–0.0660,
EGLL 0.0410–0.0565, EHAM 0.0072–0.0165, LEBL 0.0439–0.0811, LEMD 0.0354–0.0549, LFPG 0.0313–0.0581,
**LIRF 0.1453–0.2058**, LSZH 0.0069–0.0173, LTFM 0.0455–0.0561. No month is an outlier at any airport; the
within-airport max/min ratio is 1.2–2.3 and the airport ordering is identical in all 12 months.
The same holds for `dfill` (§B3): LIRF 0.0289–0.0540, LTFM 0.0057–0.0118, LEBL 0.0011–0.0184, others ≤ 0.008.

**The one unstable series is LIRF unmatched** (§B4): fill rate by month 0.4167 / 0.7586 / 0.7288 / 0.6714 /
0.4646 / 0.4865 / **0.3234 (Jul)** / 0.4518 / 0.4345 / 0.4870 / 0.6154 / **0.7614 (Dec)** on n = 60 / 58 / 59 /
70 / 99 / 185 / 337 / 197 / 168 / 115 / 52 / 88. The *rate* moves 2.4×, the *count* of filled rows only
25–109: the months with many unmatched rows (Jun–Oct) dilute the rate, the low-volume months (Feb, Mar,
Dec) are nearly all filled. Non-LIRF unmatched is flat at 0.0122–0.0296 (§B5).

Identifiable at prediction time: the *month* is, the fill state is not.

### 2.3 The fill splits into a benign and a distorting half, and only the distorting half matters (§D1–D4, §J1)

For a filled row, BLOCK ≈ SCHED. Whether that is a *fill* or an on-time pushback is decided by the real
off-block time: `dfill` = fill ∧ |BLOCK − AOBT_3| > 600 s. `dfill` and `fill ∧ far` are the same rows to
within a 600 ± 10 s boundary band (§J1: agree on 11,149 rows, differ on 998 of 2,062,577); the LIRF
`fill ∧ far` count implied here is consistent with the 6,453 in `lirf-matched-fill.md` §3.3.

| ADEP | matched n | filled | of which `dfill` | `dfill`/fill | `dfill` rate | median `resid` on `dfill` | y ≥ 2 h on `dfill` |
|---|---|---|---|---|---|---|---|
| LIRF | 159,216 | 28,369 | **6,823** | 0.2405 | 0.0429 | **+1,078** | 0.0045 (31 rows) |
| LTFM | 269,320 | 13,864 | 2,047 | 0.1476 | 0.0076 | **−662** | 0 |
| LEBL | 177,843 | 10,051 | 1,120 | 0.1114 | 0.0063 | +957 | 0 |
| LFPG | 235,779 | 9,870 | 665 | 0.0674 | 0.0028 | **−657** | 0 |
| EDDM | 166,324 | 8,341 | 562 | 0.0674 | 0.0034 | +844 | 0 |
| EGLL | 238,132 | 11,664 | 446 | 0.0382 | 0.0019 | +660 | 0.0022 (1 row) |
| LEMD | 211,295 | 9,468 | 374 | 0.0395 | 0.0018 | +841 | 0 |
| LSZH | 132,616 | 1,479 | 41 | 0.0277 | 0.0003 | +713 | 0 |
| EDDF | 228,185 | 2,826 | 38 | 0.0134 | 0.0002 | +843 | 0 |
| EHAM | 243,867 | 2,993 | 30 | 0.0100 | 0.0001 | **−655** | 0 |

On the other 86,779 filled matched rows (fill ∧ ¬`dfill`), |BLOCK − AOBT_3| has median 118–356 s per
airport (§D1) — the ordinary block-to-pushback offset. Per-airport fill rate by |AOBT_3 − SCHED| band
(§D6) shows why: at every airport except LIRF the rate falls to ≤ 0.02 beyond 600 s and to 0.000–0.005
beyond 3,600 s, whereas LIRF is 0.372 / 0.296 / 0.183 / 0.098 / 0.069 / 0.050 / 0.006 across the seven bands.

Query: `f10 & (abs((BLOCK_TIME_UTC_mvt - AOBT_3_flt).dt.total_seconds()) > 600)` on matched rows.
Identifiable at prediction time: **no** — both BLOCK and AOBT_3 are null on all 344,841 ranking rows
(§G1: `mvt_minus_aobt3` null on 5,290 unmatched rows and `AOBT_3_flt` itself unavailable as a label-time
column; `BLOCK_TIME_UTC_mvt` / `TAXITIME_SEC_mvt` null on 344,841, carried from `sched-filled-block.md` §K).

### 2.4 The 70,556 non-LIRF matched filled rows: same predicate, different and mostly harmless effect (§D)

- **92.5 %** of them (65,233) are the benign on-time component; only **5,323** are `dfill`.
- Concentration: LTFM 13,864, EGLL 11,664, LEBL 10,051, LFPG 9,870, LEMD 9,468, EDDM 8,341, EHAM 2,993,
  EDDF 2,826, LSZH 1,479.
- **The direction of the displacement is not the same as at LIRF.** At LTFM (2,047 rows), LFPG (665) and
  EHAM (30) the median `resid` on `dfill` rows is **negative** (−655 to −662), i.e. AOBT_3 precedes BLOCK
  and the label is *shorter* than the real taxi; at LIRF, LEBL, EDDM, LEMD, EGLL, LSZH, EDDF it is
  positive (+660 to +1,078), i.e. gate delay folded into the label. (The LTFM/LFPG sign matches the
  AOBT_3-earlier / BLOCK-earlier split reported in `matched-proxy-outliers.md` §3.1.)
- **No label tail follows.** Of the 5,323 non-LIRF `dfill` rows, exactly **1** has y ≥ 2 h (EGLL). y medians
  on the filled rows are 704–1,318, indistinguishable from the not-filled control (712–1,319, §D2).
- Operators: no non-LIRF operator has a high `dfill` rate. The top 20 by `dfill` among operators with
  n ≥ 30 (§D5) top out at **0.0364** (n = 110, LEBL, Non-Scheduled); most are Business / Non-Scheduled
  operators at LSZH and LTFM. Contrast LIRF, where nine operators exceed 0.06 and one reaches 0.133 (§C3).

### 2.5 Drivers, marginal by marginal (§C)

Rates are for matched rows unless stated.

- **Segment** (all airports / LIRF): Lowcost 0.0697 / 0.2219, Non-Scheduled 0.0571 / 0.4524,
  Cargo 0.0489 / 0.4685, Other 0.0450 / 0.2104, Mainline 0.0443 / 0.1364, Military 0.0379 / 0.700 (n 10),
  Regional 0.0331 / 0.1554, Business 0.0297 / 0.5294 (n 51).
- **Hour** (§C2): LIRF 0.137 (16 h) – 0.370 (1 h), other airports 0.0295 (12 h) – 0.0608 (4 h); both peak
  at night. `dfill` at other airports 0.0018 (19 h) – 0.0060 (0 h).
- **Operator** (§C3–C4): 11 operators (8,508 matched rows) have fill ≥ 0.50, 19 more (5,702 rows) in
  0.20–0.50, 122 (725,438 rows) in 0.05–0.20, 524 (1,322,929 rows) below 0.05. At LIRF, rows in operators
  with fill ≥ 0.80 are 2,717 and are 92.4 % filled; rows in operators below 0.02 are 1,276 and 1.65 % filled (§E4).
- **Stand** (§C5): a real driver only at LIRF. LIRF stands 235 / 238 / 236 / 237 / 108 / 234 are 0.494 /
  0.491 / 0.490 / 0.476 / 0.460 / 0.451 (n 322–1,367) while 405 / 403 / 404 / 407 / 409 are 0.106 / 0.098 /
  0.093 / 0.086 / 0.079 (n 2,599–2,807). At LEBL the range over stands with n ≥ 300 is 0.030–0.141, at
  LTFM 0.020–0.103, at EDDM 0.016–0.086.
- **Runway** (§C6): not a driver anywhere. LIRF 25 / 34L / 16R / 16L = 0.182 / 0.166 / 0.141 / 0.124;
  LTFM spans 0.048–0.076 over 9 runways; LEBL 0.049–0.060; EDDM 0.043–0.058.
- **Destination / wake category / flight type** (§E1): between-group R² 0.018 / 0.0004 / 0.0001 over all
  matched rows (at LIRF `ades` reaches 0.096, which tracks operator).

All of these columns are non-null in the ranking set (§G1: `STAND_mvt` 4 nulls of 344,841; `RUNWAY_mvt` 0;
`AIRCRAFT_OPERATOR_flt` and `MARKET_SEGMENT_flt` null exactly on the 5,290 unmatched rows).

### 2.6 A new, airport-free-in-direction signal: SCHED == LOBT == EOBT_1 (§F1–F3, §J3–J5)

`nosupd` = (SCHED == LOBT) ∧ (SCHED == EOBT_1) — no revision between the initial off-block estimate, the
latest, and the schedule. It lifts the fill rate at **all ten airports and in both `far` strata**:

| ADEP | ¬far ∧ nosupd | ¬far ∧ ¬nosupd | far ∧ nosupd | far ∧ ¬nosupd |
|---|---|---|---|---|
| LIRF | **0.3461** | 0.1491 | 0.1633 | 0.0658 |
| LEBL | 0.1067 | 0.0565 | 0.0268 | 0.0069 |
| EDDM | 0.0903 | 0.0544 | 0.0198 | 0.0017 |
| LFPG | 0.0908 | 0.0531 | 0.0080 | 0.0024 |
| LEMD | 0.0895 | 0.0451 | 0.0074 | 0.0023 |
| EGLL | 0.0849 | 0.0554 | 0.0089 | 0.0030 |
| LTFM | 0.0842 | 0.0679 | 0.0245 | 0.0069 |
| LSZH | 0.0252 | 0.0156 | 0.0009 | 0.0002 |
| EHAM | 0.0235 | 0.0159 | 0.0004 | 0.0002 |
| EDDF | 0.0233 | 0.0200 | 0.0004 | 0.0003 |

Cell sizes in §J4 (smallest cell 12,407). Pooled over all matched rows (§F2): `nosupd` true 1,112,736 rows
fill 0.0677, SCHED==LOBT only 413,752 rows 0.0383, neither 492,394 rows 0.0140.

**But the same flag does not predict the distorting half.** §F2's `dfill` column across the same cells is
0.0056 / 0.0065 / 0.0057 / 0.0017 — flat. The one cell with a `dfill` lift is `LOBT != IOBT`
(11,236 rows, `dfill` 0.0261 vs 0.0056 baseline, a 4.6× lift on 0.5 % of rows).

Identifiable at prediction time: **yes for matched rows** — `LOBT_flt`, `IOBT_flt`, `EOBT_1_flt` and
`SCHED_TIME_UTC_mvt` are all non-null in the ranking set except on the 5,290 unmatched rows (§G1).
Query: `(SCHED_TIME_UTC_mvt == LOBT_flt) & (SCHED_TIME_UTC_mvt == EOBT_1_flt)`.

### 2.7 The seconds-of-minute pattern is a recording grid, not a fill signal — and it biases the ±10 s threshold (§F1, §F5, §J2)

- `SCHED` seconds are **0 on 100 % of rows at all ten airports**. So |BLOCK − SCHED| ≤ 10 s can only be
  reached by a BLOCK stamp that sits within 10 s of a minute boundary.
- BLOCK within 6 s of a minute (§J2, matched / unmatched): EDDM, EGLL, LFPG, LIRF, LTFM = 1.000 / 1.000;
  LEBL 0.9996 / 0.9876; LEMD 0.9999 / 0.9989; **EDDF 0.2173 / 0.2244; LSZH 0.2490 / 0.2514;
  EHAM 0.2191 / 0.4687**. The three airports with an off-grid BLOCK are exactly the three with the lowest
  f10 (0.0112–0.0124) and the only three where f6 ≠ f10 (0.0073–0.0076 vs 0.0112–0.0124). At the ±120 s
  threshold the ordering compresses to 0.114–0.257, i.e. **f10 partly measures the BLOCK recording grid.**
- MVT seconds-of-minute — the only one of the three that survives into the ranking set — carries **no**
  fill information (§F5). Within the airports where MVT is off-grid, fill rate near vs away from a
  minute boundary: EDDF 0.0128 / 0.0121, EHAM 0.0139 / 0.0121, LEBL 0.0569 / 0.0559, LEMD 0.0460 / 0.0442,
  LSZH 0.0128 / 0.0109.
- SCHED minute-of-hour: 2,062,242 of 2,062,577 matched rows are on a 5-minute grid; the 335 that are not
  have fill 0.0776 vs 0.0480 (§F6) — too few rows to be usable.

### 2.8 Separability of the fill on MATCHED rows — the 98,925 rows never modelled (§E, §J5)

Honest leave-one-month-out smoothed group means (Beta prior, weight 20; no model fit, no tuning):

| grouping | LIRF-matched R²/AUC | other-matched R²/AUC | all-matched R²/AUC |
|---|---|---|---|
| operator × far × `nosupd` | **0.2458 / 0.8262** | 0.0319 / 0.7642 | 0.0825 / 0.7889 |
| ADEP × operator × far × `nosupd` | 0.2458 / 0.8262 | 0.0377 / 0.7815 | **0.1179 / 0.8161** |
| ADEP × operator × far | 0.2054 / 0.7944 | 0.0346 / 0.7688 | 0.1062 / 0.8044 |
| ADEP × operator × hour × far | 0.1920 / 0.7925 | 0.0331 / 0.7661 | 0.1001 / 0.8003 |
| ADEP × operator × stand × far | 0.1663 / 0.7804 | 0.0305 / 0.7575 | 0.0890 / 0.7905 |
| operator | 0.1472 / 0.7197 | 0.0055 / 0.6106 | 0.0442 / 0.6730 |
| far | 0.0566 / 0.6473 | 0.0226 / 0.6933 | 0.0232 / 0.6733 |
| ADEP × stand | 0.0483 / 0.6278 | 0.0096 / 0.6461 | 0.0498 / 0.7096 |

In-sample between-group R² (§E1, comparable to the 0.156–0.238 in `lirf-matched-fill.md` §3.5): LIRF
operator 0.1556, `+far` 0.2173, `+hour+far` 0.2349, `+stand+far` 0.2463; other-matched 0.0060 / 0.0377 /
0.0487 / 0.0609. **Adding hour or stand raises the in-sample R² and lowers the LOMO score** (LIRF
0.2054 → 0.1920 / 0.1663), so those two dimensions are in-sample noise at LIRF once operator and `far`
are in.

Continuous drivers (point-biserial corr with the fill flag, §E3), LIRF-matched / other-matched:
`ms` −0.144 / −0.073, `aobt3−sched` −0.142 / −0.076, `sched−lobt` +0.127 / +0.049,
`sched−eobt1` +0.125 / +0.051, `sched−iobt` +0.127 / +0.049, `hour` −0.037 / −0.019,
`planned_flight_s` −0.033 / −0.009, `mvt_sec` −0.002 / −0.000, `lobt−iobt` −0.003 / +0.003.

**Answer to the question asked:** matched rows are separable, but far less than unmatched rows — best
honest AUC **0.826** at LIRF and **0.782** at the other nine airports, against the 0.957–0.970 classifier
AUC reported for unmatched rows. The separating columns are, in order, operator, `far`, `nosupd`; stand,
hour, runway, segment, destination add nothing out of sample. `far` itself is ranking-visible for
matched rows (`mvt_minus_aobt3` non-null on 339,551 of 344,841).

### 2.9 Unmatched rows have no flt-side column at all; the flight designator is what is left (§I)

§I1: on all 22,470 training unmatched rows, `EOBT_1_flt`, `LOBT_flt`, `IOBT_flt`, `ARVT_1_flt`,
`AIRCRAFT_OPERATOR_flt`, `MARKET_SEGMENT_flt` are **100 % null**; `STAND_mvt` 0.049 % null,
`RUNWAY_mvt` 0 %, `FLIGHT_mvt` 0.29 % null. So §2.6's flag and the operator column do not exist there.

The 3-letter prefix of `FLIGHT_mvt` is an operator proxy that does survive, and it is the strongest
predictor of the fill on LIRF unmatched rows:

| grouping (LIRF unmatched, 1,488 rows) | groups | in-sample R² | LOMO R² | LOMO AUC |
|---|---|---|---|---|
| prefix × `ms` band | 283 | 0.5792 | 0.2166 | **0.8187** |
| prefix | 129 | 0.4479 | **0.2437** | 0.8078 |
| prefix × hour | 568 | 0.6594 | 0.0988 | 0.7700 |
| stand | 132 | 0.2157 | 0.0570 | 0.6467 |
| hour | 24 | 0.0931 | 0.0376 | 0.5929 |
| `ms` band | 6 | 0.0862 | 0.0478 | 0.5766 |
| runway | 6 | 0.0147 | −0.0154 | 0.3883 |

Single-feature AUC of `ms` alone: **0.6616** (§I4). Prefix rates (§C7, n ≥ 30): ISR 0.983 (59),
AEZ 0.765 (51), no prefix 0.754 (65), NOS 0.742 (31), WZZ 0.712 (59), WMT 0.486 (183), RYR 0.335 (239),
ITY 0.160 (94), EJU 0.100 (40). Over all 22,470 unmatched rows (§I5): ADEP × `ms` band LOMO AUC 0.9302
(R² 0.3158), ADEP × prefix 0.9086, ADEP alone 0.8778; `ms` alone 0.5274; `ADEP == LIRF` alone 0.8028.

Identifiable at prediction time: **yes** — `FLIGHT_mvt`, `STAND_mvt`, `RUNWAY_mvt`, `mvt_minus_sched`
are all non-null on the ranking unmatched rows; 358 of the 383 ranking LIRF unmatched rows carry a
prefix seen in training LIRF unmatched (§K3).

### 2.10 The ranking set: the fill itself is unobservable; two of its drivers have shifted (§G, §K)

- `BLOCK_TIME_UTC_mvt`, `TAXITIME_SEC_mvt` and `AOBT_3`-derived `far` are unavailable for the unmatched
  rows; the fill state is unobservable for all 344,841 rows.
- Expected filled matched rows under training `ADEP × operator × far × nosupd` cell rates (§K1–K2):
  **15,970** of 339,551 (flat-rate expectation 16,286), of which LIRF **2,161 (Jan) + 2,466 (Jul) = 4,627**
  and expected `dfill` **415 + 782 = 1,197** (total over all airports 2,068). Per-airport expected rates:
  LIRF 0.1973 (Jan) / 0.1585 (Jul), LTFM 0.0468 / 0.0601, LEBL 0.0592 / 0.0506, EGLL 0.0500 / 0.0446,
  EDDM 0.0455 / 0.0465, LEMD 0.0424 / 0.0393, LFPG 0.0402 / 0.0398, EDDF 0.0117 / 0.0122,
  EHAM 0.0116 / 0.0124, LSZH 0.0105 / 0.0095.
- **`far` shifts between the two ranking months** (§G2): LIRF matched `far` share Jan 2026 0.3974 vs
  Jul 2026 0.6083 (training LIRF matched 0.483, from `lirf-matched-fill.md` §3.6). Also LEBL 0.392 → 0.539,
  LTFM 0.488 → 0.355. This alone moves the expected LIRF fill rate from 0.197 to 0.159.
- **`nosupd` is lower in the ranking set at 9 of 10 airports** (§G5, SCHED == EOBT_1 share, train → rank):
  LFPG 0.6373 → 0.4506, EGLL 0.5323 → 0.4646, LSZH 0.6082 → 0.5550, LEMD 0.5352 → 0.4930,
  EDDM 0.5611 → 0.5253, LEBL 0.5401 → 0.5160, EDDF 0.5522 → 0.5270, LIRF 0.4015 → 0.3750,
  EHAM 0.5573 → 0.5554, **LTFM 0.5913 → 0.6639 (up)**. SCHED == LOBT moves −4.3 to +7.3 pp.
- Compositions that did **not** move: LIRF matched segment mix train vs rank (§G4) Mainline 0.5395 /
  0.5350, Lowcost 0.4096 / 0.4051, Regional 0.0298 / 0.0384, Cargo 0.0110 / 0.0111, all others ≤ 0.0012
  apart. LIRF unmatched profile (§G6) Jan 2025 n 60 / actype-null 1.0 / ms ≥ 2 h 0.400 vs Jan 2026
  n 107 / 1.0 / 0.486; Jul 2025 n 337 / 1.0 / 0.466 vs Jul 2026 n 276 / 1.0 / 0.467; modal runway 25 in
  all four.
- Expected filled ranking LIRF **unmatched** rows under training prefix rates: **165.4** of 383
  (flat-rate 185.6) (§K3).

### 2.11 What the fill costs on `oof_main-48c19bc830` (§H)

672,428 scored rows, total SSE 7.5275e10, RMSE 334.58.

| cell | n | RMSE | share of total SSE |
|---|---|---|---|
| matched, not filled | 632,544 | 209.0 | 0.3669 |
| matched, filled | 31,506 | 332.6 | **0.0463** |
| unmatched, not filled | 8,025 | 2,221.1 | 0.5259 |
| unmatched, filled | 353 | 3,601.1 | **0.0608** |

Filled rows total **10.71 %** of SSE on 4.74 % of scored rows. By airport (§H2), filled rows only:

| ADEP × group | n | RMSE | SSE share |
|---|---|---|---|
| LIRF unmatched | 233 | 4,428.0 | **0.0607** |
| LIRF matched | 8,950 | 522.7 | **0.0325** |
| LEBL matched | 3,339 | 342.9 | 0.0052 |
| LTFM matched | 4,779 | 198.1 | 0.0025 |
| EGLL matched | 3,585 | 192.5 | 0.0018 |
| LFPG matched | 2,977 | 198.7 | 0.0016 |
| LEMD matched | 3,017 | 182.3 | 0.0013 |
| EDDM / EDDF / LSZH / EHAM matched | 2,523 / 920 / 481 / 935 | 135.6 / 179.1 / 194.6 / 131.3 | 0.0006 / 0.0004 / 0.0002 / 0.0002 |
| all non-LIRF unmatched filled | 120 | 129.9–454.8 | 0.0001 total |

So **LIRF carries 0.0932 of the 0.1071** on 9,183 scored filled rows; the other nine airports' 22,676
scored filled rows carry 0.0139.
`dfill` rows (§H4): LIRF 1,955 rows RMSE 1,079.4 (SSE share 0.0303, median error −427), LEBL 415 rows
877.4 (0.0042, −695), LTFM 691 rows 361.9 (0.0012, **+92**), the rest ≤ 0.0007 each.

Oracle-exact replacements (§H6), total RMSE 334.58 → : all filled rows **316.15**; matched filled only
326.74; unmatched filled only 324.25; matched `dfill` only 328.11.

The 81,361 rows with 10 s < |BLOCK − SCHED| ≤ 120 s (§H5) are *not* part of the phenomenon: matched
81,082 rows RMSE **162.6** — below the matched average of 209.0 — carrying 2.85 % of SSE.

---

## 3. What is CLEAN

- **The label identity.** `(y − ms) + (BLOCK − SCHED) = 0` on all 2,085,047 rows (§A); no null BLOCK or
  SCHED in training.
- **Month-to-month stability of the matched fill rate at all ten airports.** No break, no trend, no
  outlier month; within-airport max/min 1.2–2.3, airport ordering identical in all 12 months (§B2, §B3).
- **Runway.** Not a driver at any airport, matched or unmatched: LIRF 0.124–0.182 over 4 runways,
  LTFM 0.048–0.076 over 9, LEBL 0.049–0.060, EDDM 0.043–0.058; LIRF-unmatched runway has a *negative*
  LOMO R² (−0.0154, AUC 0.388) (§C6, §C7, §I3).
- **MVT seconds-of-minute.** The one timestamp-precision signal in the ranking set carries no fill
  information at any airport (§F5); the BLOCK/MVT minute grid is an airport recording property, identical
  for matched and unmatched rows at 9 of 10 airports (§J2).
- **`LOBT` vs `IOBT`.** Equal on 0.979–0.9998 of rows at every airport; no fill signal (corr −0.003 /
  +0.003) beyond a 0.5 %-of-rows `dfill` lift (§F1, §F2, §E3).
- **Wake category, flight type, destination.** Between-group R² 0.0004 / 0.0001 / 0.018 over all matched
  rows (§E1).
- **The 70,556 non-LIRF matched filled rows do not produce the multi-hour label artefact**: 1 row with
  y ≥ 2 h among their 5,323 `dfill` members; y medians on filled rows match the not-filled control at
  every airport (§D1, §D2, §D4).
- **The 10–120 s band is benign** — matched rows there have a *lower* OOF RMSE (162.6) than the matched
  average (§H5); the ≤ 10 s threshold is the correct cut.
- **LIRF matched segment composition and LIRF unmatched profile are unchanged 2025 → 2026** (≤ 0.9 pp on
  every segment; actype-null 1.0 and modal runway 25 in both years) (§G4, §G6).
- **Flight-prefix coverage for the ranking LIRF unmatched rows**: 358 of 383 seen in training (§K3).

---

## 4. Facts and inference, kept apart

**Facts** (measured by the script; every number in §2 and §3): the four-population census and its rates;
the 12-month per-airport rate ranges; the `dfill` counts, directions and label tails per airport; all
group-mean R² and LOMO R²/AUC values; all point-biserial correlations; the `nosupd` contingency tables
and their per-airport cell rates and counts; the seconds-of-minute and stamp-equality structure; the
ranking-set null counts, composition tables and expected-fill sums; the OOF SSE decomposition and the
oracle-exact RMSE values; the identity `Σ|(y−ms)+(BLOCK−SCHED)| = 0`.

**Inference** (not measured here):

- I1. That `BLOCK == SCHED` arises from an upstream fill step rather than from genuine on-schedule
  pushback. The columns carry no provenance flag. What the data *does* show is that at nine airports the
  predicate is 92.5 % accompanied by |BLOCK − AOBT_3| ≤ 600 s (consistent with genuine on-time pushback
  plus a minute-resolution stamp), while at LIRF 24.05 % of it is not.
- I2. That `nosupd` lifts the fill rate *because* an unrevised schedule means an on-time departure. The
  measured pattern — lift on `fill` at all ten airports, no lift on `dfill` — is what that mechanism
  would produce, but the flag's behaviour on a genuine fill process is not separately observable.
- I3. That the ranking set contains ≈ 15,970 filled matched rows and ≈ 2,068 `dfill` rows. This is a
  reweighting of training cell rates onto the 2026 composition, not a measurement; the labels and BLOCK
  are absent. It assumes the per-cell fill rate is stationary across the year boundary, which §2.2
  supports within 2025 and §2.10 partially contradicts (two drivers, `far` and `nosupd`, have moved).
- I4. That the negative median `resid` on LTFM / LFPG / EHAM `dfill` rows is the same recording
  phenomenon as the positive one at the other seven airports rather than a separate AOBT_3 defect. The
  sign split matches `matched-proxy-outliers.md` §3.1's independent AOBT_3-earlier / BLOCK-earlier split,
  which is evidence but not identification.
- I5. That the LOMO AUC ceiling of 0.826 (LIRF matched) / 0.782 (other matched) is a property of the data
  rather than of group means: a fitted classifier was not permitted here and might rank better, as it
  apparently does on the unmatched rows (0.957–0.970 reported, against 0.930 for the best group-mean
  ranking measured here).
- I6. That the LIRF-unmatched month-to-month rate swing (0.323–0.761) is a dilution effect of the varying
  unmatched volume rather than a change in the fill process. The filled *count* per month (25–109) is far
  more stable than the rate, which is consistent with dilution, but the mechanism is not observable.
