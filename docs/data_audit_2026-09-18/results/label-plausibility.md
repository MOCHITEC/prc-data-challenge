# Label plausibility and row identity — audit result (2026-09-18)

Lens: **is each label physically possible, and is each row the flight it claims to be?**

Script: `docs/data_audit_2026-09-18/scripts/label-plausibility.py`
(no training, `POLARS_MAX_THREADS=1`, ~6 min wall, reads only `data/raw/training_*.parquet`,
`data/raw/ranking.parquet`, `data/external/airports.parquet`).
Every number below was produced by that script in one run; none is copied from another result file
except where a prior file is named for comparison.

**Scope constraint stated once.** All 344,841 ranking DEP rows have `BLOCK_TIME_UTC_mvt` null and
`TAXITIME_SEC_mvt` null, so **every label-valued statement here is training-only (2025)**. What *is*
available for 2026 is (i) the DEP rows' non-label columns and (ii) the **344,693 ranking ARR rows,
which are fully labelled** (`TAXITIME_SEC_mvt = BLOCK − MVT`, taxi-in, 0 nulls). §6 uses both.

**Geometry constraint stated once.** `data/external/airports.parquet` has three columns
(`icao`, `lat`, `lon`) — airport centroids for 44,239 aerodromes and **no stand or runway geometry**.
A stand→runway distance therefore cannot be computed from it. The physical floor used below is
empirical instead, from two independent sources:
(a) the DEP label distribution of the same `(ADEP_mvt, STAND_mvt, RUNWAY_mvt)` triple, and
(b) the **ARR taxi-in** distribution at the same `(airport, STAND_mvt)` — a separate traversal of the
same apron-to-runway-system path, measured on different rows.
The airport coordinates are used for what they can do: great-circle distance between `ADEP_mvt` and
`ADES_mvt` for the implied-ground-speed check in §2.13.

---

## 1. What was checked

1. Re-verification of the label identity on DEP and of the taxi-in identity on ARR.
2. The label distribution per `(ADEP, STAND, RUNWAY)` triple: pair count, pair medians, how much of
   the label's sum of squares the triple explains, and which triples have implausible medians.
3. The short tail: `y < 60 s` measured against (a) the row's own triple median and (b) the ARR
   taxi-in 5th percentile at the same stand.
4. The negative (369) and near-zero (460) labels in full: airport, runway, stand, operator, flight
   rule, market segment, aircraft type, month, hour, date concentration, seconds-of-minute
   alignment, overlap with the LIRF sched-fill predicate, and prevalence in the ranking set.
5. The long tail decomposed against the two established pathologies (`|BLOCK−SCHED| ≤ 10 s` =
   sched-fill; `y > 72,000 ∧ date(MVT)−date(BLOCK) ≥ 1` = day rollover), at five thresholds.
6. Identity: `MVT_ID_mvt` / `FLIGHT_ID_mvt` / `CALLSIGN_flt` uniqueness and collisions; same-callsign
   overlapping ground windows; same-stand overlapping pushback windows; DEP row against the ARR row
   of the same `FLIGHT_ID`; implied ground speed; the previous leg via nearest preceding same-stand
   arrival.
7. All of the above repeated or re-expressed for 2026 (composition of the ranking DEP rows; the
   labelled 2026 ARR rows; stand/triple cold start; ranking-side identity checks).

---

## 2. Findings

Layout per finding: **fact** → **count** → **identifying subset** → **identifiable at prediction
time?** → **query**. Inference is confined to the italic *Inference* lines and to §4.

### 2.1 `RUNWAY_mvt == 'H'` is a helipad, and its label is not a taxi time

| quantity | value |
|---|---|
| training DEP rows with `RUNWAY_mvt == 'H'` | **387**, all at LSZH |
| label quantiles (q0/q25/q50/q75/q95/q1) | −12 / −6 / **0** / 7 / 110 / 7,625 |
| `y < 60` | **366 / 387 (0.9457)** |
| `y ≤ 0` | 196 |
| top aircraft types | AS50 135, A109 83, B429 43, EC20 38, EC30 25, AS32 13, A139 10, B06 7 |
| stands | HPW 294, REG 75, MFG 4, W31 4, W21 4, W03 2 |
| LSZH other runways (28/32/16/10/34) | n 84,800 / 32,874 / 12,884 / 2,569 / 1,393, medians 673 / 719 / 973 / 822 / 876 |
| training ARR rows landing on `'H'` | 376, median taxi-in **2 s** |
| **ranking DEP rows** with `RUNWAY_mvt == 'H'` | **79** (all LSZH) |
| **ranking ARR rows** on `'H'` (2026, labelled) | 77, median taxi-in **2 s** |

Of the 7,108 `(ADEP, STAND, RUNWAY)` triples with n ≥ 20, exactly **two** have a median below 120 s,
and both are helipad triples: `(LSZH, HPW, H)` n 294 median **−1**, `(LSZH, REG, H)` n 75 median **+1**.
The 1st percentile of all 7,108 triple medians is 353 s.

Identifiable at prediction time: **yes** — `ADEP_mvt == 'LSZH' & RUNWAY_mvt == 'H'`, both non-null on
every ranking row.

Query: `dep.filter(RUNWAY_mvt == 'H')`; §B2 of the script.

### 2.2 A second helicopter population, at EHAM, is *not* flagged by `RUNWAY_mvt`

`EHAM` stand `HG01`: **1,001** training DEP rows, of which **990 are AIRCRAFT_TYPE `A139`**
(AW139 helicopter), median label **176 s** against the EHAM airport median 742 s; **867** of the 1,001
have `ADES_mvt == 'EHAM'`; **975** are unmatched (`FLIGHT_ID_mvt` null). These rows carry ordinary
fixed-wing runway designators (22, 24, 04, 18L …), so the §2.1 rule does not catch them. Stand `HG03`
contributes another 67 rows to the EHAM `y < 120` set. Of all 1,009 EHAM rows with `y < 120`, 291 are
on HG01, 88 on K38, 67 on HG03.

**Ranking:** 156 rows at `EHAM`/`HG01`.

Identifiable at prediction time: **yes** — `ADEP_mvt == 'EHAM' & STAND_mvt == 'HG01'`, or more
generally `AIRCRAFT_TYPE_mvt == 'A139'`.

Query: `dep.filter((ADEP_mvt=='EHAM') & (STAND_mvt=='HG01'))`; §C2, §G6.

### 2.3 Local flights (`ADEP_mvt == ADES_mvt`) have a different label law

**2,193** training DEP rows across 9 airports, median **243 s** against the all-DEP median 912 s.

| ADEP | n | median y | y < 60 | unmatched |
|---|---|---|---|---|
| EHAM | 1,549 | 190 | 92 | 1,361 |
| LSZH | 214 | 11 | 113 | 152 |
| LEBL | 137 | 506 | 13 | 113 |
| EDDM | 95 | 732 | 0 | 4 |
| LEMD | 76 | 978.5 | 0 | 1 |
| EDDF | 56 | 812.5 | 1 | 14 |
| LIRF | 41 | 604 | 1 | 31 |
| LFPG | 23 | 1,083 | 1 | 1 |
| LTFM | 2 | 326.5 | 1 | 1 |

**Ranking: 325 rows** (EHAM 236, EDDM 20, LIRF 19, LSZH 18, LEMD 17, EDDF 6, LFPG 5, LEBL 4).
Prevalence is stable: 0.0949 % of 2025 Jan+Jul DEP rows vs 0.0942 % of 2026.

Identifiable at prediction time: **yes** — `ADEP_mvt == ADES_mvt`, both non-null on every ranking row.

Query: `dep.filter(ADEP_mvt == ADES_mvt)`; §C2, §G6.

### 2.4 VFR rows (`FLIGHT_RULE_mvt == 'V'`) carry a fifth of the short tail

| group | n | median y | y < 60 | y < 0 |
|---|---|---|---|---|
| `I` | 2,081,914 | 912 | 524 | 145 |
| `V` | **2,930** | **232** | **580 (0.198)** | **224** |
| null | 203 | 662 | 1 | 0 |

**Ranking:** 419 `V` rows, 20 null, 344,402 `I`. Prevalence 0.1426 % (2025 Jan+Jul) → 0.1215 % (2026).

Identifiable at prediction time: **yes** — `FLIGHT_RULE_mvt` is present on the ranking rows.

Query: `dep.group_by('FLIGHT_RULE_mvt')`; §B1, §C2.

### 2.5 The negative and near-zero labels are one bounded ±12 s band

**This is the subset the brief asked to be characterised in full.**

Facts:

* **369** negative labels, confined **exactly to [−12, −1]**; per-value counts 11 – 44 with no
  structure (−12: 11, −11: 39, −10: 34, −9: 29, −8: 27, −7: 20, −6: 44, −5: 33, −4: 33, −3: 28,
  −2: 32, −1: 39).
* **389** labels in [0, 12] (per-value counts 14 – 47), then a cliff: **49** labels in [13, 24]
  (13: 3, 14: 7, 15: 3, 16: 2, 17: 3, 18: 4, 19: 7, 20: 6 …).
  So **758** rows sit in `|y| ≤ 12` against 49 in the next 12-second window — a step of ≈ 15×.
* 460 rows in [0, 30) (the count named in the brief); 829 rows `y < 30`; 1,105 rows `y < 60`.

Composition, 369 negatives / 460 rows in [0,30):

| attribute | negatives (369) | 0 ≤ y < 30 (460) |
|---|---|---|
| top (ADEP, RUNWAY) | LSZH/H 188, LSZH/28 102, LFPG/26R 23, LFPG/09R 16, LFPG/08L 14, LSZH/32 7 | LSZH/H 176, LSZH/28 140, LFPG/26R 22, LFPG/27L 16, LFPG/08L 15, LSZH/32 15 |
| flight rule | V 224, I 145 | V 244, I 216 |
| helipad (`RUNWAY=='H'`) | 188 | 176 |
| unmatched (`FLIGHT_ID` null) | 240 | 263 |
| `AIRCRAFT_OPERATOR_flt` null | 240 (identical to the unmatched set) | 263 (identical) |
| `AIRCRAFT_TYPE_mvt` null | 2 | 4 |
| top stands | HPW 150, MFG 43, REG 40, P73 10, null 8, GA6 7 | HPW 140, MFG 59, REG 42, GA6 10, P73 9 |
| top operators (hashed) | 240 null, then 25 / 14 / 13 / 12 / 9 | 263 null, then 37 / 19 / 10 / 9 / 8 |
| months (min–max) | 16 (Sep) – 60 (Jun), all 12 months present | 17 (Nov) – 76 (Jun) |
| distinct dates | 207 | 239 |
| max rows on one date | 19 (2025-06-03), then 7, 5 | 12 (2025-06-17), then 11, 8 |
| hour of day | daylight-shaped, peak 13 UTC (45), 0 UTC 16, 03–04 UTC 1 each | same shape, peak 13 UTC (51) |
| `\|BLOCK−SCHED\| ≤ 10 s` (LIRF sched-fill predicate) | **28 / 369** | **30 / 460** |
| `BLOCK` sec-of-minute == 0 | 16 / 369 | 14 / 460 |
| `MVT` sec-of-minute == 0 | 4 / 369 | 22 / 460 |

**No time clustering:** spread over 207 (resp. 239) distinct dates in all 12 months; the largest
single-date concentration is 19 rows. **Not the sched-fill:** only 28 of 369 satisfy the LIRF
predicate. **Not a single operator:** 240 of 369 have no operator at all (they are the unmatched
rows), and the largest named operator contributes 25.

The same band exists on the ARR side and in 2026:

* training ARR taxi-in in [−12, −1]: **209** rows; training ARR taxi-in **below −12: 11 rows, all at
  LFPG** (runways 08R 3, NA 3, 27R/26L/26R/09L/09R 1 each). Training ARR minimum −2,652.
* **2026 ranking ARR minimum is exactly −12**; 39 negatives, all in [−12, −2]; by airport LSZH 35,
  LFPG 3, LTFM 1.

Identifiable at prediction time: **partly.** The label itself is not observable, but the subset that
produces it is: 188 of the 369 negatives are `LSZH & RUNWAY=='H'` (§2.1), 224 are `FLIGHT_RULE=='V'`
(§2.4), 240 are the unmatched rows; the residue (§2.6) is not separable by any column tested here.

Query: `dep.filter(y < 0)` / `dep.filter(y.is_between(0,30,closed='left'))`; §C, §C1, §G1.

*Inference (not fact): a difference of two timestamps that is bounded at ±12 s and roughly uniform
inside that band is the signature of two independent recordings of the **same** physical event, not
of a traverse. On a helipad departure there is no traverse — `BLOCK` (lift from the pad) and `MVT`
(airborne) are the same instant, and ±12 s is the jitter between the two reporting channels. The
existence of the identical band on the ARR side, and the identical −12 floor in 2026, is consistent
with a system-wide property of the timestamp sources rather than with anything about departures.*

### 2.6 Labels too short to be a stand→runway traverse

`y < 60 s`: **1,105** training rows (0.053 %). After removing the two explained populations
(`RUNWAY == 'H'` and `FLIGHT_RULE == 'V'`): **519** rows remain, 142 of them negative.

Tested against the row's own geometry:

| test | rows |
|---|---|
| short rows whose `(ADEP, STAND, RUNWAY)` triple has n ≥ 20 | 1,047 |
| … of which the triple's own median is ≥ 300 s | **558** |
| … of which the triple's own median is ≥ 600 s | **366** |
| short rows with ARR taxi-in statistics at the same stand | 1,092 |
| … of which the ARR taxi-in 5th percentile at that stand is ≥ 120 s | **575** |
| … ≥ 240 s | 268 |
| **intersection: `y < 60` ∧ triple median ≥ 300 ∧ ARR p05 ≥ 120** | **527** |

The 527 "hard-impossible" rows by runway: LSZH/28 258 (min y −12, triple median 709), EHAM/24 45
(min 15, med 563), LFPG/26R 34 (min −11, med 972), LFPG/08L 30 (min −11, med 909), LSZH/32 23,
LFPG/09R 22 (min −12, med 903), LIRF/25 19 (min 49, med 843), LFPG/27L 17, LEBL/24L 16, EHAM/18L 16,
LEBL/24R 12, EGLL/27L 4 (med 1,467), EHAM/04 3, EHAM/36C 3.

Individual extremes (row label, triple median, ARR taxi-in median at that stand):

| ADEP | STAND | RWY | type | y | triple med | stand ARR med | time |
|---|---|---|---|---|---|---|---|
| LFPG | N05 | 09R | A332 | −11 | 1,742.5 | 909 | 2025-04-10 10:54:54 |
| LFPG | L27 | 26R | B78X | −11 | 1,023.5 | 611 | 2025-08-09 14:05:54 |
| LFPG | D12 | 27R | A319 | −11 | 1,142.0 | 652 | 2025-07-29 15:11:54 |
| LSZH | G02 | 28 | E295 | −12 | 920.5 | 408.5 | 2025-06-03 05:29:07 |
| LFPG | P73 | 09R | B738 | −12 | 902.5 | 656 | 2025-01-16 00:50:54 |
| LFPG | I36 | 09R | B738 | −12 | 714.0 | 480.5 | 2025-01-21 03:35:54 |

The 519 non-helipad non-VFR short rows are **not** only light aircraft: the commonest types are
B738 79, A320 33, C56X 25, BCS3 18, CL60 17, E190 17, A21N 16, E55P 16, **A359 16**, C68A 16, PC12 14,
**A332 12**. Market segment: Business 175, Mainline 151, Lowcost 61, null 37, Cargo 36, Regional 26,
Other 24, Non-Scheduled 7, Military 3.

Per-airport rate of "`y < 60` while the triple normally needs ≥ 5 min" (denominator = rows on a
triple with n ≥ 20):

| ADEP | n | y<60 ∧ med≥300 | y<120 ∧ med≥300 | y < 0.2 × triple median |
|---|---|---|---|---|
| LSZH | 132,893 | **285** | 289 | 482 |
| LFPG | 233,539 | **106** | 120 | 145 |
| EHAM | 244,635 | 71 | **476** | 519 |
| LEBL | 178,495 | 65 | 113 | 162 |
| LIRF | 159,272 | 19 | 108 | 189 |
| EGLL | 238,859 | 5 | 5 | 7 |
| LEMD | 208,965 | 3 | 12 | 28 |
| EDDF | 228,485 | 2 | 9 | 20 |
| EDDM | 166,809 | 2 | 4 | 4 |
| LTFM | 265,927 | **0** | 1 | 0 |

Identifiable at prediction time: **partly.** `ADEP`, `STAND`, `RUNWAY`, `AIRCRAFT_TYPE`,
`FLIGHT_RULE`, `MARKET_SEGMENT` are all present on the ranking rows, so the *population* the short
rows come from is identifiable; the individual short row is not, because the label is what makes it
short and there is no further column separating (say) the 12 A332/A359 short rows from the thousands
of normal wide-body departures from the same stands.

Query: §B1–B5 of the script (`short = dep.filter(y < 60)`, joined to `big` = triples with n ≥ 20 and
to `stand_arr` = ARR taxi-in per `(airport, stand)`).

### 2.7 How much of the label the stand×runway triple explains

12,042 distinct `(ADEP, STAND, RUNWAY)` triples with a non-null stand; **7,108** have n ≥ 20 and cover
**2,057,879** of the 2,085,047 training DEP rows. Triple-median quantiles:
q0 **−1**, q1 353, q5 543, q25 773, q50 955, q75 1,164.5, q95 1,463, q100 3,476.

On the 2,057,325 rows of those triples with `y ∈ [−100, 7200]` (i.e. with the known pathological tail
excluded), total SS = 3.516 × 10¹¹, within-triple SS = 2.034 × 10¹¹, so the triple identity alone
accounts for **42.17 %** of the label's sum of squares. Reported as a clean structural fact, not a
recommendation.

Query: §B of the script.

### 2.8 The long tail beyond the two known pathologies

Decomposition against `f10` = `|BLOCK − SCHED| ≤ 10 s` (sched-fill) and `rollover` =
`y > 72,000 ∧ date(MVT) − date(BLOCK) ≥ 1`:

| threshold | n | f10 | rollover | neither | LIRF |
|---|---|---|---|---|---|
| y ≥ 3,600 | 4,132 | 1,678 | 16 | **2,440** | 1,969 |
| y ≥ 7,200 | 584 | 430 | 16 | **140** | 480 |
| y ≥ 14,400 | 122 | 102 | 16 | **6** | 116 |
| y ≥ 43,200 | 34 | 19 | 16 | **1** | 31 |
| y ≥ 72,000 | **16** | 2 | **16** | **0** | 14 |

So above 20 h the two documented mechanisms are complete: the 16 rows with `y > 72,000` are exactly
the 16 rollover rows, with nothing left over. (Note `date(MVT) − date(BLOCK) ≥ 1` on its own holds for
**1,696** DEP rows — ordinary midnight crossings — so the rollover predicate needs the label term.)

The **140** unexplained rows at `y ≥ 7,200`:

| ADEP | n | median y | max y | unmatched |
|---|---|---|---|---|
| LIRF | 40 | 8,104 | 31,268 | 31 |
| LFPG | 36 | 8,261.5 | 58,206 | 6 |
| EGLL | 35 | 8,518 | 11,452 | 10 |
| LTFM | 21 | 7,731 | 10,309 | 19 |
| LSZH | 5 | 9,538 | 18,409 | 3 |
| EHAM | 3 | 10,972 | 11,032 | 1 |

Their `BLOCK − SCHED` (signed) quantiles are q0 −56,163 / q5 −7,376 / q25 −486 / **q50 +597** /
q75 +1,799 / q95 +6,600 / q100 +13,261 — a real off-block after schedule, not a fill. Of the 70 that
have `AOBT_3_flt`, **67 have `|BLOCK − AOBT_3| > 3,600 s`**. 28 of the 140 are members of an
`(ADEP, FLIGHT, SCHED)` group of size ≥ 2 (the return-to-stand twin shape). Month spread 1 – 31; the
largest single-date concentration is 13 rows (2025-02-23), then 5.

The six unexplained rows at `y ≥ 14,400`, listed:

| ADEP | ADES | STAND | RWY | type | y | BLOCK | MVT | unmatched |
|---|---|---|---|---|---|---|---|---|
| LFPG | GMMX | D06 | 26R | A320 | 58,206 | 2025-01-22 22:58:57 | 2025-01-23 15:09:03 | yes |
| LIRF | LROP | 310 | 25 | null | 31,268 | 2025-05-01 23:23:56 | 2025-05-02 08:05:04 | yes |
| LIRF | LGIR | 309 | 25 | null | 25,018 | 2025-08-02 03:46:01 | 2025-08-02 10:42:59 | yes |
| LFPG | LFML | P82 | 08L | B734 | 21,308 | 2025-03-07 14:11:04 | 2025-03-07 20:06:12 | no |
| LSZH | LSZF | MFG | 28 | C172 | 18,409 | 2025-02-08 10:51:47 | 2025-02-08 15:58:36 | yes |
| LFPG | KJFK | K38 | 27L | B764 | 15,059 | 2025-07-09 13:09:59 | 2025-07-09 17:20:58 | no |

Three of these six cross a calendar date but have `y < 72,000`, so they lie outside the established
16-row rollover set.

Identifiable at prediction time: **no** for the unexplained rows individually — the separating
quantities (`BLOCK`, `y`, `BLOCK − AOBT_3`) are all label-side and null on the ranking set. The one
usable marker is twin-group membership (`ADEP`, `FLIGHT_mvt`, `SCHED_TIME_UTC_mvt` are all present on
ranking rows), which covers 28 of 140.

Query: §D, §D1–D3.

### 2.9 Identity: keys

* `MVT_ID_mvt` is **unique on all 4,167,797 training rows** (0 nulls) and on all 344,841 ranking DEP
  rows. **Overlap between training MVT_IDs and ranking DEP MVT_IDs: 0.**
* `FLIGHT_ID_mvt`: 30,930 nulls; 3,821,596 distinct values. Multiplicity over (DEP count, ARR count):

  | nd | na | flights |
  |---|---|---|
  | 0 | 1 | 1,758,975 |
  | 1 | 0 | 1,747,348 |
  | 1 | 1 | 315,269 |
  | **2** | **0** | **3** |

* The **3 FLIGHT_IDs carrying two DEP rows** (all LFPG, both rows matched):

  | FLIGHT_ID | FLIGHT | date | stands | runways | BLOCKs | MVTs | y |
  |---|---|---|---|---|---|---|---|
  | 280492175 | AFR6T | 2025-01-04 | **K65 / H08** | 08L / 08L | 16:26:55 / 16:27:01 | 16:37:04 / 16:46:59 | 609 / 1,198 |
  | 287386537 | DAL81 | 2025-07-10 | **K52 / K70** | 09R / 09R | 14:26:02 / 14:26:01 | 14:43:57 / 14:44:18 | 1,075 / 1,097 |
  | 289918868 | TAY4KU | 2025-08-31 | I37 / I37 | 27R / 27R | 17:52:00 / 17:52:04 | 18:09:37 / 18:09:55 | 1,057 / 1,071 |

  **All 6 rows are already members of an `(ADEP, FLIGHT, SCHED)` group of size ≥ 2**, i.e. of the
  documented twin mechanism. There is **no FLIGHT_ID collision beyond it.**
* `(ADEP, FLIGHT_mvt, SCHED_TIME_UTC_mvt)` groups of size ≥ 2 in training: **224 groups / 450 rows**
  (223 groups of size 2, **1 group of size 4**). This reproduces `return-to-stand-twins.md`'s 223
  pairs and adds the one 4-row group. Ranking: **84 rows** in such groups.
* Ranking file: 0 duplicated DEP `MVT_ID`; **1** FLIGHT_ID with (2 DEP, 1 ARR).

Query: §E1, §D3, §F5.

### 2.10 Identity: can the same aircraft be in two places at once?

**The dataset contains no aircraft registration.** `AIRCRAFT_TYPE_mvt` + `AIRCRAFT_OPERATOR_flt` is
not an aircraft key (an operator holds many airframes of one type), so "same physical aircraft" is
not directly establishable. The two strongest available proxies were used.

* **Callsign.** Same `CALLSIGN_flt`, two DEP rows whose `[BLOCK, MVT]` ground windows overlap:
  **4 occurrences in 2,085,047 rows.** All four are **at the same airport (LFPG)**; **0 are at
  different airports**. Three are the FLIGHT_ID collisions of §2.9; the fourth is FPO701 on
  2025-03-07 (previous window 14:11:04 → 20:06:12, this row 19:46:56 → 20:06:04). In the 2026 ranking
  file, DEP rows with the same callsign and two take-offs less than 600 s apart at **different**
  airports: **0**.
* **Stand occupancy.** See §2.11 — the overlaps found there are a stand-coding artefact, not an
  aircraft being duplicated (the overlapping rows carry different flight numbers).

**Conclusion: no evidence of one aircraft in two places, in either year.**

Query: §E2, §F5.

### 2.11 `STAND_mvt` is not a unique physical stand at several airports

**2,849** DEP rows have their off-block *before* the take-off of the previous DEP from the same
`(ADEP, STAND)`. Of these, **52** carry the same `FLIGHT_mvt` as the previous row (the documented
return-to-stand twin shape, 51 of them LFPG) and **2,793 carry a different flight number**.

| ADEP | overlaps | same flight |
|---|---|---|
| EDDM | **1,735** | 0 |
| LSZH | 369 | 1 |
| LEBL | 279 | 0 |
| LIRF | 215 | 0 |
| EHAM | 93 | 0 |
| EGLL | 79 | 0 |
| LFPG | 67 | **51** |
| LTFM | 7 | 0 |
| EDDF | 3 | 0 |
| LEMD | 2 | 0 |

Overlap length (different-flight cases): q0 1 s, q25 229, q50 **422**, q75 719, q95 5,458, max 113,578.

Concentration, with each stand code's own departure volume:

| ADEP | STAND | overlaps | stand's DEP rows | rate |
|---|---|---|---|---|
| EDDM | 302 | 362 | 3,107 | 0.117 |
| EDDM | 301 | 299 | 2,859 | 0.105 |
| LEBL | **UNKNOWN** | 276 | 1,463 | 0.189 |
| LSZH | MFG | 209 | 1,107 | 0.189 |
| EDDM | 317 | 191 | 2,120 | 0.090 |
| EDDM | 318 | 183 | 1,938 | 0.094 |
| EDDM | 356 | 175 | 1,440 | 0.122 |
| EDDM | 371 | 144 | 1,241 | 0.116 |
| EDDM | 355 | 136 | 1,370 | 0.099 |
| LSZH | GA6 | 131 | 2,171 | 0.060 |
| EDDM | 354 | 102 | 1,097 | 0.093 |
| EDDM | 353 | 71 | 672 | 0.106 |
| EHAM | K38 | 43 | 1,732 | 0.025 |
| EDDM | 352 | 40 | 383 | 0.104 |
| LSZH | T46 | 13 | 550 | 0.024 |

Identifiable at prediction time: **yes** — the affected codes are literal `STAND_mvt` values present
on ranking rows (EDDM `301`, `302`, `317`, `318`, `352`–`356`, `371`; LEBL `UNKNOWN` — 249 ranking
rows carry `STAND_mvt == 'UNKNOWN'` overall; LSZH `MFG`, `GA6`).

Query: §E3, §G2.

*Inference (not fact): at EDDM the 3xx codes, at LEBL the literal string `UNKNOWN`, and at LSZH `MFG`
and `GA6`, behave like apron / area identifiers holding several aircraft at once, not like a single
physical stand. That would make the stand→runway distance they stand for a distribution rather than a
constant, which is the same thing the §2.7 within-triple variance measures.*

### 2.12 Identity: the DEP row against the ARR row of the same `FLIGHT_ID`

315,269 DEP rows join to an ARR row on `FLIGHT_ID_mvt`.

* **`FLIGHT_ID_mvt` is not a per-leg key.** `DEP.ADEP ≠ ARR.ADEP` on **1,261** joins (0.40 %);
  `DEP.ADES ≠ ARR.ADES` on **1,630** (0.52 %); `AIRCRAFT_TYPE` disagrees on **1,690** (0.54 %). Only
  **118** of the 1,261 are a return leg (`ARR.ADES == DEP.ADEP`). Largest disagreeing groups:
  (LTFM, RJBB) 184, (LIRF, LMML) 147, (LTFM, ZGGG) 111, (EDDM, LKTB) 96, (LEMD, LEZL) 95.
* **Airborne time** (`ARR.MVT − DEP.MVT`) quantiles: q0 −8,401, q0.1 % 1,707, q1 % 1,895,
  **q50 5,069**, q99 12,801, q100 39,728.
* **Arrival before its own departure: 7 rows.** `< 300 s` airborne: **14 rows**; block-to-block ≤ 0:
  **1 row**.

  | FLIGHT | ADEP | ADES | DEP take-off | ARR landing | airborne | y |
  |---|---|---|---|---|---|---|
  | WMT6025 | LIRF | LEBL | 2025-10-15 08:54:56 | 2025-10-15 06:34:55 | **−8,401** | 13,133 |
  | AIB105 | EDDF | EDDF | 2025-02-04 12:56:23 | 2025-02-04 12:43:39 | −764 | 671 |
  | AFR88WR | LFPG | LEBL | 2025-07-18 18:09:58 | 2025-07-18 18:05:38 | −260 | 5,876 |
  | LVL101P | LEBL | LEBL | 2025-01-09 12:08:10 | 2025-01-09 12:07:46 | −24 | 2,472 |
  | (null) | LIRF | LIRF | 2025-12-03 12:00:02 | 2025-12-03 11:59:56 | −6 | 605 |
  | ITY1733 | LIRF | LIRF | 2025-08-21 16:17:05 | 2025-08-21 16:17:02 | −3 | 610 |
  | (null) | LIRF | LIRF | 2025-10-31 14:52:01 | 2025-10-31 14:52:00 | −1 | 603 |
  | THY6885 | LIRF | LIRF | 2025-02-09 13:32:59 | 2025-02-09 13:33:00 | +1 | 1,198 |
  | (null) | LIRF | LIRF | 2025-09-23 13:35:02 | 2025-09-23 13:35:03 | +1 | 604 |
  | (null) | LIRF | LIRF | 2025-08-12 08:20:54 | 2025-08-12 08:20:57 | +3 | 588 |
  | (null) | LIRF | LIRF | 2025-12-13 19:12:57 | 2025-12-13 19:13:02 | +5 | 595 |
  | (null) | LIRF | LIRF | 2025-05-06 11:20:55 | 2025-05-06 11:21:04 | +9 | 51 |
  | LXFLS | EHAM | EHAM | 2025-05-19 07:43:45 | 2025-05-19 07:45:42 | +117 | 317 |
  | KL1305F | EHAM | EHAM | 2025-09-11 12:27:08 | 2025-09-11 12:30:15 | +187 | 605 |

  Eight of the fourteen are LIRF rows with `ADEP == ADES == LIRF` and `|airborne| ≤ 9 s`; **seven of
  those eight carry a label in 588–610 s** (the eighth is 51). Their DEP take-off and ARR landing are
  the same instant to within seconds.

Identifiable at prediction time: **no** — the check requires `BLOCK`/`MVT` on the DEP side, which the
ranking DEP rows lack; the FLIGHT_ID-level `ADEP`/`ADES` disagreement *is* checkable on ranking rows.

Query: §E4.

### 2.13 Identity: implied ground speed

On 312,173 legs where DEP and ARR agree on both endpoints, both endpoints are geocoded in
`airports.parquet`, and `ADEP ≠ ADES`: implied speed = great-circle(ADEP, ADES) / airborne time.

Quantiles: q0 −6,417, q0.1 % 173, q1 % 211, **q50 343 kt**, q99 438, q99.9 461, q100 1,816.
Physically impossible rows: **3 above 700 kt, 1 negative — 4 of 312,173 (1.3 × 10⁻⁵).**

| FLIGHT | ADEP | ADES | GC (km) | airborne (s) | implied kt | y |
|---|---|---|---|---|---|---|
| T7JUN | EHAM | LSZH | 603.5 | 646 | **1,816** | 1,516 |
| BAW554R | EGLL | LIRF | 1,444.1 | 3,595 | 781 | 1,687 |
| DISCV | LEBL | LEMD | 483.2 | 1,277 | 736 | 700 |
| AFR88WR | LFPG | LEBL | 858.3 | −260 | −6,417 | 5,876 |

Query: §E5.

### 2.14 Identity: consistency with the previous leg at the same stand

The previous leg was reconstructed as the **nearest preceding ARR at the same `(airport, STAND_mvt)`**
(backward as-of join on block time).

* A preceding same-stand arrival exists for **2,083,213 of 2,085,019** DEP rows that have a stand.
* The DEP's `AIRCRAFT_TYPE_mvt` equals the arriving aircraft's on **94.0 %** of them. Mismatch rate
  by airport: LEBL 1.5 %, LTFM 1.8 %, LIRF 2.8 %, LEMD 3.2 %, LFPG 5.6 %, EGLL 6.1 %, EHAM 7.3 %,
  EDDM 9.4 %, LSZH 9.5 %, **EDDF 13.3 %** — the ordering matches the stand-overlap ordering of §2.11.
* Turnaround (`DEP off-block − previous ARR on-block`) median 5,699 s; per airport 4,133 (LEBL) to
  7,016 (EGLL).
* Restricting to same aircraft type **and** same operator (**1,882,709** rows, the best same-aircraft
  proxy available): q0.1 % 1,500 s, q1 % 2,228 s, **200 rows below 300 s**, 398 below 600 s. Those 200
  rows have label median 681 s against 909 s for the whole same-type/operator set; only 1 of them has
  `y < 60`.
* Correlation between the DEP label and the preceding ARR's taxi-in at the same stand: **r = 0.135**
  (n = 1,879,655).

**Method caveat, stated explicitly:** the as-of join is *backward*, so `turnaround ≥ 0` holds by
construction and **a negative turnaround cannot be detected by this method**. The check is one-sided:
it can show absence of an implausibly short turnaround, not absence of an inversion. The 0 count
reported for `turnaround < 0` is therefore uninformative and is not claimed as a finding.

Query: §E6, §G3.

---

## 3. What is CLEAN

Stated so it need not be re-audited.

1. **Label identity.** `TAXITIME_SEC_mvt == (MVT_TIME_UTC_mvt − BLOCK_TIME_UTC_mvt)` on
   **2,085,047 / 2,085,047** training DEP rows, re-derived here. On the ARR side
   `TAXITIME_SEC_mvt == (BLOCK − MVT)` on **2,082,749 / 2,082,750** (the one exception is the single
   ARR row with a null `BLOCK_TIME_UTC_mvt`).
2. **`MVT_ID_mvt`.** Unique on all 4,167,797 training rows and on all 344,841 ranking DEP rows; zero
   nulls; **zero overlap** between training and ranking DEP MVT_IDs.
3. **No FLIGHT_ID collision beyond the documented twin mechanism.** 3 FLIGHT_IDs carry two DEP rows;
   all 6 rows are already `(ADEP, FLIGHT, SCHED)` twin-group members.
4. **No aircraft in two places.** 4 same-callsign overlapping DEP ground windows in 2,085,047 rows,
   all at one airport; 0 cross-airport occurrences in 2025 and 0 in 2026.
5. **No unexplained rows above 20 h.** The 16 rows with `y > 72,000` are exactly the 16 established
   day-rollover rows; the largest unexplained label anywhere is 58,206 s.
6. **Flight times are aerodynamically sane.** 312,169 of 312,173 geocoded legs; median implied speed
   343 kt, p99.9 461 kt.
7. **No implausibly short turnaround at scale.** On the 1,882,709 same-type-and-operator stand chains
   the 1st percentile turnaround is 2,228 s and only 200 rows fall below 300 s.
8. **Null rates on the geometry columns.** `RUNWAY_mvt` never null in training or ranking;
   `STAND_mvt` null on 28 training DEP rows (LFPG 14, LIRF 10, EGLL 4) and 4 ranking DEP rows.
9. **`FLIGHT_RULE_mvt`, `ADEP_mvt`, `ADES_mvt`, `STAND_mvt`, `RUNWAY_mvt`, `AIRCRAFT_TYPE_mvt`** are
   all present on ranking rows, so every population-level signature in §2 is checkable at prediction
   time even though the label is not.
10. **Cross-year stability of every signature tested** — see §6, all within ~20 % relative.

---

## 4. Facts and inference, separated

Everything in §2 and §3 above a marked *Inference* line is a count produced by the script. The
inferences drawn, kept apart, are:

* **I1.** The ±12 s band (§2.5) looks like two recordings of one event rather than a traverse,
  because it is bounded, flat inside, present on the ARR side, and present in 2026 at exactly the
  same −12 floor. Not proven: no column in the dataset states what `MVT_TIME_UTC_mvt` and
  `BLOCK_TIME_UTC_mvt` are sourced from.
* **I2.** `RUNWAY_mvt == 'H'` is read as "helipad" from the aircraft types on those rows (AS50, A109,
  B429, EC20, EC30, AS32, A139, B06, R66, R22, B505, B407, R44 — all rotorcraft) and from the median
  label of 0 s on DEP and 2 s on ARR. The dataset has no runway-type column stating this.
* **I3.** EDDM `3xx`, LEBL `UNKNOWN`, LSZH `MFG`/`GA6` are read as apron/area codes rather than single
  stands, from the 9–19 % same-stand overlap rates (§2.11). The dataset has no stand-geometry table
  to confirm it.
* **I4.** The 140 unexplained long rows (§2.8) are *not* the sched-fill and *not* the rollover, and
  their `BLOCK − AOBT_3` exceeds 1 h on 67 of the 70 rows where `AOBT_3` exists; what they *are* is
  not established here.
* **I5.** The 8 LIRF rows in §2.12 whose DEP take-off and ARR landing coincide to within 9 s, seven of
  which carry a label in 588–610 s, share a shape; whether they are one mechanism was not tested.
* **I6.** §2.14's negative-turnaround count is an artefact of the backward as-of join, not a finding.

Not established, and named as such:

* Whether any 2026 ranking row actually carries a short, negative or pathological label — the labels
  do not exist in the file.
* Whether the 527 "hard-impossible" short rows are a recording artefact or real ultra-short traverses;
  no column separates them from ordinary departures on the same triples.
* "Same physical aircraft" anywhere in this report — no registration column exists.

---

## 5. Queries (reproduction)

```
POLARS_MAX_THREADS=1 .venv/bin/python docs/data_audit_2026-09-18/scripts/label-plausibility.py
```

Section map: §0 shapes and both identities · §A label quantiles and short-tail thresholds ·
§B stand×runway triples, the short tail against triple median and ARR taxi-in, per-airport rates ·
§C negative and near-zero labels in full, the ±12 band, ranking-set survival of each signature ·
§D long tail against the two known pathologies · §E identity (keys, callsign, stand overlap,
DEP-vs-ARR, ground speed, previous leg) · §F 2025 vs 2026 · §G follow-ups (±12 band, EDDM stand codes,
as-of caveat, 2026 cold-start stands, the 4 callsign overlaps, local flights).

---

## 6. 2025 vs 2026

**Label-valued checks cannot be repeated on 2026** (ranking DEP rows: `BLOCK_TIME_UTC_mvt` null
344,841/344,841, `TAXITIME_SEC_mvt` null 344,841/344,841). Two things can be compared.

### 6.1 Composition of the DEP rows — stable

2025 Jan+Jul (344,419 DEP rows) vs 2026 (344,841 DEP rows):

| signature | 2025 Jan+Jul | 2026 | n (2025 / 2026) | training median y | training `y<60` rate |
|---|---|---|---|---|---|
| `ADEP=LSZH & RUNWAY=H` | 0.0279 % | 0.0229 % | 96 / **79** | 0.0 | 0.9457 |
| `FLIGHT_RULE_mvt == 'V'` | 0.1426 % | 0.1215 % | 491 / **419** | 232 | 0.1980 |
| `ADEP == ADES` | 0.0949 % | 0.0942 % | 327 / **325** | 243 | 0.1012 |
| LSZH stand `MFG` | 0.0537 % | 0.0397 % | 185 / **137** | 767 | 0.0967 |
| LSZH stands `HPW/REG/HAN` | 0.0891 % | 0.0815 % | 307 / **281** | 256 | 0.2528 |
| LFPG stands `P73/P82` | 0.0238 % | 0.0249 % | 82 / **86** | 832 | 0.0400 |
| EHAM stand `HG01` | 0.0441 % | 0.0452 % | 152 / **156** | 176 | 0.0849 |
| `STAND_mvt` null | 0.0029 % | 0.0012 % | 10 / **4** | 980.5 | 0.4286 |
| `STAND_mvt == 'UNKNOWN'` | 0.0798 % | 0.0722 % | 275 / **249** | 735 | 0.0142 |

Every signature except `STAND_mvt` null (10 rows -> 4) changes by at most ~26 % relative; none disappears.

### 6.2 The labelled 2026 quantity: ARR taxi-in

| | n | median | < 60 s | < 0 | min | max |
|---|---|---|---|---|---|---|
| 2025 Jan+Jul | 343,999 | 486 | 246 | 48 | **−1,283** | 33,601 |
| 2026 | 344,693 | 478 | 253 | 39 | **−12** | 50,517 |

Per-airport median taxi-in, 2025 Jan+Jul → 2026: LTFM 778 → 769, EHAM 502.5 → **431**, EGLL 427 → 432,
LFPG 547 → 544, EDDF 556 → 522, LEMD 504 → 484, LEBL 200 → 211, EDDM 358 → 357, LIRF 593 → 590,
LSZH 322 → 336. The `< 60 s` rate moves within 0.00002 – 0.0047 at every airport, the largest being
LSZH (0.0047 → 0.0034) and LEBL (0.0023 → 0.0028).

The −12 floor holds in 2026 with no exception: all 39 negative 2026 ARR labels lie in [−12, −2]
(LSZH 35, LFPG 3, LTFM 1). In 2025 the floor is violated by **11 ARR rows, all at LFPG**.

### 6.3 Cold start — the one real 2025 → 2026 difference

* 407 `(ADEP, STAND, RUNWAY)` triples occur in 2026 that never occur in 2025, carrying **2,495 ranking
  rows (0.724 %)**: EDDM 1,187, EDDF 861, LSZH 93, EGLL 73, LEMD 62, LEBL 61, LTFM 61, LFPG 50,
  LIRF 24, EHAM 23.
* 48 `(ADEP, STAND)` pairs are new, carrying **1,593 ranking rows**: EDDM 1,182, EDDF 300, LSZH 58,
  LEMD 14, EHAM 11, LFPG 10, LIRF 9, LTFM 6, EGLL 2, LEBL 1.
* The EDDM part is a block: stand codes **`103`, `104`, `105`, `106`, `107`, `108`** appear for the
  first time in 2026 with 239 / 228 / 250 / 164 / 140 / 156 ranking rows and **zero 2025 history**.
  EDDM also loses 27 codes present in 2025 (`120`, `121`, `206`, `603`, `61B`, `62A`, `62B`, `62N`,
  `63N`, `64A`–`67N`, `71A`, `71D`, `82A`–`84D`, `C4`, `M14`, `M16`, `TWF`).
  Next largest: EDDF `J5` 104, `J10` 98, `V333` 45, `V335` 29, `V337` 23; LSZH `T50` 38.

Identifiable at prediction time: **yes** — `(ADEP_mvt, STAND_mvt)` is present on every ranking row.

### 6.4 Identity checks on the 2026 file

`MVT_ID_mvt` duplicates 0; FLIGHT_ID multiplicity (0 DEP,1 ARR) 292,362 / (1,0) 288,635 / (1,1) 50,921
/ **(2,1) 1**; `(ADEP, FLIGHT, SCHED)` groups of size ≥ 2 cover **84 rows**; same-callsign take-offs
less than 600 s apart at different airports: **0**.
