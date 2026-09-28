# Result — row-level forensics on the 16 `y > 72,000 s` day-rollover rows (2026-09-18)

Script: `docs/data_audit_2026-09-18/scripts/forensics-rollover.py`
(no training, `POLARS_MAX_THREADS=1`, runtime ≈ 5 min; reads `data/work/feat_train.parquet`,
`data/work/feat_rank.parquet`, all 12 `data/raw/training_*.parquet`, `data/raw/ranking.parquet`,
`experiments/oof_main-48c19bc830.parquet`, `outputs/submit_v10_reparam.parquet`).
Every number below was produced by that script in one run. Context read first and used as input, not
re-derived: `docs/investigation_2026-09-16/results/block-day-rollover.md`,
`docs/investigation_2026-09-16/results/cond-mean-vs-median.md` §3.1 and §3.7,
`docs/data_audit_2026-09-18/results/label-plausibility.md` §2.8–2.12,
`docs/data_audit_2026-09-18/results/arrival-rows.md`, `.../review-far-rows.md` §3 (LB decomposition).

Notation: `y = TAXITIME_SEC_mvt`, `ms = mvt_minus_sched = MVT − SCHED`, **unmatched** = `nm_matched == 0`
(⇔ `AOBT_3_flt` null), **fill** = `|BLOCK − SCHED| ≤ 10 s`, **H_B** = "the BLOCK date is one day early",
**H_M** = "the MVT date is one day late".

---

## VERDICT

**UNDETERMINED.**

* Question (2) — *which timestamp carries the wrong date* — **is now settled**, and the earlier
  "UNDETERMINED" can be retired: on **13 of the 16 rows the BLOCK date is one day early and the MVT
  (take-off) time is correct**; on **1 row (AEZ146R, the only NM-matched one) the MVT date is one day
  late and BLOCK is correct**; 1 row (NOSOS739) is a schedule-fill with no date error at all; 1 row
  (EJU983W) is a stale BLOCK from a different movement, reconcilable by no one-day shift. §2.
* Question (1)/(3) — *targetable* — is **partly** answered: a rule computable at prediction time exists
  (`ADEP == LIRF ∧ unmatched ∧ ms ≥ 43,200`, precision 0.379, recall 0.733, 11 ranking rows; tightened to
  `ms ≥ 57,600`, precision 0.692, recall 0.600, 4 ranking rows). But **4 of the 16 rows fall outside every rule tried** —
  three of them with ms = 939 / 1,740 / 11,694 s, i.e. ordinary-looking departures — and the precision
  estimate rests on **29 training rows**.
* Question (5) — the honest expected leaderboard gain, on 344,841 scored rows from 284.3584:

| treatment | expected gain | 95 % interval from the training rate | if the 2026 rows contain no rollover at all |
|---|---|---|---|
| per-row mixture mean `π̂·87,300 + (1−π̂)·ms` on the **11**-row set, π̂ = 11/29 = 0.379 | **+8.15 s** | **+1.40 … +16.37 s** | **−8.38 s** |
| same on the **4**-row set (`ms ≥ 57,600`), π̂ = 9/13 = 0.692 | +4.76 s | +0.97 … +7.32 s | −4.78 s |
| hard rule "predict 86,400 + LIRF median taxi = 87,424" on the 11 rows | −14.79 s | — | −54.67 s |

  The headline number is therefore **+8 s expected, −8 s floor**, and the *realised* number is a lottery
  over 2¹¹ outcomes (−8.38 s if none of the 11 is a rollover, +37.56 s if all 11 are). The sign of the
  expectation depends entirely on a mixing rate measured on 29 rows in 2025, and **nothing in the 2026
  data can confirm or refute that the mechanism is still live**: the only labelled 2026 quantity is the
  344,693 ranking ARR rows, which contain 0 rollover-shaped values against an expected count of 0.66
  under the 2025 arrival rate (§4). That is why the verdict is UNDETERMINED rather than TARGETABLE.

---

## 1. Per-row forensics: what actually happened to each flight

The full dump (all non-null `feat_train` columns, the raw row, the OOF prediction, the same-`FLIGHT_mvt`
rows over ±4 days, the destination's ARR rows, the complete stand timeline over [BLOCK−12 h, MVT+12 h],
and the ±3 h departure counts at the stand and at the airport) is section **S1/S2** of the script output.
`FLIGHT_ID_mvt` is **null on 15 of the 16 rows**, so the arrival leg can only be found through
`FLIGHT_mvt` + destination; only 2 of the 16 destinations (EGLL, LFPG) are among the 10 observed airports.

Sorted by `y` descending, with the reconstruction:

| # | ADEP→ADES | FLIGHT | y | ms | OOF pred | what happened | implied true taxi |
|---|---|---|---|---|---|---|---|
| 1 | LIRF→LIMC | NOSOS739 | 131,167 | 131,163 | — | Inbound `NOS739` from OOSA (its own SCHED 02-10 03:55) in-blocks stand 704 on **02-11 16:24:03** — a 36 h late rotation. Take-off 02-11 17:51:03. `BLOCK = SCHED − 4 s` ⇒ **schedule-fill, no date error**. | ≤ 5,220 s (unobserved) |
| 2 | LIRF→MMMX | AMX071 | 88,392 | 67,686 | — | Inbound `AMX070` in-blocks stand 710 on **06-25 12:59:54**; take-off 06-25 16:03:06. BLOCK+1 d = 06-25 15:29:54 fits between them. The daily 21:15 service ran 06-20…23, 06-27, 06-28; no other row on 06-24/25. | **1,992 s** |
| 3 | LIRF→ZGSZ | CHHHH438 | 88,132 | 84,236 | 42,846 | Inbound `CHHHH437` from ZGSZ in-blocks stand 711 on **07-21 06:33:57** (its SCHED 07-21 06:20). BLOCK+1 d = 07-21 08:00:04, take-off 08:28:56. Whole rotation 23 h late. | **1,732 s** |
| 4 | LIRF→VRMM | NOSOS610 | 87,605 | 86,102 | 34,093 | Inbound `NOS610` from LIPX in-blocks stand 705 on **12-28 16:46:58** with its own SCHED **12-27 17:10** — the ARR row records the same 23.6 h delay with **no date anomaly** (its label 597 s). BLOCK+1 d = 12-28 18:14:57, take-off 18:35:02. | **1,205 s** |
| 5 | LIRF→EGLL | BAWW561D | 87,598 | 60,599 | 34,540 | Inbound `BAW560F` from EGLL in-blocks stand 825 on **11-11 21:28:02** (2 h 53 late). The recorded BLOCK 11-11 11:55:01 is 9.5 h *before* that and falls while `LAV1143` still occupied 825 (off-block 14:30:05) — impossible. BLOCK+1 d = 11-12 11:55:01 sits in the free window (21:28 → `FIE257` in-block 11-12 17:00). No EGLL ARR row matches this leg. | **1,198 s** |
| 6 | LIRF→KMIA | ITYTY630 | 87,543 | 50,581 | — | Take-off 09-28 22:38:01 from stand 609; BLOCK+1 d = 09-28 22:18:58, after `ITY793` in-block 20:30:05. H_M would put the take-off 9 h 57 *before* the 08:35 schedule. Same flight normal on 09-24 and 09-29. | **1,143 s** |
| 7 | LIRF→SAEZ | ITYTY680 | 87,480 | 56,520 | 32,352 | Take-off 11-04 13:22:00 from stand 608; BLOCK+1 d = 11-04 13:04:00, stand free since `SIA365` off-block 09:38:55. The 11-04 service is a separate matched row (stand 606, 21:51:57→22:08:02, A359). H_M ⇒ take-off 8 h 18 before schedule. | **1,080 s** |
| 8 | LIRF→LFBD | VOE6BB | 87,361 | 73,260 | 39,292 | Inbound `VOE4JT` from LFBD in-blocks stand 308 on **07-02 20:41:00** — the aircraft was not at the gate at the recorded BLOCK 16:44:59. Take-off 07-03 17:01:00; BLOCK+1 d = 07-03 16:44:59. A second, NM-matched `VOE6BB` row records the **same take-off second** (07-03 17:01:01, stand 908, SCHED 07-03 15:05, BLOCK = SCHED−2 s, y = 6,963). | **961 s** |
| 9 | LSZH→LFBE | NJE389D | 87,341 | 939 | — | `BLOCK = SCHED − 1 day − 2 s`. Stand 103 was vacated at 05-13 14:06:20, so the value is not an off-block of this aircraft on 05-13. BLOCK+1 d = 05-14 14:39:58 sits between `BOM1` off-block 12:33:29 and `NJE437A` in-block 16:17:13. **Cannot distinguish "real off-block, date one day early" from "schedule-fill, date one day early"**; the ±6 s jitter is the LIRF fill signature and LSZH's fill rate is 1.1 %. | **941 s** (under the off-block reading) |
| 10 | LIRF→LIBC | SWUU1919 | 87,247 | 65,463 | — | Take-off 03-28 10:41:03 from stand 228; BLOCK+1 d = 03-28 10:26:56, stand free since `AEZ2224` in-block 03-27 19:34:58. The 03-28 service is a separate matched row (stand 229, 16:45:57→17:05:03). H_M ⇒ take-off 5 h 49 before schedule. | **847 s** |
| 11 | LIRF→SAEZ | ITYTY680 | 87,186 | 58,563 | 51,678 | Take-off 07-30 12:46:03 from stand 606; BLOCK+1 d = 07-30 12:32:57, after `ITY621` in-block 09:58:54. At the recorded 07-29 12:32:57 stand 606 was occupied by `ITY792` (off-block 13:05:58). | **786 s** |
| 12 | LIRF→EGPF | EXSXS1GT | 87,177 | 60,118 | 34,970 | Inbound `EXS7MZ` from EGPF in-blocks stand 613 on **01-25 02:31:03** with its own SCHED **01-24 10:05** — again a day-late rotation with a clean ARR row. BLOCK+1 d = 01-25 03:24:01, take-off 03:36:58. | **777 s** |
| 13 | LIRF→LDZA | RYRR90BN | 87,170 | 93,535 | 41,118 | Inbound `RYR852` from LIPE in-blocks stand 321 on **07-14 13:17:54** and nothing else departs 321 before `VLG62RM` in-blocks 15:43:02, so that aircraft's departure is this row. BLOCK+1 d = 07-14 14:46:05, take-off 14:58:55. **The only row where H_M is arithmetically plausible** (off-block 07-13 14:46:05 = 1 h 46 after schedule); stand occupancy decides against it. | **770 s** |
| 14 | LIRF→LFPG | ITY332 | 87,168 | 11,694 | — | **Decisive.** The flight's own ARR row at LFPG: landing **06-26 02:42:00**, in-block 02:54:00. Airborne 01:13:54 → 02:42:00 = **5,286 s**, normal for the route (5,464 / 5,877 / 6,237 s on 06-24 / 06-27 / 06-28); under H_M it would be 91,686 s. Also at stand 409 a **return-to-stand pair** on 06-25 (DEP 22:05:03→22:14:57 and ARR 22:15:01→in-block 22:19:59, callsign `ITY332`) puts the aircraft on stand 409 at 22:20 on 06-25. | **768 s** |
| 15 | LIRF→LICJ | AEZ146R | 87,002 | 87,001 | 84,798 | **Decisive the other way.** The only NM-matched row. `AOBT_3` 07-14 19:57, `LOBT`/`IOBT`/`EOBT_1` 07-14 20:00, `ARVT_1` 07-14 20:57:39, **`ARVT_3` 07-14 20:49** — six clocks put the whole flight on 07-14, and the recorded arrival at LICJ is 39 min after the implied take-off 07-14 20:10:01. Stand 234: `AEZ418P` in-blocks 07-14 18:59:59, next movement 07-15 06:12:04. The 07-15 service is a separate row (stand 238, 20:15:01→20:33:58). **MVT carries a date one day late.** | **602 s** (cf. 906 / 1,203 / 907 s on 07-10/11/12) |
| 16 | LFPG→LFMN | EJU983W | 84,240 | 1,740 | 922 | Take-off 01-19 18:54:00, SCHED 01-19 18:25 — an ordinary 29 min delay. BLOCK 01-18 19:30:00. **Neither shift works**: BLOCK+1 d = 01-19 19:30 is 36 min *after* the take-off. Stand C03 held `EJU68UL` (in-block 01-18 10:23:02) and records no movement between that and this row, so a real off-block from C03 at 01-18 19:30 is possible. BLOCK is a **stale value from a previous C03 movement**, not a date-shifted copy. | unknown; label is not a taxi time |

Neighbouring traffic (script S2 §D): departures within ±3 h of the recorded BLOCK — 0–4 at the same stand,
20–175 at the same airport; within ±3 h of MVT — 1–3 at the same stand, 15–194 at the same airport. Both
instants sit inside ordinary traffic, so **neighbour density alone discriminates nothing**; the stand
timeline does.

---

## 2. Which timestamp carries the wrong date

### 2.1 Facts

* Under **either** hypothesis the implied taxi time is identical, `y − 86,400`, because both shift the
  same interval by one day (script S3: `HB_taxi == HM_taxi` on all 16 rows). The *label correction* does
  not depend on the answer; the *interpretation* does.
* Single-shift arithmetic (script S3): under H_M the implied take-off precedes the schedule by
  **298 … 74,706 s on 11 of the 13 LIRF unmatched rows** (and by 85,461 s on LSZH, 84,660 s on LFPG);
  under H_B the implied off-block follows the schedule by 10,926 … 92,765 s, i.e. a 3–26 h delay.
* Stand-occupancy test (script S3): `BLOCK + 1 day` is **after the last preceding same-stand arrival's
  in-block on 15 of the 16 rows** (the exception is row 1, the schedule-fill). On rows 5, 8, 11 the
  recorded BLOCK instant falls while a *different* aircraft occupied the stand.
* On rows 4 and 12 the **inbound leg's own ARR row carries the day-late operation with no date anomaly**
  (`NOS610` SCHED 12-27 17:10, landing 12-28 16:37:01, in-block 16:46:58, label 597 s; `EXS7MZ` SCHED
  01-24 10:05, in-block 01-25 02:31:03). On row 3 the inbound `CHHHH437` likewise.
* On row 14 the flight's own arrival at LFPG (02:42:00 on 06-26) makes the recorded MVT the true
  take-off to within the route's normal airborne time.
* On row 15, `AOBT_3`, `LOBT`, `IOBT`, `EOBT_1`, `ARVT_1` and `ARVT_3` are all present and all on 07-14,
  while MVT is on 07-15.
* On 2025 ARR rows the same shape occurs twice (script S8), and there it is the **landing time** that
  carries the early date while `BLOCK ≈ SCHED`: `IBEE0657` LEMD→LIRF (MVT 04-23 21:58:57, BLOCK 04-24
  22:12:02, SCHED 04-24 22:15, y 87,185) and `WMTMT993` EGKK→LIRF (MVT 05-18 22:40:58, BLOCK 05-19
  22:55:02, SCHED 05-19 22:55:00, y 87,244). Both are LIRF, `AIRCRAFT_TYPE_mvt` null, `AOBT_3` null.

### 2.2 Verdict per row

| attribution | rows | strength |
|---|---|---|
| **BLOCK date one day early, MVT correct** | 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 (13 rows) | row 14 decisive (arrival row of the same flight); rows 3, 4, 12 near-decisive (inbound rotation recorded a day late with a clean ARR row); rows 5, 8, 11 decisive against the recorded BLOCK (stand occupied by another aircraft); the rest rest on stand occupancy + the impossibility of departing hours before schedule |
| **MVT date one day late, BLOCK correct** | 15 (AEZ146R) | decisive — six independent clocks, including the actual arrival time at the destination |
| **no date error; BLOCK is a schedule-fill** | 1 (NOSOS739) | `BLOCK = SCHED − 4 s`; `BLOCK + 1 day` precedes the inbound's in-block by 11 h |
| **BLOCK is a stale value from another movement** | 16 (EJU983W) | no one-day shift gives a non-negative taxi |

### 2.3 Inference (not established here)

That the two attributions are two faces of one defect — a record assembled from two systems in which
*one* movement timestamp inherits the wrong calendar date — rather than two independent bugs. The
observed split (BLOCK wrong on 13 unmatched rows, MVT wrong on the 1 matched row, landing wrong on the 2
arrival rows) is consistent with "the timestamp that comes from the *second* source is the one that
rolls", but the source systems are not inspectable here and n = 16.

---

## 3. Identifiability at prediction time

`BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` are null on all 344,841 ranking DEP rows (re-verified), so
every quantity used below is one of the other 130 columns.

### 3.1 Rules tried, against the 16 known rows

| rule (ranking-computable) | n train | true pos | precision | recall | n rank |
|---|---|---|---|---|---|
| `ms ≥ 7,200` | 39,852 | 14 | 0.0004 | 0.875 | 10,180 |
| `ms ≥ 14,400` | 6,818 | 13 | 0.0019 | 0.813 | 1,861 |
| `ms ≥ 28,800` | 1,264 | 13 | 0.0103 | 0.813 | 280 |
| `ms ≥ 43,200` | 626 | 13 | 0.0208 | 0.813 | 140 |
| `ms ≥ 57,600` | 385 | 11 | 0.0286 | 0.688 | 81 |
| `ms ≥ 72,000` | 242 | 6 | 0.0248 | 0.375 | 49 |
| `ms ≥ 86,400` | 152 | 3 | 0.0197 | 0.188 | 28 |
| unmatched (any airport) | 22,470 | 15 | 0.0007 | 0.938 | 5,290 |
| LIRF ∧ unmatched | 1,488 | 13 | 0.0087 | 0.813 | 383 |
| unmatched ∧ `ms ≥ 43,200` | 75 | 12 | 0.1600 | 0.750 | 24 |
| **LIRF ∧ unmatched ∧ `ms ≥ 43,200`** | **29** | **12** | **0.4138** | **0.750** | **11** |
| unmatched ∧ `ms ≥ 57,600` | 46 | 10 | 0.2174 | 0.625 | 13 |
| **LIRF ∧ unmatched ∧ `ms ≥ 57,600`** | **13** | **10** | **0.7692** | **0.625** | **4** |
| LIRF ∧ unmatched ∧ `ms ≥ 72,000` | 5 | 5 | 1.0000 | 0.313 | 2 |
| LIRF ∧ unmatched ∧ `ms ≥ 43,200` ∧ `RUNWAY ≠ 16R` | 23 | 12 | 0.5217 | 0.750 | 8 |
| LIRF ∧ unmatched ∧ actype null ∧ `ms ≥ 43,200` | 29 | 12 | 0.4138 | 0.750 | 11 |
| LIRF ∧ unmatched ∧ `ms ≥ 43,200` ∧ stand 6xx/7xx/8xx | 22 | 9 | 0.4091 | 0.563 | 7 |

Best F1 of the sweep: **0.6897** at `LIRF ∧ unmatched ∧ ms ≥ 57,600`, then 0.6154 at
`LIRF ∧ unmatched ∧ ms ≥ 43,200 ∧ RUNWAY ≠ 16R`, then 0.5333 at `LIRF ∧ unmatched ∧ ms ≥ 43,200`.
(Precision/recall here are against the 16-row definition; §3.2 repeats the sweep against the corrected
14-row target and the true rollover count of 15.)

### 3.2 Corrected target and the `ms` gradient

The "16 rows" target mixes two things: `NOSOS739` is a **schedule-fill**, not a rollover, and predicting
≈ 87,300 for it would be badly wrong (its label is 131,167). Using the strict target
`y > 72,000 ∧ |y − ms| > 60` (14 rows; `AEZ146R` is a genuine rollover that this predicate misses because
its BLOCK happens to equal SCHED, so the true rollover count is **15**), inside the 29-row
LIRF/unmatched/`ms ≥ 12 h` population:

| `ms ≥` | n train | rollover | sched-filled | precision | recall (of 15) | n rank |
|---|---|---|---|---|---|---|
| 43,200 | 29 | 11 | 18 | 0.379 | 0.733 | 11 |
| 46,800 | 22 | 11 | 11 | 0.500 | 0.733 | 11 |
| 50,400 | 20 | 11 | 9 | 0.550 | 0.733 | 6 |
| 54,000 | 16 | 10 | 6 | 0.625 | 0.667 | 5 |
| **57,600** | **13** | **9** | **4** | **0.692** | **0.600** | **4** |
| 61,200 | 9 | 6 | 3 | 0.667 | 0.400 | 3 |
| 64,800 | 8 | 6 | 2 | 0.750 | 0.400 | 2 |
| 72,000 | 5 | 4 | 1 | 0.800 | 0.267 | 2 |
| 86,400 | 2 | 1 | 1 | 0.500 | 0.067 | 2 |

Contingency inside the 29 rows split at `ms = 57,600`: rollover 9 / filled 4 above, rollover 2 / filled 14
below; **Fisher exact two-sided p = 0.00274**. Wilson 95 % CI: 9/13 → **[0.424, 0.873]**;
11/29 → **[0.227, 0.560]**.

This is the one thing that goes beyond `block-day-rollover.md` §4 ("ms bucket gives a gradient"): the
gradient is statistically significant and gives a usable, ranking-computable threshold. It remains an
association measured on 29 rows.

### 3.3 What does NOT separate them

* Runway (rollover 25 ×10, 34L ×2; filled 25 ×11, 16R ×6 — 16R never rollover, but 6 rows).
* Stand block, destination region, take-off hour, month, day of week, aircraft-type nullness, flight
  rule, market segment, operator — all null or identical across the two classes in the unmatched
  population (all 29 rows have `AIRCRAFT_TYPE_mvt` null, `FLIGHT_RULE_mvt = "I"`, and every `*_flt`
  column null, so operator/callsign identification is 0.0000 by construction).
* A *twin take-off* (another LIRF DEP row within ±5 s of the same MVT): 2 of the 12 rollover-band rows,
  0 of the 17 others — 2 rows is not a rule.
* Turnaround proxy (MVT − previous same-stand ARR in-block): rollover-band median **7,769 s** (range
  3,955 … 54,365), others median **10,677 s** (range 124 … 354,600) — fully overlapping.
* Same-`FLIGHT_mvt` rows within ±2 days: rollover-band mean 0.500, others 0.412.
* Per-flight precedent for the 11 ranking candidates: `ITYTY672` and `ITYTY674` each have 2 training
  rows in the ≥ 12 h band, **both schedule-fills, 0 rollovers**; the other nine flight numbers have no
  training row in the band at all. No ranking candidate has a same-flight rollover precedent.

### 3.4 Rows that cannot be reached at all

| row | ADEP | matched? | ms | why invisible |
|---|---|---|---|---|
| NJE389D | LSZH | unmatched | **939** | indistinguishable from an on-time LSZH bizjet departure |
| EJU983W | LFPG | unmatched | **1,740** | indistinguishable from an ordinary 29-min-late LFPG departure |
| ITY332 | LIRF | unmatched | **11,694** | 3.2 h delay; 39,852 training rows have `ms ≥ 7,200` |
| AEZ146R | LIRF | **matched** | 87,001 | NM-matched with a full flight plan; `unmatched` filters it out. `ms ∈ [80k, 95k] ∧ matched` has precision 1/167 = 0.006 |

**4 of the 16 rows (3 of them with ms < 12,000 s) are outside every rule above.** Stated plainly: for
those rows nothing in the ranking-populated columns distinguishes them from ordinary departures.

---

## 4. How many are in the 2026 ranking set

Ranking months: **Jan 2026 (152,719 rows) and Jul 2026 (192,122 rows)**, total 344,841.

Training base rate per month (rollover-band rows, `y > 72,000`):

| month | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n | 153,706 | 143,732 | 164,449 | 175,288 | 185,202 | 183,142 | 190,713 | 191,182 | 183,950 | 185,674 | 162,332 | 165,677 |
| rollover | 2 | 1 | 1 | 0 | 1 | 2 | **5** | 0 | 1 | 0 | 2 | 1 |

* Jan + Jul 2025: 7 rows in 344,419 = 2.032 × 10⁻⁵ → **expected 7.01** in the ranking set.
* All 2025: 16 rows in 2,085,047 = 7.674 × 10⁻⁶ → **expected 2.65** in the ranking set.
  The two estimates differ by 2.6× because July 2025 alone holds 5 of the 16; with counts this small the
  month effect is only marginally separable from Poisson noise (P(≥5 in July | uniform rate) = 0.017).

Rows matching the signature in the ranking set (script S5): `ms ≥ 43,200` 140 rows; unmatched ∧
`ms ≥ 43,200` 24; **LIRF ∧ unmatched ∧ `ms ≥ 43,200` 11**; **… ∧ `ms ≥ 57,600` 4**.

| FLIGHT | month | stand | rwy | ADES | SCHED | MVT | ms | v10 prediction |
|---|---|---|---|---|---|---|---|---|
| ITYTY674 | 7 | 606 | 16R | SBGR | 2026-07-19 20:00 | 2026-07-20 09:08:57 | 47,337 | 47,890 |
| WMTMT4SG | 7 | 316 | 25 | LBWN | 2026-07-04 17:50 | 2026-07-05 07:05:05 | 47,705 | 47,850 |
| RYR4852 | 7 | 311 | 25 | LICJ | 2026-07-06 08:10 | 2026-07-06 21:38:59 | 48,539 | 47,151 |
| TARAR753 | 7 | 827 | 25 | DTTA | 2026-07-05 14:20 | 2026-07-06 03:52:56 | 48,776 | 50,468 |
| WMT402 | 1 | 826 | 25 | LRCL | 2026-01-14 00:40 | 2026-01-14 14:14:01 | 48,841 | 49,557 |
| TAY9LA | 1 | 108 | 25 | LMML | 2026-01-22 05:45 | 2026-01-22 20:37:00 | 53,520 | 56,151 |
| WMTT3508 | 1 | 801 | 25 | LWSK | 2026-01-02 19:45 | 2026-01-03 11:27:00 | 56,520 | 56,181 |
| **SVII3006** | 7 | 106 | 25 | HLLB | 2026-07-16 15:30 | 2026-07-17 08:20:56 | 60,656 | 61,662 |
| **ITYTY672** | 7 | 710 | 16R | SBGL | 2026-07-09 20:10 | 2026-07-10 13:12:02 | 61,322 | 62,050 |
| **QFAA7006** | 7 | 811 | 16R | YSSY | 2026-07-19 10:00 | 2026-07-20 10:18:04 | 87,484 | 88,042 |
| **NOSOS431** | 7 | 822 | 25 | LIMC | 2026-07-10 14:50 | 2026-07-11 21:50:54 | 111,654 | 112,771 |

(bold = also matches `ms ≥ 57,600`.) **v10 already predicts ≈ ms on all eleven** (mean |pred − ms| = 988 s),
i.e. the current submission bets "schedule-fill" on every one of them.

Expected rollover count among these 11 at the training rates: 11 × 0.379 = **4.2** by the wide rule, or
4 × 0.692 + 7 × 0.125 = **3.7** by the two-band rate. Scaling up for the 4/15 of training rollovers that
fall outside the candidate set gives **≈ 5** in the whole ranking set, of which ≈ 1.3 would be invisible.
These figures are consistent with the 2.65 / 7.01 base-rate estimates; none of them can be checked.

**Direct 2026 evidence (script S8).** The 344,693 ranking **ARR** rows are fully labelled. They contain
**0 rows with `y > 72,000`**. The 2025 ARR rollover-band rate is 4 / 2,082,749 = 1.921 × 10⁻⁶, so the
expected count in the 2026 arrivals is **0.66** and P(0) ≈ 0.52. The observation is therefore
**uninformative**: it neither confirms nor refutes that the mechanism is live in 2026.

---

## 5. The honest ceiling

Arithmetic convention (as in `review-far-rows.md` §3): N = 344,841 = matched 339,551 + unmatched 5,290;
LB 284.3584 → total SSE **2.7884 × 10¹⁰**; matched at 218.0 → SSE 1.6137 × 10¹⁰; implied unmatched SSE
**1.1747 × 10¹⁰** → unmatched RMSE 1,490.2. Rollover label model from training: `y ∈ [86,700, 88,800]`,
median **87,300**.

### 5.1 What one row is worth, and what one false positive costs

Because v10 predicts ≈ `ms` on the candidates, a missed rollover costs `(87,300 − pred)²` and a false
positive costs `(87,300 − ms)² − (pred − ms)²` — **nearly the same number**:

| FLIGHT | ms | v10 pred | gain if it IS a rollover and we fix it | loss if it is NOT and we fire |
|---|---|---|---|---|
| ITYTY674 | 47,337 | 47,890 | **+8.033 s** | **−8.028 s** |
| WMTMT4SG | 47,705 | 47,850 | +8.050 | −7.885 |
| RYR4852 | 48,539 | 47,151 | +8.342 | −7.551 |
| TARAR753 | 48,776 | 50,468 | +7.004 | −7.455 |
| WMT402 | 48,841 | 49,557 | +7.359 | −7.442 |
| TAY9LA | 53,520 | 56,151 | +4.991 | −5.725 |
| WMTT3508 | 56,520 | 56,181 | +4.981 | −4.790 |
| SVII3006 | 60,656 | 61,662 | +3.372 | −3.592 |
| ITYTY672 | 61,322 | 62,050 | +3.270 | −3.418 |
| QFAA7006 | 87,484 | 88,042 | +0.003 | +0.001 |
| NOSOS431 | 111,654 | 112,771 | +3.328 | −3.002 |

The per-row gain and loss are within 10 % of each other. **A hard "predict 87,300" rule therefore needs
precision > 0.500 to break even**, and the wide rule's measured precision is 0.379 (CI 0.227–0.560).

### 5.2 Oracle ceilings (a bound, not a plan)

| k rollover rows among the 11, perfectly predicted | LB after | gain |
|---|---|---|
| k = 1 (cheapest / costliest row) | 284.3556 / 276.0168 | +0.003 / +8.342 |
| k = 2 | 281.0858 / 267.7165 | +3.273 / +16.642 |
| k = 3 | 277.7191 / 259.1683 | +6.639 / +25.190 |
| k = 5 | 269.0976 / 243.1116 | +15.261 / +41.247 |
| k = 7 | 256.2536 / 231.2604 | +28.105 / +53.098 |

The target set is fixed at 11 rows selected before prediction, so this ceiling does **not** grow with a
threshold — but it does grow with k, which is unobservable. The LB itself bounds k: if j of the 11 were
rollovers their SSE alone would be at least 0.054 (j = 2), 0.166 (j = 4), 0.446 (j = 7), 0.832 (j = 10)
and 0.969 (j = 11) of the entire unmatched SSE budget, leaving the other ~5,280 unmatched rows an implied
RMSE of 1,449 / 1,362 / 1,110 / 611 / 262 respectively. **j = 11 is impossible** (it would require the
rest of the unmatched side to score 262, better than the matched side's 218); j ≤ 6–7 is the loose bound
this gives.

### 5.3 The actual expected gain of each treatment

`π` = true fraction of rollovers among the selected ranking rows; `π̂` = the rate fitted on training.

**Rule A — `LIRF ∧ unmatched ∧ ms ≥ 43,200` (11 rows), π̂ = 11/29 = 0.379:**

| π (true) | hard rule "predict 87,424" | mixture mean `π̂·87,300 + (1−π̂)·ms` |
|---|---|---|
| 0.00 | **−54.365 s** | **−8.377 s** |
| 0.10 | −44.343 | −4.113 |
| 0.19 (break-even of the mixture) | −35.090 | −0.236 |
| 0.20 | −34.005 | +0.215 |
| 0.30 | −23.321 | +4.610 |
| **0.379 (= π̂)** | −14.575 | **+8.145** |
| 0.50 | −0.753 | +13.614 |
| 0.80 | +36.929 | +27.711 |
| 1.00 | +65.627 | +37.557 |

**Rule B — `… ∧ ms ≥ 57,600` (4 rows), π̂ = 9/13 = 0.692:**

| π (true) | hard rule | mixture mean |
|---|---|---|
| 0.00 | −9.897 s | −4.778 s |
| 0.30 | −4.046 | −0.684 |
| 0.346 (break-even of the mixture) | −3.135 | −0.049 |
| 0.424 (CI low) | −1.672 | +0.970 |
| 0.50 | −0.078 | +2.078 |
| **0.692 (= π̂)** | +3.790 | **+4.759** |
| 0.873 (CI high) | +7.494 | +7.317 |
| 1.00 | +10.093 | +9.104 |

Facts to read off: the **hard rule is negative-expectation under every measured rate** (−14.6 s for rule A,
+3.8 s for rule B only if π̂ = 0.692 holds exactly, and −9.9 s if it does not). The **mixture mean** is the
RMSE-optimal point prediction under a two-point law and is positive-expectation for any π > π̂/2. Its
expected value is **+8.15 s (rule A)** or **+4.76 s (rule B)**, with the 95 % interval implied by the
training-rate CI of **+1.40 … +16.37 s** and **+0.97 … +7.32 s** respectively, and a floor of
**−8.38 s / −4.78 s** if no 2026 candidate is a rollover.

Two facts that shrink these numbers further: (i) `QFAA7006` (ms 87,484, v10 pred 88,042) is worth ±0.003 s
either way — it is already predicted correctly under both hypotheses, so rule B is really a bet on three
rows; (ii) `ITYTY672`'s only training precedents (2 rows, same flight number, same runway 16R, same
destination SBGL) are **both schedule-fills**, which argues its own π is below the band average.

Finally, this is not a new idea: `cond-mean-vs-median.md` §3.7 already computed the identical mixture
arithmetic on the same 11 rows at π_filled = 0.6207 and obtained ΔLB = −8.20 s against a 305.03 baseline.
The present section adds only (a) the v10 baseline, (b) the `ms ≥ 57,600` refinement and its significance
test, and (c) the sensitivity to π, which is the quantity that decides the sign.

---

## 6. Risk

* **The rule is fitted to 29 rows.** Precision 0.692 on the high-`ms` band has a Wilson 95 % CI of
  [0.424, 0.873]; precision 0.379 on the wide band has [0.227, 0.560]. Break-even for the hard rule is
  0.500 — inside both intervals. Break-even for the mixture rule is π̂/2, below both lower bounds, but
  only if π̂ itself transfers.
* **The 2026 mechanism is unverified.** No labelled 2026 departure exists; the 344,693 labelled 2026
  arrivals contain 0 rollover-shaped rows, which is what one expects (0.66) whether the mechanism is live
  or dead. If the second source's date handling was fixed between 2025 and 2026, **π = 0** and the
  mixture rule costs **−8.38 s (rule A) / −4.78 s (rule B)**, and the hard rule costs **−54.67 s**.
* **False-positive cost is essentially equal to the true-positive gain** (§5.1): −7.4 to −8.0 s for each
  of the five lowest-`ms` candidates, −3.0 to −5.7 s for the rest. On the 11-row set, firing the hard
  rule with j = 0…3 real rollovers costs −54.7 / −41.0 / −26.8 / −11.7 s (best case) and
  −54.7 / −54.7 / −49.0 / −43.6 s (worst case).
* **The final evaluation may use a different row set.** Everything above is computed on the 344,841 rows
  of the current public scoring set. A rule whose entire expected effect comes from 3–4 rows in that set
  has no measurable transfer guarantee: on a different row set the realised gain is a fresh draw from the
  same lottery, and the four rows whose identity drives the number could simply be absent.
* **The realised gain is discrete.** Even at π = π̂, the outcome is one of 2¹¹ (rule A) or 2⁴ (rule B)
  values spanning −8.38 … +37.56 s and −4.78 … +9.10 s. The expectation is not a number the leaderboard
  will show.

---

## 7. Facts vs inference

**Facts (measured by the script):** every timestamp, stand timeline, arrival row and count in §1;
`HB_taxi == HM_taxi` on all 16 rows; `BLOCK + 1 day` follows the last preceding same-stand arrival on 15
of 16 rows; the six independent clocks on `AEZ146R` all read 07-14; the LFPG arrival of `ITY332` at
02:42:00 on 06-26 and the route's normal airborne times; all precision/recall figures in §3 and the
Fisher p = 0.00274; the ranking month split and the 11/4 candidate counts; the v10 predictions; every LB
figure in §5; 0 rollover-shaped rows among the 344,693 labelled 2026 arrivals against an expected 0.66.

**Inference (not established by this data):** that the corrupted field is always the one supplied by the
"second" source; that `y − 86,400` is the true taxi time for the 13 BLOCK-shifted rows (it is the only
value consistent with the stand timeline, but the true off-block is not observed); that row 9's BLOCK is
an off-block rather than a schedule-fill carrying a wrong date; that the 2025 mixing rate π̂ applies to
the 2026 candidates; that the `ms` gradient inside the 29-row population is causal rather than a small-
sample association; that the ranking set contains rollovers at all.
