# Row-level forensics on the 141 unexplained top-200 OOF rows (2026-09-18)

Script: `docs/data_audit_2026-09-18/scripts/forensics-top200.py`
(`POLARS_MAX_THREADS=1`, no model training, ~6 min wall, writes nothing).
Inputs, read-only: `experiments/oof_main-48c19bc830.parquet`, `data/work/feat_train.parquet`,
`data/work/feat_rank.parquet`, all 12 `data/raw/training_*.parquet` (4,167,797 movement rows,
2,085,047 DEP + 2,082,750 ARR), `outputs/main.parquet`, `outputs/submit_v10_reparam.parquet`.
Every number below was produced by that script.

Notation (plan.md 2026-09-16 §0): `y` = `TAXITIME_SEC_mvt` = MVT − BLOCK; `proxy` = `mvt_minus_aobt3`
= MVT − AOBT_3; **`d` = `y` − `proxy` = AOBT_3 − BLOCK**; `ms` = `mvt_minus_sched` = MVT − SCHED;
`err` = `y` − `oof`; unmatched = `proxy` null. Two new row-level statistics are used throughout:
**`nb_med`** = median `y` of the other departures at the same airport whose MVT is within ±1 h, and
**`fn_med`** = median `y` of the same `FLIGHT_mvt` at the same airport over the whole training year.

Reproduction of the frame: 672,428 scored rows, total SSE 7.52751e10, RMSE 334.582. The top 200
squared errors carry 54.81 %; 9 are label > 72,000 s, 50 are LIRF with |BLOCK − SCHED| ≤ 10 s; the
remaining **141 carry 14.84 %** and 130 of them are unmatched. Per-airport shares match the brief
exactly (LFPG 18 / 5.85 %, LIRF 70 / 4.11 %, EHAM 10 / 1.37 %, EDDF 9 / 1.07 %, LEBL 5 / 0.79 %,
EGLL 12 / 0.59 %, LSZH 6 / 0.46 %, EDDM 4 / 0.36 %, LTFM 6 / 0.21 %, LEMD 1 / 0.03 %).

---

## The kinds

| # | kind | n | share of total OOF SSE | identifiable from columns non-null in the ranking set | expected leaderboard gain |
|---|---|---|---|---|---|
| K1 | `BLOCK` carries a timestamp from another day; `y` is not a taxi time | **1** | **4.349 %** | **no** | **0.00 s** |
| K2 | stale `SCHED`, non-LIRF: model B extrapolates on `ms`, the label is an ordinary taxi | **36** | **4.751 %** | **yes** (precision 0.88 at `ms` ≥ 28,800 s) | **0.00 s**; oracle ceiling **+0.28 s** |
| K3 | LIRF unmatched, the *non-filled* branch of a two-valued label | **64** | **3.910 %** | **no** (rule precision 0.20–0.32; the visible cell is unbiased) | **0.00 s** |
| K4 | label inflated because `BLOCK` precedes the real off-block — unmatched, inferred | **28** | **1.437 %** | **no** | **0.00 s** |
| K5 | the same, matched, with `d = AOBT_3 − BLOCK` = 3,176–7,622 s measured directly | **9** | **0.294 %** | **no** | **0.00 s** |
| K6 | model-B output of 4,900–6,100 s with no large `ms` and no visible driver | **3** | **0.097 %** | **no** | **0.00 s** |
| | total | **141** | **14.838 %** | | |

Nothing in the 141 is left unclassified. The one row I cannot explain *mechanically* is K1 (I can say
what the timestamp is not, not where it came from); the three K6 rows are unexplained in the weaker
sense that no column in `feat_train` distinguishes them from their neighbours.

---

## 0. The one structural fact the rows produce on their own

Before any hypothesis: the 141 split exactly in two by the sign of `err`, and that split coincides
with a gap in `y / nb_med` — the row's label divided by the concurrent airport median — with **no
overlap**:

| group | n | min `y/nb_med` | median | max | SSE share |
|---|---|---|---|---|---|
| over-predicted (`err` < 0) | 103 | **0.329** | 0.946 | **1.996** | 0.0876 |
| under-predicted (`err` > 0) | 38 | **2.385** | 5.991 | **69.458** | 0.0608 |

**Fact.** Every over-predicted row has a label that is *ordinary* for its airport and hour (never
more than 2.0× the concurrent median); every under-predicted row has a label 2.4–69× that median.
There is no row between 2.0 and 2.4.

**Inference.** The 141 are two disjoint failures, not one tail: in 103 of them the *prediction* is the
anomaly and the label is normal; in 38 of them the *label* is the anomaly and the prediction is close
to what that flight number normally does. Kinds K2/K3/K6 are the first group; K1/K4/K5 the second.

---

## K1 — `BLOCK` from another day (1 row, 4.35 % of all OOF squared error)

This single row is 29 % of the 141's squared error and the largest of the whole set.

**Worked example.** `MVT_ID 182377662`, LFPG → GMMX, `EJU42AY`, stand D06, runway 26R, A320,
unmatched. `SCHED` 2025-01-23 14:35:00, `MVT` 2025-01-23 15:09:03 → `ms` = 2,043 s (entirely normal).
`BLOCK` **2025-01-22 22:58:57** — the previous evening. `y` = 58,206 s. `oof` = 992.

The same flight number flew from LFPG **56 times** in Jan–Mar 2025 (script §2c). Its taxi times are
414–1,439 s, median 836, and on every other day `BLOCK` sits 10–20 min before `MVT` on the same date.
On the 23rd, and only on the 23rd, it is also the only day the flight used a D-pier stand rather than
B/G. The concurrent LFPG hour was normal (71 departures, median 838 s, p90 1,148 s, 1.4 % over 1 h).
The model's 992 s is within 156 s of that flight's own median.

**How rare.** Over the whole training year, rows with `y` > 21,600 s while |`ms`| < 7,200 s number
**5 of 2,085,047** (LFPG 84,240 and 58,206; LIRF 31,268 and 25,018; LSZH 87,341); at `y` > 10,800 s
the count is 9. Two of the five are scored, and they carry 13.57 % of total OOF SSE between them —
but one of those two (84,240 s) is a day-rollover row and belongs to the other agent, leaving this
one row at 4.35 %.

**(a) Identifiable?** No. The defining fact is a property of `BLOCK_TIME_UTC_mvt`, which is null on
all 344,841 ranking rows, and `ms` — the only visible quantity — is 2,043 s, i.e. inside the bulk of
the distribution. No rule over the ranking-visible columns can separate this row from the ~200,000
ranking rows with a similar `ms`.
**(b) Does the model handle it?** The model predicts 992 s, which is the right answer for the flight
and the wrong answer for the label. Removing the group mean of a one-row group is meaningless.
**(c) Distinguishable from a large residual?** Yes, and this is the one place in this report where
that question has a clean answer: the kind is defined by a *timestamp inconsistency* (BLOCK on a
different calendar day from MVT with a normal SCHED), not by the size of `err`, and the 5-row
population is identified without reference to `oof`.
**(d) Ranking count.** 5 rows per 2,085,047 → an expected **0.8 rows** in 344,841 ranking rows; the
count is not observable because `BLOCK` is absent.
**(e) Ceiling.** With an expected 0.8 rows and no rule, **0.00 s**. For scale: if exactly one such row
existed in the ranking set with `y` = 58,206 and the model predicted its flight's usual 1,000 s, it
alone would contribute √(57,206²/344,841) ≈ 97 s² to the MSE, i.e. ≈ 0.17 s of leaderboard, and only
an oracle could remove it.

---

## K2 — stale `SCHED` at non-LIRF airports (36 rows, 4.75 %)

**Worked example.** `MVT_ID 199363411`, EDDF → LMML, `LH8300`, stand F225, A321, unmatched.
`BLOCK` 2025-11-23 20:55:24, `MVT` 2025-11-23 21:06:25 → `y` = **661 s**. `SCHED` **2025-11-22
20:20:00** — one day and 35 min before the block. `ms` = 89,185 s. `oof` = **16,880**.
The same flight number flew 76 times from EDDF with median taxi 659.5 s; the concurrent EDDF hour had
84 departures with median 1,275.5 s. The label is unremarkable in both references; the prediction is
25× the flight's own median.

Four of the 36 have `SCHED` exactly one day early (`BLOCK − SCHED` in [86,400, 93,600] s); across the
36, `BLOCK − SCHED` runs 3,780 → 151,090 s and `ms` runs 4,360 → 151,639 s, with no day structure
outside those four. Airports: EHAM 7, LFPG 7, EDDF 7, EGLL 5, LEBL 4, LSZH 3, EDDM 2, LEMD 1.

**(a) Identifiable — yes.** The rule uses `mvt_minus_aobt3 is null` (the model's own routing rule;
`aobt3_null` is a feature, null rate 0 on ranking), `ADEP_mvt` (null rate 0) and `mvt_minus_sched`
(null rate 0). Target for the sweep = "the model over-predicts by more than 3,000 s", 146 of 672,428
scored rows, base rate 0.000217:

| `ms` ≥ T | flagged | precision | recall | bias | RMSE | bias share of MSE | SSE share |
|---|---|---|---|---|---|---|---|
| 7,200 | 1,837 | 0.0169 | 0.212 | −76.4 | 1,624.2 | 0.0022 | 0.0644 |
| 14,400 | 164 | 0.1402 | 0.158 | −1,381.5 | 4,329.5 | 0.1018 | 0.0408 |
| 21,600 | 58 | 0.3966 | 0.158 | −4,063.1 | 7,215.0 | 0.3171 | 0.0401 |
| **28,800** | **25** | **0.8800** | 0.151 | **−9,570.9** | 10,842.3 | **0.7792** | 0.0390 |
| 43,200 | 21 | 0.9048 | 0.130 | −10,203.8 | 11,306.5 | 0.8144 | 0.0357 |
| 86,400 | 6 | 1.0000 | 0.041 | −12,762.1 | 13,311.9 | 0.9191 | 0.0141 |

**(b) Bias, not spread.** At T = 28,800 the bias share of MSE is **0.78**; at T = 86,400, **0.92**.
The label is flat across the whole non-LIRF unmatched `ms` range — mean `y` by band: 763 (0–1,800),
1,205 (5,400–7,200), 1,451 (10,800–14,400), 1,184 (14,400–21,600), 1,254 (21,600–43,200), 1,261
(43,200–86,400), 1,006 (≥ 86,400) — while mean `oof` climbs to 13,769. The schedule-fill rate in
every non-LIRF band at `ms` ≥ 10,800 is **0.000**.

**(c) Is it just where the residual is large?** Partly, and the answer moves with T. At T = 21,600
the visible cell has bias −4,063 (bias share 0.317); split by the label, the 23 rows inside the 141
have bias −10,591 and the other 35 rows have bias **+226** with RMSE 1,270 — so at that threshold the
harm is in the label-selected part. At T = 28,800 the cell is 88 % kind and its own bias share is
0.78, i.e. the defect becomes a property of the *visible* cell. This is the opposite of the 39,224-row
precedent, where the best visible flag had mean `err` = −3.1 s and bias share 0.0001
(`review-far-rows.md` §4).

**(d) Ranking count.** non-LIRF unmatched: `ms` ≥ 7,200 → 1,540 rows; ≥ 14,400 → 280; ≥ 21,600 → **71**
(EHAM 41, EDDF 6, LFPG 6, EGLL 5, LEBL 5, LSZH 4, EDDM 2, LTFM 2); ≥ 43,200 → 14; ≥ 86,400 → 1.

**(e) Ceiling — the model already stopped doing this on the ranking set.** On those 71 rows:
`outputs/main.parquet` (v7) predicts median 1,524, p90 5,865, max 12,193, with 4 rows ≥ 7,200 s;
`outputs/submit_v10_reparam.parquet` (v10) predicts median 1,499, **p90 2,716, max 3,031, none ≥ 7,200 s**.
Metric arithmetic (LB 284.3584 → total SSE 2.78837e10; matched 339,551 at 218 → implied unmatched
RMSE 1,490.2, which checks out): if every one of the 71 labels were 1,200 s — the cell's training mean
— an oracle on them would remove 5.5073e7 from v10's SSE, i.e. **LB 284.0774, gain +0.281 s**. That is
the *whole* error v10 carries on the cell under the most favourable label assumption, not a gain from
any correction. Under v7 the same arithmetic gives +3.346 s, which is the size of the defect the v10
change already removed. Honest expected gain from doing anything further here: **0.00 s**.

---

## K3 — LIRF unmatched, the non-filled branch (64 rows, 3.91 %)

**Worked example.** `MVT_ID 192628836`, LIRF, `RAM941J`, stand 608, runway 25, `AIRCRAFT_TYPE_mvt`
null, unmatched. `SCHED` 2025-07-29 15:40, `BLOCK` 17:54:00, `MVT` 18:00:00 → `y` = **360 s**,
`ms` = 8,400 s, `BLOCK − SCHED` = +8,040 s. `oof` = **8,626**. Concurrent LIRF hour: median 961 s.
The flight pushed back 2 h 14 min after its schedule and then taxied for six minutes.

All 64 are LIRF, all unmatched, **all 64** have `AIRCRAFT_TYPE_mvt` null, `ms` runs 3,778 → 20,878 s,
median 9,927. Their labels are 351–3,116 s, median 1,091; their predictions 4,980–19,659, median 6,960.

**(a) Identifiable — no.** The same visible cell also contains the schedule-filled rows, where
`BLOCK` = `SCHED` and therefore `y` = `ms` exactly. The rule `unmatched ∧ ADEP = LIRF ∧ ms ≥ T`:

| `ms` ≥ T | flagged | precision for the kind | fill rate | frac `y` ≤ 1 h | bias | RMSE | bias share of MSE |
|---|---|---|---|---|---|---|---|
| 3,600 | 498 | 0.201 | 0.434 | 0.540 | **+293.2** | 7,422.8 | **0.0016** |
| 7,200 | 239 | 0.322 | 0.540 | 0.418 | +927.3 | 10,463.2 | 0.0079 |
| 14,400 | 55 | 0.218 | 0.782 | 0.073 | +8,714.2 | 20,419.8 | 0.1821 |
| 21,600 | 29 | 0.172 | 0.724 | 0.000 | +16,117.2 | 27,542.4 | 0.3424 |

**(b)/(c) The model is unbiased on the visible cell.** At T = 3,600 the bias is **+293 s in an RMSE of
7,423** — 0.16 % of MSE. The label in that cell is two-valued (43 % at `y` = `ms`, 54 % at `y` ≤ 1 h)
and the prediction sits between the two branches; the 64 rows are the low branch, selected by the
label. Removing the cell's mean changes its RMSE by 6 s in 7,423.
**This is the `review-far-rows.md` Q1 pattern exactly**: a set that looks like a category and is a
restatement of which side of a mixture a row landed on. Note the sign flip: at T ≥ 14,400 the same cell
is *under*-predicted (bias +8,714, +16,117), because there the filled branch dominates. There is no
threshold at which the cell has a consistent sign.
**(d)** 348 LIRF unmatched ranking rows with `ms` ≥ 3,600.
**(e)** **0.00 s.** v10 already reparametrises this cell (`unmatched_sched_reparam_airports: ["LIRF"]`,
`experiments.jsonl` line 43) and predicts median 4,888 / p90 14,159 on those 348 rows against v7's
4,234 / 13,584 — a deliberate change whose effect is not measurable without 2026 labels.

---

## K4 / K5 — the label is not a taxi time (28 + 9 rows, 1.44 % + 0.29 %)

**Worked example (K5, measured).** `MVT_ID 182377216`, LFPG → KATL, `DAL83`, stand K26, A339, matched.
`BLOCK` 2025-01-29 08:23:58, `AOBT_3` **10:31:00**, `MVT` 10:46:41. `y` = 8,563 s but
`proxy` = MVT − AOBT_3 = **941 s**, so `d` = AOBT_3 − BLOCK = **7,622 s**. The flight number's median
taxi at LFPG over 359 flights is 1,208 s; the concurrent LFPG hour's median was 789 s with 0 % over
1 h. `oof` = 2,503.

All 9 matched rows in the group show the same thing — `d` = 3,176 / 4,941 / 5,156 / 5,340 / 5,704 /
5,996 / 6,543 / 6,844 / 7,622 s (mean 5,702) while `proxy` is 4 / 715 / 833 / 836 / 941 / 1,135 /
1,204 / 1,260 / 7,796 s (median 941). In eight of the nine the real taxi from `AOBT_3` is ordinary;
the ninth (EHAM `OR1507`, `proxy` 7,796) has a genuinely long ground time as well.

The 28 unmatched rows (K4) have no `AOBT_3` and cannot be checked the same way; they carry the same
signature — `y / nb_med` = **2.38–17.24**, and `y / fn_med` = 1.0–44.8 with median 5.7 (the three
values at 1.0 are the rows whose flight number appears only once in the training year, so the row is
its own reference).

**A burst structure, and what it is made of.** Grouping matched scored rows by airport × 3-hour block
(7,804 blocks with ≥ 20 rows), **17 blocks** have ≥ 15 % of their departures with `d` > 1,800 s. They
hold 1,017 rows and 0.91 % of total SSE. In 15 of the 17 the concurrent *real* taxi is untouched:
median frac(`proxy` > 1 h) inside them is **0.000**, the same as the median over all blocks. Four of
the under-predicted rows sit in LTFM 2025-07-26 12:00–16:30, where `y` p90 rises to 2,471 from 1,672
the day before while `proxy` p90 is **1,205 vs 1,205** and frac(`proxy` > 1 h) is 0.010 vs 0.005; three
sit in LFPG 2025-12-05 09:00–14:00, where `y` p90 is 3,599 against 1,621 the day before while
frac(`proxy` > 1 h) is **0.000 on both days**. By contrast LFPG 2025-11-22 09:00–14:00 (two rows) is a
real event: `proxy` p90 4,141 and frac(`proxy` > 1 h) 0.129 against 0.000 the day before, with `vis_km`
2.0 and 0.3. EGLL 2025-07-30 (three rows) is mixed (`proxy` p90 2,457 vs 1,919; frac `d` > 1,800 s
0.200 vs 0.006). Episode rate by airport: EDDM 0.83 % of blocks, LIRF 0.59 %, EGLL 0.27 %, LTFM 0.21 %,
LEBL 0.12 %, LFPG 0.12 %, EHAM / EDDF / LSZH / LEMD 0.00 %.

**Fact.** In those blocks the quantity that moves is `y` − `proxy`, not `proxy`.
**Inference.** The bursts are a property of the `BLOCK` feed, not of the airfield: an airport-wide
taxi event would move `proxy` too. Whether `BLOCK` is early or `AOBT_3` late is not decidable from
these columns.

**(a) Identifiable — no.** Defining the kind over all scored rows as `y` ≥ 3 × the local 2-hour median
(2,002 rows, base rate 0.00298, **47.60 % of total SSE**), the AUC of every prediction-time-visible
column is: `|mvt_minus_sched|` 0.9037, `|mvt_minus_iobt|` 0.8835, `|mvt_minus_lobt|` 0.8831,
`|mvt_minus_eobt1|` 0.8753, `|aobt3_minus_sched|` 0.8087, `queue_ahead` 0.7955,
`|aobt3_minus_eobt1|` 0.7913, then a gap to 0.64 and below. The best one converts to nothing usable:

| `|mvt_minus_sched|` ≥ q | flagged | precision | recall |
|---|---|---|---|
| 0.50 | 336,249 | 0.0059 | 0.9905 |
| 0.90 | 67,250 | 0.0188 | 0.6329 |
| 0.99 | 6,725 | 0.0211 | 0.0709 |
| 0.999 | 673 | 0.0416 | 0.0140 |

Best precision anywhere in the sweep is 0.042 against a base rate of 0.003 — lift 14, and 96 % false
positives. For the matched half the defining quantity is `d` = `y` − `proxy`, which is the label minus
a visible feature: selecting on it is selection on the target (`review-far-rows.md` §2, Fact 2).

**(b)/(c) The decisive control.** On the label-selected kind: n 2,002, bias +1,456.6, RMSE 4,230.4,
bias share of MSE 0.1186 — 88 % spread. On the best visible approximation (q90 of `|ms|`): n 67,250,
bias **−6.0**, RMSE 797.1, bias share of MSE **0.0001**. The model is unbiased to 6 s in 797 on every
row one can actually select. This reproduces `review-far-rows.md` §4 on a different set.

**(d)** Prevalence 0.00298 → ≈ **1,027** ranking rows carry the kind, none of them nameable.
**(e)** Perfect oracle on 1,027 ranking rows at the kind's own MSE (1.79e7) gives LB 166.07, a gain of
**+118.29 s**. That number is vacuous for the reason `review-far-rows.md` §3 and §5 give: it is monotone
in the size of the set one chooses to define, it is the arithmetic of deleting label-selected error,
and no selection rule exists. Honest expected gain: **0.00 s**.

---

## K6 — three model-B outputs with no visible driver (3 rows, 0.097 %)

`MVT_ID 193071742` (LEBL, `VLG17XA`, `ms` = 1,455, `y` = 741, `oof` = 6,095);
`MVT_ID 182647144` (LIRF, `WMTMT406`, `ms` = 358, `y` = 716, `oof` = 5,797);
`MVT_ID 192619355` (LIRF, no `FLIGHT_mvt`, no stand, `ms` = **−26,577**, `y` = 1,387, `oof` = 5,656).

All three are unmatched with an ordinary `ms`, so K2's mechanism does not apply. Their feature rows are
sparse in the way model B's inputs can be: the LIRF 03:00 row has `rwy_headway_prev_s` = 32,998 s and
all four `nbr_*_30m/60m` neighbour statistics NaN (no departure in the preceding 9 h); the second LIRF
row has `STAND_mvt`, `FLIGHT_mvt` and `queue_ahead` all null and a negative `ms`.

**Size of the phenomenon.** Of the 2,245 scored unmatched rows with `ms` < 3,600 s, the model predicts
≥ 4,000 s on **13**; their labels are 716…8,103 (five of the thirteen genuinely are above 3,300 s);
together they carry **0.17 %** of total SSE.
**(a)–(e).** No rule was found that separates the 13 from the other 2,232; the group is too small and
too mixed (5 of 13 labels are in fact large) for a bias estimate to mean anything; expected ranking
count ≈ 6.7 rows; ceiling **0.00 s**.

---

## LFPG in detail (18 rows, 5.85 % — the highest damage per row)

| kind | n | SSE share | rows |
|---|---|---|---|
| K1 | 1 | **4.349 %** | `EJU42AY` `y` 58,206 `oof` 992 |
| K2 | 7 | 1.106 % | `AFR718` (`ms` 89,156), `AFR89CA` (88,502), `DAH1209` (47,338), `LGL812C` (88,202), `EJU46JM` (11,464), `TVS2XY` (9,782), `ITY333` (7,680) |
| K4 | 7 | 0.283 % | `CES570`, `SAS9230`, `DAL153`, `UPS22`, `AFR6732`, `AFR53QE`, `DLH3F` |
| K5 | 3 | 0.108 % | `DAL83` (`d` 7,622), `AFR1192` (`d` 6,543), `AFR85XG` (`d` 5,340) |

**Fact.** LFPG's damage-per-row lead over LIRF is one row: strike `EJU42AY` and LFPG falls to
1.50 % over 17 rows (0.088 % per row) against LIRF's 4.11 % over 70 (0.059 % per row). LFPG has no
schedule-fill (its seven K2 rows have `BLOCK − SCHED` = 5,340 / 8,877 / 10,316 / 46,681 / 86,458 /
87,237 / 88,504 s, never ≈ 0) and no LIRF-style
two-valued label; its 10 K4/K5 rows cluster into three windows — 2025-11-22 09:00–14:00 (a real
low-visibility event, `proxy` p90 4,141), 2025-12-05 09:00–14:00 (not a real event, `proxy` p90 1,982
and frac(`proxy` > 1 h) 0.000, `d` p90 2,701), 2025-01-17 07:00–12:00 (mixed).

**Inference.** LFPG's position in the per-airport table is not a property of LFPG. It is one corrupt
timestamp plus the same two mechanisms that appear at the other nine airports.

---

## What I could not explain

- **K1's provenance (1 row, 4.35 %).** I can show what `BLOCK` = 2025-01-22 22:58:57 is *not* — not
  this flight's off-block (the flight's other 55 records put `BLOCK` 10–20 min before `MVT` on the
  same date), not a rollover of exactly one day (the offset is 16 h 10 min), not a schedule artefact
  (`SCHED` is correct to 34 min), and not the previous arrival at stand D06 (there is none in the
  24 h before). Where the value came from is not recoverable from these tables.
- **K4's mechanism on the unmatched side (28 rows, 1.44 %).** `AOBT_3` is absent, so the `d` evidence
  that settles K5 is unavailable; the classification rests on `y / fn_med` and `y / nb_med` and is an
  inference, not a measurement.
- **K6 (3 rows, 0.097 %).** No column in the 132 of `feat_train` distinguishes these three rows from
  the 2,232 unmatched rows with a similar `ms` on which the model predicts normally.
- **Whether `BLOCK` or `AOBT_3` is the wrong clock** in K4/K5 and in the 17 burst blocks. Both readings
  fit every number here.

---

## Facts vs inference

**Facts.** The frame reproduction and per-airport table; the 103/38 split and its exact coincidence
with `y / nb_med` ≶ 2.2 with no row in between; `EJU42AY`'s 56-flight history and the 5-rows-per-year
count of the `y` > 21,600 ∧ |`ms`| < 7,200 population; the non-LIRF unmatched `ms`-band table
(flat mean `y` 763–1,451, fill rate 0.000, bias to −12,762); the K2 precision/recall sweep and its
bias shares; v7 and v10 predictions on the 71 ranking rows and the +3.346 / +0.281 s arithmetic; the
LIRF sweep with bias +293 at bias share 0.0016 and the sign flip at T ≥ 14,400; the nine measured `d`
values; the 17 burst blocks with frac(`proxy` > 1 h) = 0.000 and the four episode/control day pairs;
the kind-B AUC table, precision sweep, and the −6.0 s bias on 67,250 visibly flagged rows; every
ranking count; every ceiling figure.

**Inference.** That the 103/38 split is two failures rather than one tail. That `BLOCK` in K1 came
from a different record (nothing rules out a 16-hour pushback that then taxied 16 h; the flight's own
history and the absence of any such row elsewhere in 2,085,047 make it implausible). That the 28 K4
rows share K5's mechanism. That the 17 burst blocks are a `BLOCK`-feed phenomenon rather than an
airfield one (the `proxy` distribution is unchanged; a real taxi event would move it). That K3 is a
restatement of which branch of a two-valued label a row landed on (supported by bias share 0.0016 on
the visible cell and by the sign flip, not proved). That the honest gain is 0.00 s for K1, K3, K4, K5
and K6 — no selection rule exists for any of them — and 0.00 s for K2 because v10 already predicts
median 1,499 / max 3,031 on the cell, leaving +0.281 s of *oracle* room under the most favourable
label assumption. No correction was fitted and no model was trained.

**Not determined here.** The 2026 labels, and therefore every statement about the ranking set that is
not a count or a prediction quantile. Whether K1's population is stationary — 5 rows in one year is
too few to estimate a rate for two months.
