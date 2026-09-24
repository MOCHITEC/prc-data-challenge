# The missing off-block clock: what model B loses, and whether anything stands in for it (2026-09-24)

**Script (read-only, no submission, nothing written to `src/`, `configs/`, `outputs/` or `data/`):**
`docs/competitor_research_2026-09-21/scripts/missing_clock.py`, run as
`POLARS_MAX_THREADS=2 .venv/bin/python docs/competitor_research_2026-09-21/scripts/missing_clock.py [S1 … S7]`.
Every number below carries the section (`S1`…`S7`) that prints it. Sections S4–S6 fit models; the whole
file is about 35 minutes on a loaded 4-CPU box, S1/S2/S3/S7 alone about two.

**Inputs.** `data/work/feat_train.parquet` (2,085,047 departures), `data/work/feat_rank.parquet` (344,841
scored departures), `experiments/oof_main_reparam-bc60017878.parquet` + its `_folds` sidecar
(672,428 OOF rows over folds jan/jul/forward; `jul_fwd` excluded as elsewhere).

**Notation.** `y` = `TAXITIME_SEC_mvt` = MVT − BLOCK. `proxy` = `mvt_minus_aobt3` = MVT − AOBT_3, the
measured off-block clock model A is handed (`src/prc/train.py:40`, `:405-419`). `g` = `y − proxy` =
AOBT_3 − BLOCK, the quantity model A's *tree* actually fits. `ms` = `mvt_minus_sched` = MVT − SCHED.
`bs` = BLOCK − SCHED = `ms − y`, label-side and therefore invisible at prediction time.
**Unmatched** = `proxy` is null, which is the routing rule the code uses (`src/prc/train.py:398`).

---

## 0. The answer in six sentences

1. **What is lost is the whole NM flight-plan block, not just the measured clock, and the measured
   clock is the smaller half of it.** Denying a matched-row model everything `_flt` costs
   **222.76 → 292.53** on the same rows and folds; denying it only the AOBT_3-derived columns while
   keeping the *filed* clocks costs **222.76 → 253.50**. So of the 69.8 s, about 31 s is the measured
   off-block and about 39 s is EOBT_1 / LOBT / IOBT and the operator categoricals. [S] `S4`
2. **The blindness is a 1.3× penalty, not a 7× one, and the rest of the lane's error is composition
   and sample size.** Reweighted to the *scored* unmatched lane's airport × `ms` mix: sighted 378.4,
   blind 554.0, blind-and-trained-on-19k-rows 616.6. At the nine airports other than LIRF the
   counterfactual predicts **591.4** and the lane actually measures **569.1**. There is no residual
   mystery to explain. [S] `S4`, `S7`
3. **The substitute for the clock already exists in the feature set and is already doing its job.**
   Removing the two neighbour off-block clocks (`nbr_mvt_minus_aobt_mean_30m/60m`) from the blind
   model costs **+5.60 s** on matched rows and **+14.94 s** reweighted — the single largest
   contribution of any column tested here. [S] `S5`
4. **merry-quicksand's shipped "congestion witness" is a slightly better form of the same idea and we
   already hold nine tenths of it.** Added on top of our rolling means it is worth −3.00 / −1.78;
   substituted *for* them it is worth more than they are (553.08 against 554.03, both against 568.96
   with neither). [S] `S5`
5. **Four new candidates built for this note — surface occupancy, neighbours' actual-versus-scheduled
   pushback, the unmatched density of the last hour, and the same stand's previous departure — are at
   or just outside the noise floor of the instrument** (+0.46 … −1.80 against a noise control of
   +0.07), and **none of them survives conditioning against the OOF residual**: every partial
   Spearman is ≤ 0.038 in magnitude against a Gaussian control at 0.007. [S] `S5`, `S6`
6. **Price.** Restoring full sight to the 4,907 non-LIRF scored unmatched rows is worth
   **−4.9 to −5.5 s** of board RMSE and nothing more; what the measured columns actually recover is
   **−0.4 s** (`stand_arrival_gap`, already adopted today inside a −3.62 s feature) plus **−0.2 s**
   (witness + the four new). The old **−7 s** figure is the right order of magnitude for the *lane* but
   **22.5 of the lane's 31.7 s sits at LIRF**, whose mechanism is the schedule fill, not the clock —
   and that route closed this morning at AUC 0.9201. [S] `S7`, `results/P1-stage2-gate-FAILED.md`

**[X] The honest headline: the clock cannot be substituted, the lane's error at the nine non-LIRF
airports is very close to irreducible at this feature set, and the ceiling on the whole idea is about
five seconds.**

---

## 1. What actually remains visible on an unmatched scored row (task item 2)

`S1`. Of 344,841 scored departures, **5,290 are unmatched (1.5341 %)**:

| ADEP | EHAM | LFPG | LTFM | LSZH | LEBL | EDDF | LIRF | EGLL | EDDM | LEMD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scored unmatched | 1,206 | 734 | 721 | 659 | 442 | 421 | 383 | 248 | 238 | 238 |

Query: `feat_rank.group_by("ADEP_mvt")` on `mvt_minus_aobt3.is_null()`.

### 1.1 The 33 columns that go null together

`S1` — columns null on **100 %** of unmatched scored rows, and null on 0 % of matched ones:

* 16 raw NM fields: `LOBT_flt`, `CALLSIGN_flt`, `ADEP_flt`, `ADES_flt`, `ADES_FILED_flt`, `IOBT_flt`,
  `FLIGHT_RULE_flt`, `AIRCRAFT_TYPE_flt`, `EOBT_1_flt`, `ARVT_1_flt`, `AOBT_3_flt`, `ARVT_3_flt`,
  and the four categoricals `WK_TBL_CAT_flt`, `AIRCRAFT_OPERATOR_flt`, `MARKET_SEGMENT_flt`,
  `FLIGHT_TYPE_flt`.
* 8 derived clocks + their 7 `_slog` copies: `mvt_minus_aobt3`, `mvt_minus_eobt1`, `mvt_minus_lobt`,
  `mvt_minus_iobt`, `lobt_minus_eobt1`, `aobt3_minus_eobt1`, `aobt3_minus_sched`, `aobt3_null`.
* 4 candidate columns built from NM times: `stand_aobt_turn`, `eobt_schedule_revision`,
  `delay_neighbor_deviation`, `planned_flight_s`.

`LOBT_flt is null` ⇔ `mvt_minus_aobt3 is null` holds **exactly** on the ranking frame [S] `S1`, so the
brief's premise is confirmed: this is one event, not several. The one wrinkle: **7 unmatched scored
rows carry a non-null `FLIGHT_ID_mvt`** [S] `S1` — a movement that has a flight id but no usable NM
record. The routing key is AOBT_3, not the join, so they route to model B anyway.

### 1.2 What survives, and the three dead columns model B carries

**At nine of ten airports the movement record is untouched.** `S1`, null rate on unmatched scored rows:

| ADEP | n | AIRCRAFT_TYPE | STAND | RUNWAY | ADES | FLIGHT | SCHED | ms | stand_arrival_gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EHAM | 1,206 | 0.0000 | 0 | 0 | 0.0000 | 0 | 0 | 0 | 0.3259 |
| LFPG | 734 | 0.0000 | 0 | 0 | 0.0000 | 0 | 0 | 0 | 0.1676 |
| LTFM | 721 | 0.0055 | 0 | 0 | 0.0055 | 0 | 0 | 0 | 0.1775 |
| LSZH | 659 | 0.0015 | 0 | 0 | 0.0000 | 0 | 0 | 0 | 0.2337 |
| LEBL | 442 | 0.0000 | 0 | 0 | 0.0000 | 0 | 0 | 0 | 0.1154 |
| EDDF | 421 | 0.0000 | 0 | 0 | 0.0048 | 0 | 0 | 0 | 0.1734 |
| **LIRF** | **383** | **1.0000** | 0 | 0 | 0.0000 | **0.0601** | 0 | 0 | 0.1540 |
| EGLL | 248 | 0.0000 | 0 | 0 | 0.0000 | 0 | 0 | 0 | 0.1653 |
| LEMD | 238 | 0.0000 | 0 | 0 | 0.0000 | 0 | 0 | 0 | 0.2479 |
| EDDM | 238 | 0.0000 | 0 | 0 | 0.0000 | 0 | 0 | 0 | 0.1597 |

**[S] "Unmatched" is two populations and the split is exact.** LIRF loses `AIRCRAFT_TYPE_mvt` on
**100 %** of its 383 scored unmatched rows (99.93 % of its 1,488 training ones) and `FLIGHT_mvt` on
6.0 %; every other airport loses aircraft type on at most 0.55 %. `STAND_mvt`, `RUNWAY_mvt`,
`SCHED_TIME_UTC_mvt` and `mvt_minus_sched` are **0 % null at all ten**. `stand_arrival_gap`'s null
rate on unmatched rows is within 3 points of its matched rate at every airport except EHAM
(0.3259 vs 0.1534), i.e. it is a property of the stand link, not of the lane.

`S1`, degenerate columns. Model B's prefix rule (`src/prc/train.py:400-405`) drops anything starting
with `mvt_minus_aobt3 / aobt3 / lobt_minus / mvt_minus_lobt / mvt_minus_eobt1 / mvt_minus_iobt`, which
leaves **three columns that are constant on every row model B ever sees**: `lobt_null` ≡ 1,
`eobt1_null` ≡ 1, `nm_matched` ≡ 0 (bar the 7 rows of §1.1). **[X] Harmless — a constant column
cannot split — but it means model B's advertised 84 numeric features are 81 live ones.** Also
constant on the *whole* ranking frame, matched rows included, and therefore not an unmatched-lane
fact: `precip_mm`, `temp_x_precip`, `precip_mm_prev1h`, `temp_x_precip_prev1h` [S] `S1`. That is a
weather-cache property worth a separate look; it is out of scope here.

---

## 2. The counterfactual: how much of the lane's error is blindness (task item 1)

### 2.1 Design

`S4`. One fold — **jan** (validate 2025-01, train 2025-02…12) — **matched rows only**, so the two
lanes cannot differ by population. Training side subsampled to **247,547 rows** by a hash of
`MVT_ID_mvt` (13 %); validation is all **152,271** matched jan rows. 400 fixed boosting rounds, one
seed, production LightGBM parameters, encodings fitted on the subsample and joined onto both sides.
Five feature regimes, identical rows, identical folds:

| regime | numeric + categorical | target |
|---|---|---|
| **A** | 99 + 10 (everything, `proxy` last) | `y − proxy`, prediction `proxy + g` |
| **A_plain** | 99 + 10 | `y` directly — isolates the reparameterisation from the information |
| **P** | 92 + 10 — every AOBT_3-derived column removed, *filed* clocks kept | `y` |
| **B_cf** | 84 + 6 — model B's own rule, plus the four `_flt` candidates of §1.1 and the four `_flt` categoricals | `y` |
| **B_small** | same as B_cf, trained on **19,165** rows — model B's order of magnitude (22,470 training rows, `plans/P1-unmatched-lane.md:47`) | `y` |

The subsample and the fixed 400 rounds make every absolute number here about 3 % worse than
production (model A measures 222.76 where the shipped matched OOF is 216.34). **[X] The comparison is
paired and that is what is being read; do not quote these as production numbers.**

### 2.2 The result

`S4`. "Reweighted" = post-stratified so the matched validation rows carry the *scored* unmatched
lane's airport × `ms`-band composition (10 airports × 5 bands, cells with < 20 validation rows
dropped, coverage 1.000):

| model | RMSE, all matched | bias | reweighted to the scored lane | reweighted, non-LIRF |
|---|---:|---:|---:|---:|
| **A** (sighted) | **222.76** | +10.35 | **378.4** | **361.6** |
| A_plain | 223.38 | +10.99 | 385.3 | 369.2 |
| **P** (no measured off-block, filed clocks kept) | **253.50** | +12.45 | 446.6 | 434.4 |
| **B_cf** (blind) | **292.53** | +13.95 | **554.0** | 541.7 |
| **B_small** (blind, 19k rows) | 317.26 | +10.79 | 616.6 | **591.4** |

**And the lane itself.** `S7`, on the pooled OOF: matched 216.34, unmatched **1,632.7**, unmatched
excluding the 33 day-rollover labels **1,008.6**, **non-LIRF excluding them 566.0** (569.1 once
reweighted to the scored composition).

> **569.1 observed against 591.4 predicted by a model that is blind and starved in exactly the same
> way.** [S] `S4`, `S7`

**[X] That closes the question the brief asked.** The unmatched lane's 7.5× is not 7.5× of anything.
It is (i) two corrupted labels, (ii) LIRF's schedule fill, and (iii) at the other nine airports, a
delay-selected population — which is intrinsically about 1.6× harder at equal sight — predicted by a
model that is blind (1.31×) and trained on 1 % of the rows (a further 1.14×). Multiply the three and
you land on the observed number. Model B is not doing badly; it is doing about as well as an
off-block-blind model can.

### 2.3 Where the blindness bites, per airport and per band

`S4`, matched validation rows, A / P / B_cf:

| ADEP | n | A | P | B_cf | B_small | B−A |
|---|---:|---:|---:|---:|---:|---:|
| EDDF | 15,332 | 185.2 | 246.5 | 274.2 | 290.5 | +89.0 |
| EDDM | 11,201 | 190.6 | 228.8 | 274.0 | 307.8 | +83.4 |
| EGLL | 19,292 | 254.1 | 288.1 | 368.6 | 382.8 | **+114.4** |
| EHAM | 18,167 | 191.9 | 258.9 | 308.0 | 328.1 | **+116.1** |
| LEBL | 12,168 | 155.8 | 159.2 | 165.3 | 180.1 | **+9.5** |
| LEMD | 16,478 | 175.4 | 180.8 | 202.8 | 230.0 | +27.4 |
| LFPG | 17,684 | 277.7 | 315.9 | 365.0 | 408.7 | +87.4 |
| LIRF | 11,257 | 298.2 | 302.5 | 333.8 | 371.3 | +35.7 |
| LSZH | 9,556 | 187.5 | 251.3 | 283.4 | 305.0 | +95.9 |
| LTFM | 21,136 | 238.8 | 238.5 | 248.4 | 269.0 | **+9.6** |

**[X] The clock is worth ten times more at EHAM and EGLL than at LEBL and LTFM.** At LTFM the filed
clocks add nothing at all (P 238.5 against A 238.8) — an airport where NM's off-block times are
apparently not informative beyond the schedule. That is a fact about where an unmatched row costs us,
not a lever: `ADEP_mvt` is a categorical model B already splits on.

`S4`, per `ms` band:

| `ms` band | n | A | P | B_cf | B−A | sd(y) |
|---|---:|---:|---:|---:|---:|---:|
| < 1,800 | 105,216 | 158.4 | 168.7 | 180.3 | +21.9 | 302.8 |
| 1,800–3,600 | 33,940 | 280.5 | 329.8 | 376.0 | +95.5 | 497.6 |
| 3,600–5,400 | 7,636 | 401.6 | 479.7 | 564.4 | +162.8 | 735.3 |
| 5,400–7,200 | 2,568 | 395.7 | 482.5 | 652.3 | **+256.6** | 881.6 |
| 7,200–10,800 | 1,849 | 488.1 | 538.0 | 723.4 | +235.3 | 893.4 |
| 10,800–14,400 | 590 | 410.7 | 440.5 | 558.1 | +147.4 | 698.7 |
| ≥ 14,400 | 472 | 388.1 | 446.8 | 535.1 | +146.9 | 606.6 |

**This reproduces the brief's LFPG figure from the other side.** At `ms` 7,200–10,800 the blind model
measures 723.4 where the sighted one measures 488.1 — against the observed LFPG unmatched 899.6 vs
matched 527.1 (`lfpg-unmatched-lane.md` §3). Same ratio, same band, on rows drawn from one population.

`S4`, sliced on `bs` = BLOCK − SCHED, the axis the lane is actually selected on:

| `bs` band | n | A | P | B_cf |
|---|---:|---:|---:|---:|
| < 0 | 45,370 | 186.7 | 202.3 | 222.7 |
| 0–900 | 66,683 | 199.8 | 224.3 | 258.2 |
| 900–1,800 | 20,751 | 259.3 | 305.5 | 353.8 |
| 1,800–2,700 | 8,325 | 294.3 | 357.8 | 422.8 |
| 2,700–3,300 | 3,037 | 305.0 | 377.5 | 441.5 |
| **3,300–3,606** (the last rows before the suspension edge) | 1,232 | **383.8** | 452.5 | **505.0** |
| ≥ 3,606 (LOBT re-filed, so still matched) | 6,873 | 326.4 | 359.1 | 442.3 |

**[X] The matched rows immediately below the one-hour clamp are the honest control group for the
lane**, and a blind model scores 505.0 on them. The unmatched lane's non-LIRF 566–569 is that number
plus the small-sample penalty. Nothing else is needed to explain it.

---

## 3. Substitutes for the lost clock (task item 3)

### 3.1 What the clock is, and what would have to replace it

Model A is handed `proxy = MVT − AOBT_3`: a *measurement* of when the aircraft left the stand,
subtracted from a take-off time it also knows. Its tree then fits `g = AOBT_3 − BLOCK`, which is a
**reporting offset between two sources**, not a taxi duration. **[X] That matters for the brief's
step 3 as posed: a substitute that correlates with `g` is substituting for the two systems'
bookkeeping disagreement, which is useless on a row that has only one system. The thing a substitute
must correlate with is `proxy`.** Both are reported below.

### 3.2 The candidates, and whether each is computable on an unmatched scored row

| candidate | definition | computable without the label on an unmatched scored row? |
|---|---|---|
| `stand_arrival_gap` | MVT − block time of the latest same-stand arrival, capped 6 h (`src/prc/features.py:291`) | **Yes**, arrivals keep `BLOCK_TIME_UTC_mvt`. Null on 21.2 % of unmatched scored rows |
| `nbr_mvt_minus_aobt_mean_30m/60m` | mean of other departures' MVT − AOBT_3 in a lagged window (`features.py:200-208`) | **Yes** — 0 % null. **Already in model B's feature set** |
| `arr_taxi_in_mean_30m/60m` | mean arrival taxi-in in the window | **Yes** — 0 % null. Already in model B |
| `queue_ahead` | departures between this flight's SCHED and its take-off | **Yes** — 0 % null. A `CANDIDATE_COL`, hidden by default |
| **`witness_med` / `witness_p90` / `witness_n`** | airport × calendar-hour median / p90 of MVT − AOBT_3 over that hour's **matched** departures, n ≥ 5 — merry-quicksand's shipped "congestion witness" (`results/21-repo-merry-quicksand.md` §6.3) | **Yes**, 2.80 % null on the scored unmatched rows (LEBL 7.7 %, LIRF 8.1 %, EDDF/LTFM 0 %) |
| **`surface_occ`** *(new)* | departures already off-block (AOBT_3 ≤ t) but not yet airborne at this take-off | **Yes** — 0.00 % null/NaN |
| **`nbr_aobt_minus_sched_60m`** *(new)* | mean of neighbours' AOBT_3 − SCHED in [t−3660, t−60): "how late is everyone actually pushing back right now" | **Yes** — 0.38 % |
| **`nbr_unmatched_rate_60m`** *(new)* | share of the last hour's departures with no NM record — the disruption-density idea left untested at `lfpg-unmatched-lane.md` §4.2 | **Yes** — 0.34 % |
| **`stand_prev_dep_gap`** *(new)* | seconds since the previous departure from the same stand took off | **Yes** — 1.15 % |

[S] `S2` for every null rate. All four new columns are built only from `MVT_TIME_UTC_mvt`,
`SCHED_TIME_UTC_mvt`, `STAND_mvt`, `ADEP_mvt` and *other rows'* `AOBT_3_flt`, all of which survive in
`ranking.parquet`.

### 3.3 `stand_arrival_gap` is **not** a bound, and that should be recorded

The attractive reading — the aircraft cannot have left the stand before the previous occupant blocked
onto it, so `y ≤ stand_arrival_gap` — is **false**. `S3`, on 1,697,462 matched labelled rows where the
gap exists:

| | pooled | EGLL | EDDF | LTFM | EDDM | LSZH | EHAM | LFPG | LEBL | LEMD | LIRF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P(y > gap) | **0.1181** | **0.3146** | 0.1325 | 0.1241 | 0.1053 | 0.1040 | 0.0894 | 0.0840 | 0.0732 | 0.0520 | 0.0461 |
| median slack (gap − y) | 4,278 | 3,897 | 4,775 | 5,466 | 3,841 | 3,883 | 4,263 | 4,561 | 3,602 | 4,438 | 3,903 |

Query: `feat_train` matched labelled rows, `(TAXITIME_SEC_mvt > stand_arrival_gap).mean()` by airport.

**[X] The asof join is a stand-*context* link, not a turnaround: at EGLL nearly a third of departures
have a taxi longer than the time since the last arrival on that stand, which can only mean the join
picked an arrival that is not this aircraft's inbound leg (remote stands, towing, stand renaming).
The column is a useful feature — it is the biggest single thing tested here — but it is not a
constraint, and any rule of the form `pred := min(pred, stand_arrival_gap)` would be wrong on one row
in eight.** The same test against `proxy` violates at 0.1123, so it is not a bound on the clock
either.

### 3.4 Correlation with the thing that is missing

`S3`, Spearman on 2,062,577 matched labelled rows, and on the 131,966 of them that are delay-selected
(`bs ≥ 3,000`, the closest analogue of the lane):

| candidate | vs `proxy` | vs `y` | vs `g` | vs `proxy`, late rows only |
|---|---:|---:|---:|---:|
| **`witness_med`** *(new to us)* | **+0.6860** | +0.4611 | −0.1966 | **+0.6068** |
| **`nbr_mvt_minus_aobt_mean_30m`** | **+0.6584** | +0.4423 | −0.1886 | +0.5682 |
| **`nbr_mvt_minus_aobt_mean_60m`** | **+0.6540** | +0.4392 | −0.1872 | +0.5707 |
| `witness_p90` *(new to us)* | +0.5737 | +0.4365 | −0.1126 | +0.4975 |
| **`surface_occ`** *(new)* | **+0.4141** | +0.2864 | −0.1202 | +0.3428 |
| `arr_taxi_in_mean_60m` | +0.2932 | +0.1218 | −0.1571 | +0.2018 |
| `arr_taxi_in_mean_30m` | +0.2781 | +0.1138 | −0.1513 | +0.1932 |
| `queue_ahead` | +0.2362 | +0.3426 | +0.1399 | +0.2240 |
| `witness_n` *(new to us)* | +0.2135 | +0.1253 | −0.0910 | +0.1667 |
| `dep_tko_prev_30m` | +0.2030 | +0.1299 | −0.0722 | +0.1380 |
| `mvt_minus_sched` | +0.1946 | +0.3883 | +0.2409 | +0.1877 |
| `nbr_aobt_delay_mean_30m` | −0.1902 | +0.0668 | **+0.2746** | −0.1798 |
| `stand_arrival_gap` | +0.1384 | +0.1280 | +0.0054 | +0.0902 |
| `nbr_aobt_minus_sched_60m` *(new)* | −0.1049 | +0.0433 | +0.1606 | −0.1455 |
| `nbr_unmatched_rate_60m` *(new)* | +0.0704 | −0.0013 | −0.0948 | −0.0063 |
| `stand_prev_dep_gap` *(new)* | +0.0698 | +0.0618 | −0.0032 | +0.0949 |
| `congestion_vs_usual_30m` | +0.0082 | +0.0401 | +0.0404 | −0.0024 |

**[S] One family stands out, and it is the same family three times over.** Everything above +0.40 is
some aggregate of *other flights'* measured off-block clock: the hour-bucket median (+0.6860, the best
single number in this note), our rolling means (+0.6584 / +0.6540), the hour-bucket p90 (+0.5737) and
the count of aircraft currently between off-block and take-off (+0.4141). Nothing that is *not* built
from other flights' `AOBT_3_flt` exceeds +0.30. Note also that no candidate correlates with `g` above
+0.28, which supports §3.1: `g` is bookkeeping, not physics.

**[X] And the caution that the rest of this note turns on: `surface_occ` correlates +0.41 with the
lost clock and is worth −0.07 s as a feature (§4.1).** Correlation with the missing quantity is a
screen, not a test — the neighbour clocks and the congestion block already carry what it carries. The
same is most of the reason the witness, at +0.6860, buys only −1.78.

---

## 4. Does any of it help where it matters (task item 4)

### 4.1 Added to the blind model, on matched rows, evaluated on the lane's composition

`S5`. Same fold, same 247,547 training rows, same 400 rounds; each variant is model B's feature set
plus the named column(s). "Reweighted" is the scored-lane composition of §2.2. **`noise_control`
(a hash of `MVT_ID_mvt`, built to carry nothing, `features.py:482`) is the instrument's noise floor.**

| variant | RMSE matched | Δ | RMSE reweighted | Δ | gain share |
|---|---:|---:|---:|---:|---|
| **B_cf base** | 292.53 | — | 554.03 | — | |
| **+`stand_arrival_gap`** † | **282.66** | **−8.46** | **539.09** | **−13.95** | 2.43 % |
| +`queue_ahead` † | 290.37 | −0.75 | 549.97 | −3.07 | 1.03 % |
| +`congestion_vs_usual_30m` † | 290.46 | −0.66 | 551.66 | −1.39 | 0.34 % |
| +`dest_distance_km` † | 290.30 | −0.82 | 550.94 | −2.10 | 0.16 % |
| +`surface_occ` | 292.46 | −0.07 | 554.70 | +0.67 | 0.43 % |
| +`nbr_aobt_minus_sched_60m` | 292.43 | −0.11 | 553.29 | −0.74 | 0.54 % |
| +`nbr_unmatched_rate_60m` | 292.99 | +0.46 | 555.48 | +1.45 | 0.23 % |
| +`stand_prev_dep_gap` | 290.74 | **−1.80** | 551.84 | −2.19 | 0.83 % |
| +all four new | 290.35 | −2.18 | 548.31 | −5.72 | |
| **+congestion witness (med, p90, n)** | 289.53 | **−3.00** | 552.25 | −1.78 | **5.84 %** |
| +witness + all four new | 287.33 | −5.20 | 547.49 | −6.54 | |
| **base MINUS the two neighbour clocks** | **298.13** | **+5.60** | **568.96** | **+14.94** | — |
| base minus neighbour clocks, **+ witness instead** | 290.16 | −2.38 | 553.08 | −0.95 | |
| **+`noise_control` (control)** | 292.61 | +0.07 | 554.86 | +0.83 | 0.15 % |

Query: `S5`, `rmse(y, pred)` on the 152,271 matched jan rows, and the post-stratified version.

> **† [X] The four marked rows are not from the same `S5` run as the rest of the table, and *both* of
> their numbers are affected. Recorded unresolved: re-running `S5` is what settles it.** Their own
> `RMSE` and `Δ` disagree by a constant **1.41** matched and **0.99** reweighted — `RMSE − Δ` gives
> **291.12 / 553.04** on all four, where this table's base row gives **292.53 / 554.03**. The other ten
> rows agree with the base row to rounding.
>
> * **The Δ cannot have drifted on its own.** `S5` fits exactly one base (`missing_clock.py:421`) and
>   prints every variant as `a - b_all` on the same line as that variant's own RMSE (`:446`). There is
>   no cached Δ, no prior results file, no hardcoded table and no second `base` variable in the
>   section. Within one run the two columns *cannot* disagree — so for these four rows **both** numbers
>   come from an earlier run whose base was 291.12 / 553.04. [S] `S5`
> * **292.53 / 554.03 is the shipped harness's base.** §2.2's `B_cf` is literally the same fit — same
>   fold, same `cols_b` + `cats_mvt`, same seed, same 400 rounds — and measures 292.53 / 554.0. [S] `S4`
> * **The earlier run is the superseded harness of §7's last caveat**, whose `base` still contained the
>   four new columns (one of them the wrong first build of `surface_occ`, §7). The four columns marked
>   here are `CANDIDATE_COLS` (`features.py:276`) and are hidden from `visible_numeric_columns`
>   (`train.py:245`) in *both* harnesses, so in that run they were still genuine additions and never
>   showed the identical-RMSE no-op signature that sent the four new-column rows back to be re-run.
>   That is why exactly these four rows, and no others, survived from it.
> * **So neither column can be repaired by arithmetic.** Read against this table's base the printed
>   RMSEs would give −9.87 / −14.94, −2.16 / −4.06, −2.07 / −2.37 and −2.23 / −3.09. Those are the size
>   of the discrepancy, **not** a correction: the models behind the printed RMSEs were fitted on a base
>   that is not this table's base.
>
> **What inherits this and what does not.** Everything resting on the ten agreeing rows is untouched:
> the floor itself (`noise_control` +0.07 / +0.83), the four new columns, the witness, and the
> +5.60 / +14.94 cost of removing the neighbour clocks — so §0.1–0.5, the second and third readings
> below, §6 and §7 stand as written, and `stand_arrival_gap` is the largest single contributor by an
> order of magnitude under either base. Pending the re-run: the first reading's placement of
> `queue_ahead`, `congestion_vs_usual_30m` and `dest_distance_km` *inside* the floor (true on the Δ as
> printed, false if their RMSEs are read against 292.53), and §5's `stand_arrival_gap` price, which
> moves by at most 0.03 s (−0.41 s as printed, −0.38 s on the Δ side of the row).

**Three readings. [X]**

* **The noise floor is about ±1 s matched and ±1.5 s reweighted.** Six of the eight single-column
  additions (the control aside) — `queue_ahead` †, `congestion_vs_usual_30m` †, `dest_distance_km` †,
  `surface_occ`, `nbr_aobt_minus_sched_60m`, `nbr_unmatched_rate_60m` — sit inside it on the Δ as
  printed; for the three marked † that placement is exactly what the footnote above leaves open, since
  their RMSEs read against this table's base would put them outside it. `nbr_unmatched_rate_60m`, the
  one idea `lfpg-unmatched-lane.md` §4.2 left open, is the **worst** of them: +0.46 / +1.45, i.e. it
  hurts, at a gain share of 0.23 %. **Record §4.2 as measured and closed.**
* **The neighbour off-block clock is the clock substitute, it exists, and it is already deployed.**
  Removing `nbr_mvt_minus_aobt_mean_30m/60m` costs +5.60 / +14.94 — more than every other column
  tested here put together. This is the answer to "is there anything visible that can stand in for
  it": yes, and we have had it since the context block was built.
* **merry-quicksand's witness is a better form of the same measurement, and the margin is small
  because we already hold it.** Their board −1,775 MSE (≈ −3.15 s at their 281.5) is what the witness
  is worth to a pipeline with *no* neighbour clock: our own no-neighbour-clock baseline is 568.96 and
  the witness alone takes it to 553.08, recovering the whole 14.94 and a bit more. Layered on top of
  what we have it is worth −1.78 reweighted, which prices at −0.05 s of board RMSE (§5). **[X] That
  is the explanation for the discrepancy, and it should stop anyone importing the witness expecting
  three seconds.**

### 4.2 Association with the model-B OOF residual, and whether it survives conditioning

`S6`, on the **8,345** unmatched OOF rows with the 33 day-rollover labels removed. `rho` is Spearman
against the signed residual `oof − y`; `partial rho` residualises **both** the candidate and the
residual on what model B already holds — airport, `hour_local`, `dow`, `month`, `ms` and the whole
context block (21 columns) — with a 4-fold cross-fitted LightGBM, then correlates the residuals.

| candidate | n | rho(c, resid) | rho(c, \|resid\|) | **partial rho** | partial \|r\| |
|---|---:|---:|---:|---:|---:|
| `stand_arrival_gap` | 6,681 | +0.0168 | +0.0310 | +0.0346 | +0.0202 |
| `queue_ahead` | 8,099 | −0.0258 | +0.1206 | +0.0220 | −0.0098 |
| `congestion_vs_usual_30m` | 8,306 | −0.0146 | +0.0065 | +0.0000 | −0.0019 |
| `dest_distance_km` | 8,254 | −0.0314 | +0.0938 | −0.0372 | +0.0429 |
| `surface_occ` | 8,345 | −0.0552 | −0.0448 | −0.0380 | +0.0027 |
| `nbr_aobt_minus_sched_60m` | 8,316 | −0.0175 | **+0.1720** | +0.0120 | +0.0082 |
| `nbr_unmatched_rate_60m` | 8,322 | +0.0025 | +0.0940 | +0.0053 | +0.0041 |
| `stand_prev_dep_gap` | 8,308 | −0.0160 | +0.0328 | −0.0170 | +0.0260 |
| `witness_med` | 8,183 | −0.0521 | +0.1011 | −0.0375 | −0.0123 |
| `witness_p90` | 8,183 | −0.0511 | **+0.1294** | −0.0359 | +0.0455 |
| `witness_n` | 8,345 | −0.0404 | −0.1045 | −0.0138 | +0.0101 |
| **Gaussian noise (control)** | 8,345 | +0.0031 | — | **+0.0066** | — |

**[S] Nothing survives.** The largest partial |rho| in the table is 0.038; the noise control is 0.007;
eleven candidates against one control, so one or two of these are what 0.03 looks like by chance at
n = 8,345. Per airport (`S6` prints the full 10 × 11 grid) the only |rho| above 0.30 is
`stand_arrival_gap` at **LIRF, +0.391 on 506 rows** — which is the schedule-fill lane behaving like
the schedule-fill lane, not a clock effect.

**[X] The one thing that is real in this table is in the wrong column.** `nbr_aobt_minus_sched_60m`
(+0.172), `witness_p90` (+0.129) and `queue_ahead` (+0.121) correlate with the **magnitude** of the
residual, not its sign. That is heteroskedasticity: those columns say *when model B will be
uncertain*, not *which way it is wrong*. Under an L2 loss a variance signal with no mean signal buys
nothing in point prediction, and §4.1 confirms it — `nbr_aobt_minus_sched_60m` measures −0.11 s as a
feature. **Six candidates in this project have died at review for being descriptions of the residual;
these three would be the seventh, eighth and ninth. Do not propose them as levers.**

---

## 5. Price (task item 5)

`S7`. Board exchange rate: `N = 344,841`, RMSE 282.7371, so one second of RMSE = `N × (2R + 1)` =
**1.9534e8** of squared error. Scored unmatched rows: **5,290 = 383 LIRF + 4,907 other**.

| counterfactual | n | from | to | ΔRMSE (board) |
|---|---:|---:|---:|---:|
| **non-LIRF: observed → fully sighted** | 4,907 | 569.1 | 361.6 | **−4.85 s** |
| non-LIRF: blind-and-starved → sighted (both counterfactual, consistent pair) | 4,907 | 591.4 | 361.6 | **−5.50 s** |
| non-LIRF: the *training-size* part alone (more rows, still blind) | 4,907 | 591.4 | 541.7 | −1.42 s |
| non-LIRF: what `stand_arrival_gap` measurably recovers † | 4,907 | 554.0 | 539.1 | **−0.41 s** |
| non-LIRF: what the witness + all four new columns recover | 4,907 | 554.0 | 547.5 | **−0.18 s** |
| whole lane: observed → sighted (includes LIRF) | 5,290 | 1,141.3 | 361.6 | −31.73 s |
| whole lane: a 10 % cut of its RMSE — the shape of the old −7 s | 5,290 | 1,141.3 | 1,027.2 | −6.70 s |
| **LIRF's 383 rows alone → the non-LIRF unmatched level** | 383 | 3,437.2 | 569.1 | **−22.53 s** |

**†** The 539.1 is §4.1's `+stand_arrival_gap` row, which is flagged there: on the Δ side of that same
row the figure is 540.1 and the price −0.38 s. The range −0.38 to −0.41 s is what is settled; the exact
value waits on a re-run of `S5`. [S] `S5`, §4.1

**[X] Three conclusions from this table.**

1. **Closing the model-B-to-model-A gap entirely, at the nine airports where the gap is about the
   clock, is worth −4.9 to −5.5 s and that is the whole ceiling.** It requires an off-block
   measurement we do not have and cannot construct; §3 and §4 say the best available construction is
   already in the model.
2. **A realistic fraction is −0.2 to −0.6 s.** `stand_arrival_gap` is the only measured winner and it
   was adopted this morning at −3.62 s board — of which, by this arithmetic, about **−0.41 s comes
   from the unmatched lane** and the rest from the matched one. The witness plus everything new adds
   −0.18 s on top. A hypothetical 10 % cut of the blindness gap would be −0.57 s.
3. **The old −7 s estimate is not wrong, but it is not this lane's clock.** −7 s is a 10 % cut of the
   *whole* lane, and **22.5 of the lane's 31.7 s sits in LIRF's 383 rows**, whose mechanism is
   `BLOCK_TIME` stamped with `SCHED_TIME` and not a missing clock at all. The route to it was the
   fill classifier, which failed its pre-registered 0.93 gate at 0.9201 this morning
   (`results/P1-stage2-gate-FAILED.md`). **Anyone planning against −7 s should be told that the money
   is at LIRF and the door to it is currently closed.**

---

## 6. What this leaves, stated plainly

**[X] The clock cannot be substituted.** Enumerated (§3.2), correlated (§3.4), fitted (§4.1) and
conditioned (§4.2), the ranking is:

* `nbr_mvt_minus_aobt_mean_30m/60m` — **the** substitute, +0.658 Spearman with the lost clock, worth
  15 s of unmatched-lane RMSE, **already shipped**. Nothing here is a change.
* `stand_arrival_gap` — worth −13.95 reweighted (−14.94 on the RMSE side of the same row; see §4.1's
  † footnote, which does not change its rank), **already adopted today**, and explicitly **not** a
  bound (§3.3). Anyone tempted to clip predictions to it should read that section first.
* the congestion witness in merry-quicksand's exact hour-bucket form — real (−3.00 matched against a
  +0.07 control), **computable on every scored unmatched row at 2.8 % null**, and worth about
  **−0.05 s** of board RMSE because we already hold a rolling version of it. **[X] It is the only
  item in this note that is both new to our pipeline and measurably non-zero. It is also small enough
  that it should not be spent a slot on unless something else is already being rebuilt.** If it is
  ever tried, the honest expectation to pre-register is −0.05 s, not merry-quicksand's −3.15 s.
* `surface_occ`, `nbr_aobt_minus_sched_60m`, `nbr_unmatched_rate_60m`, `stand_prev_dep_gap` — all
  computable, all inside or barely outside the noise floor, none surviving conditioning. **Closed.**
  `surface_occ` is the instructive one: **+0.4141 Spearman with the very clock that is missing, and
  −0.07 s as a feature.** It is a genuine measurement of the same physical situation and it is
  entirely absorbed by the neighbour clocks and the congestion counts that are already there.
  This includes `lfpg-unmatched-lane.md` §4.2's unmatched-density idea, which is hereby measured:
  +0.46 / +1.45, worse than the noise control.

**The lane's error at the nine non-LIRF airports is irreducible at this feature set**, in the precise
sense that a model with the same blindness and the same training-set size reproduces it on matched
rows to within 4 % (591.4 predicted, 569.1 observed). What would change that is an *external*
off-block observation — ADS-B ground samples are the obvious one, and `results/21-...` §6.4 records
both that merry-quicksand's −7,722 MSE came from exactly that and that the organisers gave no ruling
on whether it is permitted, and that LFPG, LIRF, LEMD and LTFM had zero on-ground samples in their
2025 census. That is a governance question, not a feature-engineering one.

---

## 7. Caveats on the instrument, recorded rather than buried

* **One fold, one seed, 400 fixed rounds, 13 % of the matched training rows.** Model A measures
  222.76 here against a shipped matched OOF of 216.34, so every absolute number is about 3 % worse
  than production. All comparisons are paired on identical rows and folds; none of the conclusions
  rests on an absolute level. A second fold was not run, and the single-column deltas of §4.1 should
  be read against the ±1 s noise control, not as point estimates.
* **The reweighting is a post-stratification on 10 airports × 5 `ms` bands**, cells with fewer than
  20 matched validation rows dropped (coverage 1.000 of the scored unmatched rows). It corrects for
  composition, not for anything else the lane might differ in — and by §2.2 nothing else is needed,
  which is itself the finding.
* **The 361.6 "fully sighted" figure is model A on *matched* rows at the lane's composition.** It
  assumes an unmatched row handed a real off-block clock would be as predictable as a matched row of
  the same delay and airport. **[X] That is an assumption, not a measurement, and it is the optimistic
  end: the rows lost their flight plan because NM suspended them, which may correlate with
  operational states no clock would explain.** If it is wrong, the −4.9 s ceiling is an overestimate.
* **`surface_occ` was wrong on its first build** — both sides of the difference must be restricted to
  flights that have an `AOBT_3`, or the count drifts with the cumulative number of unmatched rows and
  becomes a monotone time index (mean −1,107, min −4,084). The corrected column has mean 10.0,
  max 45. The first version is not in the script; this is recorded so the trap is not re-entered.
* **An earlier draft of the §4.1 harness silently measured nothing** because the four new columns had
  already entered `visible_numeric_columns` via the joined frame, so "base" already contained them and
  every "+column" variant returned an identical RMSE to six figures. That signature — several
  different feature sets giving *exactly* the same number — is what a no-op looks like, and it is the
  same class of bug as the AUC-1.0000 leak in `P1-stage2-gate-FAILED.md`. The shipped script takes its
  base feature list with the candidates explicitly removed.
