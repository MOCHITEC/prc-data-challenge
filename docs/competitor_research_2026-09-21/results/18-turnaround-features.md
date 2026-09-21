# Slot 18 — turnaround, arrival and stand-linkage features

Investigator slot 18. Written 2026-09-21. Marks: **[S]** = sourced (repo file + line, published page, or
a JSON artefact in this repository), **[X]** = my inference, hypothesis or reading. No **[X]** below is
dressed up as **[S]**.

Sources cloned and read at these commits:

| Source | Repo / URL | Commit / date read |
|---|---|---|
| Phoenix-Ops (team **zestful-fountain**) | `https://github.com/Phoenix-Ops-LTD/prc2026-taxiout` | `4477d9b`, 2026-09-13 |
| OpenAir (team **likable-eagle**) | `https://github.com/ChandanHegde07/OpenAir` | `97f603c`, 2026-09-15 |
| Third PRC-2026 entrant (gultekin) | `https://github.com/ahmetabdullahgultekin/prc-taxiout-2026` | shallow clone 2026-09-21 |
| Our own prior run | `/home/user/prc-data-challenge` commit `8bdf4f9` on `codex/feature-engineering` | in-repo |

---

## 0. Headline correction to the brief, before anything else

The brief states that Phoenix-Ops's **v8 "deeper arrival residual" is worth −7.1 s and is the largest
single documented step**. The −7.1 s is right. **The attribution is wrong.**

**[S]** v8 adds **no new arrival features at all**. `arrival_boost_contest.py:15` imports `matrix` from
`arrival_contest`, i.e. the *identical* v7 feature matrix. METHOD.md:157: *"`arrival_boost_contest.py`
uses the **same 123 predictors** as the v7 global model with CatBoost depth 9, learning rate 0.05, L2
regularization 7, 254 borders and seed 20260907."*

So v8 = **the same 26 arrival features, re-learned by a much larger model**, blended 50/50 with v7.
The decomposition of the arrival family's total value in their log is therefore:

- **−2.6224 s** (v6 285.749 ← 288.371) for *introducing* the 26 arrival features into a **679-tree
  LightGBM** at 25 % blend weight (METHOD.md:153).
- **−7.1141 s** (v7 285.749 → v8 278.6349) for *re-fitting the same information* with a **4,999-tree
  depth-9 CatBoost** at 50 % blend weight (METHOD.md:168).

**[X]** My reading: the dominant lever in their largest step is **model capacity applied to the arrival
residual**, not additional arrival semantics. Anyone replicating "v8" by writing more arrival features
will replicate the −2.6, not the −7.1.

---

## 1. Phoenix-Ops v7 — the 26 arrival features, reimplementation-grade

**[S]** Whole file: `/tmp/prc2026-taxiout/arrival_features.py`, 78 lines, function
`completed_arrival_features(dep, context)`.

### 1.1 Arrival pool construction (lines 18–30)

```
cols = ['ADES_mvt','RUNWAY_mvt','STAND_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt',
        'TAXITIME_SEC_mvt','AIRCRAFT_TYPE_mvt','FLIGHT_mvt']
arr = context[context.PHASE_mvt == 'ARR'][cols]          # filter FIRST (line 20)
landed  = to_datetime(MVT_TIME_UTC_mvt)                   # ARR movement = landing
blocked = to_datetime(BLOCK_TIME_UTC_mvt)                 # ARR block   = on-blocks
taxi    = to_numeric(TAXITIME_SEC_mvt)
valid = landed.notna() & blocked.notna() & blocked >= landed & taxi.between(0, 7200)
completed_seconds = blocked (epoch seconds)               # event time = ON-BLOCK, not landing
taxi_seconds      = taxi
airport = norm(ADES_mvt); runway = norm(RUNWAY_mvt); stand = norm(STAND_mvt)
aircraft = norm(AIRCRAFT_TYPE_mvt)
carrier  = norm(FLIGHT_mvt).str.extract(r'^([A-Z]{2,4})').fillna('UNKNOWN')
```
`normalized()` (line 13) = `str.strip().str.upper()`, empty → `'UNKNOWN'`.

The **filter-first** ordering is deliberate and load-bearing: `BLOCK_TIME_UTC_mvt` and
`TAXITIME_SEC_mvt` are supplied inputs on ARR rows and blanked targets on DEP rows
(file docstring, lines 2–6; METHOD.md:139 cites the organizer schema page).

### 1.2 Query key and clock (lines 31–34)

Query time is the **departure's take-off** `MVT_TIME_UTC_mvt` (must be non-null, line 32 raises).
Keys: `airport = norm(ADEP_mvt)`, `runway = norm(RUNWAY_mvt)`, `stand = norm(STAND_mvt)`.

### 1.3 Airport- and runway-scoped blocks — 22 columns (lines 36–61)

For `label ∈ {airport (group=[airport]), runway (group=[airport,runway])}`, and windows
`m ∈ {15, 30, 60}` minutes, over arrivals whose **on-block time is strictly before** the departure's
take-off (`searchsorted(..., side='left')`, line 45 — exact ties excluded):

| column | definition |
|---|---|
| `completed_arrival_{label}_{m}m_count` | n arrivals with on-block in `[q − 60m, q)` |
| `completed_arrival_{label}_{m}m_mean` | mean arrival **taxi-in seconds** over that window |
| `completed_arrival_{label}_{m}m_std` | population sd, `sqrt(max(0, E[x²] − mean²))` (line 53) |
| `completed_arrival_{label}_last_age_seconds` | `q − t_prev`, only if `q − t_prev ≤ 7200` else NaN |
| `completed_arrival_{label}_last_taxi_seconds` | taxi-in of the last arrival, **averaged over all arrivals tied at that timestamp** (line 56, `groupby('completed_seconds').taxi_seconds.mean()`) |

2 labels × (3×3 + 2) = **22 columns**.

### 1.4 Same-stand block — 4 columns (lines 62–77) — *this is the turnaround link*

Group `['airport','stand']`, **stands equal to `'UNKNOWN'` excluded from the lookup** (line 63).

```
right     = searchsorted(t, q, side='left')        # strictly-before
previous  = max(right - 1, 0)
ambiguous = g.completed_seconds.duplicated(keep=False)   # any tied on-block time at this stand
age       = q - t[previous]
recent    = (right > 0) & (age <= 86400) & ~ambiguous[previous]      # 24-HOUR cap
```

| column | definition |
|---|---|
| `completed_arrival_stand_last_age_seconds` | seconds from previous same-stand arrival's **on-block** to this departure's **take-off** (≤ 86 400) |
| `completed_arrival_stand_last_taxi_seconds` | that arrival's **taxi-in seconds** |
| `completed_arrival_stand_aircraft_equal` | 1.0 if dep `AIRCRAFT_TYPE_mvt` == that arrival's type, else 0.0 |
| `completed_arrival_stand_carrier_equal` | 1.0 if 2–4-letter prefix of `FLIGHT_mvt` matches, else 0.0 |

All 26 emitted as `float32` (lines 61, 77). Ambiguous ties are left **missing**, not averaged —
deliberately different from the airport/runway blocks (METHOD.md:139).

**[S]** METHOD.md:141 — standalone fixed-feature comparison with these 26 added to 97 features
(→123): Jan/Jul 332.674 → **326.346**; Feb/Aug 272.272 → **267.592**; all four months improve.

**[X]** Note what is *absent*: no aircraft registration, no explicit rotation, no turnaround duration.
`aircraft_equal` and `carrier_equal` are the **only** identity-chaining proxies, and they are binary
agreement flags, not a chain.

### 1.5 How v7 used them (arrival_contest.py)

- `matrix()` (lines 32–40) = `duration_matrix` (93 cols) + `movement_service` (normalized `FLIGHT_mvt`),
  `nm_service` (normalized `CALLSIGN_flt`) + their `ADEP:value` interactions = 97, then `concat` the 26 → **123**.
- Target is a **residual**: `y − nm_schedule_fallback_baseline(x)` (line 90), where the baseline is
  `movement_time − AOBT_3` with fallbacks to `LOBT`, `EOBT_1`, 1000 s, and a LIRF-specific
  `movement − schedule` branch (`traffic_features.py:107–121`).
- LightGBM `num_leaves=63, lr=.035, min_child_samples=80, reg_lambda=10, colsample_bytree=.9,
  n_estimators=679, seed 20260907` (lines 77–83).
- Blend: `0.25 × arrival_model + 0.75 × v6`, with `specialist_scope` rows (LIRF ∧ `missing_IOBT_flt`)
  replaced by `0.5 × v6 + 0.5 × arrival_specialist` (lines 50, 138).
- `arrival_specialist.py`: 5 seeds 20260907–20260911, CatBoost `depth 5, lr .04, l2 10, 1014 iters`,
  equal weight, on carrier70 + the same 26 (lines 20, 30–34, 62–64).

---

## 2. Phoenix-Ops v8 — reimplementation-grade spec of the −7.1 s step

**[S]** `/tmp/prc2026-taxiout/arrival_boost_contest.py`, 153 lines.

| item | value | source |
|---|---|---|
| features | **exactly v7's 123** (`from arrival_contest import matrix`) | line 15, 61 |
| estimator | CatBoost `RMSE` | line 30 |
| depth | **9** | line 30 |
| learning_rate | **0.05** | line 30 |
| l2_leaf_reg | **7** | line 30 |
| border_count | **254** | line 30 |
| iterations | **4,999** (max 5,000; early stopping patience 200 chose 4,999 in the Jan/Jul validation run) | lines 24, 30; METHOD.md:157 |
| random_seed | 20260907 | line 30 |
| task_type | GPU, `thread_count=2` | lines 25, 30 |
| categoricals | every `str`/`object` column of `x` | line 70 |
| fit population | `y ≥ 0 ∧ ¬specialist_scope(x)` — 2,083,190 rows | lines 66; METHOD.md:161 |
| target | `(y − nm_schedule_fallback_baseline(x)).clip(−7200, +7200)` | line 77 |
| prediction | `max(0, model.predict(x) + nm_schedule_fallback_baseline(x))` | lines 115–116 |
| blend | `0.5 × v7 + 0.5 × candidate`; `specialist_scope` rows keep v7 **exactly** | lines 43–45 |
| `specialist_scope` | `ADEP_mvt == 'LIRF' ∧ missing_IOBT_flt == 1` | `carrier_contest.py:34–35` |

**[S]** Validation discipline (METHOD.md:159): the 50 % weight was *declared before* reading the result;
promotion required ≥ 0.5 s overall, improvement in **each** month, and ≥ 40 improved days. Achieved:
315.778 vs 318.547 on the reused Jan/Jul holdout (Jan 341.669 vs 343.676, Jul 293.251 vs 296.749,
**61 of 62 days** improved). Official: **278.6349**, −7.1141 vs v7 (METHOD.md:168).

**[S]** Feature parity check they ran: *"all 123 columns across 2,085,047 departure rows matched
exactly"* (METHOD.md:161) — the same 2,085,047 figure our own parity check used.

---

## 3. Phoenix-Ops v9/v10 — stand-level congestion via an NM proxy (not arrival)

**[S]** `nm_neighbor_features.py` (49 lines) emits **30** columns, of which **10 are stand-scoped**.

Pool: DEP rows only, with `proxy = MVT_TIME_UTC_mvt − AOBT_3_flt ∈ [0, 7200]` and
`norm(ADEP_mvt) == norm(ADEP_flt)` (lines 10–11). Groups, each **scoped to the same UTC calendar month**
(`period = strftime('%Y-%m')`, line 14): `[airport, period]`, `[airport, period, runway]`,
`[airport, period, stand]`; `UNKNOWN` runway/stand form no group (line 25).

Per group, windows 15 and 60 min, **strictly before** the query take-off:
`{15,60}m_count`, `{15,60}m_mean`, `{15,60}m_std`, `{15,60}m_own_minus_mean` (own valid proxy minus
window mean, line 41), `last_age_seconds`, `last_proxy` (ties at the latest timestamp averaged,
lines 44–45). 3 groups × 10 = 30, named `nm_neighbor_{airport|runway|stand}_*`.

**[S]** v9 = `0.75 × v8 + 0.25 × (0.5 × per-airport-expert + 0.5 × neighbour-LGBM)`
(`ordinary_ensemble.py:44–55`), official **277.7530**, −0.8819 (METHOD.md:196).
v10 = the 153-feature CatBoost replacement, official **275.1751** (METHOD.md:200, 262).

This is their **stand-congestion / apron-occupancy** family. It is built from *departure* NM clocks,
**not** from arrivals, and it is worth roughly **−0.9 s** — an order of magnitude less than v8.

---

## 4. OpenAir — the inbound/turnaround family, reimplementation-grade

**[S]** Production code: `/tmp/OpenAir/experiments/matched_submit.py:134–240`
(`load_arr_stand`, `attach_arr_stand`), driven by `experiments/submit.py:75–165`.

### 4.1 Arrival pool

```python
ARR rows → airport = ADES_mvt, STAND_mvt,
           aibt       = BLOCK_TIME_UTC_mvt        # actual in-block
           arr_taxiin = TAXITIME_SEC_mvt          # arrival taxi-in seconds
  rich:    sibt       = SCHED_TIME_UTC_mvt        # scheduled in-block
           RUNWAY_mvt, arr_type = AIRCRAFT_TYPE_mvt
drop_nulls([airport, STAND_mvt, aibt, arr_taxiin])
arr_delay   = (aibt - sibt).total_seconds()                       # INBOUND LATENESS
sort([airport, STAND_mvt, aibt])
arr_taxiin_2 = arr_taxiin.shift(1).over([airport, STAND_mvt])     # second-most-recent
```
(`matched_submit.py:138–164`)

### 4.2 Attachment — four independent as-of joins

| join | left key | right key | `by` | source |
|---|---|---|---|---|
| A. previous same-stand arrival | `MVT_TIME_UTC_mvt` (take-off) | `aibt` | `[airport, STAND_mvt]` | line 170 |
| B. same-stand arrival as of push-back | `AOBT_3_flt` | `aibt2` | `[airport, STAND_mvt]` | lines 194–200 |
| C. previous same-stand **departure** | `MVT_TIME_UTC_mvt` | `prev_mvt` (other DEP take-offs) | `[airport, STAND_mvt]` | lines 213–219 |
| D. previous same-**runway** arrival | `MVT_TIME_UTC_mvt` | `aibt` | `[airport, RUNWAY_mvt]` | lines 232–237 |

All `strategy="backward"`.

### 4.3 The eight features and their validity windows

| feature | definition | window (else NULL) | source |
|---|---|---|---|
| `since_arr` | take-off − previous same-stand arrival's in-block | `0 < since_arr < 8 h` | 171–180 |
| `arr_taxiin` | that arrival's taxi-in seconds | same 8 h window | 173–176 |
| `arr_delay` | that arrival's `AIBT − SIBT` (inbound lateness, s) | same 8 h window | 184–189 |
| `arr_taxiin_2` | taxi-in of the arrival *before* that one at the same stand | same 8 h window | 227–228 |
| `arr_type` (categorical) | that arrival's aircraft type | same 8 h window | 229–230 |
| `since_arr_aobt` | `AOBT_3 − previous same-stand arrival in-block` | `0 < x < 8 h` | 201–210 |
| `arr_taxiin_aobt` | taxi-in of the arrival preceding **push-back** rather than take-off | same | 201–210 |
| `since_dep` | take-off − previous same-stand **departure** take-off (stand re-use interval) | `1 s < since_dep < 12 h` | 219–225 |
| `rwy_arr_taxiin` | taxi-in of the last arrival on the **same runway** | none | 231–237 |

**[X]** `since_dep` (join C) uses the departure frame against itself with a backward as-of join, so a
row matches itself at gap 0; the `> 1 s` guard is what removes the self-match. Reimplementers should
exclude self explicitly rather than rely on the guard.

### 4.4 How they are consumed — a Δ-model, not the main model

**[S]** `submit.py:120–126, 165–180`: two LightGBM heads on the **matched** branch only:
`Δ = y − P` where `P = clip(MVT − AOBT_3, 0, ∞)`, and `hat = y − e20_pred` (leftover against the E20
ensemble). The inbound features enter **both** heads as plain numeric/categorical columns. Final:
`rec = clip(P + Δ̂, 0, ∞)`; `g = grec(e8, rec, λ_rec)`; `new = max(g + λ_hat·max(ĥ,0), 0)`.
Production v19 used `n_estimators=900, lr=0.03, num_leaves=63, min_child_samples=40`, 3 Δ-seeds and
2 hat-seeds, `λ_hat = 0.35` (`submit.py:104–107`, status.md:1707).

### 4.5 Measured value

| step | features added | internal (Jan+Jul matched / Dec matched) | official LB | source |
|---|---|---|---|---|
| v13 baseline | — | 238.68 / 212.08 | 284.0967 | status.md:1555, `results/E53/E53_report.md` |
| **v16** | `arr_taxiin`, `since_arr` | **236.86 / 210.54** | **282.0851 (−2.01)** | status.md:1577–1585 |
| **v17** | + `arr_delay`, `since_dep`, `arr_taxiin_aobt`, `since_arr_aobt` | 236.80 / 210.29 | **281.8799 (−0.21)** | status.md:1587–1593 |
| **v18** | + `arr_taxiin_2`, `rwy_arr_taxiin`, `arr_type` | 236.59 / 210.22 | **281.7196** | status.md:1595–1601 |

The whole inbound family is therefore worth **284.10 → 281.72 = −2.38 s** officially, of which
**−2.01 s comes from the first two columns alone** (`arr_taxiin` + `since_arr`). Everything after that
is −0.37 s of decoration.

**[S]** A second, independent use: `run_e73_lirf_g.py:20–37` puts `arr_taxiin`, `arr_taxiin2`,
`since_arr`, `arr_delay` into the **LIRF gate-hold model** (CatBoost 1200 iters, lr .04, depth 8, l2 5,
`od_wait 80`, target `G = D − T` with `D = MVT − SCHED`, prediction `T = max(D − Ĝ, 0)`).
Its window there is **36 h**, not 8 h (`run_e73_lirf_g.py:33`). Result (status.md:1727–1742):
LIRF-unmatched SSE −5.3 % Jan+Jul, −19.1 % Dec; **v18 281.72 → v20 280.1831, −1.54 s**, applied to
exactly **383 rows**. That is the single highest per-row leverage in either repo.

---

## 5. OpenAir's negative results on the same family — read these before building anything

These are the strongest **[S]** warnings available, because they are from the team that later shipped
the feature successfully.

1. **E7 (status.md:594–607).** `FLIGHT_ID_mvt` is **not** an aircraft/tail key: 315,272 IDs appear
   twice, 315,262 of those are the *same city-pair flight* seen as DEP then ARR at two of the ten
   airports (median Δt 5,065 s ≈ airborne time). Using the paired ARR for the DEP is future
   information. → **REJECT `FLIGHT_ID` as a rotation key.**
2. **E7, same-stand statistics.** 99.9 % of DEPs have *some* preceding same-stand ARR (vacuous);
   same aircraft type 94 %; same callsign **2 %**; same `FLIGHT_ID` ≈ **0 %**. Plausible 20 min–8 h
   same-type: 83 %. **corr(taxi-out, previous taxi-in) = 0.12; corr(taxi-out, turnaround duration) = 0.09.**
3. **E71 (status.md:1722).** The identity `T = since_arr − turnaround` is causally available with
   99.6 % coverage but **corr(since_arr, T) = 0.008**; *"the latest ARR at a stand is frequently a
   different aircraft (stand reuse), so `since_arr` is not the current aircraft's own turnaround."*
4. **E71c (status.md:1724).** A self-excluded **previous-DEP-at-same-stand oracle**:
   `corr(T, gap_block) = 0.008` (unmatched) / 0.043 (LIRF-u); `corr(T, prev_Taxi)` 0.13 / 0.28.
   *"The oracle has no signal."*
5. **E72 (status.md:1725).** Using the linkage as a **hard post-processing bound** — capping
   `T ≤ since_arr` — made LIRF-unmatched RMSE go **3,896 → 11,579** and non-LIRF-unmatched
   1,546 → 1,568. Rejected.
6. **Discovery report (`analysis/DISCOVERY_REPORT.md:275`).** *"Adding recent taxi-in to the OLS
   changes RMSE by < 1 s. Do not chase it."*
7. **E68 (status.md:1703).** Extra stand-history features (previous-DEP clocks, arrivals-in-3h) on top
   of the shipped set were **neutral**.
8. **E77 (status.md:1751).** External **OPDI `icao24` turnaround** — i.e. real aircraft registration
   chaining from open ADS-B data — scored **236.06 vs 235.94**, i.e. *worse than not having it*.

**[X]** The pattern across 1–8: every *marginal* statistic on this family is near zero, and every
*post-hoc* or *hard-constraint* use of it fails; it nevertheless pays −2.0 s when handed to a
sufficiently large boosted residual model as a plain column. The information is **conditional and
interactive** (it means something only jointly with stand, hour, airport, and the clock proxy), which
is exactly the regime where univariate correlation and binned corrections are blind and trees are not.

---

## 6. Broader catalogue, against the brief's checklist

| Brief item | What exists | Best available implementation | Evidence |
|---|---|---|---|
| **inbound leg: arrival time** | `since_arr` / `stand_last_age_seconds` | as-of backward join on `BLOCK_TIME_UTC_mvt` of ARR, key `(ADES, STAND)`; windows 8 h (OpenAir), 24 h (Phoenix-Ops), 36 h (OpenAir LIRF-G) | **[S]** `matched_submit.py:170–180`; `arrival_features.py:66–74` |
| **inbound leg: taxi-in** | `arr_taxiin` / `stand_last_taxi_seconds` | the joined ARR's `TAXITIME_SEC_mvt` | **[S]** same; **this is the −2.01 s column** |
| **inbound leg: delay** | `arr_delay` | `AIBT − SIBT` of the joined ARR | **[S]** `matched_submit.py:160` |
| **inbound leg: origin** | **not used by anyone** | ARR `ADEP_mvt` is available on arrival rows | **[X]** untested by any published competitor I read |
| **turnaround so far at query time** | `since_arr` (to take-off) and `since_arr_aobt` (to push-back); Phoenix-Ops has no equivalent | the `AOBT`-anchored variant is the cleaner "turnaround so far" because it stops at push-back | **[S]** `matched_submit.py:194–210` |
| **previous occupant of the stand: when it left** | `since_dep` (OpenAir), 12 h window; Phoenix-Ops has `nm_neighbor_stand_last_age_seconds` (proxy-based) | as-of join of DEP take-offs on the same stand, **self-excluded** | **[S]** `matched_submit.py:213–225`; `nm_neighbor_features.py:24,29` |
| **previous occupant: how long occupied** | only as `since_arr − since_dep` implicitly; **nobody computes stand occupancy duration directly** | — | **[X]** gap in the published work |
| **stand-to-runway distance proxy** | the **official EUROCONTROL definition is itself a stand×runway statistic** | EUROCONTROL PRU: *"one unimpeded time by departure airport, departure runway and departure stand"*; the P10 of that cell, requiring ≥10 flights at or below P10 | **[S]** [SES Performance data portal](https://www.sesperformance.eu/dataportal/metadata/additional-taxi-out-time/); P10 rule restated at `prc-taxiout-2026/docs/literature.md:62` |
| | measured strength | stand×runway group means: **η² ≈ 0.2525**; airport alone **η² ≈ 0.1121**; ~12,053 observed pairs, 6,436 with n ≥ 30; same stand / different runway differs by **302 s mean, p90 563 s** | **[S]** `OpenAir/status.md:144–150`, `analysis/DISCOVERY_REPORT.md:318–327` |
| | Phoenix-Ops equivalent | `stand_runway = airport_STAND_mvt + ':' + RUNWAY_mvt` as a single categorical | **[S]** `traffic_features.py:59` |
| **stand congestion / apron occupancy** | Phoenix-Ops `nm_neighbor_stand_*` (10 cols, monthly-scoped) | see §3 | **[S]** `nm_neighbor_features.py` |
| | but: | congestion counts correlate only **0.05–0.17** with raw taxi and are *"redundant given `MVT−AOBT`"*; traffic and queue are **not in the top 12** LightGBM gains | **[S]** `DISCOVERY_REPORT.md:292`, `status.md:526, 630` |
| **was the inbound late, by how much** | `arr_delay` | see above; worth **−0.21 s** as part of the v17 bundle | **[S]** status.md:1587–1593 |
| **aircraft-identity chaining without a registration** | three distinct attempts, all weak | (a) `FLIGHT_ID` — **rejected**, it is a city-pair ID (E7); (b) type/carrier agreement flags — Phoenix-Ops `stand_aircraft_equal` / `stand_carrier_equal`, and OpenAir `arr_type`; (c) same-callsign previous DEP, median gap **24.1 h**, `corr = 0.41` with the previous *actual* taxi — but that previous taxi is blanked in ranking, so only a `MVT − AOBT` lag is legal | **[S]** status.md:203–204, 606; `arrival_features.py:75–76` |
| | external registration | OPDI `icao24` turnaround: **236.06 vs 235.94, worse** | **[S]** status.md:1751 |

### Academic anchors (thin, and honestly so)

**[S]** The taxi-out literature does **not** treat turnaround/inbound state as a standard feature family.
The four canonical families are queue, surface traffic, geometry and weather:
Idris et al. (2002) — runway configuration, airline/terminal, downstream restrictions, **departure
queue size** (largest); Simaiakis & Balakrishnan (2015) — `taxi-out = unimpeded + queue + congestion`;
Ravizza et al. (2020) — taxi distance, sum of turning angles, dep/arr distinction, surrounding traffic;
Lee, Malik & Jung (2016) — terminal concourse, spot, runway, departure fix, weight class.
All four are catalogued with DOIs at `prc-taxiout-2026/docs/literature.md:14–131`.
**[X]** I found no taxi-out paper that uses "time since the aircraft arrived" as a predictor. The
turnaround literature I reached (TOBT/ground-time prediction) predicts *turnaround*, using inbound
delay `AIBT − SIBT` and scheduled ground time as inputs — the reverse direction of our problem.
Treat the arrival family as a **competition-data artefact** (ARR taxi-in is retained in ranking),
not as an established taxi-out feature.

---

## 7. Leakage: what is computable for a 2026 scored row

**[S] The governing asymmetry.** The organizers blank `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt`
**only on DEP rows**. On ARR rows both are supplied in `ranking.parquet`.
- `arrival_features.py:2–6`: *"Official schema removes taxi/block values only for DEP rows... Filter
  arrivals before reading taxi/block fields."*
- `matched_submit.py:135`: *"ARR in-block + taxi-in by stand. Ranking ARR TAXITIME is present and legal."*
- `OpenAir/status.md:102`: *"Ranking ARR is complete, including taxi-in."*
- `OpenAir/status.md:94`: *"Arrivals are not scored. Their taxi-in and timestamps are context."*

**[S]** OpenAir builds the ranking-time arrival pool from the **ranking file's own ARR rows**
(`submit.py:129–152`), and the training pool from the twelve `training_*.parquet` ARR rows
(`matched_submit.py:152–158`). Phoenix-Ops builds it from whichever frame is passed as `context`
(`arrival_features.py:20`), which at inference is the ranking file (`arrival_contest.py:121–122`).

| feature | 2026-scored-row availability | verdict |
|---|---|---|
| ARR `BLOCK_TIME_UTC_mvt`, `TAXITIME_SEC_mvt`, `SCHED_TIME_UTC_mvt`, `STAND_mvt`, `RUNWAY_mvt`, `AIRCRAFT_TYPE_mvt` | present in ranking | **legal** |
| `since_arr`, `arr_taxiin`, `arr_delay`, `arr_taxiin_2`, `arr_type`, `rwy_arr_taxiin` | derived from the above + the scored row's own `MVT_TIME_UTC_mvt` | **legal** |
| `since_arr_aobt`, `arr_taxiin_aobt` | need `AOBT_3_flt`, present on **98.47 %** of ranking DEP rows (status.md:102) | **legal, 1.5 % null** |
| `since_dep` (previous DEP take-off at the stand) | take-off times of other ranking DEPs are present | **legal** |
| previous DEP's **block time** or **taxi-out** at the stand | blanked for every ranking DEP | **ILLEGAL** |
| same-`FLIGHT_ID` ARR of this flight | that ARR is *after* this DEP | **ILLEGAL** (status.md:203, 855) |
| this flight's `ARVT_3` | after take-off | **ILLEGAL** (status.md:854) |
| previous same-callsign DEP's actual `TAXITIME` | blanked for other ranking DEPs | **ILLEGAL**; the legal substitute is its `MVT − AOBT` (status.md:204) |

**[X]** Two practical cautions that neither repo flags loudly:
1. **Pool-boundary truncation.** OpenAir's ranking-time arrival pool is *only* ranking ARR rows. A
   ranking DEP near the start of the ranking window sees a truncated stand history, so `since_arr`
   is null or too large there, while in training the pool spans 12 contiguous months. This is a
   train/serve distribution shift in the null rate. Phoenix-Ops's monthly `period` scoping in
   `nm_neighbor_features.py:14` is the mirror-image mitigation — they deliberately confine pools to
   one calendar month **in both** train and rank so the boundary behaviour matches.
2. **Both repos read the arrival's `BLOCK_TIME` (on-block) as the pool event time, not the landing
   time.** On-block is what the departure stand is actually released by, and it is the column that is
   supplied for ARR. Using landing time instead would be both weaker and (for the same-stand link)
   wrong.

---

## 8. Why 7 airports up and 3 down — and what it actually was

### 8.1 Our own numbers, both ways

From `docs/codex_fe_results/confirmation.json` and `continuation_evaluation.json` at commit `8bdf4f9`,
Nov–Dec 2025 confirmation, n = 328,009, 61 days. **Negative is better.**

| airport | baseline RMSE | **as post-hoc correction** Δ [99.167 % day CI] | **as a model feature** Δ [95 % day CI] |
|---|---:|---:|---:|
| LTFM | 227.22 | −3.175 [−3.980, −2.434] | −3.528 [−4.717, −2.204] |
| LIRF | 446.57 | −1.560 [−3.929, +0.309] | **−7.652** [−14.145, −0.712] |
| EGLL | 221.02 | −1.174 [−1.542, −0.820] | **−5.463** [−6.954, −3.990] |
| LFPG | 261.10 | −0.382 [−0.637, −0.142] | **−3.520** [−5.064, −2.185] |
| EDDM | 204.03 | −0.446 [−0.834, −0.117] | −2.764 [−4.111, −1.490] |
| EDDF | 179.87 | −0.147 [−0.263, −0.023] | **−3.316** [−4.770, −1.937] |
| LEBL | 172.34 | −0.272 [−0.395, −0.132] | −1.080 [−1.932, −0.382] |
| **EHAM** | 159.65 | **+0.374** [+0.194, +0.537] | **−0.722** [−1.350, −0.109] |
| **LEMD** | 171.46 | **+0.428** [+0.243, +0.597] | **−0.263** [−0.798, +0.247] |
| **LSZH** | 166.25 | **+1.035** [+0.600, +1.462] | +0.493 [−1.402, +3.460] |
| **all** | 228.32 | **−0.781** [−1.177, −0.396] | **−3.242** [−4.282, −2.225] |
| vs hash-noise control | | | −3.097 [−4.279, −1.949] |

**[S]** Three facts follow directly:
1. The three regressions were **not noise** — all three correction CIs lie entirely above zero.
2. **As a feature, the sign flips at EHAM and LEMD.** 9 of 10 airports improve; only LSZH is still
   positive, and its CI straddles zero widely ([−1.40, +3.46]), driven entirely by its unmatched
   slice (+42.7 on unmatched vs −0.853 on matched).
3. The feature form beats the correction form by a factor of **4.2×** overall (−3.242 vs −0.781),
   and beats a matched hash-noise control by −3.097, so the gain is not a fitting artefact.

**[X]** So the "7 of 10" failure mode is a property of the **post-hoc binned-residual mechanism**, not
of the information. A binned airport × regime × decile residual mean estimated on Sep+Oct and applied
to Nov+Dec adds a fixed per-cell offset. At EHAM (159.6), LSZH (166.3) and LEMD (171.5) — the three
**lowest-RMSE, most regular** airports in the set — the residual left to correct is small, so the
offset is dominated by estimation error and seasonal drift. A tree that has `ADEP_mvt` as a categorical
can learn to ignore the column at those airports; a per-cell offset table cannot.

### 8.2 A sourced structural mechanism, specifically for EHAM

**[S]** Schiphol's published stand-allocation regulation (RASAS 2025, §Towing, p. 6) sets a hard rule:

> *"To create connected gate capacity, aircraft with long ground time will temporarily be towed to a
> remote position, wait there and be towed back to a connected position before departure."*

with this norms table:

| condition | Cat. ≥5 | Cat. ≤4 |
|---|---|---|
| turnaround time triggering a tow | **≥ 210 min** | **≥ 170 min** |
| max stand occupation, inbound flight | 75 min | 55 min |
| target stand occupation, outbound flight | 85 min | 65 min |

> *"The target time for a towing procedure is 10 minutes; the time an aircraft spends at a parking
> stand should be at least 30 minutes. **An aircraft to be towed may have a different arrival and
> departure stand.**"*

Source: [Schiphol RASAS 2025 (EN), PDF](https://assets.ctfassets.net/biom0eqyyi6b/4WKbhQhXCHGBwnXhqSgWuQ/aca0896d2e7eb3311a15acbf0726ee07/Schiphol_RASAS_EN_2025_FINAL.pdf).

**[X]** My reading of what this does to `stand_arrival_gap` at EHAM: for every turnaround beyond
170/210 minutes, the departure stand is by regulation a *different* stand from the arrival stand, and
the departing aircraft has been parked there for a *target* 65–85 minutes. So in exactly the long-gap
regime our 6-hour cap admits, EHAM systematically produces a gap of ~1–1.5 h that (a) belongs to a
different aircraft and (b) says nothing about this aircraft's turnaround. A model with the airport
categorical can learn that discount; a global binned correction applies the pooled long-gap effect and
is wrong at EHAM. This is a hypothesis consistent with our numbers, **not a measured decomposition** —
I did not test it against an EHAM-only gap-length breakdown.

**[X]** I found no equivalent published stand-allocation regulation for LEMD or LSZH; the searches
returned only passenger-parking and chart material. LSZH is also the **smallest** airport in the set
(134,907 training DEP, 20,663 confirmation rows — `DISCOVERY_REPORT.md:25`), which is consistent with
its very wide feature-form CI. I am not able to give a sourced mechanism for LEMD.

### 8.3 What competitors did about per-airport heterogeneity: nothing selective

This is well evidenced and unanimous. **Nobody gated a turnaround feature per airport.**

| evidence | finding | source |
|---|---|---|
| Phoenix-Ops v9 | per-airport LightGBM experts exist, but *"the airport model alone worsens January and February"*; shipped at **12.5 % effective weight** (0.25 × 0.5) inside a global blend, never selectively | **[S]** METHOD.md:185, `ordinary_ensemble.py:52–53` |
| Phoenix-Ops v5 | *"airport-specific weights selected across the two months scored 323.712, worse than the fixed v4 blend"* | **[S]** METHOD.md:105 |
| OpenAir E67 | per-airport Δ experts *"worse on Jan+Jul (240.50) and unstable"* | **[S]** status.md:1699–1701 |
| OpenAir E75 | airport-specific Δ **237.58 vs 235.86** — *"Airport Δ hurts the ranking-like split"*; NNLS weights *"do not transfer (JJ 82 % v19; Dec 0 % v19)"* | **[S]** status.md:1746–1752, `results/E75/*.md` |
| OpenAir E25 | full per-airport architecture **473.61** vs E20 **368.03**; rejected | **[S]** status.md:275, 361 |
| OpenAir E13/E18 | every airport-conditional gate they tried had precision ≈ 0 or reversed between splits (EHAM unmatched mean y=742 vs train 565: *"the full-year 'EHAM shorter' fact is not a stable prior"*) | **[S]** status.md:227, 803, 833 |

**[S]** EHAM specifically is a repeat offender across unrelated feature families in OpenAir's log:
E14 dynamic surface state — *"All airports improve except EHAM"* (status.md:965); METAR airport expert
— EDDM −22, LFPG −15, EHAM −13 on December but Jan+Jul flat-to-worse (status.md:269, 929); E19 —
*"hurt LSZH +5.9, EHAM +1.9"* (status.md:991); E35-family — *"Matched gains: LSZH −7.4, LTFM −8.3,
EDDF −6.0; EHAM +2.2"* (status.md:1167). **[X]** EHAM is the airport where surface/context features
generically fail to transfer, which is at least consistent with the RASAS tow regime decoupling the
observable ground state from the departing aircraft's actual history.

**[X]** Conclusion for us: the correct response to our three regressions is **not** an EHAM/LEMD/LSZH
gate. Every published attempt at per-airport specialization in this competition lost. The correct
response is to use the column **as a feature in a global model that already has `ADEP_mvt` as a
categorical**, which our own measurement shows already fixes EHAM and LEMD.

---

## 9. Feature vs residual model vs post-processing — the reported difference

All three modes are documented, and the ordering is consistent across both repos and our own run.

| mode | who | measured effect |
|---|---|---|
| **Hard post-processing constraint** | OpenAir E72: cap `T ≤ since_arr` | **catastrophic** — LIRF-u 3,896 → 11,579; non-LIRF-u 1,546 → 1,568. **[S]** status.md:1725 |
| **Post-hoc binned residual correction** | our `8bdf4f9` scalar screen | **−0.781 s**, but +0.374/+0.428/+1.035 at EHAM/LEMD/LSZH. **[S]** `confirmation.json` |
| **Plain feature in a boosted residual model** | our `8bdf4f9` refit; OpenAir v16 | **−3.242 s** (ours, 9/10 airports); **−2.01 s LB** (OpenAir). **[S]** `continuation_evaluation.json`; status.md:1585 |
| **Same features, much larger residual model** | Phoenix-Ops v7 → v8 | **−7.11 s** officially, from capacity alone. **[S]** METHOD.md:157, 168 |
| **Same features, separate narrow-scope expert** | OpenAir v20 LIRF-G; Phoenix-Ops arrival specialist | **−1.54 s** from 383 rows (OpenAir); Phoenix-Ops specialist weighted 50 % inside its scope. **[S]** status.md:1742; `arrival_contest.py:138` |

**[S]** Two further corroborating data points that the *form* is what matters, not the information:
- OpenAir's own OLS check found *"< 1 s"* from recent taxi-in (`DISCOVERY_REPORT.md:275`) and their
  marginal correlations were 0.008–0.13 — yet the tree version shipped at −2.01 s.
- OpenAir E53-cascade and E54-metalayer (status.md:1605–1617): stage-3 residual-of-residual models and
  meta-layer stacking over **already-available** information gave **nothing** (all λ/gate combos worse
  or neutral). So "more post-processing" is a documented dead end; "more capacity on the primary
  residual" (Phoenix-Ops v8, OpenAir E68 `900×63` → `2000×127`) is the documented live one.

**[X]** Synthesis: the three modes are ranked **hard constraint ≪ post-hoc correction < feature <
feature + capacity**, and the gaps between adjacent rungs are large (roughly 4× from correction to
feature in our own controlled measurement, and another ~2.7× from feature to feature-plus-capacity in
Phoenix-Ops's official scores).

---

## Applicable to resilient-hedgehog

We are at 284.3584. Phoenix-Ops went 288.37 → 285.75 → **278.63** → 277.75 → 275.18 on this family
plus capacity; OpenAir went 284.10 → 282.09 → 281.72 → 280.18. Both passed straight through our
current score using the material below.

### A. Reimplementation-grade spec of Phoenix-Ops v8 (largest single documented gain, −7.1141 s)

**Understand first: v8 adds zero new features.** It is v7's 123 columns, re-learned by a 4,999-tree
depth-9 CatBoost on the same residual target, blended 50/50 with v7. Do not try to reproduce it by
writing more arrival features.

**Step 1 — the v7 feature matrix (prerequisite, worth −2.62 s on its own).**
Build the 26 columns of `arrival_features.py` exactly as specified in §1.3–1.4 above:
22 airport/runway trailing taxi-in statistics (counts, means, sds over 15/30/60 min, plus last-age and
last-taxi with a 2 h recency cap and tie-averaging), and 4 same-stand columns
(`last_age_seconds` ≤ **86,400 s**, `last_taxi_seconds`, `aircraft_equal`, `carrier_equal`),
with ambiguous same-stand timestamp ties left **missing** and `UNKNOWN` stands excluded. Event time for
the arrival pool is **`BLOCK_TIME_UTC_mvt` (on-block), not landing**. Query time is the departure's
`MVT_TIME_UTC_mvt`, comparison strictly-before (`searchsorted side='left'`). Filter `PHASE_mvt == 'ARR'`
**before** touching `BLOCK_TIME_UTC_mvt` or `TAXITIME_SEC_mvt`. Validity gate on the pool:
`landed.notna() & blocked.notna() & blocked ≥ landed & taxi ∈ [0, 7200]`. Emit `float32`.

**Step 2 — the residual target.** Fit on `(y − baseline).clip(−7200, +7200)` where `baseline` is our
own continuous off-block proxy (Phoenix-Ops use `MVT − AOBT_3`, falling back to `LOBT`, then `EOBT_1`,
then 1000 s, with a LIRF-`missing_IOBT` branch using `MVT − SCHED`). Do **not** clip the evaluation
labels — only the fitting residuals.

**Step 3 — the estimator.** CatBoost, `loss_function="RMSE"`, **`depth=9`, `learning_rate=0.05`,
`l2_leaf_reg=7`, `border_count=254`, `iterations=4999`**, `random_seed=20260907`, `task_type="GPU"`.
All string/object columns passed as `cat_features`. Fit on non-negative labels only, excluding any
separately-modelled scope.

**Step 4 — the blend.** `0.5 × current_best + 0.5 × new_candidate`, both floored at 0 before blending,
with any specialist-scope rows carried through **unchanged**.

**Step 5 — promotion criteria, declared before looking.** Phoenix-Ops pre-registered: ≥ 0.5 s overall,
improvement in **every** validation month, ≥ 40 of 62 days improved. They got 61/62. Adopt the same.

**[X]** Given our 284.36 and the fact that the −7.1 came from capacity rather than semantics, the
single highest-expected-value experiment available to us is: **take whatever residual model we
currently run, add the arrival columns, and then raise the tree count and depth by roughly an order of
magnitude** (Phoenix-Ops: 679-tree / 63-leaf LightGBM → 4,999-tree depth-9 CatBoost; OpenAir E68
independently found `2000×127` and `1500×255` both beat their shipped default
`n_estimators=400, num_leaves=31` — `matched_submit.py:41`, status.md:1703). Capacity on the
residual is the one lever both independent teams confirmed.

### B. What we already have, and the one column we are missing

**[S]** `src/prc/features.py:294–336` at `8bdf4f9` implements `stand_arrival_features`, producing
`stand_arrival_gap`, `stand_aobt_turn`, `stand_arrival_type_match`, keyed `(ADEP_mvt, STAND_mvt)` via
`join_asof(..., strategy="backward", allow_exact_matches=False)` on `BLOCK_TIME_UTC_mvt`, capped at
**21,600 s (6 h)**. That is structurally OpenAir's joins A and B and Phoenix-Ops's stand block.

**[S]** We do **not** currently ship `arr_taxiin` — the joined arrival's `TAXITIME_SEC_mvt`. It exists
as a screened candidate named `stand_arrival_taxi` (codex report line 89, correction Δ −0.441) but was
not among the five selected and was never refitted as a feature. **This is the exact column that moved
OpenAir 284.10 → 282.09 (−2.01 s LB), the largest single feature-level gain in their log.**
Likewise `stand_arrival_delay` (= OpenAir's `arr_delay`, `AIBT − SIBT`) was screened (−0.408) and
dropped.

**[X]** Highest-value cheap action: refit with `stand_arrival_gap` **plus** `stand_arrival_taxi`
(+ `stand_arrival_delay`, `stand_arrival_type_match`). Our own screen ranked them by *correction*
value, which §9 shows is the wrong ordering criterion — `stand_arrival_gap` ranked first as a
correction (−0.991) and `stand_arrival_taxi` fourth (−0.441), yet OpenAir's LB says the taxi-in column
is the load-bearing one. Re-rank by feature-refit ΔRMSE, not by correction ΔRMSE.

### C. Use it as a feature, not as a correction — settled by our own data

**[S]** `stand_arrival_gap`: **−0.781 s** as a post-hoc correction (7/10 airports), **−3.242 s
[−4.282, −2.225]** as a model feature (9/10 airports), **−3.097 s vs a matched hash-noise control**.
EHAM flips +0.374 → −0.722 and LEMD +0.428 → −0.263 purely by changing the mode of use.
Source: `docs/codex_fe_results/confirmation.json` and `continuation_evaluation.json` at `8bdf4f9`.

**[X]** Therefore: retire the residual-correction framing for this family entirely. Do not build an
EHAM/LEMD/LSZH gate — every per-airport specialization attempted by either competitor lost
(§8.3, six independent citations). Keep `ADEP_mvt` as a categorical and let the model decide.

### D. Concrete additions worth testing, in expected-value order

1. **`stand_arrival_taxi`** (the joined arrival's taxi-in). OpenAir: −2.01 s LB. **[S]**
2. **Capacity on the residual model** (see A, step 3). Phoenix-Ops: −7.11 s. **[S]**
3. **Widen the cap from 6 h**: Phoenix-Ops use 24 h for the same-stand link, OpenAir 8 h for matched
   rows and **36 h** for their LIRF gate-hold model. Our 6 h is the tightest of the four. **[S]**
4. **Add the AOBT-anchored twin** `since_arr_aobt` / `arr_taxiin_aobt` (we have `stand_aobt_turn`,
   which is the first of the pair, but not the second). OpenAir: part of the −0.21 s v17 step. **[S]**
5. **`arr_delay` = `AIBT − SIBT`** of the joined arrival, and **`arr_taxiin_2`** (`shift(1)` over
   `(airport, stand)` sorted by `aibt`), and **`rwy_arr_taxiin`** (last arrival on the same runway).
   OpenAir: the v17+v18 tail, −0.37 s combined. **[S]**
6. **Phoenix-Ops's 22 airport/runway trailing taxi-in statistics.** We appear to have none of these;
   they are half of the −2.62 s v7 step and are cheap (one `searchsorted` per group). **[S]**
7. **A LIRF-scoped variant with a 36 h window fed to a gate-hold model.** OpenAir got −1.54 s LB from
   **383 rows** this way — the highest leverage per row anywhere in either log. **[S]**

### E. Do not do these

**[S]** (a) Any hard bound of the form `taxi_out ≤ stand_arrival_gap` — OpenAir E72: LIRF-u
3,896 → 11,579. (b) Any use of `FLIGHT_ID_mvt` as a rotation/tail key — it is a city-pair flight ID and
its paired ARR is *after* the DEP. (c) Any feature reading another ranking DEP's `BLOCK_TIME_UTC_mvt`
or `TAXITIME_SEC_mvt` — blanked. (d) External ADS-B registration chaining (OPDI `icao24`) — measured
**worse** (236.06 vs 235.94). (e) Stage-3 residual-of-residual cascades or meta-layer stacking over
information the model already has — OpenAir E53/E54, uniformly neutral-to-worse.

### F. Honest limits of this slot

**[X]** (1) I did not run anything; every number above is quoted from a repo artefact. (2) The
`continuation_evaluation.json` gap refit used descriptive 95 % intervals chosen *after* seeing the
confirmation outcome, as its own report states — treat −3.242 as a strong indication, not a
pre-registered result. (3) The EHAM/RASAS mechanism is a hypothesis consistent with our per-airport
numbers; I did not test it against an EHAM gap-length breakdown. (4) I have no sourced mechanism for
LEMD. (5) Neither competitor published a per-airport breakdown *of the arrival features specifically*,
so the cross-airport evidence in §8.3 is about per-airport *models*, not per-airport arrival effects.

**Sources:**
- [Phoenix-Ops / zestful-fountain, prc2026-taxiout](https://github.com/Phoenix-Ops-LTD/prc2026-taxiout)
- [OpenAir / likable-eagle](https://github.com/ChandanHegde07/OpenAir)
- [gultekin, prc-taxiout-2026](https://github.com/ahmetabdullahgultekin/prc-taxiout-2026)
- [EUROCONTROL / SES Performance — Additional taxi-out time metadata](https://www.sesperformance.eu/dataportal/metadata/additional-taxi-out-time/)
- [Schiphol RASAS 2025 (EN)](https://assets.ctfassets.net/biom0eqyyi6b/4WKbhQhXCHGBwnXhqSgWuQ/aca0896d2e7eb3311a15acbf0726ee07/Schiphol_RASAS_EN_2025_FINAL.pdf)
- [PRC Data Challenge 2026 — ranking dataset schema](https://prc-data-challenge-2026.netlify.app/data.html#the-ranking-dataset)
