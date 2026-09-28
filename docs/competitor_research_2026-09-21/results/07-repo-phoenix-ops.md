# SLOT 07 — repo-phoenix-ops: code-level read of Phoenix-Ops-LTD/prc2026-taxiout

Team `zestful-fountain`. Public leaderboard 273.5557 (v12).
Repo cloned to `/tmp/prc2026-taxiout` at commit `4477d9b` (HEAD, 2026-09-13T20:36:08+02:00), 24 commits, GPL-3.0-only.
All line citations are against that checkout. Marks: **[S]** = read from the repo (code / METHOD.md / results snapshots), **[X]** = my own inference.

Feature counts below were **verified by execution [S]**: I stubbed `pydantic`, built a synthetic
frame with the real column names, and ran the actual `matrix()` functions
(`/tmp/claude-0/.../scratchpad/enum_features.py`). Output: `contest`=65, `carrier`=70,
`duration`=93, `arrival`=123, `ordinary`=153, `following`=183 — exactly the counts METHOD.md claims.

---

## 0. TL;DR — what this repo actually is

**[S]** It is not a taxi-out forecaster. It is a **reconstruction of the blanked `BLOCK_TIME_UTC_mvt`
from the NM (`_flt`) clocks that the organizers left in the ranking file**, plus a GBDT that learns
the residual of that reconstruction.

The chain is (`traffic_features.py:81-121`):

```
planned_baseline            = clip0(MVT_TIME - EOBT_1_flt)  if in [-7200,172800] and ADEP_flt==ADEP_mvt else 1000.0
nm_observed_baseline        = same, overridden by LOBT_flt, then overridden by AOBT_3_flt   (AOBT_3 wins)
nm_schedule_fallback_baseline = above, but for ADEP_mvt=='LIRF' & IOBT_flt missing:
                                clip(MVT_TIME - SCHED_TIME, 0, 172800)
target fed to every model   = TAXITIME_SEC_mvt - baseline        (a residual)
prediction                  = max(0, model_residual + baseline)
```

Our own `docs/CODEX_BRIEF.md:15` records `TAXITIME_SEC_mvt == MVT_TIME_UTC_mvt − BLOCK_TIME_UTC_mvt`
on 2,085,047/2,085,047 training rows, and `MVT_TIME_UTC_mvt` is take-off. So
`MVT_TIME − AOBT_3_flt` **is the label whenever NM's actual off-block equals the airport's**. [X]
Every "feature" family they later add — arrival taxi-in, NM-neighbour proxies, following-context
proxies — is a **peer-value of that same proxy or of taxi time**, i.e. a local correction to
`AOBT_3 − BLOCK` bias. That is exactly the axis our own model (`CODEX_BRIEF.md:54`) is already on,
which is why their incremental deltas (not their baseline) are the transferable part.

METHOD.md states this openly: "They are retrospective observations, unsuitable for claiming a
predeparture operational forecast" (`METHOD.md:15`). **[S]**

---

## 1. The 183 features — exact definitions

Verified names/order by execution. Blocks are concatenated in this order; the final model's
`cat_features` are all `object`-dtype columns.

### 1.1 Block A — 65 "contest" features (`pipeline.py:73-107`, `traffic_features.py:11-104`)

| # | name | definition | source |
|---|---|---|---|
|1-7|`ADEP_mvt, ADES_mvt, AIRCRAFT_TYPE_mvt, FLIGHT_RULE_mvt, MARKET_SEGMENT_flt, WK_TBL_CAT_flt, AIRCRAFT_OPERATOR_flt`|raw categories, `fillna("UNKNOWN").astype(str)`|`pipeline.py:23,79-80`|
|8-12|`hour_utc, weekday_utc, month_utc, hour_sin, hour_cos`|from `SCHED_TIME_UTC_mvt`|`pipeline.py:81-85`|
|13|`scheduled_departures_30m`|group size over (`ADEP_mvt`, `SCHED_TIME` floored to 30 min) — **includes self**|`pipeline.py:88-89`|
|14-17|`RUNWAY_mvt, airport_RUNWAY_mvt, STAND_mvt, airport_STAND_mvt`|supplied runway/stand + `ADEP:VALUE` string pairs|`pipeline.py:93-96`|
|18-21|`movement_hour_utc, movement_weekday_utc, movement_month_utc, movement_schedule_delta_sec`|from `MVT_TIME_UTC_mvt`; delta = `MVT−SCHED` clipped to [−86400, 172800]|`pipeline.py:98-102`|
|22|`movement_departures_30m`|group size over (`ADEP_mvt`, `MVT_TIME` floored 30 min), **includes self**|`pipeline.py:103-104`|
|23-38|16 trailing-traffic features|see below|`traffic_features.py:11-47`|
|39-41|`scheduled_minute_of_day, movement_minute_of_day, movement_day_of_year`||`traffic_features.py:55-57`|
|42-43|`airport_operator` = `ADEP:OPERATOR`; `stand_runway` = `ADEP:STAND:RUNWAY`||`traffic_features.py:58-59`|
|44-49|`movement_minus_IOBT_flt, schedule_minus_IOBT_flt, missing_IOBT_flt`, same trio for `EOBT_1_flt`|seconds, clipped ±604800; `missing_*` = isna indicator|`traffic_features.py:68-73`|
|50-53|`ADEP_flt, ADES_flt, FLIGHT_TYPE_flt, AIRCRAFT_TYPE_flt`|NM categories|`traffic_features.py:74-75`|
|54-55|`nm_departure_matches`, `nm_destination_matches`|`ADEP_flt==ADEP_mvt`, `ADES_flt==ADES_mvt` (after `UNKNOWN` fill)|`traffic_features.py:76-77`|
|56-63|`movement_minus_X` + `missing_X` for `AOBT_3_flt, LOBT_flt, ARVT_1_flt, ARVT_3_flt`|`MVT_TIME − X`, clipped ±604800|`traffic_features.py:97-101`|
|64|`nm_actual_vs_estimated_seconds`|`movement_minus_EOBT_1 − movement_minus_AOBT_3` (= AOBT_3 − EOBT_1)|`traffic_features.py:102`|
|65|`nm_last_vs_actual_seconds`|`movement_minus_AOBT_3 − movement_minus_LOBT` (= LOBT − AOBT_3)|`traffic_features.py:103`|

**Trailing traffic (features 23-38)** — `traffic_features.py:11-47` **[S]**:
- Context pool = **all** movement rows of the same file (DEP and ARR handled separately).
- Airport key for a pool row = `ADEP_mvt` if `PHASE=='DEP'` else `ADES_mvt` (arrivals count at their destination).
- Runway = `RUNWAY_mvt.fillna("UNKNOWN")`.
- For each of {DEP, ARR} × {airport-scope, airport+runway-scope}: counts in the trailing 15/30/60 min plus a headway.
- Window: `[q − W, q)` — `np.searchsorted(events, q, side="left")` as the right edge, so **every event at exactly the query timestamp is excluded, including the row itself** (`traffic_features.py:39-41`, docstring line 12).
- `*_headway_seconds` = `min(3600, q − previous_event)`, **3600 when no prior event** (no NaN) (`:42-43`).
- Nulls: counts default `0`, headway default `3600`; all float32.

### 1.2 Block B — +5 carrier features (66-70) (`traffic_features.py:124-139`)
`movement_carrier` = regex `^([A-Za-z]{2,4})` on `FLIGHT_mvt`, uppercased, `UNKNOWN` fallback;
`airport_movement_carrier` = `ADEP:CARRIER`; `scheduled_day` (day of month); `movement_second`;
`schedule_second` (literal seconds field of the two timestamps — a data-artefact feature).
Full flight numbers and IDs are deliberately excluded (`:127-129`).

### 1.3 Block C — +23 duration features (71-93) (`duration_contest.py:30-58`)
- 5 × 2 record-agreement pairs `{callsign, aircraft, origin, destination, rule}` × `{_both_present, _equal}`,
  comparing `FLIGHT_mvt/CALLSIGN_flt`, `AIRCRAFT_TYPE_mvt/_flt`, `ADEP_mvt/_flt`, `ADES_mvt/_flt`,
  `FLIGHT_RULE_mvt/_flt` after `.str.strip().str.upper()` (`:34-44`).
- `airport_route`, `airport_aircraft`, `airport_carrier_runway` (string interactions, `:45-47`).
- `nm_planned_flight_duration` = `movement_minus_EOBT_1 − movement_minus_ARVT_1` (= ARVT_1 − EOBT_1);
  `nm_actual_flight_duration` = ARVT_3 − AOBT_3; `nm_flight_duration_change` = actual − planned;
  `nm_duration_ratio` = actual / max(planned,120); two `_plausible` flags = `between(120, 100000)` (`:48-53`).
- `nm_actual/estimated/initial_offblock_schedule_delay` = `movement_schedule_delta_sec − movement_minus_{AOBT_3,EOBT_1,IOBT}` (`:54-56`).
- `schedule_delay_days` = `floor(movement_schedule_delta_sec / 86400)` (`:57`).
- Nulls propagate as NaN (no imputation).

### 1.4 Block D — +4 service-name features (94-97) (`arrival_contest.py:34-36`)
`movement_service` = normalized `FLIGHT_mvt`, `nm_service` = normalized `CALLSIGN_flt`, and each
prefixed with `ADEP_mvt`. `normalized()` = strip/upper/empty→`UNKNOWN` (`arrival_features.py:12-13`).
METHOD calls these "recurring service strings, not IDs" (`METHOD.md` v7 section).

### 1.5 Block E — the 26 v7 arrival / taxi-in / same-stand features (98-123)
`arrival_features.py:15-77`. **This is the interesting one.** **[S]**

**Pool construction (`:18-30`)**
- `context.loc[PHASE_mvt=='ARR']` — the ARR filter happens **before** any `TAXITIME_SEC_mvt` /
  `BLOCK_TIME_UTC_mvt` read (explicit comment `:16-17`). On ARR rows the organizers keep those fields.
- Validity: `MVT_TIME` (touchdown) and `BLOCK_TIME` (on-block) both present, `BLOCK ≥ MVT`, and
  `TAXITIME_SEC_mvt ∈ [0, 7200]` (`:24`).
- `completed_seconds` = **arrival on-block epoch** (not touchdown) — this is the "completion" time.
- `taxi_seconds` = the arrival's taxi-**in** seconds.
- Keys: `airport = normalized(ADES_mvt)`, `runway`, `stand`, `aircraft = AIRCRAFT_TYPE_mvt`,
  `carrier` = `^([A-Z]{2,4})` of `FLIGHT_mvt`.

**Query side (`:31-34`)** — query time = the departure's `MVT_TIME_UTC_mvt` (**take-off**), keys are
the departure's `(ADEP_mvt, RUNWAY_mvt, STAND_mvt)`, all normalized.

**Airport scope (11 cols, 98-108) and runway scope (11 cols, 109-119)** — `:36-61`:
- Join key: airport, or (airport, runway). No date/month restriction.
- Windows `[q − 15/30/60 min, q)` via `searchsorted(..., side='left')` ⇒ **strictly before the query
  take-off time**; prefix-sum mean and population std (`sqrt(max(0, E[x²]−E[x]²))`), plus count.
- `last_age_seconds`, `last_taxi_seconds`: the most recent completion before `q`, **only if age ≤ 7200 s**,
  otherwise NaN (`:58-59`). Simultaneous completions are **averaged**, not ordered (`:54-57`).
- Nulls: means/stds NaN when count 0; counts 0.
- Self-exclusion is structural: a DEP row can never be in an ARR pool. **[S]**

**Same-stand scope (4 cols, 120-123)** — `:62-77`:
- Key `(airport, stand)`, and groups whose stand is `UNKNOWN` are **dropped from the lookup**
  (`if key[1]!='UNKNOWN'`, `:63`), so missing stand ⇒ all four NaN.
- Takes the last arrival on-blocked at that stand before the departure's take-off, allowed age
  **≤ 86400 s** (24 h, vs 2 h for airport/runway).
- Ties at the same `completed_seconds` are **rejected**, not averaged: `ambiguous = duplicated(keep=False)`
  and the row is skipped (`:70-71`) ⇒ NaN.
- Emits `last_age_seconds`, `last_taxi_seconds` (that arrival's taxi-in), `aircraft_equal`,
  `carrier_equal` (does the departing aircraft type / carrier match the last arrival on that stand ⇒ turnaround
  identification). **[X]** `aircraft_equal=1 & small age` is a strong "this is the same airframe turning around" signal.

**Availability caveat [X]:** the cut-off is the departure's **take-off**, not its off-block. An arrival
that on-blocks *during* this flight's taxi-out is included. So these features leak information from the
interval the model is predicting. Legal under the challenge (the fields are supplied) but not
operationally causal.

### 1.6 Block F — the 30 v9 NM-neighbour peer proxies (124-153)
`nm_neighbor_features.py:6-48`. **[S]**

- **Pool**: `context.loc[PHASE_mvt=='DEP']`, columns `ADEP_mvt, ADEP_flt, RUNWAY_mvt, STAND_mvt, MVT_TIME_UTC_mvt, AOBT_3_flt` only (`:7-8`) — **no target, no block time is even selected**.
- **Proxy** = `MVT_TIME − AOBT_3_flt` in seconds, accepted only when `∈ [0, 7200]` **and**
  `norm(ADEP_mvt) == norm(ADEP_flt)` (`:10-11`). That proxy is the peer's reconstructed taxi-out.
- **`period` = `MVT_TIME.strftime('%Y-%m')`** — every group is confined to one **calendar month** (`:14, :22`).
- Three group levels (`:24`): `(airport, period)`, `(airport, period, runway)`, `(airport, period, stand)`.
  For runway/stand levels, pool rows whose key is `UNKNOWN` are dropped (`:25`), so missing runway/stand ⇒ NaN block.
- **Windows**: `[q − 15 min, q)` and `[q − 60 min, q)`; right edge `searchsorted(..., side='left')`
  ⇒ **strictly earlier; the query row and every simultaneous movement are excluded** (`:33-36`). This is
  the only self-exclusion mechanism — it is timestamp-based, not ID-based. **[X]** two departures sharing
  an exact take-off second exclude each other.
- Per group × window (10 cols each level): `count`, `mean`, population `std`,
  `own_minus_mean` = own valid proxy − window mean. `own_values` is NaN unless the row's own proxy is
  valid by the same [0,7200] + airport-match rule (`:19-20`) ⇒ `own_minus_mean` NaN for unmatched rows.
- `last_age_seconds`, `last_proxy`: most recent earlier movement, age **≤ 7200 s**; all events tied at that
  timestamp are **averaged** (`:42-47`) — order-invariant by construction.
- Names: `nm_neighbor_{airport|runway|stand}_{15m|60m}_{mean|std|own_minus_mean}`,
  `..._{15m|60m}_count`, `..._last_age_seconds`, `..._last_proxy` (10 × 3 = 30).

### 1.7 Block G — the 30 v12 following-context features (154-183)
`nm_following_features.py:14-59`. **[S]** Identical construction to Block F with three changes:

1. **Windows are strictly after the query**: left edge `searchsorted(t, q, side='right')`
   (`:45`, comment "Exclude the query itself and every movement at the exact query time"), right edge
   `searchsorted(t, q + W, side='right')` ⇒ the interval is **`(q, q + W]`**, inclusive at the top.
   Windows are **15 and 60 minutes** (`:47-48`).
2. Instead of "last earlier" it emits `next_wait_seconds` (time to the **nearest strictly later**
   movement, capped at ≤ 7200 s else NaN) and `next_proxy` (that movement's proxy, averaged over ties)
   (`:54-58`).
3. Same `%Y-%m` month confinement, same runway/stand `UNKNOWN` dropping, same own-proxy validity rule.

Names `nm_following_{airport|runway|stand}_{15m|60m}_{mean|std|own_minus_mean|count}`,
`..._next_wait_seconds`, `..._next_proxy`.

**[X]** This is a pure hindsight feature: it tells the model how congested the *next hour* was, i.e. what
happened after the flight it is predicting. It is computable on the ranking file (all DEP rows carry
`MVT_TIME` and `AOBT_3_flt`) but has no predeparture analogue.

### 1.8 v8 "deeper arrival residual" — **not a feature set** [S]
METHOD.md's "deeper arrival residual (v8): −7.1" is **not new features**. `arrival_boost_contest.py:15,61`
reuses `arrival_contest.matrix` — the **same 123 predictors** — and changes only the estimator and the
fitting target:
- CatBoost `depth=9, lr=0.05, l2_leaf_reg=7, border_count=254, seed=20260907, GPU, thread_count=2`,
  **4,999 trees fixed**, no early stopping (`:29-33`, METHOD v8 section).
- Fitting rows exclude the unmatched-LIRF specialist scope (`:66`).
- **Fitting residuals clipped to ±7200 s**: `model.fit(x, (y-offset).clip(-7200,7200))` (`:77`). Evaluation
  labels are never clipped.
- Combined as a flat **50/50 blend with v7** outside the specialist scope (`:43-45`).

So the single largest gain in their whole log (−7.11 s official) came from **a deeper/longer CatBoost on
capped residuals blended 50/50**, not from new information. **[X]** That is the most transferable finding
in this repo.

---

## 2. Models, hyperparameters, routing, seed averaging

### 2.1 Every estimator in the final stack [S]

| Component | Library / params | Trees | Rows fitted |
|---|---|---|---|
| `contest.py` NM residual (v2) | CatBoost RMSE, depth 8, lr .055, seed 20260907, GPU, `thread_count=6`, early stop 120 | ≤2200, selected 2199 | all non-negative labels (2,084,678) |
| `carrier_contest.py` global (v3) | CatBoost depth **9**, lr **.045**, `l2_leaf_reg=5`, seed 20260907, GPU, threads 4 | **2980** fixed | all non-negative (`carrier_contest.py:50-57`) |
| `carrier_contest.py` specialist | CatBoost depth **5**, lr **.04**, `l2_leaf_reg=10`, seed 20260907, **CPU**, threads 2 | **1014** fixed | **1,488** rows (`:50-57`, METHOD v3) |
| `ensemble_contest.py` LightGBM (v4) | `num_leaves=63, lr=.035, min_child_samples=80, reg_lambda=10, colsample_bytree=.9, n_jobs=2, deterministic=True, force_col_wise=True`, seed 20260907 | **679** | all non-negative (`ensemble_contest.py:57-59`) |
| `duration_contest.py` CatBoost93 (v5) | depth 9, lr .05, `l2_leaf_reg=7`, `border_count=254`, seed 20260907, GPU, threads 4 | **4998** | all non-negative (`duration_contest.py:109-112`) |
| `arrival_specialist.py` (v7 specialist) | CatBoost depth 5, lr .04, l2 10, CPU, `thread_count=1` | 1014 | 1,488 (`arrival_specialist.py:62-64`) |
| `arrival_contest.py` LightGBM123 (v7 global) | identical LightGBM params, 679 trees | 679 | all non-negative (`arrival_contest.py:77-83`) |
| `arrival_boost_contest.py` CatBoost123 (v8) | depth 9, lr .05, l2 7, borders 254, GPU, threads 2 | **4999** | 2,083,190 ordinary-scope, residual cap ±7200 |
| `ordinary_ensemble.py` LightGBM153 + 10 airport experts (v9) | same LightGBM params, 679 trees each, 11 models | 679 | ordinary scope; **airport experts use only the 123 arrival columns** (`ordinary_ensemble.py:99`) |
| `neighbor_boost_contest.py` CatBoost153 (v10) | identical to v8 CatBoost, 4999 GPU trees enforced for `data_class=="competition"` (`:39-48`) | 4999 | ordinary scope, cap ±7200 |
| `following_boost_contest.py` CatBoost183 (v12) | identical, 4999 GPU trees enforced | 4999 | ordinary scope, cap ±7200 |
| `long_following_contest.py` LightGBM183 (v13, local only) | same LightGBM params, **2716 CPU trees** (4× 679) | 2716 | ordinary scope, cap ±7200 |

Note: **the deep CatBoosts have no early stopping at all** from v8 onward; tree counts were frozen before
the validation result was read (`METHOD.md` v8/v10/v12 sections). **[S]**

### 2.2 Specialist routing rule [S]
One rule, unchanged since v3 — `carrier_contest.py:34-35`:

```python
def specialist_scope(x): return x["ADEP_mvt"].eq("LIRF") & x["missing_IOBT_flt"].eq(1)
```

i.e. **Rome Fiumicino departures with no NM `IOBT_flt`**. 1,488 non-negative training rows; 383 ranking
rows (398 rows in the Jan/Jul holdout). For that scope the baseline offset also changes to
`clip(MVT_TIME − SCHED_TIME, 0, 172800)` (`traffic_features.py:116-121`) — METHOD.md:21 says this came
from inspecting Jan–Jun 2025 rows "where ground-source records encode very long delays that tree leaves
cannot extrapolate".

**Combination** (`carrier_contest.py:38-47`): the specialist **replaces the global residual in-place**
for scope rows, then a single `+ baseline` and `clip(lower=0)` is applied. From v5 onward every later
stage instead **protects** the scope: each blend/replacement writes `result.loc[scope] = baseline.loc[scope]`
(`duration_contest.py:74-75`, `arrival_boost_contest.py:44-45`, `ordinary_ensemble.py:53-54`,
`neighbor_boost_contest.py:40-45`, `following_boost_contest.py:64-71`). v7 is the single exception: it
blends `0.5*v6 + 0.5*arrival_specialist` inside the scope (`arrival_contest.py:138`).

### 2.3 Seed averaging [S]
- v6 (`specialist_ensemble.py`): seeds **20260908-20260911** are fitted fresh; the already-submitted v5
  specialist (seed 20260907) is reused **from the v5 prediction file**, and the five values are averaged
  with equal weight — `np.mean(np.vstack([baseline[scope], experts]), axis=0)` (`:37-50`). Each constituent
  is floored at 0 **before** averaging (`:46`). A validator forbids reusing seed 20260907 and requires
  distinct seeds (`:27-34`).
- v7 (`arrival_specialist.py:20`): all five seeds `20260907..20260911` are refitted on the
  70+26 = 96-column specialist matrix and averaged equally (`:98`). No seed was selected post-hoc
  (METHOD v6/v7 sections).
- **Seed averaging is used only on the 1,488-row specialist**, never on the global models. **[S]**

### 2.4 The full v12 composition (reconstructed from code) [S for each term, [X] for the algebra]

```
v3  = Cat70_global (+specialist residual swap on LIRF∧missing-IOBT)              carrier_contest.py:38-47
v4  = 0.75·Cat70 + 0.25·LGB70            (each floored at 0 first)               ensemble_contest.py:27-31
v5  = 0.25·v4    + 0.75·Cat93                                                    duration_contest.py:66-76
v6  = v5, specialist rows -> mean(5 seeds)                                       specialist_ensemble.py:37-50
v7  = 0.75·v6 + 0.25·LGB123 ; specialist rows: 0.5·v6 + 0.5·arrivalSpec5         arrival_contest.py:43-53,138
v8  = 0.50·v7 + 0.50·Cat123deep(4999)                                            arrival_boost_contest.py:36-46
v9  = 0.75·v8 + 0.25·(0.5·LGB153neighbour + 0.5·LGB123airport-expert)            ordinary_ensemble.py:45-55
v10 = v9  + 0.375·(Cat153 − Cat123deep)                                          neighbor_boost_contest.py:29,57-72
v11 = clock overlay on 21 ranking rows                                           nm_clock_overlay.py:104-116
v12 = v11 + 0.375·(Cat183 − Cat153)   [outside specialist AND clock scopes]      following_boost_contest.py:33,82-96
v13 = v12 + 0.125·(LGB183/2716 − LGB153/679)        (local only, unsubmitted)    long_following_contest.py:32
```

The `0.375` is not tuned: it is the algebraic weight the old CatBoost already carries
(0.5 in v8 × 0.75 of v8 in v9). `0.125` = 0.25 candidate × 0.5 neighbour (METHOD v13 section). **[S]**
Each component is floored at 0 individually before the subtraction; **the replacement result itself is
never clipped — a negative result raises instead** (`neighbor_boost_contest.py:46-47`,
`following_boost_contest.py:68-69`). **[S]**

### 2.5 The v11 clock overlay, exactly [S] (`nm_clock_overlay.py:104-116`)
```python
origin_matches = ADEP_flt.fillna("UNKNOWN") == ADEP_mvt          # no normalization, exact string
special        = (ADEP_mvt == "LIRF") & IOBT_flt.isna()          # never touched
actual = clip(MVT_TIME - AOBT_3_flt, -604800, 604800)
last   = clip(MVT_TIME - LOBT_flt,   -604800, 604800)
disagreement = (AOBT_3_flt - LOBT_flt).total_seconds()           # signed, unclipped
trigger = ~special & origin_matches
        & actual.notna() & actual.between(-7200,172800)
        & last.notna()   & last.between(-7200,172800)
        & (disagreement > 7200)                                  # strict
prediction[trigger] = max(0, last)                               # full replacement, not a blend
```
Pure function: no file/model/target access (`:2-8`), and it asserts afterwards that nothing outside
`trigger` changed (`:119-120`). It fires on **21 of 344,841 ranking rows** and 31 of 344,419 Jan/Jul rows.
**[X]** Semantics: when NM's "actual" off-block is more than 2 h *later* than the last-known off-block,
AOBT_3 is treated as corrupt and LOBT is used instead. It is a bad-record repair, not modelling.

---

## 3. Complete experiment log, in order

All values **[S]**; official scores from METHOD.md + `results/leaderboard-2026-09-*.md`.
"Local" = January+July 2025 reused holdout (see §4).

### 3.1 Submitted versions

| v | date | what changed | local Jan/Jul | official RMSE | Δ official | rank |
|---|---|---|---|---|---|---|
|v1|—|first pipeline|—|**461.4635**|—|—|
|v2|2026-09-07|`contest.py` 65-feature NM residual, CatBoost depth 8|331.284|**297.5883**|−163.9|—|
|v3|09-08 04:04|+5 carrier features (70), depth 9/lr .045/l2 5/2980 trees, + LIRF specialist|324.065 (global alone 327.340)|**292.4043**|−5.18|12/69|
|v4|09-08 04:26|fixed 25 % LightGBM679 / 75 % CatBoost blend|**323.377**|**291.4829**|−0.92|12/69|
|v5|09-09 18:40|93 duration features, CatBoost 4998 trees; 75 % duration / 25 % v4|**321.589** (raw 324.864; +specialist 321.832)|**290.0659**|−1.42|19/89|
|v6|09-10 05:09|5-seed specialist average|**321.005**|**288.3714**|−1.6945|22/95|
|v7|09-10 05:58|26 arrival features → LGB123 at 25 %; arrival specialist at 50 % in scope|**318.547**|**285.7490**|−2.6224|19/96|
|v8|09-13 09:54|CatBoost123 depth 9 / 4999 trees / residual cap ±7200, 50/50 with v7|**315.778**|**278.6349**|**−7.1141**|17/127|
|v9|09-13 10:23|LGB153 (+30 neighbour) & 10 airport experts, equal, at 25 %|**315.179**|**277.7530**|−0.8819|16/129|
|v10|09-13 12:04|CatBoost153 replaces CatBoost123 at 0.375|**314.3197465013**|**275.1751**|**−2.5779**|14/130|
|v11|09-13 12:34|clock overlay, 21 rows|**313.7463286061**|**274.9738**|−0.2013|14/130|
|v12|09-13 16:53|CatBoost183 (+30 following) replaces CatBoost153 at 0.375|**312.9710103047**|**273.5557**|−1.4181|**12/131**|
|v13|prepared 09-13 20:36|LightGBM183 @2716 trees replaces LGB153@679 at 0.125|**312.4703995155**|not submitted at HEAD|—|—|

Note: METHOD.md's prose "v10: −2.6" is the **official** 277.7530→275.1751 = 2.5779 s gain, not a local one. **[S]**

### 3.2 Early feature-ablation ladder (`METHOD.md:24-33`, Jul–Dec holdout)

| candidate | RMSE |
|---|---|
| original context, 500 trees | 389.207 |
| context, 1000 trees | 382.284 |
| traffic features with **capped labels** | **434.145 — rejected** |
| planned-time residual (EOBT_1) | 355.680 |
| supplied NM records, residual | 342.865 |
| NM residual + unmatched-LIRF fallback | **267.998** |
| same, January/July split | 331.284 |

### 3.3 Everything they report as failed / rejected [S]

| experiment | result | verdict |
|---|---|---|
| Traffic features with capped **labels** (not residuals) | 434.145 vs 382.284 | rejected (`METHOD.md:28`) |
| Larger unmatched-LIRF specialist | — | not retained (v3 section) |
| Matched-LIRF LightGBM, residual fit | 395.047 vs 393.384 on that scope | rejected |
| Matched-LIRF direct-target fit | 629.654 | rejected |
| Capped-residual **global** model (v3 era) | deteriorating validation | stopped early |
| 50/50 CatBoost/LightGBM blend | weaker, worsened July | rejected (25 % chosen) |
| Non-LIRF unmatched specialist | +0.012 s overall, worse at most airports | rejected |
| 83-feature "match-context" model | 324.998 with specialist | rejected |
| Airport-specific blend weights | 323.712 vs 323.377 | rejected |
| Normalized-carrier specialist | 323.623, worsened January | rejected |
| Duration LightGBM @523 trees | 327.526; +10 % → 321.371 | not submitted |
| 119-feature NM interval / forward-traffic LightGBM | 328.146; +10 % → 321.376 | not submitted |
| 3 mixture-of-baselines classifiers for unmatched LIRF | worsened both months | rejected |
| Arrival **NM-record recovery** (≠ the v7 arrival features) | worsened both months | rejected |
| Recurring flight-service categories | local only, unscored | — |
| v7 arrival specialist, January alone | 3488.457 → **3752.507** (worse) | accepted anyway at 50 % weight |
| v9 airport-expert model **alone** | worsens January and February | only survives averaged 50/50 with the neighbour model |
| **v12 LightGBM679 following-context whole replacement** | whole gain **0.283203 s** < 0.5 gate | **rejected**; only the CatBoost version was promoted |

**[X]** Two patterns worth lifting: (a) capping the *label* is catastrophic while capping the *fitting
residual* at ±7200 is their standard and appears in every model from v8 on; (b) a feature family can be
worth nothing through one learner (LightGBM679, +0.28 s) and 1.4 s through another (CatBoost 4999) — they
only found the v12 gain because they retried the same 30 columns with the deeper model.

---

## 4. CV setup and how each gain was "validated"

**[S]**
- **Primary holdout**: calendar months **January + July 2025**, membership decided by the UTC month of
  `SCHED_TIME_UTC_mvt` (`contest.py:56`, `carrier_contest.py:75`, `ensemble_contest.py:60`,
  `duration_contest.py:99`). Fit = the **other ten months**, `y ≥ 0`. 344,419 evaluation rows,
  including **80 negative labels retained in scoring** and all positive outliers.
- **Secondary component holdout**: **February + August** (other ten months fit), used from v6 onward for
  component-level comparisons only — METHOD repeatedly states it "does not validate the complete ensemble".
- `contest.py` also supports a chronological `july` split (train Jan–Jun, validate Jul–Dec), used for the
  early ablation ladder only.
- `pipeline.py:155-160` has a flight-group purge across a chronological boundary
  (`FLIGHT_ID_mvt` may not appear on both sides) — **but none of the later contest modules use it**;
  the month-based Jan/Jul split has no group purge. **[X]** A flight that recurs daily therefore appears
  in both fit and holdout at other dates.
- **Negatives** are excluded from fitting (`y.ge(0)`) but kept in evaluation everywhere. Fitting residuals
  (not labels) clipped ±7200 from v8 on.
- **Promotion gate, predeclared before reading the result** (v8 onward, `METHOD.md` v8/v9/v10/v12/v13):
  1. ≥ **0.5 s** whole Jan/Jul RMSE gain;
  2. improvement in **both** months;
  3. **≥ 40 of 62** UTC days improved;
  4. (v9, v13) improvement in **all four** component months;
  5. (v13) also beat a 183-feature/679-tree control.
  Observed: v8 61/62 days, v9 55/62, v10 52/62, v12 59/62, v13 61/62.
- **One statistical test in the entire log**: a paired **day-cluster bootstrap, 2,000 resamples,
  seed 20260908** for the v4 blend — 95 % interval [−580.97, −301.10] s² of MSE change (`METHOD.md`, v4
  section). No other candidate got an uncertainty estimate.
- **They state plainly that none of it is an untouched test.** January/July "has repeatedly informed
  research decisions"; hyperparameters were selected on it; v11's motivating diagnostic even "contained
  previously inspected holdout information" (METHOD v11 section). **[S]**
- v13's margin is 0.5006107892629075 s against a 0.5 s threshold — **0.0006 s of slack**. **[S]** That is
  gate-fitting. **[X]**

**What is NOT determinable from the code [S]:** there is no k-fold, no repeated CV, no nested CV, and no
per-seed variance estimate for any global model. Local-to-official transfer is erratic and they never model
it: v8 gained 2.77 s locally but **7.11 s** officially; v10 gained 0.86 s locally but 2.58 s officially;
v12 gained 0.78 s locally but 1.42 s officially; v11 gained 0.57 s locally but 0.20 s officially.

---

## 5. Prediction-time availability — what depends on data you would not have

Determined from code. All of these are **supplied in the 2026 ranking file**, so they are legal; none are
available before the flight departs. The authors say so themselves (`METHOD.md:15`,
`arrival_features.py:2-7`, `nm_following_features.py:2-5`, `nm_clock_overlay.py:2-8`). **[S]**

1. **The entire baseline offset** is `MVT_TIME_UTC_mvt − AOBT_3_flt` (fallbacks LOBT, EOBT_1, then 1000 s)
   — `traffic_features.py:107-113`. `MVT_TIME` is take-off and `AOBT_3` is NM actual off-block, so this is
   the label up to the `AOBT_3 − BLOCK` discrepancy. **[X]** Everything else in the repo is a correction
   to that discrepancy.
2. **Features 56-65** (`movement_minus_AOBT_3/LOBT/ARVT_1/ARVT_3`, `nm_actual_vs_estimated_seconds`,
   `nm_last_vs_actual_seconds`) and **84-92** (NM flight durations built from ARVT_3/AOBT_3) are all
   post-flight NM records. `ARVT_3_flt` is the **arrival** time of the same flight — strictly future.
3. **Block E (arrival features)** cuts off at the departure's **take-off**, so arrivals that on-block during
   this flight's own taxi-out are inside the window (`arrival_features.py:44,57`). **[X]**
4. **Block G (30 following-context features)** is explicitly `(q, q + 15/60 min]` — the *future* hour of
   departures at the same airport/runway/stand (`nm_following_features.py:45-48`). This is the clearest
   case: −1.4181 s of official score comes from data after the query instant.
5. **The v11 clock overlay** outputs `MVT_TIME − LOBT_flt` directly as the prediction — a clock read, not
   a forecast (`nm_clock_overlay.py:116`).
6. **Block F (neighbour)** is strictly past relative to take-off, but the peer values are peers'
   `MVT − AOBT_3` — known only after those peers took off. For a flight at time `q`, peers that took off in
   `[q−60m, q)` have indeed departed, so this family is the **closest to operationally defensible** in the
   repo. **[X]**
7. Not leakage but worth noting: `scheduled_departures_30m` and `movement_departures_30m`
   (`pipeline.py:89,104`) **include the query row itself** in the count, unlike the trailing-traffic counts,
   which exclude exact-time ties (`traffic_features.py:12,39`).

**Guards that do exist [S]:** `arrival_features.py:20` filters `PHASE=='ARR'` **before** selecting
`TAXITIME_SEC_mvt`/`BLOCK_TIME_UTC_mvt`; `nm_neighbor_features.py:7` and `nm_following_features.py:15`
select a 6-column allowlist that contains no target/block field; tests poison DEP-row
`TAXITIME_SEC_mvt`/`BLOCK_TIME_UTC_mvt`/`MVT_ID_mvt` and assert the feature matrices are unchanged
(`tests/test_nm_following_features.py:15,51`, `tests/test_arrival_contest.py:66`,
`tests/test_ordinary_ensemble.py:37`, `tests/test_traffic_features.py:25,45`).

**Cannot be determined from the code:** the actual `AOBT_3 − BLOCK_TIME` distribution, per-airport null
rates, or how much of each gain is proxy repair vs genuine modelling — the repo ships no data and no
feature-importance numbers from a real fit (importances are written to private `runs/` reports only).

---

## Applicable to resilient-hedgehog

Ranked by the gain **their own log** attributes to each item (official Δ where one exists). We already
use the `MVT_TIME − AOBT_3` proxy (`docs/CODEX_BRIEF.md:54`), so items 1-6 sit directly on top of it.

1. **Deeper/longer CatBoost on ±7200-capped residuals, blended 50/50 — their v8, −7.1141 s official.**
   `depth=9, learning_rate=0.05, l2_leaf_reg=7, border_count=254, RMSE, 4999 iterations, no early stopping`,
   fitted on `clip(y − proxy, −7200, +7200)`, excluding the unmatched-LIRF scope, then
   `0.5·incumbent + 0.5·new` with both floored at 0 (`arrival_boost_contest.py:29-33,43-45,77`). Their largest
   single gain, on **zero new features**. Cheapest thing for us to try. **Cap the residual, never the label**
   (label capping scored 434.145 vs 382.284, `METHOD.md:28`).
2. **The 26 arrival / taxi-in / same-stand features — their v7, −2.6224 s official.**
   Reimplement `arrival_features.py:15-77` verbatim: ARR-only pool keyed on `ADES_mvt`, completion time =
   arrival `BLOCK_TIME` (on-block), value = arrival `TAXITIME_SEC_mvt` filtered to [0,7200] with
   `BLOCK ≥ MVT`; airport and airport+runway rolling mean/std/count over 15/30/60 min strictly before the
   departure's take-off; last completion with age ≤ 7200 s (ties averaged); plus the **same-stand block**
   (age ≤ 86400 s, ties → NaN, `aircraft_equal`/`carrier_equal` turnaround flags). 26 columns; NaN is the
   correct missing value — do not impute.
3. **Swap the incumbent component for a wider-feature refit at its algebraic weight — their v10, −2.5779 s
   official.** `new_prediction = incumbent + w·(Model_wide − Model_narrow)` where `w` is the weight the
   narrow model already carries inside the incumbent (0.375 for them), both models floored at 0, the
   difference never clipped, protected scopes skipped **before** the arithmetic
   (`neighbor_boost_contest.py:29,57-72`). This lets you upgrade one component without refitting or
   reweighting the ensemble, and keeps every previously verified prediction bit-exact.
4. **The 30 following-context NM features — their v12, −1.4181 s official.** `nm_following_features.py:14-59`.
   Pool = DEP rows with `ADEP_mvt==ADEP_flt` and proxy `MVT−AOBT_3 ∈ [0,7200]`; groups
   airport / airport+runway / airport+stand **within one `%Y-%m`**; windows `(q, q+15m]` and `(q, q+60m]`
   with count/mean/popstd/own−mean; plus `next_wait_seconds` and `next_proxy` (≤7200 s, ties averaged).
   **Caveat:** worth only +0.28 s through a 679-tree LightGBM and 1.4 s through a 4999-tree CatBoost — if we
   test it, test it with the deep model or we will wrongly reject it.
5. **The 30 strictly-past NM-neighbour peer proxies — their v9, −0.8819 s official** (plus they are the
   substrate the v10 swap exploited). `nm_neighbor_features.py:6-48`. Same construction, windows
   `[q−15m, q)` / `[q−60m, q)`, `last_age_seconds`/`last_proxy`. `own_minus_mean` (this flight's own proxy
   minus the local peer mean) is the feature that directly encodes "is this airport's AOBT_3 bias unusual
   right now". Of all their additions this is the least hindsight-dependent.
6. **Per-airport experts averaged 50/50 with a pooled model — inside their v9.** Ten LightGBM airport experts
   on the *narrow* (123) feature set, equal-weighted with the pooled wide (153) model, that average then
   entering at 25 % (`ordinary_ensemble.py:45-55,92-103`). Their airport model **alone** worsened January and
   February; only the 50/50 average helped. Do not ship per-airport experts unaveraged.
7. **The AOBT_3-vs-LOBT repair rule — their v11, −0.2013 s official on 21 rows.**
   If `AOBT_3 − LOBT > 7200 s` (signed, strict) and both `MVT−AOBT_3` and `MVT−LOBT` lie in
   `[−7200, 172800]` and `ADEP_flt == ADEP_mvt`, predict `max(0, MVT − LOBT)` outright
   (`nm_clock_overlay.py:104-116`). Tiny, high-variance (one row supplied 45.8 % of the gain), but it is a
   ~10-line pure function with an exact-preservation assertion, and our proxy has the same failure mode.
8. **Seed-bagging the small specialist — their v6, −1.6945 s official on 383 rows.** Five equal-weight
   CatBoost seeds (depth 5, lr .04, l2 10, 1014 trees) on the 1,488-row LIRF∧missing-IOBT scope, each floored
   at 0 before averaging, seeds declared up front and none discarded (`specialist_ensemble.py:37-50,80-82`).
   The generalizable rule: **any sub-model fitted on ~10³ rows should be seed-averaged**; ours at LIRF is the
   same size.
9. **Process, not score: freeze tree counts and a promotion gate before looking.** ≥0.5 s whole gain,
   improvement in *both* holdout months, ≥40/62 days improved, and (for feature families) improvement in all
   four component months. It is what kept them from shipping the 0.283 s LightGBM following-context
   candidate. Counter-example to copy carefully: v13 passes by 0.0006 s — a gate you can retry against is a
   gate you will eventually fit.
10. **Do not copy their validation design.** One reused January+July split, no k-fold, no group purge in the
    contest modules, no per-seed variance, one bootstrap in the whole log, and local→official transfer that
    ranged from 0.35× to 2.6×. Their score is evidence the *features* work, not that their selection
    procedure is sound.
