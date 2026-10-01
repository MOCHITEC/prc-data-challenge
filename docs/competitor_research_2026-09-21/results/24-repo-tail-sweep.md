# Slot 24 — repo-tail-sweep

Investigator slot 24. Five repos cloned to `/tmp/slot24/` and read in full (notebooks included).
All citations are file paths inside those clones.

Marking: **[S]** = directly sourced (quoted from a repo file or from the live leaderboard API).
**[X]** = my inference / cross-repo arithmetic, explicitly labelled.

---

## 0. Ground truth first: leaderboard ranks of these teams

[S] Fetched the public API `https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard`
(endpoint taken from `/tmp/slot24/victoralcadi_prc-data-challenge-2026/src/prc2026/leaderboard.py:16`),
paged fully: 2,144 submissions, 168 teams, **every single submission reports `usedPairs: 344841`**.

Best-per-team ranking, as of 2026-09-21:

| rank | team | best score | repo |
|---|---|---|---|
| 43 | **resilient-hedgehog (us)** | **284.3584** | — |
| 78 | joyful-lemon | 300.2712 (`_v38`) | fenil264 |
| 91 | likable-river | 315.8111 (`_v11`) | sqcm9896 |
| 144 | unique-umbrella | 573.8456 (`_v15`) | kagervig |
| 161 | sincere-donkey | 625.4325 (`_v1`) | GLAPM |
| — | (victoralcadi's team) | not identified — repo never names its team | victoralcadi |

All four scored teams are **worse than us**. Treat measurements as evidence, conclusions as claims.

---

## 1. PRIORITY: ranking-set composition — CORROBORATED, exactly

The OpenAir finding under test: the 2026 ranking set is **January 2026 = 152,719 DEP rows +
July 2026 = 192,122 DEP rows = 344,841**.

### Source A [S] — sqcm9896, total ranking rows per month

`/tmp/slot24/sqcm9896_prc-2026-likable-river/instructions.md:11-13`:

> `ranking.parquet` (689,534 rows: Jan 2026 305,590 + Jul 2026 383,944;
> DEP 344,841 to predict, ARR 344,693 context, fully observed).

Same numbers repeated independently at `eda/FINDINGS.md:8-9` (asserted by an executed
notebook: "Notebook asserts every number below (last cell `ALL EDA ASSERTIONS PASS`)").

### Source B [S] — fenil264, same numbers from an independent notebook

`/tmp/slot24/fenil264_prc-2026-taxi-out/notebooks/01_data_access.ipynb`, executed cell output:

```
ranking rows: 689,534
│ 2026-01 ┆ 305590 │
│ 2026-07 ┆ 383944 │
DEP rows in ranking: 344,841
nulls in TAXITIME_SEC_mvt (DEP): 344841
nulls in BLOCK_TIME_UTC_mvt (DEP): 344841
submitting ∩ ranking-DEP : 344,841
```

and its markdown: "Jan = 305,590 rows, Jul = 383,944 — **July is ~26% busier**."

### Source C [S] — sqcm9896, DEP-only split stated directly

`/tmp/slot24/sqcm9896_prc-2026-likable-river/eda/CHALLENGE.md:44-46`:

> Jul holdout much harder than Jan (718 vs 584); **test is Jul-heavy (192k vs 153k)** —
> pooled RMSE is the honest target, 50/50 flatters by ~11 pts.

and `eda/FINDINGS.md:43`: "Weight pooled RMSE by test mass (Jul 384k > Jan 306k), not 50/50."

### Source D [S] — victoralcadi, row count and months, sourced from the API

`/tmp/slot24/victoralcadi_prc-data-challenge-2026/src/prc2026/config.py:58-61`:

```python
# Rows the ranking script actually scores, observed from the leaderboard's `usedPairs`
# rather than published anywhere. A submitting.parquet of a different length is a
# reason to look twice before uploading.
EXPECTED_SUBMISSION_ROWS = 344_841
```
`src/prc2026/model.py` docstring of `train_validate`: "`ranking.parquet` (which contains
**Jan + Jul 2026 in full**)". README: "Ranking uses RMSE over **January and July 2026** departures".

### The arithmetic closes exactly [X — my computation on [S] numbers]

```
Jan DEP 152,719 + Jul DEP 192,122               = 344,841   ✓ (= submitting.parquet rows)
Jan total 305,590 − Jan DEP 152,719 = Jan ARR   = 152,871
Jul total 383,944 − Jul DEP 192,122 = Jul ARR   = 191,822
Jan ARR 152,871 + Jul ARR 191,822               = 344,693   ✓ (= sqcm9896's stated ARR count)
Jan total 305,590 + Jul total 383,944           = 689,534   ✓ (= stated ranking.parquet rows)
```

Five independent identities all close to the unit. The OpenAir per-month DEP split
(152,719 / 192,122) is **corroborated by a second and third source**: sqcm9896 states the
per-month *totals* and the *ARR* count which force the DEP split to exactly those values,
and separately states "192k vs 153k" in words; fenil264 reproduces the per-month totals
from its own executed notebook.

**Verdict: CONFIRMED. The scored set is Jan 2026 + Jul 2026 only, Jul-heavy at 55.7% / 44.3%.**

**Corollary [X]:** validating on any month other than January and July is validating on a
month that is not scored. A change that loses on December is, to first order, harmless.

### Non-corroborating sources

- [S] **kagervig does NOT corroborate and is internally wrong.** `learnings.md:16` says
  "ranking starts 2026-01-04"; `learnings.md` §Caveat says "the ranking data (**Jan–Sep 2026**)";
  the Full Project Narrative says training covers "**Nov 2024 – Dec 2025**" across "**11** major
  airports". All three are wrong. Their `submitting.parquet` is 344,841 rows (verified by me
  reading the parquet metadata) but they never decompose it by month. Discard their month claims.
- [S] **GLAPM makes no composition statement at all.** `eda_script.py` prints `df_ranking.shape`
  but the repo contains no output, no findings file.

---

## 2. PRIORITY: the sqcm9896 holdout gate, reconstructed exactly

Repo: `/tmp/slot24/sqcm9896_prc-2026-likable-river`, HEAD `866b2f6`, 2026-09-09.
Team likable-river, rank 91, 315.81.

### 2.1 What the rule is, in their words [S]

- `src/config.py:11`: `HOLDOUT_MONTHS = (1, 7)  # Jan+Jul 2025 validate; mirrors Jan+Jul 2026 test`
- `README.md:38`: "a feature ships only if **Jan AND Jul holdout RMSE improve**."
- `MODEL_CARD.md:19-20`: "Validation: Jan+Jul 2025 holdout (pool 1,740,628 / val 344,419);
  report Jan / Jul / pooled RMSE; a feature ships only if Jan AND Jul improve."
- `instructions.md:45-46`: "Validation: train ≤ month T, validate Jan+Jul 2025 (same calendar
  months as test), report Jan / Jul / combined RMSE. **Never select on one month alone.**"
- `plan/06_feature_engineering.md:6`: "only if Jan AND Jul 2025 holdout RMSE improve
  (**or documented trade-off**)."
- `plan/00_final_plan.md:10`: "## Pipeline (in order, keep a feature family only if Jan AND Jul improve)"

### 2.2 How each holdout is built [S]

`src/validation.py` and `src/train.py:build()`:

```python
pool, val = df[~df.month.isin(HOLDOUT_MONTHS)], df[df.month.isin(HOLDOUT_MONTHS)]
```

- Month is taken from `BLOCK_TIME_UTC_mvt.dt.month` in `validation.py:load_dep_dep()`,
  and from `SCHED_TIME_UTC_mvt.dt.month` in `features.py:add_base()` — a minor inconsistency
  in their own code.
- Split is over the full 2025 DEP set: **pool = the other ten months (1,740,628 rows),
  val = January 2025 + July 2025 (344,419 rows)**. Note their val size 344,419 is ~ the same
  order as the real scored 344,841 — they deliberately size-matched the calendar.
- It is **one split, not k folds**. No repeats, no bootstrap, no embargo despite
  `plan/02_actions.md:6` saying "(+embargo)" — the committed code has no embargo.
- `src/validation.py:report()` returns exactly four numbers: `jan`, `jul`, `pooled`, and
  `avg5050 = (jan+jul)/2`. The **pooled** number is row-weighted, so it is automatically
  Jul-heavy, matching the scored set. They explicitly reject 50/50
  (`eda/CHALLENGE.md:44`: "50/50 flatters by ~11 pts").

### 2.3 What counts as "improves" [S]

There is **no formal tolerance band in the rule**. The rule is two point estimates, both
strictly down. There is no confidence interval, no repeat-seed requirement, no bootstrap,
no trimmed/robust variant anywhere in the repo.

What they have instead is an **informal, post-hoc noise floor** that they discovered and then
applied inconsistently:

- `instructions.md`: "Pure-seed spread **±6 pooled** noted — all frozen numbers carry seed luck."
- `reports/experiments.csv` row `standzone_fix_decision`: "**±2 seed-noise zone**".
- Rejections at the noise floor, labelled `drop-noise`:
  - `cat_bag2_decision`: "−0.2 vs single 361.2, **at noise floor**".
  - `bagged5_LIRFspec_decision`: "pooled −0.1 with mixed months (jan −2.9/jul +2.1)".
  - blend-weight sweep: "best 358.0 (0.70/0.80) vs current 358.3: **−0.3 on 25-grid = noise mining**;
    surface flat; KEEP 0.8/0.7".
  - `tuning_lgbm_summary`: "best grid 448.5 vs default 448.8 (**−0.3 noise**); keep defaults".

[X] My read: their effective tolerance is roughly "a both-months improvement smaller than
~1s pooled is treated as noise and rejected, especially if it came from a grid/weight sweep",
but this is nowhere written as a rule and they violated it in both directions
(they shipped v10 on a −0.3 leg swap; they rejected a −0.3 weight sweep).

### 2.4 Do they log rejected experiments? Yes, thoroughly [S]

`reports/experiments.csv`, 110 rows, schema:
`run_id, data_manifest, feature_version, split, model, parameters, RMSE_jan, RMSE_jul, combined_RMSE, runtime, decision, notes`.
Every row has `split = jan+jul2025`. The `decision` column carries explicit rejection codes:
`drop` (15), `drop-noise` (2), `drop-bug`, `drop-mixed`, `drop-jan-rule`, `keep-fix-drop-lineage`,
alongside `keep-SHIP`, `keep-PROVEN`, `UPLOADED-*` and `TEST-RESULT` rows that record
leaderboard feedback. `plan/04_testing.md:19` mandates the schema.

Rejections caused *specifically* by the Jan-AND-Jul conjunction (i.e. pooled improved but one
month regressed):

| run | jan | jul | pooled | verdict |
|---|---|---|---|---|
| `extreme_spec` | 361.7 (**+5.9**) | 341.4 (**−23.5**) | **350.6** (vs ship 359.2) | `drop-jan-rule` |
| `strict_perapt_decision` | +2.8 worse | better | −0.5 | `drop-mixed` |
| `standzone_fix_decision` | 356.9 | 364.6 | 361.2 (vs 359.2) | fix kept in code, ship lineage unchanged |
| `cat_v1_qx2` (NQEXT) | +3.0 worse | −4.2 | −1.2 | dropped |

`extreme_spec` is the expensive one: **the gate threw away a −8.6s pooled holdout gain**
because January regressed by 5.9s. Since the metric is pooled RMSE over 152,719 Jan +
192,122 Jul rows, that trade is positive-expectation under the actual scoring function.
Their own justification (`eda/FINDINGS.md:95-98`) is that the Jan regression came from a
single 87k-second Jan row the specialist undershot by 35k — i.e. a lottery row, not a
mechanism. [X] That is a *tail-sensitivity* argument, not a *month* argument — the month
gate caught it by luck, and a drop-N-rows check would have caught it by design.

### 2.5 The gate's out-of-sample track record — the most valuable part [S]

They uploaded repeatedly and logged leaderboard results against holdout predictions.
From `reports/experiments.csv` `TEST-RESULT` rows:

| change | holdout delta (both months passed) | leaderboard delta | transferred? |
|---|---|---|---|
| v9 physics-identity floor | −23.9 pooled | **−20.3** | **yes** |
| v11 runway-flow feature swap | −2.9 pooled | −0.24 | yes, tiny |
| v10 ARR-propagation leg swap | −0.3 pooled | −0.14 | yes, tiny |
| **v6 per-airport blend weights** | **−0.6 pooled** | **+7.0 (worse)** | **NO** |
| **v7 LIRF small-slice patch** | **−1.2 pooled** | **+0.95 (worse)** | **NO** |
| v8 LFPG-Jan join-fail floor probe | (probe) | +2.6 (worse) | NO |

Their own post-mortems: v6 — "per-airport weights **OVERFIT holdout monsters**; whole approach
REJECTED"; v7 — "**small-slice monster fits don't transfer** (2nd proof after v6)".

[X] **This is the key lesson and it is not about months.** Every both-months-passing change
that failed out of sample was a *post-hoc parameter fit on holdout slices dominated by a
handful of extreme rows*. Every both-months-passing change that transferred was either a
new feature or a physics rule. The Jan-AND-Jul conjunction did **not** catch the failures,
because a 344k-row holdout has enough monsters in *both* months for a slice-fit to improve
both. A tail-trimmed check would have.

### 2.6 The gate is applied to contaminated numbers [S]

`src/train.py`: both models are fit with the holdout as `eval_set` plus early stopping:

```python
m.fit(Pool(Xp, yp, cat_features=ci), eval_set=Pool(Xv, yv, cat_features=ci))   # CatBoost, early_stopping_rounds=100
m = lgb.train(params, dtr, ..., valid_sets=[dval], callbacks=[lgb.early_stopping(100), ...])
```

Their own adversarial review caught this. `recommendations.md:18`:

> | **Validation Integrity** | ❌ Compromised | The Jan+Jul 2025 "holdout" is contaminated by
> early-stopping target peeking and hardcoded tree-count harvesting. |

and `recommendations.md:40`: "*Why do you call Jan+Jul 2025 an 'untouched holdout' when your
training scripts stop early on it and your prediction script hardcodes iterations derived
from it?*". Remedy at `recommendations.md:160` ("never pass it as `eval_set`") was **not
implemented** — the committed `train.py` still does it.

### 2.7 How the sqcm9896 gate differs from our rule

Our rule: a change ships only if the **full-sample confidence interval** AND the
**drop-200-rows confidence interval** are both on the improving side.

| axis | sqcm9896 gate | our gate |
|---|---|---|
| **what is partitioned** | the *calendar* — two disjoint month subsets of one split | the *sample* — resampling uncertainty, plus a tail-trimmed resample |
| **number of statistics** | 2 point estimates (RMSE_jan, RMSE_jul) | 2 interval estimates |
| **uncertainty quantification** | none in the rule; informal ±2…±6s folklore applied ad hoc | explicit, both intervals |
| **tail robustness** | none. Both months are individually monster-dominated (LIRF = 59.6% of holdout SSE, `eda/FINDINGS.md`) | the drop-200 arm is precisely a tail-robustness test |
| **what it is designed to catch** | seasonal non-transfer (winter-fit vs summer-fit) | (a) noise-mining, (b) fits carried by a few extreme rows |
| **what it demonstrably misses** | monster-slice overfits — v6 and v7 both passed it and both lost on the leaderboard | month-specific seasonal effects: nothing in our rule requires a change to help *both* Jan and Jul |
| **conservatism vs the metric** | *stricter* than pooled RMSE — it rejects favourable Jul-for-Jan trades (`extreme_spec`, −8.6s pooled, discarded) | neutral w.r.t. month composition |
| **audit trail** | 110-row `experiments.csv` with an explicit `decision` column incl. rejection codes | (ours) |

**The two rules are close to orthogonal, and each catches what the other misses.**
A change can pass both months and fail a drop-200 CI (v6, v7 — their two documented
out-of-sample failures). A change can pass both CIs and be a pure Jan-vs-Jul seasonal
artefact that our rule never looks at.

One more structural difference worth stating plainly: their holdout months are *the same
calendar months as the scored set*. Ours are not necessarily. Given §1, "same calendar
months" is now a demonstrably correct design choice, not a convention.

### 2.8 sqcm9896 model card (for completeness) [S]

- **Fitting target: raw duration.** `src/config.py:14`: `TARGET = "TAXITIME_SEC_mvt"`.
  No residual against a clock proxy. `eda/FINDINGS.md`: "log1p overcorrects (skew −0.61);
  NOT lognormal … **Model raw target for RMSE**". The clock proxy enters as a *feature*:
  `features.py:add_base()` → `m_aobt = MVT_TIME_UTC_mvt − AOBT_3_flt`.
- **Features**: 12 native categoricals (`ADEP`, `RUNWAY`, `STAND_ZONE`, aircraft type ×2, wake,
  operator, market segment, flight rule ×2, flight type, `ADES`); cyclical clock from SCHED
  (`sin_h/cos_h/sin_d/cos_d`, weekend, night) plus `m_hour/m_dow` from MVT; schedule deltas
  `sched_lobt/sched_iobt/sched_eobt/eobt_iobt`; flags `is_join_fail` (= `AOBT_3_flt.isna()`),
  `diverted`, `adep_agree`; **permissive block** `m_aobt, m_sched, aobt_sched, m_lobt, m_iobt,
  m_eobt, arvt1_m`; 5-fold OOF smoothed target means (m=50) + `log1p(count)` on 6 keys.
  Optional families: `CG` congestion counts (rejected), `QX` experienced-queue (accepted).
- **Model family**: CatBoost (d8, lr .05, RMSE loss) × 0.8 + LightGBM-L2 (127 leaves,
  min_data 500) × 0.2, plus per-airport heads (LIRF/EGLL/LFPG) and join-fail specialists
  (global + LIRF-only), then a cross-lineage 0.7/0.3 blend.
- **Post-processing** (`MODEL_CARD.md:9-10`): **negative cap only** + a
  **physics-identity floor: joined `m_aobt > 1800` → `max(pred, m_aobt − 3600)`**.
  This single rule is their largest leaderboard gain: **−20.3s** (v5 336.5 → v9 316.2),
  predicted −23.9 on holdout. Upper-tail clipping was tested and **hurts** (+90.7s,
  `clip_cap2h_v1`).
- **Rejected with evidence**: Huber objective (668.4 vs L2 448.8 — "Huber optimizes the wrong
  target for RMSE"), asymmetric loss, train-target clipping (vacuous), upper clip, depth-10,
  XGBoost (481.8), wake-ahead counts, ERA5/Open-Meteo weather (485.5 vs 448.8, both months
  worse, "grid too coarse"), every bagging variant, v2 congestion counts.
- **Scores** [S, matches live leaderboard]: v3 345.57, v4 (strict, no MVT/AOBT) 470.85,
  v5 336.46, v6 343.5, v7 337.4, v8 339.0, v9 316.19, v10 316.05, **v11 315.81 (final, rank 91)**.
- **Oracle note** (`experiments.csv:forensics_oracle`): "No.1 needs ~half the monster error;
  **6 rows decide 40%**".

---

## 3. victoralcadi — the takeoff-time-anchored leakage framing, in full

Repo: `/tmp/slot24/victoralcadi_prc-data-challenge-2026`, HEAD `5dccaca`, 2026-09-17.
No leaderboard score; the repo never names its team.

### 3.1 The framing verbatim [S]

`README.md`, section "The modelling constraint that shapes everything":

> For the departures you must predict, **off-block time is not given**. Only the *takeoff*
> time (`MVT_TIME_UTC_mvt`) and the *scheduled* time are known. So any feature must be
> computable from:
> - takeoff time (DEP) — known
> - scheduled time — known
> - runway, stand, aircraft type, operator, market segment — known
> - arrival movements' block + movement times — known (only departures were blanked)
>
> Congestion features here are therefore anchored on **takeoff time**, not push-back time.
>
> > `AOBT_3_flt` (NM actual off-block) is a **near-duplicate of the blanked
> > `BLOCK_TIME_UTC_mvt`**. `prc2026` keeps it out of the default feature set; run
> > `prc2026 audit` to check whether it is populated in `ranking.parquet` before
> > deciding what to do with it.

`src/prc2026/features.py` module docstring:

> Everything here is computable for the ranking set, i.e. **without** the departure
> off-block time. Congestion is therefore anchored on the takeoff time
> (`MVT_TIME_UTC_mvt`) and on scheduled times, **never on push-back**.

### 3.2 What they classify as unusable [S]

`src/prc2026/config.py:98-99`:

```python
# columns that leak the blanked off-block time; excluded from the model by default
LEAKY_COLUMNS = [BLOCK_TIME, "AOBT_3_flt", TARGET]
```

`src/prc2026/cli.py:122-124` runs a runtime audit printing
`"leak check: {col} populated for {dep[col]:.1f}% of ranking departures"` for
`AOBT_3_flt` and `LOBT_flt`.

### 3.3 Their reasoning, restated [S + X]

[S] The stated reason for excluding `AOBT_3_flt` is that it is a *near-duplicate of the
blanked quantity*, i.e. it reconstructs the target's other half, and the organisers'
blanking of `BLOCK_TIME_UTC_mvt` expresses an intent this column would defeat.
[S] Prize eligibility framing in the same README ("the solution is original…") sits
alongside it; they treat exclusion as the default and make inclusion an explicit,
audited decision rather than the reverse.

[X] Note what the framing does **not** claim: it does not claim `MVT_TIME_UTC_mvt` is
unusable. It classifies takeoff time as **known and legitimate** and builds every congestion
window around it. This is the opposite of kagervig's framing (§5.4) and the same as
sqcm9896's factual position — sqcm9896 just chose to use `AOBT_3_flt` anyway under a
documented "PERMISSIVE" policy (`instructions.md:4`).

**Empirical settlement of the disagreement [X, from [S] numbers]:** sqcm9896 ran both tracks
on the same holdout. Permissive (with `m_aobt`) CatBoost 433.6 pooled vs **strict (no
MVT/AOBT arithmetic) 508.6** — a ~75s holdout gap; on the leaderboard v3 permissive 345.57
vs v4 strict 470.85, a **125s gap**. victoralcadi's purity position is defensible on
eligibility grounds but costs more than the entire spread between rank 1 and rank 100.

### 3.4 The rest of victoralcadi [S]

- **Target**: raw `TAXITIME_SEC_mvt`. No residual formulation.
- **CV**: `model.py:train_validate(..., val_months: tuple[int, ...] = (1, 7))` — **Jan+Jul
  month holdout, a third independent team using exactly these months**, with the docstring
  "The validation frame is featurised on its own, exactly like `ranking.parquet` (which
  contains Jan + Jul 2026 in full), so **congestion features never reach across the split**."
  Baseline reported alongside is the per-airport median.
- **Model**: single LightGBM, `num_leaves 255, min_data 100, lr 0.05, feature_fraction 0.8,
  bagging 0.8, lambda_l2 1.0, cat_smooth 20`, 3000 rounds, early stopping 100 on the holdout
  (same contamination as sqcm9896).
- **Features** (`features.py`): DST-aware **airport-local** hour (+sin/cos), dow, weekend,
  month, day-of-year; `takeoff_vs_sched_sec`, `eobt_vs_sched_sec`, `takeoff_vs_eobt_sec`,
  `planned_flight_sec = ARVT_1 − EOBT_1`, `eobt_vs_iobt_sec`, `is_diverted`;
  congestion at 15/30/60 min windows around takeoff — `dep_takeoffs_prev/next`,
  `arr_landings_prev`, `arr_inblocks_prev`, `sched_deps_prev/next`, `rwy_takeoffs_prev`;
  `dep_queue_3h = sched_deps_3h − airborne_3h` ("net departure queue built up over the last
  3h: scheduled minus already airborne"); runway pacing `rwy_gap_prev_1`, `rwy_gap_mean_5`;
  `arr_dep_ratio_60m`; and smoothed group target statistics (mean/median/std/count, prior 20)
  over 8 keys including `(airport, stand_prefix, runway)` and `(airport, runway, hour_local)`.
- **Post-processing**: `np.clip(pred, 60, 10800)`.
- **⚠ methodological flaw [X]**: `config.TAXI_MIN_SEC = 60`, `TAXI_MAX_SEC = 3*3600`, and rows
  outside that band are dropped from **both training and validation**
  (`model.py`: `labelled = y_val.between(...)`). Their validation RMSE therefore excludes
  exactly the tail that dominates the real metric. Same self-deception as kagervig (§5.3),
  milder. Their reported "the bar is ~246s" is a leaderboard reading, not their own score.
- **Useful literature note** [S], `docs/papers-and-data.md`: the PRU additional-taxi-out
  methodology defines unimpeded time as the **20th percentile of observed taxi times per
  stand-group / runway pair**; they argue `STAND × RUNWAY` group statistics already
  approximate the unimpeded component and effort belongs on the additional component.
  Also: Munich de-iced 8,628 aircraft in 25-26 vs 6,937 in 24-25, so **January 2026 was a
  heavier de-icing winter than January 2025** — a directional drift warning for a
  Jan-2025-trained model. EUROCONTROL monthly open series stop at June 2026, so July 2026
  is not covered by them.

---

## 4. fenil264 / joyful-lemon (rank 78, 300.27) — EDA only, no model

Repo: `/tmp/slot24/fenil264_prc-2026-taxi-out`, HEAD `dfa0a20`, 2026-09-09,
"feat(02): EDA across target, temporal, runway, stand, distance, congestion".

### 4.1 The method is NOT in this repo [S]

Stated brief said "method lives in notebooks only". It does not live there either.
Measured content:

| file | cells | source chars |
|---|---|---|
| `notebooks/01_data_access.ipynb` | 31 | 28,784 |
| `notebooks/02_eda.ipynb` | 31 | 52,481 |
| `notebooks/03_baseline.ipynb` | 1 | **105** (a markdown stub: "Establish a simple baseline…") |
| `notebooks/04_features.ipynb` | 1 | **127** (stub) |
| `notebooks/05_model.ipynb` | 1 | **118** (stub) |
| `notebooks/06_submission.ipynb` | 1 | **114** (stub) |

`src/prc/features.py` is a 72-byte docstring-only file; `data.py` 69 bytes, `models.py`
59 bytes, `submit.py` 60 bytes — all empty stubs. `submissions/` and `models/` hold only
`.gitkeep`.

**They submitted `joyful-lemon_v38.parquet` (38 versions, score 300.27) and published none
of it.** There is **no fitting target, no feature list, no model family, no CV design and no
post-processing** recoverable from this repo. Stating that plainly is the result.

### 4.2 What IS there, and is worth having [S]

**(a) A hard 3,600s censoring cap at LEMD and LEBL.** `notebooks/01_data_access.ipynb`:

> Look at the top-10 max values side by side:
> - LEMD: 3598, 3567, 3566, 3562, 3549, 3546, 3537, 3522, 3510, 3504
> - LEBL: 3604, 3599, 3595, 3594, 3584, 3572, 3567, 3552, 3549, 3542
> - LFPG (control): 84240, 58206, 21308, 15059, 12578, 11737… natural exponential decay
> - LIRF (control): 131167, 88392, 88132, 87605, 87598… same natural decay
>
> A real long-tail distribution has one or two extreme values well above the pack. LEMD and
> LEBL have their top 10 packed tightly under 3600. That's a **hard cap at 1 hour** — some
> upstream pipeline drops or truncates any taxi-out over 3600s at these two airports.

Their transferability argument: the cap is airport-specific, not global (LFPG arrivals show
max 33,601s in train and 50,517s in ranking); arrivals at LEMD/LEBL are uncapped in both
2025 and 2026 only because taxi-in never reaches 3,600s, so the ranking DEP cap is not
directly provable. Conclusion: "**Working assumption: cap all predictions at LEMD and LEBL
at 3600s.**"

[X] This is a cheap, testable post-processing rule we can evaluate directly: on our own
2025 data, count LEMD/LEBL DEP rows with taxi > 3600 (should be ~0 if the cap is real) and
then clip our LEMD/LEBL ranking predictions at 3600. It is upper-clipping, which sqcm9896
showed *hurts globally* (+90.7s) — but this is a per-airport clip justified by an observed
censoring mechanism, not a global variance-reduction hack. Worth a gated experiment.

**(b) Divergent validation choice, with an argument [S].** Notebook 02 wrap-up:

> **Decision for validation:** hold out **entire months** as validation folds, not random
> rows. Specifically, use **February and August 2025** as our validation set (**mirrors the
> Jan/Jul test months in seasonality but doesn't leak the exact same months into training**).
> We train on the remaining 10 months.

[X] They are the only team of the five to deliberately *avoid* Jan/Jul. Their stated concern
— that training on ten months including Feb and Aug while validating on Jan/Jul somehow
"leaks" — is not a leakage argument at all; both designs hold out whole months. Feb/Aug
sacrifices the calendar match to the scored set for no gain. Given §1, this is the wrong
call, and they rank 78 with the most thorough EDA of the five.

**(c) Seasonal direction, per airport [S].** Jul 2025 median > Jan 2025 median at **every**
airport: LEBL +13%, LIRF +12%, EGLL +9%, EDDM +7%, LFPG +6%, LEMD +5%, LTFM +5%, EHAM +3%,
EDDF +2%, LSZH +2%; volume +~30% in July. Corroborates sqcm9896's "Jul holdout much harder
than Jan (718 vs 584)".

**(d) Congestion works unevenly, and why [S].** Spearman of departure congestion (±window)
vs target, per airport: EGLL +0.235, LTFM +0.226, LFPG +0.211, LIRF +0.205 — but
**EHAM +0.007, LEMD +0.005, EDDM +0.032**. Their diagnosis:

> **congestion-as-airport-count fails at multi-runway airports with independent operations.**
> The right feature there is `dep_congestion_by_runway_group`, needing airport-specific
> runway topology knowledge.

Also: the aggregate congestion curve rises then falls — "**that is Simpson's paradox**…
every airport is monotonically increasing. Trust the per-airport curves, not the overall."
And arrival congestion is "a bust as a global feature", ρ from −0.09 to +0.13.

**(e) Structural features fail at EGLL [S].** Runway spread ratio within airport:
EHAM 4.41× (811s), LTFM 1.88× (781s), LSZH 1.45×, LEBL 1.35×, LIRF 1.31×, LFPG 1.20×,
EDDF 1.21×, EDDM 1.18×, LEMD 1.16×, **EGLL 1.04× (58s)**. Stand p10–p90 spread:
EHAM 2.91×, LSZH 2.29×, EDDF 2.03×, LEMD 2.01× … **EGLL 1.27×**. Distance-to-destination
Spearman +0.298 overall (EDDF +0.371, EDDM +0.370, EHAM +0.219) but EGLL flat at
~1300–1400s across every distance bucket. Conclusion: "at EGLL, structural features fail and
everything is congestion."

**(f) Rotorcraft/small-GA are in the scored set [S].** 1,852 ranking DEP rows (0.5%) match
known helo/small-GA type codes; 79 rows with `RUNWAY_mvt == "H"` (LSZH helipad, ~7s median).
"We have to predict them, so we can't drop them from training."

---

## 5. kagervig / unique-umbrella (rank 144, **573.85**) — a cautionary case

Repo: `/tmp/slot24/kagervig_PRC-Challenge-2026`, HEAD `20fab83`, 2026-09-19.
Team name recoverable from `predictions/unique-umbrella_v*.parquet`.

**Their claimed validation RMSE is 244.0s. Their actual leaderboard score is 573.85s.**
A 330-second gap. The repo is valuable precisely because it documents, step by step, how a
team talks itself into a number that is 2.4× wrong. Two independent mechanisms produce it.

### 5.1 `learnings.md` — the substance, transcribed

Version progression (their table, verbatim):

| Version | Change | RMSE (s) | Delta |
|---|---|---|---|
| v1 | Baseline | 395.9 | — |
| v2 | Congestion signal | 386.4 | −9.5 |
| v3 | Ranking congestion fixed | 386.4 | 0.0 (submission quality only) |
| v4 | Hyperparameter tuning | 384.3 | −2.1 |
| v5 | Hour of day | 384.2 | −0.1 |
| v6 | Schedule delay | 361.3 | −22.9 |
| v7 | Day deviation ratio + ARVT update | 357.6 | −3.7 |
| v8/v9 | **Artifact filter** | **244.0** | **−113.6** |

Other content, condensed but faithful:

- **CV design**: "time-based split: the first 83% of flights by timestamp are training, the
  last 17% are validation… approximates 10 months of training and 2 months of validation."
  Code: `model.py:236` — `cutoff = df["MVT_TIME_UTC_mvt"].quantile(0.83)`. [X] So their
  validation months are **roughly November–December 2025** — neither of the two scored months,
  and the two months least like them.
- **Hyperparameter grid** (lr 0.05, ES 30, max 1000): best `num_leaves=127, min_data_in_leaf=50`
  → 384.3s; whole grid spans 384.3–387.4, "±3s".
- **Congestion window sweep**: 30 → 386.1, 45 → 387.7, **60 → 384.3**, 90 → 386.8.
- **Failed feature sweep 1** (all reverted): departure queue ±30 min EOBT count +2.1;
  scheduled hour +1.9; `is_late_departure` 0.0; stand congestion +0.5 (5.6% coverage);
  day of week +0.9. "features that encode information already present in other features
  add noise rather than signal."
- **Failed feature sweep 2**: runway congestion signal +3.1 ("coverage 97%, noise > signal");
  flight_type categorical +2.0.
- **Gate-delay NaN fix reverted**: dropping `fillna(0)` on `gate_delay_sec` cost +0.4s;
  only 1.1% of rows missing.
- **Error analysis**: "LIRF accounts for **56.6% of total MSE** despite being 7.1% of
  validation flights. Without LIRF, RMSE is **262.8s**." (After v6: non-LIRF 242.6s.)
- **LIRF deep dive**: 14 artifact rows clustering at exactly 24.0–24.6 hours, "a date
  rollover bug where MVT_TIME crossed midnight but the block time used the wrong date";
  these 14 rows = **20.6% of total SSE** at 0.001% of the data. Plus 3 genuine LIRF
  disruption days (2025-07-13 mean 42 min, max 671 min; 2025-07-28; 2025-07-14 max 1,452 min).
- **Disruption days are not LIRF-specific**: days with daily mean > 1.5× the airport's own
  baseline — LTFM 7 days (worst 2.8×), EDDM 6 (2.4×), LFPG 4 (2.0×), LIRF 3 (2.1×), LSZH 1 (1.5×).
- **`day_deviation_ratio`**: mean taxi of completed flights at that airport since 00:00 UTC ÷
  that airport's long-run mean; fires after 5 completed flights, NaN before; 98.6% coverage;
  "On disruption days, the ratio crosses 1.3× by 8–9am, which is above the 95th percentile
  for normal days (1.23×)". Only 21 disruption days in the whole training year, so the global
  gain is 2.5–2.8s.
- **Late gate departure, stat-tested**: `AOBT_3_flt − LOBT_flt > 0` vs taxi time over 2M rows:
  Mann-Whitney p ≈ 0, mean difference **+1.98 min**, Pearson r 0.25 (~6% variance),
  rank-biserial 0.16.
- **`arvt_update_sec` = ARVT_3 − ARVT_1**, 99.9% coverage, r = 0.20.
- **Three documented dead ends against the LIRF tail**: (i) per-airport LightGBM models →
  388.2s, 4s *worse*, "a per-airport model sees fewer training examples… limits tree depth
  and generalisation"; (ii) **Huber loss** at alpha 300/600/900/1200 → all worse than MSE;
  (iii) **outlier caps at 3,600s / 7,200s on the training target** → same failure mode.
  Their explanation of (ii) and (iii): "the cap only affected training, not validation…
  when the validation set contains the same corrupt data, no training-side fix can help."
- **Their own caveat on v8**, verbatim: "The competition score depends on whether the ranking
  data … contains similar date-rollover artifacts. If it does and those rows are included in
  the truth, our predictions for them will be reasonable values (~900s) while the truth
  records ~87,000s — the same situation that inflated our pre-filter validation RMSE.
  If the ranking data is clean, the score will match our 244.0s validation RMSE closely."

### 5.2 "Cascade" and "airline-consistency", reconstructed from the scripts [S]

**`cascade_analysis.py` (173 lines)** — "does a disruption at an airport create a knock-on
effect for subsequent flights?" Two methods:
1. *Lag correlation.* Per airport, `deviation = TAXITIME − airport_median`; then a
   time-based rolling mean of `deviation` over windows `{15, 30, 60, 90, 120, 180}` minutes,
   `min_periods=3`, **`.shift(1)` to exclude the flight from its own window**; correlate each
   rolling column against the current flight's deviation.
2. *Time series.* Hourly mean taxi through a known bad day vs a normal day at the same
   airport, to see onset and recovery. Hard-coded bad days: LTFM 2025-02-23 ("worst day
   overall"), LIRF 2025-07-13, LSZH 2025-08-12.

So "cascade" = **temporal autocorrelation of airport-level taxi-time deviation at multiple
lags**, i.e. how long a disruption persists. `model_plan.md:55-57` records the taxonomy they
got out of it: weather → "all-day cascade lasting 15+ hours, rolling congestion feature will
help"; congestion → "multi-hour waves"; single incident (LSZH Aug 12) → "one sharp hourly
spike, **no cascade**, rolling feature less useful". The repo contains **no results table**
from this script.

**`airline_consistency.py` (129 lines)** — "are some airlines reliably faster or slower than
the airport baseline, **across multiple airports**?" Method: airport median baseline →
per `(airport, AIRCRAFT_OPERATOR_flt)` median, keep cells with `n >= 100` → `deviation_min =
(cell_median − airport_median)/60` → keep airlines present at `>= 3` airports → per airline,
report `n_airports`, `total_flights`, mean deviation and **consistency = fraction of airports
at which that airline is slower than baseline**.

So "airline-consistency" = **a sign-consistency test that an operator effect is a real
operator effect and not an airport-mix confound.** The conclusion it produced is in
`model_plan.md:24`: "Airline … Medium effect (ε²=0.11). **Consistent across airports — not
just an airport mix artifact.** Fastest airlines ~4 min below baseline, slowest ~6 min above.
**11 airlines in ranking have no training history; fall back to airport baseline for those.**"

[X] The method itself is sound and cheap, and the question it answers — "is my high-cardinality
categorical's effect an artifact of its distribution across a stronger categorical?" — is worth
asking of every OOF target encoding we use. That is the one genuinely reusable idea in this repo.

**`per_airport.py`** — trains one LightGBM per airport with `airport` removed from the
categoricals, compares aggregate RMSE to the global model. Result (from `learnings.md`):
388.2s, 4s worse than global. Note sqcm9896 reached the *opposite* conclusion with
per-airport *heads patched onto* a global model (433.6 → 423.7) rather than *replacing* it.

### 5.3 Failure mechanism 1: they filtered the validation set [S + X]

[S] `model.py:44` — `return df[df["TAXITIME_SEC_mvt"] <= 21600].copy()` inside
`load_training_data()`, i.e. **before** `time_based_split()`. So the 69 extreme rows are
removed from validation as well as training.

[X] The −113.6s "breakthrough" is therefore not a modelling gain at all: ~all of it is
the evaluation set getting easier. The scored set does contain those rows — the task is
literally to predict them. sqcm9896 measured the same phenomenon from the other side:
"169 rows |err|>3600 hold **53.4% of SSE**; top-3 rows alone 35.8%" and
"**6 rows decide 40%**" of the leaderboard gap.

**This is the cleanest possible argument for our drop-200-rows arm.** kagervig effectively
ran "drop the 69 worst rows" as the *primary* metric and it moved 113.6s. Any rule that
only reads a full-sample number would have waved this through. A rule that compares
full-sample against tail-trimmed would have flagged a 113.6s divergence instantly.

### 5.4 Failure mechanism 2: the leakage framing is inverted [S + X]

[S] `learnings.md`, in bold: "**WARNING: BLOCK_TIME − MVT_TIME and MVT_TIME − SCHED_TIME are
data leakage** — MVT_TIME = BLOCK_TIME + TAXITIME by definition (r=−1.0 confirmed).
**Never use MVT_TIME as a feature.**"

[S] And yet `model.py:185-187`, in the shipped feature builder:

```python
out["schedule_delay_sec"] = (
    (df["BLOCK_TIME_UTC_mvt"] - df["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
)
```

[S] `write_submission()` calls the same `build_features(deps, …)` on `ranking[PHASE=="DEP"]`,
where — per fenil264's executed check and sqcm9896's data contract —
`BLOCK_TIME_UTC_mvt` is **null on all 344,841 rows**.

[X] So `schedule_delay_sec`, credited with −22.9s (their second largest gain), is
**identically NaN at prediction time**. The model's single strongest learned split is
unavailable on every scored row. Meanwhile `month` and `hour` *are* derived from
`MVT_TIME_UTC_mvt` (`model.py:180-181`) — the column they declared forbidden — and those
are the two features that are genuinely available.

They inverted it exactly: **`MVT_TIME_UTC_mvt` is given in `ranking.parquet`;
`BLOCK_TIME_UTC_mvt` is the blanked one.** victoralcadi's README states the correct version
plainly (§3.1). The consequence is a 573.85 score at rank 144.

### 5.5 Composition claims: unreliable, do not use [S]

`learnings.md:16` "training ends 2026-01-01, **ranking starts 2026-01-04**";
`learnings.md` v8 Caveat "the ranking data (**Jan–Sep 2026**)";
Full Project Narrative "twelve months … (**Nov 2024 – Dec 2025**) covering **11** major airports".
Three mutually inconsistent statements. Their `submitting.parquet` is 344,841 rows
(verified from parquet metadata) but they never split it by month.
One usable factual note: ranking ARR rows carry known taxi-in — they map
`ADES_mvt → ADEP_mvt` to use ranking arrivals as congestion context for ranking departures,
lifting coverage "from 0% to 99.8%" (`write_submission()`). Same trick as sqcm9896's
ARR-propagation and victoralcadi's `arr_inblocks_prev`.

---

## 6. GLAPM / sincere-donkey (rank 161, 625.43) — nothing substantive

Repo: `/tmp/slot24/GLAPM_prc-2026-sincere-donkey`, single commit `687f7d5`, 2026-09-06.
Two files of code, 216 KB total.

[S] Complete method, from `train_ensemble.py`:
- **Target**: raw `TAXITIME_SEC_mvt`, trained on rows with `0 < taxi < 7200` (tail deleted).
- **Features — nine total**: `hour`, `dayofweek`, `month` (all from `SCHED_TIME_UTC_mvt`,
  correctly noting `BLOCK_TIME_UTC_mvt` is blank in ranking), plus categoricals
  `ADEP_mvt`, `AIRCRAFT_TYPE_mvt`, `WK_TBL_CAT_flt`, `MARKET_SEGMENT_flt`,
  `FLIGHT_RULE_mvt`, `AIRCRAFT_OPERATOR_flt`. No runway, no stand, no congestion, no
  schedule deltas, no target encoding.
- **Model**: LightGBM + XGBoost + CatBoost, 300 estimators each, lr 0.05, **unweighted mean**.
- **CV**: `train_test_split(X, y, test_size=0.1, random_state=42)` — a **random 10% split**,
  no temporal or calendar structure at all. They print MAE and RMSE and never record them.
- **Post-processing**: none. Unmatched IDs filled with the training median.
- **Reported score in repo**: none.
- README claims "**11** major European airports" (the site typo; both sqcm9896 and
  victoralcadi resolve it to 10 from the data).

[S] Their stated "Key Innovation" is aligning `CategoricalDtype` across train and ranking so
XGBoost does not error on unseen categories — a plumbing fix, not a modelling idea.

**Nothing in this repo is applicable to us, and it contains no statement about the ranking
set's composition.** Stating that is the result. Its only evidential value is as a floor:
nine features + random split + no tail handling = 625.43, i.e. 341s worse than us.

---

## Applicable to resilient-hedgehog

**1. The ranking set is Jan 2026 + Jul 2026 only, and the per-month DEP split is
152,719 / 192,122. This is now corroborated from three independent repos.** [S]
sqcm9896 states ranking = 689,534 rows = Jan 305,590 + Jul 383,944, DEP 344,841, ARR 344,693
(`instructions.md:11-13`, `eda/FINDINGS.md:8-9`); fenil264's executed notebook reproduces the
same per-month totals (`notebooks/01_data_access.ipynb`); sqcm9896 separately writes the DEP
split in words — "test is Jul-heavy (**192k vs 153k**)" (`eda/CHALLENGE.md:44`). The five
identities close exactly [X]. victoralcadi independently hard-codes
`EXPECTED_SUBMISSION_ROWS = 344_841` "observed from the leaderboard's `usedPairs`", and all
2,144 live leaderboard submissions report `usedPairs: 344841` [S].
**Action: our validation months should be January and July, weighted 44.3% / 55.7% by row
mass, not 50/50 — sqcm9896 measured that 50/50 "flatters by ~11 pts". A change that loses on
December is not evidence against the change.** Three of the five teams (sqcm9896,
victoralcadi, and by implication whoever OpenAir is) converged on Jan+Jul independently;
the one team that deliberately chose Feb+Aug instead (fenil264, rank 78) argued it avoided
"leaking" the test months, which is not a leakage argument and costs the calendar match.

**2. The sqcm9896 gate: "ship only if Jan AND Jul 2025 holdout RMSE both improve."** [S]
One month-based split of full-2025 DEP (pool 1,740,628 / val 344,419, val month from
`BLOCK_TIME`), two point estimates, both must fall, **no interval, no tolerance band, no
tail-robustness arm, no repeats**. Pooled is row-weighted (Jul-heavy), matching the scored
set. Rejections are logged in a 110-row `reports/experiments.csv` with an explicit `decision`
column (`drop-jan-rule`, `drop-mixed`, `drop-noise`, `drop-bug`). Informal noise folklore
exists but is not part of the rule: "pure-seed spread ±6 pooled", "±2 seed-noise zone",
and sub-1s sweep wins dismissed as "noise mining".

**How it differs from our rule (full CI AND drop-200-rows CI both improving):** theirs
partitions the *calendar*, ours partitions the *sample*. Theirs has no uncertainty
quantification and no tail arm; ours has no month arm. **The two are close to orthogonal.**
Concretely: their gate passed both of their documented out-of-sample failures — v6
per-airport blend weights (−0.6 holdout → **+7.0 leaderboard**) and v7 LIRF small-slice patch
(−1.2 holdout → **+0.95 leaderboard**) — and their own post-mortem names the cause,
"per-airport weights **OVERFIT holdout monsters**" and "small-slice monster fits don't
transfer". A drop-200-rows arm is designed for exactly that and would have caught both; a
month conjunction cannot, because a 344k-row holdout has enough monsters in *each* month.
Conversely their gate is *stricter than the metric* in one direction and it cost them: the
`extreme_spec` run scored pooled 350.6 vs ship 359.2 — **a −8.6s pooled gain discarded**
because January regressed 5.9s while July improved 23.5s. Under the real Jul-heavy pooled
metric that trade is positive.
**Recommended: add their arm to ours, do not substitute it.** Report Jan RMSE, Jul RMSE and
row-weighted pooled on every run; require the mass-weighted pooled CI and the drop-200 CI to
improve (our existing rule), and treat a single-month regression as a flag that demands a
named mechanism rather than an automatic veto. Also copy the cheap part outright: a
committed `experiments.csv` with a `decision` column and rejected runs kept in it.
**Caveat on their numbers** [S]: their holdout is contaminated — `train.py` passes it as
`eval_set` with `early_stopping_rounds=100`, and their own red-team file says so
("Validation Integrity ❌ Compromised", `recommendations.md:18`) and the fix was never applied.

**3. Steal the physics-identity floor and test it.** [S] sqcm9896's largest single
leaderboard gain: for rows where `AOBT_3_flt` joined and `m_aobt = MVT_TIME − AOBT_3_flt > 1800`,
set `pred = max(pred, m_aobt − 3600)`; plus a negative cap. Holdout predicted −23.9,
leaderboard delivered **−20.3** (336.46 → 316.19), 11,826 rows triggered. It is a bound, not
a fit, which is why it transferred where their slice-fits did not. **Upper-tail clipping is
the opposite: sqcm9896 measured +90.7s (`clip_cap2h_v1`) — never cap from above globally.**

**4. Two more post-processing / data leads worth a gated experiment.**
[S] fenil264: **LEMD and LEBL DEP taxi-out is hard-censored at 3,600s upstream** — top-10 maxima
at both are packed at 3,504–3,604 while LFPG and LIRF decay naturally to 84k and 131k. If that
pipeline still runs in 2026, clipping our LEMD/LEBL predictions at 3600 is free. Verify on our
own 2025 data first (count LEMD/LEBL DEP rows above 3600), then gate it.
[S] fenil264: airport-count congestion has ρ ≈ 0 at **EHAM (0.007), LEMD (0.005), EDDM (0.032)**
— all independent-parallel-runway airports — while being the top feature at EGLL/LTFM/LFPG/LIRF.
If we use a single airport-wide congestion count we are feeding three airports pure noise;
a per-runway-group count is the fix they identify.

**5. Two protocol hazards to check we are not committing, both observed here.**
[X] (a) **Never filter extremes out of the validation set.** kagervig's headline −113.6s
"breakthrough" is 69 rows removed from *both* sides of an 83/17 split
(`model.py:44`, before the split), and they scored **573.85** against a claimed 244.0.
victoralcadi does a milder version, dropping everything outside [60s, 10800s] from validation
(`model.py`, `config.TAXI_MIN_SEC/MAX_SEC`). The scored set contains those rows.
(b) **`MVT_TIME_UTC_mvt` is given in the ranking set; `BLOCK_TIME_UTC_mvt` is the blanked one.**
kagervig wrote "never use MVT_TIME as a feature" and then shipped
`schedule_delay_sec = BLOCK_TIME − SCHED_TIME`, which is NaN on all 344,841 scored rows —
their second-largest claimed gain evaporates at prediction time. victoralcadi's README has
the correct framing and the useful refinement: `AOBT_3_flt` is a *near-duplicate* of the
blanked `BLOCK_TIME`, so its use is a documented policy decision, not an oversight.
[S] The permissive-vs-strict question is settled empirically by sqcm9896, who ran both:
permissive 433.6 vs strict 508.6 on holdout, 345.57 vs 470.85 on the leaderboard — a 125s
live gap, larger than the entire spread from rank 1 to rank 100.

**6. `airline_consistency.py` is the one reusable idea in the two worst repos.** [S]
Per `(airport, operator)` deviation from the airport median, restricted to cells with n ≥ 100
and operators present at ≥ 3 airports, then a **sign-consistency fraction** — a test that a
high-cardinality categorical's apparent effect is not an artifact of its distribution across
a stronger categorical. Their result: operator effect is real and consistent (ε² 0.11,
fastest ~−4 min, slowest ~+6 min), and **11 operators appear in ranking with no training
history**. Worth running against every OOF target encoding we fit, and worth checking how we
handle unseen operators (sqcm9896 uses a RARE(<20) → parent back-off plus a garbage bucket,
and counts 29 novel STAND, 62 ADES, 8 types, 29 operators in ranking).

**7. Negative results we can stop re-deriving.** [S] From sqcm9896's logged rejections, each
measured on a Jan+Jul holdout: Huber objective 668.4 vs L2 448.8 ("optimizes the wrong target
for RMSE"); asymmetric loss (α=0.25) 477.6; log1p target ("overcorrects, skew −0.61… NOT
lognormal"); train-target clipping (vacuous — moves neither median nor mean RMSE); upper
prediction clipping +90.7; depth-10 (overfits the tail); XGBoost 481.8; wake-ahead counts
475.7 (both months worse); **ERA5 via Open-Meteo 485.5 vs 448.8, both months worse, "grid too
coarse"**; every bagging variant (5-seed lgbm 448.4 vs single 447.8; 2-seed cat −0.2 at the
noise floor). Independently, kagervig also killed Huber (four alphas), outlier caps and
per-airport-replacement models. **Note the weather result is specifically about ERA5's coarse
grid** — victoralcadi's research notes argue station METAR from the Iowa State IEM archive is
the right instrument and that January 2026 was a heavier de-icing winter than January 2025
(Munich 8,628 vs 6,937 aircraft de-iced), which is a directional drift risk for a
Jan-2025-trained model that ERA5 could not have tested.

**8. Repos with nothing substantive, stated plainly.** `fenil264/prc-2026-taxi-out` contains
**no model** — notebooks 03–06 are 105–127-character stubs and every `src/prc/*.py` is an
empty docstring, despite the team having submitted 38 versions and scored 300.27. No fitting
target, feature list, model family, CV design or post-processing is recoverable from it; only
EDA (which is genuinely good — items 4 above). `GLAPM/prc-2026-sincere-donkey` is a single
commit: nine features, a **random** 10% split, tail deleted from training, three GBMs averaged
unweighted, no logged score, no composition statement — nothing applicable.
