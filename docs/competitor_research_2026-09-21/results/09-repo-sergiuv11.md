# SLOT 09 — repo-sergiuv11: code-level read of two public PRC-2026 repos

Date: 2026-09-21. Investigator: code-level read of cloned sources plus the live leaderboard API.

Marking convention: **[S]** = stated in the prior summary handed to me; **[X]** = verified by me
in this investigation (file + line, or a live API call). Nothing below is dressed up: where the
code cannot answer a question I say so.

Clones (read-only, outside the project tree):

- Repo A: `https://github.com/sergiuv11/prc-data-challenge-2026` → `/tmp/slot09/repoA`,
  single squashed commit `e3b653b`, 2026-09-11 16:51 +0200, author Sergiu Vincze. **[X]**
- Repo B: `https://github.com/ahmetabdullahgultekin/prc-taxiout-2026` → `/tmp/slot09/repoB`,
  author Ahmet Abdullah Gültekin, team `vibrant-lollipop`. Default branch `master` is **stale**
  (tip `9d50bf6`, 2026-09-01). All real work is on `origin/feature/pluggable-learners`
  (tip `f2fa982`, 2026-09-09, 90 commits). Everything below for Repo B is read from that
  branch unless stated. **[X]**

---

## 0. THE HEADLINE CORRECTION — Repo A is *not* level with us

The slot brief says "team `jubilant-vase`, leaderboard 286.66 — essentially level with us" **[S]**.
**That is wrong, and by a large margin.** I pulled the full live leaderboard
(`https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard`,
paged with `limit=500` + `nextCursor`, 2,144 scored rows / 168 teams, 2026-09-21): **[X]**

| team | current best | rank (of 168) | file | scored |
|---|---:|---:|---|---|
| `vigorous-whistle` (leader) | **239.1335** | 1 | `_v22` | 2026-09-20 |
| `jovial-uniform` | 240.5497 | 2 | `_v25` | 2026-09-20 |
| `jolly-lobster` | 242.0575 | 3 | `_v41` | 2026-09-20 |
| **`resilient-hedgehog` (us)** | **284.3584** | **43** | `_v10` | 2026-09-17 |
| **`jubilant-vase` (Repo A)** | **299.2862** | **76** | `_v7` | 2026-09-14 |
| **`vibrant-lollipop` (Repo B)** | **310.5502** | **85** | `_v25` | 2026-09-08 |

Every row on the current board has `usedPairs = 344841`. **[X]**

**Why the 286.66 figure is stale.** The organisers replaced the ranking set on **2026-09-04** and
reset the leaderboard: July went from 3 airports to all 10, scored departures went
215,876 → 344,841, every old submission was erased, and every team's score moved by roughly
**+25 s** (the then-leader 249.46 → 277.29). This is documented independently in Repo B
(`/tmp/slot09/repoB/docs/facts.md:161`, facts R30–R32, sourced to the bucket listing and the
leaderboard API) **[X]**. Repo A's README claims *"Official score: 286.656 seconds RMSE, scored on
all 215,876 ranking pairs"* (`/tmp/slot09/repoA/README.md:6`) **[X]** — i.e. the **old** board.
The 286.656 and 290.488 figures in `docs/RESULTS.md:7-8` do not appear anywhere in the current
board data. **[X]**

**Repo A's real trajectory on the current board** (from the API, not in the repo): **[X]**

| file | board RMSE | scored at |
|---|---:|---|
| `jubilant-vase_v4.parquet` | 327.8083 | 2026-09-11 23:34 |
| `jubilant-vase_v5.parquet` | 320.0413 | 2026-09-12 00:46 |
| `jubilant-vase_v6.parquet` | 301.2940 | 2026-09-13 21:54 |
| **`jubilant-vase_v7.parquet`** | **299.2862** | 2026-09-14 00:15 |
| `jubilant-vase_v8.parquet` | 299.5827 | 2026-09-14 02:26 |

The repository's last commit (2026-09-11 16:51) predates V6–V8, so **V4–V8 are entirely
undocumented in the repo**. The published V3 architecture corresponds to nothing currently on the
board. **[X]**

**Consequence for the slot's framing.** The question "what do they do that we do not" is still
worth answering, but the premise "they are level with us" is false: on the same 344,841 pairs we
are **14.93 s ahead of them**. The detailed architecture below is the architecture of a
**299.29-second** system, not a 286.66-second one, and it was designed for a ranking-set
composition that no longer exists (see §1.6).

---

## 1. REPO A — `jubilant-vase` (Sergiu Vincze, team 2nd place PRC 2025)

Layout: 35 scripts under `scripts/`, 4 docs under `docs/`, no package, no tests, no data.
Total 5,837 lines of Python. **[X]**

### 1.1 Exact feature list — 43 inputs

All definitions from `/tmp/slot09/repoA/scripts/features.py`. The model reads only these; the
two blanked columns `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` are never read
(`features.py:5-7`). **[X]**

**11 categorical** (`features.py:20-24`), declared to LightGBM as `categorical_feature`, encoded
as stable integer codes over the union of train+ranking, nulls → `-1` (`train_model.py:73-79`):

`AIRPORT`, `RUNWAY_mvt`, `STAND_mvt`, `stand_area`, `AIRCRAFT_TYPE_mvt`,
`AIRCRAFT_OPERATOR_flt`, `MARKET_SEGMENT_flt`, `WK_TBL_CAT_flt`, `FLIGHT_TYPE_flt`,
`FLIGHT_RULE_mvt`, `ADES_mvt`.

- `AIRPORT` = `ADEP_mvt` for DEP, `ADES_mvt` for ARR (`features.py:79-86`).
- `stand_area` = `STAND_mvt` with `[0-9].*$` stripped, i.e. the leading pier letters
  (`features.py:181`).

**17 row-level numeric** (`features.py:33-45`, computed at `features.py:151-179`). `secs(a,b)` =
`(a − b).dt.total_seconds()`:

| feature | definition | clip |
|---|---|---|
| `implied_taxi_aobt` | `MVT_TIME − AOBT_3_flt` | `(-7200, 172800)` |
| `aobt_missing` | `AOBT_3_flt is null` as Int8 | — |
| `takeoff_minus_schedule` | `MVT_TIME − SCHED_TIME` | `(-7200, 172800)` |
| `takeoff_minus_lobt` | `MVT_TIME − LOBT_flt` | `(-7200, 172800)` |
| `takeoff_minus_eobt` | `MVT_TIME − EOBT_1_flt` | `(-7200, 172800)` |
| `takeoff_minus_iobt` | `MVT_TIME − IOBT_flt` | `(-7200, 172800)` |
| `sched_minus_aobt` | `AOBT_3_flt − SCHED_TIME` | `(-14400, 14400)` |
| `eobt_minus_aobt` | `AOBT_3_flt − EOBT_1_flt` | `(-14400, 14400)` |
| `iobt_minus_aobt` | `AOBT_3_flt − IOBT_flt` | `(-14400, 14400)` |
| `lobt_minus_aobt` | `AOBT_3_flt − LOBT_flt` | `(-14400, 14400)` |
| `planned_flight_time` | `ARVT_1_flt − EOBT_1_flt` | `(0, 86400)` |
| `hour`, `minute_of_day`, `weekday`, `month`, `day_of_year`, `is_weekend` | all from `MVT_TIME` | — |

The asymmetric clip is deliberate and commented (`features.py:155-157`): *"Real taxi-out reaches
131,167 s in this dataset and the extreme records dominate RMSE, so a tight clip would erase the
signal that predicts them."* Missing numerics are **left null** so LightGBM learns the routing
(`features.py:152-154`). **[X]**

**15 congestion** (`features.py:26-31`, built at `features.py:100-143`). All counts are Polars
`rolling` over the **combined DEP+ARR movement stream**, indexed on take-off time, grouped by
airport; the `-15m/-30m/-60m` offsets are backward windows and `0s` offsets are forward:

`dep_prev_15/30/60`, `dep_next_15/30`, `arr_prev_15/30/60`, `arr_next_15/30` (airport-level,
from the combined stream — `features.py:107-123`); `runway_prev_15/30` (per `(AIRPORT, RUNWAY)`,
departures only — `features.py:125-127`); `sched_dep_prev_30`, `sched_dep_next_30` (indexed on
`SCHED_TIME`, planned pressure — `features.py:130-133`); `dep_share_runway_30` =
`runway_prev_30 / dep_prev_30` as a runway-configuration proxy (`features.py:139-142`).

Leakage control: training and ranking Parquets are windowed **separately** so a 2025 row never
sees a 2026 neighbour (`features.py:8-10`, `docs/METHODOLOGY.md:17-19`). **[X]**

**Built but excluded from the shipped feature set** (defined in `features.py:49-66`, switched on
only by CLI flags in `train_model.py:307-327`): 19 METAR features `WEATHER_NUMERIC`,
2 completed-arrival features `ARRIVAL_TAXI_NUMERIC` (`arr_taxi_mean_60`, `arr_taxi_n_60`),
1 ground-queue feature `dep_taxiing_at_aobt`, 1 `sched_minute_of_day`, plus target encodings in
`target_encoding.py`. **All of these failed their gate and are not in the model**
(`docs/RESULTS.md:45-51`). **[X]**

### 1.2 Exact hyperparameters

`train_model.py:48-66`. **One fixed configuration for all four boosters.** **[X]**

```
objective        = "regression"   # L2
metric           = "rmse"
learning_rate    = 0.06
num_leaves       = 127
min_data_in_leaf = 200
feature_fraction = 0.8
bagging_fraction = 0.8
bagging_freq     = 1
lambda_l2        = 1.0
max_cat_threshold = 64
cat_smooth       = 20.0
num_threads      = 8
seed             = 20260901
MAX_ROUNDS = 3000 ; EARLY_STOP = 100 ; BOOTSTRAP = 1000
```

Final models use a **frozen 436 trees** (`build_specialist_submission.py:26`,
`reproduce_v3.sh:54/63/110`), no early stopping — the count came from an earlier early-stopped
gate run and is then held fixed so validation and final fit are the same configuration
(`train_model.py:195-198`, `docs/RESULTS.md:24-26`). Final predictions are clipped to
`[60, ∞)` (`train_model.py:517`, `build_specialist_submission.py:94/132`). **[X]**

Capacity was swept exactly once, on a dedicated October holdout, over 5 pre-declared
configurations (`select_capacity.py:17-26`): leaves ∈ {63, 127, 255} × `min_data_in_leaf`
∈ {50, 200, 500}, with `BOOTSTRAP = 2000` and `SELECTION_CONFIDENCE = 0.9875`
(`select_capacity.py:27-29`). The incumbent (127/200) survived — `docs/RESULTS.md:48-50` lists
"model-capacity selection" among rejected branches. **[X]**

CatBoost was implemented (`train_catboost.py:57-70`: Plain boosting, depth 8, lr 0.06,
`border_count=128`, `max_ctr_complexity=1`, `one_hot_max_size=2`, `od_wait=100`) and a 50/50
LightGBM+CatBoost diversity blend was pre-declared (`blend_predictions.py:2-6`). **It failed its
gate and is not in the model** (`docs/RESULTS.md:48`). **[X]**

### 1.3 CV design

**No K-fold anywhere.** Three temporal simulations, plus two auxiliary splits used only for
selection (`train_model.py:419-489`, `docs/METHODOLOGY.md:56-71`): **[X]**

| split | fit | inner (early-stop) | test |
|---|---|---|---|
| **composition** | all months except 1, 7, 12 | month 12 | Jan (all airports) + Jul (EDDF/EGLL/EHAM only) |
| **forward** | months 1–5 | month 6 (buffer) | Jul at EDDF/EGLL/EHAM |
| **seasonal** | days 1–18 of every month | days 19–20 | days 22–31 in the scored composition (day 21 dropped as buffer) |
| `december` (aux) | months 1–10 | month 11 | month 12, all airports |
| `october_selection` (aux) | months 1–8 | month 9 | month 10 — **selection only, never a confirmation gate** |

The acceptance gate (`compare_predictions.py:9-15`, `select_airport_membership.py:60-66`,
`docs/METHODOLOGY.md:68-71`) is the interesting part and is enforced in code:

1. must improve **raw** composition RMSE **and** **overlay-adjusted** composition RMSE;
2. the LIRF overlay is re-fitted **on the composition fit pool only** for the comparison, so
   the shipped 12-month coefficients cannot leak test months in
   (`compare_predictions.py:9-12`);
3. must win ≥ 95 % of 2,000 paired bootstrap resamples (`compare_predictions.py:34`;
   ≥ 80 % for the membership selector, `select_airport_membership.py:63`);
4. **must not derive more than 50 % of its net advantage from a single row**
   (`select_airport_membership.py:64`, `maximum_row_share=50.0`, computed at
   `select_airport_membership.py:136-138`);
5. confirmation on forward **and** seasonal before any submission
   (`reproduce_v3.sh:84-99`, which `grep`s for "Verdict: PASS" and exits 1 otherwise);
6. `train_model.py:265-283` (`check_gate`) refuses a final fit unless a preserved report and its
   matching prediction file prove the model beat the control.

Local numbers (`docs/RESULTS.md:18-34`) **[X]** — note these are on the *old* composition:

| gate | global-system RMSE | V3 RMSE | gain | bootstrap wins |
|---|---:|---:|---:|---:|
| composition (with equal overlay) | 325.931 | 323.555 | 2.375 | 100.0 % |
| July forward | 250.341 | 241.598 | 8.743 | 100.0 % |
| seasonal | 403.286 | 401.471 | 1.815 | 100.0 % |

### 1.4 The LIRF overlay — exact rule, threshold, row count, gain

Implementation: `experiment_tail_overlay.py`, applied in `train_model.py:518-533` and
`build_specialist_submission.py:96-104/145-148`. **[X]**

**Trigger** (`experiment_tail_overlay.py:116-124`) — all three must hold:

```
AIRPORT ∈ {LIRF}                     (scope is LIRF only)
AND aobt_missing == 1                (AOBT_3_flt is null)
AND takeoff_minus_schedule > 14400   (4 hours; nan → -1e9, so nulls never fire)
```

**Replacement** (`experiment_tail_overlay.py:123`): the prediction is replaced by
`a + b * takeoff_minus_schedule`, where `(a, b)` is an ordinary least-squares fit of
`target ~ 1 + takeoff_minus_schedule` on 2025 LIRF rows satisfying the *same* trigger, with a
hard floor of **20 trigger rows** or the overlay is skipped
(`experiment_tail_overlay.py:127-137`). Result is then re-clipped to `[60, ∞)`.

**Threshold**: `OVERLAY_THRESHOLD = 14_400` s, hardcoded in
`build_specialist_submission.py:28`, `select_airport_membership.py:20`,
`compare_predictions.py:33`; CLI default `--overlay-threshold 14400` at `train_model.py:331`.
Chosen from a sweep over `THRESHOLDS = [3600, 5400, 7200, 10800, 14400, 21600, 28800]`
(`experiment_tail_overlay.py:38`) × scope ∈ {LIRF-only, all airports} × form ∈ {identity,
calibrated}, with a leave-one-month-out validation of the form
(`experiment_tail_overlay.py:74-106`) and a month-by-month stability table. **[X]**

**Row count**: *"It changed 15 of 215,876 ranked rows"* (`docs/METHODOLOGY.md:54`). **[X]**
Note: that is the **old** ranking set. The count on the current 344,841-row set is unknown and
is not recomputable without the data. Also `experiment_tail_overlay.py:227`: *"LIRF appears only
in January 2026, so the overlay cannot affect the July rows"* — this was true of the old
ranking file and is **false** of the current one (July now holds all 10 airports, Repo B
`docs/facts.md:161` R30). **[X]**

**Gain**: **cannot be determined from the repo.** The sweep is computed at runtime and written to
`reports/experiment_tail_overlay.md`, which is git-ignored and not committed. `docs/RESULTS.md`
reports only V2 (global + overlay) = 290.488 and V3 = 286.656; no un-overlaid submission was ever
scored, so no isolated board gain exists. Given it touches 15 of 215,876 rows, any effect is
confined to those rows. **[X]**

**Reading**: this is a hand-narrowed special case of the same underlying phenomenon Repo B
generalised (the feed writing the scheduled time into the off-block field — see §2.3). Repo A
restricts it to one airport, one missingness condition, and a 4-hour cut, and gets 15 rows.
Repo B makes it a probabilistic mixture over all rows and got **77 s on the board**. This is the
single largest methodological gap between the two repos.

### 1.5 The airport-specialist scheme

`experiment_airport_specialists.py` (validation) and `build_specialist_submission.py`
(final build). **[X]**

- **Which airports**: `DEFAULT_AIRPORTS = ("EDDF", "EGLL", "EHAM")`
  (`experiment_airport_specialists.py:20`, `build_specialist_submission.py:23`). These are
  exactly `JULY_AIRPORTS` — *"the only airports present in July 2026"* (`train_model.py:46`).
  A mechanical selector exists (`select_airport_membership.py`) that would pick membership on
  the October holdout by improvement ≥ 1.0 s, bootstrap wins ≥ 80 %, and largest-row-share
  ≤ 50 % — **but the shipped set is the hardcoded triple, not the selector's output**; the
  selector writes `membership.json` and is not invoked anywhere in `reproduce_v3.sh`. **[X]**
- **How trained**: one LightGBM per airport on that airport's rows only, from the same fit pool,
  **identical PARAMS**, 436 fixed trees, same 43 features
  (`experiment_airport_specialists.py:167-193`, `build_specialist_submission.py:121-143`).
  An `--early-stop` path exists (per-airport tree count selected on the split's inner rows,
  `experiment_airport_specialists.py:175-189`) but `reproduce_v3.sh:57-66` does **not** pass it,
  so specialists ship at the global 436.
- **How blended**: `BLEND_WEIGHT = 0.5`, literally
  `candidate[mask] = 0.5 * global_raw[mask] + 0.5 * local_prediction`
  (`build_specialist_submission.py:27/133-134`;
  `experiment_airport_specialists.py:23/198-199`; also `calibrate_v3.py:43`,
  `apply_preserved_specialists.py:82-83`). Applied per-airport by row mask; every other airport
  keeps the global prediction unchanged.
- **Order of operations**: global raw → per-airport 50/50 overwrite → LIRF overlay last →
  clip to `[60, ∞)` → round → cast to template dtype (`build_specialist_submission.py:118-149`).
- **Was the weight tuned?** **No.** It is a module-level constant in three files with no sweep
  anywhere. `blend_predictions.py` has a `--left-weight` (default 0.5) but its own docstring says
  it *"does not inspect targets or optimise a weight, so running it on the composition
  predictions cannot quietly tune the blend against the held-out answers"*
  (`blend_predictions.py:2-6`). Nothing in the repo searches over 0.5. **[X]**
- **Effect on the file**: relative to V2, 112,221 of 215,876 rounded predictions changed, all at
  the three specialist airports (`docs/RESULTS.md:36-37`); board gain V2→V3 = 3.832 s
  (`docs/RESULTS.md:10`), on the **old** board. **[X]**

### 1.6 The design is keyed to a ranking-set composition that no longer exists

This is the structural weakness behind the 286.66 → 299.29 drop. **[X]**

- `JULY_AIRPORTS = ["EDDF","EGLL","EHAM"]  # the only airports present in July 2026`
  (`train_model.py:46`) — a fact with an expiry date written in as a constant.
- The composition split (`train_model.py:404-411`, `composition_mask` at
  `train_model.py:89-94`) mirrors Jan-all + Jul-three.
- The forward split tests **only** July at those three airports (`train_model.py:424-428`).
- The specialist membership **is** that triple.
- The overlay's exposure argument (`experiment_tail_overlay.py:227`) assumes LIRF is January-only.

After 2026-09-04 July holds all 10 airports and is now the **larger** month at 55.7 %, and it is
the higher-variance one (Repo B `docs/facts.md:161-162`, R30/R32) **[X]**. Repo B hit the same
trap and removed its own three-airport exception, saying so explicitly
(`repoB src/taxiout/application/pipeline.py:231-240`): *"It used to carry an exception, because
until 2026-09-04 the ranking set held only three airports in July… the exception became a
distortion, so it is gone."* Repo A's committed code never made that change. **[X]**

### 1.7 Continuity with the same author's PRC-2025 2nd place

**The 2nd-place claim checks out.** The PRC Data Challenge 2025 outcome page lists
**`jubilant-vase`, €1,750, RMSE 211.99, 2nd of 3 prize winners**
(`https://prc-data-challenge-2025.netlify.app/outcome.html`) **[X]** — the same team name.
The 2025 repo `github.com/PRC-Data-Challenge-2025/jubilant-vase` describes an ensemble of
**CatBoost 64.1 % / LightGBM 19.9 % / XGBoost 16.0 %**, with **5-fold GroupKFold grouped by
`flight_id`**, and the blend weights fitted by **`scipy.optimize.minimize` on out-of-fold
predictions** **[X]** — matching the prior summary **[S]** precisely. (Caveat: the outcome page
describes the 2025 task as take-off weight/ATOW; Repo B's log calls it fuel burn→fuel flow. The
task name is not load-bearing here.)

**Did they carry it over? No — every one of the four elements was dropped.** **[X]**

| PRC 2025 element | PRC 2026 Repo A |
|---|---|
| 3-model CatBoost/LGBM/XGB blend | **single global LightGBM**; CatBoost built (`train_catboost.py`) but closed at its gate (`docs/RESULTS.md:48`); XGBoost absent entirely (not in `requirements.txt` paths, no script) |
| scipy-optimised OOF blend weights | **no weight optimisation anywhere**; the only blend is a hardcoded 0.5 between global and airport specialists, and `blend_predictions.py:4-6` explicitly refuses to optimise |
| 5-fold GroupKFold by flight | **no K-fold at all**; three temporal simulations (§1.3) |
| — | new: pre-declared acceptance gates + paired bootstrap + single-row-share guard |

**Is there a note in the repo explaining why? No.** I grepped the entire repo for
`catboost|xgboost|2nd place|second place|prc 2025|2025 challenge|groupkfold` — the only hits are
the CatBoost implementation files and one line in `docs/RESULTS.md:48` listing CatBoost among
rejected branches. **There is no reference to the 2025 solution anywhere in Repo A.** **[X]**
The nearest thing to a rationale is a methodological one, not a retrospective one:
`docs/METHODOLOGY.md:57-58` — *"Random row splitting would mix nearby airport operations and
overstate generalisation"* — which explains dropping K-fold but says nothing about the blend.
The CatBoost rejection is recorded only as "failed its gate"; **the measured numbers are not
committed** (they live in git-ignored `reports/`). So: *what* they dropped is verifiable, *why
the blend specifically was dropped* is **not determinable from the code** beyond "it did not
clear the pre-declared gate".

### 1.8 Repo A experiment log

The repo is a **single squashed commit**, so there is no commit-level log. The entire
reconstructable log is: **[X]**

| version | architecture | official RMSE | board | source |
|---|---|---:|---|---|
| V1 | hierarchical historical-median control (`baselines.py`) | *never reported* | — | used only as the in-code control |
| V2 | global LightGBM + LIRF overlay | 290.488 | **old** (215,876) | `docs/RESULTS.md:7` |
| V3 | V2 + EDDF/EGLL/EHAM 50/50 specialists | 286.656 | **old** | `docs/RESULTS.md:8` |
| V4 | undocumented | 327.8083 | current (344,841) | leaderboard API |
| V5 | undocumented | 320.0413 | current | leaderboard API |
| V6 | undocumented | 301.2940 | current | leaderboard API |
| V7 | undocumented | **299.2862** | current | leaderboard API |
| V8 | undocumented | 299.5827 | current | leaderboard API |

**Failures**, named but **not quantified** (`docs/RESULTS.md:45-51`): target encoding, public
METAR, CatBoost, learning-rate changes, more airport specialists, alternative early stopping,
congestion ablation, seed ensembles, wake specialists, completed-arrival state, active taxi
queue, scheduled clock time, model-capacity selection, post-hoc calibration. *"Each rejected
branch kept V3 safe. Branches that failed their gate were closed without being built or
submitted."* **Every one of these numbers is in git-ignored `reports/` and is not published.**
This is the one place Repo A is markedly less useful than Repo B. **[X]**

One honest disclosure worth noting: a clean 8-thread retrain is **not** bit-identical —
fresh V3 differed from submitted V3 on 171,075 of 215,876 rounded rows, mean |Δ| 3.37 s, max
378 s (`README.md:47-51`, `docs/REPRODUCIBILITY.md:67-82`). LightGBM's floating-point histogram
reductions resolve near-tied splits differently across runs. **[X]** For us: **a 3.37 s mean
per-row wobble from thread non-determinism alone is larger than most of the gate margins in
this repo**, and is a real caution for any of our own sub-5-second A/B calls.

---

## 2. REPO B — `vibrant-lollipop` (Ahmet Abdullah Gültekin)

Board best **310.5502** (v25, 2026-09-08), rank 85/168 **[X]**. Architecturally the more
interesting repo despite the weaker score: 90 commits, a 1,061-line experiment log, a 248-line
sourced fact register, and it settles questions with the board rather than the holdout.

### 2.1 ATXOT-style P10 unimpeded reference — full construction

`/tmp/slot09/repoB/src/taxiout/domain/reference.py`. A deliberate reimplementation of
EUROCONTROL's *additional taxi-out time* indicator (ATXOT Edition 01.00, 2023-03-16; the PDF is
committed at `docs/reference/ATXOT_indicator_documentation_mar23.pdf`). **[X]**

**Constants** (`reference.py:42-44`):
```
PERCENTILE   = 0.10                 # P10
MIN_BELOW    = 10                   # ATXOT p.15: ≥ 10 flights at or below the P10
MAX_TAXI_SEC = 120 * 60 = 7200      # ATXOT p.13 step 1: > 2 h leaves the sample
```

**Sample filter before fitting** (`reference.py:86-91`): `PHASE_mvt == "DEP"` AND
`TAXITIME_SEC_mvt` not null AND `> 0` AND `<= 7200`.

**Grouping keys and fallback chain** (`reference.py:47-52`) — most specific first, first *valid*
level wins:
```
apt_stand_rwy : (apt_mvt, STAND_mvt, RUNWAY_mvt)
apt_stand     : (apt_mvt, STAND_mvt)
apt_rwy       : (apt_mvt, RUNWAY_mvt)
apt           : (apt_mvt)
```
`apt_mvt` is the *movement* airport (ADEP for DEP, ADES for ARR), not `ADEP_mvt` — a distinction
that had caused a real bug (`reference.py:35-39`; Repo B `docs/facts.md` R02). **[X]**

**Minimum group size**: there is **no raw `n` minimum**. The validity test is ATXOT's own:
`below_<level> >= 10`, where `below` counts the flights in that group whose taxi-out is **at or
below the group's own P10** (`reference.py:66-76`). A level is usable only if
`valid_<level>` AND `p10_<level>` is non-null (`reference.py:105-108`).

**Two stated departures from the official method** (`reference.py:21-27`): ATXOT *drops* flights
with no valid reference — they cannot, hence the fallback chain; and ATXOT uses a rolling
12 months — they have only calendar 2025, so the reference is fixed rather than rolling.

**How it is used**: `apply_reference` emits three columns (`reference.py:112-121`):
- `reference_sec` (Float32) — the winning level's P10;
- `reference_level` (String) — **declared as a categorical feature**
  (`pipeline.py:46-48`), so the model knows how specific the reference was;
- `reference_sample` = `n_apt_stand_rwy` — how well-observed the most specific combo is,
  *"tells the model how far to trust the reference"*.

**Feature or offset? Both were implemented; the feature won.** `pipeline.train_predict` takes
`residual: bool` (`pipeline.py:306-345`): with `residual=True` it fits `y − reference_sec` and
adds `reference_sec` back to the prediction, filling nulls with the column mean. **The submitted
models use the raw target with `reference_sec` as a plain feature**, and the README was corrected
to say so (commit `c184a4d`: *"corrects the approach section, which claimed the model learns the
residual… The submitted models predict the raw taxi-out time and take the reference as a
feature, where it is the strongest single one"*). **[X]**

**Measured value** (`docs/experiments.md:15-16, 43-48`): **[X]**

| target parameterisation | holdout RMSE | Jan | Jul |
|---|---:|---:|---:|
| naive (`MVT − AOBT_3`, no model) | 531.40 | — | — |
| **raw target, `reference_sec` as feature** | **377.84** | 422.42 | 240.63 |
| ATXOT-P10 residual target | 379.73 | 424.87 | 240.43 |
| NM-residual target (naive prediction as base) | 382.74 | 427.05 | 247.22 |

And the author's own **retraction** (`docs/experiments.md:108-115`): those three differ by 2–5 s
against a measured paired noise floor of ~5 s, so *"no parameterisation was shown to beat any
other"*. **The honest conclusion is that the residual formulation is neither better nor worse.**

**Gain share**: `reference_sec` alone carries **18.8 % of LightGBM gain**, the strongest single
feature (`docs/experiments.md:152`, `docs/facts.md` MF22), ahead of `eobt_offset_sec` 16.0 %,
`sched_offset_sec` 15.4 %, `nm_naive_taxi_sec` 10.4 %, `STAND_mvt` 7.5 %. **[X]**

**Leakage discipline**: the reference tables are fitted **only on the non-holdout months**
(`pipeline.py:259`: `reference.fit_reference(movements.filter(~holdout_mask()))`), with an
explicit comment that including the holdout *"would flatter the out-of-fold numbers in a way the
board would later take back"* (`reference.py:80-84`). **[X]**

### 2.2 The exact weather join

Downloader `src/taxiout/adapters/metar_iem.py`, join `src/taxiout/features/weather.py`. **[X]**

- **Source**: Iowa Environmental Mesonet ASOS/METAR request endpoint
  `https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py` (`metar_iem.py:27`), public domain
  (`docs/external_data.md`).
- **Stations**: exactly the 11 challenge ICAO codes `EDDF EDDM EGLL EHAM LEBL LEMD LFPG LIRF
  LTAI LTFM LSZH` (`metar_iem.py:29-31`) — LTAI turns out not to exist in the data.
- **Station matching**: **ICAO identity**. The join key is `by_left=apt_mvt`,
  `by_right="station"` (`weather.py:47-49`). No distance/nearest-station logic — the ten
  airports are themselves ASOS stations. `apt_mvt` is the *movement* airport, and the code
  carries a warning that it is **NOT** `ADEP_mvt` (`weather.py:16-18`).
- **Time matching**: Polars `join_asof(..., strategy="backward")` on the anchor instant
  (`weather.py:43-52`). Anchor is `MVT_TIME_UTC_mvt` (take-off) for the submitted retrospective
  model and `BLOCK_TIME` for the causal variant (`weather.py:26-31`, `pipeline.py:147`).
  Timezone alignment is explicit: challenge timestamps are `datetime[us, UTC]`, IEM returns
  naive UTC, so `valid` gets `dt.replace_time_zone(tz)` — *"Same instant, two representations,
  no shift"* (`weather.py:32-38`). **No max-age cut-off**: instead `observation_age_min` is
  exposed as a feature (`weather.py:54-57`) so the model can discount stale reports. (Contrast
  Repo A, which **does** cut at 3 hours — `repoA/docs/EXTERNAL_DATA.md:37-40`.)
- **Requested raw fields** (`metar_iem.py:33`): `tmpf dwpf vsby sknt drct p01i wxcodes skyc1
  skyl1`, with `tz=UTC`, `format=comma`, `missing=empty`, `trace=empty`, `latlon=no`,
  monthly chunking and a 1 s politeness pause (`metar_iem.py:43-67, 132-138`).
  **`report_type` is deliberately omitted** so SPECI reports are kept: *"SPECI is issued exactly
  when conditions change suddenly, which makes those the most valuable observations for
  taxi-out"* (`metar_iem.py:62-65`). This is a genuinely good detail.
- **Joined variables** (`weather.py:20-23`, derived at `metar_iem.py:94-129`):
  `temperature_c`, `dewpoint_c`, `visibility_km`, `wind_ms`, `precip_mm`, `ceiling_m`,
  `freezing_precip`, `snow`, `fog`, `thunderstorm`, `deicing_proxy`, `low_visibility`;
  plus `observation_age_min` and `dewpoint_spread_c` computed at join time.
  Unit conversions: °F→°C, miles→km (×1.609344), kt→m/s (×0.514444), in→mm (×25.4),
  ft→m (×0.3048).
- **Present-weather flags** from `wxcodes` substring match (`metar_iem.py:35-40`):
  freezing `FZRA FZDZ FZFG`; snow `SN SG PL IC GS GR`; fog `FG BR`; thunder `TS`.
- **De-icing proxy** (`metar_iem.py:117-123`):
  `freezing OR snow OR (temperature_c <= 3.0 AND p01i > 0)`.
  Rationale: EUROCONTROL's own indicator *discards* flights that de-ice after off-block
  (ATXOT p.13); since the target is raw taxi-out, those rows must be modelled instead
  (`metar_iem.py:7-9`).
- Also in the weather family: EUROCONTROL's own ATMAP coefficients
  (`atmap_freezing`, `atmap_moisture`, `frost_risk`) from `features/icing.py`
  (`groups.py:85-87`) — they replaced a homemade de-icing flag (commit `a58b2ec`).

**Measured value: the weather family is worth almost nothing.** 1.8 % of gain over 14 features
(`docs/experiments.md:143`) **[X]**. And the decisive negative: the > 6 h monsters that decide
the metric sit at **20.7 °C, zero freezing, zero snow, zero de-icing proxy, below-average ATFM
delay, and are *more* common in July than January** (`docs/experiments.md`, "Are the monsters an
external-data signal? No."). **[X]** Weather does not explain the tail.

### 2.3 Repo B's one structural find — the schedule-substitution mixture

Not in the slot brief, but it is the most transferable thing in either repo. **[X]**

**The observation** (`src/taxiout/models/mixture.py:1-14`, `docs/facts.md` R40): on **4.86 %** of
2025 training rows, `TAXITIME == MVT_TIME − SCHED_TIME` to within **10 s** — i.e. the airport
feed wrote the *scheduled* off-block time into the block field. Concentration: **18.4 % at LIRF
vs 1.1 % at LSZH**, and **53.8 % at LIRF among NM-unmatched flights** where the median taxi is
7,436 s. **333 of the 435 training rows above 2 h are these rows.**

**Transfer check without touching the target** (`mixture.py:32-37`, R41): the rule can't be
tested on ranking *departures* (blanked) but can on ranking *arrivals* (not blanked). Arrival-side
substitution runs **2.31 % in Jan+Jul 2025 vs 2.42 % in Jan+Jul 2026**; LIRF 8.81 → 9.74 %,
EDDF 0.88 → 0.79 %. *"The behaviour belongs to the feed, not to the year."*

**The estimator** (`mixture.py:28-30`) — exactly what a squared metric asks for:
```
E[y|x] = P(substituted|x) · sched_offset + (1 − P(substituted|x)) · E[y | x, not substituted]
```
Two ways to get the weight:
1. `weight_mode="classifier"`: XGBClassifier (300 rounds, lr 0.08, depth 8, subsample 0.8,
   colsample 0.8, `max_bin=127`) on the ≤10 s label (`mixture.py:108-124`).
2. **`weight_mode="regression"` (shipped)**: fit the weight that *would* have been right,
   `w* = (y − rest) / (offset − rest)`, sample-weighted by `min((offset − rest)², 1e10)` — *"how
   much getting the weight wrong actually costs there"* — with `rest` computed **out of fold**
   over 3 folds so the ordinary model's opinion of its own training rows is honest
   (`mixture.py:126-188`). The **target** is clipped to `weight_clip`, the **emitted weight**
   always to `[0,1]` (`mixture.py:190-196`).

**Measured** (`docs/experiments.md`, night of 2026-09-06, XGBoost 400 rounds, 344,419-row
holdout): **[X]**

| configuration | holdout RMSE | vs plain |
|---|---:|---:|
| plain model | 474.91 | — |
| one model per airport | 484.10 | **+9.19 (loses)** |
| model per NM-match segment | 417.24 | −57.67 |
| mixture, classifier weight | 393.99 | −80.92 |
| mixture, classifier weight × 2.0 | 368.52 | −106.39 |
| **mixture, weight fitted out of fold** | **362.38** | **−112.53, no tuned parameter** |
| *oracle* told exactly which rows are substituted | 413.19 | −61.72 |

**The oracle line is the key result**: knowing the exact identity is *worse* than fitting the
weight, because *"a flight whose taxi lands within a few minutes of its schedule offset is not a
match under a ten second rule and the offset is still the better prediction for it"*
(`mixture.py:50-55`). A hard flag underperforms a calibrated soft weight. Widening the tolerance
to 30/60/120 s does not recover it (all worse at every scale).

**Board confirmation** (`docs/experiments.md`, leaderboard API **[X]**):
v12 plain XGBoost **411.23** → v19 fitted-weight mixture **334.01** (−77.2) → v21 with
`weight_clip = (−0.25, 1.25)` on the target **318.16** (−15.8) → v25 two seeds averaged
**310.55** (−5.4).

**A second identity was tried and refuted** (`src/taxiout/models/identities.py`, R46): searching
every timestamp for the same shape found `AOBT_3` matching on 8.25 % of rows — *but 0 monsters*.
Declaring it as a second mixture candidate **removes its rows from the ordinary learner's
training set**, and since those are ordinary flights the learner served fine, it was a bad trade
(>15 % worse on a synthetic population with real proportions). **Reverted.** The full identity
table (R46): SCHED 4.80 % / **430 of 584 monsters**; AOBT_3 8.25 % / 0; EOBT_1 5.83 % / 4;
IOBT 5.40 % / 14; LOBT 5.38 % / 14; ARVT_1 and ARVT_3 zero. Shift search (±30 min, 1 h, 2 h,
1 day) over 1,024,805 departures found nothing (R47).

### 2.4 The MVT_TIME zero-second probe

Prior summary **[S]**: *"MVT_TIME zero-second share about 8.3 % at EDDM/EGLL/LFPG/LIRF/LTFM vs
about 1.7 % elsewhere."* **Verified [X]** — `docs/data_probe_report_2026-09-01.md:39-51`:

| airport | n | `mvt_second_is_zero` | `block_second_is_zero` |
|---|---:|---:|---:|
| EDDF | 230,141 | 0.0177 | 0.0169 |
| **EDDM** | 167,334 | **0.0831** | 0.0834 |
| **EGLL** | 239,546 | **0.0828** | 0.0838 |
| EHAM | 247,951 | 0.0173 | 0.0169 |
| LEBL | 179,705 | 0.0166 | **0.0830** |
| LEMD | 212,242 | 0.0164 | **0.0837** |
| **LFPG** | 239,552 | **0.0832** | 0.0828 |
| **LIRF** | 160,704 | **0.0838** | 0.0835 |
| LSZH | 134,907 | 0.0168 | 0.0198 |
| **LTFM** | 272,965 | **0.0833** | 0.0830 |

Two things the repo's own reading gets only half right, worth flagging:

1. The repo concludes *"timestamps are at second precision… there is NO HH:MM problem"*
   (`docs/facts.md:151`, R07) because no share is near 1.0. That is correct as far as it goes.
2. But **uniform seconds would give 1/60 = 1.67 %**, which is exactly the low group. The high
   group at ~8.3 % is **five times** that — roughly 1/12 — which is the signature of a
   **5-second quantisation** on a sizeable fraction of rows at those five airports, not of clean
   second precision. The repo does not draw this inference, and does not act on it.
3. **`LEBL` and `LEMD` split**: MVT at 1.66/1.64 % but BLOCK at 8.30/8.37 %. The two timestamps
   at those two airports have **different precision regimes**, which means the target
   (`MVT − BLOCK`) inherits the coarser one there. This is visible in the table and is not
   commented on anywhere in Repo B. **[X]** — potentially a real, unexploited signal.

### 2.5 Repo B experiment log — full reconstruction

From `docs/experiments.md` (1,061 lines), `docs/facts.md` (248 lines, every row sourced and
dated), 90 commit messages, and cross-checked against the live board. **[X]**
Board scores below are confirmed against the leaderboard API row-for-row.

**Pre-data / ruled out before the data arrived:**

| path | outcome |
|---|---|
| OPDI ADS-B parking-position events (to recover the blanked block time) | **ruled out**: events exist at only 2 of 11 airports (LSZH, EDDF); ADS-B ground coverage at LTFM/LTAI ≈ 0 (25 and 119 runway entries in 10 days) |
| EUROCONTROL `D` de-icing reason code in arrival ATFM delay | **ruled out**: column entirely empty |
| Official EUROCONTROL taxi-out indicator as a feature | **ruled out**: monthly, ~2-month lag, July 2026 not covered, circular. Used for validation instead |

**Board log** (v-numbers are their own; scores confirmed **[X]**):

| ver | change | holdout | board | verdict |
|---|---|---:|---:|---|
| E01 | `(apt, stand, runway)` mean baseline | 628.4 | — | reference |
| v1 | LightGBM lr 0.05, 127 leaves, 800 rounds, 3 seeds, 90 feats | 378.80 | 331.23 | baseline |
| v2 | LightGBM lr 0.02, 255 leaves, 380 rounds, 5 seeds | ~372 | 331.80 | ❌ locally significant, board −0.57 |
| v3 | **XGBoost + CatBoost equal weight**, 400 rounds, 1 seed | 351.69 | **306.41** | ✅ **+24.82 s from the library alone** |
| v4 | + CatBoost depth 8→10, 1000 rounds, 2 stand features | — | **297.08** | ✅ +9.33 |
| v5 | + surface & overlap families (105 feats) | 347.74 (−10.76) | 300.51 | ❌ **largest local gain in the project, lost 3.4 on the board** |
| v6/v7/v8 | board ablation of those families | — | 317.10 / 313.35 / 310.07 | ❌ overlap costs **7.03** on the board, surface **3.75** |
| v12 | plain XGBoost on the new ranking set | — | 411.23 | baseline post-reset |
| v13/v14 | cap predictions at 7,200 s / 4,500 s | — | 488.84 / 507.28 | ❌ **+77.6 / +96.1 — extreme predictions are correct** |
| v19 | **fitted-weight schedule-substitution mixture** | — | **334.01** | ✅ **−77.2** |
| v21 | weight *target* clipped to (−0.25, 1.25) not (0, 1) | — | **318.16** | ✅ −15.8 |
| v22 | v21 averaged with a segmented-inner mixture | — | 315.94 | ✅ −2.2 (but see v26) |
| v25 | **two seeds averaged** | — | **310.55** | ✅ −5.4 — champion |
| v26 | v25 + the same partner | — | 314.71 | ❌ +4.2 |
| v28 | v25 + a *better* partner | — | 315.56 | ❌ +5.0 |
| v29 | slower rate, 800 rounds inside the mixture | 343.51 (−9.6) | 327.62 | ❌ **+17.1** |
| v30 | push extreme tail to the schedule offset (`pred > 6 h and offset > pred`) | 335.16 (−18.1) | **470.18** | ❌ **+159.6**, 13 rows locally → 29 rows on the board |
| v31 | 4 seeds instead of 2 | — | 311.75 | ❌ +1.2 |
| v32 | CatBoost inner in the mixture | 359.44 (vs XGB 357.55) | not built | ❌ tied-to-worse, + OOM on 8 GB |
| v33 | drop 17 non-substituted rows with taxi > 6 h from training | 351.27 (−2.0); **bulk 257.44 (−9.9)** | **324.22** | ❌ **+13.7 — a real bulk gain the board rejects** |
| E03a/E03b | drop training rows > 120 / > 60 min | 402.92 / 415.07 | — | ❌ **+24.1 / +36.3** |
| E02b | ATXOT-P10 residual target | 379.09 | — | ⚪ +0.29, inside noise (later retracted as inconclusive) |
| — | one model per airport | 484.10 | — | ❌ +9.19, and worse at LIRF (1377.7 → 1432.7) |
| — | day-regime family | 358.89 vs 358.11 | — | ⚪ no value, kept in code, out of submissions |
| — | second identity (`AOBT_3`) as a mixture candidate | >15 % worse (synthetic) | — | ❌ reverted |
| — | stand-occupancy upper bound | violated on 10 % of rows, 70 % of monsters | — | ❌ closed |
| — | year-on-year correction from arrival taxi-in | pooled ρ = −0.11, sign flips per airport | — | ❌ closed |
| — | raw-ADS-B trajectory recovery of the tail | — | — | ❌ **answers the wrong question**: the target is the feed's corrupted value |

**Meta-findings from Repo B that are worth more than its score:**

- **Noise floors.** Paired bootstrap on their holdout resolves ~**5 s**; a single unpaired score
  has σ ≈ **56 s**. Later, three seeds of one configuration scored 353.26 / 348.38 / 358.00 —
  a **10-second seed spread**, because the metric is dominated by a few hundred rows
  (`docs/experiments.md`, night of 2026-09-07). They then re-read several earlier conclusions as
  "not evidence either way".
- **Machine non-determinism.** The same code and the same cache scored **362.38 on Windows and
  358.11 on Hetzner** — different core count, XGBoost sums in a different order. *"Compare within
  a machine, never across one"* (R49). Together with Repo A's 3.37 s retrain wobble, this says
  **any A/B below ~5 s that crosses machines or retrains is uninterpretable**.
- **Board vs holdout.** Five documented cases where a locally significant change reversed on the
  board, including the two largest (overlap +10.76 local / −7.03 board; v30 −18.1 local /
  +159.6 board). Their rule: *"the holdout orders levers, the board decides them."*
- **Submission limit: 3 per team per day, resetting 00:00 UTC**, unpublished, found by hitting
  it (R33). **[X]** — worth knowing if we are not already budgeting for it.
- **Error concentration** after the mixture (3 seeds, 347.82): NM-unmatched rows are 1.56 % of
  rows and **51.5 %** of squared error; **24 rows above six hours carry 37.3 %**; LIRF alone is
  **64.8 %** of holdout squared error and LIRF-in-July **55.7 %** (R35). *"The competition is
  Rome in July."*
- **Long rows are under-predicted**, not over: above 6 h, median truth 45,580 s vs median
  prediction 37,970 s — 16 % short (R55). Yet v30, which pushed them up, cost 160 s. The
  fitted mixture weight is already the right answer there.
- **Data-cleaning result**: nine categorical fields, 1,899 stands, 269 aircraft types — **not one
  case or whitespace variant**. Cold start negligible: 20 unseen stands (0.106 % of ranking rows),
  3 unseen aircraft types (0.005 %), zero unseen runways or airports (DQ01/DQ02).

---

## 3. What cannot be determined from the code

- **Repo A: the isolated gain of the LIRF overlay.** Its sweep output
  (`reports/experiment_tail_overlay.md`) is git-ignored and uncommitted; no un-overlaid
  submission was ever scored. Only the row count (15 of 215,876) is published.
- **Repo A: any number attached to its 14 rejected branches.** `docs/RESULTS.md:45-51` names them
  and states they failed their gate; the measured RMSEs live in git-ignored `reports/`.
- **Repo A: V4 through V8.** The squashed commit predates them; the architecture behind the
  current 299.2862 is not in the repository.
- **Repo A: why the 2025 three-model blend was abandoned.** Verified that it *was* abandoned;
  no note, issue, commit message or doc line gives a reason beyond CatBoost failing its gate.
- **Repo A: whether `select_airport_membership.py` was ever run.** Its output `membership.json`
  is not committed and `reproduce_v3.sh` does not call it; the shipped triple is hardcoded.
- **Both: the overlay/mixture row counts on the current 344,841-row ranking set.** Both repos
  quote counts from data we would have to recompute.
- **Repo B: its board scores after 2026-09-09** — the branch stops there and the team's last
  scored submission on the API is v25/v33 (2026-09-08/09), so the repo is complete as of its
  own settling point.

---

## Applicable to `resilient-hedgehog`

### A. Side-by-side with Repo A — what they have that we lack

Ordered by my estimate of value. "Us" is read from `docs/investigation_2026-09-16/` in our own
tree; where our state is uncertain I say so rather than assert it.

| # | Repo A has | We have | Worth taking? |
|---|---|---|---|
| 1 | **A per-airport specialist layer, 50/50 with the global model at 3 airports**, gated on 3 temporal splits, each with ≥ 95 % paired-bootstrap wins (`build_specialist_submission.py:121-143`) | Single global LightGBM | **Yes, but not their membership.** Their triple (EDDF/EGLL/EHAM) was the *old* July airport set and is now meaningless. Repo B measured one-model-per-airport as **losing** (474.91 → 484.10, and worse at LIRF). The defensible version is Repo A's **mechanical selector** (`select_airport_membership.py`), not their hardcoded triple: pick membership on a disjoint month by improvement ≥ 1 s **and** bootstrap ≥ 80 % **and** largest-single-row share ≤ 50 %. |
| 2 | **A single-row-share guard**: a candidate is rejected if > 50 % of its net advantage comes from one row (`select_airport_membership.py:64, 136-138`) | Not in evidence | **Yes — cheap and high-value.** With 24 rows carrying 37 % of squared error, most of our sub-5-second gate calls are probably one-row artefacts. ~20 lines. |
| 3 | **A fail-closed gate chain**: composition must pass, then forward *and* seasonal must confirm, and `reproduce_v3.sh:78-99` literally `grep`s for "Verdict: PASS" and `exit 1`s otherwise; `check_gate()` refuses a final fit without a preserved report + its matching prediction file (`train_model.py:265-283`) | — | **Yes.** Cheap process discipline that makes "did this actually clear the bar" unfalsifiable-by-accident. |
| 4 | **Overlay re-fitted on the fit pool for the comparison arm**, so shipped 12-month coefficients cannot leak test months into the evaluation (`compare_predictions.py:9-12`) | — | **Yes** — any post-hoc rule we A/B must be re-fitted per arm or the comparison is self-scoring. |
| 5 | **A 3-hour staleness cut on METAR**, plus `wx_age_minutes` and `wx_missing` as explicit features (`repoA/docs/EXTERNAL_DATA.md:37-40`, `features.py:49-55`) | — (weather status unknown to me) | **Marginal.** Repo B measures the whole weather family at 1.8 % of gain and shows the monsters are 20.7 °C in July. Do not spend on weather. |
| 6 | **Leave-one-month-out cross-fitted target encoding** with winsorised means and parent smoothing (`target_encoding.py:1-40`) | — | **No.** It failed their gate. Recorded here so we do not rediscover it. |
| 7 | 43 features total, no external data at all | We appear to have more | Their feature set is *smaller* than ours and they score worse. Not a gap. |

**What we have that they do not**, from the same comparison:

- **We are on the current board and they are effectively not.** Their committed design targets a
  composition (Jan-all + Jul-three-airports) that ceased to exist on 2026-09-04, and the constant
  `JULY_AIRPORTS` is still in `train_model.py:46`.
- **Our "model B" / unmatched-row treatment is already ahead of their LIRF overlay.** Their rule
  fires on 15 rows at one airport. Our own notes
  (`docs/investigation_2026-09-16/hypotheses/model_b_tail.md:7`) already record that on 102 of
  119 unmatched rows with y ≥ 4 h the label is literally `MVT − SCHED` — the same identity, found
  independently, and we already have a dedicated model for it.
- **Iteration rate**: 11 scored submissions to their 5 on the current board.
- Our investigation docs quantify failures; their `docs/RESULTS.md` only names them.

### B. Concretely reimplementable, ranked

1. **Replace any hard substitution flag with a fitted soft weight.** This is the biggest
   transferable item in either repo. Repo B: hard oracle knowledge of which rows are substituted
   scores **413.19**; the fitted weight scores **362.38** — the oracle is **51 s worse**
   (`docs/experiments.md`, night of 2026-09-06). The recipe
   (`repoB/src/taxiout/models/mixture.py:126-196`), ~60 lines:
   - fit the ordinary learner on non-substituted rows only;
   - get `rest` for the training rows **out of fold** (3 folds over the ordinary rows; the
     substituted rows are already out-of-sample);
   - regression target `w* = (y − rest) / (offset − rest)`, target clipped to **(−0.25, 1.25)**
     (the (0,1) clip cost them **15.8 s on the board**), emitted weight clipped to [0,1];
   - `sample_weight = min((offset − rest)², 1e10)` so rows where the two candidates agree
     contribute nothing;
   - predict `(1 − w)·rest + w·max(offset, 0)`, weight forced to 0 where the offset is null.
   Board evidence: 411.23 → 334.01 → 318.16.
2. **Do not add a second identity candidate.** `AOBT_3` matches on 8.25 % of rows but covers
   **0 monsters**; declaring it strips 8 % of the learner's training rows for no tail benefit and
   was > 15 % worse. Refuted and reverted (`repoB/src/taxiout/models/identities.py`).
3. **Seed-average; do not blend.** Three board measurements: seed averaging transfers almost
   exactly (6 s local → 5.4 s board, one seed → two), but averaging in *any* partner **costs**
   once the base is denoised (v26 +4.2, v28 +5.0, even with a better partner). Two seeds, then
   stop — a third/fourth gave +1.2 (v31).
4. **Adopt the single-row-share guard** (Repo A, `select_airport_membership.py:64/136-138`):
   reject any candidate where one row supplies > 50 % of the net SSE advantage.
5. **Stop trusting sub-5-second differences.** Repo B: paired floor ~5 s, **seed spread 10 s**,
   **cross-machine 4 s**. Repo A: an 8-thread retrain of the identical config moves 171,075 rows
   by 3.37 s mean. Any A/B under ~10 s that is not seed-averaged and machine-matched is noise.
6. **Never clean, clip or drop the tail.** Five independent refutations: E03a +24.1, E03b +36.3,
   cap at 7,200 s **+77.6 on the board**, cap at 4,500 s +96.1, and v33 (drop just **17** rows
   with taxi > 6 h and no substitution) **+13.7 on the board despite a −9.9 s gain on the bulk**.
   The corrupted rows teach the model that the tail exists, and the board scores the corrupted
   truth.
7. **Do not push the tail up, even though it is under-predicted.** Repo B measured median truth
   45,580 s vs median prediction 37,970 s above 6 h, built the obvious correction, gated it at
   96 % local precision — and it cost **160 s on the board** (v30: 310.55 → 470.18). On 2026 the
   rows predicted above 6 h are dominated by ordinary flights with a large schedule offset that
   a fitted weight had *correctly* discounted.
8. **Consider `reference_sec` as a feature** if we do not already have an unimpeded baseline:
   P10 per `(airport, stand, runway)` fitted on training months only, validity = ≥ 10 flights at
   or below that P10, sample filtered to `0 < taxi ≤ 7200`, with the fallback chain
   `(apt,stand,rwy) → (apt,stand) → (apt,rwy) → (apt)`, and emit the **level** as a categorical
   and the **sample size** as a numeric alongside it. It is Repo B's strongest single feature at
   18.8 % of gain. **Use it as a feature, not as a residual offset** — the residual
   parameterisation measured inside the noise floor (377.84 vs 379.73) and was explicitly
   retracted as inconclusive.
9. **Check the LEBL/LEMD timestamp-precision asymmetry.** `block_second_is_zero` is 8.30/8.37 %
   at LEBL/LEMD while `mvt_second_is_zero` is 1.66/1.64 % — the two clocks have different
   precision regimes at those two airports, so the target inherits the coarser one there.
   Repo B's table shows it; Repo B does not act on it; I have not seen it in our notes.
   (`repoB/docs/data_probe_report_2026-09-01.md:39-51`.) The ~8.3 % group overall is ~5× uniform,
   consistent with 5-second quantisation rather than the "clean second precision" Repo B
   concludes.
10. **Budget submissions against the real limit**: 3 per team per day, resetting 00:00 UTC,
    unpublished (Repo B R33). And feedback lands in the bucket as
    `<name>_result.json` within ~15 s, so the board is a paired comparator on a fixed set —
    cheaper and *better-powered* than our holdout for anything above the noise floor.
