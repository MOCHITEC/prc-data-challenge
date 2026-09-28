# SLOT 06 — repo-openair: code-level read of ChandanHegde07/OpenAir

**Target:** https://github.com/ChandanHegde07/OpenAir — team **OpenAir** (submission team name in-repo is `likable-eagle`), leaderboard **276.99**.
**Clone:** `/tmp/OpenAir`, commit `97f603cb181a779e2807e358493f253ee411efc6` ("Repo cleanup: untrack catboost_info, add experiments index, order status.md", 2026-09-15). 28 commits, 2026-09-05 → 2026-09-15. 712 tracked files. License GPLv3.
**Marking:** `[S]` = read directly out of their code / logged artifacts in this repo. `[X]` = my inference, not stated by them. Every `[S]` carries a file:line.

> **Correction to the prior summary in our repo.** The prior 5-stage summary describes the pipeline as of roughly **v20 (LB 280.18)**. The repo actually runs to **E117 / v26**, production is **v24 (LB 276.9855)**, and the summary **omits the single largest cumulative gain**: the matched-branch `P+Δ` "grec" tail gate + leftover-residual chain (E48→E70, LB 288.90 → 280.18, matched RMSE 244.76 → 235.86). It also mis-states Stage 1 (the per-airport OLS has **three** regressors, not two). Details below.

---

## 0. One fact that dominates everything else

`analysis/submitting_check/tables/findings.json` → `rank_meta`:

```
ranking_rows 689534 | ranking_dep 344841 | ranking_arr 344693
unmatched 5290 | type_null 388 | lirf_unmatched 383
tmin 2026-01-01 00:01:55+00:00  tmax 2026-07-31 23:56:19+00:00
months: [{month 1, len 152719}, {month 7, len 192122}]
```

**[S] The ranking set is exactly January 2026 + July 2026.** That is why their primary holdout is Jan+Jul 2025 and why, from E108 onward, they knowingly ship changes that **lose on December** ("December is not the ranking month", `status.md:1864`; `status.md:1891`). They call Jan+Jul "the ranking analog" throughout. `status.md:98-108` also records that ranking blanks **only** `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` **for departures**; ranking ARR rows are complete including taxi-in, and `AOBT_3_flt` is present for **98.47%** of ranking DEP (the missing 1.53% = the 5,290 unmatched).

---

## 1. Pipeline architecture (what is structurally different from a single model)

Production `v24` is a **6-layer overlay stack**, each layer built by splicing rows onto the **previous submission parquet by `MVT_ID_mvt`**, not by retraining an end-to-end model. Nothing is a single fit.

```
L0  P_cal      per-airport OLS(mvt_aobt, aobt_eobt, geo_mean)            all rows
L1  E20        NNLS blend of 3 residual experts on (y - P_cal)           all rows  -> LB 316.97 (v4)
L2  LIRF D-G   T = max(D - G_hat, 0) on LIRF-unmatched only (383 rows)   -> v8 288.90, v20 280.18
L3  grec       matched-only gated blend toward P + Delta_hat             -> v11..v19, LB 287.71 -> ~280
L4  um-G blend G = 0.75*G_v20 + 0.25*G_um on the same 383 rows           -> v21 278.84, v23 278.36
L5  long-hold  OPDI icao_turn >= 7200 s -> pull 50% toward median taxi   -> v24 276.9855  (146 rows)
```

`experiments/matched_submit.py:246-273` (`pack_and_write`) is the splice primitive: read previous parquet, left-join replacements by `MVT_ID_mvt`, `np.where(new is not null, new, old)`, assert 344,841 rows, assert ID order equals `submitting.parquet`, assert no null/NaN, clip at 0. Every version-maker uses this shape.

### L0 — calibrated linear anchor `P_cal`

`experiments/run_e20_ensemble.py:111-118` and `experiments/run_e2b_e3.py:32-56`:

```python
p_ap = per_airport_ols(tr, va, ["mvt_aobt", "aobt_eobt", "geo_mean"])   # OLS per airport, min_n=200
p_cal = fill_with_fallback(fill_with_fallback(p_ap, va["geo_mean"]), airport_mean_fallback(tr, va))
```

- **[S] Three regressors**, not two: `mvt_aobt`, `aobt_eobt`, **and `geo_mean`** (`run_e20_ensemble.py:112`). The prior summary omits `geo_mean`.
- Fit rows: for each airport, rows with all three non-null, `unmatched == False` (`run_e2b_e3.py:36-37`). Airports with < 200 usable train rows fall back to the pooled global OLS (`run_e2b_e3.py:45-46`).
- Chain on missing: `p_ap` → `geo_mean` → airport mean → global mean (`common.py:232-247`). `status.md:334-336` states the rule explicitly: never dump a missing-feature row on the airport mean if a weaker predictor can score it.
- Reported OLS slope on `MVT−AOBT` ≈ 0.59 (errors-in-variables shrinkage), `status.md:118`.

### L1 — E20 residual ensemble (`experiments/run_e20_ensemble.py`)

Six experts trained on residual `y − p_cal`; NNLS blend over five of them.

| Expert | Model | Line | Jan+Jul overall / matched | Dec overall / matched |
|---|---|---|---|---|
| A | LGBM residual (E18-H baseline) | `:215` | 372.36 / 250.98 | 238.01 / 223.95 |
| A2 | A + E19-B same-runway queue cols | `:223` | 371.14 / 249.35 | 233.67 / 220.11 |
| B | LGBM **direct** `y`, `p_cal` as feature | `:231` | 394.63 / 283.08 | 237.51 / 223.62 |
| C | CatBoost residual | `:239` | 370.37 / 248.25 | 232.37 / 218.56 |
| D | XGBoost residual (numeric only) | `:249` | 385.50 / 266.24 | 246.17 / 232.70 |
| E | **per-airport** LGBM residual experts | `:259-277` | 370.52 / 247.76 | 229.94 / 215.90 |

(numbers verbatim from `experiments/results/E20/model_metrics.csv`).

**Blend (`run_e20_ensemble.py:327-361`, weights in `make_submission_e20.py:53`):**

```
NNLS over [A,B,C,D,E], non-negative, renormalised to sum 1, fitted on the Jan+Jul holdout:
A = 0.0   B = 0.0   C = 0.456   D = 0.053   E = 0.491
```

**[S]** The optimiser drops the LightGBM baseline entirely. NNLS blend: Jan+Jul 368.03 / matched 244.76; Dec 228.45 / 214.50 (`results/E20/model_metrics.csv`). Ridge(positive, α=1 and α=10) gives essentially the same answer (368.17). Expert residual correlations are **high** — A–C 0.970, A–E 0.952, C–E 0.948 (`results/E20/error_correlation.csv`): **[S]** their own read (`status.md:1064`) is that the gain comes from C and E each being individually stronger, not from decorrelated averaging.

Per-airport experts (E) fall back to expert A's prediction when the airport has `< 20,000` train rows (`run_e20_ensemble.py:265-267`); in practice all 10 airports qualify.

### L2 — LIRF gate-delay decomposition `T = D − G` (the breakthrough)

**Identity** (`status.md:1293`): `TAXITIME = D − G` with `D = MVT − SCHED` (observed on every row, ranking included) and `G = BLOCK − SCHED` (the gate hold, unobservable on unmatched rows).

**Mechanism:** train a CatBoost on **G** over *all* LIRF departures (matched, where BLOCK is known → G is a label), then apply to the LIRF **unmatched** rows and reconstruct `T = max(D − G_hat, 0)`.

Production builder `experiments/run_e74_make_v20.py`:
- Training population: all **160,704** LIRF DEP rows in `training_*.parquet` (`:57-65`).
- Target: `D − T` (i.e. G) — `:75`.
- Model: `CatBoostRegressor(iterations=1200, learning_rate=0.04, depth=8, l2_leaf_reg=5.0, loss_function="RMSE", random_seed=1, od_type="Iter", od_wait=80)` — `:73-74`.
- Applied to ranking LIRF rows; only the **383** LIRF-unmatched rows are spliced onto v19 (`:88-96`).
- `status.md:1291-1307`: why it works where the direct model (E29) failed — the target is G, the features are LIRF-specific, and the reconstruction *reuses the observed D*, which regularises the huge-D rows that dominate LIRF SSE.
- α history: v7 used `T = D − 0.60·G` (α=0.6 robust across both splits); v8+ uses α=1.0 (`status.md:1301-1318`).

### L3 — matched "grec" tail gate (the biggest gain the prior summary missed)

`experiments/submit.py` + `experiments/matched_submit.py`. Applies **only to matched ranking rows** (339,549 of 344,841), leaves unmatched at the v8/v20 value.

```python
P    = clip(MVT - AOBT, 0, None)                              # submit.py:165
dhat = LGB(Delta = y - P)                                     # submit.py:122
rec  = clip(P + dhat, 0, None)                                # submit.py:169
hat  = LGB(leftover = y - e20_pred)                           # submit.py:123
gate = (e20_pred > 1462.0) | ((rec - e20_pred) > 200.0)       # matched_submit.py:126-127
g    = e20_pred + 0.5 * (rec - e20_pred) * gate               # matched_submit.py:130-131
new  = max(g + 0.35 * max(hat, 0), 0)                         # submit.py:173-181
```

- `P90 = 1462.0`, `HTHR = 200.0` hard-coded at `matched_submit.py:32`.
- `λ_rec = 0.5`, `λ_hat = 0.35` for v19/v20 (`status.md:1701-1702`).
- **[S] The crucial trick (E49): `e20_pred` is a *feature* of the Δ model, not just the base.** `NUM = NUM0 + ["e20_pred"]` (`matched_submit.py:38`). `status.md:1530-1541`: without `e20` in Δ the gate saturates at ≈ −2 s matched; with it, matched drops 244.76 → 239.32.
- **[S] Recursive/self-referential structure.** For ranking rows, `e20_pred` is read straight out of the **previous submission parquet** (`submit.py:161-163`: `v8 = read_parquet(_v8_path()); matched.join(v8.rename({"TAXITIME_SEC_mvt": "e20_pred"}))`). For training rows it is the E20 **OOF** for Jan/Jul/Dec, and for the other nine months it is **imputed by a separate LightGBM** trained on those OOF months (`matched_submit.py:102-116`: `LGBMRegressor(n_estimators=250, lr=0.05, num_leaves=31, min_child_samples=80, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, seed=1)`).

**Exact v19/v20/v24 matched recipe** (`status.md:1701`, reproducing the CLI):
```
python experiments/submit.py --version v19 --hat-mode pos --no-operator \
       --arr-taxiin --arr-rich --lam-hat 0.35 --cap900 --seeds 3
```
- `--cap900` overrides LGB to `n_estimators=900, learning_rate=0.03, num_leaves=63, min_child_samples=40` (`submit.py:100-101`) on top of `LGB_KW = dict(n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=80, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0)` (`matched_submit.py:40-43`).
- `--seeds 3`: Δ models seeded `[1,2,3]`, leftover-hat models seeded `[101,102,103]`, predictions averaged (`submit.py:103-119`).
- **No early stopping and no eval set on these fits** — fixed 900 rounds (`submit.py:108`).
- `--hat-mode pos`: `h = max(hat, 0)` — only lift under-predictions (`submit.py:173-174`).

### L4 — soft unmatched-G blend (`run_e88c_v21_blend.py`, `run_e98_make_v23.py`, `run_e105_make_v24.py`)

Recovers `G_v20 = D − T_v20` **arithmetically from the previous parquet** (`run_e98_make_v23.py:141`: `g20 = d - t20`), then blends with a second, unmatched-only G learner:

```
G = (1 - λ) * G_v20 + λ * G_um ,   T = max(D - G, 0)
λ = 0.20 for v21, λ = 0.25 for v22/v23/v24
```

`G_um` is a LightGBM trained on the **1,488 LIRF unmatched training rows only**:
`lgb.train(dict(objective="regression", metric="rmse", learning_rate=0.05, num_leaves=31, min_data_in_leaf=10, verbosity=-1, seed=1), num_boost_round=400)` (`run_e98_make_v23.py:114-127`; v24 uses `min_data_in_leaf=8`, `run_e110_make_v24.py:50-55`).

**[S] The λ finding is the important one** (`status.md:1779-1795`): λ=1.0 (= the pure unmatched-only G of v9) **scored LB 300.95, catastrophically worse than v8's 288.90**, while λ=0.1–0.3 wins on both holdouts. Holdout LIRF-unmatched RMSE by λ: 0 → 3732.3, 0.10 → 3660.3, **0.20 → 3625.1**, 0.30 → 3627.7, 1.00 → 4581.9.

### L5 — long-hold taxi pull (`run_e110_make_v24.py`) — the final +1.37 s

```python
THR = 7200.0 ; ALPHA = 0.5
typ  = median(T of training LIRF-unmatched rows with icao_turn >= 7200)      # :47-48
T0   = max(D - (0.75*G_v20 + 0.25*G_um), 0)                                  # :78
if icao_turn >= 7200:  T = 0.5*T0 + 0.5*typ                                  # :80
```
`typ` = **1087 s** on the Jan+Jul fit and 1132 s on Dec (`results/E113/E113_results.json`). **146** ranking rows fire (`status.md:1862`). Physics stated at `status.md:1857-1859`: on LIRF unmatched with ≥ 2 h on the ground, median true T is 1132 s (a *normal* taxi) while median G is 4679 s — so a long ground turn means the delay was at the gate, and `T ≈ typical taxi`, not `T ≈ D`. **December fails this rule** (winter long-turn rows still have `T ≈ D`); they shipped it anyway because ranking is Jan+Jul.

---

## 2. Complete feature inventory with exact definitions

### 2.1 Base clock features — `experiments/common.py:84-133` (`load_dep`)

All are `(a − b).dt.total_seconds()` via `sec()` (`common.py:80-81`). Nulls propagate as NaN; the trees consume NaN natively, the OLS drops non-finite rows.

| Feature | Definition | Line |
|---|---|---|
| `y` | `TAXITIME_SEC_mvt` cast f64 (= `MVT_TIME − BLOCK_TIME`, verified 100% exact, `status.md:87`) | `:90` |
| `mvt_aobt` | `MVT_TIME_UTC_mvt − AOBT_3_flt` | `:91` |
| `mvt_eobt` / `mvt_iobt` / `mvt_lobt` / `mvt_sched` | `MVT_TIME − {EOBT_1, IOBT, LOBT, SCHED_TIME}` | `:92-95` |
| `aobt_eobt` / `aobt_iobt` / `aobt_lobt` / `aobt_sched` | `AOBT_3 − {EOBT_1, IOBT, LOBT, SCHED}` | `:96-99` |
| `eobt_iobt`, `eobt_lobt`, `iobt_lobt` | pairwise clock diffs | `:100-102` |
| `unmatched` | `AOBT_3_flt.is_null()` | `:103` |
| `month`, `hour`, `dow` | from `MVT_TIME_UTC_mvt` (`dt.weekday()`) | `:104-106` |
| `sched_hour` | hour of `SCHED_TIME_UTC_mvt` | `:107` |
| `ac_family` | `AIRCRAFT_TYPE_mvt[0:3]` | `:108` |
| `clock_std` | `nanstd` over epoch-seconds of `{AOBT_3, EOBT_1, IOBT, LOBT}`, ddof=0, non-finite → NaN | `:113-126` |
| `clock_range` | `nanmax − nanmin` of the same four | `:120` |
| `n_clocks` | count of the four that are finite | `:121` |
| `abs_mvt_aobt`, `abs_aobt_eobt`, `abs_aobt_iobt`, `abs_aobt_lobt` | absolute values | `:128-131` |
| `type_null` | `AIRCRAFT_TYPE_mvt.is_null()` — **≈ LIRF unmatched** (1,487/1,488, `status.md:74`) | `run_e12_e9.py:102` |
| `flt_prefix` / `p3` | `FLIGHT_mvt[0:3]` after `fill_null("")` — airline prefix | `run_e74_make_v20.py:43`, `run_e88c_v21_blend.py:69` |

Frame is sorted `["airport", "MVT_TIME_UTC_mvt"]` before any rolling (`common.py:133`).

### 2.2 Geometry hierarchy `geo_mean` — `experiments/run_e2b_e3.py:59-117`

Group aggregates (`n, mean, med, std, p25, p75, p90` of `y`) at four levels — `(airport, STAND, RUNWAY)`, `(airport, STAND)`, `(airport, RUNWAY)`, `(airport)` — fitted **on the training split only** and, after E18-H, **on matched train rows only** (`run_e18_unmatched_specialist.py:196`: `src = tr0.filter(~pl.col("unmatched")) if matched_geo else tr0`).

```
geo_mean = sr_mean   if sr_n >= 30 and sr_mean not null
         elif st_mean if st_n >= 30
         elif rw_mean if rw_n >= 30
         else ap_mean
```
`min_n = 30` is passed at every call site (`run_e20_ensemble.py` via `prepare_split`, `make_submission_e20.py:108-109`). `geo_med`, `geo_p90`, `geo_std`, `sr_n_used`, `sr_minus_stand`, `sr_minus_rwy`, `sr_minus_ap`, `aobt_minus_geo` are derived the same way (`:101-117`) — **[S]** the percentile/delta variants were tested and rejected as collinear in the linear core (`status.md:820`).

### 2.3 Causal rolling of `MVT−AOBT` — `common.py:136-158` (`add_causal_rolling`)

Sorted by `(airport, MVT_TIME)`, **always `.shift(1)` before the rolling window** so the current row is excluded. Row-count windows, not time windows.

`lag1_mvt_aobt`; `roll{5,10,20}_mean_mvt_aobt`; `roll10_med_mvt_aobt`; `roll10_std_mvt_aobt`; `roll10_p75_mvt_aobt` / `roll10_p90_mvt_aobt` (`rolling_quantile(..., interpolation="nearest")`); and the same over `(airport, RUNWAY_mvt)`: `lag1_rwy_mvt_aobt`, `roll10_rwy_mean_mvt_aobt`, `roll10_rwy_med_mvt_aobt`.
Only `roll10_mean_mvt_aobt` enters the production model (`run_e12_e9.py:31`). Nulls stay null (no imputation) — `status.md:813` records that filling missing roll10 with the airport mean degrades matched RMSE 379 → 398.

### 2.4 Disruption state (E16-A) — `experiments/run_e16a.py:185-283`

This is the feature the prior summary flagged, and the implementation detail matters.

`_window_from_prefix` (`:185-219`):
1. Take only **matched** rows with finite `aobt_eobt` as the "push" stream (`:235`).
2. Sort those by `AOBT_3_flt`; build prefix sums of `aobt_eobt`, of counts, and of the indicators `>600`, `>1200`, `>1800`.
3. For scored row *i*, window is `AOBT_j ∈ (MVT_i − w, MVT_i]` — **the window ends at the scored flight's takeoff time, and members are selected by *their* push time** (`hi = searchsorted(push_t, query_t, "right")`, `lo = searchsorted(push_t, query_t - w, "right")`).
4. **Self-exclusion is explicit arithmetic** (`:202-213`): if the scored row's own `AOBT` falls in the window, subtract its own value from the sum and 1 from each count, then `n = max(n, 0)`.
5. `mean = s / n` only where `n >= 1`, else **NaN** (`:215-218`). No zero-fill.

Windows: 15, 30, 60 min (`:240`). Emitted columns (`:264-282`): `dis_n_{15,30,60}m`, `dis_state_{15,30,60}m`, `dis_frac10_30m`, `dis_frac20_{15,30,60}m`, `dis_frac30_{30,60}m`, `dis_n_dly20_{30,60}m`.

- `dis_state_30m` = **mean `AOBT−EOBT` of other flights that pushed in the 30 min before this flight's takeoff, self-excluded**.
- `dis_frac20_30m` = fraction of those others with `AOBT−EOBT > 1200 s`.
- `dis_frac30_30m` = fraction with `> 1800 s`.
- `dis_state_vs_hour` = `dis_state_30m − mean(dis_state_30m)` over the **train-split** `airport × hour` climatology (`run_e16a.py:366-374`). Difference, not ratio, because `AOBT−EOBT` can be ≤ 0.
- `arr_delay_mean_30m` = mean of `MVT_TIME − SCHED_TIME` over ARR rows already landed within 30 min before the scored MVT (`run_e16a.py:330+`, ARR frame built at `:312-328`).
- `dis_med_ae_30m`, `dis_p75_ae_30m`, `dis_p90_ae_30m` via `rolling(period="30m")` + `join_asof(strategy="backward")` — **[S] their own docstring admits "one-point self contamination is possible"** (`run_e16a.py:287-289`); these three are **not** in the production feature list.

Measured effect (`status.md:200-206`): corr with own `aobt_eobt` = 0.45; corr with the frozen residual = 0.11 (own `aobt_eobt` vs residual only 0.02). Inside every own-lateness quartile, high-vs-low disruption still raises the `>30 min` rate by 8.1× / 1.6× / 1.6× / 1.4×. Jan+Jul matched 256.46 → 253.82; Dec 230.04 → 226.95.

### 2.5 Traffic / queue (kept as features, individually worthless) — `experiments/run_e4_e5.py`

`add_traffic` (`:41-84`): for w ∈ {5,15,30,60} min, `dep_{w}m` = count of DEP rows with `MVT_TIME` in the trailing w-minute window **minus 1** (self); `dep_rwy_{w}m` same within `(airport, RUNWAY)`; `arr_{w}m` via cumulative-count `join_asof` differences on the ARR stream; plus `mov_15m`, `mov_60m`, `dep_ratio_15_60`, `dep_accel = dep_5m − dep_15m/3`, `arr_dep_ratio_15`, `dep_5_15`.

`add_queue` (`:86-131`): "queue at takeoff" by interval counting — `queue = max(#{AOBT_j <= MVT_i} − #{MVT_j < MVT_i} − 1, 0)` over matched rows of that airport; `queue_rwy` the same within runway; `push5`, `push15` = pushes in the trailing 5/15 min. Uses **only AOBT and MVT**, both present in ranking, so it is computable on ranking — but it *does* count later takeoffs to decide "still taxiing", which is batch/post-hoc information (`status.md:857-859` flags it themselves as "batch-only").

**[S]** `status.md:145-152`: adding linear traffic or queue on top of clocks + geometry moves Jan+Jul matched RMSE 303.53 → 303.64 / 303.56 — i.e. **zero**. They kept them in the tree feature matrix anyway but the raw correlations (queue 0.264, rwy-dep-60m 0.167) turned out to be proxies for the AOBT duration clock and geometry.

### 2.6 Surface-event features for the LIRF G model — `run_e33_delay_decomposition.py:42-88` (`add_surface`)

Per airport, over the DEP takeoff-time array `t` and the ARR takeoff/landing array `at`:

| Column | Definition |
|---|---|
| `sdep_{5,15,30}` | # DEP takeoffs in `[t − w min, t)` (`_cnt`, side="left") |
| `sarr_{5,15,30}` | # ARR movements in the same window |
| `srdep_{10,30}` | same-runway DEP count |
| `srarr_{10,30}` | same-runway ARR count |
| `sdep_ts` | seconds since previous DEP at the airport (`_since`) |
| `sarr_ts` | seconds since previous ARR |
| `sev_ts` | seconds since previous event of either type (merged sorted stream) |
| `sst_ts` | seconds since previous DEP **from the same stand** |

`_cnt` uses `side="left"` on both bounds, so the scored row itself is excluded.

`run_e34d_enriched.py:23,30-39` adds `EXTRA`:
- `d_log = log1p(max(D, 0))`
- `d_rank = searchsorted(sort(D_train_LIRF), D, "left") / len(D_train_LIRF)` — **percentile of D against the training LIRF D distribution** (`dref` is always the training array, even when scoring ranking: `run_e74_make_v20.py:64,84`)
- `d_x_arr15 = D * sarr_15`, `d_x_dep15 = D * sdep_15`
- `arr_minus_dep_15 = sarr_15 − sdep_15`, `ev_rate_15 = (sarr_15 + sdep_15)/15`
- `hour_sin`, `hour_cos` = `sin/cos(2π·hour/24)`

### 2.7 Inbound / stand-turnaround features (E53–E55, E73–E74) — `matched_submit.py:134-238`, `run_e74_make_v20.py:29-54`

Built by `join_asof(strategy="backward")` on the **ARR** stream keyed `["airport", "STAND_mvt"]`, left key = the DEP's `MVT_TIME_UTC_mvt`, right key = the ARR's `BLOCK_TIME_UTC_mvt` (in-block, i.e. `aibt`):

| Feature | Definition | Window filter |
|---|---|---|
| `since_arr` | `MVT_TIME − aibt` of the last ARR at that stand | kept only if `0 < since_arr < 8·3600` (matched branch, `:173-181`); `< 36·3600` in the LIRF G branch (`run_e74_make_v20.py:51`) |
| `arr_taxiin` | that ARR's `TAXITIME_SEC_mvt` (taxi-**in**) | same window |
| `arr_taxiin_2` | the *previous* ARR's taxi-in at the same stand, via `shift(1).over(["airport","STAND_mvt"])` | same window |
| `arr_delay` | `aibt − SCHED_TIME` of that ARR | same window |
| `arr_type` | that ARR's `AIRCRAFT_TYPE_mvt` (categorical) | same window |
| `arr_taxiin_aobt` / `since_arr_aobt` | the same asof-join but anchored at the DEP's **AOBT** instead of MVT | `0 < · < 8·3600` |
| `since_dep` | seconds since the previous DEP from the same stand | kept if `1 < since_dep < 12·3600` |
| `rwy_arr_taxiin` | last ARR taxi-in on the **same runway** (asof on `["airport","RUNWAY_mvt"]`) | no window filter |

Out-of-window values are set to `None`, not 0 (`:173-181`). **[S] These are ranking-safe because ranking ARR rows keep `TAXITIME_SEC_mvt` and `BLOCK_TIME_UTC_mvt`** — only DEP rows are blanked (`matched_submit.py:135` docstring; `status.md:100-104`). At apply time the ARR frame is rebuilt **from `ranking.parquet` itself** (`submit.py:144-157`).

### 2.8 External OPDI features — `run_e77_icao_turn.py:42-83`, `run_e98_make_v23.py:35-49`, `run_e105_make_v24.py:38-70`

| Feature | Definition |
|---|---|
| `icao24` | OPDI hex ID, obtained by `join_asof(strategy="nearest")` on normalised callsign `cs`, then **kept only if `|MVT − first_seen| < 12 h` **and** `OPDI.adep == airport`** (`run_e77_icao_turn.py:52-59`) |
| `icao_turn` | `first_seen(current leg) − last_seen(previous leg of the same icao24)`, where "previous" is `shift(1).over("icao24")` on legs sorted by `first_seen`. **Kept only if `1200 < icao_turn < 36·3600`**, else null (`:78-83`). This is the aircraft's ground time. |
| `icao_same_apt` | `prev_ades == airport` cast Int8 |
| `typecode`, `icao_operator`, `icao_aircraft_class` | first non-null per `icao24` across all OPDI rows (`run_e105_make_v24.py:59-70`) |

Coverage (`status.md:1823-1831`, `1855-1862`): train unmatched 43%; **ranking unmatched 219/383 (57%)**; of those, **146** have `icao_turn ≥ 7200`.

### 2.9 Production feature matrices — exact lists

**E20 experts A/C/D/E** — `run_e20_ensemble.py:71` (`NUM_FEATS = NUM_COLS + MODEL_DISRUPT_COLS`):
```
NUM (run_e12_e9.py:26-43):
  mvt_aobt, aobt_eobt, mvt_sched, geo_mean, roll10_mean_mvt_aobt,
  dep_15m, dep_rwy_15m, dep_60m, arr_15m, queue, queue_rwy,
  hour, dow, month, unmatched, type_null
+ DISRUPT (run_e16a.py:75-82):
  dis_state_30m, dis_frac20_30m, dis_frac30_30m, dis_n_dly20_60m,
  dis_state_vs_hour, arr_delay_mean_30m
CAT (run_e12_e9.py:44-53):
  airport, RUNWAY_mvt, STAND_mvt, AIRCRAFT_TYPE_mvt,
  WK_TBL_CAT_flt, MARKET_SEGMENT_flt, AIRCRAFT_OPERATOR_flt, ADES_mvt
```
Expert D (XGB) gets **numeric only** (`run_e20_ensemble.py:141-142`). Expert B gets the same plus `p_cal` as a feature, target = raw `y` (`:230-232`).

**Matched Δ / leftover-hat models (v19/v20/v24)** — `matched_submit.py:33-39` + `submit.py:77-84`:
```
NUM: mvt_sched, mvt_aobt, aobt_eobt, sd_eobt, sd_lobt, sd_iobt, mvt_lobt, mvt_iobt,
     hour, dow, month, clock_std, clock_range, n_clocks, abs_aobt_eobt, e20_pred
   + arr_taxiin, since_arr, arr_delay, since_dep, arr_taxiin_aobt, since_arr_aobt,
     arr_taxiin_2, rwy_arr_taxiin
CAT: airport, RUNWAY_mvt, STAND_mvt, AIRCRAFT_TYPE_mvt, ADES_mvt,
     MARKET_SEGMENT_flt, ac_family, arr_type
     (AIRCRAFT_OPERATOR_flt is EXCLUDED in v19+: --no-operator)
```
(`sd_eobt` is a duplicate of `aobt_eobt`, `matched_submit.py:63-64` — harmless but present.)
Categoricals are `fillna("NA").astype("category")` (`submit.py:68-70`).

**LIRF all-LIRF G (CatBoost, v20)** — `run_e74_make_v20.py:66-67`:
```
NUM = [mvt_sched, hour, dow, month] + SURF(14) + EXTRA(8) + [arr_taxiin, arr_taxiin_2, since_arr, arr_delay]
CAT = [airport, RUNWAY_mvt, STAND_mvt, AIRCRAFT_TYPE_mvt, ADES_mvt, flt_prefix, arr_type]
```
Categoricals `fillna("NA").astype(str)` (`:70`).

**Unmatched-only G (LightGBM, v21–v24)** — `run_e88c_v21_blend.py:29-30` + `run_e105_make_v24.py:27-28`:
```
NUM = [D, hour, month, dow, icao_turn, icao_same_apt]
CAT = [RUNWAY_mvt, STAND_mvt, AIRCRAFT_TYPE_mvt, ADES_mvt, p3, seg,
       typecode, icao_operator, icao_aircraft_class]
```
**[S]** `seg` (= `MARKET_SEGMENT_flt`) is **NA on every unmatched row** because there is no NM join at all (`status.md:1793`) — they note it is carried but useless. That is exactly why the OPDI `typecode`/`icao_operator` cats were worth adding.

---

## 3. Exact hyperparameters (every model)

| Model | Where | Config |
|---|---|---|
| E20 expert A / A2 / B / E (LightGBM) | `run_e12_e9.py:67-87` | `LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=80, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, n_jobs=-1)`, `early_stopping(40)` on the ES fold, `categorical_feature=CAT_COLS`. Seeds: A/A2/E `SEED=1`, B `SEED+1=2` |
| E20 expert C (CatBoost) | `run_e20_ensemble.py:239-243` | `CatBoostRegressor(iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=3.0, loss_function="RMSE", random_seed=1, thread_count=-1)`, `early_stopping_rounds=60` |
| E20 expert D (XGBoost) | `run_e20_ensemble.py:249-254` | `{objective: reg:squarederror, eta: 0.05, max_depth: 7, subsample: 0.8, colsample_bytree: 0.8, min_child_weight: 80, lambda: 1.0, seed: 1, tree_method: hist}`, `num_boost_round=2000`, `early_stopping_rounds=60` |
| Matched Δ / hat base | `matched_submit.py:40-43` | `n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=80, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0` |
| Matched Δ / hat **production (cap900)** | `submit.py:101` | `n_estimators=900, learning_rate=0.03, num_leaves=63, min_child_samples=40` (rest as above). Δ seeds `[1,2,3]`, hat seeds `[101,102,103]`, **no early stopping** |
| `e20_pred` imputer for non-holdout months | `matched_submit.py:112-113` | `n_estimators=250, learning_rate=0.05, num_leaves=31, min_child_samples=80, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, seed=1` |
| LIRF all-LIRF G (CatBoost, v20/v24) | `run_e74_make_v20.py:73-74` | `iterations=1200, learning_rate=0.04, depth=8, l2_leaf_reg=5.0, RMSE, random_seed=1, od_type="Iter", od_wait=80` |
| Unmatched-only G (LightGBM, v23) | `run_e98_make_v23.py:115-127` | `lgb.train({objective: regression, metric: rmse, learning_rate: 0.05, num_leaves: 31, min_data_in_leaf: 10, seed: 1}, num_boost_round=400)` |
| Unmatched-only G (v24) | `run_e110_make_v24.py:50-55` | same but `min_data_in_leaf=8` |
| Unmatched-only G (CatBoost alt, not shipped) | `run_e88c_v21_blend.py:110-123` | `iterations=400, learning_rate=0.05, depth=6, l2_leaf_reg=5.0, od_type="Iter", od_wait=40` |
| E68 capacity probes (not shipped) | `status.md:1705` | deep `2000 × 127 leaves` → 235.79/209.75; wide `1500 × 255` → 235.83/209.29 |

**Blend / overlay constants:**
- E20 NNLS: `C=0.456, D=0.053, E=0.491` (A=B=0) — `make_submission_e20.py:53`
- grec: `λ_rec = 0.5`, gate `e20 > 1462.0 OR (rec−e20) > 200.0` — `matched_submit.py:32,126-131`
- leftover: `λ_hat = 0.35`, `pos` mode — `submit.py:173-181`, `status.md:1701`
- um-G blend: `λ = 0.20` (v21) / `0.25` (v22–v24) — `run_e88c_v21_blend.py:28`, `run_e98_make_v23.py:23`
- long-hold: `THR = 7200 s`, `ALPHA = 0.5`, `typ = 1087` (Jan+Jul) — `run_e110_make_v24.py:31-33`
- v7 gate-delay α = 0.6; v8+ α = 1.0 — `status.md:1301`, `run_e34d_enriched.py:67`

**[S] `status.md:1437` is blunt about tuning: an E44 audit found E20 used fixed HPs, one seed, no HP search, no bagging.** A deep/shallow/regularised LGB sweep plus 3-seed averaging plus NNLS bought **−0.43 s** matched on Jan+Jul and **0.00** on Dec. They explicitly concluded "no headroom" from tuning.

---

## 4. Validation / CV setup

**[S] There is no k-fold CV anywhere.** The protocol (`status.md:322-337`, `common.py:161-172`) is two fixed temporal holdouts over the 12 monthly `training_*.parquet` files (4,167,797 rows total, 2,085,047 DEP):

- **Primary ("ranking analogue"):** train = all 2025 months **except January and July**; validate = **Jan + Jul 2025**. `SPLITS["janjul"] = [1, 7]`.
- **Sanity/transfer:** train = Jan–Nov, validate = **December**. `SPLITS["dec"] = [12]`.
- Also `jan` and `jul` alone in E0.

Inside the training half, an **early-stopping fold is carved chronologically**: `time_es_split(df, frac=0.15)` sorts by `MVT_TIME_UTC_mvt` and takes the last 15% as the ES set (`run_e12_e9.py:90-94`). The remaining 85% is the fit set.

The NNLS blend weights are fitted on the **Jan+Jul holdout itself** (`run_e20_ensemble.py:343`), with December used as the honest transfer check — they acknowledge this in the module docstring (`:12-14`).

Acceptance rule (`run_e20_ensemble.py:371-388`): `ACCEPT` if Jan+Jul improvement ≥ 1.5 s **and** December regression ≤ 3.0 s; `WEAK` if ≥ 0.5 s and ≤ 5.0 s; else `REJECT`. **[S] From E108 onwards they deliberately break their own rule** and ship Jan+Jul-only wins, because the ranking set is Jan+Jul 2026.

Metrics reported everywhere (`common.py:199-229`, `run_e20_ensemble.py:83-101`): overall / matched / unmatched / **LIRF-unmatched separately** / non-LIRF-unmatched / `y>1800` / `y>3600` / per-airport, plus raw SSE. `status.md:337` calls out the "reporting trap": overall RMSE is dominated by unmatched, so matched RMSE must always be published alongside.

**Internal↔LB calibration (`status.md:1241`, `1257`):** internal Jan+Jul 368.03 ↔ LB 316.9654, so they carry a **~51 s constant offset** and use `LB ≈ internal − 51` to forecast every candidate. Later (`status.md:1713`) they observe the LB tracks **matched RMSE roughly 1:1**.

`experiments/common.py:16-36` enforces data hygiene: `TRAIN_FILES` is globbed from `training_*.parquet` and `_assert_training_path` **raises** on any parquet not starting with `training_`, or containing `ranking`/`submitting`/`submission`/`test`/`eval`. `experiments/test_e25_leakage.py` exists as a standing leakage test.

---

## 5. Full experiment sequence with scores (reconstructed from `status.md` + commit order)

### 5.1 Leaderboard ladder (every scored submission)

| Ver | Experiment | LB RMSE | Verdict | `status.md` |
|---|---|---:|---|---|
| v4 | E20 NNLS ensemble | **316.9654** | KEEP | `:1068` |
| v7 | E33 LIRF `T = D − 0.6·G` | **292.9926** | KEEP | `:1313` |
| v8 | E34 enriched CatBoost G, α=1.0 | **288.9003** | KEEP | `:1327` |
| v9 | E35 unmatched-only G | **300.9543** | **REJECT** (−12 s worse) | `:1499` |
| v10 | E47 OPDI-in + stand-release G | **296.321** | REJECT | `:1503` |
| v11 | E48 matched tail-gated `P+Δ` | **287.7101** | KEEP | `:1524` |
| v12 | E49 grec λ=0.5 with `e20` in Δ | **284.9731** | KEEP | `:1539` |
| v13 | E50 rich grec + leftover hat λ=0.25 | **284.0967** | KEEP | `:1555` |
| v14 | E51 L2 + 0.3·q65 Δ mix | **284.1341** | REJECT | `:1565` |
| v15 | E52 positive-only leftover + operator | **284.3732** | REJECT | `:1573` |
| v16 | E53 previous same-stand ARR taxi-in | **282.0851** | KEEP | `:1583` |
| v17 | E54 richer inbound | **281.8799** | KEEP | `:1591` |
| v18 | E55 2nd taxi-in + arr_type + rwy ARR | **281.7196** | KEEP | `:1599` |
| v19 | E69/E70 cap900 + seed-avg + λ=0.35 | not scored alone | folded into v20 | `:1701` |
| v20 | E73/E74 inbound features in LIRF G | **280.1831** | KEEP | `:1742` |
| v21 | E88 soft um-G blend λ=0.2 | **278.8398** | KEEP | `:1795` |
| v22 | E93 `pfx_um` cat + λ=0.25 | **not recorded** | submitted | `:1805-1815` |
| v23 | E98 OPDI `icao_turn` in um-G | **278.3641** | KEEP | `:1833` |
| v24 | E105+E110 OPDI operator/type + long-hold pull | **276.9855** | **PRODUCTION** | `:1896` |
| v25 | E112 large-G prefix taxi pull | **277.3456** | REJECT | `:1896` |
| v26 | E109 matched icao-Δ 25% blend | **279.5844** | REJECT (+2.60) | `:1880` |

### 5.2 Internal progression before any submission

`status.md:341-365` (experiment table) and `:878-905`:

| Stage | Jan+Jul overall | Jan+Jul matched |
|---|---:|---:|
| Airport mean | 660 | 440 |
| E0 AOBT proxy OLS | 632 | 396 |
| E1 + `aobt_eobt` (the second clock) | 603 | **348** |
| E2b per-airport two-clock OLS | 586 | 316 |
| E3 + `geo_mean` | 577 | 304 |
| E11 + always-on LIRF `MVT−SCHED` | **410** | 304 |
| E12 LGB direct | 468 | 288 |
| E9 LGB **residual** on `y − P_cal` | 378.28 | 256.46 |
| E16-A + disruption state | 376.04 | 253.82 |
| E18-H hygiene | 372.36 | 250.98 |
| E20 NNLS ensemble | **368.03** | **244.76** |
| E49 v12 grec | 364.50 | 239.32 |
| E50 v13 | 363.98 | 238.52 |
| E55 v18 | 362.74 | 236.59 |
| E70 v19 | **362.27** | **235.86** |

### 5.3 Rejected / falsified experiments, in order (the negative result set)

**Linear phase (E0–E13).**
- E1: raw `MVT−EOBT`/`MVT−LOBT`/`MVT−SCHED` as taxi substitutes — worse than the mean (`:811`).
- E2: airport *additive* bias — 409 vs 396 for global OLS. Calibration needs slope (`:812`).
- E3: geometry percentiles / stand−runway deltas — collinear with `geo_mean` (`:816`).
- E4/E5: **linear traffic and queue after clocks+geometry — Δ RMSE ≈ 0** (303.53 → 303.64 / 303.56) (`:147-152`).
- E7: `FLIGHT_ID` is a **city-pair NM flight, not a tail number**; 315,262 of 315,272 duplicate IDs are the DEP+ARR of the same city-pair, median 84 min apart. Same-FLIGHT_ID ARR is *after* the DEP → leakage (`:172-177`).
- E11b: clipping `MVT−SCHED` to 4 h for LIRF unmatched — worse than unclipped (`:817`).
- E13: **gating** the LIRF `MVT−SCHED` override on any threshold / stand / hour / dest / prefix — the two y-regimes exist but `MVT−SCHED ≈ 6000 s` in *both*, so no gate generalises (`:283-289`).

**Loss-function phase (E15, E24).**
- E15-A Huber (δ=228 s): `<20 min` RMSE 174.5 → 162.9 but `>30 min` 821 → 933, matched 256 → 268.
- E15-B tail-weighted L2: `>30` 821 → 654, `>60` 2376 → 1946, but bulk destroyed (`<20` 175 → 269, matched 300).
- E15-C two-stage `P(y>30)` + tail expert: matched 318, gate fires at 14% vs a true 4.5% rate.
- E24 quantile/pinball (α=0.1/0.5/0.9): Q50 overall 381 vs L2 372; Q90 fixes the tail and wrecks the bulk; **NNLS gives Q50 weight exactly 0** next to the E20 experts.
- **Their conclusion (`:181-196`): KEEP plain L2, and "do not run another residual-objective experiment."**

**Rolling-state phase (E14, E17-A, E19, E32, E41, E52, E62–E64) — the falsified family.**
- E14 (unsafe): 31 dynamic-surface features including rolling *TAXITIME* stats → 371.57 / 248.05. **Rejected for transfer: consumes other DEPs' TAXITIME, which is blanked on ranking** (`:967`).
- E17-A ranking-safe rebuild of the same idea (35 memory features on `MVT−AOBT` / `AOBT−EOBT` / `MVT−SCHED`): A1 airport memory −0.97 s Jan+Jul; A2 + runway memory **hurts December (+1.55)**. E14's level not recovered (gap +3.31). INCONCLUSIVE.
- **E19 (the "rolling queue" result the prior summary cites):** six families (neighbour state, same-runway queue, delay shock vs `airport×hour` / `airport×runway` climatology, queue/arrival pressure, interactions, tail products). Best variant −1.34 s overall, −1.76 matched. **Top-1% SSE share went from 72.25% to 72.73% — i.e. the tail got marginally *worse*** (`:1005-1031`). Verdict: "do not keep adding rolling-statistic features."
- E41 exact interval-overlap surface occupancy at `t = AOBT`: standalone matched 309.41 vs E20 244.76; optimal blend weight **0.0** on Jan+Jul.
- E52 congestion into v13's Δ: 238.50 vs v13 238.52 — **exactly redundant**.
- E62 runway service-rate/headway decomposition: −2.17 s **vs E20**, but v13 (238.5) already beats E20+E62 (242.6). E63 (continuous workload) −0.21, E64 (service process) −0.38 — both below noise once v13 is the baseline. **[S] Important methodological point: E62's apparent win was an artefact of benchmarking against the wrong baseline.**

**Structural / identity phase (E42, E57, E58, E61, E71, E77).**
- E42 flight-identity residual prior (empirical-Bayes shrinkage `n/(n+λ)·mean_resid` over CALLSIGN / ×airport / ×ADES / ×weekday / ×hour keys, λ and α auto-selected on an internal temporal holdout): selected `CALLSIGN×weekday×hour-bucket, λ=1, α=1` → matched 244.76 → **249.64 (worse)**, Dec 214.78 → **223.58 (worse)**.
- E57 latent airport regime (KMeans K=5 on causal 5-min-bin state vectors, silhouette 0.35): states persist 6–11 h but taxi medians are 829–952 s across states 0–3; state-median baseline matched 482.8 vs v13 238.5. REJECT.
- E58 forensics: stand-linked prior arrival covers 411k matched rows (80.6%) but `corr(arrival-taxi, departure-taxi) = 0.123` and `corr(arrival-taxi, E20-resid) = −0.008`. **"No second E33-style structural asymmetry exists in the supplied dataset."**
- E61 historical analogue engine (hierarchical exact-key retrieval, EB-shrunk, confidence gated): ~99–100% coverage but the analogue residual prior is ≈ 0; all corrections neutral (Δ 0.00–0.05).
- **E71c previous-departure-BLOCK oracle** (the "corr ≈ 0.008" in the prior summary): correct self-excluded previous DEP at the same stand gives `corr(T, gap_block) = 0.008` unmatched / 0.043 LIRF-u; `corr(T, prev_Taxi)` 0.13 / 0.28. **The oracle itself has no signal**, which falsified attacks 2–6 before any model was fit.
- E72 stand-release hard bound `T ≤ since_arr`: LIRF-u 3896 → 11579. Rejected — the latest ARR at a stand is usually a *different* aircraft.

**Unmatched phase (E18, E19-A/B/C, E20-prior, E21, E43, E87).**
- E18-A/B: non-LIRF unmatched airport mean/median, and a dedicated unmatched-only tree — both **fail December** even when Jan+Jul improves.
- **E19-A SSE map:** after E18-H, unmatched = 55.3% of Jan+Jul SSE; LIRF-unmatched 30.3%; **LFPG unmatched 21.8% — but 97% of that is TWO easyJet rows** (`EJU983W` y=84,240 with `MVT−SCHED`=1,740; `EJU42AY` y=58,206 with `MVT−SCHED`=2,043), i.e. wrong-day BLOCK labels.
- E19-B: a 24 h wrap on those bombs is an **oracle, not a gate** — the val-perfect rule `LFPG ∩ EJU ∩ ms<2400` fires on 2 December false positives, every train-selectable gate has precision 0. Oracle would give 372.36 → 334.39.
- **E19-C / "unmatched LFPG nearly random" (prior summary):** `corr(y, MVT−SCHED)` on LFPG unmatched = **−0.02** (full year +0.02) vs **0.90–0.92 at LIRF**. E71b generalises: `corr(T, D)` is 0.90 at LIRF and ≤ 0.44 everywhere else (EDDF 0.17, LFPG 0.02). **That single correlation is why `D − G` is a LIRF-only technique.**
- E21: every calibration of the LIRF override (mean shift, winsorised mean, OLS, median, hour/dow medians) — −14 s Jan+Jul at best, December worse. Keep raw `MVT−SCHED`.
- E43: LFPG `D − G` with the E34 CatBoost — Jan+Jul 3440.8 → 3487.3 (worse), Dec 743.4 → 1873.6 (much worse).
- E87: a dedicated non-LIRF unmatched LightGBM loses to E20's own unmatched path (Jan+Jul 1555 → 1584; Dec 527 → 712).

**External-data phase (E22, E23, E46, E59, E60).**
- E22/E23 METAR (Iowa Mesonet ASOS 2025, nearest report with `valid ≤ AOBT` else `MVT`, stale > 3 h nulled; 87,589 reports, 10/10 airports, miss ≈ 0, median age 28–30 min). Fog raises the matched residual by +47 s. **December −9 s is real; Jan+Jul is −0.56 s and the tail SSE share does not move; stacked into the E20 experts Jan+Jul goes 368.03 → 368.07 and `>30 min` RMSE 774.8 → 781.0 (hurt).** Not shipped — "ranking is Jan+Jul-like; a December-only win is the wrong split."
- E46: OpenSky/OPDI to recover a missing off-block. Flight-list coverage of unmatched rows is high (LIRF 79.4%, LFPG 93.2%) but `first_seen` is **lift-off, not push-back** (median `MVT − first_seen` = −26 s at LIRF, −42 s at LFPG). `exit-parking_position` events near FCO = **0**. REJECT.
- E59: OpenSky trajectories **blocked at audit** — the 30 challenge columns contain no ICAO24/registration, the only link key is a daily-repeating city-pair callsign; also ground tracks would make BLOCK directly observable, which their own policy forbids.
- E60: OurAirports runway geometry acquired (public domain, all 10 airports); runway mapping worked, but **stand → coordinate mapping is impossible** (no reproducible open crosswalk for the challenge STAND IDs), so taxiway graph / route distance / free-flow time are not constructible. Same blocker hit E37 and E59.

**Meta/stacking phase (E53–E55 second series, E65, E109).**
- E53 stage-3 residual-of-residual cascade with hard-tail amplification — all λ/gate combos worse or neutral.
- E54 meta-layer corrector using the individual E20 expert predictions + seed variance as meta-features — all worse.
- E55 non-linear stacking + magnitude-conditioned blend — best 250.94 vs v13 238.71, top-1% SSE +15.9%.
- E65 causal event-graph transformer (1 layer / 2 heads, context 32 same-airport events, predicting `y − e20`): 253.23 → 253.08, i.e. **−0.15 s**. Same verdict as the earlier temporal_v2 TCN, which was actually submitted and **lost on the leaderboard (317.47 vs E20's 316.97)**.
- E109 v26: the icao-Δ model is 10 s *worse* as a standalone (245.9 vs 235.9); a 25% mix won **both** internal splits but **lost 2.60 s on the leaderboard**. Their own post-mortem (`:1878`): the 2026 OPDI callsign formatting differs from 2025 (`RYRR` vs `RYR`), so the feature's coverage/quality shifts between fit and apply. **"Do not mix a globally worse model into matched again."**

---

## 6. External data sources and exact joins

**Only two external sources are used; only one reaches production.**

### 6.1 OPDI v0.0.2 (EUROCONTROL PRC + OpenSky) — **IN PRODUCTION (v23, v24)**

- Source: `https://www.eurocontrol.int/performance/data/download/OPDI/v002`, files `flight_list/flight_list_{YYYYMM}.parquet` (`experiments/download_opdi.py:22,44-45`). Downloaded to `data/external/opdi/flight_list/` (gitignored — the parquet files are **not** in the repo, only the downloader).
- Note: `download_opdi.py:5,16` disables SSL verification because "eurocontrol.int currently serves an expired cert".
- Columns used: `icao24, flt_id, adep, ades, first_seen, last_seen` (+ `typecode, icao_operator, icao_aircraft_class` from v24).
- **Months loaded: 2025-01..2025-12 for training, plus 2026-01 and 2026-07 for the ranking rows** (`status.md:1500`, `:1829`).
- **Join, step 1 (callsign → icao24)** — `run_e77_icao_turn.py:42-59`:
  ```
  cs = upper(FLIGHT_mvt).replace(" ", "")               # both sides
  join_asof(left_on = MVT_TIME (tz-naive), right_on = first_seen, by = ["cs"], strategy = "nearest")
  keep icao24 only if |MVT - first_seen| < 12*3600  AND  OPDI.adep == airport
  ```
- **Join, step 2 (previous leg of the same airframe)** — `:60-83`:
  ```
  legs = OPDI sorted by (icao24, first_seen)
  prev_last = last_seen.shift(1).over(icao24) ; prev_ades = ades.shift(1).over(icao24)
  icao_turn = first_seen - prev_last      (null unless 1200 < icao_turn < 36*3600)
  icao_same_apt = (prev_ades == airport)
  ```
- Coverage: train unmatched 43%, **ranking unmatched 219/383 (57%)**, of which 146 have `icao_turn ≥ 7200`.
- Attributed gains: v23 `278.84 → 278.36` (−0.48), v24 `278.36 → 276.99` (−1.37). **The last 1.85 LB seconds of their score come entirely from OPDI.**

### 6.2 Iowa Mesonet ASOS METAR — tested, **NOT shipped**

`status.md:1071-1089`. 2025 hourly reports, 87,589 rows, all 10 airports. Join: nearest report with `valid ≤ AOBT` (falling back to `MVT` when AOBT is null), nulled if stale > 3 h. Features: visibility, wind, gust, temperature, RH, precipitation, report age, plus fog / precip / low-vis / strong-wind flags. Outcome in §5.3 above. `experiments/download_metar.py` is referenced in the journal but **not present in the repo** (untracked).

### 6.3 OurAirports runway geometry — acquired in E60, **blocked**, not shipped.

### 6.4 Not external, but worth stating

`ranking.parquet` is read **at apply time only**, for three purposes: (a) the DEP rows being scored, (b) the **ARR** rows of the same file, to build the stand-turnaround features causally *within the 2026 stream* (`submit.py:144-157`), and (c) the rolling/traffic/disruption features, also computed within the ranking stream (`run_submitting_check.py:162-171`). `experiments/common.py:21-31` hard-blocks it from any *fit*.

---

## 7. Prediction-time availability audit

**[S] Things they identified themselves as not available and excluded from production:**

1. **E14's rolling-TAXITIME surface state** — `status.md:967`: "the winning features consume other DEPs' `TAXITIME` — NOT ranking-safe (ranking blanks DEP TAXITIME)". This produced 371.57 / 248.05 and was deliberately abandoned; E17-A is the ranking-safe rebuild and recovers only about a third of the gain.
2. **Same-FLIGHT_ID arrival** — the paired ARR is *after* the DEP (E7, `status.md:174`).
3. **`ARVT_3_flt`** — occurs after MVT for 100% of rows (`status.md:1257`).
4. **Same-callsign previous-DEP actual TAXITIME** — corr 0.41, but other ranking DEPs' TAXITIME is blank (`status.md:176`).
5. **The 24 h wrong-day-BLOCK wrap** — explicitly labelled an oracle with no train-selectable gate (E19-B).

**[S] Things that ARE available and they correctly exploit:**

- ARR `TAXITIME_SEC_mvt` and ARR `BLOCK_TIME_UTC_mvt` in ranking (only DEP rows are blanked) → all the `arr_taxiin` / `since_arr` / `arr_delay` / `arr_type` features. `status.md:100-104`, `matched_submit.py:135`.
- `AOBT_3_flt` on 98.47% of ranking DEP.
- `SCHED_TIME_UTC_mvt` on 100% → `D = MVT − SCHED` for every row, matched or not.
- **`MVT_TIME` of every other flight** (this is a post-hoc reconstruction task, not a pushback-time forecast — `status.md:106-108`).

**[X] Things I flag that they do *not* explicitly flag:**

1. **`add_queue` uses later takeoffs.** `run_e4_e5.py:116-118` computes `queue = #{AOBT_j ≤ MVT_i} − #{MVT_j < MVT_i} − 1`. It only touches AOBT and MVT, both present on ranking, so it is *computable* — but it reasons about flights that had not yet taken off when the scored flight pushed back. They note this as "batch-only" in the leakage list (`status.md:859`) yet keep `queue`/`queue_rwy` in the E20 feature matrix. Given E4/E5 measured their incremental value at ≈ 0, this is harmless in practice, but it is a genuine batch dependency.
2. **`e20_pred` has a train/serve mismatch.** In training, Jan/Jul/Dec rows get the true E20 out-of-fold prediction while the other nine months get a **LightGBM-imputed** `e20_pred` (`matched_submit.py:102-116`). At apply time, ranking rows get the actual v8 parquet value. The Δ and leftover models therefore see three different noise regimes for their single most important feature. They never audit this. **[X] This is a plausible partial explanation for why several internally-both-split-winning matched changes (v14, v15, v26) failed to transfer.**
3. **`d_rank` is computed against the training LIRF `D` distribution even when scoring ranking** (`run_e74_make_v20.py:64,84` passes `dref_tr` for both). Correct as a design (a frozen reference distribution), but it means the feature shifts if the 2026 delay distribution differs. `status.md:1259` measured ranking-vs-2025 PSI at 0.0145 (negligible) and ranking LIRF-unmatched `MVT−SCHED` median 6955 vs 6781, so the shift is small.
4. **Rolling / disruption features on the ranking set are computed within the ranking stream only**, which is Jan 2026 + Jul 2026 — a stream with a **six-month hole** between the two months. At the January→July boundary, backward `join_asof` and trailing-window features will look back across that gap. Window clamps (`< 8 h`, `< 36 h`, `1200 < icao_turn < 36 h`) null most of these out, and `_window_from_prefix` returns NaN when `n < 1`, so **[X]** the damage is probably limited to a handful of rows at the seam — but it is not audited anywhere in the repo.
5. **`typ = 1087 s` in the v24 long-hold pull is a training-set constant** applied to 146 ranking rows. It is a fitted quantity (median of training LIRF unmatched with `icao_turn ≥ 7200`), fitted on training data only, so it is legal — but it is a hard-coded 50% shrink toward a single number, selected because it wins the Jan+Jul analogue while explicitly losing December.

**Cannot be determined from the code:** (a) the actual OPDI parquet contents (files are gitignored, so the 2026-01/2026-07 coverage numbers cannot be re-verified from this repo); (b) whether the OPDI 2026 files were published before their submission deadline; (c) the v22 leaderboard score, which is never recorded; (d) any `data/` file — the entire dataset is absent, so none of the RMSE figures can be reproduced from the clone.

---

## 8. Key quantitative facts worth stealing as priors

All `[S]`, from `status.md` §Discoveries and the experiment log.

- Target identity `TAXITIME_SEC_mvt == MVT_TIME − BLOCK_TIME` verified 100% exact on training DEP. Mean 991 s, median 912, std 546, p99 2339, max 131,167.
- `MAE(BLOCK − AOBT) ≈ 238 s`, RMSE 385 s, exact match only 0.65%, median Δ +51 s, airport-dependent bias (LTFM +296 s, LIRF −118 s). **AOBT and BLOCK are genuinely different sources.**
- Anchor quality, Jan+Jul matched raw RMSE: **AOBT 428.17**, LOBT 816.96, IOBT 816.80, EOBT 745.66, SCHED 2472.
- Airport clock bias `mean(TAXITIME − (MVT−AOBT))`: LTFM −236, EHAM −152, EDDF −53, EGLL −23, LEMD +34, LFPG +35, LSZH +59, LEBL +68, EDDM +82, **LIRF +214**.
- `stand×runway` η² = 0.2525 vs airport η² = 0.1121; same-stand/different-runway taxi spread ≈ 302 s mean; ~6,436 stand-runway pairs with n ≥ 30.
- Unmatched = **total NM record miss** (every `*_flt` column null simultaneously; `partial_NM_match = 0`). 1.08–1.56% of DEP but **55–56% of total SSE**.
- **Top-10 SSE rows = 32% of total Jan+Jul SSE; top-50 = 42%; top-100 = 48.5%.** 7 of the top 10 are LIRF unmatched.
- Perfect-oracle ceilings (Jan+Jul): perfect non-LIRF unmatched → 317.7; perfect LIRF unmatched → 305.8; **perfect all unmatched → 242.8**.
- `corr(T, D)` with `D = MVT − SCHED`: **LIRF 0.90–0.92**, EDDF 0.17, LFPG 0.02, all others ≤ 0.44.
- LIRF unmatched: median T = 1447 s but median D = 6781 s — the override predicts *schedule displacement*, not taxi. 35% of rows have `|resid| < 900`; 64.5% have `>30 min` error.
- Matched SSE structure (v18, n=339,046): `y > 1800 s` is 4.5% of rows and **42% of SSE**; `Δ = y − (MVT−AOBT) > 600 s` is 4.3% of rows and **41% of SSE**. By airport: LIRF 22%, LTFM 15.8%, LFPG 14.3%, EGLL 13.1%.
- High-Δ rows carry a **clock-disagreement signature**: median `clock_range` 3,240 s vs 480 s for the rest — AOBT is the late outlier while EOBT/IOBT/LOBT agree with the airport BLOCK.
- `|AOBT − EOBT/IOBT/LOBT|` disagreement correlates **0.29** with `|E20 residual|`; `>30 min` tail rate rises 3.1% → 16.8% from below to above the p90 of disagreement. Max-disagreement quintile: RMSE 906, tail rate 5.3%, vs bottom quintile RMSE 188, tail rate 0.0%.
- Ranking vs 2025 training: **PSI 0.0145**, unseen runway 0.0%, unseen stand 0.1%, unseen type 0.0%.

---

## Applicable to resilient-hedgehog

Ranked by the gain **their own log** attributes to each item. Our current score is 284.3584; their v24 is 276.99.

### Tier 1 — large, transferred on the leaderboard

**1. LIRF-unmatched gate-delay decomposition `T = max(D − Ĝ, 0)`. Their LB gain: 316.97 → 288.90 = −28.07 s, from 383 rows.**
The single biggest lever in the whole repo. Reimplement exactly:
- Identify LIRF DEP rows where `AOBT_3_flt` is null (≈ 383 in the ranking set, 1,488 in training, ≈ 100% also `AIRCRAFT_TYPE_mvt`-null).
- Train `CatBoostRegressor(iterations=1200, learning_rate=0.04, depth=8, l2_leaf_reg=5.0, loss_function="RMSE", random_seed=1, od_type="Iter", od_wait=80)` on **all ~160k LIRF departures** with target `G = D − T` where `D = MVT − SCHED`, `T = TAXITIME`.
- Features: `[mvt_sched, hour, dow, month]` + the 14 `add_surface` counts (`run_e33_delay_decomposition.py:33-35,53-88`) + the 8 `enrich` terms (`d_log`, `d_rank` against the training-LIRF D percentile, `d × sarr_15`, `d × sdep_15`, `sarr_15 − sdep_15`, `(sarr_15+sdep_15)/15`, `hour_sin`, `hour_cos`) + the 4 stand-turnaround terms (`arr_taxiin`, `arr_taxiin_2`, `since_arr` clamped to 36 h, `arr_delay`); categoricals `[airport, RUNWAY, STAND, AIRCRAFT_TYPE, ADES, flt_prefix, arr_type]` as `fillna("NA").astype(str)`.
- Overlay `T = max(D − Ĝ, 0)` on the LIRF-unmatched rows only.
- **Guardrails from their failures:** do NOT train G on unmatched rows only (that is v9, LB 300.95, a +12 s regression). Do NOT apply `D − G` at any other airport — `corr(T, D)` is 0.02 at LFPG and ≤ 0.44 everywhere outside LIRF (E43, E71b). Do NOT gate the override on any threshold (E13, E19-C, E20).
- Sanity target from their log: LIRF-unmatched holdout RMSE 6033 → 3937 (v8) → 3732 (v20, with the inbound features).

**2. Matched "grec" tail gate: blend toward `P + Δ̂` with `e20` inside the Δ model. Their LB gain: 288.90 → 280.18 = −8.72 s** (v11 −1.19, v12 −2.74, v13 −0.88, v16 −2.01, v17 −0.21, v18 −0.17, v20 −1.54).
This is the item the prior summary entirely omits, and it is the second-largest lever.
```
P    = clip(MVT - AOBT, 0)
Δ̂   = LGB(target = y - P,      features incl. current_model_pred)   # seeds 1,2,3 averaged
rec  = clip(P + Δ̂, 0)
ĥ    = LGB(target = y - current_model_pred, same features)           # seeds 101,102,103
gate = (pred > p90_of_pred) | ((rec - pred) > 200)
out  = max( pred + 0.5*(rec - pred)*gate + 0.35*max(ĥ, 0), 0 )
```
LGB: `n_estimators=900, learning_rate=0.03, num_leaves=63, min_child_samples=40, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0`, no early stopping.
Two non-obvious details do the work: **(i)** the current model's own prediction is a *feature* of the Δ model (without it the gate saturates at −2 s; with it, matched 244.76 → 239.32), and **(ii)** the leftover residual is applied **positive-only** (`max(ĥ,0)` — lift under-predictions, never push down). Their `p90 = 1462 s` and `hthr = 200 s` are hard-coded; recalibrate `p90` on our own prediction distribution.

**3. Same-stand inbound turnaround features. Their LB gain: 281.72 → 280.18 in the LIRF G model, and 284.10 → 281.72 in the matched Δ model = −2.38 s combined.**
Backward `join_asof` from the DEP's `MVT_TIME` onto the ARR stream keyed `["airport", "STAND_mvt"]`, right key = ARR `BLOCK_TIME_UTC_mvt`. Emit `since_arr`, `arr_taxiin`, `arr_taxiin_2` (previous ARR's taxi-in at that stand), `arr_delay`, `arr_type`; a second asof anchored at the DEP's `AOBT` giving `arr_taxiin_aobt` / `since_arr_aobt`; `since_dep` (previous DEP from that stand); and `rwy_arr_taxiin` (last ARR taxi-in on the same runway). **Null everything outside `0 < since_arr < 8 h`** (36 h for the LIRF G model) — do not zero-fill. Ranking ARR rows carry taxi-in, so this is fully available at prediction time; build the ranking-side ARR frame from `ranking.parquet` itself.

### Tier 2 — moderate, transferred

**4. Overlay-and-splice architecture instead of retraining. No direct RMSE attribution, but it is what made 22 submissions possible in 9 days.**
Every improvement is a small-population row replacement on the previous submission parquet, joined by `MVT_ID_mvt`, with hard asserts (row count == 344,841, ID order identical to `submitting.parquet`, zero nulls/NaN, min ≥ 0). See `matched_submit.py:246-273`. This gives exact attribution per change (v20 changed 383 rows; v24 changed 146) and makes a bad idea trivially revertible. If we are not already doing this, adopt it before anything else in this list.

**5. Soft λ-blend of two gate models rather than replacing one with the other. Their LB gain: 280.18 → 278.36 = −1.82 s.**
`G = (1−λ)·G_all + λ·G_unmatched_only` with `λ ∈ [0.2, 0.25]`. The same `G_unmatched_only` at `λ = 1.0` cost them **+12 s** on the leaderboard (v9). Recover `G_all` arithmetically from the prior submission as `G = D − T_prev` — no refit needed. The generalisable rule they wrote down (`status.md:1795`): a specialised sub-model that is worse standalone can still be worth 20–25% of a blend; never let it take the whole weight.

**6. External OPDI `icao_turn` for unmatched rows. Their LB gain: 278.84 → 276.99 = −1.85 s.**
Download `flight_list_{YYYYMM}.parquet` from the OPDI v002 endpoint for 2025-01..12 plus 2026-01 and 2026-07. Join our callsign (`upper`, spaces stripped) to OPDI `flt_id` by `join_asof(strategy="nearest")` on `first_seen`, keeping only `|MVT − first_seen| < 12 h` and `OPDI.adep == our airport`. Then `icao_turn = first_seen − prev last_seen` of the same `icao24`, valid only in `(1200 s, 36 h)`. Use it (a) as a feature in the unmatched-G LightGBM together with `typecode` / `icao_operator` / `icao_aircraft_class`, and (b) as the **long-hold rule**: on LIRF unmatched with `icao_turn ≥ 7200 s`, set `T = 0.5·T0 + 0.5·1087 s`. Expect ~57% coverage on unmatched rows and ~146 rows firing.
**Caveat they paid for:** 2026 OPDI callsign formatting differs from 2025 (`RYRR` vs `RYR`) — this is their stated explanation for why v26 lost 2.60 s. Audit the 2026 join hit-rate separately before trusting any OPDI-derived feature.

**7. NNLS-weighted residual ensemble of heterogeneous learners. Their internal gain: 372.36 → 368.03 overall, 250.98 → 244.76 matched.**
CatBoost residual (0.456) + **per-airport** LightGBM residual experts (0.491) + XGBoost-numeric residual (0.053); NNLS on the holdout, weights renormalised to sum 1; pooled LightGBM and direct-target LightGBM both receive weight 0. Note the per-airport experts alone (370.52 / 247.76) nearly match the whole blend — **[X] if we are short on time, per-airport experts are the higher-yield half.**

**8. `dis_state_30m` airport push-delay state. Their internal gain: matched 256.46 → 253.82 (Jan+Jul), 230.04 → 226.95 (Dec); ~1% of matched RMSE.**
Mean `AOBT − EOBT` of *other* matched flights whose **AOBT** falls in `(MVT_i − 30 min, MVT_i]`, with the scored flight's own contribution subtracted arithmetically, and **NaN (not 0) when the window is empty**. Plus `dis_frac20_30m` (fraction of those with `AOBT−EOBT > 1200 s`) and `dis_state_vs_hour` (deviation from the train-split `airport × hour` mean, as a *difference*). Prefix-sum implementation at `run_e16a.py:185-219`. Their instruction: do not sweep more windows — the family saturates immediately.

### Tier 3 — cheap hygiene with measured value

**9. E18-H hygiene bundle. Their internal gain: 376.04 → 372.36 overall, 253.82 → 250.98 matched.**
Two one-line changes: (a) fit the `geo_mean` stand×runway tables on **matched training rows only**, so unmatched LIRF's multi-hour `y` cannot poison the geometry table; (b) **drop the LIRF-override rows from the residual model's training set** — the tree was being trained on rows it never scores. They never ablated these separately (`status.md:934`).

**10. Chain-don't-dump missing-value policy.** When a feature is missing, fall through to the next-best model that can still score the row (`per-airport OLS → geo_mean → airport mean → global mean`) rather than the airport mean. Measured cost of getting this wrong: matched RMSE 379 → 398.

**11. Report LIRF-unmatched RMSE as a separate metric in every experiment.** It is ~30% of total SSE from ~0.1% of rows. Any aggregate-only metric will hide both the wins and the regressions on this slice.

### Do NOT spend time on (falsified with measurements)

- Rolling/queue/congestion state features of any kind on top of a model that already has the clocks: E4/E5 (Δ≈0 linear), E19 (−1.34 s and the top-1% SSE share got *worse*), E41 (blend weight 0), E52 (exactly redundant), E62–E64 (redundant once benchmarked against the right baseline). **Beware E62's trap: it showed −2.17 s against a weak baseline and 0 against the strong one.**
- Any non-L2 residual objective: Huber, tail-weighted L2, two-stage `P(y>30)` mixture, quantile/pinball. All four trade bulk for tail at constant RMSE; NNLS assigns the quantile expert weight exactly 0.
- Identity/history priors: callsign or flight-number residual priors (E42, both splits worse), service medians (E28/E31), historical analogue retrieval (E61, prior ≈ 0), latent KMeans regimes (E57).
- Gating the LIRF `MVT−SCHED` / `D−G` override on anything: threshold, stand, hour, destination, airline prefix, neighbour taxi fraction. Three separate experiments (E13, E19-C, E20) all failed; `MVT−SCHED` is ~6000 s in *both* y-regimes.
- `FLIGHT_ID` as an aircraft tail number — it is a city-pair NM flight, and the paired row is the arrival *after* this departure.
- Stand-coordinate / taxiway-graph reconstruction — no reproducible open crosswalk exists for the challenge STAND IDs (blocked three times: E37, E59, E60).
- Sequence models (TCN, event-graph transformer) — the TCN was actually submitted and lost (317.47 vs 316.97); the transformer gave −0.15 s.
- Deeper hyperparameter search on the matched branch — a full sweep plus seed averaging bought −0.43 s Jan+Jul and 0.00 s December.
- METAR — real December effect (−9 s), essentially nothing on Jan+Jul, and it **hurts** the `>30 min` tail when stacked into the ensemble. Since the ranking set is Jan+Jul, this is the wrong split to optimise.

### One strategic observation

**[S]** Their `status.md:1711-1715` and `:1259-1265` conclude that matched RMSE is at a floor of ≈ 236 s given the 30 supplied columns, that the leaderboard tracks matched RMSE roughly 1:1, and that essentially all remaining headroom is the LIRF/unmatched gate-hold — information that is simply absent from the dataset. Their final 6 LB seconds (282 → 277) came entirely from **383 LIRF-unmatched rows plus 146 long-hold rows**, i.e. 0.15% of the submission. **[X]** If we are at 284.36, the arithmetic says the cheapest path to ~277 is not a better general model — it is the LIRF `D − G` overlay (Tier 1 item 1) plus the matched grec gate (item 2), in that order.
