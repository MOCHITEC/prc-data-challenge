# P1 — the unmatched / missing-proxy lane

Plan, 2026-09-21. No code was modified to produce it.

Marking: **[S]** = read from a source in this repository or quoted verbatim from a slot result
file that cites its own source; **[X]** = my inference or arithmetic. Every [S] carries a
`file:line` or a `file §section`. Three small read-only measurements were taken while writing
this plan (§1.9, §4.3, §5.3); they are marked **[S-new]** and the script that produced them is
named.

---

## 0. Why this lane, and what it is worth

Four independent research slots converged here. The evidence, restated with its source:

| Measurement | Slot |
|---|---|
| satam2 ablation: raw target 491.329 → +AOBT residual 486.230 (−5.1) → **+a dedicated missing-row model 361.735 (−124.5)**; per-fold missing RMSE 3,186.77 → 1,531.60 (F1) and 2,606.64 → 1,595.10 (F3) | [S] `results/23-repo-col3name-satam2.md` §2.5 |
| javidmardanov: matched RMSE **226.85** against missing **1,157.63**; 5,294 of 353,045 rows = 1.5 % of rows and **28.4 % of total SSE**; the worst 100 missing rows carry 73.9 % of the missing SSE | [S] `results/08-repo-javidmardanov.md` §2 |
| merry-quicksand: their entire July gap to the 271–274 tier was the unmatched lane **and it was entirely Rome** — LIRF alone +1,210 M SSE, the other nine airports together 65 M *better* than the competitor | [S] `results/21-repo-merry-quicksand.md` §3.6 |
| kind-mango: the LIRF regime head is their second-largest win; `R_norm` alone took LIRF clean RMSE **565.10 → 318.08** | [S] `results/22-repo-kind-mango.md` §5 |

Our own position, translated onto the board. Board decomposition with
N = 344,841 = 339,551 matched + 5,290 unmatched, LB 284.3584, matched 218.0 ⇒ implied unmatched
RMSE **1,490.2** [S] `docs/data_audit_2026-09-18/results/review-far-rows.md` §3.

**[X]** Differentiating `MSE_total = 0.9846596·m² + 0.0153404·u²` at u = 1,490:
**every 100 s off the unmatched RMSE is worth about 8.0 s of board RMSE.** Closing all the way to
javidmardanov's local missing-row level (1,490 → 1,158) would be about **−25 s**, which is larger
than our whole remaining gap to the top-10 tier. That is the size of the prize; §5 says how much
of it is actually reachable.

**Counterweight, and it is the honest number.** merry-quicksand priced the same pool from the
inside: a *perfect* model of every learnable unmatched row plus their matched ship reaches
≈ 260.6 from 281.5, and recovering 10/20/30 % of the pool gives 278.2 / 276.3 / 274.3
([S] `results/21-...` §3.7). **[X]** Scaled to us, recovering 30 % of the learnable pool is worth
roughly **−7 s**, not −25. The difference between the two figures is the rare-class rows
(24 h date-slips, "Family B") that *"every serve-time test they built fails to identify"*
([S] `results/21-...` §3.9). Plan against −7, not −25.

---

## 1. What model B is today, exactly

### 1.1 Routing

`src/prc/train.py:398` — `u_tr, u_tg = tr.filter(pl.col(PROXY).is_null()), tg.filter(pl.col(PROXY).is_null())`,
with `PROXY = "mvt_minus_aobt3"` (`train.py:40`) = `MVT_TIME_UTC_mvt − AOBT_3_flt`
(`src/prc/features.py:62`). Two-way split only: no third regime for negative or over-long proxy.
Row counts **2,062,577 matched / 22,470 unmatched** training, **339,551 / 5,290** ranking
([S] `docs/investigation_2026-09-16/results/lirf-matched-fill.md` §3.1;
`results/23-...` §2.2 confirms both counts match satam2's to the unit).

### 1.2 Target

`unmatched_target` (`train.py:265-266`) = `LABEL − sched_reparam_shift(frame, cfg)`.
`sched_reparam_shift` (`train.py:241-262`) is `mvt_minus_sched` on the airports named by
`unmatched_sched_reparam_airports` and exactly `0.0` everywhere else, including on rows whose
`mvt_minus_sched` is null. The shift is added back after prediction (`train.py:412-413`).

Defaults: `unmatched_sched_reparam: bool = False` (`train.py:108`),
`unmatched_sched_reparam_airports = ("LIRF",)` (`train.py:113`).
**`configs/main.yaml` does not set the flag** — it is on only in `configs/reparam_lirf.yaml:16`
and `configs/submit_v10_reparam.yaml:27`. See §4.1: this is a live prerequisite, not a footnote.

### 1.3 Features

`cols_b` (`train.py:400-406`) = `visible_numeric_columns` minus every column whose name starts
with `mvt_minus_aobt3`, `aobt3`, `lobt_minus`, `mvt_minus_lobt`, `mvt_minus_eobt1`,
`mvt_minus_iobt`; plus the 10 `CATEGORICALS` (`features.py:33-44`). Counted independently at
**84 numeric + 10 categorical = 94** ([S] `docs/investigation_2026-09-16/results/mms-mixture-reparam.md` §2);
`docs/CODEX_BRIEF.md:55` says 83 + 10. The discrepancy is one column and is not load-bearing here,
but it should be resolved by Stage 0's instrumentation rather than left.

`mvt_minus_sched` and `mvt_minus_sched_slog` are both visible to model B [S] (mms §2). The nine
label-derived `ENCODING_COLS` (`train.py:43`) are visible; `OFFSET_COLS` are not (`use_offset_encoding`
defaults false, `train.py:130`).

**What model B actually sees on an unmatched row** [S-new] (script:
`docs/competitor_research_2026-09-21/scripts/p1_cols.py`, a read-only null-rate scan of `data/work/feat_{train,rank}.parquet`):

| categorical | null rate, all 22,470 train unmatched | null rate, 1,488 LIRF train unmatched | null rate, 5,290 rank unmatched |
|---|---|---|---|
| `ADEP_mvt` / `RUNWAY_mvt` | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| `STAND_mvt` | 0.0005 | 0.0054 | 0.0000 |
| `AIRCRAFT_TYPE_mvt` | 0.0671 | **0.9993** | 0.0733 |
| `WK_TBL_CAT_flt`, `AIRCRAFT_OPERATOR_flt`, `MARKET_SEGMENT_flt`, `FLIGHT_TYPE_flt` | **1.0000** | **1.0000** | **1.0000** |
| `ADES_mvt` / `ADES_REGION` | 0.0019 | 0.0020 | 0.0011 |

So on a LIRF unmatched row model B's categorical vector is effectively `{RUNWAY, STAND, ADES}` —
`ADEP` is constant inside the group, aircraft type is missing on 1,487 of 1,488 rows, and all four
NM-side categoricals are structurally absent. This matches
[S] `docs/data_audit_2026-09-18/results/fill-pattern.md` §2.9 ("on all 22,470 training unmatched
rows, `EOBT_1_flt`, `LOBT_flt`, `IOBT_flt`, `ARVT_1_flt`, `AIRCRAFT_OPERATOR_flt`,
`MARKET_SEGMENT_flt` are 100 % null").

**`FLIGHT_mvt` is present in the feature frame and is not a feature.** It is 0.29 % null over all
unmatched training rows, 4.37 % null at LIRF unmatched, with **129 distinct 3-letter prefixes**
there [S-new]; it is not in `CATEGORICALS` (`features.py:33-44`) and no prefix column is derived
anywhere in `features.py`. `FLIGHT_RULE_mvt` is likewise present, 0 % null at LIRF unmatched, and
not a feature. See §3 item 1 — this is the single largest concrete omission found.

### 1.4 Hyperparameters

`for_unmatched()` (`train.py:135-145`) copies `TrainConfig` with `num_leaves`,
`min_data_in_leaf` and `num_rounds` replaced by the `unmatched_*` fields. Dataclass defaults are
255 / 20 / 3000 (`train.py:95-97`) but **`configs/main.yaml:18-20` overrides them to 127 / 50 /
2000**, with the reason recorded at `configs/main.yaml:4-9`: CV rewarded 255 leaves by 34.8 s and
the leaderboard charged 13.2 s for it (v2 312.02 → v5 325.22). Everything else is shared with
model A: `learning_rate 0.05`, `feature_fraction 0.8`, `bagging_fraction 0.8 / freq 1`,
`lambda_l2 1.0`, `deterministic`, `force_row_wise` (`train.py:55-74`). `n_seeds: 3`
(`configs/main.yaml:21`).

### 1.5 Fit protocol

`_fit` (`train.py:302-357`): early-stop on an inner holdout of the latest
`inner_holdout_months = 1` calendar month of the training side, rescale the best iteration to the
full refit size, refit on all training rows under `n_seeds` seeds and average. Model A is fitted
first in the same call chain and consumes the same `early_stopping` / `inner_holdout_months`
fields — they cannot be changed for B alone
([S] `docs/handoff.md` §9: `unmatched_num_rounds: 600` produced a bit-identical run because the
field is a cap inside a `min()`, not a tree count).

Below 200 labelled unmatched rows the branch degrades to `ref_p10_fb` with a median fallback
(`train.py:418-420`).

### 1.6 Guards on the output

`_assemble` (`train.py:427-435`) applies `fill_null(fallback).fill_nan(fallback).clip(lower_bound=0.0)`.
**There is no upper bound, no per-airport cap, no schedule floor and no tail rule of any kind.**

### 1.7 Evaluation

`run_cv` (`train.py:463-536`) records `metrics(...)` and `by_airport` per fold
(`train.py:496-500`). `by_airport` is computed over **all** rows of the fold, not over the
unmatched subset; **nothing in the experiment record isolates the unmatched lane**. The OOF
parquet written at `train.py:534` has two columns, `MVT_ID_mvt` and `oof`, and no fold label.

`prc.compare` does report the lane — `unmatched rows`, `unmatched @ LIRF` and unmatched-by-airport
— but its own docstring states that `verdict()` never sees any of it
([S] `src/prc/compare.py:143-152`: *"the subset that decides the leaderboard (1.5 % of rows,
concentrated at one airport) is never reported on its own [to the verdict]"*).

### 1.8 What it has already bought

| build | LB | note |
|---|---|---|
| v4 | 305.027 | |
| **v10 (current best)** | **284.3584** | v4's 339,551 matched predictions byte-identical + model B with the LIRF reparameterisation. The −20.67 s is model B alone. [S] `experiments/submissions.jsonl` v10 |
| v11 | 293.978 | v10 + a mixture on **11** LIRF unmatched rows with `ms ≥ 43,200`. **+9.62 s.** [S] ibid. v11 |

**[X]** v11 is the most important calibration we own for this lane: 11 rows moved the board by
9.62 s, i.e. **≈ 1.9e9 of added SSE over 11 rows ≈ 13,200 s of added RMS error per row**. The
smallest cell in this lane is worth ~1 s of board RMSE *per row*, in either direction.

### 1.9 Per-airport error of the shipped model B

[S] `docs/investigation_2026-09-16/results/reparam-implementation.md` §(forward fold, n = 3,005
unmatched). A = model B without the reparameterisation, B = with it (what v10 ships):

| airport | n | A (no reparam) | **B (shipped)** | diff | CI95 |
|---|---:|---:|---:|---:|---|
| **LIRF** | 140 | 8,578.4 | **4,124.6** | −4,453.76 | [−7,092.61, −1,152.32] |
| EHAM | 545 | 1,184.0 | 343.7 | −840.35 | [−1,291.96, −237.29] |
| LFPG | 685 | 1,217.0 | 708.9 | −508.05 | [−969.40, −15.07] |
| EDDF | 188 | 1,683.3 | 507.6 | −1,175.72 | [−2,025.94, +3.24] |
| EDDM | 165 | 1,349.8 | 855.3 | −494.52 | [−1,179.04, −7.05] |
| LEBL | 156 | 1,150.4 | 383.2 | −767.21 | [−1,498.55, +42.32] |
| EGLL | 181 | 727.0 | 584.2 | −142.76 | [−379.26, +22.91] |
| LTFM | 640 | 420.2 | 364.6 | −55.55 | [−98.47, −23.24] |
| **LSZH** | 198 | 366.8 | **418.2** | **+51.36** | [−12.67, +117.72] |
| LEMD | 107 | 360.4 | 364.7 | +4.31 | [−7.95, +18.54] |
| pooled | 3,005 | 2,109.3 | **1,028.4** | −1,080.91 | [−1,590.83, −548.98] |

**[X]** arithmetic on that table: under the shipped model B, **LIRF unmatched = 4,124.6 on 140
rows carries 75 % of the unmatched SSE**; the nine other airports pool to **527.0** on 2,865 rows.
LSZH is the one airport the reparameterisation made worse.

**[X]** Against the competitors on the same lane: merry-quicksand's July 2025 LIRF unmatched was
3,925.6 against the competitor's 3,437.9 ([S] `results/21-...` §3.6). We are at 4,124.6 — in the
same band and behind both. Our non-LIRF unmatched (527.0) is competitive with their July non-LIRF
(EHAM 245.2 / LTFM 704.4 / EDDF 387.8 against our 343.7 / 364.6 / 507.6). **The gap in this lane
is LIRF, for us exactly as for them.**

Three-fold numbers, for reference: pooled unmatched OOF **2,296.1** on `oof_main-48c19bc830`
(no reparam), reproduced at 2,326.5, and **1,632.1** under the reparameterisation
([S] mms §3.2, §3.3). The `1,875.3` at `train.py:9` is a different comparison (the
one-model-over-everything ablation) and is not the current baseline.

---

## 2. The four competitors' unmatched lanes, at reimplementation grade

### 2.1 satam2 — the dedicated missing-row model, and the "normalized" variant

[S] `results/23-repo-col3name-satam2.md` §2.2, §2.5, §3.5.

**Routing.** `proxy_status(proxy, config)` → `"missing"` if `~isfinite(proxy)`, `"invalid"` if
outside `[proxy_min=0, proxy_max=7200]`, else `"present"` (`models/residual.py:4-6`,
`configs/release.yaml:19-20`). Four inference regimes, three trained models: `residual`
(target `y − proxy`, fitted only on `status == "present"`), `direct` (raw `y`, serves negative
proxy), `missing` (raw `y`, fitted on **all 22,470** missing rows, not a sample —
`train.py:213`: `component_positions = np.arange(len(x))` when `name == "missing"`).

**Value.** The missing specialist is the largest single item in their entire log: seasonal
486.230 → 361.735 on top of the residual, 491.329 → 368.560 on top of the raw target. Their
seasonal metric is `sqrt(w_jul·MSE(F1) + w_jan·MSE(F3))` with the real ranking proportions
192122/344841 and 152719/344841 (`metrics.py:23-28`).

**The "normalized" member** (`tail240_.../models/normalized_missing_tune.py:22,27-42`):
```
scale(x)  = sqrt(3600² + (schedule_proxy_sec − 900)²)   # 3600 when schedule missing
Z         = (y − 900) / scale
weights   = scale² / mean(scale²)
reconstruct: 900 + scale·Z
```
LightGBM `lr .05, num_leaves 31`, 111 rounds, no eval set. Their justification: the weighted
scaled squared loss *equals* the raw squared loss divided by the mean, so the reparameterisation
is MSE-neutral by construction and buys only heteroscedastic capacity allocation. Blended into
the missing route at a **fixed** 25 %, with an `ExtraTreesRegressor(n_estimators=300,
max_features=0.7, min_samples_leaf=1, bootstrap=False)` at 18.75 %:
`0.5625·V2 + 0.1875·ET + 0.25·normalized`, no weight fitting.

**Value and caveat.** Removing `normalized` costs +2.41 s seasonal, removing ExtraTrees +2.52 s,
removing both +7.04 s — on 1.52 % of rows. Their own warning: *"Removing five
component-supporting rows per fold leaves only about 0.011 s of normalized benefit and reverses
the ExtraTrees benefit."* `min_samples_leaf=20` lost on all six panels; use leaf 1. And:
*"An analogous finite-clock formulation was negative"* — do not put it on the matched lane.

### 2.2 javidmardanov — the missing-timestamp mixture of experts, soft inside hard

[S] `results/08-repo-javidmardanov.md` §4.

Two components, one of which is not a model:
1. **schedule-identity expert** = the raw quantity `mvt_minus_SCHED_TIME_UTC_mvt`. No model.
2. **normal expert** = CatBoost on raw `TAXITIME_SEC_mvt`, 600 trees, depth 6, lr 0.06, L2 30,
   `one_hot_max_size=20`, `max_ctr_complexity=1`, trained on
   `training & ~tail & ~identity & (y < 12000)` — **identity rows removed, target capped at
   12,000 s** (the cap-12000 change was itself a tested improvement over 6,000).

**Routing — soft inside hard** (`scripts/missing_v2.py:220-228`):
```python
cls = CatBoostClassifier(iterations=400, depth=5, learning_rate=.06, l2_leaf_reg=30)
cls_weight = weight * np.clip(1 + (np.maximum(schedule, 0)/3600)**2, 1, 25)
cls.fit(x[train & rome], identity[train & rome], sample_weight=cls_weight)
probability = cls.predict_proba(x_valid)[:, 1]
probability[~rome | (schedule <= 0)] = 0        # HARD outer gate
raw = probability*schedule + (1-probability)*normal   # SOFT inner mix
```
* `identity := rome & (|y − (MVT − SCHED)| ≤ 6)`; the LIRF training identity fraction is **18.08 %**.
* Hard gate: `p` is forced to exactly 0 unless LIRF **and** `MVT − SCHED > 0`. At the other nine
  airports the mixture degenerates to the normal expert — no switch, no per-airport branch.
* Classifier trained on LIRF rows only, all months except the held-out ones, **121,394 rows** —
  i.e. matched LIRF rows are in the classifier's training set even though the mixture only serves
  unmatched rows.
* Regressor sample weighting: all missing-AOBT rows get weight **8** against 1 for a subsample of
  120,000 LIRF + 80,000 non-LIRF known rows.
* **Protected tail:** LIRF rows with `MVT − SCHED ≥ 24,000` are removed from the mixture entirely
  and keep a frozen earlier prediction. 25 ranking rows.
* Blended in, never substituted: `pred[gate] += α·(expert − pred)` with α = **1.0** at the nine
  airports and **0.4224** at LIRF.

**Value:** 266.052 → 263.222 local (Jul+Nov), 262.959 with cap-12000. **Warning they record
themselves:** α fitted on July gives 0.377, on November 0.939, on even dates 0.659, on odd dates
0.296 — a **3× spread** on a coefficient fitted on a few thousand rows.

### 2.3 kind-mango — the LIRF regime head and the calibrated `p_fb` gate

[S] `results/22-repo-kind-mango.md` §5, §4.2, §4.4, §1.1.

**Regime.** `fallback := |y − sd| < 60` where `sd = sched_delay`; **16.6 % of the 160,704 LIRF
training rows**, against 3.2–5.8 % at six other airports. 12 of the 14 24-hour rows in all of 2025
are at LIRF.

**Head, as served:**

| LIRF row class | served value |
|---|---|
| null `FLIGHT_ID_mvt` **and** `sd > 14,400` | Step A band table, 11 mutually-exclusive bands: `p_fb·sd + p_24h·(86,400 + mean_24h_extra) + p_norm·mixture` |
| `sd > 70,000`, outside Step A | `(5/6)·(86,400 + 1,150) + (1/6)·sd` |
| all other LIRF rows | `mixture = p_fb_cal·sd + (1 − p_fb_cal)·R_norm_LIRF` |
| every non-LIRF airport | the plain 3-seed base mean — **no head at all** |

* `R_norm_LIRF` = 5-member LightGBM `linear_tree` mean trained **only on LIRF "genuine" rows**
  (`|y − sd| ≥ 60` and `y < 80,000`), 127 leaves, 166 columns, **LIRF-only encoders**.
* `p_fb_LIRF` = LightGBM binary classifier for `|y − sd| < 60` on **all** LIRF rows, 167 columns
  including 14 leave-one-month-out fallback-rate encodings, **isotonic-calibrated**.
  Full-year in-sample AUC 0.9256.
* `fbrate_flt_prefix` — the fallback rate keyed on the flight-designator prefix — ranked
  **1 of 167** features.

**Value:** `R_norm` alone LIRF clean 565.10 → 318.08. Shipped: v22 head **−9.8 s live**, v23 rate
encodings −2.8, v16 Step A −58, v30 LIRF-only encoders −1.28, v55/v56 refits −0.42/−0.33.

**Three warnings that bound the design:**
1. *"Naive v19 mixture (uniform mixture everywhere) FAILED at LIRF (+5.3 G MSE) because `R_norm`
   underestimates on high-`sd` rows and soft mix pulls all rows toward `sd`."*
2. The same architecture applied at nine non-LIRF airports **failed live twice**: the
   fallback-and-24h classifier went hold-out −40 / live **+195**, and the LIRF+EGLL detector soft
   mix hold-out −8 / live **+5**. Their diagnosis: *"these detectors model punctuality, not
   reporting failure"* at the nine airports, where the positive class is not a distinct physical
   regime.
3. MB3 (v51, **−5.99 s live at zero offline price**): they were *training* the base on 160,704
   LIRF rows the serving stack always overwrites. *"Rows whose label the served value never
   reproduces shape splits and leaf slopes that served rows pass through."*

### 2.4 merry-quicksand — hierarchical shrunk cells, LIRF logistic, winsorised body

[S] `results/21-repo-merry-quicksand.md` §4.5, §3.5, §3.7, §3.8.

```
pred = max( p·sp + (1 − p)·nf , 1 )          sp = MVT_TIME − SCHED_TIME
```
* `p` = P(the airport stamped SCHEDULE into the off-block field), fill defined as
  `|BLOCK − SCHED| ≤ 60 s`. Rate: **LIRF 48.5 % against 0.2–9.4 % elsewhere**.
* `p` and `nf` are **hierarchically shrunk** cell estimates over three nested keys
  `[ADEP] → [ADEP, sp_band] → [ADEP, sp_fine, dayoff]`, each blended toward its parent with
  `K_SHRINK = 5.0` pseudo-counts. `SP_EDGES = [-inf, 900, 1800, 3600, 7200, 10800, 21600, 43200, inf]`;
  `dayoff = clip(date(MVT) − date(SCHED), 0, 1)`.
* **At LIRF only**, `p` comes from a logistic classifier instead of the cell rate: smoothed
  log-odds of airline / stand-first-char / destination-prefix fill rates, **plus the airline's
  "copy propensity" estimated on LIRF's 159 k MATCHED rows**, where the copy is visible as
  `|BLOCK − SCHED| ≤ 60 ∧ |BLOCK − AOBT_3| > 300`; plus `dayoff` and `log1p(max(sp,0))`. `C = 1.0`.
  Established at −81 s on the stratum.
* `nf` is a **hybrid**: `nf_hybrid(cells, fit) = where(cells < 3000, fit, cells)` — a fitted
  regressor for the body, the cell estimate for the tail.
  Body regressor = `HistGradientBoostingRegressor(max_leaf_nodes=15, max_iter=200,
  min_samples_leaf=100)` on `sp, dayoff, hr` + 5 smoothed encodings, fitted on **non-fill rows
  only**, target **winsorised at ±3,000 s for the fit only**. It *refuses* fill rows: *"a fill
  row's label is `sp`, thousands of seconds, and would poison every encoding and the fit."*
* **A plain gradient booster in place of the cell estimator scored 1,793.6 against 1,648.4** — the
  cell estimator is deliberate, because of the heavy tail.
* **Congestion witness (E3C, shipped, board −1,775 MSE):** per airport × UTC clock hour of
  take-off, the median `MVT − AOBT_3` over that hour's **matched** departures, NaN when n < 5;
  unmatched rows never contribute. As a winsorised HGB residual correction it moved the stratum
  **1,574.5 → 1,527.2 (−47.3 [−62.8, −36.8]), better in 12/12 months, seed-stable**; Spearman with
  the non-fill target +0.41.
* LIRF rules lane, applied in order: (1) `fill_classifier` arm R — stake-weighted LightGBM P(fill)
  trained on **all** LIRF departures, weight `clip(1+(max(sp,0)/3600)², 1, 25) × 8 if unmatched`;
  (2) `dateslip` inside (LIRF, `dayoff == 1`, `sp ≥ 40,000`): two-class expectation
  `q·(86,400 + m) + (1−q)·sp`; (3) **`schedule_floor`** — for LIRF unmatched with
  `24,000 ≤ sp < 86,400`, `pred = max(pred, rint(sp))`, justified by "in 2025 every LIRF unmatched
  row with `sp ≥ 24,000` had `y ≥ sp − 60`"; (4) `local_day_schedule` — `rint(sp)` when the
  schedule falls on the same Europe/Rome local day.

**Failures they recorded on this lane:** an oracle fill flag + constant non-fill scored
1,868 → **3,081 (worse)** — *"the mixture's hedging does real work"*; shrinkage/temperature
recalibration of the LIRF `p` gave −251 / +92 / −122; fine `sp` buckets rejected under LOMO;
arrival-side tell for LIRF fills NOT WORKING (a permuted control kept 71 % of the "gain").

---

## 3. What model B lacks, concretely

| # | They have | We have | Source |
|---|---|---|---|
| 1 | Operator/airline identity as the load-bearing feature of the LIRF gate (merry-quicksand: airline copy propensity; kind-mango: `fbrate_flt_prefix` ranked 1/167) | **Nothing.** All four `_flt` categoricals are 100 % null on unmatched rows and `AIRCRAFT_TYPE_mvt` is 99.93 % null at LIRF unmatched. `FLIGHT_mvt` is in the frame, has 129 distinct prefixes at LIRF unmatched, and **is not in `CATEGORICALS`** | [S-new] §1.3; `features.py:33-44`; `fill-pattern.md` §2.9 |
| 2 | A **soft** per-row fill weight `p`, so the mechanism degrades gracefully where it is absent | A **hard per-airport switch** (`unmatched_sched_reparam_airports`, `train.py:113`). It is on or off | `train.py:241-262` |
| 3 | The fill classifier trained on **matched** rows of the same airport, where the fill is directly observable (merry-quicksand 159 k; javidmardanov 121,394) | Our only fitted classifier (mms variant B) was trained on the ~22 k unmatched rows alone | mms §3.6 |
| 4 | A body regressor with **fill rows removed** and the target winsorised/capped for the fit only (all three of merry-quicksand, javidmardanov, satam2) | One regressor on all 22,470 rows, fill and non-fill together, no winsorisation | `train.py:407-411` |
| 5 | A **cell estimator for the tail** and a fitted model only for the body (`nf_hybrid` at 3,000 s) | GBDT everywhere | `results/21-...` §4.5 |
| 6 | Hierarchically shrunk cell estimates over `[ap] → [ap, sp_band] → [ap, sp_fine, dayoff]`, k = 5 | No `sp`-band or `dayoff` structure anywhere; `dayoff` is not computed | `features.py` (absent) |
| 7 | A congestion **witness**: airport × hour median `MVT − AOBT_3` over that hour's matched departures, n ≥ 5 | Partially: `nbr_mvt_minus_aobt_mean_{30,60}m`, a 60-s-lagged rolling **mean** with no n-guard, and it *is* visible to model B (`nbr_*` survives the `cols_b` prefix filter) | `features.py:200-208` |
| 8 | Multiple members on the missing route at **fixed fold-averaged weights** (satam2: V2 / ExtraTrees / normalized) | One LightGBM, seed-averaged only | `train.py:344-354` |
| 9 | A heteroscedastic, MSE-neutral target reparameterisation for the missing route | Only the additive LIRF shift | `train.py:265-266` |
| 10 | Bound-shaped overlays: schedule floor, protected tail, per-airport caps | `clip(lower_bound=0.0)` and nothing else | `train.py:433` |
| 11 | A third regime for finite-but-implausible proxy (`proxy < 0` or `> 7200`) | Two-way split only | `train.py:379`, `train.py:398` |
| 12 | Flight-ID purging between CV stages (satam2 `splits.py:26-31`) | Month-blocked folds with no flight-ID purge | `src/prc/cv.py:26-51` |

Items 1–5 are the substance. Items 6–12 are secondary.

---

## 4. Prerequisites and the two known measurement hazards

### 4.1 Prerequisite A — the shipped model is not reproducible from one config

`configs/main.yaml` does **not** carry `unmatched_sched_reparam`. The 284.3584 submission is a
hand-assembled probe: *"v4's 339,551 matched predictions kept byte-identical, only the 5,290
unmatched rows replaced by model B with the LIRF schedule reparameterisation"*
([S] `experiments/submissions.jsonl` v10 note). The reparameterisation has only ever been
cross-validated with `folds: [forward]` ([S] `configs/reparam_lirf.yaml:17`,
`configs/submit_v10_reparam.yaml:29`).

**Consequence:** there is no baseline OOF against which a Stage-2..6 change can be compared on
three folds. Producing one is Stage 0 and is not optional.

### 4.2 Prerequisite B — the decision instrument cannot resolve this lane

* `prc.compare.verdict` reads only the K=0 and K=200 entries of the whole-fold robustness curve
  (`compare.py:120-140`); the unmatched-lane numbers it prints are explicitly excluded from the
  verdict (`compare.py:143-152`).
* v10 itself — a 20.67 s live gain — passed at **K=0 −61.13, K=200 −0.38 [−0.63, −0.14]**
  ([S] `experiments/submissions.jsonl` v10 note). A change worth half of v10 lands at K=200 ≈ −0.19
  with a CI that will straddle zero. **[X] The gate will reject real wins in this lane.**
* Unmatched-only paired CIs are **385–724 s** on three-fold pairs
  ([S] `docs/CODEX_BRIEF.md` §6). merry-quicksand measured the Kish effective sample size of their
  unmatched stratum at **22.9, not 22,219**, and concluded that *"model selection by held-out
  stratum RMSE alone is not possible"* ([S] `results/21-...` §5 rules 7 and 10).
* Their measured row-bootstrap-vs-day-bootstrap width ratio on a tail-living arm was **4.60×**.

**Required before any stage is gated:** add a second arm to the decision rule for this lane —
`month_bootstrap_rmse_diff` (already implemented, `compare.py:84-113`) restricted to unmatched
rows, plus an MSE-denominated bar (merry-quicksand ship at +500 weighted-fold MSE, close a lever
below 1,000). Do **not** substitute it for the K=200 arm; slot 24 measured the two gates as
near-orthogonal and recommends adding the arm, not replacing.

### 4.3 Hazard 1 — the un-cross-fitted target encoding (`train.py:367-368`)

`enc = fit_encodings(train_f, cfg)` then `attach_encodings(train_f, enc)`: four label-dependent
tables are fitted on the training side and joined back onto those same rows. `reference_min_count: 30`
bounds the self-inclusion at 1/30 without removing it ([S] SUMMARY §C1, slot 20).

**Which stages this affects, and which it does not:**

| stage | affected? | why |
|---|---|---|
| 0 (instrument), 1 (cap sweep), 3 (schedule floor) | **No** | Post-hoc on predictions; no new label-derived table. |
| 2 (AUC pre-test) | **Partly** — it is the *point* | The pre-test measures a *new* label-derived quantity (the fill rate by prefix). It must be fitted leave-one-month-out from the start, exactly as `fill-pattern.md` §2.9 already does. |
| 4 (fill mixture), 5 (non-fill body), 6 (blend members) | **Yes** | Each introduces a new label-derived object (`p`, cell estimates, `nf` encodings) that must be fitted inside the fold like `fit_encodings` is, or the OOF will be optimistic by an unknown amount. |

**Do not "fix" the existing defect as a prerequisite.** Two independent measurements say the
repair is not free: merry-quicksand priced out-of-fold target encoding as a predictive change at
**−0.96 s (worse)** ([S] `results/21-...` §6.3 item 5), and kind-mango shipped both directions and
lost both times — more in-sample leak **+1.29 s live**, OOF encoders **+2.25 s live** — because
*"MD4 breaks the training-serving alignment… the tree learns split thresholds calibrated to the
OOF distribution and misapplies them at serve time"* ([S] `results/22-...` §4.3). Their conclusion
is that the training-serving-aligned, partly-in-sample map is the local optimum. **Treat the
defect as a known bias on the level of our OOF, not as a bug to repair inside this theme.** New
tables introduced by Stages 4–6 must be fitted the same way the existing ones are, so that any
comparison is like-for-like.

### 4.4 Hazard 2 — the OOF parquet has no fold label

`run_cv` writes `MVT_ID_mvt` + `oof` only (`train.py:486`, `train.py:534`). The folds carry
different `train_months` (`jan` trains on the future, `forward` does not), `pl.coalesce` resolves
overlaps silently in fold-list order (`train.py:507-510`), and `jul_fwd`'s 190,713 predictions are
discarded from `overall` while still counted in the fold table ([S] `train.py:501-510`).
A correction fitted on one fold's residuals and applied to another's is **measured to transfer
negatively (−0.45 to −1.63)** ([S] `docs/CODEX_BRIEF.md` §6, `results/review-refutations.md` C2).

**Which stages this affects:**

| stage | affected? |
|---|---|
| 1 (cap sweep), 3 (schedule floor) | **No** — a fixed rule with no fitted parameter; the pooled OOF is a legitimate evaluation set even if it mixes three models. |
| 2 (AUC pre-test) | **No** — it does not read the OOF at all. |
| 4, 5, 6 | **Yes** — any blend weight, α, isotonic calibration or shrinkage constant fitted on pooled OOF residuals is fitted across three different models. Stage 0 must add the `fold` column before any of these fit a parameter. |

---

## 5. The staged plan

Ordering rule: cheapest decisive test first. Stage 0 is a prerequisite for every other stage.
Effort figures are engineering hours, excluding compute wall-clock.

---

### Stage 0 — Pin the baseline and instrument the lane

**Why it is stage zero:** §4.1 and §4.4. Stages 4–6 cannot be specified, let alone measured,
without (a) a three-fold OOF from the model we actually ship and (b) a fold label in that OOF.
Today the only per-airport unmatched numbers in the repository come from a single ad-hoc
`prc.compare` run on the forward fold (§1.9).

**Files and functions.**
1. New `configs/main_reparam.yaml` = `configs/main.yaml` verbatim + `unmatched_sched_reparam: true`
   + `unmatched_sched_reparam_airports: [LIRF]`, with **no `folds` key** (all four folds).
   Run `make run CONFIG=configs/main_reparam.yaml`.
2. `src/prc/train.py::run_cv` — carry the fold name into the OOF frame. The join at
   `train.py:507-510` already knows which fold supplied each prediction (`n_new` counts exactly
   those rows); write a `fold` string column beside `oof` and include it in the parquet at
   `train.py:534`.
3. `src/prc/train.py::run_cv` — add an `unmatched` block to `fold_metrics[f.name]`
   (`train.py:496-500`): `n`, `rmse_full`, and `by_airport` restricted to
   `pl.col(PROXY).is_null()` rows, using the existing `per_airport` helper (`cv.py:120`). Mirror
   it with a `matched` block so the two lanes are always reported together.
4. `src/prc/compare.py` — add the unmatched-restricted `month_bootstrap_rmse_diff`
   (`compare.py:84`) to the printed report and to the JSON, and print the lane difference in
   **MSE** as well as seconds (merry-quicksand rule 4: *"seconds are not additive and MSE is"*).
   Leave `verdict()` untouched.
5. While in `visible_numeric_columns`, resolve the 84-vs-83 column-count discrepancy (§1.3) by
   logging `len(cols_b)` into `info` at `train.py:406`.

**Measurement that settles it.** The run must reproduce the forward-fold unmatched RMSE of
1,028.41 to within the known thread-count noise. mms §3.2 measured that changing `num_threads`
from 4 to 2 shifts the pooled unmatched OOF by +30.4 s (+1.3 %) and the early-stopping round count
substantially (jan 393 → 566); so "reproduces" means ±2 %, not bit-identical, unless `num_threads`
matches the logged run.

**Expected effect on the score:** zero. This is instrumentation.

**Risk.** The reparameterisation has never been run on `jan` or `jul`. mms §3.3 measured variant A
as better on all three folds (jan 3,247.5 → 2,904.6; jul 2,013.6 → 1,338.6; forward 2,177.4 →
1,024.6), but that was a script reproducing the model-B branch, not the pipeline. If the pipeline
disagrees, the disagreement is itself the finding and everything downstream waits.

**Effort:** 4–6 h engineering, plus ~30–60 min compute for the four-fold run at `n_seeds: 3`.

**How we would know this failed.** (a) The forward-fold unmatched RMSE comes out outside
1,008–1,049 with `num_threads: 4`. (b) `jan` or `jul` unmatched RMSE under the reparameterisation
comes out *worse* than the corresponding `oof_main-48c19bc830` fold figure (3,223.0 / 2,008.8) —
that would contradict mms §3.3 and means the pipeline and the script are not fitting the same
model. (c) The new `fold` column shows the pooled `overall` covering a different row set than
`folds_pooled` claims.

---

### Stage 1 — Re-sweep the non-LIRF unmatched upper cap

**Status: largely already decided, by a measurement taken while writing this plan.**

`tail-postprocessing-cap-shrink.md` §3 established, on the **pre-reparam** OOF, that capping
non-LIRF unmatched predictions at 3,600 gives total OOF **334.58 → 327.05** and unmatched
2,296.1 → 2,207.3 (53 rows affected, 49 helped / 4 hurt, SSE delta negative in each of the four
OOF months and at each of the eight affected airports), with the finer sweep best at 1,800–2,400
(total 326.57, unmatched 2,201.5, but 147 / 42 rows hurt). That is our own confirmed-but-never-shipped
instance of slot 15's per-airport upper bound (kind-mango v48: **−2.33 s live on 5 rows**).

**[S-new]** Counting the *shipped* submission instead (script
`docs/competitor_research_2026-09-21/scripts/p1_cap.py`, read-only over `outputs/v10_probe_reparam.parquet` joined to
`data/work/feat_rank.parquet`):

| threshold | non-LIRF unmatched ranking predictions above it, **v10 (reparam)** | v9 (v4's model B) | v7 (`outputs/main.parquet`) |
|---|---:|---:|---:|
| 3,600 s | **17** | 104 | 92 |
| 7,200 s | **0** | 2 | 10 |
| max prediction | **4,689 s** | 8,685 s | 13,730 s |

Total shift a 3,600-cap would apply: 6,108 s over 17 rows, max 1,089 s.
**[X]** Absolute ceiling on the board gain, assuming every capped row has label 0:
`ΔSSE ≤ Σ(p−c)(p+c) ≈ 1.5e8`, i.e. **≤ 0.77 s** of board RMSE; the realistic figure is ~0.1 s.

**Conclusion: the 3,600 cap is closed** — the reparameterisation already collected it, exactly as
mms §3.8 predicted (variant A took non-LIRF predictions ≥ 4 h from 10 to 0). What remains open is
the tighter end of the sweep, which moves a real number of rows: **558 rows above 1,800 and 293
above 2,400** in v10.

**Files and functions.** No source change. A read-only script over the Stage-0 OOF, in
`docs/.../scripts/`, sweeping `pred := min(pred, c)` for
`c ∈ {1,500, 1,800, 2,100, 2,400, 3,000, 3,600}` on non-LIRF unmatched rows, reporting per-fold
and per-airport SSE deltas and the helped/hurt counts.

**Measurement that settles it.** SSE delta negative in **all three folds and at every affected
airport**, with a month-block CI excluding zero, and helped/hurt no worse than the 49/4 that the
3,600 cap showed pre-reparam. A cap that helps 400 and hurts 150 is a distribution shift, not a
guard, and should be rejected.

**Expected effect and source.** −0.1 s at 3,600 [X, §5 Stage 1 arithmetic]; unknown but bounded by
roughly −1 to −3 s at 1,800–2,400 [X, extrapolating `tail-postprocessing` §3's 326.57 onto a model
whose non-LIRF predictions are already 3× tighter].

**Risk.** Low in loss, low in value. The failure mode is capping genuine long taxi times: the
`tail-postprocessing` sweep already recorded 147 rows hurt at 1,800.

**Effort:** 1 h (after Stage 0).

**How we would know this failed.** Any airport or fold with a positive SSE delta; or helped/hurt
below 3:1; or the month-block CI straddling zero. In any of those cases record the cap as CLOSED
with the reason, so it is not reopened a third time.

---

### Stage 2 — The fill-classifier AUC pre-test (gates Stages 4 and 5)

**What it decides.** Whether a per-row fill probability `p` good enough to beat the hard LIRF
switch exists at all. Stages 4 and 5 are expensive; this is 3 h and no fit enters the pipeline.

**What we already know about the ceiling.** Our own variant-B classifier — LightGBM `binary` on
`f10 := |BLOCK − SCHED| ≤ 10 s`, trained on the **unmatched rows only**, `cols_b` + categoricals —
reached validation AUC 0.9567 / 0.9681 / 0.9701 pooled and **0.8371–0.9177 within LIRF alone**,
with monotone calibration across six bins (observed f10 rate 0.010 → 0.900)
([S] mms §3.6). Separately, with no model at all, the flight-designator prefix alone gives
LIRF-unmatched **LOMO AUC 0.8078** and prefix × `ms`-band **0.8187**
([S] `fill-pattern.md` §2.9). kind-mango's full-year `p_fb` reaches 0.9256 in-sample
([S] `results/22-...` §1.1 v56).

**The hypothesis to test.** Training the classifier on **LIRF matched rows as well**, where the
fill is directly observable on 159,216 rows instead of 1,488, lifts the held-out LIRF-unmatched
AUC materially above 0.9177. This is the one ingredient all three of merry-quicksand,
javidmardanov and kind-mango have and we do not (§3 item 3). Supporting facts we already own:
LIRF matched fill rate is **17.8 %** (14.5–20.6 % by month), so the positive class is well
populated there; fill by operator on LIRF matched runs **54 / 25 / 22 / 19 / 14 / 11 %** against
≤ 5.6 % for the rest, and by segment Cargo 47 % / Non-Scheduled 45 % / Lowcost 22 % / Mainline 14 %
([S] `lirf-matched-fill.md` §1 S1, S3).

**Files and functions.** A standalone script in `docs/competitor_research_2026-09-21/scripts/`.
No `src/` change. It must:
1. Build the fill label `f10 = |BLOCK_TIME_UTC_mvt − SCHED_TIME_UTC_mvt| ≤ 10 s` on all LIRF
   training rows (both columns are in `data/work/feat_train.parquet`; `BLOCK_TIME_UTC_mvt` is in
   `KEEP_RAW_COLS`, `features.py:260`, and is excluded from features by `BLANKED_DEP_COLS`,
   `schema.py:52` + `features.py:467`, so using it as a *label* is safe and using it as a
   *feature* is already impossible).
2. Derive `flight_prefix = FLIGHT_mvt.str.slice(0,3)` and `flight_identity = FLIGHT_mvt` —
   available on ranking unmatched rows at 0.43 % null [S-new §1.3], and 358 of the 383 ranking
   LIRF unmatched rows carry a prefix seen in training LIRF unmatched
   ([S] `fill-pattern.md` §2.9).
3. Fit three classifiers under **leave-one-month-out**, evaluated on held-out LIRF **unmatched**
   rows only: (a) our current recipe, unmatched rows only; (b) the same on LIRF matched +
   unmatched with the unmatched rows up-weighted ×8 (javidmardanov/merry-quicksand); (c) (b) plus
   the stake weight `clip(1 + (max(ms,0)/3600)², 1, 25)` and the prefix features.
4. Report AUC, log-loss, and calibration by bin, plus the AUC restricted to `ms ≥ 2 h` — the band
   where the decision is expensive (mms §3.7: 239 OOF rows, 129 filled / 110 not).

**Measurement that settles it.** Held-out LIRF-unmatched AUC of recipe (c). **Threshold: 0.93.**
Rationale [X]: mms §3.3 measured the classifier mixture at 1,926.1 against the reparameterisation's
1,632.1 with an AUC of 0.84–0.92. Closing a 294 s gap requires a materially better gate, and
kind-mango's shipped gate — the one worth −9.8 s live — sits at 0.9256. Below 0.93, Stages 4 and 5
do not start.

**Expected effect.** No score effect; it is a gate. The expectation that it passes rests on the
100× increase in labelled positives, which is [X].

**Risk.** The LIRF matched and unmatched populations may not share the mechanism. Direct evidence
against the null: mms §3.1 reports LIRF **matched** corr(y, ms) = 0.2985 against 0.8964 unmatched,
and `AIRCRAFT_TYPE_mvt` is 0 % null on LIRF matched against 99.93 % on LIRF unmatched [S-new] —
the two groups are visibly different populations. The classifier must therefore be checked for a
train/serve distribution break on its own inputs (kind-mango's `check_coverage.py`, ~90 lines,
[S] `results/22-...` §6.1).

**Effort:** 3 h.

**How we would know this failed.** (a) Held-out LIRF-unmatched AUC of (c) below 0.93 — stop, and
record Stages 4–5 as CLOSED with the AUC as the reason. (b) (c) beats (b) only because of the
stake weight, with prefix features contributing nothing — then the mechanism is not the one
merry-quicksand and kind-mango describe and the transfer argument is void. (c) A permuted-prefix
negative control keeps more than ~20 % of the AUC lift (merry-quicksand's own control on an
arrival-side tell kept 71 % and killed the arm, [S] `results/21-...` §3.5).

---

### Stage 3 — The LIRF schedule-floor overlay on the high-`ms` band

**The invariant, verified on our data.** [S-new] (script `docs/competitor_research_2026-09-21/scripts/p1_floor.py`, read-only over
`data/work/feat_train.parquet`), testing merry-quicksand's justification *"in 2025 every LIRF
unmatched row with `sp ≥ 24,000` had `y ≥ sp − 60"`:

| band, LIRF unmatched training rows | n | rows violating `y ≥ ms − 60` | min(y − ms) |
|---|---:|---:|---:|
| `ms ≥ 14,400` | 127 | **16** | −21,961 |
| `ms ≥ 18,000` | 86 | 5 | −21,961 |
| **`ms ≥ 24,000`** | **56** | **1** | −6,365 |
| `ms ≥ 28,800` | 50 | 1 | −6,365 |
| non-LIRF unmatched, `ms ≥ 14,400` | 415 | **415** | — |
| non-LIRF unmatched, `ms ≥ 24,000` | 110 | **110** | — |

So the rule reproduces on our data at 55/56, their threshold of 24,000 is the right one, and it is
**strictly LIRF-only** — at the other nine airports it fails on every row, which is the same
airport-specificity that mms §3.3 measured for the reparameterisation (variant A2, all airports:
2,696.7 against 2,326.5).

**What it would move.** [S-new] On the ranking set there are **25** LIRF unmatched rows with
`ms ≥ 24,000`, of which **23** have `24,000 ≤ ms < 86,400`; v10 predicts **below** `ms` on **8** of
them, median gap **886 s**.

**Files and functions.** This is a post-hoc overlay, not a model change. Either a new function in
`src/prc/predict.py` applied after `predict_fold` returns, or — preferred for a first probe — a
standalone script producing a submission parquet from `outputs/v10_probe_reparam.parquet` with
only those 8 rows changed, plus the "rows outside the gate are bitwise unchanged" assertion that
javidmardanov make after every correction
([S] `results/08-...` Tier 5 item 14: *"they caught real bugs with this"*).

**Measurement that settles it.** On the Stage-0 three-fold OOF restricted to LIRF unmatched
`24,000 ≤ ms < 86,400`: number of rows where the floor binds, SSE delta, and the delta's sign in
each of the three folds separately. Because the ranking gate is 8 rows, a submission probe is also
legitimate and is the only instrument with real power — but see the risk.

**Expected effect and source.** [X] 8 rows × a median 886 s upward correction, on rows whose
labels satisfy the floor 55/56 of the time in training. Board arithmetic: if all 8 are genuine,
`ΔSSE ≈ 8 × 886² ≈ 6.3e6` ⇒ about **−0.03 s**. If the rows that bind are the *large*-gap ones the
effect is superlinear; the honest statement is that this is a **sub-1-second** change on its face.
The reason to consider it at all is that slot 24's physics-identity lower bound — structurally the
same kind of overlay — measured **−20.3 s live** on 11,826 rows
([S] SUMMARY §G), and slot 15's upper bound **−2.33 s live on 5 rows**.

**Risk — this is the highest-variance stage in the plan.** **v11 is the precedent**: a rule on
**11** LIRF unmatched rows with `ms ≥ 43,200` cost **+9.62 s** of board RMSE
([S] `experiments/submissions.jsonl` v11). **[X]** That is ≈ 1.9e9 of added SSE over 11 rows.
The same population is in scope here (11 of the 25 have `ms ≥ 43,200`). A floor is a *weaker*
intervention than v11's mixture — it only raises predictions that are already below `ms`, and the
training data says those are wrong 55/56 of the time — but the row count is the same order and the
per-row leverage is ~1 s of board RMSE.
**Recommendation: restrict the gate to `24,000 ≤ ms < 43,200` on the first attempt**, which
excludes every row v11 touched, and do not spend a submission on it alone — fold it into a
submission that also carries Stage 1.

**Effort:** 4 h.

**How we would know this failed.** (a) The SSE delta on the OOF LIRF `ms ≥ 24,000` band is
positive in any fold. (b) The bitwise-unchanged assertion fires on any row outside the gate.
(c) Shipped and the board moves the wrong way by more than the 8-row leverage can explain — which,
given v11, means anything worse than about +1 s.

---

### Stage 4 — Replace the hard LIRF switch with a soft fill mixture

**Gated by Stage 2 passing.** This is the theme's centre.

**The argument.** Our shipped reparameterisation is the `p ≡ 1` corner of the mixture at LIRF: it
predicts `ms + g(x)` where `g` is fitted on `y − ms`. On a filled row `y − ms = 0` exactly
([S] `shared-b-no-transfer.md` §3A: `y − ms + (BLOCK − SCHED) = 0` on every one of 2,085,047 rows),
so the model only has to output 0 — which is why it works. On a non-filled row it must output a
large negative correction, and the correction is **discontinuous**: mms §3.1 measures corr(y, ms)
= 1.0000 on the 721 filled LIRF unmatched rows and Spearman **0.0685** on the 767 non-filled ones.
A GBDT asked to straddle that discontinuity spends its capacity on the switch. The mixture moves
the switch out of the regressor and into a classifier, which is the form all four competitors
converged on.

**Where the remaining error actually is, after the reparameterisation** ([S] mms §3.7, 239 LIRF
unmatched OOF rows with `ms ≥ 2 h`):

| subset | n | y median | ms median | RMSE under variant A |
|---|---:|---:|---:|---:|
| filled | 129 | 10,560 | 10,560 | 2,158 |
| **not filled** | **110** | **1,112** | **9,838** | **8,293** |
| rollover (`y ≥ 72,000`), separately | 9 | — | — | 34,157 |

The 110 non-filled high-`ms` rows are the target. Every variant tested in mms still predicts
5,000–7,200 on them against a median label of 1,112.

**Files and functions.**
1. `src/prc/train.py` — a new `fill_gate_table(train_dep, cfg)` fitted **inside the fold**,
   alongside `fit_encodings` (`train.py:200-207`), returning a fitted classifier rather than a
   join table. It must be called from `predict_fold` after line 367 so it sees only the training
   side.
2. `src/prc/train.py::predict_fold` model-B branch (`train.py:398-422`) — after computing `g`,
   apply
   ```
   p = gate.predict_proba(u_tg)[:, 1]
   p[~(ADEP == "LIRF") | (mvt_minus_sched <= 0)] = 0.0     # hard outer gate
   pred = p * mvt_minus_sched + (1 - p) * body
   ```
   The hard outer gate is javidmardanov's (`missing_v2.py:226`) and is what keeps the nine other
   airports bitwise unchanged — the design constraint mms §3.3 established when variant A2
   (the shift everywhere) took EHAM from 954.7 to 3,408.9.
3. New `TrainConfig` fields: `unmatched_fill_mixture: bool = False`,
   `unmatched_fill_gate_airports: tuple[str, ...] = ("LIRF",)`, and the classifier's own
   hyperparameters. Default off, exactly as `unmatched_sched_reparam` is (`train.py:108`).
4. New `configs/fill_mixture.yaml` = `configs/main_reparam.yaml` + the flag, `folds: [forward]`
   for screening and all folds for the decision.

**Classifier specification**, taking the intersection of what three teams shipped:
* Label `f10 = |BLOCK − SCHED| ≤ 10 s` (our own definition, verified against the identity on all
  22,470 rows, [S] mms §0). javidmardanov use ≤ 6 s, merry-quicksand and kind-mango ≤ 60 s;
  `fill-pattern.md` §2.7 notes the ±10 s threshold is biased by a recording grid, so the
  sensitivity to 6 / 10 / 60 should be reported, not assumed.
* Training rows: **all LIRF departures, matched and unmatched** (Stage 2's recipe (c)).
* Sample weight: `clip(1 + (max(ms,0)/3600)², 1, 25) × 8 if unmatched`
  ([S] `results/21-...` §4.5 arm R; `results/08-...` §4.2).
* Features: `cols_b` + categoricals + **`flight_prefix`** + `dayoff = clip(date(MVT) − date(SCHED), 0, 1)`
  + `log1p(max(ms, 0))`.
* Isotonic calibration on the inner holdout (kind-mango; our mms §3.6 calibration was already
  monotone without it, so measure whether it earns its place).

**Measurement that settles it.** Unmatched OOF RMSE on the Stage-0 three folds against
`main_reparam`, reported in MSE, with:
- the **LIRF** and **non-LIRF** splits separately — a non-LIRF regression of any size is a stop,
  because variant A's non-LIRF gain (1,451.7 → 1,276.7, mms §3.3) is a side effect that must
  survive;
- the 110 non-filled LIRF `ms ≥ 2 h` rows as a named subset;
- the month-block bootstrap of §4.2;
- the counts of predictions ≥ 4 h and ≥ 12 h, which mms §3.8 tracks (variant A: 42 and 15, against
  53 and 17 labels, and **0** non-LIRF ≥ 4 h).

**Target to beat:** pooled unmatched **1,632.1** (mms §3.3 variant A). The mixture's measured
predecessors on the same rows were 1,926.1 (B) and 1,820.6 (B2), so **this stage must find ≈ 200 s
that the earlier classifier did not**, and Stage 2's AUC is the direct test of whether it can.

**Expected effect and source.** [X] If the mixture reaches 1,400 pooled unmatched on this OOF, and
the OOF-to-board ratio of the reparameterisation carries over (OOF 2,326.5 → 1,632.1 = −30 % came
with a board −20.67 s), the board effect is of order **−6 to −10 s**. Cross-check against
merry-quicksand: the LIRF rules lane plus arm R is about a third of their 15 s of post-v10 gain,
i.e. ~5 s ([S] `results/21-...` §6.2). Both routes land in the same place; plan on **−5 s**.

**Risk.** Three named, all with evidence:
1. **kind-mango's naive-mixture failure**: *"uniform mixture everywhere FAILED at LIRF
   (+5.3 G MSE) because `R_norm` underestimates on high-`sd` rows and soft mix pulls all rows
   toward `sd`."* Our body has the same property (mms §3.7: every variant predicts 5,000–7,200 on
   rows whose labels median 1,112). **Mitigation: Stage 5 fixes the body; consider running 5
   before 4, or together.**
2. **The oracle result.** A *perfect* fill flag plus the current body gives only 2,203.3 against
   the base's 2,326.5 (mms §3.3) — worse than variant A. merry-quicksand measured the same thing
   independently: oracle flag + constant non-fill scored 1,868 → **3,081, worse**
   ([S] `results/21-...` §3.5). **The gate is not sufficient; the body is the binding constraint.**
   This is the single strongest argument for reordering 5 ahead of 4.
3. **Coefficient instability.** javidmardanov's α on this exact population moved 0.377 / 0.939 /
   0.659 / 0.296 across four fits. Do not fit a blend α; use the classifier probability directly,
   as merry-quicksand and kind-mango do.

**Effort:** 10–14 h, plus 3 × ~15 min compute per screening run.

**How we would know this failed.** (a) Pooled unmatched worse than 1,632.1 on the same three
folds. (b) Any non-LIRF airport worse by more than its Stage-0 fold-to-fold spread. (c) The count
of non-LIRF predictions ≥ 4 h rises above 0. (d) The gain is concentrated on the 9 rollover rows —
check by re-running with those rows excluded from the metric, as mms §3.7 does. (e) A prefix-permuted
negative control keeps most of the gain.

---

### Stage 5 — The non-fill body: remove the fill rows, winsorise the fit, hand the tail to cells

**This may deserve to run before Stage 4** — see Stage 4 risk 2.

**The three teams agree on the mechanism.** merry-quicksand's body regressor *"refuses to accept
fill rows: a fill row's label is `sp`, thousands of seconds, and would poison every encoding and
the fit"*; javidmardanov's normal expert trains on `~identity & (y < 12000)`; satam2's residual
model trains only on `status == "present"`. **Model B trains on all 22,470 rows with fill and
non-fill mixed** (§3 item 4).

**Our own direct measurement of the ingredient.** mms §3.3: mixing with a non-fill-only regression
(B2, 1,820.6) beats mixing with the all-rows regression (B, 1,926.1) by **105 s** — and the
non-fill regression *alone* is worse than baseline (2,646.2), which is exactly the point: it is a
body, not a predictor.

**Files and functions.**
1. `src/prc/train.py::predict_fold`, model-B branch (`train.py:407-411`) — filter `u_tr` to
   non-fill rows before `_fit` when the new flag is on. The fill flag is computable on training
   rows from `BLOCK_TIME_UTC_mvt` and `SCHED_TIME_UTC_mvt`, both present in the feature frame and
   both excluded from the feature matrix (`schema.py:52`, `features.py:467`).
2. Winsorise the **fit target only**, at ±3,000 s (merry-quicksand) or cap at 12,000 s
   (javidmardanov). This is a two-line change inside the `target_expr` lambda passed to `_fit`
   (`train.py:409`), and it must not touch the evaluation target.
3. `nf_hybrid`: a hierarchical shrunk cell estimate over `[ADEP] → [ADEP, ms_band] →
   [ADEP, ms_band, dayoff]` with `K_SHRINK = 5.0` and
   `SP_EDGES = [-inf, 900, 1800, 3600, 7200, 10800, 21600, 43200, inf]`, fitted inside the fold
   like `fit_encodings`; final body = `where(cells < 3000, fitted, cells)`.
4. Optionally the congestion witness (§3 item 7): airport × UTC-hour median `MVT − AOBT_3` over
   that hour's **matched** departures, n ≥ 5, unmatched rows excluded from the aggregate. We
   already have a lagged rolling mean of the same quantity
   (`features.py:200-208`, `nbr_mvt_minus_aobt_mean_{30,60}m`), so measure the *replacement* and
   the *addition* separately — this is not a fresh idea for us, it is a different estimator of
   something we already compute.

**Measurement that settles it.** Same instrument as Stage 4. In addition, report merry-quicksand's
own control: **a plain gradient booster in place of the cell estimator scored 1,793.6 against
1,648.4** ([S] `results/21-...` §4.5). If our cell/GBDT comparison comes out the other way, the
heavy-tail argument does not hold on our data and the cell layer should be dropped.

**Expected effect and source.** [S] merry-quicksand's shipped E3C body change was board
**−1,775 MSE**; their RED_TEAM ancestor moved the stratum 1,574.5 → 1,527.2, **12/12 months,
seed-stable**. [S] mms §3.3's B2-vs-B gap is −105 s of unmatched RMSE on our data. **[X] Plan on
−2 to −4 s of board RMSE, and on it being the enabler that makes Stage 4 reach its number.**

**Risk.** Removing the fill rows removes 721 of 1,488 LIRF unmatched training rows ([S] mms §3.1),
leaving 767. The regressor may not have enough LIRF rows left; javidmardanov handle this by
training the normal expert on 200,285 rows across airports with the missing rows up-weighted ×8,
not on the unmatched rows alone. Follow that: keep the body global, up-weight unmatched.

**Effort:** 10–14 h.

**How we would know this failed.** (a) The non-fill body, mixed at the Stage-4 gate, is worse than
the all-rows body at the same gate — that contradicts mms §3.3's B2-vs-B and means the fill rows
were carrying information, not noise. (b) The cell layer is worse than the GBDT on the tail — drop
it and say so. (c) Winsorising the fit moves the OOF by less than the seed spread, in which case
it is free but not a result.

---

### Stage 6 — A second and third member on the unmatched route, at fixed weights

**Files and functions.** In `predict_fold`'s model-B branch, after the body is produced, fit two
further models on the same `u_tr` and combine at **fixed, unfitted** weights:
* **normalized** (satam2): LightGBM `lr .05, num_leaves 31`, target
  `Z = (y − 900)/sqrt(3600² + (ms − 900)²)` with sample weights `scale²/mean(scale²)`,
  reconstruct `900 + scale·Z`. MSE-neutral by construction. `scale = 3600` where `ms` is missing.
* **ExtraTrees** (satam2): `n_estimators=300, max_features=0.7, min_samples_leaf=1,
  criterion='squared_error', bootstrap=False`. Leaf 20 lost on all six of their panels.
* Weights `0.5625·body + 0.1875·ET + 0.25·normalized`, **not fitted** — satam2's `compose.py:23-29`
  averages per-fold calibration weights and never fits on the scored set.

**Measurement that settles it.** Same instrument. Each member ablated separately, as satam2 report
(remove normalized 271.074; remove ET 271.185; remove both 275.704, against 268.663 shipped).

**Expected effect and source.** [S] **−7.04 s seasonal** for the pair, on 1.52 % of rows
(`results/23-...` §3.5).

**Risk — this is the stage most likely to be rejected by our own gate, and it should be.**
satam2's own warning: *"Removing five component-supporting rows per fold leaves only about 0.011 s
of normalized benefit and reverses the ExtraTrees benefit."* That is precisely the tail-lottery
pattern `prc.compare.verdict` exists to reject (`compare.py:113-140`). Also: slot 11 puts the whole
sourced band for cross-library blending at 0.3–1.8 % over the best single model
([S] SUMMARY §F), and PRC 2025's rank 1 was a single LightGBM.
Also **[S]**: *"An analogous finite-clock formulation was negative"* — do not put the normalized
target on the matched lane.

**Effort:** 8 h.

**How we would know this failed.** (a) The K=200 arm rejects it — which, given satam2's own
five-row sensitivity, is the expected outcome, and the correct response is to record it CLOSED
rather than to weaken the gate. (b) The ExtraTrees member's contribution reverses when the worst
20 unmatched rows are dropped. (c) The fixed weights are worse than the body alone on any fold.

---

### Stage 7 (optional hygiene, no expected score) — label tables fitted over a regime model B never serves

kind-mango's MB3 was worth **−5.99 s live at zero offline price** because the base was being
*trained* on 160,704 rows the serving stack always overwrote
([S] `results/22-...` §4.2). Our two models are cleanly separated, so the headline defect does not
apply — but `fit_encodings` (`train.py:200-207`) is fitted on the **whole** labelled training side,
matched and unmatched together, and `airport_runway_hour_table` (`train.py:175-182`) and
`airport_median_table` (`train.py:195-197`) filter only on `LABEL.is_not_null()`. They therefore
aggregate the LIRF unmatched rows whose labels run to 87,605 s into medians that are then served to
both models. `reference_table` (`train.py:158-173`) is protected by its `0 < y < 7200` filter; the
other two are not.

Medians are robust, so the expected effect is near zero — but the check is cheap and the mechanism
is the one kind-mango measured. **Effort: 4 h. How we would know it failed: the OOF moves by less
than the seed spread, in which case record it as hygiene, not as a result.**

---

## 6. What this plan deliberately does not do

* **Does not generalise the LIRF shift to other airports.** Measured: 2,696.7 against 2,326.5,
  EHAM 954.7 → 3,408.9 ([S] mms §3.3 variant A2; `train.py:109-113` records it in the source).
* **Does not build per-airport heads at the nine non-LIRF airports.** kind-mango shipped exactly
  that architecture there and it failed live twice (+195 and +5), with the diagnosis that at those
  airports the positive class *"models punctuality, not reporting failure"*
  ([S] `results/22-...` §4.4). Slot 12 and slot 04 add that four PRC 2025 teams shipped per-airport
  specialists and none beat the global model ([S] SUMMARY §F).
* **Does not touch the matched lane.** §1.9 arithmetic says our non-LIRF unmatched lane is
  competitive and merry-quicksand's §6.2 says our matched lane at v10 was *comparable or slightly
  better* than theirs. The matched tail is a separate theme with a separate diagnosis (83 % of
  `|residual| > 1,000 s` rows have `BLOCK − AOBT_3 < −600`, and *"if any column identified
  slot-held flights it would be worth up to ~25 s — none does"*, [S] `results/21-...` §3.8).
* **Does not repair the un-cross-fitted encoding.** §4.3.
* **Does not add external data.** [S] SUMMARY §F: the largest quantified external-data gain in the
  entire public record is about −3.4 s.

---

## 7. Facts and inference, kept apart

**[S] — read from a source.** Every competitor number in §0 and §2, each attributed to its slot
result file and through it to a file:line in the cloned repository. Every description of our own
code in §1 and §5, cited to `src/prc/*.py` line numbers. Every measurement quoted from
`docs/investigation_2026-09-16/`, `docs/data_audit_2026-09-18/` and `experiments/submissions.jsonl`.

**[S-new] — measured while writing this plan, read-only, no code modified.**
(a) §1.3, the categorical null rates on unmatched rows, including `AIRCRAFT_TYPE_mvt` at 0.9993 on
LIRF unmatched and `FLIGHT_mvt` present with 129 LIRF-unmatched prefixes.
(b) §5 Stage 1, the counts of non-LIRF unmatched ranking predictions above each threshold in
`outputs/v10_probe_reparam.parquet` (17 above 3,600; 0 above 7,200; max 4,689) against v9 and v7.
(c) §5 Stage 3, the schedule-floor invariant on our training data (1 violation of 56 at
`ms ≥ 24,000` at LIRF; 110 of 110 at non-LIRF) and the 25 / 23 / 8 ranking row counts.
Scripts: `docs/competitor_research_2026-09-21/scripts/{p1_cols,p1_cap,p1_floor}.py`. They read
`data/work/feat_{train,rank}.parquet` and `outputs/*.parquet` and write nothing.

**[X] — inference and arithmetic, mine.**
The board sensitivity of 8.0 s per 100 s of unmatched RMSE, and the −25 s / −7 s framing in §0.
The LIRF 75 %-of-unmatched-SSE decomposition and the 527.0 non-LIRF pooled figure in §1.9.
The v11 per-row leverage of ~1 s of board RMSE in §1.8 and §5 Stage 3.
The 0.77 s ceiling on the 3,600 cap in Stage 1.
The 0.93 AUC threshold in Stage 2 and its justification.
The claim that Stage 5 may be the binding constraint on Stage 4 — it rests on two oracle
measurements (ours at 2,203.3, theirs at 1,868 → 3,081) that both say a perfect gate over the
current body is not enough, but no experiment has been run that isolates gate quality from body
quality on our data.
All expected-effect figures in §5 that are not quoted from a slot file.

**What cannot be specified without a measurement we do not have.**
The exact numerical target for Stages 4–6 on the `jan` and `jul` folds, because the shipped model B
has never been cross-validated on them (§4.1). The per-fold, per-airport unmatched baseline that
every "how we would know this failed" condition is stated against. Both are produced by **Stage 0**,
which is why it is stage zero.
