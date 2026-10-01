# SLOT 08 — repo-javidmardanov (team `elegant-alligator`), official RMSE 268.5583

Source: <https://github.com/javidmardanov/PRC-Data-Challenge-2026>, cloned at commit `845d0d1`
(2026-09-15 17:57 -0400), read in full. 71 Python scripts / 6,508 LOC, 10 commits, all between
2026-09-08 and 2026-09-15. All file paths below are repo-relative to that clone; line numbers are
from that commit.

Marking convention used throughout: **[S]** = stated in or directly readable from the repo (file +
line cited). **[X]** = my inference, arithmetic, or judgement — *not* claimed by the repo.

> **Team-name note [S]:** the GitHub account is `javidmardanov` but the competition team is
> `elegant-alligator` (`docs/results.json:29`, submission filenames `elegant-alligator_v{1..6}.parquet`).
> Our one-line prior [S] said "TimesFM airport series + CatBoost residual + missing-timestamp mixture
> of experts + winter-only Rome flight-identity correction." That is accurate but it *omits the single
> most important mechanism*, which is the AOBT residual baseline (§2). Everything else in this repo is
> a correction on top of that.

---

## 0. Headline facts

| Fact | Value | Source |
|---|---|---|
| Best official RMSE | **268.5583** (V6), rank 14/147 at 2026-09-15 21:56:11 UTC | `docs/autoresearch_attempts.md:6-7`, `docs/autoresearch_completed.json` |
| Official submission history | v1 285.0706 → v2 274.3377 → v3 271.0449 → **v4 271.5069 (worse, rejected)** → v5 269.0799 → v6 268.5583 | `docs/results.json:35`, `README.md:31-41`, `docs/autoresearch_attempts.md:37-124` |
| Local (Jul+Nov 2025) RMSE | v1 266.0522 → v2 262.5264 → v3 259.9769 → v4 258.0149 → v5 257.7546 → v6 **257.1287** | `docs/v1/v2/v3_ensemble.json`, `docs/autoresearch_v{4,5,6}_review.json` |
| Local→official gap | ~11.4 s at V6 (257.13 local vs 268.56 official) | [X] arithmetic from the two rows above |
| Leader at their last snapshot | 245.0207 | `docs/autoresearch_attempts.md:8` |
| Hardware | single RTX 3060 12 GB, Windows, Python 3.12, CatBoost 1.2.10, LightGBM 4.7.0, PyTorch 2.7.0+cu118 | `docs/solution.md:78-80`, `docs/research_program.md:52` |
| Process | Karpathy-`autoresearch`-style loop: 3 "GPT-5.6-sol" worker agents + 1 reviewing parent, file-backed trial logs | `docs/research_program.md:1-40` |

---

## 1. The whole pipeline in one place

V6 = V3 + LOBT-residual expert (airport-specific coefficients) + winter-only Rome identity correction.
V3 = V2 + three gated deltas. V2 = V1 + seasonal blend weights + missing-v2 MoE + tail delta. So the
apply chain, in the exact order the code executes it, is: [S] `scripts/timesfm_blend_research.py:27-40`
(`routed()`), `docs/v3_ensemble.json:"apply_order"`, `scripts/assemble_autoresearch_attempt.py:24-42`.

```
1. base  = Σ w_i · model_i                      # 4 tree models, seasonal simplex weights
2. base[missing AOBT] += 1.0·(v1_missing_specialist − base)      # alpha = 1.0 (v1_ensemble.json:14)
3. base[LIRF, sched<24000] += β·(rome_expert − base)             # β_known .16360, β_missing .90481
4. base  = max(base, 0)
5. base  = clip(base, MVT−LOBT−3606, MVT−LOBT+3606)              # timing bounds
6. base[missing AOBT] += α_region·(MoE_raw − base)               # α: LIRF .42241, other 1.0
   base[LIRF & sched≥24000] = stored v1 value                    # 25 protected ranking rows
7. base[tail gate]        += 0.5441252·Δ_catboost_tail  → clip to bounds
8. base[known & !LIRF]    += 1.0·Δ_arrival_model        → clip to bounds
9. base[known AOBT & known LOBT] += α_airport·(LOBT_expert − base) → clip to bounds   # V5/V6
10. base[missing AOBT & LIRF & month==1 & sched<24000] += 0.87886·(identity_raw − base)  # V6, 97 rows
```

Every step is a *gated convex nudge toward a separately-fitted expert*, with the coefficient fitted by
closed-form bounded least squares on calibration dates and every gate verified to leave non-gated rows
bitwise unchanged (`scripts/assemble_autoresearch_attempt.py:43-45`,
`scripts/catboost_apply_v3.py:71-72`). This "never replace, always shrink toward, and prove the rest
didn't move" discipline is arguably the most transferable thing in the repo. [X]

---

## 2. The mechanism the one-liner missed: the AOBT residual baseline

**This is the engine. TimesFM is a rounding error next to it.** [X]

[S] The target is defined by the organizers as `TAXITIME_SEC_mvt = MVT_TIME_UTC_mvt − BLOCK_TIME_UTC_mvt`
(`docs/challenge_brief.md:10-13`), and `BLOCK_TIME_UTC_mvt` is blank for every ranking departure.

[S] But `AOBT_3_flt` — "Actual off-block time for the flown (M3) trajectory", from the Network Manager
flight table — **is present in `ranking.parquet`** (`docs/data_dictionary.md:42`, `docs/data_dictionary.md:64`:
"ranking.parquet has the same 30-column schema as training"). The models therefore regress the residual
against `MVT_TIME_UTC_mvt − AOBT_3_flt`:

```python
# scripts/train.py:199-205
if args.residual:
    base = x['mvt_minus_AOBT_3_flt'].to_numpy().copy()
    missing = base < -100000
    base[(base < 0) | (base > 172800)] = 900          # 900 s fallback
    use_schedule = missing & x.ADEP_mvt.eq('LIRF').to_numpy()
    base[use_schedule] = x.loc[use_schedule, 'mvt_minus_SCHED_TIME_UTC_mvt'].to_numpy()
    base[(base < 0) | (base > 172800)] = 900
```

[S] The repo is explicit about what this means: *"This is reconstruction from supplied movement records,
not a live pre-departure forecast"* (`docs/solution.md:36-37`); *"Centered traffic windows and arrival
observations are available in ranking inputs"* (`docs/solution.md:37`).

[S] `scripts/queue_features.py:79-91` (`verify_development_bound`) asserts that on all non-{7,11,12}
training months, `|(MVT − LOBT) − TAXITIME| ≤ 3606` for **every** row — i.e. the supplied `LOBT_flt`
alone pins the answer to a ±3606 s window, with zero violations. Those bounds are shipped as
`data/processed/timing_bounds.parquet` and applied as a hard clip (`scripts/ensemble.py:153-167`).

**[X] Consequence for us:** the entire competition is a two-population problem.

- **Known-AOBT rows (~98.5 %)**: near-deterministic. v1 known-row RMSE 226.85
  (`docs/research_catboost.json:"baseline_known_rmse"`).
- **Missing-AOBT rows (~1.5 %)**: catastrophic. 5,294 of 353,045 Jul+Nov rows, RMSE **1157.63**,
  SSE 7.0945e9 out of a total 2.4990e10 (`docs/missing_v2.json` → `v1`/`missing_v1` blocks).
  **[X] That is 28.4 % of all squared error from 1.5 % of rows.**
- And it is worse than that: `docs/missing_v2.json:"missing_sse_concentration"` reports that the top
  **100** missing rows carry 73.9 % of the missing SSE, and the single worst row carries 8.0 % of it.
  **[X] ⇒ ~21 % of the entire competition SSE lives in about 100 flights.**

[S] Confirming the split in the ensemble config: the 4-model blend alone scores pooled 362.72
(`docs/v1_ensemble.json:"base_validation"`), and applying the missing-row specialist at α=1 plus the
Rome expert plus clipping brings it to 266.05 (`docs/v1_ensemble.json:"validation"`). **[X] 96.7 s of
the v1 score came purely from how 5,294 rows were handled.**

[S] Ranking-set counts: 5,290 ranking rows have missing AOBT (`docs/research_moe_final.json:"ranking_missing_rows"`);
339,551 of 344,841 ranking rows have both AOBT and LOBT (`docs/autoresearch_v6_review.json:"ranking_known_gate_rows"`).
**[X] So 5,290 / 344,841 = 1.53 % of the scored set is the high-error population.**

---

## 3. The TimesFM component — precisely

All from `scripts/timesfm_features.py` and `docs/timesfm_notes.md`.

**What series.** [S] One univariate series **per airport**, at **hourly** UTC resolution. The quantity
forecast is the **mean departure taxi-out seconds at that airport-hour**:
`target_sum/target_count` where `target` is `TAXITIME_SEC_mvt` over `PHASE_mvt=='DEP'` grouped by
`(ADEP_mvt, floor('h'))` (`scripts/timesfm_features.py:32-45`, `docs/timesfm_notes.md:8`). 10 airports ⇒
10 series.

**Covariates.** [S] Eight known-in-advance covariates spanning context **and** horizon
(`scripts/timesfm_features.py:15-16`, `59-63`, `docs/timesfm_notes.md:21-24`):
`proxy` (= mean `MVT_TIME_UTC_mvt − AOBT_3_flt` clipped 0–14400), `arrival_taxi` (mean arrival
`TAXITIME_SEC_mvt`), `departures` count, `arrivals` count, and `hour_sin/hour_cos/dow_sin/dow_cos`.
Nine total channels including the target, deliberately under the evaluator's 32-channel subsampling
threshold (`docs/timesfm_notes.md:49-50`).

**Context / horizon.** [S] Context = **512 hours**; horizon = **one whole calendar month**
(`scripts/timesfm_features.py:66-85`, `--context` default 512 at line 112). One model call per
(airport, month).

**Pretrained or fine-tuned.** [S] **Zero-shot, pretrained, frozen.** Checkpoint
`google/timesfm-3.0-pytorch` revision `43046b85ec22d584a13f8098c2ed39c889e129c2`, official source at
tag `v3.0.0` (`scripts/timesfm_features.py:13-14`, `docs/timesfm_notes.md:3-6`). Inference flags:
`per_core_batch_size=1`, `use_symmetric_averaging=False`, `make_positive=True`, `sort_quantiles=True`,
`use_znorm=False`, `padding_mode="none"` (`scripts/timesfm_features.py:161-164`). No fine-tuning code
exists anywhere in the repo; `docs/research_program.md:80` states "Real inference already done; no
fine-tuning claim." [X] I confirmed by grep that no training/optimizer step touches TimesFM.

**How it is used downstream — as 12 tabular features, never as a base prediction.** [S] The output
`data/processed/timesfm_hourly_{dev,final}.parquet` contains `tfm_q1..tfm_q9`, `tfm_median`,
`tfm_decile_mean` (equal-weight mean of the 9 deciles), `tfm_width` (= q9 − q1)
(`scripts/timesfm_features.py:168-175`). These are left-joined onto departures on
`(ADEP_mvt, MVT_TIME_UTC_mvt.floor('h'))` (`scripts/train.py:174-177`) and handed to CatBoost/LightGBM
as ordinary numeric columns. The *base prediction* in every residual model is the AOBT proxy, not the
TimesFM forecast (`scripts/train.py:199-205`). [S] The repo notes the median "need not minimize RMSE"
(`docs/timesfm_notes.md:43-44`).

**Leakage discipline (genuinely careful).** [S] Development mode globally excludes July, November and
December 2025 departure labels from *all* histories, including the histories used for training months.
When a context window would cross an excluded month it walks back to the last permitted contiguous
block (August uses June, November uses October, December uses October) —
`scripts/timesfm_features.py:72-80`, `docs/timesfm_notes.md:12-18`. A runnable mutation check
(`scripts/timesfm_features.py:88-104`) asserts that overwriting every target from 2025-07 onward cannot
change the July-2025 model inputs. January 2025 gets no TimesFM feature (no prior history) —
`docs/timesfm_notes.md:19`, which is why they could never validate a January-specific recipe
(`docs/v2_ensemble.json:"january_policy"`).

**Seasonal retrieval hack for ranking July 2026.** [S] July 2026 has no 2026 history (the labeled data
is 2025 only). They feed the **last 512 hours of June 2025** as context, then the *actual July 2026*
covariates, with context calendar covariates keeping their real 2025 dates
(`scripts/timesfm_features.py:72`, `docs/timesfm_notes.md:31-36`). January 2026 uses real history
through December 2025.

**Cost.** [S] `docs/timesfm_notes.md:56-60`: on one RTX 3060, **110 development forecasts = 46.63 s of
model-call time**, 40 additional final forecasts = 18.32 s; **peak CUDA memory 1.353 GB**; dev output
80,160 rows, final 95,040 rows. [X] i.e. the entire foundation-model component costs about **one minute
of GPU time**, plus checkpoint download and hourly aggregation. It is cheap.

**What the gain was — NOT DETERMINABLE.** [X] There is **no clean TimesFM on/off ablation anywhere in
the repo.** The two v1 CatBoost models `tfm_queue_d8` (365.51 raw pooled) and `no_tfm_full_d8` (375.56)
differ in TimesFM *and* weather *and* queue features *and* interactions *and* depth/rate/L2/iterations
(`docs/solution.md:54-55`, `docs/v1_ensemble.json:"model_validation"`), so the 10-second gap cannot be
attributed. The only matched TimesFM experiment is **context length**: 512 vs 168 hours with identical
CatBoost parameters, and context-168 was **rejected** because it did not improve both odd-date months
(`docs/research_timesfm_v3.json:"trials"`, `"selected": {"name": "none", "alpha": 0.0}`). [X] The best
available read is that TimesFM's marginal value was small enough that a halved-context variant made no
measurable difference — but the repo does not say this and I cannot prove it.

**License caveat they flag repeatedly.** [S] TimesFM-3 weights are under a non-commercial licence;
"Prize-competition clearance remains unresolved" (`README.md:79-80`, `docs/timesfm_notes.md:65-69`,
`docs/results.json:"timesfm_prize_license_clearance": "unconfirmed"`). [X] Relevant to us only if we
were considering the same checkpoint for a prize-eligible entry.

---

## 4. The "missing-timestamp mixture of experts" — precisely

Scope: **only rows where `AOBT_3_flt` is null** (detected as `mvt_minus_AOBT_3_flt < -100000`, the NaN
sentinel). 5,294 validation rows, 5,290 ranking rows.
Primary source: `scripts/missing_v2.py:211-229` (development), `scripts/moe_finalize.py:80-103` (final),
`scripts/moe_missing_arrival_v3.py:28-39` (V3 refresh), `docs/missing_v2.json:"mixture_recipe"`.

### 4.1 The two components

There are exactly **two** mixture components, and they are *not* two regressors — one of them is an
identity map:

1. **"schedule-identity" expert** = the raw quantity `mvt_minus_SCHED_TIME_UTC_mvt`
   (= `MVT_TIME_UTC_mvt − SCHED_TIME_UTC_mvt`). No model at all. [S]
2. **"normal" expert** = a CatBoost regressor predicting `TAXITIME_SEC_mvt` **directly** (not a residual)
   — 600 trees, depth 6, lr 0.06, L2 30, seed 20260909, `one_hot_max_size=20`, `max_ctr_complexity=1`. [S]
   `scripts/missing_v2.py:20-22`, `216-218`.

**Why an identity component exists.** [S] At Rome/Fiumicino (LIRF), a large sub-population of
missing-AOBT records has `TAXITIME_SEC_mvt` *exactly equal* to `MVT_TIME_UTC_mvt − SCHED_TIME_UTC_mvt`
— i.e. the airport filled the block time with the scheduled time. The repo defines the label
`identity = rome & (|y − schedule| ≤ 6)` (`scripts/missing_v2.py:212`) and reports the training
identity fraction at LIRF as **18.08 %** (`docs/lirf_identity_experiment.json:"training_identity_fraction"`).

### 4.2 Routing — a **soft** (probabilistic) gate inside a **hard** (rule) gate

```python
# scripts/missing_v2.py:220-228  (development; identical shape in moe_finalize.py:90-100)
cls = CatBoostClassifier(iterations=400, depth=5, learning_rate=.06, l2_leaf_reg=30, ...)
cls_weight = weight * np.clip(1 + (np.maximum(schedule, 0)/3600)**2, 1, 25)
cls.fit(x.loc[train & rome], identity[train & rome].astype(int), sample_weight=cls_weight[...])
probability = cls.predict_proba(x.loc[validation])[:, 1]
probability[~rome[validation] | (schedule[validation] <= 0)] = 0      # HARD gate
raw = (probability*schedule[validation] + (1-probability)*raw)        # SOFT mix
```

- **Hard outer gate:** `p` is forced to **exactly 0** unless the row is LIRF **and**
  `MVT − SCHED > 0`. So at the other nine airports the mixture degenerates to the normal expert alone.
- **Soft inner mix:** at LIRF the combination is the standard convex `p·schedule + (1−p)·normal`,
  with `p` the classifier's probability of the schedule-identity class. **Soft, not hard.**
- **Classifier training set:** LIRF rows only, all months except {7,11,12}
  (`scripts/moe_finalize.py:91`: `cls_train = training & rome & ~tail`). 121,394 final training rows.
- **Classifier sample weight:** `w · clip(1 + (max(schedule,0)/3600)², 1, 25)` — quadratic in the
  schedule delta in hours, capped at 25×. This deliberately makes the classifier care most about the
  rows where guessing wrong is worst. [S] `scripts/missing_v2.py:221`.
- **Normal-expert training set:** `training & ~tail & ~identity & (y < 12000)` — identity rows are
  *removed* so the regressor learns only the ordinary population, and the target is capped at 12,000 s
  (V3; it was 6,000 s in v1/v2 — the cap-12000 change was itself a tested improvement,
  `docs/research_moe.json:"selected": "normal_cap12000"`). `scripts/moe_missing_arrival_v3.py:29`.
- **Sample weighting for the regressor:** known-AOBT rows are sub-sampled (120,000 LIRF + 80,000
  non-LIRF, seed 20260909) and mixed in with *all* missing-AOBT rows, which get **sample weight 8**
  vs 1 (`scripts/missing_v2.py:79-83`, `:197`). 200,285 final normal-training rows.
- **Hard carve-out (the "protected tail"):** LIRF rows with `MVT − SCHED ≥ 24000 s` are removed from
  the mixture entirely and keep the frozen v1 prediction byte-for-byte. 18 validation rows, **25
  ranking rows** (`scripts/missing_v2.py:194`, `docs/v2_ensemble.json:"protected_rome_tail"`,
  `docs/research_moe_final.json:"ranking_tail_fallback_rows": 25`).

### 4.3 How the mixture is combined with the rest

Not by replacement. The MoE output is a *raw expert* that is blended into the already-complete
pipeline with a **region-specific closed-form coefficient**:

```python
# scripts/moe_finalize.py:113-115
apply = ~(rank_rome & (rank_schedule >= 24000))
alpha = np.where(rank_rome, 0.4224120030856563, 1.0)
complete[pos[apply]] += alpha[apply]*(raw[apply] - complete[pos[apply]])
```

**α(other airports) = 1.0** (full replacement), **α(LIRF) = 0.4224120030856563** (41 % shrink toward
the expert). Fitted by `blend_alpha` = `clip(δ·(y−base)/(δ·δ), 0, 1)` on the pooled Jul+Nov missing
rows (`scripts/ensemble.py:133-136`, `scripts/moe_calibrate.py:11-13`).

[S] Transfer of that α was explicitly stress-tested in `scripts/moe_calibrate.py` /
`docs/research_moe.json`: fitting on July gives α_Rome = 0.377, on November 0.939, on even dates 0.659,
on odd dates 0.296 — a **3× spread**. They shipped the pooled 0.422 anyway. **[X] That instability is a
warning: this coefficient is fitted on a few thousand rows and is not stable across periods.**

### 4.4 Measured gain

[S] Selected-mixture routed into complete v1 moves **266.052 → 263.222** on Jul+Nov
(`docs/research_program.md:64-66`, `docs/research_moe.json:"pooled"→"combined": 263.2224`), and the
later cap-12000 variant reaches 262.959 (`docs/research_moe.json` `normal_cap12000` block).
**[X] ≈ 2.8–3.1 s of local RMSE from ~5,300 rows.**

---

## 5. The "winter-only Rome flight-identity correction" — the exact rule

Source: `scripts/autoresearch_identity.py`, `scripts/autoresearch_identity_winter_check.py`,
`scripts/assemble_autoresearch_attempt.py:36-42`, `docs/autoresearch_identity*.json`.

### 5.1 What it is

It is a **second, better identity classifier** for the same LIRF missing-AOBT mixture of §4, differing
from the frozen V3 classifier in exactly two added categorical features
(`scripts/autoresearch_identity.py:45-46`):

```python
x['flight_identity']    = x.flight_prefix + '_' + x.flight_number.astype(str)   # e.g. "AZ_1234"
x['flight_destination'] = x.flight_identity + '_' + x.ADES_mvt                  # e.g. "AZ_1234_LHR"
```

[S] The rationale comment at `:44`: *"Recurring flight/route identifiers are available in ranking;
targets never enter these keys."* Classifier: CatBoost, **600 trees, depth 6**, lr 0.06, L2 30, same
quadratic schedule weight, trained on `training & rome & (schedule < 24000)`
(`scripts/autoresearch_identity.py:47-51`). The **normal regressor is unchanged** — it reloads the
frozen V3 `moe_missing_arrival_v3_normal_cap12000.cbm` (`:38-43`), so *only the routing probability
changes*.

### 5.2 The exact ranking rule

```python
# scripts/assemble_autoresearch_attempt.py:36-42  (V6 only)
winter_gate = (aobt < -100000) & (ADEP_mvt == 'LIRF') & (MVT_TIME_UTC_mvt.month == 1) & (schedule < 24000)
result[winter_gate] += 0.8788581367641393 * (identity_raw − base)
```

Four conditions, all AND: **AOBT missing**, **airport == LIRF**, **calendar month == January**,
**`MVT − SCHED` < 24000 s**. Coefficient **α = 0.8788581367641393**. In local validation the
"winter" selector is `month == 11` (November as the January proxy) —
`scripts/autoresearch_identity_winter_check.py:17`.

### 5.3 Row count

| Set | Rows in gate | Source |
|---|---:|---|
| **Ranking (the actual submission)** | **97** | `docs/autoresearch_v6_review.json:"ranking_winter_gate_rows": 97` |
| November chronological forward check | 48 (of 162,332) | `docs/autoresearch_identity_winter_forward_november.json:"gate_rows"` |

**97 rows out of 344,841 = 0.028 % of the scored set.** [X]

### 5.4 Attributed gain

| Measurement | Baseline | With correction | Δ | Source |
|---|---:|---:|---:|---|
| Local Jul+Nov pooled | 259.9769 | 259.5880 | **−0.389 s** | `docs/autoresearch_identity_winter_check.json:"local"` |
| Late-November (day ≥ 15) | 209.2171 | 208.3454 | −0.872 s | ibid. |
| Complete Nov forward check | 213.5903 | 212.6384 | **−0.952 s** | `docs/autoresearch_identity_winter_forward_november.json` |
| …on the 48 gate rows alone | **2022.44** | **1648.69** | −373.7 s | ibid. `"gate_rmse"` |
| Day-bootstrap on Nov forward SSE | −65.86 M, CI95 [−113.8 M, −25.0 M], P(improve)=1.00 | | | ibid. |
| V5 → V6 official | 269.0799 | **268.5583** | **−0.5216 s** | `docs/autoresearch_attempts.md:113-114` |

**[X] The V5→V6 official delta of 0.5216 s is the cleanest attribution available: it is exactly the
regional-vs-global LOBT coefficient change *plus* this 97-row correction, with nothing else different.
A 0.52 s official gain from ≤97+0 changed rows.** The repo separates the two locally
(`docs/autoresearch_attempts.md:76-78`: regional A2 alone 212.2474 → +winter 211.2895 in November,
i.e. ~0.96 s of the November improvement is the winter correction) but cannot separate them officially.

### 5.5 Why "winter only"

[S] The all-season version was **run first and failed**. `docs/autoresearch_identity.json`: late-July
baseline 288.0980 → candidate 288.5973 (worse), late-November 209.2171 → 208.3454 (better), so
`"eligible": false`. They then restricted the same frozen α and the same raw predictions to November /
January, which is a disjoint group from the LOBT gate — verified non-overlapping at
`scripts/assemble_autoresearch_attempt.py:43`. [X] This is post-hoc subgroup selection on repeatedly-used
tuning months; the repo says as much (`docs/autoresearch_recipe.md:41-44`). It happened to hold up
officially, once.

---

## 6. Complete feature list, CatBoost setup, CV design, final blend

### 6.1 Base feature block — `scripts/train.py:22-100`, `features()`

Built from departures only (`PHASE_mvt == 'DEP'`). Categorical columns are every `object` dtype column
except `PHASE_mvt`, NA-filled with `'__NA__'`.

**Raw categoricals (pass-through):** `FLIGHT_RULE_mvt`, `ADEP_mvt`, `ADES_mvt`, `AIRCRAFT_TYPE_mvt`,
`RUNWAY_mvt`, `STAND_mvt`, `FLIGHT_mvt`, `CALLSIGN_flt`, `ADEP_flt`, `ADES_flt`, `ADES_FILED_flt`,
`MARKET_SEGMENT_flt`, `FLIGHT_RULE_flt`, `FLIGHT_TYPE_flt`, `AIRCRAFT_TYPE_flt`, `WK_TBL_CAT_flt`,
`AIRCRAFT_OPERATOR_flt`.

**Derived categoricals/numerics (`:30-36`):** `airport_runway` = ADEP+"_"+RUNWAY; `airport_stand` =
ADEP+"_"+STAND; `stand_prefix` = leading letters of STAND; `stand_number` = digits of STAND;
`flight_prefix` = leading letters of FLIGHT; `flight_number` = digits of FLIGHT;
`callsign_prefix` = first 3 chars of CALLSIGN.

**Per-timestamp decompositions (`:37-44`)** for every tz-aware datetime column **except**
`BLOCK_TIME_UTC_mvt` (excluded by name — the target's other half):
`{col}_hour` (= hour + minute/60), `{col}_minute`, `{col}_second`, and `mvt_minus_{col}` =
`(MVT_TIME_UTC_mvt − col).total_seconds()` clipped to ±172800. That yields
`mvt_minus_AOBT_3_flt`, `mvt_minus_LOBT_flt`, `mvt_minus_IOBT_flt`, `mvt_minus_EOBT_1_flt`,
`mvt_minus_SCHED_TIME_UTC_mvt`, `mvt_minus_ARVT_1_flt`, `mvt_minus_ARVT_3_flt`.

**Pairwise timestamp differences (`:45-48`):** `LOBT−AOBT_3`, `IOBT−AOBT_3`, `EOBT_1−AOBT_3`,
`SCHED−AOBT_3`, `ARVT_3−ARVT_1`, `LOBT−IOBT`, all clipped ±172800.

**Calendar (`:49-53`):** `month`, `dow`, `day`, `dayofyear`, `week`.

**Record-quality flags (`:54-58`):** `nm_missing` (FLIGHT_ID_mvt is null), `nm_adep_match`,
`nm_ades_match`, `nm_aircraft_match` (mvt vs flt agreement), `diverted` (ADES_flt ≠ ADES_FILED_flt).

**Traffic windows (`:64-75`):** for each phase ∈ {DEP, ARR}, each width ∈ {900, 1800, 3600, 7200} s,
each direction ∈ {−1 (preceding), +1 (following)}: count of same-airport movements in that window →
`{phase}_count_{dir}_{width}`, **16 features**. Implemented with `searchsorted` on sorted second
arrays; the half-open interval excludes the row itself.

**Arrival taxi means (`:76-86`):** `arr_taxi_mean_{1800,7200,21600}` = mean *arrival*
`TAXITIME_SEC_mvt` at the same airport in ±width seconds. **Uses arrival labels — which are supplied
in ranking** (`docs/data_dictionary.md:22`: taxi-in for arrivals; only ranking *departures* are blank).

**Runway spacing (`:87-95`):** within (airport, runway): `runway_prev_gap`, `runway_next_gap`
(seconds to previous/next departure on that runway), `runway_count_30min` (count within ±900 s).

All numerics cast float32, NaN → **−999999** sentinel (`:96-98`). Asserted:
`BLOCK_TIME_UTC_mvt`, `TAXITIME_SEC_mvt` and `MVT_ID_mvt` are not in the feature frame (`:99`).

### 6.2 External / joined feature blocks

**Weather — `scripts/weather_features.py`, 22 columns.** IEM ASOS/METAR (public domain), 10 airports,
`report_type=[3,4]`, periods 2024-12-31→2026-02-01 and 2026-06-30→2026-08-01. Joined with
`merge_asof(direction='backward', tolerance=3h)` onto each airport-hour and asserted never to come from
the future (`:78-81`). Columns: `wx_temp_c`, `wx_dewpoint_c`, `wx_wind_knots`, `wx_gust_knots`,
`wx_wind_deg`, `wx_visibility_km`, `wx_pressure_hpa`, `wx_reported_{snow,freezing,rain,fog,thunder,hail}`,
`wx_dewpoint_spread_c`, `wx_freezing_temp`, `wx_cold_humid` (= `max(0,3−T)·max(0,3−spread)`, an explicit
de-icing proxy), `wx_wind_north_knots`, `wx_wind_east_knots`, `wx_age_minutes`.

**Queue — `scripts/queue_features.py`, 12 columns + timing bounds.** For `prefix ∈ {airport, runway}`
(grouped by ADEP, and by (ADEP,RUNWAY)): `{prefix}_queue_at_aobt` and `{prefix}_queue_at_takeoff` =
number of *other* flights whose [AOBT, takeoff] interval is open at that instant (an actual
concurrent-taxiing count); `{prefix}_proxy_taxi_{mean,count}_{1800,7200}` = mean/count of neighbouring
flights' observed AOBT→takeoff durations in ±width, **excluding the row itself** (`:28-38`). Valid
source intervals require `0 < duration ≤ 21600`. Also emits `timing_bounds.parquet` =
`(MVT − LOBT) ± 3606` (`:72-73`).
Note: the MoE/missing-row models **drop the `*_at_aobt` columns** (`scripts/missing_v2.py:102`,
`scripts/moe_finalize.py:47`) — correctly, since those rows have no AOBT.

**Arrival context — `scripts/arrival_context.py`, 9 columns (V3+).** Per (airport, calendar-month):
`arr_stand_{mean,median,count}` and `arr_runway_{mean,median,count}` of arrival taxi-in clipped 0–7200,
joined by the *departure's* stand/runway; plus previous-arrival-at-this-stand features
`arr_stand_previous_{age, taxi, aircraft_match}` (max age 86400 s, "not assumed to be the same
aircraft", `:36-56`). Has a mutation self-check (`--check`, `:66-76`) that corrupts departure targets
and block times and asserts the output is bit-identical.

**TimesFM — 12 columns.** `tfm_q1..q9`, `tfm_median`, `tfm_decile_mean`, `tfm_width` (§3).

**Interaction features (`--interactions`, `scripts/train.py:178-187`):** `stand_runway`,
`airport_operator`, `runway_aircraft`, plus `headwind` and `crosswind` computed by projecting the
METAR wind vector onto the runway heading parsed out of `RUNWAY_mvt` digits × π/18.

**Tail-copy features (`--tail-copies`, `:188-193`):** `aobt_delta_below_300` = `clip(aobt, upper=300)`,
`aobt_delta_above_2200` = `clip(aobt, lower=2200)`, `eobt_delta_above_3600` = `clip(eobt, lower=3600)`.
[S] Motivation at `docs/research_program.md:69-72`: CatBoost's 128 borders were landing at 304.5 /
2218.5 / 3662.5 and therefore could not resolve the extreme-delay tails; these three clipped copies
re-expose the tail at full resolution.

**LOBT-expert extras (`scripts/autoresearch_known_lobt.py:46-52`, V5/V6):** `lobt_residual_floor` =
`mvt_minus_LOBT − 3606`, `lobt_residual_ceiling` = `+3606`, `lobt_aobt_abs` = `|LOBT_delta − AOBT_delta|`,
plus the three tail-copies.

**The MoE's exact 90-feature list is published verbatim** at `docs/missing_v2.json:"feature_columns"`
— note it strips every `_flt` column and everything starting with `nm_`, plus `FLIGHT_mvt`,
`callsign_prefix`, `diverted` (`scripts/missing_v2.py:86-87`), and adds `airport_flight_prefix` and
`flight_prefix_length`.

### 6.3 CatBoost setup and hyperparameters

Common to all `train.py` fits (`scripts/train.py:214-219`): `loss_function='RMSE'`, `task_type='GPU'`,
`random_seed=42`, `border_count=128`, **`one_hot_max_size=20`, `max_ctr_complexity=1`**,
`gpu_ram_part=0.55`. [X] `max_ctr_complexity=1` is notable — it forbids multi-feature target-statistic
combinations, which for a dataset this categorical-heavy is a deliberate variance/memory trade.

| Model | Trees | Depth | LR | L2 | Features | Role / weight |
|---|---:|---:|---:|---:|---|---|
| `tfm_queue_d8` | 6000 (5995 final) | 8 | 0.0750579 | 4.090916 | base + TFM + wx + queue + interactions | v1 blend w=0.79856 |
| `no_tfm_full_d8` | 5000 | 8 | 0.06 | 10 | base only | v1 blend w=0.17672 |
| `lgbm_tfm` (LightGBM) | 188 (1000 cap, ES 100) | 63 leaves | 0.05 | λ_L2 20 | base + TFM, 500k sample | v1 blend w=0.02472 |
| `tfm_queue_d10_v2` | 4500 | 10 | 0.05 | 30 | as tfm_queue_d8 | v2 4th model |
| `catboost_tail_d8_full_v3` | 6000 (5995) | 8 | 0.07505790 | 4.09092 | + `--tail-copies` | V3 tail delta, α=0.5441252 |
| `tfm_arrival_d8_v3` | 6000 (5995) | 8 | 0.0750579 | 4.090916 | + arrival context | V3 arrival delta, α=1.0 |
| `known_lobt_d10_a2` | 3500 | 10 | 0.05 | 20 | + LOBT extras | V5/V6 LOBT expert, α global 0.63080 |
| MoE normal | 600 | 6 | 0.06 | 30 | 90-col, CPU, 3 threads | missing-row regressor |
| MoE identity classifier | 400 | 5 | 0.06 | 30 | Logloss | routing probability |
| V6 winter classifier | 600 | 6 | 0.06 | 30 | + flight_identity/destination | winter routing probability |
| v1 missing specialist | 406 | 5 | 0.04 | 25 | seed 2026 | `scripts/missing_specialist.py:20-21` |
| v1 LIRF normal / classifier | 500 / 400 | 6 / 5 | 0.08 | 30 | — | `scripts/lirf_identity_experiment.py:86-96` |

[S] Tree hyperparameters were explored with "seeded Optuna TPE"; the search itself is not in the repo
(`docs/solution.md:61-63`). [X] The odd precision of 0.0750579 / 4.090916 is the Optuna residue.

### 6.4 CV design

[S] **Not k-fold. A fixed month split, plus within-month date splits, plus a chronological backtest.**

1. **Primary split (`scripts/train.py:145-148`):** train on 2025 months ∉ {7, 11, 12}; validate on
   {7, 11}. December is *reserved* and scored exactly once, after v1 was frozen (217.5040,
   `docs/results.json:"december_audit"`); every later version declares `"december_scored": false`.
   The repo is repeatedly explicit that July/November are *reused tuning months, not a test set*
   (`docs/solution.md:12-13`, `docs/autoresearch_attempts.md:24-25`).
2. **Early stopping:** 200 rounds against the Jul+Nov eval set for development fits; **disabled** for
   `--final` and rolling fits (`scripts/train.py:221-222`).
3. **Blend-coefficient calibration — two schemes, both within-month:**
   - v2/v3: fit on **even** UTC day-of-month, check on **odd** (`docs/v3_ensemble.json:"calibration"`).
   - v5/v6: fit on **days 1–14**, check on **days 15–end** (`docs/autoresearch_known.json:"protocol"`).
   Selection rule: *retain only if BOTH July and November check RMSE improve*, then minimise their sum
   (`scripts/autoresearch_known_lobt.py:134-135`).
4. **Cross-month transfer checks:** fit on July → check November and vice versa, reported for the
   simplex weights (`docs/v2_ensemble.json:"cross_month_transfer"`) and the MoE α
   (`docs/research_moe.json:"held_checks"`).
5. **Chronological forward refits** (`scripts/forward_v3.py`, `--rolling-month`): rebuild the *entire*
   pipeline training only on strictly earlier labels, for 2025-07, 2025-10, 2025-11, with fixed tree
   counts so validation labels cannot pick the stopping point (`docs/rolling_validation.md:12-23`).
6. **Ranking-mix reweighting** (`scripts/validation_shift.py`): reweight local squared errors by the
   ranking set's observed (season × airport × AOBT-missingness) proportions, mapping ranking-January to
   local-November and ranking-July to local-July. V3 259.98 → 264.43; V2 262.53 → 266.91
   (`docs/validation_shift.json`). [X] This estimator still undershot the true official 271.04 by ~6.6 s.
7. **Day-block bootstrap** (10,000 resamples of whole UTC days) for every accepted change
   (`scripts/autoresearch_identity_winter_check.py:20-21`).

### 6.5 Final blend

[S] **Seasonal simplex weights over 4 tree models**, fitted by SLSQP on the sum-to-one simplex with
analytic gradients from a residual-covariance Gram matrix (`scripts/ensemble.py:62-78`), and — when a
specialist is present — *jointly* with the bounded specialist α by an exact routed-MSE objective with
closed-form gradient and 3 restarts at α ∈ {0, 0.5, 1} (`scripts/ensemble.py:89-130`).

Model order: `[tfm_queue, no_tfm, lgbm_tfm, tfm_queue_d10_v2]`. Pooled weights were shrunk 50 % toward
month-specific weights (`docs/v2_ensemble.json:"frozen_blend"`):

| | tfm_queue | no_tfm | lgbm | tfm_queue_d10 |
|---|---:|---:|---:|---:|
| **July weights** | 0.518131 | 0.240966 | 0.000000 | 0.240903 |
| **November/"winter" weights** | 0.607333 | 0.060370 | ~0 | 0.332297 |

[S] Ranking rows in **July 2026 use the July weights; January 2026 uses the November weights**
(`scripts/catboost_apply_v3.py:39-44`, `docs/v2_ensemble.json:"january_policy"`). LightGBM's weight
collapsed to zero once the 4th CatBoost joined — it survives in the file list but contributes nothing.

Final refits use **all 2025 labels including December**, with iteration counts frozen from the
development fits (`scripts/train.py:157-158`, `docs/solution_v3.md:54-63`).

V6's LOBT coefficients, airport-specific, ridge-shrunk toward the global 0.630798 with the equivalent
of **1,000 pseudo-rows** (`scripts/autoresearch_lobt_regional.py:49-53`, selected trial 0):

| EDDF | EDDM | EGLL | EHAM | LEBL | LEMD | LFPG | LIRF | LSZH | LTFM |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.3528 | 0.2828 | **0.8563** | 0.3846 | 0.4830 | 0.4047 | 0.6693 | **0.8354** | **0.2325** | 0.5110 |

[X] The 3.7× spread (LSZH 0.23 vs EGLL 0.86) says the LOBT timestamp is far more informative at
Heathrow and Fiumicino than at Zurich/Munich — a genuinely reusable structural finding.

---

## 7. Reconstructed experiment log, in order, with scores

Dates from `git log`; all ten commits fall in 2026-09-08 … 2026-09-15.

### Phase 1 — v1 (2026-09-08, commits `2ab5238`, `14d54e6`)

| # | Trial | Result |
|---|---|---|
| 1 | Signal audit: confirm `TAXITIME = MVT − BLOCK` exactly; profile each timestamp as a proxy | `scripts/audit_signals.py:35`; output `docs/signal_audit.json` is **gitignored / not published** |
| 2 | Proxy baseline: AOBT proxy + small CatBoost on missing-AOBT rows only | `scripts/proxy_baseline.py`; `docs/proxy_baseline.json` **not published** |
| 3 | `residual_d8` (2200 trees, d8) — needed by the Rome experiment, **absent from the final blend** | `docs/solution.md:113-121` |
| 4 | Missing-AOBT specialist (406 trees, d5) + Rome schedule-tail closed-form shrink | `scripts/missing_specialist.py` |
| 5 | LIRF identity expert: unweighted vs schedule-weighted classifier | **schedule-weighted selected**, LIRF pooled 642.61 → 614.48 (`docs/lirf_identity_experiment.json`) |
| 6 | 3-model simplex + joint α + nonnegative clip + timing bounds | 362.72 → **266.0522**; official **285.0706**, rank 14/81 |
| 7 | December frozen audit, scored once | **217.5040** on 165,677 rows |

### Phase 2 — v2 (2026-09-09, commits `284ae17`, `bc99852`, `a35d782`, `d3dff4d`)

| # | Trial | Result |
|---|---|---|
| 8 | `missing_v2` candidate **"direct"** (one regressor, no mixture) | rejected — `docs/missing_v2.json:"selected": "mixture"` |
| 9 | `missing_v2` candidate **"mixture"** (identity classifier × schedule + normal) | **accepted**; complete 266.05 → 263.22 |
| 10 | MoE classifier variants `d4_l50_i600`, `d6_l60_i400` | both **rejected**; incumbent d5/L30/i400 kept (`docs/research_moe_classifier_trials.json`) |
| 11 | MoE normal-expert target cap 6000 → **12000** | **accepted**, 263.22 → 262.96; noted "evidence is mixed by subslice" (`docs/research_timesfm.json:"missing_v2"→"evidence_limit"`) |
| 12 | Two GPU tail fits aborted on "sustained GPU throughput failure" | recorded as failures (`docs/research_catboost.json:"trials"[0..1]`) |
| 13 | Tail delta d6 vs d8, gate = AOBT present & (AOBT_delta<300 or >2200 or EOBT_delta>3600), 9,866 rows | **d6 selected**, α=0.29699 |
| 14 | 4th base model `tfm_queue_d10_v2`; ridge penalties 0 / 0.01 / 0.05 on the simplex | **unregularized** selected |
| 15 | Seasonal weight shrink 0.25 / 0.5 / 0.75 | **0.5** kept (rule: strongest tested shrink ≤ 0.5 with both odd-date checks beating pooled) |
| 16 | v2 total | **262.5264** local; official **274.3377**, rank **6/83** |

### Phase 3 — v3 (2026-09-09, commit `a35d782`)

| # | Trial | Result |
|---|---|---|
| 17 | **TimesFM context 168 vs 512**, matched CatBoost, gates `known_all` and `known_nonrome` | **REJECTED** — `docs/research_timesfm_v3.json:"selected": {"name":"none","alpha":0.0}` |
| 18 | Known-Rome expert: `gap_binary_v3b` | rejected, α≈0.033, no real gain (`docs/research_moe_v3.json`) |
| 19 | Known-Rome expert: `multisource_v3b` (4 source classes normal/schedule/AOBT/LOBT) | rejected, **α fitted to exactly 0.0** |
| 20 | Known-Rome residual model (target `y − AOBT_delta`) | rejected — Jul→Nov transfer check worsened (`docs/research_moe_known_residual_v3.json`) |
| 21 | Known-Rome + arrival context | rejected — `fit_july_check_november` worsened 290.95 → 291.62 |
| 22 | **Arrival context added to the missing MoE** ("missing-arrival normal expert") | **accepted** |
| 23 | **Full-development CatBoost tail model** (6000 trees, no 600k subsample) | **accepted**, α = 0.5441252012777676 |
| 24 | **Arrival-context model on known non-Rome rows** | **accepted**, α = 1.0, gate 321,614 rows |
| 25 | Joint 2-variable L-BFGS-B re-fit of the two α's | **REJECTED** — November odd dates regressed (`docs/research_joint_v3.json:"promoted": false`) |
| 26 | v3 total | **259.9769** local; official **271.0449** |

### Phase 4 — autoresearch, 3-attempt budget (2026-09-15, commits `5f7f293`, `ef6730b`, `a8c29c8`, `845d0d1`)

| # | Trial | Result |
|---|---|---|
| 27 | **V4**: three predeclared LIRF missing-tail shrink policies, spec (54000, 86400, shrink 0/1/3) | local 258.0149 (−1.96), **6 validation rows changed, 3 ranking rows changed**, driven by 2 late-July days (`docs/autoresearch_v4_review.json`, `docs/autoresearch_ensemble_lirf_tail_policy.json`) |
| 28 | **V4 OFFICIAL → 271.5069, WORSE than V3's 271.0449. REJECTED and excluded from V5/V6.** | `docs/autoresearch_attempts.md:44-47` |
| 29 | LOBT-residual expert `known_lobt_d8_a1` (3500/d8/.06/L12) | α=0.53613, pooled 258.1789 |
| 30 | LOBT-residual expert **`known_lobt_d10_a2`** (3500/d10/.05/L20) | **SELECTED**, α=0.630798, pooled **257.7679** |
| 31 | LOBT-residual expert `known_lobt_d10_a3` (5000/d10/.04/L30) | 257.8050, rejected as "longer 5,000-tree alternative worse overall" |
| 32 | An early LOBT trial whose eval_set leaked later check dates | discarded, **recorded** as `"contaminated_exploratory_interrupted"` |
| 33 | 3-column blend v3 + a2 + a3 | 257.6752 — better than a2 alone but **not selected** (`docs/autoresearch_lobt_blend.json`) |
| 34 | Gate choice for LOBT: `faulty_or_disagree` / `disagree` / **`all_lobt`** | all-rows gate selected |
| 35 | Airport-specific LOBT coefficients, ridge 1000 vs 5000 pseudo-rows | **1000 selected**, 257.5250; "regional gain over global is uncertain under a day-block bootstrap (95 % SSE interval −47.63 M to +7.16 M)" |
| 36 | Missing-row residual experts `missing_only_d6`, `missing_weighted_d6`, `missing_only_d4` | **all three rejected**, `"eligible": false` (`docs/autoresearch_missing.json`) |
| 37 | All-season flight-identity Rome correction | **REJECTED** (July worsened) |
| 38 | **Winter-only** flight-identity Rome correction | **accepted**, 97 ranking rows |
| 39 | **V5 OFFICIAL → 269.0799** (global LOBT α), rank 15/147 | −1.9650 vs V3 |
| 40 | **V6 OFFICIAL → 268.5583** (regional LOBT α + winter correction), rank **14/147** | −0.5216 vs V5 |
| 41 | Final artifact review caught 1 negative prediction in V5, 2 in V6 → restored the nonnegative projection | `docs/autoresearch_attempts.md:88-92` |
| 42 | Chronological forward backtest (Jul/Oct/Nov) + ranking-mix reweighting | diagnostic blend 263.9724 vs baseline 269.5916; explicitly **not** claimed as a V3 replacement |

**[X] Success rate: of roughly 26 distinct hypotheses I can count, 9 were retained. The repo logs every
failure, including an aborted GPU run and a self-detected leak.**

---

## 8. Dependence on data not available at prediction time

**Determinable, and the answer is: nothing is unavailable — but the task is reconstruction, not
forecasting.** [S] with [X] framing.

| Field used | Available in `ranking.parquet`? | Evidence |
|---|---|---|
| `BLOCK_TIME_UTC_mvt` (the target's other half) | **No** — and it is excluded by name in every feature builder | `scripts/train.py:37, 99`; `scripts/lightgbm_model.py:91`; `scripts/lirf_identity_experiment.py:48` |
| `TAXITIME_SEC_mvt` for ranking **departures** | **No**, asserted null | `scripts/timesfm_features.py:30` |
| **`AOBT_3_flt`** (actual off-block time) | **Yes** — and it is the residual base | `docs/data_dictionary.md:42, 64`; `scripts/train.py:200` |
| `MVT_TIME_UTC_mvt` (actual takeoff time) | **Yes** | `docs/data_dictionary.md:16` |
| `LOBT_flt`, `IOBT_flt`, `EOBT_1_flt`, `SCHED_TIME_UTC_mvt`, `ARVT_1/3_flt` | **Yes** | `docs/data_dictionary.md:28-43` |
| `TAXITIME_SEC_mvt` for ranking **arrivals** | **Yes** — used for `arr_taxi_mean_*` and all `arr_stand_/arr_runway_` features | `docs/data_dictionary.md:22`; `scripts/train.py:76-86`; `scripts/arrival_context.py:26` |
| Same-month/centred traffic counts (i.e. using *future* flights within the month) | **Yes** — symmetric ±window counts include later movements | `scripts/train.py:69-75`; `docs/queue_features.json` note "Symmetric proxy windows use provided post-operations times" |
| Weather at/before the hour | **Yes**, and asserted never from the future | `scripts/weather_features.py:78-81` |
| TimesFM histories | Dev: labels from {7,11,12} globally excluded, mutation-checked. Final: all 2025 labels permitted after freezing | `scripts/timesfm_features.py:88-104`, `:121` |

**[X] Bottom line: there is no rule violation and no label leakage I could find — the mutation checks
are real and they run. But this solution is only possible because the organizers ship `AOBT_3_flt` in
the ranking set. A model that ignores AOBT is solving a fundamentally harder problem and cannot reach
268 s. If our 284.36 does not already use `MVT_TIME_UTC_mvt − AOBT_3_flt` as its base, that is the
single highest-value change available to us.**

Two further points worth flagging:
- [S] The `--final` fits train on **all** 2025 labels including December and July/November. That is
  legitimate (labels are given) but it means the local 257.13 is measured with development models while
  the submitted 268.56 came from differently-fitted final models. The repo notes GPU refits can differ
  even at fixed seed (`docs/solution_v3.md:66-67`).
- [X] `arr_stand_*` features are computed per **calendar month** over the whole month including
  arrivals that land *after* the departure in question. Fine for this post-operations task; would be
  leakage in an online setting.

---

## 9. What I could **not** determine from this repo

Being explicit, since this is the repo we knew least about:

1. **The marginal value of TimesFM.** No on/off ablation exists (§3). The only matched comparison is
   512-hour vs 168-hour context, and it found no improvement. I cannot say whether dropping TimesFM
   entirely would cost 0.1 s or 5 s.
2. **Feature importances.** CatBoost writes them to `artifacts/NAME.json` (`scripts/train.py:236`) but
   `artifacts/` is gitignored. Nothing published.
3. **`docs/signal_audit.json` and `docs/proxy_baseline.json` are not in the repo** — so the actual
   AOBT-proxy residual distribution, the per-airport AOBT missingness rates, and the raw proxy RMSE are
   *not* published. I inferred the two-population structure from `docs/missing_v2.json` and
   `docs/v1_ensemble.json` instead.
4. **`results.json:"baseline_validation_rmse": 284.9253`** — no script in the repo writes or explains
   this key. Its definition is undetermined.
5. **The Optuna search** that produced `rate=0.0750579, l2=4.090916` is not in the repo
   (`docs/solution.md:61-63` says it is "unnecessary to reproduce frozen settings").
6. **Exact validation row count for the winter gate.** Only the *forward* November gate (48 rows) and
   the *ranking* gate (97 rows) are stated; the in-sample validation gate count is not.
7. **Per-airport RMSE.** `train.py:240` computes `airport_rmse` into the gitignored artifact JSON; no
   published per-airport breakdown for the full pipeline exists (only the LOBT coefficients, which are
   an indirect signal).
8. **Whether V6's official 268.5583 would replicate.** V5→V6 differ in two things at once; and V4
   demonstrated their local metric can move −1.96 s while the official moves **+0.46 s**.

---

## 10. Applicable to `resilient-hedgehog` (currently 284.3584)

Ranked by attributed gain, most reimplementable first. Gains are **their** measured numbers; our
transfer is [X] unless noted.

### Tier 1 — structural; potentially worth tens of seconds

**1. Use `MVT_TIME_UTC_mvt − AOBT_3_flt` as the residual base, not as a feature.**
[S] `scripts/train.py:199-205`. Fit trees on `y − base` with the fallbacks: base=900 s when out of
`[0, 172800]`; for LIRF with missing AOBT substitute `MVT − SCHED`. **[X] Attributed gain: this is the
difference between "predicting taxi-out" and "correcting a supplied measurement". Their known-row RMSE
is 226.85 (`docs/research_catboost.json`) and their whole-pipeline 268.56. Our 284.36 sits between —
if we already do this, skip; if not, this is the single biggest item on the page.**

**2. Split the problem explicitly by AOBT presence, and spend your effort on the ~1.5 % missing rows.**
[S] 5,294/353,045 rows carry 28.4 % of total SSE (RMSE 1157.63 vs 226.85); the worst 100 of them carry
~21 % of *all* SSE (`docs/missing_v2.json`). [X] **Action: compute our own SSE-by-AOBT-presence table
first — it tells us immediately where our 284.36 is coming from. If our missing-row RMSE is far above
~1100, that alone can explain most of our 16-second gap.**

**3. Timing bounds from LOBT: clip every prediction to `(MVT − LOBT) ± 3606 s`.**
[S] `scripts/queue_features.py:79-91` verifies **zero violations** across all non-held-out training
months; applied at `scripts/ensemble.py:153-167`. Their own measurement: `bounds_changed_rows: 55` out
of 353,045 in v1 (`docs/v1_ensemble.json`) — tiny row count, but those are exactly the catastrophic
rows. [X] Cost: ~20 lines. **Verify the ±3606 invariant on our own data before trusting it.**
Also apply `max(prediction, 0)` — they shipped V5/V6 with 1–2 negative predictions before catching it
(`docs/autoresearch_attempts.md:88-92`).

### Tier 2 — the Rome schedule-identity family; ~3 s locally, and the V6 delta

**4. Detect the "block time == scheduled time" population and route it with a soft mixture.**
[S] `identity := |y − (MVT − SCHED)| ≤ 6`, which is **18.08 %** of LIRF training rows
(`docs/lirf_identity_experiment.json`). Recipe (`scripts/missing_v2.py:211-229`):
- Train the *normal* regressor with identity rows **removed** and target capped at 12,000 s.
- Train a CatBoost **classifier** (400 trees, d5, lr .06, L2 30) on LIRF rows only, with sample weight
  `clip(1 + (max(sched_delta,0)/3600)², 1, 25)`.
- Predict `p·sched_delta + (1−p)·normal`; force `p = 0` unless (LIRF **and** sched_delta > 0).
- Blend into the complete pipeline with region α: LIRF 0.4224, others 1.0.
- Carve out LIRF with `sched_delta ≥ 24000` and leave it untouched.
[X] **First check whether the identity population exists at all in our data, and at which airports —
they only ever found it at LIRF, but `docs/research_program.md:76` flags "LEBL schedule identity is a
possible separate expert" as an untested lead we could take.** Their measured gain: 266.05 → 263.22 local.

**5. Add `flight_identity` and `flight_destination` categoricals to that classifier, for winter only.**
[S] `flight_prefix + "_" + flight_number`, and that + `"_" + ADES_mvt`
(`scripts/autoresearch_identity.py:45-46`). All-season use **hurt July** and was rejected; restricted
to January/November it gave −0.95 s on the November forward check from **48 rows**, and V5→V6 officially
moved −0.5216 s. [X] **Caveat this honestly: it is a 97-row, post-hoc, subgroup-selected correction on
repeatedly-tuned months. Reproduce the mechanism (recurring-flight identity helps predict schedule-identity
records), not the specific winter gate.**

### Tier 3 — expert deltas on the known rows; ~2.2 s local, ~2.0 s official (V3→V5)

**6. LOBT-residual expert: train a second model on `y − (MVT − LOBT)` and blend it in.**
[S] `scripts/autoresearch_known_lobt.py:70` — target is `TAXITIME − mvt_minus_LOBT`; CatBoost 3500
trees, depth 10, lr 0.05, L2 20, seed 42, fixed iteration count (no early stopping, so validation
labels can't leak into model selection). Gate = rows with both AOBT and LOBT (339,551 of 344,841).
Blend `pred += α·(expert − base)` then clip to bounds. **Measured: local 259.98 → 257.77 (−2.21 s);
chronological forward July 297.34 → 293.89 (−3.45 s), November 213.59 → 212.36 (−1.23 s); official
271.04 → 269.08 (−1.97 s).** [X] **This is the best-evidenced single change in the whole repo — it has
a forward check AND an official confirmation. Highest expected value per unit of work for us.**

**7. Airport-specific blend coefficients, ridge-shrunk toward the global one.**
[S] `α_airport = clip((d·r + k·m̄·α_global)/(d·d + k·m̄), 0, 1)` with `k = 1000` pseudo-rows and
`m̄ = mean(d²)` over the calibration rows (`scripts/autoresearch_lobt_regional.py:49-53`). Their
coefficients span 0.23 (LSZH) to 0.86 (EGLL). Local −0.24 s; part of the V5→V6 −0.52 s official.
[X] **The shrinkage form is the reusable part — it generalises to any per-group blend coefficient we fit.**

**8. Tail-copy features to defeat CatBoost's border quantisation.**
[S] `clip(aobt_delta, upper=300)`, `clip(aobt_delta, lower=2200)`, `clip(eobt_delta, lower=3600)`
(`scripts/train.py:188-193`), motivated by the observation that 128 borders landed at 304.5/2218.5/3662.5
and could not resolve the tails (`docs/research_program.md:69-72`). Paired with a gated delta
(gate = AOBT present & (aobt<300 or aobt>2200 or eobt>3600), 9,866 validation rows), their v2 tail
delta was worth 265.81 vs 266.05 and their V3 full-development version got α=0.5441. [X] **Three lines;
try it before touching hyperparameters.**

### Tier 4 — features and machinery, individually small but cheap

**9. Arrival-side features (arrival labels ARE given in ranking).** `arr_taxi_mean_{1800,7200,21600}`
(`scripts/train.py:76-86`) and the 9 arrival-context columns — per airport-month stand/runway taxi-in
mean/median/count joined on the *departure's* stand/runway, plus previous-arrival-at-this-stand
age/taxi/aircraft-match (`scripts/arrival_context.py`). [S] Their chronological backtest attributes
5.48 s to arrival context across three forward months (269.5916 → 264.1122, `docs/rolling_validation.md:49-58`)
— [X] though that is a matched-diagnostic comparison, not a V3 ablation.

**10. Queue features with correct self-exclusion.** `{airport,runway}_queue_at_takeoff` = number of
*other* aircraft whose AOBT→takeoff interval is open at your takeoff instant; `*_proxy_taxi_mean_{1800,7200}`
= neighbours' observed taxi durations excluding your own (`scripts/queue_features.py:23-38`). [X] The
"how many aircraft are taxiing right now" count is the physically right congestion variable and is
exactly computable here because AOBT and takeoff are both given.

**11. Weather from IEM METAR, with a `cold_humid` de-icing proxy.** `max(0, 3−T)·max(0, 3−dewpoint_spread)`
and runway-projected `headwind`/`crosswind` (`scripts/weather_features.py:69`, `scripts/train.py:182-187`).
[X] Public-domain, ~2 minutes to fetch, clean provenance for prize eligibility.

**12. TimesFM.** [X] **Rank this LAST.** It costs ~1 minute of GPU (§3) so it is cheap to try, but the
repo provides **no evidence** that it helped, and its own context ablation found none. If we adopt it:
one hourly series per airport of mean departure taxi-out, 512-hour context, month-long horizon, 8
known covariates, zero-shot, and use all 9 deciles + median + decile-mean + width as *features*, never
as a base prediction. Note the non-commercial weight licence (`README.md:78-80`).

### Tier 5 — process discipline (free, and it is what kept them honest)

**13. Never replace, always shrink: `pred[gate] += α·(expert[gate] − pred[gate])`** with α from
closed-form bounded least squares `clip(δ·(y−base)/(δ·δ), 0, 1)` (`scripts/ensemble.py:133-136`), fitted
on a *calibration* date subset and validated on a held-out date subset **of the same months**, retained
only if **both** months improve. [X] Trivially reimplementable and it prevents most over-fitting.

**14. Assert that rows outside the gate are bitwise unchanged** after every correction
(`scripts/catboost_apply_v3.py:71-72`, `scripts/assemble_autoresearch_attempt.py:43-45`). [X] They caught
real bugs with this, including overlapping corrections and negative outputs.

**15. Heed the V4 lesson.** A change that moved **6 validation rows / 3 ranking rows** improved local
RMSE by 1.96 s and made the **official score 0.46 s worse** (`docs/autoresearch_v4_review.json`,
`docs/autoresearch_attempts.md:44-47`). [X] **Rule for us: require a minimum changed-row count, and
require a day-block bootstrap CI that excludes zero, before spending a submission. Their bootstrap for
V4 gave P(improve)=0.867 with a CI touching zero — the evidence was visibly weak and they submitted
anyway.**

**16. Calibrate expectations about local→official transfer.** [S] Their local-to-official gap ran
~19 s at v1, ~12 s at v2/v3, ~11.4 s at V6; their ranking-mix reweighting diagnostic
(`scripts/validation_shift.py`) closed only ~4.5 s of it. [X] **A local improvement of under ~1 s should
not be assumed to survive to the leaderboard.**
