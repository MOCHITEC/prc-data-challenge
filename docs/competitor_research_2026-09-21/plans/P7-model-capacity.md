# P7 — Model capacity on clipped residuals

Written 2026-09-21. Team `resilient-hedgehog`, public LB **284.3584**, rank 43/168
(`docs/competitor_research_2026-09-21/SUMMARY.md` §A **[S]**).

Marks: **[S]** = read from a file, with `path:line`. **[X]** = my inference or arithmetic.
No code was changed and no training run was started while writing this.

The theme is *capacity*, not *information*. It splits into two claims that are usually conflated
and must be measured separately:

* **Claim A — more capacity in the family we already use.** A much larger LightGBM on the same
  2.08 M rows and the same 89 columns.
* **Claim B — a residual boost model on clipped residuals**, a different family at high capacity,
  blended at a fixed weight, following Phoenix-Ops's v8 shape.

Claim A is the control that slot 12 says nobody ran. Claim B is the shape that produced the
largest single documented gain in any competitor log. They share a measurement apparatus and a
compute budget, so they are planned together, but they must not be shipped or judged together.

---

## 1. Our current capacity, exactly

Read from `configs/main.yaml` and `src/prc/train.py` at HEAD. All **[S]**.

### 1.1 Model A (rows with an NM off-block time; ~2.06 M of 2,085,047)

| knob | value | source |
|---|---|---|
| objective | `regression` (L2 on raw seconds) | `src/prc/train.py:58` |
| learning_rate | **0.05** | `src/prc/train.py:59` |
| num_leaves | **127** | `src/prc/train.py:60` |
| min_data_in_leaf | **50** | `src/prc/train.py:61` |
| feature_fraction | 0.8 | `src/prc/train.py:62` |
| bagging_fraction / freq | 0.8 / 1 | `src/prc/train.py:63-64` |
| lambda_l2 | 1.0 | `src/prc/train.py:65` |
| num_threads | 4 | `src/prc/train.py:67` |
| determinism | `deterministic: True`, `force_row_wise: True` | `src/prc/train.py:75-76` |
| num_rounds (cap) | **2000** | `configs/main.yaml:11`, `src/prc/train.py:79` |
| early_stopping | **100** | `configs/main.yaml:12`, `src/prc/train.py:80` |
| n_seeds | **3** | `configs/main.yaml:21` |
| features given | **89 numeric** of 99 in the frame, plus **10 categoricals** | `experiments/experiments.jsonl` (`n_features` 89, `n_frame_columns` 99); `src/prc/features.py:33-44` |
| target | `LABEL − PROXY` (unclipped), prediction `PROXY + g` | `src/prc/train.py:385`, `:396` |

`configs/main.yaml` names no `params:` block, so model A runs the dataclass defaults verbatim
(`src/prc/pipeline.py:132-133` merges a config's `params` onto the defaults; main.yaml has none).

### 1.2 How the round count is actually chosen

This is the part that matters most for a capacity change, and it is not "2000".

1. `_inner_holdout_mask` takes the **latest `inner_holdout_months` = 1 calendar month of the
   training side** as a boolean mask (`src/prc/train.py:285-299`, `configs/main.yaml:17`).
2. LightGBM trains on the other ~11 months with `early_stopping(100)` against that one month
   (`src/prc/train.py:334-342`).
3. `best_iteration` is then **scaled up** to the larger refit set —
   `rounds = min(num_rounds, max(10, round(best · train.height / (train.height − n_hold))))`
   (`src/prc/train.py:345`). With 12 training months that factor is ≈ 12/11 ≈ 1.09. **[X]**
4. The model is refitted on **all** training rows at that fixed round count
   (`src/prc/train.py:348-353`). The evaluation fold never sees early stopping.

Measured on the `forward` fold (`experiments/experiments.jsonl`) **[S]**:

| record | seeds | threads | train_rows | `rounds_matched` | `rounds_unmatched` | `train_seconds` |
|---|---|---|---|---|---|---|
| `feat_base-b9e21a829d` | 1 | 4 | 1,757,038 | 978 | — | **305.5** |
| `inv_baseline_forward-9271caca22` | 3 | 2 | 1,757,038 | 939 | 45 | 1,108.6 |
| `submit_v10_reparam-77f92c802d` | 3 | 4 | 1,757,038 | 1,055 | 64 | 686.6 |
| `main-48c19bc830` (4 folds) | 3 | 4 | up to 1,931,341 | 684–1,067 | 367–479 | 2,387.4 |

**The 2000 cap does not bind on model A.** Early stopping selects ~940–1,260 rounds at lr 0.05 /
127 leaves. So "more rounds" is not a free lever: the model already stops where the inner holdout
says to stop. The capacity levers that are *not* already self-selected are **num_leaves**,
**min_data_in_leaf** and **learning_rate** (a lower lr moves the selected round count up). **[X]**

### 1.3 How `n_seeds` composes — it is not capacity

`src/prc/train.py:348-356`: **one** `lgb.Dataset` is built, the round count from step 3 is fixed,
and the loop refits `n_seeds` times changing only `seed`, `bagging_seed` and
`feature_fraction_seed`, then averages the predictions. So `n_seeds: 3` is **prediction averaging
of three draws of the same-sized model**, not a larger hypothesis class. It removes seed luck; it
does not raise capacity. Cost is linear (`src/prc/train.py:115` comment, **[S]**). Any capacity
experiment must therefore hold `n_seeds` fixed, and the cheap setting for screening is
`n_seeds: 1`.

### 1.4 Model B (rows with no NM off-block time; ~22 k train, 5,290 ranking)

`for_unmatched()` overrides three knobs and nothing else (`src/prc/train.py:135-145`).
`configs/main.yaml:18-20` sets **127 leaves / min_data 50 / 2000 rounds**, *overriding* the
dataclass defaults of 255/20/3000 (`src/prc/train.py:95-97`). The header comment on
`configs/main.yaml:4-9` records why: CV rewarded 255/20 by 34.8 s and the leaderboard charged
13.2 s for it (v2 312.02 → v5 325.22).

Measured `rounds_unmatched` is **38–64** on the forward fold and 367–479 on the jan/jul folds.
Model B is nowhere near its cap either.

### 1.5 The one capacity datapoint we already own, read correctly

The only capacity experiment in our history is model B 127/50 → 255/20. CV full said −34.8 s; the
200-row-drop robust reading said **−0.06 [−0.61, +0.52]**, i.e. nothing; the leaderboard said
**+13.20** (`docs/README.md:110`). **[S]** The robust statistic was right and was overruled.

This is a warning about **tail-driven CV on a 22 k-row model**, not evidence that capacity is
harmful on the 2.06 M-row model A. The two regimes differ by two orders of magnitude in rows, and
Montero-Manso & Hyndman's argument (§3.1 below) is specifically about pooled sample size. Do not
cite the model B result as a reason not to run Claim A — but do cite it as the reason to judge
Claim A on the robust statistic. **[X]**

---

## 2. What Phoenix-Ops actually did, and the one correction to our brief

### 2.1 The facts as recorded

* v7 → v8 is the largest single step in their log: **278.6349, Δ −7.1141 official**
  (`results/07-repo-phoenix-ops.md:310`). **[S]**
* v8 introduces **no new columns at the ensemble level**: `arrival_boost_contest.py:15,61` reuses
  `arrival_contest.matrix`, the same 123 predictors as v7's LightGBM
  (`results/07-repo-phoenix-ops.md:182-184`; `results/18-turnaround-features.md:25,518-520`). **[S]**
* Exact v8 estimator: CatBoost RMSE, `depth=9, learning_rate=0.05, l2_leaf_reg=7,
  border_count=254, iterations=4999`, `random_seed=20260907`, GPU, `thread_count=2`, **no early
  stopping** (`results/07-repo-phoenix-ops.md:186-187, 212, 218-219`). **[S]**
* Fitting target: `(y − offset).clip(-7200, 7200)`. **Evaluation labels are never clipped**
  (`results/07-repo-phoenix-ops.md:189-190`). **[S]**
* Blend: flat **0.5 · v7 + 0.5 · Cat123**, specialist-scope rows carried through unchanged
  (`results/07-repo-phoenix-ops.md:191`; `results/18-turnaround-features.md:543-544`). **[S]**
* The previous step, v7, was the *same 26 arrival features* entering a **679-tree LightGBM at 25 %**
  weight: Δ **−2.6224** (`results/07-repo-phoenix-ops.md:309`; `results/18-turnaround-features.md:32-34`). **[S]**
* Capping the **label** scored **434.145 against 382.284** and was rejected
  (`results/07-repo-phoenix-ops.md:326-327, 335`). Capping the **fitting residual** at ±7200 is
  their standard from v8 onward (`results/07-repo-phoenix-ops.md:212-216, 377`). **[S]**
  **These are different operations and the plan below never conflates them.**
* Blend weights are algebraic, not tuned: `v10 = v9 + 0.375·(Cat153 − Cat123)`, where
  0.375 = 0.5 (v8's weight) × 0.75 (v8's share of v9)
  (`results/07-repo-phoenix-ops.md:262, 268-269`). Components are floored at 0 individually; the
  **difference is never clipped** and a negative result raises rather than being clamped
  (`results/07-repo-phoenix-ops.md:270-272`). **[S]**
* Their one attempt at *tuned* weights — airport-specific blend weights — scored **323.712 against
  the fixed 323.377** and was rejected (`results/07-repo-phoenix-ops.md:343`). **[S]**

### 2.2 The correction: v8 is not purely capacity

Work their algebra forward (`results/07-repo-phoenix-ops.md:255-266`) **[X]**:

```
v5 = 0.25·v4 + 0.75·Cat93(depth 9, 4998 trees)
v7 = 0.75·v6 + 0.25·LGB123        with v6 ≈ v5
   = 0.1875·v4 + 0.5625·Cat93 + 0.25·LGB123
v8 = 0.5·v7 + 0.5·Cat123deep
   = 0.09375·v4 + 0.28125·Cat93 + 0.125·LGB123 + 0.5·Cat123deep
```

So their incumbent **already contained a 4,998-tree depth-9 CatBoost at 0.5625 weight**
(`results/07-repo-phoenix-ops.md:209, 257`). v8 did two things at once:

1. it gave the deep CatBoost **30 columns it had never seen** (93 → 123: the 26 arrival features
   plus the 4 service-name features, `results/07-repo-phoenix-ops.md:96-102`); and
2. it raised the deep-CatBoost family's total weight from 0.5625 to **0.78125**.

"Zero new features" is true of the *ensemble's* feature pool and false of the *deep model's* input.
**[X]** This matters for our expected effect in two opposite directions:

* It **lowers** the transfer of the −7.1141 number, because part of it is information, not capacity.
* It **raises** the relevance of Claim B for us, because we have **no deep model at all**: our
  entire prediction is one 127-leaf LightGBM. Handing our existing 89 columns to a 4,999-tree
  depth-9 model is the change v5 was for them (their 65→70 feature CatBoost at depth 9 / 2,980
  trees, `results/07-repo-phoenix-ops.md:206`), not the change v8 was.

**Therefore: do not budget −7 s.** The defensible band is §6.

### 2.3 The measurement trap, in their own log

Their 30 following-context features were worth **+0.283203 s** through a 679-tree LightGBM —
below their own 0.5 s gate, **rejected** — and **−1.4181 s** official through the 4,999-tree
depth-9 CatBoost (`results/07-repo-phoenix-ops.md:352, 314, 356-357`). **[S]**
Slot 17 repeats the warning and points it at us specifically: *"the gain is model-dependent
(CatBoost depth 9, 4,999 trees) and does not reproduce in a LightGBM of 679 trees … Our model B is
LightGBM; that is the closer of the two controls, and it is the one that failed"*
(`results/17-congestion-features.md:543`). **[S]**

Read plainly: **testing a feature family with a shallow model produces false negatives.** Our
10-experiment rejection log was produced entirely under a 127-leaf / ~978-round LightGBM
(`docs/features_log.md:8`: *"model B 127 葉、シード 1"*, forward fold, `--skip-predict`). **[S]**
§5 handles the consequences.

---

## 3. Claim A — a much larger LightGBM

### 3.1 Is this the control nobody ran?

Yes, with one qualification.

* Montero-Manso & Hyndman, IJF 37(4):1632-1653, verbatim: *"the complexity of the local approach
  grows with the size of the set, while the complexity of the global approach remains constant …
  a global approach can be at least as complex as the sum of all individual [local models] in the
  set. This leads us to **prescribe the use of comparatively more complex hypothesis classes for
  global algorithms**."* Formally `∏ᵢ|Hᵢ| = |J|`
  (`results/12-per-group-specialists.md:334-345`). **[S]**
* Slot 12's operational reading: for K = 10 airports the honest competitor to ten per-airport
  models *"is not our current global model — it is a substantially larger global model"*, and
  *"This is the decisive control experiment and it is missing from every competitor log I read"*
  (`results/12-per-group-specialists.md:347-352, 616-626`). **[S]**

**Qualification [S]:** two competitors did run a partial version, and both found capacity helped.
OpenAir E68 found `2000×127` and `1500×255` both beat their shipped `n_estimators=400,
num_leaves=31` (`results/18-turnaround-features.md:552-554`). Phoenix-Ops's unsubmitted v13 was a
LightGBM183 at **2,716 trees (4 × 679)** replacing LGB153@679 at 0.125 weight, worth 0.5006 s
locally (`results/07-repo-phoenix-ops.md:265, 315, 391`). Scaling that to full weight implies
roughly **4 s local** from LightGBM capacity alone **[X]** — real, but well under their CatBoost
step. So the prior is: **LightGBM capacity helps, by less than a family swap.**

Slot 18 ranks the modes and puts this rung at the top:
*"hard constraint ≪ post-hoc correction < feature < feature + capacity"*, and
*"'more post-processing' is a documented dead end; 'more capacity on the primary residual'
(Phoenix-Ops v8, OpenAir E68 900×63 → 2000×127) is the documented live one"*
(`results/18-turnaround-features.md:500-506`). **[S]**

### 3.2 The design

Model A only. Model B is untouched throughout P7 — it has 22 k training rows, its own capacity
result already went badly (§1.5), and §6 of `docs/README.md` calls changes confined to unmatched
rows unmeasurable. Every config below therefore leaves `unmatched_*` at the `configs/main.yaml`
values.

Everything is a **config change only**: `params.num_leaves`, `params.min_data_in_leaf`,
`params.learning_rate`, `num_rounds`, `early_stopping`, `n_seeds`, `folds`. No code change is
needed for Claim A (`src/prc/pipeline.py:128-137` merges a `params:` block onto the defaults).

**Stage A1 — leaves ladder, one axis, `n_seeds: 1`, `folds: [forward]`, `--skip-predict`.**

| config | num_leaves | min_data_in_leaf | lr | rounds cap / es | expected rounds **[X]** | est. wall clock **[X]** |
|---|---|---|---|---|---|---|
| `p7_a0_base` | 127 | 50 | 0.05 | 2000 / 100 | ~975 | ~5 min |
| `p7_a1_leaves255` | 255 | 50 | 0.05 | 2000 / 100 | ~800 | ~7 min |
| `p7_a2_leaves511` | 511 | 50 | 0.05 | 2000 / 100 | ~700 | ~10 min |
| `p7_a3_leaves1023` | 1023 | 30 | 0.05 | 2000 / 100 | ~600 | ~16 min |

`p7_a0_base` must be run fresh rather than reusing `feat_base-b9e21a829d`: that record carries five
`include_candidates` columns (`configs/feat_base.yaml`) and `inv_baseline_forward` runs
`num_threads: 2` and `n_seeds: 3`. One clean baseline at the exact settings of the ladder costs
5 minutes and removes every confound. **[X]**

Keep `feature_fraction: 0.8` here. It is a regulariser and changing it at the same time as leaves
would confound two axes; the `feature_fraction: 1.0` protocol belongs to feature screening
(`docs/features_log.md:66`), which is §5, not this.

**Stage A2 — lower the learning rate at the winning leaf count.** Only if A1 shows a monotone
improvement. `learning_rate: 0.02`, `num_rounds: 6000`, `early_stopping: 200`, `n_seeds: 1`.
Expected ~2,500–3,500 selected rounds; ~40 min at 511 leaves **[X]**.

**Stage A3 — seed-average the finalist.** `n_seeds: 3` on the A2 winner, forward fold. Linear cost:
~2× A2. Run only if A2 passed its gate, because seed averaging is orthogonal (§1.3) and adding it
earlier just multiplies the ladder's cost by three.

### 3.3 The two mechanisms by which Claim A can fail, named in advance

1. **The early-stopping holdout is contaminated by the label-derived encodings.**
   `predict_fold` calls `fit_encodings(train_f, cfg)` over the **whole** training side
   (`src/prc/train.py:367`) and joins the nine columns onto those same rows
   (`src/prc/train.py:368`, `ENCODING_COLS` at `:43`). `_fit` then carves the inner holdout out of
   the already-encoded frame (`src/prc/train.py:314`). So the rows early stopping is scored on
   **contributed their own labels to `ref_p10` / `arh_med` / `apt_med`**. This is the
   un-cross-fitted target encoding SUMMARY §C-1 flags (`SUMMARY.md` §C-1, citing
   `src/prc/train.py:367-368` and CatBoost arXiv:1706.09516 §3.2). **[S]**
   A larger model exploits leaky columns harder, so the selected round count will be biased
   **upward** exactly when capacity rises. **[X]** Symptom to watch: `rounds_matched` going *up*
   as leaves go up, instead of down. If that happens, the ladder is measuring leak exploitation,
   not capacity, and the fix (cross-fitted encodings) is a prerequisite, not a nicety.
2. **`min_data_in_leaf` and the day-rollover rows.** `configs/main.yaml:4-9` records that at
   255 leaves / min_data 20, model B isolated ~33 day-rollover rows and then fired on 2026 rows
   that merely resembled them. Model A has 80× the rows so the same arithmetic does not apply, but
   the failure mode is the same shape. Keep `min_data_in_leaf ≥ 30` and inspect the
   prediction distribution (count of predictions > 12 h) before shipping, as `configs/main.yaml:8`
   did. **[S]** for the precedent, **[X]** for the transfer.

---

## 4. Claim B — a residual boost model on clipped residuals

### 4.1 What the base prediction is

Our model A is **already** a residual model: it fits `g(x) = LABEL − PROXY` and predicts
`PROXY + g` (`src/prc/train.py:385, 396`), where `PROXY = mvt_minus_aobt3` (`src/prc/train.py:40`).
That is the same axis Phoenix-Ops are on (`results/07-repo-phoenix-ops.md:34-37`). **[S]**
So Claim B is not "add a residual stage"; it is **"fit the same residual with a much bigger,
different-family model and blend"**.

Define:

* `P₀` = the incumbent prediction, i.e. the current pipeline output — model A on matched rows,
  model B on unmatched rows, floored at 0 by `_assemble` (`src/prc/train.py:433`).
* `C` = the new boost model, fitted on the **matched rows only**, on the **same design matrix**
  model A gets (`cols_a` at `src/prc/train.py:382`: the 89 visible numerics with `PROXY` moved
  last, plus the 10 categoricals).
* `P_C = PROXY + g_C`, floored at 0.

### 4.2 How the residual is formed, and exactly where the ±7200 clip applies

```
fit target for C   :  clip( LABEL − PROXY , −7200 , +7200 )      ← the ONLY clip
fit target for A   :  LABEL − PROXY                              ← unchanged, NOT clipped
evaluation labels  :  LABEL                                      ← never clipped, anywhere
blend output       :  max(0, ·) only, as today (train.py:433)    ← not a ±7200 clip
```

* **Clip the fitting residual. Never the label.** `results/07-repo-phoenix-ops.md:189-190, 326-327,
  335`: label capping scored 434.145 against 382.284 and was rejected; residual capping at ±7200 is
  their standard from v8 on. **[S]**
* **Do not clip model A's target.** Leaving the incumbent bit-exact is what makes the blend
  measurable as one change. It also preserves every prediction that has already been validated —
  the same discipline as Phoenix's "protect the scope" pattern
  (`results/07-repo-phoenix-ops.md:236-239`). **[S]**
* **Do not clip the blended output** beyond the floor at 0 that `_assemble` already applies.
  Phoenix explicitly never clip a blend result (`results/07-repo-phoenix-ops.md:270-272`). **[S]**

What fraction of our matched rows the clip actually touches is **unmeasured** and is the first
thing Stage B0 should print: `mean(|LABEL − PROXY| > 7200)` over matched training rows. If it is
near zero the clip is a no-op and can be reported as such; if it is material, the clip is doing
real work and the number belongs in the record. **[X]**

### 4.3 The estimator

Phoenix's v8 spec verbatim (`results/07-repo-phoenix-ops.md:186-187`;
`results/18-turnaround-features.md:538-541`) **[S]**, with the deviations our box forces:

| knob | Phoenix | ours | why |
|---|---|---|---|
| library | CatBoost RMSE | CatBoost 1.2.10 (installed: `.venv/lib/*/site-packages/catboost-1.2.10.dist-info`) | same |
| depth | 9 | **9** | copy |
| learning_rate | 0.05 | **0.05** | copy |
| l2_leaf_reg | 7 | **7** | copy |
| border_count | 254 | **254** | copy |
| iterations | 4999, **no early stopping** | **4999, no early stopping** | copy; frozen before reading the result |
| random_seed | 20260907 | **20260907** | copy; declared up front |
| task_type | GPU | **CPU** | **`nvidia-smi` is absent on this box; there is no GPU.** `nproc` 4, 15 GB RAM **[S]** |
| cat_features | all object columns | see §4.4 | cost control |

"No early stopping" is not laziness on their part — it is the mechanism that keeps the tree count
from being fitted to the holdout (`results/07-repo-phoenix-ops.md:218-219`). Copy it, and declare
4999 in the config before the run. **[S]**

### 4.4 Categoricals — the one place we must deviate, and the fallback

`_matrix` currently converts our 10 categoricals to integer physical codes cast to float32
(`src/prc/train.py:278-281`). Two options:

* **B-cat (faithful):** pass the raw strings to CatBoost as `cat_features`. This turns on CTR
  machinery. `STAND_mvt` has 1,899 levels (`docs/features_log.md:85`) and the default
  `max_ctr_complexity=4` builds combinations across all 10 categoricals — on 2.06 M rows this is
  the main memory and time risk. **Bound it: `max_ctr_complexity=1`** (and record the deviation).
* **B-num (cheap):** pass the same integer codes model A gets, as numerics. No CTRs, cheapest,
  and the design matrix is then *literally identical* to model A's, which makes the comparison a
  clean family-and-capacity comparison with nothing else moving. Weaker categorical handling. **[X]**

**Run B-num first.** It is the cheaper arm and it is the stricter test of the capacity claim. Only
if B-num passes and we want the last of the gain is B-cat worth its cost.

### 4.5 The blend, and why the weight is fixed rather than tuned

```
matched rows   :  P = 0.5 · P₀ + 0.5 · P_C          (both floored at 0 first)
unmatched rows :  P = P₀                            (bit-identical; model B is not touched)
```

Fixed at **0.5**, declared before the run. Four reasons:

1. **Source.** Phoenix's v8 blend is a flat 0.5/0.5 declared before the result
   (`results/07-repo-phoenix-ops.md:191`; `results/18-turnaround-features.md:543-544`). **[S]**
   Their weights are algebraic throughout (`results/07-repo-phoenix-ops.md:268-269`). **[S]**
2. **Their one tuned-weight experiment lost.** Airport-specific blend weights: 323.712 against the
   fixed 323.377 (`results/07-repo-phoenix-ops.md:343`). **[S]**
3. **Our instrument cannot resolve a weight sweep.** The paired 95 % CI on matched rows is
   0.14–1.17 s wide depending on the pair (`docs/README.md:144`). **[S]** Sweeping
   {0.25, 0.5, 0.75} and keeping the best is taking the maximum of three draws from that noise — it
   is precisely the tail-lottery the §6 rule exists to block. And the leaderboard is best-of-all-
   submissions with no private set (`SUMMARY.md` §A **[S]**), which makes sweeping *there* both
   tempting and against the stated spirit of the anti-probing rule (`SUMMARY.md` §A). **[X]**
4. **The loss from a wrong fixed weight is second order.** For two predictors with equal error
   variance σ² and error correlation ρ, `MSE(w) = σ²(w² + (1−w)² + 2w(1−w)ρ)`, minimised exactly at
   w = 0.5. Using 0.5 when the true optimum is 0.35 costs `2σ²(1−ρ)(0.5−0.35)² = 0.045 σ²(1−ρ)`, a
   fraction of a percent of MSE for any plausible ρ. **[X]** The weight only matters if the two
   members are *unequally* accurate — which is what the abort condition below tests.

**One predeclared abort condition, not a sweep.** Score `P_C` **alone** on the forward fold before
blending. If `C` alone is worse than model A alone by more than **10 s** matched RMSE at the
200-row-drop level, do not blend at 0.5 — report and stop. This is a sanity gate on member quality,
declared in writing before the run; it is not weight selection. **[X]**

### 4.6 Interaction with the A/B routing

The boost applies **only to rows where `PROXY` is not null** — the model A population
(`src/prc/train.py:379`). Rows where `PROXY` is null have no residual to boost, are handled by
model B (`src/prc/train.py:398-422`), and are carried through unchanged. This is exactly Phoenix's
"protect the specialist scope" rule (`results/07-repo-phoenix-ops.md:236-239`). **[S]**

It also buys us the measurement we need: the change moves **only** matched rows, which is the
population `docs/README.md:139` calls measurable, and leaves untouched the population it calls
unmeasurable. A P7 comparison is therefore a clean matched-row comparison. **[X]**

### 4.7 Interaction with the LIRF reparameterisation

`unmatched_sched_reparam` shifts model B's target by `mvt_minus_sched` at LIRF and adds it back
after prediction (`src/prc/train.py:241-267, 409-413`; `configs/submit_v10_reparam.yaml`). It is
gated on `cfg_b`, which is only ever built inside the **unmatched** branch
(`src/prc/train.py:408-409`). The boost touches only matched rows.

**They act on disjoint row sets, so their effects are additive by construction, and no interaction
test is needed.** **[X]** What *is* needed is one assertion, cheap and worth writing: with the
boost on and the reparameterisation on, the predictions on `PROXY`-null rows must be
**bit-identical** to the reparameterisation-only run. If they are not, the boost has leaked out of
its scope.

Sequencing: measure the boost against `configs/main.yaml` (reparam **off**), one change at a time.
Ship the two together only after each has passed on its own.

### 4.8 What code would have to be added (spec only — nothing was written)

`src/prc/train.py` currently has no CatBoost import and no blending stage. The smallest honest
shape, following the file's own invariants (`src/prc/train.py:14-20`):

* A `boost_*` block on `TrainConfig` (`enabled`, `depth`, `lr`, `l2`, `border_count`, `iterations`,
  `seed`, `residual_clip`, `blend_weight`, `cat_mode`), defaulting to off so nothing changes for
  existing configs — the same pattern as `unmatched_sched_reparam`
  (`src/prc/train.py:108`, off by default, with the reason written down).
* Inside `predict_fold`'s matched branch (`src/prc/train.py:380-396`), after `g` is produced: fit
  `C` on the **same** `m_tr` rows, the **same** `cols_a` matrix and the **same** early-stopping-free
  round count, with target `np.clip(y − proxy, −7200, 7200)`; blend in prediction space.
* The blend must happen **after** each member is floored at 0 and **before** the single
  `clip(lower_bound=0)` in `_assemble` (`src/prc/train.py:433`) — floor each member, blend, do not
  clip the result (`results/07-repo-phoenix-ops.md:270-272`). **[S]**
* `info["boost_*"]` fields (iterations, clip hit-rate, member RMSEs) so the JSONL record says what
  ran — the lesson of `src/prc/train.py:526-529` (v6 shipped nine screening columns because the
  record logged frame width instead of what the model saw). **[S]**

---

## 5. The false-negative trap: re-testing what we rejected

### 5.1 The problem stated precisely

Every rejection in `docs/features_log.md` was measured under **127 leaves / lr 0.05 / ~978 rounds /
`n_seeds` 1 / forward fold**, against a hash-noise control column at `feature_fraction: 1.0`
(`docs/features_log.md:8, 66`). **[S]** Phoenix's following-context family measured **+0.283** under
a 679-tree LightGBM and **−1.4181** under a 4,999-tree depth-9 CatBoost
(`results/07-repo-phoenix-ops.md:352, 314`). **[S]** Our screening model is on the wrong side of
that comparison.

Separately, our own measured resolution floor is **0.25 s**
(`docs/features_log.md:163, 253`; `results/17-congestion-features.md:534-535`). **[S]** A family
whose true effect is 0.4 s under a bigger model is invisible to the screen we used.

### 5.2 The re-test protocol

Do **not** re-run the ten experiments one by one. That would cost ten runs at the *higher* per-run
price and would still be answering the wrong question first.

1. **Prerequisite:** a capacity winner must exist. Re-tests run under the Stage A2 (or, if Claim B
   passes, the Stage B) model, not under the 127-leaf model.
2. **Keep the noise control.** `docs/features_log.md:262` is explicit: without the matched
   noise column, experiments 1–3 would have produced a wrong "all three are harmful" conclusion,
   because adding any single column moves the score ±0.5 s at `feature_fraction: 0.8`. **[S]**
   Keep `feature_fraction: 1.0` and candidate-vs-noise, with the column count held fixed.
3. **Batch by family, not by column.** Test all six congestion columns against six noise columns in
   **one** run. If the batch moves, split it; if it does not, the family is closed at that capacity
   for one run instead of six. **[X]**
4. **Re-declare the decision rule before running**, not after (§7).

### 5.3 Which recorded rejections are candidates, ranked

| rank | candidate | recorded result | why capacity could change it | verdict |
|---|---|---|---|---|
| 1 | **`planned_flight_s`** (`ARVT_1 − EOBT_1`) | −0.21 [−0.49, **+0.08**] at 200-drop; −0.99 [−1.71,−0.35] at 20; −0.25 [−0.41,−0.10] at 2,000; matched 218.5 → **218.1**; corr +0.2575 (`docs/features_log.md:210-225`) **[S]** | The nearest miss in the whole log — three of four drop levels exclude zero. It is a *duration* feature, and slot 17's judgement 1 is that every congestion **duration** family has a measured positive while every **count** family has a strong negative (`results/17-congestion-features.md:557-562`) **[S]** | **Re-test first.** |
| 2 | **`dest_distance_km`** | +0.02 [−0.24,+0.25] at 200-drop; **unmatched −42 s**; tail-only (`docs/features_log.md:165-187`) **[S]** | A tail-only effect is what a low-capacity model does with a feature it can only use in a few regions. More capacity is the specific remedy. | **Re-test, but note** it moves the unmeasurable population (`docs/README.md:139`), so a pass there is not shippable on CV alone. |
| 3 | **`queue_ahead`, `bank_position`, `stand_rwy_share`, `active_runways_60m`, `dep_arr_ratio_30m`, `congestion_vs_usual_30m`** — one batch | −0.16 / −0.19 / −0.10 / −0.06 / +0.03 / +0.09, all CIs straddling zero (`docs/features_log.md:233-238`) **[S]** | This is the family Phoenix's trap case belongs to, which argues for a re-test. | **Re-test as ONE batch, after 1 and 2.** Counter-evidence is heavy: slot 17 puts counts in the dead column and says *"I would not spend a measurement slot here"* (`results/17-congestion-features.md:548, 557-562`) **[S]**, and Phoenix's own traffic-feature variant scored 434.145 vs 382.284 (`results/07-repo-phoenix-ops.md:326`) **[S]**. One batched run is the right price for that prior. |
| 4 | **`op_offset_med` / `op_offset_n`** | −0.17 [−0.40,+0.05] at 2 folds; **sign flipped to +0.15** [−0.21,+0.52] at 3 folds (`docs/features_log.md:200-208, 239-241`) **[S]** | **Deprioritise.** The mechanism argument is *"木は行をまたぐ集計を作れない"* (a tree cannot aggregate across rows, `docs/features_log.md:125`) **[S]** — and that is still true of a bigger tree. Capacity has no mechanism here, and the sign already flipped once resolution improved. | Skip unless 1–3 all pass. |

**One item that is on this list but is not a re-test**, and belongs to the turnaround theme rather
than P7: `stand_arrival_taxi` / `stand_arrival_delay` were screened on branch
`codex/feature-engineering` at `8bdf4f9` by *correction* value and dropped, never refitted as
features; `stand_arrival_gap` measured **−3.242 s [−4.282, −2.225]** as a feature against
**−0.781 s** as a correction (`SUMMARY.md` §D; `results/18-turnaround-features.md:559-582`). **[S]**
That is a bigger number than anything in `docs/features_log.md`, it is already measured under our
*current* model, and it does not need capacity to justify it. **If P7 competes with it for compute,
P7 loses.** Recorded here so the ordering is deliberate.

---

## 6. Compute reality

Box: **4 CPUs, 15 GB RAM, no GPU** (`nproc`, `free -g`, `nvidia-smi: command not found`). **[S]**
LightGBM 4.7.0 and **CatBoost 1.2.10** are both installed in `.venv`. **[S]**

### 6.1 Cost model, calibrated on our own logs

From `feat_base-b9e21a829d`: 305.5 s for the forward fold at `n_seeds: 1`, 978 selected rounds,
4 threads. That run performs one early-stopping fit plus one refit, so **≈ 0.156 s per boosting
round** at 127 leaves / 1.74 M matched rows / 99 columns / 4 threads. **[X]**

```
T_lgb  ≈  0.156 s × rounds × (leaves/127)^0.7 × (n_seeds + 1)  +  ~60 s overhead      [X]
```

The `^0.7` is the usual sublinear scaling from histogram subtraction; it is an assumption and
Stage A1 measures it directly.

| run | est. wall clock **[X]** |
|---|---|
| A1 ladder, 4 configs, `n_seeds: 1`, forward, `--skip-predict` | **~40 min total** |
| A2 (511 leaves, lr 0.02, ~3,000 rounds, `n_seeds: 1`) | **~40 min** |
| A3 (same, `n_seeds: 3`) | **~80 min** |
| B calibration probe (CatBoost, 200 iterations) | **~10 min** |
| B full (CatBoost depth 9, 4,999 iterations, CPU, B-num) | **0.5–2.5 h — measured by the probe** |
| submission run at current capacity (full refit + ranking) | ~15 min on top of CV (`submit_v10_reparam`: pipeline 1,598.1 s total vs CV 686.6 s) **[S]** |
| submission run at 511 leaves / 3,000 rounds / `n_seeds: 3` | **~1.5 h** **[X]** |

The CatBoost figure is the one I cannot bound from our logs — we have never run CatBoost here.
The range comes from CatBoost's oblivious-tree work per iteration at depth 9 (9 full passes over
1.74 M × 99 values at 254 bins) being roughly 1–3× LightGBM's per-tree work at 127 leaves. **[X]**
**That is why Stage B0 exists: do not guess, time 200 iterations and extrapolate.**

### 6.2 Memory

LightGBM at 511 leaves: the stacked float32 design matrix is 2.08 M × 99 × 4 B ≈ **825 MB**
(`src/prc/train.py:273-282`), the fancy-indexed inner-holdout copies are the documented peak
(`src/prc/train.py:346`) at ~+640 MB, plus LightGBM's binned copy ~206 MB and a booster of
511 × 3,000 × ~24 B ≈ 37 MB. **Peak ~2.5–3 GB.** Comfortable in 15 GB. **[X]**

CatBoost B-num: quantized 2.06 M × 99 × 1 B ≈ **206 MB**, model 4,999 × 512 leaves × 8 B ≈ 20 MB.
**Peak ~1.5–2 GB.** **[X]**
CatBoost B-cat at default `max_ctr_complexity=4`: unbounded in practice with a 1,899-level stand
column; **this is the arm that can OOM the box.** Bound it to 1 or do not run it. **[X]**

### 6.3 Staging rule: an expensive run is only reached after a cheap one justified it

```
Stage 0   free      Write the predeclared gate (§7). Print mean(|LABEL−PROXY| > 7200) on
                    matched training rows. Confirm no GPU, confirm CatBoost imports.
   ↓ gate: the gate document is written and dated BEFORE any run
Stage A1  40 min    Leaves ladder, n_seeds 1, forward, --skip-predict.
   ↓ gate: 511 (or 255) beats 127 at the 200-row-drop level with a CI excluding zero,
           AND rounds_matched does not rise with leaves (§3.3 mechanism 1)
Stage A2  40 min    lr 0.02 at the winning leaf count.
   ↓ gate: beats the A1 winner on the same statistic
Stage A3  80 min    n_seeds 3 on the A2 winner.   [optional, only for a submission]

Stage A-blend  0    Blend the A2 winner 0.5/0.5 with the incumbent — OFFLINE, from the two
                    OOF parquet files (src/prc/train.py:534). No training. This isolates
                    "blending two LightGBMs" from "a different family" for free.
   ↓ gate: does the blend beat the better single member? (§7 counter-evidence)

Stage B0  10 min    CatBoost 200-iteration timing probe on the forward fold's matched rows.
   ↓ gate: extrapolated 4,999-iteration cost ≤ 4 h. If not, drop to depth 8 or
           iterations 2,500 and record the deviation from Phoenix's spec.
Stage B1  0.5-2.5h  CatBoost 4,999, B-num, forward fold, write its OOF.
   ↓ gate: C alone is not worse than A alone by >10 s (§4.5 abort condition)
Stage B2  0         Blend 0.5/0.5 offline from the two OOF files. Judge.
Stage B3  ~1.5 h    Full-data refit + ranking prediction for whichever finalist passed.
```

**The offline-blend stages are the design's main cost saving.** `run_cv` already writes
`oof_<exp_id>.parquet` for every run (`src/prc/train.py:534`), so every blend weight, every
member combination and every drop-level bootstrap can be evaluated with arithmetic on files that
already exist. **Caveat [S]:** SUMMARY §C-2 records that the OOF parquet has no `fold` column, that
folds carry different `train_months`, and that coverage differs between experiments
(672,428 against 328,009). **All P7 runs must therefore use `folds: [forward]`**, which gives every
record identical coverage of 328,009 rows and makes an offline blend safe. Do not blend across
records with different fold lists.

---

## 7. How this will be judged — declare before running

### 7.1 The rule as coded, and the problem with it here

`src/prc/compare.py:112-137`: adopt **only** when the full-data 95 % CI *and* the 200-row-drop
95 % CI both exclude zero on the improving side. `docs/README.md:133-137` records the correction —
the code reads the full-data CI first and rejects without looking further if it straddles zero —
and gives the worked example: `det_a → main-48c19bc830` (seeds 1→3) was **−2.58 [−2.96,−2.15] at
the 200-drop level** and was still **REJECTed** because the full-row CI was [−3.75, +0.55]. **[S]**

**A capacity change will almost certainly produce exactly that pattern.** More capacity moves the
tail; the tail is where the full-row CI's width comes from (10 rows carry 24 % of the metric,
`docs/README.md:123-124`). **[S]** Running P7 under the rule as coded has a high prior of
rejecting a change worth several seconds. **[X]**

Two honest options. **Pick one and write it down before Stage A1 runs.**

* **Option 1 — apply the rule as coded.** Cheapest, most conservative, consistent with every
  previous decision. Accept that P7 may be rejected with a −3 s robust point estimate.
* **Option 2 — predeclare a P7-specific gate**, dated, written into this file before any run, and
  modelled on Phoenix's (`results/07-repo-phoenix-ops.md:378-384`) **[S]**:
  1. matched-row RMSE improves by **≥ 0.5 s** at the 200-row-drop level with a CI excluding zero;
  2. improvement at **≥ 7 of 10** airports (`metrics`/`per_airport` already computes this,
     `src/prc/train.py:499`);
  3. improvement on **≥ 40 of 62** UTC days of the forward fold;
  4. the full-row point estimate is **not worse** (the CI may straddle zero; the point estimate
     may not be on the wrong side).

  Option 2 is defensible **only** because the rule's own provenance is already in question: the
  "verified on 2 LB cases" claim is marked **UNDETERMINED** in `docs/README.md:119-121` and
  `SUMMARY.md` §B, and the "62 s unmatched CI" premise is recorded as not reproducible
  (`docs/README.md:140-146`). **[S]** It is **not** defensible after seeing a result. If Option 2
  is chosen, this section must be committed before the first run.

### 7.2 The counter-evidence this plan must keep separate

"More capacity" and "more model families" are different claims with different evidence, and the
evidence for the second is much weaker than for the first:

* Slot 11: cross-family blending buys **0.3 %–1.8 % over the best single member** — for us
  **0.9–4.5 s**, target band 283.0–279.2 (`results/11-ensembling-patterns.md:14, 333, 538, 578`). **[S]**
* PRC 2025's **rank 1 shipped a single LightGBM** with no blend, RMSE 200.83
  (`results/11-ensembling-patterns.md:143-157`). **[S]**
* Slot 20's benchmark table: cross-family portfolio 0.365 against best single-family tuned 0.524 —
  *"cross-family blending bought 0.159 — slightly more than tuning"*
  (`results/20-tabular-comp-craft.md:771-775`). **[S]**

The **Stage A-blend** step is what separates them, and it is free: it gives a
LightGBM-blended-with-LightGBM number to compare against the CatBoost-blended number. If
A-blend ≈ B-blend, the gain is capacity-plus-averaging and we never need CatBoost in production.
If B-blend ≫ A-blend, the family genuinely contributes. **[X]**

---

## 8. Expected effects, risks, effort — per stage

| stage | effort | expected effect | source of the expectation | risk |
|---|---|---|---|---|
| **0** predeclare + clip hit-rate | 30 min, no compute | — | — | Skipping it is the only way this theme becomes unfalsifiable. |
| **A1** leaves ladder | 40 min | **−0.5 to −3 s** matched | OpenAir E68 found 2000×127 and 1500×255 both beat 400×31 (`18:552-554`) **[S]**; slot 12's `∏|Hᵢ|=|J|` prescription (`12:340-345`) **[S]** | Leak-driven round inflation (§3.3-1). Day-rollover-style leaf isolation (§3.3-2). |
| **A2** lower lr | 40 min | **−0.3 to −1.5 s** on top of A1 | Standard GBDT lr/rounds trade; Phoenix's v13 LightGBM at 4× trees was worth 0.5006 s local at 0.125 weight ⇒ ~4 s at full weight (`07:265, 315`) **[S]/[X]** | Hits the 6000 cap; then the result is a floor, not an estimate. |
| **A3** seeds | 80 min | **−0.5 to −2.5 s** | `det_a → main-48c19bc830`: −2.58 [−2.96,−2.15] at 200-drop (`README:135-137`) **[S]**; 2024 winner 1,612 → 1,562 from 20 seeds (`train.py:115`) **[S]** | Rejected by the rule as coded (§7.1). Pure cost, no new information. |
| **A-blend** | free | **−0.3 to −1.5 s** over the better member | slot 11's 0.3–1.8 % band (`11:333, 538`) **[S]** | OOF coverage mismatch — mitigated by `folds: [forward]` everywhere (§6.3). |
| **B0** probe | 10 min | — | — | None. This is the stage that prevents a wasted 4 h. |
| **B1+B2** CatBoost blend | 0.5–2.5 h | **−1 to −5 s**, *not* −7.1 | Phoenix v8 −7.1141 official (`07:310`) **[S]**, discounted because part of it is +30 columns into their deep model and a reweighting from 0.5625 to 0.78125 (§2.2) **[X]**; slot 11's 0.9–4.5 s blend band (`11:538`) **[S]** | CPU cost overrun; `max_ctr_complexity` OOM in the B-cat arm; CatBoost sees the same un-cross-fitted encodings (`SUMMARY` §C-1) **[S]**. |
| **B3** submission | 1.5 h | — | — | Local→official transfer is erratic in both logs: Phoenix's v8 gained 2.77 s locally and 7.11 officially, v11 gained 0.57 locally and 0.20 officially (`07:396-397`) **[S]**. Expect the magnitude to be wrong even when the sign is right. |
| **5** re-tests | 1 run per family at the new capacity | unknown by construction | Phoenix's +0.283 → −1.4181 same-columns flip (`07:352, 314`) **[S]** | Re-testing at higher capacity is a *new* multiple-comparison surface. Cap it at three runs (§5.3 ranks 1, 2, 3) and declare them in advance. |

---

## 9. What would show this failed

Written before any run, so it cannot be renegotiated afterwards.

**Claim A has failed if:**

* The leaves ladder is **flat or non-monotone** at the 200-row-drop level — 255 and 511 both within
  ±0.25 s (our resolution floor, `docs/features_log.md:163`) **[S]** of 127. That is the direct
  refutation of slot 12's prescription on our data, and it is a publishable negative: it would mean
  our 89 columns, not our hypothesis class, are the binding constraint.
* `rounds_matched` **rises** as `num_leaves` rises. Then the ladder is measuring how much better a
  bigger model exploits the un-cross-fitted encodings (§3.3-1), and the honest next step is
  cross-fitted encodings, not more leaves.
* CV improves by several seconds while the 200-row-drop CI straddles zero — the exact signature of
  the model B 255-leaf failure (`docs/README.md:110`, CV −34.8 / robust −0.06 / LB **+13.2**). **[S]**
  **If we see that signature, stop. Do not submit.** The precedent cost us 13 s.

**Claim B has failed if:**

* `C` alone is worse than model A alone by more than 10 s matched RMSE at the 200-row-drop level
  (§4.5). The family swap is then not a peer and a 0.5 weight would be actively harmful.
* The 0.5/0.5 blend does not beat the better single member — which would put us in the
  `results/11-ensembling-patterns.md:361` regime where *"the ensemble was worse than the best
  single model on test"*. **[S]**
* The clip hit-rate `mean(|LABEL − PROXY| > 7200)` is ~0 **and** B ≈ A-blend. Then there is no
  "clipped residual" effect at all, only averaging, and the whole Phoenix v8 framing does not
  transfer to our data.

**The theme as a whole has failed if** the total from A1+A2+A-blend+B is under **1 s** at the
200-row-drop level. Slot 18 positions "feature + capacity" as the top rung
(`18:500-506`) **[S]**; under 1 s would mean we are on a different curve from both Phoenix-Ops and
OpenAir, and the compute should move to the one item with a larger already-measured number on our
own data: `stand_arrival_taxi` / `stand_arrival_gap` (§5.3, `SUMMARY.md` §D, −3.242 s as a
feature). **[S]**

**Finally — the failure mode that is not a number.** If Option 2 (§7.1) is chosen *after* a run has
been read, this plan has failed as a piece of process regardless of what the score does. The rule
in `docs/features_log.md:154` is the repo's own: *"結果を見てから規則を変えるのは危険"* — changing
the rule after seeing the result is dangerous. **[S]**
