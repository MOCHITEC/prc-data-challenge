# P10 — CatBoost as a rival single model

Plan written 2026-09-21. Baseline: `resilient-hedgehog` **284.3584** (taxi-out seconds, RMSE).
Origin: research slots 11, 20, 03, 23
(`docs/competitor_research_2026-09-21/results/11-ensembling-patterns.md`,
`20-tabular-comp-craft.md`, `03-prc-2024-writeups.md`, `23-repo-col3name-satam2.md`).

**Marking convention** (same as the research slots)

* `[S]` — read from an external source; URL given inline.
* `[S-repo]` — read from this repository, from an installed package's own source, or measured
  on this machine; `path:line` or the command given.
* `[X]` — my own inference or arithmetic. Never dressed as `[S]`.

**No code was modified in producing this plan. No training run was started on competition data.**
Section 4's numbers come from a synthetic-shape microbenchmark run in the scratchpad
(`/tmp/claude-0/.../scratchpad/bench.py`, `bench2.py`, `bench_lgb.py`) and from the existing
`experiments/experiments.jsonl` records. The method and its error bars are in §4.4.

---

## 0. The recommendation up front

This theme is **a model-family measurement, not a blend**. The distinction is load-bearing and the
sourced reason is one number: on PRC 2024 ATOW — the closest published analogue to our problem —
switching family bought **50×** what blending bought `[S]`.

| comparison | RMSE (kg) | delta |
|---|---:|---|
| their Optuna-tuned LightGBM | 2,506.43 | — |
| their Optuna-tuned CatBoost | **2,215.53** | **−11.6 %** vs their LightGBM |
| their full 5-model weighted blend | 2,210.37 | **−0.23 %** vs CatBoost alone |

`[S]` Murça et al., *Open Machine Learning Models for Actual Takeoff Weight Prediction*, JOAS 3(2)
2025, Table 6 — quoted in `results/11-ensembling-patterns.md:47-57` and `results/20-tabular-comp-craft.md:742-752`.
`[S]` Their Table 8 ablation: removing CatBoost costs **+33.90 %**; removing LightGBM, the MLP and
SAINT each cost **exactly +0.00 %**; fitted weights `[1.24e-23, 1.63e-23, 8.65e-01, 4.77e-22,
1.34e-01]` (`results/11-ensembling-patterns.md:59-79`).

The plan therefore does **not** start by building a blend harness. It builds one thing — a second
model family driven through the *identical* fold structure, feature set, routing, target
parameterisation, seed loop and evaluation path — and asks one question: **does CatBoost beat our
LightGBM at predicting taxi-out seconds on our data?** Blending is downstream of a yes, and §6 gives
the result that makes us stop instead.

Three findings from this plan change the shape of the work before a single fit is run:

1. **The comparison is nearly free to make fair.** Everything except the fitter is already shared —
   folds (`cv.py:26-51`), feature selection (`train.py:223-235`), encodings (`train.py:200-220`),
   matched/unmatched routing (`train.py:379,398`), target reparameterisation (`train.py:241-267`),
   prediction assembly (`train.py:427-437`), metrics (`cv.py:58-84`) and the adopt/reject rule
   (`compare.py:112-140`). The LightGBM assumption is confined to **five places**, listed in §1.
2. **The P2 interaction is not a side note, it is the experiment.** Our hand-rolled encodings are
   worth **−37.36 s** to LightGBM and the raw categoricals only **−7.37 s** `[S-repo]`
   (§3.3). Our encodings are doing five times the work the categorical columns are, and they are
   keyed on exactly the columns CatBoost's ordered target statistics would encode. The comparison
   that is *invariant to P2* — and therefore the one to run first — is **CatBoost-without-encodings
   vs LightGBM-without-encodings** (§3.4).
3. **CatBoost's library default is the expensive configuration and nobody has evaluated it.**
   `max_ctr_complexity` resolves to **4** at realistic iteration counts `[S-repo]`, building
   combination CTRs over up to 4 of our 10 categoricals, and that costs a measured **3.1×** in wall
   clock against complexity 1 (§4.2). Both 2026 competitor repos slot 23 read ship this default
   without measuring it `[S]`.

---

## 1. Everything in the code that assumes LightGBM

I read `src/prc/train.py` (542 lines), `src/prc/pipeline.py`, `src/prc/predict.py`, `src/prc/cv.py`,
`src/prc/compare.py`, `src/prc/features.py` and `configs/main.yaml`. The LightGBM assumption lives
in five places and nowhere else.

### 1.1 The five sites

| # | site | `file:line` | what it assumes | what a second family needs |
|---|---|---|---|---|
| 1 | import | `train.py:32` | `import lightgbm as lgb` at module scope | import the fitter lazily inside the branch, so a LightGBM run never pays for CatBoost's import and vice versa |
| 2 | params dict | `train.py:56-78` | every key is a LightGBM key: `objective, learning_rate, num_leaves, min_data_in_leaf, feature_fraction, bagging_fraction, bagging_freq, lambda_l2, verbose, num_threads, deterministic, force_row_wise` | a **separate** params namespace per family; see the merge trap in §1.2 |
| 3 | params accessor + capacity override | `train.py:135-145` (`for_unmatched`), `train.py:147-152` (`lgb_params`) | `for_unmatched()` rewrites `num_leaves` / `min_data_in_leaf` by name; `lgb_params()` adds `linear_tree` / `linear_lambda` | CatBoost's equivalents are `depth` (SymmetricTree) or `max_leaves` (Lossguide only, §2.3) and `min_data_in_leaf`; CatBoost has **no** `linear_tree`, so `use_linear_tree: true` must be an error under CatBoost, not silently ignored |
| 4 | design matrix | `train.py:273-282` (`_matrix`) | one `float32` numpy array; categoricals are cast `Utf8 → fill_null("__NA__") → Categorical → to_physical() → Float32` | **hard blocker**: CatBoost rejects float categorical values outright — verified: `Invalid type for cat_feature[...]=2.0 : cat_features must be integer or string, real number values and NaN values should be converted to string` `[S-repo]` |
| 5 | the fitter | `train.py:302-357` (`_fit`) | `lgb.Dataset(..., categorical_feature=cat_idx, params=params)` (`:334,335,348`); `lgb.train(..., valid_sets=[dva], callbacks=[lgb.log_evaluation(0), lgb.early_stopping(cfg.early_stopping, verbose=False)])` (`:336-342`); `booster.best_iteration` (`:343`); seed loop rewriting `seed`/`bagging_seed`/`feature_fraction_seed` (`:352`); `model.predict(Xall[s_:e_])` (`:355`) | a parallel implementation with the *same signature and the same contract* — see §1.3 |

### 1.2 Two traps in the config path

`[S-repo]` `pipeline.py:129`:

```python
fields = {k: v for k, v in spec.items() if k in TrainConfig.__dataclass_fields__}
```

**Unknown YAML keys are silently dropped.** A config carrying `model: catboost` before
`TrainConfig` has that field runs LightGBM and logs a record that says nothing about it. A typo
(`model_familly`) does the same. The plumbing stage must add a guard that raises on any key in the
YAML that `TrainConfig` does not know, or every negative result in this theme is uninterpretable.

`[S-repo]` `pipeline.py:132-133`:

```python
if "params" in fields:
    fields["params"] = {**TrainConfig().params, **fields["params"]}
```

A config that sets `params:` inherits the whole LightGBM default dict. Feeding that to CatBoost
fails — verified `[S-repo]`:

| passed to `CatBoostRegressor` | result |
|---|---|
| `feature_fraction=0.8` | `TypeError: unexpected keyword argument 'feature_fraction'` |
| `force_row_wise=True` | `TypeError: unexpected keyword argument 'force_row_wise'` |
| `num_leaves=127` | `CatBoostError: max_leaves option works only with lossguide tree growing` |
| `verbose=-1` | accepted |

`[X]` So `params` must become family-scoped (`params` stays LightGBM's; a new `cb_params` holds
CatBoost's) rather than one dict merged over one default. Merging is what the comment at
`pipeline.py:130-131` exists to defend, and that defence only makes sense within a family.

### 1.3 The contract `_fit` must keep

`_fit(train, target_expr, cfg, feature_cols, cat_cols, predict_frames) -> (list[np.ndarray], int)`
(`train.py:302-313`). A CatBoost sibling must preserve all six behaviours, because each is an
invariant the docstrings say was a real bug:

1. **Early stopping never sees the evaluation fold.** The stopping point comes from
   `_inner_holdout_mask` — the latest `cfg.inner_holdout_months` calendar months of the *training*
   side (`train.py:285-299`) — and the final model is refitted on all training rows at a fixed
   round count. CatBoost: `od_type="Iter"`, `od_wait=cfg.early_stopping`, `use_best_model=True`,
   `eval_set=` the holdout Pool; read `model.get_best_iteration()`.
2. **The round count is scaled to the larger refit set, then capped** (`train.py:345`):
   `rounds = min(num_rounds, max(10, round(best * n_train / (n_train - n_hold))))`. Library-agnostic;
   reuse verbatim so the two arms scale identically.
3. **Frames are stacked before encoding** (`train.py:316-317`) so train and every predict frame share
   one categorical vocabulary. Keep this; it is why an unseen 2026 stand does not silently become a
   different integer in the two frames.
4. **Seed averaging** (`train.py:350-356`): `n_seeds` refits, predictions averaged. CatBoost needs a
   fresh estimator per seed with `random_seed=cfg.seed + k`; the Pool can be reused.
5. **Predictions are returned per predict-frame in model space**, and `predict_fold` adds the proxy
   or the reparameterisation shift back (`train.py:396, 412-413`). Untouched.
6. **`free_raw_data=False`** on the full-data Dataset (`train.py:348`) exists so the same object
   survives the seed loop. CatBoost's `Pool` is already immutable-and-reusable.

`[X]` Everything else in the file — `predict_fold`, `_assemble`, `feature_parity_report`, `run_cv`,
the OOF parquet, the JSONL record — is family-agnostic today and stays that way. `predict.py:31`
calls `predict_fold` and inherits the change for free; `compare.py` reads the OOF parquet and never
knows which library wrote it.

### 1.4 One bookkeeping consequence

`[S-repo]` `TrainConfig.hash()` (`train.py:132-133`) hashes the whole `__dict__`. Adding any field
changes **every** experiment id, so post-change `main.yaml` records get a new `exp_id` even though
the model is identical. Re-run `main.yaml` once after the plumbing lands and check the RMSE is
unchanged to four decimals before trusting any comparison; that run doubles as the regression test
for §5 Stage 0.

---

## 2. Making the comparison fair

### 2.1 What is held identical, and where it is enforced

| held identical | enforced at | note |
|---|---|---|
| folds | `cv.py:26-51`, selected by `cfg.folds` (`train.py:481-485`) | both arms name the same `folds:` list in YAML |
| labelled-row filter | `train.py:365` | drops null/non-finite labels before anything else |
| label-derived encodings | `train.py:367-368` (`fit_encodings` → `attach_encodings`) | fitted on the training side of the fold, joined onto both sides |
| numeric feature list | `train.py:223-235` (`visible_numeric_columns`) | the *one* rule; `include_candidates` / `exclude_features` must match between arms |
| categorical list | `train.py:373-377` from `features.CATEGORICALS` (`features.py:33-44`) | same 10 columns to both libraries |
| matched / unmatched routing | `train.py:379` (`PROXY` not null) and `:398` (null) | 2,062,577 matched / 22,470 unmatched labelled rows `[S-repo]`, 1.08 % unmatched |
| model A's target | `train.py:385`: `y − proxy` | the reparameterisation stays; it is our largest measured win |
| model B's feature blocklist | `train.py:400-406` | the clock columns model B must not see |
| model B's target | `train.py:265-267` + `sched_reparam_shift` | same shift, added back at `:412-413` |
| seeds | `cfg.seed + k`, `k < cfg.n_seeds` (`train.py:351-352`) | same `n_seeds`, same base seed |
| early-stopping protocol | `_inner_holdout_mask` + round scaling (`train.py:314, 345`) | §1.3 items 1–2 |
| metric | `cv.metrics` (`cv.py:58-84`) | `rmse_full` is the scored one; `rmse_clean` reported alongside |
| adopt/reject rule | `compare.verdict` (`compare.py:112-140`) | full-data CI **and** the K=200-dropped CI must both exclude zero |
| threads | `num_threads: 4` / `thread_count: 4` | the box has 4 cores `[S-repo]` `nproc` = 4; a thread mismatch is a wall-clock *and* a determinism difference |

`[X]` The one thing that cannot be held identical is the hyperparameter budget, and that is the
honest hazard in the whole theme. Our LightGBM has been tuned against this leaderboard over ~10
submissions, including an explicit anti-capacity finding recorded in `configs/main.yaml:1-9`
`[S-repo]` (CV rewarded the 255-leaf model B by 34.8 s; the leaderboard charged 13.2 s for it).
CatBoost would arrive with library defaults plus whatever §5 sweeps. Cawley & Talbot's warning
applies in both directions `[S]`: the degradation from optimising a selection criterion on finite
validation data is "often of comparable magnitude to differences in performance between learning
algorithms" (`results/20-tabular-comp-craft.md:440-446`). §6 states what to conclude from a narrow
CatBoost loss, which is *not* "CatBoost is worse".

### 2.2 Categoricals to LightGBM — is our current path already fair?

`[S-repo]` Today LightGBM gets integer codes plus `categorical_feature=cat_idx`
(`train.py:280, 323, 334`). That **is** the native-categorical path, the same arm Sigrist calls
"LGBM Cat", not the "LGBM Num" arm that his benchmark shows is catastrophic at 111 % relative
difference `[S]` (`results/20-tabular-comp-craft.md:172-188`). The nulls are mapped to a real level
`"__NA__"` before coding (`train.py:280`), so they are a category, not missing. Codes start at 0, so
LightGBM's "negative values treated as missing" rule `[S]` never fires.

`[S]` Verified from LightGBM 4.7.0's own docs (`docs/Parameters.rst`, fetched at the pinned tag —
`requirements-lock.txt:44` pins `lightgbm==4.7.0`):

| parameter | default | what it does |
|---|---:|---|
| `max_cat_threshold` | **32** | "limit number of split points considered for categorical features" |
| `min_data_per_group` | **100** | "minimal number of data per categorical group" |
| `cat_smooth` | **10.0** | "reduce the effect of noises... especially for categories with few data" |
| `cat_l2` | **10.0** | L2 regularisation in the categorical split |
| `max_cat_to_onehot` | **4** | below this many categories, one-vs-other splits are used |

`[S]` And from `docs/Advanced-Topics.rst` at the same tag: *"For a categorical feature with high
cardinality (`#category` is large), it often works best to treat the feature as numeric."*

`[S-repo]` Our cardinalities, measured on `data/work/feat_train.parquet` (2,085,047 rows) and
`feat_rank.parquet` (344,841 rows):

| column | train levels | rank levels | train nulls |
|---|---:|---:|---:|
| `STAND_mvt` | **1,900** | 1,698 | 28 |
| `ADES_mvt` | **1,569** | 1,106 | 122 |
| `AIRCRAFT_OPERATOR_flt` | **677** | 580 | 22,470 |
| `AIRCRAFT_TYPE_mvt` | 270 | 200 | 1,507 |
| `ADES_REGION` | 232 | 210 | 122 |
| `RUNWAY_mvt` | 53 | 50 | 0 |
| `ADEP_mvt` | 10 | 10 | 0 |
| `MARKET_SEGMENT_flt` | 9 | 9 | 22,470 |
| `FLIGHT_TYPE_flt` | 6 | 6 | 22,470 |
| `WK_TBL_CAT_flt` | 5 | 5 | 22,544 |

`[X]` With `max_cat_threshold = 32` and `STAND_mvt` at 1,900 levels, LightGBM considers at most 32
of the 1,899 possible split points for the feature that carries the most stand-level information —
this is the concrete form of the CatBoost paper's complaint that LightGBM "groups tail categories
into one cluster and thus looses part of the information" `[S]`
(`results/20-tabular-comp-craft.md:125-128`). Sigrist's LGBM Cat arm was **grid-searched**, ours is
at library defaults. So a CatBoost-at-defaults vs LightGBM-at-defaults comparison quietly handicaps
LightGBM on exactly the axis the theme is about.

**Decision:** the LightGBM arm gets one extra variant — `max_cat_threshold: 64`, `cat_smooth: 20`,
`min_data_per_group: 50` — run at the same cost as one CatBoost fit. `[X]` If that variant alone
closes most of a CatBoost lead, the finding is "our LightGBM's categorical knobs were untouched",
which is a far cheaper fix than a second model family and belongs in the record either way.
Note the existing `abl_no_categoricals` ablation (`configs/abl_no_categoricals.yaml`) already gives
the zero-categoricals floor, so the three-point curve (none / default / widened) is nearly free.

### 2.3 Categoricals to CatBoost

**Pass the same 10 columns raw**, as `str` or `pandas.Categorical`, via `cat_features`. Do not
pre-encode; that is the whole mechanism under test. Verified on catboost 1.2.10 `[S-repo]`
(`requirements-lock.txt:18`):

* `pandas.Categorical` cat columns + `NaN` in numeric columns: **works**, `nan_mode` resolves to
  `Min`. So our null-bearing numeric features need no imputation and stay comparable with
  LightGBM's own missing handling.
* `int32` cat columns: **works**.
* `float32` cat columns: **rejected** — the error text is in §1.1 row 4. This is why `_matrix`
  cannot be reused as-is.
* Nulls in cat columns must be mapped to a literal level first. `train.py:280` already does exactly
  that with `"__NA__"`; keep the same string so the two arms see the same level set.

**CatBoost's applied defaults, verified on this machine** `[S-repo]` (fit a small model, read
`model.get_all_params()`):

| parameter | resolved default | source |
|---|---|---|
| `has_time` | **False** | also stated `[S-repo]` in the installed docstring, `catboost/core.py:4997`: "To use the order in which objects are represented in the input data (do not perform a random permutation of the dataset at the preprocessing stage)" |
| `max_ctr_complexity` | **4** at ≥200 iterations; **1** at ≤150 | docstring `catboost/core.py:4994` says `[default=4]`; the iteration-dependent downgrade is measured, see below |
| `one_hot_max_size` | **2** | docstring `catboost/core.py:5028` says `[default=None]`; the resolved value is 2 |
| `boosting_type` | **Plain** | resolved automatically, at every size I probed |
| `border_count` | 254 | `catboost/core.py:1261` |
| `bootstrap_type` / `subsample` | `MVS` / 0.8 | — |
| `depth` / `grow_policy` | 6 / `SymmetricTree` | — |
| `simple_ctr`, `combinations_ctr` | `Borders:CtrBorderCount=15:TargetBorderCount=1:TargetBorderType=MinEntropy:Prior=0/1:Prior=0.5/1:Prior=1/1` **and** `Counter:CtrBorderCount=15` | — |

**This resolves slot 23's two unverified items** (`results/23-repo-col3name-satam2.md:289-292`,
marked `[X — I did not verify these two defaults from a primary source]`): `one_hot_max_size` is
indeed **2** and `has_time` is indeed **False**, on the exact version we have pinned. Both repos
slot 23 read therefore do run random CTR permutations over a year-long series, as it suspected.

**Two further things the slot did not have, both verified here:**

1. `max_ctr_complexity` is **not** unconditionally 4. At `iterations ≤ 150` it resolves to 1; at
   `iterations ≥ 200` it resolves to 4 (probed at n = 300 / 20,000 / 200,000 / 1,000,000). `[X]`
   Practical consequence: a cheap short-iteration probe silently runs a *different* categorical
   algorithm from the production-length run. Any screening run in §5 must set
   `max_ctr_complexity` explicitly, or the screen does not measure the thing it will ship.
2. For **regression**, the default target-based CTR uses `TargetBorderCount=1` — CatBoost binarises
   the target at one MinEntropy-chosen threshold and the CTR is the smoothed fraction of rows above
   it, not the mean of `y`. `[X]` That matters for §3: CatBoost's out-of-the-box CTR is a *coarser*
   statistic than our `ref_med` / `ref_p10`, so "CatBoost's ordered TS makes our encodings
   redundant" is not obviously true, and `ctr_target_border_count` is a knob worth one variant.

**`max_ctr_complexity` is the first knob to sweep, not the last.** At 4, CatBoost builds CTRs over
combinations of up to 4 of our 10 categoricals — `C(10,1..4) = 10 + 45 + 120 + 210 = 385` candidate
combination features `[X]` — over level counts of 1,900 × 1,569 × 677. It costs a measured **3.1×**
in wall clock (§4.2) and is exactly the "large, unregularised CTR surface" slot 23 flagged as
unevaluated `[S]` (`results/23-repo-col3name-satam2.md:294-298`).

**One structural variant worth carrying.** CatBoost differs from LightGBM in *two* ways at once:
ordered target statistics **and** oblivious (symmetric) trees. `[S-repo]` verified that
`grow_policy="Lossguide", num_leaves=127, min_data_in_leaf=50` is accepted and resolves to
`max_leaves=127`, i.e. LightGBM's growth strategy inside CatBoost. Running that variant separates
the two mechanisms: if Lossguide-CatBoost ≈ LightGBM and SymmetricTree-CatBoost differs, the effect
is tree shape, not target statistics. Caveat `[S-repo]`: with Lossguide the resolved `depth` still
reported 6, so `depth` must be raised explicitly (e.g. 16) or it caps the leaf-wise growth.

### 2.4 `has_time` — the year-long-series question

**The facts.** `[S-repo]` Training data spans **2025-01-01 00:05:56 → 2025-12-31 23:57:36 UTC**,
2,085,047 rows. The ranking set spans **2026-01-01 → 2026-07-31**, 344,841 rows. So the scored set is
strictly later than every training row, and the training window is a full calendar year with strong
seasonality — which is why `cv.py:26-51` holds out January and July separately and builds a
`forward` fold from months ≥ November.

**What `has_time` does.** `[S-repo]` catboost docstring `core.py:4997`: `has_time=True` means "use
the order in which objects are represented in the input data (do not perform a random permutation of
the dataset at the preprocessing stage)". Default **False**. So by default, the permutation that
defines "rows earlier than me" for the ordered target statistic is **random**, and a January
training row's CTR can be computed from December training rows.

**Is that leakage?** `[X]` No, not in the fold sense — the fold split already decides which rows are
in `train`, and CatBoost only permutes within that set. It is the *same shape* as the leak
`cv.py:39-45` already documents and deliberately measures: the `jan` fold trains on July, November
and December, and `jul_fwd` exists precisely "so that the cost of the leak be measured rather than
assumed". So `has_time=False` does not add a new hazard on the `jan`/`jul` folds; on the `forward`
fold it does add one, because `forward` is the fold built to contain no future.

**The argument for `has_time=True`:** it is the setting whose training-time feature distribution most
resembles serve time. At serve time every 2026 row's CTR is computed from the *whole* 2025 training
set, i.e. entirely from its own past. With `has_time=True` and rows sorted by
`MVT_TIME_UTC_mvt`, training rows get the same treatment. That is the same prediction-shift argument
the CatBoost paper makes for ordered TS over Greedy TS `[S]`
(`results/20-tabular-comp-craft.md:113-124`), applied one level up.

**The argument against:** `[X]` (a) with a single fixed permutation CatBoost loses the variance
reduction it gets from averaging several random permutations; (b) rows early in January get
statistics from almost no preceding rows, so their encodings are near-prior and high-variance, and
January is one of the two scored seasons; (c) the ordered-*boosting* half of the permutation trick
is moot at our size anyway — `boosting_type` resolves to **Plain** at every size probed `[S-repo]`,
and the CatBoost paper's own Table 3 caveat is that Ordered mode's benefit shows "on Adult and
Internet datasets, which are relatively small (less than 40K training examples)" `[S]`
(`results/20-tabular-comp-craft.md:155-159`). At 2.06M rows, none of that is in play.

**Decision:** treat it as a measured variant, not a default. Run `has_time=False` (library default,
and what both 2026 competitor repos ship `[S]`) as the primary, and `has_time=True` with the Pool
rows sorted ascending by `MVT_TIME_UTC_mvt` as a named variant in Stage 3. Report both.
Implementation note `[X]`: `_fit` stacks train + predict frames and slices by offset
(`train.py:315-322`); a `has_time=True` arm must sort the *training block only*, keep the predict
blocks' order untouched (their row order must still match `MVT_ID_mvt` for `_assemble`), and the
target vector must be permuted with it.

**One thing I could not verify and am not asserting.** I did not confirm from a primary source how
CatBoost computes CTR values for *apply-time* rows. The standard account is that they come from
statistics over the whole training set, which would mean `has_time` affects training-row features
only and never the 2026 predictions. Treat that as `[X]` until checked; if it is wrong, the
`has_time` variant's interpretation changes and nothing else in this plan does.

---

## 3. The P2 interaction, head on

### 3.1 The defect, and why it is the same defect

`[S-repo]` `train.py:367-368`:

```python
enc = fit_encodings(train_f, cfg)
tr, tg = attach_encodings(train_f, enc), attach_encodings(target_f, enc)
```

`fit_encodings` builds four label-derived tables (`train.py:200-206`) and `attach_encodings` joins
them back onto **the very rows they were fitted from**. `[S]` That is textbook **Greedy TS** — the
construction the CatBoost paper names "a special kind of target leakage present in all currently
existing implementations of gradient boosting" and proves induces a prediction shift
(`results/20-tabular-comp-craft.md:113-120`, 588-592). `[S]` scikit-learn's `TargetEncoder` docs call
`fit().transform()` on training data "discouraged… because it can introduce data leakage"
(`results/20-tabular-comp-craft.md:576-584`). This is P2's subject
(`plans/P2-cross-fit-encodings.md`).

**Ordered target statistics are the mechanism that was designed to replace exactly this.** So the
naive expectation is: our encodings are redundant with CatBoost's CTRs at best, and actively
harmful at worst, because a Greedy-TS column tells a CatBoost tree something about the training row's
own label that it will not hear in 2026.

### 3.2 Why the redundancy is only partial — four specifics

`[X]`, each grounded in a `[S]` or `[S-repo]` fact:

1. **Different keys.** Our tables key on `ADEP × STAND × RUNWAY`, `ADEP × RUNWAY × hour_local`,
   `ADEP`, and `AIRCRAFT_OPERATOR × ADEP` (`train.py:162, 178, 189, 197`). CatBoost's CTRs key on
   single categoricals and on combinations **of declared categoricals**. `hour_local` is `Int8`
   `[S-repo]`, i.e. numeric, so `arh_med`'s key is **not reachable** by any CatBoost CTR
   combination unless we additionally declare an hour categorical.
2. **Different statistic.** Ours are order statistics — `p10`, `p20`, `median` (`train.py:165-167`).
   CatBoost's default regression CTR is a smoothed fraction above a **single** MinEntropy target
   border (`TargetBorderCount=1`, verified §2.3). A p10 of taxi time is not recoverable from that.
3. **Different smoothing.** Ours is a hard `min_count` cutoff — 30 for `reference_table`
   (`configs/main.yaml:16`), 10 for `airport_runway_hour_table`, 20 for `operator_offset_table` —
   producing a null that `ref_missing` flags (`train.py:218`). CatBoost's is a Bayesian prior
   (`Prior=0/1:Prior=0.5/1:Prior=1/1`) with no missing state.
4. **`op_offset_med` encodes the residual, not the label.** `operator_offset_table` takes the median
   of `LABEL − PROXY` (`train.py:188`), i.e. the clock-offset our reparameterisation is built around.
   No CTR over a raw target can express that. `[S-repo]` It is currently hidden anyway —
   `use_offset_encoding` defaults `False` (`train.py:130`) and no config sets it.

### 3.3 The size of the thing being argued about — measured on our own data

`[S-repo]` From `experiments/experiments.jsonl`, three records at `n_seeds: 1`, folds
`[jan, jul, forward]`, pooled OOF `n = 672,428`:

| record | `use_encodings` | `use_categoricals` | `rmse_full` | vs baseline |
|---|---|---|---:|---:|
| `ab_split-ca87cb5af7` (baseline, named in both ablation configs) | on | on | **301.63** | — |
| `abl_no_categoricals-7c6f287f34` | on | **off** | 309.00 | **+7.37** |
| `abl_no_encodings-8cf7918aa5` | **off** | on | 338.99 | **+37.36** |

**This is the single most important calibration in the plan** `[X]`. Our nine hand-rolled encoding
columns are worth **five times** what the ten raw categorical columns are worth. Two consequences:

* **A ceiling on the pure categorical-handling story.** Dropping the categoricals entirely costs
  7.37 s. If CatBoost's advantage were *only* that ordered TS beats Fisher-split-at-32-thresholds,
  the prize is of that order — single digits — not the 11.6 % (≈ 33 s at our scale `[X]`) PRC 2024
  reported. `[X]` This is an argument about magnitude, not an impossibility proof: CatBoost could
  also win on tree shape, on regularisation, or on the encodings the −E arm removes. But it should
  temper the 33 s figure before anyone spends four CPU-hours on it.
* **The encodings are where the categorical information actually lives in our model.** Which makes
  the encoding scheme, not the library, the first-order variable — and makes the 2×2 mandatory.

`[S-repo]` Run-to-run noise at fixed config is negligible: `det_a-cfc9417c52` 336.21 vs
`det_b-fa2cba5913` 336.23 (**0.02 s**), and `inv_baseline_forward` at two different hashes both
284.70 exactly. `deterministic: True` + `force_row_wise: True` (`train.py:75-76`) is doing its job.
So differences above ~0.1 s on the same folds are real differences in the model, not in the run —
which is what makes single-seed screening honest here.

### 3.4 The 2×2, and the cell that P2 cannot disturb

| | `use_encodings: true` | `use_encodings: false` |
|---|---|---|
| **LightGBM** | **L+E** — current production; `configs/main.yaml` | **L−E** — `configs/abl_no_encodings.yaml` exists `[S-repo]` |
| **CatBoost** | **C+E** | **C−E** |

The three contrasts it separates `[X]`:

* **family main effect** = mean(C) − mean(L)
* **encoding main effect** = mean(+E) − mean(−E) — already known to be ≈ 37 s for LightGBM (§3.3)
* **interaction** = (C+E − C−E) − (L+E − L−E) — *the answer to the question this theme asks*

Read it like this:

* **Interaction ≈ 0 and C < L in both cells** → a genuine family effect. Proceed.
* **C−E ≪ L−E but C+E ≈ L+E** → CatBoost's CTRs are substituting for our encodings and adding
  nothing on top. `[X]` Then the right action is *not* to adopt CatBoost; it is to fix the
  encodings, which is P2, at a fraction of the compute.
* **C+E > C−E by much less than L+E > L−E** → our Greedy-TS columns are hurting CatBoost
  specifically, which is the prediction-shift prediction. That is direct evidence *for* P2 and
  costs nothing extra to obtain.
* **C > L in both cells** → stop (§6).

**The ordering question the brief asks — if P2 lands first, does the expected CatBoost advantage
shrink?**

`[X]` Yes, mechanically, in the `+E` row and only there. P2 changes how the four tables are built
for *training* rows; the `−E` row hides those columns entirely (`train.py:231-232`), so **C−E and
L−E are invariant to P2 by construction**. That gives a clean scheduling answer:

> **Run the `−E` contrast first.** It is the pure family comparison, it cannot be invalidated by
> P2 landing, and it is the cheaper pair (fewer feature columns). Run the `+E` contrast against
> whatever encoding scheme is current at the time and re-run only that row if P2 changes it.

And the honest expectation: `[X]` if P2 succeeds, L+E improves and the CatBoost advantage in the
`+E` row shrinks by roughly that amount. P2's own plan pre-registers that the sourced evidence for
that exact change is a **loss** (`plans/P2-cross-fit-encodings.md:21-28`), so the shrink may be zero
or negative. Either way, the two plans should not be run concurrently against the same `main.yaml`
baseline without recording which encoding scheme each CatBoost number was measured against — that
is precisely the "OOF produced under different protocols" failure slot 11 §5.2 names
(`results/11-ensembling-patterns.md:430-436`).

---

## 4. Compute: wall clock, memory, GPU

### 4.1 GPU

`[S-repo]` **No GPU.** `nvidia-smi` is not installed and `/dev/nvidia*` does not exist. 4 CPU cores
(`nproc`), 15 GB RAM total / ~14 GB available (`free -g`). Everything below is CPU-only.

`[X]` Does it matter? It would remove the main objection to this theme, which is cost, not
correctness — CatBoost's GPU trainer is where its documented speedups live. It also changes the
defaults (CatBoost's docs give a different `max_ctr_complexity` default for GPU
`[S]`, `results/23-repo-col3name-satam2.md:287-288`), so a GPU result would not be a drop-in
substitute for a CPU result. Since no GPU is available, this is moot; do not plan around one.

### 4.2 Measured throughput

Method in §4.4. All at 4 threads, 62 numeric float32 features + the 10 categoricals at our real
cardinalities, `learning_rate=0.05`, `loss_function=RMSE`, **n = 1,000,000 rows**:

| fitter | config | ms / iteration | peak RSS |
|---|---|---:|---:|
| LightGBM 4.7.0 | 127 leaves, `min_data_in_leaf=50`, our `main.yaml` params | **137** | 1.10 GB |
| CatBoost 1.2.10 | depth 6, `max_ctr_complexity=1` | **191** | 1.51 GB |
| CatBoost 1.2.10 | depth 6, `max_ctr_complexity=4` (**library default**) | **593** | 2.74 GB |
| CatBoost 1.2.10 | depth 8, `max_ctr_complexity=4` | **694** | 2.66 GB |

Three readings `[X]`:

1. **`max_ctr_complexity` is the dominant cost knob**: 4 vs 1 is **3.1×** at the same depth. Depth 6
   → 8 is only 1.17×.
2. **At its default, CatBoost is ~4.3× LightGBM per iteration** and needs ~2.5× the memory.
3. At `max_ctr_complexity=1` it is only 1.4× LightGBM, which makes the cheap screen genuinely cheap.

### 4.3 Extrapolation to our folds, with the correction factor

`[S-repo]` Real LightGBM timings from `experiments/experiments.jsonl`: the `forward` fold at
`n_seeds: 3` (= 1 early-stopping fit + 3 refits ≈ 4 × 939 rounds on 1,757,038 training rows) takes
**1,108.6 s** (`inv_baseline_forward-9271caca22`) — call it 0.295 s/round at 1.76M, i.e. **168 ms/round
at 1M-equivalent**. My synthetic LightGBM measured 137 ms. So the synthetic benchmark runs **~1.2×
optimistic**; apply that factor to the CatBoost numbers too.

`[X]` Model A's training block is ~1.74M rows on the `forward` fold (2,062,577 matched rows overall,
1.08 % unmatched `[S-repo]`). Model B's 22k rows are negligible in every arm.

Corrected per-iteration cost on 1.74M rows, and cost of one `_fit` call
(= 1 early-stopping fit + `n_seeds` refits):

| CatBoost config | s / iter | 1 fit @ 2,000 iters | `_fit`, `n_seeds=1` (2 fits) | `_fit`, `n_seeds=3` (4 fits) |
|---|---:|---:|---:|---:|
| depth 6, `ctr=1` | ~0.40 | ~13 min | **~27 min** | ~54 min |
| depth 6, `ctr=4` | ~1.24 | ~41 min | **~83 min** | ~2 h 45 m |
| depth 8, `ctr=4` | ~1.45 | ~48 min | ~97 min | ~3 h 13 m |

Against LightGBM's measured **~19 min** for the same fold at `n_seeds=3` `[S-repo]`.

Three folds `{jan, jul, forward}`, CatBoost at `ctr=4`, `n_seeds=1`: **~4 h**. At `n_seeds=3`:
**~8 h**. `[X]` That is the real price tag of this theme, and it is the reason §5 puts a single fold
and `ctr=1` first.

**Iteration count is the biggest uncertainty in the table** `[X]`. I assumed 2,000. Our LightGBM
early-stops at 939–1,297 rounds on 127-leaf trees (`experiments.jsonl`, all recent records).
CatBoost's default depth-6 symmetric tree has 64 leaves and is *less* expressive per tree, so at the
same learning rate it plausibly wants **more** iterations, not fewer. If it wants 4,000, double the
table. Mitigation: set `learning_rate` to match round-for-round rather than copying 0.05 blindly,
and let `od_wait` decide — but **record the early-stopped iteration count in the experiment record**,
because it is the number that makes or breaks the cost estimate for every later stage.

### 4.4 Memory, and the one place it could bite

`[X]` Peak RSS scales close to linearly in rows: 2.74 GB at 1M with `ctr=4` → **~4.8 GB at 1.74M**.
Add the `feat_train` polars frame, measured at **1.44 GB** for 2,085,047 × 132 `[S-repo]`
(`df.estimated_size()`), plus `feat_rank`. That is ~7 GB against 14 GB available — workable, but not
comfortable, and `ctr=4` at depth 8 with two frames live is the worst case.

The specific hazard `[X]`: `_fit` currently materialises **one** stacked float32 matrix covering train
*and* every predict frame (`train.py:316-317`), then fancy-indexes it twice for the early-stopping
split (`train.py:333-335`) — the docstring at `:346` already calls those copies "the peak of this
function". A CatBoost path that builds a pandas frame with 10 extra `category` columns alongside,
and then a `Pool` (which quantises into its own copy), could hold three representations at once.
Build the Pool from the frame and `del` the frame before fitting; prefer `int32` codes over strings
for the cat columns (both accepted `[S-repo]`) since codes are 4 bytes against a Python string
object's ~50+.

---

## 5. Staging, cheapest decisive test first, with an abort at each stage

Stage costs use §4.3. Every stage writes a normal `experiments.jsonl` record and an OOF parquet, so
`python -m prc.compare A B` is the read-out at every stage.

### Stage 0 — plumbing. No competition-data training. ~0 compute.

Add `model: "lightgbm" | "catboost"` to `TrainConfig`, a family-scoped `cb_params`, the strict
unknown-key guard from §1.2, and a `_fit_catboost` honouring the six contract items in §1.3.

**Verify, all four required:**
1. `main.yaml` re-run reproduces `rmse_full` to 4 decimals against its existing record (the
   `cfg.hash()` change from §1.4 means a new `exp_id`; compare on the number, not the id).
2. A config with a misspelled key **raises**.
3. A config with `model: catboost` and `use_linear_tree: true` **raises** rather than silently
   ignoring it.
4. On the synthetic smoke path (`--synthetic`, `src/prc/synth.py`), both arms produce a prediction
   frame keyed by `MVT_ID_mvt` of the right height, and the `feature_cols + cat_cols` lists are
   **byte-identical** between arms. Assert this, do not eyeball it: an unequal feature list makes
   every later number meaningless.

**Abort if:** (4) fails and cannot be made to hold. A family comparison on different features is not
a family comparison.

### Stage 1 — subsample screen: does it run, and roughly where does it land? ~15 min.

`forward` fold, deterministic 20 % subsample of the matched training rows (same row set for both
arms — sample by hashing `MVT_ID_mvt`, not by `head`), `n_seeds: 1`, CatBoost at depth 6 /
`max_ctr_complexity=1` **set explicitly** (§2.3 item 1), LightGBM at `main.yaml` params.
`[X]` ~350k rows → ~5 min per CatBoost `_fit`, ~2 min for LightGBM.

**This stage is plumbing validation and a smell test, not a decision.** `[X]` Say so in the record:
a subsample *flatters* CatBoost, because the CatBoost paper's own Table 3 puts the ordered
machinery's benefit on small datasets `[S]` (`results/20-tabular-comp-craft.md:155-159`). A CatBoost
win here is not evidence; a CatBoost catastrophe here is.

**Abort if:** CatBoost is worse by more than ~30 % relative on this subsample, or the run errors, or
memory exceeds ~6 GB. A 30 % gap at 350k rows is an implementation bug or a configuration error, not
a model difference — find it before spending an hour.

### Stage 2 — the decisive cheap test: `forward` fold, full data, `−E` cells. ~1 h.

The pure family contrast from §3.4, on the fold that is shaped like the real task (`cv.py:46-48`,
and `train.py:118-121`: "forward is also the only fold shaped like the real task"), at `n_seeds: 1`:

| run | config | est. |
|---|---|---:|
| **L−E** | `abl_no_encodings.yaml` + `folds: [forward]`, `n_seeds: 1` | ~7 min |
| **C−E (a)** | CatBoost, depth 6, `max_ctr_complexity=1` | ~27 min |
| **C−E (b)** | CatBoost, depth 6, `max_ctr_complexity=4` (library default) | ~83 min |

**Read-out:** `python -m prc.compare <L−E> <C−E>`, i.e. the full-data difference, the
K∈{20,200,2000}-dropped curve, and the month-bootstrap — the same rule that governs every other
change in this repo (`compare.py:112-140`). Also record the early-stopped iteration count (§4.3) and
peak RSS.

**Abort if:** `compare.verdict` returns "B is worse on all rows" for **both** CatBoost variants.
That is slot 20's own stated rule `[S]`: *"if it loses, stop — do not blend a loser hoping for
diversity"* (`results/20-tabular-comp-craft.md:844-845`). Record the result; do not run Stage 3.

**Also abort — for a different reason — if:** `C−E (b)` is not better than `C−E (a)` by more than
noise. Then `max_ctr_complexity=4` is 3.1× the compute for nothing, and every later stage runs at
1 or 2. This is the measurement neither 2026 competitor repo made `[S]`
(`results/23-repo-col3name-satam2.md:294-298`) and it is worth having on its own.

### Stage 3 — complete the 2×2 and the named variants, still on `forward`. ~3 h.

Only if Stage 2 did not abort.

| run | why |
|---|---|
| **C+E** at the surviving `ctr` setting | completes the 2×2; tests whether our Greedy-TS columns help or hurt CatBoost (§3.4) |
| **L+E** at `folds: [forward]`, `n_seeds: 1` | the fourth cell at matched settings (the existing `inv_baseline_forward` record is `n_seeds: 3` + `num_threads: 2`, so it is a reference, not a matched arm) |
| **L+E, widened categorical knobs** — `max_cat_threshold: 64`, `cat_smooth: 20`, `min_data_per_group: 50` | §2.2: removes the handicap objection |
| **C, `has_time=True`**, Pool sorted by `MVT_TIME_UTC_mvt` | §2.4 |
| **C, `grow_policy=Lossguide`, `num_leaves=127`, `min_data_in_leaf=50`, `depth=16`** | §2.3: separates oblivious trees from ordered TS |
| **C, `one_hot_max_size=12`** | one-hots `ADEP` (10), `MARKET_SEGMENT` (9), `FLIGHT_TYPE` (6), `WK_TBL_CAT` (5) instead of CTR-ing them |

Report the interaction term from §3.4 explicitly, and per-routing-branch (matched vs unmatched) and
per-airport RMSE — `run_cv` already records `by_airport` (`train.py:499`), and slot 20's
measurement spec asks for exactly that split (`results/20-tabular-comp-craft.md:842-843`).

**Abort if:** the interaction says CatBoost's edge exists only in the `−E` cell (§3.4 case 2). Then
the finding is "fix the encodings" and the budget belongs to P2, not here. Write that up and stop —
it is a *result*, and a cheaper one than the alternative.

### Stage 4 — three folds, seeds matched. ~8 h.

Only the single best CatBoost variant, at `folds: [jan, jul, forward]` and `n_seeds: 3`, against
`main.yaml` at the same folds and seeds.

**Adopt criterion** (slot 20's, `results/20-tabular-comp-craft.md:842-845` `[S]`, tightened to this
repo's own rule): CatBoost wins under `compare.verdict` — full-data CI **and** K=200-dropped CI both
excluding zero — **and** wins on the `jan` and `jul` folds separately, not only pooled. `[X]` The
per-fold requirement matters because `overall` pools three different training protocols
(`train.py:474-477`, and slot 11 §5.3) and because the scored set is January-and-July-shaped: the
ranking set is 2026-01 → 2026-07 `[S-repo]`.

**Abort if:** the win is pooled-only, or reverses on either seasonal fold. A pooled-only win is the
shape `compare.py`'s docstring was written about (`compare.py:5-12`).

### Stage 5 — one leaderboard probe, then and only then blending.

`[X]` Our CV level (300.17 pooled on three folds `[S-repo]`, `main-f9d57a661f`) and our leaderboard
level (284.36) do not move together — `configs/main.yaml:1-9` records a case where CV said 34.8 s
and the leaderboard said 13.2 s the other way `[S-repo]`. So a CV win must be confirmed by one
submission before anything is built on top of it. Use the probe method already established in this
repo (`configs/submit_v10_reparam.yaml:6-14` `[S-repo]`): swap only the rows the change touches, so
the leaderboard delta is attributable.

Blending enters here, after a confirmed single-model result, and follows slot 11's Steps 0–6
(`results/11-ensembling-patterns.md:546-623`) — which begin with P3's fold-labelled OOF, not with
weight fitting.

---

## 6. Expected outcome, honestly, and the result that makes us stop

### 6.1 Evidence that CatBoost wins

| evidence | number | source |
|---|---|---|
| PRC 2024 ATOW, same team, same features, Optuna-tuned both | CatBoost **2,215.53** vs LightGBM **2,506.43** = **−11.6 %** | `[S]` results/20:742-752 |
| PRC 2024 ablation of the fitted blend | removing CatBoost **+33.90 %**; removing LightGBM **+0.00 %** | `[S]` results/11:65-79 |
| PRC 2024 fitted blend weights | CatBoost **0.865**, XGBoost 0.134, LightGBM/MLP/SAINT ~1e-23 | `[S]` results/11:59-63 |
| PRC 2025 rank 2, three-library blend | CatBoost best single member (244.9), largest weight (64.1 %) | `[S]` results/11:128-138 |
| TabArena v0.1 Elo, 51 datasets | CatBoost **1432** default / 1475 tuned vs LightGBM **1206** / 1450 | `[S]` results/20:262-281 |
| TabRepo, 200 OpenML datasets, normalised error | CatBoost tuned **0.534** vs LightGBM tuned **0.566** | `[S]` results/20:212-231 |
| NASA pushback Phase 1, top 2 | both CatBoost-per-airport; LightGBM 4th | `[S]` results/11:185-196 |
| PRC **2026** public competitors | both repos slot 23 read are **CatBoost-only** | `[S]` results/23:517, 228-240 |

### 6.2 Evidence that it does not

| evidence | number | source |
|---|---|---|
| PRC 2025 **rank 1** | a single LightGBM, 24 features, **200.83** — beat the fully-fitted three-library blend at rank 2 (211.99) by **11.16 RMSE** | `[S]` results/11:143-160 |
| Sigrist, 8 **regression** datasets, 553–72,000 levels | LGBM-native-categorical avg rank **2.71** / rel. diff. **17.4 %**; CatBoost **2.86** / **44.5 %** | `[S]` <https://arxiv.org/pdf/2307.02071>, results/20:166-186 |
| PRC 2024 **winner** | team `likable_jelly`, a single LightGBM, RMSE **1,561** — vs the CatBoost-blend team's 2,218 | `[S]` results/20:754-761 |
| ML Contests 2025 | a 2025 winner for whom **CatBoost underperformed** | `[S]` results/20:44, 832-836 |
| CatBoost's own size caveat | ordered *boosting*'s benefit is documented "less than 40K training examples"; at 2.06M `boosting_type` resolves to **Plain** anyway | `[S]` results/20:155-159; `[S-repo]` verified §2.3 |
| Our own ablation | the whole categorical block is worth **7.37 s** to our LightGBM; the encodings are worth 37.36 s | `[S-repo]` §3.3 |

`[X]` **Sigrist is the closest match to our data shape and it goes against the theme.** Regression,
RMSE-family metric, high-cardinality grouping variables — our `STAND_mvt` at 1,900 levels and
`ADES_mvt` at 1,569 sit inside his 553–72,000 range. PRC 2024 is the closest match to our *task* and
goes for the theme. Those are the two best sources available and they disagree. There is no honest
way to call the direction in advance.

### 6.3 The pre-registered expectation

`[X]` I will not put a point estimate on "CatBoost beats LightGBM by N seconds", because the two
closest sources point opposite ways. What I will pre-register:

1. **Most likely outcome: a small difference either way, well under the 33 s that a naive 11.6 %
   transfer implies.** The reasons are ours, not the literature's: PRC 2024's gap was partly their
   LightGBM being the *worst* of their five models (slot 11's own reading,
   `results/11-ensembling-patterns.md:81-84`), ours has been tuned against this leaderboard; our
   biggest structural wins — the `y − proxy` reparameterisation, the matched/unmatched routing, the
   LIRF schedule reparameterisation — carry over unchanged to both arms, so both start from a good
   parameterisation; and §3.3 bounds the pure categorical-handling prize at single digits.
2. **The `−E` contrast is where a real family effect would show first**, because it is the cell in
   which the categorical columns are the *only* route from category to prediction.
3. **The most likely `+E` outcome is that CatBoost's CTRs and our encodings are substitutes**, so
   `C+E ≈ L+E` even if `C−E < L−E`. That outcome is a P2 finding wearing a P10 costume, and §3.4
   says to read it that way.
4. **CatBoost at library defaults is not a tuned CatBoost.** A narrow CatBoost loss at Stage 2/3 is
   evidence about *default* CatBoost, not about the family — Cawley & Talbot's point, in reverse
   `[S]`. Say so in the write-up rather than over-claiming a null.

### 6.4 The results that stop us

Each is a stop, not a pause. Write it up and move the budget to another theme.

* **S1.** Stage 2: `compare.verdict` says CatBoost is worse on all rows at **both** `ctr` settings
  on the `forward` fold. Slot 20's rule, verbatim `[S]`: do not blend a loser hoping for diversity.
* **S2.** Stage 3: the 2×2 interaction shows CatBoost's edge lives only in the `−E` cell. The
  mechanism is the encoding scheme, P2 owns it, and P2 is ~50× cheaper in compute.
* **S3.** Stage 4: the win is pooled-only, or reverses on `jan` or `jul`. The scored set is
  January-and-July-shaped `[S-repo]`; a win that is not there is not a win.
* **S4.** Stage 4: CatBoost wins but by less than **~1 s** while costing ~4× the wall clock.
  `[X]` Then the sourced blend payoff on top is the PRC 2024 case, not the PRC 2025 case — one
  member takes ~0.87 of the weight and the blend buys **0.23 % ≈ 0.65 s** `[S]`
  (`results/20-tabular-comp-craft.md:796`). A 0.65 s expected gain does not justify doubling every
  future experiment's cost, and slot 11's own caveat applies: the remaining 6–7 points to the top of
  the public board are "in features, per-airport specialisation and target handling, not in the
  blend" `[S]` (`results/11-ensembling-patterns.md:625-631`).
* **S5.** Stage 5: the CV win does not reproduce on the leaderboard probe.

**The one result that does *not* stop us, and is worth having anyway:** `max_ctr_complexity=4`
turning out to be worthless at 3.1× the cost. That is a fact about the two competitor repos' shipped
configuration `[S]`, obtainable in Stage 2 for free, and it is the kind of thing nobody else in this
competition appears to have measured.

---

## 7. What this plan deliberately does not do

* **No blend.** Nothing here fits a weight. Slot 11 Step 0 (`results/11-ensembling-patterns.md:546-553`)
  requires a fold-labelled OOF that we do not have — that is **P3**, and it is a hard prerequisite
  for any blend, not for this comparison. This plan's comparison works fine without it because
  `compare.py` compares two OOF files fold-for-fold rather than fitting anything.
* **No XGBoost.** Its sourced marginal value is smaller than CatBoost's in every source
  (`results/20-tabular-comp-craft.md:246-247`; PRC 2024 ablation +0.16 %; Repo B's log records
  "XGBoost weight ≈ 0 in the blend" `[S]`, `results/23-repo-col3name-satam2.md:840`).
* **No GPBoost / random-effects model**, even though it is the *winner* of Sigrist's table at 7.02 %
  against LGBM Cat's 17.4 % `[S]` (`results/20-tabular-comp-craft.md:191-195`). It is a family we
  have never tried and it is not in `pyproject.toml`. Worth a line in the gap audit (theme 00); not
  worth conflating with this measurement.
* **No hyperparameter search.** Stage 3's variants are a fixed, pre-registered list of six, chosen
  because each isolates a named mechanism. An Optuna sweep would make the comparison unfair in the
  other direction and would cost multiples of the §4.3 table.

## 8. Risks that would invalidate the measurement

| risk | detection | mitigation |
|---|---|---|
| The two arms silently see different features | Stage 0 check (4): assert the `feature_cols + cat_cols` lists are equal | assert, do not inspect |
| A screening run uses `max_ctr_complexity=1` and the production run uses 4, because of the iteration-count-dependent default (§2.3) | log `model.get_all_params()` into the experiment record | set it explicitly in every config |
| The YAML key for the family is dropped silently (`pipeline.py:129`) and the "CatBoost" run is a LightGBM run | log the resolved family and the resolved params into the record | the strict unknown-key guard, Stage 0 check (2) |
| Iteration count is 2× my assumption and Stage 4 does not fit in the budget | Stage 2 reports the early-stopped count | re-cost Stage 3/4 from Stage 2's actual number before committing |
| Memory pressure with three data representations live (§4.4) | log peak RSS per stage | build the Pool, `del` the frame; `int32` codes not strings |
| `has_time=True` changes apply-time CTRs in a way I have not verified (§2.4) | read CatBoost's apply-time CTR documentation before Stage 3 | if it does, the variant's interpretation changes; flag it, do not silently reinterpret |
| P2 lands mid-theme and the `+E` numbers become incomparable | record the encoding-scheme version in every experiment record | run the `−E` contrast first — it is invariant to P2 |

---

## Appendix — benchmark method for §4

Synthetic frames of our *shape*, not our data: 62 numeric `float32` columns of standard normals plus
10 `pandas.Categorical` columns at cardinalities `[10, 40, 1500, 300, 7, 700, 12, 8, 700, 10]`,
matching the measured train-side level counts in §2.2. Target `= 100·x₀ + 900 + N(0, 200²)`, which
puts it on the taxi-seconds scale. 4 threads throughout, `allow_writing_files=False`.

* LightGBM arm: our exact `main.yaml` params (`train.py:56-78`), categoricals as integer codes with
  `categorical_feature=`, matching `_matrix`'s encoding.
* CatBoost arm: `Pool(df, y, cat_features=<the 10 names>)`, `loss_function="RMSE"`,
  `learning_rate=0.05`, `max_ctr_complexity` set explicitly.
* Timings exclude Pool/Dataset construction (measured separately at ≤0.5 s in every case). Peak RSS
  from `resource.getrusage(RUSAGE_SELF).ru_maxrss`.

**Calibration and its error bar.** Synthetic LightGBM at 1M rows = 137 ms/round; the real pipeline
runs 168 ms/round at 1M-equivalent (`inv_baseline_forward-9271caca22`: 1,108.6 s for ≈4 × 939 rounds
on 1,757,038 rows `[S-repo]`). The synthetic benchmark is therefore **~1.2× optimistic**, and §4.3
applies that factor to CatBoost as well. `[X]` That assumes the two libraries are optimistic by the
same factor on real data, which is an assumption: real features have structure (and real
categoricals have skewed level frequencies) that synthetic noise does not, and CTR construction cost
depends on level frequency in a way split-finding does not. Treat §4.3 as ±30 %, and re-cost from
Stage 2's measured wall clock before committing to Stage 4.
