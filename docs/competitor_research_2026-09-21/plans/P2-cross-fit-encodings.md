# P2 — Cross-fit the target encodings

Plan written 2026-09-21. Baseline: `resilient-hedgehog` 284.3584 (taxi-out seconds, RMSE).
Origin: research slot 20 §8.2 and §"Applicable" item 1
(`docs/competitor_research_2026-09-21/results/20-tabular-comp-craft.md:599-628, 808-824`).

**Marking convention** (same as the research slots)

* `[S]` — read from an external source; URL given inline.
* `[S-repo]` — read from this repository or from a cloned competitor repository; `path:line` given.
* `[X]` — my own inference or arithmetic. Never dressed as `[S]`.

**No code was modified in producing this plan.**

---

## 0. The recommendation up front

The defect is real and I verified it directly. But the plan is **not** "cross-fit the encodings and
ship it", because every sourced measurement of that exact change — by two independent competitors, one
of them on the actual leaderboard — is a **loss**. The plan splits the work into two things that are
usually conflated:

| | what it is | serve-side feature changes? | recommendation |
|---|---|---|---|
| **Fix 1** | the early-stopping holdout stops reading its own labels | **no** | do it; it is a correctness fix with no mechanism to hurt |
| **Fix 2** | the *training rows* stop reading their own labels | **yes** | measure it; pre-register the expectation that it loses |

Fix 1 is the defect merry-quicksand names BC-1 and wrote a repair for
`[S-repo]` (`/tmp/prc-merry-quicksand/prc/encoding.py:1-46`). Fix 2 is the Greedy TS defect the
CatBoost paper proves causes conditional shift `[S]`. They are different experiments with different
risk profiles and must not be run as one change.

---

## 1. The defect, exactly: which rows, at which points

### 1.1 The four label-dependent tables

`[S-repo]` `src/prc/train.py:158-206`. All four are built by grouping **labelled** rows and reducing
the label:

| builder | `train.py` | key | statistic | min count | feeds |
|---|---|---|---|---|---|
| `reference_table` | `:158-172` | `ADEP_mvt × STAND_mvt × RUNWAY_mvt` | `ref_p10` = quantile(y, .10), `ref_p20`, `ref_med` = median(y), `ref_n` | `cfg.reference_min_count` = 30 | `ENCODING_COLS` |
| `airport_runway_hour_table` | `:175-181` | `ADEP_mvt × RUNWAY_mvt × hour_local` | `arh_med` = median(y), `arh_n` | 10 (hard-coded default) | `ENCODING_COLS` |
| `airport_median_table` | `:195-197` | `ADEP_mvt` | `apt_med` = median(y) | none | `ENCODING_COLS` |
| `operator_offset_table` | `:184-192` | `AIRCRAFT_OPERATOR_flt × ADEP_mvt` | `op_offset_med` = median(y − proxy), `op_offset_n` | 20 (hard-coded default) | `OFFSET_COLS` |

`[S-repo]` `ENCODING_COLS` (`train.py:43`) also carries two derived columns built inside
`attach_encodings` (`train.py:215-219`): `ref_p10_fb = coalesce(ref_p10, arh_med, apt_med)` and
`ref_missing = ref_n.is_null()`. Nine columns total, all visible to the model when
`use_encodings: true` (`configs/main.yaml:14`, gate at `train.py:229-235`).

`[S-repo]` **`OFFSET_COLS` is currently hidden from the model.** `use_offset_encoding` defaults to
`False` (`train.py:130`) and `configs/main.yaml` never sets it, so `visible_numeric_columns`
(`train.py:233-234`) removes `op_offset_med` / `op_offset_n` from the feature list. The table is still
fitted on every fold (`train.py:205`) — wasted work, not a live defect. **Three** of the four tables,
not four, currently reach the production model. This matters for the effect-size ceiling in §5.

`[X]` Note the statistic types: **every one of these is an order statistic** (p10, p20, median) except
the count columns. That single fact rules out two of the four repair options in §3 — order statistics
have no additive decomposition, so neither leave-one-out in closed form nor the
subtract-the-month-aggregate trick that makes merry-quicksand's cross-fit cheap
(`/tmp/prc-merry-quicksand/prc/encoding.py:83-101` `[S-repo]`) applies here.

### 1.2 The call site

`[S-repo]` `src/prc/train.py:365-368`:

```python
train_f = train_f.filter(pl.col(LABEL).is_not_null() & pl.col(LABEL).is_finite())
info: dict = {"train_rows": train_f.height, "dropped_unlabelled": n_before - train_f.height}
enc = fit_encodings(train_f, cfg)
tr, tg = attach_encodings(train_f, enc), attach_encodings(target_f, enc)
```

`enc` is fitted on `train_f`'s labels and then joined back onto `train_f`. The docstring of
`attach_encodings` (`train.py:210`) — *"Pure join: nothing is derived from df's own labels"* — is true
of the function in isolation and false of the `fit → attach` pair on line 368. The module docstring's
first invariant (`train.py:15-16`) — *"fitted on the training side of the fold and only joined onto
the validation/ranking side"* — states the weaker property that **is** held (no leakage into the
validation fold) and is silent about the training side.

### 1.3 Exactly which rows are affected, and which already read a clean table

There are five row populations. Two are clean today, three are not.

| # | population | where it is built | encoder it reads | own labels in it? | status |
|---|---|---|---|---|---|
| 1 | **model A training rows** `m_tr` (matched, `mvt_minus_aobt3` non-null; ≈98.8 % of rows) | `train.py:379` | `fit_encodings(train_f)` | **yes** | **DIRTY — Greedy TS** |
| 2 | **model B training rows** `u_tr` (unmatched; ≈1.2 %) | `train.py:398` | same `enc` | **yes** | **DIRTY — Greedy TS** |
| 3 | **inner early-stopping holdout** — the last `inner_holdout_months` = 1 calendar months of (1) and of (2), separately per model | mask at `train.py:285-299`, used at `train.py:332-346` | same `enc` | **yes** | **DIRTY — this is merry-quicksand's BC-1** |
| 4 | **validation fold** `va_f` | `run_cv` at `train.py:491-493` → `target_f` | `fit_encodings(train months only)` | no | **CLEAN** |
| 5 | **2026 ranking frame** | `predict.py:31` → `target_f` | `fit_encodings(all 12 labelled months)` | no (unlabelled) | **CLEAN** |

Two consequences of that table that are easy to get wrong and that the plan depends on:

* **Population 3 is a subset of populations 1 and 2, not a separate frame.** `_fit` receives the
  already-encoded frame and carves the holdout out of the *design matrix*
  (`train.py:314, 332-335`: `hold = _inner_holdout_mask(train, ...)`, then `Xtrain[~hold]` /
  `Xtrain[hold]`). There is no second `attach_encodings` call, so the holdout cannot have a different
  encoder than the rows it is stopping. Repairing 3 without repairing 1–2 therefore requires a
  *second* set of encoding columns, exactly as merry-quicksand's module docstring says
  `[S-repo]` (`/tmp/prc-merry-quicksand/prc/encoding.py:28-34`): *"A single design matrix cannot serve
  both stages correctly, because a stopping row is an evaluation row in stage 1 and a training row in
  stage 2, and it needs different encodings in each."*
* **Population 4 is already clean and is guarded.** `tests/test_train.py:69-86`
  (`test_encodings_do_not_depend_on_target_side_labels`) blanks the target frame's labels and asserts
  no encoding column and no prediction changes; `docs/guard_audit.md` records guard **G1** as failing
  correctly under mutation. That test **cannot** catch the present defect, by construction: it
  perturbs the target side only. `[S-repo]`

### 1.4 Two more call sites

* `[S-repo]` `feature_parity_report` (`train.py:440-460`) repeats the same `fit → attach` on
  `train_f` at `:445-446`. It is a diagnostic — it reports per-column null rates — but it is on the
  **submission path**: `predict_ranking` (`predict.py:24-30`) raises and refuses to predict when any
  column's |null_rate_train − null_rate_target| exceeds `PARITY_GAP_LIMIT = 0.20`
  (`predict.py:17`). Any cross-fitting scheme changes train-side null rates for `ref_*` and `arh_*`
  (see §3.6), so this gate must be updated in the same change or the submission path breaks.
* `[S-repo]` The model-B degenerate branch at `train.py:418-421` predicts `u_tg["ref_p10_fb"]`
  directly when there are fewer than 200 unmatched training rows. That reads the *target-side* column,
  so it is clean; it is not reached in production (22,470 unmatched training rows,
  `docs/investigation_2026-09-16/results/holdout-encoding-self-inclusion.md` §2).

### 1.5 Why `reference_min_count: 30` does not close it

`[S-repo]` `train.py:88` states the intent: *"self-inclusion in a P10 over a small group is a leak of
1/n"*. That arithmetic is right for a **mean** and wrong for an **order statistic**. In a group of
exactly 30 rows the 10th percentile is (with polars' default linear interpolation) a function of the
3rd and 4th order statistics alone: for the ≈2 rows that sit there, `ref_p10` is *their own label*,
not 1/30 of it. `[X]` The bound is on the *average* dilution, not on the worst case, and the worst
case is concentrated in exactly the rare stand × runway cells where the encoding carries the most
information.

Measured, in our own repository `[S-repo]`
(`docs/investigation_2026-09-16/results/holdout-encoding-self-inclusion.md` §3.1, model B's 1,649
December holdout rows): median ref group size 428; median own-month share of the group 8.84 %;
**maximum share 100 %** (groups made only of the held-out month); 4.1 % of rows sit in a ref group
more than 20 % own-month. So the guard bounds the typical row and not the tail.

---

## 2. The design constraint: train-time and serve-time distributions must match

### 2.1 The measured hazard

`[S]` slot 22, `results/22-repo-kind-mango.md:423-448`, quoting their `RECAP.md:66` verbatim:

> *"Two mechanisms cooperated: (1) the 'leak' MD4 was meant to fix is small (1/N per row where N>=20),
> and removing it adds noise without removing overfitting; (2) MD4 breaks the training-serving
> alignment (training rows use per-month fits missing their own month, serving rows use the full-year
> fit), so the tree learns split thresholds calibrated to the OOF distribution and misapplies them at
> serve time. LTFM shift +11.73 s was the biggest single-airport move of any encoder lever."*

Board cost: **v60 = +2.25 s** against their then-best v57 `[S]` (`22-repo-kind-mango.md:62`). The
control is the informative part: **v58 went the opposite way — refit the encoders on all 12 months,
i.e. *more* in-sample leak — and lost +1.29 s** `[S]` (`:444-446`). Their conclusion, verbatim:
*"v57's v26 encoders (partial exclusion of months 1,7; training-serving aligned) sit at the local
optimum. Both encoder levers now closed."*

`[X]` Read carefully, that pair says the objective being optimised is **not** leak-freeness. It is
train/serve consistency, and both deviations from it — more leak and less leak — cost about the same.

### 2.2 The constraint, stated as a rule this plan is bound by

> **C1.** For every encoding column, the *conditional distribution of the column given the row's other
> features* must be the same on the rows the model is fitted on and on the rows it is served. In
> practice: the same map, or two maps built from the same number of observations with the same
> estimator.

`[S]` This is nearly but not exactly the CatBoost paper's property **P1**, stated verbatim in §3.2 of
<https://arxiv.org/pdf/1706.09516>:

> P1 `E(x̂ⁱ | y = v) = E(x̂ⁱₖ | yₖ = v)`, where (xₖ, yₖ) is the k-th training example.

and alongside it:

> P2 Effective usage of all training data for calculating TS features and for learning a model.

`[X]` **P1 and C1 are different constraints and the difference is the whole hazard.** P1 is about the
*first moment conditional on the label*; C1 is about the whole distribution, including its variance.
An 11-month-out-of-12 median satisfies P1 and violates C1, because it is noisier than the 12-month
median a served row reads — and a tree's split thresholds are calibrated to that noise level. The
paper itself concedes the variance problem for Ordered TS in the same paragraph `[S]`:

> Note that, if we use only one random permutation, then preceding examples have TS with much higher
> variance than subsequent ones. To this end, CatBoost uses different permutations for different steps
> of gradient boosting.

`[X]` So even CatBoost does not get C1 for free; it buys it back with permutation averaging inside the
booster, which is not something a hand-built polars join can do.

### 2.3 The canonical fix does not satisfy C1 either — say so plainly

`[S]` scikit-learn `TargetEncoder`
(<https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html>):
`fit_transform` cross-fits with `cv=5` and *"is the recommended method for encoding training data"*;
`fit()` is *"discouraged … because it can introduce data leakage"*; and `transform`
*"internally uses the `encodings_` attribute"*, which the docs define as *"Encodings learnt on all of
X"*.

`[X]` That is precisely kind-mango's MD4 geometry: training rows read a 4/5-of-data encoder, test rows
read a full-data encoder. The library default that slot 20 cites as the standard of care **is** the
configuration that cost kind-mango 2.25 s. Citing scikit-learn is an argument for cross-fitting the
*training* rows; it is not an argument that the resulting column is safe to serve. Any plan that
quotes the sklearn docs and stops there has skipped the part that was actually measured.

---

## 3. The alternatives

Five options. For each: mechanism, compute cost, code change, sourced evidence, and — the column this
plan adds — **whether it satisfies C1**.

### 3.1 Option A — early-stopping separation only (stage 1), training rows untouched

**Mechanism.** Build a second encoder from the training months *excluding* the inner-holdout months,
and attach it only to the holdout rows for the early-stopping fit. Fit rows, refit rows and serve rows
keep the present full-frame encoder, unchanged, byte for byte.

**C1: satisfied, trivially and by construction.** The shipped model's features are not touched. The
only channel from this change to the submission is the integer `rounds` chosen at `train.py:343-345`.

**Cost.** One extra `fit_encodings` per model per fold (two extra per `predict_fold`), plus one extra
`attach_encodings` on the holdout rows. The 1–11-month ref/arh/apt/op build over 1.9 M rows is already
timed in-repo at seconds, not minutes; the measured whole-script cost of the model-B variant study was
**103 s for 3 early-stopping fits + 2 refits** `[S-repo]`
(`docs/investigation_2026-09-16/results/holdout-encoding-self-inclusion.md`, header). `[X]` Call it
under +10 % wall clock on a full CV run.

**Code change.** `_fit` (`train.py:302-357`) currently receives one frame. It would need either a
second frame for the holdout rows, or the holdout's encoding columns passed alongside and written into
`Xtrain[hold]` before the stage-1 `lgb.Dataset` is built. merry-quicksand's module docstring recommends
exactly the second shape and gives the memory argument `[S-repo]`
(`/tmp/prc-merry-quicksand/prc/encoding.py:28-34`): *"Callers should hold the 24 encoding columns
separately and write the stage-appropriate set into the matrix before each fit; at 12 months that is
about 200 MB per stage, against several GB for a second full matrix."*

**Evidence.**
* `[S]` merry-quicksand hit the identical defect and named it BC-1
  (`results/21-repo-merry-quicksand.md:597-605`): *"`fold_masks` puts the **early-stopping**
  months *inside* the training mask, so ES rows carry encodings fitted on their own labels. Measured
  on 861,290 real rows: **24 of 24 encoding columns on ES rows move when ES labels move** (0 of 24
  after repair); mean displacement 17.8 s on `te_STAND_mvt` against sd(delta) 378.4 s."* Their scope
  statement is the one to copy: *"the outer holdout is clean, so every reported RMSE and interval
  stands — but `best_iter` was chosen against a biased signal for every arm and every shipped
  version."* **The repair exists and was never wired in.**
* `[S-repo]` We already measured this variant on model B, and it is a no-op:
  `docs/investigation_2026-09-16/results/holdout-encoding-self-inclusion.md` §3.2, variant **C**
  (holdout rows encoded from months 1-11, training rows unchanged) gives `best_iteration` **542**,
  identical to production's 542; rounds 585 in both. With the booster held fixed, the leaked
  encodings make the holdout look **1.97 s worse**, not better (1,694.16 vs 1,692.19 on 1,649 rows),
  while individual predictions move 146.40 s RMS — a near-cancellation in aggregate, not a null at the
  row level.
* `[S-repo]` **Model A is unmeasured.** Same source, §5: *"Whether the same conclusion holds for model
  A, where the encodings are fitted on 2.06 M rows and the holdout is ~170 k rows … was not run
  here."* Model A is 98.8 % of rows and carries the pipeline's sharpest paired resolution
  (0.14–1.17 s, `docs/README.md:174-186`). This is the gap §4 Step 1 fills.

**Honest objection.** Option A leaves the stage-1 fit rows dirty while making the stage-1 evaluation
rows clean, so the early-stopping metric is now computed on a column distributed differently from the
training rows' — a C1 violation *inside* stage 1. `[X]` I do not think this is a defect, and the
reason is geometric: the early-stopping metric's job is to estimate generalisation to **out-of-sample**
rows, and every out-of-sample row this pipeline will ever see (validation fold, ranking frame) reads a
clean map. Option A makes the ES rows resemble the rows the estimate is *about*. The alternative —
variant B, honest everywhere in stage 1 — makes them resemble the rows the model is *fitted on*, which
is the wrong reference and is also a predictive change in disguise (it changed `best_iteration`
542 → 596, +9.9 % rounds, and moved the ranking predictions 63.3 s RMS `[S-repo]`, same source §3.2–3.3).

### 3.2 Option B — K-fold / leave-one-month-out cross-fitting of the training rows

**Mechanism.** Partition the training frame by calendar month (12 folds) or by K blocks; for each
block, fit the four tables on the other blocks and attach to that block. Serve rows keep the full-frame
map.

**C1: violated.** This is kind-mango's MD4 exactly. Training rows read an 11/12 map, serve rows read a
12/12 map.

**Cost.** K = 12 rebuilds of four group-bys over ~2.08 M rows per `predict_fold` call, ×3 folds in CV.
`[X]` merry-quicksand's O(months × levels) subtraction trick
(`/tmp/prc-merry-quicksand/prc/encoding.py:83-101`) makes this nearly free **for sums and counts** —
and it does not apply to us, because p10/p20/median do not decompose. We would pay the full
K × group-by. On polars over 2.08 M rows × 4 tables × 12 folds, `[X]` I estimate low minutes per
`predict_fold`, i.e. a CV run going from ~20 min to ~40–60 min. Not prohibitive, but not free.

**Code change.** Moderate: `fit_encodings` gains a fold assignment; `attach_encodings` becomes
fold-aware for the training side; `feature_parity_report` needs the same treatment or its gate breaks
(§1.4, §3.6).

**Evidence — and this is the part that decides the plan.**
* `[S]` CatBoost §3.2 proves the *defect*: Greedy TS *"leads to a conditional shift: the distribution
  of x̂ⁱ|y differs for training and test examples"*, with the worked example in which one split at
  threshold `t = (0.5 + ap)/(1 + a)` perfectly classifies training data and achieves accuracy 0.5 on
  test. `[X]` Note what this proves and what it does not: it proves Greedy TS is *biased*, not that
  every alternative is better in finite samples on a 2.08 M-row regression with min_count 30.
* `[S]` slot 21, `results/21-repo-merry-quicksand.md:314, 607-609`: merry-quicksand tested
  **out-of-fold (cross-fitted) encoding as a predictive change** and it **hurt**, magnitude ≈ **0.96 s**,
  listed in their "what failed — matched lane" table. Their own reconciliation, which I adopt:
  *"BC-1 is a model-selection defect, the −0.96 s is a feature-quality measurement; they are different
  experiments."*
* `[S]` kind-mango: **+2.25 s on the leaderboard** (§2.1 above).

`[X]` Two independent teams, on this same competition, measured this exact change and both lost. That
is the strongest evidence in this document and it points against Option B as a predictive change.

### 3.3 Option C — leave-one-out with smoothing

**C1: nearly satisfied** (support n−1 vs n; for a median group size of 428 `[S-repo]` the difference is
0.2 %). This is the option that looks best on the C1 axis and it fails on a different axis.

**Rejected, on a proof.** `[S]` CatBoost §3.2, verbatim:

> **Leave-one-out TS** At first glance, a leave-one-out technique might work well: take `Dₖ = D \ xₖ`
> for training examples xₖ and `Dₖ = D` for test ones. **Surprisingly, it does not prevent target
> leakage.** Indeed, consider a constant categorical feature: `xⁱₖ = A` for all examples. Let `n⁺` be
> the number of examples with y = 1, then `x̂ⁱₖ = (n⁺ − yₖ + ap)/(n − 1 + a)` and one can perfectly
> classify the training dataset by making a split with threshold `t = (n⁺ − 0.5 + ap)/(n − 1 + a)`.

`[X]` The regression analogue is immediate and worse for us than for a mean: within a fixed group, the
LOO statistic is a monotone function of the excluded row's own label, so a tree that has already split
down to that group recovers the label's rank order from the feature. Our trees have 127 leaves and
`min_data_in_leaf` 50 (`configs/main.yaml:18-19` sets 127/50 for model B;
`train.py:60-61` for model A), on 2.08 M rows — they routinely reach group granularity.

**Additional disqualifier.** `[X]` There is no closed form for a leave-one-out quantile. LOO for
`ref_p10` means recomputing a quantile per row: O(n log n) per group per row, or a rank-based trick
that is still an order of magnitude more code than the alternatives, for a statistic the paper says is
leaky anyway.

### 3.4 Option D — CatBoost-style ordered target statistics, hand-rolled

**C1: violated, severely.** `[S]` The paper's own text (quoted in §2.2): with one permutation
*"preceding examples have TS with much higher variance than subsequent ones."* A row early in the
permutation reads a statistic from a handful of observations; a served row reads one from 2.08 M. The
paper's remedy — different permutations at different boosting steps — requires being inside the
booster's iteration loop, which a polars join is not.

**Cost.** High. Cumulative order statistics in permutation order over 2.08 M rows.

**Size caveat, sourced.** `[S]` slot 20 §2.2, reading CatBoost Table 3: *"Ordered mode is particularly
useful on small datasets … the largest benefit from Ordered is observed on Adult and Internet
datasets, which are relatively small (less than 40K training examples)."* We have 2.08 M.

**Verdict.** `[X]` Reject as a hand-built feature. The correct way to obtain ordered TS is to run
CatBoost itself, which is slot 20's separate recommendation (item 2, `20-tabular-comp-craft.md:826-845`)
and a different plan. Do not reimplement it here.

### 3.5 Option E — shrink harder instead of cross-fitting

**Mechanism.** Leave the fit/attach geometry exactly as it is and reduce the weight a row's own label
carries: raise `reference_min_count` above 30; and/or blend each cell's statistic toward its parent
(stand × runway → airport × runway → airport) with pseudo-counts, as merry-quicksand does for their
mixture cells with `K_SHRINK = 5.0` over the nested keys `[ADEP] → [ADEP, sp_band] →
[ADEP, sp_fine, dayoff]` `[S]` (`21-repo-merry-quicksand.md:624-627`).

**C1: satisfied exactly.** Train and serve read the same map. This is the only option besides A with
that property.

**Cost.** Very low. `reference_min_count` is already a config knob (`configs/main.yaml:16`); hierarchical
shrinkage of a quantile toward a parent quantile is `w·child + (1−w)·parent` with `w = n/(n+k)` —
about ten lines in `reference_table`.

**Evidence.** `[S]` scikit-learn: *"Encodings are shrunk toward the global mean"*, with
`smooth="auto"` an empirical-Bayes estimate. `[S]` The CatBoost prior term `a·p` in eq. (4) is the same
device. `[S]` kind-mango's own diagnosis of why their cross-fit failed begins *"the 'leak' MD4 was
meant to fix is small (1/N per row where N>=20)"* — i.e. they concluded the smoothing already bounded
it. **Counter-evidence I am not hiding:** `[S]` the paper is explicit that smoothing does **not**
remove the conditional shift — eq. (4) *is* Greedy TS, prior and all, and it is the construction the
paper refutes. Shrinkage reduces the magnitude of the bias; it does not change its sign or existence.
`[X]` And `reference_min_count` already sits at 30 with guard G3 pinning it
(`docs/guard_audit.md`, G3: changing it to 1 makes the tests fail), so we are on the far side of the
cheap part of that curve already.

### 3.6 Option B′ — cross-fit *both* sides, so the maps match

`[X]` This is my own construction. No source measures it, and I flag that.

**Mechanism.** Cross-fit the training rows by month as in Option B, **and** change what the serve side
reads: instead of the single full-frame map, build the K fold-maps and serve the **average of the K
fold-statistics** (for a quantile: the mean of the K leave-one-month-out quantiles for that key). Both
sides then read a statistic estimated from 11/12 of the data by the same estimator, so C1 holds to the
extent that a mean-of-K-LOMO estimate has the same variance as one LOMO estimate — which it does not
exactly (averaging K of them is *less* noisy), so an honest variant serves a **single randomly chosen**
fold-map, or accepts the residual mismatch and measures it.

**Why it is worth listing.** `[X]` kind-mango tested train-OOF / serve-full (v60, +2.25 s) and
train-full / serve-full (v58, +1.29 s vs v57) and never tested train-OOF / serve-OOF. The pair of
results they published isolates *alignment* as the operative variable; B′ is the arm that tests that
isolation. If alignment is the mechanism, B′ should not carry v60's penalty.

**Cost.** Option B's cost plus persisting K maps to serve time and a reduction step. Moderate.

**Risk.** `[X]` Serving a deliberately noisier statistic than the best available one is a real
information cost, paid on every 2026 row, in exchange for a bias correction whose measured value
elsewhere is negative. I would run this only if Step 3 in §4 shows the training-row self-inclusion on
**model A** is materially larger than the model-B numbers we already have.

### 3.7 Summary table

| option | fixes training-row Greedy TS | fixes ES-holdout leak | satisfies C1 | compute | sourced outcome |
|---|---|---|---|---|---|
| A — ES separation only | no | **yes** | **yes** | +1 encoder fit per model per fold | model B: no change to rounds `[S-repo]`; model A unmeasured |
| B — K-fold / LOMO cross-fit | yes | yes | **no** | K × group-bys, no subtraction shortcut (quantiles) | ≈ −0.96 s `[S]`; **+2.25 s board** `[S]` |
| C — leave-one-out | no (paper: LOO *"does not prevent target leakage"*) | n/a | ~yes | high (per-row quantile) | refuted by proof `[S]` |
| D — ordered TS by hand | yes | yes | **no** (high train-time variance, acknowledged in the paper) | high | benefit documented only < 40 k rows `[S]` |
| E — shrink harder | reduces, does not remove | no | **yes** | trivial | bias reduced not removed `[S]`; already at min_count 30 |
| B′ — cross-fit both sides | yes | yes | **approximately** | B + K maps at serve | **untested by anyone** `[X]` |

---

## 4. The plan

Six steps. Each is independently measurable and each has a stated stop condition. Steps 1–3 change no
production behaviour at all.

### Step 0 — record the scope correction (no code)

`[S-repo]` Only **three** of the four tables reach the model (`op_offset_*` is hidden,
`train.py:130` + `configs/main.yaml`). Every effect-size statement in this plan and in slot 20 item 1
should be read as applying to `ref_*`, `arh_*`, `apt_med` and the two derived columns, not to four
tables. Slot 20 §8.2 and item 1 say "four tables"; that is the count that is *fitted*, not the count
that is *served*.

**Measurable?** No. It is a correction to the premise. Cost: zero.

### Step 1 — quantify the defect on model A (diagnostic only)

The only quantification we have is model B: 22,470 training rows, 1.2 % of the data, and the
population with the *widest* paired CI (385 / 724 / 421 s on three-fold pairs; 5.57 s forward-only
`[S-repo]`, `docs/README.md:174-186`). Model A is 98.8 % of rows and is unmeasured
(`docs/investigation_2026-09-16/results/holdout-encoding-self-inclusion.md` §5).

Write a script under `docs/competitor_research_2026-09-21/scripts/` (not `src/`), following the house
pattern of `docs/investigation_2026-09-16/scripts/holdout-encoding-self-inclusion.py`, that reuses
`fit_encodings` / `attach_encodings` / `_inner_holdout_mask` without reimplementing them, and reports,
on the **forward** fold's model-A training rows:

1. **Self-inclusion dilution.** Per training row: its ref / arh / apt group size, and the share of that
   group contributed by the row's own month and by the row itself. Report median, mean, max, and the
   share of rows whose own-month share exceeds 20 %. (Mirrors §3.1 of the model-B study so the numbers
   are directly comparable.)
2. **Column displacement under LOMO.** For each of the nine `ENCODING_COLS`, the distribution of
   |value(full map) − value(LOMO map)| over training rows: rows changed, median, p90, and the change in
   null count. The model-B figures to beat or match: `ref_p10` 1.0 / 6.0 s, `ref_med` 1.5 / 7.5 s,
   `arh_med` 2.0 / 7.0 s, `apt_med` 1.0 / 2.0 s; `ref_n` 39 / 130 rows, `arh_n` 335 / 711 rows; nulls
   83 → 90 (ref) and 17 → 19 (arh) on 1,649 rows `[S-repo]`.
3. **The order-statistic worst case**, which the model-B study did not isolate: the number and share of
   training rows whose own label **is** one of the two order statistics its `ref_p10` interpolates
   between. This is the quantity `reference_min_count` was believed to bound and does not (§1.5).
4. **The C1 baseline.** For each encoding column, its distribution on model-A training rows versus on
   the 2026 ranking rows *today*: median, IQR, p01/p99, null rate. This is the reference any later
   change must not worsen (§4 Step 4's gate).

**Settles:** whether the defect's magnitude on the 98.8 % is the same order as on the 1.2 %.
**Stop condition:** if model-A displacement is at or below the model-B figures (1–2 s median against a
label whose sd is in the hundreds), Step 5 (Option B) does not get run at all — the evidence in §3.2
then stands unopposed.
**Cost:** `[X]` a few hours of work, ~10 min of compute (two encoder builds + joins over 2.06 M rows;
no model fit needed for items 1–4).

### Step 2 — write the tests that pin the defect *before* fixing it

Full detail in §6. The order matters: merry-quicksand's `test_incumbent_leak_is_real`
(`/tmp/prc-merry-quicksand/tests/test_encoding_separation.py:61-73`) asserts the **before** state so
that *"the repair cannot be declared unnecessary later"* `[S-repo]`. Copy that discipline.

**Settles:** nothing about the leaderboard; it makes every later step falsifiable.
**Cost:** low. No production behaviour changes.

### Step 3 — Option A: separate the early-stopping encoder (the correctness fix)

Change `predict_fold` / `_fit` so that the inner early-stopping holdout rows carry encodings fitted on
the training months **excluding** the holdout months, while fit rows, refit rows and the target frame
keep exactly what they have today. Update the module docstring's invariant list (`train.py:14-20`) to
state the property that now holds.

**Serve-side feature change: none.** The only channel to the submission is `rounds`.

**Measurement, in three reads:**

1. `info["rounds_matched"]` and `info["rounds_unmatched"]` before and after, per fold. `[S-repo]` On
   model B the expected answer is *no change* (542 → 542, variant C). If model A's rounds also do not
   move, the fix is a no-op on predictions and ships on correctness grounds alone with zero risk.
2. If rounds **do** move on model A: `make compare A=<before> B=<after>` under the standing rule
   (`src/prc/compare.py:112-140`: adopt only when the full-rows CI *and* the drop-200 CI both exclude
   zero on the improving side). Screening on `folds: [forward]` + `--skip-predict`, ~5 min per arm
   `[S-repo]` (`docs/README.md`, §6 "どの fold を判定に使うか").
3. Feature-distribution parity is unchanged by construction; assert it rather than measure it (test T4,
   §6).

**Stop condition:** ship if rounds are unchanged or the comparison adopts. If rounds move and the
comparison rejects, that is information about round-count sensitivity, not about the leak — record it
and pin `rounds` explicitly rather than reverting the correctness fix.

**Cost:** `[X]` moderate code (the two-design-matrix problem, §3.1), ~+10 % CV wall clock.

### Step 4 — build the alignment gate (instrument, no behaviour change)

Before any predictive change to the encodings, add the instrument that would have caught kind-mango's
v60. Extend `feature_parity_report` (`train.py:440-460`) from **null-rate** parity to
**distribution** parity: for each column the model actually sees, report train-side and target-side
median, IQR, p01/p99 and null rate, plus a single scalar distance (a Kolmogorov–Smirnov statistic on a
subsample is enough).

Two reasons this is its own step:
* `[S-repo]` `predict_ranking` (`predict.py:24-30`) already refuses to submit when a null-rate gap
  exceeds `PARITY_GAP_LIMIT = 0.20`. Any cross-fitting scheme moves train-side null rates for `ref_*`
  and `arh_*` — measured at 83 → 90 and 17 → 19 on 1,649 model-B rows `[S-repo]` — so this gate must be
  understood before it is tripped.
* `[X]` The count columns `ref_n` and `arh_n` are the sharpest instance of the hazard: they **are** the
  support. Under K-fold cross-fitting they shrink by ~1/K at train time and stay full size at serve
  time. merry-quicksand's own code says so in a one-line docstring `[S-repo]`
  (`/tmp/prc-merry-quicksand/prc/encoding.py:69-70`): *"Support of each row's level in the fitting set.
  Cross-fitting changes this."* Design decision this plan pre-registers: **the count columns are never
  cross-fitted.** They are counts of labelled rows, not functions of label *values*, so they carry
  almost no target information, and leaving them on the full map is the C1-preserving choice.

**Settles:** it makes C1 a checkable gate instead of a warning in a document.
**Cost:** low.

### Step 5 — Option B / B′ as a predictive change, pre-registered

Run **only if** Step 1 shows model-A displacement materially larger than model B's.

Pre-registration, written before the run, per the house rule that no threshold may be chosen after
seeing the result:

* **Prior:** the change loses. Two sourced measurements, ≈0.96 s `[S]` and +2.25 s on a real
  leaderboard `[S]`.
* **Arms:** (i) baseline; (ii) B = LOMO cross-fit training rows, full map at serve, counts not
  cross-fitted; (iii) B′ = LOMO cross-fit both sides.
* **Screening fold:** `forward` only (`docs/README.md`, §6). Decision metric: `make compare`, both CIs.
* **Alignment gate (Step 4):** an arm whose train-vs-serve distance on any encoding column exceeds the
  baseline's is reported as *misaligned* and is not eligible for adoption regardless of its CV score.
  This is the gate whose absence cost kind-mango an upload slot.
* **Per-airport read:** kind-mango's failure was concentrated — *"LTFM shift +11.73 s was the biggest
  single-airport move"* `[S]`. `run_cv` already records `by_airport` (`train.py:499`); require that no
  single airport moves more than 3× the pooled move in the harmful direction.

**Stop condition:** if B and B′ both reject, close the lever and record it, as both competitors did.

### Step 6 — Option E as the fallback, if and only if Step 5 closes

If cross-fitting is closed but Step 1 showed the order-statistic worst case is material, the remaining
C1-preserving lever is hierarchical shrinkage of `ref_p10/p20/med` toward the airport × runway and then
airport parents with pseudo-counts (§3.5). One knob, same map both sides, measured under the same rule.

---

## 5. How big is the effect? An honest answer

**I cannot give a point estimate, and no source supports one.** What I can give are three bounds and a
sign.

**Upper bound, from the encodings' own total worth.** `[S-repo]` Removing all nine encoding columns
costs **+37.37 s** on full-data CV, **+1.25 s** with the worst 200 rows removed, **+1.24 s** at 2,000
removed, and **+450 s** on unmatched rows (`docs/README.md:197`; `docs/handoff.md:141`;
`docs/investigation_2026-09-16/plan.md:390`). `[X]` A change to *how* a feature is computed cannot
plausibly be worth more than having the feature at all. On the metric the adoption rule actually reads
— the drop-200 comparison, `src/prc/compare.py:41,112-140` — that ceiling is **≈1.25 s**. The 37.37 s
full-data figure is a tail number: it collapses to 1.25 s once 200 rows go, which by this repository's
own decision rule means it is those rows.

**The one directly measured magnitude we own.** `[S-repo]`
(`docs/investigation_2026-09-16/results/holdout-encoding-self-inclusion.md` §3.1–3.3, model B):
removing one month from the encoder moves the value columns by **1–2 s at the median and 6–7.5 s at
p90**; with the booster held fixed, the leaked encodings make the ES holdout look **1.97 s worse**
(1,694.16 vs 1,692.19), i.e. the sign of the leak's effect on that metric is *opposite* to the naive
"leakage flatters the holdout" expectation; and the downstream move on the 5,290 unmatched ranking rows
from the induced round-count change is **63.3 s RMS** (91.5 on `ms ≥ 2 h`, 533.2 on the 25 rows with
`ms ≥ 12 h`, 130.9 at LIRF). For scale from the same file: the refit-seed-key change moved the same rows
by 314.5 s and v8 − v4 by 1,735.5 s.

**The two outcome measurements that exist, both negative.** `[S]` merry-quicksand: out-of-fold encoding
as a predictive change, **≈0.96 s in the harmful direction**
(`21-repo-merry-quicksand.md:314, 607`). `[S]` kind-mango: **+2.25 s on the leaderboard**
(`22-repo-kind-mango.md:62, 423-448`), with the control arm in the opposite direction also losing
(+1.29 s).

**Plain statement.** Every sourced measurement of the predictive version of this change is a loss of
order 1–2 s. The theory `[S]` says the construction is biased; the two field measurements `[S]` say
removing the bias costs more than the bias does. `[X]` My expectation is therefore: **Fix 1 is worth
approximately zero and should be done anyway because it is correct and free of risk; Fix 2 is more
likely than not to cost 1–2 s, and its upper bound on the deciding metric is ~1.25 s in either
direction.** That asymmetry — bounded upside, measured downside, at an unknown but non-trivial compute
cost — is why the plan runs Fix 1 unconditionally and gates Fix 2 behind Step 1's diagnostic.

---

## 6. Measurement: what settles it, and telling a correctness fix from a regression

### 6.1 The standing rule

`[S-repo]` `make compare A=<exp_id> B=<exp_id>` → `src/prc/compare.py`. `verdict()` (`:112-140`) adopts
B **only** when the full-data CI and the drop-200 CI both exclude zero on the improving side; drop
levels are (0, 20, 200, 2000) (`:41`). Screening on `folds: [forward]` — the only fold that is both
forward-looking and adequately trained (`docs/README.md`, §6). Resolution: matched-row paired CI width
**0.14–1.17 s**, RMSE-full **3.97 s**, forward-only unmatched **5.57 s** (`docs/README.md:174-186`).

`[S-repo]` **The rule is itself a hypothesis, and the repo says so.** `docs/README.md`'s 2026-09-17
correction: the two-case validation of "CV direction transfers to the board" *"is not reproduced by any
available pair"* and the claim remains **UNDETERMINED**
(`docs/investigation_2026-09-16/results/decision-rule-blind-to-tail.md` §4, §5). Do not present a CV
verdict on this change as a leaderboard prediction.

### 6.2 The specific reads that settle *this* change

1. **`rounds_matched` / `rounds_unmatched`, per fold, before and after** (from `info`,
   `train.py:391, 414`). For Step 3 this is the whole causal channel. If it is unchanged, the change is
   provably a no-op on predictions and the argument is over.
2. **Train-side versus holdout-side RMSE gap on model A.** The prediction-shift claim `[S]` is
   specifically that the model's conditional distribution differs between training and unseen rows. The
   falsifiable prediction is that **the gap between in-sample and out-of-sample RMSE shrinks after the
   fix even if the out-of-sample RMSE itself does not improve.** This is the only measurement that
   speaks to the *mechanism* rather than to the outcome, and neither competitor reports it.
3. **The Step 4 alignment distance**, per encoding column, before and after. Gate, not a metric.
4. **Per-airport deltas** (`by_airport`, `train.py:499`), because the one sourced failure was
   concentrated at a single airport at +11.73 s `[S]`.
5. **One leaderboard submission**, and only for an arm that has passed (1)–(4). Slot 20's item 1 calls
   this "the decisive read"; `[X]` given §6.1's UNDETERMINED status it is the *only* read that is not
   conditional on an unvalidated rule.

### 6.3 "A correctness fix can make CV look worse while making the board better" — why that defence does **not** apply here

This is the argument slot 20 anticipates, and in this pipeline's geometry it is **not available**, for a
reason worth stating precisely.

The defence works when the validation fold is contaminated the same way the training rows are: removing
the leak then costs the fold an artificial advantage it enjoyed and that the board never will. **Our
validation fold is already clean.** Population 4 in §1.3 reads an encoder fitted on training months
only; guard G1 pins it and `tests/test_train.py:69-86` tests it `[S-repo]`. The CV estimate is therefore
measured against the same out-of-sample encoder geometry the ranking frame gets.

`[X]` It follows that a pure correctness fix has **no mechanism** by which it should make CV worse. If
CV gets worse by more than the paired CI, the cause is not "we removed a crutch the fold was leaning
on"; the cause is that we changed the column the model trains on and broke C1 — which is kind-mango's
+2.25 s, a real regression that will show on the board too.

So the discriminator is:

| observation | reading |
|---|---|
| CV unchanged within the paired CI, alignment distance unchanged | correctness fix, no predictive effect. Ship Fix 1 on correctness grounds. |
| CV improves on both CIs, alignment distance unchanged | genuine gain. Ship. |
| CV worsens, **alignment distance unchanged**, `rounds` moved | the round count moved, not the feature. Re-run with `rounds` pinned to isolate; do not revert the correctness fix on this evidence. |
| CV worsens, **alignment distance increased** | C1 violated. This is a regression, it is the measured failure mode, and it will not be rescued by the board. Reject. |
| CV improves but alignment distance increased | reject anyway, per the Step 5 pre-registration. This is the combination that most needs a pre-registered gate, because it is the one where the temptation to ship is strongest. |

`[X]` The one residual case where "worse CV, better board" is genuinely possible: the fix changes the
tail-row behaviour and the CV tail is not the board's tail (`docs/README.md` §6: 10 of 672,428 rows
carry 24 % of squared error, and they are different rows in 2025 and 2026). That is exactly what the
drop-200 arm of `verdict()` exists to detect, and it is reported as *tail-only* rather than as a null
(`src/prc/compare.py:113-120`). A tail-only verdict on this change is a legitimate reason to spend an
upload slot; a whole-data rejection is not.

---

## 7. Test plan

Seven tests. The class of defect to prevent is: **a label-derived column whose value on a row depends,
directly or through an aggregate, on that row's own label — or whose distribution differs between the
rows the model is fitted on and the rows it is served.** Today's suite tests neither half of that.

Existing coverage, stated honestly `[S-repo]`:
* `tests/test_train.py:69-86` tests only that the **target** side is clean. It passes today and would
  pass with every defect in §1 present.
* `docs/guard_audit.md` G1 (encodings fitted on the training side only) and G3
  (`reference_min_count = 30`) both fail correctly under mutation. G2 (early stopping uses an inner
  holdout) originally **passed** under mutation — the test used `num_rounds=30 / early_stopping=10` and
  early stopping never fired — and was strengthened to assert that stopping actually fires before
  checking anything. Any new ES test must keep that assertion.

**T1 — the defect itself, pinned before the repair.** Sensitivity harness modelled on
`/tmp/prc-merry-quicksand/tests/test_encoding_separation.py:49-56` `[S-repo]`: perturb the labels of a
row set R by a large constant, refit, and compare a column on a watch set W. Assert that **today** the
training rows' `ref_p10` / `arh_med` / `apt_med` move when the training rows' own labels move, and that
the validation-fold rows' do not. Their stated reason for asserting the before-state: *"so the repair
cannot be declared unnecessary later."*

**T2 — early-stopping rows do not see their own labels (Fix 1's acceptance test).** With
`inner_holdout_months = 1`, perturb the labels of the holdout months only and assert that **no**
`ENCODING_COLS` value on the holdout rows moves. Must first assert that early stopping actually fired
(`num_rounds` large enough that `best_iteration < num_rounds`), per G2's recorded weakness.

**T3 — training rows do not see their own labels (Fix 2's acceptance test; skipped until Fix 2 lands).**
The same sensitivity check with R = W = a month of training rows. Two directions, as merry-quicksand's
`test_stage1_fit_rows_are_cross_fitted_by_month` does (`:88-101`): the row's **own** month's labels must
not reach it, and a **different** month's labels **must** — otherwise the test passes on a constant
column.

**T4 — the alignment invariant (the new class, and the one nobody has).** For each encoding column,
assert that the map read at train time and the map read at serve time are produced by the same
estimator from the same number of observations — concretely, that `ref_n` and `arh_n` come from the
identical table on both sides, and that the train-vs-serve distribution distance from Step 4's
instrument does not exceed a pinned bound. This is the test that would have caught kind-mango's v60
before the upload slot was spent `[S]`.

**T5 — the count columns are never cross-fitted.** A direct assertion of the Step 4 design decision, so
that a later "make it consistent" refactor cannot quietly shrink the support columns at train time
only. Cites `/tmp/prc-merry-quicksand/prc/encoding.py:69-70` `[S-repo]`.

**T6 — order-statistic self-inclusion bound.** Today's `reference_min_count` guard (G3) asserts a
config value. Add the property it was meant to buy: for a synthetic group of exactly `min_count` rows,
perturbing **one** row's label must not move that group's `ref_p10` by more than a stated bound — and
assert the bound that actually holds for an interpolated quantile, not the 1/n bound the comment at
`train.py:88` assumes. `[X]` I expect this test to fail as written against the comment's claim; that
failure is the point (§1.5).

**T7 — fold assignment is deterministic and month-derived.** If any cross-fitting lands, assert the
fold key comes from `MVT_TIME_UTC_mvt`'s calendar month and not from row order or an RNG. Two reasons:
reproducibility (the pipeline already fights LightGBM non-determinism, `train.py:68-76`), and
merry-quicksand's stated reason for choosing months over random rows `[S-repo]`
(`/tmp/prc-merry-quicksand/prc/encoding.py:43-45`): *"Whole-month exclusion is deliberate. Random row
folds would leak through same-day, same-stand structure."* Our own folds are month-blocked for the
same reason (`src/prc/cv.py:20-51`).

**T8 — mutation audit.** Run every new guard through the `docs/guard_audit.md` discipline: break the
guard, confirm the test goes red. `[S-repo]` Three of nine guards passed under mutation on the first
audit and needed strengthening; assume these will too until shown otherwise.

---

## 8. Facts versus inference

**Facts (read from code or from a cited source):** every line reference in §1; the five row populations
and which two are clean; that `OFFSET_COLS` is hidden in production; the CatBoost §3.2 text on Greedy
TS, Holdout TS, Leave-one-out TS, Ordered TS, P1 and P2, quoted verbatim from
<https://arxiv.org/pdf/1706.09516>; the scikit-learn `TargetEncoder` statements on cross-fitting,
`cv=5`, `encodings_` and `smooth`; merry-quicksand's BC-1 description, their unwired repair's two-stage
design, their `_crossfit_one` subtraction trick, their count-column docstring, and their −0.96 s
out-of-fold result; kind-mango's v60 +2.25 s, v58 +1.29 s and the verbatim RECAP.md diagnosis; all
numbers from `docs/investigation_2026-09-16/results/holdout-encoding-self-inclusion.md`; the +37.37 /
+1.25 / +1.24 / +450 encoding ablation; the `verdict()` rule, drop levels, fold choice and paired-CI
resolutions.

**Inference (mine, marked `[X]` above and repeated here so it can be attacked):**
(i) that `reference_min_count = 30` fails to bound an interpolated quantile's worst case — argued from
the definition of the statistic, **not measured** (T6 is the test that would settle it);
(ii) that a change to how a feature is computed cannot be worth more than the feature itself, giving a
~1.25 s ceiling on the deciding metric — a plausibility bound, not a theorem;
(iii) that P1 and C1 are different constraints and that C1 is the one the field measurements are
about — argued from the paper's own variance caveat plus kind-mango's diagnosis;
(iv) that a pure correctness fix has no mechanism to worsen our CV, because our validation fold is
already clean — this is the load-bearing claim of §6.3 and it rests on §1.3's population table being
complete;
(v) that Option B′ (cross-fit both sides) is the untested arm that isolates alignment — no source
measures it;
(vi) the compute estimates in §3, which are order-of-magnitude guesses from the one timed script we
have (103 s for 5 model-B fits).
