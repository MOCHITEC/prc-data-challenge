# P2 Fix 1 — the early-stopping holdout stops reading its own labels

Executed 2026-09-25 against `docs/competitor_research_2026-09-21/plans/P2-cross-fit-encodings.md`,
Step 0 → Step 1 → Step 2 → Step 3. **Steps 4–6 were not run** and the reason is recorded in §7.

**Marking convention**, as in the plan: `[S-repo]` read from this repository or measured by a script in
it, with `path:line` or the section that produced it; `[X]` my own inference or arithmetic, never
dressed as a measurement.

Artefacts this document is made of:

| what | where |
|---|---|
| Step 1 diagnostic script | `docs/competitor_research_2026-09-21/scripts/p2_fix1_diagnostic.py` |
| Step 1 raw output | `docs/competitor_research_2026-09-21/results/p2-fix1-step1.json` |
| Step 2 tests | `tests/test_encoding_separation.py` (12 tests) |
| Step 3 code change | `src/prc/train.py` — `TrainConfig.es_holdout_encodings`, `_es_encoder`, `_fit`, `_fit_grouped`, `predict_fold`, module docstring |
| Step 3 arms | `configs/p2_fix1_base.yaml`, `configs/p2_fix1_es.yaml` |
| Step 3 raw comparison | `docs/competitor_research_2026-09-21/results/p2-fix1-compare.txt` |
| Step 3 experiment ids | `p2_fix1_base-40cfc6bb39`, `p2_fix1_es-132b7a3ef0` (`experiments/experiments.jsonl`, `experiments/oof_*.parquet`) |

---

## 0. Verdict up front

**SHIP, on correctness grounds. §6.1 row 1.**

The repair works, it is not a no-op, and it buys nothing measurable — which is exactly what the plan
pre-registered (*"Fix 1 is worth approximately zero and should be done anyway because it is correct
and free of risk"*, plan §5).

| | |
|---|---|
| defect confirmed on model A | yes — every early-stopping row's encoding groups include its own month (183,759 of 183,761 for `ref`, all of them for `arh` and `apt`), median own-month share 11.5 % and maximum 100 % (§2.1, §2.2) |
| repair is not a silent no-op | yes — 127,393 of 183,761 `ref_p10` values change, and all nine columns move (§4.2, from the run's own `info` record) |
| serve-side feature change | **none**, by construction and by test (§4.1, §3) |
| round count moved | yes: `rounds_matched` 1134 → 1070, `rounds_unmatched` 82 → 80 (§4.2) |
| CV, forward fold | 219.7247 → 219.7648, **+0.0401 s**, CI95 **[−0.06, +0.17]** — zero is inside it |
| `prc.compare` verdict | REJECT — *"no difference on all rows"*. That is the null, not a regression |
| alignment distance | unchanged by construction; asserted, not measured (§4.1) |
| wall clock | 731.7 s → 703.2 s, i.e. **faster**, because the extra encoder fit costs less than the 64 rounds it removed |

`[X]` The one thing I would not let a reader take away: this is **not** a measured improvement, and
`prc.compare` says REJECT. The reason to ship is that the code was wrong, the repair costs nothing,
and the decision rule's REJECT here means *"no difference"*, which is the outcome the plan predicted.
A reader who wants the number in one line: the early-stopping holdout's leak is worth **0.04 ± 0.12 s**
on the forward fold, indistinguishable from zero.

---

## 1. Step 0 — the scope correction, confirmed

`[S-repo]` `src/prc/train.py` — `use_offset_encoding` defaults to `False` and none of
`configs/main_gap.yaml`, `p2_fix1_base.yaml`, `p2_fix1_es.yaml` sets it, so `visible_numeric_columns`
removes `op_offset_med` / `op_offset_n` from the feature list. `operator_offset_table` is still fitted
on every fold. **Three tables reach the model, not four**: `ref_*`, `arh_*`, `apt_med`, plus the two
derived columns `ref_p10_fb` and `ref_missing`. Nine visible columns.

Every effect size below is therefore about three tables. The repair re-encodes all four (including the
hidden `op` table) because leaving one of them on the leaked map for no reason would be a second
undocumented decision; it has no effect on the model while `use_offset_encoding` is off.

---

## 2. Step 1 — the defect measured on model A

Script `p2_fix1_diagnostic.py`, production forward fold: train months 202501–202510, valid 202511–202512.
Populations `[S-repo]` (§`rows` of the JSON):

| population | rows |
|---|---|
| fold training frame, labelled, both lanes | 1,757,038 |
| `A_tr` — model A's training rows (`mvt_minus_aobt3` non-null) | 1,737,573 |
| `A_es` — the inner early-stopping holdout inside `A_tr` (202510) | 183,761 |
| `A_fit` — the rows the stage-1 booster is fitted on | 1,553,812 |

Model B's comparable numbers, from `docs/investigation_2026-09-16/results/holdout-encoding-self-inclusion.md`
(22,470 training / 1,649 holdout rows), are quoted beside each of ours.

### 2.1 Self-inclusion dilution (plan Step 1 item 1)

On the 183,761 `A_es` rows `[S-repo]` (§`s1_dilution`):

| table | median group n | median own-month share | mean | max | share of rows > 20 % own month |
|---|---|---|---|---|---|
| ref (`ref_p10/p20/med/n`) | 460 | **11.50 %** | 13.78 % | 100 % | **11.30 %** |
| arh (`arh_med/arh_n`) | 3,702 | 11.18 % | 13.23 % | 100 % | 10.18 % |
| apt (`apt_med`) | 192,927 | 10.53 % | 10.58 % | 10.94 % | 0 % |

Model B's figures for the same three rows-in-group statistics were 8.84 % / 8.54 % / 8.26 % at the
median and 4.1 % / 3.4 % / 0 % above 20 %.

`[X]` Model A's holdout is **more** self-included than model B's, not less, and the reason is
arithmetic rather than interesting: the fold's training side is ten months, so one month is ~10 % of it
by construction, against model B's twelve-month frame where one month is ~8 %. The tail is the part
that matters and it is roughly 2.7× model B's: 11.3 % of `A_es` rows sit in a ref group more than a
fifth of which is their own month, against 4.1 % on model B, and the maximum share is 1.0 on both —
ref groups that consist only of the holdout month exist on model A too.

### 2.2 Column displacement (plan Step 1 item 2)

`[S-repo]` (§`s2_displacement.fix1_es_rows`). Production map (all ten training months) against the map
the repair uses (nine months, 202510 excluded), on the 183,761 `A_es` rows. Model B's figures in the
last column, from the source above, measured on 1,649 rows.

| column | rows changed | median \|Δ\| | p90 \|Δ\| | nulls before → after | model B: median / p90 |
|---|---|---|---|---|---|
| `ref_p10` | 121,527 (66 %) | **2.0 s** | 13.0 s | 5,866 → 8,756 | 1.0 / 6.0 s |
| `ref_p20` | 128,975 (70 %) | 2.0 s | 12.0 s | 5,866 → 8,756 | 1.0 / 6.0 s |
| `ref_med` | 149,390 (81 %) | 2.5 s | 15.0 s | 5,866 → 8,756 | 1.5 / 7.5 s |
| `ref_n` | 177,895 (97 %) | 58 rows | 161 rows | 5,866 → 8,756 | 39 / 130 rows |
| `arh_med` | 152,594 (83 %) | 2.0 s | 10.0 s | 86 → 339 | 2.0 / 7.0 s |
| `arh_n` | 183,675 (100 %) | 412 rows | 765 rows | 86 → 339 | 335 / 711 rows |
| `apt_med` | 169,734 (92 %) | 2.0 s | 7.0 s | 0 → 0 | 1.0 / 2.0 s |
| `ref_p10_fb` | 126,773 (69 %) | 3.0 s | 30.0 s | 0 → 0 | 1.0 / 7.0 s |
| `ref_missing` | 2,890 (1.6 %) | — | — | 0 → 0 | 7 rows |

Table sizes `[S-repo]`: ref 6,100 → 5,792 groups, arh 1,009 → 994, apt 10 → 10.

**Against the plan's stop condition.** Plan Step 1: *"if model-A displacement is at or below the
model-B figures (1–2 s median against a label whose sd is in the hundreds), Step 5 (Option B) does not
get run at all."* Model A's value columns move **2.0–3.0 s at the median and 7–30 s at p90**, i.e.
about 2× model B at the median, 2× at p90 on `ref_*`, and 4× on `ref_p10_fb`. That is inside the
plan's stated band (*1–2 s median*) at its upper edge and is the **same order**, not a different one.
The stop condition is met: **Step 5 is not run.** §7 says what would change that.

The same script also measured the **Fix 2** population for completeness — every `A_tr` row against a
map that excludes its own month `[S-repo]` (§`s2_displacement.fix2_all_training_rows_lomo`):
`ref_p10` 2.0 / 12.0 s over 1,188,553 rows, `arh_med` 3.0 / 12.0 s, `apt_med` 2.0 / 4.0 s, nulls
43,401 → 57,415 on ref. `[X]` So the training rows are displaced by the same per-row magnitude as the
holdout rows — the difference between Fix 1 and Fix 2 is the number of rows affected (1.74 M vs 184 k)
and whether the served column's train-time twin changes, not the size of the per-row move. This is a
measurement for the Fix 2 decision and **is not acted on here**.

### 2.3 The order-statistic worst case — and a correction to the plan (item 3)

**The plan's premise is wrong in a direction that makes the defect sharper, and I could not have
learned that without running it.** Plan §1.5 says *"with polars' default linear interpolation"* the
P10 of a 30-row group is a function of the 3rd and 4th order statistics. `[S-repo]` Measured on
polars 1.44.2, the default interpolation is **`nearest`**: `pl.col(y).quantile(0.10)` returns the
order statistic at rank `round(0.10·(n−1))` **verbatim**, with no blending. So on every qualifying ref
group, `ref_p10` *is* exactly one member's own label, at weight 1.0 — and `ref_p20` likewise;
`median()` for even n is the mean of the two central labels, weight ½ each.

`train.py:88`'s *"self-inclusion in a P10 over a small group is a leak of 1/n"* is the bound for a
**mean**. It is false for all three statistics, and the failure is not marginal: `[S-repo]`
`tests/test_encoding_separation.py::test_reference_min_count_does_not_bound_a_p10_to_one_over_n`
constructs a 30-row group and shows the pivot row's label moving `ref_p10` **one-for-one** (30× the
claimed bound), while any other row moves it by exactly zero.

How many rows are the pivot `[S-repo]` (§`s3_order_statistic`, 6,069 qualifying groups, median group
n = 165):

| population | rows in a qualifying ref group | own label **is** `ref_p10` | `ref_p20` | `ref_med` |
|---|---|---|---|---|
| `A_tr` | 1,693,274 | 9,897 (**0.58 %**) | 11,518 (0.68 %) | 8,868 (0.52 %) |
| `A_es` | 177,794 | 1,024 (**0.58 %**) | 1,233 (0.69 %) | 975 (0.55 %) |

`[X]` **This is the "smaller than the plan assumed" finding, and it points both ways.** Per row the
leak is far worse than `1/n` — for 0.58 % of rows the feature is the label, not a diluted version of
it. Per population it is confined: 1,024 rows of 183,761, and the count is bounded above by the number
of qualifying groups (one pivot row per group per statistic), so it can never scale with the data. A
defect that affects 0.6 % of the early-stopping rows at weight 1 and the other 99.4 % at weight
~1/460 is a defect worth fixing on correctness grounds and is not a plausible source of seconds.

### 2.4 The C1 baseline (item 4)

`[S-repo]` (§`s4_c1_baseline`). Model-A training rows under the fold encoder, against the 2026 ranking
rows under the all-12-month encoder, as production builds both **today**:

| column | train median / IQR / null | rank2026 median / IQR / null |
|---|---|---|
| `ref_p10` | 710.0 / 266.0 / 2.50 % | 712.0 / 262.0 / 3.76 % |
| `ref_p20` | 771.0 / 300.0 / 2.50 % | 773.0 / 303.0 / 3.76 % |
| `ref_med` | 908.0 / 346.0 / 2.50 % | 910.0 / 351.0 / 3.76 % |
| `ref_n` | 493.0 / 589.0 / 2.50 % | 545.0 / 695.0 / 3.76 % |
| `arh_med` | 911.0 / 246.0 / 0.03 % | 919.0 / 238.0 / 0.08 % |
| `arh_n` | 3,760.0 / 3,398.0 / 0.03 % | 4,385.0 / 4,083.0 / 0.08 % |
| `apt_med` | 953.0 / 216.0 / 0 % | 954.0 / 211.0 / 0 % |
| `ref_p10_fb` | 713.0 / 267.0 / 0 % | 715.0 / 277.0 / 0 % |

`[X]` Two things worth recording before any later change touches these numbers. The value columns are
already well aligned — medians within 2 s, IQRs within 5 %. The **count** columns are not: `ref_n` 493
vs 545 and `arh_n` 3,760 vs 4,385, because the serve-side map is fitted on twelve months and the
fold's on ten. That is a pre-existing ~16 % support gap that nothing in this plan created, and it is
the axis plan §4 Step 4 pre-registers as the one cross-fitting would widen. Fix 1 changes **none** of
these numbers, by construction and by test (§4, `test_fix1_leaves_the_parity_report_untouched`).

---

## 3. Step 2 — the tests, written before the repair

`tests/test_encoding_separation.py`. The order was: write them, run them against unrepaired `src/`,
then repair.

**State on the unrepaired code** `[S-repo]`: the module does not import — `ImportError: cannot import
name '_es_encoder' from 'prc.train'` — so every after-state test is red (eight of the eleven the file
held at that point; the twelfth, the wiring test, did not exist yet — §5 says why). Run in isolation
against unrepaired `src/`, the three before-state tests pass (`3 passed in 9.09s`): the defect is
real, the validation side is already clean, and `reference_min_count` does not bound the P10.

After the repair: **12 passed**. `FAIL` in the table below means red on the unrepaired code.

| test | what it pins | state before the repair |
|---|---|---|
| `test_es_holdout_reads_its_own_labels_today` | **T1.** With the production geometry, moving the holdout month's labels moves the holdout rows' own `ref_p10/p20/med`, `arh_med`, `apt_med`, `ref_p10_fb`. Asserted so the repair cannot be declared unnecessary later. | PASS (and must keep passing — the flag defaults off) |
| `test_validation_side_is_already_clean` | guard G1's property restated beside T1, so both halves sit together | PASS |
| `test_separated_es_encoder_ignores_the_holdout_months_labels` | **T2, direction 1.** No `ENCODING_COLS` value on the holdout rows moves when the holdout month's labels move. | FAIL |
| `test_separated_es_encoder_still_reads_the_other_months_labels` | **T2, direction 2 — the control.** The *other* months' labels must still reach those rows, or the test would pass on a constant or all-null column. | FAIL |
| `test_separated_es_encoder_is_not_a_no_op` | the separated map differs from the production map on these rows, and the per-column change counts recorded in `info` say by how much | FAIL |
| `test_separated_es_encoder_preserves_schema_and_row_order` | the block `_fit` stacks has the same columns, height and `MVT_ID_mvt` order | FAIL |
| `test_fix1_flag_defaults_to_current_behaviour` | `TrainConfig().es_holdout_encodings is False` | PASS |
| `test_fix1_early_stopping_fires_and_uses_the_separated_encoder` | asserts **stopping actually fired** before asserting anything about it (guard G2's recorded weakness), then that the encoder ran and changed columns | FAIL |
| `test_fix1_only_channel_to_the_prediction_is_the_round_count` | with the round count equal, the two arms' predictions are **bitwise** identical | FAIL |
| `test_fix1_leaves_the_parity_report_untouched` | `feature_parity_report` — which is on the submission path — is byte-identical with the flag on | FAIL |
| `test_fix1_is_wired_into_the_stage1_fit` | records the two stage-1 design matrices from inside `lgb.train`: the **fit** matrix must be identical between arms and the **stopping** matrix must differ | did not exist yet — written in response to mutant M2 (§5) |
| `test_reference_min_count_does_not_bound_a_p10_to_one_over_n` | **T6.** §2.3's property, two directions: the pivot row moves `ref_p10` one-for-one, any other row moves it by zero. | PASS |

`test_fix1_is_wired_into_the_stage1_fit` exists because of the mutation audit, not before it — see §5.

**Not written here, deliberately.** T3 (training rows do not see their own labels) is Fix 2's
acceptance test and there is nothing to accept. T4/T5 (the alignment invariant and the
never-cross-fit-the-counts rule) belong to plan Step 4, which is not in this change's scope; §2.4
records the baseline they would be written against. T7 (fold assignment is month-derived) applies only
if cross-fitting lands.

---

## 4. Step 3 — the change

### 4.1 What was touched in `src/`

`src/prc/train.py`, and nothing else in `src/`.

1. **`TrainConfig.es_holdout_encodings: bool = False`** — new field, defaulting to today's behaviour.
2. **`_es_encoder(train_f, cfg, stats)`** — new factory. Returns `build(hold_rows)`, which drops the
   eleven attached encoding/offset columns from the holdout block and re-attaches them from
   `fit_encodings` over `train_f` minus the calendar months those rows occupy. Cached by month set.
   Every build appends a record to `stats`: rows, months excluded, training rows used, resulting table
   sizes, and per-column change counts.
3. **`_fit(..., es_rows=None)`** — when given the factory and a non-degenerate holdout, the re-encoded
   block is appended to the **same** `pl.concat` that already stacks train and predict frames, and the
   stage-1 `lgb.Dataset` validation matrix is that block instead of `Xtrain[hold]`. The refit loop is
   bounded to the first `n_pred` prediction blocks so the extra block never reaches stage 2.
4. **`_fit_grouped(..., es_rows=None)`** — passes it through to the global fit and each per-airport
   fit, so the P11 branch cannot silently keep the old behaviour if it is ever turned on.
5. **`predict_fold`** — builds one factory per lane when the flag is on, passes them in, and writes
   `info["es_holdout_encodings"]`, `info["es_encoder_matched"]`, `info["es_encoder_unmatched"]`.
6. **Module docstring** — the invariant list said only that the encodings are *"fitted on the training
   side of the fold and only joined onto the validation/ranking side"*, which is the weaker property
   that was always true. It now states what the flag does hold, what it deliberately does not (the
   training rows' own self-inclusion), and why.

**One design decision worth stating, because the alternative is the obvious one.** The re-encoded block
goes into the same stacked frame rather than being matrixed on its own. `_matrix` builds each
categorical's `pl.Enum` over `sorted(unique())` of the stacked frame; a separately built matrix would
carry a different integer coding for the same category, and `train.py`'s own comment records that a
pure relabelling of those codes moved early stopping from 937 to 1101 rounds. Because the appended
rows are a subset of `train`, the level set is unchanged, so every existing row's code is unchanged and
`Xall[a:b]` is byte-identical to what it was before the flag existed. That is asserted directly by
`test_fix1_is_wired_into_the_stage1_fit`, not argued.

**One edit was made after the CV arms ran**, and it is recorded here rather than left for a reader to
find: `encoded = [c for c in [*ENCODING_COLS, *OFFSET_COLS]]` was simplified to
`encoded = [*ENCODING_COLS, *OFFSET_COLS]` — the same list, built the same way, and the only difference
between the committed `src/prc/train.py` and the file that produced §4.2's numbers. Nothing else in
`src/` changed after the arms ran.

**What was deliberately not touched.** `fit_encodings`, `attach_encodings`, `reference_table`,
`airport_runway_hour_table`, `airport_median_table`, `operator_offset_table`, `feature_parity_report`,
`predict.py`, `PARITY_GAP_LIMIT`. There is no serve-side feature change, so the parity gate needs no
update — the opposite of what plan §1.4 warns is needed for any cross-fitting scheme.

**Counts.** The holdout block's `ref_n` / `arh_n` are re-encoded along with everything else, which is
plan §3.1's Option A and the model-B study's variant C. Plan Step 4's *"the count columns are never
cross-fitted"* is a pre-registration for **Step 5**, where cross-fitting would shrink the support of
rows the model is fitted on while serve rows keep full support. It does not apply here: no fitted row
and no served row changes.

### 4.2 The measurement

Two arms, forward fold, `--skip-predict`, one process at a time, `POLARS_MAX_THREADS=2`,
LightGBM `num_threads: 4`, `lgb_deterministic` left at its default `True`. Nothing was written under
`outputs/`.

**Reproducibility check first, because this repository has twice shipped a harness that returned the
same number for every arm.** `[S-repo]` The base arm scored `rmse_full` **219.72466045903678**, which
is the archived `p4a_gap-ca4297619f` value to all sixteen digits. The base arm therefore reproduces an
independently recorded run of the same recipe exactly, and the ES arm returns a *different* number, so
the harness is neither broken nor blind.

#### (1) The round count — the whole causal channel `[S-repo]`

| | `rounds_matched` | `rounds_unmatched` | train_seconds |
|---|---|---|---|
| `p2_fix1_base-40cfc6bb39` | **1134** | **82** | 731.7 |
| `p2_fix1_es-132b7a3ef0` | **1070** | **80** | 703.2 |

Unlike model B in the 2026-09-16 study — where variant C left `best_iteration` at 542, unchanged —
**model A's round count does move**, by −64 (−5.6 %). `[X]` That is the difference the plan's Step 1
was opened to find, and it is the reason the comparison in (2) had to be run at all.

`[X]` The ES arm is **28.5 s faster**, not ~10 % slower as plan §3.1 estimated. Two `fit_encodings`
builds over 1.57 M rows cost less than the 64 boosting rounds the change removes.

#### The no-op control, from the run's own `info` record `[S-repo]`

The repair re-encoded 183,761 model-A rows and 1,913 model-B rows from a map fitted on 1,571,364 rows
(202510 excluded; ref 6,100 → 5,792 groups, arh 1,009 → 994, apt 10 → 10). Values that changed:

| column | model A (of 183,761) | model B (of 1,913) |
|---|---|---|
| `ref_p10` | 127,393 | 1,358 |
| `ref_p20` | 134,841 | 1,379 |
| `ref_med` | 155,256 | 1,588 |
| `ref_n` | 183,761 (all) | 1,912 |
| `arh_med` | 152,680 | 1,602 |
| `arh_n` | 183,761 (all) | 1,913 |
| `apt_med` | 169,734 | 1,798 |
| `ref_p10_fb` | 126,773 | 1,342 |
| `ref_missing` | 2,890 | 38 |

A build that quietly did nothing would show zeros here. It does not.

#### (2) `prc.compare p2_fix1_base-40cfc6bb39 p2_fix1_es-132b7a3ef0` `[S-repo]`

n = 328,009 rows; top-20 share of B's squared error 0.134.

| dropped | A | B | diff | CI95 | P(B better) |
|---:|---:|---:|---:|---:|---:|
| 0 | 219.72 | 219.76 | **+0.04** | **[−0.06, +0.17]** | 0.314 |
| 20 | 204.59 | 204.56 | −0.03 | [−0.06, +0.01] | 0.948 |
| 200 | 194.42 | 194.40 | −0.02 | [−0.04, +0.01] | 0.932 |
| 2000 | 174.58 | 174.58 | −0.00 | [−0.02, +0.01] | 0.710 |

Month-level bootstrap (2 months): +0.04 s [−0.08, +0.16], P(B better) = 0.290.

```
VERDICT (from the curve only, rule unchanged): REJECT - no difference on all rows: +0.04 s [-0.06, +0.17]
```

**Read the REJECT correctly.** `verdict()` has three rejecting branches (`src/prc/compare.py:112-140`):
*B is worse on all rows*, *no difference on all rows*, and *tail-only*. This is the middle one — the
full-data CI **contains zero**. It is not the "B is worse" branch. And the point estimate flips sign
the moment the tail comes off: −0.03 at K=20, −0.02 at K=200, −0.00 at K=2000, all with CIs
containing zero. `[X]` There is no drop level at which this change is measurably anything.

Lanes: matched 198.526 → 198.550 (+0.024 s on 325,004 rows); unmatched 1003.60 → 1004.04
(+0.45 s [−2.73, +3.16] on 3,005 rows).

#### (4) Per-airport, both lanes `[S-repo]`

Matched lane (pooled move +0.024 s): largest single-airport moves EDDM +0.25, LIRF +0.10, LTFM −0.07;
the other seven are within ±0.05. Unmatched lane (pooled +0.45 s), ordered by share of the
disagreement: LIRF +8.47 [−6.73, +19.55] (27.3 %), EHAM **−5.96** [−9.51, −2.78] (20.5 %),
LFPG −0.71, LTFM −1.91, EDDF −3.94, LSZH −4.26, LEBL −6.11, EDDM +0.53, EGLL −1.66, LEMD −2.30.

`[X]` Two honest notes. The per-airport moves on the unmatched lane are **mixed in sign** — one
airport up, seven down — which is the opposite of kind-mango's concentrated +11.73 s at a single
airport, and is what a round-count jitter looks like rather than a feature-quality change. And the
plan's Step-5 gate (*"no single airport moves more than 3× the pooled move in the harmful
direction"*) is **undefined here**: the pooled matched move is 0.024 s, so every ratio against it is
meaningless. The gate is written for an arm with a real pooled effect; it should not be quoted on this
one.

#### (3) Feature-distribution parity — asserted, not measured

Per plan Step 3's measurement (3). `feature_parity_report` reads `fit_encodings` / `attach_encodings`
on the full training frame and the target frame, neither of which this change touches, and
`test_fix1_leaves_the_parity_report_untouched` asserts the report is byte-identical with the flag on.
The `PARITY_GAP_LIMIT = 0.20` submission gate is therefore unaffected.

#### For scale: what a 64-round move is worth in this recipe

`[S-repo]` `results/determinism-cost.md` §3 records ten runs of `p4a_gap` with the determinism flags
**off** at three seeds: nine took 1134 rounds and one took 1113, and the whole ten-run RMSE spread was
**0.028 s**. `[X]` Our arms ran with the flags **on**, so the +0.0401 s here is not run-to-run noise —
it is the genuine consequence of −64 rounds. But the scale matches: a round-count move of this order
is worth hundredths of a second in this recipe. It is two orders below the 3.62 s the production
recipe's own feature was adopted on, and below the ~0.5 s single-column noise floor.

---

## 5. Mutation audit (plan §7 T8)

Five mutants, each applied to `src/prc/train.py` alone and reverted immediately `[S-repo]`:

| # | mutation | killed by |
|---|---|---|
| M1 | the encoder stops excluding the holdout months (`base = train_f`) | `..._ignores_the_holdout_months_labels`, `..._is_not_a_no_op`, `..._is_wired_into_the_stage1_fit` |
| M2 | `_fit` computes the block and then hands LightGBM `Xtrain[hold]` anyway | `..._is_wired_into_the_stage1_fit` — **and nothing else** |
| M3 | the encoder is fitted on an empty frame (all-null columns) | `..._still_reads_the_other_months_labels` |
| M4 | the flag defaults to `True` | `..._flag_defaults_to_current_behaviour`, `..._is_wired_into_the_stage1_fit` |
| M5 | the refit loop is not bounded, so the ES block leaks into stage 2 | `..._early_stopping_fires...`, `..._only_channel_is_the_round_count`, `..._is_wired_into_the_stage1_fit` |

**M2 survived the first audit.** Every test I had written exercised `_es_encoder` directly or read the
`info` record, and all of them pass on a build that computes the corrected block perfectly and then
discards it. That is the exact silent-no-op shape this repository has shipped twice. The audit is the
only reason it was caught. `test_fix1_is_wired_into_the_stage1_fit` was written in response: it
monkeypatches `lgb.train`, captures the two stage-1 matrices, and asserts the fit matrix is identical
between arms while the stopping matrix differs. It now kills four of the five mutants on its own.

`[S-repo]` Full suite after the change: **99 passed** (98 before, plus the new module).

---

## 6. Which §6.1 row this lands on

| observation | this run |
|---|---|
| CV unchanged within the paired CI | **yes** — +0.04 s, CI95 [−0.06, +0.17] contains zero, at every drop level |
| alignment distance unchanged | **yes** — by construction, and asserted by test |
| `rounds` moved | yes, 1134 → 1070 |

**Row 1: *"correctness fix, no predictive effect. Ship Fix 1 on correctness grounds."***

`[X]` Row 3 (*"CV worsens, alignment distance unchanged, `rounds` moved — re-run with `rounds` pinned
to isolate"*) is the row to check next, and it does not apply: it is conditioned on CV **worsening**,
and +0.04 s with a CI straddling zero is not a worsening. I did not re-run with `rounds` pinned,
because there is nothing to isolate — and the pipeline has no `rounds` override, so that re-run would
itself require a code change.

Two things this does **not** settle, stated so nobody quotes it further than it goes:

* §6.2's read (2) — *"the gap between in-sample and out-of-sample RMSE shrinks after the fix even if
  the out-of-sample RMSE itself does not improve"* — was **not run**. It is the only measurement that
  speaks to the mechanism rather than the outcome, neither competitor reports it, and it needs a
  stage-1 fit with the booster held fixed and the holdout features swapped (the model-B study's §3.2
  isolation). `[X]` It cannot change the ship/don't-ship decision, which is why it was not run inside
  Step 3's budget; it is the obvious next thing if anyone wants the mechanism rather than the number.
* §6.1's own caveat: *"Do not present a CV verdict on this change as a leaderboard prediction."*
  `docs/README.md`'s 2026-09-17 correction leaves CV→board transfer **UNDETERMINED**. Nothing here
  should be priced against the board. `[X]` If it must be: the one measured paired derivative is
  dBoard/dOOF = 0.747 at n = 1 (`results/model-a-ceiling.md` §4), which on a +0.04 s forward-fold OOF
  move — itself not a pooled-OOF move — is noise times noise. Do not spend an upload slot on this.

---

## 7. Steps 4–6: not run, and why

**Step 4 (the alignment gate) — not run.** It instruments a change this one does not make. Fix 1 moves
no train-side and no serve-side column, so there is no alignment distance to gate; §2.4 records the
baseline the instrument would be built against, including the pre-existing ~16 % `ref_n` / `arh_n`
support gap between the fold encoder and the twelve-month serve encoder.

**Step 5 (Option B / B′) — not run, and the plan's own gate is the reason.** Step 5 is conditional on
*"Step 1 shows model-A displacement materially larger than model B's"*. §2.2 measures it at about 2×
model B's median and the same order at p90 — inside the plan's stated band, not outside it. The
condition is not met.

Three further things from §2 push the same way and are recorded so the decision can be re-opened
against evidence rather than mood:

* §2.3: the population where the leak is severe rather than diluted is **0.58 % of rows**, and it is
  capped by the group count, so it cannot grow with the data.
* §2.2: the per-row displacement on the Fix 2 population is the *same* 2–3 s as on the Fix 1
  population. Fix 2 would move 1.74 M rows by that much and change the column the model is fitted on,
  which is precisely kind-mango's MD4 geometry (+2.25 s on the board) and merry-quicksand's
  out-of-fold encoding arm (≈0.96 s harmful).
* §2.4: the count columns already sit 16 % apart between train and serve. Cross-fitting the training
  rows widens exactly that gap.

`[X]` Nothing in Step 1 made me want to run Fix 2. If it had, the decision would still not be mine to
take — it is pre-registered in the plan as more-likely-than-not to cost 1–2 s, and both independent
field measurements of it are losses.

**Step 6 (Option E, hierarchical shrinkage) — not run.** It is gated behind Step 5 closing *and* the
order-statistic worst case being material. §2.3 measures the worst case at 0.58 % of rows. `[X]` If
anything is ever done about that population, the cheap and C1-preserving move is to raise the pivot
row's dilution rather than to cross-fit — but that is a new experiment with its own control arm, not a
continuation of this one.

---

## 8. Facts versus inference

**Facts (measured here or read from code).** Every number in §2 comes from
`p2_fix1_diagnostic.py` and is reproduced in `results/p2-fix1-step1.json`. Polars 1.44.2's default
quantile interpolation is `nearest` (§2.3, measured, contradicting the plan's premise). Every test
name and its before-state in §3. Every mutant and its killer in §5. The lines changed in §4.1. The CV
numbers, round counts and `prc.compare` output in §4.2.

**Inference, marked `[X]` above and repeated here so it can be attacked.**
(i) that model A's holdout is more self-included than model B's *because* the fold trains on ten
months rather than twelve — the arithmetic is obvious but I did not isolate it by re-running model B on
a ten-month frame;
(ii) that a defect affecting 0.58 % of rows at weight 1 and the rest at ~1/460 cannot be worth seconds
— a plausibility argument, and the CV measurement in §4.2 is the only thing that speaks to it directly;
(iii) that the round count is the *only* channel from this flag to a prediction — this is now asserted
by two tests rather than argued, but the tests run on synthetic data and the byte-identity assertion on
real data is inferred from the code path being identical;
(iv) that Step 5 should stay closed — this follows the plan's own stop condition, which is a
pre-registration and not a measurement.
