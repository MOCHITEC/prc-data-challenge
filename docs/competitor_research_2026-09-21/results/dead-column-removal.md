# P8-E — deleting the five dead columns: help, hurt, or nothing?

**Executed:** 2026-09-25, branch `claude/model-b-and-arrival-features`.
**Question:** the candidate is *pure deletion*. Remove `precip_mm`, `precip_mm_prev1h`, `temp_x_precip`,
`temp_x_precip_prev1h` and `runway_null` from the model-visible set; add nothing. It is the cheaper
alternative named at the end of `results/P8-B-wxcodes-repair.md` §6 and never measured.

**Marking:** `[S]` = measured here by a reproducible command, or read from a named file at a named line.
`[X]` = inference, never dressed up as a measurement. `[P]` = written down **before** any arm was fitted.

---

## 0. Pre-registration `[P]`

Written and saved before the first `prc.pipeline` call of this session. Nothing below this heading was
edited after a number existed; the later sections were appended.

### 0.1 What I expect

`[P]` **Nothing measurable, on the improving side of nothing.** Point estimate for base → drop between
−0.3 s and +0.1 s, full-data 95 % CI containing zero, `prc.compare.verdict` returning **REJECT** on its
"no difference on all rows" branch.

`[P]` The reasoning, and it is worth stating because it is *not* an argument that the deletion is free:

- The five columns carry zero information by construction, so the model can lose nothing by not seeing
  them. That bounds the *harm from lost signal* at exactly zero and nothing else.
- What is not bounded is the interaction with `feature_fraction: 0.8`. Each tree samples 80 % of the
  columns; a constant column occupies a slot in that sample and wastes it. Removing five constants changes
  which live columns co-occur in every tree of every seed of every fold. That is a real perturbation with
  no sign attached to it a priori, and it is the whole reason this needs a measurement rather than an
  argument.
- The repo's own arity evidence points very slightly the other way: adding four arbitrary columns measured
  +0.16 [−0.58, +1.06] (`P8-B-wxcodes-repair.md` §1), and `plans/EXECUTION-ORDER.md:68` puts the per-column
  noise floor at ±0.5 s. So the prior is "removing five dead columns is worth somewhere between −0.5 s and
  0, and the instrument cannot resolve it".
- `[P]` I expect the **drop** arm to differ from base by more than the 0.017–0.028 s run-to-run spread
  (`results/determinism-cost.md`) — the perturbation is real even if its expected value is zero — and by
  less than 0.5 s.

### 0.2 Failure condition — what makes this DO-NOT-SHIP `[P]`

Any one of these, and I report DO-NOT-SHIP:

1. **The deletion measurably hurts.** The base → drop full-data 95 % CI lies entirely above zero
   (`verdict()`'s "B is worse on all rows" branch).
2. **The deletion did not happen.** The `p8e_drop` record does not show `n_features` = 85 and
   `n_features_unmatched` = 70, or `tests/test_train.py::test_excluding_a_column_narrows_the_lightgbm_matrix`
   does not pass, or the coverage gate still reports the five `DEAD` failures under `p8e_drop`.
3. **The base arm does not reproduce `p8b_base`** to within the run-to-run spread. Same config bar the
   name, same code, same cache — if it drifts, the instrument moved and no number from this session is
   readable.

`[P]` **What is *not* a failure condition: `verdict()` returning REJECT.** This candidate is a correctness
change priced for safety, not a feature seeking adoption. REJECT on the "no difference" branch is the
**expected and acceptable** outcome and is compatible with SHIP. The verdict is being used here to detect
*harm*, which is the only thing that can block a deletion of provable constants.

`[P]` And the converse is pre-registered too: **if `verdict()` returns ADOPT for base → drop, I will not
report it as a win.** Five columns that are identically zero cannot carry −0.5 s of signal-by-removal; an
ADOPT would mean the `feature_fraction` reshuffle drew lucky on these two folds, and I would say so and
price it against the control rather than against zero.

### 0.3 The arms, and why the control is a *removal* `[P]`

Three arms, all `configs/main_gap.yaml` + `folds: [forward, jul_fwd]`, `n_seeds: 3`, `--skip-predict`,
differing **only** in `exclude_features`.

| arm | config | `exclude_features` | model A | model B |
|---|---|---|---|---|
| base | `configs/p8e_base.yaml` | — | 90 num + 10 cat | 75 num + 10 cat |
| **drop** (the candidate) | `configs/p8e_drop.yaml` | the five constants | **85** + 10 | **70** + 10 |
| control | `configs/p8e_ctrl.yaml` | five live columns | **85** + 10 | **70** + 10 |

`[P]` **The usual noise arm does not fit and was rejected on purpose.** The standard third arm *adds* a
signal-free column and prices arity upward. This candidate moves arity *downward*, so an additive control
measures a change this experiment never makes, and its answer — "four arbitrary columns cost +0.16 s" — is
already on file from P8-B and did not need re-buying.

`[P]` The control that earns its fit is a **five-live-column removal**, for one reason that decides it:
without it a null on the candidate is unreadable. A flat base → drop result could mean the deletion is free,
or it could mean this instrument cannot see a five-column removal at all. Only a removal arm that *does*
destroy information separates those. It is also the direction-matched analogue of P8-B's +0.16 [−0.58,
+1.06], so the two experiments' controls can be read against each other.

`[P]` **The five control columns were drawn at random and the draw is recorded here before any fit:**
`random.Random(20260925).sample(sorted(pool), 5)` over the **70 live columns visible to both lanes**, giving

```
arr_land_prev_5m     non-zero  91.441% of training rows
arr_land_prev_60m    non-zero  99.884%
lobt_null            non-zero   1.078%
rwy_headway_next_s   non-zero 100.000%
vis_km               non-zero  99.987%
```

`[P]` Pool choice: `predict_fold` hides 15 of the 90 numeric columns from the unmatched lane by prefix, and
all five dead columns are visible to **both** lanes, so drawing from the both-lane set keeps the control's
scope matched to the candidate's. Random rather than hand-picked, deliberately: selecting five
low-importance live columns would bias the control toward "removal is cheap", which is the exact
reassurance under test. **The draw is not re-rolled**, whatever it produces — an over-large control cost
still proves the instrument sees removals, which is its job.

`[P]` **Known limitation, stated in advance:** a random draw is a noisy estimate of "the cost of removing
five live columns" — n = 1 draw. It calibrates the instrument's *sensitivity*; it is not an estimate of a
population mean, and I will not quote it as one.

### 0.4 Traps this design closes `[P]`

`[P]` **A deletion is exactly the shape of change that can silently not happen.** If the drop arm still has
the columns in its matrix, every number looks plausible and means nothing. Four independent checks, none of
which is "the config says so":

1. `n_features` in `experiments/experiments.jsonl` (model A) and `n_features_unmatched` per fold (model B),
   both written by the run itself.
2. `tests/test_train.py::test_excluding_a_column_narrows_the_lightgbm_matrix` — reads
   `lgb.Dataset.num_feature()` at the LightGBM boundary inside a real `predict_fold` and requires the width
   to fall by exactly the number of columns excluded, with the categorical lane as a separate control.
3. `python -m prc.coverage --config configs/p8e_drop.yaml` must stop reporting the five `DEAD` failures,
   while `configs/p8e_ctrl.yaml` — same arity — must still report all five.
4. The three `exp_id` hashes must differ, and `p8e_base` must reproduce `p8b_base`'s RMSEs.

`[P]` **Cache:** `exclude_features` is a model-visibility field, not a frame field. All three arms read the
one `data/work` cache, whose `feat_meta.json` fingerprint (`use_weather: true`, `wx_codes: false`,
`feature_code d2c32675aa94af82`, `weather_code 6aaa29cdc6c4ceee`) is identical for all three and must print
`features: reusing the cache in data/work` on every run. The frames are byte-identical across arms by
construction, which is the strongest possible answer to the P8 fingerprint trap: there is nothing for the
fingerprint to differ on.

`[P]` **Not doing:** no submission, nothing written to `outputs/`, `--skip-predict` on every arm.

### 0.5 One correction to §0.4, made before any arm finished and recorded rather than edited away

`[S]` §0.4 check 1 names the wrong absolute number for the unmatched lane, and §0.3's table repeats it.
`run_cv` logs `n_features` on the **cached** frame, where the 9 label-derived encoding columns do not yet
exist; `predict_fold` counts on the frame **after** `attach_encodings`. So the real widths are 99 + 10 for
model A and 84 + 10 for model B, not 90 + 10 and 75 + 10, and the archived `p8b_base` record confirms it
(`n_features` 90, `n_features_unmatched` 84). The drop and control arms should therefore show
`n_features_unmatched` = **79**, not 70.

`[S]` The check is restated in the form that does not depend on my arithmetic: **base minus arm must be
exactly 5 in every logged width, in both lanes, for both `-5` arms.** §0 is left as written; this is the
correction, not a rewrite of it.

`[S]` The same slip refines, but does not change, the control draw. The pool was the **70 live both-lane
columns of the cached feature frame**, which excludes the 9 encoding columns because they are not in that
frame. The draw was made from exactly that pool and recorded before any fit; what §0.3 did not say is that
the encodings were never eligible. `[X]` That exclusion makes the control slightly *gentler* than a draw
over all 79 — `ref_p10` and its family are the highest-importance columns in this model — which matters
only in the direction of making the control an easier bar for the candidate to look good against. It is
stated here so a reader can discount it.

---

## 1. What the candidate is, mechanically — and why `src/` did not change

`[S]` **`git diff --stat src/` is empty. Nothing in `src/` was touched by this experiment.**

`exclude_features` has been a `TrainConfig` field since before this theme, and it is already honoured
everywhere the deletion needs it to be:

| where | line | what it does |
|---|---|---|
| `src/prc/train.py:312` | `visible_numeric_columns` | `hidden = ... | set(cfg.exclude_features)` — the column never enters `num_cols`, so never enters `_fit`'s stacked frame |
| `src/prc/train.py:660` | `predict_fold` | the categorical list is filtered by the same set — a separate code path |
| `src/prc/coverage.py:68` | `model_visible_columns` | the gate reads the same rule, so it sees the arm's real column set |
| `src/prc/pipeline.py:157` | recipe load | `exclude_features` is one of the four list-valued keys coerced to a tuple |

`[X]` That is the strongest possible form of this change: a candidate that needs no new code cannot ship a
new bug, and the three arms differ by nothing but a five-element list in a YAML file.

`[S]` **The frames are byte-identical across the three arms.** `exclude_features` is a model-visibility
field, not a frame field, so it is correctly absent from `cache_fingerprint` — all three arms are built from
one fingerprint (`use_weather: true`, `wx_codes: false`, `feature_code d2c32675aa94af82`, `weather_code
6aaa29cdc6c4ceee`, the same `raw_dir` and the same file list). There is nothing for the fingerprint to
differ on between arms.

`[S]` **That claim survived an attack on it mid-experiment, and §4 records how.** A sibling agent editing
`src/prc/features.py` and `src/prc/weather.py` in the shared working tree changed both hashes under my
running arms; `cache_fingerprint` caught it, printed
`cache in data/work was built from different inputs (feature_code, gust_repair, weather_code, wind_dir);
rebuilding`, and the contaminated arm was discarded rather than reported. The arms below were then re-run
from a **git worktree pinned to 95ec23e** with its own `PYTHONPATH`, its own `PRC_DATA_DIR` and its own
feature cache.

### 1.1 The no-op guard, because a deletion is the shape that can silently not happen

`[S]` New test, `tests/test_train.py::test_excluding_a_column_narrows_the_lightgbm_matrix` (the only file
this experiment modified). It monkeypatches `prc.train.lgb.train`, calls `train_set.construct()` and reads
**`lgb.Dataset.num_feature()`** inside a real `predict_fold`, requiring the width to fall by exactly the
number of columns excluded — at the LightGBM boundary, not at the config. Its control is the categorical
lane, which `predict_fold` filters in a different statement.

`[S]` **Mutation audit, two mutants, both killed:**

| mutant | result |
|---|---|
| `visible_numeric_columns` drops the `| set(cfg.exclude_features)` term | **killed** (`matrix width did not fall by 2`) |
| the categorical filter drops `and c not in set(cfg.exclude_features)` | **killed** (the categorical control assertion) |

`[S]` `src/prc/train.py` was restored byte-identical after each (`git diff --stat src/prc/train.py` empty).
`[X]` The mutant I cannot kill by test is one that excludes the *wrong five columns* — which is why the
list is asserted three further ways below rather than trusted from the YAML.

---

## 2. The coverage gate, before and after `[S]`

`.venv/bin/python -m prc.coverage --config configs/<arm>.yaml --no-strict`, on `data/work`
(2,085,047 training rows, 344,841 ranking rows):

| arm | columns | failures | report-only |
|---|---|---|---|
| `main_gap` (production today) | 100 = 90 num + 10 cat | **9** | 4 |
| `p8e_drop` (the candidate) | 95 = 85 + 10 | **4** | 4 |
| `p8e_ctrl` (arity-matched control) | 95 = 85 + 10 | **9** | 4 |

`[S]` **All five `DEAD` failures clear.** What remains under `p8e_drop` is the four `dewp_spread_c` /
`dewp_spread_c_prev1h` `NONZERO` drift cells at LEMD and LSZH in January 2026 — the ones P8-D, the plan and
kind-mango independently reproduced and agreed are real data drift, not a pipeline defect.

`[S]` **The control arm is the check that this is about *which* columns, not *how many*.** `p8e_ctrl`
removes exactly as many columns and the gate still reports all nine failures, including all five `DEAD`.
A gate that fell to 4 on both arms would have been measuring arity.

`[S]` **Note against the tasking's framing:** the deliverable was described as clearing *4* of 9. It clears
**5** of 9, because `runway_null` is one of the five identically-constant columns and the candidate deletes
it too. `[X]` That is the difference that matters operationally: P8-B's repair cleared only the four weather
ones and left `runway_null`, so the gate still could not be turned on. After this deletion there is no
`DEAD` failure left at all — the remaining four are a single, understood, non-weather finding.
