# Item 11 `linear-tree-model-b`, step (b) — result (2026-09-17)

Batch C, forward-fold run. Source item: `plan.md` §1 #11 (from M `linear-tree-modelB`; P 1; H §4).
Step (a) (file/config comparison of v4 vs HEAD) was already done by the source reports and is not repeated.
Step (c) (probe submission) is Batch D and was **not** run.
Script: `docs/investigation_2026-09-16/scripts/linear-tree-model-b.py` (thin wrapper over the shared Batch C
helper `tree-identity-saturation.py`; reads finished artefacts only, trains nothing).
Configs: `configs/inv_baseline_forward.yaml`, `configs/inv_it11_lineartree.yaml` (both new; no existing config
or `src/` file was modified).

Definitions follow plan.md §0 (`y`, `ms` = `mms` = `mvt_minus_sched`, routing definition of "unmatched",
sched-filled = |y − ms| ≤ 60 s, 4 h = 14,400 s, 12 h = 43,200 s).

## 1. Hypothesis (as stated in plan.md §1 #11)

"v4's 37 s LB advantage over v7/v8 is mostly `linear_tree` in model B extrapolating along mms on ~2 ranking
LIRF rows." The step-(b) procedure is: "forward-fold run with `use_linear_tree true` for model B (~5 min),
compare LIRF sched-filled pred/mms". M's open question, quoted in the plan: "whether linear_tree on model B
over-extrapolates on non-LIRF large-mms rows under current code (M 'Not checked')".

Sub-claims tested here:
(A) `use_linear_tree: true` can be applied to model B alone through a config key;
(B) with linear_tree, model B's predictions on LIRF rows track ms (pred/ms → 1) instead of saturating;
(C) it does not over-extrapolate on non-LIRF large-ms rows;
(D) the number of forward-fold unmatched predictions above 12 h and 4 h.
The LB attribution itself ("mostly linear_tree", "~2 ranking rows", "37 s") is not testable locally.

## 2. Finding first: there is no config path to linear_tree on model B alone

Read at HEAD (`src/prc/train.py`):

- `use_linear_tree` is a single `TrainConfig` field (line 80), and `lgb_params()` (lines 129–135) adds
  `linear_tree` / `linear_lambda` whenever it is true.
- `for_unmatched()` (lines 117–127, `replace(...)` at 119) returns `replace(self, params={... num_leaves, min_data_in_leaf ...},
  num_rounds=self.unmatched_num_rounds)` — it overrides **only** those three things, so `use_linear_tree`
  is inherited unchanged.
- `predict_fold` calls `_fit(..., cfg, ...)` for model A (line 333) and `_fit(..., cfg.for_unmatched(), ...)`
  for model B (lines 358–359); both end up calling `lgb_params()` on a config carrying the same flag.
- `pipeline.py` copies only keys that are `TrainConfig` dataclass fields, and there is no
  `unmatched_use_linear_tree` field, so no YAML key can separate the two models.

Resolved, printed from the config objects:

| | model A | model B |
|---|---|---|
| baseline | num_leaves 127, min_data 50, linear_tree **absent** | 127 / 50, linear_tree **absent** |
| `use_linear_tree: true` | 127 / 50, **linear_tree True, linear_lambda 1.0** | 127 / 50, **linear_tree True, linear_lambda 1.0** |

So the run below is the **both-models** variant, as the task instruction directs for this case. The measured
consequence is visible in the record: `rounds_matched` 939 → 231 and matched RMSE 201.6633 → 202.5132, i.e.
model A is a different model in the candidate run, which confounds every whole-fold number with a model-A
change. (This is also what M reports for v4-era code: `lgb_params()` added linear_tree and model B was fitted
through the same path, so v4's model A had linear_tree too.)

## 3. Procedure actually executed

1. Baseline: `configs/inv_baseline_forward.yaml` (= `configs/main.yaml` + `folds: [forward]` +
   `params: {num_threads: 2}`), run with
   `POLARS_MAX_THREADS=2 .venv/bin/python -m prc.pipeline --config <yaml> --skip-predict`.
2. Candidate: `configs/inv_it11_lineartree.yaml` = the same file plus one line. Run afterwards, alone.
3. `make compare A=inv_baseline_forward-67e76a5dfc B=inv_it11_lineartree-b950106ffa`.
4. The helper script for per-group RMSE, pred/ms quantiles, the tail counts and the per-row LIRF table.

Config diff (besides `name:`):

```
--- configs/inv_baseline_forward.yaml
+++ configs/inv_it11_lineartree.yaml
 n_seeds: 3
+use_linear_tree: true
```

`linear_lambda` is left at its default 1.0. Records in `experiments/experiments.jsonl`:
**`inv_baseline_forward-67e76a5dfc`** (git_sha 4bc9cb4, 1,186.5 s) and **`inv_it11_lineartree-b950106ffa`**
(git_sha 735dfaa, 419.4 s); OOF files `experiments/oof_inv_baseline_forward-67e76a5dfc.parquet` and
`experiments/oof_inv_it11_lineartree-b950106ffa.parquet` (328,009 non-null rows each = 202511 + 202512).
The differing git shas are other agents' commits during the runs; `git diff 3d7cfc7..735dfaa -- src/` is empty.
Forward fold = train Jan–Oct 2025, inner holdout October, evaluate Nov+Dec 2025. LightGBM 4.7.0.

## 4. Results

### 4.1 Forward-fold metrics (328,009 scored rows)

| quantity | baseline | linear_tree (both models) |
|---|---|---|
| `rounds_unmatched` | **45** | **70** |
| `rounds_matched` | 939 | **231** |
| RMSE total | **284.7038** | **231.1276** |
| RMSE clean | 224.5794 | 214.0295 |
| RMSE matched (n = 325,004) | 201.6633 | **202.5132** (+0.85) |
| RMSE unmatched (n = 3,005) | **2,109.32** | **1,181.29** (−928.03) |
| RMSE LIRF unmatched (n = 140) | 8,578.38 | 4,502.23 |
| RMSE non-LIRF unmatched ms ≥ 2 h (n = 596) | 2,132.29 | 1,208.95 |
| top-20 share of fold SSE | 0.4464 | 0.1600 |
| **unmatched predictions > 43,200 (12 h)** | **0** | **4** |
| **unmatched predictions > 14,400 (4 h)** | **17** | **9** |
| max unmatched prediction | 34,539.68 | 76,605.44 |
| median unmatched prediction | 1,079.51 | 1,038.81 |
| train seconds | 1,186.5 | 419.4 |

### 4.2 pred/ms on LIRF unmatched rows with ms ≥ 2 h

58 rows scored in the fold, 45 of them sched-filled. (M's 0.83 / [0.55, 1.10] reference is from the 3-fold
`oof_main-48c19bc830` sched-filled rows — a different row set.)

| row set | n | statistic | baseline | linear_tree |
|---|---|---|---|---|
| LIRF unmatched ms ≥ 2 h | 58 | median pred/ms | **0.7021** | **0.7988** |
| | | p10 | 0.4248 | 0.4577 |
| | | p90 | 0.8579 | 0.9560 |
| | | RMSE | 13,119.34 | 6,551.57 |
| | | max pred | 34,539.68 | 76,605.44 |
| …sched-filled subset | 45 | median pred/ms | **0.7893** | **0.8440** |
| | | p10 | 0.5827 | 0.6092 |
| | | p90 | 0.8779 | 0.9546 |
| | | RMSE | 4,984.98 | **2,915.44** |
| | | max pred | 32,399.49 | 44,749.57 |
| non-LIRF unmatched ms ≥ 2 h | 596 | median pred/ms | 0.1418 | 0.1398 |
| | | p10 / p90 | 0.0782 / 0.2122 | 0.0667 / 0.2266 |
| | | RMSE | 2,132.29 | 1,208.95 |
| | | max pred | 16,879.98 | **10,550.34** |

Ten largest-ms rows (item 10's column added for context — same fold, same rows):

| ym | y | ms | baseline | linear_tree | item 10 (min_data 5) |
|---|---|---|---|---|---|
| 202512 | 87,605 | 86,102 | 34,093.3 | **76,605.4** | 68,801.0 |
| 202511 | 87,598 | 60,599 | 34,539.7 | 57,896.1 | 64,488.7 |
| 202511 | 87,480 | 56,520 | 32,351.7 | 57,081.9 | 50,583.1 |
| 202512 | 47,767 | 47,763 | 32,399.5 | 42,518.6 | 48,923.0 |
| 202512 | 39,120 | 39,120 | 20,890.4 | 33,017.9 | 14,545.3 |
| 202512 | 38,633 | 38,636 | 29,713.7 | 44,749.6 | 30,395.5 |
| 202511 | 37,736 | 37,738 | 25,511.5 | 37,731.3 | 28,445.8 |
| 202511 | 29,282 | 29,281 | 24,317.9 | 27,829.4 | 26,936.9 |
| 202512 | 19,376 | 19,378 | 19,075.8 | 16,591.2 | 17,026.3 |
| 202512 | 14,471 | 14,465 | 10,065.3 | 8,789.4 | 9,156.9 |

The row `y = 37,736 / ms = 37,738` is predicted 37,731.3 by the linear-tree model — pred/ms = 0.99987.

### 4.3 `make compare` (verbatim)

```
.venv/bin/python -m prc.compare inv_baseline_forward-67e76a5dfc inv_it11_lineartree-b950106ffa
A = inv_baseline_forward-67e76a5dfc
B = inv_it11_lineartree-b950106ffa
rows = 328,009
top-20 share of B's squared error: 0.160

 dropped         A         B     diff                 CI95   P(B)
       0    284.70    231.13   -53.58    [ -96.86, -17.04]  1.000
      20    211.84    213.21    +1.37    [  +0.22,  +2.60]  0.014
     200    198.74    199.95    +1.22    [  +0.75,  +1.62]  0.000
    2000    177.66    178.77    +1.11    [  +0.88,  +1.33]  0.000

month-level bootstrap (2 months): -53.58 s [-55.42, -51.72]  P(B better) = 1.000

   matched: n=325,004  A=   201.7  B=   202.5
 unmatched: n=  3,005  A=  2109.3  B=  1181.3

VERDICT: REJECT - tail-only: -53.58 s on all rows, but +1.22 s [+0.75, +1.62] once the worst 200 rows are removed - the effect is those rows, and the ranking set has different ones
```

## 5. Verdict per sub-claim

| sub-claim | verdict | deciding observation |
|---|---|---|
| (A) a config key applies linear_tree to model B alone | **REFUTED — stated as a finding** | `use_linear_tree` is one shared field; `for_unmatched()` overrides only num_leaves / min_data_in_leaf / num_rounds; `lgb_params()` is called on both configs. Measured consequence: `rounds_matched` 939 → 231 and matched RMSE 201.6633 → 202.5132, so model A changed too. The run reported here is therefore the both-models variant |
| (B) with linear_tree model B tracks ms instead of saturating | **CONFIRMED in direction, not to pred/ms ≈ 1** | sched-filled median pred/ms 0.7893 → 0.8440, p90 0.8779 → 0.9546, their RMSE 4,984.98 → 2,915.44; the ceiling rises from 34,540 to 76,605 and 4 predictions exceed 12 h where the baseline had 0. The median is still 16 % below ms, so "≈ mms" holds for individual rows (one at 0.99987), not for the group |
| (C) no over-extrapolation on non-LIRF large-ms rows (M's "not checked") | **CONFIRMED on this fold** | non-LIRF unmatched ms ≥ 2 h (n = 596): median pred/ms 0.1418 → 0.1398, p90 0.2122 → 0.2266, max prediction **falls** 16,879.98 → 10,550.34, RMSE 2,132.29 → 1,208.95. All four >12 h predictions are LIRF rows |
| (D) forward-fold unmatched predictions > 43,200 / > 14,400 | **fact: 4 / 9** (baseline 0 / 17) | §4.1 |
| unmatched RMSE effect | **fact**: 2,109.32 → 1,181.29 (−928.03); total 284.70 → 231.13 | §4.1 |
| matched side unharmed | **REFUTED** | matched RMSE +0.85 s with 231 instead of 939 rounds; the whole-fold −53.58 s mixes a model-A regression with a model-B gain |
| does it pass the repo's decision rule | **no** | REJECT, tail-only: +1.37 s at K = 20, +1.22 s [+0.75, +1.62] at K = 200, +1.11 s at K = 2000 — the only Batch C run whose drop-2000 CI excludes zero on the worse side |
| "v4's 37 s LB advantage is mostly linear_tree on ~2 ranking rows" | **UNDETERMINED** | needs the ranking labels and a submission (step (c), Batch D) |

## 6. Not determinable locally

- **CV → LB transfer.** The −53.58 s sits on rows the adopt/reject rule removes by construction
  (`decision-rule-blind-to-tail.md`: only K = 0 and K = 200 are consulted, and the K = 200 drop set is exactly
  the rollover / LIRF sched-filled rows). Whether the ranking set contains rows of this kind, and how many, is
  unknown — `lirf-regime-tail-count.md` §5 records that the labels of the ~11–20 ranking LIRF rows do not exist
  locally; P supposes ≈ 86,400 (rollover), M supposes = mms with training p = 0.62.
- **Attribution of v4 → v8.** This run cannot separate linear_tree from the other v4-era differences, and it
  cannot reproduce v4 (v4-era code resolved model B to 127/50/1500; today's `submitted_v4.yaml` resolves to
  255/20/3000 — plan.md §4 #11). The 37 s gap is an LB quantity.
- **The model-A / model-B confound.** Isolating linear_tree on model B would need a code change
  (an `unmatched_use_linear_tree` field), which this investigation is not permitted to make.
- **The forward fold's known limits** (`fold-shape-vs-ranking.md` §3.2, §5): 10 training months, horizons 1–2,
  no year boundary, predicts Nov+Dec 2025 and never a January or July; its 58 LIRF ms ≥ 2 h rows are not the
  ranking set's 181, and its `rounds_unmatched` 45/70 are not production's 199–658.
- Whether `linear_lambda` ≠ 1.0, or linear_tree with a different leaf size, behaves differently: not swept.

## 7. Facts vs inference

**Facts (measured in this run or read from code/artefacts named above).** The code reading in §2 and the
resolved parameter table; the two exp_ids, their git shas and run times; `rounds_unmatched` 45 → 70,
`rounds_matched` 939 → 231; all RMSEs in §4.1 (total 284.7038 → 231.1276, matched 201.6633 → 202.5132,
unmatched 2,109.32 → 1,181.29, LIRF unmatched 8,578.38 → 4,502.23, non-LIRF ms ≥ 2 h 2,132.29 → 1,208.95);
tail counts 0 → 4 above 12 h and 17 → 9 above 4 h; max unmatched prediction 34,539.68 → 76,605.44; all pred/ms
quantiles in §4.2 and the per-row table; the `make compare` block quoted verbatim; that all four >12 h
predictions are LIRF rows; that `git diff 3d7cfc7..735dfaa -- src/` is empty.

**Inference (not established here).**
- That the forward fold's behaviour on 58 LIRF rows is the behaviour v4 showed on the ranking set: different
  rows, different code, unknown labels.
- That the unmatched gain is *caused by* linear extrapolation along ms rather than by the 25 extra model-B
  rounds or by the different early-stopping path; a single run confounds them.
- That model A's +0.85 s is caused by linear_tree rather than by stopping at 231 rounds; the two are the same
  change here.
- That any of this moves the leaderboard. No recommendation is made.
