# Item 23 `holdout-month-decides-rounds` — result (2026-09-17)

Batch C, two forward-fold runs. Source item: `plan.md` §1 #23 (from P 5 `holdout-six-rows-decide-rounds`;
M `early-stop-holdout-month`; H §9).
Script: `docs/investigation_2026-09-16/scripts/holdout-month-decides-rounds.py` (thin wrapper over the shared
Batch C helper `tree-identity-saturation.py`; reads finished artefacts only, trains nothing; its
`--holdout-counts` mode produces §4.1).
Configs: `configs/inv_baseline_forward.yaml`, `configs/inv_it23a_rounds600.yaml`,
`configs/inv_it23b_holdout2.yaml` (all new; no existing config or `src/` file was modified).

Definitions follow plan.md §0 (`y`, `ms` = `mms` = `mvt_minus_sched`, routing definition of "unmatched",
sched-filled = |y − ms| ≤ 60 s; 2 h = 7,200 s, 4 h = 14,400 s, 12 h = 43,200 s).

## 1. Hypothesis (as stated in plan.md §1 #23)

"Model B's round count comes from L2 early stopping on one fixed month (the latest training month) containing
only a handful of extreme rows, so the rounds — and the tail behaviour — are decided by those rows; the forward
fold (October holdout, 44–51 rounds) therefore evaluates a model B unlike production's (December holdout,
199–658 rounds)." Listed as [supposed]: "at lr 0.05 a 45-round model cannot reach leaf means of 40k+ from base
~1,400".

Sub-claims tested:
(A) the inner holdout is the latest calendar month of the fold's training side, taken per model;
(B) it contains only a handful of extreme rows, and the forward fold's October holdout has fewer than
    production's December one;
(C) `rounds_unmatched` changes when the holdout changes, i.e. the rounds are decided by that month;
(D) the round count changes the tail behaviour (predictions above 4 h / 12 h, pred/ms on LIRF) and the
    unmatched RMSE;
(E) model B's round count can be fixed from the config (the run (a) the plan proposes).

## 2. Finding first: model B's rounds cannot be fixed, and its early stopping cannot be switched off, from a config

Read at HEAD (`src/prc/train.py`):

- `_fit` (lines 252–300) runs early stopping whenever `0 < n_hold < train.height`, where
  `hold = _inner_holdout_mask(train, cfg.inner_holdout_months)` (lines 239–250). It then sets
  `rounds = int(min(cfg.num_rounds, max(10, round(best * train.height / max(1, train.height - n_hold)))))`
  — `num_rounds` is a **cap and a scaling target, never a fixed count**.
- `for_unmatched()` (lines 117–127) overrides only `num_leaves`, `min_data_in_leaf` and `num_rounds`
  (from `unmatched_num_rounds`). `inner_holdout_months` and `early_stopping` are inherited from the shared
  config, so **no YAML key switches early stopping off for model B alone**; `inner_holdout_months: 0` would
  disable it for model A as well (and make model A train the full 2,000 rounds).
- There is no `unmatched_inner_holdout_months` or `unmatched_early_stopping` field, and `pipeline.py` copies
  only existing dataclass fields.

Consequence for the plan's run (a): setting `unmatched_num_rounds: 600` cannot fix model B at 600 rounds while
early stopping is active and `best · scale ≈ 45 < 600`. Run (a) was executed anyway, as the empirical check of
exactly that statement; §4.2 shows it is a no-op. Run (b) (`inner_holdout_months: 2`) is therefore the only
config-reachable way to move model B's round count, and is the substantive run of this item.

## 3. Procedure actually executed

Three runs, one at a time, each
`POLARS_MAX_THREADS=2 .venv/bin/python -m prc.pipeline --config <yaml> --skip-predict`, then
`make compare A=<baseline> B=<run>` with `POLARS_MAX_THREADS=1`.

| run | config | diff from the baseline file (besides `name:`) | exp_id | train s |
|---|---|---|---|---|
| baseline | `configs/inv_baseline_forward.yaml` = `configs/main.yaml` + `folds: [forward]` + `params: {num_threads: 2}` | — | **`inv_baseline_forward-67e76a5dfc`** | 1,186.5 |
| (a) | `configs/inv_it23a_rounds600.yaml` | `-unmatched_num_rounds: 2000` / `+unmatched_num_rounds: 600` | **`inv_it23a_rounds600-d6b279968a`** | 1,077.3 |
| (b) | `configs/inv_it23b_holdout2.yaml` | `-inner_holdout_months: 1` / `+inner_holdout_months: 2` | **`inv_it23b_holdout2-f5a981083f`** | 1,142.3 |

Resolved parameters (model B = `for_unmatched()`): all three runs have model A 127 leaves / min_data 50 /
cap 2000, model B 127 / 50, `early_stopping: 100`, `n_seeds: 3`, `num_threads: 2`, no linear_tree. Model B's
round cap is 2000 / **600** / 2000 and `inner_holdout_months` is 1 / 1 / **2**.

Records are in `experiments/experiments.jsonl` (git shas 4bc9cb4 / 735dfaa / 735dfaa — other agents' commits
during the runs; `git diff 3d7cfc7..735dfaa -- src/` is empty, so all three are the same model code) with OOF
files `experiments/oof_inv_baseline_forward-67e76a5dfc.parquet`,
`experiments/oof_inv_it23a_rounds600-d6b279968a.parquet`, `experiments/oof_inv_it23b_holdout2-f5a981083f.parquet`
(328,009 non-null rows each = 202511 + 202512). Forward fold = train Jan–Oct 2025, holdout month for early
stopping = October, evaluate Nov+Dec 2025.

## 4. Results

### 4.1 What the inner holdout actually is (script `--holdout-counts`, no model involved)

`_inner_holdout_mask` is applied inside `_fit`, i.e. separately to model A's matched frame and model B's
unmatched frame, so the holdout is per model. Forward-fold training side (Jan–Oct 2025):

| model | `inner_holdout_months` | holdout months | train rows | holdout rows | y ≥ 2 h | y ≥ 4 h | y ≥ 12 h | max y | refit scaling |
|---|---|---|---|---|---|---|---|---|---|
| A (matched) | 1 | 2025-10 | 1,737,573 | 183,761 | 13 | 0 | 0 | 13,133 | 1.1183 |
| A (matched) | 2 | 2025-09 + 10 | 1,737,573 | 365,246 | 21 | 0 | 0 | 13,133 | 1.2662 |
| **B (unmatched)** | **1** | **2025-10** | **19,465** | **1,913** | **30** | **4** | **0** | **25,558** | **1.1090** |
| **B (unmatched)** | **2** | **2025-09 + 10** | **19,465** | **4,378** | **72** | **16** | **4** | **87,543** | **1.2902** |

M's figures for the forward fold's holdout are reproduced exactly: 4 rows ≥ 4 h, max 25,558.
"refit scaling" is the `train.height / (train.height − n_hold)` factor `_fit` multiplies `best_iteration` by.

### 4.2 Run (a): `unmatched_num_rounds: 600` is a no-op on this fold

| quantity | baseline | (a) 600 |
|---|---|---|
| `rounds_unmatched` | 45 | **45** |
| `rounds_matched` | 939 | 939 |
| RMSE total / matched / unmatched | 284.7038 / 201.6633 / 2,109.32 | **284.7038 / 201.6633 / 2,109.32** |
| unmatched predictions > 14,400 / > 43,200 | 17 / 0 | 17 / 0 |
| max unmatched prediction | 34,539.68 | 34,539.68 |
| LIRF unm ms ≥ 2 h pred/ms p10 / med / p90 | 0.4248 / 0.7021 / 0.8579 | 0.4248 / 0.7021 / 0.8579 |

Every statistic the helper computes is identical, and `make compare` reports `+0.00 [+0.00, +0.00]` at all four
drop levels, i.e. the two OOF files agree row by row. (Side fact: this also demonstrates run-to-run determinism
of the pipeline at `num_threads: 2` — two separate processes, same predictions.)

### 4.3 Run (b): `inner_holdout_months: 2`

| quantity | baseline (Oct) | (b) Sep+Oct | change |
|---|---|---|---|
| **`rounds_unmatched`** | **45** | **413** | **×9.2** |
| `rounds_matched` | 939 | 1,064 | +125 |
| RMSE total | 284.7038 | **266.9151** | −17.79 |
| RMSE clean | 224.5794 | 228.0271 | +3.45 |
| RMSE matched (n = 325,004) | 201.6633 | 201.5813 | −0.08 |
| **RMSE unmatched (n = 3,005)** | **2,109.32** | **1,838.94** | **−270.38** |
| RMSE LIRF unmatched (n = 140) | 8,578.38 | 6,877.28 | −1,701.10 |
| RMSE non-LIRF unmatched ms ≥ 2 h (n = 596) | 2,132.29 | 2,235.14 | **+102.85** |
| top-20 share of fold SSE | 0.4464 | 0.3540 | −0.0924 |
| unmatched predictions > 14,400 | 17 | 16 | −1 |
| unmatched predictions > 43,200 | 0 | **3** | +3 |
| max unmatched prediction | 34,539.68 | 50,728.42 | +16,189 |
| median unmatched prediction | 1,079.51 | 1,037.25 | −42.3 |

All three predictions above 12 h are LIRF unmatched rows (ms 56,520 / 86,102 / 38,636; predictions 50,728 /
43,343 / 46,513); the baseline has none.

pred/ms on LIRF unmatched rows with ms ≥ 2 h (58 rows; 45 sched-filled):

| row set | statistic | baseline | (b) |
|---|---|---|---|
| all 58 | p10 / median / p90 | 0.4248 / **0.7021** / 0.8579 | 0.4324 / **0.8066** / **1.1207** |
| | RMSE | 13,119.34 | 10,392.71 |
| sched-filled 45 | p10 / median / p90 | 0.5827 / **0.7893** / 0.8779 | 0.5223 / **0.8649** / **1.1413** |
| | RMSE | 4,984.98 | 4,021.58 |
| non-LIRF ms ≥ 2 h (596) | p10 / median / p90 | 0.0782 / 0.1418 / 0.2122 | 0.0583 / 0.1458 / **0.2515** |
| | max pred | 16,879.98 | **24,035.08** |

### 4.4 `make compare` (verbatim)

Run (a):

```
.venv/bin/python -m prc.compare inv_baseline_forward-67e76a5dfc inv_it23a_rounds600-d6b279968a
A = inv_baseline_forward-67e76a5dfc
B = inv_it23a_rounds600-d6b279968a
rows = 328,009
top-20 share of B's squared error: 0.446

 dropped         A         B     diff                 CI95   P(B)
       0    284.70    284.70    +0.00    [  +0.00,  +0.00]  0.000
      20    211.84    211.84    +0.00    [  +0.00,  +0.00]  0.000
     200    198.39    198.39    +0.00    [  +0.00,  +0.00]  0.000
    2000    177.23    177.23    +0.00    [  +0.00,  +0.00]  0.000

month-level bootstrap (2 months): +0.00 s [+0.00, +0.00]  P(B better) = 0.000

   matched: n=325,004  A=   201.7  B=   201.7
 unmatched: n=  3,005  A=  2109.3  B=  2109.3

VERDICT: REJECT - no difference on all rows: +0.00 s [+0.00, +0.00]
```

Run (b):

```
.venv/bin/python -m prc.compare inv_baseline_forward-67e76a5dfc inv_it23b_holdout2-f5a981083f
A = inv_baseline_forward-67e76a5dfc
B = inv_it23b_holdout2-f5a981083f
rows = 328,009
top-20 share of B's squared error: 0.354

 dropped         A         B     diff                 CI95   P(B)
       0    284.70    266.92   -17.79    [ -37.07,  +2.03]  0.942
      20    212.42    215.17    +2.75    [  +1.14,  +4.32]  0.000
     200    199.00    199.66    +0.66    [  +0.31,  +0.99]  0.000
    2000    177.68    177.95    +0.27    [  +0.17,  +0.35]  0.000

month-level bootstrap (2 months): -17.79 s [-20.41, -15.13]  P(B better) = 1.000

   matched: n=325,004  A=   201.7  B=   201.6
 unmatched: n=  3,005  A=  2109.3  B=  1838.9

VERDICT: REJECT - no difference on all rows: -17.79 s [-37.07, +2.03]
```

## 5. Verdict per sub-claim

| sub-claim | verdict | deciding observation |
|---|---|---|
| (A) the holdout is the latest calendar month of the fold's training side, per model | **CONFIRMED** | `_inner_holdout_mask` (train.py 239–250) takes `uniq[-months:]`; `_fit` is called separately for the matched and unmatched frames, so model B's holdout is October's 1,913 unmatched rows while model A's is October's 183,761 matched rows |
| (B) the holdout contains a handful of extreme rows; October has fewer than December | **CONFIRMED** | October (model B): 4 rows ≥ 4 h, 0 ≥ 12 h, max y 25,558 — M's figures reproduced exactly. P's December figures (1,649 rows, 6 ≥ 4 h, max 87,605) are not re-measured here but are the same statistic on another month; the December row with y = 87,605 is in this fold's **evaluation** set (§ item 10's per-row table) |
| (C) `rounds_unmatched` is decided by that month | **CONFIRMED** | widening the holdout by one month (same data, same code, same seeds) moves it 45 → **413**, a factor 9.2; the refit scaling factor only accounts for 1.109 → 1.290, so `best_iteration` itself moved from ≈ 41 to ≈ 320 |
| (D) the round count changes the tail behaviour and the unmatched RMSE | **CONFIRMED** | unmatched RMSE 2,109.32 → 1,838.94 (−270.38); predictions above 12 h 0 → 3; max prediction 34,540 → 50,728; LIRF sched-filled median pred/ms 0.7893 → 0.8649 with p90 crossing 1 (0.8779 → 1.1413) |
| (D′) the change is confined to the tail / to LIRF | **REFUTED** | non-LIRF unmatched ms ≥ 2 h RMSE gets **worse** (2,132.29 → 2,235.14, max prediction 16,880 → 24,035), RMSE clean gets worse (224.58 → 228.03), and at K = 20 the whole fold is +2.75 s [+1.14, +4.32] |
| (E) model B's rounds can be fixed (or its early stopping disabled) from a config | **REFUTED — stated as a finding** | `unmatched_num_rounds` is a cap inside `min(...)`; `inner_holdout_months` / `early_stopping` are shared with model A. Empirically, `unmatched_num_rounds: 600` gives a bit-identical run (§4.2) |
| "[supposed] a 45-round model at lr 0.05 cannot reach 40k+" | **CONFIRMED in direction, not as a mechanism** | at 45 rounds the largest unmatched prediction is 34,539.68 and nothing exceeds 43,200; at 413 rounds three predictions exceed 43,200 and the maximum is 50,728.42. Rounds and holdout composition moved together in this run, so "the round count alone" is not isolated |
| "forward evaluates a model B unlike production's" | **CONFIRMED for the round count; UNDETERMINED for the consequence** | 45 here vs 199–658 reported for production; whether that changes the ranking of configurations on the LB is not measurable locally |
| does either run pass the repo's decision rule | **no** | (a) "no difference" (identical); (b) REJECT, full CI [−37.07, +2.03] spans 0, and every drop level ≥ 20 is worse |

## 6. Not determinable locally

- **CV → LB transfer.** Neither run touches the leaderboard. The −17.79 s of run (b) is not significant even on
  the full-data CI, and turns positive (worse) at every drop level ≥ 20;
  `decision-rule-blind-to-tail.md` shows the K = 200 drop set is exactly the rollover / LIRF sched-filled rows
  that move here, so the rule cannot see this change either way.
- **Production's round count and holdout.** The December-holdout figures (199–658 rounds, 1,649 rows, 6 ≥ 4 h)
  are quoted from P / M / H; reproducing them needs a full 12-month fit (~20 min, not a forward fold) and was
  out of scope for this Batch C item.
- **The forward fold's known limits** (`fold-shape-vs-ranking.md` §3.2, §5): 10 training months, horizons 1–2
  months, no year boundary, evaluation months Nov+Dec 2025, and `overall` in a 3-fold run is a different
  mixture again. Whether a model B with 413 rounds generalises to Jan/Jul 2026 cannot be answered here; the
  ranking labels do not exist locally (`lirf-regime-tail-count.md` §5).
- **Separating "more rounds" from "a holdout that contains 4 rows ≥ 12 h".** Run (b) changes both at once, and
  §2 shows the config cannot change one without the other.
- Whether 3 months, or a random (non-calendar) inner holdout, behaves differently: not run.

## 7. Facts vs inference

**Facts (measured in this run or read from code/artefacts named above).** The code reading in §2 (the `min`
expression at train.py 293–296, the scaling factor, the inherited fields); every number in §4.1 (holdout row
counts, extreme-row counts, max y, scaling factors); every number in §4.2 and §4.3, including
`rounds_unmatched` 45 / 45 / 413, `rounds_matched` 939 / 939 / 1,064, forward RMSE total 284.7038 / 284.7038 /
266.9151, matched 201.6633 / 201.6633 / 201.5813, unmatched 2,109.32 / 2,109.32 / 1,838.94, the tail counts
(0 / 0 / 3 above 12 h; 17 / 17 / 16 above 4 h) and the pred/ms quantiles; both `make compare` blocks quoted
verbatim; that runs (baseline) and (a) produce identical OOF values; that `git diff 3d7cfc7..735dfaa -- src/`
is empty.

**Inference (not established here).**
- That the September rows are what moved `best_iteration` (the holdout gained 2,465 rows, of which 12 more are
  ≥ 4 h and 4 are ≥ 12 h, including one at y = 87,543); no per-row attribution of the stopping curve was made.
- That production's 199–658 rounds are caused by December's composition rather than by the extra training
  months or the different fold geometry: this item measured only the forward fold.
- That a 413-round model B would score better on the leaderboard. The forward-fold gain is entirely in the
  unmatched tail, is accompanied by a non-LIRF regression, and does not survive the removal of 20 rows.
  No recommendation is made.
