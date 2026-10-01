# P1 — the ranking refit is not reproducible from its config

Investigation, 2026-09-23. Nothing under `src/`, `configs/`, `outputs/`, `experiments/` or `data/`
was modified. No submission was built or sent. The scratch scripts behind every measurement are
listed in the appendix.

Marking: **[S]** = read from a file in this repository, or measured here by a named script;
**[X]** = my inference or arithmetic on top of [S] facts. Every [S] carries a `file:line`, an
`exp_id`, or the script that produced it.

---

## 0. The finding

**The design matrix is not reproducible.** `_matrix` encodes each categorical column with
`cast(pl.Categorical).to_physical()` (`src/prc/train.py:288-297`). On a multi-chunk column — which
every frame read from `data/work/feat_train.parquet` is, 17 chunks — polars 1.44.2 assigns those
integer codes **nondeterministically**: two casts of the same column, in the same process, return
**different permutations of the same label set**. Measured, on `AIRCRAFT_OPERATOR_flt`: a pure
bijection on all 677 categories, differing on 2,020,691 of 2,085,047 rows [S-new].

LightGBM is told those columns are categorical, so the integer labels reach the split search. A
different permutation gives a different model. Proved directly: the same matrix, the same params, the
same seed, with only the categorical codes relabelled, gives a different tree ensemble, a different
inner-holdout loss (41620.774 → 41561.491) and a different early-stopping round [S-new].

That round count then feeds `rounds`, which sizes the full-data refit, and the refit is what the
submission is. Hence two runs of the same config: `rounds_matched` 1252 against 1175, and two
submission files differing on 327,635 of 344,841 rows.

| | |
|---|---|
| The two output files differ on 327,635 of 344,841 rows, RMS 20.762 s, max 451 s — and **not on one of the 5,290 unmatched rows** | **[S]** §1 |
| The defect is `rounds_matched` 1252 → 1175, and nothing else logged differs | **[S]** §1.2 |
| Data, feature cache, library versions and model A's code path are byte-identical between the two runs | **[S]** §2 |
| **Root cause: `cast(pl.Categorical).to_physical()` on a chunked column is nondeterministic in polars 1.44.2** — a fresh relabelling every call, in the same process | **[S-new]** §3.5 |
| **Proof: relabelling alone, on a fixed matrix, changes the trees, the holdout loss and the round count** | **[S-new]** §3.6 |
| Given a *fixed* matrix the fit is bit-reproducible — 3 in-process and 2 cold-process repeats, identical `model_to_string()` SHA | **[S-new]** §3.4 |
| The amplifier: the early-stopping curve is a **plateau 270 rounds wide** worth ≤ 0.1 s of holdout RMSE, so a small perturbation moves the argmin a long way | **[S-new]** §3.2 |
| `force_row_wise` (the 2026-09-15 fix) was aimed at a different channel and did not close this; the pair that was supposed to prove it, `det_a`/`det_b`, in fact disagrees | **[S]** §3.7 |
| Cost: a rotation, not a step. Range across the paired re-runs we have: **0.00, 0.03, 0.30 s** | **[S]**/**[X]** §4 |
| CV is **not** insulated: `prc.compare` on two runs of the same recipe reports −0.27 s [−0.54, −0.07], P(B better) = 0.998, and misses ADOPT by 0.01 s | **[S-new]** §5 |
| **The fix is one line and free**: `rechunk()` before the cast, or `pl.Enum(sorted(uniques))`. Both measured deterministic | **[S-new]** §6.1 |

---

## 1. What actually differs between the two files

### 1.1 The right pair to difference

`outputs/main_reparam.parquet` has the LOBT bound applied and `outputs/submit_v10_reparam.parquet`
does not, so differencing those two mixes the bound in. The bound's own effect is logged:
`lobt_bound_rows: 52`, `lobt_bound_max_move: 2989.28` ([S] `experiments/pipeline.jsonl`). The
bound-vs-bound pair is `outputs/v13_single_config_bound.parquet` (= the v10 run's predictions with the
same bound, 47 rows moved) against `outputs/main_reparam.parquet`:

| pair | rows differing | RMS | max |
|---|---|---|---|
| `submit_v10_reparam` → `main_reparam` | 327,677 | 21.956 | 2881.0 |
| `submit_v10_reparam` → `v13_single_config_bound` (the bound alone) | 47 | 7.137 | 2881.0 |
| **`v13_single_config_bound` → `main_reparam`** (the defect alone) | **327,635** | **20.762** | **451.0** |

**[S-new]**. The third row reproduces the brief's numbers exactly, so that is the pair used from here.

### 1.2 It is model A, and only model A

Split by the routing key (`mvt_minus_aobt3` null → model B), from `data/work/feat_rank.parquet`:

| lane | n | rows differing | RMS | max | mean |
|---|---|---|---|---|---|
| matched (model A) | 339,551 | **327,635** | 20.923 | 451.0 | +0.998 |
| unmatched (model B) | 5,290 | **0** | 0.000 | 0.0 | 0.000 |

**[S-new]**. Model B's predictions are bit-identical — which is what `rounds_unmatched: 38` in both
records implied ([S] `experiments/pipeline.jsonl`), and what P1 stage 0 meant by *"forward unmatched
1028.41 against the recorded 1028.41 — an exact reproduction"* ([S] commit `80e6270`). That criterion
was true, and it happened to be checked on the one lane that is not affected.

Model A's only logged degree of freedom moved: `rounds_matched` **1252** against **1175**. Everything
else in `predict_info` is identical: `train_rows` 2,085,047, `rounds_unmatched` 38,
`sched_reparam_train_rows` 1,488, `sched_reparam_target_rows` 383, `parity_max_gap`
0.016682156222184143 [S].

### 1.3 The difference is spread, not a tail

**[S-new]**, over all 344,841 rows of the bound-vs-bound pair:

* 95.0 % of rows differ; median |Δ| **8 s**, p90 28, p95 40, p99 80, p99.9 160, max 451.
* the 100 largest disagreements carry **4.6 %** of the squared difference; the largest 1,000 carry
  18.6 %; the largest 10,000 are needed to reach half of it.
* per departure airport, RMS runs **16.1 s (LEBL) to 29.4 s (LIRF)** — every airport, not one.

**[X]** That is the signature of a different model, not of a handful of tail rows moving.

---

## 2. What it is not

| candidate | verdict | evidence |
|---|---|---|
| `lobt_bound: true` | **ruled out** | applied in `_assemble` *after* the fit (`src/prc/train.py:490,498`); moves 52 rows [S] |
| `folds: [forward]` | **ruled out** | consumed only by `run_cv` (`src/prc/train.py:564-566`); `predict_ranking` is a separate call from `pipeline.main` (`src/prc/pipeline.py:148,163`) |
| the feature cache / the data | **ruled out** | `data/work/feat_train.parquet` mtime **2026-09-15 11:38:20**, before both runs (2026-09-17 06:06, 2026-09-23 10:29); `src/prc/features.py` last changed 2026-09-15 11:37:52 (`5794f24`); `data/raw/*.parquet` all 2026-09-10 [S] |
| library / interpreter drift | **ruled out** | `lightgbm 4.7.0`, `polars 1.44.2`, `numpy 2.4.6`, `python 3.11.15`; `.venv/.../site-packages` entries last touched 2026-09-10 and 2026-09-14; the lockfile was pinned by `c61b727` (2026-09-17 04:36), *before* the v10 run [S] |
| model A's code path changed between the runs | **ruled out** | `git diff bbf788e bdb1033 -- src/` touches only `src/prc/train.py`, adding `LOBT_OFFSET`, `_lobt_bound`, `_lane_metrics`, the fold sidecar and one `info` key. `_fit`, `_inner_holdout_mask`, the params dict, `visible_numeric_columns` and `cols_a` are untouched [S] |
| `cfg.hash()` reaching a seed | **ruled out** | its only use in `src/` is `exp_id` (`src/prc/train.py:630`) [S] |
| the seed loop / `n_seeds` | **ruled out as cause, and as fix** | `rounds` is computed once at `src/prc/train.py:360`, *outside* the seed loop at `:366-368`; every seed trains to the same round count. See §6.5 |
| the inner-holdout construction | **ruled out** | `_inner_holdout_mask` (`src/prc/train.py:300-314`) is the last calendar month of the training side; 164,028 of 2,062,577 matched rows, identical on every replay [S-new] |
| thread scheduling under `deterministic` + `force_row_wise` | **real, but not this** | the trees are bit-identical across repeats; only LightGBM's *validation metric* is thread-order dependent, at ~1 ULP. §3.3 |
| **polars categorical codes** | **THIS IS IT** | I ruled it out early on a 3-element frame, where the cast is single-chunk and deterministic, and **that was wrong**. At scale, on a chunked column, it is the cause. §3.5 |

---

## 3. The mechanism

### 3.1 Where the round count comes from

`src/prc/train.py:347-360`:

```
booster = lgb.train(params, dtr, num_boost_round=cfg.num_rounds, valid_sets=[dva],
                    callbacks=[lgb.log_evaluation(0), lgb.early_stopping(cfg.early_stopping, verbose=False)])
best = int(booster.best_iteration or cfg.num_rounds)
rounds = int(min(cfg.num_rounds, max(10, round(best * train.height / max(1, train.height - n_hold)))))
```

2,062,577 matched training rows, 164,028-row inner holdout, so the scale factor is
2062577/1898549 = **1.08639** [S-new]: `rounds_matched` 1252 and 1175 correspond to `best_iteration`
**1152** and **1082** [X]. `lgb.early_stopping` runs with its default `min_delta = 0.0` and a strict
comparison, `curr_score < best_score - delta`
([S] `.venv/lib/python3.11/site-packages/lightgbm/callback.py:285,310`), so the selected round is the
exact argmin of a 1,000-plus-point curve with no tolerance.

### 3.2 The amplifier: that curve is a plateau 270 rounds wide

**[S-new]** (`/tmp/nd/long.py`, full-data model-A early-stopping fit, history saved and analysed):

| rounds within X (relative) of the minimum | count | span |
|---|---|---|
| 1e−6 | 2 | 937–938 |
| 1e−5 | 4 | 937–943 |
| 1e−4 | 17 | 929–945 |
| **1e−3** | **260** | **770–1037** |

In seconds of inner-holdout RMSE the whole 770–1037 band costs at most **0.29 s** and most of it under
**0.10 s**: round 800 is +0.098 off the best, round 850 +0.019, round 1000 +0.072.

**[X]** So any perturbation that reshuffles the curve slightly moves the argmin a long way in *rounds*
while costing almost nothing in *holdout loss* — and the refit built from either end of that band
differs by RMS ~20 s on 95 % of rows (§1). A tolerance-free argmin over a curve with no curvature is
the amplifier; §3.5 is the thing being amplified.

### 3.3 A red herring, recorded because it is real

`params` sets `deterministic: True` and `force_row_wise: True` (`src/prc/train.py:77-78`), the
`70ef994` fix. Measured (`/tmp/nd/trees.py`, three repeats of the full-data fit at 40 rounds, one
process) [S-new]:

* `model_to_string()` SHA: **identical in all three** — the learner is bit-reproducible.
* predictions on 20,000 holdout rows: **identical in all three**.
* the recorded `valid_0/l2` history: **three different SHAs**, 1–2 ULP apart (max relative difference
  3.31e−16, first divergence at round 5 of 60).

At 1 thread the history is bit-identical across three repeats (`/tmp/nd/long.py` stage B: all three
`48350.302807003594`); at 4 threads it is not.

**[X]** So LightGBM's *validation-metric reduction* — the sum of 164,028 squared errors across OpenMP
threads — is order-nondeterministic, and `deterministic` does not cover it. Worth knowing, **but not
this defect**: 7.3e−12 of noise cannot flip an argmin whose smallest competing step is 0.00196, and in
three repeats it did not move `best_iteration` at all.

### 3.4 Given a fixed matrix, the fit is bit-reproducible

`/tmp/nd/prep.py` builds model A's design matrix once through the repository's own `fit_encodings`,
`attach_encodings`, `visible_numeric_columns`, `_matrix` and `_inner_holdout_mask`, and caches it:
2,062,577 × 108 float32, inner holdout 164,028 [S-new]. Refitting from that cached matrix:

| repeats | `best_iteration` | `model_to_string()` SHA |
|---|---|---|
| 3 in one process (`long.py` stage A) | **937, 937, 937** | — |
| 2 in separate cold processes (`one.py`) | **937, 937** | **identical** (`7aff0b21…`) |

So the fit is not the problem. The input is.

### 3.5 The matrix is not reproducible — `cast(pl.Categorical).to_physical()`

`/tmp/nd/mat.py` builds the model-A design matrix **twice in one process** through the real code path
and hashes it column by column [S-new]:

```
[build1] X (2402128, 108) sha=305f06ef2da6cef5  sum=63320895211.512390
[build2] X (2402128, 108) sha=cc5b027daea67a5a  sum=63307879095.512390
columns differing: ['ADEP_mvt','RUNWAY_mvt','STAND_mvt','AIRCRAFT_TYPE_mvt',
                    'AIRCRAFT_OPERATOR_flt','FLIGHT_TYPE_flt','ADES_mvt','ADES_REGION']
```

**Every differing column is a categorical; every numeric column is identical.** Narrowed to the cast
itself (`/tmp/nd/cat2.py`, `data/work/feat_train.parquet`, 2,085,047 rows in **17 chunks**) [S-new]:

| column | distinct strings | two casts equal? | nature of the difference | fixed by `rechunk()` | fixed by `pl.Enum(sorted)` |
|---|---|---|---|---|---|
| `ADEP_mvt` | 10 | **no** | pure bijection on all 10 codes; 1,223,407 of 2,085,047 rows differ | **yes** | **yes** |
| `AIRCRAFT_OPERATOR_flt` | 677 | **no** | pure bijection on all 677 codes; 2,020,691 of 2,085,047 rows differ | **yes** | **yes** |

So polars 1.44.2 builds the category dictionary for a chunked column in a way that depends on chunk
processing order, and `to_physical()` hands out a fresh permutation each time. No values are lost —
it is a relabelling — but the labels are what `_matrix` writes into the float32 matrix and what
`lgb.Dataset(..., categorical_feature=cat_idx)` is told to treat as categories.

Both lanes are exposed: `/tmp/nd/cat3.py` finds non-reproducible codes on all nine of model A's
categoricals and all six of model B's [S-new]. §5.2 says why only model A's output moves.

### 3.6 The proof: relabel and nothing else

`/tmp/nd/perm.py` takes the *cached* matrix of §3.4 — bit-identical inputs, same params, same
`seed=42` — and applies a random permutation to each categorical column's codes, which is exactly the
freedom §3.5 shows polars exercising. [S-new]:

| codes | `best_iteration` | `rounds` | inner-holdout L2 at the best round | `model_to_string()` SHA |
|---|---|---|---|---|
| as built (5 repeats, §3.4) | 937 | 1018 | 41620.77406842256 | `7aff0b21c9422af2` |
| relabel seed 1 | **926** | 1006 | 41561.49118721816 | `ec7e036cc9196b96` |
| relabel seed 2 | **943** | 1024 | 41659.06353418459 | `55819e00bfe02df8` |
| relabel seed 3 | **1101** | **1196** | 41638.53915471405 | `95d7061edfe4f33b` |

Four distinct ensembles, four distinct losses, and `rounds_matched` spanning **1006 to 1196** — from
nothing but the integer names of the categories. Every other input, including `seed=42`, is byte-for-
byte the same. That 190-round spread is the same magnitude as the spread across the four real runs in
the table below, which is the confirmation that this is the whole story and not a contributing factor.

And the real code path, run cold, lands where the production runs land and not where the cached-matrix
runs land — `/tmp/nd/replay.py` drives the repository's own `predict_fold` on
`data/work/feat_{train,rank}.parquet` with `lgb.train` wrapped to stop after model A's early-stopping
fit, and reports `best_iteration=1064`, `es_train_rows=1898549` → `rounds_matched=1156` [S-new].
Four executions of the real path now exist:

| run | `best_iteration` | `rounds_matched` |
|---|---|---|
| `submit_v10_reparam` (2026-09-17, production) | 1152 [X] | **1252** [S] |
| `main_reparam` (2026-09-23, production) | 1082 [X] | **1175** [S] |
| `replay.py` (2026-09-23, cold process) | **1064** [S] | 1156 |
| cached-matrix repeats (§3.4) ×5 | 937 | 1018 |

**[X]** The first three are the same code on the same data and disagree; the last is stable precisely
because the matrix was built once and reused. That is the whole mechanism, end to end.

### 3.7 It has happened before, including under a control that was read too loosely

| pair | what differed | `rounds_matched` | `rounds_unmatched` |
|---|---|---|---|
| `det_a-cfc9417c52` → `det_b-fa2cba5913` | **nothing**: `git diff 70ef994 c5e9908 -- src/` is empty; configs identical but for `name` | jan 1126 → **1273**, jul 1146 → **1034**, forward 1131 → **1173** | 522 / 321 / 45, **identical** |
| `submit_v10_reparam-77f92c802d` → `main_reparam-bc60017878`, forward fold | `lobt_bound`, `folds` | 1055 → **940** | 64 → 64, **identical** |
| the two ranking refits | `lobt_bound`, `folds` | 1252 → **1175** | 38 → 38, **identical** |
| `inv_baseline_forward-67e76a5dfc` → `-9271caca22` | two different commits | 939 → 939 | 45 → 45 |

[S] `experiments/experiments.jsonl`, `experiments/pipeline.jsonl`, `git diff`.

`det_a`/`det_b` **already had `force_row_wise: True`** — both records log it — so the 2026-09-15 fix
did not close this, and the commit message claiming *"configs/det_a + det_b run the same settings twice
to prove it"* (`70ef994`) reports a pair that in fact disagrees on all three folds. The comment still
in `src/prc/train.py:70-76` — *"Two runs whose only difference was in model B's settings — which model
A is fitted before and cannot see — gave model A 1273 and 959 rounds on the same fold"* — described
this defect correctly and attributed it to the wrong cause; model B's settings were never the point,
the second run simply drew a different category permutation.

`inv_baseline_forward` reproducing exactly across two commits is not a counter-example: a relabelling
only moves the argmin when the plateau is flat enough to be tipped, and on that fold it evidently was
not.

---

## 4. What it costs

### 4.1 The naive price, and why it is wrong

The squared error carried by the difference vector is **1.4864e8** [S-new]. At RMSE 282.7371 with
N = 344,841 the exchange rate is 2·N·RMSE = **1.95e8** of SSE per second, so if the difference were
*orthogonal* to the error it would add **+0.76 s** to whichever build carried it [X].

It is not orthogonal. Write the two builds as `p_a = p̄ − d/2` and `p_b = p̄ + d/2` around their
midpoint, so their errors are `e_a = ē + d/2` and `e_b = ē − d/2`:

```
MSE_a = ( ‖ē‖² + ⟨ē,d⟩ + ‖d‖²/4 ) / N
MSE_b = ( ‖ē‖² − ⟨ē,d⟩ + ‖d‖²/4 ) / N
MSE_b − MSE_a = −2⟨ē,d⟩ / N
```

The `‖d‖²` term is carried by *both* builds and cancels — neither is systematically further from the
truth, which is what two draws of the same recipe are. **[X]** For a `d` with no preferred alignment
to `ē`, `⟨ē,d⟩` has standard deviation `‖ē‖·‖d‖/√N`, so `sd(ΔMSE) = 2·RMSE·RMS(d)/√N` and, dividing
by `dMSE/dRMSE = 2·RMSE`,

```
sd(ΔRMSE) ≈ RMS(d) / sqrt(N) = 20.762 / sqrt(344,841) = 0.035 s
```

which is, to within a constant, Blum & Hardt's `s/sqrt(n)` ladder bound slot 14 recommended
([S] `docs/competitor_research_2026-09-21/SUMMARY.md:112-115`).

### 4.2 Checked against the paired re-runs we have

| pair | rows | RMS(d) | predicted sd | **observed ΔRMSE** |
|---|---|---|---|---|
| `det_a` → `det_b`, pooled OOF | 672,428 | 25.93 | 0.032 | **+0.023** |
| `det_a` → `det_b`, matched lane | 664,050 | 26.09 | 0.032 | **+0.036** |
| v10 → main_reparam, forward fold, pooled | 328,009 | 17.66 | 0.031 | **−0.269** |
| v10 → main_reparam, forward fold, matched | 325,004 | 17.74 | 0.031 | **−0.301** |
| `inv_baseline_forward` ×2 | 328,009 | 0.00 | 0.000 | **0.000** |

[S-new] (OOF parquets joined to `data/work/feat_train.parquet`).

The isotropic model nails the `det` pair and **under-predicts the forward-fold pair tenfold**. The
reason is in the round counts: the forward fold went 1055 → 940 rounds and got **better** by 0.30 s,
so past ~940 that fit mildly overfits and the round count has a systematic component as well as a
rotational one. The `det` pair agrees in sign — more rounds was worse on jan (+147 rounds, +0.081 s)
and forward (+42, +0.007), fewer was better on jul (−112, −0.003) — at a much smaller slope [S].

### 4.3 The number to plan against

**[X]** For the ranking pair (1252 → 1175, 77 fewer rounds, favouring `main_reparam`): the rotational
floor is 0.035 s and the realised value on the one directly comparable pair — same recipe, same lane,
comparable N, comparable RMS(d) — was **0.30 s**. So:

> **re-running the same config moves the board by of order 0.0–0.3 s. The brief's "several tenths" is
> the top of that range, not its middle — but it is the right order, and it is above the 0.25 s at
> which this project adopts and rejects.**

The brief reached "several tenths" by scaling the v12 → v13 pair (RMS 64.69 ↔ 1.31 s) linearly. That
mis-reads it: v12 → v13 replaced v4's hand-assembled matched lane with the single-config one, a
*systematic* change where one build really is closer to the truth, so its 1.31 s is the cross term
`−2⟨ē,d⟩` a better model earns. A re-run has no such term by construction (§4.1). RMS(d) does not
convert to seconds at the same rate in the two cases.

Two repository claims need correcting:

* `docs/investigation_2026-09-16/hypotheses/model_b_tail.md:14` — *"det_a = det_b = repro_v2
  bit-identical"*. **False on the matched lane**: 36 of 664,050 rows agree [S-new]. True on the
  unmatched lane, which is what that file was about.
* `docs/competitor_research_2026-09-21/plans/P10-catboost-rival.md:424-425` — *"Run-to-run noise at
  fixed config is negligible: `det_a` 336.21 vs `det_b` 336.23 (0.02 s)"*. True of the **pooled full
  RMSE**, which the same round records as having a 27 s standard deviation and ~77 s resolution — the
  one statistic in the repository that cannot see this. On the matched lane the same pair is 26 s of
  RMS disagreement.

---

## 5. What it means for the measurement protocol

### 5.1 The decision rule, run on a null pair

`python -m prc.compare submit_v10_reparam-77f92c802d main_reparam-bc60017878` — the forward fold, run
twice from recipes differing only in keys the fold cannot see [S-new]:

```
 dropped         A         B     diff                 CI95   P(B)
       0    223.51    223.24    -0.27    [  -0.54,  -0.07]  0.998
      20    207.96    207.78    -0.18    [  -0.34,  -0.04]  0.996
     200    197.37    197.25    -0.12    [  -0.23,  +0.01]  0.960
    2000    177.05    176.89    -0.15    [  -0.24,  -0.06]  1.000

month-level bootstrap (2 months): -0.27 s [-0.46, -0.09]  P(B better) = 1.000
   matched: n=325,004  A=   201.6  B=   201.3
 unmatched: n=  3,005  A=  1028.4  B=  1028.4

VERDICT: REJECT - tail-only: -0.27 s on all rows, but -0.12 s [-0.23, +0.01] once the worst 200
rows are removed - the effect is those rows, and the ranking set has different ones
```

Nothing changed, and the rule reports a −0.27 s improvement with P(B better) = 0.998, a drop-2000 CI
excluding zero at P = 1.000, and a month bootstrap excluding zero at P = 1.000. It returns REJECT only
because the drop-200 CI's upper edge is **+0.01**. The project's null pair misses ADOPT by one
hundredth of a second — and the reason it prints ("the effect is those rows") is **wrong**: §1.3 shows
the difference is spread over 95 % of rows.

**[X]** At a screening resolution of 0.25 s ([S] `SUMMARY.md:109`), any A/B taken between two separate
pipeline runs is confounded until §6.1 lands.

### 5.2 Which comparisons are insulated

* **Not insulated: the matched lane, and anything pooled.** Each fold calls its own `predict_fold` →
  `_fit` → early stopping (`src/prc/train.py:580-583`), and each `_fit` rebuilds its own matrix
  through `_matrix`, so a four-fold run draws **five** independent category permutations (four folds
  plus the refit). Folds fitting their own models is not insulation — it is five chances instead of
  one.
* **Insulated in every pair observed: the unmatched lane.** `rounds_unmatched` was identical in all
  four comparable pairs (§3.7) and the predictions were bit-identical (`+0.00` with zero-width CIs at
  all ten airports, §5.1). This is **not** because model B escapes the relabelling — `/tmp/nd/cat3.py`
  shows its categoricals are relabelled too [S-new]. **[X]** The likely reason is that LightGBM orders
  categories by their gradient statistics when searching a categorical split, so the code permutation
  reaches the model only through tie-breaking and the `max_cat_threshold` cut — which bites on model
  A's long tail of rare stands and operators over 2.4 M rows and evidently not on model B's 27.8 k.
  That is an empirical regularity across four pairs, not a guarantee; it should not be relied on if
  model B's capacity grows.
* **Consequence for P1.** The unmatched-lane work this plan round is built on is measurable as things
  stand. The matched-lane and pooled numbers in the same records are not, at the resolution in use.

### 5.3 What this does *not* undermine

Both production runs agree on `train_rows`, `parity_max_gap`, `sched_reparam_target_rows` and
`rounds_unmatched`, and both P1 stage-0 criteria that were checked (unmatched forward 1028.41 exactly;
jan/jul reparam reproduction) stand [S]. v13 = 282.7371 stands too — a real 1.31 s against v12, an
order of magnitude above this noise. What is undermined is the *next* decision, if it is a few tenths
and is taken by differencing two runs.

---

## 6. The fix, ranked

### 6.1 Make the categorical encoding deterministic — *the root-cause fix, one line, free*

In `_matrix` (`src/prc/train.py:288-297`), either

```python
X = df.select(cols).rechunk()          # before the categorical casts
```

or, better, encode against an explicit sorted category list rather than order of appearance:

```python
cats = sorted(X[c].cast(pl.Utf8).fill_null("__NA__").unique().to_list())
... .cast(pl.Enum(cats)).to_physical() ...
```

* **Measured:** both make the codes bit-reproducible on the real 2,085,047-row, 17-chunk columns —
  `rechunk()` **yes**, `pl.Enum(sorted)` **yes**, for both `ADEP_mvt` (10 categories) and
  `AIRCRAFT_OPERATOR_flt` (677) [S-new] `/tmp/nd/cat2.py`.
* **Cost:** `rechunk()` copies the selected columns once — the frame is already being materialised to
  numpy on the next line, so this is a fraction of a matrix build, seconds. `pl.Enum` costs a
  `unique()` and a sort per categorical column, also seconds. Neither adds a fit.
* **What it changes about the model:** it fixes *which* permutation is used, so the next build will
  differ from both existing ones in exactly the way two current builds differ from each other — i.e.
  a fresh draw, once, and then never again. Prefer `pl.Enum(sorted(uniques))` over `rechunk()`: it is
  stable across polars versions, across chunk layouts, and across the train/rank concat order, whereas
  `rechunk()` only removes the one trigger observed today.
* **Real fix or workaround:** **a real fix.** It removes the nondeterminism at source rather than
  hiding it, and it is the only option here that makes `_matrix` a function of its inputs.
* **Caveat:** the fix must be applied *inside* `_matrix`, because `_matrix` is called once per `_fit`
  and the concat order (`train` then each predict frame) differs between folds. An `Enum` built from
  the stacked frame's own uniques is correct in every call.

### 6.2 Give early stopping a tolerance — *defence in depth, also free*

Pass a non-zero `min_delta` to `lgb.early_stopping` (`src/prc/train.py:356`). Replayed exactly on the
measured curve (`min_delta` is absolute in LightGBM, and a larger one can only stop earlier, so a
truncated history gives exact answers) [S-new]:

| `min_delta` | relative | selected round | refit rounds | holdout RMSE penalty |
|---|---|---|---|---|
| 0 (today) | — | 937 | 1018 | — |
| 1e−3 | 2.4e−8 | 937 | 1018 | +0.00000 |
| **0.1** | **2.4e−6** | **937** | **1018** | **+0.00000** |
| 1.0 | 2.4e−5 | 935 | 1016 | +0.00145 |
| 4.16 | 1e−4 | 932 | 1013 | +0.00431 |
| 41.6 | 1e−3 | 838 | 910 | +0.03628 |

`min_delta = 0.1` changes **nothing** about the selected model on this curve while sitting 1.4e10 ULPs
above the floating-point noise of §3.3 and ~50× above the smallest round-to-round step on the plateau.
It closes §3.3 outright and blunts anything else that perturbs the curve at that scale. It does **not**
close §3.5 — a relabelling moves the loss by 59 units (§3.6), which no sane tolerance absorbs — so it
is a complement to 6.1, never a substitute.

### 6.3 Pin the round counts in the config for a production build — *do this too*

`TrainConfig` has **no field that pins a round count** (`src/prc/train.py:55-145`), so
`configs/main_reparam.yaml` cannot today describe the model it produced, which is the repository's own
rule. Add `rounds_matched` / `rounds_unmatched` overrides honoured by `_fit` in place of the
early-stopping fit.

* **Cost:** negative. It removes the early-stopping fit, ~1,250 rounds on 1.9 M rows at a measured
  0.19 s/round ⇒ ~**4 minutes** saved per run for model A [S-new].
* **What it changes:** nothing, given the same number — *provided 6.1 is in first*. Without 6.1 the
  pinned round count is applied to a differently-labelled matrix each run, so the build is still not
  reproducible. **Order matters: 6.1, then 6.3.**
* Whichever number is pinned, say which: 1252 is the v13 build (282.7371 on the board), 1175 is
  `main_reparam`'s, and §4.3 puts them within ~0.3 s of each other in an unknown direction.

### 6.4 `num_threads: 1` for the refit — *reject*

* **Cost:** measured **3.05×** wall clock (60 rounds: 64 s at 1 thread against 21 s at 4) [S-new];
  the `main_reparam` run was 3,587 s, so ~3 hours.
* **What it changes:** it removes the metric-reduction nondeterminism of §3.3 (`distinct_hists` 3 → 1)
  and **nothing else** — the polars cast of §3.5 is untouched by LightGBM's thread count. It also
  changes the model, because 1-thread and 4-thread summation orders differ (holdout loss at round 60
  `48350.302807003594` against `48350.30280700396…`, ~50 ULPs apart), so every number in the
  repository, all produced at 4 threads, would need re-baselining.
* **Verdict:** three hours of wall clock aimed at the wrong channel. Reject.

### 6.5 Seed-average more — *orthogonal to this defect*

`rounds` is computed at `src/prc/train.py:360`, before the seed loop at `:366-368`, and every seed
trains to the same `rounds` on the same matrix. More seeds average over bagging and feature-fraction
draws; they leave both the round count and the category permutation exactly where they were. **[S]**
A 10-seed refit would cost roughly +24 min ([X], from 3 seeds ≈ 615 s of the `main_reparam` refit,
linear in seeds) and would not reduce the 1252-vs-1175 difference by anything.

Averaging the **round count** — one early-stopping fit per seed, take the median `best_iteration` —
would reduce the selection variance by ~1/√k at ~4 min per extra seed, but it treats the symptom.
Take it only if 6.1 somehow proves insufficient.

### 6.6 Ranked

1. **6.1** deterministic categorical encoding (`pl.Enum` over sorted uniques inside `_matrix`) —
   free, removes the cause.
2. **6.3** pin `rounds_matched`/`rounds_unmatched` in the production config — free, saves 4 min, and
   the only thing that satisfies the repository's one-config reproducibility rule. **After 6.1.**
3. **6.2** `min_delta = 0.1` on early stopping — free, closes the residual ULP channel of §3.3.
4. **6.5b** median `best_iteration` over seeds — only if 1–3 prove insufficient.
5. **6.4** `num_threads: 1` — reject.
6. **6.5a** more refit seeds — reject *for this purpose*.

**One measurement is owed after the fix**, and it is cheap: re-run `configs/main_reparam.yaml` twice
with 6.1 applied and confirm the two `outputs/*.parquet` are byte-identical. Until that passes, the
fix is argued, not demonstrated.

---

## 7. What is left open

* **The size of the permutation distribution** is characterised by four draws (`rounds` 1006, 1018,
  1024, 1196) against four real runs (1018 cached, 1156 replay, 1175 and 1252 production). The two
  sets overlap and have comparable spread, which is what a single common cause predicts, but 1252 sits
  above every draw I took. Eight points is not a distribution. It does not change the diagnosis or the
  fix; it does mean the *typical* run-to-run move is not pinned down more tightly than §4.3's
  0.0–0.3 s.
* **Which polars operation loses the ordering** is not pinned down beyond "multi-chunk column". The
  frame arrives with 17 chunks from `read_parquet` and the model-A stack has 8; `rechunk()` fixes it,
  which localises it to chunk-parallel dictionary construction, but I did not read polars' source.
  Worth a one-line upstream issue; irrelevant to the fix, which does not depend on the answer.
* **§5.2's claim that model B is insulated** is four paired observations plus a mechanism sketch, not
  a proof. After 6.1 it stops mattering.
* **Every CV record in `experiments/experiments.jsonl` predating the fix** carries one draw of this
  noise on its matched lane. Comparisons at or below ~0.3 s that were taken between two such records
  — and the plan round is full of them — should be re-read with §4.2's table in hand before being
  relied on.

---

## Appendix — the scripts behind every [S-new]

All read-only with respect to the repository; none writes under `src/`, `configs/`, `outputs/`,
`experiments/` or `data/`. They live in `/tmp/nd/` and were run with `.venv/bin/python`.

| script | what it does | claims |
|---|---|---|
| `prep.py` | builds model A's design matrix once through `prc.train`'s own `fit_encodings`, `attach_encodings`, `visible_numeric_columns`, `_matrix`, `_inner_holdout_mask`; caches `Xtr.npy`/`y.npy`/`hold.npy` | §3.4 inputs |
| `fit.py` | N repeats of the early-stopping fit at a given round budget and thread count | §3.3 |
| `trees.py` | 3 repeats at 40 rounds, hashing `model_to_string()`, predictions and the loss history | §3.3 |
| `long.py` | stage A: 3 full-length repeats. stage B: 3 repeats at 1 thread. stage C: 4-thread control | §3.2, §3.3, §3.4 |
| `one.py` | one full-length fit per process from the cached matrix | §3.4 cold-process |
| `replay.py` | the **real** `prc.train.predict_fold` on `data/work/feat_{train,rank}.parquet`, with `lgb.train` wrapped to stop after model A's early-stopping fit | §3.6 (`best_iteration=1064`) |
| `mat.py` | builds the model-A matrix twice in one process and diffs it column by column | §3.5 |
| `cat2.py` | isolates the categorical cast; tests `rechunk()` and `pl.Enum(sorted)` | §3.5, §6.1 |
| `cat3.py` | the same test on model A's and model B's actual stacked frames | §5.2 |
| `perm.py` | refits the cached matrix with the categorical codes permuted and nothing else changed | §3.6 |
| — | polars/numpy one-liners over `outputs/*.parquet`, `experiments/oof_*.parquet`, `experiments/*.jsonl` | §1, §3.7, §4.2 |
| `python -m prc.compare submit_v10_reparam-77f92c802d main_reparam-bc60017878 --max-listed 3` | the repository's own decision rule on a null pair | §5.1 |
