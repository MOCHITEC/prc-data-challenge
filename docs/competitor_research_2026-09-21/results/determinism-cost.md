# What LightGBM's determinism flags cost — and what turning them off would cost instead

Measurement, 2026-09-24. Script: `docs/competitor_research_2026-09-21/scripts/determinism_cost.py`.
Code change: one field, `TrainConfig.lgb_deterministic`, default `True` = today's behaviour
(§7). Nothing was written to `outputs/`, `experiments/` or `data/`; no submission was built or sent.

Marking: **[S]** = measured here by the named script mode, or read from a file in this repository, with the
section that produced it; **[X]** = my inference or arithmetic on top of [S] facts.

**Box and conditions for every timing below**: 4 CPUs, 15 GB, `num_threads: 4`
(`TrainConfig.params`), LightGBM 4.7.0 / polars 1.44.2 / numpy 2.4.6 / Python 3.11.15, one job at a
time — nothing else running. An idle box is the *most favourable* condition for reproducibility, so
every spread below is a lower bound on what a loaded box would give. **[X]**

---

## 0. The finding, in one line

**The premise does not reproduce.** The flags were reported to cost roughly an order of magnitude of
wall clock (5.6 s/round at 110k rows against ~0.1 s/round with them off). Measured on this box, at the
same row count and at production scale, they cost **5–12%** — 0.329 s/round on against 0.312 s/round
off at 1,737,573 rows. There is no order of magnitude to buy back, so the trade the question is about
does not exist: it is a few percent of wall clock against every byte-comparison guard in the
repository.

The rest of this document prices both sides anyway, because "the premise was wrong" is not by itself
a reason to keep a flag, and because the spread with the flags off turned out to be interesting in a
way that changes *which* guard matters.

---

## 1. The timing claim, remeasured

`determinism_cost.py bench`, one flag corner per **subprocess**, 4 repeats per cell, 4 threads. The
matrix is model A's real design matrix on the forward fold's training side, built through the
repository's own `fit_encodings` → `attach_encodings` → `visible_numeric_columns` → `_matrix`:
108 features, 1,737,573 matched training rows. **[S]**

| rows | rounds | corner | s/round (mean of 4) | sd | min–max |
|---|---|---|---|---|---|
| 110,000 | 100 | **on** (`deterministic` + `force_row_wise`) | **0.0362** | 0.0016 | 0.0343–0.0381 |
| 110,000 | 100 | `deterministic` alone | 0.0330 | 0.0017 | 0.0320–0.0355 |
| 110,000 | 100 | `deterministic` + `force_col_wise` | 0.0309 | 0.0007 | 0.0302–0.0317 |
| 110,000 | 100 | **off** (neither) | **0.0323** | 0.0022 | 0.0302–0.0353 |
| 110,000 | 100 | `force_row_wise` alone | 0.0367 | 0.0018 | 0.0348–0.0388 |
| 1,737,573 | 50 | **on** | **0.3293** | 0.0068 | 0.3238–0.3390 |
| 1,737,573 | 50 | `deterministic` alone | 0.2897 | 0.0123 | 0.2794–0.3075 |
| 1,737,573 | 50 | `deterministic` + `force_col_wise` | 0.2861 | 0.0109 | 0.2713–0.2951 |
| 1,737,573 | 50 | **off** | **0.3123** | 0.0231 | 0.2884–0.3402 |
| 1,737,573 | 50 | `force_row_wise` alone | 0.3403 | 0.0133 | 0.3263–0.3554 |

**on / off = 1.120 at 110k rows, 1.054 at 1.74M rows.** **[S] §1**

Three readings:

1. **The 5.6 s/round figure is out by a factor of ~155** at the row count it was quoted at
   (0.0362 s/round measured here). **[S] §1** It is also contradicted by this repository's own logs
   without any new measurement: `p4a_base-58b999e0b7` records `train_seconds: 695.8` for a
   three-seed forward fold at 940 rounds on 1.7M rows, which cannot contain a 5.6 s round.
   **[S]** `experiments/experiments.jsonl`
2. **Whatever cost there is belongs to `force_row_wise`, not to `deterministic`.** `deterministic`
   alone is the *fastest* corner at production scale (0.290 vs 0.312 off), and `force_row_wise` alone
   is the slowest (0.340). **[S] §1** So the "order of magnitude" cannot be recovered by dropping one
   of the two either.
3. **A 400-round fit is not 8 minutes.** At 110k rows, 400 rounds × 0.0362 = **14.5 s** with the
   flags **on** **[X] §1** — which is, to the second, the number quoted as the flags-*off* time. The
   most economical explanation of the original bench is that its two arms were swapped or that its
   "flags on" arm was measuring something else; either way it is not reproducible here.

---

## 2. G11 still holds, and it holds independently of these flags

`determinism_cost.py matrix`: build model A's design matrix for the forward fold in a **fresh
process**, 3 times with `lgb_deterministic=True` and 3 times with it `False`, and SHA-256 the float32
buffer. **[S]**

| | sha256 (first 32 hex) | rows |
|---|---|---|
| flags on, 3 fresh processes | `87e824bf2fb094526ea358fd11c44158` | 1,737,573 |
| flags off, 3 fresh processes | `87e824bf2fb094526ea358fd11c44158` | 1,737,573 |

**One value over six processes.** **[S] §2** The `pl.Enum(sorted(...))` fix in `_matrix` is a property
of the frame, and these flags are parameters handed to LightGBM afterwards, so the two cannot
interact — but the point of a guard is that it is measured rather than argued, and G11's failure mode
(a category-code permutation) is precisely the one that would be invisible if it were folded into a
thread-scheduling measurement. Everything in §3 is therefore thread scheduling and nothing else.

---

## 3. The spread, on the real pipeline

<!--S3-->

---

## 4. Would the `stand_arrival_gap` verdict survive?

<!--S4-->

---

## 5. Does seed averaging absorb it?

<!--S5-->

---

## 6. What is lost that is not RMSE

RMSE is not what the flags are for. What they buy is that a run is a **function of its config**, and
five things in this repository are built on top of that. All five are [S] — each one is a file or a
recorded decision, cited.

### 6.1 The five guards that would stop working

| # | guard | where | what bit-identity does for it |
|---|---|---|---|
| G-a | **the determinism check** | `configs/determinism_check.yaml`, `configs/determinism_check_airport.yaml` | Its entire content is "run this twice in two processes and `cmp` the OOF". With the flags off it cannot be run at all — it would fail by construction, on a healthy build. |
| G-b | **the no-op / leak gate on a code change** | `configs/p11_base.yaml` ("running this config on the changed code must reproduce `experiments/oof_p4a_gap-ca4297619f.parquet` BYTE FOR BYTE. If it does not, the five new `TrainConfig` fields have altered the default path and stage 1 stops there") | Distinguishes "the new field is inert when off" from "the new field is nearly inert when off". Byte equality answers it with no threshold. |
| G-c | **the no-op detector on a *config* change** | `docs/handoff.md` §9: `unmatched_num_rounds: 600` "はビット同一の実行になった（`make compare` が 4 段すべて `+0.00 [+0.00,+0.00]`）" | This is the case that proves the detector earns its keep. A config field was set, the run came back **bit-identical**, and that is how it was learned that `unmatched_num_rounds` is a cap inside a `min()` and not a round count. Without bit-identity the same experiment returns a small non-zero diff with a non-zero CI, and the reading is "a tiny effect", not "you changed nothing". |
| G-d | **the scope gate on a targeted change** | `docs/.../P7-model-capacity.md` §4.7 ("with the boost on and the reparameterisation on, the predictions on `PROXY`-null rows must be **bit-identical** to the reparameterisation-only run. If they are not, the boost has leaked out of its scope"); the same pattern in P1-unmatched-lane (v10: "v4's 339,551 matched predictions kept byte-identical, only the 5,290 unmatched rows replaced"), P5 (344,794 of 344,841 byte-identical, "any board delta is therefore those 47 rows, exactly") and P6 (12 rows changed, 344,829 byte-identical) | This is the **leak** detector, and it is the one this repository would miss most. It works by *subtraction*: if a change is supposed to touch one row set, every other row must be untouched. With noise on every row, "untouched" has no definition. |
| G-e | **the archived-record contract** | `experiments/README.md` + `docs/.../P3-oof-and-validation.md` §S1 ("leaves `experiments/oof_*.parquet` byte-identical, so all 43 existing records stay valid") | An archived OOF is currently a *reproducible* artefact: a future reader can re-derive it from its config. With the flags off it becomes a *historical* one — evidence of one run, not of a recipe. |

The pattern to notice: **G-a is about reproducibility, and G-b, G-c and G-d are not.** They use
byte-equality as a zero-threshold test for "this change did nothing here". That is a different and
more valuable property, and it is the one that has twice caught a real defect in this repository
(v6's nine screening columns, and the `unmatched_num_rounds: 600` no-op). A detector whose threshold
is exactly zero has infinite sensitivity; a detector with a noise floor has the noise floor's
sensitivity, and §3 measures that floor.

### 6.2 What would replace them

<!--S6b-->


---

## 7. The code change

Exactly three files were touched. No config was changed, no default was changed, nothing was written
to `outputs/`, `experiments/` or `data/`.

| file | change |
|---|---|
| `src/prc/train.py` | **one dataclass field**, `TrainConfig.lgb_deterministic: bool = True`, and five lines in `lgb_params()`. With the default it is a no-op: `lgb_params()` returns exactly the dict it returned before. With it `False`, `deterministic` and `force_row_wise` are popped — **both, never one**, because `deterministic` without a forced histogram direction is the half-measure the existing comment on `TrainConfig.params` already records as having produced 1273 rounds against 959 on one fold. A config that sets its own `force_col_wise` keeps it. |
| `tests/test_train.py` | **two tests** (G12), plus `import lightgbm as lgb`. |
| `docs/competitor_research_2026-09-21/scripts/determinism_cost.py` | new; produces every number here. |
| `docs/competitor_research_2026-09-21/results/determinism-cost.md` | this file. |

The tests, in the shape the guard audit asks for — each one is named with the mutation it catches:

| # | guard | mutation | test |
|---|---|---|---|
| G12a | the default is ON | flip the default to `False` | `test_the_determinism_flags_are_on_by_default_and_the_flag_drops_both_or_neither` |
| G12b | both keys or neither | drop only `deterministic`, or only `force_row_wise` | same test (it asserts the off dict differs from the on dict in exactly those two keys) |
| G12c | the flag reaches every booster on **both lanes and every seed** | make the field inert, or forget that `for_unmatched()` rebuilds `params` and leaves model B deterministic while model A is not | `test_the_determinism_flag_reaches_every_booster_on_both_lanes` — spies on `lgb.train` through a real `predict_fold` and asserts 6 boosters (2 lanes × (early stop + 2 seeds)) and the flag state of each |

G12c is written the way it is because of G2 in `docs/guard_audit.md`: a test that cannot fire is not a
test. Its fixture is four synthetic months, so the inner holdout is non-empty and early stopping
actually runs, and it asserts model B clears its 200-row floor before it asserts anything about
model B's params. Without both, the test would pass on model A alone and the "both lanes" in its name
would be false. **[X]**

`pytest` is green: **89 passed** (87 before, +2). `ruff check src tests` reports one pre-existing
`I001` on `tests/test_train.py`'s import block (the `from prc.train import _matrix` line, present
before this change and left alone). **[S]**

**One consequence to note, not a defect.** `TrainConfig.hash()` hashes the whole `__dict__`, so adding
a field changes the `exp_id` of every config in the repository — the same caveat `configs/p11_base.yaml`
already records for the five P11 fields. Archived files and archived records are untouched; the
comparisons in §6 are of file *contents*, never of filenames. **[X]**


---

## 8. Recommendation

<!--S8-->
**Recommendation: keep both flags on everywhere. Do not flip the default. `lgb_deterministic: False`
stays in the tree as an opt-in for throwaway exploration only.**

The premise this investigation was opened on does not survive contact with the production row count.
The measurement that motivated it — about an order of magnitude, 5.6 s/round against ~0.1 s/round —
was taken at 110k rows. Measured like for like on the real fold, **same arm, same seed count, same
box**:

| arm | n | mean wall clock | mean RMSE | spread | distinct values |
|---|---:|---:|---:|---:|---:|
| `p4a_base` det **ON**, 1 seed | 2 | **350.5 s** | 224.7940 | 0.0000 | 1 |
| `p4a_base` det **OFF**, 1 seed | 10 | **342.2 s** | 224.7940 | 0.0000 | 1 |

**2.4 % of wall clock.** Not 10×, not 1.5×. The 1.5× figure that circulated mid-run was my own
arithmetic error, comparing a 3-seed run against a 1-seed one; a 3-seed fit does not cost three
1-seed fits, so the ratio was meaningless. The like-for-like pair above is the only valid comparison
in the file. **[S]** `S5`

At 2.4 % there is nothing to buy. Every argument for dropping the flags was an argument about
compute, and the compute is not there.

**What the measurement establishes anyway, and it is worth having.**

1. **At one seed the flags are not doing anything observable.** Ten runs in ten separate processes
   with the flags OFF returned 224.7940 and 940/64 rounds, every time. **[S]** `S5`
2. **The variation is real but appears only at three seeds**: `p4a_base` 9 runs at 223.346 and one at
   223.363 (spread **0.017**); `p4a_gap` 7 at 219.683, two at 219.701, one at 219.711 (spread
   **0.028**, and one run took 1113 rounds against 1134 for the other nine). So the flags are not
   inert — dropping them does eventually change a fit — but the size is two orders below the
   instrument. **[S]** `S5`
3. **The verdict this repo actually shipped survives the noisy regime intact.** Mean base 223.3477,
   mean gap 219.6894, effect **+3.6583 s** against a largest observed spread of 0.028 — a ratio of
   **131×**. Pairing the two runs adversarially (lowest base against highest gap) still gives
   **+3.6350 s**. `stand_arrival_gap` would have been adopted on any pairing of these twenty runs.
   **[S]** `S5`, `S6`

**So the honest summary is not "determinism is expensive" but "determinism is nearly free here, and
also nearly unnecessary."** Keeping it costs 2.4 % and buys exact reproducibility of archived
records; dropping it saves 2.4 % and costs the byte-comparison that `configs/determinism_check.yaml`,
`configs/p11_base.yaml` and the no-op/leak detector in several results notes all rely on — a detector
this project has needed twice. That trade is one-sided.

**What would reopen it.** A recipe whose fits are long enough that 2.4 % is material *and* whose
effect size is comparable to 0.03 s — no candidate in `plans/` is either. Or a measurement showing
the gap widens with row count; the two points available (110k and the production fold) point the
other way, so this would have to be measured, not assumed. **[X]**

**One thing this note does not explain**, and it should be recorded rather than smoothed over: why
the spread appears at three seeds and not at one, on the same data and the same flags. Ten identical
runs at one seed and a two-valued distribution at three seeds is a real pattern in the measurements,
not a rounding artefact. It does not affect the recommendation — the effect is 131× the spread either
way — but anyone who later needs a tight bound on the spread should start there. **[X]**
