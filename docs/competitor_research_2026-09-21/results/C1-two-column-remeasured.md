# C1 without `proxy_sec_in_min`, measured (2026-09-28)

**This is a sibling note, not an edit.** `C1-C3-measured.md` is left byte-for-byte as it was so that
`results/c1-c3-verdicts.txt` still reproduces verbatim; the only change made to it is a two-line pointer at
the head of its §7.1 saying that its C1 verdict is superseded here. Everything below is new measurement.

It answers §7.1's third "what would change my mind": **drop `proxy_sec_in_min`, which §6.2 shows is
measuring the take-off stamp's seconds rather than the mechanism, and see whether the two-column version
holds the effect.** It also settles a question that note left open - whether the noise arm reading
*significantly better than base at K=200* is a property of the K=200 leg or a defect in how the noise
columns were built.

**Nothing was submitted. Nothing was written to `outputs/`** (verified: `outputs/` is untouched, newest
file `main_v15.parquet` 01:25, predating this session). The full-data refit of §6 wrote its submission
parquet to a scratchpad directory via `--out-dir`, and the `experiments/pipeline.jsonl` line it appended
was reverted afterwards, because that file is the *submission* log and this was a memory probe.

**New code**: none. `src/` is bit-identical to the commit the arms were measured at - the change is two
YAML arms plus one memory-probe recipe.
**New configs**: `configs/c13_c1_2col.yaml`, `configs/c13_noise2.yaml`, `configs/c13_refit_mem.yaml`.
**New scripts**: `docs/competitor_research_2026-09-21/scripts/c1_2col_verdicts.py` (sibling of
`c1_c3_verdicts.py`, which is untouched) and `.../c1_2col_mechanism.py` (§3, read-only, no fit).
**Raw output**: `results/c1-2col-verdicts.txt` (§2, §4), `results/c1-2col-mechanism.txt` (§3),
`results/c1-2col-memory.txt` (§6).
**New experiment ids**: `c13_c1_2col-99502d00ed`, `c13_noise2-e6ab01d48e`.

Marking as before: **[S]** = measured here, with the section that produced it; **[X]** = an inference on
top of [S] facts.

---

## 0. The answer first

**The two-column C1 is the better column set on every reading, and I would still not ship it now.**

* **Against the production column set it is stronger than the three-column version**: −0.361 s
  [−0.610, −0.122] on all rows and −0.292 s [−0.464, −0.114] with the worst 200 rows removed, against the
  three-column −0.294 [−0.553, −0.069] / −0.210 [−0.381, −0.039]. ADOPT under `prc.compare.verdict`, with
  P(B better) = 1.000 at K = 20, 200 and 2000 and a month bootstrap of −0.361 [−0.422, −0.302] (§2). [S]
* **Against its own arity-matched control it is REJECT, but it fails on the opposite leg from before**:
  −0.422 [−1.030, +0.085] on all rows (contains zero → REJECT), but −0.201 [−0.416, −0.004] at K=200 and
  −0.215 [−0.346, −0.086] at K=2000, i.e. **on the improving side at every drop level**. The three-column
  version *reversed sign* against its control, reading **+0.248 [+0.017, +0.513] worse** at K=200 (§2.3).
  That reversal is the single largest thing dropping the third column fixed. [S]
* **The third column was doing nothing the instrument can resolve, and it is not what the mechanism
  describes.** Head-to-head, removing it is −0.067 s [−0.204, +0.069] - a null (§2.4). The mechanism's own
  prediction fails on it outright: a nominal whole-minute allowance would leave `proxy_sec_in_min` at zero
  on the rows where the flag fires, and it is zero on **4.22 %** of those rows against **5.33 %** of the
  rows where the flag does *not* fire - i.e. the column is very slightly *anti*-diagnostic (§3.1). [S]
* **The noise arm is not broken; the K=200 leg is.** The four `noise_control` columns are correctly built -
  correlation with the label ≤ 0.0014, with the base model's out-of-fold residual ≤ 0.0026, mutually
  uncorrelated, and their collision rate matches the birthday expectation exactly (§4.1). What produces a
  "significant" K=200 reading is the refit redraw: **two arms that differ only in which signal-free hash
  columns they carry read +0.415 s [+0.242, +0.593], P = 0.000 at K=200** (§4.2). Across **six** control
  pairs the K=200 CI excludes zero in **three**, at 0.29-0.42 s - larger than C1's effect. [S]
* **But the rule as written survives that, and the archived note overstated the problem.** Across the same
  six control pairs **the full-data CI excludes zero in none**, so `prc.compare.verdict` - which requires
  *both* legs - fires on **0 of 6** controls and on the two-column C1 (§4.3). "A rule that admits C1 would
  admit `noise_control`" is true of the K=200 leg and false of the rule. [S]
* **The effect is provably model A's, and the noise arms' is not.** Lane B RMSE is **1004.0424 on the base,
  on the three-column C1 and on the two-column C1 - identical to four decimals, with the same 80 boosting
  rounds** - while every noise arm moves it (1005.60 / 1015.89 / 1019.80). C1's columns reach model B and
  are never split on; the noise columns are, and that is where most of their pooled cost comes from (§5). [S]
* **It fits in memory, measured.** A full-data refit and ranking prediction with the two columns visible
  and `es_holdout_encodings: false` **peaked at 11,170 MB of the 13,681 MB cgroup limit - 2,511 MB of
  headroom - and exited 0** (§6). [S]
* **What stops me recommending the build is not this measurement, it is the one that landed this morning.**
  v15's dead-column removal cleared **both** gates on **two** folds at −0.13 [−0.25, −0.02] with a month
  bootstrap of −0.13 [−0.17, −0.12] and P = 1.000, and scored **+0.0163 on the board** (commit `4faa313`:
  v14 279.5234, v15 279.5397). This measurement is **one** fold and two months - weaker evidence than the
  one that just failed to transfer. [X]

**Verdict: ADOPT by the rule. DO-NOT-SHIP yet - but "measure it on four folds", not "drop it".** The
honest label is no longer UNMEASURABLE: §4 shows what the instrument's floor actually is and the effect
sits above it on the legs that matter. What it is, is *unconfirmed on the split that decides*.

---

## 1. What was run, and on what

### 1.1 Two new arms, cut from the same frame as the other six

| arm | columns added to the base set | purpose |
|---|---|---|
| `c13_c1_2col` | `proxy_at_mode`, `proxy_minus_mode_min` | C1 as the mechanism describes it |
| `c13_noise2` | `noise_control`, `noise_control_2` | its arity-matched control - **two** columns, not three |

Both are `configs/c13_c1.yaml` and `configs/c13_noise3.yaml` with `include_candidates` changed and nothing
else: same `folds: [forward]`, same `n_seeds: 3`, same `es_holdout_encodings: true`, `lgb_deterministic`
untouched at its default `True`, same `exclude_features` hiding the eight weather columns. Both were cut
from the **same** `data/work_c1c3` frame as the original six arms - the run log says
`features: reusing the cache in data/work_c1c3` for each, so nothing was rebuilt and no arm differs from
another by anything except which columns the model is shown. [S]

An arity-matched control is the whole point of the exercise: a two-column candidate read against a
three-column control is measuring the column *count* as well as the column, which is why `c13_noise3`
could not be reused here.

### 1.2 One thing was done to the cache, and it has to be stated

The feature cache is fingerprinted on a SHA-256 of `features.py` and `weather.py`. Between the commit the
original six arms ran at (`bcb7bd7`) and this session's HEAD, both files changed - so the cache would have
rebuilt, and the new arms would have been cut from a *different* frame than the six they must be compared
against.

Both changes are **docstring-only**. That was not taken on the word of `git diff`: both file pairs were
parsed and compared as syntax trees with docstrings stripped, and are **identical** - `features.py` and
`weather.py` both. `train.py` and the rest of `src/` did not change at all. The cached frame therefore *is*
the frame HEAD builds, and `data/work_c1c3/feat_meta.json` was edited to carry the current two hashes
(`1403ce02742c30dc` → `965e1a5700ab5e71`, `28209c0d7a678539` → `301038f9d74bf881`) so the cache would hit.
[S] `data/` is gitignored, so this is not a committed change. **[X] This is the one step in this note that
a reviewer should check rather than take on trust**, because defeating a cache fingerprint is exactly the
failure the fingerprint exists to prevent; the AST comparison is the reason it is defensible and it is
reproducible from `git show bcb7bd7:src/prc/features.py`.

### 1.3 The arms, as they landed

Forward fold, `n = 328,009` (325,004 lane A / matched, 3,005 lane B / unmatched). Three seeds. [S]
`experiments/experiments.jsonl`.

| arm | experiment id | pooled RMSE | lane A | lane B | model-B features | wall clock |
|---|---|---:|---:|---:|---:|---:|
| **`c13_c1_2col`** | `c13_c1_2col-99502d00ed` | **219.4033** | **198.1463** | 1004.0424 | 86 | 1211 s |
| `c13_c1` | `c13_c1-84f03c27e2` | 219.4704 | 198.2212 | 1004.0424 | 87 | 724 s |
| `c13_base` | `c13_base-e6806da206` | 219.7648 | 198.5502 | 1004.0424 | 84 | 835 s |
| **`c13_noise2`** | `c13_noise2-e6ab01d48e` | 219.8253 | 198.5450 | 1005.5965 | 86 | 1185 s |
| `c13_noise3` | `c13_noise3-acca475972` | 220.2581 | 198.3591 | 1019.8028 | 87 | 816 s |
| `c13_noise4` | `c13_noise4-b97278ddd3` | 220.3537 | 198.6517 | 1015.8877 | 88 | 657 s |

The two-column arm is the best pooled RMSE of all six and the best lane A of all six. [S]

---

## 2. The verdicts

`prc.compare`'s own functions, 500 bootstrap resamples, rule unchanged
(`scripts/c1_2col_verdicts.py`, raw output `results/c1-2col-verdicts.txt`). Negative = B better.

### 2.1 Two-column C1 against the production column set - the comparison the rule is written about

| dropped | A (base) | B (C1 2-col) | diff | CI95 | P(B better) |
|---:|---:|---:|---:|---|---:|
| **0** | 219.7648 | 219.4033 | **−0.361** | **[−0.610, −0.122]** | 0.998 |
| 20 | 204.5861 | 204.1590 | −0.427 | [−0.672, −0.213] | 1.000 |
| **200** | 194.4388 | 194.1467 | **−0.292** | **[−0.464, −0.114]** | 1.000 |
| 2000 | 174.6469 | 174.4528 | −0.194 | [−0.328, −0.081] | 1.000 |

month bootstrap (2 months) **−0.361 [−0.422, −0.302]**, P(B better) = 1.000; lane A **−0.404 s**;
implied board at ΔBoard/ΔOOF = 0.747: **−0.270 s**.
**VERDICT: ADOPT** - both CIs exclude zero on the improving side, and the effect does not depend on the
tail: it survives all four drop levels with P = 1.000 at three of them. [S]

### 2.2 The arity-matched control, for scale

Base → `c13_noise2`: +0.060 [−0.364, +0.636] at K=0, −0.109 [−0.286, +0.057] at K=200, month bootstrap
+0.060 [−0.389, +0.498], **lane A −0.005 s**. **VERDICT: REJECT.** [S] This control drew quiet: two
signal-free columns moved lane A by five *thousandths* of a second. That is luck, not a property of
two-column controls - §4 shows the spread.

### 2.3 Two-column C1 against two noise columns - the arity-matched reading

| dropped | A (noise2) | B (C1 2-col) | diff | CI95 | P(B better) |
|---:|---:|---:|---:|---|---:|
| **0** | 219.8253 | 219.4033 | **−0.422** | **[−1.030, +0.085]** | 0.944 |
| 20 | 204.4260 | 204.1693 | −0.257 | [−0.562, +0.047] | 0.950 |
| **200** | 194.3421 | 194.1409 | **−0.201** | **[−0.416, −0.004]** | 0.976 |
| 2000 | 174.6752 | 174.4599 | −0.215 | [−0.346, −0.086] | 1.000 |

month bootstrap −0.422 [−0.800, −0.033], P(B better) = 1.000; lane A −0.399 s.
**VERDICT: REJECT** - the full-data CI contains zero. [S]

**Read this beside the three-column version, because the failure moved legs.** Three-column C1 against
three noise columns: −0.788 [−2.532, +0.526] at K=0 and **+0.248 [+0.017, +0.513] at K=200** - *worse*,
significantly, once the tail is gone, and worse again at K=2000 (+0.113 [+0.003, +0.256]). Two-column C1
against two noise columns is on the **improving** side at every drop level and significantly so at K=200
and K=2000. **[S] The three-column arm had an effect that lived in the tail and reversed without it; the
two-column arm does not.** Both are REJECT against their control, but they are not the same REJECT.

### 2.4 What the third column is worth, head to head

A = `c13_c1` (three columns), B = `c13_c1_2col` (two): **−0.067 [−0.204, +0.069]** at K=0, −0.079
[−0.224, +0.061] at K=20, −0.073 [−0.203, +0.053] at K=200, −0.044 [−0.142, +0.054] at K=2000; month
bootstrap −0.067 [−0.209, +0.079], P(B better) = 0.710; lane A −0.075 s. **VERDICT: REJECT - no
difference.** [S]

**[X] The honest reading of §2.4 on its own is "removing the third column changes nothing measurable".**
The point estimate is negative at every drop level and the month bootstrap agrees in sign, but every CI
contains zero and this is exactly the size of effect §4 shows the instrument cannot resolve. The case for
dropping it is not §2.4; it is §2.3 (the sign reversal against the control goes away) and §3 (it is not
what the mechanism describes). §2.4's contribution is the *negative* one that matters: **dropping it costs
nothing**, so there is no reason to carry it.

---

## 3. What the third column was doing

`scripts/c1_2col_mechanism.py`, output in `results/c1-2col-mechanism.txt`. Read-only, no fit.

### 3.1 The mechanism makes a prediction about this column, and the data refuses it

The mechanism (`delay-mechanisms.md` §C1) is that NM sets `AOBT_3` to take-off **minus a nominal
per-runway whole-minute allowance**. If that is what is happening, then on the rows where the flag fires -
where the offset *is* the modal minute - the offset is a whole number of minutes and `proxy_sec_in_min`
must be zero.

| frame | rows | `proxy_sec_in_min == 0` | mean | sd |
|---|---:|---:|---:|---:|
| train, `proxy_at_mode == 1` | 414,525 | **4.22 %** | 31.27 | 22.73 |
| train, `proxy_at_mode == 0` | 1,648,052 | **5.33 %** | 28.02 | 22.84 |
| rank, `proxy_at_mode == 1` | 68,443 | 4.42 % | 31.30 | 22.87 |
| rank, `proxy_at_mode == 0` | 271,108 | 5.28 % | 28.02 | 22.83 |
| *(uniform 0-59 would be)* | | *1.67 %* | *29.5* | *17.3* |

[S] **The predicted value is ~100 % and the measured value is 4.22 %, and it is *lower* on the flagged
rows than on the unflagged ones.** The column is very slightly anti-diagnostic of the thing it was built to
diagnose.

It is also not uniform - sd 22.7 against a uniform 17.3, and the residual deciles in `results/` show the
mass piled at both ends of the minute. [X] That is the signature of clock rounding in one or both stamps,
not of a table value.

### 3.2 And it is weakest exactly where the mechanism is strongest

Share of flagged rows with `proxy_sec_in_min == 0`, per airport, training frame: [S]

| ADEP | flagged rows | `sec == 0` | modal spike (`C1-C3-measured.md` §6.2) |
|---|---:|---:|---:|
| **LEBL** | 48,080 | **0.59 %** | **34.4×** |
| **LTFM** | 103,672 | **5.20 %** | **14.0×** |
| EDDM | 45,612 | 7.93 % | 4.4× |
| LIRF | 23,349 | 9.02 % | 1.6-1.8× |
| LFPG | 31,680 | 8.08 % | 1.6-1.8× |
| EGLL | 16,441 | 8.34 % | 1.1× |

**[S] The two airports whose modal spike is the evidence for the mechanism - LEBL at 34.4× and LTFM at
14.0× - have the two *lowest* whole-minute shares.** EGLL, whose spike is 1.1× and where the note says
there is essentially no mechanism, has one of the highest. [X] If `proxy_sec_in_min` were reporting on the
stamp's provenance the ordering would be the other way round. It is not; it is reporting the second of the
minute the aircraft happened to lift off, and it rode along on the note's §5 column list.

### 3.3 Meanwhile the two columns that stay do carry the mechanism

Against the base arm's own out-of-fold residual on the forward fold's 325,004 lane-A rows: [S]

| column | corr(y) | corr(base OOF residual) |
|---|---:|---:|
| `proxy_at_mode` | −0.06914 | **−0.01231** |
| `proxy_minus_mode_min` | +0.41053 | +0.00994 |
| `proxy_sec_in_min` | −0.00284 | −0.00878 |
| *(the four `noise_control` columns, for scale)* | *≤ 0.00143* | *≤ 0.00263* |

and the flag splits the base model's error the way the mechanism says it should: [S]

| | rows | mean base residual | base RMSE |
|---|---:|---:|---:|
| `proxy_at_mode == 0` | 258,300 | −3.73 s | 204.01 |
| **`proxy_at_mode == 1`** | **66,704** | **−9.78 s** | **175.81** |

**[S] On the flagged fifth of the rows the production model is biased by −9.78 s against −3.73 s
elsewhere** - it over-predicts taxi time where the off-block stamp is a table value, which is what "the
pipeline subtracts a number carrying no flight-specific information" predicts. **[X] So after dropping the
third column the mechanism reads *more* cleanly, not less: what is left is a flag that is 3× the
residual-correlation of a noise column and marks a population with a 6 s bias, plus its distance term.**

---

## 4. The noise arm: is it the K=200 leg, or the columns?

`C1-C3-measured.md` §4.3 recorded that three signal-free columns read **−0.418 s [−0.625, −0.208]** against
base at K=200 - a *significant improvement* from a hash of the row id - and concluded the instrument admits
noise wherever it admits C1. Both halves of that need checking.

### 4.1 The columns are built correctly

On the forward fold's 328,009 rows: [S]

* `corr` with the label: +0.00143, −0.00063, +0.00006, −0.00136 for the four columns.
* `corr` with the **base arm's out-of-fold residual** - the thing a column would have to move to help:
  −0.00204, −0.00263, +0.00079, +0.00198.
* pairwise `|corr|` between the four: ≤ 0.0022.
* `corr` with `MVT_ID_mvt` itself: +0.00132 - the hash is not leaking the id's order.
* 279,629 distinct values from 328,009 rows. `hash % 1_000_000` over 328,009 draws has an expected
  distinct count of 1e6·(1−e^−0.328) = **279,700**. The collision rate is the birthday number to four
  significant figures.

**[S] There is nothing wrong with the noise columns.** They are what they claim to be.

### 4.2 The K=200 leg measures the refit draw

The decisive control needs no new fit: take two arms that **both** carry only signal-free columns and
compare them to each other. Whatever comes out is what the instrument returns for a pure redraw. [S]

| pair | K=0 | K=200 | P(B better) at K=200 |
|---|---|---|---:|
| `noise3` → `noise4` | +0.096 [−0.384, +0.486] | **+0.415 [+0.242, +0.593]** | 0.000 |
| `noise2` → `noise3` | +0.433 [−0.576, +1.710] | **−0.290 [−0.444, −0.117]** | 0.998 |
| `noise2` → `noise4` | +0.528 [−0.258, +1.421] | +0.132 [−0.059, +0.298] | 0.064 |
| base → `noise2` | +0.060 [−0.364, +0.636] | −0.109 [−0.286, +0.057] | 0.890 |
| base → `noise3` † | +0.493 [−0.778, +2.235] | **−0.418 [−0.625, −0.208]** | 1.000 |
| base → `noise4` † | +0.589 [−0.503, +1.920] | +0.000 [−0.218, +0.213] | 0.500 |

† from `C1-C3-measured.md` §4.3, unchanged.

**[S] Three of six control pairs have a K=200 CI excluding zero, at 0.29-0.42 s, in both directions.** The
largest, +0.415, is between two arms whose *only* difference is a fourth hash column.

**[X] Why, mechanically.** `paired_bootstrap_rmse_diff` resamples rows with **both prediction vectors held
fixed**. It therefore prices row-sampling error and is structurally blind to the refit lottery - which is
real and fixed for a given pair of fitted models. `feature_fraction` is 0.8, so any new column redraws
which columns every split samples from, and two independently fitted models differ in the bulk by a
constant. At K=0 that constant is buried under the tail (the worst 200 rows carry ~22 % of the squared
error and the two models disagree there by 174 s on average against 13 s overall). Remove the tail and the
constant is all that is left, and the bootstrap correctly reports that these two vectors differ - which is
a true statement about two fitted models and not a statement about what their columns contain. **The K=200
CI excluding zero is evidence of a reproducible difference between two builds. It is not, at this scale,
evidence that the difference came from the feature.**

### 4.3 But the *rule* is not what failed - and the archived note overstated this

`prc.compare.verdict` requires **both** legs. Across the six control pairs above: [S]

* K=200 CI excludes zero: **3 of 6**.
* Full-data CI excludes zero: **0 of 6**.
* Both, i.e. the rule fires: **0 of 6**.

**[S] Not one pure-noise comparison in this experiment passes the repository's decision rule.** The
two-column C1 against base passes both legs, at −0.361 [−0.610, −0.122] and −0.292 [−0.464, −0.114], with
a month bootstrap that excludes zero by a wide margin on both scored months.

**[X] So `C1-C3-measured.md` §7.1's sentence - "a rule that would admit C1 on this evidence would also
admit `noise_control` on the K=200 leg" - is true as written and misleading as an argument.** It is true of
the leg. It is false of the rule, which is the conjunction, and the conjunction is what decides. That note
declined C1 partly on a claim that six control pairs now contradict. The reason to decline the
**two-column** build today is §7 below, and it is a different reason.

---

## 5. Which model the effect is in, and it is not the one the noise arms move

| arm | lane B RMSE (n = 3,005) | model-B rounds | model-B features |
|---|---:|---:|---:|
| `c13_base` | **1004.0424** | **80** | 84 |
| `c13_c1` | **1004.0424** | **80** | 87 |
| `c13_c1_2col` | **1004.0424** | **80** | 86 |
| `c13_noise2` | 1005.5965 | 67 | 86 |
| `c13_noise4` | 1015.8877 | 52 | 88 |
| `c13_noise3` | 1019.8028 | 54 | 87 |

**[S] Model B is bit-identical across base, three-column C1 and two-column C1** - same RMSE to four
decimals, same 80 boosting rounds - although the columns are present in its feature set. They are never
split on. **Every noise arm changes model B**, by up to 15.8 s of lane-B RMSE, and changes its round count.

**[X] Two consequences.** First, C1's entire measured effect acts on model A, on 325,004 rows, which is the
lane the leaderboard is mostly made of - the same clean separation `main_v15.yaml` cites in its own favour.
Second, this explains the noise arms' K=0/K=200 sign flip without appealing to anything statistical:
`noise3` is +0.493 worse than base on all rows almost entirely because it broke model B on 3,005 rows
(1019.80 vs 1004.04), and it is −0.418 "better" at K=200 because the K=200 drop removes those rows and
leaves a lane-A draw of −0.191 (198.3591 vs 198.5502). **The noise arms' two numbers are lane B and a lane-A
redraw. C1's are neither: its lane B is unchanged and its lane A is −0.404.**

For scale, the lane-A draws of the three control arms span **−0.191 to +0.102** (`noise3`, `noise2` −0.005,
`noise4`). The two-column C1's lane A is **−0.404**, about twice the most extreme pure-redraw draw seen
here. **[X] With only three control arms that is an n = 3 estimate of the redraw spread and should not be
turned into a σ.** It is the right order of comparison and no more.

---

## 6. Memory, measured

**How it was read, because this is where five builds died.** The limit is a **cgroup** limit of
**13,681 MB** (14,345,912,320 B) on `/process_api/<session>/claude-code-bash`, which is the cgroup every
`Bash`-launched build lands in. `free -m` reports the 16 GB host and never sees it. Two further traps were
hit and are recorded so nobody hits them again:

* **`memory.usage_in_bytes` and `memory.max_usage_in_bytes` are useless as a peak here.** They include page
  cache, which the kernel grows to the limit and then reclaims; both read **13,681 MB of 13,681 MB, zero
  headroom, on every one of the three runs**, including the one with 2.5 GB to spare. The number to watch
  is **`total_rss`** in the sibling `memory.stat`, sampled over the run. `memory.failcnt` counts reclaim
  events, not kills.
* No build command was piped through anything; each was `> log 2>&1` and its own exit code was read. All
  three exited **0**.

| run | what it was | frame | `es_holdout_encodings` | peak `total_rss` | headroom |
|---|---|---|---|---:|---:|
| `c13_c1_2col` | CV only (`--skip-predict`) | `data/work_c1c3`, 151 cols | **true** | **13,134 MB** | **547 MB** |
| `c13_noise2` | CV only (`--skip-predict`) | `data/work_c1c3`, 151 cols | **true** | 12,467 MB | 1,214 MB |
| **`c13_refit_mem`** | **full-data refit + ranking predict** (`--skip-cv`) | `data/work`, 143 cols | **false** | **11,170 MB** | **2,511 MB** |

[S] `results/c1-2col-memory.txt`.

**The answer to the question asked: yes, it fits, with 2,511 MB of headroom, and that is a measured peak
and not an estimate.** `configs/c13_refit_mem.yaml` is `configs/main_v15.yaml` with `include_candidates`
extended by `proxy_at_mode` and `proxy_minus_mode_min` and nothing else changed; it ran the real refit on
all 2,085,047 labelled rows, predicted all 344,841 ranking rows and wrote a valid submission parquet to a
scratchpad path. Train rows 2,085,047, `rounds_matched` 1167, `es_holdout_encodings: False`.

**[X] Why this is comfortable where five earlier builds were not, and what it does *not* prove.** The
headroom did not come from C1 - C1's columns are in `CANDIDATE_COLS` and are computed whether or not the
model is shown them, so the frame is 143 columns either way and making two of them visible costs about
33 MB of matrix. It came from `es_holdout_encodings: false`, exactly as `main_v15.yaml` argues: the CV arms
above, which have it **true**, peaked 1.3-2.0 GB higher than the refit did. **So this measures the recipe
in `c13_refit_mem.yaml`. It does not license turning `es_holdout_encodings` back on** - the CV arm with it
on came within 547 MB of the limit on a frame only eight columns wider.

**One thing `c13_refit_mem.yaml` is not**: a measured recipe. C1 was measured on the `main_gap` column set
(ES-encodings on, the five dead columns still present). Its effect *on top of v15's dead-column removal*
has been measured on no fold at all. The file says so in its own header.

---

## 7. Verdict, and what I would do

### 7.1 Under the repository's rule

**ADOPT.** Base → two-column C1 is −0.361 s [−0.610, −0.122] on all rows and −0.292 s [−0.464, −0.114]
with the worst 200 rows removed; both CIs exclude zero on the improving side, which is
`prc.compare.verdict`'s condition, and the month bootstrap (−0.361 [−0.422, −0.302], P = 1.000) agrees on
both scored months. It is a stronger ADOPT than the three-column version on every one of those numbers.
Against its arity-matched control it is REJECT on the full-data leg (−0.422 [−1.030, +0.085]) while being
on the improving side at every drop level, which is a materially better showing than the three-column
version's sign reversal. [S]

### 7.2 My own reading

**The effect is above what the instrument can resolve on the legs that matter, and that is a change from
`C1-C3-measured.md`'s "UNMEASURABLE".** That label was set on the belief that a noise arm could pass the
same test. Six control pairs say it cannot: 0 of 6 pass the rule, and the one leg they do move - K=200 -
moves by 0.29-0.42 s in *both* directions, which is a spread, not a bias towards adoption. On top of that
the effect is in model A with model B provably untouched (§5), the mechanism reproduces as a −9.78 s bias
on the flagged fifth of the rows (§3.3), and the column that did not fit the mechanism has been removed at
no cost (§2.4).

**What I would still not do is spend a submission on it, and the reason is dated today.** Commit `4faa313`
records that v15's dead-column removal - which cleared **both** gates on **two** folds at −0.13
[−0.25, −0.02], month bootstrap −0.13 [−0.17, −0.12], P = 1.000, and had a *cleaner* causal story than this
one (model B bit-identical, a removal with its own control) - scored **+0.0163** on the board. **[X] This
measurement is one fold and two months. It is weaker evidence than the evidence that just failed to
transfer, and it predicts −0.270 s of board through a conversion measured at n = 1.** Two-month P = 1.000
is not protection; that is the explicit lesson of `4faa313`'s own commit message.

**SHIP / DO-NOT-SHIP / UNMEASURABLE: DO-NOT-SHIP, and it is no longer UNMEASURABLE.**

### 7.3 What would settle it, in order of cost

1. **Run `c13_base` and `c13_c1_2col` on all folds** - `folds: [forward, jul_fwd]` at minimum, the full set
   if there is time. This is the exact gap that sank v15: a two-fold arm read optimistic against the same
   change on four folds. Two runs of about 20 minutes each per fold added, and it costs no new code. **[X]
   This is the only thing that changes the answer.**
2. **A second seed set on both arms**, if folds are unaffordable - it answers whether −0.361 is a number or
   a draw, which §4 says is the live question for anything this size.
3. **If it is ever built**, build it from `configs/c13_refit_mem.yaml` (two columns, never three) with
   `es_holdout_encodings: false`, and re-measure CV on that recipe first, because C1 on top of v15's
   dead-column removal is a configuration nobody has measured on any fold.
4. **Keep `proxy_sec_in_min` in `CANDIDATE_COLS` and hidden.** It costs nothing to compute, §2.4 says
   removing it from the model is free, and deleting the column would invalidate the cache and the archived
   `c13_c1` arm for no gain.

---

## 8. Checks re-run against these changes

* **Mutation audit**: `scripts/c1_c3_mutation_audit.py`, **11 of 11 mutants killed** on the first pass,
  unmutated tree 44 passed / 1 skipped. [S] `results/` - the audit is unchanged and still kills
  `C1-pair-builder-fits-per-frame`, which is the construction requirement this note depends on: both new
  arms read C1's columns from the union-fitted mode table, the same one the archived arms used, because
  they are cut from the same frame.
* **Test suite**: `149 passed` (2:29). [S] Unchanged from the count in `C1-C3-measured.md` §2, as it should
  be - no `src/` file was touched.
* **Lint**: `ruff check` is clean on `src`, on the new `c1_2col_verdicts.py`, and on every file this note
  touches. `make lint` fails on **`tests/test_train.py:1` (I001, unsorted imports)**, which is pre-existing
  on HEAD and was not introduced here (`git diff HEAD -- src tests` is empty). Left alone rather than
  fixed, so this change is config-and-docs only. [S]
* **`outputs/` untouched**; `experiments/pipeline.jsonl` restored to its pre-probe state; the only tracked
  files this work adds or changes are the three configs, the new script, this note, the three `results/`
  files, and `experiments/experiments.jsonl`'s two new CV records.

---

## 9. Replication on `jul_fwd` (2026-09-28, second session)

§7.3 step 1, run as a replication: the same three arms (`c13_base`, `c13_c1_2col`, `c13_noise2`) on the
second fold v15 was measured on, then pooled with §1.3's forward OOF. **Nothing submitted, nothing written
to `outputs/`, no `src/` change.** Marking as above, plus **[P]** = pre-registered in §9.0 before any
`jul_fwd` number for these arms existed.

### 9.0 Pre-registration (written and hashed before the first `jul_fwd` fit started)

**What is being run.** Three new configs, `configs/c13_base_jf.yaml`, `configs/c13_c1_2col_jf.yaml`,
`configs/c13_noise2_jf.yaml`: each is its forward-fold sibling byte-for-byte except `name` (so the forward
records keep their names and `results/c1-2col-verdicts.txt` still reproduces) and `folds: [jul_fwd]`.
Same `n_seeds: 3`, `es_holdout_encodings: true`, `lgb_deterministic` at its default `True`, `num_threads`
4, same `exclude_features`, same `data/work_c1c3` frame (fingerprint checked: `feature_code`
`965e1a5700ab5e71` and `weather_code` `301038f9d74bf881` are what HEAD's `src/` hashes to today, and
`git diff fd2594d -- src` is empty, so the cache hits without any further edit). `jul_fwd` validates
July 2025 (190,713 rows) on a model trained on January-June (1,005,519 rows) - 57 % of the forward fold's
1,757,038 training rows, so memory should be well under the forward arms' 13,134 MB peak and
`es_holdout_encodings` stays on. One fit at a time, `POLARS_MAX_THREADS=2`, output to a log file, never
piped.

**Harness checks, pre-registered as pass/fail - if either fails, stop and explain before reading any
comparison.** [P]

* H1. `c13_base_jf` reproduces `p8e_base-798baa7a90`'s `jul_fwd` fold to every printed digit:
  full 314.4240, lane A 251.8774, lane B 1333.7191, `rounds_matched` 834, `rounds_unmatched` 149.
  (Grounds: `c13_base` already reproduced `p8e_base`'s forward fold, 219.7648, on this frame.)
* H2. Lane B is **1333.7191 with 149 rounds in `c13_c1_2col_jf` as well** - C1's columns never reach a
  model-B split, as on forward. If it moves, something about `jul_fwd` differs and must be explained before
  the pooled number means anything.

**What I expect `jul_fwd` to read.** [P]

* base → C1, K=0: **about −0.2 s**, plausible range −0.45 to +0.05, **CI probably containing zero**.
  Reasons: the fold is 58 % the size of forward, its tail is heavier (full RMSE 314 against 220, lane B
  1334 against 1004, 3,938 unmatched rows), so the K=0 CI is wider; and the model is trained on six months,
  which gives C1's mode table no less information (it is fitted on the union frame) but gives model A fewer
  rows to learn the flag's interaction from. I expect shrinkage from forward's −0.36, not a sign change.
* base → C1, K=200: **about −0.2 s**, same range.
* lane A (the only lane C1 can move, by H2): **about −0.25 s**, against forward's −0.40.
* noise2 → C1: same sign as base → C1, CI containing zero at K=0.
* pooled [forward, jul_fwd], base → C1: **ADOPT, about −0.28 s**, because forward alone already clears
  both legs with room to spare and `jul_fwd` would have to be materially positive to undo it.
* `jul_fwd` alone has one month, so it has **no month bootstrap** (`month_bootstrap_rmse_diff` returns NaN
  for fewer than two months); the pooled one has three.

**What makes me say DO-NOT-SHIP.** [P] Any one of:

* D1. `jul_fwd` base → C1 reads **≥ −0.10 s at K=0 or at K=200** (point estimate) - i.e. less than about
  a third of forward's effect, or the wrong sign. That is the v15 pattern: one fold carries the pooled
  number and the other does not.
* D2. `jul_fwd` lane-A effect of C1 is no larger in magnitude than `jul_fwd`'s noise2 lane-A draw against
  base - i.e. the candidate does not beat the refit redraw on the one lane it acts on.
* D3. `jul_fwd` noise2 → C1 reads **≥ 0 at K=200** (the control beats the candidate once the tail is gone).
* D4. pooled base → C1 is REJECT under `prc.compare.verdict`.
* D5. H2 fails and cannot be explained as benign.

**What the best possible outcome is, stated now so it cannot drift.** [P] If none of D1-D5 fires, the
verdict is **NEEDS-MORE-FOLDS, not SHIP**. `[forward, jul_fwd]` passing both gates is *exactly* the
evidence v15's dead-column removal had (−0.13 [−0.25, −0.02], month P = 1.000) before it scored +0.0163
on the board and −0.01 on the four-fold CV. Passing on the same two folds cannot be sufficient here by
construction; SHIP needs the production fold set (`jan`, `jul`, `forward`) to agree.

