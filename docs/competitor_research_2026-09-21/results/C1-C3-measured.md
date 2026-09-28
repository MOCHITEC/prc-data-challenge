# C1 and C3, built and measured (2026-09-25)

Two candidates from `docs/competitor_research_2026-09-21/results/delay-mechanisms.md` §5, built as production
columns behind config flags and measured with the repository's own decision rule.

**Nothing was submitted. Nothing was written to `outputs/`.** The arms write only the usual experiment
records (`experiments/experiments.jsonl` and `experiments/oof_c13_*.parquet`) and a feature cache of their
own in `data/work_c1c3`, which is a separate directory from `data/work` so that no other run's cache was
disturbed.

**Code**: `src/prc/features.py` (C1), `src/prc/weather.py` (C3 and the `gust_ms` repair),
`src/prc/pipeline.py` (flags and cache keys), `configs/c13_*.yaml` (six arms).
**Tests**: `tests/test_nominal_proxy_and_wind.py` (16).
**Scripts**: `docs/competitor_research_2026-09-21/scripts/c1_c3_mutation_audit.py` (the mutation audit),
`.../c1_c3_column_checks.py` (sections K1-K4: what the columns contain and the coverage gate),
`.../c1_c3_verdicts.py` (the verdicts, computed by `prc.compare`'s own functions).
**Raw output**, so every number here can be checked without a rerun:
`results/c1-c3-verdicts.txt`, `results/c1-c3-column-checks.txt`.
**Experiment ids**: `c13_base-e6806da206`, `c13_c1-84f03c27e2`, `c13_c3wind-f0658b5d2d`,
`c13_gust-69fd3a7c9e`, `c13_noise3-acca475972`, `c13_noise4-b97278ddd3`.

Marking, as in the source note: **[S]** = measured here, with the section that produced it; **[X]** = an
inference on top of [S] facts.

---

## 0. The answer first

**C1 passes the repository's decision rule and I still would not build on it.** Against the production
column set C1 reads **-0.294 s [-0.553, -0.069] on all rows and -0.210 s [-0.381, -0.039] with the worst
200 rows removed** - both CIs excluding zero on the improving side, which is ADOPT under
`prc.compare.verdict`. But the arity-matched control arm, three columns that are a hash of the row id,
reads **-0.418 s [-0.625, -0.208] at K=200** in the same experiment. **[S] The instrument admits pure noise
on the same leg it admits C1, so the verdict is real and the evidence behind it is not.**

**C3 does not measure**: +0.070 s [-0.412, +0.542] against the base, the wrong sign, CI spanning zero.
**The gust repair does not measure either**, at -0.078 s [-0.579, +0.426], exactly as the source note
pre-registered (+0.000). Both are REJECT.

**All seven new columns clear the P8-D coverage gate**: 9 failures on the base column set, 9 with C1, 9
with C3, 9 with the gust repair, 9 with all three - **zero of them on a column built here** (§5). C3's
coverage caveat, the reason the note gave for not proposing it, turns out not to bind: `drct`'s worst
train-to-serve drop is 10.9 points at LSZH against a 20-point threshold.

**The construction requirement the note states was priced, not just obeyed**: a mode table fitted per frame
would put a different minute behind the same column name on **21 of the 60 (airport, runway) keys that
occur in the ranking frame** (§6.1).

**Mutation audit: 11 mutants, 9 killed on the first pass, 2 survivors - both repaired, 11/11 on the second
pass** (§2.1). One survivor was the exact defect this repository shipped last time: the values computed
correctly and the wrong ones handed over.

---

## 1. What was built

### 1.1 C1 - `proxy_is_nominal`

The mechanism is the source note's §2.3. NM does not observe the stand, so `AOBT_3` is an estimate and
`g = AOBT_3 - BLOCK` is its error (sd 399.3 s against 57.5 s at the arrival end, where both systems watch the
runway). At some airports the estimate is not an estimate of *this* flight at all: it is a nominal per-runway
allowance subtracted from the take-off time, which is why the offset's minute histogram has a spike 19.9x
(LTFM) and 30.1x (LEBL) its neighbouring bins while the label over the same range is smooth. On those rows
the pipeline subtracts at coefficient 1.0 a number that carries no flight-specific information.

Three columns, in `features.py`, all built from `MVT_TIME_UTC_mvt`, `AOBT_3_flt`, `ADEP_mvt` and
`RUNWAY_mvt`:

| column | dtype | what it is |
|---|---|---|
| `proxy_at_mode` | Int8 | 1 when the offset is exactly the modal whole minute of this airport x runway |
| `proxy_minus_mode_min` | Float64 | signed distance from that minute, in minutes |
| `proxy_sec_in_min` | Float64 | the seconds part of the offset; a table value is a whole number of minutes |

**The construction requirement, and how it is enforced.** The note states it: the mode table is a count over
feature columns - no label - and must be fitted **once on the union of the training and ranking frames**, not
per frame. The code is arranged so that the wrong version cannot be written by accident rather than merely
discouraged:

* `proxy_mode_table(frames)` takes a **sequence** of frames and raises `TypeError` on a single frame.
* `build_departure_features(...)` accepts a table and contains no code that fits one. Called without a table,
  C1's three columns are null - a visible no-op, never a per-frame fit.
* `build_feature_frames(train, rank)` is the only place a table is made, and it makes exactly one and hands
  the same object to both builds. `pipeline.build_features` calls that, so the per-frame version is
  unreachable without writing new code to reach it.
* The table is sorted on its keys before it is returned, so it is a value and not an order.

**It is not a fold artefact and not a label.** The table counts a clock difference over two columns that are
present on every scored row; `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` are not read, directly or through a
group statistic, and `bs = BLOCK - SCHED` appears nowhere. Fitting it on the ranking frame is the same
operation the scorer could run on its own copy of the data.

### 1.2 C3 - wind direction relative to the assigned runway

`drct` is fetched by `weather.py:23` and has always been dropped in the hourly aggregation, so wind
*direction* has never reached the model - although the wind is what chooses the runway configuration, and the
configuration rewrites every taxi route. `wind_dir: true` keeps the bearing through the aggregation (last
observation of the hour, sorted inside the aggregation rather than trusting the frame's order), and
`join_weather` turns it into four columns and drops the raw bearing again:

| column | what it is |
|---|---|
| `wind_dir_deg` | the bearing itself, level hour |
| `headwind_kt` | wind resolved along the runway; negative is a tailwind |
| `crosswind_kt` | the unsigned across-runway component |
| `rwy_wind_misalign` | `1 - cos(drct - runway heading)`: 0 straight down the runway, 2 straight against |

The heading comes from `RUNWAY_mvt` by a parser that reads the designator as tens of degrees and returns
**null** - not zero - for anything that does not parse, because 0 is a legal heading and a wrong angle is
worse than a missing one. The raw bearing is deliberately not a model column: 270 degrees means a headwind at
one airport and a tailwind at the next.

**The `gust_ms` hygiene repair, ridden along and measured apart.** METAR omits the gust group when there is
no gust, so `gust_ms` is null on 96.6% of scored rows and the column conflates "the wind was steady" with "we
did not observe". `gust_repair: true` adds `gust_filled` (the wind speed when there is no gust group) and
`gust_excess` (the part above the steady wind, zero rather than null on a steady hour), plus the `_prev1h`
copies the join makes of everything in the hourly table, which have the same defect and so need the same
repair. `gust_ms` itself is left alone. The note measures this at **+0.000** and calls it hygiene; it is
therefore given **its own arm** (§3) so that it cannot be read as part of C3's number.

### 1.3 Everything is behind a flag that defaults to current behaviour

`wind_dir: false` and `gust_repair: false` are the defaults, `hourly_weather_table` emits exactly the columns
it always has with them off, and C1's three columns are `CANDIDATE_COLS`, hidden from the model unless a
recipe names them. `lgb_deterministic` stays `True`. Both flags are in `weather_cache_path` (so two recipes
never share one hourly file) and in `cache_fingerprint` (so a frame built without a column is never handed to
a run that asked for it) - the trap the pipeline's own docstring calls the highest-probability way to waste a
day.

---

## 2. The tests, and the mutation audit

`tests/test_nominal_proxy_and_wind.py`, 16 tests, all passing; the full suite is **149 passed** (it was 133
before these). Each test is written against a named defect rather than against the code as written:

| test | what it kills |
|---|---|
| `test_the_mode_table_is_fitted_on_the_union` | the table fitted per frame - the note's explicit requirement |
| `test_the_pair_builder_scores_both_frames_on_one_table` | the same, asserted where the table is *used*, not where it is made |
| `test_the_mode_is_per_runway_not_per_airport` | a table keyed on the airport, which loses LTFM's second runway group |
| `test_the_three_columns_are_what_they_say` | a flipped comparison, or the raw minute handed over as the flag |
| `test_the_columns_reach_the_model_when_asked` | columns computed and then not shown (and shown when they should be hidden) |
| `test_without_a_table_the_columns_are_null` | a per-frame table invented when none was passed |
| `test_proxy_mode_table_refuses_a_single_frame` | the sequence argument called with one frame |
| `test_c1_is_blind_to_the_blanked_label_columns` | a label reaching the table or the columns |
| `test_runway_heading_parses_the_designator` | the designator read as degrees instead of tens |
| `test_wind_is_resolved_onto_the_assigned_runway` | cos/sin swapped, or the headwind sign flipped |
| `test_a_missing_bearing_stays_missing` | a null bearing silently becoming a zero angle |
| `test_the_raw_bearing_never_reaches_the_model` | the raw bearing left in the frame as a model column |
| `test_the_new_weather_columns_are_off_by_default` | a default that changes every archived recipe |
| `test_a_recipe_that_says_nothing_gets_the_old_frame` | the same, at the level of the recipe `main` reads |
| `test_the_gust_repair_separates_calm_from_unseen` | `gust_excess` that is not a difference |
| `test_every_weather_flag_is_in_the_cache_key` | two recipes sharing one hourly file or one fingerprint |

The pre-existing `test_parity_same_features_for_training_and_ranking_views` covers the new columns for free:
it compares every numeric column between a labelled build and a blanked build.

### 2.1 The mutation audit, and the two mutants that survived it

`scripts/c1_c3_mutation_audit.py` copies the source tree, applies one defect at a time as a textual patch,
runs the suite against the copy and reports whether anything failed. Eleven mutants, each one a defect a
reviewer could plausibly have written.

**First pass: 9 of 11 killed, and the two survivors were the two that matter.** [S]

* `C1-per-frame-table` **survived**. The union test exercised `proxy_mode_table`, which is where the table is
  *made*; the mutant changed `build_feature_frames`, which is where it is *used*. That is the P8 defect in
  its general form - the values are computed correctly and the wrong ones are handed over - and the suite did
  not see it. Repaired by `test_the_pair_builder_scores_both_frames_on_one_table`, which asserts the end
  state: training on a 10-minute allowance and a smaller ranking month on a 20-minute one, every ranking row
  must sit 10 minutes from the mode and none of them at it. A per-frame fit flags all of them.
* `C3-flags-default-on` **survived**. Nothing tested how `pipeline.main` reads a recipe, so a default flipped
  from `False` to `True` - which silently changes every archived recipe in the repository - passed. Repaired
  by extracting `pipeline.weather_flags(spec)` and testing it.

**Second pass: 11 of 11 killed.** [S] The audit is in the repository and reruns in about a minute.

---

## 3. How this was measured, and what the instrument can see

### 3.1 The arms

Six arms, one feature frame, the production recipe. `configs/c13_base.yaml` is `configs/main_gap.yaml`
verbatim - `es_holdout_encodings: true`, `unmatched_sched_reparam` at LIRF, `lobt_bound`, `n_seeds: 3`,
model B at 127 leaves, `lgb_deterministic` untouched - with two differences: `folds: [forward]`, which is the
repository's screening condition (`docs/features_log.md`: forward fold + `--skip-predict`, weather on, three
seeds), and an `exclude_features` list that hides the eight new weather columns. **The archived
`oof_main_gap-62dd7c6e88` was not reused**: the base arm was built here, on the same code and the same frame
as every candidate arm.

All six arms are cut from **one** feature frame (`data/work_c1c3`, built with `wind_dir` and `gust_repair`
on), so an arm differs from another by which columns the model is shown and by nothing else - not by a
rebuild, not by a seed, not by a row. That also settles the cache question the brief raises: one frame, one
fingerprint, no arm rebuilding anything.

| arm | columns added to the base set | what it is for |
|---|---|---|
| `c13_base` | - | A of every comparison |
| `c13_noise3` | `noise_control`, `_2`, `_3` | the price of three signal-free columns |
| `c13_c1` | `proxy_at_mode`, `proxy_minus_mode_min`, `proxy_sec_in_min` | C1 |
| `c13_noise4` | the four `noise_control` columns | the price of four signal-free columns |
| `c13_c3wind` | `wind_dir_deg`, `headwind_kt`, `crosswind_kt`, `rwy_wind_misalign` | C3 |
| `c13_gust` | `gust_filled`, `gust_excess` and their `_prev1h` copies | the hygiene rider, on its own |

C1 and C3 are measured **separately**, each against the base and against its arity-matched control, before
any combined arm - because a null measured on a combined arm cannot be attributed.

### 3.2 This is a different instrument from the note's screen, and the 0.57 does not travel

The source note's §4 screen fits the **OOF residual** with a small LightGBM on four base columns plus the
arm's four, and its calibration says it reads a true effect at **0.57x** (it read -1.83 for
`stand_arrival_gap`, whose true full-model gain on the same rows is -3.21), with a control-arm seed sd of
0.004.

What is measured here is the full model itself: two complete CV runs differing in one column set, compared by
`prc.compare`. That is the quantity the 0.57 was calibrated *against*, not the quantity it converts. **[X]
Multiplying anything below by 0.57 would apply the calibration twice.** The one conversion applied is the
repository's own `dBoard/dOOF = 0.747` (`model-a-ceiling.md` §8.2, **n = 1**), and never the OOF-to-board
*level*, which is unstable (+0.896 for v13, +1.984 for v14).

Running the note's own chain forward gives what this instrument should expect to see if the note is right:

| candidate | screen reading | /0.57 = lane A | x0.758 = pooled OOF | x0.747 = board |
|---|---:|---:|---:|---:|
| C1 | -0.071 | -0.12 | **-0.09** | -0.07 |
| C3 | -0.045 | -0.08 | **-0.06** | -0.045 |

So the number to look for in §4 is about **-0.09 s** for C1 and **-0.06 s** for C3 on the forward fold's
pooled OOF RMSE. One caveat on the arithmetic, stated rather than buried: 0.758 and 0.747 were measured on
the **three-fold pooled OOF** (672,428 rows), and these arms are the forward fold alone (328,009 rows), so
the conversion is being used outside the split it was measured on. **[X]** It is the right order of
magnitude, not a calibrated transfer.

### 3.3 What this instrument can resolve - the part that decides the verdict

Three numbers bound it, and they are not the same number:

1. **Run-to-run spread of an identical config**, three seeds, different processes: **0.017-0.028 s**
   (`results/determinism-cost.md` §3). That is the floor - the point below which two runs of the *same* arm
   disagree.
2. **The per-column noise floor: +/-0.5 s** (`plans/EXECUTION-ORDER.md:68`, and `docs/features_log.md`'s
   three-loss analysis, where `noise_control` cost +0.53 against candidates at +0.39 to +0.50). This is not
   sampling error: `feature_fraction` is 0.8, so adding *any* column changes which columns every split
   samples from, and the model is redrawn. It is the reason every candidate here is read against an
   arity-matched noise arm and not against the base alone.
3. **The paired-bootstrap CI** that the decision rule reads. It is much narrower than (2), because it
   resamples rows with both prediction vectors held fixed - it prices sampling error and is blind to the
   refit lottery entirely.

**[X] The consequence, stated before the numbers were seen.** Both candidates are pre-registered at -0.09 and
-0.06 s of OOF, which is **3-5x the run-to-run spread and 5-10x below the per-column noise floor**. A
difference of that size is inside the lottery that adding a column draws, so the decision rule can return
"both CIs exclude zero" for a column that carries nothing - which is exactly what §4 checks by running the
rule on the noise arms too. An instrument whose resolution is coarser than the effect it is aimed at cannot
report "absent"; it can only report **unmeasurable**. Those are different claims, and this note keeps them
apart.

---

## 4. Results

### 4.1 The arms, as they landed

Forward fold, `n = 328,009` pooled rows (`325,004` lane A / matched, `3,005` lane B). Three seeds,
`lgb_deterministic: true`. **[S]** `experiments/experiments.jsonl`, arms `c13_*`.

| arm | model-visible numerics | pooled RMSE | lane A | lane B | wall clock |
|---|---:|---:|---:|---:|---:|
| `c13_c1` | 93 | **219.4704** | 198.22 | 1004.04 | 724 s |
| `c13_gust` | 94 | 219.6865 | 198.45 | 1004.26 | 880 s |
| `c13_c3wind` | 94 | 219.8352 | 198.51 | 1006.59 | 751 s |
| `c13_noise3` | 93 | 220.2581 | 198.36 | 1019.80 | 816 s |
| `c13_noise4` | 94 | 220.3537 | 198.65 | 1015.89 | 657 s |
| `c13_base` | 90 | 219.7648 | 198.55 | 1004.04 | 835 s |

**[S] Read the two control arms first.** Three signal-free columns and four signal-free columns land
**0.79 s and 0.88 s worse** than C1, and 0.52 s worse than C3. Nothing in a noise column can explain a
0.8 s difference: this is the `feature_fraction: 0.8` redraw that `docs/features_log.md` priced at
+/-0.5 s per column, seen again, and it is four to nine times the effects being tested for.

### 4.2 The verdicts, against the arity-matched controls

`prc.compare`'s own functions, 500 bootstrap resamples, the rule unchanged
(`scripts/c1_c3_verdicts.py`). Negative = B better.

**C1 against three noise columns** [S]

| dropped | A (noise3) | B (C1) | diff | CI95 | P(B better) |
|---:|---:|---:|---:|---|---:|
| 0 | 220.2581 | 219.4704 | **-0.788** | **[-2.532, +0.526]** | 0.820 |
| 20 | 204.1882 | 204.2512 | +0.063 | [-0.291, +0.422] | 0.352 |
| **200** | 193.9949 | 194.2431 | **+0.248** | **[+0.017, +0.513]** | 0.016 |
| 2000 | 174.4110 | 174.5244 | +0.113 | [+0.003, +0.256] | 0.024 |

month bootstrap (2 months) -0.788 [-1.422, -0.135]; lane A **-0.138 s**.
**VERDICT: REJECT** - the full-data CI contains zero, and the sign reverses once twenty rows are gone:
at K=200 C1 is *worse* by +0.248 [+0.017, +0.513]. The -0.79 s is about twenty rows.

**C3 against four noise columns** [S]

| dropped | A (noise4) | B (C3) | diff | CI95 | P(B better) |
|---:|---:|---:|---:|---|---:|
| 0 | 220.3537 | 219.8352 | **-0.518** | **[-1.692, +0.360]** | 0.804 |
| 20 | 204.6571 | 204.6757 | +0.019 | [-0.235, +0.275] | 0.448 |
| **200** | 194.5031 | 194.3935 | **-0.110** | **[-0.290, +0.058]** | 0.888 |
| 2000 | 174.6002 | 174.6305 | +0.030 | [-0.078, +0.141] | 0.282 |

month bootstrap -0.518 [-1.356, +0.339]; lane A **-0.142 s**.
**VERDICT: REJECT** - neither CI excludes zero.

**The gust repair against four noise columns** [S]

| dropped | A (noise4) | B (gust) | diff | CI95 | P(B better) |
|---:|---:|---:|---:|---|---:|
| 0 | 220.3537 | 219.6865 | **-0.667** | **[-1.737, +0.171]** | 0.924 |
| 20 | 204.6228 | 204.5110 | -0.112 | [-0.368, +0.163] | 0.800 |
| **200** | 194.4391 | 194.3302 | **-0.109** | **[-0.304, +0.087]** | 0.898 |
| 2000 | 174.6181 | 174.6731 | +0.055 | [-0.045, +0.171] | 0.114 |

month bootstrap -0.667 [-1.247, -0.072]; lane A -0.199 s.
**VERDICT: REJECT** - as the note said it would: hygiene, not a gain.

### 4.3 The verdicts against the base arm

This is the comparison the decision rule is written about: A = the production column set, B = the same
plus the candidate. [S] `scripts/c1_c3_verdicts.py`.

| pair | full-data diff | CI95 | K=200 diff | CI95 | month bootstrap | verdict |
|---|---:|---|---:|---|---|---|
| base -> **C1** | **-0.294** | **[-0.553, -0.069]** | **-0.210** | **[-0.381, -0.039]** | -0.294 [-0.501, -0.093] | **ADOPT** |
| base -> C3 | +0.070 | [-0.412, +0.542] | -0.053 | [-0.211, +0.122] | +0.070 [-0.352, +0.502] | REJECT |
| base -> gust repair | -0.078 | [-0.579, +0.426] | -0.106 | [-0.263, +0.039] | -0.078 [-0.243, +0.090] | REJECT |
| base -> noise3 (control) | +0.493 | [-0.778, +2.235] | **-0.418** | **[-0.625, -0.208]** | +0.493 [-0.366, +1.329] | REJECT |
| base -> noise4 (control) | +0.589 | [-0.503, +1.920] | +0.000 | [-0.218, +0.213] | +0.589 [+0.163, +1.004] | REJECT |

**C1 passes the repository's rule outright.** Both CIs exclude zero on the improving side, P(B better) is
0.990 or better at all four drop levels, the effect survives every drop level (-0.294 / -0.348 / -0.210 /
-0.143 at K = 0 / 20 / 200 / 2000), the month bootstrap agrees on both scored months, and lane A moves
**-0.329 s**. At `dBoard/dOOF = 0.747` that prices at **-0.220 s of board**, about three times the
pre-registered -0.07.

**And the control arm says the rule cannot be trusted at this size.** Look at the last two rows.
Three signal-free columns read **+0.493 s on all rows and -0.418 s [-0.625, -0.208] once 200 rows are
gone** - a *significant improvement* from columns that are a hash of the row id. Four of them read +0.589
and +0.000. **[S]** A statistic that three hash columns move by 0.4-0.6 s in either direction cannot
resolve a 0.29 s column, and the K=200 CI excluding zero is not, at this scale, evidence of signal: the
control did it too, in the improving direction.

**One reading that is not available.** The drop-200 set is computed per pair - the rows removed are those
where *either* model of that pair errs most - so the retained rows differ between comparisons and the K=200
numbers are **not transitive**. base->C1 (-0.210), noise3->C1 (+0.248) and base->noise3 (-0.418) are three
different experiments on three different row sets, and no ordering can be read off them. Each pair is its
own measurement, and that is how they are reported here.

### 4.4 What the numbers say about the instrument, not about the features

**[S] Every full-data difference in §4.2 is between -0.5 and -0.8 s, and every one of them collapses to
within +/-0.12 s by K=20.** Twenty rows out of 328,009 carry the whole apparent difference between any two
of these arms. That is the phenomenon `prc.compare` was written for, and it is why the repository requires
both CIs.

**[X] The consequence for what can be concluded.** The quantity these arms were aimed at is -0.09 s (C1)
and -0.06 s (C3) of pooled OOF. The measured lane-A differences are -0.138 s and -0.142 s - the right
order, the right sign - but they sit inside a full-data CI of +/-1.5 s and beside a control-arm redraw of
0.5-0.9 s. **The instrument's resolution is about ten times the effect.** A reading of -0.14 s of lane A
from an instrument whose arm-to-arm scatter is 0.8 s is not evidence of -0.14 s; it is consistent with
anything between -1 s and +0.5 s.

---

## 5. The coverage gate (P8-D)

`src/prc/coverage.py` run five times on the same pair of frames (`scripts/c1_c3_column_checks.py` K4),
once per column set. The base arm is the control: **this gate already fails on production's own columns**,
so a raw count means nothing without it.

| column set | model-visible columns | failures | failures on a NEW column |
|---|---:|---:|---:|
| base (control) | 100 | **9** | 0 |
| base + C1 | 103 | **9** | **0** |
| base + C3 wind | 104 | **9** | **0** |
| base + gust repair | 104 | **9** | **0** |
| base + C1 + C3 + gust | 111 | **9** | **0** |

**[S] The failure count is 9 in every arm, and not one failure is on a column built here.** The nine are
pre-existing: five `DEAD` constants (`runway_null`, `precip_mm`, `temp_x_precip` and the two `_prev1h`
copies - P8-D's finding, unchanged) and four `NONZERO` collapses on `dewp_spread_c` and its previous-hour
copy at LEMD and LSZH in January 2026.

**C3 clears the gate the note asked about, and the reason is worth stating.** `drct` is null on 8.5% of
*rows* (the note's 10.0% is over METAR *observations*; rows are not distributed like observations) and the
worst airport is LSZH, exactly as the note said - but the number that matters to the gate is the
train-to-serve *drop*, not the level:

| airport | train-2025 null | 2026-01 null | 2026-07 null | worst non-null drop |
|---|---:|---:|---:|---:|
| **LSZH** | **29.8 %** | **40.7 %** | 29.0 % | **10.9 pts** |
| EDDF | 13.9 % | 9.5 % | 18.1 % | 4.2 pts |
| LEMD | 14.6 % | 15.8 % | 9.9 % | 1.2 pts |
| LFPG | 3.8 % | 1.0 % | 9.1 % | 5.3 pts |
| the other six | 2.5-14.9 % | - | - | < 5 pts |

**[S] The worst drop anywhere is 10.9 points at LSZH in January, against a 20-point threshold.** The feed
is equally patchy on both sides of the join, which is the condition the gate exists to check. [X] That is
a real answer to the note's caveat rather than a waiver: the coverage objection to C3 does not survive
measurement.

---

## 6. What the columns contain

### 6.1 C1's mode table, and the construction requirement priced

`scripts/c1_c3_column_checks.py` K1, on the real frames. **[S]**

* **68** (airport, runway) keys on the union; **60** of them occur in the ranking frame.
* **A table fitted per frame would put a different minute behind the same column name on 21 of those 60
  keys - 35 % of the serving keys.** Examples: EGLL 09L (union 16, rank-only 27), EDDF "NA" (10 vs 30),
  LFPG 08R (14 vs 19), LSZH "H" (-46 vs 16).
* Fitted on the training frame alone the table differs from the union on **3** of 68 keys - the union is
  dominated by the 2.09M training departures against 344k scored ones. [X] So the practical damage of the
  wrong fit is on the *ranking* side, which is exactly the side nobody can check after the fact.

### 6.2 What C1's three columns contain

| column | train null | train non-zero | rank null | rank non-zero |
|---|---:|---:|---:|---:|
| `proxy_at_mode` | 1.08 % | **19.88 %** | 1.53 % | **19.85 %** |
| `proxy_minus_mode_min` | 1.08 % | 79.04 % | 1.53 % | 78.62 % |
| `proxy_sec_in_min` | 1.08 % | 93.87 % | 1.53 % | 93.44 % |

The flag fires on **19.9 % of training rows and 19.9 % of scored rows** - the same share on both sides,
which is what a parity-safe column should look like. The null share is the `mvt_minus_aobt3` null rate
(lane B), not a defect.

Per airport, the spike the mechanism predicts is there and it is large where the note said it would be:

| ADEP | modal minute | rows | share | spike vs neighbours | `proxy_at_mode` share |
|---|---:|---:|---:|---:|---:|
| **LEBL** | 20 | 47,580 | 26.8 % | **34.4x** | 27.0 % |
| **LTFM** | 19 | 80,383 | 29.8 % | **14.0x** | 38.5 % |
| EDDM | 11 | 36,971 | 22.2 % | 4.4x | 27.4 % |
| LSZH | 11 | 25,568 | 19.3 % | 2.8x | 24.1 % |
| LEMD | 16 | 38,091 | 18.0 % | 2.6x | 22.0 % |
| LIRF / EDDF / EHAM / LFPG | 13-14 | - | 11.1-13.8 % | 1.6-1.8x | 13.0-15.4 % |
| EGLL | 22 | 16,111 | 6.8 % | 1.1x | 6.9 % |

[S] The mechanism reproduces on the full training frame (the note's §2 numbers - LEBL 30.1x, LTFM 19.9x -
were measured on the pooled OOF rows with a different neighbour window, so the two are close but not the
same statistic).

**One column is not what the mechanism describes, and it should be said.** `proxy_sec_in_min` is non-zero
on **93.9 %** of rows. A nominal allowance subtracted from a take-off stamp would leave the *seconds of the
take-off stamp* in this column, not a zero - the whole-minute property the note cites belongs to `AOBT_3`,
not to the difference. [X] So the third column is mostly "what second of the minute did this aircraft lift
off", which is a plausible nuisance regressor and not a statement about the stamp's provenance. The
mechanism is carried by the first two columns; the third rode along.

### 6.3 C3's columns

| column | train null | train non-zero | rank null | rank non-zero |
|---|---:|---:|---:|---:|
| `wind_dir_deg` | 8.49 % | 90.69 % | 8.52 % | 91.30 % |
| `headwind_kt` | 8.50 % | 90.68 % | 8.54 % | 91.28 % |
| `crosswind_kt` | 8.50 % | 85.15 % | 8.54 % | 86.05 % |
| `rwy_wind_misalign` | 8.50 % | 85.32 % | 8.54 % | 86.11 % |
| `gust_filled` | 0.02 % | 99.16 % | 0.01 % | 99.81 % |
| `gust_excess` | 0.02 % | **2.29 %** | 0.01 % | **3.38 %** |

`gust_excess` is non-zero on 2.3 % of training rows and 3.4 % of scored rows - it is above the gate's
0.1 % near-dead tier, and its rarity **is** the repair working: on the other 97 % the wind was steady and
the answer is a measured zero rather than a null that LightGBM would turn into a zero anyway without
anyone knowing which it meant.

---

## 7. Verdicts, and what I would do

### 7.1 C1 - `proxy_is_nominal`

> **Superseded, 2026-09-28: see `C1-two-column-remeasured.md`.** Step (c) below was run. The two-column
> build (`proxy_at_mode`, `proxy_minus_mode_min`) is stronger on every reading, and the argument for
> "UNMEASURABLE" given in this section does not survive: six control pairs show the rule fires on 0 of 6
> pure-noise comparisons, so the claim that it "would also admit `noise_control`" is true of the K=200 leg
> only, not of the rule. The verdict is still DO-NOT-SHIP, for a different and newer reason.

**Verdict: ADOPT by the rule, UNMEASURABLE by the instrument. I would not ship it on this evidence.**

The rule fires: -0.294 [-0.553, -0.069] on all rows and -0.210 [-0.381, -0.039] at K=200, both CIs
excluding zero on the improving side, month bootstrap agreeing, -0.220 s of board at 0.747. The mechanism
is real and reproduces (LEBL 34.4x, LTFM 14.0x), the construction is parity-safe and union-fitted, the
column set clears the coverage gate, and the measured effect is three times the pre-registration rather
than a hair under it.

**What stops me recommending a build anyway** is the control arm in the same table: three columns that are
a hash of the row id read **-0.418 s [-0.625, -0.208] at K=200**, which is *larger* than C1's -0.210 and on
the improving side. The rule was calibrated on `stand_arrival_gap`, an effect of -3.6 s, where the redraw
was 15 % of the signal. Here the redraw is bigger than the signal. **[X] A rule that admits C1 on this
evidence would also admit `noise_control` on the K=200 leg, and this repository has already shipped
`noise_control` once (v6, 350.62 against v4's 305.03).**

What would change my mind, in order of cost: (a) repeat the base and C1 arms on a second seed set - the
question is whether -0.294 reproduces or wanders by half a second, and that is two runs; (b) run both on
all three folds, where the pooled n triples; (c) drop `proxy_sec_in_min`, which §6.2 shows is measuring the
take-off stamp's seconds rather than the mechanism, and see whether the two-column version holds the effect.

### 7.2 C3 - wind direction relative to the assigned runway

**Verdict: REJECT, and on this evidence it is a clean one.** +0.070 [-0.412, +0.542] against the base,
-0.110 [-0.290, +0.058] against its noise control at K=200; the full-data CI contains zero on every
comparison and the point estimate has the wrong sign against the base. The pre-registered band was
-0.00 to -0.06 s of board and the measurement cannot distinguish that from zero, so **UNMEASURABLE is the
honest label for the pre-registered effect** and REJECT is the verdict for the column set. The coverage
objection turned out not to apply (§5), which is worth recording because it removes the reason the note
gave for not proposing C3 for production - the reason it should not go to production now is that it does
not measure.

### 7.3 The `gust_ms` hygiene repair

**Verdict: REJECT as a gain, exactly as pre-registered (+0.000 in the note, -0.078 [-0.579, +0.426] here).**
It is still correct: `gust_ms` is null on 96.6 % of scored rows and the null means "steady", and a column
that cannot tell "steady" from "unobserved" is a column whose splits mean two things. [X] If a production
build happens for another reason, the repair should ride along in it. It should never be the reason for one.

### 7.4 Is either worth a production build? My reading

**No - not on today's evidence, and the reason is the instrument rather than the candidates.**

A production build is three folds plus a refit plus a submission slot. What it would buy, at the very best
reading of these numbers, is -0.22 s of board from C1 and nothing from C3. The base-to-control gap in this
same experiment is 0.49-0.59 s. [X] Spending a submission on a change whose measured effect is half its own
instrument's noise is the trade this repository has already lost twice (`queue_ahead`, `bank_position`,
`stand_rwy_share` - all three "worse than base", all three inside the noise once the control was built).

**What I would do instead, in this order.**

1. **Keep the code.** It is behind flags that default to off, it is tested, the mutation audit kills every
   mutant, and it costs nothing until someone turns it on.
2. **If there is cheap compute, buy resolution, not another candidate.** Two more seeds on `c13_base` and
   `c13_c1` answer the only question that matters about C1 - whether -0.294 is a number or a draw - and
   cost 25 minutes. That is a far better use of a slot than measuring a fourth column against the same
   blunt instrument.
3. **If C1 is ever built for production, build it without `proxy_sec_in_min`** (§6.2) and with the gust
   repair riding along (§7.3).
4. **Do not treat the source note's §4 screen as refuted.** It predicted C1 > C3 > gust in that order and
   that is exactly the order measured here (-0.294 / +0.070 / -0.078 against base). [X] The screen ranked
   these three candidates correctly; what it could not do - and what this measurement also cannot do - is
   tell a small real effect from zero.
