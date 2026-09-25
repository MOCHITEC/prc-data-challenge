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

Marking, as in the source note: **[S]** = measured here, with the section that produced it; **[X]** = an
inference on top of [S] facts.

---

## 0. The answer first

TBD-SUMMARY

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

TBD-RESULTS

---

## 5. The coverage gate (P8-D)

TBD-COVERAGE

---

## 6. What the columns contain

TBD-COLUMNS

---

## 7. Verdicts, and what I would do

TBD-VERDICTS
