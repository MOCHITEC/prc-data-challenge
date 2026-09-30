# P8-D — the coverage-parity gate, built and run

**Executed:** 2026-09-25, branch `claude/model-b-and-arrival-features`.
**Plan:** `docs/competitor_research_2026-09-21/plans/P8-weather-and-coverage-gate.md` §5.
**Status: SHIP the module and the tests. Do NOT flip `coverage_gate` on for production yet** — the gate
fails today, on five columns P8-B is meant to remove. That is the gate working, not the gate misconfigured.

**Marking:** `[S]` = measured here by a reproducible command, or read from a named file at a named line.
`[X]` = inference, never dressed up as a measurement.

---

## 0. What was built

| thing | where |
|---|---|
| the gate | `src/prc/coverage.py` (new, 390 lines incl. the reasoning) |
| tests | `tests/test_coverage.py` (new, 12 tests, one per rule plus a control per rule) |
| wiring | `src/prc/pipeline.py`, behind `coverage_gate:` **defaulting to false** |

Run it read-only:

```
.venv/bin/python -m prc.coverage --config configs/main_gap.yaml --no-strict
```

It takes about 10 s on the real frames and writes nothing unless `--out` is given. Nothing was written to
`outputs/` in this session.

---

## 1. Ordering: D was taken first, and §5 confirms that was right

The tasking asked me to say whether D-first survives reading §5. It does, for a reason §5 states and one it
does not.

- §5 states it: the gate is read-only until the last commit, and *"its output changes what P8-B does"*
  (§7.1). It arrives at P8-B's mandate — the dead columns — from a second direction.
- The reason it does not state: the gate is also the **acceptance instrument for P8-B**. A `wxcodes` repair
  is supposed to remove five constant columns and add live ones. With the gate built first, "did the repair
  actually do that" is one command with a machine-checkable answer, rather than a hand-written assertion of
  the kind this repo has twice shipped as a silent no-op. B is measured *through* D, as the tasking put it.

---

## 2. What the gate does, and the two places it departs from kind-mango's

`[S]` The competitor's rule (`/tmp/prc-kind-mango/src/check_coverage.py:56-73`, read via
`22-repo-kind-mango.md`) fails a build when, per column × airport × scored month:

- the column is absent from the 2026 dump;
- the **non-null** share drops more than 20 points against full 2025;
- the **non-zero** share drops more than 20 points against full 2025 **and** against the month-matched 2025
  month. The month match is the good detail — it absorbs seasonality — and it is kept verbatim.

Ours adds two things and changes one threshold.

### 2.1 Change 1 — DEAD: a column that was never alive must fail

`[S]` `precip_mm` is non-null on 100 % of rows and the literal `0.00` on 100 % of rows, in **both** years.
Every drop the competitor's rule computes on it is exactly **0 points**, so their gate **passes our single
worst defect**. Their rule is built to catch a feed that decays and is structurally blind to one that was
never alive. The extension:

> **DEAD** — non-zero share below the fail threshold on the **training** side → fail as dead weight.

### 2.2 Change 2 — rises are reported, never failed

`[S]` The competitor logged but never caught `eobt1_sched` at LFPG going 35.7 → 61.6 % non-zero
(`22-repo-kind-mango.md:566`), a one-sided test's blind spot. Rises are recorded here as `RISE` warnings.
`[X]` They are not failures on purpose: a rise is not the NaN→0 failure mode, and failing on one would have
us fighting ordinary weather variation — `deicing_gate` at LSZH rising in January 2026 *is* the real world,
and it is the entire reason weather might help at all.

### 2.3 The threshold split, which is a correction to the plan and not a measurement

**The plan §5.3 contradicts itself and I had to choose.** Its rule is written as

> `DEAD: non-zero share < 0.1 % on the TRAINING side -> fail as dead weight`

and its own table immediately below marks `stand_null` (0.0013 %) and `diverted` (0.0993 %) as *"near-dead,
warn"* and *"borderline, warn"*, then §5.5 states the expected finding as **five** dead columns. `[S]` Those
statements cannot all hold: 0.0993 < 0.1, so the rule as literally written fails `diverted` and produces
**seven**, not five.

Resolution taken, and it is a judgement `[X]`: failing a build on `diverted` would be wrong — 0.0993 % of
2,085,047 rows is about 2,070 rows that genuinely carry information, just rarely. So the fail tier is set
where §5.5's acceptance criterion is, at columns that are *identically* zero:

| tier | threshold | action | what it catches today |
|---|---|---|---|
| `DEAD` | non-zero < **0.001 %** on the training side | **fail** | the five identically-zero columns |
| `NEAR_DEAD` | non-zero < **0.1 %** (the plan's literal number) | report | `stand_null`, `diverted` |

Both thresholds are parameters of `evaluate()`, so the plan's single-tier reading is one keyword argument
away. A test pins the split (`test_near_dead_is_reported_not_failed`), and the mutation audit in §5 confirms
that restoring the plan's literal 0.1 % as the *fail* tier breaks the real-frame acceptance test.

### 2.4 One semantic decision worth stating

`[S]` The non-zero share counts **a null as a zero**. That is not tidiness: it is the failure mechanism.
LightGBM's `NumericalDecision` converts NaN to 0.0 when the split's feature had no missing value in training
(`22-repo-kind-mango.md:353-358`), which is exactly why the competitor's dead feed cost −13.14 s while
median-imputing it instead cost **+58 s**. A gate that computed the non-zero share over the non-null rows
only would report `nan` for a column that has gone fully null and say nothing about what the model will
actually do with it. `test_a_null_and_an_explicit_zero_are_the_same_thing_to_the_gate` pins this, and it was
the last mutant to be killed.

### 2.5 Column set and waivers

`[S]` The gate reads `visible_numeric_columns(frame, cfg)` + `CATEGORICALS` — *what the model was given, not
how wide the frame is* (`train.py:526-528`). On `configs/main_gap.yaml` that is **100 columns (90 numeric +
10 categorical)**, against the frame's 137. The plan measured 99 (89 + 10); the difference is
`stand_arrival_gap`, which `main_gap.yaml` now names in `include_candidates`. `[S]` A candidate column not
named by the recipe is invisible to the gate, and `test_gate_reads_the_columns_the_model_sees_not_the_frame_width`
proves it by checking that the dead-by-construction `noise_control` is silent until a recipe includes it.

`[S]` Waivers are a `coverage_allow:` **mapping**, not a list: the key names what is excused
(`RULE:column`, or `RULE:column@AIRPORT@PERIOD`, or a bare column name) and the value is the written reason.
An empty reason raises. This is deliberate — the competitor marked whole column families "informational" and
then shipped the LIRF head still reading the known-bad feed (`22-repo-kind-mango.md:388-392`). A waiver
mechanism is legitimate; a silent one is not.

---

## 3. What it finds on today's production matrix `[S]`

`.venv/bin/python -m prc.coverage --config configs/main_gap.yaml` on `data/work/feat_train.parquet`
(2,085,047 rows, all of 2025) and `data/work/feat_rank.parquet` (344,841 rows, Jan + Jul 2026):

**9 failures, 4 report-only, 0 waived.**

```
DEAD     runway_null                        non-zero on 0.0000% of 2,085,047 training rows
DEAD     precip_mm                          non-zero on 0.0000% of 2,085,047 training rows
DEAD     precip_mm_prev1h                   non-zero on 0.0000% of 2,085,047 training rows
DEAD     temp_x_precip                      non-zero on 0.0000% of 2,085,047 training rows
DEAD     temp_x_precip_prev1h               non-zero on 0.0000% of 2,085,047 training rows
NONZERO  dewp_spread_c        @LEMD 202601  96.7% (all-2025) / 93.7% (202501) -> 72.5%:  -24.2 / -21.2 pts
NONZERO  dewp_spread_c        @LSZH 202601  93.2% (all-2025) / 95.9% (202501) -> 70.4%:  -22.8 / -25.6 pts
NONZERO  dewp_spread_c_prev1h @LEMD 202601  96.2% (all-2025) / 93.4% (202501) -> 70.6%:  -25.7 / -22.9 pts
NONZERO  dewp_spread_c_prev1h @LSZH 202601  92.0% (all-2025) / 95.5% (202501) -> 69.5%:  -22.5 / -26.0 pts

RISE      lobt_minus_eobt1      @LFPG 202607  23.4% (202507) -> 45.6%:  +22.2 pts month-matched
RISE      lobt_minus_eobt1_slog @LFPG 202607  23.4% (202507) -> 45.6%:  +22.2 pts month-matched
NEAR_DEAD stand_null                          non-zero on 0.0013% of training rows
NEAR_DEAD diverted                            non-zero on 0.0993% of training rows
```

Against the plan's §5.5 prediction, point by point:

1. **Four strict `dewp_spread_c` failures at LEMD/LSZH in January 2026 — reproduced, magnitudes within
   0.6 points of the plan's and within 1–2 points of kind-mango's own four** (`22-repo-kind-mango.md:536-538`).
   `[X]` Three independent implementations on the same IEM feed now agree, which makes a pipeline bug in any
   of them very unlikely; the physical reading (temperature and dewpoint reported equal in saturated air, so
   the spread is exactly 0) stays unverified. **No action beyond recording it.**
2. **Five dead columns — reproduced exactly**, and the list is confirmed against the *current* frames rather
   than assumed: `precip_mm`, `precip_mm_prev1h`, `temp_x_precip`, `temp_x_precip_prev1h`, `runway_null`.
   Four of them are 22 % of the weather block. This is P8-B's mandate, arrived at from a second direction.
3. **No `ec_*`-class catastrophe: zero `MISSING` and zero `NONNULL` failures on 100 columns.** `[S]` The
   13.14 s trap is not one we are standing in — which also means importing the gate buys almost nothing on
   today's features. Its value is prospective.
4. **The LFPG cross-theme lead is reproduced**: `lobt_minus_eobt1` (and its `_slog` copy) at LFPG, July 2026,
   **+22.2 points** month-matched, the same airport / month / field family as kind-mango's `eobt1_sched`
   35.7 → 61.6 % finding. `[X]` LFPG's plan-time reporting behaviour changed between July 2025 and July
   2026. **This is a lead for another theme, not for P8.** It is the only rise in the whole matrix above 20
   points; the plan's §5.3 list of six is a "largest rises" list, and the other five (EHAM +18.0/+17.1, LEMD
   +11.7, LSZH +9.2, LTFM +8.5) are all below the reporting threshold the plan itself set in §5.4.
5. `[S]` **Zero `NONZERO_PARTIAL` findings** — no column clears one of the two references and not the other.
   The month-matched conjunct changes nothing on today's matrix. `[X]` It is still the right rule; it costs
   nothing and it is what stops an ordinary seasonal column from failing a build.

---

## 4. Wiring, and why the gate is off

`[S]` `src/prc/pipeline.py` runs the gate immediately after `build_features` and before CV — the position
§5.4 asks for, and the right one, because **CV cannot see this class of defect**: cross-validation validates
on the training side, which is intact by construction.

It is behind `coverage_gate:`, **default false**, so every existing recipe runs byte-identically to before.
`configs/main_gap.yaml` was not modified.

**The operationally important finding: the gate cannot be turned on for production today.** `main_gap.yaml`
with `coverage_gate: true` exits 1 on the five dead columns. `[X]` That is the correct behaviour and it
should not be weakened. The three ways forward, in order of preference:

1. **P8-B lands** and deletes the four weather constants; `runway_null` is then a one-line separate fix.
   **Measured since this note was written** (`P8-B-wxcodes-repair.md` §4): with the repair on, the gate goes
   from **9 failures to 5**, all four weather DEAD failures clear, and none of the eight new columns fails
   any rule. But P8-B is REJECT as a scoring change, so shipping it is a correctness decision, not an
   adoption. The cheapest path to a switchable gate may instead be option (2).
2. **The columns are deleted without P8-B** — they carry no information by construction, so this is a
   removal, not a feature change. `[X]` It is still an 5-column arity change and the plan's §6.4 note
   applies: price it with `prc.compare`, do not just delete and ship.
3. **They are waived**, each with a written reason, which is the honest option only if 1 and 2 are both
   refused.

`[S]` A second, smaller wiring decision: `prc.submit` takes a file path and a template, not the feature
frames, so a gate call there would have meant changing its signature. Instead `pipeline.py` prints a loud
warning when `--submit` is passed with the gate off. `[X]` That is weaker than §5.4's "hard precondition"
and I have deliberately not made it hard, because a hard precondition today would block every submission on
a failure that P8-B is about to remove. Making it hard is a one-line change once (1) or (2) lands, and it
should be made then.

---

## 5. Tests, and the mutation audit

`tests/test_coverage.py`, **12 tests, all passing**. Full suite: **111 passed** before P8 began,
**123** after P8-D's 12, **132** after P8-B's further 9 — 0 existing tests changed. `ruff check` /
`ruff format` clean on every file this work touched (the one remaining `ruff` finding, `I001` in
`tests/test_train.py`, predates this work and was left alone).

Every rule has a **control column beside it** that must stay silent, because the defect this repo keeps
shipping is a check that passes everything — an AUC of exactly 1.0000 across every recipe including the
permuted control, a harness whose base already contained the columns it measured. A gate that flags nothing
and a gate that flags everything both look green in a summary line.

| test | rule | its control |
|---|---|---|
| `test_nonnull_collapse_fails_and_a_stable_column_does_not` | NONNULL (the `ec_*` mode) | a column present everywhere |
| `test_nonzero_drop_needs_both_references` | NONZERO | a seasonal column that drops vs the year but not vs the matched month |
| `test_dead_column_fails_though_its_drop_is_zero` | DEAD | a sparse-but-live column at 1 % non-zero |
| `test_near_dead_is_reported_not_failed` | NEAR_DEAD tier split | — |
| `test_a_rise_is_reported_not_failed` | RISE | — |
| `test_a_column_absent_from_the_ranking_frame_fails` | MISSING | a column that is kept |
| `test_a_waiver_moves_a_failure_and_must_carry_a_reason` | waivers | an empty reason must raise |
| `test_categoricals_are_never_reported_dead` | roll-up null handling | — |
| `test_a_null_and_an_explicit_zero_are_the_same_thing_to_the_gate` | null-as-zero semantics | — |
| `test_month_matched_reference_is_the_year_before` | period key | — |
| `test_gate_reads_the_columns_the_model_sees_not_the_frame_width` | column set | `noise_control` hidden vs included |
| `test_acceptance_on_the_real_frames` | the whole gate | the §5.5 numbers, pinned exactly |

### 5.1 Mutation audit `[S]`

Ten mutants were written into `src/prc/coverage.py` one at a time and the suite re-run
(`scratchpad/mutate.py`). **All ten are killed.** Two survived the first pass and are the reason this section
exists:

| mutant | outcome |
|---|---|
| M1 non-null rule disabled | killed |
| M2 the two-reference `and` → `or` | killed |
| M3 DEAD rule deleted | killed |
| M4 DEAD threshold restored to the plan's literal 0.1 % | killed (by the real-frame acceptance test) |
| M5 RISE promoted to a failure | killed |
| M6 roll-up reverted to a plain weighted mean | killed |
| M7 column set taken from `frame.columns` | killed |
| M8 month-matched reference → itself | killed |
| M9 MISSING rule deleted | killed |
| M10 non-zero share computed over non-null rows only | killed **on the second attempt** |

`[S]` **M6 and M10 were real bugs, not hypotheticals.**

- **M6** was in the code as first written. polars' `sum` treats a null as 0, so the n-weighted roll-up of a
  column whose non-zero share is null — which is every categorical — returned `0.0`, and the gate failed all
  ten categoricals as dead constants. Caught by running it, fixed with a null-aware roll-up, now pinned.
- **M10** survived its first kill attempt because a `row[...] or 0.0` elsewhere in the same function was
  independently coercing `None` to `0.0`, making the mutant behaviourally identical. `[X]` That is the
  interesting one: two separate sloppy null-handlings were cancelling out, and a test asserting the *outcome*
  could not see either. The coercion is now explicit (`_pct`) and the semantics are asserted on a
  partially-null column, where the two implementations genuinely differ.

---

## 6. Verdict

**SHIP** `src/prc/coverage.py`, `tests/test_coverage.py` and the default-off pipeline wiring.

`[X]` Its value on *today's* features is the dead-column finding — five real defects on the first run, one of
them (`runway_null`) not weather at all — plus the measured confidence that we are not sitting on a
13-second `ec_*` trap, plus an independent third reproduction of the LEMD/LSZH dewpoint drift. Its real value
is prospective: it is the entry condition for P8-C and for every external dataset after it, and it is the
only instrument in this repo that can see the class of defect that made `hour_utc` vanish from two
submissions.

**Follow-ups this created, none of them P8:**

1. Delete or repair the five dead columns, then flip `coverage_gate: true` in the production recipe and make
   the `--submit` warning hard. (P8-B covers four of the five.)
2. `runway_null` is a constant and is not weather. One line, priced like any other arity change.
3. The LFPG July-2026 plan-time rise (+22.2 pts on a high-rank feature family at 11.6 % of scored rows) is a
   lead for whichever theme owns the NM flight-plan fields.
