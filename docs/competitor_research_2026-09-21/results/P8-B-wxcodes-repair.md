# P8-B — repairing the four identically-zero weather columns from `wxcodes`

**Executed:** 2026-09-25, branch `claude/model-b-and-arrival-features`.
**Plan:** `docs/competitor_research_2026-09-21/plans/P8-weather-and-coverage-gate.md` §3.

## Verdict

**DO-NOT-SHIP as a scoring change. REJECT by `prc.compare.verdict` on both comparisons that matter.**
**The repair is nevertheless correct, it is never measurably worse anywhere, and it is the thing that lets
the P8-D coverage gate be turned on. That is a separate decision and it is the commander's.**

| comparison | K=0 (full) | K=200 | month bootstrap | verdict |
|---|---|---|---|---|
| base → **wxcodes repair** | **−0.18** [−0.64, +0.26] | −0.50 [−0.71, −0.24] | −0.18 [−0.65, +0.41] | **REJECT** |
| noise (arity-matched) → **wxcodes repair** | **−0.35** [−1.01, +0.22] | −0.35 [−0.59, −0.12] | −0.35 [−0.81, −0.00] | **REJECT** |
| base → noise (the price of arity alone) | +0.16 [−0.58, +1.06] | −0.12 [−0.36, +0.11] | +0.16 [−0.29, +1.00] | REJECT (as designed) |

`[S]` The full-data CI contains zero in both, so `verdict()` stops at its first gate. This is the eleventh
consecutive feature candidate to land below the instrument, and it lands in exactly the place
`docs/features_log.md` records ten others landing.

**The honest size of it: −0.35 s against the arity-matched control, on an instrument whose own measured
spread for four arbitrary columns is +0.16 [−0.58, +1.06].** `[X]` The repair's point estimate is inside
the range that adding four signal-free columns produces. It is not distinguishable from doing nothing.

---

## 1. The three arms

`[S]` `configs/p8b_{base,noise,wx}.yaml`, all `configs/main_gap.yaml` + `folds: [forward, jul_fwd]`,
`n_seeds: 3`, `--skip-predict`, the `configs/p4a_{base,noise,gap}.yaml` pattern.

| arm | model-visible features | `jul_fwd` | `forward` | pooled |
|---|---|---|---|---|
| `p8b_base-3dc917225f` | 90 | 314.4240 | 219.7648 | 258.6264 |
| `p8b_noise-b6b05e26c7` | 94 (+4 signal-free) | 314.0793 | 220.3537 | 258.7891 |
| `p8b_wx-7f7be04073` | 94 (18 weather → 22) | 314.1480 | **219.6536** | **258.4432** |

`[S]` **The arity control is not optional and it earned its place again here.** The repair's headline against
the base is −0.18 s. Against the arity-matched control it is −0.35 s — i.e. **the base comparison
*understates* it**, because the four noise columns happened to draw badly on the forward fold (+0.59 s) and
well on `jul_fwd` (−0.34 s). Without the control arm the honest number would have been unavailable in either
direction.

### 1.1 Two controls that were built in, and both passed

`[S]` **`p8b_base` reproduces `p8_wx_on` (P8-A's weather-on arm) to every printed digit**: 314.4240 /
219.7648 / 258.6264 against 314.424 / 219.7648 / 258.63, on frames rebuilt after `features.py` and
`weather.py` were both edited. This is the no-op check the repo has twice failed to make: it proves the
three extra hidden noise columns really are hidden, and that the `hourly_weather_table` refactor did not
move the default path.

`[S]` The second control is in the test suite: `test_acceptance_on_the_real_cache` asserts that
`hourly_weather_table(raw)` with the flag off is `.equals()` the shipped `data/ref/metar_hourly.parquet` —
the parquet every archived weather experiment was built from. If the refactor had moved one value, the suite
fails.

`[S]` **Note on `git_sha`:** the three records carry `3b40254`, `3b40254` and `338e020`. The second sha is
the commander's own WIP snapshot of my working tree, committed at 04:34 while the third arm was running
(`git show --stat 338e020` is exactly the files this work touched). `git diff HEAD` is empty apart from
`experiments/experiments.jsonl`. **The source was identical across all three arms**; only the recorded
string differs.

---

## 2. What the repair does (`src/prc/weather.py`, `wx_codes` flag)

`[S]` Confirmed on the rebuilt frames, `data/work_p8bwx`:

```
removed: precip_mm, precip_mm_prev1h, temp_x_precip, temp_x_precip_prev1h
added  : wet_wx, shower_wx, obscuration_wx, frozen_wx_i8   (+ their _prev1h copies)
model-visible weather columns: 18 -> 22, four of the eighteen having been constants
```

- `wet_wx` = any of `RA`, `DZ`; `shower_wx` = `SH`, `TS`; `obscuration_wx` = `FG`, `BR`, `HZ`, `FU`,
  matched as substrings, which is how METAR codes compose.
- `frozen_wx_i8` recovers a column that has existed all along and never reached the model: `frozen_wx` is
  Boolean and `numeric_feature_columns` admits only float/int.
- `deicing_gate` becomes `(temp_c ≤ 3) ∧ (frozen_wx ∨ wet_wx)`. `[S]` The `precip_mm > 0` term it replaces
  has never fired on a single one of 138,457 cached observations. On the real cache the new gate is a
  **strict superset** of the old one (0 hours lost, asserted in the test suite); on the training frame its
  non-zero share goes 0.4392 % → 0.796 %.

`[S]` Training-frame liveness of the new columns: `wet_wx` 8.59 %, `shower_wx` 5.38 %, `obscuration_wx`
3.23 %, `frozen_wx_i8` 0.51 %. **None is dead.** On the scored set:

| | 2026-01 | 2026-07 |
|---|---|---|
| `wet_wx` | 14.92 % | **2.15 %** |
| `shower_wx` | 7.25 % | **2.34 %** |
| `obscuration_wx` | 10.55 % | 0.22 % |
| `deicing_gate` (repaired) | 7.13 % | **0.00 %** |
| `low_vis` | 2.06 % | **0.00 %** |

`[S]` The plan's central claim is confirmed: in July 2026 — 55.7 % of the scored set — the repaired columns
are the **only** non-constant discrete weather channel.

---

## 3. Where the effect is, and why it is still not enough

Against the arity-matched control, on the rows the new columns are live on (both folds pooled):

```
wet_wx = 1          n=49,472   254.67 -> 254.21   -0.46 [-1.87, +0.88]
shower_wx = 1       n=29,919   268.84 -> 267.34   -1.49 [-3.23, +0.26]
obscuration_wx = 1  n=22,326   262.07 -> 261.30   -0.77 [-2.19, +0.76]
deicing_gate = 1    n= 4,602   317.98 -> 317.56   -0.42 [-2.74, +1.65]
forward  matched    n=325,004  198.65 -> 198.41   -0.24 [-0.39, -0.08]
jul_fwd  matched    n=186,775  251.71 -> 251.58   -0.13 [-0.38, +0.13]
```

`[X]` Every point estimate is on the improving side and every CI but one contains zero. The one that does
not — `forward matched`, −0.24 [−0.39, −0.08] — is a quarter of a second on 325,004 rows, which is the
instrument's floor, and `jul_fwd matched` does not reproduce it.

`[S]` **The −4.50 s that the gate rows show against the *base* arm is an artefact and should not be
quoted.** Against the arity-matched control the same cut is −0.42 [−2.74, +1.65]. The difference between
the two is the base arm's unlucky noise draw, which is exactly what the control exists to remove.

### 3.1 The one place the two decision rules disagree, stated because it is close

`[S]` `docs/features_log.md:5-6` states the adoption rule as *"`make compare` の 200 行除外後の 95% CI が
ゼロを除外したら採用"* — **K=200 alone**. `prc.compare.verdict` requires the full-data CI **and** the K=200
CI. On this candidate they disagree:

- by the prose rule, noise → wx is −0.35 [−0.59, −0.12] at K=200 → **ADOPT**;
- by the code rule, the full-data CI is −0.35 [−1.01, +0.22] → **REJECT**.

**I am reporting REJECT**, because the code rule is the one the tasking named, it is the one that produced
the calibrated −3.62 → −3.21 prediction, and it is the one every archived decision in this repo was taken
under. `[X]` But the commander should know that this is the closest any of eleven feature candidates has come,
and that the prose in `features_log.md` does not say what the code does. **Whichever rule is kept, the two
files should be made to agree.** That discrepancy is a finding of its own and is not P8's to settle.

`[X]` I would not adopt on the K=200 number even if the prose rule governed: −0.35 s is inside the ±0.5 s
that `docs/features_log.md:44-52` and `plans/EXECUTION-ORDER.md:68` measure for adding one arbitrary column,
and inside the +0.16 [−0.58, +1.06] this very experiment measured for adding four.

---

## 4. What the P8-D gate says about the repair — D measuring B

`[S]` This is what taking D first bought. `python -m prc.coverage --config configs/p8b_wx.yaml` on the
repaired frames:

```
before the repair (main_gap):  9 failures, 4 report-only   (104 columns: 90 numeric + 10 categorical -> 100)
after  the repair (p8b_wx)  :  5 failures, 4 report-only   (104 columns: 94 numeric + 10 categorical)
```

**All four weather DEAD failures are gone. None of the eight new columns fails any rule.** What remains is
`runway_null` — a constant that is not weather and was always a separate fix — and the four `dewp_spread_c`
drift cells at LEMD/LSZH that both we and kind-mango independently confirmed as data drift.

`[S]` **And the gate surfaced the caution the plan asked it to.** The largest month-matched non-zero drop
among the repaired columns:

```
shower_wx @EDDM 2026-07   23.82% -> 6.41%   -17.4 pts
shower_wx @EDDF 2026-07   18.66% -> 4.27%   -14.4 pts
wet_wx    @EDDM 2026-07   18.84% -> 6.23%   -12.6 pts
```

`[S]` **−17.4 points, against a 20-point threshold.** The new columns come within 2.6 points of failing the
gate they were built under. `[X]` This is a *support* shift and not a missingness one — July 2026 was a dry
July, the column is present and correctly zero — so passing is the right answer. But it means a July rain
feature is much less active at serve time than at train time, and it is a concrete reason not to expect the
LB to reward this even if CV had.

---

## 5. What was changed in `src/`

Everything is behind a flag whose default is current behaviour.

| file | change | default |
|---|---|---|
| `src/prc/weather.py` | `hourly_weather_table(metar, wx_codes=False)`; `WET_CODES` / `SHOWER_CODES` / `OBSCURATION_CODES`; `_any_code` | `wx_codes=False` → the shipped table, asserted `.equals()` the shipped parquet |
| `src/prc/pipeline.py` | `wx_codes` read from the recipe; `weather_cache_path()`; `maybe_join_weather(..., wx_codes)`; `build_features(..., wx_codes)` | `wx_codes: false` |
| `src/prc/pipeline.py` | **`cache_fingerprint` now hashes `weather.py` and carries `wx_codes`** | hashed only when `use_weather` is on |
| `src/prc/features.py` | `NOISE_CONTROLS`: `noise_control_2/3/4` added as candidate columns | hidden unless a recipe names them; `noise_control`'s seed unchanged so `p4a_noise` still reproduces |

`[S]` **The fingerprint fix closes the trap the plan called the highest-probability way to waste a day.**
`cache_fingerprint` hashed `features.py` only, so an edit to `weather.py` left every cached frame valid and
the next run would have reported the old numbers with the new code on disk. It now also means a recipe
cannot silently read the wrong frames: running `p8b_wx.yaml` against `data/work` detects the mismatch and
rebuilds rather than training on base's weather columns under the repaired arm's name. `[S]` Observed live:
*"cache in data/work was built from different inputs (feature_code, weather_code, wx_codes); rebuilding"*.

### Tests

`tests/test_weather.py` (+9, of which one is 13 parametrised cases) and `tests/test_feature_cache.py` (+3).
Suite: **111 → 132 passing**, 0 changed.

| test | what it pins |
|---|---|
| `test_wx_codes_defaults_to_the_shipped_columns` | the flag defaults to current behaviour |
| `test_the_repair_swaps_the_dead_columns_for_live_ones` | the column set and `Int8` dtypes (so they survive `numeric_feature_columns`) |
| `test_present_weather_codes_map_to_the_right_family` | 13 real code strings from the cache census, **including two controls** (`""`, `"SKC"`) that must flag nothing |
| `test_the_repaired_gate_fires_on_freezing_rain_where_the_old_one_could_not` | the mechanism in one assertion: −1 °C with `-RA` is gate 0 before and gate 1 after; **control**: the same code in July stays 0 while `wet_wx` turns on |
| `test_the_repaired_columns_reach_the_model_and_the_dead_ones_no_longer_do` | 18 → 22 through `numeric_feature_columns`, `frozen_wx` still invisible as a Boolean |
| `test_acceptance_on_the_real_cache` | on all 138,457 observations: the default path still `.equals()` the shipped parquet; the new gate is a strict superset of the old; `precip_mm` really is constant zero; July 2026 gate and `low_vis` are flat while `wet_wx`/`shower_wx` are live |
| `test_editing_the_weather_module_invalidates_a_weather_on_cache` | the fingerprint trap |
| `test_editing_the_weather_module_does_not_invalidate_a_weather_off_cache` | **the control** — a weather edit must not cost a `use_weather: false` recipe a rebuild |
| `test_the_wx_codes_flag_is_part_of_the_key` | and is ignored when there is no weather join to vary |

`[X]` **Would a mutant of this change survive these tests?** The ones I could construct do not: gutting
`_any_code` to return `False` fails the parametrised family test and the real-cache test; leaving
`precip_mm` in fails the column-set and model-visibility tests; keeping the old gate definition fails the
freezing-rain test; emitting the new columns as Boolean instead of `Int8` fails the model-visibility test;
dropping the `weather.py` hash from the fingerprint fails a cache test. The mutant I could **not** rule out
by test alone is a wrong *code family* — putting `BR` in `WET_CODES`, say — which is why the family test
lists literal strings taken from the census of the actual cache rather than invented ones.

---

## 6. Recommendation

**As a scoring change: DO-NOT-SHIP.** REJECT on the rule the tasking named, twice, and the point estimate is
inside the arity noise. This is loss number eleven and it fits the pattern of the other ten exactly.

**As a correctness change: the commander's call, and here is the case on each side.**

For shipping it anyway `[X]`:
- it deletes four columns that are provably constant and a gate term that provably never fires;
- it is not worse anywhere measured — every cut's point estimate is ≤ 0;
- it takes the coverage gate from 9 blocking failures to 5, and the 4 it removes are the weather ones. After
  it, `coverage_gate: true` needs only `runway_null` dealt with;
- a constant column that looks like a measurement is a trap for the next reader, and the next reader has
  twice been us.

Against `[X]`:
- it is an unmeasured change to a production recipe, and this repo's record on those is poor;
- it is a **net +4 columns**, so it spends arity budget for no measured return;
- the July support shift (−17.4 points at EDDM) means the thing it adds is much less active at serve time
  than at train time.

`[X]` **If the commander wants the gate switchable without an unmeasured scoring change, the cheapest path
is not this repair.** It is to delete `precip_mm` and `temp_x_precip` and their `_prev1h` copies and add
nothing — a pure **−4 column** removal of provable constants, which clears all four weather DEAD failures,
moves arity in the direction the evidence prefers, and can be priced against `p8b_base` with one more run.
That option was not in the plan and I did not measure it; I am naming it because §3 shows the information
the repair adds is worth about −0.35 s, which is to say nothing, while the four columns it removes are
worth removing on their own terms.

---

## 7. On P8-C

`[S]` The plan says P8-C must be judged against P8-B rather than against nothing, and that the stop rule
turns on B.

**My reading: P8-C is not worth running, and I am stopping rather than starting it.** The plan put P8-C at
`[X]` ~10 % to clear the bar and ≤ 0.5 s if it landed, on the argument that it supplies precipitation
*amount* where P8-B supplies only presence. But P8-B has now measured the value of the presence channel at
**−0.35 s [−1.01, +0.22]** against its own control, on 49,472 rain rows — and the amount is a refinement of
a signal that is itself not measurable. `[X]` A 6 km model grid refining a co-located observation's
already-null effect is not where the next second is.

**That call is the commander's, and the tasking reserved it.** The number to make it on is the −0.35 s
above; if the commander reads that as a near-miss rather than a null, then C becomes the natural next
measurement and it should be run against `p8b_wx`, not against `p8b_base`.
