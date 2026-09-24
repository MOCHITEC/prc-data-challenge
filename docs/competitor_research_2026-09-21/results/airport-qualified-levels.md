# Airport-qualified categorical levels — REJECT (diagnosis; no fit run)

Written 2026-09-24 on `claude/model-b-and-arrival-features`. Candidate: replace the global `STAND_mvt` and
`RUNWAY_mvt` Enum levels in `prc.train._matrix` with airport-qualified ones (`EDDF‖A12`), so that the same
stand name at two airports stops sharing one integer code.

Named as a control in `plans/P11-per-airport-models.md` §7.4: *"the airport-qualified categorical levels
control (`ADEP ‖ STAND`, `ADEP ‖ RUNWAY`) — one feature change, one fit. If it captures most of the gain,
ship that instead."* This file measures whether it captures anything at all.

Marks, the convention of the other files in this directory:

* **[S-repo]** — measured on this machine from a committed artefact; the command is given.
* **[X]** — my own inference or arithmetic. Never dressed as a measurement.

**No model was fitted.** No pipeline run was started. Nothing under `outputs/` was written or read. Every
number below comes from a read-only scan of `data/work/feat_train.parquet`, `data/work/feat_rank.parquet`
and the archived OOF `experiments/oof_main_gap-62dd7c6e88.parquet`. The script is
`docs/competitor_research_2026-09-21/scripts/airport_qualified_levels.py`; each section names the flag that
reproduces it. `src/` is unchanged.

**Verdict: REJECT.** The defect is real and large in raw label terms (§1, §2). It is also already paid for
(§3, §6): against a zero-information control with an identical level count, airport-qualification is worth
at most **−0.11 s** on lane A even under an in-sample ceiling that cannot be achieved, and the honest
cross-fitted estimate is **−0.06 s**. Both are below the instrument's ±0.5 s single-column noise floor
(`plans/EXECUTION-ORDER.md:68`) **[S-repo]**. The three-arm gate cannot resolve this and was not run.

---

## 1. The four motivating numbers reproduce exactly

`--sections 1`, on `data/work/feat_train.parquet` (2,085,047 rows, all `PHASE_mvt == "DEP"`). **[S-repo]**

| quantity | reported to me | measured here |
|---|---:|---:|
| `STAND_mvt` names at more than one airport | 520 of 1,899 | **520 of 1,899** |
| share of rows carrying such a stand name | 57.5 % | **57.545 %** (1,199,842 rows) |
| `RUNWAY_mvt` names at more than one airport | 15 of 53 | **15 of 53** |
| share of rows carrying such a runway name | 50.0 % | **50.007 %** (1,042,670 rows) |
| most airports pooled into one stand code | 6 | **6** (names `312` and `210`) |

All four reproduce. The mechanism is confirmed by reading `_matrix` (`src/prc/train.py:319`): the Enum is
built as `pl.Enum(sorted(s.unique().to_list()))` over the *name* column alone, so `ADEP_mvt` never enters
the code. **[S-repo]**

Qualifying would take `STAND_mvt` from 1,900 to 2,725 levels and `RUNWAY_mvt` from 53 to 70 (§4).

The worst case is as bad as described. Stand `312` **[S-repo]**:

| ADEP | rows | mean `TAXITIME_SEC_mvt` | median |
|---|---:|---:|---:|
| LIRF | 3,322 | 1,107.6 | 969 |
| EDDM | 2,390 | 825.8 | 781 |
| LEMD | 1,895 | 1,039.9 | 1,016 |
| LEBL | 488 | 642.8 | 604 |
| LTFM | 295 | 976.5 | 848 |
| LSZH | 1 | 673.0 | 673 |

Six pieces of concrete, one Enum code, 465 s between the extreme means.

---

## 2. The collisions are not cosmetic — in raw label terms

`--sections 2`. Within each colliding name, a one-way decomposition across airports, on labelled rows with
`0 < y < 7200` and cells of at least 30 rows. **[S-repo]**

| | `STAND_mvt` | `RUNWAY_mvt` |
|---|---:|---:|
| colliding names passing the 30-row filter | 444 | 13 |
| rows they cover | 1,130,945 | 986,844 |
| row-weighted η² (between-airport share of within-name variance) | **0.2169** | 0.0627 |
| row-weighted spread of group means | **433.4 s** | 220.7 s |
| median spread of group means | 407.4 s | 232.5 s |
| share of those rows with spread > 120 s | 86.3 % | 96.2 % |

**This is the strongest form of the case for the candidate, and it is genuine.** If the pooled code were
the model's only handle on the stand, 22 % of the within-name label variance would be unexplained by
construction. §3 and §6 are about whether it is the only handle.

---

## 3. 98.4 % of the colliding rows already carry an airport-qualified encoding

`--sections 3`. `reference_table` (`src/prc/train.py:206`) is keyed on **(`ADEP_mvt`, `STAND_mvt`,
`RUNWAY_mvt`)** and emits `ref_p10 / ref_p20 / ref_med / ref_n`. A row whose cell clears
`reference_min_count` (30) is therefore *handed four numbers that already distinguish LIRF's `312` from
EDDM's `312`* — the tree does not have to recover it from `ADEP_mvt` splits at all. **[S-repo]**

| | value |
|---|---:|
| `reference_table` cells at `min_count = 30` | 6,434 |
| share of all rows with a non-null `ref_n` | **97.91 %** |
| share of *colliding-stand* rows with a non-null `ref_n` | **98.40 %** |
| share of *colliding-runway* rows with a non-null `ref_n` | 97.92 % |
| share of **all** rows that are colliding **and** have no `ref_n` | **0.92 %** (stand) |

`airport_runway_hour_table` (ADEP × RUNWAY × hour), `airport_median_table` (ADEP) and
`operator_offset_table` (operator × ADEP) are airport-keyed too, and `ADEP_mvt` is itself one of the ten
categoricals. **[S-repo]**

**[X]** So the residual population for which qualification could be new information is the ~0.9–1.0 % of
rows where the reference table is missing. That is an upper bound on *where*, not on *how much*; §6 measures
the how much.

*Caveat, stated rather than buried:* the reference table here is fitted on the whole frame, while in CV it
is fitted on the training side of the fold, so fold-wise coverage is slightly lower than 97.91 %. The
direction is small and does not change the order of magnitude. **[X]**

---

## 4. The cost side: level count, rows per level, unseen levels in 2026

`--sections 4`, train frame against `data/work/feat_rank.parquet` (344,841 scored rows). **[S-repo]**

| | `STAND_mvt` | `RUNWAY_mvt` |
|---|---:|---:|
| levels, plain | 1,900 | 53 |
| levels, qualified | 2,725 | 70 |
| level multiplier | ×1.434 | ×1.321 |
| median rows per level, plain | 553 | 12,884 |
| median rows per level, qualified | **382** | 10,007 |
| qualified levels under 30 rows | 555 | 11 |
| train rows sitting in those levels | 0.188 % | 0.004 % |
| **scored rows on a level unseen in training, plain** | 284 (**0.082 %**) | 0 |
| **scored rows on a level unseen in training, qualified** | 1,589 (**0.461 %**) | 2 |

**Unseen-level handling.** `_fit` (`src/prc/train.py:372`) stacks the training frame and the target frames
and calls `_matrix` **once**, so the Enum is built over the *union* of their levels. An unseen 2026 level
therefore gets a valid code of its own that no training row carries — no NaN, no crash — and LightGBM's
categorical split, which tests membership of a learned subset of code values, routes it to the default
side. The row is predicted as if the stand were "none of the ones I learned". Qualifying multiplies the
rows that land there by **5.6×** (284 → 1,589). **[S-repo]** for the counts, **[X]** for the routing, read
from the code path rather than measured.

A second-order consequence worth recording, since it is the G11 failure mode's neighbourhood: because the
Enum spans train ∪ target, inserting a new level *shifts the codes of every level sorted after it*. This is
already true today and is not introduced by the candidate, but qualification makes new levels 5.6× more
common, so it makes the existing coupling between the target frame's content and the training encoding
correspondingly more active. **[X]**

---

## 5. Determinism (guard G11)

`--sections 5`. The candidate's encoding is
`pl.Enum(sorted((ADEP + "‖" + name).unique()))` — the same "Enum over sorted levels" construction the G11
fix installed, applied to a concatenated string. Three separate processes at three different
`POLARS_MAX_THREADS` settings: **[S-repo]**

```
run 1 (threads=1)  STAND levels_sha1 b81c5072c5b1  codes_sha1 d535398cd9e3   RUNWAY 1fefd65e0577 / f0dbc9170830
run 2 (threads=2)  STAND levels_sha1 b81c5072c5b1  codes_sha1 d535398cd9e3   RUNWAY 1fefd65e0577 / f0dbc9170830
run 3 (threads=3)  STAND levels_sha1 b81c5072c5b1  codes_sha1 d535398cd9e3   RUNWAY 1fefd65e0577 / f0dbc9170830
```

Identical. The separator is safe to concatenate with: no `STAND_mvt`, `RUNWAY_mvt` or `ADEP_mvt` value
contains `‖`, and `ADEP_mvt` is fixed-width 4, so the qualified string is unambiguous. **[S-repo]**

This is demonstrated, not asserted — but note what it demonstrates: that the *encoding step* is
deterministic. It is not a substitute for `configs/determinism_check.yaml`, which would still have to be run
twice if this ever reached a fit.

---

## 6. How much is genuinely new: the production model's own residuals

`--sections 6`, on `experiments/oof_main_gap-62dd7c6e88.parquet` joined to the feature frame by
`MVT_ID_mvt`. Recovered base RMSE **277.5398** on n = 672,428, and **213.1266** on the 664,050 lane-A rows
— matching the 277.54 / 213.1 on record, which is the join's own check. **[S-repo]**

### 6.1 The measurement

The mechanism claimed for the candidate is a level shift: a different piece of concrete is a different taxi
distance. The most a new categorical level can buy through that mechanism is a per-level intercept on the
current model's residual. So: give each level its own mean residual and measure the RMSE change, three
ways.

* **plain** — one intercept per `STAND_mvt` *name*. The model already has this code, so this arm is the
  floor: it must come out at ~0 out-of-sample, and it does.
* **qualified** — one intercept per (`ADEP_mvt`, `STAND_mvt`). The candidate.
* **noise** — the P11 `shuffled` control applied to levels. Each plain name is split into *exactly as many*
  pseudo-levels as it has real airports, membership drawn from a hash of the row alone. **Identical level
  count, near-identical size profile, zero airport information.** It is what separates "more levels" from
  "the right levels", and it is the arm that answers the built-in-control requirement.

Each is reported twice: an **in-sample ceiling** (subtract each group's own mean — unachievable, optimistic
by roughly `n_levels · σ²`) and a **cross-fitted** estimate (group mean from the other half of the rows,
2-fold by hash of `MVT_ID_mvt`), which is what a model could actually realise.

### 6.2 Lane A (matched, n = 664,050, base RMSE 213.1266) **[S-repo]**

| arm | levels | in-sample ceiling ΔRMSE | cross-fitted ΔRMSE |
|---|---:|---:|---:|
| `STAND_mvt` plain | 1,756 | −0.698 | +0.354 |
| `STAND_mvt` **qualified** | 2,529 | **−0.925** | **+0.647** |
| `STAND_mvt` noise, same level count | 2,528 | −0.855 | +0.665 |
| `RUNWAY_mvt` plain | 49 | −0.159 | −0.132 |
| `RUNWAY_mvt` **qualified** | 64 | **−0.201** | **−0.165** |
| `RUNWAY_mvt` noise, same level count | 64 | −0.165 | −0.125 |

**The contrasts that decide it:**

| contrast | ceiling | cross-fitted |
|---|---:|---:|
| `STAND` qualified − plain | −0.227 | **+0.293** |
| `STAND` qualified − **noise** | **−0.069** | −0.018 |
| `RUNWAY` qualified − plain | −0.043 | −0.033 |
| `RUNWAY` qualified − **noise** | **−0.037** | −0.040 |

Read the `qualified − noise` row. On lane A, airport-qualification beats a **meaningless** split of the
same size by **0.069 s** on stand and **0.037 s** on runway under a ceiling that cannot be reached, and by
0.018 s and 0.040 s honestly. **[X]** Taking the two together and treating them as roughly additive,
**≤ 0.11 s of ceiling and ≈ 0.06 s of realisable effect.**

**Seed robustness of the noise arm.** The noise arm's membership is random, so the contrast against it is
itself a draw. At `--seed 7` instead of 0, lane-A `STAND` qualified − noise is **−0.067 ceiling / +0.053
cross-fitted**, against −0.069 / −0.018 at seed 0. **[S-repo]** The *ceiling* contrast is stable to
0.002 s; the *cross-fitted* contrast moves 0.071 s between two seeds and changes sign. **[X]** That is the
cleanest statement of the result: the realisable effect of airport-qualification is smaller than the
seed-to-seed variation of its own control.

The cross-fitted stand numbers are *positive* — a per-stand-level intercept makes the production residual
worse out-of-sample, for the qualified level (+0.647) very slightly less than for the noise level (+0.665).
**[X]** That is the signature of a residual with no exploitable per-level structure left: what is being
fitted is 2,529 groups' worth of sampling noise. The model has already absorbed the §2 effect.

### 6.3 Pooled, all lanes (n = 672,428, base 277.5398) **[S-repo]**

Same picture: `STAND` qualified − noise = **−0.075** ceiling / −0.146 cross-fitted; `RUNWAY` qualified −
noise = −0.028 / −0.043.

### 6.4 The one subset where it could have been hiding

The `ref_n`-missing rows of §3 are the only rows where the pooled code really is the model's sole handle on
the stand. There are **14,224** of them (2.1 % of scored rows), base RMSE 326.82. **[S-repo]**

`STAND` qualified − noise: ceiling **−3.308 s**, cross-fitted −5.059 s — on that lane. **[X]** Scaled to
the full set through the SSE (14,224 rows moving from 326.82 to ~323.5 removes ≈ 3.0 × 10⁷ of a total
5.18 × 10¹⁰, i.e. ≈ 0.06 % of SSE, i.e. ≈ **−0.08 s** of pooled RMSE), it is the same ~0.1 s as §6.2 and
not a hidden reservoir. The two independent routes agreeing at ~0.08–0.11 s is the main reason I am willing
to close this without a fit.

Also note the arm counts: 1,844 levels fitted on 14,224 rows. Anything measured there is mostly
overfitting, which is exactly why the noise arm — which overfits identically — is the comparison and the
raw ceiling is not.

### 6.5 What this measurement does **not** cover

**[X]**, stated plainly because it is the one place the conclusion could be wrong. A per-level intercept on
residuals captures the *additive* value of a level. It does not capture value a qualified level might have
in **interaction** — a split that is only useful in combination with hour, weather or queue length, where
`EDDF‖A12` and `LFPG‖A12` want opposite directions. I have not measured that and cannot from an OOF alone.

Two reasons I judge it small rather than unmeasured-therefore-possible:

1. The claimed mechanism in the brief is distance — *"different pieces of concrete with different taxi
   distances"* — which is an intercept, and the intercept is measured at ~0.
2. An interaction term needs the main effect to be separable first, and the tree can already separate the
   two `A12`s by splitting on `ADEP_mvt` (one of ten categoricals, and per `src/prc/adversarial.py:24`
   `ADES_mvt` alone carries 0.18 of the gain, so airport-level splits are plentiful). Qualification makes
   that split cheaper, not newly possible.

If someone wants to close that gap, the way to do it is the fit in §7 — and §7 says the fit cannot resolve
an effect this size, which is the actual reason to stop.

---

## 7. The pre-registered price, and why the gate was not run

**Registered before any fit exists; no arm of the gate has been run.** It is *not* a clean pre-registration
in the P11 sense — §1–§6 existed first, and it is derived from them. Saying so is the point.

### 7.1 What the gate would be

Three arms, `configs/p4a_{base,noise,gap}.yaml` shape: `configs/main_gap.yaml` + `folds: [forward]`, one
setting apart, 3 seeds, same folds and same rows.

* **base** — production, global levels.
* **noise** — `STAND_mvt` and `RUNWAY_mvt` qualified by a hash of `MVT_ID_mvt` into the same number of
  levels each name really has across airports. Identical level count, no information. **It moves if, and
  only if, splitting a categorical level at all helps this model** — through extra split-search room or
  through the finer partition acting as regularisation. That is the quantity the candidate has to beat.
* **candidate** — `ADEP_mvt ‖ name`.

Decided by `prc.compare.verdict`: full CI and the drop-200 CI both on the improving side, margin reported
against base **and** against noise.

I did **not** write these three configs. They would require a `qualify_categorical_levels` flag in
`_matrix` that does not exist, and committing configs for a candidate this file rejects is dead weight.
`src/` is untouched; nothing is behind a new flag because nothing was added.

### 7.2 What I expect, and the failure condition

From §6.2, ceiling −0.11 s against noise, realisable ≈ −0.06 s, on lane A; the forward fold is roughly half
the rows, and the effect is a per-row property, so the same per-row size. **[X]**

**Point prediction: −0.1 s against the noise arm. Band: −0.4 s to +0.4 s.**

The band is set by the instrument, not by the effect: a single added column is a **±0.5 s lottery draw on a
0.25 s instrument** (`plans/EXECUTION-ORDER.md:68`) **[S-repo]**, and `configs/main_gap.yaml` records the
noise arm at +0.23 s against base on exactly this fold **[S-repo]**. The last use of this gate resolved
−3.62 s predicted against −3.21 s live — a **34× larger** effect than this one.

**Failure condition, and it is already met:** *if the ceiling contrast against the noise arm is smaller in
magnitude than the instrument's ±0.5 s single-column noise floor, the gate cannot distinguish the candidate
from its own control and must not be run.* It is −0.11 s. **The gate was not run.**

**[X]** Running it anyway would produce a number in [−0.4, +0.4] whose sign is a coin flip, and this
repository has twice shipped on a reading it could not support. A 3-seed forward-fold triple costs ~2,200 s
of the compute P7 is waiting for (`plans/P11` §8 anchors: `p4a_base` 695.8 s, `p4a_noise` 709.3 s,
`p4a_gap` 814.3 s) **[S-repo]** to measure something eight times smaller than its own resolution.

### 7.3 The no-op / leak control, built in

The requirement that the control be built in from the start is met by the §6.1 **noise** arm, not deferred
to the fit. Its numbers (`STAND` ceiling −0.855 against the candidate's −0.925; 2,528 levels against 2,529)
are distinct from both the plain and the qualified arms at six figures, so the three arms are provably not
the same computation — the failure mode where several recipes return identical numbers would have shown up
here as three identical rows, and did not.

---

## 8. Verdict

**REJECT.**

1. The defect reproduces exactly as described: 520/1,899 and 57.545 %, 15/53 and 50.007 %, one code pooling
   six stands with 465 s between their means (§1, §2).
2. It is nonetheless **already paid for**. 98.4 % of colliding rows carry a `ref_*` encoding keyed on
   (ADEP, STAND, RUNWAY) (§3), and on the production model's own residuals the qualified levels beat a
   zero-information split of the same size by ≤ 0.11 s of unachievable ceiling and ≈ 0.06 s realisable
   (§6). The subset where the encoding is missing agrees at ≈ 0.08 s (§6.4).
3. It is not free: ×1.43 levels, median rows per level 553 → 382, and **5.6× more scored 2026 rows landing
   on a level the model never saw** (284 → 1,589) (§4).
4. The effect is **eight times smaller than the resolution of the gate that would judge it** (§7.2).

**Do not ship it. Do not run the gate.** The P11 §7.4 control is answered: airport-qualified levels do not
capture the per-airport gain, because on this model there is no per-airport stand gain left to capture.

**[X]** The transferable finding is larger than this candidate: **`reference_table` is doing the
airport-qualification already, through the label rather than through the code.** Any future candidate whose
mechanism is "give the tree a finer per-airport partition of stand or runway" should be checked against §6
before it is fitted, because it is competing with four airport-keyed encoding tables and not with a naive
baseline. The same check is one command:

```
POLARS_MAX_THREADS=2 .venv/bin/python \
  docs/competitor_research_2026-09-21/scripts/airport_qualified_levels.py --sections 6
```

---

## 9. Reproduction

Read-only. All six sections end to end: **15.6 s wall, 1.20 GB peak RSS** at `POLARS_MAX_THREADS=2` on a
warm page cache, measured while two other jobs were on the machine. **[S-repo]**

```
cd /home/user/prc-data-challenge
POLARS_MAX_THREADS=2 .venv/bin/python \
  docs/competitor_research_2026-09-21/scripts/airport_qualified_levels.py --sections 1,2,3,4,5,6
```

`--sections` selects: 1 collisions, 2 label distributions, 3 existing-encoding coverage, 4 level budget and
unseen levels, 5 determinism hashes, 6 residual recovery against the OOF. Section 6 takes `--oof`
(default `experiments/oof_main_gap-62dd7c6e88.parquet`) and `--seed` for the noise arm's membership
(default 0); the noise arm is seeded, so §6 reproduces bit for bit.

Files touched by this work: this document and
`docs/competitor_research_2026-09-21/scripts/airport_qualified_levels.py`. **`src/`, `configs/`,
`outputs/`, `data/` and `experiments/` are unchanged.**
