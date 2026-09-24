# P11 — Per-airport specialist models

Written 2026-09-24 on `claude/model-b-and-arrival-features`. Baseline: `configs/main_gap.yaml`, OOF
`experiments/oof_main_gap-62dd7c6e88.parquet`, total **277.54** (matched 213.13 on 664,050 rows,
unmatched 1,606.89 on 8,378).

Marks, same convention as the other plans in this directory:

* **[S]** — read from an external source, URL or path given.
* **[S-repo]** — read from this repository, or measured on this machine during planning; the command
  is given so it can be re-run.
* **[X]** — my own inference or arithmetic. Never dressed as **[S]**.

**No model was fitted while writing this. No pipeline run was started.** The code change described in
§9 was written and its tests were run on synthetic data (`make test`, 85 passed); nothing touched
`data/raw` or `data/work` except read-only scans, and every number below marked **[S-repo]** comes from
an existing archived OOF, an existing record in `experiments/experiments.jsonl`, or a column scan of the
cached feature frame.

---

## 0. Provenance — why this file did not exist until now

This candidate was **never planned**. The ten-theme planning round of 2026-09-21 produced P1–P10 and an
auditor's list of 26 unplanned candidates (`plans/EXECUTION-ORDER.md`); per-airport *models* are not
among the ten themes and are not on the auditor's short list. **[S-repo]**

`plans/P6-per-airport-upper-bound.md` is a different object and must not be read as prior work on this.
P6 is a per-airport **cap** — `min(pred, 2025 max)` at four airports, 12 rows moved — and it is closed:
*"superseded — v10's reparameterisation already collected it"* (`EXECUTION-ORDER.md`). **[S-repo]**

The evidence base is `results/12-per-group-specialists.md`, which surveyed the field and did not measure
anything on our data — its own opening line says so. **[S-repo]** The user has approved adding the
candidate *on the grounds that the field uses it*, and has required that it be verified like everything
else. This plan is that verification, and it is designed so that the answer "the field is right but it
does not pay here" is as easy to reach as the answer "adopt".

---

## 1. What the external evidence actually says, in one table

Compressed from `results/12-per-group-specialists.md`. Every row is **[S]** to that file.

| source | group | rows per group | structure | outcome |
|---|---|---:|---|---|
| Filienko et al., KDD 2023 UC (NASA 4th) | airport, 10 | 1.2M–3.5M | global vs per-airport, same features | **global better at 8/10 and overall**, 10.7232 vs 10.7686 MAE; authors call it "at par" |
| NASA pushback 1st (CDS) | airport, 10 | — | unweighted mean of 3 per-airport CatBoosts **+ 1 global** carrying airport as a categorical | won, 10.673 |
| NASA pushback 2nd (Moles) | airport, 10 | — | per-airport only, per-airport hyperparameters, no global | 2nd, 10.728 |
| jubilant-vase, *this* competition | airport, 3 of 10 | ~130k–270k | **50/50 blend** of global and local, cold-started, identical features and hyperparameters | **290.488 → 286.656 official (−3.832)**; "more airport specialists" is on their *rejected* list |
| OpenAir, *this* competition | airport, ≥20k rows | ≥20k or fall back to global | airport expert as one of five in an NNLS blend | expert weight 0.491, blend −4.33 s Jan+Jul, −9.56 s December transfer |
| Phoenix-Ops, *this* competition | LIRF ∧ unmatched | 1,488 | specialist replaces global on scope | −1.6945 official, but *only* with 5-seed averaging; **non-LIRF unmatched specialist +0.012 s, rejected**; **matched-LIRF specialist rejected**; **airport-specific blend weights rejected** (323.712 vs 323.377) |
| Globus-Harris et al. 2024 | crowdsourced subgroups | ACS income | platform-ensembled subgroup models | **0.011 %** squared-error improvement over the best all-data model |

Two things are true at once and the design has to respect both:

1. **Every team that shipped per-airport models and reported a transferable number shipped a *blend*,
   not a replacement.** The one that used per-airport only (Moles) placed behind the one that blended.
2. **The only controlled head-to-head at this group size found parity, with the global model ahead.**
   And the KDD authors give the mechanism: their local models cannot use the departure-airport feature
   at all, and their global model's other features already encode what airport identity would say.

**[X]** The honest prior from this table is *"small, possibly zero, possibly negative"* — not
"−3.832 because jubilant-vase got −3.832". Their baseline was a single 436-tree LightGBM. Ours carries
ten categoricals, nine fold-fitted label encodings, an operator-offset table, a per-airport target
reparameterisation at LIRF and a physics bound. §4 shows that **all four** of our encoding tables are
already keyed on the airport, which is exactly the condition under which the KDD paper says specialists
stop paying.

---

## 2. The form being tested, and the forms that are not

### 2.1 The form

**One specialist per departure airport on model A, cold-started on that airport's rows, identical
feature set and identical hyperparameters, combined with the global model as a fixed
`0.5·global + 0.5·specialist` on that airport's rows. All ten airports. No tuning, no selection.**

That is `configs/p11_airport.yaml`.

### 2.2 Why the blend and not replacement

Standard two-forecast algebra, **[X]** throughout this subsection. Write σ_G, σ_S for the two
predictors' error standard deviations, r = σ_S/σ_G, ρ for their error correlation. At the fixed weight
w = 0.5,

```
MSE_blend / MSE_global = 0.25 + 0.25·r² + 0.5·ρ·r
```

The blend beats the global model iff `0.25r² + 0.5ρr − 0.75 < 0`, i.e. iff

```
r  <  r*(ρ)  =  −ρ + sqrt(ρ² + 3)
```

| ρ | break-even r* | best case (r = 1): RMSE change |
|---:|---:|---:|
| 0.95 | 1.0255 | −1.26 % |
| 0.97 | 1.0152 | −0.75 % |
| 0.98 | 1.0101 | −0.50 % |
| 0.99 | 1.0050 | −0.25 % |

This converts a vague question into a sharp one. **The whole experiment turns on whether a per-airport
specialist's matched-lane RMSE is within about 1 % of the global model's.** The KDD measurement puts
local 0.42 % behind global at 10× our per-group row count **[S]** — inside the tolerance, which is why
the blend, and not replacement, is the arm with a defensible prior.

Replacement (w = 1.0) is the arm the same algebra says to expect a *loss* from, and the KDD
side-by-side is direct evidence for that. It is not fitted as a separate arm because **it does not have
to be**: the blend is applied in model space, before `_assemble`'s floor and the LOBT bound, so

```
pure specialist  =  2 · (blend prediction)  −  (base prediction)
```

exactly, on every row where neither the zero floor nor the LOBT bound binds (both are counted in the
records: `lobt_bound_rows`). So the replacement arm is recovered in closed form from two OOFs that the
design already produces, at zero extra compute. It is a **diagnostic**, reported, and it does not enter
any verdict.

### 2.3 Why not "per-airport only where it pays"

jubilant-vase's selection gate — fit all ten, keep the airports that clear an improvement threshold, a
bootstrap win rate and a single-row-share cap — is the procedure with the best transfer record in the
survey **[S]**. It is nonetheless **excluded from this plan**, for a reason that is specific to us:

* Selecting the k best of 10 airports **on the fold that measured them** is selection on noise, and
  this repository has already paid for exactly that shape once: model B 127→255 leaves looked like
  −34.8 s in CV and cost **+13.20 s** on the board (`docs/README.md:110`). **[S-repo]**
* An honest selector needs a period it did not select on. Our fold set does not supply one: `jan` and
  `jul` train on months that come *after* the month they validate (`src/prc/cv.py:34-46`) **[S-repo]**,
  so "select on jan, confirm on forward" selects using future information. The only clean split is
  forward-only, and forward is the fold the verdict is taken on.
* `prc.compare.verdict` has no multiplicity correction and is not going to get one for this.

So: **per-airport differences are reported and never acted on in this plan.** If the candidate passes,
a selection stage becomes a separate, separately pre-registered experiment whose prerequisite is P3's
fold sidecar (`plans/P3-oof-and-validation.md`) and a holdout period reserved before any selection is
computed. That ordering is recorded here so it is not a post-hoc choice.

### 2.4 Multiplicity, explicitly

Three verdict-bearing comparisons exist in this plan, and they are a **conjunction**, not a menu:

1. stage 1, candidate vs base — must ADOPT under `verdict()`;
2. stage 1, candidate vs noise control — must improve with the K=0 CI excluding zero;
3. stage 2, candidate vs production OOF — must ADOPT under `verdict()`.

A conjunction of required tests lowers power and lowers the false-positive rate; it cannot inflate it.
Nothing else in the output — per-airport tables, the recovered replacement arm, the lane split, the
month bootstrap — can turn a failure into a pass. Every one of those is in the same category as
`compare.tail_report`, whose own heading says it does not change adopt/reject (`src/prc/compare.py`).
**[S-repo]**

---

## 3. The two lanes — and why only one of them is split

`src/prc/train.py:predict_fold` routes on `mvt_minus_aobt3`: rows with an NM off-block time go to
model A, rows without go to model B, and the two are separate fits. **[S-repo]** Per-airport splitting
could in principle apply to either.

### 3.1 Model A is estimable per airport

Labelled matched rows per airport, training side of each production fold. Measured by scanning
`data/work/feat_train.parquet`; **[S-repo]**.

| airport | forward-fold train rows (A) | of which the inner early-stopping holdout (Oct) |
|---|---:|---:|
| LSZH | 112,151 | 12,139 |
| LIRF | 135,960 | 14,027 |
| EDDM | 141,892 | 15,393 |
| LEBL | 150,832 | 16,106 |
| LEMD | 175,998 | 18,807 |
| EDDF | 192,927 | 21,112 |
| LFPG | 198,040 | 20,493 |
| EGLL | 198,890 | 20,566 |
| EHAM | 205,722 | 21,405 |
| LTFM | 225,161 | 23,713 |
| **total** | **1,737,573** | **183,761** |

Every airport clears OpenAir's 20,000-row refusal threshold by a factor of 5.6 or more **[S]**, and
clears Mansour et al.'s `m/p ≫ d_l` condition **[S]**. The smallest inner holdout (LSZH, 12,139 rows)
is larger than many whole datasets people early-stop on. **Per-airport is estimable on lane A.** **[X]**

### 3.2 Model B is not estimable per airport, and the counts say so plainly

Labelled unmatched rows per airport, by fold. **[S-repo]**

| airport | jan train | jul train | **forward train** | forward inner holdout | forward eval |
|---|---:|---:|---:|---:|---:|
| LEMD | 915 | 670 | **840** | 85 | 107 |
| EDDM | 945 | 838 | **845** | 76 | 165 |
| EGLL | 1,286 | 1,076 | **1,233** | 91 | 181 |
| LIRF | 1,428 | 1,151 | **1,348** | 115 | 140 |
| LEBL | 1,796 | 1,432 | **1,706** | 194 | 156 |
| EDDF | 1,792 | 1,521 | **1,768** | 127 | 188 |
| LSZH | 2,112 | 1,946 | **2,093** | 163 | 198 |
| LTFM | 3,413 | 3,220 | **3,005** | 367 | 640 |
| LFPG | 3,557 | 3,109 | **3,088** | 316 | 685 |
| EHAM | 3,791 | 3,569 | **3,539** | 379 | 545 |
| **total** | 21,035 | 18,532 | **19,465** | 1,923 | 3,005 |

A per-airport model B at LEMD would be a 255-leaf (or 127-leaf) GBDT fitted on **840 rows** with its
round count chosen on an **85-row** inner holdout, then evaluated on **107 rows**. Every airport is
below OpenAir's hard refusal threshold — all ten, on all three folds. **[S]** Phoenix-Ops did fit a
specialist on a 1,488-row scope and it worked, but only with five equal-weight seeds, and their
*non-LIRF* unmatched specialist came back **+0.012 s and worse at most airports** **[S]**.

**Decision: model B is not split, and the code cannot be asked to split it.** The flag reaches the
matched branch of `predict_fold` and nothing else (§9).

### 3.3 What lane B needs instead — and why that is not this plan

The lane split of the production OOF is worth writing down because it corrects a reading of the
per-airport numbers that this plan was briefed with. Measured on
`experiments/oof_main_gap-62dd7c6e88.parquet`; **[S-repo]**.

| airport | n (A) | RMSE A | % of total SSE | n (B) | RMSE B | % of total SSE | pooled RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| LFPG | 76,471 | 241.5 | 8.6 % | 1,565 | 2,622.3 | **20.8 %** | 441.7 |
| LIRF | 49,387 | 345.1 | 11.4 % | 537 | 4,059.2 | **17.1 %** | 543.2 |
| LTFM | 90,041 | 229.2 | 9.1 % | 1,297 | 503.1 | 0.6 % | 235.3 |
| EGLL | 78,986 | 222.1 | 7.5 % | 647 | 1,011.3 | 1.3 % | 239.2 |
| EHAM | 78,286 | 161.5 | 3.9 % | 1,353 | 475.0 | 0.6 % | 171.6 |
| EDDF | 71,489 | 175.3 | 4.2 % | 787 | 400.8 | 0.2 % | 179.3 |
| LEBL | 55,501 | 195.3 | 4.1 % | 652 | 362.5 | 0.2 % | 198.1 |
| LEMD | 70,316 | 171.6 | 4.0 % | 416 | 295.7 | 0.1 % | 172.6 |
| EDDM | 51,286 | 175.7 | 3.1 % | 402 | 724.6 | 0.4 % | 186.3 |
| LSZH | 42,287 | 167.2 | 2.3 % | 722 | 609.6 | 0.5 % | 183.6 |

**The pooled per-airport spread is mostly a lane-B spread.** 71 % of LFPG's SSE and 60 % of LIRF's sit
on 1,565 and 537 lane-B rows. In lane A alone the spread is 161.5 (EHAM) to 345.1 (LIRF) — still real,
but a different and much smaller problem than the pooled 171.6 → 543.2 suggests.

And the "the spread is variance, not bias" reading holds **only pooled**. Per-airport mean error
(pred − y), same file, **[S-repo]**:

| airport | pooled | lane A | **lane B** |
|---|---:|---:|---:|
| LIRF | +10.28 | +2.91 | **+688.72** |
| EDDM | −2.88 | −1.50 | **−178.23** |
| LFPG | +2.09 | +4.45 | **−113.01** |
| LSZH | +1.01 | −0.26 | **+75.50** |
| EHAM | −0.81 | −1.57 | +43.19 |
| EGLL | +10.57 | +11.00 | −41.90 |
| LEBL | +5.88 | +5.59 | +30.72 |
| LTFM | +5.65 | +6.14 | −28.00 |
| LEMD | +14.94 | +15.04 | −1.63 |
| EDDF | +2.24 | +2.25 | +0.99 |

Lane A is unbiased everywhere (−1.6 to +15.0 s). Lane B carries per-airport biases of hundreds of
seconds. **[X]** For squared loss, removing a within-group bias `b_g` removes exactly `n_g·b_g²` from
the SSE: LIRF's lane-B bias alone accounts for 537 × 688.72² ≈ **2.55e8**, about 0.49 % of the total
5.18e10 SSE, worth roughly −0.68 s pooled if it were removable. That is a **per-airport constant**, not
a per-airport model, and the 537 rows that carry it cannot support a model anyway. It belongs with P1
(the unmatched-lane specialist) and with `unmatched_sched_reparam`, which already targets exactly this
mechanism at exactly this airport. It is **out of scope here** and is recorded so it is not lost.

---

## 4. What the split costs, and what it buys

Both directions, with the mechanism named and, where possible, measured on our data.

### 4.1 Against the split

**(a) Ten times fewer rows per fit, and ten times fewer rows to early-stop on.** §3.1. The inner
holdout that picks the round count goes from 183,761 rows to 12,139–23,713. **[S-repo]** The round
count is then scaled to the refit set by `rounds = best · n/(n − n_hold)` (`src/prc/train.py`), so a
noisier `best` propagates straight into the shipped model. **[X]**

**(b) The weather response is fitted on a tenth of the distinct weather.** Weather is joined per
airport × hour, so its effective sample size is *hours*, not rows. Distinct
(temp, wind, precip, visibility) vectors: **874 (EGLL) – 1,682 (LFPG) per airport, against 12,625
pooled**. Measured **[S-repo]**. A pooled model estimates one weather response curve from 12,625
conditions; ten specialists estimate ten curves from ~1,200 each. This is the one mechanism with a
documented failure in this competition: OpenAir E23, verbatim — *"Airport experts absorb winter weather
(EDDM/LFPG/EHAM/LSZH). Jan+Jul ensemble is flat-to-worse and the >30 tail hurts. Do not ship."* **[S]**

**(c) Cross-airport physics stops being shared.** The response to the proxy offset, to queue and
congestion counts, to aircraft type and to operator is largely common across ten European hubs; a
pooled fit borrows strength for all of it. **[X]**

**(d) Thinner support for rare categorical levels.** Per airport, `AIRCRAFT_TYPE_mvt` has 42–195
levels against 270 pooled; `AIRCRAFT_OPERATOR_flt` 89–418 against 677; `ADES_mvt` 250–929 against
1,569. Measured **[S-repo]**. A type that flies twice a week from LSZH has ten times more support in
the pooled fit.

### 4.2 For the split

**(e) The categorical levels stop being diluted, and this is measurable.** `_matrix` encodes each
categorical as `pl.Enum(sorted(unique))` over the *raw string*, so one code serves one string at every
airport (`src/prc/train.py`). **[S-repo]** Measured on the feature frame **[S-repo]**:

* **520 of 1,899 `STAND_mvt` level names occur at more than one airport, and 1,199,842 of 2,085,047
  rows (57.5 %) carry such a name.** Stand "312" is one Enum code covering six physically different
  stands at six airports; "210" likewise.
* **15 of 53 `RUNWAY_mvt` names occur at more than one airport, covering 1,042,670 rows (50.0 %).**
  "34L", "26R", "08R", "27L" are each shared by two airports.

A specialist disambiguates these for free, because within its group the name is unique. The global
model can only recover it by spending splits on `ADEP_mvt` first, at 127 leaves. **[X]** This is the
strongest pro-specialist mechanism on our data and it is the one the survey does not mention.

**(f) Capacity.** Ten 127-leaf models are up to 1,270 leaves of total capacity against 127.

### 4.3 The measurements that separate the mechanisms

Three explanations predict the same sign, and each has its own separator. This matters more than the
verdict: a pass that is really (f) or (e) should be shipped as (f) or (e), which are cheaper and
carry less transfer risk.

| if the candidate passes, the gain could be… | the measurement that separates it | status |
|---|---|---|
| **generic blending** — averaging any two imperfectly correlated predictors | the **shuffled** noise arm, §5. Ten groups, same sizes, no airport information | **built into this plan** |
| **capacity** — 10×127 leaves beating 1×127 | P7's larger global model at roughly the summed capacity (`plans/P7-model-capacity.md`) | pre-existing plan; becomes a **required** follow-up before shipping |
| **categorical dilution** (e) | a global model with airport-qualified level names (`ADEP ‖ STAND`, `ADEP ‖ RUNWAY`). One feature change, one fit, no new architecture | **new, cheaper candidate**; run it *before* shipping ten models |
| **locality proper** — airports really need separate response surfaces | what is left after the three above | — |

**[X]** If the airport-qualified-levels control captures most of the gain, the answer is a two-line
feature change, not ten models — and it is the change with the better transfer story, since it does not
let any model memorise one airport's 2025 configuration.

---

## 5. Features that go constant inside an airport group

The KDD authors flag this specifically. Enumerated on our feature frame; all **[S-repo]**.

**Exactly constant within every group (1 distinct value in all ten):**

* **`ADEP_mvt`** — the categorical itself. This reproduces the KDD finding verbatim on our data: *"the
  local models cannot use it at all (it is constant within a group)"* **[S]**. LightGBM cannot split on
  a constant, so it is simply ignored. The column is **kept in the matrix anyway** so that the
  candidate's design matrix has the same shape and column order as the base arm's and the *only*
  difference between the arms is which rows enter which fit. Dropping it would be a second change.
* **`apt_med`** — the airport-median encoding, `airport_median_table` groups by `ADEP_mvt`
  (`src/prc/train.py`). One value per group, therefore dead in a specialist.
* **`ref_p10_fb`** on the rows where it falls through to `apt_med` — the coalesce is
  `ref_p10 → arh_med → apt_med`, so on rows with no stand×runway reference and no airport×runway×hour
  median it too is the group constant.

**Perfectly collinear within a group, i.e. one feature duplicated:**

* **`hour_local` and `hour_utc`.** Across airports these differ. Within an airport `hour_local` is
  `hour_utc` plus a fixed offset and a DST flag: at most **48 distinct (hour_utc, hour_local) pairs**
  per airport over 24 hour_utc values, and at **LTFM exactly 24** — Türkiye has no DST, so there the
  two columns are the same variable and the specialist loses one feature outright.

**Not constant, but thinned by an order of magnitude:** the 21 weather columns (§4.1b), and the
categorical level sets (§4.1d).

**Not constant at all:** no numeric feature comes close. The lowest within-airport-to-global variance
ratio over all 103 numeric columns is **0.674** (`rwy_tko_prev_60m`), then 0.698 (`mvt_minus_aobt3`).
There is no hidden family of airport-constant numerics.

### 5.1 The consequence that matters most

**All four label-derived encoding tables are already keyed on the airport** (`src/prc/train.py`):
`reference_table` groups by `[ADEP_mvt, STAND_mvt, RUNWAY_mvt]`; `airport_runway_hour_table` by
`[ADEP_mvt, RUNWAY_mvt, hour_local]`; `airport_median_table` by `ADEP_mvt`;
`operator_offset_table` by `[AIRCRAFT_OPERATOR_flt, ADEP_mvt]`. **[S-repo]**

So every group-level statistic a per-airport model could form from the labels is **already in the
feature matrix**, computed on the fold's training side, and handed to the global model. That is
precisely the condition the KDD paper says removes the specialists' advantage: *"the global model's
other features already encode what the airport identity would tell it"* **[S]**. It is the single
strongest reason in this document to expect a small or null result.

Consistent with that, the encodings are deliberately **fitted once on the whole training side of the
fold, before the split**, and every specialist is handed the same joined columns. Refitting them per
airport would change two things at once and would, for three of the four tables, change nothing at all.

---

## 6. The arms and the instrument

Population and shape follow the P4 stage-A pattern, whose one live calibration is the best the repo
has: it predicted −3.62 s and the production build returned −3.21 s. **[S-repo]**

### 6.1 Stage 0 — two gates, before any arm

**Gate 0a, the no-op / leak gate.** Run `configs/p11_base.yaml`. Its OOF **must be byte-identical to
`experiments/oof_p4a_gap-ca4297619f.parquet`.**

```
.venv/bin/python -m prc.pipeline --config configs/p11_base.yaml --skip-predict
cmp experiments/oof_p11_base-7708d22331.parquet experiments/oof_p4a_gap-ca4297619f.parquet
```

`p11_base.yaml` is `main_gap.yaml` + `folds: [forward]` + `airport_models: false`, which is exactly
what produced `p4a_gap-ca4297619f` (`configs/p4a_gap.yaml`). If the files differ, the five new
`TrainConfig` fields have moved the default path and **stage 1 does not run**. This is the control the
brief asks to be built in from the start, and it is the strongest form available: not "the numbers look
similar" but "the bytes are the same".

> **Note on ids.** `TrainConfig.hash()` hashes the whole `__dict__`, so adding fields changes the
> `exp_id` of *every* config in the repository. `main_gap.yaml` now hashes to `329a108956`, not
> `62dd7c6e88`; `p11_base.yaml` hashes to `7708d22331`. **The OOF contents are what is being compared,
> never the filename.** Archived files and archived records are untouched. **[S-repo]**

**Gate 0b, the determinism gate.** Run `configs/determinism_check_airport.yaml` twice, in two separate
processes, and require a byte-identical OOF.

```
for i in 1 2; do .venv/bin/python -m prc.pipeline --config configs/determinism_check_airport.yaml \
    --skip-predict --out-dir experiments; cp experiments/oof_determinism_check_airport-cd91236487.parquet \
    /tmp/det_airport_$i.parquet; done
cmp /tmp/det_airport_1.parquet /tmp/det_airport_2.parquet
```

Why a new determinism config is needed: `_matrix` builds each fit's categorical encoding as an explicit
`pl.Enum` over the frame it is handed, which is what fixed G11 (`src/prc/train.py`, and
`tests/test_train.py::test_the_design_matrix_is_a_function_of_the_data_not_of_its_chunking`)
**[S-repo]**. **A per-airport split hands every fit a different frame, so every fit gets a different
level set** — LSZH's specialist never sees LFPG's stands. That is legitimate and deterministic *as long
as* the choice of which rows go to which fit, and the order the fits run in, are functions of the data
alone. §9 makes both true by construction; gate 0b is the measurement that the construction holds on
the real 17-chunk frame, where G11 was found and where a single-chunk test could not have found it.

### 6.2 Stage 1 — three arms on the forward fold

| arm | config | what it is |
|---|---|---|
| A0 **base** | `configs/p11_base.yaml` | production recipe, splitting off |
| A1 **noise** | `configs/p11_noise_shuffled.yaml` | ten groups, same sizes, membership from a hash of `MVT_ID_mvt` |
| A2 **candidate** | `configs/p11_airport.yaml` | ten groups = ten airports |

The three configs are identical except for `airport_models` and `airport_group_key`. Population: the
`forward` fold, fit on 2025-01..10, evaluate on 2025-11..12, n = 328,009 (325,004 matched / 3,005
unmatched), 61 calendar days. One fold, `--skip-predict`, `n_seeds: 3`, `seed: 42`. Reason for one
fold: `jan` and `jul` train on months after the month they validate; forward is the only fold shaped
like the real task.

### 6.3 The noise arm, justified

The candidate changes three things against the base arm at once:

1. it averages two imperfectly correlated predictors — which lowers RMSE for **any** second model;
2. each second model trains on a tenth of the rows — which raises its error;
3. its rows are airport-homogeneous — **the only part the experiment is about**.

The shuffled arm holds (1) and (2) and removes (3). It builds the same number of groups with the same
size profile, taken from the training side's airport partition, and assigns membership from a murmur3
finaliser over `MVT_ID_mvt` — frame-independent, version-independent, written out in explicit uint64
arithmetic rather than taken from a library hash, so that a control whose membership drifted between
two runs cannot be mistaken for the effect it controls.

**What would make the noise arm move.** Genuine variance reduction from blending (down), and the cost
of fitting on ~174k rows instead of 1.74M (up). Either sign is a legitimate result. If it improves by
more than 0.5 s, the finding is *"our model wants to be an average of two models"*, which is a cheaper
and different candidate (more seeds, or a bagged pair) and routes to P7 — **not** to per-airport.

**What would make it stay put.** Those two effects cancelling. This is likely enough that **"noise ≈
base" must not be read as a no-op.** The no-op detector is structural, not numerical:

* `airport_groups` in the fold record must show **ten fitted groups**, with `train_rows` near 174k in
  A1 and spanning **112,151–225,161** in A2 (§3.1 — these are the exact expected values);
* `airport_n_groups_fitted` must be 10 and `airport_blended_rows` must be 325,004 in both;
* the three OOFs must not be equal to each other. **Three recipes returning the same number to six
  figures is this repository's known signature of a silent no-op or a leak, and it has happened
  twice.** If it happens, stop and find the bug.

### 6.4 The statistic and the decision rule — registered before any number exists

`prc.compare`, unchanged. `verdict()` unchanged: **ADOPT iff the K=0 CI and the K=200 CI both exclude
zero on the improving side.** No other statistic overrides it.

```
make compare A=<A0 exp_id> B=<A2 exp_id>      # the verdict
make compare A=<A1 exp_id> B=<A2 exp_id>      # the noise floor
make compare A=<A0 exp_id> B=<A1 exp_id>      # reported, for the record
```

Pass requires **all** of:

* (i) A2 vs A0 returns ADOPT;
* (ii) A2 vs A1 improves with the **K=0 CI excluding zero** and a point estimate of at least **0.5 s**
  — the resolution of this instrument is 0.25 s and a signal-free column has been measured at ±0.5 s
  on it (`docs/features_log.md`) **[S-repo]**;
* (iii) the month-level bootstrap agrees in sign with the row-level one;
* (iv) November and December agree in sign.

Reported and **never** allowed to change (i)–(iv): the per-airport table from `compare`'s
`per_group_diff`; the lane split; the recovered pure-specialist arm (§2.2) and its per-airport RMSE
against the global model's, which is the direct test of the `r < r*` condition of §2.2; the
`airport_groups` record; the K=200 drop list.

### 6.5 Stage 2 — confirmation on the production fold set

Only if stage 1 passes. One run: `configs/p11_airport_confirm.yaml`, folds `[jan, jul, forward]`,
compared against the **archived** production OOF `oof_main_gap-62dd7c6e88` — which gate 0a has already
shown the code change does not move, so base is not refitted.

```
make compare A=main_gap-62dd7c6e88 B=p11_airport_confirm-29d7fbcc7c
```

Must return ADOPT. `jul_fwd` is omitted because `run_cv` coalesces in fold order and `jul` reaches July
first, so `jul_fwd` never enters `overall` — the main_gap record's `folds_shadowed` says so. **[S-repo]**
The pooled OOF is identical and the run is ~13 minutes shorter.

---

## 7. The pre-registered price

**Written before any number exists. Register this block verbatim in the commit that lands this plan.**

### 7.1 What I expect

From §2.2, the forward fold's arithmetic (**[X]**, from the archived `p4a_gap-ca4297619f` record:
matched RMSE 198.526 on 325,004 rows, unmatched 1,003.595 on 3,005, pooled 219.725 on 328,009; the
unmatched lane is untouched by this change):

| assumed (ρ, r) | matched RMSE change | **pooled forward change** |
|---|---:|---:|
| ρ = 0.95, r = 1.00 | −1.26 % | **−2.23 s** |
| ρ = 0.97, r = 1.00 | −0.75 % | **−1.33 s** |
| ρ = 0.98, r = 1.00 | −0.50 % | **−0.89 s** |
| ρ = 0.99, r = 1.00 | −0.25 % | **−0.44 s** |
| ρ = 0.98, r = 1.02 | +0.10 % | **+0.18 s** |
| ρ = 0.97, r = 1.03 | +0.74 % | **+1.31 s** |

Our specialist is the same family, the same features and the same hyperparameters as the global model
on a subset of its rows, so ρ should sit **above** OpenAir's measured cross-family 0.96–0.98 **[S]** —
call it 0.98. r is the open question: KDD measured 1.004 at ten times our per-group row count **[S]**,
and §4 gives mechanisms in both directions.

**Point expectation: −0.7 s on the forward fold. Pre-registered band: −2.2 s to +1.3 s.**

**[X]** I put more mass near zero than at either end, for the reason in §5.1: all four of our encoding
tables are already keyed on the airport, which is the exact condition under which the KDD paper reports
the specialists stop paying. The single most useful number the run will produce is not the verdict — it
is the recovered specialist's matched RMSE. **If it is above ~200.5 (r > 1.01 at ρ = 0.98), the blend
cannot help and the theme is closed at every weight**, and that conclusion does not depend on the CI.

### 7.2 What counts as a failure

Any one of these. On any of them, **stop**:

1. Gate 0a: `p11_base`'s OOF is not byte-identical to `oof_p4a_gap-ca4297619f`.
2. Gate 0b: two runs of `determinism_check_airport` differ.
3. A2 vs A0: the K=0 CI includes zero, or excludes zero on the worse side.
4. A2 vs A0: K=0 passes but the **K=200 CI includes zero** — report as **tail-only**. This is the
   signature of the +13.20 s board regression (`docs/README.md:110`) **[S-repo]**, and for a change
   whose claimed mechanism is *uniform variance reduction over 325,004 rows*, a tail-only result is
   self-refuting: variance reduction that survives only on 200 rows was not variance reduction.
5. A2 does not beat A1 by ≥ 0.5 s with the K=0 CI excluding zero — result is **"indistinguishable from
   blending with any second model"**.
6. The month bootstrap and the row bootstrap disagree in sign, or November and December disagree.
7. Stage 2 does not return ADOPT.
8. Any two of the three arms produce identical OOFs, or `airport_groups` does not show ten fitted
   groups with the row counts of §3.1 and §6.3.

### 7.3 What is *not* permitted on failure

Do not re-cut the interval. Do not switch folds. Do not sweep the blend weight. Do not gate the
specialists per airport to recover the effect (§2.3) — that is the post-hoc selection the whole design
exists to refuse. Do not split model B (§3.2). Record the number, close the theme, and hand the compute
to P7, whose larger-global-model arm is the control Montero-Manso & Hyndman's result says should have
been run first anyway **[S]**.

**One pre-declared exemption, and only one.** If A2 passes (i) and fails only the ≥ 0.5 s magnitude of
(ii) while the sign of A2 − A1 is improving, one follow-up is permitted: refit A1 and A2 with
`n_seeds: 5` to see whether the margin was seed noise. Registered here so it is not a post-hoc choice;
its outcome is reported whether or not it helps.

### 7.4 If it passes

**Do not ship ten models on the strength of stage 2.** Two controls come first, both from §4.3:

* the **airport-qualified categorical levels** control (`ADEP ‖ STAND`, `ADEP ‖ RUNWAY`) — one feature
  change, one fit. If it captures most of the gain, ship that instead: it is one model, it is cheaper,
  and it does not let any model memorise one airport's 2025 configuration.
* **P7's larger global model** — if it closes the gap, the result was capacity, not locality.

---

## 8. Compute cost, honestly

Anchors, all **[S-repo]** from `experiments/experiments.jsonl`, 4 threads, warm feature cache:

| run | folds | seeds | wall clock |
|---|---|---:|---:|
| `p4a_base-58b999e0b7` | forward | 3 | 695.8 s |
| `p4a_noise-1cd24d9c30` | forward | 3 | 709.3 s |
| `p4a_gap-ca4297619f` | forward | 3 | 814.3 s |
| `feat_base-b9e21a829d` | forward | 1 | 305.5 s |
| `main_gap-62dd7c6e88` | 4 folds | 3 | 3,253.2 s |

**[X]** Scaling argument for the grouped arm. A grouped fit is **eleven** fits, not ten: the global fit
is still needed, for the blend and as the fallback. LightGBM's per-iteration cost at 127 leaves is
dominated by histogram construction, which is O(rows × features) per depth level; split finding is
O(leaves × features × bins) and is smaller by a factor of ~375 at 1.74M rows and ~37 at 174k. Summing
histogram work over ten groups of a tenth each gives **the same total as one global fit**; the
split-finding term is paid ten times but starts negligible. So the specialists together cost roughly
one global fit, ×0.7–1.5 for the per-group round counts differing from the global one, plus ten sets of
fixed overhead (frame filter, `_matrix` conversion, `lgb.Dataset` binning — all linear in rows, so
their totals are unchanged).

| item | estimate |
|---|---|
| gate 0a, base, forward, 3 seeds | ~14 min |
| gate 0b, determinism pair, forward, 1 seed, grouped | 2 × 9–13 min = **18–26 min** |
| stage 1, A0 base | ~14 min |
| stage 1, A1 shuffled | **23–40 min** |
| stage 1, A2 airport | **23–40 min** |
| stage 2, confirm, 3 folds, grouped | **60–100 min** |
| **total, sequential** | **2.5 – 4.0 h of exclusive 4-CPU time** |

**Abort criterion.** If a single grouped forward-fold arm exceeds **60 minutes**, kill it and report the
`airport_groups` round counts: that means per-group early stopping is selecting round counts far above
the global fit's ~1,134, which is itself the finding (the specialists are underfitting on their own
inner holdouts) and is worth more than finishing the run.

**Memory, and the OOM history.** Peak memory is **unchanged**, and on average lower. `_fit` stacks
train and target and materialises one float32 matrix: ~2.1M × 99 × 4 B ≈ 830 MB for the global fit,
plus the binned `lgb.Dataset`. Each grouped fit builds ~a tenth of that and is freed at the end of its
loop iteration (§9). The KDD paper's own stated reason for preferring local models is memory **[S]**.
The risk here is not this run's footprint; it is **concurrency** — this project has been OOM-killed
twice by two heavy jobs at once. **Run one arm at a time, in one process, and never alongside another
agent's measurement.**

---

## 9. The code change

`src/prc/train.py` only. Behind a flag that defaults to the current behaviour; no existing recipe
changes. Written and tested on synthetic data; **no pipeline run was started**.

### 9.1 Five new `TrainConfig` fields, all inert by default

```
airport_models: bool = False        # master flag
airport_group_key: str = "adep"     # "adep" (candidate) | "shuffled" (control)
airport_blend_weight: float = 0.5   # (1-w)*global + w*group ; 0 = global, 1 = pure specialist
airport_min_group_rows: int = 20000 # below this the group keeps the global prediction
airport_group_seed: int = 20260924  # read only by "shuffled"
```

### 9.2 Three new functions

* **`_row_uniform(ids, seed)`** — a uniform [0,1) draw per row, the murmur3 finaliser written out in
  uint64 arithmetic. Not `polars.hash`, not `numpy.random`: a library hash may change between versions,
  and a control arm whose membership moved between two runs would be indistinguishable from the effect
  it controls for. Frame-independent, so a row's group cannot depend on which other rows share its
  frame. Refuses non-integral ids.
* **`_group_labels(train, target, cfg)`** — the group of every row in both frames, plus the **sorted**
  list of groups. `adep`: `ADEP_mvt`, nulls to `__NA__`. `shuffled`: group count and size profile
  copied from the training side's airport partition, membership from `_row_uniform`. A target row whose
  group has no training rows keeps the global prediction and is counted.
* **`_fit_grouped(...)`** — the global `_fit`, then one `_fit` per group in sorted order, blended in
  **model space** (before `_assemble`'s floor and the LOBT bound, which is what makes the
  pure-specialist arm recoverable as `2B − A`). Each group's frames are filtered **inside** the loop and
  deleted at the end of it, so peak memory stays at the global fit's.

### 9.3 The call site

```python
if cfg.airport_models:
    g = _fit_grouped(m_tr, m_tg, _target_a, cfg, cols_a, cat_cols, info)
else:
    (g,), rounds = _fit(m_tr, _target_a, cfg, cols_a, cat_cols, [m_tg])
    info["rounds_matched"] = rounds
```

Matched branch only. The unmatched branch is untouched (§3.2). With the flag off the call is the single
`_fit` it has always been; the only textual change on that path is that the target lambda became a named
function.

### 9.4 What goes into the record

`info["airport_groups"]` — per group: `train_rows`, `target_rows`, `fitted`, `rounds`. Plus
`airport_group_key`, `airport_blend_weight`, `airport_n_groups_fitted`, `airport_blended_rows`,
`airport_ungrouped_target_rows`. This is §6.3's structural no-op detector: a change that silently did
nothing would still print ten groups with the right row counts, or it would print nothing at all, and
either way the record says which.

### 9.5 How determinism is preserved

1. Group membership is a pure function of the row (`ADEP_mvt`, or `_row_uniform(MVT_ID_mvt)`), never of
   frame order, chunking or process.
2. The group list is **sorted**, so the loop order is a function of the data.
3. No seed is derived from the loop index — every group is fitted at `cfg.seed`, so reordering the
   loop could not change a fit even if it happened.
4. Each group's `_fit` builds its own `pl.Enum(sorted(unique))` over its own stacked train+target
   frame, which is deterministic per the G11 fix. The level sets differ between groups; that is the
   point, and it is not the bug.
5. `configs/determinism_check_airport.yaml` is the real-data gate (§6.1b).

### 9.6 Tests added, `tests/test_train.py`

| test | what it pins |
|---|---|
| `test_airport_models_are_off_by_default_and_the_default_path_is_the_old_one` | with the flag off, moving the key, weight, floor and seed leaves every prediction bit-identical and keeps the new bookkeeping out of `info` |
| `test_a_zero_weight_blend_is_the_global_model_and_a_full_weight_blend_is_the_specialist` | w=0 is bit-identical to the unsplit model; w=0.5 is exactly the midpoint of w=0 and w=1 — the arithmetic the `2B − A` recovery rests on |
| `test_every_airport_is_its_own_group_and_the_record_says_how_many_rows_each_one_got` | the group set is the airport set, per-group `train_rows` match a direct count, every target row is accounted for |
| `test_a_group_below_the_floor_keeps_the_global_prediction_untouched` | the floor refuses rather than degrades |
| `test_the_shuffled_control_matches_the_airport_partition_in_count_and_size_but_not_in_membership` | the control costs what the candidate costs and carries no airport information |
| `test_group_membership_is_a_function_of_the_row_id_alone` | reordering and splitting the frame does not move a row's group; repeated draws agree; different seeds differ |
| `test_the_grouped_fit_refuses_a_weight_outside_the_unit_interval_and_an_unknown_key` | bad config fails loudly |
| `test_a_grouped_run_is_reproducible_across_processes` | two grouped fits of the same config agree byte for byte (in-repo twin of gate 0b) |

`make test`: **85 passed**. `ruff check src` clean; `ruff check tests` reports only the pre-existing
import-order finding on `tests/test_train.py:1`, untouched by this change.

---

## 10. Order of operations

```
0a  p11_base           forward, 3 seeds   -> byte-compare with oof_p4a_gap-ca4297619f.parquet
0b  determinism_check_airport  x2         -> byte-compare the two OOFs
1   p11_base (reuse 0a) / p11_noise_shuffled / p11_airport, one process at a time
    make compare A=base B=airport         -> verdict
    make compare A=noise B=airport        -> noise floor, >= 0.5 s
    make compare A=base B=noise           -> for the record
    report: recovered specialist RMSE (2B - A) per airport, against the global model's
2   p11_airport_confirm  (only if 1 passed)
    make compare A=main_gap-62dd7c6e88 B=p11_airport_confirm-...
3   before shipping: the airport-qualified-levels control and P7's larger global model (§7.4)
```

---

## 11. The risk this design cannot eliminate

Every fold in this design lives inside 2025. The ranking set is 2026.

A per-airport specialist is, by construction, the model best able to memorise one airport's 2025
operating configuration — its runway mode, its stand allocation, its construction works, a based
carrier's schedule. Over a 61-day validation window inside the same year, that memorisation *is* signal
and will be scored as such. If any of the ten airports changed configuration between 2025 and 2026,
the specialist's advantage at that airport in CV is exactly the part that will not transfer, and it
will transfer *worse* than the global model's, because the global model was regularised toward the
other nine airports the whole time.

No fold inside 2025 can see this. Our adversarial score measures covariate shift and our own code
records that it does not explain the CV/LB divergence (`EXECUTION-ORDER.md`) **[S-repo]**. The 2026
arrival mirror carries real taxi-**in** labels and could bound airport-level outcome drift, which is
the nearest available instrument and is on the auditor's unplanned list **[S-repo]** — but it measures
a different quantity on a different phase.

**[X]** This is the reason §7.4 requires the two cheaper controls before shipping ten models even after
a pass: airport-qualified categorical levels and a larger global model both capture the mechanisms of
§4.2 **without** giving any model a private 2025 memory of one airport. If the honest gain is in (e) or
(f), those are the versions that survive the year boundary.
