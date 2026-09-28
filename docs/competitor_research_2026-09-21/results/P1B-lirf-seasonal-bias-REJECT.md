# P1-B — the LIRF unmatched seasonal bias: the mechanism is real, the correction is **REJECT**

Measured 2026-09-24. One script, read-only:
`docs/competitor_research_2026-09-21/scripts/p1b_lirf_seasonal_bias.py` (sections `S1`–`S12`;
`.venv/bin/python <script> <section numbers>` runs a subset). Nothing was submitted, nothing was
written to `outputs/`, no config was changed, no model was refitted — the candidate is a post-hoc
overlay, so all three arms are computable from the existing OOF parquets.

---

## 0. The answer

1. **Not closed** by `P1-stage2-gate-FAILED.md`. That document closed a *per-row fill classifier*
   used to replace the hard LIRF switch; this candidate makes no per-row claim and corrects a first
   moment on a cell that is 100 % identifiable at prediction time. Different route (§1).
2. **The mechanism is real, but one link of the stated chain is false.** Fill rate is seasonal, the
   model carries a blended mixture weight, the scored months are the two lowest-fill ones, and the
   residual mean moves accordingly (§2). But *"`month` is not a feature, so the model has no way to
   track this"* does not hold: `temp_c` **is** a model-B feature, calendar month explains
   **R² = 0.735** of it at LIRF, and `temp_c` separates the fill label at in-sample AUC **0.6676**
   — slightly *better* than the calendar month's own 0.6484 (§2.5). The model is not blind to the
   season; it declines to use it. And the t is computed on rows whose Kish effective sample size is
   **27, not 383** (§2.4).
3. **The headline price is stale.** +1,356.2 s / t = 6.47 reproduces exactly, but on
   `oof_main_reparam-bc60017878`, a recipe we no longer ship. On the production OOF
   (`oof_main_gap-62dd7c6e88`) the same population's bias is **+1,206.2 s, t = 5.96**, worth
   **2.58 s** SE-adjusted, not 3.26 s: `stand_arrival_gap` has already collected part of it (§2.1).
4. **Three arms, three honest estimators, and the rule rejects all of them** (§3). So does the
   **oracle arm** — each month's own bias removed, circular by construction — which is
   `−1.75 s [−2.91, −0.59]` at K=0 and `−0.11 s [−0.53, +0.31]` at K=200. **No version of this
   candidate, including a perfect one, can be adopted by `prc.compare.verdict`.**
5. **The intervention is the second kind**: a *post-hoc constant added to a subset of predictions*,
   not a feature. Its risk is stated in those words in §4, priced in §5: the downside on the 356
   scored rows runs to **+1.92 s if the bias is simply absent in 2026 and +4.80 s if 2026 looks
   like November/December**, against an upside of −2.35 s.
6. **New evidence on the transfer risk the earlier note called unmeasurable.** A prefix fill-rate
   table fitted on 2025 and applied to the scored rows estimates 2026-07 at fill **0.290** (2025-07:
   0.326 — consistent) but 2026-**01** at fill **0.589** (2025-01: 0.455), i.e. January's scored
   population looks like a *high*-fill month, where the bias is negative. **91 of the 356 scored
   rows may carry the shift in the wrong direction** (§6).

**Verdict: REJECT.** Not "no effect" — the effect is real and is the largest identified bias in the
frame. It is rejected because no estimator that is honest about what is known before January 2026
recovers it, and because the one that comes closest is a bet whose loss branch is larger than its
win branch on the evidence available.

---

## 1. Is it already closed by `P1-stage2-gate-FAILED.md`? No.

| | what stage 2 closed | this candidate |
|---|---|---|
| object | a per-row fill probability `p`, gate 0.93, reached 0.9201 | the mean of the residual on one visible cell |
| use | `pred = p·ms + (1−p)·body` — a **row-level** reassignment | `pred = pred − S` — a **first-moment** correction |
| failure mode it must avoid | mis-assigning individual rows | none at the row level; `ΔSSE = n(S² − 2Sb)` depends only on `b` |
| where it attaches | `train.py::predict_fold`, model-B branch, inside the fold | after `predict_fold` returns |

Closing a mean bias `b` on `n` rows removes `n·b²` of SSE **whatever the per-row accuracy is**. An
AUC gate does not bound that quantity, so Stage 2's 0.9201 does not close this. Structurally the
candidate is a sibling of **Stage 3** (the LIRF schedule-floor overlay), which the plan did not
close, not of Stages 4–5.

Two things from that document do bear on it and are recorded here rather than buried:

* Calendar month, used as a fill score on the 1,488 LIRF unmatched training rows, has **in-sample
  AUC 0.6453** (`S5`) against the 0.9201 that failed. As a *classifier* the candidate's signal is
  far weaker than the one already rejected. It is only admissible because it is not being used as a
  classifier.
* The honest out-of-month version of that score **does not exist**: a calendar month has no second
  instance in a single training year, so nothing estimates month *m*'s fill rate without using
  month *m* (`S5`). This is the same structural fact that makes every feature form of this
  candidate unmeasurable under `month_block_folds` (§4).

---

## 2. Is the mechanism real, link by link?

### 2.1 The four "confirmed facts", checked

| claim as given | measured | section |
|---|---|---|
| LIRF sched-fill rate 0.323 (Jul) → 0.761 (Dec) | **exact** on all 1,488 LIRF unmatched rows: Jul 0.3234, Dec 0.7614 | `S1` |
| `month` is not a feature | **true** — `train.py:244` drops it by name. But `temp_c`, `temp_c_prev1h`, `deicing_gate`, `dewp_spread_c` **are** in model B's 75 numeric columns | `S1`, code read |
| Jan and Jul are among the lowest-fill months | **true** — Jul 0.3234 is the lowest, Jan 0.4167 the second lowest of twelve | `S1` |
| +1,356.2 s, t = 6.47, on 383 LIRF unmatched OOF rows | **exact, on `oof_main_reparam-bc60017878`**: n = 383, bias +1,356.2, SE 209.6, **t = 6.47**. (`large-ms-population.md` §6.2 prints SE 220 / t 6.15 for the same figure; the brief's 6.47 is the one that reproduces.) | `S2` |
| 356 scored rows in the corresponding population | **exact** — LIRF ∧ unmatched ∧ `ms ≥ 1800` = 356 scored rows (91 Jan, 265 Jul) | `S1` |
| `sched_reparam_train_rows: 1488`, `target_rows: 383` | **true, and the two 383s are different quantities that coincide.** 383 in the experiment record is the count of *ranking* rows the reparameterisation shifts; 383 in the bias figure is the count of *OOF* rows in months 1+7 with `ms ≥ 1800`. The pooled OOF LIRF-unmatched population is **537** rows (512 at `ms ≥ 1800`), not 383 | `S1`, `S2` |

**One "confirmed fact" is stale, and it is the price.** The +1,356.2 was measured on
`main_reparam`. Production is `main_gap`. On `oof_main_gap-62dd7c6e88`, same cell, same months:

| OOF | n | RMSE | bias | SE | t |
|---|---:|---:|---:|---:|---:|
| `main_reparam-bc60017878`, months 1+7, `ms ≥ 1800` | 383 | 4,314.4 | **+1,356.2** | 209.6 | **6.47** |
| **`main_gap-62dd7c6e88`** (production), same rows | 383 | 4,134.3 | **+1,206.2** | 202.3 | **5.96** |

SE-adjusted (`max(0, b̂² − SE²)`, the estimator `large-ms-population.md` §4.2 uses), on 356 scored
rows and `1 s = 1.9534e8` of SSE: **3.26 s → 2.58 s**. `stand_arrival_gap` has already removed
11.1 % of the raw bias and 0.68 s (21 %) of the headline price. `S2`

### 2.2 Fill rate varies by month → yes

`S1`, LIRF unmatched, `ms ≥ 1800` (n = 1,412): 0.455 / 0.788 / 0.796 / 0.672 / 0.495 / 0.483 /
**0.326** / 0.456 / 0.436 / 0.491 / 0.627 / **0.756** for months 1–12.

### 2.3 The model carries a blended mixture weight → yes, and this is the whole of it

`S3`, production OOF, B-LIRF `ms ≥ 1800`, 512 rows:

| fill status | n | bias | SE |
|---|---:|---:|---:|
| filled (`\|BLOCK−SCHED\| ≤ 10 s`) | 223 | **−1,267.5** | 93.5 |
| not filled | 289 | **+2,273.3** | 278.0 |

The two signs are the two sides of one straddled discontinuity. The month bias is then almost pure
composition — `fill·(−1,267) + (1−fill)·(+2,273)` against the observed month bias:

| month | n | fill | observed bias | composition prediction | residual |
|---|---:|---:|---:|---:|---:|
| 1 | 55 | 0.455 | +936.2 | +663.8 | +272.4 |
| 7 | 328 | 0.326 | +1,251.5 | +1,118.2 | +133.3 |
| 11 | 51 | 0.627 | −1,239.3 | +51.6 | **−1,290.9** |
| 12 | 78 | 0.756 | −313.8 | −405.0 | +91.2 |

Three of four months are explained by composition to within 275 s. **November is not** — it misses
by 1,291 s on 51 rows. So `month` is *mostly* a proxy for the month's fill rate, and there is at
least one month where something else dominates. `S3`

Note also that the pooled weights do not cancel: at the training-average fill of 0.489 the
composition formula still gives **+540 s**, so part of the B-LIRF bias is a level error that has
nothing to do with season.

### 2.4 Is the t computed on independent rows? No.

`S4`, months 1+7, B-LIRF `ms ≥ 1800`, n = 383, bias +1,206.2:

| unit resampled | blocks | CI95 on the bias | SE |
|---|---:|---|---:|
| rows (iid, the t = 5.96 assumption) | 383 | — | 202.3 |
| **day** | 54 | [+730.5, +1,641.3] | **229.3** |
| month | 2 | [+936.2, +1,251.5] | 125.0 |

**Kish effective n over day blocks = 27.0, against n = 383.** The day-block SE is 13 % wider than
the iid one, so the bias is still comfortably non-zero *within 2025*. But the quantity that has to
transfer to 2026 is the *month-level* bias, and of that there are exactly **two** observations in
the scored season — month 1 at +936.2 (iid t = **1.51**, n = 55) and month 7 at +1,251.5
(iid t = 5.89, n = 328). **The January half of the finding is not individually significant.** `S4`

### 2.5 "`month` is not a feature, so the model cannot track this" — this link is false

`S12`. `month` is indeed dropped by name at `train.py:244`. But `temp_c` and `temp_c_prev1h` are
among model B's 75 numeric columns, and at LIRF the calendar month explains **R² = 0.735** of
`temp_c` (monthly means run 10.6 °C in January to 27.0 °C in August, within-month sd 3.1–5.2).
Against the fill label on the same 1,410 rows:

| score | in-sample AUC |
|---|---:|
| `temp_c` (a column the model already has) | **0.6676** |
| the calendar month's own fill rate (the excluded variable) | 0.6484 |
| corr(month mean temp, month fill rate) | **−0.731** |

**The model holds a feature that separates the fill label slightly better than `month` would, and
still carries the bias.** So the excluded-variable story is not sufficient: the honest statement is
that the seasonal signal is available to the model and is too weak, on 1,488 LIRF unmatched training
rows, for squared-error boosting to act on it. That is also the reason a *feature* form of this
candidate is not obviously better than a shift — the information is already in the frame.

### 2.6 Conclusion on step 2

**The mechanism is real.** Fill rate is seasonal, the model carries a blended weight, the scored
months sit at the low-fill end, and the residual mean moves accordingly. It is not an artefact of
the estimator. What *is* an artefact of the estimator is the precision — t = 6 describes 27
effective units and 2 month-level draws — and one link of the stated chain (§2.5) does not hold.

---

## 3. The three-arm gate

Because the intervention is post-hoc, the three arms are three prediction vectors over the **same
672,428 OOF rows from the same fit** — a stricter like-for-like than the `stand_arrival_gap` gate
could achieve, since no refit noise enters at all. The noise arm is the *identical estimator*
applied to an equally sized random cell drawn from the non-LIRF unmatched `ms ≥ 1800` rows, which
`large-ms-population.md` §4.2 measures as unbiased (bias −37 s); five draws. `S6`, `S9`

Three honest estimators of the shift were built, because the choice of estimator *is* the design:

| form | shift for month *m* | what it assumes |
|---|---|---|
| **C1** | −mean residual of the **other OOF months** | nothing seasonal |
| **C2** | −the **other scored month's** bias (Jan ↔ Jul; Nov/Dec untouched) | that 2026 Jan/Jul behave like 2025 Jan/Jul — this is the form `large-ms-population.md` §6.2 measured |
| **C3** | −(â + ĉ·fill*ₘ*), the line `bias ~ fill` fitted on the **other three** OOF months | that the month→bias map runs through the fill rate |

### 3.1 What the shifts actually are

`S6`/`S8`. C1 is wrong-signed for July: the other three OOF months average **−196.7 s**, so C1
*adds* 196.7 s to the predictions in the month with the largest positive bias. C3's leave-one-out
predictions miss November by 1,435 s and December by 1,813 s, and hit January (+415 vs +936) and
July (+1,115 vs +1,252). **The estimator gets two of its four months badly wrong, and the two it
gets right are the two scored ones.**

### 3.2 The gate, whole OOF (`main_gap`, 672,428 rows)

| arm | K=0 | K=200 | month bootstrap (4 months) | `verdict` |
|---|---|---|---|---|
| C1 vs base | **+1.084** [+0.588, +1.688] | +0.435 [+0.184, +0.654] | +1.084 [+0.075, +1.976] | **REJECT** — worse on all rows |
| C1 vs noise | +1.083 [+0.587, +1.687] | +0.433 [+0.183, +0.652] | — | REJECT |
| C2 vs base | **−1.409** [−2.325, −0.546] | −0.201 [−0.573, +0.147] | −1.409 [−3.427, **+0.000**] | **REJECT** — tail-only |
| C2 vs noise | −1.405 [−2.324, −0.543] | −0.205 [−0.578, +0.145] | — | REJECT |
| C3 vs base | −0.717 [−1.699, +0.279] | +0.190 [−0.287, +0.655] | −0.717 [−3.449, +2.166] | **REJECT** — no difference |
| C3 vs noise | −0.715 [−1.696, +0.273] | +0.186 [−0.291, +0.656] | — | REJECT |
| **oracle** (each month's own bias, circular) | −1.750 [−2.909, −0.593] | **−0.109** [−0.532, +0.308] | — | **REJECT** — tail-only |
| noise vs base (5 draws each form) | median +0.001, range [−0.006, +0.047] | median +0.002–0.008, range [−0.002, +0.067] | — | — |

**The candidate's margin against the noise arm equals its margin against base to within 0.01 s in
every cell.** The noise arm moves the whole-OOF RMSE by at most 0.047 s, so it neither manufactures
nor masks anything here; that is what makes each rejection the candidate's own, not the control's.

**The oracle line is the decisive one.** It is not an achievable arm — it removes each OOF month's
own mean residual, which is exactly the circularity the brief warned about — and it still fails the
rule, because at K=200 the effect has gone. 512 of the 672,428 rows cannot survive a 200-row drop
when the drop order is by squared error and those 512 rows carry RMSE 4,126 against the frame's 278.
**`prc.compare.verdict` cannot adopt any member of this family.** That is `P1-unmatched-lane.md`
§4.2's prediction ("the gate will reject real wins in this lane"), now measured rather than argued.

### 3.3 Per fold view

| arm | jan+jul folds (n = 344,419) | forward fold (n = 328,009) |
|---|---|---|
| C1 | K=0 +0.578 [+0.114, +1.082], K=200 +0.136 — REJECT | K=0 **+1.909** [+0.946, +3.109], K=200 +0.248 [+0.047, +0.467] — REJECT |
| C2 | K=0 **−2.365** [−3.676, −0.784], K=200 −0.504 [−1.109, +0.084] — **REJECT (tail-only)**; vs noise −2.358, K=200 −0.512 | +0.000 — C2 leaves Nov/Dec untouched by construction |
| C3 | K=0 **−2.447** [−3.742, −0.869], K=200 **−0.675** [−1.195, −0.132] — **ADOPT**; vs noise −2.441, K=200 −0.678 [−1.199, −0.141] — **ADOPT**; month bootstrap −2.447 [−4.416, −0.314] | K=0 **+1.902** [+0.685, +3.171], K=200 +0.113 [−0.112, +0.349] — REJECT |

C1 is the only form that assumes nothing seasonal, and it is the form that loses hardest — **+1.909 s
on the forward fold**, because a year-average constant fitted across months whose biases are
`+936 / +1,252 / −1,239 / −314` is a constant fitted to a sign change.

**C3 passes the rule on the jan+jul folds and only there.** Three reasons that is not an adoption:

1. It is fold selection after the fact. The forward fold is `docs/README` §6's deciding fold, and it
   is the one where the candidate is worst, because November and December are the two *highest*-fill
   months and the bias there has the opposite sign. Choosing the fold subset on which a seasonal
   correction works is choosing the months whose season you already knew.
2. The jan and jul folds are the two anti-causal ones — each trains on the months that follow it
   (`P3-oof-and-validation.md` §3, `cv.py:39-42`).
3. C3's own leave-one-out diagnostics (§3.1) say the estimator is unreliable in half the months it
   was tested on. It is right about Jan and Jul by 2 draws out of 4.

To be explicit about what did pass: on the jan+jul rows C3 clears both arms of the rule **and** beats
the noise control by essentially its whole margin (−2.441 at K=0, −0.678 at K=200, both CIs on the
improving side) **and** its two-month bootstrap excludes zero. That is a genuine ADOPT under the
`stand_arrival_gap` protocol, on the two folds that match the scored season. It is reported as a
REJECT because the same estimator costs **+1.902 s** on the deciding fold and leaves the pooled OOF
at −0.717 [−1.699, +0.279] — no difference — and because §5 and §6 say the win branch is the one
that has to hold for both scored months and only one of them is supported.

### 3.4 Where the change lives

`S6`. C2 on the candidate cell: 512 rows, RMSE 4,126.3 → 4,000.3, **−126.0 s [−200.0, −43.2]** —
the cell-level effect is unambiguous. Split by month: January −84.5 s [−535.0, +191.0] (not
significant), July −185.9 s [−302.6, −67.1]. **The cell effect is July's.**

---

## 4. The form of the intervention and its risk class, stated plainly

**This is the second kind: a post-hoc constant added to a subset of predictions.** It is not a
feature. It does not let squared error find the conditional mean by itself. It is a bet that a bias
estimated on 2025 months persists into January and July 2026, applied to 356 rows we will not see
labelled.

It is the second kind **by necessity, not by choice.** The feature form is not merely worse — it is
**unmeasurable under this repository's CV design**:

* `month_block_folds` holds out an entire calendar month (`cv.py:26-51`). In fold `jan`, no
  January row is in training, so a `month` feature offers the model an unseen level and a
  `fill_rate_of_this_calendar_month` encoding has nothing to be computed from.
* At serve time the situation is the opposite: predicting 2026-01, the model trains on 2025-01.
* So **every fold understates the feature form and no fold can measure it.** `S5` states the same
  fact from the classifier side: a calendar month has no second instance in one training year.

The only feature form that *is* in-fold computable is a per-row fill signal keyed on something other
than the calendar — flight prefix, operator — and that is exactly the route
`P1-stage2-gate-FAILED.md` closed at AUC 0.9201.

---

## 5. What the bet costs if the bias does not persist

`S11`. C2's shifts are −1,251.5 s on January's 91 scored rows and −936.2 s on July's 265.
`ΔSSE = n(S² − 2Sb)`; `1 s = 1.9534e8`; negative is better.

| belief about the true 2026 bias `b` | month 1 | month 7 | **total** |
|---|---:|---:|---:|
| 2026 = 2025 same month (+936 / +1,252) — the bet | −0.36 s | −1.99 s | **−2.35 s** |
| §6's calibrated fill estimate (−75 / +1,384) | **+0.82 s** | −2.33 s | **−1.51 s** |
| 2025's own fill rates through the `bias ~ fill` line (+581 / +1,208) | +0.05 s | −1.88 s | −1.83 s |
| **no bias at all** (`b = 0`) | +0.73 s | +1.19 s | **+1.92 s** |
| **2026 looks like Nov/Dec** (`b = −777`) | +1.64 s | +3.16 s | **+4.80 s** |

**Break-even is `b = S/2`: +626 s for January, +468 s for July.** The upside is −2.35 s; the
downside if the seasonal regime has moved is up to +4.80 s. For comparison, v11 — a post-hoc rule on
**11** LIRF unmatched rows, with its own expected-value case written out before the upload — cost
**+9.62 s** live (`experiments/submissions.jsonl`: v10 284.3584 → v11 293.978), and the whole
reparameterisation that this cell sits on was worth −20.67 s. The board today is 282.7371 (v13).
This is a bet at roughly 1:2 stakes on a quantity with two month-level observations, on the same
population and the same kind of rule that has already cost this project 9.6 s once.

---

## 6. New: the transfer risk is no longer unmeasured

`large-ms-population.md` §7 lists "the transfer to 2026 is assumed" as inference, and
`reparam-implementation.md` §7 records the fill rate of the 383 LIRF unmatched ranking rows as
**unknown**. It cannot be measured — `BLOCK_TIME` is not in the ranking frame — but the scored
rows' *composition* can be, and it is informative. `S10`

`FLIGHT_mvt` is present on 360 of the 383 scored LIRF unmatched rows (23 null) while
`AIRCRAFT_OPERATOR_flt`, `MARKET_SEGMENT_flt` and `AIRCRAFT_TYPE_mvt` are null on all 383. A
3-character prefix fill table with shrinkage `K = 5`, validated leave-one-month-out over the twelve
2025 months, tracks the month fill rate at **corr 0.903, MAE 0.092**, with a compressed range
(predicts 0.423–0.578 where the truth runs 0.326–0.796); calibrating the compression against the
twelve months gives `actual = −0.780 + 2.638 · predicted`, R² 0.815.

Applied to the scored rows:

| | 2025 actual | 2025 predicted (LOMO) | **2026 scored, predicted** | **2026 scored, calibrated** |
|---|---:|---:|---:|---:|
| month 1 (91 scored rows) | 0.455 | 0.463 | **0.519** | **0.589** |
| month 7 (265 scored rows) | 0.326 | 0.446 | **0.406** | **0.290** |

**July's scored population looks like a low-fill month, as assumed. January's does not.** On the
same estimator, 2026-01 scores 0.056 above 2025-01 in raw prediction space and 0.135 above it after
calibration — putting it near November (0.627) and April (0.672), months whose OOF bias is negative.
Through the `bias ~ fill` line that would make January's true bias **−75 s**, against a shift of
−1,251.5 s: the row in §5 that costs +0.82 s.

This does not settle it — 25 of 356 prefixes are unseen, the estimator's month-level MAE is 0.092,
and there are no 2026 labels. It does move "the transfer is assumed" to "the transfer is measured to
be supported for July and contradicted for January".

---

## 7. What would change the answer

* **A decision rule for this lane.** `P1-unmatched-lane.md` §4.2 already specifies it: add
  `month_bootstrap_rmse_diff` restricted to unmatched rows plus an MSE-denominated bar, *in addition
  to* K=200. Under the current rule the oracle arm fails, so no work on this candidate can pay until
  the instrument changes. That is a prerequisite, not a result.
* **A second year of data**, which would give the calendar-month estimator a second instance and
  make the feature form both computable in-fold and gateable.
* **Nothing else.** More OOF months do not help: the four we have already disagree, and C3's own
  leave-one-out says the fill→bias map is not stable across them.

---

## 8. Facts and inference, kept apart

**Facts (measured here, every one with its section).** All fill rates and row counts in §2.1–§2.2
(`S1`); both OOF bias tables and the t statistics (`S2`); the fill-status decomposition and the
composition residuals (`S3`); the day/month-block bootstraps and the Kish n_eff (`S4`); the
in-sample month AUC 0.6453 (`S5`); every robustness curve, verdict and noise-arm figure in §3
(`S6`, `S8`, `S9`); the per-belief board arithmetic in §5 (`S11`); the prefix table, its
leave-one-month-out validation, its calibration and the scored-row estimates in §6 (`S10`); the
`temp_c` month-R² and the two fill AUCs in §2.5 (`S12`).
The code reads: `train.py:244` (`month` dropped), `train.py:256-283` (the reparameterisation),
`compare.py:118-140` (`verdict`), `cv.py:26-51` (the folds).

**Inference (not established here).** That the November composition residual of −1,291 s is a
second mechanism rather than 51 rows of noise. That the prefix table's calibrated estimate of the
2026 fill rate is accurate to better than its 2025 month-level MAE of 0.092. That `temp_c` is why
the model does not track the season, rather than 1,488 LIRF unmatched training rows being too few.
That the board effect of a shift is linear in the OOF bias in the way §5 assumes — the arithmetic is
exact, the belief about `b` is not.

**Out of scope by instruction.** No submission, no ranking prediction, no write to `outputs/`, no
change to any existing config, no refit.
