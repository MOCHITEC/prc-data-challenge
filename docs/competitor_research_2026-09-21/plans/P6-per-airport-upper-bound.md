# Theme P6 — a per-airport upper bound on predictions, from 2025 label support

Plan written 2026-09-21. Every claim is marked **[S]** (sourced: a file:line in this repo, a file:line in a
cloned competitor repo, or a URL) or **[X]** (my inference, arithmetic or judgement). No file in `src/` or
`configs/` was touched, no submission was created or sent. All new numbers below were produced by read-only
scripts in `/tmp/p6/` against `data/` and `outputs/`; each is labelled **[M]** with the script that made it.

Baseline: team `resilient-hedgehog`, **284.3584** on the 344,841-row Jan+Jul 2026 ranking set (v10,
`outputs/v10_probe_reparam.parquet`). **[S]** `experiments/submissions.jsonl` (v10 record,
`leaderboard_score: 284.3584`). At that baseline SSE = 2.7884e10 and **one second of RMSE = 1.958e8 of
squared error** **[M]** `/tmp/p6/deep.py`, reproducing `results/15-postprocess-overlays.md:19-28`.

**Headline, stated before the detail.** The rule as our own folds confirmed it (cap **non-LIRF unmatched**
predictions) is nearly inert on v10: that row set contains **zero** predictions above 4,800 s and 17 above
3,600 s, worth at most −0.19 s. The rule as the *label-free* evidence supports it points at a **different row
set** — 12 **matched** predictions at **LEMD and LEBL**, the two airports whose 2025 labels are hard-censored
at 3,600 s. That is the shippable version, it is worth about **−0.15 to −0.23 s** with a worst case of
**+0.03 s** under the "the model was right" counterfactual, and its whole risk is a single shared assumption,
not twelve independent ones. I recommend shipping it, priced at −0.20 s, and I say plainly in §6 why it is a
small prize.

---

## 1. What our own document established, on which rows, and why it was never shipped

### 1.1 What was measured

Source: `docs/investigation_2026-09-16/results/tail-postprocessing-cap-shrink.md`, read in full. Inputs are
`experiments/oof_main-48c19bc830.parquet` (672,428 scored rows, months 1, 7, 11, 12) joined to
`data/work/feat_train.parquet`; baseline total OOF RMSE **334.58**, unmatched (routing) **2,296.1**, matched
**216.4**, LIRF-unmatched **7,179.3**, non-LIRF-unmatched **1,450.2** **[S]** (`:46-58`).

Definitions used there, which this plan keeps **[S]** (`:9-12`): **unmatched (routing)** = `mvt_minus_aobt3`
is null (model-B rows; 22,470 train / 5,290 rank); "cap at c" = `pred := min(pred, c)` on the stated row set;
matched rows untouched unless stated.

| sub-claim | row set | verdict | key numbers | cite |
|---|---|---|---|---|
| (a) fixed cap on **all unmatched** | 8,378 OOF rows | **REFUTED** | every cap ≤ 43,200 raises total (334.58 → 452.59 / 433.64 / 404.56 / 335.42 at 3,600 / 7,200 / 14,400 / 43,200) | `:66-71`, `:259` |
| (a2) cap = quantile q of the unmatched prediction distribution | same | **REFUTED** | monotone in q; no q < 1.0 beats baseline | `:83-93`, `:260` |
| (b) shrink λ toward per-ADEP median | same | **REFUTED at every λ, under all seven target definitions** | λ=0.75 → 2,467–2,479; λ=0.5 → 2,741–2,854; λ=0 → 3,492–3,806 vs baseline 2,296 | `:100-126`, `:261` |
| (c) shrink toward (ADEP, hour) median | same | **REFUTED**; the hour dimension moves the answer by ≤ 35 s against a 180–850 s deterioration from the shrink | `:130-142`, `:262` |
| (d) cap **non-LIRF unmatched only** | 7,841 OOF rows | **CONFIRMED** | **3,600 → total 334.58 → 327.05 (−7.53), 53 rows, 49 helped / 4 hurt, ΔSSE −3.351e9, negative in all four OOF months and at all eight affected airports**; **7,200 → 329.33, 24 rows, 24 helped / 0 hurt, all 24 labels < 1,744 s** | `:152`, `:156`, `:160-164`, `:263` |
| (e) cap **matched** rows | 664,050 OOF rows | **REFUTED** | 3,600 → matched 216.4 → 243.7, total 352.64 | `:174-179`, `:264` |

The mechanism the document isolates **[S]** (`:166-168`, `:186-199`): the deterioration under any blanket cap
is *entirely* LIRF. On LIRF unmatched rows the model **under**-emits the tail (43 predictions > 4 h against 51
labels ≥ 4 h; 5 > 12 h against 15), so capping destroys real signal. On non-LIRF unmatched rows it
**over**-emits (9 predictions > 4 h against 2 labels), which is why the restricted cap works.

The document also states its own limits **[S]** (`:270-276`): the LB effect cannot be determined locally; the
non-LIRF gain "rests on 24–53 OOF rows in 2025 (4 months) whose labels were all < 4,747 s"; and it is silent
on whether the 2026 rows carry the training tail frequency.

### 1.2 Why it was never shipped — three recorded reasons, none of them a rejection of the rule

1. **The roadmap had already closed the area, one day before the measurement.** Commit `b3f4c55`
   (2026-09-15, "docs: redraw the roadmap, and cancel the phase the measurements ruled out"): "Phase 2 —
   calibrating the extreme rows — is cancelled … a paired comparison on those rows has a confidence interval
   62 s wide, so no change there can be judged at all … So half the error is conceded, and the contest is
   fought on the matched rows". **[S]** `git log -1 --format=%B b3f4c55`. The item-13 measurement landed the
   next day in `f078298` (2026-09-16) and its commit message records the result — "Cap/shrink: refuted on all
   unmatched or matched rows …; holds only as a cap on non-LIRF unmatched rows (-7.5 on this OOF, 53 rows)"
   **[S]** — but no follow-up commit re-opens the phase. **[X]** The rule was measured after the policy that
   would have carried it had been cancelled, and nothing re-litigated the cancellation.
2. **Post-processing was never put on a submission list.** `docs/investigation_2026-09-16/findings.md:560-579`
   lists everything that "needs a probe submission": items 29, 11(c) and 30. The non-LIRF cap is not among
   them. **[S]**
3. **Standing policy blocks uploads without an explicit request.** `docs/README.md:361`
   「提出は依頼があるまで行わない。」 and `docs/investigation_2026-09-16/plan.md:453` "Batch D … NOT
   scheduled — needs the user's explicit authorization". **[S]**

**[X]** There is no note anywhere in the history saying the cap was considered and declined on its merits. It
was overtaken: v10 (a model change, 284.3584) and v11 (a rollover overlay, 293.978) consumed the next two
slots **[S]** `experiments/submissions.jsonl`.

### 1.3 One correction this plan makes to §1.1

The 2025-fold verdict (d) was computed on the **v7-era** OOF file. Our production recipe is v10's. On
`experiments/oof_submit_v10_reparam-77f92c802d.parquet` (forward fold, 328,009 scored rows), the v10 recipe
emits **zero** non-LIRF unmatched predictions above 4,800 s and exactly **one** prediction anywhere above its
airport's 2025 maximum label **[M]** `/tmp/p6/oof2.py`. The fold price of any upper bound on the current
recipe is therefore ~0 — which is not evidence against the rule; it is precisely the mechanism `kind-mango`
named (§2.3).

---

## 2. `kind-mango`'s rule, exactly as implemented

Repo cloned to `/tmp/kind-mango` (`github.com/skylinkapi/prc-data-challenge-2026-kind-mango`). Line numbers
below are that working copy.

### 2.1 How the per-airport support is computed, and over which class

`src_v3/support.py:26-57` — `build_support(train_frame)`, called **once on all 2025 target-airport
departures** and frozen at fit time. Per airport it stores `class_max` for the classes `clean`, `tail` and
`24h`, `q_low_clean` (the 0.1 % quantile of clean labels), `n_over_3600` and `n_over_7200`. **[S]**

The class definition is `src_v3/labels.py:16-48`, config constants at `src_v3/config.py:38-43`: **[S]**

- `fallback` = `|y − sched_delay| ≤ 60` (per-airport tolerance if supplied) — **takes precedence over every
  band**, i.e. the support is computed with the schedule-copied rows removed;
- `low` = `0 < y < 30`; `clean` = `30 ≤ y ≤ 7200`; `tail` = `7200 < y ≤ 80000`; `24h` = `y > 80000`;
  `y ≤ 0` = `invalid`.

So "the airport's 2025 support" in their rule is **the maximum non-fallback label of the named class**, not
the raw maximum.

### 2.2 The bound itself, and how LIRF is excluded

`src_v3/postprocess.py:28-33, 36-78` **[S]**:

```
NO_LONG_TAIL  = ("EDDF","EDDM","LEBL","LEMD")   # cap at tail-class max (else clean max) + margin
HAS_LONG_TAIL = ("EGLL","EHAM","LFPG","LSZH","LTFM")
MS1_TAIL_MARGIN_S = 600 ; MS1_MVT_EOBT1_MIN = 5400
```

- `apply_ms1` loops airports and **`continue`s on `a == "LIRF"`** (`:58`) — LIRF is excluded by name, with the
  comment "LIRF rows are untouched (MH4 handles them)" (`:53`).
- At a `NO_LONG_TAIL` airport: `cap = (tail_max if tail_max > 0 else clean_max) + 600`, applied to **every**
  row, no escape hatch (`:36-41`, `:63-65`).
- At a `HAS_LONG_TAIL` airport: `cap = clean_max + 600`, but a row is **exempt** (`row_cap = inf`) when
  `flt_null == 0` **and** `mvt_eobt1 > 5400` (`:66-70`). That is the "operational reason to be long" hatch.
- The prose version, `docs/MODEL_ANALYSIS.md` §5.6, matches **[S]** and is quoted in
  `results/15-postprocess-overlays.md:63-71`.

Composition order, `postprocess.py:128-140`: **MS3 (member median on disagreement) → MS1 (cap from above) →
MS2 (floor from below)**. The floor runs last. **[S]**

### 2.3 What it bought, and the reason they shipped it

`RECAP.md:77` (v48): "**294.24 … new best, −2.33 s vs v47 (296.57)** … MS1 moved 5 non-LIRF rows (EDDM
21,465 → 7,499 the biggest −13,966 s; EHAM 11,103 → 7,311; two LEBL rows to 4,204; LEMD 4,376 → 4,198). MS2
moved 104 rows (small pushes) … **Zero training cost. MP7 gate passed. Live −2.33 s … MS1 fold price was 0 MSE
because the base does not extrapolate on 2025 rows. The label-free evidence at upload time (5 rows with
predictions 2–6× above the airport's 2025 support, all with `mvt_eobt1` below the eligibility bar) was
correct.**" **[S]**

`RECAP.md:74` (v51) is the companion caution: retraining the base to remove the cause of the extrapolation
bought **−5.99 s**, after which "MS1 finds nothing to clip". **[S]** **[X]** A cap is a patch; it is worth
less than removing what makes the model extrapolate.

### 2.4 Their pre-ship checklist (MP7), verbatim in structure

`docs/MODEL_ANALYSIS.md:832-838` **[S]**: "**Gate every file on label-free checks.** Per airport: maximum
prediction; counts above 3,600 s and 7,200 s against the 2025 support; a list of every row moved by more than
3,000 s with airport, `sd`, `mvt_eobt1` and record status; LIRF `p_fb` and `R_norm` quantiles by month; value
quantiles of every served column by airport and month against the same 2025 month. **Block the upload on an
unexplained row or a strict drift failure.**"

Implemented in `src_v3/gate.py` **[S]**: `BIG_MOVE_S = 3000` (`:23`); hard bounds `[0, 200000]` (`:27-28`,
`:85-88`); per-airport `n_over_3600` / `n_over_7200` against `allow = 2 × 2025 count + 5 / + 3` (`:29-30`,
`:99-106`); every row moved ≥ 3,000 s listed with airport, new, ref and diff (`:122-127`); `GateResult.passed
= not reasons` (`:135`). Applied in `src_v3/build_v48.py:114-134`, which also asserts one-to-one id join, row
count and no NaN before writing (`:76-105`).

The upload discipline around it, `docs/MODEL_ANALYSIS.md:1050-1051` **[S]**: "**One pre-registered price per
upload. Never build on an unscored upload.**"

**[X]** Two properties of their gate do not transfer as written: `allow = 2 × n_2025 + 5` compares a 12-month
count to a 2-month count, so it is far too lenient for us; and `UPPER_HARD_S = 200,000` is tuned to their LIRF
hedge. Our version must be rate-based (§5.4).

---

## 3. Measurement on our own current best submission (v10)

File: `outputs/v10_probe_reparam.parquet`, 344,841 rows, `MVT_ID_mvt: double`, `TAXITIME_SEC_mvt: int32`,
0 duplicate ids, min 0, max 112,771 **[M]** `/tmp/p6/rank.py`. Joined to `data/work/feat_rank.parquet` on
`MVT_ID_mvt` (0 unmatched joins). Group sizes: non-LIRF matched 313,035, non-LIRF unmatched 4,907, LIRF
matched 26,516, LIRF unmatched 383; unmatched (routing) 5,290, unmatched (fid) 5,283 — which reproduces
`tail-postprocessing-cap-shrink.md:203` exactly **[M]**.

### 3.1 The 2025 per-airport label support, rebuilt on our own data

**[M]** `/tmp/p6/support.py` over `data/work/feat_train.parquet` (2,085,047 rows, no null labels):

| ADEP | n 2025 | max label | max non-fallback | n y>3,600 | n y>7,200 | n unmatched | max unmatched |
|---|---|---|---|---|---|---|---|
| EDDF | 230,141 | 6,907 | 6,566 | 73 | **0** | 1,956 | 6,907 |
| EDDM | 167,334 | 6,899 | 6,899 | 59 | **0** | 1,010 | 6,899 |
| EGLL | 239,546 | 11,452 | 11,452 | 728 | 37 | 1,414 | 11,167 |
| EHAM | 247,951 | 11,032 | 11,032 | 82 | 3 | 4,084 | 11,032 |
| LEBL | 179,705 | **3,604** | 3,604 | **1** | **0** | 1,862 | 3,273 |
| LEMD | 212,242 | **3,598** | 3,598 | **0** | **0** | 947 | 3,465 |
| LFPG | 239,552 | 84,240 | 84,240 | 396 | 37 | 3,773 | 84,240 |
| LIRF | 160,704 | 131,167 | 88,392 | 1,967 | 480 | 1,488 | 131,167 |
| LSZH | 134,907 | 87,341 | 87,341 | 26 | 6 | 2,291 | 87,341 |
| LTFM | 272,965 | 10,309 | 10,309 | 794 | 21 | 3,645 | 10,309 |

**[X]** The four airports with **zero** 2025 labels above 7,200 s are exactly `kind-mango`'s `NO_LONG_TAIL`
set — EDDF, EDDM, LEBL, LEMD (`postprocess.py:28`). Their airport partition is reproduced on our data without
having seen it.

### 3.2 How many of our 344,841 predictions exceed their airport's 2025 support

**[M]** `/tmp/p6/exceed.py`:

| support definition | rows above it | LIRF | non-LIRF | unmatched | matched | max multiple |
|---|---|---|---|---|---|---|
| airport max label (any class) | **14** | 0 | 14 | **0** | **14** | 1.32× (LEMD) |
| airport max **non-fallback** label | 15 | 1 | 14 | 1 | 14 | 1.28× (LIRF) |
| airport max **unmatched** label | 14 | 0 | 14 | 0 | 14 | 1.37× (LEMD) |

Split of the 14 by airport: **LEMD 10** (3,878–4,745 vs support 3,598; 1.08–1.32×), **LEBL 2** (3,750 and
3,953 vs 3,604; 1.04–1.10×), **EHAM 2** (11,236 and 12,887 vs 11,032; 1.02–1.17×). Months: 8 in January, 4 in
July (the EHAM pair is January).

**The first load-bearing fact: none of the 14 is an unmatched row.** Every one has `nm_matched = 1` and a
non-null `mvt_minus_aobt3`. The row set our folds confirmed (non-LIRF **unmatched**) and the row set the
label-free evidence indicts (non-LIRF **matched**, at two airports) are disjoint on this file. **[M]**

**The second: the multiples are 1.02–1.37×, not `kind-mango`'s 2–6×.** Our model does not extrapolate on the
2026 rows the way their v47 base did. **[M]** vs **[S]** `RECAP.md:77`.

### 3.3 The confirmed row set is nearly empty on v10

Non-LIRF unmatched predictions above a threshold, v10 **[M]** `/tmp/p6/deep.py`:

| > 1,800 | > 2,400 | > 3,000 | > 3,600 | > 4,200 | > 4,800 | > 7,200 |
|---|---|---|---|---|---|---|
| 558 | 293 | 62 | **17** | 2 | **0** | **0** |

The 17 rows above 3,600 are EHAM 10, LTFM 4, LFPG 3, all January, all `unm_fid` true, predictions
3,684–4,689. Compare `tail-postprocessing-cap-shrink.md:251-252`, where v7 had 310 unmatched rows above 3,600
and 100 above 7,200, and v9 had 329 / 83 **[S]**. **[X]** The v10 LIRF reparameterisation removed the
non-LIRF unmatched tail that made the item-13 cap worth −7.53 s on the v7-era OOF. The lever the folds
confirmed has almost nothing left to grip.

Value of applying the confirmed rule anyway, at the baseline's arithmetic **[M]** `/tmp/p6/bounds.py`:

| rule | rows | ΔRMSE if y = airport 2025 unmatched median | ΔRMSE if y = p (model was right) |
|---|---|---|---|
| cap non-LIRF unmatched at 3,000 | 62 | −0.596 | +0.091 |
| cap non-LIRF unmatched at 3,600 | 17 | −0.193 | +0.017 |
| cap non-LIRF unmatched at 4,200 | 2 | −0.018 | +0.001 |
| cap non-LIRF unmatched at 7,200 | 0 | — | — |

**[X]** The 3,000 variant looks the most valuable and is the one I reject hardest: it touches 62 rows whose
median move is 155 s, i.e. it is exactly the "shifts 30 rows by 200 s" object that
`results/15-postprocess-overlays.md:31-32` prices at ~0.0000 s, and 37 of the 62 are EHAM January — see §3.5.

### 3.4 The rate check: where the file actually over-emits

Predictions above a threshold vs the count implied by the 2025 rate at the same airport **[M]**
`/tmp/p6/rates.py`:

| ADEP | n rank | p>3,600 | implied (all-2025) | implied (Jan+Jul 2025) | p>7,200 | implied (Jan+Jul) |
|---|---|---|---|---|---|---|
| EDDF | 36,317 | 27 | 11.5 | 23.7 | 0 | 0.0 |
| EDDM | 25,966 | 5 | 9.2 | 12.5 | 0 | 0.0 |
| EGLL | 39,840 | 72 | 121.1 | 237.8 | 4 | 19.8 |
| **EHAM** | 38,182 | **289** | 12.6 | 50.4 | **7** | 1.9 |
| **LEBL** | 30,080 | **2** | 0.2 | 0.0 | 0 | 0.0 |
| **LEMD** | 36,954 | **10** | 0.0 | 0.0 | 0 | 0.0 |
| LFPG | 39,872 | 101 | 65.9 | 138.9 | 2 | 9.1 |
| LIRF | 26,899 | 352 | 329.2 | 466.4 | 109 | 114.6 |
| LSZH | 23,150 | 16 | 4.5 | 12.4 | 0 | 4.1 |
| LTFM | 47,581 | 81 | 138.4 | 40.9 | 0 | 1.0 |

By month, the excess is concentrated: EHAM Jan 289 against 43.1 implied by EHAM January 2025 (July: 0 against
4.9); LTFM Jan 80 against 11.5 (July 1 against 29.0). EGLL, LFPG-July, LTFM-July and LIRF all **under**-emit.
**[M]**

### 3.5 The 2026 arrival mirror decides EHAM against the cap, and LEMD/LEBL for it

The scoring file carries **344,693 labelled 2026 arrival rows** (`data/raw/ranking.parquet`, `PHASE_mvt ==
"ARR"`, `TAXITIME_SEC_mvt` non-null on all of them; the 344,841 departure labels are null) **[M]**
`/tmp/p6/censor.py`. `kind-mango`'s MP6 is the published precedent for using it as a ship/no-ship gate
**[S]** `results/15-postprocess-overlays.md:361-367`. This repo had already measured its headline
(`docs/data_audit_2026-09-18/results/arrival-rows.md:182-183`: "EHAM Jul −69.7, EDDF Jul −66.5, **EHAM Jan
+67.0**, LIRF Jan −43.2, LSZH Jan +35.2") **[S]** but never pointed it at a post-processing decision.

Arrival tail rate per 10,000 rows, same airport, same month, 2025 vs 2026 **[M]** `/tmp/p6/mirror.py`:

| ADES | month | 2025 rate >3,600 | 2026 rate >3,600 | 2025 mean | 2026 mean |
|---|---|---|---|---|---|
| **EHAM** | 1 | 1.1 | **44.3 (40×)** | 593.3 | **660.3** |
| EHAM | 7 | 1.3 | 0.0 | 477.3 | 407.6 |
| LFPG | 1 | 1.7 | 11.1 (6.5×) | 593.8 | 612.1 |
| **LEMD** | 1 / 7 | 0.0 / 0.0 | **0.0 / 0.0** | 507 / 542 | 496 / 523 |
| **LEBL** | 1 / 7 | 0.0 / 0.0 | **0.0 / 0.0** | 222 / 244 | 232 / 241 |

And on the four specific 2026 days our rule would touch **[M]** `/tmp/p6/cluster.py`:

| airport | day | arrivals | mean taxi-in | airport 2026 mean | max | n > 1,800 |
|---|---|---|---|---|---|---|
| LEMD | 2026-01-05 | 579 | 472 | 511 | 1,299 | **0** |
| LEMD | 2026-01-28 | 509 | 512 | 511 | 1,363 | **0** |
| LEMD | 2026-07-07 | 638 | 551 | 511 | 1,630 | **0** |
| LEMD | 2026-07-15 | 649 | 502 | 511 | 1,564 | **0** |
| LEBL | 2026-07-01 | 563 | 236 | 238 | 1,140 | **0** |
| LEBL | 2026-07-20 | 563 | 277 | 238 | 1,504 | **0** |
| **EHAM** | **2026-01-05** | **201** | **2,050** | 515 | **15,526** | **84 of 201** |
| EHAM | 2026-01-07 | 199 | 684 | 515 | 2,546 | 4 |

**[X]** This is decisive in both directions. The EHAM prediction of 11,236 s on 2026-01-05 sits on a day when
the same airport's *labelled* arrivals averaged 2,050 s of taxi-in against a 515 s norm, with 42 % of them
above 30 minutes and one at 15,526 s, and with the arrival count itself collapsed to 201 from a ~525/day norm.
That prediction is **corroborated**, not anomalous, and capping it would be capping a real event. Conversely
none of the six LEMD/LEBL days shows any airport-wide congestion at all: the same-day arrival mean is within
±40 s of the airport norm and **not one arrival** exceeds 1,800 s.

This also explains our own excess counts without a model defect: the EHAM January unmatched rate is
**4.959 %** against a 2025 range of 1.127–2.502 % (`docs/data_audit_2026-09-18/results/recording-change.md`
F6 table) **[S]**, and the arrival side is 40× its 2025 tail rate. EHAM January 2026 is a real disruption
month, and our 289 predictions above 3,600 there are *less* of an excess (6.7× the 2025 rate) than the
arrival side shows (40×).

### 3.6 Is the LEMD/LEBL ceiling real, and does it survive into 2026?

Independent confirmation of `fenil264`'s claim (`results/24-repo-tail-sweep.md:436-452`) **[S]**, on our own
data **[M]** `/tmp/p6/censor.py`, `/tmp/p6/censor2.py`, `/tmp/p6/ceiling.py`:

1. **Top-10 2025 DEP labels.** LEMD `[3598, 3567, 3566, 3562, 3549, 3546, 3537, 3522, 3510, 3504]`; LEBL
   `[3604, 3599, 3595, 3594, 3584, 3572, 3567, 3552, 3549, 3542]`; controls LFPG `[84240, 58206, 21308,
   15059, …]`, LIRF `[131167, 88392, 88132, …]`. Byte-for-byte the numbers in their notebook quote.
2. **The histogram signature.** Count of 2025 DEP labels in `[3500,3600)` then `[3600,3800)`: LEMD **10 → 0**,
   LEBL **16 → 1**. Every other airport rises or holds across the boundary: EDDF 16→24, EDDM 12→17, EGLL
   83→141, EHAM 15→31, LFPG 47→103, LIRF 99→193, LSZH 2→7, LTFM 76→152. Truncation, not traffic.
3. **Twelve independent months.** Monthly maximum 2025 DEP label: LEMD `[3461, 3450, 3522, 3341, 3546, 3339,
   3537, 3478, 3567, 3598, 3258, 3465]`, LEBL `[3200, 2993, 3441, 3520, 3549, 3572, 3599, 3584, 3539, 3604,
   3459, 3540]` — 24 monthly maxima, none above 3,604, on ~15,000 rows each. Controls EDDF `[4505 … 6907]` and
   EDDM `[2698 … 6899]` wander freely.
4. **The 2026 arrival-side quantile test.** Observed 2026 maximum vs the 2025 quantile at the same sample
   size (`q = 1 − 1/n`), ratio: LEMD **1.04**, LEBL **0.71** — no tail inflation. The method detects
   inflation where it exists: LFPG **7.47**, EHAM **3.55**, LIRF **1.31**; and deflation: LSZH 0.22, LTFM
   0.65, EDDM 0.77.

**[X]** (2) and (3) establish that the 3,600 s ceiling is a stable property of the LEMD/LEBL 2025 departure
feed, not a sampling accident. (4) establishes that nothing about LEMD/LEBL taxi times changed in 2026 on the
one side we can observe. What none of them establishes is that the *departure* ceiling itself was not lifted
for 2026 — the arrival series never reaches 3,600 s at those airports in either year, so it cannot witness the
ceiling directly. `fenil264` say the same: "arrivals at LEMD/LEBL are uncapped in both 2025 and 2026 only
because taxi-in never reaches 3,600 s, so the ranking DEP cap is not directly provable" **[S]**
(`24-repo-tail-sweep.md:449-452`). That residual is the whole of the bet, and §6 prices it.

### 3.7 Worst-case bound in seconds of RMSE

**[M]** `/tmp/p6/bounds.py`, `/tmp/p6/final.py`. Positive = RMSE gets worse.

| row set | rows | y = 1,000 s | y = 3,000 s | y = cap | **y = p (every row capped wrongly, model exactly right)** | y = 7,200 | y = 86,400 on every row |
|---|---|---|---|---|---|---|---|
| all 14 above airport max | 14 | −0.466 | — | — | **+0.048** | — | +7.95 |
| **LEMD+LEBL only (12 rows)** | 12 | **−0.228** | −0.076 | −0.030 | **+0.030** | +0.244 | +6.21 |
| `kind-mango` MS1 transcribed | 7 | −0.271 | — | — | +0.036 | — | +3.56 |
| non-LIRF unmatched cap 3,600 | 17 | −0.185 | — | — | +0.017 | — | +5.19 |
| non-LIRF **all** cap 7,200 (contrast) | 13 | −2.688 | — | — | +0.600 | — | +27.10 |

Break-even for the 12-row rule, if all twelve shared a common true label: **y = 3,995 s**. The rule costs
0.1 s at a common y of 5,309 s, 0.5 s at 10,568 s and 1.0 s at 17,152 s **[M]** `/tmp/p6/final.py`. For
context, the 2025 label distribution at those airports: LEMD median 985, p99 1,932, p99.9 2,704, p99.99 3,437,
max 3,598 (n = 212,242); LEBL median 906, p99 1,962, p99.9 2,963, p99.99 3,495, max 3,604 (n = 179,705).

---

## 4. The decision, from the numbers

### 4.1 Row set

**Ship: cap at LEMD and LEBL (and, for rule generality, EDDF and EDDM — the other two airports with zero 2025
labels above 7,200 s, where the rule binds on zero rows today).** Do **not** cap EHAM, EGLL, LFPG, LSZH, LTFM
or LIRF.

Why this set and not the fold-confirmed one:

- The fold-confirmed set (non-LIRF unmatched) has **nothing left in it** on v10: 0 rows above 4,800, 17 above
  3,600 worth −0.19 s at best, and 10 of those 17 are EHAM January, the cell the arrival mirror defends
  (§3.3, §3.5). **[M]**
- The 14 rows that are above their airport's support are all matched, and capping matched rows was refuted on
  our OOF (216.4 → 243.7 at 3,600) **[S]** `tail-postprocessing-cap-shrink.md:174-179`. **[X]** That refutation
  was of a *blanket* 3,600 cap on 534 matched rows, dominated by LIRF and LFPG rows with genuinely large
  labels; it says nothing about 12 rows at two airports whose entire 2025 label range lies below the cap. The
  same document's own logic — cap where the model over-emits relative to the labels, not where it under-emits
  — selects LEMD/LEBL and rejects EHAM.
- The label-free evidence is strongest exactly where the fold evidence is silent, which is the pattern
  `kind-mango` shipped on (`0 MSE` fold price, −2.33 s live) **[S]** and the pattern our own §1.3 measurement
  reproduces (1 row, −0.048 s, on the v10-recipe OOF).
- The 2025 folds themselves do sign-check this rule, where they can see it. On
  `oof_main-48c19bc830.parquet`, capping every prediction at its own airport's 2025 maximum label touches
  **22 rows, 22 helped / 0 hurt**, total OOF RMSE **334.582 → 331.111 (−3.47)**, ΔSSE **−1.554e9**, negative
  in **all four** OOF months (Jan −9.3e7, Jul −4.1e8, Nov −4.5e8, Dec −6.0e8) and at **all five** affected
  airports (EDDF 6, EDDM 2, EHAM 4, LEBL 8, LEMD 2). The largest capped prediction was 16,880 s; the largest
  *label* among all 22 was **3,169 s**. Restricted to LEMD+LEBL: 10 rows, −1.31 s, every label below 3,169.
  **[M]** `/tmp/p6/oof2.py`. **[X]** A 22/0 hit rate is a stronger object than the 49/4 of the item-13 cap,
  and it is the rule I am proposing rather than a proxy for it.

### 4.2 Threshold

**The cap value is the airport's 2025 maximum departure label, with no margin**: LEMD **3,598**, LEBL
**3,604**, EDDF **6,907**, EDDM **6,899**. **[M]** §3.1.

- Against `kind-mango`'s `+600` margin (`postprocess.py:31`) **[S]**: a margin exists to protect a *sampled*
  tail from being mistaken for a bound. Here the bound is not sampled — it is a truncation visible in the
  histogram and stable across 12 monthly maxima (§3.6). A margin of 600 would move only 6 of the 12 rows and
  cap the largest at 4,198 instead of 3,598, cutting the expected gain roughly in half while leaving the same
  directional bet intact. **[M]** `/tmp/p6/exceed.py` (MS1 transcription: 7 rows, 6 at LEMD moving ≤ 547 s,
  1 at LFPG moving 2,496 s).
- Against a round 3,600: it differs from the per-airport maximum by 2 s at LEMD and 4 s at LEBL and changes no
  row's fate. Use the measured maximum, because it is the quantity the rule is about.
- Against the `clean`-class max (`kind-mango`'s definition, §2.1): at LEMD/LEBL every class max is the same
  number (no tail-class rows exist), so the choice is moot there. At LFPG the clean max is 7,085 while the
  overall max is 84,240 — and using the clean max is what makes the transcribed MS1 want to clip a
  legitimate LFPG row (10,181 → 7,685) at an airport that **under**-emits above 7,200 (2 predictions against
  9.1 implied). **[M]** §3.4. Use the **airport's maximum label, any class**; it is the definition under which
  our 22/22 OOF sign check was computed.

**Do not adopt `kind-mango`'s `mvt_eobt1 > 5400` escape hatch.** Under our airport set it is inert at
LEMD/LEBL (the ceiling applies to every row regardless of how late the flight is), and the one place it would
matter — EHAM — is excluded by airport anyway. Keeping it would add a parameter that does nothing. **[X]**
Note that under a faithful transcription it is the hatch that spares the two EHAM rows, so a team adopting
their rule wholesale arrives at almost the same place by a different route. **[M]**

### 4.3 Hard cap, not a shrink

Hard cap: `pred := min(pred, S_a)`.

- Our own document refutes shrink-toward-median at every λ and under all seven target definitions tried
  **[S]** `tail-postprocessing-cap-shrink.md:261-262`. **[X]** I am not departing from that. The proposal is
  not a shrink, and the refutation does not reach it: a shrink moves *every* row in the class toward a fitted
  centre (the λ = 0.75 variant moves all 8,378 unmatched rows), whereas a cap moves only the rows outside a
  measured support and moves them *to the boundary of that support*, never past it.
- The downside of a cap is bounded by construction: `|min(p, S) − y| ≤ |p − y|` for every `y ≤ S`. A blend's
  is not. `results/15-postprocess-overlays.md:411-415` records the in-competition version of this — "Hard
  swaps have unbounded downside" after a classifier-selected swap took `kind-mango` from 372 to 625 **[S]**.
- A blend toward the airport median would additionally move rows that are *inside* the support, which is the
  refuted object. There is no version of a shrink here that is not either the refuted rule or the cap in
  softer clothing, and `results/15-postprocess-overlays.md:559-562` reaches the same conclusion **[S]**.

**One-line statement of the rule.** For `a ∈ {EDDF, EDDM, LEBL, LEMD}`, `pred := min(pred, S_a)` where `S_a`
is the maximum `TAXITIME_SEC_mvt` over all 2025 departure rows at `a`. All other airports, including LIRF, and
all other rows: unchanged.

### 4.4 What it does to the file

12 rows change; 344,829 are byte-identical. Max prediction unchanged at 112,771 (LIRF); median unchanged at
959; mean 1,039.3554 → 1,039.3337; total absolute movement 7,466 s; moves 146 / 552 / 1,147 s
(min / median / max); months January 8, July 4. **[M]** `/tmp/p6/rows.py`.

The exact rows (`MVT_ID_mvt` at full precision):

| MVT_ID_mvt | ADEP | month | pred → cap | move | m_aobt | m_eobt1 |
|---|---|---|---|---|---|---|
| 213802815 | LEBL | 7 | 3,953 → 3,604 | 349 | 468 | 5,328 |
| 213791326 | LEBL | 7 | 3,750 → 3,604 | 146 | 706 | 5,926 |
| 203363474 | LEMD | 1 | 4,745 → 3,598 | 1,147 | 1,781 | 6,881 |
| 203363319 | LEMD | 1 | 4,720 → 3,598 | 1,122 | 1,816 | 6,616 |
| 203363337 | LEMD | 1 | 4,599 → 3,598 | 1,001 | 1,744 | 6,304 |
| 203349361 | LEMD | 1 | 4,418 → 3,598 | 820 | 499 | 6,319 |
| 213727911 | LEMD | 7 | 4,290 → 3,598 | 692 | 323 | 6,023 |
| 203363460 | LEMD | 1 | 4,289 → 3,598 | 691 | 1,713 | 6,093 |
| 203363459 | LEMD | 1 | 4,011 → 3,598 | 413 | 1,715 | 5,915 |
| 203363465 | LEMD | 1 | 4,001 → 3,598 | 403 | 1,834 | 5,914 |
| 213732168 | LEMD | 7 | 4,000 → 3,598 | 402 | 964 | 4,384 |
| 203349358 | LEMD | 1 | 3,878 → 3,598 | 280 | 149 | 6,209 |

**[X]** Seven of the ten LEMD rows fall on two clock windows — 2026-01-05 08:08–08:15 and 2026-01-28
10:50–11:55 — with near-consecutive `MVT_ID`s and mostly runway 36R. They are not twelve independent draws;
they are four episodes. The same-day arrival mirror is flat on all four (§3.5), which is the reason to cap
them, but the correlation means the rule's outcome is close to a single coin, not twelve.

**Pre-registered price: −0.20 s (284.36 → 284.16), with a stated range of −0.23 to +0.03 and a tail risk of
+0.24 s.** **[X]** Derived from §3.7 with the central case "the twelve labels lie in the airport's ordinary
range (median ≈ 950 s)". Applying the published offline→live transfer discount is *not* appropriate here:
this is not a fold-fitted estimate, it is arithmetic over 12 known predictions under an assumption about 12
unknown labels.

---

## 5. The plan

### 5.1 Where the cap belongs in the code

There is exactly one prediction post-processing step in the repository today: `src/prc/train.py:433`,
`.with_columns(pl.col("pred").fill_null(fallback).fill_nan(fallback).clip(lower_bound=0.0))` inside
`_assemble` **[S]** (recorded as such in `docs/investigation_2026-09-16/findings.md`, structural fact 20).
`_assemble` is reached by both `run_cv` and `predict_ranking` via `predict_fold`
(`src/prc/train.py:360`), so anything placed there is priced by the folds automatically.

**Ship in two steps, in this order.**

**Step 1 — a standalone build script, for the first upload only.** New file
`docs/competitor_research_2026-09-21/scripts/build-p6-cap.py` (a *new* file; it modifies nothing under `src/`
or `configs/`). It reads `outputs/v10_probe_reparam.parquet`, recomputes `S_a` from
`data/work/feat_train.parquet`, applies the cap, and writes `outputs/v12_airport_cap.parquet` through
`prc.predict.write_submission` (or by mirroring its template/dtype handling: template
`data/raw/submitting.parquet`, `MVT_ID_mvt: double`, `TAXITIME_SEC_mvt: int32`, template row order). This
guarantees the other 344,829 rows are byte-identical to the scored v10 file, so the leaderboard delta is
attributable to twelve numbers and nothing else. It is the same shape as `kind-mango`'s
`src_v3/build_v48.py`, which reads the shipped v47 parquet rather than re-predicting **[S]**.

**Step 2 — after the score comes back, fold it into the pipeline.** Add `src/prc/postprocess.py` holding
`build_support(train_f) -> {adep: max_label}` and `apply_upper_bound(pred, adep, support, airports)`, and call
it from `_assemble` immediately after the existing `clip(lower_bound=0.0)`, so the bound becomes two-sided.
Two constraints:

- the support must be built from **training labels only**, and inside CV from **the fold's training months
  only**, never from the validated month — otherwise the fold price is contaminated. (At LEMD/LEBL this
  changes nothing: 12/12 monthly maxima are below the bound, §3.6. It matters for the general form.)
- the airport list belongs in the config (`configs/*.yaml`), defaulting to `[]`, so a run can be reproduced
  with the bound off.

Do **not** put the cap in `write_submission` or in `validate_submission`. The gate's job is to refuse a file,
not to change one; `MAX_PLAUSIBLE_SECONDS = 3*3600` there is explicitly commented "sanity bound, not a
clipping rule" (`src/prc/validate_submission.py:19`) **[S]** and it currently fires only as a warning
(`:87-88`), which is correct and should stay that way.

### 5.2 Composition with theme P5's lower bound

P5 (`results/24-repo-tail-sweep.md:754-757`, sqcm9896's largest live gain, −20.3 s on 11,826 rows) **[S]**:
for rows where `AOBT_3_flt` joined and `m_aobt = MVT_TIME − AOBT_3 > 1800`, set `pred = max(pred, m_aobt −
3600)`.

**They cannot fight, and here is the proof rather than the assurance.**

1. **On the confirmed-row-set reading, they are disjoint by construction.** P5's gate requires
   `m_aobt > 1800`, which requires `AOBT_3_flt` to be non-null, which is exactly `nm_matched = 1`. P6's
   fold-confirmed row set is the unmatched class, `mvt_minus_aobt3` null. The repo records that these two
   nullity conditions are identical on every row of both frames, with 0 disagreements
   (`docs/investigation_2026-09-16/findings.md`, structural fact 18) **[S]**. Measured: rows with
   `m_aobt > 1800` **and** non-LIRF **and** unmatched = **0** **[M]** `/tmp/p6/deep.py`.
2. **On the shipped reading (matched rows at 4 airports), they are disjoint by value.** The maximum `m_aobt`
   at each capped airport is LEMD 3,960, LEBL 1,939, EDDF 2,891, EDDM 7,615, so the maximum P5 floor is 360 /
   −1,661 / −709 / 4,015 against caps of 3,598 / 3,604 / 6,907 / 6,899. **The P5 floor cannot reach the P6 cap
   at any of the four airports**; headroom is LEMD 3,238, LEBL 5,265, EDDF 7,616, **EDDM 2,884 (the
   tightest)**. **[M]** `/tmp/p6/rows.py`.
3. **Whole-file check.** Applying P5 first raises 177 rows (max raise 2,042 s) and leaves the count of
   predictions above their airport's 2025 maximum at 14 — unchanged — with **0 rows newly pushed above a
   support they were below**. **[M]** `/tmp/p6/bounds.py`.

**What happens to a row where both would bind, stated explicitly.** Such a row does not exist on this file,
but the rule must still be defined for the next one. **Order: P5 floor first, then P6 cap** — i.e.
`pred := min(max(pred, m_aobt − 3600), S_a)`, the opposite order to `kind-mango`'s MS1-then-MS2
(`postprocess.py:131-139`) **[S]**, and deliberately so. Rationale **[X]**: P5's floor is a *physical
identity* — the aircraft was observed off-block `m_aobt` seconds before the movement, so the taxi time cannot
be much less than that — whereas P6's cap is a *statistical support*. If the two ever conflict, the physical
observation is the better evidence and a conflict means the support is stale at that airport. So: apply the
floor, then the cap, and **emit a hard gate failure listing any row where `m_aobt − 3600 > S_a`** rather than
silently letting the cap overwrite an observation. On this file that check fires on zero rows; if it ever
fires, it is a signal to re-derive `S_a`, not to ship.

### 5.3 Shipping it as one pre-priced submission

1. Write the price **into `experiments/submissions.jsonl` before the upload**, not after. The log's
   write-ahead design already supports this (`src/prc/submit.py:89-98`) **[S]**. The `note` field carries: the
   rule in one line, the four caps, the 12 `MVT_ID`s, the expected delta −0.20 s and the stated range
   [−0.23, +0.24].
2. Upload as **v12** through `prc.submit.submit(...)`, which enforces the gate, the daily counter and version
   uniqueness (`src/prc/submit.py:54-99`) **[S]**. Change nothing else in the file. One change per upload —
   `kind-mango`'s rule, `docs/MODEL_ANALYSIS.md:1050` **[S]**.
3. **The slot is cheap and the downside is capped twice over.** The leaderboard ranks on a team's best score,
   so a worse upload costs only the slot — recorded in this repo after v11: "v10 remains our best at 284.3584
   — the leaderboard ranks on the best score, so nothing is lost but the slot" **[S]** commit `7539b14`. The
   limit is 5 uploads/day against a 2026-10-11 deadline (`docs/handoff.md` §1) **[S]**, i.e. ~100 remaining
   slots. **[X]** Spending one on a bounded-downside, mechanistically-justified 12-row change is right; it
   should not be bundled with a model change, because at this magnitude a bundle makes it unmeasurable.
4. **Do not build on it until it is scored.** If a P5 upload is also queued, ship P5 first (it is worth 100×
   more) and P6 second, on top of the scored P5 file.
5. Record the realised delta next to the pre-registered one, and add the pair to the transfer-ratio table in
   `results/15-postprocess-overlays.md:270-276`.

### 5.4 Pre-ship checklist

Structural (adapted from `src_v3/gate.py` and `build_v48.py` **[S]**, plus our own gate):

- [ ] `prc.validate_submission` passes: schema, dtypes, 344,841 rows, id set **and order** equal to
      `data/raw/submitting.parquet`, no null/NaN/inf, no negative, not all-constant, median within 30 % of the
      training median (`src/prc/validate_submission.py:39-108`).
- [ ] Row-level diff against the scored v10 file: **exactly 12 rows differ**, and their `MVT_ID`s are the
      twelve in §4.4. Any thirteenth row is a blocker.
- [ ] Maximum absolute movement **1,147 s**; no row moved by more than 3,000 s (the MP7 `BIG_MOVE_S` rule,
      `gate.py:23` **[S]**). Any row above 3,000 s must be named and explained before upload.
- [ ] Every changed prediction is **lower** than before; no prediction was raised.
- [ ] Max, median and per-airport mean of the file compared with v10: max 112,771 unchanged, median 959
      unchanged, mean shift −0.022 s.

Label-free, per airport (this is the part that decides the ship):

- [ ] For every airport: count of predictions above 3,600 and 7,200, against the count implied by **the same
      airport in the same 2025 month**, not the 12-month rate (§3.4). Flag any cell above 3× the implied
      count, and require a named mechanism for each flag. Today: EHAM Jan 289/43.1 and LTFM Jan 80/11.5 are
      flagged, and both are explained — see below.
- [ ] For every flagged cell, run the **2026 arrival mirror** on the same airport and month: tail rate above
      1,800 and 3,600, mean taxi-in, 2025 vs 2026 (§3.5, `/tmp/p6/mirror.py`). **A cell where the arrival
      mirror confirms 2026 congestion is not capped.** Today this vetoes EHAM.
- [ ] For every row the rule moves, run the **same-day arrival mirror** at that airport: arrival count, mean
      taxi-in, max, count above 1,800 (§3.5, `/tmp/p6/cluster.py`). A row on a day with a real arrival-side
      event is not capped. Today all six LEMD/LEBL days are flat.
- [ ] Re-derive `S_a` from `data/work/feat_train.parquet` in the build script itself and assert the four
      values are 6,907 / 6,899 / 3,604 / 3,598. A support table read from a stale file is the failure mode
      this catches. (Cross-check: the same four numbers are the `max_all` column of §3.1.)
- [ ] Re-assert the ceiling evidence at LEBL/LEMD before each ship: 12/12 monthly maxima below the cap, and
      the `[3500,3600)` → `[3600,3800)` histogram step (§3.6). If either breaks on a refreshed data drop,
      the rule is void.
- [ ] Assert `m_aobt − 3600 ≤ S_a` on every row at the four capped airports (§5.2). Currently the tightest
      margin is 3,238 s at EDDM.
- [ ] Worst-case arithmetic restated in the log: `Σ(p−c)²` = 5.92e6 = **+0.030 s** if the model was right on
      every capped row; +0.244 s if all twelve are two-hour taxis; +6.21 s if all twelve are day rollovers.
- [ ] Influence check (Phoenix-Ops's procedure, `results/15-postprocess-overlays.md:405-409` **[S]**): with
      the top-1 contributing row removed the expected gain falls from **−0.228 to −0.191 s**, and with the
      top-3 removed to **−0.123 s**. No single row carries the rule — the largest contributes **16.3 %** of
      the expected ΔSSE (next two 15.9 % and 13.9 %), against Phoenix-Ops's 45.8 %. **[M]** `/tmp/p6/final.py`
      with `y = 1,000 s`.

---

## 6. What would make this cost us

Stated without hedging.

1. **The prize is small, and by the file's own arithmetic it is below the bar that file set.**
   `results/15-postprocess-overlays.md:428-434` concludes "an overlay whose per-row movement is under ~3,000 s
   cannot pay for itself at any row count under ~200" **[S]**. Our per-row movement is 146–1,147 s on 12 rows.
   **[X]** By that criterion this rule does not qualify, and I am recommending it anyway for one reason: the
   criterion was written to price *risk-bearing* overlays, and this one's worst case is +0.03 s under the
   model-was-right counterfactual. It is a cheap, bounded, measurable −0.2 s, not a bet worth defending as a
   lever. If the choice is between this and any change that could move seconds, take the other one.
2. **The single assumption that carries it.** If the LEMD/LEBL upstream truncation was lifted for 2026 and
   real multi-hour taxi-outs are now recorded there, all twelve rows fail together. The 12 monthly maxima and
   the histogram step (§3.6) establish the ceiling in 2025; the arrival mirror (§3.6 item 4) establishes that
   nothing else about those airports changed; **nothing available establishes the departure ceiling itself in
   2026**, because arrival taxi-in never approaches 3,600 s at LEMD/LEBL in either year. `fenil264` say so and
   so do I. Cost if it is wrong and all twelve true labels average 7,200 s: **+0.24 s**. Cost if they average
   17,152 s: **+1.0 s**.
3. **The rows are four episodes, not twelve draws.** Seven of the ten LEMD rows are two clock windows on two
   days (§4.4). The effective sample size of the bet is closer to 4 than to 12, and the "22 helped / 0 hurt"
   OOF sign check (§4.1) is likewise concentrated — 8 of its 22 rows are LEBL. **[X]** Do not read either
   count as 12 or 22 independent confirmations.
4. **We would be capping rows our own OOF says not to cap.** Verdict (e) of
   `tail-postprocessing-cap-shrink.md` is "capping matched rows is REFUTED" **[S]** `:264`, and all 12 rows
   are matched. My argument in §4.1 is that the refutation was of a different object; if that argument is
   wrong, this plan is contradicting a measured result on our own data with an unmeasured distinction.
5. **Global upper clipping has a measured catastrophic precedent in this competition.** sqcm9896 measured
   `clip_cap2h_v1` at **+90.7 s** **[S]** `results/24-repo-tail-sweep.md:293-294, 758`. That is a global 2 h
   cap, not a per-airport support bound, and our own table 3.2 reproduces the same shape (cap all rows at
   7,200 → 444.6 from 334.58) **[S]** `tail-postprocessing-cap-shrink.md:72`. **[X]** The distance between the
   two rules is the airport partition and the support-derived value. If that distance turns out not to matter,
   we lose seconds, not tenths — which is precisely why §4.1 excludes every airport with a real 2025 tail and
   why the checklist vetoes an airport the 2026 arrival mirror defends.
6. **It patches a defect instead of removing it.** `kind-mango`'s v51 got **−5.99 s** by retraining the base
   so that the clip had nothing to clip, against **−2.33 s** for the clip **[S]** `RECAP.md:74, :77`. Our
   equivalent question — why does the model emit 3,878–4,745 s at an airport where no 2025 row exceeds 3,598 —
   is unanswered by this plan. **[X]** The likely mechanism is the shared model A seeing 2,062,577 rows across
   ten airports, seven of which do have long tails, with `ADEP` as one categorical among many. A per-airport
   monotone constraint, an airport-conditioned leaf, or training-target clipping at those two airports would
   remove the cause. Note that sqcm9896 measured train-target clipping as "vacuous" **[S]**
   (`24-repo-tail-sweep.md:799-800`), so that specific variant is already priced at zero.
7. **Opportunity cost is the real risk, not the 0.03 s.** This plan costs an upload slot and a few hours. The
   same effort spent on P5 (−20.3 s live, 11,826 rows, sourced) or on the deep-CatBoost residual blend
   (−7.11 s, sourced) is worth two orders of magnitude more **[S]** `SUMMARY.md:91-99`. **[X]** Ship P6 only
   after those are queued, and never as a substitute for them.

---

## Sources

In-repo:
`docs/investigation_2026-09-16/results/tail-postprocessing-cap-shrink.md`;
`docs/investigation_2026-09-16/findings.md`;
`docs/competitor_research_2026-09-21/results/15-postprocess-overlays.md`;
`docs/competitor_research_2026-09-21/results/24-repo-tail-sweep.md`;
`docs/competitor_research_2026-09-21/SUMMARY.md`;
`docs/data_audit_2026-09-18/results/arrival-rows.md`;
`docs/data_audit_2026-09-18/results/recording-change.md`;
`docs/README.md`; `docs/handoff.md`;
`src/prc/train.py`, `src/prc/predict.py`, `src/prc/submit.py`, `src/prc/validate_submission.py`;
`experiments/submissions.jsonl`; git commits `b3f4c55`, `f078298`, `7539b14`.

Competitor code, cloned to `/tmp/kind-mango` from
<https://github.com/skylinkapi/prc-data-challenge-2026-kind-mango>:
`src_v3/postprocess.py`, `src_v3/support.py`, `src_v3/labels.py`, `src_v3/config.py`, `src_v3/gate.py`,
`src_v3/build_v48.py`, `RECAP.md`, `docs/MODEL_ANALYSIS.md`.

Read-only measurement scripts written for this plan (scratch, not committed):
`/tmp/p6/support.py`, `/tmp/p6/rank.py`, `/tmp/p6/exceed.py`, `/tmp/p6/deep.py`, `/tmp/p6/bounds.py`,
`/tmp/p6/rates.py`, `/tmp/p6/censor.py`, `/tmp/p6/censor2.py`, `/tmp/p6/mirror.py`, `/tmp/p6/ceiling.py`,
`/tmp/p6/cluster.py`, `/tmp/p6/oof.py`, `/tmp/p6/oof2.py`, `/tmp/p6/rows.py`, `/tmp/p6/final.py`.
