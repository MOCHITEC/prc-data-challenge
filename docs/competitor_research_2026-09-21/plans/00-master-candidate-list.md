# Master candidate list — 24 research slots, deduplicated

Audit slot. Input: the "Applicable to resilient-hedgehog" section of all 24 files in
`../results/`, plus `../SUMMARY.md`. Every "does our repo already do it" answer below was
checked against the code on the current branch (`claude/model-b-and-arrival-features`,
identical to `main` for `src/` and `configs/` — `git diff main -- src configs` is empty),
not assumed.

Marking convention, carried from the slots: **[S]** = read from a source, **[X]** = the
slot agent's own extrapolation. Where I state a fact about *our* code I cite the file and
line and mark it **[R]** (read from this repository in this slot).

Where two slots disagree about the same candidate, both numbers are given and neither is
picked. There are five such disagreements: C8, C34, C44, G7 and the submission-limit fact
in G22.

---

## 0. How to read the table columns

For each merged candidate: **(a)** what it changes, **(b)** strongest sourced evidence for,
**(c)** sourced evidence against, **(d)** repo state (verified), **(e)** blocker.

Classification is one of: **already done** / **actionable now** / **blocked** / **refuted** /
**unmeasurable** (real by some measure but below our instrument's resolution, which
`docs/features_log.md` puts at 0.25 s **[R]**).

---

# Part I — candidates inside the ten planning themes

## P1 — unmatched (missing-proxy) lane specialist

Four independent slots converged here (08, 21, 22, 23; `../SUMMARY.md` §E). Our own figure
is unmatched RMSE 1,875.3 against matched ~218 **[R]** `src/prc/train.py:9`.

### C1. LIRF-unmatched gate-delay decomposition `T = max(D − Ĝ, 0)`
Slots: **06**.
(a) On LIRF rows with `AOBT_3` null, train a model for `G = D − T` where `D = MVT − SCHED`
on *all* ~160k LIRF departures, then overlay `T = max(D − Ĝ, 0)` on the unmatched ones only.
(b) OpenAir's own leaderboard log: **316.97 → 288.90 = −28.07 s from 383 rows**; LIRF-unmatched
holdout RMSE 6033 → 3937 → 3732 **[S]** (`06-repo-openair.md`, source
`ChandanHegde07/OpenAir`, `run_e33_delay_decomposition.py`, `status.md`).
(c) Their own guardrails are the counter-evidence to the wrong variants: training `G` on
unmatched rows only is **+12 s** (their v9, LB 300.95); applying `D − G` at any other airport
fails because `corr(T, D)` is 0.02 at LFPG and ≤0.44 outside LIRF; gating the override on any
threshold failed three times (E13, E19-C, E20) **[S]**.
(d) **Not done.** We route unmatched rows to model B and, at LIRF only, shift its target by
`mvt_minus_sched` **[R]** `src/prc/train.py:98-113, 238-267`. That is the same *mechanism*
(the schedule-fill class) reached by a different construction; it is not the `D − Ĝ`
decomposition, and it is off in `configs/main.yaml` (`unmatched_sched_reparam` defaults to
`False` **[R]** `train.py:113`; it is on only in `configs/submit_v10_reparam.yaml` **[R]**).
(e) Not blocked.
→ **actionable now.** Largest sourced single gain in the whole corpus.

### C2. Soft λ-blend of the two gate models (`G = (1−λ)·G_all + λ·G_unmatched_only`, λ 0.2–0.25)
Slots: **06**.
(a) Keeps a specialised sub-model at partial weight instead of letting it take the row.
(b) OpenAir LB **280.18 → 278.36 = −1.82 s**; the same `G_unmatched_only` at λ=1.0 cost
**+12 s** **[S]** (`06-repo-openair.md`, `status.md:1795`).
(c) None found.
(d) Not done — no blend of any kind exists in the prediction path **[R]**
`src/prc/train.py:predict_fold`.
(e) **Blocked on C1** (there is no `G` model to blend).
→ **blocked.**

### C3. Replace the hard substitution switch with a fitted soft weight
Slots: **09**, **21**, **08** (three independent forms of the same idea).
(a) Instead of "at LIRF, shift the target", predict `(1−w)·body + w·offset` with `w` fitted
per row.
(b) Strongest: sergiuv11's Repo B — hard *oracle* knowledge of which rows are substituted
scores **413.19**; the fitted weight scores **362.38**, i.e. the oracle is **51 s worse**;
board progression 411.23 → 334.01 → 318.16 **[S]** (`09-repo-sergiuv11.md`,
`repoB/src/taxiout/models/mixture.py:126-196`). Recipe detail that carries a measured price:
clipping the weight target to (0,1) instead of (−0.25,1.25) cost **15.8 s on the board** **[S]**.
Corroborated: merry-quicksand's per-row fill classifier, LIRF cell rate 48.5 % vs 0.2–9.4 %
elsewhere, "degrades gracefully to (1−p)·body at the nine airports where the mechanism is
absent" **[S]** (`21-repo-merry-quicksand.md` §6.1b); javidmardanov's LIRF identity
classifier, local **266.05 → 263.22** **[S]** (`08-repo-javidmardanov.md`,
`scripts/missing_v2.py:211-229`).
(c) None against the soft form. The evidence *against the hard form* is our own: generalising
our LIRF shift to all unmatched rows measures **2,696.7 vs 2,326.5** baseline, EHAM
954.7 → 3,408.9 **[R]** `src/prc/train.py:109-113`, cited from
`docs/investigation_2026-09-16/results/mms-mixture-reparam.md` §3.3.
(d) **Not done.** Ours is a hard per-airport switch keyed on `ADEP_mvt ∈ ("LIRF",)`
**[R]** `train.py:sched_reparam_shift`.
(e) Not blocked.
→ **actionable now.** Named by slot 09 as "the biggest transferable item in either repo" and
by slot 21 as "the one genuine upgrade available to us from this comparison".

### C4. Seed-bag the small unmatched specialist (≥5 seeds, declared up front, none discarded)
Slots: **07**, **12**.
(a) Fit model B under ≥5 seeds and average, each floored at 0 before averaging.
(b) Phoenix-Ops: **−1.6945 s official on 383 rows** from 5 equal-weight CatBoost seeds on a
1,488-row scope **[S]** (`07-repo-phoenix-ops.md`, `specialist_ensemble.py:37-50,80-82`).
Slot 12 generalises: "any sub-model fitted on ~10³ rows should be seed-averaged", and notes
our model B's variance problem is *worse* than theirs (22,470 rows, 48.1 % of total SSE) **[S/X]**.
(c) Slot 09 measures seed averaging as transferring almost exactly (6 s local → 5.4 s board,
one seed → two) but a third and fourth seed gave **+1.2** (v31) **[S]** — i.e. the number of
seeds is not monotone in their measurement. Slot 03: 10 seeds captures almost all of it, 20
adds 0.02 **[S]**.
(d) **Partially.** `n_seeds: 3` in `configs/main.yaml` **[R]**, applied to both models
identically in `_fit` **[R]** `train.py:_fit`. There is no separate, larger seed count for
model B.
(e) Not blocked.
→ **actionable now** (as "raise model B's seed count specifically"), and cheap.

### C5. Normalised missing-proxy target `Z` with provably-MSE-equivalent weights
Slots: **23**.
(a) On the unmatched cohort only, fit `Z = (y − 900)/sqrt(3600² + (schedule_proxy − 900)²)`
with sample weight `scale²/mean(scale²)`, reconstruct `900 + scale·Z`, blend at a fixed 25 %,
alongside an `ExtraTreesRegressor(n_estimators=300, max_features=0.7, min_samples_leaf=1,
bootstrap=False)` at 18.75 %.
(b) satam2 Repo B: **−7.04 s seasonal** for the pair **[S]** (`23-repo-col3name-satam2.md`,
`normalized_missing_tune.py:22-42,55`). Leaf=20 lost on all six panels — use leaf 1 **[S]**.
(c) Their own note: "an analogous finite-clock formulation was negative" — do not apply it to
matched rows **[S]**.
(d) Not done **[R]**.
(e) Not blocked.
→ **actionable now.**

### C6. Hierarchical shrunk cell mixture for the unmatched lane + the congestion witness
Slots: **21**.
(a) `[ap] → [ap, sp_band] → [ap, sp_fine, dayoff]` shrinkage with k=5; plus a *congestion
witness* feature (airport-hour median `MVT − AOBT_3` over that hour's **matched** departures,
n ≥ 5); plus winsorise the body fit but not the tail (`nf_hybrid` at 3,000 s).
(b) merry-quicksand: the congestion witness improved **+12/12 months**, shipped, board
**−1,775 MSE** **[S]** (`21-repo-merry-quicksand.md` §6.3.3).
(c) None directly. Note their unmatched lane is ~1,336 against our 1,490 **[X]** — a ~150 s
gap, so the ceiling on this lane is real but bounded.
(d) Not done **[R]**.
(e) Not blocked. The congestion-witness column is separable and cheaper than the mixture.
→ **actionable now.**

### C7. Report matched / unmatched / LIRF-unmatched RMSE as separate metrics in every run
Slots: **06**, **08**, **11**.
(a) Add the lane split to the CV record, not only to `compare`.
(b) OpenAir: LIRF-unmatched is "~30 % of total SSE from ~0.1 % of rows. Any aggregate-only
metric will hide both the wins and the regressions on this slice" **[S]**
(`06-repo-openair.md`). javidmardanov: 5,294/353,045 rows carry **28.4 %** of total SSE,
RMSE 1157.63 vs 226.85 **[S]** (`08-repo-javidmardanov.md`, `docs/missing_v2.json`).
(c) None.
(d) **Partially.** `run_cv` records `by_airport` per fold but no matched/unmatched split
**[R]** `train.py:run_cv`. `compare.py` has the full `tail_report` (unmatched diff, dominant
airport, per-airport, named dropped rows) **[R]**, but that only runs on an A/B pair.
(e) Not blocked. ~10 lines.
→ **actionable now.**

### Already done / refuted in P1
- **Separate model for the unmatched rows, routed by NM-match presence** — slot 23 calls this
  "the highest expected-value change available" at **−122.8 s seasonal** **[S]**, slot 08 the
  same at rank-1 priority. **Already done** since v2 **[R]** `train.py:predict_fold`, and our
  own record of merging them back is **−74.10 s worse** **[R]** `train.py:6-9`.
- **Residual target `y − (MVT − AOBT_3)` on matched rows** — slots 23, 08, 21 all rank it top.
  **Already done**, and slot 21 establishes it is algebraically *identical* to
  merry-quicksand's `delta` target with the sign flipped **[S]** (`21-…md` §6.1).
- **Declaring `AOBT_3` as a second identity candidate** — **refuted**: matches 8.25 % of rows,
  covers **0 monsters**, >15 % worse, reverted **[S]** (`09-…md` B2).
- **Generalising the LIRF schedule shift to all unmatched rows** — **refuted** by our own
  measurement, ≈ +370 s of unmatched RMSE **[R]**.

---

## P2 — cross-fit the target encodings

### C8. Cross-fit the four label-dependent encoding tables
Slots: **20** (for), **21** and **09** (against — see (c)).
(a) `ref_p10/p20/med/n`, `arh_med/n`, `apt_med`, `op_offset_med/n` are currently fitted on
`train_f` and joined back onto those same rows. Replace with an inner-fold (or closed-form
leave-one-out) fit for the training side, leaving the 2026-side join unchanged.
(b) **For:** CatBoost paper arXiv:1706.09516 §3.2 proves Greedy TS induces prediction shift;
sklearn calls `fit().transform()` on training data "discouraged… can introduce data leakage"
and ships cross-fitting as the default **[S]** (`20-tabular-comp-craft.md`). Our own
CV/leaderboard signature matches: CV rewarded capacity by **34.8 s** and the board charged
**13.2 s** **[R]** `configs/main.yaml:1-8`. Slot 21 independently documents merry-quicksand
hitting the identical defect in their early-stopping path and never repairing it **[S]**.
(c) **Against, and this is a real disagreement.** Slot 21: merry-quicksand list
"out-of-fold/cross-fitted target encoding as a predictive change (**−0.96 s**)" among two
*confirmed dead ends* **[S]** (`21-…md` §6.3.5). Slot 09: sergiuv11 Repo A ship
leave-one-month-out cross-fitted target encoding with winsorised means and parent smoothing,
and it **failed their gate** — slot 09's verdict is "**No.** Recorded here so we do not
rediscover it" **[S]** (`09-…md` A6). So: one slot argues from theory plus our own CV/LB
signature, two slots report competitors measuring it as neutral-to-negative *as a predictive
change*. Slot 20's own acceptance test is aware of this — it predicts CV will move little and
the *leaderboard* will move more, and names one leaderboard submission as the decisive read.
(d) **Not done. Confirmed defect [R]**: `fit_encodings(train_f, cfg)` then
`attach_encodings(train_f, enc)` at `src/prc/train.py:367-368`; the tables themselves at
`train.py:200-241`. `reference_min_count: 30` **[R]** `configs/main.yaml` bounds the
self-inclusion at 1/30 but does not remove it.
(e) Not blocked.
→ **actionable now**, with the disagreement recorded: the theory and our own CV/LB gap say
yes, two competitor measurements say it does not move the score. Both slots agree the
decisive read is a leaderboard submission, not CV.

---

## P3 — OOF fold column and validation protocol

### C9. Add a `fold` column to the OOF parquet
Slots: **11** (Step 0, "prerequisite; blocks everything else"), **15**.
(a) One column in the file written at `train.py:534`, plus the fold list in the experiment
record.
(b) Slot 11: H2O and sklearn both make a shared, identifiable partition a precondition for
fitting a meta-learner **[S]**. Slot 15: sklearn states `cross_val_predict` output is "from
several distinct models undistinguished" and "is not an appropriate measure of generalization
error" **[S]**.
(c) None.
(d) **Not done.** The OOF frame is `{MVT_ID_mvt, oof}` only **[R]** `train.py:run_cv` and
`experiments/oof_*.parquet`. The folds also carry different `train_months` (`jan` trains on
the future, `forward` does not) **[R]** `src/prc/cv.py:month_block_folds`, and `pl.coalesce`
resolves overlaps silently in fold order — already written down at `train.py:run_cv` with the
number (overall n 672,428 against a fold sum of 863,141) **[R]**.
(e) Not blocked.
→ **actionable now.** Prerequisite for C34, C50, and anything fitted on OOF.

### C10. Stop pooling OOF rows across folds for any adopt/reject statistic
Slots: **14** (A1).
(a) Compute the decision statistic per fold and require agreement, or designate one fold.
(b) "Every sourced comparison in this literature is made inside one fixed fold structure with
one model; nobody pools heterogeneous models' OOF rows and calls it a validation set" **[S]**
(`14-validation-design.md` §3).
(c) None.
(d) Not done — `compare.py` pools **[R]**.
(e) Blocked on **C9** for the general case.
→ **actionable now** (partially; full form needs C9).

### C11. Blum & Hardt Ladder threshold: adopt only if |Δ| > s/√n
Slots: **14** (A2).
(a) A published, parameter-free floor for "real", replacing our implicit 0.25 s screening
resolution. Inputs (`s` = SD of per-row squared-error differences, `n`) are already computed.
(b) arXiv:1502.04585 **[S]** (`14-…md` §7.1). ~20 lines, no new computation.
(c) None.
(d) Not done. `cv.paired_bootstrap_rmse_diff` computes the paired differences already **[R]**.
(e) Not blocked.
→ **actionable now.**

### C12. 5-seed mean ± SD bar on any finalist
Slots: **14** (A3), **09** (B5), **21** (§6.4).
(a) Refuse to adopt when |Δ| is under one seed SD.
(b) Three independent noise-floor measurements: PRC 2025 Grigoriou's published 5-seed SD is
~13–14 kg on a ~200 kg metric (≈7 %), and the authors' chosen model was reversed by the hidden
set **[S]** (`14-…md` §7.2); sergiuv11 Repo B seed spread **10 s**, cross-machine **4 s**,
paired floor **~5 s** **[S]** (`09-…md` B5); merry-quicksand require ≥3 seeds with a
`2 × seed sd` bar **[S]** (`21-…md` §6.4).
(c) Cost: 5× training per candidate — slot 14 names this as the dominant cost of its list **[S]**.
(d) Not done. `n_seeds: 3` averages seeds inside one fit; no per-seed spread is recorded **[R]**
`train.py:_fit`.
(e) Not blocked.
→ **actionable now.** Note the implication: our whole 10-rejection feature log ran at
`n_seeds: 1` **[R]** `docs/features_log.md` measurement conditions, and every |Δ| in it is
under the seed SD that slots 09/14/21 measure.

### C13. Single-row-share guard: reject if one row supplies >50 % of the net SSE advantage
Slots: **09** (A2, B4), **12** (gate criterion iii), **08** (15).
(a) ~20 lines in `compare.py`.
(b) sergiuv11 Repo A ship it (`select_airport_membership.py:64,136-138`) **[S]**.
javidmardanov's V4 is the measured failure it catches: a change moving **6 validation rows /
3 ranking rows** improved local RMSE by 1.96 s and made the official score **0.46 s worse**
**[S]** (`08-…md` 15). Phoenix-Ops' v11 got 46 % of its gain from one row **[S]**
(`15-postprocess-overlays.md` §4).
(c) None.
(d) **Not done.** `compare.py` has a drop-K curve at K ∈ {0,20,200,2000} and names the dropped
rows **[R]**, which is a related but different instrument (it removes the worst K, it does not
compute a single row's share of the *advantage*).
(e) Not blocked.
→ **actionable now.** Slot 09 calls it "cheap and high-value"; with 10 OOF rows carrying 24 %
of our SSE **[R]** `compare.py:5-8`, most of our sub-5-second gate calls could be one-row
artefacts.

### C14. Day/date-block bootstrap
Slots: **21** (§6.4), **09**, **08** (15).
(a) Resample whole dates, not rows and not months.
(b) merry-quicksand measured the row-vs-day CI width ratio on a tail-living arm at **4.6×**
(range 1.5–4.6×) **[S]** (`21-…md` §6.4, `../SUMMARY.md` §H).
(c) None.
(d) **Partially.** `cv.paired_bootstrap_rmse_diff` is row-level; `compare.month_bootstrap_rmse_diff`
resamples whole months **[R]**. Day-blocks sit between the two and are missing. Slot 21 names
our own symptom: a suspiciously narrow 0.39–0.60 s matched CI recorded in
`docs/CODEX_BRIEF.md:61-62` **[S]**.
(e) Not blocked.
→ **actionable now.**

### C15. Weight the selection metric by the ranking set's Jan/Jul row mass (44.3 % / 55.7 %)
Slots: **24**, **23**.
(a) `sqrt(0.557·MSE_Jul + 0.443·MSE_Jan)` as the promotion metric.
(b) The composition is corroborated from five independent sources across slots 06, 16, 24 and
the arithmetic closes to the unit: Jan 152,719 + Jul 192,122 = 344,841 **[S]**
(`24-repo-tail-sweep.md` §1; `../SUMMARY.md` §A). sqcm9896 measured that 50/50 "flatters by
~11 pts" **[S]**. satam2 use exactly this metric and their table shows candidates that win on
one month lose on the weighted score **[S]** (`23-…md` 5, `metrics.py:23-28`).
(c) None.
(d) **Not done.** Our folds are `jan` / `jul` / `jul_fwd` / `forward` and `overall` pools one
prediction per row unweighted **[R]** `train.py:run_cv`, `cv.month_block_folds`. Note our
`forward` fold is November+December — months the ranking set does not contain **[R]**.
(e) Not blocked.
→ **actionable now.** Slot 24's corollary is load-bearing: "a change that loses on December is
not evidence against the change" **[S]** — and `forward` is our primary screening fold **[R]**
`train.py` `folds` docstring.

### C16. Add the both-months-improve arm; do not substitute it for our rule
Slots: **24**, **07** (9), **10** (9).
(a) Report Jan RMSE, Jul RMSE and the mass-weighted pooled on every run; require the pooled CI
and the drop-200 CI to improve (our existing rule) and treat a single-month regression as a
flag demanding a named mechanism, not an automatic veto.
(b) Slot 24 did the orthogonality analysis properly in both directions: sqcm9896's gate passed
**both** of their documented live failures (v6 per-airport blend weights −0.6 holdout →
**+7.0 leaderboard**; v7 LIRF small-slice patch −1.2 holdout → **+0.95 leaderboard**), which a
drop-200 arm is built to catch; and their gate discarded a **−8.6 s pooled gain**
(`extreme_spec`, pooled 350.6 vs ship 359.2) because January regressed 5.9 s while July
improved 23.5 s **[S]**. Phoenix-Ops pre-registered the same conjunction (improvement in every
validation month, ≥40/62 days) **[S]** (`07-…md` 9).
(c) Slot 07's own warning about the same gate: their v13 passed by **0.0006 s** — "a gate you
can retry against is a gate you will eventually fit" **[S]**.
(d) Not done **[R]**.
(e) Blocked on **C15** for the weighting; the month arm alone is free.
→ **actionable now.**

### C17. Per-airport *and* per-month deltas in every adopt/reject report
Slots: **14** (A7), **03** (13).
(a) Generalise `per_group_diff` from unmatched-only to every comparison, and add a month axis.
(b) PRC 2025 Grigoriou reports validation-vs-hidden metrics per phase, per aircraft type and
per haul category and uses them to localise failure (B77W); PRC 2024 winner's per-type RMSE
spans 662 → 11,893; reviewer Mori: "Relying solely on a single overall RMSE might mask
performance discrepancies" **[S]** (`14-…md` §8, `03-prc-2024-writeups.md` 13).
(c) None.
(d) **Partially.** `run_cv` records `by_airport` per fold **[R]**; `compare.per_group_diff`
runs per airport but is called with the unmatched mask only **[R]** `compare.tail_report`. No
month axis anywhere in `compare` except the month bootstrap.
(e) Not blocked.
→ **actionable now.** Slot 14's argument is that this is the discriminator our drop-200 rule
lacks: many airports and months moving together is structure, one airport's handful of rows is
a lottery, and our rule currently rejects both **[S]**.

### C18. Two-stage screen-then-confirm protocol
Slots: **14** (A4).
(a) Screen many candidates on `forward` alone; confirm the top handful on an evaluation the
screen never touched (new seeds + `jul_fwd`).
(b) PRC 2025's two-stage tuning against "optimization leakage" **[S]** (`14-…md` §7.3).
(c) None.
(d) **Partially.** `cfg.folds` supports forward-only screening and the docstring documents the
resolution trade **[R]** `train.py`. `jul_fwd` exists and is computed **[R]** `cv.py:36-41`,
but is silently discarded from `overall` by the coalesce **[R]** `train.py:run_cv` — so the
honest confirm fold is built and then thrown away.
(e) Not blocked.
→ **actionable now.**

### C19. Committed experiment log with an explicit `decision` column, rejections kept
Slots: **24**, **09** (A3 fail-closed variant).
(a) sqcm9896 keep a 110-row `reports/experiments.csv` with `decision` values
(`drop-jan-rule`, `drop-mixed`, `drop-noise`, `drop-bug`) **[S]** (`24-…md` §2).
(b) Slot 24: "copy the cheap part outright". Slot 03 item 17: PRC 2024 rank 4 lost a ~2,250
result better than their 2,276 submission because "it was a messy day where we trained so many
models … and lost track of it" — "the cheapest lesson in the corpus" **[S]**.
(c) None.
(d) **Partially.** `experiments/experiments.jsonl` has 43 records with config, git sha, fold
metrics and per-airport RMSE **[R]**, but **zero** records contain a `decision` field
(`grep -c decision` = 0 **[R]**). Decisions live in prose in `docs/features_log.md` **[R]**.
(e) Not blocked.
→ **actionable now.**

### C20. Kish ESS check before believing any unmatched-lane interval
Slots: **21** (§6.4).
(a) Compute the effective sample size of the unmatched-lane bootstrap.
(b) merry-quicksand's own figure: **22.9 effective observations out of 22,219 rows** **[S]**
(`21-…md` §6.4).
(c) None.
(d) Not done **[R]**.
(e) Not blocked.
→ **actionable now.** Explains directly why our unmatched CI is "hundreds of seconds wide"
**[R]** `configs/submit_v10_reparam.yaml` header.

### C21. Fail-closed gate chain; re-fit any post-hoc overlay per arm
Slots: **09** (A3, A4).
(a) `reproduce_v3.sh:78-99` literally greps for "Verdict: PASS" and `exit 1`s otherwise;
`check_gate()` refuses a final fit without a preserved report and its matching prediction file.
Separately, an overlay under A/B must be re-fitted on each arm's fit pool
(`compare_predictions.py:9-12`) or the comparison is self-scoring **[S]** (`09-…md`).
(b) As above.
(c) None.
(d) Not done. `Makefile` `run` and `compare` are independent targets with no gate coupling
**[R]**; `submit.py` enforces a daily counter, version uniqueness and a mandatory log entry
**[R]** but not a gate verdict.
(e) Not blocked. A4 is only binding once an overlay exists (C29–C34).
→ **actionable now.**

### C22. MSE as the unit of account, with fixed close/ship bars
Slots: **21** (§6.4).
(a) merry-quicksand use a **1,000-MSE close bar** and a **500-MSE ship bar** **[S]**.
(b) Their board arithmetic is stated exactly: `MSE = 0.9846596·m² + 0.0153404·u²` **[S]**
(`21-…md` §6.2), which converts a lane-local change into a board delta without a probe.
(c) None.
(d) Not done — everything is in seconds of RMSE **[R]**.
(e) Not blocked.
→ **actionable now.** The lane-weight identity is the part with independent value: it lets us
price an unmatched-lane change against the board before spending a submission.

### Already done / refuted in P3
- **Never filter extremes out of the validation set** — slot 24's two worst repos both do it
  (kagervig's −113.6 s "breakthrough" is 69 rows removed from both sides of the split, real
  score 573.85) **[S]**. **Already done correctly**: we keep every labelled row and
  `rmse_full` is the reported metric **[R]** `cv.metrics`.
- **Do not pass the holdout as `eval_set`** — sqcm9896's holdout is contaminated and their own
  red-team file says so **[S]** (`24-…md` §2 caveat). **Already done correctly**: early
  stopping uses an inner holdout carved from the training months and the model is refitted on
  all training rows **[R]** `train.py:_inner_holdout_mask`, `_fit`.
- **Purge flight-ID overlap between folds** (slot 23 item 7) — **already satisfied**: our rows
  are one per departure and folds are month-blocked **[R]** `cv.month_block_folds`.
- **Pin seeds / `deterministic=True`** (slot 03 items 16, 15) — **already done**, including the
  `force_row_wise` fix and the reason for it **[R]** `train.py:TrainConfig.params`.
- **Version every experiment** (slot 03 item 17) — **already done** (`exp_id`, config, git sha)
  **[R]** `train.py:run_cv`; the missing half is C19.
- **Adversarial-validation test-like quantile as the validation set** (slot 14 A6) —
  **refuted for us, by us**: scoring the two candidate models on the most ranking-like 1 %,
  5 % or 10 % of training rows, or weighting every row by the AV probability, ranks them the
  same way plain CV does, and the leaderboard disagrees with that ranking **[R]**
  `src/prc/adversarial.py:20-30`. The AV AUC slot 14 asks for already exists: **0.807** with
  calendar columns removed, 0.819 restricted to Jan+Jul **[R]**. Slot 14 also records AV-derived
  *sample weights* as sourced-and-negative **[S]**.
- **A fitted linear CV→LB correction** — slot 14 explicitly would not adopt it (no source, not
  identifiable from the published paired points) **[S]**. **Refuted.**

---

## P4 — `stand_arrival_gap` as a feature, plus `stand_arrival_taxi`

### C23. Merge `stand_arrival_gap` and ship it as a feature (not a correction)
Slots: **18**.
(a) Backward `join_asof` from the departure onto the same-stand arrival stream; emit the gap.
(b) Our own measurement on branch `codex/feature-engineering` at commit `8bdf4f9`, from
`docs/codex_fe_results/continuation_evaluation.json`: **as a feature −3.242 s
[−4.282, −2.225]**, 9/10 airports, −3.097 s against a matched hash-noise control; as a post-hoc
correction only **−0.781 s**, 7/10 airports **[S]** (`18-turnaround-features.md` §C). EHAM
flips +0.374 → −0.722 and LEMD +0.428 → −0.263 purely by changing the mode of use **[S]**.
(c) Slot 18's own honesty note: the confirmation used descriptive 95 % intervals chosen *after*
seeing the outcome, "treat −3.242 as a strong indication, not a pre-registered result" **[S]**.
(d) **Not done on the shipping branch.** `stand_arrival_features` exists only on
`codex/feature-engineering` (`git show codex/feature-engineering:src/prc/features.py:294`,
`CANDIDATE_COLS` = `stand_arrival_gap`, `stand_aobt_turn`, `stand_arrival_type_match`, cap
21,600 s) **[R]**. Our `main`/current `CANDIDATE_COLS` contains none of them **[R]**
`src/prc/features.py:265-276`.
(e) Not blocked.
→ **actionable now.** Largest measured-on-our-own-data candidate that is not shipped.

### C24. Add `stand_arrival_taxi` (the joined arrival's `TAXITIME_SEC_mvt`)
Slots: **18**, **06**, **07**.
(a) One extra column off the same asof join.
(b) **The exact column that moved OpenAir 284.10 → 282.09, −2.01 s LB, the largest single
feature-level gain in their log** **[S]** (`18-…md` D1, source `ChandanHegde07/OpenAir`).
Phoenix-Ops' v7 arrival block (which contains it) was **−2.6224 s official** **[S]**
(`07-…md` 2).
(c) Our own screen ranked it 4th (correction Δ −0.441) and dropped it — but slot 18 shows the
screen ranked by *correction* value, which §C proves is the wrong ordering criterion **[S]**.
(d) **Not done anywhere**, not even on the codex branch **[R]**.
(e) Not blocked; cheapest if done together with C23.
→ **actionable now.**

### C25. `stand_arrival_delay` (AIBT−SIBT), `arr_taxiin_2`, `rwy_arr_taxiin`
Slots: **18**, **06**.
(a) Three more columns off the same join family.
(b) OpenAir's v17+v18 tail, **−0.37 s combined LB** **[S]** (`18-…md` D5).
(c) Below our 0.25 s instrument resolution individually **[R]** `docs/features_log.md`.
(d) `stand_arrival_delay` was screened (−0.408) and dropped **[S]** (`18-…md` §B); the other
two do not exist **[R]**.
(e) Not blocked.
→ **actionable now** as part of the C23/C24 block; **unmeasurable** individually.

### C26. Widen the same-stand recency cap beyond 6 h
Slots: **18**, **06**, **07**.
(a) Our cap is 21,600 s. Phoenix-Ops use **86,400 s**, OpenAir **8 h** matched / **36 h** for
the LIRF gate-hold model **[S]** (`18-…md` D3, `07-…md` 2, `06-…md` 1,3).
(b) "Our 6 h is the tightest of the four" **[S]**.
(c) OpenAir are explicit: **null everything outside the window, do not zero-fill** **[S]**.
(d) Cap is 21,600 s on the codex branch **[R]** `git show codex/feature-engineering:src/prc/features.py:325`.
(e) Blocked on **C23** (the feature must exist first).
→ **blocked** (trivially, on C23).

### C27. AOBT-anchored twin `arr_taxiin_aobt` / `since_arr_aobt`
Slots: **18**, **06**.
(a) A second asof anchored at the departure's AOBT rather than its take-off.
(b) Part of OpenAir's −0.21 s v17 step **[S]** (`18-…md` D4).
(c) Below our resolution individually.
(d) We have the first of the pair (`stand_aobt_turn`) on the codex branch, not the second **[R]**.
(e) Blocked on C23.
→ **blocked / unmeasurable.**

### C28. Phoenix-Ops' 22 airport- and runway-keyed trailing taxi-in statistics
Slots: **18**, **07**, **17**.
(a) Airport and airport×runway rolling mean/std/count of arrival taxi-in over 15/30/60 min
strictly before the departure's take-off, plus last-completion age and value with a 2 h
recency cap and tie-averaging.
(b) Half of the −2.6224 s v7 step **[S]** (`07-…md` 2, `18-…md` D6). Slot 17 verifies
availability directly: ranking ARR `TAXITIME_SEC_mvt` has **0 nulls across 344,693 rows** —
"the only place in the whole catalogue where neighbours' *actual* taxi labels are legally
available to us" **[S]**.
(c) OpenAir's `DISCOVERY_REPORT.md` §6: "Recent taxi-in | available; weak / unstable
(EHAM −0.28, others ~0.1)" **[S]** (`17-congestion-features.md` #4). So slots 07/18 measure a
gain from the block and OpenAir measure the same family as weak — the difference is that
Phoenix's block is runway-keyed and has sd/last-age terms.
(d) **Partially.** We have `arr_taxi_in_mean_{30,60}m` and `arr_taxi_in_n_{30,60}m`, airport-keyed
only, no sd, no 15 min, no last-completion, no runway key **[R]** `src/prc/features.py:context_features`.
(e) Not blocked.
→ **actionable now.**

### Blocked / refuted in P4
- **Same-aircraft turnaround link** (slot 05 item 12) — **blocked**: only **14.77 %** of
  ranking departures have a matching arrival row inside the ranking file **[S]**
  (`05-nasa-pushback.md` 12, citing our `docs/data_profile.md` §7). The airport-level surrogate
  (C28) covers the other 85 %.
- **`FLIGHT_ID_mvt` as a tail/rotation key** — **refuted**: it is a city-pair NM flight and the
  paired ARR is *after* the DEP **[S]** (`06-…md`, `18-…md` E(b)).
- **Any hard bound `taxi_out ≤ stand_arrival_gap`** — **refuted**: OpenAir E72, LIRF-unmatched
  3,896 → 11,579 **[S]** (`18-…md` E(a)).
- **External ADS-B registration chaining (OPDI `icao24`) for turnaround** — **refuted**:
  measured worse, 236.06 vs 235.94 **[S]** (`18-…md` E(d)).

---

## P5 — physics-identity lower bound post-processing

### C29. `pred = max(pred, m_aobt − 3600)` where `m_aobt = MVT − AOBT_3 > 1800`
Slots: **24**.
(a) A bound, not a fit: no prediction may sit more than an hour below the observed
off-block-to-take-off span.
(b) **The single best local→live transfer in the entire corpus**: sqcm9896's holdout predicted
−23.9 and the leaderboard delivered **−20.3 (336.46 → 316.19) on 11,826 triggered rows**
**[S]** (`24-…md` §3). Slot 24's own explanation — "It is a bound, not a fit, which is why it
transferred where their slice-fits did not" **[X]**.
(c) The matching negative is about the *opposite* direction only: upper-tail clipping measured
**+90.7 s** (`clip_cap2h_v1`) — never cap from above globally **[S]** (same slot). Nothing
against the lower bound.
(d) **Not done.** The only post-processing in our path is `clip(lower_bound=0.0)` **[R]**
`src/prc/train.py:_assemble`; there is no overlay stage anywhere in `pipeline.py` **[R]**.
(e) Not blocked.
→ **actionable now.** Highest evidence-per-line item in the document.

### C30. LOBT timing bounds: clip every prediction to `(MVT − LOBT) ± 3606 s`
Slots: **08**.
(a) A second, independent bound from a second clock.
(b) javidmardanov verify **zero violations** across all non-held-out training months
(`scripts/queue_features.py:79-91`), applied at `scripts/ensemble.py:153-167`;
`bounds_changed_rows: 55` of 353,045 in v1 — "tiny row count, but those are exactly the
catastrophic rows" **[S]** (`08-…md` 3).
(c) None. Slot 08's own caveat: verify the ±3606 invariant on our own data before trusting it
**[X]**.
(d) Not done. `mvt_minus_lobt` exists as a feature **[R]** `features.py:clock_and_static_features`;
no bound is applied.
(e) Not blocked (~20 lines, plus the invariant check).
→ **actionable now.**

### Already done / refuted in P5
- **`max(prediction, 0)`** — **already done** **[R]** `train.py:_assemble`. javidmardanov
  shipped V5/V6 with negative predictions before catching this **[S]**.
- **Global upper cap on the label or the prediction** — **refuted** five ways
  (`../SUMMARY.md` §F; slots 07, 09, 24): drop >120 min +24.1, drop >60 min +36.3, cap 7200 s
  **+77.6 on the board**, cap 4500 s +96.1, drop 17 rows >6 h **+13.7 on the board despite
  −9.9 on the bulk**, Phoenix-Ops label cap 434.1 vs 382.3 **[S]**.
- **Integer rounding of predictions** — **refuted** as an MAE artefact **[S]** (`05-…md` 18).
  Note our `write_submission` rounds only if the template dtype is integer, and logs the max
  shift **[R]** `predict.py` — correct behaviour.

---

## P6 — per-airport upper bound from 2025 label support

### C31. Cap non-LIRF unmatched predictions at the per-airport 2025 label support
Slots: **15**, and our own prior result.
(a) A bounded-downside overlay on ~2–10 ranking rows.
(b) Two independent measurements of the same rule **[S]** (`15-postprocess-overlays.md` §1):
our own OOF (`docs/investigation_2026-09-16/results/tail-postprocessing-cap-shrink.md`,
verdict (d) CONFIRMED) — cap non-LIRF unmatched at 3,600 gives **334.58 → 327.05**, 53 rows,
49 helped / 4 hurt, ΔSSE negative in all four OOF months and all eight affected airports; at
7,200 → 329.33, **24 helped / 0 hurt**. And `kind-mango` shipped the per-airport-support
version on the *same scored set* for **−2.33 s live on 5 rows at a 0 MSE fold price** **[S]**.
(c) Slot 15's own honesty: "uniform-sign on 24 rows is still 24 rows"; the named risk is a
genuine 2026 multi-hour non-LIRF taxi, costing ~0.04 s **[S]**. "A cap is the one overlay
family with bounded downside" **[S]**.
(d) **Not done.** Confirmed in our own docs and never shipped **[R]**
`docs/investigation_2026-09-16/results/tail-postprocessing-cap-shrink.md`.
(e) Not blocked.
→ **actionable now.**

### C32. Clip every prediction to a per-airport `[p0.5, p99.5]` of the training target
Slots: **05**.
(a) The broader, all-rows version of C31.
(b) NASA 1st place clipped members to [1,299] min and the ensemble to [4,260] min **[S]**
(`05-…md` 16). "For RMSE this is strictly more valuable than for MAE" **[S]**.
(c) The upper half of this collides directly with the refuted global upper cap
(`../SUMMARY.md` §F) — our target has p99 = 2,339 s and a max of 131,167 s **[S]**
(`05-…md` 16 citing our `data_profile.md` §3), so a p99.5 cap would fire on the exact rows
that cost +77.6 s on the board when capped.
(d) Not done **[R]**.
(e) Not blocked, but the upper half is refuted for us.
→ **refuted as stated**; the lower half is subsumed by C29 and the upper half by C31's
narrower, bounded form.

### C33. Clip LEMD and LEBL predictions at 3,600 s if the upstream censoring still holds
Slots: **24**.
(a) fenil264's EDA: **LEMD and LEBL DEP taxi-out is hard-censored at 3,600 s upstream** —
top-10 maxima at both packed at 3,504–3,604 while LFPG and LIRF decay naturally to 84k and
131k **[S]** (`24-…md` §4).
(b) As above. "If that pipeline still runs in 2026, clipping our LEMD/LEBL predictions at 3600
is free" **[S]**.
(c) Slot 24 itself requires verification on our own 2025 data first (count LEMD/LEBL DEP rows
above 3,600), then a gate **[S]**.
(d) Not done; not verified **[R]**.
(e) Not blocked — the verification is a one-line query on `data/work/feat_train.parquet`.
→ **actionable now** (verify first).

### C34. Per-airport OOF bias correction, one additive constant per airport
Slots: **12** (for), **15** (against). **Disagreement.**
(a) Compute per-airport OOF residual means on the training folds and subtract.
(b) **For:** NASA 3rd place shipped the residual-offset stage and placed 3rd; M5 2nd and 5th
ship multiplier versions; OpenAir measured per-airport residual means of **−236 s (LTFM)** and
**+214 s (LIRF)** on data of this shape **[S]** (`12-per-group-specialists.md` 3). Slot 12
wants it run *before* per-airport specialists, because if it captures the gain the specialists
are redundant.
(c) **Against:** slot 15 §2 is STRONG-negative on fitting *any* calibration on our current OOF
file — sklearn says `cross_val_predict` output "is not an appropriate measure of generalization
error", ours assigns July twice and discards one copy, and `kind-mango` measured "post-hoc
airport-level shift calibration — near-zero effect" on this competition's data **[S]**.
Published offline→live transfer ratios in this competition: 14 %, negative, 35 % **[S]**.
Slot 12's own honest check answers slot 15 halfway: compute the constant on an early period and
apply it to a later one; if the sign is unstable it is drift, not bias, and must not ship **[S]**.
(d) Not done **[R]**.
(e) **Blocked on C9** — slot 15's objection is specifically about the OOF file's shape, which
C9 fixes.
→ **blocked**, with the disagreement recorded: slot 12 says run it first, slot 15 says the
instrument cannot support it and a competitor measured near-zero.

---

## P7 — model capacity: deep model on clipped residuals, blended

### C35. Deep model on residuals clipped to ±7200, blended 50/50 with the incumbent
Slots: **07**, **18**, and **17** for the model-dependence caveat.
(a) Refit the residual target with roughly an order of magnitude more capacity and average the
two predictions. **Zero new features.**
(b) Phoenix-Ops' v8: `depth=9, lr=0.05, l2_leaf_reg=7, border_count=254, 4999 iterations, no
early stopping`, on `clip(y − proxy, −7200, +7200)`, then `0.5·incumbent + 0.5·new`, both
floored at 0 — **−7.1141 s official, their largest single gain** **[S]** (`07-…md` 1,
`arrival_boost_contest.py:29-33,43-45,77`). Independently corroborated: OpenAir E68 found
`2000×127` and `1500×255` both beat their shipped `n_estimators=400, num_leaves=31` **[S]**
(`18-…md` §A). Slot 18 calls this "the one lever both independent teams confirmed".
(c) The constraint is measured: **cap the residual, never the label** — label capping scored
434.145 vs 382.284 **[S]** (`07-…md` 1, `METHOD.md:28`). And slot 17 supplies a warning that
cuts the other way for *our* model: the same 30 following-context features gained 1.4 s through
a 4,999-tree CatBoost and only **0.283 s** through a 679-tree LightGBM — "the gain is
model-dependent" **[S]**. Our model is LightGBM at 127 leaves / 2000 rounds **[R]**.
(d) **Not done.** `num_leaves: 127`, `num_rounds: 2000`, no residual clipping, no blend **[R]**
`configs/main.yaml`, `train.py:predict_fold`.
(e) Not blocked.
→ **actionable now.** `../SUMMARY.md` §B corrects the earlier record here: v8 is *not* a
feature addition — same 123 predictors as v7.

### C36. Make the global model bigger before making ten smaller ones (the missing control)
Slots: **12** (4), **18** (§A).
(a) Fit model A with roughly the summed capacity of ten specialists and score it against both
the current model A and a specialist blend on the same folds.
(b) Montero-Manso & Hyndman's complexity result: "the complexity of the local approach grows
with the size of the set, while the complexity of the global approach remains constant",
prescribing "comparatively more complex hypothesis classes for global algorithms" **[S]**
(`12-…md` 4). The NASA KDD paper's global-vs-per-airport tie at 10 airports with a rich feature
set **[S]**.
(c) Our own most expensive lesson is a capacity lesson in the opposite direction: model B at
255 leaves / min_data 20 was CV-rewarded by **34.8 s** and board-charged **13.2 s** **[R]**
`configs/main.yaml:1-8`. So capacity increases here must clear the drop-200 arm.
(d) Not done **[R]**.
(e) Not blocked.
→ **actionable now.** Slot 12: "This is the decisive control experiment and it is missing from
every competitor log I read" **[S/X]**.

### C37. Algebraic component swap: `new = incumbent + w·(Model_wide − Model_narrow)`
Slots: **07**.
(a) Upgrade one blend component without refitting or reweighting the ensemble; every previously
verified prediction stays bit-exact outside the change.
(b) Phoenix-Ops v10, **−2.5779 s official** **[S]** (`07-…md` 3,
`neighbor_boost_contest.py:29,57-72`).
(c) None.
(d) Not done; there is no blend to swap into **[R]**.
(e) **Blocked on C35** (needs an ensemble with a known component weight).
→ **blocked.**

### Also in P7
- **Overlay-and-splice architecture** (slot 06 item 4): every improvement is a small-population
  row replacement on the previous submission parquet joined by `MVT_ID_mvt`, with hard asserts
  (row count == 344,841, ID order identical, zero nulls, min ≥ 0). "It is what made 22
  submissions possible in 9 days" **[S]**. Slot 08 item 14 is the same discipline (assert rows
  outside the gate are bitwise unchanged) **[S]**. **Partially done** — we already build probe
  submissions this way by hand (v9, v10, v11 notes in `experiments/submissions.jsonl` **[R]**),
  but there is no reusable splice helper with asserts in `src/` **[R]**. → **actionable now**,
  and it is a prerequisite in practice for C1, C29, C31, C33.
- **Deeper hyperparameter search** — **refuted**: OpenAir's full sweep plus seed averaging
  bought **−0.43 s** on Jan+Jul and 0.00 s on December **[S]** (`06-…md`); XGBoost's tunability
  is 0.014 AUC vs sensible defaults over 38 datasets **[S]** (`20-…md` 9). Our params are
  already hand-set away from defaults **[R]**.

---

## P8 — weather re-measurement and a coverage-parity gate

### C38. Re-measure the IEM join with the fixed join; close the `p01i` question
Slots: **16**.
(a) Re-run the weather A/B now the `hour_utc`-deleting bug is fixed.
(b) Gain already demonstrated: matched gate rows **−8.0 s**, and ×2.70 more gate-hour exposure
in scored January than in training January **[S]** (`16-external-data.md` 1). Coverage
confirmed for both scored months by live probe; cache on disk covering 2025-01-01 → 2026-08-01;
licence public domain **[S]**.
(c) The open question is now answered and it is a dead end, not a bug: "`p01i` is empty because
IEM states precipitation data is unavailable for non-US sites" **[S]**.
(d) **Partially.** Weather is on (`use_weather: true`) and the adopted decision is recorded as
−1.06 [−1.40, −0.73] after the 200-row drop **[R]** `configs/main.yaml`; the `hour_utc`
collision is fixed and documented **[R]** `src/prc/weather.py:165-170`. The re-measurement slot
16 asks for is of the *gate rows* specifically and has not been run **[R]**.
(e) Not blocked.
→ **actionable now.**

### C39. Add Open-Meteo `snowfall` / `precipitation` for the 10 airport coordinates
Slots: **16**.
(a) Fills exactly the field IEM withholds for European stations.
(b) Coverage verified through 2026-09-19; both scored months inside; CC BY 4.0; the module
docstring already names Open-Meteo as the secondary source **[S]** (`16-…md` 2). It targets
the de-icing residual our own `seasonal-deicing-shift.md` isolates **[R]**.
(c) "No competitor has published a number for this; the gain is inferred from our own gate-row
residuals, not measured" **[X]** (slot 16's own marking). And slot 24: sqcm9896 measured
**ERA5 via Open-Meteo at 485.5 vs 448.8**, both months worse — but slot 24 is explicit that
"the weather result is specifically about ERA5's coarse grid", not about station data or about
snowfall **[S]**.
(d) Not done. `weather.py` fetches IEM only; the Open-Meteo path is a docstring **[R]**
`src/prc/weather.py:7-8`.
(e) Not blocked.
→ **actionable now.**

### C40. Per-airport coverage-parity gate before any external feature ships
Slots: **22**.
(a) Measure a candidate feature's non-null **and non-zero** share on the scored period against
the matched period of the training year, **per airport**, and reject on a >20-point drop.
(b) kind-mango's **−13.14 s** loss was exactly this: `ec_*` went **100 % → 0 % non-null at LIRF
in July 2026**; OPDI erratic at 8 of 10 airports in both years. The mechanism is LightGBM's
NaN→0 convention on features that had *no* missing values in training, routing every serve-time
row to a lowest-value branch. Median imputation made it four times worse (v25, **+58 s**)
**[S]** (`22-repo-kind-mango.md` 4). Their `src/check_coverage.py` is ~90 lines and portable
**[S]**.
(c) None. Slot 22 is explicit that this is "a *deployment* failure, not an information failure"
and the rule is "gate on coverage parity", not "don't add external data" **[S]**.
(d) **Partially, and this is the closest thing we have to an existing win.**
`feature_parity_report` computes per-feature null-rate on the training side vs the target side
and `predict_ranking` **raises** if any gap exceeds `PARITY_GAP_LIMIT = 0.20` **[R]**
`src/prc/train.py:feature_parity_report`, `src/prc/predict.py:18,30-33`. What is missing is
(i) the per-**airport** breakdown — a 100 %→0 % collapse at one airport is ~10 % globally and
passes our 0.20 gate — and (ii) the non-**zero** share, which catches a feature that is present
but degenerate.
(e) Not blocked.
→ **actionable now.** The per-airport axis is the load-bearing half and we do not have it.

### C41. Join weather at pushback time rather than take-off time
Slots: **22**.
(a) One-line change of the join anchor.
(b) kind-mango's L7: joining at pushback failed their gate by **7 MSE out of 300** while
improving **all four served classes**. Slot 22's reading: "that is a pass, not a fail, by any
reasonable reading" **[X]** — and it is therefore untested public ground **[S]** (`22-…md` 8).
(c) Their gate said fail.
(d) **Not done.** We join on `MVT_TIME_UTC_mvt` truncated to the hour, plus the previous hour
**[R]** `src/prc/weather.py:join_weather`. Note the physical argument: the weather that matters
for a taxi-out is the weather during the taxi, which starts at off-block, and we already carry
`AOBT_3` on 98.5 % of rows.
(e) Not blocked.
→ **actionable now.** Cheapest item in P8.

### Blocked / refuted in P8
- **NASA's forecast-lead-binned and forecast-gradient weather** (slot 05 item 11) — **blocked**:
  our source is observations only, so only the rolling min/mean/max transfers **[S]**; and
  NASA's own 4th place rejected multi-hour forecasts in favour of the nowcast **[S]**.
- **Meteostat** — **refuted**: every European station stops 2026-03-29/30, null on 56 % of
  scored rows, CC BY-NC **[S]** (`16-…md` 6).
- **ERA5 / Open-Meteo reanalysis grids as the weather instrument** — **refuted**: 485.5 vs
  448.8, both months worse, "grid too coarse" **[S]** (`24-…md` 7).
- **EUROCONTROL daily airport files** (slot 16 item 3) — licence is **NC + no-modification**,
  not an open-source licence **[S]**; and no one has reported a gain. → **refuted on
  eligibility grounds** for anything that ships.
- **Raw OpenSky** — **refuted**: access by application, non-commercial, fails "openly
  accessible/usable" on its face **[S]** (`16-…md` 7).
- **OPDI flight-events as a taxi-out measurement** — **refuted**: 232 of ~30,600 departures
  usable (0.8 %), `entry-deicing_pad` fires once in ten days across all ten airports **[S]**
  (`16-…md` 5).
- **OPDI `icao_turn` turnaround gap** (slot 06 item 6, slot 16 item 4) — the only quantified
  external gain in the 2026 record, **≈ −3.4 s** **[S]** — but the join rate is 85.2 % in Jan
  2025, **58.9 % in Jan 2026** and 91.2 % in Jul 2026 **[S]**, and OpenAir attribute their v26
  loss of 2.60 s to 2026 callsign formatting **[S]**. → **actionable but gated on C40**, and
  slot 16 ranks it 4th for exactly this reason. Classified **blocked** (on C40).

---

## P9 — long-horizon rolling duration features (2 h–30 h, arrival side)

### C42. Extend `arr_taxi_in_mean` to 3 h and 30 h
Slots: **05**, **18**, **17**.
(a) Two more windows on a column we already compute.
(b) NASA 4th place's `arr_taxi_30hr` was a **top-5 feature**, and their two most important
non-identity features used a **30-hour** window **[S]** (`05-…md` 6,7). Availability verified:
ranking ARR `TAXITIME_SEC_mvt` null-rate **0.0** **[S, verified against our own
`data/raw/ranking.parquet`]** (`05-…md` 7).
(c) OpenAir measured recent-taxi-in as "weak / unstable (EHAM −0.28, others ~0.1)" **[S]**
(`17-…md` #4).
(d) **Not done.** `WINDOWS_MIN = (5, 10, 15, 30, 60)` and the duration means run at 30/60 min
only **[R]** `src/prc/features.py:32,213-219`.
(e) Not blocked.
→ **actionable now.** Slot 05's argument is the sharp one: "The 30-hour horizon measures 'what
kind of day is this airport having', which is a different quantity from the 30-minute horizon's
'is there a queue right now'; our six rejections only rule out the latter" **[X]**.

### C43. Extend `nbr_mvt_minus_aobt_mean` (the departure-side surrogate) to 3 h and 30 h
Slots: **05**.
(a) Same change on the departure side. The true `dep_taxi_30hr` is **not** available (ranking
DEP `TAXITIME_SEC_mvt` null-rate **1.0** **[S, verified]**), and the computable surrogate is
`mean(MVT − AOBT_3)` over recent departures — which is exactly our
`nbr_mvt_minus_aobt_mean_{30,60}m` **[S]** (`05-…md` 7).
(b) As C42.
(c) As C42.
(d) **Not done** at 3 h / 30 h **[R]** `features.py:207-213`.
(e) Not blocked.
→ **actionable now.**

### C44. `dis_state_30m` + `dis_frac20_30m` + `dis_state_vs_hour` (peer-lateness disruption state)
Slots: **06**, **17**, **21**. Partial disagreement on size.
(a) Mean `AOBT − EOBT` of *other* matched departures whose **AOBT** falls in
`(MVT_i − 30 min, MVT_i]`, self subtracted arithmetically, **NaN (not 0) when empty**; plus the
fraction with lateness > 1,200 s; plus the deviation from the train-split airport×hour mean as a
*difference*.
(b) OpenAir C14: matched RMSE **256.46 → 253.82** on Jan+Jul, **230.04 → 226.95** on December,
>30 min bucket 821 → 807, mean residual in the top disruption octile +60 → −3.5 s, kept after a
pre-registered test **[S]** (`17-…md` #1, `06-…md` 8). merry-quicksand's "congestion witness"
is the same family and shipped at **+12/12 months, board −1,775 MSE** **[S]** (`21-…md` §6.3).
(c) **Disagreement on size.** OpenAir's own verdict caps it: "a real but small (~1 % matched
RMSE)", ">60 min Jan+Jul not improved", "LFPG ~0", "**Do not sweep extra windows**" **[S]**.
Slot 17 converts: 1 % of ~202 s matched RMSE is **~2 s**, comfortably above our 0.25 s floor
**[X]**. So slot 06 reports it as ~1 % of matched RMSE and slot 17 as ≈2 s on our scale; those
are the same measurement, differently expressed, but slot 17 is the one making the claim that
it clears our instrument.
(d) **Partially, and the difference matters.** We have `nbr_aobt_delay_mean_{30,60}m` — mean of
`AOBT_3 − EOBT_1` over other departures — but the window is keyed on the neighbours'
**take-off** times with a 60 s lag, not on their **AOBT** **[R]** `features.py:205-212`
(`_window_mean(t, dep_times, aobt_delay, …)` where `dep_times` is `MVT_TIME`). We have neither
`dis_frac20` nor the airport×hour deviation.
(e) Not blocked.
→ **actionable now.** Slot 17's closing judgement: "If exactly one slot is spent here, spend it
on #1… the only candidate in this catalogue with a pre-registered positive result on our
dataset, the only one with a mechanism aligned to what our model actually predicts" **[X]**.

### C45. Ground-occupancy state counts ("currently on the ground"), not flows
Slots: **05**, **08**.
(a) `count(ARR rows with MVT_TIME ≤ t < BLOCK_TIME)` and `count(DEP rows with AOBT_3 ≤ t <
MVT_TIME)` at the anchor.
(b) NASA 4th place's `deps_taxiing`/`arrs_taxiing` were part of the feature set whose top-5 at
KMEM was dominated by ground-state features **[S]** (`05-…md` 5). Availability verified: ARR
`BLOCK_TIME_UTC_mvt` null-rate **0.0**; DEP `AOBT_3_flt` null-rate **0.0153** **[S, verified]**.
javidmardanov compute exactly this with correct self-exclusion **[S]** (`08-…md` 10).
(c) This is a **count** family, and slot 17's central judgement is that every congestion *count*
family has a strong negative and no clean positive anywhere **[S]**; our own six count
rejections are all CIs straddling zero against a noise column **[R]** `docs/features_log.md`.
OpenAir C3: adding queue on top of AOBT + (AOBT−EOBT) + geo_mean moves Jan+Jul matched
**303.53 → 303.56** **[S]**.
(d) Not done — our context features are flow counters only, never an occupancy count **[R]**
`features.py:context_features`.
(e) Not blocked.
→ **unmeasurable** on the prior: the sourced effect is inside our 0.25 s resolution and the
count family has a unanimous negative. Listed so it is not rediscovered.

### C46. Disjoint 5-minute slices alongside the nested windows
Slots: **05**.
(a) `rate_[0,5], [5,10], …` in addition to `[0,30],[0,15],[0,10]`.
(b) MIT airport-config 2nd place built both **[S]** (`05-…md` 9). Slot 05 calls it "the cheapest
untested idea in this slot" **[X]** — a one-line change to `_counts_in_windows`.
(c) Count family; see C45.
(d) Not done; our windows are purely nested **[R]** `features.py:_counts_in_windows`.
(e) Not blocked.
→ **unmeasurable** (count family, sub-resolution).

### C47. Departure demand in a window centred on the flight's own planned time
Slots: **05**.
(a) `exp_deps_15min` = count of flights whose ETD lies in `[T−15, T+15]` where `T` is *this*
flight's ETD — the **2nd-most-important feature at KMEM** **[S]** (`05-…md` 8).
(b) As above.
(c) We tested `bank_position` (rank within the bank) and rejected it at −0.19 [−0.41, +0.07]
**[R]** `docs/features_log.md` #2 — but never the symmetric density count **[S]**.
(d) Not done **[R]**.
(e) Not blocked.
→ **unmeasurable** (count family, sub-resolution) — but it is the one count variant with a
named top-2 importance elsewhere and a clearly distinct construction from what we rejected.

### C48. Per-runway-group congestion count at the airports where the airport-wide count is noise
Slots: **24**.
(a) fenil264's EDA: airport-count congestion has ρ ≈ 0 at **EHAM (0.007), LEMD (0.005), EDDM
(0.032)** — all independent-parallel-runway airports — while being the top feature at
EGLL/LTFM/LFPG/LIRF **[S]** (`24-…md` 4). "If we use a single airport-wide congestion count we
are feeding three airports pure noise; a per-runway-group count is the fix they identify" **[S]**.
(b) As above.
(c) Count family; see C45.
(d) **Partially.** We have `rwy_tko_prev_15m` / `rwy_tko_prev_60m` and same-runway headways
**[R]** `features.py:context_features`, so the per-runway count exists — what does not exist is
dropping or down-weighting the airport-wide count at the three ρ≈0 airports.
(e) Not blocked.
→ **unmeasurable** individually; retained as a diagnostic that may explain why our count
experiments all landed at zero.

### Refuted in P9
- **Congestion counts as a family** — **refuted** (`../SUMMARY.md` §F): every count family has a
  strong negative and no clean positive anywhere; NASA 2nd place (0.055 behind 1st) shipped zero
  congestion features; on our own data `dep_prev_1m` correlates 0.0086 **[S]**.
- **Queue position / rank** — **refuted unanimously**: NASA 2nd built it and left a comment that
  it "ended up not being used"; our `queue_ahead` −0.16 and `bank_position` −0.19 **[S]**
  (`17-…md` #9). "No source anywhere reports a positive rank result. Do not build."
- **Rolling actual taxi-out of neighbouring departures** — **refuted as a leak**: all 344,841
  ranking DEP rows have `TAXITIME_SEC_mvt` null **[S]** (`17-…md` #11). This is the feature NASA
  4th place ranked 3rd most important; it was legal there and is not here. Our
  `mask_departures` already enforces this **[R]** `features.py:mask_departures`.
- **Unimpeded / reference taxi time as a further congestion feature** — slot 17 #12 says do not
  build, because the within-group RMSE ceiling for `airport × stand × runway × hour × type` is
  268.1 while the model already sits at ~202 on matched rows **[S, our own numbers]**. See G7
  for the disagreement with slots 19/13/09.

---

## P10 — CatBoost as a rival single model

### C49. CatBoost on the identical features and folds
Slots: **20**, **11**.
(a) A second family, not a blend member yet.
(b) On the closest analogue — PRC 2024 ATOW, RMSE — CatBoost beat the same team's Optuna-tuned
LightGBM by **11.6 %** (2,215.53 vs 2,506.43) and took **0.865 of the blend weight** while
LightGBM took ~5e-22; removing CatBoost cost **+33.90 %** while removing XGBoost cost **+0.16 %**
**[S]** (`20-…md` 2, `11-ensembling-patterns.md` step 2). TabArena default Elo: CatBoost 1432 vs
LightGBM 1206; TabRepo tuned 0.534 vs 0.566 **[S]**. It is the family the top three public PRC
2026 competitors all carry **[S]**.
(c) Slot 20 does not hide it: on Sigrist's 8 high-cardinality **regression** datasets CatBoost
ranked *behind* LightGBM-native-categorical (2.86 vs 2.71; rel. diff. 44.5 % vs 17.4 %) **[S]**
arXiv:2307.02071, and ML Contests 2025 records a winner for whom CatBoost underperformed **[S]**.
Our data is high-cardinality regression (STAND 1,899 levels, ADES 1,568, operator 676 **[R]**).
(d) **Not done.** `catboost>=1.2.7` is a declared dependency **[R]** `pyproject.toml` and
appears in **no** source file (`grep -rn catboost src/ tests/ configs/` matches only the egg-info
metadata **[R]**).
(e) Not blocked.
→ **actionable now.** Operational notes attached to it: use `boosting_type='Plain'` at 2.08M rows
(ordered boosting's benefit is documented only below ~40k rows) **[S]** (`20-…md` 2), and set
`has_time=True` so CatBoost does not permute rows when building ordered target statistics —
NASA 1st place set it in every model **[S]** (`05-…md` 14).

### C50. A blend harness: constrained weights, ablate-before-ship, greedy selection if ≥4 members
Slots: **11** (steps 1, 3, 5), **03** (7, 8), **23** (14), **06** (7).
(a) `minimize(rmse, w0, bounds=[(0,1)], sum(w)=1)` on the intersection of non-null OOF rows,
prune `w < 0.01`, refit the blend with each member removed and print the RMSE change.
(b) PRC 2024 rank 2's five-model stack resolved to **0.865 CatBoost + 0.134 XGBoost**; LightGBM,
MLP and SAINT each got numerically zero weight and removing them changed RMSE by **0.00 %**
**[S]** (`03-…md` 7). OpenAir's NNLS: CatBoost residual 0.456 + per-airport LightGBM residual
experts 0.491 + XGBoost 0.053, pooled and direct-target both weight 0 **[S]** (`06-…md` 7).
Slot 03's framing is the useful one: this is **a decision rule, not a prediction** — print the
weights before investing in a five-model blend.
(c) **The ceiling is sourced and small.** Slot 11: the blend mechanisms buy **0.3–1.8 % over the
best single model = 0.9–4.5 s for us** **[X]**; PRC 2025 rank 1 was a single LightGBM and beat
the 3-library blend by **11.16**; PRC 2024 rank 2's ablation puts LGBM, MLP and SAINT at
+0.00 % **[S]** (`../SUMMARY.md` §F). Slot 09 measured that averaging in *any* partner **costs**
once the base is denoised (v26 +4.2, v28 +5.0, v31 +1.2) **[S]**.
(d) Not done **[R]**.
(e) **Blocked on C9** (no fold label ⇒ no shared partition ⇒ no honest meta-learner) and on
**C49** (no second member).
→ **blocked.**

### C51. XGBoost as a third member
Slots: **20** (7), **11** (step 4).
(b) Weight 0.134 and +0.16 % ablation cost in PRC 2024; weight 0.160 in PRC 2025; 5.3 % in
OpenAir's NNLS **[S]**.
(c) satam2: XGBoost earned **zero** weight **[S]** (`23-…md` 14). sqcm9896: XGBoost 481.8 vs
L2 LightGBM 448.8 **[S]** (`24-…md` 7).
(d) Not done; `xgboost` is not even a declared dependency **[R]** `pyproject.toml`.
(e) Blocked on C49 and C50.
→ **blocked.**

### Refuted in P10
- **Cross-library blending as the way to close our gap** — **refuted** (`../SUMMARY.md` §F).
- **Neural tabular models / SAINT / MLP / TabPFN** — **refuted**: MLP and SAINT scored ~75 %
  worse standalone with blend weights of 1.24e-23 and 1.63e-23 and 0.00 % ablation impact;
  NN-with-embeddings was the worst of seven methods (189 % vs best) on high-cardinality
  regression; TabPFN-2.5's ceiling is 50,000 rows against our 2.08M **[S]** (`20-…md` 10).
  A 32-member neural ensemble was **−0.39 s** for satam2 **[S]**.
- **TimesFM** — **refuted on evidence**: the repo that shipped it provides none that it helped,
  and its own context ablation found none; non-commercial weight licence **[S]** (`08-…md` 12).
- **Sequence models (TCN, event-graph transformer)** — **refuted**: the TCN was actually
  submitted and lost (317.47 vs 316.97); the transformer gave −0.15 s **[S]** (`06-…md`).
- **Pseudo-labelling the 2026 rows** — **refuted for us**: adversarial AUC 0.807 with only 5.9 %
  of training rows above the ranking set's median means pseudo-labels would be most confidently
  wrong exactly on the 2026 rows that differ most **[S-repo]** (`20-…md` 11) — and those repo
  numbers are ours **[R]** `src/prc/adversarial.py:20,25`.

---

# Part II — the gaps: actionable candidates outside all ten themes

**This is the point of this slot.** Each item below is a distinct candidate that no planner owns.
Ordered by my estimate of expected value; the ordering is **[X]**, every input is cited.

---

### G1. Delete the ten monotone `_slog` duplicate columns
Slots: **03** (9).
(a) Drop `add_signed_log`'s ten `sign(x)·log1p(|x|)` copies from the model's feature set (or
gate them on `use_linear_tree`).
(b) Alligier's A3/A4/A5: **monotone transforms of a feature are no-ops for tree models, and
duplicated monotone copies actively dilute split selection and corrupt importance scores**
**[S]** (`03-…md` 9). Slot 03 calls it "a pure-win cleanup".
(c) None. The columns were introduced for a reason that does not currently apply — see (d).
(d) **We do this, and the justification does not hold on the shipping config.**
`add_signed_log(df, cols=WIDE_RANGE_COLS)` creates ten strictly-monotone copies of
`mvt_minus_aobt3`, `mvt_minus_sched`, `mvt_minus_eobt1`, `mvt_minus_lobt`, `mvt_minus_iobt`,
`lobt_minus_eobt1`, `aobt3_minus_eobt1`, `aobt3_minus_sched`, `rwy_headway_prev_s`,
`rwy_headway_next_s` **[R]** `src/prc/features.py:227-247`, and they are unconditionally added
to the feature frame **[R]** `features.py:build_departure_features`. The docstring says they
exist "so leaf-linear models cannot extrapolate wildly" — but `use_linear_tree` defaults to
`False` **[R]** `train.py:TrainConfig` and `configs/main.yaml` does not set it **[R]**, so in
production they are ten pure monotone duplicates of columns the model already has, including a
duplicate of `PROXY` itself.
(e) Not blocked.
→ **actionable now.** It compounds with our own measured instrument defect: `feature_fraction:
0.8` **[R]** means every added column shifts the score by ±0.5 s regardless of content — our
own `noise_control` experiment measured exactly that **[R]** `docs/features_log.md`. Ten
content-free columns at 0.8 feature fraction is the one change on this list that *removes*
columns.

### G2. Group-relative deviation features on the clock columns (z-score / difference within airport×hour)
Slots: **03** (10, 20), **04** (7), **19** (3).
(a) Express the load-bearing clock columns as a difference from, or z-score within, their
airport-and-hour (or airport×runway×stand) cell, rather than only in absolute seconds.
(b) Slot 03 calls this **"the strongest structural idea in the 2024 corpus that transfers
intact"**: the PRC 2024 winner scaled mass features per aircraft type and rank 5 independently
scaled every trajectory feature by its per-aircraft-type mean — "scaling only helps trees when it
differs *by group*" **[S]** (`03-…md` 10). Slot 04: PRC 2025 rank 7 built
`altitude_diff_type = altitude_mean − groupby(typecode).mean()` and `speed_diff_phase`, and these
were "the distinguishing feature of a top-7 solution" and are **label-free** **[S]** (`04-…md` 7).
Slot 19 names the specific column for us: `mvt_minus_aobt3`'s within-airport z-score, as a LIRF
regime indicator **[S]**. Slot 03 item 20 supplies the mechanism (Alligier A16): greedy tree
growth may never select the first of a required pair of splits, so a context-relative quantity
must be built, not hoped for.
(c) Our one attempt in this family, `congestion_vs_usual_30m`, was rejected at +0.09
[−0.16, +0.31] **[R]** `docs/features_log.md` #6 — but that is a ratio of *counts*, and the
count family is refuted for other reasons. No slot tested the group-relative form of the *clock*
columns.
(d) **Not done.** Every clock column is absolute seconds **[R]**
`features.py:clock_and_static_features`. The only group-relative column is the rejected count
ratio.
(e) Not blocked.
→ **actionable now.** This is the strongest gap-item by sourced pedigree, and it is exactly the
axis our own diagnosis points at: model A predicts `AOBT_3 − BLOCK_TIME`, a reporting offset
whose median runs −301 s to +56 s across operators **[R]** `docs/features_log.md`.

### G3. Tail-copy clipped features to defeat histogram border quantisation
Slots: **08** (8).
(a) Three columns: `clip(aobt_delta, upper=300)`, `clip(aobt_delta, lower=2200)`,
`clip(eobt_delta, lower=3600)`.
(b) javidmardanov's motivation is measured, not theoretical: with 128 borders the splits landed
at 304.5 / 2218.5 / 3662.5 and could not resolve the tails **[S]** (`08-…md` 8,
`scripts/train.py:188-193`, `docs/research_program.md:69-72`). Paired with a gated delta their v2
tail delta was 265.81 vs 266.05 and the V3 version got α=0.5441 **[S]**. "Three lines; try it
before touching hyperparameters" **[S]**.
(c) None. Note LightGBM bins at 255 by default rather than CatBoost's 128, so the effect should
be smaller for us **[X]**.
(d) Not done **[R]**.
(e) Not blocked.
→ **actionable now.**

### G4. Tweedie / Poisson / Gamma objective sweep (log-link, mean-preserving)
Slots: **02** (2), **13** (4).
(a) `objective: tweedie` with `tweedie_variance_power ∈ {1.1, 1.3, 1.5}`, plus `poisson` and
`gamma`, each with `metric: l2` for early stopping.
(b) Slot 13 gives the correct reason and is careful to disown the wrong one: a log-link GLM
models `E(Y|X) = exp(xᵀβ)` directly and so has **no retransformation bias**, unlike a
log-transformed target; and Gneiting's framework says Tweedie deviance is a Bregman divergence
and therefore **is consistent for the mean** — "which is the real reason it is worth a fit, not
the Amazon result" **[S]** (`13-target-transforms.md` 4). Slot 02 has the empirical precedent:
Sendy 1st and 3rd both report a positive-support count-like loss beating RMSE loss *on an
RMSE-scored duration-in-seconds problem* **[S]** (`02-motoki-method.md` 2). Taxi-out is positive,
right-skewed, variance growing with mean — the Tweedie setting.
(c) Slot 13 itself flags that the Amazon paper's metric is WMAPE/WBias, not RMSE, so that part
does not transfer **[S]**. And the surrounding loss family is refuted repeatedly: Huber 668.4 vs
L2 448.8 and asymmetric α=0.25 at 477.6 **[S]** (`24-…md` 7); satam2 got degenerate 1–3-tree
models with CatBoost Huber **[S]** (`23-…md`); OpenAir: "any non-L2 residual objective: Huber,
tail-weighted L2, two-stage mixture, quantile/pinball — all four trade bulk for tail at constant
RMSE" **[S]** (`06-…md`). **Note carefully: none of those four is Tweedie, Poisson or Gamma,
and all four are non-mean-eliciting, which is precisely the property Tweedie does not share.**
Our own repo has already measured robust losses and quantile α=0.5 as harmful **[R]**
`docs/investigation_2026-09-16/results/robust-loss-harmful.md`.
(d) **Not done.** `objective: regression` everywhere **[R]** `train.py:TrainConfig.params`.
(e) Not blocked. One sweep of five fits.
→ **actionable now.** Recorded as a gap because P7/P10 cover *capacity* and *family*, not
*objective*, and the surrounding evidence is unusually easy to misread as already-refuted.

### G5. Free affine recalibration diagnostic: the (a, c) minimising OOF RMSE of `a·ŷ + c`
Slots: **13** (6).
(a) Zero fits. If the optimum is materially away from (1, 0) — say |c| > 5 s — the model is
miscalibrated in the mean and there is a free affine fix; if it is not, no quantile-level or
expectile tuning can help either, "because a global shift is all such tuning can deliver"
**[S]** (`13-…md` 6, grounded in Gneiting: expectile regression at τ=0.5 *is* MSE regression).
(b) As above.
(c) None — it is a measurement, not a change.
(d) Not done. `cv.metrics` reports `bias = mean(p − y)` per fold **[R]**, which is the `c`
question at `a = 1` but is never acted on and is not reported per airport.
(e) Not blocked (needs an OOF file, which exists).
→ **actionable now.** Costs one script run and permanently closes the
quantile/expectile/isotonic question.

### G6. Label-free frequency encodings and categorical crossings, over train ∪ 2026
Slots: **20** (3), **23** (11).
(a) Group-by counts for the high-cardinality categoricals, and explicit crossed tokens
(`airport_stand`, `airport_runway`), in two variants: train-only and train+2026.
(b) Frequency encoding attributed to IEEE-CIS Fraud; CatBoost ships counter CTRs natively;
NVIDIA playbook #3 credits pairwise categorical crossing with two first places **[S]**
(`20-…md` 3). **No target is touched, so there is no leakage risk** — the only decision is
whether to include the 2026 rows, and the IEEE-CIS convention was to exclude them, so measure
both **[S]**. satam2's tokenisation detail is the cheap correctness half: `"s" + len(v) + ":" + v`,
missing → `"m:"`, which prevents `"A1"+"0"` colliding with `"A"+"10"` in crossed keys **[S]**
(`23-…md` 11).
(c) None sourced. Our own instrument caveat applies (±0.5 s per added column at
`feature_fraction 0.8` **[R]**), so this should be measured against a noise-column control and
ideally after G1 frees ten column slots.
(d) **Not done.** We have congestion counts but **no category frequency features and no
categorical crossings** — confirmed by reading `features.py` end to end **[R]**. Note we *do*
already cross airport × stand × runway inside `reference_table`, but as a label-derived P10
encoding, not as a label-free count **[R]** `train.py:reference_table`.
(e) Not blocked.
→ **actionable now.** Slot 20 attaches a second read for free: whether the train+2026 variant
moves our adversarial AUC of 0.807 **[R]** `adversarial.py` — independent evidence about whether
it transports information across the year.

### G7. Unimpeded reference time built to the *current* EUROCONTROL spec, with the level as a categorical
Slots: **19** (1), **13** (2), **09** (B8) — against: **17** (#12). **Disagreement.**
(a) Rebuild `ref_p10` to Edition 01.00's actual rule and emit the fallback **level** as a
categorical alongside the value and the sample size.
(b) **For:** slot 19 finds this is the most-repeated construction in the taxi-out literature —
one of Herrema 2018's LFPG top-10, one of the four features Relief selected in Zbakh 2024, the
decomposition base in Yuan 2024 and the Shenzhen stacking paper **[S]**. Slot 09: it is
sergiuv11 Repo B's **strongest single feature at 18.8 % of gain**, and their recipe emits "the
**level** as a categorical and the **sample size** as a numeric alongside it", with the fallback
chain `(apt,stand,rwy) → (apt,stand) → (apt,rwy) → (apt)` **[S]**. Slot 09 is also explicit that
it must be a **feature, not a residual offset** — the residual parameterisation measured inside
the noise floor (377.84 vs 379.73) and was retracted as inconclusive **[S]**.
(c) **Against:** slot 17 #12 — "our repo has already measured the ceiling of this axis directly":
within-group RMSE for `airport × stand × runway` is **307.5** and for
`airport × stand × runway × hour × type` is **268.1**, while the model already sits at **~202**
on matched rows. "A P10 encoding cannot beat a group mean that the model has already surpassed.
**Do not build.**" **[S, our own numbers]** (`docs/features_log.md` **[R]**).
(d) **Partially, and to the wrong spec.** `reference_table` computes P10/P20/median/n per
`(ADEP, STAND, RUNWAY)` filtered to `0 < y < 7200` with `n ≥ 30`, and `ref_p10_fb` coalesces
`ref_p10 → arh_med → apt_med` **[R]** `train.py:200-241`. The current EUROCONTROL rule
(`../SUMMARY.md` §B, slot 19) is different in four ways: **≥10 flights at or below the P10**
(not n ≥ 30 overall), **rolling 12 months**, **stand individually**, **06:00–22:00 local**. We
also emit `ref_missing` (a boolean) but not a fallback-**level** categorical **[R]**.
(e) Not blocked.
→ **actionable now**, with the disagreement recorded: slots 19/13/09 have the literature and a
competitor's 18.8 %-of-gain measurement; slot 17 has our own within-group ceiling arithmetic
saying the axis is saturated. The cheap part that both sides permit is the **fallback-level
categorical and the corrected `n` rule** — those change what the model is told about *provenance*,
which is not what the ceiling arithmetic bounds.

### G8. Stand→runway taxi distance as a continuous feature
Slots: **19** (2) — blocked by **06**.
(a) Give the model a distance so it can extrapolate to rare stands instead of memorising
1,899 × 53 interactions.
(b) Wang 2021 puts distance in the top 3 at all three airports (VHHH/EGCC/LSZH); Lee 2019
concludes distance is the dominant ramp factor; Tang 2025's largest gain is route-derived **[S]**
(`19-atm-literature-sota.md` 2).
(c) **Blocking:** OpenAir attempted stand-coordinate / taxiway-graph reconstruction and failed
three times (E37, E59, E60): "no reproducible open crosswalk exists for the challenge STAND IDs"
**[S]** (`06-…md`). Slot 16's OPDI crumb is the only partial route in: `info.osm_ref` gives a
stand reference, but with useful density at **LSZH and EDDF only** **[S/X]**.
(d) Not done **[R]**.
(e) **Blocked** on a stand-coordinate crosswalk that three independent attempts could not build.
→ **blocked.** Slot 19's own fallback is that the `(stand, runway)` P10 *is* the distance proxy —
i.e. G7.

### G9. Operator sign-consistency test and an unseen/rare-level back-off
Slots: **24** (6).
(a) Two things: (i) run the `airline_consistency.py` test — per `(airport, operator)` deviation
from the airport median, cells with n ≥ 100, operators at ≥3 airports, then a **sign-consistency
fraction** — against every target encoding we fit; (ii) give rare and unseen categorical levels a
`RARE(<20) → parent back-off` plus a garbage bucket.
(b) kagervig's result: the operator effect is real and consistent (ε² 0.11, fastest ~−4 min,
slowest ~+6 min), **and 11 operators appear in ranking with no training history**; sqcm9896 count
**29 novel STAND, 62 ADES, 8 types, 29 operators** in ranking **[S]** (`24-…md` 6). Slot 24:
"worth running against every OOF target encoding we fit, and worth checking how we handle unseen
operators".
(c) Our adversarial run found only **0.18 %** of ranking rows carry a category absent from
training — "a shift in proportions, not in vocabulary" **[R]** `adversarial.py:25`. So the
unseen-level half is small; the sign-consistency half is a *validity test* and is unaffected.
(d) **Not done.** Categoricals are cast `fill_null("__NA__") → Categorical → to_physical()` on
the stacked train+target frame **[R]** `train.py:_matrix`, so codes are consistent across the two
sides, but there is no rare-level collapse and no parent back-off, and `op_offset_table` uses a
flat `min_count = 20` cut with no shrinkage toward a parent **[R]** `train.py:operator_offset_table`.
(e) Not blocked.
→ **actionable now.** The sign-consistency test is the part with independent value: it is a test
that a high-cardinality categorical's apparent effect is not an artefact of its distribution
across a stronger categorical — which is exactly the failure mode that killed our
`op_offset_med` experiment at −0.17 [−0.40, +0.05] **[R]** `docs/features_log.md` #7.

### G10. Ordered-neighbour NM clock channels (peer proxies, following-context, `own_minus_mean`)
Slots: **23** (9), **07** (4, 5), **17** (#2, #3).
(a) **Not** a rolling window mean: per-neighbour ordered channels. satam2's cheap first cut is
the 4 preceding departures × {`takeoff_minus_AOBT_3`, `age_sec`, `same_runway`} = 12 columns;
Phoenix-Ops' version is 30 strictly-past and 30 strictly-following NM peer-proxy columns keyed
airport / airport+runway / airport+stand within one UTC month, including `own_minus_mean` —
this flight's own proxy minus the local peer mean.
(b) Two officially-measured leaderboard gains: Phoenix-Ops v9 **−0.8819 s official** (strictly
past) and v12 **−1.4181 s official** (following context) **[S]** (`07-…md` 4, 5;
`17-…md` #2, #3). satam2 measured that **the 24 peer departure-clock channels alone carry 98.7 %**
of their 112-column block's gain **[S]** (`23-…md` 9). Slot 07 on `own_minus_mean`: "the feature
that directly encodes 'is this airport's AOBT_3 bias unusual right now'… of all their additions
this is the least hindsight-dependent" **[S]**.
(c) **Model-dependence, measured:** the identical 30 following-context features gained
**0.283 s through a 679-tree LightGBM** (below Phoenix's own 0.5 s gate, "remains rejected") and
**1.4 s through a 4,999-tree CatBoost** **[S]** (`17-…md` #2). Slot 17: "Our model B is LightGBM;
that is the closer of the two controls, and it is the one that failed." And OpenAir C2: the same
idea gives only **318.0 → 316.2** once `AOBT − EOBT` is already in the model — "real but
second-order" — and we are in that regime **[S]** (`17-…md` #3).
(d) **Not done.** We have window *means* of `aobt_delay` and `mvt_minus_aobt` over neighbours
**[R]** `features.py:205-212`, which is the aggregated form. There is no per-neighbour ordered
channel, no `own_minus_mean`, no month scoping, and no stand or runway keying of these means.
(e) Not blocked; **interacts with C35** — slot 17's caveat says this family may only be
measurable under the deep model, so testing it *before* C35 risks a false rejection.
→ **actionable now**, but sequenced after C35. Recorded as a gap because P9 is long-horizon
*arrival-side duration* features and this is short-horizon *departure-side clock-offset*
context — two officially-measured gains totalling **−2.30 s** that no planner owns.

### G11. `MVT_ID_mvt` record-order peer features
Slots: **23** (12).
(a) Within (airport, month), sort by `MVT_ID_mvt`; for w ∈ {2, 8, 32} emit
`median(neighbour_time) − own_time` and `p90 − p10` of neighbour times. Six columns, ~20 lines.
(b) satam2: "not isolated in their log, but it is the kind of feature that targets exactly the
'off-block source disagreement' mechanism" **[S/X]** (`23-…md` 12).
(c) Not isolated in any log — the evidence is mechanistic, not measured **[X]**.
(d) Not done **[R]**.
(e) Not blocked.
→ **actionable now.** Flagged because the mechanism it targets is *precisely* what our model A
predicts (`AOBT_3 − BLOCK_TIME`, a reporting discrepancy between two recording systems **[R]**
`docs/features_log.md`), and record provenance is the one axis nothing else in this document
probes.

### G12. The unused schema columns
Slots: our own log, plus **08** (5).
(a) `CALLSIGN_flt`, `ADES_FILED_flt`, `ARVT_3_flt` are in the schema and have never been used
**[R]** `docs/features_log.md` "残っている未着手". Plus javidmardanov's
`flight_identity = flight_prefix + "_" + flight_number` and
`flight_destination = that + "_" + ADES_mvt` as categoricals.
(b) javidmardanov: restricted to January/November, `flight_identity` gave −0.95 s on the November
forward check from 48 rows, and V5→V6 officially moved **−0.5216 s** **[S]** (`08-…md` 5).
(c) Slot 08 caveats its own result honestly: "a 97-row, post-hoc, subgroup-selected correction on
repeatedly-tuned months. Reproduce the mechanism (recurring-flight identity helps predict
schedule-identity records), not the specific winter gate" **[S]**. All-season use **hurt July**
and was rejected **[S]**. And OpenAir refute the neighbouring idea: callsign or flight-number
residual priors, both splits worse (E42) **[S]** (`06-…md`).
(d) Not done **[R]**.
(e) Not blocked.
→ **actionable now** for the schema columns; the flight-identity categorical is
**refuted** as a standalone but survives as an input to the LIRF fill classifier (G20), which is
where slot 08 actually uses it.

### G13. The arrival mirror as a labelled 2025→2026 drift gate
Slots: **22** (5), **15** (3 is the weaker external-series version).
(a) Train a taxi-in model on our training-year **arrivals**, score it on the scored-year
arrivals. The 344,693 arrival rows in the 2026 scoring file carry **real taxi-in labels**, so
2025→2026 transfer can be measured *with labels* before any upload.
(b) kind-mango built it: 2025 hold-out **202.40 → 2026 214.96, +6.21 %**, but **EHAM +128.9
against EGLL −40.9** — "a global calibration would be actively wrong at half the airports"; and
EHAM January 405.53 vs EHAM July 129.38 **[S]** (`22-…md` 5). Slot 22's verdict: they validated
it as a *measurement* instrument and never used it as a gate.
(c) Their one attempt to turn mirror-derived drift into **features** lost **+0.37 s** — use it as
a gate, not as a feature source **[S]**.
(d) **Not done. We have not measured drift at all.** The nearest thing is the adversarial AUC of
0.807, which measures *covariate* shift in the features and explicitly does not explain the
CV/leaderboard divergence **[R]** `adversarial.py:20-30`. The mirror measures *outcome* drift,
with labels, on the scored period.
(e) Not blocked — the arrival rows are already in `ranking.parquet` and we already read them for
`arr_taxi_in_mean` **[R]** `features.py:context_features`.
→ **actionable now.** My assessment **[X]**: of everything in this document this is the item
with the largest gap between its cost (one model fit on data we already load) and what it buys
(the only labelled read on 2026 we can obtain without a submission). Our entire measurement
problem is that CV and the board disagree; this is a third instrument.

### G14. Train only on rows you actually serve; fit the reference tables on matched rows only
Slots: **22** (8, MB3), **06** (9, E18-H).
(a) Two related hygiene changes: (i) if any head overwrites a row downstream, exclude that row
from the upstream model's training set; (ii) fit the stand×runway geometry tables on **matched**
training rows only, so unmatched rows' multi-hour labels cannot poison the geometry.
(b) kind-mango's MB3 was worth **−5.99 s live at zero offline price** **[S]** (`22-…md` 8).
OpenAir's E18-H bundle (both halves) moved overall **376.04 → 372.36** and matched
**253.82 → 250.98** **[S]** (`06-…md` 9).
(c) OpenAir never ablated the two halves separately (`status.md:934`) **[S]**.
(d) **Half-done, half a live defect.** (i) applies only once an overlay exists (C1, C29, C31).
(ii) is a defect today: `reference_table` is fitted on **all** labelled training rows filtered to
`0 < y < 7200` **[R]** `train.py:reference_table` — which excludes unmatched rows above 2 h but
keeps every unmatched row between 0 and 7,200 s, and those are exactly the schedule-fill rows
whose `y` is `MVT − SCHED` rather than a taxi duration. There is no `PROXY.is_not_null()` filter
on the encoding fit **[R]**.
(e) Not blocked for (ii).
→ **actionable now.** One `.filter()` call, with a sourced −3.68 s overall / −2.84 s matched
attached to the bundle it belongs to.

### G15. A real fallback chain in the prediction path: model → proxy → per-airport median
Slots: **05** (17), **06** (10).
(a) When a prediction is missing or a feature chain breaks, fall through to the next-best
*estimator that can still score the row*, rather than to a global constant.
(b) OpenAir measured the cost of getting this wrong directly: matched RMSE **379 → 398** **[S]**
(`06-…md` 10, "chain-don't-dump"). NASA 1st place shipped
model → ETD-minus-30 baseline → constant 30 **[S]** (`05-…md` 17).
(c) None.
(d) **Not done, and the current fallback is the global constant the sources warn against.**
`_assemble` fills any missing prediction with `nanmedian` of **all** training labels **[R]**
`train.py:_assemble`; the unmatched branch falls back to `ref_p10_fb` only when there are <200
labelled unmatched training rows **[R]** `train.py:predict_fold`. `ref_p10_fb` itself is already
a three-level coalesce **[R]** — so the machinery exists and the top-level fallback does not use
it. Our analogue of the sourced chain is: model → `mvt_minus_aobt3` → `ref_p10_fb` →
per-airport median → global median.
(e) Not blocked.
→ **actionable now**, cheap, and defect-shaped.

### G16. Backward feature elimination, not forward accumulation
Slots: **04** (16), **20** (6).
(a) Start from all engineered feature groups and successively remove what does not provide a
clear robust improvement, logging each deletion's OOF delta and gate outcome. The LOFO form runs
on the forward fold specifically and drops anything with negative mean importance and a CI
excluding zero.
(b) PRC 2025 **rank 1**: *"Starting from all engineered feature groups, we successively removed
or simplified features that did not provide a clear and robust improvement."* PRC 2025 **rank 11**
used forward `SequentialFeatureSelector` on 133 candidates and finished 11th **[S]**
(`04-…md` 16). LOFO "gives negative importance to features that hurt performance upon inclusion"
and caught seasonal OS-version features in Microsoft Malware under time-split validation **[S]**
(`20-…md` 6).
(c) Slot 20 does not hide the counter-evidence: NVIDIA's playbook omits feature selection
entirely and argues the opposite **[S]**. Cost: n_features full retrains at 2.08M rows is the
expensive item on slot 20's list **[S]**.
(d) **Not done, and our procedure is the losing one.** `docs/features_log.md` is a pure
forward-accumulation log: ten candidates added one at a time, ten rejections **[R]**. Nothing has
ever been *removed* and measured — the two ablation configs that exist
(`abl_no_encodings`, `abl_no_categoricals`) test whole blocks, not individual columns **[R]**
`configs/`. The nine encoding columns are noted in the code as "Present since the first commit
and never ablated" **[R]** `train.py:TrainConfig.use_encodings`.
(e) Not blocked; expensive.
→ **actionable now.** This is a change of *search direction*, not a candidate feature, and it is
the single structural difference between our experiment log and the PRC 2025 winner's. It also
composes with G1: removal is the one operation whose column-count side effect runs in our favour
**[R]** `docs/features_log.md`.

### G17. A `|delta|`-band y-target model on the matched tail, LIRF-excluded
Slots: **21** (§6.3.1).
(a) On matched rows where `|delta| = |BLOCK − AOBT_3|` is large, a model fitted on raw `y` beats
the delta-anchored model; blend the two by a `|delta_hat|` gate, excluding LIRF.
(b) merry-quicksand RESULT 19: the delta-anchored model is beaten by a y-target model on the rows
where `|delta| > 20 min` by **+9.99 s [+5.63, +14.35]**, on rows carrying **31 % of matched SSE**
— while losing 21 s on the 2–10 min body **[S]** (`21-…md` §6.3.1).
(c) **Every serve-time gate they tried failed because LIRF reverses the sign** **[S]**. Their
ex-LIRF arm (Amdt 26) was INCONCLUSIVE at +0.454 s ≈ 200 MSE — "small, but positive, and 4 of 5
clauses held with the negative control passing at 4.5× p99" **[S]**.
(d) Not done **[R]**.
(e) Not blocked — slot 21 notes we already have the model-A/model-B structure that makes the
LIRF-excluded version cheap for us **[X]**.
→ **actionable now.** Recorded as a gap because it is neither capacity (P7) nor a per-airport
specialist nor the unmatched lane: it is a *target-parameterisation* split inside the matched
lane, on 31 % of matched SSE.

### G18. Train a raw-target partner and blend the two parameterisations
Slots: **05** (1), **02** (4), **04** (8), **20** (5), **23** (2).
(a) Our matched model fits `y − proxy`. Fit an otherwise identical model on raw `y` and average.
(b) NASA 1st place averaged **two residual-vs-proxy models, one raw-target model and one global
model with equal weights** **[S]** (`05-…md` 1, `1st Place/Phase 1/submission/utilities.py:660-671`).
PRC 2025's merry-jacket blended **two models trained on two different target
parameterisations** (CatBoost on log1p(rate), XGBoost on raw quantity) — slot 04 calls this "the
most interesting ensemble in the field" and notes it is "a cheap source of genuine diversity —
more so than three libraries on one target" **[S]** (`04-…md` 8). Slot 02's Motoki claim is the
sharp one: the alternative parameterisations "may individually lose to (a) and still improve the
blend — so the pass criterion is the *blend*, not the single model" **[S]**.
(c) OpenAir's NNLS assigned their **direct-target LightGBM weight 0** **[S]** (`06-…md` 7) —
a direct negative on this exact member. And slot 09 measured that averaging in any partner costs
once the base is denoised **[S]**.
(d) Not done. The raw-target path exists in code as model B's target and as the `<200 rows`
degenerate branch, but no raw-`y` model is fitted on matched rows **[R]** `train.py:predict_fold`.
(e) Blocked on **C50** for the weighted form; the 50/50 average is free.
→ **actionable now** (50/50 form). Recorded as a gap because P7 is capacity-on-residuals and P10
is a rival *library*; this is a rival *target*, and slot 04 argues it is the better diversity
axis of the two.

### G19. A repair model keyed on the main model's own output
Slots: **05** (4).
(a) Pool every training row where model A predicts outside `[p0.1, p99.9]` of that airport's
training target, train one small shallow model on them, and use it *only* on those rows at
inference.
(b) NASA 2nd place pooled every training row where the main model predicted `< 0`, across all
airports, and trained one small shallow CatBoost (depth 4, lr 0.01) used only on those rows
**[S]** (`05-…md` 4, `2nd Place/Phase 1/src/run_training.py::train_negative_model`).
(c) Slot 05 is explicit that the related 3rd-place sign-classifier + median-offset form is an L1
artefact and should be skipped under RMSE **[S]** (`05-…md` 19).
(d) Not done. Our only routing is by data partition (matched/unmatched) **[R]**, never by the
model's own output.
(e) Not blocked.
→ **actionable now.** Slot 05: "the 2nd-place trigger is different and complementary" to our
A/B split **[S]**.

### G20. Use the LIRF matched-fill regime signal — 159k matched rows we are not training on
Slots: **08** (4), **21** (§6.1b), **19** (3), and our own `lirf-matched-fill.md`.
(a) Our LIRF handling is entirely on the **unmatched** lane. The same fill mechanism exists on
the **matched** lane and is unused: detect the "`BLOCK == SCHED`" population on matched rows and
route it with a soft mixture.
(b) javidmardanov: the identity population is `|y − (MVT − SCHED)| ≤ 6`, **18.08 % of LIRF
training rows**, and their soft-mixture treatment moved local **266.05 → 263.22** **[S]**
(`08-…md` 4, `docs/lirf_identity_experiment.json`). merry-quicksand name the training signal
precisely: on LIRF's **159k matched** rows the copy is visible as
`|BLOCK − SCHED| ≤ 60 ∧ |BLOCK − AOBT_3| > 300` — "a training signal we are not currently using
at all" **[S]** (`21-…md` §6.1b). Slot 19 independently proposes the same thing as a regime
indicator (a `BLOCK == SCHED` flag and the within-airport z-score of `mvt_minus_aobt3`) and
argues LIRF is "the only large remaining room" **[S/X]**.
(c) **Our own measurement bounds it and partly cuts against it** **[R]**
`docs/investigation_2026-09-16/results/lirf-matched-fill.md`: LIRF matched fill is **17.8 %**
(28,369 / 159,216), confirmed; LIRF matched OOF RMSE **347.1** is **7.90 % of SSE**, the worst of
any airport; removing filled rows lowers it to 294.4 but LIRF **still stays first**, so the fill
is "REFUTED as sufficient cause; CONFIRMED as largest single contributor". S7 — whether the fill
state is recoverable from available columns — is **UNDETERMINED**: in-sample R² of the fill flag
on group means is 0.156 (operator) to 0.238 (operator×segment×hour×far), and only 7 cells
(0.8 % of LIRF matched) reach fill ≥ 0.8. Our own file says explicitly that deciding this
"would require a model fit, which this item does not permit".
(d) **Not done.** `unmatched_sched_reparam` applies to the unmatched lane only **[R]**
`train.py:sched_reparam_shift`, called only from `unmatched_target` **[R]**. Nothing in the
feature set flags `BLOCK == SCHED` — and it could not, because `BLOCK_TIME` is blanked on scored
rows; the recoverable form is the *classifier* slot 08 builds, which is what is missing.
(e) Not blocked.
→ **actionable now.** The two competitors' recipes answer the exact question our own
investigation left open, on a population of 28,369 training rows and ~4,700–4,900 expected
ranking rows **[R]** (Ext-2, `lirf-matched-fill.md`). Slot 19 also notes the operator overlap is
99.4 %, so the classifier's main feature transfers.

### G21. LEBL / LEMD clock-precision asymmetry as a regime flag
Slots: **09** (9).
(a) `block_second_is_zero` is **8.30 / 8.37 %** at LEBL/LEMD while `mvt_second_is_zero` is
**1.66 / 1.64 %** — the two clocks have different precision regimes at those two airports, so the
target inherits the coarser one there **[S]** (`09-…md` 9, `repoB/docs/data_probe_report_2026-09-01.md:39-51`).
(b) Slot 09: "Repo B's table shows it; Repo B does not act on it; I have not seen it in our
notes." The ~8.3 % group is ~5× uniform, consistent with **5-second quantisation** rather than
the "clean second precision" Repo B concludes **[X]**.
(c) We have investigated the neighbouring question — `docs/investigation_2026-09-16/results/timestamp-precision-clusters.md`
covers MVT zero-second share by airport and AOBT_3 minute-rounding at 96.6 % **[R]** — but slot 09's
specific claim is about the **BLOCK** clock at LEBL/LEMD, which is the clock that defines the
label, and it is the asymmetry between the two that is new.
(d) Not done as a feature **[R]**; partially investigated as a diagnostic **[R]**.
(e) Not blocked.
→ **actionable now** as a check first, a feature second.

### G22. Spend the submission budget; treat the board as the better-powered comparator
Slots: **01** (2, 3, 9), **09** (B10), **14** (A9), **20**.
(a) A process change, not a model change.
(b) Ranking is **best RMSE across all submissions** with no private leaderboard, so a regression
costs nothing — our v11 at 293.98 did not hurt our 284.36 **[S]** (`01-…md` 3, 4). We have made
**11 submissions total** and 1 since 2026-09-17, against a top-10 median of 35.5, with ~20 days
remaining **[S]** — confirmed: `experiments/submissions.jsonl` holds versions 1–11 **[R]**.
Slot 09: feedback lands in the bucket as `<name>_result.json` within ~15 s, "so the board is a
paired comparator on a fixed set — cheaper and **better-powered** than our holdout for anything
above the noise floor" **[S]**. Slot 01: no team has recovered more than ~14 s through
incremental tuning after their first 3 days, and late structural jumps are still landing
(`gentle-tractor` +14.56 s and `zesty-puzzle` +13.40 s on 2026-09-20) **[S]**.
(c) **A factual disagreement between two slots on the limit itself**: slot 01 states the limit is
**5/day [S]**; slot 09 states **3 per team per day, resetting 00:00 UTC, unpublished** (Repo B
R33) **[S]**. Our code enforces `DAILY_LIMIT = 5` **[R]** `src/prc/submit.py:17`. Resolve before
planning a burst. Also binding: slot 14's anti-probing analysis — stay on the model side of the
line, submissions whose predictions are genuine model output read back only as "did this help"
**[S]**, and the rule text quoted in full at `../SUMMARY.md` §A.
(d) Not done.
(e) Not blocked.
→ **actionable now.** My assessment **[X]**: given that slots 09, 14 and 21 independently measure
our class of holdout at a 5–10 s noise floor while the board is exact on a fixed 344,841 rows,
this is the highest-leverage process item in the document — several of the candidates above are
individually unmeasurable *locally* and directly measurable *on the board*.

### G23. Repository readiness for the final GPLv3 publication — and do not publish early
Slots: **04** (17), **01** (12).
(a) Two opposite-direction actions with the same deadline.
(b) **Fourteen of thirty-three scored 2025 teams — including the would-be 3rd place (214.94) and
the JOAS-published kindly-garden (227.32) — were removed from the final ranking for not
publishing a documented GPLv3 repo** **[S]** (`04-…md` 17). Slot 04's own conclusion:
"Repository readiness is worth more RMSE-equivalent than any feature in this document."
Conversely, rank-3 `jolly-lobster` had a public repo indexed by search engines and it now returns
404; the open-source obligation is a **final**-solution requirement, not a during-competition one
**[S]** (`01-…md` 12).
(c) None.
(d) `LICENSE` is present at the repo root **[R]**; `pyproject.toml` declares
`license = { file = "LICENSE" }` **[R]**. I did not verify that the licence text is GPLv3 or that
the repository is private — both are outside a read of the code and should be checked.
(e) Not blocked.
→ **actionable now.** Outside all ten themes and it dominates them in expected value if it is not
already settled.

### G24. Switch leaderboard monitoring to the official API
Slots: **01** (1).
(a) `https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard`,
cursor-paginated via `nextCursor`: **every** competitor's full per-submission score history, 43
pages / 2,144 records, one unauthenticated pull **[S]** (`01-…md` 1). The fly.dev mirror gives
only best-per-team.
(b) As above.
(c) None.
(d) Not done **[R]** — no leaderboard fetcher exists in `src/`.
(e) Not blocked.
→ **actionable now.** Low value on its own; it is the input to G22's cadence decisions.

### G25. Read the OSN Discord `#prc-data-competition` backlog
Slots: **01** (13).
(a) `https://discord.gg/RPh89jpVVz`, channel `1304403918515605504`. The official site directs
participants there for urgent news and it is not search-indexed **[S]**.
(b) Slot 01: "it is the only material information source I could not access."
(c) None.
(d) Not done.
(e) **Blocked on a human with a Discord account** — no tool in this repo can reach it.
→ **blocked**, and worth naming because it is the only known unread source.

### G26. Keep a naive group-median control permanently in the harness
Slots: **03** (14), **04** (11).
(a) Report a per-airport (or airport × hour) median-taxi-out baseline on every validation run.
(b) PRC 2024's MTOW baseline (32,324.13 RMSE) was added at a reviewer's request and "makes every
subsequent number interpretable" **[S]** (`03-…md` 14). Slot 04: "any feature or architecture
change that does not widen the gap to it is noise" **[S]**; PRC 2025's degenerate per-type median
scored 256.27 and ranked 10 of 19 **[S]**.
(c) None.
(d) **Not done in the harness.** We have the numbers in docs (naive airport mean RMSE 514.8 vs
global mean 546.4 **[S]** via `12-…md`, from `docs/data_profile.md`) but `run_cv` reports no
control **[R]**.
(e) Not blocked.
→ **actionable now**, trivial.

### G27. Window-coverage / month-edge honesty columns
Slots: **23** (10).
(a) `observed_window_15m_sec`, `observed_window_60m_sec` = `min(seconds since month start,
w·60)`, plus `month_edge`.
(b) satam2: removing traffic counts entirely cost **+3.5 / +2.6 s**, so the counts matter; these
three columns make them honest at boundaries **[S]** (`23-…md` 10, `traffic.py:40-46`).
(c) Count family adjacency (see C45).
(d) **Partially.** We have exactly one such column, `window_coverage_60m`, computed against the
frame's own earliest departure **[R]** `features.py:context_features`, with the reason written
down. We have no per-window version and no month-edge flag — and the month scoping matters for
us because the ranking frame is two disconnected months **[R]**.
(e) Not blocked.
→ **actionable now**, and it is a correctness item for the ranking frame specifically.

### G28. A feature-starved global model as an extra ensemble member
Slots: **05** (2).
(a) NASA 1st place's global member saw only the flight-identity and proxy-revision columns plus
an airport categorical — no weather, no congestion **[S]** (`05-…md` 2,
`create_master/nodes.py::build_global_master`). Our analogue is named in the slot:
`{ADEP, AIRCRAFT_TYPE, AIRCRAFT_OPERATOR, ADES, WK_TBL_CAT, mvt_minus_aobt3, aobt3_minus_eobt1,
lobt_minus_eobt1, calendar}` only.
(b) As above. "Inputs: yes" — same feature table, only the column list changes **[S]**.
(c) None.
(d) Not done; `exclude_features` makes it a config change, not a code change **[R]**
`train.py:visible_numeric_columns`.
(e) **Blocked on C50** (no blend harness).
→ **blocked**, but note it is one YAML file away from existing once a blend exists.

### G29. GPBoost: tree boosting with random effects on the high-cardinality groups
Slots: **20** (8).
(a) A random intercept per stand / per operator × airport, with boosted trees fitting the
residual.
(b) On 8 high-cardinality tabular **regression** datasets GPBoost led at **7.02 % average
relative difference / rank 2.14**, against LGBM-Cat 17.4 % / 2.71 and CatBoost 44.5 % / 2.86
**[S]** arXiv:2307.02071 (`20-…md` 8). "The structure matches our data's shape."
(c) Slot 20 lists the caveats itself: those datasets are 28k–125k rows, the paper's author wrote
the library, and wall-clock on the largest (125k rows) was 45.9 s — scaling to 2.08M × ~4
grouping variables is untested **[S]**.
(d) Not done; not a declared dependency **[R]**.
(e) Not blocked, but high cost.
→ **actionable now** at low priority, and only in slot 20's staged form: run it on one airport
first and only scale if it wins there.

### G30. Peer-validated anomaly filtering of training rows
Slots: **02** (8) — against: **05** (20), **09** (B6), **13** (5), `../SUMMARY.md` §F.
(a) Drop from **training only** the rows with extreme `y` and no contemporaneous peer support at
the same airport ±30 min.
(b) Motoki's rule: an extreme value is unlearnable noise if no other entity at the same site/time
shows it, and real if peers do **[S]** (`02-…md` 8).
(c) **Overwhelming and specific.** Our own board: cap at 7,200 s **+77.6**, cap at 4,500 s
**+96.1**, and v33 — dropping just **17** rows with taxi > 6 h and no substitution — **+13.7 on
the board despite a −9.9 s gain on the bulk** **[S]** (`09-…md` B6). "The corrupted rows teach
the model that the tail exists, and the board scores the corrupted truth." Slot 05 item 20:
outlier trimming of the training set is defensible under MAE and dangerous under RMSE. Slot 13
item 5: the top 100 OOF rows carry **51.5 %** of total OOF squared error at 0.015 % of rows —
"under a mean-eliciting score these rows *are* the score" **[S-repo]**.
(d) Not done; we keep every labelled row **[R]** `train.py:5-6`.
(e) —
→ **refuted.** The peer-validation criterion is a narrower rule than the ones refuted, but v33 is
a direct measurement of the narrowest version (17 rows, peer-free by construction) and it cost
+13.7 s live. Listed so it is not proposed again on the strength of slot 02 alone.

---

# Part III — counts

## Distinct merged candidates by classification

| classification | count |
|---|---|
| actionable now | **46** |
| blocked | **11** |
| unmeasurable (real but below our 0.25 s instrument) | **4** |
| refuted | **3** (as merged candidates; ~25 further refuted claims are recorded inside the theme sections rather than as candidates) |
| already done (verified in code) | **9** (recorded inside the theme sections) |
| **total distinct merged candidates carried** | **64** |

## The 46 actionable candidates, distributed

| theme | actionable | blocked | unmeasurable | refuted |
|---|---|---|---|---|
| P1 unmatched lane specialist | 6 (C1, C3, C4, C5, C6, C7) | 1 (C2) | — | — |
| P2 cross-fit target encodings | 1 (C8) | — | — | — |
| P3 OOF fold column + protocol | 13 (C9–C22, less C10's blocked half) | — | — | — |
| P4 stand_arrival_gap + taxi | 4 (C23, C24, C25, C28) | 2 (C26, C27) | — | — |
| P5 physics-identity lower bound | 2 (C29, C30) | — | — | — |
| P6 per-airport upper bound | 2 (C31, C33) | 1 (C34) | — | 1 (C32) |
| P7 capacity on clipped residuals | 3 (C35, C36, splice-helper) | 1 (C37) | — | — |
| P8 weather + coverage parity | 4 (C38, C39, C40, C41) | 1 (OPDI `icao_turn`) | — | — |
| P9 long-horizon rolling durations | 3 (C42, C43, C44) | — | 4 (C45, C46, C47, C48) | — |
| P10 CatBoost as a rival | 1 (C49) | 2 (C50, C51) | — | — |
| **themes subtotal** | **39** | **8** | **4** | **1** |
| **GAPS (outside all ten)** | **20** | **3** (G8, G25, G28) | — | **1** (G30) |
| **total** | **46**¹ | **11** | **4** | **3** |

¹ 39 + 20 = 59 minus the 13 P3 items double-counted against their own sub-blockers; the
reconciled actionable count is **46** distinct candidates that can be started today with no
predecessor. The theme/gap split of those 46 is **28 inside the ten themes** and **18 in the
gaps** (G1–G7, G9–G24, G26, G27, G29 less the three blocked and one refuted).

## The gap list, one line each — this is the output of this slot

Twenty distinct candidates fall outside all ten planning themes. Eighteen are actionable now.

| # | gap candidate | class | strongest number |
|---|---|---|---|
| G1 | Delete the ten monotone `_slog` duplicate columns | actionable | Alligier A3/A4/A5 [S]; our own ±0.5 s-per-column instrument defect [R] |
| G2 | Group-relative deviation features on the clock columns | actionable | "strongest structural idea in the 2024 corpus that transfers intact" (slot 03) [S] |
| G3 | Tail-copy clipped feature triplet | actionable | javidmardanov v2 tail delta 265.81 vs 266.05 [S] |
| G4 | Tweedie / Poisson / Gamma objective sweep | actionable | Bregman-consistent for the mean, no retransformation bias (slot 13) [S] |
| G5 | Free affine recalibration `(a, c)` diagnostic | actionable | zero fits; closes the quantile/expectile question (slot 13) [S] |
| G6 | Label-free frequency encodings + categorical crossings | actionable | two Kaggle first places via crossings (slot 20) [S] |
| G7 | Unimpeded reference to the current spec + level categorical | actionable (disputed) | 18.8 % of Repo B's gain [S] **vs** slot 17 "do not build" [S] |
| G8 | Stand→runway taxi distance | **blocked** | top-3 at 3 airports (Wang 2021) [S]; no STAND crosswalk, failed 3× [S] |
| G9 | Operator sign-consistency test + rare-level back-off | actionable | operator ε² 0.11, 11 ranking operators unseen in training [S] |
| G10 | Ordered-neighbour NM clock channels (`own_minus_mean`) | actionable | −0.8819 and −1.4181 s **official** (Phoenix v9, v12) [S] |
| G11 | `MVT_ID_mvt` record-order peer features | actionable | mechanistic only [X]; targets the exact offset our model predicts |
| G12 | Unused schema columns (CALLSIGN, ADES_FILED, ARVT_3) | actionable | our own log's own open list [R] |
| G13 | **Arrival mirror as a labelled 2025→2026 drift gate** | actionable | 202.40 → 214.96 (+6.21 %), EHAM +128.9 vs EGLL −40.9 [S] |
| G14 | Train only on served rows; reference tables on matched rows only | actionable | MB3 −5.99 s live at zero offline price [S]; E18-H −3.68 s [S] |
| G15 | Real fallback chain: model → proxy → per-airport median | actionable | getting it wrong: matched 379 → 398 [S] |
| G16 | **Backward feature elimination / LOFO** | actionable | PRC 2025 rank 1 used it; rank 11 used forward selection [S] |
| G17 | `\|delta\|`-band y-target model on the matched tail, ex-LIRF | actionable | +9.99 s [+5.63, +14.35] on 31 % of matched SSE [S] |
| G18 | Raw-target partner blended with the residual model | actionable | NASA 1st blends both; merry-jacket blends two parameterisations [S] |
| G19 | Repair model keyed on the main model's own output | actionable | NASA 2nd place's `train_negative_model` [S] |
| G20 | **LIRF matched-fill regime signal (159k matched rows)** | actionable | 18.08 % of LIRF training rows; 266.05 → 263.22 local [S]; our own S7 left UNDETERMINED [R] |
| G21 | LEBL/LEMD clock-precision regime flag | actionable | 8.30/8.37 % vs 1.66/1.64 % [S] |
| G22 | **Spend the submission budget; board as comparator** | actionable | 11 of ~100 remaining slots used; best-of-all scoring [S]; 5/day [S] vs 3/day [S] |
| G23 | **GPLv3 repo readiness; do not publish early** | actionable | 14 of 33 scored 2025 teams removed from the final ranking [S] |
| G24 | Official leaderboard API | actionable | 2,144 records, one unauthenticated pull [S] |
| G25 | OSN Discord backlog | **blocked** | "the only material information source I could not access" [S] |
| G26 | Naive group-median control in the harness | actionable | PRC 2024 reviewer-requested baseline [S] |
| G27 | Window-coverage / month-edge columns | actionable | removing traffic entirely cost +3.5/+2.6 s [S] |
| G28 | Feature-starved global model as a member | **blocked** (on C50) | NASA 1st place's 4th member [S] |
| G29 | GPBoost random effects | actionable (low priority) | 7.02 % vs 17.4 % (LGBM-Cat) on high-card regression [S] |
| G30 | Peer-validated anomaly filtering of training rows | **refuted** | v33: 17 rows dropped, **+13.7 on the board** [S] |

(30 rows; G8, G25, G28 blocked and G30 refuted leaves **26** live gap candidates, of which 18
are actionable with no predecessor and 8 are actionable-but-sequenced behind an item in the
themes — principally G10 behind C35 and G18 behind C50.)

## The five recorded disagreements

| # | candidate | slot A says | slot B says |
|---|---|---|---|
| 1 | **C8** cross-fit the target encodings | slot 20: theory (arXiv:1706.09516 §3.2, sklearn default) plus our own 34.8 s CV vs 13.2 s board signature — do it, decide on the board | slots 21 and 09: merry-quicksand list it among confirmed dead ends at **−0.96 s**; sergiuv11 Repo A shipped LOMO cross-fitting and it **failed their gate** |
| 2 | **C34** per-airport bias correction | slot 12: NASA 3rd shipped it; OpenAir measured per-airport residual means of −236 s (LTFM) and +214 s (LIRF); run it *before* specialists | slot 15: STRONG negative — our OOF file cannot support a fitted correction (July assigned twice), and kind-mango measured airport-level shift calibration at **near-zero** |
| 3 | **C44** disruption-state feature size | slot 06 / OpenAir: "a real but small (~1 % of matched RMSE)", ">60 min not improved", "LFPG ~0" | slot 17: 1 % of ~202 s matched RMSE is **≈2 s**, comfortably above our 0.25 s floor — the one congestion candidate we can resolve |
| 4 | **G7** unimpeded reference feature | slots 19/13/09: most-repeated construction in the literature; **18.8 % of Repo B's total gain**; emit the level as a categorical | slot 17: our own within-group ceiling is 268.1 for `airport×stand×runway×hour×type` while the model is already at ~202 — "do not build" |
| 5 | **G22** submission limit | slot 01: **5/day [S]** (matches our `DAILY_LIMIT = 5` [R]) | slot 09: **3 per team per day, resetting 00:00 UTC, unpublished** (Repo B R33) [S] |

---

## Appendix — what I verified in the code, and what I did not

Verified by reading (`[R]` above): `src/prc/train.py`, `features.py`, `cv.py`, `compare.py`,
`predict.py`, `pipeline.py`, `weather.py`, `adversarial.py` (head), `submit.py` (limits),
`configs/main.yaml`, `configs/submit_v10_reparam.yaml`, `pyproject.toml`,
`experiments/experiments.jsonl` (43 records, no `decision` field),
`experiments/submissions.jsonl` (versions 1–11), `docs/features_log.md`,
`docs/investigation_2026-09-16/results/lirf-matched-fill.md`, and
`git show codex/feature-engineering:src/prc/features.py` for the unmerged `stand_arrival_*`
block. `git diff main -- src configs` is empty, so every code claim holds for `main` too.

**Not verified, and it matters:**
- Whether `LICENSE` is GPLv3 and whether the GitHub repository is currently private (G23).
- The ±3,606 s LOBT invariant on our own data (C30) — javidmardanov verified it on theirs.
- Whether LEMD/LEBL DEP taxi-out is censored at 3,600 s in *our* 2025 data (C33).
- The actual submission limit (G22), where two slots disagree.
- I did not run anything. No code was modified by this slot.
