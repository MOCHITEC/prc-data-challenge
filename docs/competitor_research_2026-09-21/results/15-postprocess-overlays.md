# Slot 15 — `postprocess-overlays`: post-processing of model output

Investigator note, 2026-09-21. Every claim is marked **[S]** (sourced, with URL or an in-repo path)
or **[X]** (my inference / arithmetic / judgement, not sourced). No code was modified.

Baseline for all arithmetic below: team `resilient-hedgehog`, RMSE **284.3584** on the
**344,841**-row Jan+Jul 2026 ranking set. **[S]** row count and metric:
<https://prc-data-challenge-2026.netlify.app/> — "The applied ranking will use Root Mean Square Error
(RMSE) for submitted predictions for the movements in January and July 2026"; the 344,841 figure is
confirmed independently by team `kind-mango`
(<https://github.com/skylinkapi/prc-data-challenge-2026-kind-mango>, "All are scored on the same
344,841-row 2026 Jan+Jul ranking set").

---

## 0. The arithmetic that governs every overlay decision

**[X]** (pure arithmetic from the numbers above, verifiable):
at RMSE 284.3584 over N = 344,841 rows, MSE = 80,859.70 and SSE = 2.7884e10. A gain of `d` seconds of
RMSE requires a reduction in **total** squared error of `(R² − (R−d)²)·N`:

| RMSE gain | required ΔSSE | equivalent: one row whose \|error\| falls to 0 from | equivalent: 20 rows each falling to 0 from |
|---|---|---|---|
| 0.10 s | 1.96e7 | 4,428 s | 990 s each |
| 0.20 s | 3.92e7 | 6,262 s | 1,400 s each |
| 1.00 s | 1.96e8 | 13,992 s | 3,129 s each |
| 2.33 s | 4.55e8 | 21,333 s | 4,770 s each |
| 7.53 s | 1.46e9 | 38,173 s | 8,536 s each |

Two consequences **[X]**:
1. A row-level overlay is only worth anything if it moves predictions by **thousands of seconds**.
   A rule that shifts 30 rows by 200 s each is worth ~0.0000 s of RMSE and is pure risk.
2. The payoff is *quadratic and unbounded in the wrong direction*. Replacing prediction `p` with `r`
   on a row with truth `y` changes SSE by `(p−y)² − (r−y)²`. If `r` is 20,000 s away from `y` when `p`
   was 1,000 s away, that single row costs 4.0e8 of SSE ≈ **2.0 s of RMSE**. One sign error on one row
   can wipe out a correct rule applied to fifty.

---

## 1. Technique-by-technique evidence

### 1.1 Clipping / bounding predictions to a range

**[S] Strongest evidence available, and it is from this exact competition.** Team `kind-mango`
(same 344,841-row scored set), upload v48:

> "Fifteenth-pass Phase 1: MS1 per-airport upper bounds and MS2 clean-floor post-processing on the v47
> parquet … 5 non-LIRF extreme rows capped (biggest EDDM 21,465 -> 7,499 s). MP7 label-free gate passed.
> Zero training cost. Live **-2.33 s** (about 1,400 MSE) essentially all from the five MS1 clips - MS1 fold
> price was **0 MSE** because the base doesn't extrapolate on 2025 rows, but does on 2026
> out-of-distribution rows."
> — <https://github.com/skylinkapi/prc-data-challenge-2026-kind-mango> (README, leaderboard table)

Detail of the same event from that team's RECAP **[S]**
(<https://raw.githubusercontent.com/skylinkapi/prc-data-challenge-2026-kind-mango/main/RECAP.md>):

> "MS1 moved 5 non-LIRF rows (EDDM 21,465 -> 7,499 the biggest -13,966 s; EHAM 11,103 -> 7,311; two LEBL
> rows to 4,204; LEMD 4,376 -> 4,198). MS2 moved 104 rows (small pushes). … Live -2.33 s (about 1,400 MSE)
> essentially all came from the five MS1 clips; MS1 fold price was 0 MSE because the base does not
> extrapolate on 2025 rows. The label-free evidence at upload time (5 rows with predictions 2-6x above the
> airport's 2025 support, all with `mvt_eobt1` below the eligibility bar) was correct."

**How the range was chosen [S]** (same repo, `docs/MODEL_ANALYSIS.md` §5.6):

> "**MS1. Bound predictions per airport after the ensemble.** At EDDF, EDDM, LEBL and LEMD, which have no
> 2025 row above 7,200 s outside fallback, cap at the 2025 airport maximum of that class plus a margin. At
> EGLL, EHAM, LFPG, LSZH and LTFM, allow values above 7,200 s only with a flight record and `mvt_eobt1`
> above 5,400 s. LIRF follows MH4."
> "**MS2. Clip the ensemble mean, not the members**, and floor at the per-airport 0.1 % quantile of 2025
> clean labels."

i.e. the bound is **the training-year per-airport support of the non-fallback class**, plus a margin, with
an operational escape hatch (`mvt_eobt1 > 5,400`) for rows that have a reason to be long. It is
**label-free on the test set** — it needs only the 2025 label support and the 2026 feature values.

**Failure modes of clipping [S], three separate ones:**

- *Too tight a range destroys the metric.* Same team, "Approaches tried and RULED OUT":
  "**Tighter target range `[30, 21600]`** — hurt RMSE 6+ s."
- *A cap that patches a model defect is worth less than fixing the defect.* The same team's next upload
  retrained the base excluding the rows that caused the extrapolation and gained **-5.99 s**, after which
  "MS1 finds nothing to clip … post-MS moved only 85 rows with max |diff| 304 s (v48 had 109 rows, max
  13,966 s)". **[S]** (RECAP, v51). So the clip bought 2.33 s; removing the cause bought 5.99 s and made
  the clip inert.
- *Clipping cannot rescue a bad transformation.* Their v54 clipped an anchored-offset prediction "to
  per-airport [q0.001, q0.999] offset range from training" and still scored **+35.86 s** worse. **[S]** (RECAP, v54).

**[S] Our own repository has already measured this on our OOF** —
`/home/user/prc-data-challenge/docs/investigation_2026-09-16/results/tail-postprocessing-cap-shrink.md`:

- A **blanket** cap on all unmatched predictions is *refuted*: every cap ≤ 43,200 raises total OOF RMSE
  (334.58 → 452.59 / 433.64 / 404.56 / 335.42 at 3,600 / 7,200 / 14,400 / 43,200), because the LIRF
  unmatched rows genuinely carry huge labels ("affected-row label means 9,084 – 47,417").
- A cap on **non-LIRF unmatched rows only** is *confirmed*: cap 3,600 → total OOF RMSE **334.58 → 327.05
  (−7.53 s)**, 53 rows affected, **49 helped / 4 hurt**, ΔSSE = −3.35e9, "negative in each of the four OOF
  months and at each of the eight affected airports". Cap 7,200 → 329.33 (−5.25 s), 24 rows, **24 helped /
  0 hurt**, all 24 with labels < 1,744 s.
- Capping **matched** rows is refuted (216.4 → 243.7 at cap 3,600).

**[X]** This is the same rule `kind-mango` shipped for −2.33 s live, arrived at independently, measured on
our own folds with a per-month and per-airport sign check, and never shipped.

### 1.2 Shrinking extreme predictions toward a conditional median or mean

**[S] In-competition, kept:** team `OpenAir` (<https://github.com/ChandanHegde07/OpenAir>) applies
"if OPDI `icao_turn≥7200`, blend T 50% toward train long-hold median taxi (~1131 s); ranking 146/383
long-hold rows", validated on "Jan+Jul (Dec fails, as expected)". Their leaderboard went 278.36 (v23) →
276.99 (v24) on the upload that contained it. **[X]** The 1.37 s is *not* attributable to the shrink alone —
the README does not isolate it — and the rule explicitly fails on December, i.e. it is known not to
generalise across season.

**[S] Refuted on our own OOF** (same in-repo file as above, §3.4–3.5): shrinking unmatched predictions
toward the per-airport median of unmatched training labels worsens RMSE at every λ and under **all seven**
target definitions tried (λ=0.75 → unmatched RMSE 2,467–2,479 vs baseline 2,296; λ=0.5 → 2,741–2,854;
constant → 3,492–3,806). Adding an hour dimension to the target moves the result by ≤ 35 s against a
180–850 s deterioration from the shrink itself. Verdict recorded: REFUTED.

**[X]** The mechanism is visible in the same tables: on LIRF unmatched rows the model's large predictions
have large labels (it *under*-emits the tail: 43 predictions > 4 h against 51 labels ≥ 4 h; 5 > 12 h against
15), so shrinking them destroys real signal. On non-LIRF unmatched rows it *over*-emits (9 predictions > 4 h
against 2 labels), which is why the capped-non-LIRF variant works and the blanket variant does not.

### 1.3 Isotonic regression / Platt-style calibration of predictions

**[S] scikit-learn** (<https://scikit-learn.org/stable/modules/calibration.html>):

> "The `method="isotonic"` fits a non-parametric isotonic regressor, which outputs a step-wise
> non-decreasing function" … "However, it is more prone to overfitting, especially on small datasets" …
> "Overall, `'isotonic'` will perform as well as or better than `'sigmoid'` when there is enough data
> (greater than ~ 1000 samples) to avoid overfitting" … "Ideally, the calibrator is fit on a dataset
> independent of the training data used to fit the classifier in the first place."

**[S] In-competition use is on a gate probability, not on the regression output.** `kind-mango` calibrates
a *classifier* (`p_fb`, the LIRF fallback gate) with isotonic regression and then mixes:
`mixture = p_fb_cal * sd + (1 - p_fb_cal) * R_norm_LIRF`. Refitting that gate + isotonic on all 12 months
gained **-0.33 s** (v56). A related LIRF sub-rule deliberately *de*-calibrates: "The raw probability is
shrunk by 0.6 (a conservative de-calibration). Covers 305 rows."
— <https://github.com/skylinkapi/prc-data-challenge-2026-kind-mango>

**[X]** I found **no** sourced case, in this competition or in the Kaggle write-ups I could reach, of
isotonic regression applied to the *regression output itself* against OOF truth producing a reported RMSE
gain. Under squared loss a well-fit GBDT is already conditionally-mean calibrated in-sample; an isotonic
map fitted on OOF mostly re-learns fold-specific bias. Treat "isotonic on the prediction" as unevidenced.

### 1.4 Per-group (airport / month / hour) additive or multiplicative bias correction

**[S] Explicitly measured and rejected in this competition.** `kind-mango`, "Approaches tried and RULED
OUT": "**Post-hoc airport-level shift calibration** — near-zero effect."
— <https://github.com/skylinkapi/prc-data-challenge-2026-kind-mango>

**[S] The one large documented win for a global multiplicative correction is Rossmann (RMSPE, not RMSE).**
The 1st-place winning-model documentation states:

> "I also multiplied all my predictions with a constant factor to improve them. A factor of 0.995 was
> optimal for both my validation set and the public test set. But I should have applied the mathematical
> estimate of 0.985 on the forum [5] to achieve a private leaderboard score well below 0.10."
> — <https://storage.googleapis.com/kaggle-forum-message-attachments/102102/3454/Rossmann_nr1_doc.pdf>

A top-10% competitor in the same competition **[S]**: "0.985 correction was insignificant on
cross-validation … effect was less than standard deviation of RMSPE from different folds, but helped on
both private and public leaderboards." — <https://mabrek.github.io/blog/kaggle-forecasting/>

**[X]** The instructive part is the *failure*: the factor tuned empirically on validation **and** on the
public leaderboard (0.995) was worse than the factor derived analytically from the metric (0.985). A
correction fitted to held-out data landed on the wrong side of a correction derived from first principles.
Also note RMSPE is scale-relative, so a multiplicative shrink has a first-order justification there that it
does **not** have under plain RMSE — under RMSE the optimal global scaling of an unbiased predictor is 1.0.

### 1.5 Rank-preserving distribution matching (predictions → training target distribution)

**[X]** I could not find a sourced case of quantile/histogram-matching a regression prediction vector to the
training target distribution producing a reported RMSE gain. The closest sourced material is the **negative**
result: matching the *tail frequency* of the prediction distribution to the training label frequency is what
Data Science Bowl 2019 competitors called overfitting —

**[S]** "Using training data distribution to set thresholds was considered a form of overfitting, as the
public dataset represented a rare case distribution"; top teams instead averaged the calibration over many
resampled OOF sets ("OptimizedRounder on 500 truncated oof validations", "51 validation sets per fold—one
for early stopping, 50 others for averaging").
— <https://naotaka1128.hatenadiary.jp/entry/dsb-2019-top-solution> and
<https://takaishikawa42.hatenablog.com/entry/2020/01/29/090852>

**[X]** Under RMSE specifically, distribution matching is actively harmful in expectation: the conditional
mean predictor is *supposed* to be under-dispersed relative to the target (Var(E[y|x]) ≤ Var(y)). Forcing the
prediction distribution to match the label distribution inflates variance and raises RMSE. This is the
theoretical reason the shrink results in §1.2 and the caps in §1.1 diverge — shrinking toward a median
reduces dispersion (sometimes right), matching distributions increases it (structurally wrong for RMSE).

**[S] Our own repo already has the diagnostic for this**, §3.8–3.9 of
`tail-postprocessing-cap-shrink.md`: v7 emits 34 predictions > 4 h against 20.2 implied by the all-month
training rate and 45.1 implied by the Jan+Jul rate; 3 predictions > 12 h against 5.6 / 14.0 implied. The two
reference rates disagree by more than a factor of two, so "match the training tail frequency" does not even
define a unique target here.

### 1.6 Rule overlays that override the model on a small identified row set

**[S] The best-documented in-competition instance, with both local and official numbers**, is team
`zestful-fountain` / Phoenix-Ops v11
(<https://github.com/Phoenix-Ops-LTD/prc2026-taxiout>):

> "One fixed local diagnostic improves proposed-v10 reused January/July RMSE from **314.3197465013 to
> 313.7463286061**, a **0.5734178952-second** gain, with both months improving. It changes **31 of 344,419
> rows** and retains all original labels, including 80 negatives. **The largest beneficial row supplies
> 45.8 % of net squared-error improvement**; excluding it only for influence analysis leaves a 0.311181-second
> gain … These are reused selection data … this is not an untouched validation result."
> "Target-free ranking checks found **21 eligible rows** … V11 then scored **274.9738 RMSE** officially,
> improving v10 by **0.2013 seconds**."

From their METHOD.md **[S]**: among the 31 eligible holdout rows, "**18 squared errors improve and 13
worsen**; 12 days improve, 6 worsen and 44 are unchanged", and "No thresholds were searched; the rule was
frozen before one new local calculation."

**[X]** Three numbers to keep: (a) the rule is right on only **58 %** of the rows it touches; (b) **46 %** of
the net gain comes from **one row**; (c) the realised official gain (0.2013 s) is **35 %** of the gain
measured on reused selection data (0.5734 s).

**[S] The catastrophic failure mode of the same family**, from `kind-mango`'s ruled-out list:

> "**Old tail-mixture (P(y > 3600) classifier + hard swap)** — v22/v27; live went from **372 to 625**. **Hard
> swaps have unbounded downside.**"
> — <https://github.com/skylinkapi/prc-data-challenge-2026-kind-mango>

That is **+253 s of RMSE** from a rule that replaced predictions on a classifier-selected row set.

**[S] The classic Kaggle analogue** is Elo Merchant Category Recommendation (RMSE; ~1 % of the target at
−33.2): "set top k highest outlier probability samples' target to be -33 based on submission"; the author
records that many competitors "shaked down … because they got too many false positives in outlier detection
on the private LB part", and that for this author "the submission without outlier detection won".
— <https://github.com/bangdasun/kaggle/blob/master/elo-merchant-category-recommendation/README.md>

**[S] A rule overlay that did win a competition** is David Thaler's 1st place in Walmart Recruiting — Store
Sales Forecasting: "an unweighted average of 6 component models … followed by a transformation around
Christmas to reflect that the day of the week that Christmas lands on shifts from year to year"
(<https://github.com/davidthaler/Walmart_competition_code>, `postprocess.R`). **[X]** No score attribution
for the transformation is published in that README; what distinguishes it from the failed overlays is that
the row set was identified by a **calendar fact**, not by a model score or a threshold search.

---

## 2. How practitioners avoid overfitting a calibration on out-of-fold data

Four sourced practices, in descending order of how well they are evidenced:

1. **Pre-register one price per upload and never build on an unscored upload. [S]**
   `kind-mango`, `docs/MODEL_ANALYSIS.md` §6: "One pre-registered price per upload. Never build on an
   unscored upload." and "Stop rule: if phases 1 to 3 accept less than 2,000 MSE of fold gain in total,
   skip phase 4 …". Phoenix-Ops state the converse explicitly as a limitation: "This frozen diagnostic had
   no predeclared promotion criteria; any later release decision follows observation of its results." **[S]**

2. **Gate the file on label-free checks instead of on the fitted gain. [S]**
   `kind-mango` MP7: "**Gate every file on label-free checks.** Per airport: maximum prediction; counts
   above 3,600 s and 7,200 s against the 2025 support; a list of every row moved by more than 3,000 s with
   airport, `sd`, `mvt_eobt1` and record status … Block the upload on an unexplained row or a strict drift
   failure." This is what actually justified the −2.33 s clip: the fold price was 0, the *label-free*
   evidence (predictions 2–6× above the airport's 2025 support) was the reason to ship.

3. **Average the calibration over many resampled held-out sets rather than fitting it once. [S]**
   Data Science Bowl 2019 top solutions: thresholds from "OptimizedRounder on 500 truncated oof
   validations"; "51 validation sets per fold—one for early stopping, 50 others for averaging QWK scores";
   the popular public rounder "suffered from initial value dependency, frequently falling into local
   optima". — <https://naotaka1128.hatenadiary.jp/entry/dsb-2019-top-solution>

4. **Prefer a correction derived from the metric/mechanism over one tuned on held-out data. [S]**
   Rossmann 1st place: the tuned 0.995 beat the analytic 0.985 on both validation and the public
   leaderboard, and lost to it on the private leaderboard
   (<https://storage.googleapis.com/kaggle-forum-message-attachments/102102/3454/Rossmann_nr1_doc.pdf>).

**The transfer ratios that are actually published, for this competition [S]:**

| change | offline measurement | realised live | ratio |
|---|---|---|---|
| `kind-mango` v44+v45+v46 base stack | −1,864 paired MSE | −102 MSE | **14 %** ("the paired hold-out is poorly calibrated once several base changes stack on each other") |
| `kind-mango` v49 LIRF head rebuild | −2,405 paired MSE | **+0.42 s (regression)** | **negative** |
| `kind-mango` v48 MS1 clip | **0 MSE** fold price | −2.33 s | **infinite** (offline blind to it) |
| `kind-mango` v51 MB3 base filter | zero paired price | −5.99 s | **infinite** |
| Phoenix-Ops v11 overlay | −0.5734 s on reused selection data | −0.2013 s official | **35 %** |

**[X]** The pattern across all five rows: for this competition the 2025-fold estimate of a change's value is
approximately uninformative about its 2026 value, in *both* directions. It systematically under-states
anything whose mechanism is "the model behaves differently on 2026 out-of-distribution rows" and
over-states anything fitted on the folds. That is the single most important calibration fact in this file.

---

## 3. Calibration fitted on a NON-UNIFORM out-of-fold file

**[S] The authoritative statement is scikit-learn's**
(<https://scikit-learn.org/stable/modules/cross_validation.html>):

> "The result of `cross_val_predict` may be different from those obtained using `cross_val_score` as the
> elements are grouped in different ways. The function `cross_val_score` takes an average over
> cross-validation folds, whereas `cross_val_predict` simply returns the labels (or probabilities) from
> **several distinct models undistinguished**. Thus, `cross_val_predict` **is not an appropriate measure of
> generalization error**."
> "The function `cross_val_predict` is appropriate for: Visualization of predictions obtained from different
> models. **Model blending:** when predictions of one supervised estimator are used to train another
> estimator in ensemble methods."
> "Only cross-validation strategies that **assign all elements to a test set exactly once** can be used
> (otherwise, an exception is raised)."

**[S] The stacking convention** is that the folds must be identical across every base model, so that the
meta-feature the second stage is trained on has the same distribution as the one it will see at serving
time — see the Kaggle discussion "Cross validation strategy when blending/stacking"
(<https://www.kaggle.com/general/18793>) and the standard write-up
(<https://machinelearningmastery.com/out-of-fold-predictions-in-machine-learning/>). **[X]** The page bodies
do not render to a fetcher, so I mark the *specific wording* of those two as unverified; the sklearn
quotation above carries the point on its own.

**[S] Our OOF file is exactly the non-uniform case.** From
`/home/user/prc-data-challenge/docs/investigation_2026-09-16/results/fold-shape-vs-ranking.md`:

| fold | validates | n_valid | training months | n_train |
|---|---|---|---|---|
| jan | 2025-01 | 153,706 | 2025-02 … 12 (11) | 1,931,341 |
| jul | 2025-07 | 190,713 | 2025-01…06, 08…12 (11) | 1,894,334 |
| jul_fwd | 2025-07 | 190,713 | 2025-01 … 06 (6) | 1,005,519 |
| forward | 2025-11, 2025-12 | 328,009 | 2025-01 … 10 (10) | 1,757,038 |

and: "`overall` in `run_cv` is built with `pl.coalesce` in fold order, so the `jul_fwd` predictions never
enter it (`overall.n` = 672,428 while Σ fold n = 863,141)".

**[X] Verdict on question 3.** A calibration fitted on `oof_main-48c19bc830.parquet` is **not valid** as an
estimate of the production predictor's bias, for four independent reasons:
- The July rows are predicted by **two** different models (`jul`, `jul_fwd`) and one is silently discarded
  by the coalesce; the file violates sklearn's "assign all elements to a test set exactly once" precondition.
- The four folds are trained on 6, 10, 11 and 11 months, so the four blocks of the file carry **four
  different amounts of model capacity**. A global calibration curve fitted on the pooled file is an average
  over four different estimators, none of which is the production model (trained on all 12 months).
- The `jan` fold is a **backcast** (11 training months, all later than the validated month, horizon −11).
  Any January bias it reveals is the bias of a model that saw February–December — the opposite of
  production, which forecasts January 2026 from 2025. A per-month additive correction fitted on that
  January block measures an artefact of the fold design, not a January effect.
- The file covers months 1, 7, 11, 12 only, so a per-month correction is undefined for eight of twelve
  months and a per-(airport, month) correction is estimated from four months of one year.

**[X] Standard practice** where the OOF is heterogeneous is either (a) rebuild a single clean partition with
one recipe and refit the calibration on it, or (b) do not fit a calibration at all and use only
**sign-checked, parameter-free** rules whose validity can be asserted per fold, per month and per airport
separately — which is exactly the form the §1.1 cap takes and the form a fitted isotonic/bias curve does not.

---

## 4. Detecting and correcting a level shift between the training year and the test year

Sourced methods, without test labels:

1. **Adversarial validation. [S]** Train a binary classifier to separate train rows from test rows; AUC ≈ 0.5
   means the feature distributions agree, AUC → 1 means they do not, and the rows the classifier calls
   "test-like" can be up-weighted or used as the validation set.
   — <https://medium.com/@nlztrk/adversarial-validation-a-sanity-checker-and-an-exploiter-2dff1baced19>;
   peer-reviewed application: <https://arxiv.org/pdf/2112.10078>. **[X]** This detects *covariate* shift. It
   cannot by itself measure a *label* level shift, which is what a bias correction would need.

2. **Per-column quantile monitoring against the same month of the training year. [S]** `kind-mango` MP7:
   "value quantiles of every served column by airport and month against the same 2025 month. Block the
   upload on an unexplained row or a strict drift failure." They publish the resulting drift table, e.g.
   "`eobt1_sched` non-zero share, 2025 to July 2026: LTFM 40.3 to 19.6 % (strict monitor failure), LFPG 35.7
   to 61.6 %, EGLL 46.5 to 59.3 %" and "LFPG `mvt_eobt1` median 1,441 s … shifted by 298 s".
   — <https://raw.githubusercontent.com/skylinkapi/prc-data-challenge-2026-kind-mango/main/docs/MODEL_ANALYSIS.md>

3. **A mirror target whose labels *are* available in the test period. [S]** `kind-mango` MP6: "**Build the
   arrival mirror.** Same frame builder and recipe, target arrival taxi-in on validated arrivals, trained on
   2025, scored on 2026 per airport and month. Report the arrival-side delta next to every fold price. **A
   lever that helps the 2025 folds and hurts 2026 arrivals does not ship.**" This is the cleanest published
   answer to "how do you measure a 2025→2026 level shift with no 2026 departure labels": score a *related*
   target that is observable in 2026. **[X]** Their v59 result is the caution: the arrival-side drift signal
   "does not transfer to departure taxi-out", live +0.37 s. A mirror is a gate, not a correction factor.

4. **A public reference series. [S]** EUROCONTROL publishes a monthly, per-airport **Additional Taxi-Out
   Time** dataset (columns `YEAR, MONTH_NUM, MONTH_MON, APT_ICAO, APT_NAME, STATE_NAME, TF, VALID_FL,
   NO_REF, TOTAL_REF_NB_FL, TOTAL_REF_TIME_MIN, TOTAL_ADD_TIME_MIN, COMMENT, PIVOT_LABEL`), from 2014
   onward, sourced from the same Airport Operator Data Flow that produces the challenge labels.
   — <https://ansperformance.eu/reference/dataset/taxi-out-additional-time/>,
   <https://www.eurocontrol.int/prudata/dashboard/metadata/additional-taxi-out-time/>. A search result on
   the portal states July 2026 data is already downloadable **[S, weak — I could not confirm the latest
   available month on the dataset page itself]**. `TOTAL_ADD_TIME_MIN / TOTAL_REF_NB_FL` gives a per-airport,
   per-month mean additional taxi-out in minutes for Jan 2025, Jul 2025, Jan 2026 and Jul 2026; another team
   in this challenge builds directly on it (<https://github.com/ahmetabdullahgultekin/prc-taxiout-2026>,
   "Reimplements EUROCONTROL's additional taxi-out time reference and models the residual").

5. **A global multiplicative factor derived from the metric rather than tuned.** See §1.4 (Rossmann). **[X]**
   Under plain RMSE with a conditional-mean predictor there is no analytic factor to derive, so the Rossmann
   precedent does not carry over; what carries over is the warning that the tuned factor lost to the
   derived one.

**[X]** What is *not* sourced anywhere I could reach: any competitor successfully applying an additive
per-(airport, month) level correction estimated from a public series. `kind-mango` tried the closest thing
("post-hoc airport-level shift calibration") and measured "near-zero effect".

---

## 5. How competitors decide the row count for a rule overlay

**[X] There is no published threshold.** I looked for guidance of the form "a rule must touch at least N
rows" in Kaggle write-ups and in the four public 2026 competitor repositories and found none. What exists
instead are five decision procedures, all sourced:

1. **Magnitude, not count, is the published criterion. [S]** `kind-mango`'s MP7 gate lists "a list of every
   row moved by more than **3,000 s**" and their stop rule is denominated in **MSE of fold gain** ("if phases
   1 to 3 accept less than 2,000 MSE of fold gain in total, skip phase 4"), not in rows. Their winning
   overlay touched **5 rows** and was worth −2.33 s because each moved by 5,000–14,000 s. Their 17-row MS3
   disagreement rule (v50) and 104-row MS2 floor were kept but are described as "small pushes" with no
   claimed gain.

2. **Influence analysis: what happens if the biggest row is wrong. [S]** Phoenix-Ops publish exactly this:
   "The largest beneficial row supplies 45.8 % of net squared-error improvement; excluding it only for
   influence analysis leaves a 0.311181-second gain and improvement in both months." **[X]** This is the
   procedure to copy: recompute the gain with the top-1 and top-3 contributing rows removed. If the rule
   only works because of one row, it is a coin flip, not a rule.

3. **Hard replacement vs. blending. [S]** "Hard swaps have unbounded downside" (`kind-mango`, after a
   classifier-selected hard swap took the live score from 372 to 625). The overlays that survived in this
   competition are either bounded (`min(pred, cap)`, which can only move a prediction *toward* the bulk) or
   a mixture (`p*a + (1−p)*b`, OpenAir's 50 % blend, `kind-mango`'s `p_fb_cal * sd + (1−p_fb_cal) * R_norm`).
   The one that blew up was a full replacement on a *model-selected* row set.

4. **A 1–2 row post-processing is a correctness fix, not a gain lever. [S]** javidmardanov's attempt log
   records restoring "V3's nonnegative projection" to remove "one negative ranking prediction in V5 and two
   in V6"; it was kept, with "No model or coefficient was retuned", and no separate gain is claimed for it.
   — <https://raw.githubusercontent.com/javidmardanov/PRC-Data-Challenge-2026/main/docs/autoresearch_attempts.md>

5. **Sign count on the touched rows. [S]** Phoenix-Ops report "18 squared errors improve and 13 worsen" on
   their 31 rows; our own item-13 reports "49 helped / 4 hurt" at cap 3,600 and "24 helped / 0 hurt" at cap
   7,200. **[X]** A 58 % hit rate (Phoenix) yields a gain only because the winners are larger than the
   losers; a 92–100 % hit rate (our cap) is a qualitatively different object. Publish the hit rate before
   shipping.

**[X] A usable decision rule, derived from §0 rather than sourced:** before shipping an overlay, compute
`Σ_rows [(p−r)·(p+r−2ŷ)]` under the *most pessimistic plausible* `ŷ` for each touched row (i.e. assume the
rule is wrong on that row and the model was right). If the pessimistic total is worse than −ΔSSE needed for
the claimed gain, the rule is a bet, not an improvement. With our numbers, an overlay must produce at least
2e7 of ΔSSE to be worth 0.1 s, and a *single* row that moves 10,000 s in the wrong direction costs 1e8 ≈
0.5 s. So: **an overlay whose per-row movement is under ~3,000 s cannot pay for itself at any row count
under ~200, and an overlay with any plausible wrong-sign row moving >10,000 s is dominated by its downside.**

---

## Sources

- PRC Data Challenge 2026 — <https://prc-data-challenge-2026.netlify.app/>, <https://prc-data-challenge-2026.netlify.app/data.html>
- team `kind-mango` — <https://github.com/skylinkapi/prc-data-challenge-2026-kind-mango>,
  <https://raw.githubusercontent.com/skylinkapi/prc-data-challenge-2026-kind-mango/main/RECAP.md>,
  <https://raw.githubusercontent.com/skylinkapi/prc-data-challenge-2026-kind-mango/main/docs/MODEL_ANALYSIS.md>
- team `zestful-fountain` / Phoenix-Ops — <https://github.com/Phoenix-Ops-LTD/prc2026-taxiout>,
  <https://raw.githubusercontent.com/Phoenix-Ops-LTD/prc2026-taxiout/main/METHOD.md>
- OpenAir — <https://github.com/ChandanHegde07/OpenAir>
- javidmardanov — <https://github.com/javidmardanov/PRC-Data-Challenge-2026>,
  <https://raw.githubusercontent.com/javidmardanov/PRC-Data-Challenge-2026/main/docs/autoresearch_attempts.md>
- ahmetabdullahgultekin — <https://github.com/ahmetabdullahgultekin/prc-taxiout-2026>
- Rossmann 1st place model documentation — <https://storage.googleapis.com/kaggle-forum-message-attachments/102102/3454/Rossmann_nr1_doc.pdf>
- Rossmann top-10% write-up — <https://mabrek.github.io/blog/kaggle-forecasting/>
- Elo Merchant Category Recommendation notes — <https://github.com/bangdasun/kaggle/blob/master/elo-merchant-category-recommendation/README.md>
- Walmart Store Sales 1st place — <https://github.com/davidthaler/Walmart_competition_code>
- Data Science Bowl 2019 top-solution summaries — <https://naotaka1128.hatenadiary.jp/entry/dsb-2019-top-solution>, <https://takaishikawa42.hatenablog.com/entry/2020/01/29/090852>
- scikit-learn cross-validation — <https://scikit-learn.org/stable/modules/cross_validation.html>
- scikit-learn calibration — <https://scikit-learn.org/stable/modules/calibration.html>
- Adversarial validation — <https://medium.com/@nlztrk/adversarial-validation-a-sanity-checker-and-an-exploiter-2dff1baced19>, <https://arxiv.org/pdf/2112.10078>
- EUROCONTROL additional taxi-out time — <https://ansperformance.eu/reference/dataset/taxi-out-additional-time/>, <https://www.eurocontrol.int/prudata/dashboard/metadata/additional-taxi-out-time/>
- Bojer & Meldgaard, *Learnings from Kaggle's Forecasting Competitions* — <https://arxiv.org/pdf/2009.07701>
- Blum & Hardt, *The Ladder* — <https://arxiv.org/abs/1502.04585>
- Review of PRC 2024 winners — <https://conormclaughlin.net/2024/12/reviewing-the-winning-ml-models-from-the-opensky-prc-data-challenge-what-did-they-do-differently/>

---

## Applicable to `resilient-hedgehog`

Ordered by evidence strength.

### 1. Per-airport upper bound on non-LIRF predictions, chosen from the 2025 label support — STRONG

**Evidence [S].** Two independent measurements of the same rule:
- Our own OOF (`docs/investigation_2026-09-16/results/tail-postprocessing-cap-shrink.md`, verdict (d)
  CONFIRMED): cap non-LIRF unmatched at 3,600 → total OOF RMSE **334.58 → 327.05**, 53 rows, 49 helped /
  4 hurt, ΔSSE negative in **all four** OOF months and **all eight** affected airports; at 7,200 → 329.33,
  24 rows, **24 helped / 0 hurt**, every one of those 24 rows has a label below 1,744 s.
- `kind-mango` shipped the per-airport-support version of this rule on the *same scored set* and gained
  **−2.33 s live** on 5 rows, with a **0 MSE** fold price.

**Measurement that would settle it.** Label-free, computable today, no training:
(a) on the current submission parquet, count predictions above the 2025 per-airport maximum of the
non-fallback class, split LIRF / non-LIRF and by airport; (b) for each such row, record `mvt_eobt1` (or our
equivalent long-hold evidence) and the airport's 2025 support; (c) compute ΔSSE under the worst case that
*every* capped row's true label equals the capped value's complement, i.e. bound the downside; (d) ship as
a single-change upload with the pre-registered price from (a)–(c) and compare the realised delta. Our §3.9
numbers say the current-generation files had 100 (v7) / 83 (v9) unmatched predictions > 7,200 of which 90 /
81 were LIRF — so the non-LIRF count is ~2–10 rows, the same order as `kind-mango`'s five.

**Risk, honestly.** The OOF evidence rests on 24–53 rows in four months of 2025 and cannot see 2026. The
sign is uniform across months and airports, which is the strongest thing we have, but uniform-sign on 24
rows is still 24 rows. The specific risk is that a 2026 non-LIRF row is a genuine multi-hour taxi (de-icing
queue, EHAM Jan disruption — we have a separate finding on that) and the cap turns a correct 10,000 s
prediction into a 7,200 s one, costing ~0.04 s. That is small precisely because the cap can only move a
prediction toward the bulk. **A cap is the one overlay family with bounded downside.**

### 2. Do NOT fit any calibration (isotonic, per-group shift, distribution match) on `oof_main-48c19bc830.parquet` — STRONG (negative)

**Evidence [S].** sklearn: `cross_val_predict` output is "from several distinct models undistinguished" and
"is not an appropriate measure of generalization error"; only strategies that "assign all elements to a test
set exactly once" are admissible — ours assigns July twice and discards one copy
(`fold-shape-vs-ranking.md`). `kind-mango` measured "post-hoc airport-level shift calibration — near-zero
effect" on this competition's data. Published offline→live transfer ratios in this competition: 14 %,
negative, 35 %.

**Measurement that would settle it.** Fit the candidate correction separately on each of the four folds and
on each of the four covered months, and report the four/four sets of coefficients. If a per-airport additive
shift changes sign or magnitude by more than its own size between `jan` and `forward`, it is fold structure,
not bias. **[X]** My expectation is that it will, because the `jan` fold is a backcast.

**Risk.** The risk of *not* doing it is forgoing a gain `kind-mango` has already measured as near-zero. The
risk of doing it is a silent, everywhere-applied correction on 344,841 rows with no bounded downside — the
worst possible risk shape for RMSE.

### 3. Estimate the 2025→2026 level shift from the EUROCONTROL public series, as a *gate*, not a correction — MODERATE

**Evidence [S].** The Additional Taxi-Out Time dataset is monthly, per-airport, from the same APDF source as
our labels, with `TOTAL_ADD_TIME_MIN` and `TOTAL_REF_NB_FL`
(<https://ansperformance.eu/reference/dataset/taxi-out-additional-time/>). `kind-mango`'s MP6 arrival mirror
is the sourced precedent for using an observable 2026 series as a *ship/no-ship gate* ("a lever that helps
the 2025 folds and hurts 2026 arrivals does not ship"), and their v59 is the sourced precedent for *not*
turning such a series into a feature or a correction (+0.37 s).

**Measurement that would settle it.** Download the series; compute per-airport mean additional taxi-out for
2025-01, 2025-07, 2026-01, 2026-07; compare the implied per-airport year-over-year delta with the delta
between our submission's per-airport mean prediction and the 2025 per-airport mean label. If our predictions
already move in the same direction and magnitude, there is no level shift left to correct and item 3 closes.
If they move the opposite way at some airport, that airport is where to look — but look at features, not at
an additive patch.

**Risk.** The two quantities are not the same thing: "additional taxi-out" is taxi-out minus an unimpeded
reference, over APDF-validated flights, at ~80 airports; our target is raw `TAXITIME_SEC_mvt` over the
challenge's 344,841 departures. A per-airport difference between them is expected and does not license a
correction. Using it as a correction coefficient would be fitting to a proxy of a proxy. **[X]** I would use
it only to answer yes/no: "is 2026 systematically slower than 2025 at airport X".

### 4. Rule overlays on small row sets — WEAK, and our one experiment already says so

**Evidence [S].** Our v11 rollover-mixture overlay scored **293.978**, worse than the +8.377 floor implied by
applying it to zero rows — i.e. the 11 selected rows were not what we believed. The in-competition evidence
says the successful overlays are (a) bounded or mixture-form, (b) selected by a *data fact* (a clock
inequality, a calendar fact, a value outside the training support) rather than by a model score, and (c)
worth 0.2 s, not 5 s. Phoenix-Ops's 21-row overlay bought 0.2013 s with a 58 % per-row hit rate and 46 % of
the gain from one row. `kind-mango`'s classifier-selected hard swap cost **+253 s**.

**Measurement that would settle it.** For any candidate overlay, before upload: (i) the row count on the
actual ranking parquet, not on the holdout; (ii) the per-row `|p − r|` distribution — if the median is under
3,000 s, stop (§0 arithmetic); (iii) the hit rate and the signed ΔSSE on the holdout rows, plus the same
figures with the top-1 and top-3 contributing rows removed; (iv) the worst-case ΔSSE assuming the rule is
wrong on every row. Ship only if (iv) is survivable.

**Risk.** Unbounded. A full replacement has no upper bound on the damage, and our own 293.978 result is a
measured instance of the failure mode: the selection was wrong, not the arithmetic.

### 5. Shrinking extremes toward a conditional median — DO NOT, unless restricted to the non-LIRF class

**Evidence [S].** Refuted on our OOF at every λ and under all seven target definitions tried
(`tail-postprocessing-cap-shrink.md` §3.4–3.5, verdicts (b) and (c) REFUTED). OpenAir's 50 % blend is the
only in-competition instance and it is explicitly known to fail on December.

**Measurement that would settle it.** None needed to reject the general form. If someone wants to revisit
it, the only variant with a live hypothesis is "shrink *non-LIRF* unmatched predictions above the airport's
2025 support", which is the cap of item 1 in softer clothing — and the cap dominates it because the cap's
worst case is bounded by the cap value while a blend's is not.

**Risk.** Shrinking toward a median destroys the LIRF tail, which our own tables show the model currently
*under*-predicts (43 predictions > 4 h against 51 labels; 5 > 12 h against 15). The rows most tempting to
shrink are the rows we are already too conservative on.
