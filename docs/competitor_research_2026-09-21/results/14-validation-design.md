# Slot 14 — Validation design when train and test are separated in time

Research date 2026-09-21. Marking convention: **[S]** = sourced, URL given; **[X]** = my inference or
an unsourced claim. Nothing below is dressed up: where I looked for evidence and did not find it, I say so.

## 0. What I could and could not reach

- Reachable: the PRC 2024/2025/2026 challenge sites, the Journal of Open Aviation Science (JOAS) special
  issues for PRC 2024 and PRC 2025 (full HTML galleys), DrivenData blog posts, arXiv/NeurIPS PDFs,
  FastML, KDnuggets, the ML Contests annual reports.
- **Not reachable**: kaggle.com discussion and writeup pages (server-side content is JS-only; a plain
  fetch returns a 5 kB shell), and `web.archive.org` (blocked by this session's egress policy). So the
  canonical Kaggle winner writeups (IEEE-CIS, Amex, Home Credit CRMS, Optiver) could **not** be quoted
  first-hand. Where a Kaggle practice appears below it is quoted from a secondary source that names the
  competitor, and is marked as such. The WebSearch budget for this session was exhausted part-way
  through, so a few gaps stayed gaps rather than being chased further.

---

## 1. The comparison set, and how their splits are shaped

| Competition | Train period | Public/ranking set | Final set | Time-separated? |
|---|---|---|---|---|
| PRC 2024 (ATOW) | 2022 flights | random subset of 2022 | random subset of 2022 | **No** |
| PRC 2025 (fuel burn) | Apr–Aug 2025 | Sep 2025 | Oct 2025 | **Yes**, forward |
| PRC 2026 (taxi-out) | full year 2025 | Jan + Jul 2026 | *ranking set is the final set* | **Yes**, forward |
| DrivenData Pushback (NASA) | ~2 years, 10 US airports | "withheld validation set sampled from the same time range" | — | **No** (sampled from same range) |

**[S]** PRC 2025 split: "the dataset spans seven months: training data covers April through August 2025
(five months), the ranking evaluation set covers September 2025, and the final evaluation set covers
October 2025." — PRC 2025 editorial, JOAS.
<https://journals.open.tudelft.nl/joas/article/view/8750>

**[S]** PRC 2026: "All 10 airport movements for the full year of 2025" as training; ranked on "January and
July 2026 movements"; metric RMSE. <https://prc-data-challenge-2026.netlify.app/>

**[S]** PRC 2026 has *no* held-out private set announced: "Teams will be ranked based on the best Root
Mean Square Error (RMSE) achieved across their submissions."
<https://prc-data-challenge-2026.netlify.app/ranking.html>

**[S]** Pushback: "the Open Arena provides access to a training data set including 10 airports over
approximately two years, as well as feature data for a withheld validation set sampled from the same time
range." <https://www.drivendata.org/competitions/149/competition-nasa-airport-pushback/>

**[X]** This matters for how much of the competitor literature transfers. PRC 2024 and the NASA pushback
challenge are *not* time-separated, so the validation lessons from those two are about i.i.d. holdout
noise, not about shift. Only PRC 2025 is a like-for-like precedent for our situation, and it is a strong
one — same organiser, same metric family, forward split, published post-mortems.

---

## 2. What scheme did they actually use?

### 2.1 Plain KFold, even when the test set is later in time

**[S]** PRC 2025, Kačar & Ćulibrk (a ranked team, fuel burn): "Both XGBoost and LightGBM models were
trained using K-fold cross-validation to ensure robust generalization across the [dataset]"; they
"generated out-of-fold (OOF) predictions across the entire dataset using a 10-fold cross-validation
framework". No purging, no time blocking, despite the ranking set being a later month. Their reported
numbers: XGBoost K-fold CV RMSE 232.76 kg, LightGBM 243.91 kg; "The best submission to the challenge blind
test set scored 227.32 kg RMSE."
<https://journals.open.tudelft.nl/joas/article/view/8766>

**[S]** PRC 2025, Grigoriou et al. (physics-guided GBM): feature selection by "the largest reduction in
five-fold cross-validated mean squared error"; hyperparameters by "3-fold cross-validation on an 80%
training split"; final design decisions "guided by a 20% held-out validation set of 31,306 segments".
Again a *random* split, not a temporal one, against Sep/Oct test months.
<https://journals.open.tudelft.nl/joas/article/view/8764>

**[S]** ML Contests surveyed 2022 competition winners and counted their CV method: K-fold
cross-validation 30 competitions, fixed evaluation/hold-out set 17, "online single-fold" 1. "K-fold
cross-validation proves most popular despite computational burden." Variants named among winners:
"standard, stratified, purged, stratified in groups, and multi-label stratified."
<https://mlcontests.com/state-of-competitive-machine-learning-2022/>

**[X]** The honest reading: random K-fold remains the default even where the test set is in the future.
Teams that placed well in PRC 2025 did not use temporal CV. That is not an endorsement — see §2.4 for how
badly it tracked — but it is what the field does.

### 2.2 GroupKFold — by what group

I could not reach a first-hand Kaggle source for the canonical example (IEEE-CIS Fraud Detection, where the
1st place team is widely credited with GroupKFold grouped by *month*). The claim survives only in a
secondary account:

**[S, secondary]** "GroupKFold was used with month as the group for cross-validation, and fold one in
GroupKFold trained on months 13 through 17 and predicted month 12 (for early stopping purposes only),
while the model trained on months 13 through 17 also predicted test.csv which was forward in time." —
description of the IEEE-CIS 1st place (FraudSquad, incl. Chris Deotte) approach, surfaced from
<https://www.kaggle.com/competitions/ieee-fraud-detection/writeups/fraudsquad-1st-place-solution-part-2>
(the page itself would not render for me; the summary is from a secondary account of it).

**[X]** Note what the group *is* there: the group is the time block, not an entity. Grouping by month is
a time-based split wearing GroupKFold's clothes. The other common grouping — by entity, to stop the same
customer/aircraft/airport appearing on both sides — is a leakage control, a different problem from shift.
I found no sourced PRC-family account of entity-grouped folds.

### 2.3 Time-based splits and the single forward holdout — and a sourced case where they *lost*

The strongest sourced evidence I found is negative, twice:

**[S]** DrivenData Pushback, **1st place** (Team CDS, Dhond & Sáez): "Time based cross validation:
Averaging across folds based on time splits did not improve the performance."
<https://drivendata.co/blog/airport-pushback-finalists>

**[S]** Credit-scoring study on Lending Club data (train/test separated in time), Experimental Set 2 — a
single forward holdout. Best configuration was "using the data from 2018M7 to 2019M5 as the training set
and the 2019M6 data as the validation set, the testing set AUC reaches 0.7220. This result is lower than
using all the original data directly for 5-fold cross-validation, since the 2019M6 data, which is closest
to the testing set distribution, is only involved in the validation and not in training."
The plain 5-fold baseline on all data reached **0.7237**. <https://arxiv.org/pdf/2112.10078>

**[X]** That is the cost structure of a forward holdout stated cleanly: the block you hold out is exactly
the block most like the target, so you pay for the honesty of the estimate with the loss of the most
relevant training rows. It is an argument for *estimating* on a forward block while *fitting the shipped
model* on everything, not for shrinking the training set.

### 2.4 How well did CV track the leaderboard?

Sourced, quantified instances, best first:

**[S] PRC 2024 winner (ITA team, no time separation).** Local held-out 20% vs actual ranking-set RMSE:
ensemble 2210.37 → 2217.75; CatBoost 2215.53 → 2224.69; XGBoost 2413.05 → 2403.07; LightGBM 2506.43 →
2519.17. Their statement: "The actual RMSE values for each algorithm align closely with their estimated
test RMSE, indicating that the models generalize well to unseen data."
<https://journals.open.tudelft.nl/joas/article/view/7963>
**[X]** Two things follow. (i) Agreement is ±13 units on a ~2,200 scale, i.e. within ~0.6%. (ii) The
ensemble beat CatBoost locally by 5.2 and on the leaderboard by 6.9 — *smaller than the local-vs-actual
disagreement*. So even in the best-behaved case in this literature, the top-two ordering was inside the
noise. (Leading digits for the CatBoost/LightGBM/XGBoost rows are reconstructed from the article prose,
which states CatBoost's estimated test RMSE as 2215.53 and the ensemble's as 2210.37; the HTML table drops
the thousands digit.)

**[S] PRC 2025, Kačar & Ćulibrk (forward split).** CV 232.76 kg vs blind test 227.32 kg, and they refuse
to read anything into it: "The 4.9 kg gap between the XGBoost cross-validation estimate and the blind test
result is consistent with standard generalization variance at this dataset scale, but no formal confidence
interval over folds was reported, and the gap should not be interpreted as evidence of positive
out-of-distribution behavior." <https://journals.open.tudelft.nl/joas/article/view/8766>

**[S] PRC 2025, Grigoriou et al. — CV and the hidden set actively disagreed on which model to ship.**
"While the full-feature reference model achieves a marginally lower mean RMSE on validation, the proposed
SFS-tuned version demonstrates significantly better generalization on the hidden test set (RMSE 214.9 kg
versus 223.2 kg)". Validation means were 200.3 ± 13.4 (reference) vs 203.1 ± 14.1 (proposed).
<https://journals.open.tudelft.nl/joas/article/view/8764>
**[X]** Read the error bars: the validation difference is 2.8 kg against a seed SD of ~14 kg. The hidden
test difference is 8.3 kg, and it points the other way. Their validation protocol had no power to make
this call, and they only found out from the hidden set.

**[S] PRC 2025 organisers, on generalisation across months.** "Several teams that ranked well on one
month's data dropped or rose on the other, suggesting that generalization across time periods was a real
challenge." (Sep vs Oct evaluation months, 33 teams reaching the final stage.)
<https://journals.open.tudelft.nl/joas/article/view/8750>

**[S] DrivenData Pushback, 5th place (Syracuse):** time-series features "help to reduce the MAE on Open
Area but not on Prescreened Area" — the same change helping on one evaluation set and not another.
<https://drivendata.co/blog/airport-pushback-finalists>

**[S] Across ~120 Kaggle competitions:** the public→private gaps are small and mostly not adaptive
overfitting — "we surveyed 120 competitions on Kaggle covering a wide range of classification tasks but
found little to no signs of adaptive overfitting" — but the exceptions are exactly our situation: "we
have seen multiple competitions where a non-i.i.d. split lead to substantial gaps between public and
private scores, suggesting that distribution shifts may be a more pressing problem than test set re-use."
Outlier competitions "usually have pathologies such as non-i.i.d. data splits or (effectively) small test
sets." Roelofs et al., NeurIPS 2019.
<https://proceedings.neurips.cc/paper_files/paper/2019/file/ee39e503b6bedf0c98c388b7e8589aca-Paper.pdf>

---

## 3. Same fold structure for every experiment? Comparability across experiments

**[S] PRC 2024 winner, explicitly:** "We standardized parts of our training pipeline to ensure a fair
comparison between the machine learning methods. We separated the dataset into training and test sets with
an 80-20 split using the same random seed for every technique."
<https://journals.open.tudelft.nl/joas/article/view/7963>

**[S] PRC 2025, Grigoriou et al.:** the feature-selection search "is evaluated using an XGBoost model with
fixed hyperparameters so that the feature ranking reflects the predictive signal rather than tuning
artifacts"; and hyperparameter finalists are re-ranked on a *separate* holdout, because evaluating "many
trials on the same cross-validation folds" causes "optimization leakage".
<https://journals.open.tudelft.nl/joas/article/view/8764>

**[X]** These are the two halves of the comparability rule and they pull in opposite directions:
(a) hold the split fixed so that two experiments differ only in the thing you changed; (b) do *not* let the
same fixed split be the arbiter of hundreds of decisions, because then you have fitted the split. The
resolution used by the PRC 2025 team is two layers: a cheap fixed-fold layer for screening many
candidates, and a second, *untouched* holdout that only the finalists are scored on.

**[X] Directly relevant to our error.** A fold-structure rule that our own protocol breaks: if the OOF file
is a concatenation of predictions from three differently-trained models (jan, jul, forward), then a
statistic computed over the pooled file is not a statistic of any model. Nothing in the sourced literature
does this. The PRC 2025 teams' OOF files come from one fold structure with one model family; the PRC 2024
winner's comparisons come from one fixed 80/20 split. I found no sourced account of anyone pooling OOF
rows across heterogeneous fold definitions and then treating the pool as a single validation set.

---

## 4. Adversarial validation

### 4.1 What it is, and the diagnostic reading

**[S]** "training a probabilistic classifier to distinguish train/test examples"; if train and test match,
"it would perform no better than random. This would correspond to ROC AUC of 0.5." Santander example:
LR AUC 49.82%, RF 49.95% — indistinguishable. <https://fastml.com/adversarial-validation-part-one/>

### 4.2 Using it to *build the validation set* — the reported payoff

**[S]** Numerai example where train and test genuinely differ: the train/test classifier reached AUC 0.87.
Procedure: cross-validated predictions over all training rows, sort by probability of looking like test,
take the top N as the validation set. Result — validation log loss LR 0.6922, poly+LR 0.6958; private
leaderboard LR 0.6916, poly+LR 0.6954. The AV-built validation set predicted the leaderboard to within
0.0006 and 0.0004, *and preserved the ordering*.
<https://fastml.com/adversarial-validation-part-two/>

### 4.3 Using it to select features and to weight rows — the sourced verdict is mixed-to-negative

The Lending Club study ran 92 experiments across 5 designs on a time-separated split. Adversarial
classifier AUC **0.9681** ("much higher than 0.5, which indicates that the classifier can easily
distinguish the training data from the test data"). Baseline: plain 5-fold on all data, test AUC 0.7237.

| Design | Test AUC | vs baseline |
|---|---|---|
| Plain 5-fold, all data (baseline) | 0.7237 | — |
| Single forward holdout (best of 33 variants) | 0.7220 | **−0.0017** |
| **AV probability as sample weights**, all data | **0.7202** | **−0.0035** |
| Keep top-75% most test-like rows, 5-fold on those | 0.7315 | +0.0078 |
| **All rows train; validate only on top-40% most test-like** | **0.7327** | **+0.0090** |

**[S]** "changing only the sample weights without changing the sample selection does not effectively solve
the dataset bias problem." And on the winner: "data that are not consistent with the testing set
distribution can be added to the training process as well, but not involved in the validation. This will
allow the optimal accuracy of the testing set to be further improved, and the results obtained by choosing
different quantiles are more stable." <https://arxiv.org/pdf/2112.10078>

**[S]** On feature filtering: "Uber researchers have previously proposed a method that uses adversarial
validation to filter features to deal with dataset shift. However, there is a trade-off for this method
between the improvement of generalization performance and losing information." (same paper)

**[X]** So the sourced answer to "what does adversarial validation buy": as a *diagnostic* and as a
*validation-set constructor*, a lot (0.87/0.97 AUC tells you the naive split is lying; the AV-built
validation set tracked the leaderboard to 4 decimal places). As a *row-weighting* scheme, it measurably
hurt. As a *feature filter*, acknowledged as a trade. The single best design in that study — **train on
everything, validate only on the test-like rows** — is also the design that answers §2.3's objection,
because nothing is thrown out of training.

---

## 5. Handling a shift between the training year and the test year

Sourced practices, in decreasing order of how well they are evidenced:

1. **Measure the shift first.** AV AUC is the measurement; 0.5 means no shift, 0.87 and 0.97 are the two
   sourced "large shift" cases above. **[S]** FastML, arXiv 2112.10078.
2. **Move the validation distribution toward the test distribution without shrinking the training set.**
   The 0.7327 design above. **[S]** arXiv 2112.10078.
3. **Retrain the shipped model on all data after the design decisions are made.** **[S]** PRC 2025
   Grigoriou: "For test evaluation, all ML models are retrained on the full available training data."
   **[S]** PRC 2024 winner: "Before submitting the estimated ATOW in the submission dataset for evaluation
   by the Data Challenge server, we retrained each model using the whole challenge dataset".
4. **Prefer the smaller feature set when validation cannot separate two candidates.** **[S]** PRC 2025
   Grigoriou: SFS cut 133 candidate features to 74; marginally worse on validation, 8.3 kg better on the
   hidden set, which they attribute to the compact subset "mitigat[ing] overfitting".
5. **Expect ranks to move between test periods and don't tune against a single period.** **[S]** PRC 2025
   editorial (Sep vs Oct rank churn).
6. **Discount the target's scale dependence so the shift in the mix does not become a shift in the loss.**
   **[S]** PRC 2025 Grigoriou log-transformed the target, "preventing large-fuel widebody segments from
   dominating the gradient updates of early trees."

**[X]** I found *no* sourced account in the aviation-competition literature of explicit
importance-weighting, domain-adversarial training, or recency weighting being adopted by a placing team.
The only sourced quantification of row weighting (§4.3) is negative.

---

## 6. Fitting a linear correction between local CV and leaderboard score

**I could not find a source for this practice.** Marked **[X]** throughout.

What I did find is the raw material people would fit such a line to, and two cautions:

- **[S]** The PRC 2024 winner's table is exactly a CV↔LB paired dataset over six models, and the residuals
  are signed both ways (+9.16, +7.38, −9.98, +12.74, +0.1, +0.2), i.e. an offset-plus-noise picture with
  the noise comparable to the offset. <https://journals.open.tudelft.nl/joas/article/view/7963>
- **[S]** The PRC 2025 team who had a single CV↔blind pair explicitly declined to read a direction into
  it. <https://journals.open.tudelft.nl/joas/article/view/8766>
- **[S]** Roelofs et al. warn that when many submissions cluster, rank changes between two evaluation sets
  arise from noise alone: "the public scores are only approximately equal to the private scores (even in
  the absence of any adaptive overfitting), which can lead to substantial rank changes even though all
  score deviations are small". <https://proceedings.neurips.cc/paper_files/paper/2019/file/ee39e503b6bedf0c98c388b7e8589aca-Paper.pdf>

**[X]** A fitted CV→LB line needs paired points that differ by real structural changes, not by seeds, and
needs enough of them that the slope is identified against the per-point noise. With ~6 points and noise of
the same size as the spread, the fit is not identified. I would treat "CV↔LB linear correction" as
folklore until someone produces a published fit; the defensible weaker version is a *rank agreement* check
(did CV and LB order the same pairs the same way?), which is what the PRC 2024 winner actually reports.

---

## 7. The decision rule: is a change real or is it noise?

### 7.1 A principled, sourced threshold

**[S]** Blum & Hardt, *The Ladder*: the parameter-free rule accepts a claimed improvement only when it
beats the incumbent by more than **s/√n**, where s is "the sample standard deviation of loss differences"
and n the number of scored rows; otherwise the reported score is unchanged. The theory version uses
η = O(n^(−1/3) log^(1/3)(kn)). <https://ar5iv.labs.arxiv.org/html/1502.04585>

**[X]** This is a *paired* rule: it is computed on per-row loss *differences* between the new and old
predictions, not on the two scores separately. That is the right shape for our problem and it is cheap.
For the PRC 2026 ranking set (n ≈ 3.4 × 10⁵ scored rows per our repo docs), √n ≈ 587, so the threshold is
s/587 where s is the SD of the per-row squared-error difference between two submissions. The same formula
applied to a CV fold with far fewer rows gives a proportionally larger threshold — which is the honest
statement that a fold cannot resolve what the ranking set can.

### 7.2 Repeated seeds, and the noise floor people actually report

**[S]** PRC 2025 Grigoriou et al.: "On the validation set, to account for statistical variations from the
random seed split, metrics are reported as the mean and standard deviation across 5 random seeds (42, 43,
44, 45, and 46)." Observed SDs: Ridge 903.6 ± 81.1; RF 229.5 ± 12.6; LightGBM 235.6 ± 14.6; XGBoost
reference 200.3 ± 13.4; proposed 203.1 ± 14.1 kg.
<https://journals.open.tudelft.nl/joas/article/view/8764>

**[X]** The noise floor there is ~13–14 kg on a ~200 kg metric — about **7%** — and the model-to-model
differences they were choosing between were 2.8 kg, about **1.4%**, i.e. a fifth of one seed SD. They
reported both numbers and then let the hidden set decide. This is the only sourced noise-floor measurement
I found in the PRC family, and it is far coarser than people assume.

### 7.3 Two-stage screening so the screen is not the arbiter

**[S]** "To prevent overfitting to the cross-validation folds during hyperparameter optimization, a
two-stage tuning strategy was implemented. First, Optuna was executed using 3-fold cross-validation on an
80% training split ... Second, the top 10 hyperparameter configurations from this search were retrained on
the full 80% training split and evaluated on the remaining unseen 20% validation holdout split. The single
parameter configuration achieving the lowest validation RMSE on this holdout set was selected ... This
two-stage workflow ensures that the final parameters generalize well to unseen data by mitigating
optimization leakage that can arise from evaluating many trials on the same cross-validation folds."
<https://journals.open.tudelft.nl/joas/article/view/8764>

### 7.4 What the field says about small differences generally

**[S]** Winner's public-LB rank distribution across 2022 competitions: the #1 private spot went to the #1
public team in 43% of competitions, and the private winner came from the public top 5 in over 80% of
cases — "but there's a long tail". Cited extremes: Jigsaw Toxicity winner was #1853 on public;
Blood Stroke Clot winner #471; Great Barrier Reef winner #121.
<https://mlcontests.com/state-of-competitive-machine-learning-2022/>

**[S]** Effect sizes of potential overfitting across 120 Kaggle competitions "are typically small (e.g.,
less than 1% classification accuracy)". Roelofs et al. (link above).

**[X]** None of the sourced accounts describe a numeric "adopt if the improvement exceeds X" rule of the
kind we have. The nearest things are (a) the Ladder's s/√n, (b) the PRC 2025 team's mean ± SD over 5 seeds
with the decision then deferred to a second holdout, and (c) the two-stage screen/confirm structure. The
common shape is **screen cheaply, confirm on a set the screen never touched** — not a single test made
sharper.

---

## 8. CV reliability when a handful of extreme rows dominate the squared error

This is the least-written-about item, but there is real sourced material.

**[S] The heterogeneity is enormous and is documented.** PRC 2024 winner's per-aircraft-type RMSE for the
same model: B752 **11,893**, A359 5,206, B77W 5,457 — against A319 **662**, AT76 746, CRJ9 820. An
18× spread across segments of one test set.
<https://journals.open.tudelft.nl/joas/article/view/7963>

**[S] Practitioners name the domination explicitly and act on the target, not the metric.** PRC 2025
Grigoriou: widebodies are 18.7% of segments, "yet their high per-segment fuel burns mean that prediction
errors on these segments contribute disproportionately to overall RMSE"; their responses were (i) a
log-transform of the target, "preventing large-fuel widebody segments from dominating the gradient updates
of early trees", and (ii) synthetic over-sampling of the dominating class from 18.7% to 34.1% of rows.
<https://journals.open.tudelft.nl/joas/article/view/8764>

**[S] Per-segment metrics as the reporting unit, on *both* validation and the hidden set.** The same paper
reports RMSE/MAE broken out by flight phase (climb/cruise/descent), by aircraft type, and by haul
category, side by side for validation and hidden test, and uses the comparison to detect where
generalisation fails: "For most types, validation and test metrics are broadly consistent, which confirms
that the model generalizes across the fleet. The B77W is a notable exception."
<https://journals.open.tudelft.nl/joas/article/view/8764>

**[S] Trimming for training, sourced.** DrivenData Pushback 3rd place (Oracle 2) "identified outliers
using error thresholds before training final models", filtering samples with "absolute error greater than
20 or 30" minutes. <https://drivendata.co/blog/airport-pushback-finalists>

**[S] The gap the literature itself admits.** Roelofs et al., in future work: "Our analysis here focused
on classification competitions, but Kaggle also hosts many regression competitions. Is there more adaptive
overfitting in regression? Answering this questions will likely require access to the individual
predictions of the Kaggle submissions to appropriately handle **outlier submissions**."
<https://proceedings.neurips.cc/paper_files/paper/2019/file/ee39e503b6bedf0c98c388b7e8589aca-Paper.pdf>

**[S] A sourced criticism of unsegmented RMSE, written by a PRC 2025 team about their own work:** "the
evaluation metric — RMSE in kg — does not distinguish whether residual error is distributed uniformly
across segment durations and aircraft types or concentrated in specific operational conditions such as
long-haul cruise segments or heavy wide-body types."
<https://journals.open.tudelft.nl/joas/article/view/8766>

**[X] What I did *not* find**, despite looking: no sourced competitor account of a paired bootstrap over
rows, a block bootstrap over time units, or a trimmed metric used as a *screening* statistic with the
untrimmed metric as the verdict. Our `src/prc/compare.py` (drop-K curve + month-level bootstrap +
both-sides-must-agree) appears to be ahead of the published practice on this specific point, not behind
it. The published practice is coarser: segment the *report* (§8, per-type/per-phase tables), trim the
*training set* (Oracle 2), or transform the *target* (Grigoriou) — and then let a second, untouched
evaluation set settle the argument.

---

## 9. Leaderboard probing

### 9.1 The PRC 2026 rule, verbatim

**[S]** From <https://prc-data-challenge-2026.netlify.app/ranking.html>, under a "Note / Please be aware"
callout, retrieved 2026-09-21:

> "We will monitor submissions for attempts to learn from or exploit the ranking process. We consider such
> practices unfair and inconsistent with the purposes and goals of the Data Challenge."

Immediately relevant surrounding rules, verbatim from the same page:

> "Teams will be ranked based on the best Root Mean Square Error (RMSE) achieved across their submissions."

> "There is currently a limit of 5 submissions per day (and a total max of 1GB per bucket.)"

Note the second clause we already record internally as "unfair" is longer than our stored version: the
full sentence ends "…unfair **and inconsistent with the purposes and goals of the Data Challenge**." Our
`docs/investigation_2026-09-16/` copies truncate it at "unfair."

### 9.2 What the community understands the boundary to be

**[S] Probing to extract label information is a known, sometimes winning, tactic — and it is also
sometimes fatal.** ML Contests, 2023 report:
- GoDaddy Microbusiness Density Forecasting: "The ultimate winner used a linear regression model ... and
  got an edge from probing the public leaderboard to get more recent information about a few of the most
  volatile counties in the dataset — which helped them make better forecasts for those counties on later
  data."
- ICR Identifying Age-Related Conditions: "performance on the public leaderboard did not transfer to the
  final private leaderboard score — with the eventual winner jumping 2089 places". Chris Deotte, quoted:
  "using probed positive targets from public LB actually hurt my private LB score."

<https://mlcontests.com/state-of-competitive-machine-learning-2023/>

**[S] The formal characterisation of probing is the "boosting attack".** Blum & Hardt show a submission
strategy that extracts Ω(√(k/n)) of spurious leaderboard advantage from k submissions against an n-row
holdout, using only the returned scores — "Pick u₁,…,u_k ∈ {0,1}ⁿ uniformly at random. Observe loss
estimates l₁,…,l_k ... Output u* = maj({u_i : i ∈ I})". They describe existing defences as "poorly
understood heuristics such as limiting the bit precision of answers and the rate of re-submission" — which
is precisely what PRC 2026's 5/day cap is. <https://ar5iv.labs.arxiv.org/html/1502.04585>

**[X] Where the boundary sits, as I read it.** The sourced material supports a two-part distinction that
the PRC wording tracks:

- **Learning about your model** from a submission — "did change X move the score, and by how much" — is
  the ordinary use of any leaderboard, and PRC 2026 both publishes a score per submission and permits five
  a day. The scored set is also the *final* set here, so there is no private set to protect from
  contamination.
- **Learning about the labels** — constructing submissions whose returned RMSE decodes information about
  the held-out targets (constant-prediction probes to recover the target mean/variance, segment-isolating
  probes to recover per-airport or per-month means, binary-search probes on individual rows) — is what
  "learn from or exploit the ranking process" most naturally names, and is what the boosting attack
  formalises. The GoDaddy case is exactly this and was tolerated on Kaggle; PRC's wording is stricter than
  Kaggle's.

**[X]** Practical implication for us, since we have used probe submissions: an A/B submission pair that
differs by one modelling change, where the only thing read back is the RMSE difference, is a
*model* measurement. A submission whose predictions are chosen to make the returned RMSE informative about
the labels (constants, per-segment constants, deliberate offsets fitted across submissions) is a *label*
measurement, and the second kind is what the rule is aimed at. **[X]** I could find no PRC clarification,
FAQ answer, or Discord ruling that draws this line explicitly — the sentence above is the entire published
rule, so the boundary is not authoritatively defined. A team that wants certainty has to ask the
organisers; a team that wants safety should keep the count of submissions whose *predictions are not a
genuine model output* at zero.

---

## Applicable to resilient-hedgehog

Ordered by (buy ÷ cost). Each item names the sourced basis.

**A1. Stop pooling OOF rows across the jan/jul/forward folds for any statistic that decides adopt/reject.**
Compute the decision statistic *per fold* and require agreement, or run the comparison on one designated
fold. Basis: §3 — every sourced comparison in this literature is made inside one fixed fold structure with
one model; nobody pools heterogeneous models' OOF rows and calls it a validation set. We already know this
bit us once (a correction fitted on one fold's residuals applied to another).
*Cost:* a rewrite of the comparison entry point and a re-run of the last N accept/reject decisions to see
which flip; no retraining. *Buys:* removes a known protocol error; makes every future CI mean something.

**A2. Add the Ladder threshold as a second, cheap gate: adopt only if |Δ| > s/√n on paired per-row loss
differences.** Basis: §7.1 [S] Blum & Hardt. This is a *paired* rule and is exactly the shape our
`paired_bootstrap_rmse_diff` already computes the inputs for — it needs s, the SD of the per-row squared-error
difference, and n. *Cost:* ~20 lines, no new computation. *Buys:* a principled, published floor for "real"
that scales with n, replacing our implicit 0.25 s screening resolution with something defensible; and it
makes the fold-vs-ranking-set power difference explicit rather than assumed.

**A3. Report a 5-seed mean ± SD for any change we are about to adopt, and refuse to adopt when |Δ| is
under one seed SD.** Basis: §7.2 [S] PRC 2025 Grigoriou (seeds 42–46; SD ~13–14 kg on a ~200 kg metric,
~7%), where the 2.8 kg validation "win" was a fifth of an SD and pointed the wrong way against the hidden
set. *Cost:* 5× training per candidate — the dominant cost of this list. Mitigate by applying it only to
finalists (see A4). *Buys:* the single most likely explanation for our CV/LB disagreements is that we have
been adjudicating differences below our own seed noise; this measures the floor instead of guessing it.
Our note that 0.25 s screening resolution is "too fine to see structural wins" is the same observation
without the number.

**A4. Split the protocol into screen and confirm, and never let the screen be the arbiter.** Screen many
candidates cheaply on the `forward` fold alone (it is already the only honest forward fold, and
`train.py` records it as ~324k matched rows at a third of the wall clock); confirm the top handful on a
second evaluation that the screen never touched — the natural candidate is a re-run with new seeds plus
the `jul_fwd` fold, which validates the same month trained only on what precedes it. Basis: §7.3 [S] PRC
2025 two-stage tuning against "optimization leakage"; §3 [S] fixed-hyperparameter screening.
*Cost:* restructuring the experiment loop; roughly neutral on compute (cheaper screening pays for the
confirm stage). *Buys:* stops the accumulated selection pressure of hundreds of decisions from landing on
one fixed fold set, which is the mechanism behind "CV and LB disagreed repeatedly".

**A5. Fix the anti-causal folds, or at least stop treating them as equal evidence.** Our `jan` fold trains
on Feb–Dec 2025 and `jul` on everything but July — both train on the future of what they validate, while
the ranking set is strictly after all training data. `jul_fwd` exists and is honest; `jan` has no forward
twin by construction. Basis: §1/§2.3 — the target is a forward split, and the sourced failure mode (PRC
2025 Sep-vs-Oct rank churn [S]) is precisely forward generalisation. *Cost:* a fold-weighting or
reporting change; if we go further and drop the leaky folds we lose seasonal coverage, which is real —
see A6 for the cheaper fix. *Buys:* the folds we screen on stop measuring a quantity (interpolation) that
the ranking set does not score (extrapolation).

**A6. Build a test-like validation set with adversarial validation, without shrinking the training set.**
Train a classifier to separate 2025 training rows from the 2026 submission-template rows (features only —
no labels needed, and this uses no ranking feedback at all). Then adopt the design the Lending Club study
found best: *train every fold on all rows, but compute the validation metric only on the most test-like
quantile.* Basis: §4.3 [S] arXiv 2112.10078 — that design scored 0.7327 vs 0.7237 for plain 5-fold and
0.7220 for a single forward holdout; and the same study found AV-derived *sample weights* made things
worse (0.7202), so do not do that. §4.2 [S] FastML — an AV-built validation set tracked the private
leaderboard to within 0.0006 log loss.
*Cost:* one classifier fit plus a quantile mask in the metric; no retraining, no extra model.
*Buys:* first, a *number* — the AV AUC — telling us how far 2026 is from 2025, which we currently do not
know; second, a validation set whose distribution matches what is scored, which is the direct fix for
"our CV and the leaderboard disagree". If the AUC comes back near 0.5, we learn that shift is *not* our
problem and can stop spending on it.

**A7. Make per-segment metrics part of the adopt/reject report, not a separate investigation.** Report
Δ RMSE per airport and per month alongside the pooled Δ, on the same pair. Basis: §8 [S] PRC 2025
Grigoriou reports validation-vs-hidden metrics per phase, per aircraft type and per haul category and uses
the comparison to localise failure (B77W); [S] Kačar & Ćulibrk criticise their own unsegmented RMSE for
exactly this blindness; [S] PRC 2024 winner's per-type RMSE spans 662 → 11,893.
*Cost:* small — `compare.py` already computes a per-airport difference for the unmatched rows; this
generalises it to every comparison. *Buys:* the discriminator our current rule lacks. Our drop-200 gate
cannot tell "this change helps the tail structurally" from "this change won a tail lottery", and it
rejects both. Many airports and many months moving together is the structural signature; one airport's
handful of rows is the lottery. This is how we would finally see the 2–28 s structural wins competitors
report instead of screening at 0.25 s.

**A8. Consider a non-squared or log-scaled *training* objective while keeping RMSE as the reported
metric.** Basis: §8 [S] PRC 2025 Grigoriou log-transformed the target specifically to stop the dominating
class "dominating the gradient updates of early trees", and still reported RMSE; [S] DrivenData Oracle 2
trimmed training rows by error threshold. *Cost:* a training-objective experiment plus a calibration step
on back-transform (a log-target model is biased for the mean and needs correcting before an RMSE
submission); genuine risk of losing the tail we are scored on. *Buys:* if the 10 rows carrying 24–35% of
our squared error are also steering the fit, this decouples "what the model optimises" from "what a few
rows do to the metric" — which is the one lever the literature uses on this problem that we have not
pulled.

**A9. On probe submissions — record the distinction and stay on the model side of it.** Basis: §9. The
rule text is one sentence and defines no boundary, so we should hold ourselves to the conservative reading:
submissions whose predictions are a genuine model output, read back only as "did this change help", are
model measurements; any submission whose predictions are chosen to make the returned RMSE informative
about the labels is a label measurement and should not be sent. *Cost:* none, beyond an audit of what we
have already sent and a line in the docs. *Buys:* we are obliged to publish our repository at the end
([S] PRC 2026 site: final solutions must be openly shared on public GitHub), so the probe history will be
readable. Having the distinction written down, and having stayed on the right side of it, is worth more
than any probe. Also worth noting: PRC 2026 scores the *best* submission and has no private set, so the
usual Kaggle incentive to probe (buy public-LB position at the cost of private) does not even apply here —
[S] Roelofs et al. find adaptive overfitting is rarely the binding problem anyway, while distribution
shift is.

**[X] What I would *not* adopt**: a fitted linear CV→LB correction (§6 — no source, and not identifiable
from the handful of paired points anyone has published); AV-derived sample weights (§4.3 — sourced and
negative); and dropping to a single forward holdout as the primary validator (§2.3 — sourced and negative
twice, once by the Pushback winner and once on Lending Club).
