# Slot 13 — target transforms and loss choice for a heavy-tailed duration under RMSE

Researcher: investigation agent, 2026-09-21. Web research only; no code or config in this repository was
read for measurement purposes and nothing was modified.

Marking convention: **[S]** = sourced claim with a URL, stated as the source states it. **[X]** = my
inference, arithmetic, or opinion — not measured, not sourced. Where a source's evaluation metric is *not*
RMSE I say so explicitly in the same sentence, because the whole point of this slot is that the metric
decides the answer.

Our own facts carried in from the task brief are marked **[S-repo]** (measured in this repository, not
re-derived here).

---

## 0. The one fact that orders everything else

Our metric is RMSE. **[S]** The PRC Data Challenge 2026 states the task as "predict the taxi-out time of
flights from 10 major European airports" and the ranking as "Root Mean Square Error (RMSE) for submitted
predictions for the movements in January and July 2026."
<https://prc-data-challenge-2026.netlify.app/>

**[S]** Gneiting, *Making and Evaluating Point Forecasts* (arXiv:0912.0902), establishes the framework that
settles most of this slot. From the abstract, verbatim:

> "Effective point forecasting requires that the scoring function be specified ex ante, or that the
> forecaster receives a directive in the form of a statistical functional, such as the mean or a quantile
> of the predictive distribution. […] A functional is elicitable if there exists a scoring function that is
> strictly consistent for it. Expectations, ratios of expectations and quantiles are elicitable. For
> example, a scoring function is consistent for the mean functional if and only if it is a Bregman
> function. It is consistent for a quantile if and only if it is generalized piecewise linear."

<https://arxiv.org/abs/0912.0902>

Squared error is a Bregman function; the functional it elicits is the **conditional mean**. Absolute error
elicits the median. Quantile (pinball) loss elicits a quantile. Asymmetric squared loss elicits an
**expectile** **[S]** (Newey & Powell 1987, as described in
<https://arxiv.org/html/2510.05268> and
<https://search.r-project.org/CRAN/refmans/VGAM/html/amlnormal.html>).

**[X]** Consequence for us: *every* item on this slot's list is really the same question — "does this
change what functional the model estimates, and if so have I put the mean back?" Two options on the list
(ratio-against-a-reference, minus-a-group-mean) are **affine** given the group, so they leave the mean
functional intact and are safe in principle. Four of them (log/Box-Cox/Yeo-Johnson, winsorising,
quantile regression, tail-downweighting) deliberately estimate something that is *not* the conditional
mean, and are therefore a directional bet against the metric unless corrected.

**[S]** Florian Wilhelm states this rule explicitly: only affine transformations `y' = ay + b` preserve the
expected value under `E[φ(X)] = φ(E[X])`; log and Box-Cox violate it.
<https://florianwilhelm.info/2020/05/honey_i_shrunk_the_target_variable/>

**[X]** This is exactly why our own LIRF clock-offset result **[S-repo]** (subtract a per-airport clock
offset before fitting, add it back after; worth 20.7 s) worked and was *not* a fluke: it is an affine,
group-conditional shift. It is the safest member of the whole family. It is also why the robust-loss
result **[S-repo]** (recorded harmful) was predictable: Huber and quantile losses elicit something other
than the mean.

---

## 1. Option-by-option evidence

### 1.1 No transform (train L2 on the raw seconds)

**[S]** Nuyts & Davis, *The When and How of Target Variable Transformations* (arXiv:2504.20821), is the
closest thing to a controlled experiment on this exact question. They compare a no-transform baseline
against Quantile-Normal (QN), Quantile-Uniform (QU), Yeo-Johnson (YJ) and natural log (Ln) target
transforms, on 10 public regression datasets, with four learners (Lasso, Ridge, **Gradient Boosted Trees**,
SVR), under 5×2 cross-validation. Their headline squared-error metric is **RSE**, relative squared error,
`Σ(yᵢ−ŷᵢ)² / Σ(yᵢ−ȳ)²`, computed **on the original scale** — i.e. a monotone rescaling of MSE on a fixed
test set, so it ranks exactly as RMSE does. <https://arxiv.org/pdf/2504.20821>

Reported, verbatim: "The QN transformation significantly improves performance for Lasso, Ridge, and SVR,
yielding median improvements of between 11% and 63% for RSE […]. **In contrast, the GBTR seems to be
adversely affected by the normal distribution with a median increase in RSE of 19%**, and SMAPE of 4%."
And: "The advantages for the GBTR model are less clear: while the **median RSE increases with 6% for
Yeo-Johnson and 1% for the logarithm**, the median SMAPE decreases with 2% for both transformations."

Their per-dataset GBTR numbers (Table 2, RSE, lower is better) — the two most skewed datasets are the ones
that matter to us:

| GBTR dataset | skew γ | Base | QN | QU | YJ | Ln |
|---|---|---|---|---|---|---|
| Online News Popularity (ONP) | **33.963** | **11.281** | 25.422 | 29.567 | 42.679 | **47.789** |
| Bike Sharing (BS) | 1.277 | **0.206** | 0.286 | 0.342 | 0.251 | 0.282 |
| Servo (SRV) | 1.775 | **0.197** | 0.238 | 0.226 | 0.391 | 0.199 |
| Liver Disorders (LD) | 1.537 | 1.933 | **1.417** | 1.938 | 2.056 | 2.088 |

**[S]** On the single most heavy-tailed target in their suite (ONP, skewness 34), a log transform made
original-scale squared error **4.2× worse** for gradient-boosted trees (11.281 → 47.789), and
Yeo-Johnson 3.8× worse.

**Failure mode of "no transform":** **[S]** the same paper's Heuristic 5 is that heteroscedastic or
imbalanced targets are genuinely harder to learn, and for *linear and kernel* models the transform is a
large win (Lasso/Ridge/SVR improve by 11–63% median RSE). **[X]** So "no transform" is only the right
default because we use a tree ensemble; it is not a universal law, and it would be the wrong default if we
added a linear or SVR component to a blend on the raw target.

### 1.2 log1p / Box-Cox / Yeo-Johnson, and the back-transform correction

**The trap, stated precisely. [S]** Wilhelm shows that minimising MSE on `log y` and exponentiating back
recovers (approximately) the **median** of the original distribution, by quantile equivariance, not the
mean. His worked example on a used-car price dataset with an intercept-only model:

- MSE minimised on raw prices → 6,704 (the mean)
- MSE minimised on `log` prices, exponentiated back → 4,064 (approximately the median)

i.e. the naive back-transform is **39% low**. He gives the corrections to add *before* back-transforming,
by target metric: **RMSE → +½σ²**; MAE → 0; MAPE → −σ²; RMSPE → −(3/2)σ². His categorical warning:
"Applying any non-affine transformation to your target variable might have unwanted effects on the error
measure you are minimizing. So if you don't know exactly what you are doing, just don't." He reports
log-transformation *worsening* original-scale RMSE by 2–4% in his experiments.
<https://florianwilhelm.info/2020/05/honey_i_shrunk_the_target_variable/>

**The classical correction. [S]** The parametric (log-normal) correction is `E[y|x] = exp(f(x)) · exp(½σ²)`.
<https://en.wikipedia.org/wiki/Smearing_retransformation>
**[S]** The nonparametric alternative is Duan's smearing estimate (Duan 1983, *JASA* 78(383)), "a
nonparametric estimate of the expected response on the untransformed scale after fitting a linear
regression model on a transformed scale […] consistent under mild regularity conditions".
<https://www.tandfonline.com/doi/abs/10.1080/01621459.1983.10478017>,
<https://people.stat.sc.edu/hoyen/PastTeaching/STAT704-2022/Notes/Smearing.pdf>

**Why the correction is not a free fix. [S]** Manning & Mullahy, *Estimating log models: to transform or
not to transform?* (J. Health Econ. 20(4), 2001): least-squares-on-log methods "can be biased in the face of
heteroscedasticity if not appropriately retransformed"; "no single alternative is best under all conditions
examined"; and picking a less appropriate estimator can cost precision "equivalent to losing one half or
more of one's sample."
<https://www.sciencedirect.com/science/article/abs/pii/S0167629601000868>,
<https://www.nber.org/papers/t0246>
**[X]** The practical reading: the `exp(½σ²)` factor is only right if σ² is constant in x. At our airports
σ² is emphatically *not* constant in x (LIRF sd ≈ 1,332 s vs the pooled mean ≈ 1,000 s **[S-repo]**), so a
single global smearing factor would over-correct the quiet stands and under-correct the congested ones.

**Practical size of the correction on our numbers. [X] (arithmetic on [S-repo] figures, not a measurement.)**
If the LIRF target were log-normal with mean 1,000 s and sd 1,332 s, then cv = 1.332,
σ² = ln(1+cv²) = ln(2.775) = 1.021, and the multiplicative correction is `exp(σ²/2) = 1.67`. The naive
`expm1` back-transform would therefore sit **~40% below the conditional mean** at LIRF. On a target whose
mean is ~1,000 s, that is a systematic ~400 s under-prediction at the worst airport. This is an order of
magnitude larger than the 20.7 s that our proven LIRF reparameterisation bought **[S-repo]** — i.e. the
correction is not a rounding error, it dominates.

**An alternative that sidesteps the issue entirely. [S]** More (Amazon), *Identifying and Overcoming
Transformation Bias in Forecasting Models* (arXiv:2208.12264): log and sqrt target transforms "introduce a
systematic negative bias (under-forecasting)" that persists "irrespective of the model type (XGBoost,
Random Forest, Linear Regression, Neural Networks, and ARIMAX)"; two bias-correction methods "improve model
performance (by up to 50%)"; and Tweedie-family cost functions, which model the mean directly through a log
link rather than transforming the target, "gives the best performance so far when modeling on sales".
Variance power 1.1–1.3 was best in most cases. <https://arxiv.org/pdf/2208.12264>
**Caveat, stated because it matters: [S]** this paper's reported metrics are **WMAPE and WBias, not RMSE**.
Their finding that "the model built on actual sales with MSE as cost function performs very poorly
consistently" is a statement about MAPE/bias, and MAPE is minimised by something much closer to the median
than the mean — so it does **not** transfer to our metric.
**[S]** The general claim that a log-link GLM/Tweedie models `μ = E(Y|X=x) = exp(xᵀβ)` directly and so
"avoids the bias introduced by log or square-root transformations of the response variable" is standard.
<https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html>,
<https://ar5iv.labs.arxiv.org/html/1608.04910>

**The clean contrast case, to keep the RMSE/RMSLE distinction sharp. [S]** Kaggle *New York City Taxi Trip
Duration* is a heavy-tailed positive duration target, exactly our shape, and the winning practice there was
to train on `log1p(trip_duration)` with plain MSE — because the competition metric is **RMSLE**, not RMSE:
`ε = √(1/n Σ(log(pᵢ+1) − log(aᵢ+1))²)`. Optimising MSE on the log target *is* the metric there.
<https://www.kaggle.com/c/nyc-taxi-trip-duration>
**[X]** This is the single most likely source of a wrong transfer into our problem: the two competitions
look identical (taxi, duration, seconds, right skew) and their metrics point in opposite directions.

### 1.3 Predicting a ratio or residual against a physical / unimpeded reference

**This is the strongest positive result in the whole slot, and it comes from our own competition's
predecessor.**

**[S]** Team `likable_jelly` won the PRC Data Challenge **2024** (RMSE 1,561 kg, vs 1,635 for 2nd and 2,218
for 3rd) predicting **not** the raw target but a physically normalised ratio:
`(TOW − EOW) / (MTOW − EOW)` — payload as a fraction of maximum payload — then converting back with the
per-aircraft-type constants. The stated reason: it "avoids the training process to consume a lot of splits
of aircraft_type and wtc just to get the mass range right," and keeps the modelled quantity "roughly the
same and mostly inside [0.2, 1]."
<https://github.com/PRC-Data-Challenge-2024/team_likable_jelly>,
<https://conormclaughlin.net/2024/12/reviewing-the-winning-ml-models-from-the-opensky-prc-data-challenge-what-did-they-do-differently/>

**And crucially, they corrected for the metric. [S]** They acknowledge "a 10% relative error on an A320 or
a A343 does not produce the same absolute mass error", and therefore "to optimize the root mean square
error on the absolute mass, LightGBM was trained with a vector of weights equal to the squared scaling
term: `(MTOW − EOW)**2`."
<https://github.com/PRC-Data-Challenge-2024/team_likable_jelly>

**[X]** This is the general recipe and it is exactly transferable. If you divide the target by a known
per-row constant `s_i`, squared error on the scaled target is `(1/s_i²)` times squared error on the raw
target, so to keep minimising raw-scale RMSE you must train with sample weight `s_i²`. If you only
*subtract* a per-row constant (`s_i = 1`), no weighting is needed at all — the transform is a pure shift
and squared error is unchanged. Shift-only is therefore strictly safer than ratio.

**Reported failure mode: [S]** the write-up reports **no ablation** — there is no published measurement of
what the ratio target alone bought relative to an unscaled target. The RMSE 1,561–1,564 figures are for the
final ensemble, with no unscaled baseline given. So the gain is *attested by a winner* but **not
quantified**.

**[S]** Nuyts & Davis give the same idea a name and two general forms (their Heuristic 4, "General
Contextual Dependency"): subtract the conditional expectation and divide by the conditional spread,
`f(yᵢ) = (yᵢ − E(yᵢ|φ)) / E(σᵢ|φ)`, or divide by a fitted regression baseline,
`f(yᵢ) = yᵢ / (β₀ + Σβⱼφᵢⱼ)`. They also cite the ship-arrival-at-a-lock case: predicting the *travel time*
rather than the arrival time, because arrival time "strongly depends on the departure time".
<https://arxiv.org/pdf/2504.20821>

### 1.4 Predicting the target minus a group mean / median

**[S]** The Netflix Prize is the best-documented case of this under an RMSE metric, and the numbers are
large. Koren, *The BellKor Solution to the Netflix Grand Prize* (2009), reports RMSE on the Quiz set for
pure baseline predictors — i.e. global mean plus additive group effects, with no interaction model at all:

| Baseline predictor | Quiz RMSE |
|---|---|
| μ + user bias (incl. time drift) + item bias + item-time-bin bias, eq. (9) | **0.9605** |
| …plus multiplicative user-scale term c_u(t), eq. (10) | **0.9555** |
| …plus item-bias-per-log-frequency term b_{i,f}, eq. (11) | **0.9278** |
| Netflix's own Cinematch system, same Quiz set | 0.9514 |

Koren: "this basic model, which captures just main effects disregarding user-item interactions, can explain
almost as much of the data variability as the commercial Netflix Cinematch recommender system"; and adding
the frequency term "is quite dramatic. RMSE drops from 0.9555 to 0.9278." He also states the purpose of the
offset explicitly: it "is capturing the portion of the data that is less relevant […] and in doing so
enables deriving accurate recommendations by subjecting other models to cleaner data."
<https://www2.seas.gwu.edu/~simhaweb/champalg/cf/papers/KorenBellKor2009.pdf>

**[S]** Nuyts & Davis give the same transform as their Heuristic 1 remedy — centre the target on the
subject's own mean, `f(yᵢⱼ) = yᵢⱼ − ȳᵢ` — "as was done for Netflix ratings".
<https://arxiv.org/pdf/2504.20821>

**Reported failure modes.**
- **[S]** Koren's own framing: a baseline predictor "no matter how accurate, cannot yield personalized
  recommendations on its own". **[X]** Translated: the offset removes signal from the residual, so the
  residual model must be refitted, and the offset has to be estimated out-of-fold or it leaks.
- **[S]** John Myles White, on trimming/winsorising thresholds, makes the group-estimation point that
  applies equally to group offsets: "The only way to prevent bias is to compute outlier thresholds in
  Groups A and B separately."
  <https://www.johnmyleswhite.com/notebook/2015/12/03/some-observations-on-winsorization-and-trimming/>
- **[X]** For a *mean* offset under RMSE the transform is affine and exactly metric-neutral. For a *median*
  offset it is still affine (a constant is a constant), so still metric-neutral — the median is only a worse
  choice if it estimates the group location less well, not because of any Jensen effect. Small groups are
  the real risk: a per-(stand, runway) mean over 8 flights is mostly noise, and subtracting noise adds
  variance to the target.

### 1.5 Winsorising / clipping the training target; clipping predictions at inference

**Clipping predictions to a known support. [S]** Kaggle *Predict Future Sales* is an RMSE competition on a
heavy-tailed count where the organisers themselves clip: "submissions are evaluated by root mean squared
error (RMSE) […] **true target values are clipped into [0,20] range**", and the established participant
practice is to clip predictions to the same range.
<https://www.kaggle.com/c/competitive-data-science-predict-future-sales>
**[X]** This is a special case and does not generalise: clipping predictions is guaranteed non-harmful in
squared error *only* when the truth is known to lie in the clip range, because projecting onto a convex set
containing the truth cannot increase distance to it. Our taxi-out truth has no known upper bound, so an
upper clip is a real bet. A lower clip at 0 is free (durations are non-negative).

**Winsorising / removing the training target's tail. [S]** In Kaggle *Mercedes-Benz Greener Manufacturing*
— a squared-error-family metric (R² = 1 − MSE/Var) on a duration in seconds (mean ≈ 100 s) with a single
extreme value at y = 265.32 — a widely reproduced community practice was to treat points above 150 as
outliers and drop them from the training set.
<https://medium.com/analytics-vidhya/detailed-solution-to-mercedes-benz-green-manufacturing-competition-on-kaggle-7b85c84a3ff5>,
<https://www.kaggle.com/c/mercedes-benz-greener-manufacturing>
**Caveat: [X]** these are community write-ups, not an official 1st-place solution; I did not find a
first-party winner's write-up attributing a measured gain to the outlier removal. Treat as practice, not as
a measured effect.

**Reported failure mode. [S]** John Myles White: "Both winsorization and trimming can improve the quality of
point estimates […]. But they can make inferences worse because they can introduce bias into these point
estimates," and choosing an inappropriate outlier window "makes both methods inferior to making no
adjustment at all," with the difficulty that you cannot choose the window without knowing the truth.
<https://www.johnmyleswhite.com/notebook/2015/12/03/some-observations-on-winsorization-and-trimming/>
**[S]** Winsorisation is also criticised as "not as effective as compared to outlier removal because, by
hard replacement, inaccuracies may be introduced into the data."
<https://blog.trainindata.com/winsorization-handling-outliers-in-machine-learning/>

**[X]** For our problem this is close to a settled question already. Winsorising the training target is a
deliberate downward bias on the conditional mean of exactly the rows that carry 51.5% of our OOF squared
error **[S-repo]**. It is the same bet as a robust loss, which we already measured as harmful **[S-repo]**,
implemented differently. Prediction clipping is separately measured in this repository
(`results/tail-postprocessing-cap-shrink.md`, referenced from the robust-loss report) and I have not
re-derived it.

### 1.6 Quantile / expectile regression, and a non-median quantile as the point estimate

**[S]** Gneiting: pinball loss is strictly consistent for a quantile, asymmetric squared loss for an
expectile, Bregman losses (including squared error) for the mean. Using a quantile forecast when the
scoring function is squared error is precisely the mismatch the paper warns about: "this common practice
can lead to grossly misguided inferences, unless the scoring function and the forecasting task are
carefully matched." <https://arxiv.org/abs/0912.0902>

**[S]** Expectile regression (Newey & Powell 1987) is the asymmetric-least-squares generalisation of mean
regression; at τ = 1/2 it **reduces exactly to standard mean regression / MSE**.
<https://arxiv.org/html/2510.05268>, <https://search.r-project.org/CRAN/refmans/VGAM/html/amlnormal.html>

**[X]** The implication for the "use a quantile other than the median" idea: for a right-skewed
conditional distribution the mean sits above the median, so *some* quantile level α\* > 0.5 will, at each
x, coincide with the conditional mean. But α\* is a function of the local shape of the conditional
distribution and is not constant across rows; tuning one global α is fitting a one-parameter approximation
to a per-row correction. Expectiles are the better-posed version of the same idea, and τ = 0.5 expectile
regression is just L2 — so the only thing a tuned expectile level can buy over plain L2 is a *systematic*
shift, which a tuned additive constant on the predictions would buy more cheaply. **[S-repo]** our own
quantile α = 0.5 fit was already measured and recorded as harmful.

**[S]** Multi-quantile approaches are used for extremes in other domains (e.g. precipitation downscaling,
<https://arxiv.org/html/2605.12762v1>), but those papers target the *distribution*, not a squared-error
point score.

### 1.7 Two-stage: classify "is this row in the tail", then regress separately

**[S]** Hurdle / two-part models are the standard formalisation: "Hurdle models assume all zeros arise from
a single 'non-participation' process, and once you cross the hurdle, you're in the positive regime."
<https://towardsdatascience.com/two-stage-hurdle-models-predicting-zero-inflated-outcomes/>
**[S]** A two-fold classification-then-regression approach for zero-inflated data reports accuracy/F-measure
for the classifier and RMSE for the regressor. <https://arxiv.org/pdf/2310.08088>
**[S]** In the extremes literature, a classification–regression fusion "explicitly addresses sample
imbalance […] followed by a regression stage that incorporates synthetic data to improve quantitative
estimation of heavy rainfall."
<https://www.sciencedirect.com/science/article/abs/pii/S0022169425022395>

**The one directly on-domain measurement I found, and it is negative. [S]** *Interpretable Taxi-Out Time
Prediction of Departure Flights Using Stacking Ensemble Learning and SHAP Analysis* (Sci. Rep., 2026;
Shenzhen Bao'an ZGSZ, runway 15, 5,952 departures after cleaning from 12,645 records, 26 May–8 Jun 2019)
compares, with the same stacking ensemble:

| Approach | MAPE | MAE | **RMSE** |
|---|---|---|---|
| Holistic (predict total taxi-out directly) | 10.6% | 99.7 s | **140.5 s** |
| Phase-segmented (separate models for unimpeded and dynamic components, summed) | 12.0% | 113.6 s | **156.7 s** |

The decomposed model is "slightly inferior" in accuracy but "significantly enhances model
interpretability"; the paper attributes the loss to cumulative prediction errors across the two stages.
<https://pmc.ncbi.nlm.nih.gov/articles/PMC13022428/>,
<https://www.nature.com/articles/s41598-026-40898-5>
**[X]** That is **+11.5% RMSE** for splitting the problem into two learned pieces. Note carefully what this
does and does not say: it tests *two learned models summed*, not *one model on an offset target*. It is
evidence against two-stage learning, not against subtracting a known constant.

**[X]** Generic failure mode of classify-then-regress under RMSE: the classifier's own error rate enters
the final prediction multiplicatively. At our tail rate — top 200 rows out of ~1.3 M is ~0.015%
**[S-repo]** — any realistic classifier has precision far below the level at which routing decisions help,
and a misrouted tail row gets the *bulk* model's prediction, which is the worst possible outcome for the
rows that dominate the score.

---

## 2. Air traffic specifically: the EUROCONTROL unimpeded taxi-out reference

### 2.1 What the methodology actually defines

**[S]** EUROCONTROL/PRU, *Additional Taxi-Out Time Performance Indicator*, Edition 00-04
(<https://ansperformance.eu/library/pru-tx-out-pi.pdf>). I read the PDF text directly. The definitions:

- **Grouping.** "The indicator is first calculated at disaggregated level, i.e. for a comparable grouping
  of flights with the same combination of **departure runway and stand**." On the largest airports —
  named as **EGLL, EHAM, LEMD, LFPG** — stands are replaced by **stand groups**, "virtual groupings of
  departure stands, defined from airport apron area maps […] so that their average unimpeded times have no
  statistical difference." Stand groups are calibrated by hand from airport maps (§5.1).
- **Actual taxi-out.** `AcTXOT(f) = ATOT(f) − AOBT(f)`, in minutes. Filter: `AcTXOT < 300 min`;
  helicopters excluded.
- **Congestion level per flight.** `seq(f) = ` number of departures *and* arrivals occurring between this
  flight's AOBT and its ATOT.
- **Peak hourly throughput per airport.** `hourlyMaxRate` from the truncated 20-minute window before each
  take-off, `= (n−1)/(tᵢ−fᵢ)`; `R_j = 90th percentile` of that over the reference sample.
- **First unimpeded estimate.** `U1(cᵢ) = 20th percentile` of `AcTXOT` over all flights in the grouping.
- **Saturation level.** `L(cᵢ) = round(R_j · U1(cᵢ) / 60)`.
- **Congestion limit.** `cl = 0.6` for major hubs, `0.5` for all other airports.
- **Unimpeded flight flag.** `f_u = 1` iff `seq(f) ≤ cl · L(cᵢ)`.
- **Unimpeded time.** `UTXOT(cᵢ) = median(AcTXOT)` over the unimpeded flights of the grouping, **only if
  the grouping has ≥ 10 unimpeded flights**, else `null`. Night flights excluded; day time is 06:00–22:00
  airport local, calibrated per airport.
- **Additional taxi-out.** `ATXOT(f) = AcTXOT(f) − UTXOT(cᵢ)`.
- **Reference period.** One full year (base year 2011 where available); recomputed monthly and compared
  against the reference; "a period of three months after the change is required before new robust
  unimpeded times references can be established."

**[X]** Three properties matter for us. (a) It is a **median over a filtered subsample**, not a mean —
so `UTXOT` is *not* an unbiased estimate of anything the RMSE metric elicits, but that is irrelevant if it
is used purely as a constant shift. (b) It is a **per-(runway, stand-group) constant**, refreshed only
quarterly-to-annually — so it is a static lookup table, exactly the shape of an offset. (c) It is
**undefined for thin groupings** (< 10 unimpeded flights), which in our data will be a non-trivial fraction
of (runway, stand) pairs and needs an explicit fallback.

### 2.2 The 20th-percentile and median-below-saturation conventions are standard

**[S]** NASA/Amblard et al., *Real-time Unimpeded Taxi Out Machine Learning Service* (Aviation 2021,
NTRS 20210017591) states the field's conventions: "one of the most employed estimates of the unimpeded taxi
time is to use the 20th percentile of the taxi time"; and describes the EUROCONTROL variant as "a function
of the airport congestion for a given stand-runway pair [that] assigns unimpeded taxi time as the median
value of taxi times that have a congestion below a saturation value."
<https://ntrs.nasa.gov/api/citations/20210017591/downloads/20210017591_Amblard_Aviation2021_paper.pdf>

**[S]** In that paper the ML model's *output* is the unimpeded taxi time itself — a replacement for the
statistical reference table, to be served in real time — not an offset applied to a total-taxi-out model.
Their accuracy is measured as "the STBO estimated unimpeded taxi time (our truth values) minus the
predictions from our ML model."

### 2.3 Has anyone used the unimpeded reference as a *target offset* rather than as a feature?

**Direct answer: I found no published work that subtracts a EUROCONTROL-style `UTXOT` constant from the
target, fits a model on the residual, and adds it back — and reports the RMSE delta versus fitting the raw
target.** What exists is adjacent but different:

1. **[S]** Predicting **additional taxi-out time as the modelling target in its own right** is published
   practice: "Taxi-out time is divided into unimpeded taxi-out time (UTT) and additional taxi-out time
   (ATT), where ATT equals the difference between total taxi-out time and UTT. The actual taxi-out time is
   subtracted from the unimpeded taxi-out time to obtain the historical additional taxi-out time of each
   flight." — *Additional Taxi-Out Time Prediction for Flights at Busy Airports by Fusing Flow Control
   Information*, Applied Sciences 14(21):9968, 2024. <https://doi.org/10.3390/app14219968>
   **Caveat: [X]** the MDPI full text returned HTTP 403 to every route I tried, so I could not read its
   Table 5 numbers, and critically **it reports no comparison against predicting total taxi-out directly** —
   because ATT *is* the quantity of interest there (slot allocation / performance monitoring), not an
   intermediate step towards total taxi-out. So it is not an ablation of our question.
2. **[S]** The decomposition-as-two-models version **was** ablated, and lost: holistic RMSE 140.5 s vs
   phase-segmented 156.7 s at Shenzhen (§1.7 above).
   <https://pmc.ncbi.nlm.nih.gov/articles/PMC13022428/>
3. **[S]** The closest *winning, RMSE-scored* analogue is the PRC 2024 ratio target (§1.3): a
   physically-derived per-row reference (EOW, MTOW), used as a target reparameterisation, with an explicit
   sample-weight correction so the optimisation still targets raw-scale RMSE. It won by a 74 kg margin
   (1,561 vs 1,635) — but with **no ablation isolating the target transform**.
   <https://github.com/PRC-Data-Challenge-2024/team_likable_jelly>

**[X]** My reading of the gap: ATM research treats `UTXOT` as an *output* (the reference table) or as the
*target* (when additional time is the quantity of interest), and ML-for-taxi-out research treats congestion
features as *inputs*. Using it as a pure offset on a total-taxi-out RMSE model is an underexplored
combination — which is a reason it might be worth something, and equally a reason there is no evidence it
is.

---

## 3. Applicable to resilient-hedgehog

Ordered by strength of sourced evidence. Each line names the exact measurement that would settle it. All
measurements below mean: OOF RMSE on the existing fold structure, pooled and per-airport, against the
current baseline of 284.3584, with the paired bootstrap already used in
`docs/investigation_2026-09-16/`.

**1 — Generalise the proven LIRF clock offset into a per-(airport, runway, stand-group) additive offset.
Strongest evidence, lowest risk.**
Evidence: [S] Koren — pure additive group offsets took Netflix Quiz RMSE from 0.9605 to 0.9278 and beat
Cinematch (0.9514) with no interaction model at all. [S] Nuyts & Davis Heuristic 1/4 name this transform
explicitly. [S] Wilhelm — an additive shift is affine, hence exactly mean-preserving, hence
metric-neutral by construction; **no bias correction is needed and no sample weights are needed**.
[S-repo] the affine per-airport clock offset already bought us 20.7 s at LIRF.
*Measurement that settles it:* fit the current model on `y − m_g` where `m_g` is the out-of-fold group
**mean** of `y` for group `g`, add `m_g` back, report OOF RMSE. Run the group definition ladder in one
sweep: `g` = airport; airport × runway; airport × runway × stand-group; airport × runway × stand-group ×
hour-of-day. Report per-level: OOF RMSE delta, the count of groups with < 30 training rows, and the RMSE
delta restricted to rows in those thin groups. Repeat once with the group **median** in place of the mean
to separate "location estimator" from "affine shift". Expected sign is neutral-to-positive; a *negative*
result at the finest level with a positive result at the coarsest is the signature of thin-group noise, and
the fix is shrinkage of `m_g` toward the parent level, not abandoning the idea.

**2 — Use the EUROCONTROL `UTXOT` per-(runway, stand-group) constant as that offset, computed exactly per
the PRU spec.** Evidence: [S] the spec is fully specified and I have transcribed it in §2.1 — 20th
percentile `U1`, `R_j` = 90th percentile of the 20-min truncated rate, `L = round(R_j·U1/60)`, `cl` = 0.6
hubs / 0.5 others, unimpeded iff `seq ≤ cl·L`, `UTXOT` = median over unimpeded flights, ≥ 10 flights, day
06:00–22:00. [S] The offset is a per-group constant, i.e. affine, i.e. metric-neutral. **[X] No sourced
evidence exists that anyone has used it this way** — §2.3. Weaker than item 1 only because it is unsourced
as a *target* use, not because it is riskier.
*Measurement that settles it:* implement `UTXOT` per the spec on our 2025 training data, out-of-fold;
report (a) coverage — fraction of rows whose (runway, stand) grouping reaches the ≥ 10-unimpeded-flight
threshold, per airport; (b) OOF RMSE of `y − UTXOT` refit vs baseline; (c) the same with `UTXOT` added
merely as a *feature* rather than an offset, to separate "the statistic carries signal" from "the offset
placement helps". If (c) captures the whole gain, the offset adds nothing over a feature and should be
dropped. Fallback for null `UTXOT` must be stated and measured: a parent-level median, not zero.

**3 — Do not log/Box-Cox/Yeo-Johnson the target; if tried at all, only with an explicit, group-conditional
mean correction.** Evidence: [S] Nuyts & Davis — for gradient-boosted trees the median original-scale RSE
*increases* by 19% (quantile-normal), 6% (Yeo-Johnson), 1% (log), and on their most skewed target
(skew 34) log made squared error **4.2× worse** (11.281 → 47.789). [S] Wilhelm — naive back-transform
recovers the median, 39% below the mean in his worked example; correction for RMSE is `+½σ²`.
[S] Manning & Mullahy — that correction is invalid under heteroscedasticity, which we have.
[X] Arithmetic on our numbers: a log-normal LIRF with mean 1,000 s and sd 1,332 s implies a back-transform
correction factor of **1.67**, i.e. the naive `expm1` sits ~40% below the conditional mean — 20× the size of
the gain our best target trick bought. [S-repo] our own log1p fit is already recorded as harmful.
*Measurement that settles it (only if we want the file closed):* three fits — (a) `log1p` + naive `expm1`;
(b) `log1p` + global smearing factor `mean(exp(resid))` estimated out-of-fold; (c) `log1p` + **per-airport**
smearing factor. Report OOF RMSE and, separately, mean signed residual per airport. Prediction: (a) badly
negative, (b) recovers most of it, (c) recovers more but still does not beat L2. If (c) *beats* baseline,
that is a genuinely new finding and overturns item 3.

**4 — Consider a Tweedie / Poisson / Gamma objective (log link, mean-preserving) as the *only* non-L2 loss
worth a fit.** Evidence: [S] a log-link GLM models `E(Y|X) = exp(xᵀβ)` directly and so has **no
retransformation bias** — unlike a log-transformed target. [S] More (Amazon) found Tweedie with variance
power 1.1–1.3 best among direct-on-target objectives. **Stated honestly: [S] that paper's metric is
WMAPE/WBias, not RMSE**, so it does not transfer; and [S] Gneiting's framework says Tweedie deviance is a
Bregman divergence and therefore *is* consistent for the mean — which is the real reason it is worth a fit,
not the Amazon result.
*Measurement that settles it:* LightGBM `objective: tweedie` at `tweedie_variance_power` ∈ {1.1, 1.3, 1.5},
plus `objective: poisson` and `objective: gamma`, each with `metric: l2` for early stopping so that the
stopping rule still tracks our score. Report OOF RMSE pooled and per-airport, and the mean signed residual
per airport (a log-link model that beats L2 on RMSE but has a large positive mean residual is a warning,
not a win). This is one sweep of five fits.

**5 — Do not winsorise the training target, and do not add a two-stage tail classifier.** Evidence:
[S-repo] the top 100 OOF rows carry 51.5% of total OOF squared error at 0.015% of rows — under a mean-
eliciting score these rows *are* the score. [S] Gneiting — any estimator that deliberately targets a
non-mean functional is inconsistent for squared error. [S] John Myles White — a badly chosen winsorisation
window is worse than no adjustment, and the window cannot be chosen without knowing the truth.
[S] Shenzhen — decomposing taxi-out into two learned stages cost **+11.5% RMSE** (140.5 → 156.7 s).
[S-repo] robust losses already measured harmful; prediction clipping already measured in
`results/tail-postprocessing-cap-shrink.md`.
*Measurement that would settle it if anyone insists:* train-target winsorisation at the 99.9/99.5/99.0
percentiles, OOF RMSE, plus the OOF RMSE restricted to the winsorised rows themselves. The second number is
the one that matters and I expect it to be catastrophic. I do not recommend spending a fit on this.

**6 — Do not use a tuned quantile or expectile level as the point estimate.** Evidence: [S] Gneiting —
pinball loss elicits a quantile, not the mean; using it under a squared-error score is the exact mismatch
the paper warns against. [S] Expectile regression at τ = 0.5 *is* MSE regression, so a tuned τ can only buy
a systematic shift. [S-repo] quantile α = 0.5 already measured harmful.
*Measurement that settles it more cheaply than a quantile sweep:* fit the existing L2 model, then find the
single additive constant `c` and single multiplicative constant `a` minimising OOF RMSE of `a·ŷ + c`. If
the optimal `(a, c)` is materially away from `(1, 0)` — say |c| > 5 s — then the model is miscalibrated in
the mean and there is a free affine fix; if it is not, then no quantile-level tuning can help either,
because a global shift is all such tuning can deliver, and this test costs zero fits.

**7 — If a *ratio* target is ever attempted (e.g. `y / UTXOT`), it must carry sample weights.** Evidence:
[S] PRC 2024 winner `likable_jelly` scaled the target by a physical constant and trained LightGBM with
sample weight equal to **the squared scaling term** `(MTOW − EOW)²`, explicitly "to optimize the root mean
square error on the absolute mass". [S] No ablation of the ratio target alone was published, so the gain is
attested but unquantified.
*Measurement that settles it:* if run, always run the pair — `y / UTXOT` unweighted and `y / UTXOT` with
weight `UTXOT²` — and report both against baseline. A ratio target run *without* the weight is not a test
of the idea; it is a test of a different, wrong metric. [X] My recommendation is to prefer the additive
offset (item 1/2) over the ratio for this problem, because the additive form needs no weighting at all and
our proven win was additive.

---

## 4. Sources

- PRC Data Challenge 2026 — target and metric: <https://prc-data-challenge-2026.netlify.app/>
- Gneiting, *Making and Evaluating Point Forecasts*: <https://arxiv.org/abs/0912.0902>
- Nuyts & Davis, *The When and How of Target Variable Transformations*: <https://arxiv.org/pdf/2504.20821>
- Wilhelm, *Honey, I shrunk the target variable*: <https://florianwilhelm.info/2020/05/honey_i_shrunk_the_target_variable/>
- Duan 1983, smearing estimate: <https://www.tandfonline.com/doi/abs/10.1080/01621459.1983.10478017>, notes: <https://people.stat.sc.edu/hoyen/PastTeaching/STAT704-2022/Notes/Smearing.pdf>, formula: <https://en.wikipedia.org/wiki/Smearing_retransformation>
- Manning & Mullahy 2001, *To transform or not to transform?*: <https://www.sciencedirect.com/science/article/abs/pii/S0167629601000868>, <https://www.nber.org/papers/t0246>
- More (Amazon), *Identifying and Overcoming Transformation Bias in Forecasting Models*: <https://arxiv.org/pdf/2208.12264>
- Tweedie / log-link GLM: <https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html>, <https://ar5iv.labs.arxiv.org/html/1608.04910>
- Koren, *The BellKor Solution to the Netflix Grand Prize* (2009): <https://www2.seas.gwu.edu/~simhaweb/champalg/cf/papers/KorenBellKor2009.pdf>
- PRC Data Challenge 2024, team_likable_jelly (1st, RMSE 1561): <https://github.com/PRC-Data-Challenge-2024/team_likable_jelly>; review of winners: <https://conormclaughlin.net/2024/12/reviewing-the-winning-ml-models-from-the-opensky-prc-data-challenge-what-did-they-do-differently/>
- Kaggle *Predict Future Sales* (RMSE, target clipped to [0,20]): <https://www.kaggle.com/c/competitive-data-science-predict-future-sales>
- Kaggle *NYC Taxi Trip Duration* (RMSLE, log1p target): <https://www.kaggle.com/c/nyc-taxi-trip-duration>
- Kaggle *Mercedes-Benz Greener Manufacturing* (R², duration in seconds): <https://www.kaggle.com/c/mercedes-benz-greener-manufacturing>, community write-up: <https://medium.com/analytics-vidhya/detailed-solution-to-mercedes-benz-green-manufacturing-competition-on-kaggle-7b85c84a3ff5>
- White, *Some observations on winsorization and trimming*: <https://www.johnmyleswhite.com/notebook/2015/12/03/some-observations-on-winsorization-and-trimming/>
- Winsorisation critique: <https://blog.trainindata.com/winsorization-handling-outliers-in-machine-learning/>
- Expectile regression: <https://arxiv.org/html/2510.05268>, <https://search.r-project.org/CRAN/refmans/VGAM/html/amlnormal.html>
- Hurdle / two-stage: <https://towardsdatascience.com/two-stage-hurdle-models-predicting-zero-inflated-outcomes/>, <https://arxiv.org/pdf/2310.08088>, <https://www.sciencedirect.com/science/article/abs/pii/S0022169425022395>
- EUROCONTROL/PRU, *Additional Taxi-Out Time Performance Indicator*, Ed. 00-04: <https://ansperformance.eu/library/pru-tx-out-pi.pdf>
- NASA/Amblard et al., *Real-time Unimpeded Taxi Out Machine Learning Service*: <https://ntrs.nasa.gov/api/citations/20210017591/downloads/20210017591_Amblard_Aviation2021_paper.pdf>
- *Interpretable Taxi-Out Time Prediction … Stacking Ensemble and SHAP* (Sci. Rep. 2026): <https://pmc.ncbi.nlm.nih.gov/articles/PMC13022428/>, <https://www.nature.com/articles/s41598-026-40898-5>
- *Additional Taxi-Out Time Prediction … Fusing Flow Control Information* (Appl. Sci. 14:9968, 2024; full text 403 to this agent): <https://doi.org/10.3390/app14219968>
