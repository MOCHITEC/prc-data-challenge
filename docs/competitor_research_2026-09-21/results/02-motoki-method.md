# Slot 02 — "motoki-method"

Agent: SubAgent, 2026-09-21. Legend: **[S]** = read from a source (URL given), **[X]** = my own
extrapolation / inference. Nothing marked [S] unless I actually opened the page.

---

## 0. Headline: the identification is REFUTED

**The prior claim in this repo — `youthful-giraffe` = Matt Motoki — has no primary source and is
contradicted by the one primary source that exists.**

### 0.1 What the prior note actually said
`docs/investigation_2026-09-16/hypotheses/external.md` line 15 reads
`"youthful-giraffe: … Solo, Kaggle GM (Matt Motoki)."` It carries no URL and sits under an `[S]`
heading that covers the *leaderboard numbers*, not the identity. **[S]** (local file)

### 0.2 The primary source: the official team registration page
The PRC 2026 site publishes one page per registered team. `youthful-giraffe`'s page says: **[S]**
https://prc-data-challenge-2026.netlify.app/teams/youthful-giraffe.html

> youthful-giraffe — **Indonesia** — Registered **September 1, 2026**
> Description and Rationale: "It's an interesting challenge"
> Details: **Type: Independent**, **Country: Indonesia**, **Number of team members: 1**

So: solo — yes. Indonesia — not Matt Motoki.

### 0.3 Matt Motoki's location
His own site is branded "Island & Pine" and his LinkedIn headline is "Matt M. — Shifted Energy"
(Honolulu-based company). **[S]** https://mattmotoki.github.io/ ,
https://www.linkedin.com/in/mattmotoki/ (title only; LinkedIn returned HTTP 999 to the fetcher).
He won the 2025 NASA Airport Throughput Prediction Challenge, whose eligibility was restricted to
"18+, US citizens/permanent residents, and affiliated with accredited US universities". **[S]**
https://bitgrit.com/competition/23

### 0.4 Searches that found nothing
- No GitHub repo of his mentions PRC/taxi-out. His public GitHub has exactly **3 repos**
  (`1st-place-solution-neurips-citylearn2023-forecasting`,
  `2nd-place-solution-neurips-citylearn2022`, `scripture2slides-oss`). **[S]** GitHub API
  `user:mattmotoki` search + https://github.com/mattmotoki?tab=repositories
- Web searches for "Matt Motoki + PRC Data Challenge / EUROCONTROL / taxi-out": zero hits. **[S]**
- No PRC/OpenSky paper authored by a Motoki. **[S]** (search of ansperformance.eu + OpenSky
  symposium listings)
- The live leaderboard feed carries **no identity field at all**:
  `{"team":"youthful-giraffe","best":245.0207,"n":21,"lastAt":"2026-09-10T11:13:50.050889Z"}`
  **[S]** https://prc-leaderboard.fly.dev/data.json (fetched 2026-09-21T02:57Z)

### 0.5 Verdict
**[X]** The Motoki attribution was almost certainly a hallucination or an unrecorded guess by an
earlier agent. Treat `youthful-giraffe` as *an anonymous solo Indonesian competitor*. Do not build
any inference on "this is a Kaggle GM". Note also the registration country is self-declared and
unverified, so it is evidence, not proof — but there is **zero** evidence on the other side.

### 0.6 Correction to the other half of the prior note
The one thing I *can* confirm from local snapshots **[S]**
(`refs/leaderboard_snapshots/*.json`): youthful-giraffe best = 246.3605 (v18, 09-09) →
245.2901 (v20) → 245.094 (v21) → **245.0207 (v22, processed 2026-09-10T11:13:50Z)**, 21
submissions, and **no submission since 2026-09-10** — 11 days silent. The "262.7 → 248.5 on 09-08"
step is *not* in any snapshot we hold; I could not verify it and the live feed no longer exposes
submission history. **[S]/absence**

### 0.7 Current leaderboard context (2026-09-21T02:57Z) **[S]** https://prc-leaderboard.fly.dev/data.json
| best | n | team | last submission | registered country **[S]** (team pages) |
|---|---|---|---|---|
| 239.1335 | 21 | vigorous-whistle | 09-20 18:23 | India, Independent |
| 240.5497 | 18 | jovial-uniform | 09-20 03:13 | India, Independent |
| 242.0575 | 37 | jolly-lobster | 09-20 18:58 | United States (Lee Marshall, Elysium Labs / UCF METIL Lab) |
| 243.6497 | 70 | enthusiastic-daisy | 09-20 14:57 | China, PhD student, DL trajectory prediction |
| 243.8347 | 34 | gentle-igloo | 09-20 14:19 | Portugal, aviation-occurrence corpus maintainer |
| 244.0470 | 18 | gentle-tractor | 09-20 13:06 | Vietnam, Industry (A-CDM) |
| 245.0207 | 21 | **youthful-giraffe** | **09-10 11:13** | **Indonesia, Independent, 1 member** |
| 249.7845 | 41 | zesty-puzzle | 09-20 14:05 | Germany (Luis Melzer, LU Hannover) |
| 284.3584 | 11 | **resilient-hedgehog (us)** | 09-19 01:54 | Japan, Industry, 1 member |

**[X]** youthful-giraffe has been overtaken by six teams while inactive. Whoever it is, they made a
large jump and stopped. The teams actually ahead of us and still moving are vigorous-whistle,
jovial-uniform, jolly-lobster, enthusiastic-daisy, gentle-igloo, gentle-tractor.

---

*The rest of this document is what the task asked for regardless: a study of Matt Motoki as a strong
Kaggle tabular competitor. It is NOT a study of `youthful-giraffe`.*

---

## 1. Who Matt Motoki is (handles corrected)

- Kaggle: **kaggle.com/mmotoki** (user id 331718) — *not* `kaggle.com/mattmotoki`, which does not
  exist. **[S]** https://www.kaggle.com/mmotoki
- GitHub: **github.com/mattmotoki** **[S]**
- Site: **mattmotoki.github.io** **[S]**
- Self-description: "Kaggle Grandmaster (peak world rank #18, $350k+ in prize money). I audit,
  debug, and improve machine-learning models and forecasts that aren't delivering." **[S]**
  https://mattmotoki.github.io/

### Competition record (verbatim from his own site) **[S]** https://mattmotoki.github.io/
| Year | Competition | Result |
|---|---|---|
| 2025 | **NASA Airport Throughput Prediction Challenge** | **1st, $30,000** |
| 2025 | MISO New Approaches to Baselining the Performance of DERs | Top Prize $7,000 |
| 2024 | Power Patterns Challenge (w/ Elaine Hsu) | 1st, $3,000 |
| 2024 | Predict Energy Behavior of Prosumers (Enefit, Kaggle) | 6th, $5,000 |
| 2023 | NeurIPS CityLearn Challenge — Forecasting Track | 1st, $1,000 |
| 2023 | Net Load Forecasting Prize (DOE) | 2nd, $150,000 |
| 2023 | Digitizing Utilities Prize | Phase 1 Winner $75,000 |
| 2022 | NeurIPS CityLearn Challenge | 2nd, $5,000 |
| 2022 | Solar Forecasting Prize | Runner Up $25,000 |
| 2021 | Recidivism Forecasting Challenge | Prize Winner $48,500 |
| 2019 | **ASHRAE Great Energy Predictor III** (Kaggle, w/ Isamu Yamashita) | **1st, $10,000** |

**[X]** Domain concentration is overwhelming: **energy forecasting**. Aviation appears exactly once.

---

## 2. His methodology, extracted from primary writeups

### 2.1 ASHRAE GEP III — 1st place (RMSLE, heavy-tailed positive target, ~20M rows)
Full source **[S]** https://www.kaggle.com/competitions/ashrae-energy-prediction/discussion/124709
(authored by `mmotoki`, 2020). Everything below is quoted or closely paraphrased from that post.

**Preprocessing / data cleaning — he calls this the single most important thing.**
- Removed three anomaly classes: (1) long streaks of constant values, (2) large positive/negative
  spikes, (3) anomalies found by visual inspection.
- **Cross-entity anomaly validation:** "if an anomaly showed up at the same time at multiple
  buildings, we could be reasonably certain that this was indeed a true anomaly." I.e. he uses
  *peer rows at the same site/time* to decide whether an extreme value is real or corrupt.
- Linear interpolation to impute missing weather (temperature) metadata.
- Fixed a known timezone misalignment between the movement table and the weather table.

**Target transformations — he runs several target parameterisations in parallel and blends them.**
- Baseline: `log1p(meter_reading)`.
- Unit correction for one site (a known data defect).
- **Normalised target:** `log1p(meter_reading / square_feet)` — divide the target by a per-entity
  size/exposure variable. "The models trained with the standardized target added diversity to our
  final ensemble and improved our score by about 0.002."
- He explicitly lists untried further variants (`log1p(y)/square_feet`; features built *from* the
  standardised target).

**Feature engineering — two deliberately different styles inside one team.**
- "Isamu took a conservative approach and carefully selected features; on the other hand, Matt took
  a brute force approach and used most of them." **Motoki's own style is brute-force wide.**
- Feature list: raw joined columns; **categorical interactions** (concatenation of `building_id` and
  `meter`); calendar/holiday/time-of-day; **count (frequency) encodings**; lag weather features;
  **Savitzky-Golay smoothed weather plus its 1st and 2nd derivatives**; **cyclic encoding**
  (`hour → cos(2πh/24), sin(2πh/24)`); **Bayesian/hierarchical target encoding**
  (notebook: https://www.kaggle.com/mmotoki/hierarchical-bayesian-target-encoding).

**Models — segmentation is a first-class modelling axis.**
CatBoost, LightGBM and MLP, each trained on *different partitions of the rows*:
- one model per meter type,
- one model per `site_id`,
- one model per `(building_id, meter)`.

**Validation.**
- K-fold CV where folds are **blocks of consecutive months**, rotated (he gives the exact
  `get_validation_months(n)` helper: 12 rolling windows of n consecutive months, wrapping modulo 12).
- Key sentence: *"Trying different validation schemes allowed us to train models that added
  diversity to our final ensemble."* He treats the CV split itself as a **diversity generator**, not
  just as a measurement device.
- Held out an independent, cleaned external "leak" dataset purely as a **second-stage tuning
  holdout** — a hold-out that the level-1 models never saw.

**Ensembling.**
- **Average in the transformed space**: `expm1(mean(log1p(x)))`, not a mean of raw values.
- **Generalized weighted mean** (power mean) with the power and weights **tuned by Optuna**
  (notebook: https://www.kaggle.com/mmotoki/generalized-weighted-mean — I could not read the
  notebook body, Kaggle returns `PERMISSION_DENIED` to unauthenticated API reads **[S]/absence**).
- **Hedging:** deliberately folded in three *public* leak-free kernels to reduce variance.
- Final submission: "a plain average of our top 4 submissions with respect to the public LB score."

**What did NOT work (his own negative results).**
- Tying together predictions for entities with identical target series — no LB gain.
- **Smoothing the final predictions.** "We had initial success with smoothing the final predictions
  … but after a certain point, we started to find that smoothing hurt our public LB score. Our guess
  is that smoothing helps spiky low quality predictions, but our ensemble predictions were already
  sufficiently smooth."

### 2.2 Enefit "Predict Energy Behavior of Prosumers" — 6th place (MAE)
**[S]** https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers/discussion/499397

This is the cleanest statement of his signature move: **model several different algebraic
re-parameterisations of the target, then blend across them.**

- Two segments (production, consumption). For each, **4 different targets**:
  1. `target - target_lag48`
  2. `target/(target_lag48_avg + 1) - baseline_pred`
  3. `target/(target_lag48_avg + 1) - target_lag48`
  4. `target/(target_lag48_avg + 1)`
  where `target_lag48_avg` is the daily mean of the 2-day-lagged target.
- He built **simple analytic baseline models** first (separate ones for consumption and generation)
  and then had GBDTs learn the **residual over that baseline** (targets 2 and 3 above). The baseline
  models were retrained daily; **the LightGBMs were never retrained.**
- Blend size: `160 models = 8 targets × 10 seeds × 2 hyperparameter settings`, combined with tuned
  per-segment weights (he publishes the weight table; e.g. for consumption the
  `ratio − baseline_pred` target got 0.491 of the weight, for generation the
  `ratio − lag48` target got 0.523).
- Self-criticism, worth heeding: *"In hindsight, I probably should have tried to allocate time to
  retraining LightGBM rather than creating such a big blend."*
- He also notes a *better* normaliser existed that he missed: "I didn't think to scale by
  `installed_capacity`, but from other top solutions, that seems like a good target."

**[X]** Read across to taxi-out: his instinct would be to build several targets —
`y`, `y − unimpeded(airport,stand,runway)`, `y / unimpeded`, `y − f_baseline(...)`,
`log1p(y)` — train the same learner on each, and blend, rather than to search for one perfect target.

### 2.3 NeurIPS CityLearn 2023 Forecasting — 1st place
**[S]** https://github.com/mattmotoki/1st-place-solution-neurips-citylearn2023-forecasting and
https://mattmotoki.github.io/blog/seasonal-uniform-average/

- The winning method is **not** an ML model. It is a **seasonal uniform average** — maintain 24
  hour-of-day (or 168 hour-of-week) running means, updated online via
  `update = (x - values[idx]) / counts[idx]`.
- Three refinements: (1) **informed initialisation** of the averages from entity metadata (a prior,
  to cut variance when a group has few observations — i.e. shrinkage toward a metadata-based prior);
  (2) **outlier filtering** before averaging, because large values distort the mean;
  (3) **autocorrelation adjustment** — blend the seasonal mean with the most recent observations
  using a decay factor ≈ 0.93.
- He reports the plain version alone "outpaced 75% of the more complex machine learning models
  submitted".

**[X]** The transferable lesson is the *shape*: group-conditional mean + metadata prior + outlier
filtering + recent-observation blending. That is essentially a principled version of "unimpeded
taxi time per (airport, stand, runway) shrunk toward a parent group, blended with what the last
hour actually did."

### 2.4 Beta / hierarchical Bayesian target encoding
**[S]** https://mattmotoki.github.io/blog/beta-target-encoding/
- His objection to plain target encoding: with `N=1` observations in a level, the encoding *is* the
  label — guaranteed leakage/overfit.
- His fix is a conjugate-prior posterior mean with **one interpretable hyperparameter τ = "the
  effective sample size from the prior distribution"**, rather than ad-hoc additive smoothing.
- Stated as extensible from Beta/Bernoulli (binary) to continuous targets.
- Companion notebooks: `hierarchical-bayesian-target-encoding`, `avito-target-encoding` (bodies not
  readable without a Kaggle login **[S]/absence**).

### 2.5 NeurIPS CityLearn 2022 — 2nd place
**[S]** https://github.com/mattmotoki/2nd-place-solution-neurips-citylearn2022 — a control/RL
problem (dynamic programming + policy networks + CMA-ES), **not** tabular regression. Uses
time-series cross-validation and L2 regularisation. Not transferable here.

---

## 3. Recurring methodology — the distilled pattern

All items **[S]** unless marked. Sources: §2.1 and §2.2 above.

1. **Clean the target first, aggressively, using peer rows.** He removes unlearnable anomalies and
   validates each suspected anomaly against other entities at the same site/time.
2. **Multiple target parameterisations, in parallel, as an ensemble axis.** Differences, ratios,
   residual-over-baseline, log — each becomes its own model family and they are blended.
3. **Normalise the target by an exposure/size variable** (`/square_feet`, `/lag48_avg`) — and he
   explicitly regrets not finding the best normaliser sooner.
4. **A simple analytic baseline first; GBDT learns the residual.** (Enefit baselines; CityLearn
   seasonal average.)
5. **Row-partitioned specialist models as a first-class axis**: per-meter, per-site,
   per-(entity, meter). Trained alongside, not instead of, global models.
6. **CatBoost + LightGBM + MLP**, never one library.
7. **Brute-force wide feature sets** (his own words) rather than curated ones — count/frequency
   encodings, categorical concatenations, cyclic encodings, Bayesian target encoding, smoothed
   exogenous series and their derivatives.
8. **Validation schemes are also a diversity axis.** Rotating blocks of consecutive months; multiple
   schemes trained simultaneously.
9. **Blend in the transformed space** (`expm1(mean(log1p(·)))`), with a **generalized weighted
   (power) mean** whose weights *and power* are Optuna-tuned; keep a clean, model-unseen holdout to
   fit the blend.
10. **Seed × hyperparameter-setting multiplication** (10 seeds × 2 settings in Enefit; NVIDIA's GM
    playbook reports the same, see §5.4).
11. **Post-processing smoothing helped early and hurt late** — once the ensemble was good, smoothing
    became harmful.
12. **[X]** He has *not* published anything I could find about clipping, winsorising or explicit
    tail post-processing for RMSE. His tail handling is upstream (clean the data, transform the
    target), not downstream.

### 3.1 On RMSE with heavy-tailed targets specifically
**[S]/absence** — I found **no** writing by Motoki on this topic directly. The closest primary
evidence is ASHRAE: the metric was RMSLE and he worked entirely in `log1p` space, averaged in
`log1p` space, and *filtered* extreme observations out of training rather than modelling them.
The NASA throughput competition he won scored `exp(-RMSE/10)` **[S]** https://bitgrit.com/competition/23
— a monotone transform of RMSE, so that win *is* an RMSE win, but no method description is public.

### 3.2 On duration / time prediction specifically
**[S]/absence** — none. All of his published work is on **energy quantities** (kWh, load,
generation), not durations.

---

## 4. Aviation / transport / ETA work

- **2025 NASA Airport Throughput Prediction Challenge — 1st place, $30,000.** **[S]**
  https://mattmotoki.github.io/ and https://bitgrit.com/competition/23
  Competition facts **[S]**: forecast **arrival counts in twelve 15-minute buckets, up to 3 h
  ahead**, per US airport; data = FUSER (flight/airport ops, 8 CSV types incl. runway configuration
  and timings), METAR, TAF, optional CWAM convective-weather; ~200 GB; "24 days training, 8 days
  testing" cyclical split over one year; metric `exp(-RMSE/10)`; max 3 submissions/day; **external
  datasets forbidden**; 392 participants, 746 submissions; winners announced 2025-02-19.
  **Winners transfer solution rights to bitgrit** **[S]** — which is very likely why no code or
  writeup exists.
- **No public code, notebook, paper or blog post for that win.** I checked his GitHub (3 repos, none
  aviation), his site (the entry links only to the bitgrit page), Google Scholar (geophysics +
  energy-systems papers only, no aviation) and web search. **[S]/absence**
  https://scholar.google.com/citations?user=fOWhQ3AAAAAJ
- **No other aviation/transport/ETA work found.** **[S]/absence**

**[X]** Conclusion: his aviation win is real and directly adjacent (same FUSER-style operational
tables, same RMSE family, same "airport state now → near-future outcome" structure), but it is a
**counting/throughput** problem, not a per-flight duration problem, and **none of its method is
public**. There is nothing to copy.

---

## 5. Other top competitors' methodology on duration / ETA problems

### 5.1 Zindi Sendy Logistics Challenge — predict delivery time in **seconds**, metric **RMSE**
The closest public analogue to our problem I found. Winners' interviews **[S]**
https://zindi.world/learn/meet-the-winners-of-the-sendy-logisitics-challenge

- **1st (Roman / TheRealRoman):** stacking with a **Lasso** meta-model over LightGBM base models;
  **Poisson regression objective in LightGBM "worked without outliers filtering very well"**;
  target encoding; time differences; rounded coordinates; value-counts on coordinates; several
  different outlier-filtering variants across base models.
- **2nd (Evgeny Patekha / johnpateha):** explicit **two-stage outlier architecture** — a
  **classifier** for "likelihood of short targets" plus regressors (LGBM + XGB + CatBoost) with
  **RMSE and Fair-loss** objectives; >10 regression models + 2 classifiers; **target encoding using
  average speed = target / distance** (i.e. a normalised target used as an encoder statistic);
  and the key trick: **"Predictions of regression models were multiplied by 1−probability from
  classification model — it helped to adjust predictions for outliers."**
- **3rd (Yury / kss):** **Tweedie and Poisson losses in LightGBM "actually worked better than RMSE
  loss"**; binary classifier for outliers; **isotonic regression to calibrate predictions**;
  geographic clustering into 50 "districts" with aggregated route features; a separate **speed
  prediction model**; MinMaxScaler on the target; final blend via Ridge + an `xentropy`-objective
  LightGBM.

**[X]** Three independent top-3 finishers on an RMSE-in-seconds duration problem all (a) used a
**non-Gaussian positive-support loss (Poisson/Tweedie/Fair) instead of L2**, (b) built an
**explicit outlier/short-trip classifier** and multiplied or calibrated the regression output by it,
and (c) encoded by **speed = target/distance** rather than by raw target. That is a coherent,
thrice-replicated recipe and it maps directly onto taxi-out.

### 5.2 NASA "Pushback to the Future" (DrivenData) — minutes-to-pushback, MAE
Finalist descriptions **[S]** https://drivendata.co/blog/airport-pushback-finalists ,
code at https://github.com/drivendataorg/nasa-airport-pushback (slot 05 covers this code-level).
- **1st (Team CDS):** four families of **CatBoost** models, **298 features** from historical
  windows, MAE objective; **time-based CV "did not improve the performance"**.
- **2nd (Moles):** **one CatBoost regressor per airport**, per-airport hyperparameter optimisation;
  explicit **leakage prevention** in feature extraction; **queue-length features showed no
  improvement**, and partially globalising the model also did not help.
- **3rd (Oracle 2):** per-airport **XGBoost**; features = most-recent ETD, 30-hour ETD mean and
  std, airline code, prediction time, computed taxi time; **outlier detection on absolute error
  >20–30 min**; and a **bias-correction post-process** — an XGBoost *classifier* predicts whether
  the regression over- or under-estimates, then **add or subtract the median historical error**.
- **4th (FLHuskies2):** LightGBM with **`objective=mae`** built in; tested a global model and found
  **per-airport models gave lower MAE**; future weather forecasts gave no benefit.
- **5th (Syracuse):** XGBoost on time-series signals (day-of-week, block time, pre-prediction flight
  counts); runway statistics from TFM **hurt** on the prescreened set.

**[X]** Two strong, reproducible signals: **per-airport models beat a global model** (2nd and 4th
both tested it explicitly), and **a separate classifier-driven bias correction on top of the
regressor** (3rd) is a cheap, real post-processing win.

### 5.3 Uber DeeprETA — ETA post-processing at scale
**[S]** https://arxiv.org/pdf/2206.02127 (via ar5iv) and
https://www.uber.com/en-HK/blog/deepeta-how-uber-predicts-arrival-times/
- Architecture is **residual-over-a-physical-baseline**: a routing engine produces an ETA, the ML
  model predicts only the **residual** between routing ETA and observed outcome.
- Loss: **asymmetric Huber**, two parameters — `δ` interpolates between squared and absolute error
  (robustness to outliers), `ω` weights under- vs over-prediction.
- **All continuous features are discretised** into **quantile buckets** (e.g. speed into 256
  quantiles) and embedded; **quantile bucketing outperformed equal-width bucketing**.
- Post-processing: **ReLU at the output** to force positivity, plus **clamping to reduce the effect
  of extreme values**; a calibration layer contributes measurably.
- Reported ~**7.83 % MAE improvement** over the routing-engine baseline.

### 5.4 Generic top-tabular craft (NVIDIA Kaggle Grandmasters Playbook)
**[S]** https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/
Seven techniques: (1) EDA that checks **train/test distribution shift** and temporal target drift;
(2) **diverse baselines** (linear, GBDT, NN, SVR) before any feature engineering; (3) **large-scale
feature engineering** — e.g. all pairwise concatenations of 8 categoricals → 28 new categorical
interaction features; (4) **hill-climbing ensembling** (start from the best single model, greedily
add weighted models, keep only improvements); (5) **multi-level stacking**; (6) pseudo-labelling;
(7) **many seeds** — "ensembling 100 different seeds clearly outperformed single-seed training", and
retrain on 100 % of the data after HPO.

### 5.5 NYC Taxi Trip Duration (Kaggle) and ECML/PKDD 2015 Taxi Trip Time
- NYC Taxi Trip Duration was a *playground* competition; the official closing post awards prizes for
  **datasets and kernels**, not for a 1st-place method, and the decisive contribution was an
  external dataset — **OSRM routing distances/durations** by Oscar Leo. **[S]**
  https://www.kaggle.com/c/nyc-taxi-trip-duration/discussion/39792 . **[X]** The lesson is only "a
  good physical route/time baseline as a feature dominates" — the same idea as DeeprETA §5.3.
- ECML/PKDD 2015 trip-time: the overall winner used an **MLP with embeddings**, and "although more
  sophisticated alternative architectures were tried, the simpler winning model performed better".
  The 3rd-place trip-time team (BlueTaxi) used **trip matching + ensemble learning**. **[S]**
  https://arxiv.org/pdf/1508.00021 , https://arxiv.org/abs/1509.05257 . I could not parse the CEUR
  PDF of the BlueTaxi paper for details. **[S]/absence**

---

## Applicable to resilient-hedgehog

Concrete, checkable. Each has a defined experiment and a pass/fail. Ordered by
(my estimate of) expected RMSE gain per hour of work. **[X]** on the ordering and on all expected
magnitudes; the *provenance* of each idea is [S] as cited above.

1. **Delete the Motoki premise from the working notes.** `external.md:15` should read
   "solo, Indonesia, independent, 1 member, inactive since 2026-09-10". Checkable: the line no
   longer says "Matt Motoki". Source: §0.2.

2. **Try Poisson and Tweedie objectives in LightGBM against our current L2.** Sendy 1st and 3rd both
   report a positive-support count-like loss beating RMSE loss *on an RMSE-scored duration-in-seconds
   problem* (§5.1). Taxi-out is positive, right-skewed, variance growing with mean — the exact
   Tweedie/Poisson setting. Checkable: OOF RMSE of `objective=poisson` and `objective=tweedie`
   (variance_power ∈ {1.1, 1.3, 1.5, 1.7}) vs our current best single model, same features, same
   folds. Pass = any beats L2 OOF; even a tie is valuable as a blend member.

3. **Build a short/outlier classifier and use it multiplicatively or additively on the regressor.**
   Sendy 2nd: multiply regression output by `1 − P(outlier)`. Pushback 3rd: classify the sign of the
   bias, then add/subtract the median historical error of that class. Checkable: fit
   `P(y > q95_airport)` (and/or `P(y < q05)`) with the same features; on OOF, sweep the blend
   `ŷ' = ŷ·(1−p) + p·m_tail` and the additive median-shift variant; keep only if OOF RMSE drops.

4. **Add a normalised/ratio target as a parallel model, not a replacement.** Motoki's single most
   repeated move (§2.1 `/square_feet`, §2.2 `/lag48_avg`; Sendy's speed = target/distance). For us
   the natural denominators are the ATXOT-style unimpeded reference per
   `(airport, stand_group, runway)` and the airport-hour median. Checkable: train identical LGBMs on
   (a) `y`, (b) `y − unimpeded`, (c) `y / unimpeded` (predictions mapped back by ×unimpeded),
   (d) `log1p(y)`; report the OOF RMSE of each **and of their blend**. Motoki's claim is that (b)-(d)
   may individually lose to (a) and still improve the blend — so the pass criterion is the *blend*,
   not the single model.

5. **Blend with a generalized (power) weighted mean, weights and power tuned on a held-out set the
   level-1 models never saw** — not a plain average, and average in the transformed space if the
   members were trained in it (§2.1). Checkable: on OOF, compare plain mean vs Optuna-tuned
   `(Σ wᵢ xᵢ^p)^{1/p}`; require the tuning split to be disjoint from every member's training folds.

6. **Per-airport specialists, tested explicitly against the global model — and keep both.** Pushback
   2nd and 4th both ran this A/B and both chose per-airport (§5.2); Motoki routinely trains
   per-site and per-(entity, segment) models alongside the global one (§2.1). Checkable: OOF RMSE of
   global vs 10 per-airport models vs a 50/50 blend, per airport and overall. Record the per-airport
   table, because the answer may differ by airport (LIRF especially).

7. **Seed × hyperparameter-setting multiplication on the final model.** Motoki: 10 seeds × 2 settings
   (§2.2); NVIDIA GM playbook: 100 seeds clearly beat one (§5.4). We have 11 submissions and a single
   LightGBM — this is the cheapest remaining variance reduction. Checkable: OOF RMSE of
   1 vs 5 vs 10 seeds averaged, same config; and retrain the final on 100 % of rows after HPO.

8. **Peer-validated anomaly filtering of training rows.** Motoki's rule: an extreme value is
   *unlearnable noise* if no other entity at the same site/time shows it, and *real* if peers do
   (§2.1). We already know the MVT zero-second-share anomaly (~8.3 % at EDDM/EGLL/LFPG/LIRF/LTFM vs
   ~1.7 % elsewhere, per `external.md` A3). Checkable: for candidate outlier rows, compute the
   contemporaneous peer-lateness at the same airport ±30 min; drop from *training only* the rows
   with extreme y and no peer support; measure OOF RMSE on the unfiltered validation set.

9. **CV blocks of consecutive months, rotated — and keep several schemes as a diversity axis.**
   (§2.1). Our target months are January and July 2026, i.e. two disjoint seasons; a single random
   KFold will not expose seasonal generalisation. Checkable: build a rotating consecutive-month
   fold set over 2025, report per-fold RMSE, and check whether Jan-held-out and Jul-held-out folds
   disagree about which model wins. Caution [S]: Pushback 1st place found time-based CV *did not*
   improve their result — so this is a measurement to make, not a foregone conclusion.

10. **Quantile-bucket the continuous features for any NN/embedding model we add, and ReLU + clamp the
    output.** DeeprETA found quantile bucketing beat equal-width, and clamps the tail (§5.3).
    Checkable: only relevant if we add a NN member; then A/B raw-continuous vs 256-quantile-bucketed
    inputs on OOF.

11. **Do NOT smooth the final ensemble predictions.** Motoki's explicit negative result: smoothing
    helps weak spiky predictions and *hurts* once the ensemble is already smooth (§2.1). Checkable:
    if anyone proposes prediction smoothing, require an OOF demonstration first.

12. **Stop expecting a public trace of the NASA-throughput method.** bitgrit takes solution rights
    from winners (§4); there is no repo, notebook or paper. Do not spend further slots looking.
