# Slot 11 — Ensembling patterns: how aviation and tabular winners actually build blends

Agent: SubAgent for slot 11. Date: 2026-09-21.
Marking: **[S]** = read from a named source (URL given). **[X]** = my own extrapolation or arithmetic.
All arithmetic I did myself on sourced numbers is marked [X] even when the inputs are [S].

---

## 0. One-paragraph summary

The sourced record does **not** support the idea that multi-family blending is what separates the top
of an aviation leaderboard from the middle. Across the two most directly comparable competitions
(PRC 2024 ATOW, PRC 2025 fuel burn) and one adjacent one (NASA/DrivenData pushback), the blend
buys between **0.3% and 1.8% RMSE over the best single model**, and in both PRC editions a team that
did *not* blend across families finished at or near the top. What the sources are unanimous about is
the *mechanism*: when a blend is fitted, it is fitted by constrained least squares (non-negative,
sum-to-one) on predictions produced by **one shared fold definition applied identically to every
candidate model**. That last requirement is the one our repository currently violates.

---

## 1. Concrete, sourced accounts of blend construction

### 1.1 PRC 2024 Data Challenge (ATOW, RMSE) — team_tiny_rainbow / Murça et al. [S]

Source: Murça, Maximo, Dantas, Szenczuk, Rutili de Lima, Carvalho, de Melo, "Open Machine Learning
Models for Actual Takeoff Weight Prediction", *Journal of Open Aviation Science* 3(2), 2025.
<https://journals.open.tudelft.nl/joas/article/view/7963> — full text read from
<https://www.joaopadantas.com/files/murca2025open.pdf>.
Repo: <https://github.com/PRC-Data-Challenge-2024/team_tiny_rainbow>

**Exact mechanism** (§2.4.5, quoted): the ensemble is a weighted sum
`f_ensemble(x) = Σ_h w_h · h(x)`, with weights found by **quadratic programming** solving

```
min_w  (1/n) Σ_i ( y_i − Σ_h w_h h(x_i) )²
s.t.   Σ_h w_h = 1,   w_h ≥ 0 ∀h
```

Fitted on a **held-out 20% validation set that was not used to train the individual models**
("This separation ensures that the weights are learned on data not seen during individual model
training, helping to avoid potential overfitting"). After the weights are learned, **each individual
model is retrained on the full training dataset** and the pre-computed weights are applied to
predictions on new data. They cite Breiman's stacked regression and van der Laan's Super Learner as
the justification.

**Exact numbers** (their Table 6, "Actual RMSE" = submission-set score, kg):

| Model | Estimated test RMSE | Actual RMSE |
|---|---|---|
| Baseline (MTOW) | 32,324.13 | 32,324.13 |
| MLP (FastAI tabular) | 3,864.1 | 3,864.2 |
| SAINT | 3,921.3 | 3,921.5 |
| **CatBoost** | **2,215.53** | **2,224.69** |
| LightGBM | 2,506.43 | 2,519.17 |
| XGBoost | 2,413.05 | 2,403.07 |
| **Ensemble** | **2,210.37** | **2,217.75** |

Fitted weight vector as printed: `[1.24e-23, 1.63e-23, 8.65e-01, 4.77e-22, 1.34e-01]`.
[X] Read against the table's row order (MLP, SAINT, CatBoost, LightGBM, XGBoost) this is
**CatBoost 0.865, XGBoost 0.134, everything else numerically zero**. The paper's own ablation
corroborates the ordering: removing CatBoost costs +33.90% RMSE, removing XGBoost +0.16%, and
removing LightGBM, the NN and SAINT each cost **exactly +0.00%**.

**Ablation table (Table 8), verbatim:**

| Ablated component | RMSE | Change |
|---|---|---|
| Baseline (full model) | 2,210.37 | 0.00% |
| XGBoost algorithm | 2,213.91 | +0.16% |
| CatBoost algorithm | 2,969.49 | +33.90% |
| LightGBM algorithm | 2,210.37 | +0.00% |
| Neural network | 2,210.37 | +0.00% |
| SAINT | 2,210.37 | +0.00% |
| Operational features | 2,759.18 | +24.83% |
| Geographical features | 2,390.26 | +8.14% |
| Temporal features | 2,354.97 | +6.54% |
| Aircraft characteristics | 2,266.96 | +2.56% |
| Meteorological features | 2,249.29 | +1.76% |

**What the blend bought** [X]: best single model 2,224.69 → blend 2,217.75 = **6.94 RMSE, 0.31%
relative**. Over LightGBM alone (2,519.17) the blend is 12.0% better, but that is mostly the cost of
LightGBM being their *worst* GBDT, not a blending gain. The paper itself concludes the LightGBM, NN
and SAINT members "may be redundant in the current ensemble configuration, and thus pruned."

**Load-bearing observation** [X]: one feature group (operational, +24.83%) was worth ~80× more than
the entire XGBoost member (+0.16%) and infinitely more than the LightGBM member (+0.00%).

### 1.2 PRC 2025 Data Challenge (fuel burn, RMSE) — rank 2, `jubilant-vase` [S]

Sources: <https://prc-data-challenge-2025.netlify.app/outcome.html> (official ranking: rank 2,
RMSE 211.99, €1,750); repo README and `train_blend.py` read raw from
<https://github.com/PRC-Data-Challenge-2025/jubilant-vase>.

**Exact mechanism** — from `train_blend.py`, which I read line by line:

```python
N_FOLDS = 5
kf = GroupKFold(n_splits=N_FOLDS)                 # ONE kf object, ONE groups vector
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y, groups=groups)):
    ...
    oof_preds[val_idx] = model.predict(X_val)     # same folds reused for xgb, lgb, cat

def loss_func(weights):
    final_pred = np.average(X_blend, axis=1, weights=weights)
    ...                                           # RMSE on the OOF matrix

res = minimize(loss_func, init_weights,
               bounds=[(0, 1)] * len(model_names),
               constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

for i, name in enumerate(model_names):
    weight = best_weights[i]
    if weight < 0.01: continue                    # prune negligible members
    # test prediction = mean over the 5 fold models, then × weight
```

Four mechanism details that matter:
1. **Identical folds.** All three libraries are trained inside the *same* `GroupKFold(5)` loop with
   the same `groups` (grouped by `flight_id` "to prevent leakage"), so the three OOF vectors are
   row-aligned and measured on the same partition.
2. **Constrained simplex.** `bounds=[(0,1)]` plus `sum(w)==1` — non-negative, sum-to-one, i.e. the
   same feasible set as the PRC 2024 QP above. (SciPy dispatches this to SLSQP.)
3. **Pruning.** Members with weight < 0.01 are dropped entirely at inference.
4. **Fold bagging, not refit.** Test predictions are the **mean of the 5 fold models**, not a
   single model refit on all data. (Contrast §1.1, which refits on full data.)

**Exact numbers** (repo README "Results" table):

| Model | RMSE (CV / OOF) | RMSE (Rank/LB) |
|---|---|---|
| XGBoost | ~246.8 | — |
| LightGBM | ~245.4 | — |
| CatBoost | ~244.9 | — |
| **Ensemble** | **~241.0** | **~224.0** |

Final weights: **CatBoost 64.1%, LightGBM 19.9%, XGBoost 16.0%.** (This matches the figure quoted
in the slot brief and is now sourced.)

**What the blend bought** [X]: best single (CatBoost 244.9) → 241.0 = **3.9 RMSE, 1.59% relative**.
Over LightGBM alone (245.4) → 241.0 = **4.4 RMSE, 1.79% relative**.

### 1.3 PRC 2025 Data Challenge — rank 1, `resourceful-quiver` — NO BLEND [S]

Sources: <https://prc-data-challenge-2025.netlify.app/outcome.html> (rank 1, **RMSE 200.83**,
€2,500); README and `notebooks/TrainModelAndSubmit.ipynb` read raw from
<https://github.com/PRC-Data-Challenge-2025/resourceful-quiver>.

The shipped model is a single artefact: `models/lgbm_mass_estim_24_feat.pkl`, "the lgbm mass
estimation model using 24 features". I grepped the training notebook source: **0 occurrences** of
`catboost`, `CatBoost`, `xgboost`, `XGB`, `ensemble`, `blend`, `seed`, or `weight`; the flags in the
notebook read `try_kfold = False`, `final_model = True`. The imports are `LGBMRegressor`,
`early_stopping`, `KFold`. Their edge, per the official write-up, was "a well-founded and thoughtful
methodology... strongly informed by domain knowledge": a physics-based mass estimator carried over
from PRC 2024 plus a `traffic`-library preprocessing pipeline (`preproc_v2.py`).

**[S] fact, and the single most important one in this slot:** in PRC 2025, a **single LightGBM with
24 physical features beat a fully-fitted three-library OOF-optimised blend by 11.16 RMSE**
(200.83 vs 211.99), a 5.3% gap [X]. The blend bought its owner 3.9 points; the feature/physics work
was worth roughly 3× that.

### 1.4 PRC 2026 (our competition) — public competitors [S, with one caveat]

- **OpenAir** (<https://github.com/ChandanHegde07/OpenAir>): experiment E20 is described as
  "**NNLS 0.456 CatBoost + 0.491 airport-LGB + 0.053 XGB**", i.e. non-negative least squares over
  three members, with per-airport LightGBM experts as one member. Reported for E20: overall RMSE
  368.03 on a Jan+Jul holdout, matched RMSE 244.76, leaderboard 316.97. The repo page does **not**
  state whether the NNLS was fitted on k-fold OOF or on the Jan+Jul holdout; their documented
  training convention is "train all 2025 months except January and July; hold out Jan+Jul", which
  [X] suggests the weights are fitted on that holdout, i.e. the §1.1 pattern (holdout-fitted) rather
  than the §1.2 pattern (OOF-fitted).
  *Caveat:* the brief gives OpenAir's leaderboard as 276.99; the page I read gives 316.97 **for
  experiment E20 specifically**. These are presumably different experiments; slot 06 owns the
  code-level read and should reconcile.
- **Phoenix-Ops** (CatBoost 183 features + LightGBM + specialists) — taken as [S] from the brief;
  I did not independently re-read that repo (slot 07 owns it).

### 1.5 NASA / DrivenData "Pushback to the Future" (predict pushback time, MAE minutes) [S]

Source: <https://github.com/drivendataorg/nasa-airport-pushback> (winners' repo index) and
<https://drivendata.co/blog/airport-pushback-finalists>.

Phase 1 (the open-data phase comparable to ours):

| Place | Team | MAE (min) | Modelling |
|---|---|---|---|
| 1 | CDS | 10.673 | **Ensemble of multiple CatBoost regressors per airport** |
| 2 | Moles | 10.728 | **Single CatBoost regressor per airport** |
| 3 | Oracle 2 | 11.054 | XGBoost regressor + classifier per airport (multi-stage) |
| 4 | FLHuskies2 | 11.105 | **LightGBM regressor per airport** |
| 5 | Cuong_Syr | 11.946 | XGBoost regressor per airport |

**[X] Reading:** nobody in the top 5 blended across GBDT libraries. 1st place's "ensemble" is
*within* CatBoost (seed/variant averaging), and it beat a *single* CatBoost by 0.055 min = **0.51%**.
The axis that actually spread the leaderboard was **per-airport specialisation** (every one of the
top 5 trains per airport), which is slot 12's subject, not blending.

The DrivenData benchmark post also records that a finalist chose LightGBM specifically because
"it has absolute error as a built-in objective function and is much faster for model training",
and that they tested one global model vs one model per airport and **the per-airport models won**
(<https://drivendata.co/blog/airport-pushback-benchmark>).

### 1.6 ASHRAE Great Energy Predictor III (Kaggle, RMSLE, ~3,600 teams) [S]

Source: Miller et al., "The ASHRAE Great Energy Predictor III competition: Overview and results",
<https://arxiv.org/pdf/2007.06933> (§5.1–5.5, Table 2). Full text extracted and read.

| Place | Who | Private score | Mechanism (verbatim where quoted) |
|---|---|---|---|
| 1 | Matthew Motoki & Isamu Yamashita | **1.231** | LightGBM + CatBoost + MLP trained on different subsets (by meter, primary use, site id); 12-fold CV; "individual model predictions were combined using a **generalized weighted mean** approach" |
| 2 | Rao, Isakin, Zang, Knaub | 1.232 | XGBoost + LightGBM + CatBoost + FFNN on different subsets; "ensembled using the **weighted mean** method... the weights were **determined empirically**" |
| 3 | Xavier Capdepon | 1.234 | CatBoost + NN + LightGBM; "**weighted average** method where weights were calculated with the help of the **publicly-available datasets**" |
| 4 | Jun Yang | 1.235 | 2 × XGBoost (2-fold and 5-fold) + 1 × LightGBM; "weights were determined using the **leaked data**" |
| 5 | Sano, Tomioka, Kobayashi | 1.237 | "**Unlike the other solutions, this team used only LightGBM**", per building×meter, weighted average with weights from the public dataset |

**[X] Reading:** the four-family 2nd-place blend and the LightGBM-only 5th-place solution are
**0.006 RMSLE apart — 0.49% relative**. Not one of the top five fitted blend weights by
NNLS/QP on OOF; three of five set weights using the *public leaderboard / leaked data*, which is a
holdout-fitting strategy unavailable to us. The paper's own §7.1.3 heading is
"Ensembles of models create more accurate (but slower) models".

### 1.7 Kaggle tabular regression craft [S]

- **NVIDIA "Kaggle Grandmasters Playbook"**
  (<https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/>):
  hill climbing is "a simple, but powerful way to ensemble models" — "Start with your strongest
  single model, then systematically add others with different weights, keeping only the combinations
  that improve validation." In *Predict Calorie Expenditure* they "used a hill climbing ensemble of
  XGBoost, CatBoost, neural nets, and linear models to secure first place." On seeds: "Train with
  different random seeds... averaging predictions, often improves performance", and in the fertilizer
  competition "ensembling XGBoost models across 100 different seeds clearly outperformed single-seed
  training." **No numeric gain is given for either claim** — they are directional statements.
- **MLWave Kaggle Ensembling Guide** (mirror: <https://evah.github.io/2017-02-05/ensemble/>): the
  operative rule is "creating an averaging ensemble from **less correlated** submissions"; simple
  averaging "works well for a wide range of problems". Stacked generalization = train on part A,
  predict part B, swap, train a meta-learner on the resulting OOF; blending = a ~10% holdout
  instead, simpler but uses less data, while stacking gives "more solid CV... calculated over more
  folds". **It gives no correlation threshold and no numeric gain.**
- **ML Contests tabular review** (<https://mlcontests.com/tabular-data/>): winners overwhelmingly
  ensemble; documented combos include "ensembles of XGBoost, Catboost, and LightGBM models" (NASA
  PM2.5 winner) and "ensemble of GBDTs and Recurrent Neural Networks" (AmEx 1st). The one quantified
  gain on the page: for the NASA air-quality NO2 track, "A final ensemble with a neural network
  added an additional **0.5%** to the model's perf."
- **`Matt-OP/hillclimbers`** (<https://github.com/Matt-OP/hillclimbers>): a packaged hill climber
  over a DataFrame of OOF predictions plus matching test predictions. At each iteration it perturbs
  one weight and accepts the change if the metric improves. `negative_weights` defaults to **off**
  "to reduce overfitting risk"; `precision` (weight step) defaults to 0.01, recommended 0.01–0.001.
  The author states it "chooses diverse models" rather than the highest-CV models. Reported result:
  4th place, Playground Series S3E14 (Wild Blueberry Yield).

### 1.8 The canonical algorithm papers [S]

- **Breiman (1996), "Stacked Regressions"**, *Machine Learning* 24(1):49–64
  (<https://link.springer.com/article/10.1007/BF00117832>, abstract via
  <https://mlanthology.org/mlj/1996/breiman1996mlj-stacked/>): "The idea is to use **cross-validation
  data** and **least squares under non-negativity constraints** to determine the coefficients in the
  combination." The non-negativity constraint is the load-bearing part; the sum-to-one constraint is
  not in Breiman's original formulation (empirically the fitted weights come out near 1 anyway).
- **Caruana, Niculescu-Mizil, Crew, Ksikes (2004), "Ensemble Selection from Libraries of Models"**,
  ICML (<https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf>). Full text read.
  Three explicit anti-overfitting devices, quoted:
  - **Selection with replacement** (§2.1): without replacement, "performance improves as the best
    models are added to the ensemble, peaks, and then quickly declines"; with replacement,
    "models added to the ensemble multiple times receive more weight" and this "flattens the
    performance curve past the peak", so "a test set is not needed to determine when to stop".
  - **Sorted ensemble initialization** (§2.2): rather than starting empty, "sort the models in the
    library by their performance, and put the best N models in the ensemble", typically 5–25.
    "it is more difficult for greedy selection to find models that overfit when added".
  - **Bagged ensemble selection** (§2.3): select from a random sample of p = 0.5 of the library,
    repeat 20 times, average the 20 ensembles. Measured benefit: "bagged ensemble selection reduces
    loss an additional **2.5%** at p = 0.5... about a third of the total 8.7% benefit".
  Headline measured result: over 7 problems × 10 metrics, ensemble selection "wins 64 of 70 times"
  and "reduces loss **8.76%** over the best models for each problem and metric" — from a library of
  ~2,000 models. **Their RMS column specifically** (Table 2, % loss reduction vs the best single
  model from any algorithm): adult **0.71**, cover type **3.79**, letter.p1 24.82, letter.p2 4.46,
  medis **0.04**, hyper spect 7.71, slac **0.36**; mean **5.98**. The authors' own caveat: "we don't
  want to overstate the improvement of ensemble selection. For comparison, bagging or boosting trees
  yields a 20% reduction in loss compared to raw trees, 2.5 times the benefit of ensemble selection."
- **AutoGluon-Tabular** (Erickson et al., <https://arxiv.org/pdf/2003.06505>): the final stacking
  layer "applies ensemble selection (Caruana et al., 2004) to aggregate the stacker models'
  predictions in a weighted manner", chosen for "its resilience against over-fitting". They add
  **repeated k-fold bagging**: "it repeats the k-fold bagging process on n different random
  partitions of the training data, averaging all OOF predictions over the repeated bags. ... OOF
  predictions which have been averaged across multiple k-fold bags display even less variance and
  are much less likely to be over-fit."
- **Q(D)O-ES** (Purucker, Beel et al., <https://arxiv.org/pdf/2307.08364>), 71 datasets from the
  AutoML benchmark: "for all scenarios, post hoc ensemble selection is **statistically significantly
  better than the single best model** — we can always boost the performance on average." But: "there
  are datasets on which this does not hold... Because all methods have access to the validation score
  of the single best and only propose an ensemble if it outperforms the single best, this phenomenon
  is **directly linked to overfitting**." And the closing line: "**the main challenge post hoc
  ensembling encounters is the lack of generalisation from validation to testing sets**."

---

## 2. The mechanisms, distinguished, with what each is reported to buy

| # | Mechanism | Exact definition | What sources report it buys |
|---|---|---|---|
| A | **Simple average** | `mean(preds)`, no fitting | MLWave: works broadly; strictly better than the *mean* member and often the best member when members are comparable in strength and weakly correlated. **No sourced numeric gain found.** Weakness: a weak member is given equal say (see §4). |
| B | **Weighted average, weights set by hand / on a public LB** | weights chosen empirically | ASHRAE 2nd/3rd/4th place set weights "empirically" or from public/leaked data [S]. Whole ASHRAE top 5 span 0.49% [X]. |
| C | **Weighted average, weights fitted by constrained least squares on held-out preds (NNLS/QP/SLSQP)** | `min_w ‖y − Pw‖²` s.t. `w ≥ 0` (Breiman) and optionally `Σw = 1` (PRC 2024 & 2025) | PRC 2025 rank 2: **−3.9 RMSE = −1.59%** vs best single [S+X]. PRC 2024 tiny_rainbow: **−6.94 RMSE = −0.31%** vs best single [S+X]. |
| D | **Hill climbing / greedy forward selection *with replacement* on OOF** | iteratively add the member (repeats allowed) that most improves the OOF metric; counts → weights | Caruana: mean **8.76%** loss reduction vs best single from a 2,000-model library, 64/70 wins; **RMS-specific mean 5.98%**, but only **0.71% / 0.04% / 0.36%** on the three large real-world tabular problems (adult, medis, slac) [S]. AutoGluon and Auto-sklearn both adopt it because it resists overfitting better than a fitted linear meta-model [S]. |
| E | **Stacking with a second-level model** | base models produce OOF preds via `cross_val_predict`; meta-model is fit on those OOF preds (sklearn), optionally concatenated with the raw features (AutoGluon "skip connection") | sklearn: "the `final_estimator` is trained on out-samples using `cross_val_predict` internally" to "avoid over-fitting" [S, <https://scikit-learn.org/stable/modules/ensemble.html>]. H2O default meta-learner: "**GLM with non negative weights & standardization turned off**" [S, <https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/stacked-ensembles.html>] — i.e. the production default for stacking *is* NNLS. Q(D)O-ES notes only one surveyed AutoML system uses a linear stacker; all the rest use GES (mechanism D) [S]. |
| F | **Seed averaging within one family** (what we do today) | same config, n seeds, average | NVIDIA playbook: 100-seed XGBoost "clearly outperformed single-seed training"; no number [S]. DrivenData pushback 1st place = "ensemble of multiple CatBoost regressors per airport" beat a single CatBoost per airport by **0.055 min = 0.51%** [S+X] — the closest thing to a measured seed/variant-averaging gain I found. **No source gives a clean numeric for GBDT seed averaging on an RMSE regression.** |
| G | **Bagging across CV folds** (predict with the k fold models, average) | inference = `mean_k model_k(x)` | PRC 2025 rank 2 does exactly this [S, `train_blend.py`]. AutoGluon bags and additionally *repeats* the bagging over n partitions to de-noise the OOF [S]. |
| H | **Refit on full data** (fold models discarded, weights frozen) | weights fitted on holdout, then all members retrained on 100% of data | PRC 2024 tiny_rainbow does exactly this [S]. |

**G vs H** [X]: no source I found compares them head to head. The trade is stated implicitly:
G gives a k-model average (variance reduction, and the test-time model is the same object the
weights were fitted for) at the cost of each member seeing only (k−1)/k of the data; H gives each
member all the data at the cost that the models the weights were fitted on no longer exist.

---

## 3. Specifically: what does adding CatBoost and XGBoost to a LightGBM buy on RMSE?

**Reported numbers (all [S] for the inputs, [X] for my percentage arithmetic):**

| Source | Problem | LightGBM alone | Best single | Blend of CatBoost+LGBM+XGB(+more) | Gain vs best single | Gain vs LGBM alone |
|---|---|---|---|---|---|---|
| jubilant-vase, PRC 2025 rank 2 | fuel burn, RMSE, 5-fold GroupKFold OOF | 245.4 | 244.9 (CatBoost) | 241.0 | **−3.9 (−1.59%)** | −4.4 (−1.79%) |
| Murça et al., PRC 2024 | ATOW, RMSE (kg) | 2,519.17 | 2,224.69 (CatBoost) | 2,217.75 | **−6.94 (−0.31%)** | −301 (−12.0%) |
| Caruana 2004, adult | binary, RMS | — | best of ~2,000 | ensemble selection | **−0.71%** | — |
| Caruana 2004, medis | binary, RMS | — | best of ~2,000 | ensemble selection | **−0.04%** | — |
| Caruana 2004, slac | binary, RMS | — | best of ~2,000 | ensemble selection | **−0.36%** | — |
| ASHRAE GEPIII | RMSLE | 1.237 (5th, LGBM only) | — | 1.231 (1st, 3 families) | — | **−0.006 (−0.49%)** |
| NASA pushback Ph.1 | MAE (min) | 11.105 (4th, LGBM) | 10.728 (single CatBoost) | 10.673 (CatBoost variants) | −0.055 (−0.51%) | −0.43 (−3.9%) |
| ML Contests / NASA NO2 | air quality | — | — | +NN on top | **−0.5%** | — |

**Honest reading [X]:** for a well-tuned GBDT on a large tabular regression problem, a
constrained-least-squares blend of CatBoost + LightGBM + XGBoost is reported to buy
**roughly 0.3–1.8% RMSE over the best single member**, with the top of that range achieved when the
members are genuinely of similar strength (PRC 2025: 244.9 / 245.4 / 246.8, within 0.8% of each
other) and the bottom when one member dominates (PRC 2024: CatBoost took weight 0.865 and the
LightGBM member's weight was numerically zero).

**On our number:** 1.59% of 284.3584 is **4.5 RMSE**; 0.31% is **0.9 RMSE** [X]. That is the
sourced expectation band for adding CatBoost+XGBoost to our LightGBM: roughly
**284.36 → 279.9…283.5**. It does not, by itself, reach 276.99 or 273.56.

**Folklore, explicitly labelled as such [X]:** the claim that "you always need all three GBDT
libraries" is not supported by any source I read. The counter-evidence is in the same competitions:
PRC 2025's *winner* shipped one LightGBM; ASHRAE's 5th place shipped only LightGBM and lost to the
three-family 1st place by 0.49%; NASA pushback's 2nd place shipped one CatBoost per airport. The
"XGBoost is ~5% better than LightGBM/CatBoost" figure that surfaces in search results comes from a
**classification** benchmark on ROC-AUC/PR-AUC and does not transfer to RMSE regression — I have
**not** verified that source and do not rely on it.

---

## 4. Practical failure modes practitioners report

1. **Overfitting the blend weights to the OOF/hillclimb set.** [S] Caruana §2: "The simple forward
   model selection procedure... sometimes overfits to the hillclimbing set, reducing ensemble
   performance", and the whole of §2.1–2.3 exists to fix it. Q(D)O-ES measured it across 71 datasets:
   ensembles that beat the single best *on validation* sometimes lose *on test*, and the authors call
   this "directly linked to overfitting", concluding "the main challenge post hoc ensembling
   encounters is the lack of generalisation from validation to testing sets." Caruana's Table 2
   contains negative entries (cover type acc −0.42, medis fsc −6.39, letter.p2 cal −40.03) — the
   ensemble was *worse* than the best single model on test in those cells.
   Mitigations that are sourced: selection with replacement; sorted initialization with the best
   5–25 models; bagged selection at p = 0.5 over 20 bags (worth an extra 2.5% [S]); repeated k-fold
   bagging of the OOF (AutoGluon [S]); forbidding negative weights (`hillclimbers`
   `negative_weights=False` by default "to reduce overfitting risk" [S]); the non-negativity
   constraint itself (Breiman [S]).
2. **Correlated models give no gain.** [S] MLWave's operative rule is to average "less correlated
   submissions". The PRC 2024 paper is the measured case: LightGBM, the NN and SAINT each received
   numerically zero weight and their removal changed the ensemble RMSE by **exactly 0.00%** — they
   carried nothing CatBoost and XGBoost did not already carry.
3. **A weak model drags the blend down.** [S] This is precisely the argument the PRC 2024 paper gives
   for fitted over simple weights: the QP "can give higher weight to models that perform better",
   "can deal with correlated predictions between models by adjusting their relative contributions",
   and "can **completely exclude** models that do not add predictive value beyond what is already
   captured by other models". [X] The corollary: a *simple average* has no such protection — an
   unweighted mean of their five models would have been dragged toward the 3,864/3,921 RMSE neural
   nets. `jubilant-vase` handles the same risk by pruning any member with weight < 0.01 [S].
4. **OOF predictions from inconsistent fold structures.** See §5 — this is the one that applies to us.
5. **Weights fitted on a leaderboard rather than held-out data.** [S] Three of the ASHRAE top five
   set their blend weights using public-leaderboard or leaked data. [X] That is a legitimate
   strategy where the public LB is large and drawn from the test distribution; it is a *different*
   mechanism from OOF fitting and its reported gains cannot be read as evidence for OOF fitting.
6. **Blend gains are dwarfed by feature/target work.** [S] PRC 2024's ablation: removing one feature
   group cost +24.83% while removing a whole GBDT member cost +0.16%. PRC 2025: the non-blending
   winner beat the blending runner-up by 11.16 RMSE while the runner-up's blend was worth 3.9.

---

## 5. The requirement on OOF fold structure — directly answering our question

### 5.1 What the standard requires [S]

The requirement is stated most plainly by H2O, whose Stacked Ensembles page lists it as a hard
precondition on base models
(<https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/stacked-ensembles.html>):

> "The models must be cross-validated using **the same number of folds**"
> — and identical folds are obtained by `fold_assignment = "Modulo"`, or `fold_assignment = "Random"`
> **with the same seed across all base models**, or a shared `fold_column`.
> "The cross-validated predictions from all of the models must be preserved by setting
> `keep_cross_validation_predictions` to True."

Its default meta-learner is "**GLM with non negative weights & standardization turned off**" —
i.e. NNLS, the same feasible set as Breiman's stacked regression [S].

scikit-learn's `StackingRegressor` enforces the same thing structurally: it holds a single `cv`
object and drives every base estimator through `cross_val_predict` with it, so all members share one
partition by construction; the docs state "the `final_estimator` is trained on out-samples using
`sklearn.model_selection.cross_val_predict` internally" to "avoid over-fitting" [S,
<https://scikit-learn.org/stable/modules/ensemble.html>].

The PRC 2025 rank-2 code satisfies the same requirement by construction: one `GroupKFold(5)`, one
`groups` vector, reused inside a single loop for XGBoost, LightGBM and CatBoost [S].

The Kaggle-side statement of the same rule, from the "Cross validation strategy when
blending/stacking" thread (<https://www.kaggle.com/general/18793>, surfaced in search; the page
itself is JS-rendered and I could not fetch its body, so I mark the quote **[S-weak]**): *you should
not change the folds between first-stage and second-stage models; the leakage is theoretical rather
than practical as long as you keep the same folds.*

### 5.2 Does a mixed-fold OOF file invalidate NNLS weight fitting?

Precise answer, separating two different things that "mixed folds" can mean [X, reasoning on [S]
definitions above]:

- **Mixing folds *within* one model's OOF vector is normal and does NOT invalidate anything.**
  Every k-fold OOF vector is a concatenation of predictions from k different fitted models. sklearn,
  H2O and `jubilant-vase` all do this. The invariant that must hold is only: *every row's prediction
  comes from a model that did not train on that row.*
- **Mixing fold *definitions across* candidate models DOES invalidate the fit.** NNLS solves
  `min_w ‖y − Pw‖²` where column `j` of `P` is model `j`'s prediction on the *same* rows produced
  under the *same* train/validate protocol. If model A's column was produced under an easier
  protocol than model B's (more training data, or training data that includes the future), then A's
  column has a lower error level for reasons that will not reproduce on the test set, and NNLS will
  assign A more weight than it deserves. Nothing in the objective can see this; it is a pure
  specification error.
- **Non-uniform fold *sizes / training-set composition* within one shared definition is a softer
  problem** [X]: the OOF residual variance is heteroscedastic across blocks, so a single global
  weight vector is a compromise across blocks. The fix practitioners use is not to abandon NNLS but
  to fit the weights on the block that most resembles the test set, or to fit per-block weights and
  check they agree.
- **What practitioners do about it** [S]: AutoGluon's answer is *repeated* k-fold bagging —
  "it repeats the k-fold bagging process on n different random partitions of the training data,
  averaging all OOF predictions over the repeated bags", because averaged OOF "display even less
  variance and are much less likely to be over-fit". Caruana's answer is to not trust a fitted
  linear solution at all on a small hillclimb set and to use bagged greedy selection instead.

### 5.3 Our actual situation — verified in this repository [S]

I read (did not modify) `src/prc/cv.py` and `src/prc/train.py`, and inspected the parquet files:

- `month_block_folds()` builds folds `jan`, `jul`, optionally `jul_fwd`, and `forward`. Each fold
  carries its **own** `train_months`. `jan` trains on *every other month including July, November and
  December* — i.e. on the future. `forward` trains only on months < November. The docstring says so
  explicitly: the trailing months "form a third fold that measures forward generalisation", and
  `jul_fwd` exists so that "the cost of the leak be measured rather than assumed".
- `train.py:486–510` builds the `oof` column by initialising it to null over all rows and then, fold
  by fold in list order, `coalesce`-ing in that fold's predictions. **First fold to cover a row
  wins** — so where `jul` and `jul_fwd` both validate July, the ordering silently decides which
  model's prediction lands in the file.
- Schema: the OOF parquet has exactly two columns, `MVT_ID_mvt` and `oof`. **There is no fold
  label.** Nothing downstream can tell which fold (and therefore which training protocol) produced a
  given row.
- Coverage differs between experiments. All files have 2,085,047 rows with unique, non-duplicated
  `MVT_ID_mvt`, but the non-null counts differ:
  - `oof_main-f9d57a661f.parquet`: **672,428** non-null (three folds)
  - `oof_feat_base-b9e21a829d.parquet`: **328,009** non-null (forward only)
  - `oof_inv_baseline_forward-9271caca22.parquet`: **328,009** non-null (forward only)

**Conclusion [X], grounded in §5.1–5.2:** our OOF file is *not* invalid as an OOF file — the
row-level no-leak invariant holds. But it is **not currently usable as an NNLS design matrix**, for
three separate reasons, in order of severity:
1. Candidate experiments do not share a row set (672,428 vs 328,009 non-null). A naive join would
   silently fit the weights on the 328,009-row **forward fold only**, which is the *hardest* and
   *least test-like* block, without saying so.
2. The three blocks were produced under materially different training protocols (`jan` trains on the
   future; `forward` does not), so their error levels are not comparable and a single global weight
   vector is fitted across incommensurable columns.
3. Without a fold label, none of this is detectable from the file.

All three are fixable without changing the fold design: emit a fold column, and fit weights only
across models run with an identical fold list, restricted to the intersection of non-null rows,
with per-fold weights reported alongside the pooled ones as a consistency check.

---

## 6. How competitors decide WHICH models to include

**Sourced practice:**

1. **Weight-based pruning after fitting, not correlation screening before it.** [S] The two PRC
   blends both let the optimiser do the selecting: the PRC 2024 QP drove three of five members to
   ~1e-23, and `jubilant-vase` explicitly drops any member with `weight < 0.01` before inference.
   The PRC 2024 paper states this as a design property: the QP "can completely exclude models that do
   not add predictive value beyond what is already captured by other models in the ensemble."
2. **Ablation as the acceptance test.** [S] The PRC 2024 paper's Table 8 removes each algorithm from
   the fitted ensemble and reports the RMSE change. That is the measurement that told them XGBoost
   was worth +0.16% and LightGBM +0.00%. This is the cleanest published "is this member earning its
   place?" procedure I found in the aviation literature.
3. **Diversity by construction, not by metric.** [S] MLWave: average "less correlated submissions".
   NVIDIA playbook: build "a diverse set of baselines across model types" — "linear models, GBDTs,
   and even small neural nets". `hillclimbers`: the algorithm "chooses diverse models" rather than
   the highest-CV models. ASHRAE 1st and 2nd place manufactured diversity by training the *same*
   libraries on *different data subsets* (by meter, primary use, site id) rather than only by
   swapping libraries.
4. **Greedy selection with replacement *is* the selection rule** in every major AutoML system. [S]
   Q(D)O-ES surveyed AutoML systems: all but one that do post hoc ensembling "rely on greedy ensemble
   selection with replacement (GES)"; four systems (including LightAutoML) "return the single best
   model" instead. Auto-sklearn's authors "stated that GES outperforms alternatives such as stacking
   or gradient-free numerical optimisation."

**On published correlation thresholds — an explicit negative finding [S]:**
I found **no** published numeric rule of thumb of the form "include a model if OOF residual
correlation < X". MLWave states the principle with no threshold. More importantly, the primary
literature warns against trusting diversity metrics: **Kuncheva & Whitaker (2003), "Measures of
diversity in classifier ensembles and their relationship with the ensemble accuracy",
*Machine Learning* 51:181–207** (<https://research.bangor.ac.uk/en/publications/measures-of-diversity-in-classifier-ensembles-and-their-relations/>)
examined 10 diversity measures and found no clear or definitive relationship between diversity and
ensemble accuracy; the use of diversity measures to design ensembles remains an open question.
Q(D)O-ES reaches the same place from the other direction: "respecting the diversity of ensembles
during optimisation can be beneficial but **also increases the risk of overfitting**", and it cites
a literature finding "a trade-off between ensemble diversity and performance".

**[X] Therefore:** OOF residual correlation is a useful *diagnostic* for explaining why a blend did
or didn't help, but there is no sourced basis for using a correlation threshold as an *inclusion
rule*. The sourced inclusion rule is: fit the weights, prune near-zero members, and ablate each
surviving member to confirm it earns its weight out of sample.

---

## Applicable to resilient-hedgehog

Current state: one LightGBM, `n_seeds: 3` (mechanism F only); `catboost` and `xgboost` declared in
`pyproject.toml` and used nowhere; OOF files carry no fold label and differ in coverage
(672,428 vs 328,009 non-null). LB 284.3584. Targets above us: 276.99 (−7.4) and 273.56 (−10.8).

Sourced expectation to hold in mind before starting: the blend mechanisms in §2 are reported to buy
**0.3–1.8% over the best single model** on problems like ours, i.e. **0.9–4.5 RMSE for us** [X].
That closes part of the 7.4-point gap to OpenAir and not all of it. In PRC 2025 the non-blending
winner beat the blending runner-up by 11.2 points. Blending is worth doing and is not, on this
evidence, sufficient on its own.

Ordered path. Each step names the measurement that verifies it; do not proceed past a step whose
measurement fails.

**Step 0 — Make the OOF file blendable. (Prerequisite; blocks everything else.)**
Add a `fold` column to the OOF parquet written at `train.py:534`, and record the fold list in the
experiment record. Nothing else changes.
*Verify:* re-run `main.yaml`; the new OOF has the same 672,428 non-null rows, `rmse` unchanged to
4 decimal places against the existing record, and `fold` takes exactly the values in the config's
fold list with block sizes that sum to 672,428.
*Why:* §5.1 — H2O and sklearn both make a shared, identifiable partition a precondition for fitting
a meta-learner. §5.3 — without the label, reason (3) makes reasons (1) and (2) undetectable.

**Step 1 — Establish the blend harness on models we already have, before adding any library.**
Take two existing LightGBM variants whose OOF files were produced with the *same* fold list (check
non-null counts match exactly), fit weights with SciPy exactly as PRC 2025 rank 2 does —
`minimize(rmse, w0, bounds=[(0,1)]*n, constraints={'type':'eq','fun': lambda w: sum(w)-1})` — on the
intersection of non-null rows, and prune members with `w < 0.01`.
*Verify:* report the blend's OOF RMSE **per fold** (`jan`, `jul`, `forward`) as well as pooled, plus
the weight vector. Two acceptance criteria, both required: (a) the blend beats the best member on the
pooled OOF; (b) **the per-fold weight vectors fitted independently on each fold agree to within
±0.10**. If (b) fails, §5.2's reason (2) is live and a single global weight vector is not defensible —
fit weights on the fold that most resembles the test months and say so.
*Why:* this tests the harness where we already know the answer (two near-identical LightGBMs should
produce weights near 0.5/0.5 and almost no gain), so a surprising result indicts the harness, not the
models.

**Step 2 — Add CatBoost as the second family, trained through the *identical* fold list.**
CatBoost first, not XGBoost: in the two directly comparable aviation competitions CatBoost took the
largest fitted weight both times (0.865 in PRC 2024 [S]; 0.641 in PRC 2025 [S]), was the best single
model both times, and its removal from the PRC 2024 ensemble cost +33.90% while XGBoost's cost
+0.16% [S]. It is also the family the top three public PRC 2026 competitors all carry.
*Verify:* (a) CatBoost's own pooled OOF RMSE, reported next to LightGBM's — the PRC 2025 precedent is
that the members land within ~0.8% of each other (244.9 / 245.4 / 246.8) and the blend gain came
from that parity; if CatBoost lands more than ~3% worse than our LightGBM, expect the PRC 2024
outcome instead (one member takes ~0.87 weight, gain ~0.3%). (b) The two-member blend's pooled OOF
RMSE vs LightGBM alone. Sourced target band: **−0.5% to −1.8%**, i.e. **283.0 to 279.2** at our
current level [X]. (c) Pearson correlation of the two OOF residual vectors, recorded as a
*diagnostic* — §6 is explicit that there is no sourced threshold to gate on (Kuncheva & Whitaker
found no reliable diversity→accuracy relation), so this number explains the gain, it does not
authorise the decision.

**Step 3 — Ablate before you ship, exactly as PRC 2024 did.**
Refit the blend with each member removed and report the RMSE change, reproducing the form of their
Table 8.
*Verify:* a member stays only if removing it costs a measurable amount. The PRC 2024 paper pruned
three of five members on exactly this test (+0.00% each). Applied to us: if the two-member blend's
gain over LightGBM alone is under ~0.2 RMSE, stop here — the blend is not earning its complexity and
the effort belongs in slots 12/13/17/18.

**Step 4 — Only then consider XGBoost as a third member.**
Its sourced marginal contributions: weight 0.134 and +0.16% ablation cost in PRC 2024 [S]; weight
0.160 in PRC 2025 [S]; 5.3% weight in OpenAir's NNLS [S].
*Verify:* the same ablation as Step 3. Sourced expectation is a gain in the **0.1–0.5 RMSE** range
for us [X]. Accept it only if the ablation cost exceeds the run-to-run noise of our own CV (measure
that noise first with the existing `n_seeds: 3` spread).

**Step 5 — Protect the fitted weights against §4.1.**
Once there are ≥4 candidate members, the fitted-linear solution starts to overfit the OOF — this is
the failure Caruana's §2 and Q(D)O-ES were both written about. Sourced mitigations, in the order
they cost least:
(a) keep the non-negativity and sum-to-one constraints (already in Steps 1–4; Breiman's original
point);
(b) prune `w < 0.01` (PRC 2025 rank 2);
(c) switch from SciPy least squares to **greedy selection with replacement** — this is what
AutoGluon and Auto-sklearn use *instead of* a fitted linear stacker, precisely "for its resilience
against over-fitting" [S], and Auto-sklearn's authors report it beats "stacking or gradient-free
numerical optimisation" [S];
(d) if (c) is adopted, add sorted initialization with the best N members and bagged selection at
p = 0.5 over 20 bags — Caruana measures the bagging alone at an extra 2.5% [S].
*Verify:* the honest test is not pooled OOF, which the weights were fitted on. Fit weights on
{`jan`, `jul`} and score on `forward`, and vice versa. If the held-out-fold score does not improve,
the blend gain is OOF overfitting — which is exactly the failure Q(D)O-ES measured across 71
datasets and named as "the lack of generalisation from validation to testing sets."

**Step 6 — Decide between fold-bagged and refit-on-full inference, explicitly.**
Both are used by PRC prize winners: PRC 2025 rank 2 averages the 5 fold models at inference [S];
PRC 2024 tiny_rainbow refits every member on the full dataset with the pre-computed weights frozen
[S]. No source compares them.
*Verify:* run both against the leaderboard once each and keep the winner, rather than reasoning
about it. Note that whichever we choose, the existing `n_seeds: 3` seed averaging composes with it
and should be kept — it is mechanism F and is orthogonal to C/D.

**A standing caveat to carry through all six steps.**
Every sourced blend gain in §3 is ≤1.8%, while in the same PRC 2025 competition the difference
between the winner and the blending runner-up was 5.3%, and in PRC 2024 a single feature group was
worth 24.83% against XGBoost's 0.16%. If Steps 2–4 deliver the sourced band and we land near 280,
that is the *expected* outcome of this slot, not a shortfall — the remaining 6–7 points to the top
of the public board are, on the published evidence, in features, per-airport specialisation and
target handling, not in the blend.
