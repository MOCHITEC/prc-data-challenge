# Slot 20 — tabular-comp-craft

General craft of tabular competitions (2024–2026 consensus), focused on what
`resilient-hedgehog` has **not** done.

**Marking convention**

* `[S]` = read from a source; URL given inline.
* `[S-repo]` = read from this repository; `path:line` given inline. (Facts about
  our own pipeline, verified by reading the code, not inferred.)
* `[X]` = my own extrapolation. Never dressed as `[S]`.

Every numeric gain below names **whose** result it is and **on what problem**.

A standing caveat that applies to almost every benchmark cited here, stated once:
the large public tabular benchmarks are small-data. TabArena-v0.1 spans
748–150,000 rows `[S]` (<https://arxiv.org/html/2506.16791v2>); TabRepo's 200
datasets have **median 3,800 rows**, 75th percentile 10,027, max 400,000 `[S]`
(Table 3, <https://arxiv.org/pdf/2311.02971>). Our training set is ~2.08M rows.
Ensembling and regularisation gains are largely variance-reduction gains, and
variance falls with n, so benchmark deltas measured at 4k rows are an **upper
bound** on what the same trick buys at 2M rows `[X]`.

---

## 1. What the 2024–2026 consensus actually is

`[S]` ML Contests, *The State of Machine Learning Competitions 2024*
(<https://mlcontests.com/state-of-machine-learning-competitions-2024/>):
among 2024 competition winners, GBDTs dominated tabular problems — **16 LightGBM
wins, 13 CatBoost wins, 8 XGBoost wins**. Winners commonly used *multiple*
libraries together "as their different implementations produced complementary
strengths". Neural networks appeared as **secondary components** of ensembles
(Denselight, TabM, TabNet), not as primary solutions. Tabular foundation models
(TabPFN, Chronos, Moirai) saw **no adoption among 2024 winners**.

`[S]` ML Contests, *State of Machine Learning Competitions 2025*
(<https://mlcontests.com/state-of-machine-learning-competitions-2025/>):
"XGBoost and LightGBM were most popular this year, with **14 uses each**";
CatBoost in **8** winning solutions. GBDTs remain "the go-to tabular competition
winner's modelling tool, sometimes as part of an ensemble alongside neural nets."
Two notable 2025 data points from the same report:
* One winner reported "XGBoost consistently performed best, with LightGBM adding
  useful diversity; **CatBoost underperformed**" — i.e. the library ranking is
  dataset-specific, not universal.
* The winner of Kaggle's Open Polymer Prediction 2025 found **AutoGluon beat a
  hand-built Optuna-tuned ensemble of XGBoost + LightGBM + TabM** while using a
  fraction of the training budget.
* DrivenData PREPARE challenge winners used **TabPFN to generate features that
  were fed into GBDT models** — foundation model as feature generator, not as
  predictor.

`[S]` NVIDIA, *The Kaggle Grandmasters Playbook: 7 Battle-Tested Modeling
Techniques for Tabular Data*
(<https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/>).
The seven, with what each is credited for:
1. **Smarter EDA** — train/test distribution checks and temporal target analysis.
2. **Diverse baselines** — linear + GBDT + NN simultaneously. Credited with 2nd
   place in Kaggle's Binary Prediction with Rainfall Dataset using "an ensemble
   of gradient-boosted trees, neural nets, and SVR, **without any feature
   engineering**".
3. **Generate thousands of features** — e.g. "combining pairs of eight
   categorical columns created 28 new categorical features"; credited with first
   places in the Backpack and Insurance competitions.
4. **Hill-climbing ensembles** — greedy weight search; 1st place, Predict Calorie
   Expenditure.
5. **Stacking** — meta-model on OOF predictions or on residuals; 1st place,
   Predict Podcast Listening Time.
6. **Pseudo-labelling**.
7. **Extra training** — multi-seed averaging and 100 % full-data retrain after
   HPO. Quantified: on Predicting Optimal Fertilizers, "ensembling XGBoost models
   across **100 different seeds** clearly outperformed single-seed training",
   MAP@3 ≈ 0.379 vs ≈ 0.376 single-seed.

   Note for us: we already do #7's seed-averaging, but at **n_seeds: 3**
   `[S-repo]` (`configs/main.yaml:21`), not 100. The NVIDIA figure is the only
   sourced number I found on seed-count returns and it is a classification metric
   on a different problem.

`[S]` Chris Deotte's 1st place, Kaggle Playground S5E4 (Predict Podcast Listening
Time), described in NVIDIA's *Grandmaster Pro Tip: Winning First Place ... with
Stacking Using cuML*
(<https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-stacking-using-cuml/>):
a **three-level stack of 75 Level-1 models chosen from 500 experimental models**
(GBDTs, NNs, SVR, KNN, TabPFN, Ridge, RF), Level 2 = one GBDT + one NN meta-model,
Level 3 = weighted average. CV RMSE 11.54, private LB RMSE 11.44. The article
does **not** report the best single model's score, so it does not by itself
quantify the stack's gain.
Counterpoint in the same competition: 2nd place was titled *"Single LightGBM and
Target Encoding"* `[S]` (title only; the writeup body is behind Kaggle's
JS-rendered API and I could not retrieve it —
<https://www.kaggle.com/competitions/playground-series-s5e4/writeups/farukcan-saglam-2nd-place-single-lightgbm-and-targ>).
So in that competition a **72–75-model stack beat a single target-encoded
LightGBM by one leaderboard place**, magnitude unknown to me.

**Retrieval limitation, stated honestly:** Kaggle writeup and notebook bodies are
client-rendered; `WebFetch` and `curl` both returned only the page shell (6 KB, no
content). Everything below that is attributed to Kaggle is attributed to a *title*
or to a *secondary source that quotes it*, never to my having read the writeup.

---

## 2. CatBoost's ordered target statistics, and why it differs most from LightGBM
   on many-level categoricals

Our data has airport (10), airline, aircraft type, stand, runway — exactly the
shape CatBoost was designed for. We have **no CatBoost** `[S, given]`.

### 2.1 The mechanism

`[S]` Prokhorenkova et al., *CatBoost: unbiased boosting with categorical
features* (<https://arxiv.org/pdf/1706.09516>):
* The naive category encoding ("**Greedy TS**") replaces a category with the mean
  target of training rows sharing that category, smoothed by a prior:
  `x̂ᵢₖ = (Σⱼ 1{xᵢⱼ=xᵢₖ}·yⱼ + a·p) / (Σⱼ 1{xᵢⱼ=xᵢₖ} + a)`.
  Because row *k*'s own `yₖ` is in the numerator, the training rows see their own
  label through the feature. The paper names this a **"special kind of target
  leakage present in all currently existing implementations of gradient
  boosting"** and shows it produces a **prediction shift** — the model's
  conditional distribution on training rows differs from that on test rows.
* **Ordered TS** fixes it by imposing an artificial random permutation and
  computing each row's statistic only from rows *earlier in the permutation*.
  **Ordered boosting** applies the same permutation trick to the gradient
  estimates.
* On LightGBM's handling specifically, the paper notes LightGBM "converts
  categorical features to gradient statistics at each step", which is expensive
  and so LightGBM "**groups tail categories into one cluster and thus looses part
  of the information**".

`[S]` LightGBM's own documentation states: *"For a categorical feature with high
cardinality, it often works best to treat the feature as numeric"*
(<https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html#categorical-feature-support>,
quoted and retrieved 2023-06-30 by Sigrist, below).

### 2.2 What it buys — the honest spread

**Evidence that CatBoost wins on categorical-heavy data:**

`[S]` CatBoost paper Table 2 (logloss / zero-one loss; relative increase for the
baselines; **all three libraries given ordered-TS-encoded categoricals**, so this
measures the *algorithm*, not the encoding):

| dataset | CatBoost | LightGBM | XGBoost |
|---|---|---|---|
| Amazon (all-categorical, high-cardinality) | 0.139 / 0.044 | **+17 % / +21 %** | +17 % / +21 % |
| Internet | 0.209 / 0.094 | +6.8 % / +8.6 % | +7.9 % / +8.0 % |
| Kick | 0.286 / 0.095 | +3.5 % / +4.4 % | +3.2 % / +4.1 % |
| Adult | 0.270 / 0.127 | +2.4 % / +1.9 % | +2.2 % / +1.0 % |
| Click | 0.392 / 0.156 | +1.2 % / +1.2 % | +1.2 % / +1.2 % |
| Appetency / Churn / Upselling | — | +0.1–0.4 % | +0.04–0.5 % |

Improvements are statistically significant at p ≪ 0.01 except on Appetency,
Churn, Upselling.

**Important size caveat from the same paper** `[S]`: Table 3 compares Ordered vs
Plain boosting mode inside CatBoost. "Ordered mode is particularly useful on
**small** datasets. Indeed, the largest benefit from Ordered is observed on Adult
and Internet datasets, which are relatively small (**less than 40K training
examples**)". At 2.08M rows, CatBoost's *ordered boosting* is expected to buy
little; its *ordered TS categorical encoding* is a separate mechanism and does not
carry the same size caveat `[X]`.

**Evidence that CatBoost does *not* automatically win on high-cardinality
regression:**

`[S]` Sigrist, *A Comparison of Machine Learning Methods for Data with
High-Cardinality Categorical Variables* (<https://arxiv.org/pdf/2307.02071>).
8 real regression datasets, n = 28k–125k, categorical variables with **553 to
72,000 levels** (host, director, artist, store, commodity, person). 5-fold CV,
test MSE, grid-searched tuning for every boosting method. Results (Table 2):

| method | avg. rank | avg. rel. diff. to best |
|---|---|---|
| GPBoost (tree-boosting **+ random effects**) | **2.14** | **7.02 %** |
| LGBM Cat (LightGBM native categorical) | 2.71 | 17.4 % |
| CatBoost | 2.86 | 44.5 % |
| LMMNN (NN + random effects) | 3.57 | 17.3 % |
| Linear mixed effects | 4.71 | 47.6 % |
| LGBM Num (categories as integers) | 5.57 | 111 % |
| NN with embeddings | 6.43 | **189 %** |

Three things to take from this, all `[S]`:
1. On **regression** with genuinely high-cardinality groups, LightGBM's native
   categorical handling and CatBoost are **within one rank of each other**;
   CatBoost's average relative difference is worse, driven by one dataset
   (AUimport: CatBoost 2.15 vs LGBM Cat 1.26 vs GPBoost 0.651).
2. Treating high-cardinality categories as **integers is catastrophic** (111 %).
   The paper explicitly contradicts LightGBM's own documentation on this point.
3. **Neural networks with embeddings were the worst method tested** — 189 % worse
   than best on average, and worse than a linear mixed model.
4. The winner is a family we have not tried at all: **tree-boosting with random
   effects** (GPBoost / MERF), which puts a random intercept (and optionally a
   random slope in time) on each level of the high-cardinality variable and fits
   the boosted trees to the residual. 7.02 % vs 17.4 % for LGBM Cat is the
   largest single gap in the table.

### 2.3 Net reading

`[X]` The case for adding CatBoost here is **not** "CatBoost handles categoricals
better than LightGBM" as a law — Sigrist's regression benchmark refutes that as a
law and ML Contests 2025 records a 2025 winner for whom CatBoost underperformed.
The case is (a) CatBoost is the **single most decorrelated GBDT** from LightGBM,
because ordered TS and symmetric oblivious trees differ from leaf-wise growth on
raw categoricals in both bias and variance, so it is the cheapest source of blend
diversity; and (b) in *this competition family* it was decisively the strongest
single model — see §11.

---

## 3. XGBoost's role in modern blends

`[S]` TabRepo Table 1 (200 OpenML datasets, 4 h budget, normalized error, **lower
is better**; <https://arxiv.org/pdf/2311.02971>):

| method | normalized error | rank |
|---|---|---|
| Portfolio (ensemble, cross-family) | **0.365** | 168.7 |
| AutoGluon | 0.389 | 208.2 |
| CatBoost (tuned + ensemble of 20 configs) | 0.524 | 267.3 |
| CatBoost (tuned) | 0.534 | 284.7 |
| LightGBM (tuned + ensemble) | 0.534 | 268.7 |
| LightGBM (tuned) | 0.566 | 304.2 |
| CatBoost (default) | 0.586 | 341.2 |
| MLP (tuned + ensemble) | 0.594 | 402.5 |
| **XGBoost (tuned + ensemble)** | **0.628** | 357.9 |
| XGBoost (tuned) | 0.638 | 376.5 |
| FT-Transformer (default) | 0.690 | 532.1 |
| LightGBM (default) | 0.714 | 491.5 |
| XGBoost (default) | 0.734 | 522.2 |
| MLP (default) | 0.772 | 629.4 |
| TabPFN (default) | 0.837 | 731.9 |

So **XGBoost is the weakest of the three GBDTs as a standalone** on this suite.
Its value is diversity, and TabRepo says so directly `[S]`: "some model families
are **negatively correlated** between each other such as **MLP and XGBoost**,
which hints at the potential benefit of ensembling."

`[S]` ML Contests 2025: XGBoost tied LightGBM for most-used library among
winners (14 each), and one winner reported XGBoost as their *best* single model
with LightGBM "adding useful diversity".

`[S]` PRC Data Challenge 2024, team_tiny_rainbow ablation (below, §4.2): removing
XGBoost from their weighted ensemble cost **+0.16 % RMSE** — small but non-zero,
and it was the only model besides CatBoost with non-negligible weight.

`[X]` Read: XGBoost is a cheap third opinion, not a contender for our primary
model. Its expected marginal value here is smaller than CatBoost's.

---

## 4. Neural tabular models — what they buy *in a blend*, not standalone

### 4.1 Benchmark evidence

`[S]` Grinsztajn, Oyallon & Varoquaux, NeurIPS 2022 D&B, *Why do tree-based
models still outperform deep learning on typical tabular data?*
(<https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html>):
on 45 datasets, tree-based models remain SOTA on medium-sized data (~10k rows);
the gap is **not** mostly attributable to categorical features and **does not
disappear after hyperparameter tuning**.

`[S]` TabArena v0.1, NeurIPS 2025 D&B (51 datasets, 748–150k rows, 10×3-fold CV;
<https://arxiv.org/html/2506.16791v2>, paper text and Table A.1). Elo, higher is
better:

| model | default | tuned | tuned + post-hoc ensembled |
|---|---|---|---|
| CatBoost | **1432** | **1475** | 1489 |
| TabICL | 1388 | — | — |
| TabM | 1348 | 1455 | 1552 |
| TabPFNv2 | 1317 | 1346 | — |
| ModernNCA | 1302 | — | 1435 |
| RealMLP | 1246 | 1349 | **1569** |
| XGBoost | 1229 | 1409 | 1446 |
| LightGBM | 1206 | 1450 | 1532 |

(Elo figures read from Table A.1 of the arXiv HTML; the paper's own prose states
the same ordering: *"CatBoost is ranked first in the conventional tuning regime"*
and *"the top three models in our leaderboard (TabM, LightGBM, RealMLP) would all
be worse than the actual fourth-best model (CatBoost) without post-hoc
ensembling"* `[S]`.)

Three readings, all `[S]`:
1. **CatBoost's default is far and away the best default** — 1432 vs LightGBM's
   1206, a 226-Elo gap that tuning closes to 25 Elo.
2. **Neural models only overtake GBDTs once you ensemble multiple configurations
   of them.** The deep models' tuning + post-hoc-ensembling gain is much larger
   than the GBDTs' (TabArena's own summary: tuning bought roughly 40–60 Elo for
   GBDTs and 100+ for deep models, and post-hoc ensembling "can dramatically
   improve performance").
3. "A simulated ensembling pipeline using all models in TabArena outperforms all
   individual models", and "lower-ranked models are not necessarily unnecessary"
   — they contribute ensemble diversity.

`[S]` OmniTabBench (3,030 datasets; <https://arxiv.org/html/2604.06814v1>):
"GBDT, NN, and TabPFN wins (ranks the best) on **31.6 %, 33.9 %, and 34.5 %** of
the 1,815 datasets, respectively" — a three-way near-tie overall. Conditions
favouring NNs: **larger sample sizes** (mean #rows 8,260) and higher data density.
Conditions favouring GBDTs: **feature skewness and kurtosis** (|γ| = 0.796,
κ = 1.510) and raw/unnormalised data. TabPFN wins on the *smallest* datasets
(mean 2,773 rows) and *lowest target skew*.
`[X]` Our target — taxi-out seconds — is strongly right-skewed and heavy-tailed,
which is the GBDT-favouring condition, not the NN-favouring one.

### 4.2 The decisive evidence *from this competition family*

The brief states that PRC 2024's 3rd place "stacked SAINT and an MLP with GBDTs".
The published paper shows that they **built** those models but the fitted blend
gave them **essentially zero weight**.

`[S]` Murça, Maximo, Dantas, Szenczuk, Rutili de Lima, Carvalho & de Melo (team
`tiny_rainbow`, ITA), *Open Machine Learning Models for Actual Takeoff Weight
Prediction*, Journal of Open Aviation Science
(<https://journals.open.tudelft.nl/joas/article/view/7963>, PDF galley
<https://journals.open.tudelft.nl/joas/article/download/7963/6296/31267>).
PRC Data Challenge 2024 (predict ATOW; metric RMSE). Their Table 6:

| algorithm | estimated test RMSE | actual RMSE | MAPE |
|---|---|---|---|
| Baseline (MTOW) | 32,324.13 | 32,324.13 | 26.10 % |
| MLP (FastAI Tabular Learner) | 3,864.1 | 3,864.2 | 4.54 % |
| **SAINT** | 3,921.3 | 3,921.5 | 4.63 % |
| **CatBoost** | **2,215.53** | 2,224.69 | 1.75 % |
| XGBoost | 2,413.05 | 2,403.07 | 2.42 % |
| LightGBM | 2,506.43 | 2,519.17 | 2.50 % |
| **Ensemble (weighted)** | **2,210.37** | 2,217.75 | **1.73 %** |

Fitted ensemble weights, in the table's own model order
[MLP, SAINT, CatBoost, LightGBM, XGBoost]:
**[1.24e-23, 1.63e-23, 8.65e-01, 4.77e-22, 1.34e-01]**.

Their ablation (Table 8), removing one component from the ensemble:

| ablated | RMSE | Δ |
|---|---|---|
| Baseline (full model) | 2,210.37 | 0.00 % |
| Operational (ADS-B) features | 2,759.18 | **+24.83 %** |
| Geographical features | 2,390.26 | +8.14 % |
| Temporal features | 2,354.97 | +6.54 % |
| Aircraft characteristics | 2,266.96 | +2.56 % |
| Meteorological features | 2,249.29 | +1.76 % |
| **CatBoost** | 2,969.49 | **+33.90 %** |
| **XGBoost** | 2,213.91 | +0.16 % |
| **LightGBM** | 2,210.37 | **+0.00 %** |
| **Neural Network (MLP)** | 2,210.37 | **+0.00 %** |
| **SAINT** | 2,210.37 | **+0.00 %** |

The paper's own conclusion `[S]`: "The unchanged RMSE values (0.00 % change) for
LightGBM, Neural Network, and SAINT algorithms suggest these components might
benefit from further optimization or may be **redundant in the current ensemble
configuration, and thus pruned**."

Four facts fall out of this, all `[S]`, and they are the most directly relevant
numbers in this whole document:
1. On a PRC-family regression problem, **the neural models were ~75 % worse than
   the GBDTs standalone** (3,864 / 3,921 vs 2,216) and contributed **exactly
   nothing** in the blend (weights ~1e-23).
2. **CatBoost beat their LightGBM by 290.9 RMSE = 11.6 % relative** (2,215.53 vs
   2,506.43), both Optuna-tuned with TPE over comparable search spaces (their
   Tables 3–4).
3. **The entire blend bought 5.16 RMSE over CatBoost alone** — 2,215.53 → 2,210.37,
   **0.23 %** (0.31 % on the actual-submission numbers, 2,224.69 → 2,217.75).
4. **Features beat models by an order of magnitude**: the largest model ablation
   that mattered (+33.9 %) was just "you removed the only model with weight", while
   removing the ADS-B operational feature group cost +24.83 % of a well-blended
   model.

### 4.3 Foundation models are out of range for us

`[S]` TabArena: TabPFNv2 is limited to **≤10,000 training samples**, ≤500
features; TabICL to ≤100,000 samples
(<https://arxiv.org/html/2506.16791v2>).
`[S]` TabPFN-2.5 technical report (<https://arxiv.org/abs/2511.08667>) raises this
to "**up to 50,000 data points and 2,000 features**", with an 87 % win rate vs
default XGBoost on datasets up to 100k×2k (85 % for regression), and claims to
match 4-hour AutoGluon 1.4.
`[X]` At 2.08M rows we are ~40× above TabPFN-2.5's stated ceiling. The only
in-range use is the DrivenData PREPARE pattern `[S]` (ML Contests 2025): fit
TabPFN on a small subsample **to generate features** fed to the GBDT. Whether
that survives at our scale is untested.

---

## 5. Hyperparameter tuning: how much is actually there

### 5.1 How much tuning buys

`[S]` Probst, Boulesteix & Bischl, *Tunability: Importance of Hyperparameters of
Machine Learning Algorithms*, JMLR 2019 (<https://arxiv.org/abs/1802.09596>;
38 OpenML datasets, AUC). Their Table 2 — "tunability" = mean AUC gained by
per-dataset tuning, relative to a fixed default:

| algorithm | vs **package** defaults (Tun.P) | vs **meta-learned optimal** defaults (Tun.O) |
|---|---|---|
| glmnet | 0.069 | 0.024 |
| svm | 0.056 | 0.042 |
| **xgboost** | **0.043** | **0.014** |
| rpart | 0.038 | 0.012 |
| kknn | 0.031 | 0.006 |
| ranger (RF) | 0.010 | 0.006 |

The headline: for XGBoost, **two-thirds of tuning's total value is simply
escaping bad package defaults** (0.043 → 0.014 once defaults are set sensibly).
Per-hyperparameter tunability for xgboost (their Table 3, vs optimal defaults):
`booster` 0.008, `eta` 0.005, `nrounds` 0.002, `subsample` 0.002,
`min_child_weight` 0.002, `max_depth` 0.001, `colsample_bytree` 0.001,
`colsample_bylevel` 0.001. **No single tree-shape hyperparameter is worth more
than 0.008 AUC on average.**

`[S]` TabRepo Table 1 (200 datasets, normalized error) says the same thing in a
second currency: LightGBM default 0.714 → tuned (4 h random search) 0.566 → tuned
+ ensembling its own top-20 configs 0.534. So on that suite, **tuning bought 0.148
and self-ensembling the tuned configs bought a further 0.032** — tuning is worth
~4.6× the intra-family config-ensembling, but see §11 for what cross-family
ensembling buys on top.

`[S]` TabRepo also ran fANOVA hyperparameter-importance: "hyperparameter
importance mostly follows practitioner intuition… the **depth**" is the top
hyperparameter for CatBoost.

`[S]` Holzmüller et al., *Better by Default: Strong Pre-Tuned MLPs and Boosted
Trees on Tabular Data*, NeurIPS 2024 (<https://arxiv.org/abs/2407.04491>): meta-
tuned defaults were derived on 118 datasets and validated on 90 disjoint ones
(1k–500k rows); the conclusion is that "a combination of RealMLP and GBDTs with
**improved default parameters** can achieve excellent results **without
hyperparameter tuning**."

### 5.2 Search method

`[S]` ML Contests 2025: **Optuna** appears repeatedly in winning tabular
solutions. `[S]` team_tiny_rainbow used Optuna with the **Tree-structured Parzen
Estimator** for CatBoost, LightGBM and XGBoost (their §2.4, Tables 3–5).
`[S]` Chris Deotte's S5E5 1st place is titled *"GPU Hill Climbing"* and the
NVIDIA playbook describes **hill climbing** (greedy weight search over candidate
models) as technique #4 — this is used for *blend weights*, not for
hyperparameters.

### 5.3 The standard warning, and why it applies here hard

`[S]` Cawley & Talbot, *On Over-fitting in Model Selection and Subsequent
Selection Bias in Performance Evaluation*, JMLR 11 (2010)
(<https://www.jmlr.org/papers/v11/cawley10a.html>): optimising a model-selection
criterion on finite validation data overfits that criterion, and the resulting
degradation is **"often of comparable magnitude to differences in performance
between learning algorithms"**. They recommend nested CV / an independent
hold-out, not reusing the selection folds for reporting.

`[S]` Deotte's S5E4 stack selected **75 models from 500 experimental models**
scored on the same folds — the same hazard at scale, mitigated only by the volume
of data behind each fold.

`[S-repo]` This warning is not theoretical for us. Our own notes record CV and
the leaderboard pointing in **opposite directions**: `configs/main.yaml:1-8`
states that CV rewarded the 255-leaf/min-data-20 model B by **34.8 s** while the
leaderboard **charged 13.2 s** for it (v2 312.02 → v5 325.22), and that the
robustness check turned the 34.8 s into −0.06 s once 200 rows were removed.
`src/prc/adversarial.py:19-29` records adversarial AUC **0.807** with calendar
columns removed (0.819 restricted to Jan/Jul), only **5.9 %** of training rows
scoring above the ranking set's median, and — importantly — that reweighting by
that probability **did not** reproduce the leaderboard's ranking.
`[X]` Under a CV signal that has already been caught disagreeing with the
leaderboard by a magnitude (34.8 s) far larger than XGBoost's whole tunability
budget, a large Optuna search is close to the worst possible use of compute: it
is a maximisation over exactly the noisy criterion Cawley & Talbot warn about.

---

## 6. Feature selection: does removing features help, and by what procedure?

`[S]` LOFO (Leave One Feature Out) importance, Ahmet Erdem
(<https://github.com/aerdem4/lofo-importance>): iteratively removes each feature,
retrains, and evaluates under **your own validation scheme**, reporting mean and
sd across folds. Claimed properties: it "does not favor granular features",
"generalises well to unseen test sets", is model-agnostic, and — the operative
one — "**gives negative importance to features that hurt performance upon
inclusion**". Cited competition evidence: in **Microsoft Malware Prediction**,
LOFO correctly assigned *negative* importance to seasonal OS-version features
under time-split validation, "unlike other methods that overestimated their
value"; in **TReNDS** it grouped correlated fMRI features. FLOFO permutes *within
groups of similar samples* rather than globally, to avoid scoring unrealistic
feature combinations.

`[S]` Null importance (Olivier Grellier's method,
<https://www.kaggle.com/code/ogrellier/feature-selection-with-null-importances>;
described secondhand at
<https://github.com/VirajBagal/IEEE-Fraud/blob/master/feature-selection-using-null-importance.ipynb>):
fit the model to get actual importances, then **shuffle only the target** and
refit many times to build a null distribution per feature; keep features whose
actual importance is far in the tail of their own null. Attributed to
**IEEE-CIS Fraud Detection**.

`[S]` Adversarial validation (also IEEE-CIS Fraud Detection, per
<https://dev.to/kamil_k7k/tabular-data-binary-classification-all-tips-and-tricks-from-5-kaggle-competitions-1aim>):
train a classifier to separate train rows from test rows; a high AUC means the
sets differ, and the classifier's importances name the offending columns, which
are then dropped or down-weighted. The same article attributes permutation
importance and SHAP-based selection to IEEE-CIS Fraud and Microsoft Malware
respectively.

**Counter-consensus, which must be said:** `[S]` the NVIDIA Grandmasters Playbook
does **not** list feature selection among its seven techniques at all; its
technique #3 is the opposite — "generate thousands of features… and let the model
decide what works". `[X]` The reconciliation is that selection pays when the
validation signal is time- or distribution-shifted (the Microsoft Malware case),
which is precisely our situation (2025 train → 2026 score, adversarial AUC 0.807).

`[S-repo]` We already have adversarial validation implemented
(`src/prc/adversarial.py`) and it named **STAND_mvt (0.28 of the gain)** and
**ADES_mvt (0.18)** as the heaviest shift-carrying columns; we have **no LOFO and
no null-importance pass**.

---

## 7. Pseudo-labelling and semi-supervised use of the unlabelled 2026 rows

This is the section where the sourced evidence is weakest relative to how
attractive the idea sounds. I am flagging that explicitly.

**Sourced successes:**
* `[S]` NVIDIA Grandmasters Playbook, technique #6: in **BirdCLEF 2024**,
  pseudo-labelling "helped our model generalize better to new species and
  recording conditions". Also: "Pseudo-labels can also be used for
  **pretraining**. Fine-tune on the initial data as a last step to reduce noise
  introduced earlier." (An audio problem, not tabular.)
* `[S]` Chris Deotte's public notebook *Pseudo Labeling - QDA - [0.969]*
  (<https://www.kaggle.com/code/cdeotte/pseudo-labeling-qda-0-969>) from **Instant
  Gratification** — the title carries the score; I could not retrieve the body to
  read the before/after delta, so I will not quote one.
* `[S]` The technique is recorded as originating in **Santander Customer
  Transaction Prediction** with team Wizardry (per *The Kaggle Book*, ch. 9,
  <https://www.packtpub.com/en-us/product/the-kaggle-book-9781801817479/chapter/modeling-for-tabular-competitions-9/section/pseudo-labeling-ch09lvl1sec52>;
  page returned HTTP 403 to me, so this is a secondhand attribution).

**Sourced cautions:**
* `[S]` *The Kaggle Book* (same secondhand route): "Adding pseudo-labels won't
  change the result; **it may even worsen it if there is some added noise** in
  the pseudo-labeled data."
* `[S]` The mandatory procedural rule, attributed to Deotte: with k-fold you must
  compute **k separate sets of pseudo-labels**, so that a validation fold never
  sees labels produced by a model trained on itself. Otherwise the CV estimate is
  self-confirming.
* `[S]` Seedat et al., *You can't handle the (dirty) truth: Data-centric insights
  improve pseudo-labeling* (<https://arxiv.org/abs/2406.13733>): the standard
  assumption that the *labeled* data is "gold standard and 'perfect'" is wrong,
  and selecting pseudo-labeled samples by **learning dynamics** rather than by
  model confidence improves data efficiency and shrinks the differences between
  pseudo-labelers. (I read the abstract only; I did not retrieve the quantitative
  tables and so quote no numbers.)

**The label-free half of the idea is much safer than the pseudo-label half.**
`[X]` The 2026 rows have **features** even though they have no labels. Two
distinct uses, with very different risk:
1. **Label-free transductive features** — frequency/count encodings, category
   vocabularies, per-stand and per-operator volumes, and congestion-style
   aggregates computed over **train ∪ 2026 features**. No target is touched, so
   there is no target leakage; this is legitimate under any competition rule that
   permits using the provided test features. `[S]` Note the opposite convention is
   also documented: the IEEE-CIS Fraud notes describe "generating frequency
   features **without** test data information"
   (<https://dev.to/kamil_k7k/tabular-data-binary-classification-all-tips-and-tricks-from-5-kaggle-competitions-1aim>)
   — so this is a choice to be measured, not an automatic win.
2. **Target encodings fitted on pseudo-labeled 2026 rows** — this puts the
   model's own predictions into a feature it then consumes, and is the
   self-confirming mode the sources warn about. `[X]` With adversarial AUC 0.807
   `[S-repo]`, our model is by construction *least* accurate on exactly the 2026
   rows that differ most from 2025 — so pseudo-labels would be most confident
   where they are most wrong, which is the failure mode, not the success mode.

---

## 8. Count/frequency encoding, target encoding done properly, and the leakage
   traps — including one we appear to be standing in

### 8.1 The rule

`[S]` scikit-learn `TargetEncoder` documentation
(<https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html>):
* `fit_transform` "uses a **cross fitting** scheme to prevent target leakage and
  overfitting in downstream predictors. It is the **recommended** method for
  encoding training data."
* `fit(X, y).transform(X)` **≠** `fit_transform(X, y)`, precisely because the
  cross-fitting is only in `fit_transform`.
* Using `fit().transform()` on the training data is "discouraged… because it can
  introduce data leakage."
* Encodings are **shrunk toward the global mean**; `smooth="auto"` uses an
  empirical-Bayes estimate of the shrinkage.

`[S]` CatBoost paper §3.2 (above): the un-cross-fitted version is *Greedy TS* and
it causes a **conditional/prediction shift** — a systematic difference between
the feature's distribution on training rows and on unseen rows. This is the
formal statement of why "it looked great in training" and "it did nothing on the
leaderboard" coexist.

`[S]` Fold-wise fitting is the documented competition convention too: "target
encoding cross validation" is attributed to **Home Credit Default Risk** and
**Santander Customer Transaction Prediction**
(<https://dev.to/kamil_k7k/tabular-data-binary-classification-all-tips-and-tricks-from-5-kaggle-competitions-1aim>).

### 8.2 What our pipeline does

`[S-repo]` `src/prc/train.py:200-221` and `:367-370`. `fit_encodings(train_f, cfg)`
builds four label-dependent tables —
`reference_table` (p10 of the label by ADEP × STAND × RUNWAY, `min_count = 30`),
`airport_runway_hour_table`, `airport_median_table`, and
`operator_offset_table` (median of `label − proxy` by operator × ADEP) — and then
`attach_encodings(train_f, enc)` joins those same tables back onto **the very rows
they were fitted on**. The docstring says "Pure join: nothing is derived from df's
own labels", which is true of `attach_encodings` in isolation but not of the
`fit → attach` pair on the same frame. The comment at
`src/prc/train.py:185` — "fitted on the **training side of the fold** only" —
correctly rules out leakage *into the validation fold*, which is a different and
lesser hazard.

`[X]` This is textbook **Greedy TS**, i.e. exactly the construction the CatBoost
paper proves induces prediction shift and the one scikit-learn's docs call
"discouraged". `reference_min_count: 30` `[S-repo]` (`configs/main.yaml:16`)
bounds the worst case — a group of 30 rows contributes at most 1/30 of its own
p10 — but a p10 is an order statistic, so a single row **can** be the statistic it
then reads, and the effect is largest for exactly the rare stand/runway
combinations where the encoding is doing the most work. The direction of the bias
is that the model **over-trusts** `ref_p10` in training and finds it less
informative in 2026 — which is the shape of the CV/leaderboard divergence already
recorded in `configs/main.yaml:1-8` `[S-repo]`.

I want to be precise about what is and is not claimed: this is a **hypothesis
consistent with both the theory `[S]` and our observed CV/LB gap `[S-repo]`**. It
is not established that it is *the* cause, and the fix below is cheap enough that
the measurement settles it either way.

### 8.3 Count/frequency encoding

`[S]` Frequency encoding is attributed to **IEEE-CIS Fraud Detection**
(<https://dev.to/kamil_k7k/tabular-data-binary-classification-all-tips-and-tricks-from-5-kaggle-competitions-1aim>).
`[S]` CatBoost implements counter-type CTRs natively alongside target-based ones
(<https://arxiv.org/pdf/1706.09516>).
`[S]` NVIDIA's playbook technique #3 is the generalisation: "combining **pairs**
of eight categorical columns created 28 new categorical features that captured
interactions", credited with two first places.

`[S-repo]` We compute many event-count features (`src/prc/features.py:103-190`:
departures/arrivals in ±w-minute windows, per-runway counts at 15 and 60 min) —
these are **congestion counts**, not **category frequency encodings**. There is no
"how many rows in the whole dataset share this stand / this operator × airport /
this aircraft-type × runway" feature, and no pairwise categorical crossing. Both
are label-free, so both can legitimately be computed over train ∪ 2026 features.

---

## 9. Training on all the data vs bagging fold models for the final submission

The two sourced positions point in different directions and the difference is
about *what you are averaging*.

`[S]` **Full-data retrain**: NVIDIA Grandmasters Playbook technique #7 —
"retrain on 100 % of data after hyperparameter optimization", combined with
multi-seed ensembling (the 100-seed XGBoost result on Predicting Optimal
Fertilizers, MAP@3 ≈ 0.379 vs ≈ 0.376).

`[S]` **Bagged fold models**: AutoGluon-Tabular (Erickson et al., 2020,
<https://arxiv.org/pdf/2003.06505>) bags every model at every stack layer with
k-fold, which both produces OOF predictions for stacking *and* yields an
ensemble of k models for inference. Their ablation (Table 5, AutoML Benchmark,
39 datasets, 4 h; components removed **cumulatively**, which I flag because it
changes the reading):

| variant | avg. rank | avg. rescaled loss |
|---|---|---|
| AutoGluon (full) | 1.9324 | **0.1660** |
| NoRepeat (single round of k-fold bagging) | 2.1216 | 0.2199 |
| NoMultiStack (bagging + ensemble selection only) | 2.8514 | 0.5237 |
| NoBag (single train/val split) | 3.9054 | 0.7199 |
| NoNetwork (also drop the NN from base models) | 4.1892 | 0.8171 |

and Table S5: "**even after 4 h, the NoMultiStack variant cannot outperform the
full AutoGluon trained for only 1 h**".

`[X]` Reconciliation: the playbook's full-data retrain averages over **seeds** at
fixed data; AutoGluon's bagging averages over **data subsets** and additionally
buys OOF predictions. They are not mutually exclusive — the standard competition
shape is *bag k folds to get OOF for blending, then optionally add full-data
seed-averaged models to the blend*.

`[S-repo]` We do neither of the bagging halves. `src/prc/train.py:324-353`: a
single inner hold-out (`inner_holdout_months: 1`, `configs/main.yaml:17`) picks
the round count by early stopping, then `n_seeds` models are trained on the
**full** training frame at that fixed round count and averaged. So we have:
full-data retrain ✓ (playbook #7), seed averaging ✓ but at n=3, **k-fold bagging ✗,
and — critically — no out-of-fold prediction vector for any model**. The last
point is the binding constraint: **every blending and stacking technique in §1,
§3, §4 and §11 requires OOF predictions, and we cannot currently produce them.**

---

## 10. How strong competitors get off a plateau

`[S]` NVIDIA, *Kaggle Grandmasters Unveil Winning Strategies*
(<https://developer.nvidia.com/blog/kaggle-grandmasters-unveil-winning-strategies-for-data-science-superpowers/>).
Chris Deotte's stated loop: "results of implementing new ideas will promote more
**understanding**, which will suggest even more new ideas. We repeat this process
as fast as possible!" — i.e. the generator of hypotheses is *understanding gained
from the last experiment*, not a list of techniques. Other grandmasters in the
same piece: Bhatia on "simulating the public/private LB split using multiple CV
folds to estimate potential leaderboard shakeups", and on "creative feature
engineering or **target transformations**" as the top-rank lever; Austin on
spending "the first few days of a competition visualizing and understanding the
data".

`[S]` NVIDIA Grandmasters Playbook, technique #5: when direct prediction
plateaus, **change what you predict** — train a second-stage model on the
**residuals** of the first, or feed stage-1 predictions in as features. The
Deotte S5E4 writeup describes four complementary targets for the same problem:
**direct target, ratio, residual, and missing-feature imputation**
(<https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-stacking-using-cuml/>).
`[X]` This is the same family as our own 20.7 s win from re-parameterising the
target at one airport `[S, given]` — which is direct evidence that this lever
works *on our data*, and an argument for sweeping it rather than treating it as a
one-off.

`[S]` NVIDIA Grandmasters Playbook, technique #2: when stuck, run **diverse
baselines** (linear, GBDT, NN) *before* more feature work, because "baselines
guide experimentation direction and reveal which model families fit best" — and
the Rainfall example shows an un-feature-engineered diverse ensemble reaching 2nd
place.

`[S]` Deotte's S5E4: **500 experimental models**, of which 75 were kept. The
plateau-breaking asset is throughput of experiments, and the reason he could
afford 500 was GPU acceleration (RAPIDS cuML).

`[S]` Secondary but consistent (search-result summary of grandmaster advice,
lower confidence than the primary blogs above): "read discussion forums like
research papers"; "treat models as hypotheses to test and pivot quickly";
"analyze train-test distribution differences"; "optimize directly for the
competition metric".

---

## 11. How much does a single well-tuned LightGBM lose to a proper blend on a
    large tabular regression problem?

Five sourced answers, ordered from most to least like our problem.

**(a) PRC 2024, ATOW regression, RMSE, team_tiny_rainbow (3rd place)** `[S]`
(<https://journals.open.tudelft.nl/joas/article/download/7963/6296/31267>):

| comparison | RMSE | delta |
|---|---|---|
| their tuned LightGBM alone | 2,506.43 | — |
| their tuned CatBoost alone | 2,215.53 | **−11.6 %** vs their LightGBM |
| the full weighted blend of 5 models | 2,210.37 | **−0.23 %** vs CatBoost alone |

**The single most important line in this document**: on the closest analogue to
our problem, **switching model family bought 50× what blending bought.**

**(b) PRC 2024, the winner used one model** `[S]`
(<https://conormclaughlin.net/2024/12/reviewing-the-winning-ml-models-from-the-opensky-prc-data-challenge-what-did-they-do-differently/>):
team likable_jelly won with **a single LightGBM** (RMSE 1,561); 2nd (faithful
engine, 1,635) used AutoGluon; 3rd (tiny_rainbow, reported there as 2,218) used
the 5-model blend. So in PRC 2024 the single-model solution beat the blended one
by 42 % — because of features, not models. (Note: this source reports
tiny_rainbow at 2,218 while another reports 2,281 and the paper reports actual
2,217.75; the discrepancy does not affect the ordering.)

**(c) TabRepo, 200 OpenML datasets, normalized error** `[S]`
(<https://arxiv.org/pdf/2311.02971>, Table 1):

| step | normalized error | marginal gain |
|---|---|---|
| LightGBM (default) | 0.714 | — |
| LightGBM (tuned, 4 h random search) | 0.566 | −0.148 |
| LightGBM (tuned + ensemble of its own top-20 configs) | 0.534 | −0.032 |
| best single-family tuned+ensembled (CatBoost) | 0.524 | −0.010 |
| **cross-family Portfolio (ensemble)** | **0.365** | **−0.159** |
| AutoGluon | 0.389 | — |

Read directly: **cross-family blending bought 0.159 — slightly more than tuning
bought (0.148) and 5× what same-family config-ensembling bought (0.032).** The
paper's own conclusion: "No model is able to beat state-of-the-art AutoML systems
even with tuning and ensembling… all state-of-the-art tabular methods considered
multiple model families in order to reach good performance."

**(d) AutoGluon ablation** `[S]` (<https://arxiv.org/pdf/2003.06505>, Table 5,
components removed cumulatively): full 0.1660 rescaled loss → NoMultiStack 0.5237
→ NoBag 0.7199. Multi-layer stacking + bagging is the difference between 0.166
and 0.720 on 39 AutoML Benchmark datasets.

**(e) TabArena Elo** `[S]` (<https://arxiv.org/html/2506.16791v2>): post-hoc
ensembling over hyperparameter configurations moves LightGBM 1450 → 1532 and
RealMLP 1349 → 1569; without it, the leaderboard's top three (TabM, LightGBM,
RealMLP) would all rank below CatBoost.

**Synthesis** `[X]`: on *benchmark suites of small datasets*, a proper cross-family
blend beats the best single tuned GBDT by roughly as much as tuning beats a
default — a large fraction of total achievable headroom. On *this competition
family, at scale, with heavy feature engineering already done*, the one measured
instance (a) put the blend's marginal value at **0.23 %** while the model-family
choice was worth **11.6 %**. Our 284.36 s baseline × 0.23 % ≈ **0.65 s** from
blending; × 11.6 % ≈ **33 s** from a better-suited single model. I would not bet
those exact transfers, but the *ratio* is the actionable part, and it says: try
CatBoost as a **model**, not as a **blend member**, first.

---

## Applicable to resilient-hedgehog

Ordered by (sourced evidence of gain) / (implementation cost). Each item states
the measurement that would settle it.

**1. Cross-fit the four label-dependent encoding tables (highest ratio).**
*Evidence*: `fit_encodings(train_f)` → `attach_encodings(train_f, ·)` is Greedy TS
`[S-repo]` `src/prc/train.py:200-221, 367-370`; the CatBoost paper proves Greedy TS
induces prediction shift `[S]` <https://arxiv.org/pdf/1706.09516>; scikit-learn
calls `fit().transform()` on training data "discouraged… can introduce data
leakage" and ships cross-fitting as the default for a reason `[S]`. Our CV has
already been caught over-rewarding capacity by 34.8 s that the leaderboard
charged 13.2 s for `[S-repo]` `configs/main.yaml:1-8`, which is the signature.
*Cost*: low — compute each training row's `ref_p10` / `arh_med` / `apt_med` /
`op_offset_med` from an inner fold split (or leave-one-out, closed-form for means
and counts; for the p10 use K inner folds) while leaving the 2026-side join
unchanged.
*Measurement that settles it*: month-blocked CV RMSE before/after **plus** the
gap between training-fit RMSE and validation RMSE on the encoded features — the
prediction-shift prediction is that the train/valid gap shrinks even if CV RMSE
moves little, and that the leaderboard moves **more** than CV does. One
leaderboard submission is the decisive read.

**2. CatBoost as a rival single model on the identical features and folds.**
*Evidence*: on the closest analogue — PRC 2024 ATOW, RMSE — CatBoost beat the
same team's Optuna-tuned LightGBM by **11.6 %** (2,215.53 vs 2,506.43) and took
**0.865 of the blend weight** while LightGBM took ~5e-22 `[S]`
<https://journals.open.tudelft.nl/joas/article/download/7963/6296/31267>.
TabArena default Elo: CatBoost 1432 vs LightGBM 1206 `[S]`. TabRepo tuned:
CatBoost 0.534 vs LightGBM 0.566 `[S]`. Counter-evidence I am not hiding: on
Sigrist's 8 high-cardinality **regression** datasets CatBoost ranked *behind*
LightGBM-native-categorical (2.86 vs 2.71, rel. diff. 44.5 % vs 17.4 %) `[S]`
<https://arxiv.org/pdf/2307.02071>, and ML Contests 2025 records a winner for whom
CatBoost underperformed `[S]`. So this is a **measurement**, not a foregone
conclusion — but it is the highest-expected-value single measurement available.
*Cost*: moderate — `catboost` is already a declared dependency `[S, given]`;
needs a params block, native `cat_features`, and the same routed/re-parameterised
target handling. At 2.08M rows use `boosting_type='Plain'` (ordered boosting's
benefit is documented only below ~40k rows `[S]`).
*Measurement*: CatBoost-alone vs LightGBM-alone CV RMSE on the **same** month-blocked
folds, per routing branch and per airport. Decision rule: if CatBoost wins by more
than the paired-bootstrap interval on both the Jan and Jul folds, it replaces or
joins the primary; if it loses, stop — do not blend a loser hoping for diversity.

**3. Label-free frequency/count encodings over train ∪ 2026 features.**
*Evidence*: frequency encoding attributed to IEEE-CIS Fraud `[S]`; CatBoost ships
counter CTRs natively `[S]`; NVIDIA playbook #3 credits pairwise categorical
crossing with two first places `[S]`. We have congestion counts but **no category
frequency features and no categorical crossings** `[S-repo]`
`src/prc/features.py:103-190`. Because no target is touched, there is **no
leakage risk** in using the 2026 rows — the only decision is whether to include
them, and the IEEE-CIS convention was to exclude them `[S]`, so measure both.
*Cost*: very low — group-by counts; two variants (train-only, train+2026).
*Measurement*: CV RMSE with each variant, **and** the change in adversarial AUC
(`src/prc/adversarial.py`) — if the train+2026 variant closes part of the 0.807
gap, that is independent evidence it is transporting information across the year.

**4. K-fold bagging that emits out-of-fold predictions.**
*Evidence*: AutoGluon's NoBag ablation costs 0.1660 → 0.7199 rescaled loss on 39
datasets `[S]` <https://arxiv.org/pdf/2003.06505>; and **every** blending result
in §11 presupposes OOF vectors. We currently have a single 1-month inner hold-out
and no OOF `[S-repo]` `src/prc/train.py:324-353`, `configs/main.yaml:17`.
*Cost*: moderate — the month-blocked fold machinery already exists
(`src/prc/cv.py:26-41`); what is missing is persisting each fold's validation-row
predictions.
*Measurement*: (a) does the k-fold-bagged prediction beat the current
full-data + 3-seed prediction on the leaderboard? (b) once OOF exists, the
hill-climb weight over {LightGBM, CatBoost, ±XGBoost} is nearly free to evaluate.
Expected size, honestly: item (a) in §11 puts the blend's marginal value at
**0.23 % ≈ 0.65 s** on our scale — so this item's value is mostly *enabling* #2's
comparison and #5, not the blend gain itself.

**5. A residual / alternative-target sweep, generalising the one that already
worked.**
*Evidence*: our own target re-parameterisation at one airport bought **20.7 s**
`[S, given]` — the largest single documented win in our pipeline. The playbook's
technique #5 and Deotte's S5E4 both describe running **direct target, ratio,
residual and imputation targets in parallel** as standard practice `[S]`
<https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-stacking-using-cuml/>.
Bhatia names "creative feature engineering or **target transformations**" as the
top-rank lever `[S]`.
*Cost*: low-to-moderate — `sched_reparam_shift` already implements one shift and
is configured per-airport `[S-repo]` `src/prc/train.py:243`.
*Measurement*: per-airport CV RMSE for each target parameterisation, with the
airport-level selection made on one fold and confirmed on the other — because
picking 10 airports × k parameterisations on the same folds is exactly the
selection bias Cawley & Talbot quantify `[S]`.

**6. LOFO / null-importance feature pruning under the month-blocked folds.**
*Evidence*: LOFO "gives negative importance to features that hurt performance upon
inclusion" and caught seasonal OS-version features in Microsoft Malware under
time-split validation `[S]` <https://github.com/aerdem4/lofo-importance>; null
importance from IEEE-CIS `[S]`. Our adversarial run already names STAND_mvt (0.28
of gain) and ADES_mvt (0.18) as the shift-carrying columns `[S-repo]`
`src/prc/adversarial.py:24`. Counter-evidence: NVIDIA's playbook omits feature
selection entirely and argues the opposite `[S]`.
*Cost*: moderate — n_features full retrains; at 2.08M rows this is the expensive
item on the list.
*Measurement*: LOFO run on the **jul-forward** fold specifically (the one that
measures forward generalisation, `src/prc/cv.py:36-41`); any feature with
negative mean importance and a CI excluding zero is a drop candidate, confirmed
by a single leaderboard submission with those features removed.

**7. XGBoost as a third opinion — only after #2 and #4.**
*Evidence*: removing XGBoost from tiny_rainbow's blend cost **+0.16 %** `[S]`;
TabRepo shows MLP and XGBoost are *negatively correlated* across datasets `[S]`;
ML Contests 2025 ties XGBoost with LightGBM at 14 winner-uses `[S]`. But TabRepo
also ranks XGBoost tuned+ensembled (0.628) well behind LightGBM (0.534) `[S]`.
*Cost*: low once #4 exists (xgboost is already a declared dependency).
*Measurement*: marginal CV RMSE of the hill-climbed blend with and without it.
Do not ship it for weights below ~0.05.

**8. Tree-boosting with random effects (GPBoost) on the high-cardinality groups —
speculative but the strongest benchmark number in §2.**
*Evidence*: on 8 high-cardinality tabular **regression** datasets, GPBoost led at
**7.02 % avg. relative difference / rank 2.14**, vs LGBM Cat 17.4 % / 2.71 and
CatBoost 44.5 % / 2.86 `[S]` <https://arxiv.org/pdf/2307.02071>. The structure —
a random intercept per stand / per operator × airport, with the boosted trees
fitting the residual — matches our data's shape (many levels, unbalanced, some
levels with few rows). Caveats I am not hiding: those datasets are 28k–125k rows,
the paper's own author wrote GPBoost, and the library's wall-clock on the largest
(AUimport, 125k rows) was 45.9 s — scaling to 2.08M × ~4 grouping variables is
untested `[S]`.
*Cost*: high — new library, new fitting loop, unknown scaling.
*Measurement*: run it on **one airport** first (a few hundred thousand rows) with
a random intercept on STAND and on AIRCRAFT_OPERATOR, against the LightGBM
per-airport CV RMSE for the same rows. Only scale if it wins there.

**9. Large Optuna search — explicitly deprioritised.**
*Evidence against*: XGBoost's tunability is **0.014 AUC vs sensible defaults**
(0.043 only vs the package defaults) over 38 datasets `[S]`
<https://arxiv.org/abs/1802.09596>; no single tree-shape hyperparameter exceeds
0.008. Our params are already hand-set away from defaults (127 leaves,
min_data 50, 2000 rounds) `[S-repo]` `configs/main.yaml:18-20`. And Cawley &
Talbot's selection-bias warning `[S]` <https://www.jmlr.org/papers/v11/cawley10a.html>
bites hardest when the selection criterion is already known to disagree with the
target — which ours is, by 34.8 s `[S-repo]`.
*If run anyway*: only on folds **disjoint** from those used to compare models, and
report the winner's score on a fold the search never saw.

**10. Neural tabular models and tabular foundation models — do not pursue now.**
*Evidence against*: in this exact competition family, MLP and SAINT scored ~75 %
worse standalone and received blend weights of **1.24e-23 and 1.63e-23**, with
0.00 % ablation impact `[S]`
<https://journals.open.tudelft.nl/joas/article/download/7963/6296/31267>; NN
with embeddings was the **worst of seven methods** (189 % vs best) on
high-cardinality regression `[S]` <https://arxiv.org/pdf/2307.02071>; our target is
heavy-tailed and skewed, the GBDT-favouring condition in OmniTabBench `[S]`
<https://arxiv.org/html/2604.06814v1>. TabPFN-2.5's ceiling is **50,000 rows**
`[S]` <https://arxiv.org/abs/2511.08667>; we have 2.08M.
*The one exception worth a cheap look*: the DrivenData PREPARE pattern `[S]` (ML
Contests 2025) — fit TabPFN on a ≤50k stratified subsample and feed its
prediction in as a **feature** to the GBDT.
*Measurement*: CV RMSE with that single extra column. If it is not worth >1 s,
close the question.

**11. Pseudo-labelling the 2026 rows — lowest ratio, highest risk.**
*Evidence*: the only tabular endorsements I could source are secondhand (*The
Kaggle Book* on Santander/Wizardry, retrieved via 403) and come with the caveat
that it "may even worsen" results with noisy labels `[S]`; the sourced success is
an audio competition (BirdCLEF 2024) `[S]`. The procedural requirement is k
separate pseudo-label sets `[S]`. Against it specifically here: adversarial AUC
0.807 with only 5.9 % of training rows scoring above the ranking set's median
`[S-repo]` `src/prc/adversarial.py:20,25` means pseudo-labels would be most
confidently wrong exactly on the 2026 rows that differ most from 2025.
*Measurement if attempted*: k-fold pseudo-labels, single round, and the decision
made on a **leaderboard submission** — CV cannot honestly adjudicate a technique
whose whole mechanism is to make the model agree with itself.
