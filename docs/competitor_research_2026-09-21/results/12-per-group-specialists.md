# Slot 12 — per-group specialists

Investigator note: every factual claim below is marked **[S]** (sourced, with URL or local path) or
**[X]** (my inference / unverified). Nothing here was measured on our data by me; I ran no code and
changed no code.

Local primary sources cloned during this investigation (read-only):

- `/home/user/drivendataorg/nasa-airport-pushback` — NASA "Pushback to the Future" winners' code + reports
- `/home/user/chandanhegde07/openair` — team OpenAir's public repo (`status.md` is a full experiment log)
- `/home/user/sergiuv11/prc-data-challenge-2026` — team jubilant-vase's public repo

---

## 1. How specialists are actually built and combined

### 1.1 The single most relevant academic result: per-airport vs global, side by side

**Filienko, Lin, Robison, Tomlin, De Cock, "Predicting Time to Pushback of Flights in U.S. Airports",
KDD 2023 Undergraduate Consortium** (4th place, NASA pushback). PDF in the winners' repo at
`/home/user/drivendataorg/nasa-airport-pushback/4th Place/Phase 1/reports/2023__KDD2023_Undergraduate_Consortium (2).pdf`;
repo: https://github.com/drivendataorg/nasa-airport-pushback **[S]**

Setup: 10 U.S. airports, 3,836,894 flights, 24,008,438 training instances. Per-airport training
instance counts range from **1,236,371 (KMEM)** to **3,511,956 (KSEA)**; smallest by flights is KMEM
(151,768 gufis). They trained a global model over all airports combined and 10 local models, same
feature set (>30 features), LightGBM `num_leaves=4096`, `n_estimators=128`, `regression_l1`. **[S]**

Table 4, validation MAE in minutes **[S]**:

| airport | MLP global | MLP local | LGBM global | LGBM local |
|---|---:|---:|---:|---:|
| KATL | 8.0039 | 7.9851 | **7.8549** | 7.8682 |
| KCLT | 9.4824 | 9.5111 | **9.3135** | 9.3352 |
| KDEN | 12.0029 | 12.1098 | 11.9466 | **11.9446** |
| KDFW | 12.3230 | 12.2801 | **12.0823** | 12.1768 |
| KJFK | 11.5790 | 12.0592 | 11.1881 | **11.1143** |
| KMEM | 18.7407 | 18.8432 | **17.7555** | 17.8176 |
| KMIA | 12.4397 | 12.9948 | **11.9421** | 12.0881 |
| KORD | 11.3000 | 11.4879 | **11.1351** | 11.2458 |
| KPHX | 9.7497 | 9.6736 | **9.3849** | 9.4106 |
| KSEA | 7.5612 | 9.6736 | **7.2195** | 7.2300 |
| **All** | 10.9883 | 11.0500 | **10.7232** | 10.7686 |

Global wins at 8 of 10 airports and overall, by 0.045 min ≈ **2.7 seconds** **[S]**. The paper's own
conclusion, verbatim: *"training one local model for each airport is more memory efficient, while
yielding a mean absolute error at par with a global model trained over the data of all airports
combined."* **[S]**

The load-bearing sentence for us, verbatim **[S]**:

> "While building out our feature set, as described in Sec. 3, we observed that **for smaller subsets
> of features the 'local model' based approach would result in lower MAEs than the 'global model'
> based approach.** When training LightGBM models over the entire feature set from Tab. 3 however, the
> global model and the local model based approaches are at par."

They also note that the departure-airport feature carries near-zero importance in the global model,
and explain why: the local models cannot use it at all (it is constant within a group), and the global
model's other features already encode what the airport identity would tell it **[S]**.

**[X]** Read as a rule: *per-airport specialists buy you the airport-specific information your feature
set is missing. Once the features carry it, the specialists buy nothing.* This is the cleanest
published statement of when specialists help and when they stop helping, and it comes from the same
task family as ours.

### 1.2 NASA pushback — all five finalists, grouping key = departure airport (10 groups)

Per https://drivendata.co/blog/airport-pushback-finalists and the repo README **[S]**:

| rank | team | Phase-1 MAE | structure |
|---|---|---:|---|
| 1 | CDS | 10.673 | 3 CatBoosts **per airport** (V0, V1, V2) + 1 **global** CatBoost with airport as a categorical input |
| 2 | Moles | 10.728 | one CatBoost per airport, hyperparameters tuned per airport |
| 3 | Oracle 2 | 11.054 | one XGBoost per airport + a global over/under-estimation classifier |
| 4 | FLHuskies2 | 11.105 | one LightGBM per airport (tested a global model, kept local) |
| 5 | Cuong_Syr | 11.946 | one XGBoost per airport |

**1st place (CDS) — the exact combination rule**, read from
`1st Place/Phase 1/submission/utilities.py:636-677` **[S]**:

```python
for v in [0, 1, 2]:
    master[f"pred_{v}"] = model[v].predict(...)
    if v in [0, 2]:                                   # V0/V2 predict y - ETD
        master[f"pred_{v}"] = (master[f"pred_{v}"] + master["etd_time_till_est_dep"]).clip(1, 299)
master["global_pred"] = model["global_model"].predict(...)   # airport is a categorical feature

master["minutes_until_pushback"] = ((pred_0 + pred_1 + pred_2 + global_pred) / 4).clip(4, 260)

if airport.upper() in ["KPHX", "KMIA", "KJFK"]:
    master["minutes_until_pushback"] = pred_1.clip(4, 260)   # per-airport override
```

Four facts worth extracting **[S]**:

1. The blend is a **plain unweighted mean**: 75 % specialist, 25 % global. No fitted weight.
2. The global model is *in* the ensemble, not replaced by it — it is a 4th member carrying `airport`
   as a categorical feature. Their README calls it: *"Global model: Single model trained to predict
   minutes until pushback for all airports (with airport string as an input variable)"* **[S]**.
3. Two of the three specialists (V0, V2) predict **target minus a per-flight baseline** (`y − ETD`) and
   add the baseline back — i.e. reparameterisation, the same structural trick as our LIRF change.
4. **Three of the ten airports are hard-overridden** to a single specialist, dropping the blend
   entirely. The README says the ensembling architecture was *"designed ... [to minimize] MAE in a
   held-out validation sample"* **[S]**. So: a blend by default, per-group exceptions where validation
   said so.

**2nd place (Moles)**, from their write-up PDF, verbatim **[S]**: *"we felt that a global model was not
ideal for the small number of airports we were given to work with."* They trained one CatBoost per
airport with per-airport hyperparameter optimisation and no global model at all.

**3rd place (Oracle 2)**, verbatim from their write-up **[S]**: separate XGBoost per airport, trained
after removing outliers with a **per-airport** error threshold (*"absolute error greater than 20 or 30,
depending on the airport"*), then a single global **"estimation classifier"** (AUC 0.64) predicting
whether the main model over- or under-estimates; *"Based on the classification, we either subtract or
add the median of all over/underestimated values, which were determined and saved during the training
of the 'estimation classifier.'"* That is a **residual bias correction stage bolted onto per-group
specialists** — relevant to §3(e).

Compute cost, for calibration **[S]**: 1st place reports *"Train duration: 5h for each airport and
model"*; 5th place *"1 to 2 hours per airport, 15 hours in total for 10 airports"*.

### 1.3 M5 (Walmart, 42,840 series, 30M rows) — grouping by store, 10 groups

From Makridakis, Spiliotis, Assimakopoulos, *The M5 Accuracy competition: Results, findings and
conclusions* (IJF 2022), PDF: https://statmodeling.stat.columbia.edu/wp-content/uploads/2021/10/M5_accuracy_competition.pdf **[S]**

- **1st (YJ_STU)**, verbatim: *"an equal weighted combination (arithmetic mean) of various LightGBM
  models that were trained to produce forecasts for the product-store series using **data per store
  (10 models), store-category (30 models), and store-department (70 models)**. Two variations were
  considered for each type of model... a total of 220 models were built and **each series was forecast
  using the average of 6 models**, each one exploiting a different learning approach and train set."*
- **2nd (Anderer)**: LightGBM *"trained per store (10 models)"*, then five multipliers from an N-BEATS
  model at higher aggregation levels adjust the forecasts.
- **4th (monsaraida)**: *"trained per store (10 models) ... each week of the forecasting horizon was
  forecast separately using a different model (4 models per store)"* → 40 models.
- **5th (Lahoud)**: *"trained per department (7 models)"*, then *"externally adjusted so that the mean
  of each of the series at the store-department level was the same as the one of the previous 28
  days ... using appropriate multipliers."*
- And on the rest: *"almost all of them adopted similar approaches to the winning submission, training
  recursive and non-recursive LightGBM models **per store, department, or store-department**."*

**[X]** The M5 recipe is not "specialist vs global". It is **pool the data at several different
granularities and average the resulting models with equal weights**. The group count that dominated
(10 stores) is exactly our airport count, and per-group row counts (~3M) are the same order as ours.
Nobody in the M5 top five fitted a blend weight; the winner used a plain arithmetic mean.

### 1.4 Kaggle forecasting competitions generally

Bojer & Meldgaard, *Learnings from Kaggle's Forecasting Competitions*, https://arxiv.org/pdf/2009.07701
**[S]**. Abstract, verbatim: *"we find that most of the Kaggle datasets are characterized by higher
intermittence and entropy than the M-competitions and that **global ensemble models tend to outperform
local single models**."* On Rossmann (1,115 stores), verbatim: *"Most of the top performers used
ensembles of global XGBoost models to create forecasts, but **a few of them did include local XGBoost
models as part of their ensemble**."* Ensembling multiple XGBoosts trained on **different data subsets**
bought ~5 % over the best single model **[S]**.

### 1.5 A controlled experiment in subgroup specialisation

Globus-Harris, Harrison, Kearns, Perona, Roth, *Diversified Ensembling: An Experiment in Crowdsourced
Machine Learning*, https://arxiv.org/abs/2402.10795 **[S]**. 46 teams, ACS income prediction, teams
submit (group `g`, model `h`) pairs which are ensembled into a global model by a platform that
guarantees error is non-increasing on every group named so far.

Results, verbatim **[S]**: *"the best team had an overall squared error of 17012.32 on the holdout test
data, while the global model had squared error of 17010.49 – a slight improvement but an improvement
nonetheless."* And: *"Specializing models to subgroups helped ... treating the competition as purely a
Kaggle-style exercise was a relatively competitive strategy, but was not the most competitive
strategy among our participants."*

**[X]** Quantitatively that is a **0.011 % squared-error improvement** from the whole subgroup-
specialisation machinery over the single best all-data model. It is the most honest published
magnitude estimate I found for "what specialisation buys you on a homogeneous tabular task".

### 1.6 In *this* competition

**OpenAir** (`/home/user/chandanhegde07/openair/status.md`, experiment E20) **[S]**. Their expert set,
all fitted on the same residual target, scored on a Jan+Jul holdout:

| expert | Jan+Jul overall | matched | Dec | Dec matched |
|---|---:|---:|---:|---:|
| A LGB residual | 372.36 | 250.98 | 238.01 | 223.95 |
| B LGB direct TAXITIME | 394.63 | 283.08 | 237.51 | 223.62 |
| C CatBoost residual | 370.37 | 248.25 | 232.37 | 218.56 |
| D XGBoost residual (numeric only) | 385.50 | 266.24 | 246.17 | 232.70 |
| **E airport-specific residual LGB** | **370.52** | **247.76** | **229.94** | **215.90** |

Blends, weights fitted on the Jan+Jul holdout with `scipy.optimize.nnls` (then normalised to sum 1) or
positive Ridge **[S]**:

| blend | Jan+Jul | Δ vs A | matched | Dec | Δ Dec |
|---|---:|---:|---:|---:|---:|
| equal5 | 371.99 | −0.38 | 250.86 | 232.00 | −6.01 |
| A+E only | 369.99 | −2.37 | 247.49 | 233.50 | −4.51 |
| **nnls** | **368.03** | **−4.33** | **244.76** | **228.45** | **−9.56** |
| ridge1 / ridge10 | 368.17 | −4.19 | 244.93 | 229.60 | −8.41 |

Verbatim **[S]**: *"**NNLS weights: A=0, B=0, C=0.456, D=0.053, E=0.491** — the optimizer drops the
LightGBM E18-H entirely in favour of CatBoost (C) + airport experts (E); ridge gives the same picture.
Improvement is not a Jan+Jul artifact: December improves by −9.56 s (weights were fit only on
Jan+Jul)."* And on why it works, verbatim **[S]**: *"expert residuals are positively but not perfectly
correlated (A–C ~0.97, A–E ~0.96, C–E ~0.98) — the gain comes from **C and E each being individually
stronger and slightly different**, not from low-correlation averaging of equals."*

Construction of expert E, from `experiments/run_e20_ensemble.py:258-278` **[S]**:

```python
for a in AIRPORTS:
    tr_a = tr_res.filter(pl.col("airport") == a)
    if tr_a.height < 20000 or va_a.height == 0:
        predE[va_idx] = predA[va_idx]          # fall back to the GLOBAL model
        continue
    ma = fit_lgb(...)                          # cold start, own early-stopping split, same features
    predE[va_idx] = apply_override(p_local + resid, ...)
```

So: **cold-started** (not warm-started from the global model), same feature matrix, own per-airport
early-stopping split, and a hard-coded **20,000-row minimum** below which the group falls back to the
global model. Every expert predicts a **residual over a per-airport OLS baseline `P_cal`**
(`per_airport_ols(mvt_aobt, aobt_eobt, geo_mean)`) — so the per-airport structure is *also* present in
the baseline, not only in the expert **[S]**.

Later OpenAir experiments are the cautionary half **[S]**:

- E50: *"NNLS weights unstable across splits — not shipped."* They shipped fixed λ = 0.5 and λ = 0.25
  gates instead of a fitted weight.
- E23 (stacking METAR into the experts): *"Airport experts absorb winter weather (EDDM/LFPG/EHAM/LSZH).
  Jan+Jul ensemble is flat-to-worse and the >30 tail hurts. Do not ship."* — per-airport experts can
  swallow a feature family that would otherwise pay.
- E25 (a full per-airport rebuild with a floor + log-excess target): rejected; *"Per-airport experts
  already exist as E20-E."* Their note that the schema permits full per-airport models cites the
  smallest training airport as **n = 112,561** for their Jan+Jul split **[S]**.

**jubilant-vase / sergiuv11** (`/home/user/sergiuv11/prc-data-challenge-2026`) **[S]**. This is the
cleanest end-to-end example of the pattern, with an official leaderboard delta:

| version | architecture | official RMSE |
|---|---|---:|
| V2 | Global LightGBM + narrow LIRF overlay | 290.488 s |
| V3 | V2 + EDDF, EGLL, EHAM specialists | **286.656 s** |

i.e. **−3.832 s official** from per-airport specialists at 3 of 10 airports **[S]**
(`docs/RESULTS.md`). Construction, verbatim from `docs/METHODOLOGY.md` **[S]**: *"Every model uses L2
regression, learning rate 0.06, 127 leaves, minimum 200 rows per leaf ... and 436 trees. At EDDF, EGLL
and EHAM, **the prediction is an equal blend of the global and local model**. Other airports use the
global prediction unchanged."* Specialists are **cold-started on that airport's rows only**, with the
**identical feature set and identical hyperparameters** as the global model
(`scripts/experiment_airport_specialists.py`, `BLEND_WEIGHT = 0.5`) **[S]**.

Their three predeclared gates, global-system → V3 **[S]**:

| gate | global | V3 | gain | bootstrap wins |
|---|---:|---:|---:|---:|
| composition | 325.931 | 323.555 | 2.375 | 100.0 % |
| July forward | 250.341 | 241.598 | 8.743 | 100.0 % |
| seasonal | 403.286 | 401.471 | 1.815 | 100.0 % |

The same repo lists *"more airport specialists"* among its **negative results** — extending beyond the
three did not pass **[S]** (`docs/RESULTS.md:48`). The repo also ships a second grouping key as an
experiment: `experiment_wake_specialists.py`, wake-category specialists, also `BLEND_WEIGHT = 0.5`,
also rejected **[S]**.

**Phoenix-Ops** (recorded in our own `docs/competitor_research_2026-09-21/results/07-repo-phoenix-ops.md`,
marked [S] there against their repo) **[S to that file]**:

- Specialist scope is **not** an airport: `ADEP_mvt == "LIRF" & missing_IOBT_flt == 1`, 1,488 training
  rows, 383 ranking rows. The specialist **replaces** the global residual in-place on that scope (not
  a blend) in v3; v7 is the one exception that blends `0.5·v6 + 0.5·arrival_specialist` inside the scope.
- Five-seed equal-weight averaging of that 1,488-row specialist: **−1.6945 s official on 383 rows**.
- Their **rejected** list is the most directly relevant counter-evidence we have:
  - *"Airport-specific blend weights | 323.712 vs 323.377 | rejected"* — per-airport blend weights
    were **worse** than one shared weight.
  - *"Non-LIRF unmatched specialist | +0.012 s overall, worse at most airports | rejected"*.
  - *"Matched-LIRF LightGBM, residual fit | 395.047 vs 393.384 on that scope | rejected"* — even at
    LIRF, a specialist on the *matched* rows lost.
  - *"3 mixture-of-baselines classifiers for unmatched LIRF | worsened both months | rejected"*.

---

## 2. When specialists help and when they hurt

### 2.1 Minimum group size reported as workable

| source | group | rows in group | outcome |
|---|---|---:|---|
| OpenAir `run_e20_ensemble.py:265` **[S]** | airport | **< 20,000 → refuse**, fall back to global | hard-coded guard |
| OpenAir E25 note **[S]** | airport | smallest split airport 112,561 | "schema allows full per-airport models" |
| jubilant-vase **[S]** | airport | ~130k–270k (all 10 airports of the 2025 training year) | 3 of 10 passed the gate |
| Phoenix-Ops **[S to 07-…md]** | LIRF ∧ missing IOBT | **1,488** | works, **only** with 5-seed averaging |
| NASA KDD paper **[S]** | airport | 1.2M–3.5M instances | ties the global model |
| M5 top-5 **[S]** | store | ~3M | dominates |

**[X]** Synthesis: nobody publishes a clean threshold. What the sources actually show is that the
threshold is set by **variance of the fitted specialist**, not by a row count: at ~10³ rows a
specialist is usable but needs seed-averaging to be stable (Phoenix-Ops); at 2×10⁴ a practitioner in
this very competition chose to refuse it outright (OpenAir); at 10⁵–10⁶ it is a normal model. The
group-count-to-row-count ratio matters more than either alone — see §2.3.

The theory says the same thing in a different language. Mansour, Mohri, Ro, Suresh, *Three Approaches
for Personalization with Applications to Federated Learning*, https://arxiv.org/abs/2002.10619,
Corollary 6, verbatim **[S]**: *"for models to generalize well, it is desirable to have m ≫ d_c and
**the average number of samples to be much greater than d_l, i.e., m/p ≫ d_l**. ... this bound only
depends on the **average** number of samples and not the minimum number of samples."* (m = total rows,
p = number of groups, d_c / d_l = pseudo-dimension of the central / local hypothesis class.)

**[X]** For us: m/p = 2.08M / 10 ≈ 208k rows per airport. Unless a per-airport LightGBM is enormously
more complex than the global one, this bound is comfortable — the sample size is not the binding
constraint at airport granularity. It *is* the binding constraint at the granularity Phoenix-Ops used
(LIRF ∧ unmatched, 10³).

### 2.2 The reported relationship between group count and gain

- Mansour et al., synthetic experiment, verbatim **[S]**: *"Since the clients are naturally divided
  into four clusters, **as we increase q, the test loss steadily decreases till the number of clusters
  reaches 4 and then remains constant**."* Also verbatim: *"**HYPCLUSTER performs the best when the
  number of samples per user m_k is very small. If m_k is large, MAPPER [model interpolation] performs
  the best**, followed closely by FINETUNE."* So: group *count* pays only up to the number of genuinely
  distinct regimes; past that it is free, not harmful; and the *mechanism* that wins flips from
  clustering to model interpolation as group size grows.
- jubilant-vase, empirically in our competition **[S]**: 10 candidate groups, **3 passed**, and adding
  more failed. Gains concentrate in a minority of groups.
- Phoenix-Ops **[S to 07-…md]**: the *only* group that paid was the one defined by a **data-generating
  artefact** (LIRF fill of BLOCK_TIME), not by airport identity per se. `Non-LIRF unmatched
  specialist` was +0.012 s (worse) and *"worse at most airports"*.
- M5 **[S]**: the winner did not choose a group count — he averaged three different granularities
  (10 / 30 / 70) rather than picking one.

**[X]** The pattern across all four: **the number of groups that pay is much smaller than the number of
groups available, and you cannot tell which ones in advance.** The design that survives is "fit all
of them, gate each one on held-out data, keep the survivors" (jubilant-vase's
`select_airport_membership.py`) or "average across granularities and don't choose" (M5).

### 2.3 The theoretical reason global models usually tie or win

Montero-Manso & Hyndman, *Principles and Algorithms for Forecasting Groups of Time Series: Locality and
Globality*, IJF 37(4):1632-1653, https://arxiv.org/abs/2008.00444 **[S]**:

- Verbatim: *"global methods are not more restrictive than local methods ... Global and local methods
  can produce the same forecasts **without any assumptions about similarity of the series** in the
  set."*
- Verbatim: *"the **complexity of the local approach grows with the size of the set, while the
  complexity of the global approach remains constant**. ... a global approach can be at least as
  complex as the sum of all individual [local models] in the set. This leads us to **prescribe the use
  of comparatively more complex hypothesis classes for global algorithms**."* Formally
  `∏ᵢ|Hᵢ| = |J|`: the global hypothesis class can be as large as the *product* of the local ones and
  still match their generalisation guarantee.

**[X]** Direct operational reading for us: for K = 10 airports, the honest competitor to a set of 10
per-airport LightGBMs is **not our current global model — it is a substantially larger global model**
(more leaves, more trees, more interaction depth, richer airport-conditioned features). If a per-airport
specialist beats our global model, that is at least as likely to be evidence that our global model is
under-parameterised as evidence that airports need separate models. Testing the bigger global model is
cheaper than shipping ten specialists and settles which of the two it is.

---

## 3. Alternatives to literal specialists

### (a) Global model with the group as a categorical feature

This is what we do: `ADEP_mvt` is in `CATEGORICALS` (`src/prc/features.py:33-44`) **[S, local code]**.

- NASA 1st place used exactly this as *one of four* ensemble members and averaged it with three
  per-airport models **[S]** — i.e. they did not treat it as a substitute, they treated it as a
  diversity source.
- Montero-Manso & Hyndman §2.3 gives the theory for why it can match per-group models **[S]**.
- The KDD paper found the `airport` feature had **near-zero gain importance** in their global model,
  and explains it: the other features already encode what the airport identity would say **[S]**. That
  is a diagnostic we can run: if `ADEP_mvt` carries low importance in our model *and* per-airport
  residual means are near zero, per-airport specialists have nothing to add.
- OpenAir's status.md records the opposite diagnostic result on our data, verbatim **[S]**:
  *"airport residual means (LTFM −236, LIRF +214) suggest adding a per-airport constant to the proxy."*
  Those are **large** per-airport biases in a *baseline*, which is a positive signal that group
  structure is real in this task.

### (b) Target encoding of the group

The standard smoothed form, from the CatBoost paper (Prokhorenkova et al., NeurIPS 2018,
https://arxiv.org/abs/1706.09516, eq. 4) citing Micci-Barreca (SIGKDD Explorations 3(1):27-32, 2001),
verbatim **[S]**:

> x̂ⁱ_k = ( Σⱼ 1{xⁱⱼ = xⁱ_k}·yⱼ + a·p ) / ( Σⱼ 1{xⁱⱼ = xⁱ_k} + a ) , a > 0 ... *"A common setting for
> p is the average target value in the dataset."* *"The problem of such greedy approach is **target
> leakage**: feature x̂ⁱ_k is computed using y_k, the target of x_k."*

**[X]** `a` is literally the "how many rows before I trust this group's own mean" knob: a group with
n ≫ a gets its own statistic, n ≪ a gets the global prior. It is the continuous version of the
specialist/global decision, done per group automatically, with no extra models.

Competition evidence is mixed **[S]**: jubilant-vase lists *target encoding* among its rejected
branches (`docs/RESULTS.md:47`); OpenAir's E25 ablation shows encodings mattering *inside* an
architecture that was itself rejected (`no_enc 484.32` vs `473.61` combined). We already have 9
encoding columns joined inside `predict_fold` (per `docs/handoff.md` §2 correction) **[S, local]**.

### (c) Per-group target normalisation (subtract group mean/median, add back)

This is the best-evidenced alternative in the aviation sources, and it is **structurally identical to
our LIRF reparameterisation (−20.7 s, [S] per the brief)**.

- **NASA 1st place**: models V0 and V2 predict `minutes_until_pushback − ETD` and add
  `etd_time_till_est_dep` back at inference. Two of the four ensemble members are reparameterised
  **[S]** (`utilities.py:641-647`).
- **PRC Data Challenge 2024 winner (team likable-jelly)**: modelled the derived target
  `(TOW − EOW) / (MTOW − EOW)` — a payload ratio — rather than raw take-off weight. Per aircraft type,
  EOW and MTOW are constants, so this *is* a per-aircraft-type affine normalisation of the target
  **[S]**, https://conormclaughlin.net/2024/12/reviewing-the-winning-ml-models-from-the-opensky-prc-data-challenge-what-did-they-do-differently/.
  **[X]** That team won the previous edition of *this* competition with a per-group target
  reparameterisation and, per that review, **no per-group specialists at all**.
- **M5 5th place**: *"externally adjusted so that the mean of each of the series at the store-department
  level was the same as the one of the previous 28 days ... using appropriate multipliers"* **[S]**.
- **M5 2nd place**: five multipliers from a higher-aggregation model applied to per-store LightGBM
  forecasts **[S]**.
- **Phoenix-Ops** swap the baseline itself on their specialist scope, to
  `clip(MVT_TIME − SCHED_TIME, 0, 172800)` **[S to 07-…md]** — a per-group change of offset, not a
  per-group model.
- **OpenAir** run every expert as a residual over `P_cal = per-airport OLS(mvt_aobt, aobt_eobt,
  geo_mean)` **[S]** — the per-airport structure lives in the *baseline*, and the trees only fit what
  is left.

**[X]** The common shape: **put the group effect in the offset, keep one model for the shape.** It is
strictly cheaper than 10 specialists, has no small-sample variance problem, and — per OpenAir's
`P_cal` — is already how the strongest competitor in this competition handles per-airport structure.

### (d) Per-group monotonic calibration of the global model's output

**Weakly evidenced.** I found no competition write-up in this task family that applies per-group
isotonic/monotonic recalibration to a regression output and reports a gain. The nearest [S] sources
are the fairness/calibration literature (group-wise ECE, per-group isotonic calibrators fitted on
held-out calibration sets), e.g. Luo et al., *Local Calibration: Metrics and Recalibration*,
https://proceedings.mlr.press/v180/luo22a/luo22a.pdf **[S]** — but these target *calibration error*,
not squared error, and the two are not the same objective.

The competition analogue that *does* exist is **per-group affine (not monotone-nonparametric)
overlays**: jubilant-vase fit *"a linear relationship ... from 2025 LIRF rows satisfying the same rule"*
and it *"changed 15 of 215,876 ranked rows"* **[S]**; Phoenix-Ops' baseline swap is the same idea **[S
to 07-…md]**.

**[X]** My read: per-group *monotone* calibration is a poor fit for an RMSE target whose conditional
mean is already what the model estimates; the affine version collapses to (e).

### (e) Per-group bias correction on out-of-fold residuals

- **NASA 3rd place** did exactly this (a global classifier for over/under-estimation, then add or
  subtract the stored median of over/under-estimated residuals) and finished 3rd **[S]**.
- **M5 2nd and 5th** are multiplier versions of the same idea **[S]**.
- **OpenAir** measured the per-airport residual means on our data: LTFM −236 s, LIRF +214 s **[S]**.

**[X]** For squared loss the arithmetic is exact and worth stating plainly: if the OOF residual mean
within group g is b_g and group g has n_g rows, subtracting b_g removes exactly n_g·b_g² from the SSE.
Whether this survives is entirely a question of whether b_g estimated on training folds transfers to
the ranking set — that is a two-line experiment, not a modelling project. Caveat **[X]**: a global
GBDT with the group as a feature should already have driven b_g to ≈0 *in-sample*; a large non-zero
out-of-fold b_g is therefore evidence of **temporal drift**, and the right fix may be a drift fix
rather than a constant.

### (f) Sample weighting toward the group

Mansour et al. call this **data interpolation** and give it a bound. Verbatim **[S]**: *"We propose to
minimize the loss on the concatenated data, **λ·D_k + (1−λ)·C**, where λ is a hyper-parameter and can
be obtained by either cross validation or by using the generalization bounds of Blitzer et al.
(2008)."* Their EMNIST table shows DAPPER (data interpolation) and MAPPER (model interpolation) within
0.1 % of each other and of simple fine-tuning **[S]**.

**[X]** Practical advantage over a literal specialist: one model per group but every model sees all
2.08M rows, so there is no small-sample regime; λ = 1 recovers the specialist, λ = 0 recovers the
global model. In LightGBM this is one `weight` column, no architectural change. I found **no**
competition write-up in this family that reports using it — it is the least-explored branch of the six.

---

## 4. When does blending beat both, and how is the weight chosen?

### 4.1 The condition

**[X]** (standard two-forecast algebra; I did not find a citable source for it in this investigation).
For two predictors with error variances σ₁², σ₂² and error correlation ρ, the MSE-optimal convex weight
on predictor 1 is

  w\* = (σ₂² − ρσ₁σ₂) / (σ₁² + σ₂² − 2ρσ₁σ₂),

and the blend strictly beats **both** members iff ρ < min(σ₁/σ₂, σ₂/σ₁). Two consequences:

- If the specialist is much worse than the global model (σ₁/σ₂ large), it still earns positive weight
  provided ρ is below that ratio — which is why OpenAir's XGBoost-numeric expert D (385.50, clearly the
  second-worst) still received weight 0.053 **[S]**.
- If the two are nearly equally good and nearly perfectly correlated, w\* → 0.5 and the gain → 0. This
  is the regime the KDD pushback paper sits in (global 10.7232 vs local 10.7686) **[S]** — and it
  predicts that a 50/50 global+specialist blend there would have bought roughly nothing.

The sourced empirical check: OpenAir's expert residual correlations are **0.96–0.98** **[S]**, i.e.
extremely high, and NNLS still put 0.491 on the airport experts and got −4.33 s. Their own explanation,
verbatim **[S]**: *"the gain comes from C and E each being individually stronger and slightly
different, not from low-correlation averaging of equals."*

### 4.2 How practitioners actually pick the weight

Ordered from "most evidence that it is safe" to "most evidence that it overfits":

1. **Fixed, un-tuned weight.** NASA 1st place: unweighted mean of 4 **[S]**. M5 1st place: *"equal
   weighted combination (arithmetic mean)"* of 6 models per series **[S]**. jubilant-vase:
   `BLEND_WEIGHT = 0.5`, declared in advance, never tuned **[S]**. Phoenix-Ops v7: 0.5/0.5 inside the
   specialist scope **[S to 07-…md]**. OpenAir, after NNLS proved unstable: fixed λ = 0.5 and 0.25
   gates **[S]**.
2. **Constrained regression on genuinely held-out predictions.** OpenAir: `scipy.optimize.nnls` and
   `Ridge(positive=True)` on a holdout, then **checked on a second, later period they did not fit on**
   (December improved −9.56 s with Jan+Jul-fitted weights) **[S]**. Their script header, verbatim:
   *"a genuine held-out split, so overfit risk is small; Dec is the honest transfer check."* **[S]**
3. **Cross-validated convex combination (Super Learner).** van der Laan, Polley & Hubbard,
   *Super Learner*, Statistical Applications in Genetics and Molecular Biology 6(1), 2007,
   https://biostats.bepress.com/ucbbiostat/paper222/ **[S]**: weights are chosen to minimise
   **cross-validated** risk, restricted to be non-negative and to sum to one, and the oracle inequality
   for the cross-validation selector extends to the resulting ensemble, making it asymptotically
   equivalent to the best combination in the library.
4. **Greedy hill-climbing with anti-overfit devices.** Caruana, Niculescu-Mizil, Crew, Ksikes,
   *Ensemble Selection from Libraries of Models*, ICML 2004,
   https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf **[S]**: greedy forward
   selection **with replacement** (so weights are integer multiples of 1/N), plus **bagged** ensemble
   selection over random subsets of the library, specifically because unconstrained stacking regression
   *"overfits dramatically"* when there are many highly correlated inputs and a small validation set.
5. **Per-group tuned weights.** The one measurement I have is negative: Phoenix-Ops,
   *"Airport-specific blend weights | 323.712 vs 323.377 | rejected"* **[S to 07-…md]** — per-airport
   weights were **worse** than one shared weight. NASA 1st place's per-airport decision was not a fitted
   weight but a **hard 0/1 switch** at 3 of 10 airports **[S]**.
6. **Gating model.** The only worked example I found in this task family is NASA 3rd place's
   over/under-estimation classifier (AUC 0.64, applied as a ± median offset) **[S]**. Phoenix-Ops
   tried *"3 mixture-of-baselines classifiers for unmatched LIRF"* and they *"worsened both months"*
   **[S to 07-…md]**.

**The selection rule I would actually copy**, from jubilant-vase's
`scripts/select_airport_membership.py` **[S]** — it is the only published procedure I found that gates
each group independently and rejects most of them:

- Fit specialists at **all ten** airports; blend 50/50 everywhere.
- Evaluate on a **held-out month** (October; the fit pool is months ≤ 8).
- Keep a specialist for airport *a* only if **all three** hold:
  - per-airport RMSE improvement ≥ **1.0 s** (`--minimum-improvement`),
  - paired-bootstrap win rate ≥ **0.80** over 2,000 resamples (`--minimum-win-rate`),
  - the single largest row contributes ≤ **50 %** of the net SSE advantage (`--maximum-row-share`).
- Result on their data: **3 of 10** airports selected; official LB −3.832 s.

**[X]** That third criterion is the one we most need: our OOF SSE is dominated by a handful of rows
(top 10 rows = 24.1 % of SSE per `docs/handoff.md` §3), so a per-airport gate without a
single-row-share cap will select on noise.

---

## 5. Published per-airport vs global analysis in air-traffic prediction

The only side-by-side I found is §1.1 (Filienko et al., KDD 2023): 10 airports, same features, global
LightGBM **10.7232** vs per-airport LightGBM **10.7686** MAE, global better at 8/10 airports; global
MLP 10.9883 vs local MLP 11.0500 **[S]**. Their stated conclusion is "at par", and the efficiency
argument (memory, parallelism) is what decides it for them, not accuracy **[S]**.

Supporting, but weaker **[S]**: all five NASA finalists nonetheless *shipped* per-airport models
(§1.2), and 2nd place's stated reason was the small number of airports, not a measurement. 4th place's
own paper says they tested a global model and found it equal.

Not found **[S — searched, did not find]**: any peer-reviewed taxi-out paper that reports per-airport
vs multi-airport-global side by side. Candidates that exist are single-airport studies (Shenzhen
Bao'an; Beijing Capital with a Zhengzhou generalisability check; Seattle/Tacoma; Casablanca), and both
the Scientific Reports 2026 stacking paper and the LSTM-XGBoost chapter are paywalled to this session.
The previous edition of this competition (PRC 2024, ATOW) produced a JOAS special issue
(https://journals.open.tudelft.nl/joas/issue/view/1048) with only one full research article, and the
reviewed summary of its top three teams reports **no per-group specialists** — the winner's
distinguishing move was a per-aircraft-type target reparameterisation **[S]**.

---

## Applicable to resilient-hedgehog

Context I used (all local, read-only): 10 airports, per-airport training rows **134,907 (LSZH) –
272,965 (LTFM)**, per-airport target sd **301.8 (EDDM) – 1,332 (LIRF)** (`docs/data_profile.md` §3)
**[S]**; naive `airport mean` RMSE 514.8 vs `global mean` 546.4 **[S]**; one global model routed A/B by
NM match, model A 2,062,577 rows / model B 22,470 rows **[S, brief]**; `ADEP_mvt` already a categorical
feature **[S, `src/prc/features.py:33`]**.

Ordered by the strength of the sourced evidence behind each. Each entry names the measurement that
settles it.

**1. Per-airport 50/50 specialists on model A, gated per airport.**
Evidence: strongest of any item here. jubilant-vase shipped this and moved the **official** score
290.488 → 286.656 (−3.832 s) in *this* competition, on *this* data, with 3 of 10 airports selected
**[S]**; OpenAir's NNLS assigns the airport expert **0.491**, the largest weight of five experts, with
a −9.56 s December transfer check **[S]**; all five NASA pushback finalists shipped per-airport models
**[S]**.
Build it exactly as they did: same feature matrix, same hyperparameters as model A, cold start on that
airport's rows, 50/50 with the global prediction, **no tuned weight**.
*Measurement that settles it:* run the jubilant-vase gate verbatim — fit specialists at all 10
airports, evaluate per airport on a held-out period, and keep airport *a* only if (i) per-airport RMSE
improves ≥ 1.0 s, (ii) ≥ 80 % of 2,000 paired bootstrap resamples win, (iii) no single row supplies
> 50 % of the net SSE advantage. Report the count of airports passing. If 0–1 pass, stop here; if ≥ 3
pass, ship the survivors only. Criterion (iii) is mandatory for us given that 10 OOF rows carry 24.1 %
of SSE **[S, `docs/handoff.md` §3]**.

**2. Per-airport offset instead of a per-airport model: extend the reparameterisation.**
Evidence: the KDD paper's finding that local and global tie once the feature set is rich **[S]**;
NASA 1st place puts `y − ETD` reparameterisation into 2 of its 4 members **[S]**; the PRC 2024 winner
won with a per-aircraft-type target ratio and no specialists **[S]**; OpenAir run **every** expert as a
residual over a **per-airport OLS baseline** `P_cal` **[S]**; M5 2nd and 5th place both apply per-group
multipliers **[S]**; our own LIRF reparameterisation bought 20.7 s **[S, brief]**.
This is the cheapest item on the list and the one with the best cost/evidence ratio.
*Measurement that settles it:* replace `proxy = MVT_TIME − AOBT_3` with a per-airport affine baseline
(per-airport intercept, or per-airport OLS on the two clocks as OpenAir do) and re-run `make compare`
against the current proxy. Watch specifically whether the per-airport OOF residual means collapse; if
they do and RMSE improves, item 1 becomes much less likely to pay and can be dropped.

**3. Per-airport OOF bias correction — one constant per airport.**
Evidence: NASA 3rd place shipped the residual-offset stage and placed 3rd **[S]**; M5 2nd and 5th place
ship multiplier versions **[S]**; OpenAir measured per-airport residual means of **−236 s (LTFM)** and
**+214 s (LIRF)** on data of this shape **[S]**.
*Measurement that settles it:* compute per-airport OOF residual means b_g on the training folds, apply
`pred − b_g`, score. Then the honest check: compute b_g on an early period and apply it to a later one.
If the sign of b_g is unstable across periods, the correction is drift, not bias, and must not ship.
This is a few lines and should be run **before** item 1, because if it captures most of the per-airport
gain then the specialists are redundant.

**4. Make the global model bigger before making ten smaller ones.**
Evidence: Montero-Manso & Hyndman's complexity result — *"the complexity of the local approach grows
with the size of the set, while the complexity of the global approach remains constant"*, with
`∏ᵢ|Hᵢ| = |J|`, prescribing *"comparatively more complex hypothesis classes for global algorithms"*
**[S]**; the KDD paper's tie at 10 airports with a rich feature set **[S]**; Bojer & Meldgaard's
*"global ensemble models tend to outperform local single models"* **[S]**.
*Measurement that settles it:* fit model A with roughly the summed capacity of ten specialists (leaves
and/or trees scaled up, `min_data_in_leaf` reduced) and score it against both the current model A and
the item-1 specialist blend on the same folds. If the bigger global model closes the specialist gap,
item 1 is a capacity artefact and should not ship. This is the decisive control experiment and it is
missing from every competitor log I read.

**5. Sample weighting toward the group instead of separate models (λ·D_g + (1−λ)·all).**
Evidence: theory only — Mansour et al.'s data-interpolation bound, with λ *"obtained by either cross
validation or by using the generalization bounds"*, and EMNIST results within 0.1 % of model
interpolation **[S]**. **No competition write-up in this family uses it** — which is either an
opportunity or a signal, and I cannot tell which.
*Measurement that settles it:* one LightGBM `weight` column, λ ∈ {0.2, 0.5, 0.8} at the 2–3 airports
that pass item 1's gate, scored against the item-1 blend at those airports. It costs one extra column
and no new architecture; if it matches the blend it is strictly preferable (no small-sample regime,
one model shape).

**6. Multi-granularity pooling and equal-weight averaging (the M5 recipe).**
Evidence: M5 1st place averaged models pooled per store (10), store-category (30), store-department
(70), *"equal weighted combination (arithmetic mean)"*, and *"almost all"* of the top 50 pooled at one
of those granularities **[S]**. The group counts match ours; the per-group row counts are the same
order.
*Measurement that settles it:* train model A three times — pooled on all rows, pooled per airport,
pooled per airport×runway — and average the three predictions with equal weights. Score against model
A. This tests grouping without ever choosing a group, which is the property that made it robust in M5.

**7. Per-group *tuned* blend weights, and gating models. Do not build these.**
Evidence is **negative and from this competition**: Phoenix-Ops measured *"Airport-specific blend
weights | 323.712 vs 323.377"* and rejected them **[S to
`docs/competitor_research_2026-09-21/results/07-repo-phoenix-ops.md`]**; their *"3 mixture-of-baselines
classifiers"* *"worsened both months"* **[S to same]**; OpenAir refused to ship a blend when *"NNLS
weights unstable across splits"* **[S]**; Caruana et al. document that unconstrained stacking
*"overfits dramatically"* with many correlated inputs and a small validation set **[S]**. NASA 1st
place's per-airport decision was a **hard 0/1 switch validated on a holdout**, not a fitted weight
**[S]**. If per-group weighting is attempted at all, it should be that switch, gated by item 1's
bootstrap criterion.

**8. Specialists on sub-airport scopes (our model B, 22,470 rows; or LIRF-unmatched analogues).**
Evidence: Phoenix-Ops got **−1.6945 s official on 383 ranking rows** from a specialist on a 1,488-row
scope — but **only with 5 equal-weight seeds, each floored before averaging**, and their non-LIRF
unmatched specialist was **+0.012 s (worse)** **[S to 07-…md]**. OpenAir refuse any airport group below
**20,000 rows** outright **[S]**.
*Measurement that settles it:* if any sub-airport specialist is fitted here, fit ≥ 5 declared seeds up
front, average them equally, and discard none — and gate on both holdout months plus the single-row
share cap. Our model B RMSE is 1,864.9 with 48.1 % of total SSE **[S, brief]**, so the variance
problem there is far worse than Phoenix-Ops', and an unaveraged specialist on model B rows is the most
likely way to lose score while appearing to gain it in CV.

**Not recommended on current evidence:** per-group monotonic/isotonic calibration of the global output
(§3d) — no write-up in this task family reports a squared-error gain from it, and the affine special
case is already covered by items 2 and 3.
