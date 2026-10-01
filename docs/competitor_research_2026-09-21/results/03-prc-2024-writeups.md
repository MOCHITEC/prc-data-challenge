# SLOT 03 — PRC Data Challenge 2024 write-ups (target: Actual Take-Off Weight, RMSE in kg)

Investigator: SubAgent. Date: 2026-09-21.
Marking: **[S]** = read from a source (URL given). **[X]** = my own inference/extrapolation, not stated by any source.

> **Scope warning, stated up front.** The 2024 challenge target was **ATOW (a mass, kg)**, not a duration.
> The 2026 target (`resilient-hedgehog`) is **taxi-out time in seconds at 10 European airports**. A dedicated
> "what transfers / what does not" section is at the bottom. Several of the highest-value 2024 tricks are
> **specific to a mass target with known physical bounds and do not transfer**. Do not import them blindly.

---

## 1. Official sources located

| What | URL | Status |
|---|---|---|
| PRC Data Challenge hub | https://ansperformance.eu/study/data-challenge/ | [S] live |
| 2024 challenge site (dc2024 redirects here) | https://prc-data-challenge.netlify.app/ | [S] live; `https://ansperformance.eu/study/data-challenge/dc2024/` 301s to it |
| **Final leaderboard** | https://prc-data-challenge.netlify.app/rankings.html | [S] full 42-team table, server-rendered |
| Teams page (132 teams, self-descriptions) | https://prc-data-challenge.netlify.app/teams.html | [S] |
| GitHub org (team code) | https://github.com/PRC-Data-Challenge-2024 | [S] 29 `team_*` repos + `.github` |
| Editorial / data paper (JOAS) | https://journals.open.tudelft.nl/joas/article/view/8252 — doi:10.59490/joas.2025.8252 | [S] read in full |
| Dataset archive | https://doi.org/10.4121/8cb8484b-dbe7-4750-8b87-a5b1dbc621b4 | [S] cited in editorial |
| Live rankings API (2024) | https://datacomp.opensky-network.org/api/rankings | [S] referenced by rankings.html; not fetched by me |
| Results slides (Spinielli, OpenSky Symposium, Hamburg 2024-11-08) | `https://prc-data-challenge.netlify.app/media/PRC_Data_Challenge_2024_Enrico_Spinielli.pdf` | **[S] link exists on the site but returns 404 — slides NOT retrievable** |

### The "paper track" — what it actually is

There is a JOAS special issue **Vol. 3 No. 2 (2025): EUROCONTROL PRC 2024 Data Challenge**
(https://journals.open.tudelft.nl/joas/issue/view/1048). **[S]**

It contains exactly **three** items — not a collection of top-10 papers:

1. *Aircraft Takeoff Weight Estimation: The EUROCONTROL PRC 2024 Data Challenge* — Spinielli, Sun, Strohmeier, Olive, Goens, Koelle, Tart, Fitzgerald (editorial/data paper). doi:10.59490/joas.2025.8252 **[S]**
2. *Open Machine Learning Models for Actual Takeoff Weight Prediction* — Murça, Maximo, Dantas, Szenczuk, Lima, Carvalho, Melo (= **team_tiny_rainbow**, rank 2). doi:10.59490/joas.2025.7963 **[S]**
3. *Reviews and Responses for [2]* — open peer review; reviewers **Richard Alligier** (the rank-1 winner), Ryota Mori, Junzi Sun; editor Xavier Olive. doi:10.59490/joas.2025.8111 **[S]**

**Stated explicitly: only ONE team published a peer-reviewed paper in the special issue — team_tiny_rainbow (rank 2).**
I searched JOAS, SESAR Innovation Days 2024/2025, and the web for papers by other top-10 teams and **found none**.

- The rank-1 team (Alligier & Gianazza, ENAC) has a ResearchGate entry titled *"Inference of the Actual Take-Off
  Mass from ADS-B data using Machine Learning"* (RG id 388835553, dated Feb 2025). ResearchGate returns HTTP 403
  to my tools, HAL returned zero hits, and I could not confirm venue, authorship or content. **I could not verify
  that this is the rank-1 team's PRC write-up.** Treat as unconfirmed. **[S] that the RG record exists; [X] that it is their PRC paper.**
- SID 2024 paper 003, *"Aircraft Take-off Weight Prediction with Operational Data and Supervised Learning"*
  (Gheorghe, Sun, Ribeiro, Hop, Cramet — TU Delft + EUROCONTROL),
  https://www.sesarju.eu/sites/default/files/documents/sid/2024/papers/SIDs_2024_paper_003%20final.pdf
  is **adjacent work, not a challenge entry** — it predicts TOW from flight plan + TAF at Schiphol. **[S]**

**So: the real corpus of top-10 technical write-ups is the GitHub READMEs, plus one JOAS paper + its open reviews.**

---

## 2. Verified final leaderboard (top 20 of 42 scored teams)

Source: https://prc-data-challenge.netlify.app/rankings.html **[S]**. RMSE in kg on `final_submission_set.csv`
(158,149 flights = 105,959 public + 52,190 held back). "Version" = submission number.

| Rank | Team | RMSE (kg) | Ver | Repo in org? | Write-up depth |
|---:|---|---:|---:|---|---|
| 1 | **team_likable_jelly** | **1,561.63** | 25 | yes | **excellent** (17 KB README, full method) |
| 2 | **team_tiny_rainbow** | 2,217.75 | 1 | yes | **excellent** (README + 26 p. JOAS paper + open reviews) |
| 3 | **team_gentle_elephant** | 2,252.89 | 11 | **NO** | **none found** |
| 4 | **team_brave_pillow** | 2,270.06 | 101 | yes | good (4 KB README, two architectures) |
| 5 | **team_delightful_avocado** | 2,355.61 | 16 | yes | good (README + full data-prep doc) |
| 6 | **team_youthful_xerox** | 2,386.53 | 8 | yes | good (14 KB README) |
| 7 | **team_affectionate_bridge** | 2,456.70 | 6 | yes | thin (1.3 KB) but has useful claims |
| 8 | **team_honest_turtle** | 2,479.56 | 46 | yes | verbose (32 KB) but feature-list-only, no results |
| 9 | **team_exuberant_scooter** | 2,587.50 | 7 | yes | thin (notebook index only) |
| 10 | **team_mindful_donkey** | 2,683.05 | 9 | yes | **excellent** (13 KB, staged architecture + per-stage RMSE) |
| 11 | team_diligent_igloo | 2,692.37 | 16 | yes | not read |
| 12 | team_amazing_forest | 2,695.60 | 18 | yes | **two public blog posts** (see §5) |
| 13 | team_modest_scooter | 2,696.67 | 6 | yes | not read |
| 14 | team_gentle_wreath | 2,702.16 | 20 | yes | not read |
| 15 | team_elegant_lemon | 2,746.30 | 26 | yes | not read |
| 16 | team_patient_net | 2,752.72 | 10 | yes | not read |
| 17 | team_faithful_engine | 2,784.52 | 5 | yes | good (README + project_overview.md) |
| 18 | team_mellow_barn | 2,859.32 | 14 | yes | not read |
| 19 | team_exuberant_hippo | 2,932.55 | 8 | yes | not read |
| 20 | team_zealous_watermelon | 3,024.17 | 1 | yes | not read |

Tail for scale: rank 30 = 3,960.86; rank 39 = 12,261.94; rank 42 = 89,851.11. **[S]**
Naive **MTOW baseline = 32,324.13 kg RMSE / 26.10 % MAPE** (from the tiny_rainbow paper, Table 6). **[S]**

### Corrections to the prior notes in this repo

- **[S] CONFIRMED:** 1st = `team_likable_jelly`, LightGBM + physics + METAR + normalised target. Exactly right.
- **[S] CORRECTION:** `team_tiny_rainbow` (CatBoost+LGBM+XGB+MLP+SAINT stack) was **rank 2, not rank 3**, and
  received the **2nd prize (€1,750)**. The prior note saying "3rd = tiny_rainbow" is wrong.
- **[S]** Rank 3 on RMSE was `team_gentle_elephant` (2,252.89), about which **nothing is published** — it has
  no repo in the GitHub org and does not appear in the org's 29-repo listing.
- **[S]** The **prize** podium is not the RMSE podium: prizes went to likable_jelly (€2,500),
  tiny_rainbow (€1,750), **brave_pillow (rank 4) €750**. Source: https://prc-data-challenge.netlify.app/ §"And the Winners are…".
- **[X]** My inference: gentle_elephant was skipped for the 3rd prize because prize eligibility required
  publishing all source code under GPLv3 on the OpenSky GitHub account (stated on rankings.html **[S]**) and
  gentle_elephant has no repo there **[S]**. The causal link is my inference; no source states it.

---

## 3. Rank-by-rank technical extraction

### Rank 1 — team_likable_jelly — RMSE 1,561.63
Richard Alligier & David Gianazza, ENAC Toulouse.
Source: https://github.com/PRC-Data-Challenge-2024/team_likable_jelly (README read in full) **[S]**

**Model family.** LightGBM only. No other model family. **[S]**

**Ensembling structure.** *Seed averaging of one architecture* — not stacking, not multi-family blending.
Identical data, identical hyperparameters, different `random_state`; predictions averaged. Their own numbers **[S]**:

| Models averaged | RMSE (final set) |
|---:|---:|
| 1 (seed 0) | 1,612.41 |
| 10 (seeds 0–9) | 1,564.09 |
| 20 (seeds 0–19) | **1,561.63** |

**So seed-averaging bought 50.8 kg (3.2 %), and 10→20 models bought only 2.5 kg.** Their justification is the
bias–variance decomposition, noting it applies to any model randomness, not just bagging. **[S]**
Each model: 50,000 trees, ~65 min on a ThreadRipper 1920X. **[S]**

**Target transformation — the single most distinctive choice. [S]**
```
y = (TOW - EOW) / (MTOW - EOW)          # normalised payload+fuel fraction, mostly in [0.2, 1]
sample_weight = (MTOW - EOW)**2          # so that optimising MSE on y == optimising MSE on absolute kg
```
Their stated rationale: without it "the training process consumes a lot of splits of `aircraft_type` and `wtc`
just to get the mass range right." The squared-scale sample weight is what keeps the objective aligned with the
*absolute* competition metric. **This is a matched pair — transform + reweight — not a bare target transform.**

**Feature scaling by group — second distinctive idea. [S]**
Tree models are invariant to any *strictly monotonic* transform of a feature, so rescaling does nothing —
**unless the scaling differs per group.** They scaled the physics-derived mass features per aircraft type by the
same `(mass - EOW)/(MTOW - EOW)` formula, so "1" means heavy regardless of type. They also built a general
`ScaleByGroup` class (e.g. scale `flown_distance` by aircraft type) but **did not use it in the final model** —
kept in the repo as a documented idea. **[S]**

**Feature engineering categories. [S]**
1. *Trajectory cleaning* (prerequisite, treated as essential): drop ADS-B fields repeated between consecutive
   messages when no fresh measurement arrived (detected via which fields update together per the Mode-S spec);
   own re-implementation of `traffic`'s `FilterDerivative`; **explicitly no interpolation**; isolated points
   >20 s from any neighbour discarded.
2. *Smoothing*: cubic smoothing splines (`csaps`), never across gaps >20 s.
3. *Timestamp correction*: `arrival_time`/departure in the flight list are sometimes wrong. They recompute from
   the trajectory: take the extremal timestamp among points within a 10 NM radius of the airport, then add a
   buffer implied by that point's altitude at 1,000 ft/min. Radius <10 NM left too many flights uncorrected.
4. *Altitude-sliced climb features* — 48 slices of 1,000 ft each, from [−500, 500] ft to [46500, 47500] ft.
   Per slice: point count, median ROCD, max−min ROCD, median TAS, median ΔT, min/median/max **energy rate**,
   time since departure, time to arrival, and median **equivalent mass**. "Hundreds of features."
   The alignment idea: a fixed altitude band means the same *physical concept* across flights; a temporal
   feature per slice lets the model know *when* the aircraft got there.
5. *Whole-flight temporal profile* — 20 slices of scaled completion `(t - t_adep)/(t_ades - t_adep)`; per slice
   median Mach, median altitude, altitude last−first, plus total duration. Captures cruise. They note the
   scaled slices misalign at the ends between short and long flights, and accept it because the altitude-sliced
   climb features already cover the beginning.
6. *Physics*: **energy rate** = rate of energy change divided by mass, computed from kinematics alone;
   **equivalent mass** solved from a degree-2 polynomial from a point-mass model, using **OpenAP**.
7. *Wind*: mean of `dot(wind, groundspeed)/|groundspeed|` along the trajectory — a signed head/tail-wind scalar.
8. *METAR at ADEP and ADES*: METARs are missing for some airport/times, so a `sklearn.neighbors.NearestNeighbors`
   index over the space–time domain retrieves the nearest available report.
9. *METAR thunderstorm/fog at ADES* — their explicit hypothesis: bad weather forecast at destination ⇒ holding
   risk ⇒ crew tankers extra fuel ⇒ heavier. Because the right space–time radius is unknown, they computed the
   summary at **several radii simultaneously and let LightGBM pick**. **[S]** — a concrete "don't guess the
   hyperparameter, emit the family and let the model select" pattern.

**Validation scheme.** Random search over LightGBM hyperparameters against a **validation set** (`optimparam.py`).
No cross-validation is described. **[S]** No fold structure or grouping is mentioned.

**Post-processing.** **None described.** No clipping, no per-type calibration, no blending with a physical model. **[S]**

**Stated negative results — what did NOT work. [S]**
- *Descent-phase features*: "One can proceed similarly on the descent phase but it did not provide much
  improvement in our tests, so we did not use it in the end." Their guess: descent is noisier and further in time
  from take-off, so the link to TOW is looser.
- *The physics "mass" feature may be redundant*: "in hindsight, these mass variables might not be that useful,
  considering that the energy rate variable contains very similar information." They also say a 4th-degree
  aggregate solve might have been better than a per-point degree-2 solve + median, but they did not pursue it.
- *Reproducibility failure modes, self-reported*: (a) the Iowa Mesonet METAR archive **mutates** — a station's
  location changed within a two-week window, so they had to archive the METARs to freeze results;
  (b) LightGBM's default `deterministic=False` prevents exact re-runs; with `deterministic=True` they reproduce
  1,561.48 vs the submitted 1,561.63 — and only on the same machine.

**External data. [S]** ourairports.com `airports.csv`; **Iowa State Mesonet ASOS METAR archive**
(https://mesonet.agron.iastate.edu/ASOS/); `aircraft_type_masses.csv` built from OpenAP data + Wikipedia for
MTOW/EOW not present in OpenAP. Libraries: LightGBM, scikit-learn, sklego, csaps, traffic, OpenAP, pitot.

---

### Rank 2 — team_tiny_rainbow — RMSE 2,217.75
Murça et al., Aeronautics Institute of Technology (ITA), Brazil.
Sources: repo README **[S]** + JOAS paper doi:10.59490/joas.2025.7963 (read in full) **[S]** + open reviews doi:10.59490/joas.2025.8111 **[S]**

**Model family.** CatBoost, LightGBM, XGBoost, FastAI Tabular MLP, and **SAINT** (Self-Attention and
Intersample Attention Transformer) — five learners, then an ensemble. Optuna for GBDT hyperparameters. **[S]**

**Ensembling structure and weights — the headline finding of this whole slot. [S]**
Adaptive stacking: weights `w_h` learned by **Quadratic Programming** minimising squared error on a held-out
20 % validation split, subject to `Σ w = 1`, `w ≥ 0` (a non-negative least squares / Super Learner form,
solved with `qpsolvers[proxqp]`). Two-stage: models trained on 80 %, weights fitted on the held-out 20 %, then
**every model retrained on 100 % of the data** and the pre-computed weights applied.

Reported weights (Table 6, in the table's row order MLP, SAINT, CatBoost, LightGBM, XGBoost):
```
MLP      1.24e-23   ~0
SAINT    1.63e-23   ~0
CatBoost 8.65e-01
LightGBM 4.77e-22   ~0
XGBoost  1.34e-01
```
**The five-model stack collapsed to 0.865·CatBoost + 0.134·XGBoost.** The MLP, SAINT and LightGBM members
received numerically-zero weight. Their own ablation confirms it:

| Removed | RMSE | Δ |
|---|---:|---:|
| — (full model) | 2,210.37 | 0.00 % |
| **CatBoost** | 2,969.49 | **+33.90 %** |
| XGBoost | 2,213.91 | +0.16 % |
| **LightGBM** | 2,210.37 | **+0.00 %** |
| **Neural network (MLP)** | 2,210.37 | **+0.00 %** |
| **SAINT** | 2,210.37 | **+0.00 %** |
| Operational features (trajectory-derived) | 2,759.18 | **+24.83 %** |
| Geographical features | 2,390.26 | +8.14 % |
| Temporal features | 2,354.97 | +6.54 % |
| Aircraft characteristics | 2,266.96 | +2.56 % |
| **Meteorological features** | 2,249.29 | **+1.76 %** |

The paper itself concludes those three members "may be redundant in the current ensemble configuration, and
thus pruned." **[S]**

Individual model scores (estimated test / actual on submission set) **[S]**:
CatBoost 2,215.53 / **2,224.69** (MAPE 1.75 %); XGBoost 2,413.05 / 2,403.07; LightGBM 2,506.43 / 2,519.17;
MLP 3,864.1 / 3,864.2; SAINT 3,921.3 / 3,921.5; **Ensemble 2,210.37 / 2,217.75 (MAPE 1.73 %)**.
**A single tuned CatBoost was within 7 kg (0.3 %) of the full five-model stack.**

**Validation scheme. [S]** 80/20 local split for the reported "estimated test RMSE"; a separate 5 % of training
data as a validation set for early stopping / model selection; the 20 % holdout for ensemble weights.
**No cross-validation, no grouped or temporal folds.**

**Target transformation. [S] NONE.** They regress raw ATOW in kg directly. This is the clearest single
methodological difference from rank 1, alongside the absence of physics features.

**Post-processing. [S] None described.**

**Feature engineering categories. [S]**
- Flight-phase segmentation (departure / en route / arrival), then per phase: mean vertical rate, airspeed,
  **squared airspeed**, ground speed, temperature, humidity, altitude; **max altitude as the 99th percentile**
  (explicitly to blunt noisy observations); total wind distance (Σ tail/head-wind × time); flown distance;
  specific energy `V² + gh` at the last point.
- *First-ten-points features*: for flights whose first observation is within **2 NM of the origin airport**,
  the time series of airspeed, squared airspeed, temperature, height AGL, specific energy and vertical rate for
  the first 10 post-takeoff samples.
- *Vertical-efficiency features* following EUROCONTROL's method: within 200 NM and above 3,000 ft AGL, find
  Top-of-Climb, define an exclusion box above 90 % of TOC, flag climb segments with ROC ≤ 300 ft/min lasting
  >5 min outside the box as **level-offs**; sum their time and distance. Rationale: level segments burn more
  fuel; step climbs correlate with weight.
- Literature-driven: specific energy at 10 NM flown; lift-off airspeed and groundspeed.
- *Temporal*: hour, day-of-week, month, week-of-year, season; boolean weekend; boolean rush hour (07–09, 16–18,
  chosen from their EDA).
- *Duration-based*: **taxi ratio = taxiout_time / flight_duration**; average speed = distance / duration;
  duration binned into Very Short (0–60 min) / Short (60–180) / Medium (180–300) / Long (>300);
  missing flight speed imputed with the median.
- *Geographical*: haversine ADEP–ADES distance; airport altitude difference; bearing; elevation gradient;
  **K-Means clustering of airport lat/lon**; region grouping by country code; domestic/international flag;
  same-area flag; direction-pair category; intercontinental flag.
- *Interaction*: mean airspeed × specific energy, per phase.
- *Aircraft characteristics*: MTOW, OEW, max payload, fuel capacity, engine type and count, aircraft age class
  (new / mid-life / older), seats, wing span and area, range.
- *Binning*: arrival-phase mean temperature and departure-phase mean humidity each into 5 equal-frequency bins.
- *Ratios*: vertical rate / airspeed per phase.
- Missing values imputed with the **median**.

**External data. [S]** OpenFlights Airports Database; Global Airport Database (partow.net); FAA Aircraft
Characteristics Database; plus a hand-built `aircraft_extra_v3.csv` scraped from Wikipedia, technical sheets,
manufacturer specs and EUROCONTROL data. Trajectory processing in **R** (`arrow`, `geosphere`, `airportr`,
`data.table`, `REdaS`); modelling in Python.

**Top CatBoost features named in the paper. [S]** actual distance flown, seats (typical), MZFW, cargo capacity,
average en-route altitude — plus a "significant contribution of the **airline** feature, emphasizing the impact
of different operational policies and preferences on fuel loading."

**Stated limitations. [S]** Purely data-driven, no aerodynamic constraints; ADS-B coverage gaps may hurt
generalisation across regions.

---

### Rank 2's open peer review — reviewer = the rank-1 winner. Highest-signal document in the corpus.
Source: https://journals.open.tudelft.nl/joas/article/view/8111 (doi:10.59490/joas.2025.8111), read in full **[S]**

Alligier's technical objections, all directly reusable as engineering rules:

- **A3 / A4 — monotone transforms are no-ops for trees.** `taxi_ratio` divided by its global median is "simply a
  linear transformation of the feature that will have no impact on tree-based algorithms/models (which are
  invariant to strictly monotone transformation of their feature) and neural network models in which each
  feature is typically normalized/standardized beforehand." Likewise "speed per distance" turned out to equal
  `1/flight_duration`. He asked for the claims to be removed so readers are not misled.
- **A4 (second half) — duplicated monotone features still cost you.** A strictly monotone copy of an existing
  feature has "no impact … aside from **multiplying by 2 the probability to be considered as a split feature**
  during the tree growing phase." Same point for `tas_1` vs `sqrd_tas_1` in A5: they are "basically the same
  feature, and this might have an impact on their respective importance score."
- **A11 — the ensemble theory in the paper was wrong.** The bias–variance/bagging framing does not hold when
  weights are fitted on data; the correct framing is **stacking (Breiman) / Super Learner (van der Laan)**,
  because the weights are not constants with respect to the training set. The authors amended the paper.
- **A16 — ADS-B coverage is a confound in "first N points" features.** "the first ten observed points are
  somewhat random: they are the first points inside the ADS-B coverage. The altitude 'frontier' of the ADS-B
  coverage may vary from one location to an other." Two flights differing in `VerticalRate_1` may differ only
  in where coverage started. A height-AGL feature can mitigate it, but "the tree based models will have to
  'chain' two conditions/nodes … this consumes tree nodes (and hence model complexity), and as the tree growing
  algorithm is a greedy algorithm … it is not certain that the first condition will be selected."
  **This is an argument for encoding the correction into the feature rather than hoping the tree learns it.**
- **A17 — median-impute vs leave-missing.** "sometimes if the fact they are missing is correlated to the
  variable to be predicted, it is better to left them as missing values. This way, missing values can be treated
  separately, and be put on the left or right side of the condition which should help reduce the training loss."
  He judged median imputation correct *here* only because missingness was a priori independent of mass.
- **A2 (also M-series) — unequal ADS-B coverage biases aggregate trajectory statistics.** Two flights' level-off
  time can differ purely from coverage, not from operations. The authors answered that they did **no**
  trajectory cleaning and **no** coverage handling at all (§ response to A1/A2). **[S]** — i.e. rank 2 skipped
  entirely the step rank 1 treated as foundational.
- Reviewer 2 (Mori): prefer **ISA deviation over mean temperature**; prefer max over mean altitude; flags
  `mean vertical speed` and `elevation gradient` as likely redundant; asks for **per-aircraft-type RMSE**
  because "relying solely on a single overall RMSE might mask performance discrepancies"; asks for a **naive
  baseline** in the results table (this is where the MTOW baseline of 32,324.13 came from).
- Reviewer 3 (Sun): archive the repo with a Zenodo DOI.

---

### Rank 3 — team_gentle_elephant — RMSE 2,252.89
**No write-up exists that I could find.** No repo in the PRC-Data-Challenge-2024 GitHub org (verified against
the org's 29-repo listing **[S]**), no paper, no blog, no hits in web search. Their teams-page entry exists but
I did not capture a technical description. **Stated plainly: rank 3's method is unknown.**

---

### Rank 4 — team_brave_pillow — RMSE 2,270.06  (3rd prize, €750)
Fazlur Rahman, Aidana Tassanbi (TU Delft PhD students), Bintang A.S.W.A.M (Skoltech MSc).
Source: https://github.com/PRC-Data-Challenge-2024/team_brave_pillow README **[S]**

**Two architectures built, one chosen. [S]**
1. **"Median voting"** — train many models over varying feature subsets/parameters and take the **median** of
   their predictions. Rationale given: "the deviation of each prediction using different features deviates
   differently from the ground truth. Thus, following the law of large number, the median voting can be
   helpful." Achieved 2,318 RMSE.
2. **"RMSE filtering"** — **per-aircraft-type specialist models.** Aircraft types with <1 % share merged into
   one bucket. For each type-of-interest, **undersample the other types** and train a dedicated model; log
   validation RMSE per aircraft type for every model; at prediction time select, per aircraft type, the model
   with the lowest validation RMSE. Local validation RMSE 2,066; **final submission 2,276** — a 10 % gap.

They chose RMSE filtering despite the worse local number, reasoning: "This model, in our opinion, is still
better than training more models to squeeze water out of a stone and got only slightly better results." **[S]**

**Stated negative result / honest failure. [S]** For median voting: "We got 2250-ish before the
final_submission_set was released … Unfortunately, we could not reproduce it, it was a messy day where we
trained so many models, explored so many ML techniques, and lost track of it (or you may say it was beginner's
luck)." **A direct, documented cost of not versioning experiments.**

**[X] my reading:** the 2,066 → 2,276 gap on the per-type selection scheme is the signature of selection
overfitting — choosing the argmin model per group on the same validation split used to measure it. No source
says this; it is my inference.

**Feature engineering. [S]** Resampled trajectories; a dedicated **second, non-resampled climb-phase pass**
added later ("At some point we realized that climb-phase is quite important"); airport distance; drag;
imputation. Tools: OpenAP, traffic, AirportsData.

**Target transformation / post-processing.** None described. **[S]**

---

### Rank 5 — team_delightful_avocado — RMSE 2,355.61
Felix Soest, Paul Hollmann.
Source: README + README_datapreparation.md **[S]**

**Model family. [S]** A **single** scikit-learn `HistGradientBoostingRegressor`. No ensembling at all.
**Rank 5 with one model and no blending.**

**Validation / tuning. [S]** Bayesian optimisation via `scikit-optimize`. Single train/validation setup;
no CV scheme described.

**Target transformation. [S] None** — raw kg.

**Post-processing. [S] None.**

**Feature engineering categories. [S]** (their data-prep doc is unusually complete)
- Times converted to **local time of the respective aerodrome** before extracting month/day/hour/minute/
  day-of-week/day-of-year/is_month_start/is_month_end/is_weekend/quarter/**iata_schedule_flag**
  (summer vs winter schedule)/is_holiday (minimal set: Christmas, New Year).
- `adep`, `ades` one-hot with `min_frequency=50`; **`airport_pair` one-hot with `min_frequency=450`**;
  `aircraft_type`, `airline`, country codes one-hot; `wtc` ordinal.
- **`flights_per_day`** = total flights in the challenge set on that date — a *demand/congestion* proxy.
- Trajectory: avg TAS, avg wind u/v, max cruise altitude, ADS-B in-flight time, boolean flags for whether each
  phase (ground/climb/cruise/descent) is present at all — an explicit **coverage indicator**.
- Phase features: climb_rate mean/max, **climb_efficiency = mean climb rate / mean ground speed**, mean cruise
  altitude, cruise TAS, cruise wind u/v/total, cruise ground speed,
  **`fuel_efficiency_proxy` = mean ground speed − mean TAS**, descent rate mean/min.
- Point features: TAS/temperature/humidity at landing and at departure; **`init_climb_rate` = mean climb rate
  over the first 30 s**; **`time_to_10k`** = time to reach 10,000 ft.
- Derived physics: barometric pressure at departure/landing from aerodrome elevation; air density; **CAS**
  from TAS+pressure+temperature; CAS²; and `landing_mass`/`dep_mass` proxies from `F_L ∝ ρ v_CAS²`.
- **Per-aircraft-type mean of each trajectory feature, used to scale that feature** — the same
  "scale by group" idea rank 1 described, arrived at independently.

**SHAP analysis, reported. [S]** Highest-SHAP features are `mtom`, `aircraft_type`, `oew` — all aircraft-type
proxies. Their conclusion: performance is bounded by not knowing the **aircraft variant**, and adding variant
data would be the biggest available gain. Mean climb rate has an **inverse** effect on TOW ("an empty aircraft
having better climb performance"). `airline` shows real influence, "possibly indicating different operating
specifics", though they caution it correlates with aircraft type.

**External data. [S]** OurAirports (elevation, timezone, coordinates); `masses.csv` (OEW/MTOM) hand-built from
Wikipedia and "augmented, taking into account the challenge data set takeoff weights."

---

### Rank 6 — team_youthful_xerox — RMSE 2,386.53
Source: README **[S]**

**Model family. [S]** XGBoost, Optuna-tuned.

**Target transformation — notable. [S]** "instead of predicting the TOW directly we try to predict the
**(mean(TOW@ChallengeSet) − TOW)**. This methods boosted considerable our performance in the final
submission_set." (`module/xgboost_mean_diff.py`.) A **residual-from-global-mean** target, motivated explicitly
by aircraft-type class imbalance.

**Per-group specialists. [S]** A second track (`module/xgboost_model_categories.py`) trains one model per
**group of aircraft types**, layered on top of the mean-difference target. Groups are user-redefinable.
No comparative RMSE for this track is reported.

**Feature engineering. [S]** Two explicit tracks:
1. *General extraction*: mean, max, std of every signal.
2. *Climb & take-off segmentation*: hand-crafted phase segmentation robust to noise and missing chunks, with
   two tunable knobs they flag as dataset-dependent (`vertical_rate_threshold`, `min_duration_threshold_minutes`).
   Features: altitude, groundspeed, vertical rate, TAS, groundspeed difference, **track deviation**,
   **track variance**, take-off duration. Pipeline in **Polars** for speed/memory.
Their justification for focusing on climb: "In the literature, many papers confirm that the Take-Off Weight
(TOW) is strongly related to the vertical rate and speed of the aircraft during the early stages of flight."

**Validation.** Optuna objective + RMSE evaluation; no CV scheme described. **[S]**
**Post-processing. [S] None.**

**External data. [S]** Global Airport Database (partow.net, MIT); **CADO airplane database** (~230 airplanes,
31 parameters each — geometry, mass, cruise Mach, range, approach speed, take-off/landing field length, engine
count/type, bypass ratio, max thrust), doi:10.57745/LLRJO0, ODbL 1.0.

---

### Rank 7 — team_affectionate_bridge — RMSE 2,456.70
Solo entrant ("just myself", works in data analytics).
Source: README **[S]**

**Model family. [S]** LightGBM, in **R**, tuned by `lightgbm_hyperparam_tuning.R`.

**Three claims worth carrying. [S]**
1. **"It's actually quite possible to build a decent lightweight model (around 3,000 RMSE) without the
   trajectories at all."** — a no-trajectory baseline reaching what would have been ~rank 20–21.
2. **"Generally, error was much larger with heavier aircraft"** — heteroscedasticity by aircraft class.
3. The model learned to predict between MTOW and empty weight; the apparent violations traced to **bad
   reference data**: "The A321 and 767 MTOWS are probably wrong (I pinched them off Wikipedia)."
   **A documented case of the external reference table being the error source, not the model.**

**Target transformation / ensembling / post-processing. [S] None described.**

---

### Rank 8 — team_honest_turtle — RMSE 2,479.56
Somnath Panigrahi, TU Dresden.
Source: README (32 KB) **[S]**

**Model family. [S]** AutoML for tuning and model selection, over LightGBM, CatBoost, and neural networks
(`NNTorch` / custom). No ensemble weights, no per-model scores, and **no reported RMSE anywhere in the README** —
the document is essentially an annotated feature catalogue.

**Feature engineering. [S]** ~230 features, catalogued by number and source. Categories: basic flight fields;
ADS-B derived (duration, distance, **number of position reports** — a coverage proxy, avg/max groundspeed,
avg/max vertical rate, **number of climb/descent cycles**, avg track change); per-phase climb/cruise/descent/
landing blocks (items 34–101); estimated total fuel consumption; take-off and landing environment metrics;
METAR-style weather at ADEP and ADES; **airport runway length/width/elevation**; ~30 aircraft performance
parameters from EUROCONTROL/FAA databases; **ratios of observed ADS-B metrics to the aircraft's book values**
(items 175–184) — e.g. observed climb rate ÷ spec initial climb rate; ADEP−ADES temperature/humidity/altitude
differences; flight-duration-efficiency metrics; **runway-length-to-wingspan ratios**; categorical derivations
(flight type, day/night, season, distance category); binned speed groups.

**Validation. [S]** "cross-validation, and hold-out test sets" are asserted in prose; no scheme is specified.
**Target transformation / post-processing. [S] None described.**

**External data. [S]** EUROCONTROL Aircraft Performance Database; FAA Aircraft Characteristics Database;
**Meteostat API**; OurAirports; **Eurostat airport movement data 2010–2024** (traffic/congestion context).

---

### Rank 9 — team_exuberant_scooter — RMSE 2,587.50
Swiss startup (io-swiss). Mirror at https://github.com/io-swiss/io-prc-atow.
Source: README + notebooks/README **[S]**

Thin documentation. What is stated: trajectories cleaned to keep only the airborne portion and only the fields
needed; a **normalised vertical rate** and **airspeed** feature pair developed by experimenting on random
trajectory samples via `traffic`; general aggregations plus **density altitude**; single `training.ipynb`.
**No model family, ensembling, validation scheme, target transform or post-processing is documented.** **[S]**

---

### Rank 10 — team_mindful_donkey — RMSE 2,683.05
Mark Fahey (solo). Source: README **[S]** (mirror: github.com/mtfahey/prc_challenge)

**The most explicitly staged architecture in the top 10, with per-stage RMSE — unusually honest reporting.**

**Structure. [S]**
- **Stage I** — ensemble on flight-level data **excluding** trajectories, plus coarse climb descriptors.
- **Stage II** — **one model per aircraft type (27 types)**, trained on 30-second-resampled climb-phase ADS-B,
  predicting TOW at *each point in flight*.
- **Stage III** — meta-model combining Stage I output, Stage II outputs **binned as medians at 100-second
  intervals** (`stage_two_100` … `stage_two_1000`), aircraft type, and **Stage II's own per-type
  `percent_error`** (its test RMSE as a fraction of that type's median TOW) as a feature.

**Reported numbers. [S]**
| Configuration | Submission RMSE |
|---|---:|
| Stage I alone | 2,719 |
| Median of Stage II (Stage I where trajectories missing) | **6,140** |
| Stage III (I + II combined) | **2,683** |
| Final model, local test RMSE | 2,398 |

**Read this carefully: the trajectory-based per-aircraft-type stage was, on its own, catastrophically worse
(6,140), yet combining it with the flight-level model still improved on the flight-level model alone
(2,719 → 2,683, i.e. 1.3 %).** Stage I H2O internals: 5-fold CV, **GLM metalearner**, 16 of 30 candidates in
the ensemble (8/10 GBM, 6/10 XGBoost, 1/2 DRF, 1/7 DeepLearning); training RMSE 2,036, CV RMSE 2,753, test
RMSE 2,749 — **a 700 kg train-to-CV gap, disclosed.**

**Validation — the one team with explicit leakage control. [S]** In Stage II, "Flight ids were divided into 10
fold groups **to prevent cross-training within each flight**", and training aimed at "maximizing RMSE across
fold groups without training within a flight id." This is **grouped CV with the flight as the group** — the
only such discipline documented in the top 10.

**Feature engineering. [S]** OpenAP OEW/MTOW per type (FAA data as fallback); great-circle distances
(`airportsdata` + manual additions); **an estimated idealised fuel weight per ADEP/ADES/type from
openap-top (trajectory optimizer)**, with a nearest-average-TOW aircraft substituted when a type is missing and
a linear regression on distance substituted when an airport is missing; `traffic` for filtering/resampling
(30 s) and phase labelling; max fuel flow per flight; **altitude and time to start of cruise**; cross-wind and
tail-wind via Yoshiki Kato's `weather_parameters` formulas; local-time hour, day-of-week, month.

**Target-adjacent engineering — his strongest idea. [S]**
```
est_load_lf_adjusted = (mtow_fill - oew_fill - total_fuel_fill) * monthly_European_load_factor
est_tow              = est_load_lf_adjusted + oew_fill + total_fuel_fill
```
i.e. a **physically-constructed prior estimate of TOW fed in as a feature**, using **monthly IATA European
passenger load factors** as external data. Not a target transform — a structural prior as an input.

**Stated negative result. [S]** "I have also included some code here for replacing adep/ades values in the
submission data that did not appear in the training data, but this does not appear to have improved the output
and was not included in the final model." — **unseen-category remapping did not help.**

**External data. [S]** OpenAP + openap-top; FAA aircraft data; `airportsdata`; **monthly IATA European
passenger load factors**; `country_timezones`.

---

### Rank 17 — team_faithful_engine — RMSE 2,784.52 (included: AutoML reference point)
Source: README + `documentation/project_overview.md` **[S]**

**Model family. [S]** **AutoGluon** end-to-end — "will try all kinds of gradient boosted trees, sklearn
classifiers and neural networks and finally builds a stacked ensemble automatically", run with an explicit
wall-clock budget (their final command uses 10 hours). W&B for tracking; joblib caching.

**14 sequential preprocessors. [S]** airport data (coords, timezone, distance) → OpenAP aircraft/engine
properties → **scraped per-airline aircraft cabin configurations** → **UN fuel-price data** → runway
length/elevation → **EU airport passenger-flow data** → METAR archives at destination arrival →
derived weather-safety indicators → day-of-week/route combinations → trajectory (cruise, climb, headwind) →
OpenAP fuel-flow estimates (A320 baseline) → ratio-based linearisation → "creative features" → drop
consistently-ignored columns.

**Stated negative results / deliberate abstentions. [S]**
- **They refused to fit the submission distribution:** "We could have specifically filtered our training data
  to be most similar to the samples in the submission set or we implemented different weights for sample to
  boost our score. However, we decided that this would likely lead to overfitting."
- US DOT **T100 filings** integration attempted but unfinished (time).

> **Discrepancy flagged.** The conormclaughlin.net review (§5) states team_faithful_engine scored **RMSE 1,635**.
> The official leaderboard shows **2,784.52** at rank 17. **I could not reconcile these.** The official
> rankings page is authoritative for the final ranking; treat the blog's 1,635 figure as unverified. **[S]/[X]**

---

## 4. Rank 1 vs ranks 5–10: what actually separated them

The distribution is the story. **[S]** from the leaderboard:

```
rank  1: 1,561.63   <-- alone
rank  2: 2,217.75   +42% worse than rank 1
rank  3: 2,252.89
rank  4: 2,270.06
rank  5: 2,355.61
rank 10: 2,683.05   +21% worse than rank 2
rank 20: 3,024.17
```
**Rank 1 beat rank 2 by 656 kg (29.6 % lower RMSE). Ranks 2 through 10 span only 465 kg.**
The 2024 challenge was won by one team by a margin larger than the entire spread of the rest of the top ten.

**By the teams' own accounts, rank 1 did four things nobody else in the top 10 did all of:**

1. **A physically-motivated normalised target plus a matching sample weight.** `(TOW−EOW)/(MTOW−EOW)` with
   `weight=(MTOW−EOW)²`. **[S]** Ranks 2, 4, 5, 7, 8, 9, 10 regressed raw kg. Rank 6 used a mean-difference
   target and reported it helped "considerably" **[S]** — the only other team with a deliberate target transform.
2. **Serious trajectory cleaning treated as a first-class stage.** De-duplication of repeated ADS-B messages,
   a custom derivative filter, **no interpolation**, isolated-point removal, cubic smoothing splines, and
   recomputation of departure/arrival timestamps from the trajectory itself. **[S]** Rank 2, when asked directly
   by Alligier in review, confirmed they did **no** cleaning at all. **[S]**
3. **Physics features from a point-mass model (OpenAP): energy rate and per-slice equivalent mass.** **[S]**
   Ranks 4, 10, 17 used OpenAP for MTOW/OEW/fuel-flow lookups; only rank 1 used it to solve a mass equation
   per trajectory point.
4. **Concept-aligned slicing.** 48 fixed 1,000-ft altitude slices for climb + 20 normalised-completion slices
   for the whole flight, explicitly designed so "a given feature should represent the same concept from one
   trajectory to another." **[S]** Other teams aggregated per broad phase (departure / en route / arrival).

Note what rank 1 **did not** do: no multi-family stacking (LightGBM only), no per-aircraft-type specialists,
no AutoML, no post-processing, no cross-validation (a single validation set), and no neural networks. **[S]**
Their ensembling was **20 seeds of the same model, worth 3.2 %.** **[S]**

**The mirror-image evidence from rank 2's ablation:** operational (trajectory) features were worth +24.83 % and
meteorological features only +1.76 % **[S]** — yet rank 1's distinctive weather work was *forecast-driven*
(thunder/fog at the **arrival** airport at **several** space–time radii, as a tankering proxy), not
current-conditions-at-departure. **[X] my inference:** these are not contradictory; weather helps mainly through
the fuel *decision*, and rank 2's mean-temperature-per-phase features do not capture that. No source states this.

### Consolidated "what did NOT work", verbatim-sourced

| Team | Rank | Did not work |
|---|---:|---|
| likable_jelly | 1 | Descent-phase features ("did not provide much improvement"); estimated-mass features likely redundant with energy rate; `ScaleByGroup` on distance built but unused **[S]** |
| tiny_rainbow | 2 | MLP, SAINT and **LightGBM** all got ~zero ensemble weight; paper says they "may be redundant … and thus pruned" **[S]** |
| brave_pillow | 4 | Best result (2,250-ish) **lost** through untracked experimentation; per-type model selection dropped 2,066 → 2,276 out of sample **[S]** |
| delightful_avocado | 5 | Bounded by missing **aircraft variant** — top SHAP features were all type proxies **[S]** |
| affectionate_bridge | 7 | Wikipedia-sourced MTOW for A321 and B767 were wrong and showed up as impossible predictions **[S]** |
| mindful_donkey | 10 | Remapping unseen ADEP/ADES in the submission set "does not appear to have improved the output" **[S]**; trajectory-only stage alone was 6,140 RMSE **[S]** |
| faithful_engine | 17 | Deliberately refused submission-set-matched filtering/reweighting as likely overfitting; T100 integration unfinished **[S]** |
| amazing_forest | 12 | Month/week/day/hour features "not very useful at all"; XGBoost gave no meaningful gain over ranger RF **[S]** |

### External data sources across the top 10 (consolidated) **[S]**

| Source | Used by |
|---|---|
| **OpenAP** (aircraft performance, masses, fuel flow, trajectory optimiser) | 1, 4, 10, 17 |
| `traffic` library | 1, 4, 8, 9, 10 |
| **ourairports.com** | 1, 5, 8 |
| Wikipedia (MTOW/OEW gap-filling) | 1, 2, 5 |
| **Iowa State Mesonet ASOS METAR archive** | 1, 17 |
| Meteostat API | 8 |
| FAA Aircraft Characteristics Database | 2, 8, 10 |
| EUROCONTROL Aircraft Performance Database | 2, 8 |
| Global Airport Database (partow.net) | 2, 6 |
| OpenFlights Airports Database | 2 |
| CADO airplane database (doi:10.57745/LLRJO0) | 6 |
| **Eurostat airport movement data 2010–2024** | 8 |
| **IATA monthly European passenger load factors** | 10 |
| EU airport passenger flow; UN fuel prices; scraped airline cabin configs | 17 |
| `airportsdata`, `country_timezones`, Kato `weather_parameters` | 10 |

**No team in the top 10 reported using a paid or restricted data source.** All external data was open. **[S]**

---

## 5. Secondary source: the independent review by team_amazing_forest (rank 12)

Conor McLaughlin, a working data scientist, wrote two public posts. **[S]**
- Build log: https://conormclaughlin.net/2024/11/building-a-machine-learning-model-to-predict-plane-take-off-weight-for-the-opensky-prc-data-challenge/
- Post-mortem review of the winners: https://conormclaughlin.net/2024/12/reviewing-the-winning-ml-models-from-the-opensky-prc-data-challenge-what-did-they-do-differently/

His own model: `ranger` random forest in R, 302 features, K-means clustering of aircraft types into 5 groups,
rare-category collapsing, 80/20 split, **raw kg target**, plus a **stacked feature**: "I performed a
quick-and-dirty linear regression using just aircraft_model and flown_distance to predict TOW for each
flight_id, and then fed those predictions into my primary model as a new feature!" Final 2,695 RMSE. **[S]**

His diagnosis of the gap between himself and the winners, in his own framing **[S]**: his submission lacked
(a) knowledge of trajectory-analysis libraries (OpenAP, Acropole), (b) arrival-airport weather features,
(c) ensembling or AutoML. He characterises the top three as "a tier behind" his own result.

**Caveat, stated:** he reports `team_faithful_engine` at RMSE 1,635, which contradicts the official leaderboard
(2,784.52, rank 17). His post-mortem also identifies AutoGluon-based AutoML as a top-team differentiator, an
attribution that rests on that unverified figure. **Treat his RMSE attributions as unreliable; his descriptions
of the top two teams' *methods* match their READMEs and are reliable.** **[S]/[X]**

---

## 6. Things I looked for and could NOT find — stated plainly

- **A ranking page giving separate public/private (submission_set vs final_submission_set) scores per team.**
  Only one RMSE column exists; the rankings page states final ranking uses `final_submission_set.csv`, and
  rank 1's README independently reports 1,561.63 on the final set, matching the table. So the table is the
  final ranking. **[S]**
- **The OpenSky Symposium results slides.** The site links
  `media/PRC_Data_Challenge_2024_Enrico_Spinielli.pdf`; it returns **404**. No mirror found.
- **Any write-up at all for rank 3 (team_gentle_elephant).**
- **A confirmed peer-reviewed paper from the rank-1 team.** See §1.
- **A JOAS paper from any top-10 team other than rank 2.** The "special issue" contains one research article.
- **Any team in the top 10 reporting a formal cross-validation scheme**, other than mindful_donkey's grouped
  10-fold-by-flight-id (rank 10) and H2O's internal 5-fold. Ranks 1, 2, 5, 6 all describe single
  train/validation splits. **[S]**
- **Any post-processing step (clipping, calibration, rule overlays) in the top 10.** I found **none**.
  Not "nobody used it" — nobody *documented* it.

---

## Applicable to resilient-hedgehog

Checkable items. Each is tagged **TRANSFERS**, **PARTIAL**, or **DOES NOT TRANSFER** with the reason, since the
2026 target is taxi-out **duration in seconds** at 10 named airports, not a mass.

### DOES NOT TRANSFER — do not import these

1. **`(y − EOW)/(MTOW − EOW)` target normalisation + `(MTOW−EOW)²` sample weights.** The single biggest
   contributor to rank 1's win **[S]**. It works because each aircraft type has *known, tabulated, hard physical
   bounds* on the target. **Taxi-out time has no per-group physical floor and ceiling of that kind.** There is a
   soft analogue (unimpeded taxi-out time per airport/stand/runway as a floor), but it is *estimated*, not
   tabulated, and using an estimated denominator introduces its own error. **Do not mechanically copy this.**
2. **Point-mass physics features (energy rate, equivalent mass from OpenAP).** There is no equivalent
   first-principles equation for taxi duration. The ground-movement analogue is *queueing*, not mechanics.
3. **Altitude-sliced (1,000 ft) trajectory features.** No altitude axis exists on the ground.
4. **The "predict mean(y) − y" residual target** (rank 6, helped "considerably" **[S]**). It was motivated by
   *aircraft-type* imbalance in a mass target. Tree models are invariant to this shift; any benefit came from
   interaction with XGBoost's base_score/regularisation. **[X]** Low expected value; not worth a slot.
5. **MTOW/OEW/aircraft-mass reference tables.** Relevant to 2024 only.

### TRANSFERS — directly, with a concrete check

6. **Seed averaging of one well-tuned model is worth ~3 %, and 10 models capture almost all of it.** Rank 1:
   1 seed 1,612.41 → 10 seeds 1,564.09 → 20 seeds 1,561.63 **[S]**. *Check:* train our best single config with
   10 seeds and average. Expect a small but near-free gain; **do not spend budget going past ~10**.
7. **A QP/NNLS-weighted stack over correlated GBDTs collapses onto one or two members.** Rank 2's five-model
   stack resolved to 0.865 CatBoost + 0.134 XGBoost; LightGBM, MLP and SAINT each got numerically zero weight,
   and removing them changed RMSE by 0.00 % **[S]**. A single tuned CatBoost was within 0.3 % of the stack **[S]**.
   *Check:* before investing in a 5-model blend, fit non-negative least-squares weights on a holdout and **print
   the weights**. If one member carries >0.8, spend the time on features instead. This is a decision rule, not a
   prediction.
8. **Fit blend weights on data the base models never saw, then refit base models on everything.** Rank 2's
   two-stage procedure (80 % train / 20 % weight-fitting, then retrain all members on 100 % with frozen
   weights) **[S]**, endorsed in review as the correct stacking form **[S]**.
9. **Monotone transforms of a feature are no-ops for tree models — and duplicated monotone copies actively
   dilute split selection and corrupt importance scores.** Alligier's A3/A4/A5 **[S]**. *Check:* audit our
   feature matrix for pairs related by a strictly monotone map (x and x², x and 1/x, x and x/median(x),
   seconds and minutes of the same quantity). Delete the duplicates. This is a pure-win cleanup.
10. **Scaling only helps trees when it differs *by group*.** Rank 1 scaled mass features per aircraft type
    **[S]**; rank 5 independently scaled every trajectory feature by its per-aircraft-type mean **[S]**.
    *For us:* the group is the **airport** (and plausibly runway configuration, stand area, carrier).
    Expressing a feature as a ratio to, or z-score within, its airport-and-hour cell is a real transformation,
    unlike a global rescale. **This is the strongest structural idea in the 2024 corpus that transfers intact.**
11. **Leave missing values missing when missingness may be informative; median-impute only when it is provably
    not.** Alligier A17 **[S]**. *For us:* missing trajectory/surface data very plausibly correlates with
    congestion or with a specific stand or a specific airport's feed. **Default to NaN-native GBDT handling**
    and only impute where we can argue missingness is independent of taxi-out duration.
12. **Group your CV folds by the entity that would otherwise leak.** Rank 10 folded by `flight_id` to stop
    cross-training within a flight **[S]** — the only documented leakage control in the top 10. *For us:* the
    analogous units are **date / airport-day** (shared congestion state) and possibly the aircraft registration
    turnaround chain. A random row split across an airport-day will leak the queue state.
13. **Report per-group error, not just the global metric.** Reviewer Mori: "Relying solely on a single overall
    RMSE might mask performance discrepancies" **[S]**; rank 7 observed error scaling with aircraft weight
    **[S]**; rank 2 published per-aircraft-type RMSE **[S]**. *For us:* maintain a **per-airport RMSE table** on
    every validation run. With 10 airports and a squared metric, the total is dominated by the worst airports.
14. **Publish a naive baseline number.** The MTOW baseline (32,324.13 RMSE, 26.10 % MAPE) **[S]** was added at a
    reviewer's request and makes every subsequent number interpretable. *For us:* per-airport (or
    airport × hour) median taxi-out. Record it.
15. **Freeze external data locally.** Rank 1 could not reproduce their winning score because the Mesonet METAR
    archive mutated within two weeks **[S]**. *Check:* any external feed we pull (weather, airport reference,
    schedules) gets snapshotted into the repo or a pinned artefact, with the fetch date.
16. **Set `deterministic=True` / pin seeds now, not later.** Rank 1 explicitly could not reproduce their
    submission under LightGBM defaults **[S]**.
17. **Version every experiment.** Rank 4 lost a ~2,250 result — better than their 2,276 submission — because
    "it was a messy day where we trained so many models … and lost track of it" **[S]**. This is the cheapest
    lesson in the corpus.
18. **Validate reference tables against the data before trusting them.** Rank 7's impossible predictions traced
    to wrong Wikipedia MTOWs for the A321 and B767 **[S]**. *For us:* any airport reference table (stand
    coordinates, runway assignments, taxiway distances) must be sanity-checked against observed data.
19. **Beware selecting the per-group best model on the same split used to score it.** Rank 4's per-aircraft-type
    "RMSE filtering" scored 2,066 in validation and 2,276 in the real test **[S]**. *Check:* if we build
    per-airport specialists, the choice of "use specialist vs use global" must be made on a split independent of
    the one used to measure it — or, better, blended with a learned weight rather than hard-selected.
20. **Encode a known confound into the feature instead of hoping the tree chains conditions for it.**
    Alligier's A16 **[S]** — greedy tree growth may never select the first of a required pair of splits.
    *For us:* if a feature is only interpretable relative to a context (e.g. queue length only meaningful
    relative to that airport's typical queue at that hour), build the ratio/difference directly.

### PARTIAL — the idea transfers, the implementation must be redesigned

21. **Per-group specialist models.** Three teams tried per-aircraft-type models (ranks 4, 6, 10). Evidence is
    **mixed**: rank 4 lost 10 % out of sample **[S]**; rank 10's per-type stage was 6,140 alone but still added
    1.3 % when stacked as a *feature* rather than used as a prediction **[S]**. *The transferable form is rank
    10's:* per-airport models feed their predictions (and their own per-airport validation error) **as features**
    into a global meta-model — not a hard switch on which model to trust.
22. **A physically/structurally constructed estimate fed in as a feature.** Rank 10's
    `est_tow = OEW + fuel + (payload capacity × monthly load factor)` **[S]**; rank 12's "quick-and-dirty
    linear regression prediction as a new feature" **[S]**. The 2026 analogue: **unimpeded taxi-out time**
    (per airport / stand-area / runway) as an input feature. **Safer than using it as a target denominator
    (item 1).**
23. **Weather matters through the *decision*, not the *conditions*.** Rank 1's weather features were
    thunderstorm/fog at the **arrival** airport near arrival time, as a tankering proxy **[S]**; rank 2's
    per-phase mean temperature contributed only +1.76 % **[S]**. *For taxi-out:* the analogous high-value signal
    is **conditions that change ground operations** — low visibility procedures, de-icing, snow, runway
    configuration changes, thunderstorms in the terminal area — not mean temperature. **[X] that this analogy
    holds; the 2024 sources support only the mass case.**
24. **Emit a family of features across an unknown hyperparameter and let the model choose.** Rank 1 computed
    METAR summaries at **several space–time radii** rather than picking one **[S]**. *For us:* congestion
    windows (±5/10/15/30/60 min), lookback horizons, and radii around the stand are exactly this situation.
25. **Coverage/completeness indicators as explicit features.** Rank 5 emitted booleans for whether each flight
    phase appears at all **[S]**; rank 8 used raw position-report counts **[S]**; Alligier's A16 **[S]** warns
    coverage boundaries masquerade as operational signal. *For us:* if we derive surface features from
    trajectories, emit the record count / time coverage / gap statistics alongside them.
26. **A no-trajectory baseline is worth building first.** Rank 7 got ~3,000 RMSE (≈ rank 20) from the flight
    list alone **[S]**; rank 10's Stage I (no trajectories) scored 2,719, only 1.3 % worse than his full staged
    model at 2,683 **[S]**. *Check:* establish what our schedule/airport/time features alone achieve before
    attributing gains to heavy trajectory processing.
27. **Data cleaning as a first-class stage, not a preamble.** The one difference the rank-1 and rank-2
    write-ups make starkest: rank 1 spent a documented stage on de-duplication, derivative filtering,
    no-interpolation policy, isolated-point removal, smoothing, **and recomputing wrong timestamps from the
    trajectory** **[S]**; rank 2 confirmed in review they did none **[S]**. *For us, the direct analogue:*
    the 2024 flight list's own `arrival_time` was demonstrably unreliable and rank 1 rebuilt it — **our
    off-block and take-off timestamps, which define the target and its endpoints, deserve the same audit.**

### One framing caution

28. **The 2024 result does not license "ensembling is the answer."** The winner used **one model family, one
    hyperparameter set, and 20 seeds** **[S]**; the most elaborate stack in the top 10 (5 families, QP weights)
    finished 42 % behind it and collapsed to one member **[S]**. The 2024 evidence says the margin came from
    **target formulation, data cleaning, and concept-aligned feature construction** — not from blending.
    **[X]** that the same ordering holds for a taxi-out target; the 2024 corpus cannot establish that.

---
*Compiled from: the dc2024 rankings and winners pages; the JOAS special issue (3 articles, all read in full);
9 of the top-10 GitHub READMEs plus team_faithful_engine; and two secondary blog posts. Rank 3 has no
published write-up. The official results slides are linked but return 404.*
