# Past aviation / travel-time competitions: what won, and does it transfer?

[S] = checked against the source this session, [X] = my inference or memory. Transfer = High/Med/Low; "OK" = fits our rules (no LB probing, open external data only, no side channels, no copied code).

## 1. NASA/DrivenData "Pushback to the Future" (2023), MAE in minutes, 10 US airports. The closest match
Source: https://drivendata.co/blog/airport-pushback-finalists , https://github.com/drivendataorg/nasa-airport-pushback
- 1st (Team CDS): 298 features from the history of ETD updates (latest value, plus values over the last 30 h). **Four model families ensembled**: V0 per airport predicting *pushback − ETD* (a residual target), V1 per airport predicting *absolute* minutes, V2 a re-parameterised V0, and one global model with airport as a feature. All CatBoost. [S]
- 3rd (Oracle 2): XGBoost per airport, plus an **XGB classifier that flags over- or under-estimation; the median over/under error is added to or subtracted from the regression output**. They also used an internal outlier detector with a 20–30 min error threshold. [S]
- 2nd and 4th: **one model per airport did better than one global model** (CatBoost or LightGBM). 5th: **dropped features that only helped the public LB and did not help the second arena**. [S]
- Transfer: **High**. Their residual-on-ETD setup is the same as ours (residual on AOBT_3). The new pieces for us are blending the absolute and residual targets, and blending per-airport models with the global one. OK.

## 2. DrivenData "Run-way Functions" / NASA Airport Configuration (2022), log loss
Source: https://drivendata.co/blog/airport-configuration-winners
- 1st: 120 CatBoost models (airport × lookahead), all features built from history. Chaining models did not help. [S]
- 2nd: an explicit "configuration inertia" prior. 4th: **upweighted the most recent months** to handle temporal drift, and used sin/cos for wind direction. [S]
- Transfer: **Low–Med**. The task is different. The reusable part is recency/target-period weighting for the 2025→2026 drift. OK.

## 3. PRC Data Challenge 2024 (take-off weight, RMSE)
Source: https://github.com/PRC-Data-Challenge-2024/team_likable_jelly , https://conormclaughlin.net/2024/12/reviewing-the-winning-ml-models-from-the-opensky-prc-data-challenge-what-did-they-do-differently/
- 1st (likable_jelly, ENAC): **normalised target (TOW−EOW)/(MTOW−EOW) with LightGBM sample weight = (MTOW−EOW)²**, so the loss still equals RMSE on the original scale. **Averaged 20 seeds.** Physics features (OpenAP mass), spline-smoothed trajectories, METAR. [S]
- 2nd: AutoGluon plus Acropole. 3rd: a CatBoost+LGBM+XGB+NN ensemble with ablations. [S]
- Transfer: **Med**. Seed bagging is free. Normalising by per-airport or per-runway scale with weight = scale² is an option for heteroscedastic airports. OK.

## 4. PRC Data Challenge 2025 (fuel, RMSE)
Source: https://prc-data-challenge-2025.netlify.app/outcome.html , https://github.com/PRC-Data-Challenge-2025/jubilant-vase
- 1st (resourceful-quiver, RMSE 200.8): a domain-knowledge mass estimator and feature engineering. The method sits in a PDF I did not read. The README shows a **train | rank | final split** that mirrors the leaderboard structure. [S, details partial]
- 2nd (jubilant-vase): CatBoost/LGBM/XGB with weights fitted by scipy (64/20/16), GroupKFold, automatic removal of corrupted records. [S]
- Transfer: **Med**. Domain physics won here. For us the analogue is surface geometry and queue physics, not more model capacity. OK.

## 5. Taxi-out literature
- Wang et al. 2021 (TR-C), "Aircraft taxi time prediction: feature importance". Source: https://eprints.whiterose.ac.uk/id/eprint/169934/ . A small subset reaches near-full accuracy: **taxi distance, total turns, average speed of the last 5 departing aircraft, and congestion counts** (NDepDep = departures taxiing when this one pushes back; **QDepDep = departures that take off while this one is taxiing**). [S]
- Lee, Malik & Jung 2016 (CLT, NASA). Source: https://ntrs.nasa.gov/api/citations/20180002162/downloads/20180002162.pdf . The key variables were concourse, spot, runway, departure fix and weight class. **Linear regression and RF gave the best RMSE**, and the paper says individual-flight accuracy is limited by operational randomness. [S]
- Ravizza et al. (ARN/ZRH). Source: https://nottingham-repository.worktribe.com/index.php/OutputFile/997640 . Layout-based variables (distance, turns) in a regression generalise well across airports. [S, from abstract]
- Dalmau et al. SIDs 2019 (EUROCONTROL/MUAC, NM ETOT error). Source: https://www.sesarju.eu/sites/default/files/documents/sid/2019/papers/SIDs_2019_paper_36.pdf . LightGBM cut NM ETOT error by about 30%. SHAP shows **NM planned taxi time is biased: high values are over-estimated and low values under-estimated**, and previous-leg delay matters. [S]
- Swatowska & Gabagnou, SIDs 2025 (EUROCONTROL IH, 6 A-CDM airports, GBDT). Gains over standard taxi times were only **1–6% at major airports**, versus 30–44% at regional ones. [S, from search snippet; PDF 404]
- Transfer: **High** for queue counts and geometry. Because every test row comes with its takeoff time and runway, the QDepDep-style counts and "recent proxy taxi time of the last k departures" can be computed from inputs in the test set. OK, because they use only input columns of other rows, never labels or ordering.

## 6. GE Flight Quest 1 (2012/13, gate and runway arrival RMSE)
Source: https://bpb-us-w2.wpmucdn.com/sites.udel.edu/dist/6/1855/files/2014/11/fq_bigdata_presentation-Gabor-Takacs-x6lc21.pdf (Takács)
- **Direct estimates (the airline's ETA, the flight plan) as base features. Then a 6-stage ridge → GBM → ridge … residual stack.** Most of the gain came in stage 2: the baseline-only RMSE was 93.8, and the first GBM brought it to 6.5. Training cleaning **removed extreme-delay flights**. [S]
- The 1st place (Gxav) was 40–45% better than the industry estimate. [S, headline only]
- Transfer: **Med**. It confirms the residual-on-estimate approach. A ridge on sparse stand×runway one-hots as stage 1, with GBM on its residual, could help sparse or unseen categories. OK.

## 7. ECML/PKDD 2015 taxi trip time (Porto)
Source: https://ceur-ws.org/Vol-1526/paper22.pdf (Hoch, winner)
- **Resampled training trips so their distribution matched the test snapshot process** (longer trips upweighted linearly). Used per-test-trip "expert" models trained on similar trajectories, falling back to a global model when there were too few neighbours. [S]
- Transfer: **Med**. Same logic for us: reweight 2025 so it looks like Jan+Jul 2026, using month, stand-mix and adversarial weights. Also a hierarchical fallback (stand → stand cluster → terminal) for sparse or unseen stands. OK.

## 8. Kaggle NYC Taxi Trip Duration / Taxi Fare
Source: https://queirozf.com/entries/kaggle-nyc-taxi-trips-competition-overview-and-results
- Routing distance from external OSRM and weather were the biggest gains. The top LB was inflated by leaked data; the clean 1st place was 0.362 RMSLE against 0.290 with leaks. Coordinate PCA and clusters, LGBM/XGB stacking. [S]
- Fare: drop impossible training rows and use distance to airports. [X]
- Transfer: **Med** for *routing distance* (OSM taxiway graph → shortest stand-to-runway path length). The leak lesson matches our rule. OK, because OSM is open and documented.

## 9. Flight-delay competitions in general [X]
- These were won by the official estimate plus history aggregates in GBDT. Nothing beyond the items above.

---

## Ranked actionable ideas for us

| # | Idea | Origin | Est. impact (LB RMSE) | Effort | Risk |
|---|---|---|---|---|---|
| 1 | **Explicit mixture for "BLOCK copied from SCHED" rows** (LIRF 8%, others). Classifier p = P(BLOCK==SCHED) on inputs. Prediction = p·(MVT−SCHED) + (1−p)·regular model, which is the RMSE-optimal mean of a bimodal target. Fit p calibrated per airport and month. | Pushback 3rd (classifier correction), generalised [X] | −1 to −4 s (concentrated in the heavy tail) | 4 h | Med: calibration drift in 2026 |
| 2 | **Test-time queue features from the given MVT stream**: number of takeoffs on the same runway or airport in [AOBT_3, MVT] (QDepDep), in [MVT−10/20 min, MVT], and the median (MVT−AOBT_3) of the previous k departures on the same runway. Skip any already built. | Wang 2021 [S] | −0.5 to −2 s | 3 h | Low. Compliant: uses inputs only, never ID order |
| 3 | **Geometry instead of identity for unseen stands**: shortest OSM taxiway path length and turn count from stand to the departure runway threshold, plus **stand-ID dropout** (mask ~10–15% of stand IDs while training) so trees learn the fallback. | Wang/Ravizza [S], ECML fallback [S], OSRM lesson [S] | −0.5 to −2 s (EDDF July) | 5–6 h | Med: OSM quality at EDDF T3 |
| 4 | **Seed bagging** (10–20 seeds per GBDT) | likable_jelly [S] | −0.2 to −0.8 s, and narrower CV/LB variance | 1 h + compute | Very low |
| 5 | **Blend residual-target and absolute-target models**, plus per-airport against global models, with stacking weights fit on forward folds | Pushback 1st/2nd/4th [S] | −0.3 to −1.5 s (mostly on rows where AOBT_3 is off) | 3–4 h | Low–Med |
| 6 | **Target-period reweighting**: upweight 2025 Jan/Jul and late-2025 months; adversarial-validation weights for 2025 vs 2026 inputs | ECML 1st [S], Runway 4th [S] | −0.2 to −1 s | 2 h | Med: the CV/LB gap issue already seen; pre-register |
| 7 | **NM planned taxi-time shrinkage features**: (ETOT−EOBT) or NM taxi time, its deviation from the stand×runway historical median, and turnaround or previous-leg delay if they can be derived | Dalmau 2019 [S] | −0.2 to −0.8 s | 2 h | Low |
| 8 | **Feature-adoption rule**: adopt only if it helps on ≥3 of the 4 folds, including forward. Drop features with fold-specific gains (the L5 case). | Pushback 5th [S] | Protects against +0.1–0.5 s LB regressions | 1 h | Very low |

Notes: Ridge→GBM stacking (Flight Quest) and per-airport scale normalisation with weight = scale² (likable_jelly) are worth a try only if time remains (Low impact [X]). Day-rollover (+24 h) errors cannot be fixed by any idea surveyed here. Only a mixture-style expected-value term would reduce their cost, and only if a legitimate input signal exists (for example an ADS-B pushback observed far from AOBT_3) [X].
