# Time-shifted tabular comps: what won, what transfers (for PRC 2026 taxi-out)

[S] = stated in a source I read (URL given). [X] = my inference or recollection, not checked against a primary source. Kaggle discussion pages would not render (they need JS), so most Kaggle claims come from search snippets or secondary reviews. Reviews used: Bojer & Meldgaard, "Kaggle forecasting competitions" (https://arxiv.org/pdf/2009.07701), and Jacobusse's Rossmann winner documentation (https://storage.googleapis.com/kaggle-forum-message-attachments/102102/3454/Rossmann_nr1_doc.pdf).

## Per competition

**Rossmann Store Sales (1st, Jacobusse)**: Rossmann winner PDF
- [S] Validated on one holdout: the last 6 weeks of train, matching the test horizon. He used it to select features and ensemble pairs from about 500×250 candidate pairs without overfitting. He ignored a public script that scored better on the public LB than it did on his holdout.
- [S] Added separate models trained only on May–Sep across all years, the months that cover the test period. Also added "month-ahead" models that skip the most recent month, used for the second test month.
- [S] Used a log target and a constant multiplier on predictions. 0.995 was optimal on both validation and public LB, but the theoretical 0.985 would have been better on private. Lesson: a bias correction tuned on CV or public can be wrong when the distribution shifts.
- Transfer **High**: season-matched sub-models (Jan and Jul), and derive bias corrections from theory rather than tuning them. Compliant.

**Corporación Favorita (1st)**: arxiv 2009.07701; search snippet
- [S] Used only recent data (1, 3 or 5 months), chosen by validation, even though years of history were available.
- [S, snippet] Picked the final model by mean and std over several validation sets plus the public score.
- [S] One model per forecast horizon.
- Transfer **Med**: recency weighting suits Jan 2026, which directly follows the end of the 2025 data. Selecting on mean and std is the key point. Compliant.

**M5 Accuracy (1st, YeonJun In)**: https://www.sciencedirect.com/science/article/pii/S0169207021001874
- [S] Equal-weight average of LightGBM models pooled per store, store-category and store-department. Each series is an average of about 6 models, mixing recursive and non-recursive variants.
- [S] Tuned on the last four 28-day windows, using both the mean and the std of the error.
- [S] Rescaled store-department level forecasts with multipliers to match the recent 28-day mean.
- Transfer **High** for the robust-selection rule and equal weights. **Med** for level re-anchoring: we can't use 2026 labels, but we can use 2026 ADS-B. Compliant.

**Recruit Restaurant**: arxiv 2009.07701; https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting/discussion/49174
- [S] Test contained Golden Week, which training covered only once. The winning trick was a domain mapping: treat a holiday as a Saturday, the day before as Friday, the day after as Monday.
- [S] For horizon n, features used only reservations made at least n days ahead, so train-time feature availability matched test time.
- Transfer **High** as an analogy: map unseen 2026 stands (EDDF T3) to known analogues by geometry (apron, pier, distance to runway) instead of by ID. Compliant.

**Web Traffic (1st, Suilin)**: arxiv 2009.07701
- [S] Reduced variance with 3 seeds, checkpoint averaging and SWA. Final results were insensitive to hyperparameters.
- Transfer **Med**: seed and bagging averaging stabilises how CV gains carry to the LB. Compliant.

**Walmart Store Sales (1st)**: arxiv 2009.07701
- [S] SVD denoising per department, holidays aligned across years, and simple models. Exogenous macro variables didn't help.
- Transfer **Low–Med**: align calendar effects (holidays, school breaks) between 2025 and 2026 by date.

**Elo Merchant (1st)**: https://github.com/bangdasun/kaggle/blob/master/elo-merchant-category-recommendation/README.md
- [S] Outliers were about 1% of rows but dominated RMSE. An outlier classifier gave about −0.01 on public LB, yet the winner's final submission did not use outlier post-processing. Many teams that relied on it were shaken down on private, plausibly through false positives.
- Transfer **High**: our rollover and BLOCK-copy tails are the same situation. Use soft, expected-value adjustments only. Compliant.

**Home Credit – Model Stability (2024)**: https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/discussion/482474
- [S] Teams gamed the stability metric by degrading early weeks. The hosts constant-filled WEEK_NUM and said they would review winners' code.
- [X] Top teams prioritised features whose relationship to the target is stable over time over features whose raw distribution is stable.
- Transfer **Med**, plus a rules warning: anything that keys on test time position or row order is a side channel. Avoid.

**Optiver Trading at the Close**: 9th place writeup https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/adam-9th-place-solution
- [S] Online retraining at private start and at +30 days. 1.5× sample weight on the latest 45 days.
- [S, snippet] 1st place (hyd) used about 300 features with CatBoost, GRU and Transformer.
- Transfer **Med**: recency weighting is usable. Online labels are not available to us.

**Optiver Realized Volatility (1st)**: https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970
- [S] Recovered the hidden time_id order (tick size, t-SNE) to build nearest-neighbour features and a proper time CV.
- Transfer **Low and forbidden** for us: this is exploiting a side channel.

**Ubiquant (1st)**: https://www.kaggle.com/competitions/ubiquant-market-prediction/writeups/k-i-y-1st-place-solution-our-betting-strategy
- [S, snippet] Used purged group time-series CV for feature selection and tuning, and KFold for the final fit. The title refers to a "betting strategy" across the 2 final picks.
- Transfer **Med**: choose time-based CV for feature decisions, and make the two final picks diverse.

**G-Research Crypto**: https://www.gresearch.com/news/wrapping-up-the-g-research-crypto-forecasting-competition/
- [S] The top 3 all used LightGBM. They said feature engineering beat model work. Evaluation ran live for 3 months.
- Transfer **Low–Med**.

**Santander Value / Two Sigma Rental**: https://github.com/kownse/Santander, https://github.com/plantsgo/Rental-Listing-Inquiries
- [S] Both were won by leaks: hidden time structure in Santander, image timestamps in Two Sigma.
- Transfer **forbidden**.

**atmaCup #12 and the Japanese community**: https://buildersbox.corp-sansan.com/entry/2021/12/18/110000, https://upura.hatenablog.com/entry/2019/10/27/211137
- [S, snippet] Used adversarial validation to pick test-like training rows as validation, and pseudo-labels for unseen classes. Frequency encoding was recommended for categories whose mix shifts over time.
- Transfer **High** for adversarial validation; compliant because it uses only test features, not labels. Pseudo-labelling is **Med**: allowed, but it adds noise.

**AMEX, Zillow, Instacart, Grupo Bimbo**
- [S] Grupo Bimbo used log1p and fell back to the parent group's mean for new products (https://medium.com/kaggle-blog/grupo-bimbo-inventory-demand-winners-interview-clustifier-alex-andrey-1e3b6cec8a20).
- [X] AMEX and Zillow winners mostly relied on feature engineering and ensembles. I found no time-drift insight beyond that.
- Transfer **Low**, except the hierarchical fallback for new categories, which is **High**.

## Why CV–LB disagreed in these comps, and what fixed it

1. **The validation period didn't match the test season or horizon.**
   - Seen in: Rossmann, Favorita.
   - Fix: validate on the same season and horizon as the test, and train sub-models on the matching season only.
2. **Selection noise across many candidate features.**
   - Seen in: M5, Favorita.
   - Fix: choose by mean − k·std across several folds, not by the best mean.
3. **Corrections tuned on validation: outlier hacks, constant multipliers.**
   - Seen in: Elo, Rossmann.
   - Fix: use soft or theory-derived corrections.
4. **Test-only categories that the validation never exercised.**
   - Seen in: Recruit, Bimbo.
   - Fix: fall back to group-level or analogue features, and build a validation fold that simulates unseen categories.

Our L5 result (CV −0.32 s, LB +0.12 s) sits within the noise of cause 2. A 0.3 s gain is below the LB's resolving power. [X]

## Ranked top-8 actions for us

1. **Pre-registered robust adoption rule.**
   - The rule: adopt a change only if every fold (jan, jul, jul_fwd, forward) improves and the airports improve in mostly the same direction, scored as mean − 1·std of fold deltas. Count only gains above about 1 s (M5, Favorita).
   - Directly targets the L5 failure.
   - Impact Med–High. Effort 2 h. Risk low.
2. **Make the model handle unseen stands.**
   - Replace stand ID with geometry: OSM coordinates, apron or pier cluster, taxi distance to the runway used, and terminal (Recruit analogy, Bimbo fallback).
   - Randomly mask stand ID and stand target-encodings for about 15% of training rows, so the model learns to predict without them.
   - Add a **GroupKFold-by-stand fold** (hold out 10% of stands entirely) to measure performance on unseen stands.
   - Impact High on EDDF July (12% of rows). Effort 6 h. Risk low–med.
3. **Adversarial validation, 2025 train vs 2026 test (features only).**
   - Train a classifier to tell 2025 rows from 2026 rows.
   - Features with high importance are drift candidates: drop them, or convert them to frequency or relative versions.
   - Use the classifier's probability p(test) to build a test-like validation subset and to weight training rows by importance (atmaCup #12).
   - Impact Med–High. Effort 4 h. Risk low. Compliant, since it uses no labels.
4. **Season- and recency-matched training.**
   - Sub-models trained on Jan and Jul 2025, or season-aware sample weights (Rossmann May–Sep models).
   - Upweight Q4 2025 for the Jan 2026 predictions (Favorita, Optiver 9th).
   - Blend with the full-year model.
   - Impact Med. Effort 3 h. Risk low.
5. **Per airport-month level re-anchoring from 2026 ADS-B.**
   - Estimate the 2025→2026 shift in ADS-B-derived taxi time per airport and month, on rows covered in both years. Apply it as a shrunk additive offset (the M5 multiplier analogue).
   - Use open data only and document it.
   - Impact Med. Effort 4 h. Risk med, because ADS-B coverage bias can differ by year. [X]
6. **Tail handling as an expected value, never a hard flag (Elo lesson).**
   - For rollover and BLOCK-copy prone groups, add p̂·Δ only where p̂ is calibrated across all folds.
   - Otherwise leave predictions at the conditional mean. Never post-process with thresholds.
   - Impact Med, since these rows dominate RMSE. Effort 3 h. Risk med.
7. **Variance-reducing ensembles.**
   - Equal or constrained blend weights (M5, Rossmann), multiple seeds and bagging (Web Traffic).
   - Fit any weights on forward folds only.
   - Impact Low–Med; mainly makes CV gains transfer to the LB. Effort 2 h. Risk low.
8. **Two final picks as a hedge (Ubiquant, Favorita).**
   - One pick with the best mean CV, one with the best mean − std plus the most conservative tail handling.
   - Use the public LB only as one input; no probing.
   - Impact Low–Med. Effort 0.5 h. Risk low.

Explicitly excluded under our rules: time-id or order recovery (Optiver RV), leak hunting (Santander, Two Sigma), anything keyed on test row position (Home Credit exploit).
