# Past-competition lessons where the win came from how the data was generated (DGP)
Scope: only the insight, whether it carries over to PRC taxi-out (RMSE, heavy tail), and whether it fits our rules. URLs were found by web search. Details marked "(recalled)" come from my memory of the forum writeups and were not re-read.

## Competition-by-competition

**Elo Merchant Category (RMSE, about 1.1% of targets at −33.22)** — https://github.com/bangdasun/kaggle/blob/master/elo-merchant-category-recommendation/README.md, https://towardsdatascience.com/elo-merchant-category-recommendation-a-case-study-33e84b8465c7/
- The −33.22 rows (= log2 of a near-zero quantity, an artifact of how the target was built) made up most of the RMSE: CV about 3.6 with them, about 1.55 without.
- Top solutions trained (i) a regressor on non-outlier rows only, (ii) a classifier for P(outlier), and (iii) a full-data regressor. They combined them as p·(−33.2) + (1−p)·reg_clean, or used the clean model where p was lowest. The 7th-place team overwrote the top-k highest-p rows with −33 (k tuned on CV).
- Hard snapping helped only when the classifier was very confident. The soft expectation was the robust choice.
- **Transfer: HIGH.** Our rollover rows (y ≈ +86 400 s) have exactly this structure, a known fixed offset. Compliant if p uses only allowed inputs. Tune k/thresholds on CV only, never on the LB.

**Google Brain Ventilator (MAE, discrete target)** — https://medium.com/data-science/winning-the-kaggle-google-brain-ventilator-pressure-prediction-2d4c90d831ec, https://github.com/GillesVandewiele/google-brain-ventilator
- The pressure takes only 950 evenly spaced values (a quantised sensor). Winners snapped predictions to the lattice and used median ensembles. A PID-controller-based post-process (reconstructing the simulator) gained a further 0.004.
- **Transfer: MED.** If y is quantised (whole seconds, minute-rounded BLOCK at some airports), snapping does not help RMSE: rounding a mean estimate is not optimal under squared loss. The transferable part is *identifying which rows were produced by which recording process* (minute-rounded rows are a separate population). Compliant only if the process is identified from non-blanked fields (MVT, SCHED, NM times), never from blanked-column seconds.

**IEEE-CIS Fraud (AUC)** — https://www.kaggle.com/competitions/ieee-fraud-detection/writeups/fraudsquad-1st-place-solution-part-2
- Rebuilt a hidden client ID as card1+addr1+floor(day−D1). It was used only as an aggregation key (about 40 group statistics) and dropped before training. Post-processing averaged predictions within each UID.
- **Transfer: MED.** Our analogue is a "rotation/tail entity" (registration or callsign + day) that links a departure to its inbound flight and prior turns, plus per-entity stats computed in train. Compliant as long as the key uses given fields, not ID ordering.

**Santander Customer Transaction (AUC)** — https://lovkush-a.github.io/python/data%20science/2020/07/18/santander3.html
- About 50% of test rows were synthetic (no unique values). Removing them before frequency-count features ("magic") lifted AUC by about 0.01. It also exposed the public/private split.
- **Transfer: LOW, and partly excluded [X].** Using test-set structure to infer the public/private split is LB-side-channel territory. The safe lesson is adversarial validation (train vs 2026 test, AUC 0.5 is good) to find shifted features.

**LANL Earthquake (MAE, train/test shift)** — https://medium.com/@ph_singer/1st-place-in-kaggle-lanl-earthquake-prediction-competition-15a1137c2457
- The winners subsampled training earthquakes so that feature distributions matched the test set (KS statistic), and added noise plus median removal to kill a drifting signal trend.
- **Transfer: MED-HIGH** for the CV→LB gap (item e) and EDDF T3. Reweight or select 2025 training rows so that marginals of airport×month×stand-type match Jan/Jul 2026 (inputs only, so compliant). Do not pick the subset by LB score [X].

**Allstate Claims Severity (MAE, heavy right tail)** — https://amitbera.dev/blog/allstate-casestudy
- Target log(loss+200) with a Fair/Huber objective. Heavy blending of XGB and NN.
- **Transfer: LOW under RMSE.** Log/robust losses estimate a median-like quantity and are biased low on a right tail. The lesson for us is the opposite: under RMSE keep an MSE objective (or re-add the mean correction after a log fit), because the tail mass belongs in the prediction.

**Zillow Prize (MAE on log error)** — https://blog.nycdatascience.com/blog/student-works/machine-learning/zillow-prize-competing-improve-zestimate
- Dropping or clipping |logerror| > about 0.4 in train helped *because the metric was MAE*. The private stage used external data.
- **Transfer: LOW.** Under RMSE, deleting outliers from training biases the mean. Clip only rows whose generation is impossible to predict, and add back their expected contribution as a constant (item 2 below).

**House Prices (RMSE on log)** — dropping the two GrLivArea > 4000 partial-sale rows (a documented data-entry artifact) plus a log target. **Transfer: MED.** Remove training rows that are provably mislabelled *and impossible in test*. Rollover rows are *possible* in test, so do not just remove them.

**Sberbank Housing (RMSLE)** — https://www.kaggle.com/competitions/sberbank-russian-housing-market/discussion/32993
- "Investment" prices clustered at exactly 1M/2M/3M RUB, which are placeholder or fake prices (recalled). Winners down-weighted or removed them, or modelled product_type separately.
- **Transfer: HIGH (conceptually).** This is our schedule-copied BLOCK: a *placeholder recording process* separate from the real event. Model it as its own mixture component.

**Predict Future Sales (RMSE)** (recalled) — the test set is the full shop×item grid, so winners built train as the same grid with zeros and clipped the target to [0,20] to match the clipped evaluation. **Transfer: MED.** Make train rows mirror how the test set was built (for example, how unmatched rows arise in 2026, whether cancelled or diverted flights are included).

**Mercari (RMSLE)** (recalled) — log1p target and simple sparse MLPs. The win came from the text pipeline, not the DGP. **Transfer: LOW.**

**TalkingData (AUC)** — https://github.com/flowlight0/talkingdata-adtracking-fraud-detection
- The top feature was next-click/prev-click time deltas within a (ip, app, device, os) entity, which reflect how click logs are generated.
- **Transfer: MED.** Time to the *next departure from the same stand or runway queue*, using takeoff times we are given. Compliant. Do not order by row ID.

**Mechanisms of Action (log-loss)** (recalled) — ctl_vehicle rows always have all-zero targets, so winners hard-set them to 0 and removed them from training. **Transfer: MED.** Where a deterministic rule is known (y = MVT − SCHED for copied rows), apply it *only weighted by its probability*, unless the rule's precondition can be observed in inputs.

**PetFinder 2019 (QWK)** (recalled) — OptimizedRounder thresholds. The 1st-place team was disqualified for embedding scraped test labels. **Transfer: LOW**, a cautionary case on rules [X].

**Playground Series (synthetic tabular)** (recalled) — adding the original dataset back and snapping to the original target's discrete values helped often. **Transfer: LOW-MED.** Only the "add back original-process data" idea transfers: 2025 ADS-B and NM data as extra supervision.

## General RMSE-with-rare-huge-errors principles
1. Under squared loss the optimum is E[y|x]. For a mixture: ŷ = Σ_k p_k(x)·μ_k(x) (hurdle form; https://towardsdatascience.com/two-stage-hurdle-models-predicting-zero-inflated-outcomes/). Medians, log targets and robust losses are wrong for the final layer.
2. Model each recording process (real, placeholder/copied, rollover) separately. One GBDT that sees all of them wastes splits explaining the tail and inflates variance in the bulk.
3. When p_k is near 0 and uninformative, the optimal adjustment is a **constant**: p̄·Δ (for example, p̄ = 0.1% rollover × 86 400 s gives +86 s, which is not negligible). Estimate p̄ per segment (airport × time from 00:00 UTC) and shrink it toward the global rate (empirical Bayes) because events are rare.
4. Hard snapping (Elo 7th) is only worth it if P(component) > about 0.5 at high precision. The MSE gain/loss is p·Δ² vs (1−p)·Δ², so verify on CV.
5. Blend weights per segment (matched/unmatched, airport): fit on out-of-fold predictions with a ridge penalty. Unconstrained per-segment weights overfit when few tail rows exist.
6. Check CV→LB transfer with adversarial validation, and give tail-heavy segments their own paired CI: one rollover row in a fold can swing RMSE by about 0.3 s.

## Ranked actionable list for PRC (all [S]=safe unless [X])
| # | Action | Impact (RMSE s) | Effort h | Risk |
|---|---|---|---|---|
|1|**[S] Copied-BLOCK mixture.** Train a classifier P(copied \| airport, airline, stand, SCHED fields, NM availability, hour) using train labels (y == MVT−SCHED within ±60 s). Predict p·(MVT−SCHED) + (1−p)·μ_real, where μ_real comes from the model trained on non-copied rows. Esp. LIRF (~8%). | 1–4 | 6 | Med: p must use only non-blanked inputs; **[X]** any p feature from the seconds/rounding of blanked off-block. |
|2|**[S] Rollover expected-value offset.** Do not drop rollover rows silently. Fit P(rollover) on allowed inputs (takeoff/SCHED minutes from 00:00 UTC/local, airport, overnight SCHED-vs-MVT date gap). Train the main model with rollover rows removed and add p·86 400 (or −86 400). If p is flat, use a shrunken per-segment constant. | 0.5–3 | 4 | Low-Med: high CV variance; pre-register and use paired CI. |
|3|**[S] Unmatched-row hurdle/shrinkage.** Split unmatched rows by cause (no NM record vs NM mismatch). Fit μ with an MSE objective and a separate tail-component classifier (y > 3600 s). Shrink per-segment predictions toward the segment mean by OOF-fitted ridge weight. | 1–3 | 6 | Med: few rows, so keep it simple. |
|4|**[S] Stop log/robust objectives in the final layer.** Check that every base model uses MSE or has a re-added mean correction (Allstate/Zillow lesson). | 0–1 | 1 | Low. |
|5|**[S] Adversarial validation + covariate reweighting** of 2025 train toward the 2026 input distribution (LANL), to tackle the CV→LB gap and EDDF T3. Hierarchical backoff for unseen stands: stand → OSM coordinate cluster/terminal → airport. | 0.5–2 | 5 | Low-Med; **[X]** never select weights by LB. |
|6|**[S] Entity/queue features (IEEE UID, TalkingData next-click).** Per tail/callsign-day rotation stats. Count of takeoffs on the same runway in ±10 min and gap to the next/previous takeoff from the same stand, built from given takeoff times. | 0.5–2 | 6 | Low; **[X]** no row-ID ordering. |
|7|**[S] Per-segment OOF blend weights** (matched/unmatched × airport, ridge-regularised, 3 folds). | 0.3–1 | 3 | Low-Med (overfit). |
|8|**[S] Training-set hygiene (House Prices/Sberbank).** Down-weight training rows whose label is provably a placeholder and *cannot* occur in test (for example, cancelled flights), but keep the generation types that do occur in test, via items 1–2. | 0.2–1 | 2 | Low. |

Excluded [X]: Santander-style inference of the public/private split from test structure, LB-tuned top-k snapping, any feature from ID order or second-level jitter of blanked columns, and PetFinder-style use of external labels.
