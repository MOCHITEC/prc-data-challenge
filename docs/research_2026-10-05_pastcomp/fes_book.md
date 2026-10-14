# Kuhn & Johnson, *Feature Engineering and Selection*: what applies to the PRC 2026 taxi-out model

Sources: feat.engineering chapters 3 and 5–12, read 2026-10-05. Everything below is paraphrased. The code was read but not changed.

## Per chapter

**Ch 3: modelling process and resampling**
- *Idea:* every selection step has to sit inside the resampling. Choosing the best of many configurations on the same resamples gives an optimistic estimate ("optimization bias"). The book's fix is nested or external resampling. Compare models on identical splits with paired, hierarchical inference.
- *We do:* month-blocked folds (`train.py: run_cv`, `cv.py`). We use the same folds for every model, a paired bootstrap that resamples months rather than rows, and a drop-K robustness curve (`compare.py: month_bootstrap_rmse_diff`, `robustness_curve`, `verdict`). We also pre-register.
- *Gap:* the folds have no "external" layer. Dozens of adoption decisions were made on {jan, jul, jul_fwd, forward}, so the chain of decisions as a whole is tuned to those folds. That fits the L5 case (CV −0.32 s, LB +0.12 s). We have no untouched lockbox, and we don't adjust for the number of comparisons.

**Ch 5: categorical encoding**
- *Idea 1:* at prediction time, novel levels need a planned path: an "other" slot, or a fallback to a coarser grouping.
- *Idea 2:* effect (target) encodings should be shrunk with partial pooling toward a parent group, and estimated inside resampling.
- *Idea 3:* hashing gives collisions that mean nothing. It suits huge cardinality, not about 300 stands per airport.
- *We do:* raw `STAND_mvt` as a LightGBM categorical (`features.py: CATEGORICALS`, with deterministic Enum codes in `train.py: _matrix`). We use a hard-cutoff P10/median reference per stand×runway with `min_count=30`, and a coalesce fallback stand→airport×runway×hour→airport (`reference_table`, `attach_encodings`: `ref_p10_fb`, `ref_missing`). Frequency encodings are built over train plus target (`frequency_features`). There is an unseen-stand simulation (`sim_unseen_stands`, `unseen_stand_mask`).
- *Gap:* a new stand (EDDF T3, EDDM 103–108) gets a categorical code that never appeared in training. LightGBM then sends it down whichever branch holds "other categories", which is effectively arbitrary. The fallback goes straight to airport level and skips the stand's **physical neighbourhood**. The cutoff is hard (n=29 means nothing, n=30 means a noisy P10) instead of shrinkage.

**Ch 6: numeric predictors**
- *Idea:* replace an identifier with continuous descriptors (coordinates, distances, distance to a centroid) so the model can interpolate to unseen units. Avoid binning.
- *We do:* `add_signed_log` (features.py) and `osm_stand_rwy_km` (`osm.py: add_stand_runway_distance`, `train.py: osm_distance_features`).
- *Gap:* only one scalar describes the stand. We don't give stand x/y in a runway-aligned airport frame, or a distance to the centroid of the apron or terminal cluster.

**Ch 7: interactions**
- *Idea:* trees find interactions, but approximate smooth ones poorly. Use heredity (only interact strong main effects) and control false discoveries when screening many candidates.
- *We do:* stand×runway reference tables and share (`features.py: candidate_features`, `stand_rwy_share`), optional per-airport specialists (P11).
- *Gap:* no explicit geometry × runway-in-use term, such as the along-runway signed offset between stand and threshold. A term like that would carry over to new stands.

**Ch 8: missing data**
- *Idea:* first diagnose *why* a value is missing (structural, random, or informative), then encode that as its own level or indicator. Fit imputation and encodings inside resampling, and make the training missingness rate match the scoring rate.
- *We do:* null indicators (`features.py`: `aobt3_null`, `lobt_null`, `eobt1_null`, `stand_null`, `runway_null`). Unmatched rows have their own model B, with a SCHED reparameterisation for LIRF (`unmatched_sched_reparam`). There is an out-of-fold meta prior (`meta_prior_features`) and simulated day-level missingness for OPDI (`day_null_mask`, `opdi_target_null_day_share`).
- *Gap:* "unmatched" is a single bucket, not a taxonomy. Possible causes include no flight plan, an ID or callsign mismatch, GA or positioning flights, and BLOCK copied from SCHED. ADS-B ground coverage may differ between 2025 and 2026 by airport and day, and we haven't measured whether the training missingness rate matches the 2026 rate.

**Ch 9: profile data**
- *Idea:* resample by the independent unit, not by the row. Flights on the same airport-day share weather and disruptions. Turn within-unit profiles (ADS-B tracks) into features by smoothing and differencing.
- *We do:* the month bootstrap in `compare.py` respects this, at a coarse level.
- *Gap:* only about 12–14 months, so the month CI is wide and has few degrees of freedom. An airport×day block bootstrap would be a better middle ground, since day-rollover days cluster.

**Ch 10–12: feature selection, selection bias, overfitting**
- *Idea 1:* selecting outside the resampling gives optimistic error. Ambroise & McLachlan showed near-zero error on scrambled labels.
- *Idea 2:* benchmark a selected set against random or noise sets of the same size.
- *Idea 3:* track the gap between internal and external estimates. When internal keeps improving and external stays flat, that is the overfitting signature.
- *We do:* hashed noise columns as a column-count control (`noise_features`, `fe_noise_cols`), and an adversarial module (`adversarial.py`).
- *Gap:* no running "internal vs external" ledger. The LB is the only external signal, and the rules forbid probing it, so we need our own external layer.

## Ranked top-8 actions

| # | Action | Impact | Effort | Risk |
|---|---|---|---|---|
| 1 | **Lockbox and decision ledger.** Freeze one fold no further experiment may read: a forward-style lockbox of 2025 Dec, or 20% of airport-days chosen by CRC32. Re-score the current shipped stack against the last 3–4 accepted levers on it once only. Log every comparison ever run, apply a Holm or FDR correction to the remaining adoptions, and revert any lever whose lockbox delta has the wrong sign. | High (stops the L5-type regressions; may recover 0.1–0.5 s) | 4 h | Low (costs some training data) |
| 2 | **Hierarchical stand fallback using physical neighbours.** Cluster OSM stand positions into piers or areas (k-NN or a terminal polygon). Add `stand_area`, and build the reference table stand→area→airport×runway with empirical-Bayes shrinkage (n/(n+k) weighting) instead of `min_count=30`. New EDDF T3 and EDDM 103–108 stands then inherit area statistics. | High on EDDF July (12% of rows) | 6 h | Medium (needs OSM coverage of T3; check EDDF/EDDM stand refs match) |
| 3 | **Stand dropout in training.** For a random 5–10% of training rows, set `STAND_mvt` to "__NEW__" and null all stand-keyed encodings, so the model learns a coherent path for unknown stands that relies on area, coordinates and distance. Route every 2026 stand unseen in 2025 to the same token, and validate with the existing `sim_unseen_stands`. | High for new stands; neutral elsewhere if the rate is small | 3 h | Low |
| 4 | **Continuous stand geometry.** Stand x/y in a runway-aligned metric frame, signed along- and cross-runway offsets to the departure threshold, and distance to the area centroid. This is the Ch 6/7 answer to novel levels plus the smooth interaction. | Medium–high (new stands) | 4 h | Low (OSM is open and documented) |
| 5 | **Taxonomy of unmatched reasons.** Classify unmatched rows (callsign or ID near-miss, no flight plan, GA or military by operator/type, BLOCK==SCHED) into a categorical `unmatched_reason` for model B. Per-reason baselines may beat one pooled model, given unmatched rows are 24–39% of MSE. | High (that lane dominates MSE) | 5 h | Medium (category shares must be stable 2025→2026; check via adversarial.py) |
| 6 | **Match ADS-B missingness to the test rate.** Measure ADS-B ground coverage per airport×month in 2025 vs 2026. In training, null ADS-B features on a matching share of airport-days (as `day_null_mask` does for OPDI), and add a `adsb_day_coverage` feature so the correction knows when absence is structural. | Medium | 3 h | Low |
| 7 | **Airport×day block bootstrap in `compare.py`**, alongside the month bootstrap: more resolution than 12 months while keeping disruption-day correlation. Require the sign to agree across folds and in ≥7/10 airports. | Medium (fewer false adoptions) | 2 h | Low |
| 8 | **Random-subset benchmark for feature families.** Before adopting a family of k columns, compare it with 5 sets of k noise columns, or k shuffled copies of its own columns. Adopt only if it beats all of them on the lockbox-free folds. | Low–medium | 2 h | Low |

**Order:** 1 and 7 first, because they gate everything else. Then 3, 2 and 4 together as the new-stand package, then 5 and 6. Pre-register each one and score it once on the lockbox before the 2026-10-11 deadline.
