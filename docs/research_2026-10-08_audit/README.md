# Audit of rejected levers: was the validation flawed? (2026-10-08)

Most rejections hold up. The forward fold and the unseen-flight-number share are the evidence closest to 2026, and they point the same way as each verdict.

| Lever | Audit | Next step |
|---|---|---|
| G-model, carrier echo, stamp (all / LIRF), cmix LEBL, ucday, day-demean, stage-2 corrector | Sound | none |
| Integration rejects: neighbour ADS-B residual, coverage-matched ADS-B, stamp-rate features | Sound | none |
| Labelled-row filter suspicion | No flaw. feat_train has 0 unlabelled rows, and feat_rank is the same departures-only population. | none |
| **Live-level floor** | **Flaw.** The 2025 safety test moved a different group (level 1.5–2.5 ks at EGLL/LFPG). The motivating 2026 rows are 130 non-LIRF unmatched rows with level > 3 ks on the EHAM/LFPG snow days of Jan 3–7, and 2025 has only 16 such rows. | No honest offline test exists. The swing is about −3.5 s if truth is near the level, +1.7 s if the model is right. An LB isolation submission would be needed, which is a probing decision for the user. |
| **Encoding cross-fit** (integration) | **Decided on too few rows.** It was tested on the forward fold only; the point estimate was favourable (about −0.6 s) but the CI spans 0. | Re-test with an all-fold pooled leave-month-out cross-fit. Expected −0.3 to +0.2 s, prior leaning null. |
| **Month scope / union fix** | Superseded by the integration branch's C10/C13. C13 was approximate: a2o ran out of memory and pao stood in. | Do an exact per-member re-predict, a2o included. Expected −0.1 to −0.25 s. |
| LOBT expert (integration) | Sound but set aside: about +12 MSE beyond the top-50 rows. | Overlay only. Expected about −0.1 s. |
| stand_dropout | Passed on unseen stands, −6.5 s [−7.5, −5.5]. | All-fold run on production members. Expected about −0.04 s, low priority. |

**Latent mismatch.** `stand_rwy_share` and `congestion_vs_usual_30m` are counted over the whole frame. Neither is in production today. Three `DD_SRC` columns were absent in the dd run.
