# Final phase: status of this thread (外部知見, branch claude/peaceful-babbage-md9l2x), 2026-10-13

Follows `origin/claude/model-b-and-arrival-features:docs/final_phase_2026-10-11/NO_FINAL_UPLOAD.md`.

## Uploads
- This thread has never uploaded anything to the team bucket: no `_vNN.parquet` and no `_final.parquet`.
- Submissions v40–v47 were made by the integration thread (データ分析). This thread only handed over member deltas and recipes.
- The `resilient-hedgehog_final.parquet` of 2026-10-09 was not made by this thread.
- This thread will not upload anything.

## Data access
- This container has no OSN credentials. The `AWS_*` variables present are rejected by s3.opensky-network.org with InvalidAccessKeyId.
- So `final_ranking.parquet` (Jan, Feb, Jun, Jul 2026) cannot be read here, and **no final candidate file was built in this thread.**

## Levers from this branch for the 4-month final

| Lever | In final v3 (laughing-cray 71a7d27)? | Recipe | Evidence | Expected |
|---|---|---|---|---|
| L6 gate retention | yes (members `*_l6`) | `fe_retention: true` | docs/research_2026-10-05_measure/l6_plan.md | done |
| LIRF cmix | yes (a2o `l2_opt_l6_cmix`) | `copy_mix_airports: [LIRF]` | copy_plan.md | done |
| NM-blind w = 0.25 | yes (lane B) | `nmblind_weight: 0.25` | docs/research_2026-10-06_nmblind | done |
| ucmix, LIRF unmatched | yes (lane B) | `ucmix_airports: [LIRF]` | docs/research_2026-10-07_ucmix | done |
| **cmix LTFM extension** | **no** | a2o with `copy_mix_airports: [LIRF, LTFM]` (configs/main_flseq_osm_nolirf_l2_opt_cmix_ltfm.yaml) on top of L6, 3 seeds | copy_plan.md, LTFM result: −0.25 s whole frame [−0.39, −0.12]; LTFM matched −2.26 [−3.25, −1.13]; all folds improve | about −0.03 to −0.1 s after the a2o weight |
| **Blend inner split, non-LIRF matched rows** | **no** | on non-LIRF matched rows, with c2: 0.4·(0.2·a2o + 0.8·pao) + 0.2·(p2o + c2 + m4). Simplest form: change only the inner a2o:pao split from 0.5:0.5 to 0.2:0.8 on non-LIRF matched rows; LIRF unchanged | docs/research_2026-10-08_blend: inner split alone −0.12 s (no c2), all folds; full non-LIRF reweight −0.22 [−0.30, −0.14] | −0.1 to −0.2 s. Re-measure with the c2 OOF first. |

- Both open levers act before the ADS-B layer.
- Per the team instruction, a candidate built on a changed base must rebuild the ADS-B residual and appear layers on that same base.

## Not for the final
- **Rejected after validation:** G-model, carrier echo, stamp (all and LIRF), cmix LEBL, ucday, day-demean, live-level floor.
- The restricted floor was never submitted.
- See the docs/research_2026-10-0*/ verdicts and docs/research_2026-10-08_audit.
