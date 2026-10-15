# NbrNet — results log (S1–S7)

Pre-registration: [`PREREG.md`](PREREG.md) (commit `8a95f3e`, before any fit). This file records what was run,
the numbers, and every deviation from the pre-registration with its reason.

## Deviations from PREREG.md

| # | what | why | effect on the comparisons |
|---|---|---|---|
| D1 | OSM `parking_<ICAO>.json` caches were filled from Overpass mirrors (`tools/fetch_osm_mirror.py`, same query text): EDDF and LEBL from overpass-api.de; EDDM, EGLL, EHAM, LFPG, LIRF, LSZH from maps.mail.ru; LEMD, LTFM from overpass.private.coffee | overpass-api.de reset most connections on 2026-10-15; the first pipeline attempt sat in `prc.osm`'s retry loop and was killed after 13 min and restarted (the feature cache was reused) | today's OSM differs from the one used for the logged records, so `osm_stand_rwy_km` (and hence a2o) can differ slightly from them; all NbrNet arms use the same regenerated a2o |
| D2 | the query `Preprocessor` is seeded with the recipe's `cfg.seed` (42), as m4 does, not the NN seed | one fold design shared by ctx, noctx and every NN seed | none (identical for both arms) |

## S3: a2o baseline regenerated

`python -m prc.pipeline --config configs/main_flseq_osm_nolirf_l2_opt_l6_cmix.yaml --skip-predict --fold-cache
data/nbrnet/fold_cache` (guarded, 7,739 s wall, peak own-cgroup total_rss 11,857 MB; first attempt killed, D1).
Experiment id `main_flseq_osm_nolirf_l2_opt_l6_cmix-1e01a07623`; record in `a2o_oof/cv_record.json` (not appended
to `experiments/experiments.jsonl`); fold-cache OOF parquets committed under `a2o_oof/<exp_id>/`.

| fold | n | RMSE | matched | unmatched |
|---|---|---|---|---|
| jan | 153,706 | 339.64 | 196.63 | 2,872.91 |
| jul | 190,713 | 292.04 | 235.61 | 1,223.78 |
| jul_fwd (not pooled) | 190,713 | 303.52 | 243.68 | 1,282.61 |
| forward | 328,009 | 213.80 | 192.77 | 985.18 |
| **pooled jan+jul+forward** | 672,428 | **269.92** | | |

Sanity: the plan expected ≈ 270.4–270.7 from `l2_opt_l6` (270.467) and `l2_opt_cmix` (270.702). No record of
this exact recipe (L6 + cmix together) existed; the two levers were each measured alone (cmix −0.49 s and L6 on
the same base), so a combined 269.92 is about what additive effects give (270.467 − ~0.5). Per-fold numbers are
within 0.8 s of both logged records (jan 340.4, jul 292.7–293.6, forward 214.0–214.2). Accepted as the baseline.
