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

## S1, S2, S5, S6: environment, data, tokens, benchmark

* **S1.** `torch==2.9.1+cpu` (reports `2.9.1+cpu`, CUDA unavailable, no nvidia-*/triton packages);
  `requirements-nn.txt`.
* **S2.** 12 `training_2025-*.parquet` + `ranking.parquet` + `submitting.parquet` via
  `prc.io_minio.download_files(bucket="prc-2026-datasets", dest=data/raw)`; read-only, nothing uploaded.
* **S5 tokens** (`nbr_tokens.py`, 50 s, peak 2,465 MB): 2,085,047 departures and 2,082,750 arrivals in 120
  (airport, source-file) groups. Valid departure tokens per query: mean 45.2, median 47, 14.6 % of queries fill all
  64 slots; arrival tokens: mean 37.7, median 42, 36.7 % fill all 48. Window coverage < 1 on 0.03 % of rows
  (`tokens_stats.json`). Per-airport shares of tokens on the same runway / stand / stand prefix:
  `token_share_stats.json` (departures on the same prefix: 14–43 % by airport; same stand ≈ 0, as expected).
* **S5 query design, forward fold** (`nbr_query.py`, 110 s, peak 6,348 MB): train months 202501–202510
  (1,737,573 matched rows; inner holdout 202510, 183,761 rows; fit rows 1,553,812), evaluation 202511–202512
  (325,004 matched of 328,009 labelled rows). `cols_a` has 135 numerics (incl. the fold's `osm_stand_rwy_km`);
  the Preprocessor adds 60 NaN indicators and capped one-hots of the 10 categoricals: **input width 585** (m4's
  logged width is 585–586; the plan's "125" was an estimate). Model: 302,593 parameters.
* **S6 benchmark** (200 steps, ctx, K 64/48, d 64, batch 1024, gather prefetched in one worker thread):
  0.240 s/step, 1,518 steps/epoch, holdout pass 23 s → **6.45 min/epoch estimated ≤ 15 min: no fallback**
  (full K and d, no subsampling). Measured in S7: 345–386 s per ctx epoch (gather ≈ 60 s of it, overlapped),
  27–29 s per noctx epoch.

## S7: screen — forward fold, seed 1

| run | epochs run | best epoch | best holdout RMSE (Oct, matched, s) | wall | peak own-cgroup RSS |
|---|---|---|---|---|---|
| ctx | 10 (early stop) | 8 | 213.62 | 3,690 s | 3,087 MB |
| noctx | 10 (early stop) | 8 | 219.31 | 302 s | 1,923 MB |

Holdout RMSE per epoch — ctx: 229.9, 225.3, 219.1, 224.2, 217.4, 218.9, 215.7, **213.6**, 215.1, 215.5;
noctx: 231.9, 230.9, 223.6, 228.2, 220.9, 222.9, 220.5, **219.3**, 222.6, 219.6. Checkpoints (light copies,
1.25 MB, no optimiser state) were pushed at epochs 2, 4, 6, 8; none was needed for a resume.

Forward fold, all 328,009 labelled rows (unmatched rows are a2o in every arm), after the LOBT projection and floor
(`results/s7_forward_s1.json`):

| arm | RMSE all rows | RMSE matched rows |
|---|---|---|
| a2o (regenerated OOF) | 213.801 | 192.769 |
| NN_ctx alone (descriptive) | 222.063 | 201.975 |
| NN_noctx alone (descriptive) | 226.455 | 206.839 |
| A of C1: 0.75·a2o + 0.25·NN_noctx | 213.318 | 192.227 |
| B of C1/C2: 0.75·a2o + 0.25·NN_ctx | **212.246** | 191.026 |

| comparison | full data: diff [95 % CI] | worst 200 dropped: diff [95 % CI] | `verdict` |
|---|---|---|---|
| **C1** (ctx vs noctx, both blended) | −1.072 [−1.234, −0.930] | −1.099 [−1.236, −0.972] | (True, "adopt (-1.07 s, still -1.10 s at 200 rows dropped)") |
| **C2** (blend vs a2o) | −1.556 [−1.902, −1.249] | −1.455 [−1.639, −1.259] | (True, "adopt (-1.56 s, still -1.46 s at 200 rows dropped)") |

n_boot = 1000, row bootstrap (`prc.compare.robustness_curve`). At 20 and 2,000 rows dropped C1 is −1.15 and −0.94,
C2 −1.65 and −1.42 s. **Per airport** (blend ctx − blend noctx; blend ctx − a2o), every airport improves on both:
EDDF −1.85/−1.78, EDDM −1.22/−3.70, EGLL −1.09/−1.42, EHAM −0.40/−1.17, LEBL −0.46/−1.05, LEMD −0.18/−1.54,
LFPG −1.22/−2.13, LIRF −0.85/−0.08, LSZH −1.17/−0.91, LTFM −2.21/−2.74.

**Futility rule: not triggered** (C1 point −1.07 s < −0.10 s; CI wholly below 0). These are screen verdicts on one
fold and one seed; the pre-registered decision is taken on pooled jan + jul + forward with seeds 1 and 2 (S8),
which has not been started.

Reading (descriptive): the network alone is ~9 s worse than a2o on matched rows, as m4 is, but its context
version is ~5 s better than its no-context twin, and that difference survives blending at 25 % (−1.07 s) and the
tail drop. About a third of C2 (−0.48 s, 213.80 → 213.32) is what any NN member adds to a2o; the rest is context.
