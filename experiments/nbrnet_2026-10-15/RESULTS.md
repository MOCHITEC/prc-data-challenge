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

## Tests

`.venv/bin/python -m pytest -q experiments/nbrnet_2026-10-15/tests`: 8 passed (synthetic data, 46 s): R1 (other
departures' block/taxi times shifted or nulled → tables, indices and every gathered token bit-identical; the
arrival control does change), R3 (no query in its own index), R6/R7 (same airport and source file, inside
[t0−60, t0+20] min, nearest-first, equal to a brute-force k-nearest), R5 (shuffling evaluation labels leaves every
training-side array and the target statistics unchanged), noctx = all-masked ctx and independent of tokens, LOBT
projection = `train._lobt_bound`, and an end-to-end ctx/noctx fit whose resumed re-run reproduces the predictions.

## S8 (not started) — estimate

Measured: ctx ≈ 6.0 min/epoch on 1.55 M fit rows, noctx ≈ 0.5 min/epoch; both stopped at epoch 10 here. The jan
and jul folds fit on ~10 months (inner holdout = Dec), ≈ 1.11× the forward fit rows → ≈ 6.7 min/epoch. With 10–12
epochs per run: jan + jul seed 1 ≈ 2 × (67–80) min ctx + 2 × ~6 min noctx; seed 2 on jan, jul, forward ≈ 3 × (60–80)
min ctx + 3 × ~6 min noctx; plus fold preparation (~2 min per fold, jan and jul) and evaluation (~1 min).
**Total ≈ 5.5–7 h sequential** (one job at a time, peak ≈ 3.3 GB for ctx, 6.3 GB for a fold preparation).

## S8: jan + jul at seed 1, all three folds at seed 2

Fold designs (`nbr_query.py --folds jan jul`, 150 s, peak 6,276 MB; `fold_jan.json`, `fold_jul.json`): jan trains on
202502–202512 (1,910,306 matched rows, inner holdout 202512, 1,746,278 fit rows) and scores 152,271 matched of
153,706 labelled rows; jul trains on 202501–06 + 202508–12 (1,875,802; holdout 202512; 1,711,774 fit rows) and
scores 186,775 of 190,713. Widths 587 / 586. Run by `s8_driver.sh` in the pre-registered order, one job at a time
under the 13,000 MB guard (never triggered), each run's predictions and record committed after it ended.
S8 wall clock: 10:33:56Z → 16:11:42Z (5 h 38 min).

| run | fit rows | epochs run | best epoch | best holdout RMSE | mean epoch s | wall s | peak MB | matched RMSE on fold |
|---|---|---|---|---|---|---|---|---|
| forward s1 ctx (S7) | 1,553,812 | 10 | 8 | 213.62 | 363 | 3,690 | 3,087 | 204.36 |
| forward s1 noctx (S7) | 1,553,812 | 10 | 8 | 219.31 | 28 | 302 | 1,923 | 209.55 |
| jan s1 ctx | 1,746,278 | 8 | 6 | 209.69 | 410 | 3,321 | 2,775 | 215.89 |
| jan s1 noctx | 1,746,278 | 8 | 6 | 214.12 | 32 | 275 | 1,943 | 222.11 |
| jul s1 ctx | 1,711,774 | 6 | 4 | 210.41 | 415 | 2,532 | 2,754 | 263.68 |
| jul s1 noctx | 1,711,774 | 9 | 7 | 212.61 | 32 | 311 | 1,949 | 266.27 |
| jan s2 ctx | 1,746,278 | 12 (cap) | 10 | 206.78 | 429 | 5,193 | 2,834 | 215.65 |
| jan s2 noctx | 1,746,278 | 9 | 7 | 211.35 | 33 | 321 | 1,933 | 219.70 |
| jul s2 ctx | 1,711,774 | 10 | 8 | 207.22 | 421 | 4,260 | 2,796 | 261.59 |
| jul s2 noctx | 1,711,774 | 4 | 2 | 215.70 | 32 | 141 | 1,928 | 273.20 |
| forward s2 ctx | 1,553,812 | 9 | 7 | 216.28 | 379 | 3,484 | 2,850 | 207.42 |
| forward s2 noctx | 1,553,812 | 10 | 8 | 221.79 | 29 | 312 | 1,934 | 212.57 |

All 12 runs: 24,141 s of training. No run resumed from a checkpoint. `jul s2 noctx` stopped by the pre-registered
patience rule after epoch 4 (best epoch 2) — a weak noctx run that makes its fold's C1 larger for seed 2 (see the
per-seed table); it is kept, as the rule says.

## S9: pooled verdicts — jan + jul + forward, NN mean of seeds 1 and 2

`nbr_eval.py --folds jan jul forward --seeds 1 2 --work-dir data/work --n-boot 1000`
(`results/s9_pooled_s12.json`, summary `results/s9_pooled_s12.summary.txt`). 672,428 labelled rows (664,050
matched); unmatched rows are a2o in every arm (their difference is exactly 0). The NN prediction per row is the mean
of the two seeds' predictions in seconds, then blended, projected onto the LOBT window and floored.

| arm | pooled RMSE | matched rows | jan | jul | forward |
|---|---|---|---|---|---|
| a2o | 269.919 | 206.558 | 339.642 | 292.044 | 213.801 |
| NN_ctx alone (descriptive) | 275.917 | 214.433 | 342.968 | 300.307 | 220.258 |
| NN_noctx alone (descriptive) | 280.220 | 220.010 | 345.993 | 306.455 | 224.217 |
| A of C1: 0.75·a2o + 0.25·NN_noctx | 269.691 | 206.256 | 339.354 | 292.335 | 213.196 |
| **B: 0.75·a2o + 0.25·NN_ctx** | **268.664** | 204.894 | 338.664 | 290.887 | 212.198 |

### Pre-registered decisions (`prc.compare.verdict(robustness_curve(y, A, B, n_boot=1000))`)

| | full data: diff [95 % CI] | worst 200 dropped: diff [95 % CI] | verdict tuple |
|---|---|---|---|
| **C1** ctx vs noctx (both blended) | **−1.028 [−1.213, −0.877]** | −1.289 [−1.386, −1.192] | **(True, "adopt (-1.03 s, still -1.29 s at 200 rows dropped)")** |
| **C2** blend vs a2o | **−1.255 [−1.527, −0.999]** | −1.556 [−1.770, −1.352] | **(True, "adopt (-1.26 s, still -1.56 s at 200 rows dropped)")** |

Both comparisons **adopt**. Curve at 20 / 2,000 rows dropped: C1 −1.236 / −1.154, C2 −1.509 / −1.427 s. C3 (blend5)
was not started, as instructed.

### Descriptive reports (no decision attached)

**Per fold** (curve verdict inside each fold; diff [CI], then at 200 dropped):

| fold | C1 | C1, 200 dropped | C2 | C2, 200 dropped |
|---|---|---|---|---|
| jan | −0.689 [−1.169, −0.473] adopt | −1.145 [−1.295, −1.001] | −0.978 [−1.784, −0.575] adopt | −1.558 [−1.812, −1.293] |
| jul | −1.447 [−1.636, −1.263] adopt | −1.679 [−1.884, −1.502] | −1.157 [−1.558, −0.744] adopt | −1.504 [−1.874, −1.109] |
| forward | −0.998 [−1.143, −0.864] adopt | −1.026 [−1.144, −0.911] | −1.603 [−1.958, −1.301] adopt | −1.457 [−1.657, −1.270] |

**Strict report** (`prc.compare.strict_report` / `strict_verdict`; folds, ≥ 70 % of airports, airport-day block
bootstrap over 1,230 UTC airport-days, each on all rows and with the worst 200 removed):

| | all rows: folds jan / jul / forward | airports better | airport-day CI | worst 200 removed: folds | airports | airport-day CI | strict verdict |
|---|---|---|---|---|---|---|---|
| C1 | −0.69 / −1.45 / −1.00 | 10/10 | −1.03 [−1.23, −0.84] | −1.19 / −1.67 / −1.08 | 10/10 | −1.29 [−1.44, −1.14] | adopt |
| C2 | −0.98 / −1.16 / −1.60 | 10/10 | −1.26 [−1.59, −0.97] | −1.56 / −1.45 / −1.65 | 10/10 | −1.56 [−1.86, −1.28] | adopt |

**Month bootstrap** (4 months: 202501, 202507, 202511, 202512): C1 −1.03 [−1.35, −0.77], C2 −1.26 [−1.69, −1.02].

**Per airport** (pooled, diff [paired-bootstrap 95 % CI, 500 draws]):

| airport | n | C1 diff [CI] | C2 diff [CI] |
|---|---|---|---|
| EDDF | 72,276 | −1.75 [−2.06, −1.50] | −2.08 [−2.62, −1.60] |
| EDDM | 51,688 | −1.05 [−1.40, −0.71] | −2.45 [−3.71, −1.52] |
| EGLL | 79,633 | −1.84 [−2.18, −1.53] | −0.81 [−1.76, +0.22] |
| EHAM | 79,639 | −0.52 [−0.70, −0.34] | −1.29 [−1.76, −0.85] |
| LEBL | 56,153 | −0.56 [−0.74, −0.39] | −1.36 [−1.61, −1.12] |
| LEMD | 70,732 | −0.44 [−0.56, −0.28] | −1.74 [−2.05, −1.47] |
| LFPG | 78,036 | −0.73 [−1.48, −0.46] | −0.94 [−2.00, −0.49] |
| LIRF | 49,924 | −1.30 [−1.65, −0.99] | −0.82 [−1.68, −0.07] |
| LSZH | 43,009 | −1.01 [−1.28, −0.75] | −0.87 [−1.46, −0.21] |
| LTFM | 91,338 | −1.75 [−2.01, −1.51] | −2.49 [−2.86, −2.09] |

**Lanes:** matched rows C1 −1.363 s, C2 −1.664 s; unmatched rows 0 (a2o in every arm).

**Per seed** (each seed alone, not part of the decision): seed 1 — C1 −0.78, C2 −0.97 (blend 268.953);
seed 2 — C1 −1.26, C2 −1.19 (blend 268.731). Seed 2's C1 is larger mainly through jul, where its noctx twin stopped
at epoch 4 (jul noctx alone 311.78 vs 306.33 for seed 1). Averaging the two seeds improves NN_ctx alone from
278.6 (either seed) to 275.9.

### Reading

Context information helps on all three folds, at all 10 airports and in both seeds: the ctx network is 5.6 s
better than its no-context twin on matched rows (214.43 vs 220.01), and in a 25 % blend with a2o that is −1.03 s
pooled (−1.29 s without the worst 200 rows). Against a2o alone the blend gains −1.26 s; about −0.23 s of that is
what the no-context NN already adds as a blend member (269.92 → 269.69), the rest is context. The NN alone stays
~6 s worse than a2o pooled. These are offline 2025 numbers on the same OOF rows the team used for earlier
adoptions; C3 (against the full blend5) is the step that would say whether the gain survives next to the other
members, and it was not run.
