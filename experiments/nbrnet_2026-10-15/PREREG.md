# NbrNet: a neighbour/context network for the matched lane — pre-registration (S0)

Written 2026-10-15, **before any NbrNet fit**, and committed on its own as step S0. Post-competition, offline
only: nothing from this experiment is uploaded anywhere (no leaderboard, no bucket). Designed from the plan
handed to this session only; no other team's repository was opened.

Everything a later step may not change is in §§1–6. Implementation choices that the plan left open are fixed in
§2–§3 as far as they can be before the data is on disk; anything that has to change after this commit is logged
with its reason in `RESULTS.md` ("Deviations"), never edited in here.

## 0. Question

The tree members see the traffic around a departure only through fixed-window counts and means
(`dep_tko_prev_15m`, `nbr_mvt_minus_aobt_mean_30m`, `seq_*`, ...). Does a network that attends over the
*individual* neighbouring movements (other departures and arrivals at the same airport around the take-off
time) add information that

* **C1 (primary):** the same network without the neighbours does not have, *when both are blended into a2o*;
* **C2:** a2o alone does not have?

## 1. Leakage rules (R1–R8) and how each is enforced

| rule | statement | enforcement | test |
|---|---|---|---|
| R1 | Other departures' `BLOCK_TIME_UTC_mvt` / `TAXITIME_SEC_mvt` are never used | the movement (token) tables are built from `prc.features.mask_departures(raw)`, which nulls both columns on every DEP row; the token builder reads no other label-like column | unit test: overwrite every departure's BLOCK/TAXITIME with random values (and with nulls) → token tables, neighbour indices and gathered token tensors are bit-identical |
| R2 | Arrivals' in-block time and taxi-in time are allowed | arrival tokens read `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` of ARR rows only (present in ranking.parquet) | — |
| R3 | The target row is excluded from its own tokens | the departure index excludes the query's own table row | unit test: no query's index row contains its own table row; perturbing a query's own (unmasked) label changes nothing |
| R4 | No target encodings and no neighbour OOF predictions in tokens | token fields are raw clock differences, flags and one-hots computed row by row from the masked table; no fitted statistic enters a token | code review + the R1 test (a label-derived token field would break it) |
| R5 | Normalisation and winsor bounds are fitted on the fold's training rows only | token fields use fixed transforms (signed log, fixed time scales), so nothing is fitted; the query `mlp.Preprocessor`, the target mean/scale and the winsor quantiles are fitted on the fold's training rows | unit test: the fitted target statistics do not change when the evaluation rows' labels are permuted |
| R6 | Neighbours come from the same source month (training file) as the query; a window-coverage scalar is added | movements are tagged with their source file; the index is built per (airport, source month); `cov` = share of the window [t0−60 min, t0+20 min] that lies inside that file's movement-time span at that airport | unit test: no index points into another source month |
| R7 | Movements up to t0 + 20 min are allowed | window [t0 − 60 min, t0 + 20 min] on the neighbour's movement time (departures: take-off `MVT`; arrivals: in-block time, landing time when in-block is null) | unit test: every indexed neighbour lies inside the window |
| R8 | No ADS-B in tokens | the token builder reads only the organisers' movement files | — |

## 2. Data and tokens

* Training data: the 12 `training_2025-*.parquet` files (downloaded with `prc.io_minio.download_files`, bucket
  `prc-2026-datasets`, into `data/raw`), plus `ranking.parquet`/`submitting.parquet` because `prc.pipeline`
  needs them to build its feature cache (the ranking frame is never used by NbrNet).
* **Query rows:** matched departures (`mvt_minus_aobt3` not null). Unmatched rows get no NbrNet prediction; they
  keep a2o everywhere below.
* **Query features:** the a2o member's feature frame (the `prc.pipeline` feature cache plus the L2/L6 levers the
  pipeline attaches, and the fold-level encodings/OSM distance that `predict_fold` attaches for this recipe),
  restricted to model A's visible columns (`cols_a`, as in `predict_fold`) and the categoricals, and transformed
  by `prc.mlp.Preprocessor` (m4's recipe: quantile-normal numerics, NaN indicators, capped one-hots) fitted on the
  fold's training rows. The plan's "125" input width was an estimate; the actual width is logged.
* **Movement tables** (one for departures, one for arrivals; float32 fields, int32 times in seconds since
  2025-01-01, sorted by airport, source month, time). Departure index matrix `idx_d` int32 `[n_dep, K_d=64]`:
  the 64 other departures of the same airport and source month whose take-off `MVT` is nearest to the query's
  `t0` (|MVT − t0|) among those inside [t0 − 60, t0 + 20] min, padded with −1. Arrival index `idx_a` int32
  `[n_dep, K_a=48]`: the 48 arrivals (same airport and source month) nearest by in-block time (landing time if
  in-block is null) inside the same window. Tokens are gathered per batch; the full token tensor is never
  materialised.
* **Departure token fields** (`slog(x) = sign(x)·log1p(|x|)` of seconds; a null field is 0 with its null flag
  set): Δt = (MVT_j − t0)/1800; ΔAOBT = slog(AOBT_3_j − t0); AOBT order = 1 if AOBT_3_j < AOBT_3_query, −1 if
  later, 0 if either is null; proxy slog(MVT_j − AOBT_3_j); slog(AOBT_3_j − EOBT_1_j); slog(MVT_j − SCHED_j);
  slog(MVT_j − LOBT_j); same-runway, same-stand, same-stand-prefix flags (vs the query); wake one-hot
  (`WK_TBL_CAT_flt` L/M/H/J/other-or-null); matched flag; null flags for AOBT_3, EOBT_1, LOBT, SCHED, runway,
  stand.
* **Arrival token fields:** Δ landing = (MVT_j − t0)/1800; Δ in-block = (BLOCK_j − t0)/1800; taxi-in
  slog(TAXITIME_j); arrival delay slog(BLOCK_j − SCHED_j); slog(MVT_j − SCHED_j); same-runway (landing runway =
  query's departure runway), same-stand, same-stand-prefix flags; wake one-hot; null flags for BLOCK, SCHED,
  runway, stand.
* **Stand prefix:** the leading run of letters of the stand name, or its first character when it starts with a
  digit (fixed here; the share of rows per airport sharing a prefix is logged in S5).

## 3. Model and training

* torch CPU (`torch==2.9.1+cpu`), `torch.set_num_threads(4)`, `POLARS_MAX_THREADS=1`.
* Token MLP per type (fields → 64 → 64, ReLU) plus a learned type embedding; padding mask from −1 indices.
* Query MLP: input → 256 → 128 → 64 (ReLU).
* Two rounds of single-query cross-attention: 4 heads, d = 64, key padding mask, residual + LayerNorm, then
  feed-forward 64 → 128 → 64 with residual + LayerNorm. A query with no valid token gets a zero attention output
  (no NaN).
* Masked mean pool of the departure tokens and of the arrival tokens (64 each), the valid-token counts
  (n_d/64, n_a/48) and the window-coverage scalar.
* Head: [query 64, attended query 64, pooled dep 64, pooled arr 64, n_d, n_a, cov] → 128 → 1 (ReLU).
* **Control twin `noctx`:** the same network with every token masked (attention output, pools, counts and
  coverage are all zero); same seed, same data order, same training protocol.
* Target: g = y − (MVT − AOBT_3) on matched rows (`train.py` `matched_target`, anchored), winsorised at the
  fold-train 0.05 % / 99.95 % quantiles, standardised by the fold-train mean/std of the winsorised g; loss MSE.
  Prediction in seconds = proxy + mean + std · output.
* AdamW lr 1e-3, weight decay 1e-4, batch 1024, at most 12 epochs, constant learning rate. Early stopping on the
  inner holdout (`train._inner_holdout_mask`, the latest training month; RMSE in seconds on unwinsorised g),
  patience 2, the best epoch's weights are kept (no refit on the holdout month).
* Deterministic: the epoch's row order is `default_rng([seed, epoch]).permutation`; `torch.manual_seed(seed)` for
  initialisation. Checkpoint (model, optimiser, epoch, RNG states, best weights, history) every 2 epochs;
  committed and pushed when the file is ≤ 2 MB, otherwise a weights-only copy ≤ 2 MB is pushed if one fits.

## 4. Evaluation folds and the a2o baseline

* Folds: `prc.cv.month_block_folds` — `jan`, `jul`, `forward` (Nov–Dec). (`jul_fwd` is fitted by the pipeline but
  is not used here, as in the pooled OOF.)
* a2o OOF regenerated with
  `python -m prc.pipeline --config configs/main_flseq_osm_nolirf_l2_opt_l6_cmix.yaml --skip-predict --fold-cache <dir>`.
  Sanity: pooled RMSE should be ≈ 270.4–270.7 (`l2_opt_cmix` 270.702, `l2_opt_l6` 270.467 in
  `experiments/experiments.jsonl`). The fold-cache OOF parquets are committed.
* Every blend and every NbrNet-alone prediction is, on rows with an NM record, projected onto
  [max(MVT − LOBT − 3606, 0), MVT − LOBT + 3606] (as `train._lobt_bound`), then floored at 0.

## 5. Pre-registered comparisons and decision rule

All comparisons are on **all rows of the evaluated fold(s)** (matched and unmatched; unmatched rows are a2o in
both arms), RMSE(B) − RMSE(A), negative = B better.

| id | A | B | when |
|---|---|---|---|
| **C1 (primary)** | 0.75·a2o + 0.25·NN_noctx | 0.75·a2o + 0.25·NN_ctx | screen (forward, seed 1), then jan+jul+forward |
| **C2** | a2o | 0.75·a2o + 0.25·NN_ctx | same |
| **C3** | blend5 | 0.8·blend5 + 0.2·NN_ctx | later, only if C2 adopts |

Weights are fixed here and not tuned. Decision: `prc.compare.verdict(prc.compare.robustness_curve(y, A, B,
n_boot=1000))` — adopt only if the full-data CI and the CI with the worst 200 rows dropped both exclude 0 on the
improving side. The final verdicts are taken on the pooled jan + jul + forward rows after S8 (NN predictions the
mean over seeds 1 and 2); the forward-fold verdicts at S7 are reported but are a screen only.

**Futility stop (after the S7 screen, forward fold, seed 1):** stop the experiment (no S8) if C1's point estimate
is ≥ −0.10 s, or if C1's full-data 95 % CI lies wholly above 0.

Descriptive only (no decision attached): NN_ctx and NN_noctx alone vs a2o; per-fold numbers; per-airport
breakdown; matched-lane-only RMSEs; epoch histories.

## 6. Compute gates (S6) and step plan

* Machine: 4 CPUs, ~13.6 GB cgroup limit, no GPU. One heavy job at a time; NbrNet never runs alongside a
  LightGBM CV. NbrNet runs under a guard (`run_guarded_nn.sh`) that kills it above 13,000 MB on the session's own
  cgroup.
* S6 micro-benchmark, 200 training steps of the ctx model: the extrapolated epoch time must be ≤ 15 min. If not,
  fall back in this order, re-benchmarking after each: (1) K_d/K_a = 40/24 and d = 48; (2) additionally a 50 %
  row subsample per epoch (a fresh deterministic half per (seed, epoch)). The setting used is recorded and
  applies to both twins.
* Steps: S0 this note; S1 venv + torch; S2 download; S3 METAR/OSM + feature cache + a2o CV with fold cache
  (background); S4 NbrNet code + synthetic test + leakage unit tests; S5 token tables and indices; S6
  micro-benchmark; S7 screen = forward fold, seed 1, ctx + noctx, C1 and C2 on the forward fold, futility rule.
  S8 (jan + jul seed 1, all folds seed 2) only if S7 passes the futility rule, and only on instruction.
