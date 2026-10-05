# MLP as a fourth matched-lane member (M4) of hybrid_blend_l2_pa_opt — pre-registration (2026-10-05)

**Status (2026-10-05): done. ADOPT.** `hybrid_blend_l2_pa_opt` → `hybrid_blend_l2_pa_opt_mlp` (equal 1/4 weights):
−0.75 s [−0.93, −0.58], K=200 −0.83 s [−1.00, −0.67]. M4 alone is worse than M1 by +5.59 s [+4.67, +6.69].
`outputs/hybrid_blend_l2_pa_opt_mlp.parquet` is written and validated. Nothing was uploaded or submitted. §0 was
committed as 9a8bc61 before any run. Results are in §1+, and §0 has not been edited except for this status line.

## 0. Pre-registration

Full test suite before any run: 262 passed (252 before this change, plus the 10 new tests in `tests/test_mlp_member.py`).

### 0.1 Why, and the prior

The board best is v33 = `outputs/hybrid_blend_l2_pa_opt.parquet` + the learned ADS-B residual (250.9496). Its
matched lane is the equal-weight mean of three tree members:
- M1 = 0.5 × pooled anchored LGB+L2 (retuned, `main_flseq_osm_nolirf_l2_opt-7244b3004f`) + 0.5 × per-airport anchored
  LGB+L2 (retuned, `main_flseq_osm_nolirf_l2_pa_opt-8d54effc06`)
- M2 = plain-y LGB+L2 (retuned, `main_flseq_osm_nolirf_plainy_l2_opt-dd8b8e2c7c`)
- M3 = anchored CatBoost+L2 (`main_flseq_osm_nolirf_cb_l2-4aa341b206`)

The unmatched lane is `main_flseq-8615e6a194`'s model B. Where these live: `optuna-retune.md` §3–§4.

Members that were worse alone have helped the blend:
- plain-y was +0.93 s alone and −0.78 s in the blend (`plainy-blend.md`);
- CatBoost was +0.94 s alone and −1.75 s in the blend (`catboost-blend.md`). Its error correlation with the anchored
  LGB member was 0.93, the lowest of the three.

A neural network is a different function class from all three, so its errors may be less correlated still. Against
that: `adsb-learned-residual.md` V1 found an sklearn MLP worse than LightGBM alone on a different target and feature
set. A 1/4 weight on a member that is much worse alone can also hurt the blend.

**Prior:**
- M4 alone is worse than M1 alone on the matched lane by several seconds (+3 to +15 s).
- The blend effect could have either sign. P(ADOPT) ≈ 0.35.

### 0.2 Library and code

- **Library: scikit-learn 1.9.0 `MLPRegressor`.** It is in `requirements-lock.txt` (`scikit-learn==1.9.0`) and in
  `.venv`. torch, tensorflow, jax and xgboost are not installed. Nothing is installed.
- **`TrainConfig.mlp_matched`** (bool, default off) and **`TrainConfig.mlp_params`** (dict, default `{}`). Both are in
  `_HASH_OPTIONAL`, so off they are out of the hash. On, `predict_fold` fits model A with `prc.mlp.fit_mlp` instead
  of `_fit`, and the resolved `MLP_DEFAULTS` go into the hash.
  - Model B is unchanged LightGBM with `params`.
  - Everything outside the fitter is shared with the LightGBM path: encodings, fold features, column selection, the
    anchored target and add-back, the zero floor and the LOBT projection.
  - Refused in combination: `matched_params`, `airport_models`, `use_linear_tree`, `es_holdout_encodings`, and
    `model` ≠ lightgbm. Unknown `mlp_params` keys are refused, and so is `mlp_params` set with the flag off.
- **`tests/test_mlp_member.py`:**
  - All 128 recipes in `configs/` at f6b09ef keep their experiment id and both models' resolved parameters byte for
    byte. Only `main_flseq_osm_nolirf_l2_mlp.yaml` is new.
  - `predict_fold`'s predictions and `info` hash exactly as under the f6b09ef code on a fixed synthetic fold, for
    four arms: default, plain-y, per-airport and `matched_params`. The digests were computed by running the f6b09ef
    code in a separate worktree. Under the new code all 128 ids and all four digests were identical before the test
    was written.
  - The recipe is its base plus the flag, and the settings are the ones fixed in §0.3.
  - Refusals work.
  - With the flag on, model B's predictions are byte-identical, model A moves, and its predictions are finite and
    ≥ 0.
  - Removing the target frame's labels changes nothing. A different batch of target rows changes predictions only
    by float32 rounding.
  - The preprocessor handles unseen levels and missing values.
  - A run killed mid-stage and resumed from its checkpoint equals an uninterrupted run (atol 1e-6 s). A third call
    reads every stage back.
  - The blend script's member ids, files and formulas.
- `tests/test_matched_params.py` and `tests/test_per_airport_member.py` have recipe-list guards. They are widened to
  ignore `*_mlp.yaml`, which this file checks.

### 0.3 The member M4 (fixed here)

Recipe **`configs/main_flseq_osm_nolirf_l2_mlp.yaml`**, exp id **`main_flseq_osm_nolirf_l2_mlp-2b9a3b9348`**. It is
`configs/main_flseq_osm_nolirf_l2.yaml` (`ef204b44ff`) plus `mlp_matched: true` and a new name.

The base is the un-retuned recipe. The retuned `_opt` recipe differs only by `matched_params`, which are LightGBM
model-A settings that change no column, so the two recipes give model A the same columns. `mlp_params` is not set, so
the settings are `prc.mlp.MLP_DEFAULTS`.

**Target.** The anchored target, the same as M1's: `g = TAXITIME_SEC_mvt − mvt_minus_aobt3`.
- Prediction is `mvt_minus_aobt3 + ĝ`, then the zero floor, then the LOBT projection (slack 3606 s), as for every
  member.
- For the fit, `g` is standardised by the mean and SD of the rows the stage trains on. This is undone at prediction.
- The loss is squared error.

**Feature columns.** Exactly model A's columns in `predict_fold` for this recipe: `cols_a` (the visible numeric
columns, L2 included, `mvt_minus_aobt3` last) and the 10 `CATEGORICALS`. That is the list LightGBM model A gets.
The fold's label-derived encodings come from `fit_encodings` on the fold's training rows, as for LightGBM. Their
early-stopping holdout is not re-encoded (`es_holdout_encodings` false, as in M1).

**Preprocessing.** It is fitted on the matched training rows of the fold (or of the full fit), and only on those.
- **Numeric columns:**
  - ±inf → NaN.
  - A `QuantileTransformer` with normal output and 1,000 quantiles, fitted on a fixed random subsample of 200,000
    training rows (`default_rng(seed 42)`).
  - After the transform, NaN → 0, which is the median in normal space.
  - A 0/1 missing indicator is added for every column that has at least one NaN on the training rows.
  - Nothing else is scaled. The quantile transform bounds every input to about ±5.2.
- **Categorical columns** (ADEP, RUNWAY, STAND, AIRCRAFT_TYPE, WK_TBL_CAT, OPERATOR, MARKET_SEGMENT, FLIGHT_TYPE,
  ADES, ADES_REGION):
  - Values are taken as strings, with null → `"__NA__"`.
  - Each column is one-hot encoded over its levels with at least 1,000 training rows, keeping at most the 64 most
    frequent (ties by name).
  - One "other" column per categorical takes everything else, including levels not seen in training.
  - Label-free. High-cardinality detail (stand, operator) also reaches the model through the existing numeric
    encodings (`ref_*`, `op_offset_*`).
- The design matrix is float32. It is built per chunk of 65,536 rows and never held whole. The input width is
  recorded per fold in `info["mlp_matched"]`.

**Architecture and optimisation.**
- `MLPRegressor`: hidden layers (512, 256), ReLU, linear output.
- Adam with learning rate 1e-3 and scikit-learn's default betas and epsilon. L2 `alpha` 1e-4. Batch size 1,024.
- Each epoch is one shuffled pass over the training rows. The order is `default_rng([seed, epoch]).permutation`,
  fed as chunks of 65,536 rows to `partial_fit` (one pass over each chunk in mini-batches). scikit-learn's own
  shuffle is off.

**Early stopping, on an inner split of the training fold only.**
- The holdout is the latest calendar month of the fold's training side (`_inner_holdout_mask`,
  `inner_holdout_months` 1, as for LightGBM), on matched rows. The evaluation fold is never read.
- Stage 1 trains on the remaining training rows with seed 42 for at most **40 epochs**. After each epoch it computes
  the holdout RMSE in seconds of `g`.
- Learning rate: halved each time the count of epochs since the best reaches a multiple of 2, so at 2 and 4.
- Stop: after **5** epochs without a new best.
- E* is the best epoch. The learning-rate schedule of epochs 1…E* is kept.

**Refit and seeds.**
- On all matched training rows of the fold, for E* epochs, with that schedule replayed.
- One model per seed **42, 43, 44** (`n_seeds` 3, as M1). Each seed sets the initial weights and the epoch order.
- Their predictions of `g` are averaged.
- Stage 1's model is not used for prediction.

**Folds.** The recipe has no `folds:` key, so the pipeline's four production folds are used: jan, jul, jul_fwd and
forward. The pooled OOF is jan, jul and forward, 672,428 rows; jul_fwd is shadowed, exactly as for M1 and M2. The
OOF is row for row comparable with the other members.

**Model B** is LightGBM, identical to `main_flseq_osm_nolirf_l2-ef204b44ff`'s model B. Its predictions are expected
byte-identical to that run's, and this is checked. It is not used in any blend; the unmatched lane is
`main_flseq-8615e6a194` throughout.

**Checkpoints and resume.**
- `--fold-cache logs/mlp/fold_cache` writes each fold's predictions as it finishes.
- `PRC_MLP_CKPT=logs/mlp/ckpt` writes every epoch of every stage, plus each refit's predictions. They are keyed by
  a digest of the stage's inputs and settings.
- A restarted run resumes at the next epoch of the stage it was in.

### 0.4 Candidates and the decision

Matched lane = `mvt_minus_aobt3` non-null. Unmatched lane = `main_flseq-8615e6a194` model B in every candidate.
`scripts/blend_mlp.py oof --scratch logs/mlp/scratch --m4 <id>` writes the hybrid OOFs. A dry run with a stand-in
for M4 reproduces the base's **268.32** (`optuna-retune.md` §3).

| name | matched lane | role |
|---|---|---|
| `hybrid_blend_l2_pa_opt` (base) | (M1 + M2 + M3) / 3 | the OOF of `outputs/hybrid_blend_l2_pa_opt.parquet` |
| **`hybrid_blend_l2_pa_opt_mlp`** (primary) | **(M1 + M2 + M3 + M4) / 4** | **the decision** |
| `member_m1` vs `member_m4` | M1 alone vs M4 alone | secondary, report only |
| `hybrid_blend_l2_pa_opt_mlp_lofo` | leave-one-fold-out weights | secondary, report only |

For the LOFO candidate: for each pooled fold, sum-to-one least-squares weights on (M1, M2, M3, M4), sign free, are
fitted on the matched OOF rows of the other two folds and applied to the held-out fold
(`catboost_blend.sum_to_one_weights`).

**Decision.**
- Command: `python -m prc.compare hybrid_blend_l2_pa_opt hybrid_blend_l2_pa_opt_mlp --work-dir data/work_flseq
  --exp-dir logs/mlp/scratch`.
- Rule (`prc.compare.verdict`): the full CI (K=0) and the drop-200 CI must both exclude zero on the same side.
  **ADOPT** needs the improving side. Both on the worse side is REJECT. Anything else is not adopted.
- Reported for the primary: the K = 0/20/200/2000 curve, the month bootstrap, and `scripts/blend_mlp.py detail`
  (the lane split matched / unmatched / all except LIRF, per calendar month, per airport).

**Secondary, report only, never the decision:**
- `member_m1` vs `member_m4` with `prc.compare`.
- Per-fold matched RMSE of each member, and member error correlations.
- LOFO weights per fold and on all folds, and the LOFO hybrid vs the base with `prc.compare`.

### 0.5 Output (only if ADOPT)

1. Full fit: `prc.pipeline --config configs/main_flseq_osm_nolirf_l2_mlp.yaml --work-dir data/work_flseq --skip-cv`
   → `outputs/main_flseq_osm_nolirf_l2_mlp.parquet`.
2. Blend: `scripts/blend_mlp.py full` → **`outputs/hybrid_blend_l2_pa_opt_mlp.parquet`**.
   - Matched rows: `rint(((0.5·a2o + 0.5·pao) + p2o + c2 + m4) / 4)` from the five Int32 full-fit files.
   - Unmatched rows: `outputs/main_flseq.parquet`.
3. Validate:
   - 344,841 rows, with the schema, ids and order equal to `data/raw/submitting.parquet`;
   - no nulls and no negatives;
   - unmatched rows equal to `outputs/hybrid_blend_l2_pa_opt.parquet`;
   - `prc.validate_submission` OK.

If the verdict is not ADOPT, there is no full fit and no file. Nothing is uploaded or submitted, and no ADS-B
residual is applied.

### 0.6 Runs and resources

- One training job at a time. It starts only when no other `prc.pipeline` or `adsb_learn_` process is running and
  cgroup `total_rss` is < 2,500 MB.
- `total_rss` is read from the memory cgroup in `/proc/self/cgroup`
  (`/sys/fs/cgroup/memory/process_api/<session>/claude-code-bash/memory.stat`; limit 14,345,031,680 B = 13,681 MB).
  It is sampled every 10 s, own cgroup and all sessions.
- The run is killed above 13,000 MB.
- Logs go to `logs/mlp/` (git-ignored), and every exit code is checked.
- Cost estimate from a synthetic benchmark (random data, 550 inputs, (512, 256), batch 1,024): about 20–30 s per
  1.9M-row epoch on 4 cores. That gives at most about 80 min per fold, and about 3–5 h for CV.

## 1. Run (CV)

`logs/mlp/run.sh cv --config configs/main_flseq_osm_nolirf_l2_mlp.yaml --work-dir data/work_flseq --skip-predict
--fold-cache logs/mlp/fold_cache` (with `PRC_MLP_CKPT=logs/mlp/ckpt`). Exp id **`main_flseq_osm_nolirf_l2_mlp-2b9a3b9348`**.
- Exit 0, not killed. Wall **9,037 s** (01:11:26 → 03:42:03 UTC).
- Peak cgroup `total_rss` **10,246 MB** (10,442 MB over all sessions), during feature load. MLP epochs ran at about
  9.3 GB.
- No restart happened. A first launch, attached to the tool shell, was stopped after 4 s during feature load
  (`logs/mlp/cv_attempt0.*`). It was relaunched detached. No fold was cached at that point.
- Epochs took 33–64 s each.
- **Model B is untouched.** All 8,378 pooled unmatched OOF predictions equal `main_flseq_osm_nolirf_l2-ef204b44ff`'s
  exactly.

| fold | input width | ES best epoch / epochs run | best holdout RMSE (s of g) | M4 matched RMSE | M1 (0.5 a2o + 0.5 pao) |
|---|---:|---|---:|---:|---:|
| jan | 582 | 13 / 18 | 205.67 | 202.63 | 195.24 |
| jul | 581 | 15 / 20 | 204.75 | 243.94 | 236.36 |
| jul_fwd (shadowed) | 568 | 9 / 14 | 229.64 | 253.94 | — |
| forward | 580 | 15 / 20 | 213.50 | 198.73 | 191.55 |
| pooled matched | | | | **213.27** | **205.94** |

Pooled matched RMSE of the other members: M2 208.82, M3 212.40. M4 is the worst member alone, but close to M3.

Matched-lane error correlations:

| | M1 | M2 | M3 | M4 |
|---|---:|---:|---:|---:|
| M1 | 1 | 0.9755 | 0.9414 | **0.9201** |
| M2 | | 1 | 0.9355 | **0.9098** |
| M3 | | | 1 | **0.9054** |

M4 is the least correlated member with each of the other three.

## 2. Blend vs hybrid_blend_l2_pa_opt (the decision)

`scripts/blend_mlp.py oof --scratch logs/mlp/scratch --m4 main_flseq_osm_nolirf_l2_mlp-2b9a3b9348`
(`logs/mlp/oof.txt`) reproduces the base at **268.32**. The candidate reads 267.57.

`python -m prc.compare hybrid_blend_l2_pa_opt hybrid_blend_l2_pa_opt_mlp --work-dir data/work_flseq --exp-dir
logs/mlp/scratch` (`logs/mlp/compare_primary.txt`), on 672,428 pooled OOF rows:

| K | A | B | diff [CI95] |
|---|---|---|---|
| 0 | 268.32 | 267.57 | **−0.75 [−0.93, −0.58]** |
| 20 | 223.59 | 222.70 | −0.90 [−1.07, −0.73] |
| 200 | 209.99 | 209.15 | **−0.83 [−1.00, −0.67]** |
| 2000 | 187.19 | 186.56 | −0.63 [−0.72, −0.53] |

Month bootstrap −0.75 [−1.07, −0.49]. **Verdict: ADOPT** (`ADOPT - adopt (-0.75 s, still -0.83 s at 200 rows dropped)`).

Detail (`logs/mlp/detail.txt`):
- **Lane split.** Matched 205.58 → 204.59: −0.99 [−1.18, −0.80]. Unmatched +0.00 (identical by construction). All rows
  except LIRF −0.55 [−0.73, −0.41].
- **Per calendar month**, all rows / matched. Every CI is below zero.
  - 2025-01: −0.43 [−0.79, −0.24] / −0.75
  - 2025-07: −1.13 [−1.50, −0.77] / −1.42
  - 2025-11: −0.60 [−0.89, −0.35] / −0.70
  - 2025-12: −0.83 [−1.17, −0.53] / −0.90
- **Per airport**, all rows. Eight of ten CIs are below zero; LEBL, EGLL and LSZH include zero.
  - LIRF −2.09 [−2.76, −1.33]
  - LTFM −1.05
  - LFPG −0.60
  - EGLL +0.18 [−0.29, +0.63]
  - EDDF −0.75
  - EDDM −1.30
  - EHAM −0.90
  - LEBL −0.24 [−0.49, +0.02]
  - LEMD −0.53
  - LSZH +0.01 [−0.31, +0.38]

## 3. Secondary (report only)

- **M4 alone vs M1 alone** (`member_m1` → `member_m4`, both with the same unmatched lane;
  `logs/mlp/compare_member.txt`): K=0 **+5.59 [+4.67, +6.69]**, K=200 +7.00 [+6.40, +7.66]. That is REJECT as a
  replacement. Matched 205.9 → 213.3. This falls within the prior (+3 to +15 s).
- **LOFO sum-to-one weights** on (M1, M2, M3, M4), each fitted on the other two folds:
  - held-out jan: 0.505 / 0.107 / 0.143 / 0.245
  - held-out jul: 0.524 / 0.110 / 0.146 / 0.220
  - held-out forward: 0.535 / 0.089 / 0.126 / 0.249
  - All three folds: 0.520 / 0.103 / 0.138 / 0.239.
  - LOFO hybrid vs base (`logs/mlp/compare_lofo.txt`): K=0 −1.00 [−1.23, −0.81], K=200 −1.12 [−1.31, −0.95], month
    bootstrap −1.00 [−1.33, −0.76]. That is 0.25 s better than equal weights.
  - The weights move weight from M2/M3 to M1 and keep M4 at about 1/4. They do not enter the decision; the file uses
    1/4 each.

**Reading.**
- The rule passes: the K=200 upper end is −0.67.
- As with plain-y and CatBoost, the member that is worst alone helps the blend, because its errors are the least
  correlated with the others.
- The gain is larger in July and at LIRF, and is near zero at EGLL, LSZH and LEBL.
- Caveats:
  - The MLP's settings were fixed a priori and not tuned. One seed set (42–44) was run.
  - On the board, expect less than the OOF gain.

## 4. Full fit and output (ADOPT, so §0.5 applies)

`logs/mlp/run.sh full --config configs/main_flseq_osm_nolirf_l2_mlp.yaml --work-dir data/work_flseq --skip-cv`:
- Exit 0, not killed. Wall **3,687 s**. Peak `total_rss` **9,031 MB** (9,268 MB over all sessions).
- 2,062,577 matched training rows and input width 583. The holdout is 2025-12 (164,028 rows).
- Early stopping chose E* = 15 (20 epochs run, best holdout 204.65). Three refits of 15 epochs each.
- `outputs/main_flseq_osm_nolirf_l2_mlp.parquet` passes `prc.validate_submission` (OK, with only the usual >10,800 s
  warning). Its 5,290 unmatched rows equal `outputs/main_flseq_osm_nolirf_l2.parquet` (model B untouched).

`scripts/blend_mlp.py full` → **`outputs/hybrid_blend_l2_pa_opt_mlp.parquet`** (`logs/mlp/full_blend.txt`):
- Matched rows: `rint(((0.5·a2o + 0.5·pao) + p2o + c2 + m4) / 4)`. Unmatched rows: `outputs/main_flseq.parquet`.
- 344,841 rows, the same as the template: 339,551 matched and 5,290 unmatched.
- The schema `{MVT_ID_mvt: double, TAXITIME_SEC_mvt: int32}` equals the template, and the ids and their order equal
  `data/raw/submitting.parquet`.
- 0 nulls and 0 negatives; min 0, max 113,641.
- `prc.validate_submission` OK, with only the >10,800 s warning.
- Against `outputs/hybrid_blend_l2_pa_opt.parquet`: differs on 325,625 rows, all matched (unmatched untouched), with
  max |d| 1,533 s and RMS 22.34 s.

Nothing was uploaded or submitted, and no ADS-B residual was applied. v33 = `hybrid_blend_l2_pa_opt` + the learned
ADS-B residual, so the like-for-like candidate is this file plus a residual refitted on this base, which is a
separate step.

Logs (git-ignored): `logs/mlp/` contains `cv.*`, `full.*`, `oof.txt`, `compare_{primary,member,lofo}.txt`,
`detail.txt`, `full_blend.txt` and `validate_*.txt`. Checkpoints are in `logs/mlp/ckpt`, and the fold cache in
`logs/mlp/fold_cache`.
