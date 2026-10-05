# New-stand work (`stand_dropout`) on `claude/peaceful-babbage-md9l2x`: review (2026-10-05)

## Review against the current base (written from `claude/model-b-and-arrival-features`)

This branch is our own team's work. Commits reviewed: 19968d9, 0e3de1e, 525d358, 897c872, 437effa and 9460b5a. Branch
head at review time: 3f2a820.

Scope of this review:
- Read-only: nothing was trained, and nothing in `src/` or `configs/` was changed.
- One label-free count was run (§5), peak RSS 0.38 GB.

Tags: **[S]** = read from the branch's files or records, **[new]** = computed for this review, **[X]** = inference.

### 1. Definition [S]

- **Flag.** `TrainConfig.stand_dropout: float = 0.0`, added in 19968d9 to `src/prc/train.py`.
  - It is in `_HASH_OPTIONAL` with default 0.0, so every existing id is unchanged.
  - A non-zero value changes the id (`tests/test_new_stand.py`).
- **Function.** `apply_new_stand_token(tr, tg, share)`.
  - **Training rows:** a share `share` of the rows is chosen by `crc32("newstand|<MVT_ID>")` among rows whose stand is non-null. On those rows:
    - `STAND_mvt = "__NEW__"`;
    - `ref_p10`, `ref_p20`, `ref_med` and `ref_n` are set to null;
    - `ref_p10_fb = coalesce(arh_med, apt_med)` (airport × runway × hour, else airport);
    - `ref_missing = 1`.
  - **Target rows:** a row whose (`ADEP_mvt`, `STAND_mvt`) pair never occurs in the training frame gets `STAND_mvt = "__NEW__"`. Its stand reference is already missing.
  - Nothing reads a label.
- **Where it runs.** In `predict_fold`, after `attach_encodings` and `add_fold_features`, so the OSM distance comes from the real stand. It runs before the matched/unmatched split.
- **Pre-registered setting:** `stand_dropout: 0.05` (`docs/research_2026-10-05_newstand/plan.md`, 19968d9).

### 2. Model and lane affected [S]

- **Code path.** The token is applied to `tr` and `tg` before the lane split. It therefore reaches:
  - model A (matched lane);
  - model B (unmatched lane);
  - every model-A fitter: pooled LightGBM, `airport_models` (`_fit_grouped`), `mlp_matched` and CatBoost.
- **What was measured.** Only LightGBM `main_flseq_osm_nolirf` (anchored model A, 3 seeds, no L2, no `matched_params`), on the forward fold only (train Jan–Oct, validate Nov+Dec 2025). Both lanes were measured.
- **Exception.** `es_holdout_encodings` builds from `train_f`, before the token. This flag is off in all of our member configs.

### 3. What it was measured against [S]

- **Base config.** `configs/ns_base.yaml` = `configs/main_flseq_osm_nolirf.yaml` + `folds: [forward]`. The diff is only the name, the comments and `folds`.
- **Arms** (configs in 19968d9; `git_sha` of each record in `experiments/experiments.jsonl` on that branch):

| arm | extra settings | exp id | run sha | forward full RMSE | train s |
|---|---|---|---|---:|---:|
| ns_base | — | `ns_base-2c65506a2d` | 8019757 | 217.51 | 1,491.2 |
| ns_drop | `stand_dropout: 0.05` | `ns_drop-693275dd32` | 8019757 | 217.56 | 930.8 |
| ns_base_u | `sim_unseen_stands: 0.2` | `ns_base_u-2d1a8167a5` | 0e3de1e | 221.92 | 987.6 |
| ns_drop_u | both | `ns_drop_u-f58349b6af` | 525d358 | 220.65 | 967.0 |

- `sim_unseen_stands: 0.2` removes from training every row of 20 % of the (airport, stand) pairs (CRC32 of `ADEP|STAND`). That is 343,363 of the 1,757,038 training rows.
- In ns_drop, 87,743 training rows got the token, plus 472 forward validation rows. In ns_drop_u, the counts were 70,651 and 65,513.
- The evaluation script `docs/research_2026-10-05_newstand/evaluate.py` was pushed in 0e3de1e, before any arm finished. Verdict 3 used `prc.compare ns_base ns_drop`.

### 4. Verdicts [S] (`verdict_1_2.txt`, `verdict_3.txt`, `results.md`)

**Verdicts 1 and 2: ns_base_u → ns_drop_u, on the simulated-unseen-stand rows.** There are 65,242 such rows out of 328,009 forward validation rows.

| rows | n | RMSE | K=0 | K=20 | K=200 |
|---|---:|---|---|---|---|
| unseen stands | 65,242 | 222.03 → 215.54 | **−6.49 [−7.52, −5.47]** | −7.33 [−8.17, −6.56] | **−6.99 [−7.54, −6.43]** |
| unseen, matched | 64,659 | 211.67 → 204.97 | −6.70 [−7.40, −6.01] | −7.00 [−7.71, −6.37] | −6.86 [−7.39, −6.37] |
| **seen stands** | 262,767 | 221.89 → 221.90 | **+0.00 [−0.98, +0.98]** | −0.14 [−0.72, +0.34] | **+0.04 [−0.18, +0.28]** |

- Unseen rows by airport (RMSE change):
  - EDDF −4.37, EDDM −6.72, EGLL −12.28, EHAM −7.15, LEBL −7.74
  - LEMD −6.63, LFPG −1.19, LIRF **+2.20**, LSZH −19.58, LTFM −6.07
  - 9 of 10 airports improve.
- Verdict 1 PASS (the K=200 CI excludes 0 on the improving side). Verdict 2 PASS (K=0 ≤ 0).

**Verdict 3: ns_base → ns_drop, normal forward fold (known stands).**

| K | A | B | diff [CI95] |
|---|---:|---:|---|
| 0 | 217.51 | 217.56 | +0.05 [−0.59, +0.68] |
| 20 | 202.59 | 202.43 | −0.16 [−0.43, +0.11] |
| 200 | 193.00 | 192.91 | **−0.09 [−0.28, +0.08]** |
| 2000 | 173.36 | 173.29 | −0.07 [−0.18, +0.04] |

- Pre-registered rule: the upper end of the K=200 CI must be ≤ +0.10. It is +0.08, so verdict 3 is **PASS**.
- The general `prc.compare` curve verdict is REJECT ("no difference"), and the STRICT verdict is REJECT. Neither is the pre-registered rule.
- Month bootstrap: +0.05 [−0.67, +0.76]. 7 of 10 airports are better.
- Unmatched lane: 981.7 → 985.9 (reported only, not used in the verdict).

**Board estimate in `results.md`:** about −0.04 s [X], if the −6.5 s carries over to the 1,589 rows in the 2026 data.

### 5. 2026 scored rows on unseen stands [new]

How this was counted:
- Inputs: `data/work_flseq/feat_rank.parquet`, whose 344,841 ids equal `data/raw/submitting.parquet`, against the labelled rows of `data/work_flseq/feat_train.parquet`.
- Matching on the (`ADEP_mvt`, `STAND_mvt`) pair, which is the rule `apply_new_stand_token` uses in a full fit.

Totals:
- **1,589 rows (0.461 %)** on 46 pairs. 1,546 are in the matched lane and 43 in the unmatched lane.
- 40 rows are in Jan 2026 and 1,549 in Jul 2026.
- This reproduces the 1,589 in `docs/data_audit_2026-09-18/results/recording-change.md`.

| airport | rows | stands | Jan 2026 | Jul 2026 |
|---|---:|---:|---:|---:|
| EDDM | 1,182 | 10 | 0 | 1,182 |
| EDDF | 300 | 6 | 1 | 299 |
| LSZH | 58 | 11 | 17 | 41 |
| LEMD | 14 | 7 | 4 | 10 |
| EHAM | 11 | 2 | 11 | 0 |
| LIRF | 9 | 2 | 2 | 7 |
| LFPG | 8 | 5 | 3 | 5 |
| LTFM | 6 | 2 | 1 | 5 |
| LEBL | 1 | 1 | 1 | 0 |
| EGLL | 0 | 0 | 0 | 0 |

- The largest single stands are EDDM 105 (250 rows), 103 (239), 104 (228), 106 (164) and 108 (156).
- The 2,708 rows on stands with fewer than 30 training rows (mainly EDDF T3; `plan.md`) are seen stands. The token does not reach them (`results.md`).

### 6. Compatibility with the current base (R6: a2o, pao, p2o, c2, m4 + u)

- **Code.**
  - Our branch has no `src/` commit that the new-stand branch lacks (`git log <ns-branch>..HEAD -- src/` is empty). Their `train.py` is therefore a superset of ours. [S]
  - Using the flag on our branch would need that `src/` change merged. The merge also brings `fe_month_scope` and `fe_retention`, which default to off with ids unchanged.
- **Member configs.** None of our six member configs sets `stand_dropout`, `sim_unseen_stands` or `es_holdout_encodings`. [S]
- **Not measured with any of our current members** [S]. The verdicts are on a model with:
  - no L2 (`fe_fwd_nm`);
  - no retuned `matched_params` (trial 8: `num_leaves` 357, `min_data_in_leaf` 11, `lambda_l2` 8.928);
  - no `airport_models`;
  - no plain-y target;
  - no MLP and no CatBoost.

  What carries over to each member, judged from the code path only [X]:
  - **L2:** independent columns, with no interaction in the code.
  - **Retuned params:** `min_data_in_leaf` 11 lets small leaves form, which may treat the 5 % `__NEW__` rows differently.
  - **Per-airport member:** each airport fit gets its own ~5 % token rows. The `seen` set is keyed by (airport, stand), so this stays consistent.
  - **MLP:** it one-hots at most 64 levels (`cat_max_levels`, `src/prc/mlp.py`), and an unseen stand already falls into the "other" column today. `__NEW__` at ~5 % of rows would likely get a column of its own, and the blanked `ref_*` columns would set its missing indicators.
  - **CatBoost:** it handles categoricals natively. Its behaviour on the token is unmeasured.
- **Blend weights.** The R6 weights were fitted on the current members' OOFs. A retrained member set would need new OOFs (CV runs) to refit the weights, or the old weights would have to be reused unmeasured. [X]

### 7. What integration would take

- **Retraining.**
  - The five matched members (a2o, pao, p2o, c2, m4) would need `stand_dropout: 0.05` and a new id each. Only they drive the 1,546 matched unseen rows.
  - `u` (`main_flseq`) carries the 43 unmatched rows. On the forward fold, the unmatched lane got worse with the token (981.7 → 985.9), so `u` could be left as is. [X]
- **New-stand runs' own cost [S].**
  - `train_seconds` per arm was 930.8–1,491.2 s for one forward fold of 3-seed LightGBM.
  - No peak memory is recorded for these runs, either in `experiments.jsonl` or in `docs/research_2026-10-05_newstand/`.
- **Our members' recorded cost** (wall s / peak cgroup `total_rss`) [S]:

| member | CV (4 or 3 folds) | full fit | source |
|---|---|---|---|
| a2o | 6,992 s / 11,028 MB | 1,633 s / 9,579 MB | `optuna-retune.md` |
| pao | 7,177 s / 9,007 MB | 1,263 s / 7,613 MB | `optuna-retune.md` |
| p2o | 1,302 s / 10,781 MB (resumed run, forward fold only; the pre-retune `plainy_l2` CV took 3,959 s / 11,169 MB) | 1,332 s / 9,695 MB | `optuna-retune.md`, `blend-l2.md` |
| c2 (CatBoost, 3 folds) | 10,819 s (CV + full 16,638 s) / 11,977 MB | ≈ 5,819 s (difference) | `blend-l2.md` |
| m4 (MLP) | 9,037 s / 10,246 MB | 3,687 s / 9,031 MB | `mlp-member.md` |

- **Sequential sums** [X, arithmetic on the rows above]:
  - full fits only: ≈ 13,700 s (3.8 h);
  - CV + full fit, with 3,959 s for the p2o CV: ≈ 51,700 s (14.4 h).
- **Memory.** Every one of these runs peaked at 7.6–12.0 GB. None can run alongside the current 8.1 GB CatBoost job under the 13,681 MB limit.
- **Expected effect** (`results.md`): about −0.04 s on the board [X].
