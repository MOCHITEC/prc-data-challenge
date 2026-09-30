# CatBoost as a third matched-lane blend member on hybrid_H2_plainy (2026-09-30)

**Verdict (prc.compare.verdict, `hybrid_H2_plainy` → `hybrid_plainy_cb`, equal weights 1/3, 672,428 pooled OOF
rows): full −1.75 s [−2.08, −1.47], K=200 −2.21 s [−2.40, −2.02] → ADOPT.** The month bootstrap reads −1.75
[−2.13, −1.38]. CatBoost alone is worse than the anchored LightGBM by +0.94 s [+0.49, +1.56]. The gain comes from
the blend. `outputs/hybrid_plainy_cb.parquet` is built and validated. Nothing was submitted.

§0 was committed as c6453f8 before the run started (run start 12:10:47 UTC).

## 0. Pre-registration (written before the run)

**Source of the idea.** `public-code-deep-dive-2026-09-30.md`: kind-mango uses CatBoost as a 50/50 blend member
(their v67, −2.87 s live). No competitor code was opened. Our own CatBoost history is P10
(`P10-catboost-measured.md`, ON HOLD): CatBoost was tested only as a *replacement* for LightGBM and lost lane A
by 4–6 s on `jul_fwd`. It was never tested as a blend member. A worse member can still help a blend if its errors
are not the same as the others'. The plain-y member showed this: +0.93 s alone, −0.78 s in the blend
(`plainy-blend.md`).

**Existing code.** `TrainConfig.model = "catboost"` (P10, `src/prc/train.py` `_fit_catboost`, `CB_DEFAULTS`) is
already in the tree, default `"lightgbm"`. `configs/cb_*.yaml` are **not** CatBoost: they are the clever-bell
feature-gate recipes (LightGBM). **No `src/` change is made here**, so every existing recipe keeps its id and its
predictions byte for byte. `tests/test_matched_target.py` checks the ids of all older recipes in `configs/`.

**Which P10 setting, and why.** P10 measured two CatBoost arms at full size. C+E (`configs/p10_cb_e.yaml`, with
the hand-rolled encodings) beat C−E (no encodings) on lane A on both folds: `jul_fwd` 256.868 vs 260.052, and
`forward` 198.540 vs 201.430. It is the best measured CatBoost setting. The ctr4 arm was never run, and no tuning
was measured. So the member uses C+E's family settings unchanged:
- `CB_DEFAULTS`: `learning_rate` 0.2, depth 6, SymmetricTree, `max_ctr_complexity` 1, `thread_count` 4,
  library-default `l2_leaf_reg` 3, `border_count` 254, MVS `subsample` 0.8.
- **Discrepancy, recorded:** `P10-catboost-measured.md` says the arms ran at `learning_rate` 0.1. The records
  in `experiments/experiments.jsonl` (`cb_matched.resolved.learning_rate` = 0.2 on every P10 CatBoost fold) and
  `CB_DEFAULTS` both say **0.2**. This run uses what was measured (0.2). So the note's pending "lr 0.2 re-run"
  was in effect already done: `forward` early-stopped at a best count of 4,090 rounds and ran 4,574 in the refit,
  under the 5,000 cap.
- Encodings on and `es_holdout_encodings` false, as in both P10 C+E and `main_flseq_osm_nolirf`.
- `num_rounds` 5,000 and `early_stopping` 100, P10's model-A cap. At main's 2,000 cap, CatBoost at lr 0.2 would
  be cut short on `forward` (best 4,090 in P10).
- `n_seeds` 1, as in every P10 arm. This is the only CatBoost seed count with measured numbers, and 3 seeds at up
  to 5,000 rounds is about 3× the wall clock. The two LightGBM members keep their 3 seeds. CatBoost's single-seed
  noise is large (P10 §6: 1.44 s lane A between seeds 42 and 43). Here it is one third of a blend, not the model.

**The member**, `configs/main_flseq_osm_nolirf_cb.yaml`. It is `configs/main_flseq_osm_nolirf.yaml` (the anchored
LightGBM member) plus `model: catboost`, `num_rounds: 5000`, `n_seeds: 1` and `folds: [jan, jul, forward]`, with a
new name. It uses the same features, categoricals, OSM distance (nulled at LIRF), inner holdout, round-scaling
rule, zero floor and LOBT projection. Its target is the **anchored** target, the same as the anchored LightGBM:
`g = TAXITIME_SEC_mvt − mvt_minus_aobt3`, predicted as `mvt_minus_aobt3 + ĝ`. `jul_fwd` is left out because it
is not in the pooled OOF (jul shadows it: pooled = jan, jul, forward, 672,428 rows). Its model B (CatBoost) is
not used. `tests/test_catboost_blend.py` checks that the recipe is exactly its base plus those keys.

**Primary candidate (the decision).** `hybrid_plainy_cb`:
- Matched lane (`mvt_minus_aobt3` non-null): the **equal-weight mean** `(anchored LGB + plain LGB + anchored CB) / 3`.
  - anchored LGB: `main_flseq_osm_nolirf-926741d779`
  - plain LGB: `main_flseq_osm_nolirf_plainy-b971f4b81b`
  - anchored CB: the new run
- Unmatched lane unchanged: `main_flseq-8615e6a194`'s model B.
- Baseline: `hybrid_H2_plainy`, which is 0.5 anchored + 0.5 plain on the matched lane and the same unmatched lane.
  It is rebuilt from the stored OOF by `scripts/catboost_blend.py oof`. A dry run reproduces its RMSE of 272.05.

**Comparison.** `prc.compare hybrid_H2_plainy hybrid_plainy_cb --work-dir data/work_flseq` on the 672,428 pooled
OOF rows reports:
- full, K = 20 / 200 / 2000, and the month bootstrap
- lane split, per calendar month, per airport, and all rows except LIRF (`scripts/catboost_blend.py detail`)

**Decision rule** (`prc.compare.verdict`, primary only). The full CI and the drop-200 CI must both exclude zero
on the same side. ADOPT needs the improving side. REJECT is the worsening side. Anything else is not adopted.

**Secondary (report only, not the decision).** Leave-one-fold-out weights:
- For each pooled fold, least-squares weights on (anchored, plain, CB) are fitted on the matched OOF rows of the
  other two folds. They are constrained to sum to 1, with sign free, and applied to the held-out fold.
- Reported: per-fold weights, weights fitted on all three folds, and the resulting hybrid vs `hybrid_H2_plainy`.
- Also for context: CatBoost alone on the matched lane vs anchored LGB alone, and member error correlations.

**Output, regardless of verdict.** `outputs/hybrid_plainy_cb.parquet` (`scripts/catboost_blend.py full`):
- Matched rows: `rint((a + p + c) / 3)` from the three Int32 full-fit files.
- Unmatched rows: `outputs/main_flseq.parquet`.
- Validated for rows (344,841), schema, ids and order against the template, nulls, and negatives.
- Not submitted.

**Resources.** P10 C+E peaked at 11.86 GB against LightGBM's 10.90 GB on the same recipe. The last LightGBM run on
these features peaked at 10.49 GB. The run waits until no `adsb_learn_` process is left and cgroup `total_rss` is
below 2,500 MB. `total_rss` is sampled every 30 s during the run and the peak is recorded.

## 1. Run

`.venv/bin/python -m prc.pipeline --config configs/main_flseq_osm_nolirf_cb.yaml --work-dir data/work_flseq`. The
experiment id is `main_flseq_osm_nolirf_cb-6297d62f5f`.
- It exited with 0. Wall clock was **14,242 s** (CV 10,310 s, then the full fit).
- Peak cgroup `total_rss` was **10,697 MB** (`logs/peak_cb.txt`, at 15:29 during the full fit), against a
  13,681 MB limit. LightGBM plainy peaked at 10,489 MB.
- No ADS-B job was running.
- One incident: a duplicate launch at 12:30:34 was killed after 12 s during feature load. The original run was
  never interrupted (`logs/run_cb_status.txt`).
- Resolved on every fold: `learning_rate` 0.2, `max_ctr_complexity` 1, depth 6. No fold hit the 5,000 cap.

| fold | matched RMSE, anchored LGB | plain LGB | **anchored CB** | CB rounds (ES best → refit) |
|---|---:|---:|---:|---|
| jan | 201.52 | 204.36 | 203.36 | 2,482 → 2,715 |
| jul | 241.73 | 242.49 | 244.47 | 3,736 → 4,094 |
| forward | 196.93 | 197.70 | **196.82** | 4,246 → 4,748 |
| pooled matched | 211.49 | 212.71 | 212.73 | full fit 3,659 → 3,975 |

Matched-lane error correlations: anchored–plain 0.9695, anchored–CB **0.9313**, plain–CB **0.9263**. CatBoost is
the least correlated of the three members. That is why it helps the blend while losing on its own. CatBoost alone
vs anchored LGB alone (both with the same unmatched lane): K=0 +0.94 [+0.49, +1.56], K=200 +1.02 [+0.47, +1.57],
**REJECT** as a replacement, consistent with P10.

## 2. Blend vs hybrid_H2_plainy (the decision)

`scripts/catboost_blend.py oof --cb main_flseq_osm_nolirf_cb-6297d62f5f` rebuilds the baseline. It reproduces
`plainy-blend.md`'s 272.05. `prc.compare hybrid_H2_plainy <B> --work-dir data/work_flseq`:

| B | K=0 | K=20 | K=200 | K=2000 | month bootstrap | verdict |
|---|---|---|---|---|---|---|
| **equal 1/3 (primary)** | **−1.75 [−2.08, −1.47]** | −2.09 [−2.28, −1.90] | **−2.21 [−2.40, −2.02]** | −1.97 [−2.09, −1.87] | −1.75 [−2.13, −1.38] | **ADOPT** |
| LOFO weights (secondary) | −1.81 [−2.20, −1.51] | −2.16 [−2.41, −1.92] | −2.31 [−2.56, −2.07] | −2.06 [−2.21, −1.92] | −1.81 [−2.21, −1.43] | (ADOPT; not the decision) |

RMSE: baseline 272.05, primary 270.30, LOFO 270.24. The K=200 drop removes 147 unmatched and 53 matched rows.

- **Lane split (primary).** Matched 210.48 → 208.19: −2.29 [−2.51, −2.08]. Unmatched +0.00 (identical by
  construction). All rows except LIRF −1.80 [−2.18, −1.46].
- **Per calendar month**, all rows / matched. Every month improves.
  - 2025-01: −1.19 [−2.00, −0.81] / −2.04 [−2.38, −1.69]
  - 2025-07: −1.92 [−2.31, −1.58] / −2.41 [−2.84, −1.94]
  - 2025-11: −1.94 [−2.31, −1.55] / −2.24 [−2.61, −1.87]
  - 2025-12: −2.27 [−2.59, −1.94] / −2.45 [−2.77, −2.10]
- **Per airport**, all rows, with the share of squared disagreement. All ten CIs exclude zero on the improving side.

| airport | n | share | diff | CI95 |
|---|---:|---:|---:|---|
| LIRF | 49,924 | 24.4 % | −1.88 | [−2.69, −1.06] |
| LFPG | 78,036 | 15.6 % | −1.48 | [−2.85, −0.88] |
| LTFM | 91,338 | 14.9 % | −2.75 | [−3.14, −2.39] |
| EGLL | 79,633 | 12.5 % | −2.40 | [−2.89, −1.87] |
| EHAM | 79,639 | 7.8 % | −1.45 | [−1.81, −1.11] |
| EDDF | 72,276 | 6.2 % | −1.77 | [−2.14, −1.41] |
| LEMD | 70,732 | 5.4 % | −1.67 | [−1.98, −1.35] |
| EDDM | 51,688 | 5.4 % | −2.18 | [−2.70, −1.64] |
| LEBL | 56,153 | 4.1 % | −1.33 | [−1.61, −1.06] |
| LSZH | 43,009 | 3.8 % | −2.41 | [−3.07, −1.85] |

- **Secondary: LOFO weights (anchored, plain, CB), sum to 1.** Each is fitted on the other two folds:
  - held-out jan: 0.326 / 0.254 / 0.420
  - held-out jul: 0.343 / 0.221 / 0.435
  - held-out forward: 0.386 / 0.228 / 0.387
  - All three folds: 0.350 / 0.236 / 0.415.
  - CatBoost gets the largest weight in every fold. The optimum is shallow: LOFO beats equal weights by 0.06 s.
    It does not enter the decision, and the file uses 1/3.

**Reading.** The rule passes by a wide margin: the K=200 upper end is −2.02. The gain is about 2.2× the plain-y
blend's (−0.78). It is spread evenly over months and airports, and it is a matched-lane effect at normal error
sizes (K=0 / 200 / 2000 all −1.75 to −2.21). The weights were fixed before the run, so the gain was not selected
on these rows. Caveats:
- The CatBoost member is single-seed. P10 measured a 1.44 s lane-A seed draw for CatBoost alone. A different seed
  would move the member, and the blend by less.
- The unmatched lane (v16's model B) is inherited from H2 and was post hoc there.
- On the board, expect less than the OOF gain.

## 3. File

`outputs/hybrid_plainy_cb.parquet` (`scripts/catboost_blend.py full`) has 344,841 rows: 339,551 matched and 5,290
unmatched.
- Matched rows are `rint((a + p + c) / 3)` from the Int32 full-fit files `main_flseq_osm_nolirf`,
  `main_flseq_osm_nolirf_plainy` and `main_flseq_osm_nolirf_cb`. Unmatched rows come from
  `outputs/main_flseq.parquet`.
- The schema `{MVT_ID_mvt: double, TAXITIME_SEC_mvt: int32}` equals the template, and the ids and their order equal
  the template.
- It has 0 nulls and 0 negatives, with min 0 and max 113,641. `prc.validate_submission` prints OK, with only the
  usual >10,800 s warning. `outputs/main_flseq_osm_nolirf_cb.parquet` also validates OK.
- Against `outputs/hybrid_H2_plainy.parquet`, it differs on 329,480 rows, all matched (unmatched untouched), with
  max |d| 970 s and RMS 29.02 s.
- Member differences on matched rows: CB − anchored RMS 90.50 s, CB − plain 95.68 s, anchored − plain 62.48 s.
- **No ADS-B residual has been applied.** That model would need refitting on this base. Nothing was submitted.

## 4. Code and tests

- No `src/` change. The CatBoost path is P10's `TrainConfig.model`, default `"lightgbm"`, so every existing recipe's
  id and predictions are unchanged.
- `tests/test_matched_target.py` checks the ids of all older recipes. `tests/test_catboost_blend.py` has 3 tests:
  - the recipe is its base plus the P10 keys;
  - it resolves to P10 C+E's CatBoost settings with the anchored target, and the base id is still `926741d779`;
  - the sum-to-one weight fit recovers known weights.
- Full suite: **213 passed**.

Logs (git-ignored):
- `logs/main_flseq_osm_nolirf_cb.log`, `logs/run_cb_status.txt`, `logs/peak_cb.txt` and `logs/mem_cb.txt`
- `logs/cb_oof.txt` and `logs/compare_cb_*.txt`
- `logs/cb_full.txt`, `logs/cb_validate_*.txt` and `logs/cb_pytest.txt`
