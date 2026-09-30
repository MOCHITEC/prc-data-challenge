# CatBoost as a third matched-lane blend member on hybrid_H2_plainy (2026-09-30)

**Status: PRE-REGISTERED, not yet run.** Results are appended below §0 after the run. §0 is committed before any
CatBoost number on these features exists.

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
