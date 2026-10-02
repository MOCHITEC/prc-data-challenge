# Per-airport anchored LGB+L2 member (PA) in the matched lane of hybrid_blend_l2 (2026-10-02)

**Status: pre-registered, not run.** Results go in §1 onward. §0 is not edited after it is committed.

## 0. Pre-registration (written before any run)

### 0.1 Why, and what the public evidence is

`public-code-deep-dive-2026-09-30.md` §1.3 and §2.3 [S]: kind-mango's v89 added a per-airport LightGBM member
("one member per airport with the v85 columns and v76 parameters, one seed") and served
`anchored' = 0.5 · global anchored LGB + 0.5 · per-airport anchored LGB` (and the same for plain-y), for −0.22 s on
the board. The same note says their earlier RESULT 7 found per-airport models "NOT WORKING" on LightGBM. No
competitor code was opened for this note.

### 0.2 What we measured or decided before (quotes)

Searched `docs/` for "per-airport model", "per airport model" and "airport-specific". **We have never run a
per-airport model.** What exists:

- `plans/P11-per-airport-models.md` (one global fit plus one fit per airport on model A, blended 50/50; configs
  `p11_*.yaml`, flag `airport_models`) was **withdrawn before execution** (commit 08c2a74): *"`results/model-a-ceiling.md`
  §6 measured this plan's mechanism independently and from the other side, and it came back at **0.3 s** … That sits
  below this plan's own pre-registered point of −0.7 s and inside its declared failure band"*, and
  *"`results/airport-qualified-levels.md` independently closed §7.4's first prerequisite control at −0.11 s … all four
  label-derived encoding tables are already keyed on `ADEP_mvt`"*.
- `model-a-ceiling.md` §6: *"The best cell in the whole table recovers 0.27 s and the shuffled control recovers
  0.10 s"*; *"A cell-mean correction is exactly what a target encoding, a per-group specialist, a per-group offset
  column or a per-airport recalibration would buy, so that whole family is closed at 0.3 s. It cannot see structure
  that varies continuously inside a cell"*. (C1 ADEP alone: 213.127 → 213.012.)
- `SUMMARY.md`: *"NASA pushback winners all used per-airport models | The only controlled comparison has global
  ahead (10.7232 vs 10.7686), better at 8 of 10 airports"*.
- `plans/P7-model-capacity.md` / `07-repo-phoenix-ops.md`: *"Airport-specific blend weights | 323.712 vs 323.377 |
  rejected"* (per-airport blend weights, not per-airport models).

What is different here: P11 blended inside model A of one recipe; this tests a separate member blended into the
three-member ensemble that is on the board, which is what v89 did. The ceiling note bounds a per-airport *constant*;
a per-airport member can also change interactions and adds model diversity, which the ceiling does not measure.
**Prior: |effect| ≤ 0.3 s; a REJECT or an inconclusive result is the expected outcome.**

### 0.3 Member PA and how its fits are organised

Recipe `configs/main_flseq_osm_nolirf_l2_pa.yaml`, exp id **`main_flseq_osm_nolirf_l2_pa-85f815edf4`** =
`configs/main_flseq_osm_nolirf_l2.yaml` (`ef204b44ff`) plus exactly three keys: `airport_models: true`,
`airport_blend_weight: 1.0`, `airport_specialist_only: true` (new flag, default off, out of the hash at off).
Same features, same hyperparameters (lr 0.05, 127 leaves, `min_data_in_leaf` 50, `feature_fraction` 0.8, bagging
0.8 / 1, `lambda_l2` 1, 2,000-round cap, early stopping 100, 3 seeds), same folds (no `folds:` key in either recipe,
so the four production folds jan, jul, jul_fwd, forward; pooled OOF = jan, jul, forward = 672,428 rows, jul_fwd is
shadowed exactly as in the pooled run). The OOF is row-for-row comparable with `ef204b44ff`.

Per fold (and once more for the full fit), `predict_fold` does exactly what the pooled run does up to model A:
label-derived encodings fitted on the fold's whole training frame (both lanes, all airports), fold features, the
same visible columns. Then, for model A only:

1. The matched lane (`mvt_minus_aobt3` non-null) training rows are split by `ADEP_mvt` into 10 groups (sorted
   order). **No global model-A fit is made.**
2. For each airport: early-stopping fit on that airport's matched training rows minus its latest training month
   (the inner holdout, as in the pooled fit), rounds scaled by rows and capped at 2,000; then 3 refits (seeds
   42/43/44 as in the pooled run) on all of that airport's matched training rows; their mean predicts that
   airport's matched target rows. Anchored target (`LABEL − mvt_minus_aobt3`) and add-back as in the pooled run.
3. `ADEP_mvt` is dropped from a group's columns when it is constant on that group's training rows (always, with the
   `adep` key). No other column is dropped; columns that become constant within an airport (e.g. `apt_med`) stay,
   and LightGBM ignores them.
4. Every matched target row must be predicted by its airport's fit; a row left without one is an error, not a
   fallback. Smallest matched training group: LSZH, 63,450 rows (jul_fwd); 112,151–132,616 on the pooled folds and
   the full fit. The 20,000-row floor (`airport_min_group_rows`) therefore never binds.

Model B (unmatched lane) is the pooled run's, unchanged, so its predictions are expected byte-identical to
`ef204b44ff`'s. 4 folds × 10 airports × (1 early-stopping fit + 3 refits) = 160 LightGBM fits in CV; 40 in the full
fit. Each fold's predictions are checkpointed to disk (`--fold-cache`), so a restart resumes at the next fold.

### 0.4 Candidates

Matched lane = `mvt_minus_aobt3` non-null; unmatched lane = `main_flseq-8615e6a194`'s model B in every candidate.
Members: a2 = anchored LGB+L2 `main_flseq_osm_nolirf_l2-ef204b44ff`; pa = PA `main_flseq_osm_nolirf_l2_pa-85f815edf4`;
p2 = plain-y LGB+L2 `main_flseq_osm_nolirf_plainy_l2-c74a27239b`; c2 = anchored CatBoost+L2
`main_flseq_osm_nolirf_cb_l2-4aa341b206`. All weights fixed; nothing is tuned.

| name | matched lane | role |
|---|---|---|
| `hybrid_blend_l2_P` (base) | (a2 + p2 + c2) / 3 | the OOF of `outputs/hybrid_blend_l2.parquet`; rebuild must reproduce `blend-l2.md`'s 269.98 |
| **`hybrid_blend_l2_pa`** (primary) | ((0.5·a2 + 0.5·pa) + p2 + c2) / 3 | the decision |
| `hybrid_blend_l2_pa4` (secondary) | (a2 + pa + p2 + c2) / 4 | report only |
| `member_a2` vs `member_pa` (secondary) | a2 alone vs pa alone | report only |

`scripts/blend_pa.py oof --scratch <dir> --pa main_flseq_osm_nolirf_l2_pa-85f815edf4` writes these OOF files to a
scratch directory (not `experiments/`).

### 0.5 Decision rule

`python -m prc.compare hybrid_blend_l2_P hybrid_blend_l2_pa --work-dir data/work_flseq --exp-dir <scratch>`.
`prc.compare.verdict`: the full CI (K=0) and the drop-200 CI must both exclude zero on the same side; **ADOPT** needs
the improving side, REJECT is both on the worse side, anything else keeps the base. Reported for the primary: the
K = 0 / 20 / 200 / 2000 curve, the month bootstrap, and (`scripts/blend_pa.py detail`) the lane split (matched /
unmatched / all except LIRF), per calendar month and per airport. Expected and checked: unmatched lane +0.00.

Secondary, report only, never the decision: `member_a2` vs `member_pa` and `hybrid_blend_l2_P` vs
`hybrid_blend_l2_pa4`, each with the K curve and verdict, plus per-airport member RMSE on the matched lane.

### 0.6 Output (only if ADOPT)

Run the PA full fit (`prc.pipeline --config configs/main_flseq_osm_nolirf_l2_pa.yaml --work-dir data/work_flseq
--skip-cv` → `outputs/main_flseq_osm_nolirf_l2_pa.parquet`), then `scripts/blend_pa.py full` →
**`outputs/hybrid_blend_l2_pa.parquet`**: matched rows `rint(((0.5·a2 + 0.5·pa) + p2 + c2) / 3)` from the four Int32
full-fit files, unmatched rows from `outputs/main_flseq.parquet`. Validate: 344,841 rows, schema, ids and order equal
to `data/raw/submitting.parquet`, no nulls, no negatives, unmatched rows equal to `outputs/hybrid_blend_l2.parquet`.
If not ADOPT: no full fit and no file; `hybrid_blend_l2` stays. Nothing is uploaded or submitted, and no ADS-B
residual is applied.

### 0.7 Runs and resources

`.venv/bin/python -m prc.pipeline --config configs/main_flseq_osm_nolirf_l2_pa.yaml --work-dir data/work_flseq
--skip-predict --fold-cache logs/per_airport/fold_cache` (feature cache reused), alone on the box, logged to
`logs/per_airport/` (git-ignored), exit code checked. Peak cgroup `total_rss` sampled every 10 s from `memory.stat`
of the cgroup `/proc/self/cgroup` names and of its parent (limit 13,681 MB). **If `total_rss` exceeds 13,000 MB the
run is killed and reported, not retried.** Reference: the pooled anchored+L2 CV took 4,134 s and peaked at 11,231 MB.

### 0.8 Code and tests (committed with this section)

- `src/prc/train.py`: `airport_specialist_only` (default off, out of the hash at off; refused unless
  `airport_models` and weight 1.0). On, `_fit_grouped` skips the global fit, fits groups only, drops the constant
  group key, and errors on any unpredicted matched row; `rounds_matched` becomes a per-airport dict.
- `src/prc/train.py` `run_cv(fold_cache=None)` and `prc.pipeline --fold-cache DIR` (default off: nothing read or
  written): per-fold checkpoint `<DIR>/<exp_id>/<fold>.parquet|.json`, reused on a re-run of the same id.
- `tests/test_per_airport_member.py` (8 tests): every existing recipe in `configs/` (124) keeps its id and resolved
  model A / model B LightGBM parameters byte for byte as at f3ab74f; the default, P11 w=0.5 and P11 w=1.0 paths give
  predictions byte-identical to f3ab74f on a fixed synthetic fold; the PA recipe is its base plus the three keys;
  model B is byte-identical and model A moves; every airport fitted with `ADEP_mvt` dropped; the refusal paths; a
  resumed `run_cv` gives the same OOF and record; the blend script's members and formulas match §0.4.
