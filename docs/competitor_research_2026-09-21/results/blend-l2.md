# Lever L2 (forward NM windows) in all three matched-lane members of hybrid_plainy_cb (2026-10-01)

**Status: pre-registered. No new run has started and no P or F number has been computed.** Results follow in §5
onward once the runs are done.

## 0. Pre-registration (written before any run)

**Background.** `feature-levers-L1-L4.md` adopted L2 (`fe_fwd_nm: true`, eight forward-window columns of the
peers' NM clock) in the anchored LightGBM member: `main_flseq_osm_nolirf` → `main_flseq_osm_nolirf_l2`, full
−0.14 [−0.25, −0.05], K=200 −0.20 [−0.31, −0.10] (commits 81d52da, 2199361, 6875895). L2 in model B was
rejected. Its effect on the plain-y LightGBM and the CatBoost member was not measured. The adopted base is
`outputs/hybrid_plainy_cb.parquet` (`catboost-blend.md`): matched lane = equal-weight mean of the anchored LGB
(`main_flseq_osm_nolirf-926741d779`), the plain-y LGB (`main_flseq_osm_nolirf_plainy-b971f4b81b`) and the anchored
CatBoost (`main_flseq_osm_nolirf_cb-6297d62f5f`); unmatched lane = `main_flseq-8615e6a194`'s model B.

**Existing outputs, checked before writing this.**
- The anchored+L2 OOF exists: `experiments/oof_main_flseq_osm_nolirf_l2-ef204b44ff.parquet` (four folds,
  `--skip-predict`).
- **The anchored+L2 full-fit file does not exist.** The lever runs used `--skip-predict` (§6 of the lever note),
  and no `outputs/main_flseq_osm_nolirf_l2.parquet` or other ef204b44ff artefact is on disk. Both P and F need it
  for the full-fit file. It is produced by `prc.pipeline --config configs/main_flseq_osm_nolirf_l2.yaml
  --work-dir data/work_flseq --skip-cv`, the same recipe, unchanged: the full fit (`prc.predict.predict_ranking`)
  does not read the CV record, so it is the file the original run would have written without `--skip-predict`.
  It does not change any OOF number.

**New members (recipes, no `src/` change).** Each is its base plus `fe_fwd_nm: true` and a new name, nothing
else (`tests/test_blend_l2.py`):
- `configs/main_flseq_osm_nolirf_plainy_l2.yaml` (id `main_flseq_osm_nolirf_plainy_l2-c74a27239b`): plain-y LGB,
  four folds and three seeds as in the plain-y run.
- `configs/main_flseq_osm_nolirf_cb_l2.yaml` (id `main_flseq_osm_nolirf_cb_l2-4aa341b206`): CatBoost with the
  same settings as before (P10 C+E: lr 0.2, depth 6, SymmetricTree, `max_ctr_complexity` 1, 5,000-round cap, early
  stopping 100, 1 seed, folds jan / jul / forward), anchored target.

**Candidates.** Matched lane = `mvt_minus_aobt3` non-null; all weights fixed at 1/3; unmatched lane unchanged
(`main_flseq-8615e6a194`'s model B) in every candidate.

| name | anchored LGB | plain-y LGB | CatBoost |
|---|---|---|---|
| base `hybrid_plainy_cb` | `926741d779` | `b971f4b81b` | `6297d62f5f` |
| **P** (primary) `hybrid_blend_l2_P` | **+L2** `ef204b44ff` | **+L2** `c74a27239b` | **+L2** `4aa341b206` |
| **F** (fallback) `hybrid_blend_l2_F` | **+L2** `ef204b44ff` | `b971f4b81b` | `6297d62f5f` |

OOF hybrids are built by `scripts/blend_l2.py oof` in a scratch directory (not `experiments/`), on the 672,428
pooled OOF rows (jan, jul, forward), and compared with
`python -m prc.compare <A> <B> --work-dir data/work_flseq --exp-dir <scratch>`. The rebuilt base must reproduce
`catboost-blend.md`'s RMSE of 270.30.

**Decision rule, in this order** (`prc.compare.verdict`: the full CI (K=0) and the drop-200 CI must both exclude
zero on the same side; ADOPT needs the improving side):
1. P vs `hybrid_plainy_cb`. If ADOPT, **P is chosen**.
2. Otherwise F vs `hybrid_plainy_cb`. If ADOPT, **F is chosen**.
3. Otherwise **keep `hybrid_plainy_cb`**; no new file is chosen.

**Information only (not the decision):** F vs base when P is chosen, and P vs F. Also for context: each L2 member
alone vs its no-L2 member on the matched lane (plain-y and CatBoost), with the same unmatched lane.

**Reported for every comparison:** full and K = 20 / 200 / 2000 curve, the month bootstrap, the lane split
(matched / unmatched / all except LIRF), per calendar month and per airport (`scripts/blend_l2.py detail`).

**Output.** The chosen candidate's full-fit file is `outputs/hybrid_blend_l2.parquet` (`scripts/blend_l2.py full
--which P|F`): matched rows `rint((x1 + x2 + x3) / 3)` from the members' Int32 full-fit files, unmatched rows from
`outputs/main_flseq.parquet`. It is validated for rows (344,841), schema, ids and order against
`data/raw/submitting.parquet`, nulls and negatives. If nothing is chosen, no file is built. Nothing is uploaded or
submitted, and no ADS-B residual is applied.

**Runs, sequential, one at a time:** (1) plain-y+L2 (CV + full fit), (2) CatBoost+L2 (CV + full fit),
(3) anchored+L2 full fit (`--skip-cv`). F's OOF comparison needs no new run and may be done at any point.
Peak cgroup `total_rss` is sampled every 15 s from `memory.stat` of the parent of the cgroup `/proc/self/cgroup`
names (limit 13,681 MB).
