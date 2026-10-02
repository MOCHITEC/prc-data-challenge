# Lever L2 (forward NM windows) in all three matched-lane members of hybrid_plainy_cb (2026-10-01)

**Verdict (prc.compare.verdict, 672,428 pooled OOF rows): P vs `hybrid_plainy_cb` full −0.32 s [−0.45, −0.20],
K=200 −0.41 s [−0.54, −0.27] → ADOPT, so by rule step 1 P is chosen.** Month bootstrap −0.32 [−0.38, −0.26].
Information only: F vs base −0.09 [−0.14, −0.06], K=200 −0.13 [−0.16, −0.09] (ADOPT); P vs F −0.22 [−0.35, −0.11],
K=200 −0.28 [−0.40, −0.15] (ADOPT). `outputs/hybrid_blend_l2.parquet` (P) is built and validated. Nothing was
uploaded or submitted, and no ADS-B residual was applied.

§0 was committed as c2a6d2f before the first run started (03:32:10 UTC) and has not been edited since. Results are
in §1–§4.

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

## 1. Runs

`.venv/bin/python -m prc.pipeline --config configs/<recipe>.yaml --work-dir data/work_flseq` (feature cache
reused), sequential, nothing else training. Peak cgroup `total_rss` sampled every 15 s (limit 13,681 MB).

| run | exp id | wall s (CV s) | peak total_rss |
|---|---|---:|---:|
| plain-y LGB + L2 | `main_flseq_osm_nolirf_plainy_l2-c74a27239b` | 5,151 (3,959) | 11,169 MB |
| CatBoost + L2 | `main_flseq_osm_nolirf_cb_l2-4aa341b206` | 16,638 (10,819) | 11,977 MB |
| anchored LGB + L2, full fit only (`--skip-cv`) | `main_flseq_osm_nolirf_l2` | 1,967 | 9,576 MB |

All three exited 0. Total wall clock 03:32:10 → 10:08:06 UTC (23,756 s). CatBoost+L2 peaked 1.28 GB above the
no-L2 CatBoost (10,697 MB); headroom 1.70 GB.

Per fold, overall RMSE (all rows), no-L2 → L2:
- plain-y: jan 344.20 → 344.07, jul 297.84 → 297.51, jul_fwd 309.44 → 309.26, forward 218.03 → 217.45; overall
  274.65 → 274.28. Matched rounds 955/1042/890/724 → 1239/1188/910/1247; full fit 1,049.
- CatBoost: jan 339.68 → 339.17, jul 304.82 → 304.58, forward 215.72 → 215.65; overall 274.64 → 274.39. Matched
  rounds 2715/4094/4748 → 4042/3437/4406; full fit 4,928 (under the 5,000 cap).
- Anchored+L2 full fit: 1,391 matched rounds.

Model B untouched: in each L2 member's full-fit file, the 5,290 unmatched rows equal its no-L2 file exactly
(joined by id). Matched rows differ from the no-L2 file by RMS 24.79 s (anchored), 26.97 s (plain-y), 63.07 s
(CatBoost).

Matched-lane member RMSE on the pooled scored rows (no-L2 → L2): anchored 211.49 → 211.31, plain-y 212.71 →
212.23, CatBoost 212.73 → 212.40.

## 2. Comparisons (`prc.compare <A> <B> --work-dir data/work_flseq --exp-dir <scratch>`)

`scripts/blend_l2.py oof` reproduces the base RMSE 270.30 (`catboost-blend.md`). Hybrid RMSE: base 270.30, F 270.21,
P 269.98. The K=200 drop removes the same kind of rows as before; the unmatched lane is identical by construction in
every pair (+0.00 [+0.00, +0.00]).

| A → B | K=0 | K=20 | K=200 | K=2000 | month bootstrap | verdict |
|---|---|---|---|---|---|---|
| **base → P (decision)** | **−0.32 [−0.45, −0.20]** | −0.38 [−0.51, −0.24] | **−0.41 [−0.54, −0.27]** | −0.43 [−0.53, −0.33] | −0.32 [−0.38, −0.26] | **ADOPT** |
| base → F (info) | −0.09 [−0.14, −0.06] | −0.11 [−0.15, −0.08] | −0.13 [−0.16, −0.09] | −0.15 [−0.18, −0.12] | −0.09 [−0.14, −0.05] | ADOPT |
| F → P (info) | −0.22 [−0.35, −0.11] | −0.27 [−0.39, −0.14] | −0.28 [−0.40, −0.15] | −0.27 [−0.36, −0.19] | −0.22 [−0.33, −0.12] | ADOPT |

**Lane split** (matched / unmatched / all except LIRF):
- base → P: −0.42 [−0.56, −0.29] / +0.00 / −0.35 [−0.50, −0.23]
- base → F: −0.12 [−0.16, −0.08] / +0.00 / −0.12 [−0.16, −0.08]
- F → P: −0.30 [−0.43, −0.17] / +0.00 / −0.23 [−0.35, −0.13]

**Per calendar month**, all rows (matched in brackets after the slash):

| month | base → P | base → F | F → P |
|---|---|---|---|
| 2025-01 | −0.36 [−0.67, −0.18] / −0.62 | −0.04 [−0.11, −0.00] / −0.08 | −0.31 [−0.60, −0.14] / −0.54 |
| 2025-07 | −0.25 [−0.53, +0.03] / −0.32 | −0.16 [−0.24, −0.09] / −0.20 | −0.09 [−0.34, +0.17] / −0.12 |
| 2025-11 | −0.29 [−0.49, −0.09] / −0.34 | −0.12 [−0.18, −0.06] / −0.14 | −0.17 [−0.35, +0.04] / −0.20 |
| 2025-12 | −0.43 [−0.67, −0.20] / −0.47 | −0.05 [−0.12, +0.02] / −0.06 | −0.38 [−0.60, −0.14] / −0.41 |

Every month's point estimate improves in all three comparisons.

**Per airport**, all rows, diff [CI95] (share of squared disagreement):

| airport | n | base → P | base → F | F → P |
|---|---:|---|---|---|
| EGLL | 79,633 | −0.95 [−1.46, −0.42] (19.0 %) | −0.42 [−0.52, −0.31] | −0.54 [−1.02, −0.02] |
| LFPG | 78,036 | −0.03 [−0.34, +0.24] (16.0 %) | −0.13 [−0.29, −0.05] | +0.10 [−0.13, +0.41] |
| LIRF | 49,924 | −0.21 [−0.70, +0.20] (15.1 %) | +0.01 [−0.12, +0.15] | −0.22 [−0.69, +0.15] |
| LTFM | 91,338 | −0.58 [−0.91, −0.29] (14.3 %) | +0.02 [−0.08, +0.12] | −0.61 [−0.91, −0.33] |
| EHAM | 79,639 | −0.11 [−0.37, +0.18] (9.0 %) | −0.13 [−0.20, −0.04] | +0.02 [−0.23, +0.27] |
| EDDM | 51,688 | −1.08 [−1.61, −0.62] (7.1 %) | −0.12 [−0.26, +0.01] | −0.95 [−1.44, −0.52] |
| EDDF | 72,276 | −0.11 [−0.39, +0.21] (6.3 %) | −0.05 [−0.13, +0.03] | −0.06 [−0.35, +0.28] |
| LEMD | 70,732 | −0.22 [−0.42, −0.04] (4.7 %) | +0.01 [−0.06, +0.07] | −0.22 [−0.41, −0.05] |
| LEBL | 56,153 | −0.33 [−0.59, −0.07] (4.6 %) | −0.11 [−0.20, −0.02] | −0.21 [−0.44, +0.01] |
| LSZH | 43,009 | −0.31 [−0.70, +0.00] (3.7 %) | −0.13 [−0.24, −0.05] | −0.18 [−0.51, +0.11] |

base → P: all ten point estimates negative, six CIs exclude zero.

**Context (not the decision): each L2 member alone vs its no-L2 member** (matched lane swapped, same unmatched lane):
- plain-y: K=0 −0.37 [−0.49, −0.24], K=200 −0.46 [−0.60, −0.33], month −0.37 [−0.60, −0.23] → ADOPT.
- CatBoost: K=0 −0.25 [−0.56, +0.03], K=200 −0.34 [−0.68, +0.03], month −0.25 [−0.46, +0.10] → not adopted
  (CIs cross zero). Single seed; P10 measured a 1.44 s seed draw for CatBoost alone.

**Reading.** L2 helps the plain-y member (−0.37 s) more than the anchored member (−0.14 s in the lever note). Most of
P's gain over F comes from the plain-y and CatBoost swaps together (−0.22 s). P was fixed in advance with equal
weights, so it was not selected on these rows. Caveats: the CatBoost member is single-seed, and on the board expect
less than the OOF gain.

## 3. File

`outputs/hybrid_blend_l2.parquet` (`scripts/blend_l2.py full --which P`):
- 344,841 rows (= template): 339,551 matched, 5,290 unmatched.
- Matched rows are `rint((a2 + p2 + c2) / 3)` from `outputs/main_flseq_osm_nolirf_l2.parquet`,
  `outputs/main_flseq_osm_nolirf_plainy_l2.parquet` and `outputs/main_flseq_osm_nolirf_cb_l2.parquet`. Unmatched rows
  come from `outputs/main_flseq.parquet`.
- Schema `{MVT_ID_mvt: double, TAXITIME_SEC_mvt: int32}` equals the template; ids and order equal the template.
- 0 nulls, 0 negatives, min 0, max 113,641. `prc.validate_submission` prints OK (only the usual >10,800 s warning).
  The three member files also validate OK.
- Against `outputs/hybrid_plainy_cb.parquet`: differs on 328,676 rows, all matched (unmatched untouched), max |d|
  792 s, RMS 25.61 s.
- Check: `full --which base` rebuilt from the no-L2 members equals `outputs/hybrid_plainy_cb.parquet` on every row.

## 4. Code and tests

- No `src/` change. Two recipes (`configs/main_flseq_osm_nolirf_plainy_l2.yaml`, `configs/main_flseq_osm_nolirf_cb_l2.yaml`),
  `scripts/blend_l2.py`, and `tests/test_blend_l2.py` (2 tests: each recipe is its base plus `fe_fwd_nm: true`, base
  ids unchanged; the script's member sets match §0).
- Full suite: **227 passed**.

Logs (git-ignored): `logs/blend_l2/` — `runner.log`, the three run logs, `mem_*.txt`, `oof.txt`, `compare_*.txt`,
`detail_*.txt`, `full_P.txt`, `full_base_rebuild.txt`, `validate_*.txt`, `pytest.txt`.
