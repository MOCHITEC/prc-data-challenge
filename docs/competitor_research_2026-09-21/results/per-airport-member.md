# Per-airport anchored LGB+L2 member (PA) in the matched lane of hybrid_blend_l2 (2026-10-02)

**Verdict (prc.compare.verdict, 672,428 pooled OOF rows): `hybrid_blend_l2` → `hybrid_blend_l2_pa` full −0.77 s
[−0.92, −0.64], K=200 −0.96 s [−1.05, −0.88] → ADOPT.** Month bootstrap −0.77 [−0.96, −0.69]. Matched lane −1.02
[−1.13, −0.92], unmatched +0.00. All ten airports and all four months improve. Secondary (report only): PA alone vs
a2 +0.00 [−0.54, +0.52], K=200 −0.07 [−0.58, +0.49] (no difference); 4-member mean −0.91 [−1.11, −0.74], K=200 −1.11
[−1.25, −0.98]. `outputs/hybrid_blend_l2_pa.parquet` is built and validated. Nothing was uploaded or submitted.

§0 was committed as cc8b68a before the run started (01:40:47 UTC) and has not been edited since, except this
status paragraph. Results are in §1–§4.

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

## 1. Run (CV)

`logs/per_airport/run.sh pa_cv --config configs/main_flseq_osm_nolirf_l2_pa.yaml --work-dir data/work_flseq
--skip-predict --fold-cache logs/per_airport/fold_cache` (feature cache reused), alone on the box, started
01:40:47 UTC after §0 was committed as cc8b68a. Exit 0, not killed, no resume needed.

| run | exp id | wall s (train s) | peak total_rss (own cgroup / parent) |
|---|---|---:|---:|
| PA, CV only | `main_flseq_osm_nolirf_l2_pa-85f815edf4` | 3,636 (3,561) | 8,909 MB / 9,119 MB |

Pooled anchored+L2 for reference: 4,134 s, 11,231 MB. With no global fit, the peak fell by 2.3 GB.

Per fold, pooled a2 → PA (overall RMSE / matched RMSE / matched rounds):

| fold | overall | matched | rounds_matched (PA: min–max over the 10 airports) |
|---|---|---|---|
| jan | 342.54 → 341.68 | 201.51 → 200.04 | 1,269 → 189–774 |
| jul | 297.01 → 298.94 | 241.45 → 243.87 | 1,208 → 225–779 |
| jul_fwd (shadowed) | 308.63 → 309.75 | 250.19 → 251.61 | 903 → 178–892 |
| forward | 217.16 → 216.25 | 196.73 → 195.71 | 978 → 320–933 |

`ADEP_mvt` was dropped in all 40 airport fits. Model B untouched: `rounds_unmatched` equal on every fold
(52 / 43 / 92 / 65) and all 8,378 pooled unmatched OOF predictions exactly equal to `ef204b44ff`'s. Overall
273.57 / matched 211.307 for both runs. On the 664,050 matched rows, a2 and PA differ by RMS 70.39 s
(correlation 0.984).

## 2. Comparisons (`prc.compare <A> <B> --work-dir data/work_flseq --exp-dir logs/per_airport/scratch`)

`scripts/blend_pa.py oof` reproduces the base RMSE 269.98 (`blend-l2.md`). Hybrid RMSE: base 269.98, primary
269.21, 4-member 269.07. Unmatched lane +0.00 [+0.00, +0.00] in every pair.

| A → B | K=0 | K=20 | K=200 | K=2000 | month bootstrap | verdict |
|---|---|---|---|---|---|---|
| **base → `hybrid_blend_l2_pa` (decision)** | **−0.77 [−0.92, −0.64]** | −0.93 [−1.02, −0.83] | **−0.96 [−1.05, −0.88]** | −1.01 [−1.06, −0.95] | −0.77 [−0.96, −0.69] | **ADOPT** |
| base → `hybrid_blend_l2_pa4` (secondary) | −0.91 [−1.11, −0.74] | −1.09 [−1.23, −0.95] | −1.11 [−1.25, −0.98] | −1.24 [−1.32, −1.15] | −0.91 [−1.06, −0.78] | ADOPT (report only) |
| `member_a2` → `member_pa` (secondary) | +0.00 [−0.54, +0.52] | +0.00 [−0.60, +0.64] | −0.07 [−0.58, +0.49] | −1.39 [−1.67, −1.13] | +0.00 [−1.26, +1.28] | no difference (report only) |

**Primary, lane split:** matched −1.02 [−1.13, −0.92]; unmatched +0.00; all except LIRF −0.82 [−1.01, −0.66].

**Primary, per calendar month** (all rows / matched):

| month | all rows | matched |
|---|---|---|
| 2025-01 | −0.73 [−1.23, −0.51] | −1.26 [−1.46, −1.07] |
| 2025-07 | −0.68 [−0.88, −0.48] | −0.86 [−1.10, −0.64] |
| 2025-11 | −0.72 [−0.89, −0.57] | −0.83 [−0.98, −0.69] |
| 2025-12 | −1.13 [−1.30, −0.96] | −1.21 [−1.40, −1.04] |

**Per airport**, all rows, diff [CI95] (share of squared disagreement), primary and PA member alone:

| airport | n | base → primary | a2 → PA (member, info) |
|---|---:|---|---|
| LIRF | 49,924 | −0.71 [−1.05, −0.37] (17.0 %) | +1.67 [−0.52, +3.78] |
| EGLL | 79,633 | −0.66 [−0.97, −0.36] (14.9 %) | +1.84 [−0.05, +3.89] |
| LFPG | 78,036 | −0.39 [−0.82, −0.20] (13.9 %) | +1.50 [+0.46, +3.59] |
| LTFM | 91,338 | −1.08 [−1.27, −0.87] (12.5 %) | −2.66 [−3.96, −1.10] |
| EHAM | 79,639 | −0.82 [−1.02, −0.63] (8.2 %) | +0.22 [−1.22, +1.92] |
| EDDF | 72,276 | −1.13 [−1.51, −0.85] (8.2 %) | −0.70 [−2.43, +0.75] |
| EDDM | 51,688 | −1.86 [−2.31, −1.48] (8.0 %) | −3.54 [−6.63, −1.23] |
| LEMD | 70,732 | −1.20 [−1.39, −1.03] (6.6 %) | −1.81 [−2.70, −0.95] |
| LSZH | 43,009 | −1.22 [−1.49, −0.95] (5.5 %) | −1.13 [−2.75, +0.71] |
| LEBL | 56,153 | −0.72 [−0.87, −0.56] (5.2 %) | −0.29 [−1.18, +0.71] |

Primary: all ten airports and all four months improve, every CI below zero.

**Member alone (secondary):** by month a2 → PA is −0.86 (Jan), +1.95 (Jul), +0.11 (Nov), −1.95 (Dec) on all rows;
the pooled total is +0.00. PA is not better than a2 on its own; it is a different model of equal accuracy, and the
gain is from averaging the two.

**Reading.** The prior (≤ 0.3 s, from the cell-mean ceiling) was wrong in size for this form: the ceiling bounds a
per-airport constant, and PA's value here is diversity, not a better per-airport level. The PA member's matched
predictions disagree with a2 by RMS 70 s at equal RMSE, so a 1/6 weight is enough for −1.02 s on the matched lane.
The 4-member mean is a little better (−0.91 s) but it is report-only and was not the pre-registered primary.
Caveats: on the board expect less than the OOF gain (v89's board gain was −0.22 s), and PA uses the same
early-stopping holdout month per airport as the pooled fit, so its rounds were chosen on 1/10 of the rows.

## 3. Full fit and file (ADOPT, so §0.6 applies)

`logs/per_airport/run.sh pa_full --config configs/main_flseq_osm_nolirf_l2_pa.yaml --work-dir data/work_flseq
--skip-cv`: exit 0, wall 971 s, peak total_rss 7,688 MB (parent 7,896 MB). 10 airport fits on 132,616–269,320
matched rows, rounds 220–717, `ADEP_mvt` dropped in each; model B 45 rounds. In
`outputs/main_flseq_osm_nolirf_l2_pa.parquet` the 5,290 unmatched rows equal `outputs/main_flseq_osm_nolirf_l2.parquet`'s
exactly; matched rows differ from it by RMS 73.69 s. `prc.validate_submission` OK.

`outputs/hybrid_blend_l2_pa.parquet` (`scripts/blend_pa.py full`):
- 344,841 rows (= template): 339,551 matched, 5,290 unmatched.
- Matched rows `rint(((0.5·a2 + 0.5·pa) + p2 + c2) / 3)` from `outputs/main_flseq_osm_nolirf_l2.parquet`,
  `outputs/main_flseq_osm_nolirf_l2_pa.parquet`, `outputs/main_flseq_osm_nolirf_plainy_l2.parquet`,
  `outputs/main_flseq_osm_nolirf_cb_l2.parquet`; unmatched rows from `outputs/main_flseq.parquet`.
- Schema `{MVT_ID_mvt: double, TAXITIME_SEC_mvt: int32}` equals the template; ids and order equal the template.
- 0 nulls, 0 negatives, min 0, max 113,641. `prc.validate_submission` OK (only the usual > 10,800 s warning).
- Against `outputs/hybrid_blend_l2.parquet`: differs on 317,589 rows, all matched (unmatched untouched), max |d|
  494 s, RMS 12.19 s.

Nothing was uploaded or submitted, and no ADS-B residual was applied (v28 = `hybrid_blend_l2` + the learned
ADS-B residual; applying it to this base is a separate step).

## 4. Code and tests

As §0.8. Full suite **241 passed** (`logs/per_airport/pytest.txt`), run before the CV run.

Logs (git-ignored): `logs/per_airport/` — `run.sh`, `pa_cv.{log,mem,status}`, `pa_full.{log,mem,status}`,
`fold_cache/`, `scratch/` (hybrid OOF files), `oof.txt`, `compare_{primary,pa4,member}.txt`,
`detail_{primary,member}.txt`, `full.txt`, `validate_*.txt`, `pytest.txt`.
