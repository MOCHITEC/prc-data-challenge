# Re-review of rejected and closed levers against the v35 base — inventory and OOF-only re-checks (2026-10-05)

**Status (2026-10-05): done.** R1 (LOFO weights) ADOPT −0.25 s [−0.34, −0.18], K=200 −0.29 [−0.38, −0.20]; R2 (pa as its own member) ADOPT −0.18 [−0.24, −0.13], K=200 −0.22 [−0.28, −0.17]; R3 no difference; R4 and R5 REJECT (worse). §0 and §1 were committed as a0743f5 before any re-check was computed. Results are in §2+.
§0–§1 will not be edited after that, except for this status line.

Scope: analysis only. No model was trained, nothing in `src/` or `configs/` was changed, nothing was submitted.

Current base: `hybrid_blend_l2_pa_opt_mlp` (v35's base). Its OOF is
`logs/mlp/scratch/oof_hybrid_blend_l2_pa_opt_mlp.parquet`, written by `scripts/blend_mlp.py oof`
(`mlp-member.md` §2: 672,428 pooled OOF rows, RMSE 267.57).
- Matched lane: `((0.5·a2o + 0.5·pao) + p2o + c2 + m4) / 4`.
- Unmatched lane: `main_flseq-8615e6a194` (model B).
- The OOF has **no** ADS-B residual and **no** positional overlay. Those two stages are test-time overlays with
  their own validation (`adsb-learned-residual.md` §12, `docs/posjoin_2026-10-05/`). So every re-check below is on
  the base before those overlays.

Member ids: a2o = `main_flseq_osm_nolirf_l2_opt-7244b3004f`, pao = `main_flseq_osm_nolirf_l2_pa_opt-8d54effc06`,
p2o = `main_flseq_osm_nolirf_plainy_l2_opt-dd8b8e2c7c`, c2 = `main_flseq_osm_nolirf_cb_l2-4aa341b206`,
m4 = `main_flseq_osm_nolirf_l2_mlp-2b9a3b9348`, u = `main_flseq-8615e6a194`.

## 0. Phase 1 — inventory of rejected or closed levers

Numbers are copied from the named notes. "K=0" is the full CI; "K=200" is the drop-200 CI. Negative = candidate
better. "Form" says how the lever was tested: feature (F), blend member (M), member replacement (R), override or
post-hoc overlay (O), blend weights (W). "OOF" gives the stored OOF of the arm, if it is still on disk and covers the
same pooled rows (jan / jul / forward folds; checked in §2.0 for the ones re-checked here). "Cost": (a) OOF-only, no
retraining; (b) needs retraining, with wall time from the note.

### 0.1 Levers with stored, re-usable OOF on the current four-fold row set

| # | lever (note) | measured (K=0 / K=200) | base it was measured on | form | how close | OOF on disk | cost |
|---|---|---|---|---|---|---|---|
| 1 | LOFO sum-to-one blend weights (`mlp-member.md` §3; also `plainy-blend.md`, `catboost-blend.md`) | vs `hybrid_blend_l2_pa_opt`: LOFO −1.00 [−1.23, −0.81] / −1.12 [−1.31, −0.95]; equal weights −0.75 [−0.93, −0.58] / −0.83 [−1.00, −0.67]. Earlier: CB LOFO −1.81 vs equal −1.75; plain-y LOFO −0.81 vs −0.78 | pa_opt (one step older than v35's base) | W, report-only secondary each time | both CIs excluded zero vs the older base; never compared directly to equal weights | `logs/mlp/scratch/oof_hybrid_blend_l2_pa_opt_mlp_lofo.parquet` | (a) |
| 2 | per-airport member as its own 1/N member, "pa4" (`per-airport-member.md` §2) | vs `hybrid_blend_l2`: pa4 −0.91 [−1.11, −0.74] / −1.11 [−1.25, −0.98]; chosen primary (0.5/0.5 inside M1) −0.77 [−0.92, −0.64] / −0.96 [−1.05, −0.88] | hybrid_blend_l2 (pre-retune, 3 members) | W / M, report-only secondary | both CIs excluded zero vs the older base; never compared directly to the primary | members a2o, pao, p2o, c2, m4 in `experiments/` | (a) |
| 3 | L2 forward NM windows in model B (`feature-levers-L1-L4.md` §7.5) | +0.35 [−0.22, +1.03] / **−0.21 [−0.37, −0.04]**; unmatched +4.97 [−4.16, +16.74]; LIRF unmatched +51.03 [−0.69, +108.02]; LTFM −19.92, EHAM −12.83, EDDM −16.22, LSZH −21.05, EDDF −6.64, LEMD −17.35 (CIs exclude zero) | `main_flseq` (v16; the unmatched lane is unchanged since) | F in model B, R of the unmatched lane | K=200 excluded zero; six airports' unmatched CIs exclude zero | `experiments/oof_main_flseq_l2b-e40f03b86d.parquet` | (a) as an unmatched member; (b) for other forms |
| 4 | OSM stand→runway distance in model B (`osm-on-flseq.md` §4, §7) | +0.95 [+0.28, +1.60] / −0.12 [−0.31, +0.12]; all except LIRF −0.51 [−0.75, −0.30]; LIRF unmatched +121.57 [+71.43, +173.15]; every other airport's unmatched diff negative (−0.22 to −36.54) | `main_flseq` | F in model B (matched part adopted as H2) | K=0 worse; non-LIRF subset strongly better | `experiments/oof_main_flseq_osm-bb204cd922.parquet` (pre-registered arm); `..._osm_nolirf-926741d779` (post-hoc LIRF-null arm) | (a) as an unmatched member |
| 5 | linear_tree on the anchored LGB+L2 (`linear-tree.md`) | +0.65 [+0.33, +1.02] / +0.69 [+0.38, +1.02]; matched +0.85 [+0.42, +1.32]; LTFM +5.01 [+3.77, +6.33] | anchored+L2 member alone (non-retuned) | R (single member) | worse on both CIs; never tried as an added member | `experiments/oof_main_flseq_osm_nolirf_l2_lt-a9fb149ab5.parquet` | (a) as an added member |
| 6 | A1 LIRF arrival-mirror fill (`A1-lirf-arrival-mirror.md`) | −0.09 [−0.21, +0.03] / −0.09 [−0.21, +0.04] on v16; −0.04 [−0.14, +0.06] / −0.07 [−0.18, +0.05] on osm_nolirf | v16 / osm_nolirf model A, no L2, untuned | F | both CIs cross zero | `oof_main_flseq_a1-c14b85d14f`, `oof_main_flseq_osm_nolirf_a1-e0daec598e` (non-L2, non-retuned anchored LGB; as a member it would be a near-copy of a pre-L2 M1) | (b): anchored opt CV ≈ 7,000 s, plus each other member it should enter |
| 7 | L1 `plan_nm_taxi` (`feature-levers-L1-L4.md`) | −0.03 [−0.11, +0.07] / −0.04 [−0.15, +0.06] | osm_nolirf anchored LGB | F | null | `oof_main_flseq_osm_nolirf_l1-50ebc5643d` (same caveat as #6) | (b) ≈ 4,100–7,000 s |
| 8 | L3 AOBT_3 queue counts (same) | +0.04 [−0.07, +0.15] / +0.03 [−0.09, +0.16] | same | F | null | `oof_main_flseq_osm_nolirf_l3-d08ab29d15` (same caveat) | (b) ≈ 3,800–7,000 s |
| 9 | P1 stage 1, non-LIRF unmatched upper cap (`P1-stage1-cap-CLOSED.md`) | best cap 2,400: ΔRMSE −0.11, help/hurt 1.7, month CI straddles; fails its own three conditions at every cap | `main_reparam` | O | none of its conditions met | rule is re-appliable to the current OOF | (a), but CLOSED by its own pre-registered "do not reopen" clause |
| 10 | P1-B LIRF unmatched seasonal bias (`P1B-lirf-seasonal-bias-REJECT.md`) | even the circular oracle: −1.75 [−2.91, −0.59] / −0.11 [−0.53, +0.31] | `main_gap` | O | the oracle cannot pass K=200 | re-appliable | (a), but no honest arm can beat the oracle, which fails the rule |

### 0.2 Levers that need retraining (no re-usable OOF on the current row set and recipe)

| # | lever (note) | measured (K=0 / K=200) | base | form | how close | OOF on disk | cost (b) |
|---|---|---|---|---|---|---|---|
| 11 | C1 `proxy_is_nominal`, two-column (`C1-two-column-remeasured.md`) | vs production set: **−0.361 [−0.610, −0.122] / −0.292 [−0.464, −0.114]** (ADOPT); vs arity-matched noise: −0.422 [−1.030, +0.085] / **−0.201 [−0.416, −0.004]** | `main_gap`, forward fold, single LGB | F | ADOPT vs base; vs noise K=200 excludes zero, K=0 crosses | `oof_c13_c1_2col-99502d00ed` (forward fold only; not row-aligned) | anchored opt CV ≈ 7,000 s (+ a noise arm ≈ 7,000 s); each further member adds its own CV |
| 12 | P8-E dead-column removal (`dead-column-removal.md`) | base→drop **−0.13 [−0.25, −0.02] / −0.14 [−0.28, −0.01]** (ADOPT); control→drop −0.35 [−0.92, +0.11] / +0.16 [−0.08, +0.37] | `main_gap` | F (deletion) | ADOPT by rule; its control arm does not separate it | not row-aligned | as #11; current configs still carry the five columns (no `exclude_features` in `configs/main_flseq_osm_nolirf_l2_opt.yaml`) |
| 13 | P8-B wxcodes repair (`P8-B-wxcodes-repair.md`) | noise→repair −0.35 [−1.01, +0.22] / **−0.35 [−0.59, −0.12]**; base→repair −0.18 [−0.64, +0.26] / −0.50 [−0.71, −0.24] | `main_gap`, folds forward + jul_fwd | F | K=200 excluded zero, K=0 crossed | `oof_p8b_wx-7f7be04073` (two folds; not aligned) | feature cache rebuild ≈ 4 min + anchored CV ≈ 7,000 s per arm |
| 14 | `dest_distance_km` (`features_log.md` exp. 8) | **−2.89 [−4.32, −1.56]** / +0.02 [−0.24, +0.25]; unmatched 2171.7 → 2129.3 | `main` (CV 285.5), forward fold | F, both lanes | K=0 excluded zero, tail-only; the gain was in model B | `oof_feat_f9_dist-5ed94c00a3` (old row set) | model-B run ≈ 3,800 s (L2-B's CV time) + cache rebuild |
| 15 | ADS-B residual V2 = 0.5 LGB + 0.5 MLP (`adsb-learned-residual.md` §9) | −0.25 s RMSE; paired gain +151.0 s² [−29.9, +356.5]; +1,380 s² of it at LIRF | `hybrid_blend_l2_pa` (two bases old) | M of the residual stage | CI crosses zero | rows were in another session's `$SCRATCH` | residual-stage CV only: fits 18–36 s each, about 15–30 min in total (inference from §9.2's fit times) |
| 16 | `planned_flight_s` (`features_log.md` exp. 10) | −0.06 [−0.89, +0.70] / −0.21 [−0.49, +0.08] | `main`, 3 folds | F | "closest of ten"; K=200 upper end +0.08 | old row set | as #11. The related L1 (#7) on the current family measured −0.03/−0.04 |
| 17 | `op_offset_med` (exp. 7, 9) | forward −0.26 [−0.52, −0.01] at K=200; 3-fold +0.15 [−0.21, +0.52] | `main` | F | sign reversed on more folds | old | as #11 |
| 18 | aggregate-difference / IQR features (`docs/article_fe_2026-09-24/`) | vs noise K=200 −0.25 [−0.54, +0.11] | v14 `main_gap`, forward | F | both cross | old | as #11 |
| 19 | C3 wind direction, gust repair (`C1-C3-measured.md`) | −0.110 [−0.290, +0.058] (C3 vs noise, K=200); gust −0.078 / −0.106, both cross | `main_gap` | F | null | old | as #11 |
| 20 | Feature loop 1–6, frequency encoding, sin/cos, meta-features, weather v2 B/C (`features_log.md`) | all vs noise within ±0.25, or worse (meta +1.22 [+0.76, +1.67]) | `main` / `main_gap` | F | null or worse | old | as #11 |
| 21 | ADS-B residual variant U, unmatched-only (`adsb-learned-residual.md` §11) | −43.55 s² [−251.02, +176.23] (worse side) | pa_opt | R of residual for unmatched rows | null | — | residual CV, minutes |
| 22 | P1 stage 2, fill-classifier gate (`P1-stage2-gate-FAILED.md`) | AUC 0.9201 vs gate 0.93 | `main_reparam` | O | below gate; CLOSED with stages 4–5 | — | classifier fits; CLOSED by its plan |
| 23 | airport-qualified categorical levels (`airport-qualified-levels.md`) | ceiling −0.11 s, cross-fitted −0.06 s; no fit | `main_gap` | F | below the ±0.5 s floor | — | one anchored CV ≈ 7,000 s |
| 24 | P2 fix 1 early-stop encoder (`P2-fix1-earlystop-encoder.md`) | +0.04 [−0.06, +0.17] (forward) | `main_gap` | correctness fix | null; shipped on correctness | — | n/a (already in) |
| 25 | P10 CatBoost vs LightGBM (`P10-catboost-measured.md`) | "on hold" | `main_v15` | R | superseded: CatBoost entered as M3 (`catboost-blend.md`) | — | n/a |

Not inventoried as rejected: the v35 overlays still "held" in `docs/posjoin_2026-10-05/` (LIRF schedule-copy blend,
learned residual on posjoin rows, LOBT clip). They were never measured through `prc.compare` and were not rejected.
The CatBoost M3 retune (`catboost-retune.md`) is running and not closed.

### 0.3 Ranking by (promise, cheapness)

Promise tiers use only the numbers above:
- **A**: at least one of the rule's two CIs excluded zero on the improving side in the latest measurement, or a
  lane/airport subset with several CIs excluding zero on the improving side.
- **B**: improving point estimates, both CIs crossing zero.
- **C**: null, worse, or closed by its own pre-registered condition.

| rank | # | lever | tier | cost |
|---|---|---|---|---|
| 1 | 1 | LOFO blend weights | A | (a) |
| 2 | 2 | pa as its own member (pa4 form) | A | (a) |
| 3 | 3 | L2 in model B, as an unmatched-lane member | A | (a) |
| 4 | 4 | OSM in model B, as an unmatched-lane member | A (non-LIRF subset) | (a) |
| 5 | 5 | linear_tree, as an added matched member | C as a replacement; never measured as a member | (a) |
| 6 | 11 | C1 two-column | A | (b) ≈ 2 × 7,000 s |
| 7 | 12 | dead-column removal | A (vs base), C (vs control) | (b) ≈ 7,000 s |
| 8 | 13 | P8-B wxcodes | A (K=200 only) | (b) ≈ 2 × 7,000 s |
| 9 | 15 | ADS-B residual V2 (LGB+MLP mean) | B | (b) ≈ 15–30 min |
| 10 | 14 | `dest_distance_km` in model B | A (K=0 only, tail-only) | (b) ≈ 3,800 s |
| 11 | 16, 6, 18 | `planned_flight_s`, A1, aggregate diffs | B | (b) ≈ 7,000 s each |
| 12 | 7, 8, 9, 10, 17, 19–25 | the rest | C | — |

## 1. Phase 2 — pre-registration of the OOF-only re-checks

Written before any re-check below was computed. The items are ranks 1–5 of §0.3: the five levers with stored OOF on
the current row set. Items 9 and 10 are re-appliable but excluded: each is closed by its own pre-registered
condition (P1 stage 1 forbids a third reopening; P1-B's oracle already fails the rule).

These checks are **not blind**. R1 and R2 have earlier measurements against older bases that point to the candidate
(§0.1 rows 1–2). Five primaries are tested and no multiplicity correction is applied. An ADOPT here is evidence for a
later step, not a submission.

### 1.1 Row alignment (checked before any comparison, reported in §2.0)

An arm is used only if all of these hold. Otherwise its check is not run, and the reason is reported.
- Its OOF file has one row per `MVT_ID_mvt`.
- On the 672,428 scored rows of the base (label non-null and a2o non-null, as in `scripts/blend_mlp.py`), it has a
  non-null `oof` on every row.
- The set of `MVT_ID_mvt` with non-null `oof` among labelled rows equals the base's scored set.
- The fold sidecar `_folds.parquet` (pooled rows) gives the same fold for every scored row as a2o's sidecar.
- The lane of every row is `mvt_minus_aobt3` non-null in `data/work_flseq/feat_train.parquet`, as in the base.

Reproduction check: the base OOF rebuilt by the script below must equal
`logs/mlp/scratch/oof_hybrid_blend_l2_pa_opt_mlp.parquet` to 1e-9 on every scored row, and its RMSE must be 267.57.
If not, stop.

### 1.2 The re-checks

A = `hybrid_blend_l2_pa_opt_mlp` (the base above) in every primary. M1 = 0.5·a2o + 0.5·pao, M2 = p2o, M3 = c2,
M4 = m4.

| id | candidate B | matched lane | unmatched lane |
|---|---|---|---|
| **R1** | LOFO weights | the stored `oof_hybrid_blend_l2_pa_opt_mlp_lofo.parquet` (weights fitted on the other folds per held-out fold, `mlp-member.md` §3); rebuilt by the same formula and checked equal to the stored file | u |
| **R2** | pa as its own member | (a2o + pao + p2o + c2 + m4) / 5 | u |
| **R3** | L2-B as an unmatched member | base | 0.5·u + 0.5·u_l2b, u_l2b = `main_flseq_l2b-e40f03b86d` |
| **R4** | OSM-B as an unmatched member | base | 0.5·u + 0.5·u_osm, u_osm = `main_flseq_osm-bb204cd922` (the pre-registered arm; the LIRF-null arm was post hoc) |
| **R5** | linear_tree as an added member | (M1 + M2 + M3 + M4 + lt) / 5, lt = `main_flseq_osm_nolirf_l2_lt-a9fb149ab5` | u |

Weights are fixed here and not tuned on these rows (R1's weights come from other folds).

**Decision rule (primary, each of R1–R5 separately):** `python -m prc.compare hybrid_blend_l2_pa_opt_mlp <B>
--work-dir data/work_flseq --exp-dir <scratch>`, and its `prc.compare.verdict`. **ADOPT** only if the full CI (K=0)
and the drop-200 CI both exclude zero on the improving side. **REJECT** if both exclude zero on the worse side.
Anything else: "no difference / tail-only", keep the base.

**Reported for each primary, not entering the decision:** the K = 0/20/200/2000 curve, the month bootstrap, the lane
split, and per-airport differences (`scripts/plainy_blend.py`'s `detail` on the same pair).

**Secondary, report only (no decision):**
- R3s: unmatched lane = u_l2b alone (the original replacement form, on the current base).
- R4s: unmatched lane = u_osm alone.
- R34: unmatched lane = (u + u_l2b + u_osm) / 3.
- R25: R2 and R5 together: (a2o + pao + p2o + c2 + m4 + lt) / 6.
- Error correlation of lt with M1–M4 on matched rows.

Nothing is written to `outputs/`. Hybrid OOFs go to the session scratchpad. The script is
`docs/competitor_research_2026-09-21/scripts/rejected_rereview.py` (committed with the results).

Resources: `POLARS_MAX_THREADS=1`; only the columns needed; the cgroup `total_rss` (path read from
`/proc/self/cgroup`) is checked before each step, and the step waits while it is above 10,500 MB.

## 2. Results (computed after the §0–§1 commit a0743f5)

Script: `docs/competitor_research_2026-09-21/scripts/rejected_rereview.py` (`align`, then `build --scratch DIR`).
Comparisons: `python -m prc.compare hybrid_blend_l2_pa_opt_mlp <B> --work-dir data/work_flseq --exp-dir DIR`, and
`scripts/plainy_blend.py`'s `detail` on the same pair. DIR was the session scratchpad. Peak RSS of each step was
about 0.84 GB. The cgroup `total_rss` was 7.1–7.2 GB before each step (gate 10,500 MB; it never had to wait).

### 2.0 Row alignment and reproduction

- All nine OOF files used (a2o, pao, p2o, c2, m4, u, u_l2b, u_osm, lt) have 2,085,047 rows and no duplicate ids.
- Each has a non-null `oof` on exactly the base's 672,428 scored rows: same id set, 0 missing, 0 extra.
- Each `_folds` sidecar gives the same pooled fold as a2o's on every scored row (0 mismatches).
- The base rebuilt with `scripts/blend_mlp.py`'s own `build_oof` equals the stored
  `logs/mlp/scratch/oof_hybrid_blend_l2_pa_opt_mlp.parquet` (max |diff| 0, RMSE 267.5733). The rebuilt LOFO file
  equals the stored `oof_hybrid_blend_l2_pa_opt_mlp_lofo.parquet` (max |diff| 0).

### 2.1 Primaries (the decision)

A = `hybrid_blend_l2_pa_opt_mlp` (267.57) in every row. 672,428 pooled OOF rows.

| id | B | K=0 | K=20 | K=200 | K=2000 | month bootstrap | verdict |
|---|---|---|---|---|---|---|---|
| **R1** | LOFO weights | **−0.25 [−0.34, −0.18]** | −0.30 [−0.39, −0.21] | **−0.29 [−0.38, −0.20]** | −0.36 [−0.42, −0.29] | −0.25 [−0.29, −0.19] | **ADOPT** |
| **R2** | pa as its own member, 1/5 each | **−0.18 [−0.24, −0.13]** | −0.21 [−0.27, −0.16] | **−0.22 [−0.28, −0.17]** | −0.26 [−0.30, −0.23] | −0.18 [−0.20, −0.16] | **ADOPT** |
| R3 | unmatched 0.5·u + 0.5·u_l2b | +0.08 [−0.22, +0.42] | −0.21 [−0.36, −0.05] | −0.20 [−0.29, −0.12] | −0.12 [−0.15, −0.08] | +0.08 [−0.32, +0.44] | no difference on all rows (tail-side K=200 excludes zero) |
| R4 | unmatched 0.5·u + 0.5·u_osm | +0.40 [+0.07, +0.73] | +0.06 [−0.16, +0.32] | −0.13 [−0.23, −0.02] | −0.10 [−0.13, −0.07] | +0.40 [−0.01, +0.77] | worse on all rows (REJECT) |
| R5 | lt as a 5th matched member | +0.20 [+0.13, +0.28] | +0.24 [+0.15, +0.31] | +0.23 [+0.15, +0.31] | +0.25 [+0.20, +0.31] | +0.20 [+0.11, +0.34] | **REJECT** (both CIs on the worse side) |

`prc.compare` prints "REJECT" for every non-ADOPT branch; the branch is given above.

Detail (report only):
- **R1.** Matched −0.33 [−0.42, −0.24]; unmatched +0.00 (identical). All except LIRF −0.20 [−0.28, −0.13].
  - Every month improves: 2025-01 −0.29, 2025-07 −0.29, 2025-11 −0.26, 2025-12 −0.15 [−0.29, −0.01].
  - Every airport's point estimate is ≤ 0. CIs exclude zero at LIRF −0.60, EHAM −0.48, EDDM −0.55, LEMD −0.51 and
    LSZH −0.31. LFPG, EGLL, LTFM, EDDF and LEBL cross zero.
  - The LOFO weights on (M1, M2, M3, M4) are those of `mlp-member.md` §3: per held-out fold 0.505–0.535 /
    0.089–0.110 / 0.126–0.146 / 0.220–0.249; on all folds 0.520 / 0.103 / 0.138 / 0.239.
- **R2.** Matched −0.24 [−0.30, −0.18]; unmatched +0.00.
  - Every month's CI is below zero (2025-01 −0.21, 2025-07 −0.17, 2025-11 −0.20, 2025-12 −0.14).
  - Eight of ten airports' CIs are below zero. LTFM −0.06 [−0.16, +0.04] and EDDF −0.14 [−0.32, +0.03] cross zero.
  - R2 moves the a2o + pao weight from 1/4 to 2/5. R1 moves M1 (= a2o + pao) to about 0.52. Both point the same
    way, so they are not independent evidence (inference).
- **R3.** Matched +0.00 by construction; unmatched +1.16 [−3.85, +6.74]. All except LIRF **−0.19 [−0.31, −0.08]**.
  LIRF all rows +1.70 [−0.51, +4.25] (46.1 % of the squared disagreement). CIs below zero at LTFM −0.39, EHAM
  −0.41, EDDM −0.33, LSZH −0.68, EDDF −0.12 and LEMD −0.12. Months: 2025-11 +0.61, 2025-12 −0.56, both CIs crossing.
- **R4.** Unmatched +5.51 [+1.10, +11.61]. All except LIRF **−0.26 [−0.38, −0.16]**. LIRF all rows **+4.31
  [+2.07, +6.75]** (62.1 % of the squared disagreement). Every other airport's point estimate is negative, and eight
  of nine CIs exclude zero (LFPG −0.06 [−0.28, +0.05] does not).
- **R5.** Matched +0.26 [+0.17, +0.35]. Worse at 8 of 10 airports with CIs above zero (LTFM +0.70, EDDM +0.65).
  Matched-lane error correlation of lt: 0.966 with M1, 0.961 with M2, 0.927 with M3, 0.898 with M4. Its RMSE alone
  is 212.16 (M1 205.94, M4 213.27). Inference: lt is nearly as correlated with M1 as M2 is (0.976), and it is
  6.2 s worse than M1, so it adds error without adding much diversity.

### 2.2 Secondaries (report only)

| id | B | K=0 | K=200 | month bootstrap | branch |
|---|---|---|---|---|---|
| R3s | unmatched = u_l2b alone | +0.36 [−0.22, +1.05] | −0.22 [−0.38, −0.05] | +0.36 [−0.31, +1.06] | no difference on all rows |
| R4s | unmatched = u_osm alone | +1.04 [+0.38, +1.70] | −0.08 [−0.26, +0.11] | +1.04 [+0.33, +1.76] | worse on all rows |
| R34 | unmatched = (u + u_l2b + u_osm) / 3 | +0.34 [−0.04, +0.76] | −0.21 [−0.32, −0.10] | +0.34 [−0.18, +0.69] | no difference on all rows |
| R25 | (a2o + pao + p2o + c2 + m4 + lt) / 6 | +0.05 [−0.03, +0.14] | +0.06 [−0.04, +0.15] | +0.05 [−0.03, +0.17] | no difference |

R3s and R4s reproduce the original notes' numbers on the current base: L2-B +0.35 / −0.21 and OSM +0.95 / −0.12
before; +0.36 / −0.22 and +1.04 / −0.08 now. That is expected: the unmatched lane has not changed since v16.

### 2.3 Reading

- **Truly ineffective, confirmed:** linear_tree as a member (R5, worse on both CIs) and OSM in model B (R4, worse on
  all rows). The "worse alone, helps in the blend" pattern did not hold for linear_tree. Its error correlation with
  M1 (0.966) is far above that of the members it held for (CatBoost 0.941, MLP 0.920).
- **Rejected or closed, but they pass now:** the blend-weight levers R1 (LOFO weights, −0.25 s) and R2 (pa as its
  own member, −0.18 s). Both were report-only secondaries, never rejected on their own numbers. Both pass the rule
  on the v35 base, at every month, with no unmatched-lane change.
- **Same as before:** L2 and OSM in model B. Both keep the pattern the notes described. They are better on the
  non-LIRF rows (−0.19 and −0.26 s) and worse at LIRF unmatched, and they fail the full-data leg. A LIRF-excluded
  unmatched blend was **not** pre-registered and was not run. Choosing LIRF from these rows would make it post hoc.

### 2.4 Recommendations (nothing here was run)

OOF-only follow-ups. Each needs its own pre-registration because it is chosen after §2.1.
- R1 + R2 combined: LOFO weights on (a2o, pao, p2o, c2, m4). OOF-only, a few minutes.
- R3 restricted to non-LIRF unmatched rows (LIRF keeps u). It is post hoc: LIRF was picked on these same rows, in
  `osm-on-flseq.md` §7 as well. OOF-only.
- A production version of R1 or R2 needs only `scripts/blend_mlp.py full`-style arithmetic on the existing test
  files: no retraining, minutes. Then the ADS-B residual has to be refitted on the new base
  (`adsb-learned-residual.md` §12 design). The M3 CatBoost retune in progress will change one member, so the
  weights should be redone after it lands.

Retrain-needed (ranked as in §0.3; the costs are the notes' measured wall times on this 4-CPU box):
1. C1 two-column `proxy_is_nominal` on the anchored opt member, with an arity-matched noise arm: about
   2 × 7,000 s of CV (`optuna-retune.md` §2 confirmation 6,992 s). Entering all members adds the plain-y CV (about
   1,300–4,100 s), PA (about 7,200 s), CatBoost (about 10,300 s) and MLP (about 9,000 s).
2. P8-E dead-column removal on the current recipe: one anchored opt CV (about 7,000 s) plus a control arm.
3. P8-B wxcodes repair: a feature-cache rebuild (about 4 min) plus 2 × 7,000 s.
4. ADS-B residual V2 (0.5 LGB + 0.5 MLP) on the v35 base: residual-stage CV only. The fits took 18–36 s each, so
   the whole stage is about 15–30 min (inference from §9.2's fit times).
5. `dest_distance_km` in model B: a model-B CV of about 3,800 s (L2-B's run) plus a cache rebuild.
6. `planned_flight_s`, A1 and the aggregate-difference features: about 7,000 s each. These are tier B, with both CIs
   crossing zero.

Peak `total_rss` of every retraining run in the notes was 9.0–12.5 GB. None of them can run next to the CatBoost
tuning job (8.1 GB) under the 13,681 MB limit.

## 3. R6 pre-registration (written and committed before R6 was computed)

The coordinator asked for this follow-up after §2. It is chosen after seeing R1 and R2, so it is not blind.

- **R6.** Matched lane = leave-one-fold-out sum-to-one least-squares weights over the five members
  (a2o, pao, p2o, c2, m4).
  - It uses the method of `scripts/blend_mlp.py`: `catboost_blend.sum_to_one_weights`, fitted on the matched scored
    rows of the other pooled folds (a2o's `_folds` sidecar), and applied to the held-out fold.
  - Unmatched lane = u.
- **Comparison.** A = `hybrid_blend_l2_pa_opt_mlp`. Same `prc.compare` and the same decision rule as §1.2. Report the
  full, drop-200 and month figures, plus per-airport detail.
- **Choice rule (fixed now).** Among R1, R2 and R6, keep those whose verdict is ADOPT. Pick the one with the best
  (most negative) full-data point estimate.
- **Output.** `outputs/hybrid_blend_reweight.parquet`, built from the existing full-fit member files of
  `scripts/blend_mlp.py`'s `FULL_FILES`, with no training.
  - Matched rows = `rint` of the winner's formula. Where the winner uses weights, they are the weights fitted on all
    OOF folds (stated in §4).
  - Unmatched rows = `outputs/main_flseq.parquet`.
- **Validation.** The row count equals the template. The schema equals the template. Ids and their order equal
  `data/raw/submitting.parquet`. There are 0 nulls and 0 negatives. `prc.validate_submission` passes.
- **OOF for the ADS-B refit.** The winner's OOF is copied to
  `logs/rereview/scratch/oof_hybrid_blend_reweight.parquet`.
