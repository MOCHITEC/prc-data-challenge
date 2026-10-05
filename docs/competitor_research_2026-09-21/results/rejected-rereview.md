# Re-review of rejected and closed levers against the v35 base — inventory and OOF-only re-checks (2026-10-05)

**Status (2026-10-05): done.** R1 (LOFO weights) ADOPT −0.25 s [−0.34, −0.18], K=200 −0.29 [−0.38, −0.20]; R2 (pa as its own member) ADOPT −0.18 [−0.24, −0.13], K=200 −0.22 [−0.28, −0.17]; R3 no difference; R4 and R5 REJECT (worse). R6 (pa as member + LOFO weights) ADOPT −0.32 [−0.44, −0.24], K=200 −0.36 [−0.48, −0.26], chosen by §3's rule; full-fit file not built (write to outputs/ denied by the permission classifier). §0 and §1 were committed as a0743f5 before any re-check was computed. Results are in §2+.
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

## 4. R6 result and choice (computed after 23a30d9)

`rejected_rereview.py r6`, then `prc.compare hybrid_blend_l2_pa_opt_mlp rr_R6_pa5_lofo` on 672,428 rows.

| K | A | B | diff [CI95] |
|---|---|---|---|
| 0 | 267.57 | 267.25 | **−0.32 [−0.44, −0.24]** |
| 20 | 222.70 | 222.31 | −0.39 [−0.51, −0.28] |
| 200 | 209.11 | 208.75 | **−0.36 [−0.48, −0.26]** |
| 2000 | 186.49 | 185.95 | −0.54 [−0.62, −0.47] |

Month bootstrap −0.32 [−0.38, −0.23]. **Verdict: ADOPT.**

Detail:
- Lanes: matched −0.43 [−0.55, −0.31]; unmatched +0.00; all except LIRF −0.30 [−0.42, −0.20].
- Months (all rows): 2025-01 −0.40, 2025-07 −0.20 [−0.45, +0.07], 2025-11 −0.38, 2025-12 −0.38. July's CI crosses
  zero; the other three are below zero.
- Airports (all rows): LIRF −0.55 [−0.95, −0.12], LFPG −0.14 [−0.41, +0.03], EGLL −0.05 [−0.43, +0.35], LTFM −0.29
  [−0.49, −0.09], EHAM −0.56 [−0.86, −0.27], EDDF −0.36 [−0.77, −0.00], EDDM −0.78 [−1.19, −0.37], LEMD −0.76
  [−0.95, −0.56], LSZH −0.42 [−0.76, −0.02], LEBL −0.12 [−0.32, +0.06]. Every point estimate is negative; LFPG, EGLL
  and LEBL cross zero.
- LOFO weights on (a2o, pao, p2o, c2, m4):
  - held-out forward: 0.133 / 0.345 / 0.143 / 0.134 / 0.246
  - held-out jan: 0.069 / 0.360 / 0.181 / 0.149 / 0.240
  - held-out jul: 0.015 / 0.422 / 0.196 / 0.156 / 0.211
  - The a2o weight ranges from 0.015 to 0.133 across folds.
- Weights fitted on all OOF folds: a2o 0.069428, pao 0.374785, p2o 0.176099, c2 0.147002, m4 0.232686 (sum 1).

**Choice (rule of §3).** R1 −0.25, R2 −0.18 and R6 −0.32 all pass. **R6 is chosen.**

**Full-fit file: not built.** The command that would write `outputs/hybrid_blend_reweight.parquet` was denied by the
session's permission classifier. It was not retried, and no other route was used. The build code is in the script
(`rejected_rereview.py full --weights '<json above>'`), for someone with permission to write to `outputs/`.

**R6 OOF:** it is in this session's scratchpad (`.../scratchpad/rr/oof_rr_R6_pa5_lofo.parquet`). The copy to
`logs/rereview/scratch/` was in the same denied command, so it was not made. Rebuild it anywhere with
`rejected_rereview.py r6 --scratch DIR`, which is deterministic: lstsq on fixed OOFs.

## Per-item review of retrain candidates (2026-10-05)

Scope: the seven retrain-needed items of §2.4 (inventory rows 11, 12, 13, 15, 14, 16 and 6). For each one, this
section reads the original note to find out **why** it failed. The inventory ranked them only by their CIs. Nothing
was trained, nothing in `src/` or `configs/` was changed, and nothing was submitted.

Marking: quotes are verbatim from the named note. **[new]** = a read-only check made for this section (below).
**[X]** = inference.

**The [new] checks.** One `POLARS_MAX_THREADS=1` read, peak RSS 0.58 GB, with the cgroup `total_rss` at 7.8 GB
before it.
- Inputs: columns of `data/work_flseq/feat_train.parquet`, joined to the base OOF
  `logs/mlp/scratch/oof_hybrid_blend_l2_pa_opt_mlp.parquet` and to a2o's `_folds` sidecar (pooled rows).
- Residual = `TAXITIME_SEC_mvt − oof`, the sign convention of `scripts/c1_2col_mechanism.py`.
- Lane = `mvt_minus_aobt3` null or non-null.
- No script was committed. The computation is fully described where each number is used.

Facts about the current base used below (the §0 header, `configs/`, `docs/posjoin_2026-10-05/README.md`):
- **Matched lane.** `((0.5·a2o + 0.5·pao) + p2o + c2 + m4) / 4`.
  - Every member carries L2 (`fe_fwd_nm: true`) and the OSM distance nulled at LIRF.
  - The LGB members a2o, pao and p2o carry `matched_params` with `feature_fraction: 0.4178`. The old recipes had 0.8.
  - None of the five member configs, and not `main_flseq.yaml`, has `exclude_features`, `wx_codes` or `fe_arr_fill`.
- **Unmatched lane.** u = `main_flseq-8615e6a194` (v16's model B), unchanged since v16.
- **Test-time stages.** These come after the base:
  - The learned ADS-B residual. It has `c = 0` at LTFM. The posjoin README says of unmatched 2026 rows: "the callsign
    ADS-B join, and with it the learned ADS-B residual, never reaches them".
  - The 430-row positional overlay. It covers unmatched rows only: EHAM 351, LSZH 53 and EDDF 26
    (`replacements_v33.csv`). "LIRF excluded (schedule-fill labels)".

### R-1. C1 two-column (`C1-two-column-remeasured.md`)

1. **Definition and entry.**
   - Two columns: `proxy_at_mode`, "1 when this row's offset is exactly the modal minute of its airport x runway"
     (`src/prc/features.py:367`), and `proxy_minus_mode_min`. The mode table is fitted on train ∪ rank, label-free.
   - They were added by `include_candidates` to the **`main_gap` (v14) single anchored LightGBM**, both lanes
     (`configs/c13_c1_2col.yaml`).
   - Forward fold only, 3 seeds, `es_holdout_encodings: true`, frame `data/work_c1c3`.
   - Model B never used them: "Lane B RMSE is 1004.0424 on the base, on the three-column C1 and on the two-column C1
     - identical to four decimals, with the same 80 boosting rounds".
2. **Measured.**
   - Forward fold vs `c13_base`: **−0.361 [−0.610, −0.122] / −0.292 [−0.464, −0.114]**.
     - Month bootstrap −0.361 [−0.422, −0.302].
     - Lane A −0.404. Lane B identical.
   - Vs the arity-matched `c13_noise2`: −0.422 [−1.030, +0.085] / −0.201 [−0.416, −0.004].
   - `jul_fwd` (§9.x): `c13_base_jf` 314.4240 → `c13_c1_2col_jf` 314.3446 = **−0.079**. Lane B identical
     (1333.7191). "No CI exists for it". The noise arm was never run.
   - Commits:
     - forward arms at `4faa313`; `c13_base` at `f52f741`;
     - `jul_fwd` at `42e8d95` / `cc972f1`.
   - No per-airport split was reported.
3. **Why rejected.**
   - §9.x: "the one thing this replication existed to test has an answer: the forward effect does not carry across at
     its measured size. That is the same shape as v15 … A −0.08 on the second fold is inside the ±0.5 s single-column
     noise floor".
   - The pre-registered kill D1 (§9.0): "`jul_fwd` base → C1 reads **≥ −0.10 s at K=0 or at K=200** (point estimate)".
   - The mechanism the feature targets (§3.3): "On the flagged fifth of the rows the production model is biased by
     −9.78 s against −3.73 s elsewhere". That was on the forward fold, `c13_base` OOF.
4. **What changed since.**
   - The matched lane is now a 4-member blend, retuned. It carries L2 `fw_*` columns: means of other departures'
     `MVT − AOBT_3` in forward windows, per airport and per airport × runway (`feature-levers-L1-L4.md` §3).
   - C1 has never been measured on the flseq family or on four folds.
   - **[new]** Current-base matched-lane mean residual, flagged (`proxy_at_mode = 1`) vs not:

     | fold | flagged | unflagged |
     |---|---|---|
     | forward | −10.20 s (n = 66,704) | −4.85 s (n = 258,300) |
     | jan | −5.03 | −4.76 |
     | jul | **+1.16** | −3.32 |
     | pooled | −5.90 | −4.40 |

     The pooled correlation of the flag with the residual is −0.003.
5. **Class: (a).**
   - The failure was non-transfer across folds. It is not tied to the base.
   - **[new]** The bias C1 targets is still present on the current base, at about the old size on forward: a gap of
     −5.35 s now against −6.05 s then. It is absent in jan and reversed in jul.
   - [X] So the base change has not turned a forward-only signal into a pooled one. The `jul_fwd` result is what that
     pattern predicts.
   - [X] Caveat: a first-moment check cannot rule out a gain that comes through interactions.

### R-2. P8-E dead-column removal (`dead-column-removal.md`, shipped as v15)

1. **Definition and entry.**
   - `exclude_features: [precip_mm, precip_mm_prev1h, temp_x_precip, temp_x_precip_prev1h, runway_null]`.
   - Entered both lanes of the `main_gap` single LGB: model A 90 → 85 columns, model B 84 → 79.
   - `forward` + `jul_fwd`, 3 seeds.
   - Built as `configs/main_v15.yaml`.
2. **Measured.**
   - Pooled over 518,722 rows: base → drop **−0.13 [−0.25, −0.02] / −0.14 [−0.28, −0.01]**, month −0.13
     [−0.17, −0.12].
   - Control (five live columns removed) → drop: −0.35 [−0.92, +0.11] / +0.16 [−0.08, +0.37].
   - Lanes:
     - "unmatched n= 6,943 rows where base != drop: 0";
     - forward matched −0.17 [−0.33, −0.01];
     - `jul_fwd` matched −0.15 [−0.38, +0.09].
   - Four-fold v15 CV (`b6375c8`): 277.53 against v14's 277.54. By fold: jan +0.09, jul −0.11, `jul_fwd` −0.12,
     forward 0.00.
   - Board (`4faa313`): v15 279.5397 against v14 279.5234, **+0.0163**.
   - Commits: arms at `1a80f48` and a worktree pinned to `95ec23e`; note `f45c1cb`.
3. **Why rejected.**
   - Pre-registered, §0.2: "Five columns that are identically zero cannot carry −0.5 s of signal-by-removal; an ADOPT
     would mean the `feature_fraction` reshuffle drew lucky on these two folds".
   - §3.3: "the narrow CI is a true statement about this particular pair of models … and **not** evidence that the
     effect is large".
   - `4faa313`: "it still came to nothing on four folds and on the board. A two-fold arm can read optimistic against
     the same change measured on the full fold set".
4. **What changed since.**
   - The flseq family descends from `main_gap`, not from v15. The `main_flseq.yaml` header says
     "configs/main_gap.yaml (v14, LB 279.52) + the fl and seq groups".
   - So all members still see the five columns.
   - **[new]** In `data/work_flseq` the four precip columns are 0 on every non-null row, and `runway_null` is
     constant 0.
   - The LGB members now use `feature_fraction` 0.4178. Two members are CatBoost and MLP.
5. **Class: (a).**
   - The mechanism the note gives is base-independent: the columns are provably constant, and any measured effect is
     the column-sampling redraw.
   - Four folds and the board both read null.
   - [X] A new `feature_fraction` or more members changes which redraw happens. It does not give the redraw a sign.
   - [X] The coverage-gate and correctness case (`P8-D`) is a separate decision, not a scoring re-test.

### R-3. P8-B wxcodes repair (`P8-B-wxcodes-repair.md`)

1. **Definition and entry.**
   - `wx_codes: true` swaps the four constant precip columns for `wet_wx`, `shower_wx`, `obscuration_wx` and
     `frozen_wx_i8`, plus their `_prev1h` copies.
   - It redefines `deicing_gate` as `(temp_c ≤ 3) ∧ (frozen_wx ∨ wet_wx)`.
   - Model-visible features 90 → 94, on the `main_gap` single LGB.
   - `forward` + `jul_fwd`, 3 seeds.
2. **Measured.**
   - Noise → repair: −0.35 [−1.01, +0.22] / **−0.35 [−0.59, −0.12]**.
   - Base → repair: −0.18 [−0.64, +0.26] / −0.50 [−0.71, −0.24].
   - Base → noise: +0.16 [−0.58, +1.06].
   - Lanes, vs noise: forward matched −0.24 [−0.39, −0.08]; `jul_fwd` matched −0.13 [−0.38, +0.13].
   - Subsets: `shower_wx = 1` −1.49 [−3.23, +0.26]; `deicing_gate = 1` −0.42 [−2.74, +1.65].
   - Commits `3b40254` / `338e020`.
3. **Why rejected.**
   - "The repair's point estimate is inside the range that adding four signal-free columns produces. It is not
     distinguishable from doing nothing."
   - The distribution shift (§4): "July 2026 was a dry July … a July rain feature is much less active at serve time
     than at train time, and it is a concrete reason not to expect the LB to reward this even if CV had."
     - On the scored set, 2026-07: `wet_wx` 2.15 %, `deicing_gate` 0.00 %.
     - `shower_wx @EDDM 2026-07 23.82% -> 6.41%`.
4. **What changed since.**
   - `wx_codes` is still off by default and in every current config.
   - The 2026 METAR support is a property of the scored data and cannot change.
   - The base is now a blend, followed by the ADS-B residual.
5. **Class: (a).**
   - Both stated reasons still hold: arity-level size, and the 2025 → 2026 July support shift.
   - No base change touches either.

### R-4. ADS-B residual V2 = 0.5 LGB + 0.5 MLP (`adsb-learned-residual.md` §9)

1. **Definition and entry.**
   - `c = 0.5 × c_LGB + 0.5 × c_V1`.
   - V1 is an sklearn `MLPRegressor` (128, 64), 3 seeds, 122 inputs.
   - It replaces the correction of the residual stage on ADS-B rows (286,008 in 2025). It is a stage after the base,
     for both lanes, with `c = 0` at LTFM.
   - Measured on the `hybrid_blend_l2_pa` base: matched `((0.5·a2 + 0.5·pa) + p2 + c2) / 3`, unmatched u.
2. **Measured.**
   - 344,419 jan + jul 2025 rows, day-block bootstrap.
     - V2 vs the current LGB: **+151.0 s² [−29.9, +356.5]**, RMSE 300.307 → 300.055 (−0.252).
     - V1 alone: −512.3 [−873.0, −121.0].
   - Per airport (V2, s²): LIRF **+1,380**, LFPG +159, LSZH +155, EGLL +99, LEBL +80, EDDF +39, LEMD +25, EDDM +23,
     EHAM −86.
   - corr(c_LGB, c_MLP) = 0.893.
   - Commits: pre-registration `548538c`, results `0ce97ec`.
   - Peak `total_rss` 1.45 GB. "Each fit took 18–36 s."
3. **Why rejected.**
   - "no variant passes"; "V2's point estimate is −0.25 s RMSE, but its CI includes 0."
   - "Most of V2's point gain is LIRF, the airport with the widest CIs in sections 2–8."
   - The note gives no base-specific mechanism.
4. **What changed since.**
   - The base moved twice: §10 `pa_opt`, then §12 `pa_opt_mlp`.
   - The LGB residual was refitted each time. Its gain went 8,265.3 → 8,208.1 → 8,018.2 s².
   - The matched lane now includes a tabular MLP member (m4). Its inputs are not the residual's ADS-B features.
   - Variant U was rejected (§11.2).
   - Posjoin README: LIRF ADS-B ground coverage was 0 in Jan 2025. [X] So LIRF's residual training rows are Jul 2025.
   - V2 was never re-run.
5. **Class: (c).**
   - The rejection is a power statement: the CI crosses zero, and the point gain is concentrated at LIRF.
   - It is not a mechanism that a base change could remove.
   - Nothing measured since says whether the LIRF concentration remains on the new base.
   - It is not (b) by this section's definition.
   - Fact for the coordinator: a re-run is residual-stage CV only. Fits took 18–36 s, peak 1.45 GB.
     [X] It would be a second draw at the same power, not a test of a removed mechanism.

### R-5. `dest_distance_km` in model B (`docs/features_log.md` exp. 8, 9)

1. **Definition and entry.**
   - The great-circle distance ADEP → ADES from OurAirports (`src/prc/airports.py:61`). "被覆率 99.45%".
   - Shown to both lanes of the `main` single LGB (baseline 285.52).
   - Forward fold, 1 seed, `feature_fraction` 1.0, against a noise column.
   - Arms: `feat_f9_dist-5ed94c00a3` vs `ff1_noise2-856a62b645`, both at `9e6fe61`.
   - "マッチ行 203.3 → 203.3（不変）"; "未マッチ行 2171.7 → **2129.3**（−42）". The gain was all in model B.
   - Both records have `rounds_unmatched` 45.
2. **Measured.**
   - K=0 **−2.89 [−4.32, −1.56]**, K=200 +0.02 [−0.24, +0.25].
   - Exp. 9: three folds, combined with `op_offset_med`. K=200 +0.15 [−0.21, +0.52], matched 218.2 → 218.3.
   - No per-airport split was reported.
3. **Why rejected.**
   - "典型的な「尾部のみ」。棄却。 model B の 255 葉と同じ形（CV full −34.8 → 本番 +13.2）で、規則が存在する理由そのもの。"
   - "その群は CI 幅 62 秒で測定不能なので規則上採用できない。"
   - The rule header: "未マッチ行だけを動かす変更は測定不能なので対象外。"
4. **What changed since.**
   - u is now v16's model B: fl + seq, `stand_arrival_gap`, LIRF schedule reparam, 3 seeds. `dest_distance_km` is not
     in its `include_candidates`.
   - Measurement is now four folds and 672,428 rows, with a lane split.
   - **[new]** On the current base:
     - 8,378 unmatched scored rows carry 42.3 % of the SSE.
     - Spearman(`dest_distance_km`, residual) on those rows is +0.003 (LIRF −0.022, n = 537; non-LIRF +0.015).
     - Spearman with |residual| is +0.118.
   - `model-b-ceiling.md` S1.4 (main_gap lane B): "`dest_distance_km` 0.22 %", against noise controls at 0.12–0.35 %.
   - Two model-B additions on the current model B failed the full-data leg here: R3 (L2-B) +0.08 [−0.22, +0.42] and
     R4 (OSM-B) +0.40 [+0.07, +0.73].
   - The 2026 unmatched rows get no learned residual. So a model-B change would reach the board unaltered, except on
     the 430 overlay rows.
5. **Class: (a).**
   - The stated mechanism is the lane structure plus the both-CI rule. Unmatched rows are about 1 % of rows and
     about 42 % of the SSE. Both are unchanged.
   - Current model-B additions still fail one leg (R3, R4).
   - **[new]** The column has no monotone relation to the current model B's signed residual.
   - [X] Any gain would have to come through interactions on about 8 k rows. That is the regime the note called
     unmeasurable.

### R-6. `planned_flight_s` (`docs/features_log.md` exp. 10)

1. **Definition and entry.**
   - `ARVT_1_flt − EOBT_1_flt` in seconds (`src/prc/features.py:610`). "正解との相関 +0.2575".
   - **[new]** It is null on 100 % of the 8,378 unmatched scored rows and on 0 % of matched rows. So it is a model-A
     column in effect, as `feature-levers-L1-L4.md` §3 says for EOBT_1 and ARVT_1.
   - Measured on `main`, pooled 3 folds (672k rows), 1 seed, `feature_fraction` 1.0.
   - Arms: `f10_plannedflight-b416261a17` vs `pw_noise2-7672c5b2b7`, both at `5794f24`.
2. **Measured.**
   - K=0 −0.06 [−0.89, +0.70]; K=20 −0.99 [−1.71, −0.35]; **K=200 −0.21 [−0.49, +0.08]**; K=2000 −0.25
     [−0.41, −0.10].
   - "マッチ行 218.5 → **218.1**".
   - No per-airport split was reported.
3. **Why rejected.**
   - "**10 本中もっとも本物に近い**が、判定レベルでゼロを跨ぐので棄却。"
   - Loop conclusion: "効きそうなものは全部 0.25 秒未満で、測定器の分解能を下回る。"
   - No base-specific mechanism is given.
4. **What changed since.**
   - Its residualised relative L1 `plan_nm_taxi` was measured on the current family (`main_flseq_osm_nolirf`, four
     folds). L1 is `clip(ARVT_1 − EOBT_1 − median over ADEP × ADES × type)`.
     - Result: −0.03 [−0.11, +0.07] / −0.04 [−0.15, +0.06], matched −0.04 [−0.15, +0.09].
     - Reading: "Our model already carries those clocks."
   - `delay-mechanisms.md` lists "aircraft class and speed | … `dest_distance_km`, `planned_flight_s` | yes | dead".
   - L2 has since been added to every member.
5. **Class: (a).**
   - The rejection was resolution.
   - The closest relative, re-measured on the current family, was null.
   - [X] L1 is a residualised form, not the raw column, so this is close evidence rather than the same test.

### R-7. A1 LIRF arrival mirror (`A1-lirf-arrival-mirror.md`)

1. **Definition and entry.**
   - Four columns: `arr_link_fill`, `arr_link_bs_abs`, `arr_link_same_op` and `arr_opday_fill`. They come from the
     same-stand latest arrival within 6 h; "A fill is `|in-block − sched| ≤ 12 s`".
   - **Model A only**: `unmatched_columns` excludes them. Flag `fe_arr_fill`.
   - Shown at all airports. The LIRF-only variant was "Not run".
   - Measured on `main_flseq` (v16) and on `main_flseq_osm_nolirf`. Both were non-L2 and untuned, four folds.
2. **Measured.**
   - On v16: **−0.09 [−0.21, +0.03] / −0.09 [−0.21, +0.04]**, matched −0.12 [−0.27, +0.02].
     - LIRF −0.52 [−1.07, −0.03].
     - EHAM −0.34 [−0.59, −0.12].
     - EDDF **+0.50 [+0.17, +0.91]**; LEBL +0.26 [+0.04, +0.49].
   - On osm_nolirf: −0.04 [−0.14, +0.06] / −0.07 [−0.18, +0.05], matched −0.05.
     - LSZH −0.38 [−0.71, −0.09].
     - LIRF −0.24 [−0.73, +0.25].
   - Kill (a), LIRF lane-A RMSE:
     - jul +0.43 on v16;
     - jul +0.62 and `jul_fwd` +0.77 on OSM.
   - Commits: runs at `7a9cb65`, results `f67a8c1`.
   - CV 7,750 s and 8,254 s ("about 2× the base runs"), peak 11.78 GB.
3. **Why rejected.**
   - "(a) **fails** on jul. (b) also fails: the pooled CI crosses zero."
   - On OSM: "'jul_fwd ≥ 0, close it' applies".
   - Mechanism of the failure: "It reverses in July on both configs. The note named July as the fold most likely to
     fail: LIRF's lowest-fill month".
   - Size: "At this size the ±0.5 s noise of adding any column to this model … is the same order as the effect."
4. **What changed since.**
   - All members now carry L2, the LGB members are retuned, and the matched lane is a 4-member blend.
   - A held v35 overlay, not in the base, targets the same LIRF schedule-copy population: "LIRF schedule-copy blend …
     26,277 [rows] … +390 MSE with A fixed in advance (Jan +15, Jul +375)" (`docs/posjoin_2026-10-05/deeper_checks.md`).
5. **Class: (a).**
   - The July reversal appeared on two different model-A bases, so it is not base-specific.
   - The effect shrank as the base improved (−0.09 → −0.04).
   - Its own close-out rule fired.
   - [X] If the LIRF schedule-copy overlay is adopted, it acts on the same rows from outside the model. That makes A1
     less, not more, likely to add anything.

### Summary

| item | class | key reason (quoted) |
|---|---|---|
| C1 two-column | (a) | "the forward effect does not carry across at its measured size". [new] Its target bias is still forward-only on the current base. |
| dead-column removal | (a) | "an ADOPT would mean the `feature_fraction` reshuffle drew lucky". Four-fold CV −0.01, board +0.0163. |
| P8-B wxcodes | (a) | "not distinguishable from doing nothing", plus the July 2026 support shift |
| ADS-B residual V2 | (c) | "its CI includes 0 … Most of V2's point gain is LIRF". No mechanism is given. |
| `dest_distance_km` in model B | (a) | "典型的な「尾部のみ」" and "CI 幅 62 秒で測定不能". The lane structure and the rule are unchanged. |
| `planned_flight_s` | (a) | "判定レベルでゼロを跨ぐ". Its relative L1 is null on the current family. |
| A1 | (a) | "It reverses in July on both configs" |

**No item is (b), so no re-test is pre-registered here.**
- [X] §2.4's retrain list carried these items on their CI tier alone. On the notes' own reasons, none of them failed
  for a cause that the move to the v35 base removes.
- The only one whose rejection is just a power statement is V2. It is also the only cheap one: residual-stage CV,
  18–36 s per fit, peak 1.45 GB.
