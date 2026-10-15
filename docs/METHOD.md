# Method: the model behind the final file

Team **resilient-hedgehog**, PRC Data Challenge 2026. Written 2026-10-15.

This page describes the model that produced the rows of the final file. It holds the detail that the root
`README.md` summarises in its §2 (method overview). Until 2026-10-15 this text was README §2 ("Model design"), the
EDDF and v48 parts of README §3, and the results table of README §10. It was moved here when the README was
shortened, with source references added.

- Which file was uploaded, how each month's rows were produced, and how to rebuild them:
  [`FINAL_SUBMISSION.md`](FINAL_SUBMISSION.md) (authoritative for the final file).
- The pipeline map at submission v47, with the reason for each stage: [`SOLUTION_OVERVIEW.md`](SOLUTION_OVERVIEW.md).
- How each piece was validated: `SOLUTION_OVERVIEW.md` §6 and the per-topic folders under `docs/`. The offline
  numbers are in §9 below; the leaderboard history is in `README.md` §8.

## 1. Task and target

- Target: `TAXITIME_SEC_mvt = MVT_TIME − BLOCK_TIME` (seconds) for departures at 10 airports: EDDF, EDDM, EGLL,
  EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM. Metric: RMSE over all rows.
- Training data: the organisers' 2025 movements (all 12 months) with the NM flight record joined (`AOBT_3_flt`,
  `LOBT_flt`, …).

## 2. Two lanes, split on `AOBT_3_flt`

- **Matched lane** (AOBT_3 present, about 98.5 % of rows; `SOLUTION_OVERVIEW.md` §1). In the final set, 661,713 of
  670,790 rows (`FINAL_SUBMISSION.md` §2.2 step 2).
- **Unmatched lane** (no NM record; 9,077 rows of the final set). A separate LightGBM ("model B") is fitted. At LIRF
  its target is reparameterised on the schedule (`y − (MVT − SCHED)`).

## 3. Anchored target and the five matched-lane members

**Anchored target.** On the matched lane the anchored members predict `y − proxy`, where `proxy = MVT − AOBT_3`
(`src/prc/train.py`, `matched_target`; the default is `anchored`). The plain-target member predicts `y`
(`matched_target: plain` in its config). Four of the five members (a2o, pao, c2, m4) use the anchored target.

**Blend on matched rows**: `((a2o + pao)/2 + p2o + c2 + m4) / 4`.

| member | model | production config |
|---|---|---|
| a2o | anchored LightGBM, tuned, gate retention (L6) and LIRF copy mixture (cmix) | `configs/main_flseq_osm_nolirf_l2_opt_l6_cmix.yaml` |
| pao | per-airport anchored LightGBM, L6 | `configs/main_flseq_osm_nolirf_l2_pa_opt_l6.yaml` |
| p2o | plain-target LightGBM, L6 | `configs/main_flseq_osm_nolirf_plainy_l2_opt_l6.yaml` |
| c2 | CatBoost | `configs/main_flseq_osm_nolirf_cb_l2.yaml` |
| m4 | MLP, L6 | `configs/main_flseq_osm_nolirf_l2_mlp_l6.yaml` |

For the final set, a2o, pao, p2o and m4 were run with 3 seeds each and c2 with 1 seed (`FINAL_SUBMISSION.md` §2.2).

**Features**: schedule and NM clocks, neighbouring departures and runway sequence (`src/prc/features.py`,
`src/prc/features_extra.py`, `src/prc/levers.py`), stand/arrival occupancy, hourly METAR weather
(`src/prc/weather.py`), and the OSM stand-to-runway distance (`src/prc/osm.py`).

## 4. Unmatched lane

The unmatched lane is model B, blended at weight 0.25 with an "NM-blind" model outside LIRF. At LIRF it uses a gated
copy mixture (`ucmix`) on rows with 0 < MVT − SCHED < 8 h. In the final-set chain both are applied as lane-B deltas
in the overlay stage (§7; `FINAL_SUBMISSION.md` §2.2 step 4).

## 5. Learned ADS-B residual layer

- A LightGBM fitted on 2025 to `clip(y − b, ±3600)`, where `b` is the blend.
- Its inputs are ground-movement features from adsb.lol traces: push, first move, stand dwell, post-push stops and
  positions. These are joined to departures by callsign, or by position for rows without a callsign join.
- It corrects rows with ADS-B coverage only. Its correction is clipped to ±1800 s and is 0 at LTFM.
- Components: `G12A`, `G7A` (LFPG), `GL4` (LIRF), `AIR`, `POS`, `RULE`
  (`docs/final_2026-10-11/adsb_layer_README.md` §1). `RULE` is the positional rule of submission v34 (`b + w (a − b)`
  on positional-join rule rows); `docs/DEVELOPMENT_HISTORY.md` §2.2 records how that rule was motivated.
- Code: `scripts/adsb_*.py`, `analysis/local_adsb_*.py`, `analysis/local_posjoin_*.py`, and the final-set
  rebuild `docs/final_2026-10-11/adsb_layer.py` with its README `docs/final_2026-10-11/adsb_layer_README.md`.

## 6. Appear tier and v48's EDDF coverage-matched delta

### 6.1 Appear tier

- Eight columns describing the aircraft's ADS-B position and timing relative to its own stand
  (`scripts/adsb_appear.py`), from OSM `parking_position` plus a 2026 ADS-B stand map
  (`docs/appear_tier_2026-10-05/plan.md` §0 lists the columns).
- They are added to the residual model and applied as the increment `c_A12(b) − c_P12(b)`: what the 12-month
  appear residual (A12) adds on top of a plain 12-month residual (P12). Code: `docs/appear_tier_2026-10-05/`
  (`final_adsb.py` for the final set).
- In the final file the appear tier is in the January/July rows (from v48) and, as the increment, in 226,935
  February/June rows. 2026-06-11 goes through the A12 layer instead, and only 78 appear rows exist for that day. See
  `docs/FINAL_SUBMISSION.md` §2.3 and §2.5.

### 6.2 v48's EDDF coverage-matched delta

- **The problem.** In 2026, adsb.lol has fewer than half the 2025 ground fixes per EDDF departure: 76–125 in 2025
  against 35 in 2026 (`handoff/2026-10-08/COV26_EDDF.md`).
- **The fix.**
  - The 2025 EDDF fixes were thinned to the 2026 density, counted per 100 m cell
    (`docs/appear_tier_2026-10-05/plan_cov26.md`, `cov_thin.py`).
  - The 12-month appear residual was retrained on them (A12thin).
  - The delta `c_A12thin(b) − c_A12(b)` was added on the 2026 EDDF rows only:
    `handoff/2026-10-08/cov26_eddf_delta_vs_v47.parquet`, 31,774 rows, mean −3.8 s.
- **CV.** On thinned EDDF rows, with January and July 2025 held out: EDDF −1.13 s [−1.73, −0.46], all rows −0.19 s
  [−0.39, −0.02] (`docs/appear_tier_2026-10-05/results.md` §18).
- **Leaderboard.** v48 put it on C15 together with the 12-month appear delta (`handoff/2026-10-08/ON_C15.md`). The two
  together moved the leaderboard from 234.4831 (v47) to 234.4099 (`docs/DEVELOPMENT_HISTORY.md` §1). The EDDF part
  was not scored on its own.
- **What was not adopted.**
  - Re-weighting the training months by coverage similarity was tested separately and rejected; no delta was built
    from it (`docs/covw_2026-10-08/README.md`).
  - Thinning at test time at airports whose coverage became denser was rejected too (`results.md` §20).
- **In the February/June rows** the same term enters through the increment of §6.1: `--increment` uses
  `c_A12thin(b) − c_P12(b)` at EDDF (`docs/FINAL_SUBMISSION.md` §2.5).

## 7. Overlays and the LOBT clip

Applied after the residual layer by `docs/final_2026-10-11/apply_overlays.py`, in this order
(`docs/FINAL_SUBMISSION.md` §2.2 step 4, `docs/final_2026-10-11/overlays_README.md`):

- lane-B deltas: NM-blind and ucmix (§4);
- LIRF unmatched copy band: 7 to 13.5 h late, predict `MVT − SCHED`;
- LIRF unmatched time-zone class: per-prefix q blend;
- LIRF matched schedule-copy overlay: `p' = p + 0.4·q·(MVT − SCHED − p)`, where q comes from a 2025 classifier;
- **LOBT clip** last: project onto `[MVT − LOBT − 3606, MVT − LOBT + 3606]`. This window holds on every matched
  2025 row (`docs/SOLUTION_OVERVIEW.md` §4).

The rule C7r ("one-day" class) is not applied; it was dropped after v41 (`docs/DEVELOPMENT_HISTORY.md` §2.3).

## 8. v48: how the January/July rows were built

- v48 = C15 (v47) + the 12-month appear delta + the EDDF coverage-matched delta (§6.2;
  `handoff/2026-10-08/ON_C15.md`).
- v48 itself is the end of a chain of per-submission delta layers, v33 → v48. See `docs/SOLUTION_OVERVIEW.md` §7,
  `docs/batch_2026-10-06/README.md`, `handoff/2026-10-08/ON_C15.md` and
  `docs/final_2026-10-11/adsb_layer_README.md` §1.
- The step that combined v48's Jan/Jul rows with the Feb/Jun chain rows into one file was not committed as a script
  when the files were made (`docs/FINAL_SUBMISSION.md` §6). `scripts/final/assemble_final_v48.py` reconstructs it
  from the committed post-layer files and is checked row for row against the committed files (its docstring;
  `docs/FINAL_SUBMISSION.md` §5.2).

## 9. Evidence per stage (offline measurements)

This table was README §10 until 2026-10-15; README §8 now shows a selection of its rows.
Only numbers that are in committed documents appear here; each row names its source. The ΔRMSE values are in
seconds, and negative means better. The CIs are the 95 % bootstrap intervals that each source states. These are
2025 offline measurements: each row compares one change against the base of its day, so the rows **do not add up**.

**Abbreviations.**
- *pooled OOF*: the 2025 out-of-fold predictions of the `jan`, `jul` and `forward` month folds, 672,428 rows.
- *Jan+Jul ADS-B CV*: Jan + Jul 2025, 344,419 rows, leave-days-out (5 folds of days), day-block bootstrap.
- `R = docs/competitor_research_2026-09-21/results/`, `M = docs/research_2026-10-05_measure/`,
  `A = docs/appear_tier_2026-10-05/results.md`.

| stage | change measured | ΔRMSE [95 % CI] | data | source |
|---|---|---|---|---|
| member p2o (plain target) | blend H2 → H2 + p2o, weight 0.5. Alone, p2o is worse than the anchored model by +0.93 [+0.59, +1.31] | −0.78 [−1.00, −0.60] | pooled OOF | `R/plainy-blend.md` |
| member c2 (CatBoost) | + c2, equal weights 1/3. Alone +0.94 [+0.49, +1.56] | −1.75 [−2.08, −1.47] | pooled OOF | `R/catboost-blend.md` |
| lever L2 (forward NM windows) | in all three members | −0.32 [−0.45, −0.20] | pooled OOF | `R/blend-l2.md` |
| member pao (per-airport) | blend + pao. Alone vs a2: +0.00 [−0.54, +0.52] | −0.77 [−0.92, −0.64] | pooled OOF | `R/per-airport-member.md` |
| retune of the anchored member | blend `hybrid_blend_l2_pa` → `…_opt`. The member alone gains −2.41 [−2.85, −2.02] | −0.89 [−1.10, −0.72] | pooled OOF | `R/optuna-retune.md` |
| member m4 (MLP) | + m4, equal weights 1/4. Alone it is worse than M1 by +5.59 [+4.67, +6.69] | −0.75 [−0.93, −0.58] | pooled OOF | `R/mlp-member.md` |
| lever L6 (gate retention) | one anchored member, `ret_base` → `ret_l6` | −0.76 [−0.95, −0.56] | `forward` fold, 328,009 rows | `M/l6_verdict.txt`, `M/l6_plan.md` |
| cmix (LIRF copy mixture in a2o) | a2o member. The strict verdict is REJECT: only 1 of 10 airports improves | −0.49 [−0.83, −0.15] | pooled OOF | `M/cmix_verdict.txt` |
| lane B: NM-blind (w 0.25) | non-LIRF unmatched rows (7,841): −7.73 [−20.41, −4.10] | whole frame −0.42 | pooled OOF | `docs/research_2026-10-06_nmblind/verdict.txt`, `plan.md` |
| lane B: ucmix (LIRF unmatched) | LIRF unmatched rows (537): −143.58 [−253.82, −37.74] | whole frame −1.60 [−2.95, −0.40] | pooled OOF | `docs/research_2026-10-07_ucmix/verdict.txt`, `plan.md` |
| learned ADS-B residual | on the H2 base: RMSE 316.97 → 302.94. Gain 8,697.9 s² [7,691.4, 9,828.8] | −14.03 | Jan+Jul ADS-B CV | `R/adsb-learned-residual.md` §2 |
| learned ADS-B residual, refit | on the R6-cbopt base: RMSE 311.705 → 298.596. Gain 8,000.6 s² [7,007.3, 9,060.8] | −13.11 | Jan+Jul ADS-B CV | `R/adsb-learned-residual.md` §14 |
| appear tier (4 months) | appear vs plain residual | −1.858 [−2.542, −1.346] | Jan+Jul ADS-B CV | `A` §1 |
| appear tier (12 months, A12) | vs the production mix of the day | −0.804 [−0.924, −0.692] | 2025, 12 months | `A` §12 |
| appear tier on held-out Feb+Jun 2025 | A12 vs P12 | −1.434 [−1.997, −1.056] | Feb, Jun 2025 held out | `A` §21 |
| EDDF coverage delta (in v48) | A12 trained on thinned EDDF fixes. EDDF rows −1.13 [−1.73, −0.46] | all rows −0.19 [−0.39, −0.02] | Jan+Jul 2025, held out | `A` §18 |
| LIRF matched copy overlay (0.4·q) | 2025 Jan+Jul check: gain 390 s² per row (MSE; Jan 15, Jul 375). It changes 26,277 Jan/Jul 2026 rows | — | Jan+Jul 2025 | `docs/posjoin_2026-10-05/deeper_checks.md` |
| LIRF unmatched copy band | 34 of 34 such 2025 rows are schedule copies | — | 2025 | `docs/SOLUTION_OVERVIEW.md` §4 |
| LOBT clip ±3606 s | the identity window holds on every matched 2025 row (2.06 M) | — | 2025 | `docs/SOLUTION_OVERVIEW.md` §4 |
