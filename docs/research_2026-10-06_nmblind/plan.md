# NM-blind blend for model B (nmblind_weight) — pre-registration (2026-10-06)

## Background
- **Source:** merry-mushroom v5 (LB −5.48 together with a deep CatBoost).
- **Prototype (`proto.py`, `proto.log`):** an LGB fitted on all departures with 0 < y ≤ 7200, with NM-derived columns removed (600k-row sample per fold). Blended into model B on non-LIRF unmatched rows, at w = 0.25:
  - pooled: 1273.8 → 1267.8, −6.0 [−16.7, −2.9];
  - per fold: jan −0.2, jul −12.4, forward −22.9;
  - whole frame ≈ −0.33 s.
- w = 0.5 is larger overall, but jan gets worse (+1.6). **The weight is fixed at 0.25 before the run.**

## Change
`nmblind_weight` (train.py `_nmblind_blend`; default 0, existing ids unchanged).
- Inside each fold, an LGB is fitted on all training-side departures with 0 < y ≤ 7200, using model B's columns and parameters.
- On unmatched target rows outside the reparameterised airports (LIRF): pred = 0.75·B + 0.25·NM-blind.

## Arms (all folds)
- **base:** main_flseq_osm_nolirf_l2_opt-7244b3004f (already run).
- **nmb:** main_flseq_osm_nolirf_l2_opt_nmb.

## Verdict (fixed before the run)
**ADOPT** only if all of the following hold:
1. Matched rows and LIRF unmatched rows are bit-identical.
2. On non-LIRF unmatched rows, the airport-day block-bootstrap CI excludes 0 on the improving side.
3. jul and forward improve on those rows.
4. jan is not worse by more than 2 s on those rows.

## Result (2026-10-06, `verdict.txt`)
**ADOPT.** main_flseq_osm_nolirf_l2_opt-7244b3004f → main_flseq_osm_nolirf_l2_opt_nmb-97a0e31e57.

| condition | result |
|---|---|
| 1. matched and LIRF unmatched rows bit-identical | met |
| 2. non-LIRF unmatched rows improve, CI excludes 0 | 1273.8 → 1266.1, −7.73 s [−20.41, −4.10] |
| 3. jul and forward improve | jul −12.26, forward −31.29 |
| 4. jan not worse by more than 2 s | jan −1.56 (it improves) |

- All 9 airports improve: EDDF −8.8, EDDM −27.5, EGLL −19.4, EHAM −14.1, LEBL −32.1, LEMD −18.3, LFPG −3.6, LSZH −23.5, LTFM −18.4.
- Whole frame: **−0.42 s** (270.77 vs 271.20).
- Model B is shared by every member, so this applies to the blend's unmatched lane once. The blend-level effect is about the same, roughly −0.4 s.
