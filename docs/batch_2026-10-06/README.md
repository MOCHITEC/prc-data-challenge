# Parked batch (2026-10-06): submit only when the combined expectation reaches 2–3 s

The user's rule: hold everything until the validated items add up to an expected 2–3 s. Then rebuild them on the
latest team file and submit once.

| item | expected | file / source |
|---|---|---|
| C10 = v40 + fl_/seq_ train-test fix + LIRF time-zone class + LIRF copy band | ≈ 0.66 s | `outputs/c10_v40_flfix_tz_copy.parquet` (local); docs/shift_2026-10-06, docs/lirf_wrap_2026-10-06 |
| L6 gate retention (peaceful-babbage), L6-only delta, not cmix | ≈ 0.2–0.6 s | to be committed by that session under `handoff/` |
| ADS-B residual + apron (rescat A3) | ≈ 0.6–0.8 s | `outputs/parts/apron_delta_12m.parquet`; docs/rescat_2026-10-06 |
| neighbour-clock residual (drop if L6 goes in) | ≈ 0.04 s | `outputs/parts/nbr_clock_delta.parquet`; docs/nbr_clock_2026-10-06 |
| LFPG unmatched positive-only residual | ≈ 0.03 s | `analysis/local_lfpg_unm.py lfpg_apply` |

Do not bundle: the LIRF one-day class (v41 lost 0.75 s), cmix on top of the LIRF copy overlay, or R6 on top of
L6/CatBoost members.

## Combined candidate C11 (built 2026-10-06 ~16:00 UTC)
`outputs/c11_v40_c10_apron_l6cmix_nmb.parquet` = v40 + four deltas, then the LOBT clip (27 rows). Pre-submission
gate: PASS.

| part | rows changed | Σ Δ² / N | expected |
|---|---|---|---|
| C10 (fl_/seq_ fix, LIRF time-zone class, copy band) | 81 830 | 462 | ≈ +316 MSE |
| apron in the ADS-B residual (12m-matched swap) | 230 454 | 652 | ≈ +300–400 |
| L6 + cmix member deltas (peaceful-babbage configs re-run here, weights a2o 1/8, pao 1/8, p2o 1/4; n_seeds 1, so roughly −100 MSE of extra seed noise) | 322 957 | 282 | ≈ +200–290 |
| NM-blind model-B blend (w 0.25, unmatched non-LIRF rows; `analysis/local_nmb_laneB.py`) | 4 860 | 104 | ≈ +200 |
| **total** | 335 180 | 1 512 | **≈ +1 000–1 200 MSE ≈ 2.1–2.6 s (238.77 → ≈ 236.2–236.7)** |

The neighbour-clock residual is left out (it overlaps L6), and so are the LIRF one-day class and the LFPG item.
