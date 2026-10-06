# Parked batch (2026-10-06): submit only when the combined expectation reaches 2–3 s

The user's rule: hold everything until the validated items add up to an expected 2–3 s. Then rebuild them on the
latest team file and submit once.

| item | expected | file / source |
|---|---|---|
| C10 = v40 + fl_/seq_ train-test fix + LIRF time-zone class + LIRF copy band | ≈ 0.66 s | `outputs/c10_v40_flfix_tz_copy.parquet` (local); docs/shift_2026-10-06, docs/lirf_wrap_2026-10-06 |
| L6 gate retention (peaceful-babbage), L6-only delta, not cmix | ≈ 0.2–0.6 s | to be committed by that session under `handoff/` |
| neighbour-clock residual (drop if L6 goes in) | ≈ 0.04 s | `outputs/parts/nbr_clock_delta.parquet`; docs/nbr_clock_2026-10-06 |
| LFPG unmatched positive-only residual | ≈ 0.03 s | `analysis/local_lfpg_unm.py lfpg_apply` |

Do not bundle: the LIRF one-day class (v41 lost 0.75 s), cmix on top of the LIRF copy overlay, or R6 on top of
L6/CatBoost members.
