# Neighbour ADS-B taxi features in the residual (2026-10-07): rejected

These are 20 label-free columns: neighbour ADS-B taxi duration (t_mvt − t_move) counts, means, medians and excess
over a 2025 airport×runway×hour median in windows around the take-off, aircraft taxiing at take-off and push, and
ADS-B coverage. Every single-column adversarial AUC is ≤ 0.59.

Gain (s² per row, positive = better):

| | 12-month CV (decides) | months7 CV |
|---|---|---|
| excluding LIRF | −36.2 [−62.0, −13.3] | −43.3 [−69.9, −16.6] |
| January, excluding LIRF | −135 | −155 |

EGLL is −294. ADS-B coverage-dependent congestion does not carry over between months (EGLL coverage 0.79 → 1.00,
LFPG 0.17 → 0.36), so the 2026 delta was not built. Script: `analysis/local_nbr_adsb.py`.

## Per-member fl_/seq_ fix (C13 candidate)
Each production member was refitted with the original and the union-fixed ranking rows; Σ Δ² / N of the change:

| member | Σ Δ² / N |
|---|---|
| pao | 95 |
| p2o | 104 |
| CatBoost c2 | 420 |
| MLP m4 | 561 |
| v42's LGB-mean approximation | 63 |

a2o was OOM-killed, so pao stands in for it (weight 1/8). The precise blend is Σ Δ² / N = 149, against 63
applied; correlation 0.62.

Correction (scaled by 1 − 0.4 q at LIRF): Σ Δ² / N 81, mean −0.32 s. Expected +120–240 MSE (0.25–0.5 s), using the
board/estimate ratio seen for C10.

File: `outputs/c13_v42_flfix_members.parquet` (v42 file + correction, LOBT clip).
Script: `analysis/local_flfix_member.py`.
