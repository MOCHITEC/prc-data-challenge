# Positional ADS-B overlay on the MLP + learned base (2026-10-05)

Not submitted.

## Source

The overlay comes from branch `claude/laughing-cray-vvwh0k`, where v34 = v33 + overlay scored LB 246.7493
(v33: 250.9496). Three scripts are copied from that branch unchanged:

* `analysis/local_adsb_posjoin.py`
* `analysis/local_dwell_end.py`
* `analysis/local_posjoin_overlay.py`

Its documentation is in `docs/posjoin_2026-10-05/`:

* `README.md`
* `deeper_checks.md`
* `replacements_v33.csv`

## What the overlay does

It changes 430 unmatched 2026 departures, outside LIRF, that have no callsign join. A take-off time + runway
heading + type join finds the aircraft.

| rows | rule | 2025 check |
|---|---|---|
| 332 | push signature, no later stop ≥ 900 s: `MVT − t_move` | RMSE 108 s, against 357 s for the model, n = 225 |
| 98 | later stop ≥ 900 s: `base + 0.75·(a − base)` | 84 % within 5 min, RMSE 765 s, n = 263 |

## Base dependency

The 98 long-stop rows read the base prediction. The 332 push rows overwrite it. Every other row passes
through unchanged.

## Reproduction

`$SCRATCH` was this session's `scratchpad/adsb`, which holds the 62 2026 fix and `_joined` files. The steps:

1. `local_adsb_posjoin.py` on all 62 2026 tags.
2. `local_dwell_end.py 2026`, which gave 781 rows.
3. `local_posjoin_overlay.py` on v33.

Against the real v34 downloaded from the team bucket, the result has max |diff| 0 and 0 differing rows; the frames
are equal.

## The candidate

The same overlay applied to `outputs/hybrid_blend_l2_pa_opt_mlp_adsb_learned.parquet` gives
`outputs/v35_candidate_mlp_adsb_posjoin.parquet`.

Validation:

* 344,841 rows;
* schema `MVT_ID_mvt` Float64, `TAXITIME_SEC_mvt` Int32;
* ids and order equal `data/raw/submitting.parquet`;
* 0 nulls and 0 negatives.

Rows changed vs the candidate:

* 430 rows change, the same id set as v33 → v34. The squared change is 1,623 MSE in board units, as for v34.
* The candidate equals v33 on all 430 rows. The MLP blend and the residual refit leave these unmatched,
  callsign-less rows unchanged. As a result, the 430 new values equal v34's exactly, including the 98 w = 0.75
  rows.
* The candidate differs from v33 on 328,657 other rows. The candidate's own CV gain and the overlay's 430 rows
  are therefore disjoint.
