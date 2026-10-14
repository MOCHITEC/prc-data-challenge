# External ideas A/B/C (2026-10-06)

Overlays were scored on the 7-month OOF against a v40-equivalent p (copy blend 0.4 + LOBT clip). Gains are SSE
reduction / N, in MSE (~480 MSE ≈ 1 s).

## A + C: train the matched member without LIRF schedule copies and y > 80k, then serve the full mixture (A = 1)

Real `predict_fold` refits (main_flseq_osm_nolirf, n_seeds 1), against base + copy 0.4:

| variant | Jan | Jul |
|---|---|---|
| x1 + A = 1 | +327 [+143, +517] | +835 [+483, +1 183] |
| x1 raw (no mixture) | −1 202 | −5 998 |
| base with A retuned (no refit) | ≤ +154 | ≤ +154 |

- **Where the gain is:** LIRF +227 / +703 and non-LIRF +100 / +132. The non-LIRF part is within seed noise; the
  LIRF part is not (seed control moves LIRF by +12 / +155).
- **Concentration:** the gain sits in the top 50 LIRF rows, as the production copy blend does. Those rows come from
  a pool of about 2 000 LIRF rows with a SCHED-vs-proxy gap above 1 h, and 2026 has the same size of pool.
- **Expected board effect:** about +300 to +500 MSE (0.6–1.0 s). It needs every matched member refitted.
- **Overlap:** this is the same mechanism as peaceful-babbage's `copy_mix_airports: [LIRF]` (cmix, a2o −0.49 s),
  which is independent confirmation. Take cmix INSTEAD of our A = 0.4 overlay on LIRF matched rows, never both.

## C alone (isotonic q / retuned A)
Raw q is already calibrated. On Jan+Jul the result is ≈ 0. Dropped.

## B (LOBT)
- Phoenix rule Z: −8.9 (also negative in R22 of investigation_2026-09-16).
- LOBT expert on |AOBT_3 − LOBT| > 3 600: +60, all 7 months positive, but 48 of the 60 come from the top 50 rows.
  Low priority, batch only.

Scripts: `analysis/local_ext_overlays_bc.py`, `analysis/local_ext_hygiene_refit.py`.
