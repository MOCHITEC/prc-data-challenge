# 2025 → 2026 shift (2026-10-06): a train/test mismatch in `fl_nm_rate_other` (handoff to the base pipeline)

## Finding
`src/prc/features_extra.py::flight_features` computes `fl_nm_rate_other` (the share of the other movements with
the same flight number that have a FLIGHT_ID match) over the whole frame. That frame is 12 months for training but
Jan+Jul only for ranking. With fewer movements per flight number, the rate collapses to exactly 1.0 far more often:

| share of rows with rate = 1.0 | matched lane, 2025 Jan+Jul (train frame) | 2026 |
|---|---|---|
| EDDF | 0.40 | 0.72 |
| EGLL | 0.45 | 0.82 |
| LFPG | 0.38 | 0.68 |
| unmatched lane, EDDF / LIRF | 0.17 / 0.21 | 0.48 / 0.57 |

The `seq_*` columns built on pairq share the issue, but their effect is small.

Adversarial validation: AUC 0.99 on v40's features. The top shifter is `fl_nm_rate_other`.

## Cost estimate (emulation)
Recompute the column for Jan+Jul 2025 over a 2-month frame, then score a proxy LightGBM of the base trained on the
other 10 months.

- **Total cost:** +991 / +753 MSE (2 seeds), positive in both months. `fl_nm_rate_other` alone: +980 / +696.
- **Matched lane outside LIRF:** +18 / +46.
- **LIRF matched lane:** +204 / +287.
- **Unmatched lane at LIRF:** +930, but 5 rows carry it. The size is unreliable, though the sign is consistent.

## Proposed fix (base pipeline; not applied here)
Compute `fl_nm_rate_other` for ranking rows over 2025 + 2026 departures together. Take pairq for the `seq_*`
columns from the 2025 table. Then re-predict with the trained base models.

The proxy moves the 383 LIRF unmatched rows by −117 s on average. Other airports move by 27 s or less. The
matched-lane effect needs a real re-predict.

## Other results of this pass
- **Unseen/rare stands:** a stand-group fallback is worth +8.8 MSE (leave-one-month-out), about 45 MSE in 2026.
  Not worth it. The new EDDM stands 103–108 (1 177 rows, July 2026) behave like stands 110–113.
- **Drift in the shifting features:** implies ≈ 0 bias. A residual model on the shifters fails leave-one-month-out.

Scripts: `analysis/local_shift_{unseen,adversarial,pool,pool_ablate,pool_laneB,pool_2026}.py`.

## Full feature audit (same day)
Every feature used by the production configs was checked:

- **Frame test:** a 2-month rebuild of Jan+Jul 2025 was compared with the 12-month build, row by row.
- **Label-blind test:** the same rebuild with departure labels blanked; 0 columns differ.
- **Distribution comparison:** null share, exact 0/1 share, mean and quantiles per airport, lane and feature.
- **Emulated cost:** a proxy LightGBM fitted on the other 10 months.

| family | verdict |
|---|---|
| clock / static / NM offsets / categoricals / `fl_fmt_code` / `fl_len` / `dup_*` / weather / OSM | consistent |
| `fl_nm_rate_other`, pairq `seq_hw_prev_pairq` / `seq_hw_excess` / `seq_burst_*` | **inconsistent**: +556 / +839 MSE emulated (2 seeds), both lanes, both months |
| context windows, headways, `nbr_*`, other `seq_*` | edge effects on ≤ 294 rows only (−0.2 / −2.5) |
| label encodings (`ref_*`, `arh_*`, `apt_med`) | in-sample self-inclusion; cost not reliable (LIRF lane B swings by thousands); test only in a real forward fold |

**Fix:** for the ranking rows, run `extra_candidate_features` on the masked union of the 2025 and 2026 movements,
one airport at a time. On 2025 this union equals the 12-month frame, so the emulated cost of the fix is 0. EDDF
matched lane, share of `fl_nm_rate_other` = 1.0: training 0.40, ranking before the fix 0.72, after 0.37.

The re-prediction with the fixed columns is `analysis/local_flrate_fix.py`. Each base config is fitted once and
predicts the original and the fixed ranking rows together, giving a per-row delta.
Scripts: `analysis/local_audit_{raw,frame,labelblind,dist,cost,fix2026}.py`.
