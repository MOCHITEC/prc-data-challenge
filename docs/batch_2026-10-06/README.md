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

## Submitted: v42 = C12 (2026-10-07)
C12 = C11 + L6 in the MLP member (peaceful-babbage `main_flseq_osm_nolirf_l2_mlp` → `_l2_mlp_l6`, 3 seeds each,
weight 1/4 on matched rows), then the LOBT clip (26 rows).

- vs v40: 336 939 rows changed, Σ Δ² / N = 1 840, mean −3.1 s.
- Expected about +1 000–1 300 MSE (238.77 → ≈ 236).

## Board results (2026-10-07)
v42 = **236.4369** (−2.33 s vs v40 238.7712; rank 25/232). The first read of the board for v42 said 238.7644; that
read was wrong, so the isolation submissions were not needed. They still give the attribution (all vs v40):

| sub | content | score | Δ MSE | Δ s |
|---|---|---|---|---|
| v44 | C10 only (fl_/seq_ fix, LIRF time-zone class, copy band) | 237.4070 | +650 | −1.36 |
| v45 | peaceful-babbage member deltas only (L6+cmix, L6 pao/p2o/MLP) | 238.3508 | +201 | −0.42 |
| v43 | apron residual swap only | 238.6626 | +52 | −0.11 |
| v42 − (v43+v44+v45) | ≈ NM-blind + interactions | | ≈ +207 | ≈ −0.43 |

- The pipeline-consistency fix (C10) beat its estimate by about 2×.
- Apron transferred poorly (+52 vs +300–400 expected). Candidate reasons: CV on the 7-month model but applied as a
  12-month swap, or the effect does not generalise to 2026.
- Label-encoding cross-fit (`local_xfit_enc.py`): forward fold all-rows dMSE +295 [−384, +1170]. Rejected.

## Submitted: v47 = C15 (2026-10-08)
C15 = v42 + three changes, then the LOBT clip:

- **Per-member fl_/seq_ fix correction** (C13).
- **L6 member deltas rebuilt with 3 seeds** (C14): the member change Σ Δ² / N fell from ≈ 2 200 to ≈ 1 050, so
  about half of it was seed noise. Expected ≈ +88 MSE.
- **peaceful-babbage ucmix** on LIRF unmatched rows with 0 < ms < 8 h (`analysis/local_ucmix_laneB.py`): 356 rows.
  The 6 time-zone-class rows and the 3 copy-band rows keep our values. Flights 2–4 h late: median prediction
  6 416 → 5 791 s.

vs v42: 315 356 rows changed, Σ Δ² / N = 1 131. Expected −1.0 to −1.7 s.

## v47 result and a version-collision incident (2026-10-08)
v47 = C15 scored **234.4831** (−1.95 s vs v42; rank 20/237).

**Incident: two version numbers collided with another session.** Another session uploaded its own
`resilient-hedgehog_v47.parquet` at ~01:21Z, which scored 236.3563. Ours, uploaded at 01:28Z, overwrote the object,
and the board entry (same submissionId) was re-scored with our file. The first v42 read (238.7644) was the same
thing: another session's v42 was overwritten by ours. Both overwritten files scored worse than our best, so the
board best is unaffected. But `prc.submit` only checks the local log for used versions.

**Rule from now on:** list the team bucket right before uploading and use max(existing) + 1.
