> **Status (2026-10-14): superseded — see [`docs/FINAL_SUBMISSION.md`](FINAL_SUBMISSION.md)** for the final file. This map describes the pipeline at v47; the final file's Jan/Jul rows are v48 (v47 + 12-month appear + EDDF coverage delta).

# Solution overview (state of submission v47, LB 234.48, 2026-10-08)

This is a map of the final pipeline: what each stage does, why it is there, and where its code and evidence
live. External data sources and licences are in `docs/external_data.md`.

## 1. Problem framing
- Target: `TAXITIME_SEC_mvt = MVT − BLOCK` for departures at 10 airports. Scored on Jan+Jul 2026; trained on all
  of 2025. Metric: RMSE.
- Two lanes:
  - **Matched lane (≈ 98.5 % of rows):** the NM flight record supplies AOBT_3. The base models predict y − proxy,
    where proxy = MVT − AOBT_3 (the "anchored" target).
  - **Unmatched lane (≈ 1.5 %):** no NM record. A separate model B is used, with a schedule reparametrisation
    at LIRF.
- RMSE is dominated by a few label artefacts (schedule copies, time-zone and date errors). These get dedicated,
  validated handling (§4); everything else is modelled.

## 2. Base models (`src/prc/`, `configs/`)
- **Features:**
  - flight and schedule clocks;
  - neighbouring-departure context (`nbr_*`, `seq_*`, forward NM clocks `fw_*` = lever L2);
  - gate retention (L6, `fe_retention`);
  - arrival/stand occupancy (`occ_*`, `ta_*`);
  - label encodings fitted on training months;
  - hourly METAR weather;
  - OSM stand-to-runway distance.
- **Matched-lane blend:** `((0.5·a2o + 0.5·pao) + p2o + c2 + m4) / 4`
  - a2o: anchored LightGBM, tuned, with the LIRF copy mixture (cmix);
  - pao: per-airport LightGBM;
  - p2o: plain-target LightGBM;
  - c2: CatBoost;
  - m4: MLP;
  - L6 is in a2o / pao / p2o / m4.
- **Unmatched lane:** model B (LightGBM).
  - Blended at weight 0.25 with an NM-blind model outside LIRF.
  - At LIRF, a gated copy mixture (`ucmix`) on rows with 0 < MVT−SCHED < 8 h.

## 3. Learned ADS-B residual (`scripts/adsb_learn_*.py`, `analysis/local_adsb_*`)
- **Input:** adsb.lol `globe_history` traces (ODbL), within 8 km of each airport.
- **Join:** per-departure ground-movement features (push, first move, stand dwell, post-push stops, positions),
  joined by callsign, then by position (take-off time ± 90 s, runway heading, aircraft type).
- **Model:** a LightGBM on y − b with V2 parameters, trained on all 12 months of 2025 (7 months at LFPG).
- **Where it applies:** only on rows with a ground fix. c = 0 at LTFM (no coverage); swaps exclude LIRF.

## 4. Label-artefact handling (all validated month-out on 2025)

| artefact | handling | evidence |
|---|---|---|
| LIRF matched schedule copies (BLOCK = SCHED, ~17 %) | classifier q, p' = p + 0.4·q·(MVT−SCHED − p) | `analysis/local_lirf_copy_overlay.py` |
| LIRF unmatched, 7–13.5 h late (34/34 copies in 2025) | predict MVT − SCHED | `docs/lirf_wrap_2026-10-06` |
| LIRF unmatched, BLOCK = off-block − Rome UTC offset (time-zone class; per-operator rates) | per-prefix q blend | same |
| LIRF unmatched, 0–8 h late (copy vs normal taxi) | `ucmix` gated mixture | peaceful-babbage `docs/research_2026-10-07_ucmix` |
| \|y − (MVT−LOBT)\| ≤ 3606 s (holds on all 2.06 M matched 2025 rows) | LOBT identity clip | `analysis/local_lobt_clip.py` |

Tried and rejected: the LIRF "one-day" class (v41, −0.75 s on the board), stamp-rate features, and copy blends at
other airports.

## 5. Train/test consistency (the largest single gain, −1.36 s on the board)
- **Problem:** `fl_nm_rate_other` and the pairq-based `seq_*` columns were computed over the whole frame:
  12 months for training, 2 months for ranking. That shifted their 2026 distribution (EDDF share = 1.0:
  0.40 → 0.72).
- **Fix:** ranking-row values are computed on the masked union of 2025 + 2026 movements.
- **Applied per member:** CatBoost and the MLP react 5–9× more than LightGBM.
- **Audit:** a full feature audit found no other inconsistency (`docs/shift_2026-10-06`). An audit of the ADS-B
  residual found none either; it found a receiver-coverage shift in the data instead (`docs/adsb_audit_2026-10-07`).

## 6. Validation protocol
- **Base models:** month folds (jan, jul, forward; local aso/fmj).
- **Residual and overlays:** leave-days-out K = 5 or leave-one-month-out, with a day-block bootstrap.
- **Acceptance:** pre-registered rule: CI above 0, most months positive, top-50 rows ≤ 50 % of the gain.
  Few-row, heavy-tailed gains are rejected after v41.
- **Board transfer seen so far:**
  - consistency fixes: about 2× their CV estimate;
  - model-level additions: about 1×;
  - ADS-B residual add-ons: 10–35 %.

## 7. Board history (this branch's submissions)

| version | content | LB |
|---|---|---|
| v40 | months-7/12 residual swaps | 238.77 |
| v42 | C10 + apron + L6/cmix + NM-blind + MLP-L6 | 236.44 |
| v43–v45 | isolation runs (apron / C10 / members) | 238.66 / 237.41 / 238.35 |
| v47 | per-member consistency fix, 3-seed L6 deltas, ucmix | **234.48** |

Process note: version numbers are shared by all sessions. List the team bucket right before uploading and use
max + 1 (two collisions happened; see `docs/batch_2026-10-06/README.md`).
