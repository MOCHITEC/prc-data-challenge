# Final evaluation set (2026-10-11)

- The organisers put `final_ranking.parquet` / `final_submitting.parquet` in `prc-2026-datasets` on 2026-10-08 11:04Z.
  - They cover 2026 Jan, Feb, Jun and Jul: 670,790 departures.
  - The old Jan+Jul set (344,841 departures) is included with the same MVT_IDs.
  - MVT_TIME on the Jan/Jul rows moved by up to ±12 s against the old file.
- The final file must be named `resilient-hedgehog_final.parquet` (one unique file).
- The deadline was extended to 2026-10-14 10:00Z (prc-data-challenge-2026.netlify.app).
- The ranking page says organisers monitor attempts to learn from the ranking process. Leaderboard probes (the
  LFPG-July +60 and floor isolation files) were therefore dropped.
- The team bucket already held a `resilient-hedgehog_final.parquet` (2026-10-09, origin unknown).
  - It is a single-pipeline-style file.
  - It differs from v47 by RMS 149 s on Jan/Jul.
  - The organiser scorer failed on it (`dev_sub_template.parquet` FileNotFound).

## Build (workspace `/home/user/prc-final`, members in `/home/user/pb-work/outputs/final`)
- **Members on the final raw dir:**
  - `prc.pipeline --skip-cv`.
  - Feature cache: `pb-work/data/work_final`.
  - The matched rows of `feat_rank` are patched with the union-built fl_/seq_ columns (`analysis/local_audit_fix2026.py`
    run on the final frame).
  - Members: a2o (l2_opt_l6_cmix, 3 seeds), pao_l6 (3), p2o_l6 (3), mlp_l6 (3), cb_l2 (1).
- **Blend:**
  - Matched rows: ((a2o + pao)/2 + p2o + c2 + m4) / 4.
  - Unmatched rows: LGB model B.
- **ADS-B residual layer:** `adsb_layer.py apply` (see `adsb_layer_README.md`).
  - Jan/Jul validation against v47: same row set, correlation 0.93 on the 12-month ground rows.
  - February: all 28 days ingested.
  - June: the ingest stopped at the 2 h background limit and is not yet resumed.
- **Final file:** Jan/Jul = v47; Feb/Jun = blend + layer + LOBT clip.

## Files
- `outputs/final_prov_v47_a2o.parquet`: Jan/Jul = v47, Feb/Jun = a2o alone (unfixed, 1 seed). Gate OK.
- `outputs/final_v2_v47_blend5_adsbFeb.parquet`: Jan/Jul = v47, Feb/Jun = 5-member blend + ADS-B (February only) +
  LOBT clip. Gate OK.
- **Pending:** LIRF overlays, NM-blind and ucmix for Feb/Jun; June ADS-B.

## v3 (2026-10-11 ~21:10Z)
- `outputs/final_v3_v47_full_chain_adsbFeb.parquet`: Jan/Jul = v47; Feb/Jun = blend → ADS-B layer (February) →
  lane-B (NM-blind, ucmix) → LIRF overlays → LOBT clip, using `apply_overlays.py --laneB --lobt-clip --copy-feat unfixed`
  (see `overlays_README.md`). Gate OK.
- **Check:** the same chain on Jan/Jul is within RMS 52 s of v47 (median |d| 16 s). The rest comes from the ADS-B refit
  and air-only proxy and the new MVT noise draw in the final file.
- **Still missing:** June ADS-B (ingest not resumed).
