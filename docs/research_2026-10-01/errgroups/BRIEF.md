# Shared brief: the large-error groups on GENUINE rows (PRC Data Challenge 2026, taxi-out RMSE)

## Context
Predict TAXITIME_SEC_mvt = MVT_TIME_UTC_mvt - BLOCK_TIME_UTC_mvt (s) for 344,841 departures in Jan+Jul 2026 at 10 EU airports.
Our best LB 252.43 (leader 219.6). Ranking DEP rows have BLOCK_TIME_UTC_mvt and TAXITIME_SEC_mvt blanked; every other column,
and all ARR rows (with labels), remain. The current production model (v23, branch origin/claude/beautiful-davinci-ehugk4):
model A on NM-matched rows predicts g = y - (MVT - AOBT_3) with LightGBM (3-member blend: anchored LGB, plain-y LGB, anchored
CatBoost) + a learned ADS-B residual stage (open adsb.lol data; pushback detected from ground tracks), which is switched OFF at
LTFM because 2025 LTFM has only 38 ADS-B training rows (2026: 5,161). Features include "flseq" sequence features, OSM stand
distance, stand_arrival_gap, weather etc. Read docs on that branch with
`git -C /home/user/prc-data-challenge show origin/claude/beautiful-davinci-ehugk4:<path>` (docs/ tree; start from docs/README.md,
docs/handoff.md, docs/features_log.md, docs/competitor_research_2026-09-21/results/adsb-learned-residual.md, docs/analysis/*.md).
DO NOT redo what is already measured there - build on it, and say explicitly when something is already covered by v23.

## Error anatomy we just measured (genuine rows = BLOCK not a copy of a known clock, no day shift; Jan+Jul 2025, matched lane,
simple LightGBM trained on other months, RMSE 273.8; v23 is better, so treat shares as indicative)
- BLOCK - AOBT_3 < -10 min (airport off-block much EARLIER than NM AOBT): 5.2% rows, 34% SSE, bias +360..+820 s (under-predict)
- BLOCK - AOBT_3 in +10..+30 min: 4.1% rows, 9.5% SSE, bias -310 s
- long taxis y >= 30 min: ~5% rows, ~32% SSE, strongly under-predicted
- NM delay (AOBT3 - SCHED) 30 min..3 h: 19% rows, 40% SSE
- airports: LTFM 18.2% of SSE (no ADS-B correction!), LFPG 14.9%, LIRF 13.6% (runway 25: 10.8%), EGLL 13.5%; heavy aircraft x1.3;
  specific disruption days (LIRF 2025-07-13, LTFM 07-26, LFPG 01-17) ~1% SSE each; hour of day flat.

## Shared inputs (read-only)
- /tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/errgroups/oof_matched_janjul.parquet:
  MVT_ID_mvt, ADEP_mvt, TAXITIME_SEC_mvt, pred, err (= y - pred), is_copy, is_rollover, genuine. Baseline for your analysis.
- .../errgroups/feature_gain_top40.txt (baseline model's top features)
- /home/user/prc-data-challenge/data/work/feat_train.parquet (2,085,047 DEP rows x 132 cols, v12-era features incl. raw columns,
  MVT_ID_mvt join key), data/work/feat_rank.parquet (344,841 ranking DEP rows), data/raw/*.parquet (raw DEP+ARR, training 2025 and
  ranking 2026).

## Environment rules (several agents run concurrently on a 15 GB / 4-core box)
- Python: /home/user/prc-data-challenge/.venv/bin/python (polars, numpy, pandas, duckdb, lightgbm, scikit-learn).
  Use POLARS_MAX_THREADS=1, POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage, lightgbm num_threads<=1.
  Use scan_parquet + column selection. Keep peak memory under ~2.5 GB. Models only on filtered subsets / few hundred rounds.
- Do NOT modify the repo. Work under .../scratchpad/errgroups/<your-angle>/.
- Validation: anything you claim as a gain must be measured on Jan+Jul 2025 with training on other months (or leave-one-month-out),
  as a paired difference vs the baseline pred on the same rows (bootstrap CI). Check that a feature is computable on the 2026
  ranking file with the same null rate (feature-availability parity: no DEP BLOCK/TAXITIME of other rows may be used).
- Compliance: no leaderboard probing; don't use side channels of blanked columns (ID order, second jitter).
- Distinguish measured facts from inference; give script paths.

## Deliverable
Final message in Japanese (<= ~1200 words): findings ranked by expected LB value, each with measured evidence, the parity check,
what v23 already covers, and a concrete next experiment (config/feature) with an estimated gain range in seconds.
