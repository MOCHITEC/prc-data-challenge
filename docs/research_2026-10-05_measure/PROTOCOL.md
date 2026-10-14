# Measurement protocol for competitor ideas (2026-10-05)
Goal: MEASURE, on our own data, whether each idea from other teams' repos carries signal our model lacks. Do not dismiss ideas because the source team ranks below us; they did different analyses.

Data (read-only, never copy to the git repo): /home/user/prc-data-challenge/data/work/feat_train.parquet (2025, labelled, 2.09M rows, 191 cols), data/work/feat_rank.parquet (2026 ranking set, same columns, labels blank), data/raw/*.parquet (raw incl. arrivals). Label TAXITIME_SEC_mvt = MVT_TIME_UTC_mvt - BLOCK_TIME_UTC_mvt. NM proxy column mvt_minus_aobt3 (= MVT - AOBT_3). Python: /home/user/prc-data-challenge/.venv/bin/python (polars, lightgbm, numpy).
Our model's code: /home/user/prc-data-challenge/src/prc (features.py, features_extra.py, levers.py, train.py). Check whether an idea already exists before measuring it.

Residual harness: /tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/extdata/derived/harness.py
  - base.parquet = OOF predictions of a matched-lane LightGBM on Jan+Jul 2025 (column err = y - oof), rollover rows removed.
  - test(feat_df, cols, mode='day') trains a residual LGB with CONTROL categoricals vs CONTROL+cols; reports dMSE (s^2) with a day-cluster bootstrap CI. A single noise column moves dMSE by about -36..+59 s^2, so |d| < 100 s^2 is not a clear signal. Read it before using; copy it into your own folder and extend if needed (e.g. to also cover the unmatched lane).
  - For post-processing ideas (mixtures, bounds), evaluate directly on the OOF: build the needed classifier/regressor on other months only (leave-month-out), apply to Jan/Jul OOF rows, report RMSE change with an airport-day block bootstrap CI, and the split into rows with y<=3600 vs y>3600.

For EVERY new column also report the 2025-vs-2026 drift: single-column adversarial AUC (LightGBM, 2-fold, column alone) between feat_train (Jan+Jul 2025 rows) and feat_rank (Jan+Jul 2026). Compute features identically on both frames. AUC > 0.65 = drift risk; prefer within-day ranks / day-mean-removed versions.

Constraints: 4 CPU cores and 15 GB RAM are SHARED with other agents. Use num_threads=1 for LightGBM, load only needed columns, sample if needed. Do not modify the git repo. Rules: no use of MVT_ID/FLIGHT_ID ordering, no second-level jitter or timestamp rounding/precision of blanked columns, no leaderboard information, no label-aggregate external data (EUROCONTROL pre-departure delay / taxi-out additional time series). Never copy competitor code (GPL); ideas only.

Output: a markdown report with, per idea: what exactly you built (label-free? fitted on which rows?), dMSE with CI and in RMSE seconds on the measured rows, scaled estimate of LB effect (rows affected share), AUC drift, verdict TRY/NO, and implementation notes for our pipeline (which file/function, train/rank parity). Rank ideas by expected LB gain.
