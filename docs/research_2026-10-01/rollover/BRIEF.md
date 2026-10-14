# Shared brief: the one-day "rollover" label corruption (PRC Data Challenge 2026, taxi-out RMSE)

## Task context
Predict TAXITIME_SEC_mvt = MVT_TIME_UTC_mvt - BLOCK_TIME_UTC_mvt (seconds) for 344,841 departures (Jan+Jul 2026,
10 EU airports). Metric RMSE. Our best LB is 252.43. Labels are scored *as reported by the airport*, including
corruption. In ranking.parquet the DEP rows have BLOCK_TIME_UTC_mvt and TAXITIME_SEC_mvt blanked (100% null);
every other column remains. ARR rows in ranking.parquet keep their labels (taxi-in = MVT - BLOCK, in-block).

## The problem we want to crack
A handful of departure labels equal ~86,400 + a normal taxi (BLOCK date one day off). Training 2025 has 16 DEP rows
with y > 72,000 s: 14 LIRF (13 with no NM record), 1 LSZH, 1 LFPG. In Jan+Jul 2025 there are 7. Missing one such row
by ~86,000 s costs ~21,600 MSE on the LB (~ -44 s of RMSE if fixed); LIRF rows are currently predicted near
ms = MVT - SCHED (errors 3k-30k s). Goal: find ANY prediction-time-visible signal (columns that are not blanked,
other rows in the same file, open external data) that identifies which 2026 departures are day-shifted, or that
sharpens P(day-shift | x). Also characterise the mechanism.

Known facts (docs/investigation_2026-09-16/results/block-day-rollover.md on branch origin/claude/beautiful-davinci-ehugk4):
- 16/16 rows: date(MVT) - date(BLOCK) = 1 day (UTC). 13/16: date(BLOCK)=date(SCHED) and date(MVT)=date(SCHED)+1.
  3/16 (an LIRF row, the LSZH row, the LFPG row): date(BLOCK)=date(SCHED)-1, date(MVT)=date(SCHED).
- LSZH row: BLOCK = SCHED - 86,402 s exactly (schedule fill with wrong date?). LFPG row: BLOCK 01-18 19:30, SCHED 01-19 18:25, MVT 01-19 18:54.
- All 16 BLOCK seconds-of-minute within 6 s of a minute boundary.
- LIRF unmatched rows with ms >= 12 h: 29 in training, 62% schedule-fill (y == ms), 38% rollover; ms bucket gives a gradient.
- An overlay predicting 0.379*87,300 + 0.621*ms on the 11 ranking LIRF unmatched rows with ms >= 12 h made LB WORSE
  by +9.6 s (293.98 vs 284.36) -> those 11 were probably NOT rollovers.
- LIRF unmatched rows have odd FLIGHT_mvt strings (e.g. ITYTY680, CHHHH438, NOSOS739) and null AIRCRAFT_TYPE_mvt.
- y = MVT - BLOCK holds on all training rows. ms = MVT - SCHED is visible at prediction time.

## Environment rules (IMPORTANT - several agents run concurrently on a 15 GB / 4-core box)
- Repo: /home/user/prc-data-challenge. Python: /home/user/prc-data-challenge/.venv/bin/python (polars, numpy, pandas, duckdb,
  lightgbm, scikit-learn available). Run with POLARS_MAX_THREADS=1 and POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage.
- Raw data: /home/user/prc-data-challenge/data/raw/training_2025-MM-01_*.parquet (12 files, DEP+ARR), ranking.parquet
  (DEP 344,841 blanked + ARR 344,693 labelled), submitting.parquet. Use pl.scan_parquet + select only needed columns.
  Keep your process peak memory under ~2.5 GB. No heavy model training (small models on filtered subsets are fine).
- Do NOT modify anything in the repo (no edits, no commits). Put scripts/outputs under your own folder:
  /tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/rollover/<your-angle>/
- Read prior team work with `git -C /home/user/prc-data-challenge show origin/claude/beautiful-davinci-ehugk4:<path>`
  (docs/ tree). Do not redo what is already measured there; build on it.
- Compliance: do NOT use leaderboard scores to infer labels and do not propose probe submissions. If you find a signal
  that is a leak of the blanked columns (e.g. row order or IDs encoding BLOCK_TIME), report it clearly flagged as
  "compliance-sensitive" - do not build on it; the human decides.
- Every number must come from code you ran (state the script path). Distinguish measured facts vs inference.

## Deliverable
Final message (Japanese, <= ~1200 words): findings ranked by usefulness, each with the measured evidence, precision/recall
of any candidate rule on 2025 (leave-one-month-out where relevant), how many 2026 ranking DEP rows it flags, and the
expected LB effect range under L2 (posterior-mean prediction, not hard swap). End with "what would change our prediction".
