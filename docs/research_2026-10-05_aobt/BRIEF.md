# Research brief: How reliable is the reported off-block time (AOBT) in European airport data, and what does it do to the taxi-out KPI?

Goal: an aviation-industry-relevant study (target JOAS), independent of the competition ranking. Audience: EUROCONTROL PRU / SES performance
monitoring, airports, A-CDM analysts. Key question: the SES KPI "Additional taxi-out time" and A-CDM evaluations rely on AOBT reported by
airports (APDF). How much of the reported AOBT is a measurement, how much is a copy / fill / derived value / definitional variant, and how
much does that bias taxi-out statistics per airport?

Data (local, read-only): /home/user/prc-data-challenge/data/raw/training_2025-*.parquet (12 months 2025, DEP+ARR rows, 10 airports:
EDDF EDDM EGLL EHAM LEBL LEMD LFPG LIRF LSZH LTFM). Columns: MVT_TIME_UTC_mvt (take-off for DEP), BLOCK_TIME_UTC_mvt (airport-reported
off-block = AOBT for DEP), SCHED_TIME_UTC_mvt, RUNWAY_mvt, STAND_mvt, AIRCRAFT_TYPE_mvt, FLIGHT_mvt; NM flight columns AOBT_3_flt (NM actual
off-block), EOBT_1_flt, LOBT_flt, IOBT_flt, ARVT_*, AIRCRAFT_OPERATOR_flt (hashed), MARKET_SEGMENT_flt etc. TAXITIME_SEC_mvt = MVT - BLOCK.
The competition data may not be redistributed before the organisers publish it: keep raw data local; aggregated statistics are fine.
Known facts from our earlier analysis (verify, don't just trust): ~21% of DEP BLOCK within +-60 s of NM AOBT_3, ~7.6% of SCHED, ~3.8% of
EOBT_1, ~0.5% LOBT; LIRF writes SCHED as BLOCK for many unmatched flights and stamps long-delayed flights with the scheduled LOCAL date
(one-day rollover); at LTFM, NM AOBT_3 looks like take-off minus a fixed runway-specific taxi time (19 or 24 min for ~40% of rows);
5 airports (EDDM EGLL LFPG LIRF LTFM) store MVT/BLOCK on a minute grid with a +-6 s anonymisation jitter applied in opposite directions.
Prior docs: /home/user/prc-data-challenge/docs/06_error_anatomy.md, docs/research_2026-10-01/ (rollover/, errgroups/), and on branch
origin/claude/model-b-and-arrival-features docs/investigation_2026-09-16/results/ (block-day-rollover.md, sched-filled-block.md,
timestamp-precision-clusters.md). Literature already identified: EUROCONTROL APDF Specification Ed.1.0 (2019) AOBT quality checks; PRU ATXOT
indicator documentation (2023 & older); Goens et al. OPDI (SID 2024); Schultz, Rosenow, Olive A-CDM-lite (SID 2019); Olive, Waltert,
Schultz (ATRDS 2025, Zurich); Assaia AOBT accuracy white paper.

Rules: do not modify the repo; work under /tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/aobt/<angle>/.
Python: /home/user/prc-data-challenge/.venv/bin/python (polars, numpy, lightgbm, sklearn, matplotlib). POLARS_MAX_THREADS=2,
POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage, memory < 4 GB (3 agents run concurrently on a 15 GB box). Save scripts and key
outputs (CSV tables, PNG figures) in your folder. Mark [S] measured vs [X] inference. Deliverable: Japanese report (<= ~1200 words):
findings with numbers, figures produced, how strong the evidence is, what is novel vs literature, and what a paper section would claim.
