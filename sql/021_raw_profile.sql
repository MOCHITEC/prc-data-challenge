-- 021_raw_profile.sql  Answers to docs/01 open questions U1, U1b, U5, U6, U7, U8, U11 right after RAW load.
-- Each SELECT is one MCP call. Results go to docs/data_profile.md.

-- U5: airports actually present (10 or 11?)
SELECT "ADEP_mvt" AS airport, COUNT(*) AS n FROM PRC_DC2026.RAW.TRAINING_MOVEMENTS WHERE "PHASE_mvt" = 'DEP' GROUP BY 1 ORDER BY 1;
SELECT "ADES_mvt" AS airport, COUNT(*) AS n FROM PRC_DC2026.RAW.TRAINING_MOVEMENTS WHERE "PHASE_mvt" = 'ARR' GROUP BY 1 ORDER BY 1;

-- U7: row count / uniqueness / NM match rate / inconsistency rate (training)
SELECT COUNT(*) AS n, COUNT(DISTINCT "MVT_ID_mvt") AS n_ids,
       AVG(IFF("FLIGHT_ID_mvt" IS NULL, 1, 0)) AS unmatched_rate,
       AVG(IFF("FLIGHT_ID_mvt" IS NOT NULL AND "ADEP_flt" <> "ADEP_mvt", 1, 0)) AS adep_mismatch_rate,
       AVG(IFF("FLIGHT_ID_mvt" IS NOT NULL AND "AIRCRAFT_TYPE_flt" <> "AIRCRAFT_TYPE_mvt", 1, 0)) AS actype_mismatch_rate
FROM PRC_DC2026.RAW.TRAINING_MOVEMENTS WHERE "PHASE_mvt" = 'DEP';
SELECT "ADEP_mvt" AS airport, AVG(IFF("FLIGHT_ID_mvt" IS NULL, 1, 0)) AS unmatched_rate, COUNT(*) AS n
FROM PRC_DC2026.RAW.TRAINING_MOVEMENTS WHERE "PHASE_mvt" = 'DEP' GROUP BY 1 ORDER BY 1;

-- U6: target distribution and anomalies (training DEP)
SELECT COUNT(*) AS n,
       AVG(IFF("TAXITIME_SEC_mvt" IS NULL, 1, 0)) AS null_rate,
       AVG(IFF("TAXITIME_SEC_mvt" < 0, 1, 0)) AS neg_rate,
       AVG(IFF("TAXITIME_SEC_mvt" > 7200, 1, 0)) AS gt2h_rate,
       APPROX_PERCENTILE("TAXITIME_SEC_mvt", 0.01) AS p01, APPROX_PERCENTILE("TAXITIME_SEC_mvt", 0.10) AS p10,
       APPROX_PERCENTILE("TAXITIME_SEC_mvt", 0.50) AS p50, APPROX_PERCENTILE("TAXITIME_SEC_mvt", 0.90) AS p90,
       APPROX_PERCENTILE("TAXITIME_SEC_mvt", 0.99) AS p99, STDDEV("TAXITIME_SEC_mvt") AS sd
FROM PRC_DC2026.RAW.TRAINING_MOVEMENTS WHERE "PHASE_mvt" = 'DEP';
SELECT "ADEP_mvt" AS airport, MEDIAN("TAXITIME_SEC_mvt") AS med, STDDEV("TAXITIME_SEC_mvt") AS sd, COUNT(*) AS n
FROM PRC_DC2026.RAW.TRAINING_MOVEMENTS WHERE "PHASE_mvt" = 'DEP' AND "TAXITIME_SEC_mvt" BETWEEN 0 AND 7200 GROUP BY 1 ORDER BY 1;
-- definition check: TAXITIME == MVT_TIME - BLOCK_TIME ?
SELECT AVG(IFF(ABS(DATEDIFF('second', "BLOCK_TIME_UTC_mvt", "MVT_TIME_UTC_mvt") - "TAXITIME_SEC_mvt") <= 1, 1, 0)) AS def_consistent_rate
FROM PRC_DC2026.RAW.TRAINING_MOVEMENTS WHERE "PHASE_mvt" = 'DEP' AND "TAXITIME_SEC_mvt" IS NOT NULL;

-- U1 / U11: what remains in ranking DEP rows
SELECT COUNT(*) AS n,
       AVG(IFF("MVT_TIME_UTC_mvt"   IS NULL, 1, 0)) AS mvt_time_null,
       AVG(IFF("BLOCK_TIME_UTC_mvt" IS NULL, 1, 0)) AS block_time_null,
       AVG(IFF("TAXITIME_SEC_mvt"   IS NULL, 1, 0)) AS taxitime_null,
       AVG(IFF("RUNWAY_mvt" IS NULL, 1, 0)) AS runway_null,
       AVG(IFF("STAND_mvt"  IS NULL, 1, 0)) AS stand_null,
       AVG(IFF("SCHED_TIME_UTC_mvt" IS NULL, 1, 0)) AS sched_null,
       AVG(IFF("FLIGHT_ID_mvt" IS NULL, 1, 0)) AS unmatched,
       AVG(IFF("AOBT_3_flt" IS NULL, 1, 0)) AS aobt3_null,
       AVG(IFF("LOBT_flt"   IS NULL, 1, 0)) AS lobt_null,
       AVG(IFF("EOBT_1_flt" IS NULL, 1, 0)) AS eobt1_null,
       AVG(IFF("IOBT_flt"   IS NULL, 1, 0)) AS iobt_null
FROM PRC_DC2026.RAW.RANKING WHERE "PHASE_mvt" = 'DEP';
-- same null profile in training DEP rows (for parity)
SELECT AVG(IFF("AOBT_3_flt" IS NULL, 1, 0)) AS aobt3_null, AVG(IFF("LOBT_flt" IS NULL, 1, 0)) AS lobt_null,
       AVG(IFF("EOBT_1_flt" IS NULL, 1, 0)) AS eobt1_null, AVG(IFF("RUNWAY_mvt" IS NULL, 1, 0)) AS runway_null,
       AVG(IFF("STAND_mvt" IS NULL, 1, 0)) AS stand_null
FROM PRC_DC2026.RAW.TRAINING_MOVEMENTS WHERE "PHASE_mvt" = 'DEP';

-- U1: information content of NM times vs actual off-block (training DEP, matched rows)
SELECT "ADEP_mvt" AS airport,
       MEDIAN(DATEDIFF('second', "AOBT_3_flt", "BLOCK_TIME_UTC_mvt")) AS med_block_minus_aobt3,
       AVG(IFF(ABS(DATEDIFF('second', "AOBT_3_flt", "BLOCK_TIME_UTC_mvt")) <= 60, 1, 0)) AS aobt3_within_60s,
       MEDIAN(DATEDIFF('second', "LOBT_flt", "BLOCK_TIME_UTC_mvt")) AS med_block_minus_lobt,
       AVG(IFF(ABS(DATEDIFF('second', "LOBT_flt", "BLOCK_TIME_UTC_mvt")) <= 300, 1, 0)) AS lobt_within_5min,
       COUNT(*) AS n
FROM PRC_DC2026.RAW.TRAINING_MOVEMENTS
WHERE "PHASE_mvt" = 'DEP' AND "AOBT_3_flt" IS NOT NULL GROUP BY 1 ORDER BY 1;
-- derived-value signature: is AOBT_3 just MVT_TIME minus a constant?  (low spread => derived)
SELECT "ADEP_mvt" AS airport, STDDEV(DATEDIFF('second', "AOBT_3_flt", "MVT_TIME_UTC_mvt")) AS sd_mvt_minus_aobt3,
       APPROX_PERCENTILE(DATEDIFF('second', "AOBT_3_flt", "MVT_TIME_UTC_mvt"), 0.5) AS med_mvt_minus_aobt3
FROM PRC_DC2026.RAW.TRAINING_MOVEMENTS WHERE "PHASE_mvt" = 'DEP' AND "AOBT_3_flt" IS NOT NULL GROUP BY 1 ORDER BY 1;
-- ARR semantics of ARVT_3: landing (MVT_TIME) or in-block (BLOCK_TIME)?
SELECT AVG(IFF(ABS(DATEDIFF('second', "ARVT_3_flt", "MVT_TIME_UTC_mvt")) <= 60, 1, 0)) AS arvt3_is_landing,
       AVG(IFF(ABS(DATEDIFF('second', "ARVT_3_flt", "BLOCK_TIME_UTC_mvt")) <= 60, 1, 0)) AS arvt3_is_inblock
FROM PRC_DC2026.RAW.TRAINING_MOVEMENTS WHERE "PHASE_mvt" = 'ARR' AND "ARVT_3_flt" IS NOT NULL;

-- U1b: share of ranking DEP rows whose FLIGHT_ID also appears as an ARR row in ranking (intra-network pairs)
SELECT AVG(IFF(a."FLIGHT_ID_mvt" IS NOT NULL, 1, 0)) AS dep_with_arr_row_rate
FROM PRC_DC2026.RAW.RANKING d
LEFT JOIN (SELECT DISTINCT "FLIGHT_ID_mvt" FROM PRC_DC2026.RAW.RANKING WHERE "PHASE_mvt" = 'ARR') a
  ON a."FLIGHT_ID_mvt" = d."FLIGHT_ID_mvt"
WHERE d."PHASE_mvt" = 'DEP';

-- U8: are FLIGHT_mvt / CALLSIGN_flt anonymised? (eyeball a sample; real callsigns look like BAW6VB)
SELECT "FLIGHT_mvt", "CALLSIGN_flt", "AIRCRAFT_OPERATOR_flt" FROM PRC_DC2026.RAW.TRAINING_MOVEMENTS SAMPLE (20 ROWS);

-- Timestamp sanity: min/max per month must match pyarrow (timezone trap, docs/02 §5)
SELECT _SRC_FILE, MIN("MVT_TIME_UTC_mvt") AS mn, MAX("MVT_TIME_UTC_mvt") AS mx FROM PRC_DC2026.RAW.TRAINING_MOVEMENTS GROUP BY 1 ORDER BY 1;
