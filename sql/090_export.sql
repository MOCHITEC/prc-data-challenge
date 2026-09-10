-- 090_export.sql
-- Snowflake -> sandbox export path (docs/04 section 1). The sandbox cannot connect to Snowflake directly
-- (the egress proxy does not support certificate-pinned clients), so feature tables leave the warehouse as
-- parquet on an SSE-encrypted internal stage and are downloaded over a pre-signed URL with curl.
--
-- One statement per MCP call.

-- VERIFIED 2026-09-10: 50,000 rows unloaded, pre-signed URL fetched with curl from the sandbox (HTTP 200,
-- 780 KB) and parsed with pyarrow. Two type notes from that probe:
--   * Snowflake NUMBER lands as parquet DECIMAL, which pyarrow reads as Python Decimal. Cast in the SELECT:
--     ID::INT, VAL::DOUBLE - otherwise every numeric feature arrives as Decimal and slows the trainer down.
--   * TIMESTAMP_NTZ lands as an isAdjustedToUTC timestamp, i.e. pyarrow returns tz-aware UTC datetimes.
--     Convert with .dt.replace_time_zone(None) after loading if naive UTC is wanted (docs/02 section 5).
--
-- 1) Unload a query result as a single parquet file (adjust the SELECT and the file name).
COPY INTO @PRC_DC2026.ML.EXPORT_STAGE/feat_train_v1.parquet
FROM (SELECT * FROM PRC_DC2026.FEAT.F_DEP_V1)  -- cast NUMBER columns to ::DOUBLE / ::INT here
FILE_FORMAT = (TYPE = PARQUET)
HEADER = TRUE
SINGLE = TRUE
MAX_FILE_SIZE = 5368709120
OVERWRITE = TRUE;

-- 2) Mint a pre-signed URL (seconds to live). The sandbox then runs: curl -sS -o feat_train_v1.parquet "<url>"
SELECT GET_PRESIGNED_URL(@PRC_DC2026.ML.EXPORT_STAGE, 'feat_train_v1.parquet', 3600) AS url;

-- 3) Housekeeping: the export stage is scratch space, so purge it when the file has been downloaded.
LIST @PRC_DC2026.ML.EXPORT_STAGE;
-- REMOVE @PRC_DC2026.ML.EXPORT_STAGE/feat_train_v1.parquet;

-- If a COPY exceeds the 60 s client timeout, confirm completion instead of re-running it blindly:
SELECT QUERY_ID, QUERY_TEXT, EXECUTION_STATUS, TOTAL_ELAPSED_TIME/1000 AS SEC
FROM TABLE(PRC_DC2026.INFORMATION_SCHEMA.QUERY_HISTORY_BY_USER(RESULT_LIMIT => 20))
WHERE QUERY_TEXT ILIKE 'COPY INTO @PRC_DC2026.ML.EXPORT_STAGE%'
ORDER BY START_TIME DESC;
