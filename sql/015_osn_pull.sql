-- 015_osn_pull.sql
-- Option C (docs/02 section 6): let Snowflake fetch the official files from the OpenSky MinIO bucket itself,
-- so a redistributed ranking/submitting file can be reloaded without a human in the loop.
--
-- PREREQUISITE (run once, by hand, in Snowsight - never through chat or the MCP tool):
--   CREATE SECRET PRC_DC2026.RAW.OSN_MINIO_CREDS
--     TYPE = PASSWORD USERNAME = '<OSN_ACCESS_KEY>' PASSWORD = '<OSN_SECRET_KEY>';
-- See sql/000_admin_manual.sql step 7.
--
-- One statement per MCP call. Idempotent.

CREATE OR REPLACE NETWORK RULE PRC_DC2026.RAW.OSN_S3_RULE
  MODE = EGRESS
  TYPE = HOST_PORT
  VALUE_LIST = ('s3.opensky-network.org:443')
  COMMENT = 'OpenSky Network MinIO endpoint (competition data + team submission bucket)';

CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION PRC_OSN_EAI
  ALLOWED_NETWORK_RULES = (PRC_DC2026.RAW.OSN_S3_RULE)
  ALLOWED_AUTHENTICATION_SECRETS = (PRC_DC2026.RAW.OSN_MINIO_CREDS)
  ENABLED = TRUE
  COMMENT = 'PRC Data Challenge 2026: pull official parquet files into the internal stage';

-- Fetch one object per call: the MCP client cuts off at 60 s, so never loop over all 14 files in one call.
-- Returns the staged path. Uses the minio SDK from the Snowflake Anaconda channel.
CREATE OR REPLACE PROCEDURE PRC_DC2026.RAW.FETCH_OSN_OBJECT(BUCKET STRING, OBJECT_NAME STRING)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'minio')
  HANDLER = 'run'
  EXTERNAL_ACCESS_INTEGRATIONS = (PRC_OSN_EAI)
  SECRETS = ('cred' = PRC_DC2026.RAW.OSN_MINIO_CREDS)
  COMMENT = 'Copy one object from the OSN MinIO bucket into @PRC_DC2026.RAW.OFFICIAL_STAGE'
AS
$$
import _snowflake
from minio import Minio


def run(session, bucket, object_name):
    creds = _snowflake.get_username_password('cred')
    client = Minio('s3.opensky-network.org', access_key=creds.username, secret_key=creds.password, secure=True)
    response = client.get_object(bucket, object_name)
    try:
        session.file.put_stream(
            response,
            f'@PRC_DC2026.RAW.OFFICIAL_STAGE/{object_name}',
            auto_compress=False,
            overwrite=True,
        )
    finally:
        response.close()
        response.release_conn()
    return f'@PRC_DC2026.RAW.OFFICIAL_STAGE/{object_name}'
$$;

-- List what the bucket holds (one call, small result).
CREATE OR REPLACE PROCEDURE PRC_DC2026.RAW.LIST_OSN_OBJECTS(BUCKET STRING)
  RETURNS ARRAY
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'minio')
  HANDLER = 'run'
  EXTERNAL_ACCESS_INTEGRATIONS = (PRC_OSN_EAI)
  SECRETS = ('cred' = PRC_DC2026.RAW.OSN_MINIO_CREDS)
AS
$$
import _snowflake
from minio import Minio


def run(session, bucket):
    creds = _snowflake.get_username_password('cred')
    client = Minio('s3.opensky-network.org', access_key=creds.username, secret_key=creds.password, secure=True)
    return [f'{o.object_name} {o.size}' for o in client.list_objects(bucket, recursive=True)]
$$;

-- Usage (one call each):
--   CALL PRC_DC2026.RAW.LIST_OSN_OBJECTS('prc-2026-datasets');
--   CALL PRC_DC2026.RAW.FETCH_OSN_OBJECT('prc-2026-datasets', 'ranking.parquet');
--   LIST @PRC_DC2026.RAW.OFFICIAL_STAGE;
-- For the 12 training files, wrap the calls in a TASK so the 60 s client timeout does not matter:
--   CREATE OR REPLACE TASK PRC_DC2026.RAW.T_FETCH_ALL WAREHOUSE = PRC_WH SCHEDULE = NULL AS
--     BEGIN
--       CALL PRC_DC2026.RAW.FETCH_OSN_OBJECT('prc-2026-datasets', 'training_2025-01-01_2025-02-01.parquet');
--       ... one line per file ...
--     END;
--   EXECUTE TASK PRC_DC2026.RAW.T_FETCH_ALL;
--   SELECT NAME, STATE, SCHEDULED_TIME FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY(TASK_NAME=>'T_FETCH_ALL')) ORDER BY SCHEDULED_TIME DESC LIMIT 5;
