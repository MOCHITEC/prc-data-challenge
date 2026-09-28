-- 016_osn_presigned_pull.sql
-- Load the official files into @PRC_DC2026.RAW.OFFICIAL_STAGE WITHOUT storing the OpenSky key in Snowflake
-- (chosen 2026-09-28 over option C / 015, see docs/decisions.md).
--
-- The sandbox holds the key and signs short-lived GET links (prc.io_minio.presigned_urls, 15 min); Snowflake
-- only ever sees those links. A link carries the access key id and a signature but not the secret, and it
-- lands in query history, so it must be short-lived and used once.
--
-- Run AFTER the guard rails in 000_admin_manual.sql (the CALLs below resume a warehouse: it has to be PRC_WH
-- under PRC_RM). The DDL itself uses no warehouse. One statement per MCP call. Idempotent.

CREATE NETWORK RULE IF NOT EXISTS PRC_DC2026.RAW.OSN_S3_RULE
  MODE = EGRESS
  TYPE = HOST_PORT
  VALUE_LIST = ('s3.opensky-network.org:443')
  COMMENT = 'OpenSky Network MinIO endpoint (competition data)';

-- No ALLOWED_AUTHENTICATION_SECRETS: this integration can reach the endpoint but holds no credential.
CREATE EXTERNAL ACCESS INTEGRATION IF NOT EXISTS PRC_OSN_EAI
  ALLOWED_NETWORK_RULES = (PRC_DC2026.RAW.OSN_S3_RULE)
  ENABLED = TRUE
  COMMENT = 'PRC Data Challenge 2026: fetch official parquet files via presigned links (no secret)';

-- Download one presigned link into the stage under FILE_NAME. Retries the 503s the endpoint is prone to.
CREATE OR REPLACE PROCEDURE PRC_DC2026.RAW.FETCH_URL_TO_STAGE(URL STRING, FILE_NAME STRING)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'requests')
  HANDLER = 'run'
  EXTERNAL_ACCESS_INTEGRATIONS = (PRC_OSN_EAI)
  COMMENT = 'Copy one presigned OpenSky object into @PRC_DC2026.RAW.OFFICIAL_STAGE'
AS
$$
import os
import re
import time
import urllib.parse

import requests

STAGE = '@PRC_DC2026.RAW.OFFICIAL_STAGE'


def run(session, url, file_name):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+[.]parquet', file_name):
        raise ValueError(f'unexpected file name: {file_name!r}')
    if urllib.parse.urlparse(url).hostname != 's3.opensky-network.org':
        raise ValueError('only s3.opensky-network.org links are accepted')
    local = os.path.join('/tmp', file_name)
    for attempt in range(4):
        with requests.get(url, stream=True, timeout=(10, 120)) as r:
            if r.status_code == 503 and attempt < 3:
                time.sleep(2 ** (attempt + 1))
                continue
            r.raise_for_status()
            with open(local, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        break
    size = os.path.getsize(local)
    session.file.put(local, STAGE, auto_compress=False, overwrite=True)
    os.remove(local)
    return f'{STAGE}/{file_name} {size} bytes'
$$;

-- Usage: the sandbox prints one CALL per file with fresh links, wrapped in a single anonymous block so it is
-- one MCP statement (the 60 s client cut-off does not stop it server-side; check with LIST afterwards):
--   .venv/bin/python -m prc.io_minio snowflake-fetch
--   EXECUTE IMMEDIATE $$ BEGIN CALL PRC_DC2026.RAW.FETCH_URL_TO_STAGE('<url>', 'ranking.parquet'); ... END; $$;
--   SELECT RELATIVE_PATH, SIZE FROM DIRECTORY(@PRC_DC2026.RAW.OFFICIAL_STAGE) ORDER BY 1;  -- after ALTER STAGE ... REFRESH
-- Expected: 14 files, 330 MB in total (docs/data_profile.md). Then run 020_raw_load.sql.
