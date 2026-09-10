-- 000_admin_manual.sql
-- Run MANUALLY in Snowsight as ACCOUNTADMIN (these statements need account-level privileges that the
-- SYSADMIN role used by the MCP connection does not have). Review each block before running.
-- Nothing here embeds credentials. Replace <...> placeholders.

-- 1) Session time zone for the working user (the account default is America/Los_Angeles).
--    User-level setting also applies to the non-persistent MCP sessions.
ALTER USER <ADMIN_USER> SET TIMEZONE = 'UTC';
--    The MCP session cannot persist USE WAREHOUSE, so make the competition warehouse the default for the
--    working user during the competition (or create a dedicated MCP user with DEFAULT_WAREHOUSE = PRC_WH).
ALTER USER <ADMIN_USER> SET DEFAULT_WAREHOUSE = PRC_WH;

-- 2) Resource monitor for the competition warehouse (created in 010_setup_db.sql).
--    Enable e-mail notifications for ACCOUNTADMIN in Snowsight (Admin > Cost Management) beforehand.
CREATE RESOURCE MONITOR IF NOT EXISTS PRC_RM
  WITH CREDIT_QUOTA = 50
  FREQUENCY = MONTHLY
  START_TIMESTAMP = IMMEDIATELY
  TRIGGERS ON 50 PERCENT DO NOTIFY
           ON 80 PERCENT DO NOTIFY
           ON 100 PERCENT DO SUSPEND
           ON 110 PERCENT DO SUSPEND_IMMEDIATE;
ALTER WAREHOUSE PRC_WH SET RESOURCE_MONITOR = PRC_RM;

-- 3) Account budget (covers serverless / compute-pool spend that resource monitors do not).
--    Snowsight: Admin > Cost Management > Budgets > set spending limit and notification e-mail.

-- 4) Optional hardening (recommended before the repository becomes public):
--    - Network policy restricting logins to known IPs.
--    - Authentication policy requiring MFA for TYPE=PERSON users.
--    - ALTER USER <ADMIN_USER> SET DEFAULT_ROLE = SYSADMIN;  -- avoid ACCOUNTADMIN as the everyday default

-- 5) Working role (CREATE ROLE requires USERADMIN or ACCOUNTADMIN).
CREATE ROLE IF NOT EXISTS PRC_ROLE COMMENT = 'PRC Data Challenge 2026 working role (scoped to PRC_DC2026 and PRC_WH)';
GRANT ROLE PRC_ROLE TO ROLE SYSADMIN;
GRANT USAGE, OPERATE, MONITOR ON WAREHOUSE PRC_WH TO ROLE PRC_ROLE;
GRANT USAGE ON DATABASE PRC_DC2026 TO ROLE PRC_ROLE;
GRANT ALL ON ALL SCHEMAS IN DATABASE PRC_DC2026 TO ROLE PRC_ROLE;
GRANT ALL ON FUTURE TABLES IN DATABASE PRC_DC2026 TO ROLE PRC_ROLE;

-- 6) Dedicated service user for automation (key-pair auth, no password). Public key is not a secret,
--    but generate the key pair on your own machine and keep the private key out of the repository.
-- CREATE USER IF NOT EXISTS PRC_SVC TYPE = SERVICE DEFAULT_ROLE = PRC_ROLE DEFAULT_WAREHOUSE = PRC_WH
--   RSA_PUBLIC_KEY = '<PUBLIC_KEY_BODY>';
-- GRANT ROLE PRC_ROLE TO USER PRC_SVC;

-- 7) OSN MinIO credentials as a Snowflake SECRET for option C (Snowflake pulls data directly).
--    Type this in Snowsight yourself; do not send it through chat or the MCP tool.
-- CREATE SECRET IF NOT EXISTS PRC_DC2026.RAW.OSN_MINIO_CREDS
--   TYPE = PASSWORD USERNAME = '<OSN_ACCESS_KEY>' PASSWORD = '<OSN_SECRET_KEY>';
-- GRANT USAGE ON SECRET PRC_DC2026.RAW.OSN_MINIO_CREDS TO ROLE PRC_ROLE;
