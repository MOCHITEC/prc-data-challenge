-- 000_admin_manual.sql
-- Run MANUALLY in Snowsight as ACCOUNTADMIN (these statements need account-level privileges that the
-- SYSADMIN role used by the MCP connection does not have). Review each block before running.
-- Nothing here embeds credentials. Replace <...> placeholders.

-- Budget decided 2026-09-28 (docs/decisions.md): 20 credits for the whole project, hard stop on PRC_WH.

-- 1) Route every MCP query to the competition warehouse. The MCP session cannot persist USE WAREHOUSE, so the
--    working user's default decides where the credits land (and whether PRC_RM below can stop them).
--    Side effect: other tools that log in as this user without naming a warehouse also land on PRC_WH and
--    stop with it at the cap. Revert when the project ends: ALTER USER <ADMIN_USER> SET DEFAULT_WAREHOUSE = <PREVIOUS_WH>;
ALTER USER <ADMIN_USER> SET DEFAULT_WAREHOUSE = PRC_WH;
--    Time zone: NOT changed at user level, because the same user runs other work whose LTZ output would move.
--    The loaders normalise explicitly instead (CONVERT_TIMEZONE('UTC', col)::TIMESTAMP_NTZ in STG, docs/02 §5).
--    Only if this user is dedicated to the project: ALTER USER <ADMIN_USER> SET TIMEZONE = 'UTC';

-- 2) Resource monitor = the hard stop. FREQUENCY = NEVER: the quota is for the whole project and never resets.
--    E-mail: Snowsight > your profile > Settings > Notifications, enable resource monitor e-mails beforehand.
CREATE RESOURCE MONITOR IF NOT EXISTS PRC_RM
  WITH CREDIT_QUOTA = 20
  FREQUENCY = NEVER
  START_TIMESTAMP = IMMEDIATELY
  TRIGGERS ON 50 PERCENT DO NOTIFY
           ON 80 PERCENT DO NOTIFY
           ON 100 PERCENT DO SUSPEND
           ON 110 PERCENT DO SUSPEND_IMMEDIATE;
ALTER WAREHOUSE PRC_WH SET RESOURCE_MONITOR = PRC_RM;

-- 3) Optional: a Budget as a second, notify-only alarm (it also sees serverless spend, which PRC does not plan
--    to use, and it lags by hours). Snowsight: Admin > Cost Management > Budgets > + Budget,
--    scope = database PRC_DC2026 + warehouse PRC_WH, limit 20 credits/month, your e-mail.
--    Not used by this project because PRC_RM cannot stop them: Notebooks on Container Runtime (compute pools),
--    Cortex AI functions, Snowsight Copilot, serverless tasks, automatic clustering, query acceleration.

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
