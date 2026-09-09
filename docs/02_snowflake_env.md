# 02. Snowflake 環境と設計前提

最終確認日: 2026-09-09。読み取りクエリ（SHOW / DESCRIBE / SELECT）のみで確認し、オブジェクトの作成・変更は行っていない。独立レビュー 1 回反映済み。

> **公開リポジトリ向けの記述方針**: 本リポジトリは賞金条件により GPLv3 で公開する。アカウントロケータ・ログイン名・既存の社内オブジェクト名・課金実績などの実値は載せず、プレースホルダで記す。実値はリポジトリ外の非公開メモで管理する。

## 1. アカウント概要

| 項目 | 値 |
|---|---|
| クラウド / リージョン | AWS ap-northeast-1（東京） |
| エディション | Enterprise（`SNOWFLAKE.ORGANIZATION_USAGE.ACCOUNTS` で確認） |
| Snowflake バージョン | 10.32.x |
| 作業ユーザー | `<ADMIN_USER>`（`ACCOUNTADMIN` / `ORGADMIN` を直接保持。`DEFAULT_ROLE = ACCOUNTADMIN`、`DEFAULT_SECONDARY_ROLES = ALL`） |
| MCP セッションの現在ロール | `SYSADMIN`（ユーザーへの直接付与ではなく ACCOUNTADMIN 経由の継承。セカンダリロール未適用） |
| MCP セッションの既定ウェアハウス / DB / スキーマ | `<DEFAULT_WH>`（社内分析用の X-Small）/ なし / なし |
| アカウント既定 `TIMEZONE` | **America/Los_Angeles**（Snowflake 既定のまま。§5 で対処） |
| `DATA_RETENTION_TIME_IN_DAYS` | 1（既定） |
| `STATEMENT_TIMEOUT_IN_SECONDS` | 172,800（既定 48 時間） |

`SYSADMIN` のアカウントレベル権限: `CREATE DATABASE`, `CREATE WAREHOUSE`, `CREATE COMPUTE POOL`, `CREATE EXTERNAL ACCESS INTEGRATION`, `EXECUTE TASK`。`CREATE NETWORK RULE` / `CREATE SECRET` はスキーマレベル権限なので、SYSADMIN が所有するスキーマ内なら SYSADMIN で作成できる。
→ コンペ用 DB・ウェアハウス・外部アクセス統合・network rule・secret は **SYSADMIN で完結**。`ACCOUNTADMIN` が必要なのはリソースモニター、Budget、ストレージ統合・API 統合（Git 連携）、ネットワークポリシー、ユーザー属性変更に限られる。

Snowsight にログインした場合の既定は ACCOUNTADMIN（セカンダリロール ALL）で、MCP セッション（SYSADMIN、セカンダリロールなし）と権限範囲が異なる。

## 2. 既存オブジェクトとコスト（要点のみ）

- コンペ関連のオブジェクトはまだ存在しない。既存の業務用 DB・外部アクセス統合・secret・内部ステージが別途あるが、コンペとは無関係。**コンペ用は新規 DB `PRC_DC2026` 配下で完結させ、既存 DB を参照しない。**
- Open-Meteo（`api.open-meteo.com`）向けの External Access Integration が既に存在する。過去データ取得に使う `archive-api.open-meteo.com` は network rule に含まれていないため、使う場合はコンペ DB 内に新規 rule/integration を作る。
- Snowflake 作成のシステムコンピュートプール（CPU / GPU）が存在し、Notebook の Container Runtime に使われる。**直近 30 日の課金の最大項目はこのコンピュートプールの稼働（約 10 日間ほぼ連続）で、ウェアハウス課金の約 3 倍**だった。リソースモニターはウェアハウス専用でこれを止められない。
- 直近 30 日の総消費は数十クレジット規模。ストレージは 1 GB 未満。
- リソースモニター・Budget はともに未設定。ネットワークポリシー・認証ポリシーも未設定。

## 3. Snowflake 内で使える Python パッケージ（Anaconda チャネル、`INFORMATION_SCHEMA.PACKAGES`）

Python runtime **3.11** 基準（本サンドボックスと同じ）。利用可能な版は runtime 別に異なる。`VERSION` 列は文字列なので `MAX(VERSION)` は辞書順になる（1.9.x > 1.54.0 と誤判定）。各セグメントを数値化して比較すること。

| パッケージ | 3.11 での最大版 | 備考 |
|---|---|---|
| lightgbm | 4.7.0 | 3.10〜3.14 同一 |
| xgboost | 3.2.0 | 3.12 以上では 3.3.0 |
| catboost | 1.2.10 | |
| scikit-learn | 1.9.0 | 3.10 では 1.7.2 |
| optuna | 4.6.0 | |
| shap | 0.51.0 | 3.10 では 0.48.0 |
| polars | 1.41.2 | |
| pandas | 3.0.5 | 3.10 では 2.3.3 |
| snowflake-ml-python | 1.54.0 | |
| snowflake-snowpark-python | 1.54.0 | |
| boto3 / minio / s3fs / requests | あり | 案 C（§6）で使用 |

→ ストアドプロシージャ / UDF 作成時は `RUNTIME_VERSION = '3.11'` を明示する。学習は原則サンドボックス（§7）で行い、Snowflake 内で学習する場合は Container Runtime（`CPU_X64_S` 3 vCPU/13 GiB 0.11 credits/h、`CPU_X64_M` 6 vCPU/28 GiB 0.22 credits/h）を第一候補にする。Snowpark-optimized warehouse の `MEMORY_16X` は Medium 以上（6 credits/h）でしか選べないため例外扱い。

## 4. MCP `run_sql` ツールの制約（実測）と分担

| 制約 | 実測 | 対処 |
|---|---|---|
| 1 回 1 ステートメント | 複数文は不可 | DDL/DML を 1 文ずつ。複数文スクリプトは `EXECUTE IMMEDIATE FROM @stage/file.sql`（内部ステージにアップロードした SQL ファイルを 1 文で実行。10 MB 未満・非圧縮）で実行 |
| セッション非永続 | `USE SCHEMA` / `USE WAREHOUSE` / `USE ROLE` を実行しても次の呼び出しには残らない | **常に完全修飾名**。MCP 経由の全クエリはユーザー既定ウェアハウスで走るため、コンペ用ウェアハウスに課金を寄せるには (a) MCP 用の専用ユーザー（既定 WH = `PRC_WH`、既定ロール = `PRC_ROLE`）を作って接続を切り替える、または (b) 既定 WH を一時的に `PRC_WH` にする。`RESULT_SCAN(LAST_QUERY_ID())` も使えないので、メタデータは `SNOWFLAKE.ACCOUNT_USAGE`（最大 45 分〜3 時間遅延）や `<DB>.INFORMATION_SCHEMA` を SELECT で絞る |
| タイムアウト 60 秒 | `SHOW GRANTS TO ROLE SYSADMIN` がタイムアウト | **60 秒はクライアント側の打ち切り**。サーバー側の文は最大 48 時間走り続ける（COPY / CTAS は完走または部分完了し得る）。返答がなくても `INFORMATION_SCHEMA.QUERY_HISTORY_BY_USER()` で完了を確認してから再実行する。長時間処理は `TASK` に包んで `EXECUTE TASK` → `TASK_HISTORY` をポーリング。COPY は `FORCE=FALSE`（既定）でロード履歴による重複防止に依存 |
| 結果形式 JSON | 大量行の取得に不向き | 集計して返す。SHOW 系も同様 |

分担: MCP は「対話的 EDA・小さな集計・`EXECUTE IMMEDIATE FROM` によるスクリプト実行・TASK の起動と監視」に使う。SQL スクリプトはリポジトリで管理し、Snowsight で内部ステージにアップロード（将来は Git repository stage。`CREATE API INTEGRATION` は ACCOUNTADMIN 作業）して MCP から 1 文で実行する。60 秒超が見込まれる文（大規模 CTAS）は TASK 化する。

## 5. タイムスタンプとタイムゾーン（ロード前に必ず対処）

- アカウント・ユーザーの `TIMEZONE` が `America/Los_Angeles` のまま。Parquet の `*_UTC_*` 列はファイルフォーマット設定次第で `TIMESTAMP_LTZ` / `TIMESTAMP_NTZ` / `INTEGER` のいずれにもなり、LTZ→NTZ の暗黙変換や `HOUR()` / `DAYOFWEEK()` がロサンゼルス時刻で評価されて時刻特徴量が一律ずれる罠がある。Snowsight の Load Data ウィザードはこれらのオプションを露出しない。
- 対処:
  1. ロード前にサンドボックスで `pyarrow.parquet.read_schema()` を全 14 ファイルに実行し、timestamp の unit と `isAdjustedToUTC` を記録（docs/01 U13）。
  2. 名前付きファイルフォーマットを明示: `TYPE=PARQUET, USE_LOGICAL_TYPE=TRUE, USE_VECTORIZED_SCANNER=TRUE, BINARY_AS_TEXT=FALSE`。
  3. RAW/STG の時刻列は **`TIMESTAMP_NTZ`（UTC）に固定**し、COPY の変換句で `CONVERT_TIMEZONE('UTC', col)::TIMESTAMP_NTZ` とする。
  4. `ALTER USER <ADMIN_USER> SET TIMEZONE = 'UTC'`（ユーザーレベルなら MCP の非永続セッションでも効く）。専用ユーザーを作る場合はそのユーザーにも設定。
  5. ロード後アサーション: 月別 `MIN/MAX(MVT_TIME_UTC)` を pyarrow 側の値と一致確認。
  6. 空港ローカル時刻の特徴量はセッション TZ ではなく ICAO→IANA タイムゾーンの参照表で算出する。

## 6. データ搬入経路

公式データは OSN の MinIO（S3 互換、`https://s3.opensky-network.org/`）。Snowflake の S3 互換外部ステージ（`s3compat://`）は **Snowflake Support への申請で有効化が必要**（Cloudflare R2 以外は既定で無効）。加えて仮想ホスト形式リクエスト・ポート番号なし・キーのステージ直埋めが要件で、OSN MinIO の仮想ホスト対応は未確認。コンペ期間内は採用しない。

| 案 | 流れ | 長所 | 短所 |
|---|---|---|---|
| **A′. ローカル取得 + Snowsight ステージアップロード + リポジトリ SQL** | `mc` / `pyopensky` でローカル取得 → Snowsight「Load files into a Stage」で `PRC_DC2026.RAW.OFFICIAL_STAGE` にアップロード（250 MB/ファイル、250 ファイル/回まで。対象は 14 ファイル、19〜27 MB）→ リポジトリの `sql/010_raw_load.sql`（`CREATE TABLE ... USING TEMPLATE (INFER_SCHEMA)` + `COPY INTO ... MATCH_BY_COLUMN_NAME`）を実行 | 認証情報の受け渡し不要。テーブル作成と COPY はコードで再現可能。B/C と COPY スクリプトを共用 | アップロードのみ手作業 |
| B. ローカル取得 + Python クライアント | 取得後、`snowflake-connector-python` で `PUT` → 同じ COPY スクリプト | 完全自動化 | **ユーザー PC でのみ実行**。サンドボックスに Snowflake 認証情報は置かない（§8） |
| C. Snowflake 内から直接取得 | コンペ DB 内に network rule（`s3.opensky-network.org:443`）+ External Access Integration + Secret（MinIO キー）を作り、Python ストアドプロシージャで `minio` SDK → `session.file.put_stream` でステージへ → 同じ COPY スクリプト | Snowflake 内で完結。再取得・最終評価ファイル配布時の即応に有利 | Secret 作成は **ユーザーが Snowsight で直接実行**（MCP・チャット経由禁止、§8）。60 秒制約のため 1 ファイル 1 呼び出しまたは TASK 化 |

推奨: **初回は A′ で最速ロード**し EDA・ベースラインを先に進める。その後 C を整備し、隠し最終評価ファイルが配布された際に Snowflake 内で取得→ロード→特徴量→提出まで通せる状態にする。

ロード前チェックリスト:
1. 全ファイルの schema を pyarrow で dump し差分確認（unit、tz、nullability、dictionary encoding）。
2. ファイル別行数を記録し、RAW ロード後に `METADATA$FILENAME` 別 `COUNT(*)` と照合（合計 4,167,797、Ranking の DEP 344,841 行）。
3. Ranking の空欄表現（NULL / NaN / 空文字）を確認して `NULL_IF` を決める。
4. COPY は `ON_ERROR=ABORT_STATEMENT, PURGE=FALSE, FORCE=FALSE` で実行し `COPY_HISTORY` を保存。
5. `MVT_ID_mvt` の一意性を STG のアサーションに入れる。

## 7. 本サンドボックス（Claude Code 実行環境）

| 項目 | 値 |
|---|---|
| CPU / メモリ / ディスク | 4 core / 15 GB / 約 30 GB 空き |
| Python | 3.11、`uv` あり |
| インストール確認済み | polars, pyarrow, lightgbm, duckdb, snowflake-connector-python, pyopensky |
| Snowflake 認証情報 | **なし**。Snowflake へのアクセスは MCP（SYSADMIN 権限の認証済みチャネル）のみ |
| 到達性 | `s3.opensky-network.org`、`archive-api.open-meteo.com`、Iowa State ASOS/METAR アーカイブ、GitHub |

→ DEP 約 200 万行の LightGBM 学習は本サンドボックスで実行可能。Snowflake への書き戻し（予測結果・提出ファイル）は MCP 経由の INSERT/COPY か、ユーザー PC 側の B 経路で行う。

## 8. 認証情報・セキュリティの方針

- **鍵・パスワード・アクセスキーをチャット／MCP 経由で流さない。** Secret 作成（`CREATE SECRET ... TYPE = PASSWORD USERNAME='<access_key>' PASSWORD='<secret_key>'`）はユーザーが Snowsight で直接実行し、リポジトリの DDL はプレースホルダと「手動実行」コメントのみ。
- 自動化用は専用サービスユーザー（`TYPE=SERVICE`、パスワード無し、鍵ペア認証）+ 最小権限ロール `PRC_ROLE`（`PRC_DC2026` と `PRC_WH` の所有・USAGE のみ。既存 DB へのアクセス無し）。秘密鍵は `~/.snowflake/`（chmod 600）と `.gitignore` 済みの `.env` で参照。公開リポジトリには接続名と環境変数名だけを `.env.example` に記載。
- MCP 接続も可能ならこの専用ユーザー／ロールに切り替え、MCP からの DDL を `PRC_DC2026` 配下に限定する。
- `.gitignore` に `.env*`, `*.p8`, `*.pem`, `*.key`, `connections.toml`, `.snowflake/`, `settings.conf`（pyopensky）, `.mc/`, `*.parquet`（公式データはコンペ外利用禁止のため再配布不可）を最初から含める。公開時に GitHub Secret scanning / Push protection を有効化。
- 非公開 TODO（ACCOUNTADMIN、Snowsight で実施）: アカウントのネットワークポリシー、認証ポリシー（人間ユーザーに MFA 必須）、作業ユーザーの `DEFAULT_ROLE` を `SYSADMIN` 以下へ変更、`ALTER USER ... SET TIMEZONE='UTC'`。

## 9. コンペ用オブジェクト設計（未作成）

| 種別 | 名前 | 設計 |
|---|---|---|
| Database | `PRC_DC2026` | コンペ専用。既存 DB を参照しない |
| Schema | `RAW`（permanent） | 公式 parquet をそのまま格納。**単一テーブル** `RAW.TRAINING_MOVEMENTS`（原列 + `_SRC_FILE = METADATA$FILENAME`, `_SRC_ROW = METADATA$FILE_ROW_NUMBER`, `_LOADED_AT`）、`RAW.RANKING`、`RAW.SUBMITTING`。月別 12 テーブルにはしない |
| Schema | `REF`（permanent） | 外部データの生データ（気象、空港メタデータ、タイムゾーン表、祝日） |
| Schema | `STG`（transient） | 型統一（時刻は `TIMESTAMP_NTZ` UTC）、列名を接尾辞なし大文字 snake_case に統一、DEP/ARR 分割、不整合フラグ、`MVT_ID` 一意性アサーション |
| Schema | `FEAT`（transient） | 特徴量テーブル（混雑・履歴統計・カレンダー・気象・区間制約） |
| Schema | `ML`（transient） | 学習用ワイドテーブル、CV fold、予測結果、`ML.SUBMISSIONS`（version, model_id, created_at, rows を実体保存） |
| Schema | `SUB`（transient） | 提出ファイル生成ビュー。`SELECT ... AS "MVT_ID_mvt", ... AS "TAXITIME_SEC_mvt"` と引用符付き別名で公式列名を復元し、型はテンプレートに合わせてキャスト。最終ファイルは pyarrow で `submitting.parquet` の schema と `equals()` 検証してから提出 |
| Warehouse | `PRC_WH` | X-Small **Gen1**（1.0 credits/h。Gen2 は 1.35）。`AUTO_SUSPEND=60, INITIALLY_SUSPENDED=TRUE, ENABLE_QUERY_ACCELERATION=FALSE, STATEMENT_TIMEOUT_IN_SECONDS=3600, STATEMENT_QUEUED_TIMEOUT_IN_SECONDS=600`。重い処理時のみ `ALTER WAREHOUSE` で Small（2.0 credits/h）に一時変更 |
| Resource Monitor | `PRC_RM` | **ACCOUNTADMIN が Snowsight で作成**。月次上限 50 credits（XS Gen1 で約 50 時間相当）。NOTIFY 50%/80%、SUSPEND 100%、通知先メールを有効化 |
| Budget | アカウント Budget | コンピュートプール・サーバレス課金はリソースモニター対象外のため、Budget（ACCOUNTADMIN）で上限通知を設定。Container Runtime を使う日は終了時に `SHOW COMPUTE POOLS` で ACTIVE ノード 0 を確認 |
| Role / User | `PRC_ROLE`, `PRC_SVC` | §8 |
| Stage | `PRC_DC2026.RAW.OFFICIAL_STAGE` | 内部ステージ、`ENCRYPTION=(TYPE='SNOWFLAKE_SSE')`, `DIRECTORY=(ENABLE=TRUE)` |
| File Format | `PRC_DC2026.RAW.PARQUET_FF` | §5 の設定 |

特徴量生成の設計指針（4.2M 行・約 300 MB は Snowflake では小規模。倉庫サイズより SQL の書き方が支配的）:
1. `CLUSTER BY` は付けない（自動再クラスタリングはサーバレス課金）。STG/FEAT の CTAS に `ORDER BY airport, mvt_time_utc` を付けて自然クラスタリングを得る。
2. 時間窓集計は 5〜15 分バケット化した自己結合（bin ±k）または `ASOF JOIN`、`RANGE BETWEEN INTERVAL ... PRECEDING` 枠を使う。無制限の自己結合（大空港で年 20 万行 → 4×10^10 ペア）は書かない。RANGE INTERVAL 枠は 1 空港 1 日のサンプルで結果を検証してから全体に適用。
3. 中間結果は TRANSIENT テーブルに落として段階実行し、60 秒超になる文は TASK 化。

## 10. 未確認事項

| # | 事項 | 確認方法 |
|---|---|---|
| S1 | 公式 parquet の timestamp unit / tz、Ranking の空欄表現 | データ取得後に pyarrow で確認（docs/01 U13 と同じ） |
| S2 | Notebook Container Runtime の利用可否と課金（使う場合のみ） | ACCOUNTADMIN で確認。使わない場合は不要 |
| S3 | OSN MinIO の仮想ホスト形式対応（s3compat 再検討の余地を閉じるため） | 必要になった時のみ |
| S4 | Budget の作成可否と通知設定 | ACCOUNTADMIN で Snowsight から設定 |
| S5 | Git repository stage の利用（`CREATE API INTEGRATION` は ACCOUNTADMIN） | Step 5 以降で判断 |
