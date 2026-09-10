# 05. ランブック（セッション開始時と Phase 1 の実行手順）

サンドボックスは使い捨てなので、新しいセッションは必ずここから始める。リポジトリに入っていない状態（`.venv`、`data/`、モデル）は毎回作り直す。

## 0. セッション開始（毎回、5 分以内）

```bash
cd /home/user/prc-data-challenge
make bootstrap      # uv venv + 依存インストール + 環境変数確認 + データ取得（キーがあれば）+ pytest
make doctor         # 認証情報の疎通確認とバケット名・チーム名の自動検出（読み取りのみ）
make board          # リーダーボードを取得し、前回スナップショットとの差分を表示
```

`make bootstrap` は冪等。`OSN_ACCESS_KEY` が未設定なら取得だけ飛ばす。

## 1. 認証情報の受け渡し（クラウド環境の仕様）

MinIO のキーは **環境変数としてクラウド環境に登録する**。公式ドキュメントの仕様上、

- 設定場所: claude.ai/code の環境設定（Environment）の環境変数欄。`.env` 形式で 1 行 1 件
- **セッションは起動時に一度だけ値をコピーする**。実行中のセッションは後から追加した変数を読み直さない → **登録後は新しいセッションを開始する必要がある**
- 環境を使う人は誰でも値を読める。したがってコンペ用の最小権限キーのみを置く

登録する変数（`.env.example` と同じ名前）:

```text
OSN_ACCESS_KEY=<MinIO のアクセスキー>
OSN_SECRET_KEY=<MinIO のシークレット>
```

**必要なのは MinIO（S3）のアクセスキーであって、OpenSky の API クライアントではない。** OSN アカウントは紛らわしい名前の認証情報を 2 種類発行する。API クライアント（`<user>-api-client` 形式の client_id + secret）は REST/Trino API 用で、S3 エンドポイントは `InvalidAccessKeyId` を返す。取得手順は `https://s3.opensky-network.org/` に OSN アカウントでサインイン → Access Keys → Create access key。2026-09-10 のセッションはこれを取り違えて Phase 1 が止まった（docs/decisions.md）。`make doctor` がこの取り違えを名指しで診断する。

キーの名前は複数の表記を受け付ける。アクセスキーは `OSN_ACCESS_KEY` / `OSN_ID` / `OSN_ACCESS_KEY_ID` / `OSN_KEY_ID` のいずれか、シークレットは `OSN_SECRET_KEY` / `OSN_SECRET` / `OSN_SECRET_ACCESS_KEY` のいずれかで良い（運営の表記が MinIO コンソールと OSN アカウント画面で異なるため）。

`OSN_TEAM_BUCKET`（提出先バケット）と `PRC_TEAM_NAME`（提出ファイル名に使うチーム名）は **`make doctor` が接続後に検出して提案する**ので、最初から設定しなくてよい。提出を行う前に環境変数へ追加する。`OSN_S3_ENDPOINT` は既定値 `s3.opensky-network.org` で足りる。

`OSN_COMPETITION_BUCKET` の既定値 `competition-data` は 2024 年版の手順から引いた推定で、**その名前のバケットの中身は実際には 2024 年版（ATOW）のデータセットだった**（匿名で一覧可能、日別 parquet + `challenge_set.csv`、最終更新 2024-10）。2026 年の 14 ファイルがどのバケットにあるかは鍵が通ってからでないと分からない。`make doctor` が可視バケットの一覧と 2024 年版の判定を出すので、正しい名前をそれで確認して環境変数に入れる。

サンドボックスの外向き通信は `HTTPS_PROXY` のエージェントプロキシ経由のみ。`prc.io_minio` はこれを読んで接続する（`requests` を使う `prc.leaderboard` / `prc.weather` は元から自動）。`too many 503 error responses` が出たらプロキシを迂回している合図。

ネットワークは確認済み: このクラウド環境から `s3.opensky-network.org` と `mesonet.agron.iastate.edu` に到達できる。

## 2. Phase 1: データ到着後の手順（目標 48 時間で v2 提出）

`make doctor` が緑になっていること（＝バケットが見えていること）が Phase 1 の入口条件。赤のまま先に進んでも `make download` は失敗する。

```bash
# 2-1. 取得（サイズ一致ならスキップ。14 ファイル、約 300 MB）
make download

# 2-2. プロファイル: docs/01 の未確認事項（U1, U5-U8, U11, U13）を一括で答える
.venv/bin/python -m prc.profile          # docs/data_profile.md を生成
```

プロファイルで最初に見るのは次の 3 点。

1. Ranking の出発便で `MVT_TIME_UTC_mvt` が非欠損か、`AOBT_3_flt` の充足率（公開報告では 98.5%）
2. `AOBT_3_flt` の情報量: `BLOCK_TIME − AOBT_3` の空港別分布、`AOBT_3 = MVT_TIME − 定数` の派生値シグネチャ
3. 報告エラー体制の発生率（`BLOCK == SCHED`、1 日ずれ）を空港別に

この結果で `docs/04` §4 の設計を確定する。1 が偽なら予定時刻ベースの経路に切り替える。

```bash
# 2-3. v1: 空港 × 滑走路 × 時間帯の中央値（提出経路と採点遅延の実測が目的）
# 2-4. v2: 2 つの時計の差分 LightGBM
.venv/bin/python -m prc.pipeline --config configs/v2_two_clock.yaml
```

提出は必ず次の順で行う。

1. 提出ゲート（行数・ID 一致・NaN・dtype・負値）を通す
2. `prc.submit.submit(...)` でアップロード（1 日 5 回カウンタと版番号重複を自動チェック。UTC と Brussels の両方の日境界で安全側に数える）
3. 数分後に `make board` で自分の提出が SUCCEEDED として現れることを確認（失敗提出は API に出ない）
4. `experiments/submissions.jsonl` に CV 値・変更点・モデル外スカラーの根拠が残っていることを確認

## 3. Snowflake（Track B、並走）

管理者作業（`sql/000_admin_manual.sql` の 1〜3）が済んでいることを毎セッション確認する。

```sql
-- MCP から 1 文ずつ。UTC と PRC_WH になっているか
SHOW PARAMETERS LIKE 'TIMEZONE' IN SESSION;
SELECT CURRENT_WAREHOUSE();
SHOW WAREHOUSES LIKE 'PRC_WH';   -- resource_monitor が PRC_RM になっているか
```

ロード経路は 2 つ。

- 案 C（自動）: ユーザーが Snowsight で Secret を作成 → `sql/015_osn_pull.sql` を実行 → `CALL PRC_DC2026.RAW.FETCH_OSN_OBJECT('competition-data', '<file>')` を 1 ファイル 1 コールで（12 ファイルは TASK に包む）→ `sql/020_raw_load.sql`
- 案 A′（手動）: ユーザーが Snowsight で 14 ファイルを `@PRC_DC2026.RAW.OFFICIAL_STAGE` にアップロード → `sql/020_raw_load.sql`

Snowflake からサンドボックスへの取り出しは `sql/090_export.sql`（`COPY INTO @stage` → `GET_PRESIGNED_URL` → curl）。**サンドボックスから Snowflake への接続はできない**（プロキシが証明書ピン留めクライアントを非対応）。

## 4. 毎日の定例

| 頻度 | 作業 | 担当 |
|---|---|---|
| 毎日 | `make board`（首位との差、自チームの順位） | Claude |
| 毎日 | `ranking.html` の差分確認（運営告知はここにも出る） | Claude |
| 毎日 | OSN Discord `#prc-data-competition` の確認、未確認事項（U3, U9, U12）の質問 | ユーザー |
| 提出ごと | 提出ログの記録、CV と LB の対応の更新 | Claude |

## 5. 締切（時刻は UTC と JST を併記）

| 予定 | UTC | JST |
|---|---|---|
| 公式締切 | 2026-10-11 21:59:59Z（「23:59:59 CET」の CEST 解釈） | 10-12 06:59 |
| 内部締切（最終提出完了） | 2026-10-11 12:00Z | 10-11 21:00 |
| モデル凍結 | 2026-10-09 | |

締切当日は 5 枠のうち 2 枠を予備に残す。最終候補は前日までに提出済みにする。
