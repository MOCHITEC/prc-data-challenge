# 決定記録・実行ログ

形式: 日付 / 内容 / 根拠 / 影響。運営告知（Discord・ranking.html）の記録もここに残す。

## 2026-09-09

- **進め方**: 小さく作る → 振り返る → 独立レビュー → 修正の反復。レビュー指摘は「勝敗・正しさ・安全・実行可能性に効くもの」だけ採用し、表現・丸め・出典帰属などの細部には工数を使わない。
- **リポジトリ公開前提**: 賞金条件（GPLv3 公開）のため、アカウント識別子・社内オブジェクト名・認証情報はコミットしない。識別子を含んでいた初期コミットはブランチ履歴から除去済み（リポジトリは private、フォーク 0 の時点で実施）。
- **Snowflake 基盤を作成（SYSADMIN、MCP 経由）**: DB `PRC_DC2026`（スキーマ RAW / REF は permanent、STG / FEAT / ML / SUB は transient、PUBLIC は削除）、ウェアハウス `PRC_WH`（X-Small, Gen1, auto-suspend 60 s, initially suspended, QAS 無効, statement timeout 3600 s）、ファイルフォーマット `RAW.PARQUET_FF`、ステージ `RAW.OFFICIAL_STAGE` / `RAW.SQL_STAGE`、表 `ML.EXPERIMENTS` / `ML.SUBMISSIONS`。
  - `CREATE ROLE` は SYSADMIN 不可（USERADMIN/ACCOUNTADMIN 権限）→ `PRC_ROLE` は `sql/000_admin_manual.sql` に移し、当面は SYSADMIN 所有で進める。
  - ウェアハウス世代の指定は `RESOURCE_CONSTRAINT` ではなく `GENERATION = '1'`（実行時エラーから修正）。
- **ユーザー判断待ち（docs/04 §2）**: D1 チーム登録、D2 認証経路（推奨: MinIO キーを Claude Code 環境変数に登録 + Snowflake Secret は Snowsight で作成）、D3 ACCOUNTADMIN 作業（`sql/000_admin_manual.sql`: TIMEZONE=UTC、リソースモニター、Budget、PRC_ROLE）、D4 気象データ利用、D5 作業時間。
- **リーダーボード**: 首位 246.36 秒（youthful-giraffe）。提出済み 87 チーム、承認 336〜337 チーム。
