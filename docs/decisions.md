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

## 2026-09-10

- **レビュー反映の方針**: 4 視点レビュー（推進・反対・実行可能性・リーク/検証）と追加調査（空港運用・戦術）の指摘のうち、採用したのは (1) サンドボックス⇄Snowflake の転送経路の未定義、(2) 特徴量優先順位（2 つの時計の差分・尾部体制・気象を前倒し、トランスダクティブ反復と多シードを後退）、(3) 48 時間で初 GBDT 提出の 2 トラック化、(4) D3 を Phase 0 出口条件に昇格、(5) 検証の統計的扱い（対差分ブートストラップ、全行評価、nested OOF TE、カテゴリ被覆率）、(6) 2026 の新スタンドへの階層フォールバック、(7) 相対目標。不採用: 素朴予測器を LB に投げて較正する手法（プローブに当たる）、提出アップロードの人手限定（検証ゲート + 提出ログ + 日次カウンタで代替）。
- **サンドボックスからの Snowflake 接続は不可**（プロキシが証明書ピン留めクライアントを非対応と明記）。Snowflake 認証情報はサンドボックスに置かない。Snowflake → サンドボックスは `GET_PRESIGNED_URL` で一方向。学習・提出はサンドボックス完結（Track A）、Snowflake は SQL 特徴量基盤（Track B）。
- **`AOBT_3_flt` を使う**: 運営のデータページは通常の列として説明し、空欄化対象は `BLOCK_TIME_UTC_mvt` と `TAXITIME_SEC_mvt` のみと明記。参加チームの公開プローブでは Ranking の 98.5% に存在し、相関 0.536 の「強い特徴量だが正解ではない」。使用チームに制裁は無い。使用しない場合の下限は約 300 秒と見積もられており、優勝を目指す本プロジェクトでは使用する。運営が方針を変えた場合に備え、`_flt` 非依存の経路も維持する。ユーザーには本決定を明示して確認を求める。
- **尾部方針**: 学習から外れ値を除外しない。L2・生秒・全行。報告エラー体制はソフトな混合で扱う。
- **計画 v2**（docs/04）を確定。実装フェーズのタスクを細分化して登録。
- **Snowflake → サンドボックスの転送経路を実証**（Phase 1 の完了条件を先取り）: `PRC_DC2026.ML.EXPORT_STAGE`（SSE, directory 有効）を作成し、`COPY INTO @stage/... SINGLE=TRUE` で 50,000 行を parquet 化 → `GET_PRESIGNED_URL` → サンドボックスから curl で取得（HTTP 200、780 KB）→ pyarrow で読めることを確認。型の注意 2 点を `sql/090_export.sql` に記録: Snowflake NUMBER は parquet DECIMAL になるため SELECT 側で `::DOUBLE` / `::INT` にキャストする。TIMESTAMP_NTZ は isAdjustedToUTC 付きで書かれ、pyarrow ではタイムゾーン付きになる。
- **合成データでパイプライン全体を実行**: 生成（14.5 万行）→ 特徴量（出発 7.2 万行 × 60 列）→ 3 fold CV → ランキング予測 → 提出ゲート通過まで 23 秒。実データ（出発 208 万行）でも同じ経路で回せる見込み。
- **気象データを取得・検証**: IEM ASOS（パブリックドメイン）から 10 空港・2025-01〜2026-07 の METAR を取得（138,446 時間）。防除氷ゲートの 2026 年 1 月時間割合が他チームの公開報告とほぼ一致（LSZH 18.0%、EDDM 11.2%、EDDF 8.3% など）。docs/03 §8 の気象に関する主張と自前パイプラインの相互検証になった。台帳は docs/external_data.md。
