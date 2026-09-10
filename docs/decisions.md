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

## 2026-09-10（続き）: コードの独立レビューと修正

3 視点（リーク/parity、正しさ/契約、競争力）+ 敵対的検証の 11 エージェントで実施。全て再現スクリプト付きの指摘だったため、以下を修正した（詳細はコミット `2c2efcf` と本日の後続コミット）。

**修正したもの（重大）**
1. `apt_med`（空港別ラベル中央値）を「渡されたフレーム自身のラベル」から計算していた。検証フォールドでは正解が特徴量に入り、Ranking では全行欠損になる二重の欠陥。ラベル由来の表は全て学習側で fit して結合のみに変更（`fit_encodings` / `attach_encodings`）。
2. 早期打ち切りが採点対象の検証フォールドを見ていた。加えて Ranking 経路はラベルが無く打ち切りが働かず 2000 本固定だった（「検証したモデル」と「提出するモデル」が別物）。学習側の内部ホールドアウト（直近 1 か月）で本数を決め、学習側全体で固定本数に再学習する方式に変更。合成データでの実測ラウンド数は 47〜234。
3. 予測を行の位置で運ぶ箇所が 3 つ（`predict_fold` の戻り値、`run_cv` の OOF 代入、`write_submission` の ID と値）。polars の結合は順序を保証しないため、ID と予測が入れ替わった提出が既存ゲートを素通りする経路が実在した。すべて `MVT_ID_mvt` の結合に変更し、`maintain_order="left"` を明示、ゲートに ID 順序の検証を追加。
4. 提出テンプレートが整数型または null 型だと `cast` が例外になり初回提出が不可能だった（U13 未確認）。整数型は丸め、null 型は float64 に昇格し、いずれも警告として記録。3 パターンの回帰テストを追加。
5. null ラベルが LightGBM に NaN として渡り、0 として学習されていた（実測 4% の NaN で RMSE 0.26 → 54.7）。除外して件数を記録。
6. 提出回数カウンタの CET 日境界が実装されておらず、同一 CET 日に 10 回提出できた。`ZoneInfo("Europe/Brussels")` で両方の日境界を数えて安全側を採用。dry-run は枠も版番号も消費しない方式に変更（旧テストは誤った挙動を固定していたので更新）。アップロードは write-ahead でログに残す。
7. 滑走路が欠損すると特徴量生成が `IndexError` で停止し、空港が欠損すると行が無言で消えていた。両方修正し行数・ID 一意性のアサーションを追加。
8. 検証指標がラベル 1 行の非有限値で `rmse_full` だけ NaN になり、他の指標は健全に見える状態だった。有限マスクと `n_dropped` を追加。対差分ブートストラップも同様に保護。
9. 離陸時刻が欠損した行の窓カウントが 0（＝「空いている」と誤読）になっていた。欠損に変更し、`anchor_ok` と `window_coverage_60m` を追加。
10. 月ブロック fold のキーが年を無視していた（複数年を混ぜると検証期間より後で学習し得る）。`year*100+month` に変更し、1 か月しか無い入力は明示的にエラー。
11. NM レコードはあるが `AOBT_3` が無い行が、使える `LOBT` / `EOBT_1` / `IOBT` を接頭辞除外で全部捨てていた。除外を `AOBT_3` 由来のみに限定。

**方針として採用したもの**
- `min_data_in_leaf` を 100 → 50。尾部の破滅的行（2025 年通年で 40〜50 行規模）を葉として分離できるようにする。さらなるアブレーションは実データで。
- 参照値（スタンド × 滑走路の P10）の自行包含は残す代わり、最小件数を 10 → 30 に引き上げた。完全な nested OOF は公開報告で効果が約 −1 秒（悪化）とされており、費用対効果が低い。この逸脱は意図的なものとして記録する。
- `linear_tree` と広域入力の signed-log コピーを実装。合成データでの対差分は −1.84 秒（95% CI −5.11〜+0.93）で有意ではないが、実データで再評価する。

**Phase 2 に持ち越したもの（実データが必要）**
- 報告エラー体制のソフトな体制ヘッド（校正済み確率 × 体制別予測の混合）。公開報告では決定論的な帯域ルールで LB −58 秒。校正には実データの体制発生率が必要。
- 待ち行列特徴量の D2/A2 近似の精緻化と、`AOBT_3` を近傍便のプッシュバック近似として使う設計。

## 2026-09-10（続き 2）: Phase 1 の着手と、データ取得のブロッカー

**サンドボックスのプロキシ対応（修正済み）**: MinIO クライアントはこのクラウド環境から**直接**外に出ようとして `too many 503 error responses` で失敗していた。サンドボックスの外向き通信は `HTTPS_PROXY` のエージェントプロキシ経由のみで、その TLS 傍受は `SSL_CERT_FILE` の CA バンドルで信頼する。`prc.io_minio.http_client()` を追加し、環境にプロキシがあれば `urllib3.ProxyManager`、無ければ通常の `PoolManager` を使う（ローカル実行でも同じコードで動く）。`requests` を使う `prc.leaderboard` / `prc.weather` は環境変数を自動で読むため元から動いていた。

**データ取得は未達（実 S3 鍵待ち）**: 環境変数に登録されている `OSN_ID` / `OSN_SECRET_KEY` は **OpenSky の API クライアント（OAuth2 の client_id + client_secret）**であって、MinIO の S3 アクセスキーではない。切り分けの実測:

- S3 エンドポイントは `InvalidAccessKeyId` を返す
- 同じ値を OSN の OAuth2 トークン endpoint に投げると `200` でアクセストークンが返る（＝値自体は有効な API クライアント）
- MinIO の STS `AssumeRoleWithWebIdentity` にそのトークンを渡すと `azp claim invalid, azp must match configured OpenID Client ID`。MinIO は自身のコンソール用 OIDC クライアントしか受け付けないため、API クライアントを一時的な S3 鍵に交換する迂回路も無い

したがって**ユーザー作業が必要**: `https://s3.opensky-network.org/` に OSN アカウントでサインイン → Access Keys → Create access key → その 2 値を環境変数 `OSN_ACCESS_KEY` / `OSN_SECRET_KEY` に登録 → **新しいセッションを開始**（変数はセッション開始時に一度だけコピーされる）。この診断を `prc.doctor` に実装したので、次回以降は `make doctor` が同じ結論を自分で出す。

**バケット名の注意**: `competition-data` は匿名でも一覧でき、中身は 368 オブジェクト・306 GB だが、**2024 年版（ATOW）のデータセット**だった（日別 parquet `2022-01-01.parquet` … と `challenge_set.csv` / `submission_set.csv` / `final_submission_set.csv`、最終更新 2024-10）。2026 年の 14 ファイルは別バケットにある。`.env.example` の既定値 `OSN_COMPETITION_BUCKET=competition-data` は 2024 年版の手順から引いた推定値で、**誤り**の可能性が高い。正しいバケット名は鍵が通れば `make doctor` が一覧から検出する。`prc.doctor` に 2024 年版データセットの判定を追加した。

**ブロックされていない範囲は完了**: `make bootstrap`（依存導入・35 テスト）、`make board`（首位 245.09 秒 youthful-giraffe、96 チーム。前スナップショットから 7 位が −0.04 のみ）、合成データでの v1 / v2 パイプライン全経路（特徴量 → 3 fold CV → ランキング予測 → 提出ゲート）。v1 CV full=567.91 / clean=139.32、v2 CV full=567.78 / clean=138.81。合成データは 2 つの時計の構造を模していないので、この差に意味は無い（実データでの再評価が前提）。

**次の一手**（鍵が入り次第、docs/05 §2 の順）: `make download` → `make profile` で U1・U5〜U8・U11・U13 に回答 → docs/04 §4 の設計を確定 → v1 提出で採点遅延を実測 → v2。

## 2026-09-10（続き 3）: 鍵の発行画面を一次情報で特定（前項の URL 訂正）

前項で案内した「`https://s3.opensky-network.org/` にサインイン」は**誤り**。あのホストは S3 API 専用でブラウザからは 403 を返す（Web UI が無い）。正しい発行画面は**別ホストの MinIO コンソール**:

- **https://s3-console.opensky-network.org/access-keys** → Login with SSO → OSN アカウント → 左メニュー Access Keys → 右上 Create access key
- 根拠: 2026 年公式データページの Buckets 節（`https://prc-data-challenge-2026.netlify.app/data.html#buckets`）に貼られたスクリーンショット `media/minio_access_keys.png` のアドレスバーがこの URL。運営自身が「OSN アカウントでアクセスキーを生成する必要がある（Discord の JavierIOM 提供）」と添えている
- コンソールの `/api/v1/login` は `loginStrategy: redirect` を返し、OSN の Keycloak（`client_id=minio-client`）へ飛ばす。**この `minio-client` が MinIO の STS が受け付ける唯一の OIDC クライアント**であることも、前項の STS 迂回が `azp` 不一致で失敗した理由として確定した

**2026 年版のバケット名は公式サイトに書かれていない**（提出先も「チーム専用のバケット」とだけ）。匿名での特定も不可能: エンドポイントは存在しないバケットにも既存の非公開バケットにも一律 `403 AccessDenied` を返す（`this-bucket-surely-does-not-exist-xyz` と `submissions` が同じ応答であることを確認）。鍵が通れば一覧に出るので `make doctor` で確認する。

**公式サイトの導線の罠**（2 つのサイトが並存している）:

| | 2024 年版（ATOW） | 2026 年版（taxi-out time） |
|---|---|---|
| サイト | `prc-data-challenge.netlify.app` | `prc-data-challenge-2026.netlify.app` |
| データ | 2022 年の飛行リスト + 日別軌跡（158 GiB） | 2025 年の 11 空港 movements（月次 14 ファイル） |
| バケット | `competition-data`（匿名で一覧可）+ `submissions` | 未公表 |

2026 年版の Buckets 節が「MinIO の使い方は 2024 年版を見よ」と `ansperformance.eu/study/data-challenge/dc2024/data.html#using-minio-client` を案内しており、これが 2024 年版サイトへリダイレクトする。**リンク先に書かれたバケット名（`competition-data`）は 2024 年版のもので、2026 年版のものではない**。`.env.example` の既定値がこの罠を踏んでいた。

2024 年版ページから引き継いで有効な手順は、`mc` の使い方そのもの（`mc alias set <alias> https://s3.opensky-network.org/ <ACCESS_KEY> <SECRET_KEY>` → `mc ls <alias>` でバケット一覧 → `mc cp --recursive <alias>/<bucket>/ <dir>/`）と、`pyopensky` の `S3Client`（設定ファイルの `[default]` に `access_key` / `secret_key`）。エンドポイント自体は 2026 年版でも同じ。

## 2026-09-10（続き 4）: 実データ取得とプロファイル — 未確認事項の一括解消

鍵が通り、`make download` で 14 ファイル 315 MB を取得。バケットは **`prc-2026-datasets`**（データ）と **`prc-2026-resilient-hedgehog`**（提出）。チーム名は公式 teams API で **`resilient-hedgehog`** と確認（Japan / MOCHITEC Inc. / APPROVED / 登録 2026-09-10）。提出ファイル名は `resilient-hedgehog_v<N>.parquet`。バケット名は `prc-<年>-<チーム名>` の規則で、doctor のチーム名推定もこの規則に合わせた。

`make profile` の全結果は docs/data_profile.md。決定に効くものだけ:

**U1 = 真（設計の前提が成立）**: Ranking の出発便で `MVT_TIME_UTC_mvt` の欠損率 **0.0000**。空欄化されているのは `BLOCK_TIME_UTC_mvt` と `TAXITIME_SEC_mvt` のみで、**`_flt` 列は全て残っている**（`AOBT_3_flt` / `LOBT_flt` / `EOBT_1_flt` / `IOBT_flt` / `ARVT_3_flt` の欠損率はいずれも 0.0153 = NM 未マッチ率そのもの）。docs/04 §4 の二つの時計の設計をそのまま採用する。予定時刻ベースへの切り替えは不要。

**`AOBT_3` は派生値ではない**: `MVT − AOBT_3` の空港別 sd が 171.5〜430.5 秒とばらつき、`|BLOCK − AOBT_3| ≤ 60s` の割合は 0.087〜0.264。定数オフセットの派生値なら sd はほぼ 0 になるはずで、そうなっていない。素朴予測器 `MVT − AOBT_3` の in-sample RMSE は **384.9 秒**で、docs/03 §8 の公開プローブ報告（384.9）と一致。**強い特徴量だが正解ではない**という前提が実測で裏付けられた。

**目的変数の定義が厳密に一致**: `|(MVT − BLOCK) − TAXITIME| ≤ 1s` の割合 = **1.000000**。目的変数は文字通り 2 つの時刻の差。

**尾部の主因は LIRF**: sd は他空港 300〜430 秒に対し LIRF のみ **1,332 秒**、`BLOCK == SCHED`（報告エラー体制）の発生率も **4.5%** で突出（他は 0.18〜1.4%）。`> 4h` も LIRF だけ 0.0007。**1 日ずれ体制は全空港で 0.0000** だったので、この仮説は捨てる。体制ヘッドは `BLOCK == SCHED` に絞る。

**その他の確定**: U5 = 空港は **10**（11 ではない）。U6 = 目的変数の欠損 0、負値 0.02%、`> 2h` 0.03%、最大 131,167 秒（36 時間）。U7 = training 行数 **4,167,797**（公式記載と ±0）、`MVT_ID_mvt` は一意、NM 未マッチ率 0.44〜1.70%。U11 = Ranking の `RUNWAY_mvt` / `STAND_mvt` / `SCHED_TIME_UTC_mvt` は**欠損ゼロ**（欠損行用の別経路は不要）。U13 = 全 timestamp が `timestamp[us, tz=UTC]`（Snowflake ロードの型事故の心配なし）。U1b = 域内便（到着行が Ranking 内にある出発便）は **14.77%**。

**U8 = 匿名化されていない**: `FLIGHT_mvt` = `LH982`、`CALLSIGN_flt` = `DLH8UL` と実値。ハッシュ化されているのは `AIRCRAFT_OPERATOR_flt` のみ（公式記載どおり）。ADS-B 履歴との結合で目的変数を直接復元する経路が技術的に開いていることになるが、docs/03 §8 の方針どおり**採らない**。便名は統計特徴量としてのみ使う。

**カテゴリ被覆率**: `STAND_mvt` は Ranking 行の 99.92%、`RUNWAY_mvt` 100%、`AIRCRAFT_TYPE_mvt` 99.88%、`ADES_mvt` 99.95% が学習側に存在。一方 **`FLIGHT_mvt` は 78.89%** しかないので、便名単位の統計は必ず階層フォールバックを持たせる。

**素朴予測器のベンチマーク（2025 in-sample）**: global mean 546.4 / airport mean 514.8 / airport×runway 504.5 / airport×runway×stand 469.1 / `MVT − AOBT_3` **384.9**。現在の LB 首位は 245.09 秒。

## 2026-09-10（続き 5）: 実データ初回実行で出た 3 つの欠陥と v1 の CV

**(1) 特徴量キャッシュの汚染（重大・実害寸前）**: `build_features` のキャッシュがパスだけをキーにしていた。合成データの実行と実データの実行が既定の作業ディレクトリ `data/work` を共有するため、**実データの初回実行が合成データの特徴量をそのまま読み込み、合成データで学習した**。`n=36,598`（合成データの行数）という CV 値をもっともらしく印字していた。気付けたのは提出直前の dtype 不一致で例外が出たからで、**偶然**である。ID の型が偶然一致していれば、合成データ由来の予測を提出していた。

対策: キャッシュを入力の指紋（raw ディレクトリの絶対パス、各 parquet の名前とサイズ、学習・ランキングの行数、気象フラグ、`features.py` のソースハッシュ）で照合し、不一致なら再構築して理由を印字する。`features.py` を編集した場合も無効化される。回帰テストは `tests/test_feature_cache.py`。

**(2) 提出テンプレートの ID 型**: 公式ファイルの `MVT_ID_mvt` は **float64**（整数値を保持）、`TAXITIME_SEC_mvt` は **int32**。polars は f64 と i64 を結合できず `SchemaError` で停止する。`write_submission` に共通型への整列を追加した。値が非整数の場合と float64 の正確表現範囲（2^53）を超える場合は、黙ってキャストせず例外にする。

**(3) 合成データが公式スキーマと違っていた**: `prc.synth` は ID を int64、目的変数を float64、timestamp をタイムゾーン無しで生成していた。実測した公式スキーマ（ID float64 / 目的変数 int32 / 全 timestamp `timestamp[us, tz=UTC]`）に合わせた。**合成データが実データの型を再現していなかったこと自体が (2) を実データまで持ち越した原因**である。

**メモリ: OOM で 3 回落ちた（cgroup 上限約 14 GB）**。原因は `_fit` が早期打ち切り用の内側分割を別フレームとして連結し、**学習データを行列に二重に持っていた**こと、および行列が float64 だったこと。修正は 2 点。内側分割をブールマスクにして 1 つの行列を添字で参照する（フレームの二重化と、目的変数計算のためのフレーム 2 回フィルタが消える）。行列を **float32** にする（LightGBM は内部でビン化するため精度への影響は無い）。これでピーク RSS は **9.29 GB**、v1 は完走。

**v1（実データ、提出はまだしていない）**: CV `full=342.38` / `clean=253.27`（310 秒）。fold 別 jan 376.81 / jul 388.79 / forward 293.09。学習 2,085,047 行、ランキング 344,841 行、parity gap 0.017。素朴な `MVT − AOBT_3`（384.9）より良く、LB 首位 245.09 とはまだ差がある。ゲートの警告は 2 件で、いずれも既知: 予測最大 85,322 秒（尾部の実在する外れ値。目的変数の実測最大は 131,167 秒）、int32 テンプレートへの丸め（最大 0.5 秒）。

**Snowflake は未使用（コスト 0）**: ユーザーからの費用の懸念を受けて実測した。`PRC_DC2026` のテーブルは `ML.EXPERIMENTS` と `ML.SUBMISSIONS` の 2 つで**いずれも 0 行 0 バイト**、競技データは未ロード、`PRC_WH` は SUSPENDED、過去 30 日のメータリング履歴に **`PRC_WH` の行は 1 件も無い**（消費 0 クレジット）。アカウント全体の 8.6 クレジットは全て無関係のウェアハウス（`AGENT_TEST_WH` 等）。

**未解決（ユーザー判断待ち）**: `SHOW RESOURCE MONITORS` が **0 件**で、`PRC_WH` にリソースモニターが付いていない（D3 の `PRC_RM` が未作成。ACCOUNTADMIN 権限が必要）。上限が無い状態なので、Track B を動かす前に付けるか、Track B 自体を当面見送るかを決める。Phase 1 の学習・提出はサンドボックスで完結するため、どちらでも進行は止まらない。
