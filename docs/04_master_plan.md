# 04. 全体計画（マスタープラン）

作成日: 2026-09-09。入力: docs/01（コンペ事実）、docs/02（Snowflake 環境）、docs/03（先行研究）。独立レビュー未実施（Step 4R で実施し改訂する）。

---

## 0. ゴールと成功指標

| 種別 | 内容 |
|---|---|
| 主目的 | PRC Data Challenge 2026 で **優勝**（最終評価 RMSE 1 位 + 賞金条件充足） |
| 副目的 | Snowflake を「データ基盤 + 特徴量生成基盤」として一連の流れ（ロード → EDA → 前処理 → 特徴量 → 学習 → 提出 → 改善）で使い切り、習得する |
| 締切 | 2026-10-11 23:59:59 CET（内部締切 **21:00Z**） |
| 数値目標 | 09-09 時点の LB 首位 246.4 秒。上位は日々改善しているため、**10/11 時点で ≤ 225 秒**を最終目標、中間目標を §4 の各フェーズに置く |
| 品質目標 | 隠し最終評価（別月・別サブセット）でも順位が落ちない汎化。GPLv3 公開・再現手順・JOAS 構成の README |

## 1. 前提と制約

- 期間 32 日、実働 1 人（+ Claude Code）。ユーザーは Snowflake・データコンペ初心者。
- 提出は現時点で 1 日 5 回。全提出のうち最良値で順位付け。**隠し最終評価を既定シナリオ**とする（docs/01 U2）。
- 実行環境: (a) Snowflake（MCP 経由、1 文ずつ、セッション非永続）、(b) Claude Code サンドボックス（4 core / 15 GB、Snowflake 認証情報なし）、(c) ユーザー PC（Snowsight・mc・pyopensky）。
- 外部データはオープンライセンス + 文書化が必須。ADS-B からの目的変数復元・テスト期間の集計値利用は行わない（docs/01 §8）。
- リポジトリは公開前提。機密識別子・認証情報・公式データ（再配布不可）はコミットしない。

## 2. 決定が必要な事項（ユーザー判断）

| ID | 決定事項 | 推奨 | 期限 |
|---|---|---|---|
| D1 | **チーム登録**（OSN アカウント作成 → Google Form 送信 → 承認 → バケット情報取得）。チーム種別・国・所属の記入 | 直ちに（承認まで最大 1 日。データ取得の唯一のクリティカルパス） | 09-10 |
| D2 | **データ搬入とサンドボックスの認証経路**。(i) MinIO キーを Claude Code の環境設定（環境変数）に登録し、サンドボックスが直接 MinIO からデータ取得・提出アップロードを行う。Snowflake 側は Secret を Snowsight で作成し Snowflake が MinIO から直接取得（案 C）。(ii) ユーザー PC で取得・Snowsight でステージにアップロード（案 A′）。サンドボックスは Snowflake 経由の集計のみ | **(i)**。Claude が自律的にロード・提出・再取得まで回せる。キーはチャットに貼らず環境変数で渡す。(ii) はフォールバック | 09-10 |
| D3 | Snowflake 管理作業（ACCOUNTADMIN、Snowsight で実施）: リソースモニター `PRC_RM`（月 50 credits）、Budget、`ALTER USER ... SET TIMEZONE='UTC'`、可能なら MCP 用ユーザー/ロールの分離 | 09-10〜09-11 に実施。手順書は `sql/000_admin_manual.sql` として用意 | 09-11 |
| D4 | 外部データの範囲: 気象（METAR 経由の Iowa State ASOS アーカイブ / Open-Meteo 履歴）を使うか | 使う（ライセンス記録を `docs/external_data.md` に残す）。Phase 3 で効果検証し、効かなければ外す | 09-17 |
| D5 | 作業時間の目安（平日 / 週末に割ける時間） | 計画のバッファ調整に使う | 09-10 |

## 3. アーキテクチャ

### 3.1 Snowflake（docs/02 §9 の設計をそのまま採用）

```
PRC_DC2026
 ├─ RAW   (permanent)  TRAINING_MOVEMENTS / RANKING / SUBMITTING / OFFICIAL_STAGE / PARQUET_FF
 ├─ REF   (permanent)  WEATHER_HOURLY / AIRPORTS / TIMEZONES / HOLIDAYS
 ├─ STG   (transient)  MOVEMENTS（型・列名統一、DEP/ARR、不整合フラグ、UTC 固定）
 ├─ FEAT  (transient)  F_STATIC / F_SCHEDULE / F_RUNWAY_QUEUE / F_ARR_CONTEXT / F_STAND_REF / F_WEATHER / F_NM_TIMES
 ├─ ML    (transient)  TRAIN_WIDE / FOLDS / PREDICTIONS / SUBMISSIONS / EXPERIMENTS
 └─ SUB   (transient)  V_SUBMISSION（公式列名・型に復元）
PRC_WH (XS Gen1, QAS off, timeouts) / PRC_RM / PRC_ROLE / PRC_SVC
```

- 特徴量は **SQL（window 関数・ASOF JOIN・バケット自己結合）で FEAT に作り、ML.TRAIN_WIDE に結合**。学習は原則サンドボックス（LightGBM/CatBoost）、予測結果を ML.PREDICTIONS に書き戻し、SUB ビューで提出ファイルを生成。
- Training でも DEP 行の `BLOCK_TIME` / `TAXITIME` を**マスクした派生テーブル**を作り、Ranking と同一の SQL で文脈特徴量を計算する（feature-availability parity）。

### 3.2 リポジトリ構成（Step 5 で作成）

```
README.md                 JOAS 構成で育てる（課題・データ・手法・検証・再現手順・ライセンス）
LICENSE                   GPLv3
docs/                     01〜04 + external_data.md + submissions_log.md + decisions.md
sql/                      000_admin_manual.sql（ACCOUNTADMIN 手動）, 010_setup_db.sql, 020_raw_load.sql,
                          030_stg.sql, 040_feat_*.sql, 050_ml_folds.sql, 060_sub.sql（番号順に EXECUTE IMMEDIATE FROM）
src/prc/                  config.py, io_minio.py, io_snowflake.py, features_local.py, train.py, predict.py,
                          validate_submission.py, cv.py
tests/                    合成データで SQL/Python の整合性テスト（parity 検査、提出ゲート）
notebooks/                EDA（出力はコミットしない）
configs/                  experiment yaml（特徴量セット・モデル・fold）
.env.example              必要な環境変数名のみ
```

### 3.3 実験管理

- 1 実験 = `configs/*.yaml`（特徴量セット ID・モデル・パラメータ・fold 定義・シード）。結果は `ML.EXPERIMENTS`（exp_id, config_hash, fold 別 RMSE, 尾部寄与, 学習時間, git sha）に記録。
- 提出ログ `docs/submissions_log.md`: v 番号、exp_id、ローカル CV、LB スコア、変更点、判断。
- 乱数シード固定、`deterministic=True`、依存バージョン固定（`uv lock`）。

## 4. フェーズ計画（日付は 2026 年。各フェーズに成果物・完了条件・レビュー R）

### Phase 0: 登録・基盤（09-10 〜 09-11）

| 作業 | 担当 | 成果物 / 完了条件 |
|---|---|---|
| D1 チーム登録、OSN Discord 参加 | ユーザー | 承認メール、バケット情報、チーム名 |
| D2 認証経路の決定と環境変数登録 | ユーザー | サンドボックスから `mc ls` 相当でバケット一覧が見える |
| D3 管理作業（RM / Budget / TIMEZONE） | ユーザー（手順書は Claude） | `SHOW RESOURCE MONITORS` に PRC_RM |
| リポジトリ骨格（Step 5）: README 骨格、LICENSE、.gitignore、pyproject、`sql/000〜010`、`src/prc` 雛形、`tests/` | Claude | CI 相当のローカルテストが通る |
| Snowflake 基盤 DDL（DB / スキーマ / WH / Stage / File Format / Role）を MCP で実行 | Claude | `PRC_DC2026` の各スキーマが存在 |
| Discord・ranking.html 監視の運用開始（毎日） | 両者 | `docs/decisions.md` に告知記録 |

R0: 骨格と DDL のコードレビュー（セキュリティ: 機密なし、命名、権限最小）。

### Phase 1: データ取得・RAW ロード・初回提出（09-11 〜 09-13）

| 作業 | 成果物 / 完了条件 |
|---|---|
| MinIO から 14 ファイル取得（サンドボックス）、`pyarrow` で schema dump（U13） | `docs/data_profile.md` に schema・行数・欠損表現 |
| 案 C: network rule + EAI + Secret（ユーザーが Snowsight で作成）+ Python SP で OFFICIAL_STAGE へ取得。フォールバック A′ | ステージに 14 ファイル |
| `020_raw_load.sql`: INFER_SCHEMA → RAW 3 テーブル、`METADATA$FILENAME` 付与、COPY_HISTORY 保存 | 行数照合（4,167,797 / DEP 344,841）、`MVT_ID` 一意 |
| 未確認事項の一括回答（U1, U1b, U5, U6, U7, U8, U11, U13）を SQL で集計 | `docs/data_profile.md` 更新。**U1 の結果で §5 の特徴量優先度を再決定** |
| 提出ゲート実装（行数・ID 一致・NaN ゼロ・dtype・schema equals） | `validate_submission.py` + テスト |
| **初回提出 v1**: 空港 × 滑走路 × 時間帯の中央値ベースライン（Snowflake SQL のみで生成） | LB スコア取得、`processedAt` 遅延を記録（U10）。想定 330〜400 秒 |

R1: データプロファイルと U1 結論の独立レビュー（リーク・不整合の見落とし）。

中間目標: LB 提出成功、ベースライン RMSE 把握。

### Phase 2: STG・検証設計・特徴量 v1・GBDT v1（09-13 〜 09-17）

| 作業 | 成果物 / 完了条件 |
|---|---|
| `030_stg.sql`: 型・列名統一、UTC 固定、DEP/ARR 分割、`_flt` 不整合フラグ、目的変数フィルタ（負値、> 120 分は学習から除外。閾値は分布を見て決定） | STG.MOVEMENTS + アサーション |
| **検証設計** `050_ml_folds.sql`: (a) leave-one-month-out（12 fold、主指標は Jan/Jul 2025 fold）、(b) forward holdout（Nov–Dec 2025）。fold 表を ML.FOLDS に固定 | fold 定義書 `docs/validation.md` |
| **マスク派生テーブル**: Training の DEP 行 `BLOCK_TIME`/`TAXITIME` を NULL にした STG.MOVEMENTS_MASKED。文脈特徴量は必ずこちらから計算 | parity テスト（同一 SQL を Ranking と Training に適用し列集合・欠損率が一致） |
| 特徴量 v1（`040_feat_*.sql`）: 静的（空港・滑走路・スタンド・機種・WTC・運航者・市場区分・便種）、カレンダー（空港ローカル時刻・曜日・月・祝日）、スケジュール（`SCHED`, `EOBT_1`, `LOBT`, `IOBT` と離陸時刻の差）、**離陸アンカー混雑**（同一空港/滑走路で離陸前 5/10/20/30 分の離陸数・着陸数、直前/直後離陸との間隔、離陸系列内の位置、PRU 式 20 分スループット）、**スタンド × 滑走路の unimpeded 参照値**（P10/P20、過去 12 か月ローリング、fold 内で計算） | FEAT テーブル群 + 各特徴量の「Ranking で計算可」フラグ表 |
| LightGBM v1（サンドボックス）: 目的変数 = taxi-out − 参照値（additional）、損失 RMSE、early stopping は fold 内 | CV RMSE（fold 別・空港別・尾部）、特徴量重要度 |
| 提出 v2〜v3 | LB と CV の対応を記録 |

R2: 検証設計・parity・リークの独立レビュー（最重要レビュー）。

中間目標: CV と LB の差が説明可能。LB ≤ 280 秒。

### Phase 3: 特徴量 v2・二層モデル・気象（09-17 〜 09-24）

| 作業 | 成果物 / 完了条件 |
|---|---|
| **ARR 由来区間制約**: 同一スタンドの直前インブロック + 最小折返し（空港 × 運航者 × 機種別の分布から推定）→ taxi-out 上限、次インブロック → 下限、予定時刻ソフト上限。特徴量 + 事後クリップ | 効果を CV で定量化 |
| **NM 時刻列の情報量**（U1 結果に基づく）: `AOBT_3`, `LOBT`, `EOBT_1` と実オフブロックの差の分布、派生値シグネチャ検定。`_flt` あり／なしの二層モデル | `docs/data_profile.md` 追記、モデル分岐 |
| **taxi-in ドリフト**: 2026 年 ARR の taxi-in 分布を 2025 年と比較し、空港 × 月の補正項 | ドリフト表 |
| 気象（D4）: REF.WEATHER_HOURLY（IEM ASOS: 気温・視程・降水・風、Open-Meteo 補完）、防除氷条件フラグ、低視程フラグ。ライセンス記録 | `docs/external_data.md` |
| CatBoost v1、LightGBM シード平均（5→10） | アンサンブル候補 |
| 提出 v4〜v8 | |

R3: 特徴量カタログと外部データのライセンス・再現性レビュー。

中間目標: LB ≤ 255 秒。

### Phase 4: 誤差分析・トランスダクティブ・アンサンブル（09-24 〜 10-03）

| 作業 | 成果物 / 完了条件 |
|---|---|
| 誤差分析: 空港 × 月 × 時間帯 × 尾部（> P95）の寄与分解。悪い区分に特徴量を追加 | `docs/error_analysis.md` |
| トランスダクティブ反復: 全 DEP のオフブロック予測 → N(t)・待ち行列特徴量を再計算 → 再予測（2〜3 反復）。CV でも同じ手順を踏む | 効果を fold 別に検証。効かなければ廃止 |
| 目的変数変換の比較（raw / additional / log 変換 + バイアス補正）、サンプル重み | 最良を採用 |
| アンサンブル: LightGBM × シード + CatBoost の OOF 重み最適化。特徴量集合は小さく保つ（bright-lobster の教訓） | OOF RMSE |
| 軽いハイパラ探索（Optuna ≤ 50 試行、fold 固定） | 過剰チューニング禁止 |
| 提出 v9〜v20 | |

R4: モデル・アンサンブルの過学習レビュー（LB 依存の意思決定が無いか）。

中間目標: LB ≤ 235 秒。

### Phase 5: 再現性・文書化・隠しセット対応（10-03 〜 10-09）

| 作業 | 成果物 / 完了条件 |
|---|---|
| 決定論的パイプライン: `make all`（取得 → ロード → STG → FEAT → 学習 → 予測 → 提出ファイル）を新環境で通す | 再現テスト記録 |
| **隠しセット・リハーサル**: Ranking の一部月を「新ファイル」と見立て、配布から提出までを 3 時間以内で通す | 手順書 |
| README を JOAS 構成に整備、`docs/` 整理、LICENSE、外部データの出典・ライセンス表 | レビュー可能な状態 |
| 公開準備: 機密スキャン、GitHub Secret scanning / Push protection、リポジトリ公開タイミングの決定（締切前公開が必要か Discord で確認） | 公開チェックリスト |
| 提出は改善が CV で確認できたもののみ | |

R5: 文書・再現性の独立レビュー（第三者が README だけで再現できるか）。

### Phase 6: 凍結・最終提出（10-09 〜 10-11）

- 10-09: モデル凍結。以後は再現性確認と最終評価ファイル（配布された場合）への適用のみ。
- 10-11 21:00Z までに最終提出完了。API で SUCCEEDED を確認。
- 締切後: リポジトリ公開、Discord で報告、JOAS 論文執筆の判断。

## 5. 特徴量カタログ（v1/v2。列「R」= Ranking で計算可能か）

| 群 | 特徴量 | R | 備考 |
|---|---|---|---|
| 静的 | 空港、滑走路、スタンド、スタンド群（近接クラスタ）、機種、WTC、匿名運航者、市場区分、便種、飛行規則、目的地（国・距離帯） | ○ | 高カーディナリティは fold 内ターゲットエンコーディング（過去期間のみ） |
| 参照値 | スタンド × 滑走路の unimpeded P10/P20（12 か月ローリング、≥10 便）、空港 × 滑走路 × 時間帯の中央値 | ○ | fold 内で再計算（未来を見ない） |
| カレンダー | 空港ローカル時・曜日・月・祝日・季節、離陸時刻の分・時 | ○ | IANA tz 参照表 |
| スケジュール | 離陸 − SCHED、離陸 − EOBT_1、離陸 − LOBT、離陸 − IOBT、LOBT − EOBT_1（遅延更新回数の代理） | ○ | NM 未マッチ行は欠損 |
| NM 実績 | AOBT_3 の有無・離陸 − AOBT_3、ARVT_3 − ARVT_1 | △ | U1 で情報量確認後に採否 |
| 離陸アンカー混雑 | 同一空港・同一滑走路の離陸数（前 5/10/20/30 分、後 5/10 分）、着陸数（前 10/20/30 分）、直前/直後離陸との間隔、離陸系列内順位、PRU 式 20 分スループット、飽和比 | ○ | 他 DEP の離陸時刻は Ranking にも残る |
| 到着文脈 | 同一空港の到着 taxi-in の直前 30/60 分中央値（滑走路運用・地上混雑の代理）、同一スタンド直前インブロックからの経過、次インブロックまでの余裕 | ○ | 区間制約にも使用 |
| スケジュール需要 | 予定出発数（前後 15/30 分）、予定到着数、NM EOBT_1 ベースの需要 | ○ | 実績でなく計画値 |
| 気象 | 気温、視程、降水、風速・風向、防除氷条件、低視程条件（前 1 時間） | ○ | 外部データ（D4） |
| ドリフト | 空港 × 月の taxi-in 中央値差（2026 vs 2025） | ○ | 補正項 |
| 禁止 | 他 DEP 行の実オフブロック由来の N(t)、テスト期間の外部集計値、ADS-B 復元 | × | parity と趣旨のため |

## 6. 検証設計の詳細

- **Fold A（主）**: leave-one-month-out。各月を検証、他 11 か月で学習。参照値・ターゲットエンコーディングは学習側のみで計算。主指標は Jan 2025 fold と Jul 2025 fold の RMSE（テスト月と同季節）。
- **Fold B（補）**: forward holdout（学習 Jan–Oct、検証 Nov–Dec 2025）。時間方向の汎化を確認。
- 全 fold で DEP の `BLOCK_TIME`/`TAXITIME` をマスクした派生表から文脈特徴量を計算（parity）。
- 指標: RMSE（全体・空港別・月別）、尾部寄与（残差² の上位 5% が占める割合）、CV と LB の差の推移。
- リーク検査: 特徴量ごとに「Ranking で計算した値の分布」と「マスク Training で計算した値の分布」を比較（KS 距離）。差が大きい特徴量は疑う。
- LB は sanity check のみ。モデル選択・重みは OOF で決める。

## 7. 提出運用

- 1 日 5 回の枠は「CV で改善が確認できた変更」にだけ使う。プローブ提出は禁止（docs/01 §8）。
- 提出前ゲート → アップロード → API で SUCCEEDED 確認 → `docs/submissions_log.md` 記録。
- 日境界は UTC / CET の両方で安全側に運用。
- 最終週は新規特徴量を凍結し、再現性と最終評価ファイル対応に充てる。

## 8. Snowflake 学習ロードマップ（副目的）

| Phase | 触る機能 |
|---|---|
| 0 | DB/スキーマ/ウェアハウス設計、ロール・権限、リソースモニター・Budget、TIMEZONE、内部ステージ、ファイルフォーマット |
| 1 | INFER_SCHEMA / CREATE TABLE USING TEMPLATE、COPY INTO（メタデータ列、COPY_HISTORY）、External Access Integration + Secret + Python ストアドプロシージャ、EXECUTE IMMEDIATE FROM |
| 2 | Window 関数（RANGE INTERVAL 枠）、ASOF JOIN、QUALIFY、バケット自己結合、TRANSIENT テーブル、TASK |
| 3 | 外部データの取り込み（REF）、半構造化（VARIANT）が出れば FLATTEN、時系列 JOIN、パーセンタイル関数 |
| 4 | Snowpark Python（必要なら）、結果の書き戻し、ビューでの提出生成 |
| 5 | 再現手順のスクリプト化、コスト分析（ACCOUNT_USAGE） |

## 9. リスク登録簿

| リスク | 影響 | 対策 |
|---|---|---|
| 登録・承認の遅れ | 全工程遅延 | D1 を最優先。承認待ちの間に骨格・合成データテストを進める |
| `MVT_TIME` や `_flt` 列が Ranking で想定と異なる（U1） | 戦略前提の崩れ | Phase 1 で即確認。二層モデルと予定時刻ベースの経路を用意 |
| CV が LB と乖離（リーク・parity 違反） | 誤った意思決定 | R2 レビュー、parity テスト、KS 検査 |
| 隠し最終評価で順位変動 | 最終順位低下 | 月ブロック CV、小さい特徴量集合、LB 非依存の選択、決定論パイプライン |
| 運営のルール変更（列の追加空欄化、提出制限） | 手戻り | Discord・ranking.html 毎日確認。`_flt` 非依存の構成 |
| Snowflake コスト超過（特にコンピュートプール） | 予算 | PRC_RM + Budget、QAS 無効、TASK のタイムアウト、Container Runtime 不使用が既定 |
| タイムゾーン事故 | 特徴量全滅 | docs/02 §5 の手順、ロード後アサーション |
| 機密のコミット | 公開時に致命的 | .gitignore、Secret は Snowsight で作成、公開前スキャン |
| 1 人作業の時間不足 | フェーズ遅延 | Claude が自律実行できる部分を最大化（D2 で (i) を選ぶ）。フェーズ 4 を圧縮可能に設計 |
| 外部データが効かない | 工数の無駄 | Phase 3 で早期に効果検証し撤退基準（CV 改善 < 1 秒なら外す） |

## 10. 分担

- **ユーザー**: D1〜D5 の決定と実施（登録、環境変数、Snowsight での管理作業と Secret 作成）、Discord の監視・質問（U3, U9, U12 の運営確認）、各 R レビュー結果の確認と方針判断。
- **Claude**: それ以外の全工程（DDL・SQL・Python・テスト・EDA・特徴量・学習・提出ファイル生成・文書・レビュー手配）。各 Step は小さく作り、独立レビューを通してから次へ進む。

## 11. 未確定事項の解消予定

| 事項 | 解消予定 |
|---|---|
| docs/01 U1, U1b, U5–U8, U11, U13 | Phase 1（データ取得直後） |
| docs/01 U2 | 設計で吸収。Discord 監視継続 |
| docs/01 U3, U9, U10, U12 | Phase 0〜1 で Discord 確認 |
| docs/02 S1 | Phase 1 |
| docs/02 S2〜S5 | 必要時 |
| docs/03 §7 の追加調査 | Step 3b（実行中） |
