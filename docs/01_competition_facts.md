# 01. コンペ一次情報（PRC Data Challenge 2026）

最終確認日: 2026-09-09（公式サイトを直接取得して照合）

出典（一次情報）:
- 概要: https://prc-data-challenge-2026.netlify.app/ （ansperformance.eu/study/data-challenge/dc2026/ からリダイレクト）
- データ: https://prc-data-challenge-2026.netlify.app/data.html
- ランキング・提出手順: https://prc-data-challenge-2026.netlify.app/ranking.html
- 参加資格・賞金条件: https://prc-data-challenge-2026.netlify.app/eligibility.html
- 背景: https://prc-data-challenge-2026.netlify.app/rationale.html
- 登録フォーム: Google Forms（概要ページの「Request Team Creation!」リンク）
- 公開リーダーボード: https://prc-challenge-2026.vercel.app/
- リーダーボードAPI: https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard （`nextCursor` でページング）
- MinIO 利用手順（2024年版を参照するよう公式が指示）: https://prc-data-challenge.netlify.app/data.html

「事実」は公式記載の要約。「未確認」は公式に記載が見当たらず、データ取得後または運営（Discord）で確認が必要な項目。

---

## 1. 課題

| 項目 | 事実 |
|---|---|
| 主催 | EUROCONTROL Performance Review Commission (PRC)。OpenSky Network (OSN) が支援 |
| 課題 | 欧州主要空港における **出発便の Taxi-out time（秒）** を予測する |
| 定義 | `TAXITIME_SEC_mvt = MVT_TIME_UTC_mvt - BLOCK_TIME_UTC_mvt`（離陸時刻 − オフブロック時刻） |
| 評価指標 | RMSE（単位: 秒） |
| 順位決定 | 各チームの **全提出のうち最良 RMSE** で順位付け |
| 期間 | 2026-09-01 〜 **2026-10-11 23:59:59 CET** |
| 賞金 | 上位3チーム合計 5,000 EUR |
| 空港数 | 公式サイト内で「10」と「11」が混在。掲載表は **10空港**（下表） |

対象空港（公式 Table 1）:

| ICAO | IATA | 名称 |
|---|---|---|
| EDDF | FRA | Frankfurt Main |
| EDDM | MUC | Munich |
| EGLL | LHR | London Heathrow |
| EHAM | AMS | Amsterdam Schiphol |
| LEBL | BCN | Barcelona-El Prat |
| LEMD | MAD | Madrid-Barajas |
| LFPG | CDG | Paris Charles de Gaulle |
| LIRF | FCO | Rome-Fiumicino |
| LTFM | IST | Istanbul |
| LSZH | ZRH | Zurich |

## 2. データ

### 2.1 ファイル構成

| データセット | ファイル | 内容 |
|---|---|---|
| Training | `training_YYYY-MM-01_YYYY-MM+1-01.parquet` × 12（2025-01〜2025-12、各 19〜25 MB） | 2025年の全 movement（到着 ARR と出発 DEP の両方）。合計 4,167,797 行 |
| Ranking | `ranking.parquet`（27 MB） | 2026年1月・7月の movement。列は Training と同じ。**DEP 行の `BLOCK_TIME_UTC_mvt` と `TAXITIME_SEC_mvt` のみ空欄化** |
| Submitting | `submitting.parquet`（1.1 MB） | 提出テンプレート。列は `MVT_ID_mvt`, `TAXITIME_SEC_mvt` のみ。Ranking の DEP 行と 1:1 対応 |

- 予測対象行数: **344,841**（リーダーボードAPIの `usedPairs`。全提出で同一値）
- Movement 表と NM の Flight 表を左結合（movement 基準）した形で提供。両者に不整合があり得ると公式が明記（「Real world data is messy」）
- Military / Head of State / Sensitive の movement は除外済み
- 合成データではなく実データ

### 2.2 列定義

接尾辞 `_mvt` = 報告空港側の movement 情報、`_flt` = Network Manager (NM) の flight list 情報。

Movement 列:

| 列 | 意味 |
|---|---|
| `MVT_ID_mvt` | movement レコードの一意ID |
| `FLIGHT_ID_mvt` | NM flight の一意ID（マッチした場合） |
| `FLIGHT_mvt` | 便名 |
| `FLIGHT_RULE_mvt` | I / V / NA |
| `ADEP_mvt`, `ADES_mvt` | 出発・到着空港 ICAO |
| `PHASE_mvt` | DEP / ARR |
| `MVT_TIME_UTC_mvt` | 実 movement 時刻（DEP=離陸、ARR=着陸）。**Ranking でも DEP 行に残る**（空欄化対象外） |
| `BLOCK_TIME_UTC_mvt` | ブロック時刻（DEP=オフブロック、ARR=インブロック）。Ranking の DEP 行では空欄 |
| `SCHED_TIME_UTC_mvt` | 予定時刻（DEP=出発、ARR=到着） |
| `AIRCRAFT_TYPE_mvt` | ICAO 機種コード |
| `RUNWAY_mvt` | 滑走路ID |
| `STAND_mvt` | スタンド（駐機スポット）ID |
| `TAXITIME_SEC_mvt` | タキシー秒数（DEP=taxi-out、ARR=taxi-in）。**目的変数** |

Flight 列（NM 由来）:

| 列 | 意味 |
|---|---|
| `LOBT_flt` | last known off-block time |
| `CALLSIGN_flt` | コールサイン |
| `ADEP_flt`, `ADES_flt` | NM 側の出発・到着空港 |
| `ADES_FILED_flt` | FPL で当初申請された到着空港（`ADES_flt` と異なればダイバート） |
| `MARKET_SEGMENT_flt` | Mainline / Regional / Low-Cost / Business Aviation / All-Cargo / Charter / Military / Other / Not classified |
| `IOBT_flt` | Initial Off-Block Time |
| `FLIGHT_RULE_flt` | I / V / Y / Z |
| `FLIGHT_TYPE_flt` | S / N / G / M（除外済） / X |
| `AIRCRAFT_TYPE_flt` | ICAO 機種コード（NM 側） |
| `WK_TBL_CAT_flt` | 後方乱気流区分 L / M / H / J |
| `AIRCRAFT_OPERATOR_flt` | 匿名化された ICAO 航空会社コード |
| `EOBT_1_flt` | FPL ベース (M1) の Estimated Off-Block Time |
| `ARVT_1_flt` | M1 の到着時刻 |
| `AOBT_3_flt` | 実飛行 (M3) の Actual Off-Block Time |
| `ARVT_3_flt` | M3 の到着時刻 |

### 2.3 データアクセス

- OSN の MinIO (S3 互換) バケット。エンドポイント `https://s3.opensky-network.org/`（2024年版手順より）
- OSN アカウントでアクセスキーを生成し、`mc`（MinIO client）または `pyopensky` の `S3Client` で取得
- チーム承認後にバケットのアクセス情報が届く

## 3. 提出

| 項目 | 事実 |
|---|---|
| 方法 | チーム専用バケットに parquet をアップロード |
| ファイル名 | `<team-name>_v<連番整数>.parquet` |
| 内容 | `submitting.parquet` の `TAXITIME_SEC_mvt` を予測値で埋める |
| 検証 | `MVT_ID_mvt` 不一致・行欠落・行余剰 はエラー |
| 回数制限 | **1日 5 回**。バケット容量上限 1 GB |
| 禁止事項 | ランキング処理を「学習・悪用」する行為は不正として監視対象 |

## 4. 参加資格・賞金条件

- 参加は世界に開放。ただし制裁対象国および「EUROCONTROL 加盟国／包括協定国でない国」の団体に所属するチームには制限が適用され得る、と記載（実運用は登録フォームで判定）
- 賞金対象条件（全て必須）:
  1. 制裁対象の非 EUROCONTROL 国と無関係であること
  2. 使用した外部データが全てオープンにアクセス・利用可能で文書化されていること
  3. ソースコード全体を **GitHub 上で GNU GPLv3** で公開すること（運営がフォークする）
  4. 追加データセットはオープンソースライセンスで公開されているものであること
  5. 結果を再現できる十分なドキュメント
  6. 独自の解法であること（既存実装の I/O 差し替えだけは不可。大幅な改変は可）
- EUROCONTROL / OSN 関係者は参加不可（例外的に賞金なしで許可される場合あり）
- データセットは **コンペ終了後に公開（JOAS 論文）されるまで、コンペ以外の用途に使用不可**
- 上位チームは JOAS（Journal of Open Aviation Science）への論文投稿を強く推奨

## 5. 登録手順

1. OpenSky Network アカウントを作成（登録フォームで OSN ユーザー名が必須）
2. Google Forms で「Team creation request」を送信。必須項目: 代表者の氏名・メール・OSN アカウント・所属・郵送先住所、チーム種別（Academia / Government / Industry / Independent）、国、チーム説明、追加メンバー情報（最大 14 名）、氏名公開の同意
3. 運営が審査 → 確認メール → 返信後にバケットのアクセスキー受領
4. OSN Discord サーバーに参加し `#prc-data-competition` をフォロー（重要な告知はここ）

## 6. 競合状況スナップショット（2026-09-09 14:00 UTC 時点、API より）

| 指標 | 値 |
|---|---|
| 登録チーム数（公式サイト team ページ数） | 332 |
| 提出済みチーム数 | 87 |
| 総提出数 | 558 |
| 1位 RMSE | 246.36 秒（youthful-giraffe, v18） |
| 2位 / 3位 | 260.93 / 264.90 |
| 5位 | 266.41 |
| 10位 | 278.21 |
| 提出チームの中央値 | 330.12 |
| 日本からの登録チーム | 3 |

生データ: `refs/leaderboard_snapshots/2026-09-09T14Z_leaderboard.json`

## 7. 未確認事項（データ取得後・運営確認が必要）

| # | 事項 | なぜ重要か | 確認方法 |
|---|---|---|---|
| U1 | Ranking の DEP 行で `AOBT_3_flt` / `LOBT_flt` / `IOBT_flt` / `EOBT_1_flt` が残っているか（公式は `BLOCK_TIME_UTC_mvt` と `TAXITIME_SEC_mvt` のみ空欄化と記載） | `AOBT_3_flt` は NM 側の実オフブロック時刻で、目的変数にほぼ直結する可能性 | データ取得後に欠損率を集計。Discord の運営見解も確認 |
| U2 | 最終順位に公開リーダーボードと別の隠しサブセットが使われるか | 2024年版では最終ランキングに追加便が使われた。2026年版のページには記載なし | Discord / 運営に質問 |
| U3 | 日本（非 EUROCONTROL 加盟国、非制裁国）のチームの賞金受給資格 | 資格ページの文言が曖昧 | 登録フォーム送信時または Discord で確認 |
| U4 | Ranking に含まれる ARR 行および他 DEP 行の `MVT_TIME_UTC_mvt` を特徴量に使ってよいか | 「事後分析用モデル」という趣旨に合致するが明文化はない | Discord で運営見解を確認。禁止されない限り使用前提で設計 |
| U5 | 空港数が 10 か 11 か（`ADEP_mvt` の実際のユニーク数） | データ分割・特徴量設計に影響 | データ取得後に集計 |
| U6 | Training の DEP 行に `TAXITIME_SEC_mvt` 欠損がどの程度あるか | 学習データ量・欠損処理方針 | データ取得後に集計 |

## 8. この時点での戦略的含意（計画書で詳細化）

- Ranking では **離陸時刻 `MVT_TIME_UTC_mvt` が既知**。問題は実質「離陸時刻が分かっている出発便のオフブロック時刻を推定する」課題であり、同時刻帯の他便（到着・出発）の実時刻から混雑・滑走路運用を復元できる。
- NM 由来の `AOBT_3_flt` 等が Ranking に残っていれば、モデルの中心は「NM オフブロック時刻と空港ブロック時刻の差分補正」になる（U1 の確認が最優先）。
- 1日 5 提出・全提出の最良値採用のため、ローカル検証（時系列 CV）の信頼性が勝敗を左右する。
- コード GPLv3 公開・再現性文書が賞金条件。リポジトリは最初から公開可能な構成にする。
