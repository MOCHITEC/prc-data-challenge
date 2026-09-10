# 01. コンペ一次情報（PRC Data Challenge 2026）

最終確認日: 2026-09-09（公式サイトを直接取得して照合。独立レビュー 1 回反映済み）

出典（一次情報）:
- 概要: https://prc-data-challenge-2026.netlify.app/ （ansperformance.eu/study/data-challenge/dc2026/ から 301 リダイレクト）
- データ: https://prc-data-challenge-2026.netlify.app/data.html
- ランキング・提出手順: https://prc-data-challenge-2026.netlify.app/ranking.html （運営が「緊急・重要ニュースはここに掲載」と明記。定期確認対象）
- 参加資格・賞金条件: https://prc-data-challenge-2026.netlify.app/eligibility.html （§4 の一部は登録フォームの同意文が出典。各項目に付記）
- 背景: https://prc-data-challenge-2026.netlify.app/rationale.html
- 登録フォーム: Google Forms（概要ページ「Request Team Creation!」）
- 公開リーダーボード（第三者ミラー）: https://prc-challenge-2026.vercel.app/
- 公式 API: `https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard`（`nextCursor` でページング、`?teamName=&limit=`（最大 200）で絞り込み可）、`.../teams?page=&size=`、Swagger UI `https://datacomp.opensky-network.org/api/swagger-ui/`
- MinIO 利用手順（2024 年版を参照するよう公式が指示）: https://prc-data-challenge.netlify.app/data.html
- 過去回（Step 3 の入力）: JOAS 論文 https://journals.open.tudelft.nl/joas/article/view/7963 , /8252 , /8750 , /8764 、チームリポジトリ https://github.com/prc-data-challenge-2024 , https://github.com/prc-data-challenge-2025

表記ルール: 「事実」= 公式記載の要約。「推論」= 公式記載から導いたが明示されていないもの。「未確認」= データ取得後または運営（Discord）で確認が必要なもの。

---

## 1. 課題

| 項目 | 事実 |
|---|---|
| 主催 | EUROCONTROL Performance Review Commission (PRC)。OpenSky Network (OSN) が支援 |
| 課題 | 欧州主要空港における **出発便の Taxi-out time（秒）** を予測する |
| 定義 | `TAXITIME_SEC_mvt = MVT_TIME_UTC_mvt - BLOCK_TIME_UTC_mvt`（離陸時刻 − オフブロック時刻） |
| 評価指標 | RMSE（単位: 秒） |
| 順位決定 | 各チームの **全提出のうち最良 RMSE** で順位付け |
| 期間 | 2026-09-01 〜 **2026-10-11 23:59:59 CET**（公式表記のまま） |
| 締切の UTC 換算 | 10 月 11 日はブリュッセルは夏時間（CEST, UTC+2。2026 年の夏時間終了は 10/25）。現地時刻の意図なら **21:59:59Z**、「CET」字義通りなら 22:59:59Z。**内部締切は 2026-10-11 21:00Z** とする（未確認 U9） |
| 賞金 | 上位 3 チーム合計 5,000 EUR。配分は未規定（2024/2025 年は 2,500 / 1,750 / 750 EUR） |
| 空港数 | 公式サイト内で「10」と「11」が混在。掲載表は **10 空港**（下表）。実データで確認（U5） |
| 運営の留保 | 「いかなる事情でもルール変更・チャレンジ中止の権利を留保」「どの段階でも制限対象チームの参加を拒否・停止し得る」と明記 |

対象空港（公式 Table 1。名称は原文どおり）:

| ICAO | IATA | 名称 |
|---|---|---|
| EDDF | FRA | Frankfurt Main |
| EDDM | MUC | Munich |
| EGLL | LHR | London Heathrow |
| EHAM | AMS | Amsterdam Schiphol |
| LEBL | BCN | Josep Tarradellas Barcelona-El Prat |
| LEMD | MAD | Adolfo Suárez Madrid–Barajas |
| LFPG | CDG | Charles de Gaulle |
| LIRF | FCO | Rome–Fiumicino Leonardo da Vinci |
| LTFM | IST | İstanbul |
| LSZH | ZRH | Zürich |

## 2. データ

### 2.1 ファイル構成

| データセット | ファイル | 内容 |
|---|---|---|
| Training | `training_YYYY-MM-01_YYYY-MM+1-01.parquet` × 12（2025-01〜2025-12、各 19〜25 MB） | 2025 年の movement（到着 ARR と出発 DEP の両方）。公式記載の movement 総数は **4,167,797**（「Table 1 の 10 空港における movement 総数」であり Training の合計行数とは明記されていない。行数・`MVT_ID_mvt` 一意性はデータ取得後に照合、U7） |
| Ranking | `ranking.parquet`（27 MB） | 2026 年 1 月・7 月の movement。列は Training と同じ。**DEP 行の `BLOCK_TIME_UTC_mvt` と `TAXITIME_SEC_mvt` が空欄化**（公式が空欄化を明記しているのはこの 2 列。他列の扱いは明示記載なし） |
| Submitting | `submitting.parquet`（1.1 MB） | 提出テンプレート。列は `MVT_ID_mvt`, `TAXITIME_SEC_mvt` のみ。Ranking の DEP 行と 1:1 対応 |

- 予測対象行数: **344,841**（API の `usedPairs`。全提出で同一値）
- Movement 表と NM の Flight 表を左結合（movement 基準）した形で提供。両者に不整合があり得ると公式が明記（「Real world data is messy」）
- Military / Head of State / Sensitive の movement は除外済み
- 合成データではなく実データ

### 2.2 列定義

接尾辞 `_mvt` = 報告空港側の movement 情報、`_flt` = Network Manager (NM) の flight list 情報。

Movement 列:

| 列 | 意味 |
|---|---|
| `MVT_ID_mvt` | movement レコードの一意 ID |
| `FLIGHT_ID_mvt` | NM flight の一意 ID（マッチした場合のみ。未マッチ行は `_flt` 列が全欠損と推論） |
| `FLIGHT_mvt` | 便名（「搭乗券に載る便名」。匿名化の記載なし、U8） |
| `FLIGHT_RULE_mvt` | I / V / NA |
| `ADEP_mvt`, `ADES_mvt` | 出発・到着空港 ICAO |
| `PHASE_mvt` | DEP / ARR |
| `MVT_TIME_UTC_mvt` | 「(best available) movement time」（DEP=離陸、ARR=着陸）。実測保証ではなく入手可能な最良値。公式の空欄化列挙に含まれず、Submitting 節が `TAXITIME = MVT_TIME - BLOCK_TIME` を提示していることから **Ranking の DEP 行にも残ると推論**（U1 で確認） |
| `BLOCK_TIME_UTC_mvt` | ブロック時刻（DEP=オフブロック、ARR=インブロック）。Ranking の DEP 行では空欄 |
| `SCHED_TIME_UTC_mvt` | 予定時刻（DEP=出発、ARR=到着） |
| `AIRCRAFT_TYPE_mvt` | ICAO 機種コード |
| `RUNWAY_mvt` | 滑走路 ID |
| `STAND_mvt` | スタンド（駐機スポット）ID |
| `TAXITIME_SEC_mvt` | タキシー秒数（DEP=taxi-out、ARR=taxi-in）。**目的変数** |

Flight 列（NM 由来）:

| 列 | 意味 |
|---|---|
| `LOBT_flt` | last known off-block time（最後に把握されたオフブロック予定時刻） |
| `CALLSIGN_flt` | コールサイン（例 BAW6VB。匿名化の記載なし、U8） |
| `ADEP_flt`, `ADES_flt` | NM 側の出発・到着空港 |
| `ADES_FILED_flt` | FPL で当初申請された到着空港（`ADES_flt` と異なればダイバート） |
| `MARKET_SEGMENT_flt` | Mainline / Regional / Low-Cost / Business Aviation / All-Cargo / Charter / Military / Other / Not classified |
| `IOBT_flt` | Initial Off-Block Time |
| `FLIGHT_RULE_flt` | I / V / Y / Z |
| `FLIGHT_TYPE_flt` | S / N / G / M（除外済） / X |
| `AIRCRAFT_TYPE_flt` | ICAO 機種コード（NM 側） |
| `WK_TBL_CAT_flt` | 後方乱気流区分 L / M / H / J |
| `AIRCRAFT_OPERATOR_flt` | **匿名化された** ICAO 航空会社コード（公式が匿名化を明記しているのはこの列のみ） |
| `EOBT_1_flt` | FPL ベース (M1) の Estimated Off-Block Time |
| `ARVT_1_flt` | M1 の到着時刻 |
| `AOBT_3_flt` | 実飛行 (M3) の Actual Off-Block Time |
| `ARVT_3_flt` | M3 の到着時刻 |

### 2.3 データアクセス

- OSN の MinIO（S3 互換）バケット。エンドポイント `https://s3.opensky-network.org/`（2024 年版手順より）
- アクセスキーについて公式サイト内で記述が食い違う: 概要ページは「確認メールに返信後、バケットのアクセスキーが届く」、データページは「OSN アカウント画面で自分でアクセスキーを生成する（スクリーンショット付き）」。承認メール受領後に実手順を確定する
- 取得手段（2024 年版手順）: `mc alias set <alias> https://s3.opensky-network.org/ <ACCESS_KEY> <SECRET_KEY>` → `mc cp --recursive <alias>/<bucket>/ <dir>/`。または `pyopensky` の `S3Client`（設定ファイル `[default]` に `username / password / access_key / secret_key`）
- 提出も同じ MinIO の**チーム専用バケット**へアップロード（§3）

## 3. 提出

| 項目 | 事実 |
|---|---|
| 方法 | チーム専用バケットに parquet をアップロード |
| ファイル名 | `<team-name>_v<連番整数>.parquet` |
| 内容 | `submitting.parquet` の `TAXITIME_SEC_mvt` を予測値で埋める（`MVT_ID_mvt` は変更しない） |
| 検証 | `MVT_ID_mvt` 不一致・行欠落・行余剰 はエラー。**NaN/null 値・データ型の扱いは未記載**（U10。2024 年版は欠損を 0 として採点していた） |
| 回数制限 | **現時点で 1 日 5 回**、バケット容量上限 1 GB（公式に「currently」と付記。変更され得る）。「1 日」の境界タイムゾーンは未記載（U9） |
| 採点結果の確認 | API は **status=SUCCEEDED の提出のみ**返す。失敗した提出は API に現れない。アップロード後は `?teamName=<自チーム>&limit=200` で自分の v 番号が出ることを確認する。採点までの所要時間は未記載（2024 年版は 30 分周期。2026 年 API の `processedAt` に固定周期は見られない） |
| 禁止事項 | 「ランキング処理を学習・悪用する試み」を監視し、不正とみなすと明記（§8 の提出ポリシーで対応） |

## 4. 参加資格・賞金条件

参加資格（出典: eligibility.html）:
- 参加は世界に開放。ただし「制裁対象国、または EUROCONTROL 加盟国／包括協定国でない国の団体に所属・由来するチームには制限が適用される」と記載。理由として「制裁の理由に関わらず、これらの国の受賞チームへの賞金送金が実務上困難」を挙げる。登録フォームが「可能な限り」この制限を実装する
- 実際の登録フォームの国別分岐で「Non admissible countries」に送られるのは制裁関連の約 26 か国のみで、日本は通常フロー（レビューで確認）。公式 teams ページには日本所属チームが 3 件掲載済み

賞金対象条件（出典: eligibility.html。全て必須）:
1. 制裁対象の非 EUROCONTROL 国と無関係であること
2. 使用した外部データが全てオープンにアクセス・利用可能で文書化されていること
3. ソースコード全体を **GitHub 上で GNU GPLv3** で公開すること（運営がフォークする）
4. 追加データセットはオープンソースライセンスで公開されているものであること
5. 結果を再現できる十分なドキュメント
6. 独自の解法であること。既存実装の再利用は、(a) 原著者から使用権の許諾を得ていること、かつ (b) アルゴリズム／モデルに大幅な改変を加えていること、の両方を満たす場合のみ可。I/O 部分の書き換えだけでは不十分。「パラメータ追加やデータ特性に合わせたフィルタ調整は十分とみなされ得る」（公式は "can be considered sufficient" と限定的表現）。→ 取り込む外部コード・モデル・データは **GPLv3 互換ライセンス**であることを確認する

その他（出典: 登録フォーム冒頭の必須同意文。eligibility.html には記載なし。正式条件かは運営確認が望ましい）:
- EUROCONTROL / OSN 関係者は参加不可（例外的に賞金なしで許可される場合あり）
- 「no cheating」（一般的な不正禁止）
- データセットは **コンペ終了後に公開（JOAS 論文）されるまで、コンペ以外の用途に使用不可**
- チームに関する情報はすべて公開扱い（メールアドレスは非公開）
- 上位チームは JOAS（Journal of Open Aviation Science）への論文投稿を強く推奨（出典: index.html）

## 5. 登録手順

1. OpenSky Network アカウントを作成（フォームで OSN ユーザー名が必須。メールは OSN 登録メールと一致させる）
2. Google Forms で「Team creation request」を送信
   - 必須: ルール・データ利用条件への同意（Yes。No を選ぶと受付終了）、代表者（Team Member 1）の氏名・メール・OSN ユーザー名・氏名等公開可否（Yes/No の回答必須）・郵送先住所（番地まで）、国、チームの所属団体名、チーム説明・動機、追加メンバー数（0〜14）、認知経路
   - 任意: チーム種別（Academia / Government / Industry / Independent、複数選択可）、代表者個人の所属、追加メンバー各人の氏名・メール・所属・公開可否
3. 運営が審査（「数時間〜最大 1 日で代表者に返信」）→ 確認メール → 返信後にバケット情報受領（アクセスキーの扱いは §2.3 参照）
4. OSN Discord サーバー（招待 https://discord.gg/RPh89jpVVz ）に参加し `#prc-data-competition`（ https://discord.com/channels/1252625490930307142/1304403918515605504 ）をフォロー。ranking.html も毎日確認する

補足（2026-09-10 追記）: 評価セットは 2026-09-04 に差し替えられた（215,876 → 344,841 行、参加チームの記録）。運営は Discord で「Ranking の正解は空港が報告したままの値（`BLOCK == SCHED` の代替報告行も含む）」「オープンで文書化された外部データは可」と述べたと参加チームが記録している（二次情報、docs/03 §8）。

## 6. 競合状況スナップショット（2026-09-09 15:00 UTC 頃に API 取得。最終 processedAt = 14:55 UTC）

| 指標 | 値 |
|---|---|
| 承認チーム数（teams API `totalTeams`） | 336（公式サイトの静的 team ページ数は 332 で API より遅れる。09-09 22:30 UTC 時点では 337） |
| 提出済みチーム数（SUCCEEDED のみ） | 87 |
| 総提出数（SUCCEEDED のみ） | 558 |
| 1 位 RMSE | 246.36 秒（youthful-giraffe, v18） |
| 2 位 / 3 位 | 260.93 / 264.90 |
| 5 位 | 266.40（upstanding-firefly, v8。日本チーム） |
| 10 位 | 278.21 |
| 提出チームの中央値 | 330.12 |
| 日本からの登録チーム | 3（faithful-tulip, upstanding-firefly, talented-teapot） |

生データ: `refs/leaderboard_snapshots/2026-09-09T14Z_leaderboard.json`（ファイル名の `T14Z` は時間単位の粒度。取得は 14:56〜15:02 UTC）

## 7. 未確認事項の回答（2026-09-10、実データで測定。詳細は `docs/data_profile.md`）

**解決済み**

| # | 結論 |
|---|---|
| U1 | **離陸時刻は Ranking の出発便 344,841 行すべてに存在（欠損 0%）**。`BLOCK_TIME_UTC_mvt` と `TAXITIME_SEC_mvt` は 100% 欠損。NM 由来の 5 列（`FLIGHT_ID`, `AOBT_3`, `LOBT`, `EOBT_1`, `IOBT`, `ARVT_3`）はいずれも欠損 1.53% で残存 = **98.47% で管制側のオフブロック時刻が使える**。情報量も確認: `MVT_TIME − AOBT_3` の空港別標準偏差は 171〜430 秒で、派生値（`MVT − 定数`）ではない。`BLOCK − AOBT_3` の中央値は −118〜+296 秒、60 秒以内の一致は 9〜26% にすぎず、2 つの時計のずれこそが学習対象になる。LTFM だけ中央値 +296 秒・一致 8.7% で系統的にずれている |
| U1b | 同一便の到着行が Ranking 内にある「域内便」は **14.77%**。`_flt` 列は既に 98.5% 残っているので現時点では不要。運営が将来空欄化した場合の保険として記録 |
| U5 | **10 空港**（EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM）。公式サイトの「11」は誤記 |
| U6 | 目的変数の欠損 0%、負値 0.02%、2 時間超 0.03%、最大 131,167 秒（36 時間）。中央値 912 秒、標準偏差 546 秒。空港別中央値は EHAM 742 秒〜EGLL 1,319 秒。`TAXITIME = MVT − BLOCK` の恒等式は全行で成立 |
| U7 | 訓練 **4,167,797 行**（公式記載と完全一致）、`MVT_ID_mvt` は全件一意、出発便 2,085,047 行。NM 未マッチ率は訓練 1.08% / Ranking 1.53%（Ranking の方がわずかに高い） |
| U8 | `AIRCRAFT_OPERATOR_flt` はハッシュ化（64 文字）。一方 **`CALLSIGN_flt` は匿名化されていない**（例: `DLH8UL` = ルフトハンザ）。先頭 3 文字が実在の ICAO 航空会社コードなので、ハッシュとは別に実航空会社を特徴量にできる。`FLIGHT_mvt` も実便名（`LH982`）だが Ranking 行の被覆率は 78.9% しかなく階層フォールバックが必須 |
| U11 | `RUNWAY_mvt` と `STAND_mvt` は Ranking でも欠損 0%。スタンドの水準被覆率 99.92%、滑走路 100%、機種 99.88%、運航者 98.41% |
| U13 | 全時刻列が `timestamp[us, tz=UTC]`（タイムゾーン付き）。マイクロ秒精度。Ranking の空欄は真の null。`MVT_ID_mvt` と `FLIGHT_ID_mvt` は **Float64**（整数ではない）ので提出テンプレートの型に注意 |

**報告エラー体制の実測**（尾部の主因）: `BLOCK_TIME == SCHED_TIME` の代替報告は空港別に 0.18%〜4.48%、**LIRF が 4.48% で最多**。4 時間超の taxi-out も LIRF のみ 0.07% で発生。1 日ずれ（±86,400 秒）は許容 60 秒では検出されず、別の形（不定のオフセット）である可能性が高いので Phase 2 で分布を直接見る。

**素朴な予測器のベンチマーク（2025 年 in-sample RMSE）**: 全体平均 546.4 → 空港平均 514.8 → 空港 × 滑走路 504.5 → 空港 × 滑走路 × スタンド 469.1 → **`MVT − AOBT_3` = 384.9 秒**。最後の値は他チームの公開報告と完全に一致した。

**運営確認が残るもの**

| # | 事項 | 確認方法 |
|---|---|---|
| U3 | 日本チームへの賞金送金の実務可否 | Discord |
| U9 | 締切の実 UTC（21:59:59Z か 22:59:59Z）と 1 日 5 回の日境界タイムゾーン | Discord。確認できなければ内部締切 10-11 12:00Z で運用 |
| U10 | 予測値の NaN / 負値の採点上の扱い、失敗提出が回数枠を消費するか | 初回提出で実測（ゲートで NaN と負値は事前に排除済み） |
| U12 | PRU 公開の月次集計（テスト期間を含む）の利用可否 | Discord。確認までは 2026 年の値を使わない |

## 8. この時点での戦略方針（計画書で詳細化）

**使えるもの（推論・確認待ち）**
- Ranking では離陸時刻 `MVT_TIME_UTC_mvt` が既知と推論される（U1）。問題は実質「離陸時刻が分かっている出発便のオフブロック時刻を推定する」課題になる。
- 同時間帯の他便のうち **離陸実績（他 DEP 行）・着陸/インブロック実績・taxi-in（ARR 行）・予定時刻・NM 由来の予定/実績時刻**は Ranking にも残る。滑走路運用（`RUNWAY_mvt`）や離陸系列内の位置、直前 N 分の離陸・着陸数は復元できる。
- ARR 行が完全に残るため、**オフブロック時刻の区間制約**が使える: 同一スタンドに直前にインブロックした便の時刻 + 最小折返し時間 ≤ オフブロック（taxi-out の上限）、同一スタンドに次にインブロックする便の時刻 ≥ オフブロック（下限）、予定出発時刻より大幅に早いオフブロックは稀（ソフト上限）。連結キーは STAND + 時刻近接 + 機種 + 匿名運航者（機体識別子は無い）。
- 2026 年の ARR 行の taxi-in 分布は全て見えるため、2025 年との比較で空港ごとのドリフト（工事・誘導路変更）を検知・補正できる。

**使えないもの・落とし穴**
- Ranking では**他の全 DEP 行のオフブロック実績も空欄**。文献で最重要の N(t)（プッシュバック済みで滑走路待ちの機数）は実績からは計算できない。Training で他 DEP 行の `BLOCK_TIME` から混雑特徴量を作ると、CV では良く見えて Ranking では計算不能になる（silent leakage）。**文脈特徴量は Ranking で見える列のみから構築し、Training 側でも DEP 行の `BLOCK_TIME` / `TAXITIME` をマスクした状態で同一コードで計算する（feature-availability parity）**。必要なら「全 DEP のオフブロックを予測 → N(t) 再計算 → 再予測」のトランスダクティブ反復を検討。
- `AOBT_3_flt` 等の NM 実績値は情報量が空港・経路で異なり得る（U1）。**`_flt` 実績値あり／なし（未マッチ）の二層モデルを最初から設計し、`_flt` 実績値が後日空欄化されても成立する構成**にする。

**外部データ方針**
- 主要候補はオープンな気象データ（METAR 時系列: 視程・降水・気温・風。Iowa State ASOS アーカイブ、Open-Meteo 履歴 API）。1 月は防除氷（気温・降水・湿度条件）が taxi-out の尾部を支配し得るためフラグ化する。ライセンスと取得手順を文書化する。
- **採らない手法**: ADS-B 地上トラック等からテスト期間の目的変数を直接再構成すること（趣旨違反・ライセンス不適合リスク）。PRU 公開のテスト期間集計値の利用（U12 の確認まで）。便名・コールサインは 2025 年内の統計特徴量としてのみ使う。

**検証設計**
- テストは Jan/Jul 2026 で、前年同月（Jan/Jul 2025）は学習に含まれる構造。CV は (a) 月ブロック leave-one-month-out（特に Jan 2025 / Jul 2025 fold）と (b) 末尾月 forward holdout の両方を標準とし、上記マスクを CV 側にも適用する。RMSE は尾部支配なので fold ごとの尾部寄与も記録する。

**提出ポリシー（「ランキング処理の学習・悪用」に該当しないためのガードレール）**
1. 定数提出・部分差し替え提出・診断目的の提出は行わない。提出は常にフルモデルの出力のみ。
2. 各提出に目的・変更点・ローカル CV 値を記録した提出ログを残す（再現性文書と監査対応を兼ねる）。
3. モデル選択・アンサンブル重みはローカル CV で決め、LB は sanity check に限定する。LB 選択で組んだアンサンブルは隠しセットで劣化しやすい。
4. 隠し最終評価（U2）を既定シナリオとし、**新しい ranking/submitting ファイルが配布されてから数時間以内に再学習なしで提出できる決定論的パイプライン**を維持する。締切直前週は LB 調整ではなく再現性確認に充てる。
5. アップロード前に NaN ゼロ件・dtype・行数 344,841・`MVT_ID_mvt` 一致を機械的に検証するゲートを設ける。
6. 最終提出は 2026-10-11 21:00Z までに完了する。

**公開・再現性**
- コード GPLv3 公開・再現性文書が賞金条件。リポジトリは最初から公開可能な構成にする（機密識別子を含めない。docs/02 参照）。選考委員会は独自性・文書品質も評価する前例があるため、論文（JOAS）水準の README を並行して育てる。
