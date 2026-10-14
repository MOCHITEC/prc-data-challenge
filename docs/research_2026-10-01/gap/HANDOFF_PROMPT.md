# 引き継ぎ: ADS-B／OPDI による off-block の観測範囲の拡大（2026-10-05 時点）

## 目的
首位との差（v33 250.95 対 strong-crane 213.92）を詰める。
本命は「実際にスタンドを離れた時刻（off-block）を観測できる行」を増やすこと。
モデルの調整はその次。

## 現状（2026-10-05 確認）
- チームのベストは v33 = 250.9496 で、38 位 / 230。中身は `hybrid_blend_l2_pa_opt` + 学習 ADS-B 残差。
- 首位 213.92、10 位 227.68、20 位 237.57。上位はまだ伸びている。
- 締切は 2026-10-11 21:59Z（CEST 解釈）。内部の最終提出は 10-11 21:00Z まで。
- 提出は 1 日 5 回（UTC と Brussels の両方の日境界で安全側に数える）。09-30 から MinIO CLI のみ。最終評価に別セットの告知は無い。
- 主な作業ブランチは `claude/model-b-and-arrival-features`。調査記録は `claude/peaceful-babbage-md9l2x` の `docs/research_2026-10-01/`（特に `gap/README.md`）にある。

## 調査で分かったこと（根拠: docs/research_2026-10-01/gap/README.md）
1. **ADS-B の取り出しが弱い。** ランキング行のうち、地上 fix がある行は約 71%。そのうち駐機区間を検出できたのは約 43%、プッシュバックを検出できたのは約 21% しかない。
   - 検出は後退動作を条件にしている。そのため「最初の動きが前向き」（空港により 20〜42%）と「最初に見えたとき既に動いていた」（18〜27%）を落としている（`adsb-coverage-breakdown`）。
   - LFPG は地上 fix が 58% あるのに、スタンドの検出は 3%。
2. **上位相当の公開実装**（smart-jigsaw、github.com/yookeew/dumb-puzzle、09-29 時点）。
   - スタンドで最初に見えた時刻を off-block とする appear tier を使う（プッシュバック時にトランスポンダが入る）。
   - 停止→移動の dwell tier も使う。
   - スタンド座標は OSM ではなく X-Plane Scenery Gateway（2,838 スタンド、GPLv2）。
   - ADS-B だけで LB −19 秒（296.7 → 277.7）。
   - 取得率: EHAM 71〜77%、EDDM 76%、LEBL 34〜62%、EGLL 19〜72%、LSZH 19〜32%。
   - **コードは参照のみ。コピーしない**（賞金条件: 独自性。他者コードの再利用は許諾と大幅な改変が必要）。
3. **未使用の観測源: OPDI flight events の exit-parking_position（プッシュバック）。**
   - 上限の取得率は LSZH 79%、EDDF 47%。adsb.lol の EDDF は 2〜8%。
   - Discord（09-09、kind-mango の中継）で open data と認定された、と二次情報で伝わっている。
   - ライセンス表記（CC-BY 4.0 か PRU の disclaimer か）は未確認。
   - 既存の `src/prc/opdi.py` は turnaround にしか使っていない。過去の判定「LOW YIELD」は 10 日窓での全体評価。
4. **未マッチ行（1.5%、LB MSE の 24〜39% と推定）では、ADS-B が唯一の時計になる。** v26 の分解では、LB の変化の 86% が ADS-B 残差が補正する未マッチ 2,416 行に集中していた。
5. **LTFM は 2025・2026 とも adsb.lol に地上 fix が 0**（6 日の標本で確認）。ADS-B では届かない。

## やること（優先順）
### A. ADS-B 検出の第 2 段（最優先）
- **appear tier**: 自スタンドから一定距離以内で、その便の最初の地上 fix の時刻を off-block 候補にする。
- **前向き初動・既に動いていた行**: 動き出し時刻や誘導路への進入時刻を推定し、新しい列として残差モデルの入力に加える（例: `t_first_seen_at_stand`、`t_first_move`、`t_taxiway_entry`）。
- **スタンド座標の補完**: OSM で欠けるスタンドを補う。Gateway は GPLv2 なので、使う場合はライセンス表記と外部データの台帳（`docs/external_data.md`）に記録する。
- 対象空港は LFPG・LIRF・EDDF・EHAM・LEBL・LSZH・EGLL・LEMD。
- **判定**: 既存の学習 ADS-B 残差と同じ CV（2025 年の ADS-B ありの日、日単位の fold）で、`gain s²` と対差分の CI を出す。現行の残差モデル（v33 の base）に対して改善したら採用。
- **パリティ**: 2025 年と 2026 年で、各列の被覆率が大きく違わないことを空港ごとに確認する（OPDI が 2026 年に崩落した前例がある）。

### B. OPDI の exit-parking_position を off-block の観測源に追加
- まずライセンスを一次情報で確認する。開かれていない場合は中止。
- 2025 年の学習日で、ラベル（BLOCK）との一致精度を空港ごとに測る（中央値・IQR）。
- 2026 年の被覆率を測る。
- A と同じ残差の枠組みに入力として足し、同じ判定をする。

### C. 未マッチ行専用の ADS-B 残差
- 2025 年の未マッチ行のうち ADS-B がある行で学習する（行数が少ないので、正則化を強めにする）。
- 判定は未マッチ行の中で、尾部（上位 20〜200 行）を除いた差も併記する。尾部の数行で判定がひっくり返るため。

### D. 保険
- EHAM・EDDF・LFPG の未マッチ行で ms（MVT − SCHED）が 3 万〜9.6 万秒の約 27 行を、ADS-B で前日から駐機していたかと照合する。
  - 今は普通のタクシー時間（約 2,000 秒）を予測している。
  - 1 行でも外すと LB +40 秒級だが、事前確率は低い。

## 守ること
- **リーダーボードから正解を推測しない。** 一部の行だけを動かす探りの提出（probe）はしない。提出は CV で採用されたモデル全体の出力だけ。
- **空欄化された列の副チャネルを使わない**: ID の並び、±6 秒の秒ジッタ。調査で存在は確認済みで、どちらも使っていない。
- 外部データはオープンで文書化されたものだけ（賞金条件）。コードは GPLv3 で公開されるので、他者のコードや GPLv2 データの扱いに注意する。
- 提出前に、他のセッションの提出（版番号、当日の残り枠）を `experiments/submissions.jsonl` の最新とバケットの一覧で必ず確認する。
- 事前登録してから走らせる（チームの慣例）。結果は `docs/` と `experiments/` に記録し、コミットしてプッシュする。

## 参考になる既存資料
- `docs/research_2026-10-01/gap/README.md`（今回の結論と予算表）
- `docs/research_2026-10-01/gap/public/`（smart-jigsaw と kind-mango の調査メモ）
- `docs/competitor_research_2026-09-21/results/adsb-learned-residual.md`（現行の残差モデル）
- `scripts/adsb_stage1_feasibility.py`、`scripts/adsb_learn_features.py`、`scripts/adsb_learn_cv.py`、`scripts/adsb_learn_apply.py`
- `docs/competitor_research_2026-09-21/results/external-data-survey-2026-09-29.md`（OPDI、OpenSky Trino 不可などの判断）
