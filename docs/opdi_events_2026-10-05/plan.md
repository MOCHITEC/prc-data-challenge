# OPDI の駐機場離脱イベント（exit-parking_position）を観測した off-block として使う — 事前登録（2026-10-05）

結果（パイプラインの CV）を見る前に書く。被覆と単純な精度の確認（`coverage_accuracy.txt`、`signal_check.txt`）は済ませた。

## 0. 位置づけ・ライセンス

- `peaceful-babbage` の首位との差の調査（`docs/research_2026-10-01/gap/README.md`）の推奨 2。
- **ライセンスに注意**: OPDI は EUROCONTROL の一般の免責事項（出典明記・非商用・無断改変禁止）で公開されている。
  コンペの賞の条件は「追加データはオープンソースのライセンスで公開されていること」（eligibility.html）。
  LB の順位には関係しない。賞の対象に関わる判断はユーザーが行う。

## 1. データ

- OPDI v002 flight events、10 日ごとのファイル 45 本（2024-12-26〜2026-01-30、2026-06-29〜08-08）。
  `scripts/opdi_events_ingest.py` で 10 空港の地上イベントだけを残した。**2025-10-02〜12 のファイルはサーバーで 404**、
  **2026-01-02〜10 はイベントが 0 件**（ファイルはあるが中身が 12-31 と 01-01 だけ）。
- 結びつけ（`link_events.py`）: 同じ空港・同じ callsign（未マッチ行は FLIGHT_mvt が ICAO 形式のとき）で、
  [MVT − 2 h, MVT − 30 s] の exit-parking のうち、出発便のスタンドと一致するものを優先、無ければ最も早いもの。
- 列（OPDI_EXIT_COLS）: `opdi_exit_est_s`（MVT − 離脱）、`opdi_exit_minus_aobt3_s`（離脱 − AOBT_3）、
  `opdi_exit_stand_match`、`opdi_exit_n`、`opdi_exit_in_use_s`。ラベルは読まない。

## 2. 事前の確認（モデル無し）

| | 被覆 2025 | 2026-01 | 2026-07 |
|---|---|---|---|
| LSZH | 45〜83 % | 43 % | 78 % |
| EDDF | 0〜2 %（〜8 月）、9〜19 %（9〜12 月） | 15 % | 25 % |
| 他の 8 空港 | ほぼ 0（LEBL は 1〜3 月 10〜13 %、LEMD は 11 月 15 %） | 0 | 0 |

ランキング全体: マッチ行 6.5 %、未マッチ行 1.9 %。単独の時計としては AOBT_3 より悪い（LSZH 457 s 対 251 s）が、
AOBT_3 の誤差の説明力は強い: Spearman(label − proxy, 離脱 − AOBT_3) = LSZH −0.58、EDDF の一致スタンド −0.74。
1 列群だけのモデル（1〜6 月で学習、7〜12 月で評価）で LSZH の（label − proxy）の誤差 240 → 166 s。

## 3. 群（`configs/main_flseq.yaml` ＋ 4 fold、`--work-dir data/work_flseq`）

| 群 | 変更 |
|---|---|
| base | なし（`main_flseq-8615e6a194`） |
| ox | `fe_opdi_exit: true` |
| ox_stress | ox ＋ 予測側だけ、CRC32 で選んだ日の 1/3 で 5 列を空にする（2026-01-02〜10 の欠けを模す） |

## 4. 判定

**採用候補** = 次の 3 つをすべて満たす。
1. `prc.compare base ox` が ADOPT（全行と K=200 の 95 % CI がともに改善側でゼロを除外）。
2. K=200 の差が −0.5 s より大きい改善であること（列を足すだけの揺れ ±0.3 s を十分超える）。満たさない場合は 5 列の
   ハッシュ列の群を追加して `prc.compare noise ox` の K=200 で判定する。
3. **欠けても害がない**: `prc.compare base ox_stress` が「B is worse」にならず、空にした日の行だけで base → ox_stress の
   K=0 の 95 % CI が悪化側でゼロを除外しない（`docs/opdi_2026-09-30/evaluate_stress.py` と同じ手順）。
   不合格なら、学習側でも日を空にする群（`opdi_exit_train_null_day_share` を 0.15）を同じ判定で追加する。

## 5. 範囲

本番と提出は変えない。合格したら、本番（v33 系、ADS-B 残差あり）に入れる手順と表を引き継ぎに書く。

## 6. 追記（2026-10-06 01:55 UTC、CV の前に書く）: 今の ADS-B 残差（v39 相当）の上で OPDI が効くか

`handoff/2026-10-05/README.md` の `--opdi` の確認を、この環境に取り込んだ ADS-B でそのまま測る。

- **群**（4 か月・a2・`docs/appear_tier_2026-10-05/cv_months.py` と同じ fold、V2 の設定、appear は fill = v39 の残差に相当）:
  cur、opdi（cur ＋ `scripts/adsb_opdi.py` の 6 列）、opdishuf（6 列を空港の中で並べ替えた対照）。
- **空港**: LSZH と EDDF（学習の 4 か月のうち 2025 年 11・12 月に EDDF の OPDI が 15〜19 % あり、2026 年も 15〜25 %）。他は空。
- **判定**: opdi が cur と opdishuf の両方に対して、全行の RMSE の差の日単位 95 % CI が改善側でゼロを除外すること。
- 合格しても、OPDI のライセンス（EUROCONTROL の免責事項: 非商用・無断改変禁止）の判断がユーザーから出るまでは提出候補にしない。
