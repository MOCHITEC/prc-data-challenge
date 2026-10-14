# 外部特徴量: OPDI の機体の折り返し（地上時間） — 事前登録（2026-09-30）

結果を見る前に書く。判定規則・群・列の定義は測定を始めた後に変えない。

## 0. 位置づけ

- ユーザーの指示で試す（2026-09-30）。**ライセンスに注意あり**: EUROCONTROL PRU の免責事項（非商用・改変禁止）が
  かかり、賞の条件「追加データはオープンソースライセンス」を満たさない可能性がある（`16-external-data.md` §6.1）。
  測定はするが、本番に入れるかは合否と別にユーザーが判断する。
- 公開 LB での実績: OpenAir が 281.72 → 278.36（約 −3.4 s）。ただし OpenAir は `stand_arrival_gap` を持っていない可能性がある。

## 1. 列（`src/prc/opdi.py`、読むのは OPDI と出発便の callsign・時刻だけ。ラベルは読まない）

出発便を OPDI の便に、callsign と最も近い `first_seen`（離陸、MVT との差の中央値 +12 s）で 30 分以内に結びつけ
（`adep` は問わない）、同じ機体（`icao24`）の一つ前の OPDI の便を探す。

| 列 | 定義 |
|---|---|
| `opdi_ground_s` | この便の `first_seen` − 前の便の `last_seen`（地上にいた時間） |
| `opdi_sched_slack_s` | この便の予定オフブロック（SCHED） − 前の便の `last_seen` |

前の便の到着空港がこの出発空港で、地上時間が 15 分以上 24 時間以下のときだけ値を持つ（15 分未満は 0.05 % で、
一つの飛行の追跡が切れて二つに分かれたものと判断する。この下限は測定前に決めた）。

実測の性質（モデルは未学習）:

- 被覆（行ベース）: 学習月 69〜79 %、ランキング 2026-01 **49 %**、2026-07 77 %。未マッチ行は **0 %**（callsign が無い）。
- 2026 年 1 月の欠けは**日単位**: OPDI の 1 月ファイルに 2〜10 日と 23 日の便が一件もない（24 日は一部）。2025 年は
  8 月 19〜20 日を除いてほぼ全日ある。
- `stand_arrival_gap` との順位相関 0.69。`stand_arrival_gap` が欠けている行の 66 % に値がある。

## 2. 群（すべて `configs/main_gap.yaml` ＋ `folds: [jan, forward]`）

| 群 | 変更 |
|---|---|
| base | なし（`wx2_base-f9926dfc0b` をそのまま使う: 同じ設定・同じ fold） |
| noise2 | ハッシュ列 2 本 |
| opdi | `opdi_ground_s`, `opdi_sched_slack_s` |
| opdi_stress | opdi ＋ **予測側だけ**、CRC32 で選んだ日の 1/3 について 2 列を空にする（2026 年 1 月の 10/31 日の欠けを模す） |

## 3. 判定

**採用候補** = 次の 3 つをすべて満たす。

1. `prc.compare base opdi` が ADOPT（全行と K=200 の 95 % CI がともに改善側でゼロを除外）。
2. `prc.compare noise2 opdi` の K=200 の 95 % CI が改善側でゼロを除外。
3. **欠けても害がない**: `prc.compare base opdi_stress` が「B is worse」にならず、かつ空にした日の行だけで
   base → opdi_stress の K=0 の 95 % CI が悪化側でゼロを除外しない。

## 4. 範囲

本番と提出は変えない。合格しても、ライセンスの判断が済むまで本番には入れない。

## 5. 本番の更新に合わせた基準の差し替え（2026-09-30、測定前）

§2 の群は `main_gap`（v14）の上で書いたが、測定を始める前に本番が `main_flseq`（v16）に移り、
その後の提出（v17〜v20）もすべて `main_flseq` の上にある。古い本番の上で測っても今の判断には使えないので、
**測定前に**基準だけを差し替える。判定規則（§3）・列の定義（§1）・群の中身（noise2 / opdi / opdi_stress）は変えない。

- 各群 = `configs/main_flseq.yaml` ＋ `folds: [jan, forward]` ＋ §2 の変更。`--work-dir data/work_flseq`。
- base = `main_flseq` の 4 fold の OOF（`docs/modelb_capacity_2026-09-30/` の基準と同じ実行）。fold の学習は
  fold の並びに依存しない（seed は `seed + k` だけで決まる）ので、jan と forward の行は 2 fold だけで走らせた場合と同じ。
  `prc.compare` は内部結合なので jan と forward の行だけで比べる。
