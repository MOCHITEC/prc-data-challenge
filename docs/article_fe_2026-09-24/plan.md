# 記事 2 本の未実施手法を試す — 事前登録（2026-09-24）

結果を見る前に書く。**判定規則・群・候補の定義は、測定を始めた後に変えない。**

## 0. 動機

ユーザーから 2 本の一般的な特徴量エンジニアリング記事が示され、「やっていないものは試す」ことになった。

- Zenn（colum2131）「序盤に試すテーブルデータの特徴量エンジニアリング」
- Aru「機械学習のための前処理＆特徴量エンジニアリングの手法【Python】」

両記事の手法を、このブランチの系統（`claude/model-b-and-arrival-features` の 08c2a74、v14 = LB 279.5234）と
突き合わせた。**既に測定済み**のものは再測定しない。

| 記事の手法 | このリポジトリでの状態 | 出典 |
|---|---|---|
| 四則演算（時刻差） | 実施済み（本番） | `src/prc/features.py` |
| ラグ・rolling・diff | 実施済み（本番）。追加の rolling 系は codex 63 候補で測定済み | `docs/codex_fe_results/`（8bdf4f9） |
| Label Encoding | 実施済み（本番、Enum 固定） | `src/prc/train.py:_matrix` |
| Target Encoding | 実施済み（本番 ref_*/arh_*/apt_med）。運航者×空港のずれ TE は棄却済み | `docs/features_log.md` 実験 7 |
| カテゴリ集約（中央値・分位点） | 実施済み（本番） | `train.py:reference_table` |
| カテゴリ×カテゴリ（ネイティブ categorical の空港修飾） | 診断で棄却済み（上限 −0.11 s） | `results/airport-qualified-levels.md` |
| 対数化 | 実施済み（signed-log コピー）。目的変数の log は外部報告で悪化 | `features.py:add_signed_log` |
| 欠損値処理 | 欠損フラグのみ。補完はせず LightGBM に任せる（本番） | `features.py` |
| 外れ値除去・winsorize・クリップ | **意図的に不採用**（RMSE が尾部に支配、実測で悪化） | `docs/README.md` |
| スケーリング・One-Hot | GBDT では不要（単調変換で分割が変わらない / ネイティブ categorical） | — |
| PCA | GBDT に無相関化は不要 | — |
| テキスト（tf-idf/BERT） | 該当列なし | — |
| **Count / Frequency Encoding** | **未実施**（G6「actionable now, not done」） | `plans/00-master-candidate-list.md` G6 |
| **集約値との差（groupby → 元の値との差）** | **未実施** | 本書で確認（grep） |
| **カテゴリ集約の散らばり（IQR）** | **未実施**（ref は P10/P20/中央値/件数のみ） | `train.py:reference_table` |
| **時刻の周期変換（sin/cos）** | **未実施** | 本書で確認（grep） |
| **メタ特徴量（他モデルの予測を特徴量に）** | **未実施**（P10 の CatBoost ブレンドは計画のみ） | `plans/P10-catboost-rival.md` |

## 1. 事前の見込み（外れても書き換えない）

`results/model-a-ceiling.md` S6: 空港・運航者・スタンド・滑走路・目的地・時間帯・機種・NM 遅延帯の
10 分割に対する**オラクル**の OOF セル平均補正でも、model A は **213.127 → 212.86（−0.27 s）**しか動かない。
したがって「セルの関数」でしかない特徴量（頻度エンコーディング、IQR、sin/cos 時刻）は、model A に対して
**−0.27 s 以下**と予想する。Count Encoding は 2026 の分布変化（新スタンドの頻度）を運べる点だけが S6 の外にある。

S6 の外にあるのは次の 2 つ:

- **集約値との差**（`proxy − ref_med` など）は `proxy` との**差**なので、セルの関数ではない。木は 2 列の差を
  軸平行の分割でしか近似できず、model A は**データ量が律速**（S5: 容量を 16 倍にしても 0.25 s、行を倍にすると 5.8 s）。
  構造を列として渡すと推定分散が下がる可能性がある。linear_tree（差を葉で表せる）は model A を +0.85 s
  悪化させた（`investigation_2026-09-16/findings.md` R19）ので、列として明示する方法はまだ試されていない。
- **メタ特徴量**は、NM 列を使わない「taxi の事前予測」モデルを全ラベル行（約 200 万行）で交差学習させ、
  その予測を特徴量として与える。主な狙いは 22,470 行しかない model B への**情報の移転**。
  ただし model B の改善は §6 の測定器でほぼ測れない（未マッチ行だけの CI 幅 385〜724 s）ことを先に認めておく。

## 2. 候補（すべて学習側の fold 内で作る。本番の予測経路 `predict_fold` と同一コード）

実装は `src/prc/train.py` の `add_fold_features`。**特徴量キャッシュ（`features.py`）は変更しない**ので、
base 群は本番レシピ `main_gap` と同じ特徴量行列になる。

### F: 頻度エンコーディング（Zenn: Count Encoding / カテゴリ×カテゴリ）— 6 列、ラベル不使用

学習行 ∪ 予測行（CV では 12 か月、本番では 2025 + 2026 ランキング）で数え、**空港内のシェア**に正規化する
（件数そのものは CV と本番で母数が違うため）。キーに欠損があれば null。

| 列 | キー（すべて ADEP 内） |
|---|---|
| `freq_stand` | STAND |
| `freq_stand_rwy` | STAND × RUNWAY |
| `freq_operator` | AIRCRAFT_OPERATOR |
| `freq_actype` | AIRCRAFT_TYPE |
| `freq_ades` | ADES |
| `freq_rwy_hour` | RUNWAY × 現地時 |

### D: 集約値との差と散らばり（Zenn: groupby 差分 / Aru: カテゴリ集約）— 6 列

| 列 | 定義 | ラベル | lane |
|---|---|---|---|
| `agg_proxy_minus_ref_med` | proxy − ref_med（空港×スタンド×滑走路の中央値） | 学習側で fit 済みの表 | A |
| `agg_proxy_minus_ref_p10` | proxy − ref_p10 | 同上 | A |
| `agg_proxy_minus_arh_med` | proxy − arh_med（空港×滑走路×時間帯の中央値） | 同上 | A |
| `agg_proxy_cell_dev` | proxy − 同じ空港×スタンド×滑走路の proxy 中央値（学習 ∪ 予測行、n ≥ 30） | 不使用 | A |
| `agg_sched_minus_ref_med` | mvt_minus_sched − ref_med | 学習側で fit | A, B |
| `agg_ref_iqr` | 空港×スタンド×滑走路の label の P75 − P25（学習側、0 < y < 7200、n ≥ 30） | 学習側で fit | A, B |

lane A 専用の 4 列は model B に渡さない（proxy が無いので全行 null になる）。

### C: 時刻の周期変換（Aru: 極座標変換）— 2 列

`cyc_sin` / `cyc_cos` = sin / cos(2π × 現地の分 / 1440)、現地の分 = `hour_local × 60 + 離陸時刻の分`。

### M: メタ特徴量（Aru）— 2 列

- `meta_prior`: LightGBM（63 葉、学習率 0.1、300 ラウンド固定、seed 1）で label を予測。入力は
  model B と同じ列集合から**エンコーディング 9 列を除いたもの** ＋ カテゴリ列。エンコーディングを除くのは、
  学習側のエンコーディング表が OOF 対象月のラベルを含むためで、除けば学習側の予測は厳密に OOF になる。
  学習側は**月 mod 3 の 3 分割で交差学習**、予測側は学習側全行で fit したモデルで予測する。
- `meta_prior_minus_proxy` = meta_prior − proxy（lane A のみ）。

### ノイズ対照

`noise_k`（k = 1..6）: `MVT_ID_mvt` のハッシュ（種 20260924 + k）。信号を持ちえない。
列を足すこと自体の効果（`feature_fraction: 0.8` による列抽出の変化、既知 ±0.5 s）を測る。

## 3. 群と判定

**測定条件**: `configs/main_gap.yaml`（本番、v14）＋ `folds: [forward]`。群間の違いは候補フラグ 1 つだけ。

| 群 | 追加列（A / B） | 比較するノイズ群 |
|---|---|---|
| base | 0 / 0 | — |
| noise2 | 2 / 2 | — |
| noise6 | 6 / 6 | — |
| F | 6 / 6 | noise6 |
| D | 6 / 2 | noise6 |
| C | 2 / 2 | noise2 |
| M | 2 / 1 | noise2 |

**段階 1（forward）で合格** = 次の 2 つを両方満たす。

1. `prc.compare base <候補>` が **ADOPT**（全行と K=200 の 95% CI がともに改善側でゼロを除外。既存の規則そのまま）。
2. `prc.compare <対応するノイズ群> <候補>` の **K=200 の 95% CI が改善側でゼロを除外**（列を足しただけの効果ではない）。

**段階 2（jan fold で再現）**: 段階 1 の合格候補だけ、base / 対応ノイズ / 候補を `folds: [jan]` で再実行し、
同じ 2 条件を満たせば**採用候補**とする。満たさなければ棄却（features_log 実験 7・9 の教訓: forward だけの
合格は符号が反転したことがある）。

**本番への反映はこの作業の範囲外**。採用候補が出ても `configs/main_gap.yaml` は変えず、提出もしない。
結果と推奨を記録し、判断はユーザーに委ねる。

## 4. 記録

- 各群の `experiments.jsonl` 行と `prc.compare` の出力を `results.md` に貼る。
- 群ごとに lane A / lane B / 空港別の差分を併記する（プールした数字だけでは、どちらの lane が動いたか分からない）。
