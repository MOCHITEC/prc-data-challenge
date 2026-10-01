# Slot 19 — atm-literature-sota：taxi-out 予測の学術 SOTA

調査者: SubAgent / 2026-09-21
表記規則: **[S]** = 出典から読んだ事実（引用元と URL を併記）／**[X]** = 本エージェントの推定・計算。
**[X] を [S] に見せかけない。**

---

## 0. この調査で最初に確定させたこと

依頼の中心は「報告誤差が研究間で 4 倍違う理由を、**除外条件と空港構成で説明せよ**」だった。
実際に全文を読んだ結果、**4 倍の差の主因は除外条件ではなく、もっと手前の 3 つ**だった。

| 要因 | 内容 | 影響の大きさ |
|---|---|---|
| **A. 目的変数の定義そのものが違う** | 便ごとの AOBT→ATOT か、滑走路行列を除いた区間か、1 時間平均か、additional taxi-out time か | **最大**。最大で 4 倍 |
| **B. 集計単位が違う** | 便単位 RMSE と「1 時間平均の RMSE」「1 日平均の RMSE」が同じ表に並んでいる | **大**。3 倍 |
| **C. 母集団の分散が違う** | RMSE は目的変数の SD に比例する。SD を書かない論文が大半 | **大** |
| D. 除外条件 | 3600 s 超除去・3σ 除去・欠損除去・単一滑走路限定 | 中 |
| E. 分割方法 | 1 か月データのランダム 4:1 分割 vs 時系列分割 | 中 |

**したがって、研究間で比較可能な唯一の量は RMSE ではなく R²（= 1 − MSE/Var(y)）である。**
以降、RMSE を引用するときは必ず (a) 目的変数の定義 (b) 集計単位 (c) 除外 (d) 可能なら SD か R² を併記する。

---

## 1. 比較表（全文を読んで抽出した一次情報）

**RMSE 欄の単位に注意。空港・目的変数・除外が違うものを横に比べてはいけない。**

| # | 研究 | 空港 | 期間 | サンプル | 目的変数の定義 | 分割 | 誤差（単位つき） | モデル | 除外条件 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Lee, Malik, Jung 2016 (AIAA 2016-3910) [S] | **CLT** | 2014 通年（解析）／訓練・試験は 2014/6–8 の特定日 | 通年 265,315 → **246,083**。訓練 1,944–3,361 便、試験 **621–689 便（単日）** | 便単位 AOBT→ATOT | 手選択の単日テスト（南流/北流 × 晴/雨の 4 条件） | **RMSE 4.82–5.53 min**（LR/RF）、6.10–6.83（SVM/NN）。±5 min 内 65–74% | LR, SVM, kNN, RF, NN | out/off 欠損・0 秒以下の taxi time を除去（**通年の 7.25%**） |
| 1b | 同上・ベースライン [S] | CLT | 同 | 同 | 同 | 同 | **Dead Reckoning（= unimpeded 時間そのもの）RMSE 9.68–10.20 min**、±5 min 内 34.8–38.5% | gate×spot×runway の 10 パーセンタイル | 同 |
| 2 | Lee, Coupe, Jung 2019 (AIAA 2019-2933) [S] | **CLT**（ATD-2） | 2018/8 | pushback 20,595 便 | **pushback 工程のみ**（ゲート離脱→taxi 開始） | 8/1–8/20 訓練 → 8/21–8/31 試験 | **RMSE 2.24–2.31 min**、MAE 1.40–1.49。±1 min 内 47–52% | 決定木（中央値）／LR, SVR, Lasso, kNN, RF, NN | pushback/taxi approval 欠損、**30 s 未満・1 h 超を除去** |
| 2b | 同上 [S] | CLT | 2018/8 | 同 | **ramp 区間のみ**（ゲート→spot、滑走路行列を含まない） | 同 | **RMSE 3.52–4.36 min**。定速モデル（6.6 kt）で 4.00 | LR/SVR/Lasso/kNN/RF/NN と定速 | ramp taxi 速度 >30 kt / <1 kt を除去 |
| 3 | Herrema, Curran, Visser, Huet, Lacote 2018, *J. Aerospace Information Systems* 15(3):120–130 [S] | **LFPG** | 記載を取得できず（有料） | 取得できず | 記載を取得できず | 取得できず | **「回帰木で、any given day において平均誤差 1.6 min」** | NN / 回帰木 / 強化学習 / MLP | **記載なし（Abstract・TU Delft 記録とも）** |
| 4 | Diana 2018, *Transp. Res. Part E* 119:149–164 [S] | **KSEA** | 2015/6–8 と 2016/6–8、**07:00–21:59 のみ** | 各 **1,380 = 日×時間**（便単位ではない） | **1 時間平均 taxi-out 時間** | 10-fold CV | **RMSE 1.51–2.21 min（1 時間平均に対して）**。R² 0.37–0.99 | OLS, Ridge, Lasso, ElasticNet, SVR, RF, AdaBoost, Bagging, ExtraTrees, GBR | 夜間時間帯を除外。単一滑走路運用期間を含む |
| 5 | Lordan, Sallan, Valenzuela-Arroyo 2016, *JATM* 56B:118–122 [S] | **LEBL** | 2013/6–8 | 空港の日次運用データ（件数の明記なし） | 便単位 taxi-out | in-sample | **taxi-out: 調整 R² 44.93%、残差 MAE 2.157 min**、75/85/95% 点で 1.87/2.70/4.11 min。（taxi-in: R² 76.2%、MAE 0.475 min） | 線形回帰（stand × 時刻の交互作用） | 不正レコード・重複出発通知の除去のみ |
| 6 | Wang, Brownlee, Woodward, Weiszer, Mahfouf, Chen 2021, *Transp. Res. Part C* 124:102892 [S]（本文 PDF は全リポジトリで 403、Abstract/TLDR/レビュー経由） | **VHHH / EGCC / LSZH** | 取得できず | 取得できず | taxi time（到着・出発両方） | 取得できず | **誤差の数値は取得できていない** | MLP, LR, PR, GBRT, **RF（最良）** | 取得できず |
| 7 | Ravizza, Chen, Atkin, Stewart, Burke 2014, *Applied Soft Computing*（二次出典: Diana 2018 Table 1、Yuan 2024 Table 1）[S] | **ESSA / LSZH** | — | — | taxi time | — | 数値は二次出典に無し。**TSK fuzzy が最良** | MLR, LMS 回帰, SVR, M5 model tree, TSK/Mamdani fuzzy | — |
| 8 | Zbakh et al. 2024, *IJECE* 14(2):2126–2134 [S] | **GMMN（カサブランカ）** | 2019/8 のうち **20 日**、ピーク時間帯 | **340 便** | 便単位 AOBT→ATOT | 85:15 | **RMSE 2.68 min（NN）/ 2.90（SVM）/ 3.03（RT）**、MAE 2.15/2.30/2.43 | NN, SVM, 回帰木 | AOBT または ATOT 欠損便を除去 |
| 9 | Interpretable Taxi-Out ... Stacking + SHAP, *Sci. Rep.* (2026) 10.1038/s41598-026-40898-5 [S] | **ZGSZ 深圳・Runway 15 の出発のみ** | 2019/5/26–6/8（**14 日**） | 12,645 → **5,952 出発便** | 便単位 taxi-out | 10-fold CV | **RMSE 140.5 s、MAE 99.7 s、MAPE 10.6%**。±60 s 41.0% / ±180 s 86.3% / ±300 s 96.5% | Stacking（CatBoost+SVR+KNN+MLP → Ridge） | **taxi-out < 0 s または > 3600 s を除去**、不完全・異常レコード除去、**単一滑走路限定** |
| 10 | Lu et al. 2026 DDA-SIM-ATT, *Aerospace* 13(4):314 [S] | **ZGSZ・Runway 15 のみ（#9 と同一データ）** | 2019/5/26–6/8 | 5,986 → **5,442 便** | 便単位 taxi-out | **7:3 ランダム分割**＋データ拡張 | **RMSE 125.61 s、MAE 87.55 s、MAPE 10.34%**。±120 s 74.57% / ±180 s 89.12% / ±300 s 97.76% | CatBoost ＋ 多モジュール融合 | **3σ 則で異常値除去**、欠損・重複除去 |
| 11 | Yuan, Liu, Chen 2024, *Applied Sciences* 14(21):9968 [S] | **ZSPD 上海浦東** | 2018/2/1–2/28（**1 か月**） | 明記なし（1 か月分） | **additional taxi-out time（実測 − unimpeded）** | **4:1 ランダム分割** | **RMSE 2.73 min（164 s）、MAE 2.06 min、R² 0.77**。±3 min 77% / ±5 min 94%。XGBoost 2.85 / RF 2.95 / SVR 3.19 min | NOA-XGBoost | 明記の除外なし（前処理のみ） |
| 12 | Tang, Zhang, Zhang, Wu 2025, *Aerospace* 12(8):721 [S] | **ZSPD 上海浦東** | 2023/12（**1 か月**） | **20,223 便**（クリーニング後） | 便単位 AOBT→ATOT | ランダム分割 | **RMSE 1.561 min（94 s）、MAE 1.178 min（71 s）**。±1 min 54.41% / ±3 min 94.36% / **±5 min 99.42%** | GBRT ＋ road-level link travel time 特徴 | クリーニング。**本文に「当月の PVG の天候は比較的安定」と明記** |
| 13 | Liu & Itoh 2026, *Aerospace* 13(8):684 [S] | **RJTT 東京羽田** | 訓練 2021–2022 / 試験 2023（**時系列分割**） | 明記なし（2 年分） | **pushback → 滑走路進入点の 200 m 手前**。＝ **滑走路行列を目的変数から除いている** | 時系列分割 | **RMSE 129.55 s、MAE 92.68 s、R² 0.43、MAPE 12.25%** | CatBoost（他に RF/MLP/XGBoost） | 記載なし |
| 14 | Vargo, Tien, Jafari 2022 (AIAA 2022-4026 / arXiv:2111.09139) [S] | **KEWR / KJFK / KLGA / KPHL** | 5 年 | 空港ごと 13 万〜17 万サンプル | **次の 1 時間の平均 taxi-out が閾値を超えるかの二値分類** | 60/20/20 の時系列順分割 | LGA で TPR・TNR とも約 80% | ASDE-X 画像の CNN | — |
| 15 | Balakrishna, Ganesan, Sherry 2008/2010（二次出典: Diana 2018 Table 1）[S] | **KTPA / KDTW / KDCA** | — | — | 便単位 | — | 「平均予測値が実測と一致、標準誤差 1.5 min」／DTW・DCA で平均 RMSE 2.9 min | 近似動的計画法・強化学習 | — |

**レビュー論文**: Yin, Zhang, Ma, Wu, Li, Chen 2024, *Applied Sciences* 14(3):1306,
"Prediction and Analysis of Airport Surface Taxi Time: Classification, Features, and Methodology" [S]
— 過去 20 年の手法を線形回帰／NN／SVM／アンサンブル／ファジィ／シミュレーションに分類した表 3–9 を持つ。
**ただしこのレビューも誤差の数値を横並びにしていない**（特徴量と手法だけを並べている）。これは偶然ではなく、
比較不能だからだと読める。

### 1.1 「4 倍の差」の内訳（実際に読んで判明した）

- **#12（94 s）と #1（4.82 min = 289 s）の差は手法差ではない。**
  #12 は単一空港・単月・安定天候・ランダム分割で ±5 min 内 99.42%（＝裾がほぼ無い）。
  #1 は通年の母集団で SD 8.66 min、単日テスト。
- **#13（129.55 s）が小さいのは滑走路行列を目的変数から外しているから。**
  それでも **R² は 0.43 にすぎない**。つまり行列を外しても大半は説明できていない。
- **#4（1.51 min）は便単位ではなく 1 時間平均。** 平均化で分散が 1/n に落ちている。
- **#3（LFPG, 1.6 min）は "on any given day, an average error of 1.6 min" という表現で、
  便単位 RMSE なのか日次集約の誤差なのかが原論文 Abstract の文言から確定できない [S]。**
  有料のため本文を確認できなかった。**この 1.6 min を便単位 RMSE として我々の 284 s と比較してはいけない。**
  （#4 と同じ「日/時間集約」である可能性が現実的にある。）

---

## 2. 現実的な誤差の下限（依頼 3）

### 2.1 出典つきの「取り除けない分散」に関する記述

1. **Lee, Malik, Jung 2016（CLT）[S]** — 結論部:
   > "only 65-74% departures were found within the +/– 5-minute taxi time error window. It seems that this
   > performance limit on the taxi time prediction stems from the operational complexity and uncertainties at
   > Charlotte airport."

   考察部で列挙された「データに記録されていない」要因:
   管制官による代替 taxi 経路、hardstand / holding area への送り込み、離陸順序変更、交差点離陸、
   TMI 制約（EDCT / Call For Release / Miles-in-Trail）に伴う長時間ゲートホールド。さらに
   > "Most of the uncertainties in airport operations come primarily from human operators and external
   > environment."

2. **Lordan et al. 2016（LEBL）[S]** — taxi-in の R² 76.2% に対し taxi-out は 44.93%。著者はその差を明示的に帰属:
   > "The reason to be of these differences is a factor of uncertainty that cannot be controlled in the taxi-out
   > time model. This factor ... is the waiting time that aircrafts have when there is queue at the entrance of
   > the runway and the seconds they spend on the header waiting for the take-off clearance."

3. **Liu & Itoh 2026（RJTT）[S]** — 滑走路行列を目的変数から除いてなお **R² = 0.43**。
   さらに動的な混雑特徴（前後 5 分の push back 数・到着数・経路共有機数・経路競合機数）を足しても
   **MAE は 2.11–2.57 s、RMSE は 3.42 s しか改善しない**。
   > "dynamic features provide only marginal improvements"

4. **Lee, Coupe, Jung 2019（CLT）[S]** — ramp 区間だけで RMSE 3.52–4.36 min。
   **ML 最良（3.52）と「全機を 6.6 kt の定速と仮定」（4.00）の差はわずか 0.5 min。**
   > "the main contributor to the ramp taxi time prediction is the taxi distance from gate to spot, and that the
   > benefits from considering other factors such as gate conflict, TMIs, and ramp congestion level may be
   > limited in terms of the prediction performance improvement."

5. **Tang et al. 2025（ZSPD）[S]** — 最良モデルでも ±1 min 的中は 54.41% どまり:
   > "although the airport road transport network model cannot completely eliminate the inherent uncertainty of
   > the taxi-out process, it effectively captures the microscopic variations"

6. **ラベル側のノイズ — EUROCONTROL ATXOT Edition 01.00, §4 Error description [S]**:
   > "The time stamps used for the calculation use the format HH:MM:SS ... Nevertheless, in some cases the data
   > provided might be limited to a HH:MM format, so in those cases the precision is lower. The precision in the
   > collection of the AOBT (Actual Off-Block Time) is key in the calculation of the taxi-out times."

   つまり **AOBT の記録精度そのものが空港ごとに違う**ことを、指標の作成者自身が誤差源として明記している。

### 2.2 下限の見積もり [X]

上の [S] を部品として組み立てた、本エージェントの推定である。**出典ではない。**

- CLT の分解 [S] を足し合わせると（独立を仮定）: pushback RMSE 2.24 min と ramp RMSE 3.52 min で
  √(2.24² + 3.52²) = **4.17 min ≒ 250 s**。**これは滑走路行列に入る前の段階だけの値**。[X]
- 我々の母集団の空港内 SD は、`docs/data_profile.md` の空港別 SD から計算すると **514.8 s**。[X]
  （EDDF 329.8 / EDDM 301.8 / EGLL 421.8 / EHAM 318.5 / LEBL 330.9 / LEMD 312.5 / LFPG 453.0 /
  **LIRF 1332.0** / LSZH 385.4 / LTFM 427.6 [S: data_profile.md]）
- resilient-hedgehog の 284.36 s は、空港内平均をベースラインとした **R² ≒ 1 − (284.36/514.8)² = 0.695**。[X]
- 文献の R² と並べると: Lee 2016 CLT が **0.69**（RMSE 4.82 min / SD 8.66 min から逆算 [X]）、
  Yuan 2024 ZSPD が **0.77** [S]、Liu & Itoh 2026 RJTT が **0.43** [S]、Lordan 2016 LEBL が **0.449**（in-sample）[S]。

**結論 [X]**: 便単位・AOBT→ATOT・除外なし・10 空港混在・未来区間という条件で、**R² 0.80 を超えるのは
文献上どこにも例がない**。R² 0.80 は我々の母集団では RMSE 230 s に相当する。R² 0.75 なら 257 s。
**現実的な床は 230–260 s** と見る。我々の 284 s は床から 10–24% 上にいる。
**「あと 4 分の 1 まで縮められる」という余地は文献的に存在しない。**

---

## 3. 繰り返し重要とされる特徴量族（依頼 4）

出典で 2 本以上に現れたものだけを挙げる。

| 族 | 具体的な列 | 出典 |
|---|---|---|
| **A. 幾何** | stand→runway の taxi 距離、旋回角の合計、旋回回数、管制区画の handover 回数、concourse、spot、runway | Wang 2021（3 空港すべてで distance が上位3）[S]／Tang 2025（handover + link travel time が最大の寄与）[S]／Lee 2019（ramp の支配要因は距離）[S]／Ravizza 2014（距離・旋回角）[S]／Lee 2016（concourse, spot, runway, departure fix）[S] |
| **B. 出発側行列** | 自機の AOBT〜ATOT の間に離陸した他機数、departure queue length、**直近 20 分の出発数** | Herrema 2018 の top-10 に「congestion level」「number of departures in the last 20 minutes」[S]／Wang 2021 の `NDepDep`（自機 push back 時点で滑走路へ向かっている他機数）が 3 空港共通の上位 3 [S]／深圳 SHAP [S] |
| **C. 到着側（逆流）** | arrival queue length、自機 taxi 中に着陸した機数、ramp を taxi-in 中の機数 | **深圳 SHAP で arrival queue が寄与 1 位、departure queue が 2 位** [S: PMC13022428]／Clewlow, Simaiakis, Balakrishnan 2010 "Impact of arrivals on departure taxi operations at airports"（レビュー 2024 [42]）[S]／Yuan 2024 の到着流特徴 [S] |
| **D. 直近の実測 taxi 時間** | 直近 15/30/60 分の実測 taxi-out 平均、**同一滑走路**の直近 15 分 taxi-out delay 平均、**同一 departure fix** の直近 15 分平均 | Chatterji & Zheng 2012（レビュー Table 4）[S]／深圳（`30-minute average taxi time` と `pre-pushback 30-minute average taxi time` の 2 本）[S]／Huang 2022（30 分・1 時間スライス平均）[S] |
| **E. 直近の taxi 速度** | 直近交通の平均 taxi **速度**（機数ではなく速度） | **Wang 2021 が「高精度に必要な 4 特徴」の 1 つとして明示: depart/arrival; distance; departure traffic; recent traffic average speed** [S] |
| **F. unimpeded 参照値** | stand×runway の unimpeded taxi-out time、congestion level、saturation level | Herrema 2018 top-10 に unimpeded taxi time [S]／Zbakh 2024 が Relief で選んだ 4 特徴が **congestion level, UTXOT, saturation level, aircraft type** [S]／Yuan 2024 [S] |
| **G. 運航者** | airline / carrier | **深圳 SHAP で airline が寄与 3 位** [S]／Idris 2002（airline/terminal が主要因）[S]／**FAA ASPM の unimpeded taxi time は carrier 単位で定義される** [S] |
| **H. フロー制御・規制状態** | EDCT / APREQ / Call-For-Release / Miles-in-Trail、（欧州なら ATFM regulation / CTOT） | Lee 2016 が「データに無い最大の要因」として名指し [S]／Lee 2019 で APREQ 便の pushback 5.30 min vs 他 4.67 min [S]／Yuan 2024 の題名そのものが "by Fusing Flow Control Information" [S] |
| **I. 滑走路横断・hot spot** | 経路上の滑走路横断有無、hot spot 通過数 | Zhao 2021（レビュー Table 6: "whether or not crossing the runway", "number of passes through the HS"）[S]／Wang, Gupta, Johnson 2024（hot spot 数と taxi 時間の関係）[S] |
| **J. gate conflict** | 次の到着機が同一スポットに来るか | Lee 2019: gate conflict 有りの ramp 5.06 min vs 無し 4.85 min [S] |
| **K. de-icing** | AOBT 後の de-icing の有無 | **ATXOT Ed 01.00 は AOBT 後に de-icing した便を指標から丸ごと除外している** [S]。＝別レジームとして扱うべき、という運用側の判断 |

### 3.1 文献が重視しているのに、競技参加者が使っていないと思われる族

本スロットは文献側の担当なので断定はしない。ただし `src/prc/features.py` / `src/prc/weather.py` の
alias 一覧を見る限り、**以下は我々の特徴量に見当たらない** [S: リポジトリのコード]:

- **A 幾何のすべて**（stand→runway 距離、旋回角、旋回回数、handover 回数、滑走路横断、hot spot）。
  我々は stand と runway を**カテゴリとしてのみ**与えており、**両者を結ぶ距離という連続量を与えていない**。
  文献の全空港で distance は上位 3 に入る [S: Wang 2021, Tang 2025, Lee 2019]。
- **E. 直近交通の平均 taxi 速度**（`dep_tko_prev_30m` などは機数であって速度ではない）。
- **F. unimpeded 参照値**（`stand_rwy_share` は組の**頻度**であって**基準時間**ではない）。
- **D の一部**: 直近 N 分の**実測 taxi-out 時間の平均**。`congestion_vs_usual_30m` は機数比のようで、
  実測所要時間の移動平均ではない可能性が高い（要確認）。`arr_taxi_in_mean` は到着側だけ。
- **G. AIRCRAFT_OPERATOR_flt** はスキーマにあるが、features.py の alias 一覧には現れない。

---

## 4. unimpeded 参照値の作り方（依頼 5）

### 4.1 EUROCONTROL：**チームの [S] は旧版であり、現行版は別物**

| | **Edition 00-04**（チームが参照している版, https://ansperformance.eu/library/pru-tx-out-pi.pdf） | **Edition 01.00 / 2023-03（現行）** https://ansperformance.eu/library/ATXOT_indicator_documentation_mar23.pdf |
|---|---|---|
| グルーピング | departure runway × stand。**最大空港では stand group**（近接スタンドの仮想グループ） | departure runway × **stand（個別）**。stand group は使わない。aircraft class は「サンプルが細るので使わない」と明記 |
| 参照値の統計量 | **非混雑便の中央値（median）** | **全便の 10th percentile（P10）** |
| 「非混雑」の定義 | congestion level ≤ **congestion limit(0.5 or 0.6) × saturation level**。saturation level は arrival peak throughput(90 パーセンタイル)などから推定 | **そういう機構は無い**。P10 が「optimal while possible」の近似 |
| 最小サンプル | 非混雑便が **10 便以上** | **taxi-out ≤ P10 の便が 10 便以上**（＝実質 100 便前後） |
| 参照期間 | 参照サンプル | **直近 12 か月のローリング**、毎月再計算 |
| 除外 | taxi-out **> 5 時間（300 min）**、ヘリコプター、**夜間便**、欠損 | taxi-out **> 120 min**、ヘリコプター、**AOBT 後に de-icing した便**、runway/stand/off-block 欠損 |
| 参照値が作れない組 | — | **その組の便は指標から丸ごと落とす**（=「valid reference なし」） |

> 原文 [S, Ed 01.00 §3.5]:
> "The reference time for that combo STAND—ARWY is calculated as the 10th percentile of those taxi-out times ...
> The 10th percentile is estimated to provide the best approach to the optimal (while possible) transit times"
> "there must be at least 10 flights in the sample with a taxi-out time equal or shorter than the calculated
> reference time"

> 原文 [S, Ed 01.00 §3.4 step 1]:
> "filter out the flights with some missing data (e.g. no runway information, no stand information or no
> off-block time), flights with taxi-out time longer than 120 min, helicopters and flights with de-icing after
> the off-block time"

Ed 01.00 §5 は、**taxi 経路・taxi 速度・工事等の特殊事象を「データに無いので考慮しない」と明記**している [S]。
つまり EUROCONTROL 自身が、参照値の未説明分散の所在を把握したうえで放置している。

### 4.2 FAA（ASPM / APO）

- 定義 [S: https://www.aspm.faa.gov/aspmhelp/index/ASPM_Taxi_Times__Definitions_of_Variables.html]:
  "Unimpeded Taxi Out Time is the estimated Taxi Out Time for an aircraft by carrier under optimal operating
  conditions ... estimated by calendar year and season for each carrier and airport reporting OOOI data."
- 手法: **出発キュー数・到着キュー数を説明変数にした線形回帰の切片**（キュー = 0 のときの値）。
  レビュー 2024 [S] の記述:
  > "The constant term in the regression model represents the reference time; i.e., when an aircraft is pushed
  > back from a stand without encountering any queues of aircraft either waiting to take off or taxi in."
- **粒度が carrier × airport × season × year で、stand も runway も入っていない**のが既知の弱点
  [S: Using ASDE-X Surveillance for Taxi-Out Time Benchmarking — "fails to include both start taxi-out location
  (gate/terminal) and wheels-off location (runway)"]。

**EUROCONTROL と FAA の一番大きな違い**: **EUROCONTROL は stand×runway でグループを切り、FAA は carrier で切る。**
レビュー 2024 はさらに「PRU 方式は taxi 距離を参照値の推定に織り込むが、FAA APO 方式は織り込まない」と整理している [S]。

### 4.3 unimpeded を「入力として」使うのか「目的のオフセット」か「ベンチマーク」か

**3 通りすべてが文献に存在する。**

| 使い方 | 事例 | 結果 |
|---|---|---|
| **ベンチマークのみ** | Lee, Malik, Jung 2016 の Dead Reckoning [S]。gate×spot×runway の 10 パーセンタイルをそのまま予測値とする | **RMSE 9.68–10.20 min**。ML(4.8–5.5) の約 2 倍。平均が実測より約 8 min 短い |
| **モデルの入力特徴として** | **Zbakh et al. 2024（カサブランカ）[S]**。Relief による特徴選択の結果が「congestion level / UTXOT / saturation level / aircraft type」の 4 本。**EUROCONTROL PRU の構成要素をそのまま特徴量にしている**／Herrema 2018 の top-10 にも unimpeded taxi time [S] | NN で RMSE 2.68 min（ただし 340 便） |
| **目的変数のオフセットとして** | **Yuan, Liu, Chen 2024（ZSPD）[S]**。ODP（stand group × runway）ごとに到着流・出発流の OLS で UTT を求め、`ATT = 実測 − UTT` を目的変数にする | NOA-XGBoost で ATT の RMSE 2.73 min、**R² 0.77** |
| 同上 | **深圳 Stacking + SHAP（Sci. Rep. 2026）[S]**。"decomposing taxi-out time into unimpeded taxi-out time and dynamic taxi-out time"。UTOT は出発キュー=到着キュー=0 のときの OLS 切片 | RMSE 140.5 s |

**注意 [X]**: 目的オフセット方式は、オフセットがグループごとの**定数**である限り、
総 taxi-out 時間に対する RMSE と数学的に同一である（定数を引いても誤差は変わらない）。
**利得があるとすれば、木モデルが stand×runway のカテゴリ交互作用を自力で学習しなくて済む点だけ**である。
したがって「ATT を目的にする」よりも「**UTT を特徴量として渡す**」方が、失うものが無い。

---

## 5. 対象 10 空港について（依頼 6）

**確認**: 10 番目の空港は **LEBL（バルセロナ）**[S: `src/prc/config.py` の `TARGET_AIRPORTS`]。

### 5.1 個別空港を扱った査読研究

| 空港 | 研究 | 内容 |
|---|---|---|
| **LSZH** | Ravizza, Chen, Atkin, Stewart, Burke 2014（ESSA + LSZH）[S]／Wang et al. 2021（VHHH + EGCC + LSZH）[S] | Ravizza は taxi 距離・旋回角・出発/到着・機数を説明変数に置き、TSK ファジィが最良。Wang 2021 は 3 空港共通で `depArr`, `distance`, `NDepDep` が最重要、加えて「recent traffic average speed」 |
| **LEBL** | Lordan, Sallan, Valenzuela-Arroyo 2016 [S] | **stand 単位 × 時刻の交互作用**が唯一効く構成。terminal 単位ではだめ（terminal は stand の線形結合で多重共線）。aircraft・airline・weekday はいずれも 4% 未満の説明力。taxi-out 調整 R² 44.93% |
| **LFPG** | Herrema, Curran, Visser, Huet, Lacote 2018 [S] | 42 特徴から回帰木。top-10 に unimpeded taxi time / congestion level / 直近 20 分の出発数 |
| **EGLL** | Atkin, De Maere, Burke, Greenwood "Addressing the pushback time allocation problem at Heathrow"（Zbakh 2024 の参考文献 [26] として確認）[S] | Heathrow は holding point での待機を pushback 時刻の割当で制御する研究対象になってきた |
| EHAM / EDDF / EDDM / LEMD / LTFM / **LIRF** | **見当たらなかった**。Crossref の網羅検索（"taxi time prediction Heathrow Schiphol Frankfurt Munich Madrid airport" ほか）でも便単位 taxi-out 予測の査読論文は出てこない [S: 検索結果] | |

### 5.2 EUROCONTROL PRU の実測値（10 空港の構造差を最も直接に示す）

出典 [S]: `https://ansperformance.eu/dashboard/stakeholder/airport/db/<ICAO>.html`（2026-09-21 取得）。
数値は各ダッシュボードの "Annual Taxi-Out Time (Source: APDF)" 図の埋め込みデータから抽出した。

**年次（暦年 2026 の値）**

| 空港 | 平均 taxi-out [min/dep] | additional taxi-out [min/dep] | **additional の割合** |
|---|---|---|---|
| **LIRF** | 12.1 | **6.99** | **58%** |
| LSZH | 8.7 | 3.30 | 38% |
| EGLL | **16.2** | 5.79 | 36% |
| EHAM | 9.9 | 3.52 | 36% |
| LTFM | 12.3 | 4.45 | 36% |
| LFPG | 11.8 | 3.72 | 31% |
| EDDF | 10.9 | 3.33 | 31% |
| EDDM | 9.7 | 3.00 | 31% |
| LEBL | 11.9 | 3.75 | 31% |
| LEMD | 13.3 | 3.69 | 28% |

**LIRF の additional 比率の推移** [S]: 2021 48% → 2022 53% → 2023 57% → **2024 64%** → 2025 55% → 2026 58%。
他の 9 空港はいずれも 27–48% のレンジに収まり、40% を超えるのは EGLL(2023 48%) だけ。

滑走路別（12 か月、summary tile 値は LIRF 7.12 min/dep）[S]:
LIRF **16R 総 16.01 / add 8.52**、**25 総 12.02 / add 6.59**、**34L 総 15.23 / add 8.08**。

**読み方**: LIRF は「taxi が長い空港」ではない（EGLL 16.2、LEMD 13.3、LTFM 12.3 の方が長いか同等）。
LIRF は **「stand×runway の幾何では説明できない部分が構造的に突出して大きい空港」** である。
additional taxi-out time は定義上まさに「stand×runway 参照値からの超過」なので、
**この比率は「幾何特徴でどこまで説明できるか」の上限を直接に示す指標として使える。**

---

## 6. LIRF について（依頼 7）

### 6.1 査読文献

- **便単位 taxi-out 予測を LIRF で行った査読論文は見つからなかった。**
  Crossref を "Fiumicino airport departure operations taxi runway capacity" / "Rome Fiumicino A-CDM ..." で
  検索した結果に該当なし [S]。
- LIRF を対象にした運用系の文献として確認できたのは
  **"A Simulation-Based Architecture for Supporting Strategic and Tactical Decisions in the Apron of
  Rome-Fiumicino" (Winter Simulation Conference 2005, DOI 10.1109/wsc.2005.1574428)** のみ。**非 OA** [S: Unpaywall]。
  題名が示す通り **apron（エプロン）がボトルネックとして研究対象になっている**。

### 6.2 LIRF がなぜ我々の誤差を支配するのか — 2 つの別々の理由

**理由 1：運用（文献で説明がつく側）**
PRU の実測で LIRF の taxi-out の **58%** が stand×runway 参照値からの超過である [S]。
Lordan 2016 が LEBL で「taxi-out の R² が taxi-in より 31 ポイント低いのは滑走路行列と離陸許可待ちのせい」と
書いた通り [S]、超過分は幾何から予測できない成分そのものである。
さらに LIRF は PRU が把握している出発滑走路が **3 本しかない**（16R / 25 / 34L）[S] のに対し
EHAM は 7–8 本、LTFM は 5–7 本、LEBL は 7 本。**LIRF では RUNWAY 列の情報量が構造的に小さい。**

**理由 2：報告・計測（文献では説明できず、データ品質の問題）**

`docs/data_profile.md` の実測 [S]:

| | LIRF | 他 9 空港のレンジ |
|---|---|---|
| taxi-out の SD | **1,332.0 s** | 301.8 – 453.0 s |
| 中央値 | 1,025 s（17.1 min） | 711 – 1,319 s |
| p99 | **4,019 s** | 1,707 – 2,701 s |
| `gt_2h_rate` | **0.0030** | 0.0000 – 0.0002 |
| `gt_4h_rate` | **0.0007** | 0.0000 |
| `block_eq_sched_rate` | **0.0448** | 0.0018 – 0.0143 |
| `med_block_minus_aobt3` | **−118 s**（最も負） | −54 〜 +296 s |

- **LIRF は 7.7% の行で、空港内平方和の 51.6% を占める** [X: 上表の SD と行数から計算]。
- **PRU の LIRF 平均 taxi-out は 12.1 min なのに、競技データの LIRF 中央値は 17.1 min** [S: 両者]。
  **5 分のギャップがある。**同じ空港の同じ現象を測っているはずの 2 つの系列が一致しない。
- `block_eq_sched_rate` 4.48%（他空港の 3〜25 倍）は、**BLOCK_TIME がスケジュール値で埋められている便が
  LIRF に集中している**ことを意味する。これは運用ではなく報告の問題である。
- **EUROCONTROL 自身が、この種の便を指標から除外している**:
  taxi-out > 120 min を除外、AOBT 後 de-icing を除外、AOBT の記録精度を誤差源として明記 [S: Ed 01.00 §3.4, §4]。
  **PRU の 12.1 min と競技データの 17.1 min の差は、PRU がこれらを落としていることで大部分説明がつく** [X]。

**結論 [X]**: LIRF の誤差寄与は **「運用上の行列（文献的に予測困難）」と「報告レジームの混在（文献的には
指標から除外される対象）」の 2 層**でできている。後者は運用特徴量では原理的に予測できない。
**後者をレジーム変数として明示的にモデル化しない限り、LIRF の誤差は下がらない。**

---

## 7. 手法・検証設計について文献から言えること

- **モデル族の差は小さい。** Lee 2016（CLT）で LR と RF がほぼ同点、SVM/NN が劣る [S]。
  Lee 2019（CLT ramp）で LR/Lasso/kNN/RF/NN の RMSE が 3.52–3.80 に収まり、**定速モデルが 4.00** [S]。
  Diana 2018（SEA）は「No algorithm fits data better in all cases」と結論 [S]。
  Yuan 2024 の NOA-XGBoost と素の XGBoost の差は RMSE 2.73 vs 2.85 min（4%）[S]。
  → **手法の探索から取れる利得は、文献上どこでも 5–15% 程度が上限。**
- **時系列分割を使っているのは Liu & Itoh 2026（訓練 2021–22 / 試験 2023）[S] と
  Lee 2019（8/1–8/20 訓練 → 8/21–8/31 試験）[S]、Vargo 2022（順序分割）[S] だけ。**
  RMSE が小さい研究（#9 #10 #11 #12）は**すべて 1 か月以内のデータのランダム分割**である。
  **これらの数値を、未来区間で採点される我々のスコアと比べてはいけない。**
- **Diana 2018 の learning curve の観察** [S]:
  > "Adding more training samples may not necessarily increase generalization."

---

## Applicable to resilient-hedgehog

文献が「やるべき」と言っていて、我々がやっていない（ように見える）こと。上から効きそうな順。

1. **stand×runway の unimpeded 参照時間を特徴量として渡す。**
   - 文献で最も繰り返される構成。Herrema 2018 の LFPG top-10 の 1 つ [S]、Zbakh 2024 で Relief が選んだ 4 本の
     1 つ [S]、Yuan 2024・深圳 Stacking は目的分解の土台 [S]。
   - 具体的には **EUROCONTROL Ed 01.00 の作り方をそのまま移植する**:
     (stand, runway) ごとに、**taxi-out の 10th percentile**、ただし **P10 以下の便が 10 便以上ある組だけ有効**、
     参照サンプルは**直近 12 か月ローリング**（我々の場合は訓練期間の過去側のみ＝リークしない形で）。
     有効でない組には空港×runway の P10 を fallback。
   - **注意**: `stand_rwy_share` は「組の頻度」であって参照時間ではない。別物。
   - 旧版（median × saturation level）ではなく **現行版（P10）**を使うべき。チームが [S] として持っている
     「median、10 便以上、300 分超除外」は **Edition 00-04 の記述で、2023 年 3 月に置き換わっている** [S]。

2. **stand→runway の taxi 距離（連続量）を入れる。**
   - Wang 2021 が 3 空港（VHHH/EGCC/LSZH）すべてで distance を上位 3 に挙げ [S]、
     Lee 2019 は ramp の支配要因が距離だと結論し [S]、Tang 2025 の最大の利得も経路由来の特徴 [S]。
   - 我々は stand も runway もカテゴリとしてしか与えていない。木は 1,899 × 53 の交互作用を
     サンプルから学ばねばならない。**距離を 1 本与えれば、未知 stand にも外挿が効く**
     （`STAND_mvt` は ranking で 1,698 levels、被覆 99.92% なので未知 stand は少ないが、
     稀 stand の学習サンプル不足は残る）。
   - 座標が無くても、**(stand, runway) ごとの P10 taxi-out 時間そのものが距離の代理になる**（＝項目 1 と同じ列）。

3. **LIRF の「報告レジーム」を明示的に分離する。**
   - EUROCONTROL は同種の便を指標から**除外**している [S: taxi-out > 120 min、AOBT 後 de-icing、AOBT 精度]。
     我々は採点対象なので除外できないが、**「除外されるべき便」であることをモデルに教えることはできる**。
   - 使える手掛かりはすでにデータにある [S: data_profile.md]:
     `block_eq_sched_rate`（LIRF 4.48% vs 他 0.18–1.43%）、`med_block_minus_aobt3` が LIRF だけ −118 s、
     `nm_unmatched_rate`。**「BLOCK == SCHED」フラグ、`mvt_minus_aobt3` の空港内 z-score** などを
     レジーム指標として入れる。
   - LIRF が 7.7% の行で平方和の 51.6% を占める [X] 以上、**ここが唯一の大きな余地**である。

4. **直近 N 分の「実測 taxi-out 時間の平均」を入れる（機数ではなく所要時間）。**
   - 深圳の特徴表は `30-minute average taxi time` と `pre-pushback 30-minute average taxi time` の**2 本**を持つ [S]。
     Chatterji & Zheng 2012 は「直近 15 分の taxi-out delay 平均」を**同一滑走路別・同一 departure fix 別**に
     分けて持つ [S]。
   - 我々は `dep_tko_prev_30m` などの**機数**と `arr_taxi_in_mean`（到着側）を持つが、
     **出発側の実測所要時間の移動平均**が見当たらない。ラベルのリークに見えるが、
     **自機より前に離陸済みの便だけを使えばリークしない**（ATOT 時点で確定している情報）。

5. **到着側キューを出発側と同等に扱う。**
   - **深圳の SHAP は arrival queue を寄与 1 位、departure queue を 2 位にした** [S: PMC13022428]。
     Clewlow/Simaiakis/Balakrishnan 2010 の題名も "Impact of arrivals on departure taxi operations" [S]。
   - 我々は `arr_land_prev_30m`, `dep_arr_ratio_30m` を持つが、**「自機の taxi 中に着陸した機数」**
     （AOBT < ALDT < ATOT を満たす便）という自機区間に同期した形にはなっていないように見える。
     Yuan 2024 はこの条件付きカウントを 8 通りに細分している [S]。

6. **AIRCRAFT_OPERATOR_flt を使う。**
   - 深圳 SHAP で airline が寄与 3 位 [S]。Idris 2002 は airline/terminal を主要因に挙げる [S]。
     **FAA ASPM の unimpeded taxi time は carrier 単位で定義されている** [S]。
   - スキーマにはあるが features.py の alias 一覧に無い。676 levels、ranking 被覆 98.41%。
     カテゴリのままでなくとも、**operator×airport の過去 taxi-out の P10 / 中央値**として渡せる。

7. **直近交通の平均 taxi「速度」。**
   - Wang 2021 が「高精度に必要な 4 特徴」に明示的に挙げている唯一の非自明な量 [S]。
   - 距離が無いと速度は作れないので、項目 2 とセット。
     代理として **「直近 30 分の (実測 taxi-out / その組の P10)」の平均**＝混雑倍率が作れる [X]。

8. **やっても無駄だと文献が言っていること（探索の節約）。**
   - **モデル族の探索**。LR と RF がほぼ同点、定速モデルと ML の差が 0.5 min [S: Lee 2016, Lee 2019]。
     「No algorithm fits data better in all cases」[S: Diana 2018]。
   - **動的な混雑カウント特徴をさらに増やすこと**。Liu & Itoh 2026 が静的特徴のみのモデルに
     7 種類の動的混雑特徴を足して、**MAE 2.11–2.57 s、RMSE 3.42 s しか改善しなかった** [S]。
     これは `docs/features_log.md` の #1 `queue_ahead`・#2 `bank_position` が
     「測定限界以下」に終わったのと同じ現象である。
   - **目的変数を ATT（実測 − unimpeded）に置き換えること**。オフセットがグループごとの定数なら
     RMSE は数学的に不変 [X]。**UTT は目的から引くのではなく、特徴として渡すべき。**

9. **期待値の校正。**
   - 我々の 284.36 s は、空港内平均ベースラインに対する R² ≒ **0.695** [X]。
     これは Lee 2016 の CLT（R² ≒ 0.69 [X]）と同水準、Yuan 2024 の ZSPD（0.77 [S]）より下、
     Liu & Itoh の RJTT（0.43 [S]）より上。
   - **便単位・除外なし・多空港・未来区間で R² 0.80 を超えた例は文献に無い** [S: 本調査の範囲]。
     R² 0.80 は我々の母集団で RMSE 230 s。**現実的な床は 230–260 s** [X]。
   - **したがって残り約 30–50 s のうち、約半分は LIRF の報告レジームにある** [X]。

---

## 参考文献（本文で参照したもの・URL つき）

- Lee, H., Malik, W., Jung, Y. C. (2016). *Taxi-Out Time Prediction for Departures at Charlotte Airport Using
  Machine Learning Techniques.* AIAA 2016-3910. https://ntrs.nasa.gov/api/citations/20180002162/downloads/20180002162.pdf
- Lee, H., Coupe, W. J., Jung, Y. C. (2019). *Prediction of Pushback Times and Ramp Taxi Times for Departures at
  Charlotte Airport.* AIAA 2019-2933. https://ntrs.nasa.gov/api/citations/20190027660/downloads/20190027660.pdf
- Herrema, F., Curran, R., Visser, H., Huet, D., Lacote, R. (2018). *Taxi-Out Time Prediction Model at Charles de
  Gaulle Airport.* J. Aerospace Information Systems 15(3):120–130. https://doi.org/10.2514/1.I010502
  （本文非 OA。記録: https://research.tudelft.nl/en/publications/taxi-out-time-prediction-model-at-charles-de-gaulle-airport/ ）
- Diana, T. (2018). *Can machines learn how to forecast taxi-out time? A comparison of predictive models applied
  to the case of Seattle/Tacoma International Airport.* Transp. Res. Part E 119:149–164.
  https://mdsoar.org/handle/11603/22980
- Lordan, O., Sallan, J. M., Valenzuela-Arroyo, M. (2016). *Forecasting of taxi times: The case of Barcelona-El
  Prat airport.* JATM 56B:118–122. https://hdl.handle.net/2117/91080
- Wang, X., Brownlee, A. E. I., Woodward, J. R., Weiszer, M., Mahfouf, M., Chen, J. (2021). *Aircraft taxi time
  prediction: Feature importance and their implications.* Transp. Res. Part C 124:102892.
  https://doi.org/10.1016/j.trc.2020.102892
- Ravizza, S., Chen, J., Atkin, J. A. D., Stewart, P., Burke, E. K. (2014). *Aircraft taxi time prediction:
  Comparisons and insights.* Applied Soft Computing. https://doi.org/10.1016/j.asoc.2013.10.017
- Zbakh, D. et al. (2024). *Taxi-out time prediction at Mohammed V Casablanca Airport.* IJECE 14(2):2126–2134.
  https://ijece.iaescore.com/index.php/IJECE/article/download/34030/17287
- *Interpretable Taxi-Out Time Prediction of Departure Flights Using Stacking Ensemble Learning and SHAP
  Analysis.* Scientific Reports (2026). https://doi.org/10.1038/s41598-026-40898-5 ／
  https://pmc.ncbi.nlm.nih.gov/articles/PMC13022428/
- Lu, Y. et al. (2026). *DDA-SIM-ATT: A Synergistic Multi-Module Fusion Model for High-Precision Prediction of
  Departure Flight Taxi-Out Time.* Aerospace 13(4):314. https://doi.org/10.3390/aerospace13040314
- Yuan, L., Liu, J., Chen, H. (2024). *Additional Taxi-Out Time Prediction for Flights at Busy Airports by Fusing
  Flow Control Information.* Applied Sciences 14(21):9968. https://doi.org/10.3390/app14219968
- Tang, X., Zhang, W., Zhang, S., Wu, C.-L. (2025). *A Road-Level Transport Network Model with Microscopic
  Operational Features for Aircraft Taxi-Out Time Prediction.* Aerospace 12(8):721.
  https://doi.org/10.3390/aerospace12080721
- Liu, C., Itoh, E. (2026). *Stochastic Departure Metering with Machine-Learning Taxi-Time Predictors.*
  Aerospace 13(8):684. https://doi.org/10.3390/aerospace13080684
- Vargo, E., Tien, A., Jafari, A. (2022). *Airport Taxi Time Prediction and Alerting: A Convolutional Neural
  Network Approach.* AIAA 2022-4026. https://arxiv.org/pdf/2111.09139
- Yin, J., Zhang, M., Ma, Y., Wu, W., Li, H., Chen, P. (2024). *Prediction and Analysis of Airport Surface Taxi
  Time: Classification, Features, and Methodology.* Applied Sciences 14(3):1306.
  https://doi.org/10.3390/app14031306
- EUROCONTROL PRU. *Additional taxi-out time performance indicator document — Edition 01.00 (Mar 2023).*
  https://ansperformance.eu/library/ATXOT_indicator_documentation_mar23.pdf
- EUROCONTROL PRU. *Additional Taxi-Out Time Performance Indicator document — Edition 00-04.*
  https://ansperformance.eu/library/pru-tx-out-pi.pdf
- EUROCONTROL PRU 空港ダッシュボード（10 空港）。
  https://ansperformance.eu/dashboard/stakeholder/airport/db/LIRF.html ほか（ICAO を置換）
- FAA ASPM. *Taxi Times: Definitions of Variables.*
  https://www.aspm.faa.gov/aspmhelp/index/ASPM_Taxi_Times__Definitions_of_Variables.html
- Zhang, Y., Wang, Q. (2017). *Methods for determining unimpeded aircraft taxiing time and evaluating airport
  taxiing performance.* Chinese J. Aeronautics. https://doi.org/10.1016/j.cja.2017.01.002（本文 403、レビュー経由）
- Wang, J., Gupta, S., Johnson, M. E. (2024). *Comparison of taxi-in and taxi-out times by airport hub
  classification and number of hot spots.* Collegiate Aviation Review International 42(1):75–96.
  https://ojs.library.okstate.edu/osu/index.php/CARI/article/download/9678/8600/20873

### 取得できなかったもの（正直に記録する）

- Herrema 2018（LFPG）の本文 — AIAA 有料、TU Delft リポジトリも 403。
  **「1.6 min」が便単位 RMSE か日次集約かを確定できていない。**
- Wang et al. 2021（LSZH 含む）の本文 — Elsevier / Stirling / QMUL いずれも 403。誤差の数値は未取得。
- Ravizza et al. 2014（LSZH 含む）の本文 — 二次出典のみ。
- MDPI 本体・ScienceDirect は curl / WebFetch とも 403。MDPI は `mdpi-res.com/d_attachment/...` 経由で取得した。
- 本セッションの WebSearch 枠（200 回）は他スロットと共有で途中で枯渇したため、
  後半は Crossref API / Unpaywall API / Semantic Scholar API による網羅検索に切り替えている。
