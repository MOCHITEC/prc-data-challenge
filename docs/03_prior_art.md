# 03. 先行研究・過去回調査

最終更新: 2026-09-09。4 方向の並列調査のうち **文献** と **過去回（2024/2025）** の 2 方向が完了。**空港運用知識** と **コンペ戦術** の 2 方向は実行環境の利用上限で未完了（追加調査予定、§7）。各主張には調査エージェントが実際に開いた出典 URL を付ける。独立検証（第 2 エージェントによる出典再確認）は未実施のため、本書の主張は「出典あり・未再検証」。自分で一次確認した項目のみ「確認済」と表記。

---

## 1. 目的変数の定義と公式指標（EUROCONTROL PRU）

- 欧州の実務・研究は一貫して taxi-out = ATOT − AOBT（離陸 − オフブロック）。本コンペの `TAXITIME_SEC_mvt = MVT_TIME − BLOCK_TIME` と同一。
- **PRU の Additional taxi-out time 指標（2023 年版）**: 参照値（unimpeded）は「スタンド × 出発滑走路」の組合せごとに直近 12 か月の taxi-out の **10 パーセンタイル**（その組合せで P10 以下の便が 10 便以上ある場合のみ有効）。除外: taxi-out > 120 分、スタンド/滑走路/AOBT 欠損、オフブロック後の防除氷、ヘリコプター。機種はグループ化に使わない（サンプル数のため）。AOBT の精度（分単位か秒単位か）が主要誤差源と明記。出典: https://ansperformance.eu/library/ATXOT_indicator_documentation_mar23.pdf
- **旧版 PRU 手法（Ed. 00-04）**: 便ごとの「混雑度」（その便のタキシー中に離陸・着陸した機数）と「飽和度」（滑走路–スタンド群ごとの 20 パーセンタイル taxi-out × 直前 20 分の時間当たりスループットの 90 パーセンタイル）を比較して unimpeded 便を判定し、その中央値を参照値とする。混雑限界 0.6（大規模ハブ）/ 0.5。06:00–22:00 のみ。→ **混雑度・20 分前スループット・飽和比は特徴量としてそのまま再利用可能**。出典: https://ansperformance.eu/library/pru-tx-out-pi.pdf
- SES 性能ポータルの定義も同じ（空港 × 滑走路 × スタンドごとの unimpeded）。出典: https://www.sesperformance.eu/dataportal/metadata/additional-taxi-out-time/
- **確認済**: PRU は対象空港別の月次 Additional taxi-out time を Excel で公開している（`https://ansperformance.eu/data/` の `Taxi-Out_Additional_Time.xlsx`）。テスト期間（2026 年 1 月）を含むため利用方針は docs/01 U12 のとおり「テスト期間の値は使わない」。

## 2. 空港別の taxi-out 分布（CODA、IATA Summer 2025。出典あり・未再検証）

| ICAO | 平均 (分) | SD (分) | P10 | 中央値 | P90 |
|---|---|---|---|---|---|
| EDDF | 13.74 | 5.38 | 7 | 13 | 20 |
| EDDM | 13.68 | 4.31 | 9 | 13 | 18 |
| EGLL | 20.59 | 7.82 | 13 | 19 | 29 |
| EHAM | 13.85 | 5.18 | 9 | 13 | 20 |
| LEBL | 18.62 | 6.96 | 11 | 18 | 25 |
| LEMD | 18.67 | 5.87 | 13 | 18 | 25 |
| LFPG | 17.10 | 5.91 | 11 | 16 | 24 |
| LIRF | 17.95 | 7.24 | 11 | 16 | 27 |
| LSZH | 13.29 | 4.91 | 9 | 13 | 18 |
| LTFM | 18.00 | 6.24 | 12 | 17 | 26 |

出典: https://www.eurocontrol.int/publication/taxi-times-summer-2025 （航空会社報告の AOBT/ATOT ベース。夏季のみ）。
含意: 空港内 SD は 259〜469 秒。**現リーダーボード首位（246 秒）は全空港の空港内 SD を既に下回っている**。空港平均を返すだけのベースラインは 300 秒台後半〜400 秒台になる見込み。CODA の外れ値除外は taxi-out > 180 分（PRU は 120 分）。出典: https://ansperformance.eu/reference/dataset/planning-taxi-times/

## 3. 文献から分かる有効特徴量（重要度順）

1. **実際に経験した離陸待ち行列**（最重要、ほぼ全研究で一致）
   - Idris 2002（Boston）: 「その便のプッシュバックから離陸までの間に離陸した機数」が最良。プッシュバック時点の地上機数 N は passing のため弱い（R² 0.19）。出典: https://pdfs.semanticscholar.org/cdcb/8f9380999147e162fbb4a67b9cfc889e8bc4.pdf
   - Simaiakis & Balakrishnan 2014（EWR/BOS）: taxi-out は「有効交通量」（プッシュバック時にタキシー中の機数 + 滑走路到達前にプッシュバックした機数）の凸非減少関数。出典: https://www.mit.edu/~hamsa/pubs/SimaiakisBalakrishnan_TS2014.pdf
   - 上海浦東 2024: D2 =「自便より後にプッシュバックし自便より先に離陸した他便数」が r = 0.81、A2 =「自便オフブロック後に着陸し離陸前にインブロックした到着便数」が r = 0.74。FAA 式の単純カウントは r = 0.17/0.05。**A-CDM の計算時刻（実績でなく計画値）に置き換えると MAE 2.06 → 3.94 分、RMSE 2.73 → 5.62 分に劣化** — Ranking で他便の実オフブロックが無い本コンペの状況に近い定量値。出典: https://mdpi-res.com/d_attachment/applsci/applsci-14-09968/article_deploy/applsci-14-09968.pdf
   - 深圳 2026（Sci. Rep.）: SHAP で到着待ち行列 ≈18% + 出発待ち行列 ≈18%、航空会社 ≈9%、直前 30 分平均 taxi ≈7%、距離 6%。出典: https://pmc.ncbi.nlm.nih.gov/articles/PMC13022428/
2. **スタンド × 滑走路の unimpeded 参照値**（PRU の P10、Lee 2016 のゲート/スポット/滑走路別 P10）。GBDT で「参照値 + 残差（additional）」に分解する構成が PRU 実務・上位研究と一致。
3. **固定窓の離陸・着陸数**（オフブロック前 20〜30 分、プッシュバック前後 ±5 分）、**滑走路 / 滑走路運用モード**、**航空会社**、**機種・後方乱気流区分**、**時間帯**、**直近 15〜30 分の平均 taxi-out**。
4. **気象**（風・降水・視程）は二次的だが尾部に効く。Herrema 2018（CDG）の上位特徴量に **防除氷ステーション**・出発スタンド・滑走路方位・参照時間・月・実離陸時刻。出典: https://research.tudelft.nl/en/publications/taxi-out-time-prediction-model-at-charles-de-gaulle-airport/
5. 航空会社と機種はスタンドと強く共線（Lordan 2016, LEBL: スタンド × 滑走路 + 3 時間帯だけで説明でき、航空会社・機種・ターミナル・滑走路構成の追加は < 4%。taxi-out の R²adj は 44.9% で、残差は滑走路手前の待ち行列）。出典: https://upcommons.upc.edu/bitstreams/371d032d-5e63-49bc-b5c7-1366b36f7d52/download
6. Wang et al. 2021（Manchester・Zurich・Hong Kong）: 出発/到着、タキシー距離、旋回数、平均速度、「プッシュバック時に滑走路へ向かっている他機数」の小さな特徴量集合でほぼ全精度が出る。出典: https://www.stir.ac.uk/research/hub/publication/1683284
7. 遅延（AOBT − 予定/EOBT）は混雑空港で有意（Lian 2018, 北京）。本データでは `SCHED_TIME`, `EOBT_1`, `LOBT`, `IOBT` から計算可能。出典: https://api.crossref.org/works/10.1155/2018/7509508

## 4. モデルと誤差水準

- 木系アンサンブル（RF, GBRT, XGBoost, CatBoost, LightGBM）が線形・SVR・MLP/LSTM/GRU・RL を一貫して上回る。GNN/attention は GBDT に対し 2〜10% 程度の改善で、軌跡・グラフデータが必要（本コンペでは利用不可）。出典（レビュー）: https://mdpi-res.com/d_attachment/applsci/applsci-14-01306/article_deploy/applsci-14-01306.pdf
- 運用データ（非監視データ）での実空港 RMSE: CDG 回帰木 1.6 分（Herrema 2018）、浦東 XGBoost 2.7 分、CLT RF 4.8〜5.5 分（Lee 2016: https://ntrs.nasa.gov/api/citations/20180002162/downloads/20180002162.pdf ）、Chalmers 2026 スウェーデン大空港 RF ≈ 3 分。秒単位の監視データがあると 76〜145 秒（ATL GBM 78 秒: https://www.sesarju.eu/sites/default/files/documents/sid/2021/papers/SIDs_2021_paper_78.pdf 、深圳 CatBoost 143 秒、北京 RF 160 秒）。
- 羽田 2026（Liu & Itoh）: 待ち行列を除いた unimpeded 部分だけを静的特徴量で予測すると R² 0.43 に留まり、動的特徴量の追加でも MAE 改善は 2 秒程度。**説明可能な分散の大半は待ち行列成分にある。** 出典: https://mdpi-res.com/d_attachment/aerospace/aerospace-13-00684/article_deploy/aerospace-13-00684.pdf
- 外れ値処理: PRU > 120 分、CODA > 180 分、学術研究は 3σ・IQR（上ヒゲ ≈ 27 分）・< 0 秒 / > 3600 秒。RMSE 評価では **学習の目的変数はフィルタするが、予測値の切り詰めは分布を確認してから**。

## 5. 過去の PRC Data Challenge 上位チームの共通点と教訓（2024 TOW、2025 燃料）

**共通点**
- 両年とも優勝は**小規模でドメイン専門性の高いチーム**（2024 ENAC、2025 TU Delft 博士学生）。単一 LightGBM + 徹底した特徴量工学・データクリーニング・目的変数の再スケーリングが勝因で、モデル選択の寄与は小さい。
- **2024 優勝（team_likable_jelly）**: 目的変数を (TOW−EOW)/(MTOW−EOW) に正規化しつつ、サンプル重み (MTOW−EOW)² で損失を kg 単位の RMSE に戻す。20 シードの LightGBM 平均で 1612 → 1562。誤った出発/到着時刻を軌跡から補正。METAR（雷雨・霧）を複数半径で入れてモデルに選ばせる。出典: https://github.com/prc-data-challenge-2024/team_likable_jelly 、https://journals.open.tudelft.nl/joas/article/view/8252
  - **taxi-out への類推**: 目的変数や主要特徴量を空港 × 滑走路 × スタンド群ごとに正規化し、サンプル重みで損失を秒単位 RMSE に戻す。
- **2024 準優勝（team_tiny_rainbow）**: CatBoost 主体、二次計画法で重み最適化したアンサンブル。アブレーションで運用/軌跡特徴量 +24.8%、地理 +8.1%、時間 +6.5%、気象 +1.8%。CatBoost 以外のメンバーはほぼ冗長。出典: https://journals.open.tudelft.nl/joas/article/view/7963
- **2025 優勝（resourceful-quiver）**: 燃料量ではなく**燃料流量（kg/s）を予測して積分**（220 → 200）。不自然なラベルを除外。単純で物理的に妥当な質量近似が精密だがノイズの多い物理連鎖より良い。ERA5 風は寄与なし〜負。「軌跡を補完する試みは常に悪化」。`flight_id` でグループ化した非ランダム k-fold。出典: https://raw.githubusercontent.com/meldorr/PRC-2025/main/PRC_2025_report.pdf 、https://github.com/prc-data-challenge-2025/resourceful-quiver
  - **類推**: taxi-out を「unimpeded 参照 + additional」に分解し、additional の分布が扁平な表現で学習する。
- **2025 準優勝（jubilant-vase）**: XGB/LGBM/CatBoost を OOF 上の scipy 最適化で重み付け（CatBoost 64%）。CV 241 → LB 224 → 最終 211.99（2 位）。出典: https://github.com/prc-data-challenge-2025/jubilant-vase
- **2025 3 位（outspoken-tornado）**: 運営は「ML 主導でドメイン特徴量が薄い」と評価。「全特徴量 + 160 回の Optuna」で 298〜304 に悪化した実験もあり、特徴量過多・過剰チューニングの危険。出典: https://github.com/prc-data-challenge-2025/outspoken-tornado
- bright-lobster（2025, JOAS 8764）: 133 → 逐次前向き選択した**小さい特徴量集合は検証 RMSE が僅かに悪くても隠しテストで明確に良い**（214.9 vs 223.2）。出典: https://journals.open.tudelft.nl/joas/article/view/8764

**最終評価の仕組み（確認済: 2025 ranking ページを直接取得）**
- 2024: 公開 LB は 105,959 便、最終順位は締切約 1 週間前に投入された追加 52,190 便を含む 158,149 便。3 位のチームはリポジトリ公開が無く賞金なし、4 位が 3 位賞金。
- 2025: phase 1（〜11/09）は `fuel_rank_submission`、phase 2（11/10〜11/30）は別ファイル `fuel_final_submission.parquet`（10 月分を追加、ラベル非公開）で賞金判定。脚注「リーダーボードからの学習を防ぐため」。**選考委員会が最終スコア・独自性・コード/文書の公開性を考慮して受賞者を決定し、参加者はその決定に拘束される**。JOAS 8750: 「ある月で上位のチームが他の月で順位を落とす例が複数あり、期間をまたぐ汎化が実際の課題だった」。
- 運営のリポジトリ講評（2025）: ドメイン知識に基づく手法・文書化（JOAS 論文水準）・クリーンなコードを明示的に評価。ハードコードされたパス・コメント不足・欠落データファイルは減点。

## 6. 本コンペへの含意と差別化仮説（優先順位付き）

1. **離陸時刻アンカーの待ち行列特徴量**: Ranking で既知の離陸時刻を基準に、同一滑走路で直前 10〜20 分に離陸/着陸した機数、離陸系列内の位置（直前・直後の離陸との間隔）、PRU 式の 20 分前スループットと飽和比を作る。文献の「経験した待ち行列」に最も近い代替。
2. **unimpeded 参照 + additional の分解**: スタンド × 滑走路（× 期間）の P10/P20 参照値を Snowflake で計算し、GBDT は additional を学習。目的変数のスケーリングと重み付けで RMSE（秒）を最適化。
3. **feature-availability parity**: Training でも DEP 行のオフブロックをマスクして特徴量を計算（docs/01 §8）。浦東の「計画値置換で MAE 2.06 → 3.94 分」が示すとおり、ここを誤ると CV が大幅に楽観化する。
4. **ARR 行由来の区間制約**（折返し下限・スタンド占有上限・予定時刻ソフト上限）を特徴量と事後クリップに使う。
5. **NM 時刻列の情報量の定量化**（`AOBT_3`, `LOBT`, `EOBT_1` と実オフブロックの差の空港別分布）→ 二層モデル。
6. **気象・防除氷**: METAR（Iowa State ASOS）から視程・降水・気温・風、1 月は防除氷条件フラグ。Herrema の CDG 研究で防除氷ステーションが上位特徴量。
7. **月ブロック CV（Jan/Jul 2025 fold）+ 末尾 forward holdout** を標準にし、隠し最終評価を前提に LB 過学習を避ける（2025 の順位変動の教訓）。
8. **小さく頑健な特徴量集合 + シード平均**: bright-lobster と likable_jelly の教訓。特徴量過多・Optuna 過多を避ける。
9. **単一 LightGBM を主軸、CatBoost を第 2 メンバー**にした軽量アンサンブル（OOF 重み最適化）。GNN 等は本データでは不要。
10. **文書・コード品質を選考基準として扱う**: README を JOAS 論文構成で育て、決定論的再現パイプラインを維持。

## 7. 根拠が弱い・未検証の事項と追加調査

- 本書の全引用は調査エージェントが開いた URL に基づくが、第 2 エージェントによる再検証は実行環境の上限で未完了。特に数値（CODA 表、各研究の RMSE）は計画の意思決定に使う前に原典で再確認する。
- 未完了の調査方向: (a) 空港運用知識（各空港の滑走路運用・遠隔スタンド・防除氷・A-CDM の TSAT/TOBT の影響、EHAM Polderbaan、EGLL の単一滑走路運用、LTFM の特性）、(b) 本設定特有の戦術（Discord・ブログ等で既に公開されている 2026 年の議論の有無、RMSE 向けの目的変数変換・外れ値方針）。→ Step 3 追加調査として小規模に再実行する。
- 「現 LB 首位が空港内 SD を下回る」は CODA 夏季値との比較であり、冬季（1 月）は分布が異なる。

---

## 8. 2026 年参加チームの公開知見（追加調査 2026-09-10。出典は各チームの公開 GitHub。運営確認は Discord 経由の二次情報）

計画に直結する事実（自データで再確認するまで「報告値」扱い）:

- **評価セットは 09-04 に差し替え**られた（215,876 → 344,841 行）。09-04 以前の LB スコアは比較不可。出典: https://raw.githubusercontent.com/LeeMarshall1113/prc-taxiout-2026/main/notes/2026-09-04-data-findings.md
- **`AOBT_3_flt` は Ranking 出発便の 98.52% に存在**。素朴予測 `MVT_TIME − AOBT_3` の RMSE は 384.9 秒（相関 0.536、1 分以内一致は 21%）で「強い特徴量だがリークではない」。`TAXITIME = MVT − BLOCK` は Training 全行で厳密に成立。出典: https://raw.githubusercontent.com/ahmetabdullahgultekin/prc-taxiout-2026/master/docs/data_probe_report_2026-09-01.md
- **2 つの時計の差分**（`y = (MVT − AOBT_3) − f(x)`、`f = BLOCK − AOBT_3` を GBDT で学習）: 同一 fold・特徴量・シードで 285.2 → 252.7 秒。NM 無し行（1.53%）は別モデル（単一モデルだと 144 秒悪化）。出典: https://raw.githubusercontent.com/saurabhdxt-labs/prc-challenge/main/README.md
- 同チームの負の結果: 気象 +0.5 秒、待ち行列特徴量 +2.7 秒、OOF TE −1.0 秒、winsorize −30 秒、絶対誤差損失 −15 秒（正 = 改善）。
- **RMSE は尾部に支配**: Jan+Jul 2025 の 344,336 行のうち、y > 80,000 秒の 7 行が MSE の 44%、上位 20 行で 55%。体制は「LIRF の NM レコード無し」行で、`BLOCK == SCHED`（報告の代替）や 1 日ずれ。運営（Discord、二次情報）はこれらが Ranking の正解にも含まれると述べた。決定論的な LIRF 帯域ルールで LB −58 秒、ハード切替は +195 秒悪化。出典: https://raw.githubusercontent.com/skylinkapi/prc-data-challenge-2026-kind-mango/main/docs/MODEL_ANALYSIS.md
- 学習から > 120 分の行を除くと +24 秒、> 60 分で +36 秒悪化。log 目的変数 / Huber は「MSE より大幅に悪い」。出典: https://raw.githubusercontent.com/ahmetabdullahgultekin/prc-taxiout-2026/master/docs/experiments.md
- **LightGBM `linear_tree`** で CLEAN RMSE −10.5 秒（対差分 CI −14.5〜−7.3）。木は `y ≈ proxy − f(x)` の恒等部分を表せないため。広域入力には signed-log のコピーを添えて外挿を抑える。
- **検証**: Jan+Jul 2025 ホールドアウトは LB より 12〜15% 悲観的（尾部行が多い）、Jul–Dec や Dec のみは楽観的。単一 RMSE のブートストラップ区間は約 131 秒幅で、行単位の対差分のみが約 5〜7 秒を解像する。12 fold 月ブロック CV を選択に使う。
- **ドリフト**: 到着 taxi-in モデルの 2025 → 2026 劣化は +6%（EHAM +129 秒は 1 月の降雪危機）。クリーン行に大きな運用ドリフトは無い。
- **外部データの落とし穴**: OPDI（ADS-B 由来イベント）の件数が 2026 に 5 空港で崩落し、学習で密・推論で空の特徴量が全行をずらした。被覆が変わる外部特徴量は敵対的検証で検出して除外。
- 自己参照リーク: 「他便の直近 taxi」窓に自便レコードが入って相関の 45% が目的変数そのものだった例。窓の終端を 600 秒ずらす。
- **気象**: −5〜0°C で降水ありの平均 taxi-out は乾燥時の 1.4〜1.7 倍（LFPG 2,354 vs 1,372 秒、EGLL 1,950 vs 1,416）。同じ降水でも 7°C 超では +60 秒未満。1 月の防除氷条件は LSZH 18%、EHAM 13%、EDDM 11%、LTFM 10%、EDDF 8% の時間帯で発生。IEM ASOS で 48 観測/日・欠損 < 0.03%。出典: https://raw.githubusercontent.com/ahmetabdullahgultekin/prc-taxiout-2026/master/docs/facts.md
- ATXOT P10 参照値: 残差化の目的変数としては無効（±5 秒の雑音内）、特徴量としては LightGBM gain の 18.8%。
- アンサンブル: 2026 のブレンド利得は 0.7〜1.8 秒（約 0.5%）。2024 の 20 シード平均で約 3%。
- 上位チーム（elegant-alligator、LB 271）は Jul+Nov で調整し Dec で監査する凍結手順を採用。ローカル値は公式より 15〜70 秒楽観的だった。
- 倫理面: kind-mango は `AOBT_3` / `LOBT` を「ランキング処理の悪用」に当たると読み使用を拒否し、自らの下限を約 300 秒と見積もる。運営のデータページは `AOBT_3_flt` を通常の列として説明し、使用チーム（LB 6〜21 位）に制裁は無い。→ 本プロジェクトの判断は docs/decisions.md。

## 9. 空港運用の知見（追加調査 2026-09-10。要点のみ）

- **PRU 月次 Additional taxi-out**（xlsx、2018-01〜2026-06）: 2025 年平均（分、追加 / 参照）EGLL 6.50 / 16.23、LIRF 6.74 / 12.31、LTFM 4.80 / 12.14、LFPG 4.32 / 11.80、LEBL 3.82 / 12.02、LEMD 3.62 / 13.16、EDDF 3.54 / 10.68、EDDM 3.39 / 9.54、LSZH 3.30 / 8.61、EHAM 3.25 / 9.76。**2026 年 1 月 EHAM は 7.03**（1/2〜7 の降雪・防除氷液不足）。テスト期間の値は使わない方針（docs/01 U12）だが、空港 × 月の季節性の理解には 2025 年以前の値を使う。出典: https://www.eurocontrol.int/performance/data/download/xls/Taxi-Out_Additional_Time.xlsx
- **2026 年 1 月の降雪**: EHAM 1/2〜7、EDDM 1/4〜5、LSZH 1/3・10、LFPG 1/7、EGLL 1/9〜13、EDDF 1/12〜13、LTFM 1/12、LEMD 1/28。METAR 由来の気温 × 降水特徴量が最も効く日。
- **2026 年 7 月の構造変化**: EGLL 北滑走路 09L/27R が 6〜10 月の日〜木 23:00〜06:00 現地に閉鎖（夜間単一滑走路）。**EDDF ターミナル 3**（滑走路系の南側、33 以上の新スタンド、新誘導路 S）が 4/23 開業、**EDDM T1 ピア**（西エプロン、非シェンゲン全便）が 4/21 開業 → 2025 年に存在しないスタンド ID が 7 月の Ranking に出る。
- **2025 年（学習データ）の体制変化**: LFPG 09L/27R 閉鎖 4/14〜6/5、09R/27L 閉鎖 7/15〜12/8（北側ダブレットが単一滑走路の混合運用）。LTFM は 4/17 から 3 本独立平行運用（120 → 148 機/時）。EHAM Polderbaan の着陸制限（3 月）と滑走路整備（10〜11 月）。→ LFPG の滑走路別統計は 2025 の月で偏る。
- **防除氷の場所**: リモートパッド（taxi-out に含まれる）= EDDM（滑走路端の EFM パッド）、EDDF（RWY 18 西の DP1/DP2、25 用の DP3/DP4。on-position も多い）、EHAM（4 ステーション 25 台、1 機 10〜30 分）、LSZH（パッド + on-stand）、LTFM（Deicing 1〜5 エプロン）。on-stand 主体 = EGLL（BA 専用パッドのみ例外）、LEBL、LEMD。
- **A-CDM は全 10 空港で稼働**（LTFM は 2023 年末承認、2024 に DMAN）。TSAT ±5 分でオフブロックが集約され、待ち行列の一部はスタンドで吸収される。
- **優先滑走路運用**（要点）: LEMD 北運用 = 出発 36L/36R（36R が既定、36L は長距離）、LEBL 西運用 = 到着 24R / 出発 24L（夜は到着 02 / 出発 06R）、LSZH は時間帯別コンセプト（日中出発 28/16、朝夕 32/34）、EDDF 出発 25C/07C + 18（南向き専用）、EGLL 西向き優先・15:00 に交替、EHAM Polderbaan（36L 出発は約 5 km の長距離タキシー）、LFPG 内側出発・外側着陸、EDDM 26 優先で北行きは北滑走路、LIRF 出発 25 主体、LTFM 5 本。
- **オープンデータ**: IEM ASOS（公共ドメイン、1 秒/IP のスロットル、非米国は降水量なし、LTFM の METAR 備考に滑走路別風向風速）、Open-Meteo（CC BY 4.0、視程なし、降雪あり）、OurAirports（公共ドメイン、滑走路端座標）、OSM（ODbL、parking_position / taxiway）、ENAIRE AIP のスタンド座標 PDF（LEMD / LEBL）。
