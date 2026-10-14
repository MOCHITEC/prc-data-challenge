> **最新の推奨（2026-10-14 04:45 UTC）**: `final_recommended.parquet` = v6（md5 **7e53142e478bbb022ae6584850143040**、全メンバー 5 seeds ＋ ADS-B 層 3 seeds）。1 つ前の v5（411a6aa0…）と v4（d0ce32b6…）も同じフォルダに残してある。詳細は末尾の節。それより上の節は作業の記録です。

# データ分析（laughing-cray）の最終候補（2026-10-13 00:50 UTC）

チーム指示（`claude/model-b-and-arrival-features:docs/final_phase_2026-10-11/NO_FINAL_UPLOAD.md`）に従う。
- このスレッドはバケットに何もアップロードしていない。
- バケットの `resilient-hedgehog_final.parquet`（10-09）はこのスレッドが作ったものではない。
- v48・v49 もこのスレッドの提出ではない。

## ファイル

| ファイル | md5 | 中身 |
|---|---|---|
| `final_from_v48.parquet` | be8d89dcf79bd59a6f43d80193c95685 | 1・7 月 = v48 のファイル（旧ランキングの時刻）。2 月 = 下の chain（自前の ADS-B 層）。6 月 = 下の chain（ADS-B 層は外部知見の A12 層）。validate OK |
| `pre_adsb_base_blend5_final.parquet` | 38edd5a6a6ab0328de841936cffa907a | ADS-B の層をかける前の 5 メンバー blend（4 か月、float）。外部知見の `final_adsb.py` などに渡す土台 b |

## 作り方（chain）
作業場所は `/home/user/prc-final`、メンバーの出力は `/home/user/pb-work/outputs/final`。

1. **メンバー**: final の raw に対して `prc.pipeline --skip-cv`（peaceful-babbage の pb-work、6a97bf0）。
   - 特徴量キャッシュの matched の行は、union で作り直した fl_/seq_ 列で置き換え済み（C13 と同じ修正。`analysis/local_audit_fix2026.py` を final のフレームで実行）。
   - a2o = l2_opt_l6_cmix（3 seeds）、pao_l6（3）、p2o_l6（3）、mlp_l6（3）、c2 = cb_l2（1）。
2. **blend**: matched の行は ((a2o + pao)/2 + p2o + c2 + m4) / 4。unmatched の行は LGB の model B。
3. **ADS-B の層**（2 種類）。
   - 自前: `docs/final_2026-10-11/adsb_layer.py apply`。v47 の層の組み立て（G12A / G7A / GL4 / AIR / POS / RULE、LTFM は c = 0）を作り直したもの。対象は 1・2・7 月。
   - A12: 外部知見の `docs/appear_tier_2026-10-05/final_adsb.py`。`handoff/2026-10-11/adsb` を使う。対象は 4 か月。
4. **overlay**: `docs/final_2026-10-11/apply_overlays.py --laneB --lobt-clip --copy-feat unfixed`。
   - lane-B: NM-blind（w 0.25）と ucmix。
   - LIRF: copy band、time-zone class、matched copy overlay（0.4·q）。
   - 最後に LOBT の切り詰め（±3606 s）。

## 検証（1・7 月の行で v47 / v48 と比べる）

| 比べたもの | RMS 差 | 差の中央値 |
|---|---|---|
| chain（自前の層）vs v47 | 52.5 s | 16 s |
| chain（自前の層）vs v48 | 53.2 s | 16 s |
| chain（A12 の層のみ）vs v48 | 68.8 s | 16 s |
| v48 vs v47 | 22.6 s | 6 s |

- **自前の層の方が v47 / v48 に近い。** A12 の層だけでは、posjoin、LIRF matched（GL4）、air-only などが抜けるためと考えられる。
- **6 月は自前の層がまだない**（adsb.lol の 6 月分を取り込み中）。そのため 6 月だけ A12 の層を使っている。
  - 取り込みが終わったら、6 月も自前の層（必要なら + (c_A12 − c_P12)）にした v5 を置く予定。
- **差の中央値 16 s は、主に final_ranking の MVT の ±12 s のずらし**と、ADS-B の層の作り直しの近似から来ている。
  - スクリプト自体は、旧ランキングのデータで v47 を全行そのまま再現することを確認済み。

## 更新（2026-10-13 02:30 UTC）: 6 月も自前の ADS-B の層に差し替えた
- adsb.lol の 6 月分を取り込んだ。30 日中 29 日分で、2026-06-11 はデータがなかった。
- 6 月の行は自前の層（v47 / v48 と同じ組み立て）に差し替えた。
  - 2026-06-11 の行だけは、A12 の層（外部知見の 4 か月版。この日のデータもある）のままにした。
- 1・2・7 月は変わっていない。6 月は 142,281 行が変わり、変化の RMS は 44.6 s。
- validate OK。md5 は be8d89dcf79bd59a6f43d80193c95685。

## 追加（2026-10-13 02:35 UTC）: 混ぜ方の見直しを入れた版
色々調査スレッドの提案（`peaceful-babbage:docs/final_phase_2026-10-13`）を入れた。
- **変更**: LIRF 以外の matched の行で、内側の a2o:pao を 0.5:0.5 から 0.2:0.8 にした。LIRF と unmatched の行は変えていない。
- **根拠**: 2025 年の OOF で、交差検証（fold を 1 つずつ外して評価）の見込みは −0.12〜−0.13 s。
  - すべての fold で改善した。
  - c2 は p2o で代用して測っている。

### 作り方
- **2・6 月**: 新しい blend（`pre_adsb` と同じメンバー）→ 自前の ADS-B の層 → overlay → LOBT の切り詰め、の順に作り直した。
  - 2026-06-11 の行は、A12 の層の値に blend の差を足した。
- **1・7 月**: v48 に「final の特徴量で作った chain の、新旧の差」を足した。
  - 動いたのは非 LIRF の matched の行で、変化の RMS は約 4 s。

### ファイル

| ファイル | md5 | 中身 |
|---|---|---|
| `final_from_v48_reweight.parquet` | dd361caed874a6e39cc8b7e508d9aa79 | 4 か月分。validate OK |
| `janjul_only_v48_reweight.parquet` | 1332ed10d393f247a361c55dc7517a23 | 1・7 月だけ（344,841 行、旧 submitting の形）。v48 より LB が下がらないかを確かめる用。提出するかは提出役が判断する |

### 判断の目安
- janjul_only の LB が v48（234.4099）以下なら、`final_from_v48_reweight` を使う。
- そうでなければ `final_from_v48` を使う。

## 追加（2026-10-13 03:50 UTC）: cmix の LTFM 拡張を足した版
- **変更**: a2o を `copy_mix_airports: [LIRF, LTFM]` で学習し直した（L6 あり、3 seeds、final のデータ）。設定は `pb-work/configs/main_flseq_osm_nolirf_l2_opt_l6_cmix_ltfm.yaml`。
  - a2o で変わるのは LTFM の matched の行だけ（87,446 行、a2o 単体での変化の RMS 57 s）。
- **作り方**: `final_from_v48_reweight` の LTFM の matched の行に、0.05 ×（a2o_ltfm − a2o）を足し、LOBT の切り詰めをかけ直した。
  - 0.05 は、混ぜ方を見直したあとの非 LIRF の行での a2o の重み（0.25 × 0.2）。
  - LTFM は ADS-B の補正が c = 0 なので、ADS-B の層を作り直す必要はない。
  - 4 か月とも final の特徴量で作った差を足している。1・7 月は v48 の上に足した形。
- **変化**: 69,093 行が動き、動いた行での変化の RMS は 3.3 s。
- **根拠**（色々調査、copy_plan.md）: a2o 単体で、全体 −0.25 s、LTFM matched で −2.26 s。すべての fold で改善した。
  - blend の中では重みが 0.05 なので、見込みは −0.01〜−0.04 s 程度。

| ファイル | md5 |
|---|---|
| `final_from_v48_reweight_ltfm.parquet`（4 か月、validate OK） | 155f479aa27a8b7a8f6ed5bf8355c7bf |
| `janjul_only_v48_reweight_ltfm.parquet`（1・7 月だけ、LB で確かめる用） | 45e7b291b8415b81368b3c0a94f0cc1a |

## 取り下げ（2026-10-13 09:20 UTC）: cmix の LTFM 拡張
- 色々調査の 2025 年 2・6 月の検証で不合格になった（`peaceful-babbage:docs/final_phase_2026-10-13/fj_results.md`）。
  - LTFM の matched の行で +26.1 s [+5.7, +44.9]。2 月は +46.7 s。全体で +3.66 s の悪化（a2o の重み 1/8 での測定）。
- そのため `final_from_v48_reweight_ltfm.parquet` と `janjul_only_v48_reweight_ltfm.parquet` は削除した。
  - 重みを 0.05 にしても、悪化する向きは変わらない。
- **現在の推奨は `final_from_v48_reweight.parquet`**（md5 dd361caed874a6e39cc8b7e508d9aa79）。
- 外部知見の「reweight+LTFM の上に作った最終候補」も、LTFM 抜きの reweight 版の上に作り直す必要がある。

## 取り下げ（2026-10-13 11:00 UTC）: 混ぜ方の見直し（a2o:pao = 0.2:0.8）
- 色々調査の 2025 年 2・6 月の検証で不合格だった。
  - LIRF 以外の matched の行で、内側の平均同士を比べると +2.18 s [+0.64, +3.95]。2 月は +4.30 s。
  - 2・6 月では pao の方が a2o より弱い。
- そのため `final_from_v48_reweight.parquet` と `janjul_only_v48_reweight.parquet` は削除した。
- **現在の推奨は `final_from_v48.parquet`**（md5 be8d89dcf79bd59a6f43d80193c95685）。混ぜ方は v48 と同じ。
- 進行中: CatBoost（c2）を 1 seed から 3 seeds の平均にする。この改善は `final_from_v48` の上に作る。

## 追加（2026-10-13 12:20 UTC）: CatBoost を 3 seeds の平均にした版
- **変更**: c2（cb_l2）を seed 42・43・44 の 3 本の平均にした。ほかは `final_from_v48` と同じ（混ぜ方も同じ）。
  - 設定は `pb-work/configs/local_pb/main_flseq_osm_nolirf_cb_l2_s{43,44}.yaml`（final のデータで学習）。
- **根拠**: seed を変えると、c2 単体で matched の行の差が RMS 57 s ある（42 と 43 の比較）。
  - blend での重みは 1/4。平均を 3 本にすると、この分散がおよそ 1/3 になる。
  - 見込みは約 −0.14 s。seed 平均なので、月によって良し悪しが逆転する心配はない。
- **作り方**:
  - 2・6 月: 新しい blend → 自前の ADS-B の層 → overlay → LOBT の切り詰め、で作り直した。
    - 2026-06-11 の行は、A12 の層の値に blend の差を足した。
  - 1・7 月: v48 に「final の特徴量で作った chain の、新旧の差」を足した。
- **変化の RMS**: 1 月 9.5 s、2 月 6.8 s、6 月 6.5 s、7 月 6.8 s。

| ファイル | md5 |
|---|---|
| `final_from_v48_cb3.parquet`（4 か月、validate OK）← **現在の推奨** | 6bbcb6c831d131acbd33ebb647f7d6f2 |
| `janjul_only_v48_cb3.parquet`（1・7 月だけ、LB で確かめる用） | b1e8203df971f331f5ea570ae040eb6e |

## 追加（2026-10-13 13:05 UTC）: 2・6 月に appear（スタンド出現）の補正を入れた版 ← **現在の推奨**
- 外部知見の指摘（`beautiful-davinci:handoff/2026-10-13/CANDIDATES.md` 部品 2）: v47 / v48 の 1・7 月の ADS-B の層には、
  自前の層（G12A/G7A/GL4/AIR/POS/RULE）に無い **appear の項**（12 か月、EDDF は間引き版）が入っている。
  v39 でこの層は LB −2.57 s。1・7 月で v48 − v47 ≈ (c_A12 − c_P12) + EDDF − (c_A4 − c_P4)、相関 0.989。
- **やったこと**: 外部知見の `final_adsb.py --months 02,06 --increment` を、**今の土台 `base_blend_cb3`**（3 seeds の CatBoost 入りの blend）で回し、
  増分 (c_A12 − c_P12)（EDDF は c_A12thin − c_P12）を `final_from_v48_cb3` の 2・6 月の行に足した。
  - 2026-06-11 の行は除外（A12 の層が直接入っているため）。【後日訂正: 06-11 の A12 層は実質空だったので、v2 以降は 06-11 も自前の層 + 増分にしている】LTFM、LIRF の unmatched、LOBT 境界の行は増分側で触らない。
  - 足したあとに LOBT の切り詰めをかけ直した。1・7 月は変えていない。
- **変化**: 2 月 103,343 行（RMS 39.9 s）、6 月 123,751 行（RMS 42.3 s）。Σd²/N 575。
- **検証（外部知見）**: 1・7 月で、自前の層 + 増分は v48 に近づく（RMS 差 53.2 → 43.4 s）。
- これで 2・6 月は v48 と同じ組み立て（自前の層 + 12 か月 appear + EDDF 間引き）になった。

| ファイル | md5 |
|---|---|
| `final_from_v48_cb3_appear.parquet`（4 か月、validate OK）← **推奨** | 6aa805f44a6171e5a2a79263d952baae |
| `janjul_only_v48_cb3_appear.parquet`（1・7 月だけ。`janjul_only_v48_cb3` と同一） | b1e8203df971f331f5ea570ae040eb6e |

## 追加（2026-10-13 16:00 UTC）: 候補（まだ推奨ではない）— a2o の feature_fraction 0.25 ＋ 計算ずれ修正の正確版
- `final_from_v48_cb3_appear_ff025_c13.parquet`（md5 9b09a040bd02af493704e1899b1bc67f、validate OK）、1・7 月だけ: `janjul_only_final_from_v48_cb3_appear_ff025_c13.parquet`（014e303b430e74df8c7cef1660bb6f70）。
- 変更 1: a2o を feature_fraction 0.25 で学習し直した（3 seeds、final のデータ、`configs/a2o_l6_cmix_ff025.yaml`）。
  - 根拠: 調整ワーカーの 2025 年 1・7 月 fold の検証（3 seeds 平均）で、matched −0.61 s（1 月 −0.79、7 月 −0.43）。
  - ただし検討役の見積りでは、blend の中では約 −0.03 s にしかならない（CatBoost が同じ誤差を別方向から拾っているため）。
  - 2・6 月の検証はこれから（下の「Feb+Jun screen」）。
- 変更 2: v47 の C13（学習時・予測時の計算ずれ修正）で、a2o の分を pao の差分で代用していたのを、a2o 自身の差分に置き換えた。
  - e = s·(Δa2o_new − Δpao_old)/8、LIRF では s = 1 − 0.4q。対象は 1・7 月の matched 339,551 行。Σe²/N 1.2 s²、max|e| 82 s。見込みは −0.005 s 程度（衛生的な修正）。
- 作り方: `docs/final_2026-10-11/assemble_step.py --a2o <a2o flfix parquet> --tag ... --c13`（1・7 月 = v48 + (新 chain − 旧 chain) + e、2・6 月 = 新 chain + appear 増分、最後に LOBT の切り詰め）。
- 変化（推奨ファイル比）: 1 月 RMS 4.2 s、2 月 3.7、6 月 3.7、7 月 3.2。

## 進行中（16:00 UTC）: Feb+Jun screen と cat_tight
- 調整ワーカーの検証で、LightGBM の **cat_smooth 50 / cat_l2 50 / min_data_per_group 200（cat_tight）** が最良だった:
  3 seeds 平均で matched −0.96 s（1 月 −0.86、7 月 −1.05）。3 seeds すべてで両 fold とも改善。
- 今日、2・6 月の検証で 2 件の案が落ちたので、cat_tight / ff_025 も 2025 年 2・6 月の held-out fold で検証してから使う（実行中、約 1 時間）。
- 通れば a2o と pao を cat_tight で学習し直し、推奨ファイルを更新する（見込み: blend で −0.1 s 前後）。

## 提出役への判断材料: 1・7 月の時刻のずらし（+1×δ）
- final_ranking の 1・7 月の MVT は旧 ranking から −12〜+12 s ずれている（sd 4.9 s、92 % の行が動く）。到着便の公開ラベルは毎回ずらした時刻で再計算されている。
- 検討役の結論: 出発便の正解も「ずらした時刻で再生成」の可能性が高め（p ≈ 0.7）。その場合、v48 の 1・7 月をそのまま使うと matched 行あたり +48 s² の損。
  - 対策「+1×δ を matched の 1・7 月の行に足す（= anchor を final の MVT に合わせ直す）」は期待値 −0.03 s（幅 −0.05〜+0.05 s）。+2×δ は推奨しない（隠れた正解の生成法の推測になる）。
  - **注意**: この変種を旧 LB（1・7 月）で確かめると、正解がずらされている場合は逆に約 +0.10 s 悪く見える。LB で判断できない。
- 作り方: `docs/final_2026-10-11/add_delta_janjul.py <in> <out>`（最終の推奨ファイルから作る。ファイルは最後に置く）。推奨には入れない。

## Feb+Jun screen の結果（2026-10-13 17:00 UTC）— `docs/final_2026-10-11/tune/fj_screen.md`
2025 年の 2 月・6 月を held-out にした fold（学習は残り 11 か月、ES は 12 月）、anchored LightGBM（model A）、単一 seed 42、matched RMSE。

| 設定 | 2 月 d | 6 月 d | 2+6 月 pooled d | 判定 |
|---|---|---|---|---|
| cat_tight（cat_smooth 50 / cat_l2 50 / min_data_per_group 200） | −0.55 | −0.82 | **−0.70** | 合格（1・7 月の 3 seeds でも −0.86 / −1.05） |
| ff_025（feature_fraction 0.25） | +0.68 | −0.91 | −0.20 | 2 月が悪化（seed のばらつき 0.13 s を超える）→ **使わない** |

- **決定**: a2o と pao を cat_tight で学習し直す（ff_025 は使わない。`..._ff025_c13` の候補は取り下げ）。a2o の学習を 17:01 UTC に開始（約 1.3 時間）、続けて pao（約 1.5 時間）。
- 検討役の見積り: blend での LB 改善は a2o で −0.09 s、a2o + pao で −0.13 s 程度。p2o（plain target）には適用しない（proxy が特徴量なので挙動が変わる。blend での上積みはほぼ 0 と見積り）。

## 追加（2026-10-13 18:35 UTC）: a2o を cat_tight にした候補（pao はまだ。中間版）
- a2o を cat_tight（cat_smooth 50 / cat_l2 50 / min_data_per_group 200、ほかは本番と同じ、3 seeds、final のデータ）で学習し直し、
  計算ずれ修正の正確版（C13、Σe²/N 1.2 s²、max|e| 82 s）も入れた。ff_025 は使っていない。
- a2o 単体の変化（本番 a2o 比）: 空港ごとの RMS 20〜28 s（seed の違いを含む）。
- 推奨ファイル比の変化: 1 月 RMS 4.2 s、2 月 3.7、6 月 3.7、7 月 3.1。
- ファイル: `final_from_v48_cb3_appear_cat_a2o_c13.parquet`（md5 43770572cbc1cf221bb43244396cf1f7、validate OK）、
  1・7 月だけ: `janjul_only_final_from_v48_cb3_appear_cat_a2o_c13.parquet`（a68add6555a5ea00e1fd6ffefe24f882）。
- pao の cat_tight 学習を 18:23 UTC に開始（約 1.5 時間）。終わったら a2o + pao の版を作り、独立した点検のあとで推奨を更新する。

## 追加（2026-10-13 19:10 UTC）: a2o + pao を cat_tight にした候補（独立した点検を実行中）
- `final_from_v48_cb3_appear_cat_c13.parquet`（md5 34c398af5639e62362ffe4e85b526d2c、validate OK）、1・7 月だけ: `janjul_only_final_from_v48_cb3_appear_cat_c13.parquet`（6db820cec35652e54a1103775039b15d）。
- 中身: v48 ＋ CatBoost 3 seeds ＋ 2・6 月の appear 増分 ＋ a2o/pao の cat_tight ＋ C13 正確版。作り方は `assemble_step.py --a2o a2o_cattight_flfix --pao pao_l6_cattight --c13`。
- pao 単体の変化（本番 pao 比）: 空港ごとの RMS 20〜30 s、LIRF は 53 s。推奨ファイル比の変化: 各月 RMS 4.9〜5.6 s。
- 上の a2o だけの中間版（`..._cat_a2o_c13`）は、この版に置き換えたので削除した。
- 4 観点（形式・1・7 月が v48 相当のまま・2・6 月の組み立て・再現性）の点検が通れば、`final_recommended.parquet` をこの版にする。

## 2026-10-13 20:15 UTC: 候補 v2（点検の指摘を反映。独立点検の 2 回目を実行中）
**ファイル**: `final_from_v48_cb3_appear_cat_c13_v2.parquet`（md5 ce79384d26bc0a87ddba43f19df719fa、validate OK）
1・7 月だけ: `janjul_only_final_from_v48_cb3_appear_cat_c13_v2.parquet`（4eb9c497dd175dad23a9fb069b1f21c1）。
1 回目の点検（4 観点）で見つかった 2 点を直した。19:10 の版（`..._cat_c13`）はこの版に置き換えて削除した。
1. **2026-06-11 の行に ADS-B の層が入っていなかった**（外部知見の A12 store は 06-11 の appear 行が 78 行しかなく、A12 の層は実質 0 だった）。
   → 外部知見が自前の `adsb_layer.py feat` で作った 06-11 の特徴量（`beautiful-davinci:handoff/2026-10-13/june_adsb_for_adsb_layer/feat/feat_2026-06-11.parquet`、md5 3574ba0d…）を `adsb_out/feat/` に入れ、06-11 もほかの日と同じ chain にした（AIR 2,850 行 RMS 17 s、G12A 78 行 RMS 60 s、残り 3,171 行は補正なし）。06-11 の 3,909 行が変わった（RMS 12 s）。
2. **pao の cat_tight は LIRF では本番 pao のまま**にした（LIRF の pao の変化が RMS 53 s と他空港の 2 倍以上で、per-airport モデルの LIRF での検証が無いため）。LIRF の 43,725 行が変わった（RMS 7.5 s）。
点検 1 回目の結論: 形式は合格。1・7 月は v48 相当（unmatched 5,290 行は v48 と同一、matched の変化 RMS 9.2 s、平均 −0.04 s）。2・6 月の各層は 1 回ずつ入っている（appear 増分の二重計上なし）。LTFM は ADS-B 補正なし。

### 中身（順番どおり）
1. **メンバー**（final の raw、`pb-work` 6a97bf0 のコード、特徴量キャッシュ `work_final` の matched 行は union 版の fl_/seq_ 列に置換 = `inputs/feat_rank_extra_union_final.parquet`、`analysis/local_audit_fix2026.py`）:
   - a2o = `configs/a2o_l6_cmix_cattight.yaml`（l2_opt + L6 + cmix(LIRF) + cat_smooth 50 / cat_l2 50 / min_data_per_group 200）、3 seeds。
     学習は `docs/final_2026-10-11/flfix_member_final.py`（`prc.train.predict_fold` を使い、union 版と元の rank 行の両方を予測）:
     `cd /home/user/pb-work && SCRATCH=/home/user/prc-final/scratch PYTHONPATH=src python flfix_member_final.py configs/a2o_l6_cmix_cattight.yaml 3 outputs/final/a2o_cattight_flfix.parquet`（pred_fix を使う）。
   - pao = `configs/pao_l6_cattight.yaml`、3 seeds、`prc.pipeline --skip-cv --raw-dir data/raw_final --work-dir data/work_final`。LIRF の行だけ本番 pao（`main_flseq_osm_nolirf_l2_pa_opt_l6.yaml`、3 seeds）。
   - p2o = `main_flseq_osm_nolirf_plainy_l2_opt_l6.yaml`（3 seeds）、m4 = `main_flseq_osm_nolirf_l2_mlp_l6.yaml`（3 seeds）、c2 = `main_flseq_osm_nolirf_cb_l2.yaml` seed 42 + `cb_l2_s43` + `cb_l2_s44` の平均。
   - blend（matched）: ((a2o + pao)/2 + p2o + c2 + m4)/4。unmatched: model B（a2o の run の分）。ログは `docs/final_2026-10-11/logs/`。
2. **自前の ADS-B の層**: `docs/final_2026-10-11/adsb_layer.py apply --base <blend> --out ...`（モデル: `docs/final_2026-10-11/adsb_models/`、行ごとの出力: `inputs/adsb_layer_detail_v2.parquet`（b, comp, c, new））。
3. **overlay**: `apply_overlays.py --laneB --lobt-clip --copy-feat unfixed`（lane-B の差分 = `inputs/{nmb,ucmix}_laneB_delta_final.parquet`、LIRF copy q = `inputs/copyq_LIRF_final_fixed.parquet`）。
4. **appear 増分**（2・6 月の行）: 外部知見の `final_adsb.py <blend> <out> --months 02,06 --increment`（`beautiful-davinci` 6b9d29e、`handoff/2026-10-11/adsb`）。
   増分 = 出力 − round(blend)。3 の出力に足す（出力は `inputs/appear_increment_out_v2.parquet`）。
5. **1・7 月**: v48（`beautiful-davinci:handoff/2026-10-08/C15_appear12_cov26eddf.parquet`）+ (3 の新 chain − 3 の旧 chain [本番メンバー・CatBoost 1 seed])
   + C13 の正確版 e = s·(Δa2o − Δpao_old)/8（LIRF は s = 1 − 0.4q; `inputs/c13_pao_oldframe_flfix.parquet`、`inputs/c13_copyq_LIRF_oldframe.parquet`）。
6. 最後に全 matched 行へ LOBT の切り詰め（±3606 s）、0 で下限、丸めて int32。
実行: `docs/final_2026-10-11/assemble_step.py --a2o outputs/final/a2o_cattight_flfix.parquet --pao outputs/final/pao_l6_cattight_nolirf.parquet --tag final_from_v48_cb3_appear_cat_c13_v2 --c13`。
md5 の一覧は `MD5SUMS`。

---
# ★ 提出役へ（2026-10-13 21:00 UTC 時点のまとめ）

## 推奨ファイル
- **`handoff/2026-10-13/final_recommended.parquet`**（= `final_from_v48_cb3_appear_cat_c13_v2.parquet`、md5 **ce79384d26bc0a87ddba43f19df719fa**、670,790 行、validate OK）。
- 1・7 月だけのファイル: `janjul_only_final_recommended.parquet`（md5 4eb9c497dd175dad23a9fb069b1f21c1）。旧 LB で確かめたい場合用。
- 中身 = v48 ＋ (a) CatBoost 3 seeds 平均 ＋ (b) 2・6 月の appear 増分 ＋ (c) a2o / pao（LIRF 以外）の cat_tight ＋ (d) C13 の正確版、全 matched 行に LOBT の切り詰め。
- 1・7 月の v48 からの変化: unmatched 5,290 行は同一、matched は RMS 9.1 s、平均 −0.05 s（max 600 s は LIRF 1/29 の 1 行、CatBoost の seed 平均 + ADS-B 残差の再評価によるもの）。

## 独立点検（4 観点 × 2 回、反証役つき）の結論
- 形式・網羅: 合格（行数・ID・順序・型・null・負値・空港×月×lane の件数すべて一致。LOBT の範囲違反 0）。
- 1・7 月: 合格（v48 相当。変化は記録どおりのメンバー差分 + C13 だけ）。
- 2・6 月: 合格（各層が 1 回ずつ。06-11 は地上 fix がほぼ無い日で AIR のみ。LIRF 6 月は 7 月と同じく AIR のみの受信。EDDM 2/19 は乱れの日で v48 の 1 月と同じ扱い）。
- 再現性: 値には影響しない指摘が 3 件（入力ファイルが支店外）→ 下の「再現用の同梱物」で対応した。
- 残る軽微な点: 60 s 未満の予測 32 行（主に LSZH のヘリ、v48 でも同じ）。影響は最大 0.03 s。手を入れていない。

## 再現用の同梱物（このフォルダ、md5 は `MD5SUMS`）
- `members/`: 5 メンバーの予測（a2o は `a2o_cattight_flfix.parquet` の pred_fix、pao は `pao_l6_cattight_nolirf.parquet`、c2 は cb_l2 / _s43 / _s44、p2o、m4、本番の a2o / pao）。
- `inputs/`: union 版の fl_/seq_ 列、lane-B 差分、LIRF copy q（`copyq_LIRF_final_unfixed.parquet` = 実際に使った q）、旧 chain（`orig_chain_blend_adsb2_ovl.parquet`）、ADS-B 層の行別出力、appear 増分の出力、C13 の旧フレーム入力。
- `adsb_store/`: 自前 ADS-B 層の入力（1・7 月の learn/pos/rescat、2・6 月の日別 feat/pos）。モデルは `docs/final_2026-10-11/adsb_models/`。
- 外部ブランチの入力: v48 = `beautiful-davinci:handoff/2026-10-08/C15_appear12_cov26eddf.parquet`（md5 8ee6679f43ead60db1a1aec7876caf09）、`final_adsb.py` と `handoff/2026-10-11/adsb`（同ブランチ 6b9d29e）、メンバーのコードは `peaceful-babbage` 6a97bf0。
- スクリプト: `docs/final_2026-10-11/{assemble_step.py, flfix_member_final.py, adsb_layer.py, apply_overlays.py, add_delta_janjul.py}`、設定は `docs/final_2026-10-11/configs/`、ログは `docs/final_2026-10-11/logs/`。

## 任意の変種（推奨には入れていない）
- `optional_janjul_reanchor_plus1delta.parquet`（md5 85291ccce9d21c428612bc0b6b392d55、validate OK）: 推奨ファイルの 1・7 月 matched 行に、final_ranking と旧 ranking の MVT の差（−12〜+12 s）を 1 倍だけ足して anchor を final の時刻に合わせ直したもの。
  - 検討役の見積り: 期待値 −0.03 s（幅 −0.05〜+0.05 s）。出発便の正解が「ずらした時刻で再生成」なら得、そうでなければ最大 +0.05 s の損。
  - **旧 LB では確かめられない**（正解がずらされている場合、旧 LB では約 +0.10 s 悪く見える）。使うかどうかは提出役の判断。

## 進行中（間に合えば差し替える）
- CatBoost を seed 45・46 まで足して 5 seeds 平均にする版（見込み −0.02 s 程度）。できたら同じ点検をして `final_recommended.parquet` を上書きし、この README に追記する。差し替えなければ上の md5 のままが最終。

## 追加（2026-10-13 21:45 UTC）: v3 = v2 ＋ CatBoost を 5 seeds 平均に（点検待ち。推奨はまだ v2）
- `final_v3_cb5.parquet`（md5 838d5f411b383ae9b6c1d626fa1bf04e、validate OK）、1・7 月だけ: `janjul_only_final_v3_cb5.parquet`（0354f63ef337b3074dc75a64a4eae204）。
- c2 = (seed42 + seed43 + seed44 + 2 × mean(seed45, seed46))/5（`members/cb_l2_s45.parquet` は seed 45・46 の平均、`configs/local_pb/cb_l2_s45.yaml`）。
- CatBoost 単体: mean5 − mean3 の RMS 14.4 s（matched）。推奨ファイル比の変化: 各月 RMS 3.5〜4.6 s。見込み −0.02 s 程度（seed 平均なので方向のリスクなし）。
- MLP の seed 45・46 も学習中（22:50 UTC ごろ）。両方入れた版を最終点検にかけ、通れば `final_recommended.parquet` を差し替える。

## 追加（2026-10-13 22:35 UTC）: v4 = v3 ＋ MLP も 5 seeds 平均に（最終点検を実行中）
- `final_v4_cb5_mlp5.parquet`（md5 d0ce32b68f3a7a4599d8a0982ca81988、validate OK）、1・7 月だけ: `janjul_only_final_v4_cb5_mlp5.parquet`（1e7fa78e74f4c00f66eedc9356f144bf）。
- m4 = (3 × mean(seed42–44) + 2 × mean(seed45, 46))/5（`members/mlp_l6_5seeds.parquet`。seed 45・46 は `members/mlp_l6_s45.parquet`、`configs/mlp_l6_s45.yaml`）。
- MLP 単体: mean5 − mean3 の RMS 20.9 s（matched）。推奨ファイル（v2）比の変化: 各月 RMS 5.5〜7.9 s。
- 1・7 月の v48 からの変化: unmatched 5,290 行は同一、matched は RMS 10.9 s、平均 −0.14 s、max 546 s。
- 実行: `assemble_step.py --a2o members/a2o_cattight_flfix.parquet --pao members/pao_l6_cattight_nolirf.parquet --m4 members/mlp_l6_5seeds.parquet --cb cb_l2 cb_l2_s43 cb_l2_s44 cb_l2_s45 cb_l2_s45 --tag final_v4_cb5_mlp5 --c13`。


---
# ★★ 最終版（2026-10-13 23:20 UTC）— 提出役へ

## 推奨ファイル
- **`handoff/2026-10-13/final_recommended.parquet`** — md5 **d0ce32b68f3a7a4599d8a0982ca81988**、670,790 行、validate OK（= `final_v4_cb5_mlp5.parquet` と同一）。
- 1・7 月だけ（旧 LB 用）: `janjul_only_final_recommended.parquet`（md5 1e7fa78e74f4c00f66eedc9356f144bf）。
- 任意の変種（推奨には含めない）: `optional_janjul_reanchor_plus1delta.parquet`（md5 9d0a5bf1b814e4a0cf69ac0d9d69f3c8）。1・7 月 matched 行の anchor を final の MVT に合わせ直したもの（期待値 −0.03 s、幅 −0.05〜+0.05 s、旧 LB では確かめられない）。
- 1 つ前の推奨（v2、md5 ce79384d26bc0a87ddba43f19df719fa）も `final_from_v48_cb3_appear_cat_c13_v2.parquet` として残してある。v4 との違いは CatBoost と MLP の seed 平均（3 → 5 seeds）だけ。

## 中身
v48（1・7 月の LB 最良 234.4099）を土台に、次を積んだもの。
1. 2・6 月の予測: v48 と同じ組み立て（5 メンバー blend → 自前の ADS-B 層 → lane-B（NM-blind / ucmix）＋ LIRF の overlay ＋ LOBT 切り詰め → 12 か月 appear の増分）。
2. CatBoost を 5 seeds 平均（seed 42–46）、MLP を 5 seeds 平均（seed 42–46）。
3. a2o と pao（LIRF 以外）を cat_tight（cat_smooth 50 / cat_l2 50 / min_data_per_group 200）で学習し直し。2025 年の 1・7 月（3 seeds）と 2・6 月（held-out）の検証で合格。
4. C13（学習時・予測時の計算ずれ修正）の a2o 分を正確版に。
5. 1・7 月は v48 + (新 chain − 旧 chain) + C13 補正、全 matched 行に LOBT 切り詰め。unmatched 5,290 行は v48 と同一。matched の変化は RMS 10.9 s、平均 −0.14 s、max 546 s（LIRF 1/31、seed 平均によるもの）。

## 独立点検（4 観点、反証役つき）: v2 と v4 の両方で合格
- 形式・網羅: 行数・ID・順序・型・null・負値・空港×月×lane の件数が final_ranking と完全一致。LOBT の範囲違反 0。
- 1・7 月: v48 相当（差はメンバーの seed 平均 / cat_tight / C13 で説明がつく。LIRF と LTFM の RMS が 15 s を少し超えるのは CatBoost / MLP の seed 分散）。
- 2・6 月: 各層が 1 回ずつ（appear 増分の二重計上なし、06-11 は AIR のみの日、EDDM 2/19 は乱れの日で v48 の 1 月と同じ扱い、LIRF 6 月は AIR のみの受信）。
- 残る軽微な点: 60 s 未満の予測 33 行（主に LSZH のヘリ。影響 < 0.001 s）。
- 残る再現性の指摘（値には無関係）: スクリプト内の絶対パス。下の対応表で読み替える。

## 再現
```
cd /home/user/prc-data-challenge/handoff/2026-10-13
python ../../docs/final_2026-10-11/derived_members.py --dir members    # pao_l6_cattight_nolirf, mlp_l6_5seeds
python ../../docs/final_2026-10-11/assemble_step.py \
  --a2o members/a2o_cattight_flfix.parquet --pao members/pao_l6_cattight_nolirf.parquet --m4 members/mlp_l6_5seeds.parquet \
  --cb members/main_flseq_osm_nolirf_cb_l2.parquet members/main_flseq_osm_nolirf_cb_l2_s43.parquet members/main_flseq_osm_nolirf_cb_l2_s44.parquet \
       members/cb_l2_s45.parquet members/cb_l2_s45.parquet \
  --tag final_v4_cb5_mlp5 --c13
```
パスの対応表（スクリプトは作業機のパスを直書きしている）:

| スクリプト内 | このブランチ |
|---|---|
| `/home/user/pb-work/outputs/final/*.parquet` | `handoff/2026-10-13/members/` |
| `adsb_layer.py` の `SCR`（scratchpad/adsb）の `learn/feat_2026`, `learn/feat_pos_2026`, `pos/dwell_2026`, `pos/v2026.*_pos`, `agents/rescat/{codes,vocab}` | `handoff/2026-10-13/adsb_store/{learn,pos,rescat}/` |
| `adsb_out/feat`, `adsb_out/pos`（2・6 月の日別） | `handoff/2026-10-13/adsb_store/{feat,posday}/` |
| `adsb_out/models` | `docs/final_2026-10-11/adsb_models/` |
| `overlays/val/copyq_final_unfixed.parquet` | `inputs/copyq_LIRF_final_unfixed.parquet` |
| `overlays/out/{nmb,ucmix}_laneB_delta_final.parquet` | `inputs/` |
| `assemble/base_blend.parquet`, `assemble/blend_adsb2_ovl.parquet`（旧 chain） | `pre_adsb_base_blend5_final.parquet`, `inputs/orig_chain_blend_adsb2_ovl.parquet` |
| `/home/user/pb-work/outputs/flfix/...pa_opt_l6.parquet`, scratchpad `copyq_LIRF.parquet`（C13） | `inputs/c13_pao_oldframe_flfix.parquet`, `inputs/c13_copyq_LIRF_oldframe.parquet` |
| `/home/user/bd-work/handoff/2026-10-08/C15_appear12_cov26eddf.parquet`（v48）、`final_adsb.py`、`handoff/2026-10-11/adsb` | `beautiful-davinci-ehugk4` @ 6b9d29e |
| `scratch/agents/audit/feat_rank_extra_union.parquet` | `inputs/feat_rank_extra_union_final.parquet` |

メンバーの学習: `peaceful-babbage-md9l2x` @ 6a97bf0 のコードで `prc.pipeline --skip-cv --raw-dir <final raw> --work-dir <work_final>`（設定は `docs/final_2026-10-11/configs/`、ログは `docs/final_2026-10-11/logs/`）。a2o だけは `flfix_member_final.py <config> 3 <out>`。
md5 の一覧: `MD5SUMS`（335 件）。

## このスレッドは何もアップロードしていない
チーム指示どおり、`_final.parquet` も `_vNN.parquet` もバケットには置いていない。提出は RPC 実装スレッドが行う。

## 2026-10-14 02:10 UTC: 改善を継続中（ユーザー指示: 締めるまで続ける）
実行中の項目。できたものから点検して `final_recommended.parquet` を差し替え、ここに追記する。差し替えがなければ上の ★★ の md5 が最終。
1. 自前 ADS-B 層のモデル（m12a / m7a / m4v2）を seed 42・43・44 の 3 本で学習し、補正 c を平均する（`adsb_layer.py` に `ADSB_SEED` / `ADSB_MODELS` を追加、`assemble_step.py --adsb-models`）。
   seed 違いの差は RMS 約 8 s（調整ワーカーの v2 refit と本番モデルの比較）。見込み −0.05 s 前後。
2. a2o（cat_tight）・pao（cat_tight、LIRF 以外）・p2o の seed 45・46 を足して 5 seeds 平均に（各 −0.01〜−0.02 s）。
3. 時間があれば cat_vtight（200/200/1000）の screen。

## 2026-10-14 03:00 UTC: v5 = v4 ＋ ADS-B 層を 3 seeds 平均 ＋ a2o を 5 seeds 平均（点検を実行中。推奨はまだ v4）
- `final_v5_adsb3_a2o5.parquet`（md5 411a6aa0b05b8e3b525d3d2bf4e17ea9、validate OK）、1・7 月だけ: `janjul_only_final_v5_adsb3_a2o5.parquet`（6daa4e6e2eeee0f4a79ee6cf055b9cb0）。
- 自前 ADS-B 層のモデル m12a / m7a / m4v2 を seed 43・44 でも学習（`docs/final_2026-10-11/adsb_models_s43`, `_s44`、ログ `logs/fit_s4{3,4}.log`）し、3 本の出力を平均。
  seed 違いの出力差は全行 RMS 12.5〜12.7 s → 3 本平均で分散が約 1/3 に（見込み −0.1 s 前後）。
- a2o（cat_tight）に seed 45・46 を足して 5 seeds 平均（`members/a2o_cattight_5seeds_flfix.parquet` = (3×seed42–44 + 2×seed45–46)/5、seed 45–46 は `members/a2o_cattight_s45_flfix.parquet`）。a2o 単体 mean5 − mean3 の RMS 9.7 s。
- v4 比の変化: 1 月 RMS 9.1 s、2 月 7.3、6 月 6.9、7 月 6.7。
- 実行: `assemble_step.py --a2o members/a2o_cattight_5seeds_flfix.parquet --pao members/pao_l6_cattight_nolirf.parquet --m4 members/mlp_l6_5seeds.parquet --cb <5 本> --adsb-models <models> <models_s43> <models_s44> --tag final_v5_adsb3_a2o5 --c13`。


## 2026-10-14 03:40 UTC: 推奨を v5 に切り替え（独立点検 合格）
- **`final_recommended.parquet` = `final_v5_adsb3_a2o5.parquet`（md5 411a6aa0b05b8e3b525d3d2bf4e17ea9）**、1・7 月だけ: `janjul_only_final_recommended.parquet`（6daa4e6e2eeee0f4a79ee6cf055b9cb0）。
  任意の変種 `optional_janjul_reanchor_plus1delta.parquet` も v5 から作り直した。
- v4 からの変更: 自前 ADS-B 層を 3 seeds 平均、a2o を 5 seeds 平均。
- **1・7 月の unmatched 行について（★★ の記述の訂正）**: v4 までは unmatched 5,290 行が v48 と同一だったが、v5 では ADS-B 層の seed 平均が unmatched の ground-fix 行にも効くため、1,765 行が変わる（RMS 13.9 s、平均 +0.4 s、max 173 s）。
  v48 の層はそれらの行で seed 42 の 1 本だけだったので、平均に置き換えるのは設計どおり（意図した変更）。matched 行の v48 との差は RMS 12.9 s、平均 −0.12 s、max 573 s（LIRF 1/31、CatBoost / MLP の seed 平均）。
- 点検結果: 形式・網羅 合格、LOBT 違反 0、2・6 月の層は 1 回ずつ、Jan/Jul の変化はすべてメンバー差分と層の seed 平均で説明がつく。再現性の指摘（AIR/POS 用の保存モデル 2 本が支店に無い）→ `adsb_store/learn/final_model_months_A.txt`, `final_model_7v2.txt` を追加した。
- 実行コマンド（literal）:
```
cd /home/user/prc-final && F=/home/user/pb-work/outputs/final
python assemble_step.py --a2o $F/a2o_cattight_5seeds_flfix.parquet --pao $F/pao_l6_cattight_nolirf.parquet --m4 $F/mlp_l6_5seeds.parquet \
  --cb $F/main_flseq_osm_nolirf_cb_l2.parquet $F/main_flseq_osm_nolirf_cb_l2_s43.parquet $F/main_flseq_osm_nolirf_cb_l2_s44.parquet $F/cb_l2_s45.parquet $F/cb_l2_s45.parquet \
  --adsb-models /home/user/prc-final/adsb_out/models /home/user/prc-final/adsb_out/models_s43 /home/user/prc-final/adsb_out/models_s44 \
  --tag final_v5_adsb3_a2o5 --c13
```
  stdout は `docs/final_2026-10-11/logs/assemble_final_v5_adsb3_a2o5.log`。派生メンバー（pao nolirf、MLP 5 seeds、a2o 5 seeds）は `derived_members.py`。
- 進行中: pao（cat_tight、LIRF 以外）と p2o の seed 45・46。できたら v6 として点検し、通れば差し替える。

## 2026-10-14 04:25 UTC: v6 = v5 ＋ pao・p2o も 5 seeds 平均（全メンバー 5 seeds）。点検を実行中（推奨はまだ v5）
- `final_v6_all5.parquet`（md5 7e53142e478bbb022ae6584850143040、validate OK）、1・7 月だけ: `janjul_only_final_v6_all5.parquet`（55a91f49410b93c9d97a774d76afdab2）。
- pao（cat_tight、LIRF 以外）= (3×seed42–44 + 2×seed45–46)/5（`members/pao_l6_cattight_5seeds_nolirf.parquet`、LIRF は本番 pao のまま）。pao 単体 mean5 − mean3 の RMS 9.9 s（LIRF 以外）。
- p2o = (3×seed42–44 + 2×seed45–46)/5（`members/p2o_l6_5seeds.parquet`、`configs/p2o_l6_s45.yaml`）。p2o 単体 mean5 − mean3 の RMS 10.8 s。
- v5 比の変化: 各月 RMS 2.8〜3.6 s。stdout は `docs/final_2026-10-11/logs/assemble_final_v6_all5.log`。
- 並行して cat_vtight（cat_smooth 200 / cat_l2 200 / min_data_per_group 1000）を 2025 年の 4 fold（1・2・6・7 月）で screen 中。


## 2026-10-14 04:45 UTC: 推奨を v6 に切り替え（独立点検 合格: blocker / major なし）
- **`final_recommended.parquet` = `final_v6_all5.parquet`（md5 7e53142e478bbb022ae6584850143040）**、1・7 月だけ: `janjul_only_final_recommended.parquet`（55a91f49410b93c9d97a774d76afdab2）。
  任意の変種 `optional_janjul_reanchor_plus1delta.parquet` も v6 から作り直した。
- 点検結果（4 観点）: 形式・網羅 合格（件数・ID・順序・型すべて一致、LOBT 違反 0）。1・7 月は v48 相当（matched RMS 15.2 / 11.3 s、unmatched は v5 と同じ 1,765 行が ADS-B 層の seed 平均で変化）。
  2・6 月は chain の中間ファイルから全行が正確に再現（0 行の不一致）。残る軽微な点は 60 s 未満 31 行（Jan/Jul は v47/v48 と同数の 9 行）。
- v5 → v6 の変更: pao（LIRF 以外）と p2o を 5 seeds 平均に。各月 RMS 2.8〜3.6 s の変化。
- 実行コマンド（literal）: `docs/final_2026-10-11/logs/assemble_final_v6_all5.log` の先頭を参照。
```
cd /home/user/prc-final && F=/home/user/pb-work/outputs/final && python derived_members.py --dir $F
python assemble_step.py --a2o $F/a2o_cattight_5seeds_flfix.parquet --pao $F/pao_l6_cattight_5seeds_nolirf.parquet --p2o $F/p2o_l6_5seeds.parquet --m4 $F/mlp_l6_5seeds.parquet \
  --cb $F/main_flseq_osm_nolirf_cb_l2.parquet $F/main_flseq_osm_nolirf_cb_l2_s43.parquet $F/main_flseq_osm_nolirf_cb_l2_s44.parquet $F/cb_l2_s45.parquet $F/cb_l2_s45.parquet \
  --adsb-models /home/user/prc-final/adsb_out/models /home/user/prc-final/adsb_out/models_s43 /home/user/prc-final/adsb_out/models_s44 --tag final_v6_all5 --c13
```
- 進行中: cat_vtight の screen（2025 年 1・2・6・7 月）。cat_tight より一貫して良ければ a2o / pao を学習し直す。

## 2026-10-14 05:05 UTC: cat_vtight の screen 結果 → 採用しない
anchored LightGBM、単一 seed 42、matched RMSE の base からの差（s）:

| 設定 | 1 月 | 2 月 | 6 月 | 7 月 | 平均 |
|---|---|---|---|---|---|
| cat_tight（採用中） | −1.08 | −0.55 | −0.82 | −1.00 | −0.86 |
| cat_vtight（200/200/1000） | −1.24 | −0.49 | −0.60 | −1.24 | −0.89 |

- 1・7 月は vtight、2・6 月は tight が良く、平均は同等（seed のばらつき 0.13〜0.3 s の範囲）。vtight は 2000 round の上限に達している。
- 一貫した改善ではないので、a2o / pao の学習し直し（約 2.5 時間）はしない。cat_tight のまま。
- 続けて: ADS-B 層の seed 45・46（3 → 5 本平均、見込み −0.02 s）、MLP と CatBoost の seed 47・48（各 −0.02〜−0.03 s）。

## 2026-10-14 06:45 UTC: v7a = v6 ＋ ADS-B 層を 5 seeds 平均 ＋ MLP を 7 seeds 平均（点検を実行中。推奨はまだ v6）
- `final_v7a_adsb5_mlp7.parquet`（md5 45ffc9e9f1d95f60f04e850b21377f35）、1・7 月だけ: `janjul_only_final_v7a_adsb5_mlp7.parquet`。
- v6 からの変化: 各月 RMS 4.2〜5.6 s（ADS-B 層の seed 45・46 と MLP の seed 47・48 を足しただけ）。C13 の検査は合格（e² 平均 1.21、max|e| 82.3）。
- 外部知見スレッドの補正（門・q15・+1×δ）は `build_on_base.py` で v6 から作ったファイルと md5 まで一致することを確認した（6 ファイルすべて）。v7a の上でも同じスクリプトで作れる（推奨を切り替えるときに一緒に置く）。
- 実行中: CatBoost の seed 47・48（07:40 ごろ終わる）→ ADS-B 層の seed 47・48 → v7b（CatBoost・ADS-B 層とも 7 seeds）。
