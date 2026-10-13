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
  - 2026-06-11 の行は除外（A12 の層が直接入っているため）。LTFM、LIRF の unmatched、LOBT 境界の行は増分側で触らない。
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
