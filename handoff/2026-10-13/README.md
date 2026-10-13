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
