# データ分析（laughing-cray）の最終候補（2026-10-13 00:50 UTC）

チーム指示（`claude/model-b-and-arrival-features:docs/final_phase_2026-10-11/NO_FINAL_UPLOAD.md`）に従う。
- このスレッドはバケットに何もアップロードしていない。
- バケットの `resilient-hedgehog_final.parquet`（10-09）はこのスレッドが作ったものではない。
- v48・v49 もこのスレッドの提出ではない。

## ファイル

| ファイル | md5 | 中身 |
|---|---|---|
| `final_from_v48.parquet` | c7ffa31edc2a4de5f1837682d90e14cb | 1・7 月 = v48 のファイル（旧ランキングの時刻）。2 月 = 下の chain（自前の ADS-B 層）。6 月 = 下の chain（ADS-B 層は外部知見の A12 層）。validate OK |
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
