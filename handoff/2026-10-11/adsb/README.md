# 最終フェーズ用 ADS-B の層（2026 年 1・2・6・7 月、670,790 行）

## 使い方

    SCRATCH=$PWD/handoff/2026-10-11/adsb POLARS_MAX_THREADS=2 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage \
      .venv/bin/python docs/appear_tier_2026-10-05/final_adsb.py <base.parquet> <out.parquet> --months 01,02,06,07

- **入力 `<base.parquet>`**: MVT_ID_mvt と TAXITIME_SEC_mvt。ADS-B の層をかける前の予測で、final_submitting の行。一部の行だけでもよい。
- **出力**: すべての行。補正しない行はそのまま写す。
- **補正**:
  - ADS-B の特徴量と appear の列がある行は b + c(b)。EDDF は c_A12thin、ほかは c_A12（12 か月 appear の残差）。
  - LTFM、LIRF の unmatched の行（AOBT_3 null）、LOBT 境界に乗った行は触らない。
  - 最後に LOBT の切り詰め（|y − (MVT − LOBT)| ≤ 3606 s）をかける。
- **動作確認**: 4 か月すべてで 481,191 行を補正、RMS c 134 s。
- 1・7 月は、以前の計算（ranking の時刻）と一致することを確認済み（差の中央値 0.3 s、整数への丸め）。

## 中身（`learn/`）

| ファイル | 内容 |
|---|---|
| `feat_2026.parquet` / `appear_fill_2026.parquet` | 1・7 月。final_ranking の時刻（±12 s ずらし後）で ADS-B と突き合わせ直したもの（`final_janjul_adsb.py`）。機体の対応は旧版と 99.999 % 同じ |
| `feat_2026fj.parquet` / `appear_fill_2026fj.parquet` | 2・6 月。adsb.lol を 58 日すべて取り込み、欠けた日なし。特徴量がある行は 270,901 行 |
| `final_model_a2_delta_m12_v2_A.txt` | 12 か月 appear の残差（2025 年 12 か月、V2、660 rounds） |
| `final_model_a2_delta_m12_v2_A_thinEDDF.txt` | 同じ。2025 年の EDDF の地上の fix を 2026 年並みに間引いて学習（`plan_cov26.md`） |

2 月の EDDF の受信は 1 月よりさらに薄い（地上の fix がある行 94 → 72 %、post_n の中央値 12 → 4）。

## 注意
- **b は ADS-B の層をかける前の予測であること。** ADS-B の補正（統合役の 12:v2・apron・posjoin など）がすでに入ったファイルに
  かけると二重になる。
- 統合役の 12:v2 + apron の層を使う場合は、この層とどちらか一方にする。
  - v48 の 1・7 月は、統合役の層の上に (c_A12 − c_P12) を足す形だった。
  - 12:v2 の層を使うなら、`final_adsb.py` の c を (c_A12 − c_P12) に替える必要がある（相談してください）。

Data: adsb.lol, Open Database License (ODbL) 1.0. "Contains data from adsb.lol, made available under the ODbL 1.0."
