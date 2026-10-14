# LIRF の copy overlay を「コピーがありえない行」で止める補正（beautiful-davinci、2026-10-14 01:20 UTC）— **点検中**

**状態**: 独立した点検（3 観点）を実行中。結果が出たら、この README を更新して推奨を書く。それまでは使わないこと。

- **ベース**: 統合役の推奨 v4（`laughing-cray:handoff/2026-10-13/final_recommended.parquet`、md5 d0ce32b68f3a7a4599d8a0982ca81988）。
- **中身**: LIRF の matched の行のうち |(MVT − SCHED) − (MVT − LOBT)| > 3606 s のものは、LOBT の恒等式からスケジュールコピーの正解がありえない。
  そこで、overlay p' = p + 0.4·q·(ms − p) の q を 0 にする。
  - 2025 年の検証: 12 か月とも改善、2・6 月も改善。
- **ファイル**: `final_v4_gatefull*.parquet`（4 か月）、`final_v4_gatefj*.parquet`（2・6 月だけ）。
  `_plus1delta` は、ユーザーの判断（1・7 月 +1×δ）を足したもの。md5 は `MD5SUMS`。
- スクリプト: `docs/final_phase_gate/`。
