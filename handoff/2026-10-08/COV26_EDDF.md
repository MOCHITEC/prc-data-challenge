# EDDF: 2026 年の受信状況に合わせて学習し直した ADS-B の残差（v47 に足す差分）

- **何か**: 2026 年の adsb.lol は EDDF の地上の fix が 2025 年の半分以下（出発 1 便あたり 76〜125 → 35）。2025 年の EDDF の fix を
  2026 年の密度に間引いて 12 か月 appear の残差（A12）を学習し直し、2026 年の EDDF の行だけ、今の A12 との差を足す。
- **CV**（`docs/appear_tier_2026-10-05/plan_cov26.md`、results.md §18、1・7 月を丸ごと外す、2026 年並みに間引いた EDDF の行）:
  EDDF −1.13 s [−1.73, −0.46]、全行 −0.19 s [−0.39, −0.02]（EDDF 以外は今回の差分に入れない）。
- **ファイル**: `cov26_eddf_delta_vs_v47.parquet`（MVT_ID_mvt、delta 整数秒、31,774 行、EDDF のみ、Σd²/N 45、平均 −3.8 s）、
  `v47_cov26_eddf.parquet`（v47 に足したもの、validate OK、LOBT の処理は `c12_apply.py`）。
- **見込み**: LB −0.05〜−0.1 s。1 秒未満なので他の施策と束ねて出す。v47 以降のファイルにそのまま足せる（EDDF の appear の行だけ、
  v47 の ADS-B の層に対する差し替え）。
