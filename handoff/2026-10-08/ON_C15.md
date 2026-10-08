# 12 か月 appear（A）と EDDF の受信状況の差分を C15 に載せたもの（2026-10-08 06:50 UTC）

- **バケットの v47 は laughing-cray の C15（LB 234.4831、今のチームの最良）**。こちらが 01:21 に出した v47（A、236.3563）は
  01:28 に上書きされた（`docs/batch_2026-10-06/README.md` の incident）。C15 = v42 + fl_/seq_ の修正 + L6 3 seed + ucmix で、
  **12 か月 appear（A）は入っていない**。`../2026-10-07/APPEAR12_ON_C12.md` と `COV26_EDDF.md` の「v47」は A のこと。
- **`C15_appear12_cov26eddf.parquet`** = C15 + A の差分（`../2026-10-07/A_delta_vs_c12.parquet`）+ EDDF の差分
  （`cov26_eddf_delta_vs_v47.parquet`）。足し方は `docs/appear_tier_2026-10-05/c12_apply.py`（LIRF の copy band・time-zone class・
  LOBT 境界の行は触らず、LOBT の切り詰めをかけ直す）。さらに C15 の ucmix が書き換えた LIRF の行（差分と重なるのは 65 行）は外した。
  240,976 行が変わる、Σd²/N 511、平均 −1.27 s。validate OK。
- 差分だけ: `appear12_cov26eddf_delta_vs_c15.parquet`（MVT_ID_mvt、delta 整数秒、上の 65 行を除いたもの）。
- **見込み**: LB −0.13〜−0.18 s（A は v42 の上で −0.081 s。EDDF の差分は −0.05〜−0.1 s）。1 秒未満なので他の施策と束ねて出す。
