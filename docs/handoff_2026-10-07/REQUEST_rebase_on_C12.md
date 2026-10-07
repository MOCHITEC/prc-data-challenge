# 依頼: 最新ベスト (C12 = v42, LB 236.4369) の上で ADS-B 層を作り直す

依頼元: claude/model-b-and-arrival-features のセッション (2026-10-07)

## 事実
- 現在のチームベストは v42 = laughing-cray の C12 (commit 0b16e18, docs/batch_2026-10-06)。上位10%ライン(23位)は 235.5926、差 0.84 秒。
- 古いベースで計算した差分を C12 系に足した提出は 2 回とも効かなかった。
  - v42(このセッション版, 上書きされ記録消失): v40 + R6/cbopt ベース差分 → 238.7644 (v40 比 −0.007 秒)
  - v46: C12 + 12か月 appear 差分 (dd6e60d の表をそのまま加算) → 236.4631 (C12 比 +0.03 秒)
- 原因と考えられる点: appear 差分は v39 のベース b で評価済み。C12 では該当行の b が RMS 22.9 秒動いている (docs/competitor_research_2026-09-21/results/c12-plus-appear12.md, commit 9c881cc)。
- このコンテナには 2025 年 ADS-B が 1月・7月(+11/12 の1日)しかなく、4/7/12か月版の残差モデルや 12か月 appear を C12 ベースで評価し直せない。

## お願い
1. 12か月 appear tier (beautiful-davinci 3a7af5a: ADOPT −0.804 s [−0.924, −0.692]) を、C12 のベース b で評価し直した差分として C12 に乗せる。差分は c_A12(b) − c_P12(b) − (c_A4 − c_P4)(b) を C12 の b で計算する。
2. 以降、ベース(①層)を変える改善は、その上の ADS-B 残差・appear 層(②③)を同じベースで再適合してから提出する。
3. 提出前に `python -m prc.submit --check-version N` で空き番号と当日枠を確認する (このブランチ commit 124861b で上書き防止ガードを追加済み。バケットとリーダーボードの両方を確認し、届かなければ提出を拒否)。

## 参考 (このセッションの成果、①層)
- R6 (5モデル LOFO 重み): OOF −0.32 s [−0.44, −0.24] vs hybrid_blend_l2_pa_opt_mlp。
- CatBoost 再調整 (trial 7): 単体 −1.86 s、R6 に入れて −0.07 s [−0.14, −0.01]。
- ただし laughing-cray のメモでは「R6 を L6/CatBoost メンバーの上に重ねない」とあるため、C12 のメンバー構成に合うかは要確認。
- 作り方: docs/handoff_2026-10-05/R6_for_v38_chain.md, docs/competitor_research_2026-09-21/results/catboost-retune.md
