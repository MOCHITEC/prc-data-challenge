# LIRF 照合なし行のゲート付き書き写し混合（ucmix）: 事前登録（2026-10-07、実行前）

## 出所
誤差分解の分析（サブエージェント、OOF のみ、2026-10-07）。現行最良 OOF（672,428 行、RMSE 269.69）の SSE を分解した結果:
- 照合なし（unmatched）の LIRF・通常行は 296 行で SSE の 8.0%。
- LIRF の照合なし行は二峰性。44% が y = MVT − SCHED（書き写し）で、残りは通常の taxi（約 900 秒）。
  スケジュール再パラメタ化した model B は MVT − SCHED 寄りに予測するので、遅延便（MVT − SCHED が 2〜4 時間）で大外しする。
- OOF 上の後付け試算（leave-month-out の分類器 + 非書き写し回帰、0 < MVT − SCHED < 8 h の行だけ、0.5 ブレンド）:
  全体 −2.91 秒。4 か月すべて改善。カットオフ 6/8/10/12 h で −3.11/−2.91/−2.93/−2.01、ゲート無しだと +13.7（悪化）。
- 2026 との分布差: LIRF 照合なし行は 397 vs 383 行。単一列の AUC は 0.524 以下。便名単位の率は移らないので使わない。

## 変更
`ucmix_airports: [LIRF]`、`ucmix_max_s: 28800`、`ucmix_weight: 0.5`（`train._ucmix_blend`）。
- p は LIRF の学習側にあるラベル付き出発すべて（両レーン、レーンフラグ付き）で学習した分類器から得る。入力は model B の列。
- 非書き写し回帰は、非書き写し行のうち 0 < y ≤ 7200 で学習する。
- 予測 = 0.5·B + 0.5·(p·ms + (1−p)·回帰)。対象は LIRF の照合なし行で 0 < ms < 8 h のものだけ。
- 他の行は bit 単位で同一。

## 比較
main_flseq_osm_nolirf_l2_opt_ucmix 対 base main_flseq_osm_nolirf_l2_opt-7244b3004f（全 fold）。

## 採用基準（すべて満たすこと）
1. 照合あり（matched）行と、LIRF 以外の行は bit 単位で同一。
2. LIRF 照合なし行で、airport-day ブロックブートストラップの CI が改善側で 0 を含まない。
3. jan・jul・forward の各 fold で、LIRF 照合なし行の RMSE が改善する。
4. 全行の pooled 差の CI が改善側で 0 を含まない（`prc.compare` の ADOPT）。

**選択バイアスの注記:** ゲート（8 h）と重み（0.5）は、同じ 2025 年 OOF を見てから選んだ。
ただし本実装は fold の学習側だけで分類器と回帰を作り直すので、後付け試算とは独立した再現になる。
stamp_lirf（同じ行を対象にする）とは足し算にならない。両方採用候補になった場合は、どちらか一方を選ぶ。

## Result (2026-10-07, `verdict.txt` and `compare.txt`)
**ADOPT.** opt-7244b3004f → ucmix-b581da8fb2.

| Condition | Result | Pass |
|---|---|---|
| 1. Only LIRF unmatched rows change | Matched rows and non-LIRF rows are bit-identical; 506 rows changed | yes |
| 2. LIRF unmatched airport-day CI excludes 0 | −143.6 s [−253.8, −37.7] on n = 537 | yes |
| 3. Every fold improves (LIRF unmatched) | jan −362.9, jul −138.3, forward −54.3 | yes |
| 4. `prc.compare` ADOPT (whole frame) | −1.60 s [−2.95, −0.40]; −0.49 s [−0.82, −0.16] with the worst 200 rows dropped; folds jan −1.74, jul −3.08, forward −0.41 | yes |

- The STRICT line rejects with 1/10 airports. That is expected for a single-airport change, as with cmix.
- The gain sits in the 2–4 h band of MVT − SCHED: −1.23% of whole-frame SSE.
- On the leaderboard, LIRF has 383 unmatched rows out of 345k. That is a similar or higher density than the 537 out of 672k in the OOF, so a similar per-frame effect is plausible if it transfers. **Plan on about −1 s.**
