# 最終 4 か月の候補ファイル（beautiful-davinci、2026-10-14）

**状態（03:05 UTC）**: 補正 1（LIRF の門）と補正 2（LIRF の overlay の重み直し）の 2 つ。どちらも 2025 年で検証し、独立した点検を通過。
**ユーザーの指示（03:00 UTC）: 補正 2 は「試してみて、スコアが落ちすぎないなら入れる」。** そのため、旧ボード（1・7 月、旧時計）で
試すための v48 ベースの試験ファイルを `oldboard/` に用意した。手順は下の「旧ボードでの試し方」。

## 提出役へ: 決め方

1. 旧ボードに `oldboard/v48_plus_gate.parquet`（補正 1）と `oldboard/v48_plus_gate_q15.parquet`（補正 1 + 2）を上げ、v48（234.4099）と比べる。
   番号はバケットを一覧して最大 + 1 を使う（_vNN）。
2. 結果で最終ファイルを選ぶ（**全ファイルとも 1・7 月の +1×δ 入り。`add_delta_janjul.py` を重ねないこと**）:

| 旧ボードの結果 | 最終ファイル | md5 |
|---|---|---|
| 補正 1 も 2 も悪化しない | `final_v4_gatefull_q15_plus1delta.parquet` | `d32ca3145848c9cc15888a341547d90c` |
| 補正 1 は悪化せず、2 は悪化 | `final_v4_gatefull_plus1delta.parquet` | `b7e8298d16f7e36927f8dc98738513b6` |
| 補正 1 は悪化、2 は悪化せず | `final_v4_q15_plus1delta.parquet` | `063ed43d0d13375a2252a761efac3f4c` |
| どちらも悪化 | 統合役の `optional_janjul_reanchor_plus1delta.parquet` | `9d0a5bf1b814e4a0cf69ac0d9d69f3c8` |

- 「悪化」の目安: v48 → 補正 1 の差、補正 1 → 補正 1 + 2 の差が、それぞれ **+0.00 s を超えて悪くなる**こと。
  見込みは補正 1 が −0.1〜−0.2 s、補正 2 が −0.2〜−0.4 s（1 月はほぼ 0、ほとんど 7 月）。
  ±0.05 s 程度は当たり外れの範囲だが、悪化していたら入れない（ユーザーの指示に従う）。
- 旧ボードに上げられない場合は、堅い方（`final_v4_gatefull_plus1delta.parquet`）を本命にする。
- 旧ボードは旧時計なので、+1×δ の効果はここでは測れない（別件、決定済み）。

## ファイル一覧

| 用途 | ファイル | md5 |
|---|---|---|
| 補正 1 + 1・7 月 +1×δ（堅い） | `final_v4_gatefull_plus1delta.parquet` | `b7e8298d16f7e36927f8dc98738513b6` |
| 補正 1 + 2 + 1・7 月 +1×δ（攻め） | `final_v4_gatefull_q15_plus1delta.parquet` | `d32ca3145848c9cc15888a341547d90c` |
| 補正 2 だけ + 1・7 月 +1×δ | `final_v4_q15_plus1delta.parquet` | `063ed43d0d13375a2252a761efac3f4c` |
| +1×δ なし: 補正 1 / 補正 1 + 2 | `final_v4_gatefull.parquet` / `final_v4_gatefull_q15.parquet` | `b228a45dbb38315e55ea6fb1d4be4fa2` / `abf0332842da2de1c6f594c74269a3b9` |
| 控え: 補正 1 を 2・6 月だけ、+1×δ あり / なし | `final_v4_gatefj_plus1delta.parquet` / `final_v4_gatefj.parquet` | `4e684f9d88c65aecc8a99f374c9b920d` / `a664aa7571e35005a6b9bb4608ee7bdf` |
| 旧ボード試験: v48 + 補正 1 | `oldboard/v48_plus_gate.parquet` | `b8abea954514217d984d5cf4f9787993` |
| 旧ボード試験: v48 + 補正 1 + 2 | `oldboard/v48_plus_gate_q15.parquet` | `cfd858f888e7912a9f28eaf38125dcba` |

- **ベース**: 統合役の推奨 v4（`laughing-cray:handoff/2026-10-13/final_recommended.parquet`、md5 d0ce32b68f3a7a4599d8a0982ca81988）。
  `_plus1delta` は統合役の `optional_janjul_reanchor_plus1delta.parquet`（md5 9d0a5bf1…）に補正をかけた結果と一致する（順番を入れ替えても同じ）。
- 統合役が v4 以外を最終に選んだ場合は使えない。`docs/final_phase_gate/build_gate.py`（補正 1）と
  `docs/final_phase_gate/apply_q15.py`（補正 2、差分表 `lirf_overlay_q15_rows_v4.parquet` を足す）で載せ直す。
- 最終用は全ファイル `final_submitting.parquet` と同じ形（670,790 行、行順同じ、Int32、欠損・負値なし）。
  旧ボード用は `submitting.parquet` と同じ形（344,841 行）。LOBT の制約 |y − (MVT − LOBT)| ≤ 3606 s を破る行は 0。

## 補正 1: LIRF の copy overlay を「コピーがありえない行」で止める

- LIRF の matched の行のうち |LOBT − SCHED| > 3606 s のもの（2,486 行: 1 月 471、2 月 297、6 月 639、7 月 1,079）は、
  LOBT の恒等式から、正解がスケジュールのコピー（y ≈ MVT − SCHED）になることがありえない。
- そこで、copy overlay p' = p + 0.4·q·(ms − p) の q を 0 にして、overlay の引っぱりを外す。
- 変わるのはこの 2,486 行だけ。全行で ms から離れる向き、最大 3,709 s、Σd²/N 71.7 s²。
- 2025 年（正解あり）: dSSE −119 M s²、日単位ブートストラップ 95 % 区間 [−187, −72] M、12 か月すべて改善、2・6 月 −39 M。
  2025 年の「ありえない」行 66,665 行（全空港）にコピーは 0 行。
- 見込み **−0.15〜−0.27 s**。効果は約 10 行で決まるので幅は大きいが、2025 年の同種の行（q > 0.4 の 7 行）はすべて改善した。
- 独立した点検 3 件（行の再導出、2025 年の根拠、ファイルの整合）すべて合格。

## 補正 2: LIRF の copy overlay の引っぱりを 0.4·q から q^1.5 に

- 統合役の overlay は p' = p + 0.4·q·(ms − p)。0.4 は「2025 年の月またぎの最適値が 0.25 と 0.65 だったので、前もって 0.4 に固定」したもの
  （laughing-cray `analysis/local_lirf_copy_overlay.py`）。
- q が較正されている（q の区分ごとのコピー率が q と 0.02 以内で一致）なら、二乗誤差で最適な点は p + q·(ms − p) に近い。
  実際には区分ごとの最適な重みは q が小さいほど q より小さく、q > 0.7 で q に近づく（凸）。q^1.5 がその形。
- 対象: 補正 1 の門の行を除く LIRF の matched の行 45,908 行（1 月 9,720、2 月 9,204、6 月 13,302、7 月 13,682）。
  v4 から 0.4·q の overlay を逆算して p_pre を出し、p_pre + q^1.5·(ms − p_pre) に置き換える（負なら p_pre、LOBT クリップ、0 以上）。
  Σd²/N 162 s²、最大 +1,805 s（6 月の 1 行、q 0.93、4,308 → 6,113）。
- 2025 年（final_12 を土台、門とクリップあり、LOMO の q）: dSSE **−504 M s²**、日単位ブートストラップ [−582, −430] M、
  **12 か月すべて改善**、2・6 月 −68 M [−102, −38]、1・2・6・7 月 −136 M。
  LOMO で形（係数と指数）を選ぶと 11/12 の月で q^1.5 が選ばれる。半年で選んで残り半年で試すと −319 M / −171 M。
  比較: 定数 0.8·q は −336 M（1 月は +17 M で悪化）、q^2 は −453 M（1 月 −13 M）。
- 見込み **−0.2 s**（幅 −0.18〜−0.40 s）。
- **注意（当たり外れ）**: 2025 年の利得の 85 % は上位 1 % の行（q > 0.9 で |ms − p| が 1,000〜3,000 s）に集中。
  2026 年も変化量の 86 % は上位 1 %（459 行）、41 % は 46 行。q > 0.9 の行の損益分岐のコピー率は約 0.6、2025 年の実測は 0.985。
  2026 年の出発便のコピー率は確かめられない（到着便のコピー率は 2025 年と同じ 9〜11 %、2026 年の q の分布も 2025 年と同じ）。
  1 月は 2025 年でほぼ中立（−1.3 M）で、時間を空けた検証では +6〜+25 M と悪化する。
- 統合役の 0.4 の overlay と、その q（classifier の入力に proxy_sec_in_min と minute_of_day_utc が含まれる）をそのまま使っている。
  引っぱりを強めるぶん q への依存は増える。q 自体はすでに採用済みの部品。
- 反証役 2 名（統計、適用可能性）とも棄却せず。v4 に 0.4·q の overlay が入っていること（傾き 0.386）、差分表が正確なこと、
  門の行と重ならないこと、ずらし（+1×δ）と順番を入れ替えても同じことを確認。
- スクリプトとログ: `docs/final_phase_gate/q15/`（`an25.py` 2025 年の形の比較、`boot25.py` ブートストラップと LOMO、
  `apply26.py` 2026 年の差分表、`refute_*` 反証役の再計算）。適用は `docs/final_phase_gate/apply_q15.py`。

## 旧ボードでの試し方（v48 ベース、1・7 月、旧時計）

- `docs/final_phase_gate/oldboard_test.py` で v48（md5 8ee6679f43ead60db1a1aec7876caf09）から作成。
  v48 の overlay は旧フレームの q（`c13_copyq_oldframe.parquet` の q_orig、26,516 行）で 0.4·q。それを逆算して補正をかけ、旧時計の LOBT でクリップ。
  逆算は近似（v48 の鎖は overlay を RMS 約 4 s で再現）。
- `v48_plus_gate`: 1,550 行変更（1 月 471、7 月 1,079）、Σd²/N 79.5 s²。見込み −0.1〜−0.2 s（2025 年の 1・7 月の相当分は −62 MSE ≈ −0.13 s）。
- `v48_plus_gate_q15`: さらに 23,439 行変更、Σd²/N +183 s²。見込み −0.2〜−0.4 s（2025 年の 1・7 月の相当分は −197 MSE ≈ −0.42 s、1 月はほぼ 0）。
- 事前登録: 「悪化しない」を採用条件とする。見込みより小さくても、悪化していなければ入れる。

## 再現

- 補正 1: `docs/final_phase_gate/`（`imposs25.py` 2025 年の検証、`imposs26c.py` 2026 年の行、`build_gate.py` ファイル作成）。
- 補正 2: `docs/final_phase_gate/apply_q15.py <差分表> <入力> <出力>`。差分表は v4 で計算したものを足すだけなので、
  gate や +1×δ の後に足してよい（最後に LOBT クリップと 0 床を 1 回かけ直す）。ずらした後のファイルから overlay を逆算しないこと。
- 点検で各スクリプトを再実行し、全ファイルの md5 が一致した。
