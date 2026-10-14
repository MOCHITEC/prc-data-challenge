# v7a の上の補正（先回りで作成、2026-10-14 06:50 UTC）

**統合役が推奨を v7a に切り替えたときだけ使う**（今の推奨は v6、`../v6/`）。統合役も同じスクリプトで作って一緒に置く予定（v6 では md5 まで一致を確認済み）。

- ベース: `laughing-cray:handoff/2026-10-13/final_v7a_adsb5_mlp7.parquet`（md5 45ffc9e9f1d95f60f04e850b21377f35）。
- 作り方: `python -I docs/final_phase_gate/build_on_base.py <v7a> - handoff/2026-10-14/v7a v7a`
  （v7a 用の +1×δ 版はまだ配られていないので、スクリプト内で v6 のときと同じ規則で作った: `v7a_plus1delta_built_here.parquet`）。
- 中身は v6 と同じ: 門 2,486 行（Σd²/N 71.7）、q15 45,920 行（Σd²/N 159.9）、LOBT の制約違反 0、validate OK。

| 旧ボードの結果 | 最終ファイル（v7a の上） | md5 |
|---|---|---|
| 補正 1 も 2 も悪化しない | `final_v7a_gatefull_q15_plus1delta.parquet` | `6cb95e1d74a4be88353eff63036901cd` |
| 補正 2 は悪化、または試せない | `final_v7a_gatefull_plus1delta.parquet` | `bc4da1e7ad5b98b0553ee673fa49aa12` |
| 補正 1 は悪化、2 は悪化せず | `final_v7a_q15_plus1delta.parquet` | `042081318844652b2f30b39a927325a7` |

旧ボードの試験（`../probes/`）の補正は `assemble_final.py ... --base v7a` で同じようにかけられる。
