# v28 への組み込み — 提出準備（2026-10-02、未提出）

ADS-B の素材（adsb.lol の joined ファイルと `$SCRATCH/learn/feat_2025|2026.parquet`）がある環境（v28 を作った AI）で、
下の手順を上から順に走らせると提出ファイルができる。この環境には ADS-B の素材が無いので、ここで作れる部分はすべて用意した。

## 何を組み込むか（根拠）

| 改善 | v16 の上での LB | v28 の上で必要なこと |
|---|---|---|
| `fl_nm_rate_other` の窓の修正（`fe_fl_rate_window: both`） | v26: −0.39 | 利得の **86 %** は v28 で ADS-B 残差が補正する未マッチ 2,416 行にある（v30・v31 で分解、`docs/v28_graft_2026-10-02/`）。未マッチ行の base を窓の修正版の model B にして、ADS-B 残差を学習し直す |
| 目的変数を ±1,800 s に切ったメンバー ＋ 区分ごとの重み | v27: −0.52 | v28 のマッチ行の base（3 メンバーの平均）に対して重みを決め直し、ADS-B 残差を学習し直す |

v29（L5 の変化を後から足しただけ）が +0.12 だったので、どちらも **ADS-B 残差の学習し直しまでやる**。

## 手順

```bash
git merge origin/claude/beautiful-davinci-ehugk4          # このフォルダ・scripts の変更・設定を取り込む
.venv/bin/python scripts/build_hybrid_variants.py --check # 入力の確認だけ（何も書かない）
.venv/bin/python scripts/build_hybrid_variants.py         # 4 つの base（OOF と提出形式）を書く
for v in l2m l2w l2c l2wc; do                             # l2m を最初に（l2w・l2wc の比較相手）
  POLARS_MAX_THREADS=2 .venv/bin/python scripts/adsb_learn_cv.py --base $v
  POLARS_MAX_THREADS=2 .venv/bin/python scripts/adsb_learn_apply.py --base $v
done
```

`build_hybrid_variants.py` が読むもの: `experiments/oof_main_flseq_osm_nolirf{,_plainy,_cb}_l2-*.parquet`（v28 の 3 メンバー、
そちらの環境にある）、`experiments/oof_main_flseq-8615e6a194.parquet`、`outputs/hybrid_blend_l2.parquet`（v28 の ADS-B 前の base）、
このフォルダの parquet。

| base | マッチ行 | 未マッチ行 | `adsb_learn_cv.py` の S4 の比較相手 |
|---|---|---|---|
| `l2m` | 3 メンバーの平均（v28 と同じ） | v28 と同じ model B。ただし OOF は**ランキングと同じ窓で検証した**もの | `l2` |
| `l2w` | 同上 | 窓の修正版の model B | **`l2m`**（公平な比較） |
| `l2c` | 平均 ＋ 区分ごとの重み ×（切ったメンバー − 平均） | v28 と同じ | `l2` |
| `l2wc` | `l2c` と同じ | 窓の修正版 | **`l2m`** |

**注意（重要）: `l2w` を `l2` と比べないこと。** `main_flseq` の OOF は `fl_nm_rate_other` を 12 か月の窓で検証していて、
ランキング（2 か月の窓）より楽観的になる。そのまま比べると窓の修正が CV で約 1.6 s 悪く見える（ここで確認した。
代わりの入力で `l2` 272.96、`l2m` 275.59、`l2w` 274.55）。S4 の比較相手はスクリプトの中で `l2m` にしてある。
`l2m` のランキング用ファイルは v28 の base と 1 行も違わない（検証のしかただけが違う）。

## 判定と提出

- 採用の目安: S4 の paired RMSE 差の day-bootstrap CI が改善側でゼロを除外すること（そちらの従来の基準）。
  - 窓の修正: `l2w` vs `l2m`。切ったメンバー: `l2c` vs `l2`。両方: `l2wc` vs `l2m`。
- 提出候補: `outputs/hybrid_blend_l2wc_adsb_learned.parquet`（両方が通った場合）。片方だけなら `l2w` か `l2c` の同じ名前のファイル。
- 次の版番号は **v32**（v31 まで使用済み。2026-10-02 は v28・v29・v30・v31 の 4 件を提出済み）。

## このフォルダのファイル

| ファイル | 中身 |
|---|---|
| `unm_window_oof.parquet` | 窓の修正版の model B（`flwin_both-33ed7358b9`）の未マッチ行の OOF（8,378 行、pooled の jan・jul・forward） |
| `unm_window_rank.parquet` | 同じ model B の全データ再学習の、ランキングの未マッチ行の予測（5,290 行） |
| `unm_prod_testlike_oof.parquet` | v28 と同じ model B を、ランキングと同じ窓で検証した OOF（`flwin_target-91b712363d`、8,378 行）。`l2m` 用 |
| `l2clip_oof.parquet` | `main_flseq_osm_nolirf_l2_clip`（v28 の anchored メンバー ＋ `matched_target_clip: 1800`）のマッチ行の OOF |
| `l2clip_rank.parquet` | 同じメンバーの全データ再学習の、ランキングのマッチ行の予測 |

すべてそろっている（2026-10-02 08:20 UTC）。

## 予備（近似。ADS-B 残差を作り直さない）

`outputs/fallback_v28_unm_window_all.parquet`（git には入れていない。`docs/v28_graft_2026-10-02/` の手順で作り直せる）:
v28 の未マッチ 5,290 行すべてに（v26 − v16）を足したもの。マッチ行は v28 と同一。ADS-B 残差が補正する 2,416 行では、
残差が base の変化をどれだけ吸収するかが分からないので近似。見込みは最大 −0.37 s（吸収されなければ）、吸収されれば 0、
逆向きに働けば悪化。正式な手順が回せない場合だけ使う。

## 状況

- [x] 未マッチ行の 3 ファイル、`build_hybrid_variants.py`、`--base` の 4 つの選択肢（代わりの入力で端から端まで動作確認済み）
- [x] `main_flseq_osm_nolirf_l2`（v28 の anchored メンバー）をこの環境で再現: 実験 ID `ef204b44ff` は一致、CV full 273.41
  （記録は 273.57）。マッチ行は fold ごとに 0.05〜0.26 s 良い（jan 201.25 / 201.51、jul 241.40 / 241.45、jul_fwd 250.11 / 250.19、
  forward 196.61 / 196.73）。OSM と天気のキャッシュの取り直しによる環境差（`peaceful-babbage` も 273.87 / 273.72 を報告）。
  `l2clip_*` もこの環境で作るので、そちらのメンバーとは同じ程度にずれる。重みの推定への影響は小さい見込み。
- [x] `main_flseq_osm_nolirf_l2_clip-29a6ce9882` の CV（単体 283.02）と全データ再学習 → `l2clip_oof.parquet`（664,050 行）・`l2clip_rank.parquet`（339,551 行）
- [x] 切ったメンバーが L2 のレシピでも効くかの CV 確認（b = anchored L2、c = 切った L2）: **ADOPT**、−0.32 [−0.41, −0.24]、
  K=200 −0.45、全体で 1 つの重みとの比較 −0.27（`docs/v28_graft_2026-10-02/results/stack_clip_l2.txt`）
- [x] 本物のファイルでの動作確認（3 メンバーの代わりにこの環境の anchored L2 を 3 回使った代わりの入力）:
  OOF RMSE `l2` 272.58 → `l2c` 272.26（−0.32）、`l2m` 275.21 → `l2w` 274.18（−1.03）→ `l2wc` 273.86（`l2m` 比 −1.35）。
  そちらの本物の 3 メンバーでは、平均にすでに多様性があるので `l2c` の利得はこれより小さくなりうる。
