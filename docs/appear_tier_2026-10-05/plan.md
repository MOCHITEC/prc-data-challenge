# ADS-B の appear tier（自スタンドとの位置関係）を学習 ADS-B 残差に足す — 事前登録（2026-10-05）

結果（残差の CV）を見る前に書く。首位との差の調査の推奨 A（`docs/research_2026-10-01/gap/HANDOFF_PROMPT.md`）。

## 0. 何を足すか

stage 1（`scripts/adsb_stage1_feasibility.py`）は、off-block を「止まっている区間（stand run）」でしか見つけない。
- 最初に見えたときすでに動いている便（`moving_when_first_seen`、5 空港の採点行の 18〜26 %）は、`t_move` が無い。
- stand run があっても、それが自分のスタンドか（プッシュバック後のエンジン始動の停止ではないか）を確かめていない。

`scripts/adsb_appear.py` が、OpenStreetMap の parking_position（`STAND_mvt` と照合、全体の約 9 割）との位置関係を列にする:

| 列 | 中身 |
|---|---|
| `own_ok` | `STAND_mvt` が OSM のスタンドと照合できたか |
| `d_dep1_own`、`d_dep1_near` | `t_dep1` の fix（トランスポンダが出発のために入った最初の fix）から自スタンド／最寄りのスタンドまでの距離 |
| `d_stand_own` | stand run の最後の fix から自スタンドまでの距離 |
| `n_in_own`、`t_leave_own_lo`、`t_leave_own` | 自スタンドの 50 m 以内の fix の数、最後のその fix、その次の fix（位置による離脱） |
| `t_appear_adj` | `t_dep1` − min(`d_dep1_own`, 3 km) / 6 m/s（最初の fix をスタンドまで戻した時刻） |

時刻の 3 列は、既存の stage 1 のイベントと同じく MVT − t、MVT − t − base、t − AOBT_3 にして入れる（14 列）。ラベルは読まない。

## 1. 下見（ラベルを使った、モデル無しの確認。2025-01-01〜04 の 4 日）

stand run の最後の fix から自スタンドまでの距離ごとの `t_move − BLOCK`（中央値 / IQR、秒）:

| 空港 | ≤ 50 m | 50〜300 m | > 300 m |
|---|---|---|---|
| LSZH | 35 / 24（n 421） | 330 / 223（208） | 382 / 209（196） |
| EDDM | 50 / 33（517） | 245 / 195（122） | 597 / 1,525（35） |
| EHAM | 4 / 65（1,119） | 227 / 114（546） | 355 / 146（84） |
| LEBL | 34 / 42（336） | 202 / 242（251） | 462 / 172（105） |
| EDDF | 64 / 115（152） | 327 / 290（188） | 453 / 205（196） |
| EGLL | 184 / 155（305） | 424 / 304（193） | 879 / 438（212） |

最初に見えたとき動いていた便で、その fix が自スタンドの 50 m 以内のもの: `t_dep1 − BLOCK` の IQR 29〜66 s（EDDF・EDDM・LEBL・LSZH、
各 20 行前後）。ただし該当は動いていた便の 2〜11 %。位置による離脱で新たに時刻が付く行（`t_move` 無し）は 3〜10 % だが精度は低い
（IQR 186〜569 s）。2025 年 1 月は LEMD・LFPG の行がほぼ無い（adsb.lol の 2025 年の地上の被覆が薄い）。

## 2. 群（`scripts/adsb_learn_cv.py`、2025 年 1 月・7 月の 62 日、日単位の 5 fold、本番と同じ）

本番の base（`--base opt`）のメンバーの OOF はこの環境に無いので、作れる base で測る:
**`--base a2`** = マッチ行は v28 の anchored L2 メンバー（`main_flseq_osm_nolirf_l2-ef204b44ff`）、未マッチ行は `main_flseq-8615e6a194`。

| 群 | コマンド |
|---|---|
| plain | `--base a2` |
| shuffle（対照） | `--base a2 --appear-shuffle`（同じ列を空港の中で行の間で並べ替え。時刻は各行の MVT からの差として並べ替える） |
| appear | `--base a2 --appear` |

## 3. 判定

**採用** = 次の 2 つを両方満たす（`adsb_appear.report()` が日単位の bootstrap で計算して表示）。
1. appear 対 plain: 全行（2025 年 1 月・7 月のラベルのある行すべて）の RMSE の差の 95 % CI が改善側でゼロを除外する。
2. appear 対 shuffle: 同じく改善側でゼロを除外する（列が増えただけの揺れではないこと）。

「最初に見えたとき動いていた行」「自スタンドで最初に見えた行」と空港ごとの差は説明用で、判定には使わない。

## 4. 採用なら

- 本番の base（`--base opt`）で同じ 3 本を走らせて同じ判定をする（ADS-B の素材とメンバーの OOF がある環境）。引き継ぎに手順を書く。
- `adsb_learn_apply.py --appear` の parity の行で、空港ごとの「自スタンドで最初に見えた割合」「位置による離脱がある割合」を
  2025 年と 2026 年で比べる。2026 年は LEMD・LFPG・EGLL・EDDM の受信が密になっている（adsb-learned-residual.md）ので差が出うる。
  差が 2 倍を超える空港は列を空にすることを検討する（その場合は事前に書いてから）。

## 5. ライセンス

adsb.lol は ODbL 1.0、OpenStreetMap は ODbL 1.0。どちらもオープン（賞の条件を満たす。表記は `docs/external_data.md`）。
