# appear tier を本番の学習 ADS-B 残差に入れる — 引き継ぎ（2026-10-05、未提出）

`docs/research_2026-10-01/gap/HANDOFF_PROMPT.md` の **A**（ADS-B 検出の第 2 段）。事前登録 `docs/appear_tier_2026-10-05/plan.md`、
結果 `docs/appear_tier_2026-10-05/results.md`。

## 1. 何が効くか（この環境の CV、`--base a2`）

事前登録の判定は **採用**（`docs/appear_tier_2026-10-05/results.md`）。2025 年 1 月・7 月の 62 日、全行 344,419 行、日単位の bootstrap:

| 比較 | RMSE の差 | 95 % CI |
|---|---|---|
| appear 対 plain（同じ base の今の残差） | **−1.858 s** | [−2.542, −1.346] |
| appear 対 shuffle（同じ列を並べ替えた対照） | **−1.950 s** | [−2.674, −1.405] |
| shuffle 対 plain（参考） | +0.092 s | [−0.093, +0.275] |

空港別（appear 対 plain）: EGLL −5.5、EDDF −5.5、LSZH −5.1、EHAM −4.4、EDDM −3.2、LIRF −2.2、LEBL −2.1、LEMD −0.8、LFPG −0.3 s。
残差全体は base 316.75 → 302.77（今の残差）→ 300.91（appear 入り）。

- いちばん効くのは「stage 1 の stand run が本当に自分のスタンドか」（`d_stand_own`）と、位置で見た自スタンドからの離脱
  （`t_leave_own_lo`）。stand run が自スタンドから 50 m 以上離れているとき、それはたいていプッシュバック後のエンジン始動の停止で、
  `t_move` は本当の off-block より 3〜15 分遅い（plan.md §1）。今の残差にはそれを見分ける列が無かった。
- `--base a2` はこの環境で作れる base（マッチ行 = v28 の anchored L2 メンバー、未マッチ行 = main_flseq）。本番の `--base opt` でも同じ
  3 本を走らせて同じ判定をすること（下の手順）。

## 2. 手順（ADS-B の素材と opt のメンバーの OOF がある環境）

```bash
git fetch origin claude/beautiful-davinci-ehugk4
git merge origin/claude/beautiful-davinci-ehugk4     # scripts/adsb_appear.py と adsb_learn_cv/apply のフック
                                                     # （本線 4338035 を 6624d3d で取り込み済みなので、衝突は出ないはず）
# 0) appear の列を作る（stage 1 の <tag>_joined.parquet と <tag>.parquet から。ラベルは読まない。数分）
POLARS_MAX_THREADS=2 .venv/bin/python scripts/adsb_appear.py build
POLARS_MAX_THREADS=2 .venv/bin/python scripts/adsb_appear.py concat   # -> $SCRATCH/learn/appear_{2025,2026}.parquet
# 1) 対照と本番の群（--base opt の plain = v33 の cv_rows_opt.parquet が $SCRATCH/learn にある前提。無ければ先に --base opt）
POLARS_MAX_THREADS=2 .venv/bin/python scripts/adsb_learn_cv.py --base opt --appear-shuffle
POLARS_MAX_THREADS=2 .venv/bin/python scripts/adsb_learn_cv.py --base opt --appear
```

`--appear` の最後に `DECISION (docs/appear_tier_2026-10-05/plan.md): vs plain ..., vs shuffle ... -> ADOPT / REJECT`。
採用 = 全行の RMSE の差の日単位 95 % CI が、plain と shuffle の両方に対して改善側でゼロを除外すること（事前登録どおり）。

## 3. 採用なら

```bash
POLARS_MAX_THREADS=2 .venv/bin/python scripts/adsb_learn_apply.py --base opt --appear
# -> outputs/hybrid_blend_l2_pa_opt_adsb_learned_appear.parquet
```

- `APPEAR parity <空港>: 2025 fit (at own stand, exit) (...); 2026 (...)` の行を確認する。この環境の値は PARITY_NOTE。
- v34 の位置による結びつけ（430 行）を重ねる: その 430 行は callsign の結びつけが無い未マッチ行で、残差の対象外（変わらない）。
  `handoff/2026-10-05/README.md` §4 の Python の 3 行の入力ファイル名を `..._adsb_learned_appear.parquet` に替えればそのまま使える。
- 提出前に `python -m prc.validate_submission`、版番号と当日の残り枠はバケットで確認する。

## 4. ライセンス

adsb.lol（ODbL 1.0）と OpenStreetMap（ODbL 1.0）だけ。どちらもオープンで、`docs/external_data.md` に記載済み。賞の条件の問題は無い。
