# OPDI の駐機場離脱を学習 ADS-B 残差に足す — 引き継ぎ（2026-10-05、未実行・未提出）

`docs/research_2026-10-01/gap/HANDOFF_PROMPT.md`（peaceful-babbage）の **B**「OPDI の exit-parking_position を A と同じ残差の枠組みに
入力として足し、同じ判定をする」を、ADS-B の素材がある環境（v33・v34 を作った AI）でそのまま走らせられる形にしたもの。
この環境には ADS-B の素材が無いので、残差の CV はここでは走らせられない（偽の ADS-B 列での動作確認だけ済ませた。§5）。

## 1. 根拠（`docs/opdi_events_2026-10-05/results.md`）

- ADS-B 無しの `main_flseq` に OPDI の離脱の 5 列を足すと、離脱を観測できたマッチ行（40,524 行）で RMSE 171.3 → 142.4 s
  （−28.9 s、jan・jul・forward のすべてで −24〜−38 s）。LSZH −35.1 s、EDDF −19.5 s。
- マッチ行だけ差し替えた全行の差: **−0.91 s [−1.09, −0.75]**、K=200 −1.14（事後の確認。事前登録の全行の判定は
  未マッチ行の model B の揺れで不合格）。
- 予測側で 1/3 の日の列を空にしても、マッチ行だけ差し替えで −0.46 [−0.59, −0.33]。ただし学習時にあった離脱が予測時に無い行は
  OPDI 無しより +3.5 s（LSZH +4.8 s）悪い。ランキングで実際に欠ける 2026-01-02〜10 では全体で約 +0.02 s の見込み。
- **ただし本番の残差は、すでに adsb.lol のスタンド離脱（`res_t_move_lo` など）を使っていて、LSZH で −40 s 効いている。**
  OPDI（OpenSky の受信網）との重なりは大きいはずで、v33 に対する上乗せは小さい見込み（目安 −0.05〜−0.2 s）。
  それを測るのがこの引き継ぎ。
- 被覆（出発便のうち離脱が結びついた割合）: LSZH は 2025 年 45〜83 %、2026-01 43 %、2026-07 78 %。
  **残差の学習期間（2025 年 1 月・7 月）で OPDI があるのは実質 LSZH だけ**（EDDF 0〜1 %、LEBL 1 月 13 %）なので、列は LSZH だけに入れる
  （他の空港では空。EDDF は 2026 年に 15〜25 % あるが、学習に例が無いので木が外挿してしまう）。

## 2. 手順

```bash
git fetch origin claude/beautiful-davinci-ehugk4
git checkout origin/claude/beautiful-davinci-ehugk4 -- scripts/adsb_opdi.py handoff/2026-10-05/
git apply handoff/2026-10-05/adsb_opdi.patch     # adsb_learn_cv.py / adsb_learn_apply.py に 29 行（4338035 で --check 済み）

# 0) --base opt の残差の CV（cv_rows_opt.parquet）が $SCRATCH/learn に無ければ先に作る（v33 と同じ）
POLARS_MAX_THREADS=2 .venv/bin/python scripts/adsb_learn_cv.py --base opt
# 1) 対照: 同じ 6 列で、OPDI の値を LSZH の中で行の間で並べ替えたもの（列の数・分布・埋まる行は同じ、便ごとの情報は無い）
POLARS_MAX_THREADS=2 .venv/bin/python scripts/adsb_learn_cv.py --base opt --opdi-shuffle
# 2) 本番の群
POLARS_MAX_THREADS=2 .venv/bin/python scripts/adsb_learn_cv.py --base opt --opdi
```

`--opdi` の最後に `DECISION (handoff/2026-10-05/README.md): ... -> ADOPT / REJECT` が出る。
S4（同じ base の OPDI 無しの残差との対比較）も、いつもどおり出る。

## 3. 判定（事前登録。結果を見る前にこのとおり決める）

**採用** = 次の 2 つを両方満たす（`report()` が日単位の bootstrap で計算して表示する）。
1. `--opdi` 対 `--base opt`（OPDI 無し）: 全行の RMSE の差の 95 % CI が改善側でゼロを除外する。
2. `--opdi` 対 `--opdi-shuffle`: 全行の RMSE の差の 95 % CI が改善側でゼロを除外する（列が増えただけの揺れではないこと）。

LSZH の行、離脱がある行の差も表示されるが、判定には使わない（どこで効いたかの説明用）。

## 4. 採用なら

```bash
POLARS_MAX_THREADS=2 .venv/bin/python scripts/adsb_learn_apply.py --base opt --opdi
# -> outputs/hybrid_blend_l2_pa_opt_adsb_learned_opdi.parquet（v33 と同じ作り方、残差だけ OPDI 入り）
```

- `OPDI parity LSZH: 2025 fit rows ... 2026 by month ...` の行を確認する。2026 年のどちらかの月の被覆が 2025 年の半分を下回るなら
  提出を見送る（2025-01 と 2026-01 の差は、2026-01-02〜10 にイベントが無いことによる。この環境の確認では、ADS-B の行を偽の列で作った場合、
  2025 年の学習行 0.698、2026-01 0.429、2026-07 0.777。本物の ADS-B の行でも近い値になるはず）。
- **v34 の位置による結びつけ（430 行）を重ねる**: その 430 行は callsign の結びつけが無い未マッチ行で、残差は 0（`G` に入らない）。
  OPDI 入りの残差もそこを変えないので、新しいファイルの 430 行を v34 の値で上書きすれば v34 と同じ作り方になる。
  ```python
  import polars as pl
  new = pl.read_parquet("outputs/hybrid_blend_l2_pa_opt_adsb_learned_opdi.parquet")
  v33 = pl.read_parquet("outputs/v33_blend_l2_pa_opt_adsb_learned.parquet")
  v34 = pl.read_parquet("outputs/v34_v33_posjoin.parquet")
  ow = v34["TAXITIME_SEC_mvt"] != v33["TAXITIME_SEC_mvt"]              # v34 が変えた 430 行
  assert ow.sum() == 430 and (new.filter(ow)["TAXITIME_SEC_mvt"] == v33.filter(ow)["TAXITIME_SEC_mvt"]).all()
  out = new.with_columns(pl.when(ow).then(v34["TAXITIME_SEC_mvt"]).otherwise(pl.col("TAXITIME_SEC_mvt")).alias("TAXITIME_SEC_mvt"))
  out.write_parquet("outputs/v35_v34_opdi.parquet")                    # 版番号は提出前にバケットで確認
  ```
- 提出の前に `python -m prc.validate_submission`、版番号と当日の残り枠はバケットの一覧で確認する（1 日 5 回、UTC と Brussels の両方）。

## 5. 動作確認（この環境）

`claude/model-b-and-arrival-features` の 4338035 に patch を当てた worktree で、本物のラベル・OOF の並び・ランキングと、
**偽の ADS-B 列**（`$SCRATCH/learn/feat_2025|2026.parquet` を乱数で作ったもの）を使って、`--base h2` の 4 本
（plain → `--opdi-shuffle` → `--opdi` → apply `--opdi`）を最後まで走らせた。数値は意味を持たない（コードが通ることの確認だけ）。
- 4 本とも終了コード 0。`--opdi-shuffle` は対照の表（plain との比較）を、`--opdi` は plain と shuffle の両方との比較と
  `DECISION` の行を出した。apply `--opdi` は `outputs/hybrid_H2_adsb_learned_opdi.parquet` を書き、S2 の 7 項目（行数・型・ID の並び・
  欠損なし・負なし・ADS-B の無い行は base のまま・LTFM は不変）がすべて PASS、parity の行も出た。
- 偽の列でも shuffle が plain に対して全行 −0.006 s [−0.010, −0.003] と「有意」になった。列を足すだけで小さな差の CI が
  ゼロを外れうるので、判定に shuffle との比較を入れてある。
- 偽の列で走らせるとき、`adsb_learn_features.mem_gate` は cgroup 全体のメモリで待つので、別の重い処理と並べると止まる
  （この確認では止めて走らせた。本番では止めないこと）。

## 6. ファイル

| ファイル | 中身 |
|---|---|
| `opdi_exitpark.parquet` | 出発便（学習 2,085,047 行＋ランキング 344,841 行のうち 136,388 行）ごとの OPDI の離脱の 5 列。ラベルは入っていない。`docs/opdi_events_2026-10-05/link_events.py` が作る（OPDI v002 の 44 ファイルから。`scripts/opdi_events_ingest.py`） |
| `adsb_opdi.patch` | `scripts/adsb_learn_cv.py`・`adsb_learn_apply.py` への差分（`--opdi`・`--opdi-shuffle` の読み取り、列の追加、S4 の比較相手、出力名） |
| `../../scripts/adsb_opdi.py` | 列の読み込み・LSZH 以外を空にする・並べ替えの対照・判定の表示・2025/2026 の被覆の比較 |

**Removed (English note, 2026-10-14).** `opdi_exitpark.parquet` was removed from the public tree on 2026-10-14,
because it is per-row data derived from EUROCONTROL OPDI, and the EUROCONTROL terms say that the data "may be copied in whole or in part, provided that EUROCONTROL is mentioned as the source and it is not used for commercial purposes", but "may not be modified without prior written permission from EUROCONTROL". Source: EUROCONTROL (OPDI v0.0.2).
It was research only: no submission and not the final file used it (`docs/external_data.md`). It is still in git
history: `git show 1d12abb:handoff/2026-10-05/opdi_exitpark.parquet`. To re-create it:
  1. `.venv/bin/python scripts/opdi_events_ingest.py` downloads the OPDI v002 flight-event files
     `https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_events/flight_events_{span}.parquet`
     (the 10-day spans listed in the script), keeps the ground events at the 10 airports and writes
     `data/external/opdi_events/ground_<span>.parquet`. The 2025-10-02 to 10-12 file returned 404
     (`docs/opdi_events_2026-10-05/plan.md`).
  2. `.venv/bin/python docs/opdi_events_2026-10-05/link_events.py` links the events to the departures and writes
     `data/work/opdi_exitpark.parquet`. It reads the feature cache `data/work_flseq/feat_{train,rank}.parquet`, which
     `prc.pipeline` builds from the competition files.
  3. `src/prc/train.py` (`fe_opdi_exit`) reads `data/work/opdi_exitpark.parquet`; `scripts/adsb_opdi.py` reads the
     path in `PRC_OPDI_EXIT` (its default was the removed file).

## 7. ライセンス（要判断）

OPDI は EUROCONTROL の一般の免責事項（出典明記・非商用・無断改変禁止）で公開されている。コンペの賞の条件は
「追加データはオープンソースのライセンスで公開されていること」（eligibility）。LB には関係しない。
peaceful-babbage の調査では「Discord で open data と認定された」と二次情報で伝わっているが、一次情報は未確認。
提出に使うかどうかはユーザーが決める。使う場合は `docs/external_data.md` に記録する。
