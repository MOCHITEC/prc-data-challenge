# Remediation — second instrument in `prc.compare`, `overall`'s fold set, documentation corrections (2026-09-17)

Written after `findings.md` (2026-09-17), which is the source of every correction below. Three pieces of work:
a reporting-only instrument in `prc.compare` (§1), the smallest honest fix for `overall` silently dropping
`jul_fwd` (§2), and the ten refuted documentation claims of `findings.md` §2.1 plus the four flagged in §2.3
(§4). No model was trained for this; nothing under `results/` other than this new file was touched, and
`configs/` was not touched. `experiments/experiments.jsonl` was **not** rewritten.

---

## 1. `prc.compare`: report the tail, keep the verdict

### 1.1 What was wrong (`findings.md` §1.3)

`verdict()` reads only the `dropped == 0` and `dropped == 200` entries of the curve (compare.py 100, 110), and
the K = 200 drop set is not a random 200 rows: on det_a → ab_split it removes **9 of the 10 rollover rows**,
**17 of 18** rows with y ≥ 12 h, **129 of 537** LIRF unmatched rows, **51 of 233** LIRF sched-filled rows, and
carries **53.9 % of SSE_A, 42.9 % of SSE_B and 84.5 % of Σ(p_a − p_b)²**
(`decision-rule-blind-to-tail.md` §3.2). So a rejection reading "the effect is those rows" never says which
rows, and a change that moves the tail *as a structure* is indistinguishable in the output from one that won
the tail lottery.

### 1.2 What was added

`verdict()` is **unchanged** — same signature, same branches, same thresholds; the curve is still its only
input, and the tests that pin its behaviour (`test_broad_improvement_is_adopted`,
`test_tail_only_improvement_is_rejected`, `test_no_difference_is_rejected`,
`test_a_worse_candidate_is_named_as_worse_not_as_a_null`) are untouched and still pass. What is new is four
reporting-only functions and a section `make compare` prints above the verdict:

| function | what it does |
|---|---|
| `drop_order(y, p_a, p_b)` | the shared ordering by `max(se_a, se_b)`, factored out of `robustness_curve` so the report cannot describe a different set of rows than the curve removes |
| `dropped_rows(y, p_a, p_b, k)` | the K row indices the curve removes, worst first |
| `group_diff(y, p_a, p_b, mask, …)` | the paired row bootstrap on a subset (n, RMSE A, RMSE B, diff, CI95, P(B)) |
| `per_group_diff(y, p_a, p_b, group, mask, …)` | `group_diff` per airport, ordered by that airport's share of Σ(p_a − p_b)² |
| `tail_report(...)` / `format_tail_report(rep, max_listed=None)` | assembles and renders the section |

`load_pair` now also reads `ADEP_mvt` (one more column of `feat_train.parquet`); `main` gained
`--max-listed` (default: list every dropped row).

The printed section is headed

```
tail structure (reporting only - the VERDICT below is computed by verdict() from the curve alone,
and none of these lines change adopt/reject):
```

and the verdict line itself now reads `VERDICT (from the curve only, rule unchanged): …`.

### 1.3 What it looks like on a real pair

`POLARS_MAX_THREADS=2 python -m prc.compare det_a-cfc9417c52 ab_split-ca87cb5af7 --n-boot 50 --max-listed 6`
(the item pair of `decision-rule-blind-to-tail.md`; `--n-boot 50` only to keep the CPU free, the default is 500):

```
  unmatched rows            n=  8,378  A=   2299.0  B=   1875.3    -423.69 [  -589.11,   -224.31]
  unmatched @ LIRF          n=    537  A=   7206.0  B=   5391.7   -1814.33 [ -2441.99,   -791.10]   64.9% of the unmatched disagreement

  unmatched difference by airport (ordered by share of sum (p_a - p_b)^2):
     airport       n         A         B      diff                   CI95  gap share
        LIRF     537    7206.0    5391.7  -1814.33    [-2441.99, -791.10]     64.9%
        EHAM   1,353     928.9     506.0   -422.92    [ -646.97,  -89.99]      9.3%
        LFPG   1,565    2733.2    2646.6    -86.62    [ -427.00,  -22.64]      8.5%
        LEBL     652    1068.8     572.3   -496.47    [ -859.28,   -7.28]      5.8%
        EDDF     787    1053.6     687.8   -365.84    [-1013.34,  +83.60]      5.1%
        EDDM     402    1018.7     789.1   -229.60    [ -616.11,  +22.27]      2.2%
        EGLL     647    1054.4    1104.5    +50.09    [  -41.97, +164.94]      2.1%
        LSZH     722     715.3     773.9    +58.59    [  -42.05, +144.27]      0.9%
        LEMD     416     417.3     578.6   +161.31    [  -22.76, +278.21]      0.7%
        LTFM   1,297     513.3     518.1     +4.82    [   -9.59,  +16.14]      0.4%

  the K=200 drop removes 200 rows (189 unmatched, 11 matched) - these are the rows the K=200 CI is blind to:
    rank     MVT_ID_mvt  airport       grp          y        p_a        p_b
       0      182378789     LFPG unmatched    84240.0      717.3      876.2
       1      182377662     LFPG unmatched    58206.0      906.3      995.7
       2      200297323     LIRF unmatched    87480.0    31210.8    47721.6
       3      201198951     LIRF unmatched    87605.0    32871.9    56413.4
       4      200300302     LIRF unmatched    87598.0    34328.4    53323.4
       5      182646487     LIRF unmatched    87177.0    34694.1    49158.1
    ... 194 more (raise --max-listed to see them)

VERDICT (from the curve only, rule unchanged): REJECT - tail-only: -34.58 s on all rows, but -0.27 s
[-0.83, +0.21] once the worst 200 rows are removed - the effect is those rows, and the ranking set has
different ones
```

The 189 / 11 split and the unmatched −423.7 reproduce `decision-rule-blind-to-tail.md` §3.2 and §3.3 exactly.
The verdict is the same REJECT as before this change; what is now readable is that the 255/20 build is better
on **nine of ten** airports' unmatched rows but that 64.9 % of the disagreement is LIRF and the drop set is the
rollover band.

### 1.4 Tests added (`tests/test_compare.py`, existing style)

- `test_dropped_rows_are_exactly_the_rows_the_curve_removes` — `dropped_rows` returns the very rows
  `robustness_curve` removes (RMSE on the complement is equal, not merely the count).
- `test_tail_report_names_the_rows_the_verdict_stops_looking_at` — on the tail-only fixture the verdict is
  still `tail-only`, while the report lists the 30 rows with their label and both predictions, names the
  dominant airport and reports the unmatched CI.
- `test_per_airport_breakdown_separates_structure_from_a_one_airport_tail` — ordering by gap share, shares sum
  to 1, and every other airport shows exactly 0.
- `test_group_diff_on_every_row_is_the_full_data_difference` — `group_diff` with an all-true mask equals the
  curve's K = 0 entry.
- `test_the_report_is_labelled_as_reporting_only_and_leaves_the_verdict_alone` — the heading says so, and the
  verdict is byte-identical before and after rendering the report.

---

## 2. `overall` silently dropping `jul_fwd`

### 2.1 The fix chosen

**Option taken: name it in the record, leave the pooling itself alone.** `run_cv` now tracks, per fold, how
many rows that fold actually contributed to `overall` (`n_into_overall`) and records
`overall.folds_pooled` / `overall.folds_shadowed`.

Why this one:

- It is the only option that changes **no logged number**. `pl.coalesce` in fold order already excludes
  `jul_fwd`; "exclude it explicitly" would compute the same `overall` and add a hard-coded fold name to
  `run_cv`, which is wrong the moment someone runs `folds: [jul_fwd]` (where `jul_fwd` *is* the overall).
- The general rule — *a fold contributes the rows no earlier fold predicted* — is exactly what the code does,
  so the record describes the code instead of asserting a special case.
- "Report `overall` per fold-set" would change the shape of every record and break the comparability of
  `overall.rmse_full` with the 41 existing ones; the same information is recoverable from the per-fold table.

### 2.2 What it changes about existing logged numbers

Nothing numeric. `overall.rmse_full`, `rmse_clean`, `n`, `bias`, `top20_share_of_sse` and every fold entry are
computed exactly as before, and `experiments/experiments.jsonl` was not rewritten. New keys only:

| key | meaning | value for the six four-fold records |
|---|---|---|
| `overall.folds_pooled` | folds whose rows reached `overall` | `["jan", "jul", "forward"]` |
| `overall.folds_shadowed` | folds an earlier fold had already predicted | `["jul_fwd"]` |
| `folds.<name>.n_into_overall` | rows that fold contributed | `jul_fwd`: 0 |

So the historical reading stands as `fold-shape-vs-ranking.md` §3.3 established it: `overall.n` = **672,428**
against a fold sum of **863,141**, `overall.rmse_full` = pooled{jan, jul, forward} to 3 dp, differing from
pooled{jan, jul_fwd, forward} by 7.6–14.0 s. Records written from now on say which of the two they are.
`overall`'s composition is unchanged and still leaky by construction (Jan-2025 22.9 %, Jul-2025 28.4 %,
Nov+Dec-2025 48.8 %).

### 2.3 Test added

`tests/test_train.py::test_overall_names_the_folds_it_pooled_and_the_ones_it_shadowed` — a three-month
synthetic run (folds `jan`, `jul`, `jul_fwd`) asserts `folds_pooled == ["jan", "jul"]`,
`folds_shadowed == ["jul_fwd"]`, `jul_fwd.n_into_overall == 0`, `jul.n_into_overall == jul.n`,
`overall.n == Σ n_into_overall` over the pooled folds, and `overall.n < Σ fold n`.

---

## 3. `make test` / `make lint`

Run at the end of the work, on the whole tree (another agent was training on the same box, which is why the
suite took 19 minutes instead of the usual few):

```
$ make test
.venv/bin/python -m pytest -q
....................................................................     [100%]
68 passed in 1127.14s (0:18:47)

$ make lint
.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests
All checks passed!
27 files already formatted
```

68 passed: the 62 that existed before plus the 5 added to `tests/test_compare.py` and the 1 added to
`tests/test_train.py` (the count also includes tests another agent added to `tests/test_train.py` in the same
working tree). No test was modified or removed.

---

## 4. Documentation corrections

House rule followed everywhere: **nothing was deleted**. Each original sentence stays where it was, and the
correction is appended directly under it as 「**訂正（2026-09-17）**: こう記録していた → 実測はこう
（結果ファイル）」. Both documents keep their voice: `docs/README.md` stays a decision document in Japanese
with its numbered sections, `docs/handoff.md` stays measurements-only with the correction as another measured
line, citing the result file.

A pointer to the investigation was added at `docs/README.md` line 3 (blockquote under the title) and to the
code map in `docs/handoff.md` §10.

### 4.1 The ten refuted claims (`findings.md` §2.1)

Line numbers are the pre-edit ones (`git show HEAD:<file>`).

| # | file:line | before | after (added under the original line) |
|---|---|---|---|
| R1 | README:237 (§10 Phase 2' row a) | 「極端例が 33 個 → ~2,000 個になる。**実装したが一度も測っていない**」 | 訂正ブロック: プールしても y ≥ 12h は **33 → 34**、y ≥ 4h **119 → 122**、y ≥ 2h **481 → 584**（`shared-b-no-transfer.md` §H）。施策自体は forward fold で実測され、未マッチ OOF **2,109.32 → 2,144.73**、fold 全体 284.70 → **287.12**、`prc.compare` は **REJECT**。ただし非 LIRF 未マッチ **1,034.74 → 486.57**、LIRF **8,578 → 9,690**（`shared-b-all-rows.md` §3.1–§3.3） |
| R2 | README:211 (§9-3) | 「特徴量 72 列は初日のまま」 | 訂正: 72 = **フレーム数値列 71 + `month`**、エンコーディング 9 列は**含まない**（`predict_fold` 内で後結合）。実際の行列は model A **80+10**（v1/v2/v5/v8）… **98+10**（現行）、model B 65 / 82 / 92 / 83 + 10。「初日から変更なし」も誤り（`n-features-72-mislabel.md` §3.1, §3.2） |
| R2 | handoff:38 (§2) | 「特徴量列数: 72（初日から変更なし）。うちエンコーディング 9 列」 | 同上を実測値として追記（結果ファイル併記） |
| R2 | handoff:221 (§11 レビュー用プロンプト) | 「全二乗誤差の 48.1% が model B 側にある。特徴量 72 列。」 | 同行の直下に【訂正: 72 はフレーム数値列 71 + month で、エンコーディング 9 列を含まない。実際の行列は model A 80+10〜98+10、model B 65〜83+10】 |
| R3 | README:106 (§6 使い方) | 「**未マッチ行だけを動かす変更は測れない**（対比較でも CI 幅 62 秒）。**触らない**」 | 訂正: 62 秒はどの組でも再現せず（測った組も記録に無い）、実測は **385.0 / 724.4 / 421.5**（3 fold の組）、**5.57**（forward のみ）、**0.00**（model B 予測が同一）。マッチ行は 0.77 ではなく **0.14〜1.17**（`decision-rule-blind-to-tail.md` §3.3）。「測れないから触らない」は前提ごと失効 |
| R3 | README:131–132 (§6 分解能表) | 「マッチ行 **0.77 秒** / 未マッチ行 **61.97 秒**（実質測定不能）」 | 表を「記録 → 実測」の 2 値表記に変更: マッチ 記録 0.77 → 実測 **0.14〜1.17**、未マッチ 記録 61.97 → 実測 **385 / 724 / 421**（3 fold）・**5.57**（forward のみ）。「実質測定不能」は撤回、結果ファイルを脚注に |
| R3 | handoff:82 (§5) | 「未マッチ行だけの対比較は 95% CI 幅が 62 秒。マッチ行は 0.77 秒。」 | §5 に訂正ブロック（上と同じ 5 組の実測値、組が特定できていないことも明記） |
| R3 | handoff:241 (§11) | 「未マッチ行だけの対比較は 95%CI 幅が 62 秒、マッチ行は 0.77 秒。」 | 直下に【訂正 (2)】として実測値 |
| R4 | README:188–190 (§8.1-2) | 「model B の抽選性 … `rounds_unmatched` が 199 / 212 / 585 / 658 と乱高下し … その振れ幅が ±37 秒」 | 訂正ブロック: **v2 と v8 の `rounds_unmatched` はともに 658** なので本数の乱高下では説明できない。差は d182fc1 の refit シードキー 2 つで、旧形は v2 を新形は v8 を **5,290 行すべて**で再現（`refit-seed-semantics-changed.md` §3.1, §3.2）。±37 秒が未マッチ行に載っていること（SSE 差の 96.5%）は確認済みだが本数の抽選に帰属させた測定は無い。本数がホールドアウト月で決まること自体は実測（1 → 2 か月で **45 → 413**、`holdout-month-decides-rounds.md` §4.3） |
| R4 | README:238 (§10 row b) | 「±37 秒の抽選（§8.1-2）を平均で潰す」 | 訂正: 根拠は失効。シード平均は実測済みで 3 シード **2,312.5** / 1 シード **2,299.0**（`plan.md` §2 `seed-avg-no-effect-b`） |
| R5 | README:188 / handoff:142 / handoff:267 | 「学習 22,470 行のうち極端値（y ≥ 4h 相当）は 33 行」 | 訂正: **33 は y ≥ 12h（43,200 秒）の行数**。y ≥ 4h は **119**、y ≥ 2h は **481**、y ≥ 80,000 は 15。y ≥ 43,200 の 34 行のうち日またぎ帯は 16（15 が未マッチ）（`plan.md` §4-1、`block-day-rollover.md` §3.1、`shared-b-no-transfer.md` §B） |
| R6 | README:189 / handoff:143 / handoff:268 | 「early stopping 用ホールドアウトは約 1,650 行。そこに含まれる極端値は 0〜5 行」 | 訂正: 12 月ホールドアウトは **1,649 行**で y ≥ 14,400 が **6 行**（6 は未再測定）。forward fold が使う 10 月ホールドアウトは **1,913 行・y ≥ 4h 4 行・y ≥ 12h 0 行・最大 25,558**（`holdout-month-decides-rounds.md` §4.1） |
| R7 | handoff:147 / handoff:271 | 「v5 の予測は、12 時間超の taxi 時間を学習データの出現頻度の 2 倍の頻度で出していた」 | 訂正: 逆。未マッチ 5,290 行で `v5cand` の 43,200 秒超は **2 行**、学習率 33/22,470 からの期待は **約 7.8 行** ＝ 学習頻度の **約 1/4**。v2 2 / up_v3 16 / up_v4 **18** / v7 2 / v8 2、全 344,841 行では 3 / 17 / 19 / 3 / 3 / 3（`plan.md` §4-5、`lirf-regime-tail-count.md` §3.4）。`v5_unmatched_capacity.parquet` が提出版かは未確認と明記 |
| R8 | README:72–96 (§6 見出しと 2 事例表) | 見出し「CV は本番を予測できるか → **条件付きでできる**（2 事例で検証）」＋ 表の `model B 255 葉 −34.82 / −0.06 [−0.61,+0.52] / −0.26` | 訂正: この行は **`experiments/` のどの組でも再現しない**（v2 当時の OOF が無い）。最も近い実測は **−36.60 / −0.36 [−0.89,+0.18] / −0.25** と **−34.58 / −0.27 [−0.85,+0.31] / −0.24**。「既知 2 事例で LB と方向が一致」は **UNDETERMINED**、見出しは「検証済み」ではなく「未確認の仮説」として読むこと（`decision-rule-blind-to-tail.md` §3.6, §4, §5） |
| R8 | README:165 (§8 実験表の行) | 「\| v2 → v6（model B 255 葉） \| −34.82 \| −0.06 … \| **棄却（尾部のみ）** \|」 | 行末に「（※ 再現不能、下記訂正）」を付し、表の下に同じ実測 2 組を併記。判定（棄却・尾部のみ）はどちらの組でも同じであることも明記 |
| R9 | README:104 (§6) | 「**採否は 200 行除外後の CI で決める。** CV full は参考値」 | 訂正: コードは**全行の CI を先に読み、ゼロを跨げば 200 行除外後を見ずに REJECT**。規則は「両方が改善側のときだけ ADOPT」。例: det_a → main-48c19bc830 は K=200 −2.58 [−2.96,−2.15] でも全行 [−3.75,+0.55] で REJECT（`decision-rule-blind-to-tail.md` §3.1, §4） |
| R9 | README:159 (§8) | 「**全行の CI が改善側 ∧ 上位 200 行を落とした後の CI も改善側**。」 | 「（コードは全行の CI を先に見て、そこがゼロを跨げば 200 行除外後を見ずに REJECT する）」を追記（文自体は正しいので順序だけ補足） |
| R9 | handoff:80–81 / handoff:240 (§5, §11) | 「全行の 95% CI と上位 200 行除外後の 95% CI の両方を見る／既知 2 事例で一致」 | §5 と §11 に訂正: 評価順序（全行が先）と、2 事例一致が未確定であること |
| R10 | handoff:146 (§6) | 「v5 は 255 葉 / min_data 20 で提出し、LB は 312.02 → 325.22」 | 訂正: **コード境界をまたいだ比較**で「255/20 が 13 秒のコスト」とは読めない。v2 は旧コード、v5 は現行コード系。同一コード系では v5 は v7 (342.59) / v8 (343.80) より **17〜19 秒良い**。決着には同一コードの probe 提出が必要（Batch D は未実施）（`plan.md` §1 #29） |
| R10 | handoff:271 (§11) | 同上（プロンプト内） | 【訂正 (5)】として同じ内容を 1 行で |

### 4.2 The four undecided numbers flagged in `findings.md` §2.3

| file:line | before | after |
|---|---|---|
| README:54 と README:198 | §4「首位 245.02 に届くには未マッチ RMSE ≈ **1,033**」 / §8.2「首位 245.02 → **≈ 929**」 | 両方に訂正を追記: 同じ首位スコアに対して本書の中で数字が割れており、`lirf-regime-tail-count.md` §3.5 の再計算表に首位の行が無いため**決着していない**。v4 ≈ 1,736 / v8 ≈ 2,157 は再計算で一致することも明記 |
| README:24 / handoff:18–19 (§1) | 「53 / 127 チーム、首位 youthful-giraffe 245.02 / 2 位 enthusiastic-daisy 249.20」 | 09-15 時点の値と明示し、09-16 取得の `data.json`（**154 チーム**、首位 vigorous-whistle **243.58**、2 位 enthusiastic-daisy **244.83**、youthful-giraffe 245.02）を並記（`plan.md` §4-10） |
| README:43–52 (§4) / handoff:45–54 (§3) | 誤差配分表（210 行 36.7% / 55 行 33.6% / 10 行 24.1%）にファイル名なし | 訂正: 厳密に再現するのは **`oof_main-f9d57a661f`**（total 300.17 / 未マッチ 1,864.9）。`oof_main-48c19bc830`（total 334.58 / 未マッチ 2,296.1）では同じ 10 行が **34.7%**、OOF ファイル全体で **23.7–39.9%**（`block-day-rollover.md` §3.5, §3.6） |
| handoff:41 (§2) | 「`AOBT_3` が分単位に丸められている割合 96.6%」 | 補足: 母数を明示 — :00 ちょうどは **2,015,132 行 = 全 2,085,047 行の 0.9665**（null 込み）、**非 null 2,062,577 行の 0.977**（`timestamp-precision-clusters.md` §3.1） |

### 4.3 Other corrections made in passing (same rule: record kept, measurement added)

| file:line | before | after |
|---|---|---|
| README:116 付近 (§6 fold 表) | fold 表に `overall` の説明なし | `overall` が fold 順 coalesce のため **`jul_fwd` を 1 行も含まない**（overall.n 672,428 / fold 合計 863,141）ことと、2026-09-17 に記録側を修正したことを追記（`fold-shape-vs-ranking.md` §3.3） |
| README:228 (§10 Phase 2 再開理由) | 「9-15 時点では『未マッチ行は CI 幅 62 秒で測れないから諦める』としていた」 | 「（加えて『62 秒』自体も再現しない — §6 の訂正）」を追記 |
| README:239 (§10 row c) | 「early stopping を廃止し木の本数を固定」の根拠 | 訂正: 設定では不可能（`unmatched_num_rounds` は `min(...)` の上限、600 にしてもビット同一。`inner_holdout_months` / `early_stopping` は model A と共有）（`holdout-month-decides-rounds.md` §2, §4.2, §5） |
| handoff:84 (§5 fold 行) | 「fold は 4 種。`forward` 以外は未来のデータで過去を当てている」 | fold の形の実測（学習月数 11 / 11 / 6 / 10、前向き horizon 最大 2 か月 vs 本番 1 と 7、年をまたぐ fold 無し、1 月を当てる前向き fold 無し）と `overall` の `jul_fwd` 欠落を追記 |
| handoff:172 (§8) | 「`shared_b` … 実装後に削除、**未測定**」 | 更新: forward fold で測定済み（未マッチ 2,109.32 → 2,144.73 / 2,154.29、判定 REJECT、非 LIRF 1,034.74 → 486.57、LIRF 8,578 → 9,690）（`shared-b-all-rows.md`） |
| handoff:176–177 (§9 未実施) | 「model B のシード数を増やす」「early stopping を廃止して木の本数を固定する」 | 更新: シード 3 本は測定済み（2,312.5 / 2,299.0）、本数固定は現行コードの設定では不可能 |
| handoff:194 (§10 コード一覧) | `docs/guard_audit.md` まで | `docs/investigation_2026-09-16/findings.md` を追加 |
| src/prc/compare.py モジュール docstring | 「10 行が 24%」「v6 は −34.8 → 0.06」 | ファイル名（`oof_main-f9d57a661f` で 24%、`oof_main-48c19bc830` で 34.7%）と、v6 の組が再現不能で最近傍が −34.58 → −0.27 / −36.60 → −0.36 であることを括弧書きで併記 |

Not touched, as instructed: `configs/main.yaml`'s header still carries the R5 ("~33 day-rollover rows"),
R7 ("twice the training frequency") and R10 ("255/20 cost 13 s") wordings — `findings.md` §2.1 lists them at
`configs/main.yaml` lines 4–7 and they remain to be corrected by whoever owns that file.

---

## 5. What this does not do

- It does not change any adopt/reject outcome, past or future: `verdict()` and `robustness_curve` compute
  exactly what they computed before (`drop_order` is the same expression, factored out).
- It does not answer whether a different rule would have produced a better leaderboard. The flip table is
  `decision-rule-blind-to-tail.md` §3.6 and no rule's LB outcome is measured (`findings.md` §5.2).
- It does not decide 1,033 vs 929, which file `docs/README.md` §8's v2 → v6 row came from, which pair gave
  "62 s / 0.77 s", or whether `outputs/v5_unmatched_capacity.parquet` is the uploaded v5. Those stay open in
  `findings.md` §4 items 13–18 and are now marked as open in the documents themselves.
- It does not touch `configs/main.yaml`, whose header still carries three of the refuted wordings (§4.3), and
  it does not rewrite `experiments/experiments.jsonl`.
