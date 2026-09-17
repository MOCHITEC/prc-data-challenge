# Item 30 `shared-b-all-rows` — result (2026-09-17), also settling item 31's open question

Batch D, forward-fold run. Source item: `plan.md` §1 #30 (from E H7; H §8) and #31 (from M `shared-b-no-transfer`).
Script: `docs/investigation_2026-09-16/scripts/shared-b-all-rows.py` (standalone; imports `TrainConfig`,
`fit_encodings`, `attach_encodings`, `visible_numeric_columns`, `_matrix`, `_inner_holdout_mask` from `prc.train`
and `robustness_curve`, `month_bootstrap_rmse_diff`, `verdict`, `DROP_LEVELS` from `prc.compare`, and implements the
pooled variant itself, as Batch B's scripts did). **Nothing under `src/` or `configs/` was modified**; the run reads
`configs/inv_baseline_forward.yaml` (the existing Batch C baseline config), `data/work/feat_train.parquet` and
`experiments/oof_inv_baseline_forward-67e76a5dfc.parquet`. `POLARS_MAX_THREADS=1`, LightGBM `num_threads: 2`, one fit
at a time, four fits, 476.7 s wall in total (the two pooled fits 218.6 s and 221.2 s). git sha at run time bb71045;
LightGBM 4.7.0, polars 1.44.2, numpy 2.4.6.

**Forward fold only**, as instructed: a pooled fit is a 1,757,038-row fit, and the forward fold is the only one
shaped like the real task. Train Jan–Oct 2025 (1,757,038 labelled rows), inner holdout October, evaluate Nov+Dec 2025
(328,009 rows, of which 3,005 unmatched). The 2.08 M figure in the source claim is the **whole** training set; the
forward fold's training side is 1.76 M of it (19,465 unmatched).

Definitions follow `plan.md` §0: `y` = `TAXITIME_SEC_mvt`; `ms` = `mms` = `mvt_minus_sched` = MVT − SCHED;
"unmatched" = routing definition (`mvt_minus_aobt3` null → model B); sched-filled here = |y − ms| ≤ 60 s;
2 h = 7,200 s, 4 h = 14,400 s, 12 h = 43,200 s.

Sibling reports used as inputs and **not** re-derived: `results/shared-b-no-transfer.md` (item 31 pre-check: what
pooling adds, the column classes, the regime indicators, the 2,751 vs 627 LIRF ms ≥ 2 h counts);
`results/mms-mixture-reparam.md` (item 9); `results/tree-identity-saturation.md` and `results/linear-tree-model-b.md`
(the Batch C baseline `inv_baseline_forward-67e76a5dfc` and how `make compare` judges a run).

## 1. Hypothesis — the two conflicting sides, as stated

**Side A — item 30 (E H7; H §8; `docs/README.md` Phase 2' row a; commit 5b90919).** "Training model B on all
~2.08 M rows (`shared_b`, schedule-only features) improves unmatched predictions." `docs/README.md` line 237:
「`shared_b`: model B を全 208 万行で学習（スケジュール系の特徴量のみ使用）／極端例が 33 個 → ~2,000 個になる。**実装したが一度も測っていない**」.
Commit 5b90919's reason: the discarded 2,062,577 matched rows "carry the same airport/hour/aircraft structure".
`docs/handoff.md` §8 lists it as implemented, deleted (ea04ec9) and never measured.

**Side B — item 31 (M `shared-b-no-transfer`).** "Training model B on all rows will not help the tail because the
LIRF actype-null sched-filled regime does not exist among matched rows." The pre-check
(`results/shared-b-no-transfer.md`) confirmed the data facts (pooling adds 103 rows at y ≥ 2 h, 3 at y ≥ 4 h, 1 at
y ≥ 12 h; 0 matched rows with the LIRF ∧ actype-null ∧ ms ≥ 2 h signature; at LIRF ms ≥ 2 h the 2,751 matched rows
outnumber the 627 unmatched ones 4.4 : 1 and carry the opposite mapping, P(y ≥ 2 h) 0.016 vs 0.60–0.85) and left both
A4 ("pooling improves unmatched predictions") and B6 ("pooling will not help the tail") UNDETERMINED pending this run.

Sub-claims tested here: (A4) pooled model B beats the split model B on the unmatched rows it predicts; (B6) pooling
does not help the tail; (C) how many trees the pooled model uses; (D) whether it separates the regimes on
`nm_matched`; (E) what it predicts on the LIRF unmatched ms ≥ 12 h rows.

## 2. Procedure actually executed

### 2.1 Resolved configuration

`configs/inv_baseline_forward.yaml` was read exactly as `src/prc/pipeline.py::main` reads it (only keys that are
`TrainConfig` dataclass fields; `params` merged over the defaults), then `.for_unmatched()` for model B:

| field | model A (not refitted here) | model B, all four variants |
|---|---|---|
| `params` | objective regression, learning_rate 0.05, **num_leaves 127**, **min_data_in_leaf 50**, feature_fraction 0.8, bagging_fraction 0.8, bagging_freq 1, lambda_l2 1.0, verbose −1, **num_threads 2**, deterministic true, force_row_wise true, seed 42 | identical |
| `num_rounds` (cap) | 2000 | 2000 |
| `early_stopping` / `inner_holdout_months` / `n_seeds` | 100 / 1 / 3 | 100 / 1 / 3 |
| `use_categoricals` / `use_encodings` / `use_linear_tree` / `exclude_features` | true / true / false / () | same |

Printed by the script: under this config `for_unmatched()` equals `cfg` on (`num_leaves`, `min_data_in_leaf`,
`num_rounds`) = (127, 50, 2000), because `configs/main.yaml` sets `unmatched_num_leaves: 127`,
`unmatched_min_data_in_leaf: 50`, `unmatched_num_rounds: 2000`. **The removed `shared_b`'s one design decision —
"capacity follows the training set, so use `cfg` rather than `cfg.for_unmatched()`" (5b90919) — is therefore not a
difference under today's config.** It would have been one under the deleted `configs/p1_shared_b.yaml`, which set
255 / 20 / 3000 for the unmatched model.

### 2.2 What the removed `shared_b` did (git archaeology)

`git log -S shared_b --all` → added in **5b90919** ("feat: make the A/B split a measurable choice instead of an
assumption", 2026-09-14 06:01), removed in **ea04ec9** ("refactor: delete everything that never reaches a
submission", same day 06:24, reason given: "they were never measured, so they were three code paths carrying one
experiment's worth of evidence between them"). The diff in `src/prc/train.py` added a `split_mode` field with three
values (`separate` / `shared_b` / `single`) and, in `predict_fold`, exactly two lines of behaviour for `shared_b`:

```
u_tg = tg.filter(pl.col(PROXY).is_null())
u_tr = tr if cfg.split_mode == "shared_b" else tr.filter(pl.col(PROXY).is_null())
...
b_cfg = cfg if cfg.split_mode == "shared_b" else cfg.for_unmatched()
```

i.e. model B fitted on **every labelled training row** with `cols_b` **unchanged** (no column restriction), still
predicting only the no-NM rows, model A untouched. `configs/p1_shared_b.yaml` and `tests/test_split_modes.py` were
deleted with it; the test asserted that `shared_b` leaves model A bit-identical and only moves the unmatched rows.
The variant `P-full` below is that implementation; `P-restr` is the column-restricted variant this item specifies.

### 2.3 Column list

The script rebuilds `cols_b` by the code's own rule: `visible_numeric_columns` (frame numerics minus `LABEL`,
`month`, minus the nine `CANDIDATE_COLS` that `main.yaml` does not include, minus `OFFSET_COLS`) minus the six model-B
prefixes (`mvt_minus_aobt3`, `aobt3`, `lobt_minus`, `mvt_minus_lobt`, `mvt_minus_eobt1`, `mvt_minus_iobt`), plus
`CATEGORICALS`. Measured on the fold's frames: 111 numerics after `attach_encodings` (99 frame feature columns + 9 fold-joined
encodings + 2 `OFFSET_COLS` + this script's own `ym` fold key), 100 visible (the nine `CANDIDATE_COLS` and the two
offset columns hidden), 99 after removing `ym`, and **83 numeric in `cols_b`** = the 74 non-encoding columns of
`shared-b-no-transfer.md` §3 G plus the nine fold-fitted encodings (`ref_p10`, `ref_p20`, `ref_med`, `ref_n`,
`arh_med`, `arh_n`, `apt_med`, `ref_p10_fb`, `ref_missing`), plus **10 categoricals** → **93 columns, the full set**.

The **restricted set (89 columns)** is that list minus the four categoricals that `shared-b-no-transfer.md` §3 G finds
100 % null on unmatched rows in training *and* ranking — re-verified here on this fold (null rate on matched training
rows 0.0000, on unmatched training rows 1.0000, on unmatched validation rows 1.0000):

> **dropped: `AIRCRAFT_OPERATOR_flt`, `MARKET_SEGMENT_flt`, `FLIGHT_TYPE_flt`, `WK_TBL_CAT_flt`.**

Reading of "non-null in both regimes": those four are the only columns present in one regime and absent in the other.
Kept, with the reason: `AIRCRAFT_TYPE_mvt` (null 0.0000 matched / 0.0700 unmatched training / 0.0479 unmatched
validation — non-null in both regimes, just not on the LIRF subset); `gust_ms`, `gust_ms_prev1h` (≈ 97 % null in
*both* regimes, so symmetric); `nm_matched`, `eobt1_null`, `lobt_null` (flags, never null, but constant within a
regime). The list is stated in full in the run log.

### 2.4 Variants (four fits, sequential, never two at once)

| id | model-B training rows | features |
|---|---|---|
| **S-full** | 19,465 unmatched (the code's `separate`) | 93 (full) |
| **S-restr** | the same 19,465 | 89 (restricted) |
| **P-restr** | **all 1,757,038 labelled rows** | 89 (restricted) — this item's mandated variant |
| **P-full** | **all 1,757,038 labelled rows** | 93 (full) — the literal removed `shared_b` |

Each fit follows `train._fit`: `_inner_holdout_mask` (latest 1 month of the training side) → early stop → rescale the
round count to the refit size → refit on all rows under 3 seeds and average → `np.clip(pred, 0, None)` as `_assemble`
does. Model A is not refitted: `shared_b` leaves it bit-identical, so whole-fold numbers combine each variant's
unmatched predictions with the logged matched OOF column (325,004 rows, RMSE 201.6633 — the record's figure to
4 dp). Comparisons use `prc.compare.robustness_curve` (K = 0/20/200/2000, the K rows where *either* model errs most,
500-sample paired bootstrap), `prc.compare.month_bootstrap_rmse_diff` and `prc.compare.verdict`, called directly on
the OOF arrays.

### 2.5 Baseline reproduction

`S-full` reproduces the logged record `inv_baseline_forward-67e76a5dfc` **exactly**:

| quantity | logged record | this script |
|---|---|---|
| `rounds_unmatched` | 45 | 45 (best_iter 41 on 1,913 holdout rows) |
| unmatched RMSE, n = 3,005 | 2,109.32 | **2,109.3167** (max |diff| row-by-row **0.0000**) |
| matched RMSE, n = 325,004 | 201.6633 | 201.6633 |
| forward total, n = 328,009 | 284.7038 | **284.703845** |

## 3. Results

### 3.1 Forward-fold RMSE

| variant | trees per seed | unmatched RMSE (n = 3,005) | LIRF (140) | non-LIRF (2,865) | whole fold (328,009) | pred median | pred max | n pred ≥ 2 h |
|---|---|---|---|---|---|---|---|---|
| logged OOF | 45 | 2,109.32 | 8,578.38 | 1,034.74 | 284.7038 | 1,079.5 | 34,540 | 41 |
| **S-full** (repro) | 45 | **2,109.3167** | 8,578.38 | 1,034.74 | **284.7038** | 1,079.5 | 34,540 | 41 |
| **S-restr** | 45 | 2,109.3167 | 8,578.38 | 1,034.74 | 284.7038 | 1,079.5 | 34,540 | 41 |
| **P-restr** (pooled, 89 cols) | **152** | **2,144.73** | **9,689.57** | **486.57** | **287.1174** | 1,062.6 | 38,155 | 41 |
| **P-full** (pooled, 93 cols) | **145** | **2,154.29** | **9,733.56** | **487.98** | **287.7726** | 1,050.3 | 37,752 | 42 |

`S-restr` is **bit-identical** to `S-full` (max |diff| 0.0 over all 3,005 rows): on the unmatched-only fit the four
dropped categoricals are constant (`__NA__`), and LightGBM discards single-bin features at Dataset construction, so
removing them changes nothing. The column restriction is only a real change inside a pooled fit, where those columns
vary on the matched rows.

Inner holdout for early stopping: 1,913 rows (October unmatched) in the split fits, **185,674** rows (all October
rows) in the pooled fits — which is where the jump from 45 to 152/145 trees comes from.

### 3.2 Unmatched RMSE per airport (forward fold)

| ADEP | n | logged / S-full / S-restr | P-restr | P-full |
|---|---|---|---|---|
| EDDF | 188 | 1,683.29 | **503.04** | 506.42 |
| EDDM | 165 | 1,349.80 | **729.64** | 738.81 |
| EGLL | 181 | 726.96 | **536.43** | 526.50 |
| EHAM | 545 | 1,184.00 | **286.24** | 285.05 |
| LEBL | 156 | 1,150.42 | **275.02** | 285.61 |
| LEMD | 107 | 360.39 | **319.60** | 312.66 |
| LFPG | 685 | 1,217.00 | **684.30** | 690.37 |
| **LIRF** | **140** | **8,578.38** | **9,689.57** | **9,733.56** |
| LSZH | 198 | 366.81 | **310.80** | 293.19 |
| LTFM | 640 | 420.16 | **361.26** | 359.70 |

Pooling improves nine airports out of ten and worsens LIRF. Sub-splits of LIRF (P-restr vs S-full):
ms < 2 h (82 rows) 1,974.3 → 1,914.2; ms ≥ 2 h (58 rows) 13,119.3 → **14,881.0**; sched-filled |y − ms| ≤ 60 (99 rows)
3,597.4 → 3,400.8; not sched-filled (41 rows) 14,833.4 → **17,107.5**. The LIRF share of unmatched SSE rises from
0.771 (S-full) to 0.951 (P-restr); the four worst rows' share from 0.677 to 0.878.

### 3.3 Paired comparison, the way `prc.compare` computes it

A = S-full (the reproduced baseline), B = the variant. Negative = B better. 500-sample paired bootstrap.

**Unmatched rows only (3,005):**

| B | K=0 | K=20 | K=200 | K=2000 | month bootstrap (2) | `verdict` |
|---|---|---|---|---|---|---|
| S-restr | +0.00 [+0.00, +0.00] | +0.00 | +0.00 | +0.00 | +0.00 | REJECT — no difference |
| **P-restr** | **+35.41 [−409.48, +275.50]** P(B)=0.426 | −28.09 [−69.73, +11.58] | **−25.37 [−36.36, −13.11]** | **−11.61 [−15.19, −8.40]** | +35.41 [−79.80, +143.45] P=0.290 | **REJECT** — "no difference on all rows" |
| **P-full** | **+44.98 [−397.22, +284.72]** P(B)=0.404 | −13.90 [−59.31, +34.26] | **−27.83 [−38.24, −18.05]** | **−12.01 [−15.52, −8.32]** | +44.98 [−64.98, +148.57] P=0.290 | **REJECT** — "no difference on all rows" |

(RMSE levels behind the differences, P-restr: K=0 2,109.32 → 2,144.73; K=20 722.83 → 694.74; K=200 342.45 → 317.08;
K=2000 91.87 → 80.26.)

**Whole fold (328,009), matched predictions held fixed:**

| B | K=0 | K=20 | K=200 | K=2000 | month bootstrap | `verdict` |
|---|---|---|---|---|---|---|
| P-restr | +2.41 [−20.40, +20.49] | −0.86 [−2.04, +0.30] | **−0.83 [−1.17, −0.50]** | **−0.15 [−0.24, −0.06]** | +2.41 [−5.35, +9.77] | REJECT — no difference on all rows |
| P-full | +3.07 [−19.68, +21.36] | −0.43 [−1.72, +1.08] | **−0.93 [−1.24, −0.64]** | **−0.17 [−0.26, −0.08]** | +3.07 [−4.36, +10.13] | REJECT — no difference on all rows |

Against the logged OOF column instead of the reproduction the numbers are identical (the reproduction is bit-exact).

### 3.4 Trees and how the pooled model splits on `nm_matched`

Tree structure is read from the **seed-42 refit booster** of each pooled variant (the three seeds are averaged for
the predictions; only the first is dumped).

| | P-restr | P-full |
|---|---|---|
| trees per seed (3 seeds averaged) | **152** | **145** |
| internal nodes (first seed) | 19,152 | 18,270 |
| `nm_matched`: splits / trees using it / trees with it at the **root** | 171 / 78 of 152 / **0** | 113 / 61 of 145 / **0** |
| `nm_matched`: min depth, median depth | 2, 8.0 | 3, 7.0 |
| `eobt1_null`: splits / trees / root | 10 / 7 / 0 | 3 / 1 / 0 |
| `lobt_null`: splits / trees / root | 33 / 21 / 0 | 40 / 20 / 0 |
| the three regime indicators' share of total split gain | **1.72 %** | **1.57 %** |

Root split of every tree, counted over all trees — P-restr: `AIRCRAFT_TYPE_mvt` 35, `ref_med` 31, `STAND_mvt` 19,
`arh_med` 13, `ref_p10_fb` 11, `mvt_minus_sched` 10, `ref_p20` 5, `ADES_mvt` 4, `rwy_headway_next_s` 4,
`dep_tko_next_10m` 4, others ≤ 3. P-full: `AIRCRAFT_TYPE_mvt` 32, `ref_med` 32, `arh_med` 13, `ref_p10_fb` 12,
`STAND_mvt` 11, `ADES_mvt` 8, `nbr_mvt_minus_aobt_mean_30m` 7, `mvt_minus_sched` 6, `ref_p20` 6, others ≤ 5.

Tree 0 of P-restr, depths 1–3: root `ref_med ≤ 947.5` (gain 6.55e10, 1,406,226 rows), then `ref_med ≤ 702.25` /
`ref_med ≤ 1243.5`, then `ref_med ≤ 537.5` and three `mvt_minus_sched_slog` splits (6.98 / 7.18 / 7.39).
Tree 0 of P-full: root `ref_med ≤ 947.5`, then `ref_med ≤ 702.25` / `AIRCRAFT_TYPE_mvt ∈ {185}`, then
`ref_med ≤ 537.5`, `AIRCRAFT_TYPE_mvt`, **`mvt_minus_sched ≤ 14,403` on a node of 923 rows** (gain 4.32e10), `ref_med`.

Top features by total split gain — P-restr: `mvt_minus_sched` 8.23e11 (2,580 splits), `ref_med` 8.03e11,
`AIRCRAFT_TYPE_mvt` 2.85e11, `mvt_minus_sched_slog` 1.66e11, `ref_p20` 1.14e11, `STAND_mvt` 9.50e10, `ADES_mvt`
7.62e10, `ref_p10_fb` 6.65e10, …, **`nm_matched` 4.30e10 (12th)**. P-full: `ref_med` 8.67e11, `mvt_minus_sched`
8.20e11, `AIRCRAFT_TYPE_mvt` 2.75e11, `mvt_minus_sched_slog` 2.01e11, **`AIRCRAFT_OPERATOR_flt` 8.10e10 (5th)**,
`STAND_mvt` 7.77e10, …, **`nm_matched` 3.89e10 (13th)**.

So the pooled model does **not** put the regime at the top of the tree: no tree of either pooled model splits on
`nm_matched`, `eobt1_null` or `lobt_null` at the root, the earliest such split is at depth 2, and together they carry
1.6–1.7 % of the split gain. The leaves the unmatched rows fall into are therefore shared with matched rows above
depth 2 in most trees.

### 3.5 LIRF unmatched rows with ms ≥ 12 h (forward validation set)

Four rows; predictions in seconds:

| MVT_ID | y | ms | y − ms | logged = S-full = S-restr | P-restr | P-full |
|---|---|---|---|---|---|---|
| 200297323 | 87,480 | 56,520 | 30,960 | 32,351.7 | **28,161.2** | 27,935.0 |
| 200300302 | 87,598 | 60,599 | 26,999 | 34,539.7 | **22,541.8** | 22,517.8 |
| 201198951 | 87,605 | 86,102 | 1,503 | 34,093.3 | **25,733.1** | 25,472.3 |
| 201197366 | 47,767 | 47,763 | 4 | 32,399.5 | **24,287.4** | 24,462.6 |

Pooling moves all four predictions **down**, away from labels that are all ≥ 47,767: RMSE on these four rows
47,312.7 (baseline) → **55,069.2** (P-restr) / 55,192.0 (P-full); median prediction 33,246 → 25,010 / 24,967.
Wider bands, LIRF unmatched: ms ≥ 4 h (10 rows) RMSE 30,918.3 → 35,110.6 / 35,230.5, median prediction 27,612.6 →
27,651.5 / 26,734.9; ms ≥ 2 h (58 rows) 13,119.3 → 14,881.0 / 14,954.5, median prediction 7,248.0 → 8,784.4 / 8,719.5.
72 % of LIRF unmatched predictions move up under pooling, but the rows that dominate the metric move down.

The single largest prediction is not where the baseline's was: S-full's max 34,540 sits on the y = 87,598 row;
P-restr's max 38,155 sits on a LIRF row with y = 39,120, ms = 39,120 (sched-filled, predicted almost exactly).

## 4. Verdicts per sub-claim

| # | side | sub-claim | verdict | deciding observation |
|---|---|---|---|---|
| A4 | A | Training model B on all rows **improves unmatched predictions** | **REFUTED as stated** | §3.1: unmatched RMSE 2,109.32 → 2,144.73 (restricted columns) / 2,154.29 (the literal `shared_b`); whole fold 284.7038 → 287.1174 / 287.7726. The full-data difference is +35.41 / +44.98 s with CI [−409, +276] / [−397, +285], so it is a **non-significant worsening**, not an improvement; `prc.compare.verdict` returns REJECT ("no difference on all rows") for both. |
| A4b | A | …improves them **on the rows that are not LIRF** | **CONFIRMED** | §3.2: non-LIRF unmatched RMSE 1,034.74 → 486.57; better at nine of ten airports (EDDF 1,683 → 503, EHAM 1,184 → 286, LEBL 1,150 → 275, LFPG 1,217 → 684 …). §3.3: with the worst 200 rows removed −25.37 s [−36.36, −13.11] and at 2,000 removed −11.61 s [−15.19, −8.40], both CIs excluding zero, i.e. the body of the distribution genuinely improves. |
| A-cap | A | Capacity must follow the training set (`cfg` instead of `cfg.for_unmatched()`, 5b90919) | **NOT A DIFFERENCE under this config** | §2.1: `main.yaml` sets model B to 127 / 50 / 2000, identical to the main model, so both branches resolve to the same parameters. It would differ under the deleted `configs/p1_shared_b.yaml` (255 / 20 / 3000). |
| A-cols | A | "Schedule-only features" (the restricted list) | **measured; the restriction is the better of the two pooled forms, and a no-op without pooling** | §3.1: P-restr 2,144.73 vs P-full 2,154.29 (dropping the four `_flt` categoricals is worth −9.56 s of unmatched RMSE inside a pooled fit); S-restr is bit-identical to S-full (max |diff| 0.0) because LightGBM discards those single-bin columns in the unmatched-only fit. |
| A2 | A | "Extreme examples go from 33 to ≈ 2,000" (`docs/README.md` Phase 2' a) | **REFUTED** (already by item 31's pre-check; nothing here contradicts it) | `shared-b-no-transfer.md` §3 H: pooling adds 1 row at y ≥ 12 h, 3 at y ≥ 4 h, 103 at y ≥ 2 h. This run adds the fitted consequence: the pooled model has more trees (152 vs 45) but its error concentrates *further* on the tail (LIRF share of unmatched SSE 0.771 → 0.951). |
| B6 | B | Pooling **will not help the tail** | **CONFIRMED** | §3.2/§3.5: LIRF unmatched RMSE 8,578.38 → 9,689.57 / 9,733.56; LIRF ms ≥ 2 h 13,119.3 → 14,881.0; LIRF not-sched-filled (41 rows) 14,833.4 → 17,107.5; all four LIRF ms ≥ 12 h predictions move down and away from their labels (RMSE 47,312.7 → 55,069.2). |
| B5′ | B | The mechanism named by side B — the matched LIRF rows with the same `ms` carry the opposite mapping and outnumber the unmatched ones — is what the fitted model picks up | **CONFIRMED, consistent with the pre-check** | §3.4: no tree of either pooled model splits on `nm_matched` / `eobt1_null` / `lobt_null` at the root (earliest at depth 2; 1.6–1.7 % of split gain), so the unmatched rows share leaves with matched rows; §3.5: the predictions on the LIRF tail fall by 6,000–12,000 s, in the direction of the 4.4 : 1 matched majority documented in `shared-b-no-transfer.md` §3 E/F. |
| C | — | How many trees the pooled model uses | **measured** | 152 per seed (restricted) / 145 (full), three seeds averaged, vs 45 for the split model; early stopping now reads a 185,674-row holdout instead of a 1,913-row one. |
| D | — | How it splits on `nm_matched` | **measured** | §3.4: 171 splits in 78 of 152 trees (P-restr), 113 in 61 of 145 (P-full), never at the root, min depth 2/3, median depth 8/7, 12th and 13th by total split gain. |
| E | — | What it predicts on the LIRF unmatched ms ≥ 12 h rows | **measured** | §3.5 table: 32,351.7 / 34,539.7 / 34,093.3 / 32,399.5 → 28,161.2 / 22,541.8 / 25,733.1 / 24,287.4 (P-restr). |

### Verdict on the conflict between the two source claims

**The measurement supports side B (item 31, M `shared-b-no-transfer`) on the question that decides the score, and
side A (item 30, E H7 / H §8) only on the part of the data that does not decide it.**

Pooling is not a big lever in the direction claimed: on the forward fold it makes the scored metric slightly worse
(unmatched 2,109.32 → 2,144.73; whole fold 284.7038 → 287.1174), and the worsening sits exactly where side B said it
would — the LIRF tail (8,578 → 9,690), with all four LIRF ms ≥ 12 h rows predicted lower than before. At the same
time side A's stated mechanism is real off the tail: the 1.74 M extra rows halve the non-LIRF unmatched error
(1,034.74 → 486.57) and produce an improvement that survives the K = 200 and K = 2000 robustness checks with CIs
excluding zero — the only such result among the variants. Under the project's own decision rule (`prc.compare.verdict`,
which requires the full-data comparison to exclude zero on the improving side) both pooled variants are **REJECT**.

## 5. Not determinable locally

- **The LB effect.** The ranking labels are unknown; the ranking set has 383 LIRF unmatched rows (181 with mms ≥ 2 h,
  11 with mms ≥ 12 h) whose true labels decide whether moving the LIRF tail predictions down costs or earns seconds.
  A probe submission (v4-matched + pooled-unmatched) is the only local measurement of it, and was not run.
- **The other two folds.** Only `forward` was run, as instructed. The forward fold contains 4 LIRF unmatched rows
  with ms ≥ 12 h and 58 with ms ≥ 2 h; item 9's three-fold unmatched set has 8,378 rows. Whether the +35.41 s sign
  holds on `jan` / `jul` is not measured here; the month-level bootstrap has only 2 months and gives
  [−79.80, +143.45].
- **Pooling combined with the one lever that worked.** `results/mms-mixture-reparam.md` reports its variant A
  (target y − ms on LIRF unmatched rows) at 2,326.5 → 1,632.1 on its own three-fold 8,378-row baseline. That is a
  different row set and a different baseline from this report's 2,109.32 on 3,005 forward rows; whether a pooled fit
  under that reparameterisation behaves differently was not fitted.
- **Hyper-parameters for the pooled regime.** 127 leaves / min_data 50 / early stopping on a 185,674-row holdout are
  the current config's numbers, not numbers chosen for a 1.76 M-row model B. A pooled fit with, say, a
  regime-stratified inner holdout or a sample weight on unmatched rows is a different experiment
  (`train.py::_fit` passes no sample weights).
- **Whether `AIRCRAFT_TYPE_mvt` learned on matched rows helps or hurts the LIRF rows**, where it is null on 1,487 of
  1,488 training rows: it is the most frequent root split of both pooled models, but its per-regime contribution was
  not isolated.

## 6. Facts vs inference

**Facts (produced by this run or read from the repository).** Every number in §2.5, §3.1–§3.5 and the verdict table:
the bit-exact reproduction of `inv_baseline_forward-67e76a5dfc` (2,109.3167 / 201.6633 / 284.703845, rounds 45, row-by-row
max |diff| 0.0); S-restr bit-identical to S-full; pooled unmatched RMSE 2,144.73 (89 columns) and 2,154.29
(93 columns) on 3,005 rows; whole-fold 287.1174 / 287.7726; the per-airport table; the robustness curves and
`verdict` strings; 152 / 145 trees per seed; 171 / 113 `nm_matched` splits, none at the root, 1.72 % / 1.57 % of split
gain; the four LIRF ms ≥ 12 h rows' predictions; LIRF share of unmatched SSE 0.771 → 0.951; the resolved config and
the equality of `for_unmatched()` with `cfg` under it; the four categoricals' null rates on this fold; the content of
commits 5b90919 and ea04ec9 and the deleted `configs/p1_shared_b.yaml` (255 / 20 / 3000).

**Inference (not measured here).**
- I1 The pooled model's loss on the LIRF tail and its gain elsewhere are consistent with one mechanism — leaves shared
  between regimes below depth 2 (§3.4) pulling LIRF unmatched predictions toward the matched majority at the same
  `ms` (`shared-b-no-transfer.md` §3 F: 2,751 matched vs 627 unmatched at LIRF ms ≥ 2 h, P(y ≥ 2 h) 0.016 vs
  0.60–0.85) while supplying the non-LIRF unmatched rows with the structure they lack. The run measures the two
  outcomes and the absence of a root-level regime split; it does not prove that this is the causal path.
- I2 The larger tree count (45 → 152) comes with an inner holdout in which unmatched rows are 1,913 of 185,674
  (1.03 %), so the stopping point is chosen almost entirely by matched-row error. That is a fact about the holdout composition; that
  it is *why* the LIRF tail suffers is inference.
- I3 Item 9's variant A (LIRF reparameterisation) and this item move different parts of the same 3,005 rows — the
  tail and the body respectively. Whether they compose is untested (see §5).
- I4 `results/mms-mixture-reparam.md` reports 2,177.4 for its forward-fold baseline reproduction where this script
  gets the logged 2,109.3167. Reading its script, it adds its own `ym` fold key with `with_columns` and does not
  remove it from `visible_numeric_columns`, so model B there was given one extra feature (and a differently ordered
  matrix, which `feature_fraction: 0.8` samples by index). This script drops `ym` from the feature list and preserves
  the parquet's column order, which is what makes the reproduction bit-exact here. That the `ym` column is the cause
  of the other report's +68.1 s gap is inference; the code difference is a fact.
