# 26. `fold-shape-vs-ranking` — results (2026-09-16)

Source item: plan.md §1 #26 (from P 8). Script: `docs/investigation_2026-09-16/scripts/fold-shape-vs-ranking.py`
(run with `POLARS_MAX_THREADS=1 .venv/bin/python …`; lazy scans, 5–6 columns; no model training).
Code read at HEAD `f250a9f`: `src/prc/cv.py` lines 26–51 (`month_block_folds`), `src/prc/train.py` lines
407–455 (`run_cv`), 329 / 348 (routing filters on `PROXY = "mvt_minus_aobt3"`).

## 1. Hypothesis (as consolidated in plan.md)

(a) No CV fold has the production shape (train on 12 months of 2025 → predict Jan 2026 and Jul 2026, i.e. a
1-month and a 7-month horizon across a year boundary).
(b) `overall` in `run_cv` is built with `pl.coalesce` in fold order, so the `jul_fwd` predictions never enter it
(overall.n = 672,428 while Σ fold n = 863,141).
(c) The unmatched share differs between training (1.08 %) and ranking (1.53 %).
(d) Routing rule (`mvt_minus_aobt3` null) is applied identically in CV and ranking paths; 5,290 ranking rows,
7 of them with `FLIGHT_ID_mvt` (training 46 of 22,470).

## 2. Procedure

1. Read `cv.py:month_block_folds` and `train.py:run_cv`; called `month_block_folds` (pure function) on the
   month keys actually present in `data/work/feat_train.parquet` (`year*100+month` of `MVT_TIME_UTC_mvt`).
2. Per-month row counts and unmatched counts under both §0 definitions (routing = `mvt_minus_aobt3` null;
   D = `FLIGHT_ID_mvt` null) on `feat_train.parquet` and `feat_rank.parquet`; also `MVT_ID_mvt` uniqueness,
   null/non-finite labels, and whether `AOBT_3_flt` null ≡ `mvt_minus_aobt3` null (preconditions for reading
   `overall.n` as a row count and for the two routing wordings in §0 being the same rule).
3. Fold geometry table (train-month count, horizon from last training month to each validation month,
   number of training months later than the validation month, year-boundary crossing, season) vs production.
4. All 35 records of `experiments/experiments.jsonl`: fold names present, Σ fold n, `overall.n`,
   `overall.rmse_full` vs the pooled RMSE `sqrt(Σ n_i·rmse_i² / Σ n_i)` over {jan, jul, forward} and over
   {jan, jul_fwd, forward}. The pooled RMSE identifies which fold's predictions are inside `overall`
   without re-running anything.
5. Which months carry a non-null prediction in `experiments/oof_main-48c19bc830.parquet`.
6. Replay of the exact `run_cv` join+coalesce expression on a 4-row toy frame with two folds predicting the
   same rows.

## 3. Results

### 3.1 Folds produced on the real month keys (202501 … 202512, all 12 present)

Fold order returned: `['jan', 'jul', 'jul_fwd', 'forward']` (also asserted by `tests/test_train.py:26`).

| fold | validation months | n_valid | training months | n_train |
|---|---|---|---|---|
| jan | 2025-01 | 153,706 | 2025-02 … 2025-12 (11) | 1,931,341 |
| jul | 2025-07 | 190,713 | 2025-01 … 06, 2025-08 … 12 (11) | 1,894,334 |
| jul_fwd | 2025-07 | 190,713 | 2025-01 … 2025-06 (6) | 1,005,519 |
| forward | 2025-11, 2025-12 | 328,009 | 2025-01 … 2025-10 (10) | 1,757,038 |

`n_train` here is rows in the training months; the records' `train_rows` match (1,931,341 / 1,894,334 /
1,757,038 in `ab_split-ca87cb5af7`), i.e. no labelled rows are dropped (`label_null` = 0, `label_nonfinite` = 0).

### 3.2 Fold geometry vs production geometry

Horizon = calendar months from the last training month to the validation month (negative: training data
extends past the validation month). "Train months after valid" = number of training months later than the
validation month.

| geometry | train months | predicts | horizon (months) | train months after valid | crosses year boundary | season of predicted months |
|---|---|---|---|---|---|---|
| jan | 11 | 2025-01 | −11 | 11 | no | Jan |
| jul | 11 | 2025-07 | −5 | 5 | no | Jul |
| jul_fwd | 6 | 2025-07 | +1 | 0 | no | Jul |
| forward | 10 | 2025-11, 2025-12 | +1, +2 | 0 | no | Nov, Dec |
| **production** | **12** | **2026-01, 2026-07** | **+1, +7** | **0** | **yes** | **Jan, Jul** |

Which production properties each fold reproduces:

| property | jan | jul | jul_fwd | forward |
|---|---|---|---|---|
| 12 training months | no (11) | no (11) | no (6) | no (10) |
| strictly forward (0 training months after valid) | no | no | yes | yes |
| horizon ≥ 7 months (the Jul-2026 half, 55.7 % of ranking rows) | no | no | no | no (max 2) |
| horizon 1 month (the Jan-2026 half, 44.3 % of ranking rows) | no | no | yes | yes (Nov) |
| predicts a January | yes (leaky) | no | no | no |
| predicts a July | no | yes (leaky) | yes | no |
| crosses a year boundary | no | no | no | no |

No fold satisfies "12 training months", "7-month horizon" or "year boundary"; no forward fold predicts a
January; the only fold predicting a July forward has 6 training months. Combination "forward + Jan season"
does not exist; "forward + Jul season" exists only as `jul_fwd`.

### 3.3 `overall` construction (run_cv, train.py:427–444)

Code: `oof` starts as all-null; for each fold in order the fold's predictions are left-joined and
`pl.coalesce([oof, _p])` keeps the existing value where one exists. `jul` runs before `jul_fwd` and both
predict exactly the same 190,713 rows (2025-07), so `jul_fwd`'s predictions are never written into `oof`.
`overall = metrics(...)` on `oof` after `drop_nulls`.

Toy replay (script §F): fold 1 → rows {1,2} = 10, 20; fold 2 → rows {1,2} = 99, 99; fold 3 → row 3 = 30.
Result: rows 1,2 keep 10, 20; row 3 = 30; row 4 null. Fold-2 values absent.

Experiments log (35 records; full table in script output). Preconditions hold: `MVT_ID_mvt` unique
(2,085,047 = rows), so joins do not duplicate rows and `overall.n` is a row count.

| record group | folds present | Σ fold n | overall.n | overall.rmse = pooled(jan,jul,forward)? | pooled(jan,jul_fwd,forward) |
|---|---|---|---|---|---|
| 10 records up to `det_b` (pre 09-15) | jan, jul, forward | 672,428 | 672,428 | yes (equal to 3 dp, all 10) | — |
| 18 screening records | forward only | 328,009 | 328,009 | yes | — |
| `rep_jan_off/on` | jan | 153,706 | 153,706 | yes | — |
| `leak_check-54a21538de` | jul, jul_fwd | 381,426 | **190,713** | overall 375.012 = jul 375.01 (jul_fwd = 421.69) | 421.694 |
| `pw_noise-c8f996bf3d` | jan, jul, jul_fwd, forward | 863,141 | 672,428 | 337.321 = 337.321 | 350.892 |
| `pw_both-91b711793b` | same | 863,141 | 672,428 | 336.350 = 336.350 | 349.302 |
| `pw_noise2-7672c5b2b7` | same | 863,141 | 672,428 | 337.537 = 337.537 | 351.080 |
| `f10_plannedflight-b416261a17` | same | 863,141 | 672,428 | 337.481 = 337.481 | 351.421 |
| `main-63086f0648` | same | 863,141 | 672,428 | 340.395 = 340.395 | 347.978 |
| `main-48c19bc830` | same | 863,141 | 672,428 | 334.582 = 334.582 | 347.880 |

In every 4-fold record `overall.rmse_full` equals the pooled RMSE of {jan, jul, forward} to 3 decimals and
differs from the pooled {jan, jul_fwd, forward} by 7.6–14.0 s. In `leak_check`, `overall` equals the `jul`
figure exactly. `overall.n` = 153,706 + 190,713 + 328,009 = 672,428 = Σ fold n − n(jul_fwd).

Months with a prediction in `oof_main-48c19bc830.parquet`: 2025-01 (153,706), 2025-07 (190,713), 2025-11
(162,332), 2025-12 (165,677); zero for the other eight months; 672,428 non-null of 2,085,047 rows. The
2025-07 predictions in this file are the `jul` fold's (trained on Aug–Dec 2025 among others), by the pooled-RMSE
identity above.

Composition of `overall` (672,428 rows): Jan-2025 22.9 %, Jul-2025 28.4 % (leaky fold), Nov+Dec-2025 48.8 %.
`jul_fwd` appears only in the per-fold block of the record (`folds.jul_fwd`), and `configs/main.yaml` /
`configs/submitted_v4.yaml` set no `folds:` key, so production CV runs all four and reports this `overall`.

### 3.4 Unmatched share per month, both definitions

Training (`feat_train.parquet`, 2,085,047 rows, `MVT_ID_mvt` unique, no null labels):

| month | rows | unmatched (routing: proxy null) | % | unmatched (D: FLIGHT_ID null) | % | proxy null but FLIGHT_ID present |
|---|---|---|---|---|---|---|
| 2025-01 | 153,706 | 1,435 | 0.934 | 1,432 | 0.932 | 3 |
| 2025-02 | 143,732 | 1,110 | 0.772 | 1,109 | 0.772 | 1 |
| 2025-03 | 164,449 | 1,082 | 0.658 | 1,081 | 0.657 | 1 |
| 2025-04 | 175,288 | 1,085 | 0.619 | 1,083 | 0.618 | 2 |
| 2025-05 | 185,202 | 1,605 | 0.867 | 1,603 | 0.866 | 2 |
| 2025-06 | 183,142 | 2,502 | 1.366 | 2,499 | 1.365 | 3 |
| 2025-07 | 190,713 | 3,938 | 2.065 | 3,934 | 2.063 | 4 |
| 2025-08 | 191,182 | 2,330 | 1.219 | 2,323 | 1.215 | 7 |
| 2025-09 | 183,950 | 2,465 | 1.340 | 2,458 | 1.336 | 7 |
| 2025-10 | 185,674 | 1,913 | 1.030 | 1,910 | 1.029 | 3 |
| 2025-11 | 162,332 | 1,356 | 0.835 | 1,349 | 0.831 | 7 |
| 2025-12 | 165,677 | 1,649 | 0.995 | 1,643 | 0.992 | 6 |
| **total** | **2,085,047** | **22,470** | **1.078** | **22,424** | **1.075** | **46** |

Ranking (`feat_rank.parquet`, 344,841 rows = submission row count, `MVT_ID_mvt` unique):

| month | rows | share of ranking | unmatched (routing) | % | unmatched (D) | % | proxy null but FLIGHT_ID present |
|---|---|---|---|---|---|---|---|
| 2026-01 | 152,719 | 44.3 % | 2,453 | 1.606 | 2,449 | 1.604 | 4 |
| 2026-07 | 192,122 | 55.7 % | 2,837 | 1.477 | 2,834 | 1.475 | 3 |
| **total** | **344,841** | | **5,290** | **1.534** | **5,283** | **1.532** | **7** |

`proxy_ok_but_unmatched` (FLIGHT_ID null but proxy present) = 0 in both sets; `AOBT_3_flt` null ≡
`mvt_minus_aobt3` null on every row of both sets (0 disagreements), so the two wordings of the routing rule
in §0 are the same rule.

Derived from the two tables: unmatched share of the training months with the same calendar months as the
ranking set (Jan+Jul 2025) = (1,435 + 3,938) / (153,706 + 190,713) = **1.560 %**; applying the 2025 per-month
rates to the 2026 month mix gives 1.564 %; observed ranking share 1.534 %. Per month: Jan 2026 1.606 % vs
Jan 2025 0.934 % (×1.72); Jul 2026 1.477 % vs Jul 2025 2.065 % (×0.72). The training range across months is
0.619 % (Apr) – 2.065 % (Jul) — P's "1,082 (Mar)–3,938 (Jul)" gives the count range; the minimum count is
Mar (1,082) but the minimum rate is Apr (0.619 %).

## 4. Verdict per sub-claim

| sub-claim | verdict | basis |
|---|---|---|
| (a) No fold has the production shape (12 mo → Jan+Jul 2026; horizons 1 and 7) | **CONFIRMED** | §3.1–3.2: train-month counts 11/11/6/10; max forward horizon 2; no fold crosses a year; no forward fold predicts a January |
| (b) `overall` built by coalesce in fold order; `jul_fwd` excluded; 672,428 vs 863,141 | **CONFIRMED** | code train.py:427–444; toy replay; all 6 four-fold records have overall.n 672,428 and overall.rmse ≡ pooled(jan,jul,forward) to 3 dp; `leak_check` overall = jul exactly; OOF file has predictions only in Jan/Jul/Nov/Dec |
| (c) Unmatched share train 1.08 % vs ranking 1.53 % | **CONFIRMED** as stated (1.078 / 1.534 routing; 1.075 / 1.532 D). Refinement (fact): same-calendar-month training share is 1.560 %, so the aggregate gap is month composition; per-month direction differs (Jan up ×1.72, Jul down ×0.72) |
| (d) Routing rule identical in both paths; 5,290 / 7 and 22,470 / 46 | **CONFIRMED** | train.py:329/348 is the single `predict_fold` used by run_cv and predict_ranking; counts reproduced exactly |
| P's line citations | cv.py "96–121" **REFUTED** at HEAD (folds are lines 26–51); train.py 329/348 **CONFIRMED** |
| Ranking month composition 152,719 / 192,122 | **CONFIRMED** |
| "Consequence for CV/LB agreement" | **UNDETERMINED** — see §5 |

## 5. Undeterminable parts (would need a run)

- Whether the `overall` mixture (22.9 % leaky Jan fold, 28.4 % leaky Jul fold, 48.8 % forward Nov/Dec) ranks
  configurations differently from a 12-month → 2026 evaluation cannot be established here; the LB is the only
  instrument with the production shape, and per-fold ordering of past submissions vs LB would still be an
  n = 9 comparison.
- Whether the 7-month horizon (Jul 2026) or the Jan-2026 unmatched uptick (×1.72 in count share) changes model B's
  error is unknowable without ranking labels.
- The one existing measurement of leak cost, `jul` 375.01 vs `jul_fwd` 421.69 (`leak_check-54a21538de`,
  README §"どの fold"), confounds "leak removed" with "training data 11 → 6 months" (README states this itself);
  separating them needs a run (e.g. `jul` restricted to 6 later months, or `forward` with 6 months).

## 6. Facts vs inference

**Facts (measured or read from code at HEAD f250a9f):**
- Training month keys: exactly 202501–202512; ranking: exactly 202601 (152,719) and 202607 (192,122).
- Fold definitions and row counts as in §3.1; fold order jan, jul, jul_fwd, forward.
- `run_cv` coalesces fold predictions in that order; `jul` and `jul_fwd` predict the same 190,713 rows.
- All six 4-fold records: overall.n = 672,428 = Σ n − n(jul_fwd) = 863,141 − 190,713; overall.rmse_full equals
  pooled RMSE of {jan, jul, forward} to 3 dp in all six; `leak_check` (jul + jul_fwd) has overall.n 190,713 and
  overall.rmse 375.012 = jul.
- `oof_main-48c19bc830.parquet`: predictions only for 2025-01/07/11/12; 672,428 non-null.
- Unmatched per month, both definitions, both sets: §3.4. Totals 22,470 / 22,424 and 5,290 / 5,283; the
  46 / 7 difference rows are all proxy-null-with-FLIGHT_ID; the reverse set is empty. `AOBT_3_flt` null ≡ proxy null.
- `configs/main.yaml` and `configs/submitted_v4.yaml` do not set `folds:`; 18 screening configs set `[forward]`;
  `leak_check.yaml` sets `[jul, jul_fwd]`. `jul_fwd` was added in commit 16c910e (2026-09-15); the first 10
  records predate it and have three folds.
- Labels: 0 null, 0 non-finite in feat_train; `MVT_ID_mvt` unique in both frames.

**Inference (not measured):**
- The 2025-07 predictions inside `overall` and inside `oof_main-48c19bc830.parquet` are the `jul` fold's; this
  follows from the coalesce order plus the pooled-RMSE identity, not from a per-row provenance tag (the OOF file
  has none).
- The aggregate 1.08 % vs 1.53 % gap is attributable to month composition because the same-months training share
  is 1.560 %; the per-month opposite movements (Jan up, Jul down) are a 2025→2026 change whose cause is not
  examined here.
- Whether any of the above moves the LB is not inferable from this item.
