# Can anomalous rows be told apart from information visible at prediction time? (2026-09-24)

**Script (read-only; no training run, no submission, nothing written outside `/tmp`):**
`docs/competitor_research_2026-09-21/scripts/visible-anomaly-detection.py`, run stage by stage as
`POLARS_MAX_THREADS=<1-3> .venv/bin/python docs/competitor_research_2026-09-21/scripts/visible-anomaly-detection.py <stage>`
with `<stage>` in `counts | marginal | cats | combo | widen | part2 | price | hardcase | lobt`.
Artefacts land in a scratch directory; nothing in `src/`, `configs/` or `outputs/` was touched.
Fits were kept small and, where the text says so, subsampled; a few one-off checks were run
inline from the shell and are marked **(inline)** with the query given in full.

**Inputs (read, never written).** `data/work/feat_train.parquet` (2,085,047 rows, all
`PHASE_mvt == "DEP"`, 137 columns), `data/work/feat_rank.parquet` (344,841 scored 2026 departure
rows, same 137 columns), `experiments/oof_main_reparam-bc60017878.parquet` (672,428 non-null
`oof`, months 1/7/11/12, pooled RMSE **281.8411**), `outputs/main.parquet` (the shipped v13
predictions, 344,841 rows), `data/raw/submitting.parquet` (344,841 rows — the scored set is
exactly the departure rows of `feat_rank`).

**Notation.** `y = TAXITIME_SEC_mvt`; `ms = mvt_minus_sched = MVT − SCHED`
(`src/prc/features.py:63`); lane = `nm_matched` (1 = model A, has the NM clocks; 0 = model B, all
four NM clocks null). `y = MVT − BLOCK` holds on all 2,085,047 training rows.

**What "visible" means here, and it is checked, not assumed.** [S] (inline)
`pl.scan_parquet('data/work/feat_rank.parquet').collect().null_count()` over all 137 columns:
exactly **two** columns are null on all 344,841 scored rows — `BLOCK_TIME_UTC_mvt` and
`TAXITIME_SEC_mvt`, i.e. `BLANKED_DEP_COLS` in `src/prc/schema.py:52`. Everything else is
present at the rates you would expect (`mvt_minus_aobt3` and the other NM offsets null on
5,290 / 344,841 = 1.534 %; `stand_arrival_gap` null on 18.27 %; `gust_ms` null on 96.6 %). So a
feature is *visible* iff it does not read `BLOCK_TIME_UTC_mvt` or `TAXITIME_SEC_mvt`. Every
candidate below is labelled accordingly; the standing trap in this project is `bs = BLOCK − SCHED`,
which equals `ms − y` identically and is therefore a re-expression of the label.

---

## 0. The answer

1. **Yes for the matched lane, and it is exact, not statistical.** `|y − mvt_minus_lobt| ≤ 3,606 s`
   holds on **all 2,062,623 matched training rows, 0 violations**, so `mvt_minus_lobt` — visible —
   pins `y` to a ±1 h window. `mvt_minus_lobt > 43,200` selects **1 row of 2,062,623**, and it is
   the single matched corrupted row (precision 1.000, recall 1.000, zero false positives).
   The 2026 frame contains **exactly one** such row, and the shipped v13 prediction for it is
   84,866 s — the production `lobt_bound` already acts on it. §2.1, §5.2. [S]
2. **Mostly yes for the unmatched lane too, and the claim that "visible information points the
   wrong way" is wrong as a general statement.** Of the 15 unmatched corrupted rows, **12 sit above
   the 95th percentile of `ms` in their own lane and 11 above the 99.8th**; `ms` alone scores
   **AUC 0.8896** over the 22,424-row lane and **0.9895** inside LIRF's 1,487. The two rows the
   earlier claim cited (LSZH `ms` 939, LFPG `ms` 1,740) are percentile 15.6 and 24.6 — they are
   the exceptions, quoted as if they were the rule. §2.2, §2.3. [S]
3. **But the detector buys nothing, because the model already has it and already acts on it.**
   `mvt_minus_sched` is a model feature, and `unmatched_sched_reparam` at LIRF is precisely the
   mechanism that converts it into large predictions. The shipped v13 file already predicts
   30,644–66,693 s on the eleven 2026 LIRF unmatched rows with `ms ≥ 43,200`. Replacing the
   prediction on the best-precision visible cell with 87,000 s measures **+16.2 s of OOF RMSE**
   (281.84 → 298.07); the best *additive* shift on that cell, fitted leave-one-month-out, is
   −1.5 s at one threshold, +1.2 s at another, and changes sign across folds. §5.3, §5.4. [S]
4. **The rows that actually cost us are the ones nothing visible reaches.** Two LFPG rows are
   **19.1 %** of the whole OOF squared error (one of them 13.0 % on its own). Both have a *small*
   `ms` (1,740 and 2,043 s, percentile 4.98 and 6.09 among 3,763 LFPG unmatched peers) and sit
   inside the bulk of every other visible feature; the two disagree in sign on every feature where
   either is extreme. With n = 2 there is nothing to fit and nothing to claim. §3. [S]
5. **Combination does not rescue it at this class size.** A leave-one-month-out LightGBM over the
   full visible feature set scores **below `ms` alone** on the unmatched lane. With 15 positives
   the permutation control is the point of the exercise, not a footnote. §4. [S]
6. **Part 2 (heteroscedasticity) is the part with a real, reusable answer**, and it is not a lever
   either: visible features predict *error magnitude* well, but almost all of that is the two
   things the model already conditions on — lane and airport. §6.
7. **The "well under one expected corrupt row in 2026" estimate is wrong.** Every stratification
   gives **2.6 to 7.0** expected, with Poisson 95 % lower bounds of 1.5 to 2.8. §7.1. [S]

---

## 1. The positive class, and why 16 is the number

`counts` stage. Query: counts on `feat_train.parquet` of `y > 72,000` / `y ≥ 43,200` / `≥ 14,400` /
`≥ 7,200`, split by `nm_matched == 0` and `ADEP_mvt == "LIRF"`.

| class | n | unmatched | LIRF |
|---|---:|---:|---:|
| `y > 72,000` (the day-rollover signature) | **16** | 15 | 14 |
| `y ≥ 43,200` | 34 | 33 | 31 |
| `y ≥ 14,400` | 122 | 119 | 116 |
| `y ≥ 7,200` | 584 | 481 | 480 |

[S]. Matches `docs/investigation_2026-09-16/results/block-day-rollover.md` §3.1 exactly.

**15 of the 16 are in the unmatched lane, so the lane is the analysis unit.** [S] (inline) A
projection of the 16 rows onto the clock columns shows `mvt_minus_aobt3`, `mvt_minus_lobt`,
`mvt_minus_eobt1`, `mvt_minus_iobt` all null on 15 of 16; only `AEZ146R` (LIRF, 2025-07-14) is
matched, with `mvt_minus_aobt3 = 87,181`. Comparing the 15 against the 2.08 M-row population as a
whole would therefore measure "is this row unmatched", which is already a model feature
(`nm_matched`) and already the routing key. Every per-feature number below is computed **inside the
lane**.
