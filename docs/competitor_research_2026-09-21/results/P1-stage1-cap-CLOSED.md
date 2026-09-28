# P1 stage 1 — the non-LIRF unmatched upper cap: **CLOSED**

Measured 2026-09-24 on the stage-0 OOF (`experiments/oof_main_reparam-bc60017878.parquet`,
672,428 rows, folds jan/jul/forward), on the code with the G11 categorical-encoding fix — so this
is the first sweep of this rule taken on an instrument whose two runs agree.

Script: `docs/competitor_research_2026-09-21/scripts/p1_stage1_cap_sweep.py` (read-only).

Rule under test: `pred := min(pred, c)` on rows the unmatched model served, at airports other
than LIRF. Gate population 7,841 rows; maximum label in it 84,240 s.

## The pre-registered decision, set before the measurement

From `plans/P1-unmatched-lane.md` §Stage 1, "How we would know this failed": **any** airport or
fold with a positive SSE delta; **or** helped/hurt below 3:1; **or** the month-block CI straddling
zero. Any of those → record CLOSED with the reason, so it is not reopened a third time.

## Result

| cap | moved | help/hurt | ratio | ΔRMSE | folds+ | airports+ | CI95 month | CI95 day | verdict |
|---|---:|---:|---:|---:|---:|---|---|---|---|
| 1,500 | 889 | 480/409 | 1.2 | **+0.3810** | 3 | EDDM EGLL EHAM LFPG LTFM | straddles | straddles | fails all three |
| 1,800 | 369 | 234/135 | 1.7 | **+0.0291** | 3 | EDDM LFPG LTFM | straddles | straddles | fails all three |
| 2,100 | 198 | 126/72 | 1.8 | −0.0764 | 0 | EDDM LFPG LTFM | straddles | straddles | fails airports, ratio, CI |
| 2,400 | 108 | 68/40 | 1.7 | −0.1101 | 0 | LFPG | straddles | straddles | fails airports, ratio, CI |
| 3,000 | 39 | 27/12 | 2.2 | −0.1016 | 0 | — | straddles | excludes | fails ratio and month CI |
| 3,600 | 17 | 11/6 | **1.8** | −0.0685 | 0 | — | **excludes** | **excludes** | **fails ratio** |

**Every cap fails at least one pre-registered criterion.** The best of them, 3,600, has a CI
excluding zero on both bootstraps and is still only 11 helped against 6 hurt. The 3:1 rule exists
because a cap that helps many and hurts nearly as many is a distribution shift rather than a
guard, and 11/6 over 17 rows is close to a coin toss. The effect is real and tiny; its shape is
wrong.

CLOSED. Do not reopen without a new mechanism, not a new threshold.

## What the sweep found that the plan did not expect

**LFPG is the airport that resists the cap.** It is the only positive-SSE airport at 2,400 and one
of three at 2,100 and 1,800. Its long unmatched predictions are more often right than wrong, so
truncating them costs.

That is consistent with the stage-0 per-airport breakdown, which is new:

| airport | unmatched RMSE (pooled OOF) |
|---|---:|
| LIRF | 4,232.2 |
| **LFPG** | **2,623.0** |
| EGLL | 998.4 |
| EDDM | 717.4 |
| the other six | 282–585 |

The plan expected the unmatched lane's difficulty to be LIRF. LFPG sits at the same order of
magnitude and nothing in this repository has looked at it. Whether it shares LIRF's mechanism (a
BLOCK_TIME filled with SCHED_TIME) is **not established** — that is a question for a later stage,
and it is a mechanism question, not a threshold one.

## Two caveats on the instrument, recorded rather than buried

* The month-block arm resamples **4 blocks** (Jan, Jul, Nov, Dec), which is very coarse; the day
  block resamples 122 and is reported beside it. The two agree on every row of the table, which is
  the only reason the month arm is usable here at all.
* The pooled OOF the sweep runs on is the coalesce of jan/jul/forward. The sidecar written in stage
  0 is what makes "which fold supplied this row" answerable at all, and this script uses it.
