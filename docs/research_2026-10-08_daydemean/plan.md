# Within-day demeaned context for model A (dd): pre-registration (2026-10-08, before the run)

## Why
- The gap analysis (`docs/research_2026-10-08_gap`) listed within-day-demeaned neighbour features as a legal idea. Two competitors report −1.51 s and −0.73 s with it on their own boards.
- Our error decomposition found a matched-lane airport-hour residual ceiling of −2.0 to −3.7 s.
- Raw hour and day aggregates did not help a residual model. Demeaning instead gives the main model "how busy or late is this moment relative to the rest of the day", which is a different signal.

## Change
- `fe_day_demean: true` on top of `main_flseq_osm_nolirf_l2_opt_l6` (177174f530).
- For each of 15 label-free context columns (`train.DD_SRC`: neighbour delay and proxy means, counts, queue, headway, backlog, the proxy itself, and others), add dd_<col> = col − its airport × UTC-day mean.
- Each frame is demeaned over itself.
- Model A only. Model B never sees the dd_ columns.

## Comparison
main_flseq_osm_nolirf_l2_opt_l6_dd vs main_flseq_osm_nolirf_l2_opt_l6-177174f530 (all folds).

## Verdict: ADOPT only if all of the following hold
1. Unmatched rows are bit-identical.
2. `prc.compare` returns ADOPT: the full CI and the K=200 CI both exclude 0 on the improving side.
3. jan, jul and forward each improve on the whole frame.
4. The STRICT VERDICT is ADOPT. This change touches every airport, so the strict rule applies.

## Result (2026-10-08, `compare.txt`)
**REJECT.** l6-177174f530 → l6_dd-cc501d0804.
- Whole frame: +0.09 s, airport-day CI [−0.09, +0.29].
- By fold: jan +0.04, jul +0.52, forward −0.20. 4 of 10 airports improve.
- Reading: the existing nbr_*/seq_*/L6 columns already carry the within-day relative signal, so demeaned copies add nothing. This matches the decomposition, where label-free time-local aggregates did not reach the airport-hour residual.
