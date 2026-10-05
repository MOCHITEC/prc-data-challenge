# L6 gate retention (fe_retention) — pre-registration (2026-10-05)

## Background
In `congestion.md`, the overdue counts gave −411 s² [−586, −259] on top of a rich control that already contains
the existing neighbour / runway-sequence / L2 / L3 columns. The 1 ↔ 7 month swap gave −425. Adversarial AUC ≤ 0.541.

## Change
L6, `src/prc/levers.py:_retention_frame`, model A only, default off, existing ids unchanged:
- `ret_overdue_eobt`: departures at the same airport with EOBT_1 ≤ q < AOBT_3, where q is the flight's own AOBT_3
  (pairs with 0 < AOBT_3 − EOBT_1 < 2 h).
- `ret_overdue_sched`: the same with SCHED in place of EOBT_1.
- `ret_overdue_eobt_rwy`: `ret_overdue_eobt` restricted to the same runway.
- `ret_dly15`: mean of AOBT_3 − SCHED over pushes in [q − 15 min, q).

## Arms (forward fold; production lineage includes L2)
| arm | config |
|---|---|
| ret_base | main_flseq_osm_nolirf_l2 + folds [forward] |
| ret_l6 | ret_base + fe_retention: true |

## Verdict (unchanged after the runs start)
- **ADOPT** if `prc.compare ret_base ret_l6` returns ADOPT under the team rule: the full-row CI and the K=200 CI
  both exclude 0 on the improving side.
- Report next to it: STRICT VERDICT, the body/tail split (y ≤ 3600), and per-airport results.
- If adopted, re-run all folds on the production members and re-verify with `make compare`.

## Result (2026-10-05, `l6_verdict.txt`)
- **ADOPT.** ret_base-338abfb11c → ret_l6-65265584c7.
  - Full rows −0.76 s [−0.95, −0.56]; K=200 −0.91 s [−1.10, −0.70].
- **STRICT VERDICT is also ADOPT.** 9 of 10 airports improve after the K-drop; the airport-day CI is −0.91 [−1.18, −0.67].
- Matched rows 196.6 → 195.8; unmatched rows unchanged (L6 is model A only).
- This is larger than the residual-harness estimate (−0.8 s before any haircut). After the expected CV→LB haircut,
  the LB expectation is about −0.3 to −0.6 s.
- **Next step:** turn on `fe_retention` in the production members and re-verify on all folds.
