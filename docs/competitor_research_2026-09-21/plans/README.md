# Planning round 2026-09-21 — ten themes

The 24 research slots each ended with an "Applicable to resilient-hedgehog"
list. Collapsing those across files gives ten themes with distinct mechanisms.
One planner per theme, plus an auditor (00) whose job is to name any actionable
candidate the ten themes do NOT cover.

Nothing here is the commander's own analysis. Every figure below is the
headline evidence a slot produced; the plan files carry the detail.

| # | theme | headline evidence | source slots |
|---|-------|-------------------|--------------|
| 00 | master candidate list + gap audit | — | all 24 |
| P1 | unmatched (missing-proxy) lane specialist | a dedicated model for the missing rows: **-124.5** seasonal, 25x the residual reparameterisation itself | 08, 21, 22, 23 |
| P2 | cross-fit the target encodings | defect in our own code, train.py:367-368; CatBoost paper proves prediction shift | 20, 21 |
| P3 | OOF fold column + validation protocol | our OOF pools three fold definitions with no label; blocks P1, P6, P7, P10 | 11, 14, 15, 21, 24 |
| P4 | stand_arrival_gap as a feature + stand_arrival_taxi | already measured at **-3.242** as a feature against -0.781 as a correction | 18, 06, 07 |
| P5 | physics-identity lower bound | **-20.3 live** elsewhere (336.46 -> 316.19) on 11,826 rows; a bound, not a fit | 24, 15, 08 |
| P6 | per-airport upper bound from 2025 support | our own OOF **-7.53**, 49 helped / 4 hurt, never shipped; -2.33 live elsewhere at zero fold price | 15 |
| P7 | model capacity on clipped residuals | **-7.11** elsewhere with zero new features; also un-blocks feature families we rejected under a shallow model | 07, 18, 12, 17 |
| P8 | weather re-measurement + coverage-parity gate | our join is broken and the cache is on disk; the gate is what -13.14 elsewhere was really about | 16, 22 |
| P9 | long-horizon rolling durations (2h-30h, arrival side) | duration families pay where count families never do; our windows stop at 60 min | 05, 17 |
| P10 | CatBoost as a rival single model | switching family bought ~50x what blending bought on PRC 2024's own ablation | 11, 20, 03, 23 |

## Standing constraints given to every planner

- Plans only. No code changes, no training runs, no submissions.
- Facts with sources; [S] for read, [X] for extrapolation; cite file:line.
- Every stage independently shippable and independently measurable, cheapest
  decisive test first, with an abort criterion.
- Each stage states how we would know it failed.
- Represent contradicting evidence rather than advocating.
