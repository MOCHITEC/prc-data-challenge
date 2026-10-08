# Live-level floor for unmatched rows outside LIRF: pre-registration (2026-10-08, before measuring)

## Source
- Fourth repo sweep (2026-10-08), joris-decombe/taxiout v10. That repo floors NM-unmatched rows outside LIRF at 0.95× the live NM taxi level of the airport's matched departures.
- A label-free check on our v40 ranking file found 290 rows that the floor would move. 138 of them are at EHAM in January 2026: our median prediction is 1,732 s against a level of 3,367 s, which looks like de-icing days.
- The cause is that model B cannot extrapolate. In 2025 there are only about 300 unmatched rows with a level above 2,000 s.

## Rule (fixed here)
- **Level:** the median of (MVT − AOBT_3) over matched departures at the same airport with MVT within about ±30 min. It is computed with a backward 60-min rolling median, read at t + 30 min. Label-free.
- **Floor:** pred = max(pred, k · level) on unmatched rows outside LIRF, where level > 1,500 s.
- **Primary k = 0.8, the same at every airport.** The value is conservative and is not tuned on 2025 labels. k = 0.95 (joris's value) is reported for information only.
- **Base:** model B with NM-blind, OOF main_flseq_osm_nolirf_l2_opt_nmb-97a0e31e57.

## Verdict on 2025 OOF (jan, jul, Nov, Dec)
- **ADOPT only if** both of these hold:
  1. the floor does not worsen the affected rows: pooled dSSE ≤ 0;
  2. no fold worsens by more than +0.05 s on the whole frame.
- 2025 has very few affected rows, so this is a **safety check, not a proof of gain**. The upside lies in the January 2026 EHAM de-icing rows, which 2025 barely contains.
- If the rule is adopted, it goes to the integration session as a post-processing step, with the diagnostic to be re-run on the v47 file first.

## Result (2026-10-08, `verdict.txt`)
**REJECT.** The rule fails the pre-registered safety check.

| | Primary, k = 0.8 | Info, k = 0.95 |
|---|---|---|
| Rows moved | 65 of 467 gated | 196 |
| Moved rows, RMSE change (airport-day CI) | **+75 s [+0.5, +173]** | +96 s [+36, +165] |
| Whole frame | +0.018 s | +0.063 s |
| Folds | jan +0.012, jul −0.006, forward +0.042 | — |
| EHAM moved rows, RMSE | 725 → 1,051 | — |

- When the level is high, 2025 unmatched rows at EHAM, LFPG and LSZH often sit well below it. They are not "matched-like" departures, so a floor pushes them the wrong way.
- The suspicious January 2026 EHAM rows remain an open question, but 2025 gives no support for floor-correcting them.

## Addendum (2026-10-08): restricted floor, isolation submission
- **Why:** the audit (`docs/research_2026-10-08_audit`) found that the 2025 safety test above measured a different population. It covered level 1.5–2.5 ks at EGLL and LFPG. The 2026 rows in question sit at level > 3 ks on the EHAM and LFPG snow days, and 2025 has only 16 rows like them, so there is no offline test.
- **Rule (fixed before submission):**
  - Applies to unmatched rows outside LIRF with live level > 3,000 s.
  - pred = max(pred, 0.8 · level).
  - The base is v47.
  - The level comes from `level.py` on the 2026 rank frame.
  - The usual submission gate and the LOBT clip apply afterwards.
- **Size on the v40 file:** 130 rows gated, 93 moved (EHAM 78, mean +1,273 s; LFPG 15, mean +481 s). The RMS shift over the 344,841 LB rows is 23.3 s, which bounds the LB change to about ±1.2 s.
- **Decision rule:** keep the floor if the LB score is lower than v47 (234.4831); otherwise drop it.
