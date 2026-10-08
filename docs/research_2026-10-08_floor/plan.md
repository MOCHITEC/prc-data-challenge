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
