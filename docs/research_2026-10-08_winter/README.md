# Winter regime check, 2026 vs 2025 (2026-10-08, OOF analysis, nothing adopted)

## January 2026 contains days outside the 2025 range
- EHAM had a snow and de-icing event on Jan 2–9. On Jan 3–7 the daily mean proxy was 1,940–2,600 s, about 2× the 2025 maximum.
- LFPG (Jan 5, 7), EDDF (9 days) and LSZH (Jan 8, 10) also went beyond their 2025 maximum.
- 2,805 rows sit on such days; 528 of them are unmatched.

## Matched lane: no correction needed
- Bias is flat across de-icing, frozen weather, low visibility and extreme days.
- At high proxy the 2026 predictions keep scaling with the proxy. For EHAM, pred/proxy is 0.84–0.89 against a 2025 y/proxy of 0.81–0.85.
- The leave-one-fold-out per-regime shifts were all unstable, each within about ±0.04 s.
- One weak signal: LFPG with proxy 4,000–6,000 s, 51 rows, worth at most about 0.15 s. It rests on only 28 rows from 2025.

## Unmatched lane at EHAM and LFPG in January 2026: not identifiable
- The 2025 analogues give y/level of 0.72–0.80 at EHAM. The 2026 predictions sit at pred/level 0.71, so there is no clear under-prediction.
- The floor rule was already rejected on 2025 data.
- A floor of k = 0.8 would swing about ±1.2 s in either direction, with an unknown sign. Its expected gain is about 0.

## Verdict
No legal, label-free correction worth ≥0.3 s exists.
