# Large-error rows: audit for legal, learnable structure (2026-10-08, OOF analysis, nothing adopted)

## Where the large errors sit
- Matched rows with |r| > 600 s hold 22.8% of SSE. Physically they are one mechanism: delayed flights that push back early and then hold.
- These rows break down by where BLOCK sits relative to the NM clocks:
  - BLOCK ≥ 15 min before AOBT_3: 9.9% of SSE.
  - BLOCK near a plan clock: 6.7%. Outside LIRF this is mostly coincidence.
  - BLOCK 2–15 min from AOBT_3: 4.1%.
- The 1–2 h label band is the same mechanism. In July it is concentrated in LIRF and EGLL storm days.
- Unmatched rows outside LIRF:
  - 2 day-shift rows hold 21% of SSE. They cannot be identified: the only flag has precision 0.07%.
  - Other rows near SCHED are flagged with AUC ≈ 0.95, but their residual mean changes sign by fold.

## Identifier vs gain
- The classes are easy to identify: AUC 0.94–0.96, stable across folds. ARVT_3 features add about 0.001 AUC.
- The model already prices them: the mean residual within each class is close to 0.
- A stage-2 corrector using q gives −0.011 s, CI [−0.10, +0.08], which is a no.
- Mixing toward MVT − EOBT1 gives about −0.1 s at weight 0.1, but that weight was chosen post hoc and weight 0.2 is worse.

## Verdict
- No label-free handling worth ≥ 0.1 s.
- What remains is the hold length after an early push. Only ADS-B pushback and stand-exit evidence observes it, and that work belongs to the integration session.
