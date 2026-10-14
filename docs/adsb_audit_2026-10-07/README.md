# ADS-B residual stage: 2025 vs 2026 consistency audit (2026-10-07)

No pipeline inconsistency of the base-pipeline kind was found.

- **Code:** ingest and feature code are identical for both years. Rebuilt days match the stored features exactly.
- **Base mismatch:** training the residual on one base and applying it on another moves MSE by ±25 at most.
- **CV:** leave-month-out ≈ leave-days-out. The apron gain holds month-out: Jan +328, Jul +444.
- **Data shift, not a pipeline bug:** adsb.lol receiver coverage changed.
  - EDDF ground-fix density fell from a median post_n of 34 to 8–12; stand detection is missing on 64–74 % of rows
    instead of 44 %.
  - EGLL July post_n went from 4 to 88.
  - Multivariate adversarial AUC 2025 vs 2026: 0.90.
- **Recovering b by fixed point does not converge:** mean |b̂ − b| 39 s. Swaps are still fine (≈ 1–3 MSE), but any
  2026 diagnostic built on b̂ or c_hat is distorted. Fix: store the pre-residual b next to every candidate.
- **Lesson:** discount CV gains of residual add-ons to about ⅓ until the board confirms them. Apron transferred
  +52 against +482 in CV.

Scripts: `analysis/local_adsbaudit_*.py`.
