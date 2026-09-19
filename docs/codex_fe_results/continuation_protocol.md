# Continuation — frozen before new refit outcomes, 2026-09-18

1. Complete original selected and all refits, using saved noise arm, original 742/90 rounds,
   January–August training, November–December evaluation and original 98.75% intervals.
2. Add gap-only refit at the same settings. This is exploratory: chosen after confirmation.
   Report 95% day and seven-day intervals without calling it independent confirmation.
3. Implement stand_arrival_gap in the actual feature pipeline as opt-in, verify equality
   with the saved candidate and departure-label masking; keep main.yaml unchanged.
4. Diagnose scalar failures versus airport-only corrections by regime and month, including
   null-feature collapse and SSE contributions. These are post-hoc explanations.
5. Assess training-size transfer using January–October versus January–August, identical
   November–December evaluation, fixed 742/90 rounds: baseline and gap-only. This is a
   sensitivity analysis, not new held-out confirmation. If useful, fit full-year gap features
   and baseline on ranking for operational parity/disagreement, never label-free RMSE claims.

No available labelled 2026 test can establish twelve-month → next-year performance. Never
substitute in-sample errors for this missing measurement. No submission generation/upload.
Serial training; all jobs two threads. Each completed arm writes predictions and metadata.
Original report remains intact; new evidence appended. Commit each stage if git permits.

## Adaptive extension after selected-bundle refit (07:48 UTC)
The selected five-feature bundle measured -3.663 s versus baseline. Also make those five
features available through the production pipeline and run a ten-month selected-bundle
refit against the same ten-month baseline. This added training-size comparison is explicitly
exploratory, not part of the original confirmation multiplicity family. Preserve gap-only
ablation to distinguish the scalar survivor from the selected bundle.
