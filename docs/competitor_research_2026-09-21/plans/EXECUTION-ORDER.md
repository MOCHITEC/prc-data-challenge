# Execution order — derived from the eleven plans, not from the commander

92 merged candidates, 66 actionable (plans/00-master-candidate-list.md).
40 sit in the ten themes; 26 have no planner.

Every ordering statement below is one a planner made about its own theme,
with the measurement that supports it. Where two planners disagree, both
are shown. Nothing here is the commander's own judgement.

## What changed under our own gate

Four themes were reshaped or killed by measurements taken during planning.
This is the important result of the round: the candidates with the strongest
EXTERNAL evidence moved most when put through `prc.compare`.

| candidate | external evidence | our own measurement | outcome |
|---|---|---|---|
| physics lower bound (P5 rule A) | **-20.3 s live** for sqcm9896 | +0.0985 K=0, +0.1465 K=200, P(better)=0.000 at all four levels | **REJECTED** |
| LOBT window clip (P5 rule B) | javidmardanov, zero violations | -0.0993 / -0.0941, **P(better)=1.000 at all four levels** | **ADOPT** — first in this repo's history to do that |
| per-airport cap on unmatched (P6) | our own OOF -7.53 | v10 has 17 rows over 3,600 and 0 over 7,200; v7 had 310 and 100 | **superseded** — v10's reparameterisation already collected it |
| long-horizon durations (P9) | NASA 4th place top-5 importances | corr rises 0.0887 -> 0.1569, but +0.0038 after calendar demeaning; 0.050 s incremental | **under 10% to clear the gate** |
| turnaround gap (P4) | our own -3.2424 | measured against an 8-month baseline; production's 10-month baseline is already -2.6888 better on the same rows | **unmeasured against the real baseline** |

## Blocking prerequisites

**A. Our best submission is not reproducible.** `configs/main.yaml` does not
carry `unmatched_sched_reparam`; the keys live only in `reparam_lirf.yaml`
and `submit_v10_reparam.yaml`. The live 284.3584 is a hand-assembled probe —
v4's matched predictions plus a reparam model B — and the reparameterisation
has only ever been cross-validated on `folds: [forward]`. There is no
three-fold baseline OOF from the model we ship. (P1 stage 0, 4-6 h.)

**B. The OOF file cannot carry a fitted correction.** Folds have different
`train_months`, `coalesce` resolves overlaps in list order, the schema has no
fold label, and coverage differs between experiments (672,428 against
328,009). P3 routes around it with a sidecar that leaves the wide file
byte-identical, so no archived record is invalidated. Unblocks P1, P6, P7,
P10. (P3 S1.)

**C. The decision rule will reject real wins in two of these themes.** P1:
v10, a 20.67 s live win, passed at K=200 only at -0.38 [-0.63,-0.14], so a
change worth half of v10 lands at about -0.19 with a CI straddling zero. P7:
the repo's own worked precedent is seeds 1->3 at -2.58 robust, REJECTED on a
full-row CI of [-3.75,+0.55]. Both planners require this choice to be made
BEFORE their first run, not after a result — `features_log.md:154` is the
repo's own rule against the alternative.

## Ordering

Ship-ready, no retrain, no fold structure needed:

1. **P5 rule B** — clip to `m_lobt ± 3606`. 47 rows move on the shipped file,
   344,794 byte-identical. Guaranteed at least +0.0896 s, expected +0.3 to
   +0.6 s by three independent estimates, absolute worst case +0.3683 s
   against v11's measured +9.62. Transform the parquet directly.
2. **P6 as revised** — `min(pred, 2025 max)` at EDDF/EDDM/LEBL/LEMD. 12 rows
   move. Pre-registered -0.20 s, worst case +0.030 s. EHAM is vetoed by the
   2026 arrival mirror; LEMD/LEBL cleared by it.
   P6's ordering argument versus P5 was written against rule A and needs
   redoing against rule B, whose gate is a different column.

Instrument, before anything fitted:

3. **P1 stage 0** (prerequisite A) and **P3 S1** (prerequisite B) together.
4. **P8-D** — the coverage-parity gate, 2-3 h, reusable by every future
   external feature. It already found five model-visible constant columns
   (`precip_mm`, `precip_mm_prev1h`, `temp_x_precip`, `temp_x_precip_prev1h`,
   `runway_null`), each of which is a +/-0.5 s lottery draw on a 0.25 s
   instrument.

Measurement, cheapest decisive test first:

5. **P4 stage A** — the gap against the 10-month baseline. One run. This is
   the only thing that tells us whether the -3.2424 survives.
6. **P2 fix 1** — the early-stopping holdout stops reading its own labels. No
   serve-side change. P2 recommends NOT applying fix 2, because every sourced
   measurement of that version is a loss.
7. **P10 stage 2** — the `-E` cells only, on `forward`, about 1 h. Invariant
   to P2 by construction, so it cannot be invalidated later.
8. **P1 stages 1-5** — the unmatched lane. Plan against -7 s, not -25:
   merry-quicksand priced the same pool from inside at about -7 for
   recovering 30% of the learnable rows.

Expensive, and contested:

9. **P7** — the -7.11 attribution was corrected a third time; the defensible
   band is -1 to -5 s. 8 h of compute on 4 CPUs with no GPU. P7's own author
   writes that if it competes with P4 for compute, P7 loses.
10. **P9** — run the read-only audit unconditionally (it closes slot 06's
    unaudited item and it already found the coverage-column defect). Screen
    only if a slot is spare. P9 and P7 disagree on sequencing and both
    arguments are recorded.

## The 26 candidates with no planner

Three the auditor would take first:

- **Arrival mirror as a labelled drift gate.** The 344,693 arrival rows in
  the 2026 scoring file carry real taxi-in labels, so 2025-to-2026 transfer
  is measurable WITH LABELS before any upload. We have never measured outcome
  drift; our adversarial AUC measures covariate shift and our own code says it
  does not explain the CV/LB divergence. P6 used this and it decided the
  theme.
- **Backward feature elimination / LOFO.** `features_log.md` is pure forward
  accumulation — ten candidates added one at a time, ten rejections, nothing
  ever removed and measured.
- **Spend the submission budget.** 11 used of about 100 remaining; scoring is
  best-of-all so a regression is free; feedback returns in seconds on a fixed
  344,841 rows.

## Open items that need a human answer

- The submission limit: slot 01 reads 5/day, slot 09 reads 3/day, our code
  enforces 5.
- `AOBT_3_flt` on ranking rows: the organiser stated on 2026-09-18 that no
  formal rule bans it; merry-quicksand asked in writing and got no ruling.
  Our entire matched lane rests on it.
- Prize eligibility requires a documented GPLv3 repo. 14 of 33 scored PRC
  2025 teams were struck from the final ranking for not having one,
  including the would-be 3rd place.
