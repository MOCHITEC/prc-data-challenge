# Feb/Jun 2025 check of our levers for the 4-month final: pre-registration (2026-10-13, before the runs)

## Why
- The final set scores Jan, Feb, Jun and Jul 2026.
- All our verdicts used the jan, jul and forward (Nov+Dec) folds, so Feb and Jun were never tested.

## Runs
- Leave-one-month-out folds `feb` and `jun` only (`folds: [feb, jun]`, `PRC_FEBJUN_FOLDS=1`). Each fold trains on the other 11 months of 2025.
- Configs: the production configs with `_fj` appended, in the scratch dir. They are built from these, unchanged except `folds`:
  - opt, ucmix, nmb, cmix, cmix_ltfm, l6, pa_opt_l6.

## Checks (fixed now)

| Lever | Comparison | Rows | Pass if |
|---|---|---|---|
| ucmix | ucmix vs opt | LIRF unmatched | pooled feb+jun change ≤ 0 |
| NM-blind | nmb vs opt | non-LIRF unmatched | pooled feb+jun change ≤ 0 |
| cmix | cmix vs opt | LIRF matched | pooled feb+jun change ≤ 0 |
| L6 | l6 vs opt | whole frame | pooled feb+jun change ≤ 0 |
| **cmix LTFM (open lever)** | cmix_ltfm vs cmix | LTFM matched | pooled ≤ 0 and neither month worse by more than +1 s |
| **blend inner split (open lever)** | 0.2·l6 + 0.8·pao_l6 vs 0.5/0.5, non-LIRF matched rows | — | pooled ≤ 0 and both months ≤ 0 |

- The four adopted levers (ucmix, NM-blind, cmix, L6) are already in the final. A fail is reported to the RPC thread as a risk. It does not by itself remove the lever.
