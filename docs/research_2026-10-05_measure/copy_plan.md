# LIRF matched copy mixture (copy_mix_airports) — pre-registration (2026-10-05)

## Background
`copy.md` item 1 (the two-stage mixture of outgoing-boat), measured on a simpler base:
- dMSE over all matched rows −1,017 s² [−1,453, −605].
- On the LIRF rows alone, both months improve significantly (Jan −12.3 s, Jul −19.4 s).
- 78% of the gain comes from rows with y ≤ 3600 s.

## Change
`copy_mix_airports` (train.py `_copy_mixture`, default empty, existing ids unchanged). On the matched lane, at the listed airports:
- **Copy classifier**: a LightGBM binary model on model A's columns gives p = P(|y − (MVT − SCHED)| ≤ 60 s). It is trained on the training side of the fold only.
- **Non-copy model A**: model A is refit on that airport's non-copy training rows (ŷ_nc).
- **Prediction**: p·(MVT − SCHED) + (1 − p)·ŷ_nc. p = 0 when SCHED lies outside the LOBT window. The LOBT bound is then applied as usual.

## Arms (all folds)
| arm | config |
|---|---|
| base | main_flseq_osm_nolirf_l2_opt (a2o; the same run serves as the L6 base) |
| cmix | main_flseq_osm_nolirf_l2_opt_cmix (copy_mix_airports: [LIRF]) |

## Verdict (fixed before the runs)
- **Main rule**: `prc.compare base cmix` returns ADOPT (full and K=200 CIs exclude 0 on the improving side).
- The effect lives at one airport, so the per-airport tail report is read too, along with the STRICT VERDICT line. Under the strict rule "most airports improve" cannot hold by design. The strict line is reported for information and is not part of this verdict.
- **Extra condition**: LIRF must improve in **both** the jan and jul folds. If it does not, REJECT even if the main rule ADOPTs.

## Result (2026-10-05, `cmix_verdict.txt`)
- **ADOPT.** main_flseq_osm_nolirf_l2_opt-7244b3004f → main_flseq_osm_nolirf_l2_opt_cmix-d9b7e7a288.
- **Main rule:** full rows −0.49 s [−0.83, −0.15]; K=200 −0.70 s [−1.03, −0.36]. Both CIs exclude 0 on the improving side.
- **Extra condition, met:** LIRF matched rows improve in both test-season folds.
  - jan 280.0 → 275.1 (−4.93)
  - jul 434.4 → 430.4 (−4.07)
  - forward −7.10
  - pooled −5.39 s [−8.99, −1.76], with the airport-day bootstrap
- Every fold improves overall: forward −0.69, jan −0.29, jul −0.47.
- STRICT reports 1/10 airports. That is expected for a change at one airport; the plan says it is informational only.
- This is smaller than the simple-base measurement (−2.2 s), as expected: the production model is stronger at LIRF.

## Extension to LEBL and LTFM — pre-registered before the runs (2026-10-06)
**Why:** `research_2026-10-05_aobt` F2 found a BLOCK = SCHED excess of 4.4–4.6% at LEBL in Jul/Aug. merry-mushroom's LIRF+LTFM
schedule-source mixture gave −0.33 / −0.38 s. The pooled 9-airport hedge in `copy.md` had no non-copy refit and is
a different test.

**Arms (all folds), each compared against cmix (main_flseq_osm_nolirf_l2_opt_cmix-d9b7e7a288):**
- `main_flseq_osm_nolirf_l2_opt_cmix_lebl` — copy_mix_airports: [LIRF, LEBL]
- `main_flseq_osm_nolirf_l2_opt_cmix_ltfm` — copy_mix_airports: [LIRF, LTFM]

**Verdict, per arm:** ADOPT only if both conditions hold:
1. `prc.compare cmix arm` returns ADOPT.
2. On the added airport's matched rows, the pooled airport-day CI excludes 0 on the improving side, jul improves,
   and jan is not worse by more than 1 s.
