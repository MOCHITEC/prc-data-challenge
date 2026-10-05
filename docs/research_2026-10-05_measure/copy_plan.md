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
