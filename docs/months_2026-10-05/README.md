# Finding 2: train the learned ADS-B residual on Nov+Dec 2025 as well (2026-10-05)

## Why

The learned residual is trained on Jan+Jul 2025 only. ADS-B ground coverage of matched departures by month:

| airport | Jan 2025 | Jul 2025 | Nov–Dec 2025 | Jan 2026 | Jul 2026 |
|---|---|---|---|---|---|
| LEMD | 0 | 0.49 | 0.98 | 0.97 | 0.84 |
| LIRF | 0 | 0.96 | 0.87–0.89 | 0.88 | 0 |
| LFPG | 0 | 0.33 | 0.61–0.73 | 0.57 | 0.57 |
| EGLL | 0.85 | 0.38 | 0.81–0.87 | 0.94 | 0.98 |
| EDDM | 0.98 | 0.38 | 0.96–0.99 | 0.97 | 0.97 |

Nov–Dec 2025 looks like Jan 2026, and neither looks like Jan 2025.

## Test (`analysis/local_adsb_learn_months.py`)

Inputs:

* Jan+Jul 2025 rows use the v23-equivalent cb-hybrid OOF (jan / jul folds).
* Nov+Dec 2025 rows use the `forward`-fold OOF (trained on Jan–Oct). The matched lane is the anchored LGB only,
  `main_flseq_osm_nolirf_fw`, because plainy and cb were skipped for time. The unmatched lane is `main_flseq_fw`.
* ADS-B for Nov+Dec was ingested for 60 of 61 days; 2025-12-31 has no release.

Design: leave-days-out with K = 5 per month. Both arms share the same days, base, features and parameters:

* arm A is fit on Jan+Jul days only;
* arm B is fit on all four months.

| evaluated on | RMSE base | A | B | B − A (MSE) | day-bootstrap CI95 |
|---|---|---|---|---|---|
| Nov+Dec | 217.01 | 194.61 | 190.95 | **+1,410** | [+1,201, +1,655] |
| Jan+Jul | 314.76 | 300.96 | 300.77 | +113 | [−221, +402] |

B − A on Nov+Dec, by airport: LEMD +2,356, EGLL +1,857, EDDM +1,335, EDDF +1,242, LFPG +1,124, LIRF +5,182 (A
itself was −3,833 there), LEBL +903, LSZH +773, EHAM +623.

Air-only rows (ADS-B seen, no ground fix): A beat B in Jan (+99 vs +61) and in Jul (+67 vs +24). So the swap is
applied to ground-fix rows only (`local_months_ground_only.py`).

The full output is in `learn_months_output.txt`.

## Candidate v36 (not submitted)

v36 = v35 candidate, then:

1. `local_adsb_apply_months.py`: final A and B models (386 / 586 rounds); b is recovered from the file by a fixed
   point; the new value is file + c_B(b) − c_A(b).
2. `local_months_ground_only.py`: keep the swap on ground-fix rows only.
3. `local_lobt_clip.py`.

The swap changes 240,843 rows, with a squared change of 799 MSE. Against v34, v36 changes 258,469 rows with a
squared change of 1,315 MSE. The expected board gain is about +900 MSE for finding 2 plus +450–650 for the v35
overlays.

Caveat: the production residual is the other session's model fitted on its v33 base. The A model here is a local
reproduction on the cb base, so the swap is applied as a delta (B − A), not by replacing their correction.

## Rebuilt on the other session's v35 (LB 245.935: MLP 4-member blend + its refit learned residual + the v34 overlay)

The same steps were applied in order: positional-join learned residual → LIRF copy blend → months swap (ground
rows only) → LOBT clip.

| candidate | contents | rows changed vs v35 | squared change (MSE) | expected gain |
|---|---|---|---|---|
| C1 `outputs/c1_v35theirs_overlays.parquet` | posjoin learned + LIRF copy + LOBT clip | ≈ 27,200 | ≈ 567 | +450–650 MSE |
| C2 `outputs/c2_v35theirs_overlays_months.parquet` | C1 + finding 2 | 258,567 | 1,321 | +1,300–1,500 MSE |

Both pass `validate_submission`. Neither has been submitted.

## C3 (not submitted): v37 + V2 residual settings

`local_adsb_apply_swap.py 4:cur 4:v2` on v37 (= C2), then the LOBT clip. It changes 242,813 ground-fix rows, with
a squared change of 508 MSE. The settings evidence is `deeper_checks.md` #14: +376 MSE [+193, +582] on the 4-month
leave-days-out CV. The file is `outputs/c3_v37_v2params.parquet`.

## Finding 2b: add Aug–Oct 2025 (`local_adsb_learn_months7.py`), current settings

Inputs:

* ADS-B: 92 of 92 days ingested.
* Base for Aug–Oct: a local `aso` fold that holds out Aug+Sep+Oct and trains on the other nine months
  (`local_patches/cv_aso_fold.diff`), 1 seed. Fold RMSE is 241.48 for `main_flseq_osm_nolirf_aso` and 239.94 for
  `main_flseq_aso`.

Design: leave-days-out with K = 5 per month. Arm B is fit on Jan/Jul/Nov/Dec; arm C on all seven months.

| evaluated on | C − B (MSE) | CI95 |
|---|---|---|
| Jan | **+305** | [+109, +515] |
| Jul | +410 | [−193, +1,027] |
| Jan+Jul | **+363** | [+32, +715] |
| Nov+Dec | +36 | [−177, +237] |
| Aug–Oct | +1,037 | [+692, +1,394] |
| all | **+583** | [+394, +785] |

LIRF is unstable: Nov+Dec −3,081 [−5,773, −647] and Jul +2,603 [−4,931, +10,165]. LIRF is therefore excluded from
the swap. Its Jan 2026 ground coverage matches Nov–Dec, and in Jul 2026 it has no ground rows.

The full output is in `learn_months7_output.txt`. A V2-settings rerun (`PARAMS=v2`) is under way.
