# Pre-registration: airline + apron categoricals in the learned ADS-B residual (rescat)

Written 2026-10-06 BEFORE any model in this study was fit. Script: `analysis/local_adsb_rescat.py`.

## Question
Does adding (1) an airline code and (2) a stand apron code as categorical features to the learned ADS-B residual
(production: 7 months Jan, Jul, Aug-Dec 2025, V2 params) reduce the error? Teammate prior: -0.2 to -0.8 s on ADS-B rows.
Note: the base model already uses STAND_mvt and AIRCRAFT_OPERATOR_flt as categoricals, so a small gain is plausible.

## Feature definitions (fixed before fitting)
airline (per departure, string):
1. CALLSIGN_flt if it matches `^[A-Z]{3}[0-9]` -> its first 3 letters (ICAO designator);
2. else FLIGHT_mvt: `^[A-Z]{3}[0-9]` -> first 3 letters (ICAO, e.g. DLH62X -> DLH);
   `^[A-Z]{3}[A-Z]{1,2}[0-9]` (garbled ICAO+IATA, e.g. ITYTY672, BAWW561D, WMTMT4MV) -> first 3 letters;
   `^[A-Z0-9]{2}[0-9]{1,4}[A-Z]?$` with at least one letter in the first 2 chars (IATA, e.g. LH891, 4Y068, D83539) ->
   IATA 2-char mapped to ICAO by a table learned from 2025+2026 rows that have both an IATA FLIGHT_mvt and an
   ICAO callsign (majority ICAO, >= 20 rows and majority share >= 0.6); unmapped -> "I:" + IATA code;
3. else "none" (registrations, null, private formats).
AIRCRAFT_OPERATOR_flt (hashed) is NOT used: it is null exactly where CALLSIGN_flt is null and is ~1:1 with the callsign
prefix (docs/analysis/operator_aircraft_route.md), so it adds nothing to rule 1.

apron (per departure): STAND_mvt upper-cased, stripped. Starts with a letter -> leading letter run truncated to 2
characters (V126 -> V, F223 -> F); starts with a digit -> first digit + "d" + length of the leading digit run
(137 -> 1d3, 10 -> 1d2); null -> "null". Key = ADEP + "|" + code (aprons are per airport).

Vocabulary: sorted union of the codes seen in 2025 (7 training months) and 2026 (ranking), fixed order; a code with
< 50 rows among the 2025 7-month ADS-B rows (has = ADS-B features present) is mapped to "other" (apron: ADEP|other).
Integer codes = position in the sorted kept vocabulary (identical for 2025 and 2026). Counts are computed once on all
7 months (not per fold; a count threshold, label-free). LightGBM categorical defaults otherwise.

## Arms (identical folds, rows, params, clip, early stopping, LTFM c = 0)
- A0 base: production features (`design(D,"b")`), V2 params (num_leaves 255, min_data_in_leaf 300, ff 0.7), num_threads 3
- A1 base + cat_airline
- A2 base + cat_airline + cat_apron
All trained on the 7 months' training days (leave-days-out K=5 per month, the `local_adsb_learn_months7.py` folds,
inner early-stopping split and seeds), predicted on the same test days. A0 is re-run (not reused) for exact pairing.

## Metric
Per row d_arm = (y-b-c_A0)^2 - (y-b-c_arm)^2 over ALL labelled rows (non-ADS-B rows contribute 0); pooled mean (s^2),
day-block bootstrap 95% CI (adsb_learn_cv.boot, B = 2000). Also reported: Jan, Jul, Aug-Oct, Nov-Dec, per airport,
RMSE change on ADS-B rows, gain-importance of the new columns.
Concentration: share = 1 - (sum of d over all rows except the 50 rows with the largest |d|) / (sum of d).

## Decision rule (per arm vs A0)
Pass if ALL of: (i) pooled CI95 low > 0 on all rows; (ii) Jan point gain > 0 AND Jul point gain > 0;
(iii) concentration share <= 0.5.
Adopt: if only one arm passes, that arm; if both pass, A2 only if its pooled gain exceeds A1's AND the A2-A1 paired
CI low > 0, else A1. If neither passes: reject (no 2026 delta).
If adopted: fit the adopted arm on all 7 months (rounds = median CV best iteration of that arm) and produce the
2026 delta vs the production residual (7:v2, refit with the same code path, rounds = its median) on ground-fix rows,
LTFM and production exclusions as in `analysis/local_adsb_apply_swap.py` -> `rank_delta.parquet`.

## Drift (reported, not part of the decision)
Share of 2026 ADS-B rows (feat_2026) whose airline / apron code is unseen in 2025 or rare (< 50) -> "other";
adversarial AUC (LightGBM, 5-fold CV) of 2025 7-month ADS-B rows vs 2026 ADS-B rows on [airline, apron] codes, each
alone, and ADEP alone as reference; also Jan+Jul 2025 vs 2026.
