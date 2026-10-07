# Pre-registration: neighbour ADS-B ground activity in the learned ADS-B residual (nbradsb)

Written 2026-10-07 BEFORE any model in this study was fit (features were built and summarised, no label touched).
Script: `analysis/local_nbr_adsb.py` (harness of `analysis/local_adsb_rescat.py`).

## Question
Does adding label-free aggregates of the ADS-B-observed taxi of neighbouring departures (same airport) to the learned
ADS-B residual (production 12:v2; 7:v2 at LFPG) reduce the error? Teammate prior -0.5 to -1.5 s.

## Features (fixed; 20 columns, NB_COLS in the script)
Pool = ADS-B departures of the same airport with a valid ADS-B taxi duration dur = t_mvt - t_move in [60, 3600] s; the
row itself always excluded. ref = 2025 median dur per airport x RUNWAY_mvt x hour_local (>= 30 rows, else airport x
hour, else airport; computed once on all of 2025, label-free). Windows on the neighbour's t_mvt vs own T = t_mvt:
b30 (T-30', T], f30 (T, T+30'], s15 [T-15', T+15'] -> n, mean dur, median dur, mean excess (dur - ref);
same-runway +-30' -> n, mean excess; nb_taxi_mvt = # pool aircraft taxiing at T (t_move_j <= T < t_mvt_j);
nb_taxi_start = same at own coalesce(t_push, t_move) (null if unknown); nb_cov60 = pool departures / all departures
(movement data) within +-60'; nb_taxi_adj = nb_taxi_mvt / max(nb_cov60, 0.05). (t_move used rather than t_push
because t_push covers 4-60% of rows; one consistent duration definition.)

## Arms (identical folds, rows, params, clip, early stopping, LTFM c = 0)
- A0: production features `design(D,"b")`, V2 params (255 leaves, min_data_in_leaf 300, ff 0.7), num_threads 2
- A4: A0 + the 20 NB columns
- (A5 = A4 + apron: optional, only if time remains; not part of the decision)
Settings: m12 = all 12 months 2025 (folds of local_adsb_learn_months12.py: K=5 day groups per month, same seeds);
m7 = Jan, Jul-Dec (folds of local_adsb_learn_months7 / rescat). Features stored float32 in both arms.

## Metric
d = (y-b-c_A0)^2 - (y-b-c_A4)^2 per row, over ALL labelled rows (non-ADS-B rows 0). Pooled mean, day-block bootstrap
CI95 (adsb_learn_cv.boot, B = 2000). Reported both on all airports and "ex LIRF" (LIRF rows set to 0 = what would be
applied; denominators unchanged). Top-50 share = 1 - sum(d without the 50 largest |d|)/sum(d).

## Decision rule (the 12-month setting decides)
Adopt A4 if, in m12, ex-LIRF: (i) pooled CI95 low > 0; (ii) Jan point gain > 0 AND Jul point gain > 0;
(iii) top-50 share <= 0.5. m7 is reported; if m7 contradicts (pooled ex-LIRF gain <= 0) this is flagged.
LFPG: the LFPG 7:v2 -> 7:v2+NB swap is included only if the LFPG point gain is > 0 in BOTH m7 and m12.
If adopted: production-matched 2026 swap delta (as rescat apply12): 12:v2 -> 12:v2+NB (rounds = 12:v2 rule for both),
LFPG per the rule above, LIRF/LTFM excluded, input v40 -> rank_delta_12m.parquet + summary.

## Drift (reported, not decisive)
Coverage: share of rows with >= 1 neighbour observation (b30 or f30; s15), 2025 vs 2026, all departures and ADS-B
rows, per airport. Single-feature adversarial AUC (LightGBM 5-fold, 400k random 2025 ADS-B rows vs all 2026 ADS-B
rows) for every NB column; any > 0.6 flagged.
