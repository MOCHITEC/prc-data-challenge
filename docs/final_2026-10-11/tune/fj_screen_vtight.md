# Feb+Jun screen (mode es; single seed 42 unless the arm says otherwise)

- feb: valid [202502], train 11 months, ES holdout [202512] (164028 rows), labelled matched target rows n = 142622, config main_flseq_osm_nolirf_l2_opt.yaml
- jun: valid [202506], train 11 months, ES holdout [202512] (164028 rows), labelled matched target rows n = 180640, config main_flseq_osm_nolirf_l2_opt.yaml
- pooled RMSE_m = sqrt(sum_f n_f RMSE_f^2 / sum_f n_f) over the feb+jun rows; d = arm - base; pass = pooled d <= 0 (single-seed noise reference on jan/jul: sd ~0.13 s per fold).

| arm | feb RMSE_m | d feb | jun RMSE_m | d jun | pooled RMSE_m | pooled d | w-mean d | verdict | rounds feb/jun | min/fold |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 217.493 | +0.000 | 211.528 | +0.000 | 214.180 | +0.000 | +0.000 | base | 1536/1365 | 12/10 |
| cat_tight | 216.945 | -0.548 | 210.712 | -0.817 | 213.484 | -0.696 | -0.698 | PASS | 1896/1900 | 14/13 |
| ff_025 | 218.174 | +0.681 | 210.614 | -0.914 | 213.982 | -0.198 | -0.210 | PASS | 1942/1759 | 9/9 |
| cat_vtight | 217.004 | -0.489 | 210.933 | -0.595 | 213.633 | -0.547 | -0.548 | PASS | 1987/1998 | 10/10 |
