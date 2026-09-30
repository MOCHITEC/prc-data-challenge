# Error anatomy (generated)

Source: `experiments/oof_v7_mixture_newfeats-1112223a96.parquet` joined to `data/work/feat_train.parquet`. Labelled OOF rows: 672,428. Model RMSE **305.70**. Conclusions live in `docs/06_error_anatomy.md`.

## BLOCK_TIME as a copy of a known clock

| clock | non_null_pct | |d|<=0s % | |d|<=1s % | |d|<=5s % | |d|<=60s % | |d|<=300s % |
|---|---|---|---|---|---|---|
| aobt3 (AOBT_3_flt) | 98.9 | 0.6458 | 1.9320 | 7.1114 | 20.8 | 73.8 |
| sched (SCHED_TIME_UTC_mvt) | 100.0 | 0.3896 | 1.1639 | 4.2574 | 9.5746 | 36.6 |
| eobt1 (EOBT_1_flt) | 98.9 | 0.4706 | 1.4065 | 5.1667 | 12.5 | 48.4 |
| lobt (LOBT_flt) | 98.9 | 0.4344 | 1.2959 | 4.7652 | 11.1 | 43.6 |
| iobt (IOBT_flt) | 98.9 | 0.4355 | 1.3009 | 4.7845 | 11.2 | 43.8 |

## Clock-source mix (all labelled rows)

| src | n | pct | med_y | sd_y | max_y |
|---|---|---|---|---|---|
| genuine | 1,402,350 | 67.3 | 933.0 | 439.3 | 58,206.0 |
| aobt3 | 433,576 | 20.8 | 864.0 | 292.2 | 5,876.0 |
| sched | 157,772 | 7.5668 | 962.0 | 1,080.2 | 131,167.0 |
| eobt1 | 80,008 | 3.8372 | 960.0 | 403.9 | 7,684.0 |
| lobt | 11,125 | 0.5336 | 956.0 | 409.6 | 6,358.0 |
| iobt | 201 | 0.0096 | 1,085.0 | 421.4 | 3,604.0 |
| rollover_24h | 15 | 0.0007 | 87,341.0 | 911.5 | 88,392.0 |

## Squared error by clock source (tolerance +-60s)

| src | n | pct_rows | pct_sse | model_rmse | oracle_rmse |
|---|---|---|---|---|---|
| genuine | 453,993 | 67.5 | 58.3 | 284.0 | - |
| rollover_24h | 10 | 0.0015 | 25.1 | 39,723.4 | - |
| sched | 50,490 | 7.5086 | 11.6 | 379.7 | 36.2 |
| aobt3 | 138,132 | 20.5 | 2.9762 | 116.4 | 38.8 |
| eobt1 | 26,392 | 3.9249 | 1.7128 | 201.9 | 39.3 |
| lobt | 3,358 | 0.4994 | 0.3357 | 250.7 | 37.7 |
| iobt | 53 | 0.0079 | 0.0073 | 295.1 | 42.5 |

Oracle routing (known candidate for a matched source, current model for `genuine`/`rollover_24h`): **RMSE 280.00** vs 305.70 now.

## Concentration

| top_k_rows | pct_of_sse | pct_of_rows |
|---|---|---|
| 20 | 34.6 | 0.0030 |
| 50 | 38.6 | 0.0074 |
| 100 | 41.6 | 0.0149 |
| 200 | 44.9 | 0.0297 |
| 500 | 49.4 | 0.0744 |
| 1,000 | 53.1 | 0.1487 |
| 5,000 | 62.6 | 0.7436 |

## By airport

| ADEP_mvt | n | pct_sse | rmse | no_nm_pct | y_gt3h_pct |
|---|---|---|---|---|---|
| LIRF | 49,924 | 34.6 | 660.1 | 1.0756 | 0.1422 |
| LFPG | 78,036 | 25.4 | 451.9 | 2.0016 | 0.0038 |
| LTFM | 91,338 | 8.3431 | 239.6 | 1.4069 | 0.0000 |
| EGLL | 79,633 | 7.9445 | 250.4 | 0.8125 | 0.0025 |
| EHAM | 79,639 | 4.5175 | 188.8 | 1.6989 | 0.0025 |
| EDDF | 72,276 | 4.2716 | 192.7 | 1.0889 | 0.0000 |
| LEBL | 56,153 | 4.0428 | 212.7 | 1.1576 | 0.0000 |
| LSZH | 43,009 | 4.0195 | 242.3 | 1.6787 | 0.0000 |
| LEMD | 70,732 | 3.7169 | 181.7 | 0.5853 | 0.0000 |
| EDDM | 51,688 | 3.1578 | 195.9 | 0.7758 | 0.0000 |

## By Network Manager coverage

| nm_matched | n | pct_rows | pct_sse | rmse | med_y | p99_y |
|---|---|---|---|---|---|---|
| 0 | 8,358 | 1.2430 | 49.7 | 1,933.6 | 971.0 | 10,091.0 |
| 1 | 664,070 | 98.8 | 50.3 | 218.1 | 919.0 | 2,454.0 |

## Ranking-side NM coverage

| ADEP_mvt | n | no_nm_pct |
|---|---|---|
| EHAM | 38,182 | 3.1586 |
| LSZH | 23,150 | 2.8467 |
| LFPG | 39,872 | 1.8359 |
| LTFM | 47,581 | 1.5111 |
| LEBL | 30,080 | 1.4661 |
| LIRF | 26,899 | 1.4238 |
| EDDF | 36,317 | 1.1592 |
| EDDM | 25,966 | 0.9166 |
| LEMD | 36,954 | 0.6386 |
| EGLL | 39,840 | 0.6225 |
