# データプロファイル
`python -m prc.profile` の出力。docs/01 §7 の未確認事項に対する実測の回答。
## 1. ファイル構成と型（U13）
| file | rows | size_mb | timestamps |
|---|---|---|---|
| ranking.parquet | 689,534 | 43.6 | MVT_TIME_UTC_mvt:timestamp[us, tz=UTC]; BLOCK_TIME_UTC_mvt:timestamp[us, tz=UTC] |
| submitting.parquet | 344,841 | 1.7000 | - |
| training_2025-01-01_2025-02-01.parquet | 307,257 | 21.2 | MVT_TIME_UTC_mvt:timestamp[us, tz=UTC]; BLOCK_TIME_UTC_mvt:timestamp[us, tz=UTC] |
| training_2025-02-01_2025-03-01.parquet | 287,411 | 19.8 | MVT_TIME_UTC_mvt:timestamp[us, tz=UTC]; BLOCK_TIME_UTC_mvt:timestamp[us, tz=UTC] |
| training_2025-03-01_2025-04-01.parquet | 328,622 | 22.6 | MVT_TIME_UTC_mvt:timestamp[us, tz=UTC]; BLOCK_TIME_UTC_mvt:timestamp[us, tz=UTC] |
| training_2025-04-01_2025-05-01.parquet | 350,288 | 23.8 | MVT_TIME_UTC_mvt:timestamp[us, tz=UTC]; BLOCK_TIME_UTC_mvt:timestamp[us, tz=UTC] |
| training_2025-05-01_2025-06-01.parquet | 370,105 | 25.2 | MVT_TIME_UTC_mvt:timestamp[us, tz=UTC]; BLOCK_TIME_UTC_mvt:timestamp[us, tz=UTC] |
| training_2025-06-01_2025-07-01.parquet | 365,986 | 25.0 | MVT_TIME_UTC_mvt:timestamp[us, tz=UTC]; BLOCK_TIME_UTC_mvt:timestamp[us, tz=UTC] |
| training_2025-07-01_2025-08-01.parquet | 381,161 | 26.0 | MVT_TIME_UTC_mvt:timestamp[us, tz=UTC]; BLOCK_TIME_UTC_mvt:timestamp[us, tz=UTC] |
| training_2025-08-01_2025-09-01.parquet | 382,047 | 26.0 | MVT_TIME_UTC_mvt:timestamp[us, tz=UTC]; BLOCK_TIME_UTC_mvt:timestamp[us, tz=UTC] |
| training_2025-09-01_2025-10-01.parquet | 367,679 | 25.1 | MVT_TIME_UTC_mvt:timestamp[us, tz=UTC]; BLOCK_TIME_UTC_mvt:timestamp[us, tz=UTC] |
| training_2025-10-01_2025-11-01.parquet | 371,163 | 25.4 | MVT_TIME_UTC_mvt:timestamp[us, tz=UTC]; BLOCK_TIME_UTC_mvt:timestamp[us, tz=UTC] |
| training_2025-11-01_2025-12-01.parquet | 324,530 | 22.2 | MVT_TIME_UTC_mvt:timestamp[us, tz=UTC]; BLOCK_TIME_UTC_mvt:timestamp[us, tz=UTC] |
| training_2025-12-01_2026-01-01.parquet | 331,548 | 22.7 | MVT_TIME_UTC_mvt:timestamp[us, tz=UTC]; BLOCK_TIME_UTC_mvt:timestamp[us, tz=UTC] |

- schema names differ in submitting.parquet: +[] -['ADEP_flt', 'ADEP_mvt', 'ADES_FILED_flt', 'ADES_flt', 'ADES_mvt', 'AIRCRAFT_OPERATOR_flt', 'AIRCRAFT_TYPE_flt', 'AIRCRAFT_TYPE_mvt', 'AOBT_3_flt', 'ARVT_1_flt', 'ARVT_3_flt', 'BLOCK_TIME_UTC_mvt', 'CALLSIGN_flt', 'EOBT_1_flt', 'FLIGHT_ID_mvt', 'FLIGHT_RULE_flt', 'FLIGHT_RULE_mvt', 'FLIGHT_TYPE_flt', 'FLIGHT_mvt', 'IOBT_flt', 'LOBT_flt', 'MARKET_SEGMENT_flt', 'MVT_TIME_UTC_mvt', 'PHASE_mvt', 'RUNWAY_mvt', 'SCHED_TIME_UTC_mvt', 'STAND_mvt', 'WK_TBL_CAT_flt']
- timestamp types: MVT_TIME_UTC_mvt=timestamp[us, tz=UTC], BLOCK_TIME_UTC_mvt=timestamp[us, tz=UTC], SCHED_TIME_UTC_mvt=timestamp[us, tz=UTC], LOBT_flt=timestamp[us, tz=UTC], IOBT_flt=timestamp[us, tz=UTC], EOBT_1_flt=timestamp[us, tz=UTC], ARVT_1_flt=timestamp[us, tz=UTC], AOBT_3_flt=timestamp[us, tz=UTC], ARVT_3_flt=timestamp[us, tz=UTC]

## 2. 行数・一意性・空港（U5, U7）
| metric | value | note |
|---|---|---|
| training rows | 4,167,797 | 公式記載 4,167,797 との差 +0 |
| unique MVT_ID (training) | 4,167,797 | 行数と一致すべき |
| training departures | 2,085,047 |  |
| ranking rows | 689,534 |  |
| ranking departures | 344,841 | 期待 344,841（API の usedPairs） |
| submitting rows | 344,841 |  |
| departure airports (training) | 10 | EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM |

## 3. 目的変数（U6）
| metric | value |
|---|---|
| rows | 2,085,047 |
| null_rate | 0.0000 |
| negative_rate | 0.0002 |
| gt_2h_rate | 0.0003 |
| p01 | 291.0 |
| p10 | 571.0 |
| median | 912.0 |
| p90 | 1,471.0 |
| p99 | 2,339.0 |
| max | 131,167 |
| sd | 546.4 |

空港別:
| ADEP_mvt | rows | median | sd | p99 | gt_2h_rate | null_rate |
|---|---|---|---|---|---|---|
| EDDF | 230,141 | 837.0 | 329.8 | 1,819.0 | 0.0000 | 0.0000 |
| EDDM | 167,334 | 774.0 | 301.8 | 1,921.0 | 0.0000 | 0.0000 |
| EGLL | 239,546 | 1,319.0 | 421.8 | 2,701.0 | 0.0002 | 0.0000 |
| EHAM | 247,951 | 742.0 | 318.5 | 1,775.0 | 0.0000 | 0.0000 |
| LEBL | 179,705 | 906.0 | 330.9 | 1,962.0 | 0.0000 | 0.0000 |
| LEMD | 212,242 | 985.0 | 312.5 | 1,932.0 | 0.0000 | 0.0000 |
| LFPG | 239,552 | 954.0 | 453.0 | 2,409.0 | 0.0002 | 0.0000 |
| LIRF | 160,704 | 1,025.0 | 1,332.0 | 4,019.0 | 0.0030 | 0.0000 |
| LSZH | 134,907 | 711.0 | 385.4 | 1,707.0 | 0.0000 | 0.0000 |
| LTFM | 272,965 | 963.0 | 427.6 | 2,587.0 | 0.0001 | 0.0000 |

定義の整合: `|(MVT − BLOCK) − TAXITIME| <= 1s` の割合 = 1.000000

## 4. Ranking で何が残っているか（U1, U11）
| column | null_rate_ranking_dep | null_rate_training_dep |
|---|---|---|
| MVT_TIME_UTC_mvt | 0.0000 | 0.0000 |
| BLOCK_TIME_UTC_mvt | 1.0000 | 0.0000 |
| TAXITIME_SEC_mvt | 1.0000 | 0.0000 |
| SCHED_TIME_UTC_mvt | 0.0000 | 0.0000 |
| RUNWAY_mvt | 0.0000 | 0.0000 |
| STAND_mvt | 0.0000 | 0.0000 |
| FLIGHT_ID_mvt | 0.0153 | 0.0108 |
| AOBT_3_flt | 0.0153 | 0.0108 |
| LOBT_flt | 0.0153 | 0.0108 |
| EOBT_1_flt | 0.0153 | 0.0108 |
| IOBT_flt | 0.0153 | 0.0108 |
| ARVT_3_flt | 0.0153 | 0.0108 |

## 5. NM の時計の情報量（U1）
| ADEP_mvt | rows | med_block_minus_aobt3 | aobt3_within_60s | sd_mvt_minus_aobt3 | med_block_minus_lobt | lobt_within_5min |
|---|---|---|---|---|---|---|
| EDDF | 228,185 | 87.0 | 0.2097 | 248.0 | 383.0 | 0.3806 |
| EDDM | 166,324 | -2.0000 | 0.2572 | 203.1 | 238.0 | 0.4955 |
| EGLL | 238,132 | 54.0 | 0.1935 | 430.5 | 120.0 | 0.4583 |
| EHAM | 243,867 | 107.0 | 0.2367 | 382.5 | 367.0 | 0.3910 |
| LEBL | 177,843 | -54.0 | 0.2635 | 298.6 | 237.0 | 0.4650 |
| LEMD | 211,295 | 2.0000 | 0.2625 | 227.7 | 299.0 | 0.4414 |
| LFPG | 235,779 | -4.0000 | 0.2194 | 355.9 | 297.0 | 0.4451 |
| LIRF | 159,216 | -118.0 | 0.1855 | 308.4 | 58.0 | 0.4875 |
| LSZH | 132,616 | -33.0 | 0.2423 | 171.5 | 432.0 | 0.3487 |
| LTFM | 269,320 | 296.0 | 0.0869 | 323.9 | 178.0 | 0.4886 |

## 6. 報告エラー体制（尾部の主因）
| ADEP_mvt | rows | block_eq_sched_rate | day_offset_rate | nm_unmatched_rate | gt_4h_rate |
|---|---|---|---|---|---|
| EDDF | 230,141 | 0.0018 | 0.0000 | 0.0085 | 0.0000 |
| EDDM | 167,334 | 0.0122 | 0.0000 | 0.0060 | 0.0000 |
| EGLL | 239,546 | 0.0124 | 0.0000 | 0.0059 | 0.0000 |
| EHAM | 247,951 | 0.0020 | 0.0000 | 0.0165 | 0.0000 |
| LEBL | 179,705 | 0.0143 | 0.0000 | 0.0103 | 0.0000 |
| LEMD | 212,242 | 0.0109 | 0.0000 | 0.0044 | 0.0000 |
| LFPG | 239,552 | 0.0106 | 0.0000 | 0.0157 | 0.0000 |
| LIRF | 160,704 | 0.0448 | 0.0000 | 0.0093 | 0.0007 |
| LSZH | 134,907 | 0.0018 | 0.0000 | 0.0170 | 0.0000 |
| LTFM | 272,965 | 0.0127 | 0.0000 | 0.0133 | 0.0000 |

## 7. カテゴリ被覆率（未知スタンド対策）
| column | levels_train | levels_rank | rank_rows_covered |
|---|---|---|---|
| STAND_mvt | 1,899 | 1,698 | 0.9992 |
| RUNWAY_mvt | 53 | 50 | 1.0000 |
| AIRCRAFT_TYPE_mvt | 269 | 200 | 0.9988 |
| AIRCRAFT_OPERATOR_flt | 676 | 580 | 0.9841 |
| ADES_mvt | 1,568 | 1,106 | 0.9995 |
| FLIGHT_mvt | 45,717 | 20,330 | 0.7889 |

域内便（同一 FLIGHT_ID の到着行が Ranking 内にある出発便）の割合: 0.1477（U1b）

## 8. 素朴な予測器のベンチマーク（2025 in-sample）
| predictor | rmse |
|---|---|
| global mean | 546.4 |
| global median | 552.1 |
| airport mean | 514.8 |
| airport x runway mean | 504.5 |
| airport x runway x stand mean | 469.1 |
| MVT - AOBT_3 (NM rows, 98.9% of rows) | 384.9 |

## 9. 便名・コールサインの匿名化（U8）
```
shape: (10, 3)
┌────────────┬──────────────┬─────────────────────────────────┐
│ FLIGHT_mvt ┆ CALLSIGN_flt ┆ AIRCRAFT_OPERATOR_flt           │
│ ---        ┆ ---          ┆ ---                             │
│ str        ┆ str          ┆ str                             │
╞════════════╪══════════════╪═════════════════════════════════╡
│ LH982      ┆ DLH8UL       ┆ 11dca3faf5500fdddbb17189b5d18b… │
│ LH796      ┆ DLH796       ┆ 11dca3faf5500fdddbb17189b5d18b… │
│ LH958      ┆ DLH2HU       ┆ 11dca3faf5500fdddbb17189b5d18b… │
│ LH778      ┆ DLH778       ┆ 11dca3faf5500fdddbb17189b5d18b… │
│ LH948      ┆ DLH948       ┆ 11dca3faf5500fdddbb17189b5d18b… │
│ EN8756     ┆ DLA7TA       ┆ af8b67cf63bb6f7021b7845e4f8207… │
│ LH832      ┆ DLH832       ┆ 11dca3faf5500fdddbb17189b5d18b… │
│ DE2290     ┆ CFG2290      ┆ 2cf658a7fffbe4bd9af09b8b0e3ded… │
│ LH1378     ┆ DLH1RP       ┆ 11dca3faf5500fdddbb17189b5d18b… │
│ LH506      ┆ DLH506       ┆ 11dca3faf5500fdddbb17189b5d18b… │
└────────────┴──────────────┴─────────────────────────────────┘
```
