# Result — item 13 `tail-postprocessing-cap-shrink` (2026-09-16)

Script: `docs/investigation_2026-09-16/scripts/tail-postprocessing-cap-shrink.py` (no training, `POLARS_MAX_THREADS=1`,
lazy column-selected scans, runtime ≈ 1 min). Inputs: `experiments/oof_main-48c19bc830.parquet` (2,085,047 rows, 672,428 with
non-null `oof`, joined on `MVT_ID_mvt` to `data/work/feat_train.parquet`, 8 columns), `data/work/feat_rank.parquet` (5 columns),
`outputs/main.parquet` (v7), `outputs/probe_unmatched_from_v4.parquet` (v9). Every number below is re-derived by the script;
where a source figure is quoted for comparison it is labelled (M) or (D) or (P).

Definitions (plan §0): `y = TAXITIME_SEC_mvt`; **unmatched (routing)** = `mvt_minus_aobt3` null (model-B rows; 22,470 train /
5,290 rank); **unmatched (fid)** = `FLIGHT_ID_mvt` null (22,424 / 5,283). 2 h = 7,200 s, 4 h = 14,400, 12 h = 43,200, 1 day = 86,400.
"Cap at c" means `pred := min(pred, c)` on the stated row set; "shrink λ toward t" means `pred := λ·pred + (1−λ)·t` on the stated
row set. Unless stated, post-processing is applied to unmatched (routing) rows only and matched rows are untouched. This item is
the no-training half of the item 13 / item 14 conflict; loss-function changes (Huber / quantile / log1p) are item 14 and were not fit.

## 1. Hypothesis as stated (plan §1 item 13, source E H6)

Post-processing model B's tail (cap, or shrink toward the airport-hour median) improves RMSE. Sub-claims tested:
(a) capping unmatched predictions at a fixed value improves unmatched / total RMSE; (a2) some cap quantile improves RMSE;
(b) shrinking toward the airport median improves RMSE; (c) shrinking toward the (airport, hour) median improves RMSE, and does
better than the airport median; (d) capping non-LIRF unmatched rows only improves RMSE; (e) capping matched rows improves RMSE;
(g) the submissions predict > 4 h / > 12 h more often than the training label frequency implies.

## 2. Procedure executed

1. Join OOF (672,428 scored rows; months 1, 7, 11, 12) to training labels, ADEP, both unmatched flags, `hour_utc`, `hour_local`.
   Baseline RMSE: total, unmatched (both definitions), matched, LIRF-unmatched, non-LIRF-unmatched.
2. (a) caps 3,600 / 7,200 / 14,400 / 43,200 / 86,400 on unmatched rows (routing; also fid; also all rows). Per cap: rows affected,
   rows whose |error| shrinks ("helped"), label mean / median of the affected rows, SSE delta.
3. (a2) cap = quantile q ∈ {0.90, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 1.0} of the unmatched OOF prediction distribution.
4. (b) shrink λ ∈ {0, 0.25, 0.5, 0.75, 1} toward the per-ADEP **median of unmatched training labels**, in two variants: all 12 training
   months, and only the 8 months not in this OOF (so the target is not computed on scored rows). Plus M's "oracle airport-mean"
   (mean of the OOF rows' own unmatched labels per ADEP). (b2) Reconciliation of M's figures with six alternative target definitions.
5. (c) same λ grid toward the per-(ADEP, `hour_utc`) median of unmatched training labels (all months; non-OOF months, with airport
   median as fallback where the cell is empty), and per-(ADEP, `hour_local`) as sensitivity. 219 (ADEP, hour_utc) cells, min n = 1,
   23 cells with n < 10.
6. (d) caps 3,600 … 86,400 on non-LIRF unmatched rows only; finer sweep 1,800 … 7,200 with per-month and per-airport SSE deltas;
   label distribution of non-LIRF unmatched rows predicted > 7,200. Complement: caps on LIRF unmatched rows only.
7. (e) caps on matched (routing) rows only; listing of matched rows with prediction > 3,600 and with label ≥ 4 h.
8. (f) OOF tail-frequency calibration: number of predictions > threshold vs labels ≥ threshold per group, and their overlap.
9. Ranking: per file (v7, v9) and per group (all / unmatched routing / unmatched fid / matched routing / matched fid / LIRF unmatched)
   the count of predictions > and ≥ 2 h / 4 h / 12 h; training label frequency ≥ threshold per group (all 12 months and Jan + Jul
   only) times the ranking group size = implied count; number of ranking predictions each cap would change (no labels needed).

## 3. Results

### 3.1 Baseline (OOF `main-48c19bc830`, 672,428 rows)

| group | n | RMSE |
|---|---|---|
| total | 672,428 | **334.58** |
| unmatched (routing) | 8,378 | **2,296.1** |
| unmatched (fid) | 8,358 | 2,298.6 |
| matched (routing) | 664,050 | 216.4 |
| LIRF unmatched | 537 | 7,179.3 |
| non-LIRF unmatched | 7,841 | 1,450.2 |

Unmatched-side SSE share 58.68 %. Unmatched OOF prediction quantiles: p50 1,062, p90 1,919, p99 9,609, max 51,712. Matched
prediction max 84,798 (534 matched predictions > 3,600; 36 > 7,200).

### 3.2 (a) Caps on unmatched predictions

Identical under the routing and fid definitions to 0.1 s (the 46 routing-only rows are not affected by any cap).

| variant | total | unm (routing) | unm (fid) | matched | LIRF unm | non-LIRF unm | rows affected | helped | y mean / median of affected |
|---|---|---|---|---|---|---|---|---|---|
| **baseline** | **334.58** | **2,296.1** | 2,298.6 | 216.4 | 7,179.3 | 1,450.2 | 0 | — | — |
| cap 3,600 | 452.59 | 3,567.6 | 3,571.7 | 216.4 | 13,194.8 | 1,294.5 | 363 | 160 | 9,084 / 5,510 |
| cap 7,200 | 433.64 | 3,373.4 | 3,377.3 | 216.4 | 12,296.3 | 1,343.1 | 151 | 61 | 15,944 / 9,971 |
| cap 14,400 | 404.56 | 3,069.7 | 3,073.2 | 216.4 | 10,812.0 | 1,436.1 | 52 | 17 | 33,299 / 24,805 |
| cap 43,200 | 335.42 | 2,305.8 | 2,308.4 | 216.4 | 7,227.9 | 1,450.2 | 5 | 4 | 47,417 / 43,863 |
| cap 86,400 | 334.58 | 2,296.1 | 2,298.6 | 216.4 | 7,179.3 | 1,450.2 | 0 | 0 | — |
| cap ALL rows 3,600 / 7,200 / 14,400 / 43,200 | 466.1 / 444.6 / 414.1 / 339.6 | as above | | 243.7 / 237.8 / 234.0 / 223.0 | | | | | |

M's figures (3,568 / 3,373 / 3,070 / 2,306) are reproduced. Every fixed cap ≤ 43,200 worsens unmatched and total RMSE; the
non-LIRF unmatched column improves at every cap ≤ 14,400 while the LIRF column deteriorates by a larger amount. At cap 43,200 the
5 affected rows have labels 43,863 … (4 of 5 "helped" by the |error| criterion, yet SSE rises by 3.76e8 because the one hurt row,
a rollover with y ≈ 87k predicted 51,712, loses more than the four gain).

### 3.3 (a2) Cap-quantile sweep (cap = quantile of unmatched OOF predictions)

| q | cap value | total | unm (routing) | LIRF unm | non-LIRF unm | rows affected |
|---|---|---|---|---|---|---|
| baseline (q = 1.0) | 51,712 | **334.58** | **2,296.1** | 7,179.3 | 1,450.2 | 0 |
| 0.9995 | 43,485 | 335.39 | 2,305.5 | 7,226.4 | 1,450.2 | 4 |
| 0.999 | 39,292 | 337.45 | 2,329.5 | 7,345.6 | 1,450.2 | 8 |
| 0.998 | 34,740 | 342.75 | 2,390.8 | 7,646.7 | 1,450.2 | 17 |
| 0.995 | 16,573 | 396.60 | 2,985.2 | 10,408.9 | 1,449.7 | 42 |
| 0.99 | 9,609 | 423.10 | 3,264.2 | 11,774.6 | 1,374.7 | 84 |
| 0.98 | 6,725 | 435.84 | 3,396.0 | 12,403.9 | 1,336.4 | 168 |
| 0.95 | 3,123 | 455.80 | 3,600.3 | 13,339.4 | 1,289.7 | 419 |
| 0.90 | 1,919 | 465.09 | 3,694.4 | 13,743.6 | 1,283.4 | 838 |

RMSE is monotone in q: no quantile below 1.0 improves on the baseline.

### 3.4 (b) Shrink toward the airport median

Targets = per-ADEP median of unmatched training labels (LIRF 3,922 on all months; 4,799 on the non-OOF months; other airports
539 – 1,384). M's "oracle airport-mean" = mean of the OOF rows' own unmatched labels per ADEP (LIRF 6,808).

| λ | target: airport median, all months | airport median, non-OOF months | oracle airport-mean (OOF labels) |
|---|---|---|---|
| | total / unm / LIRF unm / non-LIRF unm | total / unm / LIRF unm / non-LIRF unm | total / unm / LIRF unm / non-LIRF unm |
| **1.0 (baseline)** | **334.58 / 2,296.1 / 7,179.3 / 1,450.2** | same | same |
| 0.75 | 350.42 / 2,478.5 / 8,257.5 / 1,376.2 | 350.03 / 2,474.0 / 8,235.4 / 1,376.5 | 349.41 / 2,466.9 / 8,206.8 / 1,374.8 |
| 0.5 | 376.79 / 2,771.6 / 9,696.8 / 1,329.7 | 375.56 / 2,758.2 / 9,634.6 / 1,330.9 | 373.95 / 2,740.6 / 9,566.9 / 1,325.5 |
| 0.25 | 411.65 / 3,144.5 / 11,360.7 / 1,313.8 | 409.31 / 3,119.8 / 11,249.7 / 1,316.3 | 406.46 / 3,089.8 / 11,138.5 / 1,305.3 |
| 0.0 (constant) | 453.06 / 3,572.4 / 13,164.4 / 1,329.4 | 449.41 / 3,535.2 / 13,000.3 / 1,333.7 | 445.19 / 3,492.0 / 12,844.2 / 1,315.6 |

Every λ < 1 worsens unmatched and total RMSE; the non-LIRF unmatched RMSE improves slightly at every λ (1,450 → 1,306 – 1,376),
the LIRF unmatched RMSE deteriorates by 1,000 – 6,000 s.

(b2) Reconciliation with M's figures (constant airport-median → 3,724; 0.5 shrink → 2,825; oracle airport-mean → 3,492):

| target definition | LIRF target | λ = 0 | λ = 0.5 |
|---|---|---|---|
| median of unmatched **training** labels per ADEP (this report's (b)) | 3,922 | 3,572.4 | 2,771.6 |
| median of the **OOF rows' own** unmatched labels per ADEP | 1,729 | **3,724.2** | **2,824.7** |
| median of all training rows per ADEP (matched + unmatched) | 1,025 | 3,793.7 | 2,849.1 |
| median of OOF all-row labels per ADEP | 1,018 | 3,794.3 | 2,849.2 |
| median of OOF unmatched predictions per ADEP | 4,314 | 3,550.4 | 2,764.0 |
| mean of unmatched training labels per ADEP | 6,531 | 3,495.0 | 2,742.3 |
| global median of unmatched training labels | 951 | 3,806.1 | 2,853.8 |

M's 3,724 / 2,825 are reproduced exactly by the second row (median of the OOF rows' labels; oracle 3,492 is reproduced as well).
Across all seven target definitions the constant lies in 3,492 – 3,806 and the 0.5 shrink in 2,742 – 2,854; the ordering versus
the baseline 2,296 does not depend on the definition.

### 3.5 (c) Shrink toward the (airport, hour) median

| λ | (ADEP, hour_utc) median, all months | (ADEP, hour_utc) median, non-OOF months | (ADEP, hour_local) median, all months | (b) airport median, all months (for comparison) |
|---|---|---|---|---|
| | total / unm (routing) | total / unm | total / unm | total / unm |
| **1.0 (baseline)** | **334.58 / 2,296.1** | same | same | same |
| 0.75 | 350.10 / 2,474.8 | 350.10 / 2,474.9 | 348.09 / 2,451.9 | 350.42 / 2,478.5 |
| 0.5 | 376.40 / 2,767.4 | 376.25 / 2,765.7 | 373.61 / 2,736.9 | 376.79 / 2,771.6 |
| 0.25 | 411.42 / 3,142.1 | 410.99 / 3,137.5 | 408.91 / 3,115.7 | 411.65 / 3,144.5 |
| 0.0 | 453.14 / 3,573.2 | 452.35 / 3,565.1 | 451.70 / 3,558.5 | 453.06 / 3,572.4 |

LIRF unm / non-LIRF unm at λ = 0.5: 9,679 / 1,329 (hour_utc), 9,542 / 1,329 (hour_local) vs 9,697 / 1,330 for the airport median.
The (airport, hour) target differs from the airport target by at most 35 s of unmatched RMSE at any λ (hour_local is the better of
the two by 20 – 30 s); the fid definition gives the same numbers + 3 s. 19 OOF unmatched rows fall in (ADEP, hour_utc) cells that
are empty in the non-OOF-month variant (airport median used).

### 3.6 (d) Caps on non-LIRF unmatched rows only

| cap | total | unm (routing) | unm (fid) | non-LIRF unm | LIRF unm | affected | helped / hurt | y mean / max of affected | ΔSSE total | ΔSSE Jan / Jul / Nov / Dec |
|---|---|---|---|---|---|---|---|---|---|---|
| **baseline** | **334.58** | **2,296.1** | 2,298.6 | 1,450.2 | 7,179.3 | 0 | — | — | 0 | — |
| 1,800 | 326.57 | 2,201.5 | | 1,284.0 | 7,179.3 | 503 | 356 / 147 | | −3.563e9 | −2.82e8 / −1.04e9 / −7.70e8 / −1.47e9 |
| 2,400 | 326.57 | 2,201.5 | | 1,284.0 | 7,179.3 | 166 | 124 / 42 | | −3.563e9 | −2.98e8 / −1.02e9 / −8.07e8 / −1.44e9 |
| 3,000 | 326.78 | 2,204.0 | | 1,288.5 | 7,179.3 | 82 | 69 / 13 | | −3.471e9 | −2.81e8 / −9.76e8 / −8.13e8 / −1.40e9 |
| **3,600** | **327.05** | **2,207.3** | 2,209.7 | 1,294.5 | 7,179.3 | 53 | 49 / 4 | 1,400 / 4,747 | −3.351e9 | −2.59e8 / −9.21e8 / −8.03e8 / −1.37e9 |
| 4,200 | 327.36 | 2,211.0 | | 1,301.3 | 7,179.3 | 38 | 38 / 0 | | −3.213e9 | −2.35e8 / −8.55e8 / −7.89e8 / −1.33e9 |
| 4,800 | 327.72 | 2,215.2 | | 1,308.9 | 7,179.3 | 37 | 37 / 0 | | −3.057e9 | |
| 5,400 | 328.11 | 2,219.8 | | 1,317.2 | 7,179.3 | 34 | 34 / 0 | | −2.886e9 | |
| **7,200** | **329.33** | **2,234.3** | 2,236.8 | 1,343.1 | 7,179.3 | 24 | 24 / 0 | 996 / 1,744 | −2.345e9 | −8.26e7 / −4.76e8 / −6.74e8 / −1.11e9 |
| 14,400 | 333.87 | 2,287.8 | 2,290.3 | 1,436.1 | 7,179.3 | 9 | 9 / 0 | 897 / 1,431 | | |
| 43,200 / 86,400 | 334.58 | 2,296.1 | 2,298.6 | 1,450.2 | 7,179.3 | 0 | — | — | 0 | |

The SSE delta is negative in each of the four OOF months for every cap from 1,800 to 7,200. Cap 3,600 by airport (n affected,
helped, ΔSSE): EHAM 10 / 9 / −8.2e8, LFPG 13 / 10 / −7.9e8, EDDF 7 / 7 / −7.1e8, LEBL 8 / 8 / −5.7e8, LSZH 5 / 5 / −1.8e8, EDDM 1 / 1
/ −1.7e8, EGLL 8 / 8 / −9.9e7, LEMD 1 / 1 / −1.6e7 (all eight airports negative). The 4 hurt rows have y 3,782 – 4,747 and
predictions 3,715 – 3,979 (ΔSSE +3.0e4 … +2.5e5 each). Non-LIRF unmatched rows predicted > 7,200: 24 rows, label median 908, mean
996, max 1,744, none ≥ 2 h, prediction median 13,026 (LFPG 6, EDDF 6, EHAM 5, LEBL 4, EDDM / EGLL / LSZH 1 each).

Complement, caps on LIRF unmatched rows only: 3,600 → total 458.06 / unm 3,623.2 (310 affected, y mean 10,398); 7,200 → 437.64 /
3,414.6 (127, y mean 18,768); 14,400 → 405.14 / 3,075.9 (43, y mean 40,080); 43,200 → 335.42 / 2,305.8 (5); 86,400 → unchanged.
Table 3.2's deterioration is entirely the LIRF part.

### 3.7 (e) Caps on matched (routing) rows only

| cap | total | matched | unm (routing) | affected | helped | y mean / max of affected |
|---|---|---|---|---|---|---|
| **baseline** | **334.58** | **216.4** | 2,296.1 | 0 | — | — |
| 3,600 | 352.64 | 243.7 | 2,296.1 | 534 | 181 | 4,820 / 87,002 |
| 7,200 | 348.61 | 237.8 | 2,296.1 | 36 | 15 | 10,336 / 87,002 |
| 14,400 | 346.09 | 234.0 | 2,296.1 | 1 | 0 | 87,002 |
| 43,200 | 338.81 | 223.0 | 2,296.1 | 1 | 0 | 87,002 |
| 86,400 | 334.58 | 216.4 | 2,296.1 | 0 | — | — |

The single matched prediction > 14,400 is LIRF AEZ146R (y 87,002, pred 84,798; item 2's row 15, the only matched rollover row).
Matched OOF rows with y ≥ 4 h: 2 (that row and LFPG y 15,059 pred 11,841). The largest matched predictions after it are 8,000 –
11,800 with labels 6,500 – 15,000 (LFPG, EGLL, LIRF, July / January); every matched cap raises matched RMSE.

### 3.8 (f) OOF tail-frequency calibration (labels known)

| group | n | pred > 2 h / y ≥ 2 h | pred > 4 h / y ≥ 4 h | pred > 12 h / y ≥ 12 h |
|---|---|---|---|---|
| all | 672,428 | 187 / 210 | 53 / 55 | 6 / 18 |
| unmatched (routing) | 8,378 | 151 / 163 | 52 / 53 | 5 / 17 |
| matched | 664,050 | 36 / 47 | 1 / 2 | 1 / 1 |
| LIRF unmatched | 537 | 127 / 146 | 43 / 51 | 5 / 15 |
| non-LIRF unmatched | 7,841 | 24 / 17 | 9 / 2 | 0 / 2 |

`>` and `≥` on predictions give identical counts at 4 h and 12 h. Overlap on unmatched rows: of 52 predictions > 4 h, 39 have
y ≥ 4 h (label median 24,805); of 53 labels ≥ 4 h, 39 are predicted > 4 h (prediction median 24,287; y mean 38,198 vs pred mean
25,800 — M's figures reproduced). Of 5 predictions > 12 h, 3 have y ≥ 12 h; of 17 labels ≥ 12 h, 3 are predicted > 12 h (y mean
70,140 vs pred mean 35,261). The model emits fewer tail predictions than there are tail labels in every group except non-LIRF
unmatched, where it emits 24 predictions > 2 h against 17 labels and 9 predictions > 4 h against 2 labels.

### 3.9 Ranking set: predictions above thresholds vs training frequency

Ranking rows 344,841 (Jan 2026 + Jul 2026); unmatched routing 5,290 / fid 5,283; matched 339,551 / 339,558; LIRF unmatched 383.
Counts with `>` and `≥` are identical in every cell.

| file | group | n | pred > 2 h | pred > 4 h | pred > 12 h | pred max |
|---|---|---|---|---|---|---|
| `outputs/main.parquet` (v7) | all rows | 344,841 | 122 | **34** | **3** | 84,866 |
| | unmatched (routing) | 5,290 | 100 | 33 | 2 | 66,693 |
| | unmatched (fid) | 5,283 | 99 | 33 | 2 | 66,693 |
| | matched (routing) | 339,551 | 22 | 1 | 1 | 84,866 |
| | matched (fid) | 339,558 | 23 | 1 | 1 | 84,866 |
| | LIRF unmatched | 383 | 90 | 33 | 2 | 66,693 |
| `outputs/probe_unmatched_from_v4.parquet` (v9) | all rows | 344,841 | 106 | **32** | **19** | 108,552 |
| | unmatched (routing) | 5,290 | 83 | 31 | 18 | 108,552 |
| | unmatched (fid) | 5,283 | 83 | 31 | 18 | 108,552 |
| | matched (routing) | 339,551 | 23 | 1 | 1 | 84,814 |
| | matched (fid) | 339,558 | 23 | 1 | 1 | 84,814 |
| | LIRF unmatched | 383 | 81 | 31 | 18 | 108,552 |

Training label frequency and implied ranking counts (rate × ranking group size); "Jan+Jul" uses only the two training months
that match the ranking months:

| group | thr | train n | n y ≥ thr (all months) | rate | rank n | implied (all months) | Jan+Jul train n / n y ≥ thr | implied (Jan+Jul rate) |
|---|---|---|---|---|---|---|---|---|
| all rows | 2 h | 2,085,047 | 584 | 2.80e-4 | 344,841 | 96.6 | 344,419 / 149 | 149.2 |
| all rows | 4 h | | 122 | 5.85e-5 | | **20.2** | / 45 | **45.1** |
| all rows | 12 h | | 34 | 1.63e-5 | | **5.6** | / 14 | **14.0** |
| unmatched (routing) | 2 h | 22,470 | 481 | 2.14e-2 | 5,290 | 113.2 | 5,373 / 111 | 109.3 |
| unmatched (routing) | 4 h | | 119 | 5.30e-3 | | **28.0** | / 43 | **42.3** |
| unmatched (routing) | 12 h | | 33 | 1.47e-3 | | **7.8** | / 13 | **12.8** |
| unmatched (fid) | 2 h | 22,424 | 481 | 2.15e-2 | 5,283 | 113.3 | 5,366 / 111 | 109.3 |
| unmatched (fid) | 4 h | | 119 | 5.31e-3 | | 28.0 | / 43 | 42.3 |
| unmatched (fid) | 12 h | | 33 | 1.47e-3 | | 7.8 | / 13 | 12.8 |
| matched (routing) | 2 h | 2,062,577 | 103 | 4.99e-5 | 339,551 | 17.0 | 339,046 / 38 | 38.1 |
| matched (routing) | 4 h | | 3 | 1.45e-6 | | 0.5 | / 2 | 2.0 |
| matched (routing) | 12 h | | 1 | 4.85e-7 | | 0.2 | / 1 | 1.0 |
| matched (fid) | 2 h / 4 h / 12 h | 2,062,623 | 103 / 3 / 1 | | 339,558 | 17.0 / 0.5 / 0.2 | 339,053 / 38 / 2 / 1 | 38.1 / 2.0 / 1.0 |
| LIRF unmatched (routing) | 2 h | 1,488 | 439 | 0.295 | 383 | 113.0 | 397 / 97 | 93.6 |
| LIRF unmatched (routing) | 4 h | | 115 | 0.0773 | | 29.6 | / 41 | 39.6 |
| LIRF unmatched (routing) | 12 h | | 30 | 0.0202 | | 7.7 | / 11 | 10.6 |

Training counts with `y >` and `y ≥` are identical at all three thresholds. The two unmatched definitions give the same tail counts
(481 / 119 / 33) and implied counts equal to 0.1.

Comparison (all rows): v7 predicts > 4 h on 34 rows vs 20.2 implied by the all-month rate (45.1 by the Jan+Jul rate); > 12 h on 3
vs 5.6 (14.0). v9: 32 vs 20.2 (45.1); 19 vs 5.6 (14.0). Unmatched (routing): v7 33 / 2 vs 28.0 / 7.8 (42.3 / 12.8); v9 31 / 18 vs
28.0 / 7.8 (42.3 / 12.8). P's "v5 emits 2 predictions > 43,200 (training rate 33/22,470 ⇒ ~7.8 expected)" and D's "≥ 12 h on
3–19, implies ≈ 6" are reproduced for v7 / v9 (2 / 18 unmatched; 3 / 19 all rows; 7.8 / 5.6 implied).

Ranking predictions each cap would change (fact, no labels): v7 unmatched 310 / 100 / 33 / 2 / 0 at 3,600 / 7,200 / 14,400 / 43,200
/ 86,400, matched 677 / 22 / 1 / 1 / 0; v9 unmatched 329 / 83 / 31 / 18 / 1, matched 644 / 23 / 1 / 1 / 0. Both files contain one
matched prediction of ≈ 84.8k (v7 84,866, v9 84,814), the ranking analogue of the AEZ146R row.

## 4. Verdict per sub-claim

| sub-claim | verdict | deciding observation |
|---|---|---|
| (a) A fixed cap on unmatched predictions improves RMSE | REFUTED (on this OOF) | Every cap ≤ 43,200 raises unmatched RMSE (2,296 → 3,568 / 3,373 / 3,070 / 2,306) and total RMSE (334.6 → 452.6 / 433.6 / 404.6 / 335.4); 86,400 changes nothing (no unmatched prediction exceeds 51,712). Identical under both unmatched definitions. M's counter-evidence reproduced to the unit. |
| (a2) Some cap quantile improves RMSE | REFUTED | RMSE is monotone increasing as q falls from 1.0 to 0.9 (2,296 → 2,306 → 2,330 → … → 3,694); no q < 1 beats the baseline. |
| (b) Shrinking toward the airport median improves RMSE | REFUTED for all λ ∈ {0.25, 0.5, 0.75} and for all seven target definitions tried | 0.75 → 2,467 – 2,479; 0.5 → 2,741 – 2,854; 0.25 → 3,090 – 3,145; constant → 3,492 – 3,806; baseline 2,296. Total RMSE 349 – 453 vs 334.6. |
| (c) Shrinking toward the (airport, hour) median improves RMSE and beats the airport median | REFUTED on the first part; second part: difference ≤ 35 s, both worse than baseline | hour_utc: 0.75 → 2,474.8, 0.5 → 2,767.4, 0.25 → 3,142.1 (airport median: 2,478.5 / 2,771.6 / 3,144.5); hour_local: 2,451.9 / 2,736.9 / 3,115.7. The hour dimension moves unmatched RMSE by 4 – 35 s against a 180 – 850 s deterioration from the shrink itself. |
| (d) Capping non-LIRF unmatched rows only improves RMSE | CONFIRMED on this OOF (small effect, small n) | 3,600 → unm 2,207.3 (−88.8), total 327.05 (−7.53); 7,200 → 2,234.3 / 329.33; 14,400 → 2,287.8 / 333.87. 53 / 24 / 9 rows affected; SSE delta negative in each of the 4 OOF months and at each of the 8 affected airports. Best of the finer sweep: 1,800 – 2,400 → 2,201.5 / 326.57 (with 147 / 42 rows hurt). The 24 non-LIRF rows predicted > 7,200 all have labels < 1,744. |
| (e) Capping matched rows improves RMSE | REFUTED | Every cap raises matched RMSE (216.4 → 243.7 / 237.8 / 234.0 / 223.0) and total (334.6 → 352.6 / 348.6 / 346.1 / 338.8). Caps 14,400 / 43,200 touch one row (y 87,002, pred 84,798). |
| (g) Submissions predict > 4 h / > 12 h more often than the training frequency implies | CONFIRMED for v7 and v9 at 4 h against the all-month rate; REFUTED against the Jan+Jul rate; at 12 h split: v7 below both implied counts, v9 above both | All rows: v7 34 / v9 32 predictions > 4 h vs 20.2 (all-month) / 45.1 (Jan+Jul); > 12 h: v7 3, v9 19 vs 5.6 / 14.0. Unmatched (routing): 33 / 31 vs 28.0 / 42.3; 2 / 18 vs 7.8 / 12.8. Identical counts under the fid definition; `>` = `≥` in every cell. On the OOF, where labels exist, the model under-emits at every threshold on LIRF unmatched (43 vs 51 at 4 h, 5 vs 15 at 12 h) and over-emits on non-LIRF unmatched (9 vs 2 at 4 h). |
| Hypothesis as a whole ("post-processing model B's tail improves RMSE") | REFUTED for cap / shrink applied to all unmatched rows or to matched rows; CONFIRMED only in the restricted form "cap non-LIRF unmatched rows", worth −7.5 total RMSE on this OOF | Tables 3.2 – 3.7. |

## 5. What cannot be determined locally

- The LB effect of any of these post-processings: ranking labels are unknown, so only the number of predictions each cap would
  change is observable (v7: 310 / 100 / 33 / 2 / 0 unmatched rows; v9: 329 / 83 / 31 / 18 / 1).
- Whether the non-LIRF gain in (d) transfers to 2026: it rests on 24 – 53 OOF rows in 2025 (4 months) whose labels were all < 4,747 s;
  the ranking set has 100 − 90 = 10 (v7) and 83 − 81 = 2 (v9) non-LIRF unmatched predictions > 7,200 and 310 − ? (not split here)
  > 3,600, and their labels are unknown.
- Whether the ranking LIRF rows carry the training tail frequency (BLOCK_TIME is null on every ranking row; items 1 – 3), which
  decides whether the "implied" counts in 3.9 are the right yardstick and which of the all-month or Jan+Jul rates applies.
- Anything about the loss-function half of item 14 (Huber / quantile / log1p): not fit here.

## 6. Facts vs inference

**Facts (measured in this run):** every number in §3; the exact reproduction of M's clip figures (3,568 / 3,373 / 3,070 / 2,306),
oracle airport-mean (3,492) and — under the "median of the OOF rows' own unmatched labels" definition — M's 3,724 / 2,825; the
monotone cap-quantile curve; the sign of the non-LIRF cap effect in all four OOF months and at all eight affected airports; the
ranking counts for v7 / v9 and the training frequencies under both unmatched definitions; `>` = `≥` in every prediction count.

**Inference (not established here):** that the LIRF deterioration under caps / shrinks is because the LIRF tail is "predictable
signal" (item 14's framing) — what is measured is only that the model's large LIRF predictions have large labels on average
(affected-row label means 9,084 – 47,417 in 3.2) while its large non-LIRF predictions have small labels (means 897 – 1,400 in 3.6);
that the non-LIRF gain would appear on the LB (24 – 53 rows, one OOF, one model); that M's airport-median was computed on the OOF
rows' labels (the numbers match exactly, but M's code was not inspected); that the discrepancy between "34 vs 20.2" and "34 vs 45.1"
should be resolved in favour of either rate (the ranking months are Jan and Jul, the LIRF unmatched count per month varies 52 – 337
in training, item 3).

**Definitional notes:** the two unmatched definitions give identical counterfactual RMSEs to 0.1 s and identical tail counts, because
the 46 / 7 routing-only rows have predictions and labels below every threshold used here. "Airport median" in the plan and in M is
not the same quantity: M's figures correspond to the in-sample OOF-label median (LIRF 1,729), this report's primary variant to the
training-label median (LIRF 3,922); the verdict is the same under both. E H6's "airport-hour median" was implemented with `hour_utc`
(and `hour_local`) of the take-off time, on unmatched training labels.
