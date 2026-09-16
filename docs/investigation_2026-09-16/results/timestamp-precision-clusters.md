# Result — item 19 `timestamp-precision-clusters`

Script: `docs/investigation_2026-09-16/scripts/timestamp-precision-clusters.py` (no training, `POLARS_MAX_THREADS=1`,
lazy scans of 9 columns, runtime ≈ 1.5 min). Inputs: `data/work/feat_train.parquet`, `data/work/feat_rank.parquet`,
`experiments/oof_main-48c19bc830.parquet`. Code inspected: `src/prc/features.py`, `src/prc/train.py`. All numbers
below were re-derived by the script; none are copied from the hypothesis reports.

## 1. Hypothesis as stated (plan.md §1 item 19, source E H10)

> MVT_TIME second==0 share differs by airport (different data-source precision) and residual SD differs by cluster.

Related claims listed in the item's "Verified" block, re-checked here: MVT zero-second share ~8.3 % at
EDDM/EGLL/LFPG/LIRF/LTFM vs ~1.7 % elsewhere (E A3); AOBT_3 minute-rounded 96.6 % (H §2); EHAM AOBT_3 rounding 0.676
(Jan 2026) vs 0.749 (Jan 2025), the only airport not ~100 % rounded (D); BLOCK within 6 s of a minute boundary on
100 % of LIRF rows and 74 % of other unmatched rows (M; re-derived per airport by item 1's script).

Sub-claims tested:

- S1 MVT_TIME second == 0 share is ≈ 0.083 at the five "cluster" airports and ≈ 0.017 at the other five, in
  training and ranking, stable by month.
- S2 The difference reflects a different timestamp precision (a data-source property).
- S3 OOF residual SD differs by (airport, MVT second == 0).
- S4 OOF residual SD differs by (airport, AOBT_3 second == 0).
- S5 AOBT_3 is minute-rounded on 96.6 % of rows; EHAM is the only airport not ≈ 100 % rounded, 0.676 vs 0.749.
- S6 A second-of-minute indicator is (or is not) already a model feature.
- S7 The quantisation noise floor implied by minute-rounded stamps.

## 2. Procedure actually executed

Definitions: `sec(T)` = second-of-minute of timestamp T; `s0(T)` = `sec(T) == 0`; "cluster" = {EDDM, EGLL, LFPG,
LIRF, LTFM}; matched/unmatched = routing definition (`mvt_minus_aobt3` null → unmatched; plan §0); `resid = oof − y`;
`ym = year·100 + month` of MVT_TIME.

1. `feat_train` (2,085,047 rows): share of `s0` for MVT_TIME_UTC_mvt, BLOCK_TIME_UTC_mvt, AOBT_3_flt (over its
   non-null rows), SCHED_TIME_UTC_mvt, per ADEP, per ADEP × matched, per ADEP × month; the full 60-bin
   second-of-minute histogram of MVT for cluster vs rest and per airport (distinct values, mass on {54…59, 0…6});
   AOBT_3 and BLOCK second histograms; `y mod 60` and `(MVT − AOBT_3) mod 60` per airport.
2. `feat_rank` (344,841 rows): the same per ADEP and per ADEP × month (Jan / Jul 2026); BLOCK and TAXITIME null
   counts; side-by-side with training Jan / Jul 2025; EHAM AOBT_3 by month.
3. OOF: `oof_main-48c19bc830.parquet` (2,085,047 rows, `oof` non-null on 672,428 = months 202501/202507/202511/202512)
   joined to `feat_train` on `MVT_ID_mvt`, non-null `oof` only, 0 unjoined rows. Per (ADEP, s0(MVT)) and per
   (ADEP, matched, s0(MVT)): n, RMSE, sd(resid), mean/median resid, sd(y), median y, sd(oof); ratios s0 / other. The
   same per (ADEP, s0(AOBT_3)) on matched rows, per (ADEP, s0(BLOCK)), per (ADEP, month, s0(MVT)) matched, pooled
   cluster × s0, and the 2 × 2 cross s0(MVT) × s0(AOBT_3). Listing of the 10 scored rows with y ≥ 72,000 and each
   cell's RMSE with / without them.
4. Feature check: grep of `src/prc/` for `.dt.second`, `% 60`, `mod(60`, `sec0`, `second_zero`; the numeric feature
   list from `feat_train`'s schema (the `numeric_feature_columns` rule in `features.py`); data-side test of the
   identities `mvt_minus_sched mod 60 == 0 ⇔ s0(MVT)` and `mvt_minus_aobt3 mod 60 == 0 ⇔ sec(MVT) == sec(AOBT_3)`.
5. Quantisation arithmetic (no data).

## 3. Results

### 3.1 Share of second == 0, training (S1, S5)

| ADEP | n | unmatched | MVT s0 | BLOCK s0 | AOBT_3 s0 (non-null) | SCHED s0 | MVT within 6 s of :00 | BLOCK within 6 s | AOBT_3 within 6 s |
|---|---|---|---|---|---|---|---|---|---|
| EDDF | 230,141 | 1,956 | 0.0177 | 0.0169 | 0.9904 | 1.0 | 0.228 | 0.217 | 0.993 |
| EDDM | 167,334 | 1,010 | **0.0831** | **0.0834** | 0.9994 | 1.0 | **1.000** | **1.000** | 0.9995 |
| EGLL | 239,546 | 1,414 | **0.0828** | **0.0838** | 0.9999 | 1.0 | **1.000** | **1.000** | 0.9999 |
| EHAM | 247,951 | 4,084 | 0.0173 | 0.0169 | **0.8380** | 1.0 | 0.224 | 0.223 | **0.856** |
| LEBL | 179,705 | 1,862 | 0.0166 | **0.0830** | 0.9957 | 1.0 | 0.215 | **0.9995** | 0.996 |
| LEMD | 212,242 | 947 | 0.0164 | **0.0837** | 0.9972 | 1.0 | 0.217 | **0.9999** | 0.998 |
| LFPG | 239,552 | 3,773 | **0.0832** | **0.0828** | 0.9972 | 1.0 | **0.9996** | **1.000** | 0.998 |
| LIRF | 160,704 | 1,488 | **0.0838** | **0.0835** | 0.9957 | 1.0 | **1.000** | **1.000** | 0.9965 |
| LSZH | 134,907 | 2,291 | 0.0168 | 0.0198 | 0.9999 | 1.0 | 0.223 | 0.249 | 0.9999 |
| LTFM | 272,965 | 3,645 | **0.0833** | **0.0830** | 0.9891 | 1.0 | **1.000** | **1.000** | 0.991 |

Pooled: cluster MVT s0 0.0832 (1,080,101 rows), rest 0.0170 (1,004,946). Matched vs unmatched within an airport differ
by at most 0.025 (EHAM unmatched 0.042 vs matched 0.017; EDDM unmatched 0.064 vs 0.083; all other cells within ±0.01).

Per month (12 months): cluster MVT s0 ranges 0.0783–0.0878; rest 0.0146–0.0193. BLOCK s0: cluster + LEBL + LEMD
0.0780–0.0876; EDDF/EHAM 0.0154–0.0188; LSZH 0.0175–0.0219. AOBT_3 s0 by month: every airport ≥ 0.9748 except EHAM
0.7486 (Jan) – 0.8856 (Jul). SCHED s0 = 1.0 in every cell.

AOBT_3 exactly at :00: 2,015,132 rows. Over the 2,062,577 non-null rows that is **0.977**; over all 2,085,047 training
rows (nulls in the denominator) it is **0.9665** — this reproduces H §2's "96.6 %" and identifies its denominator.
The remaining AOBT_3 seconds are spread over all 59 other values (each ≤ 0.0007); there is no second mode at :30
(share ≤ 0.0034 at every airport).

### 3.2 What the 8.3 % is: a 13-value seconds grid, not minute rounding (S1, S2)

Full second-of-minute histogram of MVT_TIME, training:

- **Cluster** (1,080,101 rows): mass only on seconds 54–59 and 0–6. Seconds 55–59 and 0–5 each carry 0.0827–0.0836
  (1/12 = 0.0833); second 54 carries 0.0629 and second 6 carries 0.0208 (together 0.0837, one more 1/12 slot);
  seconds 7–53 carry 0.0000 (two seconds have 1 × 10⁻⁵). EDDM, LIRF, LTFM have exactly 13 distinct second values;
  EGLL 19 distinct (0.99997 of rows in {54…6}); LFPG 54 distinct (0.9996 in {54…6}).
- **Rest** (1,004,946 rows): all 60 values present, each 0.0162–0.0175 (1/60 = 0.0167). Excess mass at :00 over the
  mean of the other 59 seconds: EDDF +0.0011, EHAM +0.0007, LSZH +0.0002, LEBL −0.0001, LEMD −0.0003.

Consequently at the cluster airports "second == 0" is one of 12 equally likely slots (1/12 = 0.0833), not a
marker that distinguishes rows; and the rest's "1.7 %" is the uniform 1/60 baseline, i.e. no rows are minute-rounded
there. The naive reading "0.083 − 0.017 = 6.6 % of cluster rows are rounded" is wrong: 100 % of cluster rows sit
within 6 s of a minute boundary.

Per cluster airport, the edge cells are identical: second 54 = 0.0617–0.0634, second 6 = 0.0204–0.0215, seconds
53 and 7 = 0.0000 (LFPG 53: 0.00002).

BLOCK_TIME shows the same 13-value grid at the five cluster airports **and at LEBL and LEMD** (13 / 53 / 25 distinct
values, 0.9995–1.0 of rows in {54…6}); EDDF, EHAM, LSZH have all 60 values (0.217 / 0.223 / 0.249 within 6 s; the
uniform expectation is 13/60 = 0.217).

Label `y = MVT − BLOCK`, `y mod 60`: cluster airports have 25 distinct values (LFPG 56 distinct, 0.9997 within 12 s
of a multiple of 60; EGLL 30 distinct, 1.0 within 12 s), each interior value 0.0411–0.0419 (1/24 = 0.0417), edges
±12 at 0.021; **1.000 of cluster-airport labels lie within 12 s of a multiple of 60 s**. At EDDF/EHAM/LEBL/LEMD/LSZH
`y mod 60` is uniform (60 values, 0.415–0.422 within 12 s; expectation 25/60 = 0.417) — at LEBL/LEMD BLOCK is on
the grid but MVT is not, so their labels are not quantised.

Proxy `MVT − AOBT_3` mod 60 on matched rows: cluster airports 0.991–0.9999 within 6 s of a multiple of 60 (share
exactly 0: 0.0826–0.0836); rest 0.214–0.226 (uniform).

### 3.3 Ranking set (S1, S5)

BLOCK_TIME_UTC_mvt is null on 344,841 / 344,841 ranking rows and TAXITIME_SEC_mvt on 344,841 / 344,841, so no BLOCK
or label share can be computed for ranking. MVT and AOBT_3:

| ADEP | n | MVT s0 Jan 2026 | Jul 2026 | (train Jan / Jul 2025) | AOBT_3 s0 Jan 2026 | Jul 2026 | (train Jan / Jul 2025) |
|---|---|---|---|---|---|---|---|
| EDDF | 36,317 | 0.0184 | 0.0169 | 0.0165 / 0.0176 | 0.9984 | 0.9994 | 0.9748 / 0.9994 |
| EDDM | 25,966 | 0.0817 | 0.0815 | 0.0829 / 0.0833 | 0.9989 | 0.9996 | 0.9995 / 0.9997 |
| EGLL | 39,840 | 0.0870 | 0.0831 | 0.0851 / 0.0836 | 0.9999 | 1.0 | 0.9999 / 0.9998 |
| EHAM | 38,182 | 0.0161 | 0.0179 | 0.0166 / 0.0169 | **0.6763** | 0.9056 | **0.7486** / 0.8856 |
| LEBL | 30,080 | 0.0156 | 0.0151 | 0.0178 / 0.0171 | 0.9636 | 0.9998 | 0.9965 / 0.9972 |
| LEMD | 36,954 | 0.0166 | 0.0174 | 0.0173 / 0.0153 | 0.9932 | 0.9996 | 0.9989 / 0.9991 |
| LFPG | 39,872 | 0.0813 | 0.0816 | 0.0847 / 0.0819 | 0.9721 | 0.9978 | 0.9969 / 0.9981 |
| LIRF | 26,899 | 0.0822 | 0.0821 | 0.0847 / 0.0844 | 0.9882 | 0.9994 | 0.9984 / 0.9984 |
| LSZH | 23,150 | 0.0181 | 0.0180 | 0.0167 / 0.0167 | 0.9997 | 0.9998 | 0.9999 / 0.9998 |
| LTFM | 47,581 | 0.0841 | 0.0835 | 0.0811 / 0.0849 | 0.9956 | 0.9881 | 0.9956 / 0.9905 |

The MVT grid membership of each airport is unchanged between 2025 and 2026 (cluster 0.0813–0.0870, rest
0.0151–0.0184). EHAM AOBT_3 s0: 0.6763 (Jan 2026) vs 0.7486 (Jan 2025) — D's figures exactly; Jul 2026 0.9056 vs
Jul 2025 0.8856 (the other direction). Every other airport is ≥ 0.9636 in every ranking month (LEBL Jan 2026 0.9636
and LFPG Jan 2026 0.9721 are the lowest non-EHAM cells; in training the lowest non-EHAM month is EDDF Jan 2025 0.9748).

### 3.4 OOF residuals by (ADEP, MVT second == 0) (S3)

672,428 scored rows, RMSE 334.58; s0(MVT) share among scored rows 0.0514.

All scored rows (matched + unmatched), sd(resid) s0 / other:

| ADEP | n s0 | sd s0 | n other | sd other | ratio | sd(y) ratio |
|---|---|---|---|---|---|---|
| EDDF | 1,255 | 165.4 | 71,021 | 210.7 | 0.785 | 0.944 |
| EDDM | 4,291 | 186.6 | 47,397 | 199.2 | 0.937 | 1.006 |
| EGLL | 6,584 | 271.0 | 73,049 | 247.3 | 1.096 | 1.017 |
| EHAM | 1,338 | 167.2 | 78,301 | 201.6 | 0.829 | 0.945 |
| LEBL | 929 | 194.0 | 55,224 | 226.0 | 0.858 | 0.972 |
| LEMD | 1,182 | 173.3 | 69,550 | 174.1 | 0.995 | 1.001 |
| LFPG | 6,512 | **1,063.0** | 71,524 | 353.3 | **3.009** | 2.167 |
| LIRF | 4,206 | **1,274.6** | 45,718 | 765.6 | **1.665** | 1.413 |
| LSZH | 699 | 162.4 | 42,310 | 197.7 | 0.821 | 1.012 |
| LTFM | 7,579 | 239.6 | 83,759 | 240.4 | 0.997 | 0.996 |

**Matched rows only** (the model-A population):

| ADEP | n s0 | RMSE s0 | sd s0 | n other | RMSE other | sd other | sd ratio | sd(y) ratio |
|---|---|---|---|---|---|---|---|---|
| EDDF | 1,246 | 165.4 | 165.1 | 70,243 | 178.8 | 178.8 | 0.923 | 0.946 |
| EDDM | 4,264 | 180.4 | 180.4 | 47,022 | 178.0 | 177.9 | 1.014 | 1.013 |
| EGLL | 6,528 | 225.8 | 225.5 | 72,458 | 230.4 | 230.1 | 0.980 | 0.958 |
| EHAM | 1,287 | 163.4 | 163.4 | 76,999 | 162.1 | 162.1 | 1.008 | 0.939 |
| LEBL | 920 | 189.5 | 189.5 | 54,581 | 196.8 | 196.7 | 0.963 | 0.971 |
| LEMD | 1,173 | 170.9 | 170.4 | 69,143 | 172.3 | 171.7 | 0.992 | 0.996 |
| LFPG | 6,394 | 236.9 | 236.9 | 70,077 | 246.9 | 246.9 | 0.959 | 0.944 |
| LIRF | 4,164 | 344.6 | 344.5 | 45,223 | 347.3 | 347.3 | 0.992 | 0.814 |
| LSZH | 688 | 159.3 | 159.3 | 41,599 | 169.0 | 169.0 | 0.943 | 1.014 |
| LTFM | 7,484 | 236.3 | 236.3 | 82,557 | 234.1 | 234.0 | 1.010 | 0.999 |

Pooled matched: cluster s0 245.8 (n 28,834) vs other 248.4 (n 317,337); rest s0 169.7 (5,314) vs other 175.4
(312,565). Pooled over all matched rows without airport split: s0 235.6 (34,148) vs other 215.3 (629,902) — the
pooled gap is composition (s0 rows are 84 % cluster rows, cluster matched RMSE 248 vs rest 175), not a within-airport
effect. Per month (matched, 4 scored months × 10 airports = 40 pairs): s0 / other RMSE ratios 0.78–1.12, no airport
with a consistent sign (e.g. LIRF Jan 253 vs 288, Jul 459 vs 441, Nov 292 vs 295, Dec 295 vs 311).

**Unmatched rows only** (model-B population; cells are small):

| ADEP | n s0 | RMSE s0 | n other | RMSE other | ratio | y ≥ 72,000 rows in s0 / other cell |
|---|---|---|---|---|---|---|
| EDDF | 9 | 190.5 | 778 | 1,079.4 | 0.176 | 0 / 0 |
| EDDM | 27 | 626.2 | 375 | 1,021.3 | 0.613 | 0 / 0 |
| EGLL | 56 | 1,642.2 | 591 | 1,032.8 | 1.590 | 0 / 0 |
| EHAM | 51 | 242.5 | 1,302 | 943.6 | 0.257 | 0 / 0 |
| LEBL | 9 | 463.2 | 643 | 1,050.0 | 0.441 | 0 / 0 |
| LEMD | 9 | 396.5 | 407 | 418.8 | 0.947 | 0 / 0 |
| LFPG | 118 | 7,701.7 | 1,447 | 1,793.4 | 4.294 | **1** / 0 |
| LIRF | 42 | 12,283.2 | 495 | 6,566.1 | 1.871 | **2** / **6** |
| LSZH | 11 | 300.8 | 711 | 808.2 | 0.372 | 0 / 0 |
| LTFM | 95 | 425.7 | 1,202 | 514.4 | 0.828 | 0 / 0 |

The 10 scored rows with y ≥ 72,000 (item 2's rollover band) are LFPG unmatched 84,240 (MVT sec 0, BLOCK sec 0),
LIRF matched 87,002 (sec 1), and 8 LIRF unmatched 87,170–88,132 with MVT seconds {55, 58, 3, 0, 0, 59, 2, 56}. Excluding
them: LFPG unmatched s0 cell RMSE 7,701.7 → **699.7** (vs 1,793.4 other); LIRF unmatched s0 12,283.2 → 4,967.6 and
other 6,566.1 → 3,917.5. The all-rows ratios 3.009 (LFPG) and 1.665 (LIRF) above are these 1 and 2 rows.

### 3.5 OOF residuals by (ADEP, AOBT_3 second == 0), matched rows (S4)

| ADEP | n not-s0 | sd not-s0 | sd(y) not-s0 | n s0 | sd s0 | sd(y) s0 | sd ratio (s0 / not) |
|---|---|---|---|---|---|---|---|
| EDDF | 448 | 188.7 | 387.2 | 71,041 | 178.5 | 358.0 | 0.946 |
| EDDM | 28 | 261.1 | 355.7 | 51,258 | 178.1 | 368.5 | 0.682 |
| EGLL | 6 | 120.0 | 563.7 | 78,980 | 229.8 | 446.9 | 1.915 |
| **EHAM** | **15,011** | **155.7** | 320.4 | 63,275 | **163.5** | 346.2 | **1.050** |
| LEBL | 366 | 209.8 | 331.6 | 55,135 | 196.5 | 328.0 | 0.937 |
| LEMD | 72 | 366.5 | 465.6 | 70,244 | 171.3 | 306.3 | 0.467 |
| LFPG | 234 | 349.3 | 856.1 | 76,237 | 245.7 | 458.7 | 0.703 |
| LIRF | 95 | 359.3 | 621.1 | 49,292 | 347.1 | 678.0 | 0.966 |
| LSZH | 5 | 236.7 | 375.5 | 42,282 | 168.9 | 336.2 | 0.714 |
| LTFM | 606 | 370.1 | 391.8 | 89,435 | 233.0 | 384.7 | 0.630 |

Pooled: cluster not-s0 359.9 (n 969) vs s0 247.8 (345,202); rest not-s0 159.8 (15,902, of which 15,011 EHAM) vs s0
176.1 (301,977). EHAM is the only airport where the not-rounded cell has more than 606 rows; there the rounded rows
have 5 % higher residual sd and 8 % higher label sd. At the cluster airports the not-rounded AOBT_3 rows (0.01–0.7 % of
rows) have 1.03–1.6× the residual sd of the rounded rows (LTFM 370 vs 233 on 606 rows; LFPG 349 vs 246 on 234; LEMD
366 vs 171 on 72; EDDM/EGLL/LSZH cells have 28 / 6 / 5 rows).

2 × 2 cross on matched rows, pooled: (MVT not-s0, AOBT_3 not-s0) n 16,646 sd 176.1; (not-s0, s0) 613,256 sd 216.3;
(s0, not-s0) 225 sd 263.1; (s0, s0) 33,923 sd 235.4. `mvt_minus_aobt3 mod 60 == 0` on 1.000 of the (s0, s0) cell,
0.007 of the (not, not) cell, 0.0 of the mixed cells.

### 3.6 OOF residuals by (ADEP, BLOCK second == 0) — label side only

sd ratios s0 / other: EDDF 0.908, EDDM 0.973, EGLL 0.934, EHAM 0.913, LEBL **1.282** (4,706 vs 51,447 rows, 0 rollover
rows, sd(y) ratio 1.021), LEMD 1.016, LFPG 3.02 (→ 0.755 after removing the one rollover row: 266.0 vs 352.4), LIRF
1.34 (→ 627.4 vs 529.0 after removing 1 + 8 rollover rows), LSZH 1.015, LTFM 0.991. BLOCK is null on every ranking
row, so none of this is usable at prediction time.

### 3.7 Is a second-of-minute indicator a model feature? (S6)

- `grep -rn "\.dt\.second\|% 60\|mod(60\|sec_eq\|sec0\|second_zero" src/prc/` → no matches.
- `feat_train` numeric columns (the `numeric_feature_columns` rule in `features.py:466`) contain no second-of-minute
  column; `minute_of_day_utc` = `hour·60 + minute` (`features.py:85`), no seconds. Model B's `cols_b`
  (`train.py:350–356`) is a prefix-exclusion subset of the same list.
- Data-side identities: SCHED second == 0 on 2,085,047 / 2,085,047 rows, hence `mvt_minus_sched mod 60 == 0 ⇔
  s0(MVT)` holds on **2,085,047 / 2,085,047** rows; `mvt_minus_aobt3 mod 60 == 0 ⇔ sec(MVT) == sec(AOBT_3)` holds on
  **2,062,577 / 2,062,577** matched rows. `ADEP_mvt` is a categorical feature, so cluster membership (an airport
  property, §3.2) is available to both models directly.

### 3.8 Quantisation noise floor (S7, arithmetic)

- Rounding a single stamp to the minute: error uniform on [−30, 30) s → sd = 60 / √12 = **17.32 s**, variance 300 s².
- `y = MVT − BLOCK` with both stamps independently minute-rounded: triangular on [−60, 60] → sd = 60 / √6 =
  **24.49 s**, variance 600 s².
- Added in quadrature: RMSE 217.7 → 218.39 (+0.69 s; variance share 0.63 %); with the triangular floor +1.37 s.
  RMSE 334.58 → +0.45 s / +0.90 s. RMSE 162.1 (EHAM matched) → +0.92 s / +1.84 s. RMSE 347.1 (LIRF matched) → +0.43 s
  / +0.86 s.
- Observed cluster-airport spread of `y` around multiples of 60 s (§3.2: flat on ±11 s at 1/24 each, ±12 s at 0.021):
  sd ≈ √48 ≈ 6.9 s. This is the spread of the recorded label about the 60-s grid, not the error against the true
  taxi time, which is not observable.
- The measured pooled matched variance gap sd²(resid | s0) − sd²(resid | other) = 235.6² − 215.3² = 9,149 s² is 30×
  the 300 s² floor and is composition (§3.4), not quantisation.

## 4. Per sub-claim

| # | sub-claim | verdict | deciding observation |
|---|---|---|---|
| S1 | MVT s0 ≈ 0.083 at EDDM/EGLL/LFPG/LIRF/LTFM vs ≈ 0.017 elsewhere, in training and ranking, stable by month | **CONFIRMED** | Training 0.0828–0.0838 vs 0.0164–0.0177 (§3.1); months 0.0783–0.0878 vs 0.0146–0.0193; ranking 0.0813–0.0870 vs 0.0151–0.0184 (§3.3). E's "0.081–0.085 vs 0.015–0.018" reproduced. |
| S2 | The difference is a precision difference | **CONFIRMED as a description of the data; the "data-source" attribution is inference** | Cluster MVT seconds occupy only {54…59, 0…6} (13 distinct values at EDDM/LIRF/LTFM), each 1/12 slot 0.0827–0.0836; rest uniform over 60 values (§3.2). It is a 60-s grid with a ±6 s spread, not "8.3 % of rows minute-rounded": 100 % of cluster rows are within 6 s of :00. BLOCK is on the same grid at the cluster + LEBL + LEMD; AOBT_3 is exactly :00 on 0.977 of non-null rows everywhere except EHAM (0.838). The label `y` at the cluster airports lies within 12 s of a multiple of 60 on 1.000 of rows (25 distinct residues). |
| S3 | OOF residual SD differs by (airport, MVT s0) | **REFUTED for matched rows; UNDETERMINED for unmatched rows** | Matched within-airport sd ratios 0.923–1.014 at all 10 airports; pooled cluster 245.8 vs 248.4, rest 169.7 vs 175.4; per-month ratios 0.78–1.12 with no consistent sign (§3.4). At cluster airports s0 is a random 1/12 slot, so no difference is expected. Unmatched cells have n = 9–118 and ratios 0.18–4.3, and the two largest (LFPG 4.29, LIRF 1.87) are 1 and 2 day-rollover rows (RMSE 7,702 → 700 without the one LFPG row). |
| S3' | Residual SD differs between the cluster and the rest | **CONFIRMED as an airport-level fact, not attributable to precision** | Matched RMSE cluster 248.4 vs rest 175.5. But EDDM (cluster, 13-value grid) 178.0 ≈ EDDF (rest, full seconds) 178.8, and the cluster contains the four worst matched airports (LIRF 347, LFPG 247, LTFM 234, EGLL 230) whose excess is treated by items 4 and 5. Cluster membership is a function of `ADEP_mvt`, already a feature. |
| S4 | OOF residual SD differs by (airport, AOBT_3 s0) | **UNDETERMINED (weak)** | Only EHAM has a non-rounded cell of usable size (15,011 rows): rounded 163.5 vs not-rounded 155.7 (ratio 1.05, label sd ratio 1.08). At the other 9 airports not-rounded rows are 5–606 per airport (0.01–0.7 %) with sd 1.06–2.1× the rounded rows (LTFM 370 vs 233 on 606; LFPG 349 vs 246 on 234); too few rows to weigh, and not a cluster pattern (EDDF 0.95, LEBL 0.94). |
| S5a | AOBT_3 minute-rounded 96.6 % | **CONFIRMED with denominator identified** | 2,015,132 rows exactly at :00 = 0.9665 of all 2,085,047 training rows (nulls included) = 0.977 of 2,062,577 non-null rows. |
| S5b | EHAM AOBT_3 s0 0.676 (Jan 2026) vs 0.749 (Jan 2025); only airport not ≈ 100 % rounded | **CONFIRMED** | 0.6763 / 0.7486 (§3.3); Jul 0.9056 vs 0.8856 (opposite sign). EHAM 0.838 overall vs ≥ 0.9891 at every other airport in training; ≥ 0.9636 in every non-EHAM ranking cell. |
| S5c | BLOCK within 6 s of :00 on 100 % of LIRF rows, ≈ 74 % of other unmatched rows (M) | **CONFIRMED and sharpened** | Per airport BLOCK within 6 s: 1.000 at EDDM/EGLL/LFPG/LIRF/LTFM, 0.9995 LEBL, 0.9999 LEMD, 0.217 EDDF, 0.223 EHAM, 0.249 LSZH (§3.1). The "74 %" is the mixture of these. |
| S6 | A second-of-minute indicator is a model feature | **REFUTED as an explicit column; algebraically present** | No `.dt.second` / `% 60` in `src/prc/`; no such column in the 98 numeric features. `mvt_minus_sched mod 60 == 0 ⇔ s0(MVT)` on all rows and `mvt_minus_aobt3 mod 60 == 0 ⇔ sec(MVT) == sec(AOBT_3)` on all matched rows, and `ADEP_mvt` carries cluster membership. Whether the GBDT extracts a mod-60 pattern was not measured. |
| S7 | Quantisation noise floor | **stated (arithmetic)** | 17.32 s per stamp (300 s²); 24.49 s for a difference of two (600 s²); +0.69 s / +1.37 s on RMSE 217.7; +0.45 / +0.90 on 334.58. |

## 5. What cannot be determined locally

- Whether the 13-value seconds grid at the cluster airports arises from minute-resolution source data plus a small
  offset, from a per-airport clock correction, or from anything else: the columns carry no provenance. Only the
  distributional shape (flat 1/12 over 55…5, half-cells at 54 and 6, nothing at 7…53) is observable.
- Whether BLOCK at the ranking airports keeps the same grid in 2026: BLOCK is null on all 344,841 ranking rows.
- The LB effect of any use of the indicator (needs 2026 labels); the item's own procedure asks for none.
- Whether the trained GBDTs already exploit `mvt_minus_sched mod 60` or `ADEP_mvt` to the same effect as an
  explicit indicator (would require a model fit; not permitted for this item).
- Why the EHAM AOBT_3 rounding share moves by month (0.68–0.91) and why LEBL's BLOCK-s0 cell has residual sd 1.28×
  its complement with no label-sd difference (4,706 rows, 0 rollover rows): both are facts recorded here without a
  mechanism.

## 6. Facts vs inference

**Facts (measured by the script or read in code):** every number in §3 and in the deciding-observation column of §4;
the 13-value seconds grid and its per-airport membership (MVT: 5 airports; BLOCK: 7 airports); `y mod 60` confined to
25 residues at the cluster airports; AOBT_3 exactly at :00 on 0.977 of non-null rows (0.9665 with nulls in the
denominator); SCHED at :00 on every row; BLOCK and label null on every ranking row; absence of any second-of-minute
feature in `src/prc/` and the two mod-60 identities; the 10 rollover rows' seconds and their effect on the LFPG / LIRF
unmatched cells; the arithmetic in §3.8.

**Inference (not measured):**

- I1 That the grid reflects source-system precision (minute-resolution recording) rather than a downstream
  transformation; the data are consistent with either.
- I2 That because s0(MVT) at the cluster airports is a uniform 1/12 slot, an indicator built on it cannot carry
  information beyond `ADEP_mvt`; the matched-row ratios 0.92–1.01 are consistent with this but do not prove absence
  of any interaction.
- I3 That the quantisation floor (17–25 s) is negligible against residual sd 160–350 s: the arithmetic in §3.8 shows
  a ≤ 1.4 s RMSE contribution, which is below the 2–28 s scale of the structural changes E H12 cites and near the
  0.25 s screening resolution it mentions; whether it matters for the LB is not a local question.
- I4 That the "cluster vs rest" residual difference (248 vs 175 matched) is airport identity and the mechanisms of
  items 1, 4, 5 rather than precision: EDDM ≈ EDDF (178 vs 179) is one comparison, not a controlled test.

**Definitional notes for cross-item comparison:** H §2's 96.6 % uses all training rows as denominator (null AOBT_3
included); over matched rows it is 97.7 %. E A3's "~1.7 % elsewhere" is the uniform baseline 1/60 = 1.67 %, not a
rounded fraction. Item 1's "BLOCK within 6 s of a minute boundary on 74 % of other unmatched rows" is the mixture of
seven airports at ≈ 1.0 and three at 0.22–0.25. "Cluster" in this report is E's five-airport MVT grouping; the BLOCK
grid additionally covers LEBL and LEMD.
