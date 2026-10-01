# Data audit 2026-09-18 — lens E: where model A's remaining error lives, structurally

Script: `docs/data_audit_2026-09-18/scripts/matched-error.py`
(`POLARS_MAX_THREADS=1`, lazy scan of 33 columns, no model training, one pass, ~70 s wall).
Inputs: `experiments/oof_main-48c19bc830.parquet` ⋈ `data/work/feat_train.parquet` on `MVT_ID_mvt`,
non-null `oof` only. Every number below was produced by that one run.

**Definitions** (plan.md 2026-09-16 §0 throughout):
`y` = `TAXITIME_SEC_mvt` = MVT − BLOCK; `proxy` = `mvt_minus_aobt3` = MVT − AOBT_3;
**`d` = `y − proxy` = AOBT_3 − BLOCK** — the inter-system clock disagreement, and the only part of `y`
that is not already a visible column; `ms` = `mvt_minus_sched`; `as_` = `aobt3_minus_sched`;
`err` = `y − oof` (positive = model under-predicts); matched = `mvt_minus_aobt3` not null (routing definition);
`fill` = |BLOCK − SCHED| ≤ 10 s (label-derived, used for diagnosis only, not available at prediction time).

**Scope caveat carried from `fold-shape-vs-ranking.md`:** the OOF file scores only 2025 months 1, 7, 11, 12,
and the month-7 column is the leaky `jul` fold. All month-stability tests below therefore have four folds,
one of which is optimistic. This weakens the *positive* side of a stability claim, not the negative side:
a bias that does not survive leave-one-month-out here would not survive a clean split either.

---

## 1. What was checked

1. Frame and scope: matched/unmatched split of the 672,428 scored rows and their SSE shares.
2. Bias/noise decomposition of matched Σ`err`² for **32 groupings**: airport, stand, runway, stand×runway,
   operator, market segment, aircraft type, wake category, flight type, destination, ADES region, hour,
   weekday, month, `ms` bin, `as_` bin, proxy bin, queue bin, and 14 interactions up to
   airport × stand × runway × hour × aircraft type. For each: number of groups, in-sample between-group share
   of SSE, the share expected by chance (G·σ²/SSE), the excess, and within-group RMSE.
3. The same 32 groupings under **leave-one-month-out (LOMO)**: group means of `err` fitted on three OOF months,
   applied to the fourth, plain and with shrinkage k = 50. This is the honest "how much bias is recoverable"
   measurement; the in-sample number is not.
4. Per-group tables (share of squared error, within-group RMSE, mean and sd of `err`, mean and sd of `d`) by
   airport, segment, wake, flight type, hour, weekday, month, ADES region, top-20 operators, top-20 aircraft types.
5. Month-stable bias hunt: for 9 groupings, every group with n ≥ 400 (≥ 300 for stand×runway), all four OOF
   months present with ≥ 40 rows each, and all four monthly mean residuals of the **same sign**; ranked by the
   SSE that a perfect bias correction would remove (n·mean²).
6. Predictability of `d`: distribution; correlation against 22 visible numeric columns; in-sample and LOMO
   group-mean R² for 19 groupings including operator, airport, hour and aircraft type; `d` by airport and by
   `as_` bin; and how much of `d` model A already recovers (its implicit prediction is `oof − proxy`).
7. Ceiling: oracle group-mean predictors for 15 groupings — both "predict the group mean of `y`" (the form the
   earlier 268.1 measurement used) and **"predict proxy + the group mean of `d`"**, which is the form that uses
   the NM column. In-sample, LOMO, and LOMO-shrunk for each.
8. Tail concentration: SSE share of the top 0.01 %, 0.1 %, 1 %, 5 %, 10 % of matched rows by |`err`|;
   their composition; nine flag rates in each tail against the base rate; the 40 worst rows listed.
9. Sequential LOMO bias correction over six groupings stacked, to see what a bias-only fix would actually buy.

---

## 2. Findings

### 2.1 Scope: the matched rows are 98.75 % of scored rows and 41.3 % of the squared error

| set | n | RMSE | mean `err` | share of total OOF SSE |
|---|---|---|---|---|
| all scored | 672,428 | 334.58 | −4.48 | 1.0000 |
| **matched** | **664,050** | **216.44** | **−4.38** | **0.4132** |
| unmatched | 8,378 | 2,296.06 | −12.07 | 0.5868 |

Matched rows per OOF month: Jan 152,271 (RMSE 206.20), Jul 186,775 (247.26), Nov 160,976 (198.30),
Dec 164,028 (204.86). Reference points for everything below: **RMSE of the raw proxy as a predictor of `y` is
399.67**, of `proxy + constant` **399.28**, and model A achieves **216.44**.

### 2.2 The matched residual is almost pure noise: bias is ≤ 0.2 % of the squared error, and none of it is recoverable

**In-sample** between-group share of matched Σ`err`² (the ceiling on what subtracting group means could remove,
before any honesty correction), with the share attributable to chance:

| grouping | groups | between-group share | chance share | excess | within-group RMSE |
|---|---|---|---|---|---|
| airport × stand × runway × hour × actype | 243,464 | 0.4237 | 0.2113 | 0.2124 | 164.31 |
| airport × stand × runway × hour | 95,211 | 0.1844 | 0.1169 | 0.0675 | 195.46 |
| airport × operator × hour | 15,385 | 0.0478 | 0.0221 | 0.0257 | 211.20 |
| airport × stand × operator | 28,855 | 0.0650 | 0.0406 | 0.0244 | 209.29 |
| airport × operator × `as_` bin | 9,513 | 0.0348 | 0.0138 | 0.0210 | 212.63 |
| airport × stand × runway | 10,235 | 0.0268 | 0.0150 | 0.0118 | 213.52 |
| airport × operator | 2,452 | 0.0101 | 0.0037 | 0.0064 | 215.34 |
| airport × stand | 2,529 | 0.0086 | 0.0038 | 0.0048 | 215.51 |
| airport × actype | 1,084 | 0.0054 | 0.0016 | 0.0037 | 215.86 |
| airport × hour | 230 | 0.0034 | 0.0003 | 0.0031 | 216.06 |
| operator | 622 | 0.0034 | 0.0009 | 0.0025 | 216.06 |
| airport | 10 | 0.00098 | 0.000015 | 0.00096 | 216.33 |
| aircraft type | 207 | 0.0012 | 0.0003 | 0.0009 | 216.31 |
| proxy bin (9) | 9 | 0.00076 | 0.000014 | 0.00075 | 216.35 |
| hour | 24 | 0.00062 | 0.000036 | 0.00058 | 216.37 |
| `ms` bin | 6 | 0.00047 | 0.000009 | 0.00046 | 216.38 |
| weekday | 7 | 0.00046 | 0.000011 | 0.00045 | 216.39 |
| `as_` bin | 7 | 0.00044 | 0.000011 | 0.00043 | 216.39 |
| market segment | 8 | 0.00044 | 0.000012 | 0.00043 | 216.39 |
| wake category | 5 | 0.00043 | 0.000008 | 0.00043 | 216.39 |

Every grouping with at most ~2,500 groups explains **under 1.1 %** of matched squared error even in-sample
(airport 0.098 %, operator 0.34 %, aircraft type 0.12 %, hour 0.06 %, airport × operator 1.01 %). The larger
groupings rise only because they have thousands of groups for 664,050 rows, and about half of what they
"explain" is the chance term (0.2113 of 0.4237 for the five-way grouping).

**Honest (LOMO)** version — group means fitted on three OOF months, applied to the fourth. Positive `gain`
means RMSE fell:

| grouping | RMSE before | RMSE after | gain (s) | SSE reduction |
|---|---|---|---|---|
| flight type | 216.436 | 216.415 | +0.020 | +0.019 % |
| proxy bin | 216.436 | 216.417 | +0.018 | +0.017 % |
| weekday | 216.436 | 216.420 | +0.016 | +0.015 % |
| market segment | 216.436 | 216.422 | +0.013 | +0.012 % |
| wake category | 216.436 | 216.424 | +0.012 | +0.011 % |
| `as_` bin | 216.436 | 216.431 | +0.004 | +0.004 % |
| hour | 216.436 | 216.514 | −0.078 | −0.072 % |
| airport | 216.436 | 216.592 | −0.156 | −0.144 % |
| operator | 216.436 | 216.724 | −0.288 | −0.267 % |
| airport × hour | 216.436 | 216.890 | −0.454 | −0.420 % |
| airport × operator | 216.436 | 217.233 | −0.797 | −0.738 % |
| airport × stand | 216.436 | 217.492 | −1.057 | −0.979 % |
| airport × stand × runway | 216.436 | 220.514 | −4.078 | −3.804 % |
| airport × operator × hour | 216.436 | 221.456 | −5.020 | −4.693 % |
| airport × stand × runway × hour | 216.436 | 237.126 | −20.690 | −20.03 % |
| airport × stand × runway × hour × actype | 216.436 | 245.443 | −29.007 | −28.60 % |

With shrinkage k = 50 the damage is capped but nothing turns positive beyond +0.021 s
(best: flight type +0.0209 s, proxy bin +0.0185 s; airport −0.156, airport × stand −0.493,
airport × hour × weekday −0.713).

**The largest honest gain available from any single grouping is 0.024 s of RMSE** (month / airport × month,
which under LOMO degenerates to a global-mean correction of the −4.38 s offset). Stacking the six most
promising groupings sequentially, each fitted out-of-month and shrunk, moves matched RMSE the wrong way at
every step:

```
baseline                                     216.44
after airport x stand x runway            -> 217.08
after airport x operator                  -> 217.33
after airport x hour                      -> 217.47
after airport x actype                    -> 217.54
after aobt3-sched bin                     -> 217.56
after airport x proxy bin                 -> 217.55
```

### 2.3 Where the squared error sits (this is exposure, not bias)

Airport. `bias_share_of_own_mse` = mean²/MSE, i.e. how much of that airport's own mean squared error is a
constant offset:

| ADEP | n | share of rows | share of matched SSE | RMSE | mean `err` | median `err` | mean²/MSE |
|---|---|---|---|---|---|---|---|
| LIRF | 49,387 | 7.44 % | **19.13 %** | **347.10** | −2.70 | −24.05 | 0.00006 |
| LTFM | 90,041 | 13.56 % | 15.89 % | 234.29 | −5.37 | −24.82 | 0.00053 |
| LFPG | 76,471 | 11.52 % | 14.89 % | 246.10 | −3.45 | −17.61 | 0.00020 |
| EGLL | 78,986 | 11.89 % | 13.43 % | 230.01 | −10.92 | −14.45 | 0.00225 |
| EDDF | 71,489 | 10.77 % | 7.33 % | 178.59 | −2.09 | −14.22 | 0.00014 |
| LEBL | 55,501 | 8.36 % | 6.90 % | 196.65 | −4.95 | −17.97 | 0.00063 |
| LEMD | 70,316 | 10.59 % | 6.71 % | 172.27 | −14.55 | −24.68 | **0.00714** |
| EHAM | 78,286 | 11.79 % | 6.61 % | 162.09 | +2.00 | +5.14 | 0.00015 |
| EDDM | 51,286 | 7.72 % | 5.23 % | 178.16 | +1.60 | −9.28 | 0.00008 |
| LSZH | 42,287 | 6.37 % | 3.88 % | 168.88 | +0.99 | −13.02 | 0.00004 |

Even at the worst airport for bias (LEMD) the constant offset is 0.71 % of that airport's MSE.

Other groupings, RMSE within group (mean `err` in brackets): segment — Military 328.98 (+39.3, n 218),
Non-Scheduled 286.11 (−4.5), Other 264.65, Cargo 249.42, Lowcost 224.43, Business 218.55, Mainline 217.69,
Regional 168.61. Wake — J 264.77, H 246.45, L 218.59, M 207.10. Flight type — M 347.78 (n 177), X 358.05
(n 3,544), G 255.00, N 234.99, S 214.61 (94.8 % of SSE). Hour — 325.72 at 00 UTC down to 192.19 at 06 UTC,
back to 263.53 at 23; the SSE is spread 4–8 % per hour over 09–14. Weekday — 203.78 (Tue) to 230.95 (Sun).
Aircraft type — A320 15.2 % of SSE at RMSE 209.31, B738 9.8 % at 213.49, A21N 9.3 % at 240.69; the highest
RMSE among n ≥ 500 types is A339 278.15, A35K 272.26, A388 264.90, A332 264.35. Top operator by SSE
(`95828822…`, 74,478 rows, 12.79 % of matched SSE) has RMSE 231.11 and mean `err` **−3.79**; the second
(`8e92b7d7…`, 39,931 rows, 7.86 %) 247.50 and **−1.25**.

The pattern is uniform: groups differ a lot in **spread** and hardly at all in **centre**.

### 2.4 Month-stable bias subgroups exist, and they are tiny

Groups with n ≥ 400, all four OOF months present (≥ 40 rows each) and all four monthly means of the same sign:

| grouping | qualifying groups | Σ n·mean² as share of matched SSE | largest single group |
|---|---|---|---|
| airport | 3 | 0.0825 % | LEMD, n 70,316, mean −14.55 (monthly −20.67 … −4.05), t −22.5 |
| airport × runway | 6 | 0.0560 % | LEMD 36L, n 27,493, mean −13.87, t −13.0 |
| airport × stand | 96 | 0.0993 % | LIRF stand 606, n 471, mean −54.48, t −3.78 |
| airport × operator | 38 | 0.1165 % | LEMD × `32d3f744…`, n 9,939, mean −31.38, t −18.5 |
| airport × hour | 40 | 0.1574 % | LEMD h14, n 5,559, mean −32.09, t −12.9 |
| airport × aircraft type | 29 | 0.1057 % | LEMD B738, n 13,222, mean −26.59, t −18.7 |
| airport × stand × runway | 46 | 0.0332 % | EGLL 218L/27L, n 368, mean −47.91, t −4.56 |
| operator | 23 | 0.0586 % | `32d3f744…`, n 12,159, mean −23.94, t −14.6 |
| airport × segment | 9 | 0.1012 % | LEMD Mainline, n 38,828, mean −15.23, t −17.3 |

The largest stable mean residual among the groups listed (all families ranked by n·mean², top 25 each) is
**LEBL hour 22, n 666, mean −64.76 s** (monthly −91.4 … −19.9). Removing every stable bias in the best of these families would remove
**0.157 % of matched SSE**, i.e. 216.44 → 216.27 if the estimates were exact — and §2.2 shows that
out-of-month they are not. The signs are overwhelmingly negative (the model over-predicts slightly on
structured subsets), consistent with the global mean residual of −4.38 s.

LEMD is the one airport with a coherent, repeatable, cross-cutting offset: it appears at the top of the
airport, runway, operator, hour, aircraft-type and segment lists, all around −15 to −40 s.

### 2.5 The clock disagreement `AOBT_3 − BLOCK` is 70.6 % explained by the model already; static groupings reach 22 %

Distribution of `d` on the 664,050 matched scored rows:

| statistic | value |
|---|---|
| mean | −17.54 |
| median | −55.0 |
| sd | 399.28 |
| sd excluding the 689 rows with \|d\| > 3,600 s (0.104 %) | **362.46** |
| share \|d\| ≤ 60 s | **20.80 %** |
| share \|d\| ≤ 30 s | 10.46 % |
| share \|d\| > 600 s | 7.14 % |
| p1 / p25 / p75 / p99 / p99.9 | −786 / −185 / +121 / +1,259 / +3,185 |

(This reproduces the brief's "median −56 s, sd 364 s, 21.2 % within a minute" on the scored subset; the 364
figure is the trimmed sd — the untrimmed sd is 399.28 and 0.104 % of rows carry the difference.)

Correlation of `d` with visible numeric columns (all matched scored rows, NaN dropped):

| column | corr with `d` | corr with `err` |
|---|---|---|
| `aobt3_minus_eobt1` | **+0.516** | +0.0036 |
| `proxy` | **−0.394** | −0.0075 |
| `nbr_aobt_delay_mean_30m` | +0.287 | +0.0094 |
| `mvt_minus_lobt` | +0.265 | +0.0006 |
| `mvt_minus_eobt1` | +0.259 | −0.0010 |
| `aobt3_minus_sched` | +0.203 | −0.0034 |
| `nbr_mvt_minus_aobt_mean_30m` | −0.144 | −0.0056 |
| `mvt_minus_sched` | +0.132 | −0.0048 |
| `arr_taxi_in_mean_30m` | −0.118 | +0.0029 |
| `queue_ahead` | +0.093 | −0.0050 |
| `active_runways_60m` | −0.087 | +0.0140 |
| `dep_tko_prev_30m` | −0.073 | −0.0033 |
| `deicing_gate` / `low_vis` / `dest_distance_km` / `planned_flight_s` | +0.059 / +0.056 / +0.058 / +0.055 | ≤ 0.008 |
| `hour_utc` / `bank_position` / `stand_rwy_share` / `month` / `dow` / `rwy_headway_prev_s` | +0.017 / +0.026 / +0.015 / +0.012 / +0.003 / +0.003 | ≤ 0.004 |

`d` is visibly structured (|corr| up to 0.516). **`err` is not**: the largest |correlation| of the residual
with any of the 22 columns is **0.0140**.

Group-mean R² of `d`, in-sample vs honest LOMO (shrunk, k = 50):

| grouping | groups | in-sample R² | chance R² | LOMO R² | LOMO-shrunk R² | RMS `d` after |
|---|---|---|---|---|---|---|
| airport × operator × `as_` bin | 9,513 | 0.300 | 0.010 | **0.223** | 0.200 | **352.28** |
| airport × `as_` bin | 70 | 0.220 | 0.0001 | 0.205 | 0.205 | 356.29 |
| airport × op × hour × `as_` bin | 41,137 | 0.404 | 0.037 | 0.163 | 0.154 | 365.61 |
| airport × stand × runway | 10,235 | 0.187 | 0.013 | 0.130 | 0.121 | 372.70 |
| `as_` bin | 7 | 0.132 | 0.00001 | 0.128 | 0.128 | 373.14 |
| airport × stand | 2,529 | 0.151 | 0.003 | 0.128 | 0.123 | 373.23 |
| **airport × operator** | 2,452 | 0.139 | 0.003 | 0.114 | 0.111 | 376.23 |
| **airport × aircraft type** | 1,084 | 0.131 | 0.001 | 0.111 | 0.111 | 376.73 |
| airport × runway | 64 | 0.113 | 0.0001 | 0.102 | 0.102 | 378.78 |
| **airport × hour** | 230 | 0.110 | 0.0003 | 0.085 | 0.090 | 381.34 |
| **airport** | 10 | 0.094 | 0.00001 | 0.084 | 0.084 | 382.43 |
| **operator** | 622 | 0.066 | 0.0009 | 0.055 | 0.055 | 388.60 |
| `ms` bin | 6 | 0.052 | 0.00001 | 0.049 | 0.049 | 389.79 |
| airport × stand × rwy × hour × actype | 243,464 | 0.539 | 0.169 | **−0.115** | 0.017 | 396.27 |
| **aircraft type** | 207 | 0.020 | 0.0003 | 0.011 | 0.014 | 396.76 |
| market segment | 8 | 0.009 | 0.00001 | 0.005 | 0.005 | 398.70 |
| wake category | 5 | 0.004 | 0.00001 | 0.0008 | 0.0008 | 399.51 |
| **hour** | 24 | 0.007 | 0.00004 | −0.0007 | −0.0006 | 399.79 |

`d` by airport: LIRF mean +188.2 / median +64 / sd 529.1; LTFM −237.2 / −297 / 466.3; EHAM −143.4 / −101 /
288.0; EDDM +125.5 / +4 / 407.2; EGLL −29.0 / −56 / 445.7; LFPG +45.8 / +6 / 408.4; LSZH +66.6 / +37 / 282.3;
LEBL +55.3 / +4 / 310.4; EDDF −47.9 / −87 / 274.8; LEMD +21.8 / −4 / 269.2. Share |d| ≤ 60 s ranges from
LTFM 8.5 % to LEBL 26.8 %. `d` by `as_` bin: the mean rises monotonically with `as_`
(−630.3 at `as_` < −900 s, −191.7 at < 0, −7.2 at < 900, +96.5 at < 1,800, +151.6 at < 3,600, +181.3 at
< 7,200, +187.6 at ≥ 7,200) and so does its sd (709.6 / 245.9 / 258.5 / 373.1 / 507.7 / 710.4 / 1,089.4).

**How much of `d` is systematic, in numbers.** Predicting `y` by `proxy + constant` gives RMSE 399.28.
Model A gives 216.44. So model A already explains **70.62 %** of the variance of the clock disagreement;
`corr(oof − proxy, d) = 0.8404`. The best honest static grouping (airport × operator × `as_` bin) explains
22.3 %. Per airport, the share of `d`-variance model A explains:

| ADEP | sd `d` | corr(`oof−proxy`, `d`) | RMSE | variance of `d` explained |
|---|---|---|---|---|
| EDDM | 407.23 | 0.899 | 178.16 | 0.809 |
| LTFM | 466.29 | 0.865 | 234.29 | 0.748 |
| EGLL | 445.66 | 0.857 | 230.01 | 0.734 |
| EHAM | 287.98 | 0.827 | 162.09 | 0.683 |
| LSZH | 282.29 | 0.802 | 168.88 | 0.642 |
| LFPG | 408.44 | 0.798 | 246.10 | 0.637 |
| LEBL | 310.44 | 0.774 | 196.65 | 0.599 |
| LEMD | 269.15 | 0.770 | 172.27 | 0.590 |
| EDDF | 274.78 | 0.760 | 178.59 | 0.578 |
| LIRF | 529.15 | 0.755 | 347.10 | 0.570 |

`corr(err, d) = 0.543`: what is left of the matched residual is, to first order, the unrecovered part of the
same quantity. The residual carries no other visible structure (max |corr| with any numeric column 0.014,
§2.5 table; max honest group-mean gain 0.024 s, §2.2).

### 2.6 The ceiling: every static-grouping oracle, including ones using NM columns, is far above the model

Oracle = predict the group mean; no model fit. `meanY` is the form the earlier 268.1 measurement used;
`proxy + meanD` is the form that uses the NM column. LOMO columns fit the group means on three OOF months.

| grouping | groups | in-sample meanY | LOMO meanY | in-sample proxy+meanD | LOMO proxy+meanD | LOMO shrunk |
|---|---|---|---|---|---|---|
| airport | 10 | 408.25 | 410.20 | 380.33 | 382.43 | 382.43 |
| airport × stand | 2,529 | 375.20 | 380.53 | 368.19 | 373.23 | 374.21 |
| airport × stand × runway | 10,235 | 351.61 | 365.14 | 360.42 | 372.70 | 374.76 |
| airport × stand × runway × hour | 95,211 | 314.52 | 392.90 | 326.25 | 400.80 | 390.56 |
| **airport × stand × runway × hour × actype** | 243,464 | **265.97** | 422.03 | 271.25 | 421.99 | 396.27 |
| airport × operator | 2,452 | 383.15 | 388.84 | 370.79 | 376.23 | 376.78 |
| airport × operator × hour | 15,385 | 365.82 | 391.33 | 359.59 | 387.21 | 379.93 |
| proxy bin (16) | 17 | 361.99 | 364.02 | 356.03 | 358.56 | 358.55 |
| **airport × proxy bin** | 168 | 329.66 | 348.38 | 325.51 | **332.37** | 334.04 |
| airport × stand × runway × proxy bin | 48,910 | 264.91 | 352.36 | 266.97 | 349.10 | 377.00 |
| airport × stand × rwy × hour × actype × proxy bin | 377,823 | **157.84** | 420.96 | 158.05 | 400.44 | 397.91 |
| airport × operator × `as_` bin | 9,513 | 360.66 | 374.89 | 334.41 | **352.28** | 357.42 |
| airport × op × hour × `as_` bin | 41,137 | 336.49 | 385.27 | 308.60 | 365.61 | 367.62 |
| airport × `ms` bin × `as_` bin | 234 | 347.82 | 368.05 | 336.20 | 341.35 | 347.36 |
| airport × stand × runway × `as_` bin | 43,162 | 318.20 | 368.65 | 308.00 | 372.12 | 374.78 |

Reproduction check: airport × stand × runway × hour × aircraft type gives **265.97 in-sample**, against the
268.1 the earlier work reported for the same grouping (the small difference is the row set — this run scores
only the four OOF months).

Two things this settles:

- Adding NM columns to the static grouping does **not** produce a lower honest ceiling. The best LOMO oracle
  of any of the fifteen is **332.37** (airport × proxy bin, proxy + mean `d`), and the best `as_`-based one is
  352.28. Both are 54–63 % worse than model A's 216.44.
- The in-sample 157.84 for the six-way grouping (377,823 groups on 664,050 rows, 1.76 rows per group) is
  entirely memorisation: the same grouping scores 420.96 out-of-month.

**Model A at 216.44 sits below every one of the 45 oracle figures measured, with a single exception:** the
six-way airport × stand × runway × hour × actype × proxy-bin grouping in-sample (157.84), which has 1.76 rows
per group and scores 420.96 out-of-month. The next-lowest in-sample figures are 264.91 and 265.97, and the
lowest honest one is 332.37. There is no grouping-defined headroom above the model left to claim.

### 2.7 Tail: a third of the matched squared error is 1 % of the rows, and those rows are the clock disagreement

| tail | k | cutoff \|err\| | share of matched SSE | RMSE within | mean `err` |
|---|---|---|---|---|---|
| top 0.01 % | 66 | 3,322.0 | **3.46 %** | 4,037.3 | +3,454.4 |
| top 0.1 % | 664 | 1,756.9 | **13.35 %** | 2,500.7 | +1,869.8 |
| top 1 % | 6,640 | 732.9 | **34.02 %** | 1,262.5 | +548.1 |
| top 5 % | 33,202 | 398.1 | 56.59 % | 728.1 | +139.2 |
| top 10 % | 66,405 | 298.0 | 69.12 % | 569.0 | +64.4 |

Flag rates, tail vs base rate:

| flag | all matched | top 1 % | top 0.1 % | top 0.01 % |
|---|---|---|---|---|
| \|d\| > 600 s | 7.14 % | **80.74 %** | **95.03 %** | **98.48 %** |
| \|d\| > 1,800 s | 0.558 % | 24.32 % | 88.70 % | 95.45 % |
| \|d\| > 3,600 s | 0.104 % | 5.14 % | 24.10 % | 81.82 % |
| `fill` (\|BLOCK−SCHED\| ≤ 10 s) | 4.74 % | 14.28 % | 37.05 % | 42.42 % |
| `y` ≥ 3,600 s | 0.171 % | 11.55 % | 53.92 % | 93.94 % |
| `y` ≥ 7,200 s | 0.0071 % | 0.557 % | 3.16 % | 12.12 % |
| `ms` ≥ 7,200 s | 2.00 % | 9.01 % | 16.87 % | 28.79 % |
| LIRF | 7.44 % | **23.87 %** | **45.63 %** | **57.58 %** |
| `deicing_gate` = 1 | 0.756 % | 3.18 % | 3.61 % | 1.52 % |

Top-1 % composition by airport (SSE share of all matched SSE): LIRF 1,585 rows / 11.30 pp,
LFPG 1,111 / 5.60, LTFM 1,139 / 4.74, EGLL 805 / 3.82, LEBL 570 / 2.30, EDDF 410 / 1.83, EDDM 279 / 1.31,
LEMD 288 / 1.17, EHAM 273 / 1.11, LSZH 180 / 0.85. By segment: Mainline 4,360 rows / 21.21 pp,
Lowcost 1,371 / 8.58, Cargo 343 / 1.46, Regional 282 / 1.32. By `ms` bin: 45.9 % of top-1 % rows have
`ms` ≥ 1 h and 92.6 % have `ms` ≥ 30 min (against 2.00 % of all matched rows at `ms` ≥ 2 h). Mean `err` in the top 1 % is +548.1 — the tail is dominated by **under**-prediction.

Reading the 40 worst rows: the recurring shape is a LIRF `fill = true` Lowcost row where `y ≈ ms` and
`d ≈ as_` (e.g. LIRF Jul `15411f3d…` y 6,656, proxy 898, d 5,758, `as_` 5,760, oof 1,668) — the sched-filled
BLOCK already documented in `lirf-matched-fill.md`. Next most common is a non-filled row at LFPG/EGLL/LTFM
with a 4,000–11,000 s `d` and a normal proxy — the `|BLOCK − AOBT_3| > 1 h` set of `matched-proxy-outliers.md`.
Only 4 of the 40 are over-predictions; the largest is EDDM Dec `2c6b2d6b…` y 1,810, `ms` 12,425, oof 7,716,
`err` −5,907.

Removing tails:

| filter | n kept | share of rows | RMSE | SSE kept |
|---|---|---|---|---|
| \|err\| ≤ 600 | 652,814 | 98.308 % | 168.44 | 59.54 % |
| \|err\| ≤ 900 | 660,102 | 99.406 % | 183.72 | 71.63 % |
| \|err\| ≤ 1,800 | 663,428 | 99.906 % | 202.06 | 87.08 % |
| \|err\| ≤ 3,600 | 664,002 | 99.993 % | 213.42 | 97.22 % |
| **\|d\| ≤ 3,600** | **663,361** | **99.896 %** | **209.86** | **93.92 %** |
| \|d\| ≤ 7,200 | 663,983 | 99.990 % | 215.84 | 99.44 % |
| `fill` = false | 632,544 | 95.256 % | 208.96 | 88.79 % |

The 689 rows with |d| > 3,600 s are 0.104 % of matched rows and 6.08 % of matched SSE — the same 689 rows and
the same 6.08 % that `matched-proxy-outliers.md` §3.2 reports for `|BLOCK − AOBT_3| > 1 h`, an independent
reproduction. The 31,506 `fill = true` rows are 4.74 % of rows and 11.21 % of matched SSE.

---

## 3. What is CLEAN or irreducible, with the evidence

**CLEAN — the matched residual carries no recoverable group bias.** Over 32 groupings spanning every
categorical and binned-numeric attribute in `feat_train`, the honest (leave-one-month-out) gain from
subtracting group means is at most **+0.024 s** of RMSE and is negative for 22 of the 32. Stacking the six
most promising, shrunk and fitted out-of-month, moves 216.44 → 217.55. The in-sample between-group share is
below 1.1 % for every grouping with at most ~2,500 groups, and about half of the apparent share for the
largest groupings is the chance term G·σ²/SSE (0.2113 of 0.4237). **Facts: §2.2 tables.**

**CLEAN — the residual is orthogonal to every visible numeric column.** Across 22 numeric columns
(delay anchors, congestion counts, headways, taxi-in means, weather flags, distances, calendar), the largest
|correlation| with `err` is **0.0140** (`active_runways_60m`); every delay anchor is under 0.005. The same
columns correlate with the *label component* `d` at up to 0.516. **Facts: §2.5 table.**

**CLEAN — there is no static-grouping ceiling above the model.** The best honest oracle over fifteen
groupings, including ones built on `mvt_minus_aobt3` and `aobt3_minus_sched`, is 332.37 (airport × proxy bin),
and the classic airport × stand × runway × hour × aircraft type oracle is 265.97 in-sample / 422.03
out-of-month, against the model's 216.44. Extending the earlier 268.1 measurement to NM columns does not
lower it. **Facts: §2.6 table.**

**Month-stable bias exists but is arithmetically negligible.** The subgroups whose monthly mean residuals
all share a sign are real and statistically strong (LEMD n 70,316 mean −14.55 s, t −22.5, monthly −20.67…−4.05;
LEMD × `32d3f744…` n 9,939 mean −31.38, t −18.5; LEMD h14 n 5,559 mean −32.09, t −12.9; LEBL h22 n 666
mean −64.76). The total squared error a perfect correction of all of them would remove is **0.033–0.157 %**
of matched SSE, family by family. **Facts: §2.4 table.**

**Irreducible, with numbers.** Model A's task on matched rows is `d = AOBT_3 − BLOCK`, sd 399.28 (362.46
trimmed). The model already explains **70.62 %** of its variance (`corr(oof − proxy, d) = 0.8404`); the best
honest static grouping explains **22.3 %**. What remains — 216.44 s of RMSE, of which `corr(err, d) = 0.543`
says the bulk is still unrecovered `d` — has no group mean, no correlate and no oracle above it in this
measurement. Its concentration is extreme: 1 % of rows carry 34.02 % of the squared error, 0.1 % carry
13.35 %, 0.01 % carry 3.46 %; 80.7 % / 95.0 % / 98.5 % of those tails respectively have |d| > 600 s against a
7.14 % base rate. **Facts: §2.5, §2.7 tables.**

**Not clean, already documented elsewhere, reproduced here:** the LIRF sched-filled cell
(`lirf-matched-fill.md`) — LIRF is 7.44 % of matched rows and 19.13 % of matched SSE, 23.9 % of the top 1 %
and 57.6 % of the top 0.01 %; `fill` rows are 4.74 % of rows and 11.21 % of SSE and 42.4 % of the top 0.01 %.
The `|BLOCK − AOBT_3| > 1 h` set (`matched-proxy-outliers.md`) — 689 rows, 0.104 % of rows, 6.08 % of matched
SSE. Neither is re-audited here; both are confirmed as the dominant tail populations.

---

## 4. Facts vs inference

**Facts** (every number in §2 and §3, all produced by this run):
the matched/unmatched split and SSE shares; all 32 in-sample and 64 LOMO decompositions; the sequential
correction sequence 216.44 → 217.55; all per-group tables; all nine stable-bias families with their counts,
means, monthly ranges, t-statistics and bias shares; the distribution of `d` and its 22 correlations; the 19
group-mean R² rows for `d` in-sample, LOMO and LOMO-shrunk; the 15 oracle rows in both forms and three
fitting regimes; all tail SSE shares, cutoffs, flag rates and compositions; the tail-removal table; and the
cross-checks against `lirf-matched-fill.md` (LIRF matched RMSE 347.10 vs 347.1, SSE share 19.13 % of matched
= 7.90 % of total) and `matched-proxy-outliers.md` (689 rows, 6.08 % of matched SSE).

**Inference** (not measured here):

- I1 Calling `d = AOBT_3 − BLOCK` a "clock disagreement" is a reading of the two timestamps, not a provenance
  finding. The measurements are agnostic about which of the two systems is wrong on any given row, and
  `matched-proxy-outliers.md` §4 records the same ambiguity for the > 1 h cases.
- I2 "Bias is recoverable, noise is not" is the framing the task supplies. What is measured is that
  out-of-month group-mean corrections do not reduce SSE. A correction fitted with more folds, more
  regularisation, or interacted with continuous features has not been tested and is not excluded by these
  numbers — only the plain group-mean family is.
- I3 The LOMO measurements have four folds, one of which (2025-07) is the leaky `jul` fold per
  `fold-shape-vs-ranking.md`. A leaky fold makes `err` on those rows smaller and less structured, which would
  bias LOMO gains **toward zero** on the fold where month-7 is held out. The direction of this bias favours
  the "no recoverable bias" conclusion, so the conclusion is not established beyond the 3-clean-fold evidence.
- I4 Reading the 40 worst rows as "LIRF sched-fill" and "`|BLOCK − AOBT_3| > 1 h`" populations uses the
  mechanisms established in the two earlier result files; this run confirms their counts and SSE shares but
  did not re-derive the mechanisms.
- I5 The statement that "model A already sits below every static-attribute oracle" compares a fitted model
  scored out-of-fold against oracles scored out-of-month. The two protocols differ; the comparison is
  indicative, not a like-for-like contest.
- I6 Nothing here speaks to the 2026 ranking rows. Every measurement is on 2025 OOF months 1, 7, 11, 12.
