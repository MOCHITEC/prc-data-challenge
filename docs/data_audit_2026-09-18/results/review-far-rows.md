# Adversarial review of C3's "far-not-filled" 39,224 rows (2026-09-18)

**VERDICT: QUALIFIED.** Every number in the claim reproduces to the digit, and the half of it that says
*harm tracks `|BLOCK − AOBT_3|` and not the schedule-fill* is **confirmed and strengthened** — `|d|`
alone reproduces the whole far population's squared error to 0.13 %. The other half — that the 39,224
rows are a distinguishable, harmful, targetable set — is **refuted**: they are the large-`|d|` rows,
they carry **less** error than `|d|`-matched controls (RMSE 526.8 vs 721.9), the condition that defines
them is a function of the label, and on the best prediction-time-visible version of the set the model's
bias is **−3.1 s on 67,879 rows** (bias share 0.0001).

**NOT ACTIONABLE. Honest expected leaderboard gain: 0.00 s, uncertainty ±0.05 s** (the band is the
spread of honest group-mean corrections already measured on this residual, `matched-error.md` §2.2,
≤ +0.024 s). The oracle number is +26.0 s and is vacuous: it is the arithmetic of deleting 5.9 % of the
matched rows' error, selected by the label.

Script: `docs/data_audit_2026-09-18/scripts/review-far-rows.py`
(`POLARS_MAX_THREADS=1`, lazy scans, 25 columns, no model training, one pass, ~3 min wall).
Inputs, read-only: `experiments/oof_main-48c19bc830.parquet`, `data/work/feat_train.parquet`,
`data/work/feat_rank.parquet`. Writes nothing.

Notation (plan.md 2026-09-16 §0): `y` = `TAXITIME_SEC_mvt` = MVT − BLOCK; `proxy` = `mvt_minus_aobt3`
= MVT − AOBT_3; **`d` = `y` − `proxy` = AOBT_3 − BLOCK**; `err` = `y` − `oof`; `fill` = |BLOCK − SCHED|
≤ 10 s ≡ |`ms` − `y`| ≤ 10 s; "far" = `|d| > T`, T = 600 s unless stated; matched = `proxy` not null.

---

## 0. Frame and reproduction

672,428 scored rows, total OOF SSE **7.5275e10**, RMSE 334.582; matched 664,050 at RMSE 216.436
(41.32 % of SSE); unmatched 8,378 at 2,296.06 (58.68 %). The 2×2 reproduces exactly, including the
claim's headline cell:

| cell | n | RMSE | SSE share | mean `err` | median | bias share of MSE | y ≥ 2 h |
|---|---|---|---|---|---|---|---|
| LIRF / fill / far | 1,955 | 1,079.4 | 0.0303 | +677.7 | +427.3 | 0.394 | 11 |
| LIRF / fill / near | 6,995 | 154.6 | 0.0022 | +8.9 | +6.5 | 0.003 | 1 |
| LIRF / nofill / far | 4,350 | 562.3 | 0.0183 | +278.4 | +271.6 | 0.245 | 5 |
| LIRF / nofill / near | 36,087 | 242.9 | 0.0283 | −75.7 | −58.1 | 0.097 | 0 |
| non-LIRF / fill / far | 1,871 | 569.6 | 0.0081 | +261.8 | +83.4 | 0.211 | 1 |
| non-LIRF / fill / near | 20,685 | 144.9 | 0.0058 | +29.8 | +13.4 | 0.042 | 0 |
| **non-LIRF / nofill / far** | **39,224** | **492.7** | **0.1265** | **+67.4** | **−35.5** | **0.019** | **29** |
| non-LIRF / nofill / near | 552,883 | 162.5 | 0.1939 | −11.8 | −14.7 | 0.005 | 0 |

**Facts.** n, RMSE, SSE share and the 29 / 1 label counts are identical to C3's table.

---

## 1. Is it just "the model is worse where `|d|` is large"? — Yes, and it is *less* bad than that

All matched rows with `|d| > 600 s`: **n 47,400, RMSE 539.2, SSE share 0.1831**, 46 labels ≥ 2 h.
The claimed set is **82.75 % of those rows and 69.09 % of their squared error** — i.e. it is
sub-proportional: a random 82.75 % of the far population would carry 15.15 % of total SSE; the claimed
set carries 12.65 %.

**(a) At fixed `|d|`, being not-filled and non-LIRF is never worse** (matched scored rows, RMSE within
band):

| `|d|` band (s) | non-LIRF/nofill | non-LIRF/fill | LIRF/nofill | LIRF/fill |
|---|---|---|---|---|
| 600–900 | n 25,765 **334.6** | n 1,226 334.5 | n 2,569 342.8 | n 730 368.1 |
| 900–1,200 | n 6,487 **502.1** | n 320 600.3 | n 868 473.5 | n 382 552.7 |
| 1,200–1,800 | n 4,231 **676.8** | n 210 908.8 | n 566 680.9 | n 343 873.2 |
| 1,800–3,600 | n 2,229 **987.2** | n 95 1,325.2 | n 303 1,192.6 | n 387 1,608.3 |
| > 3,600 | n 512 **1,320.4** | n 20 844.1 | n 44 2,054.2 | n 113 2,660.4 |

**(b) A `|d|`-only model of squared error.** 40 quantile bins of `|d|` over all 664,050 matched scored
rows; each bin gets its mean `err²`; a set's predicted SSE is the sum of its rows' bin means. Ratio
actual / predicted:

| set | n | actual SSE share | `|d|`-only predicted share | ratio |
|---|---|---|---|---|
| **claimed set** | 39,224 | 0.1265 | 0.1465 | **0.863** |
| non-LIRF, fill, far | 1,871 | 0.0081 | 0.0070 | 1.147 |
| LIRF, nofill, far | 4,350 | 0.0183 | 0.0182 | 1.003 |
| LIRF, fill, far | 1,955 | 0.0303 | 0.0111 | **2.734** |
| **all far (`|d|` > 600)** | 47,400 | 0.1831 | 0.1829 | **1.001** |
| non-LIRF, nofill, near | 552,883 | 0.1939 | 0.2047 | 0.947 |

`|d|` alone reproduces the far population's squared error to **0.13 %**. The claimed set sits **13.7 %
below** what `|d|` alone predicts for it. The only cell that exceeds its `|d|`-only prediction
materially is `LIRF ∧ fill` (2.73×) — which is `fill-pattern.md`'s original harmful kind.

**(c) `|d|`-distribution-matched control.** Per `|d|` bin, draw as many rows from the *other* far cells
(filled, or LIRF) as the claimed set has, up to supply (8,176 pairs matched). Claimed-set RMSE
**526.8**, control RMSE **721.9**.

**(d)** Inside the claimed set, `corr(err, d) = 0.5679`, `corr(|err|, |d|) = 0.3797`; over all matched
rows `corr(err, d) = 0.5434` (`matched-error.md` reports 0.543). The set's internal error structure is
the same `d` relationship that already describes the whole matched residual.

**Fact:** every number above. **Inference:** that "not-filled ∧ non-LIRF" therefore names no mechanism —
the three measurements (sub-proportional SSE, ratio 0.863 against the `|d|`-only model, 526.8 vs 721.9
against `|d|`-matched controls) all point the same way, but a mechanism is not observable in these
columns. What is not an inference: the claimed set is the residual restated. `matched-error.md` §2.7
already recorded that 80.74 % of the top 1 % of matched squared error has `|d| > 600 s`; §1 here adds
that the fill/LIRF partition of those rows adds nothing to `|d|`.

---

## 2. Is the condition computable at prediction time? — No, twice over

**Fact 1 — the columns are absent.** On the 344,841 ranking rows: `BLOCK_TIME_UTC_mvt` null on
**344,841 (100 %)**, `TAXITIME_SEC_mvt` null on **344,841 (100 %)**. `AOBT_3_flt` and `mvt_minus_aobt3`
are null on 5,290 (1.53 %). So neither `far` nor `fill` is evaluable on a single scored row.

**Fact 2 — the condition is the label.** On matched rows the identity `d = y − proxy` is exact, so
`|BLOCK − AOBT_3| > 600 s` **is** `|y − proxy| > 600 s`: "the label differs from a visible feature by
more than ten minutes". Selecting rows by that and then observing the model has large error on them is
selection on the target.

**Best prediction-time-visible proxy.** AUC for the indicator of the claimed set, over all matched
scored rows (base rate 0.0591):

| column | AUC |
|---|---|
| **`|aobt3_minus_eobt1|`** | **0.7401** |
| `|aobt3_minus_sched|` | 0.6899 |
| `nbr_mvt_minus_aobt_mean_30m` | 0.6453 |
| `|mvt_minus_aobt3|` | 0.6356 |
| `|mvt_minus_eobt1|` / `|mvt_minus_iobt|` / `|mvt_minus_lobt|` | 0.6268 / 0.6265 / 0.6181 |

Precision and recall of the best one, `|aobt3_minus_eobt1| ≥ q`:

| q | flagged n | precision | recall | F1 | lift over base |
|---|---|---|---|---|---|
| 0.50 | 347,836 | 0.0911 | 0.8081 | 0.164 | 1.54 |
| 0.75 | 170,131 | 0.1430 | 0.6204 | 0.233 | 2.42 |
| **0.90** | **67,879** | **0.2051** | **0.3550** | **0.260** | 3.47 |
| 0.95 | 33,371 | 0.2491 | 0.2119 | 0.229 | 4.22 |
| 0.99 | 6,737 | 0.3632 | 0.0624 | 0.107 | 6.15 |
| 0.999 | 694 | 0.8184 | 0.0145 | 0.029 | 13.86 |

Best F1 over the sweep is **0.260** (precision 0.205, recall 0.355). Group-level rules do no better —
cells with ≥ 50 rows and rate ≥ 2× base:

| grouping | cells | max cell rate | recall | precision |
|---|---|---|---|---|
| ADEP | 10 | 0.1431 (n 90,041) | 0.329 | 0.143 |
| ADEP × operator | 820 | 0.5331 (n 242) | 0.375 | 0.153 |
| ADEP × operator × hour | 2,237 | 0.8103 (n 58) | 0.305 | 0.169 |
| ADEP × stand | 1,778 | 0.7000 (n 50) | 0.298 | 0.196 |

**Fact:** `aobt3_minus_eobt1` — the best proxy found — is already a model feature
(`src/prc/features.py`, `WIDE_RANGE_COLS`), and `matched-error.md` §2.5 measures its correlation with
the residual at **+0.0036** while its correlation with `d` is **+0.516**. The one visible column that
best flags these rows is one the model already consumes and has already extracted from.

---

## 3. Oracle ceiling — large, and meaningless

Metric arithmetic (the team's decomposition): N 344,841 = matched 339,551 + unmatched 5,290;
LB 284.3584 → total SSE 2.7884e10; matched at 218.0 → implied unmatched RMSE **1,490.2** (checks out).
Prevalence of the claimed set among matched scored rows 0.05907 → **20,057 ranking rows**, own MSE
242,772 (RMSE 492.7):

| intervention on those 20,057 rows | LB after | gain |
|---|---|---|
| **perfect oracle (MSE → 0)** | 258.3403 | **+26.018 s** |
| **halve their RMSE (MSE → ¼)** | 265.0843 | **+19.274 s** |
| bring them to the matched baseline 218 | 263.6357 | +20.723 s |
| 10 % MSE reduction | 281.8647 | +2.494 s |
| oracle on only the 35.5 % the best visible proxy recalls, ignoring the 79.5 % false positives | 275.4039 | +8.955 s |

Against the **+0.73 s** ceiling that killed the short-tail candidate: that was an oracle on **536
ranking rows identified by visible columns** (`RUNWAY=='H'`, stand, aircraft type, flight rule), on a
357.56 baseline — 0.20 % of the score, and every row of it selectable before prediction. This one is
+26.0 s = 9.15 % of a 284.36 baseline on **20,057 rows selectable only with the label in hand**. The
two numbers are not comparable, and the difference in size is the difference between an
identifiable set and a label-defined one.

That the number carries no information is shown by moving the threshold (§5 below): at T = 60 s the
identical arithmetic gives **+65.2 s** on 239,181 ranking rows (70.4 % of the matched side). The
statistic is monotone in set size and saturates at "the model's total matched error"; it measures where
error is, not what can be removed.

---

## 4. Does the model already handle them? — Yes; 98.1 % of the set's error is spread

On the 39,224 rows: RMSE **492.72**, mean `err` **+67.40**, median `err` **−35.47**, sd 488.09,
p1/p25/p75/p99 = −968 / −207 / +340 / +1,594. **Bias share of MSE 0.0187** — removing the global mean
of the set takes RMSE 492.72 → 488.09, i.e. **4.63 s** of that set's own RMSE, and only if you could
select the set.

In-sample (oracle, no honesty correction — an upper bound only) group-mean removal *inside* the set:

| grouping | groups | RMSE after | in-sample gain |
|---|---|---|---|
| ADEP | 9 | 455.15 | +37.57 |
| ADEP × operator | 1,236 | 442.61 | +50.11 |
| ADEP × operator × hour | 5,357 | 408.49 | +84.23 |
| ADEP × stand | 2,044 | 431.32 | +61.40 |
| month | 4 | 485.50 | +7.22 |

Per airport (claimed set vs that airport's own matched rows):

| ADEP | n | RMSE set | mean `err` | bias share | airport RMSE | ratio | SSE share |
|---|---|---|---|---|---|---|---|
| EDDF | 1,372 | 729.4 | +468.6 | 0.413 | 178.6 | 4.08 | 0.0097 |
| EDDM | 5,051 | 370.3 | +110.5 | 0.089 | 178.2 | 2.08 | 0.0092 |
| EGLL | 5,307 | 555.7 | −98.0 | 0.031 | 230.0 | 2.42 | 0.0218 |
| EHAM | 3,853 | 398.5 | −230.6 | 0.335 | 162.1 | 2.46 | 0.0081 |
| LEBL | 1,954 | 577.6 | +254.2 | 0.194 | 196.6 | 2.94 | 0.0087 |
| LEMD | 1,724 | 557.2 | +325.8 | 0.342 | 172.3 | 3.23 | 0.0071 |
| LFPG | 5,832 | 621.9 | +215.5 | 0.120 | 246.1 | 2.53 | 0.0300 |
| LSZH | 1,244 | 578.0 | +398.7 | 0.476 | 168.9 | 3.42 | 0.0055 |
| LTFM | 12,887 | 393.1 | +3.1 | 0.000 | 234.3 | 1.68 | 0.0265 |

**The decisive control.** Ask the same question of a set that *can* be selected at prediction time —
rows flagged by the best visible proxy:

| flag | n | RMSE | SSE share | mean `err` | bias share | RMSE after removing its mean |
|---|---|---|---|---|---|---|
| `|aobt3_minus_eobt1| ≥ q90` | 67,879 | 383.1 | 0.1323 | **−3.1** | **0.0001** | 383.1 |
| ≥ q95 | 33,371 | 450.6 | 0.0900 | +0.5 | 0.0000 | 450.6 |
| ≥ q98 | 13,513 | 549.2 | 0.0541 | +7.7 | 0.0002 | 549.1 |
| ≥ q99 | 6,737 | 636.0 | 0.0362 | +12.9 | 0.0004 | 635.9 |
| ≥ q99.9 | 694 | 1,083.4 | 0.0108 | −37.6 | 0.0012 | 1,082.8 |

A set of 67,879 rows carrying 13.23 % of total SSE — more than the claimed set — and the model is
unbiased on it to **3 s in 383**. **Inference:** the per-airport means in the table above (EDDF +468.6,
LSZH +398.7, EHAM −230.6) are a selection artefact of conditioning on `|y − proxy| > 600` with an
airport-specific sign of `d` (`matched-error.md` §2.5: LIRF `d` mean +188, LTFM −237, EHAM −143), not a
correctable model bias — they vanish the moment the same rows are approached through a visible column.

---

## 5. Is 600 s load-bearing? — No, and the contrast statistics are threshold-invariant

Non-LIRF, matched, scored; `sse_share` of *total* OOF SSE:

| T (s) | not-filled n | RMSE | SSE share | y ≥ 2 h | mean `err` | filled n | RMSE | SSE share | y ≥ 2 h | RMSE ratio f/nf | SSE ratio nf/f |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 60 | 467,760 | 217.6 | 0.2942 | **29** | −5.8 | 17,853 | 235.3 | 0.0131 | **1** | 1.081 | 22.4 |
| 120 | 358,481 | 237.6 | 0.2689 | **29** | −4.4 | 13,908 | 260.3 | 0.0125 | **1** | 1.095 | 21.5 |
| 180 | 268,712 | 261.8 | 0.2447 | **29** | −1.6 | 10,686 | 289.6 | 0.0119 | **1** | 1.106 | 20.6 |
| 300 | 149,663 | 317.9 | 0.2010 | **29** | +8.3 | 6,307 | 357.0 | 0.0107 | **1** | 1.123 | 18.8 |
| 450 | 76,350 | 395.3 | 0.1585 | **29** | +26.3 | 3,466 | 449.1 | 0.0093 | **1** | 1.136 | 17.1 |
| **600** | **39,224** | **492.7** | **0.1265** | **29** | +67.4 | **1,871** | **569.6** | **0.0081** | **1** | **1.156** | **15.7** |
| 900 | 13,459 | 702.3 | 0.0882 | **29** | +208.8 | 645 | 853.5 | 0.0062 | **1** | 1.215 | 14.1 |
| 1,200 | 6,972 | 847.1 | 0.0665 | **29** | +314.9 | 325 | 1,044.4 | 0.0047 | **1** | 1.233 | 14.1 |
| 1,800 | 2,741 | 1,057.4 | 0.0407 | **29** | +424.8 | 115 | 1,254.9 | 0.0024 | **1** | 1.187 | 16.9 |
| 2,400 | 1,392 | 1,157.2 | 0.0248 | **29** | +401.8 | 60 | 1,297.3 | 0.0013 | **1** | 1.121 | 18.5 |
| 3,600 | 512 | 1,320.4 | 0.0119 | 28 | +286.4 | 20 | 844.1 | 0.0002 | **1** | 0.639 | 62.6 |

Three facts:

1. **The 12.65 % is one point on a continuum.** The SSE share runs 0.2942 → 0.0119 over the sweep, a
   25-fold range; 600 s is not a knee (`review-refutations.md` already established the density has no
   knee outside LIRF). Nothing distinguishes it.
2. **"29 labels ≥ 2 h" is invariant at 29 for every threshold from 60 s to 2,400 s**, and the filled
   side's count is invariant at **1** for every threshold including 3,600 s. This is exactly the
   pathology C3 itself identified in `fill-pattern.md`'s "exactly one label above 2 h": the statistic
   returns the same answer whatever the cut, so it carries no information about the threshold or the
   split. C3 reproduced the pathology while criticising it.
3. **"21× more numerous, 16× more costly" is a count ratio.** The row-count ratio is 26.2 (T = 60),
   21.0 (T = 600), 21.5 (T = 1,200); the SSE ratio tracks it (22.4 / 15.7 / 14.1). The per-row
   comparison — RMSE ratio filled/not-filled — is **1.08–1.23 across the whole band**, i.e. the filled
   rows are consistently *worse per row*, by 8–23 %, at every threshold. C3's "16× more costly"
   describes only that there are 21× as many of them.

Oracle LB gain from perfect prediction on the not-filled side, over the same sweep:
T = 60 → +65.2 s (239,181 ranking rows); 300 → +42.6; **600 → +26.0**; 1,200 → +13.4; 3,600 → +2.3.
Monotone in set size, as §3 says.

---

## 6. Was C3's instrument capable of what it concluded?

For its *negative* half, yes: the 2×2 it ran is the right control, and it correctly shows `fill` moving
RMSE only 492.7 → 569.6 (1.16×) at fixed far while `far` moves it 162.5 → 492.7 (3.03×). That
conclusion — the harm is in `|d|`, not in `fill` — stands and is strengthened by §1(b) (ratio 1.001 for
the far population under a `|d|`-only model).

For its *positive* half, no. C3 named the 39,224 rows by their SSE share and label tail and stopped
there. It ran no control matched on `|d|` (§1(a)–(c) here: the cell is the least harmful of the four at
fixed `|d|`), no computability test on the identifying condition beyond noting the ranking nulls in its
own "domain of validity" paragraph, and no bias/spread split (§4: 98.1 % spread, and 0.01 % bias once
the rows are approached through a visible column). Its own §"Domain of validity" states that
`BLOCK` is null on all 344,841 ranking rows — and the report then treats the set as a finding anyway.

---

## 7. Facts vs inference

**Facts.** Every table above: the 2×2 reproduction; the far population totals (47,400 / 539.2 /
0.1831); the sub-proportional share 82.75 % of rows / 69.09 % of SSE; the banded RMSEs; the `|d|`-only
ratios (0.863 claimed set, 1.001 all-far, 2.734 LIRF∧fill); the `|d|`-matched control 526.8 vs 721.9 on
8,176 pairs; the ranking nulls (BLOCK 100 %, TAXITIME 100 %); the AUC table and the precision/recall
sweep; the group-rule ceilings; the metric arithmetic and every oracle figure; the bias/spread
decomposition and the per-airport table; the visible-flag table; the full threshold sweep.

**Inference.** That "not-filled ∧ non-LIRF" names no mechanism (the three measurements agree; the
mechanism is not observable in these columns). That the per-airport means inside the set are a
selection artefact of conditioning on the label rather than model bias (supported by the visible-flag
control showing mean `err` −3.1 s, not proved). That the honest expected leaderboard gain is 0.00 s:
this is a judgement that no correction is definable, since the set cannot be selected and the model is
unbiased on its best visible approximation; no correction was fitted and no model was trained here. That
the +26.0 s oracle is unachievable — it is the algebra of deleting label-selected error, and the
monotone sweep in §5 is the evidence that the quantity is not a ceiling on anything recoverable.

**Not determined here.** Whether a *feature* that predicts `d` better than `aobt3_minus_eobt1` exists
outside the 132 columns in `feat_train` (nothing in this run bears on that); whether the 1.16× residual
lift of `fill` at fixed far is composition (C3's own open item, unchanged).
