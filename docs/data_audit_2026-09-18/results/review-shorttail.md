# Adversarial review — short tail of the label (label-plausibility.md §§2.1–2.6)

Script: `docs/data_audit_2026-09-18/scripts/review-shorttail.py`
(no training, `POLARS_MAX_THREADS=1`, ~4 min; reads `data/work/feat_train.parquet`,
`data/work/feat_rank.parquet`, `experiments/oof_main-48c19bc830.parquet`,
`data/raw/training_*.parquet` for the ARR taxi-in floor only.)
Every number below came out of that one run. Nothing is copied from the audit.

---

## Verdicts

**Data verdict: CONFIRMED.** Every count the audit reported reproduces exactly — 387 / 1,001 / 990 /
2,193 / 2,930 / 527 in training, 79 / 156 / 325 / 419 in the ranking set, and 188 of the 369
negatives on the LSZH helipad. Not one number is wrong.

**Score verdict: NOT ACTIONABLE.** The decisive observation: on the helipad rows the model's median
out-of-fold prediction is **45.1 s**, against **738.7 s** for the non-helipad LSZH rows it is pooled
with, and its median absolute residual there is **43 s**. `RUNWAY_mvt` and `STAND_mvt` are already
LightGBM categoricals, and the per-fold `(ADEP, STAND, RUNWAY)` label encoding hands the model
`ref_med = −1.0` for `(LSZH, HPW, H)` and `+1.0` for `(LSZH, REG, H)` outright. The helipad is not a
population the model is blind to; it is a population the model was already told about. The finding is
true about the data and worth ~0 s on the score.

---

## 1. Facts: the decisive test — the model's own residual on each population

`oof_main-48c19bc830.parquet` carries a prediction for **672,428** of the 2,085,047 training rows
(the jan + jul + forward fold validation sides: 153,706 + 190,713 + 328,009). Global OOF RMSE
**334.58**, bias **+4.48**, total SSE **7.528 × 10¹⁰**. Every residual figure below is on the ~32 %
of each population that has a prediction, which is the precision limit of this whole review.

| population | n with pred | med y | **med pred** | RMSE | drop-1 | drop-5 | med \|res\| | bias | share of total SSE |
|---|---|---|---|---|---|---|---|---|---|
| `RUNWAY=='H'` (all LSZH) | 130 | 2 | **45.1** | 676.3 | **131.0** | **102.6** | **43.0** | +21.9 | 0.079 % |
| EHAM `HG01` | 310 | 173.5 | **283.7** | 640.2 | **170.8** | **150.0** | 106.2 | +45.2 | 0.169 % |
| `AIRCRAFT_TYPE=='A139'` | 315 | 171 | 281.8 | 457.1 | 165.2 | 149.0 | 107.5 | +51.3 | 0.087 % |
| `ADEP==ADES` local | 663 | 239 | 334.1 | 535.9 | 330.7 | 287.7 | 131.1 | +42.3 | 0.253 % |
| `FLIGHT_RULE=='V'` | 860 | 190 | 318.8 | 602.4 | 475.3 | 249.8 | 124.7 | +38.3 | 0.415 % |
| all rows | 672,428 | 920 | 946.5 | 334.6 | — | — | 100.0 | +4.5 | 100 % |

Reference points for the medians: LSZH excluding runway H, median prediction **738.7**; EHAM
airport-wide label median **742**; global label median **912**.

Three things follow directly.

* **The RMSEs are one row each.** Helipad 676.3 → 131.0 after dropping the single largest residual
  (|res| 7,566) and 102.6 after five. EHAM HG01 640.2 → 170.8 → 150.0. The audit's populations do
  not carry a wide error distribution; they carry a handful of long-tail rows that are *long*, not
  short, and that the audit's own §2.8 already covers under a different mechanism.
* **The bias is positive and small.** +21.9 s on the helipad, +38 to +51 s elsewhere. The model
  over-predicts these rows slightly. It is not predicting a 742 s taxi for a 0 s helicopter lift.
* **Their combined share of squared error is 0.47 %** (the deduplicated union of the four; §3).

### 1a. What the model is handed, not merely allowed to learn

`src/prc/features.py:32` `CATEGORICALS` = `ADEP_mvt, RUNWAY_mvt, STAND_mvt, AIRCRAFT_TYPE_mvt,
WK_TBL_CAT_flt, AIRCRAFT_OPERATOR_flt, MARKET_SEGMENT_flt, FLIGHT_TYPE_flt, ADES_mvt, ADES_REGION`,
and `src/prc/train.py:365` passes all ten present columns as `categorical_feature`. On top of that,
`fit_encodings` (train.py:160) builds a per-fold `(ADEP_mvt, STAND_mvt, RUNWAY_mvt)` median/P10/P20
table at `reference_min_count = 30`. Recomputing it on the full labelled training side:

| triple | n | `ref_med` | `ref_p10` |
|---|---|---|---|
| LSZH / HPW / H | 294 | **−1.0** | −10.0 |
| LSZH / REG / H | 75 | **+1.0** | −9.0 |
| EHAM / HG01 / 22 | 617 | **176.0** | 65.0 |
| EHAM / HG01 / 04 | 338 | **174.0** | 61.0 |

Ranking-side coverage of that encoding: helipad **64/79** rows get a `ref_med` (median −1.0),
EHAM HG01 **143/156** (median 176.0), local **262/325** (234.0), VFR **379/419** (176.0). The model
is literally given the number the audit proposes to discover.

Cold-start caveat, and it is the audit's one genuine gap: of the 12 distinct
`(ADEP, STAND, RUNWAY)` triples among the 79 ranking helipad rows, only **4** occur in training, so
15 of 79 rows fall back on the categoricals alone. `(LSZH, H)` as an `(ADEP, RUNWAY)` pair is
present, and all 5 EHAM HG01 ranking triples are present.

### 1b. The one thing the model genuinely does not receive

**`FLIGHT_RULE_mvt` is not a feature.** It is in `feat_train.parquet` as a `Utf8` column, it is not
in `CATEGORICALS`, and `numeric_feature_columns` selects numeric dtypes only, so it reaches neither
list. Same for `FLIGHT_RULE_flt`. This is the audit's only claim that survives the "already handled"
test — and see §4 for what it is worth.

### 1c. A fact the audit missed that undercuts its own framing

**All 2,930 VFR training rows and all 419 VFR ranking rows are unmatched** (`FLIGHT_ID_mvt` null /
`mvt_minus_aobt3` null), so every one of them is predicted by **model B**, never model A. Likewise
975/1,001 EHAM HG01, 384/387 helipad, 1,678/2,193 local. The populations live almost entirely inside
the 5,290-row model-B regime. And inside that regime they are the **easy** rows: the union's
unmatched own MSE is **366,848** (RMSE 606) against model B's overall OOF **2,298.6** (MSE
5.28 × 10⁶) — one fourteenth of the average model-B squared error. The short tail is not where
model B's error is.

---

## 2. Facts: re-derived counts — what the audit got wrong

Independently recomputed from `feat_train` / `feat_rank` (a different code path from the audit's raw
scan). All 2,085,047 DEP rows are labelled, so "rows" and "labelled rows" coincide.

| audit claim | audit | this review | verdict |
|---|---|---|---|
| `RUNWAY=='H'` training rows, all LSZH | 387 | **387** (LSZH 387) | ✔ |
| … median label | 0 | **0** | ✔ |
| … `y < 60` | 366 | **366** | ✔ |
| … share of the 369 negatives | 188 | **188** | ✔ |
| … ranking rows | 79 | **79** (LSZH 79) | ✔ |
| EHAM HG01 training rows | 1,001 | **1,001** | ✔ |
| … of which A139 | 990 | **990** (then DH8A 6, DA62 4, AS32 1) | ✔ |
| … median vs EHAM median | 176 vs 742 | **176 vs 742** | ✔ |
| … ranking rows | 156 | **156** | ✔ |
| `ADEP==ADES` training rows | 2,193 | **2,193**, 9 airports, per-airport table identical | ✔ |
| … median vs global | 243 vs 912 | **243 vs 912** | ✔ |
| … ranking rows | 325 | **325**, per-airport split identical | ✔ |
| `FLIGHT_RULE=='V'` training rows | 2,930 | **2,930** (I 2,081,914, null 203) | ✔ |
| … median / `y<60` / `y<0` | 232 / 580 / 224 | **232 / 580 / 224** | ✔ |
| … ranking rows | 419 | **419** (I 344,402, null 20) | ✔ |
| 527 "hard-impossible" rows | 527 | **527** (y<60 ∧ trip_med≥300 ∧ ARR p05≥120) | ✔ |
| … intermediate counts | 1,047 / 558 / 366 / 248 | **1,047 / 558 / 366 / 248** | ✔ |

**Nothing is arithmetically wrong.** The two things wrong are framing, not counting:

1. **"Roughly 900 ranking rows" is double counting.** §3.
2. **"Or more generally `AIRCRAFT_TYPE_mvt == 'A139'`" (§2.2) proposes an identifier the model
   already has** — `AIRCRAFT_TYPE_mvt` is categorical. And it is not even coextensive with the
   population: A139 also appears at LSZH (13 training rows, median **−1**; 11 ranking rows), so the
   "general" rule splits across two airports with medians 176 and −1.

---

## 3. Facts: overlap — the populations are mostly the same rows

Pairwise, and then the union.

| pair | training | ranking |
|---|---|---|
| helipad ∧ EHAM HG01 | 0 | 0 |
| helipad ∧ local | 26 | 0 |
| **helipad ∧ VFR** | **382 / 387** | **78 / 79** |
| **EHAM HG01 ∧ local** | **867 / 1,001** | **135 / 156** |
| **EHAM HG01 ∧ VFR** | **971 / 1,001** | **148 / 156** |
| **local ∧ VFR** | **1,620 / 2,193** | **217 / 325** |

| | naive sum of the four | deduplicated union |
|---|---|---|
| training | 6,511 | **3,533** |
| ranking | 979 | **536** |

Membership histogram, ranking: 228 rows in exactly one population, 173 in two, **135 in three**, 0
in four. So the audit's "roughly 900 ranking rows" overstates the deduplicated population by **83 %**
(979 → 536), and the single largest block — 135 ranking rows — is one row wearing three of the four
labels at once: a VFR helicopter on an EHAM local round trip out of HG01. The union's OOF residual is
RMSE 580.6, bias +37.0, **0.473 %** of total squared error; adding A139 moves that to 0.474 %.

---

## 4. Facts: prediction-time identifiability, and the effect-size ceiling

Identifiability holds. Ranking-set non-null rates: `ADEP_mvt` 1.000000, `RUNWAY_mvt` 1.000000,
`STAND_mvt` 0.999988, `ADES_mvt` 0.999962, `FLIGHT_RULE_mvt` 0.999942, `AIRCRAFT_TYPE_mvt` 0.998875.
The values themselves occur in 2026: `'H'` ✔, `'HG01'` ✔, `'V'` ✔, `'A139'` ✔.

Leaderboard arithmetic, the team's decomposition. Verified: `feat_rank` has 5,290 rows with
`mvt_minus_aobt3` null (5,283 with `nm_matched == 0`; the 7-row difference is rows with a
`FLIGHT_ID` but no `AOBT_3`), 344,841 total. Taking matched at 218 and unmatched at the OOF 2,298.6:

```
SSE = 339,551 × 218²  +  5,290 × 2,298.6²  =  1.614e10 + 2.795e10  =  4.409e10
RMSE = sqrt(4.409e10 / 344,841) = 357.56        (unmatched = 63.4 % of SSE on 1.5 % of rows)
```

Replacing each population's ranking rows with a **perfect oracle**, charging each its own measured
OOF MSE:

| population | ranking n (m / u) | own MSE m / u | 357.56 → | **ceiling gain** |
|---|---|---|---|---|
| `RUNWAY=='H'` | 79 (1 / 78) | 55,137 / 460,514 | 357.413 | **+0.146 s** |
| EHAM HG01 | 156 (8 / 148) | 43,886 / 418,268 | 357.306 | **+0.253 s** |
| A139 | 165 (18 / 147) | 55,455 / 216,082 | 357.426 | **+0.133 s** |
| `ADEP==ADES` | 325 (87 / 238) | 193,179 / 318,599 | 357.183 | **+0.376 s** |
| `FLIGHT_RULE=='V'` | 419 (0 / 419) | — / 362,869 | 356.942 | **+0.617 s** |
| **deduplicated union of four** | **536 (96 / 440)** | 186,380 / 366,848 | **356.831** | **+0.728 s** |

**The honest ceiling for the whole finding, with an oracle, is 0.73 s on a 357.6 s score — 0.20 %.**
Not the 1.39 s the individual rows sum to, because 45 % of them are the same rows.

And the achievable fraction of that 0.73 s is smaller still. Replacing the model's prediction on each
population by that population's *training median* — the strongest thing a "identify the helipad" rule
can do without an oracle — measured on the OOF rows:

| population | training median | model RMSE | median-rule RMSE | |
|---|---|---|---|---|
| `RUNWAY=='H'` | 0 | 676.3 | 670.8 | better by 5.5 (one outlier) |
| EHAM HG01 | 176 | 640.2 | 628.8 | better by 11.4 (one outlier) |
| A139 | 176 | 457.1 | 440.3 | better by 16.8 |
| `ADEP==ADES` | 243 | 535.9 | **642.0** | **worse by 106** |
| `FLIGHT_RULE=='V'` | 232 | 602.4 | **660.5** | **worse by 58** |

On the two largest populations the hand rule is decisively worse than the model. On the two smaller
ones it wins only by a margin that vanishes with the single largest residual — and on the helipad
that margin, 5.5 s on 130 rows, is 0.001 s of leaderboard.

---

## 5. Facts: the 527 rows, and whether the test is circular

Reproduced exactly: 1,105 rows with `y < 60`; 1,047 on a triple with n ≥ 20 (all 1,047 also have ARR
taxi-in statistics at their stand); 558 with triple median ≥ 300; **527** also with ARR p05 ≥ 120.

167 of the 527 have an OOF prediction. On them: median y **7**, median prediction **607**, mean
prediction 673 against mean triple median **699**, RMSE 761, bias **+659.7**, MAE 659.8 — i.e. the
residual is essentially the whole label, and the model is predicting the triple's median
(`corr(pred, trip_med) = 0.609`). Their share of total squared error is **0.128 %**.

The circularity check the brief asked for:

* **The predicate is a residual, restated.** `y < 60` while `median(y | triple) ≥ 300` *is*
  "y is far below its group's central value". The model's `ref_med` feature is that group's central
  value, and the model predicts close to it (mean 673 vs 699). So the test selects rows on which a
  group-median predictor has a large negative residual. It is the residual, computed with a weaker
  conditional (group median instead of the fitted model) and with the label on the selecting side.
* **But the rows are not the model's worst rows.** 0 of the 167 are inside the 167 largest
  \|residual\| rows; 2 inside the largest 1,000; 38 inside 5,000; 103 inside 20,000; 134 inside
  50,000. Median \|residual\| on the set is **597** against **100** over all rows — 6× worse than
  typical, but two orders below the tail the score is actually made of (global \|residual\| max
  83,319).
* **Nothing separates them at prediction time.** 247 triples produce at least one of the 527. The
  highest per-triple rate is **0.2105** (LFPG / P73 / 09R, 8 of 38); the median rate across those
  247 triples rounds to below 0.05; the largest single contributor is LSZH / MFG / 28 with 103 of the 527 at a rate
  of 0.099. So the best precision any triple-level rule can reach is **21 %**, and to collect the
  bulk you accept ~10 %. The audit says as much in its own §2.6 ("the individual short row is not
  [identifiable]"); the 527 number does not change that.

The audit's geometry caveat is real and correctly disclosed: `data/external/airports.parquet` holds
`icao, lat, lon` only, so the 300 s / 120 s floor is empirical, taken from the label itself (path a)
and from ARR taxi-in at the same stand (path b). Path b is the honest half — it is a different set of
rows — but it enters only as a ≥ 120 s gate that 1,047 of 1,047 short rows already pass statistics
for, and it eliminates just 31 of 558. **The 527 is, to within 5.6 %, the 558 that path (a) alone
selects.** The independent evidence is doing almost no work.

---

## 6. Inference (separated, as asked)

* *The helipad case is the clean refutation of the whole finding's premise.* The audit's implicit
  model is "the trainer pools helicopters with airliners and predicts 742 s for a 0 s event". The OOF
  says otherwise: median prediction 45 s, median absolute residual 43 s, 121 of 130 rows predicted
  under 300 s. Categorical `RUNWAY_mvt` plus a `ref_med` of −1 was sufficient. A 387-row population
  with a distinctive label and a distinctive column value is exactly what a 127-leaf GBM with
  `min_data_in_leaf = 50` is good at.
* *`FLIGHT_RULE_mvt` being absent from the feature list is the only actionable residue, and it is
  below the noise floor.* Its oracle ceiling is 0.617 s, of which the part not already carried by
  `RUNWAY=='H'`, `STAND=='HG01'` and `ADEP==ADES` is smaller again (78 of 419 ranking VFR rows are
  already helipad rows, 148 are HG01, 217 are local). Adding one categorical column also reshuffles
  `feature_fraction = 0.8` sampling; the repo's own `noise_control` column exists precisely because
  that reshuffle costs on the order of the effect being measured here. This is a decision for the
  §6 rule, and on these numbers the expected value is indistinguishable from zero.
* *Where the score actually is.* 63 % of leaderboard SSE sits on 5,290 unmatched rows at RMSE ~2,300.
  The five populations are 440 of those rows and carry one fourteenth of the average model-B squared
  error. Any hour spent on the short tail is an hour not spent on the other 4,850 model-B rows, which
  is where the 0.73 s ceiling should be compared against.
* *What would reverse this verdict.* (i) A 2026-only regime change at LSZH or EHAM that moves the
  helipad rows out of the triples the encoding covers — the 8 of 12 unseen ranking helipad triples
  are a small live version of this, worth 15 rows. (ii) A demonstration that the model's +22 to +51 s
  positive bias on these populations is systematic rather than one long-tail row per group; that
  needs the other two thirds of the training rows predicted, which needs a run this review was not
  permitted to make. (iii) A `FLIGHT_RULE` ablation measured under the §6 decision rule showing more
  than the 0.6 s oracle bound, which would mean the column carries signal beyond these populations.
