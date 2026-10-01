# Adversarial review — is `arr_taxi_in_mean_*` contaminated by the LIRF `BLOCK := SCHED` fill?

**VERDICT: QUALIFIED.** The mechanism is real and the columns are real — but the claim's *harm* is
misattributed. The one observation that decides it: **104 of the 119 LIRF training arrivals with
taxi-in > 2 h lie OUTSIDE the `BLOCK == SCHED` bucket** (and so do the pooled maxima, LSZH 604,797 s and
LFPG 77,756 s), so the proposed fix removes 15 of 119 extreme values while leaving the rest in the
aggregation; and the departures whose `arr_taxi_in_mean_30m` the fix moves by more than 60 s carry
**7.7 % of LIRF OOF squared error and 5.8 % of LIRF-unmatched squared error** (by more than 300 s:
1.7 % / 1.7 %), against **68.7 % / 73.5 %** sitting on rows the fix does not move at all.

Script: `docs/data_audit_2026-09-18/scripts/review-arrtaxi.py` (no training, `POLARS_MAX_THREADS=1`,
runtime ≈ 6 min, lazy scans of `data/raw/training_*.parquet` + `data/raw/ranking.parquet` (5 columns),
`data/work/feat_{train,rank}.parquet` (6 columns), `experiments/oof_main-48c19bc830.parquet`).
Every number below is from that script. `f10` = `|BLOCK_TIME_UTC_mvt − SCHED_TIME_UTC_mvt| ≤ 10 s`
on an ARR row (the audit's flag); `f6` = the same at ±6 s.

---

## 1. What I checked

1. Whether the four `arr_taxi_in*` columns exist in the shipped feature cache and survive
   `visible_numeric_columns` / `cols_a` / `cols_b` under `configs/main.yaml`; whether a gain share is
   recoverable without retraining.
2. Whether the LIRF arrival `BLOCK == SCHED` bucket is a **fill** or a **coincidence of punctuality**,
   by three independent tests: the per-airport local-density spike, a **taxi-in-stratified** spike test,
   and the implied profile of the excess rows.
3. Where the extreme taxi-in values actually sit relative to the bucket.
4. A byte-for-byte recomputation of `arr_taxi_in_mean_{30,60}m` / `_n_{30,60}m` from raw, validated
   against the shipped cache, then recomputed with the filled arrivals dropped — delta distribution per
   airport, on training and on the ranking frame.
5. The delta joined to `oof_main-48c19bc830.parquet` (672,428 scored rows), overall / LIRF / unmatched /
   LIRF-unmatched.
6. What the fix does to `arr_taxi_in_n_*` — new nulls and window occupancy per airport, both frames.

---

## 2. Facts

### 2.1 The feature is real and shipped, in **both** models (§1)

Exact columns in `data/work/feat_train.parquet` and `feat_rank.parquet`:
`arr_taxi_in_mean_30m`, `arr_taxi_in_n_30m`, `arr_taxi_in_mean_60m`, `arr_taxi_in_n_60m`.

| column | `numeric_feature_columns` | `visible_numeric_columns` | `cols_a` (model A) | `cols_b` (model B) |
|---|---|---|---|---|
| `arr_taxi_in_mean_30m` | yes | yes | yes | yes |
| `arr_taxi_in_n_30m` | yes | yes | yes | yes |
| `arr_taxi_in_mean_60m` | yes | yes | yes | yes |
| `arr_taxi_in_n_60m` | yes | yes | yes | yes |

Under `configs/main.yaml`: `exclude_features=()`, `include_candidates=()`, so nothing hides them; they
are not in `CANDIDATE_COLS` (the gated set), and none of them carries one of the six prefixes `cols_b`
strips. Counts: `numeric_feature_columns` 99, `visible_numeric_columns` 89, `cols_a` 89, `cols_b` 74.
Built in `src/prc/features.py:208-214` from `a_apt["TAXITIME_SEC_mvt"]` with in-block time
(`BLOCK_TIME_UTC_mvt`) as the event clock, `lag_sec=0`, anchored on the departure's take-off.
`mask_departures` blanks `BLOCK_TIME_UTC_mvt` / `TAXITIME_SEC_mvt` on **DEP** rows only, so arrival
taxi-in is fully available on both frames — the parity is genuine, not a leak.

**Gain share is NOT recoverable.** No booster is persisted anywhere under the repo, `src/prc/train.py`
never calls `feature_importance`, and no record in `experiments/experiments.jsonl` or
`experiments/pipeline.jsonl` carries an importance / gain field. (The only `feature_importance("gain")`
call in the tree is `src/prc/adversarial.py:114`, which scores the adversarial train-vs-rank classifier,
not the shipped model.) Reporting a gain share would have required a retrain.

### 2.2 The bucket is a FILL, not punctuality — the audit is right on this (§2a–2c)

Reproduced exactly: LIRF ARR f10 = **16,178 / 160,504 (0.1008)** training and **2,615 / 26,839 (0.0974)**
ranking; local density 3,069 / 478, spike **5.27 / 5.47**; every other airport 0.77–1.21.

The alternative explanation (equality = punctuality, taxi-in correct) is **refuted** by the
taxi-in-stratified spike test. Within a taxi-in band, count rows at `|d| ≤ 6 s` against the mean count
in the four neighbouring-minute buckets. Punctuality is a property of the *schedule*, not of the taxi-in
value, so it predicts a band-independent ratio; a fill predicts the ratio to climb, because a filled
row's "taxi-in" is `SCHED − landing`.

| taxi-in band (s) | LIRF train spike | LIRF rank spike | EGLL train | LTFM train | EGLL rank | LTFM rank |
|---|---|---|---|---|---|---|
| 0–300 | **16.4** | **13.4** | 1.06 | 1.19 | 0.96 | 1.35 |
| 300–600 | **2.3** | **2.2** | 1.08 | 1.03 | 1.08 | 1.03 |
| 600–900 | **3.4** | **3.5** | 1.22 | 1.05 | 1.18 | 0.98 |
| 900–1,200 | **11.1** | **13.1** | 1.21 | 1.00 | 1.08 | 1.07 |
| 1,200–1,800 | **117.0** | **76.3** | 0.94 | 1.05 | 1.22 | 1.02 |
| 1,800–3,600 | **648.0** | **1,184.0** | 1.20 | 1.06 | 1.40 | 0.44 |

At the two control airports the ratio is 0.94–1.40 in every band — a flat, punctuality-shaped profile.
At LIRF it is U-shaped and enormous at both ends. The mass at exactly zero is concentrated where the
implied taxi-in is *implausible in both directions*: 91.0 % of LIRF arrivals with taxi-in in
1,800–3,600 s are f6, and 22.3 % of those under 300 s, against 4.6 % in the modal 300–600 s band.

Implied profile of the excess (assuming the genuine punctual rows look like the neighbouring minutes):
training excess **13,109 rows (8.2 % of LIRF arrivals)**, mean taxi-in of the excess **1,116 s** against
604 s for the neighbouring minutes; ranking excess **2,137 rows (8.0 %)**, mean **1,100 s**. Bucket
quantiles (p01/p50/p99) 65 / 830 / 3,238 s against neighbour-minute 246 / 589 / 1,198 s. The fill
therefore also manufactures *implausibly short* taxi-in values, which the audit did not mention.

### 2.3 Where the extreme values actually live — the audit's attribution is wrong (§2d, §2e)

Training arrivals by taxi-in threshold, and how many are in the filled bucket:

| threshold | n (all apts) | of which f10 | n LIRF | LIRF f10 | pooled max | max in bucket? |
|---|---|---|---|---|---|---|
| > 1,800 s | 8,601 | 1,966 (0.229) | 2,238 | 1,869 | 604,797 (LSZH) | no |
| > 3,600 s | 440 | 95 (0.216) | 292 | **95** | 604,797 (LSZH) | no |
| > 7,200 s | 141 | **15 (0.106)** | 119 | **15** | 604,797 (LSZH) | no |
| > 86,400 s | 3 | 1 | 2 | 1 | 604,797 (LSZH) | no |

Ranking: > 7,200 s — 27 rows, 3 filled; LIRF 17, 3 filled; pooled max 50,517 s (LFPG), not filled.

Row by row at the top of the training distribution: LSZH 604,797 (`d = −62`, not filled),
**LIRF 87,244 (`d = 2`, filled)**, LIRF 87,185 (`d = −178`, not filled), LFPG 77,756 (`d = 78,298`, not
filled), LIRF 47,209 (`d = 46,494`, not filled). So the single number the claim names — 87,244 s — *is*
inside the bucket; the population it is offered as an example of is mostly not.

### 2.4 How much it moves the feature (§3)

The recomputation is exact: over all 2,085,047 training and 344,841 ranking departures,
`max |shipped − recomputed|` is 0.0026 s / 0.0002 s (float32 rounding) with perfect null agreement on
all four columns. So the deltas below are the true effect of the proposed fix, not an approximation.

`delta = mean_30m(filled arrivals dropped) − mean_30m(shipped)`, seconds:

| set | apt | n dep | Δ ≠ 0 | \|Δ\| > 10 | > 60 | > 300 | new nulls | Δ p01 | p25 | p50 | p75 | p99 | min | max | shipped mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN | LIRF | 160,704 | 0.583 | 73,488 | 30,695 | 2,854 | 415 | −393.4 | −34.0 | 0.0 | 0.0 | 74.2 | −5,787 | 4,522 | 651 |
| TRAIN | EGLL | 239,546 | 0.351 | 35,772 | 961 | 2 | 4 | −40.8 | 0.0 | 0.0 | 0.0 | 23.4 | −354 | 481 | 530 |
| TRAIN | LTFM | 272,965 | 0.249 | 40,837 | 2,422 | 42 | 101 | −45.8 | 0.0 | 0.0 | 0.0 | 34.8 | −520 | 505 | 836 |
| TRAIN | ALL | 2,085,047 | 0.268 | 255,682 | 37,190 | 2,950 | 796 | −88.7 | 0.0 | 0.0 | 0.0 | 27.7 | −5,787 | 4,522 | 547 |
| RANK | LIRF | 26,899 | 0.584 | 12,801 | 5,967 | 432 | 66 | −383.5 | −45.3 | 0.0 | 0.0 | 77.5 | −1,624 | 775 | 642 |
| RANK | ALL | 344,841 | 0.257 | 41,593 | 6,715 | 432 | 117 | −104.1 | 0.0 | 0.0 | 0.0 | 28.4 | −1,624 | 775 | 547 |

(all four `|Δ|` columns are row counts. The per-airport `Δ ≠ 0` column is a share; the LIRF training
row means 93,647 of 160,704 departures see any change at all.)
The 60 m window behaves the same way: LIRF train Δ p25 = −42.3, p50 = −8.5, 29,190 rows over 60 s,
1,616 over 300 s, min −3,416; LIRF rank Δ p25 = −50.9, p50 = −11.1, 5,944 over 60 s, min −829.

So: at LIRF the median departure's 30 m feature does not move at all, 19.1 % (train) / 22.2 % (rank)
move by more than 60 s, and 1.8 % / 1.6 % by more than 300 s, on a feature whose mean is ≈ 650 s. The
sign is predominantly negative, i.e. the fill inflates the feature, as the audit implies.

**Train and ranking move by almost the same amount** (LIRF Δ p01 −393.4 vs −383.5, p25 −34.0 vs −45.3,
share > 60 s 0.191 vs 0.222). Whatever the fill does to the feature, it does it equally in both frames.

### 2.5 What it costs on the OOF (§4)

672,428 scored rows, overall RMSE 334.58 (reproduces `sched-filled-block.md` §J).

| bucket of \|Δ30m\| | ALL n | ALL RMSE | ALL SSE share | LIRF n | LIRF RMSE | LIRF SSE share | LIRF-unm n | LIRF-unm RMSE | LIRF-unm SSE share |
|---|---|---|---|---|---|---|---|---|---|
| Δ = 0 | 503,595 | 339.1 | 0.7694 | 23,317 | 995.5 | **0.6872** | 248 | 9,055.3 | **0.7347** |
| 0 < \|Δ\| ≤ 10 | 88,666 | **254.1** | 0.0761 | 5,667 | 573.7 | 0.0555 | 64 | 4,290.4 | 0.0426 |
| 10 < \|Δ\| ≤ 60 | 69,190 | 360.0 | 0.1191 | 12,197 | 705.8 | 0.1807 | 136 | 5,790.1 | 0.1647 |
| 60 < \|Δ\| ≤ 300 | 9,947 | 457.8 | 0.0277 | 7,730 | 508.0 | 0.0593 | 75 | 3,910.2 | 0.0414 |
| \|Δ\| > 300 | 1,030 | 753.4 | 0.0078 | 1,013 | 758.9 | 0.0174 | 14 | 5,721.8 | 0.0166 |

Mean residual by bucket (all rows): −4.1, −4.1, −6.7, −13.1, −15.6 s. The ordering is **not monotone**:
the `0 < |Δ| ≤ 10` rows have the *lowest* RMSE of any bucket (254.1 against 339.1 for the untouched
rows), at LIRF (573.7 against 995.5) and on LIRF-unmatched rows (4,290 against 9,055). Where the error
actually lives — the 248 LIRF-unmatched rows with Δ = 0, RMSE 9,055, 73.5 % of that group's SSE — the
feature does not move at all.

### 2.6 Would the fix starve any cell? (§5)

New nulls (`arr_taxi_in_n` was > 0 and becomes 0):

| set | window | LIRF new nulls | LIRF share | ALL new nulls | ALL share | LIRF n p50 before → after | LIRF n mean before → after |
|---|---|---|---|---|---|---|---|
| TRAIN | 30 m | 415 | 0.00258 | 796 | 0.00038 | 11 → 10 | 11.2 → 10.3 |
| TRAIN | 60 m | 116 | 0.00072 | 185 | 0.00009 | 23 → 21 | 22.6 → 20.7 |
| RANK | 30 m | **66** | 0.00245 | 117 | 0.00034 | 11 → 10 | 11.2 → 10.2 |
| RANK | 60 m | **9** | 0.00033 | 43 | 0.00012 | 23 → 21 | 22.5 → 20.5 |

So the starvation failure mode is **small**: 66 ranking rows lose `arr_taxi_in_mean_30m` entirely
(9 lose the 60 m version), and 117 / 43 across all ten airports. The p01 of the window count at LIRF
goes 1 → 1 (30 m) and 3 → 2 (60 m); the mean drops by ~8 %. LightGBM handles the nulls natively, and
`arr_taxi_in_n_*` (also a shipped feature) tells the model when the mean is thin.

The **larger** side effect is that the fix is not LIRF-local. Because every airport has a 0.8–3.1 %
`BLOCK == SCHED` arrival rate that the spike test says is genuine punctuality (ratio 0.94–1.21), the fix
deletes ordinary arrivals from the windows at the other nine airports too: it changes
`arr_taxi_in_mean_30m` on **35.1 % of EGLL**, 32.3 % of LFPG, 31.0 % of EDDM and 24.9 % of LTFM training
departures (and 57.4 % of EGLL rows on the 60 m window), by up to ±481 s at EGLL and ±1,265 s at LEMD.
Those changes are pure noise injection: there is nothing wrong with those arrivals.

---

## 3. What the audit got wrong

1. **Attribution of the extreme values.** `arrival-rows.md` §2.6 and `timestamps.md` F4 present
   "max 87,244 s" under the fill heading, and the claim under review generalises that to "taxi-in values
   up to 87,244 s flow into `arr_taxi_in_mean_*`" *because of the fill*. Only 15 of the 119 LIRF
   training arrivals over 2 h are in the bucket; 104 are not, the pooled maximum (LSZH 604,797 s) is
   not, LFPG 77,756 s is not, and the second-largest LIRF value (87,185 s) is not. `timestamps.md` F4
   does state "119 > 2 h" and "95 of them inside the filled bucket" for the > 1 h cut, so the audit
   itself contains the counter-evidence; the claim built on it does not.
2. **"Contaminated" implies a train/rank asymmetry that is not there.** The fill rate is 10.08 % vs
   9.74 %, the spike 5.27 vs 5.47, the bucket's taxi-in quantiles agree to within 13 s at the median,
   and the feature delta distributions match closely. This is a *noisy, mis-defined* feature computed
   identically on both sides, not leakage and not a distribution shift. The word "contaminated" invites
   a remediation urgency the evidence does not support.
3. **Only the mean is discussed.** The fill corrupts the arrival's `BLOCK_TIME_UTC_mvt` itself, which is
   also the *event clock* for `arr_taxi_in_mean_*` and for `arr_inblock_prev_{5,10,15,30,60}m`
   (`features.py:172`). A filled arrival is placed in the window at its scheduled minute, not its actual
   one, on ~10 % of LIRF arrivals. Dropping the row from the mean does not address the five
   `arr_inblock_prev_*` counts, which the claim does not mention at all.
4. Minor: `arrival-rows.md` §2.6 quotes the filled/non-filled contrast as p50 830 / 589 without the
   local-density correction, which implicitly attributes all 16,178 rows to the fill. Against the
   neighbouring minutes only ≈ 13,109 (81 %) are an excess — the same correction `timestamps.md` F5
   applies to LEBL/LEMD on the departure side.

What the audit got **right**: the four columns are shipped and reach both models; the LIRF arrival
bucket is a fill and not punctuality (my stratified test is stronger evidence than their pooled spike);
the flag is observable in the ranking set; the fill inflates the feature at LIRF by a material amount
on ~19 % of that airport's departures.

---

## 4. Is the fix worth making, and what must it not break?

**Not on this evidence, and not in the form proposed.** Inference, separated from the facts above:

- The rows the fix moves are not the rows that cost us. 92.3 % of LIRF OOF squared error and 94.2 % of
  LIRF-unmatched squared error sits on rows whose `arr_taxi_in_mean_30m` changes by ≤ 60 s, and the
  largest single block (73.5 % of LIRF-unmatched SSE) does not change at all. A feature edit that
  cannot reach the error is very unlikely to move RMSE by more than noise, and this repo has already
  been burned by acting on a CV delta that a 200-row robustness check erased (`configs/main.yaml`
  header, `docs/README`).
- The fix as stated is *global*, and 9 of 10 airports have no fill to remove. Perturbing
  `arr_taxi_in_mean_*` on 25–57 % of EGLL / LFPG / EDDM / LTFM departures to fix LIRF is a bad trade
  unless measured. **If the fix is made, it must be conditioned on the airport (LIRF only), or on the
  taxi-in value being implausible, not on `BLOCK == SCHED` alone.**
- The mechanism-faithful alternative is not exclusion but **flagging**: add
  `arr_fill_share_{30,60}m` (the share of the window's arrivals with `BLOCK == SCHED`) beside the
  existing mean and count. It is computable identically on both frames, it costs no rows, it lets the
  model discount the mean where it is a schedule buffer, and it is a strictly smaller change than
  redefining a shipped feature. Untested here.
- Whatever is done must not break: (a) the exact reproducibility of the cached feature frame —
  `feat_meta.json`'s `feature_code` fingerprint changes, forcing a full rebuild of both frames;
  (b) the 66 ranking rows (30 m) / 9 (60 m) that would lose the feature, plus the 117 / 43 across all
  airports — acceptable, but they must go null, never 0; (c) the train/rank symmetry that currently
  makes this feature honest — any exclusion rule must be applied to both frames by the same code path,
  which `context_features` already guarantees since it runs per source frame.
- A precondition before spending a run on it: the gain share is unknown. If
  `arr_taxi_in_mean_{30,60}m` turns out to carry a negligible share of model A's and model B's gain,
  the whole question is moot. That is one cheap instrumentation change (log
  `booster.feature_importance("gain")` in `_fit`) and it should come first.

---

## 5. Facts and inference, separated

**Facts** (measured by the script): everything in §2 — the four column names and their membership in
`numeric_feature_columns` / `visible_numeric_columns` / `cols_a` / `cols_b`; the absence of any persisted
booster or logged gain; the f6/f10 counts and spike ratios per airport on both frames; the
taxi-in-stratified spike table; the excess-row counts (13,109 / 2,137) and implied means (1,116 / 1,100 s);
the threshold tables for extreme taxi-in and their fill membership, including the row-level listing;
the exact agreement of the recomputation with the shipped cache; all delta distributions per airport on
both frames; the OOF bucket table; the new-null and window-occupancy tables.

**Inference** (not measured):
- That the LIRF `BLOCK == SCHED` bucket is a source-system fill rather than a reporting convention that
  happens to round on-time arrivals to the scheduled minute. The stratified spike test rules out
  *punctuality-as-coincidence*; it does not identify the writing process. Either way the taxi-in value
  on those rows is not a taxi time.
- That the excess rows at the *short* end (taxi-in < 300 s, spike 16.4) are the same fill seen from the
  other side (a late landing with a soon-after scheduled in-block) rather than a second defect.
- That the 3,069 / 478 "local-density" rows are the genuine punctual population. This assumes the
  `BLOCK − SCHED` density is locally flat across ±2 min at LIRF, which is what the control airports show
  but is not proven at LIRF.
- Everything in §4: that the fix will not move RMSE, that flagging beats exclusion, and that the
  non-LIRF perturbation is harmful. **Nothing here was trained.** The SSE-share argument is a bound on
  where the error is, not a measurement of what the model would do with a different feature.
- That `arr_inblock_prev_*` is affected by the same fill. The mechanism is visible in the code
  (`BLOCK_TIME_UTC_mvt` is the event clock) but the magnitude was not computed.
