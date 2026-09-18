# Adversarial review of three negative findings (2026-09-18)

Script: `docs/data_audit_2026-09-18/scripts/review-refutations.py`
(`POLARS_MAX_THREADS=1`, lazy scans, no model training; group means, honest leave-one-month-out
group-mean subtraction, Mann-Whitney AUC, rank correlations, correlations against the **model
residual**, threshold sweeps, between-group variance shares against their chance term; ~25 min wall).
Inputs, read-only: `data/raw/training_*.parquet`, `data/work/feat_train.parquet`,
`data/work/feat_rank.parquet`, `experiments/oof_main-48c19bc830.parquet`. Writes nothing.
Notation follows `plan.md` §0. `err` = `y − oof` (positive = under-prediction).

---

## Verdicts

**C1 "arrival-derived features carry no usable signal" — QUALIFIED.**
Deciding observation: under *the audit's own* honesty protocol, the grouping
`airport × decile of time since the previous same-stand arrival` (take-off-anchored) removes
**+0.263 s** of matched RMSE out-of-month, against the audit family's own stated ceiling of +0.024 s
and a no-structure null of +0.024 s — while the one feature the audit actually measured, the previous
arrival's taxi-in, is genuinely dead against the residual (|corr| ≤ 0.025 at every airport, all three
join anchors).

**C2 "the matched side has no recoverable bias left" — REFUTED.**
Deciding observation: the four OOF months are predictions of **three different models**
(`jan`, `jul`, `forward` folds), so leave-one-month-out fits a correction on one model's residuals and
applies it to another's. Within one model, fitting on 2025-11 and applying to 2025-12 —
strictly forward, honest — the grouping `airport × operator × hour` removes **+1.016 s** of matched
RMSE (reverse direction +0.944 s; shuffle null −0.117 ± 0.019 s), against the audit's best of +0.021 s.

**C3 "the fill splits 12,146 harmful / 86,779 benign and the harmful kind does not generalise beyond
LIRF" — QUALIFIED; the second half is refuted.**
Deciding observation: non-LIRF matched rows with `|BLOCK − AOBT_3| > 600 s` that are **not** filled —
39,224 scored rows, RMSE 492.7, **12.65 % of total OOF SSE**, 29 labels above 2 h — are 21× more
numerous and 16× more costly than the 1,871 scored non-LIRF `dfill` rows the audit calls the harmful
kind (RMSE 569.6, 0.81 % of SSE, 1 label above 2 h). The harm tracks `|BLOCK − AOBT_3|`; `fill` adds
almost nothing to it.

---

## C1 — arrival-derived features

### What was checked

The stand-anchored join was rebuilt from raw (2,085,047 departures, 2,082,749 arrivals) under **three
departure anchors** instead of one: `S` = scheduled off-block (the audit's), `A` = actual off-block
`AOBT_3_flt` (non-null on 339,551 of 344,841 ranking rows), `M` = take-off `MVT_TIME_UTC_mvt`
(non-null on all 344,841 ranking rows; the project already ships `arr_land_next_*` windows, so a
take-off anchor is prediction-time legal here). Then: correlations against the **model residual** and
rank correlations, per anchor and per airport; honest group-mean subtraction on arrival-feature
groupings under two protocols; AUC on the LIRF tail instead of a median comparison; and a census of
the arrival-derived columns already in the model.

### Numbers

Reproduction. Coverage under the audit's anchor: 0.9991 any / 0.9407 within 12 h (audit 0.9991 /
0.9398), gap p50 5,215 s (audit 5,215). corr(y, taxi-in), matched, gap ≤ 12 h, on the scored rows:
**0.1076** (audit 0.1115 over all twelve months). The audit's figures reproduce.

The anchor is not innocuous. SCHED- and MVT-anchored joins select a **different arrival** on 12.15 %
of departures overall, and by departure delay `MVT − SCHED`:

| delay band | n | anchors pick a different arrival |
|---|---|---|
| < 15 min | 497,096 | 0.0389 |
| 15–30 min | 855,721 | 0.0938 |
| 30–60 min | 535,402 | 0.1486 |
| 1–2 h | 155,242 | 0.3224 |
| **> 2 h** | 39,706 | **0.6030** |

The audit's anchor-sensitivity check (`arrival-rows.md` §2.0, < 0.1 pp) measured **coverage**, not
which arrival was selected; on the delayed tail the two anchors disagree about the inbound leg on 60 %
of rows. Changing the anchor does *not*, however, produce residual signal:

| group | anchor | n | r(y, taxi-in) | r(err, taxi-in) | ρ(err, taxi-in) | r(err, gap) | ρ(err, gap) |
|---|---|---|---|---|---|---|---|
| matched | S | 619,467 | 0.1076 | −0.0042 | −0.0094 | +0.0121 | +0.0366 |
| matched | A | 620,359 | 0.1145 | −0.0014 | −0.0082 | +0.0082 | +0.0233 |
| matched | M | 620,361 | 0.1225 | −0.0079 | −0.0116 | −0.0014 | −0.0004 |
| unmatched | S | 7,275 | 0.0485 | +0.0019 | +0.0041 | +0.0100 | +0.0042 |
| unmatched | M | 7,407 | 0.0474 | −0.0070 | +0.0122 | −0.0297 | −0.0459 |

Per airport (matched, M anchor) the raw label correlation the audit reported as the strongest evidence
of an airport-dependent effect vanishes entirely against the residual: LEMD r(y, taxi-in) **+0.3306**
→ r(err, taxi-in) **−0.0106**; EDDF +0.1748 → +0.0018; EGLL +0.0776 → −0.0252 (the largest |value|
over the ten airports). Twelve arrival-derived columns are **already model features**
(`arr_land_prev_{5,15,30,60}m`, `arr_land_next_{15,30,60}m`, `arr_inblock_prev_30m`,
`arr_taxi_in_mean_{30,60}m`, `arr_taxi_in_n_{30,60}m`); their |corr(err, ·)| is ≤ 0.0029 and
|ρ| ≤ 0.0112.

Honest group-mean subtraction on arrival-feature groupings. P1 = leave-one-month-out over the four
OOF months (the audit family's protocol); SM = same-model forward transfer, fit on 2025-11 → apply to
2025-12 (see C2). Gains in seconds of matched RMSE, k = 50:

| anchor | grouping | groups | P1 | SM |
|---|---|---|---|---|
| M | apt × gap decile | 119 | **+0.2627** | **+0.6573** |
| S | apt × gap decile | 119 | +0.0669 | +0.6791 |
| A | apt × gap decile | 119 | +0.0016 | +0.4757 |
| S | apt × taxi-in × gap (5×5) | 360 | +0.0276 | +0.5970 |
| M | taxi-in decile | 11 | +0.0215 | +0.0204 |
| M | apt × taxi-in decile | 119 | −0.1538 | +0.1677 |
| M | apt × taxi-in decile × actype-agree | 356 | −0.0121 | +0.3139 |

Controls on the `apt × gap` result: `apt × ms decile` (departure delay, no arrival data at all) gives
**P1 −0.2741**; `apt` alone −0.1559; `apt × hour` −0.3090; `apt × hour × gap 5-tile` −0.1631. Per
held-out month the `apt × gap` gain is **+0.4114 (Jan) / −0.0850 (Jul, the leaky fold) / +0.3889 (Nov)
/ +0.4871 (Dec)**. With `err` permuted, 20 draws give +0.0231 ± 0.0051 s (max +0.0315).

The LIRF tail, with AUC instead of a median (all training rows):

| scope | n | y ≥ 2 h | AUC(taxi-in, S) | AUC(gap, S) | AUC(taxi-in, M) | AUC(gap, M) |
|---|---|---|---|---|---|---|
| LIRF unmatched | 1,488 | 439 | 0.5164 | **0.6344** | 0.5037 | 0.4425 |
| LIRF matched | 159,216 | 41 | 0.6015 | 0.4703 | 0.6014 | **0.6443** |
| all unmatched | 22,470 | 481 | **0.6365** | 0.5567 | 0.6365 | 0.4295 |

On all unmatched rows the previous arrival's taxi-in has AUC 0.6365 for y ≥ 2 h (median 603 s on the
tail vs 490 s on the rest). The model's own arrival-count columns on the scored LIRF unmatched rows
(n 537, 146 with y ≥ 2 h) run AUC 0.3843–0.5100, i.e. informative in the *negative* direction for two
of them (`arr_inblock_prev_30m` 0.3843, `arr_land_prev_60m` 0.3923).

### Was the original instrument capable of detecting what it concluded was absent?

Partly, and the parts matter.
- For **previous-arrival taxi-in** on matched rows: yes. A Pearson correlation against the label
  overstated the feature if anything — the residual correlation is 3–30× smaller than the label
  correlation at every airport, and LEMD's +0.35 is entirely absorbed by the model. A booster cannot
  use a quantity that is orthogonal to the residual in both Pearson and Spearman at every airport, so
  the non-linearity objection does not rescue this feature.
- For **the turnaround gap**: no. The gap was reported only as a pooled correlation (0.029–0.057) and a
  decile table of the *label*. Its honest, out-of-month, group-mean value against the residual is
  +0.263 s (M anchor) — eleven times the +0.024 s bar the sibling audit set for "no recoverable bias",
  and not reproducible by airport, hour or departure delay.
- For **the LIRF tail**: no. The instrument was a comparison of two medians on one feature at one
  airport. A median is a location statistic; AUC(gap) = 0.6344 on the same rows, and on the wider
  unmatched population AUC(taxi-in) = 0.6365, are invisible to it. The audit's own text records
  turnaround-gap medians of 9,152 vs 4,319 s on that tail and still reports the features as "flat".
- For **the single-feature-versus-26-features objection**: the audit's check was single-feature, and
  its conclusion is single-feature. The joint groupings tested here (taxi-in × gap × airport ×
  aircraft-type agreement, up to 360 cells) do beat the single-feature checks under SM
  (+0.31 to +0.60 s) but not under P1, so this review can say the audit's instrument was too narrow;
  it cannot say how a 26-feature booster would score.
- For **the join itself**: the audit's join is stand + time, most-recent-arrival-only, anchored at the
  schedule, with no uniqueness filter — it is a *candidate* link, as the audit states (74.5 % of stand
  links are non-unique within 12 h). Phoenix-Ops' 26 features and OpenAir's gate-hold model are not
  reconstructible from the public descriptions, so no statement here bears on whether the audit's join
  matches theirs.

### Facts vs inference

Facts: every number above. Inference: that a gradient-boosted tree given the turnaround gap as a
feature would realise something like the +0.26 s group-mean value — a group-mean correction applied on
top of a fitted model and a feature handed to the fit are different operations, and no training was
run. Inference: that the 0.60 anchor-disagreement rate on the > 2 h delay band matters for feature
quality; it is shown to matter for *which arrival is selected*, not for any measured gain (the M
anchor beats the S anchor under P1 by +0.196 s on the gap grouping, but the two are equal under SM).

---

## C2 — "no recoverable bias on the matched side"

### What was checked

The frame reproduces exactly: 664,050 matched scored rows, RMSE **216.436** (audit 216.44). Then:
(i) per-month residual structure split by fold leakage; (ii) group-mean subtraction under four honesty
protocols; (iii) the column sweep extended from 22 to **99** numeric columns, with Spearman as well as
Pearson, and a non-parametric decile version of the same sweep; (iv) interaction of each grouping with
the model's own prediction decile; (v) the fold-identity analysis and its controls.

### Numbers

Per OOF month, with the fold that produced it:

| month | n | RMSE | mean err | fold |
|---|---|---|---|---|
| 2025-01 | 152,271 | 206.200 | −6.442 | `jan` — trains on 11 months **including later ones** |
| 2025-07 | 186,775 | 247.259 | −3.059 | `jul` — trains on 11 months **including later ones** |
| 2025-11 | 160,976 | 198.300 | +0.436 | `forward` — trains on 2025-01…10 |
| 2025-12 | 164,028 | 204.858 | −8.705 | `forward` — same model as month 11 |

The audit's own suspicion (I3) is confirmed in the measured direction: the residual is **more**
structured on the clean months, by roughly a factor of two in between-group excess.

| grouping | leaky 1+7 excess | clean 11+12 excess |
|---|---|---|
| airport | 0.00185 | **0.00410** |
| airport × operator | 0.00781 | **0.01684** |
| airport × hour | 0.00403 | **0.01015** |
| airport × stand | 0.00749 | **0.01401** |
| operator | 0.00340 | **0.00692** |

Four honesty protocols, best gain over 17 groupings × two shrinkages (34 rows):

| protocol | best gain | grouping | rows positive |
|---|---|---|---|
| P1 — LOMO over all four OOF months (the audit's) | **+0.0209 s** | flight type, k 50 | 8 / 34 |
| P2 — P1 fits, scored only on the clean months | +0.0949 s | airport × operator, k 50 | 11 / 34 |
| P3 — leave-one-out inside 11+12 (same model) | +0.6433 s | airport × operator, k 50 | 12 / 34 |
| P4 — fit 2025-11 → apply 2025-12 (same model, forward) | **+0.7595 s** | airport × operator, k 50 | 17 / 34 |

P1 reproduces the audit (≤ +0.024 s, negative for most groupings). **The audit's positive column is its
own null**: a single global-mean LOMO correction — one group, no structure whatsoever — gains
**+0.0240 s**, i.e. exactly the audit's reported maximum over 32 groupings. Its whole positive side
is recovery of the −4.38 s global offset.

Within-model versus cross-model transfer (gain in seconds, k = 50):

| grouping | 11→12 | 12→11 | 7→12 | 1→12 | 1+7→11+12 | 11+12→1+7 |
|---|---|---|---|---|---|---|
| airport | +0.163 | +0.087 | −1.033 | +0.102 | −0.451 | −0.541 |
| operator | +0.273 | +0.075 | −0.923 | −0.152 | −0.499 | −0.590 |
| **airport × operator** | **+0.760** | **+0.521** | −1.625 | −0.206 | −0.830 | −1.123 |
| airport × hour | +0.378 | +0.381 | −1.598 | −0.132 | −0.623 | −0.996 |
| airport × stand | +0.271 | +0.129 | −1.470 | −0.692 | −1.096 | −1.074 |
| airport × aircraft type | +0.340 | +0.101 | −1.328 | −0.190 | −0.672 | −0.829 |
| airport × stand × runway | +0.276 | +0.170 | −1.292 | −0.652 | −1.147 | −1.069 |

Every same-model column is positive except airport × runway and market segment; every cross-model
column is negative except market segment. Over the full list at four shrinkages (76 rows), **33 are
positive in both same-model directions**. Best: `airport × operator × hour`, k = 20 →
**+1.0163 s (11→12)** and **+0.9437 s (12→11)**.

Controls on that figure:
- random half of 11+12 → other half (same model, same months): **+0.9245 s** (the no-month-transfer
  reference);
- `err` shuffled inside 11+12, 20 draws: **−0.1170 ± 0.0187 s**, max −0.0729 — the +0.76 s sits ~47 sd
  above the null;
- fit-set size 25 % / 50 % / 100 % of month 11: +0.4237 / +0.5977 / +0.7595 s (a learning curve, not a
  fixed offset);
- trimming to |err| ≤ 600 / 900 / 1,800 s: **+0.5551 / +0.7567 / +0.9417 s** — not a tail artefact;
- arithmetic: a 0.99 % reduction of matched SSE (41.32 % of total OOF SSE) moves total OOF RMSE
  **334.582 → 333.897**, i.e. −0.685 s.

The column sweep. 99 numeric columns exist in `feat_train`; the audit swept 22. Largest |Pearson| with
`err` among the **77 not swept**: 0.0241 (`gust_ms`, non-null on only 17,810 rows); second 0.0175
(`gust_ms_prev1h`). Largest among the audit's 22, reproduced: 0.0140 (`active_runways_60m`). So the
audit's magnitude claim survives the missing columns. Its **instrument** does not: |Spearman| reaches
**0.0538** (`mvt_minus_sched_slog`; also 0.0501 `mvt_minus_lobt_slog`, 0.0497 `mvt_minus_iobt_slog`) —
3.8× the largest |Pearson| — and none of the `*_slog` columns, the 5/10/15/60-minute neighbour counts,
the `*_next_*` windows, `mvt_minus_iobt`, `lobt_minus_eobt1`, `noise_control`,
`congestion_vs_usual_30m`, `dep_arr_ratio_30m`, `window_coverage_60m`, `minute_of_day_utc`,
`hour_local` or any weather column other than `deicing_gate` / `low_vis` were in the 22.
Non-parametric version (decile group means, 180 column × grouping combinations): 76 positive under P1,
8 above +0.024 s, **0 above +0.1 s**, best +0.0418 s (`active_runways_60m`). Under the same-model
protocol with `airport × decile`: `hour_utc` +0.355, `hour_local` +0.342, `dep_tko_prev_60m` +0.279,
`dep_tko_prev_30m` +0.268, `mvt_minus_aobt3` +0.255.

Interacting a grouping with the model's own prediction decile or with the proxy decile makes P1
*worse* in every one of the 15 combinations tried (e.g. airport plain −0.156, × oof decile −0.253,
× proxy decile −0.138; airport × stand plain −0.493, × oof decile −0.546).

### Was the original instrument capable of detecting what it concluded was absent?

No, for the load-bearing part. The audit measured "can a group-mean correction estimated on other OOF
months reduce error on this OOF month". Its months come from three different models, so for two of the
four held-out months every fitted group mean comes entirely from other models' residuals, and for the
other two, three quarters of the fitting rows do. Group-level bias is a property of a (model, training
set) pair; the protocol destroys exactly the quantity it set out to measure. The negative numbers the
audit reports are reproduced here as the cross-model columns (−0.45 to −1.63 s), and they are the
signature of the protocol, not of the residual. Two secondary instrument limits: the positive column
(≤ +0.024 s) is numerically identical to a structure-free global-mean correction, so it carried no
information either way; and the 22-column Pearson sweep is 4× less sensitive than Spearman on the same
frame, though in this case no column's non-parametric decile gain exceeded +0.042 s, so the
"orthogonal residual" conclusion is not overturned by the instrument change — only the group-mean
conclusion is.

### Facts vs inference

Facts: all tables above, including the fold-to-month attribution (read from
`fold-shape-vs-ranking.md` §3.1/§3.3 and consistent with the per-month RMSEs here) and all controls.
Inference: that a within-2025-month-to-month correction transfers across the year boundary to
2026-01 / 2026-07 — nothing here measures that, and `fold-shape-vs-ranking.md` records that no fold
has the production geometry. Inference: that the 11→12 gain is "seasonal drift at the end of the
`forward` fold's training window" rather than a stationary per-group offset; the learning curve and
the both-directions symmetry are consistent with either. Inference: that the −0.685 s total-RMSE
arithmetic is achievable — it is the algebra of the measured matched-SSE reduction, not a scored
prediction.

---

## C3 — the LIRF fill's harmful / benign split

### What was checked

Census reproduction; a twelve-point threshold sweep on `|BLOCK − AOBT_3|` scored by four different
harm measures (count, label ≥ 1 h, label ≥ 2 h, OOF RMSE and SSE share); the density of
`|BLOCK − AOBT_3|` on fill versus non-fill rows to see whether 600 s is a knee; the 2×2 control
`fill × (|BLOCK − AOBT_3| > 600)` that isolates which of the two predicates carries the harm; and the
domain of validity of the test on unmatched rows.

### Numbers

Census reproduces exactly: matched fill 98,925 rows = LIRF 28,369 (6,823 `dfill`) + non-LIRF 70,556
(5,323 `dfill`); 12,146 harmful / 86,779 benign. Unmatched fill 1,124 rows (LIRF 721, non-LIRF 403),
`|BLOCK − AOBT_3|` computable on **0** of them. Total OOF SSE 7.5275e10, matching `matched-error.md`.

Threshold sweep (matched fill rows; `sse_share` is of *total* OOF SSE):

| T (s) | set | n harmful | share of fill | y ≥ 2 h | y ≥ 1 h | n scored | RMSE | SSE share |
|---|---|---|---|---|---|---|---|---|
| 0 | LIRF | 28,211 | 0.9944 | 32 | 945 | 8,901 | 524.1 | 0.0325 |
| 300 | LIRF | 13,203 | 0.4654 | 31 | 944 | 3,903 | 779.6 | 0.0315 |
| **600** | LIRF | **6,823** | 0.2405 | 31 | 944 | 1,955 | 1,079.4 | 0.0303 |
| 1,200 | LIRF | 2,937 | 0.1035 | 30 | 940 | 843 | 1,564.1 | 0.0274 |
| 3,600 | LIRF | 365 | 0.0129 | 28 | 365 | 113 | 2,660.4 | 0.0106 |
| 0 | non-LIRF | 69,992 | 0.9920 | **1** | 48 | 22,366 | 215.5 | 0.0138 |
| 300 | non-LIRF | 18,835 | 0.2670 | **1** | 46 | 6,307 | 357.0 | 0.0107 |
| **600** | non-LIRF | **5,323** | 0.0754 | **1** | 44 | 1,871 | 569.6 | 0.0081 |
| 1,200 | non-LIRF | 891 | 0.0126 | **1** | 42 | 325 | 1,044.4 | 0.0047 |
| 3,600 | non-LIRF | 55 | 0.0008 | **1** | 15 | 20 | 844.1 | 0.0002 |

The split is not robust to the threshold in count terms — the non-LIRF harmful population is 18,835 at
300 s, 5,323 at 600 s and 891 at 1,200 s, a 21-fold range over a plausible band — and the reported
statistic *"exactly one has a label above 2 h"* is **invariant at 1 for every threshold from 0 to
3,600 s**, including T = 0 where "harmful" means all 70,556 non-LIRF fill rows. That statistic is
therefore a property of the non-LIRF fill population as a whole and carries no information about the
threshold or about the harmful/benign distinction.

Nor is 600 s a knee. Density of `|BLOCK − AOBT_3|` in bands
0–60 / 60–120 / 120–300 / 300–600 / 600–900 / 900–1,800 / 1,800–3,600 / 3,600–7,200 / > 7,200 s:

| population | n | densities |
|---|---|---|
| LIRF, fill | 28,369 | 0.127 0.120 0.281 0.229 **0.092 0.089 0.049** 0.013 0.000 |
| LIRF, not fill | 130,847 | 0.182 0.159 0.338 0.208 0.067 0.039 0.006 0.001 0.000 |
| non-LIRF, fill | 70,556 | 0.197 0.183 0.348 0.196 0.051 0.022 **0.004** 0.001 0.000 |
| non-LIRF, not fill | 1,832,805 | 0.200 0.188 0.362 0.186 0.043 0.017 **0.004** 0.001 0.000 |

At LIRF the fill rows carry a genuine excess in the tail bands (0.049 vs 0.006 at 1,800–3,600 s, 8×).
At the other nine airports the fill rows' distribution is **indistinguishable from the non-fill rows'**
at every band (0.004 vs 0.004 at 1,800–3,600 s). So outside LIRF, `dfill` is not selecting a distinct
mechanism; it is cutting the ordinary block-to-pushback offset distribution at an arbitrary point.

The 2×2 control the audit did not run (scored rows; `sse_share` of total OOF SSE):

| set | fill | \|BLOCK−AOBT_3\| > 600 | n scored | RMSE | SSE share | y ≥ 2 h |
|---|---|---|---|---|---|---|
| LIRF | yes | yes | 1,955 | 1,079.4 | 0.0303 | 11 |
| LIRF | yes | no | 6,995 | 154.6 | 0.0022 | 1 |
| LIRF | no | yes | 4,350 | 562.3 | 0.0183 | 5 |
| LIRF | no | no | 36,087 | 242.9 | 0.0283 | 0 |
| non-LIRF | yes | yes | 1,871 | 569.6 | 0.0081 | 1 |
| non-LIRF | yes | no | 20,685 | 144.9 | 0.0058 | 0 |
| **non-LIRF** | **no** | **yes** | **39,224** | **492.7** | **0.1265** | **29** |
| non-LIRF | no | no | 552,883 | 162.5 | 0.1939 | 0 |

`|BLOCK − AOBT_3| > 600 s` raises RMSE from 162.5 to 492.7 on non-LIRF non-fill rows (3.0×) and from
144.9 to 569.6 on non-LIRF fill rows (3.9×). `fill`, holding `|BLOCK − AOBT_3| > 600` fixed, moves it
from 492.7 to 569.6 (1.16×) on 4.6 % as many rows. Per airport the non-LIRF `dfill` rows run
1.49–6.41× their own airport's matched RMSE: EDDF 1,145.5 vs 178.6 (6.41×), LEBL 877.4 vs 196.6
(4.46×), LSZH 743.1 vs 168.9 (4.40×), LEMD 661.8 vs 172.3 (3.84×), EHAM 505.6 vs 162.1 (3.12×),
EGLL 582.1 vs 230.0 (2.53×), LFPG 502.4 vs 246.1 (2.04×), LTFM 361.9 vs 234.3 (1.54×),
EDDM 265.6 vs 178.2 (1.49×). The audit's own §H4 reports the same LEBL 877.4 and LTFM 361.9 figures
and still classes these rows as "mostly harmless".

Domain of validity. `AOBT_3_flt` is non-null on **0 of 22,470** training unmatched rows, so the split
cannot be evaluated there at all, and `BLOCK_TIME_UTC_mvt` is null on **all 344,841** ranking rows, so
even `fill` is unobservable at prediction time. SSE accounting:

| population | n scored | share of total OOF SSE |
|---|---|---|
| classifiable (matched) | 664,050 | 0.4132 |
| **unclassifiable (unmatched)** | **8,378** | **0.5868** |
| all matched `dfill` (the harmful side of the split) | 3,826 | 0.0383 |
| unmatched fill (unclassifiable) | 353 | **0.0608** |
| LIRF unmatched, fill | 233 | 0.0607 |
| **LIRF unmatched, not fill** | **304** | **0.3070** |

The 353 scored unmatched fill rows carry more squared error (6.08 %) than all 3,826 scored matched
`dfill` rows the split classifies (3.83 %), and the single largest population — 304 scored LIRF
unmatched non-filled rows at RMSE 8,718.9 — carries 30.7 % of total OOF SSE and lies outside both
predicates. Non-LIRF unmatched fill rows (403, 120 scored) are the one cell that is genuinely benign:
RMSE 275.8, y p50 727 s, one row above 2 h.

### Was the original instrument capable of detecting what it concluded was absent?

No. "The harmful kind does not generalise beyond LIRF" was decided by a count of labels above 2 h
among non-LIRF `dfill` rows. That count is 1 at every threshold including T = 0, so the instrument
returns the same answer whatever the split is — it cannot distinguish "the harmful mechanism is absent
outside LIRF" from "the harmful mechanism outside LIRF does not produce multi-hour labels". Measured
on OOF error instead, the same rows are 1.5–6.4× their airport's matched RMSE. The instrument also has
no purchase where the error lives: the predicate needs `AOBT_3`, which is null on 100 % of unmatched
rows (58.68 % of OOF SSE), and needs `BLOCK`, which is null on 100 % of ranking rows.

### Facts vs inference

Facts: all tables above, including the exact census reproduction, the sweep, the density bands, the
2×2 control, the per-airport ratios and the SSE accounting. Inference: that the near-identity of the
fill and non-fill `|BLOCK − AOBT_3|` densities outside LIRF means `dfill` selects no distinct
mechanism there — the densities are measured, the mechanism is not observable in these columns.
Inference: that the residual 1.16× RMSE lift of `fill` at fixed `|BLOCK − AOBT_3| > 600` is real
structure rather than composition (those 1,871 rows are concentrated at LEBL, LTFM and LFPG); no
within-airport-within-operator control was run on it.
