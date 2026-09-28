# Verification of `review-refutations.md` C2 — the +1.016 s matched-RMSE group correction (2026-09-18)

Script: `docs/data_audit_2026-09-18/scripts/verify-group-correction.py`
(`POLARS_MAX_THREADS=1`, lazy scans, no model training — group means, permutations and arithmetic only;
two independent implementations of the estimator, `np.bincount` and `polars.group_by` + join; ~14 min wall).
Inputs, read-only: `data/work/feat_train.parquet`, `experiments/oof_main-48c19bc830.parquet`. Writes nothing.
`err` = `y − oof`; matched = `mvt_minus_aobt3` non-null; "gain" = seconds of RMSE removed **on the apply block**.

---

## Verdict

**QUALIFIED.**
Deciding observation: the +1.016 s reproduces to four decimals and survives every extension asked for
(31 of 34 ordered same-model block pairs positive, a k-plateau 10–200 rather than one setting, every
shuffle null 20–90 sd away, +0.87 s after dropping the 200 largest residuals) — **but all 34 of those pairs
live inside the two months of one fold, because `forward` is the only fold that predicts more than one
month, so there is no independent replication anywhere in this OOF file; and the transfer that a submission
would actually have to make is cross-model, where all ten measured transfers are negative (−0.76 to −2.92 s).**
The measurement is right. The step from it to "recoverable" is not established by it.

**Honest expected effect on the leaderboard: −0.3 s ± 0.4 s on 284.3584, and only if a same-model
correction is built first; with the fold residuals the project has today the expected effect is
+0.6 to +2.2 s, i.e. a loss.** Arithmetic: dLB/dΔ = 0.755 s per second of matched RMSE, so the
arithmetic ceiling of a fully-transferring +1.016 s is 284.3584 → **283.592 (−0.766 s)**; restricting the
gain to the Jan-2026 half (44.3 % of ranking rows, the only horizon with a measured analogue) gives
Δ_eff = 0.449 s → **284.019 (−0.339 s)**; applying the measured cross-model transfer instead gives
284.93 to 286.56.

---

## 1. Reproduction (facts)

| quantity | this run | review-refutations.md |
|---|---|---|
| matched scored rows / RMSE | 664,050 / 216.436 | 664,050 / 216.436 |
| 2025-11 n / RMSE / mean err | 160,976 / 198.300 / +0.436 | same |
| 2025-12 n / RMSE / mean err | 164,028 / 204.858 / −8.705 | same |
| `ADEP × operator × hour`, k 20, **11→12** | **+1.0163** | +1.0163 |
| same, **12→11** | **+0.9437** | +0.9437 |
| (grouping × k) positive in both directions | **33 of 76** | 33 of 76 |
| one-group global-mean LOMO over all 4 months | **+0.0240** | +0.0240 |
| P1 LOMO, `ADEP` / `ADEP × op` k 0 / `ADEP × op × hour` k 0 | −0.1561 / −0.7975 / −5.023 | matched-error.md: −0.156 / −0.797 / −5.020 |

Both implementations agree to every printed digit. Month 12 matched RMSE 204.8578 → 203.8415.
8,032 of 15,385 groups are seen in the fit month; 3,692 of 164,028 apply rows fall in an unseen group
(correction 0 there).

**Two labelling errors in the report, neither of which changes its direction.** (i) The null quoted beside
the +1.016 figure, "−0.117 ± 0.019", is the null of a *different* configuration (`ADEP × operator`, k 50).
The headline configuration's own pooled-permutation null is **−0.553 ± 0.040** — the separation is larger,
not smaller. (ii) The fit-size curve "0.42 / 0.60 / 0.76" is likewise `ADEP × operator`, k 50; the headline
configuration's curve is **0.43 / 0.69 / 1.02** at 25 / 50 / 100 % of month 11.

## 2. Every ordered adjacent same-model pair (facts)

**First, which pairs are admissible.** `oof_main-48c19bc830.parquet` carries predictions for 2025-01
(`jan`), 2025-07 (`jul`), 2025-11 and 2025-12 (`forward`) — `fold-shape-vs-ranking.md` §3.1/§3.3.
`forward` is the only fold predicting more than one month. **The number of ordered adjacent same-model
*month* pairs in this file is exactly two: 11→12 and 12→11 — the two the report already gives.** The
extension asked for does not exist at month granularity. Replicates can only be manufactured by cutting
the same two months into smaller contiguous blocks, which is what follows; they are not independent of
the headline pair.

Gains in seconds, k = 20, every ordered adjacent pair; `mean` and `n>0` are over all ordered pairs at that
granularity (2 / 6 / 10 / 16 pairs).

| grouping | month 11→12 | month 12→11 | half-month mean (n>0) | 10-day mean (n>0) | week mean (n>0) | all 34 pairs n>0 |
|---|---|---|---|---|---|---|
| **ADEP × operator × hour** | **+1.016** | **+0.944** | +0.690 (6/6) | +0.551 (10/10) | +0.299 (13/16) | **31/34** |
| **ADEP × operator** | +0.735 | +0.440 | +0.555 (6/6) | +0.647 (10/10) | +0.612 (16/16) | **34/34** |
| ADEP × hour | +0.342 | +0.333 | +0.291 (6/6) | +0.246 (8/10) | −0.101 (7/16) | 23/34 |
| ADEP × aircraft type | +0.308 | +0.033 | +0.214 (6/6) | +0.328 (10/10) | +0.293 (15/16) | 33/34 |
| ADEP × stand | +0.095 | −0.141 | −0.175 (0/6) | −0.177 (0/10) | −0.197 (2/16) | 3/34 |
| ADEP | +0.163 | +0.086 | +0.256 (6/6) | +0.380 (10/10) | +0.408 (16/16) | 34/34 |
| operator | +0.256 | +0.026 | +0.218 (6/6) | +0.294 (10/10) | +0.257 (15/16) | 33/34 |
| GLOBAL (one group) | −0.019 | −0.210 | +0.024 (2/6) | +0.051 (3/10) | +0.037 (7/16) | 12/34 |

Individual half-month pairs, `ADEP × operator × hour`: Nov1-15→Nov16-30 +0.756, reverse +0.534;
Nov16-30→Dec1-15 +0.931, reverse +0.955; Dec1-15→Dec16-31 +0.574, reverse +0.391. Worst of the 34 pairs
for this grouping: w9→w8 −0.362.

Blocks are smaller fits, so the comparison must be size-matched. Fitting on a random fraction of month 11
and applying to month 12 (`ADEP × operator × hour`, k 20, 5 draws):

| fit rows | 10,061 | 20,122 | 40,244 | 80,488 | 160,976 |
|---|---|---|---|---|---|
| gain | +0.036 ± 0.138 | +0.199 ± 0.082 | +0.430 ± 0.041 | +0.694 ± 0.040 | +1.016 |

Half-month blocks (~80 k fit rows) return +0.690 against a size-matched +0.694; 10-day blocks (~53 k)
return +0.551 against an interpolated ~+0.52; week blocks (~37 k) return +0.299 against ~+0.41. The
sub-month replicates are therefore consistent with the headline pair, not with a fluke of it — and the
whole curve is a learning curve in fit size, so **the reported number is a property of "one month of
residuals", not a fixed quantity of bias.**

## 3. Shrinkage sweep, with the null for every cell (facts)

`ADEP × operator × hour` (15,385 groups). Null = 20 permutations of `err` **within each block** (preserves
each block's mean and marginal, destroys only the group↔residual association).

| k | 11→12 | 12→11 | null mean ± sd | z | gain − null |
|---|---|---|---|---|---|
| 0 | −3.540 | −3.338 | −4.249 ± 0.325 | 2.2 | +0.71 |
| 5 | +0.473 | +0.439 | −1.351 ± 0.088 | 20.8 | +1.82 |
| 10 | +0.844 | +0.782 | −0.926 ± 0.061 | 28.9 | +1.77 |
| **20** | **+1.016** | **+0.944** | −0.586 ± 0.044 | 36.0 | +1.60 |
| 50 | +0.973 | +0.920 | −0.273 ± 0.026 | 47.2 | +1.25 |
| 100 | +0.830 | +0.801 | −0.149 ± 0.017 | 56.3 | +0.98 |
| 200 | +0.678 | +0.661 | −0.080 ± 0.013 | 58.8 | +0.76 |
| 500 | +0.500 | +0.489 | −0.034 ± 0.009 | 59.7 | +0.53 |
| 1000 | +0.374 | +0.368 | −0.014 ± 0.005 | 76.1 | +0.39 |
| 2000 | +0.255 | +0.254 | −0.006 ± 0.004 | 65.7 | +0.26 |

`ADEP × operator` (2,452 groups): k 0 −0.113 (null −0.962 ± 0.250), 5 +0.571, 10 +0.667, 20 +0.735,
**50 +0.760** (null −0.165 ± 0.018), 100 +0.721, 200 +0.643, 500 +0.521, 1000 +0.439, 2000 +0.370.
`ADEP × hour` (230): flat +0.29 … +0.49, peak at k 500. `ADEP` (10): +0.162 … +0.195, essentially
k-independent, null −0.024 ± 0.005. `operator` (622): peak +0.275 at k 100. `ADEP × stand` (2,529):
negative below k 20, peak +0.334 at k 100. `GLOBAL`: −0.019 at every k.

k = 20 is the top of a **broad plateau**, not a single setting: every k from 10 to 200 exceeds +0.65 s in
both directions, and the sign is stable from k = 5 to k = 2000. k = 0 is catastrophic (−3.54), which is
the usual small-cell variance, not a property of the signal. Net of its own null the statistic is
monotonically *decreasing* in k above 5, i.e. the reported k = 20 is not the most favourable choice.

## 4. Nulls (facts)

All nulls above are within-block permutations (n = 20 draws each, 80 cells computed). Every one is
negative, and the spread is small relative to the gain: sd 0.004–0.33 s, largest at k = 0.
The reviewer's own null design — permuting `err` across the pooled 11+12 rows — gives, for the two
configurations the report names:

| configuration | pooled null | the report says |
|---|---|---|
| `ADEP × operator × hour`, k 20 | **−0.553 ± 0.040** (max −0.485) | −0.117 ± 0.019 (mis-attributed) |
| `ADEP × operator`, k 50 | −0.117 ± 0.019 (max −0.073) | −0.117 ± 0.019 ✓ |

Separation for the headline cell is 36 sd (within-block null) or 39 sd (pooled null). No null draw of any
configuration came within 0.5 s of its observed gain.

## 5. Where the gain lives — trimming (facts)

`ADEP × operator × hour`, k 20. The kept-row mask is fixed by the **uncorrected** |err|, so the row set
does not depend on the correction.

| trim | rows kept (of 164,028) | RMSE before | gain 11→12 | gain 12→11 |
|---|---|---|---|---|
| untrimmed | 164,028 | 204.86 | +1.016 | +0.944 |
| drop top 200 \|err\| (the team's rule) | 163,828 | 189.58 | **+0.867** | +0.791 |
| drop top 0.1 % \|err\| | 163,864 | 190.87 | +0.919 | +0.780 |
| \|err\| ≤ 1,800 | 163,908 | 192.86 | +0.942 | +0.790 |
| \|err\| ≤ 900 | 163,291 | 178.92 | +0.757 | +0.685 |
| \|err\| ≤ 600 | 161,770 | 165.80 | +0.555 | +0.624 |
| **\|err\| ≤ 300** | 149,161 | 127.01 | **−0.106** | **+0.025** |
| ONLY \|err\| > 600 | 2,258 | 1,038.80 | +8.219 | +6.687 |

`ADEP × operator`, k 50, 11→12: +0.760 / +0.632 (drop 200) / +0.651 / +0.570 / +0.521 / **+0.338**
(≤ 300). `ADEP × hour`, k 20, 11→12: +0.342 / +0.369 / +0.380 / +0.329 / +0.212 / **−0.267** (≤ 300).

Trim-independent view — which |err| bands the correction actually moves, as a share of the net SSE it
removes from month 12 (`ADEP × operator × hour`, k 20, 11→12):

| \|err\| band | 0–100 | 100–200 | 200–400 | 400–600 | 600–900 | 900–1,800 | 1,800–3,600 | > 3,600 |
|---|---|---|---|---|---|---|---|---|
| share of block rows | 0.507 | 0.283 | 0.165 | 0.030 | 0.0093 | 0.0038 | 0.0007 | 0.0001 |
| share of block SSE | 0.037 | 0.144 | 0.300 | 0.165 | 0.113 | 0.126 | 0.086 | 0.029 |
| **share of net SSE removed** | **−0.598** | +0.134 | **+0.641** | +0.259 | +0.211 | +0.224 | +0.081 | +0.047 |

The 200-row rule the team applies costs 15 % of the gain. But the honest statement about the body is
stronger than the report's "not a tail artefact": **the correction makes the smallest-residual half of the
rows worse** (−0.60 of the net), is paid for by the 200–600 s band (+0.90) and the > 600 s tail (+0.56),
and disappears or reverses once |err| ≤ 300. `ADEP × operator` at k 50 is the only tested configuration
whose gain survives the ≤ 300 trim (+0.338).

## 6. What the gain is made of (facts)

Global-offset versus structure, `ADEP × operator × hour`, k 20:

| direction | full correction | global-mean-only (1 group) | correction re-centred on the apply block | correction sd |
|---|---|---|---|---|
| 11→12 | +1.016 | **−0.019** | +1.033 | 26.79 |
| 12→11 | +0.944 | **−0.210** | +1.064 | 26.65 |

The global offset transfers *negatively* here (month means +0.436 vs −8.705), and re-centring the
correction makes it slightly better. The same-model gain is therefore entirely relative group structure —
the opposite failure mode from the audit's P1 column, where the whole +0.024 s is the global offset.

Concentration, share of the SSE removed from month 12:

| ADEP | LIRF | LTFM | LEMD | EGLL | LSZH | EDDM | LEBL | EHAM | LFPG | EDDF |
|---|---|---|---|---|---|---|---|---|---|---|
| share of removed SSE | **0.476** | **0.293** | **0.126** | 0.061 | 0.042 | 0.033 | 0.029 | −0.013 | −0.023 | −0.023 |
| gain on own rows (s) | +4.41 | +2.05 | +1.45 | +0.47 | +0.86 | +0.45 | +0.41 | −0.16 | −0.18 | −0.27 |

Leave-one-airport-out gain on the rest of month 12: excluding LIRF **+0.605**, LTFM +0.841, LEMD +0.977,
LSZH +1.026, EDDM +1.062, LEBL +1.065, EGLL +1.101, EHAM +1.133, EDDF +1.142, LFPG +1.201.
Three airports carry 89.5 % of it; three of the ten are harmed by it.

Cross-model transfer, same grouping and k, all twelve ordered month pairs:

| | apply 2025-01 | 2025-07 | 2025-11 | 2025-12 |
|---|---|---|---|---|
| fit 2025-01 (`jan`) | — | −0.982 | −1.346 | −0.756 |
| fit 2025-07 (`jul`) | −2.546 | — | −2.856 | −2.922 |
| fit 2025-11 (`forward`) | −1.623 | −1.553 | — | **+1.016** |
| fit 2025-12 (`forward`) | −1.240 | −1.619 | **+0.944** | — |

Ten of ten cross-model transfers negative; two of two same-model transfers positive. This reproduces the
report's mechanism claim exactly.

Time locality, week blocks, gain averaged over every ordered pair at each lag (k 20):

| lag (weeks) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | lag-agnostic control |
|---|---|---|---|---|---|---|---|---|---|
| ADEP × operator × hour | +0.299 | +0.225 | −0.115 | −0.145 | −0.046 | +0.027 | −0.497 | −0.116 | **+0.466 ± 0.153** |
| ADEP × operator | +0.612 | +0.408 | +0.042 | −0.034 | −0.009 | +0.103 | −0.263 | −0.183 | **+0.467 ± 0.159** |
| ADEP | +0.408 | +0.256 | −0.033 | −0.071 | −0.185 | −0.114 | −0.073 | −0.119 | **+0.307 ± 0.053** |
| n pairs | 16 | 14 | 12 | 10 | 8 | 6 | 4 | 2 | 10 |

The control is a random subset of the whole forward period of one week's size applied to a held-out week —
same fit size, no fixed lag. A single contiguous block loses to it beyond ~2 weeks and is *worse* than it
even at lag 1. Per-lag sd is 0.15–0.49 across pairs, so the decay is suggestive, not established.

## 7. Arithmetic on the competition metric (facts)

N = 344,841 = matched 339,551 + unmatched 5,290. LB 284.3584 → total SSE 2.7884e10. Matched at 218.0 →
matched SSE 1.6137e10, hence **implied unmatched RMSE 1,490.2** (the caller's ~1,490 checks out).
dLB/dΔ = **0.7549** s of leaderboard RMSE per second of matched RMSE, linear to 4 dp over Δ ∈ [0, 2].

| Δ matched RMSE | 0.1 | 0.3 | 0.449 | 0.5 | 0.75 | 1.0 | 1.016 | 1.5 |
|---|---|---|---|---|---|---|---|---|
| LB RMSE | 284.283 | 284.132 | 284.019 | 283.981 | 283.793 | 283.604 | **283.592** | 283.228 |
| LB gain (s) | 0.076 | 0.226 | 0.339 | 0.377 | 0.566 | 0.754 | **0.766** | 1.131 |

Scenarios, stated as arithmetic on the measured numbers:

| scenario | matched Δ | LB | LB change |
|---|---|---|---|
| the full +1.016 transfers to all 339,551 matched ranking rows | +1.016 | 283.592 | **−0.766** |
| transfers to the Jan-2026 half only (44.3 % of rows; the only horizon with a 1-month analogue) | +0.449 | 284.019 | **−0.339** |
| LIRF's share does not reproduce in 2026 (leave-LIRF-out figure), Jan half only | +0.267 | 284.157 | −0.201 |
| fitted on fold OOF residuals, i.e. cross-model (best of the ten) | −0.756 | 284.929 | **+0.571** |
| fitted on fold OOF residuals, cross-model (worst of the ten) | −2.922 | 286.563 | **+2.204** |

## 8. What would have to be true for this to fail

1. **A same-model correction must be constructible at submission time.** It is not, today. The submission
   model trains on all twelve months of 2025; the only residuals available for it on unseen rows come from
   the CV folds, which are different models. Every cross-model transfer measured is negative. Making the
   correction same-model costs a held-out month of training data, and that cost is not measured anywhere.
2. **The 7-month horizon must behave like the 1-month one.** 55.7 % of ranking rows are 2026-07. No fold
   has that geometry (`fold-shape-vs-ranking.md` §3.2), no measurement here spans more than 61 days, and
   the within-window lag curve is at or below zero by three weeks. If the Jul half contributes nothing the
   effect halves.
3. **The year boundary must not matter.** Nothing in this OOF file crosses one.
4. **LIRF, LTFM and LEMD must keep their 2025-11/12 offsets into 2026.** They carry 89.5 % of the gain,
   and LIRF alone 47.6 %; LIRF is the airport the sibling audits already flag for label pathology
   (`fill-pattern.md`, `review-refutations.md` C3). Three airports are harmed by the correction now.
5. **The body must not be sacrificed for the tail.** The correction currently worsens the 50.7 % of rows
   with |err| < 100 s and is repaid by the 19.5 % of rows above 200 s. Under the ≤ 300 s trim the headline
   configuration is negative; only `ADEP × operator` at k 50 survives it.
6. **The estimate must not be a learning curve.** It is one: 0.04 / 0.20 / 0.43 / 0.69 / 1.02 s as fit rows
   go 10 k → 161 k, with no sign of a plateau at the top. The +1.016 s is what one month of residuals buys,
   and would be larger with more same-model residuals — which is another reason it is not a fixed "amount
   of bias present".

## 9. Facts versus inference

**Facts.** Every number in §1–§7: the exact reproduction of +1.0163 / +0.9437 / 33-of-76 / +0.0240; the
month-to-fold attribution read from `fold-shape-vs-ranking.md` §3.1/§3.3 and consistent with the per-month
RMSEs; all 34 ordered same-model block-pair gains; the fit-size curve; the k sweep and all 80 permutation
nulls; the trim table and SSE-band decomposition; the per-airport and leave-one-airport-out figures; the
twelve cross-model transfers; the lag table; the metric arithmetic and the implied unmatched RMSE 1,490.2.

**Inference.** That the sub-month replicates constitute independent confirmation — they do not; they are
re-cuts of the same 325,004 rows from the same model, and their agreement shows internal consistency only.
That the lag decay implies the effect would not survive a 7-month horizon — the per-lag spread is large and
the pairs share blocks; the decay is directional evidence, not a measurement of a 7-month transfer. That
the cross-model negatives predict what a purpose-built production correction would score — they measure the
protocol available today, not a design in which the correction and the predictions come from one model.
That the −0.3 ± 0.4 s expectation is a forecast — it is the arithmetic of §7 under assumptions 1–4 of §8,
none of which is testable with this OOF file. That the LIRF concentration means the effect is a label
artefact — the concentration is measured, the mechanism is not.
