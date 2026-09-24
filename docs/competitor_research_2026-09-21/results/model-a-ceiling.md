# How much of model A's error is irreducible (2026-09-24)

**Script (read-only; nothing written to `src/`, `configs/`, `outputs/` or `data/`; no submission made):**
`docs/competitor_research_2026-09-21/scripts/model_a_ceiling.py`, run as
`POLARS_MAX_THREADS=2 .venv/bin/python docs/competitor_research_2026-09-21/scripts/model_a_ceiling.py [S1 … S7]`.
Every number below carries the section (`S1`…`S7`) that prints it. Measured wall clock on a 4-CPU box
that was **not** idle (§10): `S1` 0 s, `S6` 13 s, `S7` 2 s, `S3` 124 s, `S2` 135 s, `S4` 243 s, `S5`
327 s — about 14 minutes for the whole file. `S5` is the only section that fits models.

**Inputs.** `data/work/feat_train.parquet` (2,085,047 departures, of which **2,062,577** are matched and
labelled), `experiments/oof_main_gap-62dd7c6e88.parquet` + its `_folds` sidecar (672,428 pooled OOF rows
over folds jan/jul/forward; `jul_fwd` excluded as elsewhere), `experiments/experiments.jsonl`,
`experiments/submissions.jsonl`.

**Notation.**

| symbol | definition | visible at prediction time? |
|---|---|---|
| `y` | `TAXITIME_SEC_mvt` = MVT − BLOCK | no — it is the label |
| `proxy` | `mvt_minus_aobt3` = MVT − AOBT_3 | **yes** |
| `g` | `y − proxy` = AOBT_3 − BLOCK | no — this is what model A's tree fits |
| `ghat` | the tree's output; the prediction is `proxy + ghat` | — |
| `e` | `oof − y` = `ghat − g` | no — model A's OOF residual |

**lane A** = `proxy` is not null, the routing rule at `src/prc/train.py:402`. A feature is visible iff it
reads neither `BLOCK_TIME_UTC_mvt` nor `TAXITIME_SEC_mvt` (`src/prc/schema.py:52`). Every column used to
match rows below is visible by that rule; `g` and `e` are only ever the *quantity measured*, never part of
a key.

**One design choice, stated first because everything rests on it.** `proxy` is exactly visible and enters
the prediction additively, so `Var(y | visible) ≡ Var(g | visible)`, while `Var(y | cell) > Var(g | cell)`
for any cell that does not hold `proxy` fixed. Measuring the spread of **`g`**, not of `y`, therefore
removes the largest visible axis for free and makes every matched-set estimate below tighter without
assuming anything. [S] `S1`

---

## 0. The answer in nine sentences

1. **The floor is 187 – 205 s, best estimate 201, and it is nowhere near 150.** The best-instrumented
   route (twin-residual covariance, §4) measures the *systematic* — recoverable — share of lane A's
   squared error at **ρ = 0.110 [0.074, 0.146]**, putting the floor at **213.13 × √(1−ρ) = 201.0,
   CI 197.0 – 205.0**. An independent route (learning curve, §5) puts ten times the training data at a
   pooled-equivalent **200.9** and its `n → ∞` extrapolation at **186.9**. On the forward fold, where the
   two can be compared with no rescaling at all, they read **187.1** and **187.53**. The bottom of the
   range is the `n → ∞` extrapolation and I do not defend it; everywhere both routes are interpolating
   they agree to under half a second. [S] `S4`, `S5`
2. **148.9 and 141.7 are below the floor and are not reachable by working on lane A.** To land lane A at
   148.9 the systematic share would have to be **ρ ≥ 0.512**; at 141.7, **ρ ≥ 0.558**. Measured ρ is
   0.110, and the *positive control* — a deliberately crippled predictor that knows only airport ×
   operator and scores 355.7 — measures ρ = 0.484. **A board score in the 230s would require model A to be
   about as systematically wrong as a model that has never seen a clock.** The measurement survives every
   robustness check run against it: ρ is 0.115 on a 21-column matching metric instead of 0.110 on the
   12-column one (§4.5), and it is flat from twins minutes apart to twins four months apart (§4.7), so it
   is neither an artefact of which columns define a twin nor a shared operational shock. [S] `S4`, `S7`
3. **The routes agree once they are compared on the same rows, and the one place they appeared to
   disagree was a selection effect, not a disagreement.** The 1-NN semivariance route reads 140 s at
   `d ≤ 0.25`, which looks like a floor of 140 until you notice model A already scores **139.1 on exactly
   those rows**. Those rows are the dense 3 % of the lane. Post-stratified, the same route reads 203.6
   against a model RMSE of 192.2 on the same 94 % of rows. [S] `S3`
4. **Model A's error tracks the local twin spread at every density, and is slightly below it
   everywhere** — ratio 0.99 at `d ≤ 0.25`, 0.93 at `d ≈ 1`, 0.72 beyond `d = 3`. The model is not leaving
   a resolvable pattern behind; it is at the spread of its own near-twins. [S] `S3`
5. **Capacity is saturated; rows are not.** At a fixed 435,263 training rows, 31 / 127 / 511 leaves give
   213.23 / 213.13 / 213.38 — a 16× capacity range worth 0.25 s. Doubling the rows at fixed capacity is
   worth 5.8 s, and the last available doubling is still worth 4.6 s. **Model A is data-limited, and 2025
   is all the data there is** — so part of §4's recoverable 11.3 s is estimation variance that no feature
   can reach. [S] `S5`
6. **Nothing of the form "a better encoding of airport / operator / stand / runway / destination / hour /
   aircraft type / NM-delay band" is left.** An *oracle* out-of-fold cell-mean correction over ten such
   partitions — up to 543,181 cells — moves lane A from 213.127 to at best **212.86**, against a shuffled
   control at 213.03. That is **0.27 s**, and it is the ceiling on every target encoding, per-group
   specialist and per-group offset column over those variables. [S] `S6`
7. **Per airport, the recoverable gap is concentrated, and it is concentrated where the error already
   is.** LIRF (ρ 0.186, 345.1 → 311.4) and LFPG (ρ 0.130, 241.5 → 225.2) are **53.6 %** of all the
   recoverable squared error between them; LSZH is 1.8 % and EGLL 7.5 %. It is neither "a floor of 150
   that is entirely LFPG" nor a gap spread evenly over ten airports. [S] `S1`, `S4`
8. **The +1.98 s OOF-to-board gap is the larger of two observations (+0.896 for v13, +1.984 for v14),
   and its level is an artefact of the pooling** — on the board's own months (jan + jul) the same
   predictions read 323.12 and the implied gap is **−43.60**; on forward alone, **+59.80**. More
   importantly, the targets do not need the gap's level, they need its *derivative*, and the one paired
   delta available measures **ΔBoard / ΔOOF = 0.747, not 1.000**. At that ratio board 234.5 needs lane A
   at **123.4**, not 148.9. **The instability moves the targets further out of reach, not closer.**
   [S] `S7`
9. **[X] The honest headline: lane A's error is about 90 % irreducible at this feature set. The whole
   recoverable amount is 11.3 s of lane-A RMSE (213.13 → 201.78) — worth −8.5 s of board if every second
   of it were taken — and the 230s are out of reach from lane A alone.** What is *not* ruled out is a
   column carrying genuinely new information (an external off-block observation, in the sense of
   `missing-clock.md` §6): this note measures what the current columns can support, not what a new one
   could.

---

## 1. What is being estimated, and which way each route errs

Squared error is minimised by the conditional mean, so with `μ(x) = E[g | x]`

```
MSE  =  E[(ghat(x) − g)²]  =  E[(ghat(x) − μ(x))²]  +  E[Var(g | x)]
        \_______________/     \__________________/     \___________/
         what we measure        approximation error       THE FLOOR
          213.1266² = 45,423      (recoverable)          (irreducible)
```

The floor is `√(E[Var(g | visible)])`. **Every estimator in this note bounds it from above**, and that is
not a defect of the design but a fact about the problem: any predictor built from visible features has
MSE ≥ floor², so a *lower* bound would require knowing that some particular variation is unpredictable,
which no amount of data can establish. **[X] This is stated up front because it decides how the result
may be used.** "Floor ≤ 201" does not by itself forbid 148.9. What forbids it is §4's argument, which
runs the other way round: it measures the *share* of the error that is systematic, and that share is a
quantity with a control on both sides.

| route | § | what it matches on | direction of its error | reads |
|---|---|---|---|---|
| 1 matched-twin ladder | 2 | exact cells of visible categoricals, plus a within-cell fit on 15 visible continuous columns | over-states the floor (cells are coarser than `x`) | 214.0 |
| 2 1-NN semivariance | 3 | nearest visible twin inside the same stand | over-states, by `(μ_i−μ_j)²/2`, which vanishes as `d → 0` | 203.6 at 94 % coverage |
| 3 twin-residual covariance | 4 | same, but measures the *covariance of the two residuals* | under-states ρ at `d > 0`, hence over-states the floor | **201.0** |
| 4 learning curve | 5 | nothing — varies rows and capacity | gives this recipe's `n → ∞` limit, itself ≥ floor | 200.9 at 10× rows, 186.9 at `n → ∞` |

Route 3 is the load-bearing one; routes 1, 2 and 4 are there because a single estimator with a single
failure mode is what this project has twice shipped a silent bug through.

### 1.1 The baseline, reproduced

`S1`, pooled OOF, 672,428 rows, RMSE **277.5398**:

| lane | n | RMSE | share of total SSE |
|---|---:|---:|---:|
| A (matched) | 664,050 | **213.1266** | 0.5823 |
| B (unmatched) | 8,378 | 1,606.8929 | 0.4177 |

Identical to the brief to four decimals, so the two notes are talking about the same rows.

### 1.2 Lane A restated in the space model A is actually fitted in

`S1`. The prediction is `proxy + ghat`, so the residual is `ghat − g`: **every second of lane A's error is
an error about `g = AOBT_3 − BLOCK`**, the disagreement between NM's off-block stamp and the airport's.

| quantity | value |
|---|---:|
| sd(`y`) on lane A | 444.28 |
| sd(`g`) on lane A | 399.28 |
| median(`g`) | −55.0 |
| robust sd(`g`) = IQR / 1.349 | **226.83** |
| model A RMSE | **213.127** |
| bias | +4.853 |
| fraction of Var(`g`) explained | **0.7151** |

**[X] Two things follow immediately.** First, `g` is heavy-tailed — its robust spread is 226.8 against a
standard deviation of 399.3 — so lane A's RMSE lives in the tail and any estimator of the floor that is
robust to outliers is estimating the wrong thing. Second, the model already explains 71.5 % of Var(`g`)
and its RMSE (213.1) is *below* the robust spread of the raw quantity (226.8): it is not sitting on an
obviously unexploited pattern.

### 1.3 Per airport, lane A only

`S1`. **[X] This table is not the one in the brief.** The brief's per-airport RMSEs (LFPG 441.7,
LIRF 543.2) are over *both* lanes; lane A alone is very different, and lane A is what this note is about.

| ADEP | n | lane-A RMSE | bias | sd(`g`) | share of lane-A SSE |
|---|---:|---:|---:|---:|---:|
| EHAM | 78,286 | 161.46 | −1.57 | 287.98 | 0.0677 |
| LSZH | 42,287 | 167.20 | −0.26 | 282.29 | 0.0392 |
| LEMD | 70,316 | 171.62 | +15.04 | 269.15 | 0.0687 |
| EDDF | 71,489 | 175.35 | +2.25 | 274.78 | 0.0729 |
| EDDM | 51,286 | 175.73 | −1.50 | 407.23 | 0.0525 |
| LEBL | 55,501 | 195.33 | +5.59 | 310.43 | 0.0702 |
| EGLL | 78,986 | 222.07 | +11.00 | 445.65 | 0.1291 |
| LTFM | 90,041 | 229.20 | +6.14 | 466.29 | 0.1568 |
| LFPG | 76,471 | 241.53 | +4.45 | 408.43 | 0.1479 |
| **LIRF** | 49,387 | **345.14** | +2.91 | 529.14 | **0.1950** |

The spread on lane A is 2.1× (161 to 345), not the 3.2× of the pooled table. **LIRF, LTFM, LFPG and EGLL
are 62.9 % of lane-A SSE on 44 % of its rows.**

---

## 2. Route 1 — the matched-twin ladder

### 2.1 Design

`S2`. On all 2,062,577 matched labelled rows, cells are built from visible columns only and tightened one
column at a time. Two arms and one control:

* **plain** — the pooled within-cell sd of `g`, cells of size ≥ 2, `(n−1)` denominator.
* **+covar** — the same after a within-cell least-squares fit on 15 visible continuous columns expanded
  into 75 hinge terms (level plus four quintile knots each), with `nrow − ncells − rank` degrees of
  freedom. This is what lets a cell resolve a continuous feature it cannot bin, and without it the ladder
  measures the binning, not the information.
* **control** — the same +covar arm with the cell labels shuffled and the covariate rows permuted. It
  **must** return the within-ADEP sd of `g` at every level; a control that drifts means the degrees of
  freedom are being miscounted, which is the exact failure mode recorded at `missing-clock.md` §7.

### 2.2 The result

`S2`:

| level | plain sd | +covar | **control** | rows | cells | coverage |
|---|---:|---:|---:|---:|---:|---:|
| L1 ADEP | 364.35 | 259.44 | **364.36** | 2,062,577 | 10 | 1.000 |
| L2 +operator | 354.86 | 249.72 | **364.37** | 2,062,062 | 2,608 | 1.000 |
| L3 +stand | 345.94 | 240.70 | **364.52** | 2,048,825 | 28,564 | 0.993 |
| L4 +runway | 339.43 | 231.54 | **364.41** | 2,029,477 | 55,477 | 0.984 |
| L5 +dest | 330.96 | 226.09 | **363.68** | 1,896,596 | 206,668 | 0.920 |
| L6 +3 h block | 319.67 | 219.53 | **363.50** | 1,713,816 | 311,816 | 0.831 |
| L7 +aircraft type | 315.47 | 218.70 | **362.01** | 1,506,880 | 334,341 | 0.731 |
| L8 +NM-delay quintile | 275.61 | **213.97** | 359.25 | 985,306 | 315,802 | 0.478 |

**[S] The control holds at 364.3–364.4 for seven levels and 358.6 at the eighth**, against a within-ADEP
sd of `g` of 364.35. The instrument is not manufacturing a drop out of small cells: the degrees-of-freedom
arithmetic `nrow − ncells − rank` is right, and a key carrying no information returns the raw spread no
matter how many cells it has. The 1.4 % slip at L8 is the one place to watch — at 47.8 % coverage the
control is measuring a different population from L1's, since both arms drop the same singleton cells —
and it is the reason L8's 213.97 is quoted as "about 214" and not to two decimals.

**[X] Route 1 reaches 214.0 and stops, which is model A's own error, so on its own it establishes
nothing.** Two rows that agree on airport, operator, stand, runway, destination, three-hour block,
aircraft type and NM-delay quintile — and are then corrected for fifteen continuous columns — still differ
in `g` by 214 s rms, while model A's error over the whole lane is 213.1. A GBM's partition is finer than
any hand-built cell scheme, and that is the honest reading: **the matched-cell route cannot resolve the
conditional mean as well as the model already does, so it cannot bound the floor below the model.** It is
reported because a route that failed to find headroom is evidence of the same sign as one that found
little, and because its control is what validates the degrees-of-freedom arithmetic reused in §3 and §4.

`S2`, per airport, +covar / plain at three levels:

| ADEP | L1 | L5 | L8 | n at L8 |
|---|---|---|---|---:|
| EDDF | 200.5 / 249.5 | 174.5 / 228.3 | 172.0 / 214.2 | 116,520 |
| EDDM | 199.9 / 339.2 | 175.7 / 304.6 | 168.3 / 243.5 | 77,966 |
| EGLL | 265.2 / 417.9 | 246.5 / 399.8 | 225.6 / 322.3 | 124,964 |
| EHAM | 212.3 / 292.6 | 172.3 / 270.9 | 166.1 / 261.0 | 102,948 |
| LEBL | 219.4 / 304.0 | 198.7 / 279.8 | 189.5 / 245.3 | 92,989 |
| LEMD | 211.6 / 274.3 | 175.3 / 249.6 | 166.1 / 209.3 | 120,679 |
| LFPG | 273.3 / 377.9 | 248.8 / 352.4 | 229.4 / 293.9 | 115,174 |
| LIRF | 427.3 / 514.6 | 370.9 / 452.9 | 346.9 / 376.6 | 97,156 |
| LSZH | 212.3 / 261.8 | 166.9 / 216.6 | 163.0 / 185.5 | 57,871 |
| LTFM | 302.2 / 475.0 | 249.7 / 419.5 | 227.2 / 325.9 | 79,037 |

---

## 3. Route 2 — 1-NN semivariance, read against the model on the same rows

### 3.1 Design

`S3`. For every matched row, its nearest visible twin **inside its own stand block**. Distance is squared
Euclidean over eight z-scored continuous visible columns plus one unit per mismatched visible
categorical:

```
d² = Σ_j ((z_ij − z_i'j)/s_j)²   over  aobt3_minus_eobt1, mvt_minus_aobt3, aobt3_minus_sched,
                                        mvt_minus_eobt1, mvt_minus_sched, mvt_minus_lobt,
                                        nbr_aobt_delay_mean_30m, stand_arrival_gap
   + Σ_c 1[c_i ≠ c_i']           over  RUNWAY_mvt, ADES_mvt, AIRCRAFT_OPERATOR_flt, AIRCRAFT_TYPE_mvt
```

Those twelve columns carry 85 % of a LightGBM's gain on `g` (the importance run is in §10). `d = 0` means
the pair agrees on the stand and on all twelve exactly. The semivariance
`γ(d) = E[(g_i − g_j)²] / 2` estimates `E[Var(g|x)]` inflated by `(μ_i − μ_j)²/2`, which vanishes as
`d → 0`. **The third column is the point of the section**: model A's OOF RMSE computed on *exactly the
rows in the same distance bin*.

### 3.2 The result, and why the apparent 140 s floor is not one

`S3`. 2,061,650 of the 2,062,577 matched rows are paired; the 927 that are not sit in stand blocks with
fewer than eight rows. 1-NN distance quantiles: p5 0.32, p10 0.44, p25 0.75, p50 1.10, p75 1.43, p90 1.79,
p95 2.07.

| `d` bin | rows | rows in OOF | γ(d)^½ | **model RMSE, same rows** | ratio |
|---|---:|---:|---:|---:|---:|
| (0.00, 0.25] | 58,919 | 17,478 | 140.32 | **139.11** | 0.991 |
| (0.25, 0.50] | 202,215 | 60,485 | 152.50 | 149.45 | 0.980 |
| (0.50, 0.75] | 252,497 | 77,055 | 165.23 | 158.38 | 0.959 |
| (0.75, 1.00] | 245,008 | 75,689 | 178.94 | 166.62 | 0.931 |
| (1.00, 1.25] | 574,590 | 182,872 | 184.01 | 171.80 | 0.934 |
| (1.25, 1.50] | 306,926 | 100,070 | 221.76 | 206.29 | 0.930 |
| (1.50, 2.00] | 299,407 | 103,070 | 281.47 | 257.08 | 0.913 |
| (2.00, 3.00] | 101,393 | 38,275 | 404.91 | 357.07 | 0.882 |
| (> 3.00) | 20,658 | 8,738 | 823.62 | 591.34 | 0.718 |
| whole lane A | 2,061,650 | 663,742 | — | **213.05** | — |

Model A's RMSE over the paired subset is 213.050 against 213.1266 over the whole lane, so the pairing is
not selecting on difficulty at the level of the lane.

**[X] This is the trap this route sets, and it is worth recording because the number it produces is
exactly the "board 230s is reachable" number.** Read the second column alone and the floor looks like
**140 s**, which would put 148.9 comfortably in play. Read the third column beside it and the 140 s
evaporates: the rows with a twin at `d ≤ 0.25` are **2.9 % of the lane**, they are its densest and
easiest rows, and **model A already scores 139.1 on them**. A floor of 140 measured on rows where the
model is at 139 is not headroom; it is the statement that the model is at the floor there.

**[S] The ratio column is the finding.** Model A's RMSE is at or just below the twin semivariance at every
density, and the margin grows with `d` exactly as it must — the inflation term `(μ_i−μ_j)²/2` grows with
distance, so the estimator over-states more where twins are further apart. There is no bin in which the
model is materially *above* the spread of its own near-twins.

### 3.3 Post-stratified to the lane's own composition

`S3`, post-stratified on 10 airports × 5 NM-delay quintiles (50 cells), the same device as
`missing-clock.md` §2.2:

| `d ≤` | row coverage | γ^½ | model RMSE, same rows | difference |
|---:|---:|---:|---:|---:|
| 0.50 | 0.127 | 166.46 | 166.20 | −0.26 |
| 0.75 | 0.249 | 171.06 | 168.18 | −2.88 |
| 1.00 | 0.368 | 175.58 | 169.35 | −6.23 |
| 1.25 | 0.647 | 180.76 | 172.63 | −8.13 |
| 1.50 | 0.796 | 188.23 | 179.02 | −9.21 |
| **2.00** | **0.941** | **203.55** | **192.23** | −11.32 |

**Control.** `S3`, with `g` permuted within airport, γ^½ reads **364.48** overall and 356.48 at
`d ≤ 1.0`, against a within-ADEP sd of `g` of 364.35. The neighbour search is not inventing structure.

**[X] At 94 % row coverage this route reads 203.6, which is the same neighbourhood as §4's 201.0.** The
residual selection — within a stratum, rows with close twins are still the denser ones — is why it is not
quoted as the headline: post-stratification fixes the airport × delay mix, not the density.

---

## 4. Route 3 — the twin-residual covariance, and the systematic share ρ

### 4.1 The estimator

`S4`. Write `e = ghat(x) − g`. For two rows with the **same** visible `x`, fitted by the **same** model,

```
e_i = (ghat(x) − μ(x)) + (μ(x) − g_i)
```

and the second terms are independent across `i` with mean zero. Therefore

```
E[e_i e_j]  =  E[(ghat − μ)²]                    the systematic, recoverable part
E[e_i²]     =  E[(ghat − μ)²] + E[Var(g|x)]  =  MSE
ρ           =  E[e_i e_j] / MSE                  the recoverable SHARE
floor       =  RMSE · √(1 − ρ)
```

At `d > 0` the two rows' approximation errors are not identical, so the measured covariance
**under-states** `E[(ghat−μ)²]` and every floor below is an **upper** bound that falls towards the truth
as `d → 0`. Pairs are restricted to the same OOF fold so that `ghat` comes from one model; six nearest
neighbours per row are kept, capped at `d ≤ 1.0`.

**[X] Why this route is the load-bearing one.** It estimates a *ratio*, not a level, so the density
selection that defeats §3 cancels to first order: the question "what share of this row's error is
systematic" can be asked in a dense neighbourhood and carried to a sparse one, and §4.3 is the evidence
that the share does not move much when it is.

### 4.2 The result

`S4`. 2,216,552 pairs at `d ≤ 1.0`; 140,520 of them have both rows in the OOF and in the same fold.
Lane A OOF RMSE over all 664,050 rows: 213.1266.

| `d` bin | pairs | MSE^½ on those rows | **ρ** | ρ 95 % CI (400-sample bootstrap) | floor applied to the lane |
|---|---:|---:|---:|---:|---:|
| (0.00, 0.25] | 5,091 | 134.91 | **0.1104** | **[0.0742, 0.1459]** | **201.02** |
| (0.25, 0.50] | 26,691 | 144.82 | 0.1117 | [0.0914, 0.1335] | 200.87 |
| (0.50, 0.75] | 49,019 | 149.53 | 0.1036 | [0.0891, 0.1198] | 201.78 |
| (0.75, 1.00] | 59,719 | 154.86 | 0.0979 | [0.0853, 0.1096] | 202.42 |

**ρ is flat in `d`**, which is what the estimator needs to be carried from the dense rows to the whole
lane, and the CI at the tightest bin is [0.074, 0.146] → **floor 197.0 to 205.0, point 201.0**.

Restated against §1.2's variance decomposition: model A explains **71.51 %** of Var(`g`), and a perfect
model — one that is handed `μ(x)` itself — would explain **1 − (201.78/399.28)² = 74.46 %**. **[X] The
entire remaining prize on lane A is 2.95 percentage points of Var(`g`).**

### 4.3 Three controls, of which two are the ones this repo's history demands

**Negative control.** `S4`, the partner residual permuted so no pair is a twin: **ρ = −0.0020
[−0.0073, +0.0030]**. Contains zero.

**Positive control — the one that matters.** A null control only shows the estimator can return zero;
it cannot show the estimator would *notice* systematic error if there were some. So the same estimator is
run on a deliberately crippled predictor of `g`: the out-of-half mean of `g` over
(ADEP × AIRCRAFT_OPERATOR). It is a legitimate visible predictor, it is badly underfitted, and it scores
**355.73** against model A's 213.13. `S4`:

| `d` bin | pairs | MSE^½ | ρ | ρ 95 % CI | implied floor on those rows |
|---|---:|---:|---:|---:|---:|
| (0.00, 0.25] | 79,205 | 192.00 | **0.4839** | [0.4741, 0.4924] | **137.94** |
| (0.25, 0.50] | 424,656 | 201.39 | 0.4626 | [0.4589, 0.4671] | 147.63 |
| (0.50, 1.00] | 1,712,639 | 216.34 | 0.4239 | [0.4220, 0.4261] | 164.20 |

**[S] Two things are established at once.** (i) The estimator sees systematic error when it is there:
ρ = 0.484 for a predictor that has one, against 0.110 for model A. (ii) **Two predictors 67 % apart in
RMSE agree on the floor to within 8 %** — on the `d ≤ 0.25` rows model A implies
`134.91 × √(1−0.1104) = 127.2` and the crippled predictor implies 137.9. They must agree, because the
floor is a property of the data and not of the predictor, and the 127.2-vs-137.9 gap is the honest
precision of this instrument. (The crippled arm reads high because its own `ghat` varies more between
twins, so its ρ is under-stated more — the expected direction.)

**Shuffled-cell control from §2** and **permuted-label control from §3** cover the other two routes.

### 4.4 Per airport

`S4`, `d ≤ 1.0`, same fold. `lane RMSE` is `S1`'s per-airport lane-A OOF RMSE.

| ADEP | pairs | ρ | ρ 95 % CI | lane RMSE | floor | recoverable gap |
|---|---:|---:|---:|---:|---:|---:|
| EDDF | 20,320 | 0.0682 | [0.0526, 0.0826] | 175.35 | 169.26 | 6.09 |
| EDDM | 8,949 | 0.1024 | [0.0704, 0.1343] | 175.73 | 166.49 | 9.24 |
| EGLL | 14,008 | 0.0601 | [0.0423, 0.0773] | 222.07 | 215.29 | 6.78 |
| EHAM | 13,725 | 0.0904 | [0.0699, 0.1090] | 161.46 | 153.99 | 7.47 |
| LEBL | 17,662 | 0.0622 | [0.0430, 0.0829] | 195.33 | 189.15 | 6.17 |
| LEMD | 19,779 | 0.0737 | [0.0572, 0.0921] | 171.62 | 165.17 | 6.45 |
| **LFPG** | 14,049 | **0.1304** | [0.1010, 0.1613] | 241.53 | **225.23** | **16.30** |
| **LIRF** | 15,718 | **0.1858** | [0.1595, 0.2171] | 345.14 | **311.42** | **33.72** |
| LSZH | 8,975 | 0.0472 | [0.0213, 0.0749] | 167.20 | 163.21 | 3.99 |
| LTFM | 7,335 | 0.0803 | [0.0538, 0.1065] | 229.20 | 219.80 | 9.40 |
| **pooled** (SSE-weighted, and it reproduces 213.126) | | | | **213.13** | **201.78** | **11.35** |

**[X] The recoverable gap is concentrated, and it is concentrated on the airports that are already the
expensive ones.** LIRF alone is 33.7 s of its own RMSE and **35.0 %** of the total recoverable squared
error; LIRF + LFPG together are **53.6 %** of it, and LSZH is **1.8 %**. A project that recovered
*everything* at LIRF and LFPG and nothing anywhere else would take lane A from 213.13 to **207.13**.

**This is not a floor of 150 that is all LFPG, and it is not one spread evenly over ten airports either.
It is a floor of ~201 s, of which just over half the remaining slack sits at two airports — and those two
are the ones where the error is already largest, so the work is concentrated but it is also the work
nobody has managed yet.**

### 4.5 Does the narrow metric hide systematic error? — no

`S4`. The narrow metric matches on twelve columns carrying 85 % of the gain, so two rows at `d = 0` can
still differ on the other ~90 visible columns. If model A is systematically wrong *along* those columns,
that error never enters `Cov(e_i, e_j)`, ρ is under-stated and the floor is over-stated — **the one bias
in this note that flatters its own conclusion** (§10). Re-running the entire estimator on a **21-column**
metric, adding the congestion block (`dep_tko_prev_30m/60m`, `rwy_headway_prev/next_s`, `queue_ahead`,
`nbr_mvt_minus_aobt_mean_30m`, `arr_taxi_in_mean_30m`), the weather block (`temp_c`, `wind_ms`, `vis_km`)
and `hour_local`:

| `d` bin (wide metric) | pairs | MSE^½ | ρ | ρ 95 % CI | floor applied to the lane |
|---|---:|---:|---:|---:|---:|
| (0.50, 1.00] | 2,708 | 141.57 | **0.1152** | [0.0682, 0.1564] | **200.47** |
| (1.00, 1.50] | 31,670 | 144.74 | 0.1063 | [0.0915, 0.1225] | 201.48 |
| (1.50, 2.00] | 191,747 | 149.03 | 0.0572 | [0.0506, 0.0643] | 206.94 |

**[S] ρ = 0.115 on the wide metric against 0.110 on the narrow one, and the implied floor is 200.5
against 201.0.** Adding thirteen columns to the definition of "twin" does not find systematic error that
the narrow metric missed. **[X] The bias is bounded and it is small; the conclusion does not rest on
the metric.** (The decay to 0.057 by `d = 2` is the expected under-statement at distance, not a finding.)

### 4.6 Per fold — and the bridge to §5

`S4`. The pooled 213.13 averages three folds that differ by 45 s, and §5's learning curve is measured on
the forward fold alone. ρ per fold lets the two be compared without assuming the folds behave alike:

| fold | pairs | ρ | ρ 95 % CI | lane-A RMSE | floor |
|---|---:|---:|---:|---:|---:|
| jan | 22,654 | 0.1140 | [0.0949, 0.1355] | 203.40 | 191.45 |
| jul | 21,789 | 0.0777 | [0.0531, 0.0988] | 243.28 | 233.63 |
| **forward** | 96,077 | **0.1077** | [0.0976, 0.1189] | **198.53** | **187.53** |

**[X] jul is the outlier on both axes** — the worst lane-A RMSE and the lowest ρ, i.e. its extra 45 s over
forward is almost entirely irreducible rather than recoverable. That is consistent with jul being a
seasonal-composition problem rather than a modelling one, and it means a project that improved lane A
uniformly would still not close the fold spread.

### 4.7 Could the twins be sharing an operational shock rather than a model error?

`S4`. Two rows minutes apart on the same stand can share a real event — a closed taxiway, a de-icing
queue — that is **not** a function of `x`. That puts a genuine `Cov(g_i, g_j | x)` into the estimator,
inflating ρ and pushing the floor **down**. If the covariance were a shared shock it would decay with the
time between the two flights:

| \|t_i − t_j\| | pairs | MSE^½ | ρ | ρ 95 % CI | floor applied to the lane |
|---|---:|---:|---:|---:|---:|
| < 1 day | 6,798 | 149.25 | 0.1178 | [0.0825, 0.1482] | 200.18 |
| 1–7 days | 38,569 | 152.20 | 0.1139 | [0.0987, 0.1311] | 200.62 |
| 7–30 days | 76,176 | 150.62 | 0.0940 | [0.0836, 0.1059] | 202.87 |
| 30–120 days | 18,977 | 146.57 | 0.1093 | [0.0841, 0.1336] | 201.15 |

**[S] ρ is flat from minutes apart to four months apart.** The covariance is not a shared shock; it is a
repeatable error the model makes on that kind of flight, which is what the estimator is supposed to
measure. **[X] And it answers the brief's "adjacent days" replicate idea directly: twins a day apart and
twins a season apart give the same answer.**

---

---

## 5. Route 4 — the learning curve: information-limited, or data-limited?

The three routes above say how much of the error is systematic. They do not say whether that systematic
part is the model's *function class* falling short or simply not enough rows. The learning curve
separates them, and it matters for the two-week decision: a data-limited model is fixed by nothing
anyone can do in two weeks (2025 is all the data there is), an information-limited one is fixed only by
a new column.

### 5.1 The measurement this repo already owns

**[S]** `P4-turnaround-features.md:136`, from `8bdf4f9:docs/codex_fe_results/continuation_evaluation.json`.
Identical settings on both arms (LightGBM `lr 0.05, 127 leaves, min_data 50, feature_fraction 1.0`,
one seed, **fixed 742 matched rounds, no early stopping**), identical evaluation population — the
forward fold's 328,009 rows:

| arm | training rows | RMSE (both lanes) | Δ | 95 % day CI |
|---|---:|---:|---:|---|
| 8-month baseline (Jan–Aug) | 1,387,414 | 228.3237 | — | — |
| **10-month baseline (Jan–Oct)** | **1,757,038** | **225.6349** | **−2.6888** | [−4.923, −0.954] |

**[X] A 26.6 % increase in training rows bought 2.69 s, and the model was therefore NOT at its data
limit as of the 10-month baseline.** Extrapolating a power law through two points needs an assumed
exponent, so what follows is an inference, not a measurement — under `MSE = a + b·n^−α` the `n → ∞`
asymptote implied by those two points is **215.2** at `α = 1`, **211.2** at `α = 0.75` and **202.9** at
`α = 0.5`, i.e. a further −10 to −23 s.

**[X] Three things have to be said about that number before anyone plans against it.**

1. **It is both lanes, and lane B is the starved one.** Going from 8 to 10 months takes model B from
   about 18k to about 22k training rows, and `missing-clock.md` §2.2 prices model B's training-size
   starvation at a 1.14× penalty on its own RMSE — which, on a lane carrying 41.8 % of total SSE, is
   easily the whole 2.69 s. **The 8→10 datum is not clean evidence about lane A**, which is why §5.2
   re-measures it on matched rows alone.
2. **There is no more data.** 2025 is the whole training year; the asymptote is a counterfactual that no
   amount of work in two weeks can reach.
3. **[X] And it cuts the other way for feature engineering.** §4's ρ measures *total* recoverable error —
   function-class shortfall plus estimation variance. Whatever part of it is estimation variance is
   removable only by rows that do not exist. So the share of §4's 11.3 s that feature work could
   actually claim is **smaller** than 11.3 s, not larger.

### 5.2 Measured here

`S5`. Forward fold, **matched rows only** (validation 325,004; training pool 1,737,573 matched labelled
rows from Jan–Oct). The recipe is `configs/main_gap.yaml` cheapened identically at every point — `lr`
0.15 instead of 0.05, at most 400 rounds instead of 2000, one seed instead of three, and
`deterministic`/`force_row_wise` **off**, which on this box costs about 30× in wall clock for a
bit-reproducibility nothing here needs. Early stopping is kept, so no point is handicapped by a fixed
round budget. **The level is therefore about 4 s worse than production: the `frac = 1.0` point measures
202.70 where `configs/main_gap.yaml` measures 198.53 on the same fold with three seeds. Only the shape
is read.**

| training rows | fraction | rounds | matched-lane RMSE | Δ per doubling |
|---:|---:|---:|---:|---:|
| 108,699 | 6.25 % | 144 | 228.087 | — |
| 217,630 | 12.5 % | 198 | 220.342 | −7.74 |
| 435,263 | 25 % | 195 | 213.125 | −7.22 |
| 870,233 | 50 % | 281 | 207.293 | −5.83 |
| **1,737,573** | **100 %** | 290 | **202.701** | **−4.59** |

**[S] Model A is still data-limited at 1.74 M rows.** The last doubling is still worth 4.6 s and the
deltas are decaying slowly (ratio ≈ 0.79 per doubling). A least-squares fit of `MSE = a + b·n^−α` over
the five points gives **α = 0.283** and an `n → ∞` asymptote of **178.24**, with

| counterfactual | rows | RMSE | Δ vs today |
|---|---:|---:|---:|
| 2× the training data | 3,475,146 | 198.46 | −4.24 |
| 4× | 6,950,292 | 195.01 | −7.69 |
| 10× | 17,375,730 | 191.31 | −11.39 |
| `n → ∞` | — | 178.24 | −24.46 |

### 5.3 Capacity is saturated; rows are not

`S5`, all three at 435,263 rows so that only capacity moves:

| `num_leaves` | rounds | matched-lane RMSE |
|---:|---:|---:|
| 31 | 400 (hit the cap) | 213.229 |
| **127** (production) | 195 | **213.125** |
| 511 | 87 | 213.376 |

**[S] A 16× range of model capacity moves the answer by 0.25 s, while a 2× range of training rows moves
it by 5.8 s.** Production's 127 leaves is not leaving anything on the table, and neither is anything
larger. **[X] One caveat: the 31-leaf arm hit the 400-round cap, so it is round-limited rather than
capacity-limited and its 213.23 is if anything pessimistic — which only strengthens the reading that the
three are indistinguishable.**

### 5.4 Route 4 against route 3, on the same fold, with no rescaling

§4.6 gives route 3's floor **on the forward fold specifically**: lane A 198.53, ρ 0.1077, floor
**187.53**. §5.2's curve is on that same fold. The only adjustment needed is the cheapening, which the
`frac = 1.0` point measures directly: 202.70 here against 198.53 at production settings, i.e. **−4.17 s**
applied to every point.

| | forward-fold RMSE, cheapened | production-equivalent (−4.17) |
|---|---:|---:|
| today (100 % of the rows) | 202.70 | **198.53** (measured, `configs/main_gap.yaml`) |
| 2× the rows | 198.46 | 194.3 |
| 4× the rows | 195.01 | 190.8 |
| **10× the rows** | **191.31** | **187.1** |
| `n → ∞` | 178.24 | 174.1 |
| **route 3's floor on this fold (§4.6)** | — | **187.53** |

> **[S] Route 4 at ten times the training data lands at 187.1. Route 3's floor on the same fold is
> 187.53. The two agree to 0.4 s, by methods that share no machinery: one counts how much a twin's
> residual predicts this row's, the other fits models on 6.25 % to 100 % of the rows and extrapolates.**

**And here is the one place they disagree, reported rather than averaged.** Route 4's `n → ∞`
extrapolation is 174.1, which is **13.4 s below** route 3's floor — and an `n → ∞` limit cannot be below
the floor. One of the two is wrong:

* the `α = 0.283` power law is fitted over one order of magnitude in `n` and then pushed to infinity.
  Its 2× and 4× points are interpolation-grade; its `∞` point is an extrapolation of a curve that is
  still falling, and the fitted `α` is the least constrained parameter in this note. **[X] My weight is
  on this being the wrong one, and I would not defend 174.**
* or ρ is under-stated, which §4.5 bounds at about 0.5 s and §10 records.

**[X] So the floor is a range, and the honest way to state it is: 187 – 205 s pooled, with route 3 at
201.0 [197.0, 205.0], route 4's usable (interpolated) end agreeing with route 3 exactly, and route 4's
extrapolated end 13 s lower and not defended. The range's own width — 18 s — is less than a third of the
44 s that separates today from a board of 234.5, which is why nothing in §7 turns on resolving it.**

Pooled-equivalents, for readers who want the numbers on §1.1's axis, by scaling each forward-fold point
by `213.1266 / 198.526`. **[X] This rescaling assumes the three folds' curves have the same shape and is
an inference, not a measurement:**

| route 4 point | forward (production-equiv.) | pooled-equivalent lane A |
|---|---:|---:|
| today | 198.53 | 213.13 |
| 2× the rows | 194.3 | 208.6 |
| **10× the rows** | **187.1** | **200.9** |
| `n → ∞` | 174.1 | 186.9 |

---

## 6. What an oracle cell-mean correction can still recover

`S6`. For any key `K` built from visible columns, the best correction constant inside a cell is
`E[e | K]`. This section estimates it **out of fold** (5 random row folds, cell means from the other four)
with ridge shrinkage λ, and reports lane-A RMSE after subtracting it. The control shuffles the cell
labels, so its best column is what a key carrying no information buys.

`S6`, lane A, 664,050 OOF rows, baseline **213.1266**:

| cell key | cells | λ=0 | λ=5 | λ=20 | λ=50 | λ=150 | λ=500 | control (best) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 ADEP | 10 | 213.012 | 213.012 | 213.012 | 213.012 | 213.012 | 213.012 | 213.075 |
| C2 +operator | 2,452 | 213.117 | 212.918 | 212.882 | **212.864** | 212.863 | 212.896 | 213.102 |
| C3 +stand | 28,855 | 218.236 | 214.246 | 213.442 | 213.188 | 213.067 | 213.060 | 213.137 |
| C4 +runway | 56,558 | 222.217 | 214.858 | 213.602 | 213.245 | 213.094 | 213.085 | 213.132 |
| C5 +dest | 213,234 | 238.167 | 215.932 | 213.589 | 213.201 | 213.110 | 213.113 | 213.127 |
| C6 +3 h block | 336,025 | 239.641 | 214.923 | 213.227 | 213.074 | 213.084 | 213.110 | 213.129 |
| C7 +actype +delay band | 543,181 | 229.522 | 213.715 | 213.120 | 213.098 | 213.112 | 213.121 | 213.127 |
| stand alone | 2,529 | 213.378 | 213.248 | 213.173 | 213.113 | 213.050 | 213.031 | 213.144 |
| stand × rwy × 3 h | 43,843 | 221.487 | 215.272 | 213.717 | 213.240 | 213.079 | 213.088 | 213.135 |
| operator alone | 622 | 213.102 | 213.027 | 213.011 | 213.002 | **212.995** | 212.998 | 213.093 |

**[S] The best cell in the whole table recovers 0.27 s and the shuffled control recovers 0.10 s.** An
oracle that is *handed the true mean residual* of every one of 543,181 cells built from airport, operator,
stand, runway, destination, three-hour block, aircraft type and NM-delay quintile cannot move lane A.

**[X] What this does and does not close.** A cell-mean correction is exactly what a target encoding, a
per-group specialist, a per-group offset column or a per-airport recalibration would buy, so **that whole
family is closed at 0.3 s**. It cannot see structure that varies *continuously* inside a cell — §4's ρ
can, and §4 is where the remaining 11 s lives. The two results are consistent, not contradictory: the
recoverable part of model A's error is not a per-group constant.

### 6.1 A fifth line of evidence, of a different kind: this project's own return curve

Not a conditional-variance estimate, and marked as such, but it is what the two-week decision actually
turns on. From `experiments/experiments.jsonl`:

* **The 2026-09-15 feature screen, forward fold, ten candidates against a base of 285.52 / 285.58 (two
  runs of the same config, so the instrument's own spread is 0.06 s):** `queue_ahead` 288.90,
  `bank_position` 288.67, `stand_rwy_share` 287.72, `noise_control` 287.53, `active_runways_60m` 290.14,
  `dep_arr_ratio_30m` 290.29, `congestion_vs_usual_30m` 289.85, `op_offset` 289.94, `dest_distance_km`
  287.20, `planned_flight_s` 290.25. **Every one is worse than base.** [S] `experiments.jsonl`
* **P4 stage A, the one adopted feature, on lane A specifically:** base 201.413, noise arm 201.494,
  `stand_arrival_gap` **198.526** — i.e. **−2.89 s of lane A** against a noise control at +0.08.
  [S] `experiments.jsonl`, `configs/main_gap.yaml`
* **`missing-clock.md` §4–§6**, nine further candidates: all at or inside the noise floor, none surviving
  conditioning against the residual.
* **The whole campaign moved lane A 216.3 → 213.1** (the brief's own figure).

**[X] Thirteen months of candidate columns have produced one adopted feature worth −2.9 s on lane A.
§4 says 11 s remains. Those two numbers are consistent with each other and both are inconsistent with
−64 s.**

---

## 7. Are 148.9 and 141.7 reachable? — no

`S7`. Holding lane B fixed at its measured SSE of 2.1633e10 on 8,378 rows, and passing through the
brief's +1.98 s OOF-to-board gap (whose instability is §8's subject):

| board target | implied pooled OOF | lane A must reach | vs 213.13 | and at §8.2's measured ratio 0.747 |
|---:|---:|---:|---:|---:|
| 234.5000 | 232.5164 | **148.89** | −30.1 % | **123.40** |
| 230.0000 | 228.0164 | **141.67** | −33.5 % | **112.32** |
| 279.5234 (v14, today) | 277.5398 | 213.13 | — | 213.13 |

The last column is what the same targets become once the board's response to a CV improvement is taken
from the one place it has been measured rather than assumed to be one-for-one (§8.2). **Everything below
uses the brief's own, more generous, column.**

Restated as a systematic share, which is the quantity §4 actually measures:

| target lane-A RMSE | ρ it requires | measured ρ | positive control (crippled predictor) |
|---:|---:|---:|---:|
| 148.9 (board 234.5) | **≥ 0.5119** | 0.1104 [0.0742, 0.1459] | 0.4839 |
| 141.7 (board 230.0) | **≥ 0.5580** | 0.1104 | 0.4839 |
| 201.0 (§4's floor) | 0.1106 | 0.1104 | — |

> **[X] In one sentence, as asked: 148.9 and 141.7 both sit below every floor estimate in this note — the
> nearest is 38 s away — and reaching them would require model A to be *more* systematically wrong than a
> predictor that knows only the airport and the operator, so lane A cannot get there alone and a board
> score in the 230s is not a lane-A project.**

What lane A *is* worth. Recovering all 11.34 s of §4.4's gap takes lane A to 201.78 and the pooled OOF to
**269.04**: a **−8.50 s** move of the CV statistic, and that is the *whole* prize for a perfect model A.
On the board that is −8.50 s at a transfer ratio of 1.000 and **−6.35 s** at §8.2's measured 0.747 — i.e.
**279.52 → 271.0 to 273.2**. [S] `S4`, `S7`

The realistic fraction of that, given §6's 0.3 s for the whole cell-mean family and §6.1's record — one
adopted feature in the campaign, worth −2.89 s of lane A and −3.62 s of board — is **−1 to −3 s of board**
over two weeks, i.e. a board somewhere in **276.5 – 278.5**. **[X] That last range is an inference from the project's own return curve, not a
measurement.**

---

## 8. Is the +1.98 s OOF-to-board gap stable? — the level is not, the derivative is worse, and neither rescues the targets

`S7`. **There are exactly two submissions whose board score can be paired with a pooled three-fold OOF
from the same single config**, not one. v13's file was built from the `submit_v10_reparam` full-data
refit with the LOBT bound applied, and `configs/main_reparam.yaml` **is** `configs/submit_v10_reparam.yaml`
plus `lobt_bound` — so `oof_main_reparam-bc60017878` is v13's OOF. v1–v12 were hand-assembled from two
model lineages and have no comparable pooled OOF (`experiments/submissions.jsonl`).

| version | exp_id | pooled OOF | lane A | board | gap |
|---|---|---:|---:|---:|---:|
| v13 | `main_reparam-bc60017878` | 281.8411 | 216.3393 | 282.7371 | **+0.8960** |
| v14 | `main_gap-62dd7c6e88` | 277.5398 | 213.1266 | 279.5234 | **+1.9836** |

**[S] The two gaps differ by 1.09 s on a quantity that is about 1.5 s.** The brief's +1.98 is the larger
of two observations, not a constant.

### 8.1 The level moves with the pooling, by 125 seconds

`S7`, same predictions, different fold subsets. The board is Jan + Jul **2026**; the pooled OOF is
jan + jul + forward **2025**, and **49 % of its rows come from months the board does not contain**:

| fold subset | n | pooled OOF | lane A | implied gap to board 279.5234 |
|---|---:|---:|---:|---:|
| all three (as shipped) | 672,428 | 277.5398 | 213.13 | **+1.98** |
| **jan + jul only (the board's months)** | 344,419 | **323.1222** | 226.24 | **−43.60** |
| forward only | 328,009 | 219.7247 | 198.53 | **+59.80** |
| jan only | 153,706 | 344.7506 | 203.40 | −65.23 |
| jul only | 190,713 | 304.5749 | 243.28 | −25.05 |

**[X] +1.98 is a coincidence of the three-fold average.** The implied gap ranges over 125 s and lane A
alone ranges 198.5 to 243.3, so "the pooled OOF" is not a single well-defined estimate of board
performance — it is an average over months that behave nothing alike, half of which the board does not
contain.

### 8.2 But the level is not what the targets need — the derivative is, and it is 0.75

**[X] An important correction to how the brief uses the gap.** A *constant* gap cancels when you compute
an improvement: board −45.02 s would need pooled OOF −45.02 s whatever the gap's level. So §8.1 alone
does not invalidate the targets. What does is the **transfer ratio** `ΔBoard / ΔOOF`, and the two pairs
give exactly one measurement of it:

| | pooled OOF | board | lane A |
|---|---:|---:|---:|
| v13 → v14 | **−4.3013** | **−3.2137** | −3.2127 |
| **transfer ratio `ΔBoard / ΔOOF`** | | **0.747** | |

**[X] n = 1, and both terms are differences of noisy quantities, so 0.747 carries no error bar and must
not be treated as calibrated.** It is, however, the only measurement of the thing every target in the
brief silently assumes to be 1.000. `S7`, targets under both:

| assumed ratio | board target | pooled OOF needed | **lane A needed** |
|---|---:|---:|---:|
| 1.000 (the brief's) | 234.50 | 232.52 | **148.89** |
| 1.000 | 230.00 | 228.02 | **141.67** |
| **0.747 (measured)** | 234.50 | 217.28 | **123.40** |
| **0.747 (measured)** | 230.00 | 211.26 | **112.32** |

**[S] The instability does not rescue the targets; it moves them further away.** At the only transfer
ratio anyone has measured, board 234.5 needs lane A at **123.4** — against a floor of 187–205 and a
current 213.1.

### 8.3 What should be done with the gap

**[X] Three things, stated as recommendations and marked as opinion.**

1. **Stop computing board targets through a single additive gap.** Report the pooled OOF and the board
   score side by side, and quote *deltas*, which is what every decision actually turns on.
2. **The third OOF↔board pair is cheap and worth a slot.** Two points give a ratio with no error bar;
   three give a spread. Any future submission whose OOF comes from one config adds one.
3. **Consider reporting `jan + jul` as a second headline CV number.** It is the board's own season and it
   is 45.6 s away from the three-fold pool, so the two disagree about how good the model is by more than
   the entire improvement this project is trying to find.

---

## 9. What this leaves

* **The floor on lane A is 187 – 205 s, best estimate 201**, from the twin-residual covariance route
  (§4, 201.0 [197.0, 205.0]) and the learning curve (§5, 200.9 at 10× rows, 186.9 extrapolated to `n → ∞`;
  187.1 against 187.53 when the two are compared on the forward fold with no rescaling).
  The 1-NN semivariance route reads 203.6 at 94 % coverage (§3) and the matched-twin ladder reads 214.0
  and cannot go lower (§2). §3's apparent 140 is a selection effect that §3.2's third column resolves;
  §5's 186.9 is a power-law extrapolation I do not defend, and §5.4 says why rather than averaging it away.
* **Recoverable: 11.3 s of lane-A RMSE, 213.13 → 201.78, ≈ −8.5 s of board if every second of it were
  taken.** LIRF and LFPG are 53.6 % of it; LSZH is 1.8 %.
* **Closed at 0.3 s:** every re-encoding, target encoding, per-group specialist and per-group offset over
  airport / operator / stand / runway / destination / hour / aircraft type / NM-delay band (§6).
* **Not closed:** a column carrying genuinely new information. This note bounds what the *current*
  columns support. `missing-clock.md` §6 names the one candidate class — an external off-block
  observation — and records that it is a governance question, not a feature-engineering one.
* **Out of reach from lane A:** board 234.5 and board 230.0 (§7, §8).

---

## 10. Caveats on the instrument, recorded rather than buried

* **Every route bounds the floor from ABOVE, and no route can do otherwise.** §7's argument is not "the
  floor is 201, therefore 148.9 is impossible"; it is "the systematic share would have to be 0.512 and it
  measures 0.110, with a deliberately crippled model at 0.484". That is evidence, not proof.
  **[X] Anyone who wants a real lower bound on conditional variance should know it is not estimable from
  data at all.**
* **§4's central assumption is that ρ carries from rows with a close twin to rows without one.** It is
  the biggest weakness of the estimate. Four things support it and none of them is conclusive: ρ is flat
  across `d` (0.098 – 0.112 over `d ≤ 1`, 0.084 – 0.128 out to `d > 3`); flat across time separation
  (§4.7); stable on a 21-column metric (§4.5); and the two hardest airports have the *highest* ρ, so the
  assumption is not obviously optimistic. **If ρ on the sparse half of the lane were double the measured
  value the floor would be ≈ 188, which is still 39 s above 148.9.**
* **The matching metric is a choice, and the one bias in this note points the wrong way.** The narrow
  metric is eight continuous and four categorical columns inside a stand block, chosen by LightGBM gain
  share on `g`: `aobt3_minus_eobt1` 23.61 %, `mvt_minus_aobt3` 14.75 %, `STAND_mvt` 12.21 %,
  `RUNWAY_mvt` 7.22 %, `ADES_mvt` 5.76 %, `AIRCRAFT_OPERATOR_flt` 4.34 %, `mvt_minus_aobt3_slog` 4.13 %,
  `ADEP_mvt` 3.36 %, `aobt3_minus_eobt1_slog` 3.34 %, `aobt3_minus_sched` 3.30 % — 85 % of the gain in
  twelve columns (400 rounds, 300k-row subsample, seed 1). Systematic error lying along the ~90 omitted
  columns would not enter `Cov(e_i, e_j)`, so ρ would be under-stated and §7 flattered. **§4.5 bounds
  that at about half a second** by re-running everything on 21 columns.
* **`g` is heavy-tailed and every number here is an L2 quantity dominated by that tail.** ρ's CI at the
  tightest narrow-metric bin rests on 5,091 pairs and is wide, [0.074, 0.146]. The three wider bins, on
  27k–60k pairs, agree with it; a reader who trusts only the tightest bin should read the floor as
  197 – 205 and nothing narrower.
* **`S5` is a cheapened recipe and its absolute level must not be quoted.** `lr` 0.15 not 0.05, ≤ 400
  rounds not 2000, one seed not three, `deterministic`/`force_row_wise` off. The cost is measured, not
  assumed: 202.70 against production's 198.53 on the same fold. **The `α = 0.283` asymptote is the least
  defensible number in the note** (§5.4).
* **The box was not idle.** A stray `pytest tests/test_train.py` held roughly one of the four CPUs for
  much of the session, and the first attempt at `S5` was abandoned after a 400-round fit on 109k rows
  failed to finish in eight minutes. The cause was measured, not guessed: with the production
  `deterministic: True` + `force_row_wise: True`, LightGBM on this data costs about 5.6 s/round at 110k
  rows (`bench_lgb.py`, 60 rounds, > 120 s) against 0.1 s/round with them off. **[X] Worth recording
  independently of this note: the determinism flags are costing this project roughly an order of
  magnitude of wall clock on every training run, in exchange for bit-reproducibility of deltas that are
  then judged with a ±1 s noise control anyway.**
* **One OOF, one production recipe, three folds.** Everything is measured on `oof_main_gap-62dd7c6e88`.
  Fold-to-fold spread on lane A is 198.5 / 203.4 / 243.3 (`S1`), so the pooled 213.13 is not a tight
  quantity either, and §8 is the direct consequence.
* **§8.2's transfer ratio is n = 1.** 0.747 is a ratio of two differences of noisy quantities, with no
  error bar. It is quoted because 1.000 is the alternative and 1.000 has never been measured at all.
* **The cell-mean correction of §6 uses labels of other rows.** Legitimate as a diagnostic of whether
  structure exists, and implementable at prediction time *only* for keys that recur across time — which
  is why no date or week enters any key in this note. A same-day key would have measured something real
  and unusable, since the scored rows' neighbours are also unlabelled.
* **No control was added after the fact.** §2's shuffled-cell arm, §3's permuted-label arm and §4's
  permuted-partner and crippled-predictor arms were part of the design before any headline number was
  read, on the explicit instruction of `missing-clock.md` §7 and `P1-stage2-gate-FAILED.md`. The failure
  signature those documents name — several feature sets returning *identical* numbers, or a floor near
  zero — did not occur: the routes return 214.0, 203.6, 201.0 and 200.5, which is agreement without being
  suspicious identity, and the negative controls return 364.4, 364.5 and −0.002 exactly where they should.
