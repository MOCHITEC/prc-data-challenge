# ADS-B pushback: how much would it improve the score? Two days on disk, no new data (2026-09-29)

**Script:** `scripts/adsb_gain_estimate.py`. It imports the frozen detection rules from
`scripts/adsb_stage1_feasibility.py` and does not change them. It is read-only with respect to the repo:
- nothing is written to `outputs/`, `src/`, `configs/` or `data/`;
- nothing is fitted or submitted;
- nothing is downloaded.

It reads:
- the two `*_joined.parquet` files and the 2025-07-21 fixes that stage 1 left in the session scratchpad (`…/scratchpad/adsb/`);
- `experiments/oof_main_gap-62dd7c6e88{,_folds}.parquet` and `data/work/feat_train.parquet`;
- the v14 submission `outputs/main_gap.parquet`, read only, for the label-free 2026 check.

```
POLARS_MAX_THREADS=2 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage \
.venv/bin/python scripts/adsb_gain_estimate.py            # ~4 min; --skip-sweep drops S5 (~5 s)
```

**Sources.**
- `[G Sx]` is section Sx of that script's output. The run is saved at `…/scratchpad/adsb_gain_estimate.log`.
- `[St1]` is `adsb-stage1-feasibility.md`.
- `[MAC]` is `model-a-ceiling.md`.
- `[X]` marks an inference.

**Definitions.**
- Lane A means `mvt_minus_aobt3` is not null (`src/prc/train.py:402`).
- The label is `y = TAXITIME_SEC_mvt`, which equals `MVT − BLOCK` exactly on every row of the day `[G S0]`.
- The ADS-B prediction is `y_a = MVT − (t_push − b)`, where `b` is the pushback-minus-BLOCK offset. `b` is always
  estimated on rows other than the ones being scored.
- The model is the production OOF `main_gap-62dd7c6e88` (v14). Every 2025-07-21 row is in its `jul` fold `[G S0]`.

---

## 0. The answer

1. **Measured against the model's real residuals, the per-row gain is large, but only with a guard.**
   - **The comparison base is the model, not `AOBT_3`.** On the 914 lane-A flights with a detected pushback, the
     model is already at **RMSE 170.7, MAE 123 s**. `AOBT_3` is at MAE 238 s on the same flights `[G S3]`, `[St1]`.
   - **Naive replacement makes things worse:** RMSE 282.6. Five detection failures (+4,686, +5,590, −3,298 s, …)
     outweigh everything else `[G S3]`.
   - **The guarded replacement works.** It uses the pushback unless it disagrees with the model by more than 900 s.
     The 900 s threshold was cross-fitted, chosen on the other half of the rows. It gives **RMSE 98.7, MAE 38.7 s**.
   - **Gain: +19,400 s² per covered row**, 95 % CI [14,000, 24,600] by row bootstrap and [14,200, 25,100] by
     airport×hour cluster `[G S3]`. That is 43 % of lane A's average MSE (213.13² = 45,423) `[X]`.
2. **Where the gain lands.** The worry was that ADS-B only reaches airports where the model is already good. That
   is true, but the gain survives it.
   - At EDDM, EHAM, LEBL and LSZH the model's RMSE on the covered rows is 154–229 s. ADS-B's is 68–132 s.
   - Gain per covered row: EDDM +19.0k, EHAM +17.3k, LEBL +19.0k, LSZH +46.9k, EDDF +22.1k, LEMD +11.6k s² `[G S3]`.
   - The gain is broad-based, not a few rescued tails: without the 10 largest-gain rows it is still +14.7k per row
     `[G S3]`.
3. **Board estimate: −0.8 to −11 s, central ≈ −4 s**, priced with dBoard/dOOF = 0.747 (§5).
   - **LOW ends (−0.8 to −1.4 s):**
     - the lower 95 % bound of the gain;
     - the lower of the two days' coverage at each airport;
     - January at half of July;
     - zero gain at EGLL.
   - **CENTRAL (−3.9 to −5.2 s):**
     - the point gain, scaled to each airport's pooled error level;
     - 2026-07-20 coverage in July, half of it in January;
     - EGLL at the pooled gain.
   - **HIGH (−8.5 to −11.2 s):**
     - the point or upper-bound gain;
     - the higher of the two days' coverage;
     - January equal to July.
   - The three unknowns that move it most are **January coverage**, **EGLL** (58 % coverage on the 2026 day, 0.8 %
     on the 2025 day, and 5 labelled rows) and **whether the 2026 model is already closer on these rows** (§4).
4. **The in-sample rules cannot be re-measured out of sample from what is on disk.** Every labelled flight on disk
   is from 2025-07-21, the day the rules were designed on. What can be done from disk (§6):
   - **Threshold sweep, still in-sample.** The guarded gain is flat across all 18 single-threshold perturbations: 16.7k to
     21.5k s² per row. Only the structural choice matters: removing the pushback gate turns the gain to −21k.
   - **Stress test.** The guarded gain survives the detection failures that pass the guard becoming 3× as common
     (+10.8k per row). It does not survive 5× (+2.2k).
   - **The 2026 day is out of sample for the rules, but unlabelled.** On it the guard trips on 0.28 % of flights,
     against 0.33 % in 2025. `|AOBT_3 − push| > 20 min` falls from 1.5 % to 0.56 %. So there is no sign of more
     gross failures out of sample.
   - The LOW end therefore includes a failure multiplier. The range is widened, not narrowed.
5. **Given this project's shrinkage record, the estimate is worth less than its CI suggests.** Three measured
   gains shrank about tenfold on the board. One day is weaker evidence than any of them. A tenfold shrink of the
   central gives **≈ −0.4 s**, which would still beat the largest recent board gain (v16, −0.26 s). The LOW end,
   −0.8 s unshrunk, is itself larger than the shrunk central `[X]`.
6. **Recommendation: INGEST-A-SMALLER-SET-FIRST.** It needs about 7 days and ≈ 25 GB, with a hard go/kill criterion
   (§8). Then ingest the full 62 days if it passes.
   - The small set costs about 10 % of the full ingest.
   - It resolves the two things that could flip the answer: out-of-sample accuracy of the frozen rules, and January
     coverage.
   - It is needed anyway, because the bias and the guard must be calibrated on labelled days other than the one they
     were designed on.

---

## 1. Baseline: what the model already does on these airports `[G S1]`

Lane A, production OOF, reproduced: pooled n 672,428, RMSE 277.5398; lane A n 664,050, RMSE 213.1266, 58.23 % of
the SSE.

| ADEP | lane-A n (OOF) | RMSE pooled | RMSE `jul` fold | RMSE 2025-07-21 | RMSE on the day's covered rows | pushback coverage 2025-07-21 | 2026-07-20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| EDDF | 71,489 | 175.3 | 186.5 | 236.9 | 180.3 | 0.097 | 0.033 |
| EDDM | 51,286 | 175.7 | 169.8 | 165.1 | 153.7 | **0.555** | **0.604** |
| EGLL | 78,986 | 222.1 | 243.3 | 286.1 | 377.3 (n=5) | 0.008 | **0.578** |
| EHAM | 78,286 | 161.5 | 166.1 | 175.4 | 154.4 | **0.477** | **0.508** |
| LEBL | 55,501 | 195.3 | 261.2 | 240.9 | 190.5 | 0.265 | 0.333 |
| LEMD | 70,316 | 171.6 | 188.8 | 145.2 | 147.1 | 0.062 | 0.171 |
| LFPG | 76,471 | 241.5 | 242.9 | 277.7 | — | 0.000 | 0.003 |
| LIRF | 49,387 | 345.1 | 439.7 | 375.5 | 336.1 (n=7) | 0.014 | 0.000 |
| LSZH | 42,287 | 167.2 | 180.2 | 198.2 | 229.0 | 0.130 | 0.179 |
| LTFM | 90,041 | 229.2 | 267.4 | 412.9 | — | 0.000 | 0.000 |

Coverage here is the lane-A share with a detected pushback.

**2025-07-21 was a hard day.** Lane-A RMSE was 271.4 on the day, against 243.3 for the whole `jul` fold and 213.1
pooled. `[X]` An absolute per-row gain measured on this day therefore overstates the typical day. §5 carries a
"scaled" variant, which multiplies each airport's gain by (pooled lane-A MSE / day lane-A MSE). The factors are
0.55–1.40, for example EHAM 0.85, LEBL 0.66 and LSZH 0.71.

## 2. Out-of-sample bias correction `[G S2]`

The pushback is systematically later than BLOCK. Per-airport medians: EDDF +53, EDDM +50, EHAM −3, LEBL +45, LEMD
+116, LSZH +30 s. The two hash halves agree to within 5 s at every airport with ≥ 50 rows. That stability is what
makes an out-of-sample correction possible.

| bias scheme (all out of sample) | ADS-B RMSE | MAE | mean error | median error |
|---|---:|---:|---:|---:|
| none | 283.7 | 71.2 | −28.9 | −27.2 |
| pooled median of the other half | 282.2 | 64.4 | −1.2 | +0.5 |
| pooled median of the other airports (leave-airport-out) | 284.0 | 75.1 | +1.5 | −3.7 |
| **per-airport median of the other half** (used below) | **282.6** | **53.4** | +2.7 | −0.6 |
| *reference only, not used: in-sample per-airport median* | 281.2 | | | |

Bias correction cuts the MAE from 71 to 53 s, but it barely moves the RMSE. The RMSE sits in a handful of detection
failures that no offset can fix. That is the job of the guard (§3).

## 3. Ways of using it, on the 914 covered lane-A rows `[G S3]`

Every parameter below is either fixed a priori or chosen on the other hash half and applied to this one. None is
tuned on the rows it is scored on.

| method | RMSE | MAE | ΔMSE per row (s²) | row-bootstrap 95 % CI | airport×hour CI | rows improved |
|---|---:|---:|---:|---:|---:|---:|
| model alone | 170.7 | 123.2 | 0 | | | |
| replace when available | 282.6 | 53.3 | **−50,700** | [−146,900, +19,400] | [−149,600, +18,200] | 84 % |
| blend 0.5 / 0.5 (fixed) | 175.7 | 77.7 | −1,700 | [−26,800, +16,400] | | 88 % |
| blend, weight cross-fitted (w = 0.19 / 0.30) | 157.7 | 100.6 | +4,300 | [−3,900, +9,900] | | 90 % |
| replace only if \|model − ADS-B\| > 120 s | 287.3 | 75.5 | −53,400 | | | 37 % |
| same, margin cross-fitted (chose 0) | = replace | | | | | |
| **replace unless \|model − ADS-B\| > 600 s** | 105.5 | 40.8 | +18,000 | [15,000, 20,800] | [15,000, 20,900] | 84 % |
| **replace unless \|model − ADS-B\| > 1,200 s** | 98.7 | 38.7 | +19,400 | [13,900, 24,900] | [13,600, 24,900] | 84 % |
| **same, guard cross-fitted (chose 900 on both halves)**: the primary | **98.7** | **38.7** | **+19,400** | **[14,000, 24,600]** | **[14,200, 25,100]** | 84 % |

**Why the ordering is what it is.** The disagreement between model and ADS-B is large in two kinds of row: rows where
the model is badly wrong, and rows where the detection failed. On this day the failures are larger: they run to
thousands of seconds.
- Gating on large disagreement ("replace only if they disagree") therefore selects the failures, and it is the
  worst option.
- Guarding against it ("replace unless they disagree wildly") drops them. The guard is label-free: it uses only the
  prediction and the pushback.
- The cross-fitted blend weight is small (0.2–0.3) because a least-squares weight is pulled down by those same
  failures.

**Per airport, primary method** (ΔMSE per covered row, s², with the row-bootstrap CI):

| ADEP | n | model RMSE | ADS-B RMSE (bias-corrected, no guard) | guarded RMSE | ΔMSE per row | 95 % CI | unguarded replace |
|---|---:|---:|---:|---:|---:|---:|---:|
| EDDF | 62 | 180.3 | 603.8 | 101.8 | +22,100 | [3,300, 46,600] | −332,100 |
| EDDM | 279 | 153.7 | 67.7 | 67.7 | **+19,000** | [10,400, 26,100] | +19,000 |
| EGLL | 5 | 377.3 | 157.7 | 157.7 | (+117,500) | [−33,200, 398,200] | |
| EHAM | 328 | 154.4 | 80.6 | 80.6 | **+17,300** | [11,900, 22,800] | +17,300 |
| LEBL | 144 | 190.5 | 131.6 | 131.6 | **+19,000** | [10,700, 26,600] | +19,000 |
| LEMD | 38 | 147.1 | 912.3 | 100.0 | +11,600 | [−1,900, 24,800] | −810,600 |
| LIRF | 7 | 336.1 | 1,333.3 | 475.2 | (−112,800) | | |
| LSZH | 51 | 229.0 | 74.1 | 74.1 | **+46,900** | [14,500, 96,800] | +46,900 |
| pooled, airports with ≥ 30 rows (902) | | | | | +19,900 | [15,600, 24,500] | |

**MAE drops from 123 to 39 s.** Per airport: EDDM 119 → 30, EHAM 115 → 26, LEBL 132 → 59, LSZH 135 → 29 s
`[G S3]`. Stage 1's "238 → 71 s" compared ADS-B with `AOBT_3`. Against the model the starting point is 123 s, not
238, and the guarded ADS-B lands lower, at 39 s.

**Concentration.** The top 10 rows carry 25 % of the guarded gain, and the top 10 % of rows carry 72 %. Without the
top 10 rows the gain is still +14,650 per row. The 7 rows with `|model error| > 600 s` carry 20 % `[G S3]`. This
is mostly ordinary rows tightened from ~150 s to ~70 s, not late-NM rescues. The late-NM rows are where ADS-B is
blind `[St1 §4]`.

**The guard does not catch everything.** 141 of 914 rows get worse; 27 of them have an ADS-B error > 120 s, and
those 27 cost 3.9e6 s² against a gain of 21.9e6 s². Examples: EDDM LHX9YX, where the model was +64 s off and
the guarded ADS-B +901 s; EHAM UAL908, −481 → −876; LIRF BAW8CF, −538 → −1,186 `[G S3]`.

**Lane B**, rows with no `AOBT_3`, 202 on the day. Only 12 have a pushback, and the model is already at RMSE 190 on
them; the huge lane-B errors (RMSE 1,136) are on the rows ADS-B does not see. Scaled to the OOF's 8,378 lane-B rows,
the gain is **−0.04 s pooled OOF**, which is negligible `[G S3]`.

## 4. Out-of-sample, label-free: the 2026-07-20 day against the v14 submission `[G S4]`

The 2026 day was not used to set the rules, and it has no BLOCK. What can be compared is the disagreement
`dis = model prediction − ADS-B prediction`:
- in 2026, v14's submitted prediction against the pushback, bias-corrected with 2025's per-airport medians;
- in 2025, the OOF prediction against the pushback.

| ADEP | n 2025 | RMS dis 2025, inside guard | n 2026 | RMS dis 2026, inside guard | guard trips 2025 / 2026 |
|---|---:|---:|---:|---:|---:|
| EDDF | 62 | 204.0 | 22 | 196.5 | 1.6 % / 0 % |
| EDDM | 279 | 161.4 | 294 | **118.7** | 0 / 0 |
| EGLL | 5 | 351.1 | 380 | 245.7 | 0 / 0.3 % |
| EHAM | 328 | 146.5 | 363 | **129.8** | 0 / 0 |
| LEBL | 144 | 147.8 | 184 | 175.4 | 0 / 1.1 % |
| LEMD | 38 | 160.7 | 109 | 222.8 | 2.6 % / 0 |
| LSZH | 51 | 185.4 | 74 | 176.9 | 0 / 0 |
| pooled | 914 | 162.6 | 1,428 | 182.4 | 0.33 % / 0.28 % |

- **No sign of more gross failures out of sample.** The guard trips as often on 2026 as on 2025. The share with
  `|AOBT_3 − push| > 20 min` is 1.53 % in 2025 and 0.56 % in 2026.
- **But at EDDM and EHAM the model and ADS-B are closer in 2026**: 119 against 161 s, and 130 against 146 s.
  - `[X]` Either v14 is better on those flights in 2026 than the OOF was on 2025-07-21, a hard day (§1), or
    ADS-B's error shrank. The first is more likely.
  - Either way, the 2026 per-row gain at EDDM and EHAM is probably smaller than measured.
  - LEBL and LEMD go the other way.
- **A gain proxy, checked on 2025.**
  - Construction: inside the guard, if ADS-B's error `a` were independent of the model's error, then
    gain = E[dis²] − 2·E[a²].
  - On 2025 it **underestimates** the true gain: pooled 6,900 against 19,400 actual. It is negative at LEBL while
    the truth is +19,000.
  - `[X]` The model's error and ADS-B's error are positively correlated. A plausible cause is shared noise in the
    BLOCK stamp itself: when BLOCK is late or early, both predictions miss it in the same direction.
  - The proxy is therefore a **floor**, and a poor one per airport.
  - Two 2026 versions are used in §5:
    - **proxy26**, the raw proxy: pooled 13,700 per row, EGLL 40,700.
    - **adj26**, the 2025 gain moved by the change in disagreement, `actual25 + proxy26 − proxy25`. It gives
      EDDM 7,100, EHAM 12,700, LEBL 27,900, LSZH 43,800, LEMD 35,600 and EDDF 26,200.

## 5. Scale-up: lane-A SSE → pooled OOF → board `[G S6]`

**Method.**
1. ΔSSE = Σ over airports of n_A,OOF × coverage × gain per covered row. Coverage is weighted by the board's
   January/July split (152,719 / 192,122).
2. Lane-A RMSE = √((SSE_A − ΔSSE) / 664,050).
3. Pooled OOF = √((SSE_pool − ΔSSE) / 672,428).
4. **Board = 0.747 × Δpooled OOF** `[MAC §8.2, n = 1]`.

The "direct" column applies ΔMSE on the board's own lane-A composition to 279.52, with no 0.747. It is shown only
as a cross-check.

Airports with fewer than 30 covered labelled rows borrow the pooled gain of the others, or get 0 where stated.
EGLL is one of them, with 5 rows.

**Each driver moved one at a time.** The base case uses 2026-07-20 coverage in both months, the measured point gain
(absolute) and the 2026 proxy for EGLL:

| variant | covered lane-A rows | SSE removed (lane A) | Δ lane A | Δ pooled OOF | **Δ board (×0.747)** | board direct |
|---|---:|---:|---:|---:|---:|---:|
| base | 23.6 % | 13.4 % | −14.8 | −11.0 | **−8.25** | −10.7 |
| January coverage half of July's `[X]` | 18.4 % | 10.4 % | −11.4 | −8.6 | −6.39 | −8.3 |
| January coverage zero | 13.2 % | 7.5 % | −8.1 | −6.1 | −4.55 | −5.9 |
| 2025-07-21 coverage instead | 14.8 % | 6.5 % | −7.1 | −5.3 | −3.98 | −5.2 |
| min(two days) coverage | 14.0 % | 6.1 % | −6.7 | −5.0 | −3.74 | −4.9 |
| max(two days) coverage | 24.4 % | 13.8 % | −15.2 | −11.4 | −8.49 | −11.1 |
| gain scaled to pooled error level (§1) | 23.6 % | 10.2 % | −11.2 | −8.4 | −6.26 | −8.1 |
| gain at the lower 95 % bound | 23.6 % | 9.8 % | −10.7 | −8.0 | −5.99 | −7.8 |
| gain at the upper 95 % bound | 23.6 % | 17.3 % | −19.3 | −14.3 | −10.70 | −14.0 |
| EGLL at the pooled measured gain, not the proxy | 23.6 % | 10.2 % | −11.2 | −8.4 | −6.27 | −8.2 |
| EGLL and the small airports at 0 | 23.6 % | 7.2 % | −7.8 | −5.9 | −4.40 | −5.7 |
| all gains from the 2026 proxy (a floor) | 23.6 % | 8.7 % | −9.5 | −7.2 | −5.35 | −7.0 |
| gains adj26, EGLL at the pooled gain | 23.6 % | 10.8 % | −11.9 | −8.9 | −6.64 | −8.7 |
| *unguarded replacement, for contrast* | 23.6 % | **−28.2 %** | +28.2 | +21.9 | **+16.39** | +22.3 |

**The range:**

| end | construction | Δ lane A | Δ pooled OOF | **Δ board** |
|---|---|---:|---:|---:|
| LOW′ | 2026 proxy gain (a floor); min(days) coverage; January half; small airports 0 | −1.36 | −1.03 | **−0.77** |
| LOW | lower 95 % gain; scaled; min(days) coverage; January half; EGLL and small airports 0 | −2.39 | −1.81 | **−1.35** |
| CENTRAL | point gain, scaled; EGLL at the pooled gain; 2026-07-20 coverage in July, half in January | −7.01 | −5.28 | **−3.94** |
| CENTRAL′ | adj26 gain; EGLL at the pooled gain; 2026-07-20 coverage in July, half in January | −9.17 | −6.89 | **−5.15** |
| HIGH | point gain, absolute; max(days) coverage; January = July | −15.22 | −11.36 | **−8.49** |
| HIGH+ | upper 95 % gain; max(days) coverage; January = July | −20.17 | −14.97 | **−11.18** |

**Where the central lands.** Share of the SSE removed: EDDM 27 %, EHAM 23 %, EGLL 22 %, LSZH 10 %, LEBL 9 %,
LEMD 8 %, EDDF 1 %. LFPG, LIRF and LTFM contribute nothing. `[X]` The lowest-RMSE airports carry most of the gain.
That is not a contradiction: the ADS-B error there (26–30 s MAE) is far below even a good model's.

**What drives each end.**
- **LOW** needs the lower of the two days' coverage everywhere, and the loss of the EGLL upside.
  - Coverage at EGLL was 0.8 % on the 2025 day.
  - It is also the only end that prices EGLL's 5 labelled rows as worth nothing.
- **HIGH** needs coverage to hold at the better day's level through January, and the hard day's absolute s² gain.
  - `[X]` The second is unlikely, given §1's scaling and §4's EDDM/EHAM narrowing.
- **The centre is sensitive to January** (at base: −8.3 s with January = July, −6.4 s at half, −4.6 s at zero)
  **and to EGLL** (≈ 2 s at base).
  Neither has a single labelled measurement.

## 6. The in-sample rules: what could be checked, and what it does to the range

**It is not possible to re-measure the accuracy out of sample from what is on disk.** Stage 1 settled the stand rule,
the pushback gate and the dwell guard while looking at the whole of 2025-07-21's errors against BLOCK `[St1 §2.3]`.
No subset of that day was held back, and it is the only labelled day on disk. Three partial checks were possible:

**6.1 Threshold sensitivity (still in-sample).** `[G S5]` The frozen detection was re-run on the 2025-07-21 fixes.
It reproduces all 926 stage-1 pushbacks exactly. Each threshold was then moved alone, on the aircraft the label-free
join had already chosen.

| config | covered | MAE vs BLOCK | > 10 min off | guarded ΔMSE per row | guarded ΔSSE (1e6 s²) |
|---|---:|---:|---:|---:|---:|
| **frozen** | 914 | 71.2 | 8 | **19,398** | 17.73 |
| stand radius 15 / 20 / 35 / 50 m | 911 / 922 / 837 / 702 | 72.6 / 70.7 / 79.0 / 89.0 | 12 / 9 / 8 / 6 | 16,697 / 20,390 / 21,121 / 21,517 | 15.2 / 18.8 / 17.7 / 15.1 |
| pushback share 0.3 / 0.4 / 0.6 / 0.7 | 950 / 930 / 862 / 826 | 71.3 / 70.3 / 72.6 / 73.2 | 9 / 8 / 8 / 7 | 18,836 / 19,366 / 19,674 / 20,418 | 17.9 / 18.0 / 17.0 / 16.9 |
| pushback window 90 / 120 / 300 s | 934 / 933 / 571 | 70.7 / 70.9 / 83.3 | 8 / 8 / 7 | 19,032 / 19,081 / 20,213 | 17.8 / 17.8 / 11.5 |
| backwards angle 90° / 150° | 937 / 752 | 73.9 / 73.2 | 9 / 6 | 18,485 / 20,510 | 17.3 / 15.4 |
| dwell guard 600 / 1,800 s / off | 911 / 917 / 917 | 70.4 / 75.4 / 75.4 | 7 / 10 / 10 | 20,169 / 18,923 / 18,923 | 18.4 / 17.4 / 17.4 |
| silence 300 / 1,200 s | 811 / 923 | 64.1 / 72.6 | 5 / 9 | 19,247 / 19,400 | 15.6 / 17.9 |
| **no pushback gate, no dwell guard** | 2,360 | 308.5 | 299 | **−20,977** | **−49.5** |

- **The frozen point is not a sharp optimum.** The per-row gain stays within 16.7k–21.5k across every perturbation.
  Total ΔSSE stays within 15–19e6, apart from the 300 s window, which loses rows but not per-row gain.
- **The structural choice is what matters.** Without the pushback gate, the gain is negative even with the model
  guard.
- `[X]` The threshold values are not where the in-sample optimism lives. It lives in whether the same structure
  keeps catching a real pushback on other days and at other airports, and a same-day sweep cannot test that.

**6.2 Failure stress.** `[G S3]` The loss from the 27 rows whose guarded ADS-B error is above 120 s was multiplied
by k, as if detection failures were k times as common out of sample:

| k | ΔMSE per row |
|---:|---:|
| 1 (as measured) | +19,400 |
| 2 | +15,100 |
| 3 | +10,800 |
| 5 | +2,200 |

The gain survives a tripling and does not survive a quintupling.

**6.3 The 2026 day, which is out of sample for the rules but unlabelled.** `[G S4]` Gross failures do not rise: the
guard trips on 0.28 % of flights against 0.33 %. Moderate failures, 2–15 min, are invisible without labels. `[X]`
They are exactly what the k multiplier stands for.

**Effect on the range.** The LOW end is built from the lower CI bound and the lower coverage, which on the base case
corresponds roughly to k ≈ 2–3 `[X]`. A k of 5 on top of LOW would take the board gain to about −0.2 s. That is the
realistic floor if the rules are badly overfit `[X]`.

## 7. What one day can bear, and the shrinkage record

- **Coverage.** Binomial sampling error on one day is small: ±2 pp at EDDM with n = 487. The problem is the
  day-to-day feeder variance, which one or two days cannot estimate. Between two days, EGLL ground coverage went
  41 % → 99 % and its pushback rate 0.8 % → 58 %; LIRF ground coverage went 94 % → 0 % `[St1 §0.7]`.
  - `[X]` The two-day min/max used in §5 need not bracket the 62 scored days.
  - Nothing at all is known about January: not reception, not de-icing, and not whether the 15-min dwell guard drops
    de-iced flights `[St1 §8.4]`.
  - The January coverage multiplier (0.5, and 0 as a variant) is a placeholder.
- **Accuracy.** There are 914 rows on one day, 8 airports, and only 4 airports with ≥ 50 rows. EGLL, the second or
  third largest term in every scenario, has 5 labelled rows.
- **The record.** v15's dead-column removal went from −0.13 to +0.02. C1 went from −0.36 on the forward fold to
  −0.08 on `jul_fwd`. fl+seq went from −2.53 on the forward fold to −0.26 on the board. Each was measured on far
  more rows than this.
  - `[X]` Two things differ here:
    - The mechanism is a direct replacement on rows where the per-row error halves, not a learned effect.
    - The CI excludes zero by a wide margin under both bootstraps.
  - Those were the failure modes behind the earlier shrinkage: fold noise and selection among many candidates.
  - But this estimate has failure modes of its own: coverage variance, out-of-sample rules, and a better 2026 model
    on these rows (§4).
  - **Planning number:** assume the project's ~10× shrink applies. The central −3.9 s becomes **≈ −0.4 s**. That is
    still larger than any board gain shipped this month. The LOW end (−0.8 to −1.4 s) shrunk the same way would be
    −0.1 s.
  - `[X]` The estimate is good enough to decide that the ingest is worth its cost (≈ 3 h, no disk pressure). It is
    not good enough to predict the board number.

## 8. Recommendation: INGEST-A-SMALLER-SET-FIRST

**What the use would be** `[X]`. A post-hoc override on the submission:
- where a gated pushback exists on a lane-A row, predict `MVT − (t_push − b_airport)`;
- unless that disagrees with the model by more than 900 s.

This needs no retraining. It needs:
- the scored-period ADS-B (Jan and Jul 2026, the 220 GB);
- a few labelled 2025 days, to calibrate `b` and the guard out of sample and to validate.

It does **not** need a 2025-wide training ingest (≈ 1.3 TB). Using ADS-B as a model feature instead would need that
ingest, and should wait.

**Smaller set first: 7 days, ≈ 25 GB, ≈ 20–25 min at stage 1's rates** `[St1 §1]`, `[X]`.
- **3 labelled 2025 days**, to test the frozen rules, bias and guard out of sample:
  - one July weekday other than the 21st (2025-07-13 is already sized, 3.29 GB);
  - one January 2025 weekday, for the winter and de-icing check;
  - one November 2025 day, in the forward fold.
- **4 scored-period days**, a coverage census: 2026-01-19 (already sized, 3.12 GB), a second January 2026 day, and
  two more July 2026 days, for EGLL's feeder stability.
- **Airports.** The extract keeps all 10 airports at no extra cost. Evaluate EDDM, EHAM, LEBL, LSZH, EGLL and LEMD.
  LFPG, LIRF and LTFM have no pushbacks on either day and should not be counted on.
- **Go criteria**, all measured with rules, bias (from 2025-07-21) and guard (900 s) frozen:
  1. On the three held-out labelled days pooled, the guarded gain is ≥ 5,000 s² per covered lane-A row at the GO
     airports (19,400 here). Below ≈ 5,000 the CENTRAL falls to about −1 s `[X]`.
  2. On the four census days, lane-A pushback coverage is ≥ 10 % pooled in January and ≥ 15 % in July (23.6 % on
     2026-07-20).
- **If both hold:** ingest all 62 days.
- **If only July holds:** ingest July 2026 only, 31 days and ≈ 115 GB.
- **Kill:** a guarded gain below ≈ 2,000 s² per row out of sample (the k = 5 level of §6.2), or coverage below 5 %
  on the census days.
