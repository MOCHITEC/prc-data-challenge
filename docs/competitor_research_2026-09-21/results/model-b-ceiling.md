# How much of model B's error is irreducible (2026-09-25)

**Script (read-only; nothing written to `src/`, `configs/`, `outputs/` or `data/`; no submission made;
no production model refitted):**
`docs/competitor_research_2026-09-21/scripts/model_b_ceiling.py`, run as
`POLARS_MAX_THREADS=2 .venv/bin/python docs/competitor_research_2026-09-21/scripts/model_b_ceiling.py [S1 … S8]`.
Every number below carries the section (`S1`…`S8`) that prints it. **The whole file is 18 seconds**
(`S1` 7.8 s — the only section that fits anything, a 22,470-row importance model; `S4` 3.8 s, `S5` 2.8 s,
everything else under 1.5 s). Peak RSS 1.7 GB. This is the lane-B counterpart of
`results/model-a-ceiling.md`, and it reuses that note's estimators wherever they transfer.

**Inputs.** `data/work/feat_train.parquet` (2,085,047 departures, of which **22,470** are unmatched and
all 22,470 are labelled), `experiments/oof_main_gap-62dd7c6e88.parquet` + its `_folds` sidecar
(672,428 pooled OOF rows over folds jan/jul/forward), `configs/main_gap.yaml`.

**Notation.**

| symbol | definition | visible at prediction time? |
|---|---|---|
| `y` | `TAXITIME_SEC_mvt` = MVT − BLOCK | no — it is the label |
| `ms` | `mvt_minus_sched` = MVT − SCHED | **yes**, 0 % null at all ten airports |
| `bs` | `ms − y` = BLOCK − SCHED | no — **the label in disguise**, and the quantity measured |
| `e` | `oof − y` | no — model B's OOF residual |

**lane B** = `mvt_minus_aobt3` is null, the routing rule at `src/prc/train.py:686`.

**One design choice, stated first.** `ms` is exactly visible, so `Var(y | visible) ≡ Var(bs | visible)`
identically, while `Var(y | cell) > Var(bs | cell)` for any cell that does not hold `ms` fixed.
Measuring the spread of **`bs`** removes the largest visible axis for free — the same device as lane A's
use of `g`. **`bs` is never a key and never a feature; it is only ever the quantity measured.** [S] `S1`

---

## 0. The answer in eight sentences

1. **Lane B's RMSE of 1,606.89 is, in effective terms, made of 7.6 rows.** Its Kish effective sample
   size — `(Σe²)²/Σe⁴`, the number of equal rows that would give the same sampling variance for the MSE
   — is **7.58**, against lane A's **19,738**. One row is 32.0 % of lane-B SSE; the top two are 47.2 %; the
   nineteen impossible labels of §2.1 are 63.5 %. A nonparametric bootstrap puts the lane's own RMSE at **1,606.9 [1,088.2, 2,193.2]**.
   **Every number in this note inherits that interval, and lane B's "1,606.89" should never again be
   quoted to six figures.** [S] `S2`
2. **Those nineteen rows are impossible labels, not hard flights.** They carry `y` of 40,262 – 88,132 s —
   taxi times of 11 to 24 hours. The same criterion catches **1 row in 2,062,577** on the matched lane and
   **40 in 22,470** on the unmatched one: a **3,672× enrichment**. A broken `BLOCK_TIME_UTC_mvt` and a
   missing NM flight plan are not independent events, and that single fact is what makes lane B's RMSE
   what it is. [S] `S2`
3. **Two of them — 47.2 % of the whole lane — are invisible, and provably so.** The two LFPG rows carry
   `ms` of 1,740 and 2,043 s, dead in the middle of the lane's distribution (p50 5,517). Nothing visible
   marks them, because the corruption is in `BLOCK_TIME_UTC_mvt`, which is one of the two columns blanked
   on every scored row. The conditional mean — the thing the floor is *defined* as — moves by `q·jump`
   = **36.8 s** on a jump of 69,332, so **a perfect conditional-mean predictor eats essentially the whole
   squared error on these rows. That is the floor, not a modelling failure.** [S] `S6.2`
4. **The floor, measured: 1,586.2 [1,573.8, 1,597.9] against today's 1,606.89 — 20.7 s recoverable, 2.6 %
   of lane-B SSE.** Assembled block by block (§5.2), because carrying one ρ across this lane is the
   principal way to get it wrong: the naive version reads 1,372.9 and claims 234 s, all of it by crediting
   the impossible labels with a recoverable share they cannot have. [S] `S5.2`
5. **Priced through `dBoard/dOOF = 0.747`, the whole remaining prize on lane B is −1.1 s of board
   (−1.5 s at ratio 1.000), range −0.5 to −1.8 s.** Even the arithmetic limit — both unmeasurable blocks
   predicted perfectly, 88 % of lane-B SSE deleted — is board **267.4**, and lane B taken to exactly zero
   is **230.4** at the measured ratio, not 213.78. [S] `S7`, `S5.2a`
6. **The bias is real, it is large, and it is not sellable.** Per-airport mean residuals run +688.7 (LIRF)
   to −178.2 (EDDM), but an oracle per-airport correction removes only **1.4 %** of lane-B SSE, and the
   *honest* out-of-fold version of every key tested — airport, airport × ms-quintile, airport × runway,
   airport × stand, airport × month — is **worse than doing nothing** and no better than its own shuffled
   control. The first moment exists cross-sectionally (a random 5-fold split buys −30 s on the non-broken
   lane) and **does not survive a month-blocked split** (+3.2 s). That is `P1B` generalised past LIRF.
   [S] `S3`
7. **Two independent routes agree on the one block where both can be run.** On the nine non-LIRF airports
   with the impossible labels removed (7,839 rows, 5.0 % of the problem's total SSE), the twin-residual
   covariance floor is **508.5** and the learning curve — read off `missing-clock.md` §2.2's two training
   sizes, no refit — lands at **515.6 – 525.6**. They agree to 1.4 – 3.4 %, share no machinery, and the
   ordering is right (a finite-`n` curve point must sit at or above the floor). [S] `S5.2`, `S8`
8. **[X] The honest headline: the −4.9 to −5.5 s missing-clock figure is that route's ceiling, not the
   lane's — but the lane's ceiling at the current feature set is *smaller*, not larger, and the two do not
   contradict.** Re-priced on the scored composition this note measures the sight route at **−3.6 to
   −4.7 s** against `missing-clock.md`'s −4.9 to −5.5: same order, same sign. What changes is the meaning.
   The lane's headroom is not hiding somewhere the clock route missed. **47 % of lane B is two label
   corruptions that nothing visible can see; a further 16 % is seventeen more at LIRF that the `ms`
   reparameterisation already largely handles; 25 % is the rest of the LIRF switch, whose perfect version
   is worth −2.7 s of board and whose achievable version closed at AUC 0.9201; and the 12 % that is left
   is data-limited on 22,470 rows that are all the rows there are.** Lane B is close to its floor, and the question can be closed.

---

## 1. What is being estimated, and which way each route errs

Squared error is minimised by the conditional mean, so with `μ(x) = E[y | x]`

```
MSE  =  E[(yhat(x) − y)²]  =  E[(yhat(x) − μ(x))²]  +  E[Var(y | x)]
                               \__________________/     \___________/
                                approximation error       THE FLOOR
                                   (recoverable)          (irreducible)
```

**Every estimator here bounds the floor from above**, as in lane A, and for the same reason: a lower bound
would require knowing that some particular variation is unpredictable, which is not estimable from data.

| route | § | what it does | verdict on lane B |
|---|---|---|---|
| 1 k-NN semivariance of `bs` | 4 | spread of `bs` between visible twins | **cannot bound** — reads *above* the model at every density |
| 2 twin-residual covariance | 5 | `ρ = Cov(e_i,e_j)/MSE`, floor `= RMSE·√(1−ρ)` | **the load-bearing route**, 1,586.2 |
| 3 learning curve | 8 | two training sizes already measured, no refit | 515.6 – 525.6 on the one block it covers |
| 4 matched-twin ladder | — | **not run.** Lane A's ladder needed cells of size ≥ 2 over 2.06 M rows. On 22,470 rows the finest level of that ladder (L8, airport × operator × stand × runway × dest × 3 h × type × delay quintile) would have more cells than rows, and `AIRCRAFT_OPERATOR_flt` — an L2 key — is **100 % null on lane B**. There is nothing to run. [X] That is a finding about the lane, not a gap in this note. |
| 5 oracle cell-mean recovery | 3 | what a first moment is worth, in-sample and honestly | nothing survives out of fold |

### 1.1 The baseline, reproduced

`S1`, pooled OOF, 672,428 rows, RMSE **277.5398**:

| lane | n | RMSE | bias | median | share of total SSE | **Kish n_eff** |
|---|---:|---:|---:|---:|---:|---:|
| A (matched) | 664,050 | 213.1266 | +4.85 | +14.7 | 0.5823 | **19,737.9** |
| B (unmatched) | 8,378 | **1,606.8929** | +22.79 | +49.0 | 0.4177 | **7.6** |

Identical to the brief to four decimals. **[X] The last column is the finding that reorganises everything
else.** Lane A's routes had 2,600× the effective rows; lane B's have 7.6.

### 1.2 Lane B restated in the space the floor lives in

`S1.1`:

| quantity | value |
|---|---:|
| sd(`y`) | 3,774.98 |
| sd(`ms`) — **visible** | 5,887.03 |
| sd(`bs`) — the axis the metric cannot remove | 5,304.66 |
| robust sd(`bs`) = IQR/1.349 | 3,652.15 |
| model B RMSE | **1,606.89** |
| fraction of Var(`bs`) explained | **0.9082** |

**[X] Model B already explains 90.8 % of the variance of the quantity it is fitting, against model A's
71.5 %.** That is not because model B is better; it is because `bs` on this lane is a near-constant
punctuated by rare enormous jumps, and explaining "usually about zero" gets you 90 % for free. It is the
first sign that lane B's error is a tail problem, not a mean problem.

### 1.3 Per airport

`S1.2`:

| ADEP | n | RMSE | bias | median | sd(`bs`) | SSE share | n_eff |
|---|---:|---:|---:|---:|---:|---:|---:|
| **LFPG** | 1,565 | 2,622.28 | −113.0 | +31.2 | 5,812.0 | **0.4975** | **2.0** |
| **LIRF** | 537 | 4,059.20 | +688.7 | +662.0 | 5,026.5 | **0.4090** | 26.7 |
| EGLL | 647 | 1,011.34 | −41.9 | +77.9 | 2,909.6 | 0.0306 | 19.9 |
| LTFM | 1,297 | 503.15 | −28.0 | +41.9 | 2,573.9 | 0.0152 | 23.1 |
| EHAM | 1,353 | 475.00 | +43.2 | +58.1 | 7,291.7 | 0.0141 | 6.7 |
| LSZH | 722 | 609.62 | +75.5 | +91.7 | 3,232.7 | 0.0124 | 7.4 |
| EDDM | 402 | 724.56 | −178.2 | −10.1 | 3,547.8 | 0.0098 | 17.0 |
| EDDF | 787 | 400.78 | +1.0 | +43.9 | 6,319.7 | 0.0058 | 11.0 |
| LEBL | 652 | 362.45 | +30.7 | +32.0 | 4,237.6 | 0.0040 | 44.0 |
| LEMD | 416 | 295.75 | −1.6 | +15.2 | 2,511.8 | 0.0017 | 48.3 |

**[X] LFPG's 49.8 % of lane-B SSE has an effective sample size of 2.0.** The brief's framing —
"LIRF **+688.7**, EDDM **−178.2** … lane B has real per-airport bias" — is right about the bias and
points at the wrong airport for the *error*: LFPG, whose bias is a mild −113.0, carries more SSE than
LIRF does, and carries it in two rows.

### 1.4 Lane B's live visible column set

`S1.3`, the brief's "enumerate it rather than assuming". Model B is advertised as **92 numeric + 10
categorical = 102** columns. **Fourteen are dead on every row it ever sees:**

* all-null (the `_flt` block and its derivatives): `stand_aobt_turn`, `eobt_schedule_revision`,
  `delay_neighbor_deviation`, `planned_flight_s`, `WK_TBL_CAT_flt`, `AIRCRAFT_OPERATOR_flt`,
  `MARKET_SEGMENT_flt`, `FLIGHT_TYPE_flt`.
* constant: `lobt_null` ≡ 1, `eobt1_null` ≡ 1, `runway_null`, `diverted`, `actype_mismatch`, `anchor_ok`.

**==> 88 live columns, not 102.** Two things worth recording. `missing-clock.md` §1.2 found **three**
constant columns on model B's feature list; on this config there are **six**, and `runway_null`,
`diverted`, `actype_mismatch` and `anchor_ok` are new to that list. And of the fourteen dead columns,
**eight are the `_flt` block and its derivatives** — so every column that carries operator, market
segment, flight type or weight category is not merely weak on lane B, it is absent. Harmless in itself —
a constant column cannot split — but it is why §1's route 4 cannot be built (`AIRCRAFT_OPERATOR_flt` is
an L2 key in lane A's ladder). Four further columns are live but heavily null:
`stand_arrival_gap` 20.2 %, `stand_arrival_type_match` 25.9 %, `gust_ms` 97.3 %, `gust_ms_prev1h` 97.0 %.

### 1.5 The matching metric, chosen rather than assumed

`S1.4`. A 300-round LightGBM on lane B's own 22,470 rows, gain share:

| column | gain | cum |
|---|---:|---:|
| `mvt_minus_sched` | 53.87 % | 53.9 % |
| `RUNWAY_mvt` | 8.63 % | 62.5 % |
| `queue_ahead` | 7.35 % | 69.9 % |
| `STAND_mvt` | 5.77 % | 75.6 % |
| `stand_rwy_share` | 4.23 % | 79.9 % |
| `ADEP_mvt` | 2.59 % | 82.4 % |
| `ADES_mvt` | 1.96 % | 84.4 % |
| `stand_arrival_type_match` | 1.82 % | 86.2 % |
| `stand_arrival_gap` | 1.27 % | 87.5 % |

The narrow metric is twelve columns carrying **87.2 %** of the gain — deliberately the same shape as lane
A's twelve-at-85 %. Blocking is on `ADEP_mvt` and `STAND_mvt` is a metric column, because 22,470 rows over
~2,500 stands cannot be searched inside a stand.

**[S] A free sanity check falls out.** The four built-in noise controls score 0.12 / 0.21 / 0.28 / 0.35 %
of gain, and **everything below rank ~20 is at or under them** — `temp_c` 0.27 %, `wind_ms` 0.23 %,
`dest_distance_km` 0.22 %. On lane B the feature set is one column (`ms`, 53.9 %) plus a handful of
geometry, and then noise.

---

## 2. Concentration: how few rows is this RMSE made of?

`S2.1`, lane B, 8,378 OOF rows, SSE 2.1633e10:

| k worst rows | share of lane-B SSE | lane-B RMSE without them | share of **total** OOF SSE |
|---:|---:|---:|---:|
| 1 | 32.02 % | 1,325.00 | 13.37 % |
| **2** | **47.17 %** | 1,168.15 | **19.70 %** |
| 5 | 58.32 % | 1,037.69 | 24.36 % |
| 10 | 66.12 % | 935.84 | 27.62 % |
| **19** | **63.45 %** *(see below)* | **972.60** | **26.50 %** |
| 33 | 74.72 % | 809.49 | 31.21 % |
| 100 | 84.40 % | 638.42 | 35.25 % |
| 1000 | 98.03 % | 240.37 | 40.94 % |

(19 is the impossible-label set, not the top-19 by error; the two orderings overlap but are not identical.)

`S2.2`: jackknife — **the worst single deletion moves the lane's RMSE by 281.89 s**, and the jackknife sd
of the lane RMSE is **315.72 s**. Bootstrap 95 % CI **[1,088.2, 2,193.2]**.

### 2.1 The nineteen rows

`S2.3` / `S2.4`. Defined by `y > 40,000 s` — a taxi time of 11 hours is not a taxi time:

| block | n | RMSE | SSE share | n_eff | bias |
|---|---:|---:|---:|---:|---:|
| impossible labels (`y` > 11 h) | 19 | 26,877.5 | **0.6345** | 3.1 | −14,517.8 |
| &nbsp;&nbsp;of which **invisible** (`ms` < 2 h) | **2** | 71,425.5 | **0.4717** | **1.8** | −70,234.8 |
| LIRF, excl. impossible | 520 | 3,200.3 | 0.2462 | 74.9 | +971.6 |
| LFPG, excl. impossible | 1,563 | 597.6 | 0.0258 | 37.1 | −23.3 |
| other 8 airports, excl. impossible | 6,276 | 567.8 | 0.0935 | 75.1 | −0.3 |

**The threshold is not load-bearing.** `S2.4` sweeps it two ways and the alternative criterion —
`bs < −10,000`, a BLOCK stamp placed hours *before* the scheduled off-block — catches **17** rows and
**63.6 %**, against 19 and 63.5 %. The two criteria disagree about *which* rows and agree almost exactly
about *how much*. `S2.1` reaches the same rows with no threshold at all.

**Enrichment** (`S2.4`): `y > 40,000` on the **matched** lane is **1 row in 2,062,577** (0.0000 %); on the
unmatched lane it is **40 in 22,470** (0.1780 %). **3,672×.**

### 2.2 The two that matter

`S2.4`, the head of the table:

| ADEP | fold | `y` | `ms` | `bs` | `oof` | `e` |
|---|---|---:|---:|---:|---:|---:|
| **LFPG** | jan | 84,240 | **1,740** | −82,500 | 1,017.7 | **−83,222.3** |
| **LFPG** | jan | 58,206 | **2,043** | −56,163 | 958.7 | **−57,247.3** |
| LIRF | jul | 87,186 | 58,563 | −28,623 | 57,107.8 | −30,078.2 |
| LIRF | forward | 87,480 | 56,520 | −30,960 | 59,103.9 | −28,376.1 |
| … | | | | | | |
| LIRF | jul | 40,262 | 40,261 | −1 | 40,060.8 | **−201.2** |

**[S] Read the `ms` column.** The two LFPG rows look like completely ordinary flights — 29 and 34 minutes
from schedule to wheels-up, against a lane median of 5,517 s. Their `BLOCK` stamp is placed 23 and 15
hours before the schedule. The LIRF rows are the opposite: `ms` of 40,000 – 93,000 s marks them as
extraordinary from across the room, and **the model already predicts most of them well** — the bottom of
that table has errors of 161, 172, 201 s on labels of 40,000 +, because the `unmatched_sched_reparam`
switch is doing its job.

**[X] So "the 19 impossible labels" is really two facts with opposite consequences.** Seventeen at LIRF
are visible and largely handled. Two at LFPG are invisible and are 47 % of the lane.

---

## 3. Bias versus spread

`S3`. For a cell `K` with `n_K` rows and mean residual `b_K`, removing the cell mean removes exactly
`n_K·b_K²` of SSE **in sample**. That is free and, on this lane, largely fake. Every key is reported six
ways: in-sample and its shuffled control; a **random 5-fold** honest arm and its control; the
**production time-fold** honest arm and its control.

**Why two honest arms.** The production folds are month-blocked (jan / jul / forward), so any key
containing `month` is collinear with them and its time-OOF arm is structurally empty. The random split
asks *does this structure exist*; the time split asks *does it transfer forward*. `P1B` is the worked
example of a candidate that passes the first and fails the second.

`S3`, **excluding the 19 impossible labels** (n = 8,359, base 972.60):

| key | cells | in-samp | in ctl | rand5 | r5 ctl | time-OOF | t ctl |
|---|---:|---:|---:|---:|---:|---:|---:|
| K1 ADEP | 10 | 940.47 | 970.40 | **942.53** | 972.33 | **975.79** | 972.30 |
| K2 ADEP × ms-quintile | 50 | 923.79 | 968.48 | 933.77 | 974.87 | 989.83 | 977.02 |
| K3 ADEP × month | 40 | 921.76 | 969.45 | 928.93 | 974.75 | *collinear* | *collinear* |
| K4 ADEP × RUNWAY | 57 | 930.73 | 967.87 | 939.42 | 976.85 | 987.35 | 977.15 |
| K5 ADEP × STAND | 1,729 | 800.52 | **867.68** | 1,072.19 | 1,090.01 | 1,099.75 | 1,108.85 |
| K6 ADEP × ms-q × month | 200 | 889.05 | 961.45 | 933.73 | 984.25 | *collinear* | *collinear* |
| global constant (bias only) | 1 | 971.00 | — | 971.15 | — | 972.19 | — |

**[S] Three readings, in order of importance.**

1. **Not one honest arm beats base on the time split.** Every time-OOF number is ≥ 972.60, and every one
   is at or worse than its own shuffled control. **There is no first moment on lane B that transfers
   forward in time.**
2. **But the structure is real.** K1 ADEP on the random split reads **942.53 against a control at 972.33**
   — a genuine −30 s that its control does not find. The per-airport bias the brief names exists; it is
   cross-sectional, and the month-blocked fold destroys it. **[X] That is `P1B`'s finding generalised from
   LIRF's season to the whole lane, and it is the single most transferable lesson here.**
3. **K5's in-sample 800.52 is degrees of freedom, and its own control says so** (867.68 from shuffled
   labels, on the same 1,729 cells). Without the in-sample control a reader sees "an oracle stand
   correction recovers 18 %" and believes it. This is the `missing-clock.md` §7 failure mode, caught by a
   control that was in the design.

`S3.1`, the split in SSE:

| | whole lane B | excl. the 19 |
|---|---:|---:|
| total SSE | 2.1633e10 | 7.9073e9 |
| removed by the global mean (**pure bias**) | 4.353e6 = **0.020 %** | 2.6069e7 = **0.330 %** |
| removed by **oracle** per-airport means | 2.9688e8 = **1.372 %** | 5.1387e8 = **6.499 %** |
| within-airport (**spread**) | 2.1336e10 | 7.3934e9 |

> **[S] Bias is 0.02 % of lane-B SSE. An oracle per-airport correction is 1.4 %. Everything else — 98.6 %
> — is spread.** Lane B's error is not a first moment, and the +688.7 / −178.2 per-airport means that look
> so alarming are worth, at the absolute oracle limit, eleven seconds of the lane's 1,607.

---

## 4. Route 1 — k-NN semivariance, and why it cannot bound this lane

`S4`. `γ(d) = E[(bs_i − bs_j)²]/2` over 1-NN twins inside an airport block, read beside model B's OOF
RMSE on exactly the same rows. All 22,470 unmatched labelled rows are paired (a twin only has to supply a
label). 1-NN distance quantiles p5 0.41, p25 1.13, p50 1.50, p95 2.19.

| `d` bin | rows | in OOF | γ(d)^½ | **model RMSE, same rows** | ratio |
|---|---:|---:|---:|---:|---:|
| (0.00, 0.25] | 429 | 125 | 284.84 | **169.86** | 0.596 |
| (0.25, 0.50] | 1,116 | 321 | 476.17 | 206.63 | **0.434** |
| (0.50, 0.75] | 963 | 300 | 781.82 | 480.33 | 0.614 |
| (0.75, 1.00] | 950 | 312 | 937.89 | 779.35 | 0.831 |
| (1.00, 1.25] | 4,006 | 1,404 | 917.45 | 817.98 | 0.892 |
| (1.25, 1.50] | 3,729 | 1,348 | 1,760.18 | 1,234.12 | 0.701 |
| (1.50, 2.00] | 9,049 | 3,584 | 1,889.00 | 1,960.41 | 1.038 |
| whole lane B | 22,470 | 8,378 | 1,706.98 | **1,606.89** | 0.941 |

**Control** (`S4.3`): with `bs` permuted within airport, γ^½ reads **4,849.47** overall against a
within-ADEP sd(`bs`) of **4,773.81**. The neighbour search is not inventing structure — 1.6 % high, which
is the k-NN selection, not a leak.

**[X] Route 1 fails on lane B, and the failure is informative.** Its floor estimate sits **above** the
model at every density — 476 where the model scores 207, 1,707 where the model scores 1,607 — so it cannot
bound the floor below the model's own error and establishes nothing about headroom. Post-stratified
(`S4.2`) the gap is −215 s at `d ≤ 0.5` and −293 s at `d ≤ 1.0`, in the wrong direction throughout.

**Why it fails here and worked on lane A.** Lane A's metric contained the *measured off-block clock*
(`aobt3_minus_eobt1` 23.6 % + `mvt_minus_aobt3` 14.8 % of gain), which pins `g` down; two lane-A rows at
`d ≈ 0` really are near-replicates. Lane B's best column is `ms` at 53.9 %, and two lane-B rows agreeing on
`ms`, stand, runway, destination and type can still differ by whether the schedule fill fired or whether
the BLOCK stamp is broken — neither of which is in any column. **The metric does not resolve `x`, so
`(μ_i − μ_j)²/2` never vanishes.** That is a fact about lane B's feature set, and it is the same fact §1.5
records as "one column plus noise". [S] `S4`

---

## 5. Route 2 — the twin-residual covariance, and the floor

### 5.1 The estimator, and the correction the control forced

`S5`. `E[e_i e_j] = E[(yhat − μ)²]`, so `ρ = E[e_i e_j]/MSE` is the systematic, recoverable share and
`floor = RMSE·√(1−ρ)`. Six nearest neighbours per row, `d ≤ 1.0`, both rows in the OOF and the same fold:
**1,233 pairs**, against lane A's 140,520. **This route runs on 1.1 % of the data lane A had.**

**[S] The negative control failed on the first form of the estimator, and the fix is the finding.**
`S5.1`, partner residual permuted so no pair is a twin:

| form | ρ | 95 % CI | verdict |
|---|---:|---|---|
| `ρ_raw = E[e_i e_j]/MSE` | **+0.0661** | [+0.0057, +0.1218] | **EXCLUDES ZERO — fails** |
| `ρ_cen = E[(e_i−m)(e_j−m)]/MSE` | +0.0169 | [−0.0426, +0.0752] | contains zero — passes |

`E[e_i e_j]` contains the **square of the first moment**: if the model is biased by `m` on a population,
two *unrelated* rows already have `E[e_i e_j] = m² > 0`. On lane A that term was `4.85²/134.91²` = 0.13 %
and could be ignored. On lane B the paired rows carry a bias of +44.1 against an MSE of 198.7², and it is
worth 5 points of ρ. **[X] So the headline ρ is the centred one, and this is exactly the trap the brief
named — a first moment masquerading as recoverable error. §3 has already shown that first moment is not
recoverable out of fold at all, so counting it would have been wrong twice over.**

`S5.0`, ρ by distance on the rows that have twins:

| `d` bin | pairs | MSE^½ | bias | ρ_raw | **ρ_cen** | ρ_cen 95 % CI | n_eff |
|---|---:|---:|---:|---:|---:|---|---:|
| (0.00, 0.25] | 62 | 200.4 | +30.4 | 0.0341 | 0.0111 | [−0.407, +0.404] | 14.8 |
| (0.25, 0.50] | 405 | 154.6 | +17.8 | 0.1667 | 0.1535 | [+0.020, +0.287] | 79.7 |
| (0.50, 0.75] | 392 | 208.3 | +46.2 | 0.4039 | 0.3548 | [+0.158, +0.505] | 28.5 |
| (0.75, 1.00] | 374 | 228.2 | +72.6 | 0.3575 | 0.2564 | [+0.096, +0.394] | 37.4 |
| **all `d` ≤ 1.0** | **1,233** | **198.7** | +44.1 | 0.3192 | **0.2700** | **[+0.171, +0.360]** | 104.9 |

**[X] MSE^½ on the paired rows is 199 against the lane's 1,607.** The pairs are the *ordinary* rows; the
lane's error lives on rows with no twin at all. That is why §5.2 does not carry one ρ to the lane.

**Positive control** (`S5.1`) — the out-of-half mean of `y` over ADEP × ms-quintile, a legitimate visible
predictor, badly underfitted, on exactly the two blocks where §5.2 uses a measured ρ:

| block | arm | RMSE on block | pairs | ρ_cen | 95 % CI |
|---|---|---:|---:|---:|---|
| LFPG, excl. impossible | model B | 597.6 | 74 | 0.2748 | [+0.061, +0.519] |
| | crippled | 745.6 | 74 | **0.5781** | [+0.446, +0.675] |
| other 8, excl. impossible | model B | 567.8 | 1,149 | 0.1982 | [+0.102, +0.293] |
| | crippled | 599.7 | 1,149 | **0.2611** | [+0.193, +0.327] |

**[S] On both blocks the crippled predictor is worse and measures a larger systematic share. The
estimator sees systematic error when it is there.** **[X] Recorded rather than buried: pooled over the
whole lane the same positive control reads ρ_cen = −0.16, i.e. it fails.** That is not a defect of the
crippled arm; it is `n_eff = 7.6` again — a single pair containing an extreme row decides the pooled
number. It is the reason §5.2 is assembled block-wise and it belongs in §9's caveats.

### 5.2 The floor, assembled block by block

`S5.2`. `floor² = Σ_block SSE_block·(1 − ρ_block) / n`, with ρ = 0 where it cannot be measured:

| block | n | SSE | RMSE | pairs | ρ_cen | 95 % CI | block floor |
|---|---:|---:|---:|---:|---:|---|---:|
| 19 impossible labels | 19 | 1.373e10 | 26,877.5 | **0** | 0 *assumed* | — | 26,877.5 |
| LIRF, excl. impossible | 520 | 5.326e9 | 3,200.3 | **10** | 0 *assumed* | — | 3,200.3 |
| LFPG, excl. impossible | 1,563 | 5.583e8 | 597.6 | 74 | 0.2748 | [+0.061, +0.519] | 508.9 |
| other 8, excl. impossible | 6,276 | 2.023e9 | 567.8 | 1,149 | 0.1982 | [+0.102, +0.293] | 508.4 |

> ### **LANE B FLOOR: 1,586.2 [1,573.8, 1,597.9] against today's 1,606.89 — 20.7 s recoverable, 2.56 % of lane-B SSE.**

**With and without, as the brief asks:**

| scope | n | today | **floor** | recoverable |
|---|---:|---:|---:|---:|
| lane B, all rows | 8,378 | 1,606.9 | **1,586.2** | 20.7 s |
| excl. the 19 impossible labels | 8,359 | 972.6 | **937.9** | 34.7 s |
| excl. impossible labels **and LIRF** | 7,839 | 573.9 | **508.5** | 65.3 s |

**[X] And the naive version, reported rather than averaged away.** One ρ carried to the whole lane gives
`1,606.9 × √(1−0.270) = 1,372.9` and claims **234 s** recoverable against the block-wise **21 s**. The
difference is entirely the 19 impossible labels being credited with a 27 % recoverable share they cannot
have. **A reader who runs lane A's script unmodified on lane B gets 1,372.9 and a wrong answer.**

### 5.3 Sensitivity — the two blocks where ρ could not be measured

`S5.2a`. ρ = 0 on the impossible labels and at LIRF is an **assumption**, and those two blocks are
**88.1 %** of lane-B SSE. **This is the weakest link in the estimate and it is stated as such.** What
happens if the assumption is wrong in the direction that makes the floor lower:

| ρ(impossible) | ρ(LIRF) | lane-B floor | recoverable | board @0.747 |
|---:|---:|---:|---:|---:|
| **0.00** | **0.00** | **1,586.2** | **20.7** | **278.41** |
| 0.00 | 0.20 | 1,545.6 | 61.3 | 276.26 |
| 0.00 | 0.50 | 1,482.6 | 124.3 | 272.98 |
| 0.10 | 0.27 | 1,476.6 | 130.3 | 272.68 |
| 0.00 | 1.00 | 1,371.2 | 235.7 | 267.40 |
| 1.00 | 1.00 | 491.9 | 1,115.0 | 235.64 |

The last two rows are not estimates; they are the arithmetic limit — LIRF perfectly predicted, then
*both* unmeasurable blocks perfectly predicted, 88 % of lane-B SSE deleted. **[X] Even that lands the
board at 267.4, and it is there so a reader can see how little of the board a total victory on lane B is
worth.**

**The defence of ρ = 0 on the impossible labels** is §6.2: their conditional mean barely moves, so the
conditional-mean predictor — which is what ρ measures the distance to — eats their squared error. **The
defence of ρ = 0 at LIRF is weaker and is really an admission**: ten pairs is not a measurement. §6.1
bounds that block a different way.

### 5.4 Two robustness checks

`S5.3`, **shared operational shock?** ρ_cen by time separation: < 1 day 0.100, 1–7 d 0.154, 7–30 d 0.328,
30–365 d 0.438. **ρ rises with separation** — the opposite of a shared shock, which would decay. It is a
repeatable error the model makes on that kind of flight. (The CIs are wide; the *direction* is the point.)

`S5.4`, **the wide metric** (22 columns instead of 12): at `d ≤ 1.0` there are **50 pairs** and the CI is
[−0.429, +0.464]. **[X] The wide-metric check that bounded lane A's metric bias at half a second cannot be
run on lane B — adding ten columns to the definition of a twin leaves 50 pairs.** The larger bins read
ρ_cen 0.343 and 0.433, but they pool airports and therefore re-admit the between-block first moment that
§5.1 removed, so they are not evidence of a higher ρ. This is a genuine gap: **the metric-sensitivity
question is open on lane B, and it is open in the direction that would make the floor lower.**

---

## 6. Oracle bounds on the named routes

These read the label. They are **not** floors and not achievable; they bound from above what each named
route could ever return.

### 6.1 The LIRF fill gate — `P1-stage2-gate-FAILED.md`'s route

`S6.1`. Production already applies the `ms` reparameterisation at LIRF (`configs/main_gap.yaml`:
`unmatched_sched_reparam: true`, `[LIRF]`), so model B fits `y − ms` there and its prediction is
`ms + ghat`. A **fill** row — BLOCK filled from SCHED, so `y = ms` exactly — is precisely the row where the
right answer is `ghat = 0`. A perfect fill classifier *is* "set `ghat = 0` where the row is a fill".

LIRF OOF rows 537, of which fill (`|bs| ≤ 10 s`) **233 = 43.4 %**.

| | RMSE before | RMSE after |
|---|---:|---:|
| on the 233 LIRF fill rows | 1,869.0 | **3.5** |
| LIRF overall | 4,059.2 | 3,868.0 |
| **LANE B** | **1,606.89** | **1,576.37** (**−30.52 s**) |

**[S] A perfect fill classifier is worth 30.5 s of lane-B RMSE and no more** — 3.76 % of lane-B SSE, and
§7 prices it at **−2.2 s of board** (−1.6 s at 0.747). The same oracle at all ten airports (353 fill rows)
is worth −30.83 s, i.e. LIRF is essentially all of it.

**And the downside.** Predict `ms` on *every* LIRF row and lane B goes to 1,897.77, **+290.88 s**.
**[X] The route is asymmetric — +30 if the gate is perfect, −291 if it fires everywhere — and that
asymmetry is why `P1` set its gate at 0.93. The measured 0.9201 buys some fraction of 30 s, not of 300.**

### 6.2 The nineteen impossible labels

`S6.2`. 19 rows, 63.45 % of lane-B SSE; **2 are visibly ordinary and carry 47.17 %**.

| oracle | lane B |
|---|---:|
| all 19 predicted exactly | 1,606.89 → **971.50** |
| just the 2 invisible LFPG rows | 1,606.89 → **1,168.01** |

**What a real predictor would do.** The conditional mean of `y` given visible `x` on such a row is
`(1−q)·ordinary + q·(ordinary + jump)`. At LFPG, `q = 0.00053` (2 of 3,773 labelled rows) and the median
jump is 69,332 s, so **the optimal shift of every LFPG lane-B prediction is `q·jump` = 36.8 s**, leaving
99.947 % of the jump on the row that actually has it. Applying it measures **1,606.89 → 1,607.45, i.e.
+0.56 s — it makes things worse.**

> **[S] The conditional-mean predictor — the thing the floor is defined as — eats essentially the whole
> squared error on these rows. That IS the floor. And the corruption sits in `BLOCK_TIME_UTC_mvt`, one of
> the two columns blanked on every scored row (`src/prc/schema.py:52`): there is nothing to see it with,
> even in principle.**

Direct check: the best constant shift over all lane-B predictions is −22.8 s and moves 1,606.89 → 1,606.73.

### 6.3 The missing-clock route

`S6.3`. `missing-clock.md` §2.2's **reweighted, non-LIRF** column is the right source and it separates the
two penalties instead of pooling them:

```
A       (sighted, full data)  361.6
B_cf    (blind,   full data)  541.7   -> BLINDNESS  costs 541.7/361.6 = 1.498x
B_small (blind,   19k rows)   591.4   -> STARVATION costs 591.4/541.7 = 1.092x
observed lane, non-LIRF, excl. rollover, reweighted  569.1
```

(The 1.31× in that note's headline is the *matched-row, unreweighted* ratio 292.53/222.76. Both are
carried below because they bracket the answer.)

| subset | today | /1.313 | /1.498 | /1.092 |
|---|---:|---:|---:|---:|
| whole lane B | 1,606.9 | 1,223.8 | 1,072.7 | 1,471.5 |
| excl. 19 impossible labels | 972.6 | 740.7 | 649.3 | 890.7 |
| **excl. impossible and LIRF** | **573.9** | 437.1 | **383.1** | 525.5 |

**[X] The ratio cannot apply to an impossible label.** No amount of NM sight fixes it: the clock that is
missing is *NM's*, and the corruption is in the *airport's* own BLOCK stamp. The honest reading of the
missing-clock route on this axis is the third row — full NM sight at the nine ordinary airports,
573.9 → 383.1.

---

## 7. The board price

`S7`. Pooled OOF reproduces **277.5398**; lane A 213.1266 on 664,050, lane B 1,606.8929 on 8,378. The
additive gap is +1.9836 and **its level is unstable** (`model-a-ceiling.md` §8.1 moves it over 125 s by
re-pooling the folds), so the priced column is the derivative, `dBoard/dOOF = 0.747`, n = 1. Both shown.

| counterfactual on lane B | lane B | pooled OOF | dOOF | board @1.0 | **board @0.747** |
|---|---:|---:|---:|---:|---:|
| today (v14) | 1,606.9 | 277.54 | — | 279.52 | **279.52** |
| **MEASURED FLOOR (§5.2)** | **1,586.2** | 276.05 | −1.49 | 278.04 | **278.41** |
| its 95 % band, low end | 1,573.8 | 275.17 | −2.37 | 277.15 | 277.75 |
| its 95 % band, high end | 1,597.9 | 276.89 | −0.65 | 278.88 | 279.04 |
| floor if ρ(LIRF) = 0.27 like the rest | 1,476.6 | 268.37 | −9.17 | 270.35 | 272.67 |
| *oracle:* perfect LIRF fill gate (§6.1) | 1,576.4 | 275.35 | −2.19 | 277.33 | 277.89 |
| *oracle:* full NM sight, 9 non-LIRF (/1.498) | 1,552.8 | 273.68 | −3.86 | 275.66 | 276.64 |
| *oracle:* full NM sight, 9 non-LIRF (/1.313) | 1,566.1 | 274.62 | −2.92 | 276.60 | 277.34 |
| *oracle:* …and at LIRF too | 1,434.9 | 265.54 | −12.00 | 267.52 | 270.56 |
| *not a route:* the 19 predicted exactly | 971.5 | 237.94 | −39.60 | 239.93 | 249.94 |
| the brief's: lane B RMSE × 0.75 | 1,205.2 | 250.91 | −26.63 | 252.89 | 259.63 |
| the brief's: lane B RMSE × 0.50 | 803.4 | 230.00 | −47.54 | 231.98 | **244.01** |
| the brief's: lane B → 0 | 0.0 | 211.79 | −65.75 | **213.78** | **230.41** |

> ### **The answer to the brief's question 2: lane B's remaining headroom is worth −1.1 s of board (range −0.5 to −1.8 s) at the measured transfer ratio, −1.5 s (−0.6 to −2.4) at ratio 1.000.**

**[X] Note the last two rows against the brief's own figures.** "Lane B taken to zero puts the board at
213.78" and "a 50 % improvement puts it at 231.98" are both computed at a transfer ratio of 1.000. At the
only ratio anyone has measured they are **230.41** and **244.01**. The instability moves lane B's targets
further away too, exactly as it did lane A's.

### 7.1 What fraction of lane B is even in play

`S7.1`:

| block | share of lane-B SSE | share of the **whole** pooled OOF SSE |
|---|---:|---:|
| the 19 impossible labels | 63.45 % | **26.50 %** |
| LIRF, excl. impossible | 24.62 % | 10.28 % |
| the other 9 airports, excl. impossible | 11.93 % | **4.98 %** |

**[X] The block that ten months of feature work could plausibly touch — nine ordinary airports, no LIRF
switch, no broken labels — is 5.0 % of the problem's squared error, and its measured floor is 508.5
against 573.9 today.**

### 7.2 Reconciliation with `missing-clock.md`'s −4.9 to −5.5

`S7.3`. The table above prices on the **OOF's** composition, where lane B is 8,378 / 672,428 = 1.246 % of
rows. `missing-clock.md` priced on the **scored** composition, where the unmatched lane is 5,290 / 344,841
= 1.534 % — a factor 1.231 more of it. Backing lane A's board RMSE out of board 279.5234 gives **197.80**,
which is consistent with the forward fold's 198.53 and is an independent check that the assumption holds:

| counterfactual | lane B | board | dBoard | @0.747 |
|---|---:|---:|---:|---:|
| today | 1,606.9 | 279.52 | — | 279.52 |
| full NM sight, 9 non-LIRF (/1.498) | 1,552.8 | 274.80 | **−4.73** | 275.99 |
| full NM sight, 9 non-LIRF (/1.313) | 1,566.1 | 275.95 | **−3.57** | 276.85 |
| MEASURED FLOOR (§5.2) | 1,586.2 | 277.70 | −1.82 | 278.16 |
| perfect LIRF fill gate (§6.1) | 1,576.4 | 276.84 | −2.68 | 277.52 |

**[S] −3.6 to −4.7 s for the sight route on the scored composition, against `missing-clock.md` §7's
−4.9 to −5.5. Same order, same sign; the OOF table simply has 23 % less lane B in it. Nothing here
overturns that figure.**

### 7.3 The scored lane is a lottery

`S7.2`. The impossible-label rate on the 22,470 labelled unmatched rows is **0.1780 %** (40 rows), so
5,290 scored unmatched rows carry **9.4 expected**. Monte Carlo — draw `k ~ Binomial(5290, rate)`, then `k`
error magnitudes with replacement from the 19 observed ones, and ask what they alone contribute to a
344,841-row board RMSE:

| percentile | 5 % | 25 % | 50 % | 75 % | 95 % |
|---|---:|---:|---:|---:|---:|
| **seconds of board RMSE, in quadrature, from these rows alone** | **45.2** | **79.9** | **127.7** | **167.5** | **223.5** |

Mean over draws 140.4 s; `P(contribution > 100 s) = 0.64`.

> **[X] This is not a modelling uncertainty, it is a sampling one, and it is an order of magnitude larger
> than every improvement this project is chasing. The board's lane-B term is set by how many broken BLOCK
> stamps happened to land in January and July 2026 — a quantity nobody can see, influence or model.**
> **[X] Marked as inference:** this is very likely a large part of why the OOF→board gap's *level* is
> unstable (`model-a-ceiling.md` §8.1, +0.896 vs +1.984, and −43.60 vs +59.80 across fold subsets). It
> predicts that the gap will keep moving by several seconds between submissions for reasons unrelated to
> the model, and it is a second, independent argument for `model-a-ceiling.md` §8.3's recommendation to
> quote deltas rather than levels.

---

## 8. Route 3 — the learning curve, from measurements already in hand

`S8`. Lane A's learning curve needed model fits. **Lane B's is already measured**: `missing-clock.md` §2.2
ran the same blind feature regime at two training sizes on the same rows and folds, for the express
purpose of separating blindness from starvation. Reusing it costs nothing and shares no machinery with §5.

| regime | training rows | reweighted, all ten | reweighted, non-LIRF |
|---|---:|---:|---:|
| B_cf | 247,547 | 554.0 | 541.7 |
| B_small | 19,165 | 616.6 | 591.4 |

B_small was sized to model B's own training set on purpose. Over a **12.9×** range of rows the starvation
penalty is **1.113×** (all ten) / **1.092×** (non-LIRF).

`S8.1`, the two routes on the one block where both can run — the nine non-LIRF airports with the
impossible labels removed, 7,839 rows:

| | RMSE |
|---|---:|
| today | 573.9 |
| **route 3**, learning curve at 12.9× the rows | **515.6 – 525.6** |
| **route 2**, twin-residual covariance floor (§5.2) | **508.5** |
| difference | **+7.1 to +17.1 s (1.4 % – 3.4 %)** |

**[S] They agree to within 1.4–3.4 %, by methods sharing no machinery: one counts how much a twin's
residual predicts this row's, the other reads a pair of models fitted at 19k and 248k rows. The ordering
is also right — a finite-`n` learning-curve point must sit at or above the floor, and it does.**

**[X] And route 3 says something route 2 cannot.** On this block model B is **data-limited, not
information-limited**: 13× the rows is worth 10 % of its RMSE and would land it essentially *on* the
floor. There is no 13×. Lane B's training set is every unmatched departure in 2025 and there are 22,470 of
them. **So most of the 65 s that §5.2 says is recoverable at the nine airports is recoverable by rows that
do not exist, not by a feature.** That is the same conclusion `model-a-ceiling.md` §5.1 reached for lane A,
and it is far sharper here because lane B is 100× smaller.

---

## 9. What this leaves — and the four questions, answered

**1. The floor, split into bias / spread / irreducible, with LIRF separated:**

| | lane-B SSE | note |
|---|---:|---|
| pure **bias** (global mean) | **0.02 %** | §3.1 |
| removable **first moment** beyond that, oracle per-airport | **1.35 %** | §3.1 — and **0 % out of fold** (§3) |
| **recoverable spread**, measured | **2.56 %** | §5.2, the 20.7 s |
| **irreducible** | **≈ 96 %** | of which 63.45 % is 19 impossible labels and 24.62 % is LIRF |

| scope | today | floor | recoverable |
|---|---:|---:|---:|
| lane B | 1,606.9 | **1,586.2** [1,573.8, 1,597.9] | 20.7 s |
| **without the 19 impossible labels** | 972.6 | **937.9** | 34.7 s |
| **without them and without LIRF** | 573.9 | **508.5** | 65.3 s |

**2. Priced through 0.747: −1.1 s of board, range −0.5 to −1.8 s** (−1.5 s, range −0.6 to −2.4, at ratio
1.000). The two unmeasurable blocks could widen that: if LIRF's ρ were as high as the measured blocks'
the floor would be 1,476.6 and the prize −6.9 s at 0.747, and that is the honest upper end of the range
(§5.2a). **Anything beyond it requires predicting labels that nothing visible can see.**

**3. The −4.9 to −5.5 s missing-clock figure is that route's ceiling, not the lane's — and the lane's is
smaller, not larger.** Re-priced on the same composition this note measures the sight route at −3.6 to
−4.7 s (§7.2), which agrees with it. But the lane's total measured headroom at the current feature set is
−1.1 s, **below** the sight route's number, because the sight route is an *oracle over a column that does
not exist* while the floor is measured over the columns that do. **[X] The two numbers answer different
questions and the brief was right that they were being conflated: "restoring the NM flight plan is worth
−4.9 to −5.5" is a statement about a governance decision, and "lane B has −1.1 s left in it" is the
statement about modelling. Neither is the other.**

**4. Where the headroom sits, named.** Not in a feature. In order of size:

* **−2.7 s of board (@1.0; −2.0 @0.747): a perfect LIRF fill classifier** (§6.1, §7.2). Route already
  closed at AUC 0.9201 against a 0.93 gate, and it is asymmetric (+30 s of lane B if perfect, −291 s if it
  over-fires). **This is the largest single identified item and it is already closed.**
* **−1.8 s of board: reaching the measured floor everywhere else** (§7.2). §8 says most of it is rows, not
  features.
* **Nothing from a per-group offset, target encoding or specialist.** §3 closes that family on lane B the
  way `model-a-ceiling.md` §6 closed it on lane A — and closes it harder, because here every honest arm is
  *worse* than doing nothing.
* **Nothing from the 19 impossible labels** (§6.2), which are 63 % of the lane.

> ### **[X] So: yes — lane B is already near its floor, and the question can be closed.** The missing-clock
> figure was not the lane's ceiling, but the lane's ceiling turns out to be *smaller* than it, so no
> decision that was made on the −4.9/−5.5 number needs revisiting. What was genuinely open — "is there
> anything left in lane B" — is now measured at **−1.1 s of board, and at most about −7 s even if the two
> blocks that could not be measured turn out to be as recoverable as the ones that could.** Board 244 via
> a 50 % lane-B improvement, and board 230 via lane B taken to zero, both require predicting broken BLOCK
> stamps and are not reachable.

---

## 10. Caveats on the instrument, recorded rather than buried

* **`n_eff = 7.6` is the caveat that contains the others.** Lane B's bootstrap CI is [1,088, 2,193] on its
  *own RMSE*. Every derived quantity inherits it. Nothing here should be read to better than two
  significant figures, and §5.2's [1,573.8, 1,597.9] is a CI on ρ, **not** on the lane's RMSE — it is
  narrow because the lane's SSE is held fixed while ρ varies, and it would be wrong to read it as the
  total uncertainty.
* **[X] The weakest link, named: ρ = 0 at LIRF.** That block is 24.6 % of lane-B SSE and has **ten**
  usable twin pairs. Nothing in this note measures its recoverable share; §6.1 bounds the one *identified*
  mechanism there (the fill gate, 30.5 s of lane B) and §5.2a shows what a non-zero ρ would do. If LIRF's
  true ρ is 0.5, the lane floor is 1,482.6 and the prize is −6.5 s of board rather than −1.1. **If anyone
  wants one more measurement on lane B, it is this one, and the way to get it is more LIRF unmatched rows,
  which do not exist.**
* **The pooled positive control fails and is reported anyway** (§5.1). ρ_cen = −0.16 for the crippled
  predictor pooled over the lane, against +0.26 and +0.58 on the two blocks where the estimator is
  actually used. One outlier pair decides the pooled number. The block-wise controls pass; the pooled one
  does not; both are printed.
* **The wide-metric check could not be run** (§5.4). On lane A, re-running everything on 21 columns
  bounded the metric bias at about half a second. On lane B the 22-column metric leaves **50 pairs** at
  `d ≤ 1.0`. **The metric-sensitivity question is open, and it is open in the direction that would lower
  the floor** — systematic error lying along the ~76 columns the narrow metric omits would not enter
  `Cov(e_i, e_j)`. §1.5's gain table is the mitigation, not a substitute: the narrow metric holds 87.2 %
  of the gain and everything omitted is at or below the noise controls.
* **Route 1 could not bound the lane** (§4) and route 4, the matched-twin ladder, could not be built at
  all (§1). Two of lane A's four routes do not transfer. The two that do — the twin covariance and the
  learning curve — agree to 1.4–3.4 % on the one block where both run, and that agreement is the whole
  evidential basis for the headline. **It rests on 1,233 pairs and one pair of previously-fitted models.**
* **`ρ_raw` vs `ρ_cen` is a real fork and the choice is argued, not assumed** (§5.1). Using ρ_raw would
  give a lower floor and a larger prize. It is rejected because the permuted-partner control excludes
  zero under ρ_raw and contains it under ρ_cen, and because §3 independently shows the first moment ρ_raw
  is counting does not survive an out-of-fold split. **A reader who disagrees should note that ρ_raw's
  extra share is precisely the thing `P1B` was rejected for.**
* **The impossible-label class is defined on the label side.** `y > 40,000` is a description of rows
  already identified by their squared error, not a model input, and §2.1 reaches the same rows with no
  threshold. The alternative criterion `bs < −10,000` catches a different 17 rows and the same 63.6 %. But
  **it remains true that I chose where to cut, and §5.2's block assembly depends on that cut.** The
  sweep in §2.4 is the defence; the reader should check it.
* **Nothing here re-measures the fill classifier, the LIRF seasonal bias, or any per-group specialist.**
  Those are `P1-stage2-gate-FAILED.md`, `P1B-lirf-seasonal-bias-REJECT.md` and `model-a-ceiling.md` §6,
  and they are used as inputs, not repeated.
* **The transfer ratio is still n = 1.** 0.747 has no error bar. It is used because 1.000 has never been
  measured at all, and both columns are printed everywhere.
* **§8's learning curve is two points from another note's experiment**, run at 400 fixed rounds on a 13 %
  training subsample. Its *level* is about 3 % worse than production by that note's own statement; only
  the **ratio** 591.4 / 541.7 is read here, and a ratio of two arms sharing every setting is the one thing
  that design supports.
* **No control was added after the fact.** §3's shuffled-cell arms (in-sample *and* out-of-fold), §4's
  permuted-`bs` arm, and §5's permuted-partner and crippled-predictor arms were in the design before any
  headline was read, on the instruction of `missing-clock.md` §7 and `P1-stage2-gate-FAILED.md`. **One of
  them fired** — the raw negative control — and changed the estimator (§5.1). The failure signatures those
  documents name did not occur: the routes return 1,707 (cannot bound), 1,586, and 515.6–525.6 on its own
  block, which is agreement without suspicious identity; no floor is near zero; and the permutation
  controls return 4,849 against a within-airport sd of 4,774, and a ρ CI containing zero.
* **The box.** The full file is 18 s and 1.7 GB peak on a 4-CPU box. `S1`'s importance fit is the only
  model fitted anywhere in this note.
