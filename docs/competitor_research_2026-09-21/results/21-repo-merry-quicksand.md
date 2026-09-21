# Slot 21 — `saurabhdxt-labs/prc-challenge` (team **merry-quicksand**)

Investigated 2026-09-21. Clone: `git clone https://github.com/saurabhdxt-labs/prc-challenge` →
`/tmp/prc-merry-quicksand`. All line references below are to that clone at commit **`fdeb704`**
(HEAD = `origin/main`, committed **2026-09-12 16:34:39 -0400**). 349 files, 44 commits,
~21,800 lines of Python, 110 report/plan artefacts. Licence GPL-3.0.

Marking: **[S]** = stated/measured in their repo or in our own repo, with a citation.
**[X]** = my inference or arithmetic.

---

## 0. Status, and one discrepancy to record first

**[S]** The last board result recorded anywhere in this repo is
**v12 = 266.3738, rank 5 of 120, 2026-09-11 18:27 EDT** (`reports/MSE_LEDGER.md`, section
"MEASURED 2026-09-11 18:27 EDT"). The repo's final commit is 2026-09-12.

**[S]** The prior investigator reports merry-quicksand at **257.5532, rank 11 of 168** today.

**[X]** Therefore: everything in this repo is their *266.37* pipeline. The ~8.8 s between 266.37
and 257.55 was earned after 2026-09-12 and is **not in the public snapshot**. Their `reports/`
end mid-flight with three live threads that plausibly carry it (ADS-B code-2 rows, the
CAP+CatBoost bundle, the unmatched body) — see §3.1 and §3.9. Do not read this repo as their current system.

**[S]** Their own ranking history, from `reports/MSE_LEDGER.md`:
v1 689.69 (constant probe) → v2 301.7019 (rank 26/80) → v3 291.6317 → v4 289.6732 → v5 288.1406 →
v6 286.9017 → v7 285.8013 → v9 282.6790 → v10 281.5182 (rank 15/106) → v11 267.4518 (rank 7/119) →
v12 266.3738 (rank 5/120). (v8 was built and never uploaded.)

---

## 1. The reparameterisation, to reimplementation grade

### 1.1 The algebra, and the identity it rests on

**[S]** Column semantics (`prc/labels.py:30-33`):

```
LABEL     = TAXITIME_SEC_mvt      (target; 100% null on scored rows)
OFF_BLOCK = BLOCK_TIME_UTC_mvt    (100% null on scored rows)
TAKE_OFF  = MVT_TIME_UTC_mvt      (0% null on scored rows)
```

**[S]** The label identity is *enforced*, not assumed — `prc/labels.py:44-62`,
`check_label_identity()` raises unless
`(MVT_TIME_UTC_mvt − BLOCK_TIME_UTC_mvt).total_seconds() == TAXITIME_SEC_mvt` with
`tolerance_sec = 0.0`, and it is called inside `clean()` (`prc/labels.py:89`) before any
value-based filtering. Their measurement: the identity held **exactly on 100.0000 % of the
153,706 departures of 2025-01** (`prc/labels.py:18-20`).

**[S]** The two derived quantities (`scripts/build_submission.py:101`, `:494`;
`scripts/stand_ab.py:182-184`):

```
proxy = MVT_TIME_UTC_mvt − AOBT_3_flt          # seconds, known exactly at serve time
delta = BLOCK_TIME_UTC_mvt − AOBT_3_flt        # seconds, unknown at serve time (BLOCK is hidden)
```

**[X]** The algebra is exact and needs no approximation:

```
proxy − delta = (MVT − AOBT_3) − (BLOCK − AOBT_3) = MVT − BLOCK = TAXITIME_SEC_mvt = y
```

`AOBT_3` cancels identically. **Any** serve-time clock `C` present on the row works the same way;
`AOBT_3_flt` is chosen because it is the tightest: **[S]** `BLOCK − AOBT_3` has median +51 s,
p10 −310 s, p90 +363 s, p1 −1,194 s, p99 +799 s on training rows
(`reports/DATA_DICTIONARY.md:1684`), i.e. `sd(delta) ≈ 364–427 s` against `sd(y) ≈ 686 s`
(`reports/HARNESS_CALIBRATION.md` §1: std(y) = 686.07). So `delta` is the *small* part of `y`
and `proxy` carries the rest exactly.

**[S]** Their stated reason for doing it at all (`scripts/build_submission.py:156-161`, README:22-26):

> "A tree cannot represent `y = proxy − f(x)` — it must bin the identity in a wide-range variable —
> but it can represent `f(x)`. Measured on an identical fold, features and seed:
> **285.24 (predicting y) → 252.66 (predicting delta)**."

**[S]** Provenance of that number: it was measured on the 26-feature sklearn HGB, early, and was
never re-run on that learner/feature set — stated in their own prereg
(`plans/PREREG_taxiout_2026_09_08.md:1634`, `:1670`). It has been re-confirmed twice since, on
different harnesses: `reports/RED_TEAM.md` §4.8 (12 %-subsample HGB: y-target 290.9 vs
delta 252.7) and RESULT 19 / RESULT 15 at full LightGBM capacity (§3.4 below).

### 1.2 Which rows it applies to

**[S]** Exactly the rows with `AOBT_3_flt` non-null — the "matched" lane.
`scripts/build_submission.py:116` defines `unmatched = df.AOBT_3_flt.isna()`, and
`predict()` (`:406-414`) routes:

```python
matched = ~test.unmatched.to_numpy()
out[matched]  = fit_matched(train[~train.unmatched], test[matched])
out[~matched] = fit_unmatched(train[train.unmatched], test[~matched], ...)
```

**[S]** Scale: 339,551 matched / 5,290 unmatched of 344,841 scored rows =
**98.4660 % / 1.5340 %** (`scripts/lgbm_submit.py:147`; `reports/MSE_LEDGER.md` header states the
exact decomposition `RMSE² = 0.9846596·matched² + 0.0153404·unmatched²`).

### 1.3 What happens when a component is null

**[S]** Three distinct cases, all handled by arithmetic rather than by special-casing:

1. **`AOBT_3_flt` null** → `proxy` and `delta` are both NaN by subtraction
   (`scripts/stand_ab.py:170-172` says this explicitly: "on an unmatched row AOBT_3 is NaT and
   every AOBT_3-derived value here is NaN by arithmetic, not by a special case"). The row is routed
   to the *unmatched* lane and never touches the reparameterisation.
   **[S]** `AOBT_3_flt` null is identical to "all 16 `_flt` columns null"
   (`reports/DATA_DICTIONARY.md:18` — "identical to 'all 16 _flt columns null' (True)").
2. **`BLOCK_TIME_UTC_mvt` / `TAXITIME_SEC_mvt` null** → training row only; dropped by the
   admissibility filter (`scripts/build_submission.py:84-85`; `prc/labels.py:83-85`, rule C2).
   On scored rows both are always null, which is why `delta` is the thing being predicted.
3. **`MVT_TIME_UTC_mvt` or `SCHED_TIME_UTC_mvt` null** → in serve mode the row is not built at all:
   `scripts/stand_ab.py:232-233` filters
   `(PHASE_mvt == "DEP") & AOBT_3_flt.notna() & MVT_TIME_UTC_mvt.notna() & SCHED_TIME_UTC_mvt.notna()`.
   **[S]** Both are 0.000 % null on scored rows (`reports/DATA_DICTIONARY.md` §1, rows 8 and 10),
   so this filter never fires in practice.

### 1.4 What the model's fitting target literally is

**[S]** `scripts/build_submission.py:493-494` (the reference implementation):

```python
frame["y"]     = frame.TAXITIME_SEC_mvt.astype("float64")
frame["delta"] = _seconds(frame.BLOCK_TIME_UTC_mvt, frame.AOBT_3_flt)   # (a−b).dt.total_seconds()
```

and `fit_matched` (`:155-188`) fits on `train.delta.to_numpy()` — raw float64 seconds,
**not** winsorised, **not** clipped, **not** log-transformed, **not** de-meaned, every labelled
matched row kept.

**[S]** The shipped LightGBM path is the same (`scripts/lgbm_submit.py:415-423`,
`regression_label()`): under `target="delta"` the label handed to LightGBM is `delta` verbatim
("Float64, the values untouched"). Objective `regression` (L2), metric `rmse`
(`scripts/lgbm_submit.py:174-177`).

**[S]** Censoring the target was tested and is *badly* wrong: winsorising training `delta` at
[p0.5, p99.5] = [−1,558, +960] cost **−30.31 s [−33.26, −27.35]**
(`reports/MATCHED_MODEL.md` §4). Their reading: "the tails of `delta` are predictable signal, not
noise… Any robust-loss or outlier-trimming idea for this target is dead on this evidence."
Absolute-error loss: −14.9 s (README:37-40).

### 1.5 How the prediction is reassembled, and every guard on it

**[S]** Reference path (`scripts/build_submission.py:186-188`):

```python
delta_hat = (1 - BLEND) * pooled + BLEND * per_airport      # BLEND = 0.5
return np.maximum(proxy - delta_hat, 1.0)
```

then at write time `np.rint(pred).astype("int32")` (`:531`, `:546`).

**[S]** Shipped path (`scripts/lgbm_submit.py:394-435`), three functions, tested to be identical:

```python
def raw_taxi_time(pred, proxy, target="delta"):   # :426
    return np.asarray(proxy, "float64") - np.asarray(pred, "float64")

def finalise_taxi_time(raw):                      # :394
    assert np.isfinite(raw).all()
    bound = float((raw < 1.0).mean())
    assert bound < 0.005, "positivity floor binds on %.2f%% of rows"
    return np.rint(np.maximum(raw, 1.0)).astype("int32")

def recover_taxi_time(proxy, delta_hat):          # :409
    return finalise_taxi_time(proxy - delta_hat)
```

So the complete reassembly is:

```
y_hat = int32( rint( max( proxy − delta_hat , 1 ) ) )
```

**[S]** The only clipping is the **1-second positivity floor**, and it carries a *tripwire*: if the
floor binds on more than **0.5 %** of rows the run aborts — "if it binds on more than 0.5 % of rows
the model is broken, not merely imprecise" (`scripts/lgbm_submit.py:394-406`). There is **no upper
clip**, no shrinkage toward a mean, and no per-airport post-hoc offset. Nothing else touches the
value. Submission contract checks (no nulls, finite, all > 0, id set equality) are in
`scripts/build_submission.py:455-470`.

**[S]** Seed averaging happens on the **delta scale before** subtraction
(`scripts/lgbm_submit.py:1026-1037`, `mean_delta` — arithmetic mean of per-seed `delta` predictions,
summed left-to-right in float64), not on the taxi-time scale.

**[S]** They treat "which scale is a stored prediction on" as a load-bearing hazard: `prc/reference.py:24-27`
records `scale ∈ {"taxi_time","delta"}` in every saved prediction's metadata because
"reading one as the other produced a −109 s result"; `reports/bug_classes.md` BC-2 is the bug class
("two prediction columns compared under DIFFERENT recovery conventions"), surfaced twice in one night.

### 1.6 Minimal reimplementation (self-contained)

```python
# ---- build, per row, from the raw movement frame -------------------------------
proxy = (MVT_TIME_UTC_mvt - AOBT_3_flt).total_seconds()      # serve-time known
delta = (BLOCK_TIME_UTC_mvt - AOBT_3_flt).total_seconds()    # training only
matched = AOBT_3_flt is not NULL

# ---- train (matched rows only, labelled, y>0, dedup on MVT_ID) -----------------
model.fit(X[matched & labelled], delta[matched & labelled])   # L2, raw seconds, no clipping

# ---- predict (matched scored rows) --------------------------------------------
delta_hat = mean_over_seeds(model.predict(X_scored))          # average on the DELTA scale
y_hat     = np.rint(np.maximum(proxy_scored - delta_hat, 1.0)).astype("int32")
assert (proxy_scored - delta_hat < 1.0).mean() < 0.005

# ---- unmatched scored rows: a different estimator entirely (§4.5) --------------
```

### 1.7 What cannot be determined from the repo

* **The 285.24 → 252.66 run itself is not reproducible from the repo.** It predates the scripts
  that are committed (it was made on "the 26-feature HGB months ago",
  `plans/PREREG_taxiout_2026_09_08.md:1670`); no script, log or record file in the clone regenerates
  the pair. What *is* reproducible is `scripts/build_submission.py --validate` and the later
  y-vs-delta comparisons of §3.4, which point the same way with different magnitudes.
* **Data is git-ignored**, so no number here can be re-measured without re-fetching the challenge
  data (`scripts/fetch_data.py`).
* Everything else in §1.1–§1.6 is in the committed code, verbatim.

---

## 2. Is every input to `proxy` available on a 2026 scored row?

**Yes for 98.466 % of rows, and no for the other 1.534 %, all-or-nothing.** [S] throughout:

| input | source table | null % on 2026 scored rows | citation |
|---|---|---|---|
| `MVT_TIME_UTC_mvt` (take-off) | movement `_mvt` | **0.000** | `reports/DATA_DICTIONARY.md` §1 row 8 |
| `SCHED_TIME_UTC_mvt` | movement `_mvt` | **0.000** | §1 row 10 |
| **`AOBT_3_flt`** | **NM flight table `_flt`** | **1.534** | §1 row 29, §2.29 |
| `EOBT_1_flt`, `LOBT_flt`, `IOBT_flt`, `ARVT_1/3_flt`, operator, segment, wake, type | NM `_flt` | 1.534 each | §1 rows 15–30 |
| `BLOCK_TIME_UTC_mvt` | movement | **100.000** | §1 row 9 |
| `TAXITIME_SEC_mvt` (label) | movement | **100.000** | §1 row 14 |

**[S]** `AOBT_3_flt` is unambiguously the NM flight-table column: "NM field; name suggests 'actual
off-block time, message 3'" (`reports/DATA_DICTIONARY.md:1504`), and its null set is *exactly* the
"all 16 `_flt` columns null" set (`:18`). **[S]** Their prereg Amendment 3 records the same
all-or-nothing structure — "for the 1.534 % of scored rows with no Network Manager record, every
`*_flt` column is null together… Any model for that stratum is restricted to the `_mvt` columns plus
the schedule, and the operator survives only through the callsign prefix of `FLIGHT_mvt`"
(`plans/PREREG_taxiout_2026_09_08.md:433-441`).

**[X]** This matches our own data exactly (our note: `_flt` present on 98.5 % of rows,
all-or-nothing; MVT + schedule 100 %; BLOCK and label 0 %). The 1.534 % is precisely why they run a
second estimator: it is the set where the second clock does not exist.

**[S]** `AOBT_3_flt` values on scored rows are 100 % whole-second, 97.43 % on the 60-s grid
(`reports/DATA_DICTIONARY.md` §2.29) — it is a *reported* clock, not a second-resolution
measurement. `MVT_TIME` and `BLOCK_TIME` are second-resolution (5.1 % / 6.4 % whole-minute).

**[S] A governance risk they raised and never got resolved.** On 2026-09-11 the owner emailed
`enrico.spinielli@eurocontrol.int` and `challenge@opensky-network.org` asking, verbatim, "whether
`AOBT_3_flt` on ranking rows may be a model input, e.g. MVT − AOBT_3"
(`reports/MSE_LEDGER.md`, entry 2026-09-11 14:02 EDT). The reply (14:25 EDT) was **no ruling** —
a redirect to the OSN Discord. They noted "A 'no' on (1) affects every submission since v2."
**[X]** The same risk applies to us identically; nothing in the repo suggests the question was ever
answered.

---

## 3. The experiment ledger, in order, with scores — including everything that failed

Source: `reports/MSE_LEDGER.md` (754 lines), `plans/PREREG_taxiout_2026_09_08.md` (2,920 lines,
27 dated amendments + numbered RESULTs), plus the per-arm preregs. All [S].

### 3.1 Board submissions (their only real instrument)

| version | board RMSE | board MSE | Δ MSE | what changed | fold projection | transfer |
|---|---|---|---|---|---|---|
| v1 | 689.6901 | — | — | constant probe (harness calibration) | — | — |
| v2 | 301.7019 (26/80) | 91,024 | — | shipped HGB, 26 features + consolidated stratum | fold 330.81 | — |
| v3 | 291.6317 (20/87) | 85,049 | −5,975 | LightGBM on the 68-feature cache, **matched rows only** | −5,123 | **1.17×** |
| v4 | 289.6732 (19/92) | 83,911 | −1,138 | stratum hybrid (Amdt 18), unmatched rows only | — | — |
| v5 | 288.1406 (19/94) | 83,025 | −886 | 3 seeds + queue block, matched | −910 | **0.97×** |
| v6 | 286.9017 | 82,313 | −712 | arm F (record-ordering block), matched | −745 | **0.96×** |
| v7 | 285.8013 | 81,682 | −630 | Rome R2 (stake-weighted fill classifier + 24 h date-slip), **382 LIRF unmatched rows** | ≈ −4,150 | **≈ 0.15×** |
| v8 | built, **never uploaded** | | | LIRF schedule floor | | |
| v9 | 282.6790 (15/106) | 79,907 | −1,775 | E3C congestion-aware unmatched body, 4,828 non-LIRF unmatched rows | +3,960 registered | 0.45× |
| v10 | 281.5182 (15/106) | 79,252 | −655 | E1 LIRF schedule floor, **13 rows** | −5,627 / −680 / +7,012 | PARTIAL |
| v11 | **267.4518 (7/119)** | 71,530 | **−7,722** | ADS-B stack stage on 161,036 matched rows at 9 airports | −5,680 | **1.36×** |
| v12 | **266.3738 (5/120)** | 70,955 | −575 | LIRF local-day rule (3 rows) + LIRF into the ADS-B gate | −664 | 0.87× |

**[S] The single most transferable rule in the repo:** a *matched-lane* paired fold gain, measured in
MSE, lands on the board at **0.96×–1.36×** — confirmed five times (v3, v5, v6, v11 and the
correction note of 2026-09-11 ~17:00). **[S]** An *unmatched/extreme-row* fold projection does **not**:
0.15× (v7) and 0.45× (v9). Their rule: "an unmatched/extreme-row fold projection is not a board
forecast — price such lanes at a steep discount until a board reading calibrates them."

### 3.2 What worked (their own summary, `reports/WHERE_WE_STAND_2026_09_10.md` §4a)

| lever | value |
|---|---|
| **the `delta` reparameterisation** | matched **285.24 → 252.66**, "the single largest modelling decision in the repo" |
| LightGBM at capacity instead of sklearn HGB | +4.0 % matched, +4,251 fold MSE; board −5,975 |
| E1 feature block (pairwise clock differences + arrival/stand) | +8.17 s [+6.48, +10.17] |
| per-airport models + 0.5 blend (HGB era) | 235.95 → 230.41, +5.53 [+4.91, +6.14] |
| stratum hybrid (fitted body, cell tail) | board −1,138 |
| queue/surface block | real, below its own bar; shipped by owner decision |
| seed averaging (3 seeds) | ESTABLISHED, ~215 MSE |
| ADS-B stack stage (v11) | **−7,722 board MSE**, the largest single step |

### 3.3 What failed — matched lane (every one with a number)

| lever | measured | why |
|---|---|---|
| Native categoricals (LightGBM) instead of target encoding | +0.8 % vs +3.4 % | "STAND_mvt's 1,884 levels were never the problem" |
| CatBoost native `C_delta` alone | −1,422 MSE (225.78 vs F 222.56), worse at 9/10 airports | |
| Per-airport tree counts / per-airport blend on LightGBM | NOT WORKING (RESULT 7) | (it *did* work on HGB) |
| Stand-occupancy / witness block | +350 MSE marginal | 69 % redundancy with incumbent `prev_arr_*` |
| Weather, pooled | +0.49 s | wrong granularity |
| Out-of-fold target encoding | **−0.96 s** (hurts) | |
| Winsorised training targets | **−30.3 s** | tails are signal |
| Absolute-error loss | −14.9 s | metric is RMSE |
| Flight-number target encodings (K=50) | −9.1 s [−10.5, −7.6]; with STAND −100 s | self-inclusion at ~10 rows/key |
| Single undivided model (one model, all rows) | **−144 s** | |
| **Arm U** — one y-target LightGBM over all rows, at capacity, stopping rule corrected in its favour | **NOT WORKING twice** (RESULT 12, 15). matched 224.26 → 263.50; LIRF matched 384 → 623 | "the standard approach loses to the two-part pipeline" |
| **Arm Y** — y-target on matched rows only, 0.5 blend | NOT WORKING pooled: **−9.083 s [−27.996, +1.094]**; Y alone 260.651 vs delta 224.258 | see §3.4 |
| Gated blend on `|delta_hat| > 600` (Amdt 25) | NOT WORKING, −9.844 s | LIRF −72.75 sinks a rule that helps at nine airports |
| Gated blend ex-LIRF (Amdt 26) | INCONCLUSIVE, +0.454 s ≈ 200 MSE | |
| `AOBT_3` derivation fingerprints (Amdt 21) | AUC 0.48–0.51 | no signature of how NM produced the clock |
| Record-ordering as an identifier | AUC 0.53 / 0.59 | (survives as a *feature* block = arm F) |
| `FLIGHT_ID`/`MVT_ID`/row order leaking BLOCK | NOT WORKING, closed cleanly | FLIGHT_ID is **LOBT-ordered**, 1.000 concordant with LOBT, 0.50 with BLOCK |
| Airport-clock turnaround anchor | sd(turn) 13,503 s vs sd(delta) 427 s | anchor 30× noisier than the target |
| Regime-gated experts (REG) | +632.8 [+251, +1,006], below the 1,000 bar; **oracle gate +22,959** | the gate's *inputs* are the limit |
| Learned model selector over 12 stored experts (SEL) | best +253 [−16, +529]; LightGBM stacker **−24,475** | experts too alike (residual corr 0.98 with F) |
| Stacking existing arms on inner OOF | −1,093 to −1,781 split-half | same |
| Capacity sweep `127,20,0.6` (CAP) | ESTABLISHED +253 weighted fold MSE — **below the +500 ship bar**, not shipped | |
| NM-parameter diagnostic (NMD) | FALSE, no room; cross-fitted corrections −586 board MSE | |

### 3.4 The y-vs-delta head-to-head at capacity — important, and *not* one-sided

**[S]** RESULT 19 (`plans/PREREG_taxiout_2026_09_08.md:2499-2536`), matched rows, fold A, identical
design/seeds: delta-target baseline **224.258**, y-target alone **260.651**, 0.5 blend 233.341.
Pooled verdict NOT WORKING. But the band decomposition inverts:

| `|delta|` band | share of matched SSE | 0.5-blend gain vs delta-target | y alone |
|---|---|---|---|
| < 2 min | 13.7 % | −0.011 [−0.193, +0.161] | −1.010* |
| 2–10 min | 37.8 % | **−21.083** [−58.0, +0.14] | −73.096* |
| 10–20 min | 17.4 % | **+1.420 [+0.807, +2.009]*** | +0.058 |
| **> 20 min** | 31.1 % | **+9.992 [+5.628, +14.348]*** | +8.585* |
| over-10 combined | 48.5 % | **+4.459 [+2.835, +6.058]*** | +3.212* |

**[S]** Their reading: "the `proxy` anchor is exactly what fails on a held row", so a model that does
not use it is better precisely where the clocks disagree — and a blend applied *everywhere* pays for
that gain out of the body. **[S]** Every attempt to gate the blend on serve-time quantities failed
(Amdt 25/26/27), mostly because LIRF reverses the pooled sign.

**[X]** This is the single most actionable nuance in the whole repo: the reparameterisation is
right, and its residual error is *concentrated in the rows where the anchor is wrong*, which is a
different problem from "improve the model".

### 3.5 What failed — unmatched lane

| lever | measured |
|---|---|
| Fill classification with an oracle flag + constant non-fill | 1,868 → **3,081 (worse)** — the mixture's hedging does real work |
| Fine `sp` buckets | rejected under held-out LOMO |
| Mangled-designator separator at LIRF | +305 MSE, CI [−12.87, +26.58] — rests on 12 rows in 2025, ~3 expected in 2026 |
| Arrival-side tell for LIRF fills (Amdt 11) | NOT WORKING; permuted control kept 71 % of the "gain" |
| Designator-format features at LIRF | AUC 0.9061 → 0.9003, assembled −62 |
| Shrinkage / temperature recalibration of the LIRF `p` | −251 / +92 / −122 (2026-priced) |
| Turnaround linkage, runway-configuration regimes, within-hour spread | each NOT WORKING at the 1,000-MSE bar |
| E3R (offset-to-witness body) | NOT WORKING — interval spans zero, 6/12 months |
| E5 (physics + weather body) | INCONCLUSIVE, event-excluded 2026 price +208 |
| Local-day `dayoff` at Rome (LD) | NOT WORKING — Rome LOMO 3,916.6 → 4,005.5 |

### 3.6 `GAP_LOCATED_2026_09_10.md` — where their gap to the pack was

**[S]** Measured against a public GPLv3 system (`elegant-alligator`, board 274.34 → 271.04) that also
held July 2025 out. On July 2025, unmatched-lane SSE gap to their shipped S1 = **1,145 M = 6,006 MSE
in July's frame — the entire July all-rows gap — and it is entirely Rome**: LIRF alone is 1,210 M,
the other nine airports are together 65 M *better* than the competitor's.

| July 2025 unmatched | their S1 RMSE | competitor |
|---|---|---|
| **LIRF (337 rows)** | **3,925.6** | 3,437.9 |
| LTFM (425) | 704.4 | 690.0 |
| EHAM (515) | 245.2 | 329.6 |
| EDDF (435) | 387.8 | 401.2 |

**[S]** What the competitor did that they did not (read from the competitor's `missing_v2.json`):
(1) a separate normal expert trained on **non-identity rows only** (`target < 6000 and nonidentity`),
(2) a Rome-only fill classifier **weighted by the MSE stake**:
`sample_weight × clip(1 + (max(sp,0)/3600)², 1, 25)`, target `|y − sp| ≤ 6 s`,
(3) mixture `p·sp + (1−p)·normal` with `p = 0` when `sp ≤ 0`, Rome alpha 0.42.
They reimplemented (2) as arm R (`scripts/rome_fill.py:44-46`: `W_UNMATCHED=8.0`, `W_CAP=25.0`,
`stake_weight = clip(1+(max(sp,0)/3600)², 1, 25) × 8 if unmatched`) and it shipped as v7.

**[S]** Also closed in that report: "BLOCK_TIME is an exact copy of another serve-time timestamp" —
NOT WORKING, ≤ 0.7 % exact equality on any candidate.

### 3.7 `UNMATCHED_DIAGNOSTIC_2026_09_11.md` (UMD) — the bound that matters

**[S]** Verdict: **AMBIGUOUS by a defect in its own decision rule → no arm registered.** The facts:

| class (2025 label) | 2025 rows | share of 2025 SSE | top-10 share | Kish ESS | 2026 expected board MSE |
|---|---|---|---|---|---|
| 24 h date-slip | 14 | **48.0 %** | 99.8 % | 3.4 | 12,971 |
| ordinary | 20,801 | 26.9 % | 16.4 % | 217 | 8,578 |
| fill | 1,396 | 14.8 % | 42.1 % | 33 | 7,825 |
| Family B | 8 | 10.3 % | 100 % | 2.1 | 2,665 |
| **learnable** (ordinary + fills with sp < 10,800) | 22,033 | 33.6 % | 13.1 % | — | **10,809** (LIRF 6,731) |

Calibration: the 2026 expectation totals **32,038** board MSE against ≈ 30,478 inferred from the
board (+5 %). By airport: LIRF 20,724 (65 %), LSZH 4,482, LFPG 3,753, LTFM 1,431, rest ≤ 577.

**[S] The bound:** "a perfect model of every learnable row (10,809) plus the matched ship reaches
≈ **260.6**, not 245"; recovering 10/20/30 % of the pool gives ≈ 278.2 / 276.3 / 274.3.
**245 is not reachable through learnable unmatched rows.**

### 3.8 `RED_TEAM.md` — the adversarial pass

**[S]** One established positive: **the ambient proxy for the unmatched stratum** — for a row with no
`AOBT_3`, the median `MVT − AOBT_3` of *matched* departures taking off around it (backward windows
only, same airport / same runway) is serve-time observable and predicts the non-fill taxi.
As an HGB residual correction on the inner-LOMO out-of-fold residual `y − nf`, winsorised at
±3,000 s **for fitting only**: stratum **1574.5 → 1527.2 (−47.3 [−62.8, −36.8])**, non-monster
963 → 882 (−81 [−89, −73]), better in **12/12 months**, seed-stable. Spearman of runway-ambient proxy
with the non-fill target **+0.41**. This is the ancestor of the shipped E3C body.

**[S]** Negatives, each with numbers: no id/row-order leak of BLOCK (§4.1); the ARVT "shortcut" is
worth 0.4 s (§4.2); the matched tail is not the stand chain (§4.3) — **83 % of the |residual| > 1,000 s
rows have `BLOCK − AOBT_3 < −600`**, i.e. the airport stamps off-block 20–35 min *before* NM's, median
NM delay 34–53 min, July carries 73 % of tail SSE; replacing cell means by a plain HGB on the stratum
is **+408 s worse**; flight-number encodings hurt (§4.7); direct-y loses 290.9 vs 252.7 (§4.8).

**[S]** §4.3's mechanism statement, which is the most useful sentence in the file:
"BLOCK looks like an airline OUT-type event (brakes released, waiting for TSAT/CTOT) on slot-held
flights, while `AOBT_3` is the movement. If any column identified slot-held flights it would be worth
up to ~25 s on matched RMSE (197.3 without the tail rows) — none does."

### 3.9 `PIPELINE_TO_245_2026_09_10.md` — the path to 245 they describe

**[S] The answer is: there isn't one, and they say so.** From 281.5182 (79,252 MSE) to 245.0207
(60,035) is 19,217 MSE. Their staged plan and prices (event-excluded, on the 2026 composition):

| stage | expected board MSE | P(transfers) | heavy hours |
|---|---|---|---|
| 1 — E3R (unmatched body learns the offset to the congestion witness) | +0.8–1.5k | 0.6 | 0.3 |
| 2 — ADS-B measurement of the EHAM 2–9 Jan storm days | +1.0–2.5k | 0.5 | 3 (+1 day) |
| 3 — matched representation program (BC-1 repair, capacity confirmation, CatBoost, inner-OOF blend) | +0.5–2.5k | 0.5–1.0 | 8 |
| 5 — small lanes | +0.3–0.6k | 0.9 | 4.5 |
| **credible sum** | **≈ +4k (range 1.5–8k)** | | ≈ 16 |

**[S] Probabilities they attached:** top-10 ≈ 55 %, 274 ≈ 50 %, top-5 ≈ 12 %, 2nd ≈ 5 %,
**245 ≈ 2 %**. "245 needs ≈ 15k that has no located mechanism."

**[S]** Their error map for v10 (79,252 board MSE): matched ≈ 48.8–51.3k, unmatched ≈ 28–30.5k; of
the unmatched, only ≈ 14k is *mechanism* and ≈ 14–16k is rare-class rows (24 h date-slips,
Family-B monsters) that **every serve-time test they built fails to identify**.

**[S]** Matched-lane error by class (arm F, weighted to the 2026 proportion):
schedule-fill rows (`|y − sp| ≤ 60`) 30,167 rows / 7,148 wMSE (15 %); tail (non-fill, `|delta| > 600`)
22,906 rows / **17,208 wMSE (35 %)**; ordinary 285,942 rows / 24,418 (50 %).
By airport: LIRF 10,724, LTFM 7,814, LFPG 6,728, EGLL 6,054, LEBL 3,966, EDDF 3,278, EHAM 3,106,
LEMD 3,077, EDDM 2,109, LSZH 1,919.

**[S]** Eleven hypotheses closed in that one pass (H1–H11), all with evidence — including
H1 "an E3C-shaped congestion analogue exists for the matched lane": **NOT WORKING** (arm F's mean
residual is −1.6…−18.5 s in every witness bin), and H11 "245 is reachable on current evidence":
**NOT WORKING, P ≈ 2 %**.

**[X]** In hindsight they were wrong about their own ceiling: they wrote P(top-5) ≈ 12 % on 09-10 and
reached rank 5 on 09-11 — via ADS-B (their stage 2, generalised to nine airports), which delivered
−7,722 board MSE against a ≈ +1–2.5k estimate.

---

## 4. Features, model, folds, encodings — extraction

### 4.1 Feature list (83 columns in the shipped arm F design)

**[S]** Assembled in `scripts/stand_ab.py:64-79` and `:810`, `:967`:
`BASELINE_FEATS = BASE(26) + SURF(14) + STANDHIST(4) + ENC(24)` = 68, then arm F adds
`QUEUE_FEATS(12)` and `ORDER_FEATS(3)` → 83. Definitions (`scripts/stand_ab.py:166-215`):

**BASE (own-clock, 26)** — `proxy = MVT−AOBT_3`, `sp = MVT−SCHED`, `eobt_p = MVT−EOBT_1`,
`lobt_p = MVT−LOBT`, `iobt_p = MVT−IOBT`, `aobt_sched = AOBT_3−SCHED`, `eobt_sched`, `aobt_eobt =
AOBT_3−EOBT_1`, `lobt_sched`, `eobt_lobt`, `iobt_lobt`, `airborne3 = ARVT_3−MVT`, `airborne1`,
`arvt_diff = ARVT_3−ARVT_1`, `aobt_sec`/`mvt_sec`/`sched_sec` (second-within-minute of each clock),
`hr`, `tmin`, `dow`, `doy`, `actype_null`, and four record-mismatch flags
(`f_callsign`, `f_ades`, `f_type`, `f_rule`).

**SURF (14)** — `n_push`, `rwy_b30`, `apt_b30`, `arr_b30`, `sched_prev60`, `sched_next60`,
`sched_day`, `rwy_rate`, `dep_dur`, `rwy_dur`, `arr_dur`, `arr_ob60_taxiin`, `arr_ob60_cdiff`,
`arr_ob180_cdiff`.

**STANDHIST (4)** — `prev_arr_gap` (seconds since the previous arrival on-block at my stand),
`prev_arr_taxiin`, `prev_arr_cdiff`, `prev_dep_gap`.

**QUEUE (12, push-anchored; `scripts/stand_ab.py:92-113`)** — with `a = AOBT_3`, `t = MVT_TIME`:
`q_apt_at_push` `#{j: a_j < a_i < t_j}`, `q_rwy_at_push`, `q_pushed_after_me` `#{j: a_i < a_j < t_i}`,
`q_rwy_tko_in_taxi`, `q_rwy_push_pre20`, `q_dep_tko_sym15`, `q_rwy_tko_sym10`,
`q_arr_taxiing_at_push`, `q_arr_taxiin_sym30_push`, `q_arr_taxiin_sym30_tko`,
`q_rwy_ambient_proxy` (median peer `proxy` ±1 h on the runway, self excluded),
`q_next_arr_onblock_gap`.

**ORDER (3; `:955-967`)** — `o_dev_mvt` = (MVT_ID − rolling median of ±250 neighbouring rows) /
max(rolling p90−p10, 1); `o_dev_flt` same for FLIGHT_ID; `o_n_line` = window row count.

**DAY (9; `:798-810`)** and **WEATHER (10; `:1084-1096`)** blocks exist but are *not* in the shipped
arm F design.

**[S] Permutation vs drop-group value (`reports/FEATURE_IMPORTANCE_2026_09_10.md`):** the `clock`
group (17 columns incl. `proxy`) is worth **+36.6 s [31.5, 42.3]** on a drop-and-refit; every other
group is ≤ +3.9 s (`stand_id` +3.9, `stand_hist` +3.3, `airline_seg` +3.3, `queue_counts` +2.6,
`ordering` +1.9, `destination` +1.1, `actype_wake` +0.95, `recent_taxi` +0.64, `airport` +0.51,
`season_time` +0.30, `flags` −0.13). Two columns have **zero splits**: `sched_sec`, `actype_null`.
**[X]** i.e. the clock algebra is ~90 % of their matched model's feature value.

**[S]** Single-feature permutation on the HGB era (`reports/MATCHED_MODEL.md` §2):
`proxy` +117.9, **`aobt_eobt` +93.5**, `te_ars` +48.8, **`lobt_sched` +35.9**, `de_ars` +34.7,
`sp` +26.3, `aobt_sched` +15.4, `sched_prev60` +13.5. Their reading: "the gain is mostly **pairwise
differences between clocks the model already had** — `aobt_eobt` is `eobt_p − proxy`, `lobt_sched`
is `sp − lobt_p`. Trees cannot form the difference of two inputs."

### 4.2 Model family and hyperparameters

**[S]** Shipped matched lane (`scripts/lgbm_submit.py:174-178`):

```python
P = dict(objective="regression", metric="rmse", learning_rate=0.01, num_leaves=255,
         min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
         cat_smooth=20, min_data_per_group=100, max_cat_threshold=64, num_threads=4,
         verbosity=-1, seed=0)
NEST, PATIENCE = 50_000, 200        # early-stopping cap and patience
SEEDS = (0, 1, 2)                   # refit once per seed, average on the delta scale
```

Fit procedure (`scripts/lgbm_submit.py:36-45`): (1) 24 smoothed encodings fitted on all training
rows; (2) one early-stopping run with months (3, 9) held out → `best_iter` (v3: 25,689);
(3) refit on **all twelve months** at `n_ref = max(50, int(best_iter · n_all / n_fit))` (v3: 30,846
trees, 56 min, 4.51 GB), no early stopping; (4) `y_hat = max(proxy − delta_hat, 1)`, rounded.

**[S]** Reference HGB path (`scripts/build_submission.py:165-185`):
`HistGradientBoostingRegressor(learning_rate=0.05, max_leaf_nodes=127, min_samples_leaf=20,
max_iter=900, early_stopping=False)`, 3 seeds pooled + per-airport models (≥ 20,000 rows,
`max_iter=400`), `delta_hat = 0.5·pooled + 0.5·per_airport`.

**[S]** A gross-breakage guard on the stopping set (`scripts/lgbm_submit.py:196-201`):
`ES_MAX_RATIO_TO_PROXY_ONLY = 0.85` — a model that fails to beat "predict `delta_hat` = mean delta"
by 15 % on its own stopping set "has dead features, not bad luck" and the run aborts.

### 4.3 Fold structure

**[S]** `plans/PREREG_taxiout_2026_09_08.md:62-73` — group by airport-day, hold out **whole months**,
at least one winter and one summer month, report the **spread across held-out months**, never only
the mean.

* **Fold A (primary):** hold out **January + July 2025** (339,015 matched rows), train on the other
  ten (1,723,425 rows). Chosen because the evaluation set is Jan+Jul 2026.
* **Replication fold:** February + August (README:46-48). **[S]** Note: `reports/MATCHED_MODEL.md`
  §9 records the fold-B replication as **NOT RUN** at the time of writing.
* **Early stopping:** months **(3, 9)** carved out of the training months
  (`scripts/lgbm_submit.py:146`, `split_masks` at `:381-391`); ranking rows are in neither.
  So the shipped split is fit = 8 months / ES = 2 months / holdout = 2 months.
* **LOMO-12** (leave-one-month-out over all twelve) for every unmatched-lane arm.
* **Forward fold** measured once (`reports/TRAIN_SERVE_AUDIT.md` §4): at equal training volume,
  forward (train Jan–Jun → test Jul) 273.65 vs straddle 269.39 — **interleaving flatters by
  +4.26 s**, and they state plainly that every component number in the ledger was measured on the
  easier setting.

**[S]** A crucial fold caveat they discovered the hard way (`reports/MSE_LEDGER.md`,
"the fold is PESSIMISTIC" / Amendment 9): the fold's **TOTAL** has no stable absolute value — two
LFPG rows (y = 84,240 s and 58,206 s) are **54.7 % of the fold's stratum SSE and 27.0 % of its total
MSE**. The fold is sound for A/B on *matched* rows (n = 339,015) and must never be quoted as a
predicted board score.

### 4.4 Target encoding and how in-fold fitting is enforced

**[S]** 24 columns = 12 keys × 2 targets (`scripts/stand_ab.py:73-76`):
keys = `ADEP_mvt, STAND_mvt, stand_pref (STAND[:2]), airline (FLIGHT[:3]), ADES_mvt,
AIRCRAFT_TYPE_mvt, RUNWAY_mvt, AIRCRAFT_OPERATOR_flt, MARKET_SEGMENT_flt, WK_TBL_CAT_flt,
FLIGHT_TYPE_flt, ars`; targets = `te_*` (encode `y`) and `de_*` (encode `delta`).
**[S]** `ars` is `<airport>|<stand>|<runway>` on 100 % of cached rows (corrected in
`reports/PRIORITY1_IDENTITY_SCREEN.md` §2 after an earlier claim that it lacked the airport).

**[S]** The smoothing (`scripts/stand_ab.py:373-376`), `SMOOTH = k = 50.0`:

```python
enc = (sum_of_target_in_level + prior * k) / (count_in_level + k)   # prior = train mean
# unseen level -> prior
```

**[S]** Enforcement (`scripts/stand_ab.py:379-403`, `infold_encodings`): the encodings are fitted on
`keys[tr_mask]` only, and the docstring names the failure mode at the call site —
"The leak lives at this call site, not inside `_smooth_enc`: passing the full key array instead of
`keys[tr_mask]` lets held-out targets into the features and every downstream interval becomes
meaningless. Guarded by `tests/test_stand_fit.py`." A `delta_mask` argument exists so that on the
all-rows design the 12 `delta` encodings are fitted only where `delta` is defined; a NaN `delta` in
that set is **refused** rather than filled.

**[S] Their own known defect, BC-1 (`reports/bug_classes.md`, `reports/PRIORITY0_ENCODING_SEPARATION.md`):**
`fold_masks` puts the **early-stopping** months *inside* the training mask, so ES rows carry encodings
fitted on their own labels. Measured on 861,290 real rows: **24 of 24 encoding columns on ES rows move
when ES labels move** (0 of 24 after repair); mean displacement 17.8 s on `te_STAND_mvt` against
sd(delta) 378.4 s. Scope: the outer holdout is clean, so every reported RMSE and interval stands —
but **`best_iter` was chosen against a biased signal for every arm and every shipped version**
(sites: `lgbm_fold.py:1464`, `:1528`, `lgbm_submit.py:1452`, `:1106-1124`). The repair exists
(`prc/encoding.py`: stage-1 cross-fitted leave-one-**month**-out within the fit months; stage-2
leave-one-month-out within all training months) and was **never wired in**.

**[S]** Out-of-fold (cross-fitted) encoding, tested as a *predictive* change, **hurt: −0.96 s**
(README:36-38). **[X]** These two facts are not in tension: BC-1 is a model-selection defect, the
−0.96 s is a feature-quality measurement; they are different experiments.

### 4.5 The unmatched-row model (1.534 % of rows, ≈ 40 % of their board MSE)

**[S]** Base estimator, `scripts/build_submission.py:343-405` (`fit_unmatched`) — a **schedule-fill
mixture**:

```
pred = max( p · sp + (1 − p) · nf , 1 )        sp = MVT_TIME − SCHED_TIME
```

* `p` = P(the airport stamped the SCHEDULE into the off-block field), where "fill" is defined on
  training rows as `|BLOCK − SCHED| ≤ 60 s` (`schedule_fill`, `:254-258`).
  **[S]** It is violently airport-specific: **LIRF 48.5 % against 0.2–9.4 % elsewhere**, and pooling
  hides it entirely.
* `p` and `nf` are **hierarchically shrunk** cell estimates over three nested keys —
  `[ADEP] → [ADEP, sp_band] → [ADEP, sp_fine, dayoff]` — each level blended toward its parent with
  `K_SHRINK = 5.0` pseudo-counts (`:369-381`). `SP_EDGES = [-inf, 900, 1800, 3600, 7200, 10800,
  21600, 43200, inf]`; `dayoff = clip(date(MVT) − date(SCHED), 0, 1)`.
* At **LIRF only**, `p` comes from a logistic classifier instead of the cell rate
  (`_lirf_fill_model`, `:214-251`): features are smoothed log-odds of airline / stand-first-char /
  destination-prefix fill rates, **plus the airline's "copy propensity" estimated on LIRF's 159k
  MATCHED rows** where the copy is visible as `|BLOCK−SCHED| ≤ 60 ∧ |BLOCK−AOBT_3| > 300`, plus
  `dayoff` and `log1p(max(sp,0))`. `C = 1.0`. Established at −81 s on the stratum.
* `nf` (the non-fill term) is a **hybrid**: a fitted regressor for the body, the cell estimate for
  the tail — `nf_hybrid(cells, fit) = where(cells < 3000, fit, cells)` (`:307-316`). The body
  regressor (`NonFillRegressor`, `:262-296`) is `HistGradientBoostingRegressor(max_leaf_nodes=15,
  max_iter=200, min_samples_leaf=100)` on `sp, dayoff, hr` + 5 smoothed encodings, fitted on
  **non-fill rows only**, with the target **winsorised at ±3,000 s for the fit only**. It *refuses*
  to accept fill rows ("a fill row's label is `sp`, thousands of seconds, and would poison every
  encoding and the fit").
* **[S]** A plain gradient booster in place of the cell estimator scored **1793.6 vs 1648.4** — the
  cell estimator is deliberate, because of the heavy tail.

**[S]** Shipped evolution: E3C (`scripts/unm_congestion.py`) replaces the body's feature set with
`sp, dayoff, hr + hprox_med, hprox_n`, where the **witness** is, per airport × UTC clock hour of
take-off, the median `MVT − AOBT_3` over that hour's **matched** departures (NaN when fewer than 5);
unmatched rows never contribute. `p` and `nf_cells` are untouched. That shipped as v9 (−1,775 board MSE).

**[S]** LIRF gets its own rules lane (`configs/pipeline_v12.yaml:63-71`), applied in order:
1. `fill_classifier` — arm R, a **stake-weighted** LightGBM P(fill) trained on *all* LIRF departures
   (matched rows show the copy too), weight `clip(1+(max(sp,0)/3600)², 1, 25) × 8 if unmatched`
   (`scripts/rome_fill.py:44-46, 62-67`), swapped into the mixture in place of `p` only;
2. `dateslip` — inside segment G (LIRF, `dayoff == 1`, `sp ≥ 40,000`), replace the prediction by the
   two-class expectation `q·(86,400 + m) + (1−q)·sp`, `q` = Laplace date-slip share of the fold's
   training rows in the sub-band (`scripts/rome_dateslip.py`);
3. `schedule_floor` — for LIRF unmatched rows with `24,000 ≤ sp < 86,400`: `pred = max(pred, rint(sp))`
   (`scripts/rome_bandfloor.py:33-38`). Justification: in 2025 **every** LIRF unmatched row with
   `sp ≥ 24,000` had `y ≥ sp − 60`;
4. `local_day_schedule` (RLD) — `rint(sp)` when the schedule falls on the same **Europe/Rome local
   day**, added in v12.

---

## 5. Decision protocol — how a change is accepted or rejected

**[S]** The rules, in the order they were adopted:

1. **Whole months held out, never random rows**; encodings fitted inside the training fold only
   (README:44-52).
2. **Every reported RMSE carries a 95 % bootstrap interval; every A-vs-B claim is a *paired*
   bootstrap of the difference on the same rows, 2,000 resamples, seed 0.**
   "A gain whose interval spans zero is reported as not established."
   (README:50-52; `reports/MATCHED_MODEL.md` header.)
3. **Three seeds minimum, and the gain must exceed 2 × the measured seed sd.** Added after a
   self-caught failure: `cmd_fit` had hard-coded `random_state=0`, and across three seeds the same
   A1→A2 increment measured **+0.245 / −0.212 / −0.253 — sd 0.28 s, sign flips** against a published
   half-width of 0.39 s. "A single-seed paired row bootstrap certified *fitting variance* as an
   established gain." (`reports/MSE_LEDGER.md`, "The seed defect".)
4. **Gains are recorded in MSE, never in seconds** — "seconds are not additive and MSE is".
5. **A lever whose measured value is below 1,000 MSE is CLOSED**, and the reason is recorded so a
   later session does not reopen it with the same estimator. A **ship** needs **+500** weighted fold
   MSE (CAP was ESTABLISHED at +253 and *not shipped*); the bundle's Stage 1 bar was +800; the
   shortlist bar for a new arm is **+1,000**.
6. **Date-block (not row) bootstraps for any matched or tail-targeted arm.** Measured width ratios
   day-block / row: queue 1.59×, **arm F 4.60×**, W 1.58×, D 1.47×, Y-blend 1.02×. A row bootstrap is
   anti-conservative exactly where a gain lives on a few tail days.
   Rule from 2026-09-10: "any tail-targeted or matched arm is intervalled by day blocks."
7. **The heavy-tail correction to what a bootstrap even means:** the unmatched stratum's **Kish
   effective sample size on squared errors is 22.9, not 22,219** — "a 2,000-resample bootstrap over
   22,219 rows is arithmetically a bootstrap over ~23 observations."
8. **Pre-registration before every arm:** 27 numbered amendments plus 20 per-arm preregs in `plans/`,
   each with hypothesis, harness, locked TRUE/FALSE shapes and a locked verdict rule, written and
   timestamped *before* the run; RESULTs appended, never revised in place. Negative controls
   (permuted labels, permuted keys) are part of most clause sets.
9. **No leaderboard-driven tuning, at all** (`plans/PREREG_taxiout_2026_09_08.md:345-360`,
   Amendment 3 §1): "no threshold, hyper-parameter, feature, model or fold design may be chosen by
   comparing leaderboard scores. Submissions may be used only to confirm that a locally-computed
   number and the remote number agree." Amendment 13 restates it as binding.
10. **"No threshold may be narrower than the 95 % bootstrap interval of the quantity it tests."**
    On the unmatched stratum (RMSE 2,276, CI [1,576, 2,981], ±31 %) this means model selection by
    held-out stratum RMSE alone is *not possible* — stated as such (Amendment 3 §5).
11. **Harness validation against the board without spending a submission** (README:53-58): naive
    predictors reproducible by any team are matched to live leaderboard entries — constant 686.07
    fold vs 689.6901 board (two teams, identical to 4 dp), airport-mean 659.9 vs 660.5–663.0,
    airport×runway 653.1 vs 650.6, +stand 629.0 vs 625.3–627.6. **Agreement within ±5 s**, which is
    how they established the fold is not pessimistic at the *variance* level.

**[X]** The most transferable piece of this protocol is #5 + #6 together: a bar denominated in MSE
(so lanes are comparable) with an interval computed on the unit that actually varies (days, not rows).

---

## 6. Applicable to `resilient-hedgehog`

### 6.1 Side-by-side: their general reparameterisation vs ours

| | merry-quicksand | resilient-hedgehog (us) |
|---|---|---|
| Anchor clock | `AOBT_3_flt` (NM off-block) | `AOBT_3_flt` — `PROXY = "mvt_minus_aobt3"` [S] `src/prc/train.py:40` |
| Fitting target | `delta = BLOCK − AOBT_3` [S] `build_submission.py:494` | `TAXITIME − proxy` [S] `src/prc/train.py:385`, `docs/CODEX_BRIEF.md:54` |
| Reassembly | `y_hat = max(proxy − delta_hat, 1)` | `y_hat = proxy + g_hat` |
| Row scope | all matched rows, all 10 airports | all matched rows, all 10 airports |
| Second-clock offset for the fill class | `p·sp + (1−p)·(proxy − delta_hat)` mixture, per airport | model B target shifted by `mvt_minus_sched` at configured airports [S] `src/prc/train.py:98-109, 238-267` |

**The premise of this slot needs correcting, and it is the single most important finding here.**

**[X] Algebraically, our target is exactly their target with the sign flipped:**

```
ours:   TAXITIME − proxy = (MVT − BLOCK) − (MVT − AOBT_3) = AOBT_3 − BLOCK = −delta
theirs: delta            = BLOCK − AOBT_3
```

**[S]** Our own code states the identical rationale in the identical words —
`src/prc/train.py:9`: *"A tree cannot represent the identity part of `y = proxy − f(x)`, so the
reparameterisation is the point."* We adopted it as **v2, "2 クロック化", 345.35 → 312.02**
([S] `docs/handoff.md:81`), on **all matched rows**, not at LIRF only.

**So: our version of *their* reparameterisation is not a LIRF-only special case. We already have the
general form, on every matched row at every airport, and have had it since v2.** [S] `src/prc/train.py:379-396`:
`m_tr = tr.filter(PROXY.is_not_null())`, fit on `LABEL − PROXY`, predict `PROXY + g` — no airport
condition anywhere.

What *is* LIRF-only — and what bought the 20.67 s from 305.03 → 284.36 — is a **different
reparameterisation of a different lane**: model **B**'s (unmatched) target shifted by
`mvt_minus_sched` at the configured airports ([S] `src/prc/train.py:98-113`,
`unmatched_sched_reparam_airports = ("LIRF",)`; shipped in `configs/submit_v10_reparam.yaml:27-28`,
**not** in `configs/main.yaml`). Its mechanism is the schedule-fill class: at LIRF `BLOCK_TIME` is
stamped with `SCHED_TIME`, so on those rows the label degenerates to `y = MVT − SCHED = sp`
([S] `docs/data_audit_2026-09-18/README.md:3-6`).

**That mechanism is also in merry-quicksand's pipeline** — as the `p·sp` branch of the unmatched
mixture, the LIRF logistic, arm R and the schedule floor (§4.5). So both teams have both ideas;
the two systems are structurally the same family.

### 6.1b "Is ours a special case of theirs, and what would generalising ours cost?"

Precisely:

* **The matched-lane clock reparameterisation:** ours is *theirs*, sign-flipped, already general.
  Generalising costs **nothing — it is done**. The headline 285.24 → 252.66 is a gain we banked at v2.
* **The LIRF schedule-fill reparameterisation:** ours is genuinely LIRF-restricted, and
  **we have already measured what generalising it costs.** [S] `src/prc/train.py:109-113`:
  applying the same `mvt_minus_sched` shift to *all* unmatched rows measures **2,696.7** against the
  same **2,326.5** baseline, with **EHAM 954.7 → 3,408.9** — "because only LIRF has the fill
  mechanism". Restricted to LIRF it measures **2,326.5 → 1,632.1**, better on each of three folds,
  lower at nine of ten airports and higher at none
  ([S] `docs/investigation_2026-09-16/results/mms-mixture-reparam.md` §3.3, cited in that comment).
  **[X] So generalising this one costs ≈ +370 s of unmatched RMSE. Do not.** merry-quicksand reached
  the same conclusion by a different route and encoded it as a *probability* rather than a switch:
  their `p` is a per-row fill classifier whose cell rate is LIRF 48.5 % against 0.2–9.4 % elsewhere,
  so the mixture degrades gracefully to `(1−p)·body` at the nine airports where the mechanism is
  absent, instead of being on or off.
* **[X] The one genuine upgrade available to us from this comparison** is therefore not
  generalising the LIRF shift but *softening* it: replace our hard per-airport switch with their
  mixture form `pred = p·sp + (1−p)·body`, `p` fitted per row (their LIRF logistic's load-bearing
  feature is airline identity, estimated far more precisely on LIRF's **159k matched** rows where the
  copy is visible as `|BLOCK−SCHED| ≤ 60 ∧ |BLOCK−AOBT_3| > 300` — a training signal we are not
  currently using at all). Their stake weighting for the same classifier is
  `clip(1 + (max(sp,0)/3600)², 1, 25) × 8 if unmatched`.

### 6.2 Where the real difference is, then

**[X]** Board arithmetic with their exact weights `MSE = 0.9846596·m² + 0.0153404·u²`:

| | matched RMSE | unmatched RMSE | board |
|---|---|---|---|
| us, v10 | 218.0 (our OOF) → implied board consistent | implied **1,490.2** | 284.3584 [S] `docs/data_audit_2026-09-18/results/review-far-rows.md:153` |
| them, v10 | ≈ 222.6 (fold arm F) | ≈ 1,410 | 281.5182 [S] `reports/PIPELINE_TO_245_2026_09_10.md` §1a |
| them, v12 | ≈ 210.4 after ADS-B | ≈ 1,336 (back-solved) | 266.3738 |

**[X]** So at v10 our matched lane was *comparable or slightly better* than theirs, and the gap in
both lanes was small. Their 15 s of subsequent gain is **~2/3 matched (ADS-B, −7,722 MSE)** and ~1/3
unmatched/LIRF rules. Our remaining ~18 s to them is therefore *not* a target-parameterisation
question at all.

### 6.3 What is actually worth taking, ranked

1. **[X] The `|delta|`-band decomposition of our own matched error, and the y-target tail model.**
   Their RESULT 19 says the delta-anchored model is beaten by a y-target model on the rows where
   `|delta| > 20 min` by **+9.99 s [+5.63, +14.35]**, on rows carrying **31 % of matched SSE** —
   while losing 21 s on the 2–10 min body. Every *serve-time gate* they tried failed because LIRF
   reverses the sign. We have a per-airport model structure already (model A/model B), so a
   **LIRF-excluded, |delta_hat|-gated blend** is a cheap arm for us that they could not make work
   pooled (their Amdt 26 ex-LIRF arm was INCONCLUSIVE at +0.454 s ≈ 200 MSE — small, but positive,
   and 4 of 5 clauses held with the negative control passing at 4.5× p99).
2. **[S] The mechanism behind the matched tail, which we should test for directly:** 83 % of rows
   with `|residual| > 1,000 s` have `BLOCK − AOBT_3 < −600` — the airport stamps off-block 20–35 min
   *before* NM's, on flights with median NM delay 34–53 min, 73 % of it in July; the reading is
   TSAT/CTOT slot holds (`RED_TEAM.md` §4.3). Worth up to ~25 s of matched RMSE *if a column
   identified slot-held flights*. They could not find one in 30 columns. **[X]** Neither the
   `delta` nor the `y` parameterisation fixes this; it is an information problem, and it is the same
   17,208 wMSE (35 % of matched error) that their class table calls "tail".
3. **[S] The unmatched-lane structure, which is where our 1,490 vs their ~1,336 lives**: the
   hierarchical shrunk cell mixture (`[ap] → [ap, sp_band] → [ap, sp_fine, dayoff]`, k = 5), the
   **congestion witness** (airport-hour median `MVT − AOBT_3` over that hour's *matched*
   departures, n ≥ 5) as a body feature — +12/12 months, shipped, board −1,775 — and the
   **winsorise-the-body-fit-but-not-the-tail** trick (`nf_hybrid` at 3,000 s). Our own record says
   non-LIRF unmatched went 1,034.74 → 486.57 under one change and LIRF went 8,578 → 9,690
   ([S] `docs/handoff.md:242-243`), i.e. our unmatched error is LIRF-dominated exactly as theirs is
   (LIRF 65 % of their 2026 expected unmatched MSE).
4. **[S] Decision-protocol imports, cheap and immediately usable:** MSE (not seconds) as the unit of
   account; a **1,000-MSE close bar** and a **500-MSE ship bar**; **date-block** paired bootstraps for
   any matched/tail arm (their measured row-vs-day width ratio on a tail-living arm was **4.6×** —
   and our own `docs/CODEX_BRIEF.md:61-62` records a suspiciously narrow 0.39–0.60 s matched CI, which
   is exactly the symptom); ≥ 3 seeds with a `2 × seed sd` bar; and the **Kish ESS** check before
   believing any unmatched-lane interval (theirs: 22.9 effective observations out of 22,219 rows).
5. **[S] Two confirmed dead ends we should not spend a slot on:** one undivided model over all rows
   (they refuted it twice at capacity, matched 224.26 → 263.50, LIRF matched 384 → 623), and
   out-of-fold/cross-fitted target encoding as a predictive change (−0.96 s).
6. **[S] The governance flag:** the organisers were asked in writing whether `AOBT_3_flt` on ranking
   rows is a permitted input and **gave no ruling**. Our entire matched lane rests on the same
   assumption. Worth knowing; not worth acting on unilaterally.

### 6.4 One thing to *not* copy

**[S]** Their v11 gain (−7,722 board MSE, the largest single step) came from external ADS-B
(adsb.lol, ODbL-1.0) joined to the scored flights themselves — and the same unanswered organiser
question covers it ("whether openly licensed ADS-B of the scored flights themselves may be used").
**[X]** It is also, by their own census, worthless at four of our ten airports:
LFPG, LIRF, LEMD and LTFM had **zero on-ground samples** in the 2025 census; 2025 movable coverage by
airport was EHAM .72, LEBL .79, LSZH .70, EDDM .53, EDDF .51, LIRF .44, EGLL .30, LEMD .13,
LFPG .05, LTFM .00. Cost ≈ 190 GiB / ~20 h ingest plus a full stacker.
