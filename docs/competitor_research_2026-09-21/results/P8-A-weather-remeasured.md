# P8-A — the existing IEM weather join, re-measured against the current baseline

**Executed:** 2026-09-25, branch `claude/model-b-and-arrival-features`, git sha `3b40254` for **both** arms.
**Plan:** `docs/competitor_research_2026-09-21/plans/P8-weather-and-coverage-gate.md` §2.

## Verdict, plainly

**The weather join still earns its place. ADOPT, and keep `use_weather: true`.**
**The reason it earns its place is not the reason anyone thought.**

| | |
|---|---|
| full data (K=0) | **−1.81 s** [−2.76, −0.91]  P(B better) = 1.000 |
| worst 200 rows dropped (K=200) | **−1.26 s** [−1.62, −0.94]  P(B better) = 1.000 |
| month bootstrap (3 months) | **−1.81 s** [−2.47, −1.09]  P(B better) = 1.000 |
| `prc.compare.verdict` | **ADOPT** — *"−1.81 s, still −1.26 s at 200 rows dropped"* |
| re-weighted to the scored 44.3 % Jan / 55.7 % Jul | **−1.83 s** [−2.68, −1.06] |

Both halves of the decision rule are on the improving side, at every drop level (K = 0 / 20 / 200 / 2000 all
negative, all with P(B) = 1.000). The `[X]` 1-in-3 the plan put on "no longer significant" did not come in.

**But the gate-row story does not survive.** The −8.00 s that the weather block's reputation rests on was a
de-icing-gate-row figure. On gate rows today the effect is **−1.81 s [−14.10, +10.47]**, a CI twelve seconds
wide either side of zero on 2,625 rows. The gain is **uniform across gate and non-gate rows** and is **larger
in July than in winter**. Whatever weather is buying, it is not de-icing.

---

## 1. The measurement

`[S]` Two configs, `configs/p8_wx_off.yaml` and `configs/p8_wx_on.yaml`, generated from
`configs/main_gap.yaml` and differing in exactly two lines (`name`, `use_weather`) — verified by `diff` on
the uncommented bodies. `folds: [forward, jul_fwd]`, `n_seeds: 3`, `--skip-predict`. Both ran at git sha
`3b40254` with no source change between them.

`[S]` The plan specified `configs/submit_v10_reparam.yaml` as the base. That recipe is no longer production:
`main_gap.yaml` has since added `include_candidates: [stand_arrival_gap]`, `lobt_bound`,
`unmatched_sched_reparam` at LIRF and `es_holdout_encodings`. The whole point of P8-A is to measure against
the baseline we actually ship, so the pair is built from `main_gap.yaml`. That is a deliberate departure from
the plan's §2.2 and the only one.

| record | features | `jul_fwd` | `forward` | seconds |
|---|---|---|---|---|
| `p8_wx_off-f421484400` | 72 | 316.3209 | 221.5512 | 950.5 |
| `p8_wx_on-f61e61a25a` | 90 | 314.4240 | **219.7648** | 1014.0 |

`[S]` **Instrument check, free and passed.** `p8_wx_on`'s forward fold is **219.7648** against
`main_gap-62dd7c6e88`'s **219.72** and `jul_fwd` **314.4240** against **314.41**. The pair reproduces the
production record to within 0.04 s, which is inside the 0.017–0.028 s run-to-run spread plus the
four-fold/two-fold difference. The arms are what they claim to be.

---

## 2. Why the old number could not be carried forward, confirmed

`[S]` The only previous pair is `det_a-cfc9417c52` → `weather_check-1776e5e01c`: **288.880 → 285.746 =
−3.13 s**, one seed, four commits apart. Against it:

| | old pair | this pair |
|---|---|---|
| baseline forward RMSE | 288.88 | **221.55** |
| pure weather diff? | no — 4 commits (`c5e9908`, `9ab8758`, `8dc2152`, `34808da`) | **yes — one sha, one line** |
| seeds | 1 | 3 |
| forward matched effect | −0.87 | **−0.95** [−1.24, −0.72] |
| forward unmatched effect | −37.1 s on a 2,153 s unmatched RMSE | **−22.14 s** [−49.23, +9.73] on a 1,026 s one |
| gate-row effect | **−8.00 s** | **−1.81 s** [−14.10, +10.47] |

`[X]` The matched-lane number transferred almost exactly (−0.87 → −0.95). The unmatched number did not: it
halved and its CI now contains zero, which is what the plan predicted would happen once the LIRF schedule
reparameterisation removed most of the unmatched tail. The gate-row number is gone.

---

## 3. The breakdown (`scripts/p8a_weather_breakdown.py`)

Negative = weather-on better, throughout.

### 3.1 Per fold — never pooled

```
forward  n=328,009  221.55 -> 219.76   -1.79 [-3.22, -0.62]   K=200: -0.84 [-1.15, -0.53]  ADOPT
jul_fwd  n=190,713  316.32 -> 314.42   -1.90 [-2.98, -0.85]   K=200: -1.74 [-2.27, -1.17]  ADOPT
```

**Both folds adopt on their own.** `[S]` `jul_fwd` carries **zero `deicing_gate` rows in both years**, and it
is the fold with the *larger* effect and by far the more robust one (its K=200 number keeps 92 % of its
full-data effect; `forward`'s keeps 47 %).

### 3.2 Per month inside `forward`

```
2025-11  n=162,332  227.13 -> 224.66   -2.47 [-5.24, -0.27]
2025-12  n=165,677  215.95 -> 214.86   -1.09 [-2.20, -0.08]
```

`[S]` The old pair gave −1.14 / −0.61 on the same two months. Same ordering, both larger now.

### 3.3 Per de-icing gate — the finding

```
gate=0   n=516,097  259.75 -> 257.94   -1.81 [ -2.77,  -0.98]   P(B)=1.000
gate=1   n=  2,625  371.46 -> 369.65   -1.81 [-14.10, +10.47]   P(B)=0.646
```

**The two are the same number, and only one of them is measurable.** `[X]` The honest reading is that the
weather block's value is spread evenly over all rows, which is the signature of the **continuous** channels —
`temp_c`, `wind_ms`, `vis_km`, `dewp_spread_c`, `gust_ms` and their `_prev1h` copies, 10 of the 18 columns —
and not of the adverse-weather apparatus. That is consistent with three other things measured here:

- `jul_fwd`, where the gate is identically zero, has the **largest** effect (§3.1);
- on the matched lane the July fold is **−2.03** against winter's **−0.95** (§3.5) — more than double;
- four of the eighteen weather columns are constants and have been since the join landed (P8-D §3), so they
  cannot be contributing anything.

`[S]` Within gate=1 the per-airport pattern is as non-uniform as the old pair's, and worse:

```
EDDF   n=  321   -19.52 [ -33.14,   -6.34]      (old pair: -22.9)
EDDM   n=1,255   -15.79 [ -22.81,   -8.02]      (old pair:  -7.8)
LFPG   n=  138   -41.18 [-108.20,  +29.60]      (old pair: -19.4)
LSZH   n=  519    -5.08 [ -17.15,   +4.95]      (old pair:  -1.6)
LTFM   n=   83   +46.25 [  +1.84,  +94.18]      (old pair: +19.0)
EGLL   n=   64   +63.30 [ -37.51, +186.37]      (old pair: -33.2)
EHAM   n=  240  +139.54 [ +73.75, +196.99]      (old pair:  +6.7)
```

`[S]` **EHAM is +139.5 s on 240 gate rows with a CI nowhere near zero, and EGLL's sign has flipped since the
old pair.** `[X]` Two airports out of seven are being actively harmed inside gate hours. The net gate-row
effect is zero because EDDF and EDDM cancel EHAM and LTFM. This is a real lead and it is not P8-A's to chase:
the ranking set has **6,729 gate rows in January 2026 against 2,505 in January 2025** (×2.70), so whatever
EHAM's gate behaviour is, the scored set contains more of it than the training year does.

### 3.4 Per lane

```
matched    n=511,779  220.92 -> 219.52   -1.40 [ -1.61,  -1.19]   P(B)=1.000
unmatched  n=  6,943 1212.39 ->1202.18  -10.20 [-25.32,  +2.88]   P(B)=0.930
```

`[S]` **The adoption is carried entirely by the matched lane.** The unmatched effect is no longer
distinguishable from zero — the LIRF reparameterisation ate it, exactly as the plan expected.

### 3.5 Lane × fold, which is where the July result is clearest

```
forward  matched    n=325,004   199.50 -> 198.55    -0.95 [ -1.24,  -0.72]
forward  unmatched  n=  3,005  1026.19 ->1004.04   -22.14 [-49.23,  +9.73]
jul_fwd  matched    n=186,775   253.91 -> 251.88    -2.03 [ -2.37,  -1.66]
jul_fwd  unmatched  n=  3,938  1337.14 ->1333.72    -3.42 [-14.78,  +6.58]
```

`[X]` **Weather is worth more than twice as much on a July matched row as on a winter one**, on tight CIs
in both cases. Every de-icing-based account of why weather helps predicts the opposite.

### 3.6 Re-weighted to the scored composition

```
pooled (row counts as they fall)    n=518,722   260.43 -> 258.63   -1.81 [-2.71, -1.03]
re-weighted 44.3% Jan / 55.7% Jul   n=518,722   278.35 -> 276.52   -1.83 [-2.68, -1.06]
```

`[S]` The plan warned that reporting a pooled CV delta as an LB delta would overstate the winter component by
roughly 2.3×. **It does not, here, and the reason is the finding**: the two folds agree, so re-weighting them
changes the answer by 0.02 s. The 2.3× correction is only needed for a winter-only effect, and this is not
one. `[X]` The re-weighted line remains an estimate: `forward` is Nov+Dec 2025 standing in for a January we
have no honest fold for, and `jul_fwd` is Jul 2025 standing in for Jul 2026.

---

## 4. Two caveats that belong on the number

`[S]` **1. `use_weather` moves 18 columns, so the arity effect is inside this comparison.** A signal-free
column moves this model by ±0.5 s (`docs/features_log.md:44-52`), and `main_gap.yaml`'s own noise arm
measured +0.23 s. There is no 18-column noise arm and the repo has no mechanism for one. `[X]` The direction
is known: adding columns costs, so the arity effect works **against** the weather-on arm, and −1.81 s is if
anything a floor on the information value. It is not a reason to discount the result; it is a reason not to
quote −1.81 s as "the value of weather information" without saying so.

`[S]` **2. There is no honest January fold and there cannot be one** (`cv.py:39-42`). The scored set is
44.3 % January and the best winter proxy available is Nov+Dec 2025. Everything in §3 about January is a
proxy measurement. The specific thing this cannot settle is unchanged from the plan's §2.3: whether the model
extrapolates correctly at LTFM and EDDF, where January 2026 has more gate rows than the whole 2025 training
year — and §3.3 now adds EHAM to that list with a measured +139.5 s inside gate hours.

---

## 5. A correction owed to `docs/README.md` §6

`[S]` `docs/README.md:109` records the weather row as CV −3.13 [−5.23, −1.26] / K=200 −1.06 [−1.40, −0.73] /
**LB −5.73**. The CV half is `det_a` → `weather_check`, i.e. the **fixed** join. The LB half is v2 (312.02) →
v3 (306.29), and v3 was built with the **broken** join — `weather.py:167-168` says so in as many words
(*"v3 and v4 were built that way"*). **The single calibration case that underwrites our whole decision rule
pairs a fixed-join CV delta against a broken-join LB delta.** That should be recorded as a caveat whatever
else P8 does; `docs/features_log.md` has been appended with this result and the caveat.

---

## 6. SHIP / DO-NOT-SHIP

**Nothing to ship. `use_weather: true` stays as it is; no config or source change comes out of P8-A.**
`configs/p8_wx_on.yaml` / `configs/p8_wx_off.yaml` and `scripts/p8a_weather_breakdown.py` are the audit
trail, not a candidate.

**The stop rule is not triggered.** `[S]` Plan §6.4 says to stop P8-B and P8-C if the weather block is no
longer significant at K=200. It is significant at K=200 on both folds separately and pooled. **P8-B proceeds.**

`[X]` And §3 sharpens P8-B's prior rather than leaving it at the plan's ~25 %. The plan's case for the
`wxcodes` repair was that it adds a channel that is live in July, where the de-icing apparatus is flat. §3.1
and §3.5 now say that **July is where the existing weather block already earns most of its keep**, which is
the opposite of what a de-icing account predicts and is the single best argument in the theme for repairing
the July-live channel rather than the winter one.
