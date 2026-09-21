# P9 — long-horizon rolling DURATION features on the arrival side

Author: SubAgent (theme P9). Date: 2026-09-21. Team `resilient-hedgehog`, official 284.3584.

Marking: **[S]** = read from a named source (file:line, or a repo path inside a cited research slot).
**[X]** = my own inference or judgement. **[M]** = measured by me today with read-only polars/numpy against
`data/raw/*.parquet`; every [M] figure is reproducible from the scripts described in §9.

Nothing in `src/` or `configs/` was modified.

---

## 0. One-paragraph verdict, up front

The mechanism P9 is built on is real and the data supports it: extending the arrival taxi-in rolling mean
from 60 minutes to 30 hours **monotonically increases** its correlation with our model's actual target
(the clock residual `AOBT_3 − BLOCK_TIME`), from −0.089 at 30 min to **−0.157 at 30 h** [M]. But when the
same feature is measured against a residual that has already been demeaned by `airport × hour × day-of-week
× month` — all four of which the model already carries — the correlation collapses to **+0.0038** [M]. The
long-horizon arrival-duration signal is, on our data, an almost pure re-expression of "which airport, what
time of day, what season". Incremental in-sample linear value of the whole four-window duration ladder on
top of the 30/60-minute columns we already have is **0.050 s** [M], against a measurement floor of
**0.25 s** [S, `docs/features_log.md`]. I recommend P9 be **staged behind P7 and behind slot 17's
`dis_state_30m`**, and that if it is run at all it be run as the exact block specified in §5 with the gap
audit of §6, because the audit is cheap, is reusable by every other theme, and closes a defect I found in
the existing coverage column (§6.4).

---

## 1. How our windows are actually built (required reading for every number below)

All references `src/prc/features.py` unless stated.

### 1.1 The two primitives

```
:103  def _counts_in_windows(anchor, events, w_sec):
:105      events = np.sort(events)
:106      before = searchsorted(events, anchor,        'left') - searchsorted(events, anchor - w_sec, 'left')
:107      after  = searchsorted(events, anchor + w_sec,'right') - searchsorted(events, anchor,        'right')
```

```
:111  def _window_mean(anchor, times, values, w_sec, lag_sec=0):
:114      order = argsort(times); times, values = times[order], values[order]
:115      ok = ~isnan(values); t_ok, v_ok = times[ok], values[ok]
:116      csum = concat([[0.0], cumsum(v_ok)])
:120      hi = searchsorted(t_ok, anchor - lag_sec,         'left')
:121      lo = searchsorted(t_ok, anchor - lag_sec - w_sec, 'left')
:122      n = (hi - lo); s = csum[hi] - csum[lo]
:125      mean = where(n > 0, s / n, nan)
:127      return mean, n
```

Seven properties that matter for P9, all **[S]** from the code above:

1. **Both bounds are absolute timestamps.** The window is the half-open interval
   `[t − lag − W, t − lag)`. There is no forward-fill, no `join_asof`, no "most recent value". A window
   that lands in a hole in the stream returns `n = 0`, **not a stale value from before the hole.** This is
   the single most important structural fact for §6.
2. **`lag_sec` shifts the whole window backwards** — it does not widen it. `[t − lag − W, t − lag)`. At the
   two call sites it is `60` for the departure-neighbour features (`:204`, `:206`) and `0` for the arrival
   taxi-in features (`:212`).
3. **Self-exclusion is by construction, not by subtraction.** `_window_mean` uses `side='left'` on the upper
   bound, so an event at exactly `t − lag` is excluded. For the departure-neighbour block the anchor is the
   row's own `MVT_TIME` and the events are other departures' `MVT_TIME`, so `lag_sec=60` additionally
   excludes every movement in the 60 s before the row's own take-off — the comment at `:198` says this is to
   "avoid ties". For the **arrival** block there is nothing to self-exclude: the anchor is a departure's
   take-off and the events are arrivals' in-block times, two disjoint populations. `lag_sec=0` is therefore
   correct there and not an oversight.
4. **NaN values are dropped from the pool before the cumulative sum** (`:115`), so a null `TAXITIME_SEC_mvt`
   neither poisons the mean nor inflates `n`. `n` is the count of *usable* observations.
5. **Empty window ⇒ `mean = NaN`, `n = 0`** (`:125`). This is the OpenAir/Phoenix/Javid convention, not the
   NASA 4th-place `if isnan: x = 0` convention which slot 17 §4.2 flags as worse [S, slot 17]. Keep it.
6. **Anchor-null rows are blanked wholesale** (`:215-221`): if `MVT_TIME` is null, every context column on
   that row is set to null after the fact via `anchor_ok`. On ranking this fires on zero rows
   (`MVT_TIME_UTC_mvt` null-rate 0.0 [S, slot 17 §0]).
7. **Everything is computed per departure airport** (`:169-172`), with an explicit null-airport group so
   `== None` cannot silently drop rows (`:171` comment). Arrivals are keyed on `ADES_mvt` (`:172`),
   departures on `ADEP_mvt`.

### 1.2 The existing long-ish columns

```
:166  for w in WINDOWS_MIN:                      # (5, 10, 15, 30, 60)  -- :32
:198  # neighbour state from NM times of other departures; lag 60 s to avoid ties
:203  for w in (30, 60):
:204      nbr_aobt_delay_mean_{w}m       = _window_mean(t, dep_times, AOBT_3 - EOBT_1, w*60, lag_sec=60)
:206      nbr_mvt_minus_aobt_mean_{w}m   = _window_mean(t, dep_times, MVT - AOBT_3,    w*60, lag_sec=60)
:208      nbr_dep_n_{w}m                 = the count from the first call
:209  # arrival taxi-in state (labels of arrivals are visible in ranking)
:211  for w in (30, 60):
:212      arr_taxi_in_mean_{w}m = _window_mean(t, arr_inblock, TAXITIME_SEC_mvt(ARR), w*60, lag_sec=0)
:214      arr_taxi_in_n_{w}m    = the count
```

The arrival block is anchored on the **departure's take-off time** `t` and the events are the **arrivals'
in-block times** `_t_inblock` (`:145-146`, `:165`). That is the right anchor: an arrival's taxi-in duration
is only *known* at its in-block moment, so using the in-block time as the event time is exactly the
availability-honest choice. **[S]** `:212`. **[X]** It is also stricter than Javid's `arr_taxi_mean_*`,
which uses a *centred* window `[t−W, t+W]` and is therefore batch-only [S, slot 17 F8g].

`WINDOWS_MIN = (5, 10, 15, 30, 60)` (`:32`) is the ceiling: **every context window in the repo is ≤ 60
minutes.** **[S]**

### 1.3 Where new columns have to live

`train.py:230`: `hidden = set(CANDIDATE_COLS) - set(cfg.include_candidates) | set(cfg.exclude_features)`.
Columns emitted by `context_features` go straight into the model; columns listed in `CANDIDATE_COLS`
(`features.py:276-286`) are computed always and **hidden unless a config names them**. The comment at
`train.py:126-129` records why: v6 shipped with all nine candidates including `noise_control` and scored
350.62 against v4's 305.03. **[S]**

**Therefore every P9 column must be registered in `CANDIDATE_COLS`**, or it cannot be screened one block at
a time and cannot be kept out of production by accident. One feature-cache rebuild (~4 min, `pipeline.py:75-101`)
then buys config-only switching. **[X]** This is an implementation constraint, not a preference.

---

## 2. The six rejected congestion experiments, with their windows

From `docs/features_log.md` (final table). All measured **against an equal-column noise control** at
`feature_fraction: 1.0`, `folds: [forward]`, `--skip-predict`, model B 127 leaves, seed 1; judgement level is
the 200-row-trimmed RMSE delta with a 95% CI. Negative = better. **[S]**

| # | column | family | window actually used | vs noise (200-row trim) | verdict |
|---|---|---|---|---|---|
| 1 | `queue_ahead` | queue count | **no fixed window** — `[own SCHED, own MVT]`, `features.py:325-332` | −0.16 [−0.40, +0.08] | rejected |
| 2 | `bank_position` | queue position | `dep_tko_prev_60m / (prev_60m + next_60m)`, `features.py:377-379` → **±60 min** | −0.19 [−0.41, +0.07] | rejected |
| 3 | `stand_rwy_share` | route/geometry | none (a static share) | −0.10 [−0.38, +0.14] | rejected |
| 4 | `active_runways_60m` | operating mode | **±30 min centred = 60 min span**, `features.py:364-370` | −0.06 [−0.29, +0.17] | rejected |
| 5 | `dep_arr_ratio_30m` | operating mode | **30 min**, `features.py:381-386` | +0.03 [−0.16, +0.22] | rejected |
| 6 | `congestion_vs_usual_30m` | operating mode | **30 min** ÷ per-`(airport, hour_local)` median, `features.py:388-393` | +0.09 [−0.16, +0.31] | rejected |

Four further rejections followed (`op_offset_med` −0.17, `dest_distance_km` +0.02, the two re-measured at
3 folds +0.15, `planned_flight_s` −0.21), giving the 10-loss run that closed the feature loop. **[S]**

Three things this table establishes and one it does not:

- **Establishes:** the longest window ever screened here is 60 minutes, and it was screened as a *count* or a
  *ratio of counts*, never as a *duration mean*. The 2 h – 30 h band is genuinely untested. **[S]**
- **Establishes:** `queue_ahead` is the endogeneity instance. Its window is `[SCHED_i, MVT_i]`, whose length
  is `mvt_minus_sched` — a column the model already holds (`features.py:61`) and which decomposes as
  `(MVT − AOBT_3) + (AOBT_3 − SCHED)`, i.e. the proxy plus another held column. It does not break at scoring
  time; it launders columns the model already has. **[S]** for the code, **[X]** for the decomposition
  argument (slot 17 §4.1 reaches the same conclusion independently).
- **Establishes:** the measurement floor. A single added column moves the score by ±0.5 s at
  `feature_fraction: 0.8`; the corrected screening resolution with `feature_fraction: 1.0` and a noise
  control is **0.25 s**. **[S]**
- **Does not establish:** that duration features fail. None of the six is a duration aggregate. Slot 17's
  count/duration split (§3.4 "The reconciliation") reads this table as six count-family rejections and is
  right to. **[S]**

---

## 3. What I measured today, before proposing anything

All figures **[M]**, on the 12 contiguous training months (2,085,047 DEP rows) unless the row says ranking.
The target used throughout is the quantity model A actually fits:
`resid = TAXITIME_SEC_mvt − (MVT_TIME_UTC_mvt − AOBT_3_flt) = AOBT_3 − BLOCK_TIME`, the clock residual
identified in `docs/features_log.md`. Rows trimmed to `0 < y < 7200` and `|resid| < 3600`
(2,060,618 rows survive).

### 3.1 The window ladder does get better with length — on the raw residual

`arr_taxi_in_mean_W` = mean arrival `TAXITIME_SEC_mvt` over arrivals whose in-block time is in `[t−W, t)`,
keyed on airport, anchored on the departure's own take-off. Exactly the existing construction at `:212`,
only wider.

| W | corr(feature, y) | **corr(feature, resid)** | corr(feature, resid) after `airport × hour × dow × month` demeaning |
|---|---:|---:|---:|
| 30 min *(exists)* | 0.0945 | −0.0887 | — |
| 1 h *(exists)* | 0.1096 | −0.1046 | — |
| 2 h | 0.1192 | −0.1187 | — |
| 3 h | 0.1203 | −0.1275 | **+0.0091** |
| 6 h | 0.1130 | −0.1314 | — |
| 12 h | 0.1216 | −0.1406 | — |
| 24 h | 0.1237 | −0.1545 | — |
| **30 h** | 0.1269 | **−0.1569** | **+0.0038** |

The middle column is the P9 hypothesis confirmed: |corr| with the residual rises monotonically from 30 min
to 30 h, a **77% increase**. The right-hand column is the P9 hypothesis destroyed: once the four calendar/
identity variables the model already holds are removed, **nothing is left**. **[M]**

The window ladder is also extremely collinear at the long end — corr(12 h, 30 h) = **0.957**,
corr(6 h, 30 h) = 0.890, corr(3 h, 30 h) = 0.857 [M]. A four-rung ladder in 3–30 h is nearly a one-rung
ladder.

### 3.2 The same test for the other two duration families

| family | value aggregated | corr(·, resid) at 30 min | at 3 h | at 30 h | after calendar demeaning (30 h) |
|---|---|---:|---:|---:|---:|
| `nbr_mvt_minus_aobt_mean` | neighbours' `MVT − AOBT_3` | −0.1615 | −0.1599 | −0.1676 | **−0.0006** |
| `nbr_aobt_delay_mean` (= slot 17's F9 `dis_state`) | neighbours' `AOBT_3 − EOBT_1` | **+0.2905** | +0.2459 | +0.2265 | — |

**[M]** Two conclusions:

- `nbr_mvt_minus_aobt_mean` is flat in W. Extending it buys nothing, and after calendar controls it is
  exactly zero. This is our legal substitute for NASA's `dep_taxi_30hr` (slot 05 item 7, slot 17 F8d–F8f)
  and the substitute does not carry the horizon effect. **[M]**
- **Peer lateness gets monotonically *worse* with window length** — 0.291 at 30 min down to 0.227 at 30 h.
  This is an independent confirmation of OpenAir's verdict, recorded verbatim in slot 17 §2 F9:
  *"Do not sweep extra windows."* [S] **Do not apply P9's long horizon to the F9 family.** **[M]** for the
  numbers, **[X]** for the recommendation.

### 3.3 Incremental value, the honest form

OLS of the clock residual on 10 airport dummies plus the named columns; in-sample RMSE in seconds.

| model | RMSE |
|---|---:|
| airport dummies only | 334.583 |
| + the six existing short columns (`arr_taxi_in_mean`, `nbr_mvt_minus_aobt_mean`, `nbr_aobt_delay_mean` at 30 & 60 min) | **326.191** |
| ↳ + `arr_taxi 3 h` | 326.172 |
| ↳ + `arr_taxi 30 h` | 326.154 |
| ↳ + `arr_taxi {3, 6, 12, 30} h` | **326.141  (−0.050)** |
| ↳ + `nbr_mva {3, 30} h` | 326.135 (−0.056) |
| ↳ + `nbr_aod {3, 30} h` | 326.158 (−0.033) |
| ↳ + `exp_deps_{15, 30} min` (the centred count, §5.3) | **325.924  (−0.267)** |
| ↳ + all eight of the above | **325.837  (−0.354)** |

**[M]** Read this against two calibration points:

- Our floor is **0.25 s** on the 200-row-trimmed CV delta [S, `docs/features_log.md`]. The four-window
  duration ladder is worth **0.050 s of in-sample linear RMSE**, five times below the floor.
- OpenAir measured the entire congestion block at **2.6 s in-sample** on our exact data
  (`OLS(AOBT proxy, dep15) 352.51 → +rwy15,arr15,queue 349.95`) and **0.0 s out-of-sample**
  (`303.53 → 303.56`) [S, slot 17 §3.2, §3.4]. P9's duration ladder is **fifty times smaller in-sample**
  than a block that measured zero out-of-sample.

These bound the effect rather than predict it — in-sample linear ≠ out-of-sample tree — but they bound it in
one direction only: a tree cannot extract more linear signal than exists, and the nonlinear residue would
have to be an order of magnitude larger than the linear part to clear the floor. **[X]**

### 3.4 Decile structure — is there nonlinearity a tree could find?

Mean clock residual by decile of the feature, after within-airport demeaning:

| decile | `arr_taxi_30h` | `nbr_mva_30h` | `exp_deps_15min` (raw) | `exp_deps_15min` (after `airport×hour×dow×month`) |
|---|---:|---:|---:|---:|
| d0 | −1.2 | −2.4 | **−38.6** | −13.0 |
| d1 | −6.0 | +3.3 | −25.6 | −7.2 |
| d2 | +2.4 | +0.6 | −9.4 | −1.1 |
| d3 | −4.2 | −4.3 | −1.2 | +0.4 |
| d4 | −7.3 | −3.9 | +5.5 | +0.7 |
| d5 | −5.9 | +6.6 | +8.4 | +1.9 |
| d6 | +0.3 | −1.9 | +5.7 | +2.3 |
| d7 | +8.1 | +11.3 | +5.9 | +1.2 |
| d8 | +4.5 | +4.4 | +12.7 | +2.3 |
| d9 | **+9.3** | **−13.7** | **+29.9** | **+10.3** |
| span | 16.6 s, **non-monotone** | 25 s, **sign flips at d9** | 68.5 s, monotone | 23.3 s, monotone but flat in d3–d8 |

**[M]** `arr_taxi_30h` has no usable nonlinear structure — the deciles wander within ±9 s of zero and are not
ordered. `nbr_mva_30h` is worse: its top decile flips sign. The only column here with a clean monotone
profile is the **centred count**, and its span shrinks by two-thirds once calendar is controlled.

### 3.5 The unmatched lane (model B) — checked separately, because the proxy is absent there

21,439 training rows with `FLIGHT_ID_mvt` null, `y` sd 884.8 s. Correlation with the **airport-demeaned raw
label** (there is no proxy to residualise against):

| feature | matched (2,062,193 rows) | **unmatched (21,439 rows)** |
|---|---:|---:|
| `arr_taxi 30 min` | +0.0139 | **+0.0476** |
| `arr_taxi 3 h` | +0.0031 | +0.0423 |
| `arr_taxi 30 h` | −0.0141 | **+0.0025** |

**[M]** On the lane where a duration feature has the best physical case — no clock proxy, so the feature has
to carry real taxi physics — the signal is in the **short** window and is gone by 30 h. This is the opposite
of the P9 hypothesis, on the only sub-population where P9's mechanism is unobstructed. **[M]** for the
numbers; **[X]** for the reading.

---

## 4. Availability — what is and is not computable, restated against the file

Verified independently today on `data/raw/ranking.parquet`, agreeing exactly with slot 17 §0 and slot 05 §2.

| stream | rows | `BLOCK_TIME_UTC_mvt` | `TAXITIME_SEC_mvt` | `MVT_TIME_UTC_mvt` |
|---|---:|---|---|---|
| ARR | 344,693 | 0 null | 0 null | 0 null |
| DEP | 344,841 | **100% null** | **100% null** | 0 null |

**[S]** Consequences for P9, unchanged from the brief but now line-cited:

- `arr_taxi_in_mean_W` at any W is computable for a scored row. It is the **only** place in the whole
  congestion catalogue where neighbours' *actual labels* are legally available to us [S, slot 17 #4].
- NASA's `dep_taxi_30hr` (slot 05 §5.3, ranked 3rd in importance at KMEM) is **uncomputable**: it aggregates
  other departures' `TAXITIME_SEC_mvt`, which is blank for all 344,841 scored departures. It was legal in the
  NASA competition because only the *query* flight's runway time was hidden there (slot 05 §2, KDD §2). It
  would be informative in training and all-NaN at scoring — the classic CV-works/LB-collapses failure
  [S, slot 17 §4 Trap 2]. **Do not build it.**
- The legal substitute is `nbr_mvt_minus_aobt_mean_W` on `MVT − AOBT_3`, which exists at 30 and 60 min
  (`features.py:206`) and which §3.2 shows is flat in W.
- NASA's `standtime_30hr` (mean of *first position → pushback* for arriving flights) has **no analogue**:
  we have no first-position/GUFI-creation timestamp. The nearest thing, the same-aircraft turnaround link,
  is available for only 14.77% of ranking departures [S, slot 05 item 12, `docs/data_profile.md` §7].

So of NASA's four top-5 congestion features at KMEM, **one transfers cleanly (`arr_taxi_30hr`), one
transfers as a proxy (`dep_taxi_30hr` → `nbr_mvt_minus_aobt`), one does not exist for us
(`standtime_30hr`), and one is a count (`exp_deps_15min`)**. [S] for each, [X] for the tally.

---

## 5. The feature design

### 5.1 V1 — `arr_taxi_in_mean_{3h, 30h}` (the primary P9 block)

| property | choice | why |
|---|---|---|
| **quantity** | mean of `TAXITIME_SEC_mvt` over ARR rows | the only real duration label available at scoring time [S, §4] |
| **grouping key** | `ADES_mvt == ADEP_mvt` of the query, i.e. the airport, one group, no runway/stand split | at 30 h a runway split would be a slow-moving average of the runway's *usage share*, not its congestion; and slot 17 F4a's rule that a missing stand must not form an `UNKNOWN` group [S] means a stand split would silently shrink the pool |
| **anchor `t`** | the query departure's own `MVT_TIME_UTC_mvt` | matches every other context column (`features.py:150`, module docstring `:8`); `MVT_TIME` is the only timestamp with a 0.0 null-rate on ranking DEP [S] |
| **event time** | the arrival's `BLOCK_TIME_UTC_mvt` (`_t_inblock`, `features.py:145-146`) | a taxi-in duration is first *knowable* at in-block; using the landing time would credit the window with information from the future of that arrival |
| **window** | `[t − W, t)`, `lag_sec = 0` | identical to `:212`; upper bound `side='left'` |
| **self-exclusion** | none needed | anchor population (DEP) and event population (ARR) are disjoint; `lag_sec=0` is correct and is not an oversight |
| **null policy** | `mean = NaN` when `n = 0`; NaN values dropped from the pool before the sum (`:115`) | the OpenAir/Phoenix/Javid convention; NASA 4th's `if isnan: 0` substitutes a legitimate-looking taxi time and slot 17 §4.2 marks it "arguably worse" [S] |
| **companion count** | **do not ship `arr_taxi_in_n_{3h,30h}`** as a model column | §6.3: the long-window count is the one quantity whose distribution genuinely shifts between train and score |
| **dtype** | `Float32`, via the existing `arr_taxi_in_mean` prefix rule at `:226` | the prefix match already covers a wider window; no code change needed there |

**The window ladder, and why only two rungs.**

Proposed: **`3h` and `30h`.** Justification, in order of weight:

1. **Collinearity.** corr(12 h, 30 h) = 0.957 and corr(6 h, 30 h) = 0.890 [M]. Four rungs in 3–30 h is
   effectively one-and-a-bit rungs of information, and each extra column costs a full measurement slot's
   worth of noise at our resolution (`docs/features_log.md`: one column = ±0.5 s at ff 0.8).
2. **The two rungs measure different things.** 3 h is the tail of the current operational state — it is the
   nearest extension of what `arr_taxi_in_mean_60m` already says. 30 h is "what kind of day is this airport
   having", the quantity slot 05 item 6 argues our six rejections never tested [S]. A 6 h or 12 h rung sits
   between two meanings and correlates 0.89–0.96 with the 30 h rung. **[X]**
3. **3 h is the gap-free rung.** At W = 3 h only **0.029%** of scored rows have a truncated window, against
   **3.482%** at 30 h [M, §6.2]. Shipping 3 h alone is the zero-risk half of the experiment.
4. **30 h is the literal NASA number**, not 24 h or 36 h — `deps_30hr`, `dep_taxi_30hr`, `arr_taxi_30hr`,
   `standtime_30hr`, and the 30-hour bound is the competition's own availability rule
   [S, slot 05 §5.3; KDD §2]. If we are testing the transfer, test the number.

Rejected ladder variants: `{2, 6, 12, 24, 30}` (five columns, 0.050 s of total linear value, unmeasurable);
`{2h}` alone (too close to the existing 60 min to be a new band); `{30h}` alone (loses the clean-gap control
rung).

### 5.2 V2 — the anomaly form, for free

Given V1, add **`arr_taxi_anom_60m_30h = arr_taxi_in_mean_60m − arr_taxi_in_mean_30h`**: today's taxi-in
against this airport's own recent normal. This is the self-normalising shape that slot 17's F10b
(`congestion_vs_usual_30m`) tried on *counts* and that was rejected at +0.09 [−0.16, +0.31] [S].

**Honest prior: it does not work.** I measured the ratio and difference forms of a
30-min-over-30-h normalisation directly: corr with the clock residual is **+0.0185** and **+0.0156**
respectively [M] — below every raw rung in §3.1. A tree with both rungs present can form the difference at a
split anyway. **[X]** Include it only if V1 is being built regardless, and only inside the same block so it
costs no extra measurement slot.

### 5.3 V3 — the centred count, a separate and clearly distinguished feature

This is NASA 4th's `exp_deps_15min`, the 2nd-most-important feature at KMEM [S, slot 05 §5.3 report].

```
exp_deps_sched_{W}min  =  #{ other DEP rows j at the same airport with SCHED_j ∈ [SCHED_i − W, SCHED_i + W] }
W ∈ {15, 30};  own row subtracted (−1);  key = ADEP_mvt;  never null (a count)
```

**Anchor:** the query flight's own `SCHED_TIME_UTC_mvt`, null-rate 0.0 on ranking [S, slot 17 §4].
An `EOBT_1_flt`-anchored twin is available at a 1.53% null rate and measures marginally better
(corr with the demeaned residual +0.0174 vs +0.0161 [M]); **use `SCHED`** because it is complete and because
the 1.53% null population is exactly the unmatched lane, which is routed to a different model anyway.

**Is it endogenous? No — and here is the precise line.**

The trap named in slot 17 §4.1 and by `OpenAir/DISCOVERY_REPORT.md` §5 is: *"Do not count takeoffs during
`[AOBT_i, MVT_i]` as a feature: that count scales with the target."* [S] The defining property is that the
**window's length is a random variable that contains the target**. Our own `queue_ahead` is an instance with
`[SCHED_i, MVT_i]`, whose length `mvt_minus_sched` decomposes into the proxy plus a held column (§2).

`exp_deps_sched_15min` has a **window of fixed length 30 minutes**. Only the *centre* moves, and it moves to
`SCHED_i`, a published schedule time that is (a) fixed before the flight ever pushes back, (b) not a function
of the label, and (c) not a function of any quantity that contains the label. The count therefore **cannot**
scale with the target by construction. It falls on the **exogenous side** of the line. **[X]** for the
argument; **[S]** for both the trap definition and the `exp_deps` construction.

Two residual caveats that must be stated, because "not endogenous" is not "not risky":

- **It is not take-off-anchored**, and so it breaks the module's stated design rule
  (`features.py:8`: *"Windows are anchored on the known take-off time and exclude the row itself"*). It is
  still availability-safe — `SCHED` is present for every scored row — but the invariant that every context
  column shares one anchor is lost, and any future audit of anchor behaviour has to special-case it. **[X]**
- **It is a batch-only look-ahead of a mild kind.** The pool is *realised* departures in the file, not
  *scheduled* departures, so the count conditions on those neighbours having actually operated. This is the
  same class of "legal for this leaderboard, would not exist in a live system" as the `dep_tko_next_*m`
  columns already at `features.py:167` and as Phoenix's whole following-context block
  (`"These observations do not establish availability before departure"` [S, slot 17 F2]). It is legal. It is
  not live. **[X]**

**Why V3 must be measured separately from V1/V2.** Three reasons. (i) It is a **count**, and the entire
motivation for P9 is slot 17's finding that every count family has a strong negative and no clean positive
[S]. Pooling it with the durations would let a count result masquerade as a duration result, or vice versa.
(ii) It is by some distance the strongest column in the whole P9 catalogue on our data — 0.267 s of the
block's 0.354 s of linear value, and the only monotone decile profile (§3.3, §3.4). (iii) Its incremental
value **over the trailing counts we already hold** is only 0.038–0.050 s (322.330 → 322.292 with `SCHED`,
→ 322.280 with both anchors, on the calendar-demeaned residual) [M] — so it too is below the floor, and
conflating it with V1 would make a below-floor block look like a below-floor block for the wrong reason.

---

## 6. The January-to-July hole, addressed head on

Slot 06 records the problem and explicitly records it as unaudited:

> *"Rolling / disruption features on the ranking set are computed within the ranking stream only, which is
> Jan 2026 + Jul 2026 — a stream with a six-month hole between the two months. At the January→July boundary,
> backward `join_asof` and trailing-window features will look back across that gap. Window clamps ... null
> most of these out ... so **[X]** the damage is probably limited to a handful of rows at the seam — but it
> is **not audited anywhere in the repo**."*
> — [S] `results/06-repo-openair.md:542`

That paragraph is about OpenAir's code. Our code is different in one decisive respect and identical in
another. Both are established below.

### 6.1 What a 30-hour trailing window actually does at 2026-07-01 00:00 UTC

Measured [M]:

- The ranking stream contains **exactly one gap**. Last January movement `2026-01-31 23:59:44`; first July
  departure `2026-07-01 00:07:17`; last January arrival in-block `2026-01-31 23:xx`. The gap from the last
  January arrival in-block to the first July departure is **3,600.29 hours = 150.01 days**.
  *(Correction to the brief and to slot 06's wording: the hole is five calendar months — February through
  June — not six. Jan and Jul are six months apart; five months are missing.)* **[M]**
- `_window_mean` bounds the window at `[t − W, t)` in absolute epoch seconds (`features.py:120-121`). For
  `W = 30 h` and any July anchor, the lower bound is at worst `2026-06-29 18:07`. The nearest event before
  that is 150 days earlier. **`30 h ≪ 3,600 h`, so the window is structurally incapable of reaching across
  the hole.** It returns an empty slice, `n = 0`, `mean = NaN` — never a January value. **[M] + [S]** for the
  mechanism.

**This is the answer to "what does it do".** It does **not** leak January into July. Our primitive is a
two-sided absolute-time window, not a `join_asof` and not an `ffill`; slot 06's concern applies to OpenAir's
`join_asof` construction and does **not** transfer to `features.py`. **[X]** for the transfer judgement;
**[S]** for both code shapes.

**What it should do:** exactly this. Empty is the correct answer for "what has this airport's arrival taxi-in
been over the last 30 hours" when the last 30 hours are not observed. The design decision is therefore not
*whether to clamp* but *how to declare the truncation to the model*, and how to prove the declaration behaves
the same in training and at scoring.

### 6.2 What it does do: truncation, at a rate 14× higher in ranking than in training

A window is **truncated** when part of `[t − W, t)` falls before the start of the contiguous block containing
`t`. Blocks detected by a gap threshold of 6 h. [M]

| W | ranking DEP rows truncated | share | training DEP rows truncated | share | **ratio** |
|---:|---:|---:|---:|---:|---:|
| 1 h | 32 | 0.009% | 10 | 0.000% | — |
| 2 h | 60 | 0.017% | 14 | 0.001% | 25× |
| **3 h** | **100** | **0.029%** | 23 | 0.001% | 29× |
| 6 h | 888 | 0.258% | 211 | 0.010% | 26× |
| 12 h | 4,635 | 1.344% | 1,886 | 0.090% | 15× |
| 24 h | 11,033 | 3.199% | 4,791 | 0.230% | 14× |
| **30 h** | **12,006** | **3.482%** | 5,081 | 0.244% | **14.3×** |
| 48 h | 22,711 | 6.586% | 10,144 | 0.487% | 13.5× |

The ratio is ~14× for every long window and it is not caused by the hole alone: ranking is 62 days with
**two** cold starts (1 Jan and 1 Jul), training is 365 days with **one** (1 Jan 2025). The hole doubles the
cold starts and the shorter span multiplies the share. **[M]**

**What the truncation does to the feature values** (30 h window, in-block-anchored arrival taxi-in mean):

| stream | seam rows | `n` median | mean of the feature | sd of the feature | NaN |
|---|---:|---:|---:|---:|---:|
| ranking, off-seam | 332,835 (96.52%) | 779 | 538.2 s | 173.5 | 0 |
| ranking, **seam** | 12,006 (3.48%) | **304** | **545.8 s** | 164.2 | **16** |
| training, off-seam | 2,079,966 (99.76%) | 798 | 537.2 s | 170.1 | 0 |
| training, **seam** | 5,081 (0.24%) | **254** | **550.3 s** | 175.1 | 16 |

**[M]** The mean shifts by **+7.6 s on a feature with sd 174** in ranking and **+13.1 s** in training — same
sign, same order, i.e. the truncated mean is close to unbiased and behaves the *same way* in both streams.
The **count** halves (779 → 304, 798 → 254). Only 16 rows in each stream reach `n = 0`.

**Design consequence, stated as a rule:**

- **Ship the long-window mean.** Its truncated-vs-untruncated shift is small, of the same sign and magnitude
  in both streams, and affects 3.5% of scored rows.
- **Do not ship the long-window count** `arr_taxi_in_n_{3h,30h}` as a model column. A column that halves for
  3.48% of scored rows and 0.24% of training rows is a train/serve distribution mismatch with no upside:
  §3 shows counts carry nothing here anyway.
- **Ship a correct coverage column instead** (§6.4), so the tree can condition on "this window is short"
  rather than being silently fed a halved count.

### 6.3 The two conventions the competitors use, and which one to adopt

| convention | who | effect on us |
|---|---|---|
| **month-scoping**: restrict every window pool to the query's own `%Y-%m`, so training and ranking are distorted identically | Phoenix, `nm_following_features.py` / `nm_neighbor_features.py` [S, slot 17 F2c/F8e] | would throw away 10 of 12 months of context for *every* training row in order to protect 3.5% of scored rows. Cost far exceeds the exposure. **Reject.** [X] |
| **independent processing**: *"Each supplied file is processed independently; ranking contains two disconnected months"* | Javid [S, slot 17 §4 Trap 3] | this is **already what we do**: `pipeline.py:98-99` calls `build_departure_features(train)` and `build_departure_features(rank)` separately, and `build_departure_features` masks and uses `context_source` = the same frame (`features.py:446`). **Adopt by default — no change.** [S] |

There is a third option worth recording because it eliminates the seam entirely: **a count-based window**
(the last K arrivals rather than the last W hours), the shape of OpenAir's `roll10` on `MVT − AOBT`
[S, slot 17 F8d]. K = 500 arrivals ≈ 30 h at a large airport and never truncates except at the literal first
rows of a block. It is gap-immune by construction but its *duration* meaning varies by airport, which is why
I propose it only as a fallback if the §6.5 audit shows the time-based version failing parity. **[X]**

### 6.4 A defect in the existing coverage column, found while doing this

```
:194  # how much of the widest look-back window is actually inside the frame: counts near a frame edge are
:195  span_lo = np.nanmin(dep_times) if len(dep_times) else np.nan
:196  cols["window_coverage_60m"] = np.clip((t - span_lo) / 3600.0, 0.0, 1.0)
```

`span_lo` is the **minimum over the whole frame** for that airport. On the ranking frame that is
2026-01-01. For a July row, `(t − span_lo)/3600 ≈ 4,300`, clipped to `1.0`. **[S]** for the code.

Measured: of the 192,122 July ranking departures, **0 are flagged with coverage < 1**, although **14** sit in
the first 60 minutes of July and genuinely have an empty look-back. In January the column works correctly and
flags 66 rows. **[M]**

At 60 minutes this costs 14 rows and nothing else. At 30 hours the identical construction would report full
coverage for all 12,006 truncated rows — i.e. it would report the exact opposite of the truth for 3.5% of the
scored set. **Any P9 coverage column must be built from the start of the contiguous block containing `t`,
not from `nanmin` of the frame.**

Specification:

```
blocks  = the maximal contiguous runs of the airport's movement times, split wherever the gap exceeds
          GAP_SPLIT_H = 6 hours          # 6 h > the longest genuine overnight lull; 3600 h >> 6 h
b(t)    = the start of the block containing t
ctx_hours_{W} = clip((t - b(t)) / 3600, 0, W) / W      in [0, 1], one column per shipped window
```

Ship `ctx_hours_3h` and `ctx_hours_30h` alongside V1. Fixing `window_coverage_60m` itself is **out of scope
for P9** — it changes an existing production column and would confound any P9 measurement. Log it as a
separate one-line candidate. **[X]**

### 6.5 The audit that proves training-time and serving-time behaviour match

Four assertions and one experiment. All read-only; none needs a model fit. Total cost ≈ 10 minutes of
compute, and the harness is reusable by every future long-window theme.

**A1 — the gap cannot be traversed (a static assertion, not a measurement).**
For each stream, compute the sorted movement times, the maximal inter-event gap `G`, and assert
`max(W) < G_min_block_gap` for every shipped window. Today: `30 h < 3,600.29 h`. Ranking is fixed, so this
holds for any W up to 3,600 h; the assertion exists to fail loudly if the organiser replaces the ranking file
again (it was already replaced once, 2026-09-04 [S, SUMMARY §A, slot 09]).

**A2 — no January event enters a July window.**
For every July scored row, assert `searchsorted(arr_inblock, t − W, 'left') == searchsorted(arr_inblock, t, 'left')`
whenever `t − W < 2026-02-01`, i.e. the slice is empty rather than January-filled. This is the direct
falsification of the "laundered January" hypothesis. It should pass for all 12,006 seam rows.

**A3 — the coverage column is correct at the seam.**
Assert `ctx_hours_30h < 1` for exactly the 12,006 rows in the first 30 h of each block and for no others, in
ranking; and for exactly 5,081 rows in training. This is the regression test for the §6.4 defect.

**A4 — the parity experiment. This is the one that actually proves train/serve agreement.**

Build the P9 columns twice on the **training** data:

  - *(a) contiguous*: the normal build, on the full 12-month frame — what training will really see.
  - *(b) ranking-shaped*: the same build on a frame restricted to **January 2025 + July 2025 only**, an
    artificial replica of the ranking stream's shape, with a real five-month hole in the same place.

Then, on the subset of rows present in both (all Jan-2025 and Jul-2025 departures, ≈ 330 k rows), report:

  | statistic | what it proves |
  |---|---|
  | fraction of rows with `|a − b| > 1 s` per column | should equal the 30 h truncation share of the Jan+Jul-shaped frame, and no more |
  | mean and sd of `(a − b)` on those rows | should reproduce the +8…+13 s / sd-174 picture of §6.2 |
  | RMSE of the *model's* prediction using (a) vs (b), holding the model fixed | **the decisive number**: the seconds of prediction movement attributable to the hole. If it is under ~0.05 s the hole is a non-issue for P9 and the theme can be judged on its merits alone |
  | the same, restricted to the seam rows | localises any effect |

A4 is the honest form of the question because it puts a *training* row through *ranking's* stream geometry
and measures the difference in the units we care about. It is also the piece slot 06 says nobody has done.
**[X]**

**A5 — the declared exposure.** Whatever A4 says, the plan pre-registers the exposure before any score is
read: 12,006 of 344,841 scored rows (3.482%) carry a truncated 30 h window; 100 (0.029%) carry a truncated
3 h window. If P9 is adopted and the leaderboard moves the wrong way, this is the first number to look at.

---

## 7. Staged plan, with the measurement attached to each stage

Measurement protocol throughout, from `docs/features_log.md` and `docs/README §6`: `folds: [forward]`
+ `--skip-predict` (≈5 min), weather on, model B 127 leaves, **`feature_fraction: 1.0`**, candidate block
versus a noise block **with the same number of columns**, judged on the **200-row-trimmed** delta with a 95%
CI that must exclude zero. **[S]**

One prerequisite that is currently missing: `features.py:408-411` defines exactly **one** `noise_control`
column. Screening a 3-column or 4-column block requires 3–4 independent hashed noise columns, or the
comparison measures the column-count change rather than the feature — the precise failure that produced the
false "3 本とも有害" conclusion in experiments 1–3 [S, `docs/features_log.md`]. **Add `noise_control_2..4`
in the same build. This is not optional.**

| stage | what | cost | pre-registered decision rule |
|---|---|---|---|
| **S0** | Write the audit harness (§6.5 A1–A3) as a standalone read-only script. No `src/` change, no model run. | ~1 h | Must pass A1, A2, A3 before anything else is built. A2 failing would invalidate the whole theme. |
| **S1** | Run A4, the parity experiment, against the **current** feature set at the existing 60-minute windows. | ~20 min | Establishes the baseline hole-exposure of what we already ship. If the current 60 m columns already move predictions by >0.25 s under ranking-shaped geometry, that is a bug report, not a P9 result. |
| **S2** | Build the block: `arr_taxi_in_mean_3h`, `arr_taxi_in_mean_30h`, `ctx_hours_3h`, `ctx_hours_30h` — registered in `CANDIDATE_COLS`, plus `noise_control_2..4`. One cache rebuild (~4 min). Re-run A4 on the new columns. | ~1 h | Report the A4 numbers **before** running CV. Pre-register them so the CV result cannot be rationalised afterwards. |
| **S3** | Screen the 4-column block against a 4-column noise block. | ~10 min | Adopt only if the 200-row-trimmed delta's 95% CI excludes zero. **Prior: it will not** (§3.3: 0.050 s of in-sample linear value against a 0.25 s floor). |
| **S4** | Screen `exp_deps_sched_{15,30}min` as a **separate** 2-column block against a 2-column noise block. | ~10 min | Same rule. **Prior: it will not** (0.038–0.050 s incremental over the trailing counts we hold). Record it as a *count* result, not a duration result. |
| **S5** | *Conditional on P7 landing.* Re-run S3 and S4 unchanged under the larger model. | ~20 min | Same rule. This stage exists because of §8; without it, an S3/S4 rejection is a rejection *at 127 leaves*, not a rejection of the family. |

**Expected effects, with sources:**

| claim | expected | source |
|---|---|---|
| `arr_taxi_30hr` is a top-5 LightGBM feature at one airport | importance, not ablation | [S] slot 05 §5.3, KDD report; their selection was greedy bottom-up with a real inclusion test [S] |
| the 2 h – 30 h band is untested here | true | [S] `features.py:32`; `docs/features_log.md` #1–#6 all ≤ 60 min |
| our duration ladder's incremental linear value | **0.050 s** (durations) / 0.354 s (whole 8-column block) | [M] §3.3 |
| our floor | 0.25 s | [S] `docs/features_log.md` |
| the whole congestion block on our data, out of sample | **0.00 s** (303.53 → 303.56) | [S] slot 17 §3.2, OpenAir `status.md:145-152` |
| arrival taxi-in specifically, in a competitor's hands | *"weak / unstable (EHAM −0.28, others ~0.1)"* | [S] slot 17 #4, `OpenAir/DISCOVERY_REPORT.md` §6 |
| a *neighbour-duration* family can be worth 1.4 s here | yes — Phoenix v12, 273.5557 official, −1.4181 | [S] slot 17 #2, slot 07 |
| …but only through a big enough model | +0.283 s through LightGBM679, rejected under their own 0.5 s gate | [S] slot 07 line 352, `Phoenix/METHOD.md` |
| peer lateness at 30 min is the one candidate above our floor | ~2 s, ≈1% of matched RMSE | [S] slot 17 §"Three judgements" #3, OpenAir C14: matched 256.46 → 253.82 |
| peer lateness does **not** benefit from a longer window | corr with residual falls 0.291 → 0.227 from 30 min to 30 h | [M] §3.2; agrees with OpenAir's *"Do not sweep extra windows"* [S] |
| seam exposure at 30 h | 3.482% of scored rows, 14.3× the training rate | [M] §6.2 |
| January cannot leak into July | gap 3,600.29 h ≫ 30 h; `_window_mean` is absolute-time bounded | [M] + [S] `features.py:120-121` |

---

## 8. Sequencing P9 against P7 (model capacity)

The measurement trap is real and is documented at source: the identical 30 following-context columns were
worth **+0.283 s through a 679-tree LightGBM** — below Phoenix's own 0.5 s gate, and rejected — and
**−1.4181 s official through a 4,999-tree depth-9 CatBoost**. Slot 07's own words:
*"a feature family can be worth nothing through one learner and 1.4 s through another — they only found the
v12 gain because they retried the same 30 columns with the deeper model"*, and
*"if we test it, test it with the deep model or we will wrongly reject it."* [S, `results/07-repo-phoenix-ops.md`
lines 352–356, 470]

**But the trap does not transfer to P9 in the direction it first appears to, and I want to be explicit about
why.** Phoenix's v12 columns are **same-day operational state at 15 and 60 minutes** — a quantity that is
close to orthogonal to the calendar. Capacity helps there because the signal is a high-order interaction the
small model cannot reach.

P9's long rungs are the opposite case. §3.1 measures their partial correlation after removing
`airport × hour × dow × month` and gets **+0.0038**. That is not a statement about model size; it is a
statement that the *information content* of `arr_taxi_in_mean_30h` is, on our data, almost entirely the
information in four variables the model already holds as splittable columns. A **larger model gets closer to
representing that interaction directly**, which shrinks P9's headroom rather than growing it. Our 127-leaf
model B sits somewhere between the unconditional correlation (−0.157) and the fully-conditioned one (+0.004);
a bigger model sits nearer the +0.004 end. **[X]** — this is my inference from [M] numbers, and it is the one
place in this plan where I am arguing against a well-sourced cross-team pattern.

**Recommended order:**

1. **P7 first.** It is a prerequisite for interpreting any P9 result, it is the change that unlocked
   Phoenix's single largest feature gain, and it is cheap relative to a feature build.
2. **Slot 17's `dis_state_30m` + `dis_frac20_30m` before P9.** It is the only congestion candidate anywhere
   in the catalogue with a pre-registered positive result on our dataset and an expected effect (~2 s) above
   our floor [S]. My own measurement independently supports both its window choice and its value: peer
   lateness has corr +0.291 with the clock residual at 30 min, the largest of any column I measured today,
   and it decays with window length [M]. Note we already compute `nbr_aobt_delay_mean_{30,60}m`
   (`features.py:204-205`) — the *mean* half of `dis_state` exists; what is missing is the exceedance
   fractions (`dis_frac20_30m`) and the explicit self-subtraction OpenAir applies.
3. **P9's S0–S2 (the audit) at any time.** It is read-only, it is independent of model capacity, it closes
   slot 06's unaudited item, and its harness is reused by every future long-window theme. This is the part of
   P9 I would fund unconditionally.
4. **P9's S3–S4 only if a measurement slot is spare after (1) and (2)**, and always re-run at S5 under P7's
   model before the result is written down as a rejection.

---

## 9. Honest statement of prior, and the disagreement between the two supporting slots

The two slots point different ways and both are correctly reasoned from what they read.

**Slot 05 supports P9.** Items 6 and 7 of its "Applicable to resilient-hedgehog" section say, in terms:
*"Extend the look-back windows to 2 h – 30 h ... Six of our ten rejected experiments were congestion features
all at ≤60 min, so the long horizon has never been tested"* and *"Rolling mean of realised durations, not
counts — using the arrival side, which survives ... extend it to 3 h and 30 h."* Its evidence is NASA 4th
place's top-5 importance list at KMEM, where three of five are 30-hour duration means. [S]

**Slot 17 does not.** Its closing recommendation is unambiguous: *"If exactly one slot is spent here, spend
it on #1 (`dis_state_30m` + `dis_frac20_30m`) ... It is the only candidate in this catalogue with a
pre-registered positive result on our dataset, the only one with a mechanism aligned to what our model
actually predicts, and the only one whose reported effect size (~1% of matched RMSE, ≈2 s) clears our 0.25 s
measurement floor. Everything else in the count families has an expected effect that our instrument cannot
resolve."* It ranks the arrival taxi-in family **4th** and attaches OpenAir's verdict on it:
*"weak / unstable (EHAM −0.28, others ~0.1)."* [S]

**Where the disagreement resolves.** Slot 05's evidence is *feature importance inside a fitted model in a
different competition*, and it explicitly flags this: *"this is importance within a fitted model, not an
ablation"* [S, slot 17 §3.4 making the same point about the same source]. Slot 17's evidence is an *ablation
on our data*. Ablation beats importance.

**My own measurements today side with slot 17, and add a mechanism slot 17 did not have.** The horizon effect
that slot 05 predicts is *visible* — |corr| with the clock residual rises 77% from 30 min to 30 h [M], which
is exactly what the NASA importance list would lead you to expect. It is also *entirely absorbed* by
`airport × hour × dow × month` [M]. That is a specific, falsifiable explanation for why the 30-hour window
mattered at KMEM and should not here: NASA's models made ~6–20 predictions per flight on a rolling 15-minute
grid with a thin calendar block (`hour, day, weekday, month, year` as five plain integers, one model per
airport) [S, slot 05 §5.3], so a 30-hour rolling mean was carrying information the calendar columns could not.
Ours already carries `month`, `dow`, `hour_utc`, `hour_local`, `minute_of_day_utc` and a 10-level airport
categorical (`features.py:80-98`, `:28-42`), plus per-fold target encodings.

**Prior, stated as a number.** The four-column duration block: **0.050 s** of in-sample linear value on top
of what we hold, against a **0.25 s** floor. My subjective probability that S3 returns a CI excluding zero at
127 leaves is **below 10%**; under a P7-enlarged model, **below 15%** — higher only because a larger model is
noisier, not because I expect the feature to work better. For S4 (the centred count) I would say **15%**, on
the strength of its monotone decile profile, which is the only structural evidence in this whole theme that
points the right way. **[X]**

**What would change my mind.** If A4 (§6.5) shows the current 60-minute columns already moving predictions by
more than 0.25 s under ranking-shaped stream geometry, then the hole is a live defect in production and the
whole framing of this plan is wrong — the theme becomes a bug fix, not a feature addition, and it jumps to the
front of the queue.

---

## 10. Reproducing the [M] figures

Read-only, no writes to `src/` or `configs/`. Scripts were run from the repo root with `.venv/bin/python`
against `data/raw/`. Each is a 30–60-line polars + numpy script reimplementing `_window_mean` verbatim from
`features.py:111-127`:

| § | what it computes |
|---|---|
| 3.1–3.2 | the window ladder for `arr_taxi`, `nbr_mva`, `nbr_aod` at W ∈ {0.5, 1, 2, 3, 6, 12, 24, 30} h on all 12 training months, and their correlations with `y`, with `resid = y − (MVT − AOBT_3)`, and with `resid` demeaned by `airport × hour × dow × month` |
| 3.3 | OLS of `resid` on airport dummies plus the named columns, in-sample RMSE |
| 3.4 | decile means of the within-airport-demeaned residual |
| 3.5 | the same split by `FLIGHT_ID_mvt` null |
| 6.1–6.2 | block detection by a 6 h gap threshold; truncation counts per window; seam-vs-off-seam `n`, mean, sd, NaN for ranking and training |
| 6.4 | `window_coverage_60m` reproduced exactly as `features.py:195-196` and evaluated on July ranking rows |

Every [M] number in this document came from those runs and no other source.
