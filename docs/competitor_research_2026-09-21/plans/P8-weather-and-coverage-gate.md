# P8 — re-measure the weather join, and a coverage-parity gate for any external data

**Planner:** theme P8. **Date:** 2026-09-21. **Team:** `resilient-hedgehog`, best LB **284.3584**.

**Marking:** `[S]` = read from a named file at a named line, or measured by me in this sandbox with a
reproducible command. `[X]` = my inference, never dressed up as a measurement. Every number below that is not
marked `[X]` was either read out of a file in this repo or computed here today.

**Everything I ran was read-only.** No file under `src/` or `configs/` was modified.

---

## 0. Bottom line, before the detail

Three separate things are bundled under "weather", and they have very different expected values. Splitting
them is the main service this plan performs.

| step | what it is | download cost | my honest expectation |
|---|---|---|---|
| **P8-A** | Re-measure the existing IEM join with a same-sha on/off pair against the **current** baseline | zero (cache on disk) | **Worth doing, but as a defensive audit, not a gain.** The −8.00 s gate-row number was measured against a 288.88 s baseline that no longer exists; today's forward-fold baseline is **223.51** `[S]`. The gain may not survive. |
| **P8-B** | **Repair four identically-zero weather columns using `wxcodes`, which IEM already serves** | **zero** | **The best-value item in the theme, and it was not in the tasking.** I measured it today: `precip_mm` is `0.00` on **100 %** of 138,446 cached station-hours, yet `wxcodes` carries rain on **31,534 scored departure rows (9.1 % of the scored set)**, including **5,535 rows in July 2026** — the only weather channel that touches July at all. |
| **P8-C** | Add Open-Meteo `snowfall` / `precipitation` | 10 HTTP calls | **Second-order.** Verified live today; it does fill the hole. But P8-B extracts most of the same information from data already on disk, so C should be measured *after* B and judged against B, not against nothing. |
| **P8-D** | Coverage-parity gate | zero | **Do it, cheaply, and change the rule.** kind-mango's gate as written would **pass** our single worst defect (a dead column drops by 0 points). I ran their rule on our frames today; the extension it needs is specified in §5. |

**Overall verdict on the theme: worth about one day, and only because of P8-B and P8-D.** The measured
ceiling on weather in this competition is small and the evidence is mostly negative (§6). What makes this
theme non-zero is not "add weather" — we already have weather — it is that **four of our eighteen
model-visible weather columns are identically zero**, and nothing in our pipeline notices.

---

## 1. What is actually in `src/prc/weather.py`, and what "the broken join" was

### 1.1 The broken join was a name collision, and it is already fixed

`[S]` `src/prc/weather.py:165-169`, verbatim:

> `# The join key is named for what it is and never reuses a feature name. An earlier version called it`
> `# "hour_utc", which is also the integer hour-of-day feature: turning weather on overwrote that column with a`
> `# timestamp and then dropped it, so every weather run silently trained without hour_utc. v3 and v4 were built`
> `# that way.`
> `WX_KEY = "_wx_hour"`

So the defect was: the weather join created a column called `hour_utc` holding a truncated timestamp, which
collided with the existing `hour_utc` integer hour-of-day feature; the timestamp column was then dropped
before training, taking the real feature with it. **Turning weather on silently deleted the hour-of-day
feature.**

`[S]` **It is fixed, and the fix is already in the cached frames.** The fix commit is `34808da`
(`docs/investigation_2026-09-16/results/seasonal-deicing-shift.md:226`). The investigation verified the
current cache directly: *"`hour_utc` in `feat_train` is `Int8` with range 0–23, i.e. the cached features carry
the hour-of-day feature and were built with the `WX_KEY` join"* and *"re-joining the METAR table on
`(ADEP, truncate(MVT_TIME, 1h))` reproduces the cached `deicing_gate` and `temp_c` on every row of both
frames (0 mismatches)"* (`seasonal-deicing-shift.md:204-207`). `[S]` I re-confirmed the frame schema today:
`data/work/feat_train.parquet` has `hour_utc: Int8`.

`[S]` A second investigation quantified how much the bug cost, and the answer is **almost nothing**:
`hour_utc` carries **0.24 % of model B's gain**, and removing it *lowered* the December holdout RMSE from
1,694.2 to 1,653.6 (`hour-utc-absent-in-v3v4.md:59-72, 148-150`).

`[X]` **So "our weather join was broken" is true as history and misleading as a present-tense statement.**
The join is fixed; the cached features are correct; the thing that is broken *now* is different and worse
(§1.3).

### 1.2 What is computed today

`[S]` `weather.py:22-27` — IEM ASOS/METAR, `report_type=3` (routine hourly), fields
`tmpf, dwpf, relh, drct, sknt, gust, p01i, vsby, wxcodes, skyc1, feel`, 1.2 s throttle, public domain.
Cache: `data/ref/metar/{ICAO}_2025-01-01_2026-08-01.csv`, ten files, on disk `[S]` (`ls -la data/ref/metar/`),
plus the derived `data/ref/metar_hourly.parquet` (138,446 station-hours). **The cache already spans both
scored months; there is no download cost for P8-A or P8-B.**

`[S]` `weather.py:126-162` `hourly_weather_table` produces, per station-hour:
`temp_c`, `dewp_spread_c`, `wind_ms`, `gust_ms`, `precip_mm`, `vis_km`, `frozen_wx`, `deicing_gate`,
`low_vis`, `temp_x_precip`.

`[S]` `weather.py:182-203` `join_weather` attaches those ten at the take-off hour plus ten `_prev1h` copies,
keyed `(ADEP_mvt, truncate(MVT_TIME_UTC_mvt, 1h))` (`weather.py:201`), with an explicit time-zone
reconciliation (`weather.py:188-200`) because the competition parquet is tz-aware and the METAR parse is naive.

`[S]` `src/prc/pipeline.py:41-52` `maybe_join_weather` is the only caller; it is gated on `use_weather`
(`configs/main.yaml:15` = `true`).

`[S]` **Only 18 of the 20 columns reach the model.** `frozen_wx` and `frozen_wx_prev1h` are `Boolean`, and
`numeric_feature_columns` (`src/prc/features.py:466-473`) admits only float/int dtypes, so they are dropped
(`findings.md:647-649`). Confirmed by arithmetic on the records: `weather_check` `n_features` 90 minus
`det_a` 72 = **18** `[S]` (`experiments/experiments.jsonl`).

### 1.3 The defect that is live right now: four of those eighteen columns are identically zero

`[S]` **Measured today on the raw CSVs**, all ten stations, `awk` on field 9 (`p01i`):

```
EDDF tot=13848 p01i_nonnull=13848 p01i_gt0=0   sum=0.00
EDDM tot=13848 p01i_nonnull=13848 p01i_gt0=0   sum=0.00
EGLL tot=13846 p01i_nonnull=13846 p01i_gt0=0   sum=0.00
EHAM tot=13844 ...  LEBL LEMD LFPG LIRF LSZH LTFM  — all identical: gt0=0, sum=0.00
```

**This sharpens the established fact rather than merely confirming it.** The tasking says `p01i` is "empty by
design". It is not empty — IEM serves the literal string `0.00` on **100 %** of rows. The column is
**non-null and constant zero**. That distinction matters for §5: a non-null coverage check *passes* it.

`[S]` Consequence, measured today on `data/work/feat_train.parquet` and `feat_rank.parquet`:

| column | non-zero share, all 2,085,047 training rows | non-zero share, all 344,841 ranking rows | distinct values |
|---|---|---|---|
| `precip_mm` | **0.0000 %** | **0.0000 %** | 2 (`0.0`, null) |
| `precip_mm_prev1h` | **0.0000 %** | **0.0000 %** | 2 |
| `temp_x_precip` | **0.0000 %** | **0.0000 %** | 2 |
| `temp_x_precip_prev1h` | **0.0000 %** | **0.0000 %** | 2 |
| `deicing_gate` | 0.4392 % | 1.9513 % | 3 |
| `low_vis` | 0.8915 % | 0.9109 % | 3 |

**Four of the eighteen weather columns given to the model — 22 % of the weather block — are constants.**
`[X]` They cost nothing in bias but they are not free: `docs/features_log.md:44-52` measured that **adding one
signal-free column moves this model by ±0.5 s** under `feature_fraction: 0.8`, which is twice the instrument's
0.25 s resolution. Four dead columns is four draws of that lottery, baked into every run since the weather
join landed.

`[S]` And the `deicing_gate` definition at `weather.py:154` is
`(temp_c <= 3.0) & ((precip_mm > 0) | frozen_wx)`. Since `precip_mm > 0` never fires, the gate reduces
**exactly** to `frozen_wx ∧ temp_c ≤ 3` — confirmed on all 1,326 gate hours with 0 disagreements
(`seasonal-deicing-shift.md:60-62`).

### 1.4 The thing nobody has noticed: `wxcodes` already carries the missing precipitation

`[S]` **Measured today** by parsing the `wxcodes` column of all ten raw CSVs (138,446 rows). Top present-weather
codes across the cache:

```
-RA 4620 | BR 3655 | -SHRA 3301 | VCSH 1692 | BCFG 1326 | SHRA 1073 | -DZ 847 | FG 845
-RADZ 565 | RA 544 | MIFG 538 | FZFG 413 | VCFG 354 | -SN 288 | VCTS 268 | -DZRA 222
-SHSN 216 | PRFG 175 | DZ 164 | +SHRA 118 | -SHRASN 88 | HZ 71 | -TSRA 70 | SN 64 | TSRA 62
```

Present-weather codes are populated on **7.3 %–24.0 %** of station-hours depending on the airport
(LEBL 7.3 %, EGLL 10.4 %, LSZH 24.0 %) `[S]`.

`[S]` `weather.py:123` `SNOW_CODES = ("SN","SG","PL","IC","GS","GR","FZ")` — the module reads `wxcodes`
**only** for frozen precipitation. Every rain, drizzle, shower, thunderstorm, mist and fog code is parsed,
stored in the `wx` string column, and then discarded: `hourly_weather_table` never surfaces it, and `wx` is a
`String` so it could not reach the matrix anyway.

`[S]` **Row-level exposure, measured today** (rain-family codes = any of `RA`, `DZ`, `SH`, `TS`), joining the
raw `wxcodes` back onto the cached feature frames on `(ADEP_mvt, take-off hour)`:

| month | departure rows | rows in a rain-coded hour | share | rows with **any** wx code | `deicing_gate` rows | `low_vis` rows |
|---|---|---|---|---|---|---|
| **2026-01 (scored)** | 152,719 | **25,999** | **17.02 %** | 40,873 | 6,729 | 3,141 |
| **2026-07 (scored)** | 192,122 | **5,535** | **2.88 %** | 5,923 | **0** | **0** |
| 2025-01 (train) | 153,706 | 21,646 | 14.08 % | 35,193 | 2,505 | 4,396 |
| 2025-07 (train) | 190,713 | 18,512 | 9.71 % | 19,084 | 0 | 54 |
| 2025-11 (fwd fold) | 162,332 | 20,889 | 12.87 % | 29,608 | 1,184 | 3,174 |
| 2025-12 (fwd fold) | 165,677 | 18,247 | 11.01 % | 29,034 | 1,441 | 3,932 |

**Read the July row.** In July 2026 — **55.7 % of the scored set** — `deicing_gate` is zero on every row,
`low_vis` is zero on every row, and `precip_mm` is zero on every row. The entire "adverse weather" apparatus
is **provably constant** across the larger half of what we are scored on. The only weather channel that is
non-constant in scored July is the rain-code channel we currently throw away (5,535 rows) plus the continuous
`temp_c` / `wind_ms` / `vis_km` / `dewp_spread_c`.

**Scored-set totals:** rain-coded rows **31,534 = 9.14 %** of the 344,841 scored rows, against `deicing_gate`
at **6,729 = 1.95 %**. `[X]` A rain flag is therefore **4.7× more exposed** than the de-icing gate on the set
that decides our score — and unlike the gate, it is live in both scored months.

**The caution attached to it, measured today:** Jul 2026 was a dry July. Rain-coded share fell **9.71 % →
2.88 %** (−6.8 points, month-matched). Per airport, Jul 2025 → Jul 2026 rain hours: EGLL 50 → 3, LEBL 51 → 5,
LFPG 63 → 5, LIRF 14 → 2, LEMD 13 → 7, EDDF 147 → 29, LSZH 169 → 51, EDDM 175 → 52, EHAM 94 → 39, LTFM
2 → 31 `[S]`. That is a **support** shift, not a missingness one (the column is present and correctly zero),
and it passes the 20-point rule — but it means a July rain feature will be much less active at serve time
than at train time, which is exactly the class of thing §5 exists to surface.

---

## 2. P8-A — re-measuring the existing join properly

### 2.1 Why the existing number cannot be carried forward

`[S]` The only weather on/off pair in the repository is `det_a-cfc9417c52` (sha `70ef994`, `use_weather: false`,
72 features) versus `weather_check-1776e5e01c` (sha `34808da`, `use_weather: true`, 90 features), forward fold,
one seed each. Record-level **288.880 → 285.746 = −3.13 s** (`experiments/experiments.jsonl`).

Three reasons that number is not usable as today's answer:

1. `[S]` **It is not a pure weather diff.** Four commits separate the two: `c5e9908` (record), `9ab8758`
   (perf), `8dc2152` (`force_row_wise`), `34808da` (weather fix) — stated at
   `seasonal-deicing-shift.md:227-228` and listed as unresolved at `seasonal-deicing-shift.md:277-279`.
2. `[S]` **The baseline moved by 65 seconds.** `det_a` forward = 288.88; the current head-of-line experiment
   `submit_v10_reparam-77f92c802d` forward = **223.51** (`experiments/experiments.jsonl`). The LIRF schedule
   reparameterisation removed most of the unmatched tail. `[X]` A −0.87 s matched / −37.1 s unmatched effect
   measured against a 2,153 s unmatched RMSE has no reason to reproduce against a 1,028 s one.
3. `[S]` **`docs/README.md:109` mixes two different joins in one row.** The `気象` row reads
   CV −3.13 [−5.23,−1.26] / K=200 −1.06 [−1.40,−0.73] / **LB −5.73**. The CV figures are `det_a` →
   `weather_check`, i.e. the **fixed** join; the LB −5.73 is v2 (312.02) → v3 (306.29)
   (`configs/weather_check.yaml:1-3`), and v3 was built with the **broken** join
   (`weather.py:167-168`: *"v3 and v4 were built that way"*). **The one calibration case that underwrites our
   entire decision rule pairs a fixed-join CV delta against a broken-join LB delta.** That should be recorded
   as a caveat on §6 of `docs/README.md` whatever else P8 does.

### 2.2 The measurement

Two configs, identical except one line, both at the **current HEAD**, both on the **current** model
(reparameterisation on), three seeds as production runs use:

```yaml
# configs/p8_wx_off.yaml   — configs/submit_v10_reparam.yaml verbatim except:
name: p8_wx_off
use_weather: false
folds: [forward, jul_fwd]
# configs/p8_wx_on.yaml    — identical, use_weather: true
```

Then `make compare A=p8_wx_off-<hash> B=p8_wx_on-<hash>`.

**Why these folds, and the hard limitation that goes with them.** `[S]` `src/prc/cv.py:26-51` builds
`jan` (validate Jan 2025, train on Feb–Dec — **leaky**, trains on the future), `jul` (leaky), `jul_fwd`
(validate Jul 2025, train Jan–Jun — honest) and `forward` (validate Nov+Dec 2025, train Jan–Oct — honest).
**There is no `jan_fwd` fold and there cannot be one**: the code comment at `cv.py:39-42` says *"jan has
nothing before it in a single year"*. So:

- The scored set is **44.3 % January**, and we have **no honest fold that validates on a January**.
- The best honest proxy for the winter regime is `forward` (Nov + Dec 2025), which carries
  **1,184 + 1,441 = 2,625 gate rows** `[S]` — and that is precisely where the existing −8.00 s gate-row
  figure came from (2,568 matched gate rows, `seasonal-deicing-shift.md:237`). So that number *is* from an
  honest fold. Good.
- The best honest proxy for the July half is `jul_fwd`, where `deicing_gate` is **0 on every row in both
  years** `[S]`. **The de-icing half of the weather block is untestable on the July half of the scored set,
  because there is nothing to test.**

Report **the `prc.compare` triple** (K = 0 / 200 / 2000 with CIs, plus the month bootstrap — `compare.py:43-44`,
`compare.py:112`) and then, cut from the paired OOF frames, this breakdown. The adoption rule is unchanged:
`docs/features_log.md:5-6` — adopt only if the **K=200 CI excludes zero**.

**Breakdown to report (this is the deliverable, not the headline number):**

| cut | why |
|---|---|
| per **fold** (`forward`, `jul_fwd`) separately, never pooled | a pooled number mixes a regime that exists with one that does not |
| per **month** inside `forward` (Nov, Dec) | the two differ: −1.14 / −0.61 in the old pair (`seasonal-deicing-shift.md:243`) |
| per **`deicing_gate` ∈ {0,1}**, and within gate=1 **per airport** | this is where the old −8.00 s lived, and it was not uniform: EDDF −22.9, EGLL −33.2, LFPG −19.4, EDDM −7.8, LSZH −1.6, **EHAM +6.7**, **LTFM +19.0** (`seasonal-deicing-shift.md:240-242`) |
| per **matched / unmatched** | the old pair's unmatched effect (−37.1 s on 3,005 rows) is the part most likely to have been eaten by the LIRF reparameterisation |
| **re-weighted to the scored composition** | see below |

**The re-weighting, stated explicitly, because it is the single most decision-relevant line.** `[S]` The
scored set is 44.3 % January / 55.7 % July (152,719 / 192,122 rows, `16-external-data.md:16`). A de-icing gain
of Δ measured on a winter fold applies to **at most 44.3 %** of the scored rows, and within January only to
the **4.41 %** of rows in a gate hour. Reporting a pooled CV delta as if it were an LB delta will overstate
the winter component by roughly **2.3×**. The plan's own tasking says this; the measurement must print the
re-weighted number next to the pooled one so nobody has to remember.

**Cost:** `[S]` two runs × ~5 min at `--skip-predict`, per `docs/features_log.md:8`. Call it 30 minutes
including the compare. **Do this first regardless of the rest**, because everything downstream is measured
against it.

### 2.3 The one thing P8-A cannot settle

`[S]` `seasonal-deicing-shift.md:269-276`: the Jan-2026 gate-hour excess (6,729 rows vs 2,505; ×2.70) and
whether the model extrapolates correctly at **LTFM** (1,843 Jan-2026 gate rows, and training gate rows exist
only in Feb/Dec 2025 — Jan 2025 had zero) and **EDDF** (1,223 Jan-2026 gate rows against 617 in the whole
training year) **cannot be tested without ranking labels**. The only in-sample warning sign is LTFM's +208 s
forward-fold over-prediction on 81 rows. `[X]` That is a small-n signal and I would not act on it, but it is
the correct place to look if a weather change ships and the LB moves the wrong way.

---

## 3. P8-B — repair the dead columns from `wxcodes` (zero download, do it before Open-Meteo)

This step was not in the tasking. I am putting it ahead of Open-Meteo because I measured today that it
recovers most of the same information from bytes already on disk, and because leaving four constant columns
in a production matrix is a defect in its own right.

### 3.1 The change

In `hourly_weather_table` (`weather.py:126-162`), replace the dead precipitation terms with codes IEM
actually serves:

```
WET_CODES    = ("RA", "DZ")            # rain, drizzle — includes -RA/+RA/SHRA/TSRA/RADZ by substring
SHOWER_CODES = ("SH", "TS")            # showers, thunderstorms
OBSCURE      = ("FG", "BR", "HZ", "FU")  # fog / mist / haze — pairs with the existing vis_km
```

Proposed derived columns (per station-hour, `Int8`, so they survive `numeric_feature_columns`):

| new column | definition | replaces |
|---|---|---|
| `wet_wx` | any of `WET_CODES` in `wx` | `precip_mm` (dead) |
| `shower_wx` | any of `SHOWER_CODES` | — |
| `obscuration_wx` | any of `OBSCURE` | — |
| `deicing_gate` | `(temp_c ≤ 3) ∧ (frozen_wx ∨ wet_wx)` | the `precip_mm > 0` term, which never fires |
| `frozen_wx_i8` | `frozen_wx.cast(Int8)` | recovers a column currently lost to dtype (`findings.md:647-649`) |

**Delete** `precip_mm`, `temp_x_precip` and their `_prev1h` copies, or leave them and exclude them by name in
`exclude_features`. `[X]` I would delete them: a constant column that looks like a measurement is a trap for
the next reader, and `docs/features_log.md:44-52` shows each spurious column is worth ±0.5 s of model jitter.

### 3.2 How to measure it, and the trap to avoid

`[S]` `docs/features_log.md:40-52` is the governing precedent and it is the most important methodological
fact in this repo: adding **`noise_control`, a hashed column built to carry zero signal, cost +0.53 s** —
larger than any of the three real features being tested. Under `feature_fraction: 0.8` the column count itself
moves the model. Therefore:

- **Screen at `feature_fraction: 1.0`** (`docs/features_log.md:76`), and
- **hold the column count fixed**: this change *replaces* 4 dead columns with 5–8 live ones, so the control
  arm must be padded with the same number of noise columns, or the comparison measures arity, not weather.
- `[S]` The instrument's resolution at K=200 on the forward fold (328,009 rows) is about **0.25 s**
  (`docs/features_log.md:208`); at 3 folds (672,428 rows) the CI narrows by roughly √2. If the effect is under
  0.25 s we will not see it, and `docs/features_log.md` records **ten consecutive** features that died exactly
  there. `[X]` Expect the same outcome unless the July rain rows move.

### 3.3 Why this one might be different from the ten that failed

`[X]` The ten failures in `features_log.md` were all **re-derivations of information already in the file**
(queue length, bank position, runway activity, operator offsets). The model was measured to be past the
ceiling of that space: airport × stand × runway × hour × type gives a within-group RMSE of **268.1** while the
model already scores **~202** (`docs/features_log.md:90-96`). Rain is not in that space — it is the one axis
the competition files do not contain, and the one external axis that has ever moved our leaderboard
(`docs/README.md:109`). `[X]` That is an argument for *testing* it, not a prediction that it will pass.

---

## 4. P8-C — Open-Meteo `snowfall` and `precipitation`

### 4.1 The exact call, probed live today

`[S]` A **single** request per airport covers the whole training-plus-scoring window. Verified for EDDF:

```
GET https://archive-api.open-meteo.com/v1/archive
    ?latitude=50.0333&longitude=8.5706
    &start_date=2025-01-01&end_date=2026-07-31
    &hourly=precipitation,snowfall,rain
    &models=ecmwf_ifs
    &timezone=UTC
```

Response, measured: **13,848 hourly rows**, `2025-01-01T00:00` → `2026-07-31T23:00`, **zero nulls** in all
three series. `precipitation > 0` on **1,962** hours (14.2 %), `snowfall > 0` on **155**, max precipitation
7.5 mm/h, max snowfall 1.75 cm/h. Units from `hourly_units`: precipitation **mm**, snowfall **cm**,
rain **mm**, snow_depth **m**.

**13,848 is exactly the row count of our own `data/ref/metar/EDDF_*.csv`** `[S]` — the two series align
one-to-one on the hour, which makes the join trivial.

`[S]` **Independent corroboration of the winter-2026 signal.** EDDF `snowfall > 0` hours by month:
`2025-01: 19`, `2025-02: 9`, `2025-11: 10`, `2026-01: 74`, `2026-02: 33`, `2026-03: 10`, and **zero in every
July**. Jan 2026 / Jan 2025 = **×3.9**, against the ×3.65 our own METAR gate-hour ratio gives at EDDF
(`seasonal-deicing-shift.md:79`). Two unrelated instruments agree that January 2026 was a markedly snowier
January at Frankfurt. `[X]` This is the strongest single piece of evidence in the theme that the winter
signal is real rather than a parsing artefact.

### 4.2 Model choice — three findings from today's probes that change the recommendation

`[S]` **Use `models=ecmwf_ifs` (9 km), not `era5_land`, not the default.**
- `models=era5_land` (0.1°) returned **24 nulls of 24** for 2026-01-08 at EDDF. **ERA5-Land does not cover
  the scored months.** Do not use it.
- `models=ecmwf_ifs` returned a full series and resolved to grid point `50.0176, 8.6387` — about **6 km**
  from EDDF's threshold.
- The default (`best_match`) resolved to the same point today, but pinning the model makes the cache
  reproducible.

`[S]` **Operational risk, hit today:** after roughly five probes this sandbox's egress IP received
`{"error":true,"reason":"Daily API request limit exceeded. Please try again tomorrow."}` on `models=era5`
and `models=ecmwf_ifs025`. The free tier's 10,000 calls/day is **per IP and shared**. `[X]` With the
single-call-per-airport pattern the whole fetch is **10 calls**, so this is manageable — but it must be
written with retry/backoff and a persistent on-disk cache, exactly like `fetch_metar` already is
(`weather.py:54-72`), and a failed fetch must **not** silently produce a partially-null column. That is the
`ec_*` failure mode from §5.

`[S]` **Grid-scale caveat, honestly stated.** Slot 24 records a competitor measuring
*"ERA5 via Open-Meteo 485.5 vs 448.8, both months worse, 'grid too coarse'"*
(`24-repo-tail-sweep.md:800-801`). Slot 04 argues the failure reason does not transfer:
*"an airport METAR is exactly co-located with a taxiing aircraft"* (`04-prc-2025-writeups.md:769-771`).
`[X]` **Both are right, and the resolution is that the argument favours IEM over Open-Meteo, not
Open-Meteo over nothing.** The 6 km grid point is a model output, not an observation; where IEM and
Open-Meteo disagree, IEM is the better instrument for a taxiing aircraft. Open-Meteo's role is strictly to
supply the **one quantity IEM refuses to serve** — frozen and liquid precipitation amount — and nothing else.
Do not import its temperature, wind or visibility; we already have measured versions of all three.

### 4.3 Join key, time alignment, caching, composition

- **Coordinates:** `src/prc/airports.py:28-49` `airport_coords()` already caches OurAirports ICAO → lat/lon to
  `data/external/airports.parquet` `[S]`. Take the ten from `TARGET_AIRPORTS` (`src/prc/config.py:17`). No new
  reference data.
- **Join key:** `(station, hour_utc)` into `hourly_weather_table`'s output **before** `join_weather` runs —
  i.e. Open-Meteo becomes extra columns on the station-hour table, and the existing
  `(ADEP_mvt, truncate(MVT_TIME_UTC_mvt, 1h))` departure join at `weather.py:201` is untouched. **Do not add a
  second departure-level join.** `weather.py:165-169` is a monument to what happens when two joins write
  overlapping column names.
- **Time alignment:** Open-Meteo `time` is ISO-8601 naive with `timezone=UTC`, labelled at the **start** of
  the hour; IEM `valid` timestamps are METAR issue times (`:50` past the hour in our cache `[S]`, e.g.
  `EDDF,2025-01-01 00:50`) and `hourly_weather_table` truncates them to the hour (`weather.py:135`). So a
  METAR issued at `00:50` and an Open-Meteo hour labelled `00:00` both land in hour `00`. `[X]` That is a
  ≤ 50-minute offset between the two sources inside one bucket. It is acceptable for an accumulation field
  (an hour of snowfall is an hour of snowfall) and would **not** be acceptable for an instantaneous field —
  another reason to import only the accumulations.
- **Caching:** mirror `fetch_metar` exactly — one file per station,
  `data/ref/openmeteo/{ICAO}_{start}_{end}.json`, skip if present and non-trivial, sleep between requests,
  `refresh=` flag. **Add both `weather.py` and the new cache directory to `cache_fingerprint`**: `[S]`
  `src/prc/pipeline.py:55-71` currently hashes only `src/prc/features.py`, so `weather.py` edits and METAR
  cache changes **do not invalidate the feature cache** (`findings.md:667-670`). Editing `weather.py` for
  P8-B or P8-C and re-running will silently reuse the old frames. **This is the highest-probability way for
  this theme to waste a day.** Delete `data/work/feat_*.parquet` by hand, or fix the fingerprint first.
- **Composition with the existing IEM columns:** two new station-hour columns, `om_precip_mm` and
  `om_snowfall_cm`, plus `_prev1h` copies via the existing mechanism = **+4 model-visible columns**. Redefine
  `deicing_gate` as `(temp_c ≤ 3) ∧ (frozen_wx ∨ om_snowfall_cm > 0)` only if P8-B's `wet_wx` version loses
  to it head-to-head. `[X]` Keep exactly one gate definition in the tree; two correlated gates is a way to
  spend the arity budget for nothing.

### 4.4 Licence

`[S]` Data is **CC BY 4.0** (`16-external-data.md` §3.2, §6.4) — satisfies the eligibility clause
*"All additional datasets used are openly available under an open source license"*
(`16-external-data.md:27`). The free-tier non-commercial cap is on the *service*, not the data. Attribution
line goes in `docs/README`, as `weather.py:7-8` already instructs. **Clean.** No licence work needed —
unlike the EUROCONTROL/OPDI family (`16-external-data.md` §6.1).

---

## 5. P8-D — the coverage-parity gate, as a standalone deliverable

### 5.1 What kind-mango's gate does

`[S]` `/tmp/prc-kind-mango/src/check_coverage.py`, 146 lines, read in full. Per column × airport × scored
period, it computes the **non-null** and **non-zero** share and fails the build when:

```python
if nn_drop > DROP_POINTS:                      # 20.0, vs FULL 2025          (check_coverage.py:70-71)
    fails.append(...)
if nz_drop > DROP_POINTS and nz_drop_m > DROP_POINTS:   # vs full-2025 AND vs the same 2025 month  (:72-73)
    fails.append(...)
```

plus an "absent from the 2026 dump" check (`:56-57`), a JSON report, and `raise SystemExit(1)` on any strict
failure (`:140-141`). The month match on the non-zero test is the good detail: it absorbs seasonality.

`[S]` The defect it was built for: `ec_*` went **100.0 % → 0.0 %** non-null at LIRF in July 2026, their lag-1
copies **99.8 % → 3.2 %** (`22-repo-kind-mango.md:378-380`). Mechanism, from their own code reading:
*"LightGBM's `NumericalDecision` converts NaN to 0.0 when the split's feature had no missing value in
training"* (`22-repo-kind-mango.md:353-358`) — every July row routed to a lowest-traffic branch whose linear
leaf then extrapolated. Dropping the feeds was worth **−13.14 s**; **median-imputing them instead was +58 s**
(v25, 375.33). `[S]` **It is a deployment failure, not an information failure.**

### 5.2 I ran their rule on our features today. Here is what it finds.

Read-only, on `data/work/feat_train.parquet` (2,085,047 rows, all of 2025) and `data/work/feat_rank.parquet`
(344,841 rows, Jan + Jul 2026), over the **89 numeric + 10 categorical** columns the model actually sees
(`train.py:229-235`, candidates excluded).

**Strict failures: exactly four.** `[S]`

```
NONZERO dewp_spread_c        @LEMD 2026_jan: -24 pts vs all-2025, -21 pts month-matched (Jan-2025 93.7% -> 72.5%)
NONZERO dewp_spread_c_prev1h @LEMD 2026_jan: -26 pts vs all-2025, -23 pts month-matched (93.4% -> 70.6%)
NONZERO dewp_spread_c        @LSZH 2026_jan: -23 pts vs all-2025, -26 pts month-matched (95.9% -> 70.4%)
NONZERO dewp_spread_c_prev1h @LSZH 2026_jan: -22 pts vs all-2025, -26 pts month-matched (95.5% -> 69.5%)
```

**These are the same four cells kind-mango got.** `[S]` Theirs, verbatim from
`22-repo-kind-mango.md:536-538`: *"dewpt_spread at LEMD in 2026_jan: non-zero share dropped 24 points;
dewpt_spread at LSZH in 2026_jan: non-zero share dropped 22 points"*. Same columns, same two airports, same
month, magnitudes within 1–2 points. **Two independent pipelines, independently built on the same IEM feed,
produce the same four flags.** `[X]` That settles the interpretation: it is a property of the IEM dewpoint
field at LEMD and LSZH in January 2026 — temperature and dewpoint reported equal (saturated air / fog) so the
spread is exactly 0 — and **not** a bug in either repo. Their own verdict was *"data drift, not pipeline
defects"* (`22-repo-kind-mango.md:540-541`); ours should be the same. **No action beyond recording it.**

**We have no `ec_*`-class exposure.** `[S]` `16-external-data.md` §7: *"Neither OPDI nor any EUROCONTROL PRU
file is referenced anywhere in `src/` or `configs/`."* Our only external columns are IEM weather (18) and
`dest_distance_km` (OurAirports, a rejected candidate, `features_log.md:154-172`). The 13.14 s trap is not
one we are standing in — which is worth saying plainly, because it also means **importing the gate buys us
almost nothing on today's features**. Its value is prospective: it is the entry condition for P8-C and for
every future external join.

### 5.3 Where their rule is wrong for us, and the two changes I would make

**Change 1 — a dead column must fail.** `[S]` `precip_mm` has a non-zero share of **0.0000 %** in 2025 and
**0.0000 %** in 2026. Drop = **0 points**. `nn_drop` = 0. **kind-mango's gate passes it.** Their rule is
built to catch feeds that *decay*; it is blind to a feed that was never alive. Ours must add:

```
DEAD:  non-zero share < 0.1 % on the TRAINING side  ->  fail as dead weight
```

`[S]` What that would flag today, on the full 2025 training set:

| column | train non-zero | verdict |
|---|---|---|
| `precip_mm`, `precip_mm_prev1h` | 0.0000 % | **DEAD — fail** |
| `temp_x_precip`, `temp_x_precip_prev1h` | 0.0000 % | **DEAD — fail** |
| `runway_null` | 0.0000 % (1 distinct value) | **DEAD — fail** (not weather; flagged in passing) |
| `stand_null` | 0.0013 % | near-dead, warn |
| `diverted` | 0.0993 % | borderline, warn |
| `actype_mismatch` | 0.3363 % | informational |

That is **five genuinely constant columns in the production matrix**, four of them weather. `[X]` A gate that
finds five real defects on its first run has paid for its ninety lines.

**Change 2 — report rises, do not fail on them.** kind-mango logged but never caught `eobt1_sched` at LFPG
going **35.7 → 61.6 %** non-zero (`22-repo-kind-mango.md:566`), a one-sided test's blind spot. `[S]` **Our
largest rises, measured today (2026 month vs month-matched 2025, per airport):**

```
+22.2 pts  lobt_minus_eobt1        LFPG 2026_jul   (and its _slog copy)
+18.0 pts  dewp_spread_c_prev1h    EHAM 2026_jan
+17.1 pts  dewp_spread_c           EHAM 2026_jan
+11.7 pts  wind_ms                 LEMD 2026_jan
 +9.2 pts  deicing_gate_prev1h     LSZH 2026_jan
 +8.5 pts  deicing_gate_prev1h     LTFM 2026_jan
```

**The top line is an independent hit on kind-mango's LFPG finding.** `[S]` They found `eobt1_sched` non-zero
at LFPG going 35.7 → 61.6 % in July 2026; I find `lobt_minus_eobt1` at LFPG going **+22.2 points** in the same
airport, same month, same family of NM flight-plan time fields. `[X]` **LFPG's plan-time reporting behaviour
changed between July 2025 and July 2026.** LFPG is 11.6 % of scoring rows (their figure) and our
`mvt_minus_eobt1` family is a high-rank feature group. **This is a lead for another theme, not for P8** — I
am recording it here because this analysis is where it surfaced, and it should not be lost.

`[X]` Make rises **report-only**. A rise is not the NaN→0 failure mode; treating it as a build failure would
have us fighting ordinary weather variation (`deicing_gate` at LSZH rising is *the real world*, and it is the
whole reason weather might help in January 2026).

### 5.4 Where it runs and what it fails

- **File:** `src/prc/coverage.py`, `python -m prc.coverage --train data/work/feat_train.parquet --rank
  data/work/feat_rank.parquet --out outputs/coverage_monitor.json`. `[X]` ~120 lines; kind-mango's is 146 and
  half of that is their repo-specific cache plumbing (`check_coverage.py:28-36, 77-100`), which we do not need
  — our two feature frames already carry `ADEP_mvt` and `month` and have identical schemas (verified today:
  132 columns each, zero symmetric difference).
- **Column set:** exactly what the model sees — `visible_numeric_columns(feats, cfg)` plus `CATEGORICALS`
  (`train.py:229-235`, `train.py:374-377`), not the 132 frame columns. `[S]` `train.py:526-528` already
  distinguishes these two counts, with the comment *"What the model was given, not how wide the frame is"* —
  the gate must use the same definition or it will report on columns nobody trains on.
- **Where in the pipeline:** immediately after `build_features` in `pipeline.py:98-99`, before CV. A failure
  raises. `[X]` Also wire it as a **hard precondition of the submission gate** in `prc.submit`, because that
  is the step that cannot be undone.
- **Fail on:** (a) a column absent from the ranking frame; (b) non-null share down > 20 points vs full 2025,
  per airport per scored month; (c) non-zero share down > 20 points vs full 2025 **and** vs the month-matched
  2025 month; (d) **new:** non-zero share < 0.1 % on the training side (dead column).
- **Report but do not fail on:** rises > 20 points; non-zero drops that clear one of the two references but
  not both; per-airport counts so a failure can be read as "how many rows does this affect".
- **Escape hatch:** an explicit `coverage_allow:` list in the config, each entry requiring a written reason.
  `[S]` kind-mango marked whole column families "informational" (`check_coverage.py:7-9`) and then **shipped
  the LIRF head still reading `ec_*` with the known 2026 NaN pattern** — v49's attempt to remove them there
  went **+0.42 s live** (`22-repo-kind-mango.md:388-392`). `[X]` So a waiver mechanism is legitimate; a silent
  one is not.

### 5.5 Should it run against our existing features? Yes — and here is what it will say

Yes, and I have already run it, so the answer is not a guess:

1. **Four strict failures, all `dewp_spread_c` at LEMD/LSZH in January 2026, all confirmed data drift** by
   independent reproduction against kind-mango's numbers. No action.
2. **Five dead columns** — `precip_mm`, `temp_x_precip`, their `_prev1h` copies, and `runway_null` — invisible
   to the gate as kind-mango wrote it, caught by the extension in §5.3. This is P8-B's mandate, arrived at from
   a second direction.
3. **No `ec_*`-class catastrophe**, because we ship no such feed.
4. **One cross-theme lead**: LFPG plan-time drift in July 2026.

`[X]` Net: the gate's value on *today's* features is the dead-column finding and the confidence that we are
not sitting on a 13-second trap. Its real value is as the **entry condition for P8-C and everything after it**
— nothing external ships until it passes. That is worth building even if P8-A and P8-C both come back null.

---

## 6. Honest expected value — is this theme worth the effort?

The commander asked for a verdict, not advocacy. Here it is, with the evidence on both sides.

### 6.1 The case against, which is strong

`[S]` **PRC 2025's entire prize podium used no external data** (resourceful-quiver 200.83, jubilant-vase
211.99, outspoken-tornado 219.85), and the one team that reported testing weather reported *"limited
influence"* (`16-external-data.md` §2.2).

`[S]` **PRC 2026's strongest public repos credit zero external datasets with a gain** — Phoenix-Ops 273.56,
javidmardanov 268.56, sergiuv11 286.66 (`16-external-data.md` §2.3).

`[S]` **The one 2026 team that measured its weather family put it at 1.8 % of total gain over 14 features**,
and killed the mechanism story outright: the > 6 h rows that decide the metric sit at **20.7 °C, zero
freezing, zero snow, zero de-icing proxy, and are *more* common in July than January**
(`09-repo-sergiuv11.md:501-505`). **Weather does not explain the tail, and the tail is where our score is.**

`[S]` **Our own metric is a tail metric.** 10 rows of 672,428 carry 24 % of total squared error
(`compare.py:5-8`, `docs/README.md` §4). `[S]` Arithmetic in `docs/README.md:76` says reaching the leader
needs the unmatched RMSE at ≈ 1,033; **improving the matched 98.8 % of rows by 10 % is worth 11 seconds
total.** Weather acts on matched rows in ordinary conditions. It is structurally the wrong lever for the gap
we have.

`[S]` **Our own feature-engineering record is 0 for 10** (`docs/features_log.md`), every candidate landing
below the 0.25 s instrument resolution, including a rejected external one (`dest_distance_km`).

`[S]` **The de-icing mechanism is provably absent from 55.7 % of the scored set.** Zero gate rows in July 2026
and July 2025, at every airport (`seasonal-deicing-shift.md:199`, re-confirmed by me today). A winter-only
gain is worth 44.3 % of what it looks like pooled, and within January only 4.41 % of rows are in a gate hour.

`[S]` **kind-mango gated weather-at-pushback at −293 MSE against a 300 MSE threshold and rejected it**
(`22-repo-kind-mango.md:117`) — a team already shipping METAR could not clear its own bar with a better join.

### 6.2 The case for, which is narrower than it looks but not empty

`[S]` **Weather is the only change we have ever made that moved our leaderboard in the direction CV
predicted** — v2 312.02 → v3 306.29, −5.73 s (`docs/README.md:109`). It is the single calibration case
underwriting the whole decision rule. `[X]` It is also, per §2.1, a case whose CV and LB halves come from
different joins, which is a reason to re-measure it and not a reason to trust it.

`[S]` **The January-2026 exposure shift is real, large, and independently corroborated.** Ranking gate rows
Jan 2026 **6,729** vs Jan 2025 **2,505** (×2.70); at EDDF one ranking month has more gate rows (1,223) than
the entire 2025 training year (617), at LSZH 1,379 vs 1,119 (`seasonal-deicing-shift.md:197-200`). `[S]` I
corroborated this today from an unrelated source: Open-Meteo `snowfall > 0` hours at EDDF, Jan 2025 **19** vs
Jan 2026 **74** (×3.9). `[S]` A competitor's research notes independently report Munich de-icing 8,628
aircraft in 2026 vs 6,937 in 2025 (`24-repo-tail-sweep.md:804-806`).

`[S]` **Slot 04's argument that the 2025 negative does not transfer is sound as far as it goes**: ERA5's
stated failure mode is grid-scale mismatch with a fast-moving aircraft, and *"an airport METAR is exactly
co-located with a taxiing aircraft"* (`04-prc-2025-writeups.md:769-771`). `[X]` But note what that argument
actually licenses: it defends **IEM METAR**, which we already have. It is not an argument for Open-Meteo,
whose 6 km grid point is precisely the thing slot 24's competitor measured at 485.5.

`[S]` **The defect I measured today is real and is nobody's opinion.** Four of eighteen weather columns are
constants; the rain content of `wxcodes` is parsed and discarded; 31,534 scored rows (9.1 %) sit in a
rain-coded hour, 5,535 of them in July where every other weather column is flat.

### 6.3 My verdict

**Do P8-A and P8-D. Do P8-B. Treat P8-C as conditional. Expect no leaderboard gain and be pleasantly
surprised if one appears.**

| item | effort | my honest probability it clears the K=200 adoption bar | if it clears, likely size |
|---|---|---|---|
| **P8-A** re-measure | 30 min | n/a — it is an audit, not a candidate | its job is to tell us whether `use_weather: true` still earns its 18 columns against the 223.51 baseline. `[X]` I put maybe 1-in-3 on the honest answer being "no longer significant". |
| **P8-D** coverage gate | 2–3 h | n/a — infrastructure | Already paid for: it found 5 dead columns and reproduced a competitor's four drift cells today, read-only. |
| **P8-B** `wxcodes` repair | 2–3 h | `[X]` **~25 %** | `[X]` 0.3–1.0 s if it lands. It removes 4 dead columns *and* adds a July-live channel; the removal alone might be worth something via the arity effect. |
| **P8-C** Open-Meteo | 3–4 h | `[X]` **~10 %** | `[X]` ≤ 0.5 s. It is a 6 km model grid supplying a quantity P8-B mostly recovers from a co-located observation. Measure it **against P8-B**, not against nothing, or it will take credit for B's work. |

**Total: about one working day for P8-A + P8-D + P8-B, a second day if P8-C is run.**

**What I would say to the commander in one sentence:** this theme is worth a day because it closes a real
defect (four constant columns and a discarded rain signal, both measured today) and builds a reusable gate
that is a precondition for every future external dataset — **not** because weather is where sixty seconds
are hiding. The evidence that weather is *not* where the seconds are is strong, consistent across two
competition editions and four independent teams, and our own data shows the mechanism is flat on the larger
half of the scored set. **If the commander has exactly one day and another theme targets the unmatched tail,
give the day to that theme and run only P8-D here** — it is three hours, it is read-only until the last
commit, and it is the thing that stops a future external join from costing thirteen seconds.

### 6.4 The stop rule

`[X]` Following `docs/features_log.md:10` (*"5 連敗で外部調査して方針転換"*): if P8-A shows the existing
weather block is no longer significant at K=200 against the current baseline, **stop P8-B and P8-C
immediately** and open the question of whether to set `use_weather: false`. Do not spend effort improving a
feature block that has stopped paying. `[X]` Note that removing 18 columns is itself a change with an arity
effect and would need the same `prc.compare` treatment — do not just flip the flag.

---

## 7. Execution order

1. **P8-D first**, because it is read-only until the last step and its output changes what P8-B does.
   Write `src/prc/coverage.py`; reproduce the four `dewp_spread_c` failures and the five dead columns already
   measured here as its acceptance test.
2. **P8-A**: `configs/p8_wx_off.yaml` / `p8_wx_on.yaml`, same sha, `folds: [forward, jul_fwd]`,
   `--skip-predict`, then `make compare`. Report the per-fold / per-month / per-gate / per-airport /
   matched-unmatched breakdown **and the scored-composition re-weighting**. Append to
   `docs/features_log.md`.
3. **Before touching `weather.py`: delete `data/work/feat_*.parquet`.** `[S]` `cache_fingerprint`
   (`pipeline.py:55-71`) does not hash `weather.py` or the METAR cache
   (`findings.md:667-670`), so the edit will otherwise be invisible and the run will report the old numbers.
   `[X]` Better: add `sha256(weather.py)` to the fingerprint as part of this work.
4. **P8-B**: `wxcodes`-derived `wet_wx` / `shower_wx` / `obscuration_wx`, `frozen_wx` cast to `Int8`, dead
   columns deleted, gate redefined. Screen at `feature_fraction: 1.0` against a **column-count-matched**
   noise control (`features_log.md:76`).
5. **P8-C** only if P8-B lands or is a near-miss: Open-Meteo `ecmwf_ifs`, one call per airport, cached,
   `om_precip_mm` + `om_snowfall_cm` joined at the station-hour, measured **against the P8-B arm**.
6. Correct `docs/README.md` §6 to record that the weather calibration row pairs a fixed-join CV delta with a
   broken-join LB delta.

---

## Appendix — facts versus inference

**Measured by me in this sandbox today (2026-09-21), all read-only, all reproducible:**
- `p01i` is the literal `0.00` on 100 % of 138,446 raw IEM rows at all ten stations; `>0` on zero rows.
- `precip_mm`, `precip_mm_prev1h`, `temp_x_precip`, `temp_x_precip_prev1h`, `runway_null` have a non-zero
  share of 0.0000 % on both `feat_train` (2,085,047 rows) and `feat_rank` (344,841 rows).
- `wxcodes` code census over the cache (top 25 codes as listed); present-weather populated on 7.3–24.0 % of
  station-hours per airport.
- Rain-coded departure rows per month, per airport, both frames (§1.4 table).
- kind-mango's coverage rule re-implemented and run on our two frames over the 89 numeric + 10 categorical
  model-visible columns: four strict failures, listed verbatim, matching their four.
- Largest non-zero-share rises per airport per scored month (§5.3 list).
- Open-Meteo: `ecmwf_ifs` full-range single call for EDDF returns 13,848 non-null hours 2025-01-01 →
  2026-07-31; `precipitation > 0` 1,962 h, `snowfall > 0` 155 h; snowfall-hour census by month; `era5_land`
  returns all nulls for 2026-01; the daily request limit was hit from this IP.
- `feat_train` and `feat_rank` both have 132 columns with zero symmetric difference; `hour_utc` is `Int8`.

**Read from files in this repository (line-cited above):** the `WX_KEY` bug and its fix commit; the
`det_a`/`weather_check` record pair and every RMSE in it; the fold construction and the absence of a
`jan_fwd` fold; `numeric_feature_columns`' dtype filter; `cache_fingerprint`'s contents; the `noise_control`
±0.5 s result and the 0.25 s resolution; the ten-loss feature log; the gate-row and gate-hour tables; the
`compare.py` decision rule.

**Read from the competitor-research slots:** kind-mango's −13.14 s, +58 s, the NaN→0 mechanism, their four
strict failures, their L7 weather rejection at −293/300 MSE; sergiuv11's 1.8 % and 20.7 °C; slot 24's 485.5;
slot 04's grid-scale argument; slot 16's coverage verdicts and licences.

**My inference, marked `[X]` throughout and summarised here:**
- That P8-B is likelier to pay than P8-C, because a co-located observation beats a 6 km model grid for a
  quantity both can supply. Not measured — no model was fitted in this planning pass.
- The probability estimates in §6.3. They are judgements calibrated on `features_log.md`'s 0-for-10 record
  and on the competitor evidence, not measurements.
- That the four `dewp_spread_c` flags are saturated-air artefacts rather than feed defects. The independent
  reproduction across two repos makes a pipeline bug very unlikely; the physical explanation is unverified.
- That the LFPG `lobt_minus_eobt1` rise is the same phenomenon as kind-mango's LFPG `eobt1_sched` rise. Same
  airport, same month, same field family, same direction — but they are different columns and I did not join
  the two measurements.
- That removing four constant columns could itself be worth something through the arity effect. The
  `noise_control` experiment measured the cost of *adding* one column; I am assuming symmetry.
