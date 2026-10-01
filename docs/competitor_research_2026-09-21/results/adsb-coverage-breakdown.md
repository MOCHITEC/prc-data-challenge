# ADS-B pushback override: why scored departures were not overridden (2026-09-29)

Sibling note to `adsb-full-ingest-candidate.md`, `adsb-oos-validation.md`, `adsb-gain-estimate.md` and
`adsb-stage1-feasibility.md`. Read-only: nothing was fitted, submitted or written to `outputs/`, `src/`, `configs/`
or `data/`. The detection was not re-run.

- Script: `scripts/adsb_cov_breakdown.py`. Log: `$SCRATCH/adsb/adsb_cov.log`. Per-row reasons:
  `$SCRATCH/adsb/adsb_cov_reasons.parquet`. Part B table: `$SCRATCH/adsb/adsb_cov_partb.csv`.
- The reasons come from the stage-1 columns in `<tag>_joined.parquet` and the eligibility and guard logic of
  `scripts/adsb_override_candidate.py`. That logic is copied, not changed. Base = v16
  (`$SCRATCH/v16_from_bucket.parquet`), the same base as the v18 run.
- **Reproduction check.**
  - 63,737 rows are overridden at the five airports. That is the same id set as `adsb_override_rows.parquet`.
  - There are 207 guard trips.
  - The pushback counts per airport are EDDM 15,913, EGLL 16,826, EHAM 16,189, LEBL 9,403 and LSZH 5,613. They
    equal the v18 log.
- All 62 scored days were analysed. No day is missing.

## A. Reasons, scored 2026 departures

**Waterfall.** Each row gets the first reason that applies, in the order of the table. LEMD is scored *as if it
were included*. Its bias is set by the same S1 rule (median `t_push − BLOCK` over the covered rows of the 4
labelled days = +126.2 s, n 110), and the guard is applied the same way.

**What the reasons mean.**

- **A5 (no ADS-B fix carries the callsign).** No ADS-B fix at ADEP carries `CALLSIGN_flt` or `FLIGHT_mvt` that day
  (stage-1 `join` is null).
  - The data has no registration or ICAO24 key on the flight side. So "no ADS-B for the aircraft" and "callsign
    mismatch" cannot be told apart directly.
  - A5a/A5b split on an **unclaimed ADS-B take-off**: a ground→air transition at ADEP that no joined departure of
    that day claims (its `t_to` is not within 60 s). A5a means such a take-off lies within ±60 s of `MVT_TIME`.
- **A6.** The callsign did not match, but `FLIGHT_mvt` did. The rule does not use that fallback.
- **A7 (`cs_notime`).** `CALLSIGN_flt` matched an aircraft at ADEP, but none of that aircraft's fixes gave a usable
  phase.
- **A8.** The callsign matched, but no ground fix comes before the take-off.
- **A9 (no stationary stand run).** There is a ground segment, but it holds no stationary stand run (≤ 25 m).
  - A9a: the aircraft was already moving when first seen (a single fix with gs > 1 kt).
  - A9b: no qualifying run.
  - A9c: the stand run ends less than 60 s before the phase ends.
- **A10 (stand run, but no gated pushback).**
  - A10a: no usable displacement step with a track in the 180 s after the stand.
  - A10b: the tail-first share is < 0.5, i.e. the first motion was forward.
  - A10c: the tail-first share is ≥ 0.5, but the 15-min dwell guard dropped the row.

Counts, with the share of that airport's scored departures in brackets:

| reason | EDDM | EGLL | EHAM | LEBL | LSZH | **five** | LEMD (as if included) |
|---|---:|---:|---:|---:|---:|---:|---:|
| scored departures | 25,966 | 39,840 | 38,182 | 30,080 | 23,150 | 157,218 | 36,954 |
| A1 missing day | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| A2 MVT 00:00–02:00 UTC | 2 (0.01 %) | 0 | 31 (0.08 %) | 122 (0.41 %) | 0 | 155 (0.10 %) | 338 (0.91 %) |
| A3 no `CALLSIGN_flt` / `AOBT_3` (lane B) | 238 (0.92 %) | 248 (0.62 %) | 1,199 (3.14 %) | 436 (1.45 %) | 659 (2.85 %) | 2,780 (1.77 %) | 233 (0.63 %) |
| A4 other (not in the joined file) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| A5a no fix carries the callsign; unclaimed take-off within 60 s | 9 (0.03 %) | 65 (0.16 %) | 74 (0.19 %) | 7 (0.02 %) | 7 (0.03 %) | 162 (0.10 %) | 102 (0.28 %) |
| A5b no fix carries the callsign; no unclaimed take-off | 645 (2.48 %) | 982 (2.46 %) | 645 (1.69 %) | 741 (2.46 %) | 663 (2.86 %) | 3,676 (2.34 %) | 1,031 (2.79 %) |
| A6 callsign join failed, `FLIGHT_mvt` matched | 0 | 0 | 2 (0.01 %) | 0 | 0 | 2 (0.00 %) | 0 |
| A7a callsign's aircraft has no fix at ADEP within ±3 h | 2 (0.01 %) | 0 | 2 (0.01 %) | 2 (0.01 %) | 4 (0.02 %) | 10 (0.01 %) | 6 (0.02 %) |
| A7b fixes within 3 h, but no ground fix and no air fix within ±30 min | 2 (0.01 %) | 0 | 3 (0.01 %) | 0 | 3 (0.01 %) | 8 (0.01 %) | 2 (0.01 %) |
| A8 points, but no ground segment | 62 (0.24 %) | 626 (1.57 %) | 49 (0.13 %) | 202 (0.67 %) | 108 (0.47 %) | 1,047 (0.67 %) | 2,587 (7.00 %) |
| A9a ground segment, no stand: moving when first seen | 2,690 (10.36 %) | 7,492 (18.81 %) | 4,100 (10.74 %) | 7,979 (26.53 %) | 5,485 (23.69 %) | 27,746 (17.65 %) | 9,669 (26.16 %) |
| A9b ground segment, no stand: no qualifying run | 257 (0.99 %) | 1,489 (3.74 %) | 1,166 (3.05 %) | 716 (2.38 %) | 176 (0.76 %) | 3,804 (2.42 %) | 3,056 (8.27 %) |
| A9c ground segment, stand run ends < 60 s before the phase ends | 42 (0.16 %) | 1,439 (3.61 %) | 47 (0.12 %) | 1,143 (3.80 %) | 160 (0.69 %) | 2,831 (1.80 %) | 1,684 (4.56 %) |
| A10a stand, no pushback: no usable step with a track | 673 (2.59 %) | 2,169 (5.44 %) | 1,433 (3.75 %) | 815 (2.71 %) | 608 (2.63 %) | 5,698 (3.62 %) | 833 (2.25 %) |
| A10b stand, no pushback: first motion forward | 5,257 (20.25 %) | 8,389 (21.06 %) | 12,968 (33.96 %) | 8,475 (28.17 %) | 9,611 (41.52 %) | 44,700 (28.43 %) | 11,744 (31.78 %) |
| A10c stand, pushback, dwell guard | 174 (0.67 %) | 115 (0.29 %) | 274 (0.72 %) | 39 (0.13 %) | 53 (0.23 %) | 655 (0.42 %) | 18 (0.05 %) |
| A11 guard trip | 43 (0.17 %) | 67 (0.17 %) | 65 (0.17 %) | 13 (0.04 %) | 19 (0.08 %) | 207 (0.13 %) | 19 (0.05 %) |
| **not overridden (A1–A11)** | **10,096 (38.88 %)** | **23,081 (57.93 %)** | **22,058 (57.77 %)** | **20,690 (68.78 %)** | **17,556 (75.84 %)** | **93,481 (59.46 %)** | 31,322 (84.76 %) |
| A12 overridden | 15,870 (61.12 %) | 16,759 (42.07 %) | 16,124 (42.23 %) | 9,390 (31.22 %) | 5,594 (24.16 %) | 63,737 (40.54 %) | 5,632 (15.24 %) (hypothetical) |

Sums by the groups the task asked for, five airports pooled:

| group | rows | share of scored |
|---|---:|---:|
| no ADS-B fix carries the callsign that day, or none near this departure's time (A5b + A7a) | 3,686 | 2.34 % |
| points exist but no ground segment near the stand (A7b + A8) | 1,055 | 0.67 % |
| ground segment, no stationary stand run (A9a–c) | 34,381 | 21.87 % |
| stand run, but no pushback detected (A10a + A10b) | 50,398 | 32.06 % |
| pushback detected, dwell guard (A10c) | 655 | 0.42 % |
| callsign join failure, measured (A6, plus A5a) | 164 | 0.10 % |
| no callsign / `AOBT_3` (lane B, A3) | 2,780 | 1.77 % |
| guard trip (A11) | 207 | 0.13 % |
| MVT 00:00–02:00 UTC (A2) | 155 | 0.10 % |
| missing day (A1) | 0 | 0 |
| other (A4) | 0 | 0 |

The A5a split is weak evidence of a callsign mismatch. It has a **placebo**: the same count of unclaimed take-offs
within 60 s of `MVT_TIME` ± 600 s.

| | EDDM | EGLL | EHAM | LEBL | LSZH | LEMD |
|---|---:|---:|---:|---:|---:|---:|
| A5 rows (join null, lane A, in scope) | 654 | 1,047 | 719 | 748 | 670 | 1,133 |
| unclaimed take-off within ±60 s of `MVT_TIME` | 9 | 65 | 74 | 7 | 7 | 102 |
| placebo: at `MVT_TIME` + 600 s | 13 | 15 | 73 | 11 | 14 | 49 |
| placebo: at `MVT_TIME` − 600 s | 19 | 19 | 57 | 8 | 10 | 58 |
| at `MVT_TIME`, and the ADS-B callsign has the same first 3 characters | 1 | 24 | 25 | 2 | 2 | 44 |

- At EDDM, EHAM, LEBL and LSZH, the count at `MVT_TIME` is not above the placebo.
- At EGLL (65 vs 15/19) and LEMD (102 vs 49/58), it is above the placebo.
- Examples of `CALLSIGN_flt` → ADS-B callsign at EGLL: BAW81→BA81, BAW742→BA742, BAW73RP→BW73RP, ALK504→ALK5004,
  BAW560F→BAW56OF, AIC130→AI130, APK7577→7577.

### LEMD: why it was excluded

`adsb-oos-validation.md` §4 point 3: "**LEMD** is positive pooled but negative on the November day, with n = 72
over three days. I would not include it without more labelled days."

The out-of-sample result behind it (§3): "LEMD is +8,400 [−6,700, +23,300]. It is positive but not significant,
and it is negative on 2025-11-12."

The pre-registration (`adsb-full-ingest-candidate.md` §0.1) then fixed the airport list: "**EDDM, EHAM, LEBL,
LSZH.** LEMD is out."

The table above shows that LEMD's hypothetical coverage would be 15.24 % of its scored departures. LEMD also has
the largest A8 share (7.00 %) and the largest A9b share (8.27 %) of the six airports.

## B. Labelled days: ground segment present, no gated pushback, alternative events

**Population.** The four labelled 2025 days (07-21, 07-13, 01-22, 11-12). Lane A (`mvt_minus_aobt3` present),
`join == "cs"`, a ground fix before take-off, and `t_push` null.

- The phase's ground fixes were rebuilt from the fixes file, from `t_first` to `t_to`. The fix count equals
  stage-1 `n_ground` on 99.4 % of the 6,828 rows.

Rows by sub-group:

| | EDDM | EGLL | EHAM | LEBL | LSZH | LEMD |
|---|---:|---:|---:|---:|---:|---:|
| stand run, first motion forward (as A10b) | 296 | 237 | 1,006 | 501 | 702 | 360 |
| stand run, no usable step (as A10a) | 43 | 52 | 130 | 64 | 40 | 17 |
| stand run, dwell guard (as A10c) | 8 | 0 | 6 | 0 | 0 | 0 |
| no stand run (as A9) | 192 | 986 | 353 | 740 | 441 | 654 |

**Events.** Each is read from the phase's ground fixes, on or after `t_dep1` (the first fix after the last silence
of ≥ 10 min).

| event | definition |
|---|---|
| `move` | stage-1 `t_move`, the first fix outside the stand run. Exists only when there is a stand run. |
| `dep1` | `t_dep1` |
| `gsX` | the first fix with gs > X kt whose following fixes within 30 s (at least one) all have gs > X; X = 3, 5, 10 |
| `d100` | the first fix more than 100 m from the position at `t_dep1` |

**Estimate.**

- `y_e = MVT − t_event + b`.
- `b` is fitted per airport, leave one day out: the median (`t_event − BLOCK`) over the same group and event on the
  other three days.
- The residual is `y_e − y`.
- `rmse_oof` is the production OOF (`oof_main_gap-62dd7c6e88`) on the same rows.
- `guarded` applies the 900 s guard against the OOF.
- Gain = mean(e_oof² − e_guarded²), in s² per row. A positive gain means the guarded event estimate beats the OOF.

### B1. Stand run, first motion forward (the "tail-first test failed" group)

| event | airport | n | median | IQR | RMSE event | RMSE OOF | RMSE guarded | trips | gain s²/row |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| move | EDDM | 296 | 0 | 230 | 251.2 | 163.0 | 178.5 | 5 | −5,275 |
| move | EGLL | 237 | 24 | 323 | 2840.4 | 230.9 | 337.6 | 19 | −60,682 |
| move | EHAM | 1,006 | 0 | 241 | 394.2 | 178.9 | 180.5 | 19 | −560 |
| move | LEBL | 501 | 1 | 317 | 316.2 | 223.6 | 303.4 | 4 | −42,034 |
| move | LSZH | 702 | 1 | 346 | 1005.1 | 174.1 | 229.4 | 16 | −22,305 |
| move | pooled 5 | 2,742 | 1 | 271 | 1018.9 | 190.0 | 235.6 | 63 | −19,411 |
| gs5 | EDDM | 296 | 2 | 187 | 488.5 | 163.0 | 146.7 | 4 | +5,051 |
| gs5 | EGLL | 223 | 20 | 307 | 2744.4 | 232.7 | 315.1 | 16 | −45,166 |
| gs5 | EHAM | 1,006 | 1 | 121 | 532.0 | 178.9 | 165.1 | 17 | +4,744 |
| gs5 | LEBL | 498 | 3 | 291 | 318.0 | 224.0 | 305.6 | 4 | −43,223 |
| gs5 | LSZH | 700 | 1 | 312 | 1206.6 | 174.0 | 224.1 | 25 | −19,937 |
| gs5 | pooled 5 | 2,723 | 2 | 208 | 1067.7 | 190.0 | 224.8 | 66 | −14,427 |
| d100 | EDDM | 296 | 1 | 195 | 489.6 | 163.0 | 148.4 | 4 | +4,552 |
| d100 | EHAM | 1,006 | 1 | 123 | 553.5 | 178.9 | 166.7 | 18 | +4,212 |
| d100 | pooled 5 | 2,739 | 2 | 232 | 1072.3 | 190.1 | 225.7 | 70 | −14,802 |
| gs3 | pooled 5 | 2,738 | 2 | 248 | 1100.7 | 190.1 | 228.1 | 69 | −15,878 |
| gs10 | pooled 5 | 2,640 | 1 | 187 | 1083.5 | 182.3 | 221.8 | 58 | −15,942 |
| dep1 | pooled 5 | 2,742 | 1 | 303 | 1203.0 | 190.0 | 273.2 | 125 | −38,511 |
| gs5 | LEMD | 358 | −81 | 347 | 1373.1 | 171.4 | 261.4 | 5 | −38,989 |

### B2. No stand run (ground segment, but no stationary run)

| event | airport | n (of group) | median | IQR | RMSE event | RMSE OOF | RMSE guarded | trips | gain s²/row |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dep1 | pooled 5 | 2,712 (2,712) | 1 | 469 | 4857.2 | 194.8 | 303.2 | 644 | −54,006 |
| gs3 | pooled 5 | 2,234 (2,712) | −2 | 328 | 4312.4 | 189.3 | 230.0 | 547 | −17,079 |
| gs5 | EDDM | 183 (192) | 1 | 223 | 332.9 | 146.3 | 147.0 | 3 | −182 |
| gs5 | EGLL | 560 (986) | −87 | 8,250 | 7987.3 | 203.6 | 274.4 | 425 | −33,816 |
| gs5 | EHAM | 324 (353) | −1 | 172 | 471.5 | 180.2 | 162.0 | 7 | +6,232 |
| gs5 | LEBL | 711 (740) | 1 | 235 | 1604.8 | 197.4 | 233.3 | 42 | −15,467 |
| gs5 | LSZH | 403 (441) | −2 | 195 | 2806.0 | 173.1 | 209.1 | 40 | −13,740 |
| gs5 | pooled 5 | 2,181 (2,712) | −2 | 299 | 4326.4 | 188.4 | 225.5 | 517 | −15,353 |
| gs10 | pooled 5 | 1,961 (2,712) | 1 | 276 | 4097.7 | 183.2 | 210.0 | 407 | −10,513 |
| d100 | EHAM | 323 (353) | 1 | 150 | 469.8 | 180.2 | 153.6 | 7 | +8,883 |
| d100 | pooled 5 | 2,158 (2,712) | −2 | 307 | 4206.5 | 188.6 | 232.2 | 495 | −18,330 |

### B3. The whole group (B1 + B2 + the smaller stand sub-groups), pooled over the five airports

| event | n (of 5,797) | median | IQR | RMSE event | RMSE OOF | RMSE guarded | trips | gain s²/row |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| move | 3,085 | 1 | 273 | 1136.1 | 193.8 | 235.1 | 83 | −17,732 |
| dep1 | 5,797 | 1 | 376 | 3463.2 | 194.3 | 289.6 | 796 | −46,121 |
| gs3 | 5,310 | 1 | 275 | 2954.4 | 191.8 | 237.2 | 452 | −19,470 |
| gs5 | 5,237 | 1 | 232 | 2926.4 | 191.4 | 231.9 | 427 | −17,128 |
| gs10 | 4,917 | −3 | 214 | 2686.8 | 185.4 | 227.0 | 346 | −17,134 |
| d100 | 5,239 | 2 | 249 | 2846.3 | 191.6 | 233.1 | 416 | −17,627 |

**Measured facts, part B.**

- The pooled five-airport gain is negative for every event, in every group.
- Every positive airport-level gain is at EDDM or EHAM:
  - B1: gs3, gs5 and d100 at both airports, and gs10 at EDDM.
  - B2: gs3, gs5 and d100 at EHAM.
  - The largest is +8,883 s²/row (d100, EHAM, B2, n 323).
- With the bias fitted leave one day out, the median residual is within ±6 s at EDDM, EHAM, LEBL and LSZH, for
  every event and group. At EGLL it ranges from −112 to +42 s, and at LEMD from −3,524 to −46 s.
- Without the guard, the event-estimate RMSE is above the OOF RMSE in every airport × event × group cell.
- The full per-airport table for all events and groups, LEMD and six-airport pooling included, is
  `$SCRATCH/adsb/adsb_cov_partb.csv`.
