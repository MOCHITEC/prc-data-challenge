# ADS-B pushback override: out-of-sample validation on new days, rules frozen (2026-09-29)

Sibling note to `adsb-gain-estimate.md` (§8 of which pre-registered the decision applied here) and
`adsb-stage1-feasibility.md` (the detection rules).

## 0. Pre-registration (written before any of these days was downloaded)

### 0.1 What is frozen (all fixed on 2025-07-21, the only labelled day seen before this step)

| item | frozen value | source |
|---|---|---|
| detection | stage-1 stand rule, pushback gate (share ≥ 0.5 of steps > 120° off track in 180 s), 15-min dwell guard, 25 m stand radius, 10-min silence; `scripts/adsb_stage1_feasibility.py` at commit `9f29f62`, unchanged | `[St1 §2.3]` |
| join | `CALLSIGN_flt` at `ADEP_mvt`, `FLIGHT_mvt` only if the callsign is null; take-off nearest `MVT_TIME` within ±30 min | `[St1 §2.2]` |
| bias `b` | per-airport median of (pushback − BLOCK) on the 914 lane-A covered rows of 2025-07-21, the pooled median where an airport has < 10 such rows (this is the `b25` of `adsb_gain_estimate.py` S4). **Not re-estimated on any new day before scoring.** | `[G S2, S4]` |
| guard | use `y_a = MVT − (t_push − b)` unless `|model − y_a| > 900 s`, then keep the model | `[G S3]` |
| airports evaluated | EDDM, EHAM, LEBL, LSZH, EGLL, LEMD | `[G §8]` |
| model input to the guard | 2025 days: production OOF `experiments/oof_main_gap-62dd7c6e88.parquet`. 2026 days: the production submission's prediction, read from `outputs/main_gap.parquet` (v14, read only; the same file the gain estimate used) | |

### 0.2 The decision rule, copied from `adsb-gain-estimate.md` §8 and the brief, applied as written

- **Gain** = mean over covered rows of (e_model² − e_override²), s² per covered row, where a covered row is a lane-A
  departure (`mvt_minus_aobt3` not null) at one of the six airports with a gated pushback. Pooled over the three new
  labelled 2025 days.
- **Coverage** = lane-A departures with a gated pushback at the six airports ÷ all lane-A departures of the day at
  the ten airports (the definition behind the 23.6 % of 2026-07-20), pooled by month over the four census days.
- **GO**: gain ≥ 5,000 s² per covered row, and coverage ≥ 10 % in January 2026 and ≥ 15 % in July 2026.
  Both → ingest all 62 scored days; only July → ingest July only.
- **KILL**: gain below ~2,000 s² per row, or coverage below 5 %.
- **Between**: UNDECIDED, with the one measurement that would settle it.
- Reported per day and per airport with a 95 % CI, so a single day cannot carry the verdict.

### 0.3 The days

Chosen from departure counts per day only (`data/work/feat_train.parquet` for 2025, `data/raw/ranking.parquet` for
2026, all departures, all ten airports). No label, OOF error or ADS-B statistic of any candidate day was looked at.

| day | weekday | role | departures (month median) | OOF fold | why this day |
|---|---|---|---:|---|---|
| 2025-07-13 | Sun | labelled, July | — | `jul` | prescribed by the brief and already sized (3.29 GB). It is a Sunday; I kept it as prescribed rather than swap it after the fact. |
| **2025-01-22** | Wed | labelled, January | 4,900 (4,949) | `jan` | a mid-week day in deep winter, so de-icing can occur (the stage-1 concern about the 15-min dwell guard). Away from the New-Year period (1–6 Jan) and two days before Storm Éowyn (24 Jan 2025), a known disruption. Traffic 1 % below the month median. |
| **2025-11-12** | Wed | labelled, November | 5,318 (5,450; Wednesdays 5,162–5,410) | `forward` | a mid-week day in the forward fold, the fold closest to the scored period. Middle of the Wednesday range. The day after the 11 Nov holiday in France, not on it. No disruption known to me. |
| 2026-01-19 | Mon | census, January | 5,128 (4,963) | — (unlabelled) | prescribed by the brief and already sized (3.12 GB). |
| **2026-01-28** | Wed | census, January | 4,725 (4,963; Wednesdays 4,359–4,800) | — | a second January day nine days from the 19th and on a different weekday, so feeder changes within the month have a chance to show. Avoids the early-January snow week (7 Jan has the month's lowest count, 4,359). |
| **2026-07-08** | Wed | census, July | 6,227 (6,226) | — | exactly the month median; early July, twelve days before the already-read 2026-07-20. |
| **2026-07-29** | Wed | census, July | 6,153 (6,226) | — | late July, nine days after 2026-07-20, 1 % below the median. Together with 07-08 it brackets 07-20, for EGLL's feeder stability. |

2026-07-20, already on disk from stage 1, is reported as a fifth census day but is **not** part of the
pre-registered July pooled coverage (it was already used in the gain estimate).

**What was done in which order.** This section, then `scripts/adsb_oos_validation.py` (the scorer, written and run
on 2025-07-21 only, as a reproduction check), then the downloads one day at a time, then the scoring.

**Archive sizes (HEAD, before download)** `[H]`: 2025-07-13 3.288 GB, **2025-01-22 2.017 GB**, 2025-11-12 3.333 GB,
2026-01-19 3.122 GB, 2026-01-28 3.077 GB, 2026-07-08 4.030 GB, 2026-07-29 4.070 GB; 22.9 GB in all. The January 2025
archive is small because the whole month is: 2025-01-15 2.234, 01-20 2.097, 01-23 2.203 GB (01-21 has no release).
`[X]` The adsb.lol feeder network was smaller in early 2025, so lower January 2025 coverage is expected and is part of
what this day measures. The day was kept as pre-registered.

---

**Script:** `scripts/adsb_oos_validation.py` (new; read-only for the repo, imports the stage-1 module unchanged).
Run, after the stage-1 `fetch → extract → purge → analyse` loop for each day:

```
POLARS_MAX_THREADS=2 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage \
.venv/bin/python scripts/adsb_oos_validation.py          # ~1 min
```

**Sources.** `[V Sx]` is section Sx of that script's output, saved at `…/scratchpad/adsb/oos_validation.log`.
`[A day]` is the stage-1 `analyse` log for that day, `…/scratchpad/adsb/analyse_<day>.log`. `[H]` is a HEAD request.
`[G]` is `adsb-gain-estimate.md`. `[X]` marks an inference.

**Reproduction check, before any new day was on disk** `[V S0]`. With the frozen bias, the scorer gives +19,422 s² per
covered row on 2025-07-21 (all airports, 914 rows, 3 guard trips = 0.33 %). The gain estimate's cross-fitted figure
was +19,400. On the six airports it gives +20,317 (845 rows). On 2026-07-20 it gives 4 trips on 1,428 rows (0.28 %),
the same as `[G S4]`. The frozen bias is EDDF +52.9, EDDM +50.3, EHAM −3.5, LEBL +45.1, LEMD +115.7, LSZH +30.5 s,
and **+27.2 s (the pooled median) at EGLL and LIRF**, which had fewer than 10 rows on the design day.

## 1. The answer

1. **Verdict: GO, ingest all 62 scored days. This is the pre-registered rule applied as written** `[V S3]`.
   - **Gain.** Pooled over the three new labelled days, the gain is **+16,360 s² per covered lane-A row** at the six
     airports (2,022 rows). The 95 % CI is [14,100, 18,700] by row bootstrap and [12,900, 19,900] by
     airport×hour×day cluster. The GO threshold is 5,000.
   - **Coverage.** Pushback coverage is **17.6 % in January 2026** (threshold 10 %) and **22.3 % in July 2026**
     (threshold 15 %), over the four pre-registered census days.
   - **No day carries it alone.** Every labelled day clears 5,000 by itself: 2025-01-22 +21,000, 2025-07-13 +18,800,
     2025-11-12 +11,400.
2. **It shrank, but not tenfold.**
   - 2025-07-21 gave +19,400. Out of sample the pooled gain is +16,400, 16 % lower.
   - The forward-fold November day, the one nearest the scored period, is **+11,400, 41 % below the design day**.
   - All of the shrinkage is at EGLL and LEMD (point 3). At EDDM, EHAM, LEBL and LSZH the out-of-sample gain is
     +15,700 to +21,100 per row, within the design day's CIs.
   - `[X]` The tenfold shrink seen last week was board ΔRMSE against OOF ΔRMSE. This measurement is a per-row OOF
     gain on new days, so it does not test that transfer. What it rules out is the first way the estimate could have
     failed: rules that were overfit to one day. They were not.
3. **EGLL is negative under the frozen rules, and I did not fix it.**
   - Pooled over the three days, EGLL is **−6,100 per row** [−14,500, +2,300] (191 rows). On 2025-11-12 alone it is
     −15,600 [−26,800, −5,700].
   - Cause: EGLL's pushback-minus-BLOCK offset is **+133 to +177 s on every new day** (median over the three days
     +175 s). The frozen bias uses the pooled +27 s, because EGLL had only 5 rows on 2025-07-21. Those 5 rows had
     given +153 s in-sample `[G S2]`.
   - This is the biggest thing I would change (§4). It is not changed in the score.
   - LEMD is +8,400 [−6,700, +23,300]. It is positive but not significant, and it is negative on 2025-11-12.
4. **The guard trip rate rose on two of the three labelled days, but gross detection failures did not.**

   | day | trips, all airports | trips, six airports | ADS-B error > 10 min, six airports |
   |---|---:|---:|---:|
   | 2025-07-21 (design day) | 0.33 % | 0.12 % | 5 / 845 (0.59 %) |
   | 2025-07-13 | 0.62 % | 0.18 % | 2 / 560 (0.36 %) |
   | 2025-01-22 | **0.77 %** | **0.82 %** | 4 / 609 (0.66 %) |
   | 2025-11-12 | 0.19 % | 0.12 % | 4 / 853 (0.47 %) |

   On the 2026 census days the trip rate is 0.07–0.35 % `[V S1, S2]`.

   The guard catches the failures; they are not getting past it into the gain. Rows made worse by more than 2 min are
   72 of 2,022 (3.6 %), and 47 of those are at EGLL. Without EGLL it is 25 of 1,831 (1.4 %), against 11 of 845
   (1.3 %) on the design day.
5. **January.**
   - **De-icing does not trip the 15-min dwell guard in any number that matters.** On 2025-01-22 it removed 6
     gated pushbacks (5 at EDDM), against 609 kept, and all 6 were within 5 min of BLOCK. On the January 2026 days
     it removed 9 and 4 flights, against 6 and 4 on the July 2026 days `[V S2, S4]`.
   - **January coverage is lower than July's but far above the placeholder.** It is 17.6 % against 22.3 % in July.
     The gain estimate's CENTRAL assumed January at half of July, ≈ 11.5 % `[G §5]`.
   - EGLL, which sets most of the January/July difference, is 22–24 % in January and 56–61 % in July.

## 2. Labelled days: out-of-sample gain per day and per airport `[V S1]`

Covered = lane A, gated pushback, six airports. Gain = mean(e_model² − e_override²), s² per covered row. The CI is a
95 % row bootstrap (B = 2,000). The model is the production OOF `main_gap-62dd7c6e88`. Every row of each day is in
the expected fold (`jul`, `jan`, `forward`) and has an OOF prediction, and `max |MVT − BLOCK − y| = 0` on every day.

| day | lane-A RMSE, day (10 apts) | covered rows (share of lane A) | model RMSE → override | MAE | **gain / row** | 95 % CI | airport-hour CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2025-07-13 (Sun) | 307.6 | 560 (9.2 %) | 165.9 → 93.2 | 119.6 → 36.1 | **+18,834** | [14,976, 22,874] | [13,729, 23,819] |
| 2025-01-22 (Wed) | 180.8 | 609 (12.6 %) | 181.9 → 109.8 | 124.7 → 45.3 | **+21,013** | [16,595, 26,276] | [15,024, 27,940] |
| 2025-11-12 (Wed) | 176.7 | 853 (16.3 %) | 151.8 → 107.9 | 104.3 → 50.7 | **+11,419** | [8,099, 14,933] | [5,829, 17,778] |
| **pooled** | | **2,022** | 165.3 → 104.6 | 114.7 → 45.0 | **+16,362** | **[14,075, 18,732]** | **[12,900, 19,948]** |
| *2025-07-21, design day (in-sample)* | 271.4 | 845 (14.2 %) | 167.9 → 88.8 | 121.3 → 35.1 | *+20,317* | *[15,817, 25,333]* | |

The pooled airport-hour CI clusters on airport × hour × day. The equal-weight mean of the three day gains is +17,090.

**Per airport** (gain per row; n; 95 % row-bootstrap CI):

| ADEP | 2025-07-13 | 2025-01-22 | 2025-11-12 | **pooled** | pooled 95 % CI | design day `[G S3]` |
|---|---:|---:|---:|---:|---:|---:|
| EDDM | — (n 0) | +22,992 (219) | +18,097 (245) | **+20,407** (464) | [16,889, 24,583] | +19,000 |
| EHAM | +21,251 (330) | +21,188 (195) | +15,255 (267) | **+19,214** (792) | [15,716, 22,893] | +17,300 |
| LEBL | +14,315 (118) | +13,979 (62) | +17,871 (126) | **+15,711** (306) | [9,628, 23,261] | +19,000 |
| LSZH | +13,465 (61) | +34,248 (68) | +14,685 (68) | **+21,060** (197) | [13,727, 30,962] | +46,900 |
| EGLL | +15,147 (12) | +6,687 (65) | **−15,630** (114) | **−6,102** (191) | [−14,523, +2,297] | (+117,500, n 5) |
| LEMD | +21,586 (39) | — (n 0) | −7,137 (33) | **+8,421** (72) | [−6,663, +23,260] | +11,600 |

- **EDDM had no pushbacks at all on 2025-07-13.** Its coverage fell from 55–61 % to 0 `[V S1]`. `[X]` The feeder
  that covers the Munich aprons was down that day. This is the day-to-day feeder variance of `[St1 §0.7]` again, now
  at a GO airport.
- **Concentration.** The top 10 rows carry 13.7 % of the pooled gain; without them it is +14,190. 376 rows are made
  worse, costing 8.0e6 s² against a gain of 41.1e6 s² `[V S1]`.
- **The gain is not driven by hard days.** The design day had lane-A RMSE 271; the January and November days are at
  177–181, below the pooled 213 `[G §1]`, and still give +11k to +21k.
- **The model is not closer to ADS-B in 2026 than in 2025.** RMS disagreement inside the guard is 151–164 s on the
  three 2025 days, and 159–182 s on the five 2026 days `[V S2, S4]`. So the worry in `[G §4]`, that the 2026 gain
  would be smaller because v14 is already closer to ADS-B, does not show on these days.

## 3. Census days, 2026: coverage and trips `[V S2]`

For the 2026 days, the guard compares the pushback with the **production submission's prediction**, read from
`outputs/main_gap.parquet` (v14, read only). The board's current best, v16 (279.26), comes from a parallel session,
and its file was not located here. `[X]` v14 and v16 differ by 0.26 s on the board, which is immaterial to a
900 s guard. Every lane-A departure in scope has a prediction.

Figures below are the share of lane-A departures with a gated pushback.

| day | EDDM | EGLL | EHAM | LEBL | LEMD | LSZH | **pooled** (six ÷ ten airports' lane A) | after guard | trips, six airports | trips, all airports |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-01-19 (Mon) | 0.608 | 0.217 | 0.378 | 0.274 | 0.160 | 0.294 | **0.177** (890 / 5,030) | 0.176 | 3 / 890 (0.34 %) | 0.32 % |
| 2026-01-28 (Wed) | 0.662 | 0.236 | 0.401 | 0.180 | 0.101 | 0.343 | **0.175** (816 / 4,669) | 0.174 | 3 / 816 (0.37 %) | 0.35 % |
| 2026-07-08 (Wed) | 0.610 | 0.555 | 0.380 | 0.486 | 0.109 | 0.182 | **0.221** (1,356 / 6,144) | 0.220 | 2 / 1,356 (0.15 %) | 0.14 % |
| 2026-07-29 (Wed) | 0.598 | 0.612 | 0.484 | 0.369 | 0.085 | 0.193 | **0.225** (1,368 / 6,067) | 0.225 | 1 / 1,368 (0.07 %) | 0.07 % |
| *2026-07-20 (Mon), not pre-registered* | 0.604 | 0.578 | 0.508 | 0.333 | 0.171 | 0.179 | 0.229 (1,404 / 6,120) | 0.229 | 3 / 1,404 (0.21 %) | 0.28 % |
| **January, pooled** | | | | | | | **0.176** | 0.175 | | |
| **July, pooled** (07-08, 07-29) | | | | | | | **0.223** (incl. 07-20: 0.225) | 0.223 | | |

- **EGLL's feeder is stable across July 2026: 56–61 % on all three days.** In January it is 22–24 %.
- EDDM's coverage is the steadiest of the six in 2026, at 60–66 % on every day.
- The label-free share with `|AOBT_3 − push| > 20 min` is 1.7 % on 2026-01-19, and 0.1–0.9 % on the other 2026
  days. `[X]` This could be either late NM stamps or detection failures. The guard trip rate that day is unremarkable,
  at 0.34 %.
- Outside the six airports, LIRF shows 6.8–9.0 % pushback in January 2026, against 0 % on every July day. It is not
  evaluated and it is not counted.

## 4. What I wanted to change after seeing the days, and did not

None of this enters the verdict. The figures are from `[V S4]`.

1. **EGLL's bias.** The frozen rule falls back to the pooled median (+27 s) below 10 design-day rows. EGLL's
   offset is +170 s (2025-01-22), +133 s (07-13) and +177 s (11-12).
   - With EGLL's bias set to its own median on these same days (+175 s, **in-sample**), EGLL goes from −6,100 to
     **+19,100** [12,600, 26,000].
   - The per-day medians agree within 45 s. `[X]` An EGLL bias estimated on labelled 2025 days other than the ones
     scored would therefore very likely hold, but that is a changed rule and needs its own out-of-sample test.
   - `[X]` For production, `b` should come from all the labelled days available (now 4), not from 2025-07-21 alone.
2. **Dropping EGLL**, or keeping only the four GO airports: the pooled gain would be +18,700 [16,300, 21,100], or
   +19,100 [16,700, 21,600]. It passes either way, so the verdict does not depend on this.
3. **LEMD** is positive pooled but negative on the November day, with n = 72 over three days. I would not include
   it without more labelled days.
4. **The dwell guard.** No change is wanted. It drops about 1 % of pushbacks in January and in July, and the
   January drops are accurate stand exits, so keeping them would add a few rows at most.

## 5. Disk and cost

- **Peak raw footprint:** one day at a time, at most **4.07 GB** (2026-07-29, 3 parts). `fetch` checked for raw +
  3 GB free before each day, and each day's parts were deleted by `purge` before the next `fetch`.
- **Free space:** `df` showed 13.53 GB free before the first day and 13.43 GB after the last `[run_oos.log]`.
- **Downloaded:** 22.9 GB in 18.9 min for seven days, 2.5–3.5 min per day including extraction.
- **Kept:** in `…/scratchpad/adsb/`, 130 MB in total. That is seven filtered fix files (6.8–17.4 MB each), seven
  joined files, and the logs.
- **Written in the repo:** this note and `scripts/adsb_oos_validation.py`. Nothing went to `outputs/` or `src/`, and
  nothing was fitted or submitted.

## 6. Verdict

**GO: ingest all 62 scored days.** Every pre-registered condition holds, and it holds on each labelled day separately:

| condition | threshold | measured |
|---|---|---|
| out-of-sample gain | ≥ 5,000 s² per covered row | +16,400 [14,100, 18,700]; each day ≥ +11,400 |
| coverage, January 2026 | ≥ 10 % | 17.6 % |
| coverage, July 2026 | ≥ 15 % | 22.3 % |

**Caveats that travel with the GO:**
- EGLL is negative under the frozen bias. Before any submission it needs a bias calibrated on labelled days, and
  that calibration needs its own out-of-sample check (§4).
- Single-day feeder outages happen, even at a GO airport (EDDM on 2025-07-13). Coverage must be read per day from the
  full ingest, not assumed.
- This measures the per-row gain against the OOF, not the board transfer. The ≈10× board shrinkage of earlier
  changes is untested here. `[X]`
