# ADS-B stage 1: can adsb.lol see departures leave the stand at our 10 airports? (2026-09-29)

**Script:** `scripts/adsb_stage1_feasibility.py` (written from scratch; no third-party extractor code was read
or copied). It is read-only with respect to the repo: nothing goes to `outputs/`, `src/`, `configs/` or `data/`,
and no model is fitted. Run it with the repo venv:

```
S=<scratch dir>  # default: the session scratchpad .../scratchpad/adsb
.venv/bin/python scripts/adsb_stage1_feasibility.py sizes   globe_history_2025 v2025.07.21-planes-readsb-prod-0
.venv/bin/python scripts/adsb_stage1_feasibility.py fetch   globe_history_2025 v2025.07.21-planes-readsb-prod-0
.venv/bin/python scripts/adsb_stage1_feasibility.py extract v2025.07.21-planes-readsb-prod-0
.venv/bin/python scripts/adsb_stage1_feasibility.py purge   v2025.07.21-planes-readsb-prod-0
POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage \
.venv/bin/python scripts/adsb_stage1_feasibility.py analyse v2025.07.21-planes-readsb-prod-0 2025-07-21
```

The same commands were run for `globe_history_2026` / `v2026.07.20-planes-readsb-prod-0` / `2026-07-20`.

**Sources.** Every number carries its source:
- `[L25 §x]` is section `[x]` of the `analyse` output for 2025-07-21.
- `[L26 §x]` is the same for 2026-07-20.
- `[H]` is a HEAD request on a release asset, or a `git ls-remote --tags` listing.
- `[inline]` is a one-off polars query on the script's saved `*_joined.parquet`. Each is described where it is used.
- `[X]` marks an inference.

**Data and licence.** The data is the adsb.lol `globe_history_2025` and `globe_history_2026` GitHub releases,
from pod `planes-readsb-prod-0`. Each day is one tar of readsb `traces/*/trace_full_<icao24>.json` (gzip JSON),
split into 2 GB parts. Each release's `README.txt` says: *"This database is made available under the Open
Database License: http://opendatacommons.org/licenses/odbl/1.0/"*, and it ships `LICENSE-ODbL.txt`. That is
ODbL 1.0, and it matches `https://www.adsb.lol/docs/open-data/historical/`. **Attribution to carry in any
write-up or submission:** "Contains data from adsb.lol (globe_history), made available under the Open Database
License (ODbL) 1.0."

**Timestamps.** Everything is UTC. readsb's `timestamp` is the Unix epoch of UTC midnight of the archive day,
and each trace point's first element is seconds after it. Our parquet datetimes are tz-aware UTC and were
converted with `dt.epoch`. The check below shows the clocks agree. ADS-B take-off minus `MVT_TIME_UTC_mvt` has a
median of −1.7 s, and it is within ±2 min on **99.7 %** of 4,122 flights on the 2025 day and 99.9 % of 4,480 on
the 2026 day `[inline]`.

**Leakage.** `BLOCK_TIME_UTC_mvt` is read only to *evaluate* on 2025. The stand and pushback detection reads
ADS-B positions, the transponder's own track, `CALLSIGN_flt`, `FLIGHT_mvt`, `ADEP_mvt` and `MVT_TIME_UTC_mvt`,
all of which are present on scored rows.

---

## 0. The answer

1. **One day's archive is 3.1–4.0 GB**: 2025-07-13 3.288 GB, 2025-07-21 3.552 GB, 2026-01-19 3.122 GB, 2026-07-20
   4.017 GB, in two or three 2 GB parts `[H]`. That fits in the 13 GB of free disk with a large margin. The peak
   raw footprint here was 4.02 GB, one day at a time.
2. **Every day of January and July 2026 exists**: 31 + 31 `…-planes-readsb-prod-0` tags in `adsblol/globe_history_2026` `[H]`.
3. **The join works where there is any data.** `CALLSIGN_flt` finds the aircraft on **95–100 %** of matched-lane
   departures at 9 airports, on both days. The ADS-B take-off agrees with `MVT_TIME` within 2 min on 99.7 %.
   **LTFM is unjoinable.** adsb.lol holds 270 fixes near LTFM on the 2025 day, none of them on the ground, and on
   the 2026 day only 15 % of departures are seen, airborne only `[L25 §fixes]`, `[L26 §coverage]`.
4. **Ground coverage is common, but a usable stand exit is not.** A ground fix before take-off exists for 68 %
   (2025) and 73 % (2026) of matched departures. In most of them the transponder is first received mid-taxi, so
   the first fix is **not** an off-block: it is 150 s late at the median, and within ±5 min only 43 % of the time
   `[L25 §accuracy:dep1]`. **The usable signal is an observed pushback:** a stationary run at the stand followed
   by backwards motion. It is found on **15.4 %** (2025) and **23.3 %** (2026) of matched departures
   `[L25/L26 §coverage matched]`.
5. **When a pushback is seen, it is much closer to BLOCK than `AOBT_3` is.** On 914 such flights on 2025-07-21
   `[L25 §accuracy:push]`:

   | | median | IQR | within ±2 min | within ±5 min | MAE |
   |---|---:|---:|---:|---:|---:|
   | ADS-B pushback − BLOCK | +27 s | 64 s | **90.3 %** | **97.3 %** | **71 s** |
   | `AOBT_3` − BLOCK (same flights) | +2 s | 317 s | 39.5 % | 74.9 % | 238 s |

   ADS-B is closer than `AOBT_3` on **83 %** of these flights.
6. **On the late-NM-stamp flights, ADS-B rarely sees them, and when it does it is right.** This is the most
   important number in the report. On the day there were 124 flights with `g = AOBT_3 − BLOCK > 1,200 s` and no
   schedule fill. ADS-B **saw a pushback on 3 of them (2.4 %)**. On those 3 its error was +35, −71 and +486 s,
   against `AOBT_3` errors of +1,366, +1,277 and +1,433 s `[L25 §late NM]`, `[inline]`.

   The reason is where these flights are. They concentrate at LTFM, EGLL, LFPG and LIRF, the airports with the
   least surface coverage.

   Projected onto the 6,152-row OOF population with each airport's pushback-detection rate `[X]`, ADS-B reaches
   **≈1,040 rows (17 %)** carrying **≈1.5 pp of the 16.7 pp** of lane-A SSE at the 2025-07-21 rates. At the
   2026-07-20 rates it reaches ≈1,730 rows and ≈2.8 pp. **It does not solve the late-NM bucket.** At most it trims
   the EDDM/LEBL/EGLL part of it.
7. **Coverage is not stable between days, because it depends on individual volunteer feeders.**
   - EGLL had ground fixes on 41 % of departures on 2025-07-21 and 99 % on 2026-07-20. Its pushback rate went
     from 0.8 % to 58 %.
   - LIRF went the other way, from 94 % ground fixes to **0 %**.
   - The 2026 figures that matter for scoring can only come from a census of more 2026 days `[L25/L26 §coverage]`.
8. **Verdict: GO-FOR-SOME-AIRPORTS.**
   - **GO:** EDDM and EHAM, with pushback on 48–60 % of departures on both days and a 26–58 s MAE against 207–278
     s for `AOBT_3`.
   - **GO, smaller:** LEBL and LSZH, with 13–33 % seen and a 52–77 s MAE.
   - **CONDITIONAL:** EGLL, which was strong on the 2026 day and nearly absent on the 2025 day. It needs a
     multi-day census, and the model needs 2025 training rows that have the feature.
   - **MARGINAL:** EDDF (3–10 %) and LEMD (6–17 %, with worse tails).
   - **NO-GO:** LFPG (0–0.3 %), LIRF (0–1.4 %) and LTFM (0 %).

**The main caveat on these numbers.** I designed the stand rule, the pushback gate and the 15-min dwell guard
while looking at the 2025-07-21 errors against BLOCK. §2.3 lists the three revisions. The accuracy in point 5 is
therefore **in-sample for the design choices** and optimistic. It must be re-measured on a different 2025 day
before stage 2. The 2026 day confirms coverage and stability (§5), not accuracy.

---

## 1. Archive size, availability, and cost

| day | parts | bytes | source |
|---|---:|---:|---|
| 2025-07-13 | 2 | 3,288,320,000 | `[H]` |
| 2025-07-21 | 2 | 3,551,692,800 | `[H]` |
| 2026-01-19 | 2 | 3,121,659,392 | `[H]` |
| 2026-07-20 | 3 | 4,016,839,168 | `[H]` |

- **Availability.** `git ls-remote --tags https://github.com/adsblol/globe_history_2026` lists 31 `prod-0` tags
  for January 2026 and 31 for July 2026 `[H]`. The GitHub releases API and HTML pages were not reachable from this
  session (HTTP 403 at the proxy). Download URLs (`releases/download/<tag>/<tag>.tar.aa`, …) worked, and part
  sizes came from `Content-Length` on HEAD.
- **Throughput.**
  - Download: a 200 MB range read took 2.7 s, about 75 MB/s `[inline]`.
  - Extraction: streams the parts back-to-back through `tarfile` in `r|` mode, never re-joining them on disk, with
    3 parse workers. It took **97 s** for 70,818 trace files on 2025-07-21 `[inline]`. About 3,800–4,100 aircraft
    per day come near one of the 10 airports.
- **Filtered output:** 15.0 MB (2025-07-21, 1,255,164 fixes) and 17.3 MB (2026-07-20, 1,475,133 fixes)
  `[extract logs]`.
- **Cost of a full run** `[X]`:
  - Jan+Jul 2026 is 62 days × ≈3.6 GB ≈ 220 GB of download. At the measured rates that is ≈1 min of download and
    ≈1.6 min of extraction per day, about 3 h in total, with ≈1 GB of filtered output.
  - All of 2025 would be ≈1.3 TB and ≈10 h. A subset of 2025 days is the realistic training set.
  - Disk is **not** the limit if each day is fetched, filtered and purged in turn. The extractor could also read
    `curl` output directly, so no raw part would be stored at all. Network volume and wall time are the limits.

## 2. Method

### 2.1 Filter (the `extract` step)

A fix is kept if it lies within **8 km** of an airport's ARP (`data/external/airports.parquet`) and is either on
the ground (`alt == "ground"`) or at ≤ 4,000 ft. The low-altitude part is kept so that a take-off time can be
read. `[X]` Every runway end of the 10 airports is within ~7 km of its ARP.

The callsign appears only in the occasional aircraft-state dict of a trace point. The script forward-fills and
back-fills it within a readsb leg (flag bit 2 starts a leg).

### 2.2 Join (the `analyse` step)

Candidate aircraft are those carrying `CALLSIGN_flt` at `ADEP_mvt`. If `CALLSIGN_flt` is null, `FLIGHT_mvt` is
tried instead. A candidate must have fixes within 3 h of `MVT_TIME`. The departure then takes the aircraft's
take-off nearest `MVT_TIME` within ±30 min. The take-off is the first airborne fix after a ground fix.

ICAO24 cannot be derived from our files, because they carry no registration. The ICAO24 comes out of the join.

Scope: departures with `MVT_TIME` in [day 02:00, day 24:00) UTC, so that the stand phase falls inside the day's
archive. That is 6,153 departures on 2025-07-21 and 6,203 on 2026-07-20.

### 2.3 Definitions

- **Ground phase.** The departure's ground fixes before take-off, back to the previous airborne fix (the arrival
  leg), for at most 4 h.
- **Silence.** A gap of ≥ 10 min between ground fixes: the transponder was off, or nothing was received.
  - `t_dep1` is the first fix after the last silence, i.e. when the departure was first heard.
- **Run.** Consecutive ground fixes that all stay within **25 m** of the run's first fix. A run may span a
  silence if the fixes either side of it are in the same place.
  - `[X]` 25 m is above the ADS-B surface position noise at NACp ≥ 8 (< 30 m 95 %, typically much less), and
    well below a push of one aircraft length.
- **Stand.**
  - If the departure starts after a silence, or the phase does not open with a landing: the run that holds the
    `t_dep1` fix. It counts only if it has ≥ 2 fixes, or 1 fix at ≤ 1 kt.
  - If the aircraft stayed visible from landing to take-off: the longest run, which must be ≥ 10 min.
  - `t_move` is the first fix outside the stand run. `t_move_lo` is its last fix inside. Their gap is the time
    resolution: median 7.3 s and p90 26 s on the 926 pushback flights of 2025 `[inline]`.
- **Pushback signature (the gate).** Take the displacement steps of ≥ 3 m in the 180 s after the stand run. The
  signature is the share of them whose direction is more than 120° off the track the transponder reports. The
  gate requires ≥ 0.5, i.e. the aircraft moved tail-first. It is label-free.
  - `[X]` I read this as the surface-message track carrying the aircraft's heading while it is pushed. The
    evidence is empirical (§3.2): gated exits are accurate, and ungated ones are not.
- **Dwell guard.** No later stop of ≥ 15 min before take-off. A second long stop means the push that was seen may
  not be the one that started this departure. It removed 3 of 917 flights, 2 of them > 10 min wrong `[inline]`.
- **Estimators** reported in the script:
  - `move`: any stand exit.
  - `push`: the gated `t_move`. **This is the primary estimator.**
  - `push_lo`: `t_move_lo` on the same flights.
  - `dep1`: the first fix, with no stand test at all.

**How these definitions were reached, stated plainly.** The first version took the first ground fix of the phase
as the stand. Because it walked back to the arrival leg, it anchored on the landing roll, and a move was found on
3 % of flights. The second version took the longest stationary run ≥ 60 s. Its tails picked holding points and
taxi-in stops.

The third version (the stand is the run at the first departure fix, with the pushback gate and the dwell guard)
was chosen after inspecting 2025-07-21 errors against BLOCK. Hence the in-sample warning in §0.

## 3. 2025-07-21: coverage, join rate, accuracy

### 3.1 Coverage and join rate (matched lane: `CALLSIGN_flt` and `AOBT_3` present) `[L25 §coverage matched]`

| airport | n | join (seen, `CALLSIGN_flt`) | ground fix before take-off | stand run found | **pushback detected** |
|---|---:|---:|---:|---:|---:|
| EDDF | 642 | 0.974 | 0.967 | 0.484 | 0.097 |
| EDDM | 503 | 0.996 | 0.990 | 0.837 | **0.555** |
| EGLL | 660 | 0.989 | 0.414 | 0.082 | 0.008 |
| EHAM | 688 | 0.981 | 0.980 | 0.856 | **0.477** |
| LEBL | 544 | 0.967 | 0.965 | 0.618 | 0.265 |
| LEMD | 608 | 0.979 | 0.615 | 0.178 | 0.063 |
| LFPG | 623 | 0.989 | 0.393 | 0.008 | 0.000 |
| LIRF | 492 | 0.963 | 0.943 | 0.539 | 0.014 |
| LSZH | 392 | 0.982 | 0.954 | 0.691 | 0.130 |
| LTFM | 799 | **0.000** | 0.000 | 0.000 | 0.000 |
| all | 5,951 | 0.849 | 0.680 | 0.397 | **0.154** |

**All departures.** Over all 6,153 departures, 96.7 % carry a callsign. Where `CALLSIGN_flt` is null (202 rows),
`FLIGHT_mvt` joins 130, 80 have ground fixes, and 12 have a pushback, with errors of −54 to +244 s
`[L25 §coverage all]`, `[inline]`. `FLIGHT_mvt` is therefore a usable fallback only on the unmatched lane.

**Where the fixes come from** `[L25 §fixes]` (ground fixes / aircraft):

| EDDF | EDDM | EGLL | EHAM | LEBL | LEMD | LFPG | LIRF | LSZH | LTFM |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 55,509 / 479 | 115,563 / 433 | 10,071 / 535 | 292,647 / 613 | 46,318 / 423 | 18,258 / 444 | 1,087 / 616 | 25,858 / 353 | 54,606 / 321 | **0 / 97** |

**Where the first fix falls.** The median time from first departure fix to take-off is 9–13 min at EDDF, EDDM,
EGLL, EHAM, LEBL and LSZH, 23 min at LEMD and 66 min at LFPG `[L25 §sampling]`. At LIRF it is 416 s, against a
median BLOCK-to-take-off of 1,143 s `[inline]`. `[X]` At LIRF, EDDF,
LEBL and LSZH, reception begins mid-taxi on most flights, which is why a ground fix is common but a pushback is
rare.

### 3.2 Accuracy against BLOCK, compared with `AOBT_3` on the same flights `[L25 §accuracy:push]`

Errors are in seconds; +ve = later than BLOCK.

| airport | n | ADS-B median | ADS-B IQR | ADS-B ±2 min | ADS-B ±5 min | ADS-B MAE | `AOBT_3` median | `AOBT_3` IQR | `AOBT_3` ±2 min | `AOBT_3` ±5 min | `AOBT_3` MAE | ADS-B closer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EDDF | 62 | +55 | 85 | 0.79 | 0.95 | 154 | −69 | 325 | 0.39 | 0.85 | 190 | 0.76 |
| EDDM | 279 | +50 | 40 | 0.96 | **0.99** | **58** | +116 | 249 | 0.43 | 0.77 | 207 | 0.78 |
| EGLL | 5 | +153 | 98 | 0.00 | 1.00 | 176 | +535 | 309 | 0.20 | 0.40 | 481 | 0.80 |
| EHAM | 328 | −3 | 18 | 0.97 | **0.97** | **26** | −144 | 307 | 0.38 | 0.72 | 278 | **0.95** |
| LEBL | 144 | +45 | 58 | 0.85 | 0.97 | 77 | +174 | 306 | 0.35 | 0.67 | 225 | 0.79 |
| LEMD | 38 | +118 | 92 | 0.50 | 0.95 | 282 | −3 | 302 | 0.45 | 0.89 | 188 | 0.55 |
| LIRF | 7 | +91 | 1,123 | 0.57 | 0.57 | 746 | −4 | 1,142 | 0.29 | 0.57 | 755 | 0.43 |
| LSZH | 51 | +30 | 15 | 0.92 | 0.98 | 52 | +126 | 236 | 0.47 | 0.84 | 190 | 0.84 |
| **all** | **914** | **+27** | **64** | **0.90** | **0.97** | **71** | +2 | 317 | 0.39 | 0.75 | 238 | **0.83** |

- **The last fix still at the stand** (`push_lo`, same 914 flights) is slightly better: median +15 s, ±2 min
  0.919, MAE 65 s `[L25 §accuracy:push_lo]`.
- **The positive offset** (+30 to +55 s at most airports) is systematic. `[X]` BLOCK is stamped at push start or
  brake release, before the aircraft has moved 25 m. A model would absorb it per airport.
- **Without the gate it does not work** `[L25 §accuracy:move]`, `[L25 §accuracy:dep1]`:
  - Any stand exit (`move`, 2,360 flights): median +137 s, ±5 min 0.63, MAE 309 s. It is **worse than
    `AOBT_3`** (MAE 250).
  - The first fix (`dep1`, 4,048 flights): median +151 s, ±5 min 0.43.
  - At LIRF, `move` is +682 s at the median. `[X]` It is a taxiway stop, not the stand.
- **Tails.** 8 of 914 gated flights are > 10 min off `[inline]`. In 4 of the 8, `AOBT_3` agrees with ADS-B
  within 5 min, so BLOCK is the outlier; 2 of those are schedule fills. The other 4 are detection failures, e.g.
  a pushback seen 77–91 min before take-off with a normal BLOCK (LEMD IBB37VK, EDDF CFG5YW).
- **Early NM stamps.** On 14 gated flights, `AOBT_3` and the pushback disagree by > 20 min `[inline]`. Two are
  the detection failures above (IBB37VK, CFG5YW) and two are late-NM rows (§4). **The other 10 are all at EHAM,
  with `AOBT_3` 24.5–38.9 min *early* (−1,471 to −2,334 s).** On 9 of those 10, ADS-B is within 43 s of BLOCK.
  That is the mirror of the late-NM bucket: `g < −1,200 s`, not a fill, 1,733 OOF rows, 1.8 % of lane-A SSE,
  755 of them at EHAM `[inline]`. It is the one place where coverage and error coincide.

### 3.3 What it would do to the target, raw `[L25 §target:push]`

This compares, on the gated flights, `y_adsb = MVT − t_push` with the production OOF prediction
(`experiments/oof_main_gap-62dd7c6e88.parquet`).

| | EDDF | EDDM | EGLL | EHAM | LEBL | LEMD | LIRF | LSZH | pooled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| n | 62 | 279 | 5 | 328 | 144 | 38 | 7 | 51 | 914 |
| OOF RMSE | 180 | 154 | 377 | 154 | 190 | 147 | 336 | 229 | **171** |
| raw ADS-B y RMSE | 600 | **85** | 184 | **81** | **144** | 901 | 1,347 | **87** | 284 |

Raw ADS-B halves the RMSE at EDDM, EHAM and LSZH. It loses where a single detection failure (−4.6 k s, −5.5 k s)
dominates a small cell. `[X]` This is the case for using it as a gated feature inside the model, not as a
replacement for the prediction.

## 4. The late-NM-stamp flights (`g = AOBT_3 − BLOCK > 1,200 s`, not a schedule fill)

**Definition.** Schedule fill = `|BLOCK − SCHED| ≤ 12 s`, as in `scripts/lirf_lfpg_mechanisms.py`. I checked the
definition against the pooled OOF: it gives **6,152 rows, 16.69 % of lane-A SSE**. Of that, LFPG is 3.77 %, LIRF
2.87 % and LTFM 2.73 % `[inline]`, which matches the brief. Excluding only LIRF fills gives 6,443 rows and 17.8 %.

### 4.1 Measured on 2025-07-21: 124 rows `[L25 §late NM]`

| | EDDF | EDDM | EGLL | EHAM | LEBL | LEMD | LFPG | LIRF | LSZH | LTFM | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| n | 13 | 2 | 36 | 1 | 4 | 2 | 11 | 3 | 4 | 48 | 124 |
| seen at all | 13 | 2 | 36 | 1 | 4 | 2 | 11 | 3 | 4 | 0 | 76 |
| ground fix | 13 | 2 | 19 | 1 | 4 | 1 | 4 | 3 | 4 | 0 | 51 |
| **pushback** | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | **3** |

On the 3 flights with a pushback `[inline]`:

| flight | ADS-B − BLOCK | `AOBT_3` − BLOCK | y |
|---|---:|---:|---:|
| EDDF TUA476 | +35 s | +1,366 s | 2,397 s |
| EHAM DAL47 | −71 s | +1,277 s | 1,370 s |
| LSZH EJU52FK | +486 s | +1,433 s | 2,275 s |

On these three, OOF RMSE is 801 s and raw ADS-B RMSE 285 s `[L25 §target:push]`.

**Is 3 of 124 just the airport mix?** Applying each airport's all-flight pushback rate to the 124 rows predicts
4.9 detections `[X]`, against 3 observed, which is consistent at this n. The late rows are rarely seen because
**they sit at LTFM (48), EGLL (36) and LFPG (11), the airports where ADS-B has no surface picture**.

The flights with a ground fix but no pushback (48 of 51) do not help either. Their first fix is 1,409 s after
BLOCK at the median, and within ±5 min on 10 % `[L25 §late NM dep1]`. For the same aircraft, NM's stamp is a
similar distance late. `[X]` Both are observing the aircraft mid-taxi, not at the stand.

### 4.2 Projected onto the 6,152-row population `[X]`

Method: n and SSE share per airport `[inline]`, multiplied by that airport's matched-lane pushback rate. This
assumes the late rows are detected at the airport's average rate.

| | EDDF | EDDM | EGLL | EHAM | LEBL | LEMD | LFPG | LIRF | LSZH | LTFM | **total** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| late rows | 273 | 1,440 | 968 | 44 | 467 | 514 | 677 | 910 | 111 | 748 | **6,152** |
| lane-A SSE share | 1.24 % | 1.37 % | 2.05 % | 0.28 % | 0.95 % | 0.84 % | 3.77 % | 2.87 % | 0.59 % | 2.73 % | **16.69 %** |
| rows reached, 2025-07-21 rates | 26 | 799 | 7 | 21 | 124 | 32 | 0 | 13 | 14 | 0 | **≈1,037 (17 %)** |
| SSE reached, 2025-07-21 rates | 0.12 | 0.76 | 0.02 | 0.14 | 0.25 | 0.05 | 0 | 0.04 | 0.08 | 0 | **≈1.45 pp** |
| rows reached, 2026-07-20 rates | 9 | 869 | 560 | 22 | 155 | 88 | 2 | 0 | 20 | 0 | **≈1,725 (28 %)** |
| SSE reached, 2026-07-20 rates | 0.04 | 0.83 | 1.19 | 0.14 | 0.32 | 0.14 | 0.01 | 0 | 0.10 | 0 | **≈2.78 pp** |

SSE rows are in percentage points of lane-A SSE.

"Reached" is an upper bound on what the feature can fix, not a gain. **LFPG, LIRF and LTFM carry 9.4 pp of the
16.7 pp, and ADS-B reaches essentially none of it on either day.** `[X]` EDDM is the one airport where late rows
and coverage overlap.

## 5. 2026-07-20: the scored period

**Coverage** `[L26 §coverage matched]`:

| airport | n | join (seen) | ground fix | stand run | **pushback** | vs 2025-07-21 |
|---|---:|---:|---:|---:|---:|---|
| EDDF | 675 | 0.970 | 0.930 | 0.218 | 0.033 | down (0.097) |
| EDDM | 487 | 0.955 | 0.951 | 0.823 | **0.604** | same |
| EGLL | 657 | 0.994 | **0.994** | 0.833 | **0.578** | **up from 0.008** |
| EHAM | 715 | 0.985 | 0.985 | 0.866 | **0.508** | same |
| LEBL | 553 | 0.962 | 0.957 | 0.745 | 0.333 | up |
| LEMD | 639 | 0.978 | 0.837 | 0.382 | 0.171 | up (0.063) |
| LFPG | 677 | 0.978 | 0.774 | 0.027 | 0.003 | ≈0 |
| LIRF | 523 | 0.975 | **0.000** | 0.000 | 0.000 | **down from 0.943 ground** |
| LSZH | 414 | 0.988 | 0.988 | 0.606 | 0.179 | up |
| LTFM | 780 | 0.154 | 0.000 | 0.000 | 0.000 | airborne only |
| all | 6,120 | 0.872 | 0.726 | 0.431 | **0.233** | |

**Stability of the ADS-B-to-`AOBT_3` relation.** BLOCK is blank on this day, so this uses `AOBT_3 − t_push`, the
one statistic available in both years `[L26 §2026]`, `[inline]`:

| | EDDF | EDDM | EHAM | LEBL | LEMD | LSZH | pooled |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2025 median (n) | −131 (62) | +45 (279) | −134 (328) | +88 (144) | −139 (38) | +47 (51) | −33 (914), IQR 290 |
| 2026 median (n) | −123 (22) | +4 (294) | −90 (363) | +83 (184) | −104 (109) | +75 (74) | −66 (1,428), IQR 287 |

The relation holds within 45 s per airport, with the same pooled IQR. EGLL and LFPG have too few 2025 rows to
compare. The share of gated flights where `AOBT_3` is > 20 min later than the pushback is 0.4 % in 2025 and 0.2 %
in 2026. `[X]` This is consistent with the late-NM bucket being where ADS-B is blind.

## 6. Disk used

- **Peak:** one raw day at a time, at most 4.02 GB (2026-07-20). Each day's parts were deleted by `purge` right
  after `extract`, and before the next fetch. `fetch` refuses to start unless raw size + 3 GB fits.
- **Kept in the scratchpad** (`…/scratchpad/adsb/`, 33 MB in total):
  - two filtered fix files (15.0 and 17.3 MB)
  - two joined per-departure files (0.65 and 0.56 MB)
  - logs
- **Nothing is written to the repo** except this note and the script. `df` shows 13 GB available afterwards, as
  before.

## 7. GO / NO-GO

**GO-FOR-SOME-AIRPORTS.**

| airport | call | why |
|---|---|---|
| EDDM | **GO** | pushback on 55–60 % of departures on both days; MAE 58 s vs 207 s for `AOBT_3`; raw RMSE 85 vs OOF 154 |
| EHAM | **GO** | pushback on 48–51 %; MAE 26 s vs 278; ADS-B closer on 95 %; also sees EHAM's early-NM rows (`g < −1,200`, 755 OOF rows) |
| LEBL | GO (smaller) | 26–33 %; MAE 77 vs 225 |
| LSZH | GO (smaller) | 13–18 %; MAE 52 vs 190 |
| EGLL | CONDITIONAL | 0.8 % on the 2025 day, 58 % on the 2026 day. It depends on the feeder; census more days in both years first |
| EDDF | MARGINAL | 3–10 %; MAE 154 vs 190 |
| LEMD | MARGINAL | 6–17 %; median +118 s and a −5.5 k s failure |
| LFPG | NO-GO | 0–0.3 % pushback; ground fixes are sparse (median 2–4 per departure) |
| LIRF | NO-GO | 1.4 % in 2025 and **no ground fixes at all** on the 2026 day |
| LTFM | NO-GO | no ground fixes on either day; unjoinable in 2025 |

**What it is and what it is not** `[X]`:
- **It is:** a high-precision independent off-block observation on ~15–23 % of matched departures, concentrated at
  EDDM, EHAM, LEBL and LSZH, and possibly EGLL.
- **It is not:** the fix for the largest error bucket. The late-NM rows sit where ADS-B cannot see: 9.4 of their
  16.7 pp are at LFPG, LIRF and LTFM.
- **Why the upside is limited:** the reachable part of lane-A SSE is small.
  - ≈1.5–2.8 pp from the late-NM rows.
  - Some of EHAM's early-NM 1.8 %.
  - Ordinary rows at the four GO airports, whose OOF error is already moderate (RMSE 150–230).
- **A concrete number needs stage 2.** That means a gated per-airport feature, measured on the OOF protocol.
  Merry-quicksand's reported −7,722 board MSE from a 9-airport ADS-B stack is **not** contradicted by this. Their
  gain was spread over 161 k matched rows, most of them ordinary.

## 8. What I did not establish, and the next step if you go further

1. **Accuracy out of sample.** Re-run `analyse` on a second 2025 day with the rules frozen as committed.
   2025-07-13 has the most late-NM rows (240, 168 of them LIRF) but a poor airport mix. A day with many EDDM/LEBL
   late rows is the better test.
2. **Coverage census over the scored period.** Run `extract` on all 62 Jan/Jul 2026 days: ≈220 GB streamed, ≈3 h,
   ≈1 GB kept. The feeder dependence (EGLL, LIRF) makes a two-day read unreliable for any single airport.
3. **Training coverage.** The model trains on 2025. At airports where 2025 coverage differs from 2026 (EGLL), the
   feature would be rare in training and common at scoring. Measure 2025 coverage on a sample of days per month.
4. **January.** Both days read here are July. Winter de-icing adds a second long stop after pushback, and the
   15-min dwell guard would then drop those flights. The guard needs a January check.
5. **The unmatched lane** (no `AOBT_3`): 202 rows on the day. `FLIGHT_mvt` joined 130, with 12 pushbacks at −54 to
   +244 s from BLOCK, all within ±5 min `[inline]`. Small, but it is the lane with no NM clock at all.
6. **Rules.** The organiser's 2026-09-18 remark (*"if you can find open trajectory data with good ground coverage
   and extract off-block times, we are ok"*) is relayed second-hand. The official eligibility page's
   open-licence condition is met by ODbL 1.0 with the attribution above.
