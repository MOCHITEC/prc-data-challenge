# ADS-B pushback override: full ingest of the 62 scored days, and a submission candidate (2026-09-29)

Sibling note to `adsb-oos-validation.md` (the out-of-sample GO this acts on), `adsb-gain-estimate.md` and
`adsb-stage1-feasibility.md`. Nothing here was submitted. Nothing was written to `outputs/` or `src/`, and nothing
was fitted.

## 0. Pre-registration (written before any new 2026 day was ingested and before the EGLL check was run)

At the time this section was written, the scratch dir already held the five 2026 census days of the validation
(01-19, 01-28, 07-08, 07-20, 07-29). None of the other 57 scored days had been downloaded. The EGLL
leave-one-day-out check below had not been run. The only thing looked at for the 57 days was a HEAD of each
release: all 62 days have one, 2.20–4.28 GB each, 213.2 GB in total `[H]`.

### 0.1 Production settings, fixed here

| item | value |
|---|---|
| detection | stage-1 rules, `scripts/adsb_stage1_feasibility.py` unchanged (last changed in `ae98d1f`; identical at `9f29f62` and `b807fa8`). Pushback gate share ≥ 0.5, 15-min dwell guard, 25 m stand radius, 10-min silence. Scope is the stage-1 scope: departures with `MVT_TIME_UTC_mvt` in [D 02:00, D+1 00:00) UTC, read from archive day D. Departures between 00:00 and 02:00 UTC keep the base prediction, and their count is reported. |
| join | `CALLSIGN_flt` at `ADEP_mvt`. A row qualifies only if stage-1's `join == "cs"`. The stage-1 fallback to `FLIGHT_mvt` is not used: it produced 1 of 9,269 covered lane-A rows over the nine days on disk, and none at a GO airport. On the scored set, `CALLSIGN_flt` is present exactly when `AOBT_3_flt` is (lane A), so the override is lane A only, which is the population the gain was validated on. |
| guard | `y_a = MVT − t_push + b`. Use `y_a` unless `|base − y_a| > 900 s`, in which case keep the base. |
| airports | **EDDM, EHAM, LEBL, LSZH.** LEMD is out. EGLL is conditional (0.2). Every other airport keeps the base. |
| bias `b` | per airport, the median of (`t_push` − `BLOCK`) over the covered rows (lane A, `join == "cs"`, gated pushback) of **all four labelled days**: 2025-07-21, 2025-07-13, 2025-01-22, 2025-11-12. Pooled over rows, not averaged over days. |
| base prediction | `outputs/main_gap.parquet` (v14, board 279.5234), read only. |
| candidate value | `round(y_a)` cast to Int32, the template's dtype. Every other row is the base value, bit for bit. |
| missing days | if a day cannot be fetched, its flights keep the base, and the day is listed. |

What the override reads is `t_push` (ADS-B), `MVT_TIME_UTC_mvt`, `ADEP_mvt`, the join key, the base prediction and
`b`. `BLOCK_TIME_UTC_mvt` is read only on the four labelled 2025 days, to estimate `b` and to score. On the scored
rows it is blank (0 of 344,841 non-null).

### 0.2 The EGLL rule, fixed before the check was run

- **Check.** Leave one labelled day out, over the four labelled days. For each held-out day `h`:
  - `b_EGLL(h)` is the median (`t_push` − `BLOCK`) over EGLL's covered rows on the other three days.
  - Apply the override with the 900 s guard against the production OOF (`experiments/oof_main_gap-62dd7c6e88`) on
    EGLL's covered rows of day `h`.
- **Gain.** The mean over the pooled held-out rows of (e_model² − e_override²), in s² per covered row. A tripped row
  counts as 0.
- **CI.** A 95 % row bootstrap, B = 2,000, the validation's primary CI. The airport × hour × day cluster CI and the
  per-day gains are reported but do not decide.
- **Include EGLL iff the pooled gain is ≥ 5,000 s² per covered row and the row-bootstrap CI's lower bound is > 0.**
  Otherwise EGLL is out.
- If EGLL is in, its production bias is the four-day median of 0.1, like every other airport.
- The same leave-one-day-out figure is reported for the four GO airports, for information only. Their inclusion is
  already decided `[OOS §4]`.

---

**Scripts.**
- `scripts/adsb_ingest_all.py` (new) is the resumable driver. Its log is `…/scratchpad/adsb/ingest/ingest.log`, with
  per-day state in `…/ingest/state/<day>.json`.
- `scripts/adsb_override_candidate.py` (new) runs the EGLL check, the bias and the candidate. Its output is
  `…/scratchpad/adsb/override_candidate.log`.

```
POLARS_MAX_THREADS=2 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage .venv/bin/python scripts/adsb_ingest_all.py run
POLARS_MAX_THREADS=2 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage .venv/bin/python scripts/adsb_override_candidate.py
```

**Sources.**
- `[C Sx]` is section Sx of `override_candidate.log`.
- `[I]` is `ingest.log`.
- `[OOS]` is `adsb-oos-validation.md`.
- `[H]` is a HEAD request.
- `[X]` marks an inference.

## 1. The answer

1. **The candidate is written and not submitted.**
   - Path: `…/scratchpad/adsb/adsb_override_candidate.parquet`, 1.10 MB, sha256 `e8dc5b12…a3bd5`.
   - It has the template's schema (`MVT_ID_mvt` Float64, `TAXITIME_SEC_mvt` Int32), 344,841 rows, and the template's
     row order.
   - It is v14 (`outputs/main_gap.parquet`) plus the override. **63,533 rows (18.42 % of scored rows) differ from
     v14.** The 281,308 rows that do not change are bit-identical to v14 `[C S3, S4]`.
2. **EGLL is in.** The leave-one-day-out gain is **+21,718 s² per covered row** [12,989, 31,852] (row bootstrap;
   airport × hour × day cluster [12,556, 33,673]), on 196 rows with 0 guard trips. The pre-registered threshold was
   ≥ 5,000 with a lower bound > 0 `[C S0]`.
3. **All 62 scored days were ingested. None is missing** `[I]`.
4. **Coverage on the scored set is 18.5 %** of scored rows overridden. It is 15.9 % in January and 21.1 % in July,
   on the validation's definition (included airports ÷ lane A at ten airports). The validation measured 17.6 % and
   22.3 % with LEMD in and on 4 census days `[C S2]`, `[OOS §3]`.

## 2. EGLL: leave one labelled day out `[C S0]`

Covered = lane A, `join == "cs"`, gated pushback. `b` is estimated on the other three days, and the model is the
production OOF.

| held-out day | b from other three (n) | EGLL rows | gain / row | 95 % row CI |
|---|---:|---:|---:|---:|
| 2025-07-21 | 174.9 s (191) | 5 | +139,520 | [−310, +411,915] |
| 2025-07-13 | 175.8 s (184) | 12 | +26,368 | [7,106, 50,043] |
| 2025-01-22 | 174.9 s (131) | 65 | +31,997 | [19,391, 46,954] |
| 2025-11-12 | 162.6 s (82) | 114 | +10,201 | [2,788, 17,478] |
| **pooled** | | **196** | **+21,718** | **[12,989, 31,852]** |

- **Decision: INCLUDE.** Both conditions hold.
- On 2025-11-12, the day where the frozen bias gave −15,600 `[OOS §2]`, it is now +10,200, with the lower bound
  above 0. The whole of the validation's EGLL loss was the bias. `[X]`
- Pooled RMSE on these rows goes from 176.9 to 97.9 s.

**The same check for the four GO airports**, for information only: EDDM +19,903 (743 rows), EHAM +18,670 (1,120),
LEBL +16,573 (450), LSZH +26,463 (248). All four pooled: **+19,414** [17,281, 21,623]. All five: **+19,578**
[17,440, 21,956], with 5 trips in 2,757 rows. **Every held-out airport-day is positive.** The
leave-one-day-out gain is about the four-day in-sample figure (+19,624). So the move from the frozen
one-day bias to the four-day bias did not overfit `[C S0, S1]`.

## 3. Bias table (production) `[C S1]`

Median (`t_push` − BLOCK) over covered rows of the four labelled days, pooled over rows.

| ADEP | n | **b (s), used** | 2025-07-21 | 2025-07-13 | 2025-01-22 | 2025-11-12 |
|---|---:|---:|---:|---:|---:|---:|
| EDDM | 743 | **+47.50** | +50.3 | — (0 rows) | +43.5 | +46.3 |
| EGLL | 196 | **+173.74** | +152.7 | +133.3 | +170.0 | +176.9 |
| EHAM | 1,120 | **−4.43** | −3.5 | −3.3 | −4.8 | −6.7 |
| LEBL | 450 | **+40.23** | +45.1 | +46.7 | +38.5 | +32.6 |
| LSZH | 248 | **+34.19** | +30.5 | +29.3 | +34.6 | +46.5 |
| *not used:* EDDF | 340 | +39.5 | | | | |
| *not used:* LEMD | 110 | +126.2 | | | | |
| *not used:* LIRF | 36 | +116.0 | | | | |

Exact values used: EDDM 47.50, EGLL 173.74, EHAM −4.43, LEBL 40.23, LSZH 34.185.

## 4. Ingest `[I]`

- **Days.** 62 of 62. 57 were downloaded, 194.9 GB in total. The other five are the validation's census days
  (2026-01-19, 01-28, 07-08, 07-20, 07-29): their filtered fixes were reused, and their joined files were
  re-analysed. All five re-analysed files are `DataFrame.equals` to the validation's copies, so the pipeline is
  deterministic.
- **Wall clock.** 06:50–10:37 UTC, 3 h 47 min. About 45 min of that was lost to a crash (below). Days took
  about 2.5–3.5 min each, fetch plus extract.
- **Disk.**
  - One raw day at a time: parts were deleted after `extract` and before the next `fetch`. The largest day was
    4.28 GB (2026-07-23) `[H]`.
  - Free space before each fresh fetch was 12.50–13.43 GB. The minimum ever logged was 8.67 GB, at the restart with
    2026-07-17's 4.04 GB still on disk. No `raw + 3 GB > free` refusal was triggered.
  - Free at the end: 12.49 GB. Kept: `…/scratchpad/adsb/` is 1.1 GB, of which filtered fixes for 62 days are 976 MB.
    Joined files, logs and the candidate make up the rest.
- **Downloads.** One transient failure: 2026-07-11 part `.ab`, curl 56 "connection reset by peer" at 497 MB of
  1,635 MB. The part was resumed on the first retry. No proxy block and no 403/407.
- **Resume, tested three times, twice on purpose.**
  1. `kill -9` of the process group while 2026-01-01's last part was finishing, before the state recorded
     `fetched`. The restart found both parts at their HEAD sizes, downloaded nothing, and extracted.
  2. `kill -9` with 2026-01-02 `.aa` at 411.8 MB of 2,000 MB. The restart reported "still to fetch 2.610 GB",
     resumed the part with `curl -C -`, and finished it 20 s later. That day's extract (55,983 traces, 3,474 near an
     airport) is in line with its neighbours. `[X]` A corrupt resumed byte range would have broken the tar stream.
     The GitHub API (asset digests) is not attached to this session, so no checksum was compared.
  3. **A real crash.** Stage-1 `extract` on 2026-07-17 raised `ArrowInvalid` in `pa.concat_tables`.
     - Cause: the last 500k-row flush fell exactly on the last kept trace, which left an empty final chunk with
       all-null column types.
     - The driver died with it at 09:09. My wait loop missed the death: its `pgrep -f` pattern matched the loop's
       own command line. It was noticed at 09:54.
     - The restart went straight to re-extracting 07-17 from the raw parts still on disk (1,502,118 fixes), with no
       re-download.
- **One change to `scripts/adsb_stage1_feasibility.py`, made after the pre-registration.**
  - `extract` now skips an empty final chunk:
    `if out["t"] or not chunks: chunks.append(pa.table(out))`.
  - It changes nothing when the final chunk is non-empty, so the output of every other day, and of every day read
    before, is byte-for-byte what the old code produced. No detection rule changed.
- **Driver hardening, after the crash.** A failing `extract`/`analyse` is now recorded in the day's state, its parts
  are purged, and the run moves on (a second pass retries it). The driver writes a pid file for watchers.

## 5. The candidate on the scored set `[C S2, S3]`

**Per airport** (all 62 days; coverage = overridden ÷ lane-A departures; share of scored = overridden ÷ all scored
departures at the airport):

| ADEP | scored dep | lane A | pushbacks | guard trips | overridden | changed | coverage (lane A) | share of scored | airport-day range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EDDM | 25,966 | 25,728 | 15,913 | 47 (0.30 %) | 15,866 | 15,807 | **61.7 %** | 61.1 % | 53–69 % |
| EGLL | 39,840 | 39,592 | 16,826 | 66 (0.39 %) | 16,760 | 16,716 | **42.3 %** | 42.1 % | 14–71 % (Jan median 24 %, Jul 59 %) |
| EHAM | 38,182 | 36,976 | 16,189 | 70 (0.43 %) | 16,119 | 16,063 | **43.6 %** | 42.2 % | 28–51 % |
| LEBL | 30,080 | 29,638 | 9,403 | 9 (0.10 %) | 9,394 | 9,365 | **31.7 %** | 31.2 % | 18–59 % |
| LSZH | 23,150 | 22,491 | 5,613 | 16 (0.29 %) | 5,597 | 5,582 | **24.9 %** | 24.2 % | 16–40 % (Jan median 33 %, Jul 19 %) |
| EDDF, LEMD, LFPG, LIRF, LTFM | 187,623 | | 0 | 0 | 0 | 0 | — | 0 | |
| **all** | **344,841** | | **63,944** | **208 (0.33 %)** | **63,736** | **63,533** | | **18.48 %** | |

**Per month.** Overridden ÷ lane A at the ten airports (the validation's definition) is **15.9 %** in January
(23,822 / 150,266) and **21.1 %** in July (39,914 / 189,285). At the five airports alone it is 35.4 % and 45.8 %.

**No single-day feeder outage.** No airport has a zero day in 2026. The lowest airport-days are EGLL 14 %
(2026-01-01), LSZH 16 % and LEBL 18 %. Contrast EDDM's 0 % on 2025-07-13 `[OOS §2]`.

**Distribution of the change** (the 63,533 changed rows, candidate − v14, in s):
- |change|: median **91**, p90 **259**, p99 535, max **897**.
- Signed change: median −11, mean −7.7, p10 −193, p90 +182.
- Mean prediction over all scored rows: 1039.29 in v14, 1037.87 in the candidate.
- Per airport, median |change| / p90: EDDM 82 / 238, EGLL 120 / 311, EHAM 78 / 220, LEBL 87 / 245, LSZH 88 / 254.

**Why "overridden" ≠ "changed":** 203 overridden rows round to exactly the v14 value.

**Rows kept at v14 by scope.**
- 155 departures at the included airports have `MVT_TIME` between 00:00 and 02:00 UTC (EDDM 2, EHAM 31, LEBL 122).
  They are outside the stage-1 scope.
- Lane-B rows (no `CALLSIGN_flt`: 2,793 rows, 1.8 % of scored rows at the five airports) are also kept at v14.

**One implausible value, left as the rule gives it.**
- One overridden row is negative: LEBL, 2026-01-09, v14 708 s, candidate **−46 s**.
- Only 2 overridden rows are below 120 s and 5 below 240 s. On the labelled days no overridden row had `y_a` < 180 s,
  and the smallest label was 78 s.
- `[X]` This is a detection error that the 900 s guard does not catch. At worst it adds ≈ 754² / 344,841 ≈ 1.6 s²
  to the board MSE.
- I did not clip it: a floor was not pre-registered. Clipping to a floor (for example the v14 value when
  `y_a` < 120 s) is a one-line choice for whoever submits.

**Per day** (per airport: overridden / guard trips / coverage of lane A; the last column pools the five airports):

| day | EDDM | EGLL | EHAM | LEBL | LSZH | overridden | trips | cov. (lane A, 5 apts) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-01-01 | 207 / 0 / 65 % | 92 / 0 / 14 % | 200 / 1 / 33 % | 148 / 0 / 33 % | 104 / 0 / 33 % | 751 | 1 | 32.2 % |
| 2026-01-02 | 208 / 1 / 56 % | 105 / 1 / 16 % | 142 / 1 / 38 % | 145 / 0 / 29 % | 116 / 0 / 33 % | 716 | 3 | 32.1 % |
| 2026-01-03 | 232 / 0 / 66 % | 153 / 4 / 25 % | 96 / 11 / 34 % | 120 / 0 / 27 % | 112 / 0 / 33 % | 713 | 15 | 35.1 % |
| 2026-01-04 | 235 / 1 / 62 % | 119 / 2 / 20 % | 60 / 7 / 28 % | 120 / 0 / 25 % | 102 / 0 / 30 % | 636 | 10 | 31.7 % |
| 2026-01-05 | 236 / 2 / 63 % | 152 / 1 / 25 % | 36 / 5 / 29 % | 134 / 0 / 28 % | 115 / 0 / 34 % | 673 | 8 | 35.2 % |
| 2026-01-06 | 223 / 1 / 65 % | 170 / 0 / 28 % | 82 / 2 / 36 % | 118 / 1 / 30 % | 99 / 0 / 33 % | 692 | 4 | 37.0 % |
| 2026-01-07 | 238 / 2 / 64 % | 164 / 1 / 27 % | 50 / 1 / 28 % | 122 / 0 / 35 % | 94 / 0 / 34 % | 668 | 4 | 37.5 % |
| 2026-01-08 | 203 / 3 / 58 % | 144 / 2 / 24 % | 255 / 2 / 45 % | 109 / 0 / 29 % | 87 / 1 / 32 % | 798 | 8 | 36.5 % |
| 2026-01-09 | 207 / 0 / 58 % | 157 / 1 / 26 % | 182 / 4 / 41 % | 107 / 0 / 26 % | 116 / 0 / 38 % | 769 | 5 | 36.3 % |
| 2026-01-10 | 169 / 0 / 53 % | 170 / 1 / 28 % | 231 / 1 / 46 % | 92 / 0 / 29 % | 47 / 2 / 19 % | 709 | 4 | 35.8 % |
| 2026-01-11 | 219 / 1 / 60 % | 185 / 0 / 29 % | 272 / 1 / 47 % | 127 / 0 / 30 % | 93 / 0 / 28 % | 896 | 2 | 38.3 % |
| 2026-01-12 | 181 / 5 / 57 % | 162 / 1 / 26 % | 236 / 1 / 40 % | 103 / 0 / 26 % | 109 / 0 / 36 % | 791 | 7 | 35.3 % |
| 2026-01-13 | 199 / 1 / 63 % | 148 / 0 / 24 % | 242 / 1 / 45 % | 79 / 1 / 26 % | 100 / 0 / 38 % | 768 | 3 | 37.7 % |
| 2026-01-14 | 212 / 0 / 61 % | 132 / 0 / 21 % | 243 / 0 / 44 % | 82 / 0 / 26 % | 96 / 0 / 33 % | 765 | 0 | 35.8 % |
| 2026-01-15 | 207 / 0 / 58 % | 137 / 0 / 22 % | 229 / 0 / 39 % | 82 / 0 / 23 % | 102 / 1 / 32 % | 757 | 1 | 33.7 % |
| 2026-01-16 | 230 / 0 / 59 % | 164 / 0 / 26 % | 256 / 0 / 43 % | 116 / 0 / 28 % | 110 / 0 / 34 % | 876 | 0 | 37.2 % |
| 2026-01-17 | 218 / 0 / 69 % | 107 / 0 / 18 % | 226 / 0 / 43 % | 88 / 0 / 28 % | 104 / 1 / 36 % | 743 | 1 | 36.4 % |
| 2026-01-18 | 231 / 1 / 66 % | 107 / 1 / 17 % | 270 / 0 / 46 % | 112 / 0 / 26 % | 117 / 0 / 33 % | 837 | 2 | 35.6 % |
| 2026-01-19 | 217 / 0 / 61 % | 136 / 0 / 22 % | 223 / 1 / 38 % | 117 / 0 / 27 % | 111 / 0 / 29 % | 804 | 1 | 33.7 % |
| 2026-01-20 | 199 / 0 / 60 % | 160 / 1 / 26 % | 235 / 0 / 42 % | 100 / 1 / 31 % | 97 / 0 / 33 % | 791 | 2 | 37.5 % |
| 2026-01-21 | 230 / 0 / 67 % | 151 / 0 / 24 % | 240 / 1 / 43 % | 102 / 0 / 28 % | 89 / 0 / 29 % | 812 | 1 | 36.7 % |
| 2026-01-22 | 240 / 0 / 65 % | 137 / 0 / 21 % | 247 / 0 / 41 % | 141 / 1 / 34 % | 96 / 1 / 29 % | 861 | 2 | 36.5 % |
| 2026-01-23 | 204 / 0 / 53 % | 103 / 1 / 16 % | 215 / 0 / 35 % | 99 / 0 / 22 % | 100 / 0 / 25 % | 721 | 1 | 28.9 % |
| 2026-01-24 | 218 / 0 / 67 % | 168 / 0 / 28 % | 231 / 0 / 44 % | 94 / 0 / 30 % | 105 / 0 / 36 % | 816 | 0 | 39.7 % |
| 2026-01-25 | 217 / 1 / 63 % | 133 / 0 / 23 % | 245 / 6 / 44 % | 120 / 0 / 29 % | 112 / 0 / 35 % | 827 | 7 | 37.1 % |
| 2026-01-26 | 218 / 1 / 62 % | 187 / 0 / 31 % | 258 / 0 / 44 % | 108 / 0 / 26 % | 116 / 0 / 39 % | 887 | 1 | 39.2 % |
| 2026-01-27 | 203 / 0 / 63 % | 147 / 1 / 25 % | 214 / 1 / 39 % | 84 / 0 / 27 % | 99 / 0 / 35 % | 747 | 2 | 36.2 % |
| 2026-01-28 | 233 / 0 / 66 % | 144 / 2 / 23 % | 223 / 1 / 40 % | 60 / 0 / 18 % | 106 / 0 / 34 % | 766 | 3 | 35.3 % |
| 2026-01-29 | 188 / 2 / 53 % | 132 / 0 / 21 % | 189 / 1 / 36 % | 98 / 0 / 24 % | 101 / 0 / 33 % | 708 | 3 | 32.0 % |
| 2026-01-30 | 233 / 1 / 62 % | 123 / 1 / 19 % | 175 / 2 / 29 % | 129 / 0 / 28 % | 110 / 1 / 33 % | 770 | 5 | 31.7 % |
| 2026-01-31 | 220 / 0 / 67 % | 152 / 0 / 25 % | 170 / 0 / 33 % | 95 / 0 / 28 % | 117 / 0 / 40 % | 754 | 0 | 36.4 % |
| 2026-07-01 | 243 / 0 / 62 % | 380 / 1 / 59 % | 239 / 1 / 35 % | 130 / 0 / 24 % | 56 / 0 / 16 % | 1048 | 2 | 40.2 % |
| 2026-07-02 | 313 / 0 / 62 % | 410 / 0 / 62 % | 323 / 1 / 46 % | 199 / 0 / 34 % | 78 / 0 / 18 % | 1323 | 1 | 46.0 % |
| 2026-07-03 | 311 / 1 / 62 % | 386 / 2 / 58 % | 344 / 1 / 48 % | 206 / 1 / 34 % | 83 / 0 / 19 % | 1330 | 5 | 45.6 % |
| 2026-07-04 | 285 / 0 / 63 % | 399 / 1 / 62 % | 334 / 0 / 49 % | 225 / 0 / 40 % | 77 / 2 / 19 % | 1320 | 3 | 48.3 % |
| 2026-07-05 | 320 / 2 / 67 % | 384 / 0 / 59 % | 356 / 1 / 51 % | 319 / 1 / 57 % | 83 / 0 / 20 % | 1462 | 4 | 52.0 % |
| 2026-07-06 | 291 / 1 / 59 % | 385 / 0 / 58 % | 345 / 2 / 49 % | 347 / 1 / 59 % | 65 / 0 / 16 % | 1433 | 4 | 50.0 % |
| 2026-07-07 | 319 / 1 / 66 % | 396 / 0 / 60 % | 324 / 1 / 48 % | 313 / 0 / 56 % | 77 / 0 / 20 % | 1429 | 2 | 51.5 % |
| 2026-07-08 | 293 / 0 / 61 % | 366 / 0 / 55 % | 271 / 0 / 38 % | 277 / 0 / 49 % | 79 / 1 / 18 % | 1286 | 1 | 44.9 % |
| 2026-07-09 | 300 / 1 / 60 % | 383 / 1 / 57 % | 307 / 1 / 44 % | 323 / 0 / 56 % | 84 / 0 / 19 % | 1397 | 3 | 48.4 % |
| 2026-07-10 | 293 / 1 / 58 % | 408 / 0 / 61 % | 295 / 1 / 43 % | 295 / 0 / 51 % | 86 / 0 / 20 % | 1377 | 2 | 47.8 % |
| 2026-07-11 | 271 / 1 / 59 % | 406 / 2 / 64 % | 307 / 1 / 46 % | 181 / 0 / 34 % | 75 / 0 / 19 % | 1240 | 4 | 46.1 % |
| 2026-07-12 | 308 / 0 / 64 % | 422 / 3 / 64 % | 341 / 0 / 48 % | 169 / 0 / 30 % | 86 / 1 / 21 % | 1326 | 4 | 47.2 % |
| 2026-07-13 | 310 / 2 / 62 % | 384 / 3 / 58 % | 329 / 0 / 48 % | 167 / 0 / 29 % | 84 / 1 / 19 % | 1274 | 6 | 44.4 % |
| 2026-07-14 | 294 / 4 / 62 % | 393 / 1 / 60 % | 313 / 0 / 47 % | 127 / 0 / 23 % | 76 / 0 / 19 % | 1203 | 5 | 43.4 % |
| 2026-07-15 | 298 / 2 / 60 % | 359 / 6 / 55 % | 335 / 0 / 49 % | 255 / 0 / 46 % | 74 / 0 / 19 % | 1321 | 8 | 47.6 % |
| 2026-07-16 | 297 / 1 / 60 % | 375 / 2 / 56 % | 335 / 0 / 48 % | 142 / 0 / 26 % | 78 / 1 / 19 % | 1227 | 4 | 43.6 % |
| 2026-07-17 | 305 / 1 / 63 % | 429 / 0 / 65 % | 308 / 0 / 45 % | 172 / 0 / 30 % | 82 / 0 / 20 % | 1296 | 1 | 45.8 % |
| 2026-07-18 | 271 / 0 / 62 % | 439 / 2 / 70 % | 315 / 2 / 47 % | 133 / 0 / 25 % | 79 / 0 / 20 % | 1237 | 4 | 46.2 % |
| 2026-07-19 | 309 / 0 / 65 % | 466 / 3 / 70 % | 321 / 2 / 46 % | 164 / 0 / 30 % | 86 / 0 / 20 % | 1346 | 5 | 47.9 % |
| 2026-07-20 | 294 / 0 / 60 % | 379 / 1 / 58 % | 363 / 0 / 51 % | 182 / 2 / 33 % | 74 / 0 / 18 % | 1292 | 3 | 45.7 % |
| 2026-07-21 | 298 / 0 / 63 % | 381 / 2 / 57 % | 297 / 0 / 43 % | 157 / 0 / 28 % | 71 / 0 / 18 % | 1204 | 2 | 43.2 % |
| 2026-07-22 | 291 / 1 / 60 % | 352 / 2 / 53 % | 335 / 1 / 47 % | 153 / 0 / 27 % | 84 / 0 / 20 % | 1215 | 4 | 42.6 % |
| 2026-07-23 | 313 / 0 / 64 % | 376 / 1 / 56 % | 309 / 1 / 44 % | 136 / 0 / 23 % | 70 / 0 / 17 % | 1204 | 2 | 42.2 % |
| 2026-07-24 | 307 / 0 / 61 % | 387 / 2 / 58 % | 348 / 0 / 49 % | 132 / 0 / 22 % | 85 / 0 / 19 % | 1259 | 2 | 43.0 % |
| 2026-07-25 | 283 / 0 / 62 % | 437 / 1 / 68 % | 316 / 0 / 47 % | 114 / 0 / 22 % | 71 / 0 / 18 % | 1221 | 1 | 45.2 % |
| 2026-07-26 | 292 / 0 / 61 % | 470 / 1 / 71 % | 310 / 0 / 44 % | 143 / 0 / 26 % | 70 / 1 / 17 % | 1285 | 2 | 45.6 % |
| 2026-07-27 | 303 / 1 / 63 % | 403 / 0 / 61 % | 332 / 0 / 47 % | 103 / 0 / 18 % | 75 / 0 / 18 % | 1216 | 1 | 42.6 % |
| 2026-07-28 | 290 / 0 / 60 % | 364 / 2 / 55 % | 310 / 0 / 45 % | 170 / 0 / 31 % | 77 / 0 / 19 % | 1211 | 2 | 43.6 % |
| 2026-07-29 | 290 / 0 / 60 % | 404 / 1 / 61 % | 334 / 0 / 48 % | 205 / 0 / 37 % | 80 / 0 / 19 % | 1313 | 1 | 46.7 % |
| 2026-07-30 | 290 / 0 / 60 % | 391 / 1 / 58 % | 308 / 2 / 44 % | 258 / 0 / 45 % | 87 / 0 / 21 % | 1334 | 3 | 46.9 % |
| 2026-07-31 | 309 / 4 / 63 % | 405 / 4 / 60 % | 342 / 1 / 48 % | 146 / 0 / 26 % | 83 / 2 / 19 % | 1285 | 11 | 44.8 % |

The per-day CSVs are `…/scratchpad/adsb/adsb_override_perday.csv` and `adsb_override_perday_airport.csv`. The
per-row audit (every overridden row, with `t_push`, `b`, v14, `y_a` and the candidate) is `adsb_override_rows.parquet`.

## 6. Sanity checks `[C S4]` (all PASS)

- The schema equals the template's. There are 344,841 rows, the row order equals the template's, and there are no
  nulls.
- The changed rows (63,533) are a subset of the overridden rows (63,736). Re-reading the written file gives exactly
  63,533 rows that differ from `outputs/main_gap.parquet`.
- **No row outside EDDM / EGLL / EHAM / LEBL / LSZH changed.** That excludes LEMD, EDDF, LFPG, LIRF and LTFM.
- **Every changed row has a gated pushback, lane A, and a `join == "cs"` (`CALLSIGN_flt`) join.**
- **max |candidate − v14| = 897 s ≤ 900.** No guard-tripped row changed.
- No changed row has `MVT` between 00:00 and 02:00 UTC. No day was missing, so there was no missing day to protect.
- The unchanged rows are bit-identical to v14.
- **Independent recomputation.** Every changed value equals `round(MVT − t_push + b)`, with `MVT` read fresh from
  `data/raw/ranking.parquet`, not from the joined file.
- **BLOCK is not read by the override.** The 2026 frame is reduced to `MVT_ID_mvt, ADEP_mvt, t_mvt, join, AOBT_3_flt`
  plus the detection columns, and an assertion checks that no BLOCK column is present. BLOCK is blank on every
  scored row anyway (0 / 344,841).

## 7. What this candidate is, and is not

- **The base is v14** (`outputs/main_gap.parquet`, board 279.5234).
  - The team's current best is **v16** (279.259, recipe `main_flseq`), submitted from a parallel session on branch
    `claude/clever-bell-jz8lut`. Its file is not in this checkout.
  - So this candidate is **v14 + override**. A v16 + override needs that file:
    `scripts/adsb_override_candidate.py` with `outputs/main_gap.parquet` swapped for it. The guard compares against
    whichever base is used, so the file must be the base itself.
- **Expected effect, not a promise** `[X]`.
  - The per-row gain, out of sample, is about +19,500 s² on covered rows `[C S0]`. Times 63,736 rows ÷ 344,841,
    that is ≈ 3,600 s² of MSE, which would be a large drop from 279.5² ≈ 78,100.
  - The ≈10× board shrinkage seen on earlier changes is untested for this one `[OOS §6]`. So the board is the only
    measurement that counts. No board number is claimed here.
- **Not done.** No submission, nothing written to `outputs/` or `src/`, no fit. The version number and today's
  quota are for the submitter to check.
