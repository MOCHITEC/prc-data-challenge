# OSM stand-to-runway distance on top of v16 (main_flseq): REJECT on all four folds (2026-09-30)

The OSM feature (`osm_stand_rwy_km`, `docs/osm_stand_2026-09-28/`) passed its three pre-registered conditions on
`main_gap` (v14) with `folds: [forward]`. Its note asked for a full-fold measurement before production. This is that
measurement, on the current team-best model (v16 = `configs/main_flseq.yaml`). The candidate is
`configs/main_flseq_osm.yaml` = `main_flseq` + `fe_osm_dist: true`. Nothing else differs.

Nothing was submitted. The candidate file `outputs/main_flseq_osm.parquet` was built anyway, as requested.

## 1. Code line and reproduction of v16

- This branch merged `origin/claude/clever-bell-jz8lut` (v16, `1fdaca7`: `configs/main_flseq.yaml`,
  `prc.features_extra`) and `origin/claude/beautiful-davinci-ehugk4` (`f7b3c47`: `prc.osm`, `fe_osm_dist` and the
  other fold-time families). Conflicts were resolved keeping both sides (`464c539`, `2d14f9a`).
- The first `main_flseq` run on the merged code was OOM-killed (exit 137, anon RSS 13.9 GB) in its first fold.
  `prc.pipeline.main` now frees the raw frames after the feature build and drops the candidate columns the recipe
  does not include. `visible_numeric_columns` never gives those columns to the model and keeps the rest in frame
  order, so the design matrix is unchanged. A test checks this (`tests/test_feature_cache.py`). The cgroup's
  `memory.max_usage_in_bytes` was then 11.91 GB for the `main_flseq` run and 12.15 GB for the `main_flseq_osm` run.
- **Reproduction.** `main_flseq` rebuilt on this branch (`--work-dir data/work_flseq`) gives
  `outputs/main_flseq.parquet`. Compared with the v16 file from the bucket (`resilient-hedgehog_v16.parquet`):
  **max abs diff 0 s, 0 of 344,841 rows differ**, same ids in the same order. The CV also matches v16's record
  exactly: full 272.96, jan 342.16, jul 295.32, jul_fwd 308.29, forward 217.20 (`main_flseq-8615e6a194`). The id differs from v16's `main_flseq-70b25859ec`
only because this branch's TrainConfig has more fields (for example `es_holdout_encodings`), and those fields go into the hash.

## 2. OSM data used

Fetched 2026-09-29 23:42 to 2026-09-30 00:14 UTC with `scripts/fetch_osm_stands.py`. EDDF, EDDM, EGLL, EHAM, LIRF
and LTFM came through Overpass. LEBL, LEMD, LFPG and LSZH came through the OSM `map` API, because Overpass reset the
connection or answered 504. Stand refs per airport: EDDF 284, EDDM 195, EGLL 267, EHAM 276, LEBL 186, LEMD 406,
LFPG 484, LIRF 147, LSZH 254, LTFM 454. The feature is non-null on 90.7% of labelled rows and 91.5% of scored rows.
EDDM is lowest (57.4% / 57.6%) and EDDF next (70.4% / 72.9%). The earlier session reported 90.5% OSM coverage on
its unseen-stand rows and 50.6% at EDDM. Licences and attribution are in `docs/external_data.md`.

## 3. Folds

| fold | main_flseq | main_flseq_osm | diff | matched A → B | unmatched A → B |
|---|---:|---:|---:|---|---|
| jan | 342.16 | 343.14 | +0.98 | 201.67 → 201.58 | 2867.76 → 2880.94 |
| jul | 295.32 | 297.26 | +1.94 | 241.90 → 241.90 | 1203.41 → 1226.49 |
| jul_fwd | 308.29 | 308.29 | 0.00 | 250.57 → 250.27 | 1274.69 → 1277.52 |
| forward | 217.20 | 217.38 | +0.18 | 197.14 → 196.98 | 972.67 → 980.35 |
| **overall** (pooled jan, jul, forward) | **272.96** | **273.91** | **+0.95** | 211.67 → 211.58 | 1558.4 → 1572.7 |

## 4. `prc.compare` (main_flseq-8615e6a194 → main_flseq_osm-bb204cd922, 672,428 rows)

| dropped | A | B | diff | CI95 | P(B better) |
|---:|---:|---:|---:|---|---:|
| 0 | 272.96 | 273.91 | **+0.95** | **[+0.28, +1.60]** | 0.002 |
| 20 | 229.21 | 229.43 | +0.22 | [−0.20, +0.65] | 0.166 |
| 200 | 215.82 | 215.70 | **−0.12** | **[−0.31, +0.12]** | 0.816 |
| 2000 | 192.80 | 192.51 | −0.29 | [−0.42, −0.17] | 1.000 |

Month-level bootstrap (4 months): +0.95 s [+0.18, +1.71], P(B better) = 0.022.

**VERDICT: REJECT.** The verdict line reads "B is worse on all rows: +0.95 s [+0.28, +1.60]". The full CI excludes
zero on the worse side. The drop-200 CI crosses zero.

### Per airport, all rows (B − A; ordered by share of squared disagreement)

| airport | n | A | B | diff | CI95 | gap share |
|---|---:|---:|---:|---:|---|---:|
| LIRF | 49,924 | 517.39 | 527.03 | **+9.65** | [+5.29, +14.56] | 32.5% |
| LTFM | 91,338 | 233.65 | 232.70 | −0.95 | [−1.76, −0.37] | 13.9% |
| LFPG | 78,036 | 439.39 | 439.14 | −0.25 | [−0.84, +0.08] | 12.9% |
| EGLL | 79,633 | 239.04 | 238.69 | −0.35 | [−0.86, +0.20] | 9.7% |
| EHAM | 79,639 | 171.07 | 170.49 | −0.58 | [−1.11, −0.09] | 7.1% |
| EDDF | 72,276 | 178.76 | 178.61 | −0.15 | [−0.61, +0.29] | 5.4% |
| LEMD | 70,732 | 171.63 | 170.99 | −0.64 | [−0.96, −0.35] | 5.2% |
| EDDM | 51,688 | 185.69 | 185.36 | −0.34 | [−1.11, +0.30] | 5.0% |
| LEBL | 56,153 | 197.33 | 196.63 | −0.70 | [−1.84, −0.04] | 4.3% |
| LSZH | 43,009 | 183.06 | 181.40 | −1.66 | [−3.46, −0.36] | 3.9% |

### Per airport, matched lane only

| airport | n | A | B | diff | CI95 |
|---|---:|---:|---:|---:|---|
| LTFM | 90,041 | 227.11 | 226.54 | −0.57 | [−0.88, −0.21] |
| LFPG | 76,471 | 237.04 | 236.61 | −0.42 | [−0.76, −0.07] |
| LIRF | 49,387 | 344.57 | 344.78 | +0.21 | [−0.49, +0.89] |
| EGLL | 78,986 | 221.57 | 221.60 | +0.04 | [−0.28, +0.41] |
| EHAM | 78,286 | 160.70 | 160.69 | −0.01 | [−0.26, +0.27] |
| LEMD | 70,316 | 170.54 | 170.13 | −0.41 | [−0.69, −0.14] |
| EDDF | 71,489 | 174.37 | 174.82 | +0.45 | [+0.16, +0.75] |
| EDDM | 51,286 | 174.79 | 174.87 | +0.09 | [−0.42, +0.54] |
| LEBL | 55,501 | 193.98 | 194.07 | +0.10 | [−0.13, +0.32] |
| LSZH | 42,287 | 166.72 | 166.71 | −0.01 | [−0.42, +0.37] |

Matched lane pooled: −0.09 [−0.21, +0.04]. All rows except LIRF: −0.51 [−0.75, −0.30].

### Unmatched lane (from `prc.compare`'s tail report)

Unmatched pooled: +14.28 [+5.39, +26.98]. **LIRF unmatched (537 rows): 3737.3 → 3858.8, +121.57 [+71.43, +173.15],
62.1% of the unmatched disagreement.** Every other airport's unmatched difference is negative, from −0.22 (LFPG) to
−36.54 (LEBL). Of the 200 rows the K=200 drop removes, 150 are unmatched. 136 are LIRF rows (106 unmatched, 30 matched).

## 5. What the numbers say

- The loss is one cell: LIRF's unmatched lane (537 rows, +121.6 s there). It is enough to make the whole-fold
  difference +0.95 s with a CI that excludes zero. Without LIRF the feature is −0.51 [−0.75, −0.30] on all rows.
  On the matched lane it is −0.09 [−0.21, +0.04], which is no difference.
- The forward-fold measurement on `main_gap` (base → dist −0.02 [−0.73, +0.89] at K=0) did not show the LIRF
  loss. This run's forward fold is also small (+0.18). The loss is concentrated in jan (+0.98) and jul (+1.94),
  the months the forward fold does not validate.
- Under the decision rule (`prc.compare.verdict`, `docs/features_log.md`), this is REJECT. The candidate file exists
  only because it was asked for. It is not a production candidate as measured.

## 6. The candidate file

`outputs/main_flseq_osm.parquet` (full fit on all labelled months, `fold_feature_cols: ['osm_stand_rwy_km']`,
model A 1477 rounds, model B 108 features / 37 rounds). Validation against `data/raw/submitting.parquet`:
344,841 rows. Schema `{MVT_ID_mvt: Float64, TAXITIME_SEC_mvt: Int32}` equals the template. Ids and order equal the
template. 0 nulls, 0 negatives, min 0, max 113,713. Against `outputs/main_flseq.parquet` (= v16): 334,408 rows
differ, max abs diff 910 s, RMS difference 26.05 s. It has no ADS-B override applied. v18 is v16 + the override, and
the same override would be `scripts/adsb_override_candidate.py --base outputs/main_flseq_osm.parquet --out ...`.

Logs: `logs/main_flseq.log`, `logs/main_flseq_osm.log`, `logs/compare_osm.txt`,
`logs/compare_osm_per_airport.txt` (all git-ignored). Records are in `experiments/experiments.jsonl` and
`experiments/pipeline.jsonl`.

## 7. Follow-up: OSM distance with LIRF nulled (`main_flseq_osm_nolirf`), POST HOC, REJECT (2026-09-30)

**This variant is post hoc.** LIRF was chosen as the airport to exclude *after* section 4 showed the loss at LIRF.
It was not pre-registered. Any gain it showed would be partly selected on the same OOF rows it is scored on, so
it could not be read at face value. It shows no gain, so the question does not come up.

- **Change.** New `TrainConfig.fe_osm_null_airports` (default `[]`). After `osm_stand_rwy_km` is computed, it is
  set to null on rows whose `ADEP_mvt` is in the list (`prc.train.null_osm_at_airports`). It applies to the
  training and target frames of every fold and to the full fit. The field is in `_HASH_OPTIONAL` at `[]`, so the
  ids of existing configs do not change. `main_flseq` still hashes to `8615e6a194` and `main_flseq_osm` to
  `bb204cd922` (tested). Setting the field without `fe_osm_dist` raises. `configs/main_flseq_osm_nolirf.yaml` =
  `main_flseq_osm` + `fe_osm_null_airports: [LIRF]`.
- **Check.** On `feat_train` the feature is non-null at LIRF on 99.0% of rows with the option off and 0.0% with
  it on. Every other airport's column is identical.
- **Run.** `main_flseq_osm_nolirf-926741d779` took 3,627 s. The cgroup's `memory.max_usage_in_bytes` afterwards
  was 12.46 GB.

### Folds

| fold | main_flseq | main_flseq_osm | main_flseq_osm_nolirf | nolirf − flseq | matched (nolirf) | unmatched (nolirf) |
|---|---:|---:|---:|---:|---:|---:|
| jan | 342.16 | 343.14 | 342.54 | +0.38 | 201.52 | 2873.79 |
| jul | 295.32 | 297.26 | 297.23 | +1.91 | 241.73 | 1227.62 |
| jul_fwd | 308.29 | 308.29 | 308.66 | +0.37 | 250.24 | 1282.18 |
| forward | 217.20 | 217.38 | 217.35 | +0.15 | 196.93 | 980.73 |
| **overall** | **272.96** | **273.91** | **273.72** | **+0.76** | | |

### `prc.compare` (main_flseq-8615e6a194 → main_flseq_osm_nolirf-926741d779, 672,428 rows)

| dropped | A | B | diff | CI95 | P(B better) |
|---:|---:|---:|---:|---|---:|
| 0 | 272.96 | 273.72 | **+0.76** | **[+0.16, +1.38]** | 0.006 |
| 20 | 229.21 | 229.28 | +0.06 | [−0.31, +0.51] | 0.386 |
| 200 | 215.84 | 215.66 | **−0.19** | **[−0.39, +0.03]** | 0.942 |
| 2000 | 192.81 | 192.52 | −0.29 | [−0.39, −0.18] | 1.000 |

Month-level bootstrap (4 months): +0.76 s [−0.06, +1.55], P(B better) = 0.054.

**VERDICT: REJECT.** The verdict line reads "B is worse on all rows: +0.76 s [+0.16, +1.38]".

### Per airport (B − A, B = nolirf; ordered by share of squared disagreement)

| airport | all rows n | all rows diff | CI95 | matched diff | CI95 | unmatched n | unmatched diff | CI95 |
|---|---:|---:|---|---:|---|---:|---:|---|
| LIRF | 49,924 | **+8.30** | [+4.27, +13.13] | −0.28 | [−0.80, +0.33] | 537 | **+108.49** | [+59.96, +163.87] |
| LTFM | 91,338 | −0.94 | [−1.76, −0.34] | −0.58 | [−0.90, −0.29] | 1,297 | −12.43 | [−39.37, +3.98] |
| LFPG | 78,036 | −0.15 | [−0.74, +0.18] | −0.33 | [−0.70, +0.03] | 1,565 | +0.18 | [−7.22, +2.52] |
| EGLL | 79,633 | −0.33 | [−0.86, +0.15] | +0.06 | [−0.28, +0.42] | 647 | −11.39 | [−22.60, −0.04] |
| EHAM | 79,639 | −0.75 | [−1.23, −0.35] | −0.15 | [−0.50, +0.12] | 1,353 | −12.98 | [−24.50, −6.03] |
| LEMD | 70,732 | −0.80 | [−1.10, −0.52] | −0.57 | [−0.83, −0.30] | 416 | −22.74 | [−31.30, −13.69] |
| EDDM | 51,688 | −0.75 | [−1.54, +0.01] | −0.12 | [−0.64, +0.31] | 402 | −21.01 | [−45.74, −2.96] |
| EDDF | 72,276 | +0.04 | [−0.29, +0.42] | +0.38 | [+0.09, +0.68] | 787 | −12.90 | [−19.96, −7.06] |
| LEBL | 56,153 | −0.78 | [−2.03, −0.00] | +0.05 | [−0.20, +0.29] | 652 | −37.95 | [−83.24, −5.43] |
| LSZH | 43,009 | −1.31 | [−2.87, −0.05] | +0.13 | [−0.24, +0.49] | 722 | −26.08 | [−58.88, −5.53] |

Pooled: matched lane −0.18 [−0.30, −0.06]. Unmatched lane +12.54 [+4.20, +24.80]. All rows except LIRF −0.51
[−0.76, −0.29]. LIRF unmatched is 61.3% of the unmatched disagreement. The K=200 drop removes 151 unmatched and
49 matched rows.

Directly against `main_flseq_osm` (`logs/compare_osm_vs_nolirf.txt`): −0.19 s [−0.47, +0.02] at K=0 and −0.05
[−0.16, +0.09] at K=200, which is no difference. LIRF unmatched is 3858.8 → 3845.8 (−13.09 [−31.26, +3.81]).

### What the numbers say

- Nulling the feature at LIRF does **not** remove the LIRF loss. LIRF's unmatched lane is still +108.5 s against
  v16, compared with +121.6 s with the feature on at LIRF. So the loss does not come from LIRF's own OSM
  distances. It comes from what the column does to the models through the other nine airports' rows: the shared
  trees and the stopping points. LIRF's 537 unmatched rows carry large errors (RMSE about 3,700 s), so a small
  change in the fit moves them a lot. This matches section 5 and the note in `main_flseq`'s config that a
  signal-free column moves this model by about ±0.5 s. It is not a test of the noise-column hypothesis, which
  would need its own control arm.
- The matched lane improves slightly (−0.18 [−0.30, −0.06]; it was −0.09 [−0.21, +0.04] in section 4). The
  whole-fold result is still decided by the LIRF unmatched cell.
- Under the rule this is REJECT. The post-hoc exclusion does not rescue the feature.

### The candidate file

`outputs/main_flseq_osm_nolirf.parquet` (full fit, `fold_feature_cols: ['osm_stand_rwy_km']`, model A 1212 rounds,
model B 108 features / 45 rounds), built as requested whatever the verdict. Validation against
`data/raw/submitting.parquet`: 344,841 rows. Schema `{MVT_ID_mvt: Float64, TAXITIME_SEC_mvt: Int32}` equals the
template. Ids and order equal the template. 0 nulls, 0 negatives, min 0, max 113,966.
`python -m prc.validate_submission` prints OK, with only the usual >10,800 s max warning. Against
`outputs/main_flseq.parquet` (v16): 334,299 rows differ, max abs diff 1,137 s, RMS difference 26.54 s. Against
`outputs/main_flseq_osm.parquet`: 333,382 rows differ, max 1,036 s, RMS 24.78 s. It has no ADS-B override applied.
Nothing was submitted.

Logs (git-ignored): `logs/main_flseq_osm_nolirf.log`, `logs/compare_osm_nolirf.txt`,
`logs/compare_osm_nolirf_per_airport.txt`, `logs/compare_osm_vs_nolirf.txt`.
