# ADS-B ingest of the labelled days of January and July 2025 (2026-09-30)

This note sits alongside `adsb-full-ingest-candidate.md` (the 62 scored 2026 days) and `adsb-oos-validation.md`.
The aim is labelled ADS-B coverage of the kind merry-quicksand trained on (Jan and Jul 2025, calendar-day folds,
see `public-code-deep-dive-2026-09-30.md`), so that an ADS-B model can be trained on labelled rows later.
Nothing was fitted or submitted. Nothing was written to `src/`, `configs/`, `outputs/` or `data/`.

## 1. Facts established before the ingest

**Labelled data in `data/raw/`.** There are 12 monthly training files, `training_2025-MM-01_…parquet`, covering
**2025-01-01 to 2025-12-31**, every month of 2025. `BLOCK_TIME_UTC_mvt` is non-null on every departure row:
153,706 departures in Jan and 190,713 in Jul. `ranking.parquet` covers the 62 scored days of 2026-01 and 2026-07.
It has 344,841 departures and 0 non-null `BLOCK_TIME_UTC_mvt` on them, so it is unlabelled.
`submitting.parquet` is the template (`MVT_ID_mvt`, `TAXITIME_SEC_mvt`).

**Archive availability (adsb.lol `globe_history_2025`).** The existing scripts list a day by sending HEAD requests
to `releases/download/<tag>/<tag>.tar.aa, .ab, …` (`s1.sizes`).
- All 62 days have a `v2025.MM.DD-planes-readsb-prod-0` tag (checked with `git ls-remote --tags`).
- 57 days are split into 2 parts (`.tar.aa` + `.tar.ab`).
- **Five January days are one unsplit `<tag>.tar`** (< 2 GB): 01-01, 01-11, 01-18, 01-19 and 01-21. The part
  listing in `s1.sizes` reports these five as absent, because it only tries `.tar.aa`, `.tar.ab`, and so on.
- Size per day ranges from 1.68 to 3.85 GB. The totals are 66.40 GB for Jan, 108.57 GB for Jul and **174.97 GB**
  overall.
- The GitHub API and `expanded_assets` pages return 403 through the proxy, so asset names were found with HEAD
  probes of `.tar`, `.tar.gz` and `.zip`.

## 2. Method

- **Driver.** `scripts/adsb_ingest_2025.py` (new) is a thin wrapper around the unchanged `adsb_ingest_all.do_day`
  and the unchanged `adsb_stage1_feasibility.py`. The per-day steps are fetch → extract → purge → analyse. It shares
  the state in `…/scratchpad/adsb/ingest/state/<D>.json`, the lock and `ingest.log`. It is resumable. Days run one
  at a time, in order, with one raw day on disk at a time; the raw parts are deleted after extraction.
- **What the wrapper changes, and only this.**
  - The 2025 release repository: `fetch_day` hard-codes `globe_history_2026`. `s1.sizes` and `s1.part_url` are
    wrapped so that v2025 tags use `globe_history_2025`.
  - The five unsplit days: `<tag>.tar` is saved as `<tag>.tar.aa`, so the extractor streams it unchanged.
  - A memory gate: before each day, it waits until the cgroup's `total_rss` is at or below 9,000 MB.
  - A per-day status line in `…/ingest/status_2025.jsonl`.
- **Filters.** These are exactly the stage-1 filters.
  - **All 10 challenge airports** (EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM).
  - Fixes are kept within 8 km of the ARP, on the ground or at ≤ 4,000 ft baro.
  - `analyse` joins to the training departures with `MVT_TIME_UTC_mvt` in [D 02:00, D+1 00:00) UTC, the
    stage-1 scope. It writes `<tag>_joined.parquet`, which includes BLOCK, AOBT_3, `t_move` and `t_move_lo`, the
    pushback signature, and `e_to`.
- **Reused days.** 2025-01-22, 2025-07-13 and 2025-07-21 already had filtered fixes on disk from the validation
  run. `do_day` reused them without downloading again (8.86 GB streamed then) and only re-ran `analyse`. The new
  `2025-01-22_joined` is identical to the validation copy (`DataFrame.equals` = True). The three pre-run joined
  files were backed up to `…/ingest/validation_joined_backup_2025/`.

```
POLARS_MAX_THREADS=2 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage \
  .venv/bin/python scripts/adsb_ingest_2025.py run     # resumable; `status` / `table` to inspect
```

## 3. Result

| | Jan 2025 | Jul 2025 | total |
|---|---|---|---|
| days analysed / failed | 31 / 0 | 31 / 0 | **62 / 0** |
| bytes streamed this run | 64.38 GB | 101.73 GB | **166.11 GB** (+ 8.86 GB earlier for the 3 reused days) |
| fixes kept | 15,863,496 | 37,000,914 | 52,864,410 |
| filtered fixes parquet | 203.4 MB | 438.4 MB | 641.8 MB |
| joined departures parquet (rows) | 15.6 MB (152,952) | 19.6 MB (189,690) | 35.2 MB (342,642) |
| wall time in steps | 50.8 min | 90.5 min | 141 min, plus about 2.8 h of memory-gate waits (run 05:09–10:19 UTC) |

- **Outputs.** The 124 files total 0.68 GB, in `/tmp/claude-0/-home-user-prc-data-challenge/ca703460-62c0-5c84-91fe-d76c3d0e1e20/scratchpad/adsb/`:
  - `v2025.MM.DD-planes-readsb-prod-0.parquet` holds the fixes;
  - `v2025.MM.DD-planes-readsb-prod-0_joined.parquet` holds the per-departure join.

  These are scratch files, not committed.
- **Excluded departures.** There are 1,777 labelled departures with MVT before 02:00 UTC (344,419 labelled minus
  342,642 joined). They are outside the stage-1 scope and are not in the joined files.
- **Download retries.** One transient HTTP 500 happened on 2025-01-16 `.tar.aa`, and the first retry succeeded.
- **Memory.** The peak RSS summed over the ingest process tree (it over-counts pages shared by forked workers) was
  ≤ 1.1 GB on every July day. On 5 January days it was 1.55–2.59 GB, with the maximum on 2025-01-16. The spike is
  one stage-1 extract worker parsing a large trace file: a measured PSS of 1.46 GB in total, 1.21 GB in one worker,
  on 2025-01-13. The spikes are transient. The cgroup's `total_rss` was about 7.4 GB when those days started.

**Coverage over the 62 joined files.** Shares are of all departures in scope. `stand` = a stand episode was found
(`t_move`). `push` = the gated pushback estimator `t_push` of stage 1.

| ADEP | n | join_cs | seen | ground | stand | push |
|---|---|---|---|---|---|---|
| EDDF | 36,830 | 0.969 | 0.970 | 0.958 | 0.459 | 0.087 |
| EDDM | 27,090 | 0.979 | 0.988 | 0.632 | 0.526 | 0.360 |
| EGLL | 40,209 | 0.980 | 0.991 | 0.609 | 0.190 | 0.062 |
| EHAM | 40,926 | 0.969 | 0.970 | 0.968 | 0.797 | 0.365 |
| LEBL | 28,883 | 0.969 | 0.986 | 0.973 | 0.508 | 0.236 |
| LEMD | 35,096 | 0.837 | 0.845 | 0.263 | 0.102 | 0.035 |
| LFPG | 39,314 | 0.962 | 0.984 | 0.178 | 0.005 | 0.000 |
| LIRF | 26,479 | 0.945 | 0.957 | 0.551 | 0.324 | 0.014 |
| LSZH | 22,345 | 0.962 | 0.974 | 0.955 | 0.696 | 0.171 |
| LTFM | 45,470 | 0.001 | 0.001 | 0.000 | 0.000 | 0.000 |
| **all** | 342,642 | 0.826 | 0.835 | 0.574 | 0.333 | 0.124 |

The pooled figures are Jan 0.316 stand / 0.130 push and Jul 0.346 stand / 0.120 push. At LTFM, `CALLSIGN_flt` is
present on 98.6 % of rows. The archive holds only 22,290 LTFM fixes across all 62 days, out of 52.9 M, so almost
no LTFM departure is seen.

## 4. Per-day status

`peak RSS` is the peak RSS summed over the ingest process tree, sampled every 3 s. The source is
`…/ingest/status_2025.jsonl`, and `scripts/adsb_ingest_2025.py table` regenerates the table. "reused" = filtered
fixes from the validation run, not downloaded again.

| day | result | parts | GB streamed | fixes kept | fixes MB | departures joined | elapsed s | peak RSS MB |
|---|---|---|---|---|---|---|---|---|
| 2025-01-01 | done | 1 (1 tar) | 1.683 | 447466 | 5.9 | 4774 | 78 | 466 |
| 2025-01-02 | done | 2 | 2.171 | 505318 | 6.6 | 5336 | 101 | 585 |
| 2025-01-03 | done | 2 | 2.324 | 545851 | 7.0 | 5474 | 100 | 645 |
| 2025-01-04 | done | 2 | 2.187 | 492530 | 6.5 | 5070 | 95 | 527 |
| 2025-01-05 | done | 2 | 2.122 | 595359 | 7.3 | 5156 | 101 | 632 |
| 2025-01-06 | done | 2 | 2.054 | 500347 | 6.5 | 5281 | 90 | 532 |
| 2025-01-07 | done | 2 | 2.100 | 458847 | 6.1 | 4857 | 99 | 1836 |
| 2025-01-08 | done | 2 | 2.104 | 455713 | 6.0 | 4818 | 100 | 542 |
| 2025-01-09 | done | 2 | 2.108 | 457894 | 6.1 | 4905 | 111 | 1809 |
| 2025-01-10 | done | 2 | 2.077 | 483870 | 6.3 | 5127 | 94 | 545 |
| 2025-01-11 | done | 1 (1 tar) | 1.877 | 422064 | 5.7 | 4626 | 86 | 528 |
| 2025-01-12 | done | 2 | 2.090 | 510736 | 6.6 | 5023 | 99 | 719 |
| 2025-01-13 | done | 2 | 2.197 | 527133 | 6.8 | 4887 | 102 | 1555 |
| 2025-01-14 | done | 2 | 2.105 | 502980 | 6.5 | 4595 | 91 | 546 |
| 2025-01-15 | done | 2 | 2.234 | 567336 | 7.0 | 4656 | 112 | 1554 |
| 2025-01-16 | done | 2 | 2.234 | 497504 | 6.4 | 4836 | 159 | 2586 |
| 2025-01-17 | done | 2 | 2.445 | 530892 | 6.8 | 5117 | 102 | 652 |
| 2025-01-18 | done | 1 (1 tar) | 1.995 | 498213 | 6.4 | 4583 | 99 | 547 |
| 2025-01-19 | done | 1 (1 tar) | 1.889 | 541193 | 6.9 | 5003 | 87 | 655 |
| 2025-01-20 | done | 2 | 2.097 | 549783 | 6.9 | 5033 | 97 | 652 |
| 2025-01-21 | done | 1 (1 tar) | 1.961 | 503563 | 6.4 | 4561 | 86 | 546 |
| 2025-01-22 | done | 0 | reused | 543337 | 6.8 | 4873 | 5 | 410 |
| 2025-01-23 | done | 2 | 2.203 | 540793 | 6.8 | 4989 | 98 | 647 |
| 2025-01-24 | done | 2 | 2.427 | 549220 | 6.9 | 5179 | 109 | 651 |
| 2025-01-25 | done | 2 | 2.192 | 471534 | 6.1 | 4668 | 98 | 535 |
| 2025-01-26 | done | 2 | 2.114 | 530158 | 6.7 | 5057 | 100 | 636 |
| 2025-01-27 | done | 2 | 2.235 | 540039 | 6.8 | 4980 | 106 | 638 |
| 2025-01-28 | done | 2 | 2.202 | 510906 | 6.5 | 4596 | 99 | 552 |
| 2025-01-29 | done | 2 | 2.224 | 530052 | 6.7 | 4757 | 111 | 639 |
| 2025-01-30 | done | 2 | 2.408 | 503642 | 6.4 | 4921 | 112 | 547 |
| 2025-01-31 | done | 2 | 2.325 | 549223 | 7.0 | 5214 | 119 | 654 |
| 2025-07-01 | done | 2 | 3.423 | 1208858 | 14.5 | 6045 | 189 | 757 |
| 2025-07-02 | done | 2 | 3.538 | 1187856 | 14.2 | 6100 | 204 | 799 |
| 2025-07-03 | done | 2 | 3.579 | 1094439 | 12.8 | 5905 | 209 | 768 |
| 2025-07-04 | done | 2 | 3.147 | 1115114 | 13.2 | 5926 | 184 | 813 |
| 2025-07-05 | done | 2 | 3.009 | 1075920 | 12.5 | 5934 | 153 | 1023 |
| 2025-07-06 | done | 2 | 3.231 | 1098229 | 12.9 | 6098 | 155 | 762 |
| 2025-07-07 | done | 2 | 3.408 | 1191381 | 14.1 | 6196 | 168 | 994 |
| 2025-07-08 | done | 2 | 3.573 | 1174137 | 13.9 | 6084 | 168 | 771 |
| 2025-07-09 | done | 2 | 3.555 | 1181579 | 14.0 | 6153 | 174 | 790 |
| 2025-07-10 | done | 2 | 3.654 | 1160641 | 13.8 | 6277 | 190 | 777 |
| 2025-07-11 | done | 2 | 3.670 | 1088339 | 12.7 | 6367 | 182 | 762 |
| 2025-07-12 | done | 2 | 3.335 | 1034036 | 11.8 | 5889 | 174 | 885 |
| 2025-07-13 | done | 0 | reused | 1131291 | 13.3 | 6168 | 8 | 667 |
| 2025-07-14 | done | 2 | 3.438 | 1158597 | 13.8 | 6230 | 171 | 906 |
| 2025-07-15 | done | 2 | 3.547 | 1145015 | 13.5 | 6092 | 176 | 768 |
| 2025-07-16 | done | 2 | 3.540 | 1200094 | 14.3 | 6176 | 198 | 760 |
| 2025-07-17 | done | 2 | 3.671 | 1162680 | 13.7 | 6245 | 186 | 1073 |
| 2025-07-18 | done | 2 | 3.849 | 1187158 | 14.2 | 6378 | 208 | 1007 |
| 2025-07-19 | done | 2 | 3.389 | 1221502 | 14.6 | 5873 | 177 | 1040 |
| 2025-07-20 | done | 2 | 3.328 | 1162280 | 13.9 | 6176 | 191 | 786 |
| 2025-07-21 | done | 0 | reused | 1255164 | 15.0 | 6153 | 8 | 721 |
| 2025-07-22 | done | 2 | 3.607 | 1138222 | 13.6 | 6110 | 186 | 804 |
| 2025-07-23 | done | 2 | 3.712 | 1149056 | 13.7 | 6169 | 188 | 766 |
| 2025-07-24 | done | 2 | 3.831 | 1264597 | 15.1 | 6215 | 201 | 785 |
| 2025-07-25 | done | 2 | 3.723 | 1256104 | 15.0 | 6197 | 210 | 1051 |
| 2025-07-26 | done | 2 | 3.390 | 1189197 | 14.3 | 5949 | 183 | 788 |
| 2025-07-27 | done | 2 | 3.274 | 1230797 | 14.7 | 6062 | 200 | 758 |
| 2025-07-28 | done | 2 | 3.573 | 1426127 | 16.8 | 6205 | 207 | 789 |
| 2025-07-29 | done | 2 | 3.625 | 1363378 | 16.0 | 6062 | 189 | 769 |
| 2025-07-30 | done | 2 | 3.593 | 1377352 | 16.3 | 6130 | 164 | 772 |
| 2025-07-31 | done | 2 | 3.519 | 1371774 | 16.2 | 6126 | 229 | 806 |

Data: adsb.lol, Open Database License (ODbL) 1.0. "Contains data from adsb.lol, made available under the ODbL 1.0."
