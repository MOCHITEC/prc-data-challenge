# Item 28 `cache-fingerprint-gaps` — result (2026-09-16)

Code/log item. No training, no cache rebuild, no data frame opened (only `data/work/feat_meta.json` was read as
text). Script: `docs/investigation_2026-09-16/scripts/cache-fingerprint-gaps.py` (runs in ~10 s; reads git
objects, `experiments/*.jsonl` current and historical, file mtimes, `feat_meta.json`). All times UTC.
HEAD moved during the investigation (f250a9f → 1d646dd, other investigators' commits); `src/prc/features.py`
was not touched by any of them (last change 5794f24).

## 1. Hypothesis (plan.md §1 item 28, source P 10)

The feature-cache fingerprint hashes only `features.py`; changes to `weather.py`, `airports.py`, the METAR cache
or `train.py` do not invalidate it. P's own status: current cache consistent; risk for future runs only.
Sub-claims checked here:

- A. the fingerprint covers only raw-file names/sizes, row counts, `use_weather`, sha256(`features.py`);
- B–E. `weather.py`, `airports.py` (+ `data/external/airports.parquet`), the METAR cache
  (`data/ref/metar/*.csv`, `data/ref/metar_hourly.parquet`) and `train.py` are outside the key;
- F. whether any experiment recorded in `experiments/experiments.jsonl` (or in the records dropped at 4e807b5)
  ran on a cache built from feature code other than the `features.py` at its recorded `git_sha`;
- G. the cache on disk today is consistent with the code on disk today.

## 2. Procedure

1. Read `src/prc/pipeline.py` (`cache_fingerprint`, `_features_version`, `build_features`, `maybe_join_weather`)
   and its history: `git log -L '/def cache_fingerprint/,/^def _features_version/:src/prc/pipeline.py'`,
   `git show a96e43a:src/prc/pipeline.py` (pre-fingerprint cache), `git show c2cd247` (introduction).
2. `git log --since=2026-09-09 --name-only` over `features.py weather.py airports.py train.py pipeline.py schema.py`;
   sha256 of `features.py`, `weather.py`, `airports.py` at every such commit (`git show <sha>:<path>`).
3. Diff-read the behavioural changes of the un-fingerprinted inputs: `git show 34808da -- src/prc/weather.py`,
   `git show ea04ec9 -- src/prc/weather.py`, `diff` of `hourly_weather_table` between e00876a and HEAD,
   `git show d75c9f9 / 9e6fe61 -- src/prc/features.py`, `src/prc/airports.py` in full.
4. Experiment records: 35 lines of `experiments/experiments.jsonl` + 6 lines recovered with
   `git show 4e807b5^:experiments/experiments.jsonl`; `experiments/pipeline.jsonl`, `experiments/submissions.jsonl`.
   Neither log carries a timestamp or a `feature_code`; run end time = mtime of `experiments/oof_<exp_id>.parquet`
   (or `outputs/<name>.parquet` for the dropped records); `train_seconds` from the record;
   `start_est = end − train_seconds − 60 s`. `git_sha` is evaluated in `run_cv` after the fold loop
   (`src/prc/train.py:448`), i.e. it is HEAD at the **end** of CV, so for each run the script also reports HEAD at
   `start_est` and `features.py` hash there.
5. Mtimes of `data/work/feat_*.parquet`, `feat_meta.json`, `data/ref/metar/*.csv`, `data/ref/metar_hourly.parquet`,
   `data/external/airports.parquet`, `outputs/*.parquet`; `docs/features_log.md:15` (rebuild ≈ 4 min),
   `docs/handoff.md` §7 bugs 4–5, `tests/test_feature_cache.py`.
6. `n_features` in each record compared with the count implied by `features.py` at its `git_sha`
   (72 base numeric; weather adds 18 = 9 numeric hourly fields × {now, prev1h}, `frozen_wx` is Boolean and
   excluded by `numeric_feature_columns`; with the pre-34808da join it is 17 because `hour_utc` was dropped —
   P 7 measured 80 → 97 for v3/v4 and 80 → 98 today).

## 3. Results

### 3.1 What the fingerprint covers (fact, `src/prc/pipeline.py:55-71`, unchanged since c2cd247 2026-09-10 08:32:10)

| key | value | covers |
|---|---|---|
| `raw_dir` | resolved path | raw location |
| `files` | sorted (name, byte size) of every `*.parquet` under raw_dir | raw files (size only, no content hash) |
| `train_rows`, `rank_rows` | frame heights | raw content coarsely |
| `use_weather` | config flag | weather on/off |
| `feature_code` | sha256(`src/prc/features.py`)[:16] of the **working-tree** file | feature builder source |

Not in the key: `weather.py` (join and hourly aggregation code), `airports.py`, `schema.py` (imported by
`features.py`: `BLANKED_DEP_COLS`), `data/ref/metar/*.csv`, `data/ref/metar_hourly.parquet`,
`data/external/airports.parquet`, `train.py`, `pipeline.py`, library versions. Before c2cd247 (a96e43a
05:17:48 → 08:32:10 on 09-10) the cache was keyed by path alone (`if f_tr_path.exists() and f_rk_path.exists()`).
Rebuild is triggered whenever any key differs, so alternating `use_weather` configs rebuild every time (thrash,
not staleness).

### 3.2 Commits since 09-10 touching the un-fingerprinted or fingerprinted feature inputs

| commit | time | files | features.py | weather.py | airports.py | would the cache have rebuilt on the next run? |
|---|---|---|---|---|---|---|
| e00876a | 09-10 03:54:06 | weather.py (created) | b9e407e0 | 9354abf4 | – | n/a (no cache yet) |
| 2312457 | 09-10 05:45:45 | features.py, train.py | 309f8269 | 9354abf4 | – | path-only cache at the time |
| c2cd247 | 09-10 08:32:10 | pipeline.py (fingerprint introduced), train.py | 309f8269 | 9354abf4 | – | – |
| ea04ec9 | 09-14 06:24:10 | features.py, weather.py (docstring only), schema.py, pipeline.py, train.py | 519c7592 | 3259c8db | – | yes (features.py changed) — observed: ab_split run start ≈06:30 |
| 34808da | 09-15 07:49:36 | weather.py (`join_weather` fix: key renamed `_wx_hour`, tz alignment; `hourly_weather_table` untouched) | 519c7592 | 3f0d9e84 | – | **no** by key; a pre-fix weather cache would have survived. None existed (see 3.4) |
| 66c088b | 09-15 09:12:35 | features.py (+3 candidates), pipeline.py, train.py | 24ed31fc | 3f0d9e84 | – | yes |
| 5347c3b | 09-15 09:37:00 | features.py (+noise_control) | 57a9e319 | 3f0d9e84 | – | yes |
| 30e21e2 | 09-15 10:01:44 | features.py (+3) | 85122e20 | 3f0d9e84 | – | yes |
| e5af4b5 | 09-15 10:25:46 | train.py (offset encoding) | 85122e20 | 3f0d9e84 | – | no — not needed, encodings are fitted at train time, not cached |
| d75c9f9 | 09-15 10:46:18 | airports.py (created), features.py (+dest_distance_km) | d7fde923 | 3f0d9e84 | 7cba67ec | yes (features.py changed at the same time); the two runs on this code "died before training" (9e6fe61 message) → no cache written from d7fde923 |
| 9e6fe61 | 09-15 10:47:43 | features.py (move join into candidate_features) | 78fafe39 | 3f0d9e84 | 7cba67ec | yes |
| 5794f24 | 09-15 11:37:52 | features.py (+planned_flight_s) | 590ceb8f | 3f0d9e84 | 7cba67ec | yes — observed: `feat_meta.json` written 11:38:20 |
| f9fff60, 97e67c6, fd848bf | 09-15 14:58–15:54 | pipeline.py / train.py (column visibility) | 590ceb8f | 3f0d9e84 | 7cba67ec | no — not needed (visibility decided at train time) |

`weather.py` has three commits in total (e00876a, ea04ec9, 34808da); `airports.py` one (d75c9f9). The
`hourly_weather_table` body differs between e00876a and HEAD only in one docstring line, so the content of
`metar_hourly.parquet` does not depend on any weather.py change made since its creation.

### 3.3 Experiment timeline (41 runs; end = OOF/output mtime; `gap−train` = seconds between previous run end and this run end minus this run's `train_seconds`, an upper bound on load + feature build + idle)

| # | exp_id | git_sha (commit time) | features.py @sha | wx | n_feat | end | train_s | gap−train | rebuild expected (key changed vs previous run) | HEAD @start_est → features.py there | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | v1_baseline-cf5c3c4bb9 † | e0099d7 (09-10 07:59:27) | 309f8269 | off | 72 | 09-10 08:28:53 | 310 | – | first real cache (path-only era; fingerprint fix in working tree, committed 08:32:10) | e0099d7 → 309f8269 | dropped record |
| 2 | v2_two_clock-ce1c9c23ce † | 07ae6a3 (09-10 09:42:03) | 309f8269 | off | 72 | 09-10 10:13:46 | 1372 | 4922 | no | 07ae6a3 → 309f8269 | dropped record |
| – | v3 / v4 (weather, linear_tree) | none recorded | 309f8269 (at the time) | on | 97+10 matrix (P 7) | uploaded 09-10 13:21 / 15:42 UTC (log says +9 h, JST mislabelled, P 11) | – | – | on/off flips would have rebuilt | – | **no experiment, pipeline or output record in any commit; not determinable from this repo** |
| 3 | v5_unmatched_capacity-01023da84f † | 0305847 (09-14 01:43:23) | 309f8269 | off | 72 | 09-14 02:04:34 | 941 | – | no (if v4's wx-on cache had been left, yes) | 0305847 → 309f8269 | dropped |
| 4 | v6_unmatched_wider-fffd3895cb † | 79d6b60 (09-14 03:36:34) | 309f8269 | off | 72 | 09-14 04:00:45 | 1013 | 5957 | no | 79d6b60 | dropped |
| 5 | v7_unmatched_wider2-bf90b7f8fb † | d150a6c (09-14 04:01:49) | 309f8269 | off | 72 | 09-14 04:27:06 | 1091 | 490 | no | d150a6c | dropped |
| 0 | v8_seed_avg-6b88e350f8 † | d182fc1 (09-14 04:28:57) | 309f8269 | off | 72 | (no output file) | 2054 | – | no | – | dropped; end unknown |
| 6 | ab_split-ca87cb5af7 | 4e807b5 (06:30:33) | 519c7592 | off | 72 | 09-14 06:48:14 | 1035 | 7433 | **yes** (ea04ec9 changed features.py) | ea04ec9 → 519c7592 | sha committed 33 s after start_est; same features.py |
| 7 | ab_single-3978ee26d2 | 4e807b5 | 519c7592 | off | 72 | 06:59:16 | 292 | 370 | no | 4e807b5 | |
| 8 | abl_vanilla_params-c2251eb080 | c0084dd (07:37:31) | 519c7592 | off | 72 | 07:53:00 | 903 | 2321 | no | dda7e81 → 519c7592 | sha +33 s |
| 9 | abl_no_encodings-8cf7918aa5 | 4c2b2ca (08:14:39) | 519c7592 | off | 72 | 08:16:32 | 1053 | 358 | no | c0084dd → 519c7592 | sha committed 999 s after start_est (train.py-only commits in between) |
| 10 | abl_no_categoricals-7c6f287f34 | 4c2b2ca | 519c7592 | off | 72 | 08:44:03 | 1195 | 457 | no | 4c2b2ca | |
| 11 | main-f9d57a661f (LB v5) | 949d33b (09:06:27) | 519c7592 | off | 72 | 09:40:07 | 2216 | 1148 | no | 5f52eb4 → 519c7592 | sha +255 s |
| 12 | main-ab56551b5e | f5b950f (09:57:20) | 519c7592 | off | 72 | 10:31:22 | 2084 | 991 | no | 7fee7e4 → 519c7592 | sha +101 s |
| 13 | repro_v2-26c5b53b4d | 347ef81 (09-15 06:30:36) | 519c7592 | off | 72 | 09-15 06:47:41 | 1025 | 71954 | no | 53d931c → 519c7592 | sha +59 s |
| 14 | det_a-cfc9417c52 (LB v8) | 70ef994 (06:56:01) | 519c7592 | off | 72 | 07:11:21 | 1150 | 270 | no | 92623df → 519c7592 | sha +289 s |
| 15 | det_b-fa2cba5913 | c5e9908 (07:18:41) | 519c7592 | off | 72 | 07:34:13 | 970 | 402 | no | a7a300d → 519c7592 | sha +97 s |
| 16 | weather_check-1776e5e01c | 34808da (07:49:36) | 519c7592 | **on** | **90** | 07:54:51 | 295 | 944 | **yes** (wx flipped; METAR fetched 07:48:10–07:48:35 inside this window) | 8dc2152 → 519c7592 | sha +39 s; 90 = 72+18 ⇒ `hour_utc` retained ⇒ fixed `join_weather` ran |
| 17 | leak_check-54a21538de | 16c910e (08:15:45) | 519c7592 | off | 72 | 08:29:08 | 435 | 1621 | **yes** (wx flipped back) | 16c910e | |
| 18 | feat_base-b9e21a829d | def61ae (09:12:56) | 24ed31fc | on | 93 | 09:18:15 | 306 | 2642 | **yes** (features.py 66c088b + wx flip) | b3f4c55 → **519c7592** | sha +45 s; 93 = 90+3 matches @sha, not @HEAD-at-start (90) |
| 19 | feat_f1_queue-da758e547b | 629dea8 (09:18:47) | 24ed31fc | on | 93 | 09:24:23 | 357 | 11 | no | def61ae → 24ed31fc | sha +80 s |
| 20 | feat_f2_bank-5eb8fdddfd | 629dea8 | 24ed31fc | on | 93 | 09:29:49 | 322 | 4 | no | 629dea8 | |
| 21 | feat_f3_share-15cc219230 | 629dea8 | 24ed31fc | on | 93 | 09:35:10 | 318 | 4 | no | 629dea8 | |
| 22 | feat_base-4908e7408f | 5347c3b (09:37:00) | 57a9e319 | on | 94 | 09:42:00 | 302 | 108 | **yes** (features.py) | 629dea8 → **24ed31fc** | sha +61 s; 94 = 93+1 matches @sha |
| 23 | feat_f4_noise-dfae91b597 | 5347c3b | 57a9e319 | on | 94 | 09:47:41 | 337 | 4 | no | 5347c3b | |
| 24 | ff1_base-b60b8da07b | a418a88 (09:49:37) | 57a9e319 | on | 94 | 09:55:15 | 353 | 100 | no | 5347c3b → 57a9e319 | sha +75 s |
| 25 | ff1_noise-36511caad3 | a418a88 | 57a9e319 | on | 94 | 10:00:09 | 290 | 4 | no | a418a88 | |
| 26 | ff1_noise-47e9566782 | 30e21e2 (10:01:44) | 85122e20 | on | 97 | 10:07:24 | 312 | 123 | **yes** (features.py) | a418a88 → **57a9e319** | sha +31 s; 97 = 94+3 matches @sha |
| 27–29 | feat_f5 / f6 / f7 | 30e21e2 | 85122e20 | on | 97 | 10:11:45 / 10:16:38 / 10:21:47 | 257 / 288 / 305 | 4 / 4 / 4 | no | 30e21e2 | |
| 30 | feat_f8_opoffset-c4dc7f0a64 | e5af4b5 (10:25:46) | 85122e20 | on | 97 | 10:29:49 | 275 | 208 | no (train.py only) | 30e21e2 → 85122e20 | sha +91 s |
| 31 | rep_jan_off-ef82a28b63 | 14f563b (10:31:16) | 85122e20 | on | 97 | 10:36:43 | 347 | 67 | no | e5af4b5 → 85122e20 | sha +79 s |
| 32 | rep_jan_on-9db9f2476e | 14f563b | 85122e20 | on | 97 | 10:42:27 | 340 | 4 | no | 14f563b | |
| – | (two runs on d75c9f9 code) | d75c9f9 (10:46:18) | d7fde923 | on | – | crashed before training (9e6fe61 message) | – | – | build reached `add_destination_distance` (airports.parquet written 10:45:37, 41 s **before** d75c9f9 was committed) then failed in `candidate_features.select` → **no cache written** | – | not recorded |
| 33 | ff1_noise2-856a62b645 | 9e6fe61 (10:47:43) | 78fafe39 | on (inferred from n_feat) | 98 | 10:52:47 | 276 | 344 | **yes** (features.py) | d75c9f9 → d7fde923 (no cache from it existed) | sha +31 s |
| 34 | feat_f9_dist-5ed94c00a3 | 9e6fe61 | 78fafe39 | on | 98 | 10:57:33 | 281 | 4 | no | 9e6fe61 | |
| 35 | pw_noise-c8f996bf3d | 50b574d (10:58:55) | 78fafe39 | on | 98 | 11:17:08 | 1087 | 88 | no | 9e6fe61 → 78fafe39 | sha +53 s |
| 36 | pw_both-91b711793b | 50b574d | 78fafe39 | on | 98 | 11:35:47 | 1115 | 4 | no | 50b574d | |
| 37 | pw_noise2-7672c5b2b7 | 5794f24 (11:37:52) | 590ceb8f | on (inferred) | 99 | 11:58:22 | 1202 | 153 | **yes** — observed: `feat_meta.json` / `feat_*.parquet` written **11:38:20**, feature_code 590ceb8f | 50b574d → **78fafe39** | sha +31 s; 99 = 98+1 matches @sha |
| 38 | f10_plannedflight-b416261a17 | 5794f24 | 590ceb8f | on | 99 | 12:18:04 | 1177 | 4 | no | 5794f24 | |
| 39 | main-63086f0648 (LB v6) | 062a6f0 (12:19:35) | 590ceb8f | on | 99 | 14:39:00 | 2758 | 5699 | no | 062a6f0 | |
| 40 | main-48c19bc830 (LB v7) | 97e67c6 (14:59:33) | 590ceb8f | on | 99 | 15:39:29 | 2387 | 1241 | no | f9fff60 → 590ceb8f | sha +51 s |

† records deleted at 4e807b5 ("drop the v1-v8 history"), recovered from `4e807b5^`. Observed `gap−train` on
back-to-back cache hits is 4–11 s; on expected rebuilds 108–153 s where no idle time intervened (rows 22, 26, 37);
`docs/features_log.md:15` states "約 4 分" for a rebuild.

### 3.4 State of the un-fingerprinted inputs today

| input | mtime | relation to the current cache (11:38:20) |
|---|---|---|
| `data/ref/metar/*.csv` (10 stations, 2025-01-01→2026-08-01) | 09-15 07:48:10–07:48:34 | older; never re-fetched; first METAR data ever present in this environment (`data/ref` dir created 07:48:35) |
| `data/ref/metar_hourly.parquet` (670,968 B) | 09-15 07:48:35 | older; built by `hourly_weather_table`, whose code is unchanged (docstring only) since e00876a |
| `src/prc/weather.py` | last commit 34808da 07:49:36 | older; the `join_weather` fix is applied at cache build time, so the current cache carries the fixed join |
| `data/external/airports.parquet` (705,518 B) | 09-15 10:45:37 | older; written by the first (crashed) distance run |
| `src/prc/airports.py` | d75c9f9 10:46:18, never changed since | older |
| `data/work/feat_meta.json` | 09-15 11:38:20 | `feature_code` 590ceb8f8170ff6b = sha256(features.py) at HEAD; `use_weather: true`; rows 4,167,797 / 689,534; `git status` clean apart from this item's files |

## 4. Verdicts

| sub-claim | verdict | basis |
|---|---|---|
| A. key = raw names/sizes, row counts, `use_weather`, sha256(features.py) only | **CONFIRMED** | §3.1; function text identical since c2cd247 (`git log -L`) |
| B. `weather.py` changes do not invalidate the cache | **CONFIRMED** (code). Exposure in history: **none found** | only behavioural change 34808da; no weather cache existed before it (METAR first fetched 07:48 that morning; pre-fix join crashed — commit message, handoff §7 bug 5; weather_check n_features 90 = 72+18 ⇒ fixed join) |
| C. `airports.py` / `airports.parquet` not in key | **CONFIRMED** (code). Exposure: **none found** | airports.py created together with a features.py change and never modified; airports.parquet predates every surviving cache. Silent-null path (`except Exception` → null `dest_distance_km`) exists and would not be detected by the key; whether the cached column is non-null was **not verified** (no frame load) |
| D. METAR cache not in key | **CONFIRMED** (code). Content not stale | built once 07:48:35; aggregation code unchanged since e00876a; CSVs unchanged |
| E. `train.py` not in key | **CONFIRMED** as a fact, but **no staleness route**: `fit_encodings`/`attach_encodings` (train.py:182/191) run inside `predict_fold`/`predict_ranking` on the cached frame; column visibility (`visible_numeric_columns`) is decided at train time; nothing train.py computes is written into `feat_*.parquet` |
| F. some recorded experiment ran on a stale cache (feature code ≠ features.py at its `git_sha`) | **REFUTED for all 41 recorded runs, on indirect evidence** | the fingerprint ties the cache to the working-tree `features.py` at run start; `n_features` in every record equals the count implied by `features.py` at the recorded `git_sha` (72; 90; 93; 94; 97; 98; 99), including the five runs (18, 22, 26, 33, 37) whose `git_sha` was committed 31–61 s after `start_est` and whose HEAD-at-start had an older `features.py` (which would have given 90, 93, 94, –, 98). The pre-fingerprint synthetic-cache incident (09-10, c2cd247 message) crashed before any record was written |
| G. current cache consistent with current code (P) | **CONFIRMED** | §3.4 |
| Side finding: `git_sha` is HEAD at CV **end**, not start (train.py:448) | **CONFIRMED** | 17 of 35 current records have their `git_sha` committed after `start_est` (31–999 s); for features.py this never mattered (see F); for train.py/pipeline.py content it means the recorded sha may post-date the code that ran (e.g. row 9: three train.py commits inside the window) |
| Side finding: `use_weather` in the key causes full rebuilds when configs alternate | **CONFIRMED** (rows 16→17→18; det_a 07:11 wx off → main wx on) | cost only, not staleness |

## 5. Undeterminable parts

- **v3 / v4** (uploaded 09-10 13:21 / 15:42 UTC, LB 306.29 / 305.03, weather on): no `experiments.jsonl`,
  `pipeline.jsonl` or `outputs/` record exists in any commit (`git log --all -S'v3_two_clock_weather'` hits only
  config add/delete and the submissions backfill); the METAR files on disk date from 09-15. Whether they were built
  in this environment, on which `features.py` (309f8269 by date), on which weather cache, and whether their cache
  was reused for v5–v8 (`use_weather` differs, so by key: no) cannot be settled from the repository.
- Historical `feat_meta.json` contents: `data/` is git-ignored; only the current file exists. Every "rebuilt at"
  entry in §3.3 other than 11:38:20 is inferred from the key rule plus run timings.
- Whether `dest_distance_km` in the current cache is populated (the `except Exception` fallback would have cached a
  null column with an identical fingerprint): needs a frame load, out of scope here. Indirect: exp 34 vs 33 differ
  by 2.9 s with the column as the only difference.
- `start_est` assumes 60 s for raw load before `run_cv`'s `t0`; the "sha committed N s after start" figures carry
  that uncertainty.
- v8_seed_avg (dropped record) has no output file; its end time is unknown.

## 6. Facts vs inference

Facts (read from code, git objects, file metadata, logs):
- Fingerprint keys and the hashed file (§3.1); function unchanged since c2cd247; path-only cache before it.
- Commit list, times and per-commit hashes of `features.py`, `weather.py`, `airports.py` (§3.2).
- 34808da changed only `join_weather` (+ new `WX_KEY`, `_utc_hour`); `hourly_weather_table` differs from e00876a by
  one docstring line.
- `git_sha` is computed after the fold loop in `run_cv`; `experiments.jsonl` has no timestamp and no `feature_code`.
- OOF / output / cache / METAR / airports mtimes and sizes (§3.3, §3.4); `feat_meta.json` content.
- `n_features` per record; weather adds 18 numeric columns (9 fields × 2, `frozen_wx` Boolean excluded).
- 9e6fe61 message: the two d75c9f9-code runs died before training; c2cd247 message: the first real run trained on
  synthetic features and crashed at the submission step; handoff §7 bug 5: pre-fix weather join crashed on tz.

Inference:
- "Rebuild expected" per run (key rule applied to consecutive runs) and the resulting cache timeline; rebuild
  durations from `gap−train`.
- No weather cache existed before 34808da (from the METAR mtimes + the crash statements + n_features 90).
- No cache was written from d7fde923 code (from the crash statement and where the crash is in the call chain).
- Each run's working tree contained the `features.py` of its recorded `git_sha` (from `n_features` agreement),
  hence no recorded run used a cache built from other feature code.
- v1's recorded run (342.38) is the "v1 completes" run of the c2cd247 message, i.e. ran with the fingerprint fix
  in the working tree before it was committed.
